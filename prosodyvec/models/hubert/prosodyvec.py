# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import librosa
import scipy

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from fairseq import utils
from fairseq.data.data_utils import compute_mask_indices
from fairseq.data.dictionary import Dictionary
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.models import BaseFairseqModel, register_model
from fairseq.models.wav2vec.wav2vec2 import (
    ConvFeatureExtractionModel,
    TransformerEncoder,
    TransformerEncoderLimitedContext,
)
from fairseq.modules import GradMultiply, LayerNorm
from fairseq.tasks.prosodyvec_pretraining import (
    ProsodyvecPretrainingConfig,
    ProsodyvecPretrainingTask,
)
from omegaconf import II, ListConfig

logger = logging.getLogger(__name__)

EXTRACTOR_MODE_CHOICES = ChoiceEnum(["default", "layer_norm"])
MASKING_DISTRIBUTION_CHOICES = ChoiceEnum(
    ["static", "uniform", "normal", "poisson"]
)

def print_memory_usage(tag: str=""):
    """Print GPU memory usage for debugging."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        logger.info(f"[{tag}] GPU memory allocated: {allocated:.3f} GB, reserved: {reserved:.3f} GB")

@dataclass
class ProsodyvecConfig(FairseqDataclass):
    label_rate: float = II("task.label_rate")

    extractor_mode: EXTRACTOR_MODE_CHOICES = field(
        default="default",
        metadata={
            "help": "mode for feature extractor. default has a single group "
            "norm with d groups in the first conv block, whereas layer_norm "
            "has layer norms in every block (meant to use with normalize=True)"
        },
    )
    encoder_layers: int = field(
        default=12, metadata={"help": "num encoder layers in the transformer"}
    )
    encoder_embed_dim: int = field(
        default=768, metadata={"help": "encoder embedding dimension"}
    )
    encoder_ffn_embed_dim: int = field(
        default=3072, metadata={"help": "encoder embedding dimension for FFN"}
    )
    encoder_attention_heads: int = field(
        default=12, metadata={"help": "num encoder attention heads"}
    )
    activation_fn: ChoiceEnum(utils.get_available_activation_fns()) = field(
        default="gelu", metadata={"help": "activation function to use"}
    )
    num_spkrs: int = field(
        default=0,
        metadata={
            "help": "number of speakers in the dataset. 0 means no speaker classification"
        },
    )
    spkr_class_stop_grad: bool = field(
        default=False,
        metadata={"help": "stop gradient for speaker classification, instead of gradient reversal"},
    )
    spkr_class_pooling: bool = field(
        default=False,
        metadata={"help": "apply pooling to speaker classification"},
    )
    spkr_class_pooling_pos_enc: bool = field(
        default=False,
        metadata={"help": "apply positional encoding to speaker classification pooling"},
    )
    spkr_class_layer_mode: str = field(
        default="final",
        metadata={"help": "mode for speaker classification layer, 'final' means final layer, 'all' means all layers, 'random' means a randomly selected layer, '(int)' means that specific layer"},
    )

    # dropouts
    dropout: float = field(
        default=0.1,
        metadata={"help": "dropout probability for the transformer"},
    )
    attention_dropout: float = field(
        default=0.1,
        metadata={"help": "dropout probability for attention weights"},
    )
    activation_dropout: float = field(
        default=0.0,
        metadata={"help": "dropout probability after activation in FFN"},
    )
    encoder_layerdrop: float = field(
        default=0.0,
        metadata={"help": "probability of dropping a tarnsformer layer"},
    )
    dropout_input: float = field(
        default=0.0,
        metadata={"help": "dropout to apply to the input (after feat extr)"},
    )
    dropout_features: float = field(
        default=0.0,
        metadata={
            "help": "dropout to apply to the features (after feat extr)"
        },
    )

    final_dim: int = field(
        default=0,
        metadata={
            "help": "project final representations and targets to this many "
            "dimensions. set to encoder_embed_dim is <= 0"
        },
    )
    untie_final_proj: bool = field(
        default=False,
        metadata={"help": "use separate projection for each target"},
    )
    layer_norm_first: bool = field(
        default=False,
        metadata={"help": "apply layernorm first in the transformer"},
    )
    conv_feature_layers: str = field(
        default="[(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2",
        metadata={
            "help": "string describing convolutional feature extraction "
            "layers in form of a python list that contains "
            "[(dim, kernel_size, stride), ...]"
        },
    )
    conv_bias: bool = field(
        default=False, metadata={"help": "include bias in conv encoder"}
    )
    logit_temp: float = field(
        default=0.1, metadata={"help": "temperature to divide logits by"}
    )
    target_glu: bool = field(
        default=False, metadata={"help": "adds projection + glu to targets"}
    )
    feature_grad_mult: float = field(
        default=1.0,
        metadata={"help": "multiply feature extractor var grads by this"},
    )

    # masking
    mask_length: int = field(default=10, metadata={"help": "mask length"})
    mask_prob: float = field(
        default=0.65,
        metadata={"help": "probability of replacing a token with mask"},
    )
    mask_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static", metadata={"help": "how to choose mask length"}
    )
    mask_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument "
            "(used for more complex distributions), "
            "see help in compute_mask_indicesh"
        },
    )
    no_mask_overlap: bool = field(
        default=False, metadata={"help": "whether to allow masks to overlap"}
    )
    mask_min_space: int = field(
        default=1,
        metadata={
            "help": "min space between spans (if no overlap is enabled)"
        },
    )

    # channel masking
    mask_channel_length: int = field(
        default=10,
        metadata={"help": "length of the mask for features (channels)"},
    )
    mask_channel_prob: float = field(
        default=0.0,
        metadata={"help": "probability of replacing a feature with 0"},
    )
    mask_channel_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static",
        metadata={"help": "how to choose mask length for channel masking"},
    )
    mask_channel_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument "
            "(used for more complex distributions), "
            "see help in compute_mask_indicesh"
        },
    )
    no_mask_channel_overlap: bool = field(
        default=False,
        metadata={"help": "whether to allow channel masks to overlap"},
    )
    mask_channel_min_space: int = field(
        default=1,
        metadata={
            "help": "min space between spans (if no overlap is enabled)"
        },
    )

    # positional embeddings
    conv_pos: int = field(
        default=128,
        metadata={
            "help": "number of filters for convolutional positional embeddings"
        },
    )
    conv_pos_groups: int = field(
        default=16,
        metadata={
            "help": "number of groups for convolutional positional embedding"
        },
    )

    latent_temp: Tuple[float, float, float] = field(
        default=(2, 0.5, 0.999995),
        metadata={"help": "legacy (to be removed)"},
    )

    # loss computation
    skip_masked: bool = field(
        default=False,
        metadata={"help": "skip computing losses over masked frames"},
    )
    skip_nomask: bool = field(
        default=False,
        metadata={"help": "skip computing losses over unmasked frames"},
    )
    span_loss: bool = field(
        default=False,
        metadata={"help": "use span-based loss"},
    )
    span_loss_max_pos: int = field(
        default=20,
        metadata={"help": "maximum span pos embedding index for span-based loss"},
    )

    # pretrained model
    pretrained_encoder: Optional[str] = field(
        default=None,
        metadata={"help": "path to pretrained encoder model"},
    )

def normalized_sinusoidal_encoding(pos, C):
    """
    input:
        pos (B, T): tensor of normalized positions in [0, 1]
        C: integer dimension of the positional encoding
    
    returns:
        pos_enc (B, T, C)
    """
    div_term = torch.exp(
        torch.arange(0, C, 2, dtype=pos.dtype, device=pos.device) * -np.log(10000.0) / C
    )
    sinusoid_inp = pos.unsqueeze(-1) * div_term # (B, T, C//2)
    pos_enc = torch.zeros(pos.size(0), pos.size(1), C, dtype=pos.dtype, device=pos.device)
    pos_enc[:, :, 0::2] = torch.sin(sinusoid_inp)
    pos_enc[:, :, 1::2] = torch.cos(sinusoid_inp)
    return pos_enc

class SelfAttentionPooling(nn.Module):
    """
    Computes a weighted average of the input features using self-attention.
    """
    def __init__(self, embed_dim: int, num_heads: int=8, add_pos_enc: bool=False):
        super().__init__()
        self.activation = nn.Tanh()
        self.W = nn.Linear(embed_dim, embed_dim)
        self.W_a = nn.Linear(embed_dim, 1)
        self.num_heads = num_heads
        self.softmax = nn.functional.softmax

        self.add_pos_enc = add_pos_enc

    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        input:
            x: shape (B, T, C)
            padding_mask: shape (B, T)

        returns:
            x_pooled, shape (B, 1, C)
        """
        B, T, C = x.shape
        if padding_mask is None:
            padding_mask = torch.zeros((B, T), dtype=torch.bool, device=x.device)
        if self.add_pos_enc:
            rel_pos = torch.zeros((B, T), device=x.device, dtype=x.dtype)
            seq_lens = torch.sum(~padding_mask, dim=1)
            for b in range(B):
                rel_pos[b, :seq_lens[b]] = torch.linspace(0, 1, seq_lens[b], dtype=x.dtype, device=x.device) # (B, T)
            pos_emb = normalized_sinusoidal_encoding(rel_pos, C) # (B, T, C)
            x = x + pos_emb
            
        attn_logits = self.W_a(
            self.activation(
                self.W(x)
            )
        ).squeeze(-1) # (B, T)
        attn_logits = attn_logits.masked_fill(padding_mask, torch.finfo(attn_logits.dtype).min) # (B, T)
        attn_weights = self.softmax(
            attn_logits, dim=1
        ) # (B, T)
        x_pooled = torch.sum(
            x * attn_weights.unsqueeze(-1), dim=1
        ) # (B, C)
        return x_pooled.unsqueeze(1)  # (B, 1, C)

# speaker classification module
class SpeakerClassLayer(nn.Module):
    def __init__(
            self,
            encoder_embed_dim, num_spkrs,
            stopgrad: bool = False,
            pooling: bool = False,
            pooling_pos_enc: bool = False,
            layer_mode: str = "final",
            encoder_layers: int = None
    ):
        """
        single linear classification layer to map encoder output to speaker id
        """
        super().__init__()
        self.spkr_class_layer = nn.Linear(encoder_embed_dim, num_spkrs)
        self.stopgrad = stopgrad
        self.layer_combiner = None
        if layer_mode == "all":
            # create learnable weights for linear combination of all layers
            self.layer_combiner = nn.Parameter(
                torch.FloatTensor(encoder_layers).uniform_()
            )
        self.pooling = None
        if pooling:
            self.pooling = SelfAttentionPooling(encoder_embed_dim, add_pos_enc=pooling_pos_enc)

    def forward(self, x):
        """
        inputs:
            x: (L,B,T,C) or (B,T,C) tensor, with number of layers L, batch size B, seq length T, feat dim C 
        """
        # grad reversal or stopgrad
        if not self.stopgrad:
            # adversarial speaker classification
            x = GradMultiply.apply(x, -1.0)
        else:
            # gradient stop
            x = x.detach()

        # apply layer weights if needed
        if self.layer_combiner is not None:
            assert x.dim() == 4, f"expected 4D tensor, got {x.dim()}D tensor"
            # weights are softmax of self.layer_combiner
            layer_wts = self.layer_combiner.softmax(dim=0)  # (L,)
            x = torch.einsum("l,lbtc->btc", layer_wts, x) # (B, T, C)
        else:
            assert x.dim() == 3, f"expected 3D tensor, got {x.dim()}D tensor"

        # apply pooling if needed
        if self.pooling is not None:
            # apply pooling before classification
            x = self.pooling(x)

        x = self.spkr_class_layer(x)
        return x

def compute_span_start_end(mask_indices):
    """
    input:
        mask_indices: (B, T) bool tensor of masked frames
    output:
        starts: (B, T) long tensor of start indices of masked spans, -1 if not masked
        ends: (B, T) long tensor of end indices of masked spans, -1 if not masked
        from_starts: (B, T) long tensor of positions from start of masked spans, 0 if not masked
        to_ends: (B, T) long tensor of positions to end of masked spans, 0 if not masked
    """
    B, T = mask_indices.shape
    device = mask_indices.device

    mask_prev = torch.zeros_like(mask_indices)
    mask_prev[:, 1:] = mask_indices[:, :-1]
    start_mask = mask_indices & ~mask_prev

    mask_next = torch.zeros_like(mask_indices)
    mask_next[:, :-1] = mask_indices[:, 1:]
    end_mask = mask_indices & ~mask_next

    time_idx = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)

    # start indices
    start_fill = (start_mask * (time_idx + 1).to(torch.long))
    last_seen_start, _ = torch.cummax(start_fill, dim=1)
    starts = torch.where(mask_indices, last_seen_start - 1, torch.full_like(time_idx, -1))

    # end indices
    end_fill = torch.where(end_mask, time_idx, T+1)
    nearest_future_end = torch.flip(
        torch.cummin(
            torch.flip(end_fill, dims=[1]), dim=1
        )[0],
        dims=[1],
    )
    ends = torch.where(mask_indices, nearest_future_end, torch.full_like(time_idx, -1))

    # switch from start/end indices to outer boundaries
    starts = torch.where(mask_indices, starts-1, torch.full_like(time_idx, -1))
    ends = torch.where(mask_indices, ends+1, torch.full_like(time_idx, -1))

    # within-span positions
    from_starts = torch.where(
        mask_indices,
        time_idx - starts,
        torch.zeros_like(time_idx),
    )
    to_ends = torch.where(
        mask_indices,
        ends - time_idx,
        torch.zeros_like(time_idx),
    )

    return starts, ends, from_starts, to_ends

def deduplicate_span_start_ends(starts, ends, mask_indices):
    """
    converts tensors of span start/ends to only keep one start/end per span for more efficient processing
    input:
        starts: (B, T) long tensor of start indices of masked spans, -1 if not masked
        ends: (B, T) long tensor of end indices of masked spans, -1 if not masked
        mask_indices: (B, T) bool tensor of masked frames
    output:
        dedup_starts_flat: (num_masked_spans,) long tensor of batch-flattened start indices
        dedup_ends_flat: (num_masked_spans,) long tensor of batch-flattened end indices
        dedup2orig_idx_flat: (num_masked_frames,) long tensor mapping from deduplicated_{starts,ends} to original masked frames
        batch_start_idx_flat: (num_masked_frame,) long tensor of batch-flattened start indices indicating which batch item each masked frame belongs to
        batch_end_idx_flat: (num_masked_frame,) long tensor of batch-flattened end indices indicating which batch item each masked frame belongs to
    """
    B, T = starts.shape
    device = starts.device
    assert ends.shape == (B, T)
    assert mask_indices.shape == (B, T)

    batch_add = (torch.arange(B).unsqueeze(1) * T).to(device)  # (B, 1)

    starts_flat = (starts + batch_add).masked_select(mask_indices)  # (num_masked_frames,)
    ends_flat = (ends + batch_add).masked_select(mask_indices)  # (num_masked_frames,)

    # get unique start indices and their first occurrence positions
    dedup_starts_flat, dedup2orig_idx_flat = torch.unique_consecutive(
        starts_flat, return_inverse=True
    )
    dedup_ends_flat = torch.unique_consecutive(
        ends_flat,
    )

    # get batch indices for each masked frame
    span_ids = dedup2orig_idx_flat  # (num_masked_frames,)
    batch_ids = batch_add.expand(-1, T).masked_select(mask_indices)  # (num_masked_frames,)
    num_spans = dedup_starts_flat.size(0)
    frame_indices = torch.arange(len(span_ids), device=device)  # (num_masked_frames,)
    first_occ = torch.full((num_spans,), len(span_ids), device=device)  # (num_spans,)
    first_occ.scatter_reduce_(
        0, # reduce by span_id
        span_ids, # frame belongs to this span
        frame_indices, # minimize over frame indices
        reduce="amin",
    )
    dedup_batch_start_idx_flat = batch_ids[first_occ]  # (num_spans,)

    return dedup_starts_flat, dedup_ends_flat, dedup2orig_idx_flat, dedup_batch_start_idx_flat

@register_model("prosodyvec", dataclass=ProsodyvecConfig)
class ProsodyvecModel(BaseFairseqModel):
    def __init__(
        self,
        cfg: ProsodyvecConfig,
        task_cfg: ProsodyvecPretrainingConfig,
        dictionaries: List[Dictionary],
    ) -> None:
        super().__init__()
        logger.info(f"ProsodvecModel Config: {cfg}")

        feature_enc_layers = eval(cfg.conv_feature_layers)  # noqa
        self.embed = feature_enc_layers[-1][0]

        self.feature_extractor = ConvFeatureExtractionModel(
            conv_layers=feature_enc_layers,
            dropout=0.0,
            mode=cfg.extractor_mode,
            conv_bias=cfg.conv_bias,
        )
        feature_ds_rate = np.prod([s for _, _, s in feature_enc_layers])
        self.feature_rate = task_cfg.sample_rate / feature_ds_rate
        self.feat2tar_ratio = (
            cfg.label_rate * feature_ds_rate / task_cfg.sample_rate
        )

        self.post_extract_proj = (
            nn.Linear(self.embed, cfg.encoder_embed_dim)
            if self.embed != cfg.encoder_embed_dim
            else None
        )

        self.mask_prob = cfg.mask_prob
        self.mask_selection = cfg.mask_selection
        self.mask_other = cfg.mask_other
        self.mask_length = cfg.mask_length
        self.no_mask_overlap = cfg.no_mask_overlap
        self.mask_min_space = cfg.mask_min_space

        self.mask_channel_prob = cfg.mask_channel_prob
        self.mask_channel_selection = cfg.mask_channel_selection
        self.mask_channel_other = cfg.mask_channel_other
        self.mask_channel_length = cfg.mask_channel_length
        self.no_mask_channel_overlap = cfg.no_mask_channel_overlap
        self.mask_channel_min_space = cfg.mask_channel_min_space

        self.dropout_input = nn.Dropout(cfg.dropout_input)
        self.dropout_features = nn.Dropout(cfg.dropout_features)

        self.feature_grad_mult = cfg.feature_grad_mult
        self.logit_temp = cfg.logit_temp
        self.skip_masked = cfg.skip_masked
        self.skip_nomask = cfg.skip_nomask
        self.span_loss = cfg.span_loss
        self.span_loss_max_pos = cfg.span_loss_max_pos

        final_dim = (
            cfg.final_dim if cfg.final_dim > 0 else cfg.encoder_embed_dim
        )

        self.mask_emb = nn.Parameter(
            torch.FloatTensor(cfg.encoder_embed_dim).uniform_()
        )

        self.encoder = TransformerEncoder(cfg)

        self.layer_norm = LayerNorm(self.embed)

        self.target_glu = None
        if cfg.target_glu:
            self.target_glu = nn.Sequential(
                nn.Linear(final_dim, final_dim * 2), nn.GLU()
            )

        self.untie_final_proj = cfg.untie_final_proj
        if self.untie_final_proj:
            self.final_proj = nn.Linear(
                cfg.encoder_embed_dim, final_dim * len(dictionaries)
            )
            if self.span_loss:
                self.final_span_proj = nn.Linear(
                    cfg.encoder_embed_dim * 2, final_dim * len(dictionaries)
                )
        else:
            self.final_proj = nn.Linear(cfg.encoder_embed_dim, final_dim)
            if self.span_loss:
                self.final_span_proj = nn.Linear(
                    cfg.encoder_embed_dim * 2, final_dim
                )

        # modules below are not needed during fine-tuning
        if any([d is None for d in dictionaries]):
            logger.info(
                "cannot find dictionary. assume will be used for fine-tuning"
            )
        else:
            self.num_classes = [len(d) for d in dictionaries]
            self.label_embs_concat = nn.Parameter(
                torch.FloatTensor(sum(self.num_classes), final_dim)
            )
            nn.init.uniform_(self.label_embs_concat)

        # use boundary embeddings for span loss in cases where masked span begins at frame 0 or ends at last frame
        if self.span_loss:
            self.left_boundary_emb = nn.Parameter(
                torch.FloatTensor(cfg.encoder_embed_dim).uniform_()
            )
            self.right_boundary_emb = nn.Parameter(
                torch.FloatTensor(cfg.encoder_embed_dim).uniform_()
            )
            self.left_rel_pos_emb = nn.Embedding(
                self.span_loss_max_pos, cfg.encoder_embed_dim
            )
            self.right_rel_pos_emb = nn.Embedding(
                self.span_loss_max_pos, cfg.encoder_embed_dim
            )

            self.label_embs_concat_span = nn.Parameter(
                torch.FloatTensor(sum(self.num_classes), final_dim)
            )
            nn.init.uniform_(self.label_embs_concat_span)

        # speaker classification layer
        if cfg.num_spkrs > 0:
            self.num_spkrs = cfg.num_spkrs
            logger.info(f"cfg.spkr_class_layer_mode type: {type(cfg.spkr_class_layer_mode)}, value: {cfg.spkr_class_layer_mode}, isdigit? {cfg.spkr_class_layer_mode.isdigit()}")
            self.spkr_class_layer_mode = cfg.spkr_class_layer_mode
            if cfg.spkr_class_layer_mode in ["all", "random"]:
                self.encoder_output_layer = cfg.encoder_layers
            elif cfg.spkr_class_layer_mode.isdigit():
                self.spkr_class_layer_mode = int(cfg.spkr_class_layer_mode)
                logger.info(f"self.spkr_class_layer_mode type: {type(self.spkr_class_layer_mode)}, value: {self.spkr_class_layer_mode}, isint? {isinstance(self.spkr_class_layer_mode, int)}")
                # return all layers and select the specified layer before passing to speaker classifier
                self.encoder_output_layer = cfg.encoder_layers
            else:
                self.encoder_output_layer = None
            self.spkr_class_layer = SpeakerClassLayer(
                cfg.encoder_embed_dim,
                cfg.num_spkrs,
                stopgrad=cfg.spkr_class_stop_grad,
                pooling=cfg.spkr_class_pooling,
                pooling_pos_enc=cfg.spkr_class_pooling_pos_enc,
                layer_mode=cfg.spkr_class_layer_mode,
                encoder_layers=cfg.encoder_layers,
            )
            logger.info(f"added speaker classification layer ({type(self.spkr_class_layer)}) with {cfg.num_spkrs} speakers, stopgrad={cfg.spkr_class_stop_grad}, pooling={cfg.spkr_class_pooling}, pooling_pos_enc={cfg.spkr_class_pooling_pos_enc}")
        else:
            self.num_spkrs = 0
            self.spkr_class_layer = None
            self.spkr_class_layer_mode = None
            self.encoder_output_layer = None
        
        self.pretrained_encoder = None
        if cfg.pretrained_encoder is not None:
            self.pretrained_encoder = torch.load(cfg.pretrained_encoder)["model"]
            new_state_dict = self.get_pretrained_state_dict()
            print(self.load_state_dict(new_state_dict, strict=False))

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""

        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    def get_pretrained_state_dict(self):

        new_state_dict = {}
        for idx, (nc, c) in enumerate(self.state_dict().items()):
            copy_this = False
            if nc in self.pretrained_encoder.keys():
                if c.shape == self.pretrained_encoder[nc].shape:
                    copy_this = True
                    logger.info(f'{nc} ({c.shape}) copy? {copy_this}')
                else:
                    logger.info(f'{nc} ({c.shape}) copy? {copy_this} (shape mismatch {self.pretrained_encoder[nc].shape})')
            else:
                logger.info(f'{nc} ({c.shape}) copy? {copy_this} (not found in pretrained)')

            if copy_this:
                new_state_dict[nc] = self.pretrained_encoder[nc]
            else:
                new_state_dict[nc] = c
        
        return new_state_dict

    @classmethod
    def build_model(cls, cfg: ProsodyvecConfig, task: ProsodyvecPretrainingTask):
        """Build a new model instance."""

        model = ProsodyvecModel(cfg, task.cfg, task.dictionaries)
        return model

    def set_num_updates(self, num_updates):
        self.num_updates = num_updates

    def apply_mask(self, x, padding_mask, target_list,):
        B, T, C = x.shape
        if self.mask_prob > 0:
            mask_indices = compute_mask_indices(
                (B, T),
                padding_mask,
                self.mask_prob,
                self.mask_length,
                self.mask_selection,
                self.mask_other,
                min_masks=2,
                no_overlap=self.no_mask_overlap,
                min_space=self.mask_min_space,
            )
            mask_indices = torch.from_numpy(mask_indices).to(x.device)
            x[mask_indices] = self.mask_emb
        else:
            mask_indices = None

        if self.mask_channel_prob > 0:
            mask_channel_indices = compute_mask_indices(
                (B, C),
                None,
                self.mask_channel_prob,
                self.mask_channel_length,
                self.mask_channel_selection,
                self.mask_channel_other,
                no_overlap=self.no_mask_channel_overlap,
                min_space=self.mask_channel_min_space,
            )
            mask_channel_indices = (
                torch.from_numpy(mask_channel_indices)
                .to(x.device)
                .unsqueeze(1)
                .expand(-1, T, -1)
            )
            x[mask_channel_indices] = 0

        return x, mask_indices

    def compute_nce(self, x, pos, negs):
        neg_is_pos = (pos == negs).all(-1)
        pos = pos.unsqueeze(0)
        targets = torch.cat([pos, negs], dim=0)

        logits = torch.cosine_similarity(
            x.float(), targets.float(), dim=-1
        ).type_as(x)
        logits /= self.logit_temp
        if neg_is_pos.any():
            logits[1:][neg_is_pos] = float("-inf")
        logits = logits.transpose(0, 1)  # (num_x, num_cls+1)
        return logits

    def forward_features(self, source: torch.Tensor) -> torch.Tensor:
        if self.feature_grad_mult > 0:
            features = self.feature_extractor(source)
            if self.feature_grad_mult != 1.0:
                features = GradMultiply.apply(features, self.feature_grad_mult)
        else:
            with torch.no_grad():
                features = self.feature_extractor(source)
        # check if an entire row of features is NaN
        entire_row_nans = torch.all(
            torch.all(torch.isnan(features), dim=2), dim=1
        )
        if entire_row_nans.any():
            logger.info(f"all NaN in features row(s) {entire_row_nans}")
            # set this row to random values
            features = features.clone()
            features[entire_row_nans] = torch.randn(
                features[entire_row_nans].size(), device=features.device, dtype=features.dtype
            )
        return features, entire_row_nans

    def forward_targets(
        self, features: torch.Tensor, target_list: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Trim features to ensure labels exist and then get aligned labels
        feat_tsz = features.size(2)
        targ_tsz = min([t.size(1) for t in target_list])
        if self.feat2tar_ratio * feat_tsz > targ_tsz:
            feat_tsz = int(targ_tsz / self.feat2tar_ratio)
            features = features[..., :feat_tsz]
        target_inds = torch.arange(feat_tsz).float() * self.feat2tar_ratio
        target_list = [t[:, target_inds.long()] for t in target_list]
        
        return features, target_list

    def forward_padding_mask(
        self, features: torch.Tensor, padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        extra = padding_mask.size(1) % features.size(1)
        if extra > 0:
            padding_mask = padding_mask[:, :-extra]
        padding_mask = padding_mask.view(
            padding_mask.size(0), features.size(1), -1
        )
        padding_mask = padding_mask.all(-1)
        return padding_mask

    def forward(
        self,
        source: torch.Tensor,
        spk: torch.Tensor,
        id: torch.LongTensor,
        target_list: Optional[List[torch.Tensor]] = None,
        padding_mask: Optional[torch.Tensor] = None,
        mask: bool = True,
        features_only: bool = False,
        output_layer: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """output layer is 1-based"""
        features, entire_row_nans = self.forward_features(source)

        features, target_list = self.forward_targets(features, target_list)

        features_pen = features.float().pow(2).mean()

        features = features.transpose(1, 2)
        features = self.layer_norm(features)
        unmasked_features = features.clone()

        if padding_mask is not None:
            padding_mask = self.forward_padding_mask(features, padding_mask)

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        features = self.dropout_input(features)
        unmasked_features = self.dropout_features(unmasked_features)

        if mask:
            x, mask_indices = self.apply_mask(
                features, padding_mask, target_list,
            )
        else:
            x = features
            mask_indices = None

        # feature: (B, T, D), float
        # target: (B, T), long
        # x: (B, T, D), float
        # padding_mask: (B, T), bool
        # mask_indices: (B, T), bool
        x, layer_results = self.encoder(
            x,
            padding_mask=padding_mask,
            layer=self.encoder_output_layer,
            dropout_return_x=(self.spkr_class_layer_mode in ["all", "random"] or isinstance(self.spkr_class_layer_mode, int)),
            # layer=None if output_layer is None else output_layer - 1
        )

        if features_only:
            return {"x": x, "padding_mask": padding_mask, "features": features}

        def compute_pred(proj_x, target, label_embs):
            # compute logits for the i-th label set
            y = torch.index_select(label_embs, 0, target.long())
            negs = label_embs.unsqueeze(1).expand(-1, proj_x.size(0), -1)
            if self.target_glu:
                y = self.target_glu(y)
                negs = self.target_glu(negs)
            # proj_x: (S, D)
            # y: (S, D)
            # negs: (Neg, S, D)
            return self.compute_nce(proj_x, y, negs)

        label_embs_list = self.label_embs_concat.split(self.num_classes, 0)


        masked_indices = torch.logical_and(~padding_mask, mask_indices)
        if not self.skip_masked:
            proj_x_m = self.final_proj(x[masked_indices])
            if self.untie_final_proj:
                proj_x_m_list = proj_x_m.chunk(len(target_list), dim=-1)
            else:
                proj_x_m_list = [proj_x_m for _ in range(len(target_list))]
            logit_m_list = [
                compute_pred(proj_x_m, t[masked_indices], label_embs_list[i])
                for i, (proj_x_m, t) in enumerate(
                    zip(proj_x_m_list, target_list)
                )
            ]
        else:
            logit_m_list = [None for _ in target_list]

        nomask_indices = torch.logical_and(~padding_mask, ~mask_indices)
        if not self.skip_nomask:
            proj_x_u = self.final_proj(x[nomask_indices])
            if self.untie_final_proj:
                proj_x_u_list = proj_x_u.chunk(len(target_list), dim=-1)
            else:
                proj_x_u_list = [proj_x_u for _ in range(len(target_list))]

            logit_u_list = [
                compute_pred(proj_x_u, t[nomask_indices], label_embs_list[i])
                for i, (proj_x_u, t) in enumerate(
                    zip(proj_x_u_list, target_list)
                )
            ]
        else:
            logit_u_list = [None for _ in target_list]

        if self.span_loss:
            label_embs_span_list = self.label_embs_concat_span.split(self.num_classes, 0)
            # concatenate left and right boundaries to x to handle edge cases where span begins at frame 0 or ends at last frame
            x_span_padded = torch.cat(
                [
                    self.left_boundary_emb.expand(x.size(0), 1, x.size(2)),
                    x,
                    self.right_boundary_emb.expand(x.size(0), 1, x.size(2)),
                ], dim=1
            )
            mask_indices_padded = torch.cat(
                [
                    torch.zeros((mask_indices.size(0), 1), dtype=torch.bool, device=mask_indices.device),
                    mask_indices,
                    torch.zeros((mask_indices.size(0), 1), dtype=torch.bool, device=mask_indices.device),
                ], dim=1
            )

            span_starts, span_ends, from_starts, to_ends = compute_span_start_end(mask_indices_padded)
            # get deduplicated span starts/ends for efficient processing
            dedup_span_starts_flat, dedup_span_ends_flat, dedup2orig_idx_flat, dedup_batch_start_idx_flat = deduplicate_span_start_ends(
                span_starts, span_ends, mask_indices_padded
            )
            dedup_batch_end_idx_flat = dedup_batch_start_idx_flat + mask_indices_padded.size(1) - 1
            
            dedup_span_reps = torch.cat(
                [
                    x_span_padded.view(-1, x_span_padded.size(2)).index_select(0, dedup_span_starts_flat),
                    x_span_padded.view(-1, x_span_padded.size(2)).index_select(0, dedup_span_ends_flat),
                ], dim=-1,
            ) # (num_masked_spans, 2*C)
            # map back to original masked frames
            span_reps = dedup_span_reps.index_select(0, dedup2orig_idx_flat) # (num_masked_frames, 2*C)

            # get relative position embeddings
            from_starts_masked = from_starts.masked_select(mask_indices_padded) - 1
            to_ends_masked = to_ends.masked_select(mask_indices_padded) - 1
            # clamp from_starts, to_ends to max pos embedding index
            from_starts_masked = torch.clamp(from_starts_masked, 0, self.span_loss_max_pos - 1)
            to_ends_masked = torch.clamp(to_ends_masked, 0, self.span_loss_max_pos - 1)
            rel_pos_embs = torch.cat(
                [
                    self.left_rel_pos_emb(from_starts_masked),
                    self.right_rel_pos_emb(to_ends_masked),
                ], dim=-1,
            )
            span_reps = span_reps + rel_pos_embs

            proj_x_span = self.final_span_proj(span_reps)
            if self.untie_final_proj:
                proj_x_span_list = proj_x_span.chunk(len(target_list), dim=-1)
            else:
                proj_x_span_list = [proj_x_span for _ in range(len(target_list))]
            logit_span_list = [
                compute_pred(proj_x_span, t[mask_indices], label_embs_span_list[i])
                for i, (proj_x_span, t) in enumerate(
                    zip(proj_x_span_list, target_list)
                )
            ]
        else:
            logit_span_list = [None for _ in target_list]

        # print_memory_usage("before spkr class layer")

        # speaker classification prediction
        if self.spkr_class_layer is not None:
            if self.spkr_class_layer_mode == "final":
                spkr_class_input = x
                # spkr_adv_logits = self.spkr_class_layer(x)
            elif self.spkr_class_layer_mode == "random":
                # select a random layer from layer_results
                rand_layer_idx = np.random.randint(len(layer_results))
                spkr_class_input = layer_results[rand_layer_idx][0].transpose(0, 1)
            elif self.spkr_class_layer_mode == "all":
                # stack all layers
                spkr_class_input = torch.stack(
                    [lr[0].transpose(0, 1) for lr in layer_results], dim=0
                )
            elif isinstance(self.spkr_class_layer_mode, int):
                spkr_class_input = layer_results[self.spkr_class_layer_mode][0].transpose(0, 1)
            spkr_adv_logits = self.spkr_class_layer(spkr_class_input)
        else:
            spkr_adv_logits = None

        result = {
            "logit_m_list": logit_m_list,
            "logit_u_list": logit_u_list,
            "logit_span_list": logit_span_list,
            "padding_mask": padding_mask,
            "features_pen": features_pen,
            "spkr_adv_logits": spkr_adv_logits,
            "mask_indices": mask_indices,
            "nomask_indices": nomask_indices,
        }
        return result

    def extract_features(
        self,
        source: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        mask: bool = False,
        ret_conv: bool = False,
        output_layer: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        res = self.forward(
            source,
            padding_mask=padding_mask,
            mask=mask,
            features_only=True,
            output_layer=output_layer,
        )
        feature = res["features"] if ret_conv else res["x"]
        return feature, res["padding_mask"]

    def get_logits(self, net_output, is_masked=True):
        if is_masked:
            logits_list = net_output["logit_m_list"]
        else:
            logits_list = net_output["logit_u_list"]
        logits_list = [x.float() for x in logits_list if x is not None]
        return logits_list
    
    def get_span_logits(self, net_output):
        logits_list = net_output["logit_span_list"]
        logits_list = [x.float() for x in logits_list if x is not None]
        return logits_list

    def get_targets(self, net_output, is_masked=True):
        logits_list = self.get_logits(net_output, is_masked)
        targets_list = [
            x.new_zeros(x.size(0), dtype=torch.long) for x in logits_list
        ]
        return targets_list
    
    def get_spkr_adv_logits(self, net_output):
        return net_output["spkr_adv_logits"]
    
    def get_spkr_adv_targets(self, net_output, spk):
        """
        input:
            spk: tensor (B,) of speaker ids
            net_output["spkr_adv_logits"]: tensor (B,T,S) of logits over S speakers, used for shape (T=1 if pooling layer is used)
        output:
            spkr_adv_targets: tensor (B,T) of speaker ids
        """
        if net_output["spkr_adv_logits"] is not None:
            spkr_adv_targets = spk.unsqueeze(1).expand(-1, net_output["spkr_adv_logits"].size(1))
        else:
            spkr_adv_targets = None
        return spkr_adv_targets
    
    def spkr_adv_loss(self, spkr_adv_logits, spkr_adv_targets):
        """
        input:
            spkr_adv_logits: tensor (B,T,S) of logits over S speakers
            spkr_adv_targets: tensor (B,T) of speaker ids
        output:
            spkr_adv_loss: tensor (1,) of loss
        """
        loss = torch.nn.functional.cross_entropy(
            spkr_adv_logits.flatten(0, 1),
            spkr_adv_targets.flatten(),
            reduction='mean',
        )
        return loss

    def get_extra_losses(self, net_output):
        extra_losses = []
        names = []

        if "features_pen" in net_output:
            extra_losses.append(net_output["features_pen"])
            names.append("features_pen")

        return extra_losses, names

    def remove_pretraining_modules(self):
        self.target_glu = None
        self.final_proj = None
