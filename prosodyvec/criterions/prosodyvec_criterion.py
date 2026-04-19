# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import re
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.criterions.hubert_criterion import HubertCriterionConfig
from fairseq.dataclass import FairseqDataclass

@dataclass
class ProsodyvecCriterionConfig(FairseqDataclass):
    pred_masked_weight: float = field(
        default=1.0,
        metadata={"help": "weight for predictive loss for masked frames"},
    )
    pred_nomask_weight: float = field(
        default=0.0,
        metadata={"help": "weight for predictive loss for unmasked frames"},
    )
    loss_weights: Optional[List[float]] = field(
        default=None,
        metadata={"help": "weights for additional loss terms (not first one)"},
    )
    log_keys: List[str] = field(
        default_factory=lambda: [],
        metadata={"help": "output keys to log"},
    )

@register_criterion("prosodyvec", dataclass=ProsodyvecCriterionConfig)
class ProsodyvecCriterion(FairseqCriterion):
    def __init__(self, task, pred_masked_weight, pred_nomask_weight, loss_weights=None, log_keys=None, streaming_chunk_size=None):
        super().__init__(task)
        self.pred_masked_weight = pred_masked_weight
        self.pred_nomask_weight = pred_nomask_weight
        self.pred_span_weight = task.cfg.pred_span_weight
        self.spkr_adv_weight = task.cfg.spkr_adv_weight
        self.loss_weights = loss_weights
        self.log_keys = [] if log_keys is None else log_keys

    def forward(self, model, sample, split="train", reduce=True, log_pred=False, **kwargs):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(target_list=sample["target_list"], **sample["net_input"], id=sample["id"])
        loss = 0.
        sample_size = 0
        num_masked = 0
        num_unmasked = 0
        logging_output = {}
        reduction = "sum" if reduce else "none"

        mask_indices = net_output["mask_indices"]
        nomask_indices = net_output["nomask_indices"]
        num_mask = mask_indices.sum().item()
        num_nomask = nomask_indices.sum().item()

        logging_output["num_masked"] = num_mask
        logging_output["num_unmasked"] = num_nomask

        loss_m_list = []
        logp_m_list = model.get_logits(net_output, True)
        targ_m_list = model.get_targets(net_output, True)
        assert self.pred_masked_weight == 0 or len(logp_m_list) > 0
        for i, (logp_m, targ_m) in enumerate(zip(logp_m_list, targ_m_list)):
            loss_m = F.cross_entropy(logp_m, targ_m, reduction=reduction)
            loss_m_list.append(loss_m)
            logging_output[f"loss_m_{i}"] = loss_m.detach().item()
        if self.pred_masked_weight > 0:
            loss += self.pred_masked_weight * sum(loss_m_list)
            sample_size += num_masked

        loss_u_list = []
        logp_u_list = model.get_logits(net_output, False)
        targ_u_list = model.get_targets(net_output, False)
        assert self.pred_nomask_weight == 0 or len(logp_u_list) > 0
        for i, (logp_u, targ_u) in enumerate(zip(logp_u_list, targ_u_list)):
            loss_u = F.cross_entropy(logp_u, targ_u, reduction=reduction)
            loss_u_list.append(loss_u)
            logging_output[f"loss_u_{i}"] = loss_u.detach().item()
        if self.pred_nomask_weight > 0:
            loss += self.pred_nomask_weight * sum(loss_u_list)
            sample_size += num_unmasked

        # span loss
        loss_span_list = []
        logp_span_list = model.get_span_logits(net_output)
        targ_span_list = model.get_targets(net_output, True)
        assert self.pred_span_weight == 0 or len(logp_span_list) > 0
        for i, (logp_span, targ_span) in enumerate(zip(logp_span_list, targ_span_list)):
            loss_span = F.cross_entropy(logp_span, targ_span, reduction=reduction)
            loss_span_list.append(loss_span)
            logging_output[f"loss_span_{i}"] = loss_span.detach().item()
        if self.pred_span_weight > 0:
            loss += self.pred_span_weight * sum(loss_span_list)
            sample_size += targ_span_list[0].numel()

        if self.loss_weights is not None:
            assert hasattr(model, "get_extra_losses")
            extra_losses, names = model.get_extra_losses(net_output)
            if torch.is_tensor(extra_losses):
                extra_losses = [extra_losses]
                names = [names]
            if len(self.loss_weights) == 1 and len(extra_losses) != 1:
                self.loss_weights = [self.loss_weights[0]] * len(extra_losses)
            assert len(extra_losses) == len(self.loss_weights), f"{len(extra_losses)}, {len(self.loss_weights)}"
            for p, n, coef in zip(extra_losses, names, self.loss_weights):
                if coef != 0 and p is not None:
                    p = coef * p.float() * sample_size
                    loss += p
                    logging_output[f"loss_{n}"] = p.item()

        # speaker classification loss
        if self.spkr_adv_weight > 0 and split == "train":
            spkr_adv_logits = model.get_spkr_adv_logits(net_output)
            spkr_adv_targets = model.get_spkr_adv_targets(net_output, sample["net_input"]["spk"])
            spkr_adv_loss = model.spkr_adv_loss(spkr_adv_logits, spkr_adv_targets)
            p_spkr_adv = spkr_adv_loss.float() * self.spkr_adv_weight * sample_size
            loss += p_spkr_adv
        else:
            p_spkr_adv = torch.tensor(0.0)
        logging_output["loss_spkr_adv"] = p_spkr_adv.item()

        logging_output = {
            "loss": loss.item() if reduce else loss,
            "ntokens": sample_size,
            "nsentences": sample["id"].numel(),
            "sample_size": sample_size,
            **logging_output,
        }

        for lk in self.log_keys:
            if lk in net_output:
                logging_output[lk] = float((net_output[lk]))

        def compute_pct_masked(n_masked_frames, n_unmasked_frames):
            denom = n_masked_frames + n_unmasked_frames
            if denom == 0:
                return 0.0
            else:
                return float(n_masked_frames) / denom

        def compute_correct(logits):
            if logits.numel() == 0:
                return 0, 0
            else:
                assert logits.dim() > 1, logits.shape
                max = logits.argmax(-1) == 0
                min = logits.argmin(-1) == 0
                both = max & min
                corr = max.long().sum().item() - both.long().sum().item()
                count = max.numel()
                return corr, count

        def compute_spkr_adv_correct(logits, targets):
            if logits.numel() == 0:
                return 0, 0
            else:
                assert logits.dim() > 1, logits.shape
                corr = (logits.argmax(-1) == targets).long().sum().item()
                count = logits.size(0) * logits.size(1)
                return corr, count

        with torch.no_grad():
            for i, logp_m in enumerate(logp_m_list):
                corr_m, count_m = compute_correct(logp_m)
                logging_output[f"correct_m_{i}"] = corr_m
                logging_output[f"count_m_{i}"] = count_m

            for i, logp_u in enumerate(logp_u_list):
                corr_u, count_u = compute_correct(logp_u)
                logging_output[f"correct_u_{i}"] = corr_u
                logging_output[f"count_u_{i}"] = count_u

            for i, logp_span in enumerate(logp_span_list):
                corr_span, count_span = compute_correct(logp_span)
                logging_output[f"correct_span_{i}"] = corr_span
                logging_output[f"count_span_{i}"] = count_span

            if self.spkr_adv_weight > 0 and split == "train":
                corr_spkr_adv, count_spkr_adv = compute_spkr_adv_correct(spkr_adv_logits, spkr_adv_targets)
                logging_output["correct_spkr_adv"] = corr_spkr_adv
                logging_output["count_spkr_adv"] = count_spkr_adv

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training (copied from normal cross entropy)."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar("loss", loss_sum / sample_size / math.log(2), sample_size, round=3)
        if sample_size != ntokens:
            metrics.log_scalar("nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3)
            metrics.log_derived("ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg))
        else:
            metrics.log_derived("ppl", lambda meters: utils.get_perplexity(meters["loss"].avg))

        counts = {}
        for lk in logging_outputs[0].keys():
            if lk.startswith("count_"):
                val = sum(log[lk] for log in logging_outputs)
                metrics.log_scalar(lk, val)
                counts[lk] = val

        for lk in logging_outputs[0].keys():
            if lk.startswith("loss_"):
                val = sum(log[lk] for log in logging_outputs)
                metrics.log_scalar(lk, val / sample_size / math.log(2), round=3)
            elif lk.startswith("correct_"):
                val = sum(log[lk] for log in logging_outputs)
                metrics.log_scalar(lk, val / counts[re.sub("correct", "count", lk)])

        # add pct_masked
        total_masked = sum(log.get("num_masked", 0) for log in logging_outputs)
        total_unmasked = sum(log.get("num_unmasked", 0) for log in logging_outputs)
        denom = total_masked + total_unmasked
        if denom == 0:
            metrics.log_scalar("pct_masked", 0.0)
        else:
            metrics.log_scalar(
                "pct_masked",
                sum(log.get("num_masked", 0) for log in logging_outputs) / denom
            )

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        raise NotImplementedError()

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return False
