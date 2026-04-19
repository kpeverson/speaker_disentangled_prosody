# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import json
import logging
import os
import sys
from typing import Any, List, Optional, Union

import librosa
import numpy as np
import scipy

import torch
import torch.nn.functional as F
import torchaudio
from fairseq.data import data_utils
from fairseq.data.fairseq_dataset import FairseqDataset
from fairseq.data.audio.hubert_dataset import HubertDataset

logger = logging.getLogger(__name__)


def load_audio(manifest_path, max_keep, min_keep, utt2spkr):
    n_long, n_short = 0, 0
    names, inds, sizes, spks = [], [], [], []
    with open(manifest_path) as f:
        root = f.readline().strip()
        for ind, line in enumerate(f):
            items = line.strip().split("\t")
            assert len(items) == 2, line
            name = items[0]
            sz = int(items[1])
            if utt2spkr is not None:
                spk = utt2spkr.get(name, -1)
            else:
                spk = -1
            if min_keep is not None and sz < min_keep:
                n_short += 1
            elif max_keep is not None and sz > max_keep:
                n_long += 1
            else:
                names.append(name)
                inds.append(ind)
                sizes.append(sz)
                spks.append(spk)
    tot = ind + 1
    logger.info(
        (
            f"max_keep={max_keep}, min_keep={min_keep}, "
            f"loaded {len(names)}, skipped {n_short} short and {n_long} long, "
            f"longest-loaded={max(sizes)}, shortest-loaded={min(sizes)}"
        )
    )
    return root, names, inds, tot, sizes, spks


def load_label(label_path, inds, tot):
    with open(label_path) as f:
        labels = [line.rstrip() for line in f]
        assert (
            len(labels) == tot
        ), f"number of labels does not match ({len(labels)} != {tot})"
        labels = [labels[i] for i in inds]
    return labels


def load_label_offset(label_path, inds, tot):
    with open(label_path) as f:
        code_lengths = [len(line.encode("utf-8")) for line in f]
        assert (
            len(code_lengths) == tot
        ), f"number of labels does not match ({len(code_lengths)} != {tot})"
        offsets = list(itertools.accumulate([0] + code_lengths))
        offsets = [(offsets[i], offsets[i + 1]) for i in inds]
    return offsets


def verify_label_lengths(
    audio_sizes,
    audio_rate,
    label_path,
    label_rate,
    inds,
    tot,
    tol=0.1,  # tolerance in seconds
    allowed_factors=[1],
):
    if label_rate < 0:
        logger.info(f"{label_path} is sequence label. skipped")
        return

    with open(label_path) as f:
        lengths = [len(line.rstrip().split()) for line in f]
        assert len(lengths) == tot
        lengths = [lengths[i] for i in inds]
    num_invalid = 0
    for i, ind in enumerate(inds):
        dur_from_audio = audio_sizes[i] / audio_rate
        dur_from_label = lengths[i] / label_rate
        # if abs(dur_from_audio - dur_from_label) > tol:
        if all([abs(dur_from_audio*float(f)-dur_from_label) > tol for f in allowed_factors]):
            logger.warning(
                (
                    f"audio and label duration differ too much "
                    f"(|{dur_from_audio} - {dur_from_label}| > {tol}) "
                    f"in line {ind+1} of {label_path}. Check if `label_rate` "
                    f"is correctly set (currently {label_rate}). "
                    f"num. of samples = {audio_sizes[i]}; "
                    f"label length = {lengths[i]}"
                )
            )
            num_invalid += 1
    if num_invalid > 0:
        logger.warning(
            f"total {num_invalid} (audio, label) pairs with mismatched lengths"
        )

class ProsodyvecDataset(HubertDataset):
    def __init__(
        self,
        manifest_path: str,
        utt2spkr_path: Optional[str],
        sample_rate: float,
        label_paths: List[str],
        label_rates: Union[List[float], float],  # -1 for sequence labels
        pad_list: List[str],
        eos_list: List[str],
        label_processors: Optional[List[Any]] = None,
        max_keep_sample_size: Optional[int] = None,
        min_keep_sample_size: Optional[int] = None,
        max_sample_size: Optional[int] = None,
        shuffle: bool = True,
        pad_audio: bool = False,
        normalize: bool = False,
        store_labels: bool = True,
        random_crop: bool = False,
        crop: bool = False,
        single_target: bool = False,
        use_glottal_source: bool = True,
        lpc_order: int = 16,
        p_half_band: float = 0.0,
        mask: bool = False,
    ):
        super().__init__(
            manifest_path,
            sample_rate,
            label_paths,
            label_rates,
            pad_list,
            eos_list,
            label_processors,
            max_keep_sample_size,
            min_keep_sample_size,
            max_sample_size,
            shuffle,
            pad_audio,
            normalize,
            store_labels,
            random_crop,
            single_target,
        )
        self.metadata_dir = os.path.dirname(manifest_path)
        self.utt2spkr_path = utt2spkr_path
        self.utt2spkr = self.get_utt2spkr()
        self.audio_root, self.audio_names, inds, tot, self.sizes, self.spks = load_audio(
            manifest_path, max_keep_sample_size, min_keep_sample_size, self.utt2spkr
        )
        self.crop = crop
        self.use_glottal_source = use_glottal_source
        self.lpc_order = lpc_order
        self.p_half_band = p_half_band
        assert 0 <= p_half_band <= 1

        self.mask = mask
        for label_path, label_rate in zip(label_paths, self.label_rates):
            verify_label_lengths(
                self.sizes, sample_rate, label_path, label_rate, inds, tot,
                allowed_factors=[1, 1/2, 2/3, 3/2, 2]
            )

    def get_utt2spkr(self):
        if self.utt2spkr_path is None:
            return None
        # assert that utt2spkr_path is a file
        assert os.path.isfile(self.utt2spkr_path)
        with open(self.utt2spkr_path) as f:
            utt2spkr = json.load(f)
        return utt2spkr

    def half_band_audio(self, wav):
        """
        make audio half-band by downsampling then upsampling by factor of 2
        """
        wav = torchaudio.transforms.Resample(
            self.sample_rate, self.sample_rate // 2
        )(wav)
        wav = torchaudio.transforms.Resample(
            self.sample_rate // 2, self.sample_rate
        )(wav)
        return wav
    
    def postprocess(self, wav, curr_sample_rate):

        if curr_sample_rate != self.sample_rate:
            wav = torchaudio.transforms.Resample(
                curr_sample_rate, self.sample_rate
            )(wav)

        if self.normalize:
            with torch.no_grad():
                wav = F.layer_norm(wav, wav.shape)

        return wav

    def get_audio(self, index):
        fileName = self.audio_names[index]
        fileLen = self.sizes[index]
        spk = self.spks[index]
        wav_path = os.path.join(self.audio_root, fileName)

        # load audio with torchaudio
        wav, curr_sample_rate = torchaudio.load(wav_path)
        if wav.ndim == 2:
            wav = wav.mean(0)
        assert wav.ndim == 1, wav.ndim
        if self.crop:
            wav = wav[:fileLen]
        wav = wav.float()

        wav = self.postprocess(wav, curr_sample_rate)
        if self.p_half_band > 0:
            if torch.rand(1).item() < self.p_half_band:
                wav = self.half_band_audio(wav)

        spk_emb = torch.zeros(1)

        return wav, spk_emb, spk
    
    def collater(self, samples):
        # target = max(sizes) -> random_crop not used
        # target = max_sample_size -> random_crop used for long
        samples = [s for s in samples if s["source"] is not None]
        if len(samples) == 0:
            return {}

        audios = [s["source"] for s in samples]
        audio_sizes = [len(s) for s in audios]
        if self.pad_audio:
            audio_size = min(max(audio_sizes), self.max_sample_size)
        else:
            audio_size = min(min(audio_sizes), self.max_sample_size)
        collated_audios, padding_mask, audio_starts = self.collater_audio(
            audios, audio_size
        )

        # speakers
        spks = [s["spk"] for s in samples]
        collated_spks = self.collater_speaker(spks)

        targets_by_label = [
            [s["label_list"][i] for s in samples]
            for i in range(self.num_labels)
        ]
        targets_list, lengths_list, ntokens_list = self.collater_label(
            targets_by_label, audio_size, audio_starts
        )

        net_input = {
            "source": collated_audios, 
            "padding_mask": padding_mask,
            "spk": collated_spks,
        }
        batch = {
            "id": torch.LongTensor([s["id"] for s in samples]),
            "net_input": net_input,
        }

        if self.single_target:
            batch["target_lengths"] = lengths_list[0]
            batch["ntokens"] = ntokens_list[0]
            batch["target"] = targets_list[0]
        else:
            batch["target_lengths_list"] = lengths_list
            batch["ntokens_list"] = ntokens_list
            batch["target_list"] = targets_list
        return batch
    
    def collater_speaker(self, spks):
        return torch.LongTensor(spks)

    def get_glottal(self, wav):
        """
        obtains estimated glottal source from audio using inverse filtering from librosa.lpc
        """
        a = librosa.lpc(wav.numpy(), order=self.lpc_order)
        wav_hat = torch.from_numpy(
            scipy.signal.lfilter(
                np.hstack([[0], -1 * a[1:]]), [1], wav.numpy()
            )
        ).to(wav.dtype)
        glottal = wav - wav_hat

        # in very few cases, glottal may contain NaNs
        # replace NaNs with zeros
        if torch.isnan(glottal).any():
            glottal[torch.isnan(glottal)] = 0

        return glottal
    
    def __getitem__(self, index):
        wav, spk_emb, spk = self.get_audio(index)
        labels = self.get_labels(index)
        if self.use_glottal_source:
            glottal = self.get_glottal(wav)
            return {
                "id": index,
                "source": glottal,
                "label_list": labels,
                "spk_emb": spk_emb,
                "spk": spk,
            }
        else:
            return {
                "id": index,
                "source": wav,
                "label_list": labels,
                "spk_emb": spk_emb,
                "spk": spk,
            }