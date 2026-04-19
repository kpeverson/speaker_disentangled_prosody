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
from scipy import signal

import h5pickle as h5py
# import h5py
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
    # names, inds, spks = [], [], []
    with open(manifest_path) as f:
        # root = f.readline().strip()
        for ind, line in enumerate(f):
            items = line.strip().split("\t")
            assert len(items) == 2, line
            name = items[0]
            sz = int(items[1])
            if utt2spkr is not None:
                spk = utt2spkr.get(name, 0) #-1)
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
    return names, inds, tot, sizes, spks


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

class ProsodyvecDatasetH5(HubertDataset):
    def __init__(
        self,
        manifest_path: str,
        h5_path: str,
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
        p_half_band: float = 0.0,
        mask: bool = False,
        lpf_cutoff: Optional[float] = None,
        syl_segments_h5_path: Optional[str] = None,
        syl_segments_filter_threshold: Optional[float] = None,
    ):
        self.h5_path = h5_path
        self.h5_file = None
        self.syl_segments_h5_path = syl_segments_h5_path
        self.syl_segments_h5_file = None
        self.metadata_dir = os.path.dirname(manifest_path)
        self.utt2spkr_path = utt2spkr_path
        self.utt2spkr = self.get_utt2spkr()
        self.audio_names, inds, tot, self.sizes, self.spks = load_audio(
            manifest_path, max_keep_sample_size, min_keep_sample_size, self.utt2spkr
        )

        # copied from HubertDataset
        self.sample_rate = sample_rate
        self.shuffle = shuffle
        self.random_crop = random_crop

        self.num_labels = len(label_paths)
        self.pad_list = pad_list
        self.eos_list = eos_list
        self.label_processors = label_processors
        self.single_target = single_target
        self.label_rates = (
            [label_rates for _ in range(len(label_paths))]
            if isinstance(label_rates, float)
            else label_rates
        )
        self.store_labels = store_labels
        if store_labels:
            self.label_list = [load_label(p, inds, tot) for p in label_paths]
        else:
            self.label_paths = label_paths
            self.label_offsets_list = [
                load_label_offset(p, inds, tot) for p in label_paths
            ]
        assert (
            label_processors is None
            or len(label_processors) == self.num_labels
        )

        self.max_sample_size = (
            max_sample_size if max_sample_size is not None else sys.maxsize
        )
        self.pad_audio = pad_audio
        self.normalize = normalize
        logger.info(
            f"pad_audio={pad_audio}, random_crop={random_crop}, "
            f"normalize={normalize}, max_sample_size={self.max_sample_size}"
        )

        self.crop = crop
        self.p_half_band = p_half_band
        assert 0 <= self.p_half_band <= 1
        self.lpf_cutoff = lpf_cutoff
        if self.lpf_cutoff is not None:
            assert self.lpf_cutoff > 0

        self.mask = mask

        self.syl_segments_filter_threshold = syl_segments_filter_threshold

    def _get_h5_file(self):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, "r")
        return self.h5_file

    def _get_syl_segments_h5_file(self):
        if self.syl_segments_h5_file is None and self.syl_segments_h5_path is not None:
            self.syl_segments_h5_file = h5py.File(self.syl_segments_h5_path, "r")
        return self.syl_segments_h5_file

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
    
    def lpf(self, wav):
        """
        apply low-pass butterworth filter using signal.sosfiltfilt
        """
        sos = signal.butter(4, self.lpf_cutoff, "low", fs=self.sample_rate, output="sos")
        wav = signal.sosfiltfilt(sos, wav.numpy())
        return torch.from_numpy(wav.copy())
    
    def postprocess(self, wav): 

        if self.normalize:
            with torch.no_grad():
                wav = F.layer_norm(wav, wav.shape)

        return wav

    def get_audio(self, index):
        fileName = self.audio_names[index]
        fileLen = self.sizes[index]
        spk = self.spks[index]

        h5_file = self._get_h5_file()

        # load audio with torchaudio
        try:
            wav = h5_file[fileName][:]
        except:
            raise Exception(f"Error getting key {fileName} from h5_file (type {type(self.h5_file)}) {self.h5_file}")
        wav = torch.from_numpy(wav)
        if self.lpf_cutoff is not None:
            wav = self.lpf(wav)
        if wav.ndim == 2:
            wav = wav.mean(0)
        assert wav.ndim == 1, wav.ndim
        if self.crop:
            wav = wav[:fileLen]
        wav = wav.float()

        wav = self.postprocess(wav)
        if self.p_half_band > 0:
            if torch.rand(1).item() < self.p_half_band:
                wav = self.half_band_audio(wav)

        spk_emb = torch.zeros(1)

        return wav, spk_emb, spk
    
    def get_syl_segments(self, index):
        fileName = self.audio_names[index]

        syl_segments_h5_file = self._get_syl_segments_h5_file()

        if syl_segments_h5_file is None:
            return None
        else:
            try:
                syl_segments = syl_segments_h5_file[fileName][:]
                syl_segments = torch.tensor(syl_segments.copy(), dtype=torch.float32)
            except:
                raise Exception(f"Error getting key {fileName} from syl_segments_h5_file (type {type(self.syl_segments_h5_file)}) {self.syl_segments_h5_file}")
            return syl_segments
    
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
        if "syl_segments" in samples[0]:
            syl_segments = [s["syl_segments"] for s in samples]
            syl_segments, syl_segments_padding_mask = self.collater_syl_segments(syl_segments, audio_starts, audio_size)
            net_input["syl_segments"] = syl_segments
            net_input["syl_segments_padding_mask"] = syl_segments_padding_mask

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

    def collater_syl_segments(self, syl_segments, audio_starts, audio_size):
        # subtract audio_starts (in seconds) from syl_segments
        syl_segments = [
            s - (start / self.sample_rate) for s, start in zip(syl_segments, audio_starts)
        ]
        # filter segments with start < 0
        syl_segments = [s[s[:,0] >= 0] for s in syl_segments]
        # filter segments with end > audio_size / sample_rate
        syl_segments = [s[s[:,1] <= (audio_size / self.sample_rate)] for s in syl_segments]
        if self.syl_segments_filter_threshold is not None:
            syl_segments = [s[s[:,1] - s[:,0] >= self.syl_segments_filter_threshold] for s in syl_segments]
        syl_lens = [s.shape[0] for s in syl_segments]
        max_syl_len = max(syl_lens)
        syl_segments_mask = torch.zeros(
            (len(syl_segments), max_syl_len), dtype=torch.bool
        )
        for i, l in enumerate(syl_lens):
            syl_segments_mask[i, l:] = 1
        syl_segments = torch.nn.utils.rnn.pad_sequence(
            syl_segments, batch_first=True, padding_value=-1.0
        )
        return syl_segments, syl_segments_mask
    
    def __getitem__(self, index):
        ret = {}
        wav, spk_emb, spk = self.get_audio(index)
        ret["id"] = index
        ret["source"] = wav
        ret["spk_emb"] = spk_emb
        ret["spk"] = spk
        labels = self.get_labels(index)
        ret["label_list"] = labels
        syl_segments = self.get_syl_segments(index)
        if syl_segments is not None:
            ret["syl_segments"] = syl_segments
        return ret
        
    def __del__(self):
        if self.h5_file is not None:
            self.h5_file.close()
        if self.syl_segments_h5_file is not None:
            self.syl_segments_h5_file.close()
