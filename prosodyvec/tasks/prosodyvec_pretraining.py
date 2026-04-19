# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import logging
import os
import sys
from typing import Dict, List, Optional, Tuple

import h5pickle as h5py
import numpy as np

from dataclasses import dataclass, field
from fairseq.data import Dictionary, ProsodyvecDataset, ProsodyvecDatasetH5
from fairseq.dataclass.configs import FairseqDataclass
from fairseq.tasks import register_task
from fairseq.tasks.fairseq_task import FairseqTask
from omegaconf import MISSING

logger = logging.getLogger(__name__)


class LabelEncoder(object):
    def __init__(self, dictionary: Dictionary) -> None:
        self.dictionary = dictionary

    def __call__(self, label: str) -> List[str]:
        return self.dictionary.encode_line(
            label, append_eos=False, add_if_not_exist=False,
        )


@dataclass
class ProsodyvecPretrainingConfig(FairseqDataclass):
    data: str = field(
        default=MISSING, metadata={"help": "path to data directory"}
    )
    use_h5: bool = field(
        default=False, metadata={"help": "use h5 dataset if true"}
    )
    h5_path: str = field(
        default="",
        metadata={"help": "path to h5 dataset"},
    )
    syl_segments_h5_path: Optional[str] = field(
        default=None,
        metadata={"help": "path to h5 file containing syllable segments"},
    )
    utt2spkr_path: Optional[str] = field(
        default="",
        metadata={
            "help": "path to utt2spkr, if it exists",
        },
    )    
    fine_tuning: bool = field(
        default=False, metadata={"help": "set to true if fine-tuning Prosodyvec"}
    )
    labels: List[str] = field(
        default_factory=lambda: ["ltr"],
        metadata={
            "help": (
                "extension of the label files to load, frame-level labels for"
                " pre-training, and sequence-level label for fine-tuning"
            )
        },
    )
    label_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "if set, looks for labels in this directory instead",
        },
    )
    label_rate: float = field(
        default=-1,
        metadata={"help": "label frame rate. -1 for sequence label"},
    )
    sample_rate: int = field(
        default=16_000,
        metadata={
            "help": "target sample rate. audio files will be up/down "
            "sampled to this rate"
        },
    )
    normalize: bool = field(
        default=False,
        metadata={
            "help": "if set, normalizes input to have 0 mean and unit variance"
        },
    )
    enable_padding: bool = field(
        default=False,
        metadata={"help": "pad shorter samples instead of cropping"},
    )
    max_keep_size: Optional[int] = field(
        default=None,
        metadata={"help": "exclude sample longer than this"},
    )
    max_sample_size: Optional[int] = field(
        default=None,
        metadata={"help": "max sample size to crop to for batching"},
    )
    min_sample_size: Optional[int] = field(
        default=None,
        metadata={"help": "min sample size to crop to for batching"},
    )
    single_target: Optional[bool] = field(
        default=False,
        metadata={
            "help": "if set, AddTargetDatasets outputs same keys "
            "as AddTargetDataset"
        },
    )
    random_crop: Optional[bool] = field(
        default=True,
        metadata={"help": "always crop from the beginning if false"},
    )
    crop: Optional[bool] = field(
        default=False,
        metadata={"help": "crop using reference lengths in metadata"}
    )
    pad_audio: Optional[bool] = field(
        default=False,
        metadata={"help": "pad audio to the longest one in the batch if true"},
    )
    # use_glottal_source: Optional[bool] = field(
    #     default=True,
    #     metadata={"help": "use glottal source as input if true. otherwise, use raw waveform"},
    # )
    glottal_prob: Optional[float] = field(
        default=1.0,
        metadata={"help": "probability of using glottal source as input"},
    )
    lpc_order: Optional[int] = field(
        default=16,
        metadata={"help": "order of LPC glottal estimation"},
    )
    p_half_band: Optional[float] = field(
        default=0.0,
        metadata={"help": "probability of half band filter via downsampling and upsampling"},
    )
    lpf_cutoff: Optional[float] = field(
        default=None,
        metadata={"help": "cutoff frequency for low pass filter"},
    )
    pred_span_weight: float = field(
        default=0.0,
        metadata={"help": "weight for predictive loss for masked spans"},
    )
    spkr_adv_weight: Optional[float] = field(
        default=0.0,
        metadata={"help": "speaker classification loss weight"}
    )
    syl_segments_filter_threshold: Optional[float] = field(
        default=None,
        metadata={"help": "filter threshold for syllable segments"},
    )


@register_task("prosodyvec_pretraining", dataclass=ProsodyvecPretrainingConfig)
class ProsodyvecPretrainingTask(FairseqTask):

    cfg: ProsodyvecPretrainingConfig

    def __init__(
        self,
        cfg: ProsodyvecPretrainingConfig,
    ) -> None:
        super().__init__(cfg)

        logger.info(f"current directory is {os.getcwd()}")
        logger.info(f"ProsodyvecPretrainingTask Config {cfg}")

        self.cfg = cfg
        self.fine_tuning = cfg.fine_tuning

        if cfg.fine_tuning:
            self.state.add_factory("target_dictionary", self.load_dictionaries)
        else:
            self.state.add_factory("dictionaries", self.load_dictionaries)

        self.blank_symbol = "<s>"

        assert self.cfg.glottal_prob >= 0.0 and self.cfg.glottal_prob <= 1.0

    @property
    def source_dictionary(self) -> Optional[Dictionary]:
        return None

    @property
    def target_dictionary(self) -> Optional[Dictionary]:
        return self.state.target_dictionary

    @property
    def dictionaries(self) -> List[Dictionary]:
        return self.state.dictionaries

    @classmethod
    def setup_task(
        cls, cfg: ProsodyvecPretrainingConfig, **kwargs
    ) -> "ProsodyvecPretrainingTask":
        return cls(cfg)

    def load_dictionaries(self):
        label_dir = self.cfg.data if self.cfg.label_dir is None else self.cfg.label_dir
        dictionaries = [Dictionary.load(f"{label_dir}/dict.{label}.txt") for label in self.cfg.labels]
        return dictionaries[0] if self.cfg.fine_tuning else dictionaries

    def get_label_dir(self) -> str:
        if self.cfg.label_dir is None:
            return self.cfg.data
        return self.cfg.label_dir

    def load_dataset(self, split: str, **kwargs) -> None:
        manifest = f"{self.cfg.data}/{split}.tsv"
        utt2spkr_path = os.path.join(self.cfg.data, self.cfg.utt2spkr_path) if self.cfg.utt2spkr_path else None
        dicts = [self.target_dictionary] if self.cfg.fine_tuning else self.dictionaries
        pad_list = [dict.pad() for dict in dicts]
        eos_list = [dict.eos() for dict in dicts]
        procs = [LabelEncoder(dict) for dict in dicts]
        paths = [
            f"{self.get_label_dir()}/{split}.{l}" for l in self.cfg.labels
        ]

        # hubert v1: pad_audio=True, random_crop=False;
        if self.cfg.use_h5:
            self.datasets[split] = ProsodyvecDatasetH5(
                manifest,
                self.cfg.h5_path,
                syl_segments_h5_path=self.cfg.syl_segments_h5_path,
                utt2spkr_path=utt2spkr_path,
                sample_rate=self.cfg.sample_rate,
                label_paths=paths,
                label_rates=self.cfg.label_rate,
                pad_list=pad_list,
                eos_list=eos_list,
                label_processors=procs,
                max_keep_sample_size=self.cfg.max_keep_size,
                min_keep_sample_size=self.cfg.min_sample_size,
                max_sample_size=self.cfg.max_sample_size,
                pad_audio=self.cfg.pad_audio,
                normalize=self.cfg.normalize,
                store_labels=False,
                random_crop=self.cfg.random_crop,
                crop=self.cfg.crop,
                single_target=self.cfg.single_target,
                p_half_band=self.cfg.p_half_band,
                lpf_cutoff=self.cfg.lpf_cutoff,
                syl_segments_filter_threshold=self.cfg.syl_segments_filter_threshold,
            )
        else:
            print(f"loading dataset from {manifest}")
            self.datasets[split] = ProsodyvecDataset(
                manifest,
                utt2spkr_path=utt2spkr_path,
                sample_rate=self.cfg.sample_rate,
                label_paths=paths,
                label_rates=self.cfg.label_rate,
                pad_list=pad_list,
                eos_list=eos_list,
                label_processors=procs,
                max_keep_sample_size=self.cfg.max_keep_size,
                min_keep_sample_size=self.cfg.min_sample_size,
                max_sample_size=self.cfg.max_sample_size,
                pad_audio=self.cfg.pad_audio,
                normalize=self.cfg.normalize,
                store_labels=False,
                random_crop=self.cfg.random_crop,
                crop=self.cfg.crop,
                single_target=self.cfg.single_target,
                use_glottal_source=self.cfg.use_glottal_source,
                lpc_order=self.cfg.lpc_order,
                p_half_band=self.cfg.p_half_band,
            )

    def max_positions(self) -> Tuple[int, int]:
        return (sys.maxsize, sys.maxsize)

    def filter_indices_by_size(
        self, indices: np.array, *args, **kwargs
    ) -> np.array:
        return indices
