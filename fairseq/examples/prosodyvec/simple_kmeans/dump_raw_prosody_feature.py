# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import logging
import os
import sys

import fairseq
# import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
import torchcrepe

from feature_utils import get_path_iterator, get_h5_iterator, dump_feature, dump_feature_from_h5


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("dump_hubert_feature")

class AmplitudeHistogram(torch.nn.Module):
    
    def __init__(self, sr=16000, feature_rate=50):
        super().__init__()
        hop_length = int(sr / feature_rate)
        kernel = torch.ones(hop_length)/hop_length
        kernel = kernel.unsqueeze(0)
        self.conv = torch.nn.Conv1d(1, 1, hop_length, stride=hop_length, padding=hop_length//2, bias=False)
        self.conv.weight.data = kernel.unsqueeze(1)
        self.conv.requires_grad_(False)
        
    def forward(self, x):
        return self.conv(x.unsqueeze(1).abs()).squeeze(1)

class RawProsodyFeatureReader(object):
    """
    class to extract raw prosody features from wav files
    """
    def __init__(
        self,
        spectral_tilt=False,
        sr=16000,
        max_chunk=1600000,
        feature_rate=50,
        pitch_q=1,
        fmin=50,
        fmax=550,
        crepe_model="full",
        crepe_device="cuda:0",
        min_points=5,
        periodicity_threshold=0.0,
        loudness_threshold=0.1,
        load_from_h5_file=False,
        h5_file_path=None,
    ):
        self.max_chunk = max_chunk
        self.sr = sr
        self.feature_rate = feature_rate
        self.fmin = fmin
        self.fmax = fmax

        self.spectral_tilt = spectral_tilt
        if self.spectral_tilt:
            self.mfcc_transform = torchaudio.transforms.MFCC(
                sample_rate=sr,
                n_mfcc=13,
                log_mels=True,
                melkwargs={
                    "n_mels": 20,
                    "hop_length": int(sr / feature_rate),
                    # "f_min": fmin,
                    # "f_max": fmax,
                },
            )

        self.crepe_model = crepe_model
        self.crepe_device = crepe_device
        self.min_points = min_points
        self.periodicity_threshold = periodicity_threshold
        self.pitch_q = pitch_q
        if not self.spectral_tilt:
            self.intensity_model = AmplitudeHistogram(sr, feature_rate)
            self.intensity_model.eval().to(self.crepe_device)
            self.loudness_threshold = loudness_threshold

        self.load_from_h5_file = load_from_h5_file
        self.h5_file_path = h5_file_path

    # read audio using torchaudio
    def read_audio(self, path, ref_len=None):
        wav, sr = torchaudio.load(path)
        if sr != self.sr:
            sr_ratio = sr / self.sr
            wav = torchaudio.transforms.Resample(sr, self.sr)(wav)
            sr = self.sr
        else:
            sr_ratio = 1
        # average over 2 channels (also works for mono)
        if wav.ndim == 2:
            wav = wav.mean(0)
        assert wav.ndim == 1, wav.ndim
        if ref_len is not None and abs(ref_len - len(wav)*sr_ratio) > 160:
            logging.warning(f"ref {ref_len} != read {len(wav)} ({path})")
        return wav
    
    def read_audio_from_h5(self, h5_file, key):
        wav = h5_file[key][:]
        return torch.from_numpy(wav).squeeze(0)
    
    def get_f0_nccf(self, wav, log=False):
        """
        uses torchaudio.functional.compute_kaldi_pitch to extract f0 and NCCF features

        Args:
            wav (torch tensor, shape (T_w,)): waveform

        Returns:
            f0 (torch tensor, shape (T_f,)): f0 contour
        """
        frame_shift = int(1000* (1 / self.feature_rate)) # ms
        frame_length = frame_shift * 2
        pitch_feats = torchaudio.functional.compute_kaldi_pitch(
            wav, 
            sample_rate=self.sr,
            frame_length=frame_length,
            frame_shift=frame_shift,
            min_f0=self.fmin,
            max_f0=self.fmax,
        )
        nccf, f0 = pitch_feats[ ... , 0], pitch_feats[ ... , 1]
        if log:
            f0 = f0.log()
        delta_f0 = torch.cat(
            (
                f0[0].unsqueeze(0),
                f0[1:] - f0[:-1],
            ), 0
        )
        return f0, delta_f0, nccf
    
    def get_sparc_f0_periodicity(self, wav, log=True):
        """
        uses torchcrepe to extract f0 and periodicity features

        Args:
            wav (torch tensor, shape (T_w,)): waveform

        Returns:
            f0 (torch tensor, shape (T_f,)): f0 contour
            periodicity (torch tensor, shape (T_f,)): periodicity contour
        """

        def _reshape(arr,q):
            b = arr.shape[0]
            l = arr.shape[1]
            arr = arr[:,:int(l//q)*q]
            arr = arr.reshape(b,l//q,q)
            arr = arr.mean(-1)
            return arr
        
        def _threshold_periodicity(periodicity):
            if self.min_points >= 1:
                min_points = self.min_points
            else:
                min_points = int(self.min_points * len(periodicity))
            if (periodicity < self.periodicity_threshold).sum() < min_points:
                periodicity[periodicity < self.periodicity_threshold] = 0.0
            return periodicity

        pitch_hop_length = int(self.sr / (self.feature_rate * self.pitch_q))
        try:
            f0, periodicity = torchcrepe.predict(
                wav.unsqueeze(0),
                self.sr,
                pitch_hop_length,
                self.fmin,
                self.fmax,
                self.crepe_model,
                batch_size=2048,
                device=self.crepe_device,
                return_periodicity=True,
            )
            f0 = _reshape(f0, self.pitch_q) if self.pitch_q > 1 else f0
            periodicity = _reshape(periodicity, self.pitch_q) if self.pitch_q > 1 else periodicity
            periodicity = _threshold_periodicity(periodicity)
        except:
            # return all zeros
            output_size = int(wav.shape[0] * self.feature_rate // self.sr)
            logger.error(f"Error in torchcrepe.predict. returning zeros for f0 and periodicity with shape {(1, output_size)}")
            f0 = torch.zeros((1, output_size))
            periodicity = torch.zeros((1, output_size))
        if log:
            f0 = f0.log()
        delta_f0 = torch.cat(
            (
                f0[0].unsqueeze(0),
                f0[1:] - f0[:-1],
            ), 0
        )


        return f0.squeeze(0), delta_f0.squeeze(0), periodicity.squeeze(0)

    def get_fbank(self, wav, max_freq=500, num_bins=20, log=True):
        """
        Args:
            wav (torch tensor, shape (T_w,)): waveform
            
        Returns:
            fbank (torch tensor, shape (T_f, C)): log-mel filterbank energies <500 Hz
        """
        frame_shift = int(self.sr / self.feature_rate)
        frame_length = frame_shift * 2
        mel_feats = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sr,
            hop_length=frame_shift,
            n_mels=num_bins,
            f_max=max_freq,
        )(wav)
        if log:
            mel_feats = mel_feats.log()
        # remove first bin (DC component) since it's -inf
        mel_feats = mel_feats[1:]

        return mel_feats
    
    def get_energy(self, wav):
        """
        Args:
            wav (torch tensor, shape (T_w,)): waveform

        Returns:
            energy (torch tensor, shape (T_f,)): energy computed via sliding window RMS
        """
        frame_shift = int(self.sr / self.feature_rate)
        frame_length = frame_shift * 2
        # compute sample-level RMS
        sliding_windows = wav.unfold(0, frame_length, frame_shift)
        rms = sliding_windows.pow(2).mean(1).sqrt()
        return rms
    
    def get_sparc_loudness(self, wav):
        """
        Args:
            wav (torch tensor, shape (T_w,)): waveform

        Returns:
            loudness (torch tensor, shape (T_f,)): loudness contour
        """
        return self.intensity_model(
            wav.unsqueeze(0).to(self.crepe_device)
        ).cpu().squeeze(0)
    
    def get_spectral_tilt(self, wav):
        """
        Args:
            wav (torch tensor, shape (T_w,)): waveform
            
        Returns:
            tilt (torch tensor, shape (T_f,)): spectral tilt (first cepstral coefficient)
        """
        mfccs = self.mfcc_transform(wav)
        # return first cepstral coefficient (spectral tilt)
        tilt = mfccs[1, :]
        return tilt

    def get_feats(self, path, ref_len=None, h5_file=None):
        """
        Args:
            path (str): path to wav file

        Returns:
            feats (torch tensor, shape ): tensor of frame-level raw prosody features
                - log(F0): log of fundamental frequency
                - delta(log(F0)): delta of log of fundamental frequency
                - energy: energy
                - NCCF: normalized cross-correlation function
                - spectral: log-mel filterbank energies, <500 Hz
        """
        if self.load_from_h5_file:
            x = self.read_audio_from_h5(h5_file, path)
        else:
            x = self.read_audio(path, ref_len)
        f0, delta_f0, periodicity = self.get_sparc_f0_periodicity(x)
        if self.spectral_tilt:
            energy = self.get_spectral_tilt(x)
        else:
            energy = self.get_sparc_loudness(x)
        # concatenate features, using minimum length
        min_len = min(
            f0.shape[0],
            delta_f0.shape[0],
            energy.shape[0],
            periodicity.shape[0],
        )
        feats = torch.cat(
            (
                f0[:min_len].unsqueeze(0),
                delta_f0[:min_len].unsqueeze(0),
                energy[:min_len].unsqueeze(0),
                periodicity[:min_len].unsqueeze(0),
            ),
            0
        ).transpose(0, 1)
        return feats

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("tsv_dir")
    parser.add_argument("split")
    parser.add_argument("nshard", type=int)
    parser.add_argument("rank", type=int)
    parser.add_argument("feat_dir")
    parser.add_argument("--spectral_tilt", action="store_true", help="use spectral tilt (proxy - first cepstral coefficient) instead of RMS energy")
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--max_chunk", type=int, default=1600000)
    parser.add_argument("--feature_rate", type=float, default=50)
    parser.add_argument("--pitch_q", type=int, default=4)
    parser.add_argument("--fmin", type=int, default=50)
    parser.add_argument("--fmax", type=int, default=550)
    parser.add_argument("--crepe_model", type=str, default="full")
    parser.add_argument("--crepe_device", type=str, default="cuda:0")
    parser.add_argument("--min_points", type=int, default=5)
    parser.add_argument("--periodicity_threshold", type=float, default=0.0)
    parser.add_argument("--loudness_threshold", type=float, default=0.05)
    parser.add_argument("--load_from_h5_file", action="store_true", help="load from h5 file (supply key list dir in 'tsv_dir' argument)")
    parser.add_argument("--h5_file_path", type=str, default=None, help="h5 file to load from")
    args = parser.parse_args()
    logger.info(args)

    reader = RawProsodyFeatureReader(
        spectral_tilt=args.spectral_tilt,
        sr=args.sr,
        max_chunk=args.max_chunk,
        feature_rate=args.feature_rate,
        pitch_q=args.pitch_q,
        fmin=args.fmin,
        fmax=args.fmax,
        crepe_model=args.crepe_model,
        crepe_device=args.crepe_device,
        min_points=args.min_points,
        periodicity_threshold=args.periodicity_threshold,
        loudness_threshold=args.loudness_threshold,
        load_from_h5_file=args.load_from_h5_file,
        h5_file_path=args.h5_file_path,
    )
    if args.load_from_h5_file:
        assert args.h5_file_path is not None, "h5 file not provided"
        # warn that sampling rate must be consistent with the one used for h5 file
        Warning(f"Please verify that sampling rate {args.sr} is consistent with the one used for h5 file")
        generator, num = get_h5_iterator(f"{args.tsv_dir}/{args.split}.txt", args.nshard, args.rank)
        dump_feature_from_h5(reader, generator, num, args.split, args.nshard, args.rank, args.feat_dir, args.h5_file_path)
    else:
        generator, num = get_path_iterator(f"{args.tsv_dir}/{args.split}.tsv", args.nshard, args.rank)
        dump_feature(reader, generator, num, args.split, args.nshard, args.rank, args.feat_dir)

if __name__ == "__main__":

    main()
