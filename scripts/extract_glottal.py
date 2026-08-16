import argparse
import os

import h5pickle as h5py
import librosa
import numpy as np
import scipy
import torch
import torchaudio
import tqdm

def inverse_filter(x, a, energy_threshold=1e-4):
    """
    return estimated glottal source using inverse filtering
    input:
        x: np.ndarray (T,) of source signal
        a: np.ndarray (L,) of LPC coefficients
    output:
        estimated glottal source: torch.Tensor (T,) of glottal source
    """
    if np.sum(x**2) < energy_threshold:
        # return x
        print(f"Energy of x is too low, returning x")
        return x
    x_hat = scipy.signal.lfilter(
                np.hstack([[0], -1 * a[1:]]), [1], x
            )
    glottal = x - x_hat

    return glottal

def forward_glottal(x, args, key=None):
    """
    return estimated glottal source using inverse filtering from librosa
    input:
        source: numpy array, shape (T,)
        args: argparse object
    output:
        glottal: numpy array, shape (T,)
    """
    glottal_source = np.zeros_like(x)
    frames = librosa.util.frame(x, frame_length=args.lpc_frame_length_samples, hop_length=args.lpc_frame_shift_samples).T
    
    if args.lpc_window == "hamming":
        window = np.hamming(args.lpc_frame_length_samples)
    else:
        raise ValueError(f"Unsupported window type: {args.lpc_window}")
    
    for i, frame in enumerate(frames):
        frame = frame*window
        a = librosa.lpc(frame, order=args.lpc_order)
        frame_glottal_source = inverse_filter(frame, a)
        glottal_source[i*args.lpc_frame_shift_samples:i*args.lpc_frame_shift_samples+args.lpc_frame_length_samples] += frame_glottal_source

    # print if any nans or large numbers exist
    if np.isnan(glottal_source).any():
        print(f"Bad glottal source (nans) for {key}")
    elif np.abs(glottal_source).max() > 50.0:
        print(f"Bad glottal source (max value > 50) for {key}")
    return glottal_source

def main(args):
    split_file = os.path.join(args.dataset_dir, f"{args.split}.txt")
    with open(split_file, "r") as f:
        keys_list = [line.strip() for line in f.readlines()]
    # shard the keys
    keys_list = keys_list[args.rank::args.nshards]

    output_hdf5_path = os.path.join(args.output_hdf5_dir, f"{args.split}_{args.rank}_{args.nshards}.hdf5")
    os.makedirs(args.output_hdf5_dir, exist_ok=True)
    with h5py.File(args.input_hdf5_path, "r") as rf, h5py.File(output_hdf5_path, "w") as wf:
        for i, key in tqdm.tqdm(enumerate(keys_list), total=len(keys_list)):
            x = rf[key][:].squeeze()
            glottal = forward_glottal(x, args, key=key)
            wf.create_dataset(key, data=glottal)
    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_hdf5_path", type=str, required=True)
    parser.add_argument("--output_hdf5_dir", type=str, required=True)
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--nshards", type=int, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--lpc_order", type=int, default=16)
    parser.add_argument("--lpc_frame_length", type=float, default=0.025)
    parser.add_argument("--lpc_frame_shift", type=float, default=0.010)
    parser.add_argument("--lpc_window", type=str, default="hamming")
    parser.add_argument("--sr", type=int, default=16000)
    args = parser.parse_args()

    args.lpc_frame_length_samples = int(args.lpc_frame_length * args.sr)
    args.lpc_frame_shift_samples = int(args.lpc_frame_shift * args.sr)


    main(args)