import argparse
import glob
import json
import os

import numpy as np
from npy_append_array import NpyAppendArray
import tqdm

def compute_feat_stats_by_periodicity(feats, F0_idx=0, energy_idx=2, periodicity_idx=3):
    """
    computes weighted mean and std of F0 and energy using periodicity as weights
    """
    f0_avg = np.average(feats[:, F0_idx], weights=feats[:, periodicity_idx])
    f0_std = np.sqrt(np.average((feats[:, F0_idx] - f0_avg) ** 2, weights=feats[:, periodicity_idx]))
    energy_avg = np.average(feats[:, energy_idx], weights=feats[:, periodicity_idx])
    energy_std = np.sqrt(np.average((feats[:, energy_idx] - energy_avg) ** 2, weights=feats[:, periodicity_idx]))
    return f0_avg, f0_std, energy_avg, energy_std

def compute_single_feat_stats_by_periodicity(feat, periodicity):
    feat_avg = np.average(feat, weights=periodicity)
    feat_std = np.sqrt(np.average((feat - feat_avg) ** 2, weights=periodicity))
    return feat_avg, feat_std

def normalize_utt_feats(utt_feats, log_domain=True):
    normalized_feats = utt_feats.copy()
    unnormalized_feats = utt_feats.copy()

    if log_domain:
        # transform F0 to log domain
        normalized_feats[:, 0] = np.log(utt_feats[:, 0] + 1e-6)
        unnormalized_feats[:, 0] = np.log(utt_feats[:, 0] + 1e-6)

    # compute utterance-level weighted mean and std
    # f0_avg, f0_std, energy_avg, energy_std = compute_feat_stats_by_periodicity(normalized_feats[:, :])
    f0_avg, _ = compute_single_feat_stats_by_periodicity(
        normalized_feats[:, 0], normalized_feats[:, 3]
    )

    # mean-normalize F0 on utterance-level
    normalized_feats[:, 0] = normalized_feats[:, 0] - f0_avg

    # compute delta_F0 as difference between consecutive F0 frames
    normalized_feats[:, 1] = np.concatenate(
        [np.zeros(1), np.diff(normalized_feats[:, 0])]
    )
    unnormalized_feats[:, 1] = np.concatenate(
        [np.zeros(1), np.diff(unnormalized_feats[:, 0])]
    )

    # EDIT: don't normalize energy
    # z-normalize energy on utterance-level
    # normalized_feats[:, 2] = (utt_feats[:, 2] - energy_avg) / energy_std

    return normalized_feats, unnormalized_feats

def normalize_by_rank(args):
    """
    compute stats of features in given directory
    features should have size (nframes, 4) with each column containing:
    0: F0 (Hz)
    1: delta_F0 (Hz), will be changed after normalizin F0
    2: energy (actually first MFCC coefficient)
    3: periodicity

    nframes contains frames for many utterances, with their lengths stored in the corresponding .len file

    before clustering, the features will be normalized as follows:
    -- F0 and delta_F0 --
    1. transform F0 to log domain
    2. calculate utterance-level weighted mean using periodicity as weights
    3. mean-normalize F0 on utterance-level
    4. compute delta_F0 as difference between consecutive F0 frames
    5. normalize F0 by corpus-wide maximum abs value log+normalized F0 (can happen in learn_kmeans.py and dump_km_label.py)
    6. normalize delta_F0 by corpus-wide maximum abs value (can happen in learn_kmeans.py and dump_km_label.py)

    -- energy --
    1. calculate utterance-level weighted mean and std using periodicity as weights
    2. z-normalize energy on utterance-level
    3. normalize energy by corpus-wide maximum abs value energy (can happen in learn_kmeans.py and dump_km_label.py)

    -- periodicity --
    ~ none ~
    """
    normalized_feats_out_dir = os.path.join(args.feats_dir, "utt_normalized")
    unnormalized_feats_out_dir = os.path.join(args.feats_dir, "utt_unnormalized")
    os.makedirs(normalized_feats_out_dir, exist_ok=True)
    os.makedirs(unnormalized_feats_out_dir, exist_ok=True)
    for rank in range(args.nshard):
        # input files
        feat_path = os.path.join(
            args.feats_dir, f"{args.split}_{rank}_{args.nshard}.npy"
        )
        feats = np.load(feat_path, mmap_mode="r")
        len_path = os.path.join(
            args.feats_dir, f"{args.split}_{rank}_{args.nshard}.len"
        )
        with open(len_path, "r") as f:
            lens = [int(line.strip()) for line in f]

        # output files
        normalized_npy_out_file = os.path.join(
            normalized_feats_out_dir, f"{args.split}_{rank}_{args.nshard}.npy"
        )
        normalized_len_out_file = os.path.join(
            normalized_feats_out_dir, f"{args.split}_{rank}_{args.nshard}.len"
        )
        unnormalized_npy_out_file = os.path.join(
            unnormalized_feats_out_dir, f"{args.split}_{rank}_{args.nshard}.npy"
        )
        unnormalized_len_out_file = os.path.join(
            unnormalized_feats_out_dir, f"{args.split}_{rank}_{args.nshard}.len"
        )
        # remove these if they exist
        for filename in [
            normalized_npy_out_file,
            normalized_len_out_file,
            unnormalized_npy_out_file,
            unnormalized_len_out_file,
        ]:
            if os.path.exists(filename):
                os.remove(filename)

        # create symlinks from original len files to new ones
        if not os.path.exists(normalized_len_out_file):
            os.symlink(len_path, normalized_len_out_file)
        if not os.path.exists(unnormalized_len_out_file):
            os.symlink(len_path, unnormalized_len_out_file)

        with NpyAppendArray(normalized_npy_out_file) as normalized_npy_out, \
                NpyAppendArray(unnormalized_npy_out_file) as unnormalized_npy_out, \
                open(normalized_len_out_file, "r") as normalized_len_out, \
                open(unnormalized_len_out_file, "r") as unnormalized_len_out:
            
            idx2utt = np.concatenate([np.repeat(i, l) for i, l in enumerate(lens)])
            utts = set(idx2utt)
            # normalized_feats = np.zeros_like(feats)
            # unnormalized_feats = np.zeros_like(feats)
            for utt in tqdm.tqdm(
                utts, desc=f"normalizing {args.split} rank {rank}", ncols=80, total=len(utts)
            ):
                idxs = np.where(idx2utt == utt)[0]
                utt_feats = feats[idxs]

                # normalize features
                normalized_feats, unnormalized_feats = normalize_utt_feats(utt_feats)

                # append features
                normalized_npy_out.append(np.ascontiguousarray(normalized_feats))
                unnormalized_npy_out.append(np.ascontiguousarray(unnormalized_feats))

def compute_stats_bu_radio(args):
    # load utt2spkr.json
    utt2spkr_path = os.path.join(args.metadata_dir, "utt2spkr.json")
    with open(utt2spkr_path, "r") as f:
        utt2spkr = json.load(f)
    # load {split}.tsv
    split_tsv_path = os.path.join(args.metadata_dir, f"train.tsv")
    with open(split_tsv_path, "r") as f:
        utt_ids = [line.strip().split("\t")[0] for line in f]
    spkr_by_idx = [utt2spkr[utt_id] for utt_id in utt_ids]

    feat_path = os.path.join(args.feats_dir, f"train_0_1.npy")
    feats = np.load(feat_path, mmap_mode="r")
    len_path = os.path.join(args.feats_dir, f"train_0_1.len")
    with open(len_path, "r") as f:
        lens = [int(line.strip()) for line in f]
    idx2spk = np.concatenate([np.repeat(spk, l) for spk, l in zip(spkr_by_idx, lens)])
    print(f"idx2spk (shape {idx2spk.shape}): {idx2spk}")
    spk_f0_avg = {}
    for spk in tqdm.tqdm(
        set(spkr_by_idx), desc=f"Computing stats for BU Radio features", ncols=120, total=len(set(spkr_by_idx))
    ):
        idxs = np.where(idx2spk == spk)[0]
        spk_feats = feats[idxs]

        # compute stats
        f0_avg, _ = compute_single_feat_stats_by_periodicity(
            spk_feats[:, 0], spk_feats[:, 3]
        )
        spk_f0_avg[spk] = f0_avg

    return spk_f0_avg

def normalize_bu_radio(args):
    spk_f0_avg = compute_stats_bu_radio(args)
    utt2spkr_path = os.path.join(args.metadata_dir, "utt2spkr.json")
    with open(utt2spkr_path, "r") as f:
        utt2spkr = json.load(f)
    split_tsv_path = os.path.join(args.metadata_dir, f"{args.split}.tsv")
    with open(split_tsv_path, "r") as f:
        utt_ids = [line.strip().split("\t")[0] for line in f]
    spkr_by_idx = [utt2spkr[utt_id] for utt_id in utt_ids]

    normalized_feats_out_dir = os.path.join(args.feats_dir, "utt_normalized")
    unnormalized_feats_out_dir = os.path.join(args.feats_dir, "utt_unnormalized")
    os.makedirs(normalized_feats_out_dir, exist_ok=True)
    os.makedirs(unnormalized_feats_out_dir, exist_ok=True)

    # output files
    normalized_npy_out_file = os.path.join(
        normalized_feats_out_dir, f"{args.split}_0_1.npy"
    )
    normalized_len_out_file = os.path.join(
        normalized_feats_out_dir, f"{args.split}_0_1.len"
    )
    unnormalized_npy_out_file = os.path.join(
        unnormalized_feats_out_dir, f"{args.split}_0_1.npy"
    )
    unnormalized_len_out_file = os.path.join(
        unnormalized_feats_out_dir, f"{args.split}_0_1.len"
    )
    # remove these if they exist
    for filename in [
        normalized_npy_out_file,
        normalized_len_out_file,
        unnormalized_npy_out_file,
        unnormalized_len_out_file,
    ]:
        if os.path.exists(filename):
            os.remove(filename)

    all_feats = np.load(os.path.join(args.feats_dir, f"{args.split}_0_1.npy"), mmap_mode="r")
    len_path = os.path.join(args.feats_dir, f"{args.split}_0_1.len")
    with open(len_path, "r") as f:
        all_lens = [int(line.strip()) for line in f]
    # write lens
    with open(normalized_len_out_file, "w") as n_f, open(unnormalized_len_out_file, "w") as un_f:
        for l in all_lens:
            n_f.write(f"{l}\n")
            un_f.write(f"{l}\n")
    all_normalized_feats = np.zeros_like(all_feats)
    all_unnormalized_feats = np.zeros_like(all_feats)
    idx2spk = np.concatenate([np.repeat(spk, l) for spk, l in zip(spkr_by_idx, all_lens)])
    print(f"idx2spk (shape {idx2spk.shape}): {idx2spk}")
    for spk in tqdm.tqdm(
        set(spkr_by_idx), desc=f"Normalizing {args.split} features by speaker", ncols=120, total=len(set(spkr_by_idx))
    ):
        idxs = np.where(idx2spk == spk)[0]
        spk_feats = all_feats[idxs]

        # normalize features
        normalized_spk_feats = spk_feats.copy()
        unnormalized_spk_feats = spk_feats.copy()

        # mean-normalize F0
        normalized_spk_feats[:, 0] = normalized_spk_feats[:, 0] - spk_f0_avg[spk]

        # compute delta_F0 as difference between consecutive F0 frames
        normalized_spk_feats[:, 1] = np.concatenate(
            [np.zeros(1), np.diff(normalized_spk_feats[:, 0])]
        )
        unnormalized_spk_feats[:, 1] = np.concatenate(
            [np.zeros(1), np.diff(unnormalized_spk_feats[:, 0])]
        )

        all_normalized_feats[idxs] = normalized_spk_feats
        all_unnormalized_feats[idxs] = unnormalized_spk_feats

    # save normalized features
    np.save(normalized_npy_out_file, all_normalized_feats)
    np.save(unnormalized_npy_out_file, all_unnormalized_feats)

def normalize(args):
    """
    """
    # load utt2spkr.json
    utt2spkr_path = os.path.join(args.metadata_dir, "utt2spkr.json")
    with open(utt2spkr_path, "r") as f:
        utt2spkr = json.load(f)
    # load {split}.tsv
    split_tsv_path = os.path.join(args.metadata_dir, f"{args.split}.tsv")
    with open(split_tsv_path, "r") as f:
        utt_ids = [line.strip().split("\t") for line in f]
        utt_ids = [utt_id[0] for utt_id in utt_ids if len(utt_id) == 2]  # filter first line if necessary
    print(f"Number of utterances in {args.split}: {len(utt_ids)}")
    if args.split in ["training", "train"]:
        spkr_by_idx = [utt2spkr[utt_id] for utt_id in utt_ids]
    else: # elif args.split == "validation":
        # labels don't exist for validation set, so just use utterance-level normalization (one spkr per utt)
        spkr_by_idx = [i for i, _ in enumerate(utt_ids)]

    normalized_feats_out_dir = os.path.join(args.feats_dir, "utt_normalized")
    unnormalized_feats_out_dir = os.path.join(args.feats_dir, "utt_unnormalized")
    os.makedirs(normalized_feats_out_dir, exist_ok=True)
    os.makedirs(unnormalized_feats_out_dir, exist_ok=True)

    # output files
    normalized_npy_out_file = os.path.join(
        normalized_feats_out_dir, f"{args.split}_0_1.npy"
    )
    normalized_len_out_file = os.path.join(
        normalized_feats_out_dir, f"{args.split}_0_1.len"
    )
    unnormalized_npy_out_file = os.path.join(
        unnormalized_feats_out_dir, f"{args.split}_0_1.npy"
    )
    unnormalized_len_out_file = os.path.join(
        unnormalized_feats_out_dir, f"{args.split}_0_1.len"
    )
    # remove these if they exist
    for filename in [
        normalized_npy_out_file,
        normalized_len_out_file,
        unnormalized_npy_out_file,
        unnormalized_len_out_file,
    ]:
        if os.path.exists(filename):
            os.remove(filename)

    all_feats = []
    all_lens = []
    for rank in tqdm.tqdm(range(args.nshard), desc=f"Loading {args.split} features", ncols=80):
        feat_path = os.path.join(args.feats_dir, f"{args.split}_{rank}_{args.nshard}.npy")
        feats = np.load(feat_path, mmap_mode="r")
        len_path = os.path.join(args.feats_dir, f"{args.split}_{rank}_{args.nshard}.len")
        with open(len_path, "r") as f:
            lens = [int(line.strip()) for line in f]
        all_feats.append(feats)
        all_lens += lens
    # write lens
    with open(normalized_len_out_file, "w") as n_f, open(unnormalized_len_out_file, "w") as un_f:
        for l in all_lens:
            n_f.write(f"{l}\n")
            un_f.write(f"{l}\n")
    all_feats = np.concatenate(all_feats, axis=0)
    all_normalized_feats = np.zeros_like(all_feats)
    all_unnormalized_feats = np.zeros_like(all_feats)
    idx2spk = np.concatenate([np.repeat(spk, l) for spk, l in zip(spkr_by_idx, all_lens)])
    print(f"idx2spk (shape {idx2spk.shape}): {idx2spk}")
    for spk in tqdm.tqdm(
        set(spkr_by_idx), desc=f"Normalizing {args.split} features by speaker", ncols=120, total=len(set(spkr_by_idx))
    ):
        idxs = np.where(idx2spk == spk)[0]
        spk_feats = all_feats[idxs]

        # normalize features
        normalized_feats, unnormalized_feats = normalize_utt_feats(spk_feats, log_domain=args.log_pitch)

        # append features
        all_normalized_feats[idxs] = normalized_feats
        all_unnormalized_feats[idxs] = unnormalized_feats
    # save normalized features
    np.save(normalized_npy_out_file, all_normalized_feats)
    np.save(unnormalized_npy_out_file, all_unnormalized_feats)

def save_max(args):
    """
    save max absolute values of F0 and energy in the corpus
    """
    for n_un in ["normalized", "unnormalized"]:
        feats_dir = os.path.join(args.feats_dir, f"utt_{n_un}")
        max_f0 = 0
        max_energy = 0
        for rank in tqdm.tqdm(range(args.nshard), desc=f"Finding max F0 and energy in {args.split}", ncols=80):
            feat_path = os.path.join(feats_dir, f"{args.split}_{rank}_{args.nshard}.npy")
            feats = np.load(feat_path, mmap_mode="r")
            max_f0 = max(max_f0, np.max(np.abs(feats[:, 0])))
            max_energy = max(max_energy, np.max(np.abs(feats[:, 2])))
        out_file = os.path.join(feats_dir, f"{args.split}_max_F0_energy.npy")
        print(f"Saving max F0 {max_f0} and max energy {max_energy} to {out_file}")
        np.save(out_file, np.array([max_f0, max_energy]))

def save_percentiles(args):
    """
    save 0, 25, 50, 75, 95, 100 percentiles of abs(F0) and abs(energy) in the corpus
    """
    pcts = [0, 25, 50, 75, 95, 100]
    pcts_array = np.array(pcts, dtype=np.float32).reshape(-1, 1)
    for n_un in ["normalized", "unnormalized"]:
        feats_dir = os.path.join(args.feats_dir, f"utt_{n_un}")
        all_feats = []
        for rank in tqdm.tqdm(range(args.nshard), desc=f"Finding percentiles of F0 and energy", ncols=180):
            feat_path = os.path.join(feats_dir, f"{args.split}_{rank}_{args.nshard}.npy")
            feats = np.load(feat_path, mmap_mode="r")
            all_feats.append(feats)
        all_feats = np.concatenate(all_feats, axis=0)
        percentiles = np.percentile(np.abs(all_feats[:, [0, 2]]), pcts, axis=0)
        out_file = os.path.join(feats_dir, f"{args.split}_percentiles_F0_energy.npy")
        print(f"Saving percentiles (shape {percentiles.shape}) of F0 and energy to {out_file}")
        percentiles = np.concatenate(
            [pcts_array, percentiles], axis=1,
        )
        np.save(out_file, percentiles)

def save_mean_std(args, normalized_unnormalized=True):

    if normalized_unnormalized:
        norm_unnorm_strs = ["normalized", "unnormalized"]
    else:
        norm_unnorm_strs = [None]
    for n_un in norm_unnorm_strs:
        if n_un is not None:
            feats_dir = os.path.join(args.feats_dir, f"utt_{n_un}")
        else:
            feats_dir = args.feats_dir
        all_feats = []
        for rank in tqdm.tqdm(range(args.nshard), desc=f"Computing stats of F0 and energy", ncols=120):
            feat_path = os.path.join(feats_dir, f"{args.split}_{rank}_{args.nshard}.npy")
            feats = np.load(feat_path, mmap_mode="r")
            all_feats.append(feats)
        all_feats = np.concatenate(all_feats, axis=0)
        # compute weighted avg of columns 0, 1, 2, using column 3 as weights
        f0_avg, f0_std = compute_single_feat_stats_by_periodicity(all_feats[:, 0], all_feats[:, 3])
        deltaf0_avg, deltaf0_std = compute_single_feat_stats_by_periodicity(all_feats[:, 1], all_feats[:, 3])
        energy_avg, energy_std = compute_single_feat_stats_by_periodicity(all_feats[:, 2], all_feats[:, 3])
        out_file = os.path.join(feats_dir, f"{args.split}_mean_std_F0_deltaF0_energy.npy")
        print(f"Saving mean and std of F0, delta_F0, and energy to {out_file}")
        stats = np.array([[f0_avg, f0_std], [deltaf0_avg, deltaf0_std], [energy_avg, energy_std]], dtype=np.float32)
        np.save(out_file, stats)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--feats_dir", type=str, required=True, help="Directory of features")
    parser.add_argument("--metadata_dir", type=str, default=None, help="Directory of metadata files (.tsvs and utt2spkr.json)")
    parser.add_argument("--normalize_by_speaker", action="store_true", help="Whether to normalize features by speaker (requires metadata directory)")
    parser.add_argument("--split", type=str, default="training", help="Data split to process")
    parser.add_argument("--nshard", type=int, default=40, help="Number of shards to process")
    parser.add_argument("--mode", type=str, default="normalize", choices=["normalize", "save_max", "save_percentiles", "print_raw_percentiles", "save_mean_std", "save_mean_std_raw"],)
    parser.add_argument("--log_pitch", action="store_true", help="Whether to log-transform F0 before normalization")
    args = parser.parse_args()

    if args.mode == "normalize" and not args.normalize_by_speaker:
        normalize_by_rank(args)
    elif args.mode == "normalize" and args.normalize_by_speaker:
        assert args.metadata_dir is not None, "Metadata directory must be specified when normalizing by speaker"
        normalize(args)
    elif args.mode == "save_max":
        save_max(args)
    elif args.mode == "save_percentiles":
        save_percentiles(args)
    elif args.mode == "save_mean_std":
        save_mean_std(args)
    elif args.mode == "save_mean_std_raw":
        save_mean_std(args, normalized_unnormalized=False)