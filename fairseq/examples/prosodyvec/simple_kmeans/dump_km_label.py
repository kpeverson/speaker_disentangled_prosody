# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import glob
import logging
import os
import sys

import numpy as np

import joblib
import torch
import tqdm

from learn_kmeans import delete_inf_feat

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("dump_km_label")


class ApplyKmeans(object):
    def __init__(self, km_path):
        self.km_model = joblib.load(km_path)
        self.C_np = self.km_model.cluster_centers_.transpose()
        self.Cnorm_np = (self.C_np ** 2).sum(0, keepdims=True)

        self.C = torch.from_numpy(self.C_np)
        self.Cnorm = torch.from_numpy(self.Cnorm_np)
        if torch.cuda.is_available():
            self.C = self.C.cuda()
            self.Cnorm = self.Cnorm.cuda()

    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            dist = (
                x.pow(2).sum(1, keepdim=True)
                - 2 * torch.matmul(x, self.C)
                + self.Cnorm
            )
            return dist.argmin(dim=1).cpu().numpy()
        else:
            dist = (
                (x ** 2).sum(1, keepdims=True)
                - 2 * np.matmul(x, self.C_np)
                + self.Cnorm_np
            )
            return np.argmin(dist, axis=1)


def get_feat_iterator(feat_dir, split, nshard, rank, z_normalize=False, z_normalize_idxs=None, recompute_deltaf0=False):
    if nshard == 1:
        feat_path = f"{feat_dir}/{split}.npy"
        leng_path = f"{feat_dir}/{split}.len"
    else:
        feat_path = f"{feat_dir}/{split}_{rank}_{nshard}.npy"
        leng_path = f"{feat_dir}/{split}_{rank}_{nshard}.len"
    with open(leng_path, "r") as f:
        lengs = [int(line.rstrip()) for line in f]
        offsets = [0] + np.cumsum(lengs[:-1]).tolist()

    if z_normalize:
        if split != "training":
            train_feat_paths = glob.glob(f"{feat_dir}/training*.npy")
            # train_feat_path = feat_path.replace(split, "training")
            train_feat_path = train_feat_paths[0]
            train_feat = np.load(train_feat_path, mmap_mode="r")
        else:
            train_feat = np.load(feat_path, mmap_mode="r")
        train_feat_mean = np.mean(train_feat, axis=0, dtype=np.float64)
        train_feat_std = np.std(train_feat, axis=0, dtype=np.float64)
        logger.info(f"mean {train_feat_mean}")
        logger.info(f"std {train_feat_std}")

    def iterate():
        feat = np.load(feat_path, mmap_mode="r")
        # feat = delete_inf_feat(feat)
        assert feat.shape[0] == (offsets[-1] + lengs[-1]), f"feat shape {feat.shape[0]} does not match offsets {offsets[-1]} + lengths {lengs[-1]} = {offsets[-1] + lengs[-1]}"
        for offset, leng in zip(offsets, lengs):
            chunk_feat = feat[offset: offset + leng].copy()
            if z_normalize:
                if z_normalize_idxs is not None:
                    # logger.info(f"z-normalizing features {z_normalize_idxs}")
                    chunk_feat[:, z_normalize_idxs] = (chunk_feat[:, z_normalize_idxs] - train_feat_mean[z_normalize_idxs]) / train_feat_std[z_normalize_idxs]
                else:
                    # logger.info(f"z-normalizing all features")
                    chunk_feat = (chunk_feat - train_feat_mean) / train_feat_std
            if recompute_deltaf0:
                # logger.info(f"recomputing delta f0 for chunk with shape {chunk_feat.shape}")
                chunk_feat[:, 1] = np.concatenate([[chunk_feat[0, 0]], np.diff(chunk_feat[:, 0])])
            yield chunk_feat

    return iterate, len(lengs)


def dump_label(feat_dir, split, km_path, nshard, rank, lab_dir, max_normalization_path=None, z_normalize=False, z_normalize_idxs=None, recompute_deltaf0=False):
    apply_kmeans = ApplyKmeans(km_path)
    generator, num = get_feat_iterator(feat_dir, split, nshard, rank, z_normalize=z_normalize, z_normalize_idxs=z_normalize_idxs, recompute_deltaf0=recompute_deltaf0)
    iterator = generator()

    if max_normalization_path is not None:
        max_normalization = np.load(max_normalization_path, mmap_mode="r")
        logger.info(f"normalizing columns 0 and 1 by {max_normalization[0]} and column 2 by {max_normalization[1]}")
        max_normalization = np.array([max_normalization[0], max_normalization[0], max_normalization[1], 1.0])
    if nshard == 1:
        lab_path = f"{lab_dir}/{split}.km"
    else:
        lab_path = f"{lab_dir}/{split}_{rank}_{nshard}.km"
    os.makedirs(lab_dir, exist_ok=True)
    with open(lab_path, "w") as f:
        for feat in tqdm.tqdm(iterator, total=num):
            if max_normalization_path is not None:
                lab = apply_kmeans(feat / max_normalization).tolist()
            else:
                lab = apply_kmeans(feat).tolist()
            f.write(" ".join(map(str, lab)) + "\n")
    logger.info("finished successfully")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("feat_dir")
    parser.add_argument("split")
    parser.add_argument("km_path")
    parser.add_argument("nshard", type=int)
    parser.add_argument("rank", type=int)
    parser.add_argument("lab_dir")
    parser.add_argument("--max_normalization_path", type=str, default=None,)
    parser.add_argument("--z_normalize", action="store_true")
    parser.add_argument("--z_normalize_idxs", nargs="+", type=int, default=None, help="only z-normalize specified features (0-indexed), e.g. --z_normalize 0 2 to only z-normalize log(F0) and energy/c1")
    parser.add_argument("--recompute_deltaf0", action="store_true", help="recompute delta f0 as diff of log f0, instead of using pre-computed delta f0 feature")
    args = parser.parse_args()

    if args.z_normalize:
        assert args.max_normalization_path is None

    logging.info(str(args))

    dump_label(**vars(args))
