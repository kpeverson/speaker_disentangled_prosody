# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys

import numpy as np
from sklearn.cluster import MiniBatchKMeans

import joblib

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("learn_kmeans")


def get_km_model(
    n_clusters,
    init,
    max_iter,
    batch_size,
    tol,
    max_no_improvement,
    n_init,
    reassignment_ratio,
):
    return MiniBatchKMeans(
        n_clusters=n_clusters,
        init=init,
        max_iter=max_iter,
        batch_size=batch_size,
        verbose=1,
        compute_labels=False,
        tol=tol,
        max_no_improvement=max_no_improvement,
        init_size=None,
        n_init=n_init,
        reassignment_ratio=reassignment_ratio,
    )

def delete_inf_time(feat):
    # remove time idx if any element is -inf
    if np.any(np.isinf(feat)):
        feat = np.delete(
            feat, 
            np.unique(np.where(np.isinf(feat))[0]), 
            axis=0
        )
    return feat

def delete_inf_feat(feat, feat_idx=4):
    # remove feat. index 4 if any element is -inf
    if np.any(np.isinf(feat[ ... , feat_idx])):
        feat = np.delete(feat, feat_idx, axis=-1)
    return feat

def delete_inf(feat):
    return delete_inf_time(delete_inf_feat(feat))

def load_feature_shard(feat_dir, split, nshard, rank, percent, delete_inf=False):
    if nshard == 1:
        feat_path = f"{feat_dir}/{split}.npy"
        leng_path = f"{feat_dir}/{split}.len"
    else:
        feat_path = f"{feat_dir}/{split}_{rank}_{nshard}.npy"
        leng_path = f"{feat_dir}/{split}_{rank}_{nshard}.len"
    with open(leng_path, "r") as f:
        lengs = [int(line.rstrip()) for line in f]
        offsets = [0] + np.cumsum(lengs[:-1]).tolist()

    if percent < 0:
        feats = np.load(feat_path, mmap_mode="r")
        if delete_inf:
            return delete_inf(feats)
        else:
            return feats
    else:
        nsample = int(np.ceil(len(lengs) * percent))
        indices = np.random.choice(len(lengs), nsample, replace=False)
        feat = np.load(feat_path, mmap_mode="r")
        if delete_inf:
            feat = delete_inf(feat)
        sampled_feat = np.concatenate(
            [feat[offsets[i]: offsets[i] + lengs[i]] for i in indices], axis=0
        )
        logger.info(
            (
                f"sampled {nsample} utterances, {len(sampled_feat)} frames "
                f"from shard {rank}/{nshard}"
            )
        )
        return sampled_feat


def load_feature(feat_dir, split, nshard, seed, percent, delete_inf=False, max_normalization_path=None, z_normalize=False, z_normalize_idxs=None, recompute_deltaf0=False):
    assert percent <= 1.0
    feat = np.concatenate(
        [
            load_feature_shard(feat_dir, split, nshard, r, percent, delete_inf=delete_inf)
            for r in range(nshard)
        ],
        axis=0,
    )
    logging.info(f"loaded feature with dimension {feat.shape}")
    if max_normalization_path is not None:
        max_normalization = np.load(max_normalization_path)
        logging.info(f"normalizing columns 0 and 1 by {max_normalization[0]} and column 2 by {max_normalization[1]}")
        feat[:, 0] = feat[:, 0] / max_normalization[0]
        feat[:, 1] = feat[:, 1] / max_normalization[0]
        feat[:, 2] = feat[:, 2] / max_normalization[1]
    elif z_normalize:
        if z_normalize_idxs is None:
            # normalize all features
            logging.info(f"z-normalizing features")
            means = np.mean(feat, axis=0, dtype=np.float64)
            logging.info(f"mean: {means}")
            stds = np.std(feat, axis=0, dtype=np.float64)
            logging.info(f"stds: {stds}")
            feat = (feat - means) / stds
        else:
            # only z-normalize specified features
            logging.info(f"z-normalizing features {z_normalize_idxs}")
            means = np.mean(feat[:, z_normalize_idxs], axis=0, dtype=np.float64)
            logging.info(f"mean (idxs {z_normalize_idxs}): {means}")
            stds = np.std(feat[:, z_normalize_idxs], axis=0, dtype=np.float64)
            logging.info(f"stds (idxs {z_normalize_idxs}): {stds}")
            feat[:, z_normalize_idxs] = (feat[:, z_normalize_idxs] - means) / stds
        # feat = (feat - np.mean(feat, axis=0)) / np.std(feat, axis=0)
    if recompute_deltaf0:
        logging.info(f"recomputing delta f0. feats shape before: {feat.shape}")
        feat[:, 1] = np.concatenate([[feat[0, 0]], np.diff(feat[:, 0])])
        logging.info(f"feats shape after: {feat.shape}")

    return feat


def learn_kmeans(
    feat_dir,
    split,
    nshard,
    km_path,
    n_clusters,
    seed,
    percent,
    init,
    max_iter,
    batch_size,
    tol,
    n_init,
    reassignment_ratio,
    max_no_improvement,
    delete_inf=False,
    max_normalization_path=None,
    z_normalize=False,
    z_normalize_idxs=None,
    recompute_deltaf0=False,
):
    np.random.seed(seed)
    feat = load_feature(feat_dir, split, nshard, seed, percent, delete_inf=delete_inf, max_normalization_path=max_normalization_path, z_normalize=z_normalize, z_normalize_idxs=z_normalize_idxs, recompute_deltaf0=recompute_deltaf0)
    km_model = get_km_model(
        n_clusters,
        init,
        max_iter,
        batch_size,
        tol,
        max_no_improvement,
        n_init,
        reassignment_ratio,
    )
    km_model.fit(feat)
    # make dir of km_path if not exist
    os.makedirs(os.path.dirname(km_path), exist_ok=True)
    joblib.dump(km_model, km_path)

    inertia = -km_model.score(feat) / len(feat)
    logger.info("total intertia: %.5f", inertia)
    logger.info("finished successfully")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("feat_dir", type=str)
    parser.add_argument("split", type=str)
    parser.add_argument("nshard", type=int)
    parser.add_argument("km_path", type=str)
    parser.add_argument("n_clusters", type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument(
        "--percent", default=-1, type=float, help="sample a subset; -1 for all"
    )
    parser.add_argument("--init", default="k-means++")
    parser.add_argument("--max_iter", default=100, type=int)
    parser.add_argument("--batch_size", default=10000, type=int)
    parser.add_argument("--tol", default=0.0, type=float)
    parser.add_argument("--max_no_improvement", default=100, type=int)
    parser.add_argument("--n_init", default=20, type=int)
    parser.add_argument("--reassignment_ratio", default=0.0, type=float)
    parser.add_argument("--delete_inf", action="store_true")
    parser.add_argument("--max_normalization_path", type=str, default=None,)
    parser.add_argument("--z_normalize", action="store_true")
    parser.add_argument("--z_normalize_idxs", nargs="+", type=int, default=None, help="only z-normalize specified features (0-indexed), e.g. --z_normalize 0 2 to only z-normalize log(F0) and energy/c1")
    parser.add_argument("--recompute_deltaf0", action="store_true", help="recompute delta f0 as diff of log f0, instead of using pre-computed delta f0 feature")
    args = parser.parse_args()

    if args.z_normalize:
        assert args.max_normalization_path is None

    logging.info(str(args))

    learn_kmeans(**vars(args))
