"""Compute directional joint statistics for one DexBench task set.

Scans every data parquet shard of a DexBench LeRobot v3 task set and writes
``{dataset_dir}/meta/dexbench_normalization_stats.npy`` (auto-built by
``DexBenchDataset`` on first use — rank 0 scans, other ranks wait)::

    {
      "dexbench_joint62":       {mean, std, min, max, q01, q99},  # from action
      "dexbench_joint62_state": {mean, std, min, max, q01, q99},  # from observation.state
      "n_action_samples": int,
      "n_state_samples": int,
    }

Actions and states are kept apart rather than pooled: commanded targets and
measured positions differ on the hand joints. Deploy unnormalizes model output
with the action block and normalizes proprio with the ``_state`` block
(``load_mode_stats``). Exact quantiles over every row; a task set is small.

Usage:
    python -m openwam.dataloader.utils.stats_computation.dexbench_stats_computation \
        --dataset-dir /path/to/dexbench_task_set
    ... --dry-run          # print the table, write nothing
"""

from __future__ import annotations

import argparse
import logging
import os
import tempfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger("dexbench_stats_computation")

ACTION_COL = "action"
STATE_COL = "observation.state"
JOINT_DIM = 62
ACTION_MODE = "dexbench_joint62"
NORMALIZATION_STATS_FILENAME = "dexbench_normalization_stats.npy"


def _load_column(path: Path, column: str) -> np.ndarray:
    table = pq.read_table(path, memory_map=True, columns=[column])
    arr = np.asarray(table.column(column).to_pylist(), dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != JOINT_DIM:
        raise ValueError(f"{path.name}: column {column!r} must be (N, {JOINT_DIM}), got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{path.name}: column {column!r} contains non-finite values")
    return arr


def _summary(arr: np.ndarray) -> dict:
    return {
        "mean": arr.mean(axis=0).astype(np.float64).tolist(),
        "std": arr.std(axis=0).astype(np.float64).tolist(),
        "min": arr.min(axis=0).astype(np.float64).tolist(),
        "max": arr.max(axis=0).astype(np.float64).tolist(),
        "q01": np.quantile(arr, 0.01, axis=0).astype(np.float64).tolist(),
        "q99": np.quantile(arr, 0.99, axis=0).astype(np.float64).tolist(),
    }


def compute_dataset_stats(dataset_dir: Path) -> dict:
    """Walk the data parquets and summarize ``action`` and ``observation.state`` separately."""
    paths = sorted((Path(dataset_dir) / "data").rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No parquet shards under {dataset_dir}/data")
    logger.info("DexBench: scanning %d parquet shards under %s/data", len(paths), dataset_dir)
    actions = np.concatenate([_load_column(p, ACTION_COL) for p in paths], axis=0)
    states = np.concatenate([_load_column(p, STATE_COL) for p in paths], axis=0)
    return {
        ACTION_MODE: _summary(actions),
        f"{ACTION_MODE}_state": _summary(states),
        "n_action_samples": int(len(actions)),
        "n_state_samples": int(len(states)),
    }


def build_and_save_dexbench_stats(dataset_dir, output=None) -> Path:
    """Compute and atomically write the stats payload; returns the output path."""
    dataset_dir = Path(dataset_dir)
    out_path = Path(output) if output else dataset_dir / "meta" / NORMALIZATION_STATS_FILENAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stats = compute_dataset_stats(dataset_dir)
    fd, tmp_name = tempfile.mkstemp(dir=str(out_path.parent), suffix=".npy.tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            np.save(handle, stats, allow_pickle=True)
        os.replace(tmp_name, out_path)
    except BaseException:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise
    logger.info("wrote %s", out_path)
    return out_path


def _print_stats_table(stats: dict) -> None:
    for key in (ACTION_MODE, f"{ACTION_MODE}_state"):
        block = stats[key]
        print(f"\n  {key}")
        print(f"  {'dim':<4} {'min':>9} {'max':>9} {'q01':>9} {'q99':>9} {'mean':>9} {'std':>9}")
        for i in range(JOINT_DIM):
            print(
                f"  {i:<4} {block['min'][i]:>9.4f} {block['max'][i]:>9.4f} {block['q01'][i]:>9.4f} "
                f"{block['q99'][i]:>9.4f} {block['mean'][i]:>9.4f} {block['std'][i]:>9.4f}"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=str, required=True, help="DexBench LeRobot v3 task set root")
    parser.add_argument(
        "--dry-run", action="store_true", help=f"print stats without writing meta/{NORMALIZATION_STATS_FILENAME}"
    )
    args = parser.parse_args()
    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.is_dir():
        parser.error(f"dataset dir does not exist: {dataset_dir}")
    if args.dry_run:
        _print_stats_table(compute_dataset_stats(dataset_dir))
        return
    out_path = build_and_save_dexbench_stats(dataset_dir)
    _print_stats_table(np.load(out_path, allow_pickle=True).item())


if __name__ == "__main__":
    main()
