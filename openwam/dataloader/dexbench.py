"""DexBench RBY1-Wuji2 reader: 62 absolute joint positions, ego and wrist cameras.

DexBench (RBY1-M with two Wuji Hand 2, simulated in Isaac Lab) publishes each
task's demonstrations as LeRobot v3 task sets whose ``action`` and
``observation.state`` columns are the 62 policy joints in radians — the four
wheels are excluded — in the order::

    [ 0: 7) left arm      left_arm_0 .. left_arm_6
    [ 7:14) right arm     right_arm_0 .. right_arm_6
    [14:16) head          head_0, head_1
    [16:56) hands         stage-major: cmc_flex/mcp_flex, cmc_abd/mcp_abd, mcp/pip,
                          ip/dip; within a stage the five left fingers, then the
                          five right fingers (thumb, index, middle, ring, pinky)
    [56:62) torso         torso_0 .. torso_5

``observation.state`` uses the same order with a ``_Pos`` suffix on each name.
The joint names are part of the run's contract: ``configs/dataloader/dexbench.yaml``
lists them and this reader refuses a task set whose ``meta/info.json`` names
differ, so a trained checkpoint's ``config.yaml`` states the order its actions
use (``benchmarks/dexbench/server.py`` reports it to the evaluation client).

Cameras
-------
The ego (head) left camera fills the head slot and the two palm cameras fill
the wrist slots of the L-shape multiview canvas, the layout every OpenWAM-α
pretraining source uses. The ego right camera is not read. All three views are
model inputs, not optional extras: a wrist clip that fails to decode skips the
window (the base reader retries another one) instead of training on a black
slot.

Action space
------------
The 62 raw joints are normalized first, then scattered into the unified 80-D
slots by ``unify_action_map``: the arm joints into the EEF pose slots, the 40
hand joints into the dexterous-hand slots, and head and torso into the reserved
slots (see the yaml for the full map).

Windows
-------
Only full ``num_frames`` windows are sampled, so every window carries a
supervised target on all ``num_frames - 1`` action steps.

Normalization
-------------
``meta/dexbench_normalization_stats.npy`` holds directional statistics: the
``dexbench_joint62`` block (from ``action``) normalizes actions and
unnormalizes model output at deploy, and the ``dexbench_joint62_state`` block
(from ``observation.state``) normalizes proprio. The file is auto-built on
first use (rank 0 scans the task set, other ranks wait) and is itself the
deploy artifact copied into each checkpoint as ``normalization_stats.npy``.
One task set per run: point ``dataset_dir`` at a single set.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from openwam.dataloader.bases import LeRobotV3Reader
from openwam.dataloader.utils.normalization import apply_normalization, load_stats_file

logger = logging.getLogger(__name__)

DEXBENCH_JOINT_DIM = 62
ACTION_MODE = "dexbench_joint62"
STATE_STATS_KEY = f"{ACTION_MODE}_state"
NORMALIZATION_STATS_FILENAME = "dexbench_normalization_stats.npy"
ACTION_COL = "action"
STATE_COL = "observation.state"
STATE_NAME_SUFFIX = "_Pos"


def _image_hw(feature: dict) -> list[int]:
    """``[height, width]`` of a LeRobot video feature (``shape`` is ``[C, H, W]`` or ``[H, W, C]``)."""
    shape = [int(v) for v in feature["shape"]]
    names = feature.get("names") or []
    if len(shape) == 3 and list(names)[:1] == ["channels"]:
        return shape[1:]
    if len(shape) == 3 and list(names)[-1:] == ["channels"]:
        return shape[:2]
    info = feature.get("info") or {}
    return [int(info["video.height"]), int(info["video.width"])]


class DexBenchDataset(LeRobotV3Reader):
    """Single task set reader for the DexBench RBY1-Wuji2 62-joint LeRobot v3 layout."""

    DATASET_NAME = "DexBench"
    HEAD_CAMERA = "observation.images.camera_ego_left"
    LEFT_WRIST_CAMERA = "observation.images.camera_wrist_left"
    RIGHT_WRIST_CAMERA = "observation.images.camera_wrist_right"
    NEEDED_COLS = (STATE_COL, ACTION_COL, "task_index")
    ACTION_DIM = DEXBENCH_JOINT_DIM
    PROMPT_FILE_REQUIRED = True
    DEFAULT_NORMALIZE_MODE = "min-max"
    STATS_DIM = DEXBENCH_JOINT_DIM
    DEPLOY_ACTION_MODE = ACTION_MODE
    # Wrist views are required inputs of this recipe (see module docstring).
    WRIST_DECODE_TOLERATED = ()

    CONFIG_KEYS = LeRobotV3Reader.CONFIG_KEYS + ("joint_names", "camera_frame_hw", "fps")

    def __init__(
        self,
        dataset_dir: str,
        *,
        joint_names: Optional[Sequence[str]] = None,
        camera_frame_hw: Optional[Sequence[int]] = None,
        fps: Optional[float] = None,
        **kwargs: Any,
    ):
        self._expected_joint_names = list(joint_names) if joint_names is not None else None
        self._expected_fps = float(fps) if fps is not None else None
        self._expected_frame_hw = [int(v) for v in camera_frame_hw] if camera_frame_hw is not None else None
        self._state_normalization_stats: Optional[dict] = None
        super().__init__(dataset_dir, **kwargs)

    # ----- contract checks --------------------------------------------------

    def _post_init(self, info: dict) -> None:
        if self._expected_fps is not None and self._fps != self._expected_fps:
            raise ValueError(
                f"{self.DATASET_NAME}({self._dataset_id}): task set is {self._fps} fps, fps expects {self._expected_fps}"
            )
        features = info["features"]
        for column in (ACTION_COL, STATE_COL):
            shape = list(features.get(column, {}).get("shape") or [])
            if shape != [DEXBENCH_JOINT_DIM]:
                raise ValueError(f"{self.DATASET_NAME}({self._dataset_id}): {column!r} must be 62-D, got {shape}")
        action_names = list(features[ACTION_COL].get("names") or [])
        state_names = [
            name[: -len(STATE_NAME_SUFFIX)] if name.endswith(STATE_NAME_SUFFIX) else name
            for name in features[STATE_COL].get("names") or []
        ]
        if state_names != action_names:
            raise ValueError(
                f"{self.DATASET_NAME}({self._dataset_id}): observation.state and action list different joints"
            )
        if self._expected_joint_names is not None and action_names != self._expected_joint_names:
            raise ValueError(
                f"{self.DATASET_NAME}({self._dataset_id}): meta/info.json action names {action_names} differ "
                f"from the configured joint_names {self._expected_joint_names}"
            )
        self.joint_names = action_names
        cameras = [c for c in (self._head_camera, self._left_wrist_camera, self._right_wrist_camera) if c]
        for camera in cameras:
            if camera not in features:
                raise ValueError(f"{self.DATASET_NAME}({self._dataset_id}): no {camera!r} video in meta/info.json")
            hw = _image_hw(features[camera])
            if self._expected_frame_hw is not None and hw != self._expected_frame_hw:
                raise ValueError(
                    f"{self.DATASET_NAME}({self._dataset_id}): {camera} frames are {hw[1]}x{hw[0]}, "
                    f"camera_frame_hw expects {self._expected_frame_hw[1]}x{self._expected_frame_hw[0]}"
                )

    def _train_min_window_len(self) -> int:
        return self._num_frames

    # ----- statistics -------------------------------------------------------

    def _stats_path(self) -> Path:
        return self._dataset_dir / "meta" / NORMALIZATION_STATS_FILENAME

    def _build_stats_rank0(self, path: Path) -> None:
        """Rank 0 scans the task set and writes the stats file; other ranks wait for it."""
        try:
            import torch.distributed as dist

            dist_ready = dist.is_available() and dist.is_initialized()
        except Exception:
            dist_ready = False
        rank = dist.get_rank() if dist_ready else int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
        if rank == 0:
            from openwam.dataloader.utils.stats_computation.dexbench_stats_computation import (
                build_and_save_dexbench_stats,
            )

            build_and_save_dexbench_stats(self._dataset_dir, output=path)
            return
        deadline = time.monotonic() + float(os.environ.get("OPENWAM_STATS_WAIT_TIMEOUT_S", 12 * 60 * 60))
        poll_interval_s = float(os.environ.get("OPENWAM_STATS_POLL_INTERVAL_S", 10))
        while not path.is_file():
            if time.monotonic() >= deadline:
                raise TimeoutError(f"timed out waiting for rank 0 to build DexBench stats: {path}")
            time.sleep(poll_interval_s)

    def _load_stats(self, info: dict) -> Optional[dict]:
        if not self._normalize_mode or self._normalize_mode in ("none", "null"):
            return None
        path = self._stats_path()
        if not path.is_file():
            self._build_stats_rank0(path)
        action_stats = load_stats_file(
            path, action_mode=ACTION_MODE, normalize_mode=self._normalize_mode, dim=DEXBENCH_JOINT_DIM
        )
        self._state_normalization_stats = load_stats_file(
            path, action_mode=STATE_STATS_KEY, normalize_mode=self._normalize_mode, dim=DEXBENCH_JOINT_DIM
        )
        # The source file already carries both deploy blocks (action and _state);
        # checkpoint saving copies it as normalization_stats.npy.
        self.normalization_stats_path = str(path)
        return action_stats

    # ----- payloads ---------------------------------------------------------

    @staticmethod
    def _column(win: pd.DataFrame, column: str, rows: Optional[int] = None) -> np.ndarray:
        values = win[column].values if rows is None else win[column].values[:rows]
        arr = np.stack(values).astype(np.float32)
        if arr.ndim != 2 or arr.shape[1] != DEXBENCH_JOINT_DIM:
            raise ValueError(f"DexBench {column} must be (T, {DEXBENCH_JOINT_DIM}), got {arr.shape}")
        return arr

    def _action_20d(self, win: pd.DataFrame) -> np.ndarray:
        return apply_normalization(self._column(win, ACTION_COL), self._normalization_stats, self._normalize_mode)

    def _proprio_20d(self, win: pd.DataFrame) -> np.ndarray:
        return apply_normalization(
            self._column(win, STATE_COL, rows=1), self._state_normalization_stats, self._normalize_mode
        )

    # ----- config -----------------------------------------------------------

    @classmethod
    def from_config(cls, config, split: str = "train"):
        from openwam.dataloader.utils import get_cfg

        action_mode = get_cfg(config, "action_mode", ACTION_MODE)
        if action_mode != ACTION_MODE:
            raise ValueError(f"{cls.__name__}: action_mode must be {ACTION_MODE!r}, got {action_mode!r}")
        dataset_dir = get_cfg(config, "dataset_dir")
        if dataset_dir is not None and not (Path(dataset_dir) / "meta" / "info.json").is_file():
            raise FileNotFoundError(
                f"{cls.__name__}: {dataset_dir}/meta/info.json missing; dataset_dir must be one DexBench task set"
            )
        return super().from_config(config, split=split)


__all__ = ["DexBenchDataset", "ACTION_MODE", "DEXBENCH_JOINT_DIM", "NORMALIZATION_STATS_FILENAME"]
