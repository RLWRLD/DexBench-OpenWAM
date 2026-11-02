"""Synthetic DexBench task set (LeRobot v3, 62 joints, four cameras) shared by the DexBench tests.

Video files are empty placeholders: tests patch ``_decode_video_frames`` with
:func:`fake_decode`, which draws a deterministic 320x180 source frame per
(camera, frame index) and resizes it exactly as the real decoder does
(``Image.resize((w, h), LANCZOS)``). :func:`source_frame` returns the same
source frame, so a test can push it through another path (the deploy server)
and compare canvases bitwise.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from omegaconf import OmegaConf
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "dataloader" / "dexbench.yaml"
CAMERAS = (
    "observation.images.camera_ego_left",
    "observation.images.camera_ego_right",
    "observation.images.camera_wrist_left",
    "observation.images.camera_wrist_right",
)
FPS = 30
FRAME_HW = (180, 320)
LENGTHS = (40, 36)
PROMPTS = ("take the can out of the fridge", "put the bag on the floor")


def config_joint_names() -> list:
    return list(OmegaConf.load(CONFIG_PATH).joint_names)


def dexbench_config(dataset_dir: Path):
    cfg = OmegaConf.load(CONFIG_PATH)
    cfg.dataset_dir = str(dataset_dir)
    return cfg


def source_frame(camera: str, index: int) -> Image.Image:
    """Deterministic textured 320x180 RGB frame for one camera and absolute frame index."""
    seed = int.from_bytes(hashlib.sha256(f"{camera}:{index}".encode()).digest()[:4], "little")
    rng = np.random.RandomState(seed)
    coarse = rng.randint(0, 256, size=(9, 16, 3), dtype=np.uint8)
    frame = Image.fromarray(coarse).resize((FRAME_HW[1], FRAME_HW[0]), Image.NEAREST)
    noise = rng.randint(0, 24, size=(FRAME_HW[0], FRAME_HW[1], 3), dtype=np.uint8)
    return Image.fromarray(np.asarray(frame) // 2 + noise)


def _camera_from_path(path: str) -> str:
    return next(c for c in CAMERAS if f"/{c}/" in path)


def fake_decode(video_path: str, frame_indices, height: int, width: int):
    camera = _camera_from_path(str(video_path))
    return [source_frame(camera, i).resize((width, height), Image.LANCZOS) for i in frame_indices]


def write_bucket(bucket: Path, *, joint_names=None, fps: int = FPS, frame_hw=FRAME_HW, seed: int = 0) -> dict:
    """Write the task set; returns the raw ``action`` / ``observation.state`` arrays."""
    names = list(joint_names) if joint_names is not None else config_joint_names()
    (bucket / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (bucket / "data" / "chunk-000").mkdir(parents=True)
    for camera in CAMERAS:
        video_dir = bucket / "videos" / camera / "chunk-000"
        video_dir.mkdir(parents=True)
        (video_dir / "file-000.mp4").write_bytes(b"")
    video = {"dtype": "video", "shape": [3, frame_hw[0], frame_hw[1]], "names": ["channels", "height", "width"]}
    info = {
        "codebase_version": "v3.0",
        "fps": fps,
        "total_episodes": len(LENGTHS),
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {
            "action": {"dtype": "float32", "shape": [62], "names": names},
            "observation.state": {"dtype": "float32", "shape": [62], "names": [f"{n}_Pos" for n in names]},
            "task_index": {"dtype": "int64", "shape": [1]},
            **{camera: dict(video) for camera in CAMERAS},
        },
    }
    (bucket / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")

    lengths = np.asarray(LENGTHS, dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    rows = {
        "episode_index": np.arange(len(LENGTHS), dtype=np.int64),
        "length": lengths,
        "dataset_from_index": starts,
        "dataset_to_index": starts + lengths,
        "data/chunk_index": np.zeros(len(LENGTHS), dtype=np.int64),
        "data/file_index": np.zeros(len(LENGTHS), dtype=np.int64),
    }
    for camera in CAMERAS:
        rows[f"videos/{camera}/chunk_index"] = np.zeros(len(LENGTHS), dtype=np.int64)
        rows[f"videos/{camera}/file_index"] = np.zeros(len(LENGTHS), dtype=np.int64)
        rows[f"videos/{camera}/from_timestamp"] = starts / float(fps)
        rows[f"videos/{camera}/to_timestamp"] = (starts + lengths) / float(fps)
    pq.write_table(
        pa.Table.from_pandas(pd.DataFrame(rows)), bucket / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    )
    pd.DataFrame({"task_index": list(range(len(PROMPTS)))}, index=pd.Index(PROMPTS, name="task")).to_parquet(
        bucket / "meta" / "tasks.parquet"
    )

    rng = np.random.RandomState(seed)
    n = int(lengths.sum())
    action = rng.uniform(-1.5, 1.5, size=(n, 62)).astype(np.float32)
    state = (action + rng.normal(0, 0.05, size=(n, 62))).astype(np.float32)
    action[:, 14] = 0.25  # a constant joint: the degenerate-range path
    state[:, 14] = 0.25
    task = np.concatenate([np.full(int(length), i, dtype=np.int64) for i, length in enumerate(LENGTHS)])
    frame = pd.DataFrame({"action": list(action), "observation.state": list(state), "task_index": task})
    pq.write_table(pa.Table.from_pandas(frame), bucket / "data" / "chunk-000" / "file-000.parquet")
    return {"action": action, "state": state, "starts": starts}
