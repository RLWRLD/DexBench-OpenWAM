"""DexBenchDataset over a synthetic DexBench task set (``_dexbench_bucket``).

Covers what the recipe promises: the 62-joint contract checks, directional
statistics, normalization then the unify scatter into the configured slots,
full windows only, the L-shape canvas from ego left + both wrists, wrist decode
failures that are not papered over, and the deploy normalizer inverting it all.
"""

from __future__ import annotations

import json
import shutil
from unittest.mock import patch

import numpy as np
import pytest
from omegaconf import OmegaConf
from PIL import Image

from openwam.dataloader.dexbench import ACTION_MODE, NORMALIZATION_STATS_FILENAME, DexBenchDataset
from openwam.dataloader.registry import build_dataset, list_registered_datasets
from openwam.dataloader.transforms.multiview import assemble_multiview_layout
from openwam.dataloader.utils.normalization import apply_normalization
from openwam.dataloader.utils.unify_action import map_to_unify, parse_unify_spec
from tests.dataloader._dexbench_bucket import (
    LENGTHS,
    PROMPTS,
    dexbench_config,
    fake_decode,
    source_frame,
    write_bucket,
)

DECODE = "openwam.dataloader.bases.lerobot_v3_reader._decode_video_frames"


@pytest.fixture
def bucket(tmp_path):
    root = tmp_path / "task_set"
    raw = write_bucket(root)
    return root, raw


def _build(root):
    with patch(DECODE, side_effect=fake_decode):
        return build_dataset(dexbench_config(root), split="train")


def _stats(root, key):
    raw = np.load(root / "meta" / NORMALIZATION_STATS_FILENAME, allow_pickle=True).item()[key]
    return {k: np.asarray(v, dtype=np.float32) for k, v in raw.items()}


def test_registered():
    assert "dexbench" in list_registered_datasets()


def test_unify_map_is_the_documented_slot_layout():
    spec = parse_unify_spec(dexbench_config("unused").unify_action_map)
    hands = [
        (10 if side == 0 else 44) + 4 * finger + stage for stage in range(4) for side in (0, 1) for finger in range(5)
    ]
    assert spec.tolist() == list(range(0, 7)) + list(range(34, 41)) + [68, 69] + hands + list(range(70, 76))


def test_directional_stats_and_deploy_artifact(bucket):
    root, raw = bucket
    ds = _build(root)
    assert ds.normalization_stats_path == str(root / "meta" / NORMALIZATION_STATS_FILENAME)
    action, state = _stats(root, ACTION_MODE), _stats(root, f"{ACTION_MODE}_state")
    np.testing.assert_array_equal(action["min"], raw["action"].min(axis=0))
    np.testing.assert_array_equal(state["max"], raw["state"].max(axis=0))
    assert not np.array_equal(action["min"], state["min"])


def test_sample_is_normalized_then_scattered(bucket):
    root, raw = bucket
    ds = _build(root)
    assert len(ds) == sum(length - 32 for length in LENGTHS)  # full 33-frame windows only
    dst = parse_unify_spec(dexbench_config(root).unify_action_map)
    action_stats, state_stats = _stats(root, ACTION_MODE), _stats(root, f"{ACTION_MODE}_state")
    for idx, (start, prompt) in {0: (0, PROMPTS[0]), len(ds) - 1: (LENGTHS[0] + LENGTHS[1] - 33, PROMPTS[1])}.items():
        with patch(DECODE, side_effect=fake_decode):
            sample = ds[idx]
        expected, _ = map_to_unify(
            apply_normalization(raw["action"][start : start + 32], action_stats, "min-max"), dst, 80
        )
        np.testing.assert_array_equal(sample["action"].numpy(), expected)
        proprio, _ = map_to_unify(apply_normalization(raw["state"][start : start + 1], state_stats, "min-max"), dst, 80)
        np.testing.assert_array_equal(sample["proprio"].numpy(), proprio)
        mask = sample["action_mask"].numpy()
        assert mask.all(axis=0)[dst].all() and mask.sum() == 32 * 62
        assert sample["proprio_mask"].numpy().sum() == 62
        assert sample["prompt"] == prompt
        assert len(sample["video"]) == 9 and sample["video"][0].size == (320, 384)


def test_canvas_is_ego_left_head_and_both_wrists(bucket):
    root, raw = bucket
    ds = _build(root)
    with patch(DECODE, side_effect=fake_decode):
        sample = ds[3]
    cams = list(dexbench_config(root).camera_layout)
    for k, frame in enumerate(sample["video"]):
        index = 3 + 4 * k
        sizes = {cams[0]: (320, 256), cams[1]: (160, 128), cams[2]: (160, 128)}
        tiles = {c: source_frame(c, index).resize(size, Image.LANCZOS) for c, size in sizes.items()}
        expected = assemble_multiview_layout(tiles, cams, out_h=384, out_w=320)
        assert np.array_equal(np.asarray(frame), np.asarray(expected))


def test_wrist_decode_failure_is_not_a_black_slot(bucket):
    root, _ = bucket
    ds = _build(root)

    def no_wrist(path, *args):
        if "wrist" in str(path):
            raise FileNotFoundError(path)
        return fake_decode(path, *args)

    with patch(DECODE, side_effect=no_wrist), pytest.raises(FileNotFoundError):
        ds[0]


@pytest.mark.parametrize(
    "edit, message",
    [
        (lambda info: info["features"]["action"]["names"].reverse(), "differ from the configured joint_names"),
        (
            lambda info: info["features"]["observation.images.camera_wrist_left"].update(shape=[3, 360, 640]),
            "camera_frame_hw",
        ),
    ],
)
def test_refuses_a_task_set_outside_the_contract(bucket, edit, message):
    root, _ = bucket
    info = json.loads((root / "meta" / "info.json").read_text())
    edit(info)
    # Keep state and action names paired so only the contract check under test can fire.
    info["features"]["observation.state"]["names"] = [f"{n}_Pos" for n in info["features"]["action"]["names"]]
    (root / "meta" / "info.json").write_text(json.dumps(info))
    with pytest.raises(ValueError, match=message):
        _build(root)


def test_refuses_another_control_rate(bucket):
    root, _ = bucket
    cfg = dexbench_config(root)
    cfg.fps = 15
    with patch(DECODE, side_effect=fake_decode), pytest.raises(ValueError, match="fps expects 15"):
        build_dataset(cfg, split="train")


def test_refuses_another_action_mode(bucket):
    root, _ = bucket
    cfg = dexbench_config(root)
    cfg.action_mode = "joint"
    with pytest.raises(ValueError, match="action_mode"):
        DexBenchDataset.from_config(cfg)


def test_deploy_normalizer_inverts_training(bucket, tmp_path):
    from openwam.deploy.model_loader import _build_normalizer

    root, raw = bucket
    ds = _build(root)
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    shutil.copy(ds.normalization_stats_path, ckpt / "normalization_stats.npy")
    normalizer = _build_normalizer(OmegaConf.create({"dataloader": dexbench_config(root)}), str(ckpt))
    with patch(DECODE, side_effect=fake_decode):
        sample = ds[5]
    np.testing.assert_array_equal(normalizer.normalize(raw["state"][5:6]), sample["proprio"].numpy())
    restored = normalizer.unnormalize(sample["action"].numpy())
    assert restored.shape == (32, 62)
    np.testing.assert_allclose(restored, raw["action"][5:37], atol=1e-5)
