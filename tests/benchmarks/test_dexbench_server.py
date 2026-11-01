"""DexBench HTTP server: contract, canvas parity with the training reader, and the wire protocol.

The model is a stub engine; everything between the HTTP request and
``engine.generate`` (and back) is the real server code.
"""

from __future__ import annotations

import base64
import io
import json
import threading
import urllib.error
import urllib.request
from unittest.mock import patch

import numpy as np
import pytest
from omegaconf import OmegaConf
from PIL import Image

from benchmarks.dexbench.server import DexBenchContract, DexBenchPolicy, RequestError, compose_canvas, make_http_server
from openwam.dataloader.registry import build_dataset
from tests.dataloader._dexbench_bucket import (
    config_joint_names,
    dexbench_config,
    fake_decode,
    source_frame,
    write_bucket,
)

DECODE = "openwam.dataloader.bases.lerobot_v3_reader._decode_video_frames"
CAMERA_SET = ["camera_ego_left", "camera_wrist_left", "camera_wrist_right"]
TOKEN = "test-token"


def _contract():
    return DexBenchContract.from_cfg(OmegaConf.create({"dataloader": dexbench_config("/unused")}))


def _png(image: Image.Image, fmt: str = "PNG") -> str:
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return base64.b64encode(buffer.getvalue()).decode()


class StubEngine:
    def __init__(self):
        self.calls = []

    def generate(self, conditions):
        self.calls.append(conditions)
        return {"actions": np.tile(np.linspace(-1, 1, 62, dtype=np.float32), (32, 1))}


def test_contract_comes_from_the_training_config():
    contract = _contract()
    assert contract.camera_set == CAMERA_SET
    assert contract.frame_hw == (180, 320)
    assert (contract.height, contract.width, contract.action_horizon, contract.fps) == (384, 320, 32, 30.0)
    assert list(contract.joint_names) == config_joint_names()


def test_contract_refuses_another_dataloader():
    cfg = dexbench_config("/unused")
    cfg.type = "robotwin"
    with pytest.raises(ValueError, match="dataloader=dexbench"):
        DexBenchContract.from_cfg(OmegaConf.create({"dataloader": cfg}))


def test_server_canvas_equals_the_training_canvas(tmp_path):
    root = tmp_path / "task_set"
    write_bucket(root)
    with patch(DECODE, side_effect=fake_decode):
        dataset = build_dataset(dexbench_config(root), split="train")
        sample = dataset[7]
    contract = _contract()
    for k, frame in enumerate(sample["video"]):
        index = 7 + 4 * k
        frames = {cam: source_frame(cam, index) for cam in contract.camera_layout}
        assert np.array_equal(np.asarray(compose_canvas(frames, contract)), np.asarray(frame))


@pytest.fixture
def served():
    engine = StubEngine()
    policy = DexBenchPolicy(engine, _contract(), checkpoint="stub", num_steps=10, seed=42)
    server = make_http_server(policy, "127.0.0.1", 0, TOKEN)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield engine, server.server_address[1]
    server.shutdown()


def _call(port, method, path, payload=None, token=TOKEN):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    data = None if payload is None else (payload if isinstance(payload, bytes) else json.dumps(payload).encode())
    request = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def _request(index=0, **overrides):
    contract = _contract()
    images = {key: _png(source_frame(cam, index)) for key, cam in zip(CAMERA_SET, contract.camera_layout)}
    return {"images": images, "state": np.linspace(-0.5, 0.5, 62).tolist(), "prompt": "take the can", **overrides}


def test_info(served):
    _, port = served
    status, info = _call(port, "GET", "/info")
    assert status == 200
    assert info["model_family"] == "openwam"
    assert info["embodiment"] == "dexbench_rby1_wuji2"
    assert info["camera_set"] == CAMERA_SET and info["frame_hw"] == [180, 320]
    assert info["action_horizon"] == 32 and info["fps"] == 30.0
    assert info["num_steps"] == 10 and info["seed"] == 42
    assert info["action_representation"] == {"type": "absolute"}
    assert info["joint_names"] == config_joint_names()


def test_predict_returns_a_chunk_and_feeds_the_training_canvas(served):
    engine, port = served
    status, body = _call(port, "POST", "/predict", _request(index=11))
    assert status == 200
    assert np.asarray(body["action"]).shape == (32, 62)
    conditions = engine.calls[-1]
    contract = _contract()
    expected = compose_canvas({cam: source_frame(cam, 11) for cam in contract.camera_layout}, contract)
    assert np.array_equal(np.asarray(conditions["first_frame_image"][0]), np.asarray(expected))
    assert conditions["proprio"].shape == (1, 62) and conditions["prompt"] == "take the can"
    assert conditions["seed"] == 42 and conditions["denoise_steps"] == 10


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"images": {}}, "missing"),
        ({"state": [0.0] * 61}, "62 finite"),
        ({"state": [float("nan")] * 62}, "62 finite"),
        ({"prompt": " "}, "prompt"),
    ],
)
def test_malformed_requests_are_400(served, overrides, message):
    _, port = served
    status, body = _call(port, "POST", "/predict", _request(**overrides))
    assert status == 400 and message in body["error"]


def test_wrong_frame_size_and_format_are_400(served):
    _, port = served
    request = _request()
    request["images"]["camera_wrist_left"] = _png(Image.new("RGB", (640, 360)))
    status, body = _call(port, "POST", "/predict", request)
    assert status == 400 and "trained on 320x180" in body["error"]
    request = _request()
    request["images"]["camera_ego_left"] = _png(source_frame("observation.images.camera_ego_left", 0), "JPEG")
    status, body = _call(port, "POST", "/predict", request)
    assert status == 400 and "PNG" in body["error"]


def test_auth_path_and_body_errors(served):
    _, port = served
    assert _call(port, "GET", "/info", token=None)[0] == 401
    assert _call(port, "GET", "/nope")[0] == 404
    assert _call(port, "POST", "/predict", b"not json")[0] == 400


def test_policy_parse_is_usable_without_http():
    policy = DexBenchPolicy(StubEngine(), _contract(), checkpoint="stub", num_steps=4, seed=0)
    with pytest.raises(RequestError):
        policy.predict({"images": "x"})
    assert np.asarray(policy.predict(_request())["action"]).shape == (32, 62)
