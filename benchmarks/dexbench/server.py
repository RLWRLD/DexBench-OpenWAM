"""DexBench HTTP policy server for an OpenWAM checkpoint fine-tuned with ``dataloader=dexbench``.

The DexBench evaluation client is DexBench-Arena's joint HTTP client (the one
that also drives the Cosmos3 DexBench server). It asks the server for its
contract once and then requests one action chunk per re-plan:

* ``GET /info`` — the contract: ``model_family``, the checkpoint, the 62 joint
  names in the order ``state`` and ``action`` use, the camera keys in training
  order (``camera_set``), the per-camera frame size the client must send
  (``frame_hw``), the action horizon, fps, sampling settings and the action
  representation (absolute joint targets).
* ``POST /predict`` with ``{"images": {<camera key>: <base64 RGB PNG>}, "state":
  [62 floats], "prompt": "<task text>"}`` returns ``{"action": [[62 floats] x
  horizon]}`` in radians. A malformed request gets ``{"error": ...}`` with 400,
  an inference failure 500.

Everything the server states comes from the checkpoint's saved ``config.yaml``
(the ``dataloader`` block of the training run) and its ``normalization_stats.npy``;
nothing is configured twice. The client area-downscales each 640x360 render to
``frame_hw`` (the stored frame size of the training videos). The server then
builds the canvas exactly as the training reader did: each frame LANCZOS-resized
to its slot, then composed by ``assemble_multiview_layout``.

Inference goes through the native deploy stack (``load_from_checkpoint_dir``,
``JointInferenceEngine``): the unify-aware normalizer scatters the raw state
into the 80-D model space, pins the unmapped action slots, and gathers and
unnormalizes the model output back to the 62 raw joints. The server keeps no
episode state; every request generates a full chunk with the same seed.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import io
import json
import logging
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image

from openwam.dataloader.bases.lerobot_v3_reader import _HEAD_SLOT_H, _HEAD_SLOT_W, _WRIST_SLOT_H, _WRIST_SLOT_W
from openwam.dataloader.transforms.multiview import assemble_multiview_layout

logger = logging.getLogger("dexbench_server")

MODEL_FAMILY = "openwam"
EMBODIMENT = "dexbench_rby1_wuji2"
ACTION_MODE = "dexbench_joint62"
CAMERA_PREFIX = "observation.images."
REPO_ROOT = Path(__file__).resolve().parents[2]


class RequestError(ValueError):
    """A malformed ``/predict`` request (answered with 400)."""


@dataclass(frozen=True)
class DexBenchContract:
    """The checkpoint's DexBench contract, read from its saved ``dataloader`` config."""

    joint_names: tuple
    camera_layout: tuple
    frame_hw: tuple
    height: int
    width: int
    action_horizon: int
    fps: float

    @property
    def camera_set(self) -> list:
        return [key[len(CAMERA_PREFIX) :] if key.startswith(CAMERA_PREFIX) else key for key in self.camera_layout]

    @classmethod
    def from_cfg(cls, cfg) -> "DexBenchContract":
        from omegaconf import OmegaConf

        dl = OmegaConf.to_container(OmegaConf.select(cfg, "dataloader"), resolve=True)
        if dl.get("type") != "dexbench" or dl.get("action_mode") != ACTION_MODE:
            raise ValueError(
                f"checkpoint was not trained with dataloader=dexbench (type={dl.get('type')!r}, "
                f"action_mode={dl.get('action_mode')!r})"
            )
        if not dl.get("multiview") or len(dl.get("camera_layout") or []) != 3:
            raise ValueError("checkpoint dataloader must be multiview with a 3-camera camera_layout")
        if not dl.get("unify_action"):
            raise ValueError("checkpoint dataloader must use unify_action")
        names = tuple(dl["joint_names"])
        if len(names) != 62 or len(set(names)) != 62:
            raise ValueError("checkpoint dataloader.joint_names must list 62 distinct joints")
        return cls(
            joint_names=names,
            camera_layout=tuple(dl["camera_layout"]),
            frame_hw=tuple(int(v) for v in dl["camera_frame_hw"]),
            height=int(dl["height"]),
            width=int(dl["width"]),
            action_horizon=int(dl["num_frames"]) - 1,
            fps=float(dl["fps"]),
        )


def compose_canvas(frames: dict, contract: DexBenchContract) -> Image.Image:
    """Training canvas from one frame per contract camera (keys as in ``camera_layout``).

    Mirrors ``LeRobotV3Reader``: each decoded frame is LANCZOS-resized to its
    slot (head 256x320, wrists 128x160), then ``assemble_multiview_layout``
    pastes the slot-sized tiles.
    """
    head, left, right = contract.camera_layout
    slots = {
        head: (_HEAD_SLOT_W, _HEAD_SLOT_H),
        left: (_WRIST_SLOT_W, _WRIST_SLOT_H),
        right: (_WRIST_SLOT_W, _WRIST_SLOT_H),
    }
    tiles = {key: frames[key].resize(size, Image.LANCZOS) for key, size in slots.items()}
    return assemble_multiview_layout(tiles, list(contract.camera_layout), out_h=contract.height, out_w=contract.width)


def decode_png(data: Any, expected_hw: tuple, name: str) -> Image.Image:
    """An RGB image from a base64 PNG of exactly ``expected_hw`` (height, width)."""
    if not isinstance(data, str):
        raise RequestError(f"images.{name} must be a base64 PNG string")
    try:
        raw = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise RequestError(f"images.{name}: invalid base64 ({exc})") from exc
    try:
        with Image.open(io.BytesIO(raw)) as img:
            if img.format != "PNG" or img.mode != "RGB":
                raise RequestError(f"images.{name}: expected an RGB PNG, got {img.format} {img.mode}")
            image = img.copy()
    except RequestError:
        raise
    except Exception as exc:
        raise RequestError(f"images.{name}: cannot decode PNG ({exc})") from exc
    if (image.height, image.width) != tuple(expected_hw):
        raise RequestError(
            f"images.{name}: frame is {image.width}x{image.height}, the checkpoint was trained on "
            f"{expected_hw[1]}x{expected_hw[0]} frames"
        )
    return image


class DexBenchPolicy:
    """Request validation, canvas composition and one engine call per request."""

    def __init__(self, engine, contract: DexBenchContract, *, checkpoint: str, num_steps: int, seed: int):
        self.engine = engine
        self.contract = contract
        self.checkpoint = checkpoint
        self.num_steps = int(num_steps)
        self.seed = int(seed)
        self._lock = threading.Lock()
        self._requests = 0

    def info(self) -> dict:
        c = self.contract
        return {
            "model_family": MODEL_FAMILY,
            "checkpoint": self.checkpoint,
            "embodiment": EMBODIMENT,
            "joint_names": list(c.joint_names),
            "camera_set": c.camera_set,
            "frame_hw": list(c.frame_hw),
            "canvas_hw": [c.height, c.width],
            "action_horizon": c.action_horizon,
            "fps": c.fps,
            "num_steps": self.num_steps,
            "seed": self.seed,
            "action_representation": {"type": "absolute"},
        }

    def _parse(self, request: Any):
        if not isinstance(request, dict):
            raise RequestError("request body must be a JSON object")
        images = request.get("images")
        if not isinstance(images, dict):
            raise RequestError("images must be an object keyed by camera")
        missing = [key for key in self.contract.camera_set if key not in images]
        if missing:
            raise RequestError(f"images is missing {missing}; this checkpoint takes {self.contract.camera_set}")
        frames = {
            layout_key: decode_png(images[key], self.contract.frame_hw, key)
            for layout_key, key in zip(self.contract.camera_layout, self.contract.camera_set)
        }
        try:
            state = np.asarray(request.get("state"), dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise RequestError(f"state must be a list of numbers ({exc})") from exc
        if state.shape != (len(self.contract.joint_names),) or not np.isfinite(state).all():
            raise RequestError(
                f"state must be {len(self.contract.joint_names)} finite joint positions, got {state.shape}"
            )
        prompt = request.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise RequestError("prompt must be a non-empty string")
        return frames, state, prompt

    def predict(self, request: Any) -> dict:
        frames, state, prompt = self._parse(request)
        canvas = compose_canvas(frames, self.contract)
        t0 = time.monotonic()
        with self._lock:
            self._requests += 1
            result = self.engine.generate(
                {
                    "prompt": prompt,
                    "first_frame_image": [canvas],
                    "proprio": state[None],
                    "seed": self.seed,
                    "denoise_steps": self.num_steps,
                }
            )
        actions = result["actions"]
        if hasattr(actions, "detach"):
            actions = actions.detach().cpu().numpy()
        actions = np.asarray(actions, dtype=np.float64)
        expected = (self.contract.action_horizon, len(self.contract.joint_names))
        if actions.shape != expected or not np.isfinite(actions).all():
            raise RuntimeError(f"model returned an action chunk of shape {actions.shape}, expected {expected}")
        logger.info("request %d: %.3fs", self._requests, time.monotonic() - t0)
        return {"action": actions.tolist()}


class _Handler(BaseHTTPRequestHandler):
    def _send(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _authorized(self) -> bool:
        token = self.server.token  # type: ignore[attr-defined]
        if token is None or self.headers.get("Authorization", "") == f"Bearer {token}":
            return True
        self._send(401, {"error": "missing or wrong bearer token"})
        return False

    def do_GET(self) -> None:  # noqa: N802
        if not self._authorized():
            return
        if self.path == "/info":
            self._send(200, self.server.policy.info())  # type: ignore[attr-defined]
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if not self._authorized():
            return
        if self.path != "/predict":
            self._send(404, {"error": "not found"})
            return
        try:
            request = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
        except (ValueError, UnicodeDecodeError) as exc:
            self._send(400, {"error": f"request body is not JSON ({exc})"})
            return
        try:
            self._send(200, self.server.policy.predict(request))  # type: ignore[attr-defined]
        except RequestError as exc:
            self._send(400, {"error": str(exc)})
        except Exception as exc:
            logger.exception("inference failed")
            self._send(500, {"error": f"{type(exc).__name__}: {exc}"})

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return


def make_http_server(policy: DexBenchPolicy, host: str, port: int, token: Optional[str]) -> ThreadingHTTPServer:
    httpd = ThreadingHTTPServer((host, int(port)), _Handler)
    httpd.policy = policy  # type: ignore[attr-defined]
    httpd.token = token  # type: ignore[attr-defined]
    return httpd


def build_engine(
    ckpt_dir: str,
    *,
    ckpt_name: Optional[str],
    device: str,
    num_steps: int,
    dit_cache: bool,
    compile_enabled: bool,
    overrides: list,
):
    """Load the checkpoint through the native deploy stack; returns ``(engine, merged_cfg)``."""
    from omegaconf import OmegaConf

    from openwam.deploy import JointInferenceEngine
    from openwam.deploy.model_loader import load_from_checkpoint_dir
    from openwam.deploy.server import _normalize_compile_enabled_in_cfg, _validate_inference_config, merge_deploy_cfg

    deploy_cfg = OmegaConf.load(REPO_ROOT / "configs" / "deploy.yaml")
    OmegaConf.update(deploy_cfg, "inference.denoise_steps", int(num_steps), merge=False)
    OmegaConf.update(deploy_cfg, "inference.denoise_mode", "sync", merge=False)
    OmegaConf.update(deploy_cfg, "optimization.decode_video", False, merge=False)
    OmegaConf.update(deploy_cfg, "optimization.dit_cache.enabled", bool(dit_cache), merge=False)
    OmegaConf.update(deploy_cfg, "optimization.compile.enabled", bool(compile_enabled), merge=False)
    if overrides:
        deploy_cfg = OmegaConf.merge(deploy_cfg, OmegaConf.from_dotlist(list(overrides)))
    _validate_inference_config(deploy_cfg)
    _normalize_compile_enabled_in_cfg(deploy_cfg)
    training_cfg, architecture = load_from_checkpoint_dir(ckpt_dir, device=device, ckpt_name=ckpt_name)
    merged = merge_deploy_cfg(training_cfg, deploy_cfg)
    normalizer = getattr(architecture, "normalizer", None)
    if len(getattr(normalizer, "_dst_index", ())) != 62:
        raise ValueError("checkpoint has no 62-joint unify-aware normalizer; is normalization_stats.npy missing?")
    return JointInferenceEngine(cfg=merged, architecture=architecture), merged


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--ckpt-dir", required=True, help="checkpoint directory (config.yaml + checkpoint_step_*.safetensors)"
    )
    parser.add_argument("--ckpt-name", default=None, help="checkpoint file (default: the latest checkpoint_step_*)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default="127.0.0.1", help="bind address; keep it local unless the port is firewalled")
    parser.add_argument("--port", type=int, default=5816, help="DexBench registers OpenWAM at 127.0.0.1:5816")
    parser.add_argument(
        "--api-token-file", type=Path, default=None, help="file holding a bearer token every request must carry"
    )
    parser.add_argument("--num-steps", type=int, default=10, help="denoising steps per chunk")
    parser.add_argument("--seed", type=int, default=42, help="sampling seed, the same for every request")
    parser.add_argument("--dit-cache", action="store_true", help="enable the deploy DiT velocity cache (approximate)")
    parser.add_argument("--compile", action="store_true", help="enable the architecture's compile fast paths")
    parser.add_argument("overrides", nargs="*", help="extra OmegaConf dotlist overrides for the deploy config")
    args = parser.parse_args(argv)
    logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)

    token = args.api_token_file.read_text().strip() if args.api_token_file else None
    engine, cfg = build_engine(
        args.ckpt_dir,
        ckpt_name=args.ckpt_name,
        device=args.device,
        num_steps=args.num_steps,
        dit_cache=args.dit_cache,
        compile_enabled=args.compile,
        overrides=args.overrides,
    )
    contract = DexBenchContract.from_cfg(cfg)
    checkpoint = str(Path(args.ckpt_dir) / args.ckpt_name) if args.ckpt_name else str(args.ckpt_dir)
    policy = DexBenchPolicy(engine, contract, checkpoint=checkpoint, num_steps=args.num_steps, seed=args.seed)
    httpd = make_http_server(policy, args.host, args.port, token)
    logger.info(
        "DexBench server on http://%s:%d (token %s): %s",
        args.host,
        args.port,
        "required" if token else "off",
        json.dumps(policy.info()),
    )
    httpd.serve_forever()


if __name__ == "__main__":
    main()
