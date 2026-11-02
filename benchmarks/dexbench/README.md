# DexBench (RBY1-Wuji2) Fine-Tuning and Serving

This guide fine-tunes the OpenWAM-α foundation checkpoint on a DexBench task
set (an RBY1-M robot with two Wuji Hand 2, simulated in Isaac Lab) and serves
the result to the DexBench evaluation client. The setup uses the ego left
camera and both wrist (palm) cameras, composed into the same L-shape canvas
that every OpenWAM-α pretraining source uses.

The DexBench additions on this branch are upstream OpenWAM at commit
`3dfc9a7d9230f0e1382e6b22639278f1a86e5eb5` plus:

| File | Purpose |
| --- | --- |
| [`openwam/dataloader/dexbench.py`](../../openwam/dataloader/dexbench.py) | LeRobot v3 reader (`dataloader.type: dexbench`) |
| [`configs/dataloader/dexbench.yaml`](../../configs/dataloader/dexbench.yaml) | joint contract, camera layout, 62→80 slot map, normalization |
| [`openwam/dataloader/utils/stats_computation/dexbench_stats_computation.py`](../../openwam/dataloader/utils/stats_computation/dexbench_stats_computation.py) | action and state statistics |
| [`train.sh`](train.sh) | fine-tuning launcher with the recipe below |
| [`server.py`](server.py) | HTTP policy server for the DexBench client |

Follow the [installation guide](../../README.md#installation) and prepare the
Wan2.2-TI2V-5B video backbone ([Assets Preparation](../../README.md#assets-preparation))
before running the commands below from the repository root.

## Dataset

Use one DexBench task set in the LeRobot v3 layout:

- `action` and `observation.state`: 62 absolute joint positions in radians,
  30 fps. The 62 policy joints exclude the four wheels.
- Four 320×180 videos: `observation.images.camera_ego_left`,
  `observation.images.camera_ego_right`, `observation.images.camera_wrist_left`,
  `observation.images.camera_wrist_right`. The ego right camera is not read.
- `meta/tasks.parquet`: one task sentence per `task_index`. The sentence is the
  model's prompt as is, with no template.

DexBench publishes each task's demonstrations in this layout as
`lerobot/cosmos_62d_320x180_4cam`. DexBench-Arena's
`scripts/training/relayout_lerobot_62d.py` produces it from a raw recording:
it reorders the joints and area-downscales the simulator's 640×360 renders to
320×180.

The reader finds each episode's rows from the `index` range each data parquet
holds, not from `data/file_index` in `meta/episodes`. Sets converted from
LeRobot v2.1 with LeRobot before 0.5.0 list the first episode of each new data
file under the previous file
([huggingface/lerobot#2401](https://github.com/huggingface/lerobot/issues/2401));
the reader logs a warning for such episodes and reads their rows correctly.

The joint order is:

| Group | Slice | Joint order |
| --- | --- | --- |
| Left arm | `0:7` | `left_arm_0` through `left_arm_6` |
| Right arm | `7:14` | `right_arm_0` through `right_arm_6` |
| Head | `14:16` | `head_0`, `head_1` |
| Hands | `16:56` | Joint stage, then left/right, then thumb/index/middle/ring/pinky |
| Torso | `56:62` | `torso_0` through `torso_5` |

Slices exclude the end index. The four stages are `cmc_flex`, `cmc_abd`, `mcp`,
`ip` for the thumb, and `mcp_flex`, `mcp_abd`, `pip`, `dip` for the other
fingers. For each stage, the five left-hand joints are followed by the five
right-hand joints. `observation.state` lists the same joints with a `_Pos`
suffix.

[`configs/dataloader/dexbench.yaml`](../../configs/dataloader/dexbench.yaml)
lists the 62 names, the frame size (`camera_frame_hw: [180, 320]`) and the
rate (`fps: 30`). The reader refuses a task set whose `meta/info.json` differs
from any of them. One task set per run: point `dataset_dir` at a single set.

## Cameras and Canvas

| Camera | Canvas slot (H×W) |
| --- | --- |
| `camera_ego_left` | head, top, 256×320 |
| `camera_wrist_left` | bottom left, 128×160 |
| `camera_wrist_right` | bottom right, 128×160 |

Each frame is resized to its slot with LANCZOS, without preserving the aspect
ratio, and the three tiles are composed into the 384×320 canvas by
`assemble_multiview_layout`. This is the generic LeRobot v3 multiview path. A
window whose wrist video fails to decode is skipped instead of being trained
with a black slot. Each window holds 33 rows: 32 actions and 9 video frames
(every 4th row). Only full windows are sampled.

## Action Space

The 62 joints are normalized first and then scattered into the shared 80-D
action vector (`unify_action_map` in the yaml):

| Joints | Unified slots |
| --- | --- |
| Left arm (7) | `0-6`, the left EEF pose slots |
| Right arm (7) | `34-40`, the right EEF pose slots |
| Head (2) | `68`, `69`, reserved |
| Left hand (20) | `10 + 4f + s` for finger `f` (thumb = 0) and stage `s` |
| Right hand (20) | `44 + 4f + s` |
| Torso (6) | `70-75`, reserved |

The remaining slots (grippers 9 and 43, EEF slots 7, 8, 41, 42, hand slots
30-33 and 64-67, reserved 76-79) are masked out of the loss. At inference, the
deploy normalizer gathers the 62 joints back from the model output.

## Normalization

`normalize_mode: min-max` maps each joint's `[min, max]` onto `[-1, 1]` with
clipping. Actions use the `action` statistics. Proprio uses the
`observation.state` statistics. The reader builds
`<dataset_dir>/meta/dexbench_normalization_stats.npy` on first load: rank 0
scans the set and the other ranks wait. Every checkpoint carries a copy as
`normalization_stats.npy`. To build the file ahead of training:

```bash
python -m openwam.dataloader.utils.stats_computation.dexbench_stats_computation \
  --dataset-dir /path/to/task_set
```

## Fine-Tuning

Download the OpenWAM-α foundation checkpoint (select OpenWAM_Alpha and
OpenWAM-Alpha-Pretrain-Foundation-Model):

```bash
python scripts/download_assets/download_openwam_checkpoints.py
```

Then run [train.sh](train.sh):

```bash
DEXBENCH_DATASET=/path/to/task_set \
OPENWAM_ALPHA_CKPT=<foundation_ckpt_dir_path> \
  bash benchmarks/dexbench/train.sh
```

| Setting | Value |
| --- | --- |
| Global batch | 256 = micro-batch × GPUs × gradient accumulation (default 4 × 8 × 8) |
| Optimizer updates | 16,000 |
| Optimizer | AdamW, betas (0.9, 0.95), weight decay 0.01, gradient clip 1.0 |
| Learning rate | 1e-4 for both the action and the video parameters |
| Schedule | 5% linear warmup, cosine decay to 1e-6 |
| Precision / sharding | bf16, DeepSpeed ZeRO-2, gradient checkpointing |
| Augmentation | none |
| Checkpoints | every 2,000 updates |

The model configuration is the default `dual_system` (Wan2.2-TI2V-5B, joint
self-attention, 80-D action and proprio). `training.max_steps` and
`training.save_steps` count micro-steps, so `train.sh` scales both by the
accumulation. It accepts these environment variables:

| Variable | Default | Meaning |
| --- | --- | --- |
| `NPROC_PER_NODE` | all visible GPUs | GPUs per node |
| `NNODES` | 1 | nodes; also set `NODE_RANK` and `MASTER_ADDR` as for `scripts/train.sh` |
| `MICROBATCH` | 4 | per-GPU batch; 256 must divide by micro-batch × total GPUs |
| `UPDATES` | 16000 | optimizer updates |
| `SAVE_EVERY` | 2000 | updates between checkpoints |

Additional arguments are forwarded as Hydra overrides, for example
`training.output_path=/path/to/runs` or `training.save_full_states_for_resume=true`.

Run two nodes with eight GPUs each (micro-batch 4, accumulation 4):

```bash
NNODES=2 NODE_RANK=0 MASTER_ADDR=<node0 address> DEXBENCH_DATASET=... OPENWAM_ALPHA_CKPT=... \
  bash benchmarks/dexbench/train.sh
NNODES=2 NODE_RANK=1 MASTER_ADDR=<node0 address> DEXBENCH_DATASET=... OPENWAM_ALPHA_CKPT=... \
  bash benchmarks/dexbench/train.sh
```

## Inference Server

[server.py](server.py) serves a checkpoint directory (`config.yaml`,
`checkpoint_step_*.safetensors` and `normalization_stats.npy`, as training
writes it) over HTTP. The DexBench evaluation client looks for OpenWAM at
`127.0.0.1:5816`:

```bash
CUDA_VISIBLE_DEVICES=0 python benchmarks/dexbench/server.py \
  --ckpt-dir <ckpt_dir_path> \
  --port 5816
```

- `GET /info` states the contract: `model_family: "openwam"`, the checkpoint,
  the 62 joint names in the order of `state` and `action`, `camera_set`
  (`camera_ego_left`, `camera_wrist_left`, `camera_wrist_right`, in training
  order), `frame_hw` (`[180, 320]`, the frame size the client must send),
  `action_horizon` (32), `fps`, `num_steps`, `seed` and
  `action_representation` (absolute).
- `POST /predict` with
  `{"images": {<camera>: <base64 RGB PNG>}, "state": [62 floats], "prompt": "<task sentence>"}`
  returns `{"action": [[62 floats] x 32]}`, absolute joint positions in
  radians. A malformed request gets `{"error": ...}` with status 400, and a
  failed inference gets status 500.

The client area-downscales each 640×360 render to `frame_hw`, the same
filter the dataset relayout used. The server then builds the canvas with the
reader's code path. Inference uses the native deploy stack
(`load_from_checkpoint_dir`, `JointInferenceEngine`). The server keeps no
episode state: each request generates a full chunk, and the client decides how
many actions to execute before asking again.

| Option | Default | Meaning |
| --- | --- | --- |
| `--num-steps` | 10 | denoising steps per chunk |
| `--seed` | 42 | sampling seed, the same for every request |
| `--ckpt-name` | latest | a specific `checkpoint_step_*.safetensors` |
| `--api-token-file` | none | require `Authorization: Bearer <token>` on every request |
| `--host` | `127.0.0.1` | bind address; keep it local unless the port is firewalled |
| `--dit-cache`, `--compile` | off | the deploy fast paths of `configs/deploy.yaml`; the DiT cache approximates the output |

## Tests

```bash
pytest tests/dataloader/test_dexbench_dataset.py tests/benchmarks/test_dexbench_server.py
```

The tests build a synthetic task set. They check the contract refusals, the
statistics, normalization followed by the slot scatter, the canvas, the
server's canvas against the reader's (bitwise), the deploy normalizer round
trip and the HTTP protocol with a stub engine.
