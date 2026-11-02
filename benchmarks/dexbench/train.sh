#!/usr/bin/env bash
# Fine-tune OpenWAM-α on one DexBench task set (ego left + two wrist cameras).
#
#   DEXBENCH_DATASET=/path/to/task_set OPENWAM_ALPHA_CKPT=/path/to/foundation_ckpt_dir \
#     bash benchmarks/dexbench/train.sh
#
# Global batch 256 and 16,000 optimizer updates at any GPU count: the per-GPU
# micro-batch (MICROBATCH, default 4) and the gradient accumulation are chosen
# so that MICROBATCH x GPUs x accumulation = 256. training.max_steps and
# training.save_steps count micro-steps, so both are scaled by the accumulation.
# Multi-node: set NNODES, NODE_RANK and MASTER_ADDR as for scripts/train.sh.
# Extra Hydra overrides are forwarded, e.g. training.output_path=/path/to/runs.
set -euo pipefail
cd "$(dirname "$0")/../.."

: "${DEXBENCH_DATASET:?set DEXBENCH_DATASET to a DexBench task set (LeRobot v3, 62 joints, four cameras)}"
: "${OPENWAM_ALPHA_CKPT:?set OPENWAM_ALPHA_CKPT to the OpenWAM-Alpha foundation checkpoint directory}"

export NPROC_PER_NODE="${NPROC_PER_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
export NNODES="${NNODES:-1}"
GLOBAL_BATCH=256
MICROBATCH="${MICROBATCH:-4}"
UPDATES="${UPDATES:-16000}"
SAVE_EVERY="${SAVE_EVERY:-2000}"

WORLD=$((NPROC_PER_NODE * NNODES))
if (( WORLD < 1 || GLOBAL_BATCH % (WORLD * MICROBATCH) != 0 )); then
    echo "global batch ${GLOBAL_BATCH} is not divisible by ${WORLD} GPUs x micro-batch ${MICROBATCH}" >&2
    exit 1
fi
ACCUM=$((GLOBAL_BATCH / (WORLD * MICROBATCH)))
echo "[dexbench] ${WORLD} GPUs x micro-batch ${MICROBATCH} x accumulation ${ACCUM} = ${GLOBAL_BATCH}; ${UPDATES} updates"

exec bash scripts/train.sh \
    dataloader=dexbench \
    "dataloader.dataset_dir=${DEXBENCH_DATASET}" \
    "training.finetune_ckpt_path=${OPENWAM_ALPHA_CKPT}" \
    training.num_epochs=null \
    "training.max_steps=$((UPDATES * ACCUM))" \
    "training.save_steps=$((SAVE_EVERY * ACCUM))" \
    "training.batch_size=${MICROBATCH}" \
    "training.gradient_accumulation_steps=${ACCUM}" \
    training.learning_rate=1e-4 \
    training.action_lr=1e-4 \
    training.video_lr=1e-4 \
    "$@"
