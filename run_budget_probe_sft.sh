#!/bin/bash
# SFT warm-up for budget probe estimator.
# Trains Qwen2.5-7B-Instruct on the reserved SFT split so the model learns
# the output format (<think>...</think><answer>[L, H]|impossible</answer>)
# and basic token-budget reasoning before RL starts.
#
# Prerequisites (produces a DISJOINT SFT/RL split at trajectory level):
#   python prepare_budget_probe.py \
#       --input qwen-2.5-7b-sokoban-128.jsonl \
#       --output-dir ./budget_probe_data \
#       --max-tokens 32768 --probe-every-n 1 --margin 0.1 \
#       --balance --balance-ratio 1.0 --oversample-possible 2 \
#       --sft-fraction 0.3
#
# After SFT finishes, point the RL script at the checkpoint:
#   MODEL=/workspace/verl-x/checkpoints/budget_probe_sft/qwen2.5_7b_sft bash run_budget_probe_grpo.sh

set -x

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/budget_probe_sft_v2"
SAVE_DIR="${SCRIPT_DIR}/checkpoints/budget_probe_sft_v2/qwen2.5_7b_sft"

# H200 is SM 9.0
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"

NGPUS=${NGPUS:-8}
MODEL=${MODEL:-"Qwen/Qwen2.5-7B-Instruct"}
LR=${LR:-5e-6}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-5}
MICRO_BS=${MICRO_BS:-2}

mkdir -p "${SAVE_DIR}"

torchrun --standalone --nnodes=1 --nproc_per_node=${NGPUS} \
    -m verl.trainer.sft_trainer \
    data.train_files="${DATA_DIR}/train.parquet" \
    data.val_files="${DATA_DIR}/test.parquet" \
    data.messages_key=messages \
    data.train_batch_size=16 \
    data.micro_batch_size_per_gpu=${MICRO_BS} \
    data.max_length=9216 \
    data.ignore_input_ids_mismatch=True \
    optim.lr=${LR} \
    engine=fsdp \
    model.path=${MODEL} \
    model.use_remove_padding=True \
    trainer.default_local_dir="${SAVE_DIR}" \
    trainer.project_name=budget_probe_sft \
    trainer.experiment_name=qwen2.5_7b_sft_warmup \
    trainer.logger=console \
    trainer.total_epochs=${TOTAL_EPOCHS} "$@"
