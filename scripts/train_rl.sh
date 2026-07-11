#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERL_ROOT="${VERL_ROOT:-${ROOT}/third_party/verl}"
DATA_DIR="${DATA_DIR:-${ROOT}/artifacts/data/rl}"
MODEL="${MODEL:-${ROOT}/artifacts/models/sft_interval_pct30_e5}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/outputs/rl_pct30e5_kl005}"
REWARD_FN="${ROOT}/reward/budget_probe_reward.py"

NGPUS="${NGPUS:-8}"
NNODES="${NNODES:-1}"
TP_SIZE="${TP_SIZE:-4}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-64}"
PPO_MICRO_BATCH_SIZE="${PPO_MICRO_BATCH_SIZE:-4}"
ROLLOUT_N="${ROLLOUT_N:-16}"
LR="${LR:-5e-7}"
KL_COEF="${KL_COEF:-0.05}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-5}"
SAVE_FREQ="${SAVE_FREQ:-10}"
TEST_FREQ="${TEST_FREQ:-5}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-rl_pct30e5_kl005_repro}"
CENTRALITY_LAMBDA="${CENTRALITY_LAMBDA:-0}"
SEED="${SEED:-42}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.4}"
ROLLOUT_MAX_MODEL_LEN="${ROLLOUT_MAX_MODEL_LEN:-8192}"

for path in "${VERL_ROOT}/verl" "${DATA_DIR}/train.parquet" "${DATA_DIR}/test.parquet" \
            "${MODEL}/config.json" "${REWARD_FN}"; do
  test -e "${path}" || { echo "ERROR: required path missing: ${path}" >&2; exit 1; }
done

mkdir -p "${OUTPUT_DIR}"
export PYTHONPATH="${VERL_ROOT}:${PYTHONPATH:-}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
export REWARD_METRICS_LOG="${REWARD_METRICS_LOG:-${OUTPUT_DIR}/reward_metrics.jsonl}"
export CENTRALITY_LAMBDA

if test "${DRY_RUN:-0}" = "1"; then
  set -- "$@" --cfg job
fi

cd "${VERL_ROOT}"
python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files="${DATA_DIR}/train.parquet" \
  data.val_files="${DATA_DIR}/test.parquet" \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.seed="${SEED}" \
  data.max_prompt_length=8192 \
  data.max_response_length=1024 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  actor_rollout_ref.model.path="${MODEL}" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr="${LR}" \
  actor_rollout_ref.actor.data_loader_seed="${SEED}" \
  actor_rollout_ref.actor.fsdp_config.seed="${SEED}" \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE}" \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef="${KL_COEF}" \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${TP_SIZE}" \
  actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION}" \
  actor_rollout_ref.rollout.max_model_len="${ROLLOUT_MAX_MODEL_LEN}" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE}" \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE}" \
  actor_rollout_ref.ref.fsdp_config.seed="${SEED}" \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  reward.custom_reward_function.path="${REWARD_FN}" \
  reward.custom_reward_function.name=compute_score \
  trainer.critic_warmup=0 \
  'trainer.logger=["console"]' \
  trainer.project_name=budget_probe_grpo \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.n_gpus_per_node="${NGPUS}" \
  trainer.nnodes="${NNODES}" \
  trainer.default_local_dir="${OUTPUT_DIR}" \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.test_freq="${TEST_FREQ}" \
  trainer.val_before_train=True \
  trainer.resume_mode="${RESUME_MODE:-disable}" \
  trainer.total_epochs="${TOTAL_EPOCHS}" \
  "$@"
