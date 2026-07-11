#!/bin/bash
set -uo pipefail
cd /workspace/verl-x
BASE="/workspace/verl-x/ablation_data_v3"
RESULTS_MD="${BASE}/overnight_results.md"
PORT=8765

cat > "${RESULTS_MD}" << 'HEADER'
# Overnight RL Experiment Results

## Setup
- Base model: Qwen/Qwen2.5-7B-Instruct
- Task: Sokoban budget estimation (budget=2500)
- Data: v3 (new prompt with example + total + tightness hint)
- RL: GRPO, 8 GPU, rollout_n=16, batch=64, lr=5e-7, 5 epochs
- Reward: strict cover-only (no format bonus, no proximity)
- Eval: 380 balanced test samples (190 possible + 190 impossible)

## SFT Baselines
| Variant | cls_acc | cover | MRE_med | R |
|---|---|---|---|---|
| baseline | 25.5% | 10.5% | 65.5% | 0.021 |
| pct10_e5 | 90.3% | 26.3% | 45.1% | 0.277 |
| fix100_e3 | 90.5% | 52.6% | 42.9% | 0.095 |
| pct30_e5 | 91.1% | 37.4% | 53.9% | 0.210 |

---

HEADER

echo "=== Overnight RL experiments starting at $(date) ===" | tee -a "${RESULTS_MD}"

run_rl() {
  local NAME=$1 MODEL=$2 KL=$3
  local CKPT_DIR="${BASE}/checkpoints/${NAME}"
  local LOG="${BASE}/eval_results/_${NAME}.log"

  echo "" | tee -a "${RESULTS_MD}"
  echo "[$(date +%H:%M:%S)] === Starting ${NAME} (kl=${KL}, 8 GPU) ===" | tee -a "${RESULTS_MD}"

  rm -rf "${CKPT_DIR}"

  REWARD_METRICS_LOG="/tmp/reward_metrics_${NAME}.jsonl" \
  DATA_DIR="${BASE}" \
  MODEL="${MODEL}" \
  NGPUS=8 TP_SIZE=4 \
  TRAIN_BATCH_SIZE=64 LR=5e-7 ROLLOUT_N=16 \
  TOTAL_EPOCHS=5 \
  EXPERIMENT_NAME="${NAME}" \
  RESUME_MODE=disable \
  bash run_budget_probe_grpo.sh \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.kl_loss_coef=${KL} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.max_model_len=8192 \
    trainer.default_local_dir="${CKPT_DIR}" \
    trainer.save_freq=10 \
    'trainer.logger=["console","wandb"]' \
    trainer.project_name=budget_probe_grpo_overnight \
    > "${LOG}" 2>&1

  local RC=$?
  if [ $RC -eq 0 ]; then
    echo "[$(date +%H:%M:%S)] ✅ ${NAME} training done" | tee -a "${RESULTS_MD}"
  else
    echo "[$(date +%H:%M:%S)] ❌ ${NAME} training failed (rc=${RC})" | tee -a "${RESULTS_MD}"
    # Show error
    grep -E "Error|OOM|FAILED" "${LOG}" | tail -3 >> "${RESULTS_MD}"
    return $RC
  fi
}

convert_and_eval() {
  local NAME=$1
  local CKPT_DIR="${BASE}/checkpoints/${NAME}"
  local LATEST=$(cat "${CKPT_DIR}/latest_checkpointed_iteration.txt" 2>/dev/null)

  if [ -z "${LATEST}" ]; then
    echo "  no checkpoint found for ${NAME}" | tee -a "${RESULTS_MD}"
    return 1
  fi

  local ACTOR_DIR="${CKPT_DIR}/global_step_${LATEST}/actor"

  # Find a reference HF config from any SFT checkpoint
  local REF_HF=""
  for d in ${BASE}/checkpoints/sft_interval_pct10/huggingface_e*; do
    [ -f "${d}/config.json" ] && REF_HF="${d}" && break
  done
  [ -z "${REF_HF}" ] && REF_HF="${BASE}/checkpoints/sft_interval_pct30/huggingface_e3"

  [ ! -d "${ACTOR_DIR}/huggingface" ] && ln -sf "${REF_HF}" "${ACTOR_DIR}/huggingface"

  local HF_OUT="${CKPT_DIR}/huggingface_final"
  echo "  converting step ${LATEST} -> HF ..."
  python3 -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "${ACTOR_DIR}" \
    --target_dir "${HF_OUT}" 2>&1 | tail -2

  if [ ! -f "${HF_OUT}/config.json" ]; then
    echo "  HF conversion failed for ${NAME}" | tee -a "${RESULTS_MD}"
    return 1
  fi

  echo "  evaluating ${NAME} ..."
  python3 -m vllm.entrypoints.openai.api_server \
    --model "${HF_OUT}" --served-model-name "${NAME}" \
    --port ${PORT} --gpu-memory-utilization 0.6 --max-model-len 16384 \
    --dtype bfloat16 > /dev/null 2>&1 &
  local VPID=$!

  for _ in $(seq 1 60); do
    curl -sf "http://localhost:${PORT}/v1/models" > /dev/null 2>&1 && break
    kill -0 $VPID 2>/dev/null || break
    sleep 5
  done

  if curl -sf "http://localhost:${PORT}/v1/models" > /dev/null 2>&1; then
    python3 eval_checkpoint.py \
      --vllm-url "http://localhost:${PORT}" --model-name "${NAME}" \
      --test-parquet "${BASE}/eval_test/train.parquet" \
      --output "${BASE}/eval_results/${NAME}.json" \
      --max-tokens 512 --max-concurrency 8 2>&1 | tail -5

    python3 -c "
import json
with open('${BASE}/eval_results/${NAME}.json') as f:
    s = json.load(f).get('summary', {})
print(f'''
### ${NAME} (step ${LATEST})
| cls_acc | recall(pos) | recall(imp) | cover | MRE_med | R |
|---|---|---|---|---|---|
| {s.get('class_accuracy',0):.1%} | {s.get('pred_hit_possible',0):.1%} | {s.get('pred_hit_impossible',0):.1%} | {s.get('cover_rate_possible',0):.1%} | {s.get('median_relative_error',0):.1%} | {s.get('mean_reward',0):.3f} |
''')
" >> "${RESULTS_MD}"
  fi

  kill $VPID 2>/dev/null; wait $VPID 2>/dev/null; sleep 5
}

# === Run 4 experiments sequentially (8 GPU each) ===

EXPERIMENTS=(
  "rl_pct10e5_kl005|${BASE}/checkpoints/sft_interval_pct10/huggingface_e5|0.05"
  "rl_fix100e3_kl005|${BASE}/checkpoints/sft_interval_fix100/huggingface_e3|0.05"
  "rl_pct30e5_kl005|${BASE}/checkpoints/sft_interval_pct30/huggingface_e5|0.05"
  "rl_pct10e5_kl01|${BASE}/checkpoints/sft_interval_pct10/huggingface_e5|0.1"
)

for entry in "${EXPERIMENTS[@]}"; do
  IFS='|' read -r NAME MODEL KL <<< "$entry"
  run_rl "${NAME}" "${MODEL}" "${KL}"
  convert_and_eval "${NAME}"
done

# === Final summary ===
echo "" | tee -a "${RESULTS_MD}"
echo "## Final Comparison" | tee -a "${RESULTS_MD}"
echo "" >> "${RESULTS_MD}"

python3 << 'PY' >> "${RESULTS_MD}"
import json, os
base = '/workspace/verl-x/ablation_data_v3/eval_results'
labels = ['baseline', 'sft_interval_pct10_e5', 'sft_interval_fix100_e3', 'sft_interval_pct30_e5',
          'rl_pct10e5_kl005', 'rl_pct10e5_kl01', 'rl_fix100e3_kl005', 'rl_pct30e5_kl005']
print('| Model | cls_acc | recall(pos) | recall(imp) | cover | MRE_med | R |')
print('|---|---|---|---|---|---|---|')
for label in labels:
    path = f'{base}/{label}.json'
    if not os.path.exists(path): continue
    with open(path) as f:
        s = json.load(f).get('summary', {})
    print(f'| {label} | {s.get("class_accuracy",0):.1%} | {s.get("pred_hit_possible",0):.1%} | {s.get("pred_hit_impossible",0):.1%} | {s.get("cover_rate_possible",0):.1%} | {s.get("median_relative_error",0):.1%} | {s.get("mean_reward",0):.3f} |')
PY

echo ""
echo "=== Overnight experiments completed at $(date) ===" | tee -a "${RESULTS_MD}"
