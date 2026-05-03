#!/bin/bash
# Evaluate baseline + all SFT ablation checkpoints on the held-out test set.
#
# For each model:
#   1. Start vLLM OpenAI-API server
#   2. Wait until /v1/models responds
#   3. Run eval_checkpoint.py
#   4. Stop server
#
# Aggregates per-variant summaries into one CSV.
#
# Override: BASE, BASELINE_MODEL, VLLM_PORT, GPU.
set -uo pipefail  # not -e: we want to continue past failures

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE="${BASE:-${SCRIPT_DIR}/ablation_data}"
RESULTS_DIR="${BASE}/eval_results"
TEST_PARQUET="${BASE}/eval_test/train.parquet"

BASELINE_MODEL="${BASELINE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
VLLM_PORT="${VLLM_PORT:-8765}"
GPU="${GPU:-0}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-8}"
MAX_TOKENS="${MAX_TOKENS:-512}"

mkdir -p "${RESULTS_DIR}"

if [ ! -f "${TEST_PARQUET}" ]; then
  echo "ERROR: ${TEST_PARQUET} not found. Run prepare_all_ablations.sh first." >&2
  exit 1
fi

start_vllm() {
  local model_path="$1"
  local served_name="$2"
  echo
  echo "=== Starting vLLM server for ${served_name} ==="
  CUDA_VISIBLE_DEVICES="${GPU}" python3 -m vllm.entrypoints.openai.api_server \
    --model "${model_path}" \
    --served-model-name "${served_name}" \
    --port "${VLLM_PORT}" \
    --gpu-memory-utilization 0.6 \
    --max-model-len 8192 \
    --dtype bfloat16 \
    > "${RESULTS_DIR}/_vllm_${served_name}.log" 2>&1 &
  VLLM_PID=$!
  echo "vLLM PID=${VLLM_PID}, waiting for /v1/models ..."

  # Wait up to 5 minutes for ready
  local waited=0
  while ! curl -sf "http://localhost:${VLLM_PORT}/v1/models" >/dev/null 2>&1; do
    sleep 5
    waited=$((waited + 5))
    if [ $waited -gt 300 ]; then
      echo "ERROR: vLLM did not become ready in 5 minutes" >&2
      kill "${VLLM_PID}" 2>/dev/null || true
      return 1
    fi
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
      echo "ERROR: vLLM process died early. Check ${RESULTS_DIR}/_vllm_${served_name}.log" >&2
      tail -30 "${RESULTS_DIR}/_vllm_${served_name}.log" >&2
      return 1
    fi
  done
  echo "vLLM ready (waited ${waited}s)"
}

stop_vllm() {
  if [ -n "${VLLM_PID:-}" ] && kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "Stopping vLLM PID=${VLLM_PID}"
    kill "${VLLM_PID}" 2>/dev/null || true
    sleep 5
    kill -9 "${VLLM_PID}" 2>/dev/null || true
    wait "${VLLM_PID}" 2>/dev/null || true
  fi
  unset VLLM_PID
}

trap 'stop_vllm; exit' INT TERM

eval_one() {
  local label="$1" model_path="$2" served_name="$3"
  echo
  echo "########################################"
  echo "# Evaluating: ${label}"
  echo "# model: ${model_path}"
  echo "########################################"
  if ! start_vllm "${model_path}" "${served_name}"; then
    echo "skip ${label} (vLLM failed to start)"
    return 1
  fi

  python3 eval_checkpoint.py \
    --vllm-url "http://localhost:${VLLM_PORT}" \
    --model-name "${served_name}" \
    --test-parquet "${TEST_PARQUET}" \
    --output "${RESULTS_DIR}/${label}.json" \
    --max-tokens "${MAX_TOKENS}" \
    --max-concurrency "${MAX_CONCURRENCY}" || echo "WARN: eval failed for ${label}"

  stop_vllm
}

# 1. Baseline
eval_one "baseline" "${BASELINE_MODEL}" "baseline"

# 2. Each SFT ablation checkpoint
for ABL in sft_point sft_interval_pct10 sft_interval_pct30 sft_interval_pct50 \
           sft_interval_fix100 sft_interval_fix500 sft_interval_fix1000; do
  CKPT="${BASE}/checkpoints/${ABL}"
  # VeRL writes final checkpoint under ${SAVE_DIR}/global_step_<N>/actor or similar
  # Try common patterns:
  if [ -d "${CKPT}/huggingface" ]; then
    MODEL_PATH="${CKPT}/huggingface"
  elif ls "${CKPT}"/global_step_*/actor 2>/dev/null | tail -1 >/dev/null; then
    MODEL_PATH=$(ls -d "${CKPT}"/global_step_*/actor 2>/dev/null | tail -1)
  else
    MODEL_PATH="${CKPT}"
  fi

  if [ ! -f "${MODEL_PATH}/config.json" ]; then
    echo "skip ${ABL} (no config.json at ${MODEL_PATH})"
    continue
  fi
  eval_one "${ABL}" "${MODEL_PATH}" "${ABL}"
done

# 3. Aggregate
echo
echo "=== Aggregate results ==="
python3 - <<PY
import json, os, glob

results_dir = "${RESULTS_DIR}"
rows = []
for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
    if path.endswith("_vllm.json"): continue
    with open(path) as f:
        d = json.load(f)
    s = d.get("summary", {})
    label = os.path.splitext(os.path.basename(path))[0]
    rows.append({
        "label": label,
        "n": s.get("n_samples", 0),
        "mean_reward": s.get("mean_reward", 0),
        "fmt_valid": s.get("format_valid_rate", 0),
        "class_acc": s.get("class_accuracy", 0),
        "phit_pos": s.get("pred_hit_possible", 0),
        "phit_imp": s.get("pred_hit_impossible", 0),
        "cover": s.get("cover_rate_possible", 0),
        "mre_mean": s.get("mean_relative_error", 0),
        "mre_med": s.get("median_relative_error", 0),
    })

if rows:
    headers = list(rows[0].keys())
    csv_path = os.path.join(results_dir, "_summary.csv")
    with open(csv_path, "w") as f:
        f.write(",".join(headers) + "\n")
        for r in rows:
            f.write(",".join(f"{r[h]:.4f}" if isinstance(r[h], float) else str(r[h]) for h in headers) + "\n")
    print(f"summary -> {csv_path}")
    # Pretty print
    widths = [max(len(h), max((len(f"{r[h]:.4f}" if isinstance(r[h], float) else str(r[h])) for r in rows), default=0)) for h in headers]
    print(" | ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("-+-".join("-"*w for w in widths))
    for r in rows:
        cells = [f"{r[h]:.4f}" if isinstance(r[h], float) else str(r[h]) for h in headers]
        print(" | ".join(c.ljust(w) for c, w in zip(cells, widths)))
PY
