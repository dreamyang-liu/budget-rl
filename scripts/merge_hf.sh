#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERL_ROOT="${VERL_ROOT:-${ROOT}/third_party/verl}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/outputs/rl_pct30e5_kl005}"
BASE_MODEL="${MODEL:-${ROOT}/artifacts/models/sft_interval_pct30_e5}"
TARGET_DIR="${TARGET_DIR:-${OUTPUT_DIR}/huggingface_final}"

TRACKER="${OUTPUT_DIR}/latest_checkpointed_iteration.txt"
test -f "${TRACKER}" || { echo "ERROR: missing ${TRACKER}" >&2; exit 1; }
STEP="$(tr -d '[:space:]' < "${TRACKER}")"
ACTOR_DIR="${OUTPUT_DIR}/global_step_${STEP}/actor"
test -d "${ACTOR_DIR}" || { echo "ERROR: missing ${ACTOR_DIR}" >&2; exit 1; }
test -f "${BASE_MODEL}/config.json" || { echo "ERROR: invalid base model ${BASE_MODEL}" >&2; exit 1; }

if test ! -e "${ACTOR_DIR}/huggingface"; then
  ln -s "${BASE_MODEL}" "${ACTOR_DIR}/huggingface"
fi

export PYTHONPATH="${VERL_ROOT}:${PYTHONPATH:-}"
cd "${VERL_ROOT}"
python3 -m verl.model_merger merge \
  --backend fsdp \
  --local_dir "${ACTOR_DIR}" \
  --target_dir "${TARGET_DIR}"

echo "Merged checkpoint: ${TARGET_DIR}"
