#!/bin/bash
set -uo pipefail
cd /workspace/verl-x
BASE="/workspace/verl-x/ablation_data_v3"
VARIANTS=(sft_interval_pct10 sft_interval_pct30 sft_interval_pct50 sft_interval_fix100 sft_interval_fix500 sft_interval_fix1000)
NGPUS_PER=4
SLOTS=2

run_slot() {
  local V=$1 SLOT=$2
  local GPU_START=$((SLOT * NGPUS_PER))
  local GPU_END=$((GPU_START + NGPUS_PER - 1))
  local GPUS=$(seq -s, ${GPU_START} ${GPU_END})
  local LOG="${BASE}/eval_results/_sft_${V}.log"
  echo "[$(date +%H:%M:%S)] Starting ${V} on GPUs ${GPUS}"
  CUDA_VISIBLE_DEVICES=${GPUS} NGPUS=${NGPUS_PER} BASE=${BASE} \
    bash run_sft_ablation.sh ${V} data.max_length=16384 data.max_token_len_per_gpu=16384 > "${LOG}" 2>&1
  echo "[$(date +%H:%M:%S)] $([ $? -eq 0 ] && echo '✅' || echo '❌') ${V}"
}

mkdir -p "${BASE}/eval_results"
i=0
while [ $i -lt ${#VARIANTS[@]} ]; do
  PIDS=()
  for s in $(seq 0 $((SLOTS-1))); do
    idx=$((i + s))
    [ $idx -lt ${#VARIANTS[@]} ] && { run_slot "${VARIANTS[$idx]}" $s & PIDS+=($!); }
  done
  for pid in "${PIDS[@]}"; do wait $pid; done
  i=$((i + SLOTS))
done

echo ""
echo "=== ALL 6 SFT COMPLETE ==="
for V in "${VARIANTS[@]}"; do
  VLOSS=$(grep "Final validation" "${BASE}/eval_results/_sft_${V}.log" 2>/dev/null)
  printf "%-25s %s\n" "${V}" "${VLOSS:-(failed)}"
done
