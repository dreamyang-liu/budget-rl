#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REGION="${AWS_REGION:-us-east-2}"
S3_ROOT="${S3_ROOT:-s3://drmyang-training-data-241580540779-us-east-2-an/agent-budget-control-ckpt}"
MODE="${1:-data}"

command -v aws >/dev/null || { echo "ERROR: aws CLI not found" >&2; exit 1; }

download_data() {
  mkdir -p "${ROOT}/artifacts/data"
  aws s3 sync "${S3_ROOT}/v3/data/rl/" "${ROOT}/artifacts/data/rl/" --region "${REGION}" --only-show-errors
  aws s3 sync "${S3_ROOT}/v3/data/eval_test/" "${ROOT}/artifacts/data/eval_test/" --region "${REGION}" --only-show-errors
  aws s3 sync "${S3_ROOT}/v3/data/splits/" "${ROOT}/artifacts/data/splits/" --region "${REGION}" --only-show-errors
}

download_starter() {
  mkdir -p "${ROOT}/artifacts/models/sft_interval_pct30_e5"
  echo "Downloading approximately 14.2 GiB..."
  aws s3 sync \
    "${S3_ROOT}/v3/sft_interval_pct30/huggingface_e5/" \
    "${ROOT}/artifacts/models/sft_interval_pct30_e5/" \
    --region "${REGION}" --only-show-errors
}

case "${MODE}" in
  data) download_data ;;
  starter) download_starter ;;
  all) download_data; download_starter ;;
  *) echo "Usage: $0 {data|starter|all}" >&2; exit 2 ;;
esac

echo "Done: ${ROOT}/artifacts"
