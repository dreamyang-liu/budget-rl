#!/usr/bin/env python3
"""Check the local environment and downloaded artifacts before training."""

import importlib.util
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERL_ROOT = Path(os.environ.get("VERL_ROOT", ROOT / "third_party/verl"))
DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT / "artifacts/data/rl"))
MODEL = Path(os.environ.get("MODEL", ROOT / "artifacts/models/sft_interval_pct30_e5"))


def status(ok, message):
    print(f"[{'OK' if ok else 'FAIL'}] {message}")
    return ok


def main():
    checks = []
    checks.append(status((VERL_ROOT / "verl").is_dir(), f"verl root: {VERL_ROOT}"))
    checks.append(status((DATA_DIR / "train.parquet").is_file(), f"train data: {DATA_DIR / 'train.parquet'}"))
    checks.append(status((DATA_DIR / "test.parquet").is_file(), f"validation data: {DATA_DIR / 'test.parquet'}"))
    checks.append(status((MODEL / "config.json").is_file(), f"starter model: {MODEL}"))

    if (MODEL / "model.safetensors.index.json").is_file():
        index = json.loads((MODEL / "model.safetensors.index.json").read_text())
        shards = sorted(set(index.get("weight_map", {}).values()))
        missing = [name for name in shards if not (MODEL / name).is_file()]
        checks.append(status(not missing, f"model shards: {len(shards)} expected, {len(missing)} missing"))

    try:
        import pyarrow.parquet as pq

        for name in ("train", "test"):
            path = DATA_DIR / f"{name}.parquet"
            if not path.is_file():
                continue
            rows = pq.read_table(path).to_pylist()
            possible = sum(bool(row["extra_info"].get("is_possible")) for row in rows)
            impossible = len(rows) - possible
            checks.append(status(bool(rows), f"{name}: {len(rows)} rows, {possible} possible, {impossible} impossible"))
    except Exception as exc:
        checks.append(status(False, f"parquet validation failed: {exc}"))

    sys.path.insert(0, str(VERL_ROOT))
    spec = importlib.util.find_spec("verl")
    vendored_import = spec is not None and spec.origin is not None and Path(spec.origin).resolve().is_relative_to(
        VERL_ROOT.resolve()
    )
    checks.append(status(vendored_import, f"vendored verl is importable from {spec.origin if spec else 'missing'}"))

    try:
        import torch

        count = torch.cuda.device_count()
        checks.append(status(torch.cuda.is_available(), f"CUDA available; visible GPUs={count}"))
        if count:
            print("     " + ", ".join(torch.cuda.get_device_name(i) for i in range(count)))
    except Exception as exc:
        checks.append(status(False, f"torch/CUDA check failed: {exc}"))

    if not all(checks):
        print("\nPreflight failed. Fix the FAIL items before starting a full run.")
        return 1
    print("\nPreflight passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
