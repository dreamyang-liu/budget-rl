#!/usr/bin/env python3
"""
Read reward metrics JSONL and log batch summaries to W&B.
Runs as a daemon alongside GRPO training.

Usage:
    python wandb_reward_sync.py \
        --metrics-log /tmp/reward_metrics_pct10.jsonl \
        --wandb-project budget_probe_grpo \
        --wandb-run rl_pct10_e3 \
        --poll-interval 30
"""
import argparse
import json
import os
import time

import wandb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-log", required=True)
    ap.add_argument("--wandb-project", default="budget_probe_grpo")
    ap.add_argument("--wandb-run", required=True)
    ap.add_argument("--poll-interval", type=int, default=30)
    ap.add_argument("--max-idle", type=int, default=600,
                    help="Exit after N seconds of no new data")
    args = ap.parse_args()

    run = wandb.init(
        project=args.wandb_project,
        name=f"{args.wandb_run}_reward_detail",
        group=args.wandb_run,
        job_type="reward_metrics",
        reinit=True,
    )

    lines_read = 0
    last_data_time = time.time()
    step = 0

    print(f"Watching {args.metrics_log} → W&B {args.wandb_project}/{args.wandb_run}")

    while True:
        if not os.path.exists(args.metrics_log):
            time.sleep(args.poll_interval)
            continue

        new_lines = 0
        with open(args.metrics_log) as f:
            for i, line in enumerate(f):
                if i < lines_read:
                    continue
                lines_read = i + 1
                new_lines += 1
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                if d.get("type") != "batch_summary":
                    continue

                step += 1
                wandb.log({
                    "reward_detail/class_accuracy": d.get("class_accuracy", 0),
                    "reward_detail/pred_hit_possible": d.get("pred_hit_possible", 0),
                    "reward_detail/pred_hit_impossible": d.get("pred_hit_impossible", 0),
                    "reward_detail/cover_rate": d.get("cover_rate", 0),
                    "reward_detail/reward_all_mean": d.get("reward_all_mean", 0),
                    "reward_detail/reward_possible_mean": d.get("reward_possible_mean", 0),
                    "reward_detail/reward_impossible_mean": d.get("reward_impossible_mean", 0),
                    "reward_detail/width_mean": d.get("width_mean", 0),
                    "reward_detail/width_median": d.get("width_median", 0),
                    "reward_detail/mre_mean": d.get("mre_mean", 0),
                    "reward_detail/mre_median": d.get("mre_median", 0),
                    "reward_detail/n_covered": d.get("n_covered", 0),
                    "reward_detail/n_interval": d.get("n_interval", 0),
                }, step=step)

        if new_lines > 0:
            last_data_time = time.time()
            print(f"  logged {new_lines} batch summaries (total step={step})")

        if time.time() - last_data_time > args.max_idle:
            print(f"No new data for {args.max_idle}s, exiting.")
            break

        time.sleep(args.poll_interval)

    run.finish()


if __name__ == "__main__":
    main()
