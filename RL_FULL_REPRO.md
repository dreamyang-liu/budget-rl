# 5-epoch Budget RL reproduction

Date: 2026-07-10

## Status

The full GRPO training run completed successfully on 8 H100 80 GB GPUs:

- 5 epochs, 75 optimizer steps (15 steps per epoch)
- one FSDP checkpoint saved at the end of every epoch
- all five checkpoints merged to Hugging Face format
- all five checkpoints evaluated on the same 380-sample held-out set
- no failed evaluation requests; format-valid rate was 100% for every epoch
- training wall clock: 2:12:54

The final result closely reproduces the archived `rl_pct30e5_kl005` result. The
classification accuracy is identical (90.00%); final mean reward differs by
0.00188 and coverage differs by 1.05 percentage points.

## Training configuration

The experiment-defining parameters match the archived run:

| Parameter | Value |
|---|---:|
| Algorithm | GRPO, no critic |
| Starter | `sft_interval_pct30_e5` |
| GPUs | 8 x H100 80 GB |
| Train batch | 64 |
| Rollouts per prompt | 16 |
| Tensor parallel | 4 |
| Learning rate | `5e-7` |
| KL coefficient | `0.05` |
| Epochs | 5 |
| Max prompt / rollout model length | 8192 |
| Checkpoint frequency | 15 steps (one epoch) |

The original H200 recipe used PPO/log-prob micro-batch 4 and vLLM memory
utilization 0.4. On H100 80 GB that configuration completed its first step but
OOMed while vLLM remapped weights during wake-up. The successful run used
micro-batch 2 and vLLM memory utilization 0.3. Effective train batch, rollout
count, optimizer settings, reward, and number of updates were unchanged.

Online validation was disabled (`test_freq=-1`, `val_before_train=False`) so
that every saved epoch could instead be evaluated with the archived standalone
evaluation script under exactly the same inference settings.

## Evaluation results

Evaluation settings: vLLM 0.12.0, bfloat16, max model length 16,384, greedy
decoding (`temperature=0`), max 512 generated tokens, concurrency 8, and the
archived `eval_checkpoint.py` plus strict reward function.

| Checkpoint | Accuracy | Recall possible | Recall impossible | Coverage possible | Median MRE | Mean MRE | Mean reward |
|---|---:|---:|---:|---:|---:|---:|---:|
| SFT reproduction | 91.05% | 95.26% | 86.84% | 37.37% | 53.90% | 63.16% | 0.20995 |
| RL epoch 1 (step 15) | 89.74% | 92.63% | 86.84% | 36.84% | 53.89% | 47.34% | 0.20823 |
| RL epoch 2 (step 30) | 89.74% | 91.05% | 88.42% | 40.00% | 52.39% | 44.18% | 0.22826 |
| RL epoch 3 (step 45) | 90.26% | 93.68% | 86.84% | 44.21% | 38.14% | 42.02% | 0.24949 |
| RL epoch 4 (step 60) | 90.00% | 93.16% | 86.84% | 45.26% | 33.05% | 47.76% | 0.25711 |
| RL epoch 5 (step 75) | 90.00% | 91.58% | 88.42% | 45.79% | 30.41% | 56.29% | 0.26183 |
| Archived RL final | 90.00% | 92.63% | 87.37% | 46.84% | 28.22% | 56.05% | 0.26371 |

The optimization signal improves consistently by the primary measures: mean
reward rises from 0.20823 at epoch 1 to 0.26183 at epoch 5, coverage rises by
8.95 percentage points, and median MRE falls by 23.48 percentage points. Mean
MRE is less stable because a few large relative errors dominate the mean; the
median and strict coverage are more representative here.

Relative to the reproduced SFT baseline, the final checkpoint gains 8.42
percentage points of strict coverage and 0.05188 mean reward, while classification
accuracy falls by 1.05 percentage points. This is the same trade-off shown in
the archived report.

## Archived-final comparison

| Metric | Archived final | This run | Difference |
|---|---:|---:|---:|
| Accuracy | 90.00% | 90.00% | 0.00 pp |
| Recall possible | 92.63% | 91.58% | -1.05 pp |
| Recall impossible | 87.37% | 88.42% | +1.05 pp |
| Coverage possible | 46.84% | 45.79% | -1.05 pp |
| Median MRE | 28.22% | 30.41% | +2.19 pp |
| Mean MRE | 56.05% | 56.29% | +0.23 pp |
| Mean reward | 0.26371 | 0.26183 | -0.00188 |

This is a close statistical reproduction, not a bit-exact replay. The hardware
(H100 versus archived H200), smaller gradient micro-batch, floating-point
accumulation order, and vLLM sampling/batching nondeterminism can all move a few
borderline examples.

## Artifacts

All paths below are relative to this repository and are intentionally ignored
by Git because of their size:

```text
outputs/rl_pct30e5_kl005_full/train.log
outputs/rl_pct30e5_kl005_full/reward_metrics.jsonl
outputs/rl_pct30e5_kl005_full/global_step_{15,30,45,60,75}/
outputs/rl_pct30e5_kl005_full/huggingface_epoch{1,2,3,4,5}/
outputs/rl_pct30e5_kl005_full/eval/epoch{1,2,3,4,5}.json
```

Each raw FSDP checkpoint is about 86 GB and includes optimizer state for exact
resume. Each merged Hugging Face checkpoint is about 15 GB and contains four
safetensor shards. At completion, all training and evaluation processes exited
cleanly and all eight GPUs returned to zero allocated memory.

The failed H100 memory-0.4 attempt is retained for diagnosis as
`train_failed_h100_mem04.log` and `reward_metrics_failed_h100_mem04.jsonl` in
the same output directory. It did not produce a checkpoint and is not included
in the results above.
