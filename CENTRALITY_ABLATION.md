# Centrality-aware reward ablation

Date: 2026-07-11

## Setup

This single-seed exploratory ablation compares the reproduced original reward
against two midpoint-centrality penalties. For a covered interval `[L,H]`, true
remaining budget `y`, width `w=H-L`, and midpoint `m=(L+H)/2`:

```text
R_lambda = 1.8 * max(0, 1 - w/y - lambda * abs(m-y)/y)
```

An uncovered interval still receives zero. Impossible-label scoring and all
format rules are unchanged. `lambda=0` is exactly the original reward.

All three runs use seed 42, the same SFT starter, data, 8 H100 GPUs, batch 64,
rollout N=16, learning rate 5e-7, KL coefficient 0.05, and 5 epochs. The
Original row reuses the completed reproduction; the two centrality runs each
completed 75 steps in about 2 hours 13 minutes.

## Common evaluation

All models were evaluated on the same 380 rows using vLLM greedy decoding and
the archived evaluation script. The reported mean reward below is the original
reward (`lambda=0`) for every model, so it is directly comparable.

| Model | Accuracy | Recall possible | Recall impossible | Coverage | Median MRE | Mean MRE | Mean reward |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original | 90.00% | 91.58% | 88.42% | 45.79% | 30.41% | 56.29% | 0.26183 |
| Central, lambda=1.0 | 90.00% | 92.63% | 87.37% | 46.32% | 31.42% | 47.26% | 0.25712 |
| Central, lambda=0.5 | 90.00% | 92.63% | 87.37% | 44.74% | 35.95% | 50.10% | 0.25271 |

Additional interval diagnostics:

| Model | Valid intervals on possible GT | Mean relative width | Median relative width | P90 MRE | Conditional coverage |
|---|---:|---:|---:|---:|---:|
| Original | 174 | 60.05% | 50.68% | 87.17% | 50.00% |
| Central, lambda=1.0 | 176 | 52.03% | 52.94% | 85.92% | 50.00% |
| Central, lambda=0.5 | 176 | 53.67% | 50.45% | 85.97% | 48.30% |

Cross-scoring every set of outputs with each reward definition:

| Model | Reward lambda=0 | Reward lambda=0.5 | Reward lambda=1.0 |
|---|---:|---:|---:|
| Original | 0.26183 | 0.23445 | 0.21073 |
| Central, lambda=1.0 | 0.25712 | 0.23384 | 0.21419 |
| Central, lambda=0.5 | 0.25271 | 0.22683 | 0.20574 |

## Interpretation

`lambda=1.0` is the more promising centrality variant. Relative to Original it:

- preserves classification accuracy at 90.00%;
- raises coverage by 0.53 percentage points;
- reduces mean MRE by 9.03 percentage points and P90 MRE by 1.25 points;
- reduces mean relative interval width by 8.02 percentage points;
- improves the common `lambda=1` cross-score from 0.21073 to 0.21419.

However, it does **not** improve the pre-specified median MRE: median MRE rises
from 30.41% to 31.42%. `lambda=0.5` is weaker, with both lower coverage and
higher median MRE than Original. The correct single-seed conclusion is therefore
that lambda=1 improves mean/tail centrality and produces narrower intervals
without hurting classification, but the evidence does not establish a median
centrality improvement.

The changes in coverage and median MRE are small enough to overlap known vLLM
evaluation nondeterminism. A reviewer-facing confirmatory claim should repeat
Original and lambda=1 across multiple training seeds (or at minimum repeat
inference) rather than treating this exploratory seed as conclusive.

## Artifacts

```text
outputs/rl_pct30e5_kl005_central_lam1_seed42/
  global_step_{15,30,45,60,75}/
  huggingface_final/
  eval.json

outputs/rl_pct30e5_kl005_central_lam05_seed42/
  global_step_{15,30,45,60,75}/
  huggingface_final/
  eval.json
```
