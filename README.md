# Budget RL: Reproduction and Centrality-Reward Ablation

This branch contains the code, pinned framework sources, experiment commands,
logs, and per-sample evaluation outputs needed to reproduce the Budget RL
experiments for Sokoban remaining-token estimation.

The main pipeline is:

```text
Sokoban rollout
  -> budget-probe dataset
  -> pct30 SFT checkpoint
  -> 5-epoch GRPO training
  -> Hugging Face checkpoint merge
  -> held-out evaluation
```

The repository also contains the reviewer-requested reward ablation that adds a
midpoint-centrality penalty to the original interval reward.

## Main results

All rows below use seed 42 and the same 380-example held-out test set (190
possible and 190 impossible examples).

| Model | Classification accuracy | Possible recall | Impossible recall | Coverage | Median MRE | Mean MRE | Original reward |
|---|---:|---:|---:|---:|---:|---:|---:|
| Reproduced SFT starter | 91.05% | 95.26% | 86.84% | 37.37% | 53.90% | 63.16% | 0.20995 |
| Original RL reward | 90.00% | 91.58% | 88.42% | 45.79% | 30.41% | 56.29% | 0.26183 |
| Centrality reward, lambda=1.0 | 90.00% | 92.63% | 87.37% | 46.32% | 31.42% | **47.26%** | 0.25712 |
| Centrality reward, lambda=0.5 | 90.00% | 92.63% | 87.37% | 44.74% | 35.95% | 50.10% | 0.25271 |

The original 5-epoch RL result is closely reproduced: the archived final reward
is 0.26371 and this run obtains 0.26183. Classification accuracy is exactly the
same (90.00%), and coverage differs by 1.05 percentage points.

For the centrality ablation, `lambda=1.0` is the more promising variant. It
preserves classification accuracy, slightly improves coverage, reduces mean MRE
by 9.03 percentage points, and reduces mean relative interval width from 60.05%
to 52.03%. It does not improve median MRE, so this single-seed experiment does
not support a claim that every measure of centrality improves.

Detailed reports:

- [Full 5-epoch RL reproduction](RL_FULL_REPRO.md)
- [Centrality-aware reward ablation](CENTRALITY_ABLATION.md)
- [SFT checkpoint evaluation reproduction](EVAL_REPRO.md)
- [One-step end-to-end smoke test](SMOKE_TEST.md)

## Reward definitions

Let the true remaining token budget be `y`, the predicted interval be `[L,H]`,
its width be `w = H-L`, and its midpoint be `m = (L+H)/2`.

### Original reward

For a possible example:

```text
R_original = 1.8 * I[L <= y <= H] * max(0, 1 - w/y)
```

For an impossible example, a correct `impossible` prediction receives 0.2.
Invalid formats, scalar predictions, incorrect feasibility predictions, and
uncovered intervals receive zero.

The original reward scores intervals only by coverage and width. Two covered
intervals with the same width receive the same reward even if one midpoint is
closer to the true value.

### Centrality-aware reward

The ablation adds midpoint relative error:

```text
R_lambda = 1.8 * I[L <= y <= H]
           * max(0, 1 - w/y - lambda * abs(m-y)/y)
```

`lambda=0` is exactly the original reward. The main centrality experiment uses
`lambda=1.0`; `lambda=0.5` is a sensitivity run. The impossible-label and format
rules remain unchanged.

The implementation is in
[`reward/budget_probe_reward.py`](reward/budget_probe_reward.py), controlled by
the `CENTRALITY_LAMBDA` environment variable.

## Repository contents

```text
reward/
  budget_probe_reward.py        Original + parameterized centrality reward

scripts/
  download_s3.sh                Download prepared data and SFT starter
  preflight.py                  Validate data, model, CUDA, and vendored verl
  train_rl.sh                   GRPO training entry point
  merge_hf.sh                   Merge FSDP shards into an HF checkpoint
  eval_checkpoint.py            Standalone held-out evaluation
  analyze_eval.py               Evaluation report formatter
  prepare_budget_probe.py       Build SFT/RL budget-probe datasets
  convert_estimation_dialogues.py

third_party/verl/               Exact verl source used by the H100 runs
third_party/agent-budget-control/
                                Sokoban rollout-generation source
reference/original_budget_rl/   Exact original budget-rl source snapshot
environment/repro-versions.txt  Versions in the successful environment
outputs/                        Lightweight logs and per-sample results
```

The branch vendors source snapshots instead of Git submodules. A normal clone
therefore contains the complete code path without additional repository clones.

## Pinned source versions

| Component | Commit |
|---|---|
| `dreamyang-liu/budget-rl` original source | `1d2ebfb5a9cc83a931d0293cffc83f3bb38dbd33` |
| Active `verl-project/verl` | `b9d71f9a84ef89ec7f5a946cd277b35165a3daae` |
| `EarthRecovery/agent-budget-control` | `23826211b7415260e4a27169f3befb1c496499a0` |
| agent-budget-control's historical verl submodule | `d62da4950573d7a4b7ef2362337952e7ab59e78d` |

See [PROVENANCE.md](PROVENANCE.md) for artifact and configuration provenance.

## Hardware and software

The archived experiment used 8 H200 GPUs. The reproduction and centrality runs
in this branch used 8 H100 80 GB GPUs.

Important runtime versions:

```text
Python          3.12
PyTorch         2.9.0+cu129
flash-attn      2.8.1
vLLM            0.12.0
Ray             2.52.1
Transformers    4.57.3
Datasets        4.4.2
Hydra           1.3.2
```

The complete audit list is in
[`environment/repro-versions.txt`](environment/repro-versions.txt). Vendoring
the Python source does not replace platform-specific CUDA, PyTorch, flash-attn,
or vLLM installations.

## Quick reproduction

### 1. Clone the rebuttal branch

```bash
git clone --branch rebuttal --single-branch \
  https://github.com/dreamyang-liu/budget-rl.git
cd budget-rl
```

No separate verl or agent-budget-control clone is required. Training scripts
default to `third_party/verl`. `VERL_ROOT` may be set to override it.

### 2. Configure AWS access

The prepared data and SFT starter are stored at:

```text
s3://drmyang-training-data-241580540779-us-east-2-an/agent-budget-control-ckpt/
```

Verify access:

```bash
aws sts get-caller-identity
```

### 3. Download data and the SFT starter

```bash
bash scripts/download_s3.sh all
```

This downloads:

```text
artifacts/data/rl/train.parquet
artifacts/data/rl/test.parquet
artifacts/data/eval_test/train.parquet
artifacts/data/splits/
artifacts/models/sft_interval_pct30_e5/
```

The prepared data are small; the SFT starter is approximately 14.2 GiB.

Data statistics:

- 966 raw RL training probes: 483 possible and 483 impossible;
- 963 probes remain after the 8,192-token prompt filter;
- 380 held-out probes: 190 possible and 190 impossible.

### 4. Run preflight checks

```bash
python3 scripts/preflight.py
python3 -m unittest discover -s tests -v
DRY_RUN=1 bash scripts/train_rl.sh
```

Preflight checks the parquet schema, class balance, model shard index, CUDA
visibility, and that Python imports verl from this repository.

### 5. Reproduce the original 5-epoch RL run

The H100-stable configuration is:

```bash
CENTRALITY_LAMBDA=0 \
SEED=42 \
EXPERIMENT_NAME=rl_pct30e5_kl005_full \
OUTPUT_DIR="$PWD/outputs/rl_pct30e5_kl005_full" \
NGPUS=8 \
NNODES=1 \
TP_SIZE=4 \
TRAIN_BATCH_SIZE=64 \
PPO_MINI_BATCH_SIZE=64 \
PPO_MICRO_BATCH_SIZE=2 \
ROLLOUT_N=16 \
LR=5e-7 \
KL_COEF=0.05 \
TOTAL_EPOCHS=5 \
SAVE_FREQ=15 \
TEST_FREQ=-1 \
RESUME_MODE=disable \
ROLLOUT_GPU_MEMORY_UTILIZATION=0.3 \
bash scripts/train_rl.sh trainer.val_before_train=False
```

This produces 75 optimizer steps, 15 per epoch, and saves checkpoints at steps
15, 30, 45, 60, and 75. The measured wall time was 2:12:54.

The archived H200 recipe used PPO micro-batch 4 and vLLM memory utilization
0.4. That combination OOMed on H100 80 GB while vLLM remapped weights after the
first update. Micro-batch 2 and utilization 0.3 leave the effective train batch,
rollout count, optimizer, reward, and number of updates unchanged.

### 6. Run the centrality ablations

For `lambda=1.0`:

```bash
CENTRALITY_LAMBDA=1.0 \
SEED=42 \
EXPERIMENT_NAME=rl_pct30e5_kl005_central_lam1_seed42 \
OUTPUT_DIR="$PWD/outputs/rl_pct30e5_kl005_central_lam1_seed42" \
NGPUS=8 TP_SIZE=4 \
TRAIN_BATCH_SIZE=64 PPO_MINI_BATCH_SIZE=64 PPO_MICRO_BATCH_SIZE=2 \
ROLLOUT_N=16 LR=5e-7 KL_COEF=0.05 TOTAL_EPOCHS=5 \
SAVE_FREQ=15 TEST_FREQ=-1 RESUME_MODE=disable \
ROLLOUT_GPU_MEMORY_UTILIZATION=0.3 \
bash scripts/train_rl.sh trainer.val_before_train=False
```

For `lambda=0.5`, change `CENTRALITY_LAMBDA`, `EXPERIMENT_NAME`, and
`OUTPUT_DIR`:

```bash
CENTRALITY_LAMBDA=0.5 \
SEED=42 \
EXPERIMENT_NAME=rl_pct30e5_kl005_central_lam05_seed42 \
OUTPUT_DIR="$PWD/outputs/rl_pct30e5_kl005_central_lam05_seed42" \
NGPUS=8 TP_SIZE=4 \
TRAIN_BATCH_SIZE=64 PPO_MINI_BATCH_SIZE=64 PPO_MICRO_BATCH_SIZE=2 \
ROLLOUT_N=16 LR=5e-7 KL_COEF=0.05 TOTAL_EPOCHS=5 \
SAVE_FREQ=15 TEST_FREQ=-1 RESUME_MODE=disable \
ROLLOUT_GPU_MEMORY_UTILIZATION=0.3 \
bash scripts/train_rl.sh trainer.val_before_train=False
```

Each centrality run took approximately 2 hours 13 minutes on 8 H100 GPUs.

### 7. Merge an FSDP checkpoint

`merge_hf.sh` reads `latest_checkpointed_iteration.txt` and merges the latest
actor checkpoint:

```bash
OUTPUT_DIR="$PWD/outputs/rl_pct30e5_kl005_central_lam1_seed42" \
TARGET_DIR="$PWD/outputs/rl_pct30e5_kl005_central_lam1_seed42/huggingface_final" \
bash scripts/merge_hf.sh
```

The merged 24B checkpoint is approximately 15 GB in four safetensor shards.

### 8. Evaluate a merged checkpoint

Start a vLLM OpenAI-compatible server:

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m vllm.entrypoints.openai.api_server \
  --model outputs/rl_pct30e5_kl005_central_lam1_seed42/huggingface_final \
  --served-model-name central-lam1 \
  --host 127.0.0.1 \
  --port 8765 \
  --dtype bfloat16 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.6
```

In another shell:

```bash
python3 scripts/eval_checkpoint.py \
  --vllm-url http://127.0.0.1:8765 \
  --model-name central-lam1 \
  --test-parquet artifacts/data/eval_test/train.parquet \
  --output outputs/rl_pct30e5_kl005_central_lam1_seed42/eval.json \
  --max-tokens 512 \
  --temperature 0 \
  --max-concurrency 8
```

Every reported evaluation contains 380 responses, has a 100% format-valid
rate, and contains no failed API requests.

## Rebuilding the prepared dataset

The published S3 parquet files are sufficient to reproduce RL training. To
rebuild them from rollout dialogues, use:

```text
third_party/agent-budget-control/       generate Sokoban dialogues
scripts/convert_estimation_dialogues.py convert dialogues to JSONL
scripts/prepare_budget_probe.py         split and emit SFT/RL parquet
```

The exact original scripts and their historical verl snapshot are retained in
`reference/original_budget_rl/`. The active RL runs use the newer vendored
`third_party/verl/` snapshot documented above.

## Evaluation details and caveats

- `coverage` in `eval_checkpoint.py` uses all 190 possible-ground-truth
  examples as its denominator. The archived prose description instead suggests
  conditioning on numeric interval predictions; those are different metrics.
- MRE is computed on possible examples with numeric interval predictions.
  Median MRE is robust to outliers; mean MRE can be dominated by a small number
  of large relative errors.
- Concurrent greedy vLLM decoding is not perfectly bit-deterministic across
  repeated runs. The SFT reproduction changed a small number of predictions
  between runs while preserving the headline metrics.
- The centrality comparison currently has one training seed per reward. Small
  differences in coverage and median MRE should not be treated as conclusive
  without additional seeds or repeated inference.
- The parser intentionally preserves original behavior, including accepting an
  interval embedded inside other text within `<answer>...</answer>`.
- Scalar predictions receive zero reward on possible examples.

## Tracked and external artifacts

Git tracks:

- all source code required by this pipeline;
- pinned third-party source snapshots;
- training and evaluation logs;
- reward batch metrics;
- per-sample evaluation JSON;
- checkpoint iteration manifests.

Git does not track:

- the 14.2 GiB SFT starter;
- raw FSDP checkpoints (about 86 GB per epoch);
- merged Hugging Face weights (about 15 GB per model);
- the approximately 1.38 TB of local model artifacts from all three full runs.

Those files exceed GitHub's 100 MB per-file limit and are obtained from S3 or
regenerated locally. See [`outputs/README.md`](outputs/README.md) for the exact
lightweight outputs committed to this branch.
