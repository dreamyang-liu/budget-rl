# Budget RL Training Reproduction

这个仓库用于复现 Sokoban budget estimation 的 **rollout → data → SFT
checkpoint → GRPO → evaluation** 代码链。默认训练入口从实验中表现最好的
`sft_interval_pct30/huggingface_e5` 开始；原始数据和模型权重通过 S3 获取。

## 已固定的实验配置

- Base model: Qwen2.5-7B-Instruct
- Starter checkpoint: `sft_interval_pct30/huggingface_e5`
- Data: 966 raw train probes（483 possible + 483 impossible）；其中 963 条通过
  `max_prompt_length=8192` 过滤并实际参与训练
- Validation: 380 probes（190 possible + 190 impossible）
- Algorithm: GRPO, no critic
- GPUs: 8×H200（原实验环境）；当前机器为 8×H100 80GB
- Train batch size: 64
- Rollout samples: 16
- Tensor parallel size: 4
- Learning rate: `5e-7`
- KL coefficient: `0.05`
- Epochs: 5
- Reward: strict interval coverage

## 目录

```text
reward/                 实验使用的 reward function
third_party/verl/       实际训练使用的完整 verl 源码快照
third_party/agent-budget-control/  Sokoban rollout 生成源码
reference/original_budget_rl/      原始 budget-rl 脚本快照
scripts/download_s3.sh  下载训练数据或 starter checkpoint
scripts/preflight.py    数据、环境和 GPU 检查
scripts/train_rl.sh     GRPO 训练入口
scripts/merge_hf.sh     将最终 FSDP checkpoint 转为 HF 格式
scripts/eval_checkpoint.py  standalone checkpoint evaluation
scripts/prepare_budget_probe.py  probe dataset preparation
tests/test_reward.py    reward 行为测试
environment/            实际运行环境的版本记录
artifacts/              下载的数据和模型（不纳入 Git）
outputs/                轻量结果纳入 Git；大型 checkpoint 不纳入
```

已完成的单步端到端验证结果见 [SMOKE_TEST.md](SMOKE_TEST.md)。
下载 checkpoint 的评测复现结果见 [EVAL_REPRO.md](EVAL_REPRO.md)。
完整 5 epoch RL 训练和逐 epoch 评测结果见
[RL_FULL_REPRO.md](RL_FULL_REPRO.md)。
Centrality-aware reward 的单 seed ablation 见
[CENTRALITY_ABLATION.md](CENTRALITY_ABLATION.md)。

## 1. 前置条件

Clone 后无需另外 clone verl 或 agent-budget-control；两者的实际源码版本已
vendor 在 `third_party/`。默认训练入口使用：

```bash
export VERL_ROOT="$PWD/third_party/verl"
```

通常无需显式设置该变量。运行环境使用 Python 3.12、PyTorch 2.9、vLLM
0.12 和 Ray 2.52.1；完整版本见
[`environment/repro-versions.txt`](environment/repro-versions.txt)。CUDA、
PyTorch、flash-attn 和 vLLM 仍需针对目标 GPU 正确安装，vendor 源码不会
替代这些编译依赖。

AWS CLI 需要能读取：

```text
s3://drmyang-training-data-241580540779-us-east-2-an/agent-budget-control-ckpt/
```

## 2. 下载数据

```bash
bash scripts/download_s3.sh data
```

约下载 2 MB，包括：

- `artifacts/data/rl/train.parquet`
- `artifacts/data/rl/test.parquet`
- `artifacts/data/eval_test/train.parquet`
- trajectory split manifests

## 3. 下载 SFT starter checkpoint

```bash
bash scripts/download_s3.sh starter
```

这一步约下载 14.2 GiB。

也可以使用已经存在的本地 checkpoint：

```bash
export MODEL=/path/to/huggingface_checkpoint
```

## 4. 预检

```bash
python3 scripts/preflight.py
python3 -m unittest discover -s tests -v
DRY_RUN=1 bash scripts/train_rl.sh
```

预检会检查 parquet schema、类别平衡、trajectory split、verl import、CUDA 和 GPU 数量。

## 5. 启动训练

```bash
bash scripts/train_rl.sh
```

训练输出默认写入：

```text
outputs/rl_pct30e5_kl005/
```

所有关键参数都可以通过环境变量修改，例如做小规模 smoke run：

```bash
NGPUS=2 \
TP_SIZE=1 \
TRAIN_BATCH_SIZE=8 \
PPO_MINI_BATCH_SIZE=8 \
PPO_MICRO_BATCH_SIZE=1 \
ROLLOUT_N=2 \
TOTAL_EPOCHS=1 \
SAVE_FREQ=1 \
bash scripts/train_rl.sh trainer.total_training_steps=1
```

这个 smoke 配置只用于验证流水线，不等价于原实验。

当前机器的 H100 显存小于原实验的 H200。如果发生 OOM，优先降低
`PPO_MICRO_BATCH_SIZE` 和 `actor_rollout_ref.rollout.gpu_memory_utilization`，
不要先改变 batch size、rollout N 或 KL 等影响实验定义的参数。

## 6. 转换最终 checkpoint

```bash
bash scripts/merge_hf.sh
```

默认读取最新的 `outputs/rl_pct30e5_kl005/global_step_*`，输出到：

```text
outputs/rl_pct30e5_kl005/huggingface_final/
```

## 7. Evaluation

启动 OpenAI-compatible vLLM server 后，使用仓库内的原始评测脚本：

```bash
python3 scripts/eval_checkpoint.py \
  --vllm-url http://localhost:8765 \
  --model-name budget-rl \
  --test-parquet artifacts/data/eval_test/train.parquet \
  --output outputs/eval.json \
  --max-tokens 512 \
  --temperature 0 \
  --max-concurrency 8
```

生成原始 rollout 的代码也已保留在
`third_party/agent-budget-control/`；从已发布的 parquet 和 SFT starter
复现 RL 时不需要重新生成 rollout。

## 已知注意事项

- 原始 reward parser 会接受 `<answer>` 内夹杂文本的区间；仓库为了复现实验而保留该行为。
- Scalar prediction 在 possible 样本上奖励为 0。
- Reward metrics 每 200 条写一次，进程退出时不足 200 条的尾部不会写入。
- S3 中没有六组 SFT 的 e2 checkpoint，只有 e3/e5。
- S3 中部分旧 eval JSON 的 reward 来自旧 reward 版本，训练复现应以本仓库 reward 为准。
- GitHub 不保存 FSDP/HF 权重：三个完整实验目录本地合计约 1.38 TB，且单个
  权重文件超过 GitHub 100 MB 限制。代码、轻量日志、per-sample eval 和下载
  入口均已包含。
