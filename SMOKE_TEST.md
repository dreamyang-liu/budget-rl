# One-step GRPO smoke test

Date: 2026-07-10

## Environment

- 8× NVIDIA H100 80GB HBM3
- PyTorch 2.9.0+cu129
- Ray 2.52.1
- vLLM 0.12.0
- verl 0.8.0.dev workspace checkout

## Command profile

The real SFT pct30 e5 checkpoint, parquet data, vLLM rollout, strict reward,
reference log-prob, GRPO advantage, FSDP backward pass, and actor update were
used. The smoke-only reductions were:

- train batch: 8
- rollout N: 2
- PPO mini batch: 8
- PPO micro batch per GPU: 1
- total training steps: 1
- validation and checkpoint saving disabled

## Result

The training step completed successfully without OOM.

- usable train rows after 8192-token filtering: 963/966
- usable validation rows after filtering: 379/380
- reward mean/min/max: 0.0625 / 0.0 / 0.2
- actor loss: 0.0187064
- actor grad norm: 2.74827
- actor learning rate: 5e-7
- response length mean/min/max: 11.8125 / 9 / 16
- prompt length mean/min/max: 1473.75 / 561 / 3958
- total processed tokens: 23,769
- optimizer step time: 43.48 seconds
- throughput: 68.33 tokens/second
- peak observed GPU memory was below 16 GiB per GPU during initialization and
  no OOM occurred

The run reached `Training Progress: 100% (1/1)` and emitted final step metrics.
During process teardown, Python multiprocessing resource-tracker messages and a
killed DataLoader-worker warning were emitted after the completed step. Ray and
vLLM exited, all GPU memory was released, and no residual processes remained.

## Non-fatal warnings observed

- ANTLR runtime/generated version mismatch (4.7.2 vs 4.9.3)
- Transformers tokenizer regex warning
- FlashAttention dtype warning during pre-FSDP model construction
- deprecated FSDP state-dict API warning
- multiprocessing resource-tracker warnings during vLLM teardown
