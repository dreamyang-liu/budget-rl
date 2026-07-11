# Artifact provenance

- Source repository: https://github.com/dreamyang-liu/budget-rl
- Source commit: `1d2ebfb5a9cc83a931d0293cffc83f3bb38dbd33`
- Budget scripts commit: `4ea61d2084afe0887021566c18b8aea30f83bcf2`
- Active verl repository: https://github.com/verl-project/verl
- Active verl commit: `b9d71f9a84ef89ec7f5a946cd277b35165a3daae`
- Rollout repository: https://github.com/EarthRecovery/agent-budget-control
- Rollout repository commit: `23826211b7415260e4a27169f3befb1c496499a0`
- S3 prefix: `s3://drmyang-training-data-241580540779-us-east-2-an/agent-budget-control-ckpt/`
- S3 region: `us-east-2`
- Artifact timestamp: 2026-04-29

The S3 scripts and the files in the source repository were byte-identical when
this reproduction repository was assembled.

The source repository's old nested verl snapshot is version 0.6.1. This branch
instead vendors the exact newer verl tree used by the successful H100 runs under
`third_party/verl`. No tracked files below the vendored `verl/` package differed
from the upstream commit during the experiments.

The complete tracked `agent-budget-control` tree is vendored under
`third_party/agent-budget-control`. Two local configuration additions used by
the rollout environment are overlaid on its base commit: OpenRouter Qwen model
entries in `config/evaluate_api_llm.yaml` and the Qwen3-8B dispatch case in
`scripts/evaluation-scripts/eval/sokoban.sh`. Neither changes RL reward or
optimization code.

Its verl submodule is expanded at the upstream-pinned commit
`d62da4950573d7a4b7ef2362337952e7ab59e78d`; it is retained for historical
rollout compatibility and is not used by the active GRPO entry point.

The exact original top-level budget scripts are preserved under
`reference/original_budget_rl`. Active copies of evaluation and data-preparation
scripts live under `scripts/`.
