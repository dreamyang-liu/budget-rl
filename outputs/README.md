# Tracked experiment outputs

This directory contains the lightweight outputs needed to inspect and reproduce
the reported metrics:

- per-sample evaluation JSON;
- training console logs;
- reward batch metrics;
- final checkpoint iteration manifests; and
- vLLM evaluation server logs for the centrality runs.

Raw FSDP checkpoints and merged Hugging Face model weights are intentionally not
tracked. The local artifacts are approximately 440 GB per centrality run and
497 GB for the original five-epoch run, and contain individual files larger
than GitHub's 100 MB file limit.

The centrality comparison uses:

```text
rl_pct30e5_kl005_full/eval/epoch5.json
rl_pct30e5_kl005_central_lam1_seed42/eval.json
rl_pct30e5_kl005_central_lam05_seed42/eval.json
```
