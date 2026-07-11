# Vendored third-party code

| Directory | Upstream | Commit | Purpose |
|---|---|---|---|
| `verl` | `verl-project/verl` | `b9d71f9a84ef89ec7f5a946cd277b35165a3daae` | GRPO trainer, FSDP workers, vLLM rollout and checkpoint merger |
| `agent-budget-control` | `EarthRecovery/agent-budget-control` | `23826211b7415260e4a27169f3befb1c496499a0` plus the two config overlays documented in `PROVENANCE.md` | Sokoban rollout generation |

These are source snapshots rather than Git submodules so that the `rebuttal`
branch remains cloneable as one repository. Upstream license and notice files
are retained inside each directory.

The `agent-budget-control/verl` submodule is also expanded at its pinned commit
`d62da4950573d7a4b7ef2362337952e7ab59e78d`. Its unrelated optional environment
submodules (WebShop, Lean, and ToS Base) are not needed by the Sokoban pipeline.
RL training in this repository uses the newer top-level `third_party/verl`, not
the older rollout repository submodule.
