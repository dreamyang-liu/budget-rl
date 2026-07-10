# Artifact provenance

- Source repository: https://github.com/dreamyang-liu/budget-rl
- Source commit: `1d2ebfb5a9cc83a931d0293cffc83f3bb38dbd33`
- Budget scripts commit: `4ea61d2084afe0887021566c18b8aea30f83bcf2`
- S3 prefix: `s3://drmyang-training-data-241580540779-us-east-2-an/agent-budget-control-ckpt/`
- S3 region: `us-east-2`
- Artifact timestamp: 2026-04-29

The S3 scripts and the files in the source repository were byte-identical when
this reproduction repository was assembled.

The nested verl snapshot in the source repository is version 0.6.1. It is not
vendored here because the active workspace uses verl 0.8.0.dev. The training
entry point has been checked against the current configuration keys, but a full
GPU run is still required to establish runtime compatibility.
