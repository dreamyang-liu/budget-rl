# SFT pct30 e5 evaluation reproduction

Date: 2026-07-10

The downloaded `sft_interval_pct30_e5` checkpoint was evaluated twice against
the downloaded 380-row held-out parquet file with:

- vLLM 0.12.0 on one H100
- max model length: 16,384
- max generated tokens: 512
- temperature: 0
- request concurrency: 8
- the S3 `eval_checkpoint.py` and strict reward function

## Comparison

| Metric | Published report | Reproduction run 1 | Reproduction run 2 |
|---|---:|---:|---:|
| Format valid | 100.00% | 100.00% | 100.00% |
| Classification accuracy | 91.05% | 90.79% | 91.05% |
| Recall possible | 95.26% | 95.26% | 95.26% |
| Recall impossible | 86.84% | 86.32% | 86.84% |
| `cover_rate_possible` | 37.37% | 38.42% | 37.37% |
| Median MRE | 53.90% | 52.78% | 53.90% |
| Mean MRE | 63.89% | 52.23% | 63.16% |
| Mean reward | 0.21043 | 0.21142 | 0.20995 |

Run 2 reproduces every discrete headline metric and median MRE exactly. Mean
MRE differs by 0.73 percentage points and mean reward differs by 0.00048.

The report and run 2 contain the same 380 custom IDs. Eight generated responses
differ from the archived report; none changes the possible/impossible class and
two change coverage/reward. Repeating the same evaluation also changes a small
number of outputs, so the residual difference is consistent with batching and
floating-point nondeterminism in concurrent vLLM greedy decoding rather than a
checkpoint or dataset mismatch.

## Coverage denominator caveat

`eval_checkpoint.py` reports `cover_rate_possible` as covered predictions divided
by all 190 possible-ground-truth rows. In run 2 this is 71/190 = 37.37%.

The prose report defines coverage over numeric interval predictions only. Under
that definition, the value is 71/181 = 39.23%. The published 37.37% number and
the script are internally consistent, but the prose definition uses a different
denominator.
