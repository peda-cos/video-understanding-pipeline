# Evaluation report

Measured locally on the deterministic synthetic suite. No public benchmark claims.

| Metric | Value |
| --- | --- |
| Queries evaluated | 100 / 100 |
| Skipped | 0 |
| Answer accuracy | 97.0% |
| Retrieval scene hit@1 | 100.0% |
| Retrieval scene hit@5 | 100.0% |
| Valid citations | 100.0% |
| Counting error rate | 6.7% |
| Action error rate | 0.0% |
| Grounding mean IoU (action) | 0.734 |
| Grounding IoU@0.3 (action) | 100.0% |
| Grounding IoU@0.5 (action) | 100.0% |
| Ask latency median | 1.795 ms |
| Ask latency p95 | 3.966 ms |

## Per-class accuracy

| Class | n | Accuracy | Hit@1 | Hit@5 |
| --- | ---: | ---: | ---: | ---: |
| counting | 30 | 93.3% | 100.0% | 100.0% |
| action | 30 | 100.0% | 100.0% | 100.0% |
| descriptive | 40 | 97.5% | 100.0% | 100.0% |

Counting and action error rates are reported separately as hallucination proxies;
they are the complement of accuracy on those subsets, not calibrated probabilities.
Grounding IoU is computed only on the action subset with temporal gold windows.
