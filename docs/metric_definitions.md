# Metric Definitions

PCCA is evaluated as a ranked blind-spot localization task.

## Localization Metrics

- Precision: fraction of reported findings that match ground-truth evidence.
- Recall: fraction of ground-truth evidence recovered by the findings.
- F1-score: harmonic mean of precision and recall.
- AUC-PR: area under the precision-recall curve computed from ranked findings.

## Ranking Metrics

NDCG@K measures whether high-value evidence appears near the top of the ranked finding list. In the released strict evaluator, the relevance score is based on evidence importance:

- evidence-importance gain: `high=3`, `medium=2`, `low=1`, `unknown=0`

The legacy compatibility mode can additionally include the matched finding's unified score:

```text
relevance = 1.0 * evidence_importance_gain + 20.0 * model_ranking_signal
```

This legacy mode is disabled by default and is exposed only through `--legacy-compatibility-matching` for inspecting older result artifacts. New public runs should use strict matching.

## Diagnostic Metrics

Some scripts also report diagnostic measures such as weighted F1, evidence MRR, blind-spot recall, and zone evidence coverage rate. These are useful for analysis but the main paper tables focus on Precision, Recall, F1-score, AUC-PR, NDCG@15, and NDCG@25.
