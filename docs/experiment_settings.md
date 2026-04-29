# Experiment Settings

This file summarizes the released experiment settings used by the public code.

## Data Layout

The full benchmark should be downloaded from:

<https://huggingface.co/datasets/Yukko-kurisu/PCCA>

After extraction, the default layout expected by the pipeline is:

```text
data/full_benchmark/
|-- smallcase/
|-- mediumcase/
`-- bigcase/
```

Each scale directory contains `BCSA`, `CEDA`, and `Mixed` subdirectories.

## Main Pipeline

The released orchestration entry point is:

```bash
python BCSA_11_Complete_Experiment_Pipeline.py --preset main_medium --base-data-dir data/full_benchmark
```

The public presets are:

- `quick_test`: one small Mixed case for smoke testing.
- `main_medium`: medium Mixed benchmark used for the accepted main table.
- `cross_scale_small`: small Mixed benchmark.
- `cross_scale_large`: large Mixed benchmark.
- `eval_only`: aggregate existing result artifacts without rerunning models.

## Fixed Settings

- Random seed: `42`
- PCCA audit methods: `PCGATE`, `PCVGAE`, or explicit `BOTH`
- Default PCCA audit method: `PCGATE`
- Uncertainty aggregation: `weighted_max`
- Monte Carlo samples: `50`
- Baseline `top_k`: `35`
- Default baseline methods: `VGAE`, `GATE`, `Peter-Clark`, `CommonNeighbors`, `GAT`, `CDHC`, `IF-PDAG`, `DistMult`, `GES`, `NOTEARS`, `GOLEM`

Code identifiers `PCGATE` and `PCVGAE` correspond to the paper-table labels `PC-GATE` and `PC-VGAE`.

The output-selection rules are fixed before evaluation: baseline runners report the top-ranked candidate edges with `top_k=35`, while PCCA reports findings that pass its high-priority uncertainty rule. Precision, recall, AUC-PR, and nDCG are computed from the resulting ranked lists.

## Reported Metrics

The paper-result tables use:

- Evidence precision
- Evidence recall
- F1 score
- AUC-PR
- Global nDCG@15
- Global nDCG@25

Metric definitions are documented in `docs/metric_definitions.md`.

## Evaluation Policy

The released evaluator defaults to strict evidence matching. Legacy semantic compatibility matching is disabled unless `--legacy-compatibility-matching` is passed explicitly to `BCSA_03_Quantitative_Evaluator.py`. The result JSON records the active `matching_policy`, whether legacy matching was available, and how many legacy semantic matches were applied.

The `BOTH` audit option runs PC-GATE and PC-VGAE separately and keeps separate metrics. It does not select the better method by ground-truth weighted F1 in the released default workflow.

## Result Tables

Curated paper-result tables are stored under `results/paper_tables/`. The accepted manuscript table is preserved in `main_table_medium_accepted.csv`. The per-case extraction and recomputed statistics are included to make the reported aggregates auditable without rerunning all models. The strict recheck table applies the released strict evaluator to the saved medium-scale artifacts as a conservative audit view.

The release package does not include exploratory ablation, extension, sensitivity, visualization, or auxiliary plotting scripts. Those are not required to run the main PCCA pipeline or inspect the accepted paper-result tables.
