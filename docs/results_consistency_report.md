# Results Consistency Report

This report records the release-time consistency checks used to prepare the open-source package.

## Checks Performed

- Counted all benchmark cases on disk.
- Exported a 225-case dataset manifest.
- Compared `cross_scale_summary.json` with paper-table values.
- Extracted medium-scale per-case metrics for the accepted main table.
- Recomputed medium-scale means, sample standard deviations, and population standard deviations.
- Added a strict recheck table computed from the same saved medium-scale artifacts with the released strict evaluator.
- Removed or excluded partial development outputs from curated paper tables.
- Set the released evaluator to strict matching by default and documented the legacy compatibility switch.
- Verified the release Python files with syntax compilation.

## Confirmed Consistencies

- Dataset on disk contains 225 cases: 3 scales x 3 case groups x 25 cases.
- The accepted manuscript's medium-scale main table is recorded in `results/paper_tables/main_table_medium_accepted.csv`.
- PC-GATE and PC-VGAE medium-scale means and standard deviations match the accepted manuscript values when recomputed from `results/paper_tables/main_table_medium_per_case_metrics.csv`.
- Accepted-table means match the extracted per-case metrics. Standard deviations are auditable with the sample and population columns in `main_table_medium_recomputed_stats.csv`.
- `main_table_medium_strict_recheck.csv` provides a conservative strict-matching audit of the same medium-scale saved artifacts.
- The cross-scale PC-GATE and PC-VGAE values match `results/paper_tables/cross_scale_selected_methods.csv`.
- Core release Python files compile successfully.

## Provenance Notes

The release keeps accepted-paper tables and paper-result artifacts separate from recomputed helper tables. This avoids changing accepted results while still making the available artifacts auditable.

- `n_cases` records the number of valid evaluation artifacts used in each aggregate.
- Mean-only tables are provided for quick inspection; accepted manuscript tables and recomputed per-case statistics are provided separately.
- Code identifiers `PCGATE` and `PCVGAE` correspond to the paper labels `PC-GATE` and `PC-VGAE`.
- Legacy development outputs are excluded from the curated paper tables.
- Accepted paper-result tables are archival artifacts. New reruns use strict matching unless legacy compatibility is explicitly requested.
- The strict recheck table is provided for audit transparency and does not supersede the accepted manuscript table.

## Release Decision

The release keeps exact aggregate values and exact `n_cases` values from available artifacts. No values are imputed or regenerated during packaging.
