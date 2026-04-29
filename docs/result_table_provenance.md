# Result Table Provenance

This note explains the curated result tables included in `results/paper_tables/`.

## Files

- `main_table_medium_accepted.csv`: accepted manuscript main-table values for the medium-scale benchmark.
- `main_table_medium_per_case_metrics.csv`: per-case metrics extracted from available medium-scale evaluation artifacts.
- `main_table_medium_recomputed_stats.csv`: means, sample standard deviations, and population standard deviations recomputed from the per-case metrics.
- `main_table_medium_strict_recheck.csv`: conservative aggregate table computed by applying the released strict evaluator to the saved medium-scale artifacts.
- `main_table_medium_strict_recheck_per_case.csv`: per-case metrics behind the strict recheck table.
- `main_medium_results.csv`: mean-only medium-scale summary exported from the cross-scale aggregate.
- `cross_scale_selected_methods.csv`: cross-scale summary for the selected methods.

## Reproducibility Notes

The accepted manuscript reports the medium-scale benchmark. The release keeps the accepted table unchanged and also provides per-case metrics so readers can independently recompute statistics.

The table values for PC-GATE and PC-VGAE match the accepted manuscript when recomputed from the extracted per-case metrics. For all accepted-table rows, the means match the extracted per-case metrics; the reported standard deviations correspond to either the sample or population convention depending on the source artifact. The `main_table_medium_recomputed_stats.csv` file includes both conventions so that readers can audit the values without changing the accepted manuscript table.

For cross-scale summaries, `n_cases` records the number of valid evaluation artifacts used for each method/scale aggregate.

The public evaluator now defaults to strict evidence matching. Older artifacts can be inspected with the explicit legacy matching switch, but the accepted table files are kept as immutable paper-result records rather than silently regenerated under a different policy.

The strict recheck table uses the same case/method artifact set as the accepted medium table. It is included as an audit aid for readers who want a conservative matching policy. It does not replace the accepted manuscript table.

Practical interpretation:

- Use `main_table_medium_accepted.csv` when citing or checking the accepted manuscript values.
- Use the default strict evaluator for new public reruns.
- Use `main_table_medium_strict_recheck.csv` when auditing saved artifacts under the conservative matching policy.
- Use `--legacy-compatibility-matching` only when inspecting historical artifacts that were produced under the earlier compatibility policy.
