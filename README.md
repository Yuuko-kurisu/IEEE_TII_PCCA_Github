# PCCA: Prompt-Conditioned Causal Auditing

This repository contains the source code, benchmark data description, demo case, and paper-result summaries for:

> PCCA: A Hypothesis-Driven Prompt-Conditioned Causal Auditing Framework for Industrial Knowledge Graphs

PCCA audits existing industrial causal knowledge graphs (CKGs). It first generates structured audit hypotheses, then injects each hypothesis as a prompt into graph autoencoder backbones, and finally ranks candidate missing or unreliable causal relationships for expert review.

## Repository Contents

```text
.
|-- BCSA_*.py / SEEK_*.py              Core implementation and evaluation modules
|-- data/
|   |-- demo/Mixed_small_01/           Small demo case included in git
|   |-- benchmark_manifest.csv         Manifest for the full 225-directory benchmark
|   `-- dataset_summary.json           Dataset scale/type summary
|-- results/paper_tables/              Curated result tables used for paper checks
|   |-- main_table_medium_accepted.csv Accepted manuscript main-table values
|   |-- main_table_medium_per_case_metrics.csv
|   |-- main_table_medium_recomputed_stats.csv
|   `-- main_table_medium_strict_recheck.csv
|-- HUGGINGFACE_DATASET_CARD.md        README content for the dataset repository
|-- docs/
|   |-- experiment_settings.md
|   |-- metric_definitions.md
|   |-- result_table_provenance.md
|   `-- results_consistency_report.md
`-- requirements.txt                   Python dependency list
```

## Installation

Python 3.10 is recommended.

```bash
pip install -r requirements.txt
```

The graph models use PyTorch and PyTorch Geometric. If your platform requires a specific PyTorch wheel, install `torch` and `torch_geometric` first following their official instructions, then install the remaining dependencies.

## Quick Start

The demo case is included under `data/demo/Mixed_small_01`. The full benchmark is hosted on Hugging Face:

[https://huggingface.co/datasets/Yukko-kurisu/PCCA](https://huggingface.co/datasets/Yukko-kurisu/PCCA)

Download the complete benchmark archive with:

```bash
pip install huggingface_hub
huggingface-cli download Yukko-kurisu/PCCA pcca_full_benchmark_cases.zip --repo-type dataset --local-dir data
python -m zipfile -e data/pcca_full_benchmark_cases.zip data/full_benchmark
```

```bash
python BCSA_11_Complete_Experiment_Pipeline.py --preset quick_test --base-data-dir data/full_benchmark --case-scale smallcase --case-type Mixed --case-ids Mixed_small_01
```

For direct metric evaluation on existing findings:

```bash
python BCSA_03_Quantitative_Evaluator.py results/conditioned_uncertainty_analysis/Mixed_small_01 Mixed_small_01 --base-data-dir data/full_benchmark
```

## Data

The full benchmark manifest lists 225 curated semi-synthetic industrial CKG audit case directories, including 224 complete evaluation cases with CKG and ground-truth files:

- 3 scales: `smallcase`, `mediumcase`, `bigcase`
- 3 scenario groups: `BCSA`, `CEDA`, `Mixed`
- 25 cases per scale/group combination

The manifest records file availability for each case. Reproduction scripts should select cases with the required `causal_knowledge_graph.json` and `processed_ground_truth.json` files. See `data/README.md` for the file schema and privacy notes.

The dataset card content for Hugging Face is provided in `HUGGINGFACE_DATASET_CARD.md`.

## Results and Reproducibility

Curated paper-result tables are in `results/paper_tables/`. `main_table_medium_accepted.csv` records the accepted manuscript main table, while `main_table_medium_per_case_metrics.csv` and `main_table_medium_recomputed_stats.csv` provide the extracted per-case metrics and recomputed statistics used for release checks. `main_table_medium_strict_recheck.csv` is a conservative audit table produced by applying the released strict evaluator to the saved medium-scale artifacts. Cross-scale summaries preserve exact `n_cases` values from the available artifacts.

Additional table provenance notes are documented in `docs/results_consistency_report.md` and `docs/result_table_provenance.md`.
The released experiment presets and default method settings are summarized in `docs/experiment_settings.md`.

The public evaluator uses strict evidence matching by default. A legacy compatibility switch (`--legacy-compatibility-matching`) is available only for inspecting older result artifacts that used semantic compatibility matching; new public runs should keep the strict default.

The result files serve different purposes:

| File | Purpose |
| --- | --- |
| `main_table_medium_accepted.csv` | Accepted manuscript values. Use this file when citing the paper table. |
| `main_table_medium_strict_recheck.csv` | Conservative audit of the same saved medium-scale artifacts under the released strict evaluator. |
| `main_table_medium_per_case_metrics.csv` | Per-case metrics extracted from the accepted medium-scale artifacts. |
| `main_table_medium_recomputed_stats.csv` | Mean/std recomputation helper, including standard-deviation convention checks. |

For new reruns, use the strict default evaluator. For historical artifact inspection, add `--legacy-compatibility-matching` explicitly.

## Citation

If you use this repository, please cite the TII paper. A machine-readable citation template is provided in `CITATION.cff`.
