# PCCA: Prompt-Conditioned Causal Auditing

This repository contains the source code, benchmark description, demo case, and paper-result summaries for:

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
|-- results/paper_tables/              Paper-result tables
|-- HUGGINGFACE_DATASET_CARD.md        README content for the dataset repository
|-- docs/                              Experiment settings, metrics, and result notes
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

After running an audit method, evaluate its output directory with:

```bash
python BCSA_03_Quantitative_Evaluator.py <finding_dir> <case_id> --base-data-dir data/full_benchmark
```

## Data

The full benchmark manifest lists 225 industrial CKG audit case directories:

- 3 scales: `smallcase`, `mediumcase`, `bigcase`
- 3 scenario groups: `BCSA`, `CEDA`, `Mixed`
- 25 cases per scale/group combination

The manifest records file availability for each case. See `data/README.md` for the file schema and dataset scope.

The dataset card content for Hugging Face is provided in `HUGGINGFACE_DATASET_CARD.md`.

## Results and Reproducibility

Paper-result tables are in `results/paper_tables/`. `main_table_medium_accepted.csv` records the accepted manuscript main table.

Experiment settings, metric definitions, and result notes are documented under `docs/`.

## Citation

If you use this repository, please cite the TII paper. A machine-readable citation template is provided in `CITATION.cff`.
