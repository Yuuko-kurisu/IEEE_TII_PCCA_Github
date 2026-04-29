# Dataset Description

The benchmark is a curated semi-synthetic industrial causal knowledge graph audit dataset. It is designed to preserve realistic industrial structure while avoiding direct disclosure of private production records.

## Included Data

This git-ready release includes one demo case:

```text
data/demo/Mixed_small_01/
```

The complete benchmark is hosted on Hugging Face:

```text
https://huggingface.co/datasets/Yukko-kurisu/PCCA
```

Download and extract it with:

```bash
pip install huggingface_hub
huggingface-cli download Yukko-kurisu/PCCA pcca_full_benchmark_cases.zip --repo-type dataset --local-dir data
python -m zipfile -e data/pcca_full_benchmark_cases.zip data/full_benchmark
```

When extracted, the full dataset contains only the benchmark case tree. Local analysis folders, intermediate experiment workspaces, and development logs are not part of the public dataset archive. The archive is about 520 MB. The case-level manifest in this repository is kept as `benchmark_manifest.csv`.

## Full Benchmark Scale

The full benchmark manifest lists 225 case directories, including 224 complete evaluation cases with CKG and ground-truth files:

- `smallcase`: 75 cases
- `mediumcase`: 75 cases
- `bigcase`: 75 cases
- each scale has `BCSA`, `CEDA`, and `Mixed` subsets with 25 cases each

See `benchmark_manifest.csv` for case-level file availability, file counts, and sizes. One retained sensor-only directory is marked in the manifest and is excluded by evaluation filters. For evaluation runs, filter to cases with `causal_knowledge_graph.json`, `processed_ground_truth.json`, and result findings available. The expected extracted layout is:

```text
data/full_benchmark/
|-- bigcase/
|-- mediumcase/
`-- smallcase/
```

## Case Schema

Each case may contain:

- `causal_knowledge_graph.json`: initial CKG nodes, edges, mappings, and blind-spot markers
- `case_data.json`: scenario and case metadata
- `sensor_data.csv`: time-series sensor observations
- `fault_tickets.json`: maintenance-ticket style records
- `expert_summary.json`: document coverage summary
- `ground_truth.json`: raw ground-truth audit targets
- `processed_ground_truth.json`: evaluation-ready evidence zones and target edges
- `expert_knowledge/`: case documents in text, markdown, and JSON forms

## Privacy and Scope

The released cases are curated semi-synthetic benchmark cases. They do not contain real personal emails, phone numbers, passwords, or raw partner production logs. The dataset is intended for method verification, reproduction of paper tables, and development of industrial CKG audit methods.
