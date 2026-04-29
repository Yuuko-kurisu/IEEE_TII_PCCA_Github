---
license: mit
task_categories:
  - graph-machine-learning
  - tabular-classification
  - text-classification
language:
  - en
tags:
  - causal-discovery
  - knowledge-graph
  - industrial-ai
  - audit
  - graph-neural-networks
pretty_name: PCCA Industrial Causal Knowledge Graph Audit Benchmark
size_categories:
  - 100K<n<1M
---

# PCCA Industrial Causal Knowledge Graph Audit Benchmark

This dataset accompanies the paper:

> PCCA: A Hypothesis-Driven Prompt-Conditioned Causal Auditing Framework for Industrial Knowledge Graphs

The benchmark contains curated semi-synthetic industrial causal knowledge graph audit cases. It is designed for evaluating methods that identify missing, unreliable, or under-supported causal relationships in industrial causal knowledge graphs.

## Dataset Summary

- Case directories: 225
- Complete evaluation cases with CKG and ground-truth files: 224
- Scales: `smallcase`, `mediumcase`, `bigcase`
- Scenario groups: `BCSA`, `CEDA`, `Mixed`
- Case directories per scale/group pair: 25
- Included modalities: causal knowledge graphs, sensor time series, fault tickets, expert-knowledge documents, and ground-truth audit targets
- Demo case in the code repository: `Mixed_small_01`

## Repository

Code, documentation, demo data, and paper-result tables are available at:

<https://github.com/Yuuko-kurisu/IEEE_TII_PCCA_Github>

## Download

```bash
pip install huggingface_hub
huggingface-cli download Yukko-kurisu/PCCA pcca_full_benchmark_cases.zip --repo-type dataset --local-dir data
python -m zipfile -e data/pcca_full_benchmark_cases.zip data/full_benchmark
```

Expected extracted layout:

```text
data/full_benchmark/
|-- smallcase/
|   |-- BCSA/
|   |-- CEDA/
|   `-- Mixed/
|-- mediumcase/
`-- bigcase/
```

The public archive contains the benchmark case tree only. Local analysis folders,
intermediate experiment workspaces, and development logs are intentionally
excluded from the dataset release.

## Case Schema

Each case directory may contain:

- `causal_knowledge_graph.json`: initial causal knowledge graph nodes, edges, mappings, and blind-spot markers
- `case_data.json`: case metadata and scenario description
- `sensor_data.csv`: sensor time-series observations
- `fault_tickets.json`: maintenance-ticket style records
- `expert_summary.json`: summary of case documents
- `ground_truth.json`: raw ground-truth audit targets
- `processed_ground_truth.json`: evaluation-ready evidence zones and target edges
- `expert_knowledge/`: expert documents in text, markdown, and JSON forms

The code repository includes `data/benchmark_manifest.csv`, which records case-level file availability. One retained sensor-only directory is marked in the manifest and is excluded by evaluation filters. Evaluation runs should use cases with the required CKG and processed ground-truth files available.

## Intended Use

The dataset is intended for:

- Reproducing and auditing the PCCA paper-result tables
- Evaluating industrial causal knowledge graph audit methods
- Testing graph neural network, causal discovery, and knowledge graph completion baselines on a common benchmark

## Privacy and Provenance

The benchmark is semi-synthetic. It does not contain real personal emails, phone numbers, passwords, or raw partner production logs. The benchmark design is intended to preserve realistic audit patterns without disclosing private operational records.

## Limitations

The cases are designed for method verification and academic benchmarking. They should not be treated as direct production data or as a substitute for domain-specific industrial validation.

## Citation

If you use this dataset, please cite the associated IEEE TII paper. A machine-readable citation file is included in the code repository as `CITATION.cff`.
