#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Baseline runner for the released PCCA experiments."""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import argparse
import json
import logging
import random

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from BCSA_02_Baselines import AVAILABLE_BASELINES
from BCSA_03_Quantitative_Evaluator import calculate_quantitative_metrics


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


DEFAULT_BASELINE_NAMES = [
    "VGAE",
    "GATE",
    "Peter-Clark",
    "CommonNeighbors",
    "GAT",
    "CDHC",
    "IF-PDAG",
    "DistMult",
    "GES",
    "NOTEARS",
    "GOLEM",
]


def set_global_determinism(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_baseline_runners(method_names: Optional[Iterable[str]] = None, top_k: int = 35) -> Dict[str, Any]:
    selected = list(method_names) if method_names is not None else list(DEFAULT_BASELINE_NAMES)
    runners: Dict[str, Any] = {}
    for name in selected:
        if name not in AVAILABLE_BASELINES:
            logger.warning("Skipping unavailable baseline: %s", name)
            continue
        runners[name] = AVAILABLE_BASELINES[name](name, top_k=top_k)
    return runners


def get_available_cases(base_data_dir: Path, case_scale: str, case_type: str) -> List[str]:
    cases_dir = base_data_dir / case_scale / case_type
    if not cases_dir.is_dir():
        logger.warning("Case directory does not exist: %s", cases_dir)
        return []

    required_files = ["causal_knowledge_graph.json", "sensor_data.csv"]
    case_ids = [
        case_path.name
        for case_path in cases_dir.iterdir()
        if case_path.is_dir() and all((case_path / file_name).exists() for file_name in required_files)
    ]
    return sorted(case_ids)


def _build_node_mappings(ckg: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    id_to_text: Dict[str, str] = {}
    text_to_id: Dict[str, str] = {}
    for nodes in ckg.get("nodes_by_type", {}).values():
        if not isinstance(nodes, list):
            continue
        for node in nodes:
            if not isinstance(node, dict):
                continue
            node_id = node.get("id", "")
            node_text = node.get("text", "")
            if node_id and node_text:
                id_to_text[node_id] = node_text
                text_to_id[node_text] = node_id
    return {"id_to_text": id_to_text, "text_to_id": text_to_id}


def evaluate_findings(case_data_dir: str, findings_file: str, case_id: str, output_file: str) -> Dict[str, Any]:
    case_path = Path(case_data_dir)
    output_path = Path(output_file)
    result: Dict[str, Any]

    try:
        with open(case_path / "processed_ground_truth.json", "r", encoding="utf-8") as f:
            gt_data = json.load(f)
        with open(case_path / "causal_knowledge_graph.json", "r", encoding="utf-8") as f:
            ckg = json.load(f)
        with open(findings_file, "r", encoding="utf-8") as f:
            findings = json.load(f)

        metrics = calculate_quantitative_metrics(
            {
                "gt_data": gt_data,
                "node_mappings": _build_node_mappings(ckg),
                "findings": findings,
                "impact_data": {},
            }
        )
        result = {
            "case_id": case_id,
            "evaluation_timestamp": datetime.now().isoformat(),
            "quantitative_metrics": metrics,
        }
    except Exception as exc:
        logger.exception("Evaluation failed for %s using %s", case_id, findings_file)
        result = {
            "case_id": case_id,
            "evaluation_timestamp": datetime.now().isoformat(),
            "status": "failed",
            "error": str(exc),
            "quantitative_metrics": {},
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    return result


def run_baselines_pipeline(
    case_ids: Optional[List[str]] = None,
    case_scale: str = "smallcase",
    case_type: str = "Mixed",
    base_data_dir: Path = Path("./seek_data_v3_deep_enhanced/cases"),
    output_base_dir: Path = Path("./seek_data_v3_deep_enhanced/results"),
    baselines_to_run: Optional[Dict[str, Any]] = None,
    method_names: Optional[Iterable[str]] = None,
    top_k: int = 35,
) -> Dict[str, Dict[str, Any]]:
    set_global_determinism(42)

    if baselines_to_run is None:
        baselines_to_run = create_baseline_runners(method_names=method_names, top_k=top_k)
    if case_ids is None:
        case_ids = get_available_cases(base_data_dir, case_scale, case_type)
    if not case_ids:
        logger.error("No available cases for %s/%s", case_scale, case_type)
        return {}
    if not baselines_to_run:
        logger.error("No baseline methods selected")
        return {}

    output_base_dir.mkdir(parents=True, exist_ok=True)
    all_results: Dict[str, Dict[str, Any]] = {}

    for case_id in case_ids:
        case_data_dir = base_data_dir / case_scale / case_type / case_id
        case_results: Dict[str, Any] = {}
        if not case_data_dir.exists():
            case_results["_case"] = {"status": "failed", "error": f"missing case directory: {case_data_dir}"}
            all_results[case_id] = case_results
            continue

        for baseline_name, baseline_runner in baselines_to_run.items():
            method_output_dir = output_base_dir / case_scale / case_type / case_id / f"{baseline_name}_Analysis"
            try:
                method_output_dir.mkdir(parents=True, exist_ok=True)
                findings_file = baseline_runner.run(case_data_dir, method_output_dir)
                evaluation_file = method_output_dir / "evaluation_results.json"
                evaluation_result = evaluate_findings(
                    case_data_dir=str(case_data_dir),
                    findings_file=str(findings_file),
                    case_id=case_id,
                    output_file=str(evaluation_file),
                )
                case_results[baseline_name] = {
                    "method": baseline_name,
                    "case_id": case_id,
                    "findings_file": str(findings_file),
                    "evaluation_file": str(evaluation_file),
                    "evaluation_result": evaluation_result,
                    "status": "success",
                }
            except Exception as exc:
                logger.exception("Baseline %s failed on case %s", baseline_name, case_id)
                case_results[baseline_name] = {
                    "method": baseline_name,
                    "case_id": case_id,
                    "status": "failed",
                    "error": str(exc),
                }

        all_results[case_id] = case_results

    summary_file = output_base_dir / f"baselines_pipeline_summary_{case_scale}_{case_type}.json"
    summary = {
        "timestamp": datetime.now().isoformat(),
        "case_scale": case_scale,
        "case_type": case_type,
        "processed_cases": len(case_ids),
        "baselines_used": list(baselines_to_run.keys()),
        "results": all_results,
    }
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    successful = sum(
        1
        for case_result in all_results.values()
        for method_result in case_result.values()
        if method_result.get("status") == "success"
    )
    failed = sum(
        1
        for case_result in all_results.values()
        for method_result in case_result.values()
        if method_result.get("status") != "success"
    )
    logger.info("Baseline pipeline finished: %s successful, %s failed", successful, failed)
    logger.info("Summary saved to %s", summary_file)
    return all_results


def _parse_csv_arg(value: Optional[str]) -> Optional[List[str]]:
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run released PCCA baseline methods")
    parser.add_argument("--scale", default="smallcase", choices=["smallcase", "mediumcase", "bigcase"])
    parser.add_argument("--type", default="Mixed", choices=["Mixed", "BCSA", "CEDA"])
    parser.add_argument("--case-ids", help="Comma-separated case ids")
    parser.add_argument("--case-limit", type=int)
    parser.add_argument("--methods", help="Comma-separated baseline names")
    parser.add_argument("--top-k", type=int, default=35)
    parser.add_argument("--base-data-dir", default="./seek_data_v3_deep_enhanced/cases")
    parser.add_argument("--output-dir", default="./seek_data_v3_deep_enhanced/results")
    parser.add_argument("--list-methods", action="store_true")
    args = parser.parse_args()

    if args.list_methods:
        print("\n".join(sorted(AVAILABLE_BASELINES)))
        return 0

    case_ids = _parse_csv_arg(args.case_ids)
    if case_ids is not None and args.case_limit:
        case_ids = case_ids[: args.case_limit]

    results = run_baselines_pipeline(
        case_ids=case_ids,
        case_scale=args.scale,
        case_type=args.type,
        base_data_dir=Path(args.base_data_dir),
        output_base_dir=Path(args.output_dir),
        method_names=_parse_csv_arg(args.methods),
        top_k=args.top_k,
    )
    print(json.dumps({"processed_cases": len(results)}, ensure_ascii=False))
    return 0 if results else 1


if __name__ == "__main__":
    raise SystemExit(main())
