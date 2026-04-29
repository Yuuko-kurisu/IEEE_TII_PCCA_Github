#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Released PCCA experiment pipeline.

The release pipeline covers the paper-facing workflow only: PCCA audit runs,
baseline runs, and quantitative aggregation. Additional exploratory experiments
used during development are not part of the public execution surface.
"""

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import argparse
import json
import logging
import random
import sys

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - torch is an optional runtime dependency for import checks.
    torch = None

from BCSA_04_audit_pipeline import run_complete_batch_evaluation
from BCSA_04_Baselines_run_pipeline import get_available_cases, run_baselines_pipeline
from BCSA_12_Experiment_Configs import ExperimentConfigManager

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    case_scale: str = "mediumcase"
    case_type: str = "Mixed"
    case_ids: Optional[List[str]] = None
    case_limit: Optional[int] = None
    seed: int = 42
    run_audit_pipeline: bool = True
    run_baselines_pipeline: bool = True
    run_quantitative_eval: bool = True
    audit_method: str = "PCGATE"
    aggregation_method: str = "weighted_max"
    mc_samples: int = 50
    base_data_dir: Path = Path("./seek_data_v3_deep_enhanced/cases")
    output_dir: Path = Path("./seek_data_v3_deep_enhanced/results")
    continue_on_error: bool = True


def set_global_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class CompleteExperimentPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.timestamp = datetime.now().isoformat()
        self.results: Dict[str, Any] = {}
        set_global_determinism(config.seed)

    def get_case_ids(self) -> List[str]:
        if self.config.case_ids:
            case_ids = self.config.case_ids
        else:
            case_ids = get_available_cases(
                base_data_dir=self.config.base_data_dir,
                case_scale=self.config.case_scale,
                case_type=self.config.case_type,
            )
        if self.config.case_limit:
            case_ids = case_ids[: self.config.case_limit]
        return case_ids

    def run_audit(self, case_ids: List[str]) -> Dict[str, Any]:
        if not self.config.run_audit_pipeline:
            return {"status": "skipped"}
        result = run_complete_batch_evaluation(
            case_ids=case_ids,
            case_scale=self.config.case_scale,
            case_type=self.config.case_type,
            base_data_dir=self.config.base_data_dir,
            output_dir=self.config.output_dir,
            method_choice=self.config.audit_method,
            aggregation_method=self.config.aggregation_method,
            mc_samples=self.config.mc_samples,
        )
        self.results["audit"] = result
        return result

    def run_baselines(self, case_ids: List[str]) -> Dict[str, Any]:
        if not self.config.run_baselines_pipeline:
            return {"status": "skipped"}
        result = run_baselines_pipeline(
            case_ids=case_ids,
            case_scale=self.config.case_scale,
            case_type=self.config.case_type,
            base_data_dir=self.config.base_data_dir,
            output_base_dir=self.config.output_dir,
        )
        self.results["baselines"] = result
        return result

    def run_quantitative_summary(self) -> Dict[str, Any]:
        if not self.config.run_quantitative_eval:
            return {"status": "skipped"}
        try:
            from BCSA_05_Unified_Quantitative_Evaluator import run_unified_analysis
        except ImportError as exc:
            return {"status": "skipped", "reason": f"quantitative module unavailable: {exc}"}

        output_dir = self.config.output_dir / "Quantitative_analysis" / f"{self.config.case_scale}_{self.config.case_type}"
        result = run_unified_analysis(
            results_base_dir=self.config.output_dir,
            case_scale=self.config.case_scale,
            case_type=self.config.case_type,
            output_dir=output_dir,
        )
        self.results["quantitative"] = result
        return result

    def run_complete_pipeline(self) -> Dict[str, Any]:
        case_ids = self.get_case_ids()
        if not case_ids:
            return {"status": "failed", "error": "no_available_cases", "config": asdict(self.config)}

        status: Dict[str, Any] = {
            "status": "completed",
            "timestamp": self.timestamp,
            "case_count": len(case_ids),
            "case_ids": case_ids,
            "config": asdict(self.config),
            "stages": {},
        }

        stage_calls = [
            ("audit", lambda: self.run_audit(case_ids)),
            ("baselines", lambda: self.run_baselines(case_ids)),
            ("quantitative", self.run_quantitative_summary),
        ]
        for stage_name, call in stage_calls:
            try:
                status["stages"][stage_name] = call()
            except Exception as exc:
                status["stages"][stage_name] = {"status": "failed", "error": str(exc)}
                if not self.config.continue_on_error:
                    status["status"] = "failed"
                    break
        return status

    def save_summary(self, summary: Dict[str, Any], filename: str = "complete_pipeline_results.json") -> Path:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.config.output_dir / filename
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        return output_path


def create_preset_configurations() -> Dict[str, PipelineConfig]:
    manager = ExperimentConfigManager()
    presets: Dict[str, PipelineConfig] = {}
    for name, preset in manager.builtin_configs.items():
        presets[name] = PipelineConfig(
            case_scale=preset.case_scale,
            case_type=preset.case_type,
            case_ids=preset.case_ids,
            case_limit=preset.case_limit,
            seed=preset.random_seed,
            run_audit_pipeline=preset.audit_pipeline.enabled,
            run_baselines_pipeline=preset.baselines_pipeline.enabled,
            run_quantitative_eval=preset.quantitative_evaluation.enabled,
            audit_method=preset.module_configs.audit_method,
            aggregation_method=preset.module_configs.aggregation_method,
            mc_samples=preset.module_configs.mc_samples,
            base_data_dir=Path(preset.base_data_dir),
            output_dir=Path(preset.base_output_dir),
        )
    return presets


def parse_case_ids(raw: Optional[str]) -> Optional[List[str]]:
    if not raw:
        return None
    return [item.strip() for item in raw.split(",") if item.strip()]


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    presets = create_preset_configurations()

    parser = argparse.ArgumentParser(description="Run released PCCA experiments")
    parser.add_argument("--preset", choices=sorted(presets), default="quick_test")
    parser.add_argument("--case-scale", choices=["smallcase", "mediumcase", "bigcase"])
    parser.add_argument("--case-type", choices=["Mixed", "BCSA", "CEDA"])
    parser.add_argument("--case-ids", help="Comma-separated case ids")
    parser.add_argument("--audit-method", choices=["PCVGAE", "PCGATE", "BOTH"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--base-data-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--skip-audit", action="store_true")
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--skip-quant", action="store_true")
    parser.add_argument("--save-summary", action="store_true")
    args = parser.parse_args()

    config = presets[args.preset]
    if args.case_scale:
        config.case_scale = args.case_scale
    if args.case_type:
        config.case_type = args.case_type
    if args.case_ids:
        config.case_ids = parse_case_ids(args.case_ids)
    if args.audit_method:
        config.audit_method = args.audit_method
    if args.seed is not None:
        config.seed = args.seed
    if args.base_data_dir:
        config.base_data_dir = Path(args.base_data_dir)
    if args.output_dir:
        config.output_dir = Path(args.output_dir)
    if args.skip_audit:
        config.run_audit_pipeline = False
    if args.skip_baselines:
        config.run_baselines_pipeline = False
    if args.skip_quant:
        config.run_quantitative_eval = False

    pipeline = CompleteExperimentPipeline(config)
    summary = pipeline.run_complete_pipeline()
    if args.save_summary:
        output_path = pipeline.save_summary(summary)
        print(f"Summary saved to {output_path}")
    print(json.dumps({"status": summary["status"], "case_count": summary.get("case_count", 0)}, ensure_ascii=False))
    return 0 if summary["status"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
