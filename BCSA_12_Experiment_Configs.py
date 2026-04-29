#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Experiment presets for the released PCCA code."""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import json


PAPER_METHODS = [
    "PCGATE",
    "PCVGAE",
    "CommonNeighbors",
    "Peter-Clark",
    "GAT",
    "GATE",
    "VGAE",
    "DistMult",
    "NOTEARS",
    "GES",
    "GOLEM",
    "CDHC",
    "IF-PDAG",
]

PAPER_METRICS = [
    "evidence_precision",
    "evidence_recall",
    "f1_score",
    "auc_pr",
    "global_ndcg_at_15",
    "global_ndcg_at_25",
]


@dataclass
class ExperimentStageConfig:
    enabled: bool = True
    priority: int = 1
    case_limit: Optional[int] = None
    continue_on_error: bool = True
    timeout_minutes: Optional[int] = None


@dataclass
class ModuleSpecificConfig:
    audit_method: str = "PCGATE"
    aggregation_method: str = "weighted_max"
    mc_samples: int = 50
    baselines_to_run: List[str] = field(default_factory=lambda: list(PAPER_METHODS))
    baseline_top_k: int = 35
    evaluation_metrics: List[str] = field(default_factory=lambda: list(PAPER_METRICS))


@dataclass
class CompleteExperimentConfig:
    config_name: str
    description: str
    version: str = "1.0"
    case_scale: str = "mediumcase"
    case_type: str = "Mixed"
    case_ids: Optional[List[str]] = None
    case_limit: Optional[int] = None
    random_seed: int = 42
    base_data_dir: str = "./seek_data_v3_deep_enhanced/cases"
    base_output_dir: str = "./seek_data_v3_deep_enhanced/results"
    audit_pipeline: ExperimentStageConfig = field(default_factory=lambda: ExperimentStageConfig(enabled=True, priority=1))
    baselines_pipeline: ExperimentStageConfig = field(default_factory=lambda: ExperimentStageConfig(enabled=True, priority=2))
    quantitative_evaluation: ExperimentStageConfig = field(default_factory=lambda: ExperimentStageConfig(enabled=True, priority=3))
    module_configs: ModuleSpecificConfig = field(default_factory=ModuleSpecificConfig)
    parallel_execution: bool = False
    max_parallel_jobs: int = 1
    save_intermediate_results: bool = True
    cleanup_temp_files: bool = True
    log_level: str = "INFO"
    progress_reporting: bool = True

    def get_enabled_stages(self) -> List[str]:
        stages = [
            ("audit_pipeline", self.audit_pipeline),
            ("baselines_pipeline", self.baselines_pipeline),
            ("quantitative_evaluation", self.quantitative_evaluation),
        ]
        enabled = [(name, config) for name, config in stages if config.enabled]
        enabled.sort(key=lambda x: x[1].priority)
        return [name for name, _ in enabled]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CompleteExperimentConfig":
        converted = dict(data)
        for key in ["audit_pipeline", "baselines_pipeline", "quantitative_evaluation"]:
            if key in converted and not isinstance(converted[key], ExperimentStageConfig):
                converted[key] = ExperimentStageConfig(**converted[key])
        if "module_configs" in converted and not isinstance(converted["module_configs"], ModuleSpecificConfig):
            converted["module_configs"] = ModuleSpecificConfig(**converted["module_configs"])
        return cls(**converted)


class ExperimentConfigManager:
    def __init__(self, config_dir: Path = None):
        self.config_dir = config_dir or Path("./experiment_configs")
        self.config_dir.mkdir(exist_ok=True)
        self.builtin_configs = self._create_builtin_configs()

    def _create_builtin_configs(self) -> Dict[str, CompleteExperimentConfig]:
        return {
            "quick_test": CompleteExperimentConfig(
                config_name="quick_test",
                description="Single-case smoke test for the released pipeline.",
                case_scale="smallcase",
                case_type="Mixed",
                case_ids=["Mixed_small_01"],
                case_limit=1,
                baselines_pipeline=ExperimentStageConfig(enabled=False, priority=2),
                quantitative_evaluation=ExperimentStageConfig(enabled=True, priority=3, case_limit=1),
            ),
            "main_medium": CompleteExperimentConfig(
                config_name="main_medium",
                description="Medium-scale Mixed benchmark used for the accepted main table.",
                case_scale="mediumcase",
                case_type="Mixed",
                case_limit=None,
            ),
            "cross_scale_small": CompleteExperimentConfig(
                config_name="cross_scale_small",
                description="Small-scale Mixed benchmark run.",
                case_scale="smallcase",
                case_type="Mixed",
            ),
            "cross_scale_large": CompleteExperimentConfig(
                config_name="cross_scale_large",
                description="Large-scale Mixed benchmark run.",
                case_scale="bigcase",
                case_type="Mixed",
            ),
            "eval_only": CompleteExperimentConfig(
                config_name="eval_only",
                description="Aggregate existing result artifacts without re-running models.",
                audit_pipeline=ExperimentStageConfig(enabled=False, priority=1),
                baselines_pipeline=ExperimentStageConfig(enabled=False, priority=2),
                quantitative_evaluation=ExperimentStageConfig(enabled=True, priority=3),
            ),
        }

    def get_config(self, config_name: str) -> CompleteExperimentConfig:
        if config_name not in self.builtin_configs:
            available = ", ".join(sorted(self.builtin_configs))
            raise ValueError(f"Unknown config '{config_name}'. Available presets: {available}")
        return self.builtin_configs[config_name]

    def list_configs(self) -> List[str]:
        return sorted(self.builtin_configs)

    def save_config(self, config: CompleteExperimentConfig, filename: str = None) -> Path:
        filename = filename or f"{config.config_name}.json"
        output_path = self.config_dir / filename
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(config.to_dict(), f, indent=2, ensure_ascii=False)
        return output_path

    def load_config(self, filename: str) -> CompleteExperimentConfig:
        with open(self.config_dir / filename, "r", encoding="utf-8") as f:
            return CompleteExperimentConfig.from_dict(json.load(f))


def get_config(config_name: str) -> CompleteExperimentConfig:
    return ExperimentConfigManager().get_config(config_name)


def list_available_configs() -> List[str]:
    return ExperimentConfigManager().list_configs()


def print_config_summary(config: CompleteExperimentConfig) -> None:
    print(f"Config: {config.config_name}")
    print(f"Description: {config.description}")
    print(f"Dataset: {config.case_scale}/{config.case_type}")
    print(f"Seed: {config.random_seed}")
    print(f"Enabled stages: {', '.join(config.get_enabled_stages())}")
    print(f"Audit method: {config.module_configs.audit_method}")
    print(f"Aggregation: {config.module_configs.aggregation_method}")
    print(f"MC samples: {config.module_configs.mc_samples}")
    print(f"Baseline top_k: {config.module_configs.baseline_top_k}")
    print(f"Baselines: {', '.join(config.module_configs.baselines_to_run)}")
