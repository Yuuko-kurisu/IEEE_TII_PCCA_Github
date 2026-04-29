#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared utilities for released PCCA experiments."""

from pathlib import Path
from typing import Any, Dict, List, Tuple
import json
import logging

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

BASE_DATA_DIR = Path("./seek_data_v3_deep_enhanced/cases")
BASE_RESULTS_DIR = Path("./seek_data_v3_deep_enhanced/results")


CORE_METRICS_6 = {
    "evidence_precision": "Evidence Precision",
    "evidence_recall": "Evidence Recall",
    "evidence_f1": "Evidence F1 Score",
    "auc_pr": "AUC-PR",
    "ndcg15": "NDCG@15",
    "ndcg25": "NDCG@25",
}


def normalize_scale_name(case_scale: str) -> str:
    return case_scale if case_scale.endswith("case") else f"{case_scale}case"


class PathManager:
    """Path helper for the released case and result layout."""

    def __init__(self, case_scale: str, case_type: str, case_id: str):
        self.case_scale = case_scale
        self.case_type = case_type
        self.case_id = case_id

    @property
    def scale_folder_name(self) -> str:
        return normalize_scale_name(self.case_scale)

    def get_case_dir(self) -> Path:
        return BASE_DATA_DIR / self.scale_folder_name / self.case_type / self.case_id

    def get_case_results_dir(self) -> Path:
        return BASE_RESULTS_DIR / self.scale_folder_name / self.case_type / self.case_id


def setup_ieee_style() -> None:
    """Set a stable matplotlib style for paper-style plots."""
    plt.style.use("default")
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "mathtext.fontset": "stix",
            "font.size": 12,
            "axes.labelsize": 13,
            "axes.titlesize": 14,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "figure.dpi": 300,
            "text.usetex": False,
            "axes.unicode_minus": False,
            "lines.linewidth": 1.8,
            "lines.markersize": 6,
            "grid.alpha": 0.35,
            "grid.linestyle": "--",
        }
    )


def get_method_styles(methods: List[str]) -> Dict[str, Dict[str, Any]]:
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]
    markers = ["o", "s", "^", "D", "v", "p", "*"]
    styles: Dict[str, Dict[str, Any]] = {}
    for i, method in enumerate(methods):
        styles[method] = {
            "color": "#1f77b4" if method in {"PCGATE", "PC-GATE"} else colors[i % len(colors)],
            "marker": markers[i % len(markers)],
            "linestyle": "-",
            "linewidth": 2.0 if method in {"PCGATE", "PC-GATE"} else 1.5,
            "markersize": 6,
        }
    return styles


def discover_case_structure(base_data_dir: Path = BASE_DATA_DIR) -> Dict[str, List[str]]:
    case_structure: Dict[str, List[str]] = {}
    if not base_data_dir.exists():
        return case_structure
    for scale_dir in sorted(base_data_dir.iterdir()):
        if scale_dir.is_dir() and scale_dir.name.endswith("case"):
            scale_name = scale_dir.name.replace("case", "")
            case_structure[scale_name] = sorted([p.name for p in scale_dir.iterdir() if p.is_dir()])
    return case_structure


def get_available_cases(base_data_dir: Path, case_scale: str, case_type: str) -> List[str]:
    case_dir = base_data_dir / normalize_scale_name(case_scale) / case_type
    if not case_dir.exists():
        return []
    return sorted(
        case_path.name
        for case_path in case_dir.iterdir()
        if case_path.is_dir() and (case_path / "causal_knowledge_graph.json").exists()
    )


def select_case_scope() -> Tuple[str, str, List[str]]:
    case_structure = discover_case_structure(BASE_DATA_DIR)
    if not case_structure:
        raise ValueError(f"No case structure found under {BASE_DATA_DIR}")

    available_scales = list(case_structure.keys())
    print("\nAvailable scales:")
    for i, scale in enumerate(available_scales, 1):
        print(f"  {i}. {scale}")
    scale_idx = int(input(f"Select scale (1-{len(available_scales)}, default 1): ").strip() or "1") - 1
    selected_scale = available_scales[scale_idx]

    available_types = case_structure[selected_scale]
    print(f"\nAvailable case types for {selected_scale}:")
    for i, type_name in enumerate(available_types, 1):
        print(f"  {i}. {type_name}")
    type_idx = int(input(f"Select type (1-{len(available_types)}, default 1): ").strip() or "1") - 1
    selected_type = available_types[type_idx]

    available_cases = get_available_cases(BASE_DATA_DIR, selected_scale, selected_type)
    print(f"\nSelected {selected_scale}/{selected_type}: {len(available_cases)} cases")
    return selected_scale, selected_type, available_cases


def select_batch_cases(available_cases: List[str]) -> List[str]:
    if not available_cases:
        return []
    print("\nBatch case selection:")
    print(f"  1. all cases ({len(available_cases)})")
    print("  2. first N cases")
    print("  3. comma-separated case ids")
    mode = input("Select mode (1/2/3, default 1): ").strip() or "1"
    if mode == "2":
        n = int(input(f"N (1-{len(available_cases)}): ").strip())
        return available_cases[:n]
    if mode == "3":
        requested = [x.strip() for x in input("Case ids: ").split(",") if x.strip()]
        valid = [x for x in requested if x in available_cases]
        invalid = [x for x in requested if x not in available_cases]
        if invalid:
            print(f"Ignoring unknown case ids: {', '.join(invalid)}")
        return valid
    return available_cases


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
