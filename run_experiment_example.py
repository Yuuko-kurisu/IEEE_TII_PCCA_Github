#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Minimal examples for the released PCCA pipeline."""

from pathlib import Path

from BCSA_11_Complete_Experiment_Pipeline import CompleteExperimentPipeline, PipelineConfig
from BCSA_12_Experiment_Configs import get_config, print_config_summary


def quick_demo():
    config = PipelineConfig(
        case_scale="smallcase",
        case_type="Mixed",
        case_ids=["Mixed_small_01"],
        run_audit_pipeline=True,
        run_baselines_pipeline=False,
        run_quantitative_eval=True,
        base_data_dir=Path("data/full_benchmark"),
        seed=42,
    )
    pipeline = CompleteExperimentPipeline(config)
    return pipeline.run_complete_pipeline()


def show_main_config():
    config = get_config("main_medium")
    print_config_summary(config)


if __name__ == "__main__":
    show_main_config()
    print(quick_demo())
