#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
三规模数据汇总脚本 (正式版)
汇总 small / medium / large 三个规模的全部基线 + PCCA 方法的评估结果
输出: 
  - seek_data_v3_deep_enhanced/results/Quantitative_analysis/all_scales_summary.csv
  - 终端对比表
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RESULTS_BASE = Path("./seek_data_v3_deep_enhanced/results")
SCALES = ["smallcase", "mediumcase", "bigcase"]
CASE_TYPE = "Mixed"

ALL_METHODS = [
    "PCGATE", "PCVGAE",
    "TransE", "CommonNeighbors", "DistMult",
    "GAT", "GATE", "VGAE",
    "Peter-Clark", "GES",
    "CDHC", "IF-PDAG", "NOTEARS",
]

METHOD_CATEGORY = {
    "PCGATE": "Prompt-Conditioned (Ours)",
    "PCVGAE": "Prompt-Conditioned (Ours)",
    "TransE": "KG Completion",
    "CommonNeighbors": "KG Completion",
    "DistMult": "KG Completion",
    "GAT": "Graph Representation",
    "GATE": "Graph Representation",
    "VGAE": "Graph Representation",
    "Peter-Clark": "Data-Driven Causal",
    "GES": "Data-Driven Causal",
    "CDHC": "Causal Optimization",
    "IF-PDAG": "Causal Optimization",
    "NOTEARS": "Causal Optimization",
}


def collect_evaluation_results(scale: str) -> pd.DataFrame:
    """收集某个规模下全部方法的评估结果"""
    results_dir = RESULTS_BASE / scale / CASE_TYPE
    rows = []
    
    if not results_dir.exists():
        logger.warning(f"Results directory not found: {results_dir}")
        return pd.DataFrame()
    
    for case_dir in sorted(results_dir.iterdir()):
        if not case_dir.is_dir():
            continue
        case_id = case_dir.name
        
        for method in ALL_METHODS:
            possible_dirs = [
                case_dir / f"{method}_Analysis",
                case_dir / f"{method}_audit",
                case_dir / f"PCGATE_Analysis" if method == "PCGATE" else None,
                case_dir / f"PCVGAE_Analysis" if method == "PCVGAE" else None,
            ]
            
            eval_file = None
            for d in possible_dirs:
                if d and (d / "evaluation_results.json").exists():
                    eval_file = d / "evaluation_results.json"
                    break
            
            if eval_file is None:
                continue
            
            try:
                with open(eval_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                perf = data.get('quantitative_metrics', {}).get('performance_metrics', {})
                basic = data.get('quantitative_metrics', {}).get('basic_metrics', {})
                
                rows.append({
                    'scale': scale,
                    'case_id': case_id,
                    'method': method,
                    'category': METHOD_CATEGORY.get(method, 'Unknown'),
                    'f1_score': perf.get('f1_score', 0),
                    'blind_spot_recall': perf.get('blind_spot_recall', 0),
                    'auc_pr': perf.get('auc_pr', 0),
                    'ndcg_10': perf.get('ndcg_10', 0),
                    'total_findings': basic.get('total_findings', 0),
                })
            except Exception as e:
                logger.warning(f"Error reading {eval_file}: {e}")
    
    return pd.DataFrame(rows)


def generate_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """生成三规模汇总表"""
    if df.empty:
        return pd.DataFrame()
    
    summary = df.groupby(['scale', 'method', 'category']).agg(
        f1_mean=('f1_score', 'mean'),
        f1_std=('f1_score', 'std'),
        bsr_mean=('blind_spot_recall', 'mean'),
        n_cases=('case_id', 'nunique'),
    ).reset_index()
    
    return summary


def print_comparison_table(summary: pd.DataFrame):
    """打印三规模对比表"""
    if summary.empty:
        print("No data to display")
        return
    
    print("\n" + "=" * 90)
    print("THREE-SCALE BASELINE COMPARISON TABLE")
    print("=" * 90)
    
    header = f"{'Method':15s} | {'Category':22s}"
    for scale in SCALES:
        scale_label = scale.replace("case", "")
        header += f" | {scale_label:>12s}"
    header += f" | {'N':>3s}"
    print(header)
    print("-" * len(header))
    
    for method in ALL_METHODS:
        row = f"{method:15s} | {METHOD_CATEGORY.get(method, ''):22s}"
        for scale in SCALES:
            mask = (summary['method'] == method) & (summary['scale'] == scale)
            if mask.any():
                f1 = summary.loc[mask, 'f1_mean'].values[0]
                std = summary.loc[mask, 'f1_std'].values[0]
                row += f" | {f1:.3f}±{std:.3f}"
            else:
                row += f" |     —      "
        
        n = summary.loc[summary['method'] == method, 'n_cases'].max()
        row += f" | {int(n):3d}" if pd.notna(n) else " |   0"
        print(row)
    
    print("=" * 90)


def main():
    print("Collecting evaluation results across all scales...")
    
    all_dfs = []
    for scale in SCALES:
        df = collect_evaluation_results(scale)
        if not df.empty:
            print(f"  {scale}: {len(df)} records ({df['case_id'].nunique()} cases × {df['method'].nunique()} methods)")
            all_dfs.append(df)
        else:
            print(f"  {scale}: No data found")
    
    if not all_dfs:
        print("No data collected!")
        return
    
    combined = pd.concat(all_dfs, ignore_index=True)
    
    output_dir = RESULTS_BASE / "Quantitative_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    raw_file = output_dir / "all_scales_raw_evaluation.csv"
    combined.to_csv(raw_file, index=False, encoding='utf-8')
    print(f"\nRaw data saved to: {raw_file}")
    
    summary = generate_summary_table(combined)
    summary_file = output_dir / "all_scales_summary.csv"
    summary.to_csv(summary_file, index=False, encoding='utf-8')
    print(f"Summary saved to: {summary_file}")
    
    print_comparison_table(summary)
    
    print("\n--- Per-Scale Statistics ---")
    for scale in SCALES:
        scale_data = combined[combined['scale'] == scale]
        if not scale_data.empty:
            best = scale_data.groupby('method')['f1_score'].mean().idxmax()
            best_f1 = scale_data.groupby('method')['f1_score'].mean().max()
            print(f"  {scale:12s}: {scale_data['case_id'].nunique()} cases, "
                  f"{scale_data['method'].nunique()} methods, "
                  f"Best: {best} (F1={best_f1:.4f})")


if __name__ == "__main__":
    main()
