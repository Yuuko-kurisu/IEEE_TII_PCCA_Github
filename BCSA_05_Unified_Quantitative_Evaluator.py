#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pipeline
统一分析器
汇总BCSA和基线方法的所有结果，进行对比分析和可视化
"""

import json
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime
import warnings
import openpyxl
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
from BCSA_07_Experiment_Utils import select_case_scope


import random
import numpy as np
import torch
def set_global_determinism(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"--- Global seed set to {seed} ---")

set_global_determinism(42)

global PRESET_METHODS
PRESET_METHODS = [
    'PCGATE',
    'PCVGAE',
    'CommonNeighbors', 
    'Peter-Clark',
    'GAT',
    'GATE',
    'VGAE'
    'CDHC',
    'IF-PDAG'
]


CORE_METRICS_6 = {
    'evidence_p': 'Evidence Precision',
    'evidence_r': 'Evidence Recall',
    'evidence_f1': 'Evidence F1 Score',
    'aucpr': 'AUC-PR',
    'ndcg15': 'NDCG@15',
    'ndcg25': 'NDCG@25'
}

TASK1_METRICS_6 = {k: v for k, v in CORE_METRICS_6.items() if 'ndcg' not in k}
TASK2_METRICS_6 = {k: v for k, v in CORE_METRICS_6.items() if 'ndcg' in k}

plt.style.use('default')  # 重置为默认样式
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "lines.linewidth": 1.5,
    "lines.markersize": 5,
    "axes.linewidth": 0.8,
    "grid.alpha": 0.3
})


def setup_ieee_style():
    """设置统一的IEEE论文绘图风格 (更新版)"""
    plt.style.use('default')  # Reset to default style
    plt.rcParams.update({
        "font.family": "Times New Roman",
        'mathtext.fontset': 'stix', # For LaTeX-style math fonts
        "font.size": 16,
        "axes.labelsize": 18,
        "axes.titlesize": 20,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 16,
        'figure.titlesize': 22,
        'text.usetex': False,
        'axes.unicode_minus': False,
        "lines.linewidth": 2.0, # Thicker lines
        "lines.markersize": 7, # Larger markers
        "axes.linewidth": 0.8,
        "grid.alpha": 0.5,
        "grid.linestyle": '--'
    })

def get_method_styles(methods: List[str]) -> Dict[str, Dict[str, Any]]:
    """获取统一的方法样式配置"""
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', 
              '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', 'X', '+']
    linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--']
    
    styles = {}
    
    if 'BCSA' in methods:
        styles['BCSA'] = {
            'color': '#FF6B6B',
            'marker': 'o',
            'linestyle': '-',
            'linewidth': 2.0,
            'markersize': 6
        }
    
    for i, method in enumerate(methods):
        if method not in styles:
            styles[method] = {
                'color': colors[i % len(colors)],
                'marker': markers[i % len(markers)],
                'linestyle': linestyles[i % len(linestyles)],
                'linewidth': 1.5,
                'markersize': 5
            }
    
    return styles

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_all_evaluation_results(results_base_dir: Path, case_scale: str, case_type: str) -> pd.DataFrame:
    """
    扫描整个结果目录，智能地找到所有方法的评估报告，并整合成DataFrame。
    """
    logger.info(f"开始加载评估结果: {results_base_dir} / {case_scale} / {case_type}")
    
    all_results = []
    target_dir = results_base_dir / case_scale / case_type
    
    if not target_dir.exists():
        logger.error(f"目标目录不存在: {target_dir}")
        return pd.DataFrame()
    
    for case_dir in target_dir.iterdir():
        if not case_dir.is_dir():
            continue
            
        case_id = case_dir.name
        logger.debug(f"处理案例: {case_id}")
        
        pcvgae_eval_file = case_dir / "PCVGAE_Analysis" / "1_Final_Results" / "evaluation" / "quantitative_evaluation_report.json"
        
        if not pcvgae_eval_file.exists():
            pcvgae_eval_file_old = case_dir / "BCSA_Analysis" / "1_Final_Results" / "evaluation" / "quantitative_evaluation_report.json"
            if pcvgae_eval_file_old.exists():
                pcvgae_eval_file = pcvgae_eval_file_old
                logger.debug(f"  使用兼容PCVGAE路径: {pcvgae_eval_file}")
        
        if pcvgae_eval_file.exists():
            try:
                with open(pcvgae_eval_file, 'r', encoding='utf-8') as f:
                    pcvgae_data = json.load(f)
                
                result_row = _extract_metrics_from_result(pcvgae_data, 'PCVGAE', case_id, case_scale, case_type)
                if result_row:
                    all_results.append(result_row)
                    logger.debug(f"  ✓ 加载PCVGAE结果")
            except Exception as e:
                logger.warning(f"  ❌ 加载PCVGAE结果失败: {e}")
        
        pcgate_eval_file = case_dir / "PCGATE_Analysis" / "1_Final_Results" / "evaluation" / "quantitative_evaluation_report.json"
        
        if pcgate_eval_file.exists():
            try:
                with open(pcgate_eval_file, 'r', encoding='utf-8') as f:
                    pcgate_data = json.load(f)
                
                result_row = _extract_metrics_from_result(pcgate_data, 'PCGATE', case_id, case_scale, case_type)
                if result_row:
                    all_results.append(result_row)
                    logger.debug(f"  ✓ 加载PCGATE结果")
            except Exception as e:
                logger.warning(f"  ❌ 加载PCGATE结果失败: {e}")
        
        for method_dir in case_dir.iterdir():
            if not method_dir.is_dir() or method_dir.name in ["PCVGAE_Analysis", "PCGATE_Analysis", "BCSA_Analysis"]:
                continue
                
            if method_dir.name.endswith("_Analysis"):
                method_name = method_dir.name.replace("_Analysis", "")
                eval_file = method_dir / "evaluation_results.json"
                
                if eval_file.exists():
                    try:
                        with open(eval_file, 'r', encoding='utf-8') as f:
                            baseline_data = json.load(f)
                        
                        result_row = _extract_metrics_from_result(baseline_data, method_name, case_id, case_scale, case_type)
                        if result_row:
                            all_results.append(result_row)
                            logger.debug(f"  ✓ 加载{method_name}结果")
                    except Exception as e:
                        logger.warning(f"  ❌ 加载{method_name}结果失败: {e}")
    
    if not all_results:
        logger.warning("未找到任何有效的评估结果")
        return pd.DataFrame()
    
    df = pd.DataFrame(all_results)
    logger.info(f"成功加载 {len(df)} 条评估记录，涵盖 {df['method'].nunique()} 种方法，{df['case_id'].nunique()} 个案例")
    logger.info(f"发现的方法: {sorted(df['method'].unique())}")
    
    return df

def plot_pr_curves(df: pd.DataFrame, output_dir: Path):
    """绘制PR曲线对比图 - 本（解决曲线形态问题）"""
    logger.info("生成PR曲线对比图（本）...")
    
    if df.empty:
        logger.warning("DataFrame为空，跳过PR曲线绘图")
        return
    
    setup_ieee_style()
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    global PRESET_METHODS
    if 'PRESET_METHODS' in globals():
        preset_methods = PRESET_METHODS
    else:
        preset_methods = [
            'PCGATE', 'PCVGAE', 'CommonNeighbors', 'Peter-Clark',
            'GAT', 'GATE', 'VGAE', 'No_Prompt (PCVGAE)', 'No_Prompt (PCGATE)', 'No_Audit'
        ]
    
    available_methods = df['method'].unique()
    unique_methods = [method for method in preset_methods if method in available_methods]
    remaining_methods = [method for method in available_methods if method not in unique_methods]
    unique_methods.extend(sorted(remaining_methods))  # 剩余方法按字母排序添加到末尾
    
    method_styles = get_method_styles(unique_methods)
    
    plotted_methods = []
    
    for method in unique_methods:
        method_data = df[df['method'] == method]
        
        all_precision_curves = []
        all_recall_curves = []
        auc_pr_values = []
        
        recall_base = np.linspace(0, 1, 101)
        
        for _, row in method_data.iterrows():
            pr_data = row.get('pr_curve_data', {})
            auc_pr = row.get('aucpr', 0.0)  # 使用标准化的列名
            
            if isinstance(pr_data, dict) and 'precision' in pr_data and 'recall' in pr_data:
                precision = np.array(pr_data['precision'])
                recall = np.array(pr_data['recall'])
                
                if len(precision) > 0 and len(recall) > 0 and len(precision) == len(recall):
                    sorted_indices = np.argsort(recall)
                    recall_sorted = recall[sorted_indices]
                    precision_sorted = precision[sorted_indices]
                    
                    if len(recall_sorted) > 1:
                        unique_recall, unique_indices = np.unique(recall_sorted, return_index=True)
                        unique_precision = precision_sorted[unique_indices]
                        
                        if len(unique_recall) > 1:
                            precision_interp = np.interp(recall_base, unique_recall, unique_precision)
                            all_precision_curves.append(precision_interp)
                            all_recall_curves.append(recall_base)
                            auc_pr_values.append(auc_pr)
                            
                            logger.debug(f"  {method} - 案例 {row['case_id']}: "
                                       f"原始点数={len(recall)}, 排序后={len(recall_sorted)}, "
                                       f"去重后={len(unique_recall)}, AUC-PR={auc_pr:.3f}")
        
        if all_precision_curves:
            precision_matrix = np.array(all_precision_curves)
            avg_precision = np.mean(precision_matrix, axis=0)
            avg_auc_pr = np.mean(auc_pr_values)
            
            style = method_styles.get(method, {})
            ax.plot(recall_base, avg_precision, 
                   color=style.get('color', 'gray'), 
                   linewidth=style.get('linewidth', 1.5), 
                   linestyle=style.get('linestyle', '-'),
                   label=f'{method} (AUC={avg_auc_pr:.3f})')
            
            plotted_methods.append(method)
            logger.debug(f"✓ 绘制 {method} PR曲线：{len(all_precision_curves)} 个案例平均，AUC-PR={avg_auc_pr:.3f}")
        else:
            logger.warning(f"⚠️ {method} 没有有效的PR曲线数据")
    
    if plotted_methods:
        ax.set_xlabel('Recall', fontsize=18)
        ax.set_ylabel('Precision', fontsize=18)
        ax.set_title('Precision-Recall Curves Comparison', fontsize=20, fontweight='bold')
        
        ncol = min(len(plotted_methods), 3)  # 最多4列，超过就换行
        ax.legend(bbox_to_anchor=(0.5, -0.10), loc='upper center', fontsize=14, ncol=ncol, framealpha=0.9)
        
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        
        logger.info(f"✅ 成功绘制 {len(plotted_methods)} 个方法的PR曲线")
    else:
        ax.text(0.5, 0.5, 'No valid PR curve data available', 
               ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_title('Precision-Recall Curves (No Data)', fontsize=14)
        logger.warning("❌ 没有找到任何有效的PR曲线数据")
    
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    
    chart_file = output_dir / "pr_curves_comparison_fixed.png"
    chart_file_pdf = output_dir / "pr_curves_comparison_fixed.pdf"
    try:
        plt.savefig(chart_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(chart_file_pdf, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"PR曲线对比图已保存: {chart_file}")
        logger.info(f"PR曲线对比图已保存: {chart_file_pdf}")
    except Exception as e:
        logger.error(f"保存PR曲线图表失败: {e}")
    finally:
        plt.close()


def plot_ndcg_curves(df: pd.DataFrame, output_dir: Path):
    """绘制NDCG@k曲线对比图"""
    logger.info("生成NDCG@k曲线对比图...")
    
    if df.empty:
        logger.warning("DataFrame为空，跳过NDCG@k曲线绘图")
        return
    
    ndcg_data = []
    
    for _, row in df.iterrows():
        method = row['method']
        case_id = row['case_id']
        ndcg_at_k = row.get('ndcg_at_k', {})
        
        if isinstance(ndcg_at_k, dict):
            for k_str, ndcg_score in ndcg_at_k.items():
                try:
                    k = int(k_str)
                    ndcg_data.append({
                        'method': method,
                        'case_id': case_id,
                        'k': k,
                        'ndcg_score': float(ndcg_score)
                    })
                except (ValueError, TypeError):
                    continue
    
    if not ndcg_data:
        logger.warning("未找到有效的NDCG@k数据，跳过绘图")
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.text(0.5, 0.5, 'No NDCG@k data available', 
               ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_title('NDCG@k Curves (No Data)', fontsize=14)
        
        chart_file = output_dir / "ndcg_at_k_comparison.png"
        plt.savefig(chart_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        return
    
    ndcg_df = pd.DataFrame(ndcg_data)
    
    avg_ndcg = ndcg_df.groupby(['method', 'k'])['ndcg_score'].mean().reset_index()
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    global PRESET_METHODS
    if 'PRESET_METHODS' in globals():
        preset_methods = PRESET_METHODS
    else:
        preset_methods = [
            'PCGATE', 'PCVGAE', 'CommonNeighbors', 'Peter-Clark',
            'GAT', 'GATE', 'VGAE', 'No_Prompt (PCVGAE)', 'No_Prompt (PCGATE)', 'No_Audit'
        ]
    
    available_methods = avg_ndcg['method'].unique()
    unique_methods = [method for method in preset_methods if method in available_methods]
    remaining_methods = [method for method in available_methods if method not in unique_methods]
    unique_methods.extend(sorted(remaining_methods))  # 剩余方法按字母排序添加到末尾
    
    method_styles = get_method_styles(unique_methods)
    
    for method in unique_methods:
        method_data = avg_ndcg[avg_ndcg['method'] == method].sort_values('k')
        
        if not method_data.empty:
            style = method_styles.get(method, {})
            ax.plot(method_data['k'], method_data['ndcg_score'], 
                   color=style.get('color', 'gray'),
                   marker=style.get('marker', 'o'),
                   linestyle=style.get('linestyle', '-'),
                   linewidth=style.get('linewidth', 1.5), 
                   markersize=style.get('markersize', 5), 
                   alpha=0.8, clip_on=False,
                   label=method)
    
    ax.set_xlabel('K Value (Top-k)', fontsize=18)
    ax.set_ylabel('NDCG@k Score', fontsize=18)
    ax.set_title('NDCG@k Curves Comparison', fontsize=20, fontweight='bold')
    
    ncol = min(len(unique_methods), 5)  # 最多4列，超过就换行
    ax.legend(bbox_to_anchor=(0.5, -0.10), loc='upper center', fontsize=14, ncol=ncol, framealpha=0.9)
    
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    
    if not avg_ndcg.empty:
        k_values = sorted(avg_ndcg['k'].unique())
        ax.set_xticks(k_values)
        ax.set_xlim(min(k_values) - 1, max(k_values) + 1)
    
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    
    chart_file = output_dir / "ndcg_at_k_comparison.png"
    chart_file_pdf = output_dir / "ndcg_at_k_comparison.pdf"
    try:
        plt.savefig(chart_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(chart_file_pdf, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"NDCG@k曲线对比图已保存: {chart_file}")
        logger.info(f"NDCG@k曲线对比图已保存: {chart_file_pdf}")
    except Exception as e:
        logger.error(f"保存NDCG@k图表失败: {e}")
    finally:
        plt.close()


def _extract_metrics_from_result(result_data: Dict[str, Any], method: str, case_id: str, 
                                case_scale: str, case_type: str) -> Dict[str, Any]:
    """从评估结果中提取6个核心指标 + 边际效益指标"""
    try:
        metrics = result_data.get('quantitative_metrics', {})
        perf_metrics = metrics.get('performance_metrics', {})
        
        marginal_benefit = metrics.get('marginal_benefit_metrics', {})
        
        result = {
            'method': method,
            'case_id': case_id,
            'case_scale': case_scale,
            'case_type': case_type,
            
            'evidence_p': perf_metrics.get('evidence_precision', 0.0),
            'evidence_r': perf_metrics.get('evidence_recall', 0.0),
            'evidence_f1': perf_metrics.get('f1_score', 0.0),
            'aucpr': perf_metrics.get('auc_pr', 0.0),
            
            'ndcg10': perf_metrics.get('global_ndcg_at_10', 0.0),
            'ndcg15': perf_metrics.get('global_ndcg_at_15', 0.0), # 修改
            'ndcg20': perf_metrics.get('global_ndcg_at_20', 0.0),
            'ndcg25': perf_metrics.get('global_ndcg_at_25', 0.0), # 修改
            
            'evidence_precision_marginal': marginal_benefit.get('evidence_precision_marginal', 0.0),
            'evidence_recall_marginal': marginal_benefit.get('evidence_recall_marginal', 0.0),
            'f1_score_marginal': marginal_benefit.get('f1_score_marginal', 0.0),
            'auc_pr_marginal': marginal_benefit.get('auc_pr_marginal', 0.0),
            'global_ndcg_at_10_marginal': marginal_benefit.get('global_ndcg_at_10_marginal', 0.0),
            'global_ndcg_at_20_marginal': marginal_benefit.get('global_ndcg_at_20_marginal', 0.0),
            'global_ndcg_at_15_marginal': marginal_benefit.get('global_ndcg_at_15_marginal', 0.0), # 修改
            'global_ndcg_at_25_marginal': marginal_benefit.get('global_ndcg_at_25_marginal', 0.0), # 修改
            
            'evidence_level_composite_marginal': marginal_benefit.get('evidence_level_composite_marginal', 0.0),
            'blind_spot_level_composite_marginal': marginal_benefit.get('blind_spot_level_composite_marginal', 0.0),
            'overall_composite_marginal_benefit_score': marginal_benefit.get('overall_composite_marginal_benefit_score', 0.0),
            
            'pr_curve_data_marginal': marginal_benefit.get('pr_curve_data_marginal', {}),
            'ndcg_at_k_marginal': marginal_benefit.get('ndcg_at_k_marginal', {}),

            'pr_curve_data': perf_metrics.get('pr_curve_data', {}),
            'ndcg_at_k': perf_metrics.get('ndcg_at_k', {}),
        }
        
        return result
        
    except Exception as e:
        logger.error(f"提取指标失败 ({method}, {case_id}): {e}")
        logger.debug(f"原始数据结构: {result_data}")
        
        return {
            'method': method,
            'case_id': case_id,
            'case_scale': case_scale,
            'case_type': case_type,
            'evidence_p': 0.0,
            'evidence_r': 0.0,
            'evidence_f1': 0.0,
            'aucpr': 0.0,
            'ndcg10': 0.0,
            'ndcg20': 0.0,
            'evidence_precision_marginal': 0.0,
            'evidence_recall_marginal': 0.0,
            'f1_score_marginal': 0.0,
            'auc_pr_marginal': 0.0,
            'global_ndcg_at_10_marginal': 0.0,
            'global_ndcg_at_20_marginal': 0.0,
            'evidence_level_composite_marginal': 0.0,
            'blind_spot_level_composite_marginal': 0.0,
            'overall_composite_marginal_benefit_score': 0.0,
            'pr_curve_data': {},
            'ndcg_at_k': {},
        }

def generate_visualization_data(case_id: str, results_base_dir: Path, 
                              case_scale: str = "smallcase", case_type: str = "Mixed",
                              output_dir: Path = None) -> Dict[str, Any]:
    """
    为指定案例生成认知不确定性地图的可视化数据
    
    Args:
        case_id: 案例ID
        results_base_dir: 结果基础目录
        case_scale: 案例规模
        case_type: 案例类型
        output_dir: 输出目录
    
    Returns:
        可视化数据字典
    """
    logger.info(f"为案例 {case_id} 生成可视化数据...")
    
    if output_dir is None:
        output_dir = results_base_dir / "visualization_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        case_dir = results_base_dir / case_scale / case_type / case_id
        bcsa_dir = case_dir / "BCSA_Analysis" / "1_Final_Results" / "pc_vgae_audit"
        
        uncertainty_map_file = None
        possible_paths = [
            bcsa_dir / "aggregated_uncertainty_map.json",
            case_dir / "BCSA_Analysis" / "1_Final_Results" / "aggregated_uncertainty_map.json",
            case_dir / "BCSA_Analysis" / "aggregated_uncertainty_map.json",
            bcsa_dir / "final_aggregated_uncertainty_map.json"
        ]
        
        for path in possible_paths:
            if path.exists():
                uncertainty_map_file = path
                logger.debug(f"找到不确定性地图文件: {path}")
                break
        
        if uncertainty_map_file is None:
            logger.warning(f"未找到不确定性地图文件，尝试的路径:")
            for path in possible_paths:
                logger.warning(f"  - {path}")
            return {'status': 'failed', 'error': 'uncertainty_map_not_found'}
        
        with open(uncertainty_map_file, 'r', encoding='utf-8') as f:
            uncertainty_data = json.load(f)
        
        case_data_dir = Path("./seek_data_v3_deep_enhanced/cases") / case_scale / case_type / case_id
        
        g_init_file = case_data_dir / "G_init.json"
        g_true_file = case_data_dir / "G_true.json"
        
        graph_data = {}
        if g_init_file.exists():
            with open(g_init_file, 'r', encoding='utf-8') as f:
                graph_data['G_init'] = json.load(f)
        
        if g_true_file.exists():
            with open(g_true_file, 'r', encoding='utf-8') as f:
                graph_data['G_true'] = json.load(f)
        
        visualization_data = {
            'case_id': case_id,
            'uncertainty_map': uncertainty_data,
            'graph_data': graph_data,
            'edges_with_uncertainty': [],
            'missing_edges': [],
            'existing_edges': []
        }
        
        if isinstance(uncertainty_data, list):
            logger.debug("不确定性数据是列表格式，直接遍历")
            for uncertainty_info in uncertainty_data:
                if isinstance(uncertainty_info, dict):
                    source_id = uncertainty_info.get('source_id', '')
                    target_id = uncertainty_info.get('target_id', '')
                    edge_id = f"{source_id}->{target_id}"
                    
                    edge_data = {
                        'edge_id': edge_id,
                        'uncertainty_score': uncertainty_info.get('uncertainty_score', 0.0),
                        'confidence': uncertainty_info.get('confidence', 0.0),
                        'source_id': source_id,
                        'target_id': target_id,
                        'edge_type': uncertainty_info.get('edge_type', 'unknown')
                    }
                    
                    if edge_data['uncertainty_score'] > 0.7:
                        visualization_data['missing_edges'].append(edge_data)
                    elif edge_data['uncertainty_score'] > 0.3:
                        visualization_data['edges_with_uncertainty'].append(edge_data)
                    else:
                        visualization_data['existing_edges'].append(edge_data)
        
        elif isinstance(uncertainty_data, dict):
            logger.debug("不确定性数据是字典格式，使用items()遍历")
            for edge_id, uncertainty_info in uncertainty_data.items():
                if isinstance(uncertainty_info, dict):
                    edge_data = {
                        'edge_id': edge_id,
                        'uncertainty_score': uncertainty_info.get('uncertainty_score', 0.0),
                        'confidence': uncertainty_info.get('confidence', 0.0),
                        'source_id': uncertainty_info.get('source_id', ''),
                        'target_id': uncertainty_info.get('target_id', ''),
                        'edge_type': uncertainty_info.get('edge_type', 'unknown')
                    }
                    
                    if edge_data['uncertainty_score'] > 0.7:
                        visualization_data['missing_edges'].append(edge_data)
                    elif edge_data['uncertainty_score'] > 0.3:
                        visualization_data['edges_with_uncertainty'].append(edge_data)
                    else:
                        visualization_data['existing_edges'].append(edge_data)
        else:
            logger.warning(f"不确定性数据格式无效: {type(uncertainty_data)}")
            return {'status': 'failed', 'error': 'invalid_uncertainty_data_format'}
        
        vis_data_file = output_dir / f"visualization_data_{case_id}.json"
        with open(vis_data_file, 'w', encoding='utf-8') as f:
            json.dump(visualization_data, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"可视化数据已保存: {vis_data_file}")
        logger.info(f"  - 高不确定性边: {len(visualization_data['missing_edges'])}")
        logger.info(f"  - 中等不确定性边: {len(visualization_data['edges_with_uncertainty'])}")
        logger.info(f"  - 低不确定性边: {len(visualization_data['existing_edges'])}")
        
        return {
            'status': 'success',
            'case_id': case_id,
            'output_file': str(vis_data_file),
            'summary': {
                'high_uncertainty_edges': len(visualization_data['missing_edges']),
                'medium_uncertainty_edges': len(visualization_data['edges_with_uncertainty']),
                'low_uncertainty_edges': len(visualization_data['existing_edges'])
            }
        }
        
    except Exception as e:
        logger.error(f"生成可视化数据失败: {e}")
        return {'status': 'failed', 'error': str(e)}
    
    
        
        
        
        
        
        
        
    
    
    

def generate_comprehensive_report(df: pd.DataFrame, summary_tables: Dict[str, pd.DataFrame], 
                                 method_analysis: Dict[str, Any], output_dir: Path):
    """生成综合分析报告"""
    logger.info("生成综合分析报告...")
    
    report_lines = []
    
    report_lines.append("# BCSA与基线方法综合性能分析报告")
    report_lines.append(f"## 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    report_lines.append("## 数据概览")
    report_lines.append(f"- **分析案例数**: {df['case_id'].nunique()}")
    report_lines.append(f"- **对比方法数**: {df['method'].nunique()}")
    report_lines.append(f"- **总评估记录**: {len(df)}")
    report_lines.append(f"- **涵盖方法**: {', '.join(sorted(df['method'].unique()))}")
    report_lines.append("")
    
    if 'rankings' in summary_tables:
        report_lines.append("## 整体性能排名")
        report_lines.append("基于盲区召回率、加权F1分数、平均倒数排名的综合排名:")
        report_lines.append("")
        
        rankings = summary_tables['rankings'].sort_values('average_rank')
        for idx, (method, row) in enumerate(rankings.iterrows(), 1):
            report_lines.append(f"{idx}. **{method}** (平均排名: {row['average_rank']:.1f})")
        report_lines.append("")
    
    report_lines.append("## 关键指标宏平均对比")
    
    key_metrics = [
        ('blind_spot_recall', '盲区召回率 (BSR)'),
        ('weighted_f1_score', '加权F1分数'),
        ('mean_reciprocal_rank', '平均倒数排名 (MRR)')
    ]
    
    for metric, description in key_metrics:
        report_lines.append(f"### {description}")
        method_means = df.groupby('method')[metric].mean().sort_values(ascending=False)
        for method, score in method_means.items():
            status = "🏆" if score == method_means.max() else "📊"
            report_lines.append(f"- {status} **{method}**: {score:.3f}")
        report_lines.append("")
    
    report_lines.append("## 方法特点分析")
    for method, analysis in method_analysis.items():
        report_lines.append(f"### {method}")
        report_lines.append(f"- **稳定性得分**: {analysis['consistency_score']:.3f}")
        report_lines.append(f"- **平均发现数**: {analysis['avg_findings_count']:.1f}")
        report_lines.append(f"- **主要特点**: {', '.join(analysis['characteristics'])}")
        report_lines.append("")
    
    report_lines.append("## 改进建议")
    
    bsr_ranking = df.groupby('method')['blind_spot_recall'].mean().sort_values(ascending=False)
    best_method = bsr_ranking.index[0]
    worst_method = bsr_ranking.index[-1]
    
    report_lines.append(f"1. **优势方法学习**: {best_method} 在盲区召回率方面表现最佳 ({bsr_ranking.iloc[0]:.3f})，值得深入研究其机制")
    report_lines.append(f"2. **劣势方法改进**: {worst_method} 在盲区召回率方面需要改进 ({bsr_ranking.iloc[-1]:.3f})")
    
    stability_scores = {method: analysis['consistency_score'] for method, analysis in method_analysis.items()}
    most_stable = max(stability_scores, key=stability_scores.get)
    least_stable = min(stability_scores, key=stability_scores.get)
    
    report_lines.append(f"3. **稳定性优化**: {most_stable} 最稳定，{least_stable} 需要提升跨案例一致性")
    report_lines.append("")
    
    report_content = '\n'.join(report_lines)
    report_file = output_dir / "comprehensive_analysis_report.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    logger.info(f"综合分析报告已保存: {report_file}")


def create_summary_tables(df: pd.DataFrame, output_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    创建6个核心指标对比表格 - 支持2x3布局的版本（包含标准差）
    """
    logger.info("生成6个核心指标对比表格...")

    summary_tables = {}

    core_metrics = list(CORE_METRICS_6.keys())

    if df.empty:
        logger.warning("DataFrame为空，跳过表格生成")
        return summary_tables

    available_metrics = [metric for metric in core_metrics if metric in df.columns]

    if not available_metrics:
        logger.warning("未找到任何核心指标列")
        return summary_tables

    macro_avg_table = df.groupby('method')[available_metrics].mean().round(4)

    macro_std_table = df.groupby('method')[available_metrics].std().round(4)

    headers = []
    metric_mapping = {
        'evidence_p': ('任务一', '证据 P'),
        'evidence_r': ('任务一', '证据 R'),
        'evidence_f1': ('任务一', '证据 F1'),
        'aucpr': ('任务一', 'AUC-PR'),
        'ndcg15': ('任务二', 'nDCG@15'),
        'ndcg25': ('任务二', 'nDCG@25')
    }

    for metric in available_metrics:
        if metric in metric_mapping:
            headers.append(metric_mapping[metric])

    macro_avg_table.columns = pd.MultiIndex.from_tuples(headers, names=['任务', '指标'])

    macro_std_table.columns = pd.MultiIndex.from_tuples(headers, names=['任务', '指标'])

    excel_file = output_dir / "core_metrics_summary.xlsx"
    try:
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            macro_avg_table.to_excel(writer, sheet_name='宏平均')
            macro_std_table.to_excel(writer, sheet_name='宏标准差')

        logger.info(f"6个核心指标表格（含标准差）已保存: {excel_file}")
        summary_tables['core_metrics'] = macro_avg_table
        summary_tables['core_metrics_std'] = macro_std_table
    except Exception as e:
        logger.error(f"保存Excel文件失败: {e}")

    csv_avg_file = output_dir / "core_metrics_macro_average.csv"
    try:
        macro_avg_table.to_csv(csv_avg_file)
        logger.info(f"核心指标宏平均已保存: {csv_avg_file}")
        summary_tables['core_metrics_csv_avg'] = macro_avg_table
    except Exception as e:
        logger.error(f"保存平均值CSV文件失败: {e}")

    csv_std_file = output_dir / "core_metrics_macro_std.csv"
    try:
        macro_std_table.to_csv(csv_std_file)
        logger.info(f"核心指标宏标准差已保存: {csv_std_file}")
        summary_tables['core_metrics_csv_std'] = macro_std_table
    except Exception as e:
        logger.error(f"保存标准差CSV文件失败: {e}")

    return summary_tables


def plot_unified_overview(df: pd.DataFrame, output_dir: Path, preset_methods):
    """绘制2x3布局的统一概览图表：任务一4个指标（左侧），任务二2个指标（右侧）"""
    logger.info("生成2x3统一概览图表...")
    
    if df.empty:
        logger.warning("DataFrame为空，跳过统一概览绘图")
        return

    global PRESET_METHODS
    if 'PRESET_METHODS' in globals():
        preset_methods = PRESET_METHODS
    else:
        preset_methods = [
            'PCGATE', 'PCVGAE', 'CommonNeighbors', 'Peter-Clark',
            'GAT', 'GATE', 'VGAE', 'No_Prompt (PCVGAE)', 'No_Prompt (PCGATE)', 'No_Audit'
        ]
    task1_metrics = TASK1_METRICS_6
    task2_metrics = TASK2_METRICS_6
    
    available_task1 = [m for m in task1_metrics.keys() if m in df.columns]
    available_task2 = [m for m in task2_metrics.keys() if m in df.columns]
    
    if not available_task1 and not available_task2:
        logger.warning("未找到任何可绘制的指标")
        return
    
    df_filtered = df[df['method'].isin(preset_methods)].copy()
    if df_filtered.empty:
        logger.warning("没有匹配预设方法的数据")
        return
    
    unique_methods = [method for method in preset_methods if method in df_filtered['method'].unique()]
    
    method_styles = get_method_styles(unique_methods)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Core Metrics Comparison Overview', fontsize=24, fontweight='bold', y=0.98)
    
    fig.text(0.325, 0.93, 'Task 1: Blind Spot Localization Performance', ha='center', va='top', 
             fontsize=20, fontweight='bold', color='darkblue')
    fig.text(0.83, 0.93, 'Task 2: Ranking Performance', ha='center', va='top', 
             fontsize=20, fontweight='bold', color='darkgreen')
    
    method_averages = df_filtered.groupby('method')[available_task1 + available_task2].mean()
    
    task1_positions = [(0,0), (1,0), (0,1), (1,1)]
    for i, (metric, title) in enumerate(task1_metrics.items()):
        if metric in available_task1 and i < len(task1_positions):
            row, col = task1_positions[i]
            ax = axes[row, col]
            
            values = [method_averages.loc[method, metric] if method in method_averages.index else 0 
                     for method in unique_methods]
            
            bars = ax.bar(range(len(unique_methods)), values, 
                         color=[method_styles.get(method, {}).get('color', 'gray') for method in unique_methods],
                         alpha=0.8)
            
            ax.set_title(f'{title}', fontsize=18, fontweight='bold')
            ax.set_ylabel('Score', fontsize=16)
            ax.set_xticks(range(len(unique_methods)))
            ax.set_xticklabels(unique_methods, rotation=45, ha='right', fontsize=16)
            ax.tick_params(axis='y', labelsize=16)
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim(0, 1.05)
            
            for bar, value in zip(bars, values):
                if value > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom', fontsize=12)
    
    task2_positions = [(0,2), (1,2)]
    for i, (metric, title) in enumerate(task2_metrics.items()):
        if metric in available_task2 and i < len(task2_positions):
            row, col = task2_positions[i]
            ax = axes[row, col]
            
            values = [method_averages.loc[method, metric] if method in method_averages.index else 0 
                     for method in unique_methods]
            
            bars = ax.bar(range(len(unique_methods)), values, 
                         color=[method_styles.get(method, {}).get('color', 'gray') for method in unique_methods],
                         alpha=0.8)
            
            ax.set_title(f'{title}', fontsize=18, fontweight='bold')
            ax.set_ylabel('Score', fontsize=16)
            ax.set_xticks(range(len(unique_methods)))
            ax.set_xticklabels(unique_methods, rotation=45, ha='right', fontsize=16)
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim(0, 1.05)
            
            for bar, value in zip(bars, values):
                if value > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom', fontsize=12)
    
    line = plt.Line2D((0.665, 0.665), (0.12, 0.88), transform=fig.transFigure, 
                     color="gray", linestyle='--', linewidth=2, alpha=0.7)
    fig.add_artist(line)
    
    legend_elements = [plt.Rectangle((0, 0), 1, 1, 
                                   facecolor=method_styles.get(method, {}).get('color', 'gray'), 
                                   alpha=0.8, label=method) for method in unique_methods]
    ncol = min(len(unique_methods), 10)  # 最多6列，超过就换行
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.02), 
              fontsize=16, framealpha=0.9, ncol=ncol)
    
    plt.tight_layout(rect=[0, 0.08, 1, 0.92])  # 调整边距为标题和图例留空间
    
    chart_file = output_dir / "unified_overview_comparison.png"
    chart_file_pdf = output_dir / "unified_overview_comparison.pdf"
    plt.savefig(chart_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(chart_file_pdf, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"2x3统一概览柱状图已保存: {chart_file}")
    logger.info(f"2x3统一概览柱状图已保存: {chart_file_pdf}")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Core Metrics Stability Overview - Cross-Case Performance', fontsize=16, fontweight='bold', y=0.98)
    
    fig.text(0.325, 0.93, 'Task 1: Evidence Discovery Performance', ha='center', va='top', 
             fontsize=14, fontweight='bold', color='darkblue')
    fig.text(0.83, 0.93, 'Task 2: Ranking Performance', ha='center', va='top', 
             fontsize=14, fontweight='bold', color='darkgreen')
    
    for i, (metric, title) in enumerate(task1_metrics.items()):
        if metric in available_task1 and i < len(task1_positions):
            row, col = task1_positions[i]
            ax = axes[row, col]
            
            for method in unique_methods:
                method_data = df_filtered[df_filtered['method'] == method].copy()
                if method_data.empty or metric not in method_data.columns:
                    continue
                
                method_data = method_data.sort_values('case_id')
                style = method_styles.get(method, {})
                
                ax.plot(range(len(method_data)), method_data[metric], 
                       color=style.get('color', 'gray'),
                       marker=style.get('marker', 'o'), 
                       linestyle=style.get('linestyle', '-'),
                       linewidth=style.get('linewidth', 1.5), 
                       markersize=style.get('markersize', 5), 
                       alpha=0.8, clip_on=False)
            
            ax.set_title(f'{title}', fontsize=12, fontweight='bold')
            ax.set_ylabel('Score', fontsize=10)
            ax.set_xlabel('Case Index', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1.05)
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    for i, (metric, title) in enumerate(task2_metrics.items()):
        if metric in available_task2 and i < len(task2_positions):
            row, col = task2_positions[i]
            ax = axes[row, col]
            
            for method in unique_methods:
                method_data = df_filtered[df_filtered['method'] == method].copy()
                if method_data.empty or metric not in method_data.columns:
                    continue
                
                method_data = method_data.sort_values('case_id')
                style = method_styles.get(method, {})
                
                ax.plot(range(len(method_data)), method_data[metric], 
                       color=style.get('color', 'gray'),
                       marker=style.get('marker', 'o'), 
                       linestyle=style.get('linestyle', '-'),
                       linewidth=style.get('linewidth', 1.5), 
                       markersize=style.get('markersize', 5), 
                       alpha=0.8, clip_on=False)
            
            ax.set_title(f'{title}', fontsize=12, fontweight='bold')
            ax.set_ylabel('Score', fontsize=10)
            ax.set_xlabel('Case Index', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1.05)
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    line = plt.Line2D((0.665, 0.665), (0.12, 0.88), transform=fig.transFigure, 
                     color="gray", linestyle='--', linewidth=2, alpha=0.7)
    fig.add_artist(line)
    
    legend_elements = [plt.Line2D([0], [0], 
                                 color=method_styles.get(method, {}).get('color', 'gray'),
                                 marker=method_styles.get(method, {}).get('marker', 'o'),
                                 linestyle=method_styles.get(method, {}).get('linestyle', '-'),
                                 label=method, 
                                 linewidth=method_styles.get(method, {}).get('linewidth', 1.5), 
                                 markersize=6) 
                      for method in unique_methods]
    ncol = min(len(unique_methods), 8)  # 最多6列，超过就换行
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.02), 
              fontsize=10, framealpha=0.9, ncol=ncol)
    
    plt.tight_layout(rect=[0, 0.08, 1, 0.92])  # 调整边距为标题和图例留空间
    
    chart_file = output_dir / "unified_overview_stability.png"
    chart_file_pdf = output_dir / "unified_overview_stability.pdf"
    plt.savefig(chart_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(chart_file_pdf, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"2x3统一概览稳定性图已保存: {chart_file}")
    logger.info(f"2x3统一概览稳定性图已保存: {chart_file_pdf}")


def _run_analysis_for_version(df: pd.DataFrame, version_output_dir: Path, version: str):
    """
    为指定版本运行分析的辅助函数 - 使用新的2x3统一绘图函数和6指标体系
    """
    version_output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"开始处理{version}分析...")
    
    try:
        summary_tables = create_summary_tables(df, version_output_dir)
        logger.info(f"✅ 核心指标表格生成完成")
    except Exception as e:
        logger.error(f"生成核心指标表格失败: {e}")
        summary_tables = {}
    
    try:
        plot_unified_overview(df, version_output_dir, preset_methods=PRESET_METHODS)
        logger.info(f"✅ 2x3统一概览图表生成完成")
    except Exception as e:
        logger.error(f"生成2x3统一概览图表失败: {e}")
    
    try:
        plot_pr_curves(df, version_output_dir)
        logger.info(f"✅ PR曲线对比图生成完成")
    except Exception as e:
        logger.error(f"生成PR曲线图表失败: {e}")
    
    try:
        plot_ndcg_curves(df, version_output_dir)
        logger.info(f"✅ NDCG@k曲线对比图生成完成")
    except Exception as e:
        logger.error(f"生成NDCG@k图表失败: {e}")
    
    try:
        method_analysis = analyze_method_characteristics_for_6_metrics(df, version_output_dir)
        logger.info(f"✅ 方法特点分析完成")
    except Exception as e:
        logger.error(f"方法特点分析失败: {e}")
        method_analysis = {}
    
    try:
        generate_comprehensive_report_for_6_metrics(df, summary_tables, method_analysis, version_output_dir)
        logger.info(f"✅ 综合报告生成完成")
    except Exception as e:
        logger.error(f"生成综合报告失败: {e}")
    
    return summary_tables, method_analysis


def analyze_method_characteristics_for_6_metrics(df: pd.DataFrame, output_dir: Path) -> Dict[str, Any]:
    """分析各方法的特点 - 基于6个核心指标的版本"""
    logger.info("分析各方法特点（基于6个核心指标）...")
    
    analysis = {}
    core_metrics = list(CORE_METRICS_6.keys())
    
    available_metrics = [m for m in core_metrics if m in df.columns]
    
    if not available_metrics:
        logger.warning("未找到可用于分析的核心指标")
        return analysis
    
    for method in df['method'].unique():
        method_data = df[df['method'] == method]
        
        stats = {
            'case_count': len(method_data),
        }
        
        for metric in available_metrics:
            if metric in method_data.columns:
                stats[f'avg_{metric}'] = method_data[metric].mean()
                stats[f'std_{metric}'] = method_data[metric].std()
        
        std_values = [stats.get(f'std_{metric}', 0) for metric in available_metrics 
                     if f'std_{metric}' in stats]
        if std_values:
            avg_std = np.mean(std_values)
            stats['consistency_score'] = max(0, 1 - avg_std)  # 标准差越小，一致性越高
        else:
            stats['consistency_score'] = 0.5
        
        characteristics = []
        
        if 'evidence_f1' in stats:
            if stats['avg_evidence_f1'] > 0.7:
                characteristics.append("高证据发现质量")
            elif stats['avg_evidence_f1'] < 0.3:
                characteristics.append("低证据发现质量")
        
        if 'aucpr' in stats:
            if stats['avg_aucpr'] > 0.7:
                characteristics.append("优秀PR性能")
            elif stats['avg_aucpr'] < 0.3:
                characteristics.append("较差PR性能")
        
        if 'ndcg10' in stats:
            if stats['avg_ndcg10'] > 0.6:
                characteristics.append("优秀排序性能")
            elif stats['avg_ndcg10'] < 0.3:
                characteristics.append("较差排序性能")
        
        if stats['consistency_score'] > 0.8:
            characteristics.append("高稳定性")
        elif stats['consistency_score'] < 0.6:
            characteristics.append("低稳定性")
        
        stats['characteristics'] = characteristics
        analysis[method] = stats
    
    analysis_file = output_dir / "method_characteristics_analysis_6metrics.json"
    with open(analysis_file, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False, default=str)
    
    logger.info(f"6个核心指标方法特点分析已保存: {analysis_file}")
    
    return analysis


def generate_comprehensive_report_for_6_metrics(df: pd.DataFrame, summary_tables: Dict[str, pd.DataFrame], 
                                               method_analysis: Dict[str, Any], output_dir: Path):
    """生成基于6个核心指标的综合分析报告"""
    logger.info("生成6个核心指标综合分析报告...")
    
    report_lines = []
    
    report_lines.append("# 6个核心指标综合性能分析报告")
    report_lines.append(f"## 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    report_lines.append("## 数据概览")
    report_lines.append(f"- **分析案例数**: {df['case_id'].nunique()}")
    report_lines.append(f"- **对比方法数**: {df['method'].nunique()}")
    report_lines.append(f"- **总评估记录**: {len(df)}")
    report_lines.append(f"- **涵盖方法**: {', '.join(sorted(df['method'].unique()))}")
    report_lines.append("")
    
    report_lines.append("## 6个核心指标宏平均对比")
    
    for metric_key, metric_name in CORE_METRICS_6.items():
        if metric_key in df.columns:
            report_lines.append(f"### {metric_name}")

            if metric_key in df.columns:
                method_means = df.groupby('method')[metric_key].mean().sort_values(ascending=False)
            else:
                continue  # 跳过不存在的指标
            for method, score in method_means.items():
                status = "🏆" if score == method_means.max() else "📊"
                report_lines.append(f"- {status} **{method}**: {score:.3f}")
            report_lines.append("")
    
    if method_analysis:
        report_lines.append("## 方法特点分析")
        for method, analysis in method_analysis.items():
            report_lines.append(f"### {method}")
            report_lines.append(f"- **稳定性得分**: {analysis.get('consistency_score', 0):.3f}")
            report_lines.append(f"- **案例数量**: {analysis.get('case_count', 0)}")
            characteristics = analysis.get('characteristics', [])
            if characteristics:
                report_lines.append(f"- **主要特点**: {', '.join(characteristics)}")
            report_lines.append("")
    
    report_lines.append("## 改进建议")
    
    if 'evidence_f1' in df.columns:
        f1_ranking = df.groupby('method')['evidence_f1'].mean().sort_values(ascending=False)
        if len(f1_ranking) > 0:
            best_method = f1_ranking.index[0]
            report_lines.append(f"1. **证据发现优势**: {best_method} 在证据F1分数方面表现最佳 ({f1_ranking.iloc[0]:.3f})，值得深入研究其机制")
    
    if 'aucpr' in df.columns:
        aucpr_ranking = df.groupby('method')['aucpr'].mean().sort_values(ascending=False)
        if len(aucpr_ranking) > 0:
            best_aucpr_method = aucpr_ranking.index[0]
            report_lines.append(f"2. **PR性能优势**: {best_aucpr_method} 在AUC-PR方面表现最佳 ({aucpr_ranking.iloc[0]:.3f})")
    
    if 'ndcg10' in df.columns:
        ndcg_ranking = df.groupby('method')['ndcg10'].mean().sort_values(ascending=False)
        if len(ndcg_ranking) > 0:
            best_ndcg_method = ndcg_ranking.index[0]
            report_lines.append(f"3. **排序性能优势**: {best_ndcg_method} 在nDCG@10方面表现最佳 ({ndcg_ranking.iloc[0]:.3f})")
    
    if method_analysis:
        stability_scores = {method: analysis.get('consistency_score', 0) 
                          for method, analysis in method_analysis.items()}
        if stability_scores:
            most_stable = max(stability_scores, key=stability_scores.get)
            least_stable = min(stability_scores, key=stability_scores.get)
            
            report_lines.append(f"4. **稳定性优化**: {most_stable} 最稳定，{least_stable} 需要提升跨案例一致性")
    
    report_lines.append("")
    
    report_content = '\n'.join(report_lines)
    report_file = output_dir / "comprehensive_analysis_report_6metrics.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    logger.info(f"6个核心指标综合分析报告已保存: {report_file}")


def load_results_from_data_file(data_file_path: str) -> pd.DataFrame:
    """
    从已有数据文件加载评估结果

    Args:
        data_file_path: 数据文件路径（支持CSV或Excel格式）

    Returns:
        包含评估结果的DataFrame
    """
    logger.info(f"从数据文件加载评估结果: {data_file_path}")

    data_path = Path(data_file_path)
    if not data_path.exists():
        logger.error(f"数据文件不存在: {data_file_path}")
        return pd.DataFrame()

    try:
        if data_path.suffix.lower() == '.csv':
            df = pd.read_csv(data_path)
        elif data_path.suffix.lower() in ['.xlsx', '.xls']:
            df = pd.read_excel(data_path)
        else:
            logger.error(f"不支持的文件格式: {data_path.suffix}")
            return pd.DataFrame()

        logger.info(f"成功加载 {len(df)} 条记录，涵盖 {df['method'].nunique()} 种方法")
        logger.info(f"发现的方法: {sorted(df['method'].unique())}")

        return df

    except Exception as e:
        logger.error(f"加载数据文件失败: {e}")
        return pd.DataFrame()

def run_unified_analysis(results_base_dir: Path = Path("./seek_data_v3_deep_enhanced/results"),
                        case_scale: str = "smallcase",
                        case_type: str = "Mixed",
                        output_dir: Path = None,
                        selected_methods: List[str] = None,
                        input_df: pd.DataFrame = None) -> Dict[str, Any]:
    """
    运行完整的统一分析流程 - 支持双版本输出，聚焦6个核心指标
    """
    if output_dir is None:
        output_dir = results_base_dir / "Quantitative_analysis" / f"{case_scale}_{case_type}"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("🚀 开始统一分析流程（聚焦6个核心指标，双版本输出）")
    logger.info(f"目标范围: {case_scale}/{case_type}")
    logger.info(f"输出目录: {output_dir}")
    if selected_methods:
        logger.info(f"指定对比方法: {selected_methods}")
    else:
        logger.info("对比所有发现的方法")
    
    if input_df is not None:
        logger.info("使用预加载的DataFrame进行分析")
        df = input_df
    else:
        logger.info("从原始文件加载评估结果...")
        try:
            df = load_all_evaluation_results(results_base_dir, case_scale, case_type)
        except Exception as e:
            logger.error(f"加载评估结果失败: {e}")
            return {'status': 'failed', 'error': f'data_loading_failed: {str(e)}'}
    
    if df.empty:
        logger.error("未找到任何评估结果，分析终止")
        return {'status': 'failed', 'error': 'no_data_found'}
    
    logger.info(f"✅ 成功加载 {len(df)} 条记录")
    
    expected_core_cols = list(CORE_METRICS_6.keys())
    available_core_cols = [col for col in expected_core_cols if col in df.columns]
    logger.info(f"📊 检测到6个核心指标: {available_core_cols}")
    
    tuple_cols = [col for col in df.columns if isinstance(col, tuple)]
    if tuple_cols:
        logger.warning(f"⚠️ 检测到未清理的元组列名: {tuple_cols[:3]}...")
    else:
        logger.info("✅ 数据结构简化成功，无元组列名")
    
    all_methods = sorted(df['method'].unique())
    logger.info(f"📊 发现的所有方法: {all_methods}")
    
    if selected_methods:
        missing_methods = [method for method in selected_methods if method not in all_methods]
        available_methods = [method for method in selected_methods if method in all_methods]
        
        if missing_methods:
            logger.warning(f"⚠️ 以下指定方法未找到: {missing_methods}")
        
        if not available_methods:
            logger.error("❌ 指定的方法都不存在，分析终止")
            return {'status': 'failed', 'error': 'selected_methods_not_found'}
        
        df = df[df['method'].isin(available_methods)].copy()
        logger.info(f"✅ 筛选后保留方法: {available_methods}")
        logger.info(f"✅ 筛选后数据量: {len(df)} 条记录")
    
    logger.info("📊 数据质量检查:")
    logger.info(f"  - 方法数: {df['method'].nunique()}")
    logger.info(f"  - 案例数: {df['case_id'].nunique()}")
    logger.info(f"  - 最终对比方法: {sorted(df['method'].unique())}")
    
    logger.info("📊 分离原始指标和边际效益指标...")

    df_original = df.copy()

    marginal_columns = []
    for col in df.columns:
        if isinstance(col, str) and col.endswith('_marginal'):
            marginal_columns.append(col)
        elif isinstance(col, tuple) and len(col) >= 2 and 'Marginal' in str(col):
            marginal_columns.append(col)

    if marginal_columns:
        logger.info(f"📊 发现边际效益相关列: {marginal_columns}")
        
        marginal_rename_mapping = {
            'evidence_precision_marginal': 'evidence_p',
            'evidence_recall_marginal': 'evidence_r',
            'f1_score_marginal': 'evidence_f1',
            'auc_pr_marginal': 'aucpr',
            'global_ndcg_at_15_marginal': 'ndcg15', # 修改
            'global_ndcg_at_25_marginal': 'ndcg25'  # 修改
        }
        
        existing_marginal_columns = {old_name: new_name for old_name, new_name in marginal_rename_mapping.items() 
                                if old_name in df.columns}
        
        if existing_marginal_columns:
            base_columns = ['method', 'case_id', 'case_scale', 'case_type']
            
            df_marginal = df[base_columns + list(existing_marginal_columns.keys())].copy()
            df_marginal = df_marginal.rename(columns=existing_marginal_columns)
            
            for core_metric in CORE_METRICS_6.keys():
                if core_metric not in df_marginal.columns:
                    df_marginal[core_metric] = 0.0
            
            plot_columns = ['pr_curve_data', 'ndcg_at_k']
            for col in plot_columns:
                marginal_col_name = f"{col}_marginal" # e.g., 'pr_curve_data_marginal'
                if marginal_col_name in df.columns:
                    df_marginal[col] = df[marginal_col_name]
                    logger.info(f"  ✓ 使用 '{marginal_col_name}' 作为边际效益版本的 '{col}' 数据")
                elif col in df.columns:
                    df_marginal[col] = df[col]
                    logger.warning(f"  ⚠️ 未找到 '{marginal_col_name}'，为 '{col}' 回退使用原始绘图数据")
                else:
                    df_marginal[col] = [{}] * len(df_marginal)
            
            logger.info(f"✅ 边际效益数据重命名完成: {list(existing_marginal_columns.values())}")
            logger.info(f"📊 边际效益数据框形状: {df_marginal.shape}")
            logger.info(f"📊 边际效益数据列: {sorted(df_marginal.columns.tolist())}")
        else:
            logger.warning("⚠️ 未找到需要重命名的边际效益列，将使用原始数据作为边际效益版本")
            df_marginal = df_original.copy()
    else:
        logger.warning("⚠️ 未找到边际效益相关列，将使用原始数据作为边际效益版本")
        df_marginal = df_original.copy()
    
    original_output_dir = output_dir / "original_metrics"
    marginal_output_dir = output_dir / "marginal_benefit_metrics"
    
    logger.info("--- 处理原始指标分析 ---")
    original_summary, original_analysis = _run_analysis_for_version(
        df_original, original_output_dir, 'original'
    )
    
    logger.info("--- 处理边际效益分析 ---")
    marginal_summary, marginal_analysis = _run_analysis_for_version(
        df_marginal, marginal_output_dir, 'marginal_benefit'
    )
    
    
        
            
    
    try:
        raw_data_file = output_dir / "raw_evaluation_data.csv"
        df.to_csv(raw_data_file, index=False, encoding='utf-8-sig')
        logger.info(f"💾 原始数据已保存: {raw_data_file}")
    except Exception as e:
        logger.error(f"保存原始数据失败: {e}")
        raw_data_file = None
    
    key_findings = {}
    try:
        if not df.empty and available_core_cols:
            for metric in available_core_cols:
                if metric in df.columns:
                    best_idx = df[metric].idxmax()
                    if pd.notna(best_idx):
                        key_findings[f'best_method_{metric}'] = df.loc[best_idx, 'method']
            
            if original_analysis:
                most_stable = max(original_analysis, 
                                key=lambda x: original_analysis[x].get('consistency_score', 0))
                key_findings['most_stable_method'] = most_stable
    except Exception as e:
        logger.error(f"计算关键发现失败: {e}")
    
    result = {
        'status': 'success',
        'timestamp': datetime.now().isoformat(),
        'analysis_mode': '8_core_metrics_dual_version',
        'data_summary': {
            'total_records': len(df),
            'methods_count': df['method'].nunique(),
            'cases_count': df['case_id'].nunique(),
            'methods': sorted(df['method'].unique().tolist()),
            'selected_methods': selected_methods if selected_methods else 'all_available',
            'available_core_metrics': available_core_cols,
            'marginal_columns_found': len(marginal_columns) > 0
        },
        'output_files': {
            'raw_data': str(raw_data_file) if raw_data_file else None,
            'original_core_metrics_summary': str(original_output_dir / "core_metrics_summary.xlsx"),
            'original_unified_overview_comparison': str(original_output_dir / "unified_overview_comparison.png"),
            'original_unified_overview_stability': str(original_output_dir / "unified_overview_stability.png"),
            'original_pr_curves': str(original_output_dir / "pr_curves_comparison_fixed.png"),
            'original_ndcg_curves': str(original_output_dir / "ndcg_at_k_comparison.png"),
            'original_characteristics': str(original_output_dir / "method_characteristics_analysis_6metrics.json"),
            'original_report': str(original_output_dir / "comprehensive_analysis_report_6metrics.md"),
            'marginal_core_metrics_summary': str(marginal_output_dir / "core_metrics_summary.xlsx"),
            'marginal_unified_overview_comparison': str(marginal_output_dir / "unified_overview_comparison.png"),
            'marginal_unified_overview_stability': str(marginal_output_dir / "unified_overview_stability.png"),
            'marginal_pr_curves': str(marginal_output_dir / "pr_curves_comparison_fixed.png"),
            'marginal_ndcg_curves': str(marginal_output_dir / "ndcg_at_k_comparison.png"),
            'marginal_characteristics': str(marginal_output_dir / "method_characteristics_analysis_6metrics.json"),
            'marginal_report': str(marginal_output_dir / "comprehensive_analysis_report_6metrics.md")
        },
        'key_findings': key_findings,
        'dual_version_analysis': {
            'original_metrics_dir': str(original_output_dir),
            'marginal_benefit_metrics_dir': str(marginal_output_dir)
        }
    }
    
    try:
        metadata_file = output_dir / "analysis_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        result['output_files']['metadata'] = str(metadata_file)
    except Exception as e:
        logger.error(f"保存分析元数据失败: {e}")
    
    logger.info("✅ 双版本统一分析流程完成")
    logger.info(f"📊 处理了 {len(df)} 条记录，涵盖 {df['method'].nunique()} 种方法")
    logger.info(f"📁 输出目录: {output_dir}")
    logger.info(f"🎯 8个核心指标: {', '.join(available_core_cols)}")
    logger.info(f"📂 原始指标分析: {original_output_dir}")
    logger.info(f"📂 边际效益分析: {marginal_output_dir}")
    
    if key_findings:
        logger.info("🏆 关键发现:")
        for finding, method in key_findings.items():
            logger.info(f"  - {finding}: {method}")
    
    logger.info(f"📋 查看原始指标报告: {result['output_files']['original_report']}")
    logger.info(f"📋 查看边际效益报告: {result['output_files']['marginal_report']}")
    
    return result







        
        


        
        
        
            


def main():
    """主函数 - 已修改为交互式模式，支持从头分析或从文件分析"""
    import argparse

    print("=" * 80)
    print("📊 BCSA 定量实验统一分析器")
    print("=" * 80)

    global PRESET_METHODS
    PRESET_METHODS = [
        'PCGATE', 'PCVGAE', 'CommonNeighbors', 'Peter-Clark',
        'GAT', 'GATE', 'VGAE', 'CDHC', 'IF-PDAG'
    ]

    print("\nPlease select an analysis mode:")
    print("  [1] Run full analysis from scratch (loads raw result files)")
    print("  [2] Analyze from an existing data file (e.g., .csv or .xlsx)")
    
    mode_choice = input("Enter your choice [1]: ").strip() or "1"
    
    df_to_analyze = None
    analysis_params = {}

    try:
        if mode_choice == '1':
            print("\n--- Mode 1: Run full analysis from scratch ---")
            case_scale, case_type, available_cases = select_case_scope()
            if not available_cases:
                print("❌ 未找到可用案例")
                return
            
            print(f"\n发现 {len(available_cases)} 个案例。此脚本将以批量模式分析指定范围内的所有案例。")
            confirm = input("是否继续? (Y/n): ").strip().lower()
            if confirm not in ['', 'y', 'yes']:
                print("分析已取消。")
                return

            analysis_params = {"case_scale": f"{case_scale}case", "case_type": case_type}
            df_to_analyze = None  # Signal to load from scratch

        elif mode_choice == '2':
            print("\n--- Mode 2: Analyze from existing data file ---")
            print("首先，请指定案例范围以构建默认文件路径和输出目录。")
            case_scale, case_type, _ = select_case_scope()

            default_path = f"./seek_data_v3_deep_enhanced/results/Quantitative_analysis/{case_scale}case_{case_type}/raw_evaluation_data.csv"
            
            print(f"\n请输入数据文件路径 (.csv or .xlsx)。")
            data_file_path = input(f"按 Enter 使用默认路径 [{default_path}]: ").strip()
            if not data_file_path:
                data_file_path = default_path

            df_to_analyze = load_results_from_data_file(data_file_path)
            if df_to_analyze.empty:
                return

            analysis_params = {"case_scale": f"{case_scale}case", "case_type": case_type}
        
        else:
            print("❌ 无效选择，程序退出。")
            return

        all_methods_choice = input("\n是否分析所有发现的方法 (而不是预设列表)? (y/N): ").strip().lower()
        if all_methods_choice in ['y', 'yes']:
            selected_methods = None
            print("🎯 对比所有发现的方法。")
        else:
            selected_methods = PRESET_METHODS
            print(f"🎯 使用预设方法列表: {', '.join(PRESET_METHODS)}")

        results_base_dir = Path('./seek_data_v3_deep_enhanced/results')
        
        print(f"\n📂 开始分析...")
        print(f"分析范围: {analysis_params.get('case_scale')}/{analysis_params.get('case_type')}")
        
        result = run_unified_analysis(
            results_base_dir=results_base_dir,
            case_scale=analysis_params.get("case_scale"),
            case_type=analysis_params.get("case_type"),
            output_dir=None,
            selected_methods=selected_methods,
            input_df=df_to_analyze
        )
        
        if result['status'] == 'success':
            methods_info = result['data_summary']['methods']
            print(f"\n🎉 双版本分析完成!")
            print(f"📊 实际对比了 {len(methods_info)} 种方法: {', '.join(methods_info)}")
            print(f"📂 原始指标分析: {result['dual_version_analysis']['original_metrics_dir']}")
            print(f"📂 边际效益分析: {result['dual_version_analysis']['marginal_benefit_metrics_dir']}")
            print(f"📋 原始指标报告: {result['output_files']['original_report']}")
            print(f"📋 边际效益报告: {result['output_files']['marginal_report']}")
        else:
            print(f"\n❌ 分析失败: {result.get('error', 'Unknown error')}")
            
    except KeyboardInterrupt:
        print(f"\n⏹️ 用户中断分析")
    except Exception as e:
        print(f"\n❌ 分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
