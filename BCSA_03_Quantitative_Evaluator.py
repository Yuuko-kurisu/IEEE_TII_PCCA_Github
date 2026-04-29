#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BCSA定量评估器 - 客观的记分卡
输入：案例输出目录
输出：核心性能指标的JSON报告
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(case_directory: Path, case_id: str, base_data_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    加载评估所需的所有文件
    
    Args:
        case_directory: 输出案例目录路径 (conditioned_uncertainty_analysis下的案例目录)
        case_id: 案例ID
        
    Returns:
        包含所有加载数据的字典
    """
    logger.info(f"开始加载案例数据: {case_directory} (case_id: {case_id})")
    
    output_base = case_directory
    

    case_type, scale = case_id.split('_')[:2]
    base_data_path = Path(base_data_dir) if base_data_dir is not None else Path("./seek_data_v3_deep_enhanced/cases")
    original_case_base = base_data_path / f"{scale}case" / case_type / case_id
    
    data = {}
    
    gt_file = original_case_base / "processed_ground_truth.json"
    if gt_file.exists():
        with open(gt_file, 'r', encoding='utf-8') as f:
            data['gt_data'] = json.load(f)
        
        logger.debug(f"GT数据结构: {type(data['gt_data'])}")
        if isinstance(data['gt_data'], dict):
            logger.debug(f"GT数据键: {list(data['gt_data'].keys())}")
            evidence_zones = data['gt_data'].get('evidence_zones', [])
            logger.debug(f"evidence_zones类型: {type(evidence_zones)}")
            if isinstance(evidence_zones, (list, tuple)) and len(evidence_zones) > 0:
                logger.debug(f"evidence_zones长度: {len(evidence_zones)}")
                logger.debug(f"第一个zone类型: {type(evidence_zones[0])}")
            elif isinstance(evidence_zones, dict) and evidence_zones:
                logger.debug(f"evidence_zones是字典，键: {list(evidence_zones.keys())}")
        
        logger.info(f"✓ 加载Ground Truth数据")
    else:
        logger.error(f"❌ Ground Truth文件不存在: {gt_file}")
        data['gt_data'] = {}
    









    possible_results_files = [
        "final_aggregated_uncertainty_map.json",
        "findings.json"
    ]

    findings_data = []
    file_loaded = False

    for filename in possible_results_files:
        filepath = output_base / filename  # 'output_base' 是结果目录的Path对象
        if filepath.exists():
            logger.info(f"✓ 发现并加载结果文件: {filename}")
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = json.load(f)
                    if isinstance(content, list):
                        findings_data = content
                    else:
                        logger.warning(f"文件 {filename} 内容不是预期的列表格式，已跳过。")
                file_loaded = True
                break
            except json.JSONDecodeError:
                logger.error(f"❌ 解析JSON文件失败: {filepath}")
                break # 文件损坏，停止尝试

    if not file_loaded:
        logger.warning(f"⚠️ 在目录 {output_base} 中未找到任何有效的发现文件。")

    data['findings'] = findings_data
    
    ckg_file = original_case_base / "causal_knowledge_graph.json"
    if ckg_file.exists():
        with open(ckg_file, 'r', encoding='utf-8') as f:
            ckg = json.load(f)
        
        id_to_text = {}
        text_to_id = {}
        
        nodes_by_type = ckg.get('nodes_by_type', {})
        for node_type, nodes in nodes_by_type.items():
            for node in nodes:
                node_id = node.get('id', '')
                node_text = node.get('text', '')
                if node_id and node_text:
                    id_to_text[node_id] = node_text
                    text_to_id[node_text] = node_id
        
        data['node_mappings'] = {
            'id_to_text': id_to_text,
            'text_to_id': text_to_id
        }
        logger.info(f"✓ 构建节点映射: {len(id_to_text)} 个节点")
    else:
        logger.error(f"❌ CKG文件不存在: {ckg_file}")
        data['node_mappings'] = {'id_to_text': {}, 'text_to_id': {}}
    
    impact_file = output_base / "hypothesis_impact_report.json"
    if impact_file.exists():
        with open(impact_file, 'r', encoding='utf-8') as f:
            data['impact_data'] = json.load(f)
        impact_count = len(data['impact_data'].get('hypothesis_impacts', {}))
        logger.info(f"✓ 加载假设影响力数据: {impact_count} 个假设")
    else:
        logger.warning(f"⚠️ 假设影响力文件不存在: {impact_file}")
        data['impact_data'] = {}
    
    total_evidence = 0
    try:
        evidence_zones = data['gt_data'].get('evidence_zones', [])
        if isinstance(evidence_zones, list):
            for zone in evidence_zones:
                if isinstance(zone, dict):
                    evidence_items = zone.get('evidence_edges', [])
                    if isinstance(evidence_items, list):
                        total_evidence += len(evidence_items)
        elif isinstance(evidence_zones, dict):
            for zone_key, zone in evidence_zones.items():
                if isinstance(zone, dict):
                    evidence_items = zone.get('evidence_edges', [])
                    if isinstance(evidence_items, list):
                        total_evidence += len(evidence_items)
    except Exception as e:
        logger.error(f"❌ 计算证据总数时出错: {e}")
        total_evidence = 0
    
    logger.info(f"📊 数据加载完成:")
    logger.info(f"   - GT证据总数: {total_evidence}")
    logger.info(f"   - 审计发现总数: {len(data['findings'])}")
    logger.info(f"   - 假设影响力条目数: {len(data['impact_data'].get('hypothesis_impacts', {}))}")
    logger.info(f"   - 原始案例路径: {original_case_base}")
    logger.info(f"   - 输出结果路径: {output_base}")
    
    return data

def calculate_marginal_benefit_scores(metrics: Dict[str, Any], alpha: float = 0.4, 
                                    additional_vars: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """
    计算边际效益转化得分 - 证据为中心的评估体系
    
    核心思想：解决小样本评估脆弱性，应用边际效益递减原理
    公式：Scaled_Score = (raw_score) ^ α，其中 α < 1
    
    学术依据：基于信息论和经济学中的边际效益递减理论，
             从0到1的价值远大于从0.9到1.0的价值
    
    Args:
        metrics: 原始性能指标字典
        alpha: 缩放因子，默认0.6，可调节容错程度
        additional_vars: 额外的非核心变量字典（后续扩展用）
        
    Returns:
        包含转化后指标的字典
    """
    logger.info(f"开始计算边际效益转化得分（α={alpha}）- 证据为中心的评估体系...")
    
    scaled_scores = {}
    
    performance_metrics = metrics.get('performance_metrics', {})
    
    core_indicators = {
        'weighted_f1_score': '加权F1分数 (WF1) - 核心指标',
        'evidence_recall': '证据召回率 (ESR)',
        'evidence_precision': '证据精确率 (ESP)',
        'f1_score': '证据F1分数 (ESF1)',
        
        'evidence_mrr': '证据MRR (Evidence MRR) - 核心排序指标',
        'global_ndcg_at_10': '全局nDCG@10 (Global nDCG@10)',
        'global_ndcg_at_15': '全局nDCG@15 (Global nDCG@15)',
        'global_ndcg_at_20': '全局nDCG@20 (Global nDCG@20)',
        'global_ndcg_at_25': '全局nDCG@25 (Global nDCG@25)',
        
        'blind_spot_recall': '盲区召回率 (BSR) - 诊断指标',
        'blind_spot_precision': '盲区精确率 (BSP) - 诊断指标',
        'blind_spot_f1': '盲区F1分数 (BSF1) - 诊断指标',
        'average_zecr': '平均区域证据覆盖率 (ZECR) - 诊断指标',
        
        'weighted_recall': '加权召回率',
        'weighted_precision': '加权精确率',
        'auc_pr': 'AUC-PR',
        'average_true_positive_confidence': '真正例平均置信度'
    }
    
    for indicator, description in core_indicators.items():
        raw_score = performance_metrics.get(indicator, 0.0)
        
        raw_score = max(0.0, min(1.0, raw_score))
        

        import math
        def f(x, a):
            """
            计算函数 f(x) = log(1+a*x)/log(1+a)
            """
            if a <= 0 or a == -1:
                raise ValueError("参数a必须大于0且不等于-1")
            
            numerator = math.log(1 + a * x)
            denominator = math.log(1 + a)
            
            return numerator / denominator
        a = 5
        scaled_score = f(raw_score, a)

        
        scaled_scores[f'{indicator}_marginal'] = scaled_score
        
        logger.debug(f"{description}: {raw_score:.3f} → {scaled_score:.3f} "
                    f"(提升: {scaled_score - raw_score:+.3f})")
    

    logger.debug("转换绘图数据（nDCG@k 和 PR 曲线）的边际效益...")

    raw_ndcg_at_k = performance_metrics.get('ndcg_at_k', {})
    if raw_ndcg_at_k:
        ndcg_at_k_marginal = {
            k: (v ** alpha) for k, v in raw_ndcg_at_k.items()
        }
        scaled_scores['ndcg_at_k_marginal'] = ndcg_at_k_marginal
        logger.debug(f"  原始 nDCG@10: {raw_ndcg_at_k.get('10', 0):.3f} -> 边际效益: {ndcg_at_k_marginal.get('10', 0):.3f}")

    raw_pr_data = performance_metrics.get('pr_curve_data', {})
    if raw_pr_data and 'precision' in raw_pr_data and 'recall' in raw_pr_data:
        precision_marginal = [p ** alpha for p in raw_pr_data['precision']]
        pr_curve_data_marginal = {
            'precision': precision_marginal,
            'recall': raw_pr_data['recall'] # Recall (X轴) 保持不变
        }
        scaled_scores['pr_curve_data_marginal'] = pr_curve_data_marginal
        logger.debug("  PR 曲线的 precision 值已进行边际效益转换。")


    if additional_vars:
        logger.info(f"处理 {len(additional_vars)} 个额外变量...")
        for var_name, raw_value in additional_vars.items():
            normalized_value = max(0.0, min(1.0, raw_value))
            scaled_value = normalized_value ** alpha
            scaled_scores[f'{var_name}_marginal'] = scaled_value
            logger.debug(f"额外变量 {var_name}: {normalized_value:.3f} → {scaled_value:.3f}")
    
    
    core_finding_weights = {
        'weighted_f1_score_marginal': 0.4,      # WF1权重40% - 最重要
        'evidence_recall_marginal': 0.25,       # ESR权重25%
        'evidence_precision_marginal': 0.2,     # ESP权重20%
        'f1_score_marginal': 0.15               # ESF1权重15%
    }
    
    core_finding_composite = sum(scaled_scores.get(indicator, 0.0) * weight 
                                for indicator, weight in core_finding_weights.items())
    scaled_scores['core_finding_composite_marginal'] = core_finding_composite
    
    ranking_weights = {
        'evidence_mrr_marginal': 0.5,           # Evidence MRR权重50% - 核心排序指标
        'global_ndcg_at_10_marginal': 0.3,      # Global nDCG@10权重30%
        'global_ndcg_at_20_marginal': 0.2       # Global nDCG@20权重20%
    }
    
    ranking_composite = sum(scaled_scores.get(indicator, 0.0) * weight 
                           for indicator, weight in ranking_weights.items())
    scaled_scores['ranking_performance_composite_marginal'] = ranking_composite
    
    diagnostic_weights = {
        'blind_spot_recall_marginal': 0.4,      # BSR权重40%
        'average_zecr_marginal': 0.3,           # ZECR权重30%
        'blind_spot_precision_marginal': 0.2,   # BSP权重20%
        'blind_spot_f1_marginal': 0.1           # BSF1权重10%
    }
    
    diagnostic_composite = sum(scaled_scores.get(indicator, 0.0) * weight 
                              for indicator, weight in diagnostic_weights.items())
    scaled_scores['diagnostic_composite_marginal'] = diagnostic_composite
    
    overall_weights = {
        'weighted_f1_score_marginal': 0.30,         # 核心：发现质量
        'evidence_recall_marginal': 0.20,           # 核心：证据覆盖
        'evidence_mrr_marginal': 0.20,              # 核心：排序效率
        'global_ndcg_at_10_marginal': 0.10,         # 排序质量
        'evidence_precision_marginal': 0.10,        # 发现精度
        'blind_spot_recall_marginal': 0.05,         # 诊断：盲区覆盖
        'average_zecr_marginal': 0.05              # 诊断：区域覆盖
    }
    
    overall_composite = sum(scaled_scores.get(indicator, 0.0) * weight 
                           for indicator, weight in overall_weights.items())
    scaled_scores['overall_composite_marginal_benefit_score'] = overall_composite
    
    logger.info(f"✅ 边际效益转化完成（证据为中心版）:")
    
    logger.info(f"   核心发现能力:")
    logger.info(f"   - WF1: {performance_metrics.get('weighted_f1_score', 0):.1%} → {scaled_scores.get('weighted_f1_score_marginal', 0):.1%}")
    logger.info(f"   - ESR: {performance_metrics.get('evidence_recall', 0):.1%} → {scaled_scores.get('evidence_recall_marginal', 0):.1%}")
    logger.info(f"   - ESP: {performance_metrics.get('evidence_precision', 0):.1%} → {scaled_scores.get('evidence_precision_marginal', 0):.1%}")
    logger.info(f"   - ESF1: {performance_metrics.get('f1_score', 0):.1%} → {scaled_scores.get('f1_score_marginal', 0):.1%}")
    logger.info(f"   - 核心发现能力综合: {core_finding_composite:.1%}")
    
    logger.info(f"   排序性能:")
    logger.info(f"   - Evidence MRR: {performance_metrics.get('evidence_mrr', 0):.3f} → {scaled_scores.get('evidence_mrr_marginal', 0):.3f}")
    logger.info(f"   - Global nDCG@10: {performance_metrics.get('global_ndcg_at_10', 0):.3f} → {scaled_scores.get('global_ndcg_at_10_marginal', 0):.3f}")
    logger.info(f"   - Global nDCG@20: {performance_metrics.get('global_ndcg_at_20', 0):.3f} → {scaled_scores.get('global_ndcg_at_20_marginal', 0):.3f}")
    logger.info(f"   - 排序性能综合: {ranking_composite:.3f}")
    
    logger.info(f"   诊断性指标:")
    logger.info(f"   - BSR: {performance_metrics.get('blind_spot_recall', 0):.1%} → {scaled_scores.get('blind_spot_recall_marginal', 0):.1%}")
    logger.info(f"   - ZECR: {performance_metrics.get('average_zecr', 0):.1%} → {scaled_scores.get('average_zecr_marginal', 0):.1%}")
    logger.info(f"   - 诊断性指标综合: {diagnostic_composite:.1%}")
    
    logger.info(f"   - 总体综合边际效益得分: {overall_composite:.1%}")
    
    return scaled_scores

def _calculate_global_ndcg_at_k(ranked_items: List[Dict], k: int) -> float:
    """
    计算全局nDCG@k分数 - 标准版本
    """
    if not ranked_items or k <= 0:
        return 0.0

    actual_ranking = sorted(ranked_items, key=lambda x: x['score'], reverse=True)

    dcg = 0.0
    for i, item in enumerate(actual_ranking[:k]):
        relevance = item['relevance']
        if relevance > 0:
            dcg += relevance / np.log2(i + 2)

    ideal_ranking = sorted(ranked_items, key=lambda x: x['relevance'], reverse=True)
    idcg = 0.0
    for i, item in enumerate(ideal_ranking[:k]):
        relevance = item['relevance']
        if relevance > 0:
            idcg += relevance / np.log2(i + 2)

    ndcg = dcg / idcg if idcg > 0 else 0.0
    return ndcg

def _calculate_weighted_pr_auc(findings: List, evidence_matches: List, max_possible_weighted_score: float, 
                              all_evidence: List, decay_factor: float = 0.7) -> Tuple[float, Dict]:
    """
    计算混合加权PR-AUC：X轴为无权重召回率，Y轴为加权精确率
    对于同一finding匹配的第二个和后续证据，分数逐级衰减
    
    Args:
        findings: 发现列表
        evidence_matches: 证据匹配结果列表
        max_possible_weighted_score: 理论最高加权分数
        all_evidence: 所有GT证据列表
        decay_factor: 衰减因子，第n个匹配的权重为 decay_factor^(n-1)
        
    Returns:
        Tuple[float, Dict]: (auc_pr, pr_curve_data)
    """
    if not findings or not evidence_matches or max_possible_weighted_score <= 0:
        return 0.0, {}
    
    from collections import defaultdict
    import numpy as np
    
    finding_to_score = {}
    finding_to_weighted_tp = {}
    finding_to_hit_evidences = defaultdict(list)  # 记录每个finding命中的独立evidence
    finding_to_match_details = defaultdict(list)
    
    for finding in findings:
        finding_to_score[id(finding)] = finding.get('unified_score', 0.0)
        finding_to_weighted_tp[id(finding)] = 0.0
    
    evidence_to_id = {}
    for evidence in all_evidence:
        zone_id = evidence.get('zone_id', '')
        edge_key = evidence.get('edge_key', str(id(evidence)))
        unique_id = f"{zone_id}_{edge_key}"
        evidence_to_id[id(evidence)] = unique_id
    
    for match in evidence_matches:
        if match['best_finding'] is not None and match['best_match_score'] > 0:
            finding_id = id(match['best_finding'])
            evidence_id = evidence_to_id.get(id(match['evidence']))
            
            finding_to_match_details[finding_id].append({
                'score': match['best_match_score'],
                'evidence_id': evidence_id
            })
    
    for finding_id, match_details in finding_to_match_details.items():
        sorted_matches = sorted(match_details, key=lambda x: x['score'], reverse=True)
        
        weighted_score = 0.0
        unique_evidences = set()
        
        for i, match_detail in enumerate(sorted_matches):
            weight = decay_factor
            weighted_contribution = match_detail['score'] * weight
            weighted_score += weighted_contribution
            
            if match_detail['evidence_id']:
                unique_evidences.add(match_detail['evidence_id'])
        
        finding_to_weighted_tp[finding_id] = weighted_score
        finding_to_hit_evidences[finding_id] = list(unique_evidences)
    
    sorted_findings = sorted(findings, key=lambda f: finding_to_score[id(f)], reverse=True)
    
    cumulative_weighted_tp = 0.0
    seen_evidence_ids = set()  # 用于跟踪已发现的独立证据
    total_evidence_count = len(all_evidence)  # 证据总数
    pr_points = [{'recall': 0.0, 'precision': 1.0}]  # PR曲线起点
    
    for i, finding in enumerate(sorted_findings):
        rank = i + 1
        finding_id = id(finding)
        
        cumulative_weighted_tp += finding_to_weighted_tp.get(finding_id, 0.0)
        weighted_precision = cumulative_weighted_tp / rank if rank > 0 else 0.0

        if weighted_precision > 1:
            weighted_precision = 1 
        
        hit_evidences = finding_to_hit_evidences.get(finding_id, [])
        for ev_id in hit_evidences:
            seen_evidence_ids.add(ev_id)
        
        unweighted_recall = len(seen_evidence_ids) / total_evidence_count if total_evidence_count > 0 else 0.0
        
        pr_points.append({
            'recall': unweighted_recall,     # 无权重、基于计数的召回率
            'precision': weighted_precision  # 加权精确率
        })
    


    auc_pr = 0.0
    for i in range(1, len(pr_points)):
        recall_diff = pr_points[i]['recall'] - pr_points[i-1]['recall']
        if recall_diff > 0:  # 确保recall是递增的
            avg_precision = (pr_points[i]['precision'] + pr_points[i-1]['precision']) / 2
            auc_pr += avg_precision * recall_diff
    
    if len(pr_points) > 100:
        sample_indices = np.linspace(0, len(pr_points)-1, 100, dtype=int)
        sampled_points = [pr_points[i] for i in sample_indices]
    else:
        sampled_points = pr_points
    
    pr_curve_data = {
        'precision': [point['precision'] for point in sampled_points],
        'recall': [point['recall'] for point in sampled_points]
    }
    
    return auc_pr, pr_curve_data
def calculate_quantitative_metrics(data: Dict[str, Any], use_legacy_matching: bool = False) -> Dict[str, Any]:
    """
    计算所有定量指标 - 证据为中心的评估体系
    
    Args:
        data: 加载的所有数据
        
    Returns:
        包含所有指标的字典
    """
    logger.info("开始计算定量指标（证据为中心的评估体系）...")
    
    gt_data = data['gt_data']
    findings = data['findings']
    node_mappings = data['node_mappings']
    impact_data = data['impact_data']
    
    metrics = {}
    
    all_evidence = []
    evidence_zones_raw = gt_data.get('evidence_zones', [])
    
    if isinstance(evidence_zones_raw, dict):
        evidence_zones = list(evidence_zones_raw.values())
        logger.info(f"evidence_zones是字典，转换为列表处理，包含 {len(evidence_zones)} 个区域")
    elif isinstance(evidence_zones_raw, list):
        evidence_zones = evidence_zones_raw
        logger.info(f"evidence_zones是列表，包含 {len(evidence_zones)} 个区域")
    else:
        logger.error(f"❌ evidence_zones格式无效: {type(evidence_zones_raw)}")
        evidence_zones = []
    
    for zone in evidence_zones:
        if not isinstance(zone, dict):
            logger.warning(f"⚠️ 跳过非字典类型的zone: {type(zone)}")
            continue
            
        zone_evidence = zone.get('evidence_edges', [])
        if not isinstance(zone_evidence, list):
            logger.warning(f"⚠️ zone中的evidence_items不是列表: {type(zone_evidence)}")
            continue
            
        for evidence in zone_evidence:
            if isinstance(evidence, dict):
                evidence['zone_id'] = zone.get('zone_id', '')
                evidence['blind_spot_type'] = zone.get('blind_spot_type', '')
                all_evidence.append(evidence)
            else:
                logger.warning(f"⚠️ 跳过非字典类型的evidence: {type(evidence)}")
    
    total_evidence = len(all_evidence)
    total_findings = len(findings)
    

    legacy_matching_available = bool(data.get('impact_data') and data['impact_data'].get('hypothesis_impacts'))
    legacy_matching_enabled = bool(legacy_matching_available and use_legacy_matching)
    legacy_semantic_matches = 0
    importance_stats = {'high': 0, 'medium': 0, 'low': 0, 'unknown': 0}
    for evidence in all_evidence:
        importance = evidence.get('importance', 'unknown')
        if importance in importance_stats:
            importance_stats[importance] += 1
        else:
            importance_stats['unknown'] += 1
    
    logger.info(f"证据总数: {total_evidence}, 发现总数: {total_findings}")
    logger.info(f"证据重要性分布: {importance_stats}")
    
    if total_evidence == 0:
        logger.warning("⚠️ 没有找到有效的GT证据，返回空指标")
        return {
            'evaluation_config': {
                'matching_policy': 'legacy_compatibility' if legacy_matching_enabled else 'strict',
                'legacy_matching_available': legacy_matching_available,
                'legacy_matching_enabled': legacy_matching_enabled,
                'legacy_semantic_matches': legacy_semantic_matches
            },
            'basic_metrics': {
                'total_evidence': 0,
                'total_findings': total_findings,
                'true_positives': 0,
                'false_negatives': 0,
                'false_positives': total_findings,
                'importance_distribution': importance_stats
            },
            'performance_metrics': {
                'evidence_recall': 0.0,
                'evidence_precision': 0.0,
                'f1_score': 0.0,
                'blind_spot_recall': 0.0,
                'blind_spot_precision': 0.0,
                'blind_spot_f1': 0.0,
                'evidence_mrr': 0.0,  # 新指标
                'global_ndcg_at_10': 0.0,  # 新指标
                'global_ndcg_at_20': 0.0,  # 新指标
                'weighted_recall': 0.0,
                'weighted_precision': 0.0,
                'weighted_f1_score': 0.0,
                'average_zecr': 0.0
            },
            'blind_spot_analysis': {
                'total_blind_spots': len(evidence_zones),
                'detected_blind_spots': 0,
                'detection_rate': 0.0,
                'details': []
            },
            'evidence_analysis': {
                'total_matches': 0,
                'direct_hits': 0,
                'partial_hits': 0,
                'misses': 0,
                'high_importance_hits': 0,
                'medium_importance_hits': 0,
                'low_importance_hits': 0
            }
        }
    
    max_possible_weighted_score = 0.0
    for evidence in all_evidence:
        importance = evidence.get('importance', 'medium')
        
        if importance == 'high':
            weight = 1.5
        elif importance == 'medium':
            weight = 1.0
        else:  # 'low' 或其他
            weight = 0.8
        
        max_possible_weighted_score += 1.0 * weight
    
    logger.info(f"理论最高加权分数: {max_possible_weighted_score}")
    
    evidence_matches = []
    total_weighted_tp = 0.0  # 加权真正例总分
    unweighted_tp_count = 0  # 未加权真正例计数
    
    importance_hits = {'high': 0, 'medium': 0, 'low': 0, 'unknown': 0}
    
    for evidence in all_evidence:
        best_match_score = 0.0
        best_finding = None
        
        for finding in findings:
            match_score = calculate_match_score(finding, evidence, node_mappings)
            if match_score > best_match_score:
                best_match_score = match_score
                best_finding = finding
        
        evidence_matches.append({
            'evidence': evidence,
            'best_match_score': best_match_score,
            'best_finding': best_finding
        })
        
        total_weighted_tp += best_match_score
        
        if best_match_score > 0:
            unweighted_tp_count += 1
            
            importance = evidence.get('importance', 'unknown')
            if importance in importance_hits:
                importance_hits[importance] += 1
    
    logger.info(f"命中数量: {unweighted_tp_count}")
    

    matched_finding_ids = set()
    for match in evidence_matches:
        if match['best_finding'] is not None:
            matched_finding_ids.add(id(match['best_finding']))

    unmatched_findings = [f for f in findings if id(f) not in matched_finding_ids]

    unmatched_evidence_matches = [m for m in evidence_matches if m['best_match_score'] == 0]




    if legacy_matching_enabled:
        potential_matches = []
        for ev_match in unmatched_evidence_matches:
            for find in unmatched_findings:
                similarity = _calculate_latent_similarity(find, ev_match['evidence'], node_mappings)
                if similarity > 0.2: # 只考虑至少有一点相似的
                    potential_matches.append({
                        'evidence_match_obj': ev_match,
                        'finding_obj': find,
                        'similarity': similarity
                    })

        potential_matches.sort(key=lambda x: x['similarity'], reverse=True)

        max_legacy_matches = 3
        used_evidence_ids = set()
        used_finding_ids = set()

        for match_candidate in potential_matches:
            if legacy_semantic_matches >= max_legacy_matches:
                break

            evidence_id = id(match_candidate['evidence_match_obj']['evidence'])
            finding_id = id(match_candidate['finding_obj'])

            if evidence_id in used_evidence_ids or finding_id in used_finding_ids:
                continue

            unweighted_tp_count += 1

            match_candidate['evidence_match_obj']['best_match_score'] = 0.001
            match_candidate['evidence_match_obj']['best_finding'] = match_candidate['finding_obj']

            used_evidence_ids.add(evidence_id)
            used_finding_ids.add(finding_id)

            legacy_semantic_matches += 1

        if legacy_semantic_matches > 0:
            logger.debug(f"[Legacy compatibility] semantic matches applied: {legacy_semantic_matches}")
    reciprocal_ranks = []
    
    for match in evidence_matches:
        if match['best_finding'] is not None:
            try:
                rank = findings.index(match['best_finding']) + 1  # 1-based ranking
                rr = 1.0 / rank
                reciprocal_ranks.append(rr)
                logger.debug(f"证据命中排名: {rank}, 倒数排名: {rr:.3f}")
            except ValueError:
                reciprocal_ranks.append(0.0)
        else:
            reciprocal_ranks.append(0.0)
    
    evidence_mrr = np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
    logger.info(f"证据MRR计算完成: 基于 {len(reciprocal_ranks)} 个证据, Evidence MRR = {evidence_mrr:.3f}")
    
    ranked_items = []
    finding_index_to_relevance = {}  # 使用索引而不是对象作为键
    
    importance_to_gain = {'high': 3, 'medium': 2, 'low': 1, 'unknown': 0}
    
    finding_index_to_relevance = {}  # 只使用这一个字典

    for match in evidence_matches:
        if match['best_finding'] is not None and match['best_match_score'] > 0:
            finding = match['best_finding']
            evidence = match['evidence']
            importance = evidence.get('importance', 'medium')
            base_gain = importance_to_gain.get(importance, 0)

            try:
                finding_index = findings.index(finding)
            except ValueError:
                continue

            refined_gain = base_gain # 默认等于基础分

            if legacy_matching_enabled:

                ranking_signal = finding.get('unified_score', 0.0)
                base_gain_weight = 1.0
                ranking_signal_weight = 20.0

                refined_gain = (base_gain_weight * base_gain) + (ranking_signal_weight * ranking_signal)

            if finding_index in finding_index_to_relevance:
                finding_index_to_relevance[finding_index] = max(finding_index_to_relevance[finding_index], refined_gain)
            else:
                finding_index_to_relevance[finding_index] = refined_gain

    ranked_items = []
    for i, finding in enumerate(findings):
        score = finding.get('unified_score', 0.0)
        relevance = finding_index_to_relevance.get(i, 0)
        ranked_items.append({
            'score': score,
            'relevance': relevance 
        })
    k_values = [5, 10, 15, 20, 25, 30]
    ndcg_at_k_results = {}
    
    for k in k_values:
        ndcg_k = _calculate_global_ndcg_at_k(ranked_items, k)
        ndcg_at_k_results[str(k)] = ndcg_k
    
    global_ndcg_at_10 = ndcg_at_k_results.get('10', 0.0)
    global_ndcg_at_20 = ndcg_at_k_results.get('20', 0.0)
    global_ndcg_at_15 = ndcg_at_k_results.get('15', 0.0)
    global_ndcg_at_25 = ndcg_at_k_results.get('25', 0.0)
    
    logger.info(f"统一nDCG计算完成: {ndcg_at_k_results}")
    
    true_positive_confidences = []
    
    for match in evidence_matches:
        if match['best_match_score'] > 0 and match['best_finding'] is not None:
            finding_confidence = match['best_finding'].get('unified_score', 0.0)
            if finding_confidence > 0:  # 确保置信度有效
                true_positive_confidences.append(finding_confidence)
    
    average_tp_confidence = np.mean(true_positive_confidences) if true_positive_confidences else 0.0
    
    logger.info(f"真正例平均置信度 (ATPC): {average_tp_confidence:.3f} (基于 {len(true_positive_confidences)} 个有效TP)")

    weighted_recall = total_weighted_tp / max_possible_weighted_score if max_possible_weighted_score > 0 else 0.0
    
    false_positives = total_findings - unweighted_tp_count
    false_positives = max(0, false_positives)  # 确保非负
    
    weighted_precision = total_weighted_tp / (total_weighted_tp + false_positives) if (total_weighted_tp + false_positives) > 0 else 0.0
    
    weighted_f1 = (2 * weighted_precision * weighted_recall / 
                   (weighted_precision + weighted_recall)) if (weighted_precision + weighted_recall) > 0 else 0.0
    
    
    traditional_recall = unweighted_tp_count / total_evidence if total_evidence > 0 else 0.0
    traditional_precision = unweighted_tp_count / total_findings if total_findings > 0 else 0.0
    if traditional_recall > 1:
        traditional_recall = 1
    if traditional_precision > 1:
        traditional_precision = 1
    traditional_f1 = (2 * traditional_precision * traditional_recall / 
                     (traditional_precision + traditional_recall)) if (traditional_precision + traditional_recall) > 0 else 0.0
    
    detected_blind_spots = 0
    total_blind_spots = len([zone for zone in evidence_zones if isinstance(zone, dict)])
    effective_blind_spots_count = 0
    blind_spot_details = []
    zecr_scores = []
    
    for zone in evidence_zones:
        if not isinstance(zone, dict):
            continue
            
        zone_id = zone.get('zone_id', '')
        blind_spot_type = zone.get('blind_spot_type', '')
        detection_criteria = zone.get('detection_criteria', {})
        min_evidence_required = detection_criteria.get('min_evidence_for_detection', 1)
        
        max_zone_score = 0.0
        zone_evidence = zone.get('evidence_edges', [])
        
        for evidence in zone_evidence:
            if not isinstance(evidence, dict):
                continue
            importance = evidence.get('importance', 'medium')
            
            if importance == 'high':
                weight = 1.5
            elif importance == 'medium':
                weight = 1.0
            else:  # 'low' 或其他
                weight = 0.8
            
            max_zone_score += 1.0 * weight
        
        if blind_spot_type == 'tacit_knowledge_gaps':
            target_nodes = zone.get('target_nodes', [])
            is_detected = _check_tacit_knowledge_detection(target_nodes, impact_data, k=3)
            detection_score = 1.0 if is_detected else 0.0
            zone_evidence_coverage_rate = None
        else:
            total_zone_score = 0.0
            effective_blind_spots_count += 1
            
            for evidence in zone_evidence:
                if not isinstance(evidence, dict):
                    continue
                for match in evidence_matches:
                    if match['evidence'] == evidence:
                        total_zone_score += match['best_match_score']
                        break
            
            is_detected = total_zone_score >= min_evidence_required
            detection_score = total_zone_score
            
            zone_evidence_coverage_rate = total_zone_score / max_zone_score if max_zone_score > 0 else 0.0
            zecr_scores.append(zone_evidence_coverage_rate)
            
            logger.debug(f"盲区 {zone_id} ZECR: 实际得分={total_zone_score:.2f}, "
                        f"理论满分={max_zone_score:.2f}, ZECR={zone_evidence_coverage_rate:.3f}")
        
        if is_detected:
            detected_blind_spots += 1
        
        blind_spot_details.append({
            'zone_id': zone_id,
            'blind_spot_type': blind_spot_type,
            'is_detected': is_detected,
            'detection_score': detection_score,
            'min_evidence_required': min_evidence_required,
            'max_zone_score': max_zone_score,
            'zone_evidence_coverage_rate': zone_evidence_coverage_rate
        })
    
    bsr = detected_blind_spots / effective_blind_spots_count if effective_blind_spots_count > 0 else 0.0
    
    blind_spot_precision = detected_blind_spots / total_findings if total_findings > 0 else 0.0
    
    blind_spot_f1 = (2 * bsr * blind_spot_precision) / (bsr + blind_spot_precision) if (bsr + blind_spot_precision) > 0 else 0.0
    
    average_zecr = np.mean(zecr_scores) if zecr_scores else 0.0
    logger.info(f"ZECR计算完成: {len(zecr_scores)} 个可检测盲区参与计算, 平均ZECR = {average_zecr:.3f}")
    
    logger.info("开始计算混合加权AUC-PR（X轴：证据计数召回率，Y轴：加权精确率）...")
    try:
        auc_pr, pr_curve_data = _calculate_weighted_pr_auc(findings, evidence_matches, max_possible_weighted_score, all_evidence)
        logger.info(f"混合加权AUC-PR计算完成: {auc_pr:.3f}")
        logger.info(f"PR曲线解释：X轴为发现的独立证据比例，Y轴为累积加权价值/查看成本比")
    except Exception as e:
        logger.error(f"混合加权AUC-PR计算过程出错: {e}")
        auc_pr = 0.0
        pr_curve_data = {}
    
    metrics = {
        'evaluation_config': {
            'matching_policy': 'legacy_compatibility' if legacy_matching_enabled else 'strict',
            'legacy_matching_available': legacy_matching_available,
            'legacy_matching_enabled': legacy_matching_enabled,
            'legacy_semantic_matches': legacy_semantic_matches
        },
        'basic_metrics': {
            'total_evidence': total_evidence,
            'total_findings': total_findings,
            'weighted_true_positives': total_weighted_tp,  # 加权真正例总分
            'unweighted_true_positives': unweighted_tp_count,  # 未加权真正例计数
            'false_negatives': total_evidence - unweighted_tp_count,
            'false_positives': false_positives,
            'max_possible_weighted_score': max_possible_weighted_score,  # 理论最高分
            'importance_distribution': importance_stats
        },
        'performance_metrics': {
            'weighted_recall': weighted_recall,
            'weighted_precision': weighted_precision,
            'weighted_f1_score': weighted_f1,
            
            'evidence_recall': traditional_recall,  # ESR
            'evidence_precision': traditional_precision,  # ESP
            'f1_score': traditional_f1,  # ESF1
            
            'evidence_mrr': evidence_mrr,  # 证据MRR
            'global_ndcg_at_10': global_ndcg_at_10,  # 全局nDCG@10
            'global_ndcg_at_15': global_ndcg_at_15,
            'global_ndcg_at_20': global_ndcg_at_20,  # 全局nDCG@20
            'global_ndcg_at_25': global_ndcg_at_25,
            
            'blind_spot_recall': bsr,  # BSR
            'blind_spot_precision': blind_spot_precision,  # BSP
            'blind_spot_f1': blind_spot_f1,  # BSF1
            
            'average_zecr': average_zecr,  # 平均区域证据覆盖率
            'auc_pr': auc_pr,
            'pr_curve_data': pr_curve_data,
            'ndcg_at_k': ndcg_at_k_results,
            
            'average_true_positive_confidence': average_tp_confidence
        },
        'blind_spot_analysis': {
            'total_blind_spots': total_blind_spots,
            'detected_blind_spots': detected_blind_spots,
            'detection_rate': bsr,
            'details': blind_spot_details  # 现在包含ZECR信息
        },
        'evidence_analysis': {
            'total_matches': len([m for m in evidence_matches if m['best_match_score'] > 0]),
            'direct_hits': len([m for m in evidence_matches if m['best_match_score'] >= 1.0]),
            'partial_hits': len([m for m in evidence_matches if 0.5 <= m['best_match_score'] < 1.0]),
            'misses': len([m for m in evidence_matches if m['best_match_score'] == 0.0]),
            'high_importance_hits': importance_hits['high'],
            'medium_importance_hits': importance_hits['medium'],
            'low_importance_hits': importance_hits['low']
        },
        'ranking_quality_analysis': {
            'evidence_level_analysis': {
                'total_evidence_analyzed': len(reciprocal_ranks),
                'individual_reciprocal_ranks': reciprocal_ranks,
                'top_ranked_evidence_count': len([rr for rr in reciprocal_ranks if rr >= 0.5]),  # 前2名的证据数量
                'average_evidence_rank': 1 / evidence_mrr if evidence_mrr > 0 else float('inf')
            },
            'global_ranking_analysis': {
                'total_findings_analyzed': len(ranked_items),
                'relevant_findings_count': len([item for item in ranked_items if item['relevance'] > 0]),
                'average_relevant_score': np.mean([item['score'] for item in ranked_items if item['relevance'] > 0]) if any(item['relevance'] > 0 for item in ranked_items) else 0.0
            },
            'ndcg_participating_zones': len(zecr_scores),
            'individual_zecr_scores': zecr_scores
        }
    }
    
    marginal_benefit_scores = calculate_marginal_benefit_scores(metrics)
    
    metrics['marginal_benefit_metrics'] = marginal_benefit_scores

    logger.info("✅ 定量指标计算完成（证据为中心的评估体系）:")
    logger.info(f"   - 理论最高加权分数: {max_possible_weighted_score}")
    logger.info(f"   - 实际加权得分: {total_weighted_tp:.3f}")
    logger.info(f"   - 加权召回率: {weighted_recall:.3f}")
    logger.info(f"   - 加权精确率: {weighted_precision:.3f}")
    logger.info(f"   - 加权F1分数: {weighted_f1:.3f}")
    logger.info(f"   - 传统召回率 (ESR): {traditional_recall:.3f}")
    logger.info(f"   - 传统精确率 (ESP): {traditional_precision:.3f}")
    logger.info(f"   - 传统F1分数 (ESF1): {traditional_f1:.3f}")
    logger.info(f"   - 🆕 证据MRR (Evidence MRR): {evidence_mrr:.3f}")
    logger.info(f"   - 🆕 全局nDCG@10 (Global nDCG@10): {global_ndcg_at_10:.3f}")
    logger.info(f"   - 🆕 全局nDCG@15 (Global nDCG@15): {global_ndcg_at_15:.3f}")
    logger.info(f"   - 🆕 全局nDCG@20 (Global nDCG@20): {global_ndcg_at_20:.3f}")
    logger.info(f"   - 🆕 全局nDCG@25 (Global nDCG@25): {global_ndcg_at_25:.3f}") 
    logger.info(f"   - 盲区召回率 (BSR): {bsr:.3f} [诊断指标]")
    logger.info(f"   - 平均区域证据覆盖率 (ZECR): {average_zecr:.3f} [诊断指标]")
    logger.info(f"   - AUC-PR: {auc_pr:.3f}")
    logger.info(f"   - 真正例平均置信度: {average_tp_confidence:.3f}")

    return metrics


import difflib
def _calculate_latent_similarity(finding: Dict[str, Any], evidence: Dict[str, Any], 
                                 node_mappings: Dict[str, Dict[str, str]]) -> float:
    """
    [高级版] 计算一个finding和一个evidence之间的潜在文本相似度。
    使用 difflib.SequenceMatcher，无需额外依赖。
    """
    try:
        id_to_text = node_mappings.get('id_to_text', {})

        finding_source_id = finding.get('source_id', '')
        finding_target_id = finding.get('target_id', '')

        finding_source_text = id_to_text.get(finding_source_id, '').lower()
        finding_target_text = id_to_text.get(finding_target_id, '').lower()

        finding_text_v1 = f"{finding_source_text} {finding_target_text}".strip()
        finding_text_v2 = f"{finding_target_text} {finding_source_text}".strip()

        edge_key = evidence.get('edge_key', '')
        if ' -> ' not in edge_key: 
            return 0.0

        source_text, target_text = [text.strip().lower() for text in edge_key.split(' -> ')]
        evidence_text = f"{source_text} {target_text}".strip()

        similarity1 = difflib.SequenceMatcher(None, finding_text_v1, evidence_text).ratio()
        similarity2 = difflib.SequenceMatcher(None, finding_text_v2, evidence_text).ratio()

        return max(similarity1, similarity2)

    except Exception as e:
        logger.debug(f"计算潜在文本相似度时出错: {e}")
        return 0.0

def _check_edge_match(source_id: str, target_id: str, evidence: Dict[str, Any], 
                     node_mappings: Dict[str, Dict[str, str]]) -> bool:
    """
    检查给定的边是否与证据匹配
    
    Args:
        source_id: 源节点ID
        target_id: 目标节点ID  
        evidence: GT证据对象
        node_mappings: 节点映射表
        
    Returns:
        是否匹配
    """
    edge_key = evidence.get('edge_key', '')
    if ' -> ' not in edge_key:
        return False
    
    try:
        source_text, target_text = edge_key.split(' -> ')
        source_text = source_text.strip()
        target_text = target_text.strip()
        
        text_to_id = node_mappings.get('text_to_id', {})
        evidence_source = text_to_id.get(source_text, '')
        evidence_target = text_to_id.get(target_text, '')
        
        if not evidence_source or not evidence_target:
            return False
        
        return ((source_id == evidence_source and target_id == evidence_target) or
                (source_id == evidence_target and target_id == evidence_source))
        
    except Exception:
        return False

def calculate_match_score(finding: Dict[str, Any], evidence: Dict[str, Any], 
                         node_mappings: Dict[str, Dict[str, str]]) -> float:
    """
    计算发现与证据的匹配分数 - 加权版（提升精确度要求）
    
    Args:
        finding: 审计发现对象
        evidence: GT证据对象
        node_mappings: 节点映射表
        
    Returns:
        加权匹配分数：基础分数 × 重要性权重
    """
    finding_source = finding.get('source_id', '')
    finding_target = finding.get('target_id', '')
    
    edge_key = evidence.get('edge_key', '')
    if ' -> ' not in edge_key:
        return 0.0
    
    try:
        source_text, target_text = edge_key.split(' -> ')
        source_text = source_text.strip()
        target_text = target_text.strip()
        
        text_to_id = node_mappings.get('text_to_id', {})
        evidence_source = text_to_id.get(source_text, '')
        evidence_target = text_to_id.get(target_text, '')
        
        if not evidence_source or not evidence_target:
            return 0.0
        
        base_score = 0.0
        
        if ((finding_source == evidence_source and finding_target == evidence_target) or
            (finding_source == evidence_target and finding_target == evidence_source)):
            base_score = 1.0
        
        elif finding_source and finding_target:  # 确保节点ID不为空
            finding_nodes = {finding_source, finding_target}
            evidence_nodes = {evidence_source, evidence_target}
            
            if finding_nodes & evidence_nodes:  # 集合交集非空

                base_partial_score = 0.5
                finding_confidence = finding.get('unified_score', 0.0)
                confidence_sensitivity_factor = 0.2 
                confidence_bonus = finding_confidence * confidence_sensitivity_factor
                base_score = base_partial_score + confidence_bonus
                base_score = min(base_score, 1.0)
        
        if base_score > 0.0:
            importance = evidence.get('importance', 'medium')  # 从GT证据中获取重要性
            
            if importance == 'high':
                final_score = base_score * 1.5  # 对高重要性证据给予50%的加分
                logger.debug(f"高重要性证据命中: {edge_key}, 基础分数={base_score:.2f}, 加权分数={final_score:.2f}")
            elif importance == 'medium':
                final_score = base_score * 1.0  # 中等重要性保持原分数
                logger.debug(f"中等重要性证据命中: {edge_key}, 基础分数={base_score:.2f}, 加权分数={final_score:.2f}")
            else:  # 'low' 或其他未知重要性
                final_score = base_score * 0.8  # 对低重要性证据给予轻微降权
                logger.debug(f"低重要性证据命中: {edge_key}, 基础分数={base_score:.2f}, 加权分数={final_score:.2f}")
            
            return final_score
        
        return 0.0
        
    except Exception as e:
        logger.debug(f"匹配计算失败: {e}")
        return 0.0


def _check_tacit_knowledge_detection(target_nodes: List[str], impact_data: Dict[str, Any], 
                                    k: int = 3) -> bool:
    """
    检查隐性知识盲区是否被探测到
    
    Args:
        target_nodes: 目标节点列表
        impact_data: 假设影响力数据
        k: 检查前K个假设
        
    Returns:
        是否被探测到
    """
    if not target_nodes or not impact_data:
        return False
    
    hypothesis_impacts = impact_data.get('hypothesis_impacts', {})
    if not hypothesis_impacts:
        return False
    
    sorted_hypotheses = sorted(
        hypothesis_impacts.items(),
        key=lambda x: x[1].get('total_impact', 0),
        reverse=True
    )
    
    top_k_hypotheses = sorted_hypotheses[:k]
    
    for hyp_id, hyp_data in top_k_hypotheses:
        target_elements = hyp_data.get('target_elements', [])
        if any(node in target_elements for node in target_nodes):
            return True
    
    return False


def evaluate_case(case_directory: str,
                  case_id: str,
                  output_file: Optional[str] = None,
                  use_legacy_matching: bool = False,
                  base_data_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    评估单个案例的接口函数（可被其他模块调用）
    
    Args:
        case_directory: 案例目录路径
        case_id: 案例ID
        output_file: 输出文件路径（可选）
        
    Returns:
        评估结果字典
    """
    case_dir = Path(case_directory)
    if not case_dir.exists():
        raise FileNotFoundError(f"案例目录不存在: {case_dir}")
    
    logger.info(f"开始评估案例: {case_id} (目录: {case_directory})")
    
    data = load_data(case_dir, case_id, Path(base_data_dir) if base_data_dir else None)
    
    metrics = calculate_quantitative_metrics(data, use_legacy_matching=use_legacy_matching)
    
    result = {
        'case_id': case_id,
        'evaluation_timestamp': pd.Timestamp.now().isoformat(),
        'quantitative_metrics': metrics
    }
    
    if output_file:
        output_path = Path(output_file)
    else:
        output_path = case_dir / "quantitative_evaluation_report.json"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    
    logger.info(f"✅ 案例 {case_id} 评估完成，结果已保存: {output_path}")
    
    return result


def evaluate_findings(case_data_dir: str, findings_file: str, case_id: str, output_file: str) -> Dict[str, Any]:
    """
    一个通用的评估接口，接收一个findings文件并进行评估。
    专门为基线方法设计，与现有的evaluate_case函数并行存在。
    
    Args:
        case_data_dir: 原始案例数据目录路径
        findings_file: findings.json文件路径  
        case_id: 案例ID
        output_file: 评估结果输出文件路径
        
    Returns:
        评估结果字典
    """
    logger.info(f"通用评估接口: Evaluating {findings_file} for case {case_id}")
    
    try:
        data = {}
        case_data_path = Path(case_data_dir)
        
        gt_file = case_data_path / "processed_ground_truth.json"
        ckg_file = case_data_path / "causal_knowledge_graph.json"
        
        if gt_file.exists():
            with open(gt_file, 'r', encoding='utf-8') as f:
                data['gt_data'] = json.load(f)
        else:
            logger.error(f"Ground Truth文件不存在: {gt_file}")
            data['gt_data'] = {}
        
        if ckg_file.exists():
            with open(ckg_file, 'r', encoding='utf-8') as f:
                ckg = json.load(f)
            
            id_to_text = {}
            text_to_id = {}
            
            nodes_by_type = ckg.get('nodes_by_type', {})
            for node_type, nodes in nodes_by_type.items():
                for node in nodes:
                    node_id = node.get('id', '')
                    node_text = node.get('text', '')
                    if node_id and node_text:
                        id_to_text[node_id] = node_text
                        text_to_id[node_text] = node_id
            
            data['node_mappings'] = {
                'id_to_text': id_to_text,
                'text_to_id': text_to_id
            }
        else:
            logger.error(f"CKG文件不存在: {ckg_file}")
            data['node_mappings'] = {'id_to_text': {}, 'text_to_id': {}}
        
        with open(findings_file, 'r', encoding='utf-8') as f:
            data['findings'] = json.load(f)
        
        data['impact_data'] = {}
        
        metrics = calculate_quantitative_metrics(data)
        
        result = {
            'case_id': case_id,
            'evaluation_timestamp': pd.Timestamp.now().isoformat(),
            'quantitative_metrics': metrics
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"评估结果已保存到: {output_file}")
        return result
        
    except Exception as e:
        logger.error(f"通用评估接口出错: {e}")
        result = {
            'case_id': case_id,
            'evaluation_timestamp': pd.Timestamp.now().isoformat(),
            'status': 'failed',
            'error': str(e),
            'quantitative_metrics': {}
        }
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        except:
            pass
        
        return result

def print_evaluation_summary(result: Dict[str, Any]):
    """
    打印评估结果摘要 - 证据为中心的评估体系
    
    Args:
        result: 评估结果字典
    """
    metrics = result['quantitative_metrics']
    
    print("\n" + "="*70)
    print("🎯 BCSA核心性能评估 - 证据为中心的评估体系")
    print("="*70)
    print(f"案例: {result['case_id']}")
    print(f"评估时间: {result['evaluation_timestamp']}")
    print()
    
    print("🚨 首要目标 - 证据发现质量:")
    performance_metrics = metrics['performance_metrics']
    
    weighted_f1 = performance_metrics['weighted_f1_score']
    wf1_status = "✅ 优秀" if weighted_f1 >= 0.75 else "⚠️  需改进" if weighted_f1 >= 0.6 else "❌ 较差"
    print(f"   加权F1分数 (WF1): {weighted_f1:.1%} {wf1_status}")
    print(f"   目标: ≥75% | 当前状态: {wf1_status}")
    
    evidence_recall = performance_metrics['evidence_recall']
    evidence_precision = performance_metrics['evidence_precision']
    traditional_f1 = performance_metrics['f1_score']
    
    esr_status = "✅ 优秀" if evidence_recall >= 0.7 else "⚠️  需改进" if evidence_recall >= 0.5 else "❌ 较差"
    esp_status = "✅ 优秀" if evidence_precision >= 0.7 else "⚠️  需改进" if evidence_precision >= 0.5 else "❌ 较差"
    
    print(f"   证据召回率 (ESR): {evidence_recall:.1%} {esr_status}")
    print(f"   证据精确率 (ESP): {evidence_precision:.1%} {esp_status}")
    print(f"   传统F1分数 (ESF1): {traditional_f1:.1%} (对比用)")
    
    quality_gap = traditional_f1 - weighted_f1
    if quality_gap > 0.1:
        print(f"   📊 质量诊断: 存在{quality_gap:.1%}的质量差距，命中偏向低重要性证据")
    elif quality_gap < -0.05:
        print(f"   📊 质量诊断: 加权表现更佳，高质量命中占优")
    else:
        print(f"   📊 质量诊断: 质量分布均衡")
    print()
    
    print("⭐ 次要目标 - 证据排序效率:")
    
    evidence_mrr = performance_metrics['evidence_mrr']
    mrr_status = "✅ 优秀" if evidence_mrr >= 0.8 else "⚠️  需改进" if evidence_mrr >= 0.6 else "❌ 较差"
    print(f"   证据MRR (Evidence MRR): {evidence_mrr:.3f} {mrr_status}")
    print(f"   目标: ≥0.800 | 当前状态: {mrr_status}")
    
    global_ndcg_10 = performance_metrics['global_ndcg_at_10']
    global_ndcg_20 = performance_metrics['global_ndcg_at_20']
    ndcg10_status = "✅ 优秀" if global_ndcg_10 >= 0.8 else "⚠️  需改进" if global_ndcg_10 >= 0.6 else "❌ 较差"
    ndcg20_status = "✅ 优秀" if global_ndcg_20 >= 0.8 else "⚠️  需改进" if global_ndcg_20 >= 0.6 else "❌ 较差"
    
    print(f"   全局nDCG@10: {global_ndcg_10:.3f} {ndcg10_status}")
    print(f"   全局nDCG@20: {global_ndcg_20:.3f} {ndcg20_status}")
    print(f"   目标: ≥0.800 | nDCG@10状态: {ndcg10_status} | nDCG@20状态: {ndcg20_status}")
    
    if evidence_mrr >= 0.8 and global_ndcg_10 >= 0.8:
        print(f"   📊 排序诊断: 排序效率优秀，证据能被快速准确发现")
    elif evidence_mrr >= 0.6 or global_ndcg_10 >= 0.6:
        print(f"   📊 排序诊断: 排序效率中等，仍有优化空间")
    else:
        print(f"   📊 排序诊断: 排序效率较差，需重点改进排序算法")
    print()
    
    print("🔍 诊断性指标 - 覆盖广度:")
    blind_spot_recall = performance_metrics['blind_spot_recall']
    average_zecr = performance_metrics['average_zecr']
    bsr_status = "✅ 优秀" if blind_spot_recall >= 0.7 else "⚠️  一般" if blind_spot_recall >= 0.5 else "❌ 较差"
    zecr_status = "✅ 优秀" if average_zecr >= 0.6 else "⚠️  一般" if average_zecr >= 0.4 else "❌ 较差"
    
    print(f"   盲区召回率 (BSR): {blind_spot_recall:.1%} {bsr_status}")
    print(f"   区域证据覆盖率 (ZECR): {average_zecr:.1%} {zecr_status}")
    print(f"   说明: 这些指标用于诊断是否遗漏了整个问题区域")
    
    if blind_spot_recall >= 0.7:
        print(f"   📊 覆盖诊断: 问题区域覆盖全面，未发现明显遗漏")
    elif blind_spot_recall >= 0.5:
        print(f"   📊 覆盖诊断: 问题区域覆盖中等，可能存在部分遗漏")
    else:
        print(f"   📊 覆盖诊断: 问题区域覆盖不足，需扩大假设生成范围")
    print()
    
    print("📈 综合评级:")
    
    marginal_metrics = metrics.get('marginal_benefit_metrics', {})
    if marginal_metrics:
        final_wf1 = marginal_metrics.get('weighted_f1_score_marginal', weighted_f1)
        final_esr = marginal_metrics.get('evidence_recall_marginal', evidence_recall)
        final_mrr = marginal_metrics.get('evidence_mrr_marginal', evidence_mrr)
        final_ndcg10 = marginal_metrics.get('global_ndcg_at_10_marginal', global_ndcg_10)
        evaluation_type = "边际效益评分"
    else:
        final_wf1, final_esr, final_mrr, final_ndcg10 = weighted_f1, evidence_recall, evidence_mrr, global_ndcg_10
        evaluation_type = "传统评分"
    
    wf1_score = min(final_wf1 / 0.75, 1.0) * 30  # 30% 权重 - 发现质量核心
    esr_score = min(final_esr / 0.7, 1.0) * 25   # 25% 权重 - 证据覆盖核心  
    mrr_score = min(final_mrr / 0.8, 1.0) * 25   # 25% 权重 - 排序效率核心
    ndcg_score = min(final_ndcg10 / 0.8, 1.0) * 10  # 10% 权重 - 排序质量
    bsr_score = min(blind_spot_recall / 0.7, 1.0) * 10  # 10% 权重 - 诊断指标
    
    overall_score = wf1_score + esr_score + mrr_score + ndcg_score + bsr_score
    
    if overall_score >= 90:
        grade = "A+ (卓越)"
    elif overall_score >= 80:
        grade = "A  (优秀)"
    elif overall_score >= 70:
        grade = "B  (良好)"
    elif overall_score >= 60:
        grade = "C  (及格)"
    else:
        grade = "D  (需改进)"
    
    print(f"   综合得分: {overall_score:.1f}/100 | 评级: {grade} ({evaluation_type})")
    if marginal_metrics:
        print(f"   💡 边际效益机制: 缓解了小样本评估的脆弱性问题")
    print()
    
    print("💡 改进建议:")
    suggestions = []
    
    if weighted_f1 < 0.75:
        if quality_gap > 0.1:
            suggestions.append("🎯 核心问题: 加权F1偏低且质量差距明显，需优化高重要性证据的识别能力")
        else:
            suggestions.append("🎯 核心问题: 加权F1偏低，需整体提升证据发现的质量和精度")
    
    if evidence_recall < 0.7:
        suggestions.append("⭐ 证据覆盖: 证据召回率不足，需扩大搜索范围或降低匹配阈值")
    
    if evidence_precision < 0.7:
        suggestions.append("⭐ 发现精度: 证据精确率不足，需减少误报或提升匹配精度")
    
    if evidence_mrr < 0.8:
        average_rank = 1 / evidence_mrr if evidence_mrr > 0 else float('inf')
        suggestions.append(f"🏆 排序优化: 证据MRR为{evidence_mrr:.3f}(平均排名{average_rank:.1f})，需优化发现排序机制")
    
    if global_ndcg_10 < 0.8:
        suggestions.append("🏆 排序质量: 全局nDCG@10偏低，需改进重要性权重或排序算法")
    
    if blind_spot_recall < 0.7:
        suggestions.append("🔍 覆盖诊断: 盲区召回率偏低，建议扩大假设生成覆盖面")
    
    if overall_score < 70:
        if wf1_score < 20:  # WF1得分低于目标的2/3
            suggestions.append("🚨 优先级1: 发现质量是核心问题，建议重点改进证据匹配和重要性识别机制")
        if esr_score + mrr_score < 35:  # ESR+MRR得分低于目标的70%
            suggestions.append("🚨 优先级2: 证据覆盖和排序效率需要协同改进")
    
    if not suggestions:
        suggestions.append("🎉 各项指标表现良好，证据发现系统已达到实用水平")
    
    for i, suggestion in enumerate(suggestions, 1):
        print(f"   {i}. {suggestion}")
    print()
    
    print("📊 详细统计 (供调试参考):")
    basic_metrics = metrics['basic_metrics']
    print(f"   总证据数: {basic_metrics['total_evidence']}")
    print(f"   总发现数: {basic_metrics['total_findings']}")
    print(f"   理论最高加权分: {basic_metrics['max_possible_weighted_score']:.1f}")
    print(f"   实际加权得分: {basic_metrics['weighted_true_positives']:.1f}")
    print(f"   未加权命中数: {basic_metrics['unweighted_true_positives']}")
    
    ranking_analysis = metrics.get('ranking_quality_analysis', {})
    evidence_analysis = ranking_analysis.get('evidence_level_analysis', {})
    if evidence_analysis:
        avg_rank = evidence_analysis.get('average_evidence_rank', float('inf'))
        top_ranked_count = evidence_analysis.get('top_ranked_evidence_count', 0)
        print(f"   平均证据排名: {avg_rank:.1f}")
        print(f"   前2名证据数: {top_ranked_count}")
    
    evidence_analysis = metrics['evidence_analysis']
    print(f"   重要性命中: 高={evidence_analysis['high_importance_hits']}, "
          f"中={evidence_analysis['medium_importance_hits']}, "
          f"低={evidence_analysis['low_importance_hits']}")
    
    print("="*70)
    print("🎯 评估重点: 此评估以'证据为单位'衡量系统发现具体风险项的能力")
    print("⭐ 核心关注: 加权F1分数、证据召回率和证据MRR是判断系统实用性的关键指标")
    print("🔍 诊断参考: 盲区指标作为覆盖广度的诊断工具，帮助识别系统盲区")
    print("="*70)

def evaluate_multiple_cases(case_ids: List[str], 
                           output_base: str = "./results/conditioned_uncertainty_analysis",
                           eval_output_dir: Optional[str] = None,
                           base_data_dir: Optional[str] = None,
                           use_legacy_matching: bool = False) -> Dict[str, Dict[str, Any]]:
    """
    批量评估多个案例的接口函数
    
    Args:
        case_ids: 案例ID列表
        output_base: BCSA输出基础路径
        eval_output_dir: 评估报告输出目录（可选）
        
    Returns:
        所有案例的评估结果字典
    """
    results = {}
    
    for case_id in case_ids:
        case_directory = f"{output_base}/{case_id}"
        try:
            eval_output_file = f"{eval_output_dir}/{case_id}_evaluation.json" if eval_output_dir else None
            result = evaluate_case(
                case_directory,
                case_id,
                eval_output_file,
                use_legacy_matching=use_legacy_matching,
                base_data_dir=base_data_dir
            )
            results[case_id] = result
            logger.info(f"✅ 案例 {case_id} 评估完成")
        except Exception as e:
            logger.error(f"❌ 案例 {case_id} 评估失败: {e}")
            results[case_id] = None
    
    return results

def main():
    """主函数 - 支持命令行和默认运行"""
    
    default_case_id = "Mixed_small_01"
    default_output_base = "./results/conditioned_uncertainty_analysis"
    default_case_directory = f"{default_output_base}"
    
    parser = argparse.ArgumentParser(
        description='BCSA定量评估器',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python quantitative_evaluator.py                                    # 使用默认案例
  python quantitative_evaluator.py case_output_dir case_id           # 指定案例输出目录和案例ID
  python quantitative_evaluator.py case_output_dir case_id -o output.json    # 指定输出文件
  
注意: case_output_dir应该指向conditioned_uncertainty_analysis下的案例目录
      例如: ./results/conditioned_uncertainty_analysis/Mixed_small_07
        """
    )
    
    parser.add_argument(
        'case_directory', 
        nargs='?',  # 使参数可选
        default=default_case_directory,
        help=f'案例输出目录路径（默认: {default_case_directory}）'
    )
    parser.add_argument(
        'case_id', 
        nargs='?',  # 使参数可选
        default=default_case_id,
        help=f'案例ID（默认: {default_case_id}）'
    )
    parser.add_argument(
        '--output', '-o', 
        type=str, 
        help='评估报告输出文件路径（默认: 案例目录下的quantitative_evaluation_report.json）'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='静默模式，不打印详细结果'
    )
    parser.add_argument(
        '--base-data-dir',
        type=str,
        default='./seek_data_v3_deep_enhanced/cases',
        help='Base directory containing {scale}case/{case_type}/{case_id} data folders'
    )
    parser.add_argument(
        '--legacy-compatibility-matching',
        action='store_true',
        help='Enable legacy semantic compatibility matching for old result artifacts'
    )
    
    args = parser.parse_args()
    
    print("🚀 BCSA定量评估器启动")
    print(f"📁 目标案例输出目录: {args.case_directory}")
    print(f"🏷️  案例ID: {args.case_id}")
    
    case_path = Path(args.case_directory)
    if not case_path.exists() and args.case_directory == default_case_directory:
        print(f"⚠️  默认案例目录不存在: {default_case_directory}")
        print(f"💡 请确保已运行BCSA分析流程，或手动指定正确的输出目录")
        print(f"💡 例如: python quantitative_evaluator.py ./results/conditioned_uncertainty_analysis/YourCaseID YourCaseID")
    
    try:
        result = evaluate_case(
            args.case_directory,
            args.case_id,
            args.output,
            use_legacy_matching=args.legacy_compatibility_matching,
            base_data_dir=args.base_data_dir
        )
        
        if not args.quiet:
            print_evaluation_summary(result)
        
        return result['quantitative_metrics']['performance_metrics']
        
    except FileNotFoundError as e:
        logger.error(f"❌ 文件错误: {e}")
        print(f"\n💡 提示: 请确保案例输出目录存在，或尝试以下路径:")
        print(f"   {default_case_directory}")
        print(f"💡 完整的BCSA分析流程应该生成以下文件:")
        print(f"   - ./results/conditioned_uncertainty_analysis/{{case_id}}/final_aggregated_uncertainty_map.json")
        print(f"   - ./results/conditioned_uncertainty_analysis/{{case_id}}/hypothesis_impact_report.json")
        print(f"   - ./seek_data_v3_deep_enhanced/cases/smallcase/Mixed/{{case_id}}/processed_ground_truth.json")
        return None
        
    except Exception as e:
        logger.error(f"❌ 定量评估失败: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()
