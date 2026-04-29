#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BCSA共享数据结构模块
定义系统中所有模块共享的核心数据结构，遵循"单一数据源"原则
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

@dataclass
class HypothesisContext:
    """假设生成上下文"""
    ckg: Dict[str, Any]  # 因果知识图谱
    graph: nx.Graph  # NetworkX图对象
    data: pd.DataFrame  # 原始数据
    correlation_matrix: pd.DataFrame  # 相关性矩阵

@dataclass
class Hypothesis:
    """假设数据结构 - 统一的原子假设表示"""
    id: str
    rule_name: str
    rule_category: str  # "structural" 或 "data_driven"
    hypothesis_type: str  # "degree_outlier", "correlation_distance", 等
    description: str
    target_elements: List[str]  # 目标节点或边
    evidence: Dict[str, Any]
    confidence_score: float
    priority: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    explicitly_targeted_edges: List[Tuple[str, str]] = field(default_factory=list)

@dataclass
class DataContext:
    """数据上下文 - 统一的数据访问接口"""
    ckg: Dict[str, Any]
    sensor_data: Optional[pd.DataFrame] = None
    expert_docs: Optional[List[Dict]] = None
    node_mappings: Dict[str, Dict[str, str]] = field(default_factory=dict)
    graph: Optional[nx.DiGraph] = None
    correlation_matrix: Optional[pd.DataFrame] = None

@dataclass
class EdgeUncertaintyResult:
    """边不确定性结果"""
    source_id: str
    target_id: str
    source_text: str
    target_text: str
    edge_exists: bool  # 边是否在原图中存在
    reconstruction_prob: float  # 重构概率
    uncertainty_score: float   # 不确定性分数
    edge_type: str  # 'existing_unreliable' 或 'missing_potential'
    confidence_interval: Tuple[float, float]
    unified_score: float = 0.0  # 统一评分 [-1, 1]
    contributing_hypotheses: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class CognitiveUncertaintyMap:
    """认知不确定性地图"""
    case_id: str
    node_uncertainties: List[Dict[str, Any]]
    edge_uncertainties: List[EdgeUncertaintyResult]
    high_uncertainty_nodes: List[Dict[str, Any]]  # 不确定性最高的节点
    type_a_edges: List[EdgeUncertaintyResult]     # 类型A：现有但不可靠的边
    type_b_edges: List[EdgeUncertaintyResult]     # 类型B：缺失但应存在的边
    global_uncertainty_score: float
    summary_report: Dict[str, Any]

@dataclass
class CoreEvidence:
    """核心证据数据结构"""
    evidence_id: str
    evidence_type: str  # 'node', 'edge'
    content: str  # 节点ID或边描述
    blind_spot_type: str
    importance: str  # 'high', 'medium', 'low'
    expected_change: str  # 'increase', 'decrease', 'detect'
    reason: str
    source_region: str

@dataclass
class AuditFinding:
    """审计发现数据结构"""
    finding_id: str
    finding_type: str  # 'node_uncertainty', 'edge_uncertainty'
    content: str
    uncertainty_score: float
    confidence_level: str  # 'high', 'medium', 'low'
    detection_method: str
    supporting_evidence: Dict[str, Any]

@dataclass
class ComparisonResult:
    """比对结果数据结构"""
    evidence_id: str
    gt_evidence: CoreEvidence
    matched_findings: List[AuditFinding]
    match_quality: float  # 0-1
    match_type: str  # 'exact', 'partial', 'missed'
    analysis: str

@dataclass
class QualitativeEvaluationReport:
    """定性评价报告数据结构"""
    case_id: str
    evaluation_summary: Dict[str, Any]
    node_discovery_analysis: Dict[str, Any]
    edge_discovery_analysis: Dict[str, Any]
    blindspot_matching_analysis: Dict[str, Any]
    detailed_comparisons: List[ComparisonResult]
    expert_diagnosis: str
    recommendations: List[str]
    evidence_comparison: Optional[Dict[str, Any]] = None
    gt_comparison: Optional[Dict[str, Any]] = None

@dataclass
class HypothesisPrompt:
    """假设软提示数据结构"""
    hypothesis_id: str
    text_embedding: np.ndarray  # Sentence-BERT编码
    node_features: np.ndarray   # 节点特征编码
    rule_type_onehot: np.ndarray  # 规则类型独热编码
    confidence_score: float
    priority_score: float
    target_elements: List[str]
    combined_prompt: np.ndarray  # 组合软提示向量

@dataclass
class TrainingResult:
    """模型训练结果"""
    hypothesis_id: str
    final_loss: float
    loss_history: List[float]
    converged: bool
    model_params: Optional[Dict[str, Any]] = None

@dataclass
class AggregatedUncertaintyResult:
    """聚合后的不确定性分析结果"""
    case_id: str
    aggregated_uncertainty_map: List[EdgeUncertaintyResult]
    training_results: Dict[str, TrainingResult]
    aggregation_method: str
    high_priority_findings: List[EdgeUncertaintyResult]
    metadata: Dict[str, Any] = field(default_factory=dict)

def convert_numpy_types(obj):
    """
    递归转换numpy类型为Python原生类型，确保JSON序列化兼容性
    """
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    elif hasattr(obj, '__dict__'):
        return convert_numpy_types(obj.__dict__)
    else:
        return obj