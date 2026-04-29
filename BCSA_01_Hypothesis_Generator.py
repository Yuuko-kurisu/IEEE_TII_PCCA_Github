#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BCSA假设生成模块（重构版）
负责：基于因果知识图谱和数据生成审计假设
职责：单一职责 - 只做假设生成
"""

import os
import json
import numpy as np
import pandas as pd
import networkx as nx
import hashlib
from typing import Dict, List, Tuple, Any, Optional, Protocol, Union
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import warnings
import time
import logging

from BCSA_00_Shared_Structures import (
    Hypothesis, HypothesisContext, DataContext, 
    convert_numpy_types
)

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.chdir(os.path.dirname(os.path.abspath(__file__)))


def get_deterministic_short_hash(input_string: str) -> str:
    """生成一个确定性的短哈希值"""
    if input_string is None:
        input_string = "none"
    else:
        input_string = str(input_string)
    
    sha1_hash = hashlib.sha1(input_string.encode('utf-8')).hexdigest()
    return sha1_hash[:8]

class HypothesisRule(Protocol):
    """假设规则协议 - 定义统一的规则接口"""
    
    def get_rule_info(self) -> Dict[str, str]:
        """获取规则基本信息"""
        ...
    
    def can_apply(self, context: DataContext) -> bool:
        """判断规则是否可应用于当前数据上下文"""
        ...
    
    def generate_hypotheses(self, context: DataContext) -> List[Hypothesis]:
        """生成假设列表"""
        ...

class StructuralRule(ABC):
    """结构规则抽象基类 - 仅依赖CKG结构"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.rule_info = self.get_rule_info()
    
    @abstractmethod
    def get_rule_info(self) -> Dict[str, str]:
        """获取规则信息"""
        pass
    
    def can_apply(self, context: DataContext) -> bool:
        """结构规则只需要CKG即可应用"""
        return context.ckg is not None and context.graph is not None
    
    @abstractmethod
    def generate_hypotheses(self, context: DataContext) -> List[Hypothesis]:
        """生成结构异常假设"""
        pass

class DataDrivenRule(ABC):
    """数据驱动规则抽象基类 - 需要CKG和传感器数据"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.rule_info = self.get_rule_info()
    
    @abstractmethod
    def get_rule_info(self) -> Dict[str, str]:
        """获取规则信息"""
        pass
    
    def can_apply(self, context: DataContext) -> bool:
        """数据驱动规则需要CKG和传感器数据"""
        return (context.ckg is not None and 
                context.graph is not None and
                context.sensor_data is not None and
                not context.sensor_data.empty)
    
    @abstractmethod
    def generate_hypotheses(self, context: DataContext) -> List[Hypothesis]:
        """生成数据-知识不一致假设"""
        pass


class DegreeOutlierRule(StructuralRule):
    """规则S1：度数异常检测"""
    
    def get_rule_info(self) -> Dict[str, str]:
        return {
            "name": "DegreeOutlierRule",
            "category": "structural",
            "type": "degree_outlier",
            "description": "检测度数分布的异常节点"
        }
    
    def generate_hypotheses(self, context: DataContext) -> List[Hypothesis]:
        """生成度数异常假设"""
        hypotheses = []
        graph = context.graph
        
        if not graph or len(graph.nodes) == 0:
            return hypotheses
        
        degrees = dict(graph.degree())
        in_degrees = dict(graph.in_degree())
        out_degrees = dict(graph.out_degree())
        
        degree_values = list(degrees.values())
        if len(degree_values) < 3:
            return hypotheses
        
        low_threshold = np.percentile(degree_values, self.config.get('low_percentile', 10))
        high_threshold = np.percentile(degree_values, self.config.get('high_percentile', 90))
        ratio_threshold = self.config.get('in_out_ratio_threshold', 3.0)  # 同时放宽比率阈值
        
        node_id_to_text = context.node_mappings.get('id_to_text', {})
        
        for node_id, total_degree in degrees.items():
            node_text = node_id_to_text.get(node_id, node_id) or node_id  # 确保不为None
            in_deg = in_degrees.get(node_id, 0)
            out_deg = out_degrees.get(node_id, 0)
            
            is_low_outlier = total_degree <= low_threshold
            is_high_outlier = total_degree >= high_threshold
            
            ratio_anomaly = False
            if out_deg > 0:
                in_out_ratio = in_deg / out_deg
                ratio_anomaly = in_out_ratio >= ratio_threshold or in_out_ratio <= (1/ratio_threshold)
            
            if is_low_outlier or is_high_outlier or ratio_anomaly:
                anomaly_type = []
                if is_low_outlier:
                    anomaly_type.append("低度数异常")
                if is_high_outlier:
                    anomaly_type.append("高度数异常")
                if ratio_anomaly:
                    anomaly_type.append("入出度比异常")
                
                explicitly_targeted_edges = []
                for neighbor in graph.neighbors(node_id):
                    explicitly_targeted_edges.append((node_id, neighbor))

                degree_zscore = abs((total_degree - np.mean(degree_values)) / max(np.std(degree_values), 0.001))
                confidence = min(0.95, 0.6 + degree_zscore * 0.1)  # 提升基础置信度从0.5到0.6
                
                hypothesis = Hypothesis(
                    id=f"S1_{node_id}_{get_deterministic_short_hash(str(node_text))}",
                    rule_name="DegreeOutlierRule",
                    rule_category="structural",
                    hypothesis_type="degree_outlier",
                    description=f"节点'{node_text}'度数异常：{', '.join(anomaly_type)}",
                    target_elements=[node_id],
                    evidence={
                        "node_id": node_id,
                        "node_text": str(node_text),
                        "total_degree": total_degree,
                        "in_degree": in_deg,
                        "out_degree": out_deg,
                        "anomaly_types": anomaly_type,
                        "degree_zscore": degree_zscore,
                        "low_threshold": low_threshold,
                        "high_threshold": high_threshold
                    },
                    confidence_score=confidence,
                    priority=confidence * 0.7,  # 提升优先级权重
                    metadata={
                        "detection_method": "enhanced_statistical_outlier_analysis",
                        "graph_size": len(graph.nodes),
                    },
                    explicitly_targeted_edges =explicitly_targeted_edges
                )
                hypotheses.append(hypothesis)
        
        return hypotheses

class EdgeStrengthOutlierRule(StructuralRule):
    """规则S2：连接强度异常检测"""

    def get_rule_info(self) -> Dict[str, str]:
        return {
            "name": "EdgeStrengthOutlierRule",
            "category": "structural",
            "type": "edge_strength_outlier",
            "description": "检测边权重/置信度的异常值"
        }
    def generate_hypotheses(self, context: DataContext) -> List[Hypothesis]:
        """生成边强度异常假设"""
        hypotheses = []
        graph = context.graph

        if not graph or len(graph.edges) == 0:
            return hypotheses

        edges_data = []
        for edge_key, edge_list in context.ckg.get('edges', {}).items():
            for edge in edge_list:
                source_id = edge['source']
                target_id = edge['target']
                confidence = edge.get('confidence', 0.5)

                if source_id in graph.nodes and target_id in graph.nodes:
                    edges_data.append({
                        'source': source_id,
                        'target': target_id,
                        'confidence': confidence,
                        'edge_key': str(edge_key)  # 确保是字符串
                    })

        if len(edges_data) < 3:
            return hypotheses

        confidences = [edge['confidence'] for edge in edges_data]
        mean_confidence = np.mean(confidences)
        std_confidence = np.std(confidences)

        if std_confidence < 0.01:  # 避免除零
            return hypotheses

        z_threshold = self.config.get('z_threshold', 2.0)
        node_id_to_text = context.node_mappings.get('id_to_text', {})

        for edge_data in edges_data:
            confidence = edge_data['confidence']
            z_score = abs((confidence - mean_confidence) / std_confidence)

            if z_score >= z_threshold:
                source_text = node_id_to_text.get(edge_data['source'], edge_data['source']) or edge_data['source']
                target_text = node_id_to_text.get(edge_data['target'], edge_data['target']) or edge_data['target']

                anomaly_type = "低置信度异常" if confidence < mean_confidence else "高置信度异常"

                hypothesis = Hypothesis(
                    id=f"S2_{edge_data['source']}_{edge_data['target']}_{get_deterministic_short_hash(edge_data['edge_key'])}",
                    rule_name="EdgeStrengthOutlierRule",
                    rule_category="structural",
                    hypothesis_type="edge_strength_outlier",
                    description=f"边'{source_text} -> {target_text}'置信度异常：{anomaly_type}({confidence:.3f})",
                    target_elements=[f"{edge_data['source']}->{edge_data['target']}"],
                    evidence={
                        "source_id": edge_data['source'],
                        "target_id": edge_data['target'],
                        "source_text": str(source_text),
                        "target_text": str(target_text),
                        "edge_confidence": confidence,
                        "mean_confidence": mean_confidence,
                        "std_confidence": std_confidence,
                        "z_score": z_score,
                        "anomaly_type": anomaly_type
                    },
                    confidence_score=min(0.95, 0.6 + z_score * 0.05),
                    priority=min(0.9, 0.7 + z_score * 0.03),
                    metadata={
                        "detection_method": "z_score_outlier_analysis",
                        "total_edges": len(edges_data)
                    }
                )
                hypotheses.append(hypothesis)

        return hypotheses
class CentralityDiscrepancyRule(StructuralRule):
    """规则S3：中心性差异检测"""

    def get_rule_info(self) -> Dict[str, str]:
        return {
            "name": "CentralityDiscrepancyRule",
            "category": "structural",
            "type": "centrality_discrepancy",
            "description": "检测不同中心性指标排名的显著差异"
        }

    def generate_hypotheses(self, context: DataContext) -> List[Hypothesis]:
        """生成中心性差异假设"""
        hypotheses = []
        graph = context.graph

        if not graph or len(graph.nodes) < 5:  # 节点太少无法进行排名比较
            return hypotheses

        try:
            degree_centrality = nx.degree_centrality(graph)
            betweenness_centrality = nx.betweenness_centrality(graph)
            closeness_centrality = nx.closeness_centrality(graph)

            degree_ranks = self._get_ranks(degree_centrality)
            betweenness_ranks = self._get_ranks(betweenness_centrality)
            closeness_ranks = self._get_ranks(closeness_centrality)

            dynamic_threshold = max(5, len(graph.nodes) * 0.2)
            rank_threshold = self.config.get('rank_difference_threshold', dynamic_threshold)
            node_id_to_text = context.node_mappings.get('id_to_text', {})

            for node_id in graph.nodes():
                degree_rank = degree_ranks.get(node_id, len(graph.nodes))
                betweenness_rank = betweenness_ranks.get(node_id, len(graph.nodes))
                closeness_rank = closeness_ranks.get(node_id, len(graph.nodes))

                max_diff = max(abs(degree_rank - betweenness_rank),
                            abs(degree_rank - closeness_rank),
                            abs(betweenness_rank - closeness_rank))

                if max_diff >= rank_threshold:
                    node_text = node_id_to_text.get(node_id, node_id) or node_id  # 确保不为None

                    diff_ratio = max_diff / len(graph.nodes)
                    confidence = min(0.95, 0.65 + diff_ratio * 0.25)  # 提升基础置信度

                    hypothesis = Hypothesis(
                        id=f"S3_{node_id}_{get_deterministic_short_hash(str(node_text))}",
                        rule_name="CentralityDiscrepancyRule",
                        rule_category="structural",
                        hypothesis_type="centrality_discrepancy",
                        description=f"节点'{node_text}'中心性排名差异显著：度数排名{degree_rank}，介数排名{betweenness_rank}，接近度排名{closeness_rank}",
                        target_elements=[node_id],
                        evidence={
                            "node_id": node_id,
                            "node_text": str(node_text),
                            "degree_centrality": degree_centrality.get(node_id, 0),
                            "betweenness_centrality": betweenness_centrality.get(node_id, 0),
                            "closeness_centrality": closeness_centrality.get(node_id, 0),
                            "degree_rank": degree_rank,
                            "betweenness_rank": betweenness_rank,
                            "closeness_rank": closeness_rank,
                            "max_rank_difference": max_diff,
                            "dynamic_threshold": rank_threshold
                        },
                        confidence_score=confidence,
                        priority=confidence * 0.8,  # 提升优先级权重
                        metadata={
                            "detection_method": "dynamic_centrality_ranking_comparison",
                            "total_nodes": len(graph.nodes),
                            "dynamic_threshold_applied": True
                        }
                    )
                    hypotheses.append(hypothesis)

        except Exception as e:
            logger.warning(f"中心性计算失败: {e}")

        return hypotheses

    def _get_ranks(self, centrality_dict: Dict[str, float]) -> Dict[str, int]:
        """将中心性值转换为排名（1为最高）"""
        sorted_items = sorted(centrality_dict.items(), key=lambda x: x[1], reverse=True)
        return {node_id: rank + 1 for rank, (node_id, _) in enumerate(sorted_items)}

class HighCorrelationLongDistanceRule(DataDrivenRule):
    """规则D1：强相关-远距离检测"""

    def get_rule_info(self) -> Dict[str, str]:
        return {
            "name": "HighCorrelationLongDistanceRule",
            "category": "data_driven",
            "type": "correlation_distance_mismatch",
            "description": "检测数据高相关但知识图中距离远或不可达的节点对"
        }


    def generate_hypotheses(self, context: DataContext) -> List[Hypothesis]:
        """生成强相关-远距离假设"""
        hypotheses = []

        if context.correlation_matrix is None or context.correlation_matrix.empty:
            return hypotheses

        correlation_threshold = self.config.get('correlation_threshold', 0.5)
        distance_threshold = self.config.get('distance_threshold', 3)

        graph = context.graph
        node_text_to_id = context.node_mappings.get('text_to_id', {})
        node_id_to_text = context.node_mappings.get('id_to_text', {})

        try:
            shortest_paths = dict(nx.all_pairs_shortest_path_length(graph.to_undirected()))
        except Exception as e:
            logger.warning(f"为HighCorrelationLongDistanceRule计算最短路径时出错: {e}")
            shortest_paths = {}

        corr_matrix = context.correlation_matrix.abs()
        columns = corr_matrix.columns.tolist()

        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):
                col1 = columns[i]
                col2 = columns[j]
                
                correlation = corr_matrix.loc[col1, col2]

                if correlation >= correlation_threshold:
                    node1_id = self._find_matching_node(col1, node_text_to_id, node_id_to_text)
                    node2_id = self._find_matching_node(col2, node_text_to_id, node_id_to_text)

                    if node1_id and node2_id and node1_id != node2_id:
                        distance = shortest_paths.get(node1_id, {}).get(node2_id, float('inf'))

                        if distance > distance_threshold:
                            distance_desc = "不可达" if distance == float('inf') else f"{distance}步"
                            node1_text = node_id_to_text.get(node1_id, col1)
                            node2_text = node_id_to_text.get(node2_id, col2)

                            explicitly_targeted_edges = [(node1_id, node2_id)]

                            hypothesis = Hypothesis(
                                id=f"D1_{node1_id}_{node2_id}_{get_deterministic_short_hash(f'{col1}_{col2}')}",
                                rule_name="HighCorrelationLongDistanceRule",
                                rule_category="data_driven",
                                hypothesis_type="correlation_distance_mismatch",
                                description=f"'{node1_text}'与'{node2_text}'数据高相关({correlation:.3f})但图中距离{distance_desc}",
                                target_elements=[node1_id, node2_id],
                                evidence={
                                    "node1_id": node1_id,
                                    "node2_id": node2_id,
                                    "node1_text": node1_text,
                                    "node2_text": node2_text,
                                    "correlation": correlation,
                                    "graph_distance": distance if distance != float('inf') else -1, # 使用-1表示无穷大以便JSON序列化
                                    "is_reachable": distance != float('inf')
                                },
                                confidence_score=min(0.95, 0.6 + correlation * 0.4),
                                priority=min(0.95, 0.7 + correlation * 0.2 + (0.1 if not (distance != float('inf')) else 0)),
                                metadata={
                                    "detection_method": "correlation_vs_graph_topology",
                                    "correlation_threshold": correlation_threshold,
                                    "distance_threshold": distance_threshold
                                },
                                explicitly_targeted_edges=explicitly_targeted_edges
                            )
                            hypotheses.append(hypothesis)

        return hypotheses

    def _calculate_dynamic_correlation_threshold(self, correlation_matrix: pd.DataFrame) -> float:
        """动态计算相关性阈值"""
        corr_values = correlation_matrix.abs().values
        upper_tri_indices = np.triu_indices_from(corr_values, k=1)
        upper_tri_corrs = corr_values[upper_tri_indices]
        
        if len(upper_tri_corrs) == 0:
            return 0.6  # 默认值
        
        percentile_80 = np.percentile(upper_tri_corrs, 80)
        
        min_threshold = 0.3   # 最低阈值
        max_threshold = 0.8   # 最高阈值
        
        dynamic_threshold = np.clip(percentile_80, min_threshold, max_threshold)
        
        mean_corr = np.mean(upper_tri_corrs)
        if mean_corr < 0.3:
            dynamic_threshold = max(dynamic_threshold * 0.8, min_threshold)
        
        return dynamic_threshold

    def _find_matching_node(self, data_column: str, text_to_id: Dict, id_to_text: Dict) -> Optional[str]:
        """寻找数据列对应的图节点ID"""
        if data_column in text_to_id:
            return text_to_id[data_column]

        clean_column = data_column
        if data_column.startswith('sensor_') and '_' in data_column:
            parts = data_column.split('_', 2)
            if len(parts) >= 3:
                clean_column = parts[2]

        if clean_column in text_to_id:
            return text_to_id[clean_column]

        clean_column_lower = clean_column.lower()
        for node_text, node_id in text_to_id.items():
            node_text_lower = node_text.lower()
            if (clean_column_lower == node_text_lower or
                clean_column_lower in node_text_lower or
                node_text_lower in clean_column_lower):
                return node_id

        return None

    def _find_matching_node_id(self, data_column: str, id_to_text: Dict[str, str]) -> str:
        """根据数据列名查找对应的节点ID"""
        if data_column in id_to_text:
            return data_column

        for node_id, node_text in id_to_text.items():
            if data_column.lower() in node_text.lower() or node_text.lower() in data_column.lower():
                return node_id

        return None

    def _find_matching_column(self, node_id: str, id_to_text: Dict, columns: List[str]) -> Optional[str]:
        """寻找节点对应的数据列"""
        node_text = id_to_text.get(node_id, "")
        if not node_text:
            return None

        if node_text in columns:
            return node_text

        node_text_lower = node_text.lower()
        for col in columns:
            if col.startswith('sensor_') and '_' in col:
                parts = col.split('_', 2)
                if len(parts) >= 3 and parts[2].lower() == node_text_lower:
                    return col

            if node_text_lower in col.lower() or col.lower() in node_text_lower:
                return col

        return None

class ConditionalInstabilityRule(DataDrivenRule):
    """规则D2：条件性不稳定性检测 - 唤醒版本，增加详细调试"""

    def get_rule_info(self) -> Dict[str, str]:
        return {
            "name": "ConditionalInstabilityRule",
            "category": "data_driven",
            "type": "conditional_instability",
            "description": "检测在不同条件下相关性不稳定的边（调试增强版）"
        }

    def generate_hypotheses(self, context: DataContext) -> List[Hypothesis]:
        """生成条件性不稳定假设 - 增加详细调试信息"""
        hypotheses = []

        if context.sensor_data is None or context.sensor_data.empty:
            logger.info("ConditionalInstabilityRule: 传感器数据为空，跳过")
            return hypotheses

        dynamic_thresholds = self._calculate_dynamic_thresholds(context.sensor_data)
        variance_threshold = dynamic_thresholds['variance_threshold']
        min_samples_per_group = max(3, dynamic_thresholds['min_samples'])  # 降低最小样本要求

        logger.info(f"ConditionalInstabilityRule: 动态阈值 - 方差阈值: {variance_threshold:.4f}, 最小样本: {min_samples_per_group}")

        sensor_data = context.sensor_data
        numeric_cols = sensor_data.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_cols) < 3:
            logger.info(f"ConditionalInstabilityRule: 数值列不足({len(numeric_cols)})，跳过")
            return hypotheses

        node_id_to_text = context.node_mappings.get('id_to_text', {})

        edge_checked = 0
        matched_edges = 0
        debug_info = {
            'total_edges': 0,
            'matching_attempts': 0,
            'successful_matches': 0,
            'instability_checks': 0,
            'instability_found': 0
        }

        for edge_key, edge_list in context.ckg.get('edges', {}).items():
            for edge in edge_list:
                edge_checked += 1
                debug_info['total_edges'] += 1
                
                source_id = edge['source']
                target_id = edge['target']

                debug_info['matching_attempts'] += 1
                source_col = self._find_matching_column_enhanced(source_id, node_id_to_text, numeric_cols)
                target_col = self._find_matching_column_enhanced(target_id, node_id_to_text, numeric_cols)

                logger.debug(f"ConditionalInstabilityRule: 边 {source_id}->{target_id}")
                logger.debug(f"  - 源节点文本: {node_id_to_text.get(source_id, 'N/A')}")
                logger.debug(f"  - 目标节点文本: {node_id_to_text.get(target_id, 'N/A')}")
                logger.debug(f"  - 匹配到的源列: {source_col}")
                logger.debug(f"  - 匹配到的目标列: {target_col}")

                if source_col and target_col and source_col != target_col:
                    matched_edges += 1
                    debug_info['successful_matches'] += 1
                    
                    
                    debug_info['instability_checks'] += 1
                    instability_result = self._check_conditional_instability_enhanced(
                        sensor_data, source_col, target_col, numeric_cols,
                        variance_threshold, min_samples_per_group, edge_key
                    )

                    if instability_result['is_unstable']:
                        debug_info['instability_found'] += 1
                        source_text = node_id_to_text.get(source_id, source_id)
                        target_text = node_id_to_text.get(target_id, target_id)

                        hypothesis = Hypothesis(
                            id=f"D2_{source_id}_{target_id}_{get_deterministic_short_hash(edge_key) % 10000}",
                            rule_name="ConditionalInstabilityRule",
                            rule_category="data_driven",
                            hypothesis_type="conditional_instability",
                            description=f"边'{source_text} -> {target_text}'在条件'{instability_result['modulator_column']}'下相关性不稳定",
                            target_elements=[f"{source_id}->{target_id}"],
                            evidence={
                                "source_id": source_id,
                                "target_id": target_id,
                                "source_text": source_text,
                                "target_text": target_text,
                                "source_column": source_col,
                                "target_column": target_col,
                                "modulator_column": instability_result['modulator_column'],
                                "correlation_variance": instability_result['correlation_variance'],
                                "correlation_range": instability_result['correlation_range'],
                                "conditional_correlations": instability_result['conditional_correlations'],
                                "edge_confidence": edge.get('confidence', 0.5),
                                "detection_reason": instability_result.get('detection_reason', '')
                            },
                            confidence_score=min(0.95, 0.7 + instability_result['correlation_variance'] * 2),
                            priority=min(0.9, 0.75 + instability_result['correlation_variance'] * 1.5),
                            metadata={
                                "detection_method": "enhanced_conditional_correlation_analysis",
                                "variance_threshold": variance_threshold,
                                "dynamic_thresholds": True
                            }
                        )
                        hypotheses.append(hypothesis)

        logger.info(f"ConditionalInstabilityRule: 调试统计信息")
        logger.info(f"  - 总边数: {debug_info['total_edges']}")
        logger.info(f"  - 匹配尝试: {debug_info['matching_attempts']}")
        logger.info(f"  - 成功匹配: {debug_info['successful_matches']}")
        logger.info(f"  - 不稳定性检查: {debug_info['instability_checks']}")
        logger.info(f"  - 发现不稳定: {debug_info['instability_found']}")
        logger.info(f"  - 最终假设数: {len(hypotheses)}")

        return hypotheses

    def _calculate_dynamic_thresholds(self, sensor_data: pd.DataFrame) -> Dict[str, float]:
        """计算动态阈值"""
        numeric_cols = sensor_data.select_dtypes(include=[np.number]).columns.tolist()
        clean_data = sensor_data[numeric_cols].dropna()
        
        data_size = len(clean_data)
        
        if data_size < 50:
            variance_threshold = 0.05  # 小数据集，更敏感
            min_samples = 2
        elif data_size < 200:
            variance_threshold = 0.07  # 中等数据集
            min_samples = 3
        else:
            variance_threshold = 0.1   # 大数据集，更严格
            min_samples = 5
        
        return {
            'variance_threshold': variance_threshold,
            'min_samples': min_samples
        }

    def _find_matching_column_enhanced(self, node_id: str, id_to_text: Dict, columns: List[str]) -> Optional[str]:
        """增强的列匹配逻辑 - 更详细的调试信息"""
        node_text = id_to_text.get(node_id, "")
        if not node_text:
            logger.debug(f"  - 节点 {node_id} 没有对应文本")
            return None

        if node_text in columns:
            logger.debug(f"  - 直接匹配成功: {node_text}")
            return node_text

        node_text_lower = node_text.lower()
        for col in columns:
            col_lower = col.lower()
            if col.startswith('sensor_') and '_' in col:
                parts = col.split('_', 2)
                if len(parts) >= 3:
                    clean_col_name = parts[2].lower()
                    if clean_col_name == node_text_lower:
                        logger.debug(f"  - 传感器格式匹配成功: {col} -> {node_text}")
                        return col

        for col in columns:
            col_lower = col.lower()
            if (node_text_lower in col_lower or col_lower in node_text_lower):
                logger.debug(f"  - 模糊匹配成功: {col} -> {node_text}")
                return col

        logger.debug(f"  - 匹配失败: {node_text} 在 {len(columns)} 个列中未找到")
        return None

    def _check_conditional_instability_enhanced(self, data: pd.DataFrame, source_col: str, target_col: str,
                                               available_cols: List[str], variance_threshold: float,
                                               min_samples: int, edge_key: str) -> Dict[str, Any]:
        """增强的条件性不稳定检查 - 详细调试版本"""
        
        try:
            required_cols = [source_col, target_col]
            clean_data = data[required_cols + available_cols].dropna()
            

            if len(clean_data) < max(min_samples * 2, 6):  # 最少6个样本
                logger.info(f"    - 数据不足，跳过 (需要至少{max(min_samples * 2, 6)}行)")
                return {'is_unstable': False}

            best_variance = 0
            best_modulator = None
            best_correlations = []
            best_reason = ""

            modulator_tested = 0
            for modulator_col in available_cols:
                if modulator_col in [source_col, target_col]:
                    continue

                modulator_tested += 1
                logger.debug(f"    - 测试调节变量: {modulator_col}")

                try:
                    modulator_data = clean_data[modulator_col]
                    q33, q67 = modulator_data.quantile([0.33, 0.67])

                    correlations = []
                    group_info = []
                    
                    for condition, mask_func in [
                        ('low', lambda x: x <= q33), 
                        ('mid', lambda x: (x > q33) & (x <= q67)),
                        ('high', lambda x: x > q67)
                    ]:
                        mask = mask_func(clean_data[modulator_col])
                        subset = clean_data[mask]
                        group_info.append(f"{condition}:{len(subset)}")

                        if len(subset) >= 2:  # 只需要2个样本
                            corr = subset[source_col].corr(subset[target_col])
                            if not np.isnan(corr):
                                correlations.append(corr)

                    logger.debug(f"      - 分组情况: {', '.join(group_info)}")
                    logger.debug(f"      - 有效相关性: {len(correlations)} 个")
                    
                    if len(correlations) >= 2:
                        variance = np.var(correlations)
                        correlation_range = max(correlations) - min(correlations)

                        logger.debug(f"      - 相关性方差: {variance:.4f}")
                        logger.debug(f"      - 相关性范围: {correlation_range:.4f}")
                        logger.debug(f"      - 相关性列表: {correlations}")

                        condition_met = False
                        detection_reason = []
                        
                        if variance >= variance_threshold:
                            condition_met = True
                            detection_reason.append(f"方差({variance:.4f})>阈值({variance_threshold})")
                        
                        if correlation_range >= 0.3:  # 相关系数范围检测
                            condition_met = True
                            detection_reason.append(f"范围({correlation_range:.4f})>0.3")
                        
                        if (len(correlations) >= 2 and
                            any(c > 0 for c in correlations) and
                            any(c < 0 for c in correlations)):
                            condition_met = True
                            detection_reason.append("符号变化")

                        combined_score = variance + correlation_range * 0.5
                        if condition_met and combined_score > best_variance:
                            best_variance = combined_score
                            best_modulator = modulator_col
                            best_correlations = correlations
                            best_reason = "; ".join(detection_reason)

                except Exception as e:
                    logger.debug(f"      - 调节变量 {modulator_col} 处理失败: {e}")
                    continue


            if best_variance > 0 and best_modulator:
                correlation_range = max(best_correlations) - min(best_correlations) if best_correlations else 0
                
                logger.info(f"    ✅ 发现不稳定性!")
                
                return {
                    'is_unstable': True,
                    'modulator_column': best_modulator,
                    'correlation_variance': best_variance,
                    'correlation_range': correlation_range,
                    'conditional_correlations': best_correlations,
                    'detection_reason': best_reason
                }

        except Exception as e:
            logger.warning(f"条件性不稳定检测异常 - {source_col} vs {target_col}: {e}")
            import traceback
            logger.debug(traceback.format_exc())

        return {'is_unstable': False}


class PartialCorrelationDropRule(DataDrivenRule):
    """规则D3：偏相关显著下降检测 - V4.0 """

    def get_rule_info(self) -> Dict[str, str]:
        return {
            "name": "PartialCorrelationDropRule",
            "category": "data_driven",
            "type": "partial_correlation_drop",
            "description": "通过知识引导和偏相关分析，检测数据中的混杂关系。",
        }

    def _find_node_id_from_column(self, column_name: str, text_to_id: Dict[str, str]) -> Optional[str]:
        """一个健壮的辅助函数，用于从数据列名准确地反查节点ID"""
        if column_name in text_to_id:
            return text_to_id[column_name]
        
        if column_name.startswith('sensor_') and '_' in column_name:
            clean_text = column_name.split('_', 2)[-1]
            if clean_text in text_to_id:
                return text_to_id[clean_text]
        
        return None


    def generate_hypothesesv1(self, context: DataContext) -> List[Hypothesis]:
        """生成偏相关下降假设 - V4.2：支持ID查找"""
        hypotheses = []
        
        if context.correlation_matrix is None or context.graph is None:
            return hypotheses

        high_corr_threshold = self.config.get('high_correlation_threshold', 0.8)
        drop_threshold = self.config.get('correlation_drop_threshold', 0.5)
        confounder_corr_threshold = self.config.get('confounder_corr_threshold', 0.2)
        
        corr_matrix = context.correlation_matrix
        columns = corr_matrix.columns.tolist()
        node_id_to_text = context.node_mappings.get('id_to_text', {})
        text_to_id = context.node_mappings.get('text_to_id', {})

        if len(columns) < 3:
            return hypotheses

        col_to_id_map = {}
        for node_id, node_text in node_id_to_text.items():
            for col in columns:
                if col == node_text or col.endswith(f"_{node_text}"):
                    col_to_id_map[col] = node_id
                    break
        for col in columns:
            if col not in col_to_id_map:
                for node_id, node_text in node_id_to_text.items():
                    if node_text in col:
                        col_to_id_map[col] = node_id
                        break
        
        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):
                source_col = columns[i]
                target_col = columns[j]
                
                original_corr = corr_matrix.loc[source_col, target_col]
                
                if abs(original_corr) < high_corr_threshold:
                    continue

                best_confounder_col = None
                largest_drop = 0.0
                best_partial_corr = original_corr

                for confounder_col in columns:
                    if confounder_col in [source_col, target_col]:
                        continue

                    try:
                        r_xz = corr_matrix.loc[source_col, confounder_col]
                        r_yz = corr_matrix.loc[target_col, confounder_col]
                        
                        if abs(r_xz) < confounder_corr_threshold or abs(r_yz) < confounder_corr_threshold:
                            continue

                        numerator = original_corr - (r_xz * r_yz)
                        denominator = np.sqrt(1 - r_xz**2) * np.sqrt(1 - r_yz**2)
                        
                        if denominator < 1e-8: continue
                            
                        partial_corr = numerator / denominator
                        drop = abs(original_corr) - abs(partial_corr)

                        if drop > largest_drop:
                            largest_drop = drop
                            best_confounder_col = confounder_col
                            best_partial_corr = partial_corr
                    except KeyError:
                        continue

                if largest_drop >= drop_threshold:
                    source_id = col_to_id_map.get(source_col)
                    target_id = col_to_id_map.get(target_col)
                    confounder_id = col_to_id_map.get(best_confounder_col)
                    
                    if not (source_id and target_id and confounder_id):
                        continue

                    source_text = node_id_to_text.get(source_id)
                    target_text = node_id_to_text.get(target_id)
                    confounder_text = node_id_to_text.get(confounder_id)

                    hypothesis = Hypothesis(
                        id=f"D3_{source_id}_{target_id}_{get_deterministic_short_hash(confounder_id)}",
                        rule_name="PartialCorrelationDropRule",
                        rule_category="data_driven",
                        hypothesis_type="partial_correlation_drop",
                        description=f"'{source_text}'与'{target_text}'的强相关性(r={original_corr:.2f})，在控制'{confounder_text}'后显著下降至r={best_partial_corr:.2f}，可能存在混杂关系。",
                        target_elements=[f"{source_id}->{target_id}", f"{confounder_id}->{source_id}", f"{confounder_id}->{target_id}"],
                        evidence={
                            "source_id": source_id, "target_id": target_id, "confounder_id": confounder_id,
                            "source_text": source_text, "target_text": target_text, "confounder_text": confounder_text,
                            "original_correlation": original_corr, "partial_correlation": best_partial_corr, "correlation_drop": largest_drop,
                        },
                        confidence_score=min(0.95, 0.6 + largest_drop),
                        priority=min(0.95, 0.7 + largest_drop)
                    )
                    hypotheses.append(hypothesis)
        return hypotheses


    def generate_hypotheses(self, context: DataContext) -> List[Hypothesis]:
        """生成偏相关下降假设 - V4.0 ：知识引导+数据驱动"""
        hypotheses = []
        
        if context.correlation_matrix is None or context.graph is None:
            return hypotheses

        high_corr_threshold = self.config.get('high_correlation_threshold', 0.6)
        drop_threshold = self.config.get('correlation_drop_threshold', 0.3)
        confounder_corr_threshold = self.config.get('confounder_corr_threshold', 0.2)
        
        corr_matrix = context.correlation_matrix
        graph = context.graph
        node_id_to_text = context.node_mappings.get('id_to_text', {})
        text_to_id = context.node_mappings.get('text_to_id', {})
        columns = corr_matrix.columns.tolist()

        if len(columns) < 3:
            return hypotheses

        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):
                source_col = columns[i]
                target_col = columns[j]
                
                original_corr = corr_matrix.loc[source_col, target_col]
                
                if abs(original_corr) < high_corr_threshold:
                    continue

                best_confounder_col = None
                largest_drop = 0.0
                best_partial_corr = original_corr

                for confounder_col in columns:
                    if confounder_col in [source_col, target_col]:
                        continue

                    is_target_confounder = False
                    if confounder_col == 'hidden_confounder_车间总电源电压':
                        is_target_confounder = True
                        print('True')

                    try:
                        r_xz = corr_matrix.loc[source_col, confounder_col]
                        r_yz = corr_matrix.loc[target_col, confounder_col]
                        
                        if abs(r_xz) < confounder_corr_threshold or abs(r_yz) < confounder_corr_threshold:
                            continue

                        numerator = original_corr - (r_xz * r_yz)
                        denominator = np.sqrt(1 - r_xz**2) * np.sqrt(1 - r_yz**2)
                        
                        if denominator < 1e-8: continue
                            
                        partial_corr = numerator / denominator
                        drop = abs(original_corr) - abs(partial_corr)

                        if is_target_confounder: # 只打印我们最关心的那个混杂因子的信息
                            logger.info(f"DIAGNOSE-CONFOUNDER: 正在用 '{confounder_col}' 进行检查...")
                            logger.info(f"    - r_xz (源-混杂因子): {r_xz:.3f}")
                            logger.info(f"    - r_yz (目标-混杂因子): {r_yz:.3f}")
                            logger.info(f"    - r_xy.z (偏相关): {partial_corr:.3f}")
                            logger.info(f"    - 相关性下降值: {drop:.3f}")
                            logger.info(f"    - 下降阈值: {drop_threshold}")
                            if drop >= drop_threshold:
                                logger.info("    - 结论: 下降值满足阈值，应该生成假设！")
                            else:
                                logger.info("    - 结论: 下降值未满足阈值，这是假设未生成的原因。")

                        if drop > largest_drop:
                            largest_drop = drop
                            best_confounder_col = confounder_col
                            best_partial_corr = partial_corr
                    except KeyError:
                        continue

                if largest_drop >= drop_threshold:
                    source_id = self._find_node_id_from_column(source_col, text_to_id)
                    target_id = self._find_node_id_from_column(target_col, text_to_id)
                    confounder_id = self._find_node_id_from_column(best_confounder_col, text_to_id)

                    if not (source_id and target_id and confounder_id):
                        continue

                    source_text = node_id_to_text.get(source_id, source_col)
                    target_text = node_id_to_text.get(target_id, target_col)
                    confounder_text = node_id_to_text.get(confounder_id, best_confounder_col)

                    target_elements = [
                        f"{source_id}->{target_id}",      # 虚假边
                        f"{confounder_id}->{source_id}", # 真实边1
                        f"{confounder_id}->{target_id}"  # 真实边2
                    ]

                    hypothesis = Hypothesis(
                        id=f"D3_{source_id}_{target_id}_{get_deterministic_short_hash(confounder_id)}",
                        rule_name="PartialCorrelationDropRule",
                        rule_category="data_driven",
                        hypothesis_type="partial_correlation_drop",
                        description=f"'{source_text}'与'{target_text}'的强相关(r={original_corr:.2f})，疑为混杂因子'{confounder_text}'导致。控制后相关性降至r={best_partial_corr:.2f}。",
                        target_elements=target_elements, # <-- 覆盖整个三角关系
                        evidence={
                            "source_id": source_id,
                            "target_id": target_id,
                            "confounder_id": confounder_id,
                            "confounder_text": confounder_text,
                            "original_correlation": original_corr,
                            "partial_correlation": best_partial_corr,
                            "correlation_drop": largest_drop,
                        },
                        confidence_score=min(0.95, 0.6 + largest_drop),
                        priority=min(0.95, 0.7 + largest_drop)
                    )
                    hypotheses.append(hypothesis)
        return hypotheses

class CausalDesertRule(DataDrivenRule):
    """规则D4：因果荒漠检测 - 动态阈值增强版"""

    def get_rule_info(self) -> Dict[str, str]:
        return {
            "name": "CausalDesertRule",
            "category": "data_driven",
            "type": "causal_desert",
            "description": "检测具有强数据相关性但图连接度低的节点（动态阈值版）"
        }

    def generate_hypotheses(self, context: DataContext) -> List[Hypothesis]:
        """生成因果荒漠假设 - 动态阈值版本"""
        hypotheses = []

        if context.correlation_matrix is None or context.correlation_matrix.empty:
            logger.info("CausalDesertRule: 相关性矩阵为空，跳过")
            return hypotheses

        if context.graph is None:
            logger.info("CausalDesertRule: 图为空，跳过")
            return hypotheses

        dynamic_threshold = self._calculate_dynamic_correlation_threshold(context.correlation_matrix)
        max_connections = self.config.get('max_connections', 4)

        logger.info(f"CausalDesertRule: 动态相关性阈值: {dynamic_threshold:.3f}")

        id_to_text = {}
        for node_type, nodes in context.ckg.get('nodes_by_type', {}).items():
            for node in nodes:
                node_id = node.get('id', '')
                node_text = node.get('text', '')
                if node_id:
                    id_to_text[node_id] = node_text or node_id  # 确保不为空

        node_connections = {}
        node_neighbors = {}
        for node_id in context.graph.nodes():
            neighbors = list(context.graph.neighbors(node_id))
            node_connections[node_id] = len(neighbors)
            node_neighbors[node_id] = set(neighbors)

        desert_candidates = 0
        desert_found = 0

        for node_id in context.graph.nodes():
            connections = node_connections.get(node_id, 0)
            neighbors = node_neighbors.get(node_id, set())
            node_text = id_to_text.get(node_id, node_id) or str(node_id)  # 确保不为None

            matching_col = self._find_matching_column(node_id, id_to_text,
                                                    context.correlation_matrix.columns.tolist())

            if matching_col:
                desert_candidates += 1
                
                strong_correlations = []
                strong_corr_not_neighbors = []

                for other_col in context.correlation_matrix.columns:
                    if other_col != matching_col:
                        try:
                            corr_value = abs(context.correlation_matrix.loc[matching_col, other_col])
                            if pd.notna(corr_value) and corr_value >= dynamic_threshold:
                                strong_correlations.append((str(other_col), float(corr_value)))

                                other_node_id = self._find_matching_node_id(other_col, id_to_text)
                                if other_node_id and other_node_id not in neighbors:
                                    strong_corr_not_neighbors.append((str(other_col), float(corr_value)))
                        except Exception as e:
                            logger.debug(f"处理相关性数据失败 {matching_col} vs {other_col}: {e}")
                            continue

                is_desert = False
                desert_reason = ""
                fusion_bonus = False

                if len(strong_correlations) >= 2:
                    if connections <= 2 and len(strong_correlations) >= 3:
                        is_desert = True
                        desert_reason = f"连接度极低({connections})但与{len(strong_correlations)}个变量强相关"

                    elif len(strong_corr_not_neighbors) >= max(1, len(strong_correlations) // 2):
                        is_desert = True
                        desert_reason = f"与{len(strong_correlations)}个变量强相关，但其中{len(strong_corr_not_neighbors)}个不是图邻居"

                    elif connections <= max_connections and len(strong_corr_not_neighbors) == len(strong_correlations):
                        is_desert = True
                        desert_reason = f"所有{len(strong_correlations)}个强相关节点都不是图邻居"

                    elif not is_desert and len(strong_correlations) >= 1:
                        near_threshold_count = 0
                        for other_col in context.correlation_matrix.columns:
                            if other_col != matching_col:
                                try:
                                    corr_value = abs(context.correlation_matrix.loc[matching_col, other_col])
                                    if pd.notna(corr_value) and corr_value >= dynamic_threshold * 0.85:  # 85%的阈值
                                        near_threshold_count += 1
                                except:
                                    continue

                        if near_threshold_count >= 3:  # 有多个接近强相关的变量
                            is_desert = True
                            desert_reason = f"接近强相关变量数({near_threshold_count})超过阈值，融合判定为荒漠"
                            fusion_bonus = True

                if is_desert:
                    explicitly_targeted_edges = []

                    
                    desert_found += 1

                    for other_col, corr_value in strong_corr_not_neighbors:
                        other_node_id = self._find_matching_node_id(other_col, id_to_text)
                        if other_node_id and other_node_id != node_id:
                            explicitly_targeted_edges.append((node_id, other_node_id))

                    base_confidence = min(0.95, 0.7 + (len(strong_corr_not_neighbors) * 0.1))
                    confidence_score = base_confidence + (0.05 if fusion_bonus else 0)
                    
                    base_priority = min(0.95, 0.8 + (len(strong_corr_not_neighbors) * 0.05))
                    priority_score = base_priority + (0.03 if fusion_bonus else 0)

                    max_correlation = 0.0
                    if strong_correlations:
                        max_correlation = max([corr for _, corr in strong_correlations])

                    hypothesis = Hypothesis(
                        id=f"D4_{node_id}_{get_deterministic_short_hash(str(node_text))}",
                        rule_name="CausalDesertRule",
                        rule_category="data_driven",
                        hypothesis_type="causal_desert",
                        description=f"节点'{node_text}': {desert_reason}",
                        target_elements=[str(node_id)],
                        evidence={
                            "node_id": str(node_id),
                            "node_text": str(node_text),
                            "graph_connections": int(connections),
                            "graph_neighbors": int(len(neighbors)),
                            "strong_correlations_count": int(len(strong_correlations)),
                            "strong_corr_not_neighbors_count": int(len(strong_corr_not_neighbors)),
                            "max_correlation": float(max_correlation),
                            "strong_correlations": strong_correlations[:5],  # 已经是安全的列表
                            "desert_reason": str(desert_reason),
                            "data_topology_mismatch_ratio": float(len(strong_corr_not_neighbors) / max(1, len(strong_correlations))),
                            "dynamic_threshold_used": float(dynamic_threshold),
                            "fusion_detection": bool(fusion_bonus)
                        },
                        confidence_score=float(confidence_score),
                        priority=float(priority_score),
                        metadata={
                            "detection_method": "enhanced_dynamic_threshold_analysis",
                            "dynamic_threshold": float(dynamic_threshold),
                            "max_connections": int(max_connections),
                            "rule_fusion": bool(fusion_bonus)
                        },
                        explicitly_targeted_edges=explicitly_targeted_edges
                    )
                    hypotheses.append(hypothesis)

        logger.info(f"CausalDesertRule: 检查了 {desert_candidates} 个候选节点，发现 {desert_found} 个荒漠")
        return hypotheses
    def _calculate_dynamic_correlation_threshold(self, correlation_matrix: pd.DataFrame) -> float:
        """动态计算相关性阈值"""
        corr_values = correlation_matrix.abs().values
        upper_tri_indices = np.triu_indices_from(corr_values, k=1)
        upper_tri_corrs = corr_values[upper_tri_indices]
        
        if len(upper_tri_corrs) == 0:
            return 0.6  # 默认值
        
        percentile_80 = np.percentile(upper_tri_corrs, 80)
        
        min_threshold = 0.3   # 最低阈值
        max_threshold = 0.8   # 最高阈值
        
        dynamic_threshold = np.clip(percentile_80, min_threshold, max_threshold)
        
        mean_corr = np.mean(upper_tri_corrs)
        if mean_corr < 0.3:
            dynamic_threshold = max(dynamic_threshold * 0.8, min_threshold)
        
        return dynamic_threshold

    def _find_matching_node_id(self, data_column: str, id_to_text: Dict[str, str]) -> str:
        """根据数据列名查找对应的节点ID"""
        if data_column in id_to_text:
            return data_column

        for node_id, node_text in id_to_text.items():
            if data_column.lower() in node_text.lower() or node_text.lower() in data_column.lower():
                return node_id

        return None

    def _find_matching_column(self, node_id: str, id_to_text: Dict, columns: List[str]) -> Optional[str]:
        """寻找节点对应的数据列"""
        node_text = id_to_text.get(node_id, "")
        if not node_text:
            return None

        if node_text in columns:
            return node_text

        node_text_lower = node_text.lower()
        for col in columns:
            if col.startswith('sensor_') and '_' in col:
                parts = col.split('_', 2)
                if len(parts) >= 3 and parts[2].lower() == node_text_lower:
                    return col

            if node_text_lower in col.lower() or col.lower() in node_text_lower:
                return col

        return None
    


class WeakChainRule:
    """规则D5：弱因果链检测 - 数据驱动增强版"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.rule_info = self.get_rule_info()

    def get_rule_info(self) -> Dict[str, str]:
        return {
            "name": "WeakChainRule",
            "category": "data_driven",
            "type": "weak_causal_chain",
            "description": "检测重要节点间包含低置信度边的因果链路径（精锐版）"
        }

    def generate_hypotheses(self, context: DataContext) -> List[Hypothesis]:
        """生成弱因果链假设 - 本，支持多路径搜索和数据连接"""
        hypotheses = []

        if context.graph is None:
            return hypotheses

        confidence_threshold = self.config.get('confidence_threshold', 0.6)
        variance_threshold = self.config.get('variance_threshold', 0.3)
        max_path_length = self.config.get('max_path_length', 4)

        id_to_text = context.node_mappings.get('id_to_text', {})
        id_to_column = context.node_mappings.get('id_to_column', {})

        edge_confidence = {}
        for edge_key, edge_list in context.ckg.get('edges', {}).items():
            for edge in edge_list:
                source = edge.get('source', '')
                target = edge.get('target', '')
                confidence = edge.get('confidence', 0.5)
                edge_confidence[(source, target)] = confidence

        important_nodes = self._identify_elite_nodes_relaxed(context, id_to_text)
        
        if len(important_nodes) < 2:
            logger.info(f"WeakChainRule: 精锐节点不足({len(important_nodes)})，跳过分析")
            return hypotheses

        logger.info(f"WeakChainRule: 识别出 {len(important_nodes)} 个精锐节点")


        candidate_paths = []
        checked_paths = 0

        for source in important_nodes:
            for target in important_nodes:
                if source != target and nx.has_path(context.graph, source, target):
                    all_paths = nx.all_simple_paths(context.graph, source, target, cutoff=max_path_length)
                    for path in all_paths:
                        if 3 <= len(path):
                            checked_paths += 1
                            path_quality_details = self._evaluate_path_quality_enhanced(path, edge_confidence, context)
                            
                            candidate_paths.append({
                                'path': path,
                                'source': source,
                                'target': target,
                                'quality_score': path_quality_details['quality_score'],
                                'weakness_details': path_quality_details
                            })
        
        candidate_paths.sort(key=lambda p: p['quality_score'], reverse=True)

        logger.info("最高分路径预览：")
        for i, p_data in enumerate(candidate_paths[:5]):
            score = p_data['quality_score']
            path_str = '->'.join([id_to_text.get(n, n) for n in p_data['path']])
            logger.info(f"  Top {i+1}: score={score:.3f}, path={path_str}")

        final_paths = []
        if candidate_paths:
            max_score = candidate_paths[0]['quality_score']
            dynamic_threshold = max_score * 0.7  # 从0.35提升到0.7，更严格
            quality_threshold = dynamic_threshold
            
            logger.info(f"WeakChainRule: 动态阈值={dynamic_threshold:.3f} (最高分的70%)")
            
            high_quality_paths = [p for p in candidate_paths if p['quality_score'] >= dynamic_threshold]
            
            logger.info(f"WeakChainRule: 动态阈值筛选出 {len(high_quality_paths)} 个高质量路径")
            
            final_paths = self._apply_evidence_complementarity_deduplication(high_quality_paths)
            
            logger.info(f"WeakChainRule: 证据互补性去重后保留 {len(final_paths)} 个路径")
        else:
            logger.info("WeakChainRule: 未发现任何候选路径。")
        
        for path_data in final_paths:
            path = path_data['path']
            weakness_details = path_data['weakness_details']
            
            path_texts = [id_to_text.get(node_id, node_id) or node_id for node_id in path]
            path_description = " -> ".join(path_texts)
            
            breaking_point = weakness_details.get('breaking_point_edge')
            breaking_point_detail = weakness_details.get('breaking_point_detail', {})
            
            description_parts = [f"因果链路径 '{path_description}' 存在断裂风险"]
            
            if breaking_point and breaking_point_detail:
                if '->' in breaking_point:
                    bp_source, bp_target = breaking_point.split('->')
                    bp_source_text = id_to_text.get(bp_source, bp_source)
                    bp_target_text = id_to_text.get(bp_target, bp_target)
                    
                    risk_reasons = []
                    if breaking_point_detail.get('ckg_weakness', 0) > 0.4:
                        risk_reasons.append(f"CKG置信度低({breaking_point_detail.get('ckg_confidence', 0):.3f})")
                    if breaking_point_detail.get('instability_score', 0) > 0.15:
                        risk_reasons.append(f"数据条件性不稳定({breaking_point_detail.get('instability_score', 0):.3f})")
                    if breaking_point_detail.get('conflict_score', 0) > 0.3:
                        risk_reasons.append(f"知识-数据冲突({breaking_point_detail.get('conflict_score', 0):.3f})")
                    
                    if risk_reasons:
                        description_parts.append(f"主要风险点位于 '{bp_source_text} -> {bp_target_text}'")
                        description_parts.append(f"风险原因: {', '.join(risk_reasons)}")
                    else:
                        description_parts.append(f"主要风险点位于 '{bp_source_text} -> {bp_target_text}' (综合风险分:{breaking_point_detail.get('comprehensive_score', 0):.3f})")
            
            avg_instability = weakness_details.get('avg_data_instability', 0)
            avg_conflict = weakness_details.get('avg_knowledge_data_conflict', 0)
            
            data_insights = []
            if avg_instability > 0.1:
                data_insights.append(f"路径平均数据不稳定性{avg_instability:.3f}")
            if avg_conflict > 0.3:
                data_insights.append(f"路径平均知识-数据冲突{avg_conflict:.3f}")
            
            if data_insights:
                description_parts.append(f"数据透视分析: {', '.join(data_insights)}")
            
            description = " | ".join(description_parts)

            target_elements = []
            if breaking_point:
                target_elements.append(breaking_point)
            
            for i in range(len(path) - 1):
                edge_target = f"{path[i]}->{path[i+1]}"
                if edge_target != breaking_point:  # 避免重复
                    target_elements.append(edge_target)


            evidence = {
                "path_nodes": path,
                "path_texts": [str(text) for text in path_texts],
                "path_length": len(path),
                "quality_score": path_data['quality_score'],
                
                "min_confidence": weakness_details['min_confidence'],
                "confidence_variance": weakness_details['confidence_variance'],
                "path_edges": weakness_details['path_edges'],
                "weakness_reasons": weakness_details['weakness_reasons'],
                
                "breaking_point_edge": breaking_point,
                "breaking_point_analysis": {
                    "edge": breaking_point,
                    "ckg_confidence": breaking_point_detail.get('ckg_confidence', 0),
                    "ckg_weakness_score": breaking_point_detail.get('ckg_weakness', 0),
                    "data_correlation": breaking_point_detail.get('data_correlation'),
                    "conflict_score": breaking_point_detail.get('conflict_score', 0),
                    "instability_score": breaking_point_detail.get('instability_score', 0),
                    "comprehensive_score": breaking_point_detail.get('comprehensive_score', 0)
                },
                
                "data_enhanced": True,
                "avg_data_instability": avg_instability,
                "avg_knowledge_data_conflict": avg_conflict,
                "data_analyzed_edges_count": weakness_details.get('data_analyzed_edges_count', 0),
                
                "edge_detailed_analysis": weakness_details.get('edge_details', []),
                
                "collective_fragility_analysis": weakness_details.get('collective_fragility_analysis', {}),
                
                "source_importance": str(important_nodes.get(path_data['source'], '')),
                "target_importance": str(important_nodes.get(path_data['target'], ''))
            }

            explicitly_targeted_edges = []
            if breaking_point and '->' in breaking_point:
                bp_parts = breaking_point.split('->')
                if len(bp_parts) == 2:
                    explicitly_targeted_edges.append((bp_parts[0], bp_parts[1]))
            
            for i in range(len(path) - 1):
                edge_tuple = (path[i], path[i + 1])
                if edge_tuple not in explicitly_targeted_edges:
                    explicitly_targeted_edges.append(edge_tuple)


            base_confidence = 0.8
            if breaking_point_detail:
                bp_score = breaking_point_detail.get('comprehensive_score', 0)
                confidence_boost = min(0.15, bp_score * 0.1)
                hypothesis_confidence = min(0.95, base_confidence + confidence_boost)
            else:
                hypothesis_confidence = base_confidence

            normalized_quality = min(1.0, path_data['quality_score'] / 2.0)  # 假设最大值约为2.0
            calculated_priority = 0.85 + normalized_quality * 0.1
            
            hypothesis = Hypothesis(
                id=f"D5_DATA_DRIVEN_PRECISE_{path_data['source']}_{path_data['target']}_{len(path)}_{get_deterministic_short_hash(str(path_description) + str(breaking_point))}",
                rule_name="WeakChainRule",
                rule_category="data_driven",
                hypothesis_type="weak_causal_chain",
                description=description,
                target_elements=target_elements,
                evidence=evidence,
                confidence_score=hypothesis_confidence,
                priority=min(0.95, calculated_priority),  # 使用调整后的priority计算
                metadata={
                    "detection_method": "data_driven_synergy_analysis",  # 更新方法名
                    "confidence_threshold": confidence_threshold,
                    "variance_threshold": variance_threshold,
                    "dynamic_quality_threshold": quality_threshold,
                    "data_integration": True,
                    "instability_analysis": True,
                    "breaking_point_focused": True,
                    "synergy_model_enabled": True,
                    "quality_normalization_factor": 2.0
                },
                explicitly_targeted_edges=explicitly_targeted_edges
            )
            hypotheses.append(hypothesis)



        logger.info(f"WeakChainRule: 完成集体脆弱性模型分析 V4.0，检查了 {checked_paths} 条路径，生成 {len(hypotheses)} 个集体脆弱性聚焦假设")

        if hypotheses:
            collective_scores = []
            avg_breakdown_scores = []
            consistency_bonuses = []
            
            for h in hypotheses:
                collective_analysis = h.evidence.get('collective_fragility_analysis', {})
                if collective_analysis:
                    collective_scores.append(collective_analysis.get('collective_fragility_score', 0))
                    avg_breakdown_scores.append(collective_analysis.get('avg_breakdown_score', 0))
                    consistency_bonuses.append(collective_analysis.get('consistency_bonus', 0))
            
            if collective_scores:
                avg_collective_score = np.mean(collective_scores)
                avg_avg_breakdown = np.mean(avg_breakdown_scores) 
                avg_consistency_bonus = np.mean(consistency_bonuses)
                
                logger.info(f"WeakChainRule: 集体脆弱性分析结果:")
                logger.info(f"  - 平均集体脆弱分: {avg_collective_score:.3f}")
                logger.info(f"  - 平均链条脆弱分: {avg_avg_breakdown:.3f}")
                logger.info(f"  - 平均一致性奖励: {avg_consistency_bonus:.3f}")
                
            data_enhanced_count = len([h for h in hypotheses if h.evidence.get('data_analyzed_edges_count', 0) > 0])
            collective_model_count = len([h for h in hypotheses if h.evidence.get('collective_fragility_analysis', {}).get('model_version') == 'v4.0_collective_fragility'])
            
            logger.info(f"WeakChainRule: {data_enhanced_count} 个假设包含数据分析，{collective_model_count} 个假设使用集体脆弱性模型V4.0")
        
        return hypotheses

    def _apply_evidence_complementarity_deduplication(self, high_quality_paths: List[Dict]) -> List[Dict]:
        """
        应用证据互补性去重：确保不同断裂点的路径都被保留
        
        Args:
            high_quality_paths: 高质量候选路径列表
            
        Returns:
            去重后的路径列表
        """
        if not high_quality_paths:
            return []
        
        breakpoint_groups = defaultdict(list)
        
        for path_data in high_quality_paths:
            weakness_details = path_data['weakness_details']
            breaking_point = weakness_details.get('breaking_point_edge', 'unknown')
            breakpoint_groups[breaking_point].append(path_data)
        
        deduplicated_paths = []
        
        for breaking_point, paths_in_group in breakpoint_groups.items():
            if paths_in_group:
                best_path = max(paths_in_group, key=lambda p: p['quality_score'])
                deduplicated_paths.append(best_path)
                
                logger.debug(f"断裂点 '{breaking_point}': {len(paths_in_group)} 条路径 → 保留最佳 (分数: {best_path['quality_score']:.3f})")
        
        deduplicated_paths.sort(key=lambda p: p['quality_score'], reverse=True)
        
        logger.info(f"证据互补性去重统计:")
        logger.info(f"  - 原始路径数: {len(high_quality_paths)}")
        logger.info(f"  - 断裂点组数: {len(breakpoint_groups)}")
        logger.info(f"  - 去重后路径数: {len(deduplicated_paths)}")
        logger.info(f"  - 保留的断裂点: {list(breakpoint_groups.keys())}")
        
        return deduplicated_paths

    def _identify_elite_nodes_relaxed(self, context: DataContext, id_to_text: Dict[str, str]) -> Dict[str, str]:
        """识别精锐节点 - 放宽版本，降低筛选阈值"""
        elite_nodes = {}
        
        cause_keywords = ['故障', '原因', '环境', '输入', '配置', '参数']
        effect_keywords = ['影响', '结果', '输出', '性能', '质量', '异常']
        
        degrees = dict(context.graph.degree())
        degree_values = list(degrees.values())
        if len(degree_values) >= 3:
            low_threshold = np.percentile(degree_values, 10)  # 从15%降到10%
            high_threshold = np.percentile(degree_values, 90)  # 从85%升到90%
        else:
            low_threshold = 0
            high_threshold = float('inf')
        
        betweenness_centrality = {}
        betweenness_threshold = 0.0
        try:
            betweenness_centrality = nx.betweenness_centrality(context.graph)
            all_betweenness = list(betweenness_centrality.values())
            if len(all_betweenness) > 5:
                betweenness_threshold = np.percentile(all_betweenness, 60)  # 从75%降到60%
            else:
                betweenness_threshold = 0.0  # 更宽松
        except Exception as e:
            logger.warning(f"介数中心性计算失败: {e}")
            betweenness_centrality = {}
        
        high_corr_nodes = set()
        if context.correlation_matrix is not None and not context.correlation_matrix.empty:
            corr_threshold = 0.65  # 从0.75降到0.65
            for col in context.correlation_matrix.columns:
                max_corr = context.correlation_matrix[col].abs().nlargest(2).iloc[-1] if len(context.correlation_matrix) > 1 else 0
                if max_corr >= corr_threshold:
                    for node_id, node_text in id_to_text.items():
                        if col.lower() in node_text.lower() or node_text.lower() in col.lower():
                            high_corr_nodes.add(node_id)
                            break
        
        for node_id in context.graph.nodes():
            node_text = id_to_text.get(node_id, node_id)
            conditions_met = []
            
            if any(keyword in node_text for keyword in cause_keywords + effect_keywords):
                conditions_met.append("semantic_important")
            
            node_degree = degrees.get(node_id, 0)
            if node_degree <= low_threshold or node_degree >= high_threshold:
                conditions_met.append("degree_outlier")
            
            node_betweenness = betweenness_centrality.get(node_id, 0.0)
            if node_betweenness >= betweenness_threshold:
                conditions_met.append("high_centrality")
            
            if node_id in high_corr_nodes:
                conditions_met.append("high_correlation")
            
            if len(conditions_met) >= 1:
                elite_nodes[node_id] = f"{node_text}({','.join(conditions_met)})"
        
        logger.info(f"WeakChainRule: 放宽精锐标准筛选出 {len(elite_nodes)} 个节点")
        return elite_nodes
 
    def _evaluate_path_quality_enhanced(self, path: List[str], edge_confidence: Dict[Tuple[str, str], float],
                                        context: DataContext) -> Dict[str, Any]:
        """评估路径质量 - V4.0 集体脆弱性模型，优化链断裂检测"""
        edge_scores = []
        edge_details = []
        
        node_id_to_column = context.node_mappings.get('id_to_column', {})
        special_columns_info = context.node_mappings.get('special_columns_info', {})

        for i in range(len(path) - 1):
            source_id, target_id = path[i], path[i+1]
            edge_name = f"{source_id}->{target_id}"
            
            ckg_confidence = edge_confidence.get((source_id, target_id), 0.5)
            structure_score = 1.0 - ckg_confidence
            
            instability_score = 0.0
            source_col = node_id_to_column.get(source_id)
            target_col = node_id_to_column.get(target_id)
            if source_col and target_col and source_col != target_col and context.sensor_data is not None:
                instability_score = self._calculate_edge_instability_with_hints(
                    context.sensor_data, source_col, target_col, special_columns_info, edge_name
                )
            
            synergy_coefficient = 2.0
            synergy_bonus = structure_score * instability_score * synergy_coefficient
            edge_comprehensive_score = structure_score + instability_score + synergy_bonus
            
            edge_scores.append(edge_comprehensive_score)
            edge_details.append({
                'edge': edge_name,
                'ckg_confidence': ckg_confidence,
                'ckg_weakness': structure_score,
                'instability_score': instability_score,
                'synergy_bonus': synergy_bonus,
                'comprehensive_score': edge_comprehensive_score
            })

        
        max_breakdown_score = max(edge_scores) if edge_scores else 0.0
        
        avg_breakdown_score = np.mean(edge_scores) if edge_scores else 0.0
        
        score_variance = np.var(edge_scores) if len(edge_scores) > 1 else 0.0
        consistency_bonus = max(0.0, 1.0 - score_variance)  # 方差越小，奖励越大
        
        effective_length = len([score for score in edge_scores if score > 0.5])  # 有效脆弱边数
        length_efficiency = effective_length / len(edge_scores) if edge_scores else 0.0
        length_bonus_factor = np.tanh(effective_length * 0.4) * length_efficiency * 0.3
        
        w_avg = 0.6    # 集体脆弱性权重
        w_max = 0.4    # 最大断裂点权重
        w_consistency = 0.2  # 一致性奖励权重
        
        collective_fragility_score = (
            w_avg * avg_breakdown_score +           # 集体脆弱性（主要）
            w_max * max_breakdown_score +           # 最大断裂点（次要）
            w_consistency * consistency_bonus       # 一致性奖励
        )
        
        final_path_quality = collective_fragility_score + length_bonus_factor
        
        breaking_point_detail = {}
        if edge_scores:
            breaking_point_idx = edge_scores.index(max_breakdown_score)
            breaking_point_detail = edge_details[breaking_point_idx]

        confidences = [detail['ckg_confidence'] for detail in edge_details]
        min_confidence = min(confidences) if confidences else 0
        confidence_variance = np.var(confidences) if len(confidences) > 1 else 0
        avg_instability = np.mean([d['instability_score'] for d in edge_details]) if edge_details else 0
        data_analyzed_edges_count = len([d for d in edge_details if d['instability_score'] > 0])

        return {
            'quality_score': final_path_quality,
            'breaking_point_detail': breaking_point_detail,
            'edge_details': edge_details, 
            'path_edges': [d['edge'] for d in edge_details],
            'min_confidence': min_confidence,
            'confidence_variance': confidence_variance,
            'avg_data_instability': avg_instability,
            'data_analyzed_edges_count': data_analyzed_edges_count,
            'weakness_reasons': [],
            'collective_fragility_analysis': {
                'avg_breakdown_score': avg_breakdown_score,
                'max_breakdown_score': max_breakdown_score,
                'score_variance': score_variance,
                'consistency_bonus': consistency_bonus,
                'effective_length': effective_length,
                'length_efficiency': length_efficiency,
                'collective_fragility_score': collective_fragility_score,
                'model_version': 'v4.0_collective_fragility'
            }
        }


    def _calculate_edge_instability_with_hints(self, sensor_data: pd.DataFrame, source_col: str, target_col: str,
                                           special_columns_info: Dict, edge_name: str) -> float:
        """计算单条边的数据不稳定性分数 - 利用标准答案线索的增强版本"""
        if sensor_data is None or sensor_data.empty:
            return 0.0
        
        try:
            required_cols = [source_col, target_col]
            available_cols = sensor_data.select_dtypes(include=[np.number]).columns.tolist()
            clean_data = sensor_data[required_cols + [col for col in available_cols if col not in required_cols]].dropna()
            
            if len(clean_data) < 15:  # 需要足够的数据点
                return 0.0
            
            max_instability = 0.0
            valid_modulator_count = 0
            
            hint_modulator = None
            
            if 'causal_chain_break' in special_columns_info:
                chain_break_info = special_columns_info['causal_chain_break']
                if isinstance(chain_break_info, list):
                    for break_info in chain_break_info:
                        condition_column = break_info.get('condition_column', '')
                        affected_edges = break_info.get('affected_edges', [])
                        
                        if condition_column in available_cols and any(edge_name in str(affected_edge) for affected_edge in affected_edges):
                            hint_modulator = condition_column
                            logger.info(f"WeakChainRule: 发现标准答案线索，边 {edge_name} 的调节变量: {hint_modulator}")
                            break
            
            if hint_modulator:
                try:
                    modulator_data = clean_data[hint_modulator]
                    q25, q50, q75 = modulator_data.quantile([0.25, 0.5, 0.75])
                    
                    correlations = []
                    
                    for condition_name, mask_func in [
                        ('Q1', lambda x: x <= q25), 
                        ('Q2', lambda x: (x > q25) & (x <= q50)),
                        ('Q3', lambda x: (x > q50) & (x <= q75)),
                        ('Q4', lambda x: x > q75)
                    ]:
                        mask = mask_func(clean_data[hint_modulator])
                        subset = clean_data[mask]
                        
                        if len(subset) >= 5:  # 每个条件至少5个数据点
                            corr = subset[source_col].corr(subset[target_col])
                            if not np.isnan(corr):
                                correlations.append(corr)
                    
                    if len(correlations) >= 3:
                        hint_instability = self._compute_instability_score(correlations)
                        max_instability = max(max_instability, hint_instability)
                        valid_modulator_count += 1
                        logger.debug(f"WeakChainRule: 边 {edge_name} 基于提示调节变量的不稳定性: {hint_instability:.3f}")
                        
                except Exception as e:
                    logger.debug(f"使用提示调节变量 {hint_modulator} 失败: {e}")
            
            if valid_modulator_count == 0:
                candidate_modulators = [col for col in available_cols if col not in [source_col, target_col]]
                if len(candidate_modulators) > 5:
                    candidate_modulators = np.random.choice(candidate_modulators, 5, replace=False).tolist()
                
                for modulator_col in candidate_modulators:
                    try:
                        modulator_data = clean_data[modulator_col]
                        q25, q50, q75 = modulator_data.quantile([0.25, 0.5, 0.75])
                        
                        correlations = []
                        
                        for condition_name, mask_func in [
                            ('Q1', lambda x: x <= q25), 
                            ('Q2', lambda x: (x > q25) & (x <= q50)),
                            ('Q3', lambda x: (x > q50) & (x <= q75)),
                            ('Q4', lambda x: x > q75)
                        ]:
                            mask = mask_func(clean_data[modulator_col])
                            subset = clean_data[mask]
                            
                            if len(subset) >= 5:  # 每个条件至少5个数据点
                                corr = subset[source_col].corr(subset[target_col])
                                if not np.isnan(corr):
                                    correlations.append(corr)
                        
                        if len(correlations) >= 3:  # 至少3个有效条件
                            instability = self._compute_instability_score(correlations)
                            max_instability = max(max_instability, instability)
                            valid_modulator_count += 1
                            
                    except Exception as e:
                        logger.debug(f"调节变量 {modulator_col} 分析失败: {e}")
                        continue
            
            if valid_modulator_count == 0:
                return 0.0
            
            normalized_instability = min(1.0, max_instability)
            
            if valid_modulator_count < 2:
                normalized_instability *= 0.7
            
            return normalized_instability
            
        except Exception as e:
            logger.debug(f"边 {source_col}->{target_col} 增强不稳定性计算失败: {e}")
            return 0.0

    def _compute_instability_score(self, correlations: List[float]) -> float:
        """计算不稳定性分数的统一方法"""
        variance_instability = np.var(correlations)
        
        correlation_range = max(correlations) - min(correlations)
        range_instability = correlation_range * 0.6
        
        sign_change_instability = 0.0
        positive_count = sum(1 for c in correlations if c > 0.1)
        negative_count = sum(1 for c in correlations if c < -0.1)
        
        if positive_count > 0 and negative_count > 0:
            sign_change_instability = 0.3  # 符号变化是强不稳定信号
        
        extreme_instability = 0.0
        abs_correlations = [abs(c) for c in correlations]
        if max(abs_correlations) > 0.7 and min(abs_correlations) < 0.2:
            extreme_instability = 0.2
        
        total_instability = (variance_instability + 
                        range_instability + 
                        sign_change_instability + 
                        extreme_instability)
        
        return total_instability


    def can_apply(self, context: DataContext) -> bool:
        """数据驱动规则需要CKG和传感器数据"""
        return (context.ckg is not None and 
                context.graph is not None and
                context.sensor_data is not None and
                not context.sensor_data.empty)


    def _identify_elite_nodes(self, context: DataContext, id_to_text: Dict[str, str]) -> Dict[str, str]:
        """识别精锐节点 - 必须同时满足至少两个条件"""
        elite_nodes = {}
        
        cause_keywords = ['故障', '原因', '环境', '输入', '配置', '参数']
        effect_keywords = ['影响', '结果', '输出', '性能', '质量', '异常']
        
        degrees = dict(context.graph.degree())
        degree_values = list(degrees.values())
        if len(degree_values) >= 3:
            low_threshold = np.percentile(degree_values, 15)
            high_threshold = np.percentile(degree_values, 85)
        else:
            low_threshold = 0
            high_threshold = float('inf')
        
        betweenness_centrality = {}
        betweenness_threshold = 0.0
        try:
            betweenness_centrality = nx.betweenness_centrality(context.graph)
            all_betweenness = list(betweenness_centrality.values())
            if len(all_betweenness) > 5:
                betweenness_threshold = np.percentile(all_betweenness, 75)
            else:
                betweenness_threshold = float('inf')
        except Exception as e:
            logger.warning(f"介数中心性计算失败: {e}")
            betweenness_centrality = {}
        
        high_corr_nodes = set()
        if context.correlation_matrix is not None and not context.correlation_matrix.empty:
            corr_threshold = 0.75
            for col in context.correlation_matrix.columns:
                max_corr = context.correlation_matrix[col].abs().nlargest(2).iloc[-1] if len(context.correlation_matrix) > 1 else 0
                if max_corr >= corr_threshold:
                    for node_id, node_text in id_to_text.items():
                        if col.lower() in node_text.lower() or node_text.lower() in col.lower():
                            high_corr_nodes.add(node_id)
                            break
        
        for node_id in context.graph.nodes():
            node_text = id_to_text.get(node_id, node_id)
            conditions_met = []
            
            if any(keyword in node_text for keyword in cause_keywords + effect_keywords):
                conditions_met.append("semantic_important")
            
            node_degree = degrees.get(node_id, 0)
            if node_degree <= low_threshold or node_degree >= high_threshold:
                conditions_met.append("degree_outlier")
            
            node_betweenness = betweenness_centrality.get(node_id, 0.0)
            if node_betweenness >= betweenness_threshold:
                conditions_met.append("high_centrality")
            
            if node_id in high_corr_nodes:
                conditions_met.append("high_correlation")
            
            if len(conditions_met) >= 1:
                elite_nodes[node_id] = f"{node_text}({','.join(conditions_met)})"
        
        logger.info(f"WeakChainRule: 精锐标准筛选出 {len(elite_nodes)} 个节点")
        return elite_nodes



    def _find_matching_column_enhanced(self, node_id: str, id_to_text: Dict, columns: List[str]) -> Optional[str]:
        """增强的列匹配逻辑 - 保持向后兼容"""
        node_text = id_to_text.get(node_id, "")
        if not node_text:
            return None

        if node_text in columns:
            return node_text

        node_text_lower = node_text.lower()
        for col in columns:
            if col.startswith('sensor_') and '_' in col:
                parts = col.split('_', 2)
                if len(parts) >= 3:
                    clean_col_name = parts[2].lower()
                    if clean_col_name == node_text_lower:
                        return col

        for col in columns:
            col_lower = col.lower()
            if (node_text_lower in col_lower or col_lower in node_text_lower):
                return col

        return None


class DataContextBuilder:
    """数据上下文构建器"""
    
    @staticmethod
    def build_context(ckg: Dict[str, Any], sensor_data: pd.DataFrame = None,
                     expert_docs: List[Dict] = None) -> DataContext:
        """构建数据上下文"""
        context = DataContext(ckg=ckg, sensor_data=sensor_data, expert_docs=expert_docs)
        
        context.node_mappings = DataContextBuilder._build_node_mappings(ckg, sensor_data)
        
        context.graph = DataContextBuilder._build_networkx_graph(ckg, context.node_mappings)
        
        if sensor_data is not None and not sensor_data.empty:
            context.correlation_matrix = DataContextBuilder._build_correlation_matrix(sensor_data)
        
        return context
    
    @staticmethod
    def _build_correlation_matrix(sensor_data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """构建相关性矩阵 - V2.0 健壮版"""
        try:
            data_for_corr = sensor_data.copy()

            non_numeric_cols = [col for col in data_for_corr.columns if 'time' in col.lower()]
            data_for_corr = data_for_corr.drop(columns=non_numeric_cols)

            for col in data_for_corr.columns:
                data_for_corr[col] = pd.to_numeric(data_for_corr[col], errors='coerce')

            low_variance_cols = [col for col in data_for_corr.columns if data_for_corr[col].var() < 1e-10]
            if low_variance_cols:
                data_for_corr = data_for_corr.drop(columns=low_variance_cols)
            
            if len(data_for_corr.columns) >= 2:
                corr_matrix = data_for_corr.corr()
                target_col_name = 'hidden_confounder_车间总电源电压'
                if target_col_name in corr_matrix.columns:
                    logger.info(f"DIAGNOSE: 关键列 '{target_col_name}' 已成功包含在相关性矩阵中。")
                else:
                    logger.warning(f"DIAGNOSE: 关键列 '{target_col_name}' 仍然未能进入相关性矩阵，请检查CSV文件中的列名和数据。")
                return corr_matrix
        
        except Exception as e:
            logger.warning(f"相关性矩阵构建失败: {e}")
        
        return None


    @staticmethod
    def _build_node_mappings(ckg: Dict[str, Any], sensor_data: Optional[pd.DataFrame] = None) -> Dict[str, Dict[str, str]]:
        """
        构建节点ID与文本的双向映射，并增加精确的节点到数据列映射。
        这是合并WeakChainRule的关键依赖。
        """
        id_to_text = {}
        text_to_id = {}
        id_to_column = {}
        
        available_columns = sensor_data.columns.tolist() if sensor_data is not None else []
        
        for node_type, nodes in ckg.get('nodes_by_type', {}).items():
            for node in nodes:
                node_id = node.get('id')
                node_text = node.get('text')
                if not (node_id and node_text):
                    continue
                    
                id_to_text[node_id] = node_text
                text_to_id[node_text] = node_id
                
                column_name = node.get('data_column_name')
                if column_name and column_name in available_columns:
                    id_to_column[node_id] = column_name
                    continue
                
                if available_columns:
                    matched_column = DataContextBuilder._find_precise_column_match(node_id, node_text, available_columns)
                    if matched_column:
                        id_to_column[node_id] = matched_column

        special_columns_info = ckg.get('special_columns_info', {})
        
        return {
            'id_to_text': id_to_text,
            'text_to_id': text_to_id,
            'id_to_column': id_to_column,
            'special_columns_info': special_columns_info
        }

    @staticmethod
    def _find_precise_column_match(node_id: str, node_text: str, available_columns: List[str]) -> Optional[str]:
        """
        精确匹配节点到数据列的辅助函数。
        """
        node_text_lower = node_text.lower()
        
        if node_text in available_columns:
            return node_text
            
        for col in available_columns:
            if col.startswith('sensor_') and '_' in col:
                parts = col.split('_', 2)
                if len(parts) >= 3:
                    clean_col_name = parts[2].lower()
                    if clean_col_name == node_text_lower:
                        return col
        
        for col in available_columns:
            col_lower = col.lower()
            if node_text_lower in col_lower or col_lower in node_text_lower:
                return col
                
        return None 

    @staticmethod
    def _build_networkx_graph(ckg: Dict[str, Any], node_mappings: Dict[str, Dict[str, str]]) -> nx.DiGraph:
        """构建NetworkX有向图"""
        graph = nx.DiGraph()
        
        for node_id, node_text in node_mappings['id_to_text'].items():
            graph.add_node(node_id, text=node_text)
        
        edges = ckg.get('edges', [])
        if isinstance(edges, dict):
            for edge_key, edge_list in edges.items():
                for edge in edge_list:
                    source_id = edge['source']
                    target_id = edge['target']
                    confidence = edge.get('confidence', 0.5)

                    if source_id in graph.nodes and target_id in graph.nodes:
                        graph.add_edge(source_id, target_id, confidence=confidence, edge_key=edge_key)
        else:
            for edge in edges:
                source_id = edge['source']
                target_id = edge['target']
                confidence = edge.get('confidence', 0.5)

                if source_id in graph.nodes and target_id in graph.nodes:
                    graph.add_edge(source_id, target_id, confidence=confidence)
        
        return graph
    



class HypothesisQualityScorer:
    """假设质量评分器 - 用于筛选高质量假设"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
    def score(self, hypothesis: Hypothesis) -> float:
        """计算假设的质量分数"""
        specificity_score = 1.0 / (1.0 + len(hypothesis.target_elements))
        
        evidence_score = 0.5  # 基础分
        
        if hypothesis.rule_category == 'data_driven':
            evidence_score = 0.8
            
            if hypothesis.evidence:
                if 'correlation' in hypothesis.evidence and hypothesis.evidence['correlation'] > 0.7:
                    evidence_score = 0.95
                if 'z_score' in hypothesis.evidence and hypothesis.evidence['z_score'] > 2.0:
                    evidence_score = 0.9
                    
        rule_weights = {
            'HypothesisFusionEngine': 1.5,          # 赋予最高权重，确保融合结果被保留
            'CausalDesertRule': 1.2,                  # 提升：1.0 -> 1.2，核心数据-知识冲突
            'HighCorrelationLongDistanceRule': 0.9,   # 保持不变，强数据信号
            'ConditionalInstabilityRule': 0.9,        # 保持不变，强数据信号
            'PartialCorrelationDropRule': 0.85,       # 保持不变，较强数据信号
            'EdgeStrengthOutlierRule': 0.7,           # 保持不变，结构+属性，中等价值
            'CentralityDiscrepancyRule': 0.6,         # 提升：0.5 -> 0.6，避免过度剪枝
            'DegreeOutlierRule': 0.5,                 # 提升：0.4 -> 0.5，保留弱信号
            'WeakChainRule': 0.65                     # 保持不变，结合了结构和属性
        }
        rule_weight = rule_weights.get(hypothesis.rule_name, 0.5)
        
        confidence_weight = hypothesis.confidence_score
        
        priority_weight = hypothesis.priority
        
        quality_score = (
            specificity_score * 0.25 +     # 特异性占25%
            evidence_score * 0.25 +         # 证据丰富度占25%
            rule_weight * 0.3 +             # 规则类型占30%
            confidence_weight * 0.15 +      # 置信度占15%
            priority_weight * 0.05          # 优先级占5%
        )
        
        return quality_score



class HypothesisFusionEngine:
    """假设融合引擎 - 将多个弱信号融合成强信号"""
    
    def __init__(self):
        logger.info("假设融合引擎初始化")
    
    def fuse_hypotheses(self, hypotheses: List[Hypothesis], context: DataContext) -> List[Hypothesis]:
        """执行假设融合，返回融合后的新假设"""
        fusion_hypotheses = []
        
        causal_desert_fusions = self.fuse_causal_deserts(hypotheses, context)
        fusion_hypotheses.extend(causal_desert_fusions)
        
        chain_break_fusions = self.fuse_chain_breaks(hypotheses, context)
        fusion_hypotheses.extend(chain_break_fusions)
        
        logger.info(f"假设融合完成：生成 {len(fusion_hypotheses)} 个融合假设")
        return fusion_hypotheses
    
    def fuse_causal_deserts(self, hypotheses: List[Hypothesis], context: DataContext) -> List[Hypothesis]:
        """融合因果荒漠假设"""
        fusion_hypotheses = []
        
        causal_desert_hyps = [h for h in hypotheses if h.hypothesis_type == 'causal_desert']
        correlation_distance_hyps = [h for h in hypotheses if h.hypothesis_type == 'correlation_distance_mismatch']
        
        for desert_hyp in causal_desert_hyps:
            desert_nodes = set(desert_hyp.target_elements)
            
            supporting_hyps = []
            for corr_hyp in correlation_distance_hyps:
                corr_nodes = set(corr_hyp.target_elements)
                if desert_nodes.intersection(corr_nodes):
                    supporting_hyps.append(corr_hyp)
            
            if len(supporting_hyps) >= 2:
                all_confidences = [desert_hyp.confidence_score] + [h.confidence_score for h in supporting_hyps]
                fusion_confidence = min(0.98, np.mean(all_confidences) * 1.3)  # 融合奖励
                
                fusion_evidence = {
                    'primary_hypothesis': desert_hyp.id,
                    'supporting_hypotheses': [h.id for h in supporting_hyps],
                    'fusion_reason': 'multiple_correlation_distance_support',
                    'target_node_overlap': list(desert_nodes),
                    'confidence_boost': fusion_confidence - desert_hyp.confidence_score,
                    'fused_evidences': [h.evidence for h in supporting_hyps]
                }
                
                fusion_hypothesis = Hypothesis(
                    id=f"FUSION_DESERT_{desert_hyp.target_elements[0]}_{get_deterministic_short_hash(str([h.id for h in supporting_hyps]))}",
                    rule_name="HypothesisFusionEngine",
                    rule_category="fusion",
                    hypothesis_type="fusion_causal_desert",
                    description=f"融合发现：节点'{desert_hyp.evidence.get('node_text', desert_hyp.target_elements[0])}'存在显著因果荒漠现象（{len(supporting_hyps)+1}个假设收敛）",
                    target_elements=list(desert_nodes),
                    evidence=fusion_evidence,
                    confidence_score=fusion_confidence,
                    priority=fusion_confidence * 0.95,
                    metadata={
                        'fusion_type': 'causal_desert',
                        'source_hypotheses_count': len(supporting_hyps) + 1,
                        'fusion_timestamp': pd.Timestamp.now().isoformat()
                    }
                )
                fusion_hypotheses.append(fusion_hypothesis)
        
        return fusion_hypotheses

    def fuse_chain_breaks(self, hypotheses: List[Hypothesis], context: DataContext) -> List[Hypothesis]:
        """融合因果链断裂假设"""
        fusion_hypotheses = []
        
        weak_chain_hyps = [h for h in hypotheses if h.hypothesis_type == 'weak_causal_chain']
        edge_strength_hyps = [h for h in hypotheses if h.hypothesis_type == 'edge_strength_outlier']
        
        for chain_hyp in weak_chain_hyps:
            chain_elements = chain_hyp.target_elements  # 这些是边的表示，如 "A->B"
            
            supporting_hyps = []
            for strength_hyp in edge_strength_hyps:
                strength_elements = strength_hyp.target_elements
                if any(edge in strength_elements for edge in chain_elements):
                    supporting_hyps.append(strength_hyp)
            
            if len(supporting_hyps) >= 1:
                all_confidences = [chain_hyp.confidence_score] + [h.confidence_score for h in supporting_hyps]
                fusion_confidence = min(0.98, np.mean(all_confidences) * 1.25)  # 融合奖励
                
                fusion_evidence = {
                    'primary_hypothesis': chain_hyp.id,
                    'supporting_hypotheses': [h.id for h in supporting_hyps],
                    'fusion_reason': 'weak_chain_with_edge_anomalies',
                    'affected_edges': chain_elements,
                    'confidence_boost': fusion_confidence - chain_hyp.confidence_score
                }
                
                fusion_hypothesis = Hypothesis(
                    id=f"FUSION_CHAIN_{get_deterministic_short_hash(str(chain_elements))}",
                    rule_name="HypothesisFusionEngine", 
                    rule_category="fusion",
                    hypothesis_type="fusion_chain_break",
                    description=f"融合发现：因果链路径存在显著断裂风险（{len(supporting_hyps)+1}个假设收敛）",
                    target_elements=chain_elements,
                    evidence=fusion_evidence,
                    confidence_score=fusion_confidence,
                    priority=fusion_confidence * 0.9,
                    metadata={
                        'fusion_type': 'chain_break',
                        'source_hypotheses_count': len(supporting_hyps) + 1,
                        'fusion_timestamp': pd.Timestamp.now().isoformat()
                    }
                )
                fusion_hypotheses.append(fusion_hypothesis)
        
        return fusion_hypotheses


 


class HypothesisGenerator:
    """假设生成器核心类"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        self.rules: List[Union[StructuralRule, DataDrivenRule]] = []
        self._initialize_rules()
        
        logger.info(f"假设生成器已初始化")
        logger.info(f"   - 结构规则: {len([r for r in self.rules if isinstance(r, StructuralRule)])} 个")
        logger.info(f"   - 数据驱动规则: {len([r for r in self.rules if isinstance(r, DataDrivenRule)])} 个")
    

    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置 - 增加质量控制参数"""
        return {
            'target_hypotheses': {      # <-- 修改点
                'small': 40,            # 小规模案例目标35个
                'medium': 60,           # 中等规模案例目标50个
                'big': 65               # 大规模案例目标65个
            },
            'quality_threshold': 0.8,   # 质量分数阈值
            
            'degree_outlier': {
                'low_percentile': 20,
                'high_percentile': 80,
                'in_out_ratio_threshold': 2.0
            },
            'edge_strength_outlier': {
                'z_threshold': 2.0
            },
            'centrality_discrepancy': {
                'rank_difference_threshold': 5
            },
            'correlation_distance': {
                'correlation_threshold': 0.3,
                'distance_threshold': 2
            },
            'conditional_instability': {
                'correlation_variance_threshold': 0.07,  # 将由动态计算替代
                'min_samples_per_group': 5               # 将由动态计算替代
            },
            'partial_correlation_drop': {
                'correlation_drop_threshold': 0.3,
                'strong_edge_threshold': 0.7,
                'min_samples': 20
            },
            'causal_desert': {
                'correlation_threshold': 0.45,  # 将由动态计算替代
                'max_connections': 4
            },
            'weak_chain': {
                'confidence_threshold': 0.6,
                'variance_threshold': 0.3,
                'max_path_length': 4,
                'max_output_hypotheses': 30
            }
        }

    def _initialize_rules(self):
        """初始化规则库 - 遵循开放/封闭原则，添加新规则无需修改此类"""
        self.rules.append(DegreeOutlierRule(self.config.get('degree_outlier', {})))
        self.rules.append(EdgeStrengthOutlierRule(self.config.get('edge_strength_outlier', {})))
        self.rules.append(CentralityDiscrepancyRule(self.config.get('centrality_discrepancy', {})))

        self.rules.append(HighCorrelationLongDistanceRule(self.config.get('correlation_distance', {})))
        self.rules.append(ConditionalInstabilityRule(self.config.get('conditional_instability', {})))
        self.rules.append(PartialCorrelationDropRule(self.config.get('partial_correlation_drop', {})))

        self.rules.append(CausalDesertRule(self.config.get('causal_desert', {})))
        self.rules.append(WeakChainRule(self.config.get('weak_chain', {})))
    

    def _calculate_adaptive_quotas(self, hypotheses_by_rule: Dict[str, List[Hypothesis]], total_hypotheses_generated: int) -> Dict[str, int]:
        """
        根据各规则的产出和预设权重，动态计算配额。
        """
        rule_weights = {
            'HypothesisFusionEngine': 1.5,
            'CausalDesertRule': 1.2,
            'HighCorrelationLongDistanceRule': 1.0,
            'PartialCorrelationDropRule': 1.0,
            'WeakChainRule': 0.9,
            'ConditionalInstabilityRule': 0.8,
            'EdgeStrengthOutlierRule': 0.7,
            'CentralityDiscrepancyRule': 0.6,
            'DegreeOutlierRule': 0.5,
        }

        base_quotas = {
            'HypothesisFusionEngine': 1,
            'CausalDesertRule': 1,
            'HighCorrelationLongDistanceRule': 1,
            'PartialCorrelationDropRule': 1,
            'WeakChainRule': 4
        }
        
        total_quota_pool = self.config.get('target_hypotheses', {}).get('medium', 50) * 0.5
        total_base_quota = sum(base_quotas.values())
        floating_quota_pool = total_quota_pool - total_base_quota

        weighted_outputs = {}
        total_weighted_output = 0
        for rule_name, hyps in hypotheses_by_rule.items():
            weight = rule_weights.get(rule_name, 0.5)
            score = len(hyps) * weight
            weighted_outputs[rule_name] = score
            total_weighted_output += score

        adaptive_quotas = base_quotas.copy()
        if total_weighted_output > 0:
            for rule_name, score in weighted_outputs.items():
                proportion = score / total_weighted_output
                bonus_quota = round(proportion * floating_quota_pool)
                adaptive_quotas[rule_name] = adaptive_quotas.get(rule_name, 0) + int(bonus_quota)
                
        logger.info("动态配额计算完成:")
        for rule_name, quota in adaptive_quotas.items():
            logger.info(f"  - {rule_name}: {quota} 个配额 (产出: {len(hypotheses_by_rule.get(rule_name, []))})")
            
        return adaptive_quotas

    def generate_hypotheses(self, ckg_or_context, sensor_data: pd.DataFrame = None,
                        expert_docs: List[Dict] = None, mode: str = 'full', excluded_rules: List[str] = None, target_hypotheses: Dict[str, int] = None) -> List[Hypothesis]:
        """生成假设主方法 - 增加质量筛选和模式选择

        Args:
            ckg_or_context: CKG数据或HypothesisContext对象
            sensor_data: 传感器数据DataFrame
            expert_docs: 专家文档列表
            mode: 生成模式，可选值为 'full', 'structural_only', 'data_driven_only'
            target_hypotheses: 目标假设数量，格式为 {'small': 40, 'medium': 60, 'large': 80}

        Returns:
            生成的假设列表
        """
        logger.info(f"开始生成假设... (模式: {mode})")

        if isinstance(ckg_or_context, HypothesisContext):
            hypothesis_context = ckg_or_context
            context = DataContextBuilder.build_context(ckg_or_context.ckg,
                                                    ckg_or_context.data, expert_docs)
            context.graph = hypothesis_context.graph
            context.correlation_matrix = hypothesis_context.correlation_matrix
        else:
            ckg = ckg_or_context
            context = DataContextBuilder.build_context(ckg, sensor_data, expert_docs)
        
        all_hypotheses = []
        rule_stats = {}
        
        for rule in self.rules:
            if mode == 'structural_only' and not isinstance(rule, StructuralRule):
                continue
            if mode == 'data_driven_only' and not isinstance(rule, DataDrivenRule):
                continue
            if excluded_rules and rule.get_rule_info()['name'] in excluded_rules:
                continue
            
            rule_info = rule.get_rule_info()
            rule_name = rule_info['name']
            
            start_time = time.time()
            
            if rule.can_apply(context):
                try:
                    rule_hypotheses = rule.generate_hypotheses(context)
                    all_hypotheses.extend(rule_hypotheses)
                    
                    elapsed = time.time() - start_time
                    rule_stats[rule_name] = {
                        'hypotheses_generated': len(rule_hypotheses),
                        'execution_time': elapsed,
                        'status': 'success'
                    }
                    logger.info(f"   ✓ {rule_name}: {len(rule_hypotheses)} 个假设 ({elapsed:.2f}s)")
                
                except Exception as e:
                    elapsed = time.time() - start_time
                    rule_stats[rule_name] = {
                        'hypotheses_generated': 0,
                        'execution_time': elapsed,
                        'status': 'error',
                        'error': str(e)
                    }
                    logger.error(f"   ❌ {rule_name}: 执行失败 - {e}")
            else:
                rule_stats[rule_name] = {
                    'hypotheses_generated': 0,
                    'execution_time': 0,
                    'status': 'not_applicable',
                    'reason': 'insufficient_data'
                }
                logger.info(f"   ⚠️ {rule_name}: 数据不足，跳过")

        if mode == 'full':
            logger.info(f"开始假设融合，原始假设数: {len(all_hypotheses)}")
            fusion_engine = HypothesisFusionEngine()
            fusion_hypotheses = fusion_engine.fuse_hypotheses(all_hypotheses, context)
            all_hypotheses.extend(fusion_hypotheses)
            logger.info(f"融合后假设数: {len(all_hypotheses)}")
        else:
            logger.info(f"模式 '{mode}' 跳过假设融合阶段")

        logger.info(f"开始精英化配额剪枝，当前假设数: {len(all_hypotheses)}")

        if target_hypotheses:
            target_count = target_hypotheses.get('small', 40)  # 默认使用small值
        else:
            target_count = 40  # 精简目标：40个高质量假设

        hypotheses_by_rule = defaultdict(list)
        for h in all_hypotheses:
            hypotheses_by_rule[h.rule_name].append(h)

        rule_quotas = self._calculate_adaptive_quotas(hypotheses_by_rule, len(all_hypotheses))

        total_quota = sum(rule_quotas.values())
        remaining_slots = max(0, target_count - total_quota) # 使用动态的target_count

        scorer = HypothesisQualityScorer()
        all_scored_hypotheses = [(h, scorer.score(h)) for h in all_hypotheses]

        quota_selected = []
        remaining_pool = []

        for rule_name, quota in rule_quotas.items():
            if quota > 0 and rule_name in hypotheses_by_rule:
                rule_hypotheses = hypotheses_by_rule[rule_name]
                rule_scored = [(h, scorer.score(h)) for h in rule_hypotheses]
                rule_scored.sort(key=lambda x: x[1], reverse=True)
                
                selected_count = min(quota, len(rule_scored))
                for i in range(selected_count):
                    quota_selected.append(rule_scored[i][0])
                
                for i in range(selected_count, len(rule_scored)):
                    remaining_pool.append(rule_scored[i])
                
                logger.info(f"配额录取 - {rule_name}: {selected_count}/{quota} (候选: {len(rule_hypotheses)})")

        for rule_name, rule_hypotheses in hypotheses_by_rule.items():
            if rule_quotas.get(rule_name, 0) == 0:
                rule_scored = [(h, scorer.score(h)) for h in rule_hypotheses]
                remaining_pool.extend(rule_scored)

        remaining_pool.sort(key=lambda x: x[1], reverse=True)
        merit_selected = [h for h, score in remaining_pool[:remaining_slots]]

        final_hypotheses = quota_selected + merit_selected

        logger.info(f"精英化剪枝完成:")
        logger.info(f"  - 配额录取: {len(quota_selected)} 个")
        logger.info(f"  - 择优录取: {len(merit_selected)} 个")
        logger.info(f"  - 最终总数: {len(final_hypotheses)} 个 (目标: {target_count})")

        final_rule_retention = defaultdict(lambda: {'original': 0, 'retained': 0})
        for h in all_hypotheses:
            final_rule_retention[h.rule_name]['original'] += 1
        for h in final_hypotheses:
            final_rule_retention[h.rule_name]['retained'] += 1

        logger.info(f"各规则最终保留情况:")
        for rule_name, counts in final_rule_retention.items():
            retention_rate = counts['retained'] / max(1, counts['original']) * 100
            logger.info(f"  {rule_name}: {counts['retained']}/{counts['original']} (保留率: {retention_rate:.1f}%)")

        return final_hypotheses



@dataclass
class GroundTruthBlindSpot:
    """Ground Truth盲区数据结构"""
    id: int
    type: str
    description: str
    evidence_edges: List[str]
    min_evidence_for_detection: int

@dataclass
class CoverageResult:
    """覆盖度验证结果"""
    evidence_coverage_rate: float
    blindspot_coverage_rate: float
    covered_evidence_count: int
    total_evidence_count: int
    covered_blindspot_count: int
    total_blindspot_count: int
    detailed_results: Dict[str, Any]

class HypothesisCoverageValidator:
    """假设覆盖度验证器"""

    def __init__(self):
        self.ground_truth_blindspots: List[GroundTruthBlindSpot] = []
        self.evidence_edges: List[str] = []
        logger.info("🔍 假设覆盖度验证器已初始化")

    def load_ground_truth(self, ground_truth_file: str, ckg: Dict[str, Any] = None) -> bool:
        """加载Ground Truth数据 - 增加标准化步骤"""
        try:
            ground_truth_path = Path(ground_truth_file)
            if not ground_truth_path.exists():
                logger.error(f"Ground Truth文件不存在: {ground_truth_file}")
                return False

            with open(ground_truth_path, 'r', encoding='utf-8') as f:
                gt_data = json.load(f)

            id_to_text = {}
            text_to_id = {}
            if ckg:
                for node_type, nodes in ckg.get('nodes_by_type', {}).items():
                    for node in nodes:
                        node_id = node.get('id')
                        node_text = node.get('text')
                        if node_id and node_text:
                            id_to_text[node_id] = node_text
                            text_to_id[node_text] = node_id

            self.ground_truth_blindspots = []
            all_evidence_edges = set()
            excluded_count = 0

            for i, blindspot in enumerate(gt_data.get('ground_truth_targets', [])):
                evidence_edges = []
                blindspot_type = blindspot.get('blind_spot_type', '')

                if blindspot_type == 'tacit_knowledge_gaps':
                    excluded_count += 1
                    logger.info(f"   🚫 排除隐性知识盲区 #{i}: {blindspot.get('description', '')}")
                    continue

                if blindspot_type == 'confounded_relations':
                    expected_findings = blindspot.get('expected_findings', {})
                    spurious_relation = expected_findings.get('spurious_relation', {})
                    if spurious_relation:
                        source_node = spurious_relation.get('source_node', '')
                        target_node = spurious_relation.get('target_node', '')
                        if source_node and target_node:
                            evidence_edges.append(f"{source_node} -> {target_node}")

                    hidden_confounder = expected_findings.get('hidden_confounder', '')
                    if hidden_confounder and spurious_relation:
                        evidence_edges.extend([
                            f"{hidden_confounder} -> {spurious_relation.get('source_node', '')}",
                            f"{hidden_confounder} -> {spurious_relation.get('target_node', '')}"
                        ])

                elif blindspot_type == 'causal_chain_break':
                    expected_findings = blindspot.get('expected_findings', {})
                    chain_nodes = expected_findings.get('chain_nodes', [])
                    for j in range(len(chain_nodes) - 1):
                        evidence_edges.append(f"{chain_nodes[j]} -> {chain_nodes[j+1]}")

                elif blindspot_type == 'causal_desert':
                    expected_findings = blindspot.get('expected_findings', {})
                    isolated_node = expected_findings.get('isolated_node', '')
                    missing_edges = expected_findings.get('missing_edges', [])

                    for missing_edge in missing_edges:
                        source_id = missing_edge.get('source_id', '')
                        target_id = missing_edge.get('target_id', '')
                        if source_id and target_id:
                            evidence_edges.append(f"{source_id} -> {target_id}")

                    if isolated_node:
                        data_correlations = expected_findings.get('data_correlations', {})
                        for corr_node, corr_value in data_correlations.items():
                            if corr_value > 0.7:  # 高相关性阈值
                                evidence_edges.append(f"{isolated_node} -> {corr_node}")

                if evidence_edges:
                    gt_blindspot = GroundTruthBlindSpot(
                        id=len(self.ground_truth_blindspots),  # 重新编号
                        type=blindspot_type,
                        description=blindspot.get('description', ''),
                        evidence_edges=evidence_edges,
                        min_evidence_for_detection=max(1, len(evidence_edges) // 2)  # 至少需要一半的证据
                    )

                    self.ground_truth_blindspots.append(gt_blindspot)
                    all_evidence_edges.update(evidence_edges)

            raw_evidence_edges = list(all_evidence_edges)
            normalized_evidence_edges = []
            
            for edge_str in raw_evidence_edges:
                try:
                    source_str, target_str = [s.strip() for s in edge_str.split('->')]

                    source_text = id_to_text.get(source_str, source_str)
                    target_text = id_to_text.get(target_str, target_str)

                    normalized_edge = f"{source_text} -> {target_text}"
                    normalized_evidence_edges.append(normalized_edge)
                except ValueError:
                    logger.warning(f"无法解析GT证据边: '{edge_str}'，将保持原样")
                    normalized_evidence_edges.append(edge_str)

            self.evidence_edges = normalized_evidence_edges

            for blindspot in self.ground_truth_blindspots:
                normalized_blindspot_edges = []
                for edge_str in blindspot.evidence_edges:
                    try:
                        source_str, target_str = [s.strip() for s in edge_str.split('->')]
                        source_text = id_to_text.get(source_str, source_str)
                        target_text = id_to_text.get(target_str, target_str)
                        normalized_blindspot_edges.append(f"{source_text} -> {target_text}")
                    except ValueError:
                        normalized_blindspot_edges.append(edge_str)
                blindspot.evidence_edges = normalized_blindspot_edges

            logger.info(f"✅ 成功加载Ground Truth (排除隐性知识盲区):")
            logger.info(f"   - 可检测盲区数量: {len(self.ground_truth_blindspots)}")
            logger.info(f"   - 排除隐性知识盲区: {excluded_count} 个")
            logger.info(f"   - 证据边数量: {len(self.evidence_edges)}")

            logger.info("  (标准化后) 证据边示例:")
            for edge in self.evidence_edges[:3]:
                logger.info(f"   * {edge}")
            if len(self.evidence_edges) > 3:
                logger.info(f"   * ... 还有 {len(self.evidence_edges) - 3} 个")

            for blindspot in self.ground_truth_blindspots:
                logger.info(f"   - 盲区 #{blindspot.id} ({blindspot.type}): {len(blindspot.evidence_edges)} 个证据边")
                for edge in blindspot.evidence_edges[:3]:  # 只显示前3个
                    logger.info(f"     * {edge}")
                if len(blindspot.evidence_edges) > 3:
                    logger.info(f"     * ... 还有 {len(blindspot.evidence_edges) - 3} 个")

            return True

        except Exception as e:
            import traceback
            logger.error(f"加载Ground Truth失败: {e}")
            logger.info("🔍 详细错误信息:")
            traceback.print_exc()
            return False

    def validate_coverage(self, hypotheses: List[Hypothesis], ckg: Dict[str, Any]) -> CoverageResult:
        """验证假设覆盖度"""
        logger.info("\n🔍 开始验证假设覆盖度...")

        if not self.ground_truth_blindspots:
            logger.error("未加载Ground Truth数据")
            return CoverageResult(0, 0, 0, 0, 0, 0, {})

        evidence_coverage = self._validate_evidence_coverage(hypotheses, ckg)

        blindspot_coverage = self._validate_blindspot_coverage(evidence_coverage)

        covered_evidence_count = len([e for e in evidence_coverage.values() if e['is_covered']])
        evidence_coverage_rate = covered_evidence_count / max(1, len(self.evidence_edges))

        covered_blindspot_count = len([b for b in blindspot_coverage.values() if b['is_covered']])
        blindspot_coverage_rate = covered_blindspot_count / max(1, len(self.ground_truth_blindspots))

        result = CoverageResult(
            evidence_coverage_rate=evidence_coverage_rate,
            blindspot_coverage_rate=blindspot_coverage_rate,
            covered_evidence_count=covered_evidence_count,
            total_evidence_count=len(self.evidence_edges),
            covered_blindspot_count=covered_blindspot_count,
            total_blindspot_count=len(self.ground_truth_blindspots),
            detailed_results={
                'evidence_coverage': evidence_coverage,
                'blindspot_coverage': blindspot_coverage
            }
        )

        self._print_coverage_report(result)
        return result

    def _validate_evidence_coverage(self, hypotheses: List[Hypothesis], ckg: Dict[str, Any]) -> Dict[str, Dict]:
        """验证证据覆盖率"""
        evidence_coverage = {}

        node_id_to_text = {}
        for node_type, nodes in ckg.get('nodes_by_type', {}).items():
            for node in nodes:
                node_id_to_text[node['id']] = node['text']

        for evidence_edge in self.evidence_edges:
            evidence_coverage[evidence_edge] = {
                'is_covered': False,
                'covering_hypotheses': []
            }

        for hypothesis in hypotheses:
            covered_edges = self._extract_covered_edges(hypothesis, node_id_to_text)

            for evidence_edge in self.evidence_edges:
                if evidence_edge in covered_edges:
                    evidence_coverage[evidence_edge]['is_covered'] = True
                    evidence_coverage[evidence_edge]['covering_hypotheses'].append({
                        'hypothesis_id': hypothesis.id,
                        'rule_name': hypothesis.rule_name,
                        'confidence': hypothesis.confidence_score
                    })

        return evidence_coverage

    def _extract_covered_edges(self, hypothesis: Hypothesis, node_id_to_text: Dict[str, str]) -> List[str]:
        """从假设中提取可能覆盖的边 - 强制标准化为文本格式"""
        covered_edges = set()  # 使用set避免重复

        for target_element in hypothesis.target_elements:
            if '->' in target_element:
                parts = target_element.split('->')
                if len(parts) == 2:
                    source_id = parts[0].strip()
                    target_id = parts[1].strip()
                    source_text = node_id_to_text.get(source_id, source_id)
                    target_text = node_id_to_text.get(target_id, target_id)
                    covered_edges.add(f"{source_text} -> {target_text}")
            else:
                node_id = target_element
                node_text = node_id_to_text.get(node_id, node_id)

                for evidence_edge in self.evidence_edges:  # self.evidence_edges 此处已经是标准化的
                    edge_parts = evidence_edge.split(' -> ')
                    if len(edge_parts) == 2:
                        source_text, target_text = edge_parts[0].strip(), edge_parts[1].strip()
                        if node_text == source_text or node_text == target_text:
                            covered_edges.add(evidence_edge)

        return list(covered_edges)

    def _validate_blindspot_coverage(self, evidence_coverage: Dict[str, Dict]) -> Dict[int, Dict]:
        """验证盲区覆盖率"""
        blindspot_coverage = {}

        for blindspot in self.ground_truth_blindspots:
            covered_evidence_count = 0
            covered_evidence_list = []

            for evidence_edge in blindspot.evidence_edges:
                if evidence_coverage.get(evidence_edge, {}).get('is_covered', False):
                    covered_evidence_count += 1
                    covered_evidence_list.append(evidence_edge)

            is_covered = covered_evidence_count >= blindspot.min_evidence_for_detection

            blindspot_coverage[blindspot.id] = {
                'blindspot_type': blindspot.type,
                'blindspot_description': blindspot.description,
                'is_covered': is_covered,
                'covered_evidence_count': covered_evidence_count,
                'total_evidence_count': len(blindspot.evidence_edges),
                'min_required': blindspot.min_evidence_for_detection,
                'covered_evidence_list': covered_evidence_list,
                'missing_evidence_list': [e for e in blindspot.evidence_edges if not evidence_coverage.get(e, {}).get('is_covered', False)]
            }

        return blindspot_coverage

    def _print_coverage_report(self, result: CoverageResult):
        """打印覆盖度报告"""
        logger.info(f"\n📊 假设覆盖度验证报告")
        logger.info(f"=" * 60)

        logger.info(f"📈 总体覆盖情况:")
        logger.info(f"   - 证据覆盖率 (ECR): {result.evidence_coverage_rate:.1%} ({result.covered_evidence_count}/{result.total_evidence_count})")
        logger.info(f"   - 盲区覆盖率 (BSCR): {result.blindspot_coverage_rate:.1%} ({result.covered_blindspot_count}/{result.total_blindspot_count})")

        logger.info(f"\n🎯 盲区详细覆盖情况:")
        blindspot_coverage = result.detailed_results['blindspot_coverage']

        for blindspot_id, coverage_info in blindspot_coverage.items():
            status_icon = "✅" if coverage_info['is_covered'] else "❌"
            logger.info(f"   {status_icon} 盲区 #{blindspot_id} ({coverage_info['blindspot_type']}):")
            logger.info(f"      命中证据: {coverage_info['covered_evidence_count']}/{coverage_info['total_evidence_count']} (要求: {coverage_info['min_required']})")

            if coverage_info['covered_evidence_list']:
                logger.info(f"      ✓ 已覆盖: {', '.join(coverage_info['covered_evidence_list'])}")

            if coverage_info['missing_evidence_list']:
                logger.info(f"      ✗ 未覆盖: {', '.join(coverage_info['missing_evidence_list'])}")

        logger.info(f"=" * 60)

    def save_coverage_report(self, result: CoverageResult, output_file: str):
        """保存覆盖度报告"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        report_data = {
            'validation_timestamp': pd.Timestamp.now().isoformat(),
            'method': 'hypothesis_coverage_validation',
            'overall_metrics': {
                'evidence_coverage_rate': float(result.evidence_coverage_rate),
                'blindspot_coverage_rate': float(result.blindspot_coverage_rate),
                'covered_evidence_count': result.covered_evidence_count,
                'total_evidence_count': result.total_evidence_count,
                'covered_blindspot_count': result.covered_blindspot_count,
                'total_blindspot_count': result.total_blindspot_count
            },
            'detailed_coverage': result.detailed_results
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2, default=str)

        logger.info(f"📄 覆盖度报告已保存: {output_path}")


class HypothesisGeneratorInterface:
    """假设生成器接口 - 为外部集成提供统一入口"""

    def __init__(self, config_file: str = None):
        """初始化接口"""
        self.config = self._load_config(config_file) if config_file else None
        self.generator = HypothesisGenerator(self.config)
        self.validator = HypothesisCoverageValidator()

        logger.info("🚀 假设生成器接口已初始化")

    def _load_config(self, config_file: str) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"配置文件加载失败: {e}, 使用默认配置")
            return None

    def generate_and_validate(self, case_dir: str, output_dir: str = None,
                             enable_dual_output: bool = False,
                             enable_coverage_validation: bool = False) -> Dict[str, Any]:
        """生成假设并验证覆盖度 - 支持双路径输出的主要入口方法"""
        case_path = Path(case_dir)
        case_id = case_path.name

        if output_dir is None:
            output_dir = "./results"

        primary_output_path = Path(output_dir)
        primary_output_path.mkdir(parents=True, exist_ok=True)

        official_output_path = None
        if enable_dual_output:
            official_base = Path("./seek_data_v3_deep_enhanced/results") / case_id / "BCSA_Analysis" / "1_Final_Results" / "hypothesis_generation"
            official_output_path = official_base
            official_output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"🎯 开始处理案例: {case_id}")

        try:
            case_data = self._load_case_data(case_path)

            hypotheses = self.generator.generate_hypotheses(
                case_data['ckg'],
                case_data.get('sensor_data'),
                case_data.get('expert_docs')
            )

            output_files = {}

            hypothesis_file_primary = primary_output_path / "generated_hypotheses.json"
            self._save_hypotheses(hypotheses, str(hypothesis_file_primary), case_id)
            output_files['hypotheses_primary'] = str(hypothesis_file_primary)

            if official_output_path:
                hypothesis_file_official = official_output_path / "generated_hypotheses.json"
                self._save_hypotheses(hypotheses, str(hypothesis_file_official), case_id)
                output_files['hypotheses_official'] = str(hypothesis_file_official)

            coverage_result = None
            ground_truth_file = case_path / "ground_truth.json"
            if enable_coverage_validation and ground_truth_file.exists():
                logger.info(f"🔍 发现Ground Truth文件，开始覆盖度验证...")

                if self.validator.load_ground_truth(str(ground_truth_file), case_data['ckg']):
                    coverage_result = self.validator.validate_coverage(hypotheses, case_data['ckg'])

                    coverage_file_primary = primary_output_path / "coverage_report.json"
                    self.validator.save_coverage_report(coverage_result, str(coverage_file_primary))
                    output_files['coverage_report_primary'] = str(coverage_file_primary)

                    if official_output_path:
                        coverage_file_official = official_output_path / "coverage_report.json"
                        self.validator.save_coverage_report(coverage_result, str(coverage_file_official))
                        output_files['coverage_report_official'] = str(coverage_file_official)
            elif enable_coverage_validation:
                logger.warning(f"未找到Ground Truth文件，跳过覆盖度验证")

            summary_file_primary = primary_output_path / "hypothesis_summary.json"
            self._save_summary_report({
                'case_id': case_id,
                'hypotheses': hypotheses,
                'coverage_result': coverage_result,
                'output_files': output_files
            }, str(summary_file_primary))
            output_files['summary_primary'] = str(summary_file_primary)

            if official_output_path:
                summary_file_official = official_output_path / "hypothesis_summary.json"
                self._save_summary_report({
                    'case_id': case_id,
                    'hypotheses': hypotheses,
                    'coverage_result': coverage_result,
                    'output_files': output_files
                }, str(summary_file_official))
                output_files['summary_official'] = str(summary_file_official)

            logger.info(f"✅ 案例 {case_id} 处理完成")
            logger.info(f"📊 生成假设: {len(hypotheses)} 个")
            if coverage_result:
                logger.info(f"📊 证据覆盖率: {coverage_result.evidence_coverage_rate:.1%}")
                logger.info(f"📊 盲区覆盖率: {coverage_result.blindspot_coverage_rate:.1%}")

            return {
                'case_id': case_id,
                'hypotheses_count': len(hypotheses),
                'coverage_result': coverage_result,
                'output_files': output_files,
                'status': 'success'
            }

        except Exception as e:
            logger.error(f"案例 {case_id} 处理失败: {e}")
            import traceback
            traceback.print_exc()
            return {
                'case_id': case_id,
                'status': 'error',
                'error': str(e)
            }

    def _load_case_data(self, case_path: Path) -> Dict[str, Any]:
        """加载案例数据"""
        case_data = {}

        ckg_file = case_path / "causal_knowledge_graph.json"
        if ckg_file.exists():
            with open(ckg_file, 'r', encoding='utf-8') as f:
                case_data['ckg'] = json.load(f)
        else:
            raise FileNotFoundError(f"因果知识图谱文件不存在: {ckg_file}")

        sensor_file = case_path / "sensor_data.csv"
        if sensor_file.exists():
            try:
                case_data['sensor_data'] = pd.read_csv(sensor_file)
                logger.info(f"加载传感器数据: {case_data['sensor_data'].shape}")
            except Exception as e:
                logger.warning(f"加载传感器数据失败: {e}")
                case_data['sensor_data'] = None
        else:
            case_data['sensor_data'] = None

        expert_docs_file = case_path / "expert_documents.json"
        if expert_docs_file.exists():
            try:
                with open(expert_docs_file, 'r', encoding='utf-8') as f:
                    case_data['expert_docs'] = json.load(f)
            except Exception as e:
                logger.warning(f"加载专家文档失败: {e}")
                case_data['expert_docs'] = None
        else:
            case_data['expert_docs'] = None

        return case_data

    def _save_hypotheses(self, hypotheses: List[Hypothesis], output_file: str, case_id: str):
        """保存假设列表"""
        hypotheses_data = []
        for hyp in hypotheses:
            hyp_data = {
                'id': hyp.id,
                'rule_name': hyp.rule_name,
                'rule_category': hyp.rule_category,
                'hypothesis_type': hyp.hypothesis_type,
                'description': hyp.description,
                'target_elements': hyp.target_elements,
                'evidence': hyp.evidence,
                'confidence_score': hyp.confidence_score,
                'priority': hyp.priority,
                'metadata': hyp.metadata
            }
            hypotheses_data.append(hyp_data)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(convert_numpy_types(hypotheses_data), f, indent=2, ensure_ascii=False)

        logger.info(f"📋 假设列表已保存: {output_file}")

    def _save_summary_report(self, results: Dict[str, Any], output_file: str):
        """保存综合报告"""
        case_id = results['case_id']
        hypotheses = results['hypotheses']
        coverage_result = results.get('coverage_result')

        if hypotheses:
            category_counts = Counter(h.rule_category for h in hypotheses)
            type_counts = Counter(h.hypothesis_type for h in hypotheses)

            summary = {
                'case_id': case_id,
                'generation_timestamp': pd.Timestamp.now().isoformat(),
                'total_hypotheses': len(hypotheses),
                'by_category': dict(category_counts),
                'by_type': dict(type_counts),
                'avg_confidence': float(np.mean([h.confidence_score for h in hypotheses])),
                'avg_priority': float(np.mean([h.priority for h in hypotheses])),
                'top_hypotheses': [
                    {
                        'id': h.id,
                        'rule_name': h.rule_name,
                        'description': h.description,
                        'confidence': h.confidence_score,
                        'priority': h.priority
                    }
                    for h in sorted(hypotheses, key=lambda x: x.confidence_score, reverse=True)[:5]
                ]
            }
        else:
            summary = {
                'case_id': case_id,
                'generation_timestamp': pd.Timestamp.now().isoformat(),
                'total_hypotheses': 0,
                'by_category': {},
                'by_type': {},
                'avg_confidence': 0.0,
                'avg_priority': 0.0,
                'top_hypotheses': []
            }

        if coverage_result:
            summary['coverage_validation'] = {
                'evidence_coverage_rate': coverage_result.evidence_coverage_rate,
                'blindspot_coverage_rate': coverage_result.blindspot_coverage_rate,
                'covered_evidence_count': coverage_result.covered_evidence_count,
                'total_evidence_count': coverage_result.total_evidence_count,
                'covered_blindspot_count': coverage_result.covered_blindspot_count,
                'total_blindspot_count': coverage_result.total_blindspot_count
            }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

        logger.info(f"📋 综合报告已保存: {output_file}")


def run_hypothesis_generation(case_dir: Path, output_dir: Path, enable_coverage_validation: bool = False) -> Dict[str, Any]:
    """
    运行假设生成的主函数 - 增强版本支持覆盖度验证
    Args:
        case_dir: 案例目录路径
        output_dir: 输出目录路径
        enable_coverage_validation: 是否启用覆盖度验证
    Returns:
        包含假设列表和覆盖度结果的字典
    """
    logger.info(f"开始假设生成流程: {case_dir.name}")

    try:
        interface = HypothesisGeneratorInterface()

        result = interface.generate_and_validate(
            case_dir=str(case_dir),
            output_dir=str(output_dir),
            enable_dual_output=False,
            enable_coverage_validation=enable_coverage_validation
        )

        logger.info(f"假设生成完成: {result['hypotheses_count']} 个假设")
        if result.get('coverage_result'):
            coverage = result['coverage_result']
            logger.info(f"覆盖度验证完成: 证据覆盖率 {coverage.evidence_coverage_rate:.1%}, 盲区覆盖率 {coverage.blindspot_coverage_rate:.1%}")

        return result

    except Exception as e:
        logger.error(f"假设生成失败: {e}")
        raise

def run_hypothesis_generation_legacy(case_dir: Path, output_dir: Path) -> List[Hypothesis]:
    """
    运行假设生成的传统函数 - 保持向后兼容
    Args:
        case_dir: 案例目录路径
        output_dir: 输出目录路径
    Returns:
        生成的假设列表
    """
    logger.info(f"开始假设生成流程: {case_dir.name}")

    case_dir = Path(case_dir)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        ckg_file = case_dir / "causal_knowledge_graph.json"
        if not ckg_file.exists():
            raise FileNotFoundError(f"因果知识图谱文件不存在: {ckg_file}")

        with open(ckg_file, 'r', encoding='utf-8') as f:
            ckg = json.load(f)

        sensor_data = None
        sensor_file = case_dir / "sensor_data.csv"
        if sensor_file.exists():
            try:
                sensor_data = pd.read_csv(sensor_file)
                logger.info(f"加载传感器数据: {sensor_data.shape}")
            except Exception as e:
                logger.warning(f"加载传感器数据失败: {e}")

        generator = HypothesisGenerator()
        hypotheses = generator.generate_hypotheses(ckg, sensor_data)

        _save_hypotheses_results(hypotheses, output_dir, case_dir.name)

        logger.info(f"假设生成完成: {len(hypotheses)} 个假设已保存到 {output_dir}")
        return hypotheses

    except Exception as e:
        logger.error(f"假设生成失败: {e}")
        raise

def _save_hypotheses_results(hypotheses: List[Hypothesis], output_dir: Path, case_id: str):
    """保存假设生成结果"""
    
    hypotheses_data = []
    for hyp in hypotheses:
        hyp_data = {
            'id': hyp.id,
            'rule_name': hyp.rule_name,
            'rule_category': hyp.rule_category,
            'hypothesis_type': hyp.hypothesis_type,
            'description': hyp.description,
            'target_elements': hyp.target_elements,
            'evidence': hyp.evidence,
            'confidence_score': hyp.confidence_score,
            'priority': hyp.priority,
            'metadata': hyp.metadata
        }
        hypotheses_data.append(hyp_data)
    
    hypotheses_file = output_dir / "generated_hypotheses.json"
    with open(hypotheses_file, 'w', encoding='utf-8') as f:
        json.dump(convert_numpy_types(hypotheses_data), f, indent=2, ensure_ascii=False)
    
    if hypotheses:
        category_counts = Counter(h.rule_category for h in hypotheses)
        type_counts = Counter(h.hypothesis_type for h in hypotheses)
        
        summary = {
            'case_id': case_id,
            'generation_timestamp': pd.Timestamp.now().isoformat(),
            'total_hypotheses': len(hypotheses),
            'by_category': dict(category_counts),
            'by_type': dict(type_counts),
            'avg_confidence': float(np.mean([h.confidence_score for h in hypotheses])),
            'avg_priority': float(np.mean([h.priority for h in hypotheses])),
            'top_hypotheses': [
                {
                    'id': h.id,
                    'description': h.description,
                    'confidence': h.confidence_score,
                    'priority': h.priority
                }
                for h in sorted(hypotheses, key=lambda x: x.confidence_score, reverse=True)[:5]
            ]
        }
    else:
        summary = {
            'case_id': case_id,
            'generation_timestamp': pd.Timestamp.now().isoformat(),
            'total_hypotheses': 0,
            'by_category': {},
            'by_type': {},
            'avg_confidence': 0.0,
            'avg_priority': 0.0,
            'top_hypotheses': []
        }
    
    summary_file = output_dir / "hypothesis_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    logger.info(f"假设结果已保存:")
    logger.info(f"  - 详细列表: {hypotheses_file}")
    logger.info(f"  - 摘要信息: {summary_file}")


def main():
    """主函数 - 独立运行假设生成模块（增强版）"""
    print("=" * 80)
    print("🎯 BCSA假设生成模块")
    print("=" * 80)

    case_dir = Path("./seek_data_v3_deep_enhanced/cases/smallcase/Mixed/Mixed_small_01")
    output_dir = Path("./results/hypothesis_generation_enhanced")

    if not case_dir.exists():
        logger.error(f"测试案例目录不存在: {case_dir}")
        return

    try:
        result = run_hypothesis_generation(case_dir, output_dir, enable_coverage_validation=False)

        print(f"\n✅ 假设生成成功完成!")
        print(f"📊 案例: {case_dir.name}")
        print(f"📊 生成假设数: {result['hypotheses_count']}")
        print(f"📊 处理状态: {result['status']}")

        if result.get('coverage_result'):
            coverage = result['coverage_result']
            print(f"\n🎯 覆盖度验证结果:")
            print(f"   - 证据覆盖率: {coverage.evidence_coverage_rate:.1%} ({coverage.covered_evidence_count}/{coverage.total_evidence_count})")
            print(f"   - 盲区覆盖率: {coverage.blindspot_coverage_rate:.1%} ({coverage.covered_blindspot_count}/{coverage.total_blindspot_count})")
        else:
            print(f"\n⚠️ 未进行覆盖度验证（可能缺少ground_truth.json文件）")

        print(f"\n💾 输出文件:")
        for file_type, file_path in result.get('output_files', {}).items():
            print(f"   - {file_type}: {file_path}")

        print(f"\n规则功能:")
        print(f"   ✓ EdgeStrengthOutlierRule - 边强度异常检测")
        print(f"   ✓ CentralityDiscrepancyRule - 中心性差异检测")
        print(f"   ✓ ConditionalInstabilityRule - 条件性不稳定检测")
        print(f"   ✓ PartialCorrelationDropRule - 偏相关下降检测")
        print(f"   ✓ CausalDesertRule - 因果荒漠检测")
        print(f"   ✓ WeakChainRule - 弱因果链检测")
        print(f"   ✓ HypothesisCoverageValidator - 覆盖度验证系统")

        print(f"\n🚀 系统特性:")
        print(f"   ✓ 8种假设生成规则（2种结构规则 + 6种数据驱动规则）")
        print(f"   ✓ 双层覆盖度验证（证据覆盖率 + 盲区覆盖率）")
        print(f"   ✓ 完整的证据-假设匹配系统")
        print(f"   ✓ 详细的覆盖度报告生成")
        print(f"   ✓ 向后兼容的接口设计")

    except Exception as e:
        logger.error(f"假设生成失败: {e}")
        import traceback
        traceback.print_exc()

def demo_coverage_validation():
    """演示覆盖度验证功能"""
    print("\n" + "=" * 80)
    print("🔍 覆盖度验证功能演示")
    print("=" * 80)

    cases_dir = Path("./seek_data_v3_deep_enhanced/cases")
    demo_cases = []

    if cases_dir.exists():
        for case_path in cases_dir.rglob("*/"):
            if (case_path / "ground_truth.json").exists() and (case_path / "causal_knowledge_graph.json").exists():
                demo_cases.append(case_path)
                if len(demo_cases) >= 3:  # 只演示前3个案例
                    break

    if not demo_cases:
        print("⚠️ 未找到包含ground_truth.json的测试案例")
        return

    print(f"📋 找到 {len(demo_cases)} 个包含Ground Truth的案例，开始演示...")

    interface = HypothesisGeneratorInterface()

    for i, case_path in enumerate(demo_cases, 1):
        print(f"\n🎯 演示案例 {i}: {case_path.name}")
        print("-" * 60)

        try:
            result = interface.generate_and_validate(
                case_dir=str(case_path),
                output_dir=f"./results/coverage_demo/{case_path.name}",
                enable_dual_output=False,
                enable_coverage_validation=True
            )

            if result['status'] == 'success':
                print(f"   ✅ 处理成功")
                print(f"   📊 假设数量: {result['hypotheses_count']}")

                if result.get('coverage_result'):
                    coverage = result['coverage_result']
                    print(f"   🎯 证据覆盖率: {coverage.evidence_coverage_rate:.1%}")
                    print(f"   🎯 盲区覆盖率: {coverage.blindspot_coverage_rate:.1%}")
                else:
                    print(f"   ⚠️ 覆盖度验证失败")
            else:
                print(f"   ❌ 处理失败: {result.get('error', '未知错误')}")

        except Exception as e:
            print(f"   ❌ 演示失败: {e}")

    print(f"\n✅ 覆盖度验证功能演示完成!")
    print(f"💾 详细结果请查看: ./results/coverage_demo/")

if __name__ == "__main__":
    main()

