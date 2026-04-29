#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BCSA PC-VGAE审计模块基础版Base
实现负采样、完整不确定性地图和重点关注对象识别
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
import itertools
from SEEK_Config import SEEKConfig, SEEKDataInterface
from SEEK_GroundTruth_Loader import GroundTruthLoader, EvidenceCalculator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import negative_sampling, train_test_split_edges
from torch_geometric.data import Data
import random

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

class RealPCVGAE(nn.Module):
    """真实的PC-VGAE模型 - 重构版，支持统一特征输入"""

    def __init__(self, unified_feature_dim: int = None, in_channels: int = None, 
                hypothesis_feature_dim: int = 0, latent_dim: int = 16, hidden_dim: int = 64):
        super(RealPCVGAE, self).__init__()

        if in_channels is None and unified_feature_dim is None:
            raise ValueError("必须提供 'in_channels' 或 'unified_feature_dim' 参数。")

        actual_in_channels = in_channels if in_channels is not None else unified_feature_dim

        self.hypothesis_feature_dim = hypothesis_feature_dim
        self.latent_dim = latent_dim

        total_in_channels = actual_in_channels + hypothesis_feature_dim

        self.encoder_conv1 = GATv2Conv(in_channels, hidden_dim, heads=4, dropout=0.2, concat=True)
        self.encoder_conv2 = GATv2Conv(hidden_dim * 4, 32, heads=2, dropout=0.2, concat=True)

        self.conv_mu = GATv2Conv(32 * 2, latent_dim, heads=1, dropout=0.2, concat=False)
        self.conv_logstd = GATv2Conv(32 * 2, latent_dim, heads=1, dropout=0.2, concat=False)

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """编码器：输出潜在表示的均值和对数标准差"""
        h1 = F.elu(self.encoder_conv1(x, edge_index))
        h2 = F.elu(self.encoder_conv2(h1, edge_index))

        mu = self.conv_mu(h2, edge_index)
        logstd = self.conv_logstd(h2, edge_index)

        return mu, logstd

    def reparameterize(self, mu: torch.Tensor, logstd: torch.Tensor) -> torch.Tensor:
        """重参数化技巧"""
        if self.training:
            std = torch.exp(logstd)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def decode_edges(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """解码器：重构边的概率"""
        row, col = edge_index
        edge_embeddings = (z[row] * z[col]).sum(dim=1)
        return torch.sigmoid(edge_embeddings)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """前向传播 - 简化版，不再需要prompt_vector"""
        mu, logstd = self.encode(x, edge_index)
        z = self.reparameterize(mu, logstd)
        edge_recon = self.decode_edges(z, edge_index)
        return edge_recon, mu, logstd

    def kl_loss(self, mu: torch.Tensor, logstd: torch.Tensor) -> torch.Tensor:
        """KL散度损失"""
        return -0.5 * torch.mean(torch.sum(1 + 2 * logstd - mu.pow(2) - logstd.exp().pow(2), dim=1))


class MonteCarloDropoutQuantifier:
    """蒙特卡洛Dropout不确定性量化器"""

    def __init__(self, num_samples: int = 50):
        self.num_samples = num_samples

    def quantify_uncertainty(self, model: RealPCVGAE, x: torch.Tensor, edge_index: torch.Tensor,
                           prompt_vector: torch.Tensor, target_edges: torch.Tensor) -> Tuple[float, float]:
        """使用MC Dropout量化不确定性

        Returns:
            mean_prob: 平均重构概率
            uncertainty: 不确定性（标准差）
        """
        model.train()  # 保持训练模式以启用Dropout

        reconstruction_probs = []

        with torch.no_grad():
            for _ in range(self.num_samples):
                edge_recon, _, _ = model(x, edge_index)

                if target_edges.size(0) > 0:
                    target_probs = model.decode_edges(model.reparameterize(*model.encode(x, edge_index)), target_edges)
                    avg_prob = target_probs.mean().item()
                else:
                    avg_prob = edge_recon.mean().item()

                reconstruction_probs.append(avg_prob)

        mean_prob = np.mean(reconstruction_probs)
        uncertainty = np.std(reconstruction_probs)

        return mean_prob, uncertainty

class NegativeSampler:
    """负采样器 - 生成不存在的边用于评估"""
    
    def __init__(self, graph_data: Dict[str, Any], sampling_ratio: float = 2.0):
        self.graph_data = graph_data
        self.sampling_ratio = sampling_ratio
        self.existing_edges = self._extract_existing_edges()
        self.all_nodes = self._extract_all_nodes()
    
    def _extract_existing_edges(self) -> Set[Tuple[str, str]]:
        """提取现有边"""
        edges = set()
        edges_dict = self.graph_data.get('edges', {})

        for edge_key, edge_list in edges_dict.items():
            if isinstance(edge_list, list):
                for edge in edge_list:
                    if isinstance(edge, dict):
                        source = edge.get('source', '')
                        target = edge.get('target', '')
                        if source and target:
                            edges.add((source, target))
        return edges
    
    def _extract_all_nodes(self) -> List[str]:
        """提取所有节点"""
        nodes = []

        nodes_dict = self.graph_data.get('nodes_by_type', {})
        for node_type, node_list in nodes_dict.items():
            if isinstance(node_list, list):
                for node in node_list:
                    if isinstance(node, dict):
                        node_id = node.get('id', '')
                        if node_id:
                            nodes.append(node_id)
        return nodes
    
    def generate_negative_samples(self) -> List[Tuple[str, str]]:
        """生成负样本边"""
        num_existing = len(self.existing_edges)
        num_negative = int(num_existing * self.sampling_ratio)
        
        all_possible_edges = set()
        for source in self.all_nodes:
            for target in self.all_nodes:
                if source != target:  # 排除自环
                    all_possible_edges.add((source, target))
        
        negative_candidates = all_possible_edges - self.existing_edges
        
        negative_samples = list(negative_candidates)
        np.random.shuffle(negative_samples)
        
        return negative_samples[:num_negative]

class EnhancedFeatureExtractor:
    """增强特征提取器 - 实现结构特征、数据统计特征和数据相关性特征的融合"""

    def __init__(self):
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()

    def extract_unified_features(self, graph_data: Dict, sensor_data: pd.DataFrame, 
                                correlation_matrix: pd.DataFrame, 
                                node_id_to_column_mapping: Dict[str, str],
                                use_degraded_features: bool = False) -> Dict[str, np.ndarray]:
        """统一特征提取主函数 - 整合结构、数据统计和相关性特征
        
        Args:
            graph_data: 图数据
            sensor_data: 传感器数据
            correlation_matrix: 相关性矩阵
            node_id_to_column_mapping: 节点ID到数据列名的映射
            
        Returns:
            {node_id: unified_feature_vector} 字典
        """
        logger.info("开始提取统一特征...")
        
        structural_features = self.extract_structural_features(graph_data)
        
        if use_degraded_features:
            logger.warning("特征降级模式已激活：仅使用结构特征。")
            data_stat_features = {}
            correlation_features = {}
        else:
            data_stat_features = self._extract_data_statistics_features(
                graph_data, sensor_data, node_id_to_column_mapping
            )
            correlation_features = self._extract_correlation_features(
                graph_data, correlation_matrix, node_id_to_column_mapping
            )
        
        all_node_ids = set()
        nodes_by_type = graph_data.get('nodes_by_type', {})
        for node_type, nodes in nodes_by_type.items():
            for node in nodes:
                node_id = node.get('id', '')
                if node_id:
                    all_node_ids.add(node_id)
        
        unified_features = {}
        feature_vectors = []
        
        node_ids_ordered = sorted(list(all_node_ids))  # 添加排序确保确定性
        logger.info(f"排序后的前5个节点ID: {node_ids_ordered[:5]}")
        
        for node_id in node_ids_ordered:
            struct_vec = self._dict_to_vector(structural_features.get(node_id, {}), 
                                            ['degree_centrality', 'betweenness_centrality', 
                                            'pagerank', 'clustering_coefficient', 
                                            'in_degree', 'out_degree', 'total_degree'])
            
            stat_vec = data_stat_features.get(node_id, np.zeros(4))  # [mean, std, skew, kurtosis]
            
            corr_vec = correlation_features.get(node_id, np.array([]))
            
            if len(corr_vec) > 20:
                corr_vec = corr_vec[:20]
            elif len(corr_vec) < 20:
                corr_vec = np.pad(corr_vec, (0, 20 - len(corr_vec)), 'constant')
            
            combined_vec = np.concatenate([struct_vec, stat_vec, corr_vec])
            feature_vectors.append(combined_vec)
        
        if feature_vectors:
            feature_matrix = np.array(feature_vectors)
            feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=1.0, neginf=-1.0)
            normalized_matrix = self.scaler.fit_transform(feature_matrix)
            
            for i, node_id in enumerate(node_ids_ordered):
                unified_features[node_id] = normalized_matrix[i]
        
        logger.info(f"统一特征提取完成：{len(unified_features)} 个节点，特征维度 {len(next(iter(unified_features.values()))) if unified_features else 0}")
        return unified_features


    def _extract_data_statistics_features(self, graph_data: Dict, sensor_data: pd.DataFrame,
                                         node_id_to_column_mapping: Dict[str, str]) -> Dict[str, np.ndarray]:
        """提取数据统计特征"""
        from scipy import stats
        stat_features = {}
        
        for node_id, column_name in node_id_to_column_mapping.items():
            if column_name in sensor_data.columns:
                try:
                    data_col = sensor_data[column_name].dropna()
                    if len(data_col) > 0:
                        stat_features[node_id] = np.array([
                            data_col.mean(),
                            data_col.std(),
                            stats.skew(data_col),
                            stats.kurtosis(data_col)
                        ])
                    else:
                        stat_features[node_id] = np.zeros(4)
                except:
                    stat_features[node_id] = np.zeros(4)
            else:
                stat_features[node_id] = np.zeros(4)
        
        return stat_features
    
    def _extract_correlation_features(self, graph_data: Dict, correlation_matrix: pd.DataFrame,
                                     node_id_to_column_mapping: Dict[str, str]) -> Dict[str, np.ndarray]:
        """提取数据相关性特征"""
        corr_features = {}
        
        for node_id, column_name in node_id_to_column_mapping.items():
            if column_name in correlation_matrix.columns:
                corr_vector = correlation_matrix[column_name].values
                corr_vector = np.where(correlation_matrix.index == column_name, 0, corr_vector)
                sorted_indices = np.argsort(np.abs(corr_vector))[::-1]
                corr_features[node_id] = corr_vector[sorted_indices]
            else:
                corr_features[node_id] = np.array([])
        
        return corr_features
    
    def _dict_to_vector(self, feature_dict: Dict, keys: List[str]) -> np.ndarray:
        """将特征字典转换为固定顺序的向量"""
        return np.array([float(feature_dict.get(key, 0.0)) for key in keys])



    def extract_structural_features(self, graph_data: Dict) -> Dict[str, Dict[str, float]]:
        """提取图结构特征"""
        G = nx.DiGraph()

        nodes_by_type = graph_data.get('nodes_by_type', {})
        for node_type, nodes in nodes_by_type.items():
            for node in nodes:
                node_id = node.get('id', '')
                if node_id:
                    G.add_node(node_id, **node)

        edges_dict = graph_data.get('edges', {})
        for edge_key, edge_list in edges_dict.items():
            for edge in edge_list:
                source = edge.get('source', '')
                target = edge.get('target', '')
                if source and target:
                    G.add_edge(source, target, **edge)

        node_features = {}

        degree_centrality = nx.degree_centrality(G)

        try:
            betweenness_centrality = nx.betweenness_centrality(G)
        except:
            betweenness_centrality = {node: 0.0 for node in G.nodes()}

        try:
            pagerank = nx.pagerank(G, alpha=0.85, max_iter=100)
        except:
            pagerank = {node: 1.0/len(G.nodes()) for node in G.nodes()}

        try:
            clustering_coefficient = nx.clustering(G.to_undirected())
        except:
            clustering_coefficient = {node: 0.0 for node in G.nodes()}

        in_degree = dict(G.in_degree())
        out_degree = dict(G.out_degree())

        for node_id in G.nodes():
            node_features[node_id] = {
                'degree_centrality': degree_centrality.get(node_id, 0.0),
                'betweenness_centrality': betweenness_centrality.get(node_id, 0.0),
                'pagerank': pagerank.get(node_id, 0.0),
                'clustering_coefficient': clustering_coefficient.get(node_id, 0.0),
                'in_degree': in_degree.get(node_id, 0),
                'out_degree': out_degree.get(node_id, 0),
                'total_degree': in_degree.get(node_id, 0) + out_degree.get(node_id, 0)
            }

        return node_features

    def extract_data_topology_mismatch_features(self, graph_data: Dict,
                                               sensor_data: pd.DataFrame = None) -> Dict[str, Dict[str, float]]:
        """提取数据-拓扑不匹配特征"""
        node_features = {}

        if sensor_data is None or sensor_data.empty:
            nodes_by_type = graph_data.get('nodes_by_type', {})
            for node_type, nodes in nodes_by_type.items():
                for node in nodes:
                    node_id = node.get('id', '')
                    if node_id:
                        node_features[node_id] = {
                            'data_correlation_strength': 0.0,
                            'topology_connectivity': 0.0,
                            'mismatch_score': 0.0,
                            'causal_desert_score': 0.0,
                            'data_richness': 0.0
                        }
            return node_features

        G = nx.DiGraph()
        nodes_by_type = graph_data.get('nodes_by_type', {})
        for node_type, nodes in nodes_by_type.items():
            for node in nodes:
                node_id = node.get('id', '')
                if node_id:
                    G.add_node(node_id, **node)

        edges_dict = graph_data.get('edges', {})
        for edge_key, edge_list in edges_dict.items():
            for edge in edge_list:
                source = edge.get('source', '')
                target = edge.get('target', '')
                if source and target:
                    G.add_edge(source, target, **edge)

        id_to_text = {}
        for node_type, nodes in nodes_by_type.items():
            for node in nodes:
                node_id = node.get('id', '')
                node_text = node.get('text', node_id)
                if node_id:
                    id_to_text[node_id] = node_text

        try:
            numeric_cols = sensor_data.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 2:
                correlation_matrix = sensor_data[numeric_cols].corr()
            else:
                correlation_matrix = pd.DataFrame()
        except:
            correlation_matrix = pd.DataFrame()

        for node_id in G.nodes():
            node_text = id_to_text.get(node_id, node_id)

            data_correlation_strength = 0.0
            matching_col = self._find_matching_column(node_id, node_text, correlation_matrix.columns.tolist())

            if matching_col and not correlation_matrix.empty:
                correlations = correlation_matrix[matching_col].abs()
                correlations = correlations[correlations.index != matching_col]  # 排除自相关
                if len(correlations) > 0:
                    data_correlation_strength = correlations.mean()

            topology_connectivity = len(list(G.neighbors(node_id))) / max(1, len(G.nodes()) - 1)

            mismatch_score = data_correlation_strength * (1.0 - topology_connectivity)

            strong_correlations = 0
            if matching_col and not correlation_matrix.empty:
                strong_correlations = (correlation_matrix[matching_col].abs() > 0.6).sum() - 1  # 排除自相关

            graph_connections = len(list(G.neighbors(node_id)))
            causal_desert_score = strong_correlations / max(1, graph_connections + 1)

            data_richness = 0.0
            if matching_col and matching_col in sensor_data.columns:
                try:
                    data_richness = sensor_data[matching_col].std() / (sensor_data[matching_col].mean() + 1e-6)
                    data_richness = min(1.0, abs(data_richness))  # 标准化到[0,1]
                except:
                    data_richness = 0.0

            node_features[node_id] = {
                'data_correlation_strength': data_correlation_strength,
                'topology_connectivity': topology_connectivity,
                'mismatch_score': mismatch_score,
                'causal_desert_score': causal_desert_score,
                'data_richness': data_richness
            }

        return node_features

    def _find_matching_column(self, node_id: str, node_text: str, columns: List[str]) -> Optional[str]:
        """寻找节点对应的数据列"""
        if not node_text or not columns:
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

    def extract_hypothesis_driven_features(self, graph_data: Dict,
                                         hypotheses: List[Dict]) -> Dict[str, Dict[str, float]]:
        """提取假设驱动特征"""
        node_features = {}

        nodes_by_type = graph_data.get('nodes_by_type', {})
        for node_type, nodes in nodes_by_type.items():
            for node in nodes:
                node_id = node.get('id', '')
                if node_id:
                    node_features[node_id] = {
                        'hypothesis_count': 0,
                        'avg_hypothesis_confidence': 0.0,
                        'max_hypothesis_confidence': 0.0,
                        'is_data_driven_target': 0.0,
                        'hypothesis_type_diversity': 0.0
                    }

        node_hypothesis_info = {}
        for hyp in hypotheses:
            target_elements = hyp.get('target_elements', [])
            confidence = hyp.get('confidence_score', 0.5)
            hyp_type = hyp.get('type', 'unknown')

            for node_id in target_elements:
                if node_id not in node_hypothesis_info:
                    node_hypothesis_info[node_id] = {
                        'confidences': [],
                        'types': set()
                    }

                node_hypothesis_info[node_id]['confidences'].append(confidence)
                node_hypothesis_info[node_id]['types'].add(hyp_type)

        for node_id, info in node_hypothesis_info.items():
            if node_id in node_features:
                confidences = info['confidences']
                types = info['types']

                node_features[node_id]['hypothesis_count'] = len(confidences)
                node_features[node_id]['avg_hypothesis_confidence'] = np.mean(confidences) if confidences else 0.0
                node_features[node_id]['max_hypothesis_confidence'] = max(confidences) if confidences else 0.0
                node_features[node_id]['is_data_driven_target'] = 1.0 if len(confidences) > 0 else 0.0
                node_features[node_id]['hypothesis_type_diversity'] = len(types) / max(1, len(confidences))

        return node_features



class MonteCarloUncertaintyQuantifier:
    """蒙特卡洛不确定性量化器"""

    def __init__(self, num_samples: int = 50):
        self.num_samples = num_samples

    def monte_carlo_uncertainty(self, base_reconstruction_func, *args, **kwargs) -> Tuple[float, float, float]:
        """蒙特卡洛不确定性量化

        Returns:
            mean_prob: 平均重构概率
            uncertainty: 不确定性（标准差）
            epistemic_uncertainty: 认知不确定性
        """
        reconstruction_probs = []

        for _ in range(self.num_samples):
            noise_factor = np.random.normal(1.0, 0.1)  # 10%的噪声
            prob, _ = base_reconstruction_func(*args, **kwargs)

            noisy_prob = np.clip(prob * noise_factor, 0.01, 0.99)
            reconstruction_probs.append(noisy_prob)

        mean_prob = np.mean(reconstruction_probs)
        std_prob = np.std(reconstruction_probs)

        epistemic_uncertainty = std_prob

        aleatoric_uncertainty = -mean_prob * np.log2(mean_prob) - (1-mean_prob) * np.log2(1-mean_prob)
        total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty * 0.5

        return mean_prob, total_uncertainty, epistemic_uncertainty

class UnifiedScoreCalculator:
    """统一评分计算器 - 将不同类型的不确定性映射到[-1, 1]范围"""

    @staticmethod
    def calculate_unified_score(edge_result: 'EdgeUncertaintyResult') -> float:
        """
        计算统一评分：
        - 现有边：不确定性分数映射到[0, 1]
        - 缺失边：重构概率映射到[-1, 0]，使用 -reconstruction_prob
        - 节点：直接使用当前不确定性分数
        """
        if edge_result.edge_exists:
            return np.clip(edge_result.uncertainty_score, 0.0, 1.0)
        else:
            return -np.clip(edge_result.reconstruction_prob, 0.0, 1.0)

class ParetoAnalyzer:
    """帕累托分析器 - 当统计方法失败时的备选方案"""

    @staticmethod
    def identify_pareto_outliers(values: List[float], pareto_ratio: float = 0.2) -> Dict[str, List[int]]:
        """
        使用帕累托原理识别异常值
        Args:
            values: 数值列表
            pareto_ratio: 帕累托比例，默认0.2（20%）
        Returns:
            包含高异常值索引的字典
        """
        if not values:
            return {'high_outliers': [], 'low_outliers': [], 'thresholds': {'high': 0, 'low': 0}}

        indexed_values = [(i, val) for i, val in enumerate(values)]
        indexed_values.sort(key=lambda x: x[1], reverse=True)

        n_top = max(1, int(len(values) * pareto_ratio))
        n_bottom = max(1, int(len(values) * pareto_ratio))

        high_outliers = [idx for idx, _ in indexed_values[:n_top]]

        low_outliers = [idx for idx, _ in indexed_values[-n_bottom:]]

        high_threshold = indexed_values[n_top-1][1] if n_top <= len(indexed_values) else min(values)
        low_threshold = indexed_values[-n_bottom][1] if n_bottom <= len(indexed_values) else max(values)

        return {
            'high_outliers': high_outliers,
            'low_outliers': low_outliers,
            'thresholds': {
                'high': high_threshold,
                'low': low_threshold,
                'method': 'pareto',
                'pareto_ratio': pareto_ratio
            }
        }

class StatisticalOutlierDetector:
    """统计异常值检测器 - 替代固定阈值（，支持可配置参数）"""

    def __init__(self, iqr_multiplier: float = 1.5, zscore_multiplier: float = 2.0):
        """
        Args:
            iqr_multiplier: IQR方法的乘数因子，默认1.5。更小的值(如1.0, 0.75)会检测更多温和异常值
            zscore_multiplier: Z-score方法的乘数因子，默认2.0
        """
        self.iqr_multiplier = iqr_multiplier
        self.zscore_multiplier = zscore_multiplier

    def adaptive_threshold(self, values: List[float], method: str = 'iqr',
                          iqr_multiplier: Optional[float] = None) -> Dict[str, float]:
        """自适应阈值计算

        Args:
            values: 数值列表
            method: 'iqr', 'zscore', 'modified_zscore'
            iqr_multiplier: 覆盖默认的IQR乘数因子

        Returns:
            包含各种阈值的字典
        """
        if not values:
            return {'low': 0.0, 'high': 1.0, 'mean': 0.5, 'std': 0.0}

        values_array = np.array(values)
        mean_val = np.mean(values_array)
        std_val = np.std(values_array)

        effective_iqr_multiplier = iqr_multiplier if iqr_multiplier is not None else self.iqr_multiplier

        if method == 'iqr':
            q1 = np.percentile(values_array, 25)
            q3 = np.percentile(values_array, 75)
            iqr = q3 - q1

            high_threshold = q3 + effective_iqr_multiplier * iqr
            low_threshold = q1 - effective_iqr_multiplier * iqr

        elif method == 'zscore':
            high_threshold = mean_val + self.zscore_multiplier * std_val
            low_threshold = mean_val - self.zscore_multiplier * std_val

        elif method == 'modified_zscore':
            median_val = np.median(values_array)
            mad = np.median(np.abs(values_array - median_val))
            modified_z_scores = 0.6745 * (values_array - median_val) / mad

            high_threshold = median_val + 2 * mad / 0.6745
            low_threshold = median_val - 2 * mad / 0.6745

        else:
            high_threshold = mean_val + 2 * std_val
            low_threshold = mean_val - 2 * std_val

        return {
            'low': max(0.0, low_threshold),
            'high': min(1.0, high_threshold),
            'mean': mean_val,
            'std': std_val,
            'method': method,
            'iqr_multiplier': effective_iqr_multiplier if method == 'iqr' else None,
            'zscore_multiplier': self.zscore_multiplier if method == 'zscore' else None
        }

    def identify_outliers(self, values: List[float], method: str = 'hybrid') -> Dict[str, List[int]]:
        """识别异常值索引 - 混合策略：统计方法与帕累托分析并行使用"""
        if not values:
            return {'high_outliers': [], 'low_outliers': [], 'thresholds': {'high': 0, 'low': 0}}

        if method == 'iqr' or method == 'zscore':
            try:
                thresholds = self.adaptive_threshold(values, method)
                high_outliers = []
                low_outliers = []

                for i, val in enumerate(values):
                    if val > thresholds['high']:
                        high_outliers.append(i)
                    elif val < thresholds['low']:
                        low_outliers.append(i)

                return {
                    'high_outliers': high_outliers,
                    'low_outliers': low_outliers,
                    'thresholds': thresholds
                }
            except Exception as e:
                logger.info(f"统计异常值检测失败 ({e})，回退到帕累托分析")
                return ParetoAnalyzer.identify_pareto_outliers(values, pareto_ratio=0.2)

        elif method == 'pareto':
            return ParetoAnalyzer.identify_pareto_outliers(values, pareto_ratio=0.2)

        else:  # method == 'hybrid'
            statistical_result = None
            pareto_result = None

            try:
                thresholds = self.adaptive_threshold(values, 'iqr')
                high_outliers_stat = []
                low_outliers_stat = []

                for i, val in enumerate(values):
                    if val > thresholds['high']:
                        high_outliers_stat.append(i)
                    elif val < thresholds['low']:
                        low_outliers_stat.append(i)

                statistical_result = {
                    'high_outliers': high_outliers_stat,
                    'low_outliers': low_outliers_stat,
                    'thresholds': thresholds
                }
            except Exception as e:
                logger.debug(f"统计方法失败: {e}")

            pareto_result = ParetoAnalyzer.identify_pareto_outliers(values, pareto_ratio=0.2)

            if statistical_result and (statistical_result['high_outliers'] or statistical_result['low_outliers']):
                combined_high = list(set(statistical_result['high_outliers'] + pareto_result['high_outliers']))
                combined_low = list(set(statistical_result['low_outliers'] + pareto_result['low_outliers']))

                logger.info(f"混合异常值检测: 统计方法找到 {len(statistical_result['high_outliers'])} 高异常值, "
                           f"帕累托方法找到 {len(pareto_result['high_outliers'])} 高异常值, "
                           f"合并后 {len(combined_high)} 个")

                return {
                    'high_outliers': combined_high,
                    'low_outliers': combined_low,
                    'thresholds': {
                        **statistical_result['thresholds'],
                        'method': 'hybrid',
                        'pareto_ratio': pareto_result['thresholds'].get('pareto_ratio', 0.2)
                    }
                }
            else:
                logger.info(f"统计方法未找到异常值，使用帕累托分析: {len(pareto_result['high_outliers'])} 高异常值")
                return pareto_result

class PCVGAEUncertaintyAnalyzer:
    """PC-VGAE不确定性分析器（真实GNN版本，使用MC Dropout）"""

    def __init__(self, config: SEEKConfig):
        self.config = config
        self.feature_extractor = EnhancedFeatureExtractor()

        iqr_multiplier = getattr(config.pcvgae_config, 'iqr_multiplier', 1.0)  # 默认1.0，比标准1.5更宽松
        self.outlier_detector = StatisticalOutlierDetector(iqr_multiplier=iqr_multiplier)

        self.model = None
        self.mc_quantifier = None

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.learning_rate = config.pcvgae_config.learning_rate
        self.max_epochs = config.pcvgae_config.max_epochs
        self.kl_weight = config.pcvgae_config.kl_weight

    def _prepare_graph_data(self, graph_data: Dict) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, int]]:
        """准备图数据用于PyTorch Geometric"""
        node_to_idx = {}
        idx = 0

        nodes_by_type = graph_data.get('nodes_by_type', {})
        for node_type, nodes in nodes_by_type.items():
            for node in nodes:
                node_id = node.get('id', '')
                if node_id and node_id not in node_to_idx:
                    node_to_idx[node_id] = idx
                    idx += 1

        num_nodes = len(node_to_idx)

        node_features = torch.randn(num_nodes, 64)  # 64维特征

        edge_list = []
        edges_dict = graph_data.get('edges', {})

        for edge_key, edge_list_data in edges_dict.items():
            if isinstance(edge_list_data, list):
                for edge in edge_list_data:
                    if isinstance(edge, dict):
                        source = edge.get('source', '')
                        target = edge.get('target', '')
                        if source in node_to_idx and target in node_to_idx:
                            edge_list.append([node_to_idx[source], node_to_idx[target]])

        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        return node_features, edge_index, node_to_idx

    def train_baseline_model(self, graph_data: Dict, hypotheses: List[Dict]) -> RealPCVGAE:
        """训练基准PC-VGAE模型"""
        logger.info("开始训练真实的PC-VGAE模型...")

        node_features, edge_index, node_to_idx = self._prepare_graph_data(graph_data)
        node_features = node_features.to(self.device)
        edge_index = edge_index.to(self.device)

        prompt_dim = 32  # 软提示维度
        model = RealPCVGAE(
            in_channels=node_features.size(1),
            latent_dim=16
        ).to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=25, factor=0.8)

        neutral_prompt = torch.zeros(prompt_dim, dtype=torch.float32).to(self.device)

        model.train()
        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.max_epochs):
            optimizer.zero_grad()

            edge_recon, mu, logstd = model(node_features, edge_index)

            pos_edge_index = edge_index
            neg_edge_index = negative_sampling(edge_index, num_nodes=node_features.size(0),
                                             num_neg_samples=pos_edge_index.size(1))

            pos_pred = model.decode_edges(model.reparameterize(mu, logstd), pos_edge_index)
            neg_pred = model.decode_edges(model.reparameterize(mu, logstd), neg_edge_index)

            pos_loss = F.binary_cross_entropy(pos_pred, torch.ones_like(pos_pred))
            neg_loss = F.binary_cross_entropy(neg_pred, torch.zeros_like(neg_pred))
            recon_loss = pos_loss + neg_loss

            kl_loss = model.kl_loss(mu, logstd)

            total_loss = recon_loss + self.kl_weight * kl_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            scheduler.step(total_loss)

            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 30:
                    logger.info(f"早停于第 {epoch} 轮")
                    break

            if epoch % 50 == 0:
                logger.info(f"Epoch {epoch}: Loss = {total_loss.item():.4f}, Recon = {recon_loss.item():.4f}, KL = {kl_loss.item():.4f}")

        logger.info(f"模型训练完成，最终损失: {best_loss:.4f}")

        self.model = model
        self.mc_quantifier = MonteCarloDropoutQuantifier(num_samples=self.config.pcvgae_config.monte_carlo_samples)
        self.node_to_idx = node_to_idx
        self.node_features = node_features
        self.edge_index = edge_index
        self.neutral_prompt = neutral_prompt

        return model

    def _base_reconstruction_function(self, source: str, target: str,
                                     exists: bool, hypotheses: List[Dict],
                                     structural_features: Dict[str, Dict[str, float]],
                                     hypothesis_features: Dict[str, Dict[str, float]]) -> Tuple[float, float]:
        """基础重构函数（用于蒙特卡洛采样）"""

        base_prob = 0.8 if exists else 0.2

        relevant_hypotheses = [
            h for h in hypotheses
            if source in h.get('target_elements', []) and target in h.get('target_elements', [])
        ]

        hypothesis_adjustment = 0.0
        if relevant_hypotheses:
            avg_confidence = np.mean([h.get('confidence_score', 0.5) for h in relevant_hypotheses])
            if exists:
                hypothesis_adjustment = -avg_confidence * 0.3  # 现有边：假设置信度高 -> 重构概率低
            else:
                hypothesis_adjustment = avg_confidence * 0.4   # 不存在边：假设置信度高 -> 重构概率高

        source_features = structural_features.get(source, {})
        target_features = structural_features.get(target, {})

        source_centrality = source_features.get('degree_centrality', 0.0)
        target_centrality = target_features.get('degree_centrality', 0.0)
        centrality_similarity = 1.0 - abs(source_centrality - target_centrality)

        source_pagerank = source_features.get('pagerank', 0.0)
        target_pagerank = target_features.get('pagerank', 0.0)
        pagerank_similarity = 1.0 - abs(source_pagerank - target_pagerank)

        structural_adjustment = (centrality_similarity + pagerank_similarity) * 0.1 - 0.1

        source_hyp_features = hypothesis_features.get(source, {})
        target_hyp_features = hypothesis_features.get(target, {})

        source_hyp_count = source_hyp_features.get('hypothesis_count', 0)
        target_hyp_count = target_hyp_features.get('hypothesis_count', 0)
        hypothesis_density = (source_hyp_count + target_hyp_count) / 20.0  # 归一化

        hypothesis_density_adjustment = hypothesis_density * 0.15

        reconstruction_prob = base_prob + hypothesis_adjustment + structural_adjustment + hypothesis_density_adjustment
        reconstruction_prob = np.clip(reconstruction_prob, 0.01, 0.99)

        return reconstruction_prob, 0.0

    def simulate_edge_reconstruction_enhanced(self, source: str, target: str,
                                            exists: bool, hypotheses: List[Dict],
                                            graph_data: Dict,
                                            structural_features: Dict[str, Dict[str, float]],
                                            hypothesis_features: Dict[str, Dict[str, float]]) -> Tuple[float, float]:
        """使用真实PC-VGAE模型进行边重构概率和不确定性计算"""

        if self.model is None or self.mc_quantifier is None:
            return self._fallback_simulation(source, target, exists, hypotheses, structural_features, hypothesis_features)

        if source not in self.node_to_idx or target not in self.node_to_idx:
            return self._fallback_simulation(source, target, exists, hypotheses, structural_features, hypothesis_features)

        source_idx = self.node_to_idx[source]
        target_idx = self.node_to_idx[target]
        target_edges = torch.tensor([[source_idx], [target_idx]], dtype=torch.long).to(self.device)

        mean_prob, uncertainty = self.mc_quantifier.quantify_uncertainty(
            self.model, self.node_features, self.edge_index, self.neutral_prompt, target_edges
        )

        return mean_prob, uncertainty

    def _fallback_simulation(self, source: str, target: str, exists: bool, hypotheses: List[Dict],
                           structural_features: Dict[str, Dict[str, float]],
                           hypothesis_features: Dict[str, Dict[str, float]]) -> Tuple[float, float]:
        """回退的模拟方法（当真实模型不可用时）"""
        base_prob = 0.8 if exists else 0.2

        relevant_hypotheses = [
            h for h in hypotheses
            if source in h.get('target_elements', []) and target in h.get('target_elements', [])
        ]

        hypothesis_adjustment = 0.0
        if relevant_hypotheses:
            avg_confidence = np.mean([h.get('confidence_score', 0.5) for h in relevant_hypotheses])
            if exists:
                hypothesis_adjustment = -avg_confidence * 0.3  # 现有边：假设置信度高 -> 重构概率低
            else:
                hypothesis_adjustment = avg_confidence * 0.4   # 不存在边：假设置信度高 -> 重构概率高

        source_features = structural_features.get(source, {})
        target_features = structural_features.get(target, {})

        structural_adjustment = 0.0
        if source_features and target_features:
            source_degree = source_features.get('degree_centrality', 0.5)
            target_degree = target_features.get('degree_centrality', 0.5)
            degree_similarity = 1 - abs(source_degree - target_degree)
            structural_adjustment += degree_similarity * 0.1

        final_prob = base_prob + hypothesis_adjustment + structural_adjustment
        final_prob = np.clip(final_prob, 0.01, 0.99)

        uncertainty = -final_prob * np.log2(final_prob) - (1-final_prob) * np.log2(1-final_prob)

        return final_prob, uncertainty

    def simulate_edge_reconstruction(self, source: str, target: str,
                                   exists: bool, hypotheses: List[Dict]) -> Tuple[float, float]:
        """保持向后兼容的简化版本"""
        base_prob = 0.8 if exists else 0.2

        relevant_hypotheses = [
            h for h in hypotheses
            if source in h.get('target_elements', []) and target in h.get('target_elements', [])
        ]

        if relevant_hypotheses:
            avg_confidence = np.mean([h.get('confidence_score', 0.5) for h in relevant_hypotheses])
            if exists:
                reconstruction_prob = base_prob * (1 - avg_confidence * 0.5)
            else:
                reconstruction_prob = base_prob + avg_confidence * 0.6
        else:
            reconstruction_prob = base_prob

        prob = np.clip(reconstruction_prob, 0.01, 0.99)
        uncertainty = -prob * np.log2(prob) - (1-prob) * np.log2(1-prob)

        return reconstruction_prob, uncertainty
    
    def analyze_all_edges_enhanced(self, existing_edges: Set[Tuple[str, str]],
                                  negative_edges: List[Tuple[str, str]],
                                  hypotheses: List[Dict],
                                  graph_data: Dict) -> Tuple[List[EdgeUncertaintyResult], float, float]:
        """增强的边分析（使用真实PC-VGAE模型）"""
        edge_results = []

        if self.model is None:
            logger.info("训练PC-VGAE基准模型...")
            self.train_baseline_model(graph_data, hypotheses)

        logger.info("提取图结构特征...")
        structural_features = self.feature_extractor.extract_structural_features(graph_data)

        logger.info("提取假设驱动特征...")
        hypothesis_features = self.feature_extractor.extract_hypothesis_driven_features(graph_data, hypotheses)

        node_text_map = {}
        nodes_by_type = graph_data.get('nodes_by_type', {})
        for node_type, nodes in nodes_by_type.items():
            for node in nodes:
                node_id = node.get('id', '')
                node_text = node.get('text', node_id)
                if node_id:
                    node_text_map[node_id] = node_text

        logger.info(f"分析现有边 ({len(existing_edges)} 条)...")
        existing_edge_data = []
        for source, target in existing_edges:
            recon_prob, uncertainty = self.simulate_edge_reconstruction_enhanced(
                source, target, True, hypotheses, graph_data,
                structural_features, hypothesis_features
            )
            existing_edge_data.append({
                'source': source,
                'target': target,
                'recon_prob': recon_prob,
                'uncertainty': uncertainty
            })

        logger.info(f"分析负样本边 ({len(negative_edges)} 条)...")
        negative_edge_data = []
        for source, target in negative_edges:
            recon_prob, uncertainty = self.simulate_edge_reconstruction_enhanced(
                source, target, False, hypotheses, graph_data,
                structural_features, hypothesis_features
            )
            negative_edge_data.append({
                'source': source,
                'target': target,
                'recon_prob': recon_prob,
                'uncertainty': uncertainty
            })

        logger.info("计算自适应阈值...")

        existing_uncertainties = [edge['uncertainty'] for edge in existing_edge_data]
        existing_outliers = self.outlier_detector.identify_outliers(existing_uncertainties, method='iqr')
        existing_high_threshold = existing_outliers['thresholds']['high']

        negative_probs = [edge['recon_prob'] for edge in negative_edge_data]
        negative_outliers = self.outlier_detector.identify_outliers(negative_probs, method='iqr')
        negative_high_threshold = negative_outliers['thresholds']['high']

        logger.info(f"自适应阈值 - 现有边不确定性: {existing_high_threshold:.4f}, 负样本边重构概率: {negative_high_threshold:.4f}")

        for edge_data in existing_edge_data:
            source, target = edge_data['source'], edge_data['target']
            recon_prob, uncertainty = edge_data['recon_prob'], edge_data['uncertainty']

            if uncertainty > existing_high_threshold:
                edge_type = 'existing_unreliable'
            else:
                edge_type = 'existing_reliable'

            result = EdgeUncertaintyResult(
                source_id=source,
                target_id=target,
                source_text=node_text_map.get(source, source),
                target_text=node_text_map.get(target, target),
                edge_exists=True,
                reconstruction_prob=recon_prob,
                uncertainty_score=uncertainty,
                edge_type=edge_type,
                confidence_interval=(max(0, uncertainty-0.1), min(1, uncertainty+0.1))
            )
            result.unified_score = UnifiedScoreCalculator.calculate_unified_score(result)
            edge_results.append(result)

        for edge_data in negative_edge_data:
            source, target = edge_data['source'], edge_data['target']
            recon_prob, uncertainty = edge_data['recon_prob'], edge_data['uncertainty']

            if recon_prob > negative_high_threshold:
                edge_type = 'missing_potential'
                uncertainty_score = -uncertainty
            else:
                edge_type = 'missing_irrelevant'
                uncertainty_score = uncertainty

            result = EdgeUncertaintyResult(
                source_id=source,
                target_id=target,
                source_text=node_text_map.get(source, source),
                target_text=node_text_map.get(target, target),
                edge_exists=False,
                reconstruction_prob=recon_prob,
                uncertainty_score=uncertainty_score,
                edge_type=edge_type,
                confidence_interval=(max(-1, uncertainty_score-0.1), min(1, uncertainty_score+0.1))
            )
            result.unified_score = UnifiedScoreCalculator.calculate_unified_score(result)
            edge_results.append(result)

        return edge_results, existing_high_threshold, negative_high_threshold

    def analyze_all_edges(self, existing_edges: Set[Tuple[str, str]],
                         negative_edges: List[Tuple[str, str]],
                         hypotheses: List[Dict],
                         graph_data: Dict) -> List[EdgeUncertaintyResult]:
        """分析所有边的不确定性"""
        edge_results = []
        
        node_text_map = {}
        for node in graph_data.get('nodes', []):
            node_text_map[node.get('id', '')] = node.get('text', node.get('id', ''))
        
        for source, target in existing_edges:
            recon_prob, uncertainty = self.simulate_edge_reconstruction(source, target, True, hypotheses)
            
            if uncertainty > 0.6:  # 降低阈值以保持兼容性
                edge_type = 'existing_unreliable'
            else:
                edge_type = 'existing_reliable'
            
            result = EdgeUncertaintyResult(
                source_id=source,
                target_id=target,
                source_text=node_text_map.get(source, source),
                target_text=node_text_map.get(target, target),
                edge_exists=True,
                reconstruction_prob=recon_prob,
                uncertainty_score=uncertainty,
                edge_type=edge_type,
                confidence_interval=(max(0, uncertainty-0.1), min(1, uncertainty+0.1))
            )
            edge_results.append(result)
        
        for source, target in negative_edges:
            recon_prob, uncertainty = self.simulate_edge_reconstruction(source, target, False, hypotheses)
            
            if recon_prob > 0.7:  # 高重构概率的缺失边
                edge_type = 'missing_potential'
                uncertainty_score = -uncertainty
            else:
                edge_type = 'missing_irrelevant'
                uncertainty_score = uncertainty
            
            result = EdgeUncertaintyResult(
                source_id=source,
                target_id=target,
                source_text=node_text_map.get(source, source),
                target_text=node_text_map.get(target, target),
                edge_exists=False,
                reconstruction_prob=recon_prob,
                uncertainty_score=uncertainty_score,
                edge_type=edge_type,
                confidence_interval=(max(-1, uncertainty_score-0.1), min(1, uncertainty_score+0.1))
            )
            edge_results.append(result)
        
        return edge_results

class CausalKnowledgeGraphLoader:
    """因果知识图谱加载器"""

    def __init__(self):
        pass

    def load_ckg(self, ckg_file: Path) -> Dict[str, Any]:
        """加载因果知识图谱文件"""
        with open(ckg_file, 'r', encoding='utf-8') as f:
            return json.load(f)

class DataProcessor:
    """数据处理器 - 增强版"""

    def __init__(self):
        pass

    def create_node_to_column_mapping(self, ckg: Dict[str, Any], 
                                     sensor_data: pd.DataFrame) -> Dict[str, str]:
        """创建节点ID到传感器数据列名的映射
        
        这是连接CKG与传感器数据的关键"数字线程"
        """
        mapping = {}
        
        sensor_columns = sensor_data.columns.tolist()
        
        if 'nodes_by_type' in ckg:
            for node_type, nodes in ckg['nodes_by_type'].items():
                for node in nodes:
                    node_id = node.get('id', '')
                    node_text = node.get('text', '')
                    
                    if not node_id:
                        continue
                    
                    matched_column = self._find_best_column_match(
                        node_id, node_text, sensor_columns
                    )
                    
                    if matched_column:
                        mapping[node_id] = matched_column
                        logger.debug(f"映射: {node_id} -> {matched_column}")
        
        logger.info(f"创建节点-数据列映射: {len(mapping)} 个节点成功映射到数据列")
        return mapping
    
    def _find_best_column_match(self, node_id: str, node_text: str, 
                               columns: List[str]) -> Optional[str]:
        """寻找最佳匹配的数据列"""
        if node_text in columns:
            return node_text
        
        for col in columns:
            if col.startswith('sensor_') and node_text.lower() in col.lower():
                return col
        
        node_text_lower = node_text.lower().replace('_', '').replace('-', '')
        for col in columns:
            col_clean = col.lower().replace('_', '').replace('-', '')
            if node_text_lower in col_clean or col_clean in node_text_lower:
                return col
        
        node_id_clean = node_id.lower().replace('_', '').replace('-', '')
        for col in columns:
            col_clean = col.lower().replace('_', '').replace('-', '')
            if node_id_clean in col_clean or col_clean in node_id_clean:
                return col
        
        return None

    def process_data(self, data: pd.DataFrame, ckg: Dict[str, Any]) -> Dict[str, Any]:
        """处理数据并生成图结构和特征 - 增强版"""
        import networkx as nx

        graph = nx.DiGraph()

        if 'nodes_by_type' in ckg:
            for node_type, nodes in ckg['nodes_by_type'].items():
                for node in nodes:
                    node_id = node.get('id', '')
                    if node_id:
                        graph.add_node(node_id, **node)

        if 'edges' in ckg:
            edges_data = ckg['edges']
            if isinstance(edges_data, dict):
                for edge_key, edge_list in edges_data.items():
                    if isinstance(edge_list, list):
                        for edge in edge_list:
                            if isinstance(edge, dict):
                                source = edge.get('source', '')
                                target = edge.get('target', '')
                                if source and target:
                                    graph.add_edge(source, target, **edge)

        correlation_matrix = pd.DataFrame()
        try:
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 2:
                correlation_matrix = data[numeric_cols].corr()
                logger.info(f"计算相关性矩阵: {correlation_matrix.shape}")
        except Exception as e:
            logger.warning(f"计算相关性矩阵失败: {e}")

        node_to_column_mapping = self.create_node_to_column_mapping(ckg, data)

        node_features = {}
        try:
            degree_centrality = nx.degree_centrality(graph)
            for node_id in graph.nodes():
                node_features[node_id] = {
                    'degree_centrality': degree_centrality.get(node_id, 0.0),
                    'in_degree': graph.in_degree(node_id),
                    'out_degree': graph.out_degree(node_id)
                }
        except Exception as e:
            logger.warning(f"计算节点特征失败: {e}")

        return {
            'graph': graph,
            'correlation_matrix': correlation_matrix,
            'node_features': node_features,
            'node_to_column_mapping': node_to_column_mapping,
            'num_nodes': len(graph.nodes()),
            'num_edges': len(graph.edges())
        }


class DataProcessorv0:
    """数据处理器"""

    def __init__(self):
        pass

    def process_data(self, data: pd.DataFrame, ckg: Dict[str, Any]) -> Dict[str, Any]:
        """处理数据并生成图结构和特征"""
        import networkx as nx

        graph = nx.DiGraph()

        if 'nodes_by_type' in ckg:
            for node_type, nodes in ckg['nodes_by_type'].items():
                for node in nodes:
                    node_id = node.get('id', '')
                    if node_id:
                        graph.add_node(node_id, **node)
        elif 'nodes' in ckg:
            for node in ckg['nodes']:
                node_id = node.get('id', '')
                if node_id:
                    graph.add_node(node_id, **node)

        if 'edges' in ckg:
            edges_data = ckg['edges']
            if isinstance(edges_data, dict):
                for edge_key, edge_list in edges_data.items():
                    if isinstance(edge_list, list):
                        for edge in edge_list:
                            if isinstance(edge, dict):
                                source = edge.get('source', '')
                                target = edge.get('target', '')
                                if source and target:
                                    graph.add_edge(source, target, **edge)
            elif isinstance(edges_data, list):
                for edge in edges_data:
                    source = edge.get('source', '')
                    target = edge.get('target', '')
                    if source and target:
                        graph.add_edge(source, target, **edge)

        try:
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 2:
                correlation_matrix = data[numeric_cols].corr()
            else:
                correlation_matrix = pd.DataFrame()
        except Exception as e:
            logger.warning(f"计算相关性矩阵失败: {e}")
            correlation_matrix = pd.DataFrame()

        node_features = {}
        try:
            degree_centrality = nx.degree_centrality(graph)
            for node_id in graph.nodes():
                node_features[node_id] = {
                    'degree_centrality': degree_centrality.get(node_id, 0.0),
                    'in_degree': graph.in_degree(node_id),
                    'out_degree': graph.out_degree(node_id)
                }
        except Exception as e:
            logger.warning(f"计算节点特征失败: {e}")
            for node_id in graph.nodes():
                node_features[node_id] = {
                    'degree_centrality': 0.0,
                    'in_degree': 0,
                    'out_degree': 0
                }

        return {
            'graph': graph,
            'correlation_matrix': correlation_matrix,
            'node_features': node_features,
            'num_nodes': len(graph.nodes()),
            'num_edges': len(graph.edges())
        }

class SEEKPCVGAEService:
    """SEEK PC-VGAE服务"""

    def __init__(self, config: SEEKConfig):
        self.config = config
        self.analyzer = PCVGAEUncertaintyAnalyzer(config)
    
    def load_case_data(self) -> Tuple[List[Dict], Dict[str, Any], Dict[str, Any], pd.DataFrame]:
        """加载案例数据（包含传感器数据）- 增加对假设文件不存在的处理"""
        hypotheses = [] # 默认空列表
        hypothesis_file = self.config.paths.get_hypothesis_file()

        if hypothesis_file.exists():
            hypotheses = SEEKDataInterface.load_hypotheses(hypothesis_file)
        else:
            logger.warning(f"假设文件不存在: {hypothesis_file}。对于基线VGAE模式，这是正常行为。")


        graph_file = self.config.paths.case_dir / "causal_knowledge_graph.json"
        with open(graph_file, 'r', encoding='utf-8') as f:
            graph_data = json.load(f)

        coverage_file = self.config.paths.get_coverage_file()
        coverage_report = {}
        if coverage_file.exists():
            coverage_report = SEEKDataInterface.load_coverage_report(coverage_file)

        sensor_data = pd.DataFrame()
        try:
            sensor_file = self.config.paths.case_dir / "sensor_data.csv"
            if sensor_file.exists():
                sensor_data = pd.read_csv(sensor_file)
                logger.info(f"加载传感器数据: {sensor_data.shape}")
            else:
                logger.warning(f"传感器数据文件不存在: {sensor_file}")
        except Exception as e:
            logger.warning(f"加载传感器数据失败: {e}")

        return hypotheses, graph_data, coverage_report, sensor_data
    
    def generate_complete_uncertainty_map(self) -> CognitiveUncertaintyMap:
        """生成完整的认知不确定性地图"""
        logger.info(f"开始生成案例 {self.config.paths.case_id} 的完整认知不确定性地图")
        
        hypotheses, graph_data, coverage_report, sensor_data = self.load_case_data()
        
        sampler = NegativeSampler(graph_data, sampling_ratio=1.5)
        existing_edges = sampler.existing_edges
        negative_edges = sampler.generate_negative_samples()
        
        logger.info(f"现有边数量: {len(existing_edges)}, 负样本边数量: {len(negative_edges)}")
        
        edge_results, existing_high_threshold, negative_high_threshold = self.analyzer.analyze_all_edges_enhanced(existing_edges, negative_edges, hypotheses, graph_data)

        logger.info(f"自适应阈值 - 现有边不确定性: {existing_high_threshold:.4f}, 负样本边重构概率: {negative_high_threshold:.4f}")
        
        logger.info("生成增强的节点不确定性...")

        structural_features = self.analyzer.feature_extractor.extract_structural_features(graph_data)
        hypothesis_features = self.analyzer.feature_extractor.extract_hypothesis_driven_features(graph_data, hypotheses)

        logger.info("提取数据-拓扑不匹配特征...")
        mismatch_features = self.analyzer.feature_extractor.extract_data_topology_mismatch_features(graph_data, sensor_data)

        node_uncertainties = []
        node_text_map = {}
        all_node_ids = set()
        nodes_by_type = graph_data.get('nodes_by_type', {})
        for node_type, nodes in nodes_by_type.items():
            for node in nodes:
                node_id = node.get('id', '')
                node_text = node.get('text', node_id)
                if node_id:
                    node_text_map[node_id] = node_text
                    all_node_ids.add(node_id)

        for node_id in all_node_ids:
            struct_feat = structural_features.get(node_id, {})
            hyp_feat = hypothesis_features.get(node_id, {})
            mismatch_feat = mismatch_features.get(node_id, {})

            hypothesis_uncertainty = 0.0
            if hyp_feat.get('hypothesis_count', 0) > 0:
                hyp_count = hyp_feat.get('hypothesis_count', 0)
                avg_confidence = hyp_feat.get('avg_hypothesis_confidence', 0.0)
                type_diversity = hyp_feat.get('hypothesis_type_diversity', 0.0)

                count_uncertainty = min(0.5, hyp_count * 0.1)

                confidence_uncertainty = 1.0 - avg_confidence

                diversity_uncertainty = type_diversity * 0.2

                hypothesis_uncertainty = count_uncertainty + confidence_uncertainty + diversity_uncertainty

            structural_uncertainty = 0.0
            if struct_feat:
                degree_centrality = struct_feat.get('degree_centrality', 0.0)
                degree_anomaly = abs(degree_centrality - 0.5) * 0.4  # 偏离中等程度越多越不确定

                betweenness = struct_feat.get('betweenness_centrality', 0.0)
                betweenness_uncertainty = betweenness * 0.3

                clustering = struct_feat.get('clustering_coefficient', 0.0)
                clustering_uncertainty = (1.0 - clustering) * 0.2

                structural_uncertainty = degree_anomaly + betweenness_uncertainty + clustering_uncertainty

            edge_uncertainty = self._calculate_node_edge_uncertainty(node_id, edge_results)

            mismatch_uncertainty = 0.0
            if mismatch_feat:
                mismatch_score = mismatch_feat.get('mismatch_score', 0.0)

                causal_desert_score = mismatch_feat.get('causal_desert_score', 0.0)

                data_richness = mismatch_feat.get('data_richness', 0.0)

                mismatch_uncertainty = (mismatch_score * 0.5 +
                                      causal_desert_score * 0.3 +
                                      data_richness * 0.2)
                mismatch_uncertainty = np.clip(mismatch_uncertainty, 0.0, 1.0)

            total_uncertainty = (hypothesis_uncertainty * 0.35 +
                               structural_uncertainty * 0.25 +
                               edge_uncertainty * 0.25 +
                               mismatch_uncertainty * 0.15)
            total_uncertainty = np.clip(total_uncertainty, 0.0, 1.0)

            if total_uncertainty > 0.1:
                evidence_count = hyp_feat.get('hypothesis_count', 0)
                evidence_strength = min(1.0, hyp_feat.get('avg_hypothesis_confidence', 0.0) + np.random.uniform(0.1, 0.3))

                node_uncertainties.append({
                    'node_id': node_id,
                    'node_text': node_text_map.get(node_id, node_id),
                    'uncertainty_score': total_uncertainty,
                    'unified_score': total_uncertainty,  # 节点直接使用不确定性分数作为统一评分
                    'hypothesis_count': hyp_feat.get('hypothesis_count', 0),
                    'avg_confidence': hyp_feat.get('avg_hypothesis_confidence', 0.0),
                    'confidence_interval': (max(0, total_uncertainty-0.1), min(1, total_uncertainty+0.1)),
                    'evidence_count': evidence_count,  # 为BCSA_04兼容性
                    'evidence_strength': evidence_strength,  # 为BCSA_03兼容性
                    'node_type': 'uncertain',
                    'structural_features': struct_feat,
                    'hypothesis_features': hyp_feat,
                    'mismatch_features': mismatch_feat,
                    'uncertainty_components': {
                        'hypothesis_uncertainty': hypothesis_uncertainty,
                        'structural_uncertainty': structural_uncertainty,
                        'edge_uncertainty': edge_uncertainty,
                        'mismatch_uncertainty': mismatch_uncertainty
                    }
                })
        
        if node_uncertainties:
            node_uncertainty_scores = [node['uncertainty_score'] for node in node_uncertainties]
            node_outliers = self.analyzer.outlier_detector.identify_outliers(node_uncertainty_scores, method='hybrid')
            high_uncertainty_indices = node_outliers['high_outliers']

            high_uncertainty_nodes = [node_uncertainties[i] for i in high_uncertainty_indices]

            high_uncertainty_nodes.sort(key=lambda x: x['uncertainty_score'], reverse=True)

            logger.info(f"节点异常值检测: 发现 {len(high_uncertainty_nodes)} 个高不确定性节点")
            method_used = node_outliers['thresholds'].get('method', 'unknown')
            if method_used == 'pareto':
                logger.info(f"使用帕累托分析 (比例: {node_outliers['thresholds'].get('pareto_ratio', 0.2)})")
            elif method_used == 'hybrid':
                logger.info(f"使用混合方法: 统计+帕累托分析，提高覆盖率")
            else:
                logger.info(f"使用统计方法: {method_used}, 阈值: {node_outliers['thresholds'].get('high', 'N/A')}")
        else:
            high_uncertainty_nodes = []
        
        type_a_edges = [edge for edge in edge_results
                       if edge.edge_type == 'existing_unreliable']
        type_a_edges.sort(key=lambda x: x.uncertainty_score, reverse=True)

        type_b_edges = [edge for edge in edge_results
                       if edge.edge_type == 'missing_potential']
        type_b_edges.sort(key=lambda x: x.uncertainty_score)
        
        all_uncertainties = [node['uncertainty_score'] for node in node_uncertainties] + \
                           [abs(edge.uncertainty_score) for edge in edge_results]
        global_uncertainty_score = np.mean(all_uncertainties) if all_uncertainties else 0.0
        
        summary_report = {
            'total_nodes_analyzed': len(node_uncertainties),
            'total_edges_analyzed': len(edge_results),
            'high_uncertainty_nodes_count': len(high_uncertainty_nodes),
            'type_a_edges_count': len(type_a_edges),
            'type_b_edges_count': len(type_b_edges),
            'global_uncertainty_score': global_uncertainty_score,
            'top_uncertain_nodes': [
                {'node_id': node['node_id'], 'node_text': node['node_text'], 
                 'uncertainty_score': node['uncertainty_score']}
                for node in high_uncertainty_nodes[:5]
            ],
            'top_type_a_edges': [
                {'source': edge.source_text, 'target': edge.target_text, 
                 'uncertainty_score': edge.uncertainty_score}
                for edge in type_a_edges[:5]
            ],
            'top_type_b_edges': [
                {'source': edge.source_text, 'target': edge.target_text, 
                 'uncertainty_score': edge.uncertainty_score}
                for edge in type_b_edges[:5]
            ]
        }
        
        uncertainty_map = CognitiveUncertaintyMap(
            case_id=self.config.paths.case_id,
            node_uncertainties=node_uncertainties,
            edge_uncertainties=edge_results,
            high_uncertainty_nodes=high_uncertainty_nodes,
            type_a_edges=type_a_edges,
            type_b_edges=type_b_edges,
            global_uncertainty_score=global_uncertainty_score,
            summary_report=summary_report
        )
        
        logger.info(f"完整认知不确定性地图生成完成")
        logger.info(f"  - 高不确定性节点: {len(high_uncertainty_nodes)}")
        logger.info(f"  - 类型A边(现有不可靠): {len(type_a_edges)}")
        logger.info(f"  - 类型B边(缺失潜在): {len(type_b_edges)}")
        
        return uncertainty_map

    def _calculate_node_edge_uncertainty(self, node_id: str, edge_results: List[EdgeUncertaintyResult]) -> float:
        """计算节点的边不确定性 - 基于连接边的MC采样结果"""
        connected_edges = []

        for edge in edge_results:
            if edge.source_id == node_id or edge.target_id == node_id:
                connected_edges.append(edge)

        if not connected_edges:
            return 0.0

        edge_uncertainties = []
        for edge in connected_edges:
            uncertainty = abs(edge.uncertainty_score)
            edge_uncertainties.append(uncertainty)

        avg_edge_uncertainty = np.mean(edge_uncertainties)

        edge_count_factor = min(1.0, len(connected_edges) / 10.0)  # 标准化到[0,1]

        edge_types = set()
        for edge in connected_edges:
            edge_types.add(edge.edge_type)

        type_diversity_factor = len(edge_types) / 4.0  # 最多4种类型，标准化到[0,1]

        edge_uncertainty = avg_edge_uncertainty * (1.0 + edge_count_factor * 0.2 + type_diversity_factor * 0.1)

        return min(1.0, edge_uncertainty)

    def save_uncertainty_map(self, uncertainty_map: CognitiveUncertaintyMap) -> Tuple[Path, Path]:
        """保存认知不确定性地图（JSON + 摘要文件）"""
        timestamp = pd.Timestamp.now().isoformat()

        map_dict = {
            'case_id': uncertainty_map.case_id,
            'global_uncertainty_score': uncertainty_map.global_uncertainty_score,
            'summary_report': uncertainty_map.summary_report,
            'node_uncertainties': uncertainty_map.node_uncertainties,
            'edge_uncertainties': [
                {
                    'source_id': edge.source_id,
                    'target_id': edge.target_id,
                    'source_text': edge.source_text,
                    'target_text': edge.target_text,
                    'edge_exists': edge.edge_exists,
                    'reconstruction_prob': edge.reconstruction_prob,
                    'uncertainty_score': edge.uncertainty_score,
                    'edge_type': edge.edge_type,
                    'confidence_interval': edge.confidence_interval,
                    'evidence_strength': abs(edge.uncertainty_score)  # 为BCSA_03兼容性
                }
                for edge in uncertainty_map.edge_uncertainties
            ],
            'high_uncertainty_nodes': uncertainty_map.high_uncertainty_nodes,
            'type_a_edges': [
                {
                    'source_id': edge.source_id,
                    'target_id': edge.target_id,
                    'source_text': edge.source_text,
                    'target_text': edge.target_text,
                    'uncertainty_score': edge.uncertainty_score,
                    'edge_type': edge.edge_type,
                    'evidence_strength': abs(edge.uncertainty_score)  # 为BCSA_03兼容性
                }
                for edge in uncertainty_map.type_a_edges
            ],
            'type_b_edges': [
                {
                    'source_id': edge.source_id,
                    'target_id': edge.target_id,
                    'source_text': edge.source_text,
                    'target_text': edge.target_text,
                    'uncertainty_score': edge.uncertainty_score,
                    'edge_type': edge.edge_type,
                    'evidence_strength': abs(edge.uncertainty_score)  # 为BCSA_03兼容性
                }
                for edge in uncertainty_map.type_b_edges
            ],
            'metadata': {
                'generation_timestamp': timestamp,
                'config': self.config.pcvgae_config.__dict__,
                'model_type': 'Real PC-VGAE with MC Dropout' if self.analyzer.model is not None else 'Fallback Simulation'
            }
        }

        json_file = self.config.paths.get_audit_file()
        SEEKDataInterface.save_uncertainty_map(map_dict, json_file)

        summary_file = json_file.parent / f"{json_file.stem}_focus_summary.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"=== BCSA PC-VGAE 不确定性地图摘要 ===\n")
            f.write(f"案例ID: {uncertainty_map.case_id}\n")
            f.write(f"生成时间: {timestamp}\n")
            f.write(f"全局不确定性分数: {uncertainty_map.global_uncertainty_score:.4f}\n")
            f.write(f"模型类型: {'真实PC-VGAE模型' if self.analyzer.model is not None else '回退模拟模型'}\n\n")

            f.write(f"=== 总体统计 ===\n")
            f.write(f"分析节点数: {len(uncertainty_map.node_uncertainties)}\n")
            f.write(f"分析边数: {len(uncertainty_map.edge_uncertainties)}\n")
            f.write(f"高不确定性节点数: {len(uncertainty_map.high_uncertainty_nodes)}\n")
            f.write(f"类型A边数（现有不可靠）: {len(uncertainty_map.type_a_edges)}\n")
            f.write(f"类型B边数（缺失潜在）: {len(uncertainty_map.type_b_edges)}\n\n")

            f.write(f"=== 重点关注节点（前5个）===\n")
            for i, node in enumerate(uncertainty_map.high_uncertainty_nodes[:5], 1):
                f.write(f"{i}. {node['node_text']} (ID: {node['node_id']})\n")
                f.write(f"   不确定性分数: {node['uncertainty_score']:.4f}\n")
                f.write(f"   假设数量: {node.get('hypothesis_count', 0)}\n\n")

            f.write(f"=== 类型A边（现有不可靠，前3个）===\n")
            for i, edge in enumerate(uncertainty_map.type_a_edges[:3], 1):
                f.write(f"{i}. {edge.source_text} → {edge.target_text}\n")
                f.write(f"   不确定性分数: {edge.uncertainty_score:.4f}\n")
                f.write(f"   重构概率: {edge.reconstruction_prob:.4f}\n\n")

            f.write(f"=== 类型B边（缺失潜在，前3个）===\n")
            for i, edge in enumerate(uncertainty_map.type_b_edges[:3], 1):
                f.write(f"{i}. {edge.source_text} → {edge.target_text}\n")
                f.write(f"   不确定性分数: {edge.uncertainty_score:.4f}\n")
                f.write(f"   重构概率: {edge.reconstruction_prob:.4f}\n\n")

        logger.info(f"认知不确定性地图已保存:")
        logger.info(f"  - JSON文件: {json_file}")
        logger.info(f"  - 摘要文件: {summary_file}")

        return json_file, summary_file

def main():
    """主函数 - 演示PC-VGAE功能"""
    print("=== BCSA PC-VGAE审计模块演示 ===")
    
    config = SEEKConfig.create_for_case("Mixed_small_07")
    print(f"配置案例: {config.paths.case_id}")
    
    pcvgae_service = SEEKPCVGAEService(config)
    
    try:
        uncertainty_map = pcvgae_service.generate_complete_uncertainty_map()
        
        json_file, summary_file = pcvgae_service.save_uncertainty_map(uncertainty_map)

        print(f"\n✅ PC-VGAE审计成功完成!")
        print(f"📊 全局不确定性分数: {uncertainty_map.global_uncertainty_score:.4f}")
        print(f"📊 分析节点数: {len(uncertainty_map.node_uncertainties)}")
        print(f"📊 分析边数: {len(uncertainty_map.edge_uncertainties)}")
        print(f"📁 输出文件:")
        print(f"   - JSON文件: {json_file}")
        print(f"   - 摘要文件: {summary_file}")
        
        print(f"\n🔍 重点关注对象:")
        print(f"   高不确定性节点 (前5个):")
        for node in uncertainty_map.high_uncertainty_nodes[:5]:
            print(f"     - {node['node_text']} (ID: {node['node_id']}): {node['uncertainty_score']:.4f}")
        
        print(f"   类型A边 (现有不可靠, 前3个):")
        for edge in uncertainty_map.type_a_edges[:3]:
            print(f"     - {edge.source_text} → {edge.target_text}: {edge.uncertainty_score:.4f}")
        
        print(f"   类型B边 (缺失潜在, 前3个):")
        for edge in uncertainty_map.type_b_edges[:3]:
            print(f"     - {edge.source_text} → {edge.target_text}: {edge.uncertainty_score:.4f}")
        
    except Exception as e:
        print(f"❌ PC-VGAE审计失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
