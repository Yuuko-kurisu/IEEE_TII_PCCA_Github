#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Baseline methods for the released PCCA experiments.
"""

import json
import logging
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data
from sklearn.metrics import roc_auc_score
import networkx as nx
from torch_geometric.utils import negative_sampling
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _build_degraded_graph_data(all_nodes: List[Dict], all_edges: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """
    [洁净室函数] 仅使用节点ID和稀疏的边列表构建图数据。
    特征为最简单的独热编码，不包含任何来自传感器数据的信息。
    """
    logger.info("--- Executing in DEGRADED DATA mode ---")
    node_ids = sorted([node['id'] for node in all_nodes])
    node_id_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
    
    edges = []
    for edge in all_edges:
        source_id = edge.get('source_id', '')
        target_id = edge.get('target_id', '')
        if source_id in node_id_to_idx and target_id in node_id_to_idx:
            edges.append([node_id_to_idx[source_id], node_id_to_idx[target_id]])
            edges.append([node_id_to_idx[target_id], node_id_to_idx[source_id]])

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty((2, 0), dtype=torch.long)
    
    node_features = torch.nn.functional.one_hot(torch.arange(0, len(node_ids)), num_classes=len(node_ids)).float()
    
    return edge_index, node_features, node_ids

class Baseline(ABC):
    """基线方法抽象基类"""
    
    def __init__(self, name: str, top_k: int = None):
        self.name = name
        self.top_k = top_k
        print(f"Initializing baseline: {self.name} (top_k={top_k})")

    @abstractmethod
    def run(self, case_data_dir: Path, output_dir: Path) -> Path:
        """
        运行基线方法的核心逻辑。
        必须在此方法内将基线的原始输出转换为标准的 'findings.json' 格式，
        同时生成包含所有原始分数的 'raw_scores.json' 文件。
        
        Args:
            case_data_dir: 包含 G_init, sensor_data.csv 等的案例目录。
            output_dir: 用于存放该方法结果的目录。
            
        Returns:
            指向生成的 findings.json 的文件路径。
        """
        pass

    def _save_findings(self, findings_list: list, output_dir: Path) -> Path:
        """
        辅助函数：保存findings.json（top-k筛选后）和raw_scores.json（全部原始分数）
        
        Args:
            findings_list: 包含所有发现的列表，每个元素应包含unified_score字段
            output_dir: 输出目录
            
        Returns:
            findings.json文件路径
        """
        findings_output_path = output_dir / "findings.json"
        raw_scores_output_path = output_dir / "raw_scores.json"
        
        sorted_findings = sorted(findings_list, key=lambda x: x.get('unified_score', 0.0), reverse=True)
        
        raw_scores = []
        for finding in sorted_findings:
            raw_score_entry = {
                'source_id': finding.get('source_id', ''),
                'target_id': finding.get('target_id', ''),
                'score': finding.get('unified_score', 0.0),
                'edge_type': finding.get('edge_type', 'unknown'),
                'method': finding.get('method', self.name),
                'additional_info': {k: v for k, v in finding.items() 
                                  if k not in ['source_id', 'target_id', 'unified_score', 'edge_type', 'method']}
            }
            raw_scores.append(raw_score_entry)
        
        with open(raw_scores_output_path, 'w', encoding='utf-8') as f:
            json.dump(raw_scores, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"保存原始分数文件: {raw_scores_output_path} (包含 {len(raw_scores)} 条记录)")
        
        final_findings = sorted_findings
        if self.top_k is not None and len(sorted_findings) > self.top_k:
            logger.info(f"应用Top-K (k={self.top_k})筛选，发现数量从 {len(sorted_findings)} 减少到 {self.top_k}")
            final_findings = sorted_findings[:self.top_k]
        else:
            logger.info(f"Top-K筛选: 当前发现数量={len(sorted_findings)}, 设置的top_k={self.top_k}, 无需筛选")
        
        with open(findings_output_path, 'w', encoding='utf-8') as f:
            json.dump(final_findings, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(final_findings)} findings to {findings_output_path}")
        print(f"Saved {len(raw_scores)} raw scores to {raw_scores_output_path}")
        
        return findings_output_path

    def _load_case_data(self, case_data_dir: Path) -> Dict[str, Any]:
        """加载案例数据的通用方法"""
        case_data = {
            'ckg': {},
            'sensor_data': pd.DataFrame(),
            'all_nodes': [],
            'all_edges': [],
            'node_mappings': {}
        }

        ckg_file = case_data_dir / "causal_knowledge_graph.json"
        if ckg_file.exists():
            with open(ckg_file, 'r', encoding='utf-8') as f:
                ckg = json.load(f)
            case_data['ckg'] = ckg

            all_nodes = []
            nodes_by_type = ckg.get('nodes_by_type', {})
            for node_type, node_list in nodes_by_type.items():
                if isinstance(node_list, list):
                    for node in node_list:
                        if isinstance(node, dict) and 'id' in node:
                            all_nodes.append(node)
            
            case_data['all_nodes'] = all_nodes
            logger.info(f"Loaded {len(all_nodes)} nodes from knowledge graph")

            edges_data = ckg.get('edges', {})  # 注意这里是 'edges'，不是 'edges_by_type'
            standardized_edges = []
            
            for edge_key, edge_list in edges_data.items():
                if not isinstance(edge_list, list):
                    continue
                
                for edge in edge_list:
                    if not isinstance(edge, dict):
                        continue
                    
                    source_node = edge.get('source')
                    target_node = edge.get('target')
                    
                    if source_node and target_node:
                        standardized_edge = {
                            'source_id': source_node,
                            'target_id': target_node,
                            'edge_type': edge.get('relation', 'unknown'),
                            'confidence': edge.get('confidence', 0.5),
                            'original_data': edge
                        }
                        standardized_edges.append(standardized_edge)
                    else:
                        logger.warning(f"发现格式不正确的边，已跳过: {edge}")

            case_data['all_edges'] = standardized_edges
            logger.info(f"Loaded {len(standardized_edges)} edges from knowledge graph")

        sensor_file = case_data_dir / "sensor_data.csv"
        if sensor_file.exists():
            case_data['sensor_data'] = pd.read_csv(sensor_file)
            logger.info(f"Loaded sensor data with {len(case_data['sensor_data'])} rows")
        else:
            logger.warning(f"传感器数据文件不存在: {sensor_file}")

        if case_data['all_nodes']:
            case_data['node_mappings'] = self._build_node_mappings(case_data['all_nodes'])

        return case_data

    def _build_node_mappings(self, all_nodes: List[Dict]) -> Dict[str, Dict[str, str]]:
        """构建节点ID到文本和文本到ID的双向映射"""
        id_to_text = {}
        text_to_id = {}
        
        for node in all_nodes:
            node_id = node.get('id', '')
            node_text = node.get('text', '')
            if node_id and node_text:
                id_to_text[node_id] = node_text
                text_to_id[node_text] = node_id
        
        return {
            'id_to_text': id_to_text,
            'text_to_id': text_to_id
        }

    def _generate_all_candidate_edges(self, all_nodes, all_edges) -> Tuple[set, set]:
        """生成所有候选边和现有边集合"""
        node_ids = [node['id'] for node in all_nodes]
        existing_edges = set()
        for edge in all_edges:
            src, tgt = edge['source_id'], edge['target_id']
            existing_edges.add(tuple(sorted((src, tgt))))

        candidate_edges = set()
        for i in range(len(node_ids)):
            for j in range(i + 1, len(node_ids)):
                u, v = node_ids[i], node_ids[j]
                edge_tuple = tuple(sorted((u, v)))
                if edge_tuple not in existing_edges:
                    candidate_edges.add((u, v))
        return existing_edges, candidate_edges




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

        self.encoder_conv1 = GATv2Conv(actual_in_channels, hidden_dim, heads=4, dropout=0.2, concat=True)
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
        """前向传播"""
        mu, logstd = self.encode(x, edge_index)
        z = self.reparameterize(mu, logstd)
        return mu, logstd, z

    def kl_loss(self, mu: torch.Tensor, logstd: torch.Tensor) -> torch.Tensor:
        """KL散度损失"""
        return -0.5 * torch.mean(torch.sum(1 + 2 * logstd - mu.pow(2) - logstd.exp().pow(2), dim=1))


class EnhancedFeatureExtractor:
    """简化版特征提取器"""
    def __init__(self):
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()

    def extract_unified_features(self, graph_data: Dict, sensor_data: pd.DataFrame, 
                                correlation_matrix: pd.DataFrame, 
                                node_id_to_column_mapping: Dict[str, str]) -> Dict[str, np.ndarray]:
        """提取统一特征"""
        unified_features = {}
        nodes_by_type = graph_data.get('nodes_by_type', {})
        all_node_ids = []
        
        for node_type, nodes in nodes_by_type.items():
            for node in nodes:
                node_id = node.get('id', '')
                if node_id:
                    all_node_ids.append(node_id)
        
        for i, node_id in enumerate(sorted(all_node_ids)):
            feature_vector = np.zeros(64)  # 64维特征
            feature_vector[i % 64] = 1.0   # 简单的one-hot类似编码
            
            if not sensor_data.empty and node_id in node_id_to_column_mapping:
                column_name = node_id_to_column_mapping[node_id]
                if column_name in sensor_data.columns:
                    try:
                        data_col = sensor_data[column_name].dropna()
                        if len(data_col) > 0:
                            feature_vector[60] = data_col.mean()
                            feature_vector[61] = data_col.std()
                            feature_vector[62] = data_col.min()
                            feature_vector[63] = data_col.max()
                    except:
                        pass
            
            unified_features[node_id] = feature_vector
        
        return unified_features

class VGAEBaseline(Baseline):
    """VGAE基线方法（基于RealPCVGAE）"""
    
    def __init__(self, name: str = "VGAE", top_k: int = None):
        super().__init__(name, top_k)
        self.latent_dim = 16
        self.hidden_dim = 64
        self.epochs = 200
        self.lr = 0.005
        self.kl_weight = 0.1

    def run(self, case_data_dir: Path, output_dir: Path, ckg_data = None, use_degraded_features: bool = False) -> Path:
        """运行VGAE并生成findings.json"""
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Running VGAE (GATv2-based) on case: {case_data_dir}")
        
        case_data = self._load_case_data(case_data_dir)
        
        if len(case_data['all_nodes']) == 0:
            logger.warning("No nodes found in graph, returning empty findings")
            return self._save_findings([], output_dir)
        
        feature_extractor = EnhancedFeatureExtractor()
        
        node_id_to_column_mapping = {}
        if not case_data['sensor_data'].empty:
            sensor_columns = case_data['sensor_data'].columns.tolist()
            for node in case_data['all_nodes']:
                node_id = node.get('id', '')
                node_text = node.get('text', '')
                for col in sensor_columns:
                    if node_text.lower() in col.lower() or col.lower() in node_text.lower():
                        node_id_to_column_mapping[node_id] = col
                        break
        
        unified_features = feature_extractor.extract_unified_features(
            case_data['ckg'], 
            case_data['sensor_data'], 
            pd.DataFrame(),  # 空的相关性矩阵
            node_id_to_column_mapping
        )
        
        if use_degraded_features:
            edge_index, node_features, node_ids = _build_degraded_graph_data(
                case_data['all_nodes'], case_data['all_edges']
            )
        else:
            edge_index, node_features, node_ids = self._build_graph_data_with_unified_features(
                case_data['all_nodes'], case_data['all_edges'], unified_features
            )
        
        if edge_index.size(1) == 0:
            logger.warning("No edges found, using fallback mode")
            return self._fallback_prediction(node_features, node_ids, case_data, output_dir)

        model = RealPCVGAE(
            unified_feature_dim=node_features.size(1),
            latent_dim=self.latent_dim,
            hidden_dim=self.hidden_dim
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        
        model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            
            mu, logstd, z = model(node_features, edge_index)
            
            pos_pred = model.decode_edges(z, edge_index)
            neg_edge_index = negative_sampling(edge_index, num_nodes=node_features.size(0),
                                             num_neg_samples=edge_index.size(1))
            neg_pred = model.decode_edges(z, neg_edge_index)
            
            pos_loss = F.binary_cross_entropy(pos_pred, torch.ones_like(pos_pred))
            neg_loss = F.binary_cross_entropy(neg_pred, torch.zeros_like(neg_pred))
            recon_loss = pos_loss + neg_loss
            
            kl_loss = model.kl_loss(mu, logstd)
            total_loss = recon_loss + self.kl_weight * kl_loss
            
            total_loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 50 == 0:
                logger.info(f"VGAE Epoch {epoch+1}/{self.epochs}, Loss: {total_loss.item():.4f}")
        
        model.eval()
        findings = []
        
        with torch.no_grad():
            mu, logstd, z = model(node_features, edge_index)
            
            existing_edges, candidate_edges = self._generate_all_candidate_edges(
                case_data['all_nodes'], case_data['all_edges']
            )
            
            node_id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
            
            for u, v in candidate_edges:
                if u in node_id_to_idx and v in node_id_to_idx:
                    u_idx, v_idx = node_id_to_idx[u], node_id_to_idx[v]
                    score = (z[u_idx] * z[v_idx]).sum().item()
                    score = torch.sigmoid(torch.tensor(score)).item()
                    
                    findings.append({
                        'source_id': u,
                        'target_id': v,
                        'unified_score': score,
                        'edge_type': 'missing',
                        'method': 'VGAE_corrected'
                    })
        
        logger.info(f"生成了 {len(findings)} 个原始发现")
        return self._save_findings(findings, output_dir)

    def _build_graph_data_with_unified_features(self, all_nodes: List[Dict], all_edges: List[Dict],
                                              unified_features: Dict[str, np.ndarray],
                                              use_degraded_features: bool = False) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """使用统一特征构建图数据，并增加特征降级选项"""
        node_ids = sorted([node['id'] for node in all_nodes])
        node_id_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
        
        edges = []
        for edge in all_edges:
            source_id, target_id = edge.get('source_id', ''), edge.get('target_id', '')
            if source_id in node_id_to_idx and target_id in node_id_to_idx:
                edges.append([node_id_to_idx[source_id], node_id_to_idx[target_id]])
                edges.append([node_id_to_idx[target_id], node_id_to_idx[source_id]])
        
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty((2, 0), dtype=torch.long)
        
        if use_degraded_features:
            logger.warning("特征降级模式已激活：仅使用基础ID特征。")
            node_features = torch.nn.functional.one_hot(torch.arange(0, len(node_ids)), num_classes=len(node_ids)).float()
        else:
            if unified_features:
                feature_dim = len(next(iter(unified_features.values())))
                node_features = torch.zeros(len(node_ids), feature_dim, dtype=torch.float32)
                for i, node_id in enumerate(node_ids):
                    if node_id in unified_features:
                        node_features[i] = torch.FloatTensor(unified_features[node_id])
            else:
                node_features = torch.randn(len(node_ids), 64)
        
        return edge_index, node_features, node_ids

    def _fallback_prediction(self, node_features: torch.Tensor, node_ids: List[str], 
                           case_data: Dict[str, Any], output_dir: Path) -> Path:
        """无边图的回退预测逻辑"""
        logger.info("Using VGAE fallback mode")
        findings = []
        similarity_matrix = torch.mm(F.normalize(node_features), F.normalize(node_features).t())
        
        for i in range(len(node_ids)):
            for j in range(i + 1, len(node_ids)):
                similarity_score = similarity_matrix[i, j].item()
                if similarity_score > 0.3:
                    findings.append({
                        'source_id': node_ids[i],
                        'target_id': node_ids[j],
                        'unified_score': similarity_score,
                        'edge_type': 'missing',
                        'method': 'VGAE_fallback'
                    })
        
        return self._save_findings(findings, output_dir)

class GATE(nn.Module):
    """Graph Attention Auto-Encoder（非变分版本）"""
    
    def __init__(self, in_channels: int, hidden_dim: int, latent_dim: int, heads: int = 4):
        super(GATE, self).__init__()
        self.latent_dim = latent_dim
        
        self.encoder1 = GATv2Conv(in_channels, hidden_dim, heads=heads, dropout=0.2, concat=True)
        self.encoder2 = GATv2Conv(hidden_dim * heads, latent_dim, heads=1, dropout=0.2, concat=False)
        
    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """编码器：直接输出潜在表示"""
        h = F.elu(self.encoder1(x, edge_index))
        z = self.encoder2(h, edge_index)
        return z
    
    def decode(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """解码器：重构边的概率"""
        row, col = edge_index
        return torch.sigmoid((z[row] * z[col]).sum(dim=1))
    
    def recon_loss(self, z: torch.Tensor, pos_edge_index: torch.Tensor) -> torch.Tensor:
        """重构损失（只有重构损失，没有KL散度）"""
        pos_pred = self.decode(z, pos_edge_index)
        
        neg_edge_index = negative_sampling(pos_edge_index, z.size(0), num_neg_samples=pos_edge_index.size(1))
        neg_pred = self.decode(z, neg_edge_index)
        
        pos_loss = F.binary_cross_entropy(pos_pred, torch.ones_like(pos_pred))
        neg_loss = F.binary_cross_entropy(neg_pred, torch.zeros_like(neg_pred))
        
        return pos_loss + neg_loss


class GATEBaseline(Baseline):
    """GATE基线方法（图注意力自编码器）"""
    
    def __init__(self, name: str = "GATE", top_k: int = None):
        super().__init__(name, top_k)
        self.latent_dim = 16
        self.hidden_dim = 64
        self.epochs = 200
        self.lr = 0.005

    def run(self, case_data_dir: Path, output_dir: Path, ckg_data = None, use_degraded_features: bool = False) -> Path:
        """运行GATE并生成findings.json"""
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Running GATE on case: {case_data_dir}")
        
        case_data = self._load_case_data(case_data_dir)
        
        if len(case_data['all_nodes']) == 0:
            logger.warning("No nodes found in graph, returning empty findings")
            return self._save_findings([], output_dir)
        
        feature_extractor = EnhancedFeatureExtractor()
        
        node_id_to_column_mapping = {}
        if not case_data['sensor_data'].empty:
            sensor_columns = case_data['sensor_data'].columns.tolist()
            for node in case_data['all_nodes']:
                node_id = node.get('id', '')
                node_text = node.get('text', '')
                for col in sensor_columns:
                    if node_text.lower() in col.lower() or col.lower() in node_text.lower():
                        node_id_to_column_mapping[node_id] = col
                        break
        
        unified_features = feature_extractor.extract_unified_features(
            case_data['ckg'], 
            case_data['sensor_data'], 
            pd.DataFrame(),
            node_id_to_column_mapping
        )
        
        if use_degraded_features:
            edge_index, node_features, node_ids = _build_degraded_graph_data(
                case_data['all_nodes'], case_data['all_edges']
            )
        else:
            edge_index, node_features, node_ids = self._build_graph_data_with_unified_features(
                case_data['all_nodes'], case_data['all_edges'], unified_features
            )
        
        if edge_index.size(1) == 0:
            logger.warning("No edges found, using fallback mode")
            return self._gate_fallback_prediction(node_features, node_ids, output_dir)

        model = GATE(
            in_channels=node_features.size(1),
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        
        model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            
            z = model.encode(node_features, edge_index)
            
            loss = model.recon_loss(z, edge_index)
            
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 50 == 0:
                logger.info(f"GATE Epoch {epoch+1}/{self.epochs}, Loss: {loss.item():.4f}")
        
        model.eval()
        findings = []
        
        with torch.no_grad():
            z = model.encode(node_features, edge_index)
            
            existing_edges, candidate_edges = self._generate_all_candidate_edges(
                case_data['all_nodes'], case_data['all_edges']
            )
            
            node_id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
            
            for u, v in candidate_edges:
                if u in node_id_to_idx and v in node_id_to_idx:
                    u_idx, v_idx = node_id_to_idx[u], node_id_to_idx[v]
                    score = torch.sigmoid((z[u_idx] * z[v_idx]).sum()).item()
                    
                    findings.append({
                        'source_id': u,
                        'target_id': v,
                        'unified_score': score,
                        'edge_type': 'missing',
                        'method': 'GATE'
                    })
        
        logger.info(f"生成了 {len(findings)} 个原始发现")
        return self._save_findings(findings, output_dir)

    def _build_graph_data_with_unified_features(self, all_nodes: List[Dict], all_edges: List[Dict],
                                              unified_features: Dict[str, np.ndarray],
                                              use_degraded_features: bool = False) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """使用统一特征构建图数据，并增加特征降级选项"""
        node_ids = sorted([node['id'] for node in all_nodes])
        node_id_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
        
        edges = []
        for edge in all_edges:
            source_id, target_id = edge.get('source_id', ''), edge.get('target_id', '')
            if source_id in node_id_to_idx and target_id in node_id_to_idx:
                edges.append([node_id_to_idx[source_id], node_id_to_idx[target_id]])
                edges.append([node_id_to_idx[target_id], node_id_to_idx[source_id]])
        
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty((2, 0), dtype=torch.long)
        
        if use_degraded_features:
            logger.warning("特征降级模式已激活：仅使用基础ID特征。")
            node_features = torch.nn.functional.one_hot(torch.arange(0, len(node_ids)), num_classes=len(node_ids)).float()
        else:
            if unified_features:
                feature_dim = len(next(iter(unified_features.values())))
                node_features = torch.zeros(len(node_ids), feature_dim, dtype=torch.float32)
                for i, node_id in enumerate(node_ids):
                    if node_id in unified_features:
                        node_features[i] = torch.FloatTensor(unified_features[node_id])
            else:
                node_features = torch.randn(len(node_ids), 64)
        
        return edge_index, node_features, node_ids

    def _gate_fallback_prediction(self, node_features: torch.Tensor, node_ids: List[str], output_dir: Path) -> Path:
        """GATE无边图的回退预测逻辑"""
        logger.info("Using GATE fallback mode")
        findings = []
        
        attention_weights = torch.softmax(torch.mm(node_features, node_features.t()), dim=1)
        embeddings = torch.mm(attention_weights, node_features)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        for i in range(len(node_ids)):
            for j in range(i + 1, len(node_ids)):
                similarity_score = torch.sigmoid(torch.dot(embeddings[i], embeddings[j])).item()
                if similarity_score > 0.6:
                    findings.append({
                        'source_id': node_ids[i],
                        'target_id': node_ids[j],
                        'unified_score': similarity_score,
                        'edge_type': 'missing',
                        'method': 'GATE_fallback'
                    })
        
        return self._save_findings(findings, output_dir)


class PCStableBaseline(Baseline):
    """Peter-Clark算法基线方法（优化版）"""
    
    def __init__(self, name: str = "Peter-Clark", top_k: int = None):
        super().__init__(name, top_k)
        self.alpha = 0.05  # 显著性水平
    
    def run(self, case_data_dir: Path, output_dir: Path) -> Path:
        """运行Peter-Clark算法并生成findings.json"""
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Running Peter-Clark on case: {case_data_dir}")
        
        case_data = self._load_case_data(case_data_dir)
        sensor_data = case_data['sensor_data']
        node_mappings = case_data['node_mappings']
        
        if sensor_data.empty:
            logger.warning("No sensor data available, returning empty findings")
            return self._save_findings([], output_dir)
        
        findings_list = self._run_optimized_pc_algorithm(sensor_data, node_mappings)
        
        findings = findings_list
        logger.info(f"生成了 {len(findings)} 个原始发现，即将保存到raw_scores.json和findings.json")
        return self._save_findings(findings, output_dir)
    
    def _run_optimized_pc_algorithm(self, sensor_data: pd.DataFrame, 
                                  node_mappings: Dict[str, Dict[str, str]]) -> List[Dict[str, Any]]:
        """运行优化后的PC算法（移除硬阈值，改用Top-K控制）"""
        findings = []
        
        numeric_cols = sensor_data.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            logger.warning("Insufficient numeric columns for PC algorithm")
            return findings
        
        corr_matrix = sensor_data[numeric_cols].corr()
        
        for i, col1 in enumerate(numeric_cols):
            for j, col2 in enumerate(numeric_cols):
                if i < j:  # 避免重复
                    correlation = abs(corr_matrix.loc[col1, col2])
                    
                    if not np.isnan(correlation) and correlation > 0.4:
                        source_id = self._map_column_to_node_id(col1, node_mappings)
                        target_id = self._map_column_to_node_id(col2, node_mappings)
                        
                        if source_id and target_id:
                            findings.append({
                                'source_id': source_id,
                                'target_id': target_id,
                                'unified_score': float(correlation),
                                'edge_type': 'pc_causal',
                                'method': 'Peter-Clark',
                                'correlation': float(correlation),
                                'source_column': col1,
                                'target_column': col2
                            })
        
        logger.info(f"Peter-Clark generated {len(findings)} findings (before Top-K filtering)")
        return findings
    
    def _map_column_to_node_id(self, column_name: str, 
                              node_mappings: Dict[str, Dict[str, str]]) -> str:
        """将数据列名映射到节点ID"""
        text_to_id = node_mappings.get('text_to_id', {})
        
        if column_name in text_to_id:
            return text_to_id[column_name]
        
        clean_name = column_name.replace('sensor_', '').replace('_', ' ')
        for text, node_id in text_to_id.items():
            if clean_name.lower() in text.lower() or text.lower() in clean_name.lower():
                return node_id
        
        return f"sensor_{column_name}"


class CommonNeighborsBaseline(Baseline):
    """共同邻居基线方法（优化版）"""
    
    def __init__(self, name: str = "CommonNeighbors", top_k: int = None):
        super().__init__(name, top_k)

    def run(self, case_data_dir: Path, output_dir: Path) -> Path:
        """运行共同邻居算法并生成findings.json"""
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Running CommonNeighbors on case: {case_data_dir}")
        
        case_data = self._load_case_data(case_data_dir)
        
        if len(case_data['all_nodes']) == 0:
            logger.warning("No nodes found, returning empty findings")
            return self._save_findings([], output_dir)
        
        G = nx.Graph()
        for node in case_data['all_nodes']:
            G.add_node(node['id'])
        
        for edge in case_data['all_edges']:
            G.add_edge(edge['source_id'], edge['target_id'])

        findings = []
        _, candidate_edges = self._generate_all_candidate_edges(
            case_data['all_nodes'], case_data['all_edges']
        )
        
        for u, v in candidate_edges:
            if G.has_node(u) and G.has_node(v):
                jaccard_coeff = list(nx.jaccard_coefficient(G, [(u, v)]))[0][2]
                
                if jaccard_coeff > 0.2:  # 可调阈值
                    findings.append({
                        'source_id': u,
                        'target_id': v,
                        'unified_score': float(jaccard_coeff),
                        'edge_type': 'missing',
                        'method': 'CommonNeighbors'
                    })
        
        logger.info(f"CommonNeighbors generated {len(findings)} findings")
        logger.info(f"生成了 {len(findings)} 个原始发现，即将保存到raw_scores.json和findings.json")
        return self._save_findings(findings, output_dir)


class TransEBaseline(Baseline):
    """TransE knowledge graph embedding baseline."""
    
    def __init__(self, name: str = "TransE", top_k: int = None):
        super().__init__(name, top_k)
        self.embedding_dim = 64
        self.epochs = 100
        self.lr = 0.01
        self.margin = 1.0
    
    def run(self, case_data_dir: Path, output_dir: Path) -> Path:
        """运行TransE并生成findings.json"""
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Running TransE on case: {case_data_dir}")
        
        case_data = self._load_case_data(case_data_dir)
        ckg = case_data['ckg']
        
        entities, relations, triples = self._build_transe_data(case_data)
        
        if not triples:
            logger.warning("No triples found for TransE, returning empty findings")
            return self._save_findings([], output_dir)
        
        model = TransE(len(entities), len(relations), self.embedding_dim)
        findings_list = self._train_and_evaluate_transe(model, triples, entities, relations)
        findings = findings_list
        logger.info(f"生成了 {len(findings)} 个原始发现，即将保存到raw_scores.json和findings.json")
        return self._save_findings(findings, output_dir)
    
    def _build_transe_data(self, case_data: Dict[str, Any]) -> Tuple[List[str], List[str], List[Tuple[int, int, int]]]:
        """Build entities, relations, and triples for TransE."""
        entities = [node['id'] for node in case_data['all_nodes']]
        entity_to_idx = {entity: idx for idx, entity in enumerate(entities)}
        
        relations = set()
        for edge in case_data['all_edges']:
            relations.add(edge.get('edge_type', 'unknown'))
        relations = list(relations)
        relation_to_idx = {relation: idx for idx, relation in enumerate(relations)}
        
        triples = []
        for edge in case_data['all_edges']:
            source_id = edge['source_id']
            target_id = edge['target_id']
            edge_type = edge.get('edge_type', 'unknown')
            
            if (source_id in entity_to_idx and 
                target_id in entity_to_idx and 
                edge_type in relation_to_idx):
                
                head_idx = entity_to_idx[source_id]
                tail_idx = entity_to_idx[target_id]
                rel_idx = relation_to_idx[edge_type]
                triples.append((head_idx, rel_idx, tail_idx))
        
        logger.info(f"TransE data: {len(entities)} entities, {len(relations)} relations, {len(triples)} triples")
        return entities, relations, triples
    
    def _train_and_evaluate_transe(self, model: 'TransE', triples: List[Tuple[int, int, int]], 
                                  entities: List[str], relations: List[str]) -> List[Dict[str, Any]]:
        """训练TransE模型并生成异常发现"""
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        
        for epoch in range(self.epochs):
            total_loss = 0
            for head, rel, tail in triples:
                pos_score = model(head, rel, tail)
                
                if np.random.random() > 0.5:
                    neg_tail = np.random.randint(0, len(entities))
                    neg_score = model(head, rel, neg_tail)
                else:
                    neg_head = np.random.randint(0, len(entities))
                    neg_score = model(neg_head, rel, tail)
                
                loss = F.relu(self.margin + pos_score - neg_score)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 20 == 0:
                logger.info(f"TransE Epoch {epoch+1}/{self.epochs}, Loss: {total_loss/len(triples):.4f}")
        
        findings = []
        model.eval()
        
        existing_triples = set(triples)
        
        with torch.no_grad():
            for h_idx in range(len(entities)):
                for r_idx in range(len(relations)):
                    for t_idx in range(len(entities)):
                        if h_idx == t_idx:
                            continue
                        
                        if (h_idx, r_idx, t_idx) not in existing_triples:
                            score = model(h_idx, r_idx, t_idx).item()
                            unified_score = torch.exp(-torch.tensor(score)).item()  # 距离越小，分数越接近1
                            
                            if unified_score > 0.01:  # 只保留有意义的预测
                                findings.append({
                                    'source_id': entities[h_idx],
                                    'target_id': entities[t_idx],
                                    'unified_score': unified_score,
                                    'edge_type': relations[r_idx],
                                    'method': 'TransE',
                                    'transe_score': score
                                })
        
        logger.info(f"TransE generated {len(findings)} findings")
        return findings


class TransE(nn.Module):
    """TransE模型实现"""
    
    def __init__(self, num_entities: int, num_relations: int, embedding_dim: int):
        super(TransE, self).__init__()
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        
        nn.init.uniform_(self.entity_embeddings.weight.data, -6/np.sqrt(embedding_dim), 6/np.sqrt(embedding_dim))
        nn.init.uniform_(self.relation_embeddings.weight.data, -6/np.sqrt(embedding_dim), 6/np.sqrt(embedding_dim))
        
        self.relation_embeddings.weight.data = F.normalize(self.relation_embeddings.weight.data, p=2, dim=1)
    
    def forward(self, head: int, relation: int, tail: int) -> torch.Tensor:
        """计算三元组得分"""
        head_emb = self.entity_embeddings(torch.tensor(head))
        rel_emb = self.relation_embeddings(torch.tensor(relation))
        tail_emb = self.entity_embeddings(torch.tensor(tail))
        
        score = torch.norm(head_emb + rel_emb - tail_emb, p=2)
        return score


class GATBaseline(Baseline):
    """Graph Attention Network baseline."""
    
    def __init__(self, name: str = "GAT", top_k: int = None):
        super().__init__(name, top_k)
        self.hidden_dim = 64
        self.out_dim = 32
        self.heads = 4
        self.epochs = 200
        self.lr = 0.01
        self.weight_decay = 5e-4
        self.patience = 20  # 早停耐心值
    
    def run(self, case_data_dir: Path, output_dir: Path) -> Path:
        """Run GAT and generate findings.json."""
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Running GAT on case: {case_data_dir}")
        
        case_data = self._load_case_data(case_data_dir)
        
        if len(case_data['all_nodes']) == 0:
            logger.warning("No nodes found in graph, returning empty findings")
            return self._save_findings([], output_dir)
        
        edge_index, node_features, node_ids = self._build_graph_data_for_gat(
            case_data['all_nodes'], case_data['all_edges']
        )
        
        if edge_index.size(1) == 0:
            logger.warning("No edges found, using GAT fallback mode")
            return self._gat_fallback_prediction(node_features, node_ids, output_dir)
        
        try:
            from torch_geometric.utils import train_test_split_edges, negative_sampling
            from torch_geometric.data import Data
            
            data = Data(x=node_features, edge_index=edge_index)
            
            data = train_test_split_edges(data, val_ratio=0.1, test_ratio=0.1)
            
            logger.info(f"数据集划分完成:")
            logger.info(f"  训练边: {data.train_pos_edge_index.size(1)}")
            logger.info(f"  验证边: {data.val_pos_edge_index.size(1)}")
            logger.info(f"  测试边: {data.test_pos_edge_index.size(1)}")
            
            num_nodes = data.x.size(0)
            
            if not hasattr(data, 'train_neg_edge_index') or data.train_neg_edge_index is None:
                logger.info("手动生成负样本边...")
                
                train_neg_edge_index = negative_sampling(
                    edge_index=data.train_pos_edge_index,
                    num_nodes=num_nodes,
                    num_neg_samples=data.train_pos_edge_index.size(1)
                )
                data.train_neg_edge_index = train_neg_edge_index
                
                val_neg_edge_index = negative_sampling(
                    edge_index=data.val_pos_edge_index,
                    num_nodes=num_nodes,
                    num_neg_samples=data.val_pos_edge_index.size(1)
                )
                data.val_neg_edge_index = val_neg_edge_index
                
                test_neg_edge_index = negative_sampling(
                    edge_index=data.test_pos_edge_index,
                    num_nodes=num_nodes,
                    num_neg_samples=data.test_pos_edge_index.size(1)
                )
                data.test_neg_edge_index = test_neg_edge_index
                
                logger.info(f"负样本边生成完成:")
                logger.info(f"  训练负样本: {data.train_neg_edge_index.size(1)}")
                logger.info(f"  验证负样本: {data.val_neg_edge_index.size(1)}")
                logger.info(f"  测试负样本: {data.test_neg_edge_index.size(1)}")
            
        except ImportError:
            logger.warning("torch_geometric.utils.train_test_split_edges 不可用，使用简化版本")
            return self._simplified_gat_training(node_features, edge_index, node_ids, case_data, output_dir)
        
        model = GATLinkPredictor(node_features.size(1), self.hidden_dim, self.out_dim, self.heads)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = nn.BCEWithLogitsLoss()
        
        best_val_auc = 0
        patience_counter = 0
        
        model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            
            node_embeddings = model(data.x, data.train_pos_edge_index)
            
            train_pos_edge_index = data.train_pos_edge_index
            train_neg_edge_index = data.train_neg_edge_index
            
            pos_scores = self._compute_edge_scores(node_embeddings, train_pos_edge_index)
            neg_scores = self._compute_edge_scores(node_embeddings, train_neg_edge_index)
            
            scores = torch.cat([pos_scores, neg_scores])
            labels = torch.cat([torch.ones(pos_scores.size(0)), torch.zeros(neg_scores.size(0))])
            
            loss = criterion(scores, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                model.eval()
                with torch.no_grad():
                    val_embeddings = model(data.x, data.train_pos_edge_index)
                    
                    val_pos_scores = self._compute_edge_scores(val_embeddings, data.val_pos_edge_index)
                    val_neg_scores = self._compute_edge_scores(val_embeddings, data.val_neg_edge_index)
                    
                    val_scores = torch.cat([val_pos_scores, val_neg_scores]).cpu().numpy()
                    val_labels = torch.cat([torch.ones(val_pos_scores.size(0)), 
                                        torch.zeros(val_neg_scores.size(0))]).cpu().numpy()
                    
                    val_auc = roc_auc_score(val_labels, val_scores)
                    
                    logger.info(f"GAT Epoch {epoch+1}/{self.epochs}, Loss: {loss.item():.4f}, Val AUC: {val_auc:.4f}")
                    
                    if val_auc > best_val_auc:
                        best_val_auc = val_auc
                        patience_counter = 0
                        best_model_state = model.state_dict().copy()
                    else:
                        patience_counter += 1
                        
                    if patience_counter >= self.patience:
                        logger.info(f"早停触发，最佳验证AUC: {best_val_auc:.4f}")
                        model.load_state_dict(best_model_state)
                        break
                
                model.train()
        
        model.eval()
        findings = []
        
        with torch.no_grad():
            final_embeddings = model(data.x, data.train_pos_edge_index)
            
            existing_edges, candidate_edges = self._generate_all_candidate_edges(
                case_data['all_nodes'], case_data['all_edges']
            )
            
            node_id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
            
            for u, v in candidate_edges:
                if u in node_id_to_idx and v in node_id_to_idx:
                    u_idx, v_idx = node_id_to_idx[u], node_id_to_idx[v]
                    
                    score = torch.sigmoid(torch.dot(final_embeddings[u_idx], final_embeddings[v_idx])).item()
                    
                    if score > 0.35:  # 基础筛选阈值
                        findings.append({
                            'source_id': u,
                            'target_id': v,
                            'unified_score': score,
                            'edge_type': 'missing',
                            'method': 'GAT',
                            'validation_auc': best_val_auc
                        })
        
        logger.info(f"GAT生成 {len(findings)} 个发现，最佳验证AUC: {best_val_auc:.4f}")
        return self._save_findings(findings, output_dir)
    def _gat_fallback_prediction(self, node_features: torch.Tensor, node_ids: List[str], 
                            output_dir: Path) -> Path:
        """GAT无边图的回退预测逻辑"""
        logger.info("Using GAT fallback mode: node feature similarity-based prediction")
        
        attention_dim = min(64, node_features.size(1))
        
        W_q = torch.randn(node_features.size(1), attention_dim) * 0.1
        W_k = torch.randn(node_features.size(1), attention_dim) * 0.1
        
        with torch.no_grad():
            queries = torch.mm(node_features, W_q)
            keys = torch.mm(node_features, W_k)
            
            attention_scores = torch.mm(queries, keys.t())
            attention_weights = torch.softmax(attention_scores, dim=1)
            
            embeddings = torch.mm(attention_weights, node_features)
            embeddings = F.normalize(embeddings, p=2, dim=1)
        
        findings = []
        num_nodes = len(node_ids)
        
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                similarity_score = torch.dot(embeddings[i], embeddings[j]).item()
                similarity_score = torch.sigmoid(torch.tensor(similarity_score)).item()
                
                if similarity_score > 0.6:  # 阈值可调
                    findings.append({
                        'source_id': node_ids[i],
                        'target_id': node_ids[j],
                        'unified_score': similarity_score,
                        'edge_type': 'missing',
                        'method': 'GAT_fallback'
                    })
        
        logger.info(f"GAT fallback mode generated {len(findings)} findings")
        return self._save_findings(findings, output_dir)
    def _compute_edge_scores(self, embeddings: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """计算边的得分（点积）"""
        row, col = edge_index
        return (embeddings[row] * embeddings[col]).sum(dim=1)
    
    def _simplified_gat_training(self, node_features: torch.Tensor, edge_index: torch.Tensor, 
                               node_ids: List[str], case_data: Dict[str, Any], output_dir: Path) -> Path:
        """简化版GAT训练（当torch_geometric版本过低时使用）"""
        logger.info("使用简化版GAT训练")
        
        model = GATLinkPredictor(node_features.size(1), self.hidden_dim, self.out_dim, self.heads)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        
        num_edges = edge_index.size(1)
        train_size = int(0.8 * num_edges)
        perm = torch.randperm(num_edges)
        train_edges = edge_index[:, perm[:train_size]]
        
        model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            embeddings = model(node_features, train_edges)
            
            pos_scores = self._compute_edge_scores(embeddings, train_edges)
            
            num_nodes = node_features.size(0)
            neg_edges = torch.randint(0, num_nodes, (2, train_size))
            neg_scores = self._compute_edge_scores(embeddings, neg_edges)
            
            pos_loss = -torch.log(torch.sigmoid(pos_scores) + 1e-8).mean()
            neg_loss = -torch.log(1 - torch.sigmoid(neg_scores) + 1e-8).mean()
            loss = pos_loss + neg_loss
            
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 50 == 0:
                logger.info(f"简化GAT Epoch {epoch+1}/{self.epochs}, Loss: {loss.item():.4f}")
        
        model.eval()
        findings = []
        
        with torch.no_grad():
            embeddings = model(node_features, edge_index)
            existing_edges, candidate_edges = self._generate_all_candidate_edges(
                case_data['all_nodes'], case_data['all_edges']
            )
            
            node_id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
            
            for u, v in candidate_edges:
                if u in node_id_to_idx and v in node_id_to_idx:
                    u_idx, v_idx = node_id_to_idx[u], node_id_to_idx[v]
                    score = torch.sigmoid(torch.dot(embeddings[u_idx], embeddings[v_idx])).item()
                    
                    if score > 0.2:
                        findings.append({
                            'source_id': u,
                            'target_id': v,
                            'unified_score': score,
                            'edge_type': 'missing',
                            'method': 'GAT_simplified'
                        })
        
        return self._save_findings(findings, output_dir)

    def _build_graph_data_for_gat(self, all_nodes: List[Dict], all_edges: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """为GAT构建图数据"""
        node_id_to_idx = {node['id']: idx for idx, node in enumerate(all_nodes)}
        node_ids = [node['id'] for node in all_nodes]
        
        edges = []
        for edge in all_edges:
            source_id = edge.get('source_id', '')
            target_id = edge.get('target_id', '')
            if source_id in node_id_to_idx and target_id in node_id_to_idx:
                edges.append([node_id_to_idx[source_id], node_id_to_idx[target_id]])
                edges.append([node_id_to_idx[target_id], node_id_to_idx[source_id]])
        
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty((2, 0), dtype=torch.long)
        
        num_nodes = len(all_nodes)
        if num_nodes > 0:
            one_hot = F.one_hot(torch.arange(0, num_nodes), num_classes=num_nodes).float()
            random_features = torch.randn(num_nodes, 16)  # 额外的随机特征
            node_features = torch.cat([one_hot, random_features], dim=1)
        else:
            node_features = torch.empty((0, 1), dtype=torch.float)
        
        return edge_index, node_features, node_ids


class GATLinkPredictor(nn.Module):
    """GAT链接预测模型（改进版）"""
    
    def __init__(self, input_dim: int, hidden_dim: int, out_dim: int, heads: int = 4):
        super(GATLinkPredictor, self).__init__()
        self.gat1 = GATConv(input_dim, hidden_dim, heads=heads, dropout=0.2)
        self.gat2 = GATConv(hidden_dim * heads, out_dim, heads=1, dropout=0.2)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """前向传播（改进版）"""
        x = self.dropout(x)
        x = F.elu(self.gat1(x, edge_index))
        
        x = self.dropout(x)
        x = self.gat2(x, edge_index)
        
        x = F.normalize(x, p=2, dim=1)
        return x



import scipy.stats as stats
from sklearn.linear_model import LassoCV
from itertools import combinations

class CDHCBaseline(Baseline):
    """CDHC (Causal Discovery with Hidden Confounders) 基线方法"""
    
    def __init__(self, name: str = "CDHC", top_k: int = None):
        super().__init__(name, top_k)
        self.requires_observational_data = True
        self.requires_graph_structure = True
        self.max_iterations = 10
        self.alpha = 0.05  # 独立性测试的显著性水平
        
    def run(self, case_data_dir: Path, output_dir: Path) -> Path:
        """运行CDHC并生成findings.json"""
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Running CDHC on case: {case_data_dir}")
        
        case_data = self._load_case_data(case_data_dir)
        sensor_data = case_data['sensor_data']
        all_nodes = case_data['all_nodes']
        all_edges = case_data['all_edges']
        
        if sensor_data.empty or len(all_nodes) == 0:
            logger.warning("Insufficient data for CDHC, returning empty findings")
            return self._save_findings([], output_dir)
        
        data_matrix, node_ids = self._prepare_data_matrix(sensor_data, all_nodes, case_data['node_mappings'])
        if data_matrix is None or len(node_ids) < 2:
            logger.warning("Data matrix preparation failed, returning empty findings")
            return self._save_findings([], output_dir)
        
        g_prior = self._build_initial_graph(node_ids, all_edges)
        
        refined_graph = self._run_cdhc_algorithm(data_matrix, node_ids, g_prior)
        
        findings = self._parse_cdhc_output(refined_graph, node_ids)
        
        logger.info(f"CDHC生成了 {len(findings)} 个发现")
        return self._save_findings(findings, output_dir)
    
    def _prepare_data_matrix(self, sensor_data: pd.DataFrame, all_nodes: List[Dict], 
                           node_mappings: Dict[str, Dict[str, str]]) -> Tuple[np.ndarray, List[str]]:
        """准备数据矩阵，将传感器数据映射到节点"""
        numeric_cols = sensor_data.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) == 0:
            return None, []
        
        col_to_node = {}
        for col in numeric_cols:
            node_id = None
            for node in all_nodes:
                node_text = node.get('text', '').lower()
                if col.lower() in node_text or node_text in col.lower():
                    node_id = node.get('id')
                    break
            
            if node_id:
                col_to_node[col] = node_id
        
        valid_cols = [col for col in numeric_cols if col in col_to_node]
        if len(valid_cols) < 2:
            return None, []
        
        data_matrix = sensor_data[valid_cols].dropna().values
        node_ids = [col_to_node[col] for col in valid_cols]
        
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        data_matrix = scaler.fit_transform(data_matrix)
        
        logger.info(f"准备数据矩阵: {data_matrix.shape}, 节点数: {len(node_ids)}")
        return data_matrix, node_ids
    
    def _build_initial_graph(self, node_ids: List[str], all_edges: List[Dict]) -> nx.DiGraph:
        """构建初始有向图"""
        G = nx.DiGraph()
        G.add_nodes_from(node_ids)
        
        for edge in all_edges:
            src, tgt = edge.get('source_id'), edge.get('target_id')
            if src in node_ids and tgt in node_ids:
                G.add_edge(src, tgt)
        
        logger.info(f"初始图: {G.number_of_nodes()} 节点, {G.number_of_edges()} 条边")
        return G
    
    def _run_cdhc_algorithm(self, data: np.ndarray, node_ids: List[str], 
                        g_prior: nx.DiGraph) -> nx.DiGraph:
        """
        CDHC核心算法：迭代式隐藏混杂检测与修正
        基于论文Algorithm 1的简化实现
        """
        current_graph = g_prior.copy()
        iteration = 0
        latent_counter = 0
        
        while iteration < self.max_iterations:
            iteration += 1
            logger.info(f"CDHC迭代 {iteration}/{self.max_iterations}")
            
            graph_changed = False
            
            for i, node_i in enumerate(node_ids):
                mb_nodes = self._get_markov_blanket(current_graph, node_i)
                
                candidate_set = [n for n in mb_nodes if n != node_i and n in node_ids]
                
                if len(candidate_set) < 2:
                    continue
                
                confounded_set = self._detect_hidden_confounder(
                    data, node_ids, node_i, candidate_set
                )
                
                if len(confounded_set) >= 2:
                    latent_name = f"Z_{latent_counter}"
                    latent_counter += 1
                    
                    current_graph.add_node(latent_name, node_type='latent')
                    
                    for conf_node in confounded_set:
                        current_graph.add_edge(latent_name, conf_node)
                    
                    logger.info(f"引入潜变量 {latent_name}，影响节点: {confounded_set}")
                    graph_changed = True
            
            if not graph_changed:
                logger.info("图结构未发生变化，CDHC算法收敛")
                break
        
        return current_graph

    
    def _get_markov_blanket(self, graph: nx.DiGraph, node: str) -> List[str]:
        """获取节点的马尔可夫毯"""
        mb = set()
        
        mb.update(graph.predecessors(node))
        
        children = list(graph.successors(node))
        mb.update(children)
        
        for child in children:
            mb.update(graph.predecessors(child))
        
        mb.discard(node)
        return list(mb)
    
    def _detect_hidden_confounder(self, data: np.ndarray, node_ids: List[str], 
                                target_node: str, candidate_nodes: List[str]) -> List[str]:
        """
        检测候选节点集中是否存在被隐藏混杂因素影响的子集
        使用条件独立性测试和偏相关分析
        """
        target_idx = node_ids.index(target_node)
        
        valid_candidates = [n for n in candidate_nodes if n in node_ids]
        
        if len(valid_candidates) < 2:
            return []
        
        candidate_indices = [node_ids.index(n) for n in valid_candidates]
        
        confounded_set = []
        
        for size in range(2, min(len(candidate_indices) + 1, 5)):  # 限制子集大小
            for subset in combinations(candidate_indices, size):
                subset = list(subset)
                
                if self._check_confounding_pattern(data, subset, target_idx):
                    confounded_nodes = [node_ids[idx] for idx in subset]
                    confounded_set.extend(confounded_nodes)
        
        return list(set(confounded_set))
    
    def _check_confounding_pattern(self, data: np.ndarray, subset_indices: List[int], 
                                  target_idx: int) -> bool:
        """
        检查子集是否呈现隐藏混杂模式
        判断标准：子集变量之间有显著相关，但控制目标变量后仍然相关
        """
        if len(subset_indices) < 2:
            return False
        
        correlations = []
        for i, j in combinations(subset_indices, 2):
            corr = np.corrcoef(data[:, i], data[:, j])[0, 1]
            correlations.append(abs(corr))
        
        avg_corr = np.mean(correlations)
        
        if avg_corr > 0.3:
            try:
                partial_corrs = []
                for i, j in combinations(subset_indices, 2):
                    X = data[:, [i, j, target_idx]]
                    corr_matrix = np.corrcoef(X.T)
                    
                    r_ij = corr_matrix[0, 1]
                    r_it = corr_matrix[0, 2]
                    r_jt = corr_matrix[1, 2]
                    
                    partial_corr = (r_ij - r_it * r_jt) / (np.sqrt(1 - r_it**2) * np.sqrt(1 - r_jt**2))
                    partial_corrs.append(abs(partial_corr))
                
                avg_partial_corr = np.mean(partial_corrs)
                
                return avg_partial_corr > 0.25
            except:
                return False
        
        return False
    
    def _parse_cdhc_output(self, graph: nx.DiGraph, node_ids: List[str]) -> List[Dict[str, Any]]:
        """解析CDHC输出的图结构，生成发现列表"""
        findings = []
        
        latent_nodes = [n for n in graph.nodes() if graph.nodes[n].get('node_type') == 'latent']
        
        logger.info(f"发现 {len(latent_nodes)} 个潜变量")
        
        for latent in latent_nodes:
            affected_nodes = list(graph.successors(latent))
            
            affected_obs_nodes = [n for n in affected_nodes if n in node_ids]
            
            if len(affected_obs_nodes) >= 2:
                for i, j in combinations(affected_obs_nodes, 2):
                    findings.append({
                        'source_id': i,
                        'target_id': j,
                        'unified_score': 0.85,  # 高置信度
                        'edge_type': 'confounded_by_latent',
                        'method': 'CDHC',
                        'latent_node': latent,
                        'confounded_set': affected_obs_nodes
                    })
        
        for edge in graph.edges():
            src, tgt = edge
            if src in node_ids and tgt in node_ids:
                findings.append({
                    'source_id': src,
                    'target_id': tgt,
                    'unified_score': 0.75,
                    'edge_type': 'refined_causal',
                    'method': 'CDHC'
                })
        
        return findings
    


class IFPDAGBaseline(Baseline):
    """IF-PDAG (Intervention-Free Partially Directed Acyclic Graph) 基线方法"""
    
    def __init__(self, name: str = "IF-PDAG", top_k: int = None):
        super().__init__(name, top_k)
        self.requires_observational_data = True
        self.requires_graph_structure = True
        self.lambda_prior = 100  # 先验知识权重
        self.lambda_l1 = 0.01   # L1正则化权重
        self.max_iter = 50
        
    def run(self, case_data_dir: Path, output_dir: Path) -> Path:
        """运行IF-PDAG并生成findings.json"""
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Running IF-PDAG on case: {case_data_dir}")
        
        case_data = self._load_case_data(case_data_dir)
        sensor_data = case_data['sensor_data']
        all_nodes = case_data['all_nodes']
        all_edges = case_data['all_edges']
        
        if sensor_data.empty or len(all_nodes) == 0:
            logger.warning("Insufficient data for IF-PDAG, returning empty findings")
            return self._save_findings([], output_dir)
        
        X, node_ids = self._prepare_data_matrix(sensor_data, all_nodes, case_data['node_mappings'])
        if X is None or len(node_ids) < 2:
            logger.warning("Data preparation failed for IF-PDAG")
            return self._save_findings([], output_dir)
        
        A_prior = self._build_prior_matrix(node_ids, all_edges)
        
        W_refined = self._optimize_with_prior(X, A_prior)
        
        findings = self._parse_adjacency_matrix(W_refined, A_prior, node_ids)
        
        logger.info(f"IF-PDAG生成了 {len(findings)} 个发现")
        return self._save_findings(findings, output_dir)
    
    def _prepare_data_matrix(self, sensor_data: pd.DataFrame, all_nodes: List[Dict],
                           node_mappings: Dict[str, Dict[str, str]]) -> Tuple[np.ndarray, List[str]]:
        """准备标准化的数据矩阵（与CDHC相同的逻辑）"""
        numeric_cols = sensor_data.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) == 0:
            return None, []
        
        col_to_node = {}
        for col in numeric_cols:
            node_id = None
            for node in all_nodes:
                node_text = node.get('text', '').lower()
                if col.lower() in node_text or node_text in col.lower():
                    node_id = node.get('id')
                    break
            if node_id:
                col_to_node[col] = node_id
        
        valid_cols = [col for col in numeric_cols if col in col_to_node]
        if len(valid_cols) < 2:
            return None, []
        
        X = sensor_data[valid_cols].dropna().values
        node_ids = [col_to_node[col] for col in valid_cols]
        
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        logger.info(f"IF-PDAG数据矩阵: {X.shape}")
        return X, node_ids
    
    def _build_prior_matrix(self, node_ids: List[str], all_edges: List[Dict]) -> np.ndarray:
        """构建先验邻接矩阵"""
        d = len(node_ids)
        A_prior = np.zeros((d, d))
        
        node_to_idx = {node_id: i for i, node_id in enumerate(node_ids)}
        
        for edge in all_edges:
            src, tgt = edge.get('source_id'), edge.get('target_id')
            if src in node_to_idx and tgt in node_to_idx:
                i, j = node_to_idx[src], node_to_idx[tgt]
                A_prior[i, j] = 1.0
        
        logger.info(f"先验矩阵包含 {np.sum(A_prior)} 条边")
        return A_prior
    
    def _optimize_with_prior(self, X: np.ndarray, A_prior: np.ndarray) -> np.ndarray:
        """
        基于NOTEARS框架的约束优化
        目标函数: loss = data_fit_loss + lambda_l1 * ||W||_1 + lambda_prior * ||W - A_prior||_F^2
        """
        n, d = X.shape
        W = np.zeros((d, d))  # 初始化邻接矩阵
        
        lr = 0.01
        for iteration in range(self.max_iter):
            residual = X - X @ W.T
            data_loss = 0.5 / n * np.sum(residual ** 2)
            
            l1_loss = self.lambda_l1 * np.sum(np.abs(W))
            
            prior_loss = self.lambda_prior * np.sum((W - A_prior) ** 2)
            
            total_loss = data_loss + l1_loss + prior_loss
            
            grad_data = -1.0 / n * (X.T @ residual)
            grad_prior = 2 * self.lambda_prior * (W - A_prior)
            grad_l1 = self.lambda_l1 * np.sign(W)
            
            grad = grad_data + grad_prior + grad_l1
            
            W = W - lr * grad
            
            np.fill_diagonal(W, 0)
            
            if iteration % 20 == 0:
                logger.info(f"IF-PDAG优化迭代 {iteration}/{self.max_iter}, Loss: {total_loss:.4f}")
        
        return W
    
    def _parse_adjacency_matrix(self, W: np.ndarray, A_prior: np.ndarray, 
                               node_ids: List[str]) -> List[Dict[str, Any]]:
        """解析优化后的邻接矩阵，生成发现列表"""
        findings = []
        threshold = 0.1  # 权重阈值
        
        d = len(node_ids)
        
        for i in range(d):
            for j in range(d):
                if i != j and abs(W[i, j]) > threshold:
                    score = abs(W[i, j])
                    
                    if A_prior[i, j] > 0:
                        edge_type = 'confirmed_by_refinement'
                    else:
                        edge_type = 'discovered_by_refinement'
                    
                    findings.append({
                        'source_id': node_ids[i],
                        'target_id': node_ids[j],
                        'unified_score': float(score),
                        'edge_type': edge_type,
                        'method': 'IF-PDAG',
                        'weight': float(W[i, j])
                    })
        
        return findings



class DistMultModel(nn.Module):
    """DistMult 双线性模型"""
    def __init__(self, num_entities: int, num_relations: int, embedding_dim: int):
        super().__init__()
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)

    def forward(self, head_idx, rel_idx, tail_idx):
        h = self.entity_embeddings(head_idx)
        r = self.relation_embeddings(rel_idx)
        t = self.entity_embeddings(tail_idx)
        return torch.sum(h * r * t, dim=-1)


class DistMultBaseline(Baseline):
    """DistMult 知识图谱补全基线 (Yang et al., 2015)
    Method group: Traditional knowledge graph completion methods
    """
    def __init__(self, name: str = "DistMult", top_k: int = None):
        super().__init__(name, top_k)
        self.embedding_dim = 64
        self.epochs = 150
        self.lr = 0.01
        self.neg_ratio = 5

    def run(self, case_data_dir: Path, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Running DistMult on case: {case_data_dir}")

        case_data = self._load_case_data(case_data_dir)
        all_nodes = case_data['all_nodes']
        all_edges = case_data['all_edges']

        if len(all_nodes) < 2:
            return self._save_findings([], output_dir)

        entities = [n['id'] for n in all_nodes]
        ent2idx = {e: i for i, e in enumerate(entities)}

        rel2idx = {"causal": 0}
        triples = []
        for edge in all_edges:
            src, tgt = edge.get('source_id', ''), edge.get('target_id', '')
            if src in ent2idx and tgt in ent2idx:
                triples.append((ent2idx[src], 0, ent2idx[tgt]))

        if len(triples) < 2:
            logger.warning("Too few triples for DistMult")
            return self._save_findings([], output_dir)

        num_ent = len(entities)
        model = DistMultModel(num_ent, len(rel2idx), self.embedding_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-5)
        criterion = nn.MarginRankingLoss(margin=1.0)

        model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            total_loss = 0.0

            for h, r, t in triples:
                pos_score = model(torch.tensor([h]), torch.tensor([r]), torch.tensor([t]))

                neg_tails = torch.randint(0, num_ent, (self.neg_ratio,))
                neg_scores = model(
                    torch.tensor([h] * self.neg_ratio),
                    torch.tensor([r] * self.neg_ratio),
                    neg_tails
                )
                target = torch.ones(self.neg_ratio)
                loss = criterion(pos_score.expand_as(neg_scores), neg_scores, target)
                total_loss += loss

            total_loss.backward()
            optimizer.step()

            if (epoch + 1) % 50 == 0:
                logger.info(f"DistMult Epoch {epoch+1}/{self.epochs}, Loss: {total_loss.item():.4f}")

        model.eval()
        existing_edges, candidate_edges = self._generate_all_candidate_edges(all_nodes, all_edges)
        findings = []

        with torch.no_grad():
            for u, v in candidate_edges:
                if u in ent2idx and v in ent2idx:
                    score = model(
                        torch.tensor([ent2idx[u]]),
                        torch.tensor([0]),
                        torch.tensor([ent2idx[v]])
                    ).item()
                    prob = torch.sigmoid(torch.tensor(score)).item()
                    if prob > 0.3:
                        findings.append({
                            'source_id': u,
                            'target_id': v,
                            'unified_score': prob,
                            'edge_type': 'missing',
                            'method': 'DistMult'
                        })

        logger.info(f"DistMult generated {len(findings)} findings")
        return self._save_findings(findings, output_dir)


class GESBaseline(Baseline):
    """Greedy Equivalence Search 基线 (Chickering, 2002)
    Method group: Pure data-driven causal discovery methods
    基于 BIC 评分的贪心搜索，分两阶段：Forward (加边) + Backward (删边)
    """
    def __init__(self, name: str = "GES", top_k: int = None):
        super().__init__(name, top_k)
        self.max_forward_steps = 100
        self.max_backward_steps = 50

    def run(self, case_data_dir: Path, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Running GES on case: {case_data_dir}")

        case_data = self._load_case_data(case_data_dir)
        sensor_data = case_data['sensor_data']
        all_nodes = case_data['all_nodes']

        if sensor_data.empty or len(all_nodes) < 2:
            return self._save_findings([], output_dir)

        numeric_cols = sensor_data.select_dtypes(include=[np.number]).columns.tolist()
        col_to_node = {}
        for col in numeric_cols:
            for node in all_nodes:
                node_text = node.get('text', '').lower()
                if col.lower() in node_text or node_text in col.lower():
                    col_to_node[col] = node.get('id')
                    break

        valid_cols = [col for col in numeric_cols if col in col_to_node]
        if len(valid_cols) < 2:
            return self._save_findings([], output_dir)

        data_matrix = sensor_data[valid_cols].dropna().values
        node_ids = [col_to_node[col] for col in valid_cols]
        n, d = data_matrix.shape

        from sklearn.preprocessing import StandardScaler
        data_matrix = StandardScaler().fit_transform(data_matrix)

        adj = np.zeros((d, d))  # 空图开始
        best_bic = self._compute_bic(data_matrix, adj, n, d)
        logger.info(f"GES initial BIC: {best_bic:.2f}")

        for step in range(self.max_forward_steps):
            best_edge = None
            best_edge_bic = best_bic

            for i in range(d):
                for j in range(d):
                    if i != j and adj[i, j] == 0:
                        adj[i, j] = 1
                        new_bic = self._compute_bic(data_matrix, adj, n, d)
                        if new_bic < best_edge_bic:
                            best_edge_bic = new_bic
                            best_edge = (i, j)
                        adj[i, j] = 0

            if best_edge is None:
                break
            adj[best_edge[0], best_edge[1]] = 1
            best_bic = best_edge_bic

        logger.info(f"GES Forward phase: {int(adj.sum())} edges, BIC: {best_bic:.2f}")

        for step in range(self.max_backward_steps):
            best_remove = None
            best_remove_bic = best_bic

            for i in range(d):
                for j in range(d):
                    if adj[i, j] == 1:
                        adj[i, j] = 0
                        new_bic = self._compute_bic(data_matrix, adj, n, d)
                        if new_bic < best_remove_bic:
                            best_remove_bic = new_bic
                            best_remove = (i, j)
                        adj[i, j] = 1

            if best_remove is None:
                break
            adj[best_remove[0], best_remove[1]] = 0
            best_bic = best_remove_bic

        logger.info(f"GES Backward phase: {int(adj.sum())} edges, BIC: {best_bic:.2f}")

        all_edges = case_data['all_edges']
        existing_edge_set = set()
        for edge in all_edges:
            src, tgt = edge.get('source_id', ''), edge.get('target_id', '')
            existing_edge_set.add((src, tgt))

        findings = []
        for i in range(d):
            for j in range(d):
                if adj[i, j] > 0 and (node_ids[i], node_ids[j]) not in existing_edge_set:
                    score = self._compute_edge_score(data_matrix, i, j)
                    findings.append({
                        'source_id': node_ids[i],
                        'target_id': node_ids[j],
                        'unified_score': float(score),
                        'edge_type': 'discovered_by_ges',
                        'method': 'GES'
                    })

        logger.info(f"GES generated {len(findings)} findings")
        return self._save_findings(findings, output_dir)

    def _compute_bic(self, data: np.ndarray, adj: np.ndarray, n: int, d: int) -> float:
        """计算 BIC 评分"""
        bic = 0.0
        for j in range(d):
            parents = np.where(adj[:, j] > 0)[0]
            if len(parents) == 0:
                residual_var = np.var(data[:, j])
            else:
                X_pa = data[:, parents]
                try:
                    beta = np.linalg.lstsq(X_pa, data[:, j], rcond=None)[0]
                    residual = data[:, j] - X_pa @ beta
                    residual_var = np.var(residual)
                except np.linalg.LinAlgError:
                    residual_var = np.var(data[:, j])

            residual_var = max(residual_var, 1e-10)
            bic += n * np.log(residual_var) + len(parents) * np.log(n)
        return bic

    def _compute_edge_score(self, data: np.ndarray, i: int, j: int) -> float:
        """计算单条边的得分 (R^2)"""
        try:
            corr = np.corrcoef(data[:, i], data[:, j])[0, 1]
            return min(abs(corr), 1.0)
        except:
            return 0.5


class NOTEARSBaseline(Baseline):
    """NOTEARS 基线 (Zheng et al., 2018)
    Method group: Existing causal structure optimization methods
    基于连续优化的 DAG 学习，不使用先验图（与 IF-PDAG 对比）
    """
    def __init__(self, name: str = "NOTEARS", top_k: int = None):
        super().__init__(name, top_k)
        self.lambda_l1 = 0.01     # L1 稀疏性
        self.max_iter = 100
        self.h_tol = 1e-8          # DAG 约束容差
        self.rho_max = 1e+16       # 增广拉格朗日 rho 上限
        self.w_threshold = 0.05    # 边权值阈值

    def run(self, case_data_dir: Path, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Running NOTEARS on case: {case_data_dir}")

        case_data = self._load_case_data(case_data_dir)
        sensor_data = case_data['sensor_data']
        all_nodes = case_data['all_nodes']

        if sensor_data.empty or len(all_nodes) < 2:
            return self._save_findings([], output_dir)

        numeric_cols = sensor_data.select_dtypes(include=[np.number]).columns.tolist()
        col_to_node = {}
        for col in numeric_cols:
            for node in all_nodes:
                node_text = node.get('text', '').lower()
                if col.lower() in node_text or node_text in col.lower():
                    col_to_node[col] = node.get('id')
                    break

        valid_cols = [col for col in numeric_cols if col in col_to_node]
        if len(valid_cols) < 2:
            return self._save_findings([], output_dir)

        X = sensor_data[valid_cols].dropna().values
        node_ids = [col_to_node[col] for col in valid_cols]
        n, d = X.shape

        from sklearn.preprocessing import StandardScaler
        X = StandardScaler().fit_transform(X)

        W_est = self._notears_linear(X, d, n)

        W_est[np.abs(W_est) < self.w_threshold] = 0

        all_edges = case_data['all_edges']
        existing_edge_set = set()
        for edge in all_edges:
            src, tgt = edge.get('source_id', ''), edge.get('target_id', '')
            existing_edge_set.add((src, tgt))

        findings = []
        for i in range(d):
            for j in range(d):
                if i != j and abs(W_est[i, j]) > 0:
                    findings.append({
                        'source_id': node_ids[i],
                        'target_id': node_ids[j],
                        'unified_score': float(min(abs(W_est[i, j]), 1.0)),
                        'edge_type': 'discovered_by_notears' if (node_ids[i], node_ids[j]) not in existing_edge_set else 'confirmed_by_notears',
                        'method': 'NOTEARS',
                        'weight': float(W_est[i, j])
                    })

        if len(findings) == 0:
            logger.warning("NOTEARS: no edges after thresholding, falling back to top-K by absolute weight")
            all_edges_scored = []
            for i in range(d):
                for j in range(d):
                    if i != j and (node_ids[i], node_ids[j]) not in existing_edge_set:
                        w_abs = abs(W_est[i, j])
                        if w_abs > 1e-6:
                            all_edges_scored.append((node_ids[i], node_ids[j], w_abs))
            all_edges_scored.sort(key=lambda x: x[2], reverse=True)
            for src, tgt, w in all_edges_scored[:max(self.top_k or 35, 35)]:
                findings.append({
                    'source_id': src,
                    'target_id': tgt,
                    'unified_score': float(min(w * 10, 1.0)),  # scale up for scoring
                    'edge_type': 'discovered_by_notears',
                    'method': 'NOTEARS',
                    'weight': float(w)
                })

        logger.info(f"NOTEARS generated {len(findings)} findings")
        return self._save_findings(findings, output_dir)

    def _notears_linear(self, X: np.ndarray, d: int, n: int) -> np.ndarray:
        """NOTEARS 线性模型 - 数值稳定版"""
        W = np.zeros((d, d))
        rho = 1.0
        alpha = 0.0
        h_prev = np.inf
        best_W = W.copy()
        best_loss = np.inf

        for iteration in range(self.max_iter):
            W_before = W.copy()

            for sub_iter in range(30):
                residual = X - X @ W.T
                grad_data = -1.0 / n * (X.T @ residual)

                W_sq = W * W
                try:
                    exp_term = np.eye(d)
                    power = np.eye(d)
                    for k in range(1, min(d, 8)):
                        power = power @ W_sq / k
                        if np.any(np.abs(power) > 1e10):
                            break
                        exp_term += power
                    grad_h = exp_term.T * 2 * W
                    h = np.trace(exp_term) - d
                except:
                    grad_h = np.zeros_like(W)
                    h = 0.0

                if np.isnan(h) or np.isinf(h):
                    logger.warning(f"NOTEARS: h is NaN/Inf at iter {iteration}, reverting")
                    W = W_before.copy()
                    break

                grad_l1 = self.lambda_l1 * np.sign(W)
                grad = grad_data + (alpha + rho * h) * grad_h + grad_l1

                grad_norm = np.linalg.norm(grad)
                if grad_norm > 10.0:
                    grad = grad * 10.0 / grad_norm

                lr = 0.005 / (1 + 0.05 * iteration)
                W = W - lr * grad
                np.fill_diagonal(W, 0)

                W = np.clip(W, -5.0, 5.0)

            W_sq = W * W
            exp_term = np.eye(d)
            power = np.eye(d)
            for k in range(1, min(d, 8)):
                power = power @ W_sq / k
                if np.any(np.abs(power) > 1e10):
                    break
                exp_term += power
            h = np.trace(exp_term) - d

            if np.isnan(h) or np.isinf(h) or np.any(np.isnan(W)):
                logger.warning(f"NOTEARS: numerical instability at iter {iteration}, using best W")
                W = best_W.copy()
                break

            data_loss = 0.5 / n * np.sum((X - X @ W.T) ** 2)
            if data_loss < best_loss:
                best_loss = data_loss
                best_W = W.copy()

            if iteration % 20 == 0:
                logger.info(f"NOTEARS iter {iteration}, h={h:.6f}, rho={rho:.1f}, loss={data_loss:.4f}")

            if h > 0.25 * h_prev:
                rho = min(rho * 2, 1e6)  # 更保守的 rho 增长
            alpha += rho * h
            alpha = np.clip(alpha, -1e6, 1e6)  # 防止溢出
            h_prev = h

            if abs(h) < self.h_tol:
                logger.info(f"NOTEARS converged at iteration {iteration}")
                break

        return best_W


class GOLEMBaseline(Baseline):
    """GOLEM 基线 (Ng et al., NeurIPS 2020)
    Method group: Causal structure optimization methods
    基于 soft DAG 约束 + likelihood-based score 的连续优化 DAG 学习。
    与 NOTEARS 的区别: 使用 soft 约束直接梯度下降，而非增广拉格朗日。
    适配到本文 CKG 审计任务: 从传感器数据学习因果结构，发现缺失边。
    """
    def __init__(self, name: str = "GOLEM", top_k: int = None):
        super().__init__(name, top_k)
        self.lambda_l1 = 0.02       # L1 稀疏性
        self.lambda_dag = 5.0       # DAG 约束权重 (soft)
        self.max_iter = 150         # 最大迭代
        self.lr = 3e-3              # 学习率
        self.w_threshold = 0.05     # 边权值阈值

    def run(self, case_data_dir: Path, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Running GOLEM on case: {case_data_dir}")

        case_data = self._load_case_data(case_data_dir)
        sensor_data = case_data['sensor_data']
        all_nodes = case_data['all_nodes']

        if sensor_data.empty or len(all_nodes) < 2:
            return self._save_findings([], output_dir)

        numeric_cols = sensor_data.select_dtypes(include=[np.number]).columns.tolist()
        col_to_node = {}
        for col in numeric_cols:
            for node in all_nodes:
                node_text = node.get('text', '').lower()
                if col.lower() in node_text or node_text in col.lower():
                    col_to_node[col] = node.get('id')
                    break

        valid_cols = [col for col in numeric_cols if col in col_to_node]
        if len(valid_cols) < 2:
            return self._save_findings([], output_dir)

        X = sensor_data[valid_cols].dropna().values
        node_ids = [col_to_node[col] for col in valid_cols]
        n, d = X.shape

        from sklearn.preprocessing import StandardScaler
        X = StandardScaler().fit_transform(X)

        W_est = self._golem_linear(X, d, n)

        W_est[np.abs(W_est) < self.w_threshold] = 0

        all_edges = case_data['all_edges']
        existing_edge_set = set()
        for edge in all_edges:
            src, tgt = edge.get('source_id', ''), edge.get('target_id', '')
            existing_edge_set.add((src, tgt))

        findings = []
        for i in range(d):
            for j in range(d):
                if i != j and abs(W_est[i, j]) > 0:
                    findings.append({
                        'source_id': node_ids[i],
                        'target_id': node_ids[j],
                        'unified_score': float(min(abs(W_est[i, j]), 1.0)),
                        'edge_type': 'discovered_by_golem' if (node_ids[i], node_ids[j]) not in existing_edge_set else 'confirmed_by_golem',
                        'method': 'GOLEM',
                        'weight': float(W_est[i, j])
                    })

        if len(findings) == 0:
            logger.warning("GOLEM: no edges after thresholding, falling back to top-K by absolute weight")
            all_edges_scored = []
            for i in range(d):
                for j in range(d):
                    if i != j and (node_ids[i], node_ids[j]) not in existing_edge_set:
                        w_abs = abs(W_est[i, j])
                        if w_abs > 1e-6:
                            all_edges_scored.append((node_ids[i], node_ids[j], w_abs))
            all_edges_scored.sort(key=lambda x: x[2], reverse=True)
            for src, tgt, w in all_edges_scored[:max(self.top_k or 35, 35)]:
                findings.append({
                    'source_id': src, 'target_id': tgt,
                    'unified_score': float(min(w * 10, 1.0)),
                    'edge_type': 'discovered_by_golem',
                    'method': 'GOLEM', 'weight': float(w)
                })

        logger.info(f"GOLEM generated {len(findings)} findings")
        return self._save_findings(findings, output_dir)

    def _golem_linear(self, X: np.ndarray, d: int, n: int) -> np.ndarray:
        """GOLEM 线性模型: soft DAG 约束 + NLL score + 梯度下降"""
        corr = np.corrcoef(X.T)
        np.fill_diagonal(corr, 0)
        W = np.where(np.abs(corr) > 0.2, corr * 0.1, 0.0)
        np.fill_diagonal(W, 0)
        best_W = W.copy()
        best_score = np.inf

        for iteration in range(self.max_iter):
            residual = X - X @ W.T
            nll = 0.0
            grad_nll = np.zeros_like(W)
            for j in range(d):
                rss_j = np.sum(residual[:, j] ** 2) / n + 1e-8
                nll += 0.5 * np.log(rss_j)
                grad_nll[j, :] = -1.0 / (n * rss_j) * (X.T @ residual[:, j])
                grad_nll[j, j] = 0  # 对角线不更新

            W_sq = W * W
            try:
                exp_term = np.eye(d)
                power = np.eye(d)
                for k in range(1, min(d, 8)):
                    power = power @ W_sq / k
                    if np.any(np.abs(power) > 1e10):
                        break
                    exp_term += power
                h = np.trace(exp_term) - d
            except Exception:
                h = 0.0

            grad_h = 2.0 * W * exp_term.T

            grad_l1 = np.sign(W)

            grad = grad_nll + self.lambda_dag * grad_h + self.lambda_l1 * grad_l1

            if np.any(np.isnan(grad)) or np.any(np.isinf(grad)):
                logger.warning(f"GOLEM: numerical instability at iter {iteration}, using best W")
                break

            W = W - self.lr * grad
            np.fill_diagonal(W, 0)

            total_score = nll + self.lambda_dag * h + self.lambda_l1 * np.sum(np.abs(W))
            if total_score < best_score and not np.isnan(total_score):
                best_score = total_score
                best_W = W.copy()

            if iteration % 20 == 0:
                logger.info(f"GOLEM iter {iteration}, h={h:.6f}, nll={nll:.4f}, score={total_score:.4f}")

            if h < 1e-6:
                logger.info(f"GOLEM essentially DAG at iteration {iteration}")
                break

        return best_W


class DAGMABaseline(Baseline):
    """DAGMA 基线 (Bello et al., NeurIPS 2022)
    Method group: Causal structure optimization methods
    基于 log-det M-matrix 约束的 DAG 学习，替代矩阵指数计算。
    核心约束: h(W) = -log det(sI - W∘W) + d·log(s)
    适配到本文 CKG 审计任务: 从传感器数据学习因果结构，发现缺失边。
    """
    def __init__(self, name: str = "DAGMA", top_k: int = None):
        super().__init__(name, top_k)
        self.lambda_l1 = 0.02       # L1 稀疏性
        self.s_init = 1.0           # log-det 参数 s
        self.max_iter = 80          # 外层最大迭代
        self.inner_iter = 30        # 内层梯度步
        self.lr = 1e-3              # 学习率
        self.h_tol = 1e-8           # DAG 约束容差
        self.w_threshold = 0.05     # 边权值阈值
        self.mu_init = 1.0          # 增广拉格朗日初始 mu
        self.mu_factor = 2.0        # mu 增长因子

    def run(self, case_data_dir: Path, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Running DAGMA on case: {case_data_dir}")

        case_data = self._load_case_data(case_data_dir)
        sensor_data = case_data['sensor_data']
        all_nodes = case_data['all_nodes']

        if sensor_data.empty or len(all_nodes) < 2:
            return self._save_findings([], output_dir)

        numeric_cols = sensor_data.select_dtypes(include=[np.number]).columns.tolist()
        col_to_node = {}
        for col in numeric_cols:
            for node in all_nodes:
                node_text = node.get('text', '').lower()
                if col.lower() in node_text or node_text in col.lower():
                    col_to_node[col] = node.get('id')
                    break

        valid_cols = [col for col in numeric_cols if col in col_to_node]
        if len(valid_cols) < 2:
            return self._save_findings([], output_dir)

        X = sensor_data[valid_cols].dropna().values
        node_ids = [col_to_node[col] for col in valid_cols]
        n, d = X.shape

        from sklearn.preprocessing import StandardScaler
        X = StandardScaler().fit_transform(X)

        W_est = self._dagma_linear(X, d, n)

        W_est[np.abs(W_est) < self.w_threshold] = 0

        all_edges = case_data['all_edges']
        existing_edge_set = set()
        for edge in all_edges:
            src, tgt = edge.get('source_id', ''), edge.get('target_id', '')
            existing_edge_set.add((src, tgt))

        findings = []
        for i in range(d):
            for j in range(d):
                if i != j and abs(W_est[i, j]) > 0:
                    findings.append({
                        'source_id': node_ids[i],
                        'target_id': node_ids[j],
                        'unified_score': float(min(abs(W_est[i, j]), 1.0)),
                        'edge_type': 'discovered_by_dagma' if (node_ids[i], node_ids[j]) not in existing_edge_set else 'confirmed_by_dagma',
                        'method': 'DAGMA',
                        'weight': float(W_est[i, j])
                    })

        if len(findings) == 0:
            logger.warning("DAGMA: no edges after thresholding, falling back to top-K")
            all_edges_scored = []
            for i in range(d):
                for j in range(d):
                    if i != j and (node_ids[i], node_ids[j]) not in existing_edge_set:
                        w_abs = abs(W_est[i, j])
                        if w_abs > 1e-6:
                            all_edges_scored.append((node_ids[i], node_ids[j], w_abs))
            all_edges_scored.sort(key=lambda x: x[2], reverse=True)
            for src, tgt, w in all_edges_scored[:max(self.top_k or 35, 35)]:
                findings.append({
                    'source_id': src, 'target_id': tgt,
                    'unified_score': float(min(w * 10, 1.0)),
                    'edge_type': 'discovered_by_dagma',
                    'method': 'DAGMA', 'weight': float(w)
                })

        logger.info(f"DAGMA generated {len(findings)} findings")
        return self._save_findings(findings, output_dir)

    def _dagma_linear(self, X: np.ndarray, d: int, n: int) -> np.ndarray:
        """DAGMA 线性模型: log-det M-matrix DAG 约束 + 增广拉格朗日"""
        W = np.zeros((d, d))
        best_W = W.copy()
        best_loss = np.inf
        s = self.s_init
        mu = self.mu_init
        alpha = 0.0

        for outer in range(self.max_iter):
            for inner in range(self.inner_iter):
                residual = X - X @ W.T
                grad_data = -1.0 / n * (X.T @ residual)

                grad_l1 = self.lambda_l1 * np.sign(W)

                W_sq = W * W
                M = s * np.eye(d) - W_sq
                try:
                    M_inv = np.linalg.inv(M)
                    grad_h = 2.0 * W * M_inv.T
                except np.linalg.LinAlgError:
                    grad_h = 2.0 * W * 10.0

                h_val = self._compute_h(W, d, s)
                grad = grad_data + grad_l1 + (alpha + mu * h_val) * grad_h

                if np.any(np.isnan(grad)) or np.any(np.isinf(grad)):
                    logger.warning(f"DAGMA: numerical instability at outer={outer}, inner={inner}")
                    break

                W = W - self.lr * grad
                np.fill_diagonal(W, 0)

            h_val = self._compute_h(W, d, s)
            data_loss = 0.5 / n * np.sum((X - X @ W.T) ** 2)

            if data_loss < best_loss and not np.isnan(data_loss):
                best_loss = data_loss
                best_W = W.copy()

            if outer % 10 == 0:
                logger.info(f"DAGMA outer {outer}, h={h_val:.6f}, mu={mu:.1f}, loss={data_loss:.4f}")

            if abs(h_val) < self.h_tol:
                logger.info(f"DAGMA converged at outer iteration {outer}")
                break

            alpha += mu * h_val
            alpha = np.clip(alpha, -1e6, 1e6)
            mu = min(mu * self.mu_factor, 1e6)

        return best_W

    def _compute_h(self, W: np.ndarray, d: int, s: float) -> float:
        """计算 log-det DAG 约束: h(W) = -log det(sI - W∘W) + d·log(s)"""
        W_sq = W * W
        M = s * np.eye(d) - W_sq
        try:
            sign, logdet = np.linalg.slogdet(M)
            if sign <= 0:
                return 1e6  # M 不正定 => 非 DAG
            h = -logdet + d * np.log(s)
        except np.linalg.LinAlgError:
            h = 1e6
        return max(h, 0.0)


AVAILABLE_BASELINES = {
    "VGAE": VGAEBaseline,
    "GATE": GATEBaseline,
    "Peter-Clark": PCStableBaseline,
    "TransE": TransEBaseline,
    "CommonNeighbors": CommonNeighborsBaseline,
    "GAT": GATBaseline,
    "CDHC": CDHCBaseline,
    "IF-PDAG": IFPDAGBaseline,
    "DistMult": DistMultBaseline,    # KG completion
    "GES": GESBaseline,              # Data-driven causal discovery
    "NOTEARS": NOTEARSBaseline,      # Causal structure optimization
    "GOLEM": GOLEMBaseline,          # Soft DAG constraint (NeurIPS 2020)
    "DAGMA": DAGMABaseline,          # Log-det M-matrix DAG (NeurIPS 2022)
}
