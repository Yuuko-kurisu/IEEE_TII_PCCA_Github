#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BCSA_02_PCGATE.py
BCSA假设条件化PC-GATE模块
负责：基于假设进行条件化训练和不确定性聚合（使用图注意力自编码器）
职责：单一职责 - 假设条件化的审计分析（非变分版本）
"""

import os
import sys
import numpy as np
import pandas as pd
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from collections import defaultdict
from BCSA_00_Shared_Structures import (
    Hypothesis, EdgeUncertaintyResult, AggregatedUncertaintyResult,
    HypothesisPrompt, TrainingResult, convert_numpy_types
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.chdir(os.path.dirname(os.path.abspath(__file__)))

TORCH_AVAILABLE = False
SENTENCE_TRANSFORMERS_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GATv2Conv
    from torch_geometric.data import Data
    from torch_geometric.utils import negative_sampling
    TORCH_AVAILABLE = True
    logger.info("✓ PyTorch和torch_geometric可用")
except ImportError as e:
    logger.warning(f"⚠️ PyTorch不可用，将使用简化模式: {e}")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
    logger.info("✓ SentenceTransformers可用")
except ImportError as e:
    logger.warning(f"⚠️ SentenceTransformers不可用，文本嵌入功能降级: {e}")

try:
    import importlib
    bcsa_module = importlib.import_module('BCSA_02_PC_VGAE_Base')
    DataProcessor = bcsa_module.DataProcessor
    EnhancedFeatureExtractor = bcsa_module.EnhancedFeatureExtractor
    logger.info("✓ 成功导入基础PC-VGAE模块")
except ImportError as e:
    logger.warning(f"⚠️ 无法导入基础PC-VGAE模块: {e}")
    class DataProcessor:
        def process_data(self, data, ckg):
            return {'graph': None, 'node_features': {}}

class HypothesisEncoder:
    """假设到软提示编码器"""
    
    def __init__(self, prompt_dim: int = 64):
        self.prompt_dim = prompt_dim
        
        try:
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
                self.text_embed_dim = 384
            else:
                logger.warning("SentenceTransformers不可用，使用随机编码")
                self.sentence_model = None
                self.text_embed_dim = 128
        except Exception as e:
            logger.warning(f"无法加载Sentence-BERT模型: {e}，使用随机编码")
            self.sentence_model = None
            self.text_embed_dim = 128
        
        self.rule_types = [
            'degree_outlier', 'edge_strength_outlier', 'centrality_discrepancy',
            'correlation_distance_mismatch', 'conditional_instability', 
            'partial_correlation_drop', 'causal_desert', 'weak_causal_chain'
        ]
        self.rule_type_dim = len(self.rule_types)
        
        self.node_feature_dim = 32
        
        if TORCH_AVAILABLE:
            text_dim = prompt_dim // 3
            node_dim = prompt_dim // 3
            rule_dim = prompt_dim - text_dim - node_dim
            
            self.text_projector = nn.Linear(self.text_embed_dim, text_dim)
            self.node_projector = nn.Linear(self.node_feature_dim, node_dim)  # 修改：使用固定的节点特征维度
            self.rule_projector = nn.Linear(self.rule_type_dim, rule_dim)
        
        logger.info(f"假设编码器初始化完成，软提示维度: {prompt_dim}")
    
    def _encode_target_nodes(self, target_elements: List[str],
                           node_features: Dict[str, np.ndarray]) -> np.ndarray:
        """编码目标节点特征"""
        if not target_elements:
            return np.zeros(self.node_feature_dim)

        node_embeddings = []
        for element in target_elements:
            if element in node_features:
                features_data = node_features[element]
                
                if isinstance(features_data, dict):
                    feature_values = [
                        features_data.get('degree_centrality', 0.0),
                        features_data.get('in_degree', 0),
                        features_data.get('out_degree', 0)
                    ]
                    feature_array = np.array(feature_values + [0.0] * (self.node_feature_dim - len(feature_values)))
                    node_embeddings.append(feature_array)
                else:
                    feature_array = np.array(features_data).flatten()
                    
                    if len(feature_array) >= self.node_feature_dim:
                        truncated_array = feature_array[:self.node_feature_dim]
                        node_embeddings.append(truncated_array)
                    else:
                        padded_array = np.pad(feature_array, 
                                            (0, self.node_feature_dim - len(feature_array)), 
                                            'constant', constant_values=0.0)
                        node_embeddings.append(padded_array)
            else:
                hash_val = hash(element) % (2**31)
                np.random.seed(hash_val)
                node_embeddings.append(np.random.normal(0, 0.1, self.node_feature_dim))

        if node_embeddings:
            return np.mean(node_embeddings, axis=0)
        else:
            return np.zeros(self.node_feature_dim)


    def encode_hypothesis_to_prompt(self, hypothesis: Hypothesis, 
                                  node_features: Dict[str, np.ndarray]) -> HypothesisPrompt:
        """将假设编码为软提示向量"""
        text_embedding = self._encode_text(hypothesis.description)
        
        node_embedding = self._encode_target_nodes(hypothesis.target_elements, node_features)
        
        rule_onehot = self._encode_rule_type(hypothesis.hypothesis_type)
        
        if TORCH_AVAILABLE:
            with torch.no_grad():
                text_proj = self.text_projector(torch.FloatTensor(text_embedding))
                node_proj = self.node_projector(torch.FloatTensor(node_embedding))
                rule_proj = self.rule_projector(torch.FloatTensor(rule_onehot))
                
                combined_prompt = torch.cat([text_proj, node_proj, rule_proj], dim=0)
                combined_prompt = F.normalize(combined_prompt, p=2, dim=0)
                combined_prompt_np = combined_prompt.numpy()
        else:
            combined_prompt_np = np.concatenate([
                text_embedding[:self.prompt_dim//3],
                node_embedding[:self.prompt_dim//3],
                rule_onehot[:self.prompt_dim - 2*(self.prompt_dim//3)]
            ])
        
        return HypothesisPrompt(
            hypothesis_id=getattr(hypothesis, 'id', f"hyp_{hash(hypothesis.description) % 10000}"),
            text_embedding=text_embedding,
            node_features=node_embedding,
            rule_type_onehot=rule_onehot,
            confidence_score=hypothesis.confidence_score,
            priority_score=hypothesis.priority,
            target_elements=hypothesis.target_elements,
            combined_prompt=combined_prompt_np
        )
    
    def _encode_text(self, text: str) -> np.ndarray:
        """编码假设描述文本"""
        if self.sentence_model:
            try:
                embedding = self.sentence_model.encode(text)
                return embedding
            except Exception as e:
                logger.warning(f"文本编码失败: {e}，使用随机编码")
        
        hash_val = hash(text) % (2**31)
        np.random.seed(hash_val)
        return np.random.normal(0, 1, self.text_embed_dim)
    


    def _encode_rule_type(self, hypothesis_type: str) -> np.ndarray:
        """编码规则类型为独热向量"""
        onehot = np.zeros(self.rule_type_dim)
        
        type_mapping = {
            'degree_outlier': 'degree_outlier',
            'edge_strength_outlier': 'edge_strength_outlier',
            'centrality_discrepancy': 'centrality_discrepancy',
            'correlation_distance_mismatch': 'correlation_distance_mismatch',
            'conditional_instability': 'conditional_instability',
            'partial_correlation_drop': 'partial_correlation_drop',
            'causal_desert': 'causal_desert',
            'weak_causal_chain': 'weak_causal_chain'
        }
        
        rule_type = type_mapping.get(hypothesis_type, 'degree_outlier')
        if rule_type in self.rule_types:
            idx = self.rule_types.index(rule_type)
            onehot[idx] = 1.0
        
        return onehot


class HypothesisConditionedPCGATE(nn.Module):
    """假设条件化PC-GATE模型 - 非变分版本"""
    
    def __init__(self, node_features_dim: int, hypothesis_feature_dim: int = 16, 
                 hidden_dim: int = 256, latent_dim: int = 16, num_heads: int = 4):
        super().__init__()
        
        self.node_features_dim = node_features_dim
        self.hypothesis_feature_dim = hypothesis_feature_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        input_dim = node_features_dim + hypothesis_feature_dim
        
        self.encoder1 = GATv2Conv(
            input_dim,
            hidden_dim,
            heads=num_heads,
            dropout=0.1
        )
        self.encoder2 = GATv2Conv(
            hidden_dim * num_heads, 
            latent_dim,
            heads=1, 
            dropout=0.1
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim * num_heads)
        self.dropout = nn.Dropout(0.1)
        
        logger.info(f"假设条件化PC-GATE初始化: 节点特征{node_features_dim}维, "
                   f"假设特征{hypothesis_feature_dim}维, 输入总维度{input_dim}维, "
                   f"隐藏层{hidden_dim}维, 潜在空间{latent_dim}维")
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """前向传播 - 修改为非变分版本"""
        
        h1 = F.elu(self.encoder1(x, edge_index))
        h1 = self.dropout(h1)
        h1 = self.layer_norm(h1)
        
        z = self.encoder2(h1, edge_index)
        
        return z
    
    
    def decode(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """解码器：重构边概率"""
        row, col = edge_index
        edge_logits = (z[row] * z[col]).sum(dim=1)
        return torch.sigmoid(edge_logits)


class HypothesisConditionedUncertaintyAnalyzerPCGATE:
    """假设条件化不确定性分析器 - PC-GATE版本"""

    def __init__(self, prompt_dim: int = 64, device: str = 'cpu', enable_hypothesis_fasttrack: bool = True):
        self.prompt_dim = prompt_dim
        self.device = torch.device(device) if TORCH_AVAILABLE else 'cpu'
        self.torch_available = TORCH_AVAILABLE
        
        self.enable_hypothesis_fasttrack = enable_hypothesis_fasttrack

        self.hypothesis_encoder = HypothesisEncoder(prompt_dim)

        self.unified_model = None
        self.baseline_map = None
        self.hypothesis_impacts = {}

        self.learning_rate = 5e-4
        self.max_epochs = 800

        logger.info(f"假设条件化不确定性分析器PC-GATE初始化完成，直升通道: {'启用' if enable_hypothesis_fasttrack else '禁用'}")

    def train_unified_model(self, graph_data: Any, node_features: Dict[str, np.ndarray],
                        negative_edges: Any, unified_features: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """训练统一的PC-GATE模型（一次训练）- 非变分版本"""
        logger.info("开始训练统一PC-GATE模型（非变分版本）...")
        
        if not self.torch_available:
            logger.warning("PyTorch不可用，跳过实际训练")
            return {}
        
        node_list = list(unified_features.keys())
        feature_dim = len(next(iter(unified_features.values())))
        hypothesis_feature_dim = 16  # 假设特征维度
        
        X_base = torch.zeros(len(node_list), feature_dim, dtype=torch.float32)
        for i, node_id in enumerate(node_list):
            X_base[i] = torch.FloatTensor(unified_features[node_id])
        
        zeros_for_hypothesis = torch.zeros(len(node_list), hypothesis_feature_dim, dtype=torch.float32)
        X_train = torch.cat([X_base, zeros_for_hypothesis], dim=1)
        
        self.unified_model = HypothesisConditionedPCGATE(
            node_features_dim=feature_dim,
            hypothesis_feature_dim=hypothesis_feature_dim,
            hidden_dim=256,
            latent_dim=16
        ).to(self.device)
        
        optimizer = torch.optim.Adam(self.unified_model.parameters(), lr=self.learning_rate)
        
        x = X_train.to(self.device)
        edge_index = graph_data.edge_index.to(self.device)
        neg_edges = negative_edges.to(self.device)
        
        best_loss = float('inf')
        patience_counter = 0
        patience = 30
        best_model_path = 'best_unified_pcgate_model.pth'
        
        losses = []
        logger.info(f"开始训练：最大轮数 {self.max_epochs}，早停阈值 {patience}")
        
        for epoch in range(self.max_epochs):
            optimizer.zero_grad()
            
            z = self.unified_model(x, edge_index)
            
            pos_pred = self.unified_model.decode(z, edge_index)
            neg_pred = self.unified_model.decode(z, neg_edges)
            
            pos_loss = F.binary_cross_entropy(pos_pred, torch.ones_like(pos_pred))
            neg_loss = F.binary_cross_entropy(neg_pred, torch.zeros_like(neg_pred))
            total_loss = pos_loss + neg_loss
            
            total_loss.backward()
            optimizer.step()
            
            current_loss = total_loss.item()
            losses.append(current_loss)
            
            if current_loss < best_loss:
                best_loss = current_loss
                patience_counter = 0
                torch.save(self.unified_model.state_dict(), best_model_path)
                if epoch % 50 == 0:
                    logger.info(f"Epoch {epoch}: 新的最佳损失 {best_loss:.4f}")
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                logger.info(f"训练损失已稳定，提前停止在第 {epoch} 轮")
                break
            
            if epoch % 50 == 0:
                logger.info(f"Epoch {epoch}: Loss={current_loss:.4f}, Best={best_loss:.4f}, Patience={patience_counter}")
        
        if os.path.exists(best_model_path):
            self.unified_model.load_state_dict(torch.load(best_model_path))
            logger.info(f"已加载最佳PC-GATE模型，最终损失: {best_loss:.4f}")
            os.remove(best_model_path)
        
        converged = patience_counter >= patience or (len(losses) > 20 and abs(losses[-1] - losses[-20]) < 0.001)
        
        logger.info(f"统一PC-GATE模型训练完成，最终损失: {best_loss:.4f}，是否收敛: {converged}")
        
        return {
            'final_loss': best_loss,
            'loss_history': losses,
            'converged': converged,
            'total_epochs': len(losses),
            'stopped_early': patience_counter >= patience
        }

    def quantify_impact_and_aggregate(self, high_quality_hypotheses: List[Hypothesis],
                                    graph_data: Any, unified_features: Dict[str, np.ndarray],
                                    negative_edges: Any, node_to_idx: Dict[str, int],
                                    idx_to_node: Dict[int, str], data_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """量化假设影响力并进行加权聚合（PC-GATE版本）"""
        logger.info(f"开始量化 {len(high_quality_hypotheses)} 个高质量假设的影响力（PC-GATE版本）...")
        
        if not self.torch_available or self.unified_model is None:
            logger.error("PC-GATE模型未准备好，无法进行推理")
            return {}
        
        promoted_findings, remaining_hypotheses = self._promote_hypotheses_to_findings(
            high_quality_hypotheses, node_to_idx, idx_to_node, data_context
        )
        
        if not remaining_hypotheses:
            logger.info("所有假设都已通过直升通道处理，跳过常规PC-GATE流程")
            return {
                'edge_results': promoted_findings,
                'high_priority_edges': promoted_findings,
                'hypothesis_impacts': {},
                'candidate_pool_stats': {
                    'total_candidates': len(promoted_findings),
                    'hypothesis_driven': len(promoted_findings),
                    'conflict_driven': 0,
                    'fasttrack_promoted': len(promoted_findings)
                }
            }
        
        logger.info(f"对剩余 {len(remaining_hypotheses)} 个假设进行常规PC-GATE分析...")
        
        node_list = sorted(list(unified_features.keys()))
        feature_dim = len(next(iter(unified_features.values())))
        hypothesis_feature_dim = 16
        
        X_base = torch.zeros(len(node_list), feature_dim, dtype=torch.float32)
        for i, node_id in enumerate(node_list):
            X_base[i] = torch.FloatTensor(unified_features[node_id])
        X_base = X_base.to(self.device)
        
        edge_index = graph_data.edge_index.to(self.device)
        
        with torch.no_grad():
            zeros_for_hypothesis = torch.zeros(len(node_list), hypothesis_feature_dim).to(self.device)
            X_baseline = torch.cat([X_base, zeros_for_hypothesis], dim=1)
            
            z_base = self.unified_model(X_baseline, edge_index)
            
            all_edges = torch.cat([edge_index, negative_edges], dim=1)
            baseline_probs = self.unified_model.decode(z_base, all_edges)
            self.baseline_map = baseline_probs.cpu().numpy()
        
        hypothesis_deltas = {}
        hypothesis_weights = {}
        
        sorted_remaining_hypotheses = sorted(remaining_hypotheses, key=lambda h: h.id)
        
        for hyp_idx, hypothesis in enumerate(sorted_remaining_hypotheses):
            logger.info(f"处理剩余假设 {hyp_idx+1}/{len(sorted_remaining_hypotheses)}: {hypothesis.hypothesis_type}")
            
            hypothesis_prompt = self.hypothesis_encoder.encode_hypothesis_to_prompt(
                hypothesis, unified_features
            )
            h_vector = torch.FloatTensor(hypothesis_prompt.combined_prompt[:hypothesis_feature_dim]).to(self.device)
            
            H_features = torch.zeros(len(node_list), hypothesis_feature_dim).to(self.device)
            for target_element in hypothesis.target_elements:
                if target_element in node_to_idx:
                    node_idx = node_to_idx[target_element]
                    H_features[node_idx] = h_vector
            
            X_conditioned = torch.cat([X_base, H_features], dim=1)
            
            with torch.no_grad():
                z_cond = self.unified_model(X_conditioned, edge_index)
                conditioned_probs = self.unified_model.decode(z_cond, all_edges)
                conditioned_map = conditioned_probs.cpu().numpy()
            
            delta_map = conditioned_map - self.baseline_map
            
            total_impact = np.linalg.norm(delta_map)
            
            quality_score = hypothesis.confidence_score * hypothesis.priority
            final_weight = quality_score * total_impact
            
            hypothesis_deltas[hypothesis.id] = delta_map
            hypothesis_weights[hypothesis.id] = final_weight
            
            self.hypothesis_impacts[hypothesis.id] = {
                'hypothesis': hypothesis,
                'delta_map': delta_map,
                'total_impact': total_impact,
                'weight': final_weight
            }
        
        conflict_scores = self.calculate_direct_conflict_scores(
            graph_data, unified_features, negative_edges, 
            node_to_idx, idx_to_node, data_context or {}
        )
        
        logger.info("生成常规的发现候选池...")
        candidate_pool = {}
        
        num_hypotheses_to_select = max(10, int(len(sorted_remaining_hypotheses) * 0.2))
        top_k_hypotheses = min(num_hypotheses_to_select, len(sorted_remaining_hypotheses))
        sorted_hypotheses = sorted(
            self.hypothesis_impacts.items(),
            key=lambda x: (x[1]['total_impact'], x[0]),
            reverse=True
        )[:top_k_hypotheses]
        
        for hyp_id, impact_data in sorted_hypotheses:
            delta_map = impact_data['delta_map']
            edge_impacts = [(i, abs(delta_map[i])) for i in range(len(delta_map))]

            num_edges = len(delta_map)
            num_to_select = max(5, min(20, int(num_edges * 0.001))) 

            top_edges = sorted(edge_impacts, key=lambda x: (x[1], x[0]), reverse=True)[:num_to_select]
            
            for edge_idx, impact_value in top_edges:
                if edge_idx < edge_index.size(1):
                    source_idx = edge_index[0, edge_idx].item()
                    target_idx = edge_index[1, edge_idx].item()
                else:
                    neg_idx = edge_idx - edge_index.size(1)
                    source_idx = negative_edges[0, neg_idx].item()
                    target_idx = negative_edges[1, neg_idx].item()
                
                source_id = idx_to_node[source_idx]
                target_id = idx_to_node[target_idx]
                edge_key = (source_id, target_id)
                
                if edge_key not in candidate_pool:
                    candidate_pool[edge_key] = {
                        'sources': [],
                        'conflict_score': conflict_scores.get(edge_key, 0.0),
                        'impact_scores': [],
                        'contributing_hypotheses': []
                    }
                
                candidate_pool[edge_key]['sources'].append(f'high_impact_hypothesis_{hyp_id}')
                candidate_pool[edge_key]['impact_scores'].append(impact_value)
                candidate_pool[edge_key]['contributing_hypotheses'].append({
                    'hypothesis_id': hyp_id,
                    'contribution': delta_map[edge_idx],
                    'weighted_contribution': delta_map[edge_idx] * impact_data['weight']
                })
        
        num_conflicts_to_select = max(20, int(len(conflict_scores) * 0.05))
        sorted_conflicts = sorted(
            conflict_scores.items(),
            key=lambda x: (x[1], x[0]),
            reverse=True
        )[:num_conflicts_to_select]
        
        for edge_key, conflict_score in sorted_conflicts:
            if edge_key not in candidate_pool:
                candidate_pool[edge_key] = {
                    'sources': [],
                    'conflict_score': conflict_score,
                    'impact_scores': [],
                    'contributing_hypotheses': []
                }
            candidate_pool[edge_key]['sources'].append('direct_conflict')
        
        logger.info("计算常规发现的多维证据收敛分数...")
        regular_candidates = []
        
        for edge_key in sorted(candidate_pool.keys()):
            candidate_data = candidate_pool[edge_key]
            final_score = self.calculate_final_finding_score(
                edge_key, candidate_data, self.hypothesis_impacts
            )
            
            source_id, target_id = edge_key
            node_text_map = {}
            if data_context and 'ckg' in data_context:
                ckg = data_context['ckg']
                nodes_by_type = ckg.get('nodes_by_type', {})
                for node_type, nodes in nodes_by_type.items():
                    for node in nodes:
                        node_id = node.get('id', '')
                        node_text = node.get('text', node_id)
                        if node_id:
                            node_text_map[node_id] = node_text
            
            source_text = node_text_map.get(source_id, source_id)
            target_text = node_text_map.get(target_id, target_id)
            
            edge_exists = any('high_impact_hypothesis' in source for source in candidate_data['sources'])
            
            if edge_exists:
                edge_type = 'existing_unreliable' if final_score > 0.6 else 'existing_reliable'
            else:
                edge_type = 'missing_potential' if final_score > 0.6 else 'missing_irrelevant'
            
            edge_result = EdgeUncertaintyResult(
                source_id=source_id,
                target_id=target_id,
                source_text=source_text,
                target_text=target_text,
                edge_exists=edge_exists,
                reconstruction_prob=0.5,  # 占位值
                uncertainty_score=candidate_data['conflict_score'],
                edge_type=edge_type,
                confidence_interval=(0, 1),
                unified_score=final_score  # 使用最终分数作为统一分数
            )
            edge_result.contributing_hypotheses = candidate_data['contributing_hypotheses']
            
            regular_candidates.append(edge_result)
        
        logger.info("合并直升发现与常规发现...")
        combined_edge_results = self._merge_findings(promoted_findings, regular_candidates)
        
        if combined_edge_results:
            all_scores = [res.unified_score for res in combined_edge_results]
            threshold_75 = np.percentile(all_scores, 60) 
            high_priority_edges = [res for res in combined_edge_results if res.unified_score >= threshold_75]

            min_findings = min(5, len(combined_edge_results))
            if len(high_priority_edges) < min_findings:
                high_priority_edges = sorted(combined_edge_results, key=lambda r: r.unified_score, reverse=True)[:min_findings]

            logger.info(f"动态阈值筛选：75分位数阈值={threshold_75:.3f}, "
                        f"筛选出 {len(high_priority_edges)} 个高优先级发现")
        else:
            high_priority_edges = []
            threshold_75 = 0.0
        
        logger.info(f"多维证据收敛分析完成：{len(candidate_pool)} 个常规候选，"
                f"{len(promoted_findings)} 个直升发现，{len(high_priority_edges)} 个最终发现")
        
        
        return {
            'edge_results': combined_edge_results,
            'high_priority_edges': high_priority_edges,
            'hypothesis_impacts': self.hypothesis_impacts,
            'candidate_pool_stats': {
                'total_candidates': len(candidate_pool) + len(promoted_findings),
                'hypothesis_driven': len([c for c in candidate_pool.values() if any('high_impact' in s for s in c['sources'])]),
                'conflict_driven': len([c for c in candidate_pool.values() if 'direct_conflict' in c['sources']]),
                'fasttrack_promoted': len(promoted_findings),
                'dynamic_threshold': threshold_75
            }
        }


    def _promote_hypotheses_to_findings(self, hypotheses: List[Hypothesis], 
                                      node_to_idx: Dict[str, int],
                                      idx_to_node: Dict[int, str], 
                                      data_context: Dict[str, Any]) -> Tuple[List[EdgeUncertaintyResult], List[Hypothesis]]:
        """假设直升通道：将高质量假设直接转化为发现"""
        if not self.enable_hypothesis_fasttrack:
            logger.info("假设直升通道已禁用，跳过假设提拔")
            return [], hypotheses
        
        logger.info("开始假设直升通道处理...")
        
        promoted_findings = []
        remaining_hypotheses = []
        
        node_text_map = {}
        if data_context and 'ckg' in data_context:
            ckg = data_context['ckg']
            nodes_by_type = ckg.get('nodes_by_type', {})
            for node_type, nodes in nodes_by_type.items():
                for node in nodes:
                    node_id = node.get('id', '')
                    node_text = node.get('text', node_id)
                    if node_id:
                        node_text_map[node_id] = node_text
        
        eligible_types = {'correlation_distance_mismatch', 'fusion_causal_desert', 'high_correlation_long_distance'}
        eligible_rules = {'HighCorrelationLongDistanceRule', 'HypothesisFusionEngine'}
        
        for hypothesis in hypotheses:
            is_eligible = (
                hypothesis.hypothesis_type in eligible_types and
                hypothesis.confidence_score >= 0.85 and
                hypothesis.priority >= 0.85 and
                hasattr(hypothesis, 'rule_name') and hypothesis.rule_name in eligible_rules
            )
            
            if is_eligible:
                logger.info(f"提拔假设: {hypothesis.id} ({hypothesis.hypothesis_type})")
                
                findings = self._convert_hypothesis_to_findings(
                    hypothesis, node_to_idx, idx_to_node, node_text_map
                )
                promoted_findings.extend(findings)
            else:
                remaining_hypotheses.append(hypothesis)
        
        logger.info(f"假设直升通道完成：提拔 {len(promoted_findings)} 个发现，剩余 {len(remaining_hypotheses)} 个假设进入常规流程")
        return promoted_findings, remaining_hypotheses

    def _convert_hypothesis_to_findings(self, hypothesis: Hypothesis,
                                      node_to_idx: Dict[str, int],
                                      idx_to_node: Dict[int, str],
                                      node_text_map: Dict[str, str]) -> List[EdgeUncertaintyResult]:
        """将单个假设转化为边发现"""
        findings = []
        
        if hypothesis.hypothesis_type == 'correlation_distance_mismatch':
            evidence = hypothesis.evidence
            if isinstance(evidence, dict):
                node1_id = evidence.get('node1_id') or evidence.get('source_node')
                node2_id = evidence.get('node2_id') or evidence.get('target_node')
                correlation = evidence.get('correlation', 0.0)
                
                if node1_id and node2_id and node1_id in node_to_idx and node2_id in node_to_idx:
                    source_text = node_text_map.get(node1_id, node1_id)
                    target_text = node_text_map.get(node2_id, node2_id)
                    
                    unified_score = (hypothesis.confidence_score + hypothesis.priority) / 2 * abs(correlation)
                    
                    finding = EdgeUncertaintyResult(
                        source_id=node1_id,
                        target_id=node2_id,
                        source_text=source_text,
                        target_text=target_text,
                        edge_exists=False,  # 通常是缺失边
                        reconstruction_prob=abs(correlation),
                        uncertainty_score=0.1,  # 低不确定性，因为来自高质量假设
                        edge_type='missing_potential',
                        confidence_interval=(0.8, 1.0),
                        unified_score=unified_score
                    )
                    finding.contributing_hypotheses = [{
                        'hypothesis_id': hypothesis.id,
                        'contribution': unified_score,
                        'weighted_contribution': unified_score,
                        'source': 'hypothesis_fasttrack'
                    }]
                    findings.append(finding)
                    
                    logger.debug(f"转化相关性假设 {hypothesis.id}: {source_text} → {target_text} (分数: {unified_score:.3f})")
        
        elif hypothesis.hypothesis_type == 'fusion_causal_desert':
            target_elements = hypothesis.target_elements
            if target_elements:
                center_node = target_elements[0]  # 核心问题节点
                
                evidence = hypothesis.evidence
                if isinstance(evidence, dict) and 'fused_evidences' in evidence:
                    fused_evidences = evidence['fused_evidences']
                    
                    best_correlation = 0.0
                    best_edge = None
                    
                    for fused_evidence in fused_evidences:
                        if isinstance(fused_evidence, dict):
                            correlation = abs(fused_evidence.get('correlation', 0.0))
                            source_node = fused_evidence.get('source_node') or fused_evidence.get('node1_id')
                            target_node = fused_evidence.get('target_node') or fused_evidence.get('node2_id')
                            
                            if correlation > best_correlation and source_node and target_node:
                                if source_node in node_to_idx and target_node in node_to_idx:
                                    best_correlation = correlation
                                    best_edge = (source_node, target_node, correlation)
                    
                    if best_edge:
                        source_id, target_id, correlation = best_edge
                        source_text = node_text_map.get(source_id, source_id)
                        target_text = node_text_map.get(target_id, target_id)
                        
                        unified_score = (hypothesis.confidence_score + hypothesis.priority) / 2 * correlation * 1.2
                        
                        finding = EdgeUncertaintyResult(
                            source_id=source_id,
                            target_id=target_id,
                            source_text=source_text,
                            target_text=target_text,
                            edge_exists=False,
                            reconstruction_prob=correlation,
                            uncertainty_score=0.05,  # 极低不确定性
                            edge_type='missing_potential',
                            confidence_interval=(0.9, 1.0),
                            unified_score=unified_score
                        )
                        finding.contributing_hypotheses = [{
                            'hypothesis_id': hypothesis.id,
                            'contribution': unified_score,
                            'weighted_contribution': unified_score,
                            'source': 'hypothesis_fasttrack_fusion'
                        }]
                        findings.append(finding)
                        
                        logger.debug(f"转化融合假设 {hypothesis.id}: {source_text} → {target_text} (分数: {unified_score:.3f})")
        
        return findings



    def _merge_findings(self, promoted_findings: List[EdgeUncertaintyResult], 
                       regular_findings: List[EdgeUncertaintyResult]) -> List[EdgeUncertaintyResult]:
        """合并直升发现与常规发现，并去重"""
        if not promoted_findings:
            return sorted(regular_findings, key=lambda x: x.unified_score, reverse=True)
        
        if not regular_findings:
            return sorted(promoted_findings, key=lambda x: x.unified_score, reverse=True)
        
        promoted_edges = {(finding.source_id, finding.target_id): finding for finding in promoted_findings}
        
        merged_findings = list(promoted_findings)
        
        for regular_finding in regular_findings:
            edge_key = (regular_finding.source_id, regular_finding.target_id)
            
            if edge_key in promoted_edges:
                promoted_finding = promoted_edges[edge_key]
                if regular_finding.unified_score > promoted_finding.unified_score:
                    logger.debug(f"常规发现覆盖直升发现: {edge_key}, "
                               f"常规分数: {regular_finding.unified_score:.3f} > "
                               f"直升分数: {promoted_finding.unified_score:.3f}")
                    merged_findings = [f for f in merged_findings if (f.source_id, f.target_id) != edge_key]
                    merged_findings.append(regular_finding)
                else:
                    logger.debug(f"保留直升发现: {edge_key}, "
                               f"直升分数: {promoted_finding.unified_score:.3f} >= "
                               f"常规分数: {regular_finding.unified_score:.3f}")
            else:
                merged_findings.append(regular_finding)
        
        merged_findings = sorted(merged_findings, key=lambda x: x.unified_score, reverse=True)
        
        logger.info(f"发现合并完成: {len(promoted_findings)} 个直升 + {len(regular_findings)} 个常规 → {len(merged_findings)} 个最终")
        return merged_findings





    def train_hypothesis_conditioned_models(self, hypotheses: List[Hypothesis],
                                          graph_data: Any, node_features: Dict[str, np.ndarray],
                                          negative_edges: Any) -> Dict[str, Any]:
        """为每个假设训练独立的条件化模型"""
        logger.info(f"开始训练 {len(hypotheses)} 个假设条件化模型")
        
        if not self.torch_available:
            logger.warning("PyTorch不可用，跳过实际训练")
            return {}
        
        training_results = {}
        
        for i, hypothesis in enumerate(hypotheses):
            logger.info(f"训练假设 {i+1}/{len(hypotheses)}: {hypothesis.hypothesis_type}")
            
            hypothesis_prompt = self.hypothesis_encoder.encode_hypothesis_to_prompt(
                hypothesis, node_features
            )
            
            model = HypothesisConditionedPCVGAE(
                node_features_dim=graph_data.x.size(1),
                prompt_dim=self.prompt_dim
            ).to(self.device)
            
            training_result = self._train_single_model(
                model, hypothesis_prompt, graph_data, negative_edges
            )
            
            self.conditioned_models[hypothesis_prompt.hypothesis_id] = model
            training_results[hypothesis_prompt.hypothesis_id] = {
                'hypothesis': hypothesis,
                'prompt': hypothesis_prompt,
                'training_result': training_result
            }

        self.training_results = training_results
        logger.info(f"完成 {len(self.conditioned_models)} 个条件化模型训练")
        return training_results


    def generate_hypothesis_conditioned_uncertainty_maps(self, 
                                                       graph_data: Any,
                                                       negative_edges: Any,
                                                       mc_samples: int = 50,
                                                       data_context: Dict = None) -> Dict[str, List[EdgeUncertaintyResult]]:
        """使用训练好的模型进行真实的MC Dropout推理来生成不确定性地图"""
        logger.info(f"生成 {len(self.training_results)} 个假设条件化不确定性地图（MC采样：{mc_samples}次）")
        
        if not self.torch_available:
            logger.error("PyTorch不可用，无法进行真实推理")
            return {}
        
        if not self.conditioned_models:
            logger.error("没有训练好的模型可用")
            return {}
        
        hypothesis_uncertainty_maps = {}
        
        x = graph_data.x.to(self.device)
        edge_index = graph_data.edge_index.to(self.device)
        neg_edges = negative_edges.to(self.device)
        
        idx_to_node = data_context.get('idx_to_node', {}) if data_context else {}
        
        node_text_map = {}
        if data_context and 'ckg' in data_context:
            ckg = data_context['ckg']
            nodes_by_type = ckg.get('nodes_by_type', {})
            for node_type, nodes in nodes_by_type.items():
                for node in nodes:
                    node_id = node.get('id', '')
                    node_text = node.get('text', node_id)
                    if node_id:
                        node_text_map[node_id] = node_text
        
        for hypothesis_id, model in self.conditioned_models.items():
            logger.info(f"处理假设 {hypothesis_id} 的不确定性推理")
            
            if hypothesis_id in self.training_results:
                prompt = self.training_results[hypothesis_id]['prompt']
                prompt_tensor = torch.FloatTensor(prompt.combined_prompt).to(self.device)
            else:
                logger.warning(f"找不到假设 {hypothesis_id} 的软提示，跳过")
                continue
            
            model.train()
            
            with torch.no_grad():
                predictions = []
                
                for sample_idx in range(mc_samples):
                    mu, logvar, z = model(x, edge_index, prompt_tensor)
                    
                    all_edges = torch.cat([edge_index, neg_edges], dim=1)
                    reconstruction_probs = model.decode(z, all_edges)
                    predictions.append(reconstruction_probs)
                
                predictions_tensor = torch.stack(predictions)  # Shape: [mc_samples, num_all_edges]
                
                mean_probs = predictions_tensor.mean(dim=0)  # Shape: [num_all_edges]
                std_devs = predictions_tensor.std(dim=0)      # Shape: [num_all_edges]
                
                edge_results = []
                
                num_existing_edges = edge_index.size(1)
                for i in range(num_existing_edges):
                    source_idx = edge_index[0, i].item()
                    target_idx = edge_index[1, i].item()
                    
                    source_id = idx_to_node.get(source_idx, f"node_{source_idx}")
                    target_id = idx_to_node.get(target_idx, f"node_{target_idx}")
                    
                    source_text = node_text_map.get(source_id, source_id)
                    target_text = node_text_map.get(target_id, target_id)
                    
                    reconstruction_prob = mean_probs[i].item()
                    uncertainty_score = std_devs[i].item()
                    
                    if uncertainty_score > 0.3:  # 高不确定性阈值
                        edge_type = 'existing_unreliable'
                    else:
                        edge_type = 'existing_reliable'
                    
                    edge_result = EdgeUncertaintyResult(
                        source_id=source_id,
                        target_id=target_id,
                        source_text=source_text,
                        target_text=target_text,
                        edge_exists=True,
                        reconstruction_prob=reconstruction_prob,
                        uncertainty_score=uncertainty_score,
                        edge_type=edge_type,
                        confidence_interval=(
                            max(0, reconstruction_prob - 2*uncertainty_score),
                            min(1, reconstruction_prob + 2*uncertainty_score)
                        ),
                        unified_score=uncertainty_score  # 现有边使用不确定性作为统一分数
                    )
                    edge_results.append(edge_result)
                
                num_neg_edges = neg_edges.size(1)
                for i in range(num_neg_edges):
                    source_idx = neg_edges[0, i].item()
                    target_idx = neg_edges[1, i].item()
                    
                    source_id = idx_to_node.get(source_idx, f"node_{source_idx}")
                    target_id = idx_to_node.get(target_idx, f"node_{target_idx}")
                    
                    source_text = node_text_map.get(source_id, source_id)
                    target_text = node_text_map.get(target_id, target_id)
                    
                    prob_idx = num_existing_edges + i
                    reconstruction_prob = mean_probs[prob_idx].item()
                    uncertainty_score = std_devs[prob_idx].item()
                    
                    if reconstruction_prob > 0.7:  # 高重构概率阈值
                        edge_type = 'missing_potential'
                        unified_score = -reconstruction_prob  # 负值表示应该存在
                    else:
                        edge_type = 'missing_irrelevant'
                        unified_score = uncertainty_score
                    
                    edge_result = EdgeUncertaintyResult(
                        source_id=source_id,
                        target_id=target_id,
                        source_text=source_text,
                        target_text=target_text,
                        edge_exists=False,
                        reconstruction_prob=reconstruction_prob,
                        uncertainty_score=uncertainty_score,
                        edge_type=edge_type,
                        confidence_interval=(
                            max(0, reconstruction_prob - 2*uncertainty_score),
                            min(1, reconstruction_prob + 2*uncertainty_score)
                        ),
                        unified_score=unified_score
                    )
                    edge_results.append(edge_result)
                
                hypothesis_uncertainty_maps[hypothesis_id] = edge_results
                logger.info(f"假设 {hypothesis_id} 生成 {len(edge_results)} 条边的不确定性结果")
        
        logger.info(f"完成所有假设的不确定性地图生成")
        return hypothesis_uncertainty_maps

    def aggregate_hypothesis_uncertainty_maps(self, hypothesis_uncertainty_maps: Dict[str, List],
                                            training_results: Dict[str, Any],
                                            aggregation_method: str = 'weighted_max') -> List[EdgeUncertaintyResult]:
        """聚合多个假设条件化不确定性地图"""
        logger.info(f"使用 {aggregation_method} 方法聚合 {len(hypothesis_uncertainty_maps)} 个不确定性地图")

        if not hypothesis_uncertainty_maps:
            return []

        all_edges = set()
        for uncertainty_map in hypothesis_uncertainty_maps.values():
            for edge_result in uncertainty_map:
                edge_key = (edge_result.source_id, edge_result.target_id)
                all_edges.add(edge_key)

        aggregated_results = []

        for edge_key in all_edges:
            source_id, target_id = edge_key

            edge_results = []
            hypothesis_weights = []

            for hypothesis_id, uncertainty_map in hypothesis_uncertainty_maps.items():
                for edge_result in uncertainty_map:
                    if (edge_result.source_id, edge_result.target_id) == edge_key:
                        edge_results.append(edge_result)

                        if hypothesis_id in training_results:
                            hypothesis = training_results[hypothesis_id]['hypothesis']
                            training_quality = 1.0 - training_results[hypothesis_id]['training_result']['final_loss']
                            weight = hypothesis.confidence_score * max(0.1, training_quality)
                        else:
                            weight = 0.5

                        hypothesis_weights.append(weight)
                        break

            if not edge_results:
                continue

            if aggregation_method == 'weighted_max':
                aggregated_result = self._weighted_max_aggregation(edge_results, hypothesis_weights)
            elif aggregation_method == 'attention':
                aggregated_result = self._attention_aggregation(edge_results, hypothesis_weights)
            else:  # simple_max
                aggregated_result = self._simple_max_aggregation(edge_results)

            aggregated_results.append(aggregated_result)

        logger.info(f"聚合完成，生成 {len(aggregated_results)} 个边不确定性结果")
        return aggregated_results

    def _weighted_max_aggregation(self, edge_results: List, weights: List[float]):
        """加权最大值聚合策略"""
        if not edge_results:
            return None

        weighted_scores = []
        for result, weight in zip(edge_results, weights):
            weighted_score = abs(result.unified_score) * weight
            weighted_scores.append((weighted_score, result))

        best_score, best_result = max(weighted_scores, key=lambda x: x[0])

        return best_result

    def _attention_aggregation(self, edge_results: List, weights: List[float]):
        """注意力机制聚合策略"""
        if not edge_results:
            return None

        weights = np.array(weights)
        weights = weights / (np.sum(weights) + 1e-8)

        avg_uncertainty = sum(r.uncertainty_score * w for r, w in zip(edge_results, weights))
        avg_recon_prob = sum(r.reconstruction_prob * w for r, w in zip(edge_results, weights))
        avg_unified_score = sum(r.unified_score * w for r, w in zip(edge_results, weights))

        max_weight_idx = np.argmax(weights)
        base_result = edge_results[max_weight_idx]

        aggregated_result = EdgeUncertaintyResult(
            source_id=base_result.source_id,
            target_id=base_result.target_id,
            source_text=base_result.source_text,
            target_text=base_result.target_text,
            edge_exists=base_result.edge_exists,
            uncertainty_score=avg_uncertainty,
            reconstruction_prob=avg_recon_prob,
            edge_type=base_result.edge_type,
            confidence_interval=base_result.confidence_interval,
            unified_score=avg_unified_score
        )

        return aggregated_result

    def _simple_max_aggregation(self, edge_results: List):
        """简单最大值聚合策略"""
        if not edge_results:
            return None

        best_result = max(edge_results, key=lambda x: abs(x.unified_score))
        return best_result

    def calculate_direct_conflict_scores(self, graph_data: Any, unified_features: Dict[str, np.ndarray],
                                    negative_edges: Any, node_to_idx: Dict[str, int],
                                    idx_to_node: Dict[int, str], data_context: Dict[str, Any]) -> Dict[Tuple[str, str], float]:
        """计算所有边的数据-知识直接冲突分数"""
        logger.info("计算数据-知识直接冲突分数...")
        
        conflict_scores = {}
        
        correlation_matrix = data_context['processed_data'].get('correlation_matrix', pd.DataFrame())
        ckg = data_context['ckg']
        node_to_column_mapping = data_context['processed_data'].get('node_to_column_mapping', {})
        
        correlation_matrix_available = (
            isinstance(correlation_matrix, pd.DataFrame) and 
            not correlation_matrix.empty
        )
        
        if not isinstance(node_to_column_mapping, dict):
            logger.warning(f"node_to_column_mapping不是字典类型: {type(node_to_column_mapping)}")
            node_to_column_mapping = {}
        
        logger.info(f"相关性矩阵可用: {correlation_matrix_available}, 节点映射数量: {len(node_to_column_mapping)}")
        
        ckg_edge_confidence = {}
        if 'edges' in ckg:
            edges_data = ckg['edges']
            
            if isinstance(edges_data, dict):
                for edge_category, edge_list in edges_data.items():
                    if isinstance(edge_list, list):
                        for edge in edge_list:
                            if isinstance(edge, dict):
                                source_id = edge.get('source', '')
                                target_id = edge.get('target', '')
                                confidence = edge.get('weight', edge.get('confidence', 0.8))
                                if source_id and target_id:
                                    ckg_edge_confidence[(source_id, target_id)] = confidence
                                    ckg_edge_confidence[(target_id, source_id)] = confidence  # 无向边
            elif isinstance(edges_data, list):
                for edge in edges_data:
                    if isinstance(edge, str):
                        if '->' in edge:
                            parts = edge.split('->')
                            if len(parts) == 2:
                                source_id = parts[0].strip()
                                target_id = parts[1].strip()
                                confidence = 0.8  # 默认置信度
                                ckg_edge_confidence[(source_id, target_id)] = confidence
                                ckg_edge_confidence[(target_id, source_id)] = confidence
                        elif '--' in edge:
                            parts = edge.split('--')
                            if len(parts) == 2:
                                source_id = parts[0].strip()
                                target_id = parts[1].strip()
                                confidence = 0.8  # 默认置信度
                                ckg_edge_confidence[(source_id, target_id)] = confidence
                                ckg_edge_confidence[(target_id, source_id)] = confidence
                    elif isinstance(edge, dict):
                        source_id = edge.get('source_id', edge.get('source', ''))
                        target_id = edge.get('target_id', edge.get('target', ''))
                        confidence = edge.get('confidence', edge.get('weight', 0.8))
                        if source_id and target_id:
                            ckg_edge_confidence[(source_id, target_id)] = confidence
                            ckg_edge_confidence[(target_id, source_id)] = confidence
                    elif isinstance(edge, (list, tuple)) and len(edge) >= 2:
                        source_id = str(edge[0])
                        target_id = str(edge[1])
                        confidence = edge[2] if len(edge) > 2 else 0.8
                        ckg_edge_confidence[(source_id, target_id)] = confidence
                        ckg_edge_confidence[(target_id, source_id)] = confidence
        
        if not ckg_edge_confidence and 'graph' in data_context['processed_data']:
            graph = data_context['processed_data']['graph']
            if hasattr(graph, 'edges'):
                for edge in graph.edges(data=True):
                    source_id, target_id, edge_data = edge
                    confidence = edge_data.get('weight', edge_data.get('confidence', 0.8))
                    ckg_edge_confidence[(source_id, target_id)] = confidence
                    ckg_edge_confidence[(target_id, source_id)] = confidence
        
        edge_index = graph_data.edge_index
        all_edges = torch.cat([edge_index, negative_edges], dim=1)
        
        for i in range(all_edges.size(1)):
            source_idx = all_edges[0, i].item()
            target_idx = all_edges[1, i].item()
            source_id = idx_to_node[source_idx]
            target_id = idx_to_node[target_idx]
            
            edge_key = (source_id, target_id)
            
            data_correlation = 0.0
            
            if (source_id in node_to_column_mapping and 
                target_id in node_to_column_mapping and
                correlation_matrix_available):
                
                source_col = node_to_column_mapping[source_id]
                target_col = node_to_column_mapping[target_id]
                
                try:
                    if source_col in correlation_matrix.columns and target_col in correlation_matrix.columns:
                        data_correlation = correlation_matrix.loc[source_col, target_col]
                        if pd.isna(data_correlation):
                            data_correlation = 0.0
                except (KeyError, IndexError) as e:
                    logger.debug(f"获取相关系数失败 {source_col}-{target_col}: {e}")
                    data_correlation = 0.0
            
            edge_exists_in_ckg = edge_key in ckg_edge_confidence or (target_id, source_id) in ckg_edge_confidence
            
            if edge_exists_in_ckg:
                ckg_confidence = ckg_edge_confidence.get(edge_key, ckg_edge_confidence.get((target_id, source_id), 0.8))
                conflict_score = abs(ckg_confidence - abs(data_correlation))
            else:
                conflict_score = abs(data_correlation)
            
            conflict_scores[edge_key] = conflict_score
        
        logger.info(f"计算完成 {len(conflict_scores)} 条边的冲突分数，CKG边数: {len(ckg_edge_confidence)}")
        return conflict_scores




    def calculate_final_finding_score(self, edge_key: Tuple[str, str], 
                                    candidate_data: Dict[str, Any],
                                    hypothesis_impacts: Dict[str, Any]) -> float:
        """
        基于证据档案的多维度评分机制 - V3.0 层次化结构重构
        采用 "基础分 x 协同奖励" 的层次化评分模式
        
        Args:
            edge_key: 边键值对 (source_id, target_id)
            candidate_data: 候选边数据
            hypothesis_impacts: 假设影响力数据
        
        Returns:
            最终发现分数
        """
        source_id, target_id = edge_key
        
        evidence_profile = {
            'direct_impacts': [],  # 来自精准归因的"影响力"证据
            'conflict_score': candidate_data.get('conflict_score', 0.0),  # "冲突"证据
            'contributing_hypotheses': defaultdict(list),  # 按类型组织的"假设"证据
            'proximity_hypotheses': []  # 结构邻近假设
        }
        
        for contrib in candidate_data.get('contributing_hypotheses', []):
            hyp_id = contrib.get('hypothesis_id', '')
            if hyp_id in hypothesis_impacts:
                hypothesis = hypothesis_impacts[hyp_id]['hypothesis']
                impact_data = hypothesis_impacts[hyp_id]
                
                evidence_profile['contributing_hypotheses'][hypothesis.hypothesis_type].append(hypothesis)
                
                edge_directly_targeted = (
                    hasattr(hypothesis, 'explicitly_targeted_edges') and (
                        (source_id, target_id) in hypothesis.explicitly_targeted_edges or 
                        (target_id, source_id) in hypothesis.explicitly_targeted_edges
                    )
                )
                
                if edge_directly_targeted:
                    evidence_profile['direct_impacts'].append(impact_data['total_impact'])
                
                target_elements = hypothesis.target_elements
                edge_nodes = {source_id, target_id}
                
                hypothesis_nodes = set()
                for element in target_elements:
                    if '->' not in element:  # 这是一个节点
                        hypothesis_nodes.add(element)
                    else:  # 这是一个边，提取其端点
                        try:
                            edge_source, edge_target = element.split('->')
                            hypothesis_nodes.add(edge_source.strip())
                            hypothesis_nodes.add(edge_target.strip())
                        except ValueError:
                            continue
                
                relevant_proximity_types = {
                    'causal_desert', 'correlation_distance_mismatch', 
                    'fusion_causal_desert', 'partial_correlation_drop'
                }
                
                if (hypothesis_nodes.intersection(edge_nodes) and 
                    hypothesis.hypothesis_type in relevant_proximity_types):
                    evidence_profile['proximity_hypotheses'].append(hypothesis)
        
        
        max_impact = max(evidence_profile['direct_impacts']) if evidence_profile['direct_impacts'] else 0.0
        impact_score = 1 / (1 + np.exp(-(max_impact - 15) / 5))
        
        conflict_score = min(1.0, evidence_profile['conflict_score'])
        
        w_impact = 0.6  # 模型证据权重
        w_conflict = 0.4  # 数据证据权重
        base_score = (impact_score * w_impact) + (conflict_score * w_conflict)
        
        
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
        
        weighted_diversity_sum = 0.0
        max_possible_weight = sum(rule_weights.values())
        
        for hyp_type, hyp_list in evidence_profile['contributing_hypotheses'].items():
            if hyp_list:  # 如果该类型有假设
                rule_name = hyp_list[0].rule_name
                weight = rule_weights.get(rule_name, 0.5)
                weighted_diversity_sum += weight
        
        normalized_diversity = min(1.0, weighted_diversity_sum / max_possible_weight)
        diversity_bonus = normalized_diversity * 0.3  # 最高0.3的奖励
        
        all_qualities = []
        for hyp_list in evidence_profile['contributing_hypotheses'].values():
            for hyp in hyp_list:
                all_qualities.append(hyp.confidence_score * hyp.priority)
        
        if all_qualities and len(all_qualities) > 1:
            quality_variance = np.var(all_qualities)
            consistency_bonus = (1 - min(1.0, quality_variance)) * 0.2  # 最高0.2的奖励
        else:
            consistency_bonus = 0.0
        
        proximity_bonus = 0.0
        if evidence_profile['proximity_hypotheses']:
            proximity_qualities = []
            for hyp in evidence_profile['proximity_hypotheses']:
                quality = hyp.confidence_score * hyp.priority
                proximity_qualities.append(quality)
            
            if proximity_qualities:
                sorted_qualities = sorted(proximity_qualities, reverse=True)
                weights = [1.0 / (i + 1) for i in range(len(sorted_qualities))]  # 权重递减
                weighted_avg = np.average(sorted_qualities, weights=weights)
                proximity_bonus = min(1.0, weighted_avg) * 0.25  # 最高0.25的奖励
        
        synergy_multiplier = 1.0 + diversity_bonus + consistency_bonus + proximity_bonus
        
        unified_score = base_score * synergy_multiplier
        
        logger.debug(f"边 {edge_key} 层次化评分分析 (V3.0):")
        logger.debug(f"  【基础证据层】")
        logger.debug(f"    直接影响数: {len(evidence_profile['direct_impacts'])}, 最大值: {max_impact:.4f}")
        logger.debug(f"    影响力得分: {impact_score:.3f} (权重{w_impact})")
        logger.debug(f"    冲突得分: {conflict_score:.3f} (权重{w_conflict})")
        logger.debug(f"    基础分数: {base_score:.3f}")
        
        logger.debug(f"  【协同奖励层】")
        logger.debug(f"    多样性奖励: {diversity_bonus:.3f} (加权多样性: {weighted_diversity_sum:.3f})")
        logger.debug(f"    一致性奖励: {consistency_bonus:.3f} (质量方差: {np.var(all_qualities) if len(all_qualities) > 1 else 0:.3f})")
        logger.debug(f"    邻近度奖励: {proximity_bonus:.3f} (邻近假设: {len(evidence_profile['proximity_hypotheses'])})")
        logger.debug(f"    协同乘数: {synergy_multiplier:.3f}")
        
        logger.debug(f"  【最终结果】")
        logger.debug(f"    最终分数: {unified_score:.3f} = {base_score:.3f} × {synergy_multiplier:.3f}")
        
        hyp_type_counts = {hyp_type: len(hyps) for hyp_type, hyps in evidence_profile['contributing_hypotheses'].items()}
        logger.debug(f"    假设类型分布: {dict(hyp_type_counts)}")
        
        return unified_score





def _load_case_data(case_dir: Path,use_degraded_features: bool = False) -> Dict[str, Any]:
    """加载并处理真实的案例数据 - 增强版"""
    logger.info(f"加载真实案例数据: {case_dir}")
    
    ckg_file = case_dir / "causal_knowledge_graph.json"
    with open(ckg_file, 'r', encoding='utf-8') as f:
        ckg = json.load(f)
    
    data_file = case_dir / "sensor_data.csv"
    if data_file.exists():
        data = pd.read_csv(data_file)
        logger.info(f"加载传感器数据: {data.shape}")
    else:
        data = pd.DataFrame()
        logger.warning("传感器数据文件不存在")
    
    data_processor = DataProcessor()
    processed_data = data_processor.process_data(data, ckg)
    
    graph = processed_data['graph']
    real_node_features = processed_data['node_features']
    node_to_column_mapping = processed_data.get('node_to_column_mapping', {})
    
    node_list = sorted(list(graph.nodes()))
    node_to_idx = {node_id: i for i, node_id in enumerate(node_list)}
    idx_to_node = {i: node_id for node_id, i in node_to_idx.items()}
    num_nodes = len(node_list)
    
    logger.info(f"图统计: {num_nodes} 个节点, {len(graph.edges())} 条边")
    logger.info(f"节点-数据映射: {len(node_to_column_mapping)} 个节点映射到数据列")
    
    if TORCH_AVAILABLE:
        feature_extractor = EnhancedFeatureExtractor()
        unified_features = feature_extractor.extract_unified_features(
            ckg, data, processed_data['correlation_matrix'], node_to_column_mapping,
            use_degraded_features=use_degraded_features # <-- 传递开关
        )
        
        if unified_features:
            feature_dim = len(next(iter(unified_features.values())))
        else:
            feature_dim = 3  # 默认最小特征维度
        
        x = torch.zeros(num_nodes, feature_dim, dtype=torch.float32)
        for i, node_id in enumerate(node_list):
            if node_id in unified_features:
                x[i] = torch.FloatTensor(unified_features[node_id])
        
        edge_list = sorted(list(graph.edges()))
        logger.info(f"排序后的前5条边: {edge_list[:5]}")
        
        if edge_list:
            edge_index_list = [[node_to_idx[u], node_to_idx[v]] for u, v in edge_list]
            edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        
        torch_data = Data(x=x, edge_index=edge_index)
        
        num_neg_samples = edge_index.size(1) if edge_index.size(1) > 0 else num_nodes
        negative_edges = negative_sampling(
            edge_index=edge_index,
            num_nodes=num_nodes,
            num_neg_samples=num_neg_samples
        )
        
        logger.info(f"创建PyTorch几何数据: 节点特征 {x.shape}, 边索引 {edge_index.shape}, 负采样边 {negative_edges.shape}")
    else:
        torch_data = None
        negative_edges = None
        unified_features = {}
        logger.warning("PyTorch不可用，跳过张量数据创建")
    
    return {
        'ckg': ckg,
        'data': data,
        'processed_data': processed_data,
        'torch_data': torch_data,
        'negative_edges': negative_edges,
        'node_features': real_node_features,
        'unified_features': unified_features,
        'node_to_idx': node_to_idx,
        'idx_to_node': idx_to_node,
        'num_nodes': num_nodes
    }



def _save_enhanced_analysis_results(result: AggregatedUncertaintyResult, 
                                   output_dir: Path,
                                   analysis_results: Dict[str, Any]):
    """保存增强的分析结果（包含可解释性信息）"""
    
    uncertainty_map_data = []
    for edge_result in result.aggregated_uncertainty_map:
        edge_data = {
            'source_id': edge_result.source_id,
            'target_id': edge_result.target_id,
            'source_text': edge_result.source_text,
            'target_text': edge_result.target_text,
            'edge_exists': edge_result.edge_exists,
            'uncertainty_score': edge_result.uncertainty_score,
            'reconstruction_prob': edge_result.reconstruction_prob,
            'edge_type': edge_result.edge_type,
            'confidence_interval': edge_result.confidence_interval,
            'unified_score': edge_result.unified_score,
            'contributing_hypotheses': edge_result.contributing_hypotheses
        }
        uncertainty_map_data.append(edge_data)
    
    uncertainty_file = output_dir / "aggregated_uncertainty_map.json"
    with open(uncertainty_file, 'w', encoding='utf-8') as f:
        json.dump(convert_numpy_types(uncertainty_map_data), f, indent=2, ensure_ascii=False)
    
    high_priority_data = []
    for finding in result.high_priority_findings:
        main_contributors = sorted(
            finding.contributing_hypotheses, 
            key=lambda x: abs(x['weighted_contribution']), 
            reverse=True
        )[:3]  # 前3个主要贡献者
        
        finding_data = {
            'source_id': finding.source_id,
            'target_id': finding.target_id,
            'source_text': finding.source_text,
            'target_text': finding.target_text,
            'edge_exists': finding.edge_exists,
            'uncertainty_score': finding.uncertainty_score,
            'unified_score': finding.unified_score,
            'edge_type': finding.edge_type,
            'main_contributors': [
                {
                    'hypothesis_id': contrib['hypothesis_id'],
                    'contribution': contrib['weighted_contribution']
                }
                for contrib in main_contributors
            ],
            'explanation': _generate_finding_explanation(finding, main_contributors)
        }
        high_priority_data.append(finding_data)
    
    priority_file = output_dir / "final_aggregated_uncertainty_map.json"
    with open(priority_file, 'w', encoding='utf-8') as f:
        json.dump(convert_numpy_types(high_priority_data), f, indent=2, ensure_ascii=False)
    
    impact_report = {
        'case_id': result.case_id,
        'hypothesis_impacts': {}
    }
    
    for hyp_id, impact_data in analysis_results['hypothesis_impacts'].items():
        hypothesis = impact_data['hypothesis']
        impact_report['hypothesis_impacts'][hyp_id] = {
            'hypothesis_type': hypothesis.hypothesis_type,
            'description': hypothesis.description,
            'target_elements': hypothesis.target_elements,
            'confidence_score': hypothesis.confidence_score,
            'priority': hypothesis.priority,
            'total_impact': impact_data['total_impact'],
            'normalized_weight': impact_data['weight'],
            'num_affected_edges': np.sum(np.abs(impact_data['delta_map']) > 0.01)
        }
    
    impact_file = output_dir / "hypothesis_impact_report.json"
    with open(impact_file, 'w', encoding='utf-8') as f:
        json.dump(convert_numpy_types(impact_report), f, indent=2, ensure_ascii=False)
    
    explainability_file = output_dir / "explainability_summary.txt"
    with open(explainability_file, 'w', encoding='utf-8') as f:
        f.write("=== BCSA 可解释性分析摘要 ===\n")
        f.write(f"案例ID: {result.case_id}\n")
        f.write(f"生成时间: {result.metadata['generation_timestamp']}\n\n")
        
        f.write("=== 假设影响力排名 ===\n")
        sorted_impacts = sorted(
            analysis_results['hypothesis_impacts'].items(),
            key=lambda x: x[1]['total_impact'],
            reverse=True
        )
        
        for rank, (hyp_id, impact_data) in enumerate(sorted_impacts[:10], 1):
            hypothesis = impact_data['hypothesis']
            f.write(f"\n{rank}. 假设 {hyp_id}\n")
            f.write(f"   类型: {hypothesis.hypothesis_type}\n")
            f.write(f"   描述: {hypothesis.description[:100]}...\n")
            f.write(f"   总影响力: {impact_data['total_impact']:.4f}\n")
            f.write(f"   归一化权重: {impact_data['weight']:.4f}\n")
        
        f.write("\n=== 高优先级发现的因果链解释 ===\n")
        for i, finding in enumerate(result.high_priority_findings[:5], 1):
            f.write(f"\n{i}. {finding.source_text} → {finding.target_text}\n")
            f.write(f"   边类型: {finding.edge_type}\n")
            f.write(f"   统一分数: {finding.unified_score:.4f}\n")
            
            if finding.contributing_hypotheses:
                f.write("   主要贡献假设:\n")
                for contrib in finding.contributing_hypotheses[:3]:
                    f.write(f"     - 假设 {contrib['hypothesis_id']}: "
                           f"贡献度 {contrib['weighted_contribution']:.4f}\n")
    
    logger.info(f"增强的分析结果已保存:")
    logger.info(f"  - 聚合不确定性地图: {uncertainty_file}")
    logger.info(f"  - 高优先级发现: {priority_file}")
    logger.info(f"  - 假设影响力报告: {impact_file}")
    logger.info(f"  - 可解释性摘要: {explainability_file}")

def _generate_finding_explanation(finding: EdgeUncertaintyResult, 
                                 main_contributors: List[Dict]) -> str:
    """生成发现的自然语言解释"""
    if finding.edge_exists:
        if finding.edge_type == 'existing_unreliable':
            explanation = f"现有边 {finding.source_text} → {finding.target_text} 存在高不确定性。"
        else:
            explanation = f"现有边 {finding.source_text} → {finding.target_text} 相对可靠。"
    else:
        if finding.edge_type == 'missing_potential':
            explanation = f"缺失边 {finding.source_text} → {finding.target_text} 可能应该存在。"
        else:
            explanation = f"缺失边 {finding.source_text} → {finding.target_text} 确实不应存在。"
    
    if main_contributors:
        explanation += f" 此判断主要基于 {len(main_contributors)} 个假设的综合分析。"
    
    return explanation



def _convert_training_result(training_result: Dict[str, Any], hypothesis_id: str) -> TrainingResult:
    """转换训练结果为标准格式"""
    return TrainingResult(
        hypothesis_id=hypothesis_id,
        final_loss=training_result['final_loss'],
        loss_history=training_result['loss_history'],
        converged=training_result['converged']
    )



def run_conditioned_uncertainty_analysis_PCGATE(hypotheses: List[Hypothesis],
                                               case_dir: Path,
                                               output_dir: Path,
                                               aggregation_method: str = 'weighted_max',
                                               mc_samples: int = 50,
                                               quality_threshold: float = 0.7,
                                               enable_hypothesis_fasttrack: bool = True,
                                               use_degraded_features: bool = False,
                                               scoring_params: Optional[Dict[str, float]] = None) -> AggregatedUncertaintyResult:
    """
    运行假设条件化不确定性分析的主函数 - PC-GATE版本

    Args:
        hypotheses: 假设列表
        case_dir: 案例目录路径
        output_dir: 输出目录路径
        aggregation_method: 聚合方法
        mc_samples: 蒙特卡洛采样次数
        quality_threshold: 假设质量阈值
        enable_hypothesis_fasttrack: 是否启用假设直升通道机制
        use_degraded_features: 是否使用降级特征
        scoring_params: 评分参数，用于调整假设评分 {'alpha': 0.6, 'beta': 0.4}
    Returns:
        聚合后的不确定性分析结果
    """
    logger.info(f"开始假设条件化不确定性分析（PC-GATE版本）: {case_dir.name}, "
               f"直升通道: {'启用' if enable_hypothesis_fasttrack else '禁用'}")
    
    case_dir = Path(case_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        data_context = _load_case_data(case_dir)
        
        high_quality_hypotheses = [
            h for h in hypotheses 
            if h.confidence_score * h.priority >= quality_threshold
        ]
        logger.info(f"筛选出 {len(high_quality_hypotheses)}/{len(hypotheses)} 个高质量假设")
        
        feature_extractor = EnhancedFeatureExtractor()
        unified_features = feature_extractor.extract_unified_features(
            graph_data=data_context['ckg'],
            sensor_data=data_context['data'],
            correlation_matrix=data_context['processed_data']['correlation_matrix'],
            node_id_to_column_mapping=data_context['processed_data']['node_to_column_mapping']
        )
        
        analyzer = HypothesisConditionedUncertaintyAnalyzerPCGATE(enable_hypothesis_fasttrack=enable_hypothesis_fasttrack)
        
        logger.info("步骤1: 训练统一PC-GATE模型")
        training_result = analyzer.train_unified_model(
            data_context['torch_data'],
            data_context['node_features'],
            data_context['negative_edges'],
            unified_features,
        )
        
        logger.info("步骤2: 量化假设影响力并进行加权聚合（PC-GATE版本）")
        analysis_results = analyzer.quantify_impact_and_aggregate(
            high_quality_hypotheses,
            data_context['torch_data'],
            unified_features,
            data_context['negative_edges'],
            data_context['node_to_idx'],
            data_context['idx_to_node'],
            data_context,
        )
        
        result = AggregatedUncertaintyResult(
            case_id=case_dir.name,
            aggregated_uncertainty_map=analysis_results['edge_results'],
            training_results={'unified_model': _convert_training_result(training_result, 'unified_pcgate')},
            aggregation_method='weighted_impact_pcgate_with_fasttrack' if enable_hypothesis_fasttrack else 'weighted_impact_pcgate',
            high_priority_findings=analysis_results['high_priority_edges'],
            metadata={
                'mc_samples': mc_samples,
                'total_hypotheses': len(hypotheses),
                'high_quality_hypotheses': len(high_quality_hypotheses),
                'quality_threshold': quality_threshold,
                'enable_hypothesis_fasttrack': enable_hypothesis_fasttrack,
                'fasttrack_promoted_count': analysis_results['candidate_pool_stats'].get('fasttrack_promoted', 0),
                'generation_timestamp': datetime.now().isoformat(),
                'model_type': 'PC-GATE',
                'hypothesis_impacts': {
                    hyp_id: {
                        'total_impact': impact_data['total_impact'],
                        'weight': impact_data['weight']
                    }
                    for hyp_id, impact_data in analysis_results['hypothesis_impacts'].items()
                }
            }
        )
        
        _save_enhanced_analysis_results(result, output_dir, analysis_results)
        
        logger.info(f"假设条件化不确定性分析完成（PC-GATE版本）: "
                   f"{len(analysis_results['edge_results'])} 个边，"
                   f"{len(analysis_results['high_priority_edges'])} 个高优先级发现")
        
        return result
        
    except Exception as e:
        logger.error(f"假设条件化不确定性分析失败（PC-GATE版本）: {e}")
        raise

def main():
    """主函数 - 独立运行假设条件化分析模块（PC-GATE版本）"""
    print("=" * 60)
    print("🎯 BCSA假设条件化PC-GATE模块 - 独立运行")
    print("=" * 60)
    
    case_id = 'Mixed_small_01'
    case_dir = Path("./seek_data_v3_deep_enhanced/cases/smallcase/Mixed") / case_id
    hypotheses_dir = Path("./results/hypothesis_generation_enhanced")
    output_dir = Path("./results/conditioned_uncertainty_analysis_pcgate")
    enable_hypothesis_fasttrack = True
    
    if not case_dir.exists():
        logger.error(f"测试案例目录不存在: {case_dir}")
        return
    
    try:
        hypotheses_file = hypotheses_dir / "generated_hypotheses.json"
        if not hypotheses_file.exists():
            logger.error(f"假设文件不存在: {hypotheses_file}")
            logger.info("请先运行 BCSA_01_hypothesis_generator.py 生成假设")
            return
        
        with open(hypotheses_file, 'r', encoding='utf-8') as f:
            hypotheses_data = json.load(f)
        
        hypotheses = []
        for hyp_data in hypotheses_data:
            hypothesis = Hypothesis(
                id=hyp_data['id'],
                rule_name=hyp_data['rule_name'],
                rule_category=hyp_data['rule_category'],
                hypothesis_type=hyp_data['hypothesis_type'],
                description=hyp_data['description'],
                target_elements=hyp_data['target_elements'],
                evidence=hyp_data['evidence'],
                confidence_score=hyp_data['confidence_score'],
                priority=hyp_data['priority'],
                metadata=hyp_data.get('metadata', {})
            )
            hypotheses.append(hypothesis)
        
        logger.info(f"加载 {len(hypotheses)} 个假设")
        
        result = run_conditioned_uncertainty_analysis_PCGATE(
            hypotheses, case_dir, output_dir, enable_hypothesis_fasttrack
        )
        
        print(f"\n✅ 假设条件化不确定性分析成功完成（PC-GATE版本）!")
        print(f"📊 案例: {result.case_id}")
        print(f"📊 聚合方法: {result.aggregation_method}")
        print(f"📊 训练模型数: {len(result.training_results)}")
        print(f"📊 聚合边数: {len(result.aggregated_uncertainty_map)}")
        print(f"📊 高优先级发现: {len(result.high_priority_findings)}")
        
        if result.training_results:
            converged_count = len([r for r in result.training_results.values() if r.converged])
            print(f"📊 收敛模型数: {converged_count}/{len(result.training_results)}")
            avg_loss = np.mean([r.final_loss for r in result.training_results.values()])
            print(f"📊 平均最终损失: {avg_loss:.4f}")
        
        if result.high_priority_findings:
            print(f"\n🔍 前5个高优先级发现:")
            for i, finding in enumerate(result.high_priority_findings[:5], 1):
                print(f"   {i}. {finding.source_text} → {finding.target_text}")
                print(f"      统一分数: {finding.unified_score:.3f}, 类型: {finding.edge_type}")
        
        print(f"\n💾 输出目录: {output_dir}")
        
    except Exception as e:
        logger.error(f"假设条件化不确定性分析失败（PC-GATE版本）: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
