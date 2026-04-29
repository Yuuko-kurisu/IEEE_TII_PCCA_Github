#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SEEK框架Ground Truth数据加载器
规范化处理预置的Ground Truth文件，支持证据区域计算
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
import logging

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EvidenceRegion:
    """证据区域数据结构"""
    region_id: str
    blind_spot_type: str
    description: str
    required_evidence: Set[str]  # 该区域要求的证据集合
    evidence_nodes: Set[str]     # 涉及的节点
    evidence_edges: Set[Tuple[str, str]]  # 涉及的边
    severity: float
    detection_difficulty: str

@dataclass
class ProcessedGroundTruth:
    """处理后的Ground Truth数据结构"""
    case_id: str
    evidence_regions: List[EvidenceRegion]
    total_required_evidence: Set[str]
    blind_spot_distribution: Dict[str, int]
    metadata: Dict[str, Any]

class GroundTruthLoader:
    """Ground Truth数据加载器"""
    
    def __init__(self, case_dir: Path):
        self.case_dir = Path(case_dir)
        self.case_id = self.case_dir.name
        
    def load_original_ground_truth(self) -> Dict[str, Any]:
        """加载原始Ground Truth文件"""
        gt_file = self.case_dir / "ground_truth.json"
        if not gt_file.exists():
            raise FileNotFoundError(f"Ground Truth文件不存在: {gt_file}")
        
        with open(gt_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def load_processed_ground_truth(self) -> Dict[str, Any]:
        """加载处理后的Ground Truth文件"""
        processed_gt_file = self.case_dir / "processed_ground_truth.json"
        if not processed_gt_file.exists():
            raise FileNotFoundError(f"处理后Ground Truth文件不存在: {processed_gt_file}")
        
        with open(processed_gt_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def extract_evidence_regions(self, processed_gt: Dict[str, Any]) -> List[EvidenceRegion]:
        """从处理后的GT中提取证据区域 - 支持新的evidence_zones格式"""
        evidence_regions = []

        evidence_zones = processed_gt.get('evidence_zones', {})
        if evidence_zones:
            logger.info(f"使用新的evidence_zones格式，发现 {len(evidence_zones)} 个证据区域")

            for zone_id, zone_data in evidence_zones.items():
                region_id = zone_data.get('zone_id', zone_id)
                blind_spot_type = zone_data.get('blind_spot_type', 'unknown')
                description = zone_data.get('description', '')
                severity = zone_data.get('severity', 0.5)

                required_evidence = set()
                evidence_nodes = set()
                evidence_edges = set()

                target_nodes = zone_data.get('target_nodes', [])
                for node in target_nodes:
                    evidence_nodes.add(node)
                    required_evidence.add(f"node:{node}")

                evidence_edges_data = zone_data.get('evidence_edges', [])
                for edge_data in evidence_edges_data:
                    if isinstance(edge_data, dict):
                        edge_key = edge_data.get('edge_key', '')
                        edge_type = edge_data.get('edge_type', 'unknown')

                        if ' -> ' in edge_key:
                            source, target = edge_key.split(' -> ', 1)
                            source = source.strip()
                            target = target.strip()

                            if source and target:
                                evidence_edges.add((source, target))

                                if edge_type == 'wrong_edge':
                                    required_evidence.add(f"spurious_edge:{source}->{target}")
                                elif edge_type == 'missing_edge':
                                    required_evidence.add(f"missing_edge:{source}->{target}")
                                else:
                                    required_evidence.add(f"{edge_type}:{source}->{target}")

                region = EvidenceRegion(
                    region_id=region_id,
                    blind_spot_type=blind_spot_type,
                    description=description,
                    required_evidence=required_evidence,
                    evidence_nodes=evidence_nodes,
                    evidence_edges=evidence_edges,
                    severity=severity,
                    detection_difficulty=zone_data.get('detection_criteria', {}).get('detection_difficulty', 'medium')
                )

                evidence_regions.append(region)
                logger.debug(f"解析证据区域 {region_id}: {len(evidence_nodes)} 节点, {len(evidence_edges)} 边")

        else:
            logger.info("未找到evidence_zones，回退到原始格式")
            for i, target in enumerate(processed_gt.get('original_ground_truth_targets', [])):
                region_id = f"region_{i+1}_{target.get('blind_spot_type', 'unknown')}"

                required_evidence = set()
                evidence_nodes = set()
                evidence_edges = set()

                target_nodes = target.get('target_nodes', [])
                for node in target_nodes:
                    evidence_nodes.add(node)
                    required_evidence.add(f"node:{node}")

                expected_findings = target.get('expected_findings', {})

                blind_spot_type = target.get('blind_spot_type', 'unknown')

                if blind_spot_type == 'causal_chain_break':
                    missing_edges = expected_findings.get('missing_edges', [])
                    for edge in missing_edges:
                        if isinstance(edge, dict):
                            source = edge.get('source', '')
                            target_node = edge.get('target', '')
                            if source and target_node:
                                evidence_edges.add((source, target_node))
                                required_evidence.add(f"missing_edge:{source}->{target_node}")

                elif blind_spot_type == 'confounded_relations':
                    spurious_edges = expected_findings.get('spurious_edges', [])
                    for edge in spurious_edges:
                        if isinstance(edge, dict):
                            source = edge.get('source', '')
                            target_node = edge.get('target', '')
                            if source and target_node:
                                evidence_edges.add((source, target_node))
                                required_evidence.add(f"spurious_edge:{source}->{target_node}")

                    true_edges = expected_findings.get('true_causal_edges', [])
                    for edge in true_edges:
                        if isinstance(edge, dict):
                            source = edge.get('source', '')
                            target_node = edge.get('target', '')
                            if source and target_node:
                                evidence_edges.add((source, target_node))
                                required_evidence.add(f"true_edge:{source}->{target_node}")

                elif blind_spot_type == 'causal_desert':
                    missing_connections = expected_findings.get('missing_connections', [])
                    for conn in missing_connections:
                        if isinstance(conn, dict):
                            source = conn.get('source', '')
                            target_node = conn.get('target', '')
                            if source and target_node:
                                evidence_edges.add((source, target_node))
                                required_evidence.add(f"desert_edge:{source}->{target_node}")

                region = EvidenceRegion(
                    region_id=region_id,
                    blind_spot_type=blind_spot_type,
                    description=target.get('description', ''),
                    required_evidence=required_evidence,
                    evidence_nodes=evidence_nodes,
                    evidence_edges=evidence_edges,
                    severity=target.get('severity', 0.5),
                    detection_difficulty=target.get('detection_difficulty', 'medium')
                )

                evidence_regions.append(region)

        logger.info(f"成功解析 {len(evidence_regions)} 个证据区域")
        total_edges = sum(len(region.evidence_edges) for region in evidence_regions)
        logger.info(f"总计 {total_edges} 条边证据")

        return evidence_regions
    
    def load_complete_ground_truth(self) -> ProcessedGroundTruth:
        """加载完整的Ground Truth数据"""
        logger.info(f"加载案例 {self.case_id} 的Ground Truth数据")
        
        processed_gt = self.load_processed_ground_truth()
        
        evidence_regions = self.extract_evidence_regions(processed_gt)
        
        total_required_evidence = set()
        for region in evidence_regions:
            total_required_evidence.update(region.required_evidence)
        
        blind_spot_distribution = processed_gt.get('evaluation_metrics', {}).get('blind_spot_distribution', {})
        
        complete_gt = ProcessedGroundTruth(
            case_id=self.case_id,
            evidence_regions=evidence_regions,
            total_required_evidence=total_required_evidence,
            blind_spot_distribution=blind_spot_distribution,
            metadata=processed_gt.get('dataset_metadata', {})
        )
        
        logger.info(f"成功加载 {len(evidence_regions)} 个证据区域，总计 {len(total_required_evidence)} 个必需证据")
        return complete_gt

class EvidenceCalculator:
    """证据计算器 - 用于计算EDR、EDP、F1等指标"""
    
    def __init__(self, ground_truth: ProcessedGroundTruth):
        self.ground_truth = ground_truth
    
    def extract_found_evidence(self, cognitive_map: Dict[str, Any]) -> Set[str]:
        """从认知不确定性地图中提取发现的证据"""
        found_evidence = set()
        
        for node in cognitive_map.get('node_uncertainties', []):
            if node.get('uncertainty_score', 0) > 0.5:  # 高不确定性节点
                node_id = node.get('node_id', '')
                found_evidence.add(f"node:{node_id}")
        
        for edge in cognitive_map.get('edge_uncertainties', []):
            uncertainty_score = edge.get('uncertainty_score', 0)
            source_id = edge.get('source_id', '')
            target_id = edge.get('target_id', '')
            
            if uncertainty_score > 0.8:  # 类型A：高度不可靠的现有边
                found_evidence.add(f"spurious_edge:{source_id}->{target_id}")
            elif uncertainty_score < -0.8:  # 类型B：应该存在的缺失边
                found_evidence.add(f"missing_edge:{source_id}->{target_id}")
        
        return found_evidence
    
    def calculate_evidence_metrics(self, cognitive_map: Dict[str, Any]) -> Dict[str, float]:
        """计算证据发现指标"""
        found_evidence = self.extract_found_evidence(cognitive_map)
        
        region_found_evidence = []
        for region in self.ground_truth.evidence_regions:
            region_found = found_evidence & region.required_evidence
            region_found_evidence.append(region_found)
        
        total_required = len(self.ground_truth.total_required_evidence)
        total_found_in_regions = len(set().union(*region_found_evidence)) if region_found_evidence else 0
        edr = total_found_in_regions / total_required if total_required > 0 else 0.0
        
        total_found = len(found_evidence)
        edp = total_found_in_regions / total_found if total_found > 0 else 0.0
        
        f1_score = 2 * edr * edp / (edr + edp) if (edr + edp) > 0 else 0.0
        
        return {
            'evidence_discovery_recall': edr,
            'evidence_discovery_precision': edp,
            'evidence_f1_score': f1_score,
            'total_required_evidence': total_required,
            'total_found_evidence': total_found,
            'valid_found_evidence': total_found_in_regions
        }

def test_ground_truth_loader():
    """测试Ground Truth加载器"""
    print("=== Ground Truth加载器测试 ===")
    
    case_dir = Path("./seek_data_v3_deep_enhanced/cases/smallcase/Mixed/Mixed_small_01")
    
    try:
        loader = GroundTruthLoader(case_dir)
        
        gt = loader.load_complete_ground_truth()
        
        print(f"✅ 成功加载案例: {gt.case_id}")
        print(f"📊 证据区域数量: {len(gt.evidence_regions)}")
        print(f"📊 总必需证据数: {len(gt.total_required_evidence)}")
        print(f"📊 盲区分布: {gt.blind_spot_distribution}")
        
        print(f"\n🔍 证据区域详情:")
        for i, region in enumerate(gt.evidence_regions, 1):
            print(f"   {i}. {region.region_id}")
            print(f"      类型: {region.blind_spot_type}")
            print(f"      必需证据数: {len(region.required_evidence)}")
            print(f"      涉及节点数: {len(region.evidence_nodes)}")
            print(f"      涉及边数: {len(region.evidence_edges)}")
        
        calculator = EvidenceCalculator(gt)
        
        mock_cognitive_map = {
            'node_uncertainties': [
                {'node_id': '温度', 'uncertainty_score': 0.8},
                {'node_id': 'Laser-C', 'uncertainty_score': 0.6}
            ],
            'edge_uncertainties': [
                {'source_id': 'n001', 'target_id': 'n002', 'uncertainty_score': 0.9},
                {'source_id': 'n003', 'target_id': 'n004', 'uncertainty_score': -0.9}
            ]
        }
        
        metrics = calculator.calculate_evidence_metrics(mock_cognitive_map)
        print(f"\n📈 证据发现指标:")
        print(f"   - EDR (召回率): {metrics['evidence_discovery_recall']:.4f}")
        print(f"   - EDP (精确率): {metrics['evidence_discovery_precision']:.4f}")
        print(f"   - F1分数: {metrics['evidence_f1_score']:.4f}")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_ground_truth_loader()
