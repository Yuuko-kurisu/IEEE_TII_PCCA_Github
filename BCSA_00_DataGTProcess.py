#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BCSA证据区域处理器 - 构建结构化Ground Truth
将概念级盲区映射为CKG上的具体证据区域
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict
import glob
import os
import pandas as pd  # 用于时间戳处理

os.chdir(os.path.dirname(os.path.abspath(__file__)))
print("当前工作目录:", os.getcwd())

class EvidenceZoneProcessor:
    """证据区域处理器 - 构建结构化Ground Truth"""
    
    def __init__(self):
        print("证据区域处理器已初始化")
    def process_ground_truth_to_evidence_zones(self, ground_truth_file: str, 
                                            ckg_file: str, 
                                            output_file: str = None) -> Dict[str, Any]:
        """
        将Ground Truth处理为结构化证据区域
        
        Args:
            ground_truth_file: Ground Truth JSON文件路径
            ckg_file: 因果知识图谱JSON文件路径  
            output_file: 输出的处理后GT文件路径
        
        Returns:
            处理后的结构化证据区域数据
        """
        print(f"开始处理Ground Truth到证据区域...")
        
        with open(ground_truth_file, 'r', encoding='utf-8') as f:
            ground_truth = json.load(f)
        
        with open(ckg_file, 'r', encoding='utf-8') as f:
            ckg = json.load(f)
        
        enhanced_ckg = self._enhance_ckg_with_hidden_factors(ckg, ckg_file)
        
        node_mappings = self._build_node_mappings(enhanced_ckg)
        
        evidence_zones = {}
        processed_targets = []
        
        for i, gt_target in enumerate(ground_truth.get('ground_truth_targets', [])):
            zone_id = f"evidence_zone_{i}"
            
            evidence_zone = self._create_evidence_zone(
                gt_target, node_mappings, enhanced_ckg, zone_id
            )
            
            evidence_zones[zone_id] = evidence_zone
            processed_targets.append({
                'original_target': gt_target,
                'evidence_zone_id': zone_id,
                'evidence_zone': evidence_zone
            })
        
        processed_gt = {
            'case_id': ground_truth.get('case_id'),
            'dataset_metadata': ground_truth.get('dataset_metadata'),
            'evaluation_metrics': ground_truth.get('evaluation_metrics'),
            'original_ground_truth_targets': ground_truth.get('ground_truth_targets'),
            
            'evidence_zones': evidence_zones,
            'processed_targets': processed_targets,
            
            'evaluation_config': {
                'hit_threshold': 2,  # 命中阈值：至少需要发现2个证据
                'evidence_types': ['wrong_edge', 'missing_edge'],
                'scoring_method': 'weighted_evidence',
                'quality_weights': {
                    'high_confidence_evidence': 1.0,
                    'medium_confidence_evidence': 0.7,
                    'low_confidence_evidence': 0.4
                }
            },
            
            'validation_guide': self._generate_validation_guide(evidence_zones),
            
            'processing_metadata': {
                'processor_version': 'v1.0',
                'processing_timestamp': pd.Timestamp.now().isoformat(),
                'total_evidence_zones': len(evidence_zones),
                'total_evidence_edges': sum(len(zone.get('evidence_edges', [])) 
                                        for zone in evidence_zones.values()),
                'ckg_enhanced_with_hidden_factors': True,
                'enhanced_ckg_path': ckg_file  # 记录增强后的CKG保存位置
            }
        }
        
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(processed_gt, f, ensure_ascii=False, indent=2, default=str)
            
            print(f"✓ 处理后的Ground Truth已保存: {output_path}")
        
        print(f"✓ 证据区域处理完成: {len(evidence_zones)}个证据区域")
        return processed_gt
    
    def _enhance_ckg_with_hidden_factors(self, ckg: Dict, ckg_file: str) -> Dict[str, Any]:
        """
        增强CKG：识别并添加隐藏因子节点
        
        Args:
            ckg: 原始因果知识图谱数据
            ckg_file: CKG文件路径，用于保存增强后的版本
        
        Returns:
            增强后的CKG数据
        """
        print("🔍 识别并添加隐藏因子节点...")
        
        import copy
        enhanced_ckg = copy.deepcopy(ckg)
        
        hidden_factors = []
        hidden_factor_names = set()
        
        special_columns_info = enhanced_ckg.get('special_columns_info', {})
        confounded_relations = special_columns_info.get('confounded_relations', {})
        
        if 'hidden_variable_name' in confounded_relations:
            hidden_factor_names.add(confounded_relations['hidden_variable_name'])
        
        blind_spot_markers = enhanced_ckg.get('blind_spot_markers', [])
        for marker in blind_spot_markers:
            if marker.get('type') == 'confounded_relations':
                if 'hidden_confounder' in marker:
                    hidden_factor_names.add(marker['hidden_confounder'])
                if 'hidden_variable_name' in marker:
                    hidden_factor_names.add(marker['hidden_variable_name'])
        
        edges = enhanced_ckg.get('edges', {})
        for edge_key, edge_list in edges.items():
            for edge in edge_list:
                if edge.get('is_confounded') and 'hidden_confounder' in edge:
                    hidden_factor_names.add(edge['hidden_confounder'])
        
        if not hidden_factor_names:
            print("   ⚠️  未发现隐藏因子，跳过CKG增强")
            return enhanced_ckg
        
        print(f"   🎯 发现 {len(hidden_factor_names)} 个隐藏因子: {list(hidden_factor_names)}")
        
        existing_node_ids = []
        nodes_by_type = enhanced_ckg.get('nodes_by_type', {})
        for node_type, nodes in nodes_by_type.items():
            for node in nodes:
                node_id = node.get('id', '')
                if node_id.startswith('n') and node_id[1:].isdigit():
                    existing_node_ids.append(int(node_id[1:]))
        
        next_node_number = max(existing_node_ids) + 1 if existing_node_ids else 1
        next_creation_order = max(
            node.get('creation_metadata', {}).get('creation_order', 0)
            for node_type_nodes in nodes_by_type.values()
            for node in node_type_nodes
        ) + 1 if nodes_by_type else 1
        
        for hidden_factor_name in sorted(hidden_factor_names):  # 排序确保一致性
            node_id = f"n{next_node_number:03d}"
            data_column_name = f"hidden_confounder_{hidden_factor_name}"
            
            hidden_factor_node = {
                "id": node_id,
                "text": hidden_factor_name,
                "confidence": 0.9,  # 默认置信度
                "node_type": "hidden",
                "scenario_specific": False,
                "data_column_name": data_column_name,
                "creation_metadata": {
                    "creation_order": next_creation_order,
                    "node_type": "hidden",
                    "original_text": hidden_factor_name
                }
            }
            
            hidden_factors.append(hidden_factor_node)
            next_node_number += 1
            next_creation_order += 1
            
            print(f"   ✅ 创建隐藏因子节点: {node_id} - {hidden_factor_name}")
        
        if 'nodes_by_type' not in enhanced_ckg:
            enhanced_ckg['nodes_by_type'] = {}
        
        enhanced_ckg['nodes_by_type']['hidden_factors'] = hidden_factors
        
        if 'node_id_to_column_mapping' not in enhanced_ckg:
            enhanced_ckg['node_id_to_column_mapping'] = {}
        if 'column_to_node_mapping' not in enhanced_ckg:
            enhanced_ckg['column_to_node_mapping'] = {}
        
        for hidden_factor in hidden_factors:
            node_id = hidden_factor['id']
            column_name = hidden_factor['data_column_name']
            
            enhanced_ckg['node_id_to_column_mapping'][node_id] = column_name
            enhanced_ckg['column_to_node_mapping'][column_name] = hidden_factor
        
        if 'total_nodes' in enhanced_ckg:
            enhanced_ckg['total_nodes'] += len(hidden_factors)
        
        metadata = enhanced_ckg.setdefault('dataset_metadata', {})
        metadata['hidden_factors_added'] = len(hidden_factors)
        metadata['hidden_factor_names'] = list(hidden_factor_names)
        
        try:
            with open(ckg_file, 'w', encoding='utf-8') as f:
                json.dump(enhanced_ckg, f, ensure_ascii=False, indent=2, default=str)
            print(f"   💾 增强后的CKG已保存: {ckg_file}")
        except Exception as e:
            print(f"   ⚠️  保存增强CKG失败: {e}")
        
        print(f"✅ CKG增强完成: 添加了 {len(hidden_factors)} 个隐藏因子节点")
        return enhanced_ckg
    

    


    def process_ground_truth_to_evidence_zonesv0(self, ground_truth_file: str, 
                                             ckg_file: str, 
                                             output_file: str = None) -> Dict[str, Any]:
        """
        将Ground Truth处理为结构化证据区域
        
        Args:
            ground_truth_file: Ground Truth JSON文件路径
            ckg_file: 因果知识图谱JSON文件路径  
            output_file: 输出的处理后GT文件路径
        
        Returns:
            处理后的结构化证据区域数据
        """
        print(f"开始处理Ground Truth到证据区域...")
        
        with open(ground_truth_file, 'r', encoding='utf-8') as f:
            ground_truth = json.load(f)
        
        with open(ckg_file, 'r', encoding='utf-8') as f:
            ckg = json.load(f)
        
        node_mappings = self._build_node_mappings(ckg)
        
        evidence_zones = {}
        processed_targets = []
        
        for i, gt_target in enumerate(ground_truth.get('ground_truth_targets', [])):
            zone_id = f"evidence_zone_{i}"
            
            evidence_zone = self._create_evidence_zone(
                gt_target, node_mappings, ckg, zone_id
            )
            
            evidence_zones[zone_id] = evidence_zone
            processed_targets.append({
                'original_target': gt_target,
                'evidence_zone_id': zone_id,
                'evidence_zone': evidence_zone
            })
        
        processed_gt = {
            'case_id': ground_truth.get('case_id'),
            'dataset_metadata': ground_truth.get('dataset_metadata'),
            'evaluation_metrics': ground_truth.get('evaluation_metrics'),
            'original_ground_truth_targets': ground_truth.get('ground_truth_targets'),
            
            'evidence_zones': evidence_zones,
            'processed_targets': processed_targets,
            
            'evaluation_config': {
                'hit_threshold': 2,  # 命中阈值：至少需要发现2个证据
                'evidence_types': ['wrong_edge', 'missing_edge'],
                'scoring_method': 'weighted_evidence',
                'quality_weights': {
                    'high_confidence_evidence': 1.0,
                    'medium_confidence_evidence': 0.7,
                    'low_confidence_evidence': 0.4
                }
            },
            
            'validation_guide': self._generate_validation_guide(evidence_zones),
            
            'processing_metadata': {
                'processor_version': 'v1.0',
                'processing_timestamp': pd.Timestamp.now().isoformat(),
                'total_evidence_zones': len(evidence_zones),
                'total_evidence_edges': sum(len(zone.get('evidence_edges', [])) 
                                          for zone in evidence_zones.values())
            }
        }
        
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(processed_gt, f, ensure_ascii=False, indent=2, default=str)
            
            print(f"✓ 处理后的Ground Truth已保存: {output_path}")
        
        print(f"✓ 证据区域处理完成: {len(evidence_zones)}个证据区域")
        return processed_gt
    
    def _build_node_mappings(self, ckg: Dict) -> Dict[str, Any]:
        """构建节点映射关系"""
        node_id_to_text = {}
        node_text_to_id = {}
        node_types = {}
        
        for node_type, nodes in ckg.get('nodes_by_type', {}).items():
            for node in nodes:
                node_id = node['id']
                node_text = node['text']
                
                node_id_to_text[node_id] = node_text
                node_text_to_id[node_text] = node_id
                node_types[node_id] = node_type
        
        return {
            'id_to_text': node_id_to_text,
            'text_to_id': node_text_to_id,
            'node_types': node_types
        }
    
    def _create_evidence_zone(self, gt_target: Dict, node_mappings: Dict, 
                            ckg: Dict, zone_id: str) -> Dict[str, Any]:
        """为GT盲区创建证据区域"""
        blind_spot_type = gt_target.get('blind_spot_type')
        target_nodes = gt_target.get('target_nodes', [])
        expected_findings = gt_target.get('expected_findings', {})
        
        evidence_zone = {
            'zone_id': zone_id,
            'blind_spot_type': blind_spot_type,
            'description': gt_target.get('description', ''),
            'severity': gt_target.get('severity', 0.5),
            'target_nodes': target_nodes,
            'evidence_edges': [],
            'target_regions': {},
            'detection_criteria': {}
        }
        
        if blind_spot_type == 'confounded_relations':
            evidence_zone = self._create_confounded_relations_zone(
                evidence_zone, expected_findings, node_mappings, ckg
            )
        
        elif blind_spot_type == 'causal_chain_break':
            evidence_zone = self._create_causal_chain_break_zone(
                evidence_zone, expected_findings, node_mappings, ckg
            )
        
        elif blind_spot_type == 'causal_desert':
            evidence_zone = self._create_causal_desert_zone(
                evidence_zone, expected_findings, node_mappings, ckg
            )
        
        elif blind_spot_type == 'tacit_knowledge_gaps':
            evidence_zone = self._create_tacit_knowledge_zone(
                evidence_zone, expected_findings, node_mappings, ckg
            )
        
        return evidence_zone
    
    def _create_confounded_relations_zone(self, evidence_zone: Dict, 
                                        expected_findings: Dict, 
                                        node_mappings: Dict, 
                                        ckg: Dict) -> Dict[str, Any]:
        """创建混杂关系证据区域"""
        spurious_relation = expected_findings.get('spurious_relation', {})
        hidden_confounder = expected_findings.get('hidden_confounder', '')
        
        source_node = spurious_relation.get('source_node', '')
        target_node = spurious_relation.get('target_node', '')
        
        evidence_edges = []
        
        if source_node and target_node:
            evidence_edges.append({
                'edge_key': f"{source_node} -> {target_node}",
                'edge_type': 'wrong_edge',
                'expected_change': 'decrease',  # 概率应该下降
                'importance': 'high',
                'reason': f'虚假关联：受{hidden_confounder}混杂'
            })
        
        if hidden_confounder:
            if source_node:
                evidence_edges.append({
                    'edge_key': f"{hidden_confounder} -> {source_node}",
                    'edge_type': 'missing_edge', 
                    'expected_change': 'increase',  # 概率应该上升
                    'importance': 'high',
                    'reason': f'混杂因子{hidden_confounder}对{source_node}的真实影响'
                })
            
            if target_node:
                evidence_edges.append({
                    'edge_key': f"{hidden_confounder} -> {target_node}",
                    'edge_type': 'missing_edge',
                    'expected_change': 'increase',
                    'importance': 'high', 
                    'reason': f'混杂因子{hidden_confounder}对{target_node}的真实影响'
                })
        
        evidence_zone['evidence_edges'] = evidence_edges
        evidence_zone['detection_criteria'] = {
            'spurious_correlation_threshold': 0.85,
            'partial_correlation_threshold': 0.2,
            'simpson_paradox_indicator': True,
            'expected_evidence_count': len(evidence_edges),
            'min_evidence_for_detection': 2
        }
        
        evidence_zone['target_regions'] = {
            'spurious_edge_region': [f"{source_node} -> {target_node}"],
            'confounder_region': [f"{hidden_confounder} -> {source_node}", 
                                f"{hidden_confounder} -> {target_node}"]
        }
        
        return evidence_zone
    
    def _create_causal_chain_break_zone(self, evidence_zone: Dict,
                                      expected_findings: Dict,
                                      node_mappings: Dict,
                                      ckg: Dict) -> Dict[str, Any]:
        """创建因果断链证据区域"""
        chain_nodes = expected_findings.get('chain_nodes', [])
        instability_indicators = expected_findings.get('instability_indicators', {})
        
        evidence_edges = []
        
        for i in range(len(chain_nodes) - 1):
            source_node = chain_nodes[i]
            target_node = chain_nodes[i + 1]
            
            evidence_edges.append({
                'edge_key': f"{source_node} -> {target_node}",
                'edge_type': 'wrong_edge',  # 不稳定的链被视为错边
                'expected_change': 'decrease',  # 置信度应该下降
                'importance': 'high',
                'reason': f'因果链{source_node}->{target_node}传导不稳定'
            })
        
        evidence_zone['evidence_edges'] = evidence_edges
        evidence_zone['detection_criteria'] = {
            'low_confidence_threshold': 0.5,
            'instability_indicator': 'conditional_dependency',
            'success_rate_range': [0.2, 0.7],
            'expected_evidence_count': len(evidence_edges),
            'min_evidence_for_detection': max(1, len(evidence_edges) // 2)
        }
        
        evidence_zone['target_regions'] = {
            'unstable_chain_region': [f"{chain_nodes[i]} -> {chain_nodes[i+1]}" 
                                    for i in range(len(chain_nodes) - 1)]
        }
        
        return evidence_zone
    
    def _create_causal_desert_zone(self, evidence_zone: Dict,
                                 expected_findings: Dict,
                                 node_mappings: Dict,
                                 ckg: Dict) -> Dict[str, Any]:
        """创建因果荒漠证据区域"""
        isolated_node = expected_findings.get('isolated_node', '')
        missing_edges = expected_findings.get('missing_edges', [])
        data_correlations = expected_findings.get('data_correlations', {})
        
        evidence_edges = []
        
        for missing_edge in missing_edges:
            source_id = missing_edge.get('source_id')
            target_id = missing_edge.get('target_id')
            
            source_text = node_mappings['id_to_text'].get(source_id, source_id)
            target_text = node_mappings['id_to_text'].get(target_id, target_id)
            
            evidence_edges.append({
                'edge_key': f"{source_text} -> {target_text}",
                'edge_type': 'missing_edge',
                'expected_change': 'increase',  # 缺失边的概率应该上升
                'importance': 'high',
                'reason': f'荒漠节点{isolated_node}的缺失连接'
            })
        
        for corr_node, corr_strength in data_correlations.items():
            if corr_strength > 0.7:  # 强相关性
                evidence_edges.append({
                    'edge_key': f"{isolated_node} -> {corr_node}",
                    'edge_type': 'missing_edge',
                    'expected_change': 'increase',
                    'importance': 'high',
                    'reason': f'强数据相关性({corr_strength:.2f})但图结构缺失'
                })
        
        evidence_zone['evidence_edges'] = evidence_edges
        evidence_zone['detection_criteria'] = {
            'data_correlation_threshold': 0.7,
            'graph_connectivity_threshold': 2,
            'isolation_indicator': 'low_degree_high_correlation',
            'expected_evidence_count': len(evidence_edges),
            'min_evidence_for_detection': max(1, len(evidence_edges) // 2)
        }
        
        evidence_zone['target_regions'] = {
            'desert_center': [isolated_node],
            'missing_connections_region': [edge['edge_key'] for edge in evidence_edges]
        }
        
        return evidence_zone
    
    def _create_tacit_knowledge_zone(self, evidence_zone: Dict,
                                   expected_findings: Dict,
                                   node_mappings: Dict,
                                   ckg: Dict) -> Dict[str, Any]:
        """创建隐性知识缺口证据区域"""
        
        evidence_edges = []
        
        
        
        evidence_zone['evidence_edges'] = evidence_edges
        evidence_zone['detection_criteria'] = {
            'complexity_gap_threshold': 0.5,
            'expert_vs_system_performance_gap': 0.3,
            'multi_step_logic_indicator': True,
            'expected_evidence_count': len(evidence_edges),
            'min_evidence_for_detection': 1
        }
        
        return evidence_zone
    
    def _generate_validation_guide(self, evidence_zones: Dict) -> Dict[str, Any]:
        """生成验证指南"""
        guide = {
            'total_zones': len(evidence_zones),
            'validation_steps': [
                "1. 加载BCSA分析结果中的显著证据",
                "2. 对每个证据区域计算命中率",
                "3. 判断整体盲区召回率",
                "4. 生成诊断对比仪表盘"
            ],
            'zone_summaries': {}
        }
        
        for zone_id, zone in evidence_zones.items():
            guide['zone_summaries'][zone_id] = {
                'type': zone['blind_spot_type'],
                'evidence_count': len(zone.get('evidence_edges', [])),
                'detection_difficulty': 'hard',  # 默认为困难
                'key_indicators': [edge.get('reason', '') for edge in zone.get('evidence_edges', [])]
            }
        
        return guide

    def batch_process_all_cases(self, base_data_dir: str = "./seek_data_v3_deep_enhanced/cases") -> Dict[str, Any]:
        """
        批量处理所有案例的Ground Truth
        
        Args:
            base_data_dir: 案例数据根目录路径
        
        Returns:
            处理结果统计报告
        """
        print("=" * 80)
        print("🚀 开始批量处理所有案例的Ground Truth到证据区域")
        print("=" * 80)
        
        scales = ['smallcase', 'mediumcase', 'bigcase']
        methods = ['BCSA', 'CEDA', 'Mixed']
        
        processing_stats = {
            'total_cases': 0,
            'successful_cases': 0,
            'failed_cases': 0,
            'by_scale': {scale: {'total': 0, 'success': 0, 'failed': 0} for scale in scales},
            'by_method': {method: {'total': 0, 'success': 0, 'failed': 0} for method in methods},
            'failed_cases_detail': [],
            'processing_summary': {}
        }
        
        base_path = Path(base_data_dir)
        if not base_path.exists():
            print(f"❌ 数据目录不存在: {base_data_dir}")
            return processing_stats
        
        print(f"📂 数据根目录: {base_path.absolute()}")
        
        for scale in scales:
            scale_path = base_path / scale
            if not scale_path.exists():
                print(f"⚠️  规模目录不存在: {scale_path}")
                continue
                
            print(f"\n📊 处理规模: {scale.upper()}")
            
            for method in methods:
                method_path = scale_path / method
                if not method_path.exists():
                    print(f"⚠️  方法目录不存在: {method_path}")
                    continue
                
                print(f"  🔧 处理方法: {method}")
                
                case_dirs = [d for d in method_path.iterdir() if d.is_dir()]
                case_dirs.sort()  # 按名称排序
                
                print(f"    📁 发现 {len(case_dirs)} 个案例目录")
                
                for case_dir in case_dirs:
                    case_id = case_dir.name
                    processing_stats['total_cases'] += 1
                    processing_stats['by_scale'][scale]['total'] += 1
                    processing_stats['by_method'][method]['total'] += 1
                    
                    print(f"      🔄 处理案例: {case_id}")
                    
                    gt_file = case_dir / "ground_truth.json"
                    ckg_file = case_dir / "causal_knowledge_graph.json"
                    output_file = case_dir / "processed_ground_truth.json"
                    
                    if not gt_file.exists():
                        print(f"        ❌ Ground Truth文件不存在: {gt_file}")
                        self._record_failed_case(processing_stats, case_id, scale, method, "Ground Truth文件不存在")
                        continue
                    
                    if not ckg_file.exists():
                        print(f"        ❌ CKG文件不存在: {ckg_file}")
                        self._record_failed_case(processing_stats, case_id, scale, method, "CKG文件不存在")
                        continue
                    
                    try:
                        processed_gt = self.process_ground_truth_to_evidence_zones(
                            str(gt_file), str(ckg_file), str(output_file)
                        )
                        
                        processing_stats['successful_cases'] += 1
                        processing_stats['by_scale'][scale]['success'] += 1
                        processing_stats['by_method'][method]['success'] += 1
                        
                        processing_stats['processing_summary'][case_id] = {
                            'scale': scale,
                            'method': method,
                            'evidence_zones_count': len(processed_gt.get('evidence_zones', {})),
                            'total_evidence_edges': processed_gt.get('processing_metadata', {}).get('total_evidence_edges', 0),
                            'status': 'success'
                        }
                        
                        print(f"        ✅ 处理成功 - 证据区域: {len(processed_gt.get('evidence_zones', {}))}")
                        
                    except Exception as e:
                        error_msg = str(e)
                        print(f"        ❌ 处理失败: {error_msg}")
                        self._record_failed_case(processing_stats, case_id, scale, method, error_msg)
        
        self._generate_batch_processing_report(processing_stats)
        
        return processing_stats
    
    def _record_failed_case(self, stats: Dict, case_id: str, scale: str, method: str, error_msg: str):
        """记录失败案例"""
        stats['failed_cases'] += 1
        stats['by_scale'][scale]['failed'] += 1
        stats['by_method'][method]['failed'] += 1
        stats['failed_cases_detail'].append({
            'case_id': case_id,
            'scale': scale,
            'method': method,
            'error': error_msg
        })
        stats['processing_summary'][case_id] = {
            'scale': scale,
            'method': method,
            'status': 'failed',
            'error': error_msg
        }
    
    def _generate_batch_processing_report(self, stats: Dict):
        """生成批量处理报告"""
        print("\n" + "=" * 80)
        print("📋 批量处理完成报告")
        print("=" * 80)
        
        print(f"📊 总体统计:")
        print(f"   总案例数: {stats['total_cases']}")
        print(f"   成功处理: {stats['successful_cases']} ({stats['successful_cases']/max(stats['total_cases'],1)*100:.1f}%)")
        print(f"   处理失败: {stats['failed_cases']} ({stats['failed_cases']/max(stats['total_cases'],1)*100:.1f}%)")
        
        print(f"\n📈 按规模统计:")
        for scale, scale_stats in stats['by_scale'].items():
            if scale_stats['total'] > 0:
                success_rate = scale_stats['success'] / scale_stats['total'] * 100
                print(f"   {scale.upper():12} - 总计: {scale_stats['total']:2d}, 成功: {scale_stats['success']:2d}, 失败: {scale_stats['failed']:2d} ({success_rate:.1f}%)")
        
        print(f"\n🔧 按方法统计:")
        for method, method_stats in stats['by_method'].items():
            if method_stats['total'] > 0:
                success_rate = method_stats['success'] / method_stats['total'] * 100
                print(f"   {method:5} - 总计: {method_stats['total']:2d}, 成功: {method_stats['success']:2d}, 失败: {method_stats['failed']:2d} ({success_rate:.1f}%)")
        
        if stats['failed_cases'] > 0:
            print(f"\n❌ 失败案例详情:")
            for failed in stats['failed_cases_detail'][:10]:  # 只显示前10个
                print(f"   {failed['case_id']} ({failed['scale']}/{failed['method']}): {failed['error'][:50]}...")
            
            if len(stats['failed_cases_detail']) > 10:
                print(f"   ... 还有 {len(stats['failed_cases_detail']) - 10} 个失败案例")
        
        total_zones = sum(
            summary.get('evidence_zones_count', 0) 
            for summary in stats['processing_summary'].values() 
            if summary.get('status') == 'success'
        )
        total_edges = sum(
            summary.get('total_evidence_edges', 0) 
            for summary in stats['processing_summary'].values() 
            if summary.get('status') == 'success'
        )
        
        print(f"\n🎯 证据区域统计:")
        print(f"   总证据区域数: {total_zones}")
        print(f"   总证据边数: {total_edges}")
        if stats['successful_cases'] > 0:
            print(f"   平均每案例区域数: {total_zones/stats['successful_cases']:.1f}")
            print(f"   平均每案例证据边数: {total_edges/stats['successful_cases']:.1f}")
        
        print("\n✅ 批量处理报告完成!")

    def scan_available_cases(self, base_data_dir: str = "./seek_data_v3_deep_enhanced/cases") -> Dict[str, Any]:
        """
        扫描可用案例，生成目录结构报告
        
        Args:
            base_data_dir: 案例数据根目录路径
        
        Returns:
            案例目录结构统计
        """
        print("🔍 扫描可用案例...")
        
        base_path = Path(base_data_dir)
        if not base_path.exists():
            print(f"❌ 数据目录不存在: {base_data_dir}")
            return {}
        
        scan_result = {
            'base_dir': str(base_path.absolute()),
            'scales': {},
            'total_cases': 0,
            'total_complete_cases': 0  # 同时拥有GT和CKG文件的案例
        }
        
        scales = ['smallcase', 'mediumcase', 'bigcase']
        methods = ['BCSA', 'CEDA', 'Mixed']
        
        for scale in scales:
            scale_path = base_path / scale
            if not scale_path.exists():
                continue
                
            scan_result['scales'][scale] = {'methods': {}, 'total': 0, 'complete': 0}
            
            for method in methods:
                method_path = scale_path / method
                if not method_path.exists():
                    continue
                
                case_dirs = [d for d in method_path.iterdir() if d.is_dir()]
                case_info = []
                complete_count = 0
                
                for case_dir in sorted(case_dirs):
                    gt_exists = (case_dir / "ground_truth.json").exists()
                    ckg_exists = (case_dir / "causal_knowledge_graph.json").exists()
                    processed_exists = (case_dir / "processed_ground_truth.json").exists()
                    
                    is_complete = gt_exists and ckg_exists
                    if is_complete:
                        complete_count += 1
                    
                    case_info.append({
                        'name': case_dir.name,
                        'path': str(case_dir),
                        'has_gt': gt_exists,
                        'has_ckg': ckg_exists,
                        'has_processed_gt': processed_exists,
                        'is_complete': is_complete
                    })
                
                scan_result['scales'][scale]['methods'][method] = {
                    'total_cases': len(case_dirs),
                    'complete_cases': complete_count,
                    'cases': case_info
                }
                
                scan_result['scales'][scale]['total'] += len(case_dirs)
                scan_result['scales'][scale]['complete'] += complete_count
        
        for scale_info in scan_result['scales'].values():
            scan_result['total_cases'] += scale_info['total']
            scan_result['total_complete_cases'] += scale_info['complete']
        
        print(f"\n📂 案例目录扫描结果:")
        print(f"   根目录: {scan_result['base_dir']}")
        print(f"   总案例数: {scan_result['total_cases']}")
        print(f"   完整案例数: {scan_result['total_complete_cases']} (同时拥有GT和CKG)")
        
        for scale, scale_info in scan_result['scales'].items():
            print(f"\n   📊 {scale.upper()}:")
            print(f"      总计: {scale_info['total']} 案例, 完整: {scale_info['complete']} 案例")
            
            for method, method_info in scale_info['methods'].items():
                print(f"        🔧 {method}: {method_info['total_cases']} 案例 ({method_info['complete_cases']} 完整)")
        
        return scan_result


def process_single_case_example():
    """处理单个案例的示例"""
    processor = EvidenceZoneProcessor()
    
    case_dir = "./seek_data_v3_deep_enhanced/cases/smallcase/Mixed/Mixed_small_04"
    
    ground_truth_file = f"{case_dir}/ground_truth.json"
    ckg_file = f"{case_dir}/causal_knowledge_graph.json"
    output_file = f"{case_dir}/processed_ground_truth.json"
    
    try:
        processed_gt = processor.process_ground_truth_to_evidence_zones(
            ground_truth_file, ckg_file, output_file
        )
        
        print(f"✅ 证据区域处理完成")
        print(f"   - 证据区域数量: {len(processed_gt['evidence_zones'])}")
        print(f"   - 总证据边数: {processed_gt['processing_metadata']['total_evidence_edges']}")
        
        return processed_gt
        
    except Exception as e:
        print(f"❌ 处理失败: {e}")
        return None


def main():
    """主函数 - 批量处理所有案例"""
    import pandas as pd  # 用于时间戳
    
    processor = EvidenceZoneProcessor()
    
    print("第一步：扫描可用案例")
    scan_result = processor.scan_available_cases()
    
    if scan_result['total_complete_cases'] == 0:
        print("❌ 没有发现完整的案例（需要同时有GT和CKG文件）")
        print("请检查数据目录结构和文件完整性")
        return
    
    print(f"\n发现 {scan_result['total_complete_cases']} 个完整案例可供处理")
    
    
    print("\n第二步：执行批量处理")
    processing_stats = processor.batch_process_all_cases()
    
    report_file = "./batch_processing_report.json"
    try:
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump({
                'scan_result': scan_result,
                'processing_stats': processing_stats,
                'timestamp': pd.Timestamp.now().isoformat()
            }, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n📄 详细报告已保存至: {report_file}")
    except Exception as e:
        print(f"⚠️  保存报告失败: {e}")
    
    print("\n🎉 全部处理完成!")


if __name__ == "__main__":
    main()
