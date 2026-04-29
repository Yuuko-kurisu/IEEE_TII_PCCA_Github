#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BCSA审计流程编排器（重构版）
pipeline
负责：连接所有模块，编排完整的审计流程
职责：单一职责 - 流程编排和最终报告生成
"""

import os
import sys
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
import logging
from collections import defaultdict

from BCSA_00_Shared_Structures import (
    Hypothesis, EdgeUncertaintyResult, QualitativeEvaluationReport,
    AggregatedUncertaintyResult, convert_numpy_types
)

from BCSA_01_Hypothesis_Generator import run_hypothesis_generation
from BCSA_02_PCVGAE import run_conditioned_uncertainty_analysis
from BCSA_02_PCGATE import run_conditioned_uncertainty_analysis_PCGATE
from BCSA_03_Quantitative_Evaluator import evaluate_case, print_evaluation_summary

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


try:
    from BCSA_03_Qualitative_Evaluation import generate_qualitative_report
    QUALITATIVE_AVAILABLE = True
except ImportError:
    logger.warning("定性报告生成器不可用，将跳过定性分析步骤")
    QUALITATIVE_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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

class AuditPipelineOrchestrator:
    """审计流程编排器"""
    
    def __init__(self, base_output_dir: Path = None):
        if base_output_dir is None:
            self.base_output_dir = Path("./seek_data_v3_deep_enhanced/results")
        else:
            self.base_output_dir = base_output_dir
        logger.info("审计流程编排器初始化完成")
    
    def create_output_structure(self, case_scale: str, case_type: str, case_id: str, method_name: str) -> Dict[str, Path]:
        """创建标准的输出目录结构，反映输入层级和方法名称"""
        case_output_dir = self.base_output_dir / case_scale / case_type / case_id / f"{method_name}_Analysis"
        
        output_dirs = {
            'main': case_output_dir,
            'hypothesis_generation': case_output_dir / "1_Final_Results" / "hypothesis_generation",
            'conditioned_uncertainty': case_output_dir / "1_Final_Results" / f"pc_{method_name.lower()}_audit",
            'qualitative_evaluation': case_output_dir / "1_Final_Results" / "evaluation",
            'case_validation': case_output_dir / "1_Final_Results" / "case_validation",
            'visualizations': case_output_dir / "2_Result_Visualizations",
            'process_visualizations': case_output_dir / "3_Process_Visualizations",
            'metrics': case_output_dir / "4_Evaluation_Metrics"
        }
        
        for dir_path in output_dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"创建输出目录结构: {case_output_dir}")
        return output_dirs
    
    def run_complete_audit_pipeline(self, case_dir: Path, method_choice: str = 'PCGATE',
                                    aggregation_method: str = 'weighted_max',
                                    mc_samples: int = 50) -> Dict[str, Any]:
        """
        运行完整审计流程
        
        Args:
            case_dir: 案例目录路径
            method_choice: 选择的方法 ('PCVGAE', 'PCGATE', 'BOTH')
            aggregation_method: 聚合方法
            mc_samples: 蒙特卡洛采样次数
        """
        case_dir = Path(case_dir)
        case_id = case_dir.name
        case_type = case_dir.parent.name
        case_scale = case_dir.parent.parent.name

        logger.info(f"开始完整审计流程: scale='{case_scale}', type='{case_type}', case='{case_id}', method='{method_choice}'")
        
        if method_choice == 'BOTH':
            pcvgae_result = self._run_single_method_pipeline(
                case_dir, case_scale, case_type, case_id, 'PCVGAE', aggregation_method, mc_samples
            )
            pcgate_result = self._run_single_method_pipeline(
                case_dir, case_scale, case_type, case_id, 'PCGATE', aggregation_method, mc_samples
            )
            
            pcvgae_status = pcvgae_result.get('pipeline_status', 'failed')
            pcgate_status = pcgate_result.get('pipeline_status', 'failed')
            
            if pcvgae_status == 'completed' and pcgate_status == 'completed':
                overall_status = 'completed'
            elif pcvgae_status == 'completed' or pcgate_status == 'completed':
                overall_status = 'partially_completed'
            else:
                overall_status = 'failed'
            
            pcvgae_quant = pcvgae_result.get('quantitative_evaluation', {})
            pcgate_quant = pcgate_result.get('quantitative_evaluation', {})
            
            if pcvgae_quant or pcgate_quant:
                merged_quant = {
                    'method_used': 'BOTH',
                    'selection_policy': 'separate_results_only',
                    'status': 'separate_results_only',
                    'PCVGAE': pcvgae_quant,
                    'PCGATE': pcgate_quant
                }
            else:
                merged_quant = {'method_used': 'BOTH', 'status': 'no_results'}
            
            return {
                'combined_results': True,
                'method_choice': method_choice,
                'case_id': case_id,
                'pipeline_status': overall_status,  # 添加整体状态
                'PCVGAE_result': pcvgae_result,
                'PCGATE_result': pcgate_result,
                'timestamp': datetime.now().isoformat(),
                'hypothesis_generation': pcvgae_result.get('hypothesis_generation', pcgate_result.get('hypothesis_generation', {})),
                'quantitative_evaluation': merged_quant,
                'qualitative_evaluation': pcvgae_result.get('qualitative_evaluation', pcgate_result.get('qualitative_evaluation', {}))
            }
        else:
            return self._run_single_method_pipeline(
                case_dir, case_scale, case_type, case_id, method_choice, aggregation_method, mc_samples
            )
    
    def _run_single_method_pipeline(self, case_dir: Path, case_scale: str, case_type: str, 
                                   case_id: str, method_name: str, aggregation_method: str, 
                                   mc_samples: int) -> Dict[str, Any]:
        """运行单个方法的完整流程"""
        output_dirs = self.create_output_structure(case_scale, case_type, case_id, method_name)
        
        audit_results = {
            'case_id': case_id,
            'method_name': method_name,
            'case_dir': str(case_dir),
            'output_dirs': {k: str(v) for k, v in output_dirs.items()},
            'timestamp': datetime.now().isoformat(),
            'config': {
                'aggregation_method': aggregation_method,
                'mc_samples': mc_samples
            },
            'pipeline_status': 'running'
        }
        
        try:
            logger.info(f"🎯 步骤1: 运行假设生成模块 (方法: {method_name})")
            logger.info("-" * 40)

            hypothesis_result = run_hypothesis_generation(
                case_dir=case_dir,
                output_dir=output_dirs['hypothesis_generation'],
                enable_coverage_validation=False
            )



            if isinstance(hypothesis_result, dict):
                hypotheses = hypothesis_result.get('hypotheses_in_memory', [])
                if not hypotheses: # 作为备用方案，如果内存对象为空，再尝试从文件加载
                    hypotheses = self._load_hypotheses_from_files(hypothesis_result.get('output_files', {}))
            else:
                hypotheses = hypothesis_result

            if not hypotheses:
                logger.warning("未生成任何假设，审计流程提前结束")
                audit_results['pipeline_status'] = 'failed'
                audit_results['error'] = 'no_hypotheses_generated'
                return audit_results
            
            hypotheses_count = len(hypotheses)
            audit_results['hypothesis_generation'] = {
                'status': 'success',
                'hypotheses_count': hypotheses_count,
                'hypothesis_types': list(set(h.hypothesis_type for h in hypotheses)) if hypotheses else [],
                'avg_confidence': float(sum(h.confidence_score for h in hypotheses) / len(hypotheses)) if hypotheses else 0.0,
                'output_dir': str(output_dirs['hypothesis_generation'])
            }



            logger.info(f"✅ 假设生成完成: {len(hypotheses)} 个假设")
            
            logger.info(f"\n🔬 步骤2: 运行假设条件化不确定性分析模块 (方法: {method_name})")
            logger.info("-" * 40)
            
            if method_name == 'PCVGAE':
                uncertainty_result = run_conditioned_uncertainty_analysis(
                    hypotheses=hypotheses,
                    case_dir=case_dir,
                    output_dir=output_dirs['conditioned_uncertainty'],
                    aggregation_method=aggregation_method,
                    mc_samples=mc_samples
                )
            elif method_name == 'PCGATE':
                uncertainty_result = run_conditioned_uncertainty_analysis_PCGATE(
                    hypotheses=hypotheses,
                    case_dir=case_dir,
                    output_dir=output_dirs['conditioned_uncertainty'],
                    aggregation_method=aggregation_method,
                    mc_samples=mc_samples
                )
            else:
                raise ValueError(f"未知的方法选择: {method_name}")
            
            audit_results['conditioned_uncertainty_analysis'] = {
                'status': 'success',
                'aggregated_edges_count': len(uncertainty_result.aggregated_uncertainty_map),
                'high_priority_findings_count': len(uncertainty_result.high_priority_findings),
                'trained_models': len(uncertainty_result.training_results),
                'converged_models': len([r for r in uncertainty_result.training_results.values() if r.converged]),
                'aggregation_method': uncertainty_result.aggregation_method,
                'output_dir': str(output_dirs['conditioned_uncertainty'])
            }
            
            logger.info(f"✅ 假设条件化不确定性分析完成: {len(uncertainty_result.aggregated_uncertainty_map)} 个边")
            
            logger.info(f"\n📊 步骤3: 运行定量评价模块 (方法: {method_name})")
            logger.info("-" * 40)
            
            quantitative_result = evaluate_case(
                case_id=case_id,
                case_directory=str(output_dirs['conditioned_uncertainty']),
                output_file=str(output_dirs['qualitative_evaluation'] / "quantitative_evaluation_report.json"),
                use_legacy_matching=False,
                base_data_dir=str(case_dir.parent.parent.parent)
            )

            print_evaluation_summary(quantitative_result)
            audit_results['quantitative_evaluation'] = quantitative_result.get('quantitative_metrics', {})
            audit_results['quantitative_evaluation']['status'] = 'success'
            audit_results['quantitative_evaluation']['output_dir'] = str(output_dirs['conditioned_uncertainty'])
            
            logger.info(f"✅ 定量评价完成: F1分数 {quantitative_result['quantitative_metrics']['performance_metrics']['f1_score']:.1%}")
            
            if QUALITATIVE_AVAILABLE:
                logger.info(f"\n📝 步骤4: 生成定性分析报告 (方法: {method_name})")
                logger.info("-" * 40)
                
                try:
                    qualitative_report_content = generate_qualitative_report(
                        case_id=case_id,
                        case_directory=str(output_dirs['conditioned_uncertainty']),
                        output_file=str(output_dirs['qualitative_evaluation'])
                    )
                    
                    audit_results['qualitative_evaluation'] = {
                        'status': 'success',
                        'report_generated': True,
                        'report_length': len(qualitative_report_content.split('\n')) if qualitative_report_content else 0,
                        'output_dir': str(output_dirs['qualitative_evaluation'])
                    }
                    
                    logger.info(f"✅ 定性分析报告生成完成")
                except Exception as e:
                    logger.warning(f"定性分析报告生成失败: {e}")
                    audit_results['qualitative_evaluation'] = {
                        'status': 'failed',
                        'error': str(e),
                        'output_dir': str(output_dirs['qualitative_evaluation'])
                    }
            else:
                logger.info("\n⚠️ 跳过步骤4: 定性分析报告生成器不可用")
                audit_results['qualitative_evaluation'] = {
                    'status': 'skipped',
                    'reason': 'module_not_available'
                }
            
            logger.info(f"\n📋 步骤5: 生成综合审计报告 (方法: {method_name})")
            logger.info("-" * 40)
            
            comprehensive_report_path = self._generate_comprehensive_report(
                audit_results, hypotheses, uncertainty_result, quantitative_result, output_dirs['main']
            )
            
            audit_results['comprehensive_report_path'] = str(comprehensive_report_path)
            audit_results['pipeline_status'] = 'completed'
            
            final_results_file = output_dirs['main'] / f"{case_id}_{method_name}_complete_audit_results.json"
            with open(final_results_file, 'w', encoding='utf-8') as f:
                json.dump(convert_numpy_types(audit_results), f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ 完整审计流程成功完成! (方法: {method_name})")
            logger.info(f"📁 主要输出:")
            logger.info(f"   - 综合报告: {comprehensive_report_path}")
            logger.info(f"   - 完整结果: {final_results_file}")
            logger.info(f"   - 输出目录: {output_dirs['main']}")
            
            return audit_results
            
        except Exception as e:
            logger.error(f"审计流程失败 (方法: {method_name}): {e}")
            import traceback
            audit_results.update({
                'pipeline_status': 'failed',
                'error': str(e),
                'traceback': traceback.format_exc()
            })
            
            try:
                failed_results_file = output_dirs['main'] / f"{case_id}_{method_name}_failed_audit_results.json"
                with open(failed_results_file, 'w', encoding='utf-8') as f:
                    json.dump(convert_numpy_types(audit_results), f, indent=2, ensure_ascii=False)
                logger.info(f"失败结果已保存: {failed_results_file}")
            except:
                pass
            
            return audit_results
    def _calculate_node_uncertainties_from_edges(self, aggregated_uncertainty_map: List[EdgeUncertaintyResult]) -> List[Dict[str, Any]]:
        """
        基于边的不确定性计算节点不确定性
        
        Args:
            aggregated_uncertainty_map: 聚合后的边不确定性结果列表
            
        Returns:
            节点不确定性列表，每个元素包含node_id和uncertainty_score
        """
        node_edge_uncertainties = defaultdict(list)
        
        for edge_result in aggregated_uncertainty_map:
            score = abs(edge_result.uncertainty_score)
            
            node_edge_uncertainties[edge_result.source_id].append(score)
            node_edge_uncertainties[edge_result.target_id].append(score)
        
        result = []
        
        for node_id, scores in node_edge_uncertainties.items():
            if scores:  # 确保有分数可计算
                uncertainty_score = sum(scores) / len(scores)
                
                result.append({
                    'node_id': node_id,
                    'uncertainty_score': uncertainty_score,
                    'edge_count': len(scores),  # 记录关联的边数量
                    'max_edge_uncertainty': max(scores),  # 记录最大边不确定性
                    'min_edge_uncertainty': min(scores)   # 记录最小边不确定性
                })
        
        result.sort(key=lambda x: x['uncertainty_score'], reverse=True)
        
        logger.info(f"基于边计算了 {len(result)} 个节点的不确定性")
        if result:
            top_uncertain = result[:5]
            top_nodes_str = [f"{n['node_id']} ({n['uncertainty_score']:.3f})" for n in top_uncertain]
            logger.info(f"前5个最不确定的节点: {top_nodes_str}")    
        
        return result

    def _load_hypotheses_from_files(self, output_files: Dict[str, str]) -> List[Hypothesis]:
        """从输出文件中加载假设对象列表"""
        hypotheses = []

        hypothesis_file = output_files.get('hypotheses_primary') or output_files.get('hypotheses_official')

        if hypothesis_file and Path(hypothesis_file).exists():
            try:
                with open(hypothesis_file, 'r', encoding='utf-8') as f:
                    hypotheses_data = json.load(f)

                for hyp_data in hypotheses_data:
                    hypothesis = Hypothesis(
                        id=hyp_data.get('id', ''),
                        rule_name=hyp_data.get('rule_name', ''),
                        rule_category=hyp_data.get('rule_category', ''),
                        hypothesis_type=hyp_data.get('hypothesis_type', ''),
                        description=hyp_data.get('description', ''),
                        target_elements=hyp_data.get('target_elements', []),
                        evidence=hyp_data.get('evidence', {}),
                        confidence_score=hyp_data.get('confidence_score', 0.0),
                        priority=hyp_data.get('priority', 0.0),
                        metadata=hyp_data.get('metadata', {})
                    )
                    hypotheses.append(hypothesis)

                logger.info(f"从文件加载了 {len(hypotheses)} 个假设对象")

            except Exception as e:
                logger.error(f"加载假设文件失败: {e}")
        else:
            logger.warning(f"假设文件不存在: {hypothesis_file}")

        return hypotheses

    def _generate_comprehensive_report(self, audit_results: Dict[str, Any],
                                    hypotheses: List[Hypothesis],
                                    uncertainty_result: AggregatedUncertaintyResult,
                                    quantitative_result: Dict[str, Any],
                                    output_dir: Path) -> Path:
        """生成综合Markdown审计报告 - 证据为中心的版本"""
        report_content = []
        
        report_content.append(f"# BCSA完整审计报告")
        report_content.append(f"## 案例: {audit_results['case_id']}")
        report_content.append(f"**生成时间**: {audit_results['timestamp']}")
        report_content.append(f"**审计状态**: {audit_results['pipeline_status']}")
        report_content.append(f"**评估体系**: 证据为中心 (Evidence-Centric)")
        report_content.append("")
        
        report_content.append("## 执行摘要")
        
        hyp_gen = audit_results.get('hypothesis_generation', {})
        cond_analysis = audit_results.get('conditioned_uncertainty_analysis', {})
        quant_eval = audit_results.get('quantitative_evaluation', {})
        qual_eval = audit_results.get('qualitative_evaluation', {})
        
        performance_metrics = quant_eval.get('performance_metrics', {})
        weighted_f1 = performance_metrics.get('weighted_f1_score', 0)
        evidence_recall = performance_metrics.get('evidence_recall', 0)
        evidence_mrr = performance_metrics.get('evidence_mrr', 0)
        
        report_content.append(f"- **假设生成**: 生成 {hyp_gen.get('hypotheses_count', 0)} 个假设")
        report_content.append(f"- **条件化分析**: 分析 {cond_analysis.get('aggregated_edges_count', 0)} 条边，发现 {cond_analysis.get('high_priority_findings_count', 0)} 个高优先级异常")
        report_content.append(f"- **证据发现质量**: 加权F1分数 {weighted_f1:.1%}，证据召回率 {evidence_recall:.1%}，证据MRR {evidence_mrr:.3f}")
        report_content.append(f"- **定性分析**: 已生成 {qual_eval.get('report_length', 0)} 行分析报告")
        report_content.append("")
        
        report_content.append("## 关键审计发现")
        
        if uncertainty_result.high_priority_findings:
            report_content.append(f"### 高优先级异常边 (前10个)")
            for i, finding in enumerate(uncertainty_result.high_priority_findings[:10], 1):
                edge_desc = f"{finding.source_id} → {finding.target_id}"
                if hasattr(finding, 'source_text') and hasattr(finding, 'target_text'):
                    edge_desc = f"{finding.source_text} → {finding.target_text}"
                
                report_content.append(f"{i}. **{edge_desc}**")
                report_content.append(f"   - 统一分数: {finding.unified_score:.3f}")
                report_content.append(f"   - 不确定性: {finding.uncertainty_score:.3f}")
                report_content.append(f"   - 类型: {finding.edge_type}")
                report_content.append("")
        else:
            report_content.append("未发现高优先级异常")
            report_content.append("")
        
        if hypotheses:
            report_content.append("### 假设分析摘要")
            from collections import Counter
            type_counts = Counter(h.hypothesis_type for h in hypotheses)
            category_counts = Counter(h.rule_category for h in hypotheses)
            
            report_content.append(f"**假设类别分布**:")
            for category, count in category_counts.items():
                report_content.append(f"- {category}: {count} 个")
            
            report_content.append(f"\n**假设类型分布**:")
            for hyp_type, count in type_counts.most_common():
                report_content.append(f"- {hyp_type}: {count} 个")
            
            report_content.append(f"\n**平均置信度**: {sum(h.confidence_score for h in hypotheses) / len(hypotheses):.3f}")
            report_content.append("")
        
        if uncertainty_result.training_results:
            report_content.append("### 模型训练摘要")
            training_results = uncertainty_result.training_results
            converged_count = len([r for r in training_results.values() if r.converged])
            avg_loss = sum(r.final_loss for r in training_results.values()) / len(training_results)
            
            report_content.append(f"- **训练模型数**: {len(training_results)}")
            report_content.append(f"- **收敛模型数**: {converged_count} ({converged_count/len(training_results):.1%})")
            report_content.append(f"- **平均最终损失**: {avg_loss:.4f}")
            report_content.append(f"- **聚合方法**: {uncertainty_result.aggregation_method}")
            report_content.append("")
        
        if quantitative_result:
            report_content.append("## 详细评价结果 - 证据为中心的评估")
            
            report_content.append("### 核心发现能力")
            report_content.append(f"- **加权F1分数 (WF1)**: {performance_metrics.get('weighted_f1_score', 0):.1%} - 最重要的质量指标")
            report_content.append(f"- **证据召回率 (ESR)**: {performance_metrics.get('evidence_recall', 0):.1%} - 证据覆盖能力")
            report_content.append(f"- **证据精确率 (ESP)**: {performance_metrics.get('evidence_precision', 0):.1%} - 发现精度")
            report_content.append(f"- **传统F1分数 (ESF1)**: {performance_metrics.get('f1_score', 0):.1%} - 对比基准")
            report_content.append("")
            
            report_content.append("### 排序性能")
            report_content.append(f"- **证据MRR (Evidence MRR)**: {performance_metrics.get('evidence_mrr', 0):.3f} - 核心排序指标")
            report_content.append(f"- **全局nDCG@10**: {performance_metrics.get('global_ndcg_at_10', 0):.3f} - 前10排序质量")
            report_content.append(f"- **全局nDCG@20**: {performance_metrics.get('global_ndcg_at_20', 0):.3f} - 前20排序质量")
            
            evidence_mrr = performance_metrics.get('evidence_mrr', 0)
            avg_evidence_rank = 1 / evidence_mrr if evidence_mrr > 0 else float('inf')
            report_content.append(f"- **平均证据排名**: {avg_evidence_rank:.1f}")
            report_content.append("")
            
            basic_metrics = quantitative_result.get('quantitative_metrics', {}).get('basic_metrics', {})
            blindspot_analysis = quantitative_result.get('quantitative_metrics', {}).get('blind_spot_analysis', {})
            
            report_content.append("### 诊断性指标 - 覆盖广度")
            report_content.append(f"- **盲区召回率 (BSR)**: {performance_metrics.get('blind_spot_recall', 0):.1%} - 问题区域覆盖")
            report_content.append(f"- **平均区域证据覆盖率 (ZECR)**: {performance_metrics.get('average_zecr', 0):.1%} - 区域覆盖深度")
            report_content.append(f"- **总证据数**: {basic_metrics.get('total_evidence', 0)}")
            report_content.append(f"- **总发现数**: {basic_metrics.get('total_findings', 0)}")
            report_content.append(f"- **总盲区数**: {blindspot_analysis.get('total_blind_spots', 0)}")
            report_content.append(f"- **已探测盲区**: {blindspot_analysis.get('detected_blind_spots', 0)}")
            report_content.append("")
        
        report_content.append("## 改进建议 - 证据为中心的优化方向")
        
        improvement_suggestions = []
        
        if performance_metrics:
            if performance_metrics.get('weighted_f1_score', 0) < 0.75:
                improvement_suggestions.append("1. **提升发现质量**: 加权F1分数偏低，重点优化重要性证据的识别和匹配机制")
            
            if performance_metrics.get('evidence_recall', 0) < 0.7:
                improvement_suggestions.append("2. **扩大证据覆盖**: 证据召回率不足，建议扩大搜索范围或降低匹配阈值")
            
            if performance_metrics.get('evidence_mrr', 0) < 0.8:
                avg_rank = 1 / performance_metrics.get('evidence_mrr', 0.001)
                improvement_suggestions.append(f"3. **优化排序算法**: 证据MRR偏低(平均排名{avg_rank:.1f})，需改进统一分数计算和排序机制")
            
            if performance_metrics.get('global_ndcg_at_10', 0) < 0.8:
                improvement_suggestions.append("4. **提升排序质量**: 全局nDCG@10偏低，需优化重要性权重分配")
            
            if performance_metrics.get('blind_spot_recall', 0) < 0.7:
                improvement_suggestions.append("5. **扩大覆盖广度**: 盲区召回率偏低，建议增强假设生成的多样性")
        
        if not improvement_suggestions:
            improvement_suggestions.append("🎉 各项指标表现良好，系统已达到实用水平")
        
        for suggestion in improvement_suggestions:
            report_content.append(suggestion)
        
        report_content.append("")
        
        report_content.append("## 技术细节")
        report_content.append(f"- **评估体系**: 证据为中心 (Evidence-Centric)")
        report_content.append(f"- **核心指标**: 加权F1分数、证据召回率、证据MRR")
        report_content.append(f"- **聚合方法**: {audit_results['config']['aggregation_method']}")
        report_content.append(f"- **蒙特卡洛采样次数**: {audit_results['config']['mc_samples']}")
        report_content.append(f"- **输出目录**: {audit_results['output_dirs']['main']}")
        report_content.append("")
        
        report_content.append("## 输出文件清单")
        report_content.append("### 假设生成")
        report_content.append("- `generated_hypotheses.json` - 详细假设列表")
        report_content.append("- `hypothesis_summary.json` - 假设摘要")
        report_content.append("")
        
        report_content.append("### 条件化不确定性分析")
        report_content.append("- `aggregated_uncertainty_map.json` - 聚合不确定性地图")
        report_content.append("- `final_aggregated_uncertainty_map.json` - 高优先级发现")
        report_content.append("- `training_summary.json` - 训练摘要")
        report_content.append("")
        
        report_content.append("### 证据为中心的定量评价")
        report_content.append("- `quantitative_evaluation_report.json` - 详细定量评价报告")
        report_content.append("- 包含证据MRR、全局nDCG、加权F1等核心指标")
        report_content.append("")
        
        report_content.append("### 定性分析")
        report_content.append("- `qualitative_analysis_{case_id}.md` - 定性分析报告")
        report_content.append("")
        
        report_content.append("---")
        report_content.append(f"*报告由BCSA审计流程编排器自动生成于 {audit_results['timestamp']}*")
        report_content.append(f"*评估体系: 证据为中心 (Evidence-Centric) - 专注于具体证据发现能力*")
        
        report_path = output_dir / f"{audit_results['case_id']}_comprehensive_audit_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_content))
        
        logger.info(f"综合审计报告已生成: {report_path}")
        return report_path

def run_batch_audit_pipeline(case_ids: List[str], 
                             case_scale: str,  
                             case_type: str,   
                             base_data_dir: Path = Path("./seek_data_v3_deep_enhanced/cases"),
                             output_dir: Path = None,
                             method_choice: str = 'PCGATE',
                             aggregation_method: str = 'weighted_max',
                             mc_samples: int = 50,
                             continue_on_error: bool = True) -> Dict[str, Dict[str, Any]]:
    """
    批量运行完整审计流程
    ...
    """
    orchestrator = AuditPipelineOrchestrator(output_dir)
    batch_results = {}
    
    logger.info(f"开始批量审计流程: {len(case_ids)} 个案例 (scale: {case_scale}, type: {case_type}, method: {method_choice})")
    logger.info(f"案例列表: {case_ids}")
    logger.info(f"=" * 80)
    
    successful_cases = 0
    failed_cases = 0
    
    for i, case_id in enumerate(case_ids, 1):
        logger.info(f"\n🎯 处理案例 {i}/{len(case_ids)}: {case_id}")
        logger.info(f"-" * 60)
        
        case_dir = base_data_dir / case_scale / case_type / case_id
        
        if not case_dir.exists():
            logger.error(f"案例目录不存在: {case_dir}")
            batch_results[case_id] = {
                'status': 'failed',
                'error': f'case_directory_not_found: {case_dir}',
                'case_id': case_id
            }
            failed_cases += 1
            if continue_on_error:
                continue
            else:
                break
        
        try:
            case_result = orchestrator.run_complete_audit_pipeline(
                case_dir=case_dir,
                method_choice=method_choice,
                aggregation_method=aggregation_method,
                mc_samples=mc_samples
            )
            
            batch_results[case_id] = case_result
            
            if case_result['pipeline_status'] == 'completed':
                successful_cases += 1
                logger.info(f"✅ 案例 {case_id} 处理完成")
                
                hyp_gen = case_result.get('hypothesis_generation', {})
                cond_analysis = case_result.get('conditioned_uncertainty_analysis', {})
                quant_eval = case_result.get('quantitative_evaluation', {})
                
                logger.info(f"   📊 假设生成: {hyp_gen.get('hypotheses_count', 0)} 个")
                logger.info(f"   📊 条件化分析: {cond_analysis.get('aggregated_edges_count', 0)} 条边")
                logger.info(f"   📊 定量评价: F1分数 {quant_eval.get('f1_score', 0):.1%}")
            else:
                failed_cases += 1
                logger.error(f"❌ 案例 {case_id} 处理失败: {case_result.get('error', '未知错误')}")
        
        except Exception as e:
            failed_cases += 1
            logger.error(f"❌ 案例 {case_id} 处理异常: {e}")
            batch_results[case_id] = {
                'status': 'failed',
                'error': str(e),
                'case_id': case_id
            }
            
            if not continue_on_error:
                break
    
    logger.info(f"\n" + "=" * 80)
    logger.info(f"📊 批量审计流程完成")
    logger.info(f"=" * 80)
    logger.info(f"总案例数: {len(case_ids)}")
    logger.info(f"成功案例: {successful_cases}")
    logger.info(f"失败案例: {failed_cases}")
    logger.info(f"成功率: {successful_cases/len(case_ids):.1%}")
    
    if output_dir:
        batch_summary_file = output_dir / f"batch_audit_summary_{case_scale}.json"
        batch_summary = {
            'batch_timestamp': datetime.now().isoformat(),
            'total_cases': len(case_ids),
            'successful_cases': successful_cases,
            'failed_cases': failed_cases,
            'success_rate': successful_cases / len(case_ids),
            'case_ids': case_ids,
            'method_choice': method_choice,
            'config': {
                'aggregation_method': aggregation_method,
                'mc_samples': mc_samples,
                'continue_on_error': continue_on_error
            },
            'case_results_summary': {
                case_id: {
                    'status': result.get('pipeline_status', result.get('status', 'unknown')),
                    'hypotheses_count': result.get('hypothesis_generation', {}).get('hypotheses_count', 0),
                    'aggregated_edges_count': result.get('conditioned_uncertainty_analysis', {}).get('aggregated_edges_count', 0),
                    'f1_score': result.get('quantitative_evaluation', {}).get('f1_score', 0)
                }
                for case_id, result in batch_results.items()
            }
        }
        
        with open(batch_summary_file, 'w', encoding='utf-8') as f:
            json.dump(convert_numpy_types(batch_summary), f, indent=2, ensure_ascii=False)
        
        logger.info(f"📁 批量摘要已保存: {batch_summary_file}")
    
    return batch_results

def get_available_cases(base_data_dir: Path, case_scale: str, case_type: str) -> List[str]:
    """
    根据指定的scale和type，获取可用的案例列表。
    Args:
        base_data_dir: 基础数据目录 (e.g., ./seek_data_v3_deep_enhanced/cases)
        case_scale: 案例规模 (e.g., "smallcase")
        case_type: 案例类型 (e.g., "Mixed")
    Returns:
        可用的案例ID列表
    """
    cases_dir = base_data_dir / case_scale / case_type
    
    if not cases_dir.is_dir(): # 增加对目录是否存在的检查
        logger.warning(f"案例目录不存在: {cases_dir}")
        return []
    
    available_cases = []
    for case_path in cases_dir.iterdir():
        if case_path.is_dir():
            required_files = [
                "causal_knowledge_graph.json",
                "sensor_data.csv"
            ]
            
            if all((case_path / file).exists() for file in required_files):
                available_cases.append(case_path.name)
    
    available_cases.sort()
    logger.info(f"在 {cases_dir} 发现 {len(available_cases)} 个可用案例")
    return available_cases

def analyze_batch_results(batch_results: Dict[str, Dict[str, Any]], 
                         output_dir: Path = None, case_scale = 'small') -> Dict[str, Any]:
    """
    分析批量评估结果，计算宏平均指标 - 证据为中心的评估体系
    
    Args:
        batch_results: 批量评估结果字典
        output_dir: 输出目录路径
        case_scale: 案例规模标识
        
    Returns:
        宏平均分析结果
    """
    logger.info("开始分析批量评估结果（证据为中心的评估体系）...")
    
    evidence_centric_metrics = []  # 证据为中心的指标
    marginal_benefit_metrics = []  # 边际效益指标
    successful_cases = []
    failed_cases = []
    
    for case_id, result in batch_results.items():
        if result.get('combined_results'):
            pipeline_status = result.get('pipeline_status')
            if pipeline_status in ['completed', 'partially_completed']:
                successful_cases.append(case_id)
                quant_eval = result.get('quantitative_evaluation', {})
            else:
                failed_cases.append(case_id)
                continue
        else:
            if result.get('pipeline_status') == 'completed':
                successful_cases.append(case_id)
                quant_eval = result.get('quantitative_evaluation', {})
            else:
                failed_cases.append(case_id)
                continue
        if quant_eval:
            perf_metrics = quant_eval.get('performance_metrics', {})

            evidence_centric_dict = {
                'case_id': case_id,
                'weighted_f1_score': perf_metrics.get('weighted_f1_score', 0.0),
                'evidence_recall': perf_metrics.get('evidence_recall', 0.0),
                'evidence_precision': perf_metrics.get('evidence_precision', 0.0),
                'f1_score': perf_metrics.get('f1_score', 0.0), # ESF1

                'evidence_mrr': perf_metrics.get('evidence_mrr', 0.0),
                'global_ndcg_at_10': perf_metrics.get('global_ndcg_at_10', 0.0),
                'global_ndcg_at_20': perf_metrics.get('global_ndcg_at_20', 0.0),

                'blind_spot_recall': perf_metrics.get('blind_spot_recall', 0.0),
                'blind_spot_precision': perf_metrics.get('blind_spot_precision', 0.0),
                'blind_spot_f1': perf_metrics.get('blind_spot_f1', 0.0),
                'average_zecr': perf_metrics.get('average_zecr', 0.0),
                
                'weighted_recall': perf_metrics.get('weighted_recall', 0.0),
                'weighted_precision': perf_metrics.get('weighted_precision', 0.0),
                'auc_pr': perf_metrics.get('auc_pr', 0.0),
                'average_true_positive_confidence': perf_metrics.get('average_true_positive_confidence', 0.0)
            }
            evidence_centric_metrics.append(evidence_centric_dict)
            
            marginal_benefit_data = {}
            
            potential_marginal_source = result.get('quantitative_evaluation', {})
            
            marginal_indicators = [
                'weighted_f1_score_marginal', 'evidence_recall_marginal', 
                'evidence_precision_marginal', 'f1_score_marginal',
                
                'evidence_mrr_marginal', 'global_ndcg_at_10_marginal', 'global_ndcg_at_20_marginal',
                'auc_pr_marginal',
                
                'blind_spot_recall_marginal', 'blind_spot_precision_marginal', 
                'blind_spot_f1_marginal', 'average_zecr_marginal',
                
                'core_finding_composite_marginal', 'ranking_performance_composite_marginal',
                'diagnostic_composite_marginal', 'overall_composite_marginal_benefit_score'
            ]
            
            for indicator in marginal_indicators:
                marginal_value = (
                    potential_marginal_source.get(indicator, 0.0) or  # 直接在 quantitative_evaluation 中
                    potential_marginal_source.get('marginal_benefit_metrics', {}).get(indicator, 0.0)  # 在子字典中
                )
                marginal_benefit_data[indicator] = marginal_value
            
            marginal_benefit_data['case_id'] = case_id
            marginal_benefit_metrics.append(marginal_benefit_data)
            
        else:
            failed_cases.append(case_id)
    
    if not evidence_centric_metrics:
        logger.error("没有成功的案例可用于分析")
        return {
            'status': 'failed',
            'error': 'no_successful_cases',
            'total_cases': len(batch_results),
            'successful_cases': 0,
            'failed_cases': len(batch_results)
        }
    
    df_evidence = pd.DataFrame(evidence_centric_metrics)
    df_marginal = pd.DataFrame(marginal_benefit_metrics) if marginal_benefit_metrics else None
    
    logger.info(f"DataFrame列名: {list(df_evidence.columns)}")
    logger.info(f"成功案例数: {len(df_evidence)}")
    if df_marginal is not None:
        logger.info(f"边际效益指标数: {len([col for col in df_marginal.columns if col != 'case_id'])}")
    
    macro_averages = {}
    
    core_finding_metrics = {
        'weighted_f1_score': {
            'target': 0.75,
            'description': '加权F1分数 (核心指标)'
        },
        'evidence_recall': {
            'target': 0.7,
            'description': '证据召回率'
        },
        'evidence_precision': {
            'target': 0.7,
            'description': '证据精确率'
        },
        'f1_score': {
            'target': 0.6,
            'description': '传统F1分数 (对比用)'
        }
    }
    
    for metric, config in core_finding_metrics.items():
        if metric in df_evidence.columns:
            macro_averages[metric] = {
                'mean': float(df_evidence[metric].mean()),
                'std': float(df_evidence[metric].std()),
                'min': float(df_evidence[metric].min()),
                'max': float(df_evidence[metric].max()),
                'target': config['target'],
                'target_met_count': int((df_evidence[metric] >= config['target']).sum()),
                'category': 'core_finding'
            }
            
            marginal_metric = f"{metric}_marginal"
            if df_marginal is not None and marginal_metric in df_marginal.columns:
                macro_averages[metric]['marginal'] = {
                    'mean': float(df_marginal[marginal_metric].mean()),
                    'std': float(df_marginal[marginal_metric].std()),
                    'min': float(df_marginal[marginal_metric].min()),
                    'max': float(df_marginal[marginal_metric].max())
                }
    
    ranking_metrics = {
        'evidence_mrr': {
            'target': 0.8,
            'description': '证据MRR (核心排序指标)'
        },
        'global_ndcg_at_10': {
            'target': 0.8,
            'description': '全局nDCG@10'
        },
        'global_ndcg_at_20': {
            'target': 0.8,
            'description': '全局nDCG@20'
        },
        'auc_pr': {
            'target': 0.7,
            'description': 'AUC-PR (排序综合性能)'
        }
    }
    
    for metric, config in ranking_metrics.items():
        if metric in df_evidence.columns:
            macro_averages[metric] = {
                'mean': float(df_evidence[metric].mean()),
                'std': float(df_evidence[metric].std()),
                'min': float(df_evidence[metric].min()),
                'max': float(df_evidence[metric].max()),
                'target': config['target'],
                'target_met_count': int((df_evidence[metric] >= config['target']).sum()),
                'category': 'ranking_performance'
            }
            
            marginal_metric = f"{metric}_marginal"
            if df_marginal is not None and marginal_metric in df_marginal.columns:
                macro_averages[metric]['marginal'] = {
                    'mean': float(df_marginal[marginal_metric].mean()),
                    'std': float(df_marginal[marginal_metric].std()),
                    'min': float(df_marginal[marginal_metric].min()),
                    'max': float(df_marginal[marginal_metric].max())
                }
    
    diagnostic_metrics = {
        'blind_spot_recall': {
            'target': 0.7,
            'description': '盲区召回率 (诊断指标)'
        },
        'blind_spot_precision': {
            'target': 0.6,
            'description': '盲区精确率 (诊断指标)'
        },
        'blind_spot_f1': {
            'target': 0.6,
            'description': '盲区F1分数 (诊断指标)'
        },
        'average_zecr': {
            'target': 0.6,
            'description': '平均区域证据覆盖率 (诊断指标)'
        }
    }
    
    for metric, config in diagnostic_metrics.items():
        if metric in df_evidence.columns:
            macro_averages[metric] = {
                'mean': float(df_evidence[metric].mean()),
                'std': float(df_evidence[metric].std()),
                'min': float(df_evidence[metric].min()),
                'max': float(df_evidence[metric].max()),
                'target': config['target'],
                'target_met_count': int((df_evidence[metric] >= config['target']).sum()),
                'category': 'diagnostic'
            }
            
            marginal_metric = f"{metric}_marginal"
            if df_marginal is not None and marginal_metric in df_marginal.columns:
                macro_averages[metric]['marginal'] = {
                    'mean': float(df_marginal[marginal_metric].mean()),
                    'std': float(df_marginal[marginal_metric].std()),
                    'min': float(df_marginal[marginal_metric].min()),
                    'max': float(df_marginal[marginal_metric].max())
                }
    
    if df_marginal is not None:
        composite_metrics = [
            'core_finding_composite_marginal',
            'ranking_performance_composite_marginal', 
            'diagnostic_composite_marginal',
            'overall_composite_marginal_benefit_score'
        ]
        
        for metric in composite_metrics:
            if metric in df_marginal.columns:
                macro_averages[metric] = {
                    'mean': float(df_marginal[metric].mean()),
                    'std': float(df_marginal[metric].std()),
                    'min': float(df_marginal[metric].min()),
                    'max': float(df_marginal[metric].max()),
                    'category': 'composite'
                }
    
    quality_gap_mean = float(df_evidence['f1_score'].mean() - df_evidence['weighted_f1_score'].mean())
    quality_gap_std = float((df_evidence['f1_score'] - df_evidence['weighted_f1_score']).std())
    
    total_cases = len(evidence_centric_metrics)
    
    wf1_target_rate = macro_averages.get('weighted_f1_score', {}).get('target_met_count', 0) / total_cases
    esr_target_rate = macro_averages.get('evidence_recall', {}).get('target_met_count', 0) / total_cases
    esp_target_rate = macro_averages.get('evidence_precision', {}).get('target_met_count', 0) / total_cases
    
    mrr_target_rate = macro_averages.get('evidence_mrr', {}).get('target_met_count', 0) / total_cases
    ndcg10_target_rate = macro_averages.get('global_ndcg_at_10', {}).get('target_met_count', 0) / total_cases
    ndcg20_target_rate = macro_averages.get('global_ndcg_at_20', {}).get('target_met_count', 0) / total_cases
    auc_pr_target_rate = macro_averages.get('auc_pr', {}).get('target_met_count', 0) / total_cases
    
    bsr_target_rate = macro_averages.get('blind_spot_recall', {}).get('target_met_count', 0) / total_cases
    zecr_target_rate = macro_averages.get('average_zecr', {}).get('target_met_count', 0) / total_cases
    
    def get_effective_score(metric_name: str, target: float) -> float:
        """获取有效得分，优先使用边际效益版本"""
        metric_data = macro_averages.get(metric_name, {})
        
        if 'marginal' in metric_data:
            return metric_data['marginal']['mean']
        else:
            return metric_data.get('mean', 0.0)
    
    wf1_effective = get_effective_score('weighted_f1_score', 0.75)
    esr_effective = get_effective_score('evidence_recall', 0.7)
    mrr_effective = get_effective_score('evidence_mrr', 0.8)
    ndcg10_effective = get_effective_score('global_ndcg_at_10', 0.8)
    esp_effective = get_effective_score('evidence_precision', 0.7)
    
    wf1_score = min(wf1_effective / 0.75, 1.0) * 30   # 30%权重 - 发现质量核心
    esr_score = min(esr_effective / 0.7, 1.0) * 25    # 25%权重 - 证据覆盖核心
    mrr_score = min(mrr_effective / 0.8, 1.0) * 25    # 25%权重 - 排序效率核心
    ndcg_score = min(ndcg10_effective / 0.8, 1.0) * 10  # 10%权重 - 排序质量
    esp_score = min(esp_effective / 0.7, 1.0) * 10    # 10%权重 - 发现精度
    
    overall_score = wf1_score + esr_score + mrr_score + ndcg_score + esp_score
    
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
    
    has_marginal_data = df_marginal is not None and len(df_marginal.columns) > 1
    evaluation_method = "基于边际效益的证据为中心评估" if has_marginal_data else "传统证据为中心评估"
    
    analysis_result = {
        'status': 'success',
        'analysis_timestamp': datetime.now().isoformat(),
        'evaluation_method': evaluation_method,
        'evaluation_philosophy': 'evidence_centric',
        'summary': {
            'total_cases': len(batch_results),
            'successful_cases': len(successful_cases),
            'failed_cases': len(failed_cases),
            'success_rate': len(successful_cases) / len(batch_results),
            'cases_analyzed': total_cases,
            'has_marginal_benefit_data': has_marginal_data
        },
        'macro_averages': macro_averages,
        'quality_analysis': {
            'quality_gap_mean': quality_gap_mean,
            'quality_gap_std': quality_gap_std,
            'interpretation': 'high_quality_hits' if quality_gap_mean < -0.05 else 
                           'balanced_quality' if abs(quality_gap_mean) <= 0.1 else 
                           'low_quality_hits'
        },
        'target_achievement': {
            'wf1_target_rate': wf1_target_rate,
            'esr_target_rate': esr_target_rate,
            'esp_target_rate': esp_target_rate,
            'mrr_target_rate': mrr_target_rate,
            'ndcg10_target_rate': ndcg10_target_rate,
            'ndcg20_target_rate': ndcg20_target_rate,
            'auc_pr_target_rate': auc_pr_target_rate,
            'bsr_target_rate': bsr_target_rate,
            'zecr_target_rate': zecr_target_rate
        },
        'overall_assessment': {
            'score': overall_score,
            'grade': grade,
            'evaluation_method': evaluation_method,
            'component_scores': {
                'wf1_score': wf1_score,
                'esr_score': esr_score,
                'mrr_score': mrr_score,
                'ndcg_score': ndcg_score,
                'esp_score': esp_score
            },
            'effective_scores_used': {
                'wf1_effective': wf1_effective,
                'esr_effective': esr_effective,
                'esp_effective': esp_effective,
                'esf1_effective': get_effective_score('f1_score', 0.6),
                
                'mrr_effective': mrr_effective,
                'ndcg10_effective': ndcg10_effective,
                'ndcg20_effective': get_effective_score('global_ndcg_at_20', 0.8),
                'auc_pr_effective': get_effective_score('auc_pr', 0.7),
                
                'zecr_effective': get_effective_score('average_zecr', 0.6),
                'bsr_effective': get_effective_score('blind_spot_recall', 0.7)
            }
        },
        'successful_cases': successful_cases,
        'failed_cases': failed_cases,
        'detailed_metrics': evidence_centric_metrics,
        'marginal_benefit_details': marginal_benefit_metrics if has_marginal_data else None
    }
    
    if output_dir:
        analysis_file = output_dir / f"batch_evaluation_analysis_{case_scale}.json"
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(convert_numpy_types(analysis_result), f, indent=2, ensure_ascii=False)
        logger.info(f"批量分析结果已保存: {analysis_file}")
    
    return analysis_result

def print_macro_average_scorecard(analysis_result: Dict[str, Any]):
    """
    打印宏平均记分卡 - 证据为中心的评估体系
    
    Args:
        analysis_result: 批量分析结果
    """
    if analysis_result.get('status') != 'success':
        print(f"❌ 批量分析失败: {analysis_result.get('error', 'Unknown error')}")
        return
    
    summary = analysis_result['summary']
    macro_avg = analysis_result['macro_averages']
    target_achieve = analysis_result['target_achievement']
    overall = analysis_result['overall_assessment']
    quality = analysis_result['quality_analysis']
    
    print("\n" + "="*80)
    print("🏆 BCSA宏平均性能记分卡 - 证据为中心的评估体系")
    print("="*80)
    print(f"分析时间: {analysis_result['analysis_timestamp']}")
    print(f"评估哲学: 证据为中心 (Evidence-Centric)")
    print(f"案例统计: {summary['successful_cases']}/{summary['total_cases']} 成功 "
          f"(成功率: {summary['success_rate']:.1%})")
    print()
    
    print("🎯 核心发现能力 (Core Finding Metrics)")
    print("="*60)
    print("    关注系统发现具体风险证据的质量和覆盖能力")
    print("-" * 40)
    
    wf1 = macro_avg.get('weighted_f1_score', {})
    if wf1:
        wf1_status = "✅ 达标" if wf1['mean'] >= 0.75 else "❌ 未达标"
        print(f"加权F1分数 (WF1) - 核心指标:")
        print(f"   平均值: {wf1['mean']:.1%} (±{wf1['std']:.1%}) {wf1_status}")
        print(f"   目标: ≥75% | 达标案例: {target_achieve.get('wf1_target_rate', 0):.1%}")
        print(f"   范围: {wf1['min']:.1%} ~ {wf1['max']:.1%}")
        print(f"   说明: 重要性加权的综合发现质量，是最重要的评估指标")
        print()
    
    esr = macro_avg.get('evidence_recall', {})
    if esr:
        esr_status = "✅ 达标" if esr['mean'] >= 0.7 else "❌ 未达标"
        print(f"证据召回率 (ESR):")
        print(f"   平均值: {esr['mean']:.1%} (±{esr['std']:.1%}) {esr_status}")
        print(f"   目标: ≥70% | 达标案例: {target_achieve.get('esr_target_rate', 0):.1%}")
        print(f"   范围: {esr['min']:.1%} ~ {esr['max']:.1%}")
        print(f"   说明: 系统找到所有GT证据的比例")
        print()
    
    esp = macro_avg.get('evidence_precision', {})
    if esp:
        esp_status = "✅ 达标" if esp['mean'] >= 0.7 else "❌ 未达标"
        print(f"证据精确率 (ESP):")
        print(f"   平均值: {esp['mean']:.1%} (±{esp['std']:.1%}) {esp_status}")
        print(f"   目标: ≥70% | 达标案例: {target_achieve.get('esp_target_rate', 0):.1%}")
        print(f"   范围: {esp['min']:.1%} ~ {esp['max']:.1%}")
        print(f"   说明: 系统发现中真正有效证据的比例")
        print()
    
    esf1 = macro_avg.get('f1_score', {})
    if esf1:
        print(f"传统F1分数 (ESF1) - 对比用:")
        print(f"   平均值: {esf1['mean']:.1%} (±{esf1['std']:.1%})")
        print(f"   范围: {esf1['min']:.1%} ~ {esf1['max']:.1%}")
        print(f"   说明: 不考虑重要性权重的传统F1分数")
        print()
    
    print("🏆 排序性能 (Ranking Performance)")
    print("="*60)
    print("    关注系统将正确证据排在前面的能力")
    print("-" * 40)
    
    mrr = macro_avg.get('evidence_mrr', {})
    if mrr:
        mrr_status = "✅ 达标" if mrr['mean'] >= 0.8 else "❌ 未达标"
        avg_rank = 1 / mrr['mean'] if mrr['mean'] > 0 else float('inf')
        print(f"证据MRR (Evidence MRR) - 核心排序指标:")
        print(f"   平均值: {mrr['mean']:.3f} (±{mrr['std']:.3f}) {mrr_status}")
        print(f"   目标: ≥0.800 | 达标案例: {target_achieve.get('mrr_target_rate', 0):.1%}")
        print(f"   范围: {mrr['min']:.3f} ~ {mrr['max']:.3f}")
        print(f"   平均证据排名: {avg_rank:.1f}")
        print(f"   说明: 每个GT证据在排序列表中位置的平均倒数")
        print()
    
    ndcg10 = macro_avg.get('global_ndcg_at_10', {})
    if ndcg10:
        ndcg10_status = "✅ 达标" if ndcg10['mean'] >= 0.8 else "❌ 未达标"
        print(f"全局nDCG@10 (Global nDCG@10):")
        print(f"   平均值: {ndcg10['mean']:.3f} (±{ndcg10['std']:.3f}) {ndcg10_status}")
        print(f"   目标: ≥0.800 | 达标案例: {target_achieve.get('ndcg10_target_rate', 0):.1%}")
        print(f"   范围: {ndcg10['min']:.3f} ~ {ndcg10['max']:.3f}")
        print(f"   说明: 前10个发现的整体排序质量，考虑重要性权重")
        print()
    
    ndcg20 = macro_avg.get('global_ndcg_at_20', {})
    if ndcg20:
        ndcg20_status = "✅ 达标" if ndcg20['mean'] >= 0.8 else "❌ 未达标"
        print(f"全局nDCG@20 (Global nDCG@20):")
        print(f"   平均值: {ndcg20['mean']:.3f} (±{ndcg20['std']:.3f}) {ndcg20_status}")
        print(f"   目标: ≥0.800 | 达标案例: {target_achieve.get('ndcg20_target_rate', 0):.1%}")
        print(f"   范围: {ndcg20['min']:.3f} ~ {ndcg20['max']:.3f}")
        print(f"   说明: 前20个发现的整体排序质量，考虑重要性权重")
        print()
    
    auc_pr = macro_avg.get('auc_pr', {})
    if auc_pr:
        auc_pr_status = "✅ 达标" if auc_pr['mean'] >= 0.7 else "❌ 未达标"
        print(f"AUC-PR (排序综合性能):")
        print(f"   平均值: {auc_pr['mean']:.3f} (±{auc_pr['std']:.3f}) {auc_pr_status}")
        print(f"   目标: ≥0.700 | 达标案例: {target_achieve.get('auc_pr_target_rate', 0):.1%}")
        print(f"   范围: {auc_pr['min']:.3f} ~ {auc_pr['max']:.3f}")
        print(f"   说明: 精确率-召回率曲线下的面积，综合反映排序性能")
        print()
    
    print("📊 发现质量对比分析:")
    print("-" * 40)
    
    if wf1 and esf1:
        quality_gap = quality['quality_gap_mean']
        gap_interpretation = quality['interpretation']
        
        print(f"质量差距: ESF1({esf1['mean']:.1%}) - WF1({wf1['mean']:.1%}) = {quality_gap:.1%}")
        
        if gap_interpretation == 'high_quality_hits':
            quality_msg = "✅ 高质量命中占优，系统倾向于发现重要证据"
        elif gap_interpretation == 'balanced_quality':
            quality_msg = "⚖️ 质量分布均衡，各重要性证据发现均衡"
        else:
            quality_msg = "⚠️ 偏向低重要性命中，需优化高重要性证据识别"
        
        print(f"解读: {quality_msg}")
        
        if mrr and ndcg10:
            if mrr['mean'] >= 0.8 and ndcg10['mean'] >= 0.8:
                ranking_msg = "🏆 排序效率优秀，证据能被快速准确发现"
            elif mrr['mean'] >= 0.6 or ndcg10['mean'] >= 0.6:
                ranking_msg = "🔄 排序效率中等，仍有优化空间"
            else:
                ranking_msg = "⚠️ 排序效率较差，需重点改进排序算法"
            print(f"排序效率: {ranking_msg}")
        print()
    
    print("🔍 诊断性指标 - 覆盖广度分析")
    print("="*60)
    print("    这些指标用于诊断是否遗漏了整个问题区域")
    print("-" * 40)
    
    bsr = macro_avg.get('blind_spot_recall', {})
    zecr = macro_avg.get('average_zecr', {})
    
    if bsr:
        bsr_status = "✅ 达标" if bsr['mean'] >= 0.7 else "❌ 未达标"
        print(f"盲区召回率 (BSR): {bsr['mean']:.1%} (±{bsr['std']:.1%}) {bsr_status}")
        print(f"   目标: ≥70% | 达标案例: {target_achieve.get('bsr_target_rate', 0):.1%}")
    
    if zecr:
        zecr_status = "✅ 达标" if zecr['mean'] >= 0.6 else "❌ 未达标"
        print(f"平均区域证据覆盖率 (ZECR): {zecr['mean']:.1%} (±{zecr['std']:.1%}) {zecr_status}")
        print(f"   目标: ≥60% | 达标案例: {target_achieve.get('zecr_target_rate', 0):.1%}")
    
    if bsr and zecr:
        if bsr['mean'] >= 0.7 and zecr['mean'] >= 0.6:
            coverage_msg = "✅ 覆盖广度优秀，问题区域识别全面"
        elif bsr['mean'] >= 0.5 or zecr['mean'] >= 0.4:
            coverage_msg = "⚠️ 覆盖广度中等，可能存在部分遗漏"
        else:
            coverage_msg = "❌ 覆盖广度不足，需扩大假设生成范围"
        print(f"覆盖广度评估: {coverage_msg}")
    print()
    
    print("📈 综合评级:")
    print("-" * 40)
    
    evaluation_method = analysis_result.get('evaluation_method', '传统证据为中心评估')
    has_marginal_data = analysis_result.get('summary', {}).get('has_marginal_benefit_data', False)
    
    print(f"评估方法: {evaluation_method}")
    print(f"评估哲学: 证据为中心 (Evidence-Centric)")
    if has_marginal_data:
        print("💡 边际效益机制: 缓解了小样本评估的脆弱性问题")
    
    print(f"综合得分: {overall['score']:.1f}/100")
    print(f"评级: {overall['grade']}")
    
    component_scores = overall['component_scores']
    print(f"组成部分: WF1({component_scores['wf1_score']:.1f}) + "
          f"ESR({component_scores['esr_score']:.1f}) + "
          f"MRR({component_scores['mrr_score']:.1f}) + "
          f"nDCG({component_scores['ndcg_score']:.1f}) + "
          f"ESP({component_scores['esp_score']:.1f})")
    
    if has_marginal_data and 'effective_scores_used' in overall:
        effective_scores = overall['effective_scores_used']
        print(f"实际有效得分 (边际效益转化后):")
        
        wf1_val = effective_scores.get('wf1_effective', 0)
        esr_val = effective_scores.get('esr_effective', 0)
        esp_val = effective_scores.get('esp_effective', 0)
        esf1_val = effective_scores.get('esf1_effective', 0)
        print(f"  - 核心发现: WF1={wf1_val:.1%}, ESR={esr_val:.1%}, ESP={esp_val:.1%}, ESF1={esf1_val:.1%}")
        
        mrr_val = effective_scores.get('mrr_effective', 0)
        ndcg10_val = effective_scores.get('ndcg10_effective', 0)
        ndcg20_val = effective_scores.get('ndcg20_effective', 0)
        auc_pr_val = effective_scores.get('auc_pr_effective', 0)
        print(f"  - 排序性能: MRR={mrr_val:.3f}, nDCG@10={ndcg10_val:.3f}, nDCG@20={ndcg20_val:.3f}, AUC-PR={auc_pr_val:.3f}")
        
        zecr_val = effective_scores.get('zecr_effective', 0)
        bsr_val = effective_scores.get('bsr_effective', 0)
        print(f"  - 诊断指标: ZECR={zecr_val:.1%}, BSR={bsr_val:.1%}")
    print()
    
    print("📊 稳定性分析:")
    print("-" * 40)
    
    stability_status = []
    
    if wf1 and 'mean' in wf1 and 'std' in wf1 and wf1['mean'] > 0:
        wf1_cv = wf1['std'] / wf1['mean']
        esr_cv = esr['std'] / esr['mean'] if esr and esr['mean'] > 0 else float('inf')
        
        if wf1_cv < 0.3 and esr_cv < 0.3:
            stability_status.append("核心发现能力稳定")
        else:
            stability_status.append("核心发现能力波动较大")
        print(f"核心发现能力: WF1变异系数={wf1_cv:.2f}, ESR变异系数={esr_cv:.2f}")
    
    if mrr and ndcg10:
        mrr_cv = mrr['std'] / mrr['mean'] if mrr['mean'] > 0 else float('inf')
        ndcg_cv = ndcg10['std'] / ndcg10['mean'] if ndcg10['mean'] > 0 else float('inf')
        
        if mrr_cv < 0.3 and ndcg_cv < 0.3:
            stability_status.append("排序性能稳定")
        else:
            stability_status.append("排序性能波动较大")
        print(f"排序性能: MRR变异系数={mrr_cv:.2f}, nDCG@10变异系数={ndcg_cv:.2f}")
    
    if bsr and zecr:
        bsr_cv = bsr['std'] / bsr['mean'] if bsr['mean'] > 0 else float('inf')
        zecr_cv = zecr['std'] / zecr['mean'] if zecr['mean'] > 0 else float('inf')
        
        if bsr_cv < 0.3 and zecr_cv < 0.3:
            stability_status.append("诊断指标稳定")
        else:
            stability_status.append("诊断指标波动较大")
        print(f"诊断指标: BSR变异系数={bsr_cv:.2f}, ZECR变异系数={zecr_cv:.2f}")
    
    print(f"稳定性评估: {', '.join(stability_status) if stability_status else '指标不足'}")
    print()
    
    print("💡 改进建议:")
    print("-" * 40)
    
    suggestions = []
    
    if wf1 and wf1['mean'] < 0.75:
        gap_mean = quality['quality_gap_mean']
        if gap_mean > 0.1:
            suggestions.append(f"🎯 核心问题: WF1为{wf1['mean']:.1%}且质量差距{gap_mean:.1%}，需重点优化高重要性证据识别")
        else:
            suggestions.append(f"🎯 核心问题: WF1为{wf1['mean']:.1%}，需整体提升证据发现质量")
    
    if esr and esr['mean'] < 0.7:
        suggestions.append(f"⭐ 证据覆盖: ESR为{esr['mean']:.1%}，需扩大搜索范围或降低匹配阈值")
    
    if esp and esp['mean'] < 0.7:
        suggestions.append(f"⭐ 发现精度: ESP为{esp['mean']:.1%}，需减少误报或提升匹配精度")
    
    if mrr and mrr['mean'] < 0.8:
        avg_rank = 1 / mrr['mean'] if mrr['mean'] > 0 else float('inf')
        suggestions.append(f"🏆 排序优化: Evidence MRR为{mrr['mean']:.3f}(平均排名{avg_rank:.1f})，需优化排序算法")
    
    if ndcg10 and ndcg10['mean'] < 0.8:
        suggestions.append(f"🏆 排序质量: Global nDCG@10为{ndcg10['mean']:.3f}，需改进重要性权重或排序算法")
    
    if auc_pr and auc_pr['mean'] < 0.7:
        suggestions.append(f"🏆 综合性能: AUC-PR为{auc_pr['mean']:.3f}，需改进精确率-召回率平衡")
    
    component_scores = overall.get('component_scores', {})
    if overall['score'] < 70:
        if component_scores.get('wf1_score', 0) < 20:  # WF1得分低于目标的2/3
            suggestions.append("🚨 优先级1: 发现质量是核心问题，建议重点改进证据匹配和重要性识别机制")
        if component_scores.get('esr_score', 0) + component_scores.get('mrr_score', 0) < 35:  # ESR+MRR得分低于目标的70%
            suggestions.append("🚨 优先级2: 证据覆盖和排序效率需要协同改进")
    
    if bsr and bsr['mean'] < 0.7:
        suggestions.append("🔍 覆盖诊断: BSR偏低，建议扩大假设生成覆盖面，避免遗漏整个问题区域")
    
    high_cv_indicators = []
    if wf1 and (wf1['std'] / wf1['mean']) > 0.3:
        high_cv_indicators.append("WF1")
    if esr and (esr['std'] / esr['mean']) > 0.3:
        high_cv_indicators.append("ESR")
    if mrr and (mrr['std'] / mrr['mean']) > 0.3:
        high_cv_indicators.append("MRR")
    
    if high_cv_indicators:
        suggestions.append(f"📊 稳定性: {', '.join(high_cv_indicators)}波动较大，需增强方法的跨案例稳定性")
    
    if not suggestions:
        suggestions.append("🎉 各项指标表现良好且稳定，证据发现系统已达到实用水平")
    
    for i, suggestion in enumerate(suggestions, 1):
        print(f"{i}. {suggestion}")
    
    print("\n" + "="*80)
    print("💡 提示: 此评估采用证据为中心的体系，专注于系统发现具体风险项的能力")
    print("🎯 核心关注: 加权F1分数、证据召回率和证据MRR是判断系统实用性的关键指标")
    print("🏆 排序重要性: 证据MRR和全局nDCG反映系统将正确答案排在前面的能力")
    print("🔍 诊断参考: 盲区指标作为覆盖广度的诊断工具，确保未遗漏整个问题区域")
    print("="*80)


def run_complete_batch_evaluation(case_ids: List[str] = None,
                                  case_scale: str = "smallcase",
                                  case_type: str = "Mixed",
                                 base_data_dir: Path = Path("./seek_data_v3_deep_enhanced/cases"),
                                 output_dir: Path = None,
                                 method_choice: str = 'PCGATE',
                                 aggregation_method: str = 'weighted_max',
                                 mc_samples: int = 50) -> Dict[str, Any]:
    """
    运行完整的批量评估流程 (流程 + 分析)
    
    Args:
        case_ids: 案例ID列表，如果为None则自动获取所有可用案例
        case_scale: 案例规模
        case_type: 案例类型
        output_dir: 输出目录路径
        method_choice: 方法选择 ('PCVGAE', 'PCGATE', 'BOTH')
        aggregation_method: 聚合方法
        mc_samples: 蒙特卡洛采样次数
        
    Returns:
        包含批量结果和宏平均分析的完整结果
    """
    logger.info("🚀 启动完整批量评估流程")
    
    if case_ids is None:
        case_ids = get_available_cases(base_data_dir, case_scale, case_type)
        logger.info(f"自动发现 {len(case_ids)} 个可用案例")
    
    if not case_ids:
        logger.error("没有可用的案例进行批量评估")
        return {'status': 'failed', 'error': 'no_available_cases'}
    
    logger.info(f"步骤1: 运行批量审计流程 ({len(case_ids)} 个案例, 方法: {method_choice})")
    batch_results = run_batch_audit_pipeline(
        case_ids=case_ids,
        case_scale=case_scale,
        case_type=case_type,
        base_data_dir=base_data_dir,
        output_dir=output_dir,
        method_choice=method_choice,  # 传递方法选择参数
        aggregation_method=aggregation_method,
        mc_samples=mc_samples,
        continue_on_error=True
    )
    
    logger.info("步骤2: 分析批量结果，计算宏平均指标")
    analysis_result = analyze_batch_results(batch_results, output_dir, case_scale)
    
    print_macro_average_scorecard(analysis_result)
    
    complete_result = {
        'batch_results': batch_results,
        'macro_analysis': analysis_result,
        'timestamp': datetime.now().isoformat(),
        'config': {
            'case_ids': case_ids,
            'method_choice': method_choice,
            'aggregation_method': aggregation_method,
            'mc_samples': mc_samples
        }
    }
    
    if output_dir:
        complete_results_file = output_dir / f"complete_batch_evaluation_{case_scale}.json"
        with open(complete_results_file, 'w', encoding='utf-8') as f:
            json.dump(convert_numpy_types(complete_result), f, indent=2, ensure_ascii=False)
        logger.info(f"完整批量评估结果已保存: {complete_results_file}")
    
    logger.info("✅ 完整批量评估流程完成")
    return complete_result




def discover_case_structure(base_data_dir: Path) -> Dict[str, List[str]]:
    """
    扫描基础数据目录，自动发现所有可用的 case_scale 和 case_type。
    Returns:
        一个字典，键是 scale，值是该 scale 下的所有 type 列表。
        e.g., {'smallcase': ['Mixed', 'BCSA'], 'mediumcase': ['CEDA']}
    """
    structure = defaultdict(list)
    if not base_data_dir.is_dir():
        return {}
    
    for scale_path in base_data_dir.iterdir():
        if scale_path.is_dir():
            for type_path in scale_path.iterdir():
                if type_path.is_dir():
                    structure[scale_path.name].append(type_path.name)
    return dict(structure)


def run_analysis_only_evaluation(case_ids: List[str],
                                 case_scale: str,
                                 case_type: str,
                                 output_dir: Path) -> Dict[str, Any]:
    """
    仅分析模式：基于已有结果直接进行宏平均分析，不执行审计流程
    
    Args:
        case_ids: 案例ID列表
        case_scale: 案例规模
        case_type: 案例类型
        output_dir: 输出目录路径
        
    Returns:
        包含宏平均分析的结果
    """
    logger.info("📊 启动仅分析评估流程")
    
    batch_results = {}
    
    for case_id in case_ids:
        case_result_dir = output_dir / case_scale / case_type / case_id / "BCSA_Analysis"
        result_file = case_result_dir / f"{case_id}_complete_audit_results.json"
        
        if result_file.exists():
            try:
                with open(result_file, 'r', encoding='utf-8') as f:
                    case_result = json.load(f)
                batch_results[case_id] = case_result
                logger.info(f"✅ 加载案例结果: {case_id}")
            except Exception as e:
                logger.warning(f"⚠️ 无法加载案例 {case_id} 的结果: {e}")
                batch_results[case_id] = {
                    'status': 'failed',
                    'error': f'result_file_load_failed: {e}',
                    'case_id': case_id
                }
        else:
            logger.warning(f"⚠️ 未找到案例 {case_id} 的结果文件: {result_file}")
            batch_results[case_id] = {
                'status': 'failed',
                'error': f'result_file_not_found: {result_file}',
                'case_id': case_id
            }
    
    if not batch_results:
        logger.error("没有找到任何有效的案例结果")
        return {'status': 'failed', 'error': 'no_valid_case_results'}
    
    valid_results = [r for r in batch_results.values() if r.get('pipeline_status') == 'completed' or r.get('status') != 'failed']
    logger.info(f"📊 找到 {len(valid_results)} 个有效的案例结果，共 {len(batch_results)} 个案例")
    
    if not valid_results:
        logger.error("没有找到任何有效的完成状态案例结果")
        return {'status': 'failed', 'error': 'no_completed_case_results'}
    
    logger.info("步骤1: 分析批量结果，计算宏平均指标")
    analysis_result = analyze_batch_results(batch_results, output_dir, case_scale)
    
    print_macro_average_scorecard(analysis_result)
    
    complete_result = {
        'batch_results': batch_results,
        'macro_analysis': analysis_result,
        'timestamp': datetime.now().isoformat(),
        'mode': 'analysis_only',
        'config': {
            'case_ids': case_ids,
            'case_scale': case_scale,
            'case_type': case_type
        }
    }
    
    if output_dir:
        complete_results_file = output_dir / f"analysis_only_evaluation_{case_scale}.json"
        with open(complete_results_file, 'w', encoding='utf-8') as f:
            json.dump(convert_numpy_types(complete_result), f, indent=2, ensure_ascii=False)
        logger.info(f"仅分析评估结果已保存: {complete_results_file}")
    
    logger.info("✅ 仅分析评估流程完成")
    return complete_result


def run_quantitative_qualitative_batch_evaluation(case_ids: List[str],
                                                  case_scale: str,
                                                  case_type: str,
                                                  output_dir: Path,
                                                  base_data_dir: Path = Path("./seek_data_v3_deep_enhanced/cases")) -> Dict[str, Any]:
    """
    批量定量定性分析模式：假定已经完成了假设生成和条件化不确定性分析，
    批量执行定量评价和定性分析
    
    Args:
        case_ids: 案例ID列表
        case_scale: 案例规模
        case_type: 案例类型
        output_dir: 输出目录路径
        
    Returns:
        包含批量定量定性分析的结果
    """
    logger.info("🔬 启动批量定量定性分析流程")
    
    batch_results = {}
    successful_cases = 0
    failed_cases = 0
    
    for i, case_id in enumerate(case_ids, 1):
        logger.info(f"\n🎯 处理案例 {i}/{len(case_ids)}: {case_id}")
        logger.info(f"-" * 60)
        
        case_output_dir = output_dir / case_scale / case_type / case_id / "BCSA_Analysis"
        conditioned_uncertainty_dir = case_output_dir / "1_Final_Results" / "pc_vgae_audit"
        qualitative_evaluation_dir = case_output_dir / "1_Final_Results" / "evaluation"
        
        required_files = [
            conditioned_uncertainty_dir / "aggregated_uncertainty_map.json",
            conditioned_uncertainty_dir / "final_aggregated_uncertainty_map.json"
        ]
        
        missing_files = [f for f in required_files if not f.exists()]
        if missing_files:
            logger.error(f"案例 {case_id} 缺少必要文件: {[str(f) for f in missing_files]}")
            batch_results[case_id] = {
                'status': 'failed',
                'error': f'missing_required_files: {[str(f) for f in missing_files]}',
                'case_id': case_id
            }
            failed_cases += 1
            continue
        
        try:
            qualitative_evaluation_dir.mkdir(parents=True, exist_ok=True)
            
            case_result = {
                'case_id': case_id,
                'timestamp': datetime.now().isoformat(),
                'quantitative_qualitative_status': 'running'
            }
            
            logger.info("📊 步骤1: 运行定量评价模块")
            
            quantitative_result = evaluate_case(
                case_id=case_id,
                case_directory=str(conditioned_uncertainty_dir),
                output_file=str(qualitative_evaluation_dir / "quantitative_evaluation_report.json"),
                use_legacy_matching=False,
                base_data_dir=str(base_data_dir)
            )
            
            print_evaluation_summary(quantitative_result)
            
            case_result['quantitative_evaluation'] = quantitative_result.get('quantitative_metrics', {})
            case_result['quantitative_evaluation']['status'] = 'success'
            case_result['quantitative_evaluation']['output_dir'] = str(conditioned_uncertainty_dir)
            
            logger.info(f"✅ 定量评价完成: F1分数 {quantitative_result['quantitative_metrics']['performance_metrics']['f1_score']:.1%}")
            
            if QUALITATIVE_AVAILABLE:
                logger.info("📝 步骤2: 生成定性分析报告")
                
                try:
                    qualitative_report_content = generate_qualitative_report(
                        case_id=case_id,
                        case_directory=str(conditioned_uncertainty_dir),
                        output_file=str(qualitative_evaluation_dir)
                    )
                    
                    case_result['qualitative_evaluation'] = {
                        'status': 'success',
                        'report_generated': True,
                        'report_length': len(qualitative_report_content.split('\n')) if qualitative_report_content else 0,
                        'output_dir': str(qualitative_evaluation_dir)
                    }
                    
                    logger.info(f"✅ 定性分析报告生成完成")
                except Exception as e:
                    logger.warning(f"定性分析报告生成失败: {e}")
                    case_result['qualitative_evaluation'] = {
                        'status': 'failed',
                        'error': str(e),
                        'output_dir': str(qualitative_evaluation_dir)
                    }
            else:
                logger.info("⚠️ 跳过定性分析报告生成: 模块不可用")
                case_result['qualitative_evaluation'] = {
                    'status': 'skipped',
                    'reason': 'module_not_available'
                }
            
            case_result['quantitative_qualitative_status'] = 'completed'
            batch_results[case_id] = case_result
            successful_cases += 1
            
            logger.info(f"✅ 案例 {case_id} 定量定性分析完成")
            
        except Exception as e:
            failed_cases += 1
            logger.error(f"❌ 案例 {case_id} 处理异常: {e}")
            batch_results[case_id] = {
                'status': 'failed',
                'error': str(e),
                'case_id': case_id
            }
    
    logger.info(f"\n" + "=" * 80)
    logger.info(f"📊 批量定量定性分析完成")
    logger.info(f"=" * 80)
    logger.info(f"总案例数: {len(case_ids)}")
    logger.info(f"成功案例: {successful_cases}")
    logger.info(f"失败案例: {failed_cases}")
    logger.info(f"成功率: {successful_cases/len(case_ids):.1%}")
    
    batch_summary_file = output_dir / f"batch_quantitative_qualitative_summary_{case_scale}.json"
    batch_summary = {
        'batch_timestamp': datetime.now().isoformat(),
        'mode': 'quantitative_qualitative_batch',
        'total_cases': len(case_ids),
        'successful_cases': successful_cases,
        'failed_cases': failed_cases,
        'success_rate': successful_cases / len(case_ids),
        'case_ids': case_ids,
        'case_results_summary': {
            case_id: {
                'status': result.get('quantitative_qualitative_status', result.get('status', 'unknown')),
                'f1_score': result.get('quantitative_evaluation', {}).get('performance_metrics', {}).get('f1_score', 0),
                'weighted_f1_score': result.get('quantitative_evaluation', {}).get('performance_metrics', {}).get('weighted_f1_score', 0),
                'qualitative_status': result.get('qualitative_evaluation', {}).get('status', 'unknown')
            }
            for case_id, result in batch_results.items()
        }
    }
    
    with open(batch_summary_file, 'w', encoding='utf-8') as f:
        json.dump(convert_numpy_types(batch_summary), f, indent=2, ensure_ascii=False)
    
    logger.info(f"📁 批量摘要已保存: {batch_summary_file}")
    
    return {
        'status': 'success',
        'batch_results': batch_results,
        'summary': {
            'total_cases': len(case_ids),
            'successful_cases': successful_cases,
            'failed_cases': failed_cases,
            'success_rate': successful_cases / len(case_ids)
        },
        'timestamp': datetime.now().isoformat(),
        'mode': 'quantitative_qualitative_batch'
    }

def main():
    """主函数 - 支持按 scale/type 选择，并保留所有原有运行模式"""
    
    print("=" * 80)
    print("🎯 BCSA完整审计流程编排器 (V2.0 灵活选择版)")
    print("=" * 80)
    
    print("\n🔬 请选择要运行的方法:")
    print("   1. PCVGAE (Point-wise Conditional Variational Graph AutoEncoder)")
    print("   2. PCGATE (Point-wise Conditional Graph Attention Networks)")  
    print("   3. 同时运行两种方法")
    
    method_choice_str = input("请输入选择 (1/2/3, 默认为 1): ").strip() or "1"
    method_choices = {
        '1': 'PCVGAE',
        '2': 'PCGATE', 
        '3': 'BOTH'
    }
    selected_method = method_choices.get(method_choice_str, 'PCVGAE')
    
    print(f"\n✅ 已选择方法: {selected_method}")
    
    base_data_dir = Path("./seek_data_v3_deep_enhanced/cases")
    output_dir = Path("./seek_data_v3_deep_enhanced/results")

    try:
        case_structure = discover_case_structure(base_data_dir)
        if not case_structure:
            print(f"❌ 错误: 在目录 '{base_data_dir}' 下未发现任何有效的案例结构。")
            print("   请确认数据目录结构为: .../cases/<scale>/<type>/<case_id>")
            return

        available_scales = list(case_structure.keys())
        print("\n🔍 发现以下案例规模 (Scale):")
        for i, scale in enumerate(available_scales, 1):
            print(f"   {i}. {scale}")
        
        scale_choice_str = input(f"请选择一个 Scale (输入序号, 默认为 1): ").strip() or "1"
        selected_scale = available_scales[int(scale_choice_str) - 1]

        available_types = case_structure[selected_scale]
        print(f"\n🔬 在 '{selected_scale}' 下发现以下案例类型 (Type):")
        for i, type_name in enumerate(available_types, 1):
            print(f"   {i}. {type_name}")
        
        type_choice_str = input(f"请选择一个 Type (输入序号, 默认为 1): ").strip() or "1"
        selected_type = available_types[int(type_choice_str) - 1]

        print(f"\n✅ 已选定范围 -> Scale: '{selected_scale}', Type: '{selected_type}', Method: '{selected_method}'")

        available_cases = get_available_cases(base_data_dir, selected_scale, selected_type)
        if not available_cases:
            print(f"\n❌ 在 '{base_data_dir / selected_scale / selected_type}' 路径下未找到可运行的案例。")
            return
            
        print(f"\n📋 在选定范围内发现 {len(available_cases)} 个可用案例:")
        for i, case_id in enumerate(available_cases, 1):
            if i <= 10 or len(available_cases) <= 15:
                print(f"   {i:2d}. {case_id}")
            elif i == 11:
                print(f"   ... (及其他 {len(available_cases) - 10} 个)")
        
        print("\n🚀 请选择运行模式:")
        print("   1. 单个案例模式 (运行一个指定的案例)")
        print("   2. 批量案例模式 (对选定范围内的案例进行批量处理)")
        print("   3. 🏆 宏平均评估模式 (对选定范围内的案例进行批量处理并生成稳定评估报告)")
        print("   4. 🔬 批量定量定性分析模式 (假定已完成前期处理，批量执行定量评价和定性分析)")
        print("   5. 📊 直接评估模式 (基于已有的处理结果直接进行评估分析)")

        mode = input("\n请输入选择 (1/2/3/4/5, 默认为 3): ").strip() or "3"
        
        if mode == '1':
            print("\n🎯 单个案例模式")
            case_idx_str = input(f"请输入案例序号 (1-{len(available_cases)}, 默认为 1): ").strip() or "1"
            case_idx = int(case_idx_str) - 1
            case_id = available_cases[case_idx]
            
            case_dir = base_data_dir / selected_scale / selected_type / case_id
            
            print(f"\n🚀 开始处理单个案例: {case_id}")
            print(f"📁 案例目录: {case_dir}")
            print(f"🔬 使用方法: {selected_method}")
            
            orchestrator = AuditPipelineOrchestrator(output_dir)
            
            orchestrator.run_complete_audit_pipeline(
                case_dir=case_dir,
                method_choice=selected_method,
                aggregation_method='weighted_max',
                mc_samples=50
            )

        elif mode == '2':
            print("\n🎯 批量案例模式")
            print("   请选择批量处理方式:")
            print(f"   1. 处理当前选定的所有案例 ({len(available_cases)} 个)")
            print("   2. 处理前 N 个案例")
            print("   3. 处理指定范围的案例 (按序号)")
            print("   4. 手动输入案例 ID 列表 (逗号分隔)")
            
            batch_mode = input("   请输入选择 (1/2/3/4, 默认为 1): ").strip() or "1"
            
            case_ids_to_run = []
            if batch_mode == '1':
                case_ids_to_run = available_cases
            elif batch_mode == '2':
                n = int(input(f"   请输入要处理的案例数量 N (1-{len(available_cases)}): "))
                case_ids_to_run = available_cases[:n]
            elif batch_mode == '3':
                start = int(input(f"   请输入起始序号 (1-{len(available_cases)}): ")) - 1
                end = int(input(f"   请输入结束序号 (1-{len(available_cases)}): "))
                case_ids_to_run = available_cases[start:end]
            elif batch_mode == '4':
                ids_input = input("   请输入案例ID (用逗号分隔):\n   ").strip()
                input_ids = [cid.strip() for cid in ids_input.split(',') if cid.strip()]
                case_ids_to_run = [cid for cid in input_ids if cid in available_cases]
                invalid_ids = [cid for cid in input_ids if cid not in case_ids_to_run]
                if invalid_ids:
                    print(f"   ⚠️ 警告: 以下ID无效或不在当前选择范围内，将被忽略: {', '.join(invalid_ids)}")
            
            if not case_ids_to_run:
                print("❌ 未选择任何有效案例进行批量处理。")
                return
            
            print(f"\n将对以下 {len(case_ids_to_run)} 个案例进行批量处理:")
            print("   " + ", ".join(case_ids_to_run[:5]) + ("..." if len(case_ids_to_run) > 5 else ""))
            
            confirm = input("确认开始? (y/N): ").strip().lower()
            if confirm in ['y', 'yes']:
                run_batch_audit_pipeline(
                    case_ids=case_ids_to_run,
                    case_scale=selected_scale,
                    case_type=selected_type,
                    output_dir=output_dir,
                    method_choice=selected_method,  
                    aggregation_method='weighted_max',
                    mc_samples=50,
                    continue_on_error=True
                )
            else:
                print("已取消批量处理。")

        elif mode == '3':
            print("\n🏆 宏平均评估模式")
            print("   此模式将对选定范围内的案例进行批量处理，并计算宏平均指标以提供稳定的性能评估。")
            print(f"   将使用选定的方法: {selected_method}")
            
            confirm = input(f"确认对当前选定的 {len(available_cases)} 个案例进行宏平均评估? (Y/n): ").strip().lower()
            if confirm in ['n', 'no']:
                print("已取消宏平均评估。")
                return

            print(f"\n{'='*80}")
            print(f"🚀 开始宏平均评估流程 ({len(available_cases)} 个案例, 方法: {selected_method})")
            print(f"{'='*80}")
            print("💡 此过程可能需要较长时间，请耐心等待...")

            run_complete_batch_evaluation(
                case_ids=available_cases,
                case_scale=selected_scale,
                case_type=selected_type,
                output_dir=output_dir,
                method_choice=selected_method,  # 传递用户选择的方法
                aggregation_method='weighted_max',
                mc_samples=50
            )

        elif mode == '4':
            print("\n🔬 批量定量定性分析模式")
            print("   此模式假定案例已完成假设生成和条件化不确定性分析，")
            print("   将批量执行定量评价和定性分析步骤。")
            print("   前提：目标案例必须已经通过模式2完成过前期审计流程。")
            
            confirm = input(f"确认对当前选定的 {len(available_cases)} 个案例进行批量定量定性分析? (Y/n): ").strip().lower()
            if confirm in ['n', 'no']:
                print("已取消批量定量定性分析。")
                return

            print(f"\n{'='*80}")
            print(f"🔬 开始批量定量定性分析流程 ({len(available_cases)} 个案例)")
            print(f"{'='*80}")
            print("💡 正在检查必要的输入文件...")

            run_quantitative_qualitative_batch_evaluation(
                case_ids=available_cases,
                case_scale=selected_scale,
                case_type=selected_type,
                output_dir=output_dir,
                base_data_dir=base_data_dir
            )

        elif mode == '5':
            print("\n📊 直接评估模式")
            print("   此模式将基于已有的处理结果直接进行评估分析，无需重新运行案例。")
            print("   前提：目标案例必须已经通过模式2或模式3完成过审计流程。")
            
            confirm = input(f"确认对当前选定的 {len(available_cases)} 个案例进行仅分析评估? (Y/n): ").strip().lower()
            if confirm in ['n', 'no']:
                print("已取消仅分析评估。")
                return

            print(f"\n{'='*80}")
            print(f"📊 开始仅分析评估流程 ({len(available_cases)} 个案例)")
            print(f"{'='*80}")
            print("💡 正在查找已有的审计结果...")

            run_analysis_only_evaluation(
                case_ids=available_cases,
                case_scale=selected_scale,
                case_type=selected_type,
                output_dir=output_dir
            )
        else:
            print("❌ 无效的模式选择。")
            return

        print(f"\n{'='*80}")
        print("🎉 BCSA审计流程完成!")
        print(f"📁 详细结果请查看输出目录: {output_dir}")
        print(f"{'='*80}")

    except (ValueError, IndexError):
        print("\n❌ 输入错误: 请输入有效的序号。程序已退出。")
    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断，程序退出。")
    except Exception as e:
        print(f"\n❌ 程序执行期间发生意外错误: {e}")
        import traceback
        traceback.print_exc()



if __name__ == "__main__":
    main()
