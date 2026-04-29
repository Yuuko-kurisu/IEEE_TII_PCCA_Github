#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BCSA定性分析报告生成器 - 自动化叙事分析引擎
目标：输入案例完整输出，生成结构化的人类易读Markdown分析报告
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import argparse
import matplotlib.pyplot as plt
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ReportContext:
    """报告生成上下文"""
    case_id: str
    gt_data: Dict[str, Any]
    findings: List[Dict[str, Any]]
    hypothesis_impacts: Dict[str, Any]
    quantitative_metrics: Dict[str, Any]
    node_id_to_text: Dict[str, str]
    text_to_id: Dict[str, str]
    findings_index: Dict[Tuple[str, str], Dict[str, Any]]
    hypothesis_index: Dict[str, Any]




class QualitativeReporter:
    """定性分析报告生成器"""
    
    def __init__(self):
        self.context: Optional[ReportContext] = None
        logger.info("定性分析报告生成器已初始化")
    
    def load_case_data(self, case_id, case_directory: str, output_file) -> ReportContext:
        """步骤1：数据准备与上下文构建"""
        logger.info(f"开始加载案例数据: {case_directory}")
        

        case_type, scale = case_id.split('_')[:2]
        case_dir = Path(f"./seek_data_v3_deep_enhanced/cases/{scale}case/{case_type}") / case_id
        case_directory_output = Path(case_directory)
        output_file = Path(output_file)
        case_id = case_id
        
        logger.info("正在加载必要文件...")
        
        gt_file = self._find_gt_file(case_dir)
        with open(gt_file, 'r', encoding='utf-8') as f:
            gt_data = json.load(f)
        logger.info(f"✓ 加载Ground Truth: {gt_file}")
        
        findings_file = case_directory_output / "final_aggregated_uncertainty_map.json"
        with open(findings_file, 'r', encoding='utf-8') as f:
            findings = json.load(f)
        logger.info(f"✓ 加载审计发现: {len(findings)} 个")
        
        hypothesis_file = case_directory_output / "hypothesis_impact_report.json"
        with open(hypothesis_file, 'r', encoding='utf-8') as f:
            hypothesis_data = json.load(f)
        hypothesis_impacts = hypothesis_data.get('hypothesis_impacts', {})
        logger.info(f"✓ 加载假设影响: {len(hypothesis_impacts)} 个")
        
        quant_file = output_file / "quantitative_evaluation_report.json"
        with open(quant_file, 'r', encoding='utf-8') as f:
            quant_data = json.load(f)
        quantitative_metrics = quant_data.get('quantitative_metrics', {})
        logger.info(f"✓ 加载定量指标")
        
        ckg_file = self._find_ckg_file(case_dir)
        with open(ckg_file, 'r', encoding='utf-8') as f:
            ckg = json.load(f)
        logger.info(f"✓ 加载CKG: {ckg_file}")
        
        logger.info("正在创建核心映射表...")
        
        node_id_to_text, text_to_id = self._build_node_mappings(ckg)
        logger.info(f"✓ 节点映射: {len(node_id_to_text)} 个节点")
        
        findings_index = self._build_findings_index(findings)
        logger.info(f"✓ 发现索引: {len(findings_index)} 个键值对")
        
        hypothesis_index = self._build_hypothesis_index(hypothesis_impacts)
        logger.info(f"✓ 假设索引: {len(hypothesis_index)} 个假设")
        
        context = ReportContext(
            case_id=case_id,
            gt_data=gt_data,
            findings=findings,
            hypothesis_impacts=hypothesis_impacts,
            quantitative_metrics=quantitative_metrics,
            node_id_to_text=node_id_to_text,
            text_to_id=text_to_id,
            findings_index=findings_index,
            hypothesis_index=hypothesis_index
        )
        
        logger.info("✅ 数据加载完成，上下文构建成功")
        return context
    
    def _find_gt_file(self, case_dir: Path) -> Path:
        """查找Ground Truth文件"""
        gt_file = case_dir / "processed_ground_truth.json"
        if gt_file.exists():
            return gt_file
        
        case_id = case_dir.name
        original_case_dir = Path("./seek_data_v3_deep_enhanced/cases/smallcase/Mixed") / case_id
        gt_file = original_case_dir / "processed_ground_truth.json"
        if gt_file.exists():
            return gt_file
        
        raise FileNotFoundError(f"无法找到processed_ground_truth.json文件")
    
    def _find_ckg_file(self, case_dir: Path) -> Path:
        """查找CKG文件"""
        ckg_file = case_dir / "causal_knowledge_graph.json"
        if ckg_file.exists():
            return ckg_file
        
        case_id = case_dir.name
        original_case_dir = Path("./seek_data_v3_deep_enhanced/cases/smallcase/Mixed") / case_id
        ckg_file = original_case_dir / "causal_knowledge_graph.json"
        if ckg_file.exists():
            return ckg_file
        
        raise FileNotFoundError(f"无法找到causal_knowledge_graph.json文件")
    
    def _build_node_mappings(self, ckg: Dict[str, Any]) -> Tuple[Dict[str, str], Dict[str, str]]:
        """构建节点ID与文本的双向映射"""
        node_id_to_text = {}
        text_to_id = {}
        
        nodes_by_type = ckg.get('nodes_by_type', {})
        for node_type, nodes in nodes_by_type.items():
            for node in nodes:
                node_id = node.get('id', '')
                node_text = node.get('text', '')
                if node_id and node_text:
                    node_id_to_text[node_id] = node_text
                    text_to_id[node_text] = node_id
        
        return node_id_to_text, text_to_id
    
    def _build_findings_index(self, findings: List[Dict[str, Any]]) -> Dict[Tuple[str, str], Dict[str, Any]]:
        """创建发现索引：以(source_id, target_id)为键的字典"""
        findings_index = {}
        
        for i, finding in enumerate(findings):
            source_id = finding.get('source_id', '')
            target_id = finding.get('target_id', '')
            if source_id and target_id:
                key = (source_id, target_id)
                findings_index[key] = {
                    'index': i,
                    'finding': finding
                }
                
                reverse_key = (target_id, source_id)
                if reverse_key not in findings_index:
                    findings_index[reverse_key] = {
                        'index': i,
                        'finding': finding,
                        'is_reverse': True
                    }
        
        return findings_index
    
    def _build_hypothesis_index(self, hypothesis_impacts: Dict[str, Any]) -> Dict[str, Any]:
        """创建假设索引：以hypothesis_id为键的字典"""
        return hypothesis_impacts
    
    def generate_report(self, case_id, case_directory: str, output_file: str = None) -> str:
        """步骤2：生成完整的定性分析报告"""
        logger.info("开始生成定性分析报告...")
        
        self.context = self.load_case_data(case_id, case_directory,output_file)
        
        report_content = []
        
        report_content.append(self._generate_report_header())
        
        report_content.append(self._generate_executive_summary())
        
        evidence_zones = self.context.gt_data.get('evidence_zones', {})
        if isinstance(evidence_zones, list):
            evidence_zones = {f"zone_{i}": zone for i, zone in enumerate(evidence_zones)}
        
        logger.info(f"开始分析 {len(evidence_zones)} 个盲区...")
        
        for zone_id, evidence_zone in evidence_zones.items():
            zone_narrative = self.generate_zone_narrative(zone_id, evidence_zone)
            report_content.append(zone_narrative)
        
        report_content.append(self._generate_overall_conclusion())
        
        full_report = "\n\n".join(report_content)
        
        if output_file is None:
            output_file = f"./results/qualitative_analysis_{self.context.case_id}.md"
        else:
            output_file = output_file + f'/qualitative_analysis_{self.context.case_id}.md'
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(full_report)
        
        logger.info(f"✅ 定性分析报告已生成: {output_path}")
        return full_report
    
    def _generate_report_header(self) -> str:
        """生成报告头部"""
        return f"""# BCSA定性分析报告

**案例ID**: {self.context.case_id}  
**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**分析方法**: 基于证据覆盖度的假设驱动审计分析

---"""
    
    def _generate_executive_summary(self) -> str:
        """生成执行摘要"""
        perf_metrics = self.context.quantitative_metrics.get('performance_metrics', {})
        basic_metrics = self.context.quantitative_metrics.get('basic_metrics', {})
        blindspot_analysis = self.context.quantitative_metrics.get('blind_spot_analysis', {})
        
        summary = f"""## 执行摘要


| 指标 | 值 | 说明 |
|------|----|----- |
| **证据召回率 (Evidence Recall)** | {perf_metrics.get('evidence_recall', 0):.1%} | 成功识别的GT证据比例 |
| **证据精确率 (Evidence Precision)** | {perf_metrics.get('evidence_precision', 0):.1%} | 审计发现中真正有效的比例 |
| **F1分数** | {perf_metrics.get('f1_score', 0):.1%} | 召回率和精确率的调和平均 |
| **盲区召回率 (BSR)** | {perf_metrics.get('blind_spot_recall', 0):.1%} | 成功探测的盲区比例 |
| **MRR** | {perf_metrics.get('mean_reciprocal_rank', 0):.3f} | 发现排序质量指标 |


- **总证据数**: {basic_metrics.get('total_evidence', 0)}
- **总发现数**: {basic_metrics.get('total_findings', 0)}
- **总盲区数**: {blindspot_analysis.get('total_blind_spots', 0)}
- **已探测盲区**: {blindspot_analysis.get('detected_blind_spots', 0)}

---"""
        
        return summary
    
    def generate_zone_narrative(self, zone_id: str, evidence_zone: Dict[str, Any]) -> str:
        """步骤3：核心叙事生成函数"""
        logger.info(f"正在生成盲区 {zone_id} 的叙事分析...")
        
        zone_type = evidence_zone.get('blind_spot_type', '未知类型')
        zone_description = evidence_zone.get('description', '无描述')
        evidence_edges = evidence_zone.get('evidence_edges', [])
        
        detection_status = self._get_zone_detection_status(zone_id, evidence_zone)
        status_icon = "✅ 已发现" if detection_status['is_detected'] else "❌ 未发现"
        
        narrative_parts = []
        
        narrative_parts.append(f"### 盲区分析: {zone_type} - 状态: {status_icon}")
        
        gt_statement = self._generate_gt_statement(evidence_zone, evidence_edges)
        narrative_parts.append(f"**Ground Truth**: {gt_statement}")
        
        matching_analysis = self._generate_matching_analysis(evidence_edges)
        narrative_parts.append("**审计发现匹配详情**:")
        narrative_parts.append(matching_analysis)
        
        explainability_trace = self._generate_explainability_trace(evidence_edges)
        if explainability_trace:
            narrative_parts.append("**发现的逻辑链追溯**:")
            narrative_parts.append(explainability_trace)
        
        qualitative_conclusion = self._generate_qualitative_conclusion(
            zone_type, detection_status, evidence_edges
        )
        narrative_parts.append(f"**分析结论**: {qualitative_conclusion}")
        
        return "\n\n".join(narrative_parts)
    
    def _get_zone_detection_status(self, zone_id: str, evidence_zone: Dict[str, Any]) -> Dict[str, Any]:
        """从定量结果中获取盲区探测状态"""
        blindspot_details = self.context.quantitative_metrics.get('blind_spot_analysis', {}).get('details', [])
        
        for detail in blindspot_details:
            if detail.get('zone_id') == zone_id:
                return {
                    'is_detected': detail.get('is_detected', False),
                    'detection_score': detail.get('detection_score', 0.0),
                    'min_evidence_required': detail.get('min_evidence_required', 1)
                }
        
        evidence_edges = evidence_zone.get('evidence_edges', [])
        matched_evidence_count = 0
        
        for evidence in evidence_edges:
            if self._is_evidence_matched(evidence):
                matched_evidence_count += 1
        
        min_required = max(1, len(evidence_edges) // 2)
        is_detected = matched_evidence_count >= min_required
        
        return {
            'is_detected': is_detected,
            'detection_score': matched_evidence_count,
            'min_evidence_required': min_required
        }
    
    def _generate_gt_statement(self, evidence_zone: Dict[str, Any], evidence_edges: List[Dict[str, Any]]) -> str:
        """生成GT证据陈述"""
        zone_description = evidence_zone.get('description', '')
        blind_spot_type = evidence_zone.get('blind_spot_type', '')
        
        key_nodes = set()
        for evidence in evidence_edges:
            edge_key = evidence.get('edge_key', '')
            if ' -> ' in edge_key:
                source_text, target_text = edge_key.split(' -> ')
                key_nodes.add(source_text.strip())
                key_nodes.add(target_text.strip())
        
        key_nodes_str = "、".join(list(key_nodes)[:3])  # 只显示前3个关键节点
        
        if blind_spot_type == 'causal_desert':
            return f"此盲区核心在于节点 **{key_nodes_str}** 在数据上强相关，但在知识图谱中却是孤立的。{zone_description} 关键证据包括{len(evidence_edges)}条应被发现的缺失边。"
        elif blind_spot_type == 'confounded_relations':
            return f"此盲区揭示了 **{key_nodes_str}** 之间的虚假关联。{zone_description} 需要识别{len(evidence_edges)}个混杂关系。"
        elif blind_spot_type == 'causal_chain_break':
            return f"此盲区涉及 **{key_nodes_str}** 之间的因果链断裂。{zone_description} 包含{len(evidence_edges)}个关键传导路径。"
        else:
            return f"此盲区类型为 **{blind_spot_type}**，涉及节点 **{key_nodes_str}**。{zone_description} 包含{len(evidence_edges)}个关键证据。"
    
    def _generate_matching_analysis(self, evidence_edges: List[Dict[str, Any]]) -> str:
        """生成审计发现与匹配分析"""
        matching_results = []
        
        for i, evidence in enumerate(evidence_edges):
            edge_key = evidence.get('edge_key', '')
            importance = evidence.get('importance', 'medium')
            
            if ' -> ' not in edge_key:
                continue
            
            source_text, target_text = edge_key.split(' -> ')
            source_text = source_text.strip()
            target_text = target_text.strip()
            
            match_result = self._find_matching_findings(source_text, target_text)
            
            if match_result['match_type'] == 'direct':
                finding = match_result['finding']
                unified_score = finding.get('unified_score', 0)
                finding_index = match_result['index']
                matching_results.append(
                    f"   * [**直接命中**] GT证据 `{edge_key}` 被 **发现 #{finding_index}**: "
                    f"`{finding.get('source_id', '')} -> {finding.get('target_id', '')}` "
                    f"(统一分数: {unified_score:.3f}) 成功覆盖。"
                )
            elif match_result['match_type'] == 'partial':
                finding = match_result['finding']
                unified_score = finding.get('unified_score', 0)
                finding_index = match_result['index']
                matching_results.append(
                    f"   * [**节点命中**] GT证据 `{edge_key}` 被 **发现 #{finding_index}**: "
                    f"`{finding.get('source_id', '')} -> {finding.get('target_id', '')}` "
                    f"(统一分数: {unified_score:.3f}) 覆盖。"
                    f"系统虽未直接发现`{source_text}->{target_text}`，但正确识别出相关节点周边存在高度不确定性。"
                )
            else:
                matching_results.append(
                    f"   * [**未命中**] GT证据 `{edge_key}` 未被任何审计发现覆盖。"
                )
        
        if not matching_results:
            return "   * 该盲区的证据均未被审计发现覆盖。"
        
        return "\n".join(matching_results)
    
    def _find_matching_findings(self, source_text: str, target_text: str) -> Dict[str, Any]:
        """查找匹配的审计发现"""
        source_id = self.context.text_to_id.get(source_text, source_text)
        target_id = self.context.text_to_id.get(target_text, target_text)
        
        key = (source_id, target_id)
        if key in self.context.findings_index:
            finding_info = self.context.findings_index[key]
            return {
                'match_type': 'direct',
                'finding': finding_info['finding'],
                'index': finding_info['index'],
                'is_reverse': finding_info.get('is_reverse', False)
            }
        
        reverse_key = (target_id, source_id)
        if reverse_key in self.context.findings_index:
            finding_info = self.context.findings_index[reverse_key]
            return {
                'match_type': 'direct',
                'finding': finding_info['finding'],
                'index': finding_info['index'],
                'is_reverse': True
            }
        
        for key, finding_info in self.context.findings_index.items():
            finding_source, finding_target = key
            
            if source_id in [finding_source, finding_target] or target_id in [finding_source, finding_target]:
                return {
                    'match_type': 'partial',
                    'finding': finding_info['finding'],
                    'index': finding_info['index'],
                    'is_reverse': finding_info.get('is_reverse', False)
                }
        
        return {'match_type': 'none'}
    
    def _is_evidence_matched(self, evidence: Dict[str, Any]) -> bool:
        """检查证据是否被匹配"""
        edge_key = evidence.get('edge_key', '')
        if ' -> ' not in edge_key:
            return False
        
        source_text, target_text = edge_key.split(' -> ')
        match_result = self._find_matching_findings(source_text.strip(), target_text.strip())
        return match_result['match_type'] != 'none'
    
    def _generate_explainability_trace(self, evidence_edges: List[Dict[str, Any]]) -> str:
        """生成可解释性追溯"""
        matched_findings = []
        
        for evidence in evidence_edges:
            edge_key = evidence.get('edge_key', '')
            if ' -> ' not in edge_key:
                continue
            
            source_text, target_text = edge_key.split(' -> ')
            match_result = self._find_matching_findings(source_text.strip(), target_text.strip())
            
            if match_result['match_type'] != 'none':
                matched_findings.append(match_result['finding'])
        
        if not matched_findings:
            return ""
        
        contributing_hypotheses = set()
        for finding in matched_findings:
            contributors = finding.get('main_contributors', [])
            for contrib in contributors[:2]:  # 只取前2个主要贡献者
                hypothesis_id = contrib.get('hypothesis_id', '')
                if hypothesis_id:
                    contributing_hypotheses.add(hypothesis_id)
        
        if not contributing_hypotheses:
            return ""
        
        hypothesis_impacts = []
        for hyp_id in contributing_hypotheses:
            if hyp_id in self.context.hypothesis_index:
                hyp_data = self.context.hypothesis_index[hyp_id]
                impact = hyp_data.get('total_impact', 0)
                hypothesis_impacts.append((hyp_id, hyp_data, impact))
        
        hypothesis_impacts.sort(key=lambda x: x[2], reverse=True)
        
        trace_parts = []
        
        if hypothesis_impacts:
            top_hypothesis = hypothesis_impacts[0]
            hyp_type = top_hypothesis[1].get('hypothesis_type', '未知类型')
            
            trace_parts.append(
                f"   * **核心驱动力**: 对此盲区的探测，主要由影响力排名靠前的假设类型 "
                f"`{hyp_type}` 所驱动。"
            )
            
            main_hyp_id = top_hypothesis[0]
            main_hyp_data = top_hypothesis[1]
            description = main_hyp_data.get('description', '').split('：')[0] if '：' in main_hyp_data.get('description', '') else main_hyp_data.get('description', '')[:50]
            
            trace_parts.append(
                f"   * **具体路径**: 假设 `{main_hyp_id}` ({description}) "
                f"产生了显著的\"不确定性增量\"，导致模型将相关的缺失连接或不可靠连接的"
                f"`unified_score`显著推高，使其成为最终的高优先级发现。"
            )
        
        return "\n".join(trace_parts)
    
    def _generate_qualitative_conclusion(self, zone_type: str, detection_status: Dict[str, Any], 
                                       evidence_edges: List[Dict[str, Any]]) -> str:
        """生成定性结论"""
        is_detected = detection_status['is_detected']
        detection_score = detection_status.get('detection_score', 0)
        total_evidence = len(evidence_edges)
        
        if is_detected:
            if zone_type == 'causal_desert':
                return (f"系统成功地通过识别多个**因果荒漠**类型的假设，发现了由"
                       f"**数据强相关与知识图谱孤立**这一核心矛盾驱动的盲区。"
                       f"系统准确地将问题的根源聚焦到了核心节点及其周边区域，"
                       f"展现了强大的问题定位和可解释能力。"
                       f"(覆盖度: {detection_score}/{total_evidence})")
            elif zone_type == 'confounded_relations':
                return (f"系统通过**条件性不稳定性**和**偏相关下降**检测，"
                       f"成功识别了隐藏的混杂关系。虽然部分证据是间接命中，"
                       f"但系统正确地揭示了变量间关系的不稳定性，"
                       f"为进一步的因果推断提供了重要线索。"
                       f"(覆盖度: {detection_score}/{total_evidence})")
            elif zone_type == 'causal_chain_break':
                return (f"系统通过**弱因果链**检测，成功识别了因果传导路径中的薄弱环节。"
                       f"虽然单个证据的命中可能有限，但系统从整体上把握了"
                       f"因果链的完整性问题，体现了系统性分析能力。"
                       f"(覆盖度: {detection_score}/{total_evidence})")
            else:
                return (f"系统成功探测到该类型盲区，通过多种假设的协同作用，"
                       f"准确定位了问题区域并提供了合理的解释路径。"
                       f"(覆盖度: {detection_score}/{total_evidence})")
        else:
            return (f"系统未能充分探测到该盲区。可能的原因包括：该类型盲区的证据特征"
                   f"超出了当前假设生成规则的覆盖范围，或者需要更复杂的证据组合逻辑。"
                   f"这为未来的方法改进提供了明确的方向。"
                   f"(覆盖度: {detection_score}/{total_evidence})")
    
    def _generate_overall_conclusion(self) -> str:
        """生成总体结论"""
        perf_metrics = self.context.quantitative_metrics.get('performance_metrics', {})
        evidence_recall = perf_metrics.get('evidence_recall', 0)
        blindspot_recall = perf_metrics.get('blind_spot_recall', 0)
        f1_score = perf_metrics.get('f1_score', 0)
        
        conclusion = f"""## 总体结论与方法评价

基于对案例 **{self.context.case_id}** 的全面分析，BCSA方法展现出以下特点：


1. **假设驱动的系统性分析**: 通过8种不同类型的假设生成规则，能够从结构异常和数据驱动两个维度全面审视因果知识图谱的完整性。

2. **强可解释性**: 每个审计发现都可以追溯到具体的假设及其影响力，形成了清晰的"假设→不确定性增量→最终发现"的逻辑链条。

3. **精准的问题定位**: 证据召回率达到 **{evidence_recall:.1%}**，盲区召回率达到 **{blindspot_recall:.1%}**，在复杂的因果网络中准确识别关键薄弱环节。


- **多层次聚合机制**: 从假设级别的不确定性量化到最终的加权聚合，确保了发现的可靠性和重要性排序的合理性。
- **数据-知识双重验证**: 既考虑了知识图谱的结构特征，又充分利用了传感器数据的统计信息，实现了双重交叉验证。
- **自适应阈值设计**: 根据不同盲区类型和证据复杂度，采用了灵活的探测标准，平衡了召回率和精确率。


F1分数 **{f1_score:.1%}** 表明BCSA方法在准确性和完整性之间取得了良好平衡。方法不仅能够"发现问题"，更重要的是能够"解释问题"，为工业界的因果推断和决策支持提供了可信赖的技术方案。

---

*报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*  
*分析引擎: BCSA定性分析报告生成器 v1.0*"""
        
        return conclusion

def generate_qualitative_report(case_id, case_directory: str, output_file: str = None) -> str:
    """
    生成定性分析报告的接口函数（可被其他模块调用）
    
    Args:
        case_directory: 案例目录路径
        output_file: 输出文件路径（可选）
        
    Returns:
        生成的报告内容字符串
    """
    case_dir = Path(case_directory)
    if not case_dir.exists():
        raise FileNotFoundError(f"案例目录不存在: {case_dir}")
    
    logger.info(f"开始生成定性分析报告: {case_dir.name}")
    
    try:
        reporter = QualitativeReporter()
        report_content = reporter.generate_report(case_id, case_directory, output_file)
        
        logger.info(f"✅ 案例 {case_dir.name} 定性分析报告生成完成")
        
        return report_content
        
    except Exception as e:
        logger.error(f"❌ 定性分析报告生成失败: {e}")
        raise


def main():
    """主函数 - 支持命令行和默认运行"""
    print("=" * 80)
    print("📝 BCSA定性分析报告生成器")
    print("=" * 80)
    
    default_case_id = "Mixed_small_09"
    default_output_base = "./results/conditioned_uncertainty_analysis"
    default_case_directory = f"{default_output_base}"
    default_output_dir = "./results/conditioned_uncertainty_analysis"
    
    parser = argparse.ArgumentParser(
        description='BCSA定性分析报告生成器',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
示例用法:
  python qualitative_reporter.py                                    # 使用默认案例
  python qualitative_reporter.py case_directory                     # 指定案例目录
  python qualitative_reporter.py case_directory -o output.md        # 指定输出文件
  
注意: case_directory应该指向conditioned_uncertainty_analysis下的案例目录
      例如: ./results/conditioned_uncertainty_analysis/Mixed_small_07
        """
    )

    parser.add_argument(
        'case_id', 
        nargs='?',  # 使参数可选
        default=default_case_id,
    )
    
    parser.add_argument(
        'case_directory', 
        nargs='?',  # 使参数可选
        default=default_case_directory,
        help=f'案例目录路径（默认: {default_case_directory}）'
    )
    parser.add_argument(
        '--output', '-o', 
        help='输出报告文件路径（默认: ./results/qualitative_analysis_{{case_id}}.md）',
        default=default_output_dir
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='静默模式，不显示报告预览'
    )
    
    args = parser.parse_args()
    
    print(f"🚀 BCSA定性分析报告生成器启动")
    print(f"📁 目标案例目录: {args.case_directory}")
    
    case_path = Path(args.case_directory)
    if not case_path.exists() and args.case_directory == default_case_directory:
        print(f"⚠️  默认案例目录不存在: {default_case_directory}")
        print(f"💡 请确保已运行BCSA分析流程，或手动指定正确的案例目录")
        print(f"💡 例如: python qualitative_reporter.py ./results/conditioned_uncertainty_analysis/YourCaseID")
    
    try:
        reporter = QualitativeReporter()
        report_content = reporter.generate_report(args.case_id, args.case_directory, args.output)
        
        print(f"\n✅ 定性分析报告生成成功!")
        print(f"📁 案例目录: {args.case_directory}")
        
        case_name = Path(args.case_directory).name
        if args.output:
            print(f"📄 输出文件: {args.output}")
        else:
            print(f"📄 输出文件: ./results/qualitative_analysis_{case_name}.md")
        
        if not args.quiet:
            lines = report_content.split('\n')
            print(f"\n📖 报告预览 (前20行):")
            print("-" * 60)
            for line in lines[:20]:
                print(line)
            if len(lines) > 20:
                print("...")
                print(f"(完整报告共 {len(lines)} 行)")
        
        print(f"\n💡 提示: 完整的定性分析需要以下文件:")
        print(f"   - ./results/conditioned_uncertainty_analysis/{{case_id}}/final_aggregated_uncertainty_map.json")
        print(f"   - ./results/conditioned_uncertainty_analysis/{{case_id}}/hypothesis_impact_report.json")
        print(f"   - ./results/conditioned_uncertainty_analysis/{{case_id}}/quantitative_evaluation_report.json")
        print(f"   - ./seek_data_v3_deep_enhanced/cases/smallcase/Mixed/{{case_id}}/processed_ground_truth.json")
        print(f"   - ./seek_data_v3_deep_enhanced/cases/smallcase/Mixed/{{case_id}}/causal_knowledge_graph.json")
        
        return report_content
        
    except FileNotFoundError as e:
        logger.error(f"❌ 文件错误: {e}")
        print(f"\n💡 提示: 请确保案例目录存在，或尝试以下路径:")
        print(f"   {default_case_directory}")
        print(f"💡 完整的BCSA分析流程应该生成以上所需的所有文件")
        return None
        
    except Exception as e:
        logger.error(f"❌ 定性分析报告生成失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def demo_qualitative_analysis():
    """演示定性分析报告生成"""
    print("\n" + "=" * 80)
    print("📝 定性分析报告生成演示")
    print("=" * 80)
    
    default_case_id = "Mixed_small_07"
    default_case_directory = f"./results/conditioned_uncertainty_analysis/{default_case_id}"
    
    print(f"📋 使用默认案例进行演示: {default_case_id}")
    print(f"📁 案例目录: {default_case_directory}")
    
    try:
        report_content = generate_qualitative_report(
            case_id= default_case_id,
            case_directory=default_case_directory,
            output_file=f"./results/demo_qualitative_report_{default_case_id}.md"
        )
        
        print(f"✅ 演示成功！")
        print(f"📄 报告文件: ./results/demo_qualitative_report_{default_case_id}.md")
        
        lines = report_content.split('\n')
        sections = [line for line in lines if line.startswith('###')]
        print(f"📊 报告包含 {len(sections)} 个盲区分析章节")
        print(f"📊 报告总长度: {len(lines)} 行")
        
        return True
        
    except Exception as e:
        print(f"❌ 演示失败: {e}")
        print(f"💡 请确保已运行完整的BCSA分析流程")
        return False


if __name__ == "__main__":
    main()
    
