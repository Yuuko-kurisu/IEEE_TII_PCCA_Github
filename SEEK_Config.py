#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SEEK框架统一配置管理模块
提供标准化的配置、路径管理和数据接口
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import json

class SEEKModule(Enum):
    """SEEK模块枚举"""
    HYPOTHESIS_GENERATION = "BCSA_01"
    PC_VGAE_AUDIT = "BCSA_02" 
    EVALUATION = "BCSA_03"
    CASE_VALIDATION = "BCSA_04"

@dataclass
class SEEKPaths:
    """SEEK框架路径配置"""
    base_dir: Path = field(default_factory=lambda: Path(os.getcwd()))
    data_dir: Path = field(default_factory=lambda: Path("./seek_data_v3_deep_enhanced"))
    
    case_scale: str = "smallcase"
    case_type: str = "Mixed"
    case_id: str = "Mixed_small_01"
    case_dir: Path = field(init=False)
    results_dir: Path = field(init=False)
    official_output_dir: Path = field(init=False)
    
    hypothesis_output_dir: Path = field(init=False)
    audit_output_dir: Path = field(init=False)
    evaluation_output_dir: Path = field(init=False)
    case_output_dir: Path = field(init=False)
    
    def __post_init__(self):
        """初始化后设置派生路径"""


        self.case_dir = self.data_dir / "cases" / self.case_scale / self.case_type / self.case_id
        self.results_dir = self.data_dir / "results" / self.case_scale / self.case_type / self.case_id


        self.official_output_dir = self.results_dir / "BCSA_Analysis" / "1_Final_Results"
        
        self.hypothesis_output_dir = self.official_output_dir / "hypothesis_generation"
        self.audit_output_dir = self.official_output_dir / "pc_vgae_audit"
        self.evaluation_output_dir = self.official_output_dir / "evaluation"
        self.case_output_dir = self.official_output_dir / "case_validation"
        
        for dir_path in [self.hypothesis_output_dir, self.audit_output_dir, 
                        self.evaluation_output_dir, self.case_output_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def get_hypothesis_file(self) -> Path:
        """获取假设生成输出文件路径"""
        return self.hypothesis_output_dir / f"generated_hypotheses.json"
    
    def get_coverage_file(self) -> Path:
        """获取覆盖度报告文件路径"""
        return self.hypothesis_output_dir / f"coverage_report_{self.case_id}.json"
    
    def get_audit_file(self) -> Path:
        """获取审计输出文件路径"""
        return self.audit_output_dir / f"uncertainty_map_{self.case_id}.json"
    
    def get_evaluation_file(self) -> Path:
        """获取评价输出文件路径"""
        return self.evaluation_output_dir / f"evaluation_report_{self.case_id}.json"
    
    def get_case_file(self) -> Path:
        """获取案例验证输出文件路径"""
        return self.case_output_dir / f"case_validation_{self.case_id}.json"

@dataclass
class HypothesisGenerationConfig:
    """假设生成模块配置"""
    target_coverage: float = 0.95
    enable_dual_output: bool = True
    
@dataclass
class PCVGAEConfig:
    """PC-VGAE审计模块配置"""
    latent_dim: int = 16
    learning_rate: float = 0.005
    weight_decay: float = 1e-4
    max_epochs: int = 300
    early_stopping_patience: int = 30
    kl_weight: float = 0.01
    monte_carlo_samples: int = 100
    grad_clip_norm: float = 1.0
    min_clamp_value: float = 1e-8
    max_clamp_value: float = 1e8
    iqr_multiplier: float = 1.0  # IQR乘数因子，1.0比标准1.5更宽松，检测更多温和异常值
    zscore_multiplier: float = 2.0  # Z-score乘数因子
    
@dataclass
class EvaluationConfig:
    """评价模块配置"""
    target_accuracy: float = 0.65  # 调整到更现实的目标
    confidence_threshold: float = 0.8
    uncertainty_threshold: float = 0.3
    
@dataclass
class CaseValidationConfig:
    """案例验证模块配置"""
    validation_scenarios: List[str] = field(default_factory=lambda: [
        "before_optimization", "after_optimization", "comparative_analysis"
    ])
    
@dataclass
class SEEKConfig:
    """SEEK框架总配置"""
    paths: SEEKPaths
    hypothesis_config: HypothesisGenerationConfig = field(default_factory=HypothesisGenerationConfig)
    pcvgae_config: PCVGAEConfig = field(default_factory=PCVGAEConfig)
    evaluation_config: EvaluationConfig = field(default_factory=EvaluationConfig)
    case_config: CaseValidationConfig = field(default_factory=CaseValidationConfig)
    
    device: str = "cpu"  # 由于torch依赖问题，默认使用CPU
    random_seed: int = 42
    debug_mode: bool = False
    
    
    @classmethod
    def create_for_case(cls, case_id: str, case_scale: str, case_type: str, **kwargs) -> 'SEEKConfig':
        """为特定案例创建配置，现在需要 scale 和 type"""
        paths = SEEKPaths(case_id=case_id, case_scale=case_scale, case_type=case_type)
        config = cls(paths=paths, **kwargs)
        return config

    def save_config(self, file_path: Optional[Path] = None) -> Path:
        """保存配置到文件"""
        if file_path is None:
            file_path = self.paths.official_output_dir / "seek_config.json"
        
        config_dict = {
            'case_id': self.paths.case_id,
            'hypothesis_config': self.hypothesis_config.__dict__,
            'pcvgae_config': self.pcvgae_config.__dict__,
            'evaluation_config': self.evaluation_config.__dict__,
            'case_config': self.case_config.__dict__,
            'device': self.device,
            'random_seed': self.random_seed,
            'debug_mode': self.debug_mode
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, ensure_ascii=False, indent=2)
        
        return file_path
    
    @classmethod
    def load_config(cls, file_path: Path) -> 'SEEKConfig':
        """从文件加载配置"""
        with open(file_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        case_id = config_dict['case_id']
        paths = SEEKPaths(case_id=case_id)
        
        return cls(
            paths=paths,
            hypothesis_config=HypothesisGenerationConfig(**config_dict['hypothesis_config']),
            pcvgae_config=PCVGAEConfig(**config_dict['pcvgae_config']),
            evaluation_config=EvaluationConfig(**config_dict['evaluation_config']),
            case_config=CaseValidationConfig(**config_dict['case_config']),
            device=config_dict['device'],
            random_seed=config_dict['random_seed'],
            debug_mode=config_dict['debug_mode']
        )

@dataclass
class SEEKDataInterface:
    """SEEK框架数据接口"""
    
    @staticmethod
    def load_hypotheses(file_path: Path) -> List[Dict[str, Any]]:
        """加载假设生成结果"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    @staticmethod
    def load_coverage_report(file_path: Path) -> Dict[str, Any]:
        """加载覆盖度报告"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    @staticmethod
    def save_uncertainty_map(data: Dict[str, Any], file_path: Path) -> None:
        """保存不确定性地图"""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    
    @staticmethod
    def save_evaluation_report(data: Dict[str, Any], file_path: Path) -> None:
        """保存评价报告"""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    
    @staticmethod
    def save_case_validation(data: Dict[str, Any], file_path: Path) -> None:
        """保存案例验证结果"""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)

def create_seek_config_for_batch_test(case_ids: List[str]) -> List[SEEKConfig]:
    """为批量测试创建配置"""
    configs = []
    for case_id in case_ids:
        config = SEEKConfig.create_for_case(case_id)
        configs.append(config)
    return configs

DEFAULT_TEST_CASES = [f"Mixed_small_{i:02d}" for i in range(1, 11)]

if __name__ == "__main__":
    print("=== SEEK框架配置管理演示 ===")
    
    config = SEEKConfig.create_for_case("Mixed_small_01")
    print(f"案例配置: {config.paths.case_id}")
    print(f"假设文件路径: {config.paths.get_hypothesis_file()}")
    print(f"审计输出路径: {config.paths.get_audit_file()}")
    
    config_file = config.save_config()
    print(f"配置已保存: {config_file}")
    
    batch_configs = create_seek_config_for_batch_test(DEFAULT_TEST_CASES[:3])
    print(f"批量配置创建: {len(batch_configs)} 个案例")
    
    print("✅ 配置管理模块演示完成")
