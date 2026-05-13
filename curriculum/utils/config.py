import json
import os
from typing import Dict, Any, Optional, Union

class CurriculumConfig:
    """Curriculum learning configuration management
    
    Used to manage and store configuration parameters for the curriculum learning framework
    """
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """Initialize configuration
        
        Args:
            config_dict: Configuration dictionary, use default if None
        """
        self.config = config_dict or self._default_config()
        
    @staticmethod
    def _default_config() -> Dict[str, Any]:
        """Create default configuration
        
        Returns:
            Default configuration dictionary
        """
        return {
            "task_generator": {
                "type": "default",
                "variation_ranges": {
                    "obstacle_count": (5, 20),
                    "map_size": (500, 700),
                    "target_distance": (200, 600),
                    "uav_count": (2, 5),
                    "uav_speed": (5, 15)
                }
            },
            
            "task_sequencer": {
                "type": "linear",
                "learning_progress_window": 10,
                "performance_threshold": 0.9
            },
            
            "knowledge_transfer": {
                "type": "policy",
                "transfer_ratio": 1.0
            },
            
            "curriculum_manager": {
                "num_initial_tasks": 5,
                "max_episodes_per_task": 1000,
                "max_curriculum_steps": 20
            },
            
            "evaluation": {
                "eval_episodes": 10,
                "eval_frequency": 5,
                "success_threshold": 0.9
            },
            
            "experiment": {
                "num_runs": 5,
                "random_seeds": [42, 123, 456, 789, 1024]
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value
        
        Supports multilevel keys like "task_generator.type"
        
        Args:
            key: Configuration key, supports dot-separated multilevel keys
            default: Default value to return if key doesn't exist
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    def set(self, key: str, value: Any) -> None:
        """设置配置值
        
        支持多级键，如"task_generator.type"
        
        Args:
            key: 配置键，支持点号分隔的多级键
            value: 要设置的值
        """
        keys = key.split('.')
        config = self.config
        for i, k in enumerate(keys[:-1]):
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value
    
    def update(self, config_dict: Dict[str, Any]) -> None:
        """更新配置
        
        Args:
            config_dict: 要更新的配置字典
        """
        def _update_dict(target, source):
            for key, value in source.items():
                if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                    _update_dict(target[key], value)
                else:
                    target[key] = value
        
        _update_dict(self.config, config_dict)
    
    @classmethod
    def from_file(cls, file_path: str) -> 'CurriculumConfig':
        """从文件加载配置
        
        支持JSON文件
        
        Args:
            file_path: 配置文件路径
            
        Returns:
            加载的配置对象
            
        Raises:
            ValueError: 如果文件格式不支持
        """
        _, ext = os.path.splitext(file_path)
        
        if ext.lower() in ['.json']:
            with open(file_path, 'r') as f:
                config_dict = json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
        
        return cls(config_dict)
    
    def to_file(self, file_path: str) -> None:
        """将配置保存到文件
        
        支持JSON文件
        
        Args:
            file_path: 保存路径
            
        Raises:
            ValueError: 如果文件格式不支持
        """
        _, ext = os.path.splitext(file_path)
        
        if ext.lower() in ['.json']:
            with open(file_path, 'w') as f:
                json.dump(self.config, f, indent=2)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
    
    def validate(self) -> Dict[str, str]:
        """验证配置
        
        检查必要的配置项是否存在，类型是否正确
        
        Returns:
            错误信息字典，键为配置路径，值为错误描述。如果没有错误，返回空字典
        """
        errors = {}
        
        # 检查任务生成器配置
        if not isinstance(self.get('task_generator.type'), str):
            errors['task_generator.type'] = "必须是字符串"
        
        # 检查任务排序器配置
        if not isinstance(self.get('task_sequencer.type'), str):
            errors['task_sequencer.type'] = "必须是字符串"
        
        # 检查知识迁移配置
        if not isinstance(self.get('knowledge_transfer.type'), str):
            errors['knowledge_transfer.type'] = "必须是字符串"
        
        # 检查课程管理配置
        if not isinstance(self.get('curriculum_manager.num_initial_tasks'), int) or self.get('curriculum_manager.num_initial_tasks') <= 0:
            errors['curriculum_manager.num_initial_tasks'] = "必须是正整数"
        
        return errors 