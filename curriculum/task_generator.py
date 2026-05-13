import numpy as np
import random
import uuid
from typing import Dict, List, Any, Optional, Tuple, Union
from .task import Task
from .utils.config import CurriculumConfig

class TaskGenerator:
    """任务生成器基类
    
    负责生成不同难度的任务，由具体子类实现生成策略
    """
    
    def __init__(self, config: Optional[CurriculumConfig] = None):
        """初始化任务生成器
        
        Args:
            config: 配置对象，如果为None则创建默认配置
        """
        self.config = config or CurriculumConfig()
        
    def generate_task(self, difficulty: Optional[float] = None) -> Task:
        """生成一个指定难度的任务
        
        Args:
            difficulty: 任务难度-1之间的浮点数表示最简单
            
        Returns:
            生成的Task对象
        """
        raise NotImplementedError("子类必须实现此方法")
    
    def generate_tasks(self, count: int, difficulty_range: Optional[Tuple[float, float]] = None) -> List[Task]:
        """批量生成指定难度范围内的任务
        
        Args:
            count: 要生成的任务数量
            difficulty_range: 难度范围元组(min_difficulty, max_difficulty)，默认为(0, 1)
            
        Returns:
            生成的Task对象列表
        """
        if difficulty_range is None:
            difficulty_range = (0.0, 1.0)
            
        min_diff, max_diff = difficulty_range
        tasks = []
        
        for _ in range(count):
            difficulty = min_diff + random.random() * (max_diff - min_diff)
            tasks.append(self.generate_task(difficulty))
            
        return tasks


class DefaultTaskGenerator(TaskGenerator):
    """默认任务生成器
    
    基于环境参数的线性插值生成任务
    """
    
    def __init__(self, config: Optional[CurriculumConfig] = None):
        """初始化默认任务生成器
        
        Args:
            config: 配置对象
        """
        super().__init__(config)
        self.variation_ranges = self.config.get("task_generator.variation_ranges", {
            "leader_count": (1, 1),        # 主机数量，默认固定为1
            "follower_count": (1, 4),      # 从机数量
            "obstacle_count": (1, 3),      # 障碍物数量
            "map_size": (700, 1000),       # 地图尺寸，应C.SCREEN_SIZE一致
            "target_distance": (200, 600), # 目标距离
            "uav_speed": (10, 20)          # 无人机速度
        })
        
        # 从配置中获取是否需要渲染
        self.render = self.config.get("render", False)
        print(f"TaskGenerator - 渲染设置: {'开' if self.render else '关闭'}")
        
    def generate_task(self, difficulty: Optional[float] = None) -> Task:
        """生成一个指定难度的任务
        
        根据难度系数线性插值环境参数
        
        Args:
            difficulty: 任务难度-1之间的浮点数表示最简单
            
        Returns:
            生成的Task对象
        """
        if difficulty is None:
            difficulty = random.random()
            
        # 确保难度在0-1范围内
        difficulty = max(0.0, min(1.0, difficulty))
        
        # 根据难度插值生成环境参数
        env_params = self._interpolate_params(difficulty)
        
        # 生成唯一ID
        task_id = f"task_{uuid.uuid4().hex[:8]}"
        
        return Task(task_id=task_id, env_params=env_params, difficulty=difficulty)
    
    def _interpolate_params(self, difficulty: float) -> Dict[str, Any]:
        """根据难度插值生成环境参数
        
        Args:
            difficulty: 0-1之间的难度系数
            
        Returns:
            环境参数字典，与RlGame类兼容
        """
        env_params = {}
        
        # 主机数量 - 使用阈值函数使难度变化更明显
        min_leader, max_leader = self.variation_ranges["leader_count"]
        if difficulty < 0.3:
            env_params["leader_count"] = min_leader
        elif difficulty < 0.7:
            env_params["leader_count"] = min(max_leader, min_leader + 1)
        else:
            env_params["leader_count"] = max_leader
        
        # 从机数量 - 使用指数函数使变化更明显
        min_follower, max_follower = self.variation_ranges["follower_count"]
        difficulty_squared = difficulty ** 2  # 使曲线更陡峭
        follower_count = int(min_follower + difficulty_squared * (max_follower - min_follower))
        env_params["follower_count"] = max(min_follower, min(max_follower, follower_count))
        
        # 障碍物数量 - 使用阶梯函数
        min_obs, max_obs = self.variation_ranges["obstacle_count"]
        if difficulty < 0.2:
            obs_factor = 0.0  # 低难度无障碍
        elif difficulty < 0.4:
            obs_factor = 0.2  # 中低难度少量障碍
        elif difficulty < 0.6:
            obs_factor = 0.5  # 中等难度适量障碍
        elif difficulty < 0.8:
            obs_factor = 0.8  # 中高难度较多障碍
        else:
            obs_factor = 1.0  # 高难度大量障碍物
            
        obstacle_count = int(min_obs + obs_factor * (max_obs - min_obs))
        env_params["obstacle_count"] = obstacle_count
        
        # 使用TaskGenerator的渲染设置
        env_params["render"] = self.render
        
        # 无人机速度 - 随难度增加而增加
        min_speed, max_speed = self.variation_ranges["uav_speed"]
        uav_speed = min_speed + difficulty * (max_speed - min_speed)
        
        # 使用PositionGenerator统一生成位置
        from rl_env.components.position_generator import PositionGenerator
        
        # 生成主机位置
        env_params["leader_init_pos"] = PositionGenerator.generate_leader_position()
        
        # 生成目标位置
        env_params["goal_init_pos"] = PositionGenerator.generate_goal_position()
        
        # 生成从机位置
        env_params["follower_init_pos"] = PositionGenerator.generate_follower_position()
        
        # 生成障碍物位置
        env_params["obstacle_init_pos"] = PositionGenerator.generate_obstacle_position()
        
        # 保存难度信息，可以在创建环境后用于调整其他参数
        env_params["difficulty"] = difficulty
        env_params["uav_speed"] = uav_speed
        
        return env_params


class ProgressiveTaskGenerator(TaskGenerator):
    """渐进式任务生成器
    
    基于学习进度动态调整任务难度
    """
    
    def __init__(self, config: Optional[CurriculumConfig] = None, base_generator: Optional[TaskGenerator] = None):
        """初始化渐进式任务生成器
        
        Args:
            config: 配置对象
            base_generator: 基础任务生成器，用于实际生成任务
        """
        super().__init__(config)
        self.base_generator = base_generator or DefaultTaskGenerator(config)
        
    def generate_task(self, difficulty: Optional[float] = None) -> Task:
        """生成一个任务
        
        Args:
            difficulty: 任务难度
            
        Returns:
            生成的Task对象
        """
        return self.base_generator.generate_task(difficulty)
    
    def generate_next_task(self, tasks: List[Task], agent_performance: Dict[str, float], window: int = 15) -> Task:
        """根据已有任务和智能体性能生成下一个任务
        
        基于学习曲线和性能调整难度
        
        Args:
            tasks: 已有任务列表
            agent_performance: 智能体当前性能指标
            window: 用于计算学习进度的窗口大小
            
        Returns:
            生成的下一个Task对象
        """
        # 计算平均学习进度 - 使用指定的窗口大小
        learning_progress = 0.0
        if tasks:
            progress_values = [task.calculate_learning_progress(window=window) for task in tasks]
            learning_progress = np.mean([p for p in progress_values if p > 0])
        
        # 计算已完成任务的平均难度
        solved_tasks = [task for task in tasks if task.is_solved(window=window)]
        avg_solved_difficulty = 0.3  # 默认中等难度
        if solved_tasks:
            difficulties = [task.difficulty for task in solved_tasks if task.difficulty is not None]
            if difficulties:
                avg_solved_difficulty = np.mean(difficulties)
        
        # 考虑当前性能
        success_rate = agent_performance.get('success_rate', 0.0)
        reward = agent_performance.get('reward', 0.0)
        
        # 根据学习进度和性能动态调整难度增加量
        difficulty_increment = 0.1  # 默认增加0.1
        
        # 如果学习进度好且表现优秀，大幅增加难度
        if learning_progress > 0.1 and (success_rate > 0.8 or reward > 100):
            difficulty_increment = 0.2
        # 如果学习进度一般，小幅增加难度
        elif learning_progress > 0.05 or (success_rate > 0.6 or reward > 50):
            difficulty_increment = 0.1
        # 如果学习进度差，微小增加难度
        else:
            difficulty_increment = 0.05
        
        # 计算新难度
        new_difficulty = min(1.0, avg_solved_difficulty + difficulty_increment)
        
        # 随机扰动，避免任务总是在同一难度级别
        if random.random() < 0.3:  # 30%概率增加更大的随机扰动
            new_difficulty = new_difficulty + random.uniform(-0.1, 0.2)
        else:  # 常规随机扰动
            new_difficulty = new_difficulty + random.uniform(-0.05, 0.05)
        
        # 确保难度在有效范围内
        new_difficulty = max(0.1, min(1.0, new_difficulty))
        
        print(f"生成新任务，基础难度: {avg_solved_difficulty:.2f}，增加 {difficulty_increment:.2f}，最终难度 {new_difficulty:.2f}")
        
        # 生成新任务
        return self.generate_task(new_difficulty)


class FixedTaskGenerator(TaskGenerator):
    """固定任务生成器
    
    提供一组预定义的任务序列
    """
    
    # 预定义默认难度级别，子类可继承并覆盖
    DEFAULT_DIFFICULTY_LEVELS = [] # 保持类默认为空
    
    # 特定任务序列配置
    SPECIFIC_TASKS_CONFIG = [
        # 任务 1: 1 主机, 1 从机, 0 障碍物
        {"task_id_suffix": "L1_F1_O0", "hero_count": 1, "enemy_count": 1, "obstacle_count": 0, "map_size": (800, 600), "target_distance": (300, 300)},
        # 任务 2: 1 主机, 1 从机, 1 障碍物
        {"task_id_suffix": "L1_F1_O1", "hero_count": 1, "enemy_count": 1, "obstacle_count": 1, "map_size": (800, 600), "target_distance": (300, 300)},
        # 任务 3: 1 主机, 1 从机, 2 障碍物
        {"task_id_suffix": "L1_F1_O2", "hero_count": 1, "enemy_count": 1, "obstacle_count": 2, "map_size": (800, 600), "target_distance": (300, 300)},
        # 任务 4: 1 主机, 2 从机, 2 障碍物
        {"task_id_suffix": "L1_F2_O2", "hero_count": 1, "enemy_count": 2, "obstacle_count": 2, "map_size": (800, 600), "target_distance": (300, 300)},
        # 任务 5: 1 主机, 3 从机, 2 障碍物
        {"task_id_suffix": "L1_F3_O2", "hero_count": 1, "enemy_count": 3, "obstacle_count": 2, "map_size": (800, 600), "target_distance": (300, 300)},
    ]
    
    def __init__(self, config: Optional[CurriculumConfig] = None):
        """初始化固定任务生成器
        
        Args:
            config: 配置对象
        """
        super().__init__(config)
        self.config = config if config else CurriculumConfig()
        self.predefined_task_configs: Dict[float, Task] = {}

        # 从配置获取预定义难度级别
        difficulty_keys = self._get_predefined_difficulty_levels()

        if not difficulty_keys:
            print("FixedTaskGenerator: 配置未设置或为空。无法创建预定义任务")
            return

        # 严格要求难度键数量与特定任务配置数量一致
        if len(difficulty_keys) != len(self.SPECIFIC_TASKS_CONFIG):
            print(f"FixedTaskGenerator: 配置的难度键数量 ({len(difficulty_keys)}) 与SPECIFIC_TASKS_CONFIG 中的任务数量 ({len(self.SPECIFIC_TASKS_CONFIG)}) 不匹配。请检查配置")
            return # 提前返回

        # 从配置中获取是否需要渲染
        self.render = self.config.get("render", False)
        print(f"FixedTaskGenerator - 渲染设置: {'开' if self.render else '关闭'}")

        print("FixedTaskGenerator: 正在使用预定义的特定任务序列进行初始化")
        for i, task_spec_params in enumerate(self.SPECIFIC_TASKS_CONFIG):
            difficulty_key = difficulty_keys[i] # 使用配置中对应顺序的难度键

            # 将task_spec_params与CurriculumConfig的默认值构造成env_params
            env_params = {
                "leader_count": task_spec_params["hero_count"],
                "follower_count": task_spec_params["enemy_count"],
                "obstacle_count": task_spec_params["obstacle_count"],
                "map_size": task_spec_params.get("map_size", self.config.get("task_generator.variation_ranges.map_size", [(800, 600)])[0]),
                "target_distance": task_spec_params.get("target_distance", self.config.get("task_generator.variation_ranges.target_distance", [(300, 300)])[0]),
                # 确保包含渲染设置
                "render": self.render,
            }

            # 获取预定义的实体位置；如果特定难度键没有对应位置，则默认为空字典
            current_positions = self._create_predefined_positions().get(difficulty_key, {})
            
            # 将位置信息添加到 env_params
            for pos_key, pos_value in current_positions.items():
                if pos_key == "leader_init_pos":
                    env_params["leader_init_pos"] = pos_value
                elif pos_key == "follower_init_pos":
                    env_params["follower_init_pos"] = pos_value
                elif pos_key == "obstacle_init_pos":
                    env_params["obstacle_init_pos"] = pos_value
                elif pos_key == "goal_init_pos":
                    env_params["goal_init_pos"] = pos_value
            
            task_id_str = f"fixed_seq_task_{task_spec_params.get('task_id_suffix', str(i + 1))}_d{difficulty_key:.2f}" # 确保ID唯一且包含难度信息

            # 直接创建 Task 对象
            new_task = Task(
                task_id=task_id_str,
                env_params=env_params,
                difficulty=difficulty_key
            )
            self.predefined_task_configs[difficulty_key] = new_task
            print(f"创建固定序列任务: ID={new_task.id}, 难度={new_task.difficulty:.2f}, 参数={env_params}")
        
        # 当前任务索引，用于generate_next_task方法
        self.current_task_index = 0

    def _create_predefined_positions(self) -> Dict[float, Dict[str, List[Tuple[float, float]]]]:
        """创建预定义位置
        
        Returns:
            按难度为键的位置字典
        """
        # 使用PositionGenerator生成位置
        from rl_env.components.position_generator import PositionGenerator
        
        # 对每个难度级别创建固定位置
        positions_by_difficulty = {}
        
        # 获取预定义难度级别
        difficulty_keys = self._get_predefined_difficulty_levels()
        if not difficulty_keys or not self.SPECIFIC_TASKS_CONFIG or len(difficulty_keys) != len(self.SPECIFIC_TASKS_CONFIG):
            print("Warning: 难度键与特定任务数量不匹配，无法创建完全匹配的预定义位置")
            # 此时返回一个空字典，后续初始化流程会检测并提前返回
            return {}
            
        # 创建固定位置
        for i, task_spec in enumerate(self.SPECIFIC_TASKS_CONFIG):
            difficulty_key = difficulty_keys[i]
            
            # 获取环境参数
            leader_count = task_spec["hero_count"]
            follower_count = task_spec["enemy_count"]
            obstacle_count = task_spec["obstacle_count"]
            
            # 为简化起见，直接生成位置
            positions = {
                "leader_init_pos": [PositionGenerator.generate_leader_position() for _ in range(leader_count)],
                "follower_init_pos": [PositionGenerator.generate_follower_position() for _ in range(follower_count)],
                "obstacle_init_pos": [PositionGenerator.generate_obstacle_position() for _ in range(obstacle_count)],
                "goal_init_pos": [PositionGenerator.generate_goal_position()]  # 总是只有一个目标
            }
            
            positions_by_difficulty[difficulty_key] = positions
        
        return positions_by_difficulty
    
    def generate_task(self, difficulty: Optional[float] = None) -> Task:
        """生成指定难度的任务
        
        从预定义任务集中选择难度最接近的任务
        
        Args:
            difficulty: 任务难度，为None时返回第一个任务
            
        Returns:
            生成的Task对象
        """
        if not self.predefined_task_configs:
            print("错误: 没有可用的预定义任务，请检查是否正确初始化")
            raise ValueError("FixedTaskGenerator 未包含任何预定义任务")
            
        if difficulty is None:
            # 返回第一个预定义任务
            first_key = list(sorted(self.predefined_task_configs.keys()))[0]
            return self.predefined_task_configs[first_key]
        
        # 找到难度最接近的任务
        closest_key = min(self.predefined_task_configs.keys(), 
                          key=lambda d: abs(d - difficulty))
        
        return self.predefined_task_configs[closest_key]
    
    def generate_tasks(self, count: int, difficulty_range: Optional[Tuple[float, float]] = None) -> List[Task]:
        """批量生成指定难度范围内的任务
        
        对于固定任务集，将返回预定义任务集的子集
        
        Args:
            count: 要生成的任务数量
            difficulty_range: 难度范围
            
        Returns:
            生成的Task对象列表
        """
        if not self.predefined_task_configs:
            print("错误: 没有可用的预定义任务，请检查是否正确初始化")
            return []
            
        if count >= len(self.predefined_task_configs):
            # 如果请求的数量大于预定义任务数，返回所有预定义任务
            return list(self.predefined_task_configs.values())
        
        # 否则，选择难度适合范围的任务
        if difficulty_range is None:
            # 无范围限制，选择前count个任务
            sorted_keys = sorted(self.predefined_task_configs.keys())
            selected_tasks = [self.predefined_task_configs[k] for k in sorted_keys[:count]]
        else:
            min_diff, max_diff = difficulty_range
            selected_keys = [
                k for k in self.predefined_task_configs.keys()
                if min_diff <= k <= max_diff
            ]
            sorted_selected_keys = sorted(selected_keys)[:count]
            selected_tasks = [self.predefined_task_configs[k] for k in sorted_selected_keys]
        
        return selected_tasks
    
    def generate_next_task(self, tasks: List[Task], agent_performance: Dict[str, float], window: int = 15) -> Task:
        """生成下一个任务
        
        按预定义顺序返回下一个任务
        
        Args:
            tasks: 已完成任务列表
            agent_performance: 智能体性能指标
            window: 用于计算性能的窗口大小
            
        Returns:
            下一个Task对象
        """
        if not self.predefined_task_configs:
            print("错误: 没有可用的预定义任务，请检查是否正确初始化")
            raise ValueError("FixedTaskGenerator 未包含任何预定义任务")
            
        # 检查当前任务是否已解决
        sorted_keys = sorted(self.predefined_task_configs.keys())
        
        if tasks and self.current_task_index < len(sorted_keys):
            latest_task = tasks[-1]
            
            # 如果找到当前任务，检查是否已解决
            if latest_task and latest_task.is_solved(window=window):
                self.current_task_index += 1
                print(f"当前任务已解决，进入下一个任务，难度级别: {self.current_task_index + 1}/{len(sorted_keys)}")
        
        # 返回下一个任务，或者如果已完成所有任务，则返回最后一个任务
        next_index = min(self.current_task_index, len(sorted_keys) - 1)
        next_key = sorted_keys[next_index]
        return self.predefined_task_configs[next_key]
    
    # 获取预定义难度级别列表
    def _get_predefined_difficulty_levels(self) -> List[float]:
        """获取预定义的难度级别列表。
        
        Returns:
            List[float]: 按从易到难排序的难度级别列表
        """
        # 优先从配置获取
        levels_from_config = self.config.get("fixed_tasks_config.difficulty_levels")
        
        if levels_from_config is not None and isinstance(levels_from_config, list) and levels_from_config:
            print(f"使用配置中的难度级别: {levels_from_config}")
            return levels_from_config
        
        # 如果配置中没有，尝试使用类属性默认值
        if FixedTaskGenerator.DEFAULT_DIFFICULTY_LEVELS:
            print(f"使用类属性中的难度级别: {FixedTaskGenerator.DEFAULT_DIFFICULTY_LEVELS}")
            return FixedTaskGenerator.DEFAULT_DIFFICULTY_LEVELS
        
        # 如果都没有，则返回空列表
        print("警告: 未能获取到任何预定义难度级别。")
        return []
