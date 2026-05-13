import os
import json
import time
import numpy as np
import random
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from .task import Task
from .task_generator import TaskGenerator, DefaultTaskGenerator, FixedTaskGenerator
from .task_sequencer import TaskSequencer, LinearTaskSequencer
from .knowledge_transfer import KnowledgeTransfer, PolicyTransfer
from .utils.config import CurriculumConfig

class CurriculumManager:
    """课程学习管理器
    
    整合课程学习的各个组件，管理整个课程学习流程
    """
    
    def __init__(self,
                 config: Optional[CurriculumConfig] = None,
                 task_generator: Optional[TaskGenerator] = None,
                 task_sequencer: Optional[TaskSequencer] = None,
                 knowledge_transfer: Optional[KnowledgeTransfer] = None):
        """初始化课程管理器
        
        Args:
            config: 配置对象
            task_generator: 任务生成器
            task_sequencer: 任务排序器
            knowledge_transfer: 知识迁移器
        """
        self.config = config or CurriculumConfig()
        
        # 默认使用FixedTaskGenerator而不是DefaultTaskGenerator
        if task_generator is None:
            print("默认使用固定任务生成器FixedTaskGenerator")
            task_generator = FixedTaskGenerator(self.config)
            
        self.task_generator = task_generator
        self.task_sequencer = task_sequencer or LinearTaskSequencer(self.config)
        self.knowledge_transfer = knowledge_transfer or PolicyTransfer(self.config)
        
        # 课程参数
        self.num_initial_tasks = self.config.get("curriculum_manager.num_initial_tasks", 5)
        self.max_episodes_per_task = self.config.get("curriculum_manager.max_episodes_per_task", 200)
        self.max_curriculum_steps = self.config.get("curriculum_manager.max_curriculum_steps", 20)
        self.max_total_episodes = self.config.get("curriculum_manager.max_total_episodes", 1000)
        
        # 添加评估窗口大小配置，确保任务评估和生成使用一致的窗口大小
        self.evaluation_window = self.config.get("curriculum_manager.evaluation_window", 15)
        self.min_training_rounds = self.config.get("curriculum_manager.min_training_rounds", 50)
        self.reward_stability_threshold = self.config.get("curriculum_manager.reward_stability_threshold", 0.7)
        self.success_rate_threshold = self.config.get("curriculum_manager.success_rate_threshold", 0.9)
        
        # 添加停滞检测相关变量
        self.stagnation_counter = 0
        self.stagnation_threshold = self.config.get("curriculum_manager.stagnation_threshold", 3)
        self.progress_threshold = self.config.get("curriculum_manager.progress_threshold", 0.05)
        
        # 回退机制相关参数 - 强制禁用
        self.backtrack_enabled = False  # 强制禁用回退机制
        print("回退机制已禁用")
        self.consecutive_fails = 0
        self.task_history = []  # 记录已完成的任务，用于记录不用于回退
        
        # 多维度评估参数
        self.min_success_rate = self.config.get("curriculum_manager.min_success_rate", 0.2)
        self.max_variation_coef = self.config.get("curriculum_manager.max_variation_coef", 0.5)
        
        # 当前课程状态
        self.tasks = []
        self.current_task = None
        self.task_episodes = 0
        self.total_episodes = 0  # 添加总回合数计数器
        self.curriculum_step = 0
        self.history = []
        
    def initialize(self) -> Task:
        """初始化课程
        
        生成初始任务并选择第一个任务
        
        Returns:
            初始任务
        """
        # 获取初始任务难度范围配置，默认为(0.0, 0.3)
        initial_difficulty_range = self.config.get("curriculum_manager.initial_difficulty_range", (0.0, 0.3))
        
        # 生成初始任务集
        initial_tasks = self.task_generator.generate_tasks(
            count=self.num_initial_tasks, 
            difficulty_range=initial_difficulty_range  # 使用配置的初始难度范围
        )
        
        self.tasks = initial_tasks
        
        # 排序任务
        sorted_tasks = self.task_sequencer.sort_tasks(self.tasks)
        
        # 选择第一个任务
        self.current_task = sorted_tasks[0]
        self.task_episodes = 0
        self.curriculum_step = 0
        
        return self.current_task
    
    def update_task_performance(self, metrics: Dict[str, float]) -> bool:
        """更新当前任务的性能指标，并记录成功/失败
        
        Args:
            metrics: 性能指标字典，如{'reward': 100, 'success_rate': 0.8}
            
        Returns:
            bool: 当总回合数达到最大限制时返回True，否则返回False
        """
        if self.current_task:
            self.current_task.add_performance(metrics)
            self.task_episodes += 1
            self.total_episodes += 1  # 同时增加总回合数
            
            # 检查是否达到总回合数限制
            if self.total_episodes >= self.max_total_episodes:
                print(f"达到最大总回合数限制({self.max_total_episodes})，训练将结束")
                return True
            
            # 记录连续失败次数 - 用于回退机制
            if 'success_rate' in metrics and metrics['success_rate'] < 0.5:
                self.consecutive_fails += 1
                print(f"任务失败，连续失败次数: {self.consecutive_fails}")
            else:
                self.consecutive_fails = 0  # 重置连续失败计数
            
            # 记录历史
            self.history.append({
                "task_id": self.current_task.id,
                "difficulty": self.current_task.difficulty,
                "episode": self.task_episodes,
                "total_episodes": self.total_episodes,  # 添加总回合数到历史记录
                "curriculum_step": self.curriculum_step,
                "metrics": metrics,
                "timestamp": time.time()
            })
            
        return False  # 默认情况下不结束训练
    
    def should_switch_task(self) -> bool:
        """判断是否应该切换到新任务
        
        只在当前任务已解决或达到最大训练轮数时切换，移除所有学习停滞检测逻辑
        对于最后一个课程步骤，仅当达到指定回合数时才切换。
        
        Returns:
            是否应该切换任务
        """
        if not self.current_task:
            return True

        # 检查是否是最后一个课程步骤
        is_last_curriculum_step = (self.curriculum_step == self.max_curriculum_steps - 1)

        if is_last_curriculum_step:
            # 对于最后一个任务，只检查是否达到了设定的回合数
            if self.task_episodes >= self.max_episodes_per_task:
                print(f"最后一个课程步骤 ({self.curriculum_step + 1}/{self.max_curriculum_steps})，已达到指定训练回合数 ({self.task_episodes}/{self.max_episodes_per_task})，准备切换...")
                return True
            else:
                # print(f"最后一个课程步骤 ({self.curriculum_step + 1}/{self.max_curriculum_steps})，继续训练至指定回合数 ({self.task_episodes}/{self.max_episodes_per_task})...")
                return False
            
        # 对于非最后一个任务，执行原有逻辑
        # 如果没有达到最小训练轮数，不切换任务
        if self.task_episodes < self.min_training_rounds:
            return False
            
        # 增加训练初期保护机制，防止过早切换任务
        # 即使任务已解决，也至少保证一定轮数的训练
        if self.task_episodes < self.min_training_rounds * 1.5:
            # 确保初期训练不会因简单的随机运气好而过早切换
            print(f"任务仍处于训练前期({self.task_episodes}/{self.min_training_rounds * 1.5})，继续训练")
            return False
            
        # 任务已解决
        if self.current_task.is_solved(
            success_threshold=self.success_rate_threshold,
            window=self.evaluation_window,
            reward_stability_threshold=self.reward_stability_threshold
        ):
            print("任务已解决，准备切换...")
            return True
            
        # 达到最大训练轮数
        if self.task_episodes >= self.max_episodes_per_task:
            print(f"达到最大训练轮数({self.max_episodes_per_task})，准备切换...")
            return True
            
        return False
    
    def get_next_task(self, model: Any = None) -> Tuple[Optional[Task], Any]:
        """获取下一个任务
        
        根据当前任务的完成情况，确定是否需要切换到新任务。
        增加了最高难度任务的终止逻辑，避免重新生成初级任务。
        
        Args:
            model: 当前模型，用于知识迁移
            
        Returns:
            (next_task, transferred_model)元组
        """
        if not self.current_task:
            # 如果当前没有任务，从任务列表中选择第一个，或创建一个新任务
            if self.tasks:
                return self.tasks[0], model
            else:
                # 确保使用FixedTaskGenerator生成固定难度梯度任务
                return self._get_initial_task(), model
        
        # 检查当前任务是否需要切换
        if self.should_switch_task():
            print("当前任务需要切换...")
            
            # 如果任务已解决，添加到历史记录（仅用于记录，不用于回退）
            if self.current_task.is_solved():
                self.task_history.append(self.current_task)
                # 只保留最近5个任务历史
                if len(self.task_history) > 5:
                    self.task_history = self.task_history[-5:]
                print(f"任务已解决，添加到历史记录: {self.current_task.id}")
            
            # 使用固定任务生成器创建下一个难度级别的任务
            from curriculum.task_generator import FixedTaskGenerator
            
            # 获取当前任务的难度级别
            current_difficulty = self.current_task.difficulty
            
            # 创建固定任务生成器（如果还没有）
            if not hasattr(self, '_fixed_task_generator'):
                self._fixed_task_generator = FixedTaskGenerator(self.config)
                
            # 增加难度比较的容差 (此容差用于匹配 current_difficulty 到 difficulty_levels 中的索引)
            DIFFICULTY_TOLERANCE = 0.05
                
            # 找到当前任务的难度级别在预定义难度梯度中的索引
            try:
                difficulty_levels = self._fixed_task_generator._get_predefined_difficulty_levels()
                print(f"成功使用 _get_predefined_difficulty_levels 获取难度级别: {difficulty_levels}")
            except Exception as e:
                print(f"错误：调用 _get_predefined_difficulty_levels 方法失败: {str(e)}")
                difficulty_levels = []
                # 此处可以添加从其他来源获取 difficulty_levels 的回退逻辑，但根据当前上下文，主要依赖 _get_predefined_difficulty_levels
                if not difficulty_levels: # 再次检查，如果仍然为空
                    print("错误：无法从 FixedTaskGenerator 的任何来源获取预定义难度级别！")
                    return self.current_task, model

            if not difficulty_levels: # 再次检查，以防上面try-except后仍为空
                print("错误：无法从 FixedTaskGenerator 的任何来源获取预定义难度级别！")
                return self.current_task, model
            
            print(f"最终使用的难度级别列表: {difficulty_levels}")

            current_index = -1
            for i, level in enumerate(difficulty_levels):
                 if abs(current_difficulty - level) < DIFFICULTY_TOLERANCE:
                      current_index = i
                      print(f"当前任务难度{current_difficulty}匹配到预定义难度级别{level}，索引{i}")
                      break
            
            if current_index == -1:
                print(f"警告：当前任务难度{current_difficulty}在预定义难度梯度中找不到匹配项")
                closest_index = 0
                min_diff = float('inf')
                for i, level in enumerate(difficulty_levels):
                    diff = abs(current_difficulty - level)
                    if diff < min_diff:
                        min_diff = diff
                        closest_index = i
                current_index = closest_index
                print(f"使用最接近的难度级别{difficulty_levels[current_index]}，索引{current_index}")
            
            if current_index >= len(difficulty_levels) - 1:
                print(f"已达到最高难度级别({difficulty_levels[current_index]})，课程学习完成!")
                # 检查最后一个任务是否已解决
                if self.current_task and self.current_task.is_solved():
                    print("最后一个任务已解决，整个课程学习流程完成！")
                    return None, model  # 返回None触发终止条件
                else:
                    print(f"训练将继续使用最后一个任务，直到解决为止")
                    return self.current_task, model
            
            # 准备下一个任务
            next_task = None # 初始化 next_task
            next_difficulty = difficulty_levels[current_index + 1]
            print(f"当前任务(难度:{current_difficulty})已完成，尝试切换到下一目标难度级别({next_difficulty})")

            # 严格尝试从 self.predefined_task_configs 字典中直接获取下一个任务
            if hasattr(self._fixed_task_generator, 'predefined_task_configs') and self._fixed_task_generator.predefined_task_configs:
                found_key = None
                # 使用非常小的容差进行精确匹配 next_difficulty
                for key_difficulty in self._fixed_task_generator.predefined_task_configs.keys():
                    if abs(key_difficulty - next_difficulty) < 0.001: 
                        found_key = key_difficulty
                        break
                
                if found_key is not None:
                    task_candidate = self._fixed_task_generator.predefined_task_configs.get(found_key)
                    if task_candidate:
                        next_task = task_candidate
                        if next_task not in self.tasks:
                            self.tasks.append(next_task)
                        print(f"从 predefined_task_configs 成功获取预定义任务 (难度: {found_key:.2f}) ID: {next_task.id}")
                    else:
                        print(f"错误: 在 predefined_task_configs 中找到了键 {found_key:.2f}，但对应的值为 None。无法切换到此任务。")
                        # next_task 保持为 None
                else:
                    print(f"错误: 未能在 predefined_task_configs 中找到与下一目标难度 {next_difficulty:.2f} 精确匹配的预定义任务。")
                    # next_task 保持为 None
            else:
                print("错误: _fixed_task_generator 没有 predefined_task_configs 属性或该属性为空。无法查找预定义任务。")
                # next_task 保持为 None

            # 如果未能获取到下一个预定义任务
            if next_task is None:
                print(f"关键错误: 无法为难度 {next_difficulty:.2f} 找到或确定下一个预定义任务。课程学习无法继续按计划进行。")
                print(f"将保持当前任务: {self.current_task.id} (难度: {self.current_task.difficulty:.2f})")
                return self.current_task, model 
            
            # 如果成功获取到 next_task，则继续后续的知识迁移和任务更新逻辑
            # (知识迁移和状态更新逻辑)
            if model is not None and self.current_task is not None:
                old_hero_count = self.current_task.hero_count
                old_enemy_count = self.current_task.enemy_count
                new_hero_count = next_task.hero_count
                new_enemy_count = next_task.enemy_count
                
                # ---- BEGIN DEBUG PRINT ----
                print(f"[DEBUG CM] Next Task Details Before Agent Count Log:")
                print(f"[DEBUG CM]   next_task.id: {next_task.id}")
                print(f"[DEBUG CM]   next_task.difficulty: {next_task.difficulty}")
                print(f"[DEBUG CM]   next_task.hero_count (property): {next_task.hero_count}")
                print(f"[DEBUG CM]   next_task.enemy_count (property): {next_task.enemy_count}")
                print(f"[DEBUG CM]   next_task.env_params: {next_task.env_params}")
                # ---- END DEBUG PRINT ----
                
                current_task_agents = old_hero_count + old_enemy_count
                next_task_agents = new_hero_count + new_enemy_count
                
                print(f"任务切换 - 智能体数量: {current_task_agents} -> {next_task_agents}")
                
                transferred_result = self.knowledge_transfer.transfer(self.current_task, next_task, model)
                
                # PolicyTransfer._do_transfer 现在始终返回参数字典，而不是模型实例
                if isinstance(transferred_result, dict):
                    # 这是预期的行为 - PolicyTransfer返回参数字典
                    print("知识迁移返回了参数字典（正确的行为）")
                    if 'agent_counts' in transferred_result:
                        agent_counts = transferred_result.get('agent_counts', {})
                        print(f"智能体数量变化: {agent_counts.get('source', 'N/A')} -> {agent_counts.get('target', 'N/A')}")
                    # 返回参数字典，让调用者(main_SAC_curriculum.py)处理参数应用
                    model = transferred_result
                else:
                    # 兼容旧版本行为 - 如果返回了模型实例
                    print("警告：知识迁移返回了模型实例而不是参数字典（旧版本行为）")
                    if hasattr(transferred_result, 'adapt_to_agent_count'):
                        transferred_result.adapt_to_agent_count(new_hero_count + new_enemy_count)
                    elif hasattr(transferred_result, 'reset_noise'):
                        transferred_result.reset_noise(new_hero_count + new_enemy_count)
                    model = transferred_result
                    
                print(f"任务切换完成: {self.current_task.id} -> {next_task.id}")
            
            self.current_task = next_task
            self.task_episodes = 0
            self.curriculum_step += 1
            self.consecutive_fails = 0 # 重置连续失败计数
            
            return next_task, model
        
        # 如果当前任务未完成，继续使用当前任务
        return self.current_task, model
    
    def _get_initial_task(self) -> Task:
        """获取初始任务
        
        创建一个新的初始固定任务
        
        Returns:
            返回初始任务
        """
        # 确保使用FixedTaskGenerator
        from curriculum.task_generator import FixedTaskGenerator
        
        # 创建固定任务生成器（如果还没有）
        if not hasattr(self, '_fixed_task_generator'):
            self._fixed_task_generator = FixedTaskGenerator(self.config)
            
        # 获取最简单的任务（第一个预定义任务）
        if self._fixed_task_generator.predefined_tasks:
            initial_task_config = self._fixed_task_generator.predefined_tasks[0]
            initial_task = self._fixed_task_generator._create_task_from_config(initial_task_config)
            self.tasks.append(initial_task)
            return initial_task
        
        # 如果没有预定义任务，使用generate_next_task创建一个新任务
        initial_task = self._fixed_task_generator.generate_next_task([], {}, window=self.evaluation_window)
        self.tasks.append(initial_task)
        return initial_task
    
    def get_current_task(self) -> Optional[Task]:
        """获取当前任务
        
        Returns:
            当前任务，如果没有则返回None
        """
        return self.current_task
    
    def get_all_tasks(self) -> List[Task]:
        """获取所有任务
        
        Returns:
            任务列表
        """
        return self.tasks
    
    def save_curriculum_state(self, path: str) -> None:
        """保存课程状态到文件
        
        Args:
            path: 保存路径
        """
        state = {
            "tasks": [task.to_dict() for task in self.tasks],
            "current_task_id": self.current_task.id if self.current_task else None,
            "task_episodes": self.task_episodes,
            "total_episodes": self.total_episodes,  # 添加总回合数
            "curriculum_step": self.curriculum_step,
            "history": self.history
        }
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_curriculum_state(self, path: str) -> None:
        """从文件加载课程状态
        
        Args:
            path: 加载路径
        """
        with open(path, 'r') as f:
            state = json.load(f)
        
        # 恢复任务
        self.tasks = [Task.from_dict(task_dict) for task_dict in state["tasks"]]
        
        # 恢复当前任务
        if state["current_task_id"]:
            for task in self.tasks:
                if task.id == state["current_task_id"]:
                    self.current_task = task
                    break
        
        # 恢复其他状态
        self.task_episodes = state["task_episodes"]
        self.total_episodes = state.get("total_episodes", 0)  # 使用 get 方法兼容旧版本保存的状态
        self.curriculum_step = state["curriculum_step"]
        self.history = state["history"]
    
    def get_curriculum_progress(self) -> Dict[str, Any]:
        """获取课程学习进度信息
        
        Returns:
            进度信息字典
        """
        solved_tasks = sum(1 for task in self.tasks if task.is_solved())
        total_tasks = len(self.tasks)
        
        return {
            "curriculum_step": self.curriculum_step,
            "max_curriculum_steps": self.max_curriculum_steps,
            "solved_tasks": solved_tasks,
            "total_tasks": total_tasks,
            "progress_percentage": (solved_tasks / total_tasks * 100) if total_tasks > 0 else 0,
            "current_task": str(self.current_task) if self.current_task else None,
            "task_episodes": self.task_episodes,
            "total_episodes": self.total_episodes  # 添加总回合数
        } 
    
    def _get_current_performance(self) -> Dict[str, float]:
        """获取当前任务的平均性能数据
        
        Returns:
            当前任务的平均性能指标字典
        """
        current_performance = {}
        if self.current_task and self.current_task.performance_history:
            window_size = min(self.evaluation_window, len(self.current_task.performance_history))
            recent_history = self.current_task.performance_history[-window_size:]
            if recent_history:
                metrics_keys = recent_history[0]['metrics'].keys()
                for key in metrics_keys:
                    current_performance[key] = np.mean([h['metrics'][key] for h in recent_history if key in h['metrics']])
        return current_performance 
    
    def is_learning_stagnant(self) -> bool:
        """检测是否出现学习停滞
        
        已禁用学习停滞检测，始终返回False
        
        Returns:
            False，不再检测学习停滞
        """
        # 学习停滞检测功能已禁用
        return False
    
    def backtrack_to_previous_task(self, model: Any = None) -> Tuple[Optional[Task], Any]:
        """回退到上一个已完成的任务
        
        该功能已被禁用，不会执行实际回退
        
        Args:
            model: 当前模型，用于知识迁移
            
        Returns:
            (current_task, model)元组，不执行回退
        """
        # 直接打印警告信息，返回当前任务和模型
        print("警告：回退机制已禁用，忽略回退请求，继续当前任务")
        print(f"当前任务: {self.current_task.id} (难度: {self.current_task.difficulty:.2f})")
        
        # 返回当前任务和模型，不执行回退
        return self.current_task, model 