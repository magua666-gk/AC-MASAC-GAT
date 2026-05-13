import numpy as np
import random
from typing import List, Dict, Any, Optional, Callable, Tuple
from .task import Task
from .utils.config import CurriculumConfig

class TaskSequencer:
    """任务排序器基类
    
    负责确定任务的学习顺序
    """
    
    def __init__(self, config: Optional[CurriculumConfig] = None):
        """初始化任务排序器
        
        Args:
            config: 配置对象，如果为None则创建默认配置
        """
        self.config = config or CurriculumConfig()
        self.learning_progress_window = self.config.get("task_sequencer.learning_progress_window", 10)
        self.performance_threshold = self.config.get("task_sequencer.performance_threshold", 0.9)
        
        # 添加评估窗口大小配置，确保与CurriculumManager使用一致的窗口大小
        self.evaluation_window = self.config.get("curriculum_manager.evaluation_window", 20)
        
    def select_next_task(self, tasks: List[Task], current_task: Optional[Task] = None) -> Optional[Task]:
        """从任务列表中选择下一个要学习的任务
        
        Args:
            tasks: 候选任务列表
            current_task: 当前任务(可选)，用于确定下一个任务
            
        Returns:
            选择的任务，如果没有合适的任务则返回None
        """
        raise NotImplementedError("子类必须实现此方法")
    
    def sort_tasks(self, tasks: List[Task]) -> List[Task]:
        """对任务列表进行排序
        
        Args:
            tasks: 待排序的任务列表
            
        Returns:
            排序后的任务列表
        """
        raise NotImplementedError("子类必须实现此方法")
    
    def is_curriculum_completed(self, tasks: List[Task]) -> bool:
        """判断课程是否已完成
        
        默认实现：当所有任务都被解决时，课程完成
        
        Args:
            tasks: 任务列表
            
        Returns:
            课程是否完成
        """
        if not tasks:
            return False
            
        return all(task.is_solved(self.performance_threshold) for task in tasks)


class LinearTaskSequencer(TaskSequencer):
    """线性任务排序器
    
    按照任务难度从易到难排序
    """
    
    def select_next_task(self, tasks: List[Task], current_task: Optional[Task] = None) -> Optional[Task]:
        """选择下一个要学习的任务
        
        选择未解决的任务中难度最接近但略高于当前任务的任务
        增强与FixedTaskGenerator的兼容性
        
        Args:
            tasks: 候选任务列表
            current_task: 当前任务，用于确定下一个任务
            
        Returns:
            选择的任务，如果没有合适的任务则返回None
        """
        if not tasks:
            return None
        
        # 过滤出未解决的任务，使用统一的评估窗口大小
        unsolved_tasks = [task for task in tasks if not task.is_solved(
            self.performance_threshold, 
            window=self.evaluation_window
        )]
        
        if not unsolved_tasks:
            return None
        
        # 如果有当前任务，选择难度略高于当前任务的任务
        if current_task is not None:
            current_difficulty = current_task.difficulty or 0.0
            
            # 寻找难度适当高于当前任务的未解决任务
            suitable_tasks = [task for task in unsolved_tasks 
                          if (task.difficulty or 0.0) > current_difficulty and 
                             (task.difficulty or 0.0) <= current_difficulty + 0.3]  # 从0.2增加到0.3，以适应固定任务梯度
            
            # 如果找到合适的任务，按难度排序并返回最简单的
            if suitable_tasks:
                selected_task = sorted(suitable_tasks, key=lambda t: t.difficulty or 0.0)[0]
                print(f"LinearTaskSequencer: 选择难度为 {selected_task.difficulty:.2f} 的下一个任务 (当前难度: {current_difficulty:.2f})")
                return selected_task
            else:
                print(f"LinearTaskSequencer: 未找到难度适合的下一个任务 (当前难度: {current_difficulty:.2f})")
        
        # 如果没有当前任务或没有找到合适的后续任务，按难度排序并返回最简单的未解决任务
        simplest_task = sorted(unsolved_tasks, key=lambda task: task.difficulty or 0.0)[0]
        print(f"LinearTaskSequencer: 返回难度最低的未解决任务，难度: {simplest_task.difficulty:.2f}")
        return simplest_task
    
    def sort_tasks(self, tasks: List[Task]) -> List[Task]:
        """按难度从易到难排序任务
        
        Args:
            tasks: 待排序的任务列表
            
        Returns:
            排序后的任务列表
        """
        return sorted(tasks, key=lambda task: task.difficulty or 0.0)


class LearningProgressTaskSequencer(TaskSequencer):
    """学习进度任务排序器
    
    基于智能体在任务上的学习进度选择任务
    """
    
    def select_next_task(self, tasks: List[Task], current_task: Optional[Task] = None) -> Optional[Task]:
        """选择下一个要学习的任务
        
        选择学习进度最高的未解决任务
        
        Args:
            tasks: 候选任务列表
            current_task: 当前任务(可选)，用于确定下一个任务
            
        Returns:
            选择的任务，如果没有合适的任务则返回None
        """
        if not tasks:
            return None
            
        # 过滤出未解决的任务，使用统一的评估窗口大小
        unsolved_tasks = [task for task in tasks if not task.is_solved(
            self.performance_threshold, 
            window=self.evaluation_window
        )]
        
        if not unsolved_tasks:
            return None
            
        # 计算每个任务的学习进度，使用统一的评估窗口
        task_progress = [(task, task.calculate_learning_progress(window=self.evaluation_window)) for task in unsolved_tasks]
        
        # 按学习进度排序（从高到低）
        sorted_tasks = sorted(task_progress, key=lambda x: x[1], reverse=True)
        
        return sorted_tasks[0][0]
    
    def sort_tasks(self, tasks: List[Task]) -> List[Task]:
        """按学习进度排序任务
        
        Args:
            tasks: 待排序的任务列表
            
        Returns:
            排序后的任务列表
        """
        # 计算每个任务的学习进度，使用统一的评估窗口
        task_progress = [(task, task.calculate_learning_progress(window=self.evaluation_window)) for task in tasks]
        
        # 按学习进度排序（从高到低）
        sorted_tasks = sorted(task_progress, key=lambda x: x[1], reverse=True)
        
        return [task for task, _ in sorted_tasks]


class AdaptiveTaskSequencer(TaskSequencer):
    """自适应任务排序器
    
    综合考虑任务难度和学习进度，动态调整任务顺序
    """
    
    def __init__(self, config: Optional[CurriculumConfig] = None):
        """初始化自适应任务排序器
        
        Args:
            config: 配置对象
        """
        super().__init__(config)
        # 学习进度权重，越大表示越重视学习进度
        self.progress_weight = 0.7
        # 难度权重，越大表示越倾向于从易到难
        self.difficulty_weight = 0.3
        
    def select_next_task(self, tasks: List[Task], current_task: Optional[Task] = None) -> Optional[Task]:
        """选择下一个要学习的任务
        
        基于难度和学习进度的加权得分选择最佳任务
        
        Args:
            tasks: 候选任务列表
            current_task: 当前任务，用于确定下一个任务
            
        Returns:
            选择的任务，如果没有合适的任务则返回None
        """
        if not tasks:
            return None
            
        # 过滤出未解决的任务，使用统一的评估窗口大小
        unsolved_tasks = [task for task in tasks if not task.is_solved(
            self.performance_threshold, 
            window=self.evaluation_window
        )]
        
        if not unsolved_tasks:
            return None
            
        if len(unsolved_tasks) == 1:
            return unsolved_tasks[0]
            
        current_difficulty = current_task.difficulty if current_task else 0.0
            
        # 计算每个任务的综合得分
        task_scores = []
        for task in unsolved_tasks:
            # 难度得分：优先选择略高于当前任务难度的任务
            difficulty = task.difficulty or 0.0
            if difficulty <= current_difficulty:
                difficulty_score = 0.0  # 难度低于当前任务的得分为0
            else:
                # 难度比当前任务高，但不要高太多
                difficulty_delta = difficulty - current_difficulty
                # 难度差越小越好，但不要为0
                difficulty_score = max(0.0, 1.0 - difficulty_delta)
                
            # 学习进度得分：优先选择学习进度高的任务
            learning_progress = task.calculate_learning_progress(window=self.evaluation_window)
            progress_score = max(0.0, learning_progress)  # 确保非负
            
            # 综合得分：加权平均
            score = self.difficulty_weight * difficulty_score + self.progress_weight * progress_score
            task_scores.append((task, score))
            
        # 按综合得分排序（从高到低）
        sorted_tasks = sorted(task_scores, key=lambda x: x[1], reverse=True)
        
        # 返回得分最高的任务
        return sorted_tasks[0][0]
    
    def sort_tasks(self, tasks: List[Task]) -> List[Task]:
        """按综合得分排序任务
        
        Args:
            tasks: 待排序的任务列表
            
        Returns:
            排序后的任务列表
        """
        # 计算每个任务的综合得分
        task_scores = []
        
        for task in tasks:
            # 学习进度得分
            progress = task.calculate_learning_progress(self.learning_progress_window)
            progress_score = progress if progress > 0 else 0
            
            # 难度得分
            difficulty = task.difficulty or 0.5
            difficulty_score = 1.0 - difficulty
            
            # 综合得分
            combined_score = (self.progress_weight * progress_score + 
                             self.difficulty_weight * difficulty_score)
            
            task_scores.append((task, combined_score))
        
        # 按综合得分排序（从高到低）
        sorted_tasks = sorted(task_scores, key=lambda x: x[1], reverse=True)
        
        return [task for task, _ in sorted_tasks] 