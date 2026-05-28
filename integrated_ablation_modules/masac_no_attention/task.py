import numpy as np
import time
import copy
import os
from typing import Dict, List, Any, Optional, Union

class Task:
    """表示课程学习中的一个任务
    
    每个任务包含环境参数配置、难度估计和性能历史记录
    
    Attributes:
        id: 任务唯一标识符
        env_params: 环境参数字典，用于创建环境实例
        difficulty: 任务难度系数(0-1)，1表示最难
        performance_history: 历史性能记录列表
    """
    def __init__(self, task_id: str, env_params: Dict[str, Any], difficulty: Optional[float] = None):
        """初始化一个任务
        
        Args:
            task_id: 任务ID
            env_params: 环境参数字典
            difficulty: 0-1之间的难度系数，默认为None
        """
        self.id = task_id
        self.env_params = env_params
        self.difficulty = difficulty
        self.performance_history = []

        # 从 env_params 中提取核心计数参数并设置为 Task 对象的直接属性
        # 这使得在其他模块（如 CurriculumManager）中可以方便地直接访问这些计数
        # 例如： next_task.hero_count, next_task.enemy_count
        
        # "leader_count" 对应主机 (hero) 数量
        self.hero_count = self.env_params.get("leader_count", 1) # 默认1个主机
        
        # "follower_count" 对应从机 (enemy) 数量
        # 注意：在 SPECIFIC_TASKS_CONFIG 中我们使用的是 "enemy_count"
        # 但在 Task 的 env_params 内部，FixedTaskGenerator 存储时使用的是 "follower_count"
        # Task 类应该从其自身的 self.env_params["follower_count"] 读取
        self.enemy_count = self.env_params.get("follower_count", 1) # 默认1个从机
        
        # "obstacle_count" 对应障碍物数量
        self.obstacle_count = self.env_params.get("obstacle_count", 0) # 默认0个障碍物
        
        # 打印调试信息，确认这些属性已设置 (可选，但建议在开发阶段保留)
        print(f"[Task Init DEBUG] Task \'{self.id}\': hero_count={self.hero_count}, enemy_count={self.enemy_count}, obstacle_count={self.obstacle_count} (from env_params[\'leader_count\'], env_params[\'follower_count\'], env_params[\'obstacle_count\'])")
    
    def create_env(self):
        """根据参数创建具体环境实例
        
        Returns:
            RlGame环境实例
        """
        from rl_env.path_env import RlGame
        from main_SAC import Switch
        
        # 从环境参数中提取RlGame构造函数所需的参数
        leader_count = self.env_params.get("leader_count", 1)  # 主机数量
        follower_count = self.env_params.get("follower_count", 1)  # 从机数量
        obstacle_num = self.env_params.get("obstacle_count", 1)  # 障碍物数量
        render = self.env_params.get("render", False)  # 是否渲染
        
        print(f"创建环境 - 主机: {leader_count}, 从机: {follower_count}, 障碍物: {obstacle_num}")
        
        # 创建预定义位置字典
        predefined_positions = {}
        if "goal_init_pos" in self.env_params:
            predefined_positions["goals"] = self.env_params["goal_init_pos"]
        
        # 创建RlGame环境，传入预定义位置
        env = RlGame(leader_count=leader_count, follower_count=follower_count, obstacle_num=obstacle_num, render=render, predefined_positions=predefined_positions)
        
        # 根据训练/测试模式设置dt
        if 'Switch' in globals() and globals()['Switch'] == 0:
            # 测试模式，使用dt=1.0
            env.set_time_step(1.0)
            print("测试模式: 时间步长dt设置为1.0")
        else:
            # 训练模式，使用dt=1.0
            env.set_time_step(1.0)
            print("训练模式: 时间步长dt设置为1.0")
        
        # 直接使用重构后的reconfigure方法来设置环境
        env.reconfigure(
            leader_count=leader_count,
            follower_count=follower_count,
            obstacle_count=obstacle_num
        )
        
        # 验证实际创建的智能体数量
        initial_state = env.reset()
        actual_agents = len(initial_state)
        if actual_agents != leader_count + follower_count:
            print(f"警告: 环境创建的智能体数量({actual_agents})与请求的数量({leader_count+follower_count})不一致")
            
        # 手动设置固定位置（如果提供）
        if hasattr(env, 'entity_manager'):
            # 设置主机位置
            if "leader_init_pos" in self.env_params:
                for i, leader in enumerate(env.entity_manager.leaders):
                    if i < len(self.env_params["leader_init_pos"]):
                        pos = self.env_params["leader_init_pos"][i]  # 获取第i个位置，这是一个(x,y)元组
                        leader.set_position(*pos)  # 解包元组为x,y参数
            
            # 设置从机位置
            if "follower_init_pos" in self.env_params:
                for i, follower in enumerate(env.entity_manager.followers):
                    if i < len(self.env_params["follower_init_pos"]):
                        pos = self.env_params["follower_init_pos"][i]  # 获取第i个位置
                        follower.set_position(*pos)  # 解包元组为x,y参数
            
            # 设置目标位置
            if "goal_init_pos" in self.env_params and env.entity_manager.goals:
                for i, goal in enumerate(env.entity_manager.goals):
                    if i < len(self.env_params["goal_init_pos"]):
                        pos = self.env_params["goal_init_pos"][i]  # 获取第i个位置
                        goal.set_position(*pos)  # 解包元组为x,y参数
            
            # 设置障碍物位置
            if "obstacle_init_pos" in self.env_params:
                for i, obstacle in enumerate(env.entity_manager.obstacles):
                    if i < len(self.env_params["obstacle_init_pos"]):
                        pos = self.env_params["obstacle_init_pos"][i]  # 获取第i个位置
                        obstacle.set_position(*pos)  # 解包元组为x,y参数
            
            # 不再设置无人机速度，使用实体类默认值
            # 领导者(LeaderAgent)初始速度默认为15，速度范围10-20，加速度系数0.3
            # 跟随者(FollowerAgent)初始速度默认为15，速度范围10-20，加速度系数0.3
            # if "uav_speed" in self.env_params:
            #     speed = self.env_params["uav_speed"]
            #     for leader in env.entity_manager.leaders:
            #         leader.speed = speed
            #     
            #     # 从机与主机使用相同的初始速度
            #     for follower in env.entity_manager.followers:
            #         follower.speed = speed
            #         
            #     print(f"设置无人机速度: 主机={speed}, 从机={speed}")
        
        return env
    
    def add_performance(self, metrics: Dict[str, float]):
        """添加性能指标到历史记录
        
        Args:
            metrics: 性能指标字典，如{'reward': 100, 'success_rate': 0.8}
        """
        self.performance_history.append({
            "metrics": metrics,
            "timestamp": time.time()
        })
    
    def get_average_performance(self, window: int = 15) -> Optional[Dict[str, float]]:
        """获取最近window次的平均性能
        
        Args:
            window: 用于计算平均值的窗口大小
            
        Returns:
            平均性能指标字典，如果没有性能记录则返回None
        """
        if not self.performance_history:
            return None
        
        recent = self.performance_history[-window:] if len(self.performance_history) >= window else self.performance_history
        return {
            k: np.mean([r["metrics"][k] for r in recent if k in r["metrics"]]) 
            for k in recent[0]["metrics"].keys()
        }
    
    def is_solved(self, success_threshold: float = 0.9, window: int = 15, reward_stability_threshold: float = 0.7) -> bool:
        """判断任务是否已经被解决（成功率达到阈值且奖励稳定）
        
        Args:
            success_threshold: 成功率阈值
            window: 用于计算平均成功率和奖励稳定性的窗口大小
            reward_stability_threshold: 奖励稳定性阈值，值越高要求越稳定
            
        Returns:
            任务是否已解决
        """
        # 如果历史记录不足，返回 False
        if len(self.performance_history) < window:
            return False
            
        # 获取平均性能
        avg_perf = self.get_average_performance(window)
        if not avg_perf:
            return False
        
        # 检查成功率
        success_rate_passed = False
        if 'success_rate' in avg_perf:
            success_rate_passed = avg_perf['success_rate'] >= success_threshold
            
            # 检查最近连续成功次数
            recent_successes = [
                entry['metrics']['success_rate'] >= 0.9  # 单次任务成功率
                for entry in self.performance_history[-window:]
                if 'success_rate' in entry['metrics']
            ]
            
            # 增加成功比例检查
            success_count = sum(1 for success in recent_successes if success)
            success_ratio = success_count / len(recent_successes) if recent_successes else 0
            success_ratio_passed = success_ratio >= success_threshold
            
            # 要求至少有一半的连续成功
            consecutive_success_count = 0
            max_consecutive_successes = 0
            for success in recent_successes:
                if success:
                    consecutive_success_count += 1
                    max_consecutive_successes = max(max_consecutive_successes, consecutive_success_count)
                else:
                    consecutive_success_count = 0
            
            # 连续成功要求 - 对于固定任务集，减少连续成功的要求
            # 检查任务ID是否包含特定标记，以识别固定任务
            is_fixed_task = 'task_' not in self.id  # 固定任务的ID通常是描述性的，如"入门级任务_abc123"
            
            if is_fixed_task:
                # 对于固定任务，减少连续成功要求，但不要太宽松
                min_required_consecutive = window // 3  # 从window//4调整到window//3
                print(f"固定任务宽松评估: 要求连续成功{min_required_consecutive}次 (当前连续成功{max_consecutive_successes}次)")
            else:
                # 对于普通任务，维持原始连续成功要求
                min_required_consecutive = window // 2
            
            consecutive_success_passed = max_consecutive_successes >= min_required_consecutive
            
            # 同时要求连续成功和整体成功率
            success_rate_passed = success_rate_passed and consecutive_success_passed and success_ratio_passed
        
        # 如果没有成功率指标，则使用奖励
        reward_passed = False
        if 'reward' in avg_perf and not success_rate_passed:
            # 难度越高，所需的奖励阈值越低
            difficulty_factor = self.difficulty if self.difficulty is not None else 0.5
            adaptive_threshold = success_threshold * (1 - 0.3 * difficulty_factor)
            reward_passed = avg_perf['reward'] >= adaptive_threshold
        
        # 计算奖励稳定性
        rewards = []
        for entry in self.performance_history[-window:]:
            if 'reward' in entry['metrics']:
                rewards.append(entry['metrics']['reward'])
        
        reward_stability_passed = False
        if rewards:
            reward_mean = np.mean(rewards)
            reward_std = np.std(rewards)
            
            # 计算稳定性得分：1 表示完全稳定，0 表示极不稳定
            # 使用修正的稳定性计算方法
            if abs(reward_mean) < 1e-6:  # 避免除零
                reward_stability = 0.0 if reward_std > 0 else 1.0
            else:
                coefficient_of_variation = reward_std / abs(reward_mean)
                reward_stability = 1.0 - min(1.0, coefficient_of_variation)
            
            # 另一种稳定性指标：最近窗口内奖励是否有上升趋势
            reward_trend = 0.0
            if len(rewards) >= 5:
                # 使用最近5个奖励计算趋势
                recent_rewards = rewards[-5:]
                x = np.arange(len(recent_rewards))
                try:
                    A = np.vstack([x, np.ones(len(x))]).T
                    slope, _ = np.linalg.lstsq(A, recent_rewards, rcond=None)[0]
                    reward_trend = slope
                except:
                    reward_trend = 0.0
            
            # 奖励是否相对稳定并且非下降趋势
            # 对固定任务集，适当降低奖励稳定性要求
            is_fixed_task = 'task_' not in self.id
            
            if is_fixed_task:
                # 固定任务使用较低的稳定性要求
                actual_stability_threshold = max(0.5, reward_stability_threshold - 0.2)
                print(f"固定任务宽松评估: 稳定性要求{actual_stability_threshold:.2f} (当前{reward_stability:.2f})")
            else:
                actual_stability_threshold = reward_stability_threshold
                
            reward_stability_passed = (reward_stability >= actual_stability_threshold and reward_trend >= -0.5)
            
            # 打印调试信息，帮助分析
            print(f"任务 {self.id} 评估:")
            print(f"  - 成功率: {avg_perf.get('success_rate', 0):.2f} (目标: {success_threshold})")
            print(f"  - 最大连续成功次数: {max_consecutive_successes if 'success_rate' in avg_perf else 'N/A'}")
            print(f"  - 奖励均值: {reward_mean:.2f}, 标准差: {reward_std:.2f}")
            print(f"  - 奖励稳定性: {reward_stability:.2f} (目标: {actual_stability_threshold})")
            print(f"  - 奖励趋势: {reward_trend:.4f}")
        
        # 只有当成功率条件满足且奖励稳定时，才认为任务解决
        if (success_rate_passed or reward_passed) and reward_stability_passed:
            print(f"任务 {self.id} 已解决: 成功率/奖励达标 + 奖励稳定")
            return True
        
        return False
    
    def calculate_learning_progress(self, window: int = 10) -> float:
        """计算任务的学习进度（奖励斜率）
        
        Args:
            window: 用于计算斜率的窗口大小
            
        Returns:
            学习进度值，正值表示在进步，负值表示在退步
        """
        # 确保有足够的样本用于计算学习进度
        if len(self.performance_history) < window:
            print(f"性能历史记录不足({len(self.performance_history)}/{window})，无法计算学习进度")
            return 0.0
        
        # 计算最近窗口内的奖励斜率，确保使用传入的窗口大小
        recent = self.performance_history[-window:]
        if 'reward' not in recent[0]['metrics']:
            return 0.0
            
        rewards = [r['metrics']['reward'] for r in recent]
        x = np.arange(len(rewards))
        
        # 检查样本是否有足够的变化
        reward_std = np.std(rewards)
        reward_mean = np.mean(rewards)
        
        # 如果奖励几乎没有变化，则无法判断是否有进步
        if reward_std < 0.01 * abs(reward_mean) and len(rewards) < window * 2:
            print(f"奖励波动过小(std={reward_std:.2f}, mean={reward_mean:.2f})，需要更多样本确定学习进度")
            return 0.0
        
        # 使用线性回归计算斜率
        try:
            # 简单线性回归: y = ax + b
            A = np.vstack([x, np.ones(len(x))]).T
            slope, _ = np.linalg.lstsq(A, rewards, rcond=None)[0]
            
            # 计算斜率的相对值，相对于均值的百分比
            relative_slope = slope / (abs(reward_mean) + 1e-6)
            
            # 根据相对斜率判断学习进度
            if abs(relative_slope) < 0.01:  # 相对斜率非常小
                print(f"学习进度接近平稳 (相对斜率={relative_slope:.4f})")
                return 0.0
                
            return float(slope)
        except:
            return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """将任务转换为字典表示，用于序列化
        
        Returns:
            任务的字典表示
        """
        return {
            'id': self.id,
            'env_params': self.env_params,
            'difficulty': self.difficulty,
            # 不包含性能历史，因为可能很大
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Task':
        """从字典创建任务
        
        Args:
            data: 任务字典表示
            
        Returns:
            创建的Task对象
        """
        return cls(
            task_id=data['id'],
            env_params=data['env_params'],
            difficulty=data.get('difficulty')
        )
    
    def __hash__(self):
        """实现哈希方法以便在集合中使用"""
        return hash(self.id)
    
    def __eq__(self, other):
        """实现相等比较"""
        if not isinstance(other, Task):
            return False
        return self.id == other.id
    
    def __str__(self):
        """字符串表示"""
        return f"Task({self.id}, difficulty={self.difficulty})"
    
    def __repr__(self):
        """调试表示"""
        return self.__str__() 