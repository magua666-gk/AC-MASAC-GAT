import numpy as np
import torch
import os
import copy
from typing import Dict, Any, Optional, Union, List, Tuple
from .task import Task
from .utils.config import CurriculumConfig

class KnowledgeTransfer:
    """Knowledge transfer base class
    
    Defines basic interface and methods for knowledge transfer
    """
    
    def __init__(self, config=None):
        """Initialize knowledge transfer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.transfer_threshold = self.config.get("knowledge_transfer.similarity_threshold", 0.5)
        self.verbose = self.config.get("knowledge_transfer.verbose", True)
        
    def transfer(self, source_task, target_task, model):
        """Transfer knowledge from source task to target task
        
        Args:
            source_task: Source task
            target_task: Target task
            model: Model to transfer
            
        Returns:
            Transferred model
        """
        if self.verbose:
            print("\n" + "="*60)
            print(f"Knowledge transfer debug info:")
            print(f"Model type: {type(model).__name__}")
            print(f"Supported methods:")
            for method_name in dir(model):
                if not method_name.startswith('_') and callable(getattr(model, method_name)):
                    print(f"  - {method_name}")
            print(f"Supports transfer_policy_parameters: {hasattr(model, 'transfer_policy_parameters')}")
            print("="*60 + "\n")
            
        similarity = self._calculate_task_similarity(source_task, target_task)
        
        if self.verbose:
            print(f"\n{'='*50}")
            print(f"Transfer analysis - source '{source_task.id}' to target '{target_task.id}'")
            print(f"{'='*50}")
            print(f"Task similarity: {similarity:.4f}")
            
            print("\nTask parameters comparison:")
            print(f"{'Parameter':20} | {'Source':15} | {'Target':15} | {'Diff':10}")
            print("-" * 65)
            
            all_keys = set(source_task.env_params.keys()) | set(target_task.env_params.keys())
            
            for key in sorted(all_keys):
                source_value = source_task.env_params.get(key, "N/A")
                target_value = target_task.env_params.get(key, "N/A")
                
                diff = "N/A"
                if isinstance(source_value, (int, float)) and isinstance(target_value, (int, float)):
                    diff = f"{target_value - source_value:+.2f}"
                
                print(f"{key:20} | {str(source_value):15} | {str(target_value):15} | {diff:10}")
        
        if similarity >= self.transfer_threshold:
            if self.verbose:
                print(f"\nā Task similarity ({similarity:.4f}) above threshold ({self.transfer_threshold}), performing knowledge transfer")
                
            if self.verbose and hasattr(model, 'get_parameters'):
                try:
                    params = model.get_parameters()
                    print(f"Model parameters before transfer:")
                    for key, value in params.items():
                        if isinstance(value, dict):
                            print(f"  - {key}: {len(value)} items")
                        else:
                            print(f"  - {key}: {type(value)}")
                except Exception as e:
                    print(f"Error extracting model parameters: {e}")
                    import traceback
                    traceback.print_exc()
            
            result_model = self._do_transfer(source_task, target_task, model, similarity)
            
            # 迁移后检查模型状态
            if self.verbose:
                if result_model is not model:
                    print(f"警告: 迁移返回了新的模型实例而不是更新原模型")
                    
                if hasattr(result_model, 'get_parameters'):
                    try:
                        params = result_model.get_parameters()
                        print(f"迁移后模型参数统计:")
                        for key, value in params.items():
                            if isinstance(value, dict):
                                print(f"  - {key}: {len(value)} 项")
                            else:
                                print(f"  - {key}: {type(value)}")
                    except Exception as e:
                        print(f"提取迁移后模型参数时出错: {e}")
            
            return result_model
        else:
            if self.verbose:
                print(f"\n✗ 任务相似度 ({similarity:.4f}) 低于阈值 ({self.transfer_threshold})，不执行知识迁移")
            return model
    
    def _calculate_task_similarity(self, source_task, target_task):
        """Calculate similarity between two tasks
        
        Task similarity is based on environment parameters. Range: 0-1, higher means more similar.
        Enhanced robustness to ensure small differences don't result in too low similarity.
        
        Args:
            source_task: Source task
            target_task: Target task
            
        Returns:
            Similarity score (0-1)
        """
        source_params = source_task.env_params
        target_params = target_task.env_params
        
        difficulty_diff = abs(source_task.difficulty - target_task.difficulty)
        normalized_difficulty_diff = min(difficulty_diff / 1.0, 0.5)  
        
        source_leader_count = source_params.get("leader_count", 1)
        target_leader_count = target_params.get("leader_count", 1)
        source_follower_count = source_params.get("follower_count", 0)
        target_follower_count = target_params.get("follower_count", 0)
        
        leader_count_diff = abs(source_leader_count - target_leader_count)
        follower_count_diff = abs(source_follower_count - target_follower_count)
        agent_count_similarity = 1.0 - min((leader_count_diff + follower_count_diff) / 8.0, 0.5)
        
        source_obstacle_count = source_params.get("obstacle_count", 0)
        target_obstacle_count = target_params.get("obstacle_count", 0)
        obstacle_diff = abs(source_obstacle_count - target_obstacle_count)
        obstacle_similarity = 1.0 - min(obstacle_diff / 15.0, 0.5)
        
        # 地图大小相似度
        source_map_size = source_params.get("map_size", (10, 10))
        target_map_size = target_params.get("map_size", (10, 10))
        
        # 获取宽高
        if isinstance(source_map_size, (list, tuple)) and len(source_map_size) >= 2:
            source_width, source_height = source_map_size[0], source_map_size[1]
        else:
            source_width = source_height = source_map_size
            
        if isinstance(target_map_size, (list, tuple)) and len(target_map_size) >= 2:
            target_width, target_height = target_map_size[0], target_map_size[1]
        else:
            target_width = target_height = target_map_size
            
        width_diff = abs(source_width - target_width)
        height_diff = abs(source_height - target_height)
        map_size_similarity = 1.0 - min((width_diff + height_diff) / 20.0, 0.5)
        
        # 算法参数相似度
        source_algo_params = source_params.get("algo_params", {})
        target_algo_params = target_params.get("algo_params", {})
        
        algo_similarity = 1.0
        if source_algo_params and target_algo_params:
            # 比较关键算法参数
            key_params = ["learning_rate", "gamma", "clip_norm"]
            param_diffs = []
            
            for param in key_params:
                source_value = source_algo_params.get(param)
                target_value = target_algo_params.get(param)
                
                if source_value is not None and target_value is not None:
                    if isinstance(source_value, (int, float)) and isinstance(target_value, (int, float)):
                        # 计算相对差异
                        max_value = max(abs(source_value), abs(target_value))
                        if max_value > 0:
                            diff = abs(source_value - target_value) / max_value
                            param_diffs.append(min(diff, 1.0))
            
            # 计算平均差异
            if param_diffs:
                algo_similarity = 1.0 - sum(param_diffs) / len(param_diffs)
        
        # 奖励设置相似度
        reward_similarity = 1.0
        source_reward_conf = source_params.get("reward_params", {})
        target_reward_conf = target_params.get("reward_params", {})
        
        if source_reward_conf and target_reward_conf:
            reward_diffs = []
            
            # 比较奖励权重
            for key in set(source_reward_conf.keys()) | set(target_reward_conf.keys()):
                source_weight = source_reward_conf.get(key, 0)
                target_weight = target_reward_conf.get(key, 0)
                
                if source_weight != 0 or target_weight != 0:
                    max_weight = max(abs(source_weight), abs(target_weight))
                    if max_weight > 0:
                        diff = abs(source_weight - target_weight) / max_weight
                        reward_diffs.append(min(diff, 1.0))
            
            if reward_diffs:
                reward_similarity = 1.0 - sum(reward_diffs) / len(reward_diffs)
        
        # 调整权重，提高知识迁移的可能性
        weights = {
            "difficulty": 0.15,  # 降低难度的权重
            "agent_count": 0.2,  # 降低智能体数量的权重
            "obstacle": 0.1,     # 降低障碍物的权重
            "map_size": 0.1,     # 降低地图大小的权重
            "algo": 0.2,         # 增加算法参数的权重
            "reward": 0.25       # 增加奖励设置的权重
        }
        
        # 计算加权相似度
        similarity = (
            weights["difficulty"] * (1.0 - normalized_difficulty_diff) +
            weights["agent_count"] * agent_count_similarity +
            weights["obstacle"] * obstacle_similarity +
            weights["map_size"] * map_size_similarity +
            weights["algo"] * algo_similarity +
            weights["reward"] * reward_similarity
        )
        
        # 设置最小相似度值，确保至少有一些知识迁移
        MIN_SIMILARITY = 0.3
        similarity = max(similarity, MIN_SIMILARITY)
        
        # 输出详细的相似度计算
        if self.verbose:
            print("\n各项相似度权重和得分:")
            print(f"难度相似度: {1.0 - normalized_difficulty_diff:.4f} (权重: {weights['difficulty']})")
            print(f"智能体数量相似度: {agent_count_similarity:.4f} (权重: {weights['agent_count']})")
            print(f"障碍物相似度: {obstacle_similarity:.4f} (权重: {weights['obstacle']})")
            print(f"地图大小相似度: {map_size_similarity:.4f} (权重: {weights['map_size']})")
            print(f"算法参数相似度: {algo_similarity:.4f} (权重: {weights['algo']})")
            print(f"奖励设置相似度: {reward_similarity:.4f} (权重: {weights['reward']})")
            print(f"最终相似度(加最小阈值): {similarity:.4f}")
        
        return similarity
    
    def _do_transfer(self, source_task, target_task, model, similarity):
        """执行实际的知识迁移过程
        
        在子类中实现具体的迁移逻辑
        
        Args:
            source_task: 源任务
            target_task: 目标任务
            model: 要迁移的模型
            similarity: 任务相似度
            
        Returns:
            迁移后的模型
        """
        # 基类仅返回原模型
        if self.verbose:
            print("知识迁移基类不执行实际迁移操作，返回原模型")
        return model


class PolicyTransfer(KnowledgeTransfer):
    """策略迁移
    
    通过迁移策略网络参数实现知识迁移
    """
    
    def __init__(self, config=None):
        """初始化策略迁移类
        
        Args:
            config: 配置字典
        """
        super().__init__(config)
        self.transfer_ratio = self.config.get("knowledge_transfer.policy_transfer_ratio", 0.8)
        
    def _do_transfer(self, source_task, target_task, model, similarity):
        """执行策略网络参数迁移
        
        根据任务相似度计算迁移比例，迁移Actor、Entropy和Critic参数
        
        Args:
            source_task: 源任务
            target_task: 目标任务
            model: 要迁移的模型（一个包装器，包含get_parameters方法）
            similarity: 任务相似度
            
        Returns:
            Dict: 包含迁移后的参数字典，而不是返回模型实例
        """
        # 根据相似度调整迁移比例
        # 相似度越高，迁移比例越接近设定的transfer_ratio
        # 相似度越低，迁移比例越接近0
        adjusted_ratio = self.transfer_ratio * similarity
        
        if self.verbose:
            print(f"知识迁移参数 - 基础迁移比例: {self.transfer_ratio:.4f}, 相似度: {similarity:.4f}")
            print(f"调整后迁移比例: {adjusted_ratio:.4f}")
            
        # 初始化结果参数字典
        result_params = {
            'actors': {},
            'entropy': {},
            'critic': None
        }
        
        # 获取源模型参数
        source_params = None
        if hasattr(model, 'get_parameters'):
            try:
                source_params = model.get_parameters()
                if self.verbose:
                    print(f"成功提取源模型参数用于迁移")
                    if isinstance(source_params, dict):
                        print(f"源模型参数包含以下键: {list(source_params.keys())}")
            except Exception as e:
                if self.verbose:
                    print(f"提取源模型参数失败: {e}")
                    import traceback
                    traceback.print_exc()
        
        if source_params is None:
            if self.verbose:
                print("无法获取源模型参数，无法执行知识迁移")
            return None
            
        # 处理Actor参数
        if 'actors' in source_params and isinstance(source_params['actors'], dict):
            result_params['actors'] = source_params['actors'].copy()  # 使用浅拷贝
            if self.verbose:
                print(f"迁移Actor参数: {len(source_params['actors'])} 项")
            
            # 新增：处理从机数量增加的情况
            # 获取源任务和目标任务的智能体数量
            source_leader_count = source_task.env_params.get("leader_count", 1)
            source_follower_count = source_task.env_params.get("follower_count", 0)
            target_leader_count = target_task.env_params.get("leader_count", 1)
            target_follower_count = target_task.env_params.get("follower_count", 0)
            
            # 计算源任务和目标任务的从机数量
            source_follower_num = source_follower_count
            target_follower_num = target_follower_count
            
            if target_follower_num > source_follower_num and source_follower_num > 0:
                # 从机数量增加，需要复用最后一个从机的参数
                if self.verbose:
                    print(f"\n检测到从机数量增加: {source_follower_num} -> {target_follower_num}")
                    print(f"将复用最后一个已有从机的参数到新增的从机上")
                
                # 找到最后一个从机的Actor参数键
                last_follower_key = None
                for i in range(source_follower_num):
                    follower_key = f'actor_{i+1}'  # 从机索引从1开始(actor_1, actor_2, ...)
                    if follower_key in result_params['actors']:
                        last_follower_key = follower_key
                
                if last_follower_key:
                    # 复用最后一个从机的参数到新增的从机
                    last_follower_params = copy.deepcopy(result_params['actors'][last_follower_key])
                    
                    for i in range(source_follower_num, target_follower_num):
                        new_follower_key = f'actor_{i+1}'
                        result_params['actors'][new_follower_key] = copy.deepcopy(last_follower_params)
                        if self.verbose:
                            print(f"  - 复用 {last_follower_key} 的参数到 {new_follower_key}")
                else:
                    if self.verbose:
                        print(f"警告: 找不到最后一个从机的参数，无法复用")
                
        # 处理Entropy参数
        if 'entropy' in source_params and isinstance(source_params['entropy'], dict):
            result_params['entropy'] = source_params['entropy'].copy()  # 使用浅拷贝
            if self.verbose:
                print(f"迁移Entropy参数: {len(source_params['entropy'])} 项")
            
            # 新增：处理从机数量增加时的Entropy参数复用
            # 注意：通常Entropy参数中，entropy_0是Leader的，entropy_1是Follower的
            # 所有Follower通常共享同一个Entropy参数
            if target_follower_num > source_follower_num and source_follower_num > 0:
                # 检查是否需要为新增的从机复制Entropy参数
                # 通常只有entropy_0(Leader)和entropy_1(Follower共享)
                if 'entropy_1' in result_params['entropy']:
                    if self.verbose:
                        print(f"注意: 从机Entropy参数通常是共享的(entropy_1)，无需为每个从机单独复制")
                
                # 如果未来需要每个从机独立的Entropy参数，可以使用以下代码：
                # for i in range(source_follower_num, target_follower_num):
                #     new_entropy_key = f'entropy_{i+1}'
                #     if 'entropy_1' in result_params['entropy']:
                #         result_params['entropy'][new_entropy_key] = copy.deepcopy(result_params['entropy']['entropy_1'])
                #         if self.verbose:
                #             print(f"  - 复用 entropy_1 的参数到 {new_entropy_key}")
                
        # 处理Critic参数
        if 'critic' in source_params:
            result_params['critic'] = source_params['critic']
            if self.verbose:
                print(f"迁移Critic参数: {'成功' if source_params['critic'] is not None else '失败'}")
        
        # 增加兼容代码，处理旧版本的返回
        # 如果无法按新格式处理，则提取智能体信息
        if result_params['actors'] == {} and result_params['entropy'] == {} and result_params['critic'] is None:
            # 使用兼容的适配方法
            if self.verbose:
                print("无法按新格式提取参数，使用兼容的适配方法")
                
            # 获取源任务和目标任务的智能体数量
            source_leader_count = source_task.env_params.get("leader_count", 1)
            source_follower_count = source_task.env_params.get("follower_count", 1)
            target_leader_count = target_task.env_params.get("leader_count", 1)
            target_follower_count = target_task.env_params.get("follower_count", 1)
            
            # 检查智能体数量是否变化
            source_agent_count = source_leader_count + source_follower_count
            target_agent_count = target_leader_count + target_follower_count
            
            if source_agent_count != target_agent_count:
                if self.verbose:
                    print(f"检测到智能体数量变化: {source_agent_count} -> {target_agent_count}")
                
                # 添加智能体数量信息到结果中
                result_params['agent_counts'] = {
                    'source': source_agent_count,
                    'target': target_agent_count
                }
        
        # 始终添加智能体数量信息（无论是否进行了参数复用）
        if 'agent_counts' not in result_params:
            # 确保智能体数量变量已定义
            if 'source_follower_num' not in locals():
                source_leader_count = source_task.env_params.get("leader_count", 1)
                source_follower_count = source_task.env_params.get("follower_count", 0)
                target_leader_count = target_task.env_params.get("leader_count", 1)
                target_follower_count = target_task.env_params.get("follower_count", 0)
                source_follower_num = source_follower_count
                target_follower_num = target_follower_count
            
            source_agent_count = source_leader_count + source_follower_num
            target_agent_count = target_leader_count + target_follower_num
            
            result_params['agent_counts'] = {
                'source': source_agent_count,
                'target': target_agent_count,
                'source_followers': source_follower_num,
                'target_followers': target_follower_num
            }
            
            if self.verbose:
                print(f"\n智能体数量信息:")
                print(f"  源任务: {source_agent_count} 个智能体 (1个Leader + {source_follower_num}个Follower)")
                print(f"  目标任务: {target_agent_count} 个智能体 (1个Leader + {target_follower_num}个Follower)")
        
        if self.verbose:
            print(f"知识迁移完成，返回参数字典而不是模型实例")
            
        return result_params


class ValueFunctionTransfer(KnowledgeTransfer):
    """价值函数迁移
    
    通过迁移价值函数/评论家网络参数实现知识迁移
    """
    
    def __init__(self, config=None):
        """初始化价值函数迁移类
        
        Args:
            config: 配置字典
        """
        super().__init__(config)
        self.transfer_ratio = self.config.get("knowledge_transfer.value_transfer_ratio", 0.7)
        
    def _do_transfer(self, source_task, target_task, model, similarity):
        """执行价值函数/评论家网络参数迁移
        
        根据任务相似度计算迁移比例，并迁移评论家网络参数
        
        Args:
            source_task: 源任务
            target_task: 目标任务
            model: 要迁移的模型
            similarity: 任务相似度
            
        Returns:
            迁移后的模型
        """
        # 检查模型是否有迁移评论家网络参数的方法
        if not hasattr(model, 'transfer_critic_parameters'):
            if self.verbose:
                print("警告: 模型没有transfer_critic_parameters方法，无法执行评论家参数迁移")
            return model
        
        # 根据相似度调整迁移比例
        adjusted_ratio = self.transfer_ratio * similarity
        
        if self.verbose:
            print(f"执行评论家网络参数迁移，基础迁移比例: {self.transfer_ratio:.4f}, 相似度: {similarity:.4f}")
            print(f"调整后迁移比例: {adjusted_ratio:.4f}")
        
        # 执行评论家网络参数迁移
        model.transfer_critic_parameters(adjusted_ratio)
        
        if self.verbose:
            print(f"评论家网络参数迁移完成")
        
        return model


class HybridTransfer(KnowledgeTransfer):
    """混合知识迁移
    
    综合使用策略迁移和价值函数迁移
    """
    
    def __init__(self, config=None):
        """初始化混合迁移类
        
        Args:
            config: 配置字典
        """
        super().__init__(config)
        self.policy_transfer = PolicyTransfer(config)
        self.value_transfer = ValueFunctionTransfer(config)
        
    def _do_transfer(self, source_task, target_task, model, similarity):
        """执行混合知识迁移
        
        先执行策略迁移，再执行价值函数迁移
        
        Args:
            source_task: 源任务
            target_task: 目标任务
            model: 要迁移的模型
            similarity: 任务相似度
            
        Returns:
            迁移后的模型
        """
        if self.verbose:
            print("执行混合知识迁移，包含策略迁移和价值函数迁移")
        
        # 执行策略迁移
        model = self.policy_transfer._do_transfer(source_task, target_task, model, similarity)
        
        # 执行价值函数迁移
        model = self.value_transfer._do_transfer(source_task, target_task, model, similarity)
        
        if self.verbose:
            print("混合知识迁移完成")
        
        return model


class NoTransfer(KnowledgeTransfer):
    """无知识迁移
    
    完全不执行知识迁移，仅用于对比实验
    """
    
    def _do_transfer(self, source_task, target_task, model, similarity):
        """不执行任何迁移
        
        Args:
            source_task: 源任务
            target_task: 目标任务
            model: 要迁移的模型
            similarity: 任务相似度
            
        Returns:
            原始模型
        """
        if self.verbose:
            print("已配置为不执行任何知识迁移，返回原始模型")
        return model


def create_knowledge_transfer(transfer_type, config=None):
    """创建知识迁移对象工厂函数
    
    根据指定的迁移类型创建相应的知识迁移对象
    
    Args:
        transfer_type: 迁移类型，可选值: "policy", "value", "hybrid", "none"
        config: 配置字典
        
    Returns:
        知识迁移对象
    """
    if transfer_type == "policy":
        return PolicyTransfer(config)
    elif transfer_type == "value":
        return ValueFunctionTransfer(config)
    elif transfer_type == "hybrid":
        return HybridTransfer(config)
    elif transfer_type == "none":
        return NoTransfer(config)
    else:
        print(f"警告: 未知的知识迁移类型 '{transfer_type}'，使用基本知识迁移")
        return KnowledgeTransfer(config) 