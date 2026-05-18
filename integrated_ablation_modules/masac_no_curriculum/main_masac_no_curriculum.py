# -*- coding: utf-8 -*-
import argparse
import gym
import numpy as np
import os
import sys
import time
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import json
import uuid
import random
import matplotlib.pyplot as plt
import shutil
from tqdm.auto import tqdm, trange
import pickle as pkl
import re
from typing import Dict, Any

# 添加上级目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入环境
from rl_env.path_env import RlGame

# 导入网络组件
from integrated_ablation_modules.masac_no_curriculum.role_specific_networks import RoleEmbedding, PolicyNetFlatRole, SharedEncoder, QHead, CriticNetAttentionFlat, ROLE_EMBED_DIM, EMBED_DIM
from integrated_ablation_modules.masac_no_curriculum.masac_adapter import MASACEntroy,set_log_level, MultiHeadAttention, max_action, min_action, LEADER_TYPE_ID, FOLLOWER_TYPE_ID, log, LOG_INFO, LOG_WARNING, LOG_DEBUG, LOG_ERROR, clear_log_history
from main_SAC import Ornstein_Uhlenbeck_Noise
from integrated_ablation_modules.masac_no_curriculum.smer_memory import SMERMemory

# 导入新的 Actor 和 Critic 网络
from integrated_ablation_modules.masac_no_curriculum.actor_networks import LeaderActorNet, FollowerActorNet, AttentionLeaderActorNet, AttentionFollowerActorNet
from integrated_ablation_modules.masac_no_curriculum.critic_networks import StructuredAttentionCriticNet

# 设置随机种子
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

def check_numerical_stability(tensor, name="tensor"):
    """Check numerical stability of tensor
    
    Args:
        tensor: Tensor or numpy array to check
        name: Tensor name for logging
        
    Returns:
        bool: True if stable, False if contains NaN or Inf
    """
    if isinstance(tensor, np.ndarray):
        has_nan = np.isnan(tensor).any()
        has_inf = np.isinf(tensor).any()
    elif isinstance(tensor, torch.Tensor):
        has_nan = torch.isnan(tensor).any().item()
        has_inf = torch.isinf(tensor).any().item()
    elif isinstance(tensor, dict):
        # Recursively check all values in dict
        for key, value in tensor.items():
            if not check_numerical_stability(value, f"{name}.{key}"):
                return False
        return True
    elif isinstance(tensor, (list, tuple)):
        # Recursively check all elements in list or tuple
        for i, item in enumerate(tensor):
            if not check_numerical_stability(item, f"{name}[{i}]"):
                return False
        return True
    else:
        # For scalar values
        try:
            has_nan = np.isnan(float(tensor))
            has_inf = np.isinf(float(tensor))
        except (TypeError, ValueError):
            return True  # Objects that can't convert to numeric are considered stable
    
    if has_nan:
        print(f"Warning: {name} contains NaN values")
        return False
    if has_inf:
        print(f"Warning: {name} contains Inf values")
        return False
    return True

# Role-based MASAC controller
class MASACController:
    """Role-based Multi-Agent SAC Controller"""
    
    def __init__(self, n_agents=1, state_dim=17, action_dim=4, memory_size=int(2e6), \
            batch_size=256, gamma=0.99, tau=0.01, value_lr=3e-4, policy_lr=1e-4, \
            hidden_dim=256, target_update_interval=2, reward_scale=0.1, \
            auto_entropy=True, entropy_lr=3e-4, target_entropy=-0.1, device=None, 
            memory_capacity=None, max_replay_ratio=10.0):  # Add max_replay_ratio parameter
        """Initialize role-based MASAC controller
        
        Args:
            n_agents: Number of agents
            state_dim: State dimension per agent
            action_dim: Action dimension per agent
            memory_size: Replay buffer size (new parameter name)
            memory_capacity: Replay buffer capacity (old parameter name, backward compatible)
            batch_size: Training batch size
            gamma: Discount factor
            tau: Soft update coefficient
            value_lr: Critic learning rate
            policy_lr: Actor learning rate
            hidden_dim: Hidden layer dimension
            target_update_interval: Target network update interval
            reward_scale: Reward scaling factor
            auto_entropy: Whether to auto-adjust entropy
            entropy_lr: Entropy adjustment learning rate
            target_entropy: Target entropy value
            device: Training device
            max_replay_ratio: Maximum replay ratio, limiting upper bound for set_replay_ratio
        """
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.target_update_interval = target_update_interval
        self.reward_scale = reward_scale
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.entropy_lr = entropy_lr # Save entropy_lr as instance attribute
        
        # Handle memory size parameter (backward compatible)
        if memory_capacity is not None:
            print(f"Warning: Deprecated parameter 'memory_capacity' used, please use 'memory_size' instead")
            self.memory_capacity = memory_capacity
            memory_size = memory_capacity
        else:
            self.memory_capacity = memory_size  # Save copy for reset_memory method
        
        print(f"Initializing MASAC controller with graph attention (GAT): {n_agents} agents, state_dim={state_dim}/agent, action_dim={action_dim}/agent")
        print(f"Using device: {self.device}")
        
        # 初始化经验回放缓冲区
        obs_dims = {"leader": state_dim, "followers": state_dim}
        action_dims = {"leader": action_dim, "followers": action_dim}
        
        self.memory = SMERMemory(
            capacity=memory_size,
            obs_dims=obs_dims,
            action_dims=action_dims,
            device=self.device
        )
        
        # Record current agent count version for replay buffer
        self.memory_n_agents_version = n_agents
        
        # === Graph-attention-based Actor network hyperparameters ===
        embed_dim = 128  # Embedding dimension
        actor_hidden_dims = [256, 128]  # Actor hidden layer dimensions
        actor_n_heads = 4  # Number of attention heads
        actor_dropout = 0.1  # Dropout probability
        use_shared_layer = True  # Whether to use shared layer for fusion features
        
        # === Create graph-attention-based Leader/Follower Actor networks ===
        # Leader Actor network
        self.leader_actor = AttentionLeaderActorNet(
            state_dim=state_dim,
            action_dim=action_dim,
            embed_dim=embed_dim,
            hidden_dims=actor_hidden_dims,
            n_heads=actor_n_heads,
            dropout=actor_dropout,
            use_shared_layer=use_shared_layer
        )
        
        # 目标 Leader Actor 网络
        self.target_leader_actor = AttentionLeaderActorNet(
            state_dim=state_dim,
            action_dim=action_dim,
            embed_dim=embed_dim,
            hidden_dims=actor_hidden_dims,
            n_heads=actor_n_heads,
            dropout=actor_dropout,
            use_shared_layer=use_shared_layer
        )
        
        # 加载初始目标 Leader Actor 参数
        self.target_leader_actor.load_state_dict(self.leader_actor.state_dict())
        
        # 确定要创建的最大从机数量
        max_followers = max(n_agents - 1, 3)  # 至少支持3个从机，或根据初始n_agents计算
        
        # 为每个可能的从机创建单独的 Actor 网络
        self.follower_actors = nn.ModuleList([
            AttentionFollowerActorNet(
                state_dim=state_dim,
                action_dim=action_dim,
                embed_dim=embed_dim,
                hidden_dims=actor_hidden_dims,
                n_heads=actor_n_heads,
                dropout=actor_dropout,
                use_shared_layer=use_shared_layer
            ) for _ in range(max_followers)
        ])
        
        # 为每个可能的从机创建单独的目标 Actor 网络
        self.target_follower_actors = nn.ModuleList([
            AttentionFollowerActorNet(
                state_dim=state_dim,
                action_dim=action_dim,
                embed_dim=embed_dim,
                hidden_dims=actor_hidden_dims,
                n_heads=actor_n_heads,
                dropout=actor_dropout,
                use_shared_layer=use_shared_layer
            ) for _ in range(max_followers)
        ])
        
        # 加载初始目标 Follower Actor 参数
        for i in range(max_followers):
            self.target_follower_actors[i].load_state_dict(self.follower_actors[i].state_dict())
            
        # 旧的 LeaderActorNet 和 FollowerActorNet 代码（已注释掉）
        """
        # Leader Actor 网络
        self.leader_actor = LeaderActorNet(
            state_dim=state_dim,
            action_dim=action_dim,
            embed_dim=embed_dim,
            hidden_dims=actor_hidden_dims
        )
        
        # Follower Actor 网络
        self.follower_actor = FollowerActorNet(
            state_dim=state_dim,
            action_dim=action_dim,
            embed_dim=embed_dim,
            hidden_dims=actor_hidden_dims
        )
        
        # 目标 Actor 网络
        self.target_leader_actor = LeaderActorNet(
            state_dim=state_dim,
            action_dim=action_dim,
            embed_dim=embed_dim,
            hidden_dims=actor_hidden_dims
        )
        
        self.target_follower_actor = FollowerActorNet(
            state_dim=state_dim,
            action_dim=action_dim,
            embed_dim=embed_dim,
            hidden_dims=actor_hidden_dims
        )
        
        # 加载初始目标 Actor 参数
        self.target_leader_actor.load_state_dict(self.leader_actor.state_dict())
        self.target_follower_actor.load_state_dict(self.follower_actor.state_dict())
        """
        
        # === 创建图注意力 Critic 网络 ===
        critic_hidden_dims = [256, 128]  # Critic 隐藏层维度
        n_heads = 4  # 注意力头数
        dropout = 0.1  # Dropout 概率
        
        self.critic = StructuredAttentionCriticNet(
            state_dim=state_dim,
            action_dim=action_dim,
            embed_dim=embed_dim,
            n_heads=n_heads,
            hidden_dims=critic_hidden_dims,
            dropout=dropout
        )
        
        # === 初始化熵调整 ===
        # Leader 和 Follower 分别有一个熵参数
        self.entroy_leader = MASACEntroy(action_dim=action_dim)
        self.entroy_follower = MASACEntroy(action_dim=action_dim)
        
        # 设置目标熵
        if target_entropy < 0:
            self.entroy_leader.target_entropy = -0.1
            self.entroy_follower.target_entropy = -0.1
        else:
            self.entroy_leader.target_entropy = target_entropy
            self.entroy_follower.target_entropy = target_entropy
        
        # === 初始化优化器 ===
        # Leader Actor 优化器
        self.leader_actor_optimizer = optim.Adam(self.leader_actor.parameters(), lr=policy_lr)
        
        # Follower Actors 优化器（整合所有Follower Actor参数）
        follower_actor_parameters = []
        for actor_instance in self.follower_actors:
            follower_actor_parameters.extend(list(actor_instance.parameters()))
        self.follower_actor_optimizer = optim.Adam(follower_actor_parameters, lr=policy_lr)
        
        # Critic 优化器 (包含所有 Critic 组件)
        critic_params = []
        critic_params.extend(list(self.critic.leader_encoder.parameters()))
        critic_params.extend(list(self.critic.follower_encoder.parameters()))
        # 更新: 使用重命名后的leader_sees_followers_attention
        critic_params.extend(list(self.critic.leader_sees_followers_attention.parameters()))
        # 添加: 新增的follower_context_attention
        critic_params.extend(list(self.critic.follower_context_attention.parameters()))
        critic_params.extend(list(self.critic.leader_q_head.parameters()))
        critic_params.extend(list(self.critic.follower_q_head.parameters()))
        
        self.critic_optimizer = optim.Adam(critic_params, lr=value_lr)
        
        # 熵参数优化器
        self.leader_alpha_optimizer = optim.Adam([self.entroy_leader.log_alpha], lr=entropy_lr)
        self.follower_alpha_optimizer = optim.Adam([self.entroy_follower.log_alpha], lr=entropy_lr)
        
        # 初始化噪声发生器
        self.noises = [Ornstein_Uhlenbeck_Noise(mu=np.zeros(action_dim)) for _ in range(n_agents)]
        
        # 移动到指定设备
        self.to(self.device)
        
        # 记录训练状态
        self.train_step = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_successes = []
        
        # 初始化回放比例参数
        self.min_replay_ratio = 0.1  # 最小回放比例
        self.max_replay_ratio = max_replay_ratio  # 最大回放比例
        self.replay_ratio = 1.0  # 当前回放比例
        
        # 初始化训练步数计数器
        self.steps_done = 0
    
    def set_replay_ratio(self, ratio):
        """设置重放比例
        
        Args:
            ratio: 新的重放比例
        """
        if ratio < self.min_replay_ratio:
            ratio = self.min_replay_ratio
            print(f"重放比例过小，设置为最小值 {self.min_replay_ratio}")
        elif ratio > self.max_replay_ratio:
            ratio = self.max_replay_ratio
            print(f"重放比例过大，设置为最大值 {self.max_replay_ratio}")
            
        old_ratio = self.replay_ratio
        self.replay_ratio = ratio
        print(f"重放比例从 {old_ratio} 调整为 {self.replay_ratio}")
    
    def reset_memory(self):
        """重置经验回放缓冲区"""
        # 获取观测和动作维度
        obs_dims = {"leader": self.state_dim, "followers": self.state_dim}
        action_dims = {"leader": self.action_dim, "followers": self.action_dim}
        
        # 重新初始化SMERMemory
        self.memory = SMERMemory(
            capacity=self.memory_capacity, # 使用保存的容量
            obs_dims=obs_dims,
            action_dims=action_dims,
            device=self.device
        )
        print(f"重置 SMERMemory 缓冲区，容量: {self.memory_capacity}")
        
        # 重置噪声生成器
        self._reset_noise()
    
    def _reset_noise(self):
        """重置噪声生成器"""
        # 确保有足够的噪声生成器
        if len(self.noises) < self.n_agents:
            # 需要添加新的噪声生成器
            while len(self.noises) < self.n_agents:
                self.noises.append(Ornstein_Uhlenbeck_Noise(mu=np.zeros(self.action_dim)))
        else:
            # 需要减少噪声生成器
            self.noises = self.noises[:self.n_agents]
            
        # 重置所有噪声生成器的状态
        for noise in self.noises:
            noise.reset()
    
    def adapt_to_agent_count(self, n_agents):
        """适应新的智能体数量
        
        Args:
            n_agents: 新的智能体数量
        """
        if n_agents == self.n_agents:
            return
                
        print(f"适应智能体数量变化: {self.n_agents} -> {n_agents}")
        
        # 更新智能体数量
        self.n_agents = n_agents
        
        # 计算从机数量
        num_followers = max(n_agents - 1, 0)
        
        # 检查从机Actor网络的数量是否足够
        if num_followers > len(self.follower_actors):
            log(f"警告: 请求的从机数量 ({num_followers}) 超过了预分配的从机Actor网络数量 ({len(self.follower_actors)})", LOG_WARNING)
            log(f"将使用 {len(self.follower_actors)} 个从机Actor网络处理最多 {len(self.follower_actors)} 个从机", LOG_WARNING)
            log(f"请考虑重新初始化MASACController，提供更大的预分配从机网络数量", LOG_WARNING)
        
        # 调整噪声生成器
        self._reset_noise()
            
        # 更新记忆版本 - 这很重要，确保训练时正确解析状态和动作维度
        self.memory_n_agents_version = n_agents
            
        print(f"成功调整为 {n_agents} 个智能体 (1个Leader + {min(num_followers, len(self.follower_actors))}个可用Follower)")
    
    def to(self, device):
        """将所有网络移动到指定设备
        
        Args:
            device: 目标设备
            
        Returns:
            self: 支持链式调用
        """
        self.device = device
        
        # 移动 Actor 网络
        self.leader_actor = self.leader_actor.to(device)
        self.follower_actors = self.follower_actors.to(device)
        
        # 移动目标 Actor 网络
        self.target_leader_actor = self.target_leader_actor.to(device)
        self.target_follower_actors = self.target_follower_actors.to(device)
        
        # 移动 Critic 网络
        self.critic = self.critic.to(device)
        
        # 移动熵参数
        if hasattr(self.entroy_leader, 'log_alpha') and torch.is_tensor(self.entroy_leader.log_alpha):
            self.entroy_leader.log_alpha = self.entroy_leader.log_alpha.to(device)
            self.entroy_leader.alpha = self.entroy_leader.alpha.to(device)
        if hasattr(self.entroy_follower, 'log_alpha') and torch.is_tensor(self.entroy_follower.log_alpha):
            self.entroy_follower.log_alpha = self.entroy_follower.log_alpha.to(device)
            self.entroy_follower.alpha = self.entroy_follower.alpha.to(device)
        
        # 移动优化器状态
        optimizers_to_move = [
            self.leader_actor_optimizer,
            self.follower_actor_optimizer,
            self.critic_optimizer,
            self.leader_alpha_optimizer,
            self.follower_alpha_optimizer
        ]
        
        for opt in optimizers_to_move:
            # 检查优化器实例是否存在
            if opt is not None:
                # 遍历优化器状态字典中的每个参数组的状态
                for state in opt.state.values():
                    # 遍历状态字典中的每个状态张量
                    for k, v in state.items():
                        # 只移动张量类型的状态
                        if torch.is_tensor(v):
                            # 将张量状态移动到目标设备
                            state[k] = v.to(device)
        
        print(f"MASAC控制器 (包括优化器状态) 已移动到设备: {device}")
        return self
    
    def select_actions(self, observation, add_noise=False, noise_scale=0.1, evaluate=False):
        """为所有智能体选择动作
        
        Args:
            observation: 环境返回的结构化观测 {"leader": obs_leader, "followers": [obs_f1, obs_f2, ...]}
                或旧版的扁平化状态列表/数组 (n_agents, state_dim) 或 (state_dim * n_agents,)
            add_noise: 是否添加探索噪声
            noise_scale: 噪声缩放系数
            evaluate: 是否为评估模式
            
        Returns:
            actions: 结构化动作字典 {"leader": action_leader, "followers": [action_f1, action_f2, ...]}
        """
        # 检查输入是否为结构化格式
        if isinstance(observation, dict) and "leader" in observation and "followers" in observation:
            # 处理结构化输入
            leader_obs = observation["leader"]
            follower_obs_list = observation["followers"]
            
            # 确保是numpy数组
            leader_obs = np.array(leader_obs, dtype=np.float32)
            follower_obs_list = [np.array(obs, dtype=np.float32) for obs in follower_obs_list]
            
            # 创建从机观测的numpy数组和掩码
            num_followers = len(follower_obs_list)
            if num_followers > 0:
                # 创建从机观测数组 [1, num_followers, state_dim]
                followers_obs_array = np.stack(follower_obs_list).reshape(1, num_followers, -1)
                # 创建从机掩码 [1, num_followers]
                followers_mask = np.ones((1, num_followers), dtype=bool)
            else:
                # 如果没有从机，创建空数组
                followers_obs_array = np.zeros((1, 0, self.state_dim), dtype=np.float32)
                followers_mask = np.zeros((1, 0), dtype=bool)
            
            # 选择 Leader 动作
            try:
                # 如果是 AttentionLeaderActorNet，需要提供上下文信息
                if isinstance(self.leader_actor, AttentionLeaderActorNet):
                    leader_action = self.leader_actor.choose_action(
                        leader_obs, 
                        followers_obs_array, 
                        evaluate=evaluate
                    )
                else:
                    # 兼容旧版 LeaderActorNet
                    leader_action = self.leader_actor.choose_action(leader_obs, evaluate=evaluate)
            except Exception as e:
                log(f"选择 Leader 动作时出错: {e}", LOG_ERROR)
                leader_action = np.zeros(self.action_dim)
            
            # 对 Leader 添加噪声 (如果需要)
            if add_noise and not evaluate:
                noise = self.noises[0]() * noise_scale
                leader_action += noise
                leader_action = np.clip(leader_action, min_action, max_action)
            
            # 选择 Followers 动作
            follower_actions = []
            for i, follower_obs in enumerate(follower_obs_list):
                # 确保有足够的噪声生成器
                if i >= len(self.noises):
                    self.noises.append(Ornstein_Uhlenbeck_Noise(mu=np.zeros(self.action_dim)))
                
                try:
                    # 如果是 AttentionFollowerActorNet，需要提供上下文信息
                    if isinstance(self.follower_actors[i], AttentionFollowerActorNet):
                        # 准备 Leader 上下文观测
                        leader_obs_for_context = leader_obs.reshape(1, -1)  # [1, state_dim]
                        
                        # 准备其他从机上下文观测
                        other_followers_indices = [j for j in range(num_followers) if j != i]
                        
                        if other_followers_indices:
                            other_followers_obs = np.stack([follower_obs_list[j] for j in other_followers_indices])
                            other_followers_obs = other_followers_obs.reshape(1, len(other_followers_indices), -1)
                        else:
                            other_followers_obs = np.zeros((1, 0, self.state_dim), dtype=np.float32)
                        
                        # 调用从机Actor网络
                        follower_action = self.follower_actors[i].choose_action(
                            follower_obs,
                            leader_obs_for_context,
                            other_followers_obs,
                            evaluate=evaluate
                        )
                    else:
                        # 兼容旧版 FollowerActorNet
                        follower_action = self.follower_actors[i].choose_action(follower_obs, evaluate=evaluate)
                except Exception as e:
                    log(f"选择第 {i} 个 Follower 动作时出错: {e}", LOG_ERROR)
                    follower_action = np.zeros(self.action_dim)
                
                # 添加探索噪声
                if add_noise and not evaluate:
                    noise = self.noises[i+1]() * noise_scale
                    follower_action += noise
                    follower_action = np.clip(follower_action, min_action, max_action)
                
                follower_actions.append(follower_action)
            
            # 返回结构化动作字典
            return {
                "leader": leader_action,
                "followers": follower_actions
            }
            
        else:
            # 旧版扁平化输入，保持兼容性
            # 确保states是numpy数组
            states = observation
            if not isinstance(states, np.ndarray):
                states = np.array(states)
            
            # 处理不同的输入形状
            if len(states.shape) == 1:
                # 单个向量包含所有智能体的状态
                # 推断智能体数量
                inferred_n_agents = states.shape[0] // self.state_dim
                if inferred_n_agents * self.state_dim != states.shape[0]:
                    log(f"警告: 状态维度 {states.shape[0]} 不是状态维度 {self.state_dim} 的整数倍", LOG_WARNING)
                    inferred_n_agents = max(1, states.shape[0] // self.state_dim)
                    
                # 如果实际智能体数量与控制器设置的不同，动态适应
                if inferred_n_agents != self.n_agents:
                    log(f"检测到智能体数量变化: 控制器={self.n_agents}, 输入状态={inferred_n_agents}", LOG_INFO)
                    self.adapt_to_agent_count(inferred_n_agents)
                    
                # 拆分成n_agents个智能体的状态
                states_list = []
                for i in range(self.n_agents):
                    if i * self.state_dim < states.shape[0]:
                        agent_state = states[i*self.state_dim:(i+1)*self.state_dim]
                        # 确保状态维度正确
                        if len(agent_state) < self.state_dim:
                            agent_state = np.pad(agent_state, (0, self.state_dim - len(agent_state)))
                        states_list.append(agent_state)
                    else:
                        # 如果状态不足，使用零填充
                        states_list.append(np.zeros(self.state_dim))
                    
            elif len(states.shape) == 2:
                # 已经按智能体排列的状态列表
                actual_n_agents = states.shape[0]
                
                # 如果实际智能体数量与控制器设置的不同，动态适应
                if actual_n_agents != self.n_agents:
                    log(f"检测到智能体数量变化: 控制器={self.n_agents}, 输入状态={actual_n_agents}", LOG_INFO)
                    self.adapt_to_agent_count(actual_n_agents)
                
                states_list = [states[i] if i < actual_n_agents else np.zeros(self.state_dim) 
                              for i in range(self.n_agents)]
            else:
                log(f"错误: 状态的形状 {states.shape} 无法解析为智能体的状态", LOG_ERROR)
                # 返回零动作
                return {
                    "leader": np.zeros(self.action_dim),
                    "followers": [np.zeros(self.action_dim) for _ in range(self.n_agents-1)]
                }
                
            # 为每个智能体选择动作
            leader_action = None
            follower_actions = []
            
            # 获取Leader状态和所有从机状态
            leader_state = states_list[0]
            follower_states = states_list[1:] if len(states_list) > 1 else []
            
            # 构建从机状态数组和掩码 (用于注意力机制)
            if follower_states:
                follower_states_array = np.stack(follower_states).reshape(1, len(follower_states), -1)
                follower_mask = np.ones((1, len(follower_states)), dtype=bool)
            else:
                follower_states_array = np.zeros((1, 0, self.state_dim), dtype=np.float32)
                follower_mask = np.zeros((1, 0), dtype=bool)
            
            # Leader (第一个智能体)
            if len(states_list) > 0:
                try:
                    # 如果是 AttentionLeaderActorNet，需要提供上下文信息
                    if isinstance(self.leader_actor, AttentionLeaderActorNet):
                        leader_action = self.leader_actor.choose_action(
                            leader_state, 
                            follower_states_array, 
                            evaluate=evaluate
                        )
                    else:
                        # 兼容旧版 LeaderActorNet
                        leader_action = self.leader_actor.choose_action(leader_state, evaluate=evaluate)
                    
                    # 添加探索噪声
                    if add_noise and not evaluate:
                        noise = self.noises[0]() * noise_scale
                        leader_action += noise
                        leader_action = np.clip(leader_action, min_action, max_action)
                except Exception as e:
                    log(f"选择 Leader 动作时出错: {e}", LOG_ERROR)
                    leader_action = np.zeros(self.action_dim)
            
            # Followers (其余智能体)
            for i in range(1, len(states_list)):
                follower_idx = i - 1  # 从机索引 (0-based)
                
                try:
                    # 如果是 AttentionFollowerActorNet，需要提供上下文信息
                    if isinstance(self.follower_actors[follower_idx], AttentionFollowerActorNet):
                        # 准备 Leader 上下文观测
                        leader_obs_for_context = leader_state.reshape(1, -1)  # [1, state_dim]
                        
                        # 准备其他从机上下文观测
                        other_followers_indices = [j-1 for j in range(1, len(states_list)) if j != i]
                        
                        if other_followers_indices:
                            other_followers_obs = np.stack([states_list[j+1] for j in other_followers_indices])
                            other_followers_obs = other_followers_obs.reshape(1, len(other_followers_indices), -1)
                        else:
                            other_followers_obs = np.zeros((1, 0, self.state_dim), dtype=np.float32)
                        
                        # 调用从机Actor网络
                        follower_action = self.follower_actors[follower_idx].choose_action(
                            states_list[i],
                            leader_obs_for_context,
                            other_followers_obs,
                            evaluate=evaluate
                        )
                    else:
                        # 兼容旧版 FollowerActorNet
                        follower_action = self.follower_actors[follower_idx].choose_action(states_list[i], evaluate=evaluate)
                    
                    # 添加探索噪声
                    if add_noise and not evaluate:
                        # 确保有足够的噪声生成器
                        if i >= len(self.noises):
                            self.noises.append(Ornstein_Uhlenbeck_Noise(mu=np.zeros(self.action_dim)))
                        
                        noise = self.noises[i]() * noise_scale
                        follower_action += noise
                        follower_action = np.clip(follower_action, min_action, max_action)
                    
                    follower_actions.append(follower_action)
                except Exception as e:
                    log(f"选择第 {i} 个 Follower 动作时出错: {e}", LOG_ERROR)
                    follower_actions.append(np.zeros(self.action_dim))

            # 检查动作的数值稳定性
            actions_dict = {
                "leader": leader_action,
                "followers": follower_actions
            }
            
            # 进行数值稳定性检查
            if not check_numerical_stability(actions_dict, "actions"):
                print("检测到动作中存在NaN或Inf值，返回零动作")
                return {
                    "leader": np.zeros(self.action_dim),
                    "followers": [np.zeros(self.action_dim) for _ in range(len(follower_actions))]
                }
            
            return actions_dict
    
    def store_transition(self, states, actions, rewards, next_states, done=False, current_stage_tag: str = "default_stage"):
        """存储转换到经验回放缓冲区
        
        Args:
            states: 当前状态数组 [n_agents, state_dim] 或结构化字典
            actions: 动作数组 [n_agents, action_dim] 或结构化字典
            rewards: 奖励数组 [n_agents] 或结构化字典
            next_states: 下一状态数组 [n_agents, state_dim] 或结构化字典
            done: 是否结束标志
            current_stage_tag: 当前课程阶段标签，用于标记经验所属阶段
        """
        # 检查输入是否已经是结构化格式
        if (isinstance(states, dict) and "leader" in states and "followers" in states and
            isinstance(actions, dict) and "leader" in actions and "followers" in actions and
            isinstance(rewards, dict) and "leader" in rewards and "followers" in rewards and
            isinstance(next_states, dict) and "leader" in next_states and "followers" in next_states):
            # 已经是结构化格式，直接存储
            self.memory.store_transition(states, actions, rewards, next_states, done, stage_tag=current_stage_tag)
            return
            
        # 处理扁平化格式
        # 确保输入是 numpy 数组
        states = np.asarray(states)
        actions = np.asarray(actions)
        rewards = np.asarray(rewards)
        next_states = np.asarray(next_states)
        
        n_agents = states.shape[0]
        
        if n_agents == 0:
            log("store_transition: 接收到 0 个智能体的状态，跳过存储。", LOG_WARNING)
            return

        # 分离 leader 和 followers
        leader_state = states[0]
        follower_states = states[1:] if n_agents > 1 else []

        leader_action = actions[0]
        follower_actions = actions[1:] if n_agents > 1 else []

        leader_reward = rewards[0] 
        follower_rewards = rewards[1:] if n_agents > 1 else []
        
        leader_next_state = next_states[0]
        follower_next_states = next_states[1:] if n_agents > 1 else []

        # 构建字典
        observation = {"leader": leader_state, "followers": list(follower_states)}
        action = {"leader": leader_action, "followers": list(follower_actions)}
        reward = {"leader": leader_reward, "followers": list(follower_rewards)}
        next_observation = {"leader": leader_next_state, "followers": list(follower_next_states)}
        
        # 调用 SMERMemory 的存储方法，并传递课程阶段标签
        self.memory.store_transition(observation, action, reward, next_observation, done, stage_tag=current_stage_tag)
    
    def train(self, batch_size=None, current_stage_tag: str = "default_stage", current_stage_number: int = 0):
        """训练网络

        Args:
            batch_size: 批次大小，如果为None则使用self.batch_size
            current_stage_tag: 当前课程阶段标签，用于区分新旧经验
            current_stage_number: 当前课程阶段编号，用于计算新旧经验采样比例
        """
        if batch_size is None:
            batch_size = self.batch_size

        # 从经验回放缓冲区采样，传递阶段信息
        sampled_data = self.memory.sample(
            batch_size,
            current_stage_tag=current_stage_tag,
            current_stage_number=current_stage_number
        )
        
        if sampled_data is None:
            log(f"MASACController: 阶段 '{current_stage_tag}' (编号 {current_stage_number}) 因样本不足或采样错误跳过训练步骤。", LOG_DEBUG)
            return  # 经验不足或采样失败，跳过此次训练

        batch_data, batch_masks = sampled_data
        
        # 获取 leader 和 followers 的数据
        obs_leader = batch_data["observation"]["leader"]
        obs_followers = batch_data["observation"]["followers"]
        mask_followers = batch_masks["followers"]
        
        act_leader = batch_data["action"]["leader"]
        act_followers = batch_data["action"]["followers"]
        
        reward_leader = batch_data["reward"]["leader"]
        reward_followers = batch_data["reward"]["followers"]
        
        next_obs_leader = batch_data["next_observation"]["leader"]
        next_obs_followers = batch_data["next_observation"]["followers"]
        
        done = batch_data["done"]
        
        # 获取从机的数量（batch中可能有不同数量的从机）
        B, max_F, _ = obs_followers.shape
        num_active_followers = min(max_F, len(self.follower_actors))
        
        # ===== 计算 Critic 损失 =====
        # 1. 计算目标 Q 值
        with torch.no_grad():
            # Leader的下一状态动作评估
            if isinstance(self.target_leader_actor, AttentionLeaderActorNet):
                # 使用注意力Leader Actor网络
                next_act_leader, next_log_prob_leader = self.target_leader_actor.evaluate(
                    next_obs_leader, next_obs_followers
                )
            else:
                # 使用旧版Leader Actor网络
                next_act_leader, next_log_prob_leader = self.target_leader_actor.evaluate(next_obs_leader)
            
            # Followers的下一状态动作评估
            next_act_followers_list = []
            next_log_prob_followers_list = []
            
            for k in range(num_active_followers):
                # 准备从机k的上下文数据
                follower_self_obs_k = next_obs_followers[:, k, :]  # [B, state_dim]
                leader_for_context_k = next_obs_leader  # [B, state_dim]
                
                # 准备其他从机上下文
                other_follower_indices = [j for j in range(max_F) if j != k]
                if other_follower_indices:
                    other_followers_context_k = next_obs_followers[:, other_follower_indices, :]  # [B, max_F-1, state_dim]
                else:
                    other_followers_context_k = torch.zeros(B, 0, self.state_dim, device=self.device)
                
                if isinstance(self.target_follower_actors[k], AttentionFollowerActorNet):
                    # 使用注意力Follower Actor网络
                    act_k, log_prob_k = self.target_follower_actors[k].evaluate(
                        follower_self_obs_k,
                        leader_for_context_k,
                        other_followers_context_k
                    )
                else:
                    # 使用旧版Follower Actor网络
                    act_k, log_prob_k = self.target_follower_actors[k].evaluate(follower_self_obs_k)
                
                # 收集结果
                next_act_followers_list.append(act_k.unsqueeze(1))  # 添加从机维度 [B, 1, action_dim]
                next_log_prob_followers_list.append(log_prob_k.unsqueeze(1))  # [B, 1, 1]
            
            # 如果没有足够的从机Actor网络，用零填充
            if num_active_followers < max_F:
                # 创建填充动作
                action_padding = torch.zeros(
                    B, max_F - num_active_followers, self.action_dim, 
                    device=self.device
                )
                # 创建填充对数概率
                log_prob_padding = torch.zeros(
                    B, max_F - num_active_followers, 1, 
                    device=self.device
                )
                
                # 如果有实际的从机动作，则与填充连接
                if next_act_followers_list:
                    next_act_followers = torch.cat(next_act_followers_list + [action_padding], dim=1)
                    next_log_prob_followers = torch.cat(next_log_prob_followers_list + [log_prob_padding], dim=1)
                else:
                    # 如果没有实际从机动作，直接使用填充
                    next_act_followers = action_padding
                    next_log_prob_followers = log_prob_padding
            else:
                # 如果有足够的从机Actor网络，直接连接所有结果
                if next_act_followers_list:
                    next_act_followers = torch.cat(next_act_followers_list, dim=1)  # [B, max_F, action_dim]
                    next_log_prob_followers = torch.cat(next_log_prob_followers_list, dim=1)  # [B, max_F, 1]
                else:
                    # 极端情况：没有从机
                    next_act_followers = torch.zeros(B, 0, self.action_dim, device=self.device)
                    next_log_prob_followers = torch.zeros(B, 0, 1, device=self.device)
            
            # 计算目标 Q 值
            target_q1_leader, target_q2_leader, target_q1_followers, target_q2_followers = self.critic.forward_target(
                next_obs_leader, next_obs_followers, 
                next_act_leader, next_act_followers
            )
            
            # 使用 min Q
            target_q_leader = torch.min(target_q1_leader, target_q2_leader)
            target_q_followers = torch.min(target_q1_followers, target_q2_followers)
            
            # 计算目标值 (奖励 + gamma * (Q - alpha * log_prob))
            # Leader 目标
            target_leader = reward_leader + self.gamma * (1 - done) * (
                target_q_leader - self.entroy_leader.alpha * next_log_prob_leader
            )
            
            # Followers 目标 (应用掩码)
            target_followers = reward_followers + self.gamma * (1 - done).unsqueeze(1) * (
                target_q_followers - self.entroy_follower.alpha * next_log_prob_followers
            )
        
        # 2. 计算当前 Q 值
        current_q1_leader, current_q2_leader, current_q1_followers, current_q2_followers = self.critic(
            obs_leader, obs_followers,
            act_leader, act_followers
        )
        
        # 3. 计算 Critic 损失 (MSE)
        # Leader 损失
        critic_loss_leader = F.mse_loss(current_q1_leader, target_leader) + F.mse_loss(current_q2_leader, target_leader)
        
        # Followers 损失
        critic_loss_followers_q1 = F.mse_loss(current_q1_followers, target_followers)
        critic_loss_followers_q2 = F.mse_loss(current_q2_followers, target_followers)
        
        critic_loss_followers = critic_loss_followers_q1 + critic_loss_followers_q2
        
        # 总 Critic 损失
        critic_loss = critic_loss_leader + critic_loss_followers
        
        # 更新 Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # ===== 计算 Actor 损失 =====
        # Leader Actor 损失
        # 1. 生成新动作
        if isinstance(self.leader_actor, AttentionLeaderActorNet):
            # 使用注意力Leader Actor网络
            new_act_leader, log_prob_leader = self.leader_actor.evaluate(
                obs_leader, obs_followers
            )
        else:
            # 使用旧版Leader Actor网络
            new_act_leader, log_prob_leader = self.leader_actor.evaluate(obs_leader)
        
        # 2. 计算 Q 值
        q1_leader, q2_leader, _, _ = self.critic(
            obs_leader, obs_followers,
            new_act_leader, act_followers  # 使用新的 Leader 动作，保持 Followers 动作不变
        )
        min_q_leader = torch.min(q1_leader, q2_leader)
        
        # 3. 计算 Leader Actor 损失 (策略梯度，最大化 Q - alpha * log_prob)
        actor_loss_leader = (self.entroy_leader.alpha * log_prob_leader - min_q_leader).mean()
        
        # 更新 Leader Actor
        self.leader_actor_optimizer.zero_grad()
        actor_loss_leader.backward()
        self.leader_actor_optimizer.step()
        
        # Follower Actors 损失
        # 为每个从机生成新动作
        new_act_followers_list = []
        log_prob_followers_list = []
        
        for k in range(num_active_followers):
            # 准备从机k的上下文数据
            follower_self_obs_k = obs_followers[:, k, :]  # [B, state_dim]
            leader_for_context_k = obs_leader  # [B, state_dim]
            
            # 准备其他从机上下文
            other_follower_indices = [j for j in range(max_F) if j != k]
            if other_follower_indices:
                other_followers_context_k = obs_followers[:, other_follower_indices, :]  # [B, max_F-1, state_dim]
            else:
                other_followers_context_k = torch.zeros(B, 0, self.state_dim, device=self.device)
            
            if isinstance(self.follower_actors[k], AttentionFollowerActorNet):
                # 使用注意力Follower Actor网络
                act_k, log_prob_k = self.follower_actors[k].evaluate(
                    follower_self_obs_k,
                    leader_for_context_k,
                    other_followers_context_k
                )
            else:
                # 使用旧版Follower Actor网络
                act_k, log_prob_k = self.follower_actors[k].evaluate(follower_self_obs_k)
            
            # 收集结果
            new_act_followers_list.append(act_k.unsqueeze(1))  # 添加从机维度 [B, 1, action_dim]
            log_prob_followers_list.append(log_prob_k.unsqueeze(1))  # [B, 1, 1]
        
        # 如果没有足够的从机Actor网络，用零填充
        if num_active_followers < max_F:
            # 创建填充动作
            action_padding = torch.zeros(
                B, max_F - num_active_followers, self.action_dim, 
                device=self.device
            )
            # 创建填充对数概率
            log_prob_padding = torch.zeros(
                B, max_F - num_active_followers, 1, 
                device=self.device
            )
            
            # 如果有实际的从机动作，则与填充连接
            if new_act_followers_list:
                new_act_followers = torch.cat(new_act_followers_list + [action_padding], dim=1)
                log_prob_followers = torch.cat(log_prob_followers_list + [log_prob_padding], dim=1)
            else:
                # 如果没有实际从机动作，直接使用填充
                new_act_followers = action_padding
                log_prob_followers = log_prob_padding
        else:
            # 如果有足够的从机Actor网络，直接连接所有结果
            if new_act_followers_list:
                new_act_followers = torch.cat(new_act_followers_list, dim=1)  # [B, max_F, action_dim]
                log_prob_followers = torch.cat(log_prob_followers_list, dim=1)  # [B, max_F, 1]
            else:
                # 极端情况：没有从机
                new_act_followers = torch.zeros(B, 0, self.action_dim, device=self.device)
                log_prob_followers = torch.zeros(B, 0, 1, device=self.device)
        
        # 2. 计算 Q 值
        _, _, q1_followers, q2_followers = self.critic(
            obs_leader, obs_followers,
            act_leader, new_act_followers  # 使用新的 Followers 动作，保持 Leader 动作不变
        )
        min_q_followers = torch.min(q1_followers, q2_followers)
        
        # 3. 计算 Follower Actor 损失 (策略梯度，最大化 Q - alpha * log_prob)
        actor_loss_followers = (self.entroy_follower.alpha * log_prob_followers - min_q_followers).mean()
        
        # 更新 Follower Actor
        self.follower_actor_optimizer.zero_grad()
        actor_loss_followers.backward()
        self.follower_actor_optimizer.step()
        
        # ===== 更新熵权重 alpha =====
        # Leader alpha
        alpha_loss_leader = -(self.entroy_leader.log_alpha * (
            log_prob_leader.detach() + self.entroy_leader.target_entropy
        )).mean()
        
        self.leader_alpha_optimizer.zero_grad()
        alpha_loss_leader.backward()
        self.leader_alpha_optimizer.step()
        
        # 更新 alpha 值
        self.entroy_leader.alpha = self.entroy_leader.log_alpha.exp()
        
        # Follower alpha
        alpha_loss_followers = -(self.entroy_follower.log_alpha * (
            log_prob_followers.detach() + self.entroy_follower.target_entropy
        )).mean()
        
        self.follower_alpha_optimizer.zero_grad()
        alpha_loss_followers.backward()
        self.follower_alpha_optimizer.step()
        
        # 更新 alpha 值
        self.entroy_follower.alpha = self.entroy_follower.log_alpha.exp()
        
        # ===== 软更新目标网络 =====
        # 更新 Critic 目标网络
        self.critic.soft_update(self.tau)
        
        # 更新 Leader Actor 目标网络
        for target_param, param in zip(self.target_leader_actor.parameters(), self.leader_actor.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
        
        # 更新 Follower Actors 目标网络
        for i in range(len(self.follower_actors)):
            for target_param, param in zip(self.target_follower_actors[i].parameters(), self.follower_actors[i].parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
            
        # 更新训练步数
        self.train_step += 1
    
    def save_models(self, path):
        """保存模型
        
        Args:
            path: 保存路径
        """
        # 确保目录存在
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # 保存网络参数
        torch.save({
            # Actor 网络参数
            'leader_actor': self.leader_actor.state_dict(),
            'follower_actors': [actor.state_dict() for actor in self.follower_actors],
            'target_leader_actor': self.target_leader_actor.state_dict(),
            'target_follower_actors': [actor.state_dict() for actor in self.target_follower_actors],
            
            # Critic 网络参数
            'critic': self.critic.state_dict(),
            
            # 熵参数
            'entroy_leader': self.entroy_leader.__dict__,
            'entroy_follower': self.entroy_follower.__dict__,
            
            # 优化器参数
            'leader_actor_optimizer': self.leader_actor_optimizer.state_dict(),
            'follower_actor_optimizer': self.follower_actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'leader_alpha_optimizer': self.leader_alpha_optimizer.state_dict(),
            'follower_alpha_optimizer': self.follower_alpha_optimizer.state_dict(),
            
            # 训练统计信息
            'train_step': self.train_step,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'episode_successes': self.episode_successes,
            
            # 配置
            'n_agents': self.n_agents,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'gamma': self.gamma,
            'tau': self.tau,
            'batch_size': self.batch_size,
            'memory_capacity': self.memory_capacity,
            'replay_ratio': self.replay_ratio
        }, path)
        
        print(f"模型已保存到 {path}")
    
    def load_models(self, path, strict=False):
        """加载模型
        
        Args:
            path: 模型路径
            strict: 是否严格加载（若为False，则允许部分参数不匹配）
            
        Returns:
            bool: 是否成功加载
        """
        if not os.path.exists(path):
            print(f"模型文件不存在: {path}")
            return False
            
        try:
            checkpoint = torch.load(path, map_location=self.device)
            
            # 加载 Actor 网络参数
            self.leader_actor.load_state_dict(checkpoint['leader_actor'], strict=strict)
            for i, actor in enumerate(self.follower_actors):
                actor.load_state_dict(checkpoint['follower_actors'][i], strict=strict)
            
            # 加载目标 Actor 网络参数
            if 'target_leader_actor' in checkpoint:
                self.target_leader_actor.load_state_dict(checkpoint['target_leader_actor'], strict=strict)
            else:
                self.target_leader_actor.load_state_dict(self.leader_actor.state_dict())
                
            if 'target_follower_actors' in checkpoint:
                for i, actor in enumerate(self.target_follower_actors):
                    actor.load_state_dict(checkpoint['target_follower_actors'][i], strict=strict)
            else:
                for i, actor in enumerate(self.follower_actors):
                    actor.load_state_dict(self.follower_actors[i].state_dict())
            
            # 加载 Critic 网络参数
            self.critic.load_state_dict(checkpoint['critic'], strict=strict)
            
            # 加载熵参数
            if 'entroy_leader' in checkpoint:
                # 处理张量
                for key, value in checkpoint['entroy_leader'].items():
                    if isinstance(value, torch.Tensor):
                        setattr(self.entroy_leader, key, value.to(self.device))
                    else:
                        setattr(self.entroy_leader, key, value)
                        
            if 'entroy_follower' in checkpoint:
                # 处理张量
                for key, value in checkpoint['entroy_follower'].items():
                    if isinstance(value, torch.Tensor):
                        setattr(self.entroy_follower, key, value.to(self.device))
                    else:
                        setattr(self.entroy_follower, key, value)
            
            # 加载优化器参数（如果存在）
            if 'leader_actor_optimizer' in checkpoint:
                self.leader_actor_optimizer.load_state_dict(checkpoint['leader_actor_optimizer'])
                
            if 'follower_actor_optimizer' in checkpoint:
                self.follower_actor_optimizer.load_state_dict(checkpoint['follower_actor_optimizer'])
                
            if 'critic_optimizer' in checkpoint:
                self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
                
            if 'leader_alpha_optimizer' in checkpoint:
                self.leader_alpha_optimizer.load_state_dict(checkpoint['leader_alpha_optimizer'])
                
            if 'follower_alpha_optimizer' in checkpoint:
                self.follower_alpha_optimizer.load_state_dict(checkpoint['follower_alpha_optimizer'])
            
            # 加载训练统计信息（如果存在）
            if 'train_step' in checkpoint:
                self.train_step = checkpoint['train_step']
                
            if 'episode_rewards' in checkpoint:
                self.episode_rewards = checkpoint['episode_rewards']
                
            if 'episode_lengths' in checkpoint:
                self.episode_lengths = checkpoint['episode_lengths']
                
            if 'episode_successes' in checkpoint:
                self.episode_successes = checkpoint['episode_successes']
            
            # 加载配置（如果存在）
            if 'n_agents' in checkpoint:
                self.n_agents = checkpoint['n_agents']
                
            if 'replay_ratio' in checkpoint:
                self.replay_ratio = checkpoint['replay_ratio']
            
            print(f"成功加载模型: {path}")
            return True
            
        except Exception as e:
            print(f"加载模型时出错: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def track_episode_rewards(self, rewards):
        """跟踪回合奖励
        
        Args:
            rewards: 每个智能体的奖励列表或数组
        """
        # 计算总奖励（所有智能体奖励之和）
        total_reward = sum(rewards) if isinstance(rewards, (list, np.ndarray)) else rewards
        self.episode_rewards.append(total_reward)
        
    def track_episode_length(self, length):
        """跟踪回合长度
        
        Args:
            length: 回合中步骤数
        """
        self.episode_lengths.append(length)
        
    def track_episode_success(self, success):
        """跟踪回合成功标志
        
        Args:
            success: 是否成功完成任务
        """
        self.episode_successes.append(1 if success else 0)
        
    def get_training_stats(self, window=100):
        """获取训练统计信息
        
        Args:
            window: 计算平均值的窗口大小
            
        Returns:
            dict: 统计信息字典
        """
        stats = {}
        
        # 计算平均奖励
        if len(self.episode_rewards) > 0:
            stats['last_reward'] = self.episode_rewards[-1]
            stats['avg_reward'] = np.mean(self.episode_rewards[-window:])
            
        # 计算平均步长
        if len(self.episode_lengths) > 0:
            stats['last_length'] = self.episode_lengths[-1]
            stats['avg_length'] = np.mean(self.episode_lengths[-window:])
            
        # 计算成功率
        if len(self.episode_successes) > 0:
            stats['last_success'] = self.episode_successes[-1]
            stats['success_rate'] = np.mean(self.episode_successes[-window:])
            
        # 训练步骤
        stats['train_step'] = self.train_step
        
        return stats

def ensure_dir_exists(dir_path):
    """确保目录存在，如果不存在则创建
    
    Args:
        dir_path: 目录路径
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
        print(f"创建目录: {dir_path}")
    return dir_path

def get_timestamp():
    """获取格式化的时间戳
    
    Returns:
        格式化的时间戳字符串: YYYYMMDD_HHMMSS
    """
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def convert_to_json_compatible(obj):
    """将对象转换为JSON兼容格式
    
    处理numpy数组、列表、字典等数据类型，使其可以被JSON序列化
    
    Args:
        obj: 需要转换的对象
        
    Returns:
        转换后的JSON兼容对象
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, 
                          np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_compatible(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_json_compatible(value) for key, value in obj.items()}
    else:
        return obj

def create_test_results_index():
    """创建测试结果索引文件
    
    生成一个HTML文件，列出所有测试结果，便于浏览
    """
    # 确保测试结果目录存在
    ensure_dir_exists(TEST_RESULTS_BASE)
    
    # 查找所有测试结果目录
    test_dirs = []
    for item in os.listdir(TEST_RESULTS_BASE):
        item_path = os.path.join(TEST_RESULTS_BASE, item)
        if os.path.isdir(item_path):
            # 获取目录信息
            try:
                # 检查是否有JSON结果文件
                json_files = [f for f in os.listdir(item_path) if f.endswith('.json')]
                info_file = os.path.join(item_path, "test_info.json")
                
                if os.path.exists(info_file):
                    with open(info_file, 'r', encoding='utf-8') as f:
                        info = json.load(f)
                    
                    # 收集测试信息
                    test_dirs.append({
                        'dir_name': item,
                        'timestamp': info.get('timestamp', ''),
                        'date': info.get('date', ''),
                        'config': info.get('config', {}),
                        'images': [f for f in os.listdir(item_path) if f.endswith('.png')],
                        'success_rate': info.get('success_rate', 'N/A'),
                        'path': item_path
                    })
            except Exception as e:
                print(f"处理目录 {item_path} 时出错: {e}")
    
    # 按时间戳排序
    test_dirs.sort(key=lambda x: x['timestamp'], reverse=True)
    
    # 生成HTML内容
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>测试结果索引</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #333; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .img-link {{ margin-right: 10px; }}
        </style>
    </head>
    <body>
        <h1>测试结果索引</h1>
        <p>生成时间: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        <p>共找到 {len(test_dirs)} 个测试结果</p>
        
        <table>
            <tr>
                <th>测试日期</th>
                <th>配置</th>
                <th>成功率</th>
                <th>结果图片</th>
                <th>详细信息</th>
            </tr>
    """
    
    for test in test_dirs:
        config_str = ""
        config = test.get('config', {})
        if config:
            config_str = f"友方:{config.get('hero_count', 'N/A')}, " \
                        f"敌方:{config.get('enemy_count', 'N/A')}, " \
                        f"障碍:{config.get('obstacle_count', 'N/A')}"
            if config.get('uav_speed', 'N/A') != 'N/A':
                config_str += f", 速度:{config['uav_speed']}"
        
        # 图片链接
        img_links = ""
        for img in test.get('images', []):
            img_path = os.path.join(test['path'], img).replace('\\', '/')
            img_links += f'<a href="file:///{img_path}" class="img-link" target="_blank">{img}</a>'
        
        # 详细信息链接
        info_link = os.path.join(test['path'], "test_info.json").replace('\\', '/')
        result_link = os.path.join(test['path'], "test_results.json").replace('\\', '/')
        
        html_content += f"""
            <tr>
                <td>{test.get('date', test.get('timestamp', 'N/A'))}</td>
                <td>{config_str}</td>
                <td>{test.get('success_rate', 'N/A')}</td>
                <td>{img_links}</td>
                <td>
                    <a href="file:///{info_link}" target="_blank">测试信息</a> | 
                    <a href="file:///{result_link}" target="_blank">详细结果</a>
                </td>
            </tr>
        """
    
    html_content += """
        </table>
    </body>
    </html>
    """
    
    # 保存HTML文件
    index_path = os.path.join(TEST_RESULTS_BASE, "index.html")
    with open(index_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"测试结果索引已生成: {index_path}")
    return index_path

def print_task_details(task, title="任务详情"):
    """打印任务的详细信息
    
    增强了对固定任务的支持
    
    Args:
        task: 要打印的任务
        title: 显示标题
    """
    # 识别是否为固定任务
    is_fixed_task = 'task_' not in task.id
    task_type = "固定任务" if is_fixed_task else "动态任务"
    
    log(f"\n=== {title} ({task_type}) ===", LOG_INFO)
    log(f"任务ID: {task.id}", LOG_INFO)
    log(f"难度: {task.difficulty:.2f}", LOG_INFO)
    
    # 打印环境参数
    log("环境参数:", LOG_INFO)
    log(f"  - 友方无人机: {task.env_params.get('hero_count', 1)}", LOG_INFO)
    log(f"  - 敌方无人机: {task.env_params.get('enemy_count', 1)}", LOG_INFO)
    log(f"  - 障碍物数量: {task.env_params.get('obstacle_count', 1)}", LOG_INFO)
    uav_speed_param = task.env_params.get('uav_speed')
    if uav_speed_param is not None:
        log(f"  - 无人机速度 (任务指定): {uav_speed_param}", LOG_INFO)
    else:
        log(f"  - 无人机速度 (Agent默认): 由Agent自行随机初始化", LOG_INFO)
    # 如果有性能历史，打印最近的性能
    if task.performance_history:
        recent = task.performance_history[-1]
        log("最近性能:", LOG_INFO)
        for k, v in recent['metrics'].items():
            log(f"  - {k}: {v:.2f}", LOG_INFO)
    
    # 如果是固定任务，显示特殊提示
    if is_fixed_task:
        log("注意: 这是预定义的固定难度梯度任务，使用宽松的评估标准", LOG_INFO)
    
    log("=" * (len(title) + 10), LOG_INFO)
    log("", LOG_INFO)

def calculate_five_performance_metrics(episode_data_list):
    """计算五项性能评估指标
    
    Args:
        episode_data_list: 包含每个回合数据的列表
        
    Returns:
        dict: 包含五项指标的字典
    """
    import numpy as np
    import math
    
    total_episodes = len(episode_data_list)
    
    # 1. 任务完成率 (Mission Completion Rate, MCR)
    successful_episodes = sum(1 for episode in episode_data_list if episode.get('success', False))
    mcr = successful_episodes / total_episodes if total_episodes > 0 else 0.0
    
    # 2. 编队保持率 (Formation Keeping Rate, FKR)
    formation_rates = [episode.get('formation_rate', 0.0) for episode in episode_data_list]
    fkr_mean = np.mean(formation_rates) if formation_rates else 0.0
    fkr_std = np.std(formation_rates) if formation_rates else 0.0
    
    # 3. 成功率加权探索时间 (Success-weighted Exploration Time, SET)
    exploration_times = []
    successful_times = []
    
    for episode in episode_data_list:
        time_steps = episode.get('steps', 0)
        success = episode.get('success', False)
        exploration_times.append(time_steps)
        if success:
            successful_times.append(time_steps)
    
    avg_exploration_time = np.mean(successful_times) if successful_times else np.mean(exploration_times) if exploration_times else 0.0
    set_score = mcr * avg_exploration_time
    
    # 4. 飞行轨迹平滑性指标 (Trajectory Smoothness, J_S)
    trajectory_smoothness_values = []
    
    for episode in episode_data_list:
        if 'trajectory_data' in episode and len(episode['trajectory_data']) > 2:
            # 计算轨迹的角度变化方差作为平滑性指标
            trajectory = episode['trajectory_data']
            angle_changes = []
            
            for i in range(1, len(trajectory) - 1):
                prev_pos = trajectory[i-1]
                curr_pos = trajectory[i]
                next_pos = trajectory[i+1]
                
                # 计算前后两个线段的角度
                vec1 = (curr_pos[0] - prev_pos[0], curr_pos[1] - prev_pos[1])
                vec2 = (next_pos[0] - curr_pos[0], next_pos[1] - curr_pos[1])
                
                # 计算角度变化
                if np.linalg.norm(vec1) > 0 and np.linalg.norm(vec2) > 0:
                    cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                    cos_angle = np.clip(cos_angle, -1.0, 1.0)
                    angle_change = np.arccos(cos_angle)
                    angle_changes.append(angle_change)
            
            if angle_changes:
                # 使用角度变化的标准差作为平滑性指标（越小越平滑）
                smoothness = np.std(angle_changes)
                trajectory_smoothness_values.append(smoothness)
    
    j_s_mean = np.mean(trajectory_smoothness_values) if trajectory_smoothness_values else 0.0
    j_s_std = np.std(trajectory_smoothness_values) if trajectory_smoothness_values else 0.0
    
    # 5. 能量消耗指标 (Energy Consumption, J_C)
    energy_consumption_values = []
    
    for episode in episode_data_list:
        if 'action_sequence' in episode:
            # 基于动作序列计算能量消耗
            actions = episode['action_sequence']
            total_energy = 0.0
            
            for action in actions:
                if isinstance(action, (list, np.ndarray)) and len(action) >= 2:
                    # 能量消耗 = 加速度^2 + 角速度^2 的积分
                    acceleration = action[0]
                    angular_velocity = action[1]
                    energy_step = acceleration**2 + angular_velocity**2
                    total_energy += energy_step
            
            energy_consumption_values.append(total_energy)
        else:
            # 如果没有动作序列，使用步数作为能量消耗的代理指标
            steps = episode.get('steps', 0)
            # 假设每步的平均能量消耗
            estimated_energy = steps * 1.0
            energy_consumption_values.append(estimated_energy)
    
    j_c_mean = np.mean(energy_consumption_values) if energy_consumption_values else 0.0
    j_c_std = np.std(energy_consumption_values) if energy_consumption_values else 0.0
    
    return {
        'MCR': mcr,  # 任务完成率
        'FKR': {'mean': fkr_mean, 'std': fkr_std},  # 编队保持率
        'SET': set_score,  # 成功率加权探索时间
        'J_S': {'mean': j_s_mean, 'std': j_s_std},  # 飞行轨迹平滑性
        'J_C': {'mean': j_c_mean, 'std': j_c_std},  # 能量消耗
        'success_rate': mcr,  # 成功率（与MCR相同）
        'avg_exploration_time': avg_exploration_time  # 平均探索时间
    }


def run_direct_training(args, n_agent, m_enemy, seed=42, run_id=None):
    """直接训练MASAC（无课程学习）
    
    Args:
        args: 命令行参数
        n_agent: 友方数量
        m_enemy: 敌方数量
        seed: 随机种子
        run_id: 运行ID，用于多次训练时区分不同的运行
    """
    # 设置随机种子
    set_seed(seed)
    print(f"设置随机种子: {seed}")
    if run_id is not None:
        print(f"训练运行ID: {run_id}")
    
    # 确保结果目录存在
    global RESULTS_DIR, TEST_RESULTS_BASE, TRAINING_RESULTS_FILE, RENDER, state_number, action_number, MemoryCapacity, BATCH, EP_LEN, device
    ensure_dir_exists(RESULTS_DIR)
    ensure_dir_exists(TEST_RESULTS_BASE)
    ensure_dir_exists(os.path.dirname(TRAINING_RESULTS_FILE))
    
    # 初始化历史记录列表
    alpha_history = []
    reward_history = []
    success_rate_history = []
    
    # 初始化训练记录数组
    all_ep_r = [[] for _ in range(TRAIN_NUM)]
    all_ep_r0 = [[] for _ in range(TRAIN_NUM)]
    all_ep_r1 = [[] for _ in range(TRAIN_NUM)]
    k = 0  # 使用索引0，因为TRAIN_NUM=1
    
    # 清除所有旧模型
    models_base_dir = "D:/pa/path planning2/models"
    if os.path.exists(models_base_dir):
        import shutil
        try:
            log(f"正在清除旧模型目录: {models_base_dir}", LOG_INFO)
            shutil.rmtree(models_base_dir)
            log(f"旧模型目录已清除", LOG_INFO)
        except Exception as e:
            log(f"清除旧模型目录时出错: {e}", LOG_ERROR)
    
    # 重新创建模型基础目录
    os.makedirs(models_base_dir, exist_ok=True)
    log(f"创建新的模型目录: {models_base_dir}", LOG_INFO)
    
    # 设置日志级别
    set_log_level(LOG_DEBUG)  # 从INFO改为DEBUG级别，显示更详细的日志信息
    
    # 清除之前的日志历史
    if hasattr(globals(), 'clear_log_history'):
        clear_log_history()
    
    # 确保模型保存目录存在
    log(f"模型将保存在: {models_base_dir}", LOG_INFO)
    
    # 使用最终环境的固定配置
    print("使用最终环境的固定配置进行直接训练")
    print(f"友方无人机数量: {n_agent}")
    print(f"敌方无人机数量: {m_enemy}")
    print(f"障碍物数量: {args.obstacle_count}")
    
    # 创建环境
    env = RlGame(leader_count=n_agent, follower_count=m_enemy, obstacle_num=args.obstacle_count, render=RENDER).unwrapped
    
    # 训练模式下设置dt=1.0
    env.set_time_step(1.0)
    print(f"训练模式：时间步长dt设置为1.0")
    
    # 创建MASAC控制器
    n_agents = n_agent + m_enemy
    state_dim = state_number
    action_dim = action_number
    print("初始化MASAC控制器...")
    masac_controller = MASACController(
        n_agents=n_agents, 
        state_dim=state_dim, 
        action_dim=action_dim, 
        device=device,
        memory_capacity=MemoryCapacity,
        max_replay_ratio=20  # 允许较高的重放比例以减轻过拟合
    )
    
    # 开始直接训练
    max_episodes = 800  # 直接训练的总回合数
    print(f"开始直接训练，总回合数: {max_episodes}")
    
    # 训练循环
    for episode in range(max_episodes):
        print(f"回合 {episode+1}/{max_episodes}")
        
        # 重置环境
        observation = env.reset()
        total_reward = 0
        reward_totle0 = 0 # 初始化领导者回合奖励累加器
        reward_totle1 = 0 # 初始化第一个跟随者回合奖励累加器
        done = False
        step_count = 0
        team_formation_time = 0
        last_distance = None
        
        # 在每个时间步上进行训练
        while not done and step_count < EP_LEN:
            # 根据回合数决定是否添加噪声
            should_add_noise = episode < 20
            # 选择动作
            action = masac_controller.select_actions(observation, add_noise=should_add_noise, noise_scale=0.1, evaluate=False)
            
            # 执行动作
            observation_, reward, done, win, team_counter, dis = env.step(action)
            
            # 存储经验到回放缓冲区
            masac_controller.store_transition(
                observation, action, reward, observation_, done,
                current_stage_tag="direct_training"
            )
            
            # 记录最后一步的距离
            last_distance = dis
            
            # 更新状态和统计
            observation = observation_
            # 累加当前时间步的总奖励 (Leader + Followers)
            current_step_reward = 0.0
            if isinstance(reward, dict):
                leader_r = reward.get("leader", 0.0)
                followers_r = reward.get("followers", [])
                # 确保是数值
                if isinstance(leader_r, (int, float, np.number)):
                     current_step_reward += float(leader_r)
                if isinstance(followers_r, list):
                    for r in followers_r:
                         if isinstance(r, (int, float, np.number)):
                             current_step_reward += float(r)
            total_reward += current_step_reward 
            step_count += 1
            
            # 计算编队时间
            if team_counter > 0:
                team_formation_time += 1
            
            # 渲染环境
            if RENDER:
                env.render()
            
            # 如果缓冲区足够大，开始学习
            # 使用10%的经验池大小作为开始学习的阈值
            training_start_size = int(MemoryCapacity * 0.3)
            if len(masac_controller.memory.buffer) > training_start_size:  # 使用实际的缓冲区长度
                try:
                    masac_controller.train(
                        batch_size=BATCH,
                        current_stage_tag="direct_training",
                        current_stage_number=0
                    )
                except Exception as e:
                    print(f"训练过程中出现错误: {e}")
                    import traceback
                    traceback.print_exc()
        
        # 记录回合统计
        all_ep_r[k].append(total_reward)
        
        # 计算成功率
        success = win if 'win' in locals() else False
        success_rate_history.append(1 if success else 0)
        
        # 记录训练统计
        masac_controller.track_episode_rewards([total_reward])
        masac_controller.track_episode_length(step_count)
        masac_controller.track_episode_success(success)
        
        # 每100回合保存一次模型
        if (episode + 1) % 100 == 0:
            save_dir = os.path.join(RESULTS_DIR, "models", f"direct_training_episode_{episode+1}")
            os.makedirs(save_dir, exist_ok=True)
            masac_controller.save_models(f"{save_dir}/model_ep{episode+1}")
            
            # 保存训练结果
            timestamp = get_timestamp()
            seed_suffix = f"_seed{seed}" if seed != 42 else ""
            run_suffix = f"_run{run_id}" if run_id is not None else ""
            result_filename = f"MASAC_direct_training_{timestamp}{run_suffix}{seed_suffix}_episode_{episode+1}.pkl"
            result_path = os.path.join(RESULTS_DIR, result_filename)
            
            training_results = {
                'episode': episode + 1,
                'total_reward': total_reward,
                'step_count': step_count,
                'success': success,
                'team_formation_time': team_formation_time,
                'last_distance': last_distance,
                'all_ep_r': all_ep_r,
                'success_rate_history': success_rate_history,
                'timestamp': timestamp
            }
            
            with open(result_path, 'wb') as f:
                pkl.dump(training_results, f)
            
            print(f"训练结果已保存: {result_path}")
            
            # 绘制训练曲线
            import matplotlib.pyplot as plt
            plt.figure(figsize=(12, 8))
            
            # 奖励曲线
            ax1 = plt.subplot(2, 2, 1)
            if all_ep_r[k]:
                # 计算移动平均
                window_size = 50
                rewards = all_ep_r[k]
                if len(rewards) >= window_size:
                    moving_avg = [np.mean(rewards[max(0, i-window_size):i+1]) for i in range(len(rewards))]
                    plt.plot(moving_avg, label=f'移动平均 (窗口={window_size})')
                plt.plot(rewards, alpha=0.3, label='原始奖励')
            plt.xlabel('回合')
            plt.ylabel('奖励')
            plt.title('MASAC Direct Training - Rewards')
            plt.legend()
            plt.grid(True)
            
            # 成功率曲线
            ax2 = plt.subplot(2, 2, 2)
            if success_rate_history:
                window_size = 50
                if len(success_rate_history) >= window_size:
                    moving_avg = [np.mean(success_rate_history[max(0, i-window_size):i+1]) for i in range(len(success_rate_history))]
                    plt.plot(moving_avg, label=f'移动平均 (窗口={window_size})')
                plt.plot(success_rate_history, alpha=0.3, label='原始成功率')
            plt.xlabel('回合')
            plt.ylabel('成功率')
            plt.title('Success Rate')
            plt.legend()
            plt.grid(True)
            
            # 回合长度曲线
            ax3 = plt.subplot(2, 2, 3)
            if masac_controller.episode_lengths:
                window_size = 50
                lengths = masac_controller.episode_lengths
                if len(lengths) >= window_size:
                    moving_avg = [np.mean(lengths[max(0, i-window_size):i+1]) for i in range(len(lengths))]
                    plt.plot(moving_avg, label=f'移动平均 (窗口={window_size})')
                plt.plot(lengths, alpha=0.3, label='原始长度')
            plt.xlabel('回合')
            plt.ylabel('回合长度')
            plt.title('Episode Length')
            plt.legend()
            plt.grid(True)
            
            # 训练统计
            ax4 = plt.subplot(2, 2, 4)
            stats = masac_controller.get_training_stats()
            stats_text = f"训练步数: {stats.get('train_step', 0)}\n"
            stats_text += f"平均奖励: {stats.get('avg_reward', 0):.2f}\n"
            stats_text += f"平均长度: {stats.get('avg_length', 0):.2f}\n"
            stats_text += f"成功率: {stats.get('success_rate', 0):.2f}"
            plt.text(0.1, 0.5, stats_text, transform=ax4.transAxes, fontsize=12, verticalalignment='center')
            plt.title('Training Statistics')
            plt.axis('off')
            
            plt.tight_layout()
            plot_filename = f"MASAC_direct_training_{timestamp}{run_suffix}{seed_suffix}_episode_{episode+1}.png"
            plot_path = os.path.join(RESULTS_DIR, plot_filename)
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"训练曲线已保存: {plot_path}")
        
        # 打印回合摘要
        # 处理最终距离的显示
        distance_str = "N/A"
        if last_distance is not None:
            if isinstance(last_distance, dict):
                distance_parts = []
                if "leader_to_goal" in last_distance:
                    distance_parts.append(f"goal={float(last_distance['leader_to_goal']):.2f}")

                follower_distances = [
                    float(value)
                    for key, value in last_distance.items()
                    if key.startswith("leader_to_follower_")
                ]
                if follower_distances:
                    distance_parts.append(f"follower={float(np.mean(follower_distances)):.2f}")

                if "leader_to_obstacle" in last_distance:
                    distance_parts.append(f"obstacle={float(last_distance['leader_to_obstacle']):.2f}")

                distance_str = ", ".join(distance_parts) if distance_parts else "N/A"
            elif isinstance(last_distance, (list, tuple, np.ndarray)):
                # 如果是列表、元组或数组，取最小值
                distance_str = f"{min(last_distance):.2f}"
            else:
                # 如果是标量，直接显示
                distance_str = f"{last_distance:.2f}"
        
        print(f"回合摘要 - 回合: {episode+1}/{max_episodes}, "
              f"总奖励: {total_reward:.2f}, "
              f"步数: {step_count}, "
              f"成功: {success}, "
              f"编队时间: {team_formation_time}, "
              f"最终距离: {distance_str}")
    
    # 训练完成，保存最终模型
    final_model_dir = os.path.join(RESULTS_DIR, "models", "direct_training_final")
    os.makedirs(final_model_dir, exist_ok=True)
    masac_controller.save_models(f"{final_model_dir}/final_model")
    
    # 保存最终训练结果
    timestamp = get_timestamp()
    seed_suffix = f"_seed{seed}" if seed != 42 else ""
    run_suffix = f"_run{run_id}" if run_id is not None else ""
    final_result_filename = f"MASAC_direct_training_{timestamp}{run_suffix}{seed_suffix}_final.pkl"
    final_result_path = os.path.join(RESULTS_DIR, final_result_filename)
    
    final_training_results = {
        'total_episodes': max_episodes,
        'all_ep_r': all_ep_r,
        'success_rate_history': success_rate_history,
        'final_success_rate': np.mean(success_rate_history[-100:]) if len(success_rate_history) >= 100 else np.mean(success_rate_history),
        'final_avg_reward': np.mean(all_ep_r[k][-100:]) if len(all_ep_r[k]) >= 100 else np.mean(all_ep_r[k]),
        'timestamp': timestamp,
        'seed': seed,
        'run_id': run_id,
        'training_config': {
            'n_agent': n_agent,
            'm_enemy': m_enemy,
            'obstacle_count': args.obstacle_count,
            'max_episodes': max_episodes,
            'seed': seed,
            'run_id': run_id
        }
    }
    
    with open(final_result_path, 'wb') as f:
        pkl.dump(final_training_results, f)
    
    print(f"最终训练结果已保存: {final_result_path}")
    print(f"最终模型已保存: {final_model_dir}/final_model")
    print("直接训练完成！")
    
    # 返回训练结果用于多次训练汇总
    return final_training_results


def run_multi_seed_training(args, n_agent, m_enemy):
    """多次训练模式，使用不同随机种子进行多次训练
    
    Args:
        args: 命令行参数
        n_agent: 友方数量
        m_enemy: 敌方数量
    """
    print("=== 多次训练模式 ===")
    
    # 解析种子列表
    if args.seeds:
        try:
            seeds = [int(s.strip()) for s in args.seeds.split(',')]
            print(f"使用自定义种子: {seeds}")
        except ValueError:
            print("种子格式错误，使用默认种子")
            seeds = [42, 123, 456][:args.num_runs]
    else:
        # 自动生成种子
        import random
        random.seed(42)  # 确保可重现
        seeds = [random.randint(1, 10000) for _ in range(args.num_runs)]
        print(f"自动生成种子: {seeds}")
    
    # 确保种子数量与运行次数匹配
    if len(seeds) != args.num_runs:
        print(f"警告: 种子数量({len(seeds)})与运行次数({args.num_runs})不匹配")
        if len(seeds) < args.num_runs:
            # 补充种子
            import random
            random.seed(seeds[-1] if seeds else 42)
            while len(seeds) < args.num_runs:
                seeds.append(random.randint(1, 10000))
        else:
            # 截取种子
            seeds = seeds[:args.num_runs]
        print(f"调整后的种子: {seeds}")
    
    all_results = []
    
    print(f"开始进行 {args.num_runs} 次训练")
    for i, seed in enumerate(seeds):
        print(f"\n{'='*60}")
        print(f"第 {i+1}/{args.num_runs} 次训练 - 种子: {seed}")
        print(f"{'='*60}")
        
        # 运行单次训练
        result = run_direct_training(args, n_agent, m_enemy, seed=seed, run_id=i+1)
        all_results.append(result)
        
        print(f"第 {i+1} 次训练完成")
        if result:
            final_reward = result.get('final_avg_reward', 'N/A')
            final_success = result.get('final_success_rate', 'N/A')
            print(f"最终平均奖励: {final_reward}")
            print(f"最终成功率: {final_success}")
    
    # 汇总所有运行的结果
    print(f"\n{'='*60}")
    print("多次训练汇总")
    print(f"{'='*60}")
    
    # 计算统计信息
    final_rewards = [r.get('final_avg_reward', 0) for r in all_results if r]
    final_success_rates = [r.get('final_success_rate', 0) for r in all_results if r]
    
    if final_rewards:
        reward_mean = np.mean(final_rewards)
        reward_std = np.std(final_rewards)
        reward_95_ci = 1.96 * reward_std / np.sqrt(len(final_rewards))
        
        print(f"最终平均奖励: {reward_mean:.3f} ± {reward_std:.3f}")
        print(f"95% 置信区间: [{reward_mean - reward_95_ci:.3f}, {reward_mean + reward_95_ci:.3f}]")
    
    if final_success_rates:
        success_mean = np.mean(final_success_rates)
        success_std = np.std(final_success_rates)
        success_95_ci = 1.96 * success_std / np.sqrt(len(final_success_rates))
        
        print(f"最终成功率: {success_mean:.3f} ± {success_std:.3f}")
        print(f"95% 置信区间: [{success_mean - success_95_ci:.3f}, {success_mean + success_95_ci:.3f}]")
    
    # 保存汇总结果
    timestamp = get_timestamp()
    summary_filename = f"MASAC_multi_run_summary_{timestamp}.pkl"
    summary_path = os.path.join(RESULTS_DIR, summary_filename)
    
    summary_results = {
        'num_runs': args.num_runs,
        'seeds': seeds,
        'individual_results': all_results,
        'summary_stats': {
            'final_rewards': {
                'mean': reward_mean if final_rewards else 0,
                'std': reward_std if final_rewards else 0,
                'ci_95': reward_95_ci if final_rewards else 0,
                'values': final_rewards
            },
            'final_success_rates': {
                'mean': success_mean if final_success_rates else 0,
                'std': success_std if final_success_rates else 0,
                'ci_95': success_95_ci if final_success_rates else 0,
                'values': final_success_rates
            }
        },
        'timestamp': timestamp,
        'config': {
            'n_agent': n_agent,
            'm_enemy': m_enemy,
            'obstacle_count': args.obstacle_count,
            'seeds': seeds,
            'num_runs': args.num_runs
        }
    }
    
    with open(summary_path, 'wb') as f:
        pkl.dump(summary_results, f)
    
    print(f"\n多次训练汇总结果已保存: {summary_path}")
    print("多次训练完成！")
    
    return summary_results


    # 添加奖励分布分析函数
    def analyze_reward_distribution(reward, n_friendly_agents, episode, timestep):
        """分析奖励分布情况，记录主机和从机奖励的比例和统计特性
        
        Args:
            reward: 环境返回的奖励字典 {"leader": r_l, "followers": [r_f1, ...]}
            n_friendly_agents: 友方智能体数量
            episode: 当前回合数
            timestep: 当前时间步
            
        Returns:
            dict or None: 奖励分布统计信息，如果输入无效则返回None
        """
        # 强制要求输入为结构化字典
        if not (isinstance(reward, dict) and "leader" in reward and "followers" in reward):
            log(f"[错误] analyze_reward_distribution 期望结构化奖励字典，收到: {type(reward)}", LOG_ERROR)
            return None
        
        leader_reward = reward.get("leader", 0.0)
        follower_rewards_raw = reward.get("followers", []) # 使用新变量名
        
        # 确保提取的值是数值
        all_reward_values = []
        valid_leader = False
        if isinstance(leader_reward, (int, float, np.number)):
            all_reward_values.append(float(leader_reward))
            valid_leader = True
        else:
             log(f"[警告] Leader 奖励不是有效数值: {leader_reward}", LOG_WARNING)
             leader_reward = 0.0 # 使用默认值
             
        valid_follower_rewards = []
        if isinstance(follower_rewards_raw, list):
            for r in follower_rewards_raw:
                if isinstance(r, (int, float, np.number)):
                    all_reward_values.append(float(r))
                    valid_follower_rewards.append(float(r))
                else:
                    log(f"[警告] Follower 奖励包含无效数值: {r}", LOG_WARNING)
        else:
             log(f"[警告] Followers 奖励不是列表: {follower_rewards_raw}", LOG_WARNING)
             # follower_rewards_raw = [] # 这里不需要重置，因为后面会检查 all_reward_values

        # 如果没有有效的奖励值
        if not all_reward_values:
            log("[警告] analyze_reward_distribution 未找到有效奖励值", LOG_WARNING)
            # 返回一个带有默认值的字典可能比返回None更好
            return {
                'episode': episode,
                'timestep': timestep,
                'n_friendly': n_friendly_agents,
                'total_agents_rewarded': 0,
                'reward_mean': 0.0, 'reward_std': 0.0, 'reward_min': 0.0, 'reward_max': 0.0,
                'friendly_mean': 0.0, 'friendly_std': 0.0
            }

        # 计算整体统计数据
        reward_values_np = np.array(all_reward_values)
        stats = {
            'episode': episode,
            'timestep': timestep,
            'n_friendly': n_friendly_agents,
            'total_agents_rewarded': len(all_reward_values), # 实际收到奖励的智能体数
            'reward_mean': float(np.mean(reward_values_np)),
            'reward_std': float(np.std(reward_values_np)),
            'reward_min': float(np.min(reward_values_np)),
            'reward_max': float(np.max(reward_values_np)),
        }
        
        # 计算友方（Leader + Followers）统计信息
        # 假设 all_reward_values 只包含友方奖励 (Leader + Followers)
        stats['friendly_mean'] = float(np.mean(reward_values_np))
        stats['friendly_std'] = float(np.std(reward_values_np))

        # 记录具体的 Leader 和 Follower 奖励
        if valid_leader:
            stats['leader_reward'] = float(leader_reward)
        if valid_follower_rewards: # 使用处理后的有效奖励列表
            stats['follower_rewards'] = valid_follower_rewards
            
        # 打印主机和从机奖励信息 (保持不变)
        if episode % 50 == 0 and timestep == 0: 
            log(f"奖励分布分析 (EP {episode}, 步 {timestep}):", LOG_INFO)
            if 'leader_reward' in stats:
                log(f"  领导者奖励: {stats['leader_reward']:.2f}", LOG_INFO)
            if 'follower_rewards' in stats:
                log(f"  跟随者奖励: {stats['follower_rewards']}", LOG_INFO)
            if 'friendly_mean' in stats:
                log(f"  友方平均/标准差: {stats['friendly_mean']:.2f}/{stats['friendly_std']:.2f}", LOG_INFO)
        
        return stats
    
    # 在外层循环检查是否需要跳出
    break_outer = False
    for curriculum_step in range(curriculum_manager.max_curriculum_steps):
        print(f"\n课程步骤 {curriculum_step+1}/{curriculum_manager.max_curriculum_steps}")
        
        # 添加这个检查，确保当前任务有效
        if curriculum_manager.get_current_task() is None:
            print("没有更多有效任务，课程学习已完成！")
            break
        
        # 获取当前任务的阶段标签和编号
        current_task = curriculum_manager.get_current_task()
        current_task_id = current_task.id if current_task and hasattr(current_task, 'id') else f"stage_{curriculum_step}"
        current_stage_tag_for_memory = current_task_id
        current_stage_number_for_memory = curriculum_step  # 使用循环索引作为阶段编号
        
        log(f"当前课程阶段: '{current_stage_tag_for_memory}' (编号 {current_stage_number_for_memory})", LOG_INFO)
        
        # 在每个任务上进行训练
        for episode in range(curriculum_manager.max_episodes_per_task):
            print(f"任务 {episode+1}/{curriculum_manager.max_episodes_per_task}")
            
            # 重置环境
            observation = env.reset()
            total_reward = 0
            reward_totle0 = 0 # 初始化领导者回合奖励累加器
            reward_totle1 = 0 # 初始化第一个跟随者回合奖励累加器
            done = False
            step_count = 0
            team_formation_time = 0
            last_distance = None
            
            # 在每个时间步上进行训练
            while not done and step_count < EP_LEN:
                # 根据回合数决定是否添加噪声
                should_add_noise = episode < 20
                # 选择动作
                action = masac_controller.select_actions(observation, add_noise=should_add_noise, noise_scale=0.1, evaluate=False)
                
                # 执行动作
                observation_, reward, done, win, team_counter, dis = env.step(action)
                
                # 存储经验到回放缓冲区，传递阶段标签
                masac_controller.store_transition(
                    observation, action, reward, observation_, done,
                    current_stage_tag=current_stage_tag_for_memory
                )
                
                # 记录最后一步的距离
                last_distance = dis
                
                # 更新状态和统计
                observation = observation_
                # 累加当前时间步的总奖励 (Leader + Followers)
                current_step_reward = 0.0
                if isinstance(reward, dict):
                    leader_r = reward.get("leader", 0.0)
                    followers_r = reward.get("followers", [])
                    # 确保是数值
                    if isinstance(leader_r, (int, float, np.number)):
                         current_step_reward += float(leader_r)
                    if isinstance(followers_r, list):
                        for r in followers_r:
                             if isinstance(r, (int, float, np.number)):
                                 current_step_reward += float(r)
                total_reward += current_step_reward 
                step_count += 1
                
                # 计算编队时间
                if team_counter > 0:
                    team_formation_time += 1
                
                # 渲染环境
                if RENDER:
                    env.render()
                
                # 如果缓冲区足够大，开始学习
                # 使用10%的经验池大小作为开始学习的阈值
                training_start_size = int(MemoryCapacity * 0.3)
                if len(masac_controller.memory.buffer) > training_start_size:  # 使用实际的缓冲区长度
                    try:
                        masac_controller.train(
                            batch_size=BATCH,
                            current_stage_tag=current_stage_tag_for_memory,
                            current_stage_number=current_stage_number_for_memory
                        )
                    except Exception as e:
                        print(f"训练过程中出现错误: {e}")
                        import traceback
                        traceback.print_exc()
                
                # 更新状态和统计 - 这部分 observation = observation_ 已在上文处理
                # observation = observation_
                
                # 分别计算各个智能体的奖励 - 原 UnboundLocalError 行已删除
                # reward_totle += reward.mean() if isinstance(reward, np.ndarray) else reward
                
                # 确保安全地获取奖励值并累加到 reward_totle0 (Leader) 和 reward_totle1 (First Follower)
                if isinstance(reward, dict):
                    # Leader 奖励
                    leader_r_val = reward.get("leader")
                    if isinstance(leader_r_val, (int, float, np.number)):
                        reward_totle0 += float(leader_r_val)

                    # First Follower 奖励
                    followers_r_list = reward.get("followers")
                    if isinstance(followers_r_list, list) and len(followers_r_list) > 0:
                        first_follower_r_val = followers_r_list[0]
                        if isinstance(first_follower_r_val, (int, float, np.number)):
                            reward_totle1 += float(first_follower_r_val)
                
                # 渲染环境
                if RENDER:
                    env.render()
                
                # 结束判断
                if done:
                    # 显示回合结束状态
                    if win:
                        print(f"回合 {episode+1} 成功! 智能体到达目标!")
                    else:
                        print(f"回合 {episode+1} 失败")
                    break
            
            # 收集温度系数 Alpha 统计数据
            alpha_stats = []
            if hasattr(masac_controller, 'entroy_leader') and n_agents >= 1:
                alpha_stats.append(masac_controller.entroy_leader.get_alpha_stats())
            if hasattr(masac_controller, 'entroy_follower') and n_agents > 1:
                # 假设所有follower共享一个entroy对象
                # 如果需要每个follower独立，需要修改MASACController
                alpha_stats.append(masac_controller.entroy_follower.get_alpha_stats())
            
            avg_alpha = np.mean([stat["current"] for stat in alpha_stats]) if alpha_stats else 0.0
            alpha_history.append(avg_alpha)
            
            # 获取并打印各个飞机的速度和编队率
            leader_speeds = [f"{leader.speed:.1f}" for leader in env.entity_manager.leaders]
            follower_speeds = [f"{follower.speed:.1f}" for follower in env.entity_manager.followers]
            formation_rate = env.entity_manager.get_formation_rate()
            
            # 创建综合信息并使用log函数打印
            if episode % 10 == 0 or win:
                status = "成功" if win else "进行中"
                # 整合所有回合信息到一个日志消息中
                log_message = (
                    f"回合摘要 - 任务: {curriculum_step+1}/{curriculum_manager.max_curriculum_steps}, "
                    f"回合: {episode+1}/{curriculum_manager.max_episodes_per_task}, "
                    f"总回合: {curriculum_manager.total_episodes}, "
                    f"状态: {status}, 步数: {step_count}, 总奖励: {total_reward:.1f}, " # 在这里添加了步数
                    f"主机速度: [{', '.join(leader_speeds)}], "
                    f"从机速度: [{', '.join(follower_speeds)}], "
                    f"编队率: {formation_rate:.2f}, "
                    f"Alpha: {avg_alpha:.4f}, 重放比例: {masac_controller.replay_ratio}"
                )
                log(log_message, LOG_INFO)
            
            # 记录奖励
            all_ep_r[k].append(total_reward) # 使用 total_reward 记录总奖励
            all_ep_r0[k].append(reward_totle0)
            all_ep_r1[k].append(reward_totle1)
            
            # 记录历史数据
            reward_history.append(total_reward)
            success_rate_history.append(float(win))
            
            # 更新任务性能
            metrics = {
                'reward': total_reward,
                'success_rate': float(win),
                'team_coordination': team_counter  # 编队保持率
            }
            # 检查返回值，如果为True则说明达到最大总回合数，需终止训练
            if curriculum_manager.update_task_performance(metrics):
                print("达到总回合数限制，终止训练！")
                # 保存最终模型
                final_save_dir = f"{models_base_dir}/final"
                os.makedirs(final_save_dir, exist_ok=True)
                final_save_path = f"{final_save_dir}/final_model"
                try:
                    masac_controller.save_models(final_save_path)
                    print(f"最终模型已保存到: {final_save_path}")
                except Exception as e:
                    print(f"保存最终模型失败: {e}")
                    traceback.print_exc()
                # 直接跳出两层循环
                break_outer = True
                break
            
            # 检查是否需要切换任务
            if curriculum_manager.should_switch_task():
                log(f"任务切换，完成 {episode+1} 轮训练", LOG_INFO)
                
                # 保存任务完成时的模型
                task_complete_dir = f"{models_base_dir}/curriculum_step{curriculum_step}_complete"
                os.makedirs(task_complete_dir, exist_ok=True)
                task_complete_path = f"{task_complete_dir}/model_ep{episode}"
                try:
                    masac_controller.save_models(task_complete_path)
                    log(f"模型已保存到: {task_complete_path}", LOG_INFO)
                except Exception as e:
                    log(f"保存模型失败: {e}", LOG_ERROR)
                    traceback.print_exc()
                break
            
            # 定期保存模型 - 改为每100回合保存一次
            if episode % 100 == 0 and episode > 0:
                # 创建专门的保存目录
                save_dir = os.path.join(RESULTS_DIR, "models", f"curriculum_step{curriculum_step}")
                os.makedirs(save_dir, exist_ok=True)
                
                # 构建完整的保存路径
                save_path = f"{save_dir}/model_ep{episode}"
                try:
                    masac_controller.save_models(save_path)
                    print(f"模型已保存到: {save_path}")
                except Exception as e:
                    print(f"保存模型失败: {e}")
                    traceback.print_exc()
        
        # 在外层循环检查是否需要跳出
        if break_outer:
            print("由于达到最大总回合数限制，终止整个课程训练")
            break
                
        # 准备参数以进行知识迁移
        current_policy_and_critic_params = masac_controller.get_policy_parameters_for_curriculum()

        # 定义一个临时的包装器类，仅用于传递参数给 PolicyTransfer
        class TempPolicyWrapperForCurriculum:
            def __init__(self, params):
                self._params = params
            def get_parameters(self):
                return self._params

        policy_wrapper_for_transfer = TempPolicyWrapperForCurriculum(current_policy_and_critic_params)

        # 获取下一个任务和迁移后的参数字典
        # 假设 curriculum_manager.get_next_task 内部的 PolicyTransfer.transfer 返回参数字典
        next_task, transferred_params_dict = curriculum_manager.get_next_task(policy_wrapper_for_transfer)

        if next_task is None:
            print("没有更多任务，课程完成!")
            # ... (保存最终模型的逻辑不变) ...
            final_save_dir = f"{models_base_dir}/final"
            os.makedirs(final_save_dir, exist_ok=True)
            final_save_path = f"{final_save_dir}/final_model"
            try:
                masac_controller.save_models(final_save_path) # 使用 save_models
                print(f"最终模型已保存到: {final_save_path}")
            except Exception as e:
                print(f"保存最终模型失败: {e}")
                traceback.print_exc()
                
            # 保存最后一个任务的训练结果
            all_ep_r_mean = np.mean((np.array(all_ep_r)), axis=0)
            all_ep_r_std = np.std((np.array(all_ep_r)), axis=0)
            all_ep_L_mean = np.mean((np.array(all_ep_r0)), axis=0)
            all_ep_L_std = np.std((np.array(all_ep_r0)), axis=0)
            all_ep_F_mean = np.mean((np.array(all_ep_r1)), axis=0)
            all_ep_F_std = np.std((np.array(all_ep_r1)), axis=0)
            
            # 保存训练结果
            d = {
                "all_ep_r_mean": all_ep_r_mean, 
                "all_ep_r_std": all_ep_r_std,
                "all_ep_L_mean": all_ep_L_mean, 
                "all_ep_L_std": all_ep_L_std,
                "all_ep_F_mean": all_ep_F_mean, 
                "all_ep_F_std": all_ep_F_std,
                "alpha_history": alpha_history,
                "reward_history": reward_history,
                "success_rate_history": success_rate_history
            }
            
            # 创建唯一的结果文件名（添加时间戳和课程步骤）
            timestamp = get_timestamp()
            # 确保最后一个任务使用正确的步骤编号（curriculum_step + 1，而不是 curriculum_step）
            result_filename = f"MASAC_curriculum_{timestamp}_step{curriculum_step+1}.pkl"
            result_path = os.path.join(RESULTS_DIR, result_filename)
            
            # 确保结果目录存在
            ensure_dir_exists(RESULTS_DIR)
            log(f"保存最终任务训练结果到 {result_path}", LOG_INFO)
            
            try:
                with open(result_path, 'wb') as f:
                    pkl.dump(d, f, pkl.HIGHEST_PROTOCOL)
                log(f"最终任务训练结果已成功保存", LOG_INFO)
            except Exception as e:
                log(f"保存最终任务训练结果时出错: {e}", LOG_ERROR)
                import traceback
                traceback.print_exc()
            
            # 绘制训练曲线
            import matplotlib.pyplot as plt
            
            # 创建一个包含两个子图的图形
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
            
            # 获取实际有数据的回合数
            actual_episodes = len(all_ep_r_mean)
            print(f"实际训练回合数: {actual_episodes}")
            
            # 绘制第一个子图：奖励曲线
            x_range = np.arange(actual_episodes)
            ax1.plot(x_range, all_ep_r_mean[:actual_episodes], label='Average Reward')
            ax1.fill_between(x_range,
                             all_ep_r_mean[:actual_episodes] - all_ep_r_std[:actual_episodes], 
                             all_ep_r_mean[:actual_episodes] + all_ep_r_std[:actual_episodes],
                             alpha=0.1, color='blue')
            
            ax1.set_title('MASAC with Curriculum Learning - Rewards')
            ax1.set_ylabel('Moving averaged episode reward')
            ax1.legend()
            ax1.grid(True, linestyle='--', alpha=0.7)
            
            # 绘制第二个子图：温度系数Alpha曲线
            if alpha_history:
                alpha_x = np.arange(len(alpha_history))
                ax2.plot(alpha_x, alpha_history, color='green', label='Alpha')
                ax2.set_title('Temperature Coefficient Alpha')
                ax2.set_xlabel('Episode')
                ax2.set_ylabel('Alpha value')
                ax2.legend()
                ax2.grid(True, linestyle='--', alpha=0.7)
            
            # 调整布局
            plt.tight_layout()
            
            # 保存图表 - 使用与训练结果数据文件匹配的文件名（不含扩展名）
            plot_filename = f"MASAC_curriculum_{timestamp}_step{curriculum_step+1}.png"
            plot_path = os.path.join(RESULTS_DIR, plot_filename)
            
            try:
                plt.savefig(plot_path)
                log(f"最终任务训练曲线已保存到: {plot_path}", LOG_INFO)
            except Exception as e:
                log(f"保存最终任务训练曲线时出错: {e}", LOG_ERROR)
                import traceback
                traceback.print_exc()
            finally:
                plt.close()  # 确保图表资源被释放
                
            break # 跳出 curriculum_step 循环
        
        # 使用迁移后的参数更新控制器
        if transferred_params_dict is not None:
            # PolicyTransfer._do_transfer 现在返回参数字典
            actual_params_to_pass = transferred_params_dict
            
            # 如果是包装器类型（兼容旧版本）
            if isinstance(transferred_params_dict, TempPolicyWrapperForCurriculum):
                log("检测到 TempPolicyWrapperForCurriculum，正在提取内部参数进行迁移。", LOG_DEBUG)
                actual_params_to_pass = transferred_params_dict.get_parameters()
            elif isinstance(transferred_params_dict, dict):
                # 这是预期的返回类型 - 参数字典
                log("知识迁移返回了参数字典（正确的行为）", LOG_INFO)
                # 检查是否有智能体数量信息
                if 'agent_counts' in transferred_params_dict:
                    agent_counts = transferred_params_dict.get('agent_counts', {})
                    log(f"智能体数量变化: {agent_counts.get('source', 'N/A')} -> {agent_counts.get('target', 'N/A')}", LOG_DEBUG)
            else:
                # 如果不是包装器也不是字典，记录警告
                log(f"警告: 知识迁移返回的参数类型为 {type(transferred_params_dict)}，期望是 dict。继续尝试，但可能导致错误。", LOG_WARNING)
                
            # 使用正确的参数调用 update_components_from_transfer
            try:
                masac_controller.update_components_from_transfer(actual_params_to_pass)
                log("已成功从迁移的参数更新 MASACController 组件。", LOG_INFO)
            except Exception as e:
                log(f"从迁移参数更新组件时出错: {e}", LOG_ERROR)
                import traceback
                traceback.print_exc()
        else:
            log("知识迁移未返回有效参数，控制器组件保持不变。", LOG_INFO)
            
        # 如果智能体数量变化，控制器进行适应
        # 使用 masac_controller 内部的 n_agents 计数进行比较和更新
        new_agent_count = next_task.env_params.get("leader_count", 1) + next_task.env_params.get("follower_count", 0)
        current_agent_count = masac_controller.n_agents # 获取控制器当前的 n_agents
        if new_agent_count != current_agent_count:
            log(f"适应新的智能体数量: {current_agent_count} -> {new_agent_count}", LOG_INFO)
            masac_controller.adapt_to_agent_count(new_agent_count)
            # n_agents 变量（如果之前在 run_with_curriculum 作用域中使用）也应更新
            n_agents = new_agent_count
    
        # 更新环境到下一个任务
        env.close() # 关闭旧环境
        env = next_task.create_env()
        env.set_time_step(1.0) # 确保新环境时间步正确
        print_task_details(next_task, f"切换到新任务 (课程步骤 {curriculum_step + 2})")
            
        # 训练完成，记录结果
        all_ep_r_mean = np.mean((np.array(all_ep_r)), axis=0)
        all_ep_r_std = np.std((np.array(all_ep_r)), axis=0)
        all_ep_L_mean = np.mean((np.array(all_ep_r0)), axis=0)
        all_ep_L_std = np.std((np.array(all_ep_r0)), axis=0)
        all_ep_F_mean = np.mean((np.array(all_ep_r1)), axis=0)
        all_ep_F_std = np.std((np.array(all_ep_r1)), axis=0)
        
        # 保存训练结果
        d = {
            "all_ep_r_mean": all_ep_r_mean, 
            "all_ep_r_std": all_ep_r_std,
            "all_ep_L_mean": all_ep_L_mean, 
            "all_ep_L_std": all_ep_L_std,
            "all_ep_F_mean": all_ep_F_mean, 
            "all_ep_F_std": all_ep_F_std,
            "alpha_history": alpha_history,
            "reward_history": reward_history,
            "success_rate_history": success_rate_history
        }
        
        # 创建唯一的结果文件名（添加时间戳和课程步骤）
        timestamp = get_timestamp()
        result_filename = f"MASAC_curriculum_{timestamp}_step{curriculum_step+1}.pkl"
        result_path = os.path.join(RESULTS_DIR, result_filename)
        
        # 确保结果目录存在
        ensure_dir_exists(RESULTS_DIR)
        log(f"保存训练结果到 {result_path}", LOG_INFO)
        
        try:
            with open(result_path, 'wb') as f:
                pkl.dump(d, f, pkl.HIGHEST_PROTOCOL)
            log(f"训练结果已成功保存", LOG_INFO)
        except Exception as e:
            log(f"保存训练结果时出错: {e}", LOG_ERROR)
            import traceback
            traceback.print_exc()
        
        # 绘制训练曲线
        import matplotlib.pyplot as plt
        
        # 创建一个包含两个子图的图形
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # 获取实际有数据的回合数
        actual_episodes = len(all_ep_r_mean)
        print(f"实际训练回合数: {actual_episodes}")
        
        # 绘制第一个子图：奖励曲线
        x_range = np.arange(actual_episodes)
        ax1.plot(x_range, all_ep_r_mean[:actual_episodes], label='Average Reward')
        ax1.fill_between(x_range,
                         all_ep_r_mean[:actual_episodes] - all_ep_r_std[:actual_episodes], 
                         all_ep_r_mean[:actual_episodes] + all_ep_r_std[:actual_episodes],
                         alpha=0.1, color='blue')
        
        ax1.set_title('MASAC with Curriculum Learning - Rewards')
        ax1.set_ylabel('Moving averaged episode reward')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # 绘制第二个子图：温度系数Alpha曲线
        if alpha_history:
            alpha_x = np.arange(len(alpha_history))
            ax2.plot(alpha_x, alpha_history, color='green', label='Alpha')
            ax2.set_title('Temperature Coefficient Alpha')
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Alpha value')
            ax2.legend()
            ax2.grid(True, linestyle='--', alpha=0.7)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图表 - 使用与训练结果数据文件匹配的文件名（不含扩展名）
        plot_filename = f"MASAC_curriculum_{timestamp}_step{curriculum_step+1}.png"
        plot_path = os.path.join(RESULTS_DIR, plot_filename)
        
        try:
            plt.savefig(plot_path)
            log(f"训练曲线已保存到: {plot_path}", LOG_INFO)
        except Exception as e:
            log(f"保存训练曲线时出错: {e}", LOG_ERROR)
            import traceback
            traceback.print_exc()
        finally:
            plt.close()  # 确保图表资源被释放
            
        # 分析课程学习过程
        analyze_curriculum_learning(curriculum_manager)
      
    # 在for curriculum_step循环结束后添加
    print("\n=========================================")
    print("课程学习训练总结:")
    print(f"- 完成的课程步骤: {curriculum_step+1}/{curriculum_manager.max_curriculum_steps}")
    print(f"- 总训练回合数: {curriculum_manager.total_episodes}")
    print(f"- 解决的任务数量: {len(curriculum_manager.task_history)}/{len(curriculum_manager.task_generator.predefined_task_configs) if hasattr(curriculum_manager.task_generator, 'predefined_task_configs') else 'N/A'}")
    print("=========================================\n")
    
    # 训练完成，记录结果
    all_ep_r_mean = np.mean((np.array(all_ep_r)), axis=0)



def run_monte_carlo_test(model_path, test_episodes=None, test_options=None, collect_formation_data=True):
    """运行蒙特卡洛测试以评估模型性能
    
    Args:
        model_path: 要测试的模型路径
        test_episodes: 测试回合数，如果为None则使用全局变量TEST_EPIOSDE
        test_options: 测试选项字典，包含如障碍物数量、无人机速度等配置
        collect_formation_data: 是否收集详细的编队数据，默认为True
        
    Returns:
        测试结果统计字典，包含五项性能指标
    """
    import numpy as np  # 移到函数开始处避免UnboundLocalError
    global RENDER, N_Agent, M_Enemy, action_number, TEST_EPIOSDE, state_number, device
    
    if test_episodes is None:
        test_episodes = TEST_EPIOSDE
    
    if test_options is None:
        test_options = {}
        
    # 从测试选项中提取环境参数
    n_agent = test_options.get('hero_count', N_Agent)  # 使用传入参数，回退到全局变量
    m_enemy = test_options.get('enemy_count', M_Enemy)  # 使用传入参数，回退到全局变量
    obstacle_count = test_options.get('obstacle_count', 1)
    uav_speed = test_options.get('uav_speed', None)
        
    print(f"开始蒙特卡洛测试，测试回合数: {test_episodes}")
    print(f"加载模型: {model_path}")
    print(f"测试配置: 友方={n_agent}, 敌方={m_enemy}, 障碍物={obstacle_count}")
    if uav_speed:
        print(f"无人机速度: {uav_speed}")
    if collect_formation_data:
        print("启用详细编队数据收集")
    
    # 创建固定目标位置的预定义位置字典
    predefined_positions = {
        "goals": [(500, 200)]  # 与训练时相同的固定目标位置
    }
    print(f"使用固定目标位置: (500, 200)")
    
    # 创建环境，添加预定义位置参数
    env = RlGame(leader_count=n_agent, follower_count=m_enemy, obstacle_num=obstacle_count, render=RENDER, 
                predefined_positions=predefined_positions).unwrapped
    
    # 测试模式下设置dt=1.0，与训练模式保持一致
    env.set_time_step(0.1)
    print(f"测试模式：时间步长dt设置为1")
    # 测试阶段启用 Leader safety shield
    if hasattr(env, "entity_manager"):
        env.entity_manager.enable_leader_safety_shield = True
        env.entity_manager.leader_safe_distance = 32.0
        env.entity_manager.leader_warning_distance = 95.0
        env.entity_manager.leader_safety_horizon = 4
        print("[Test] Leader safety shield enabled.")
    
    # 创建MASAC控制器 (用于运行测试)
    # 这里主要是模仿训练环境创建起基本的控制器结构，以便加载模型
    n_agents_total = n_agent + m_enemy  # 总智能体数量 = 友方 + 敌方
    state_dim = state_number
    action_dim = action_number
    masac_controller = MASACController(n_agents_total, state_dim, action_dim, device=device)
    masac_controller.load_models(model_path, strict=False)
    
    # 初始化统计数据
    win_count = 0
    rewards = []
    steps = []
    formation_rates = []
    distance_metrics = {
        "leader_to_goal": [],
        "leader_to_follower": [],
        "leader_to_obstacle": [],
    }
    
    # 初始化五项指标所需的数据收集列表
    episode_data_list = []
    
    # 初始化编队数据收集结构
    formation_data = {
        'test_info': {
            'test_id': get_timestamp(),
            'timestamp': time.time(),
            'total_episodes': test_episodes,
            'agents_config': {
                'leader_count': N_Agent,
                'follower_count': M_Enemy,
                'obstacle_count': obstacle_count,
                'uav_speed': uav_speed
            }
        },
        'episodes': [],
        'summary_stats': {}
    } if collect_formation_data else None
    
    # 运行测试
    for episode in range(test_episodes):
        observation = env.reset()
        total_reward = 0
        done = False
        step_count = 0
        team_formation_time = 0
        last_distance = None
        
        # 初始化当前回合的数据收集
        episode_formation_data = {
            'episode_id': episode,
            'timesteps': [],
            'episode_summary': {}
        } if collect_formation_data else None
        
        # 初始化当前回合的五项指标数据收集
        episode_trajectory = []  # 轨迹数据
        episode_actions = []     # 动作序列
        
        while not done and step_count < EP_LEN:
            # 收集当前时间步的编队状态数据
            if collect_formation_data:
                timestep_state = env.get_formation_state()
                episode_formation_data['timesteps'].append(timestep_state)
            
            # 收集轨迹数据（主机位置）
            if env.entity_manager.leaders:
                leader = env.entity_manager.leaders[0]
                episode_trajectory.append((leader.pos_x, leader.pos_y))
            
            # 选择动作（无噪声且使用确定性策略）
            action = masac_controller.select_actions(observation, add_noise=False, evaluate=True)
            
            # 收集动作数据
            if isinstance(action, np.ndarray) and len(action) >= 2:
                episode_actions.append(action[:2])  # 只记录前两个动作分量
            
            # 执行动作
            observation_, reward, done, win, team_counter, dis = env.step(action)
            
            # 记录最后一步的距离
            last_distance = dis
            
            # 更新状态和统计
           # 更新状态和统计
            observation = observation_
            
# 累加当前时间步的总奖励
            current_step_reward = 0.0
            if isinstance(reward, dict):
                leader_r = reward.get("leader", 0.0)      # 安全获取leader奖励，默认为0.0
                followers_r = reward.get("followers", []) # 安全获取followers奖励列表，默认为空列表

    # 累加 Leader 奖励 (确保是数字)
                if isinstance(leader_r, (int, float, np.number)):
                    current_step_reward += float(leader_r)

    # 累加 Followers 奖励 (确保列表中的元素是数字)
                if isinstance(followers_r, list):
                    for r in followers_r:
                        if isinstance(r, (int, float, np.number)):
                            current_step_reward += float(r)
            elif isinstance(reward, (np.ndarray, list)): # 兼容旧格式
                try:
                    current_step_reward = np.mean(reward)
                except:
        # 忽略无法处理的旧格式
                    pass
            elif isinstance(reward, (int, float, np.number)): # 单个数值奖励
                current_step_reward = float(reward)

            total_reward += current_step_reward
            step_count += 1
            
            # 计算编队时间
            if team_counter > 0:
                team_formation_time += 1
            
            # 渲染环境
            if RENDER:
                env.render()
        
        # 完成当前回合的数据收集
        if collect_formation_data and episode_formation_data:
            # 计算回合级别的编队质量指标
            if episode_formation_data['timesteps']:
                timesteps_data = episode_formation_data['timesteps']
                
                # 计算平均编队距离误差
                formation_errors = []
                for ts in timesteps_data:
                    for follower in ts['followers']:
                        formation_errors.append(follower['formation_distance_error'])
                
                avg_formation_error = np.mean(formation_errors) if formation_errors else 0.0
                
                episode_formation_data['episode_summary'] = {
                    'total_steps': len(timesteps_data),
                    'formation_rate': team_formation_time / step_count if step_count > 0 else 0,
                    'avg_formation_error': float(avg_formation_error),
                    'total_reward': float(total_reward),
                    'success': bool(win)
                }
            
            formation_data['episodes'].append(episode_formation_data)
        
        # 收集统计信息
        win_count += int(win)
        rewards.append(total_reward)
        steps.append(step_count)
        
        # 计算这一回合的编队保持率
        formation_rate = team_formation_time / step_count if step_count > 0 else 0
        formation_rates.append(formation_rate)
        
        # 记录最终距离
        if last_distance is not None:
            if isinstance(last_distance, dict):
                if "leader_to_goal" in last_distance:
                    distance_metrics["leader_to_goal"].append(float(last_distance["leader_to_goal"]))

                follower_distances = [
                    float(value)
                    for key, value in last_distance.items()
                    if key.startswith("leader_to_follower_")
                ]
                if follower_distances:
                    distance_metrics["leader_to_follower"].append(float(np.mean(follower_distances)))

                if "leader_to_obstacle" in last_distance:
                    distance_metrics["leader_to_obstacle"].append(float(last_distance["leader_to_obstacle"]))
        
        # 收集当前回合的五项指标数据
        episode_data = {
            'success': bool(win),
            'steps': step_count,
            'formation_rate': formation_rate,
            'trajectory_data': episode_trajectory,
            'action_sequence': episode_actions,
            'total_reward': total_reward,
            'final_distance': last_distance if last_distance is not None else 0.0
        }
        episode_data_list.append(episode_data)
        
        # 输出回合信息
        status = "成功" if win else "失败"
        print(f"测试回合 {episode+1}/{test_episodes}, 状态: {status}, 奖励: {total_reward:.1f}, "
              f"步数: {step_count}, 编队率: {formation_rate:.2f}")
    
    # 计算编队数据的汇总统计
    if collect_formation_data and formation_data:
        # 计算所有回合的平均编队质量指标
        all_heading_angles = {'leaders': [], 'followers': []}
        all_speeds = {'leaders': [], 'followers': []}
        all_distances = []
        
        for episode_data in formation_data['episodes']:
            for timestep in episode_data['timesteps']:
                # 收集航向角数据
                for leader in timestep['leaders']:
                    all_heading_angles['leaders'].append(leader['heading_angle'])
                for follower in timestep['followers']:
                    all_heading_angles['followers'].append(follower['heading_angle'])
                
                # 收集速度数据
                for leader in timestep['leaders']:
                    all_speeds['leaders'].append(leader['speed'])
                for follower in timestep['followers']:
                    all_speeds['followers'].append(follower['speed'])
                
                # 收集距离数据
                for follower in timestep['followers']:
                    all_distances.append(follower['leader_distance'])
        
        formation_data['summary_stats'] = {
            'mean_formation_quality': float(np.mean(formation_rates)) if formation_rates else 0.0,
            'avg_heading_angles': {
                'leaders': float(np.mean(all_heading_angles['leaders'])) if all_heading_angles['leaders'] else 0.0,
                'followers': float(np.mean(all_heading_angles['followers'])) if all_heading_angles['followers'] else 0.0
            },
            'avg_speeds': {
                'leaders': float(np.mean(all_speeds['leaders'])) if all_speeds['leaders'] else 0.0,
                'followers': float(np.mean(all_speeds['followers'])) if all_speeds['followers'] else 0.0
            },
            'avg_leader_follower_distance': float(np.mean(all_distances)) if all_distances else 0.0
        }
    
    # 计算统计结果
    success_rate = win_count / test_episodes
    avg_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    avg_steps = np.mean(steps)
    std_steps = np.std(steps)
    avg_formation_rate = np.mean(formation_rates)
    std_formation_rate = np.std(formation_rates)
    
    # 计算最终距离的平均值和标准差
    distance_stats = {
        metric_name: {
            "mean": float(np.mean(values)) if values else 0.0,
            "std": float(np.std(values)) if values else 0.0,
            "values": [float(value) for value in values]
        }
        for metric_name, values in distance_metrics.items()
    }
    
    # 计算五项性能指标
    five_metrics = calculate_five_performance_metrics(episode_data_list)
    
    # 输出总体统计信息
    print("\n蒙特卡洛测试结果统计 (平均值±标准差):")
    print(f"测试回合数: {test_episodes}")
    print(f"成功率: {success_rate:.2f}")
    print(f"平均奖励: {avg_reward:.2f}±{std_reward:.2f}")
    print(f"平均步数: {avg_steps:.2f}±{std_steps:.2f}")
    print(f"平均编队保持率: {avg_formation_rate:.2f}±{std_formation_rate:.2f}")
    print(
        "平均最终距离: "
        f"leader_to_goal={distance_stats['leader_to_goal']['mean']:.2f}±{distance_stats['leader_to_goal']['std']:.2f}, "
        f"leader_to_follower={distance_stats['leader_to_follower']['mean']:.2f}±{distance_stats['leader_to_follower']['std']:.2f}, "
        f"leader_to_obstacle={distance_stats['leader_to_obstacle']['mean']:.2f}±{distance_stats['leader_to_obstacle']['std']:.2f}"
    )
    
    # 输出五项性能评估指标
    print("\n五指标性能评估:")
    print(f"1. 任务完成率(MCR): {five_metrics['MCR']:.2f}")
    print(f"2. 编队保持率(FKR): {five_metrics['FKR']['mean']:.2f}±{five_metrics['FKR']['std']:.2f}")
    print(f"3. 成功率加权探索时间(SET): {five_metrics['SET']:.2f} (SR: {five_metrics['success_rate']:.2f} × 平均时间: {five_metrics['avg_exploration_time']:.2f})")
    print(f"4. 飞行轨迹(J_S): {five_metrics['J_S']['mean']:.2f}±{five_metrics['J_S']['std']:.2f}")
    print(f"5. 能量消耗(J_C): {five_metrics['J_C']['mean']:.2f}±{five_metrics['J_C']['std']:.2f}")
    
    # 构建结果字典
    results = {
        "test_episodes": test_episodes,
        "success_rate": success_rate,
        "rewards": {
            "mean": float(avg_reward),
            "std": float(std_reward),
            "values": [float(r) for r in rewards]
        },
        "steps": {
            "mean": float(avg_steps),
            "std": float(std_steps),
            "values": [float(s) for s in steps]
        },
        "formation_rates": {
            "mean": float(avg_formation_rate),
            "std": float(std_formation_rate),
            "values": [float(f) for f in formation_rates]
        },
        "distances": distance_stats,
        "test_config": {
            "hero_count": n_agent,
            "enemy_count": m_enemy,
            "obstacle_count": obstacle_count,
            "uav_speed": uav_speed
        },
        "five_metrics": five_metrics,  # 添加五项性能指标
        "timestamp": time.time()
    }
    
    # 如果收集了编队数据，将其添加到结果中
    if collect_formation_data and formation_data:
        results["formation_data"] = formation_data
    
    # 创建唯一的测试ID和保存目录
    timestamp = get_timestamp()
    config_str = f"_h{N_Agent}_e{M_Enemy}_o{obstacle_count}"
    if uav_speed:
        config_str += f"_s{int(uav_speed)}"
    
    # 创建保存目录
    test_dir_name = f"test_{timestamp}{config_str}"
    test_dir = os.path.join(TEST_RESULTS_BASE, test_dir_name)
    ensure_dir_exists(test_dir)
    
    # 保存测试结果 (pickle格式)
    pickle_path = os.path.join(test_dir, "test_results.pkl")
    with open(pickle_path, 'wb') as f:
        pkl.dump(results, f, pkl.HIGHEST_PROTOCOL)
    print(f"测试结果已保存到: {pickle_path}")
    
    # 同时保存为JSON格式
    json_path = os.path.join(test_dir, "test_results.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(convert_to_json_compatible(results), f, ensure_ascii=False, indent=4)
    print(f"测试结果(JSON格式)已保存到: {json_path}")
    
    # 保存测试信息
    date_str = datetime.datetime.fromtimestamp(results["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
    model_name = os.path.basename(model_path)
    
    test_info = {
        "timestamp": timestamp,
        "date": date_str,
        "model": model_name,
        "config": results["test_config"],
        "success_rate": success_rate,
        "avg_reward": float(avg_reward),
        "avg_steps": float(avg_steps),
        "formation_rate": float(avg_formation_rate),
        "distance_metrics": {
            metric_name: {
                "mean": metric_stats["mean"],
                "std": metric_stats["std"]
            }
            for metric_name, metric_stats in distance_stats.items()
        }
    }
    
    info_path = os.path.join(test_dir, "test_info.json")
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(test_info, f, ensure_ascii=False, indent=4)
    
    # 单独保存编队数据 (PKL格式)
    if collect_formation_data and formation_data:
        formation_pkl_path = os.path.join(test_dir, "formation_data.pkl")
        with open(formation_pkl_path, 'wb') as f:
            pkl.dump(formation_data, f, pkl.HIGHEST_PROTOCOL)
        print(f"编队数据已保存到: {formation_pkl_path}")
        
        # 保存编队数据汇总 (PKL格式)
        formation_summary_path = os.path.join(test_dir, "formation_summary.pkl")
        formation_summary = {
            'summary_stats': formation_data['summary_stats'],
            'test_info': formation_data['test_info'],
            'episode_summaries': [ep['episode_summary'] for ep in formation_data['episodes']]
        }
        with open(formation_summary_path, 'wb') as f:
            pkl.dump(formation_summary, f, pkl.HIGHEST_PROTOCOL)
        print(f"编队数据汇总已保存到: {formation_summary_path}")
    
    # 绘制测试奖励分布直方图
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 8))
        
        # 奖励分布直方图
        plt.subplot(2, 2, 1)
        plt.hist(rewards, bins=min(20, test_episodes//5), alpha=0.7)
        plt.title('奖励分布')
        plt.xlabel('奖励')
        plt.ylabel('频次')
        plt.axvline(avg_reward, color='r', linestyle='dashed', linewidth=1, label=f'平均值: {avg_reward:.2f}')
        plt.legend()
        
        # 步数分布直方图
        plt.subplot(2, 2, 2)
        plt.hist(steps, bins=min(20, test_episodes//5), alpha=0.7)
        plt.title('步数分布')
        plt.xlabel('步数')
        plt.ylabel('频次')
        plt.axvline(avg_steps, color='r', linestyle='dashed', linewidth=1, label=f'平均值: {avg_steps:.2f}')
        plt.legend()
        
        # 编队率分布直方图
        plt.subplot(2, 2, 3)
        plt.hist(formation_rates, bins=min(20, test_episodes//5), alpha=0.7)
        plt.title('编队保持率分布')
        plt.xlabel('编队保持率')
        plt.ylabel('频次')
        plt.axvline(avg_formation_rate, color='r', linestyle='dashed', linewidth=1, label=f'平均值: {avg_formation_rate:.2f}')
        plt.legend()
        
        # 距离分布直方图
        if any(metric_stats["values"] for metric_stats in distance_stats.values()):
            plt.subplot(2, 2, 4)
            for metric_name, metric_stats in distance_stats.items():
                values = metric_stats["values"]
                if not values:
                    continue
                plt.hist(values, bins=min(20, test_episodes//5), alpha=0.45, label=metric_name)
                plt.axvline(
                    metric_stats["mean"],
                    linestyle='dashed',
                    linewidth=1,
                    label=f'{metric_name} mean: {metric_stats["mean"]:.2f}'
                )
            plt.title('最终距离分布')
            plt.xlabel('距离')
            plt.ylabel('频次')
            plt.legend()
        
        plt.tight_layout()
        title = f"蒙特卡洛测试结果 (友方:{N_Agent}, 敌方:{M_Enemy}, 障碍:{obstacle_count})"
        if uav_speed:
            title += f", 速度:{uav_speed}"
        plt.suptitle(title)
        
        # 保存图片到测试结果目录
        save_img_path = os.path.join(test_dir, "histogram.png")
        plt.savefig(save_img_path)
        plt.close()
        print(f"测试结果直方图已保存到: {save_img_path}")
    except Exception as e:
        print(f"绘制直方图时出错: {e}")
        import traceback
        traceback.print_exc()
    
    # 更新测试结果索引
    create_test_results_index()
    
    return results

def analyze_test_results(result_files=None, base_path=None):
    """分析和比较一组测试结果
    
    Args:
        result_files: 测试结果文件列表，不提供则自动搜索base_path下的所有测试结果
        base_path: 测试结果文件的基本路径，默认为TEST_RESULTS_BASE目录
        
    Returns:
        分析结果汇总
    """
    import glob
    import os
    import matplotlib.pyplot as plt
    
    if base_path is None:
        base_path = TEST_RESULTS_BASE
        
    if not os.path.exists(base_path):
        print(f"测试结果目录不存在: {base_path}")
        ensure_dir_exists(base_path)
        print("没有找到任何测试结果文件")
        return None
    
    if result_files is None:
        # 自动搜索所有测试结果文件 - 优先查找JSON文件
        all_results = []
        
        # 遍历所有测试目录
        for root, dirs, files in os.walk(base_path):
            # 查找JSON结果文件
            json_results = [os.path.join(root, f) for f in files 
                           if f in ["test_results.json", "all_results.json"]]
            
            # 如果找到了JSON文件，优先使用
            if json_results:
                all_results.extend(json_results)
            else:
                # 否则查找pickle文件
                pkl_results = [os.path.join(root, f) for f in files 
                              if f in ["test_results.pkl", "all_results.pkl"]]
                all_results.extend(pkl_results)
        
        result_files = all_results
        
        if not result_files:
            print(f"在{base_path}目录及其子目录下未找到测试结果文件")
            return None
    
    print(f"找到{len(result_files)}个测试结果文件:")
    for i, file in enumerate(result_files):
        print(f"{i+1}. {file}")
    
    # 加载测试结果
    test_results = []
    for file in result_files:
        try:
            # 根据文件扩展名决定加载方式
            if file.endswith('.json'):
                with open(file, 'r', encoding='utf-8') as f:
                    result = json.load(f)
            else:  # 假设是pickle文件
                with open(file, 'rb') as f:
                    result = pkl.load(f)
                    
            # 处理多难度测试结果
            if "level_1" in result:
                # 多难度测试结果 - 提取每个难度级别作为单独的结果
                for level_name, level_result in result.items():
                    if isinstance(level_result, dict) and "test_config" in level_result:
                        level_result['level_name'] = level_name
                        level_result['file_name'] = os.path.basename(file) + f":{level_name}"
                        test_results.append(level_result)
                        print(f"已加载测试结果: {level_result['file_name']}")
            else:
                # 单一难度测试结果
                result['file_name'] = os.path.basename(file)
                test_results.append(result)
                print(f"已加载测试结果: {result['file_name']}")
                
        except Exception as e:
            print(f"加载文件{file}时出错: {e}")
            import traceback
            traceback.print_exc()
    
    if not test_results:
        print("没有成功加载任何测试结果")
        return None
    
    # 分析和比较测试结果
    print("\n测试结果比较:")
    print("-" * 80)
    print(f"{'配置信息':^40} | {'成功率':^10} | {'平均奖励':^15} | {'平均步数':^15} | {'编队率':^15}")
    print("-" * 80)
    
    for result in test_results:
        config = result.get('test_config', {})
        hero_count = config.get('hero_count', 'N/A')
        enemy_count = config.get('enemy_count', 'N/A')
        obstacle_count = config.get('obstacle_count', 'N/A')
        uav_speed = config.get('uav_speed', 'N/A')
        
        # 添加难度级别标识（如果有）
        level_prefix = ""
        if 'level_name' in result:
            level_prefix = f"{result['level_name']}: "
        
        config_str = f"{level_prefix}友方:{hero_count}, 敌方:{enemy_count}, 障碍:{obstacle_count}"
        if uav_speed != 'N/A':
            config_str += f", 速度:{uav_speed}"
            
        success_rate = result.get('success_rate', 0)
        reward_mean = result.get('rewards', {}).get('mean', 0)
        reward_std = result.get('rewards', {}).get('std', 0)
        steps_mean = result.get('steps', {}).get('mean', 0)
        steps_std = result.get('steps', {}).get('std', 0)
        formation_mean = result.get('formation_rates', {}).get('mean', 0)
        formation_std = result.get('formation_rates', {}).get('std', 0)
        
        print(f"{config_str:40} | {success_rate:10.2f} | {reward_mean:6.2f}±{reward_std:6.2f} | "
              f"{steps_mean:6.2f}±{steps_std:6.2f} | {formation_mean:6.2f}±{formation_std:6.2f}")
    
    print("-" * 80)
    
    # 绘制对比图表
    try:
        plt.figure(figsize=(16, 10))
        
        # 提取数据
        labels = []
        success_rates = []
        reward_means = []
        reward_stds = []
        steps_means = []
        formation_means = []
        
        for result in test_results:
            config = result.get('test_config', {})
            hero_count = config.get('hero_count', 'N/A')
            enemy_count = config.get('enemy_count', 'N/A')
            obstacle_count = config.get('obstacle_count', 'N/A')
            
            # 添加难度级别标识（如果有）
            if 'level_name' in result:
                label = f"{result['level_name']}"
            else:
                label = f"H{hero_count}E{enemy_count}O{obstacle_count}"
                
            if config.get('uav_speed', 'N/A') != 'N/A':
                label += f"S{config['uav_speed']}"
                
            labels.append(label)
            success_rates.append(result.get('success_rate', 0))
            reward_means.append(result.get('rewards', {}).get('mean', 0))
            reward_stds.append(result.get('rewards', {}).get('std', 0))
            steps_means.append(result.get('steps', {}).get('mean', 0))
            formation_means.append(result.get('formation_rates', {}).get('mean', 0))
        
        # 绘制成功率对比
        plt.subplot(2, 2, 1)
        plt.bar(labels, success_rates, alpha=0.7)
        plt.title('成功率对比')
        plt.ylabel('成功率')
        plt.xticks(rotation=45)
        plt.ylim(0, 1.1)
        
        # 绘制平均奖励对比
        plt.subplot(2, 2, 2)
        plt.bar(labels, reward_means, yerr=reward_stds, alpha=0.7, capsize=5)
        plt.title('平均奖励对比')
        plt.ylabel('平均奖励')
        plt.xticks(rotation=45)
        
        # 绘制平均步数对比
        plt.subplot(2, 2, 3)
        plt.bar(labels, steps_means, alpha=0.7)
        plt.title('平均步数对比')
        plt.ylabel('平均步数')
        plt.xticks(rotation=45)
        
        # 绘制编队率对比
        plt.subplot(2, 2, 4)
        plt.bar(labels, formation_means, alpha=0.7)
        plt.title('编队保持率对比')
        plt.ylabel('编队保持率')
        plt.xticks(rotation=45)
        plt.ylim(0, 1.1)
        
        plt.tight_layout()
        plt.suptitle('测试结果比较', fontsize=16)
        
        # 保存比较图
        analysis_dir = os.path.join(TEST_RESULTS_BASE, "analysis")
        ensure_dir_exists(analysis_dir)
        comparison_path = os.path.join(analysis_dir, f"test_comparison_{get_timestamp()}.png")
        plt.savefig(comparison_path)
        plt.close()
        print(f"测试结果比较图已保存到: {comparison_path}")
    except Exception as e:
        print(f"绘制比较图时出错: {e}")
        import traceback
        traceback.print_exc()
    
    return test_results

def monte_carlo_test(actor_path, critic_path=None, test_nums=100, base_difficulty_levels=None):
    """执行蒙特卡洛测试
    
    Args:
        actor_path: Actor路径
        critic_path: Critic路径(可选)
        test_nums: 测试次数
        base_difficulty_levels: 基础难度级别列表
    """
    # 声明需要的全局变量
    global TEST_RESULTS_BASE, N_Agent, M_Enemy
    
    # 如果没有提供难度级别，使用默认的1-5个障碍物配置
    if base_difficulty_levels is None:
        base_difficulty_levels = [
            {'obstacle_count': 2, 'uav_speed': 1.0},  # 默认从2个障碍物开始
            {'obstacle_count': 3, 'uav_speed': 1.0},
            {'obstacle_count': 4, 'uav_speed': 1.0},
            {'obstacle_count': 5, 'uav_speed': 1.0},
            {'obstacle_count': 6, 'uav_speed': 1.0}
        ]
    
    # 加载环境和模型
    from rl_env.path_env import RlGame
    
    # 创建多难度测试专用目录
    timestamp = get_timestamp()
    model_name = os.path.basename(actor_path)
    multi_test_dir = os.path.join(TEST_RESULTS_BASE, f"multi_diff_test_{timestamp}_{model_name}")
    ensure_dir_exists(multi_test_dir)
    
    # 保存测试配置信息
    test_config = {
        "timestamp": timestamp,
        "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": model_name,
        "test_episodes": test_nums,
        "difficulty_levels": base_difficulty_levels,
        "hero_count": N_Agent,
        "enemy_count": M_Enemy,
        "goal_position": (500, 200)  # 记录固定目标位置
    }
    
    config_path = os.path.join(multi_test_dir, "test_config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(convert_to_json_compatible(test_config), f, ensure_ascii=False, indent=4)
    
    # 初始化结果存储
    all_results = {}
    
    # 对每个难度级别进行测试
    for difficulty_idx, difficulty_config in enumerate(base_difficulty_levels):
        print(f"\n{'-'*50}")
        print(f"测试难度级别 {difficulty_idx+1}/{len(base_difficulty_levels)}")
        print(f"配置: {difficulty_config}")
        print(f"{'-'*50}")
        
        # 运行测试 - 使用评估模式加载模型（仅加载Actor）
        result = run_monte_carlo_test(
            model_path=actor_path,
            test_episodes=test_nums,
            test_options=difficulty_config
        )
        
        # 存储结果
        level_key = f"level_{difficulty_idx+1}"
        all_results[level_key] = result
        
        # 打印当前难度级别的主要指标
        print(f"\n难度级别 {difficulty_idx+1} 测试结果摘要:")
        print(f"成功率: {result['success_rate']:.2f}")
        print(f"平均奖励: {result['rewards']['mean']:.2f}±{result['rewards']['std']:.2f}")
        print(f"平均步数: {result['steps']['mean']:.2f}±{result['steps']['std']:.2f}")
        print(f"平均编队率: {result['formation_rates']['mean']:.2f}±{result['formation_rates']['std']:.2f}")
        print(f"平均路径效率: {1.0 / result['steps']['mean']:.4f}")
        print(f"平均碰撞率: {1.0 - result['success_rate']:.2f}")
        
        # 打印五项性能指标
        if 'five_metrics' in result:
            five_metrics = result['five_metrics']
            print(f"\n五指标性能评估:")
            print(f"1. 任务完成率(MCR): {five_metrics['MCR']:.2f}")
            print(f"2. 编队保持率(FKR): {five_metrics['FKR']['mean']:.2f}±{five_metrics['FKR']['std']:.2f}")
            print(f"3. 成功率加权探索时间(SET): {five_metrics['SET']:.2f}")
            print(f"4. 飞行轨迹(J_S): {five_metrics['J_S']['mean']:.2f}±{five_metrics['J_S']['std']:.2f}")
            print(f"5. 能量消耗(J_C): {five_metrics['J_C']['mean']:.2f}±{five_metrics['J_C']['std']:.2f}")
    
    # 输出整体结果摘要
    print(f"\n{'='*60}")
    print(f"多难度蒙特卡洛测试完成")
    print(f"{'='*60}\n")
    
    print("各难度级别测试结果汇总:")
    print(f"{'难度级别':^12} | {'成功率':^8} | {'平均奖励':^15} | {'平均步数':^15} | {'编队率':^15}")
    print("-" * 75)
    
    for level_idx, (level_name, level_result) in enumerate(all_results.items()):
        print(f"{level_name:^12} | {level_result['success_rate']:8.2f} | "
              f"{level_result['rewards']['mean']:6.2f}±{level_result['rewards']['std']:6.2f} | "
              f"{level_result['steps']['mean']:6.2f}±{level_result['steps']['std']:6.2f} | "
              f"{level_result['formation_rates']['mean']:6.2f}±{level_result['formation_rates']['std']:6.2f}")
    
    print("-" * 75)
    
    # 绘制不同难度级别的对比图表
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        # 从结果中提取数据
        levels = list(all_results.keys())
        success_rates = [all_results[level]['success_rate'] for level in levels]
        rewards = [all_results[level]['rewards']['mean'] for level in levels]
        reward_stds = [all_results[level]['rewards']['std'] for level in levels]
        steps = [all_results[level]['steps']['mean'] for level in levels]
        step_stds = [all_results[level]['steps']['std'] for level in levels]
        formation_rates = [all_results[level]['formation_rates']['mean'] for level in levels]
        formation_stds = [all_results[level]['formation_rates']['std'] for level in levels]
        
        # 计算路径效率和碰撞率
        path_efficiencies = [1.0 / step if step > 0 else 0 for step in steps]
        collision_rates = [1.0 - sr for sr in success_rates]
        
        # 创建图表
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # 成功率
        axes[0, 0].bar(levels, success_rates, color='green')
        axes[0, 0].set_title('成功率')
        axes[0, 0].set_ylim(0, 1.0)
        for i, v in enumerate(success_rates):
            axes[0, 0].text(i, v + 0.02, f'{v:.2f}', ha='center')
        
        # 平均奖励
        axes[0, 1].bar(levels, rewards, color='blue', yerr=reward_stds, alpha=0.7, capsize=5)
        axes[0, 1].set_title('平均奖励')
        for i, v in enumerate(rewards):
            axes[0, 1].text(i, v + 0.5 if v >= 0 else v - 1.5, f'{v:.2f}', ha='center')
        
        # 平均步数
        axes[0, 2].bar(levels, steps, color='orange', yerr=step_stds, capsize=5)
        axes[0, 2].set_title('平均步数')
        for i, v in enumerate(steps):
            axes[0, 2].text(i, v + 2, f'{v:.2f}', ha='center')
        
        # 编队率
        axes[1, 0].bar(levels, formation_rates, color='purple', yerr=formation_stds, capsize=5)
        axes[1, 0].set_title('编队保持率')
        axes[1, 0].set_ylim(0, 1.0)
        for i, v in enumerate(formation_rates):
            axes[1, 0].text(i, v + 0.02, f'{v:.2f}', ha='center')
        
        # 路径效率
        axes[1, 1].bar(levels, path_efficiencies, color='teal')
        axes[1, 1].set_title('路径效率')
        for i, v in enumerate(path_efficiencies):
            axes[1, 1].text(i, v + 0.002, f'{v:.4f}', ha='center')
        
        # 碰撞率
        axes[1, 2].bar(levels, collision_rates, color='red')
        axes[1, 2].set_title('碰撞率')
        axes[1, 2].set_ylim(0, 1.0)
        for i, v in enumerate(collision_rates):
            axes[1, 2].text(i, v + 0.02, f'{v:.2f}', ha='center')
        
        # 设置整体标题
        model_name = actor_path.split('/')[-1] if '/' in actor_path else actor_path
        plt.suptitle(f'模型 {model_name} 在不同难度级别的性能', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # 保存图表到多难度测试目录
        comparison_img_path = os.path.join(multi_test_dir, "difficulty_comparison.png")
        plt.savefig(comparison_img_path)
        plt.close()
        print(f"难度级别比较图已保存到: {comparison_img_path}")
    except Exception as e:
        print(f"绘制性能比较图时出错: {e}")
        import traceback
        traceback.print_exc()
    
    # 保存完整结果 (Pickle格式)
    try:
        pickle_results_path = os.path.join(multi_test_dir, "all_results.pkl")
        with open(pickle_results_path, 'wb') as f:
            pkl.dump(all_results, f, pkl.HIGHEST_PROTOCOL)
        print(f"完整测试结果(Pickle格式)已保存到: {pickle_results_path}")
        
        # 同时保存为JSON格式
        json_results_path = os.path.join(multi_test_dir, "all_results.json")
        with open(json_results_path, 'w', encoding='utf-8') as f:
            json.dump(convert_to_json_compatible(all_results), f, ensure_ascii=False, indent=4)
        print(f"完整测试结果(JSON格式)已保存到: {json_results_path}")
    except Exception as e:
        print(f"保存完整测试结果时出错: {e}")
    
    # 更新测试结果索引
    create_test_results_index()
    
    return all_results

def main():
    """主函数 - 无课程学习版本
    """
    # 添加命令行参数解析
    parser = argparse.ArgumentParser(description='MASAC without Curriculum Learning')
    parser.add_argument('--render', action='store_true', help='是否渲染环境')
    parser.add_argument('--no_render', action='store_true', help='测试模式下关闭渲染窗口')
    parser.add_argument('--test', action='store_true', help='测试模式（加载已训练的模型）')
    parser.add_argument('--model_path', type=str, default='models/direct_training_final/final_model', 
                        help='测试模式下加载的模型路径')
    # 添加更多测试相关选项
    parser.add_argument('--test_episodes', type=int, default=None, 
                       help='蒙特卡洛测试的回合数，默认使用全局变量TEST_EPIOSDE')
    parser.add_argument('--hero_count', type=int, default=1,
                       help='测试或训练时使用的友方无人机数量，默认为1')
    parser.add_argument('--enemy_count', type=int, default=3,
                       help='测试或训练时使用的敌方无人机数量，默认为3')
    parser.add_argument('--obstacle_count', type=int, default=2,
                       help='测试时使用的障碍物数量，默认为2')
    parser.add_argument('--test_speed', type=float, default=None,
                       help='测试时使用的无人机速度，不设置则使用默认速度')
    # 添加多难度测试选项
    parser.add_argument('--multi_difficulty_test', action='store_true',
                       help='在多个难度级别上进行蒙特卡洛测试')
    parser.add_argument('--max_obstacle', type=int, default=5,
                       help='多难度测试时的最大障碍物数量，默认为5')
    parser.add_argument('--step_size', type=float, default=1.0,
                       help='多难度测试时的障碍物数量步长，默认为1')
    parser.add_argument('--test_difficulty', type=str, default=None,
                       help='自定义难度测试，格式为逗号分隔的难度值，例如"1,2,3,4,5"')
    # 添加测试结果分析选项
    parser.add_argument('--analyze', action='store_true', 
                       help='分析已有的测试结果而非进行新的测试')
    parser.add_argument('--result_path', type=str, default=None,
                       help='测试结果文件路径，默认自动搜索')
    # 添加结果保存目录选项
    parser.add_argument('--results_dir', type=str, default=None,
                       help='测试结果保存目录，默认为"D:/pa/path planning2/results"')
    parser.add_argument('--create_index', action='store_true',
                       help='生成测试结果索引HTML文件')
    # 添加日志级别控制
    parser.add_argument('--log_level', type=str, choices=['debug', 'info', 'warning', 'error'], default='info',
                      help='设置日志级别：debug(调试), info(信息), warning(警告), error(错误)')
    # 添加多次训练模式参数
    parser.add_argument('--multi_run', action='store_true',
                       help='启用多次训练模式，使用不同随机种子进行多次训练')
    parser.add_argument('--num_runs', type=int, default=3,
                       help='多次训练模式下的训练次数，默认为3次')
    parser.add_argument('--seeds', type=str, default=None,
                       help='自定义随机种子列表，格式为逗号分隔的数字，例如"42,123,456"。如不指定则自动生成')
                       
    args = parser.parse_args()
    
    # 设置全局日志级别
    log_level_map = {
        'debug': LOG_DEBUG,
        'info': LOG_INFO,
        'warning': LOG_WARNING,
        'error': LOG_ERROR
    }
    set_log_level(log_level_map.get(args.log_level, LOG_INFO))
    log(f"日志级别设置为: {args.log_level.upper()}", LOG_INFO)
    
    global RENDER, action_number, N_Agent, M_Enemy, TRAINING_RESULTS_FILE, TEST_RESULTS_BASE, RESULTS_DIR
    
    # 如果指定了结果目录，更新全局变量
    if args.results_dir:
        RESULTS_DIR = args.results_dir
        TEST_RESULTS_BASE = os.path.join(RESULTS_DIR, "test_results")
        TRAINING_RESULTS_FILE = os.path.join(RESULTS_DIR, "MASAC_no_curriculum.pkl")
        print(f"结果将保存在: {RESULTS_DIR}")
        ensure_dir_exists(RESULTS_DIR)
        ensure_dir_exists(TEST_RESULTS_BASE)
    
    # 如果只是创建索引，直接调用索引创建函数后返回
    if args.create_index:
        print("正在创建测试结果索引...")
        index_path = create_test_results_index()
        print(f"索引已创建: {index_path}")
        return
    
    # 如果只是分析结果，直接调用分析函数
    if args.analyze:
        print("分析测试结果模式")
        if args.result_path:
            analyze_test_results([args.result_path])
        else:
            analyze_test_results()
        return
    
    # 设置渲染标志：测试时默认渲染，训练时根据参数决定
    if args.no_render:
        RENDER = False
    elif args.test or args.multi_difficulty_test:
        RENDER = True
    else:
        RENDER = args.render
    
    # 使用解析后的参数或默认值
    n_agent = args.hero_count
    m_enemy = args.enemy_count
    print(f"设置友方无人机数量 (n_agent): {n_agent}")
    print(f"设置敌方无人机数量 (m_enemy): {m_enemy}")
    
    # 更新全局变量以确保所有函数使用正确的值
    N_Agent = n_agent
    M_Enemy = m_enemy
    
    # 创建一个临时环境实例以获取动作空间信息
    # 使用 n_agent 和 m_enemy 变量
    temp_env = RlGame(leader_count=n_agent, follower_count=m_enemy, obstacle_num=args.obstacle_count, render=False).unwrapped
    action_number = temp_env.action_space.shape[0]
    temp_env.close()
    
    # 更新main_SAC模块中的全局变量action_number
    import main_SAC
    main_SAC.action_number = action_number
    
    if args.multi_difficulty_test:
        # 多难度测试模式
        print("启动多难度蒙特卡洛测试模式")
        # 设置难度级别配置
        difficulty_levels = []
        
        # 如果提供了自定义难度列表
        if args.test_difficulty:
            try:
                custom_difficulties = [int(d) for d in args.test_difficulty.split(',')]
                print(f"使用自定义难度级别: {custom_difficulties}")
                difficulty_levels = [
                    {'obstacle_count': d, 'uav_speed': args.test_speed or 1.0}
                    for d in custom_difficulties
                ]
            except ValueError:
                print(f"无效的自定义难度格式: {args.test_difficulty}，将使用默认设置")
                difficulty_levels = []
        
        # 如果没有提供自定义难度或解析失败，使用默认生成的难度级别
        if not difficulty_levels:
            max_obstacle = args.max_obstacle
            step = args.step_size
            
            print(f"生成难度级别 - 最大障碍物: {max_obstacle}, 步长: {step}")
            obstacle_counts = [int(1 + i * step) for i in range(int((max_obstacle - 1) / step) + 1)]
            
            difficulty_levels = [
                {'obstacle_count': d, 'uav_speed': args.test_speed or 1.0}
                for d in obstacle_counts
            ]
        
        # 修正模型路径 - 如果是相对路径且不存在，尝试上级目录
        model_path = args.model_path
        if not os.path.isabs(model_path) and not os.path.exists(model_path):
            parent_model_path = os.path.join('..', model_path)
            if os.path.exists(parent_model_path):
                model_path = parent_model_path
                print(f"使用修正后的模型路径: {model_path}")
        
        # 输出最终使用的难度级别
        print(f"将测试的难度级别配置: {difficulty_levels}")
                
        # 运行多难度测试
        monte_carlo_test(
            actor_path=model_path,
            critic_path=model_path,  # 使用相同的路径前缀，加载函数会自动添加_critic_{i}.pth
            test_nums=args.test_episodes or TEST_EPIOSDE,
            base_difficulty_levels=difficulty_levels
        )
    elif args.test:
        # 单难度测试模式
        print(f"启动单一难度蒙特卡洛测试模式: 友方={n_agent}, 敌方={m_enemy}, 障碍物={args.obstacle_count}")
        if args.test_speed:
            print(f"指定无人机速度: {args.test_speed}")
        
        # 修正模型路径 - 如果是相对路径且不存在，尝试上级目录
        model_path = args.model_path
        if not os.path.isabs(model_path) and not os.path.exists(model_path):
            parent_model_path = os.path.join('..', model_path)
            if os.path.exists(parent_model_path):
                model_path = parent_model_path
                print(f"使用修正后的模型路径: {model_path}")
        
        # 设置测试选项
        test_options = {
            'hero_count': n_agent, # 使用 n_agent
            'enemy_count': m_enemy, # 使用 m_enemy
            'obstacle_count': args.obstacle_count,
            'uav_speed': args.test_speed
        }
        
        # 运行测试
        run_monte_carlo_test(
            model_path=model_path,
            test_episodes=args.test_episodes or TEST_EPIOSDE, # TEST_EPIOSDE 也需要定义
            test_options=test_options
        )
    else:
        # 训练模式 - 直接训练（无课程学习）
        if args.multi_run:
            print("使用多次训练模式（无课程学习）")
            run_multi_seed_training(args, n_agent, m_enemy)
        else:
            print("使用单次直接训练模式（无课程学习）")
            # 使用 n_agent 和 m_enemy 变量进行直接训练
            run_direct_training(args, n_agent, m_enemy)

if __name__ == "__main__":
    # 定义一些全局常量（如果测试模式需要）
    # 这些值应该与 main_SAC.py 中的默认值或 args 的默认值对齐
    set_seed(42) # 添加set_seed调用

    TRAIN_NUM = 1 # 添加 TRAIN_NUM 定义
    TEST_EPIOSDE = 100 
    # 定义 state_number 和 action_number 的默认值，它们会被覆盖
    state_number = 7 # 假设默认状态维度
    action_number = 2 # 假设默认动作维度
    
    # 设置默认的智能体数量，与命令行参数默认值保持一致
    N_Agent = 1  # 主机数量默认为1
    M_Enemy = 3  # 从机数量默认为3
    
    # 定义设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 定义其他可能在全局范围使用的常量
    EP_LEN = 1000
    MemoryCapacity = 50000
    BATCH = 256
    # 设置保存路径到masac_no_curriculum文件夹下
    MASAC_NO_CURRICULUM_DIR = os.path.dirname(os.path.abspath(__file__))
    RESULTS_DIR = os.path.join(MASAC_NO_CURRICULUM_DIR, "results")
    TEST_RESULTS_BASE = os.path.join(RESULTS_DIR, "test_results")
    TRAINING_RESULTS_FILE = os.path.join(RESULTS_DIR, "MASAC_no_curriculum.pkl")

    main()
    
# 使用示例:
# 1. 单次训练: python main_masac_no_curriculum.py
# 2. 多次训练(默认3次): python main_masac_no_curriculum.py --multi_run
# 3. 自定义多次训练: python main_masac_no_curriculum.py --multi_run --num_runs 5 --seeds "42,123,456,789,999"
# 4. 测试模型: python main_masac_no_curriculum.py --test --model_path models/your_model
# 5. 绘制置信区间: python plot_confidence_intervals.py --results_dir masac_no_curriculum/results
