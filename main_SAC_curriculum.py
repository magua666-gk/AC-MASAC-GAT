# -*- coding: utf-8 -*-
import argparse
import gym
import numpy as np
import os
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
from visualization.matplotlib_fonts import configure_matplotlib_fonts
configure_matplotlib_fonts()
import matplotlib.pyplot as plt
from tqdm.auto import tqdm, trange
import pickle as pkl
import re
import sys
import atexit
from typing import Dict, Any

# Import environment
from rl_env.path_env import RlGame

# Import network components
from masac_adapter.role_specific_networks import RoleEmbedding, PolicyNetFlatRole, SharedEncoder, QHead, CriticNetAttentionFlat, ROLE_EMBED_DIM, EMBED_DIM
from masac_adapter.masac_adapter import MASACEntroy,set_log_level, MultiHeadAttention, max_action, min_action, LEADER_TYPE_ID, FOLLOWER_TYPE_ID, log, LOG_INFO, LOG_WARNING, LOG_DEBUG, LOG_ERROR, clear_log_history
from main_SAC import Ornstein_Uhlenbeck_Noise
from masac_adapter.smer_memory import SMERMemory

# Import new Actor and Critic networks
from masac_adapter.actor_networks import LeaderActorNet, FollowerActorNet, AttentionLeaderActorNet, AttentionFollowerActorNet
from masac_adapter.critic_networks import StructuredAttentionCriticNet

# Import curriculum learning manager
from curriculum import CurriculumManager, FixedTaskGenerator, CurriculumConfig, LinearTaskSequencer,PolicyTransfer

# Set random seed
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

# Role-based MASAC controller
class MASACController:
    """Role-based Multi-Agent SAC Controller"""
    
    def __init__(self, n_agents=1, state_dim=17, action_dim=4, memory_size=int(2e6), \
            batch_size=256, gamma=0.99, tau=0.01, value_lr=3e-4, policy_lr=1e-4, \
            hidden_dim=256, target_update_interval=2, reward_scale=0.1, \
            auto_entropy=False, entropy_lr=3e-4, target_entropy=-0.1, alpha_min=0.01, alpha_max=1.0, device=None, 
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
        self.auto_entropy = bool(auto_entropy)
        self.entropy_lr = entropy_lr # Save entropy_lr as instance attribute
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        
        if self.alpha_min <= 0:
            self.alpha_min = 1e-6
        
        if self.alpha_max < self.alpha_min:
            raise ValueError(f"alpha_max({self.alpha_max}) must be >= alpha_min({self.alpha_min})")

        # Handle memory size parameter (backward compatible)
        if memory_capacity is not None:
            print(f"Warning: Deprecated parameter 'memory_capacity' used, please use 'memory_size' instead")
            self.memory_capacity = memory_capacity
            memory_size = memory_capacity
        else:
            self.memory_capacity = memory_size  # Save copy for reset_memory method
        
        print(f"Initializing MASAC controller with graph attention (GAT): {n_agents} agents, state_dim={state_dim}/agent, action_dim={action_dim}/agent")
        print(f"Using device: {self.device}")
        print(f"Alpha mode: {'adaptive' if self.auto_entropy else 'fixed at 1.0'}")
        
        # Initialize experience replay buffer
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
        
        # Target Leader Actor network
        self.target_leader_actor = AttentionLeaderActorNet(
            state_dim=state_dim,
            action_dim=action_dim,
            embed_dim=embed_dim,
            hidden_dims=actor_hidden_dims,
            n_heads=actor_n_heads,
            dropout=actor_dropout,
            use_shared_layer=use_shared_layer
        )
        
        # Load initial target Leader Actor parameters
        self.target_leader_actor.load_state_dict(self.leader_actor.state_dict())
        
        # Determine maximum number of followers to create
        max_followers = max(n_agents - 1, 3)  # Support at least 3 followers, or based on initial n_agents
        
        # Create separate Actor network for each possible follower
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
        
        # Create separate target Actor network for each possible follower
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
        
        # Load initial target Follower Actor parameters
        for i in range(max_followers):
            self.target_follower_actors[i].load_state_dict(self.follower_actors[i].state_dict())
            
        # Old LeaderActorNet and FollowerActorNet code (commented out)
        """
        # Leader Actor network
        self.leader_actor = LeaderActorNet(
            state_dim=state_dim,
            action_dim=action_dim,
            embed_dim=embed_dim,
            hidden_dims=actor_hidden_dims
        )
        
        # Follower Actor network
        self.follower_actor = FollowerActorNet(
            state_dim=state_dim,
            action_dim=action_dim,
            embed_dim=embed_dim,
            hidden_dims=actor_hidden_dims
        )
        
        # Target Actor networks
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
        
        # Load initial target Actor parameters
        self.target_leader_actor.load_state_dict(self.leader_actor.state_dict())
        self.target_follower_actor.load_state_dict(self.follower_actor.state_dict())
        """
        
        # === Create graph-attention Critic network ===
        critic_hidden_dims = [256, 128]  # Critic hidden layer dimensions
        n_heads = 4  # Number of attention heads
        dropout = 0.1  # Dropout probability
        
        self.critic = StructuredAttentionCriticNet(
            state_dim=state_dim,
            action_dim=action_dim,
            embed_dim=embed_dim,
            n_heads=n_heads,
            hidden_dims=critic_hidden_dims,
            dropout=dropout
        )
        
        # === Initialize entropy adjustment ===
        # Leader and Follower each have an entropy parameter
        self.entroy_leader = MASACEntroy(action_dim=action_dim)
        self.entroy_follower = MASACEntroy(action_dim=action_dim)
        
        # Set alpha clamp range
        self.entroy_leader.min_alpha = self.alpha_min
        self.entroy_leader.max_alpha = self.alpha_max
        self.entroy_follower.min_alpha = self.alpha_min
        self.entroy_follower.max_alpha = self.alpha_max
        
        # Set target entropy
        # 注意：不要再写 if target_entropy < 0 就强制改成 -0.1
        # 否则你传 -2.0 也会被覆盖成 -0.1
        target_entropy = float(target_entropy)
        self.entroy_leader.target_entropy = target_entropy
        self.entroy_follower.target_entropy = target_entropy
        
        self._ensure_entropy_parameters()
        
        # === Initialize optimizers ===
        # Leader Actor optimizer
        self.leader_actor_optimizer = optim.Adam(self.leader_actor.parameters(), lr=policy_lr)
        
        # Follower Actors optimizer (combine all Follower Actor parameters)
        follower_actor_parameters = []
        for actor_instance in self.follower_actors:
            follower_actor_parameters.extend(list(actor_instance.parameters()))
        self.follower_actor_optimizer = optim.Adam(follower_actor_parameters, lr=policy_lr)
        
        # Critic optimizer (includes all Critic components)
        critic_params = []
        critic_params.extend(list(self.critic.leader_encoder.parameters()))
        critic_params.extend(list(self.critic.follower_encoder.parameters()))
        # Update: Use renamed leader_sees_followers_attention
        critic_params.extend(list(self.critic.leader_sees_followers_attention.parameters()))
        # Add: New follower_context_attention
        critic_params.extend(list(self.critic.follower_context_attention.parameters()))
        critic_params.extend(list(self.critic.leader_q_head.parameters()))
        critic_params.extend(list(self.critic.follower_q_head.parameters()))
        
        self.critic_optimizer = optim.Adam(critic_params, lr=value_lr)
        
        # Entropy parameter optimizers
        self.leader_alpha_optimizer = optim.Adam([self.entroy_leader.log_alpha], lr=entropy_lr)
        self.follower_alpha_optimizer = optim.Adam([self.entroy_follower.log_alpha], lr=entropy_lr)
        
        # Initialize noise generators
        self.noises = [Ornstein_Uhlenbeck_Noise(mu=np.zeros(action_dim)) for _ in range(n_agents)]
        
        # Move to specified device
        self.to(self.device)
        
        # Record training state
        self.train_step = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_successes = []
        
        # Initialize replay ratio parameters
        self.min_replay_ratio = 0.1  # Minimum replay ratio
        self.max_replay_ratio = max_replay_ratio  # Maximum replay ratio
        self.replay_ratio = 1.0  # Current replay ratio
        
        # Initialize training step counter
        self.steps_done = 0
        self.max_grad_norm = 10.0
    
    def set_replay_ratio(self, ratio):
        """Set replay ratio
        
        Args:
            ratio: New replay ratio
        """
        if ratio < self.min_replay_ratio:
            ratio = self.min_replay_ratio
            print(f"Replay ratio too small, set to minimum value {self.min_replay_ratio}")
        elif ratio > self.max_replay_ratio:
            ratio = self.max_replay_ratio
            print(f"Replay ratio too large, set to maximum value {self.max_replay_ratio}")
            
        old_ratio = self.replay_ratio
        self.replay_ratio = ratio
        print(f"Replay ratio adjusted from {old_ratio} to {self.replay_ratio}")
    
    def reset_memory(self):
        """Reset replay buffer"""
        obs_dims = {"leader": self.state_dim, "followers": self.state_dim}
        action_dims = {"leader": self.action_dim, "followers": self.action_dim}
        
        self.memory = SMERMemory(
            capacity=self.memory_capacity,
            obs_dims=obs_dims,
            action_dims=action_dims,
            device=self.device
        )
        print(f"Reset SMERMemory buffer, capacity: {self.memory_capacity}")
        
        self._reset_noise()
    
    def _reset_noise(self):
        """Reset noise generators"""
        # Ensure sufficient noise generators
        if len(self.noises) < self.n_agents:
            # Add new noise generators
            while len(self.noises) < self.n_agents:
                self.noises.append(Ornstein_Uhlenbeck_Noise(mu=np.zeros(self.action_dim)))
        else:
            # Reduce noise generators
            self.noises = self.noises[:self.n_agents]
            
        # Reset all noise generator states
        for noise in self.noises:
            noise.reset()
    
    def adapt_to_agent_count(self, n_agents):
        """Adapt to new agent count
        
        Args:
            n_agents: New agent count
        """
        if n_agents == self.n_agents:
            return
                
        print(f"Adapting to agent count change: {self.n_agents} -> {n_agents}")
        
        self.n_agents = n_agents
        num_followers = max(n_agents - 1, 0)
        
        if num_followers > len(self.follower_actors):
            log(f"Warning: Requested follower count ({num_followers}) exceeds pre-allocated follower actor networks ({len(self.follower_actors)})", LOG_WARNING)
            log(f"Will use {len(self.follower_actors)} follower actor networks to handle up to {len(self.follower_actors)} followers", LOG_WARNING)
            log(f"Consider reinitializing MASACController with larger pre-allocated follower network count", LOG_WARNING)
        
        self._reset_noise()
        self.memory_n_agents_version = n_agents
            
        print(f"Successfully adjusted to {n_agents} agents (1 Leader + {min(num_followers, len(self.follower_actors))} available Followers)")

    def _entropy_tensor(self, value):
        if isinstance(value, torch.Tensor):
            tensor = value.detach().clone().to(self.device).float()
        else:
            tensor = torch.tensor(value, device=self.device, dtype=torch.float32)
        if tensor.dim() == 0:
            tensor = tensor.view(1)
        return tensor

    def _set_entropy_log_alpha(self, entroy, value=None):
        source = getattr(entroy, 'log_alpha', torch.zeros(1)) if value is None else value
        tensor = self._entropy_tensor(source)
        current = getattr(entroy, 'log_alpha', None)
        if (
            isinstance(current, nn.Parameter)
            and current.is_leaf
            and current.shape == tensor.shape
            and current.device == tensor.device
        ):
            with torch.no_grad():
                current.copy_(tensor)
            current.requires_grad_(True)
            entroy.log_alpha = current
        else:
            entroy.log_alpha = nn.Parameter(tensor, requires_grad=True)
        # 关键保护逻辑：
        # 1）如果开启自适应 alpha，则按 alpha_min / alpha_max 裁剪
        # 2）如果没有开启自适应 alpha，则强制 alpha = 1
        if self.auto_entropy:
            self._clamp_entropy_alpha(entroy, record_history=False)
        else:
            with torch.no_grad():
                entroy.log_alpha.fill_(0.0)
            entroy.alpha = entroy.log_alpha.detach().exp()
    def _clamp_entropy_alpha(self, entroy, record_history=True):
        """Clamp log_alpha so that alpha stays in [alpha_min, alpha_max]."""
        min_alpha = float(getattr(entroy, 'min_alpha', self.alpha_min))
        max_alpha = float(getattr(entroy, 'max_alpha', self.alpha_max))
    
        if min_alpha <= 0:
            min_alpha = 1e-6
    
        if max_alpha < min_alpha:
            max_alpha = min_alpha
    
        with torch.no_grad():
            lower = float(np.log(min_alpha))
            upper = float(np.log(max_alpha))
            entroy.log_alpha.clamp_(lower, upper)
            entroy.alpha = entroy.log_alpha.detach().exp()
    
        if record_history and hasattr(entroy, 'alpha_history'):
            entroy.alpha_history.append(float(entroy.alpha.item()))
            if len(entroy.alpha_history) > 1000:
                entroy.alpha_history = entroy.alpha_history[-1000:]
    
        return entroy.alpha

    def _ensure_entropy_parameters(self):
        if self.auto_entropy:
            self._set_entropy_log_alpha(self.entroy_leader)
            self._set_entropy_log_alpha(self.entroy_follower)
        else:
            fixed_log_alpha = torch.zeros(1, device=self.device, dtype=torch.float32)
            self._set_entropy_log_alpha(self.entroy_leader, fixed_log_alpha)
            self._set_entropy_log_alpha(self.entroy_follower, fixed_log_alpha)

    def _reset_entropy_optimizers(self):
        self.leader_alpha_optimizer = optim.Adam([self.entroy_leader.log_alpha], lr=self.entropy_lr)
        self.follower_alpha_optimizer = optim.Adam([self.entroy_follower.log_alpha], lr=self.entropy_lr)

    def _entropy_optimizer_uses_param(self, optimizer, param):
        return any(p is param for group in optimizer.param_groups for p in group["params"])

    def _ensure_entropy_optimizers_current(self):
        if not self._entropy_optimizer_uses_param(self.leader_alpha_optimizer, self.entroy_leader.log_alpha):
            self.leader_alpha_optimizer = optim.Adam([self.entroy_leader.log_alpha], lr=self.entropy_lr)
        if not self._entropy_optimizer_uses_param(self.follower_alpha_optimizer, self.entroy_follower.log_alpha):
            self.follower_alpha_optimizer = optim.Adam([self.entroy_follower.log_alpha], lr=self.entropy_lr)

    def _alpha_for(self, entroy, reference_tensor):
        if not isinstance(getattr(entroy, "alpha", None), torch.Tensor):
            entroy.alpha = entroy.log_alpha.detach().exp()
        if entroy.alpha.device != reference_tensor.device or entroy.alpha.dtype != reference_tensor.dtype:
            entroy.alpha = entroy.alpha.to(device=reference_tensor.device, dtype=reference_tensor.dtype)
        return entroy.alpha

    def _entropy_tensors_need_device_sync(self):
        entropy_items = (self.entroy_leader, self.entroy_follower)
        for entroy in entropy_items:
            log_alpha = getattr(entroy, "log_alpha", None)
            alpha = getattr(entroy, "alpha", None)
            if not isinstance(log_alpha, torch.Tensor) or log_alpha.device != self.device:
                return True
            if not isinstance(alpha, torch.Tensor) or alpha.device != self.device:
                return True
        return False

    def _sanitize_training_tensor(self, tensor, name):
        if not torch.isfinite(tensor).all():
            log(
                f"Warning: {name} contains NaN/Inf; skipped this training step.",
                LOG_WARNING,
                throttle=10
            )
            return None
        return tensor

    def _optimizer_step_if_finite(self, optimizer, loss, name):
        optimizer.zero_grad()
        loss.backward()
        params_with_grad = [
            p
            for group in optimizer.param_groups
            for p in group["params"]
            if p.grad is not None
        ]
        try:
            torch.nn.utils.clip_grad_norm_(params_with_grad, self.max_grad_norm, error_if_nonfinite=True)
        except RuntimeError:
            log(f"Warning: {name} produced NaN/Inf gradients; skipped this optimizer step.", LOG_WARNING, throttle=10)
            optimizer.zero_grad(set_to_none=True)
            return False
        optimizer.step()
        return True
    def _array_like_is_finite(self, value):
        """检查 numpy/list/dict/tensor 中是否包含 NaN/Inf。"""
        try:
            if isinstance(value, dict):
                return all(self._array_like_is_finite(v) for v in value.values())
    
            if isinstance(value, (list, tuple)):
                return all(self._array_like_is_finite(v) for v in value)
    
            if torch.is_tensor(value):
                return torch.isfinite(value).all().item()
    
            arr = np.asarray(value, dtype=np.float32)
            return np.isfinite(arr).all()
        except Exception:
            return False
    
    def to(self, device):
        """Move all networks to specified device
        
        Args:
            device: Target device
            
        Returns:
            self: Support chaining
        """
        self.device = device
        
        self.leader_actor = self.leader_actor.to(device)
        self.follower_actors = self.follower_actors.to(device)
        
        self.target_leader_actor = self.target_leader_actor.to(device)
        self.target_follower_actors = self.target_follower_actors.to(device)
        
        self.critic = self.critic.to(device)
        self._ensure_entropy_parameters()
        self._reset_entropy_optimizers()
        
        # Move optimizer state
        optimizers_to_move = [
            self.leader_actor_optimizer,
            self.follower_actor_optimizer,
            self.critic_optimizer,
            self.leader_alpha_optimizer,
            self.follower_alpha_optimizer
        ]
        
        for opt in optimizers_to_move:
            if opt is not None:
                for state in opt.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.to(device)
        
        print(f"MASAC controller (including optimizer state) moved to device: {device}")
        return self
    
    def select_actions(self, observation, add_noise=False, noise_scale=0.1, evaluate=False):
        """Select actions for all agents
        
        Args:
            observation: Structured observation from environment {"leader": obs_leader, "followers": [obs_f1, obs_f2, ...]}
                or legacy flattened state list/array (n_agents, state_dim) or (state_dim * n_agents,)
            add_noise: Whether to add exploration noise
            noise_scale: Noise scaling factor
            evaluate: Whether in evaluation mode
            
        Returns:
            actions: Structured action dict {"leader": action_leader, "followers": [action_f1, action_f2, ...]}
        """
        # Check if input is structured format
        if isinstance(observation, dict) and "leader" in observation and "followers" in observation:
            # Handle structured input
            leader_obs = observation["leader"]
            follower_obs_list = observation["followers"]
            
            # Ensure they are numpy arrays
            leader_obs = np.array(leader_obs, dtype=np.float32)
            follower_obs_list = [np.array(obs, dtype=np.float32) for obs in follower_obs_list]
            
            # Create numpy array and mask for follower observations
            num_followers = len(follower_obs_list)
            if num_followers > 0:
                # Create follower observation array [1, num_followers, state_dim]
                followers_obs_array = np.stack(follower_obs_list).reshape(1, num_followers, -1)
                # Create follower mask [1, num_followers]
                followers_mask = np.ones((1, num_followers), dtype=bool)
            else:
                # If no followers, create empty array
                followers_obs_array = np.zeros((1, 0, self.state_dim), dtype=np.float32)
                followers_mask = np.zeros((1, 0), dtype=bool)
            
            try:
                if isinstance(self.leader_actor, AttentionLeaderActorNet):
                    leader_action = self.leader_actor.choose_action(
                        leader_obs, 
                        followers_obs_array, 
                        followers_mask,
                        evaluate=evaluate
                    )
                else:
                    leader_action = self.leader_actor.choose_action(leader_obs, evaluate=evaluate)
            except Exception as e:
                log(f"Error selecting Leader action: {e}", LOG_ERROR)
                leader_action = np.zeros(self.action_dim)
            
            if add_noise and not evaluate:
                noise = self.noises[0]() * noise_scale
                leader_action += noise
            leader_action = np.clip(leader_action, min_action, max_action)
            
            follower_actions = []
            for i, follower_obs in enumerate(follower_obs_list):
                if i >= len(self.noises):
                    self.noises.append(Ornstein_Uhlenbeck_Noise(mu=np.zeros(self.action_dim)))
                
                try:
                    # If AttentionFollowerActorNet, need to provide context information
                    if isinstance(self.follower_actors[i], AttentionFollowerActorNet):
                        # Prepare Leader context observations
                        leader_obs_for_context = leader_obs.reshape(1, -1)  # [1, state_dim]
                        leader_mask = np.ones((1, 1), dtype=bool)
                        
                        other_followers_indices = [j for j in range(num_followers) if j != i]
                        
                        if other_followers_indices:
                            other_followers_obs = np.stack([follower_obs_list[j] for j in other_followers_indices])
                            other_followers_obs = other_followers_obs.reshape(1, len(other_followers_indices), -1)
                            other_followers_mask = np.ones((1, len(other_followers_indices)), dtype=bool)
                        else:
                            other_followers_obs = np.zeros((1, 0, self.state_dim), dtype=np.float32)
                            other_followers_mask = np.zeros((1, 0), dtype=bool)
                        
                        follower_action = self.follower_actors[i].choose_action(
                            follower_obs,
                            leader_obs_for_context,
                            other_followers_obs,
                            leader_mask,
                            other_followers_mask,
                            evaluate=evaluate
                        )
                    else:
                        follower_action = self.follower_actors[i].choose_action(follower_obs, evaluate=evaluate)
                except Exception as e:
                    log(f"Error selecting Follower {i} action: {e}", LOG_ERROR)
                    follower_action = np.zeros(self.action_dim)
                
                # Add exploration noise
                if add_noise and not evaluate:
                    noise = self.noises[i+1]() * noise_scale
                    follower_action += noise
                follower_action = np.clip(follower_action, min_action, max_action)
                
                follower_actions.append(follower_action)
            
            # Return structured action dict
            return {
                "leader": leader_action,
                "followers": follower_actions
            }
            
        else:
            # Legacy flattened input, maintain compatibility
            # Ensure states is numpy array
            states = observation
            if not isinstance(states, np.ndarray):
                states = np.array(states)
            
            # Handle different input shapes
            if len(states.shape) == 1:
                # Single vector contains all agent states
                # Infer agent count
                inferred_n_agents = states.shape[0] // self.state_dim
                if inferred_n_agents * self.state_dim != states.shape[0]:
                    log(f"Warning: State dimension {states.shape[0]} is not an integer multiple of state_dim {self.state_dim}", LOG_WARNING)
                    inferred_n_agents = max(1, states.shape[0] // self.state_dim)
                    
                if inferred_n_agents != self.n_agents:
                    log(f"Detected agent count change: controller={self.n_agents}, input state={inferred_n_agents}", LOG_INFO)
                    self.adapt_to_agent_count(inferred_n_agents)
                    
                # Split into n_agents individual agent states
                states_list = []
                for i in range(self.n_agents):
                    if i * self.state_dim < states.shape[0]:
                        agent_state = states[i*self.state_dim:(i+1)*self.state_dim]
                        # Ensure correct state dimension
                        if len(agent_state) < self.state_dim:
                            agent_state = np.pad(agent_state, (0, self.state_dim - len(agent_state)))
                        states_list.append(agent_state)
                    else:
                        # If states insufficient, use zero padding
                        states_list.append(np.zeros(self.state_dim))
                    
            elif len(states.shape) == 2:
                actual_n_agents = states.shape[0]
                
                if actual_n_agents != self.n_agents:
                    log(f"Detected agent count change: controller={self.n_agents}, input_state={actual_n_agents}", LOG_INFO)
                    self.adapt_to_agent_count(actual_n_agents)
                
                states_list = [states[i] if i < actual_n_agents else np.zeros(self.state_dim) 
                              for i in range(self.n_agents)]
            else:
                log(f"Error: State shape {states.shape} cannot be parsed as agent states", LOG_ERROR)
                # Return zero actions
                return {
                    "leader": np.zeros(self.action_dim),
                    "followers": [np.zeros(self.action_dim) for _ in range(self.n_agents-1)]
                }
                
            # Select actions for each agent
            leader_action = None
            follower_actions = []
            
            # Get Leader state and all follower states
            leader_state = states_list[0]
            follower_states = states_list[1:] if len(states_list) > 1 else []
            
            # Build follower state array and mask (for attention mechanism)
            if follower_states:
                follower_states_array = np.stack(follower_states).reshape(1, len(follower_states), -1)
                follower_mask = np.ones((1, len(follower_states)), dtype=bool)
            else:
                follower_states_array = np.zeros((1, 0, self.state_dim), dtype=np.float32)
                follower_mask = np.zeros((1, 0), dtype=bool)
            
            if len(states_list) > 0:
                try:
                    if isinstance(self.leader_actor, AttentionLeaderActorNet):
                        leader_action = self.leader_actor.choose_action(
                            leader_state, 
                            follower_states_array, 
                            follower_mask,
                            evaluate=evaluate
                        )
                    else:
                        # Compatible with old LeaderActorNet
                        leader_action = self.leader_actor.choose_action(leader_state, evaluate=evaluate)
                    
                    # Add exploration noise
                    if add_noise and not evaluate:
                        noise = self.noises[0]() * noise_scale
                        leader_action += noise
                    leader_action = np.clip(leader_action, min_action, max_action)
                except Exception as e:
                    log(f"Error selecting Leader action: {e}", LOG_ERROR)
                    leader_action = np.zeros(self.action_dim)
            
            # Followers (remaining agents)
            for i in range(1, len(states_list)):
                follower_idx = i - 1  # Follower index (0-based)
                
                try:
                    # If AttentionFollowerActorNet, need to provide context information
                    if isinstance(self.follower_actors[follower_idx], AttentionFollowerActorNet):
                        # Prepare Leader context observation
                        leader_obs_for_context = leader_state.reshape(1, -1)  # [1, state_dim]
                        leader_mask = np.ones((1, 1), dtype=bool)  # [1, 1]
                        
                        # Prepare other follower context observation
                        other_followers_indices = [j-1 for j in range(1, len(states_list)) if j != i]
                        
                        if other_followers_indices:
                            other_followers_obs = np.stack([states_list[j+1] for j in other_followers_indices])
                            other_followers_obs = other_followers_obs.reshape(1, len(other_followers_indices), -1)
                            other_followers_mask = np.ones((1, len(other_followers_indices)), dtype=bool)
                        else:
                            other_followers_obs = np.zeros((1, 0, self.state_dim), dtype=np.float32)
                            other_followers_mask = np.zeros((1, 0), dtype=bool)
                        
                        # Call follower Actor network
                        follower_action = self.follower_actors[follower_idx].choose_action(
                            states_list[i],
                            leader_obs_for_context,
                            other_followers_obs,
                            leader_mask,
                            other_followers_mask,
                            evaluate=evaluate
                        )
                    else:
                        # Compatible with old FollowerActorNet
                        follower_action = self.follower_actors[follower_idx].choose_action(states_list[i], evaluate=evaluate)
                    
                    # Add exploration noise
                    if add_noise and not evaluate:
                        # Ensure sufficient noise generators
                        if i >= len(self.noises):
                            self.noises.append(Ornstein_Uhlenbeck_Noise(mu=np.zeros(self.action_dim)))
                        
                        noise = self.noises[i]() * noise_scale
                        follower_action += noise
                    follower_action = np.clip(follower_action, min_action, max_action)
                    
                    follower_actions.append(follower_action)
                except Exception as e:
                    log(f"Error selecting Follower {i} action: {e}", LOG_ERROR)
                    follower_actions.append(np.zeros(self.action_dim))

            return {
                "leader": leader_action,
                "followers": follower_actions
            }
    
    def store_transition(self, states, actions, rewards, next_states, done=False, current_stage_tag: str = "default_stage"):
        """Store transition to replay buffer
        
        Args:
            states: Current state array [n_agents, state_dim] or structured dict
            actions: Action array [n_agents, action_dim] or structured dict
            rewards: Reward array [n_agents] or structured dict
            next_states: Next state array [n_agents, state_dim] or structured dict
            done: Done flag
            current_stage_tag: Current curriculum stage tag for marking experience stage
        """
        # 如果环境状态、动作或奖励中已经出现 NaN/Inf，不写入经验池
        if not (
            self._array_like_is_finite(states)
            and self._array_like_is_finite(actions)
            and self._array_like_is_finite(rewards)
            and self._array_like_is_finite(next_states)
        ):
            log(
                "store_transition: detected NaN/Inf in transition; skipped storing this sample.",
                LOG_WARNING,
                throttle=10
            )
            return
        if (isinstance(states, dict) and "leader" in states and "followers" in states and
            isinstance(actions, dict) and "leader" in actions and "followers" in actions and
            isinstance(rewards, dict) and "leader" in rewards and "followers" in rewards and
            isinstance(next_states, dict) and "leader" in next_states and "followers" in next_states):
            self.memory.store_transition(states, actions, rewards, next_states, done, stage_tag=current_stage_tag)
            return
            
        states = np.asarray(states)
        actions = np.asarray(actions)
        rewards = np.asarray(rewards)
        next_states = np.asarray(next_states)
        
        n_agents = states.shape[0]
        
        if n_agents == 0:
            log("store_transition: Received 0 agent states, skipping storage.", LOG_WARNING)
            return

        leader_state = states[0]
        follower_states = states[1:] if n_agents > 1 else []

        leader_action = actions[0]
        follower_actions = actions[1:] if n_agents > 1 else []

        leader_reward = rewards[0] 
        follower_rewards = rewards[1:] if n_agents > 1 else []
        
        leader_next_state = next_states[0]
        follower_next_states = next_states[1:] if n_agents > 1 else []

        observation = {"leader": leader_state, "followers": list(follower_states)}
        action = {"leader": leader_action, "followers": list(follower_actions)}
        reward = {"leader": leader_reward, "followers": list(follower_rewards)}
        next_observation = {"leader": leader_next_state, "followers": list(follower_next_states)}
        
        self.memory.store_transition(observation, action, reward, next_observation, done, stage_tag=current_stage_tag)
    
    def train(self, batch_size=None, current_stage_tag: str = "default_stage", current_stage_number: int = 0):
        """Train networks

        Args:
            batch_size: Batch size, use self.batch_size if None
            current_stage_tag: Current curriculum stage tag for distinguishing old/new experiences
            current_stage_number: Current curriculum stage number for calculating old/new experience sampling ratio
        """
        if batch_size is None:
            batch_size = self.batch_size

        if self._entropy_tensors_need_device_sync():
            self._ensure_entropy_parameters()
            self._ensure_entropy_optimizers_current()

        sampled_data = self.memory.sample(
            batch_size,
            current_stage_tag=current_stage_tag,
            current_stage_number=current_stage_number
        )
        
        if sampled_data is None:
            log(f"MASACController: Stage '{current_stage_tag}' (Number {current_stage_number}) skipped training step due to insufficient samples or sampling error.", LOG_DEBUG)
            return  

        batch_data, batch_masks = sampled_data
        
        # Get leader and followers data
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

        tensors_to_check = [
            (obs_leader, "obs_leader"),
            (obs_followers, "obs_followers"),
            (next_obs_leader, "next_obs_leader"),
            (next_obs_followers, "next_obs_followers"),
            (act_leader, "act_leader"),
            (act_followers, "act_followers"),
            (reward_leader, "reward_leader"),
            (reward_followers, "reward_followers"),
            (done, "done"),
            (mask_followers, "mask_followers"),
        ]
        
        checked_tensors = [
            self._sanitize_training_tensor(t, name)
            for t, name in tensors_to_check
        ]
        
        if any(t is None for t in checked_tensors):
            return
        
        (
            obs_leader,
            obs_followers,
            next_obs_leader,
            next_obs_followers,
            act_leader,
            act_followers,
            reward_leader,
            reward_followers,
            done,
            mask_followers
        ) = checked_tensors
        
        act_leader = torch.clamp(act_leader, min_action, max_action)
        act_followers = torch.clamp(act_followers, min_action, max_action)
        done = torch.clamp(done, 0.0, 1.0)
        mask_followers = torch.clamp(mask_followers, 0.0, 1.0)
        
        # Get number of followers (batch may have different number of followers)
        B, max_F, _ = obs_followers.shape
        num_active_followers = min(max_F, len(self.follower_actors))
        
        # ===== Calculate Critic loss =====
        # 1. Calculate target Q value
        with torch.no_grad():
            # Evaluate leader's next state action
            if isinstance(self.target_leader_actor, AttentionLeaderActorNet):
                # Use attention-based Leader Actor network
                next_act_leader, next_log_prob_leader = self.target_leader_actor.evaluate(
                    next_obs_leader, next_obs_followers, mask_followers
                )
            else:
                # Use legacy Leader Actor network
                next_act_leader, next_log_prob_leader = self.target_leader_actor.evaluate(next_obs_leader)
            
            # Evaluate followers' next state actions
            next_act_followers_list = []
            next_log_prob_followers_list = []
            
            for k in range(num_active_followers):
                # Prepare context data for follower k
                follower_self_obs_k = next_obs_followers[:, k, :]  # [B, state_dim]
                leader_for_context_k = next_obs_leader  # [B, state_dim]
                valid_leader_mask_k = torch.ones(B, 1, dtype=torch.bool, device=self.device)  # [B, 1]
                
                # Prepare other followers' context
                other_follower_indices = [j for j in range(max_F) if j != k]
                if other_follower_indices:
                    other_followers_context_k = next_obs_followers[:, other_follower_indices, :]  # [B, max_F-1, state_dim]
                    valid_other_followers_mask_k = mask_followers[:, other_follower_indices]  # [B, max_F-1]
                else:
                    other_followers_context_k = torch.zeros(B, 0, self.state_dim, device=self.device)
                    valid_other_followers_mask_k = torch.zeros(B, 0, dtype=torch.bool, device=self.device)
                
                if isinstance(self.target_follower_actors[k], AttentionFollowerActorNet):
                    # Use attention-based Follower Actor network
                    act_k, log_prob_k = self.target_follower_actors[k].evaluate(
                        follower_self_obs_k,
                        leader_for_context_k,
                        other_followers_context_k,
                        valid_leader_mask_k,
                        valid_other_followers_mask_k
                    )
                else:
                    # Use legacy Follower Actor network
                    act_k, log_prob_k = self.target_follower_actors[k].evaluate(follower_self_obs_k)
                
                next_act_followers_list.append(act_k.unsqueeze(1))  # Add follower dimension [B, 1, action_dim]
                next_log_prob_followers_list.append(log_prob_k.unsqueeze(1))  # [B, 1, 1]
            
            if num_active_followers < max_F:
                # Create padding actions
                action_padding = torch.zeros(
                    B, max_F - num_active_followers, self.action_dim, 
                    device=self.device
                )
                # Create padding log probabilities
                log_prob_padding = torch.zeros(
                    B, max_F - num_active_followers, 1, 
                    device=self.device
                )
                
                # Concatenate with padding if there are actual follower actions
                if next_act_followers_list:
                    next_act_followers = torch.cat(next_act_followers_list + [action_padding], dim=1)
                    next_log_prob_followers = torch.cat(next_log_prob_followers_list + [log_prob_padding], dim=1)
                else:
                    # If no actual follower actions, use padding directly
                    next_act_followers = action_padding
                    next_log_prob_followers = log_prob_padding
            else:
                # If there are enough follower Actor networks, concatenate all results directly
                if next_act_followers_list:
                    next_act_followers = torch.cat(next_act_followers_list, dim=1)  # [B, max_F, action_dim]
                    next_log_prob_followers = torch.cat(next_log_prob_followers_list, dim=1)  # [B, max_F, 1]
                else:
                    # Edge case: no followers
                    next_act_followers = torch.zeros(B, 0, self.action_dim, device=self.device)
                    next_log_prob_followers = torch.zeros(B, 0, 1, device=self.device)
            
            # Calculate target Q values
            target_q1_leader, target_q2_leader, target_q1_followers, target_q2_followers = self.critic.forward_target(
                next_obs_leader, next_obs_followers, 
                next_act_leader, next_act_followers,
                mask_followers
            )
            
            # Use min Q
            target_q_leader = torch.min(target_q1_leader, target_q2_leader)
            target_q_followers = torch.min(target_q1_followers, target_q2_followers)
            alpha_leader = self._alpha_for(self.entroy_leader, next_log_prob_leader)
            alpha_follower = self._alpha_for(self.entroy_follower, next_log_prob_followers)
            
            # Calculate target value (reward + gamma * (Q - alpha * log_prob))
            # Leader target
            target_leader = reward_leader + self.gamma * (1 - done) * (
                target_q_leader - alpha_leader * next_log_prob_leader
            )
            
            # Followers target (apply mask)
            target_followers = reward_followers + self.gamma * (1 - done).unsqueeze(1) * (
                target_q_followers - alpha_follower * next_log_prob_followers
            )
        
        # 2. Calculate current Q values
        current_q1_leader, current_q2_leader, current_q1_followers, current_q2_followers = self.critic(
            obs_leader, obs_followers,
            act_leader, act_followers,
            mask_followers
        )
        
        # 3. Calculate Critic loss (MSE)
        # Leader loss
        critic_loss_leader = F.mse_loss(current_q1_leader, target_leader) + F.mse_loss(current_q2_leader, target_leader)
        
        # Followers loss (apply mask)
        # First, calculate element-wise MSE loss
        critic_loss_followers_q1 = F.mse_loss(
            current_q1_followers, target_followers, reduction='none'
        )
        critic_loss_followers_q2 = F.mse_loss(
            current_q2_followers, target_followers, reduction='none'
        )
        
        # Apply mask and calculate average
        # Ensure mask shape is correct [B, max_F, 1]
        mask_3d = mask_followers.unsqueeze(-1)
        critic_loss_followers_q1 = (critic_loss_followers_q1 * mask_3d).sum() / (mask_3d.sum() + 1e-8)
        critic_loss_followers_q2 = (critic_loss_followers_q2 * mask_3d).sum() / (mask_3d.sum() + 1e-8)
        
        critic_loss_followers = critic_loss_followers_q1 + critic_loss_followers_q2
        
        # Total Critic loss
        critic_loss = critic_loss_leader + critic_loss_followers
        
        # Update Critic
        if not self._optimizer_step_if_finite(self.critic_optimizer, critic_loss, "critic_loss"):
            return
        
        # ===== Calculate Actor loss =====
        # Leader Actor loss
        # 1. Generate new actions
        if isinstance(self.leader_actor, AttentionLeaderActorNet):
            # Use attention Leader Actor network
            new_act_leader, log_prob_leader = self.leader_actor.evaluate(
                obs_leader, obs_followers, mask_followers
            )
        else:
            # Use legacy Leader Actor network
            new_act_leader, log_prob_leader = self.leader_actor.evaluate(obs_leader)
        
        # 2. Calculate Q values
        q1_leader, q2_leader, _, _ = self.critic(
            obs_leader, obs_followers,
            new_act_leader, act_followers,  # Use new Leader action, keep Followers actions unchanged
            mask_followers
        )
        min_q_leader = torch.min(q1_leader, q2_leader)
        
        # 3. Calculate Leader Actor loss (policy gradient, maximize Q - alpha * log_prob)
        actor_loss_leader = (self._alpha_for(self.entroy_leader, log_prob_leader) * log_prob_leader - min_q_leader).mean()
        
        # Update Leader Actor
        if not self._optimizer_step_if_finite(self.leader_actor_optimizer, actor_loss_leader, "actor_loss_leader"):
            return
        
        # Follower Actors loss
        # Generate new actions for each follower
        new_act_followers_list = []
        log_prob_followers_list = []
        
        for k in range(num_active_followers):
            # Prepare follower k's context data
            follower_self_obs_k = obs_followers[:, k, :]  # [B, state_dim]
            leader_for_context_k = obs_leader  # [B, state_dim]
            valid_leader_mask_k = torch.ones(B, 1, dtype=torch.bool, device=self.device)  # [B, 1]
            
            # Prepare other followers context
            other_follower_indices = [j for j in range(max_F) if j != k]
            if other_follower_indices:
                other_followers_context_k = obs_followers[:, other_follower_indices, :]  # [B, max_F-1, state_dim]
                valid_other_followers_mask_k = mask_followers[:, other_follower_indices]  # [B, max_F-1]
            else:
                other_followers_context_k = torch.zeros(B, 0, self.state_dim, device=self.device)
                valid_other_followers_mask_k = torch.zeros(B, 0, dtype=torch.bool, device=self.device)
            
            if isinstance(self.follower_actors[k], AttentionFollowerActorNet):
                # Use attention Follower Actor network
                act_k, log_prob_k = self.follower_actors[k].evaluate(
                    follower_self_obs_k,
                    leader_for_context_k,
                    other_followers_context_k,
                    valid_leader_mask_k,
                    valid_other_followers_mask_k
                )
            else:
                # Use legacy Follower Actor network
                act_k, log_prob_k = self.follower_actors[k].evaluate(follower_self_obs_k)
            
            # Collect results
            new_act_followers_list.append(act_k.unsqueeze(1))  # Add follower dimension [B, 1, action_dim]
            log_prob_followers_list.append(log_prob_k.unsqueeze(1))  # [B, 1, 1]
        
        # Pad with zeros if not enough follower Actor networks
        if num_active_followers < max_F:
            # Create padding actions
            action_padding = torch.zeros(
                B, max_F - num_active_followers, self.action_dim, 
                device=self.device
            )
            # Create padding log probabilities
            log_prob_padding = torch.zeros(
                B, max_F - num_active_followers, 1, 
                device=self.device
            )
            
            # Concatenate with padding if there are actual follower actions
            if new_act_followers_list:
                new_act_followers = torch.cat(new_act_followers_list + [action_padding], dim=1)
                log_prob_followers = torch.cat(log_prob_followers_list + [log_prob_padding], dim=1)
            else:
                # If no actual follower actions, use padding directly
                new_act_followers = action_padding
                log_prob_followers = log_prob_padding
        else:
            # If there are enough follower Actor networks, concatenate all results directly
            if new_act_followers_list:
                new_act_followers = torch.cat(new_act_followers_list, dim=1)  # [B, max_F, action_dim]
                log_prob_followers = torch.cat(log_prob_followers_list, dim=1)  # [B, max_F, 1]
            else:
                # Edge case: no followers
                new_act_followers = torch.zeros(B, 0, self.action_dim, device=self.device)
                log_prob_followers = torch.zeros(B, 0, 1, device=self.device)
        
        # 2. Calculate Q values
        _, _, q1_followers, q2_followers = self.critic(
            obs_leader, obs_followers,
            act_leader, new_act_followers,  # Use new Followers actions, keep Leader action unchanged
            mask_followers
        )
        min_q_followers = torch.min(q1_followers, q2_followers)
        
        # 3. Calculate Follower Actor loss (policy gradient, maximize Q - alpha * log_prob)
        # Apply mask
        actor_loss_followers = (self._alpha_for(self.entroy_follower, log_prob_followers) * log_prob_followers - min_q_followers) * mask_3d
        # Average over masked loss
        actor_loss_followers = actor_loss_followers.sum() / (mask_3d.sum() + 1e-8)
        
        # Update Follower Actor
        if not self._optimizer_step_if_finite(self.follower_actor_optimizer, actor_loss_followers, "actor_loss_followers"):
            return
        
        # ===== Update entropy weight alpha =====
        if self.auto_entropy:
            # Leader alpha
            alpha_loss_leader = -(self.entroy_leader.log_alpha * (
                log_prob_leader.detach() + self.entroy_leader.target_entropy
            )).mean()
        
            if self._optimizer_step_if_finite(
                self.leader_alpha_optimizer,
                alpha_loss_leader,
                "alpha_loss_leader"
            ):
                self._clamp_entropy_alpha(self.entroy_leader, record_history=True)
        
            # Follower alpha
            alpha_loss_followers = -(self.entroy_follower.log_alpha * (
                log_prob_followers.detach() + self.entroy_follower.target_entropy
            )) * mask_3d
        
            alpha_loss_followers = alpha_loss_followers.sum() / (mask_3d.sum() + 1e-8)
        
            if self._optimizer_step_if_finite(
                self.follower_alpha_optimizer,
                alpha_loss_followers,
                "alpha_loss_followers"
            ):
                self._clamp_entropy_alpha(self.entroy_follower, record_history=True)
        
        else:
            # 固定 alpha 模式：强制保持 alpha = 1
            with torch.no_grad():
                self.entroy_leader.log_alpha.fill_(0.0)
                self.entroy_follower.log_alpha.fill_(0.0)
        
            self.entroy_leader.alpha = torch.ones_like(self.entroy_leader.log_alpha).detach()
            self.entroy_follower.alpha = torch.ones_like(self.entroy_follower.log_alpha).detach()
                        
        # ===== Soft update target networks =====
        # Update Critic target network
        self.critic.soft_update(self.tau)
        
        # Update Leader Actor target network
        for target_param, param in zip(self.target_leader_actor.parameters(), self.leader_actor.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
        
        # Update Follower Actors target networks
        for i in range(len(self.follower_actors)):
            for target_param, param in zip(self.target_follower_actors[i].parameters(), self.follower_actors[i].parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
            
        # Update training step count
        self.train_step += 1
    def save_models(self, path):
        """Save model
        
        Args:
            path: Save path
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save network parameters
        torch.save({
            # Actor network parameters
            'leader_actor': self.leader_actor.state_dict(),
            'follower_actors': [actor.state_dict() for actor in self.follower_actors],
            'target_leader_actor': self.target_leader_actor.state_dict(),
            'target_follower_actors': [actor.state_dict() for actor in self.target_follower_actors],
            
            # Critic network parameters
            'critic': self.critic.state_dict(),
            
            # Entropy parameters
            'entroy_leader': self.entroy_leader.__dict__,
            'entroy_follower': self.entroy_follower.__dict__,
            'auto_entropy': self.auto_entropy,
            
            # Optimizer parameters
            'leader_actor_optimizer': self.leader_actor_optimizer.state_dict(),
            'follower_actor_optimizer': self.follower_actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'leader_alpha_optimizer': self.leader_alpha_optimizer.state_dict(),
            'follower_alpha_optimizer': self.follower_alpha_optimizer.state_dict(),
            
            # Training statistics
            'train_step': self.train_step,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'episode_successes': self.episode_successes,
            
            # Configuration
            'n_agents': self.n_agents,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'gamma': self.gamma,
            'tau': self.tau,
            'batch_size': self.batch_size,
            'memory_capacity': self.memory_capacity,
            'replay_ratio': self.replay_ratio
        }, path)
        
        print(f"Model saved to {path}")
    
    def load_models(self, path, strict=False):
        """Load model
        
        Args:
            path: Model path
            strict: Whether to strictly load (if False, allows partial parameter mismatch)
            
        Returns:
            bool: Whether successfully loaded
        """
        if not os.path.exists(path):
            print(f"Model file does not exist: {path}")
            return False
            
        try:
            checkpoint = torch.load(path, map_location=self.device)
            
            # Load Actor network parameters
            self.leader_actor.load_state_dict(checkpoint['leader_actor'], strict=strict)
            for i, actor in enumerate(self.follower_actors):
                actor.load_state_dict(checkpoint['follower_actors'][i], strict=strict)
            
            # Load target Actor network parameters
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
            
            # Load Critic network parameters
            self.critic.load_state_dict(checkpoint['critic'], strict=strict)
            
            # Load entropy parameters
            if 'entroy_leader' in checkpoint:
                # Handle tensors
                for key, value in checkpoint['entroy_leader'].items():
                    if key == 'log_alpha':
                        self._set_entropy_log_alpha(self.entroy_leader, value)
                    elif key == 'alpha':
                        continue
                    elif isinstance(value, torch.Tensor):
                        setattr(self.entroy_leader, key, value.to(self.device))
                    else:
                        setattr(self.entroy_leader, key, value)
                        
            if 'entroy_follower' in checkpoint:
                # Handle tensors
                for key, value in checkpoint['entroy_follower'].items():
                    if key == 'log_alpha':
                        self._set_entropy_log_alpha(self.entroy_follower, value)
                    elif key == 'alpha':
                        continue
                    elif isinstance(value, torch.Tensor):
                        setattr(self.entroy_follower, key, value.to(self.device))
                    else:
                        setattr(self.entroy_follower, key, value)
            self._ensure_entropy_parameters()
            self._reset_entropy_optimizers()
            
            # Load optimizer parameters (if exists)
            if 'leader_actor_optimizer' in checkpoint:
                self.leader_actor_optimizer.load_state_dict(checkpoint['leader_actor_optimizer'])
                
            if 'follower_actor_optimizer' in checkpoint:
                self.follower_actor_optimizer.load_state_dict(checkpoint['follower_actor_optimizer'])
                
            if 'critic_optimizer' in checkpoint:
                self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
                
            if self.auto_entropy and 'leader_alpha_optimizer' in checkpoint:
                self.leader_alpha_optimizer.load_state_dict(checkpoint['leader_alpha_optimizer'])
                
            if self.auto_entropy and 'follower_alpha_optimizer' in checkpoint:
                self.follower_alpha_optimizer.load_state_dict(checkpoint['follower_alpha_optimizer'])
            
            # Load training statistics (if exists)
            if 'train_step' in checkpoint:
                self.train_step = checkpoint['train_step']
                
            if 'episode_rewards' in checkpoint:
                self.episode_rewards = checkpoint['episode_rewards']
                
            if 'episode_lengths' in checkpoint:
                self.episode_lengths = checkpoint['episode_lengths']
                
            if 'episode_successes' in checkpoint:
                self.episode_successes = checkpoint['episode_successes']
            
            # Load configuration (if exists)
            if 'n_agents' in checkpoint:
                self.n_agents = checkpoint['n_agents']
                
            if 'replay_ratio' in checkpoint:
                self.replay_ratio = checkpoint['replay_ratio']
            
            print(f"Successfully loaded model: {path}")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def track_episode_rewards(self, rewards):
        """Track episode rewards
        
        Args:
            rewards: List or array of rewards for each agent
        """
        # Calculate total reward (sum of all agent rewards)
        total_reward = sum(rewards) if isinstance(rewards, (list, np.ndarray)) else rewards
        self.episode_rewards.append(total_reward)
        
    def track_episode_length(self, length):
        """Track episode length
        
        Args:
            length: Number of steps in episode
        """
        self.episode_lengths.append(length)
        
    def track_episode_success(self, success):
        """Track episode success flag
        
        Args:
            success: Whether task completed successfully
        """
        self.episode_successes.append(1 if success else 0)
        
    def get_training_stats(self, window=100):
        """Get training statistics
        
        Args:
            window: Window size for calculating averages
            
        Returns:
            dict: Statistics dictionary
        """
        stats = {}
        
        # Calculate average reward
        if len(self.episode_rewards) > 0:
            stats['last_reward'] = self.episode_rewards[-1]
            stats['avg_reward'] = np.mean(self.episode_rewards[-window:])
            
        # Calculate average step length
        if len(self.episode_lengths) > 0:
            stats['last_length'] = self.episode_lengths[-1]
            stats['avg_length'] = np.mean(self.episode_lengths[-window:])
            
        # Calculate success rate
        if len(self.episode_successes) > 0:
            stats['last_success'] = self.episode_successes[-1]
            stats['success_rate'] = np.mean(self.episode_successes[-window:])
            
        # Training step
        stats['train_step'] = self.train_step
        
        return stats

    def get_policy_parameters_for_curriculum(self):
        """Export policy and Critic parameters for curriculum learning knowledge transfer"""
        # Create dictionary to save follower Actor parameters
        follower_actors_dict = {}
        for i, actor in enumerate(self.follower_actors):
            follower_actors_dict[f'actor_{i+1}'] = actor.state_dict()
        
        params = {
            'actors': {
                'actor_0': self.leader_actor.state_dict(),
                **follower_actors_dict  # Add all follower Actor parameters
            },
            'entropy': {
                'entropy_0': {
                    'log_alpha': self.entroy_leader.log_alpha.data.clone(),
                    'target_entropy': self.entroy_leader.target_entropy,
                    'alpha': self.entroy_leader.alpha.data.clone()
                },
                'entropy_1': {
                    'log_alpha': self.entroy_follower.log_alpha.data.clone(),
                    'target_entropy': self.entroy_follower.target_entropy,
                    'alpha': self.entroy_follower.alpha.data.clone()
                }
            },
            'critic': self.critic.state_dict(), # Export Critic parameters
            'num_follower_actors': len(self.follower_actors)  # Add follower Actor count information
        }
        log("Exported parameters for curriculum transfer from MASACController (includes Critic and multiple Follower Actors)", LOG_DEBUG)
        return params
        
    def update_components_from_transfer(self, transferred_params: Dict[str, Any]):
        """从课程学习迁移的参数更新控制器组件"""
        log("开始从迁移的参数更新MASACController组件...", LOG_INFO)
        updated_components = []
        try:
            # 检查是否有智能体数量信息
            if 'agent_counts' in transferred_params:
                agent_counts = transferred_params['agent_counts']
                log(f"检测到智能体数量变化信息:", LOG_INFO)
                log(f"  - 源任务: {agent_counts.get('source', 'N/A')} 个智能体", LOG_INFO)
                log(f"  - 目标任务: {agent_counts.get('target', 'N/A')} 个智能体", LOG_INFO)
                log(f"  - 源任务从机数: {agent_counts.get('source_followers', 'N/A')}", LOG_INFO)
                log(f"  - 目标任务从机数: {agent_counts.get('target_followers', 'N/A')}", LOG_INFO)
            
            # 更新 Actor 参数
            if 'actors' in transferred_params:
                actor_params = transferred_params['actors']
                
                # 更新Leader Actor
                if 'actor_0' in actor_params:
                    self.leader_actor.load_state_dict(actor_params['actor_0'])
                    self.target_leader_actor.load_state_dict(self.leader_actor.state_dict()) # 更新目标网络
                    updated_components.append("Leader Actor")
                
                # 更新Follower Actors
                for i in range(len(self.follower_actors)):
                    follower_key = f'actor_{i+1}'
                    if follower_key in actor_params:
                        self.follower_actors[i].load_state_dict(actor_params[follower_key])
                        self.target_follower_actors[i].load_state_dict(self.follower_actors[i].state_dict()) # 更新目标网络
                        
                        # 检查是否是复用的参数
                        if 'agent_counts' in transferred_params:
                            source_followers = transferred_params['agent_counts'].get('source_followers', 0)
                            if i >= source_followers:
                                log(f"  - Follower Actor {i+1} 使用了从最后一个已有从机复用的参数", LOG_INFO)
                        
                        updated_components.append(f"Follower Actor {i+1}")
                    else:
                        # 如果参数中没有对应的actor，说明这是新增的从机，应该由知识迁移系统处理了复用
                        log(f"  - Follower Actor {i+1} 在迁移参数中未找到（可能是新增的从机）", LOG_WARNING)

            # 更新 Entropy 参数
            if 'entropy' in transferred_params:
                entropy_params = transferred_params['entropy']
                
                # 更新Leader Entropy
                if 'entropy_0' in entropy_params:
                    leader_entropy = entropy_params['entropy_0']
                    for key, value in leader_entropy.items():
                        if key == 'log_alpha':
                            self._set_entropy_log_alpha(self.entroy_leader, value)
                        elif key == 'alpha':
                            continue
                        elif isinstance(value, torch.Tensor):
                            setattr(self.entroy_leader, key, value.detach().clone().to(self.device))
                        else:
                            setattr(self.entroy_leader, key, value)
                    
                    # 确保重新设置优化器
                    self.leader_alpha_optimizer = optim.Adam([self.entroy_leader.log_alpha], lr=self.entropy_lr)
                    updated_components.append("Leader Entropy")
                
                # 更新Follower Entropy
                if 'entropy_1' in entropy_params:
                    follower_entropy = entropy_params['entropy_1']
                    for key, value in follower_entropy.items():
                        if key == 'log_alpha':
                            self._set_entropy_log_alpha(self.entroy_follower, value)
                        elif key == 'alpha':
                            continue
                        elif isinstance(value, torch.Tensor):
                            setattr(self.entroy_follower, key, value.detach().clone().to(self.device))
                        else:
                            setattr(self.entroy_follower, key, value)
                    
                    # 确保重新设置优化器
                    self.follower_alpha_optimizer = optim.Adam([self.entroy_follower.log_alpha], lr=self.entropy_lr)
                    updated_components.append("Follower Entropy (所有从机共享)")
                if not self.auto_entropy:
                    self._ensure_entropy_parameters()
                    self._reset_entropy_optimizers()
            
            # 更新 Critic 参数
            if 'critic' in transferred_params:
                self.critic.load_state_dict(transferred_params['critic'])
                updated_components.append("Critic")
            
            log(f"已成功更新MASACController组件: {', '.join(updated_components)}", LOG_INFO)
            
        except Exception as e:
            log(f"更新组件时出错: {e}", LOG_ERROR)
            import traceback
            traceback.print_exc()

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
                # 检查是否有JSON结果文件（兼容带时间戳的新命名）
                info_file = _find_first_prefixed_file(item_path, "test_info", ".json")
                
                if info_file and os.path.exists(info_file):
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
            img_path = os.path.relpath(os.path.join(test['path'], img), TEST_RESULTS_BASE).replace('\\', '/')
            img_links += f'<a href="{img_path}" class="img-link" target="_blank">{img}</a>'
        
        # 详细信息链接
        info_file = _find_first_prefixed_file(test['path'], "test_info", ".json") or os.path.join(test['path'], "test_info.json")
        result_file = _find_first_prefixed_file(test['path'], "test_results", ".json") or os.path.join(test['path'], "test_results.json")
        info_link = os.path.relpath(info_file, TEST_RESULTS_BASE).replace('\\', '/')
        result_link = os.path.relpath(result_file, TEST_RESULTS_BASE).replace('\\', '/')
        
        html_content += f"""
            <tr>
                <td>{test.get('date', test.get('timestamp', 'N/A'))}</td>
                <td>{config_str}</td>
                <td>{test.get('success_rate', 'N/A')}</td>
                <td>{img_links}</td>
                <td>
                    <a href="{info_link}" target="_blank">测试信息</a> | 
                    <a href="{result_link}" target="_blank">详细结果</a>
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

def run_with_curriculum(args, initial_n_agent, initial_m_enemy, seed=42, run_id=None):
    """使用课程学习训练MASAC
    
    Args:
        args: 命令行参数
        initial_n_agent: 初始友方数量
        initial_m_enemy: 初始敌方数量
        seed: 随机种子
        run_id: 运行ID，用于多次训练时区分不同的运行
    """
    # 设置随机种子
    set_seed(seed)
    print(f"设置随机种子: {seed}")
    if run_id is not None:
        print(f"训练运行ID: {run_id}")
    
    # 确保结果目录存在
    global RESULTS_DIR, TEST_RESULTS_BASE, TRAINING_RESULTS_FILE, RENDER
    RENDER = False
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
    
    # 模型保存在当前运行目录下，避免覆盖其他训练/测试模式
    model_scope = f"run{run_id}_seed{seed}" if run_id is not None else None
    models_base_dir = os.path.join(RESULTS_DIR, "model", model_scope) if model_scope else os.path.join(RESULTS_DIR, "model")
    
    # 创建模型基础目录；不删除旧模型，每次训练目录本身带时间戳
    os.makedirs(models_base_dir, exist_ok=True)
    log(f"模型目录已准备好: {models_base_dir}", LOG_INFO)
    
    # 设置日志级别
    
    # 清除之前的日志历史
    if hasattr(globals(), 'clear_log_history'):
        clear_log_history()
    
    # 确保模型保存目录存在
    log(f"模型将保存在: {models_base_dir}", LOG_INFO)
    
    # 创建课程学习组件
    config = CurriculumConfig()
    
    # 修改变化范围配置
    config.set("task_generator.variation_ranges", {
        "hero_count": (1, 3),       # 友方无人机数量1-3
        "enemy_count": (1, 5),      # 敌方无人机数量1-5
        "obstacle_count": (0, 10),   # 障碍物数量0-10
        "map_size": (700, 1000),     # 地图尺寸
        "target_distance": (200, 600), # 目标距离
        "uav_speed": (10, 20)       # 无人机速度10-20
    })
    
    print("智能体奖励处理说明:")
    print("- 环境返回的奖励包含所有智能体 (友方+敌方) 的奖励")
    print("- 控制器会根据实际友方智能体数量自动调整，并从中提取所需的奖励")
    print("- 当智能体数量从1主机+1从机变为1主机+2从机时，控制器会自动适应")
    
    # 增加课程步骤和每个任务的训练轮数
    config.set("curriculum.max_curriculum_steps", 6)  # 从15减少到2
    config.set("curriculum_manager.max_episodes_per_task", 300)  # 设置为200
    
    # 设置评估窗口大小和稳定性阈值
    config.set("curriculum_manager.evaluation_window", 20)  # 从20减少到5
    config.set("curriculum_manager.min_training_rounds", 60)  # 从40减少到5
    config.set("curriculum_manager.reward_stability_threshold", 0.6)  # 从0.75降低到0.5
    config.set("curriculum_manager.success_rate_threshold", 0.8)  # 从0.8降低到0.5

    # 添加停滞检测相关配置
    config.set("curriculum_manager.progress_threshold", 0.05)  # 学习进度阈值(从0.01提高到0.05)
    config.set("curriculum_manager.stagnation_threshold", 3)  # 连续停滞检测阈值
    
    # 在配置中设置渲染选项，确保所有任务都使用同样的渲染设置
    config.set("render", RENDER)
    print(f"渲染设置: {'开启' if RENDER else '关闭'}")
    
    # 修改初始难度范围，使其更广泛
    config.set("curriculum_manager.initial_difficulty_range", (0.0, 0.3))  # 从(0.0, 0.1)扩展到(0.0, 0.3)
    
    # 设置固定任务的相关配置参数
    config.set("use_fixed_tasks", True) # 启用了固定任务
    print("固定任务集已启用")
    
    # 定义预设的固定任务难度级别
    # 这些值应该与FixedTaskGenerator中的SPECIFIC_TASKS_CONFIG任务数量匹配
    predefined_task_difficulties = [0.1, 0.2, 0.3, 0.4, 0.5]
    config.set("fixed_tasks_config.difficulty_levels", predefined_task_difficulties)
    log(f"已为FixedTaskGenerator配置预定义难度级别: {predefined_task_difficulties}", LOG_INFO)
    
    # 使用固定难度梯度任务生成器
    print("使用固定难度梯度任务生成器...")
    task_generator = FixedTaskGenerator(config)
    task_sequencer = LinearTaskSequencer(config)
    knowledge_transfer = PolicyTransfer(config)
    
    # 创建课程管理器
    curriculum_manager = CurriculumManager(
        config=config,
        task_generator=task_generator,
        task_sequencer=task_sequencer,
        knowledge_transfer=knowledge_transfer
    )
    
    # 初始化课程
    initial_task = curriculum_manager.initialize()
    print_task_details(initial_task, "初始任务详情")
    
    # 根据任务创建环境
    env = initial_task.create_env()
    
    # 训练模式下设置dt=1.0
    env.set_time_step(1.0)
    print(f"训练模式：时间步长dt设置为1.0")
    
    # 创建MASAC控制器
    n_agents = initial_n_agent + initial_m_enemy
    state_dim = state_number
    action_dim = action_number
    print("初始化MASAC控制器...")
    masac_controller = MASACController(
        n_agents=n_agents, 
        state_dim=state_dim, 
        action_dim=action_dim, 
        device=device,
        memory_capacity=MemoryCapacity,
        auto_entropy=getattr(args, "adaptive_alpha", False),
        target_entropy=getattr(args, "target_entropy", -0.1),
        alpha_min=getattr(args, "alpha_min", 0.01),
        alpha_max=getattr(args, "alpha_max", 3.0),
        max_replay_ratio=20  # 允许较高的重放比例以减轻过拟合
    )
    print(
    f"Alpha模式: {'自适应' if getattr(args, 'adaptive_alpha', False) else '固定为1'}, "
    f"target_entropy={getattr(args, 'target_entropy', -0.1)}, "
    f"alpha范围=[{getattr(args, 'alpha_min', 0.01)}, {getattr(args, 'alpha_max', 3.0)}]"
)
    
    # 添加奖励分配验证函数
    def verify_reward_allocation(rewards, n_friendly_agents):
        """验证奖励分配是否正确
        
        Args:
            rewards: 环境返回的奖励字典 {"leader": r_l, "followers": [r_f1, ...]} 或旧的扁平化数组
            n_friendly_agents: 友方智能体数量
            
        Returns:
            bool: 奖励分配是否有效
        """
        all_reward_values = []
        total_rewards_received = 0

        if isinstance(rewards, dict) and "leader" in rewards and "followers" in rewards:
            # 处理结构化奖励字典
            leader_reward = rewards.get("leader", 0.0)
            follower_rewards = rewards.get("followers", [])
            
            # 确保 leader_reward 是数值
            if isinstance(leader_reward, (int, float, np.number)):
                all_reward_values.append(float(leader_reward))
            
            # 确保 follower_rewards 是列表，且内部是数值
            if isinstance(follower_rewards, list):
                for r in follower_rewards:
                    if isinstance(r, (int, float, np.number)):
                        all_reward_values.append(float(r))
            
            # 结构化数据中，友方奖励数量 = 1 (leader) + len(followers)
            # 注意：这里我们不再严格检查 n_friendly_agents 和接收到的奖励数量是否完全匹配，
            # 因为结构化奖励明确区分了leader和followers，数量不匹配可能是环境设计问题
            total_rewards_received = 1 + len(follower_rewards)
            # 可以在这里添加更灵活的检查，例如 n_friendly_agents 是否等于 total_rewards_received
            if n_friendly_agents != total_rewards_received:
                log(f"[信息] 结构化奖励数量({total_rewards_received})与友方智能体数量({n_friendly_agents})不一致", LOG_INFO)

        elif isinstance(rewards, (np.ndarray, list)):
            # 尝试处理扁平化数组/列表
            try:
                rewards_array = np.asarray(rewards, dtype=float)
                all_reward_values = rewards_array.flatten().tolist()
                total_rewards_received = len(all_reward_values)
                # 扁平化数据下，检查长度
                if total_rewards_received < n_friendly_agents:
                    log(f"[信息] 扁平奖励数组长度({total_rewards_received})小于友方智能体数量({n_friendly_agents})", LOG_INFO)
            except (TypeError, ValueError):
                log(f"[错误] 无法将奖励转换为数值数组: {rewards}", LOG_ERROR)
                return False
        else:
            # 不支持的奖励格式
            log(f"[错误] 不支持的奖励格式: {type(rewards)}", LOG_ERROR)
            return False
            
        # 如果没有提取到任何有效的奖励值
        if not all_reward_values:
            log("[警告] 未能从奖励数据中提取任何有效数值", LOG_WARNING)
            # 根据情况决定是否返回 True 或 False，这里暂时返回 True，允许空奖励
            return True 
        
        # 将提取的奖励值转换为 NumPy 数组进行检查
        reward_values_np = np.array(all_reward_values)
        
        # 检查奖励是否包含NaN值
        if np.isnan(reward_values_np).any():
            log(f"[错误] 奖励包含NaN值: {reward_values_np}", LOG_ERROR)
            return False
            
        # 检查奖励值是否在合理范围
        if np.max(np.abs(reward_values_np)) > 1000:
            log(f"[警告] 奖励值超出正常范围: {reward_values_np}", LOG_WARNING)
            # 这只是警告，不是错误
                
        return True
    
    # 添加奖励记录功能
    reward_allocation_issues = []  # 记录奖励分配问题
    reward_distribution_stats = []  # 记录奖励分布统计信息
    
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
                final_save_dir = os.path.join(models_base_dir, "final")
                os.makedirs(final_save_dir, exist_ok=True)
                final_save_path = os.path.join(final_save_dir, f"final_model_{get_timestamp()}")
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
                task_complete_dir = os.path.join(models_base_dir, f"curriculum_step{curriculum_step}_complete")
                os.makedirs(task_complete_dir, exist_ok=True)
                task_complete_path = os.path.join(task_complete_dir, f"model_ep{episode}_{get_timestamp()}")
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
                save_dir = os.path.join(models_base_dir, f"curriculum_step{curriculum_step}")
                os.makedirs(save_dir, exist_ok=True)
                
                # 构建完整的保存路径
                save_path = os.path.join(save_dir, f"model_ep{episode}_{get_timestamp()}")
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
            final_save_dir = os.path.join(models_base_dir, "final")
            os.makedirs(final_save_dir, exist_ok=True)
            final_save_path = os.path.join(final_save_dir, f"final_model_{get_timestamp()}")
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
            seed_suffix = f"_seed{seed}" if seed != 42 else ""
            run_suffix = f"_run{run_id}" if run_id is not None else ""
            # 确保最后一个任务使用正确的步骤编号（curriculum_step + 1，而不是 curriculum_step）
            result_filename = f"MASAC_curriculum_{timestamp}{run_suffix}{seed_suffix}_step{curriculum_step+1}.pkl"
            result_path = os.path.join(TEST_RESULTS_BASE, result_filename)

            # 确保结果目录存在
            ensure_dir_exists(TEST_RESULTS_BASE)
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
            plot_filename = f"MASAC_curriculum_{timestamp}{run_suffix}{seed_suffix}_step{curriculum_step+1}.png"
            plot_path = os.path.join(TEST_RESULTS_BASE, plot_filename)
            
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
        seed_suffix = f"_seed{seed}" if seed != 42 else ""
        run_suffix = f"_run{run_id}" if run_id is not None else ""
        result_filename = f"MASAC_curriculum_{timestamp}{run_suffix}{seed_suffix}_step{curriculum_step+1}.pkl"
        result_path = os.path.join(TEST_RESULTS_BASE, result_filename)

        # 确保结果目录存在
        ensure_dir_exists(TEST_RESULTS_BASE)
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
        plot_filename = f"MASAC_curriculum_{timestamp}{run_suffix}{seed_suffix}_step{curriculum_step+1}.png"
        plot_path = os.path.join(TEST_RESULTS_BASE, plot_filename)
        
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
    
    # 构建最终训练结果并返回
    final_training_results = {
        'all_ep_r': all_ep_r,
        'success_rate_history': success_rate_history,
        'reward_history': reward_history,
        'alpha_history': alpha_history,
        'final_success_rate': np.mean(success_rate_history[-100:]) if len(success_rate_history) >= 100 else np.mean(success_rate_history) if success_rate_history else 0,
        'final_avg_reward': np.mean(all_ep_r[0][-100:]) if len(all_ep_r[0]) >= 100 else np.mean(all_ep_r[0]) if all_ep_r[0] else 0,
        'seed': seed,
        'run_id': run_id,
        'total_episodes': len(all_ep_r[0]) if all_ep_r[0] else 0,
        'curriculum_steps_completed': curriculum_step + 1,
        'timestamp': get_timestamp()
    }
    
    return final_training_results


def run_multi_seed_curriculum(args, initial_n_agent, initial_m_enemy):

    print("=== 多次课程学习训练模式 ===")
    
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
    
    print(f"开始进行 {args.num_runs} 次课程学习训练")
    for i, seed in enumerate(seeds):
        print(f"\n{'='*60}")
        print(f"第 {i+1}/{args.num_runs} 次训练 - 种子: {seed}")
        print(f"{'='*60}")
        
        # 运行单次课程学习训练
        result = run_with_curriculum(args, initial_n_agent, initial_m_enemy, seed=seed, run_id=i+1)
        all_results.append(result)
        
        print(f"第 {i+1} 次课程学习训练完成")
        if result:
            final_reward = result.get('final_avg_reward', 'N/A')
            final_success = result.get('final_success_rate', 'N/A')
            curriculum_steps = result.get('curriculum_steps_completed', 'N/A')
            total_episodes = result.get('total_episodes', 'N/A')
            print(f"最终平均奖励: {final_reward}")
            print(f"最终成功率: {final_success}")
            print(f"完成课程步骤: {curriculum_steps}")
            print(f"总训练回合: {total_episodes}")
    
    # 汇总所有运行的结果
    print(f"\n{'='*60}")
    print("多次课程学习训练汇总")
    print(f"{'='*60}")
    
    # 计算统计信息
    final_rewards = [r.get('final_avg_reward', 0) for r in all_results if r]
    final_success_rates = [r.get('final_success_rate', 0) for r in all_results if r]
    curriculum_steps_list = [r.get('curriculum_steps_completed', 0) for r in all_results if r]
    total_episodes_list = [r.get('total_episodes', 0) for r in all_results if r]
    
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
    
    if curriculum_steps_list:
        steps_mean = np.mean(curriculum_steps_list)
        steps_std = np.std(curriculum_steps_list)
        print(f"平均完成课程步骤: {steps_mean:.1f} ± {steps_std:.1f}")
    
    if total_episodes_list:
        episodes_mean = np.mean(total_episodes_list)
        episodes_std = np.std(total_episodes_list)
        print(f"平均总训练回合: {episodes_mean:.1f} ± {episodes_std:.1f}")
    
    # 保存汇总结果
    timestamp = get_timestamp()
    summary_filename = f"MASAC_curriculum_multi_run_summary_{timestamp}.pkl"
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
            },
            'curriculum_steps': {
                'mean': steps_mean if curriculum_steps_list else 0,
                'std': steps_std if curriculum_steps_list else 0,
                'values': curriculum_steps_list
            },
            'total_episodes': {
                'mean': episodes_mean if total_episodes_list else 0,
                'std': episodes_std if total_episodes_list else 0,
                'values': total_episodes_list
            }
        },
        'timestamp': timestamp,
        'config': {
            'initial_n_agent': initial_n_agent,
            'initial_m_enemy': initial_m_enemy,
            'seeds': seeds,
            'num_runs': args.num_runs,
            'use_curriculum': True
        }
    }
    
    with open(summary_path, 'wb') as f:
        pkl.dump(summary_results, f)
    
    print(f"\n多次课程学习训练汇总结果已保存: {summary_path}")
    print("多次课程学习训练完成！")
    
    return summary_results


def analyze_curriculum_learning(manager: CurriculumManager):
    """分析课程学习过程
    
    Args:
        manager: 课程管理器实例
    """
    tasks = manager.get_all_tasks()
    print(f"\n课程学习统计:")
    print(f"- 总任务数: {len(tasks)}")
    print(f"- 总训练回合数: {manager.total_episodes}")  # 添加总回合数信息
    
    # 计算难度相关统计
    difficulties = [task.difficulty for task in tasks if task.difficulty is not None]
    if difficulties:
        print(f"- 难度范围: {min(difficulties):.2f} - {max(difficulties):.2f}")
        print(f"- 平均难度: {np.mean(difficulties):.2f}")
    
    # 分析任务变化维度
    hero_counts = [task.env_params.get("hero_count", 1) for task in tasks]
    enemy_counts = [task.env_params.get("enemy_count", 1) for task in tasks]
    obstacle_counts = [task.env_params.get("obstacle_count", 0) for task in tasks]
    uav_speeds = [task.env_params.get("uav_speed", 10.0) for task in tasks]
    
    print(f"- 友方无人机数量变化: {min(hero_counts)} - {max(hero_counts)}")
    print(f"- 敌方无人机数量变化: {min(enemy_counts)} - {max(enemy_counts)}")
    print(f"- 障碍物数量变化: {min(obstacle_counts)} - {max(obstacle_counts)}")
    print(f"- 无人机速度变化: {min(uav_speeds):.1f} - {max(uav_speeds):.1f}")
    
    # 任务解决情况
    solved_tasks = [task for task in tasks if task.is_solved()]
    print(f"- 已解决任务数: {len(solved_tasks)}/{len(tasks)} ({len(solved_tasks)/len(tasks)*100:.1f}%)")
    
    # 任务难度分布
    difficulty_ranges = {
        "低难度(0.0-0.3)": len([t for t in tasks if t.difficulty is not None and t.difficulty <= 0.3]),
        "中难度(0.3-0.7)": len([t for t in tasks if t.difficulty is not None and 0.3 < t.difficulty <= 0.7]),
        "高难度(0.7-1.0)": len([t for t in tasks if t.difficulty is not None and t.difficulty > 0.7])
    }
    print("- 任务难度分布:")
    for range_name, count in difficulty_ranges.items():
        print(f"  - {range_name}: {count} 个任务 ({count/len(tasks)*100:.1f}%)")
        
    # 任务解决情况与难度的关系
    if solved_tasks:
        solved_difficulties = [task.difficulty for task in solved_tasks if task.difficulty is not None]
        if solved_difficulties:
            print(f"- 已解决任务的平均难度: {np.mean(solved_difficulties):.2f}")
            
    # 任务解决所需的平均训练轮数
    episodes_to_solve = []
    for task in solved_tasks:
        if task.performance_history:
            episodes_to_solve.append(len(task.performance_history))
    
    if episodes_to_solve:
        print(f"- 任务解决平均训练轮数: {np.mean(episodes_to_solve):.1f}")
        print(f"- 任务解决最少训练轮数: {min(episodes_to_solve)}")
        print(f"- 任务解决最多训练轮数: {max(episodes_to_solve)}")
    
    print("\n课程学习分析完成")

def run_monte_carlo_test(model_path, test_episodes=None, test_options=None, collect_formation_data=True, experiment_type="curriculum"):
    """运行蒙特卡洛测试以评估模型性能
    
    Args:
        model_path: 要测试的模型路径
        test_episodes: 测试回合数，如果为None则使用全局变量TEST_EPIOSDE
        test_options: 测试选项字典，包含如障碍物数量、无人机速度等配置
        collect_formation_data: 是否收集详细的编队数据，默认为True
        
    Returns:
        测试结果统计字典
    """
    import numpy as np  # 移到函数开始处避免UnboundLocalError
    global RENDER, action_number
    
    if test_episodes is None:
        test_episodes = TEST_EPIOSDE
    
    if test_options is None:
        test_options = {}
    model_path = _resolve_model_path(model_path, experiment_type)
    if not model_path or not os.path.isfile(model_path):
        default_train_root = os.path.join("outputs", "train", _safe_experiment_name(experiment_type))
        raise FileNotFoundError(
            f"找不到模型文件: {model_path or '未指定'}。"
            f"请传入具体模型文件，或传入训练输出目录；默认会在 {default_train_root} 下查找最新模型。"
        )
        
    # 从测试选项中提取环境参数
    obstacle_count = test_options.get('obstacle_count', 1)
    uav_speed = test_options.get('uav_speed', None)
    hero_count = test_options.get('hero_count', 1)
    enemy_count = test_options.get('enemy_count', 3) 
        
    print(f"开始蒙特卡洛测试，测试回合数: {test_episodes}")
    print(f"加载模型: {model_path}")
    print(f"测试配置: 友方={hero_count}, 敌方={enemy_count}, 障碍物={obstacle_count}")
    if uav_speed:
        print(f"无人机速度: {uav_speed}")
    if collect_formation_data:
        print("启用详细编队数据收集")
    
    predefined_positions = {
        "goals": [(500, 200)]  # 与训练时相同的固定目标位置
    }
    print(f"使用固定目标位置: (500, 200)")
    
    # 创建环境，添加预定义位置参数
    env = RlGame(leader_count=hero_count, follower_count=enemy_count, obstacle_num=obstacle_count, render=RENDER, 
                predefined_positions=predefined_positions).unwrapped
    
    env.set_time_step(0.1)
    print(f"测试模式：时间步长dt设置为1")

    # 测试阶段启用 Leader safety shield
    if hasattr(env, "entity_manager"):
        env.entity_manager.enable_leader_safety_shield = True
        env.entity_manager.leader_safe_distance = 32.0
        env.entity_manager.leader_warning_distance = 95.0
        env.entity_manager.leader_safety_horizon = 4
        print("[Test] Leader safety shield enabled.")
    
    if hasattr(env, 'entity_manager') and hasattr(env.entity_manager, 'images_loaded'):
        env.entity_manager.images_loaded = False
        print("已重置EntityManager的图像加载状态，确保测试时图像正确加载")
    

    n_agents = hero_count
    state_dim = state_number
    action_dim = action_number
    masac_controller = MASACController(n_agents, state_dim, action_dim, device=device)
    if not masac_controller.load_models(model_path, strict=False):
        raise RuntimeError(f"模型加载失败: {model_path}")
    
    # 初始化统计数据
    win_count = 0
    rewards = []
    steps = []
    formation_rates = []
    distances = []  # 智能体最终与目标的距离
    trajectory_lengths = []  # 飞行轨迹长度列表
    energy_consumptions = []  # 能量消耗列表
    
    # 初始化编队数据收集结构
    formation_data = {
        'test_info': {
            'test_id': get_timestamp(),
            'timestamp': time.time(),
            'total_episodes': test_episodes,
            'agents_config': {
                'leader_count': hero_count,
                'follower_count': enemy_count,
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
        
        episode_trajectory_length = 0.0  # 当前回合轨迹长度
        episode_energy_consumption = 0.0  # 当前回合能量消耗
        
        while not done and step_count < EP_LEN:
            # 收集当前时间步的编队状态数据
            if collect_formation_data:
                timestep_state = env.get_formation_state()
                episode_formation_data['timesteps'].append(timestep_state)
            
            # 选择动作（无噪声且使用确定性策略）
            action = masac_controller.select_actions(observation, add_noise=False, evaluate=True)
            
            # 执行动作
            observation_, reward, done, win, team_counter, dis = env.step(action)
            
            # 获取速度和控制输入数据
            if hasattr(env, 'entity_manager') and env.entity_manager.leaders:
                leader = env.entity_manager.leaders[0]
                # 计算轨迹长度增量
                speed = getattr(leader, 'speed', 0.0)
                episode_trajectory_length += float(speed)
                
                # 计算能量消耗增量（基于动作）
                if isinstance(action, dict) and "leader" in action:
                    leader_action = action["leader"]
                else:
                    leader_action = action[0] if len(action) > 0 else [0, 0]
                
                # 假设动作为[加速度, 角速度]
                u = abs(float(leader_action[0])) if len(leader_action) > 0 else 0.0
                omega = abs(float(leader_action[1])) if len(leader_action) > 1 else 0.0
                episode_energy_consumption += (u + omega)
            
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
                # 如果是字典，取所有值的最小值
                if last_distance:
                    distances.append(min(last_distance.values()))
                else:
                    distances.append(0)  # 如果字典为空，使用0
            elif isinstance(last_distance, (list, tuple, np.ndarray)):
                # 如果是列表、元组或数组，取最小值
                distances.append(min(last_distance))
            else:
                # 如果是标量，直接添加
                distances.append(last_distance)
        
        # 存储轨迹长度和能量消耗
        trajectory_lengths.append(episode_trajectory_length)
        energy_consumptions.append(episode_energy_consumption)
        
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
    avg_distance = np.mean(distances) if distances else 0
    std_distance = np.std(distances) if distances else 0
    
    # 计算新增指标
    avg_trajectory_length = np.mean(trajectory_lengths) if trajectory_lengths else 0
    std_trajectory_length = np.std(trajectory_lengths) if trajectory_lengths else 0
    avg_energy_consumption = np.mean(energy_consumptions) if energy_consumptions else 0
    std_energy_consumption = np.std(energy_consumptions) if energy_consumptions else 0
    
    # 计算成功率加权探索时间(SET)
    set_score = success_rate * avg_steps
    
    # 输出总体统计信息
    print("\n蒙特卡洛测试结果统计 (平均值±标准差):")
    print(f"测试回合数: {test_episodes}")
    print(f"1. 任务完成率(MCR): {success_rate:.2f}")
    print(f"2. 编队保持率(FKR): {avg_formation_rate:.2f}±{std_formation_rate:.2f}")
    print(f"3. 成功率加权探索时间(SET): {set_score:.2f} (SR: {success_rate:.2f} × 平均时间: {avg_steps:.2f})")
    print(f"4. 飞行轨迹(J_S): {avg_trajectory_length:.2f}±{std_trajectory_length:.2f}")
    print(f"5. 能量消耗(J_C): {avg_energy_consumption:.2f}±{std_energy_consumption:.2f}")
    print(f"平均奖励: {avg_reward:.2f}±{std_reward:.2f}")
    print(f"平均最终距离: {avg_distance:.2f}±{std_distance:.2f}")
    
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
        "set_score": {
            "value": float(set_score),
            "success_rate": float(success_rate),
            "avg_exploration_time": float(avg_steps)
        },
        "formation_rates": {
            "mean": float(avg_formation_rate),
            "std": float(std_formation_rate),
            "values": [float(f) for f in formation_rates]
        },
        "distances": {
            "mean": float(avg_distance),
            "std": float(std_distance),
            "values": [float(d) for d in distances]
        },
        "trajectory_lengths": {
            "mean": float(avg_trajectory_length),
            "std": float(std_trajectory_length),
            "values": [float(t) for t in trajectory_lengths]
        },
        "energy_consumptions": {
            "mean": float(avg_energy_consumption),
            "std": float(std_energy_consumption),
            "values": [float(e) for e in energy_consumptions]
        },
        "test_config": {
            "hero_count": hero_count,
            "enemy_count": enemy_count,
            "obstacle_count": obstacle_count,
            "uav_speed": uav_speed
        },
        "timestamp": time.time()
    }
    
    # 如果收集了编队数据，将其添加到结果中
    if collect_formation_data and formation_data:
        results["formation_data"] = formation_data
    
    # 创建唯一的测试ID和保存目录
    timestamp = get_timestamp()
    config_str = f"_h{hero_count}_e{enemy_count}_o{obstacle_count}"
    if uav_speed:
        config_str += f"_s{int(uav_speed)}"
    
    # 创建保存目录
    test_dir_name = f"single_test_{timestamp}{config_str}"
    test_dir = os.path.join(TEST_RESULTS_BASE, test_dir_name)
    ensure_dir_exists(test_dir)
    
    # 保存测试结果 (pickle格式)
    pickle_path = os.path.join(test_dir, _timestamped_filename("test_results", ".pkl", timestamp))
    with open(pickle_path, 'wb') as f:
        pkl.dump(results, f, pkl.HIGHEST_PROTOCOL)
    print(f"测试结果已保存到: {pickle_path}")
    
    # 同时保存为JSON格式
    json_path = os.path.join(test_dir, _timestamped_filename("test_results", ".json", timestamp))
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
        "metrics": {
            "MCR": float(success_rate),
            "FKR": {
                "mean": float(avg_formation_rate),
                "std": float(std_formation_rate)
            },
            "SET": {
                "value": float(set_score),
                "success_rate": float(success_rate),
                "avg_exploration_time": float(avg_steps)
            },
            "J_S": {
                "mean": float(avg_trajectory_length),
                "std": float(std_trajectory_length)
            },
            "J_C": {
                "mean": float(avg_energy_consumption),
                "std": float(std_energy_consumption)
            },
            "avg_reward": {
                "mean": float(avg_reward),
                "std": float(std_reward)
            },
            "avg_steps": {
                "mean": float(avg_steps),
                "std": float(std_steps)
            },
            "avg_distance": {
                "mean": float(avg_distance),
                "std": float(std_distance)
            }
        }
    }
    
    info_path = os.path.join(test_dir, _timestamped_filename("test_info", ".json", timestamp))
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(test_info, f, ensure_ascii=False, indent=4)
    
    # 单独保存编队数据 (PKL格式)
    if collect_formation_data and formation_data:
        formation_pkl_path = os.path.join(test_dir, _timestamped_filename("formation_data", ".pkl", timestamp))
        with open(formation_pkl_path, 'wb') as f:
            pkl.dump(formation_data, f, pkl.HIGHEST_PROTOCOL)
        print(f"编队数据已保存到: {formation_pkl_path}")
        # 生成热力图（基于 formation_data.pkl）
        try:
            from visualization.plot_heatmaps import generate_heatmaps
            heatmap_paths = generate_heatmaps(formation_pkl_path, test_dir, timestamp)
            if heatmap_paths:
                print("已生成热力图:")
                for k, v in heatmap_paths.items():
                    print(f" - {k}: {v}")
        except Exception as e:
            print(f"生成热力图时出错: {e}")
            
        # 生成综合分析曲线图
        try:
            from visualization.plot_formation_curves import generate_formation_curves
            curves_path = generate_formation_curves(formation_pkl_path, test_dir, "AC-MASAC", timestamp)
            if curves_path:
                print(f"已生成综合分析曲线图: {curves_path}")
        except Exception as e:
            print(f"生成综合分析曲线图时出错: {e}")
        
        # 保存编队数据汇总 
        formation_summary_path = os.path.join(test_dir, _timestamped_filename("formation_summary", ".pkl", timestamp))
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
        if distances:
            plt.subplot(2, 2, 4)
            plt.hist(distances, bins=min(20, test_episodes//5), alpha=0.7)
            plt.title('最终距离分布')
            plt.xlabel('距离')
            plt.ylabel('频次')
            plt.axvline(avg_distance, color='r', linestyle='dashed', linewidth=1, label=f'平均值: {avg_distance:.2f}')
            plt.legend()
        
        plt.tight_layout()
        title = f"蒙特卡洛测试结果 (友方:{hero_count}, 敌方:{enemy_count}, 障碍:{obstacle_count})"
        if uav_speed:
            title += f", 速度:{uav_speed}"
        plt.suptitle(title)
        
        # 保存图片到测试结果目录
        save_img_path = os.path.join(test_dir, _timestamped_filename("histogram", ".png", timestamp))
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
            json_results = [
                os.path.join(root, f) for f in files
                if (f.startswith("test_results") or f.startswith("all_results")) and f.endswith(".json")
            ]
            
            # 如果找到了JSON文件，优先使用
            if json_results:
                all_results.extend(json_results)
            else:
                # 否则查找pickle文件
                pkl_results = [
                    os.path.join(root, f) for f in files
                    if (f.startswith("test_results") or f.startswith("all_results")) and f.endswith(".pkl")
                ]
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

def monte_carlo_test(actor_path, critic_path=None, test_nums=100, base_difficulty_levels=None, experiment_type="curriculum"):
    """执行蒙特卡洛测试
    
    Args:
        actor_path: Actor路径
        critic_path: Critic路径(可选)
        test_nums: 测试次数
        base_difficulty_levels: 基础难度级别列表
    """
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
    # from curriculum.fixed_task_generator import FixedTaskGenerator  # 已在顶部导入
    from rl_env.path_env import RlGame
    actor_path = _resolve_model_path(actor_path, experiment_type)
    if not actor_path or not os.path.isfile(actor_path):
        default_train_root = os.path.join("outputs", "train", _safe_experiment_name(experiment_type))
        raise FileNotFoundError(
            f"找不到模型文件: {actor_path or '未指定'}。"
            f"请传入具体模型文件，或传入训练输出目录；默认会在 {default_train_root} 下查找最新模型。"
        )
    
    # 创建多难度测试专用目录
    timestamp = get_timestamp()
    model_name = os.path.basename(actor_path)
    safe_model_name = _safe_experiment_name(model_name)
    multi_test_dir = os.path.join(TEST_RESULTS_BASE, f"multi_diff_test_{timestamp}_{safe_model_name}")
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
    
    config_path = os.path.join(multi_test_dir, _timestamped_filename("test_config", ".json", timestamp))
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
            test_options=difficulty_config,
            experiment_type=experiment_type
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
        comparison_img_path = os.path.join(multi_test_dir, _timestamped_filename("difficulty_comparison", ".png", timestamp))
        plt.savefig(comparison_img_path)
        plt.close()
        print(f"难度级别比较图已保存到: {comparison_img_path}")
    except Exception as e:
        print(f"绘制性能比较图时出错: {e}")
        import traceback
        traceback.print_exc()
    
    # 保存完整结果 (Pickle格式)
    try:
        pickle_results_path = os.path.join(multi_test_dir, _timestamped_filename("all_results", ".pkl", timestamp))
        with open(pickle_results_path, 'wb') as f:
            pkl.dump(all_results, f, pkl.HIGHEST_PROTOCOL)
        print(f"完整测试结果(Pickle格式)已保存到: {pickle_results_path}")
        
        # 同时保存为JSON格式
        json_results_path = os.path.join(multi_test_dir, _timestamped_filename("all_results", ".json", timestamp))
        with open(json_results_path, 'w', encoding='utf-8') as f:
            json.dump(convert_to_json_compatible(all_results), f, ensure_ascii=False, indent=4)
        print(f"完整测试结果(JSON格式)已保存到: {json_results_path}")
    except Exception as e:
        print(f"保存完整测试结果时出错: {e}")
    
    # 更新测试结果索引
    create_test_results_index()
    
    return all_results


# ==================== Integrated ablation/baseline entry helpers ====================
def _format_seconds(seconds):
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    sec = seconds % 60
    return f"{h:02d}:{m:02d}:{sec:05.2f}"


def _safe_experiment_name(name):
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", str(name or "run")).strip("_") or "run"


def _make_timestamped_results_dir(experiment_type, mode, base_dir=None):
    ts = time.strftime("%Y%m%d_%H%M%S")
    safe_experiment = _safe_experiment_name(experiment_type)
    safe_mode = _safe_experiment_name(mode)
    root = base_dir or "outputs"
    path = os.path.join(root, safe_mode, safe_experiment, f"{safe_experiment}_{safe_mode}_{ts}")
    ensure_dir_exists(path)
    return path


def _timestamped_filename(prefix, ext, timestamp=None):
    ts = timestamp or get_timestamp()
    safe_prefix = _safe_experiment_name(prefix)
    clean_ext = ext if str(ext).startswith(".") else f".{ext}"
    return f"{safe_prefix}_{ts}{clean_ext}"


def _find_first_prefixed_file(directory, prefix, ext):
    if not os.path.isdir(directory):
        return None
    candidates = [
        f for f in os.listdir(directory)
        if f.startswith(prefix) and f.endswith(ext)
    ]
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return os.path.join(directory, candidates[0])


def _find_latest_model_file(search_roots, prefixes=("final_model", "model_ep")):
    candidates = []
    seen_roots = set()
    for root in search_roots:
        if not root:
            continue
        root = os.path.normpath(root)
        if root in seen_roots:
            continue
        seen_roots.add(root)

        if os.path.isfile(root):
            if os.path.basename(root).startswith(prefixes):
                candidates.append(root)
            continue
        if not os.path.isdir(root):
            continue

        for current_dir, _, files in os.walk(root):
            for name in files:
                if name.startswith(prefixes):
                    candidates.append(os.path.join(current_dir, name))

    if not candidates:
        return None
    candidates.sort(key=lambda path: (os.path.getmtime(path), path), reverse=True)
    return candidates[0]


def _resolve_model_path(model_path=None, experiment_type="curriculum"):
    """Resolve a timestamped model file from a file path, model dir, or default output tree."""
    legacy_default = os.path.normpath(os.path.join("models", "final", "final_model"))
    safe_experiment = _safe_experiment_name(experiment_type or "curriculum")

    if model_path and os.path.isfile(model_path):
        return model_path

    search_roots = []
    should_auto_search = not model_path
    if model_path and os.path.isdir(model_path):
        search_roots.append(model_path)
        should_auto_search = True
    elif model_path and os.path.normpath(model_path) == legacy_default:
        should_auto_search = True

    if should_auto_search:
        search_roots.extend([
            os.path.join("outputs", "train", safe_experiment),
            os.path.join("outputs", "train"),
        ])
        resolved = _find_latest_model_file(search_roots)
        if resolved:
            if model_path != resolved:
                print(f"自动选择最新模型文件: {resolved}")
            return resolved

    return model_path


def _register_run_timer(results_dir, experiment_type, mode):
    start_wall = time.time()
    start_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    record_ts = time.strftime("%Y%m%d_%H%M%S")
    ensure_dir_exists(results_dir)
    print(f"[记录] 实验类型: {experiment_type}")
    print(f"[记录] 文件保存目录: {results_dir}")
    print(f"[记录] 启动时间: {start_str}")

    def _finish():
        end_wall = time.time()
        end_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        elapsed = end_wall - start_wall
        record = {
            "experiment_type": experiment_type,
            "mode": mode,
            "start_time": start_str,
            "end_time": end_str,
            "elapsed_seconds": elapsed,
            "elapsed_hms": _format_seconds(elapsed),
            "results_dir": os.path.normpath(results_dir),
        }
        try:
            ensure_dir_exists(results_dir)
            record_path = os.path.join(results_dir, _timestamped_filename("run_time_record", ".json", record_ts))
            with open(record_path, "w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            print(f"[警告] 运行时间记录保存失败: {exc}")
        print(f"[记录] 结束时间: {end_str}")
        print(f"[记录] 统计耗时: {_format_seconds(elapsed)}")
    atexit.register(_finish)
    return start_wall


def _strip_dispatch_args(argv):
    cleaned = []
    skip_next = False
    dispatch_flags = {"--experiment_type", "--run_tag"}
    for i, item in enumerate(argv):
        if skip_next:
            skip_next = False
            continue
        if item in dispatch_flags:
            skip_next = True
            continue
        if any(item.startswith(flag + "=") for flag in dispatch_flags):
            continue
        cleaned.append(item)
    return cleaned






def _filter_hcrrt_args(argv):
    allowed_with_value = {"--hero_count", "--enemy_count", "--obstacle_count", "--test_episodes", "--replan_horizon", "--dt", "--results_dir", "--seed"}
    allowed_flags = {"--use_formation", "--no_use_formation", "--render", "--no_render"}
    cleaned = []
    i = 0
    while i < len(argv):
        item = argv[i]
        if item in allowed_flags:
            cleaned.append(item); i += 1; continue
        if item in allowed_with_value:
            cleaned.append(item)
            if i + 1 < len(argv):
                cleaned.append(argv[i + 1])
                i += 2
            else:
                i += 1
            continue
        if any(item.startswith(flag + "=") for flag in allowed_with_value):
            cleaned.append(item); i += 1; continue
        # Drop unsupported unified/training args and their value if present.
        if item.startswith("--") and i + 1 < len(argv) and not argv[i + 1].startswith("--"):
            i += 2
        else:
            i += 1
    return cleaned

def _prepare_integrated_module_defaults(module, experiment_type, results_dir):
    """Initialize globals that used to be created in each standalone script's __main__ block."""
    try:
        module.set_seed(42)
    except Exception:
        pass
    module.TRAIN_NUM = getattr(module, "TRAIN_NUM", 1)
    module.TEST_EPIOSDE = getattr(module, "TEST_EPIOSDE", 100)
    module.state_number = getattr(module, "state_number", 7)
    module.action_number = getattr(module, "action_number", 2)
    module.N_Agent = getattr(module, "N_Agent", 1)
    module.M_Enemy = getattr(module, "M_Enemy", 3)
    module.EP_LEN = getattr(module, "EP_LEN", 1000)
    module.MemoryCapacity = getattr(module, "MemoryCapacity", 50000)
    module.BATCH = getattr(module, "BATCH", 256)
    module.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module.RESULTS_DIR = results_dir
    module.TEST_RESULTS_BASE = os.path.join(results_dir, "test_results")
    if experiment_type == "no_attention":
        module.TRAINING_RESULTS_FILE = os.path.join(results_dir, _timestamped_filename("MASAC_no_attention_curriculum", ".pkl"))
    elif experiment_type == "no_curriculum":
        module.TRAINING_RESULTS_FILE = os.path.join(results_dir, _timestamped_filename("MASAC_no_curriculum", ".pkl"))
    else:
        module.TRAINING_RESULTS_FILE = os.path.join(results_dir, _timestamped_filename(experiment_type, ".pkl"))
    ensure_dir_exists(module.RESULTS_DIR)
    ensure_dir_exists(module.TEST_RESULTS_BASE)
    ensure_dir_exists(os.path.dirname(module.TRAINING_RESULTS_FILE))

def _delegate_integrated_experiment(experiment_type, original_argv):
    """Run ablation/baseline implementations from the unified main entry."""
    mode = "test" if ("--test" in original_argv or "--multi_difficulty_test" in original_argv) else "train"
    argv = _strip_dispatch_args(original_argv)
    # Give every experiment its own timestamped result folder when the user does not provide one.
    if "--results_dir" not in argv and not any(a.startswith("--results_dir=") for a in argv):
        argv.extend(["--results_dir", _make_timestamped_results_dir(experiment_type, mode)])
    # H-CRRT is a test-only baseline; normalize its argument names/defaults.
    if experiment_type == "h_crrt":
        if "--test" in argv:
            argv.remove("--test")
        if "--multi_difficulty_test" in argv:
            argv.remove("--multi_difficulty_test")
        argv = _filter_hcrrt_args(argv)
        if "--results_dir" not in argv and not any(a.startswith("--results_dir=") for a in argv):
            argv.extend(["--results_dir", _make_timestamped_results_dir(experiment_type, "test")])
        # H-CRRT has no training stage, so always record as test/baseline.
        mode = "test"
        from integrated_ablation_modules.H_CRRT import run_hcrrt as hcrrt_main
        target_module = hcrrt_main
        target_main = hcrrt_main.main
    elif experiment_type == "no_attention":
        if not any(flag in argv for flag in ["--use_curriculum", "--test", "--multi_difficulty_test", "--analyze", "--create_index"]):
            argv.append("--use_curriculum")
        from integrated_ablation_modules.masac_no_attention import main_masac_no_attention as no_attention_main
        target_module = no_attention_main
        target_main = no_attention_main.main
    elif experiment_type == "no_curriculum":
        from integrated_ablation_modules.masac_no_curriculum import main_masac_no_curriculum as no_curriculum_main
        target_module = no_curriculum_main
        target_main = no_curriculum_main.main
    else:
        raise ValueError(f"未知实验类型: {experiment_type}")

    results_dir = None
    for idx, item in enumerate(argv):
        if item == "--results_dir" and idx + 1 < len(argv):
            results_dir = argv[idx + 1]
        elif item.startswith("--results_dir="):
            results_dir = item.split("=", 1)[1]
    results_dir = results_dir or _make_timestamped_results_dir(experiment_type, mode)
    if experiment_type in ["no_attention", "no_curriculum"]:
        _prepare_integrated_module_defaults(target_module, experiment_type, results_dir)
    _register_run_timer(results_dir, experiment_type, mode)

    old_argv = sys.argv[:]
    try:
        sys.argv = [old_argv[0]] + argv
        return target_main()
    finally:
        sys.argv = old_argv

def main():
    """主函数
    """
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument('--experiment_type', type=str, default='curriculum',
                            choices=['curriculum', 'standard', 'no_attention', 'no_curriculum', 'h_crrt'],
                            help='统一入口实验类型：curriculum/standard/no_attention/no_curriculum/h_crrt')
    pre_parser.add_argument('--run_tag', type=str, default=None, help='可选运行标记，会写入时间记录')
    pre_args, _ = pre_parser.parse_known_args()
    if pre_args.experiment_type in ['no_attention', 'no_curriculum', 'h_crrt']:
        return _delegate_integrated_experiment(pre_args.experiment_type, sys.argv[1:])

    # 添加命令行参数解析
    parser = argparse.ArgumentParser(description='MASAC with Curriculum Learning', parents=[pre_parser])
    parser.add_argument('--use_curriculum', action='store_true', help='使用课程学习框架')
    parser.add_argument('--render', action='store_true', help='是否渲染环境')
    parser.add_argument('--adaptive_alpha', action='store_true',
                        help='开启自适应alpha；默认关闭，alpha固定为1')
    parser.add_argument('--target_entropy', type=float, default=-0.1,
                    help='自适应alpha的目标熵；论文设置为-0.1，二维动作常用可尝试-2.0')
    parser.add_argument('--alpha_min', type=float, default=0.01,
                        help='自适应alpha的下限')
    parser.add_argument('--alpha_max', type=float, default=3.0,
                    help='自适应alpha的上限，建议先用1.0，最多可尝试2.0')
    parser.add_argument('--test', action='store_true', help='测试模式（加载已训练的模型）')
    parser.add_argument('--model_path', type=str, default=None, 
                        help='测试模式下加载的模型文件或模型目录；不指定时会从 outputs/train/<experiment_type>/ 自动选择最新模型')
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
                       help='结果保存目录，默认自动生成到 outputs/<train|test>/<experiment>/<experiment>_<mode>_<timestamp>')
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
    if args.experiment_type == 'standard':
        args.use_curriculum = False
    mode = 'test' if (args.test or args.multi_difficulty_test) else 'train'
    if args.results_dir is None:
        args.results_dir = _make_timestamped_results_dir(args.experiment_type, mode)
    _register_run_timer(args.results_dir, args.experiment_type, mode)
    
    # 设置全局日志级别
    log_level_map = {
        'debug': LOG_DEBUG,
        'info': LOG_INFO,
        'warning': LOG_WARNING,
        'error': LOG_ERROR
    }
    set_log_level(log_level_map.get(args.log_level, LOG_INFO))
    log(f"日志级别设置为: {args.log_level.upper()}", LOG_INFO)
    
    global RENDER, action_number, TRAINING_RESULTS_FILE, TEST_RESULTS_BASE, RESULTS_DIR
    
    # 如果指定了结果目录，更新全局变量
    if args.results_dir:
        RESULTS_DIR = args.results_dir
        TEST_RESULTS_BASE = os.path.join(RESULTS_DIR, "test_results")
        training_prefix = "MASAC_curriculum" if args.experiment_type == "curriculum" else f"MASAC_{args.experiment_type}"
        TRAINING_RESULTS_FILE = os.path.join(RESULTS_DIR, _timestamped_filename(training_prefix, ".pkl"))
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
    
    # 设置渲染标志：测试时默认渲染，训练时固定关闭渲染
    if args.test or args.multi_difficulty_test:
        RENDER = True
    else:
        RENDER = False
    
    # 使用解析后的参数或默认值
    n_agent = args.hero_count
    m_enemy = args.enemy_count
    print(f"设置友方无人机数量 (n_agent): {n_agent}")
    print(f"设置敌方无人机数量 (m_enemy): {m_enemy}")
    
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
        
        # 输出最终使用的难度级别
        print(f"将测试的难度级别配置: {difficulty_levels}")
                
        # 运行多难度测试
        monte_carlo_test(
            actor_path=args.model_path,
            critic_path=args.model_path,  # 使用相同的路径前缀，加载函数会自动添加_critic_{i}.pth
            test_nums=args.test_episodes or TEST_EPIOSDE,
            base_difficulty_levels=difficulty_levels,
            experiment_type=args.experiment_type
        )
    elif args.test:
        # 单难度测试模式
        print(f"启动单一难度蒙特卡洛测试模式: 友方={n_agent}, 敌方={m_enemy}, 障碍物={args.obstacle_count}")
        if args.test_speed:
            print(f"指定无人机速度: {args.test_speed}")
        
        # 设置测试选项
        test_options = {
            'hero_count': n_agent, # 使用 n_agent
            'enemy_count': m_enemy, # 使用 m_enemy
            'obstacle_count': args.obstacle_count,
            'uav_speed': args.test_speed
        }
        
        # 运行测试
        run_monte_carlo_test(
            model_path=args.model_path,
            test_episodes=args.test_episodes or TEST_EPIOSDE, # TEST_EPIOSDE 也需要定义
            test_options=test_options,
            experiment_type=args.experiment_type
        )
    else:
        # 训练模式
        if args.use_curriculum:
            if args.multi_run:
                print("使用多次课程学习框架进行训练")
                # 传递 n_agent 和 m_enemy 给 run_multi_seed_curriculum
                run_multi_seed_curriculum(args, n_agent, m_enemy)
            else:
                print("使用单次课程学习框架进行训练")
                # 传递 n_agent 和 m_enemy 给 run_with_curriculum
                run_with_curriculum(args, n_agent, m_enemy)
        else:
            print("使用标准MASAC进行训练")
            import main_SAC
            main_SAC.default_model_dir = os.path.join(RESULTS_DIR, "model")
            main_SAC.shoplistfile = os.path.join(RESULTS_DIR, _timestamped_filename("MASAC_standard", ".pkl"))
            main_SAC.shoplistfile_test = os.path.join(RESULTS_DIR, _timestamped_filename("MASAC_standard_test", ".pkl"))
            main_SAC.shoplistfile_test1 = os.path.join(RESULTS_DIR, _timestamped_filename("MASAC_standard_compare", ".pkl"))
            main_SAC.ADAPTIVE_ALPHA = args.adaptive_alpha
            main_SAC.RENDER = False
            # 创建环境实例并传递给run函数
            # 使用 n_agent 和 m_enemy 变量
            env = RlGame(leader_count=n_agent, follower_count=m_enemy, obstacle_num=args.obstacle_count, render=RENDER).unwrapped
            main_SAC.run(env)

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
    RESULTS_DIR = os.path.join("outputs", "train", "curriculum", "manual")
    TEST_RESULTS_BASE = os.path.join(RESULTS_DIR, "test_results")
    TRAINING_RESULTS_FILE = os.path.join(RESULTS_DIR, _timestamped_filename("MASAC_curriculum", ".pkl"))

    main()
    
