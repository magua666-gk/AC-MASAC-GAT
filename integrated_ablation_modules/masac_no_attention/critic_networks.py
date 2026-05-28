import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 定义角色ID常量
LEADER_TYPE_ID = 0
FOLLOWER_TYPE_ID = 1

class CriticObsActEncoder(nn.Module):
    """Critic 观测动作编码器
    
    将观测与动作编码为隐藏表示
    """
    def __init__(self, state_dim, action_dim, embed_dim, hidden_dims=[256, 128]):
        """初始化观测动作编码器
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            embed_dim: 输出嵌入维度
            hidden_dims: 隐藏层维度列表
        """
        super(CriticObsActEncoder, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.embed_dim = embed_dim
        
        # 构建 MLP 层
        layers = []
        input_dim = state_dim + action_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        # 最终输出层
        layers.append(nn.Linear(input_dim, embed_dim))
        
        self.mlp = nn.Sequential(*layers)
        
        # 使用 Xavier 初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, obs, act):
        """前向传播
        
        Args:
            obs: 观测张量 [..., state_dim]
            act: 动作张量 [..., action_dim]
            
        Returns:
            embedding: 编码后的嵌入 [..., embed_dim]
        """
        # 拼接观测和动作
        sa = torch.cat([obs, act], dim=-1)
        
        # 编码
        return self.mlp(sa)

class QHead(nn.Module):
    """Q 值头部
    
    双 Q 结构，用于计算 Q 值
    """
    def __init__(self, input_dim, hidden_dims=[256, 128]):
        """初始化 Q 值头部
        
        Args:
            input_dim: 输入维度
            hidden_dims: 隐藏层维度列表
        """
        super(QHead, self).__init__()
        
        # 共享层
        layers = []
        current_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            current_dim = hidden_dim
        
        self.shared_layers = nn.Sequential(*layers)
        
        # 双 Q 输出层
        self.q1_out = nn.Linear(hidden_dims[-1], 1)
        self.q2_out = nn.Linear(hidden_dims[-1], 1)
        
        # 使用 Xavier 初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Q 值输出层使用较小的初始化值
        nn.init.uniform_(self.q1_out.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.q2_out.weight, -3e-3, 3e-3)
        nn.init.zeros_(self.q1_out.bias)
        nn.init.zeros_(self.q2_out.bias)
    
    def forward(self, x):
        """前向传播
        
        Args:
            x: 输入特征 [..., input_dim]
            
        Returns:
            q1: 第一个 Q 值 [..., 1]
            q2: 第二个 Q 值 [..., 1]
        """
        features = self.shared_layers(x)
        q1 = self.q1_out(features)
        q2 = self.q2_out(features)
        return q1, q2

class SimpleCriticNet(nn.Module):
    """简单的Critic网络，无注意力机制
    
    使用简单的全连接层替代注意力机制
    """
    def __init__(self, state_dim, action_dim, hidden_dims=[256, 128]):
        """初始化简单Critic网络
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            hidden_dims: 隐藏层维度列表
        """
        super(SimpleCriticNet, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        embed_dim = 128  # 固定嵌入维度
        
        # 编码器
        self.leader_encoder = CriticObsActEncoder(state_dim, action_dim, embed_dim)
        self.follower_encoder = CriticObsActEncoder(state_dim, action_dim, embed_dim)
        
        # 简单聚合层（替代注意力）
        # Leader聚合：自身 + 3个follower的平均
        self.leader_aggregator = nn.Linear(embed_dim * 2, embed_dim)
        
        # Follower聚合：自身 + leader
        self.follower_aggregator = nn.Linear(embed_dim * 2, embed_dim)
        
        # Q头
        self.leader_q_head = QHead(embed_dim, hidden_dims)
        self.follower_q_head = QHead(embed_dim, hidden_dims)
        
        # 目标网络
        self.target_leader_encoder = CriticObsActEncoder(state_dim, action_dim, embed_dim)
        self.target_follower_encoder = CriticObsActEncoder(state_dim, action_dim, embed_dim)
        self.target_leader_aggregator = nn.Linear(embed_dim * 2, embed_dim)
        self.target_follower_aggregator = nn.Linear(embed_dim * 2, embed_dim)
        self.target_leader_q_head = QHead(embed_dim, hidden_dims)
        self.target_follower_q_head = QHead(embed_dim, hidden_dims)
        
        # 加载目标网络初始参数
        self._load_initial_target_params()
    
    def _load_initial_target_params(self):
        """加载目标网络初始参数"""
        self.target_leader_encoder.load_state_dict(self.leader_encoder.state_dict())
        self.target_follower_encoder.load_state_dict(self.follower_encoder.state_dict())
        self.target_leader_aggregator.load_state_dict(self.leader_aggregator.state_dict())
        self.target_follower_aggregator.load_state_dict(self.follower_aggregator.state_dict())
        self.target_leader_q_head.load_state_dict(self.leader_q_head.state_dict())
        self.target_follower_q_head.load_state_dict(self.follower_q_head.state_dict())
    
    def forward(self, obs_leader, obs_followers, act_leader, act_followers, mask_followers):
        """前向传播
        
        Args:
            obs_leader: Leader 的观测 [batch_size, state_dim]
            obs_followers: Followers 的观测 [batch_size, max_followers, state_dim]
            act_leader: Leader 的动作 [batch_size, action_dim]
            act_followers: Followers 的动作 [batch_size, max_followers, action_dim]
            mask_followers: Followers 的掩码 [batch_size, max_followers]
            
        Returns:
            q1_leader: Leader 的第一个 Q 值 [batch_size, 1]
            q2_leader: Leader 的第二个 Q 值 [batch_size, 1]
            q1_followers: Followers 的第一个 Q 值 [batch_size, max_followers, 1]
            q2_followers: Followers 的第二个 Q 值 [batch_size, max_followers, 1]
        """
        B = obs_leader.shape[0]
        max_F = obs_followers.shape[1]
        
        # 编码 Leader
        leader_embedding = self.leader_encoder(obs_leader, act_leader)  # [B, E]
        
        # 编码 Followers
        obs_f_flat = obs_followers.reshape(B * max_F, self.state_dim)  # [B*max_F, D_s]
        act_f_flat = act_followers.reshape(B * max_F, self.action_dim)  # [B*max_F, D_a]
        follower_embeds_flat = self.follower_encoder(obs_f_flat, act_f_flat)  # [B*max_F, E]
        follower_embeddings = follower_embeds_flat.reshape(B, max_F, 128)  # [B, max_F, E]
        
        # === Leader Q 值计算 ===
        # 计算 Follower 平均嵌入（替代注意力）
        follower_avg = torch.mean(follower_embeddings, dim=1)  # [B, E]
        
        # 聚合 Leader 信息
        leader_context = torch.cat([leader_embedding, follower_avg], dim=-1)  # [B, E*2]
        leader_context = self.leader_aggregator(leader_context)  # [B, E]
        
        # 计算 Leader Q 值
        q1_leader, q2_leader = self.leader_q_head(leader_context)  # [B, 1], [B, 1]
        
        # === Follower Q 值计算 ===
        # 扩展 Leader 嵌入以匹配 Follower 数量
        leader_expanded = leader_embedding.unsqueeze(1).expand(-1, max_F, -1)  # [B, max_F, E]
        
        # 聚合每个 Follower 的信息
        follower_context = torch.cat([follower_embeddings, leader_expanded], dim=-1)  # [B, max_F, E*2]
        follower_context = self.follower_aggregator(follower_context)  # [B, max_F, E]
        
        # 计算 Follower Q 值
        follower_context_flat = follower_context.reshape(B * max_F, 128)  # [B*max_F, E]
        q1_followers_flat, q2_followers_flat = self.follower_q_head(follower_context_flat)  # [B*max_F, 1], [B*max_F, 1]
        q1_followers = q1_followers_flat.reshape(B, max_F, 1)  # [B, max_F, 1]
        q2_followers = q2_followers_flat.reshape(B, max_F, 1)  # [B, max_F, 1]
        
        return q1_leader, q2_leader, q1_followers, q2_followers
    
    def forward_target(self, obs_leader, obs_followers, act_leader, act_followers, mask_followers):
        """目标网络前向传播（别名方法）
        
        Args:
            同 target_forward 方法
            
        Returns:
            同 target_forward 方法
        """
        return self.target_forward(obs_leader, obs_followers, act_leader, act_followers, mask_followers)
    
    def target_forward(self, obs_leader, obs_followers, act_leader, act_followers, mask_followers):
        """目标网络前向传播
        
        Args:
            同 forward 方法
            
        Returns:
            同 forward 方法
        """
        B = obs_leader.shape[0]
        max_F = obs_followers.shape[1]
        
        # 编码 Leader
        leader_embedding = self.target_leader_encoder(obs_leader, act_leader)  # [B, E]
        
        # 编码 Followers
        obs_f_flat = obs_followers.reshape(B * max_F, self.state_dim)  # [B*max_F, D_s]
        act_f_flat = act_followers.reshape(B * max_F, self.action_dim)  # [B*max_F, D_a]
        follower_embeds_flat = self.target_follower_encoder(obs_f_flat, act_f_flat)  # [B*max_F, E]
        follower_embeddings = follower_embeds_flat.reshape(B, max_F, 128)  # [B, max_F, E]
        
        # === Leader Q 值计算 ===
        # 计算 Follower 平均嵌入
        follower_avg = torch.mean(follower_embeddings, dim=1)  # [B, E]
        
        # 聚合 Leader 信息
        leader_context = torch.cat([leader_embedding, follower_avg], dim=-1)  # [B, E*2]
        leader_context = self.target_leader_aggregator(leader_context)  # [B, E]
        
        # 计算 Leader Q 值
        q1_leader, q2_leader = self.target_leader_q_head(leader_context)  # [B, 1], [B, 1]
        
        # === Follower Q 值计算 ===
        # 扩展 Leader 嵌入以匹配 Follower 数量
        leader_expanded = leader_embedding.unsqueeze(1).expand(-1, max_F, -1)  # [B, max_F, E]
        
        # 聚合每个 Follower 的信息
        follower_context = torch.cat([follower_embeddings, leader_expanded], dim=-1)  # [B, max_F, E*2]
        follower_context = self.target_follower_aggregator(follower_context)  # [B, max_F, E]
        
        # 计算 Follower Q 值
        follower_context_flat = follower_context.reshape(B * max_F, 128)  # [B*max_F, E]
        q1_followers_flat, q2_followers_flat = self.target_follower_q_head(follower_context_flat)  # [B*max_F, 1], [B*max_F, 1]
        q1_followers = q1_followers_flat.reshape(B, max_F, 1)  # [B, max_F, 1]
        q2_followers = q2_followers_flat.reshape(B, max_F, 1)  # [B, max_F, 1]
        
        return q1_leader, q2_leader, q1_followers, q2_followers
    
    def soft_update_targets(self, tau):
        """软更新目标网络
        
        Args:
            tau: 软更新系数
        """
        for target_param, param in zip(self.target_leader_encoder.parameters(), self.leader_encoder.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        
        for target_param, param in zip(self.target_follower_encoder.parameters(), self.follower_encoder.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        
        for target_param, param in zip(self.target_leader_aggregator.parameters(), self.leader_aggregator.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        
        for target_param, param in zip(self.target_follower_aggregator.parameters(), self.follower_aggregator.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        
        for target_param, param in zip(self.target_leader_q_head.parameters(), self.leader_q_head.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        
        for target_param, param in zip(self.target_follower_q_head.parameters(), self.follower_q_head.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data) 
    
    def soft_update(self, tau):
        """软更新目标网络（别名方法）
        
        Args:
            tau: 软更新系数
        """
        self.soft_update_targets(tau) 