import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 从 masac_adapter 导入注意力机制和角色 ID 常量
from masac_adapter.masac_adapter import GraphAttentionLayer, build_spatial_adjacency, build_formation_adjacency, LEADER_TYPE_ID, FOLLOWER_TYPE_ID

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

class StructuredAttentionCriticNet(nn.Module):
    """结构化注意力 Critic 网络
    
    处理结构化的观测和动作，使用注意力机制聚合信息
    """
    def __init__(self, state_dim, action_dim, embed_dim=128, n_heads=4, hidden_dims=[256, 128], dropout=0.1):
        """初始化结构化注意力 Critic 网络
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            embed_dim: 嵌入维度
            n_heads: 注意力头数
            hidden_dims: 隐藏层维度列表
            dropout: Dropout 概率
        """
        super(StructuredAttentionCriticNet, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        
        # 编码器
        self.leader_encoder = CriticObsActEncoder(state_dim, action_dim, embed_dim)
        self.follower_encoder = CriticObsActEncoder(state_dim, action_dim, embed_dim)
        
        # 注意力层 (Leader关注Followers - 重命名)
        self.leader_sees_followers_attention = GraphAttentionLayer(embed_dim, n_heads, dropout)
        
        # 新增: Follower上下文注意力层
        self.follower_context_attention = GraphAttentionLayer(embed_dim, n_heads, dropout)
        self.leader_context_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid()
        )
        self.follower_context_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid()
        )
        
        # Q 头
        # Leader的输入维度保持不变: embed_dim * 2 (自身编码 + Follower上下文)
        q_input_dim_leader = embed_dim * 2
        # Follower的输入维度从embed_dim变为embed_dim * 2 (自身编码 + 全局上下文)
        q_input_dim_follower = embed_dim * 2
        
        self.leader_q_head = QHead(q_input_dim_leader, hidden_dims)
        self.follower_q_head = QHead(q_input_dim_follower, hidden_dims)
        
        # 目标网络
        self.target_leader_encoder = CriticObsActEncoder(state_dim, action_dim, embed_dim)
        self.target_follower_encoder = CriticObsActEncoder(state_dim, action_dim, embed_dim)
        
        # 目标注意力层 (Leader关注Followers - 重命名)
        self.target_leader_sees_followers_attention = GraphAttentionLayer(embed_dim, n_heads, dropout)
        
        # 新增: 目标Follower上下文注意力层
        self.target_follower_context_attention = GraphAttentionLayer(embed_dim, n_heads, dropout)
        self.target_leader_context_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid()
        )
        self.target_follower_context_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid()
        )
        
        self.target_leader_q_head = QHead(q_input_dim_leader, hidden_dims)
        self.target_follower_q_head = QHead(q_input_dim_follower, hidden_dims)
        
        # 加载目标网络初始参数
        self._load_initial_target_params()
    
    def _load_initial_target_params(self):
        """加载目标网络初始参数"""
        self.target_leader_encoder.load_state_dict(self.leader_encoder.state_dict())
        self.target_follower_encoder.load_state_dict(self.follower_encoder.state_dict())
        self.target_leader_sees_followers_attention.load_state_dict(self.leader_sees_followers_attention.state_dict())
        self.target_follower_context_attention.load_state_dict(self.follower_context_attention.state_dict())
        self.target_leader_context_gate.load_state_dict(self.leader_context_gate.state_dict())
        self.target_follower_context_gate.load_state_dict(self.follower_context_gate.state_dict())
        self.target_leader_q_head.load_state_dict(self.leader_q_head.state_dict())
        self.target_follower_q_head.load_state_dict(self.follower_q_head.state_dict())
    
    def forward(self, obs_leader, obs_followers, act_leader, act_followers, mask_followers):
        """前向传播
        
        Args:
            obs_leader: Leader 的观测 [batch_size, state_dim]
            obs_followers: Followers 的观测 [batch_size, max_followers, state_dim]
            act_leader: Leader 的动作 [batch_size, action_dim]
            act_followers: Followers 的动作 [batch_size, max_followers, action_dim]
            mask_followers: Followers 的掩码 [batch_size, max_followers]，True表示有效
            
        Returns:
            q1_leader: Leader 的第一个 Q 值 [batch_size, 1]
            q2_leader: Leader 的第二个 Q 值 [batch_size, 1]
            q1_followers: Followers 的第一个 Q 值 [batch_size, max_followers, 1]
            q2_followers: Followers 的第二个 Q 值 [batch_size, max_followers, 1]
        """
        B = obs_leader.shape[0]
        max_F = obs_followers.shape[1]
        device = obs_leader.device
        mask_followers_bool = mask_followers.to(device=device).bool()
        
        # 编码 Leader
        leader_embedding = self.leader_encoder(obs_leader, act_leader)  # [B, E]
        
        # 编码 Followers (处理 B, max_F, D -> B*max_F, D -> B*max_F, E -> B, max_F, E)
        if max_F > 0:
            obs_f_flat = obs_followers.reshape(B * max_F, self.state_dim)  # [B*max_F, D_s]
            act_f_flat = act_followers.reshape(B * max_F, self.action_dim)  # [B*max_F, D_a]
            follower_embeds_flat = self.follower_encoder(obs_f_flat, act_f_flat)  # [B*max_F, E]
            follower_embeddings = follower_embeds_flat.reshape(B, max_F, self.embed_dim)  # [B, max_F, E]
        else:
            follower_embeddings = torch.empty(B, 0, self.embed_dim, device=device, dtype=leader_embedding.dtype)
        
        # === 构建智能体图：节点0为Leader，后续为Followers ===
        all_agents_embeddings = torch.cat([leader_embedding.unsqueeze(1), follower_embeddings], dim=1)  # [B, 1+F, E]
        all_agents_obs = torch.cat([obs_leader.unsqueeze(1), obs_followers], dim=1)  # [B, 1+F, S]
        leader_mask = torch.ones(B, 1, dtype=torch.bool, device=device)
        node_mask = torch.cat([leader_mask, mask_followers_bool], dim=1)  # [B, 1+F]
        adjacency = build_formation_adjacency(
            all_agents_obs,
            node_mask=node_mask,
            leader_index=0,
            distance_threshold=0.65,
            k_neighbors=2,
            force_leader_edges=True
        )
        
        # Leader与Follower分别使用独立GAT参数，保留原有结构中的角色差异。
        leader_graph_embeddings = self.leader_sees_followers_attention(
            all_agents_embeddings,
            adjacency=adjacency,
            node_mask=node_mask
        )
        follower_graph_embeddings = self.follower_context_attention(
            all_agents_embeddings,
            adjacency=adjacency,
            node_mask=node_mask
        )
        
        # === Leader Q 值计算 ===
        raw_leader_context = leader_graph_embeddings[:, 0, :]  # [B, E]
        leader_gate = self.leader_context_gate(torch.cat([leader_embedding, raw_leader_context], dim=-1))
        leader_context = leader_gate * raw_leader_context + (1.0 - leader_gate) * leader_embedding
        fused_leader_feature = torch.cat([leader_embedding, leader_context], dim=-1)  # [B, 2E]
        q1_leader, q2_leader = self.leader_q_head(fused_leader_feature)
        
        # === Follower Q 值计算 ===
        raw_follower_contexts = follower_graph_embeddings[:, 1:, :]  # [B, F, E]
        follower_gate = self.follower_context_gate(torch.cat([follower_embeddings, raw_follower_contexts], dim=-1))
        follower_contexts = follower_gate * raw_follower_contexts + (1.0 - follower_gate) * follower_embeddings
        fused_follower_feature = torch.cat([follower_embeddings, follower_contexts], dim=-1)  # [B, F, 2E]
        
        q1_followers = torch.zeros(B, max_F, 1, device=device, dtype=leader_embedding.dtype)
        q2_followers = torch.zeros(B, max_F, 1, device=device, dtype=leader_embedding.dtype)
        for i in range(max_F):
            f_fused_i = fused_follower_feature[:, i, :]  # [B, 2E]
            q1_f_i, q2_f_i = self.follower_q_head(f_fused_i)
            valid_i = mask_followers_bool[:, i:i+1].to(q1_f_i.dtype)
            q1_followers[:, i, :] = q1_f_i * valid_i
            q2_followers[:, i, :] = q2_f_i * valid_i
        
        return q1_leader, q2_leader, q1_followers, q2_followers
    
    def forward_target(self, obs_leader, obs_followers, act_leader, act_followers, mask_followers):
        """使用目标网络计算 Q 值"""
        with torch.no_grad():
            B = obs_leader.shape[0]
            max_F = obs_followers.shape[1]
            device = obs_leader.device
            mask_followers_bool = mask_followers.to(device=device).bool()
            
            # 编码 Leader
            leader_embedding = self.target_leader_encoder(obs_leader, act_leader)  # [B, E]
            
            # 编码 Followers
            if max_F > 0:
                obs_f_flat = obs_followers.reshape(B * max_F, self.state_dim)
                act_f_flat = act_followers.reshape(B * max_F, self.action_dim)
                follower_embeds_flat = self.target_follower_encoder(obs_f_flat, act_f_flat)
                follower_embeddings = follower_embeds_flat.reshape(B, max_F, self.embed_dim)
            else:
                follower_embeddings = torch.empty(B, 0, self.embed_dim, device=device, dtype=leader_embedding.dtype)
            
            # 构建目标网络智能体图
            all_agents_embeddings = torch.cat([leader_embedding.unsqueeze(1), follower_embeddings], dim=1)
            all_agents_obs = torch.cat([obs_leader.unsqueeze(1), obs_followers], dim=1)
            leader_mask = torch.ones(B, 1, dtype=torch.bool, device=device)
            node_mask = torch.cat([leader_mask, mask_followers_bool], dim=1)
            adjacency = build_formation_adjacency(
                all_agents_obs,
                node_mask=node_mask,
                leader_index=0,
                distance_threshold=0.65,
                k_neighbors=2,
                force_leader_edges=True
            )
            
            leader_graph_embeddings = self.target_leader_sees_followers_attention(
                all_agents_embeddings,
                adjacency=adjacency,
                node_mask=node_mask
            )
            follower_graph_embeddings = self.target_follower_context_attention(
                all_agents_embeddings,
                adjacency=adjacency,
                node_mask=node_mask
            )
            
            # Leader Q
            raw_leader_context = leader_graph_embeddings[:, 0, :]
            leader_gate = self.target_leader_context_gate(torch.cat([leader_embedding, raw_leader_context], dim=-1))
            leader_context = leader_gate * raw_leader_context + (1.0 - leader_gate) * leader_embedding
            fused_leader_feature = torch.cat([leader_embedding, leader_context], dim=-1)
            q1_leader, q2_leader = self.target_leader_q_head(fused_leader_feature)
            
            # Follower Q
            raw_follower_contexts = follower_graph_embeddings[:, 1:, :]
            follower_gate = self.target_follower_context_gate(torch.cat([follower_embeddings, raw_follower_contexts], dim=-1))
            follower_contexts = follower_gate * raw_follower_contexts + (1.0 - follower_gate) * follower_embeddings
            fused_follower_feature = torch.cat([follower_embeddings, follower_contexts], dim=-1)
            q1_followers = torch.zeros(B, max_F, 1, device=device, dtype=leader_embedding.dtype)
            q2_followers = torch.zeros(B, max_F, 1, device=device, dtype=leader_embedding.dtype)
            for i in range(max_F):
                f_fused_i = fused_follower_feature[:, i, :]
                q1_f_i, q2_f_i = self.target_follower_q_head(f_fused_i)
                valid_i = mask_followers_bool[:, i:i+1].to(q1_f_i.dtype)
                q1_followers[:, i, :] = q1_f_i * valid_i
                q2_followers[:, i, :] = q2_f_i * valid_i
            
            return q1_leader, q2_leader, q1_followers, q2_followers
    
    def soft_update(self, tau):
        """软更新目标网络参数
        
        Args:
            tau: 软更新系数
        """
        # 更新 Leader 编码器
        for target_param, param in zip(self.target_leader_encoder.parameters(), self.leader_encoder.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        
        # 更新 Follower 编码器
        for target_param, param in zip(self.target_follower_encoder.parameters(), self.follower_encoder.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        
        # 更新 Leader关注Followers注意力层
        for target_param, param in zip(self.target_leader_sees_followers_attention.parameters(), self.leader_sees_followers_attention.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        
        # 更新 Follower上下文注意力层 (新增)
        for target_param, param in zip(self.target_follower_context_attention.parameters(), self.follower_context_attention.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        
        for target_param, param in zip(self.target_leader_context_gate.parameters(), self.leader_context_gate.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        
        for target_param, param in zip(self.target_follower_context_gate.parameters(), self.follower_context_gate.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        
        # 更新 Q 头
        for target_param, param in zip(self.target_leader_q_head.parameters(), self.leader_q_head.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        for target_param, param in zip(self.target_follower_q_head.parameters(), self.follower_q_head.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau) 