import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Import attention mechanism and role ID constants from masac_adapter
from integrated_ablation_modules.masac_no_curriculum.masac_adapter import GraphAttentionLayer, build_spatial_adjacency, build_formation_adjacency, LEADER_TYPE_ID, FOLLOWER_TYPE_ID

class CriticObsActEncoder(nn.Module):
    """Critic observation-action encoder
    
    Encodes observations and actions into hidden representations
    """
    def __init__(self, state_dim, action_dim, embed_dim, hidden_dims=[256, 128]):
        """Initialize observation-action encoder
        
        Args:
            state_dim: State dimension
            action_dim: Action dimension
            embed_dim: Output embedding dimension
            hidden_dims: Hidden layer dimension list
        """
        super(CriticObsActEncoder, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.embed_dim = embed_dim
        
        # Build MLP layers
        layers = []
        input_dim = state_dim + action_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        # Final output layer
        layers.append(nn.Linear(input_dim, embed_dim))
        
        self.mlp = nn.Sequential(*layers)
        
        # Use Xavier initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, obs, act):
        """Forward propagation
        
        Args:
            obs: Observation tensor [..., state_dim]
            act: Action tensor [..., action_dim]
            
        Returns:
            embedding: Encoded embedding [..., embed_dim]
        """
        # Concatenate observation and action
        sa = torch.cat([obs, act], dim=-1)
        
        # Encode
        return self.mlp(sa)

class QHead(nn.Module):
    """Q value head
    
    Dual Q structure for computing Q values
    """
    def __init__(self, input_dim, hidden_dims=[256, 128]):
        """Initialize Q value head
        
        Args:
            input_dim: Input dimension
            hidden_dims: Hidden layer dimension list
        """
        super(QHead, self).__init__()
        
        # Shared layers
        layers = []
        current_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            current_dim = hidden_dim
        
        self.shared_layers = nn.Sequential(*layers)
        
        # Dual Q output layers
        self.q1_out = nn.Linear(hidden_dims[-1], 1)
        self.q2_out = nn.Linear(hidden_dims[-1], 1)
        
        # Use Xavier initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Use smaller initialization values for Q value output layers
        nn.init.uniform_(self.q1_out.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.q2_out.weight, -3e-3, 3e-3)
        nn.init.zeros_(self.q1_out.bias)
        nn.init.zeros_(self.q2_out.bias)
    
    def forward(self, x):
        """Forward propagation
        
        Args:
            x: Input features [..., input_dim]
            
        Returns:
            q1: First Q value [..., 1]
            q2: Second Q value [..., 1]
        """
        features = self.shared_layers(x)
        q1 = self.q1_out(features)
        q2 = self.q2_out(features)
        return q1, q2

class StructuredAttentionCriticNet(nn.Module):
    """Structured Attention Critic Network
    
    Processes structured observations and actions, using attention mechanism to aggregate information
    """
    def __init__(self, state_dim, action_dim, embed_dim=128, n_heads=4, hidden_dims=[256, 128], dropout=0.1):
        """Initialize structured attention Critic network
        
        Args:
            state_dim: State dimension
            action_dim: Action dimension
            embed_dim: Embedding dimension
            n_heads: Number of attention heads
            hidden_dims: Hidden layer dimension list
            dropout: Dropout probability
        """
        super(StructuredAttentionCriticNet, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        
        # Encoders
        self.leader_encoder = CriticObsActEncoder(state_dim, action_dim, embed_dim)
        self.follower_encoder = CriticObsActEncoder(state_dim, action_dim, embed_dim)
        
        # Attention layer (Leader attends to Followers - renamed)
        self.leader_sees_followers_attention = GraphAttentionLayer(embed_dim, n_heads, dropout)
        
        # New: Follower context attention layer
        self.follower_context_attention = GraphAttentionLayer(embed_dim, n_heads, dropout)
        self.leader_context_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid()
        )
        self.follower_context_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid()
        )
        
        # Q heads
        # Leader input dimension remains unchanged: embed_dim * 2 (self encoding + Follower context)
        q_input_dim_leader = embed_dim * 2
        # Follower input dimension changes from embed_dim to embed_dim * 2 (self encoding + global context)
        q_input_dim_follower = embed_dim * 2
        
        self.leader_q_head = QHead(q_input_dim_leader, hidden_dims)
        self.follower_q_head = QHead(q_input_dim_follower, hidden_dims)
        
        # Target networks
        self.target_leader_encoder = CriticObsActEncoder(state_dim, action_dim, embed_dim)
        self.target_follower_encoder = CriticObsActEncoder(state_dim, action_dim, embed_dim)
        
        # Target attention layer (Leader attends to Followers - renamed)
        self.target_leader_sees_followers_attention = GraphAttentionLayer(embed_dim, n_heads, dropout)
        
        # New: Target Follower context attention layer
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
        
        # Load initial target network parameters
        self._load_initial_target_params()
    
    def _load_initial_target_params(self):
        """Load initial target network parameters"""
        self.target_leader_encoder.load_state_dict(self.leader_encoder.state_dict())
        self.target_follower_encoder.load_state_dict(self.follower_encoder.state_dict())
        self.target_leader_sees_followers_attention.load_state_dict(self.leader_sees_followers_attention.state_dict())
        self.target_follower_context_attention.load_state_dict(self.follower_context_attention.state_dict())
        self.target_leader_context_gate.load_state_dict(self.leader_context_gate.state_dict())
        self.target_follower_context_gate.load_state_dict(self.follower_context_gate.state_dict())
        self.target_leader_q_head.load_state_dict(self.leader_q_head.state_dict())
        self.target_follower_q_head.load_state_dict(self.follower_q_head.state_dict())
    
    def forward(self, obs_leader, obs_followers, act_leader, act_followers):
        """Forward propagation"""
        B = obs_leader.shape[0]
        max_F = obs_followers.shape[1]
        device = obs_leader.device
        
        # 编码 Leader
        leader_embedding = self.leader_encoder(obs_leader, act_leader)  # [B, E]
        
        # 编码 Followers
        if max_F > 0:
            obs_f_flat = obs_followers.reshape(B * max_F, self.state_dim)
            act_f_flat = act_followers.reshape(B * max_F, self.action_dim)
            follower_embeds_flat = self.follower_encoder(obs_f_flat, act_f_flat)
            follower_embeddings = follower_embeds_flat.reshape(B, max_F, self.embed_dim)
        else:
            follower_embeddings = torch.empty(B, 0, self.embed_dim, device=device, dtype=leader_embedding.dtype)
        
        # === 构建智能体图：节点0为Leader，后续为Followers ===
        all_agents_embeddings = torch.cat([leader_embedding.unsqueeze(1), follower_embeddings], dim=1)
        all_agents_obs = torch.cat([obs_leader.unsqueeze(1), obs_followers], dim=1)
        node_mask = torch.ones(B, 1 + max_F, dtype=torch.bool, device=device)
        adjacency = build_formation_adjacency(
            all_agents_obs,
            node_mask=node_mask,
            leader_index=0,
            distance_threshold=0.65,
            k_neighbors=2,
            force_leader_edges=True
        )
        
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
        
        raw_leader_context = leader_graph_embeddings[:, 0, :]
        leader_gate = self.leader_context_gate(torch.cat([leader_embedding, raw_leader_context], dim=-1))
        leader_context = leader_gate * raw_leader_context + (1.0 - leader_gate) * leader_embedding
        fused_leader_feature = torch.cat([leader_embedding, leader_context], dim=-1)
        q1_leader, q2_leader = self.leader_q_head(fused_leader_feature)

        raw_follower_contexts = follower_graph_embeddings[:, 1:, :]
        follower_gate = self.follower_context_gate(torch.cat([follower_embeddings, raw_follower_contexts], dim=-1))
        follower_contexts = follower_gate * raw_follower_contexts + (1.0 - follower_gate) * follower_embeddings
        fused_follower_feature = torch.cat([follower_embeddings, follower_contexts], dim=-1)
        q1_followers = torch.zeros(B, max_F, 1, device=device, dtype=leader_embedding.dtype)
        q2_followers = torch.zeros(B, max_F, 1, device=device, dtype=leader_embedding.dtype)
        for i in range(max_F):
            f_fused_i = fused_follower_feature[:, i, :]
            q1_f_i, q2_f_i = self.follower_q_head(f_fused_i)
            q1_followers[:, i, :] = q1_f_i
            q2_followers[:, i, :] = q2_f_i
        
        return q1_leader, q2_leader, q1_followers, q2_followers
    
    def forward_target(self, obs_leader, obs_followers, act_leader, act_followers):
        """使用目标网络计算 Q 值"""
        with torch.no_grad():
            B = obs_leader.shape[0]
            max_F = obs_followers.shape[1]
            device = obs_leader.device
            
            leader_embedding = self.target_leader_encoder(obs_leader, act_leader)
            if max_F > 0:
                obs_f_flat = obs_followers.reshape(B * max_F, self.state_dim)
                act_f_flat = act_followers.reshape(B * max_F, self.action_dim)
                follower_embeds_flat = self.target_follower_encoder(obs_f_flat, act_f_flat)
                follower_embeddings = follower_embeds_flat.reshape(B, max_F, self.embed_dim)
            else:
                follower_embeddings = torch.empty(B, 0, self.embed_dim, device=device, dtype=leader_embedding.dtype)
            
            all_agents_embeddings = torch.cat([leader_embedding.unsqueeze(1), follower_embeddings], dim=1)
            all_agents_obs = torch.cat([obs_leader.unsqueeze(1), obs_followers], dim=1)
            node_mask = torch.ones(B, 1 + max_F, dtype=torch.bool, device=device)
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
            
            raw_leader_context = leader_graph_embeddings[:, 0, :]
            leader_gate = self.target_leader_context_gate(torch.cat([leader_embedding, raw_leader_context], dim=-1))
            leader_context = leader_gate * raw_leader_context + (1.0 - leader_gate) * leader_embedding
            fused_leader_feature = torch.cat([leader_embedding, leader_context], dim=-1)
            q1_leader, q2_leader = self.target_leader_q_head(fused_leader_feature)
            
            raw_follower_contexts = follower_graph_embeddings[:, 1:, :]
            follower_gate = self.target_follower_context_gate(torch.cat([follower_embeddings, raw_follower_contexts], dim=-1))
            follower_contexts = follower_gate * raw_follower_contexts + (1.0 - follower_gate) * follower_embeddings
            fused_follower_feature = torch.cat([follower_embeddings, follower_contexts], dim=-1)
            q1_followers = torch.zeros(B, max_F, 1, device=device, dtype=leader_embedding.dtype)
            q2_followers = torch.zeros(B, max_F, 1, device=device, dtype=leader_embedding.dtype)
            for i in range(max_F):
                f_fused_i = fused_follower_feature[:, i, :]
                q1_f_i, q2_f_i = self.target_follower_q_head(f_fused_i)
                q1_followers[:, i, :] = q1_f_i
                q2_followers[:, i, :] = q2_f_i
            
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
