import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional

# 基本配置常量
ROLE_EMBED_DIM = 16    # 角色嵌入维度
EMBED_DIM = 128        # 通用嵌入维度

# 导入常量
from masac_adapter.masac_adapter import LEADER_TYPE_ID, FOLLOWER_TYPE_ID

class RoleEmbedding(nn.Module):
    """角色嵌入模块
    
    为主机(Leader)和从机(Follower)提供可学习的嵌入表示
    """
    def __init__(self, num_embeddings=2, embedding_dim=ROLE_EMBED_DIM):
        """初始化角色嵌入模块
        
        Args:
            num_embeddings: 嵌入表大小，默认为2（主机和从机）
            embedding_dim: 嵌入维度
        """
        super(RoleEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        
        # 使用Xavier初始化
        nn.init.xavier_uniform_(self.embedding.weight)
        
    def forward(self, role_ids):
        """获取角色嵌入
        
        Args:
            role_ids: 角色ID张量 [batch_size] 或 [batch_size, 1]
            
        Returns:
            嵌入张量 [batch_size, embedding_dim]
        """
        # 确保role_ids是适当的形状
        if role_ids.dim() > 1 and role_ids.size(1) == 1:
            role_ids = role_ids.squeeze(1)
            
        return self.embedding(role_ids)

class PolicyNetFlatRole(nn.Module):
    """基于角色的策略网络，处理扁平状态输入
    
    接收扁平状态和角色嵌入，输出动作均值和标准差
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256, role_embed_dim=ROLE_EMBED_DIM):
        """初始化策略网络
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            hidden_dim: 隐藏层维度
            role_embed_dim: 角色嵌入维度
        """
        super(PolicyNetFlatRole, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.role_embed_dim = role_embed_dim
        
        # 状态和角色信息的MLP处理网络
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim + role_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 动作均值和对数标准差输出
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)
        
        # 使用Xavier初始化
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        """前向传播
        
        Args:
            x: 输入张量 [batch_size, state_dim + role_embed_dim]
                这已经是拼接过的状态和角色嵌入
            
        Returns:
            mean: 动作均值 [batch_size, action_dim]
            std: 动作标准差 [batch_size, action_dim]
        """
        # 通过MLP网络
        features = self.policy_net(x)
        
        # 生成均值（使用tanh限制范围）
        mean = torch.tanh(self.mean_layer(features))
        
        # 计算对数标准差，并限制在合理范围内，避免训练不稳定
        log_std = self.log_std_layer(features)
        log_std = torch.clamp(log_std, -20, 2)  # 防止标准差过大或过小
        
        # 计算标准差
        std = torch.exp(log_std)
        
        return mean, std

class SharedEncoder(nn.Module):
    """共享编码器
    
    将状态-动作对编码为隐藏表示
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256, output_dim=EMBED_DIM):
        """初始化共享编码器
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            hidden_dim: 隐藏层维度
            output_dim: 输出维度
        """
        super(SharedEncoder, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.output_dim = output_dim
        
        # 状态-动作编码网络
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # 使用Xavier初始化
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        """前向传播
        
        Args:
            x: 状态-动作张量 [batch_size, state_dim + action_dim]
            
        Returns:
            encoder_output: 编码后的张量 [batch_size, output_dim]
        """
        return self.encoder(x)

class QHead(nn.Module):
    """Q值头部
    
    实现双Q结构，用于计算Q值
    """
    def __init__(self, input_dim=EMBED_DIM*2, hidden_dim=256):
        """初始化Q值头部
        
        Args:
            input_dim: 输入维度，默认为EMBED_DIM*2（编码+注意力输出的拼接）
            hidden_dim: 隐藏层维度
        """
        super(QHead, self).__init__()
        
        # Q网络共享的初始层
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 双Q结构的两个独立输出层
        self.q1_out = nn.Linear(hidden_dim, 1)
        self.q2_out = nn.Linear(hidden_dim, 1)
        
        # 使用Xavier初始化
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        
        # Q值输出层使用较小的初始化值
        nn.init.uniform_(self.q1_out.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.q2_out.weight, -3e-3, 3e-3)
        nn.init.zeros_(self.q1_out.bias)
        nn.init.zeros_(self.q2_out.bias)
    
    def forward(self, x):
        """前向传播
        
        Args:
            x: 输入特征 [batch_size, input_dim]
            
        Returns:
            q1: 第一个Q值 [batch_size, 1]
            q2: 第二个Q值 [batch_size, 1]
        """
        # 通过共享层
        features = self.shared_layers(x)
        
        # 计算两个Q值输出
        q1 = self.q1_out(features)
        q2 = self.q2_out(features)
        
        return q1, q2

class CriticNetAttentionFlat(nn.Module):
    """基于注意力的扁平Critic网络
    
    使用共享编码器和注意力机制处理多智能体，以及角色特定的Q头
    """
    def __init__(self, shared_encoder, shared_attention, q_head, target_q_head, target_encoder=None, target_attention=None):
        """初始化Critic网络
        
        Args:
            shared_encoder: 共享的状态-动作编码器
            shared_attention: 共享的注意力机制
            q_head: 当前Q值头
            target_q_head: 目标Q值头
            target_encoder: 目标编码器，如果不提供则使用shared_encoder
            target_attention: 目标注意力机制，如果不提供则使用shared_attention
        """
        super(CriticNetAttentionFlat, self).__init__()
        
        self.shared_encoder = shared_encoder
        self.shared_attention = shared_attention
        self.q_head = q_head
        self.target_q_head = target_q_head
        
        # 存储目标编码器和注意力机制（如果提供）
        self.target_encoder = target_encoder if target_encoder is not None else shared_encoder
        self.target_attention = target_attention if target_attention is not None else shared_attention
        
        # 保存编码器输出维度，用于后续处理
        self.embed_dim = shared_encoder.output_dim
    
    def forward(self, all_s, all_a, agent_index):
        """前向传播，计算当前Q值
        
        Args:
            all_s: 所有智能体的状态 [batch_size, n_agents * state_dim]
            all_a: 所有智能体的动作 [batch_size, n_agents * action_dim]
            agent_index: 需要计算Q值的智能体索引
            
        Returns:
            q1, q2: 两个Q值 [batch_size, 1]
        """
        batch_size = all_s.shape[0]
        
        # 计算智能体数量
        state_dim = self.shared_encoder.state_dim
        action_dim = self.shared_encoder.action_dim
        n_agents = min(all_s.shape[1] // state_dim, all_a.shape[1] // action_dim)
        
        # 编码所有智能体的状态-动作对
        embeddings = []
        for j in range(n_agents):
            # 提取第j个智能体的状态和动作
            s_j = all_s[:, j*state_dim : (j+1)*state_dim]
            a_j = all_a[:, j*action_dim : (j+1)*action_dim]
            
            # 拼接状态和动作
            sa_j = torch.cat([s_j, a_j], dim=1)
            
            # 编码
            embed_j = self.shared_encoder(sa_j)  # [batch_size, embed_dim]
            embeddings.append(embed_j)
        
        # 将所有嵌入堆叠成3D张量
        embeddings_tensor = torch.stack(embeddings, dim=1)  # [batch_size, n_agents, embed_dim]
        
        # 应用注意力机制
        attended_embeddings = self.shared_attention(
            embeddings_tensor, embeddings_tensor, embeddings_tensor
        )  # [batch_size, n_agents, embed_dim]
        
        # 获取指定智能体的嵌入和注意力输出
        agent_embedding = embeddings_tensor[:, agent_index, :]  # [batch_size, embed_dim]
        agent_attended = attended_embeddings[:, agent_index, :]  # [batch_size, embed_dim]
        
        # 拼接嵌入和注意力输出
        combined_features = torch.cat([agent_embedding, agent_attended], dim=1)  # [batch_size, embed_dim*2]
        
        # 计算Q值
        q1, q2 = self.q_head(combined_features)
        
        return q1, q2
    
    def forward_target(self, all_s, all_a, agent_index):
        """使用目标网络计算Q值
        
        Args:
            all_s: 所有智能体的状态 [batch_size, n_agents * state_dim]
            all_a: 所有智能体的动作 [batch_size, n_agents * action_dim]
            agent_index: 需要计算Q值的智能体索引
            
        Returns:
            q1, q2: 两个Q值 [batch_size, 1]
        """
        batch_size = all_s.shape[0]
        
        # 计算智能体数量
        state_dim = self.shared_encoder.state_dim
        action_dim = self.shared_encoder.action_dim
        n_agents = min(all_s.shape[1] // state_dim, all_a.shape[1] // action_dim)
        
        # 使用目标编码器和注意力机制
        with torch.no_grad():
            embeddings = []
            for j in range(n_agents):
                # 提取第j个智能体的状态和动作
                s_j = all_s[:, j*state_dim : (j+1)*state_dim]
                a_j = all_a[:, j*action_dim : (j+1)*action_dim]
                
                # 拼接状态和动作
                sa_j = torch.cat([s_j, a_j], dim=1)
                
                # 使用目标编码器进行编码
                embed_j = self.target_encoder(sa_j)  # [batch_size, embed_dim]
                embeddings.append(embed_j)
            
            # 将所有嵌入堆叠成3D张量
            embeddings_tensor = torch.stack(embeddings, dim=1)  # [batch_size, n_agents, embed_dim]
            
            # 应用目标注意力机制
            attended_embeddings = self.target_attention(
                embeddings_tensor, embeddings_tensor, embeddings_tensor
            )  # [batch_size, n_agents, embed_dim]
            
            # 获取指定智能体的嵌入和注意力输出
            agent_embedding = embeddings_tensor[:, agent_index, :]  # [batch_size, embed_dim]
            agent_attended = attended_embeddings[:, agent_index, :]  # [batch_size, embed_dim]
            
            # 拼接嵌入和注意力输出
            combined_features = torch.cat([agent_embedding, agent_attended], dim=1)  # [batch_size, embed_dim*2]
            
            # 使用目标Q头计算Q值
            q1, q2 = self.target_q_head(combined_features)
        
        return q1, q2 