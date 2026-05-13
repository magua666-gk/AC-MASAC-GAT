import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from main_SAC import max_action, min_action
from integrated_ablation_modules.masac_no_curriculum.masac_adapter import GraphAttentionLayer, build_spatial_adjacency, build_formation_adjacency


LOG_STD_MIN = -20
LOG_STD_MAX = 2

class ActorObsEncoder(nn.Module):
    """Actor 观测编码器
    
    将原始观测编码为隐藏表示
    """
    def __init__(self, state_dim, embed_dim, hidden_dims=[256, 128]):
        """初始化观测编码器
        
        Args:
            state_dim: 状态维度
            embed_dim: 输出嵌入维度
            hidden_dims: 隐藏层维度列表
        """
        super(ActorObsEncoder, self).__init__()
        
        layers = []
        input_dim = state_dim
        
        # 构建 MLP 层
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        # 最终输出层
        layers.append(nn.Linear(input_dim, embed_dim))
        
        self.encoder = nn.Sequential(*layers)
        
        # 使用 Xavier 初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, obs):
        """前向传播
        
        Args:
            obs: 观测张量 [batch_size, state_dim]
            
        Returns:
            embedding: 编码后的嵌入 [batch_size, embed_dim]
        """
        return self.encoder(obs)

class LeaderActorNet(nn.Module):
    """Leader Actor 网络
    
    处理 Leader 的观测，输出动作分布参数
    """
    def __init__(self, state_dim, action_dim, embed_dim=128, hidden_dims=[256, 128]):
        """初始化 Leader Actor 网络
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            embed_dim: 嵌入维度
            hidden_dims: 隐藏层维度列表
        """
        super(LeaderActorNet, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.embed_dim = embed_dim
        
        # 观测编码器
        self.encoder = ActorObsEncoder(state_dim, embed_dim)
        
        # 特征处理 MLP
        layers = []
        input_dim = embed_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        self.mlp = nn.Sequential(*layers)
        
        # 均值和对数标准差输出层
        self.mean_layer = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std_layer = nn.Linear(hidden_dims[-1], action_dim)
        
        # 使用较小的初始化值，以控制初始输出范围
        nn.init.uniform_(self.mean_layer.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.log_std_layer.weight, -3e-3, 3e-3)
        nn.init.zeros_(self.mean_layer.bias)
        # 初始化log_std偏置为较大的负值，确保初始标准差较小但不会过小
        nn.init.constant_(self.log_std_layer.bias, -2.0)
    
    def forward(self, obs_leader):
        """前向传播
        
        Args:
            obs_leader: Leader 的观测 [batch_size, state_dim]
            
        Returns:
            mean: 动作均值 [batch_size, action_dim]
            std: 动作标准差 [batch_size, action_dim]
        """
        # 编码观测
        embed = self.encoder(obs_leader)
        
        # 特征处理
        features = self.mlp(embed)
        
        # 计算均值（使用 tanh 限制范围）
        mean = torch.tanh(self.mean_layer(features)) * max_action
        
        # 计算对数标准差，并限制在合理范围内
        log_std = self.log_std_layer(features)
        log_std = torch.clamp(log_std, -20, 2)  # 防止标准差过大或过小
        
        # 计算标准差
        std = torch.exp(log_std)
        
        return mean, std
    
    def evaluate(self, obs_leader):
        """评估动作和对数概率
        
        Args:
            obs_leader: Leader 的观测 [batch_size, state_dim]
            
        Returns:
            action: 采样的动作 [batch_size, action_dim]
            log_prob: 对数概率 [batch_size, 1]
        """
        # 获取动作分布参数
        mean, std = self.forward(obs_leader)
        
        # 创建正态分布
        dist = torch.distributions.Normal(mean, std)
        
        # 使用重参数化技巧采样动作
        z = dist.rsample()
        
        # 应用 tanh 变换
        action = torch.tanh(z)
        
        # 限制在更安全的范围内，避免数值不稳定
        action = torch.clamp(action, -0.99, 0.99)
        
        # 缩放到动作范围
        scaled_action = action * max_action
        
        # 计算 tanh 校正的对数概率
        log_prob = dist.log_prob(z)
        
        # 应用 tanh 校正项
        # log prob = log prob - log(1 - tanh(z)^2) 
        # = log prob - log(1 - action^2)
        # 使用更稳健的数值稳定性计算
        log_prob -= torch.log(torch.clamp(1 - action.pow(2), min=1e-4, max=1.0))
        
        # 对动作维度求和
        log_prob = log_prob.sum(dim=1, keepdim=True)
        
        return scaled_action, log_prob
    
    def choose_action(self, obs_leader, evaluate=False):
        """选择动作
        
        Args:
            obs_leader: Leader 的观测 (numpy 数组)
            evaluate: 是否为评估模式（使用确定性策略）
            
        Returns:
            action: 动作 (numpy 数组)
        """
        # 确保输入是正确形状的张量
        if isinstance(obs_leader, np.ndarray):
            obs_leader = torch.FloatTensor(obs_leader)
        
        if obs_leader.dim() == 1:
            obs_leader = obs_leader.unsqueeze(0)
        
        # 移动到正确的设备
        device = next(self.parameters()).device
        obs_leader = obs_leader.to(device)
        
        with torch.no_grad():
            mean, std = self.forward(obs_leader)
            
            if evaluate:
                # 评估模式：使用确定性策略（均值）
                action = mean
            else:
                # 训练模式：使用随机采样
                dist = torch.distributions.Normal(mean, std)
                action = dist.sample()
            
            # 应用tanh变换并限制范围
            action = torch.tanh(action)
            action = torch.clamp(action, -0.99, 0.99)
            action = action * max_action
        
        # 返回 numpy 数组
        return action.cpu().numpy().squeeze()

class FollowerActorNet(nn.Module):
    """Follower Actor 网络
    
    处理 Follower 的观测，输出动作分布参数
    与 LeaderActorNet 结构相同，用于 Follower 智能体
    参数将在所有 Follower 间共享
    """
    def __init__(self, state_dim, action_dim, embed_dim=128, hidden_dims=[256, 128]):
        """初始化 Follower Actor 网络
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            embed_dim: 嵌入维度
            hidden_dims: 隐藏层维度列表
        """
        super(FollowerActorNet, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.embed_dim = embed_dim
        
        # 观测编码器
        self.encoder = ActorObsEncoder(state_dim, embed_dim)
        
        # 特征处理 MLP
        layers = []
        input_dim = embed_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        self.mlp = nn.Sequential(*layers)
        
        # 均值和对数标准差输出层
        self.mean_layer = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std_layer = nn.Linear(hidden_dims[-1], action_dim)
        
        # 使用较小的初始化值，以控制初始输出范围
        nn.init.uniform_(self.mean_layer.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.log_std_layer.weight, -3e-3, 3e-3)
        nn.init.zeros_(self.mean_layer.bias)
        # 初始化log_std偏置为较大的负值，确保初始标准差较小但不会过小
        nn.init.constant_(self.log_std_layer.bias, -2.0)
    
    def forward(self, obs_follower):
        """前向传播
        
        Args:
            obs_follower: Follower 的观测 [batch_size, state_dim] 或 [batch_size, max_F, state_dim]
            
        Returns:
            mean: 动作均值 [batch_size, action_dim] 或 [batch_size, max_F, action_dim]
            std: 动作标准差 [batch_size, action_dim] 或 [batch_size, max_F, action_dim]
        """
        is_batched_obs = obs_follower.dim() == 3  # [B, max_F, D]
        
        if is_batched_obs:
            B, max_F, D = obs_follower.shape
            
            if max_F == 0:
                # 获取动作维度
                action_dim = self.action_dim
                
                # 创建期望形状的零元素/单位标准差张量
                # 使用 obs_follower 的 device 和 dtype 保持一致性
                mean_output = torch.zeros((B, 0, action_dim), device=obs_follower.device, dtype=obs_follower.dtype)
                
                # 为标准差使用 ones，以避免潜在的 log(0) 或除以0的问题
                std_output = torch.ones((B, 0, action_dim), device=obs_follower.device, dtype=obs_follower.dtype)
                
                return mean_output, std_output
            
            # 展平批次和 max_F 维度
            obs_flat = obs_follower.reshape(-1, D)  # [B*max_F, D]
            
            # 编码观测
            embed = self.encoder(obs_flat)  # [B*max_F, E]
            
            # 特征处理
            features = self.mlp(embed)  # [B*max_F, H]
            
            # 计算均值
            mean = torch.tanh(self.mean_layer(features)) * max_action  # [B*max_F, A]
            
            # 计算对数标准差
            log_std = self.log_std_layer(features)  # [B*max_F, A]
            log_std = torch.clamp(log_std, -20, 2)
            
            # 计算标准差
            std = torch.exp(log_std)  # [B*max_F, A]
            
            # 重塑回原始维度
            mean = mean.reshape(B, max_F, -1)  # [B, max_F, A]
            std = std.reshape(B, max_F, -1)  # [B, max_F, A]
        else:
            # 标准处理，与 LeaderActorNet 相同
            embed = self.encoder(obs_follower)
            features = self.mlp(embed)
            mean = torch.tanh(self.mean_layer(features)) * max_action
            log_std = self.log_std_layer(features)
            log_std = torch.clamp(log_std, -20, 2)
            std = torch.exp(log_std)
        
        return mean, std
    
    def evaluate(self, obs_follower):
        """评估动作和对数概率
        
        Args:
            obs_follower: Follower 的观测 [batch_size, state_dim] 或 [batch_size, max_F, state_dim]
            
        Returns:
            action: 采样的动作 [batch_size, action_dim] 或 [batch_size, max_F, action_dim]
            log_prob: 对数概率 [batch_size, 1] 或 [batch_size, max_F, 1]
        """
        # 检查输入维度
        is_batched_obs = obs_follower.dim() == 3  # [B, max_F, D]
        
        # 获取动作分布参数
        mean, std = self.forward(obs_follower)
        
        if is_batched_obs:
            B, max_F, A = mean.shape
            
            # 展平批次和 max_F 维度，方便处理
            mean_flat = mean.reshape(-1, A)  # [B*max_F, A]
            std_flat = std.reshape(-1, A)  # [B*max_F, A]
            
            # 创建正态分布
            dist = torch.distributions.Normal(mean_flat, std_flat)
            
            # 使用重参数化技巧采样动作
            z = dist.rsample()  # [B*max_F, A]
            
            # 应用 tanh 变换
            action = torch.tanh(z)  # [B*max_F, A]
            
            # 缩放到动作范围
            scaled_action = action * max_action  # [B*max_F, A]
            
            # 计算 tanh 校正的对数概率
            log_prob = dist.log_prob(z)  # [B*max_F, A]
            
            # 应用 tanh 校正项
            # 使用更稳健的数值稳定性计算
            log_prob -= torch.log(torch.clamp(1 - action.pow(2), min=1e-4, max=1.0))  # [B*max_F, A]
            
            # 对动作维度求和
            log_prob = log_prob.sum(dim=1, keepdim=True)  # [B*max_F, 1]
            
            # 重塑回原始维度
            scaled_action = scaled_action.reshape(B, max_F, A)  # [B, max_F, A]
            log_prob = log_prob.reshape(B, max_F, 1)  # [B, max_F, 1]
        else:
            # 创建正态分布
            dist = torch.distributions.Normal(mean, std)
            
            # 使用重参数化技巧采样动作
            z = dist.rsample()
            
            # 应用 tanh 变换
            action = torch.tanh(z)
            
            # 限制在更安全的范围内，避免数值不稳定
            action = torch.clamp(action, -0.99, 0.99)
            
            # 缩放到动作范围
            scaled_action = action * max_action
            
            # 计算 tanh 校正的对数概率
            log_prob = dist.log_prob(z)
            
            # 应用 tanh 校正项
            # 使用更稳健的数值稳定性计算
            log_prob -= torch.log(torch.clamp(1 - action.pow(2), min=1e-4, max=1.0))
            
            # 对动作维度求和
            log_prob = log_prob.sum(dim=1, keepdim=True)
        
        return scaled_action, log_prob
    
    def choose_action(self, obs_follower, evaluate=False):
        """选择动作
        
        Args:
            obs_follower: Follower 的观测 (numpy 数组)
            evaluate: 是否为评估模式（使用确定性策略）
            
        Returns:
            action: 动作 (numpy 数组)
        """
        # 确保输入是正确形状的张量
        if isinstance(obs_follower, np.ndarray):
            obs_follower = torch.FloatTensor(obs_follower)
        
        if obs_follower.dim() == 1:
            obs_follower = obs_follower.unsqueeze(0)
        
        # 移动到正确的设备
        device = next(self.parameters()).device
        obs_follower = obs_follower.to(device)
        
        with torch.no_grad():
            mean, std = self.forward(obs_follower)
            
            if evaluate:
                # 评估模式：使用确定性策略（均值）
                action = mean
            else:
                # 训练模式：使用随机采样
                dist = torch.distributions.Normal(mean, std)
                action = dist.sample()
            
            # 应用tanh变换并限制范围
            action = torch.tanh(action)
            action = torch.clamp(action, -0.99, 0.99)
            action = action * max_action
        
        # 返回 numpy 数组
        return action.cpu().numpy().squeeze()

class AttentionLeaderActorNet(nn.Module):
    """基于注意力机制的Leader Actor网络
    
    处理Leader的观测，并利用注意力机制获取从Followers中的上下文信息，
    输出动作分布参数。
    """
    def __init__(self, state_dim, action_dim, embed_dim=128, hidden_dims=[256, 128], 
                 n_heads=4, dropout=0.05, use_shared_layer=True):
        """初始化基于注意力的Leader Actor网络
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            embed_dim: 嵌入维度
            hidden_dims: 隐藏层维度列表
            n_heads: 注意力头数
            dropout: Dropout概率
            use_shared_layer: 是否使用共享层处理融合特征
        """
        super(AttentionLeaderActorNet, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.embed_dim = embed_dim
        self.use_shared_layer = use_shared_layer
        
        # Leader观测编码器
        self.self_obs_encoder = ActorObsEncoder(state_dim, embed_dim, hidden_dims)
        
        # Follower观测编码器
        self.follower_view_obs_encoder = ActorObsEncoder(state_dim, embed_dim, hidden_dims)
        
        # 注意力层
        self.attention_on_followers = GraphAttentionLayer(embed_dim, n_heads, dropout)
        self.gat_context_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid()
        )
        
        # 特征融合后的共享层
        if use_shared_layer:
            self.fc_shared_after_attention = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.ReLU()
            )
            final_dim = embed_dim
        else:
            final_dim = embed_dim * 2
        
        # 均值和对数标准差输出层
        self.mu_head = nn.Linear(final_dim, action_dim)
        self.log_std_head = nn.Linear(final_dim, action_dim)
        
        # 使用较小的初始化值
        nn.init.uniform_(self.mu_head.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.log_std_head.weight, -3e-3, 3e-3)
        nn.init.zeros_(self.mu_head.bias)
        nn.init.zeros_(self.log_std_head.bias)
    
    def forward(self, obs_leader, obs_followers):
        """前向传播
        
        Args:
            obs_leader: Leader的观测 [batch_size, state_dim]
            obs_followers: Followers的观测 [batch_size, max_F, state_dim]
            
        Returns:
            mean: 动作均值 [batch_size, action_dim]
            std: 动作标准差 [batch_size, action_dim]
        """
        # 编码Leader观测
        leader_self_embedding = self.self_obs_encoder(obs_leader)  # [B, E_dim]
        
        # 获取批次大小和维度信息
        B, max_F, S_dim = obs_followers.shape
        E_dim = leader_self_embedding.shape[-1]
        
        # 如果没有Followers，直接使用Leader的自身嵌入
        if max_F == 0:
            if self.use_shared_layer:
                # 使用零向量作为上下文
                context_from_followers = torch.zeros_like(leader_self_embedding)
                fused_feature = torch.cat([leader_self_embedding, context_from_followers], dim=-1)
                fused_feature = self.fc_shared_after_attention(fused_feature)
            else:
                # 直接重复Leader嵌入以满足维度要求
                fused_feature = torch.cat([leader_self_embedding, leader_self_embedding], dim=-1)
                
            mu = self.mu_head(fused_feature)
            log_std = self.log_std_head(fused_feature)
            log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
            std = torch.exp(log_std)
            
            return mu, std
        
        # 编码Followers观测
        follower_view_embeddings = self.follower_view_obs_encoder(
            obs_followers.reshape(B * max_F, S_dim)
        ).reshape(B, max_F, E_dim)  # [B, max_F, E_dim]
        
        # === 图注意力：Leader + Followers 作为一个动态图 ===
        leader_valid_mask = torch.ones(B, 1, dtype=torch.bool, device=obs_leader.device)
        follower_valid_mask = torch.ones(B, max_F, dtype=torch.bool, device=obs_leader.device)
        node_mask = torch.cat([leader_valid_mask, follower_valid_mask], dim=1)
        node_embeddings = torch.cat([leader_self_embedding.unsqueeze(1), follower_view_embeddings], dim=1)
        node_obs = torch.cat([obs_leader.unsqueeze(1), obs_followers], dim=1)
        adjacency = build_formation_adjacency(
            node_obs,
            node_mask=node_mask,
            leader_index=0,
            distance_threshold=0.65,
            k_neighbors=2,
            force_leader_edges=True
        )
        graph_embeddings = self.attention_on_followers(
            node_embeddings,
            adjacency=adjacency,
            node_mask=node_mask
        )  # [B, 1+max_F, E_dim]
        raw_context_from_followers = graph_embeddings[:, 0, :]  # Leader节点聚合后的图上下文
        gate = self.gat_context_gate(torch.cat([leader_self_embedding, raw_context_from_followers], dim=-1))
        context_from_followers = gate * raw_context_from_followers + (1.0 - gate) * leader_self_embedding
        
        # 特征融合
        fused_feature = torch.cat([leader_self_embedding, context_from_followers], dim=-1)  # [B, E_dim*2]
        
        # 可选的共享层处理
        if self.use_shared_layer:
            fused_feature = self.fc_shared_after_attention(fused_feature)  # [B, E_dim]
        
        # 计算动作分布参数
        mu = self.mu_head(fused_feature)
        log_std = self.log_std_head(fused_feature)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        
        # 生成均值和标准差
        return mu, std
    
    def evaluate(self, obs_leader, obs_followers):
        """评估动作和对数概率
        
        Args:
            obs_leader: Leader的观测 [batch_size, state_dim]
            obs_followers: Followers的观测 [batch_size, max_F, state_dim]
            
        Returns:
            action: 采样的动作 [batch_size, action_dim]
            log_prob: 对数概率 [batch_size, 1]
        """
        # 获取动作分布参数
        mean, std = self.forward(obs_leader, obs_followers)
        
        # 创建正态分布
        dist = torch.distributions.Normal(mean, std)
        
        # 使用重参数化技巧采样动作
        z = dist.rsample()
        
        # 应用tanh变换
        action = torch.tanh(z)
        
        # 限制在更安全的范围内，避免数值不稳定
        action = torch.clamp(action, -0.99, 0.99)
        
        # 缩放到动作范围
        scaled_action = action * max_action
        
        # 计算tanh校正的对数概率
        log_prob = dist.log_prob(z)
        
        # 应用tanh校正项
        # 使用更稳健的数值稳定性计算
        log_prob -= torch.log(torch.clamp(1 - action.pow(2), min=1e-4, max=1.0))
        
        # 对动作维度求和
        log_prob = log_prob.sum(dim=1, keepdim=True)
        
        return scaled_action, log_prob
    
    def choose_action(self, obs_leader, obs_followers, evaluate=False):
        """选择动作
        
        Args:
            obs_leader: Leader的观测 (numpy数组或张量)
            obs_followers: Followers的观测 (numpy数组或张量)
            evaluate: 是否为评估模式（使用确定性策略）
            
        Returns:
            action: 动作 (numpy数组)
        """
        # 确保输入是正确形状的张量
        if isinstance(obs_leader, np.ndarray):
            obs_leader = torch.FloatTensor(obs_leader)
        
        if isinstance(obs_followers, np.ndarray):
            obs_followers = torch.FloatTensor(obs_followers)
        
        # 确保单样本输入形状正确
        if obs_leader.dim() == 1:
            obs_leader = obs_leader.unsqueeze(0)
            
        if obs_followers.dim() == 2:
            obs_followers = obs_followers.unsqueeze(0)
        
        # 移动到正确的设备
        device = next(self.parameters()).device
        obs_leader = obs_leader.to(device)
        obs_followers = obs_followers.to(device)
        
        with torch.no_grad():
            mean, std = self.forward(obs_leader, obs_followers)
            
            if evaluate:
                # 评估模式：使用确定性策略（均值）
                action = mean
            else:
                # 训练模式：使用随机采样
                dist = torch.distributions.Normal(mean, std)
                action = dist.sample()
        
        # 返回numpy数组
        return action.cpu().numpy().squeeze()

class AttentionFollowerActorNet(nn.Module):
    """基于注意力机制的Follower Actor网络
    
    处理Follower的观测，并利用注意力机制获取来自Leader和其他Followers的上下文信息，
    输出动作分布参数。
    """
    def __init__(self, state_dim, action_dim, embed_dim=128, hidden_dims=[256, 128], 
                 n_heads=4, dropout=0.05, use_shared_layer=True):
        """初始化基于注意力的Follower Actor网络
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            embed_dim: 嵌入维度
            hidden_dims: 隐藏层维度列表
            n_heads: 注意力头数
            dropout: Dropout概率
            use_shared_layer: 是否使用共享层处理融合特征
        """
        super(AttentionFollowerActorNet, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.embed_dim = embed_dim
        self.use_shared_layer = use_shared_layer
        
        # Follower自身观测编码器
        self.self_obs_encoder = ActorObsEncoder(state_dim, embed_dim, hidden_dims)
        
        # Leader上下文观测编码器
        self.context_leader_obs_encoder = ActorObsEncoder(state_dim, embed_dim, hidden_dims)
        
        # 其他Followers上下文观测编码器
        self.context_other_followers_obs_encoder = ActorObsEncoder(state_dim, embed_dim, hidden_dims)
        
        # 全局上下文注意力层
        self.attention_on_global_context = GraphAttentionLayer(embed_dim, n_heads, dropout)
        self.gat_context_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid()
        )
        
        # 特征融合后的共享层
        if use_shared_layer:
            self.fc_shared_after_attention = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.ReLU()
            )
            final_dim = embed_dim
        else:
            final_dim = embed_dim * 2
        
        # 均值和对数标准差输出层
        self.mu_head = nn.Linear(final_dim, action_dim)
        self.log_std_head = nn.Linear(final_dim, action_dim)
        
        # 使用较小的初始化值
        nn.init.uniform_(self.mu_head.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.log_std_head.weight, -3e-3, 3e-3)
        nn.init.zeros_(self.mu_head.bias)
        nn.init.zeros_(self.log_std_head.bias)
    
    def forward(self, obs_follower_self, obs_leader_for_context, obs_other_followers_for_context):
        """前向传播
        
        Args:
            obs_follower_self: 当前Follower的观测 [batch_size, state_dim]
            obs_leader_for_context: Leader的观测 [batch_size, state_dim]
            obs_other_followers_for_context: 其他Followers的观测 [batch_size, num_other_F, state_dim]
            
        Returns:
            mean: 动作均值 [batch_size, action_dim]
            std: 动作标准差 [batch_size, action_dim]
        """
        # 编码Follower自身观测
        follower_self_embedding = self.self_obs_encoder(obs_follower_self)  # [B, E_dim]
        
        # 编码Leader上下文观测
        context_leader_embedding = self.context_leader_obs_encoder(obs_leader_for_context)  # [B, E_dim]
        
        # 获取批次大小和维度信息
        B, E_dim = follower_self_embedding.shape
        
        # 检查是否存在其他Followers
        has_other_followers = obs_other_followers_for_context.size(1) > 0
        
        # 编码其他Followers观测
        if has_other_followers:
            B, num_other_F, S_dim = obs_other_followers_for_context.shape
            context_other_followers_embeddings = self.context_other_followers_obs_encoder(
                obs_other_followers_for_context.reshape(B * num_other_F, S_dim)
            ).reshape(B, num_other_F, E_dim)  # [B, num_other_F, E_dim]
        else:
            # 创建空张量
            context_other_followers_embeddings = torch.empty((B, 0, E_dim), 
                                                           device=follower_self_embedding.device, 
                                                           dtype=follower_self_embedding.dtype)
            num_other_F = 0
        
        # 组合Leader和其他Followers的上下文嵌入
        global_context_embeddings = torch.cat(
            [context_leader_embedding.unsqueeze(1), context_other_followers_embeddings], 
            dim=1
        )  # [B, 1+num_other_F, E_dim]
        
        # === 图注意力：当前Follower + Leader + 其他Followers 作为局部动态图 ===
        self_valid_mask = torch.ones(B, 1, dtype=torch.bool, device=obs_follower_self.device)
        context_valid_mask = torch.ones(B, 1 + num_other_F, dtype=torch.bool, device=obs_follower_self.device)
        node_mask = torch.cat([self_valid_mask, context_valid_mask], dim=1)
        node_embeddings = torch.cat([follower_self_embedding.unsqueeze(1), global_context_embeddings], dim=1)
        node_obs = torch.cat(
            [obs_follower_self.unsqueeze(1), obs_leader_for_context.unsqueeze(1), obs_other_followers_for_context],
            dim=1
        )
        adjacency = build_formation_adjacency(
            node_obs,
            node_mask=node_mask,
            leader_index=1,
            distance_threshold=0.65,
            k_neighbors=2,
            force_leader_edges=True
        )
        graph_embeddings = self.attention_on_global_context(
            node_embeddings,
            adjacency=adjacency,
            node_mask=node_mask
        )  # [B, 1+1+num_other_F, E_dim]
        raw_context_from_global = graph_embeddings[:, 0, :]  # 当前Follower节点聚合后的图上下文
        gate = self.gat_context_gate(torch.cat([follower_self_embedding, raw_context_from_global], dim=-1))
        context_from_global = gate * raw_context_from_global + (1.0 - gate) * follower_self_embedding
        
        # 特征融合
        fused_feature = torch.cat([follower_self_embedding, context_from_global], dim=-1)  # [B, E_dim*2]
        
        # 可选的共享层处理
        if self.use_shared_layer:
            fused_feature = self.fc_shared_after_attention(fused_feature)  # [B, E_dim]
        
        # 计算动作分布参数
        mu = self.mu_head(fused_feature)
        log_std = self.log_std_head(fused_feature)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        
        # 生成均值和标准差
        return mu, std
    
    def evaluate(self, obs_follower_self, obs_leader_for_context, obs_other_followers_for_context):
        """评估动作和对数概率
        
        Args:
            obs_follower_self: 当前Follower的观测 [batch_size, state_dim]
            obs_leader_for_context: Leader的观测 [batch_size, state_dim]
            obs_other_followers_for_context: 其他Followers的观测 [batch_size, num_other_F, state_dim]
            
        Returns:
            action: 采样的动作 [batch_size, action_dim]
            log_prob: 对数概率 [batch_size, 1]
        """
        # 获取动作分布参数
        mean, std = self.forward(
            obs_follower_self, 
            obs_leader_for_context, 
            obs_other_followers_for_context
        )
        
        # 创建正态分布
        dist = torch.distributions.Normal(mean, std)
        
        # 使用重参数化技巧采样动作
        z = dist.rsample()
        
        # 应用tanh变换
        action = torch.tanh(z)
        
        # 限制在更安全的范围内，避免数值不稳定
        action = torch.clamp(action, -0.99, 0.99)
        
        # 缩放到动作范围
        scaled_action = action * max_action
        
        # 计算tanh校正的对数概率
        log_prob = dist.log_prob(z)
        
        # 应用tanh校正项
        # 使用更稳健的数值稳定性计算
        log_prob -= torch.log(torch.clamp(1 - action.pow(2), min=1e-4, max=1.0))
        
        # 对动作维度求和
        log_prob = log_prob.sum(dim=1, keepdim=True)
        
        return scaled_action, log_prob
    
    def choose_action(self, obs_follower_self, obs_leader_for_context, obs_other_followers_for_context, evaluate=False):
        """选择动作
        
        Args:
            obs_follower_self: 当前Follower的观测 (numpy数组或张量)
            obs_leader_for_context: Leader的观测 (numpy数组或张量)
            obs_other_followers_for_context: 其他Followers的观测 (numpy数组或张量)
            evaluate: 是否为评估模式（使用确定性策略）
            
        Returns:
            action: 动作 (numpy数组)
        """
        # 确保输入是正确形状的张量
        if isinstance(obs_follower_self, np.ndarray):
            obs_follower_self = torch.FloatTensor(obs_follower_self)
        
        if isinstance(obs_leader_for_context, np.ndarray):
            obs_leader_for_context = torch.FloatTensor(obs_leader_for_context)
            
        if isinstance(obs_other_followers_for_context, np.ndarray):
            obs_other_followers_for_context = torch.FloatTensor(obs_other_followers_for_context)
        
        # 确保单样本输入形状正确
        if obs_follower_self.dim() == 1:
            obs_follower_self = obs_follower_self.unsqueeze(0)
            
        if obs_leader_for_context.dim() == 1:
            obs_leader_for_context = obs_leader_for_context.unsqueeze(0)
            
        if obs_other_followers_for_context.dim() == 2:
            obs_other_followers_for_context = obs_other_followers_for_context.unsqueeze(0)
        
        # 移动到正确的设备
        device = next(self.parameters()).device
        obs_follower_self = obs_follower_self.to(device)
        obs_leader_for_context = obs_leader_for_context.to(device)
        obs_other_followers_for_context = obs_other_followers_for_context.to(device)
        
        with torch.no_grad():
            mean, std = self.forward(
                obs_follower_self, 
                obs_leader_for_context, 
                obs_other_followers_for_context
            )
            
            if evaluate:
                # 评估模式：使用确定性策略（均值）
                action = mean
            else:
                # 训练模式：使用随机采样
                dist = torch.distributions.Normal(mean, std)
                action = dist.sample()
        
        # 返回numpy数组
        return action.cpu().numpy().squeeze() 