import numpy as np
import torch
import torch.nn as nn
import os
import traceback
import copy
import time
from typing import List, Dict, Any, Tuple, Optional

# 日志级别
LOG_DEBUG = 0
LOG_INFO = 1
LOG_WARNING = 2
LOG_ERROR = 3

# 全局日志级别设置
CURRENT_LOG_LEVEL = LOG_INFO

# 默认gamma值
GAMMA = 0.99

# 默认动作范围
min_action = -5.0
max_action = 5.0

# 最大智能体数量上限
MAX_N_AGENTS = 3

# 智能体类型定义
LEADER_TYPE_ID = 0  # 主机
FOLLOWER_TYPE_ID = 1  # 从机
NUM_AGENT_TYPES = 2 

# 嵌入维度定义
TYPE_EMBEDDING_DIM = 16  # 类型嵌入维度
AGENT_EMBEDDING_DIM = 128  # 智能体嵌入维度

# 日志历史
_log_history = {}
_log_throttle_times = {}

# 用于抑制重复消息的字典
_last_log_messages = {}
_log_counts = {}
_last_throttled_time = {}

def set_log_level(level):
    """设置日志级别
    
    Args:
        level: 日志级别，如LOG_ERROR, LOG_WARNING, LOG_INFO, LOG_DEBUG
    """
    global CURRENT_LOG_LEVEL
    CURRENT_LOG_LEVEL = level
    
def log(message, level=LOG_INFO, throttle=0, suppress_repeat=True):
    """按日志级别打印消息，支持频率控制和消息抑制
    
    Args:
        message: 要打印的消息
        level: 消息的日志级别
        throttle: 抑制频率(秒)，0表示不限制
        suppress_repeat: 是否抑制重复消息
    """
    import time
    current_time = time.time()
    
    # 修改：过滤掉比全局设置更详细的日志级别
    # （即，只有当消息级别 level >= CURRENT_LOG_LEVEL 时才继续）
    if level < CURRENT_LOG_LEVEL:
        return
        
    # 消息抑制逻辑 - 避免打印重复消息
    if suppress_repeat:
        # 进一步简化键生成方式，只使用消息的内容类型作为键
        # 对于特定模式的消息，提取关键特征
        if "奖励维度" in message or "维度处理" in message:
            msg_key = "rewards_dim_warning"
        elif "形状" in message and "调整" in message:
            msg_key = "shape_adjustment"
        else:
            # 对于其他消息，使用前30个字符作为键
            msg_key = message[:30]
        
        # 全局键，便于跨批次抑制相似消息
        key = f"{msg_key}"
        
        # 检查是否是重复消息
        if key in _last_log_messages:
            # 更新计数
            _log_counts[key] = _log_counts.get(key, 1) + 1
            
            # 降低重复消息的输出频率，只在100的倍数时才输出
            # 修改：确保重复消息也检查日志级别（即使它们已通过了第一次级别检查）
            if _log_counts[key] % 100 == 0 and level <= CURRENT_LOG_LEVEL:
                # 使用与级别相同的前缀
                prefix = {
                    LOG_ERROR: "[错误] ",
                    LOG_WARNING: "[警告] ",
                    LOG_INFO: "[信息] ",
                    LOG_DEBUG: "[调试] "
                }.get(level, "")
                print(f"{prefix}{message} (重复 {_log_counts[key]} 次)")
            return
        else:
            # 新消息，重置计数
            _last_log_messages[key] = current_time
            _log_counts[key] = 1
    
    # 频率控制逻辑
    if throttle > 0:
        # 简化键生成，使用消息内容的前50个字符
        throttle_key = message[:50]
        
        # 检查上次输出时间
        if throttle_key in _last_throttled_time:
            last_time = _last_throttled_time[throttle_key]
            # 如果未到输出时间，跳过
            if current_time - last_time < throttle:
                return
        
        # 更新最后输出时间
        _last_throttled_time[throttle_key] = current_time
    
    # 根据日志级别添加前缀
    prefix = {
        LOG_ERROR: "[错误] ",
        LOG_WARNING: "[警告] ",
        LOG_INFO: "[信息] ",
        LOG_DEBUG: "[调试] "
    }.get(level, "")
    
    # 打印消息
    print(f"{prefix}{message}")

def clear_log_history():
    """清除日志历史记录"""
    global _last_log_messages, _log_counts, _last_throttled_time
    _last_log_messages = {}
    _log_counts = {}
    _last_throttled_time = {}

# 导入现有的SAC类
from main_SAC import Actor, Critic, Entroy, Memory, Ornstein_Uhlenbeck_Noise as OUNoise
from main_SAC import state_number, action_number, max_action, min_action
from main_SAC import GAMMA, tau, CriticNet

# 导入新创建的类
from .agent_pool import AgentPool, DynamicActor
from .smer_memory import SMERMemory

# 智能体类型嵌入模块，用于区分主机和从机
class AgentTypeEmbedding(nn.Module):
    """智能体类型嵌入模块
    
    为主机(LEADER_TYPE_ID=0)和从机(FOLLOWER_TYPE_ID=1)提供可学习的嵌入表示
    """
    def __init__(self, num_types=NUM_AGENT_TYPES, embedding_dim=TYPE_EMBEDDING_DIM):
        """初始化智能体类型嵌入模块
        
        Args:
            num_types: 智能体类型数量
            embedding_dim: 嵌入维度
        """
        super(AgentTypeEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_types, embedding_dim)
        
    def forward(self, type_ids):
        """获取类型嵌入
        
        Args:
            type_ids: 类型ID张量 [batch_size] 或单个整数
            
        Returns:
            嵌入张量 [batch_size, embedding_dim]
        """
        return self.embedding(type_ids)

# 图注意力模块，用于处理可变数量的智能体
class GraphAttentionLayer(nn.Module):
    """稠密图多头注意力层（GATv2 风格）。

    与普通 Transformer 注意力不同，这里先根据智能体有效性/空间距离/编队关系
    构造邻接矩阵，然后只在图边允许的位置计算注意力权重。相比上一版经典 GAT，
    本层使用 GATv2 的动态注意力打分：e_ij = a^T LeakyReLU(W_src h_i + W_dst h_j)，
    使邻居重要性能够随当前节点状态变化，更适合编队保持中的 Leader-Follower 关系。
    输入输出形状均为 [B, N, d_model]。
    """
    def __init__(self, d_model, num_heads, dropout=0.05, negative_slope=0.2,
                 residual=True, add_self_loops=True):
        super(GraphAttentionLayer, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        assert self.head_dim * num_heads == d_model, "d_model必须能被num_heads整除"

        self.residual = residual
        self.add_self_loops = add_self_loops

        # value 分支负责最终信息聚合；src/dst 分支负责动态注意力打分。
        self.node_linear = nn.Linear(d_model, d_model, bias=False)
        self.node_linear_src = nn.Linear(d_model, d_model, bias=False)
        self.node_linear_dst = nn.Linear(d_model, d_model, bias=False)
        self.attn_vec = nn.Parameter(torch.empty(num_heads, self.head_dim))

        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.dropout = nn.Dropout(dropout)
        self.out_linear = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.activation = nn.ELU()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.node_linear.weight)
        nn.init.xavier_uniform_(self.node_linear_src.weight)
        nn.init.xavier_uniform_(self.node_linear_dst.weight)
        nn.init.xavier_uniform_(self.attn_vec)
        nn.init.xavier_uniform_(self.out_linear.weight)
        if self.out_linear.bias is not None:
            nn.init.zeros_(self.out_linear.bias)

    def _normalize_masks(self, node_features, adjacency=None, node_mask=None):
        B, N, _ = node_features.shape
        device = node_features.device

        if node_mask is None:
            node_mask = torch.ones(B, N, dtype=torch.bool, device=device)
        else:
            node_mask = node_mask.to(device=device).bool()

        if adjacency is None:
            adjacency = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
        else:
            adjacency = adjacency.to(device=device).bool()
            if adjacency.dim() == 2:
                adjacency = adjacency.unsqueeze(0).expand(B, -1, -1)
            adjacency = adjacency & node_mask.unsqueeze(1) & node_mask.unsqueeze(2)

        if self.add_self_loops and N > 0:
            eye = torch.eye(N, dtype=torch.bool, device=device).unsqueeze(0)
            adjacency = adjacency | (eye & node_mask.unsqueeze(1) & node_mask.unsqueeze(2))

        return adjacency, node_mask

    def forward(self, node_features, adjacency=None, node_mask=None, create_graph=True):
        """图注意力前向传播。

        Args:
            node_features: 节点特征 [B, N, d_model]
            adjacency: 邻接矩阵 [B, N, N]，True 表示允许消息传递
            node_mask: 节点有效性 [B, N]，True 表示有效节点
            create_graph: False 时会分离输入，保留原代码中的梯度隔离语义
        """
        if node_features.dim() != 3:
            raise ValueError(f"GraphAttentionLayer需要[B,N,D]输入，收到形状: {tuple(node_features.shape)}")

        B, N, _ = node_features.shape
        if N == 0:
            return torch.zeros_like(node_features)

        if not create_graph:
            node_features = node_features.detach()

        adjacency, node_mask = self._normalize_masks(node_features, adjacency, node_mask)

        # value 分支
        h_value = self.node_linear(node_features)
        h_value = h_value.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, Dh]

        # GATv2 动态注意力分支：不同 query 节点会得到不同的邻居排序。
        h_src = self.node_linear_src(node_features)
        h_dst = self.node_linear_dst(node_features)
        h_src = h_src.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, Dh]
        h_dst = h_dst.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, Dh]

        pair_features = self.leaky_relu(h_src.unsqueeze(3) + h_dst.unsqueeze(2))  # [B, H, N, N, Dh]
        scores = (pair_features * self.attn_vec.view(1, self.num_heads, 1, 1, self.head_dim)).sum(dim=-1)

        edge_mask = adjacency.unsqueeze(1)  # [B, 1, N, N]
        scores = scores.masked_fill(~edge_mask, -1e9)

        # 极端情况下某行没有有效边时避免 softmax 产生 NaN；正常有自环不会触发。
        empty_rows = ~edge_mask.any(dim=-1, keepdim=True)
        scores = scores.masked_fill(empty_rows, 0.0)

        attention = torch.softmax(scores, dim=-1)
        attention = attention.masked_fill(~edge_mask, 0.0)
        attention = self.dropout(attention)

        out = torch.matmul(attention, h_value)  # [B, H, N, Dh]
        out = out.transpose(1, 2).contiguous().view(B, N, self.d_model)
        out = self.out_linear(out)

        if self.residual:
            out = self.norm(node_features + out)
        else:
            out = self.norm(out)
        out = self.activation(out)
        out = out * node_mask.unsqueeze(-1).to(out.dtype)
        return out


def build_valid_adjacency(node_mask, include_self=True):
    """根据节点有效性构造全连接图邻接矩阵。True 表示有效边。"""
    node_mask = node_mask.bool()
    adjacency = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
    if not include_self and adjacency.size(-1) > 0:
        eye = torch.eye(adjacency.size(-1), dtype=torch.bool, device=adjacency.device).unsqueeze(0)
        adjacency = adjacency & ~eye
    return adjacency


def build_spatial_adjacency(agent_obs, node_mask=None, distance_threshold=0.45, include_self=True):
    """根据智能体观测中的二维位置构造空间图。

    观测前两维在环境中是归一化 x/y 坐标；当距离小于阈值时建立边。
    为了避免早期训练图过稀导致信息断开，若某个有效节点只有自环，则自动连接到所有有效节点。
    """
    if agent_obs.dim() != 3:
        raise ValueError(f"build_spatial_adjacency需要[B,N,S]输入，收到形状: {tuple(agent_obs.shape)}")

    B, N, _ = agent_obs.shape
    device = agent_obs.device
    if node_mask is None:
        node_mask = torch.ones(B, N, dtype=torch.bool, device=device)
    else:
        node_mask = node_mask.to(device=device).bool()

    if N == 0:
        return torch.zeros(B, 0, 0, dtype=torch.bool, device=device)

    positions = agent_obs[..., :2]
    distances = torch.cdist(positions, positions, p=2)
    adjacency = distances <= distance_threshold
    adjacency = adjacency & node_mask.unsqueeze(1) & node_mask.unsqueeze(2)

    if include_self:
        eye = torch.eye(N, dtype=torch.bool, device=device).unsqueeze(0)
        adjacency = adjacency | (eye & node_mask.unsqueeze(1) & node_mask.unsqueeze(2))

    # 稳定性兜底：如果某个有效节点没有邻居（除自环外），连接到全部有效节点。
    valid_full = build_valid_adjacency(node_mask, include_self=True)
    degree = adjacency.sum(dim=-1)
    sparse_rows = node_mask & (degree <= 1)
    if sparse_rows.any():
        adjacency = torch.where(sparse_rows.unsqueeze(-1), valid_full, adjacency)

    return adjacency


def build_formation_adjacency(
    agent_obs,
    node_mask=None,
    leader_index=0,
    distance_threshold=0.65,
    k_neighbors=2,
    force_leader_edges=True,
    include_self=True
):
    """面向编队保持的混合图邻接矩阵。

    仅改变图注意力中的信息传递结构，不改变环境、奖励和状态定义。
    设计原则：
    1. 距离阈值边：保留局部避障/局部协同能力；
    2. k近邻边：避免训练中队形拉开后图过稀；
    3. Leader强制边：保证Leader作为编队锚点始终传递给有效Follower；
    4. 自环与兜底全连接：避免有效节点无入边导致注意力退化。

    Args:
        agent_obs: [B, N, S]，默认前两维为归一化 x/y 坐标
        node_mask: [B, N]，True 表示有效节点
        leader_index: Leader 在当前节点序列中的位置
            - Leader Actor / Critic: 节点0为Leader
            - Follower Actor: 节点0为当前Follower，节点1为Leader
    """
    if agent_obs.dim() != 3:
        raise ValueError(f"build_formation_adjacency需要[B,N,S]输入，收到形状: {tuple(agent_obs.shape)}")

    B, N, _ = agent_obs.shape
    device = agent_obs.device

    if node_mask is None:
        node_mask = torch.ones(B, N, dtype=torch.bool, device=device)
    else:
        node_mask = node_mask.to(device=device).bool()

    if N == 0:
        return torch.zeros(B, 0, 0, dtype=torch.bool, device=device)

    valid_pair = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
    positions = agent_obs[..., :2]
    distances = torch.cdist(positions, positions, p=2)  # [B, N, N]

    # 1. 距离阈值边
    adjacency = (distances <= distance_threshold) & valid_pair

    # 2. k近邻补边，保持编队信息通路稳定。
    if N > 1 and k_neighbors > 0:
        eye = torch.eye(N, dtype=torch.bool, device=device).unsqueeze(0)
        masked_distances = distances.masked_fill(~valid_pair, float("inf"))
        masked_distances = masked_distances.masked_fill(eye, float("inf"))

        k = min(k_neighbors, N - 1)
        knn_dist, knn_idx = torch.topk(masked_distances, k=k, dim=-1, largest=False)
        finite_knn = torch.isfinite(knn_dist)

        knn_adj = torch.zeros(B, N, N, dtype=torch.bool, device=device)
        knn_adj.scatter_(dim=2, index=knn_idx, src=finite_knn)
        knn_adj = (knn_adj | knn_adj.transpose(1, 2)) & valid_pair
        adjacency = adjacency | knn_adj

    # 3. Leader强制连接所有有效节点，强化编队锚点信息。
    if force_leader_edges and 0 <= leader_index < N:
        leader_valid = node_mask[:, leader_index]  # [B]
        adjacency[:, leader_index, :] = adjacency[:, leader_index, :] | (leader_valid.unsqueeze(-1) & node_mask)
        adjacency[:, :, leader_index] = adjacency[:, :, leader_index] | (node_mask & leader_valid.unsqueeze(-1))

    # 4. 自环
    if include_self:
        eye = torch.eye(N, dtype=torch.bool, device=device).unsqueeze(0)
        adjacency = adjacency | (eye & valid_pair)

    # 5. 兜底：如果某个有效节点仍只有自环，则连接所有有效节点。
    valid_full = build_valid_adjacency(node_mask, include_self=True)
    degree = adjacency.sum(dim=-1)
    sparse_rows = node_mask & (degree <= 1)
    if sparse_rows.any():
        adjacency = torch.where(sparse_rows.unsqueeze(-1), valid_full, adjacency)

    return adjacency


# 保留旧类名，兼容旧接口；新主线 Actor/Critic 已改用 GraphAttentionLayer。
class MultiHeadAttention(nn.Module):
    """兼容旧代码的多头注意力层。

    mask 语义统一为 True/1 表示允许注意力，False/0 表示屏蔽。
    新增的 GAT 主线不再依赖这个类，但保留它避免旧实验入口失效。
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        assert self.head_dim * num_heads == d_model, "d_model必须能被num_heads整除"
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out_linear = nn.Linear(d_model, d_model)

    def forward(self, queries, keys, values, mask=None, create_graph=True):
        batch_size = queries.shape[0]
        n_queries = queries.shape[1]
        n_keys = keys.shape[1]
        n_values = values.shape[1]

        if n_keys == 0 or n_values == 0:
            return torch.zeros((batch_size, n_queries, self.d_model), device=queries.device, dtype=queries.dtype)

        if not create_graph:
            queries = queries.detach()
            keys = keys.detach()
            values = values.detach()

        q = self.q_linear(queries).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(keys).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(values).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        scale = torch.sqrt(torch.tensor(self.head_dim, dtype=q.dtype, device=q.device))
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale
        if mask is not None:
            mask = mask.to(device=scores.device).bool()
            scores = scores.masked_fill(~mask, -1e9)
            empty_rows = ~mask.any(dim=-1, keepdim=True)
            scores = scores.masked_fill(empty_rows, 0.0)

        attention = torch.softmax(scores, dim=-1)
        if mask is not None:
            attention = attention.masked_fill(~mask, 0.0)
        attention = self.dropout(attention)
        output = torch.matmul(attention, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.out_linear(output)


# 观测动作编码器
class ObservationActionEncoder(nn.Module):
    """智能体观测与动作编码器
    
    将智能体的观测和动作编码为隐藏表示，并融合类型嵌入信息
    """
    def __init__(self, state_dim, action_dim, hidden_dim=AGENT_EMBEDDING_DIM, embedding_dim=TYPE_EMBEDDING_DIM):
        """初始化编码器
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            hidden_dim: 隐藏层维度
            embedding_dim: 类型嵌入维度
        """
        super(ObservationActionEncoder, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        
        # 智能体类型嵌入
        self.type_embedding = AgentTypeEmbedding(num_types=NUM_AGENT_TYPES, embedding_dim=embedding_dim)
        
        # 状态-动作编码网络
        self.sa_encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim - embedding_dim)  # 预留embedding_dim维度给类型信息
        )
        
        # 输出编码与类型嵌入融合网络
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
    
    def forward(self, states, actions, agent_type_ids):
        """前向传播
        
        Args:
            states: 状态张量 [batch_size, state_dim]
            actions: 动作张量 [batch_size, action_dim]
            agent_type_ids: 智能体类型ID [batch_size] 或单个整数
            
        Returns:
            编码后的表示 [batch_size, hidden_dim]
        """
        # 状态和动作拼接
        sa_input = torch.cat([states, actions], dim=1)  # [batch_size, state_dim + action_dim]
        
        # 编码状态-动作对
        sa_encoding = self.sa_encoder(sa_input)  # [batch_size, hidden_dim - embedding_dim]
        
        # 获取类型嵌入
        # 确保agent_type_ids形状正确
        if isinstance(agent_type_ids, int):
            type_ids = torch.tensor([agent_type_ids], device=states.device).expand(states.shape[0])
        else:
            type_ids = agent_type_ids
            
        type_embed = self.type_embedding(type_ids)  # [batch_size, embedding_dim]
        
        # 拼接状态-动作编码和类型嵌入
        combined = torch.cat([sa_encoding, type_embed], dim=1)  # [batch_size, hidden_dim]
        
        # 融合
        output = self.fusion_layer(combined)  # [batch_size, hidden_dim]
        
        return output


# 自注意力Critic网络
class AttentionCriticNet(nn.Module):
    """基于自注意力的Critic网络
    
    能够处理任意数量的智能体，无需填充策略
    支持异构智能体(主机/从机)，通过类型嵌入区分不同角色
    """
    def __init__(self, state_dim_per_agent, action_dim_per_agent, hidden_dim=256, num_heads=4, dropout=0.1, 
                 agent_embed_dim=AGENT_EMBEDDING_DIM):
        """初始化自注意力Critic网络
        
        Args:
            state_dim_per_agent: 每个智能体的状态维度
            action_dim_per_agent: 每个智能体的动作维度
            hidden_dim: 隐藏层维度
            num_heads: 注意力头数
            dropout: Dropout概率
            agent_embed_dim: 智能体嵌入维度
        """
        super(AttentionCriticNet, self).__init__()
        
        # 观测-动作编码器
        self.encoder = ObservationActionEncoder(
            state_dim=state_dim_per_agent,
            action_dim=action_dim_per_agent,
            hidden_dim=agent_embed_dim
        )
        
        # 图注意力层
        self.attention = GraphAttentionLayer(agent_embed_dim, num_heads, dropout)
        
        # Q1值输出层
        self.q1_layers = nn.Sequential(
            nn.Linear(agent_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Q2值输出层(双Q网络结构)
        self.q2_layers = nn.Sequential(
            nn.Linear(agent_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, states_list, actions_list, agent_types_list, create_graph=True):
        """前向传播
        
        Args:
            states_list: 各智能体状态列表 [B, state_dim] * n_agents
            actions_list: 各智能体动作列表 [B, action_dim] * n_agents
            agent_types_list: 各智能体类型列表 (主机=0，从机=1)
            create_graph: 是否创建可导的计算图，用于智能体间梯度隔离
            
        Returns:
            q1, q2: 两个Q值 [batch_size, 1]
        """
        batch_size = states_list[0].shape[0]
        n_agents = len(states_list)
        
        # 编码每个智能体的状态-动作对
        encoded_agents = []
        for i in range(n_agents):
            state_i = states_list[i]
            action_i = actions_list[i]
            agent_type = agent_types_list[i]
            # 使用正确的编码器API调用
            agent_encoding = self.encoder(
                states=state_i,
                actions=action_i,
                agent_type_ids=agent_type
            )  # [batch_size, agent_embed_dim]
            encoded_agents.append(agent_encoding)
        
        # 将编码后的智能体表示堆叠为一个3D张量
        agents_tensor = torch.stack(encoded_agents, dim=1)  # [batch_size, n_agents, agent_embed_dim]
        
        # 图注意力处理：基于观测位置构造动态图，每个智能体只从图邻居接收信息
        states_tensor = torch.stack(states_list, dim=1)  # [batch_size, n_agents, state_dim]
        node_mask = torch.ones(batch_size, n_agents, dtype=torch.bool, device=agents_tensor.device)
        adjacency = build_spatial_adjacency(states_tensor, node_mask=node_mask)
        attended = self.attention(
            agents_tensor,
            adjacency=adjacency,
            node_mask=node_mask,
            create_graph=create_graph
        )  # [batch_size, n_agents, agent_embed_dim]
        
        # 全局池化，得到所有智能体的聚合表示
        global_feature = attended.mean(dim=1)  # [batch_size, agent_embed_dim]
        
        # 计算Q值
        q1 = self.q1_layers(global_feature)
        q2 = self.q2_layers(global_feature)
        
        return q1, q2


# 基于自注意力的中心化Critic
class AttentionCritic:
    """基于自注意力的中心化Critic
    
    使用自注意力机制处理可变数量的智能体，无需填充策略
    支持异构智能体(主机/从机)，通过类型嵌入区分不同角色
    """
    def __init__(self, state_dim_per_agent, action_dim_per_agent, hidden_dim=256, num_heads=4, value_lr=3e-4, tau=1e-2):
        """初始化自注意力Critic
        
        Args:
            state_dim_per_agent: 每个智能体的状态维度
            action_dim_per_agent: 每个智能体的动作维度
            hidden_dim: 隐藏层维度
            num_heads: 注意力头数
            value_lr: 学习率
            tau: 软更新参数
        """
        self.state_dim_per_agent = state_dim_per_agent
        self.action_dim_per_agent = action_dim_per_agent
        self.tau = tau
        self.device = torch.device("cpu")
        
        # 初始化网络
        self.critic_v = AttentionCriticNet(state_dim_per_agent, action_dim_per_agent, hidden_dim, num_heads)
        self.target_critic_v = AttentionCriticNet(state_dim_per_agent, action_dim_per_agent, hidden_dim, num_heads)
        
        # 初始化目标网络参数
        self.target_critic_v.load_state_dict(self.critic_v.state_dict())
        
        # 创建优化器
        self.optimizer = torch.optim.Adam(self.critic_v.parameters(), lr=value_lr, eps=1e-5)
        
        log(f"自注意力Critic初始化完成: 状态维度={state_dim_per_agent}/智能体, 动作维度={action_dim_per_agent}/智能体", LOG_INFO)
        
    def to(self, device):
        """将Critic模型移动到指定设备
        
        Args:
            device: 目标设备
            
        Returns:
            self: 支持链式调用
        """
        self.device = device
        self.critic_v = self.critic_v.to(device)
        self.target_critic_v = self.target_critic_v.to(device)
        
        # 手动移动优化器状态
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)
                    
        log(f"自注意力Critic已移动到设备: {device}", LOG_DEBUG)
        return self
    
    def _decompose_batch(self, global_state, global_action, n_agents):
        """将批次拆分为各智能体的状态和动作列表，并确定智能体类型
        
        Args:
            global_state: 全局状态张量 [batch_size, state_dim * n_agents]
            global_action: 全局动作张量 [batch_size, action_dim * n_agents]
            n_agents: 智能体数量
            
        Returns:
            states_list: 各智能体状态列表
            actions_list: 各智能体动作列表
            agent_types_list: 各智能体类型列表(主机=0，从机=1)
        """
        batch_size = global_state.shape[0]
        
        states_list = []
        actions_list = []
        agent_types_list = []
        
        for i in range(n_agents):
            # 按智能体拆分状态和动作
            start_idx_s = i * self.state_dim_per_agent
            end_idx_s = start_idx_s + self.state_dim_per_agent
            
            start_idx_a = i * self.action_dim_per_agent
            end_idx_a = start_idx_a + self.action_dim_per_agent
            
            states_list.append(global_state[:, start_idx_s:end_idx_s])
            actions_list.append(global_action[:, start_idx_a:end_idx_a])
            
            # 确定智能体类型：第一个智能体是主机(0)，其余是从机(1)
            agent_type = LEADER_TYPE_ID if i == 0 else FOLLOWER_TYPE_ID
            agent_types_list.append(agent_type)
        
        return states_list, actions_list, agent_types_list
    
    def get_v(self, global_state, global_action, create_graph=True):
        """计算Q值
        
        Args:
            global_state: 全局状态张量 [batch_size, state_dim * n_agents]
            global_action: 全局动作张量 [batch_size, action_dim * n_agents]
            create_graph: 是否创建可导的计算图，用于智能体间梯度隔离
            
        Returns:
            q1, q2: 两个Q值
        """
        batch_size = global_state.shape[0]
        
        # 计算智能体数量
        n_agents = min(
            global_state.shape[1] // self.state_dim_per_agent, 
            global_action.shape[1] // self.action_dim_per_agent
        )
        
        # 拆分批次数据
        states_list, actions_list, agent_types_list = self._decompose_batch(
            global_state, global_action, n_agents
        )
        
        # 使用注意力网络计算Q值
        return self.critic_v(states_list, actions_list, agent_types_list, create_graph=create_graph)
    
    def target_get_v(self, global_state, global_action, create_graph=False):
        """使用目标网络计算Q值
        
        Args:
            global_state: 全局状态张量 [batch_size, state_dim * n_agents]
            global_action: 全局动作张量 [batch_size, action_dim * n_agents]
            create_graph: 是否创建可导的计算图，用于智能体间梯度隔离 (默认为False，目标网络不需要梯度)
            
        Returns:
            q1, q2: 两个Q值
        """
        batch_size = global_state.shape[0]
        
        # 计算智能体数量
        n_agents = min(
            global_state.shape[1] // self.state_dim_per_agent, 
            global_action.shape[1] // self.action_dim_per_agent
        )
        
        # 拆分批次数据
        states_list, actions_list, agent_types_list = self._decompose_batch(
            global_state, global_action, n_agents
        )
        
        # 使用目标网络计算Q值
        return self.target_critic_v(states_list, actions_list, agent_types_list, create_graph=create_graph)
        
    def learn(self, current_q1, current_q2, target_q):
        """更新Critic网络
        
        Args:
            current_q1: 当前Q1值 [batch_size, 1]
            current_q2: 当前Q2值 [batch_size, 1]
            target_q: 目标Q值 [batch_size, 1]
            
        Returns:
            loss: 损失值
        """
        # 计算损失
        loss = torch.nn.MSELoss()(current_q1, target_q) + torch.nn.MSELoss()(current_q2, target_q)
        
        # 执行优化
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
        
    def soft_update(self):
        """软更新目标网络参数"""
        for target_param, param in zip(self.target_critic_v.parameters(), self.critic_v.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
            
    def state_dict(self):
        """返回状态字典，用于保存
        
        Returns:
            dict: 包含网络参数和元数据的字典
        """
        return {
            'critic_v': self.critic_v.state_dict(),
            'target_critic_v': self.target_critic_v.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'state_dim_per_agent': self.state_dim_per_agent,
            'action_dim_per_agent': self.action_dim_per_agent
        }
        
    def load_state_dict(self, state_dict):
        """从状态字典加载参数
        
        Args:
            state_dict: 包含网络参数和元数据的字典
            
        Returns:
            bool: 是否成功加载
        """
        # 检查维度是否匹配
        if state_dict.get('state_dim_per_agent') != self.state_dim_per_agent or state_dict.get('action_dim_per_agent') != self.action_dim_per_agent:
            log(f"警告: 维度不匹配，无法加载网络权重。模型: [{state_dict.get('state_dim_per_agent')}/智能体, {state_dict.get('action_dim_per_agent')}/智能体], 当前: [{self.state_dim_per_agent}/智能体, {self.action_dim_per_agent}/智能体]", LOG_WARNING)
            return False
                
        # 维度匹配，加载网络权重和优化器状态
        try:
            self.critic_v.load_state_dict(state_dict['critic_v'])
            self.target_critic_v.load_state_dict(state_dict['target_critic_v'])
            self.optimizer.load_state_dict(state_dict['optimizer'])
            log(f"自注意力Critic参数加载成功", LOG_INFO)
            return True
        except Exception as e:
            log(f"加载自注意力Critic参数失败: {e}", LOG_ERROR)
            return False


# 创建一个动态Critic类，可以处理可变数量的智能体
class DynamicCritic:
    """动态Critic类，能够处理可变数量的智能体输入
    
    这个类与原始Critic类接口兼容，但内部实现能处理不同数量的智能体状态和动作。
    在course learning框架中使用这个类替代原始Critic类。
    """
    
    def __init__(self, state_dim_per_agent: int, action_dim_per_agent: int):
        """初始化动态Critic
        
        Args:
            state_dim_per_agent: 每个智能体的状态维度
            action_dim_per_agent: 每个智能体的动作维度
        """
        self.state_dim_per_agent = state_dim_per_agent
        self.action_dim_per_agent = action_dim_per_agent
        
        # 创建一个每个智能体自己的Q网络
        self.critic_v = CriticNet(state_dim_per_agent, action_dim_per_agent)
        self.target_critic_v = CriticNet(state_dim_per_agent, action_dim_per_agent)
        
        # 复制参数到目标网络
        self.target_critic_v.load_state_dict(self.critic_v.state_dict())
        
        # 创建优化器
        self.optimizer = torch.optim.Adam(self.critic_v.parameters(), lr=3e-3, eps=1e-5)
        
    def soft_update(self):
        """对目标网络进行软更新"""
        for target_param, param in zip(self.target_critic_v.parameters(), self.critic_v.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
            
    def get_v(self, s, a):
        """获取Q值
        
        Args:
            s: 智能体的状态，形状为 [batch_size, state_dim_per_agent]
            a: 智能体的动作，形状为 [batch_size, action_dim_per_agent]
            
        Returns:
            (q1, q2) Q值对
        """
        return self.critic_v(s, a)
        
    def target_get_v(self, s, a):
        """获取目标网络的Q值
        
        Args:
            s: 智能体的状态
            a: 智能体的动作
            
        Returns:
            (q1, q2) 目标Q值对
        """
        return self.target_critic_v(s, a)
        
    def learn(self, current_q1, current_q2, target_q):
        """更新Critic网络
        
        Args:
            current_q1: A Tensor of shape [batch_size, 1]
            current_q2: A Tensor of shape [batch_size, 1]
            target_q: A Tensor of shape [batch_size, 1] or [batch_size, n]
        """
        # 检查并确保张量形状一致
        if current_q1.shape != target_q.shape:
            log(f"调整目标Q值形状: {target_q.shape} -> {current_q1.shape}", LOG_INFO, throttle=5)
            try:
                # 特殊处理[batch_size, 2]到[batch_size, 1]的情况
                if target_q.shape[1] == 2 and current_q1.shape[1] == 1:
                    # 方法1: 使用第一列
                    target_q = target_q[:, 0:1]
                    log(f"使用第一列调整目标Q值形状: {target_q.shape}", LOG_DEBUG, throttle=5)
                elif len(current_q1.shape) == len(target_q.shape):
                    # 维度数相同但大小不同
                    if target_q.shape[1] == 1 and current_q1.shape[1] > 1:
                        # 如果目标是单列但当前Q值有多列，则扩展目标
                        target_q = target_q.expand(-1, current_q1.shape[1])
                    elif target_q.shape[1] > 1 and current_q1.shape[1] == 1:
                        # 多列目标，单列当前值
                        # 方法1: 取平均值
                        target_q = target_q.mean(dim=1, keepdim=True)
                        # 方法2: 使用第一列
                        # target_q = target_q[:, 0:1]
                    else:
                        # 尝试安全转换
                        target_q = target_q[:, :current_q1.shape[1]]
                elif len(current_q1.shape) > len(target_q.shape):
                    # 当前Q值维度高于目标，增加目标维度
                    if len(target_q.shape) == 1:  # 1D变2D
                        target_q = target_q.unsqueeze(1)
                    else:
                        log(f"无法自动调整形状，使用备用方案", LOG_WARNING)
                        raise ValueError("维度不匹配")
                else:
                    # 目标维度高于当前Q值，减少目标维度
                    log(f"无法自动调整形状，使用备用方案", LOG_WARNING)
                    raise ValueError("维度不匹配")
                    
            except Exception as e:
                log(f"调整目标Q值形状失败: {e}", LOG_ERROR)
                # 备用方案：创建与current_q1相同形状的零张量并进行值复制
                backup_target = torch.zeros_like(current_q1)
                
                # 尝试复制尽可能多的值
                min_batch = min(target_q.shape[0], current_q1.shape[0])
                
                # 对于2D目标但形状不兼容的情况
                if len(target_q.shape) == 2 and target_q.shape[1] > 0:
                    # 使用第一列，确保兼容性
                    if target_q.shape[1] >= 1:
                        col_idx = 0
                        backup_target[:min_batch, 0] = target_q[:min_batch, col_idx]
                # 对于1D目标
                elif len(target_q.shape) == 1:
                    backup_target[:min_batch, 0] = target_q[:min_batch]
                
                target_q = backup_target
                log(f"已创建备用目标Q值: {target_q.shape}", LOG_WARNING)
            
        # 计算TD误差
        loss = torch.nn.MSELoss()(current_q1, target_q) + torch.nn.MSELoss()(current_q2, target_q)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


# 类型感知Actor网络
class TypeAwareActorNet(nn.Module):
    """类型感知的Actor网络
    
    支持异构智能体(主机/从机)，通过类型嵌入区分不同角色
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        """初始化类型感知的Actor网络
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            hidden_dim: 隐藏层维度
        """
        super(TypeAwareActorNet, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 类型嵌入层
        self.type_embed = nn.Embedding(NUM_AGENT_TYPES, TYPE_EMBEDDING_DIM)
        
        # 状态和类型信息的组合处理网络
        combined_input_dim = state_dim + TYPE_EMBEDDING_DIM
        
        # 策略网络
        self.policy_net = nn.Sequential(
            nn.Linear(combined_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 均值和标准差输出
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state, agent_type_id):
        """前向传播
        
        Args:
            state: 状态张量 [batch_size, state_dim]
            agent_type_id: 智能体类型ID (整数或张量)
            
        Returns:
            mean: 动作均值
            std: 动作标准差
        """
        # 将类型ID转换为张量并确保设备一致
        if isinstance(agent_type_id, int):
            type_ids = torch.tensor([agent_type_id], device=state.device).expand(state.shape[0])
        else:
            type_ids = agent_type_id.to(state.device)
            
        # 获取类型嵌入
        type_embedding = self.type_embed(type_ids)  # [batch_size, embed_dim]
        
        # 连接状态和类型嵌入
        combined_input = torch.cat([state, type_embedding], dim=1)
        
        # 策略网络处理
        features = self.policy_net(combined_input)
        
        # 计算均值和标准差
        mean = torch.tanh(self.mean_layer(features)) * max_action
        log_std = self.log_std_layer(features)
        log_std = torch.clamp(log_std, -20, 2)  # 避免数值不稳定
        std = torch.exp(log_std)
        
        return mean, std


# 类型感知Actor
class TypeAwareActor:
    """类型感知的Actor
    
    考虑智能体类型(主机/从机)的Actor，使用类型嵌入区分不同角色
    """
    def __init__(self, state_dim, action_dim, agent_type_id=LEADER_TYPE_ID, lr=3e-4):
        """初始化类型感知Actor
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            agent_type_id: 智能体类型ID (0=主机，1=从机)
            lr: 学习率
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.agent_type_id = agent_type_id
        self.device = torch.device("cpu")
        
        # 创建策略网络
        self.action_net = TypeAwareActorNet(state_dim, action_dim)
        
        # 创建优化器
        self.optimizer = torch.optim.Adam(self.action_net.parameters(), lr=lr)
        
        log(f"创建类型感知Actor: 类型={'主机' if agent_type_id == LEADER_TYPE_ID else '从机'}, 状态维度={state_dim}, 动作维度={action_dim}", LOG_INFO)
    
    def to(self, device):
        """将Actor模型移动到指定设备
        
        Args:
            device: 目标设备
            
        Returns:
            self: 支持链式调用
        """
        self.device = device
        self.action_net = self.action_net.to(device)
        
        # 手动移动优化器状态
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)
                    
        return self
    
    def choose_action(self, s, evaluate=False):
        """选择动作
        
        Args:
            s: 状态
            evaluate: 是否为评估模式
            
        Returns:
            选择的动作
        """
        # 确保状态是numpy数组
        if not isinstance(s, np.ndarray):
            s = np.array(s)
            
        # 确保状态维度正确
        if s.size != self.state_dim:
            log(f"警告: 状态维度不匹配，调整维度 {s.size} -> {self.state_dim}", LOG_WARNING)
            # 尝试调整维度
            if s.size > self.state_dim:
                s = s[:self.state_dim]
            else:
                padded_s = np.zeros(self.state_dim)
                padded_s[:s.size] = s
                s = padded_s
        
        # 转换为张量并移动到正确设备
        state = torch.FloatTensor(s).unsqueeze(0).to(self.device)  # [1, state_dim]
        
        # 策略网络前向传播
        with torch.no_grad():
            mean, std = self.action_net(state, self.agent_type_id)
            
            if evaluate:
                # 评估模式使用均值
                action = mean
            else:
                # 训练模式使用随机采样
                normal = torch.distributions.Normal(mean, std)
                action = normal.sample()
                
            # 限制动作范围
            action = torch.clamp(action, min_action, max_action)
        
        return action.squeeze(0).cpu().numpy()
    
    def evaluate(self, s, create_graph=True):
        """评估状态，获取动作和日志概率
        
        Args:
            s: 智能体的状态，形状为 [batch_size, state_dim]
            create_graph: 是否创建可导的计算图
            
        Returns:
            (action, log_prob) 动作和日志概率
        """
        # 检查状态维度
        if len(s.shape) != 2:
            raise ValueError(f"期望状态形状为 [batch_size, state_dim]，实际为 {s.shape}")
            
        if s.shape[1] != self.state_dim:
            raise ValueError(f"状态维度不匹配，期望 {self.state_dim}，实际 {s.shape[1]}")
        
        # 确保状态在正确的设备上
        s = s.to(self.device)
        
        # 如果不需要创建计算图，则分离输入状态
        if not create_graph:
            s = s.detach()
        
        # 获取策略的均值和标准差，传入智能体类型
        mean, std = self.action_net(s, self.agent_type_id)
        
        # 创建正态分布
        dist = torch.distributions.Normal(mean, std)
        
        # 使用重参数化技巧采样
        noise = torch.distributions.Normal(0, 1).sample(mean.shape).to(self.device)
        z = mean + std * noise
        
        # 应用tanh变换
        action = torch.tanh(z)
        # 将动作缩放到正确的范围
        action = torch.clamp(action, min_action / max_action, max_action / max_action) * max_action
        
        # 计算对数概率，考虑tanh变换的雅可比行列式
        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        
        return action, log_prob
    
    def load_state_dict(self, state_dict):
        """加载状态字典
        
        Args:
            state_dict: 包含网络参数的字典
        """
        # 这里假设state_dict包含'action_net'和'optimizer'键
        self.action_net.load_state_dict(state_dict['action_net'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        
    def state_dict(self):
        """返回状态字典
        
        Returns:
            包含网络参数的字典
        """
        return {
            'action_net': self.action_net.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }


class MASACEntroy():
    def __init__(self, action_dim=2):
        """初始化温度系数管理器
        
        Args:
            action_dim: 动作维度，用于计算初始的目标熵
        """
        # 基于动作空间大小动态设置目标熵
        # 对于连续动作空间，一个常用启发式是 -dim(A)，即动作空间的负维度
        self.target_entropy = -2
        
        # 初始化为保守的alpha值，确保在训练初期有足够的探索
        self.initial_alpha = 1
        
        # 创建可学习的log_alpha参数
        self.log_alpha = nn.Parameter(torch.log(torch.ones(1) * self.initial_alpha), requires_grad=True)
        self.alpha = self.log_alpha.detach().exp()
        
        # 小一点的学习率，使得alpha的更新更平滑
        self.optimizer = torch.optim.Adam([self.log_alpha], lr=3e-4)
        
        # 设置允许的alpha值范围
        self.min_alpha = 0.01  # 最小alpha值
        self.max_alpha = 1.0  # 最大alpha值
        
        # 监控数据
        self.alpha_history = []
        self.entropy_history = []
    
    def update_target_entropy(self, new_target):
        """更新目标熵
        
        Args:
            new_target: 新的目标熵值
        """
        old_target = self.target_entropy
        self.target_entropy = new_target
        return old_target, new_target
    
    def learn(self, entroy_loss):
        """更新温度系数alpha
        
        Args:
            entroy_loss: 熵损失
            
        Returns:
            当前alpha值和更新后的loss
        """
        # 计算alpha的梯度并更新
        loss = entroy_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 更新alpha值并限制在合理范围内
        self.alpha = self.log_alpha.detach().exp()
        
        # 限制alpha在有效范围内（直接裁剪）
        alpha_value = self.alpha.item()
        if alpha_value < self.min_alpha or alpha_value > self.max_alpha:
            # 记录过大/过小的值
            limited_alpha = max(min(alpha_value, self.max_alpha), self.min_alpha)
            log(f"Alpha值超出范围[{self.min_alpha}, {self.max_alpha}]，从{alpha_value:.4f}限制为{limited_alpha:.4f}", LOG_WARNING, throttle=100)
            
            # 手动设置限制后的alpha
            with torch.no_grad():
                self.log_alpha[:] = torch.log(torch.tensor(limited_alpha))
            self.alpha = self.log_alpha.detach().exp()
        
        # 记录历史
        self.alpha_history.append(self.alpha.item())
        if len(self.alpha_history) > 1000:  # 限制历史长度
            self.alpha_history = self.alpha_history[-1000:]
            
        return self.alpha.item(), loss.item()
    
    def get_alpha_stats(self):
        """获取alpha统计信息
        
        Returns:
            字典，包含当前值、均值和方差
        """
        if not self.alpha_history:
            return {"current": self.alpha.item(), "mean": self.alpha.item(), "std": 0.0}
            
        return {
            "current": self.alpha.item(),
            "mean": np.mean(self.alpha_history),
            "std": np.std(self.alpha_history)
        }

class MASACAdapter:
    """MASAC适配器类
    
    将现有的Actor、Critic和Entropy类包装为课程学习框架需要的接口
    """
    
    def __init__(self, n_agents: int, state_dim: int, action_dim: int, batch_size=256, policy_lr=1e-4, entropy_lr=1e-4, target_entropy=-1):
        """初始化MASAC适配器
        
        Args:
            n_agents: 智能体数量
            state_dim: 状态维度
            action_dim: 动作维度
        """
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device("cpu")  # 默认使用CPU
        
        # 创建智能体池，使用DynamicActor处理单个智能体的状态
        self.agent_pool = AgentPool(max_agents=n_agents, state_dim=state_dim, action_dim=action_dim)
        
        # 为兼容性保留actors变量
        try:
            # 确保 actors 列表长度与 n_agents 一致
            pool_actors = self.agent_pool.agents
            self.actors = pool_actors[:n_agents]
            
            log(f"初始化 actors 列表，长度: {len(self.actors)}", LOG_DEBUG)
            if len(self.actors) != n_agents:
                log(f"警告: 初始化的 actors 列表长度 ({len(self.actors)}) 与 n_agents ({n_agents}) 不匹配", LOG_WARNING)
        except AttributeError:
            # 向后兼容，使用旧方法
            log(f"AgentPool 对象没有 'agents' 属性，使用 get_all_agents() 方法", LOG_INFO)
            self.actors = self.agent_pool.get_all_agents()
        except Exception as e:
            log(f"初始化 actors 列表时出错: {e}", LOG_ERROR)
            self.actors = []
        
        # 创建Entropy实例列表 - 为每个智能体创建一个独立的Entropy，使用实际动作维度
        self.entroys = [MASACEntroy(action_dim=action_dim) for _ in range(n_agents)]
        
        # 定义主要组件（用于知识迁移）
        self.actor = self.actors[0] if self.actors else None
        
        print(f"MASAC适配器初始化完成，支持 {n_agents} 个智能体 (CTDE范式)")
        print(f"状态维度: {state_dim}, 动作维度: {action_dim}")
        print(f"使用去中心化Actor，每个智能体只接收其自身的观测")
        print(f"初始温度系数Alpha: {self.entroys[0].alpha.item() if self.entroys else 'N/A'}, 目标熵: {self.entroys[0].target_entropy if self.entroys else 'N/A'}")
        
    def to(self, device):
        """将所有模型移动到指定设备
        
        Args:
            device: 目标设备
            
        Returns:
            self: 允许链式调用
        """
        self.device = device
        
        # 移动所有Actor的模型
        for actor in self.actors:
            if hasattr(actor, 'action_net'):
                actor.action_net.to(device)
            if hasattr(actor, 'optimizer'):
                # 优化器状态需要手动移动
                for state in actor.optimizer.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.to(device)
        
        # 移动所有Entropy的张量
        for entroy in self.entroys:
            if hasattr(entroy, 'log_alpha') and torch.is_tensor(entroy.log_alpha):
                entroy.log_alpha = nn.Parameter(entroy.log_alpha.detach().clone().to(device), requires_grad=True)
                entroy.alpha = entroy.log_alpha.detach().exp()
            if hasattr(entroy, 'optimizer'):
                entroy.optimizer = torch.optim.Adam([entroy.log_alpha], lr=3e-4)
                # 优化器状态需要手动移动
                for state in entroy.optimizer.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.to(device)
        
        # 移动智能体池
        if hasattr(self.agent_pool, 'to'):
            self.agent_pool.to(device)
            
        log(f"MASAC适配器已移动到设备: {device}", LOG_INFO)
        return self
    
    def select_action(self, state, evaluate=False):
        """选择动作
        
        Args:
            state: 观察状态数组，形状为[n_agents, state_dim]
            evaluate: 是否为评估模式
            
        Returns:
            numpy数组形状为[n_agents, action_dim]
        """
        try:
            # 检查状态是否有效
            if state is None:
                log(f"错误: 状态为空", LOG_ERROR)
                return np.zeros((self.n_agents, self.action_dim))
                
            # 检查实际智能体数量
            actual_n_agents = len(state)
            if actual_n_agents != self.n_agents:
                log(f"检测到智能体数量不匹配: 预期={self.n_agents}, 实际={actual_n_agents}", LOG_INFO, throttle=60)
                
                # 自动处理智能体数量变化
                if hasattr(self, 'actors') and hasattr(self, 'entroys'):
                    # 检查当前有多少个可用的Actor
                    available_actors = len(self.actors)
                    
                    # 准备动作数组，初始大小为实际智能体数量
                    actions = np.zeros((actual_n_agents, self.action_dim))
                    
                    # 根据智能体数量的变化情况来处理
                    if actual_n_agents <= available_actors:
                        # 如果实际智能体数量 <= 可用Actor数量，直接使用前actual_n_agents个Actor
                        for i in range(actual_n_agents):
                            # 获取Actor并选择动作
                            actor = self.actors[i]
                            actions[i] = actor.choose_action(state[i], evaluate)
                    else:
                        # 如果实际智能体数量 > 可用Actor数量，需要重复使用Actor
                        log(f"智能体数量({actual_n_agents})大于可用Actor数量({available_actors})，循环使用Actor", LOG_WARNING)
                        for i in range(actual_n_agents):
                            # 计算要使用的Actor索引，循环利用可用Actor
                            actor_idx = i % available_actors
                            # 获取Actor并选择动作
                            actor = self.actors[actor_idx]
                            actions[i] = actor.choose_action(state[i], evaluate)
                    
                    return actions
                else:
                    log(f"错误: actors或entroys不存在", LOG_ERROR)
                    return np.zeros((actual_n_agents, self.action_dim))
            else:
                # 标准流程 - 智能体数量匹配时
                actions = np.zeros((self.n_agents, self.action_dim))
                for i, actor in enumerate(self.actors):
                    actions[i] = actor.choose_action(state[i], evaluate)
                return actions
        except Exception as e:
            log(f"选择动作时出错: {e}", LOG_ERROR)
            import traceback
            traceback.print_exc()
            # 返回一个安全的空动作
            if hasattr(state, '__len__'):
                return np.zeros((len(state), self.action_dim))
            else:
                return np.zeros((1, self.action_dim))
    
    def update(self, batch=None):
        """更新模型
        
        Args:
            batch: 经验批次
            
        Returns:
            损失值元组
        """
        # 这里简化了更新逻辑，实际实现需要和main_SAC.py中对应
        # 通常在main_SAC.py的训练循环中完成，而不是单独的方法
        return 0.0, 0.0, 0.0
        
    def clone(self):
        """创建当前适配器的深拷贝
        
        Returns:
            MASACAdapter的深拷贝
        """
        new_adapter = MASACAdapter(self.n_agents, self.state_dim, self.action_dim)
        
        # 复制所有智能体的网络参数
        for i in range(self.n_agents):
            # 复制Actor
            new_adapter.actors[i].action_net.load_state_dict(
                copy.deepcopy(self.actors[i].action_net.state_dict())
            )
            new_adapter.actors[i].optimizer.load_state_dict(
                copy.deepcopy(self.actors[i].optimizer.state_dict())
            )
            
            # 复制Entropy
            new_adapter.entroys[i].log_alpha = torch.tensor(
                self.entroys[i].log_alpha.clone().detach().item(),
                requires_grad=True
            )
            new_adapter.entroys[i].alpha = torch.tensor(
                self.entroys[i].alpha.clone().detach().item()
            )
            new_adapter.entroys[i].optimizer.load_state_dict(
                copy.deepcopy(self.entroys[i].optimizer.state_dict())
            )
            
        return new_adapter
    
    def save(self, path_prefix: str):
        """保存模型
        
        Args:
            path_prefix: 保存路径前缀
        """
        # 保存Actor和Entropy的逻辑保持不变
        try:
            # 保存Actor和Entropy
            for i, actor in enumerate(self.masac_adapter.actors):
                try:
                    save_data = {
                        'net': actor.action_net.state_dict(),
                        'opt': actor.optimizer.state_dict()
                    }
                    torch.save(save_data, f"{path_prefix}_actor_{i}.pth")
                except Exception as e:
                    print(f"保存Actor {i}失败: {e}")
                
            for i, entroy in enumerate(self.masac_adapter.entroys):
                try:
                    save_data = {
                        'alpha': entroy.alpha,
                        'log_alpha': entroy.log_alpha,
                        'opt': entroy.optimizer.state_dict()
                    }
                    torch.save(save_data, f"{path_prefix}_entroy_{i}.pth")
                except Exception as e:
                    print(f"保存Entroy {i}失败: {e}")
                    
            # 保存中心化Critic
            critic_data = self.multi_agent_critic.state_dict()
            torch.save(critic_data, f"{path_prefix}_central_critic.pth")
            print(f"中心化Critic保存成功: {path_prefix}_central_critic.pth")
            
            print(f"模型保存完成: {path_prefix}")
        except Exception as e:
            print(f"保存模型时出错: {e}")
            traceback.print_exc()
    
    def load(self, path_prefix: str, mode: str = 'train'):
        """加载模型
        
        Args:
            path_prefix: 加载路径前缀
            mode: 加载模式，'train'表示加载所有组件用于训练，'eval'表示仅加载Actor用于评估
        """
        # 保持Actor和Entropy的加载逻辑不变
        try:
            # 加载Actor
            for i, actor in enumerate(self.actors):
                try:
                    checkpoint = torch.load(f"{path_prefix}_actor_{i}.pth")
                    actor.action_net.load_state_dict(checkpoint['net'])
                    actor.optimizer.load_state_dict(checkpoint['opt'])
                    print(f"成功加载Actor {i}")
                except Exception as e:
                    print(f"无法加载模型: {path_prefix}_actor_{i}.pth, 错误: {e}")
            
            # 仅在训练模式下加载Entropy和Critic
            if mode == 'train':
                # 加载Entropy
                for i, entroy in enumerate(self.entroys):
                    try:
                        checkpoint = torch.load(f"{path_prefix}_entroy_{i}.pth")
                        entroy.alpha = checkpoint['alpha']
                        entroy.log_alpha = checkpoint['log_alpha']
                        entroy.optimizer.load_state_dict(checkpoint['opt'])
                        print(f"成功加载Entroy {i}")
                    except Exception as e:
                        print(f"无法加载模型: {path_prefix}_entroy_{i}.pth, 错误: {e}")
                        traceback.print_exc()
                        
                # 加载中心化Critic
                try:
                    critic_path = f"{path_prefix}_central_critic.pth"
                    critic_checkpoint = torch.load(critic_path)
                    
                    # 检查维度是否匹配
                    if ('state_dim_per_agent' in critic_checkpoint and 'action_dim_per_agent' in critic_checkpoint and
                        (critic_checkpoint['state_dim_per_agent'] != self.multi_agent_critic.state_dim_per_agent or 
                         critic_checkpoint['action_dim_per_agent'] != self.multi_agent_critic.action_dim_per_agent)):
                        
                        # 维度不匹配，打印警告
                        print(f"警告: 自注意力Critic维度不匹配")
                        print(f"模型维度: 状态={critic_checkpoint.get('state_dim_per_agent')}/智能体, 动作={critic_checkpoint.get('action_dim_per_agent')}/智能体")
                        print(f"当前维度: 状态={self.multi_agent_critic.state_dim_per_agent}/智能体, 动作={self.multi_agent_critic.action_dim_per_agent}/智能体")
                        
                        # 尝试只加载优化器状态
                        try:
                            if 'optimizer' in critic_checkpoint:
                                print("尝试只加载优化器状态...")
                                self.multi_agent_critic.optimizer.load_state_dict(critic_checkpoint['optimizer'])
                                print("优化器状态加载成功")
                        except Exception as e:
                            print(f"加载优化器状态失败: {e}")
                    else:
                        # 尝试加载
                        try:
                            load_success = self.multi_agent_critic.load_state_dict(critic_checkpoint)
                            if load_success or load_success is None:  # None表示方法没有返回值
                                print(f"自注意力Critic加载成功")
                            else:
                                print(f"自注意力Critic加载失败: 返回False")
                        except Exception as e:
                            print(f"加载自注意力Critic参数失败: {e}")
                            
                            # 尝试只加载优化器状态
                            try:
                                if 'optimizer' in critic_checkpoint:
                                    print("尝试只加载优化器状态...")
                                    self.multi_agent_critic.optimizer.load_state_dict(critic_checkpoint['optimizer'])
                                    print("优化器状态加载成功")
                            except Exception as e:
                                print(f"加载优化器状态失败: {e}")
                    
                except FileNotFoundError:
                    print(f"找不到中心化Critic文件: {critic_path}")
                    print("可能是使用旧版模型，跳过加载中心化Critic")
                except Exception as e:
                    print(f"加载中心化Critic时出错: {e}")
                    traceback.print_exc()
            else:
                # 评估模式，跳过加载Entropy和Critic
                print(f"以评估模式加载模型，仅加载Actor，跳过Entropy和Critic")
                    
            print(f"模型加载完成: {path_prefix}, 模式: {mode}")
            
        except Exception as e:
            print(f"加载模型时出错: {e}")
            traceback.print_exc()

    def _validate_agent_structures(self):
        """验证智能体相关的数据结构是否一致
        
        检查actors和entroys列表的长度是否与n_agents匹配，并打印警告日志
        
        Returns:
            bool: 如果数据结构一致返回True，否则返回False
        """
        valid = True
        
        # 首先检查基本结构是否存在
        if not hasattr(self, 'actors') or self.actors is None:
            log(f"严重错误: actors列表不存在或为None", LOG_ERROR)
            return False
            
        if not hasattr(self, 'entroys') or self.entroys is None:
            log(f"严重错误: entroys列表不存在或为None", LOG_ERROR)
            return False
        
        # 检查长度
        try:
            if len(self.actors) != self.n_agents:
                log(f"警告: actors列表长度 ({len(self.actors)}) 与n_agents ({self.n_agents}) 不一致", LOG_WARNING)
                valid = False
        except Exception as e:
            log(f"检查actors列表时出错: {e}", LOG_ERROR)
            valid = False
            
        try:
            if len(self.entroys) != self.n_agents:
                log(f"警告: entroys列表长度 ({len(self.entroys)}) 与n_agents ({self.n_agents}) 不一致", LOG_WARNING)
                valid = False
        except Exception as e:
            log(f"检查entroys列表时出错: {e}", LOG_ERROR)
            valid = False
            
        # 检查actors元素是否有效
        for i, actor in enumerate(self.actors):
            if actor is None:
                log(f"警告: actors[{i}]为None", LOG_WARNING)
                valid = False
            elif not hasattr(actor, 'action_net'):
                log(f"警告: actors[{i}]没有action_net属性", LOG_WARNING)
                valid = False
                
        # 检查entroys元素是否有效
        for i, entroy in enumerate(self.entroys):
            if entroy is None:
                log(f"警告: entroys[{i}]为None", LOG_WARNING)
                valid = False
            elif not hasattr(entroy, 'alpha') or not hasattr(entroy, 'log_alpha'):
                log(f"警告: entroys[{i}]缺少必要的属性", LOG_WARNING)
                valid = False
                
        return valid
        
    def _log_system_state(self):
        """记录系统状态
        
        输出适配器当前的主要组件状态
        """
        try:
            # 记录actors列表状态
            actors_count = len(self.actors) if hasattr(self, 'actors') else 0
            log(f"Actors列表长度: {actors_count} (必需的: {self.n_agents})", LOG_INFO)
            actors_ok = actors_count == self.n_agents
            
            # 记录entroys列表状态
            entroys_count = len(self.entroys) if hasattr(self, 'entroys') else 0
            log(f"Entroys列表长度: {entroys_count} (必需的: {self.n_agents})", LOG_INFO)
            entroys_ok = entroys_count == self.n_agents
            
            # 记录温度系数状态
            if entroys_ok and entroys_count > 0:
                alpha_values = [entroy.alpha.item() for entroy in self.entroys]
                avg_alpha = np.mean(alpha_values)
                min_alpha = np.min(alpha_values)
                max_alpha = np.max(alpha_values)
                alpha_std = np.std(alpha_values)
                log(f"温度系数 Alpha: 均值={avg_alpha:.4f}, 最小={min_alpha:.4f}, 最大={max_alpha:.4f}, 标准差={alpha_std:.4f}", LOG_INFO)
                
                # 记录目标熵
                target_entropy_values = [entroy.target_entropy for entroy in self.entroys]
                avg_target = np.mean(target_entropy_values)
                log(f"目标熵: 均值={avg_target:.4f}", LOG_INFO)
            
            # 检查智能体池
            pool_exists = hasattr(self, 'agent_pool')
            pool_size = len(self.agent_pool.agents) if pool_exists and hasattr(self.agent_pool, 'agents') else 0
            log(f"AgentPool状态: 存在={pool_exists}, 大小={pool_size}", LOG_INFO)
            pool_ok = pool_exists and pool_size >= self.n_agents
            
            # 总结系统状态
            system_ok = actors_ok and entroys_ok and pool_ok
            if system_ok:
                log(f"系统状态: 正常", LOG_INFO)
            else:
                log(f"系统状态: 问题! actors_ok={actors_ok}, entroys_ok={entroys_ok}, pool_ok={pool_ok}", LOG_WARNING)
            
            return system_ok
            
        except Exception as e:
            log(f"记录系统状态时出错: {e}", LOG_ERROR)
            traceback.print_exc()
            return False
    
    def update_agent_count(self, n_agents):
        """适应新的智能体数量
        
        Args:
            n_agents: 新的智能体数量
            
        Returns:
            bool: 操作是否成功
        """
        try:
            # 如果数量没变，不做任何事
            if n_agents == self.n_agents:
                log(f"智能体数量没有变化: {n_agents}", LOG_INFO)
                return True
                
            log(f"适应新的智能体数量: {self.n_agents} -> {n_agents}", LOG_INFO)
            
            # 扩展智能体池
            if hasattr(self, 'agent_pool'):
                self.agent_pool.expand(n_agents)
                try:
                    # 直接访问 AgentPool 内部的 agents 列表
                    pool_actors_list = self.agent_pool.agents
                    pool_capacity = len(pool_actors_list)

                    if n_agents > pool_capacity:
                        # 如果请求的数量超过池的实际容量，记录警告并使用所有可用的 Actor
                        log(f"警告: 请求的智能体数量 {n_agents} 超过 AgentPool 实际容量 {pool_capacity}。将使用所有可用的 {pool_capacity} 个 Actor。", LOG_WARNING)
                        self.actors = pool_actors_list[:] # 获取所有可用的 Actor
                    else:
                        # 否则，从池中获取前 n_agents 个 Actor
                        self.actors = pool_actors_list[:n_agents]
                        log(f"从智能体池中获取 {n_agents} 个智能体", LOG_INFO)
                except Exception as e:
                    log(f"访问智能体池失败: {e}", LOG_ERROR)
                    traceback.print_exc()
                    return False
            
            # 扩展或缩减entropy列表
            if hasattr(self, 'entroys'):
                current_entropy_count = len(self.entroys)
                
                if n_agents > current_entropy_count:
                    # 需要创建新的Entropy
                    log(f"扩展温度系数列表: {current_entropy_count} -> {n_agents}", LOG_INFO)
                    
                    # 如果已有Entropy，获取其平均值作为新Entropy的初始值
                    if current_entropy_count > 0:
                        # 获取当前系统中所有Entropy的平均目标值和平均Alpha值
                        avg_target_entropy = sum(e.target_entropy for e in self.entroys) / current_entropy_count
                        avg_alpha = sum(e.alpha.item() for e in self.entroys) / current_entropy_count
                        
                        log(f"使用现有温度系数的平均值初始化新智能体: 目标熵={avg_target_entropy:.4f}, Alpha={avg_alpha:.4f}", LOG_INFO)
                        
                        # 创建新的Entropy并设置为平均值
                        for _ in range(n_agents - current_entropy_count):
                            new_entroy = MASACEntroy(action_dim=self.action_dim)
                            # 设置为平均目标熵
                            new_entroy.update_target_entropy(avg_target_entropy)
                            # 设置为平均Alpha值
                            with torch.no_grad():
                                new_entroy.log_alpha[:] = torch.log(torch.tensor(avg_alpha))
                                new_entroy.alpha = new_entroy.log_alpha.exp()
                            self.entroys.append(new_entroy)
                    else:
                        # 没有现有的Entropy，创建新的
                        self.entroys = [MASACEntroy(action_dim=self.action_dim) for _ in range(n_agents)]
                elif n_agents < current_entropy_count:
                    # 需要缩减Entropy列表
                    log(f"缩减温度系数列表: {current_entropy_count} -> {n_agents}", LOG_INFO)
                    self.entroys = self.entroys[:n_agents]
            else:
                # 创建新的Entropy列表
                log(f"创建新的温度系数列表: {n_agents}个", LOG_INFO)
                self.entroys = [MASACEntroy(action_dim=self.action_dim) for _ in range(n_agents)]
                
            # 更新智能体数量
            self.n_agents = n_agents
            
            # 检查actors和entroys长度是否匹配
            if len(self.actors) != n_agents:
                log(f"警告: actors列表长度 ({len(self.actors)}) 与n_agents ({n_agents}) 不一致", LOG_WARNING)
            
            if len(self.entroys) != n_agents:
                log(f"警告: entroys列表长度 ({len(self.entroys)}) 与n_agents ({n_agents}) 不一致", LOG_WARNING)
                
            # 设置主组件（用于知识迁移）
            self.actor = self.actors[0] if self.actors else None
            
            return True
            
        except Exception as e:
            log(f"适应新的智能体数量时出错: {e}", LOG_ERROR)
            traceback.print_exc()
            return False
    
    def transfer_policy_parameters(self, source_params=None, target_params=None, transfer_ratio=0.6):
        """实现策略参数迁移
        
        这个方法允许在不同任务之间迁移策略参数，以促进知识转移。
        当从一个任务切换到另一个任务时，可以部分保留原任务的策略知识。
        
        Args:
            source_params: 源参数(如果为None，则使用当前模型参数)
            target_params: 目标参数(如果为None，则使用当前模型参数)
            transfer_ratio: 迁移比例，1.0表示完全使用源参数，0.0表示完全使用目标参数
            
        Returns:
            self，允许方法链式调用
        """
        try:
            # 详细记录参数情况
            print(f"开始执行策略参数迁移，迁移比例: {transfer_ratio:.4f}")
            
            # 如果没有提供源参数，使用当前参数作为源
            if source_params is None:
                print(f"未提供源参数，使用当前模型参数作为源")
                source_params = self.get_parameters()
            
            # 确保有 actors 属性
            if not hasattr(self, 'actors') or not self.actors:
                print(f"模型没有 actors 属性或为空，无法执行参数迁移")
                return self
            
            # 对每个 actor 网络应用迁移
            for i, actor in enumerate(self.actors):
                if hasattr(actor, 'action_net'):
                    # 获取当前网络参数
                    current_params = actor.action_net.state_dict()
                    
                    # 检查源参数中是否有对应的参数
                    if 'actors' in source_params and f'actor_{i}' in source_params['actors']:
                        source_actor_params = source_params['actors'][f'actor_{i}']
                        
                        # 创建混合参数
                        mixed_params = {}
                        for key in current_params:
                            if key in source_actor_params:
                                # 按比例混合参数
                                mixed_params[key] = source_actor_params[key] * transfer_ratio + \
                                                  current_params[key] * (1 - transfer_ratio)
                            else:
                                mixed_params[key] = current_params[key]
                        
                        # 应用混合参数
                        actor.action_net.load_state_dict(mixed_params)
                        print(f"已完成 Actor {i} 的参数迁移，迁移比例: {transfer_ratio:.4f}")
                    else:
                        print(f"源参数中找不到 Actor {i} 的参数，跳过此 Actor 的迁移")
            
            print(f"策略参数迁移完成")
            return self
            
        except Exception as e:
            print(f"参数迁移过程中出错: {e}")
            import traceback
            traceback.print_exc()
            return self

    def get_parameters(self):
        """获取MASACAdapter的参数
        
        用于保存和进行知识迁移。获取所有Actor的参数。
        
        Returns:
            包含模型参数的字典
        """
        try:
            params = {'actors': {}}
            
            # 获取所有Actor的参数
            for i, actor in enumerate(self.actors):
                if hasattr(actor, 'action_net'):
                    params['actors'][f'actor_{i}'] = actor.action_net.state_dict()
            
            # 获取Entropy的参数
            params['entropy'] = {}
            for i, entropy in enumerate(self.entroys):
                if hasattr(entropy, 'log_alpha'):
                    params['entropy'][f'entropy_{i}'] = {
                        'log_alpha': entropy.log_alpha.data,
                        'target_entropy': entropy.target_entropy
                    }
            
            return params
            
        except Exception as e:
            log(f"获取参数时出错: {e}", LOG_ERROR)
            import traceback
            traceback.print_exc()
            return {}


# 多智能体类型Critic管理器
class MultiAgentCritic:
    """多智能体类型Critic管理器
    
    管理不同类型智能体(主机/从机)的Critic网络，每种类型使用独立的注意力Critic
    该类包装了两个AttentionCritic实例，分别用于主机和从机
    """
    def __init__(self, state_dim_per_agent, action_dim_per_agent, hidden_dim=256, num_heads=4, value_lr=3e-4, tau=1e-2):
        """初始化多智能体类型Critic管理器
        
        Args:
            state_dim_per_agent: 每个智能体的状态维度
            action_dim_per_agent: 每个智能体的动作维度
            hidden_dim: 隐藏层维度
            num_heads: 注意力头数
            value_lr: 学习率
            tau: 软更新参数
        """
        self.state_dim_per_agent = state_dim_per_agent
        self.action_dim_per_agent = action_dim_per_agent
        self.tau = tau
        self.device = torch.device("cpu")
        
        # 创建主机的Critic网络
        self.leader_critic = AttentionCritic(state_dim_per_agent, action_dim_per_agent, hidden_dim, num_heads, value_lr, tau)
        
        # 创建从机的Critic网络
        self.follower_critic = AttentionCritic(state_dim_per_agent, action_dim_per_agent, hidden_dim, num_heads, value_lr, tau)
        
        log(f"多智能体类型Critic初始化完成: 状态维度={state_dim_per_agent}/智能体, 动作维度={action_dim_per_agent}/智能体", LOG_INFO)
        log(f"创建了独立的主机Critic和从机Critic", LOG_INFO)
        
    def to(self, device):
        """将所有Critic网络移动到指定设备
        
        Args:
            device: 目标设备
            
        Returns:
            self: 支持链式调用
        """
        self.device = device
        self.leader_critic.to(device)
        self.follower_critic.to(device)
        
        log(f"多智能体类型Critic已移动到设备: {device}", LOG_INFO)
        return self
    
    def _decompose_batch(self, global_state, global_action, n_agents):
        """将批次拆分为各智能体的状态和动作列表，并确定智能体类型
        
        Args:
            global_state: 全局状态张量 [batch_size, state_dim * n_agents]
            global_action: 全局动作张量 [batch_size, action_dim * n_agents]
            n_agents: 智能体数量
            
        Returns:
            states_list: 各智能体状态列表
            actions_list: 各智能体动作列表
            agent_types_list: 各智能体类型列表(主机=0，从机=1)
            leader_indices: 主机智能体的索引列表
            follower_indices: 从机智能体的索引列表
        """
        batch_size = global_state.shape[0]
        
        states_list = []
        actions_list = []
        agent_types_list = []
        leader_indices = []
        follower_indices = []
        
        for i in range(n_agents):
            # 按智能体拆分状态和动作
            start_idx_s = i * self.state_dim_per_agent
            end_idx_s = start_idx_s + self.state_dim_per_agent
            
            start_idx_a = i * self.action_dim_per_agent
            end_idx_a = start_idx_a + self.action_dim_per_agent
            
            states_list.append(global_state[:, start_idx_s:end_idx_s])
            actions_list.append(global_action[:, start_idx_a:end_idx_a])
            
            # 确定智能体类型：第一个智能体是主机(0)，其余是从机(1)
            agent_type = LEADER_TYPE_ID if i == 0 else FOLLOWER_TYPE_ID
            agent_types_list.append(agent_type)
            
            # 记录主机和从机的索引
            if agent_type == LEADER_TYPE_ID:
                leader_indices.append(i)
            else:
                follower_indices.append(i)
        
        return states_list, actions_list, agent_types_list, leader_indices, follower_indices
    
    def get_v(self, global_state, global_action, agent_index=None, create_graph=True):
        """计算指定智能体的Q值
        
        Args:
            global_state: 全局状态张量 [batch_size, state_dim * n_agents]
            global_action: 全局动作张量 [batch_size, action_dim * n_agents]
            agent_index: 指定智能体的索引，None表示获取所有智能体的Q值
            create_graph: 是否创建可导的计算图，用于智能体间梯度隔离
            
        Returns:
            agent_type: 智能体类型 (LEADER_TYPE_ID 或 FOLLOWER_TYPE_ID)
            q1, q2: 两个Q值
        """
        batch_size = global_state.shape[0]
        
        # 计算智能体数量
        n_agents = min(
            global_state.shape[1] // self.state_dim_per_agent, 
            global_action.shape[1] // self.action_dim_per_agent
        )
        
        # 拆分批次数据
        states_list, actions_list, agent_types_list, leader_indices, follower_indices = self._decompose_batch(
            global_state, global_action, n_agents
        )
        
        # 如果指定了智能体索引
        if agent_index is not None:
            if agent_index >= n_agents:
                raise ValueError(f"智能体索引 {agent_index} 超出范围 (0-{n_agents-1})")
                
            # 确定智能体类型
            agent_type = agent_types_list[agent_index]
            
            if agent_type == LEADER_TYPE_ID:
                # 使用主机Critic
                return agent_type, self.leader_critic.get_v(global_state, global_action, create_graph)
            else:
                # 使用从机Critic
                return agent_type, self.follower_critic.get_v(global_state, global_action, create_graph)
        else:
            # 返回所有智能体的Q值（为兼容性保留）
            # 默认使用主机Critic
            return LEADER_TYPE_ID, self.leader_critic.get_v(global_state, global_action, create_graph)
    
    def target_get_v(self, global_state, global_action, agent_index=None, create_graph=False):
        """使用目标网络计算指定智能体的Q值
        
        Args:
            global_state: 全局状态张量 [batch_size, state_dim * n_agents]
            global_action: 全局动作张量 [batch_size, action_dim * n_agents]
            agent_index: 指定智能体的索引，None表示获取所有智能体的Q值
            create_graph: 是否创建可导的计算图，用于智能体间梯度隔离 (默认为False，目标网络不需要梯度)
            
        Returns:
            agent_type: 智能体类型 (LEADER_TYPE_ID 或 FOLLOWER_TYPE_ID)
            q1, q2: 两个Q值
        """
        batch_size = global_state.shape[0]
        
        # 计算智能体数量
        n_agents = min(
            global_state.shape[1] // self.state_dim_per_agent, 
            global_action.shape[1] // self.action_dim_per_agent
        )
        
        # 拆分批次数据
        states_list, actions_list, agent_types_list, leader_indices, follower_indices = self._decompose_batch(
            global_state, global_action, n_agents
        )
        
        # 如果指定了智能体索引
        if agent_index is not None:
            if agent_index >= n_agents:
                raise ValueError(f"智能体索引 {agent_index} 超出范围 (0-{n_agents-1})")
                
            # 确定智能体类型
            agent_type = agent_types_list[agent_index]
            
            if agent_type == LEADER_TYPE_ID:
                # 使用主机Critic
                return agent_type, self.leader_critic.target_get_v(global_state, global_action, create_graph)
            else:
                # 使用从机Critic
                return agent_type, self.follower_critic.target_get_v(global_state, global_action, create_graph)
        else:
            # 返回所有智能体的Q值（为兼容性保留）
            # 默认使用主机Critic
            return LEADER_TYPE_ID, self.leader_critic.target_get_v(global_state, global_action, create_graph)
    
    def learn(self, current_q1, current_q2, target_q, agent_type):
        """更新指定类型的Critic网络
        
        Args:
            current_q1: 当前Q1值 [batch_size, 1]
            current_q2: 当前Q2值 [batch_size, 1]
            target_q: 目标Q值 [batch_size, 1]
            agent_type: 智能体类型 (LEADER_TYPE_ID 或 FOLLOWER_TYPE_ID)
            
        Returns:
            loss: 损失值
        """
        # 根据智能体类型选择对应的Critic
        if agent_type == LEADER_TYPE_ID:
            return self.leader_critic.learn(current_q1, current_q2, target_q)
        else:
            return self.follower_critic.learn(current_q1, current_q2, target_q)
        
    def soft_update(self):
        """软更新所有Critic网络的目标网络参数"""
        self.leader_critic.soft_update()
        self.follower_critic.soft_update()
            
    def state_dict(self):
        """返回状态字典，用于保存
        
        Returns:
            dict: 包含网络参数和元数据的字典
        """
        return {
            'leader_critic': self.leader_critic.state_dict(),
            'follower_critic': self.follower_critic.state_dict(),
            'state_dim_per_agent': self.state_dim_per_agent,
            'action_dim_per_agent': self.action_dim_per_agent
        }
        
    def load_state_dict(self, state_dict):
        """从状态字典加载参数
        
        Args:
            state_dict: 包含网络参数和元数据的字典
            
        Returns:
            bool: 是否成功加载
        """
        # 检查维度是否匹配
        if state_dict.get('state_dim_per_agent') != self.state_dim_per_agent or state_dict.get('action_dim_per_agent') != self.action_dim_per_agent:
            log(f"警告: 维度不匹配，无法加载网络权重。模型: [{state_dict.get('state_dim_per_agent')}/智能体, {state_dict.get('action_dim_per_agent')}/智能体], 当前: [{self.state_dim_per_agent}/智能体, {self.action_dim_per_agent}/智能体]", LOG_WARNING)
            return False
                
        # 维度匹配，加载网络权重
        try:
            if 'leader_critic' in state_dict:
                self.leader_critic.load_state_dict(state_dict['leader_critic'])
            if 'follower_critic' in state_dict:
                self.follower_critic.load_state_dict(state_dict['follower_critic'])
            log(f"多智能体类型Critic参数加载成功", LOG_INFO)
            return True
        except Exception as e:
            log(f"加载多智能体类型Critic参数失败: {e}", LOG_ERROR)
            return False


# MASAC控制器

    def save(self, path):
        """保存模型和参数到文件
        
        Args:
            path: 保存路径
            
        Returns:
            bool: 是否保存成功
        """
        try:
            # 构建保存数据
            save_data = {
                'masac_adapter': self.masac_adapter.state_dict() if hasattr(self, 'masac_adapter') else None,
                'multi_agent_critic': self.multi_agent_critic.state_dict() if hasattr(self, 'multi_agent_critic') else None,
                'n_agents': self.n_agents,
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'memory_n_agents_version': self.memory_n_agents_version,
                'version': '2.0.0'  # 添加版本信息便于未来兼容性检查
            }
            
            # 保存文件
            torch.save(save_data, path)
            log(f"保存模型成功: {path}", LOG_INFO)
            
            return True
            
        except Exception as e:
            log(f"保存模型失败: {e}", LOG_ERROR)
            import traceback
            traceback.print_exc()
            return False
            
    def load(self, path):
        """从文件加载模型和参数
        
        Args:
            path: 模型文件路径
            
        Returns:
            bool: 是否加载成功
        """
        try:
            # 读取数据
            save_data = torch.load(path)
            
            # 检查维度匹配
            saved_state_dim = save_data.get('state_dim')
            saved_action_dim = save_data.get('action_dim')
            
            if saved_state_dim != self.state_dim or saved_action_dim != self.action_dim:
                log(f"维度不匹配! 模型: [{saved_state_dim}, {saved_action_dim}], 当前: [{self.state_dim}, {self.action_dim}]", LOG_WARNING)
                return False
                
            # 获取保存的智能体数量
            saved_n_agents = save_data.get('n_agents', self.n_agents)
            
            # 调整当前控制器适应保存的智能体数量
            if saved_n_agents != self.n_agents:
                log(f"调整控制器以匹配保存的智能体数量: {self.n_agents} -> {saved_n_agents}", LOG_INFO)
                self.adapt_to_agent_count(saved_n_agents)
            
            # 加载MASAC适配器
            if 'masac_adapter' in save_data and save_data['masac_adapter'] is not None and hasattr(self, 'masac_adapter'):
                self.masac_adapter.load_state_dict(save_data['masac_adapter'])
                log("MASAC适配器加载成功", LOG_INFO)
                
            # 加载多智能体类型Critic
            if 'multi_agent_critic' in save_data and save_data['multi_agent_critic'] is not None and hasattr(self, 'multi_agent_critic'):
                self.multi_agent_critic.load_state_dict(save_data['multi_agent_critic'])
                log("多智能体类型Critic加载成功", LOG_INFO)
            
            # 更新记忆版本
            if 'memory_n_agents_version' in save_data:
                self.memory_n_agents_version = save_data['memory_n_agents_version']
                
            # 记录系统状态
            self._log_system_state()
                
            log(f"加载模型成功: {path}", LOG_INFO)
            return True
            
        except Exception as e:
            log(f"加载模型失败: {e}", LOG_ERROR)
            import traceback
            traceback.print_exc()
            return False
