import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 导入动作边界常量
try:
    from masac_adapter.masac_adapter import max_action, min_action
except ImportError:
    # 如果导入失败，使用默认值
    max_action = 1.0
    min_action = -1.0

# 定义角色ID常量
LEADER_TYPE_ID = 0
FOLLOWER_TYPE_ID = 1

class SimpleLeaderActorNet(nn.Module):
    """简单的Leader Actor网络，无注意力机制"""
    
    def __init__(self, state_dim, action_dim, hidden_dims=[256, 128]):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 构建特征提取层
        layers = []
        input_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        self.feature_layers = nn.Sequential(*layers)
        
        # 动作均值和对数标准差输出层
        self.mean_layer = nn.Linear(input_dim, action_dim)
        self.log_std_layer = nn.Linear(input_dim, action_dim)
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, state):
        """前向传播
        
        Args:
            state: 输入状态 [batch_size, state_dim]
            
        Returns:
            mean: 动作均值 [batch_size, action_dim]
            std: 动作标准差 [batch_size, action_dim]
        """
        # 通过特征提取层
        features = self.feature_layers(state)
        
        # 生成均值（使用tanh限制范围并缩放到动作范围）
        mean = torch.tanh(self.mean_layer(features)) * max_action
        
        # 计算对数标准差，并限制在合理范围内，避免训练不稳定
        log_std = self.log_std_layer(features)
        log_std = torch.clamp(log_std, -20, 2)  # 防止标准差过大或过小
        
        # 计算标准差
        std = torch.exp(log_std)
        
        return mean, std
    
    def choose_action(self, obs, evaluate=False):
        """选择动作
        
        Args:
            obs: 观测 (numpy 数组)
            evaluate: 是否为评估模式（使用确定性策略）
            
        Returns:
            action: 动作 (numpy 数组)
        """
        # 确保输入是正确形状的张量
        if isinstance(obs, np.ndarray):
            obs = torch.FloatTensor(obs)
        
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        
        # 移动到正确的设备
        device = next(self.parameters()).device
        obs = obs.to(device)
        
        with torch.no_grad():
            mean, std = self.forward(obs)
            
            if evaluate:
                # 评估模式：使用确定性策略（均值）
                action = mean
            else:
                # 训练模式：使用随机采样
                dist = torch.distributions.Normal(mean, std)
                action = dist.sample()
        
        # 返回 numpy 数组
        return action.cpu().numpy().squeeze()
    
    def evaluate(self, obs):
        """评估动作和对数概率（用于SAC训练）
        
        Args:
            obs: 观测张量 [batch_size, state_dim]
            
        Returns:
            action: 动作张量 [batch_size, action_dim]
            log_prob: 对数概率张量 [batch_size, 1]
        """
        # 确保输入是张量格式
        if isinstance(obs, np.ndarray):
            obs = torch.FloatTensor(obs)
        
        # 移动到正确的设备
        device = next(self.parameters()).device
        obs = obs.to(device)
        
        # 获取动作分布参数
        mean, std = self.forward(obs)
        
        # 创建正态分布
        dist = torch.distributions.Normal(mean, std)
        
        # 使用重参数化技巧采样动作
        z = dist.rsample()
        
        # 应用 tanh 变换
        action = torch.tanh(z)
        
        # 缩放到动作范围
        scaled_action = action * max_action
        
        # 计算 tanh 校正的对数概率
        log_prob = dist.log_prob(z)
        
        # 应用 tanh 校正项
        # log prob = log prob - log(1 - tanh(z)^2) 
        # = log prob - log(1 - action^2)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        
        # 对动作维度求和
        log_prob = log_prob.sum(dim=1, keepdim=True)
        
        return scaled_action, log_prob

class SimpleFollowerActorNet(nn.Module):
    """简单的Follower Actor网络，无注意力机制"""
    
    def __init__(self, state_dim, action_dim, hidden_dims=[256, 128]):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 构建特征提取层
        layers = []
        input_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        self.feature_layers = nn.Sequential(*layers)
        
        # 动作均值和对数标准差输出层
        self.mean_layer = nn.Linear(input_dim, action_dim)
        self.log_std_layer = nn.Linear(input_dim, action_dim)
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, state):
        """前向传播
        
        Args:
            state: 输入状态 [batch_size, state_dim]
            
        Returns:
            mean: 动作均值 [batch_size, action_dim]
            std: 动作标准差 [batch_size, action_dim]
        """
        # 通过特征提取层
        features = self.feature_layers(state)
        
        # 生成均值（使用tanh限制范围并缩放到动作范围）
        mean = torch.tanh(self.mean_layer(features)) * max_action
        
        # 计算对数标准差，并限制在合理范围内，避免训练不稳定
        log_std = self.log_std_layer(features)
        log_std = torch.clamp(log_std, -20, 2)  # 防止标准差过大或过小
        
        # 计算标准差
        std = torch.exp(log_std)
        
        return mean, std
    
    def choose_action(self, obs, evaluate=False):
        """选择动作
        
        Args:
            obs: 观测 (numpy 数组)
            evaluate: 是否为评估模式（使用确定性策略）
            
        Returns:
            action: 动作 (numpy 数组)
        """
        # 确保输入是正确形状的张量
        if isinstance(obs, np.ndarray):
            obs = torch.FloatTensor(obs)
        
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        
        # 移动到正确的设备
        device = next(self.parameters()).device
        obs = obs.to(device)
        
        with torch.no_grad():
            mean, std = self.forward(obs)
            
            if evaluate:
                # 评估模式：使用确定性策略（均值）
                action = mean
            else:
                # 训练模式：使用随机采样
                dist = torch.distributions.Normal(mean, std)
                action = dist.sample()
        
        # 返回 numpy 数组
        return action.cpu().numpy().squeeze()
    
    def evaluate(self, obs):
        """评估动作和对数概率（用于SAC训练）
        
        Args:
            obs: 观测张量 [batch_size, state_dim]
            
        Returns:
            action: 动作张量 [batch_size, action_dim]
            log_prob: 对数概率张量 [batch_size, 1]
        """
        # 确保输入是张量格式
        if isinstance(obs, np.ndarray):
            obs = torch.FloatTensor(obs)
        
        # 移动到正确的设备
        device = next(self.parameters()).device
        obs = obs.to(device)
        
        # 获取动作分布参数
        mean, std = self.forward(obs)
        
        # 创建正态分布
        dist = torch.distributions.Normal(mean, std)
        
        # 使用重参数化技巧采样动作
        z = dist.rsample()
        
        # 应用 tanh 变换
        action = torch.tanh(z)
        
        # 缩放到动作范围
        scaled_action = action * max_action
        
        # 计算 tanh 校正的对数概率
        log_prob = dist.log_prob(z)
        
        # 应用 tanh 校正项
        # log prob = log prob - log(1 - tanh(z)^2) 
        # = log prob - log(1 - action^2)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        
        # 对动作维度求和
        log_prob = log_prob.sum(dim=1, keepdim=True)
        
        return scaled_action, log_prob