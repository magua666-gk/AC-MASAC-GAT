import numpy as np
import torch
from typing import Dict, Any, List, Optional, Tuple
from main_SAC import Actor, policy_lr, max_action, min_action
import traceback

# 创建一个动态Actor类，可以处理单个智能体的状态
class DynamicActor(Actor):
    """动态调整的Actor，能够处理不同维度的输入状态"""
    
    def __init__(self, state_dim: int, action_dim: int):
        """初始化动态Actor
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
        """
        super().__init__(state_dim, action_dim)
        self.device = torch.device("cpu")
        self.state_dim = state_dim
        self.action_dim = action_dim
        
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
        
    def state_dict(self):
        """获取网络参数状态字典，用于知识迁移
        
        Returns:
            包含网络参数的状态字典
        """
        if hasattr(self.action_net, 'state_dict'):
            return self.action_net.state_dict()
        else:
            raise AttributeError("action_net 没有 state_dict 方法")
        
    def load_state_dict(self, state_dict):
        """从状态字典加载网络参数
        
        Args:
            state_dict: 包含网络参数的状态字典
        """
        if hasattr(self.action_net, 'load_state_dict'):
            self.action_net.load_state_dict(state_dict)
        else:
            raise AttributeError("action_net 没有 load_state_dict 方法")
        
    def named_parameters(self):
        """获取命名参数，用于知识迁移
        
        Returns:
            命名参数的迭代器
        """
        if hasattr(self.action_net, 'named_parameters'):
            return self.action_net.named_parameters()
        else:
            raise AttributeError("action_net 没有 named_parameters 方法")
        
    def evaluate(self, s, create_graph=True):
        """评估状态，获取动作和日志概率
        
        Args:
            s: 智能体的状态，形状为 [batch_size, state_dim]
            create_graph: 是否创建可导的计算图
            
        Returns:
            (action, log_prob) 动作和日志概率
        """
        # 检查维度
        if len(s.shape) != 2:
            raise ValueError(f"期望状态形状为 [batch_size, state_dim]，实际为 {s.shape}")
            
        if s.shape[1] != self.state_dim:
            raise ValueError(f"状态维度不匹配，期望 {self.state_dim}，实际 {s.shape[1]}")
        
        # 确保状态在正确的设备上
        s = s.to(self.device)
        
        # 如果不需要创建计算图，则分离输入状态
        if not create_graph:
            s = s.detach()
            
        # 调用原始Actor的evaluate方法
        return super().evaluate(s)
        
    def choose_action(self, s, evaluate=False):
        """选择动作
        
        Args:
            s: 智能体的状态
            evaluate: 是否为评估模式，评估模式下不添加随机性
            
        Returns:
            选择的动作
        """
        # 确保状态是numpy数组
        if not isinstance(s, np.ndarray):
            s = np.array(s)
            
        # 确保状态维度正确
        if s.size != self.state_dim:
            print(f"警告: 状态维度不匹配，期望 {self.state_dim}，实际 {s.size}，尝试调整")
            # 尝试调整状态维度
            if s.size > self.state_dim:
                # 截断多余的维度
                s = s[:self.state_dim]
            else:
                # 补充缺失的维度
                padded_state = np.zeros(self.state_dim)
                padded_state[:s.size] = s
                s = padded_state
        
        # 转换为张量并移动到正确设备
        inputstate = torch.FloatTensor(s).to(self.device)
        
        # 获取策略的均值和标准差
        mean, std = self.action_net(inputstate)
        
        if evaluate:
            # 评估模式下，使用确定性策略（直接使用均值）
            action = mean
        else:
            # 训练模式下，使用随机策略
            # 直接在目标设备上创建随机张量
            z = torch.randn_like(mean)
            action = torch.tanh(mean + std * z)
            
        # 裁剪动作范围
        action = torch.clamp(action, min_action, max_action)
        
        return action.cpu().detach().numpy()

class AgentPool:
    """智能体池管理类，负责创建和分配不同类型的智能体策略"""
    
    def __init__(self, max_agents: int, state_dim: int, action_dim: int):
        """初始化智能体池
        
        Args:
            max_agents: 最大支持的智能体数量
            state_dim: 状态维度
            action_dim: 动作维度
        """
        self.max_agents = max_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 创建基础智能体策略池 - 使用DynamicActor替代原始Actor
        # DynamicActor能更好地处理不同维度的状态
        self.agents = []
        for i in range(max_agents):
            self.agents.append(DynamicActor(state_dim=state_dim, action_dim=action_dim))
            
        # 特定类型的智能体映射 
        # 修改为更灵活的智能体类型映射，支持多种不同类型
        self.type_mapping = {
            "hero": 0,     # 英雄类型使用第0个策略
            "enemy": 1,    # 敌人类型使用第1个策略
            "default": 0,  # 默认使用第0个策略
            # 可以添加更多类型
            "scout": 2,    # 侦察类型使用第2个策略
            "support": 3   # 支援类型使用第3个策略
        }
        
        # 确保有足够的智能体策略
        for type_name, policy_idx in self.type_mapping.items():
            if policy_idx >= max_agents:
                print(f"警告: 类型 {type_name} 的策略索引 {policy_idx} 超出最大智能体数量 {max_agents}")
                self.type_mapping[type_name] = 0  # 回退到第一个策略
                
        print(f"已创建智能体池，支持 {max_agents} 个智能体，类型映射: {self.type_mapping}")
        print(f"正在使用动态调整的Actor，可处理任意维度的状态")
        
    def get_agent(self, agent_type: str, agent_idx: int) -> DynamicActor:
        """获取指定类型和索引的智能体策略
        
        Args:
            agent_type: 智能体类型 ('hero', 'enemy', 等)
            agent_idx: 智能体在环境中的索引
        
        Returns:
            对应的DynamicActor实例
        """
        try:
            # 检查输入
            if agent_type is None:
                print(f"警告: agent_type为None，使用'default'类型")
                agent_type = "default"
                
            if agent_idx is None or agent_idx < 0:
                print(f"警告: agent_idx无效({agent_idx})，使用索引0")
                agent_idx = 0
                
            # 确定使用哪种策略
            if agent_type in self.type_mapping:
                policy_idx = self.type_mapping[agent_type]
            else:
                # 未知的智能体类型，输出警告并使用默认策略
                print(f"警告: 未知的智能体类型 '{agent_type}'，使用默认类型 'default'")
                policy_idx = self.type_mapping.get("default", 0)
                
            # 为特定的智能体类型应用特殊规则
            if agent_type == "hero":
                # 所有英雄都使用同一个策略
                policy_idx = self.type_mapping["hero"]
            elif agent_type == "enemy":
                # 敌人可以根据索引使用不同的策略
                # 例如，每个敌人使用不同策略，或者所有敌人共享一个策略
                policy_idx = self.type_mapping["enemy"]
                
                # 实现更复杂的逻辑，如每个敌人使用不同的策略：
                # policy_idx = self.type_mapping["enemy"] + agent_idx % 3  # 循环使用3种敌人策略
            
            # 确保索引在有效范围内
            policy_idx = max(0, min(policy_idx, self.max_agents - 1))
            
            return self.agents[policy_idx]
            
        except Exception as e:
            # 如果发生错误，记录并返回默认智能体
            print(f"获取智能体时出错: {e}")
            traceback.print_exc()
            return self.agents[0]  # 始终返回第一个智能体作为后备
    
    def get_all_agents(self) -> List[DynamicActor]:
        """获取所有智能体策略
        
        Returns:
            所有DynamicActor实例的列表
        """
        return self.agents
        
    def get_agent_by_index(self, idx: int) -> DynamicActor:
        """直接通过索引获取智能体策略
        
        Args:
            idx: 智能体策略索引
            
        Returns:
            对应的DynamicActor实例
        """
        idx = min(idx, self.max_agents - 1)
        return self.agents[idx] 

    def expand(self, new_max_agents: int):
        """扩展智能体池以适应更多智能体
        
        Args:
            new_max_agents: 新的最大智能体数量
            
        Returns:
            自身，允许链式调用
        """
        if new_max_agents <= self.max_agents:
            print(f"智能体池已经支持 {self.max_agents} 个智能体，无需扩展")
            return self
            
        print(f"扩展智能体池: {self.max_agents} -> {new_max_agents}")
        
        # 添加新的智能体
        for i in range(self.max_agents, new_max_agents):
            self.agents.append(DynamicActor(state_dim=self.state_dim, action_dim=self.action_dim))
            print(f"  创建智能体 {i}")
            
        # 更新最大智能体数量
        self.max_agents = new_max_agents
        
        # 确保类型映射有效
        for type_name, policy_idx in self.type_mapping.items():
            if policy_idx >= new_max_agents:
                print(f"警告: 类型 {type_name} 的策略索引 {policy_idx} 超出最大智能体数量 {new_max_agents}")
                self.type_mapping[type_name] = 0  # 回退到第一个策略
                
        return self
        
    def to(self, device):
        """将所有智能体移动到指定设备
        
        Args:
            device: 目标设备
            
        Returns:
            self: 支持链式调用
        """
        for agent in self.agents:
            agent.to(device)
            
        print(f"智能体池已移动到设备: {device}")
        return self 