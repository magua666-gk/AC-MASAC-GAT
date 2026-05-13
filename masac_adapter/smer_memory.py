import collections
import random
import numpy as np
import torch
from typing import Dict, List, Tuple, Any, Optional

try:
    from .masac_adapter import log, LOG_INFO, LOG_WARNING, LOG_ERROR, LOG_DEBUG
except ImportError:
    LOG_INFO, LOG_WARNING, LOG_ERROR, LOG_DEBUG = 1, 2, 3, 4
    def log(message, level=LOG_INFO, throttle=0, suppress_repeat=True):
        if level >= LOG_INFO:
            print(message)

class SMERMemory:
    """Structured Masked Experience Replay Memory
    
    Used to store and sample structured experience data containing leader and followers.
    Supports dynamic number of followers and outputs padded batch tensors with masks during sampling.
    """
    
    def __init__(self, capacity: int, obs_dims: Dict[str, int], action_dims: Dict[str, int], device: torch.device):
        """Initialize SMERMemory
        
        Args:
            capacity: Memory capacity
            obs_dims: Observation dimension dictionary, format: {"leader": dim, "followers": dim}
            action_dims: Action dimension dictionary, format: {"leader": dim, "followers": dim}
            device: Device to store tensors
        """
        self.buffer = collections.deque(maxlen=capacity)
        self.capacity = capacity
        self.obs_dims = obs_dims
        self.action_dims = action_dims
        self.device = device
        self.memory_counter = 0
        
        log(f"SMERMemory initialized: capacity={capacity}, obs_dims={obs_dims}, action_dims={action_dims}", LOG_INFO)

    def store_transition(self, observation: Dict, action: Dict, reward: Dict, 
                        next_observation: Dict, done: bool, stage_tag: str = "default_stage"):
        """Store single transition to experience replay buffer
        
        Args:
            observation: Observation dict {"leader": obs_leader, "followers": [obs_f1, obs_f2, ...]}
            action: Action dict {"leader": act_leader, "followers": [act_f1, act_f2, ...]}
            reward: Reward dict {"leader": rew_leader, "followers": [rew_f1, rew_f2, ...]}
            next_observation: Next observation dict {"leader": next_obs_leader, "followers": [next_obs_f1, ...]}
            done: 终止标志
            stage_tag: 阶段标签，用于标记经验所属的课程阶段
        """
        # 验证和转换数据
        obs_converted = self._validate_and_convert_data(observation, self.obs_dims)
        action_converted = self._validate_and_convert_data(action, self.action_dims)
        next_obs_converted = self._validate_and_convert_data(next_observation, self.obs_dims)

        # 将奖励标准化为统一维度 (1)
        converted_reward = self._validate_and_convert_data(reward, {"leader": 1, "followers": 1})
        
        # 创建经验元组
        experience = {
            "observation": obs_converted,
            "action": action_converted,
            "reward": converted_reward,
            "next_observation": next_obs_converted,
            "done": done,
            "stage_tag": stage_tag  # 新增字段：阶段标签
        }
        
        # 添加到缓冲区
        self.buffer.append(experience)
        
        # 更新计数器 - 使用缓冲区实际长度
        self.memory_counter = len(self.buffer)
        
        # 记录日志
        if self.memory_counter % 1000 == 0 and self.memory_counter > 0:
            log(f"SMERMemory: 已存储 {self.memory_counter} 条经验 (容量: {self.capacity})", LOG_DEBUG)

    def _validate_and_convert_data(self, data_dict: Dict, expected_dims_dict: Dict) -> Dict:
        """验证并转换输入数据的类型和形状
        
        Args:
            data_dict: 输入数据字典
            expected_dims_dict: 预期的维度字典（obs_dims或action_dims）
            
        Returns:
            转换后的数据字典
        """
        converted = {}
        for key, value in data_dict.items():
            if key == "leader":
                # 处理单个实体的数据
                if isinstance(value, (list, tuple)):
                    value = np.array(value, dtype=np.float32)
                elif isinstance(value, (int, float)):
                    value = np.array([value], dtype=np.float32)
                elif not isinstance(value, np.ndarray):
                    raise TypeError(f"leader {key} 的数据类型必须是 numpy.ndarray, list, tuple, int 或 float")
                
                # 确保形状正确
                if key in expected_dims_dict and value.shape != (expected_dims_dict[key],):
                    value = value.reshape(expected_dims_dict[key])
                
            elif key == "followers":
                # 处理实体列表的数据
                if not isinstance(value, list):
                    raise TypeError("followers 必须是列表")
                
                converted_followers = []
                for item in value:
                    if isinstance(item, (list, tuple)):
                        item = np.array(item, dtype=np.float32)
                    elif isinstance(item, (int, float)):
                        item = np.array([item], dtype=np.float32)
                    elif not isinstance(item, np.ndarray):
                        raise TypeError(f"follower 数据项的类型必须是 numpy.ndarray, list, tuple, int 或 float")
                    
                    # 确保形状正确
                    if key in expected_dims_dict and item.shape != (expected_dims_dict[key],):
                        item = item.reshape(expected_dims_dict[key])
                    
                    converted_followers.append(item)
                value = converted_followers
            
            converted[key] = value
        
        return converted

    def _pad_and_stack_sequence(self, data_sequences: List[List[np.ndarray]], max_len: int, 
                              feature_dim: int, dtype=np.float32) -> Tuple[np.ndarray, np.ndarray]:
        """填充和堆叠序列数据
        
        Args:
            data_sequences: 数据序列列表，每个元素是一个样本的 follower 数据列表
            max_len: 最大序列长度
            feature_dim: 特征维度
            dtype: 数据类型
            
        Returns:
            padded_batch: 填充后的批次数据
            masks: 对应的掩码
        """
        batch_size = len(data_sequences)
        padded_batch = np.zeros((batch_size, max_len, feature_dim), dtype=dtype)
        masks = np.zeros((batch_size, max_len), dtype=np.float32)
        
        for i, seq in enumerate(data_sequences):
            if not seq:  # 处理空序列
                continue
                
            num_entities = len(seq)
            valid_len = min(num_entities, max_len)
            
            # 提取并填充有效数据
            valid_data = seq[:valid_len]
            for j, data in enumerate(valid_data):
                if data.shape != (feature_dim,):
                    data = data.reshape(feature_dim)
                padded_batch[i, j] = data
            
            # 设置掩码
            masks[i, :valid_len] = 1.0
        
        return padded_batch, masks

    def _stack_single_entity(self, data_list: List[np.ndarray], feature_dim: int, 
                           dtype=np.float32) -> np.ndarray:
        """堆叠单个实体的数据
        
        Args:
            data_list: 数据列表
            feature_dim: 特征维度
            dtype: 数据类型
            
        Returns:
            堆叠后的数据数组
        """
        # 过滤无效数据
        valid_data = [data for data in data_list if data is not None and data.size > 0]
        
        if not valid_data:
            return np.zeros((len(data_list), feature_dim), dtype=dtype)
        
        # 确保所有数据形状一致
        processed_data = []
        for data in valid_data:
            if data.shape != (feature_dim,):
                try:
                    data = data.reshape(feature_dim)
                except ValueError as e:
                    log(f"数据形状转换错误: {e}", LOG_WARNING)
                    continue
            processed_data.append(data)
        
        if not processed_data:
            return np.zeros((len(data_list), feature_dim), dtype=dtype)
        
        # 堆叠数据
        stacked = np.stack(processed_data)
        
        # 处理数据丢失情况
        if len(stacked) < len(data_list):
            result = np.zeros((len(data_list), feature_dim), dtype=dtype)
            result[:len(stacked)] = stacked
            return result
        
        return stacked

    def sample(self, batch_size: int, current_stage_tag: str = "default_stage", current_stage_number: int = 0) -> Optional[Tuple[Dict[str, Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]]:
        """从经验回放缓冲区中采样一批经验
        
        Args:
            batch_size: 采样批次大小
            current_stage_tag: 当前课程阶段标签，用于区分新旧经验
            current_stage_number: 当前课程阶段编号，用于计算新旧经验采样比例
            
        Returns:
            批次数据元组 (batch_data, batch_masks) 或 None (如果缓冲区为空)
        """
        buffer_len = self.memory_counter
        min_buffer_size_for_sampling = batch_size  # 最小采样阈值

        if buffer_len < min_buffer_size_for_sampling:
            log(f"SMERMemory: 缓冲区大小 {buffer_len} 小于最小采样阈值 {min_buffer_size_for_sampling}，无法采样。", LOG_DEBUG)
            return None
        
        # 1. 计算采样比例 - 旧经验比例随阶段编号递减，新经验比例增加
        old_experience_ratio = max(0.8 - 0.1 * current_stage_number, 0.2)
        old_experience_ratio = min(max(old_experience_ratio, 0.0), 1.0)  # 确保在 [0,1] 范围内

        num_old_samples_target = int(batch_size * old_experience_ratio)
        num_new_samples_target = batch_size - num_old_samples_target

        # 2. 将经验分为新旧两部分
        all_experiences_list = list(self.buffer)  # 转换为列表便于筛选

        new_experiences_pool = [exp for exp in all_experiences_list if exp.get("stage_tag") == current_stage_tag]
        old_experiences_pool = [exp for exp in all_experiences_list if exp.get("stage_tag") != current_stage_tag]

        len_new_pool = len(new_experiences_pool)
        len_old_pool = len(old_experiences_pool)

        # 3. 根据可用性调整实际采样数量
        actual_num_new_samples = min(num_new_samples_target, len_new_pool)
        actual_num_old_samples = min(num_old_samples_target, len_old_pool)

        # 4. 尝试互相补充以达到 batch_size
        # 如果新经验不足，尝试从旧经验中补充
        if actual_num_new_samples < num_new_samples_target:
            needed_from_old = num_new_samples_target - actual_num_new_samples
            actual_num_old_samples = min(actual_num_old_samples + needed_from_old, len_old_pool)
        
        # 如果旧经验不足，尝试从新经验中补充
        if actual_num_old_samples < num_old_samples_target:
            needed_from_new = num_old_samples_target - actual_num_old_samples
            actual_num_new_samples = min(actual_num_new_samples + needed_from_new, len_new_pool)

        # 5. 确保总数不超过 batch_size，并尽可能等于 batch_size
        current_total_target = actual_num_new_samples + actual_num_old_samples
        if current_total_target > batch_size:
            if current_total_target > 0:  # 避免除以零
                scale_factor = batch_size / current_total_target
                actual_num_new_samples = int(actual_num_new_samples * scale_factor)
                actual_num_old_samples = batch_size - actual_num_new_samples  # 确保精确等于 batch_size
            else:  # 极少情况
                log("SMERMemory: 采样逻辑错误，目标总采样数为零。", LOG_ERROR)
                return None

        # 6. 从各自的池中随机采样
        sampled_new_experiences = []
        if len_new_pool > 0 and actual_num_new_samples > 0:
            sampled_new_experiences = random.sample(new_experiences_pool, actual_num_new_samples)
        
        sampled_old_experiences = []
        if len_old_pool > 0 and actual_num_old_samples > 0:
            sampled_old_experiences = random.sample(old_experiences_pool, actual_num_old_samples)

        # 合并采样结果
        sampled_experiences = sampled_new_experiences + sampled_old_experiences
        final_sampled_count = len(sampled_experiences)

        if final_sampled_count == 0:
            log(f"SMERMemory: 阶段 {current_stage_tag} (编号 {current_stage_number}) 没有采样到任何经验。"
                f"新池: {len_new_pool}, 旧池: {len_old_pool}. 目标新: {num_new_samples_target}, 旧: {num_old_samples_target}. "
                f"实际新: {actual_num_new_samples}, 旧: {actual_num_old_samples}", LOG_WARNING)
            return None
        
        # 如果采样数量少于 batch_size 但大于0，则接受当前数量
        if final_sampled_count < batch_size:
            log(f"SMERMemory: 阶段 {current_stage_tag} 采样数量 {final_sampled_count} 小于 batch_size {batch_size}。使用可用样本继续。", LOG_DEBUG)
        
        # --- 记录采样比例日志 ---
        if len_new_pool > 0 or len_old_pool > 0:
            actual_new_ratio = actual_num_new_samples / final_sampled_count if final_sampled_count > 0 else 0
            log(f"SMERMemory: 阶段 {current_stage_tag} (编号 {current_stage_number}) 采样 - 目标新旧比例: {1-old_experience_ratio:.2f}:{old_experience_ratio:.2f}, "
                f"实际新旧比例: {actual_new_ratio:.2f}:{1-actual_new_ratio:.2f}, 新池: {len_new_pool}, 旧池: {len_old_pool}", LOG_DEBUG)
        
        # --- 后续的数据堆叠和转换逻辑 ---
        # A. 确定批次内 max_followers
        max_followers = 0
        if "followers" in self.obs_dims:
            for exp in sampled_experiences:
                if "observation" in exp and "followers" in exp["observation"] and exp["observation"]["followers"] is not None:
                   max_followers = max(max_followers, len(exp["observation"]["followers"]))
        
        # B. 初始化 batch_data 和 batch_masks
        batch_data = {"observation": {}, "action": {}, "reward": {}, "next_observation": {}, "done": None}
        batch_masks = {}
        
        # 获取实际采样到的批次大小
        actual_sampled_batch_size = len(sampled_experiences)

        # C. 处理 Leader 数据
        batch_data["observation"]["leader"] = self._stack_single_entity(
            [exp["observation"]["leader"] for exp in sampled_experiences], self.obs_dims["leader"])
        batch_data["action"]["leader"] = self._stack_single_entity(
            [exp["action"]["leader"] for exp in sampled_experiences], self.action_dims["leader"])
        leader_rewards_raw = [exp["reward"]["leader"] for exp in sampled_experiences]
        batch_data["reward"]["leader"] = self._stack_single_entity(leader_rewards_raw, 1).reshape(-1, 1)
        batch_data["next_observation"]["leader"] = self._stack_single_entity(
            [exp["next_observation"]["leader"] for exp in sampled_experiences], self.obs_dims["leader"])
        
        # D. 处理 Followers 数据
        if "followers" in self.obs_dims and self.obs_dims["followers"] > 0:
            if max_followers > 0:
                follower_obs_list = [exp["observation"].get("followers", []) for exp in sampled_experiences]
                follower_action_list = [exp["action"].get("followers", []) for exp in sampled_experiences]
                follower_reward_raw_list = [exp["reward"].get("followers", []) for exp in sampled_experiences]
                next_follower_obs_list = [exp["next_observation"].get("followers", []) for exp in sampled_experiences]
            
                padded_obs, follower_mask = self._pad_and_stack_sequence(
                        follower_obs_list, max_followers, self.obs_dims["followers"])
                batch_data["observation"]["followers"] = padded_obs
                batch_masks["followers"] = follower_mask
                
                padded_action, _ = self._pad_and_stack_sequence(
                        follower_action_list, max_followers, self.action_dims["followers"])
                batch_data["action"]["followers"] = padded_action
                # 处理 follower 奖励
                processed_follower_reward_list = []
                for rew_list_for_exp in follower_reward_raw_list:
                    current_exp_rewards = []
                    if isinstance(rew_list_for_exp, (list, np.ndarray)):
                        for r_val in rew_list_for_exp:
                                current_exp_rewards.append(np.array([r_val], dtype=np.float32))
                        processed_follower_reward_list.append(current_exp_rewards)
                    
                padded_reward, _ = self._pad_and_stack_sequence(
                        processed_follower_reward_list, max_followers, 1)  # 奖励维度为1
                batch_data["reward"]["followers"] = padded_reward
                
                padded_next_obs, _ = self._pad_and_stack_sequence(
                        next_follower_obs_list, max_followers, self.obs_dims["followers"])
                batch_data["next_observation"]["followers"] = padded_next_obs
            else:
                # 没有 follower 的情况，创建空占位符
                batch_data["observation"]["followers"] = np.zeros((actual_sampled_batch_size, 0, self.obs_dims["followers"]))
                batch_data["action"]["followers"] = np.zeros((actual_sampled_batch_size, 0, self.action_dims.get("followers", self.obs_dims["followers"])))
                batch_data["reward"]["followers"] = np.zeros((actual_sampled_batch_size, 0, 1))
                batch_data["next_observation"]["followers"] = np.zeros((actual_sampled_batch_size, 0, self.obs_dims["followers"]))
                batch_masks["followers"] = np.zeros((actual_sampled_batch_size, 0))
        else:
            # 不需要处理 follower 的情况，创建默认空占位符
            batch_data["observation"]["followers"] = np.zeros((actual_sampled_batch_size, 0, 1))
            batch_data["action"]["followers"] = np.zeros((actual_sampled_batch_size, 0, 1))
            batch_data["reward"]["followers"] = np.zeros((actual_sampled_batch_size, 0, 1))
            batch_data["next_observation"]["followers"] = np.zeros((actual_sampled_batch_size, 0, 1))
            batch_masks["followers"] = np.zeros((actual_sampled_batch_size, 0))
        
        # E. 处理 done 标志
        batch_data["done"] = np.array([exp["done"] for exp in sampled_experiences], dtype=np.float32).reshape(-1, 1)
        
        # F. 将 NumPy 数组转换为 PyTorch 张量
        tensor_batch_data = {}
        tensor_batch_masks = {}
        for key_major, value_major in batch_data.items():
            if isinstance(value_major, dict):
                tensor_batch_data[key_major] = {}
                for key_minor, value_arr in value_major.items():
                    tensor_batch_data[key_major][key_minor] = torch.from_numpy(value_arr).to(self.device).float()
            else:
                tensor_batch_data[key_major] = torch.from_numpy(value_major).to(self.device).float()
        
        for key, value_mask_arr in batch_masks.items():
            tensor_batch_masks[key] = torch.from_numpy(value_mask_arr).to(self.device).float()
        
        return tensor_batch_data, tensor_batch_masks

    def __len__(self) -> int:
        """返回当前存储的经验数量"""
        return len(self.buffer) 