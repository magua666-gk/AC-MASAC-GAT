class MASACController:
    def transfer_critic_parameters(self, source_params=None, target_params=None, transfer_ratio=0.3):
        """实现中央化Critic参数迁移
        
        在CTDE（集中训练，分散执行）范式下，这个方法允许在不同任务之间迁移Critic参数，
        以促进知识转移。当从一个任务切换到另一个任务时，可以部分保留原任务的评价知识。
        
        Args:
            source_params: 源参数(如果为None，则使用当前模型参数)
            target_params: 目标参数(如果为None，则使用当前模型参数)
            transfer_ratio: 迁移比例，1.0表示完全使用源参数，0.0表示完全使用目标参数
            
        Returns:
            self，允许方法链式调用
        """
        try:
            # 详细记录参数情况
            print(f"开始执行中央化Critic参数迁移，迁移比例: {transfer_ratio:.4f}")
            
            # 如果没有提供源参数，使用当前参数作为源
            if source_params is None:
                print(f"未提供源参数，使用当前模型参数作为源")
                source_params = self.get_parameters()
            
            # 确保有 centralized_critic 属性
            if not hasattr(self, 'centralized_critic') or self.centralized_critic is None:
                print(f"模型没有centralized_critic属性或为空，无法执行参数迁移")
                return self
            
            # 获取当前网络参数
            current_params = self.centralized_critic.critic_net.state_dict()
            
            # 检查源参数中是否有对应的参数
            if 'centralized_critic' in source_params:
                source_critic_params = source_params['centralized_critic']
                
                # 创建混合参数
                mixed_params = {}
                for key in current_params:
                    if key in source_critic_params:
                        # 按比例混合参数
                        mixed_params[key] = source_critic_params[key] * transfer_ratio + \
                                          current_params[key] * (1 - transfer_ratio)
                    else:
                        mixed_params[key] = current_params[key]
                
                # 应用混合参数
                self.centralized_critic.critic_net.load_state_dict(mixed_params)
                print(f"已完成中央化Critic的参数迁移，迁移比例: {transfer_ratio:.4f}")
            else:
                print(f"源参数中找不到中央化Critic的参数，跳过参数迁移")
            
            print(f"中央化Critic参数迁移完成")
            return self
            
        except Exception as e:
            print(f"参数迁移过程中出错: {e}")
            import traceback
            traceback.print_exc()
            return self
            
    def get_parameters(self):
        """获取MASACController的参数
        
        在CTDE范式下，此方法返回中央化Critic的参数，用于保存或迁移。
        
        Returns:
            包含模型参数的字典
        """
        params = {}
        
        if hasattr(self, 'centralized_critic') and self.centralized_critic is not None:
            # 获取中央化critic的参数
            params['centralized_critic'] = {}
            if hasattr(self.centralized_critic, 'critic_net'):
                params['centralized_critic'] = self.centralized_critic.critic_net.state_dict()
        
        return params 