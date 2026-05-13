# GAT 编队率增强版改动说明

本版本在上一版 `AC-MASAC-main_GAT优化版` 基础上增强图注意力机制，目标是在不改变环境、奖励函数和训练流程的前提下提升编队率。

## 主要改动

1. **新增 `build_formation_adjacency`**
   - 文件：`masac_adapter/masac_adapter.py`
   - 同步文件：`integrated_ablation_modules/masac_no_curriculum/masac_adapter.py`
   - 使用距离阈值 + k近邻 + Leader强制连边 + 自环兜底构造编队图。

2. **Actor/Critic 统一使用编队图**
   - Leader Actor / Critic：`leader_index=0`。
   - Follower Actor：`leader_index=1`，因为节点0是当前Follower，节点1是Leader。

3. **GAT 升级为 GATv2 风格动态注意力**
   - 由 `a_src^T Wh_i + a_dst^T Wh_j` 改为 `a^T LeakyReLU(W_src h_i + W_dst h_j)`。
   - 更适合根据当前智能体状态动态判断 Leader、邻近 Follower 的重要性。

4. **加入门控融合 gate**
   - Actor 和 Critic 都加入 `gate * GAT_context + (1-gate) * self_embedding`。
   - 避免图上下文过度覆盖自身控制信息，有利于提高编队稳定性。

5. **降低 GAT dropout**
   - 默认从 `0.1` 调为 `0.05`，减少编队保持时邻居信息抖动。

## 同步范围

已同步修改：

- `masac_adapter/`：课程学习主线
- `integrated_ablation_modules/masac_no_curriculum/`：无课程学习分支

未修改环境、奖励、经验池、训练入口。

## 注意

由于 GAT 层参数结构从经典 GAT 改成 GATv2，并新增 gate 层，旧模型 checkpoint 不能完整直接复用。建议重新训练，或只加载兼容参数后微调。
