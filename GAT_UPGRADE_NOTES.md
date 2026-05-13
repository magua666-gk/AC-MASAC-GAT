# GAT upgrade notes

本版本将原先的普通多头注意力主线升级为图注意力机制（GAT 风格），核心改动如下：

1. 在 `masac_adapter/masac_adapter.py` 中新增 `GraphAttentionLayer`、`build_spatial_adjacency`、`build_valid_adjacency`。
2. Actor 不再只做 Query-Key-Value 的全连接注意力，而是将 `Leader/Follower` 组成动态图节点：
   - Leader Actor：节点为 `Leader + Followers`，取 Leader 节点的 GAT 聚合结果作为图上下文。
   - Follower Actor：节点为 `当前Follower + Leader + 其他Followers`，取当前 Follower 节点的 GAT 聚合结果作为图上下文。
3. Critic 使用同一批智能体节点构图，分别通过 Leader-GAT 与 Follower-GAT 聚合上下文，再进入双 Q 头。
4. 图边由观测前两维的归一化空间位置构造，同时受有效节点 mask 约束；默认距离阈值为 `0.45`，并在图过稀时自动回退为有效节点连接，避免训练早期信息断开。
5. 保留 `MultiHeadAttention` 兼容旧入口，但主线 Actor/Critic 已改为 `GraphAttentionLayer`。
6. 同步修改了 `integrated_ablation_modules/masac_no_curriculum` 中对应的 GAT 实现，避免消融入口仍旧使用普通注意力。

注意：由于注意力层参数结构已改变，旧 checkpoint 的普通注意力参数不能完整复用。代码默认 `strict=False` 加载模型时会跳过不匹配的注意力参数；建议重新训练或至少重新微调。
