---
name: gpu-kernel-ablation-study
description: Design, audit, and report controlled ablations and optimization costs for GPU kernel studies. Use for model and experience comparisons, structural retrieval, pruning, or token, GPU, and elapsed-time accounting.
---

# GPU Kernel 消融与成本分析

按研究主张确定要切换的变量、共同目标和成本口径。一个主模型冻结协议并解释证据，可用同一模型配置的子 agent 分别核对比较组、收据和成本边界。

## 工作步骤

1. 取得目标清单、主比较、候选流程、经验快照与可见反馈。分别登记模型指导、跨目标经验、结构检索或剪枝要回答的问题。
2. 固定共同起点、模型配置和编辑权、可见数据、Profile、真实合法池、机会或资源、停止条件、设备及计时范围。各组局部反馈写回本组。
3. 主比较覆盖全登记套件。消融可取事先登记的共同分层子集，覆盖几何、来源、边界和模型支持域。扩样按结构规则登记新增项、分层和权重。
4. 完成各组实际优化与配置确认后冻结，在同一 A 实例上配对。保留全部登记项并区分失败、待测和后备。分别报告性能、首次达标和成本。
5. 核对总墙时、阶段尝试、已记活动、设备等待与占用、编译、Profile、实调和复测。按请求或会话去重输入、输出、缓存和重试，子 agent 费用归到对应组与目标。
6. 单列建库、校准和公共基线，说明目标增量及摊销。剪枝保留合法池、实际少测、影子审计、撤回和选择损失，净节省依据同范围全池实测及实际开销。

交付逐组协议及绑定、配对性能和成本、固定分母、负载层次和可回查收据。未执行的环节标记计划或待测。缺用量时保留未知。[比较组、消融与成本手册](references/ablation-and-cost.md)给出各控制因素和报告边界。
