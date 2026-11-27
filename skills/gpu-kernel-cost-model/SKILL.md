---
name: gpu-kernel-cost-model
description: Build or apply hardware cost models for the GPU kernel branch that actually runs. Use for branch execution graphs, calibration, bottleneck diagnosis, candidate ranking, and validated reversible pruning.
---

# GPU Kernel 分支成本建模

分析目标结构实际命中的路径和配置，输出可以核证的成本、资源限制和竞争假设。适用于诊断已测实现、预测候选或审计模型剪枝。成本模型与负责工作流的主模型各有证据。可委派同一模型配置的子 agent 核对工具采集、执行图或校准域。

## 操作步骤

1. 绑定目标、源码、真实配置、后端、设备、计时范围及模型。分开记录源代码、编译器、实际 Profile 和时延来源。
2. 绘制当前执行图，含 CTA 分解、有效计算与掩码、访存、寄存器/共享内存、同步、阶段依赖、真实主机与辅助成本。共享组件各自匹配工作单位和约束。
3. 校准需要的新组件或复核已登记的相容证据。用独立样本核对覆盖、误差与排序。记录未知项、超域以及适合区分竞争解释的下一测量。
4. 候选实测前保存预测和使用依据。缺独立准入时仅建议优先级。正式暂缓依赖已登记的环境与支持域、风险、探索、审计和撤销规则。实际少测与全池影子判断分别登记。
5. 移交能够验证的机制、原合法池、待测配置和优先级。之后用同范围观测更新诊断。用于接纳的时延和正确性来自真实运行。

[分支成本、校准和剪枝手册](references/cost-model-and-pruning.md)描述证据来源、资源与时延的关系、联合服务、准入门槛和审计。输出绑定执行图、支持域、预测及决策收据。
