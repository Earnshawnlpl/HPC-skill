---
name: gpu-kernel-optimization-workflow
description: Design or run a reproducible GPU kernel optimization study under one controlling model. Use for workload and baseline construction, shape specialization, GPU measurement and tuning, hardware cost modeling, independent acceptance, or ablations.
---

# GPU Kernel 优化工作流

面向独立课题的算子优化与实验工作流。**一个主模型**负责从契约到验收，也可以用同一模型配置的子 agent 分担研究、诊断、实现和审阅。主模型维护阶段状态、合并候选和核对实际收据。

先读项目约定，确定本次交付是方案、实现、实测还是审计。登记算子语义、数值要求、数据目录、真实基线、设备、计时范围和可用资源。使用新课题独立的负载、经验、模型与运行记录。复用外来源码或机制时登记来源、许可证和适用范围。缺少实测入口时先接通最小测试，报告已有证据与待验证项。

## 阶段与交接

| 阶段 | 做什么 | 推进前确认 |
|---|---|---|
| 契约与工作负载 | 构建真实来源、受控配对和边界样本 | 拟比较的语义、目标分母、结构范围、阶段实例和验证政策已登记 |
| 共同起点与强基线 | 跑通独立参考、实际库入口和基线调参 | 有正确性、设备和真实计时收据 |
| O 诊断与特化 | 分析实际分支，设计守卫，生成竞争路线并逐版本实调 | 同目标独立复测支持接纳、保留或回退，原始候选与失败可追溯 |
| 集成与 T | 接入统一入口，验证交叉影响，冻结算法后重新选参 | 统一入口可运行，最终配置完成独立确认，已登记的留用规则已应用 |
| A 验收 | 冻结所有方法，在隔离输入上做同范围配对实测 | 逐负载的正确性、加速、后备、缺失和验收分母都有凭据 |
| 报告 | 汇总主对照、消融、模型覆盖和总成本 | 结果能追到输入、源码、配置、环境和测量收据 |

## 按任务读取

- [负载、参考与基线](references/workload-and-baselines.md)覆盖契约登记、真实与合成结构、覆盖检查和基线接入。
- **第一次做特化必读** [特化实现与统一分派](references/shape-specialization.md)。从判断价值开始，落实守卫、机制选择、候选实验矩阵、代码修改、局部接纳和回归。
- [分支成本、经验与搜索](references/optimization-and-search.md)覆盖成本图、校准、经验检索、可撤销剪枝和有限池搜索。
- [执行、子 agent 与收据](references/execution-and-records.md)覆盖编排、测试入口、设备预约、阶段交接字段和恢复。
- [计时、最终调参与独立验收](references/acceptance-and-cost.md)覆盖正确性、对称计时、O/T 可比复测和正式 A 报告。
- [实验比较、消融与成本](references/ablation-and-cost.md)覆盖各组设置、扩样、token、GPU 与端到端墙时。

## 贯穿各阶段的检查

- 区分结构、数值实例、实现分支和参数配置。相同外层维度可以命中不同分支。接纳依据绑定当前目标和真实完整调用。
- 经验、预测和 Profile 各自服务于候选设计。每个有资格接纳的新版本实际验证和调参。收益以相同输入、物理设备和计时范围的独立配对证据为准。
- 先在 CPU 准备下一批 GPU 操作。就绪后登记预计占卡时间，在设备锁内核对物理设备和占用，批次结束后同步、保存并释放。[GPU 资源登记](../gpu-resource-scheduling/SKILL.md)是本仓库提供的一个可选调度入口。
- 失败、超出模型支持域、未测和合法后备各自登记。按终态收据恢复，发布时保留固定的目标分母。A 的输入、答案和反馈由独立评估步骤管理。
