---
name: gpu-kernel-optimization-workflow
description: Coordinate a GPU kernel optimization study across platform porting, workload design, experience, cost modeling, specialization, actual tuning, independent evaluation, and ablations. Use for end-to-end planning, implementation, execution, or audit across these stages.
---

# GPU Kernel 优化总控

**一个主模型**负责读项目约定、拆分任务、汇合证据和作阶段决策。可让同一模型配置的子 agent 独立研究、诊断、实现或审阅。主模型也可以按需逐个调用下列子 Skill。

## 执行顺序与子 Skill

| 阶段 | 子 Skill | 交给下阶段的内容 |
|---|---|---|
| 新设备或加速器移植 | [平台适配](../gpu-kernel-platform-porting/SKILL.md) | 本机能力、真实派发、通用实现与基线。特化及验收按本平台重新建立 |
| 契约、负载与基线 | [负载设计](../gpu-kernel-workload-design/SKILL.md) | 可重建结构、独立参考、实际共同起点和强基线 |
| 获取或沉淀机制 | [专家经验](../gpu-kernel-experience/SKILL.md) | 可见快照、适用条件、反例与复用候选 |
| 预测、诊断与搜索次序 | [硬件成本模型](../gpu-kernel-cost-model/SKILL.md) | 绑定真实分支的执行图、支持域、未知项和选择依据 |
| O 代码和分派设计 | [核函数特化](../gpu-kernel-shape-specialization/SKILL.md) | 守卫和后备、候选源码与合法池、具体实测矩阵 |
| O 搜索与 T 最终选参 | [实测调参](../gpu-kernel-autotuning/SKILL.md) | 逐版本试验、O 独立确认、统一入口 T 配对和冻结参数 |
| 设备与可恢复执行 | [实验执行](../gpu-kernel-experiment-execution/SKILL.md) | 设备队列、批次、真实收据与断点 |
| 正确性、计时和 A | [实测验收](../gpu-kernel-evaluation/SKILL.md) | 同边界计时、冻结方案的独立逐目标结果 |
| 实验控制与成本 | [消融研究](../gpu-kernel-ablation-study/SKILL.md) | 匹配的比较组、结论范围、可审计的耗时与用量 |

可按任务直接调用子 Skill。若对整个流程有授权，先登记比较组和预算，再冻结负载。共同起点和真实基线通过后开展 O。O 每轮把特化设计、实际搜索和独立复测连起来，局部接纳后回归统一入口。T 在统一入口上给已冻结的代码和分派选参。A 在所有方法冻结后独立执行。模型、经验、剪枝根据研究问题有条件启用。

目标平台变化时，在开展专用分支 O 前完成本平台适配的 P0–P2。原架构可提供契约、结构、机制假设和参考，目标机的合法配置、真实基线、模型及性能证据分别重新绑定。用 P3–P5 的收据界定特化和整套支持范围。

## 主模型的阶段决策

- **明确本轮模式。** 设计交付方案和待补项，实现在当前项目里补工具和测试，实测遵循授权的设备、预算和可见证据，审计复核既有收据。用本课题独立的结构、模型、经验及工件目录。
- **区分不同证据。** 等待运行的候选、静态合法、正确、已测、独立确认和已接受分别登记。模型估算与子 agent 报告用于提出或排序方案。正确性和性能依据实际工具收据。
- **完成局部与整套检查。** O 只针对有证据的目标和域接纳。回归覆盖守卫冲突、共享依赖与后备。统一入口 T 在同一批输入上配对 O 保留参数与 T 选中参数。算法或分派改变回到相应的 O。
- **保持可恢复。** 待测批次 CPU 准备完成再申请设备。故障后从已核对的收据恢复。保留失败、超域、未测和后备，按登记目标报告总分母。
- **保护独立验收。** A 的数值输入、参考和结果按权限隔离。据 A 修改方案时登记新的候选和新验收。同源样本的泛化结论另验证结构或来源留出。

[阶段交接契约](references/handoff-contract.md)给出稳定标识、证据边界和放行条件。保留各 Skill 的同级目录布局时，总控链接可以直接打开各子 Skill 及其手册。
