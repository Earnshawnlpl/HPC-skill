---
name: gpu-kernel-autotuning
description: Tune actual GPU kernel candidates and frozen complete-entry paths. Use for finite configuration pools, measured selection, independent O/T confirmation, actual parameter binding, or resumable tuning.
---

# GPU Kernel 实测调参

O 为每个候选源码版本实际选参。T 给冻结的统一入口选参。A 使用最终冻结方案进行独立评估。一个主模型确认搜索及接纳政策。可用同一模型配置的子 agent 核对参数空间、失败和搜索证据。

## 开始前

取得目标结构、数学参考、资源规则、实际调用边界、当前保留方案和允许的试验。为改过的源码登记新绑定、参数生效方式和有限池。只有设计授权时，交付搜索矩阵、合法规则、配对方案和缺项。实际耗时、正确性或性能结论由真实收据支持。

| 阶段 | 动作 | 输出 |
|---|---|---|
| O 搜索 | 按政策测本版代码允许的配置并检查正确性及资源 | 各项状态、实际参数、选中理由 |
| O 确认 | 用独立块配对当前最佳与选中实现，检查真实调用 | 独立确认收据、局部接纳建议与待补证据 |
| T 搜索与确认 | 用统一入口搜索并直接配对 O 保留参数与 T 选中参数 | 逐路由配置、差异、联合检查 |
| A 交接 | 锁定统一入口、参数、实际依赖与可比基线 | 可独立验收的正式提交 |

[搜索与最终选参手册](references/search-and-finalization.md)描述状态、接纳和续跑。搜索政策完成、全池穷尽、正确性和性能达标分别报告。
