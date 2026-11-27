---
name: gpu-kernel-experience
description: Extract, validate, retrieve, and audit reusable implementation experience for GPU kernels. Use for mechanism cards, verified source reuse, structural retrieval, failure learning, and isolated experience libraries.
---

# GPU Kernel 专家经验与机制库

用于把已确认的机制和反例沉淀为能检索的规则。主模型可以亲自查证来源，也可以用同一模型配置的子 agent 分开做技术研究、失败分析和复核。

## 录入与使用

1. 核对来源、版本、许可证和语义。用结构、当前分支、约束、改动、参数域、观测、反例和开销形成经验卡。
2. 区分机制假设、静态或编译验证、独立实测及准入。未完成的证据保留对应状态。经验卡要能说明何时适用、如何在当前目标推翻它。
3. 开始优化前绑定本方法可见快照。按结构、限制、语义和成本范围找候选，记录入选、排除和待查项。
4. 复用代码或变换后，为当前目标建立新绑定并实际调参。测得的来源收益留在原范围，目标劣化与副作用另记。
5. 晋升或修改规则时保留反例、实验和资格。比较组仅共享事先登记的快照。局部失败与待测项写入本方法账本。

[经验卡、检索与版本证据](references/mechanism-library.md)有可填写字段和分支处理。交付检索轨迹、机制候选、反例和后续验证办法。
