---
name: gpu-kernel-evaluation
description: Evaluate GPU kernels with contract-based correctness, fair paired timing, and independent acceptance. Use to register or audit measurement scope, compare implementations, validate frozen configurations, or report suite results.
---

# GPU Kernel 正确性与独立验收

取得算子契约、独立参考、目标实例、参与方法、真实入口和配置、物理设备、计时范围和统计政策。一个主模型核对实际收据，可用同一模型配置的子 agent 分开审阅参考、计时和汇总。区分 O 独立确认、T 参数确认、A 正式验收和证据审计。

仅获设计授权时，登记验证和配对协议，标记待测及缺失证据。实际运行按项目授权执行。

## 实测方法

1. 核对实际调用的目标、路由、源码、配置、输入和设备。用独立参考检查完整输出、容差、异常值、辅助输出和输入副作用，并覆盖边界与后备。
2. 固定冷或热、状态还原、workspace、预热、编译和同步政策。完整入口涵盖真实规划、分派、转换、分配、核函数和合并。设备段和 Graph 时延单列。
3. 在同一实例和物理设备上按登记的次序交错比较，保留原始样本、配对块、实际顺序、异常及补测凭据。
4. 按明确方向计算配对效果，保留不确定性。用固定分母汇总覆盖、正确性、效应和分层结果。重复计时和同源结构派生保留各自分组。

## 阶段边界与交付

- **O** 使用搜索外的确认块复测候选和当前保留实现。
- **T** 在同一统一入口和同批确认输入上配对 O 保留参数与 T 选中参数，复核配置组和分派。
- **A** 在参与方法和基线冻结后，从独立评估范围取得输入和参考，报告每项目实际命中及完整调用。基于 A 结果的开发另行登记身份及验收。

交付技术绑定、参考检查、原始计时、逐目标结果、统计口径与分母。失败、未知、后备和待测有各自的状态与凭据。[实测、最终调参与独立验收手册](references/measurement-and-acceptance.md)展开测量、确认和报告。
