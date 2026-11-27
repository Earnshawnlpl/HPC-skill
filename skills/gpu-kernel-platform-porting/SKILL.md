---
name: gpu-kernel-platform-porting
description: Port a GPU kernel or custom operator to a new GPU or accelerator. Use for target SDK and device qualification, a portable contract and backend adapter, native or compatibility paths, per-platform specialization, and real-device correctness and acceptance.
---

# GPU 与加速器平台适配

每个目标平台都有自己的设备、工具链、后端和验证身份。主模型负责按证据推进，必要时委派同一模型配置的子 agent 核对官方资料、实现平台适配或审阅。先确定本轮是设计、实现还是授权实测。

## 目标与交付

登记目标型号及架构、可访问的计算节点、实际 SDK/驱动/固件、编译环境、原生或兼容路径、算子契约和验收指标。交付可重现的构建、调用和验证入口，记录实际命中的设备与分支、覆盖范围和待补证据。

## 放行顺序

| 门控 | 条件与产物 |
|---|---|
| P0 环境 | 核实目标型号对应文档、SDK、设备身份及能力清单 |
| P1 设备探针 | 编译、加载、提交和同步后，取得目标设备执行凭据及与独立参考一致的输出 |
| P2 通用入口 | 用目标后端跑通完整算子及支持域，采集本机基线、合法后备和正确性 |
| P3 特化与 O | 按实机能力构造分支，逐源码版本验证、实调及独立确认 |
| P4 统一入口与 T | 回归分派和共享开销，在同入口、同输入配对保留参数与新参数，再冻结 |
| P5 目标机 A | 独立配对验收，给出实机功能、合格加速和可支持范围 |

任一门缺证据时标明状态，按实际授权继续独立的设计、构建或测试。设备库缺失时可接通同语义的通用设备路径，记录可比基线的覆盖。主机参考或模拟的结果按其实际范围报告。

- [平台档案、后端接口与放行细则](references/platform-bringup.md)用于从未知平台跑通最小设备探针、通用入口并确定可宣称范围。
- [变长 Grouped GEMM 的移植示例](references/grouped-gemm-porting.md)用于复用逻辑任务映射，再按设备能力实现新特化和重新测量。
