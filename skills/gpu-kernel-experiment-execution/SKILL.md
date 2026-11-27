---
name: gpu-kernel-experiment-execution
description: Prepare, schedule, resume, and audit executable GPU kernel experiment batches. Use for runner setup, single-model task coordination, shared GPU scheduling, attempt receipts, failure handling, and checkpoints.
---

# GPU Kernel 实验执行与恢复

用于建立最小运行链路、调度已就绪批次或核对断点。一个主模型决定任务、可见材料、修改边界、测量时机和阶段推进。可由同一模型配置的子 agent 完成互相独立的分析、候选或审阅。

## 推进方式

1. 取得可复现输入、独立参考、基线和实际调用入口。找不到可用 runner 时先做一个合法代表目标的调用、同步、正确性和计时。设计任务可以交付实际入口和验证计划。
2. 在本课题数据目录保存来源、编译、测量、环境和阶段工件。用逻辑任务身份和各次物理尝试连接记录。CPU 准备输入、代码、合法池和可退出的批次。
3. GPU 操作就绪后才预约，按实测需要估时。遵循项目的物理卡与占用规则。取锁后核对设备、预约和实际进程。与基线配对测量时对齐设备和计时范围。
4. 写入原始收据，等待同步和设备进程退出后释放预约。已有可信终态的工作从恢复清单跳过。发生派发前拒绝、错误、重测或不明状态时保留分别可审计的证据。
5. 为主模型提供当前阶段、实际测量、剩余任务、等待原因、追加资源和总耗时。设计模式与实际设备执行各自遵守用户权限。

[执行与实验收据手册](references/execution-and-records.md)包含本地或集群入口、GPU 协作、续跑状态和字段示例。交付运行入口与实际收据，具体数值接受边界使用本项目的测量政策。
