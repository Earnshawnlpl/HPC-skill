# 接入指南

## 控制器

1. **CPU 准备。** 在预约前完成负载与候选生成、输入摘要校验、静态验证和合法配置池确定。核验目标 shape、版本、设备绑定、已完成收据及续跑位置，形成可直接执行的就绪批次。等待 LLM 交付和分析也放在预约之外。若原入口混合了 CPU 准备和 GPU 执行，先让控制器拆出准备阶段，再让设备入口读取并复核该批次的身份及续跑位置。
2. **预约与执行。** 用 `run` 为就绪批次登记 `task_id`、用途和预计时长。获得预约并通过设备复核后，经 `GPUExecutionQueue` 执行本批次。设备占用或日历冲突时等待，再复查。
3. **保存与释放。** 停止派发后同步设备，原子保存结果、检查点、测量及失败收据，等待 GPU 子进程和其设备 worker 退出。让 `run` 随即结束本地租约并取消本次 Pulse 登记。按预约 ID 将收据关联到测量、Profile、编译和阶段账本。需要后续 CPU 分析或新一轮编辑时，先归还设备，再按新就绪批次预约。恢复时跳过已有终态收据的配置。上次派发结果不明确时，先核对收据和实际进程状态。

## DeepSeek worker

Researcher、Analyst、Optimizer 和 Reviewer 可以并行完成编辑和分析。这些 CPU 工作无需预约。涉及设备执行的编译取证、正确性、调参、Profile 和完整入口复测由受信控制器为对应批次登记，交串行队列处理。

## 预计时间

按就绪批次估计设备编译、预热、测量、同步、退出及收据写入时间，单次最多 1 小时。需要更久时先拆成可恢复批次，并在每批次结束后按进度释放。预计操作放不进剩余时段时，停止新派发并保存续跑点。已经开始的操作在安全边界退出，超时按 `overdue` 如实保留记录。

## GPU Pulse 日历

对提供 GPU Pulse 的环境，固定设备用 `--gpu-uuid` 和同一物理 UUID 的 `--pulse-gpu-id`，另传入部署方给出的 `--pulse-url`。公开标题可用 `--pulse-title 5090` 指定，缺省沿用 `--task-id`。项目内部的真实任务 ID 仍由 `--task-id` 决定。无日历的环境可使用共享目录登记。

独立登记为设备可选的开发任务，可以改用两项或多项 `--eligible-gpu-uuid` 和 `--pulse-url`。按登记顺序尝试各设备的本地 FIFO、日历和设备检查，选中后清理本任务在其余设备的等待记录。启动后将实际选中 UUID 交给子进程。已经绑定设备的模型、校准和正式实验沿用单 UUID 入口。

本地 FIFO 取得优先权且设备就绪后，脚本读取所选 UUID 的日历时段，将登记窗口控制在 1 小时内，检查重叠，并以 `force=false` 提交。`--pulse-minimum-seconds` 允许使用当前到下一已登记时段之间的空档，前提是空档达到预先声明的最短时长。当前时段被占用时等待后再查。启用 `--pulse-wait` 或设备可选模式后，日历冲突、明确的 HTTP 409 和可信设备占用会默认等 30 秒后再查，本地仅保留等待登记。请求结果不明时按原身份核对创建状态，确认是本次创建的日历项再尝试清理。结果未确认时保留记录并停止，避免重复提交。提交成功后核对 group ID、GPU、起止时间和 `mine` 身份，并复核设备就绪。

`run` 向子进程传入 `SKOPA_GPU_RESERVATION_END_MS`，值为预约截止的 UTC epoch 毫秒。驱动应在每个可恢复的设备操作之前比较预计耗时和剩余时段，提前停止派发并留出同步和清理时间。后续阶段在新预约中恢复。正在执行的测量可安全退出后再释放。执行跨过预约结束时间时，本地状态保持 `overdue`，而日历原时段已结束，报告应保留这个区别。

受信驱动可在项目 GPU 锁内调用 `verify_active_reservation(root, gpu_uuid, reservation_id, minimum_remaining_seconds=...)`，或运行 `gpu_reservation.py check --root "$SKOPA_GPU_RESERVATION_ROOT" --gpu-uuid "$SKOPA_GPU_UUID" --reservation-id "$SKOPA_GPU_RESERVATION_ID" --minimum-remaining-seconds 90`。校验本地活动记录、相同的 Pulse 登记及日历冲突，成功时报告 `expected_end_ms` 和 `pulse_verified`。Python 调用可区分 `ReservationWindowEnding`、`ReservationUnavailable` 和 `gpu_pulse.PulseConflict`；无法确定的传输或数据错误停止新派发。该检查由受信控制器执行，CUDA 子进程和编辑 worker 只接收各自执行所需的信息。

批次退出、收据落盘后查询原登记时段，只取消同时匹配 group ID、GPU、时段、任务以及 `mine=true` 的本次登记。取消使用创建时的持久化客户端 ID。网络错误或归属不明时保留原记录和恢复证据，标记清理未完成。

## 与现有队列的关系

登记解决跨项目和跨控制器的排队。`GPUExecutionQueue` 仍负责项目内部文件锁、设备 UUID 复核、外部 PID 检查、CUDA lease 和清理。两者都通过后才能运行。登记脚本故障、设备查询失败或记录身份不一致时拒绝开始 GPU 工作。
