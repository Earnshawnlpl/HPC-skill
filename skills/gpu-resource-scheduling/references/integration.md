# 接入指南

## 控制器

控制器在创建 GPU callback 前调用登记脚本或等价 Python API，绑定 `task_id`、目标 shape、版本、用途和预计时长。登记返回 `reservation_id` 后才允许把任务提交给现有 GPU 队列。callback 完成后在 `finally` 中释放登记，并把 reservation 收据链接到测量、Profile、编译和 DeepSeek 角色账本。

## DeepSeek worker

Researcher、Analyst、Optimizer 和 Reviewer 默认不需要 GPU，不得因为编辑或分析任务创建 GPU 登记。只有编译、正确性、调参、Profile 和完整入口复测阶段才登记。多个 worker 的代码修改可以并行，GPU callback 必须通过同一 GPU 登记和现有串行队列。

## 预计时间

预计时长来自控制器对本阶段的粗略估计，应该覆盖编译、预热、测量、清理和收据写入。估计过大时拆分成可恢复的小阶段，并在每阶段结束后释放租约。估计不足进入 `overdue` 后继续持有，不允许让后续请求误以为设备已空闲。

## GPU Pulse 日历

对提供 GPU Pulse 的环境，给 `run` 或 `acquire` 同时传入 `--pulse-url` 和 `--pulse-gpu-id`。URL 使用部署方给出的 HTTP(S) 服务地址。Pulse 快照中该 GPU 的 `id` 必须匹配 `--gpu-uuid`。登记用户名和任务标题分别来自 `--owner-name` 和 `--task-id`。无日历的环境可使用共享目录登记。

本地 FIFO 取得优先权且设备就绪后，脚本读取所选 UUID 的日历时段，将登记窗口控制在 1 小时内，检查重叠，并以 `force=false` 提交。提交后，脚本查询并确认本次 group ID、GPU、起止时间和 `mine` 身份，复核设备就绪后才启动。已存在的同名登记也参与冲突检查。接口拒绝、响应异常或无法确认归属时停止启动。Pulse 日历的登记与执行队列的设备锁都要满足。

`run` 向子进程传入 `SKOPA_GPU_RESERVATION_END_MS`，值为预约截止的 UTC epoch 毫秒。驱动应在每个可恢复的设备操作之前比较预计耗时和剩余时段，提前停止派发并留出同步和清理时间。后续阶段在新预约中恢复。正在执行的测量可安全退出后再释放。执行跨过预约结束时间时，本地状态保持 `overdue`，而日历原时段已结束，报告应保留这个区别。

正常完成后查询原登记时段，只删除同时匹配 group ID、GPU、时段、任务以及 `mine=true` 的本次登记。网络错误保留登记和恢复证据。在能够确认同一 group 的归属前保持本阶段未启动或清理未完成。任务对自己创建的记录使用登记时的持久化客户端 ID，已有手工预约按日历中的原归属处理。

## 与现有队列的关系

登记解决跨项目和跨控制器的排队。`GPUExecutionQueue` 仍负责项目内部文件锁、设备 UUID 复核、外部 PID 检查、CUDA lease 和清理。两者都通过后才能运行。登记脚本故障、设备查询失败或记录身份不一致时拒绝开始 GPU 工作。
