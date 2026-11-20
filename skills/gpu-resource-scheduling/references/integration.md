# 接入指南

## 控制器

控制器在创建 GPU callback 前调用登记脚本或等价 Python API，绑定 `task_id`、目标 shape、版本、用途和预计时长。登记返回 `reservation_id` 后才允许把任务提交给现有 GPU 队列。callback 完成后在 `finally` 中释放登记，并把 reservation 收据链接到测量、Profile、编译和 DeepSeek 角色账本。

## DeepSeek worker

Researcher、Analyst、Optimizer 和 Reviewer 默认不需要 GPU，不得因为编辑或分析任务创建 GPU 登记。只有编译、正确性、调参、Profile 和完整入口复测阶段才登记。多个 worker 的代码修改可以并行，GPU callback 必须通过同一 GPU 登记和现有串行队列。

## 预计时间

预计时长来自控制器对本阶段的粗略估计，应该覆盖编译、预热、测量、清理和收据写入。估计过大时拆分成可恢复的小阶段，并在每阶段结束后释放租约。估计不足进入 `overdue` 后继续持有，不允许让后续请求误以为设备已空闲。

## 与现有队列的关系

登记解决跨项目和跨控制器的排队。`GPUExecutionQueue` 仍负责项目内部文件锁、设备 UUID 复核、外部 PID 检查、CUDA lease 和清理。两者都通过后才能运行。登记脚本故障、设备查询失败或记录身份不一致时拒绝开始 GPU 工作。
