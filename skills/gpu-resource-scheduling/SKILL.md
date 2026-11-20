---
name: gpu-resource-scheduling
description: Coordinate short GPU reservations for shared experiments, wait for active or overlapping requests, verify the selected device, and release the reservation after execution.
---

# GPU 资源登记与排队

在任何需要使用共享 GPU 的调参、Profile、校准或验收任务前使用本 Skill。任务必须先在 5090 的共享目录登记目标 GPU、用途和预计时长，再等待当前占用、较早登记或设备上的外部计算进程结束。登记不自动切换 GPU，也不替代项目已有的 GPU 执行队列。

## 登记规则

- 明确传入物理 GPU UUID、任务标识、用途和 `estimated_duration_s`。不根据空闲程度自动选择 GPU。
- 默认单次预约最长 30 分钟。超过该时长的任务拆成多个可恢复阶段或显式调整政策，不能静默持有长租约。
- 预约按同一 GPU 的登记时间排队。较早的等待者优先，活动租约、未完成的较早预约和设备上的外部计算进程都会阻塞新的登记。
- 获得租约后再次核对 UUID、计算利用率和计算进程。复核失败时撤销本次激活并回到等待状态。
- 推荐使用脚本的 `run` 子命令，它会在子进程运行期间定期心跳，并在成功、失败、中断或异常退出时释放租约。手动 `acquire` 必须在 `finally` 中调用 `release`。
- 预计时长到期不会把仍在运行的任务强行释放。租约进入 `overdue`，继续阻塞后续任务，直到任务主动释放或看守程序确认持有者已经退出。
- 所有登记、等待、激活、心跳、释放、回收和错误事件写入 `/data03/liupl` 下的登记根目录。状态输出不得包含租约 token。

## 推荐入口

在 5090 上运行 [gpu_reservation.py](scripts/gpu_reservation.py)。常用形式如下，其中 `--` 后是实际 GPU 命令。

```bash
python gpu_reservation.py run \
  --root /data03/liupl/code-opt/gpu-reservations \
  --gpu-uuid GPU-ca94ba5a-1a2f-8928-b793-076fe9f6e0a4 \
  --task-id moe150-calibration-01 \
  --purpose calibration \
  --estimate-seconds 1800 \
  -- python -B run_task.py
```

查询和人工释放使用 `status`、`acquire`、`heartbeat`、`release` 与 `cancel` 子命令。取消仍在等待的请求使用 `cancel`，释放已激活的租约使用 `release`。实际任务仍需设置 `SKOPA_GPU_UUID` 和 `CUDA_VISIBLE_DEVICES`，并通过现有 `GPUExecutionQueue` 获取设备执行租约。

读取[登记记录与恢复规范](references/record-schema.md)来审查状态、过期回收和成本字段。读取[接入指南](references/integration.md)来把登记接入优化控制器、DeepSeek worker 和 GPU 队列。
