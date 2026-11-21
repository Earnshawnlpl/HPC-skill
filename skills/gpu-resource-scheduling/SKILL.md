---
name: gpu-resource-scheduling
description: Coordinate short GPU reservations for shared experiments, wait for active or overlapping requests, verify the selected device, and release the reservation after execution.
---

# GPU 资源登记与排队

在需要使用共享 GPU 的调参、Profile、校准或验收任务前使用本 Skill。先在执行服务器登记目标 GPU、用途和预计时长。服务器提供 GPU Pulse 日历时，同时校验并登记该日历上的时段，然后取得项目已有的 GPU 执行队列租约。

## 登记规则

- 明确传入物理 GPU UUID、任务标识、用途和 `estimated_duration_s`。不根据空闲程度自动选择 GPU。
- GPU Pulse 单次预约最长 1 小时。长任务分成可恢复阶段，按登记结束时间安排新设备工作并保留清理时间。
- 预约按同一 GPU 的登记时间排队。较早的等待者优先，活动租约、未完成的较早预约和设备上的外部计算进程都会阻塞新的登记。
- 获得租约前核对 UUID、计算利用率和计算进程。配置 GPU Pulse 时，还要确认日历中已存在起止时间匹配的本次登记。发生冲突或无法确认日历状态时停止激活。
- 推荐使用脚本的 `run` 子命令，它会在子进程运行期间定期心跳，并在成功、失败、中断或异常退出时释放租约。手动 `acquire` 必须在 `finally` 中调用 `release`。
- 预计时长到期不会把仍在运行的任务强行释放。租约进入 `overdue`，继续阻塞后续任务，直到任务主动释放或看守程序确认持有者已经退出。
- 所有登记、等待、激活、心跳、释放、回收和错误事件写入 `/data03/liupl` 下的登记根目录。状态输出不得包含租约 token。

## 推荐入口

在执行服务器上运行脚本。设置 GPU Pulse 时传入日历地址与快照中对应所选物理 UUID 的 GPU ID。`--` 后是实际 GPU 命令。[gpu_reservation.py](scripts/gpu_reservation.py)

```bash
python gpu_reservation.py run \
  --gpu-uuid "$SELECTED_GPU_UUID" \
  --pulse-url "$GPU_PULSE_URL" \
  --pulse-gpu-id "$PULSE_GPU_ID" \
  --owner-name "$RESERVATION_OWNER" \
  --task-id calibration-01 \
  --purpose calibration \
  --estimate-seconds 1800 \
  -- python -B run_task.py
```

查询和人工释放使用 `status`、`acquire`、`heartbeat`、`release` 与 `cancel` 子命令。取消仍在等待的请求使用 `cancel`，释放已激活的租约使用 `release`。实际任务仍需设置 `SKOPA_GPU_UUID` 和 `CUDA_VISIBLE_DEVICES`，并通过现有 `GPUExecutionQueue` 获取设备执行租约。

登记状态、过期回收和成本字段见[登记记录与恢复规范](references/record-schema.md)。GPU Pulse 配置、冲突处理与分段续跑见[接入指南](references/integration.md)。
