---
name: gpu-resource-scheduling
description: Reserve a shared GPU only for a ready tuning, profiling, calibration, or validation batch, wait for an available slot, and release it when device work ends.
---

# GPU 资源登记与排队

在共享 GPU 上执行调参、Profile、校准或验收时使用本 Skill。先在 CPU 上准备并核验输入、代码和批次清单。核验通过后才进入预约队列，登记目标 GPU、用途和预计时长，并取得设备执行租约。

## 登记规则

- 固定设备实验明确传入物理 GPU UUID、任务标识、用途和预计时间。仅已独立登记为可选设备的开发任务允许从显式 UUID 列表选择。
- 按就绪批次估计需要占卡的时间，加上同步、写收据和退出的余量。1 小时是单次上限。估计超过上限时，先拆成可恢复的批次。
- 预约按同一 GPU 的登记时间排队。较早的等待者优先，活动租约、未完成的较早预约和设备上的外部计算进程都会阻塞新的登记。
- 获得租约前核对 UUID、计算利用率和计算进程，并核对匹配的 Pulse 登记。启用 `--pulse-wait` 后，已确认冲突或设备占用默认间隔 30 秒重查。接口返回不明时停止激活并保留记录。
- 用 `run` 启动已就绪批次。批次结束时先同步设备并保存结果、检查点和测量收据，再退出 GPU 进程并释放本次预约。手动 `acquire` 用 `finally` 处理清理。
- 预计时长到期不会把仍在运行的任务强行释放。租约进入 `overdue`，继续阻塞后续任务，直到任务主动释放或看守程序确认持有者已经退出。
- 所有登记、等待、激活、心跳、释放、回收和错误事件写入 `/data03/liupl` 下的登记根目录。状态输出不得包含租约 token。

## 推荐入口

在执行服务器上完成实验自己的 CPU 准备阶段后，再运行下方命令。设置 GPU Pulse 时传入日历地址与快照中对应所选物理 UUID 的 GPU ID。`--` 后的程序读取已核验的批次，执行本次需要 GPU 的工作。[gpu_reservation.py](scripts/gpu_reservation.py)

```bash
# 先完成 CPU 准备与核验，再将本批预计占卡秒数写入 GPU_BATCH_ESTIMATE_SECONDS。
python gpu_reservation.py run \
  --gpu-uuid "$SELECTED_GPU_UUID" \
  --pulse-url "$GPU_PULSE_URL" \
  --pulse-gpu-id "$PULSE_GPU_ID" \
  --pulse-title 5090 \
  --pulse-wait \
  --owner-name "$RESERVATION_OWNER" \
  --task-id "$READY_BATCH_ID" \
  --purpose tuning \
  --estimate-seconds "$GPU_BATCH_ESTIMATE_SECONDS" \
  -- python -B run_task.py
```

查询和人工释放使用 `status`、`acquire`、`heartbeat`、`release` 与 `cancel` 子命令。取消仍在等待的请求使用 `cancel`，释放已激活的租约使用 `release`。实际任务仍需设置 `SKOPA_GPU_UUID` 和 `CUDA_VISIBLE_DEVICES`，并通过现有 `GPUExecutionQueue` 获取设备执行租约。

阶段驱动按需接入 `check`，在设备锁内、提交实际回调前复核自己的预约、日历冲突和剩余时间。参数和返回语义见[接入指南](references/integration.md)。

登记状态、过期回收和成本字段见[登记记录与恢复规范](references/record-schema.md)。GPU Pulse 配置、冲突处理与分段续跑见[接入指南](references/integration.md)。
