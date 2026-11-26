# 登记记录与恢复规范

登记根目录默认是 `/data03/liupl/code-opt/gpu-reservations`，可通过 `--root` 指定。每个 GPU 使用 UUID 的规范化值计算目录名。目录内使用一个 OS 文件锁保护状态迁移，记录写入采用临时文件、`fsync` 和原子替换。

## 生命周期

```text
pending → active → released
                 ↘ overdue → released
                 ↘ abandoned
```

`pending` 表示已经登记但尚未占用设备。只有登记时间最早、没有活动租约且设备复核通过的请求才能进入 `active`。等待者主动退出时使用 `cancel`。持有进程死亡且心跳超过回收阈值时，控制器把 `active` 迁移到 `abandoned`，把无人维护的 `pending` 迁移到 `canceled`，新请求才能继续。`overdue` 只表示预计结束时间已过，仍然阻塞同一 GPU。

## 关键字段

| 字段 | 说明 |
| --- | --- |
| `reservation_id` | 随机生成的登记身份 |
| `task_id` | 实验或控制器任务身份 |
| `owner_name` | 登记负责人姓名 |
| `gpu_uuid` | 物理 UUID，必须和设备查询结果一致 |
| `purpose` | `correctness`、`tuning`、`profile`、`replay`、`integration`、`acceptance` 或 `calibration` |
| `estimated_duration_s` | 本次登记的预计时长 |
| `requested_at` / `activated_at` / `released_at` | UTC epoch 秒 |
| `expected_end_at` | 已确认的日历截止时间，使用本地租约时为激活时间加预计时长 |
| `heartbeat_at` | 最近一次持有者心跳 |
| `owner_pid` / `owner_start_ticks` | 用于防止 PID 重用的持有者身份 |
| `state` | 当前生命周期状态 |
| `device_preflight` | 激活前的 UUID、利用率和计算进程检查 |
| `last_wait_reason` | 最近等待原因，例如 `prior_lease`、`device_busy`、`pulse_conflict` |
| `wait_reason_since` / `last_wait_at` | 当前等待原因首次记录及最近复核的 UTC epoch 秒 |
| `exit_code` / `actual_duration_s` | 释放时的执行结果和实际时长 |
| `events` | 状态变化的不可变事件文件引用 |

手动 `acquire` 将 token 交给调用方，`run` 在内部使用 token。记录中仅保存 `token_sha256`，公开状态和报告使用登记 ID。

启用 GPU Pulse 后，`pulse/<reservation_id>.json` 保存提交意图、周期和已确认的 group ID。等待、确认和取消用同一请求的客户端 ID。客户端 ID 仅放在 `pulse-private/<reservation_id>/client-id`，远端文件权限为 `0600`。公开收据使用 group ID 和状态，无需读取客户端 ID。

已核对的冲突拒绝或已清理的时段会记录在 `pulse/attempts/<reservation_id>/`，随后才能重新计算可用窗口并提交。无法确认创建或清理时，保留最近的提交意图和本地状态供恢复。

## 恢复与成本

分别记录 CPU 准备、预约排队、设备锁等待、持有租约和 GPU 执行的耗时，以及外部进程阻塞次数、回收次数和退出码。每次重试保留对应租约和收据。中断、设备复核失败、持有者异常退出和超时都保留原记录，恢复时创建新的激活事件并沿用原 `task_id`。
