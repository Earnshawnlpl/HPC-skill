# 主 Skill 与子 Skill 的阶段交接

主模型维护一份项目账本，子 Skill 提供本职责的输入、动作和证据。主模型核对身份及收据后更新阶段。可选子 agent 只消费该任务可见的材料。

## 版本与身份

| 标识 | 绑定内容 |
|---|---|
| 实验与方法 | 课题、比较组、权限、公共快照、预算或停止政策、实际起点 |
| 平台实现 | 目标设备与 ISA、工具链、编译/执行路径、能力与支持域、P0–P5 收据 |
| 目标结构 | 算子契约、维度及有序元数据、来源谱系与重建依据 |
| 阶段实例 | O 搜索、O 复测、T 搜索、T 复测、A 中的阶段、工作结构、数值配方和审批可见域 |
| 源码与配置 | 父版本、实际调用入口、依赖与编译状态、守卫、合法池、参数和运行时行为 |
| 执行与测量 | 物理设备、测量和统计政策、独立尝试 ID、起止、路径、状态及原始收据 |

配置与执行路径变化时新建相应的绑定。试验代码、守卫和合法参数域变更时重新划分确认范围。维度相同的不同有序结构各有结构身份。

## 阶段放行条件

| 从哪里交接 | 放行条件与产物 | 受领子 Skill |
|---|---|---|
| 设计去构建 | 语义、候选覆盖、阶段数据与比较边界有登记，待定政策显式列出 | 负载设计、实验执行 |
| 新平台去优化 | 已确认真实设备派发，本机通用入口正确且有可比基线。记录支持域与未确认能力 | 平台适配、核函数特化 |
| 构建去优化 | 输入可重建，独立参考和共同起点可调用，实际基线有收据 | 核函数特化、专家经验、成本模型 |
| 特化去搜索 | 提供版本与结构、预期真实路径、安全域、有限合法池、参考与配对口径 | 实测调参、实验执行、实测验收 |
| 搜索去接纳 | 选中参数有实际试验与独立完整入口确认，瓶颈假设与观测分别保留 | 总控、核函数特化 |
| O 去 T | 通过集成回归，保存冻结的代码和守卫、后备及 O 参数表 | 实测调参 |
| T 去 A | 交付唯一的冻结入口、整套参数、受影响路径的联合确认和可比基线 | 实测验收 |
| A 去结论 | 全目标的正确性、失败、后备、可比范围和隔离输入可追溯 | 消融研究、总控 |

关键入口或判据缺失时，受影响的阶段留待补证据，其余可执行事项按授权推进。保留实际测量与推测之间的界线。

## 可使用的最小记录

```yaml
experiment:
  run_id: workflow-example
  method_id: full-system
trial:
  target_id: ordered-structure
  stage: O-confirm
  candidate: source-and-dependency-reference
  prior_choice: registered-incumbent-reference
  configuration: actual-configuration-reference
  instance: stage-bound-input-reference
  environment: physical-device-and-toolchain-reference
  timing_scope: complete-entry
  status: pending
  receipts: []
decision:
  policy: registered-rule-reference
  next_action: gather-paired-evidence
```

请求记录、编译产物、Profile、可执行代码和最终收据可以分别存储，通过引用和版本稳定连接。已知输入阶段与未见结构验证分开登记，隔离各比较组的局部反馈。
