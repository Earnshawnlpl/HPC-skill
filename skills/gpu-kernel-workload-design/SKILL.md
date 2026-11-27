---
name: gpu-kernel-workload-design
description: Define, build, validate, and freeze workload suites and comparable baselines for GPU kernel studies. Use for operator contracts, real and controlled inputs, structural coverage, references, or baseline admission.
---

# GPU Kernel 负载与基线设计

把研究问题变成可重建的目标集、独立正确性参考和实际完整调用基线。一个主模型负责冻结及核对证据，可用同一模型配置的子 agent 审阅来源、配对设计和边界。

## 工作步骤

1. 取得契约及实际调用约定，确定本轮验证已知结构特化、未见结构泛化或受控机制。记录 dtype、累加、输出、布局、别名和索引语义。缺少关键语义时标待定并完成可独立确定的覆盖。
2. 用维度、有序长度、offsets、页表或稀疏索引建立结构身份。同结构在不同数值种子与测量重复中沿用目标 ID。
3. 为真实回放、合成和压力用例登记来源、转换和合法域。指定受控配对固定量与变化量，对生成产物检查其实际覆盖。保留合法困难项、失败和后备条件。
4. 分配 O 开发、T 选参和确认、A 评估的数值实例及可见性。未见结构另留，相关派生按谱系分组。
5. 接通独立数学参考、共同起点和真实基线。核实实际库后端、默认配置及可调空间。逐结构校验支持域和实际可比计时。
6. 核对重建、配对不变量、语义和参考后，锁定结构、权重、分母、基线版本与阶段可见性。

交付契约、结构清单及其重建、来源谱系、覆盖及合法检查、参考、基线入口和选择凭据。待测与后端不支持分别保留。按照权限推进设计、实现或实测。

[负载、参考与基线手册](references/workload-and-baselines.md)含分阶段隔离、不同算子类型的覆盖示例及基线绑定。
