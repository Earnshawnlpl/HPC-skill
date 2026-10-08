---
name: adtuner-tuning-model
description: 给定任意 ADTuner 的 config.yaml，生成与目标输入结构同构的多负载训练数据、用采集工具（cpm/hipprof 等）采集、训练可泛化的性能模型，并通过 ADTuner runner_strategy 0/2 检验。Use when the user gives an ADTuner config.yaml (any kernel, any device) and asks to collect kernel data, train a generalizable kernel performance model, or verify model-guided ADTuner tuning. Platform- and kernel-agnostic — cluster paths live in a profile YAML.
---

# ADTuner 调优模型构建

典型输入：`<path/to/config.yaml> --workdir <dir> [--profile <yaml>] [--levels 6] [--blocks 32,64,...]`

## 适用范围

**任何** ADTuner 的 `config.yaml`（`kernel` + `inputs` + `tuning` 三段式），
**任何**支持 ADTuner 的平台（HIP/CUDA/OpenCL，SLURM/PBS/LSF/本地）。

与内核、平台**无关**的部分是流程与判据；与平台相关的路径/调度器集中在
`profiles/*.yaml`，与内核相关的缩放规则通过 `--roles` 显式声明。

| 支持 | 说明 |
|---|---|
| 内核形态 | CSR/稀疏类（有行指针）、元素级（vecadd/GEMM 类，无行指针）、任意多输入 |
| 输入类型 | `int/float/double/short/...` 及指针形式；标量取原值 |
| 平台 | 只要能把「采集工具」和「adtuner」跑起来即可（本 skill 不假定厂商） |
| 调度器 | profile 里声明；`slurm` 可直接用，其他类型脚本会给出改写提示 |

**不适用**：内核没有可缩放的问题规模（`problem_size` 无法变小）、
或输入数据无法用「前缀」表达更小的问题。

### ⚠️ 两种流程变体，先判断走哪条

| 情形 | 走哪条 | Phase 2/3 |
|---|---|---|
| 平台有 `cpm`/`hipprof` 之类的采集工具 | 本文件的标准流程 | 用 `collect.json` / `train.json` |
| 平台**没有**采集工具，或它无法表达目标内核 | `references/variant-adtuner-side-collection.md` | 改成 **ADTuner 侧采集**（`runner_strategy: 0` 的 cache 即数据集）+ **离线 torch→onnx 训练** |

`profiles/mt3000.yaml`（长沙超算 MT-3000 / hthreads）就是第二种变体的完整示例。
该变体下 Phase 0 / Phase 1 / Phase 4 的判据不变，但 §3 之后把"采集工具"
一律替换成 ADTuner 自己 —— 对应的新陷阱见 `pitfalls.md` §L–§P。

---

## 第 0 步 — 建立平台档案

集群路径**不要写死在流程里**，填一份 profile：

```bash
cp scripts/../profiles/TEMPLATE.yaml profiles/<my-cluster>.yaml
# 填 paths / env / scheduler / kwargs
```

已提供的示例：
* `profiles/sugon8000.yaml`（曙光8000，Hygon DCU + SLURM）—— 标准流程
* `profiles/mt3000.yaml`（长沙超算 MT-3000 / hthreads）—— **无 cpm 变体**，
  注意它用 `adtuner-run` 启动、必须清空 `PYTHONPATH`、且没有 grid/block 特征

之后所有脚本都用 `--profile profiles/<my-cluster>.yaml`；不给则用该示例。

profile 关键字段：

| 段 | 字段 |
|---|---|
| `paths` | `cpm` / `cpm_python` / `adtuner` / `adtuner_python` / `adtuner_source` |
| `env` | `conda_sh` / `cpm_env` / `adtuner_env` / `modules[]` / `adtuner_modules[]` |
| `scheduler` | `type` / `partition` / `gres` / `collect_time` / `verify_time` |
| `kwargs` | `launch_bounds`（block 超过该值时自动加 `__launch_bounds__`） |

> 采集工具不一定是 cpm —— profile 里换掉 `paths.cpm` 即可；
> 本流程只要求它「吃一个 JSON 配置、吐一个含特征与耗时的 CSV」。

---

## Phase 0 — 读懂 config.yaml

```bash
$PYPY scripts/probe_config.py <config.yaml> [--profile <yaml>]
```

记录三件事：

1. `kernel.path` / `kernel.problem_size` / `kernel.lang`
2. `inputs[]` 的**顺序**（= 内核形参顺序，必须严格保持）、类型、`shape`
3. `tuning.parameters`（可调参数空间）

**关键产出：每个指针输入相对 `problem_size` 的「尺寸比例指纹」。**
后面的数据必须复现这些比例。脚本会直接算给你，并标出异常比例
（例如某数组是 `14 × problem_size` —— 这类数组最容易被写错）。

---

## Phase 1 — 生成多负载训练数据 ⭐最关键

### 通用铁律：训练数据的输入结构必须与目标 config **同构**

具体做法：**前缀缩放**。逐字节沿用真实数组，只按问题规模 `M'` 截断。
这样在 `M' = problem_size` 处与原数据**完全一致**，

```
M' = problem_size   →   特征向量与 adtuner 推理时逐位相同
```

对于不随问题规模变化的工作区数组，保持全长、各负载共用。

> **为什么不能随便造数据**（与内核无关的教训）：自造同形状随机数据会改变
> - 数组相对 `problem_size` 的**长度比例**
> - 索引数组的**取值范围** → 访存/缓存行为
>
> 二者都会让模型变成分布外。实测案例见 `references/pitfalls.md` §B（MAPE 264%）。

### 缩放角色（自动推断，`--roles` 可覆盖）

| 角色 | 判别 | 缩放 |
|---|---|---|
| `rowptr` | 整型，长度 = `problem_size+1`，单调不减 | 截到 `M'+1` |
| `csr` | 长度 = `rowptr[P]-1` 的配套数组 | 截到 `nnz(M')` |
| `perrow` | 长度 = `problem_size` | 截到 `M'` |
| `prop` | 长度 = `k×problem_size`（2≤k≤4） | 截到 `round(k·M')` |
| `fixed` | 其余（独立于问题规模的工作区） | 保持全长，共用 |

无行指针的元素级内核自动进入 **no-CSR 模式**（`perrow`/`prop`/`fixed` 仍可用）。
**脚本会打印推断结果，必须人工确认**；不确定就用 `--roles name=role,...` 显式指定。

```bash
$PYPY scripts/gen_dataset.py <config.yaml> --workdir <dir> \
      [--profile <yaml>] [--levels 6] [--blocks 32,64,...] [--roles f=fixed]
```

自动产出 `data/`、`collect.json`、`levels.json`、`collect.slurm`。

### 规模设计

- 训练负载 **5~6 个**，**必须包含目标 `problem_size` 本身**
- 留出负载 **1 个**中间规模，**不参与训练**（泛化测试）
- 线程配置 **≥8 个**，覆盖调优空间（默认 12 个 block 值）
- **总样本数 ≥ 60**（16 个样本时任何模型都赢不过「预测均值」）

### 结构自检（脚本自动跑，必须全过）

行指针单调、`rowptr[M']-1 == nnz'`、索引数组不越界、**`M'=P` 时与原数据一致**。

---

## Phase 2 — 采集

```bash
sbatch <dir>/collect.slurm        # 脚本已按平台档案渲染（模块/conda/分区/gres）
```

写采集配置的通用要点：

- 指针输入：`"value": "<绝对路径>"`（按文件字节数 / `sizeof(T)` 定长度）
- 标量输入：`"value": <数值>`，**不写 `size`**
- 输出数组（`zero` 初始化）：`"value": 0`
- **`grid_size = ceil(problem_size / block_size)`**
  —— 必须与 adtuner 的算法一致。adtuner 侧：`grid_div_x=None` → 除数取
  `block_size_x` → `ceil(problem_size/block_size_x)`。若你的 config 显式设了
  `grid_div_*`，要按它反推。
- 每次配置通常要重新编译一次内核 → 耗时 ≈ 配置数 × 编译时间
  （参考：84 个配置约 11 分钟）

---

## Phase 3 — 训练

```bash
$PYPY scripts/split_and_check.py <config.yaml> --workdir <dir> [--profile <yaml>]
# 然后到采集环境（profile 的 cpm_env）里跑 cpm <dir>/train.json
```

### 三条铁律

1. **`model_type` 优先用 `MLPModel`**
   > 某些实现里 `PolynomialLasso` 的超参（如 `Lasso(alpha=0.1)`）是硬编码的，
   > 配上高次多项式展开（15 维 → 815 项）时，**即使 60 个样本也会把系数全压成 0**，
   > 退化成常数预测器。实测：`MLPModel` MAPE 4.33% vs `PolynomialLasso` 退化。
   > 选型前先用 `eval_model.py` 的退化检测确认。

2. **训练前必须切掉留出负载**（采集/训练工具会训练整个 CSV）

3. **特征必须逐位一致** —— `split_and_check.py` 会把你的数据集与
   「原始 config 推理时会算出的特征」逐项对比，**全部相同**才算过关。
   不一致说明比例错了，回 Phase 1。

### 特征语义（cpm ↔ adtuner 已逐行核对，通用）

| 项 | 采集端 | adtuner 推理端 |
|---|---|---|
| 指针 | `元素数 × sizeof(type)` 字节 | `arguments[i].nbytes` |
| 标量 | 原值 | `arguments[i]` |
| 顺序 | `grid_x,y,z, block_x,y,z, workload_*` | 同左 |

---

## Phase 4 — 用 adtuner 检验

```bash
$PYPY scripts/make_verify.py <config.yaml> --workdir <dir> --model <dir>/model.onnx \
      [--profile <yaml>] [--kernel <带 launch_bounds 的内核副本>]
sbatch <dir>/verify.slurm
$PYPY scripts/analyze_adtuner.py <dir>/job_verify_<jobid>.out
```

生成两份 config，**只差 `runner_strategy`**：

| 目录 | runner_strategy | 作用 |
|---|---|---|
| `gt/` | **0** | 全空间实测 = 真值 |
| `topk/` | **2** | 模型预测 top-k → 再实测 |

> ⚠️ 两份的 cache 文件名必须不同且跑前清空。adtuner 的 `model_predict` 优先读 cache，
> 若 cache 里是实测值，phase-1 的"预测"就变成读实测值，**对照实验失效**。
> 判别方法见 `references/pitfalls.md` §I。

---

## 必查陷阱（按概率排序）

| # | 陷阱 | 症状 | 处置 |
|---|---|---|---|
| 1 | **输入结构不同构** | 预测与实测差 2~3 倍 | 前缀缩放；`--roles` 确认每个数组的角色 |
| 2 | **特征不一致** | 同上 | `split_and_check.py` 逐项校验 |
| 3 | **模型类型/超参不当** | 预测值全部相同（退化） | 换 `MLPModel`；用退化检测确认 |
| 4 | **样本太少 / 单一负载** | 排序差 | ≥60 样本，多负载 × 多线程 |
| 5 | **亚毫秒内核的计时开销** | 实测被抬高、差异被压平 | 见 `pitfalls.md` §A（批量计时补丁） |
| 6 | **launch bounds** | `Launch params (N,1,1) larger than launch bounds (M)` | 内核副本加 `__launch_bounds__`；或把调优空间限制到合法范围 |
| 7 | **测量分辨力不足** | 小负载下排序随机 | 内核过短时放弃模型排序，直接全空间实测 |
| 8 | **cache 污染** | 预测"完美"命中 | gt/topk 用不同 cache 且跑前清空 |
| 9 | **`PYTHONPATH` 抢占换掉后端** | 特征/计时/采样行为与文档不符 | 断言生效树 + md5 锚定（§L） |
| 10 | **`predictor` 三处静默陷阱** | `.onnx` 找不到；`model_type` 不生效 | 传绝对 `.onnx`；写 `doc_string`（§M） |
| 11 | **`random_sample` 忽略 `max_fevals`** | 采到的条数与设定不符（324 vs 200） | 改用 `fraction=(target-0.5)/size`（§N） |
| 12 | **YAML 键序 = 特征顺序** | 模型精度莫名差 | 契约测试直接调 `get_features`（§O） |
| 13 | **hthreads 必须 `adtuner-run`** | 作业永远卡在第一个配置 | 别直接 `sbatch`（§P） |
| 14 | **远端脚本卫生** | `sed 's/\r$//'` 删掉行尾 `r`；PS 替换毁 UTF-8 | 只传 LF，改完 `md5sum`/定点 grep 核对（§Q） |

---

## 已知共性偏差（与具体平台无关）

| 偏差 | 典型量值 | 说明 |
|---|---|---|
| 采集工具（如 hipprof） vs adtuner 自计时 | 约 **+10%** | 两者计时口径不同（单次 kernel vs 批量） |
| 同配置重复采集 | 中位 **~2%**，最差 **>20%** | 真实差异小于它时，排名不可信 |
| MT3000（adtuner 侧采集 vs `srun sgemm`，25 对共享配置） | 中位 **+0.10%**，绝对值均值 3.14% | 两套口径基本一致；但 **512³ 上有 +8.8%~+11.5% 的系统性偏移** |
| MT3000 评测仪器自身组内相对标准差 | 均值 **2.09%**，最大 13.66% | 这是**噪声底**：小于它的差异不可作为结论 |
| MODELFIT 每次预测重建 ONNX session | **127 ms/次** | 决定 `runner_strategy: 2` 的 phase-1 预算，见 §M(4) |

> 要让模型与 adtuner **完全同口径**，应改用 **adtuner 侧采集**的数据训练。
> 具体量值随平台不同，务必在你自己的平台上实测一遍（`pitfalls.md` §H 给了方法）。

---

## 脚本清单

全部脚本用**同一个** python 运行 —— 需要 `yaml + numpy + pandas + onnxruntime`；
profile 的 `paths.adtuner_python` 通常满足。

| 脚本 | 平台/内核相关性 |
|---|---|
| `scripts/profile.py` | 平台档案加载 + 作业脚本渲染（改平台只改 YAML） |
| `scripts/probe_config.py` | 无（只依赖 config.yaml 结构） |
| `scripts/gen_dataset.py` | 内核相关性集中在「角色推断」，可用 `--roles` 覆盖 |
| `scripts/split_and_check.py` | 无 |
| `scripts/eval_model.py` | 无 |
| `scripts/make_verify.py` | 无（作业脚本按 profile 渲染） |
| `scripts/analyze_adtuner.py` | 无 |
| `profiles/sugon8000.yaml` | 平台示例（标准流程，有 cpm） |
| `profiles/mt3000.yaml` | 平台示例（**无 cpm 变体**） |
| `profiles/TEMPLATE.yaml` | 新平台模板 |
| `references/variant-adtuner-side-collection.md` | **无采集工具时的完整替代流程**（ADTuner 侧采集 + 离线 ONNX） |
| `references/pitfalls.md` | 原理 + 量化案例（§L–§P 为无 cpm 变体专用） |

---

## 最短可执行流程

```bash
PYPY=<profile.paths.adtuner_python>
W=<工作目录>

$PYPY scripts/probe_config.py    <config.yaml> --profile <prof>
$PYPY scripts/gen_dataset.py     <config.yaml> --workdir $W --profile <prof>
sbatch $W/collect.slurm                                   # Phase 2
$PYPY scripts/split_and_check.py <config.yaml> --workdir $W --profile <prof>
cpm $W/train.json                                         # Phase 3（MLPModel）
$PYPY scripts/eval_model.py --workdir $W
$PYPY scripts/make_verify.py <config.yaml> --workdir $W --model $W/model.onnx --profile <prof>
sbatch $W/verify.slurm                                    # Phase 4
$PYPY scripts/analyze_adtuner.py $W/job_verify_<jobid>.out
```

> **无 cpm 的变体**（`profiles/mt3000.yaml`）不适用上面这条链：
> 采集换成跑 `runner_strategy: 0` 的调优作业（cache 即数据集），
> 训练换成离线 `torch.onnx.export`（归一化与逆变换必须烘进图内），
> 评测另配一条独立仪器。完整步骤与检查清单见
> `references/variant-adtuner-side-collection.md`。

`train.json`：

```json
{
  "task": "train_kernel_model",
  "dataset_file_path": "<W>/dataset_train.csv",
  "model_type": "MLPModel",
  "model_save_path": "<W>/model.onnx",
  "kernel_name": "<内核名>",
  "visit_file_path": "<W>/visit.json"
}
```

---

## 检查清单

**标准流程（有 cpm）**

```
[ ] 第0步  平台档案就绪（paths/env/scheduler），能提交作业
[ ] Phase 0 读懂 config.yaml，记录尺寸比例指纹（L/P）
[ ] Phase 1 角色推断已人工确认；结构自检全过；M'=P 与原数据一致
[ ] Phase 1 训练负载含目标 problem_size；留出 1 个中间规模；样本 ≥60
[ ] Phase 2 grid = ceil(problem_size/block)；采集完成
[ ] Phase 3 切掉留出负载；特征逐项一致；模型非退化；留出集精度可接受
[ ] Phase 4 gt(strategy0) + topk(strategy2)，cache 清空
[ ] Phase 4 top-k 重合 / Spearman / 最优命中 均已量化
[ ] 交付    模型 + 数据集 + 验证报告（含已知偏差）
```

**无采集工具的变体（见 `references/variant-adtuner-side-collection.md`）**

```
[ ] 平台确实没有可用采集工具，或它无法表达目标内核
[ ] 查源码锚定 cache `time` 口径（聚合方式 / 是否含 memcpy / iterations）
[ ] 量出 cache `time` vs 独立仪器的偏差，以及独立仪器自身的重复性（噪声底）
[ ] 数据集只取 runner_strategy=0 的 cache（strategy=2 的 cache 只有预测值）
[ ] 采样用 A/B/C 三段式；random_sample 用 fraction；每形状不同种子
[ ] 划分按形状分组；单形状训练无法得到形状泛化（不要用它验证机器）
[ ] ONNX 图自包含：log1p(跨数量级特征) + 归一化 + 逆变换
[ ] std 用相对下限；验证指标非有限必须显式失败
[ ] 导出后用 adtuner 自己的 performancePredictor 做契约测试
[ ] 基线 ≥3 个（含形状均值 oracle）；另给 leave-one-shape-out CV
[ ] 评测用独立仪器 + 专门生成的无偏样本（不含模型预测最优）
[ ] 模型引导 top-k 由独立仪器实测（不依赖解析 stdout），并与 strategy=2 交叉验证
[ ] 报告区分 in-sample / out-of-sample / oracle 对齐三种口径
[ ] strategy=2 前先量 predictor 构造成本，据此定 max_fevals
```
