# ADTuner 调优模型 —— 通用原理与量化案例

本文件分两部分：
- **原理**（与内核/平台无关，可直接套用）
- **案例数据**（来自一个具体内核 + 具体平台，用于给出量级参考）

> 案例环境：曙光8000（Hygon DCU / HIP）。内核 `gauss_all_seidel_backfor`（CSR 稀疏，
> 9 个形参，`problem_size=373569`）。**换平台/换内核时量值会变，但结论方向通常不变。**

---

## §0 先在自己的平台上量三个基准数

在信任任何模型之前，先量这三个数（方法适用于任何平台/内核）：

| 基准 | 怎么量 | 用途 |
|---|---|---|
| **① 采集工具的重复性** | 同一配置连采 2~3 次，比较差异 | 决定"真实差异小于多少时排名不可信" |
| **② 采集工具 vs adtuner 的计时差** | 同一份数据、同一 kernel、同一配置，两边各测一遍 | 决定模型训练口径是否要换成 adtuner 侧 |
| **③ 空内核 / 极短内核的耗时** | 用最小问题规模跑一遍 | 判定是否被计时开销主导 |

案例量值（供量级参考）：① 中位 2.2%，最差 22%；② adtuner 比采集工具高 **+9.9%**；
③ 旧 adtuner 对亚毫秒内核有 **0.19 ms** 固定开销。

---

## §A 亚毫秒内核的计时开销（先查这个）

### 原理
若计时实现是「**每次 launch 计时一次**」，那么 host 侧的同步/轮询/事件开销会
**按 launch 次数**叠加，而不是被平均掉。提高重复次数只能降随机噪声，
**消不掉**这个固定开销。判定方法：**跑一个空内核**——如果空内核也测出可观的耗时，
说明固定开销占主导。

### 案例
`benchmark_default()` 每次 launch 都执行 `start_event → launch → stop_event → 轮询`，
轮询里有 `time.sleep(1e-6)`。实测：

```
time.sleep(1e-6) 单次实际耗时 : 0.0547 ms  (54.7 us)   ← 名义值的 54 倍
每次 launch 轮询 3.5 次       : 0.19 ms
实测到的固定开销              : 0.189 ± 0.010 ms         ← 完全吻合
```

内核真值 ~0.16 ms，**固定开销比信号还大**。

| 指标 | 修前 | 修后 |
|---|---:|---:|
| 实测时间范围 | 0.361 ~ 0.384 ms | 0.071 ~ 0.120 ms |
| 实测相对跨度 | 6.4% | **69.0%** |
| top-6 极差 | 0.83%（不可区分） | **16.90%** |
| vs 参考真值 MAPE | **102%** | **10.7%** |
| Spearman ρ | +0.20 | **+0.876** |

### 处置
改成**批量计时**：一对 event 包住 N 次 launch，再除以 N。

```python
self.dev.synchronize()
obs.before_start()
self.dev.start_event()
for _ in range(self.iterations):
    self.dev.run_kernel(func, gpu_args, threads, grid)
    obs.after_start()
self.dev.stop_event()
self.dev.synchronize()
obs.after_finish()
r = obs.get_results()
per = r["time"] / self.iterations
r["time"] = per
r["times"] = [per]
```

**仅在**「语言/后端匹配 + 只有 1 个计时 observer + 该 observer 声明支持批量」时启用，
否则回退旧路径。给 observer 加一个能力标记（如 `supports_batch_timing = True`）来安全识别。

**副作用**：`times` 从 N 个样本变成 1 个批量平均样本；连续 N 次 launch 会反复写输出数组
→ **非幂等内核（原子累加/原地更新）需先确认或加数据复位**。

**回滚**：改的是共享源码，先备份；且工作区常非干净 git 状态，**不要用 `git checkout --`**。

---

## §B 训练数据必须与目标输入「结构同构」⭐

### 原理
模型学到的是「特征 → 耗时」的映射。若训练数据的输入**结构**与推理时不同，
模型就处在**分布外**，预测会差 2~3 倍。

两个最容易漏掉的结构属性：

1. **数组相对 `problem_size` 的长度比例**
   （例如某个数组是 `14 × P` 而不是 `1 × P`）
2. **索引数组的取值范围** —— 它决定**访存/缓存行为**

第 2 条最隐蔽：如果索引只落在很小的范围（如 `[1,256]`），被索引的数组只有
几 KB 被访问（L1/L2 常驻）；如果索引铺满整个数组，就是跨数组随机 gather（打显存）。
**两者的性能特征完全不同**，模型学到的规律不通用。

### 案例
真实数据 vs 自造随机数据：

| 属性 | 真实 | 自造 | 倍差 |
|---|---:|---:|---:|
| `f` 长度 | 5,229,966（=14×P） | 373,569（=1×P） | **14.00** ❌ |
| `nc` 长度 | 373,570（=P+1） | 373,570 | 1.00 |
| `a_ae`/`ne` 长度 | 2,000,000（≈5.35×P） | 1,998,594 | 1.00 |
| **`ne` 取值范围** | **[1, 256]** | **[1, P]** | ❌ |

结果：

| 训练数据 | 对真实配置的 MAPE |
|---|---:|
| 自造随机（结构失配） | **264.3%** |
| 真实数据前缀缩放 | **8.0%** |

旁证：真实数据实测（0.072~0.119 ms）比同规模自造数据（0.12~0.15 ms）**更快** ——
因为 `ne∈[1,256]` 让被索引数组几乎全在 cache 里。

### 处置：前缀缩放（通用做法）

逐字节沿用真实数组，只按规模 `M'` 截断；不随规模变化的工作区数组保持全长、各负载共用。

```
rowptr:   arr[0 : M'+1]
csr:      arr[0 : nnz(M')]        nnz(M') = rowptr[M'] - 1
perrow:   arr[0 : M']
prop(k):  arr[0 : round(k*M')]
fixed:    全长, 各负载共用
```

这样 **`M' = problem_size` 时与原数据逐位一致**。

### 必做校验
把训练数据的特征向量与「原始 config 推理时会算出的特征」**逐项对比**，全部相同才算过关。
（`scripts/split_and_check.py` 做了这件事；正/反两向都已实测。）

---

## §C 模型类型与超参：先做退化检测

### 原理
「预测值全部相同」= 模型退化成常数预测器（输出恒等于训练集均值）。
检测只需要两行：

```python
len(set(pred)) == 1        # 或 nunique
pred.std() == 0
```

**更强的判据**：把模型 MAPE 与「直接预测均值的常数基线 MAPE」并排比。
**模型不显著优于基线 = 模型无用**，哪怕它"跑成功"了、ONNX 也存下来了。

### 案例
某实现里 `PolynomialLasso` 的 `Lasso(alpha=0.1)` 是**硬编码**的，
配上 15 维 `degree=3` 展开的 **815 项**：

| 样本数 | 非零系数 | 预测唯一值 | 结果 |
|---:|---:|---:|---|
| 16 | 0 / 815 | **1** | 退化 |
| 61 | 0 / 815 | **1** | **仍退化** |

同数据三种模型（61 样本）：

| 类型 | 预测唯一值 | 10 折 CV MAPE |
|---|---:|---:|
| `PolynomialLasso`（默认） | **1** | 15.38% |
| **`MLPModel`** | 61 | **9.89%** |
| `TwoStageModel` | 61 | 17.23% |

> 注意：MAPE 单独看分不出"真模型"和"均值"——必须和常数基线对比。

---

## §D 样本数与有效自由度

### 原理
两个独立问题，别混为一谈：

1. **样本数太少** → 任何模型都学不到（连简单模型都赢不过均值）
2. **有效自由度太少** → 特征看着多，其实只有一个在变 → **加样本也救不了**

判据：`各列 nunique()`。若 15 个特征里只有 1 个在变，那就是 1 维曲线拟合问题。

### 案例

| 估计器 | 16 样本 | 61 样本 |
|---|---:|---:|
| 常数基线 | 17.16% | 14.96% |
| LinearRegression(仅 block) | 18.6% ❌ | **14.1%** ✅ |
| Ridge + poly2 | 37.5% ❌ | **13.8%** ✅ |

**16 个样本时任何模型都赢不过均值。**

而只扫 `block_size`、`workload` 恒定时，**真实自由度 = 1**，
模型在结构上无法学习负载效应 —— 必须让**多负载**参与。

### 推荐配比
训练负载 5~6 个（含目标规模）+ 留出 1 个 + 每负载 ≥8 个线程配置 → **≥60 样本**。

---

## §E 测量分辨力决定模型能力上限

### 原理
若「配置之间的真实差异」小于「测量重复性」，那么**真实排名本身就是噪声**，
模型排不准不是模型的问题。

判定：算该负载下**相邻配置耗时差的中位数**，与该负载的**测量重复性**比。

### 案例

| 负载 | 内核耗时 | 相邻配置差异中位 | 可分辨 | 模型 ρ |
|---:|---:|---:|:---:|---:|
| 50k | **21.3 µs** | **0.06 µs（0.3%）** | ❌ | −0.413 |
| 100k | **38.0 µs** | **0.09 µs（0.2%）** | ❌ | +0.000 |
| 200k | 77.5 µs | 1.33 µs（1.7%） | ✅ | +0.874 |
| 800k | 363.1 µs | 6.02 µs（1.7%） | ✅ | +0.902 |

**模型排名能力与真值可分辨性完全对应。**

### 处置
小规模区间**放弃模型排序，直接全空间实测**。

---

## §F launch bounds / 编译期最大线程数

### 原理
若内核编译时确定了 max workgroup/threads-per-block（例如默认值或 `__launch_bounds__`），
那么请求更大的 block 会报警告甚至失败。**不是所有配置都合法**。

### 处置
1. **给内核副本加** `__launch_bounds__(N)`（N ≥ 最大 block），不改原文件便于 A/B；
   或按编译器提示用等价编译选项。
2. 或**把调优空间限制到合法范围**。
3. 若配置里 block 超过合法值，其"实测值"可信度存疑。

### 案例
```
Launch params (320, 1, 1) are larger than launch bounds (256) ...
```
加 `__launch_bounds__(1024)` 后警告消失，且实测影响很小：

| block | 无 lb | 加 lb1024 | 差 |
|---:|---:|---:|---:|
| 32 | 0.074 | 0.078 | +0.004 |
| 96 | **0.071** | **0.072** | +0.001 |
| 512 | 0.110 | 0.114 | +0.004 |

**差异 ≤4%，最优配置一致** → 可安全固定使用。

---

## §G 特征语义必须与推理端逐行核对

### 原理
训练端和推理端各自从配置构造特征向量。**必须逐行核对**，不能想当然。
通常需要核对：**顺序**、**指针用字节数还是元素数**、**标量用原值还是别的**。

### 案例（已核对）

| 项 | 训练端 | 推理端 |
|---|---|---|
| 指针 | `元素数 × sizeof(type)` | `arguments[i].nbytes` |
| 标量 | 原值 | `arguments[i]` |
| 顺序 | `grid_x,y,z, block_x,y,z, workload_*` | 同左 |

**grid 规则**：推理端 `grid_div_*=None` → 除数取 `block_size*` → `ceil(problem_size/block_size)`。
生成训练配置时**必须**用同一规则手算 `grid_size`；若 config 显式设了 `grid_div_*`，
要按它反推。

⚠️ 但这只保证「特征算法一致」。**首先要确认这个规则是否忠实于真实程序** ——
见 §K，约定选错会直接反转 block 偏好。

---

## §H 跨工具偏差与重复性

### 原理
「采集工具测出来的时间」与「adtuner 用来排名的时间」是**两套计时口径**。
模型若在其中一套上训练、在另一套上使用，就会带上一个**系统性偏差**。

### 案例

| 偏差 | 量值 |
|---|---|
| adtuner 自计时 vs 采集工具 | **+9.9%**（范围 0.94 ~ 1.30） |
| 同配置重复采集 | 平均 4.89%，中位 **2.16%**，最差 **22.2%** |

### 处置
- 若偏差可接受：直接用（并在报告里注明）
- 若要**完全同口径**：改用 **adtuner 侧采集**的数据训练
- 无论如何：**先按 §0 量出你自己平台上的这两个数**

---

## §I cache 污染会让对照实验失效

### 原理
`modelfit` 的 `model_predict()` 会**先查 cache**：

```python
if tuning_options.cache and x_int in tuning_options.cache:
    params.update(tuning_options.cache[x_int])   # 直接复用
```

若 cache 里存的是**实测值**，那么 strategy=2 的 phase-1 "预测"其实在读实测值
→ top-k 选择必然完美 → **对照实验毫无意义**。

### 判别方法（通用）
看 cache 条目的字段：

| 字段特征 | 判定 |
|---|---|
| 含 `compile_time` / `benchmark_time` / `verification_time` / **多元素** `times` | **实测** |
| 只有调优参数 + `strategy_time` + `time` + `timestamp` | **模型预测** |

### 处置
gt 与 topk 用**不同的 cache 文件名**，且**跑前删除**。

---

## §J runner_strategy 语义（一般情况）

| 值 | 名称 | 行为 |
|---|---|---|
| 0 | run directly | 全空间实测 |
| 1 | model predict | 模型预测全空间，不实测 |
| 2 | top K | 模型预测 → 取 top-k → **清空 cache、切到策略 0 实测这 k 个** |

strategy=2 的实现要点（`interface.py`）：

```python
top_k_res = util.get_topk_results(unique_results, topk)
new_searchspace.list = param_configs
tuning_options.cache = {}                 # 清空
new_tuning_options["runner_strategy"] = 0
results = topk_stategy.tune(new_searchspace, runner, new_tuning_options)
```

因此 strategy=2 天然就是"**预测 top-k vs 实测**"的对照实验 ——
本 skill 的 Phase 4 就是把它跑出来并量化。

---

## §K 固定 grid 还是自适应 grid —— 会反转 block 偏好 ⭐

### 原理
ADTuner 默认 `grid = ceil(problem_size / block_size_x)`（`grid_div_x=["block_size_x"]`），
意味着**线程总数 ≡ problem_size，与 block 无关**。

但真实程序常把 grid 与 block 当**两个独立旋钮**：

```cuda
OTFsolver<<<_B, _T>>>(...)     // _B=grid, _T=block，都来自输入卡
```

此时**线程总数 = B × T ∝ T**。若内核每线程占大量私有内存
（如 `dev_segment segs[5000]` = 5000×16 B = **80 KB/线程**），
线程总数就直接决定总 scratch 足迹 → **小 block 因线程总数少而占优**。

两种约定对 block 的偏好**可以完全相反**：自适应 grid 抹掉了"小 block 线程更少"
这一优势，只剩占用率劣势（受硬件 max workgroups/CU 上限，典型 16：
block=32 → 16×32 = 512 线程/CU = **25%** 占用），于是小 block 必输约 10%。

### 判据（动手前先做这一步）
去真实程序里找启动语句和输入卡：

```bash
grep -rn '<<<\|dim3' src/            # 看 grid 怎么来
grep -rn '_B\|_T\|block_size' run/config.yaml
```

- grid 由 `ceil(N/block)` 算出 → **自适应**，用 `problem_size = N`
- grid 是**独立输入** → **固定**，必须用 `problem_size = "block_size_x*B"`

### 案例（ANT-MOC OTFsolver，DCU "BW"，`_segments[5000]`=80 KB/线程）
真实输入卡 `_B: 512, _T: 512`（独立旋钮）。真实程序历史实测：

| 每域径迹数 | B | T=32 | T=64 | T=256 | T=512 |
|---|---|---|---|---|---|
| 2892 | 512 | **2.06 ms** | 2.53 ms | 150.61 ms | 150.82 ms |
| 71112 | 512 | 27.48 ms | **24.88 ms** | — | 170.25 ms |

线程总数 = B×T，scratch = 线程数 × 80 KB：
T=32 → 16384 线程 → 1.31 GB；T=512 → 262144 线程 → **21.0 GB** → **73× 断崖**。
所以小 block 胜出的条件是"**线程数 ≫ 径迹数**"（scratch 主导）；当线程数不足时
（71112 径迹、T=32 只有 16384 线程），并行度不足反而不如 T=64。

对照：同一个内核在**自适应 grid** 下，block=32 稳定比最优差 **9~11%**，
且把 `problem_size` 从 32928 改到 262144（grid 变 8×）**排名完全不变** ——
说明此时主导因素是占用率而非 grid。**排查顺序因此是：先定约定，再查 grid 大小。**

### 处置
```yaml
kernel:
  problem_size: "block_size_x*512"   # grid ≡ 512，忠实复现固定 _B
```

`problem_size` 与 `grid_div_x` 都支持**算术字符串表达式**：
`util.py` 用 `eval(replace_param_occurrences(s, params))` 求值，可用可调参数名。
故 `problem_size="block_size_x*512"` + 默认 `grid_div_x` →
`grid = ceil(512·block / block) = 512` 恒定。

**注意**：采集训练数据时也必须用同一约定（`problem_size` 随 block 变），
否则模型学到的 block 偏好建立在错约定上，**不可迁移**。cpm 的 collect 配置
每个只能有一个 `problem_size`，所以需要**为每个 block 值生成一份采集配置**。

### 附带检查：launch bounds
真实输入卡的大 `_T` 可能**违反程序自身 kernel 的 `__launch_bounds__`**：

```
Launch params (512, 1, 1) are larger than launch bounds (256) for kernel calsegmentsnum
```

这类配置能跑但既非法又低效，是"应该减小线程数"的硬证据 —— 顺手在日志里 grep 一下。

---

## §L `PYTHONPATH` 抢占会静默换掉后端实现 ⭐

### 原理
`adtuner` 可能同时存在**多份**源码树：pip/editable 安装的那份，和工程目录里
`vendor/` 下的副本。`sys.path` 里 `PYTHONPATH` 的条目**优先于** `site-packages`
的 `.pth` editable 路径 -- 所以只要某个脚本 `export PYTHONPATH=<vendor>`，
生效的实现就整体换掉了，**特征定义、计时方式、采样行为都可能不同**，而运行
不会报任何错。

### 判别方法
永远在作业开头打印生效路径，并断言：

```bash
PY=<venv>/bin/python
EFF=$($PY -c 'import adtuner; print(adtuner.__file__)')
case "$EFF" in *<期望的源码树名>*) echo "[env ] $EFF" ;;
  *) echo "[FATAL] effective adtuner is $EFF"; exit 2 ;; esac
```

再用 `md5sum` 锚定关键文件（本例 `backends/hthreads.py`：生效树
17,592 B / `8642a568…`，vendor 副本 58,070 B / `bc007a7f…`）。

### 案例
`scripts/env.sh` 末段 `export PYTHONPATH="$ADTUNER_SOURCE:$PYTHONPATH"`；
而 `run_yaml.slurm` 只 `activate` venv、**不** source 它。于是：

| 环境 | 生效模块根 |
|---|---|
| 仅 activate venv | `/vol8/…/mt-adtuner/adtuner/__init__.py` |
| 再 `source scripts/env.sh` | `/vol8/…/mt3000_sgemm/vendor/adtuner/adtuner/__init__.py` |

### 处置
1. 采集/训练/评测脚本**不** source 那个 env 脚本，只复刻真正需要的导出项；
2. 显式 `export PYTHONPATH=`；
3. 启动断言 + md5 锚定。
4. 顺带：这类 env 脚本常 `source /etc/profile.d/module.sh`，里面含未绑定变量，
   **在 `set -u` 下会直接终止整个 shell**（且错误被 `2>/dev/null` 吞掉时表现为
   "什么都没发生、退出码 1"）。不要对它们开 `set -u`。

---

## §M `predictor` 的加载语义：三处静默陷阱

### 1) 自动补 `.onnx`
`performancePredictor.__init__`：

```python
base_path = os.path.splitext(model_path)[0]      # 先剥扩展名
single_onnx_path = f"{base_path}.onnx"          # 再拼 .onnx
```

所以传 `.pt` / `.joblib` **不会报"类型不对"**，而是去找同名 `.onnx` 并抛
`FileNotFoundError`。→ 永远传存在的 `.onnx`（绝对路径最稳）。

### 2) `model_type` 只在 `doc_string` 里
`model_type` 取自 ONNX 的 `doc_string`（JSON），**YAML 里的 `model_type` 不参与**。
只有 `"TwoStageModel"` 走特殊分支（需 `_stage1.onnx` + `_stage2.onnx` +
`y_scale`/`y_mean`）；其他任意值都走单输入单输出。

→ 导出时务必 `graph.doc_string = json.dumps({"model_type": "MLPModel", ...})`
并 `onnx.save`；文件名**不要**带 `_stage1`/`_stage2` 后缀。

### 3) `model_map` 的回退是坏的
`runners/modelfit.py` 的 `model_map` 里 `"GENERIC"` 指向 `../models/best_model_fold.pt`，
而发行版 `models/` 目录里往往**一个 `.onnx` 都没有**。→ 走回退必
`FileNotFoundError`。→ **必须**在 YAML 里显式给 `modelfit.model_path`。

### 4) 每次预测都重建 ONNX session —— 决定 strategy=2 的 phase-1 预算 ⭐`modelfit.model_predict()` 对 `parameter_space` 的每个元素都会被
`CostFunc.__call__` 以 `runner.run([params], ...)` 单独调用一次，而
`model_predict` 内部**每次都新建** `performancePredictor(model_path)`
（= 新建一个 onnxruntime `InferenceSession`）。实测（77 KB MLP，MT3000 登录节点）：

| 操作 | 耗时 |
|---|---|
| `performancePredictor(path)` 构造 | **127 ms** |
| 单次 `predict()` | 0.098 ms |
| 构造 + 一次预测 | 126 ms |

所以 phase 1 的墙钟 ≈ `max_fevals × 127 ms`：

| `max_fevals` | phase-1 估算 |
|---|---|
| 256 | 32 s |
| 512 | 65 s |
| 1024 | 129 s |
| 3240（全空间） | **409 s** |

处置：**先用本文的方法量一遍，再定 `max_fevals`**。别想当然把 `max_fevals`
设成搜索空间大小 —— 那会把"模型筛选"从秒级变成分钟级，而 `greedy_mls`
带 `randomize=True` 随机重启，筛到 512 之后的边际收益很小。
（顺便：这也是一个可以提给 ADTuner 的优化点 —— session 应该缓存。）

---

## §N `random_sample` 会静默忽略 `strategy_options.max_fevals` ⭐

### 原理
不同策略读 `max_fevals` 的**位置不同**：

| 策略 | 读取位置 | 是否生效 |
|---|---|---|
| `greedy_mls` / `greedy_ils` | `tuning_options.strategy_options` | ✅ |
| `random_sample` | **顶层 `tuning_options`** | ❌（一般情况） |

而 `interface.py` 只在**同时满足** `modelfit_mode` 时才把
`strategy_options["max_fevals"]` 写进顶层：

```python
if (strategy_options and "max_fevals" in strategy_options
        and not simulation_mode and modelfit_mode and runner_strategy != 2):
    tuning_options["max_fevals"] = strategy_options["max_fevals"]
```

所以普通采集（`modelfit.enabled: false`）写着 `max_fevals: 200`，
`random_sample` 仍然用它自己的默认 `fraction=0.1`。

### 案例（实测）
`strategy_comparison/random_sample/config.yaml` 写 `max_fevals: 200`，
三个种子实际都采到 **324** 条 = `ceil(3240 × 0.1)`；
同一批实验里 `greedy_mls` 写 200 得到 200/200/203 条。

### 处置
用 `random_sample` 就用 **`fraction`**，并反算：

```python
fraction = (target - 0.5) / size     # 不要写 target/size
```

因为 `num_samples = ceil(size * fraction)`，而 `size*(target/size)` 在浮点下
可能略大于 `target`，`ceil` 后多一条（3240×96/3240 → 97）。

---

## §O YAML 的**键序**决定特征顺序（hthreads 变体尤其）

### 原理
`get_features` 用 `params.items()` 的**插入序**生成 `tune_para_0..N`，
而 `params` 来自 `dict(zip(tuning_options.tune_params.keys(), element))`
→ `tune_params` 来自 YAML `tuning.parameters` 的键序。

**两个顺序约束**（写错不会报错，只会让模型精度莫名其妙地差）：

1. `inputs[]` 的顺序 = `workload_0..N` 的顺序。
   例如 GEMM 必须是 `M, N, K, A, B, C`（标量原值，数组取 `nbytes`）。
2. `tuning.parameters` 的键序 = `tune_para_0..N` 的顺序。

### hthreads 变体的特征布局（与 CUDA/HIP 不同！）
```
[0]      hthreads_num_threads       <- 来自 params 的独立字段
[1..6]   M, N, K, A.nbytes, B.nbytes, C.nbytes     <- inputs 顺序
[7..13]  BM, BN, BK, MR, NR, KU, DOUBLE_BUFFER     <- parameters 键序（去掉线程数）
```
**没有 grid_size_*/block_size_***：`get_features` 显式把它们和
`hthreads_num_threads` 一起跳过。所以 `input_size = 1 + len(inputs) + (len(params)-1)` = 14。

### 处置
写一个**契约测试**：用桩对象直接调用生效实现的 `get_features` 与
`get_prediction`，比对键序、数值，并断言最终喂给 ONNX 的向量长度与顺序。
桩只需要 `kernel_options.arguments`（`kernel_instance` 未被使用），
因此无需初始化设备：

```python
features = DeviceInterface.get_features(None, None, ko, params)   # 真实现
out      = DeviceInterface.get_prediction(None, features, stub_predictor)
```

`DeviceInterface` 里可能有**多个同名 `get_prediction`** 定义（后定义者生效），
这正是在 `backends/*.py` 里 grep 不到的原因 —— 实现继承自 `core.py` 的基类。

---

## §P hthreads：必须 `adtuner-run`，直接 `sbatch` 会死等

### 原理
hthreads 后端把**宿主端 DSP 编译**委托给**登录节点**：作业在
`__ADTUNER_MT_CACHE__/start_compile.txt` 上等待，由登录侧的
`adtuner-run`（`adtuner/hthreads_cli.py`）轮询该目录并执行 `make`，
作业结束後 `rmtree` 整个 `__ADTUNER_MT_CACHE__`。

`adtuner-run <slurm脚本>` 做三件事：
1. `sbatch` 提交；
2. 循环 `while True`：发现 `start_compile.txt` → `cd __ADTUNER_MT_CACHE__ && make`
   → 删信号 → 写 `compile_finish.txt`；每 5 次循环检查一次作业是否结束；
3. 作业结束后清理缓存目录。

### 症状与处置
* 直接 `sbatch run_yaml.slurm` → 作业**永远卡在**第一个配置，不报错，
  `squeue` 里长期 R 状态。
* 必须 `cd <含 slurm 与 config 的目录> && adtuner-run run_yaml.slurm`
  —— `adtuner-run` 与作业的 CWD 必须是同一个（缓存目录就在 CWD 下）。
* 生成脚本时记得把上一轮的 `__ADTUNER_MT_CACHE__` 一并 `rm -rf`
  （正常结束会被 `adtuner-run` 清掉，异常退出会残留）。
* 这类"登录侧帮忙编译"的架构下，**登录侧的环境**（工具链变量，如 `MT_ROOT`）
  才是编译环境；作业侧只需要能等信号。别把环境配在错的一侧。
* 因为要交互式轮询，长任务要脱离 ssh 会话跑：
  `nohup setsid bash collect.sh > /tmp/collect.log 2>&1 < /dev/null &`
  —— 否则 ssh 一断，`adtuner-run` 死掉，作业就永久挂住。

---

## §Q 远端脚本卫生：三个会静默损坏文件的操作 ⭐

### 1) `sed -i 's/\r$//'` 在某些环境里等价于 `s/r$//`
在该集群上实测：它**不是**"删除行尾 CR"，而是**删掉每一行末尾的字母 `r`**：

| 原意 | 实际结果 |
|---|---|
| `latest[(r["shape"], r["tag"])] = r` | `latest[(r["shape"], r["tag"])] = ` → 末尾汇总 Python 报 `SyntaxError` |
| `#SBATCH -e error_a.err` | `#SBATCH -e error_a.er`（SLURM 的 stderr 文件名被改，且**不报错**） |

`bash -n` 检查不出这种损坏（它只查 shell 语法，不查 heredoc 里的 Python）。

**处置**：不要做"换行符清理"。生成时就用 LF（Python `open(..., newline="\n")`），
上传即用。确需清理用 `tr -d '\r'` 或 `perl -pe 's/\r$//'`。

### 2) PowerShell 文本替换会破坏 UTF-8 中文
`(Get-Content f -Raw) -replace ... | Set-Content f -Encoding UTF8` 会把中文变成
乱码并使文件不可导入。**改文件只用结构化编辑器/`edit` 工具，不要用 shell 文本替换。**

### 3) PowerShell 剥掉传给原生程序的 `"`
`ssh host 'python -c "..."'` 里的 `"` 常常消失，导致远端语法错误；
`|` 也可能被当成本地管道。→ 远端命令只写**简单 token**；任何复杂逻辑
**写成脚本文件再 `scp`**，用 `bash script.sh` 调用。

### 纪律：改完必须"本地 ↔ 远端"逐行核对
```bash
md5sum <remote>/script.sh          # 与本地副本比对
grep -n '可疑行' <remote>/script.sh
diff <(tr -d '\r' < local) <(tr -d '\r' < remote)
```
本项目靠 `md5sum` + 定点 `grep` 才发现上述损坏 —— **不要假设 scp 之后文件是完好的**。
