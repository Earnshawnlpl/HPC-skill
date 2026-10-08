# 变体：没有 cpm / 没有采集工具时，用 ADTuner 自己当采集器

适用场景（`profiles/mt3000.yaml` 即此类）：

* 平台**没有** `cpm` / `hipprof` 这类"吃配置吐 CSV"的采集工具；
* 或者采集工具**无法**表达目标内核的输入（例如 hthreads 的内核必须由
  ADTuner 的 `lang: Hthreads` 后端自己去驱动 DSP 编译与计时）；
* 但 `adtuner` 本身能跑，且能把每次评估的**参数 + 实测耗时**落盘。

此时本 skill 的标准 Phase 2/3（`cpm collect.json` → `cpm train.json`）
**不适用**，改用下面的流程。Phase 0 / Phase 1 / Phase 4 的判据不变。

---

## 1. 核心反转：ADTuner 自己就是采集器

跑 `runner_strategy: 0`（`directly_run`，真实编译+测量）的调优作业，
它产出的 `tuning_cache.json` 就是数据集：

```json
{ "objective": "time",
  "device_name": "MT-3000",
  "problem_size": [1024, 1024],
  "tune_params_keys": ["BM","BN","BK","MR","NR","KU","DOUBLE_BUFFER","hthreads_num_threads"],
  "cache": { "64,256,64,4,2,4,1,16": {
      "BM": 64, "BN": 256, "...": "…", "time": 1.7423,
      "times": [1.7467, 1.7404, "...7 个..."],
      "compile_time": 1228.5, "benchmark_time": 171.1,
      "verification_time": 0, "strategy_time": 38.2, "timestamp": "…" } } }
```

三点必须记住：

1. **只有 `runner_strategy: 0` 的 cache 能当标签。**
   `runner_strategy: 2` 时 `interface.py` 会把 phase 2 的实测值丢掉
   （`tuning_options.cache = {}`），cache 里只剩 **phase 1 的模型预测值**。
   拿它训练 = 自举污染。
2. **`time` 的口径必须查源码确认**（见 §2），不要猜。
3. 键是**逗号连接的参数元组**，顺序 = `tune_params_keys`；条目里同时有
   8 个参数名可以互校（键序与条目不一致说明有人改过 config）。

---

## 2. 先锚定 `time` 的口径（对应 §0）

去生效源码里找三处：

```bash
grep -rn 'def compile_and_benchmark' <adtuner_source>/adtuner/
grep -rn 'def benchmark'           <adtuner_source>/adtuner/core.py
cat <adtuner_source>/adtuner/observers/<backend>.py
```

MT3000 的答案（可作为"该怎么查"的范例）：

```
core.py:849  compile_and_benchmark  -> result.update(self.benchmark(...))
core.py:671  benchmark_default      -> 跑 self.iterations 次（YAML tuning.iterations）
observers/hthreads.py               -> after_finish: times.append((dev.end-dev.start)*1000)
                                       get_results : {"time": np.average(times), "times": [...]}
```

⇒ `time` = **`iterations`(默认 7) 次单次 kernel 墙钟的算术平均**，单位 ms，
纯 kernel、**不含 memcpy**。

然后按 §H 量出「这套口径」与「你的独立评测仪器」之间的偏差 —— 这是后面
解释 MAPE 的前提。MT3000 的实测结果（25 对共享配置）：

| 指标 | 值 |
|---|---|
| ratio(adtuner/独立仪器) 中位 | **1.0010** |
| 相对差中位 | **+0.10%** |
| 相对差绝对值均值 | 3.14% |
| 独立仪器自身组内相对标准差 | 均值 **2.09%**（噪声底） |
| 小规模系统性偏移 | 512³ 上 **+8.8%~+11.5%**（冷启动） |

---

## 3. 采样设计：怎么让 ADTuner 采到"设计好的"样本

`runner_strategy: 0` 时，样本由**策略**决定，配置里给不了任意参数列表
（`parsers/` 里没有 `parameter_space` 之类的键）。可用的旋钮：

| 策略 | 控制手段 | 特点 |
|---|---|---|
| `brute_force` | 收窄 `tuning.parameters` 的取值集合 | 只走笛卡尔积 → 适合"小网格定向覆盖" |
| `random_sample` | **`strategy_options.fraction`** | 从合法空间**均匀随机不放回**抽 `ceil(size×fraction)` 个 |
| `greedy_mls` | `strategy_options.max_fevals` | 自适应爬山，预算集中在低耗时区 |

⚠️ **`random_sample` 的 `max_fevals` 会被静默忽略**（见 pitfalls §N）。
要用它就写 `fraction`，并用 `(target-0.5)/size` 反算以保证条数精确。

### 推荐的三段式（本项目做法）

单一策略必然偏：`random_sample` 均匀但**几乎撞不到最优邻域**（3240 里抽 162，
命中某个 24 点邻域的期望不到 1 个点）；`greedy_mls` 精于最优区但覆盖不到全局。

| 运行 | 策略 | 条数 | 作用 |
|---|---|---|---|
| A | `random_sample` + `fraction` | 162 | 全局无偏覆盖 |
| B | `brute_force` + 受限网格 | 24 | 强制覆盖历史最优邻域 |
| C | `greedy_mls` + `max_fevals` | 40 | 低耗时区加密（代理模型最需要精度的区域） |

每个**训练形状**三次运行，共 226 条；B 的受限网格要覆盖已知最优邻域，
例如 `BM[16,32,64]×BN[128,256]×BK[32,64]×MR[4]×NR[2]×KU[4]×DB[1]×threads[8,16]=24`。

**每个形状用不同种子**，让 10 个形状的采样并集远大于单形状 —— 形状×参数交互
才学得到。

### 形状划分

按**形状**分组，绝不按配置随机划分（否则测的是插值不是泛化）。
本项目 10 训练（含 2 个内部早停验证形状）+ 3 留出；留出形状刻意选
①非整除边界（513×769×1003）②未见长宽比（1536×384×2560）
③量级外推（8192×512×1024）。

---

## 4. 离线训练：ONNX 图必须**自包含**

没有 `cpm train.json` 时训练自己在 venv 里做（`torch` + `onnx` 足够；
`tf2onnx`/`skl2onnx` 通常没装，所以走 `torch.onnx.export`）。

**关键约束来自推理端**：非 `TwoStageModel` 路径**不做任何反变换**，只
`max(res, 0)`。所以：

1. 训练目标可以取 `log(time)`，但**逆变换 `exp` 必须放进图内**；
2. **归一化必须放进图内**（`Sub`/`Div` + initializer）。跨形状的
   `A.nbytes` 可达 3.4e7 而 `BM` 只有 16~64，量级差 6 个数量级，
   推理端没有归一化入口。

```python
class _Wrap(nn.Module):
    def __init__(self):
        self.register_buffer("mean", torch.tensor(mean)); self.register_buffer("std", torch.tensor(std))
        self.net = nn.Sequential(nn.Linear(14,128), nn.GELU(),
                                 nn.Linear(128,128), nn.GELU(), nn.Linear(128,1))
    def forward(self, x):
        y_log = self.net((x - self.mean) / self.std).squeeze(-1)
        return torch.exp(y_log).clamp(min=0.0)     # 直接输出毫秒

torch.onnx.export(model, dummy, path, input_names=["input"], output_names=["output"],
                  dynamic_axes={"input":{0:"batch"}, "output":{0:"batch"}}, opset_version=17,
                  dynamo=False)
graph = onnx.load(path); graph.doc_string = json.dumps({"model_type":"MLPModel", ...})
onnx.save(graph, path)      # model_type 只能从 doc_string 读，见 pitfalls §M
```

**导出后必做契约测试**：用 `adtuner.predictor.performancePredictor` 真的加载一次，
断言 `model_type` 正确、ONNX 与 torch 输出最大偏差 < 1e-3、输出非负。

### 图内变换的两个必做防护（都踩过）

**(1) `std` 的下限必须是相对值。**
`std = max(std, 1e-8)` 看着无害，但当某特征在训练集里是常数时（例如只有一个
训练形状，`workload_*` 全部相同），测试集上 0.69 的差异会被放大成 `6.9e7`，
MLP 直接外推出 `inf`。用 `std = max(std, 0.02 × (|mean| + 1))`。

**(2) "验证集指标非有限"必须显式失败。**
早停初值 `best = inf`；若每轮验证指标也是 `inf`，则 `x < best - 1e-6` 永假 →
`best_state` 保持 `None` → 函数返回**最后一个 epoch（等于未训练）的模型**，
而流程若无 check 就会继续往下走并产出垃圾报告。
务必：单独计数非有限轮次；结束时若从未出现有限改进，直接抛错。

**顺带一条纪律**：验证"训练/导出机器能跑"时，**不要**用"单形状训练 + 另一形状
验证"的划分 —— 那必然外推失败（形状特征方差为 0，对其它形状是纯外推），
`inf` 会掩盖真正的实现错误。验证机器请用**同形状**的 fit/val 划分，
泛化能力另行用留出形状评测。

**(3) 跨数量级的特征先取对数。**
若某些特征跨好几个数量级（如 `A.nbytes` 从 1e5 到 3.4e7），在 MLP 前加
`log1p`（图内），否则跨形状内插会非常差。注意 `log1p` 要**只作用在那些列**上。

**上界对照**：至少报三个基线 —— 全局均值、**每个形状自身的标签均值**
（oracle，代表"只用形状信息"的信息上界）、Ridge/log 线性。
若 MLP 赢不过"形状均值"，说明模型没学到参数效应，别急着上调优。

**泛化估计**：除了固定的留出形状，再跑一次 **leave-one-shape-out CV**
（每次留一个训练形状），给出的是"完全没见过的形状"上的分布而非单点。

---

## 5. 评测：一定要有一个**独立仪器**

用 `adtuner` 自己的 `time` 训练、又用 `adtuner` 自己的 `time` 评测，
只能证明"自洽"，不能证明"对"。所以要另找一条测量通路。本项目的做法：

* 独立仪器 = 宿主程序 + `srun`，`--warmup 3 --repeat 20 --verify`
  （复用项目已有的 `run_retest.py` 的 `benchmark()`，不新写一套）；
* **无偏样本**必须专门生成：从合法空间均匀随机不放回抽 48 个配置，
  **不要**让"模型预测最优"混进这批（会让 MAPE 系统性乐观）；
* **模型引导的结论**用同一台独立仪器实测模型全局 top-k（例如 top-16）——
  这样"模型引导能达到什么水平"就**不依赖解析 ADTuner 的 stdout**；
* 若还想跑真实 `runner_strategy: 2` 端到端，注意它的实测值只在 stdout 里
  （见 §1 第 1 点），解析后要与上面那批独立实测**互相校验**。

三个口径分开报告，不许混：

| 口径 | 含义 |
|---|---|
| in-sample | 同形状同区间的拟合误差，**不是泛化指标** |
| out-of-sample（固定留出形状 / LOSO） | 真正的泛化 |
| `mape_after_median_alignment` | 用实测中位比值做 oracle 对齐后的 MAPE，用于把"绝对量级偏差"与"排序/趋势"分开看 —— **必须标注是 oracle 口径** |

---

## 6. 预算规划

单次评估成本 = 编译 + 计时（MT3000 实测：`compile_time≈1.23 s` +
`benchmark_time≈0.17 s`）。此外**每次 `adtuner-run` 调用**还有固定开销
（排队 + 启动 + 建缓存目录，MT3000 约 **60 s**）：

```
总时长 ≈ 运行次数 × 60 s  +  总评估条数 × 1.5 s
```

本项目：10 形状 × 3 运行 × 60 s + 2260 × 1.5 s ≈ 30 + 57 = **87 分钟**。
（只按"条数×1.5 s"估会低估 30%。）

---

## 7. 本变体的检查清单（替代 SKILL.md 的 Phase 2/3 项）

```
[ ] 确认平台确实没有可用的采集工具（或它无法表达目标内核）
[ ] 查源码锚定 cache `time` 的口径（聚合方式 / 是否含 memcpy / iterations）
[ ] 量出 cache `time` 与独立仪器之间的偏差 + 独立仪器自身的重复性
[ ] 数据集只取 runner_strategy=0 的 cache；strategy=2 的 cache 只含预测值
[ ] 采样用 A/B/C 三段式；random_sample 用 fraction 而非 max_fevals
[ ] 每个形状不同种子；划分按形状分组
[ ] ONNX 图自包含（归一化 + 逆变换都在图内）；doc_string 写 model_type
[ ] 导出后用 adtuner 自己的 performancePredictor 做契约测试
[ ] 基线 ≥3 个（含形状均值 oracle）；另给 LOSO CV
[ ] 评测用独立仪器 + 专门生成的无偏样本（不含模型预测最优）
[ ] 模型引导 top-k 由独立仪器实测（不依赖解析 stdout）
[ ] 报告里区分 in-sample / out-of-sample / oracle 对齐三种口径
```
