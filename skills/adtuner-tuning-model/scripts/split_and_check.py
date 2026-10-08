#!/usr/bin/env python
"""Phase 3 前置 —— 切分训练/留出集, 并校验特征向量与原始 config 逐位一致。

这是**最重要的一道防线**: 如果训练数据的特征向量与 adtuner 推理时的特征向量
不一致 (尺寸比例不同), 模型对真实问题就是分布外的, 预测会差 2~3 倍。

用法:
    <adtuner-env>/python split_and_check.py <config.yaml> --workdir <dir>
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import yaml

DT_SZ = {"int32": 4, "int": 4, "int64": 8, "double": 8, "float": 4}


def features_from_config(cfg, block, M):
    """按 adtuner core.get_features + predictor.predict 的顺序构造特征向量。

    顺序: [grid_x, grid_y, grid_z, block_x, block_y, block_z, workload_0..N]
      - 指针 -> shape 元素数 x sizeof(type) (等价于 ndarray.nbytes)
      - 标量 -> 原值
    行指针 (shape[0]==problem_size+1) 与等于 problem_size 的标量随 M 缩放。
    """
    P = int(cfg["kernel"]["problem_size"])
    wl = []
    for inp in cfg["inputs"]:
        if isinstance(inp.get("shape"), list):
            n = int(np.prod(inp["shape"]))
            # 行指针随 M 缩放
            if n == P + 1:
                n = M + 1
            elif n == P:
                n = M
            wl.append(float(n * DT_SZ[inp["type"]]))
        else:
            v = inp.get("value", 0)
            if isinstance(v, (int, float)) and int(v) == P:
                v = M
            wl.append(float(v))
    return [int(np.ceil(M / block)), 1, 1, block, 1, 1] + wl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--workdir", required=True)
    a = ap.parse_args()

    W = os.path.abspath(a.workdir)
    cfg = yaml.safe_load(open(a.config))
    lv = json.load(open(f"{W}/levels.json"))
    hold = int(lv["holdout"])

    df = pd.read_csv(f"{W}/dataset.csv")
    print(f"总行数 = {len(df)}")
    print("各负载行数:")
    print(df.groupby("workload_0").size().to_string())

    tr = df[df["workload_0"] != hold].copy()
    ho = df[df["workload_0"] == hold].copy()
    if len(ho) == 0:
        raise SystemExit(f"❌ 找不到留出负载 {hold} 的数据")
    tr.to_csv(f"{W}/dataset_train.csv", index=False)
    ho.to_csv(f"{W}/dataset_holdout.csv", index=False)
    print()
    print(f"训练集: {len(tr)} 行  负载 {sorted(tr['workload_0'].unique().astype(int))}")
    print(f"留出集: {len(ho)} 行  负载 {sorted(ho['workload_0'].unique().astype(int))}")

    # ---------- 特征一致性校验 ----------
    cols = [c for c in df.columns if c != "time(ms)"]
    P = int(cfg["kernel"]["problem_size"])
    blocks = [int(b) for b in lv["blocks"]]
    b0 = blocks[0]
    g = df[(df["workload_0"] == P) & (df["block_size_x"] == b0)]
    if len(g) == 0:
        print(f"\n⚠️ 训练集里没有 problem_size={P} 这一档, 无法做一致性校验。")
        print("   建议在 Phase 1 让训练负载包含 problem_size 本身。")
        return
    row = g[cols].values[0]
    ref = features_from_config(cfg, b0, P)

    names = (["grid_x", "grid_y", "grid_z", "block_x", "block_y", "block_z"]
             + [f"workload_{i}" for i in range(len(ref) - 6)])
    print()
    print("=" * 82)
    print(f"特征一致性校验  (problem_size={P}, block={b0})")
    print("=" * 82)
    print(f"{'特征':<14} {'我的数据集':>18} {'adtuner 推理时':>18} {'一致':>6}")
    print("-" * 82)
    same = True
    for n, u, v in zip(names, row, ref):
        ok = abs(float(u) - float(v)) < 1e-6
        same &= ok
        print(f"{n:<14} {float(u):>18.0f} {float(v):>18.0f} "
              f"{'是' if ok else '否 <<<':>6}")
    print("-" * 82)
    if same:
        print("✅ 全部一致 —— 模型对真实问题是分布内的")
    else:
        print("❌ 存在不一致! 训练数据与推理特征不匹配,")
        print("   请回到 Phase 1 检查尺寸比例 / 换用前缀缩放重建数据。")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
