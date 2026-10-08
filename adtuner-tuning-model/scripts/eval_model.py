#!/usr/bin/env python
"""Phase 3 后 —— 评估模型: 分负载精度/排序 + 退化检测 + 留出泛化。

用法:
    <adtuner-env>/python eval_model.py --workdir <dir> [--model <model.onnx>]
"""
import argparse
import json
import os

import numpy as np
import onnxruntime as ort
import pandas as pd


def spearman(a, b):
    n = len(a)
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return 1 - 6 * np.sum((ra - rb) ** 2) / (n * (n * n - 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--model", default=None)
    a = ap.parse_args()

    W = os.path.abspath(a.workdir)
    model = a.model or f"{W}/model.onnx"
    if not os.path.isfile(model):
        raise SystemExit(f"❌ 找不到模型: {model}")

    lv = json.load(open(f"{W}/levels.json"))
    P, hold = int(lv["problem_size"]), int(lv["holdout"])

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    opts.log_severity_level = 3
    sess = ort.InferenceSession(model, sess_options=opts, providers=["CPUExecutionProvider"])
    iname, oname = sess.get_inputs()[0].name, sess.get_outputs()[0].name

    def predict(X):
        return np.array([float(np.array(sess.run([oname], {iname: X[i:i + 1]})[0]).flatten()[0])
                         for i in range(len(X))])

    frames = []
    for tag, fn in [("训练", "dataset_train.csv"), ("留出", "dataset_holdout.csv")]:
        p = f"{W}/{fn}"
        if os.path.isfile(p):
            d = pd.read_csv(p)
            d["_split"] = tag
            frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    cols = [c for c in df.columns if c not in ("time(ms)", "_split")]
    df["pred"] = predict(df[cols].values.astype(np.float32))

    print("=" * 92)
    print(f"模型评估: {os.path.basename(model)}   样本 {len(df)}")
    print("=" * 92)

    # ---- 退化检测 ----
    nun = df["pred"].nunique()
    base = float(np.mean(np.abs((df["time(ms)"].mean() - df["time(ms)"]) / df["time(ms)"])) * 100)
    mape = float(np.mean(np.abs((df["pred"] - df["time(ms)"]) / df["time(ms)"])) * 100)
    print(f"整体 MAPE = {mape:.2f}%      常数基线 = {base:.2f}%      "
          f"预测唯一值个数 = {nun} / {len(df)}")
    if nun == 1:
        print("❌ **退化为常数预测器** —— 检查 model_type 是否用了 PolynomialLasso;")
        print("   应改用 MLPModel。")
    elif mape >= base:
        print("⚠️ 不比常数基线好 —— 数据可能缺少有效自由度或样本太少。")
    else:
        print("✅ 优于常数基线")

    print()
    print(f"{'负载 M':>10} {'类型':>5} {'n':>3} {'MAPE':>8} {'Spearman':>9} "
          f"{'真最优':>7} {'预测最优':>8} {'命中':>5} {'top3重合':>9}")
    print("-" * 92)
    hit = tot = 0
    for M, g in df.groupby("workload_0"):
        g = g.sort_values("block_size_x")
        t, p = g["time(ms)"].values, g["pred"].values
        if len(g) < 2:
            continue
        mp = float(np.mean(np.abs((p - t) / t)) * 100)
        r = spearman(p, t)
        b = g["block_size_x"].values
        tb, pb = int(b[np.argmin(t)]), int(b[np.argmin(p)])
        t3 = set(b[np.argsort(t)[:3]])
        p3 = set(b[np.argsort(p)[:3]])
        ok = tb == pb
        hit += ok
        tot += 1
        kind = "留出" if int(M) == hold else "训练"
        print(f"{int(M):>10,} {kind:>5} {len(g):>3} {mp:>7.1f}% {r:>+9.3f} "
              f"{tb:>7} {pb:>8} {'是' if ok else '否':>5} {len(t3 & p3):>6}/3")
    print("-" * 92)
    print(f"最优 block 命中 = {hit}/{tot}")

    # ---- 大负载子集（模型真正要用的区间） ----
    big = df[df["workload_0"] >= P / 2]
    if len(big):
        m2 = float(np.mean(np.abs((big["pred"] - big["time(ms)"]) / big["time(ms)"])) * 100)
        print()
        print(f"⚠️ 只看负载 >={P//2:,} 的区间 (模型实际使用区间): MAPE = {m2:.2f}%")

    print()
    print("提示: 若某些负载 MAPE 很高但 Spearman 也低, 先检查该负载的内核耗时量级 ——")
    print("      内核 < 50us 时配置间真实差异低于测量分辨力, 排序本身即噪声。")


if __name__ == "__main__":
    main()
