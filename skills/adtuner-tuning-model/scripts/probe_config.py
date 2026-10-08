#!/usr/bin/env python
"""Phase 0 —— 解析 ADTuner config.yaml, 打印内核/输入/调优空间与"尺寸比例指纹"。

与内核、平台无关：只依赖 adtuner 的 config.yaml 结构。

用法:
    <python-with-yaml> probe_config.py <config.yaml> [--profile <yaml>]

注意: 需要 pyyaml。若采集环境（cpm 所在环境）没有 yaml，
      用 adtuner 环境的 python，或任意带 pyyaml 的解释器即可。
"""
import argparse
import os
import sys

import numpy as np
import yaml


def fmt(n):
    return f"{n:,}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", help="adtuner config.yaml")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    k = cfg["kernel"]
    inputs = cfg["inputs"]
    tuning = cfg.get("tuning", {})

    print("=" * 78)
    print("内核")
    print("=" * 78)
    print(f"  name          : {k['name']}")
    print(f"  path          : {k['path']}")
    print(f"  lang          : {k.get('lang')}")
    print(f"  problem_size  : {k['problem_size']}")

    P = int(k["problem_size"])
    scalars, pointers = [], []
    for inp in inputs:
        if isinstance(inp.get("shape"), list):
            pointers.append(inp)
        else:
            scalars.append(inp)

    print()
    print("=" * 78)
    print("输入参数 (顺序即内核形参顺序, 必须严格保持)")
    print("=" * 78)
    for i, inp in enumerate(inputs):
        if isinstance(inp.get("shape"), list):
            n = int(np.prod(inp["shape"]))
            src = inp.get("source", {})
            tag = "指针"
            print(f"  [{i}] {inp['name']:<12} {tag} {inp['type']:<9} shape={inp['shape']} "
                  f"n={fmt(n)}")
            print(f"        source.type={src.get('type')}  path={src.get('path')}")
        else:
            print(f"  [{i}] {inp['name']:<12} 标量 {inp['type']:<9} value={inp.get('value')}")

    print()
    print("=" * 78)
    print("尺寸比例指纹  (Phase 3 的特征一致性校验必须以此为准)")
    print("=" * 78)
    print(f"  problem_size = {fmt(P)}")
    print(f"  {'输入':<12} {'元素数':>14} {'/ problem_size':>16}")
    for inp in pointers:
        n = int(np.prod(inp["shape"]))
        print(f"  {inp['name']:<12} {fmt(n):>14} {n / P:>16.2f}")

    # 关键提示: f 类数组的比例与 ne 的取值范围
    print()
    print("  ⚠️ 生成训练数据时, 以上每个比例都必须逐位复现。")
    print("     尤其是 '远大于 problem_size' 的数组 (如上表中的 14.00) ——")
    print("     这类数组若在训练数据里被改成 = problem_size, 模型会严重失配。")

    # ne 取值范围检查
    print()
    print("=" * 78)
    print("指针输入的取值范围 (决定访存行为, 也必须复现)")
    print("=" * 78)
    for inp in pointers:
        src = inp.get("source", {})
        path = src.get("path")
        if not path or not os.path.isfile(path):
            print(f"  {inp['name']:<12} (无源文件或不存在, 跳过)")
            continue
        dt = np.dtype({"int32": np.int32, "int": np.int32, "double": np.float64,
                       "float": np.float32, "int64": np.int64}[inp["type"]])
        n = int(np.prod(inp["shape"]))
        arr = np.fromfile(path, dtype=dt, count=n, offset=int(src.get("offset", 0)))
        if np.issubdtype(dt, np.integer):
            print(f"  {inp['name']:<12} min={arr.min():<12} max={arr.max():<12} "
                  f"唯一值={len(np.unique(arr))}")
        else:
            print(f"  {inp['name']:<12} min={arr.min():<12.4f} max={arr.max():<12.4f}")

    print()
    print("=" * 78)
    print("调优空间")
    print("=" * 78)
    for kk, vv in (tuning.get("parameters") or {}).items():
        print(f"  {kk} = {vv}")
    mf = tuning.get("modelfit", {}) or {}
    print(f"  strategy          : {tuning.get('strategy')}")
    print(f"  iterations        : {tuning.get('iterations')}")
    print(f"  modelfit.enabled  : {mf.get('enabled')}")
    print(f"  modelfit.model    : {mf.get('model_path')}")
    print(f"  runner_strategy   : {mf.get('runner_strategy')}")
    print(f"  topk              : {mf.get('topk')}")

    print()
    print("=" * 78)
    print("下一步")
    print("=" * 78)
    print("  Phase 1: gen_dataset.py  用上面的比例做【真实数据前缀缩放】生成多负载数据")
    print("           —— 不要自己造合成 CSR")


if __name__ == "__main__":
    sys.exit(main())
