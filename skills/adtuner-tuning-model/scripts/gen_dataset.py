#!/usr/bin/env python
"""Phase 1 —— 从真实数据生成多负载训练数据, 并写出 cpm 采集配置与作业脚本。

**核心原则（与具体内核无关）**
    训练数据必须与目标 config 的输入【结构同构】，否则模型是分布外的。
    最稳的做法是【前缀缩放】: 逐字节沿用真实数组, 只按问题规模 M' 截断,
    这样在 M' = problem_size 处与原数据完全一致。

    反例（踩过的坑）: 用随机数自造同形状数据会改变
      - 数组相对 problem_size 的长度比例
      - 索引数组的取值范围 → 访存/缓存行为
    两者都会让模型学到的规律不适用（实测见过 MAPE 264%）。

**角色（决定每个输入怎么缩放）** —— 自动推断，可用 --roles 覆盖:

    rowptr   长度 = problem_size+1 的整型行指针        -> 截到 M'+1
    csr      长度 = rowptr[P]-1 的配套数组             -> 截到 nnz(M')
    perrow   长度 = problem_size                       -> 截到 M'
    prop    长度 = k×problem_size (2<=k<=4)            -> 截到 round(k*M')
    fixed   其余（例如独立于问题规模的工作区数组）      -> 保持全长, 各负载共用

    无行指针的内核（如 vecadd / GEMM 类）自动进入 no-CSR 模式，
    此时 perrow / prop / fixed 仍然可用。

用法:
    <adtuner-env>/python gen_dataset.py <config.yaml> --workdir <dir> [选项]

选项:
    --profile <yaml>    平台档案（默认 profiles/sugon8000.yaml）
    --levels N          训练负载个数（默认 6），最后一个 = problem_size
    --holdout-frac F    留出负载比例（默认取相邻两个训练比例的中间值）
    --blocks "32,64,…"  线程配置列表
    --kernel PATH       覆盖内核路径
    --launch-bounds N   给内核副本加 __launch_bounds__(N)（0=不处理）
    --roles a=csr,b=fixed   手工指定角色（覆盖自动推断）
"""
import argparse
import json
import math
import os
import re
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import profile as prof_mod  # noqa: E402

DT = {"int32": np.int32, "int": np.int32, "int64": np.int64, "int16": np.int16,
      "short": np.int16, "uint32": np.uint32,
      "double": np.float64, "float": np.float32, "float64": np.float64,
      "half": np.float16}
SZ = {"int32": 4, "int": 4, "int64": 8, "int16": 2, "short": 2, "uint32": 4,
      "double": 8, "float": 4, "float64": 8, "half": 2}
ISINT = {"int32", "int", "int64", "int16", "short", "uint32"}


def read_input(inp):
    src = inp.get("source", {})
    dt = DT.get(inp["type"])
    if dt is None:
        return None
    n = int(np.prod(inp["shape"]))
    p = src.get("path")
    if not p or not os.path.isfile(p):
        return None
    return np.fromfile(p, dtype=dt, count=n, offset=int(src.get("offset", 0)))


def make_kernel(src, out, lb):
    txt = open(src).read()
    if "__launch_bounds__" in txt:
        print(f"  内核已自带 __launch_bounds__，沿用原文件")
        return src
    new = re.sub(r"__global__\s+void\s+",
                 f"__global__ void __launch_bounds__({lb}) ", txt, count=1)
    if new == txt:
        print("  ⚠️ 未找到 __global__ void，无法插入 launch_bounds，沿用原内核")
        return src
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write(new)
    print(f"  已生成内核副本（含 __launch_bounds__({lb})）")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--profile", default=None)
    ap.add_argument("--levels", type=int, default=6)
    ap.add_argument("--holdout-frac", type=float, default=None)
    ap.add_argument("--blocks", default="32,64,96,128,160,192,224,256,320,384,448,512")
    ap.add_argument("--kernel", default=None)
    ap.add_argument("--launch-bounds", type=int, default=None)
    ap.add_argument("--roles", default="")
    a = ap.parse_args()

    prof = prof_mod.load_profile(a.profile)
    lb = a.launch_bounds
    if lb is None:
        lb = int((prof.get("kwargs") or {}).get("launch_bounds", 0) or 0)

    cfg = yaml.safe_load(open(a.config))
    k = cfg["kernel"]
    P = int(k["problem_size"])
    inputs = cfg["inputs"]
    W = os.path.abspath(a.workdir)
    DATA = f"{W}/data"
    os.makedirs(DATA, exist_ok=True)
    blocks = [int(x) for x in a.blocks.split(",")]
    KEYS = ["cw", "cn", "ca", "cf", "cc", "cb", "cg", "ch", "ci"]
    KEY = {i: (KEYS[i] if i < len(KEYS) else f"k{i}") for i in range(len(inputs))}

    print("=" * 78)
    print(f"平台档案: {prof['name']}  ({prof.get('description','')})")
    print(f"  文件    : {prof['_path'] or '(内置缺省)'}")
    print(f"  调度器  : {(prof['scheduler'] or {}).get('type')}  "
          f"分区={(prof['scheduler'] or {}).get('partition','-')}")
    print("=" * 78)
    print(f"内核: {k['name']}   problem_size = {P:,}")
    print()

    # ---------- 读取真实数据 ----------
    arrays, scalars, shapes = {}, {}, {}
    for i, inp in enumerate(inputs):
        if isinstance(inp.get("shape"), list):
            arrays[i] = read_input(inp)
            shapes[i] = int(np.prod(inp["shape"]))
            tag = "" if arrays[i] is not None else "  ⚠️ 源文件缺失"
            print(f"  [{i}] {inp['name']:<14} {inp['type']:<8} n={shapes[i]:>12,}  "
                  f"({shapes[i]/P:.2f} x P){tag}")
        else:
            scalars[i] = inp.get("value")
            print(f"  [{i}] {inp['name']:<14} 标量 = {inp.get('value')}")

    missing = [inputs[i]["name"] for i, v in arrays.items() if v is None]
    if missing:
        raise SystemExit(
            f"\n❌ 以下输入的源文件缺失，无法做前缀缩放: {missing}\n"
            "   前缀缩放必须读取真实数据；请先修正 config.yaml 里的 source.path。")

    # ---------- 角色推断 ----------
    roles = {}
    rowptr_i = None
    for i, arr in arrays.items():
        if inputs[i]["type"] in ISINT and len(arr) == P + 1 and np.all(np.diff(arr) >= 0):
            rowptr_i = i
            break
    nnz_full = int(arrays[rowptr_i][P]) - 1 if rowptr_i is not None else None

    for i, arr in arrays.items():
        L = len(arr)
        if i == rowptr_i:
            roles[i] = "rowptr"
        elif L == P:
            roles[i] = "perrow"
        elif nnz_full is not None and L == nnz_full:
            roles[i] = "csr"
        elif L > P and L % P == 0 and 2 <= L // P <= 4:
            roles[i] = "prop"
        else:
            roles[i] = "fixed"
    for kv in filter(None, a.roles.split(",")):
        nm, rl = kv.split("=")
        hit = False
        for i, inp in enumerate(inputs):
            if inp["name"] == nm:
                roles[i] = rl
                hit = True
        if not hit:
            raise SystemExit(f"❌ --roles 里的输入名不存在: {nm}")

    mode = "CSR" if rowptr_i is not None else "无行指针(元素级)"
    print()
    print(f"  结构模式: {mode}")
    if nnz_full is not None:
        print(f"  nnz(全量) = {nnz_full:,}   nnz/行 = {nnz_full/P:.2f}")
    print("  角色推断（可用 --roles name=role 覆盖）:")
    for i in arrays:
        extra = f"   k={len(arrays[i])//P}" if roles[i] == "prop" else ""
        print(f"    {inputs[i]['name']:<14} -> {roles[i]:<8}{extra}")
    if all(r == "fixed" for r in roles.values()):
        raise SystemExit("❌ 所有数组都被判为 fixed，无法构造更小的问题。"
                         "请用 --roles 指定至少一个 rowptr/perrow/prop 数组。")

    # ---------- 规模设计 ----------
    fracs = sorted({round(x, 4) for x in np.linspace(1.0 / 8, 1.0, a.levels)})
    if 1.0 not in fracs:
        fracs.append(1.0)
    hf = a.holdout_frac if a.holdout_frac is not None else round((fracs[-1] + fracs[-2]) / 2, 4)
    trainM = sorted({int(round(f * P)) for f in fracs})
    holdM = int(round(hf * P))
    if holdM in trainM:
        holdM += 1
    allM = sorted(trainM + [holdM])

    print()
    print("=" * 78)
    print(f"规模设计: 训练 {len(trainM)} + 留出 1, x {len(blocks)} 线程 "
          f"= {len(allM)*len(blocks)} 次采集")
    print("=" * 78)
    print(f"  训练: {trainM}")
    print(f"  留出: {holdM}   （不参与训练，用于泛化测试）")

    # ---------- 生成前缀数据 ----------
    meta, shared = {}, {}
    print()
    print("生成前缀数据 ...")
    for M in allM:
        d = f"{DATA}/M{M}"
        os.makedirs(d, exist_ok=True)
        nnz = int(arrays[rowptr_i][M]) - 1 if rowptr_i is not None else None
        info = {}
        for i, inp in enumerate(inputs):
            if i not in arrays:
                continue
            arr, r, key = arrays[i], roles[i], KEY[i]
            ext = "i32" if inp["type"] in ISINT else "f64"
            if r == "rowptr":
                seg = arr[: M + 1]
            elif r == "perrow":
                seg = arr[:M]
            elif r == "csr":
                seg = arr[:nnz]
            elif r == "prop":
                seg = arr[: max(1, round(len(arr) / P * M))]
            else:  # fixed
                if key not in shared:
                    fp = f"{DATA}/{key}_full.{ext}"
                    arr.tofile(fp)
                    shared[key] = (fp, len(arr))
                info[i] = ("shared", key)
                continue
            fp = f"{d}/{key}.{ext}"
            seg.tofile(fp)
            info[i] = ("local", fp, len(seg))
        meta[M] = info
        rng = ""
        if nnz is not None:
            for i in arrays:
                if roles[i] == "csr" and inputs[i]["type"] in ISINT:
                    rng = (f"  索引范围[{int(arrays[i][:nnz].min())},"
                           f"{int(arrays[i][:nnz].max())}]")
        nnzs = f"nnz={nnz:>11,}" if nnz is not None else "nnz=        -"
        print(f"  M={M:>10,}  {nnzs}{rng}")

    # ---------- 结构自检 ----------
    print()
    print("=" * 78)
    print("结构自检")
    print("=" * 78)
    ok = True
    for M in allM:
        checks = {}
        if rowptr_i is not None:
            rp = arrays[rowptr_i][: M + 1]
            nnz = int(rp[M]) - 1
            checks["行指针单调"] = bool(np.all(np.diff(rp) >= 0))
            for i in arrays:
                if roles[i] == "csr" and inputs[i]["type"] in ISINT:
                    mx = int(arrays[i][:nnz].max())
                    ln = shared[KEY[i]][1] if KEY[i] in shared else nnz
                    checks[f"{inputs[i]['name']}索引不越界"] = mx <= ln
        if not checks:
            checks["可生成更小问题"] = True
        good = all(checks.values())
        ok &= good
        print(f"  M={M:>10,}  " + "  ".join(f"{n}={'✓' if v else '✗'}"
                                           for n, v in checks.items()))

    # ---- 目标规模一致性（最关键的自检）: M'=P 时写出的文件必须与原始数组逐字节一致 ----
    print()
    print("目标规模 M'=P 一致性（读回文件、与原始数组逐字节比对）")
    Mp = P
    ident = True
    for i, arr in arrays.items():
        r, key = roles[i], KEY[i]
        ext = "i32" if inputs[i]["type"] in ISINT else "f64"
        if r == "fixed":
            fp, _ = shared[key]
            want = arr
        elif r == "rowptr":
            fp, want = f"{DATA}/M{Mp}/{key}.{ext}", arr[: Mp + 1]
        elif r == "perrow":
            fp, want = f"{DATA}/M{Mp}/{key}.{ext}", arr[:Mp]
        elif r == "csr":
            nnz = int(arrays[rowptr_i][Mp]) - 1
            fp, want = f"{DATA}/M{Mp}/{key}.{ext}", arr[:nnz]
        else:  # prop
            fp = f"{DATA}/M{Mp}/{key}.{ext}"
            want = arr[: max(1, round(len(arr) / P * Mp))]
        got = np.fromfile(fp, dtype=DT[inputs[i]["type"]])
        same = got.shape == want.shape and bool(np.array_equal(got, want))
        ident &= same
        print(f"  {inputs[i]['name']:<14} {r:<8} {len(want):>12,} 元素   "
              f"{'一致 ✓' if same else '不一致 ✗'}")
    ok &= ident
    print(f"  总体: {'✅ 通过' if ok else '❌ 有问题（检查 --roles）'}")
    if not ok:
        raise SystemExit(2)

    # ---------- 写 cpm 采集配置 ----------
    kern = a.kernel or k["path"]
    if lb and max(blocks) > 256:
        base = os.path.basename(kern).rsplit(".", 1)[0]
        kern = make_kernel(kern, f"{W}/kernels/{base}_lb{lb}.cpp", lb)

    def mk_inputs(M):
        out = []
        for i, inp in enumerate(inputs):
            if i not in arrays:
                v = scalars[i]
                if isinstance(v, (int, float)) and int(v) == P:
                    v = M
                out.append({"type": inp["type"], "value": v})
                continue
            kind = meta[M][i]
            if kind[0] == "shared":
                fp, ln = shared[kind[1]]
            else:
                _, fp, ln = kind
            out.append({"type": inp["type"], "size": ln, "value": fp})
        return out

    clist = [{"inputs": mk_inputs(M),
              "tune_parameters": [{"block_size": [b, 1, 1],
                                   "grid_size": [math.ceil(M / b), 1, 1]}
                                  for b in blocks]}
             for M in allM]

    jcfg = {"task": "collect_kernel_data", "kernel_name": k["name"],
            "kernel_file_path": kern, "bbc": False,
            "config_list": clist, "result_file_path": "dataset.csv"}
    with open(f"{W}/collect.json", "w") as f:
        json.dump(jcfg, f, indent=2)
    with open(f"{W}/levels.json", "w") as f:
        json.dump({"problem_size": P, "train": trainM, "holdout": holdM,
                   "blocks": blocks, "roles": {inputs[i]["name"]: roles[i]
                                               for i in roles},
                   "n_configs": len(clist) * len(blocks)}, f, indent=2)
    with open(f"{W}/collect.slurm", "w") as f:
        f.write(prof_mod.render_collect_slurm(W, prof))

    print()
    print("=" * 78)
    print("产出")
    print("=" * 78)
    print(f"  {W}/collect.json    采集配置（{len(clist)} 负载 x {len(blocks)} 线程 "
          f"= {len(clist)*len(blocks)} 次）")
    print(f"  {W}/levels.json     规模划分 + 角色（供后续脚本自动识别）")
    print(f"  {W}/collect.slurm   采集作业脚本（按平台档案渲染）")
    print(f"  内核: {kern}")
    print(f"  grid 规则: ceil(problem_size / block_size)  —— 必须与 adtuner 一致")
    print()
    print(f"  下一步: sbatch {W}/collect.slurm")


if __name__ == "__main__":
    main()
