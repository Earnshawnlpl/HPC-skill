#!/usr/bin/env python
"""Phase 4 —— 解析 adtuner 验证作业输出 (strategy=0 真值 + strategy=2 模型调优), 给出结论。

用法:
    <adtuner-env>/python analyze_adtuner.py <job.out> [--topk 5]

期望 job.out 含两段 (templates/verify.slurm 会产生):
    ########## [1/2] runner_strategy = 0 ...
    ########## [2/2] runner_strategy = 2 ...
"""
import argparse
import re
import statistics

PAT = re.compile(r"block_size_x=(\d+),\s*time=([\d.]+)ms")


def split_out(txt):
    i1, i2 = txt.find("[1/2]"), txt.find("[2/2]")
    if i1 < 0 or i2 < 0:
        raise SystemExit("❌ 找不到 [1/2] / [2/2] 分段标记, 请确认 job.out 来自 "
                         "templates/verify.slurm")
    return txt[i1:i2], txt[i2:]


def parse(section):
    """返回 (phase1 预测 或 None, 实测) —— 截掉结尾的 best performing configuration 行。"""
    best = section.find("best performing configuration:")
    body = section[:best] if best >= 0 else section
    items = [(m.start(), int(m.group(1)), float(m.group(2))) for m in PAT.finditer(body)]
    items.sort()
    mk = body.find("New Searchspace created with")
    if mk < 0:
        return None, [(b, t) for _, b, t in items]
    return ([(b, t) for p, b, t in items if p < mk],
            [(b, t) for p, b, t in items if p > mk])


def rho(a, b, keys):
    n = len(keys)
    ra = {k: i for i, k in enumerate(sorted(keys, key=lambda x: a[x]))}
    rb = {k: i for i, k in enumerate(sorted(keys, key=lambda x: b[x]))}
    return 1 - 6 * sum((ra[k] - rb[k]) ** 2 for k in keys) / (n * (n * n - 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobout")
    ap.add_argument("--topk", type=int, default=5)
    a = ap.parse_args()

    txt = open(a.jobout, errors="replace").read()
    gt_sec, tk_sec = split_out(txt)
    _, gt = parse(gt_sec)
    pr_list, tk = parse(tk_sec)
    true = dict(gt)
    pred = dict(pr_list or [])

    print("=" * 82)
    print("数据规模")
    print("=" * 82)
    print(f"  strategy=0 实测配置数 : {len(true)}")
    print(f"  模型预测配置数        : {len(pred)}")
    print(f"  strategy=2 复测配置数 : {len(tk)}")

    if not true:
        raise SystemExit("❌ strategy=0 段没有解析到实测值")

    k = len(tk) or a.topk
    t_top = [b for b, _ in sorted(true.items(), key=lambda x: x[1])[:k]]
    p_top = [b for b, _ in sorted(pred.items(), key=lambda x: x[1])[:k]] if pred else []

    print()
    print("=" * 82)
    print(f"top-{k} 集合对比")
    print("=" * 82)
    print(f"  实测 top-{k} : " + "  ".join(f"{b}({true[b]:.4f})" for b in t_top))
    if p_top:
        print(f"  预测 top-{k} : " + "  ".join(f"{b}({pred[b]:.4f})" for b in p_top))
        inter = sorted(set(t_top) & set(p_top))
        print(f"  交集        : {inter}   ->  {len(inter)}/{k} 重合")
        print(f"  预测漏掉    : {sorted(set(t_top) - set(p_top))}")
        print(f"  预测多选    : {sorted(set(p_top) - set(t_top))}")
        rank = sorted(pred, key=lambda x: pred[x]).index(t_top[0]) + 1
        print(f"  真最优 {t_top[0]} 在预测中排名: 第 {rank} 名  "
              f"-> {'已选入 top-k ✅' if t_top[0] in p_top else '未选入 ❌'}")

    if pred:
        common = sorted(set(true) & set(pred))
        errs = [abs(pred[b] - true[b]) / true[b] * 100 for b in common]
        rr = [true[b] / pred[b] for b in common]
        print()
        print("=" * 82)
        print(f"逐配置 预测 vs 实测  (n={len(common)})")
        print("=" * 82)
        for b in common:
            print(f"  block={b:>5}  预测 {pred[b]:>8.4f}  实测 {true[b]:>8.4f}  "
                  f"{(pred[b]-true[b])/true[b]*100:>+8.1f}%")
        print(f"  MAPE = {statistics.mean(errs):.1f}%   中位 = {statistics.median(errs):.1f}%")
        print(f"  系统性偏差 实测/预测 = {statistics.mean(rr):.2f}")
        print(f"  Spearman ρ = {rho(pred, true, common):+.3f}")

    if tk:
        print()
        print("=" * 82)
        print("strategy=2 最终结论")
        print("=" * 82)
        best = min(tk, key=lambda x: x[1])
        opt = min(true, key=lambda x: true[x])
        print(f"  strategy=2 报出的最优 : block={best[0]}, {best[1]:.4f} ms")
        print(f"  全空间真实最优        : block={opt}, {true[opt]:.4f} ms")
        d = (true[best[0]] - true[opt]) / true[opt] * 100
        print(f"  相对真实最优          : {d:+.2f}%")
        if abs(d) < 5:
            print("  -> ✅ 命中")
        elif opt in [b for b, _ in tk]:
            print("  -> ⚠️ 真实最优在复测集合内, 但复测排序把它排后了 (测量噪声)")
        else:
            print("  -> ❌ 真实最优不在预测 top-k 内 (模型排序问题)")


if __name__ == "__main__":
    main()
