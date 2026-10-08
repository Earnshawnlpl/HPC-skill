#!/usr/bin/env python
"""Phase 4 准备 —— 生成 adtuner 验证所需的两份 config.yaml 与作业脚本。

  gt/config.yaml   : runner_strategy=0  全空间实测 = 真值
  topk/config.yaml : runner_strategy=2  模型预测 top-k 再实测

两份除 runner_strategy / cache 外完全一致。

⚠️ cache 文件名必须不同且跑前清空 —— adtuner 的 model_predict 会优先读 cache，
   若 cache 里是实测值，phase-1 的"预测"就变成了读实测值，对照实验失效。

用法:
    <adtuner-env>/python make_verify.py <config.yaml> --workdir <dir> --model <model.onnx>
                                        [--profile <yaml>] [--kernel <带 launch_bounds 的副本>]
"""
import argparse
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import profile as prof_mod  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--profile", default=None)
    ap.add_argument("--kernel", default=None,
                    help="覆盖内核路径（例如 Phase 1 生成的 __launch_bounds__ 副本）")
    a = ap.parse_args()

    prof = prof_mod.load_profile(a.profile)
    W = os.path.abspath(a.workdir)
    cfg = yaml.safe_load(open(a.config))
    if a.kernel:
        cfg["kernel"]["path"] = os.path.abspath(a.kernel)

    tun = cfg.setdefault("tuning", {})
    mf = tun.setdefault("modelfit", {})
    mf["enabled"] = True
    mf["model_path"] = os.path.abspath(a.model)

    print(f"平台档案: {prof['name']}  ({prof.get('description','')})")
    for sub, rs, cache in [("gt", 0, "gt_cache.json"), ("topk", 2, "topk_cache.json")]:
        d = f"{W}/{sub}"
        os.makedirs(d, exist_ok=True)
        c = yaml.safe_load(yaml.safe_dump(cfg))          # 深拷贝
        c["tuning"]["modelfit"]["runner_strategy"] = rs
        c["tuning"]["modelfit"]["model_path"] = os.path.abspath(a.model)
        c["tuning"]["cache"] = cache
        with open(f"{d}/config.yaml", "w") as f:
            yaml.safe_dump(c, f, allow_unicode=True, sort_keys=False,
                           default_flow_style=False)
        print(f"  写出 {d}/config.yaml   runner_strategy={rs}  cache={cache}")

    with open(f"{W}/verify.slurm", "w") as f:
        f.write(prof_mod.render_verify_slurm(W, prof))

    print()
    print(f"✅ 已生成 {W}/verify.slurm")
    print(f"   提交: sbatch {W}/verify.slurm")
    print(f"   分析: scripts/analyze_adtuner.py <job.out>")


if __name__ == "__main__":
    main()
