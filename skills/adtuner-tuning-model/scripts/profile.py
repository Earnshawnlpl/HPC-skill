"""平台档案加载 + 作业脚本渲染。

**不把平台写死**：集群相关的路径/调度器/模块全部放进 profile YAML，
脚本只依赖 profile 的字段。内置 `profiles/sugon8000.yaml` 作为示例，
新平台照 `profiles/TEMPLATE.yaml` 填一份即可。

profile 字段（全部可选，缺省见 DEFAULTS）:
    name, description
    paths:      cpm, cpm_python, adtuner, adtuner_python, adtuner_source
    env:        conda_sh, cpm_env, adtuner_env, modules[], adtuner_modules[]
    scheduler:  type(slurm), partition, gres, nodes, ntasks, collect_time, verify_time
    slurm_extra:  "任意附加到 #SBATCH 之后的 shell 片段"
    env_exports:  {VAR: value}   作业内 export
    kwargs:     launch_bounds, collect_time 等
"""
import os

DEF = {
    "name": "default",
    "description": "未命名平台",
    "paths": {
        "cpm": "cpm",
        "cpm_python": "python3",
        "adtuner": "adtuner",
        "adtuner_python": "python3",
        "adtuner_source": "",
    },
    "env": {
        "conda_sh": "",
        "cpm_env": "",
        "adtuner_env": "",
        "modules": [],
        "adtuner_modules": [],
    },
    "scheduler": {
        "type": "slurm",
        "partition": "",
        "gres": "",
        "nodes": 1,
        "ntasks": 1,
        "collect_time": "02:00:00",
        "verify_time": "00:30:00",
    },
    "slurm_extra": "",
    "env_exports": {},
    "kwargs": {"launch_bounds": 1024},
}


def _merge(base, over):
    out = dict(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def default_profile_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "profiles", "sugon8000.yaml")


def load_profile(path=None):
    import yaml  # adtuner 环境自带
    p = path or default_profile_path()
    prof = DEF
    if p and os.path.isfile(p):
        prof = _merge(DEF, yaml.safe_load(open(p)) or {})
    prof["_path"] = os.path.abspath(p) if p and os.path.isfile(p) else None
    return prof


def _preamble(prof, which):
    e = prof["env"]
    lines = [
        'source /etc/profile 2>/dev/null',
        'if [ -n "${MODULESHOME:-}" ]; then source "$MODULESHOME/init/bash" 2>/dev/null; fi',
        'export TERM=xterm',
    ]
    mods = list(e.get("modules") or [])
    if which == "verify":
        mods += list(e.get("adtuner_modules") or [])
    if mods:
        lines.append("module purge 2>/dev/null || true")
        for m in mods:
            lines.append(f"module load {m}")
    if e.get("conda_sh"):
        lines.append(f'source {e["conda_sh"]}')
    if which == "collect" and e.get("cpm_env"):
        lines.append(f'conda activate {e["cpm_env"]}')
    if which == "verify" and e.get("adtuner_env"):
        lines.append(f'conda activate {e["adtuner_env"]}')
    return lines


def _sbatch(prof, name, out, err, tm):
    s = prof["scheduler"]
    L = [f"#SBATCH -J {name}"]
    if s.get("partition"):
        L.append(f"#SBATCH -p {s['partition']}")
    if s.get("nodes"):
        L.append(f"#SBATCH -N {s['nodes']}")
    if s.get("ntasks"):
        L.append(f"#SBATCH -n {s['ntasks']}")
    if s.get("gres"):
        L.append(f"#SBATCH --gres={s['gres']}")
    L += [f"#SBATCH -t {tm}", f"#SBATCH -o {out}", f"#SBATCH -e {err}"]
    return L


def note_scheduler(prof):
    t = (prof["scheduler"] or {}).get("type", "slurm")
    if t != "slurm":
        return [f"# ⚠️ 本 profile 的调度器是 {t}；render_* 生成的是 SLURM 脚本，",
                f"#    请按该调度器改写提交头（PBS: #PBS / LSF: #BSUB）。"]
    return []


def render_collect_slurm(W, prof):
    ap = prof["paths"]
    s = prof["scheduler"]
    L = ["#!/bin/bash"] + note_scheduler(prof) + _sbatch(
        prof, "adtuner_collect", f"{W}/job_collect_%j.out", f"{W}/job_collect_%j.err",
        s.get("collect_time", "02:00:00"))
    L.append("")
    L.append('echo "[$(date \'+%F %T\')] start (job ${SLURM_JOB_ID:-local})"')
    L += _preamble(prof, "collect")
    for k, v in (prof.get("env_exports") or {}).items():
        L.append(f'export {k}="{v}"')
    L += [
        "",
        f"cd {W} || {{ echo FATAL; exit 1; }}",
        'echo "node : $(hostname)"',
        'echo "cpm  : $(command -v cpm || echo ' + ap["cpm"] + ')"',
        "echo '============================================================'",
        "",
        "rm -f dataset.csv",
        "cpm collect.json",
        'echo "[cpm] exit=$?"',
        "",
        "if [ -f dataset.csv ]; then",
        '    echo "行数: $(wc -l < dataset.csv)"',
        "    head -2 dataset.csv",
        "else",
        '    echo "WARN: dataset.csv 未生成"',
        "fi",
        '[ -d __CPM_CACHE__ ] && echo "WARN: __CPM_CACHE__ 残留"',
        'echo "[$(date \'+%F %T\')] done"',
        "",
    ]
    return "\n".join(L)


def render_verify_slurm(W, prof):
    ap = prof["paths"]
    s = prof["scheduler"]
    L = ["#!/bin/bash"] + note_scheduler(prof) + _sbatch(
        prof, "adtuner_verify", f"{W}/job_verify_%j.out", f"{W}/job_verify_%j.err",
        s.get("verify_time", "00:30:00"))
    L.append("")
    L.append('echo "[$(date \'+%F %T\')] start (job ${SLURM_JOB_ID:-local})"')
    L += _preamble(prof, "verify")
    # adtuner 环境常见的 PATH / LD_LIBRARY_PATH 补充（arch 由 profile 提供）
    L.append('export PATH="$(dirname "$(command -v adtuner || echo ' + ap["adtuner"] + ')")":$PATH')
    ld = (prof.get("env") or {}).get("adtuner_ld_library_path")
    if ld:
        L.append(f'export LD_LIBRARY_PATH={ld}:${{LD_LIBRARY_PATH:-}}')
    L.append('export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}')
    for k, v in (prof.get("env_exports") or {}).items():
        L.append(f'export {k}="{v}"')
    L += [
        "",
        f"W={W}",
        'cd "$W" || { echo FATAL; exit 1; }',
        'echo "node : $(hostname)"',
        "echo '============================================================'",
        "",
        'echo',
        'echo "########## [1/2] runner_strategy = 0  (真值, 全空间实测) ##########"',
        'rm -f "$W/gt/gt_cache.json" "$W/gt"/*.log 2>/dev/null',
        'cd "$W/gt" && adtuner config.yaml',
        'echo "[gt] exit=$?"',
        "",
        'echo',
        'echo "########## [2/2] runner_strategy = 2  (模型预测 top-k 后实测) ##########"',
        'rm -f "$W/topk/topk_cache.json" "$W/topk"/*.log 2>/dev/null',
        'cd "$W/topk" && adtuner config.yaml',
        'echo "[topk] exit=$?"',
        "",
        'echo',
        'echo "[$(date \'+%F %T\')] done"',
        "",
    ]
    return "\n".join(L)
