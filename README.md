# HPC Skill

面向高性能计算（HPC）场景的 Codex skills 合集。每个 skill 均以独立目录存放在 `skills/` 下，目录名与 `SKILL.md` 中的 `name` 保持一致。

```text
HPC-skill/
├── README.md
└── skills/
    ├── adtuner-tuning-model/
    │   ├── SKILL.md
    │   ├── profiles/
    │   ├── references/
    │   └── scripts/
    └── <future-skill>/
        └── SKILL.md
```

## Skills

### adtuner-tuning-model

根据任意 ADTuner `config.yaml` 构造同结构的多负载训练数据，完成性能数据采集、泛化性能模型训练，并通过 ADTuner `runner_strategy` 0/2 验证模型引导的调优效果。

主要内容：

- `skills/adtuner-tuning-model/SKILL.md`：完整工作流与检查清单
- `skills/adtuner-tuning-model/scripts/`：配置探测、数据集生成、模型评估和调优验证脚本
- `skills/adtuner-tuning-model/profiles/`：集群环境配置模板及 MT3000、曙光 8000 示例
- `skills/adtuner-tuning-model/references/`：ADTuner 侧采集变体和常见问题说明

## 安装到 Codex

将 skill 目录复制到 Codex skills 目录：

```powershell
git clone https://github.com/Earnshawnlpl/HPC-skill.git
Copy-Item -Recurse -Force `
  .\HPC-skill\skills\adtuner-tuning-model `
  "$HOME\.codex\skills\adtuner-tuning-model"
```

重启 Codex 或新建会话后即可使用 `adtuner-tuning-model`。

## 添加新 skill

在 `skills/` 下创建新的同级目录，并确保至少包含有效的 `SKILL.md`：

```text
skills/<skill-name>/SKILL.md
```

`skill-name` 使用小写字母、数字和连字符，并与 `SKILL.md` frontmatter 中的 `name` 一致。脚本、参考资料和输出资产可分别放入该 skill 自己的 `scripts/`、`references/` 和 `assets/` 目录。
