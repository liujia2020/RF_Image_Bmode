# dual_input_v1 训练运行目录规范

本目录只放双输入 v1 的运行记录。正式训练必须使用四段式结构，并在日期后加入两位编号，便于判断顺序。

## 命名规则

```text
YYYY-MM-DD_NN_short_name/
```

- `YYYY-MM-DD`：创建或正式开跑日期。
- `NN`：当天第几个正式 run，从 `01` 开始递增。
- `short_name`：短说明，只写关键变量，例如 `baseline_50epoch`。

示例：

```text
2026-06-13_01_baseline_50epoch/
2026-06-13_02_fusion_v2/
```

## 正式 run 结构

```text
YYYY-MM-DD_NN_short_name/
├── 01_config/          训练前配置 (README / config.yaml / manifest)
├── 02_train/           训练产物 (train.ipynb / checkpoints / probes / logs / metrics)
├── 03_validate/        验证结果 (nii / figures / metrics / verdict)
└── 04_proma_verify/    Proma 独立验证
```

## 文件原则

- `01_config/config.yaml` 是唯一参数源。
- `02_train/train.ipynb` 是训练核心文件，一个 notebook 贯穿到底。
- 正式结果必须由 `train.ipynb` 一次性 Restart & Run All 产出。
- 新建 run 时必须先清空 notebook output，避免复制旧输出误导。
- 临时监控脚本、旧 lock、`__pycache__` 不作为正式产物保留。

## 归档区

旧结构、smoke、早期 pilot 放入：

```text
00_archive_legacy_runs/
```

归档区只保留历史参考，不作为当前正式训练入口。
