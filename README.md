# RF_Image_Bmode

> 线二：RF + B-mode 双输入 → 高质量 B-mode 图像重建

Created: 2026-06-12

This repository is the **second research line** of the RF_Image project.
It uses a **dual-input** architecture (RF channel data + baseline B-mode image)
to predict high-quality B-mode images, inspired by the Hy-PCF paper (IEEE TMI 2025).

## What this line is about

- **Input 1**: 3-angle raw RF [1536, 64, 32, 32]
- **Input 2**: 3-angle baseline B-mode image [1, 64, 32, 32]
- **Output**: High-quality B-mode [1, 64, 32, 32] matching 33-angle DAS quality
- **Loss**: 0.84*(1-SSIM3D) + 0.16*L1 (pure supervised, no adversarial)
- **Architecture**: Lightweight dual-encoder + fusion + decoder

## Why dual-input

Pure RF (channel-data-only) models tend to oversmooth — this was confirmed by both
the Hy-PCF paper's ablation study and our own Line-1 experiments (TinyBModeRFNet
outputting gray haze at pred_mean=0.528, Light3DUNet val SSIM declining).

The Hy-PCF paper demonstrated that adding a low-quality B-mode image as a second
input provides structural guidance that prevents oversmoothing while preserving
speckle texture.

## Relationship to RF_Image_GAN (Line 1)

This line shares:
- Same physical setup (RC6gV RCA probe, Verasonics Vantage 256)
- Same raw H5 data source (1500 patches)
- Same B-mode conversion standard (REF=64407.58, 5-step)
- Same loss formula
- Same collaboration rules (四方角色, 八条铁律)

This line is INDEPENDENT in:
- Code repository
- Data cache paths
- Experiment records
- Training machine (Win11 WSL / RTX 4060)

## Getting started

```bash
# WSL
cd /mnt/g/code/RF_Image_Bmode
conda activate rf-cgan-clean

# Documents
docs/
  ├── 项目笔记/      (8 docs — project knowledge)
  ├── 原则与规范/    (5 docs — collaboration rules, shared with Line 1)
  ├── Claude参考/    (strategy context for Claude)
  └── Codex参考/     (execution specs for Codex)
```

## Quick links

- Route map: `docs/项目笔记/00_路线决策树.md`
- Current status: `docs/项目笔记/04_当前状态与路线图.md`
- Lessons from Line 1: `docs/项目笔记/06_关键发现与教训.md`
- For Claude: `docs/Claude参考/README.md`
- For Codex: `docs/Codex参考/README.md`
