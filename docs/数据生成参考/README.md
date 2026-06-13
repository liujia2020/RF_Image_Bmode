# 数据生成参考（MATLAB）

> 线二训练数据的**源头脚本**归档。取自线一仓库 [github.com/liujia2020/RF_Image](https://github.com/liujia2020/RF_Image)（2026-06-13）。
>
> ⚠️ **仅供查阅，不在线二运行**——线二无 MATLAB / Verasonics 环境。脚本里的 `F:\`、`G:\` 路径是线一采集机的本地路径，原样保留以忠实记录。

## 为什么放进线二

线二的 cache（`G:\rf_training_cache\dual_input_bmode\`）由线一的 H5 patch cp / 转换而来，而那些 H5 正是下列 MATLAB 脚本生成的。把它们存进线二仓库，是为了让「RF / baseline / label 三组数据到底怎么算出来」可追溯——尤其 **1536 通道的内部顺序**、**DAS 复合方式**、**patch 索引约定** 这些直接影响架构和评估的细节。

## 文件清单

| 文件 | 作用 | 与线二的关系 |
|------|------|-------------|
| ⭐ **`build_RF_random64_full1500.m`** | **生成线二的 1500 样本**：carotid/muscle/phantom 各 100 文件 × 每文件 5 个随机 `[64,32,32]` patch（train 1050 / val 225 / test 225）。**完全自包含**——31 个本地函数（patch 提取、`extract_rf_subset` 波束形成、H5 写入、manifest、audit）全在一个文件里，只依赖 Verasonics 商业库。 | **线二 cache 的直接源头** |
| `build_RF_random64_pilot80.m` | Pilot 小样本（81 文件 × 1 patch），流程同上，用于早期打通。 | 流程参考 |
| `build_RF_datase_V2.m` | 早期 V2：patch `[16,8,8]`，含 `build_RF_dataset_manifest_v2_simple` 清单构建。 | 历史演进 |
| `build_RF_V3.m` | V3 迭代版本。 | 历史演进 |
| `MATLAB_dense_generation/` | 密集滑窗采样：整卷 `[1024,128,128]` 无重叠切块（如 `[32,16,16]`），用于**全容积推理 / 重建检查**。 | 线二 Phase 4 全容积拼接的思路参考 |

## 从脚本提取的关键约定（权威）

- **1536 通道内部顺序** → 详见 [`../项目笔记/02_数据管线详解.md`](../项目笔记/02_数据管线详解.md)「1536 通道内部顺序」节。源头是 `build_RF_random64_full1500.m`：`F_RC_patch/F_CR_patch : [Nz,Nx,Ny,128,N_angle]` → `stack([RC_re,RC_im,CR_re,CR_im])` → reshape。结论：**128 阵元 × 3 角度 × 4 分量，C-order**。
- **输入角度**：`input_angle_set = [3, 38, 73]`（alpha/beta 数组的角度索引，非度数本身）。
- **目标角度**：`target_angle_set = round(linspace(1, 75, 33))` → 33 角度均匀分布，GT 的来源。
- **patch 尺寸**：full1500 = `[64,32,32] = [Nz,Nx,Ny]`。
- **⚠️ 1-based 索引坑**：H5 meta 的 `z_idx/x_idx/y_idx` 是 **MATLAB 1-based**。Python（线二 Phase 4 全容积拼接）当数组下标用前**必须减 1**。dense 脚本注释专门强调。
- **DAS 复合**：baseline/label = `sum over (通道, 角度)` of `F_RC + F_CR` 再平均——这也是「第 4 维=通道、第 5 维=角度」的物理反证。

## 忠实性

原样归档，未改脚本内容。完整线一仓库（含 .py 训练代码、notebooks、docs）见 [github.com/liujia2020/RF_Image](https://github.com/liujia2020/RF_Image)。
