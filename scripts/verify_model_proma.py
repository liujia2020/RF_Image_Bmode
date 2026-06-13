#!/usr/bin/env python3
"""Proma 独立验证脚本 —— 测 rf_cgan_models + rf_cgan_losses，只报数不下结论。

9 项验收标准，逐项打印实测值。不读 Codex smoke_test，不改生产代码。
用法：source ... && conda run -n rf-bmode python scripts/verify_model_proma.py
"""

from __future__ import annotations

import sys, time
from pathlib import Path

import torch
import torch.nn as nn

# --- 确保能 import 生产代码 ---
PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from rf_cgan_models import DualBranchBModeUNet, BN_MOMENTUM
from rf_cgan_losses import BModeLoss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

results = {}


def header(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def report(item: str, value: str) -> None:
    results[item] = value
    print(f"  [{item}] {value}")


# ===================================================================
# 1. [契约] 输出 shape
# ===================================================================
header("1. [契约] 输出 shape")

model = DualBranchBModeUNet().to(DEVICE)
model.eval()

input_rf = torch.randn(2, 1536, 64, 32, 32, device=DEVICE)
input_bmode = torch.rand(2, 1, 64, 32, 32, device=DEVICE)

with torch.no_grad():
    output = model(input_rf, input_bmode)

expected = (2, 1, 64, 32, 32)
match = "MATCH" if output.shape == expected else f"MISMATCH"
report("output.shape", f"{tuple(output.shape)}  [{match}]  expected={expected}")

# ===================================================================
# 2. [范围] 输出 min/max
# ===================================================================
header("2. [范围] 输出 min/max 是否在 [0,1]")

omin = float(output.min().item())
omax = float(output.max().item())
in_range = "YES" if omin >= 0.0 and omax <= 1.0 else "NO"
report("output.min", f"{omin:.6f}")
report("output.max", f"{omax:.6f}")
report("in [0,1]", f"{in_range}")

# ===================================================================
# 3. [防灰雾] 输出头 bias
# ===================================================================
header("3. [防灰雾] 输出头 bias")

head: nn.Conv3d = model.output_head
bias = head.bias
report("output_head class", type(head).__name__)
report("output_head bias mean", f"{bias.mean().item():.4f}")
report("output_head bias values", f"{bias.detach().cpu().flatten().tolist()}")

# ===================================================================
# 4. [规范] BN / IN 统计
# ===================================================================
header("4. [规范] BatchNorm3d / InstanceNorm3d 统计")

bn_count = 0
in_count = 0
bn_momentums = []
for name, module in model.named_modules():
    if isinstance(module, nn.BatchNorm3d):
        bn_count += 1
        bn_momentums.append(float(module.momentum))
    if isinstance(module, nn.InstanceNorm3d):
        in_count += 1
        report(f"  ⚠ IN found: {name}", type(module).__name__)

report("BatchNorm3d count", str(bn_count))
report("InstanceNorm3d count", str(in_count))
if bn_momentums:
    report("BN momentum (first)", f"{bn_momentums[0]:.4f}")
    all_same = all(abs(m - bn_momentums[0]) < 1e-6 for m in bn_momentums)
    report("BN momentum all same", "YES" if all_same else f"NO — values: {bn_momentums}")

# ===================================================================
# 5. [规模] 总参数量
# ===================================================================
header("5. [规模] 总参数量")

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
report("total params", f"{total_params:,} ({total_params/1e6:.4f} M)")
report("trainable params", f"{trainable_params:,} ({trainable_params/1e6:.4f} M)")

# ===================================================================
# 6. [反传] backward + 梯度 NaN/Inf
# ===================================================================
header("6. [反传] BModeLoss backward")

model.train()
# 用新的随机输入，重新做一次 forward（需要 grad）
input_rf_2 = torch.randn(2, 1536, 64, 32, 32, device=DEVICE)
input_bmode_2 = torch.rand(2, 1, 64, 32, 32, device=DEVICE)
gt = torch.rand(2, 1, 64, 32, 32, device=DEVICE)

criterion = BModeLoss()
pred = model(input_rf_2, input_bmode_2)
loss, terms = criterion(pred, gt)

try:
    loss.backward()
    report("backward", "SUCCESS")
except Exception as e:
    report("backward", f"FAILED: {e}")

grad_nan = False
grad_inf = False
grad_nan_layers = []
grad_inf_layers = []
for name, param in model.named_parameters():
    if param.grad is not None:
        if torch.any(torch.isnan(param.grad)):
            grad_nan = True
            grad_nan_layers.append(name)
        if torch.any(torch.isinf(param.grad)):
            grad_inf = True
            grad_inf_layers.append(name)

report("any grad NaN", f"{'YES ⚠' if grad_nan else 'none'}" + (f" layers: {grad_nan_layers}" if grad_nan else ""))
report("any grad Inf", f"{'YES ⚠' if grad_inf else 'none'}" + (f" layers: {grad_inf_layers}" if grad_inf else ""))

# ===================================================================
# 7. [损失自洽] pred=gt vs random pred
# ===================================================================
header("7. [损失自洽]")

pred_eq_gt = gt.clone().detach()
loss_eq, terms_eq = criterion(pred_eq_gt, gt)
report("pred=gt total loss", f"{loss_eq.item():.8f}")
report("pred=gt ssim_term", f"{terms_eq['ssim_term'].item():.8f}")
report("pred=gt l1_term", f"{terms_eq['l1_term'].item():.8f}")

pred_rand = torch.rand(2, 1, 64, 32, 32, device=DEVICE)
loss_rand, terms_rand = criterion(pred_rand, gt)
report("pred_rand total loss", f"{loss_rand.item():.6f}")
report("pred_rand ssim_term", f"{terms_rand['ssim_term'].item():.6f}")
report("pred_rand l1_term", f"{terms_rand['l1_term'].item():.6f}")

# ===================================================================
# 8. [显存] micro-batch=2, fp32, forward+backward
# ===================================================================
header("8. [显存] fp32, batch=2, forward+backward max memory")

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats(DEVICE)

model_2 = DualBranchBModeUNet().to(DEVICE)
model_2.train()
rf_8 = torch.randn(2, 1536, 64, 32, 32, device=DEVICE, dtype=torch.float32)
bm_8 = torch.rand(2, 1, 64, 32, 32, device=DEVICE, dtype=torch.float32)
gt_8 = torch.rand(2, 1, 64, 32, 32, device=DEVICE, dtype=torch.float32)

criterion_8 = BModeLoss()
pred_8 = model_2(rf_8, bm_8)
loss_8, _ = criterion_8(pred_8, gt_8)
loss_8.backward()

peak_gb = torch.cuda.max_memory_allocated(DEVICE) / 1e9
peak_all_gb = torch.cuda.max_memory_reserved(DEVICE) / 1e9
report("max_memory_allocated (GB)", f"{peak_gb:.4f}")
report("max_memory_reserved (GB)", f"{peak_all_gb:.4f}")

# cleanup
del model_2, rf_8, bm_8, gt_8, pred_8, loss_8, criterion_8
torch.cuda.empty_cache()

# ===================================================================
# 9. [数值] 输出 NaN/Inf
# ===================================================================
header("9. [数值] 输出 NaN/Inf 计数")

model.eval()
with torch.no_grad():
    out9 = model(input_rf, input_bmode)
    nan_count = int(torch.isnan(out9).sum().item())
    inf_count = int(torch.isinf(out9).sum().item())
    report("output NaN count", str(nan_count))
    report("output Inf count", str(inf_count))

# ===================================================================
# 汇总
# ===================================================================
header("汇总清单")

for i, (k, v) in enumerate(results.items(), 1):
    print(f"  {k}:  {v}")

print(f"\n{'='*60}")
print(f"  脚本完成 @ {time.strftime('%Y-%m-%d %H:%M:%S +08:00')}")
print(f"  device: {DEVICE}  |  cuda: {torch.cuda.is_available()}")
print(f"  torch: {torch.__version__}  |  cuda ver: {torch.version.cuda}")
print(f"  以上为 Proma 独立实测值，不做 PASS/FAIL 结论。")
print(f"{'='*60}")
