from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rf_cgan_losses import BModeLoss
from rf_cgan_models import DualBranchBModeUNet


def tensor_min_max_mean(tensor: torch.Tensor) -> tuple[float, float, float]:
    return float(tensor.min().item()), float(tensor.max().item()), float(tensor.mean().item())


def tensor_has_nan_inf(tensor: torch.Tensor) -> tuple[bool, bool]:
    return bool(torch.isnan(tensor).any().item()), bool(torch.isinf(tensor).any().item())


def count_modules(model: nn.Module, module_type: type[nn.Module]) -> int:
    return sum(1 for module in model.modules() if isinstance(module, module_type))


def gradients_have_nan_inf(model: nn.Module) -> tuple[bool, bool]:
    has_nan = False
    has_inf = False
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        has_nan = has_nan or bool(torch.isnan(parameter.grad).any().item())
        has_inf = has_inf or bool(torch.isinf(parameter.grad).any().item())
    return has_nan, has_inf


def main() -> None:
    print(f"torch.__version__={torch.__version__}", flush=True)
    print(f"torch.cuda.is_available()={torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"torch.cuda.get_device_name(0)={torch.cuda.get_device_name(0)}", flush=True)
    else:
        print("torch.cuda.get_device_name(0)=NA", flush=True)
        raise RuntimeError("CUDA is not available; smoke_test_model.py requires CUDA for this test")

    torch.manual_seed(0)
    torch.cuda.reset_peak_memory_stats()
    device = torch.device("cuda")

    model = DualBranchBModeUNet().to(device=device, dtype=torch.float32)
    loss_fn = BModeLoss().to(device=device)
    total_params = sum(parameter.numel() for parameter in model.parameters())
    print(f"total_params={total_params}", flush=True)

    input_rf = torch.randn(2, 1536, 64, 32, 32, device=device, dtype=torch.float32)
    input_bmode = torch.rand(2, 1, 64, 32, 32, device=device, dtype=torch.float32)

    pred = model(input_rf, input_bmode)
    pred_min, pred_max, pred_mean = tensor_min_max_mean(pred)
    pred_nan, pred_inf = tensor_has_nan_inf(pred)
    print(
        f"output.shape={tuple(pred.shape)} min={pred_min:.6g} max={pred_max:.6g} "
        f"mean={pred_mean:.6g} nan={pred_nan} inf={pred_inf}",
        flush=True,
    )

    gt = torch.rand(2, 1, 64, 32, 32, device=device, dtype=torch.float32)
    total, terms = loss_fn(pred, gt)
    print(
        f"loss total={float(total.item()):.6g} "
        f"ssim_term={float(terms['ssim_term'].item()):.6g} "
        f"l1_term={float(terms['l1_term'].item()):.6g}",
        flush=True,
    )

    total.backward()
    grad_nan, grad_inf = gradients_have_nan_inf(model)
    print(f"backward_success=True grad_nan={grad_nan} grad_inf={grad_inf}", flush=True)

    max_memory_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"cuda_max_memory_allocated_gb={max_memory_gb:.6g}", flush=True)

    head_bias_mean = float(model.output_head.bias.detach().mean().item())
    bn_count = count_modules(model, nn.BatchNorm3d)
    instance_norm_count = count_modules(model, nn.InstanceNorm3d)
    print(f"output_head_bias_mean={head_bias_mean:.6g}", flush=True)
    print(f"batchnorm3d_count={bn_count}", flush=True)
    print(f"instancenorm_count={instance_norm_count}", flush=True)


if __name__ == "__main__":
    main()
