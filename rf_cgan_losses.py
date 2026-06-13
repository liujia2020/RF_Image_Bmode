from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim


class BModeLoss(nn.Module):
    def __init__(self, ssim_weight: float = 0.84, l1_weight: float = 0.16, win_size: int = 7):
        super().__init__()
        self.ssim_weight = ssim_weight
        self.l1_weight = l1_weight
        self.win_size = win_size

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        ssim_value = ssim(pred, gt, data_range=1.0, win_size=self.win_size)
        ssim_term = 1.0 - ssim_value
        l1_term = F.l1_loss(pred, gt)
        total = self.ssim_weight * ssim_term + self.l1_weight * l1_term
        return total, {"ssim_term": ssim_term, "l1_term": l1_term}
