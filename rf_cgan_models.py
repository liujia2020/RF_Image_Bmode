from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


BN_MOMENTUM = 0.9


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(cin, cout, kernel_size=3, padding=1),
            nn.BatchNorm3d(cout, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv3d(cout, cout, kernel_size=3, padding=1),
            nn.BatchNorm3d(cout, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv3d(cin, cout, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(cout, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            ConvBlock(cout, cout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(x)


class Up(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(cin, cout, kernel_size=3, padding=1),
            nn.BatchNorm3d(cout, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="trilinear", align_corners=False)
        return self.conv(x)


class DualBranchBModeUNet(nn.Module):
    """Dual-input RF + baseline B-mode 3D U-Net for B-mode prediction."""

    def __init__(self):
        super().__init__()
        self.rf_stem = nn.Sequential(
            nn.Conv3d(1536, 64, kernel_size=1),
            nn.BatchNorm3d(64, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )
        self.bmode_stem = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )

        self.enc0 = ConvBlock(96, 48)
        self.enc1 = Down(48, 96)
        self.enc2 = Down(96, 128)
        self.bottleneck = Down(128, 160)

        self.up2 = Up(160, 128)
        self.dec2 = ConvBlock(256, 128)
        self.up1 = Up(128, 96)
        self.dec1 = ConvBlock(192, 96)
        self.up0 = Up(96, 48)
        self.dec0 = ConvBlock(96, 48)

        self.output_head = nn.Conv3d(48, 1, kernel_size=1)

        self._init_weights()

    def forward(self, input_rf: torch.Tensor, input_bmode: torch.Tensor) -> torch.Tensor:
        self._check_inputs(input_rf, input_bmode)

        rf_feat = self.rf_stem(input_rf)
        bm_feat = self.bmode_stem(input_bmode)
        x = torch.cat([rf_feat, bm_feat], dim=1)

        skip0 = self.enc0(x)
        skip1 = self.enc1(skip0)
        skip2 = self.enc2(skip1)
        bott = self.bottleneck(skip2)

        d2 = self.up2(bott)
        d2 = self.dec2(torch.cat([d2, skip2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, skip1], dim=1))
        d0 = self.up0(d1)
        d0 = self.dec0(torch.cat([d0, skip0], dim=1))

        out = self.output_head(d0)
        return torch.sigmoid(out)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm3d):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)

        if self.output_head.bias is not None:
            nn.init.constant_(self.output_head.bias, -2.0)

    @staticmethod
    def _check_inputs(input_rf: torch.Tensor, input_bmode: torch.Tensor) -> None:
        if input_rf.ndim != 5:
            raise ValueError(f"input_rf must be 5D [B,1536,64,32,32], got shape={tuple(input_rf.shape)}")
        if input_bmode.ndim != 5:
            raise ValueError(f"input_bmode must be 5D [B,1,64,32,32], got shape={tuple(input_bmode.shape)}")
        if input_rf.shape[1:] != (1536, 64, 32, 32):
            raise ValueError(f"input_rf shape mismatch: got {tuple(input_rf.shape)}")
        if input_bmode.shape[1:] != (1, 64, 32, 32):
            raise ValueError(f"input_bmode shape mismatch: got {tuple(input_bmode.shape)}")
        if input_rf.shape[0] != input_bmode.shape[0]:
            raise ValueError(f"batch size mismatch: input_rf={input_rf.shape[0]} input_bmode={input_bmode.shape[0]}")
