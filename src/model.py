"""
model.py — Csi1DCNN core + ExportWrapper (README §5, §6).

Input tensor: [batch, 64, 100] = [batch, subcarrier (channels), time (length)].
The subcarrier-major window maps onto this directly; no transpose needed.

The core returns RAW LOGITS. Softmax and input normalization live in ExportWrapper
so they ship inside the ONNX graph — never in the training loss, never in C#.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Csi1DCNN(nn.Module):
    """Compact 1D-CNN. Channels = 64 subcarriers, length = time steps. Outputs logits."""

    def __init__(self, num_classes: int, in_channels: int = 64):
        super().__init__()

        # Block 1: Conv(64->64, k=5, pad=2) -> BN -> ReLU -> MaxPool(2)
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )
        # Block 2: Conv(64->128, k=3, pad=1) -> BN -> ReLU -> MaxPool(2)
        self.block2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )
        # Block 3: Conv(128->128, k=3, pad=1) -> BN -> ReLU
        self.block3 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )
        # Head: AdaptiveAvgPool1d(1) -> Flatten -> Dropout(0.3) -> Linear(128 -> num_classes)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B, 64, T] -> logits [B, num_classes]
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.head(x)


class ExportWrapper(nn.Module):
    """
    Wrap the trained core so the exported graph is: normalize -> core -> softmax.

    mean/std are train-split-only per-subcarrier stats shaped [1, 64, 1]; registered
    as buffers so torch.onnx.export bakes them into the graph as Constants. The
    backend then feeds RAW filtered windows and the graph normalizes internally —
    train/serve skew becomes impossible.
    """

    def __init__(self, core: nn.Module, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        self.core = core
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B, 64, T] RAW filtered window
        x = (x - self.mean) / self.std
        return torch.softmax(self.core(x), dim=1)  # [B, num_classes] probabilities
