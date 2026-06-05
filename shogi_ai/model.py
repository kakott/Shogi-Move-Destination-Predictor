"""
model.py
========
将棋の次の一手予測 CNN (PyTorch)。
AlphaGo / AlphaZero スタイルの ResNet を縮小したもの。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.bn1(self.conv1(x)), inplace=True)
        h = self.bn2(self.conv2(h))
        return F.relu(x + h, inplace=True)


class ShogiPolicyNet(nn.Module):
    """次の一手 (729 クラス) を予測する Policy Network。"""

    def __init__(self, in_channels: int = 42, ch: int = 128,
                 n_blocks: int = 6, n_labels: int = 729) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[ResBlock(ch) for _ in range(n_blocks)])
        # policy head: 1x1 conv で 9ch にしてから flatten -> linear で 729
        self.policy_conv = nn.Conv2d(ch, 9, 1)
        self.policy_bn = nn.BatchNorm2d(9)
        self.policy_fc = nn.Linear(9 * 9 * 9, n_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        h = self.blocks(h)
        h = F.relu(self.policy_bn(self.policy_conv(h)), inplace=True)
        h = h.flatten(1)
        return self.policy_fc(h)
