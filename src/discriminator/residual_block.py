import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ResidualBlockD(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.residual_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.scale_conv = None
        if in_channels != out_channels:
            self.scale_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

        self.gamma = nn.Parameter(torch.zeros(1))

    def _shortcut(self, x: Tensor) -> Tensor:
        if self.scale_conv is not None:
            x = self.scale_conv(x)

        return F.avg_pool2d(x, 2)

    def forward(self, x: Tensor) -> Tensor:
        return self._shortcut(x) + self.gamma * self.residual_conv(x)
