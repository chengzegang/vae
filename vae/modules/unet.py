from typing import List

from torch import Tensor, nn
from .blocks import ResidualBlock


class ConvDown(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        eps: float = 1e-4,
    ):
        super().__init__()
        self.res = ResidualBlock(in_channels, out_channels, eps)
        self.down = nn.Conv2d(out_channels, out_channels, kernel_size=2, stride=2)
        self.compile(fullgraph=True, dynamic=False, backend="aot_ts_nvfuser")

    def forward(self, x: Tensor) -> Tensor:
        x = self.res(x)
        x = self.down(x)
        return x


class ConvUp(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        eps: float = 1e-4,
    ):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
        )
        self.res = ResidualBlock(out_channels, out_channels, eps)
        self.compile(fullgraph=True, dynamic=False, backend="aot_ts_nvfuser")

    def forward(self, x: Tensor) -> Tensor:
        x = self.up(x)
        x = self.res(x)
        return x


class UNetEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: List[int],
        latent_size: int,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.in_conv = nn.Conv2d(in_channels, channels[0], kernel_size=1)
        for i in range(len(channels) - 1):
            self.layers.append(ConvDown(channels[i], channels[i + 1], eps))
        self.layers.append(ResidualBlock(channels[-1], channels[-1], eps))
        self.attn = nn.TransformerEncoderLayer(
            d_model=channels[-1],
            nhead=8,
            dim_feedforward=channels[-1] * 4,
            dropout=0.1,
            activation=nn.SiLU(True),
            batch_first=True,
        )
        self.out_norm = nn.InstanceNorm2d(channels[-1], eps=eps)
        self.out_act = nn.SiLU(True)
        self.out_conv = nn.Conv2d(channels[-1], latent_size, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.in_conv(x)
        for layer in self.layers:
            x = layer(x)
        a_val = x.flatten(2).transpose(-1, -2)
        a_val = self.attn(a_val)
        a_val = a_val.transpose(-1, -2).reshape(x.shape)
        x = x + a_val
        z = self.out_conv(x)
        return z


class UNetDecoder(nn.Module):
    def __init__(
        self,
        out_channels: int,
        channels: List[int],
        latent_size: int,
        eps: float = 1e-4,
    ):
        super().__init__()
        channels = channels[::-1]
        self.in_conv = nn.Conv2d(latent_size, channels[0], kernel_size=1)
        self.attn = nn.TransformerEncoderLayer(
            d_model=channels[0],
            nhead=8,
            dim_feedforward=channels[0] * 4,
            dropout=0.1,
            activation=nn.SiLU(True),
            batch_first=True,
        )
        self.layers = nn.ModuleList()
        self.layers.append(ResidualBlock(channels[0], channels[0], eps))
        for i in range(len(channels) - 1):
            self.layers.append(ConvUp(channels[i], channels[i + 1], eps))
        self.out_norm = nn.InstanceNorm2d(channels[-1], eps=eps)
        self.out = nn.Conv2d(channels[-1], out_channels, kernel_size=1)

    def forward(self, qz: Tensor) -> Tensor:
        x = self.in_conv(qz)
        a_val = x.flatten(2).transpose(-1, -2)
        a_val = self.attn(a_val)
        a_val = a_val.transpose(-1, -2).reshape(x.shape)
        x = x + a_val
        for layer in self.layers:
            x = layer(x)
        x = self.out(x)
        return x
