from typing import List
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from .unet import ConvUp, ConvDown, UNet
from .convolutions import QuantConv2d, ResidualBlock


class BinaryCodebook(nn.Module):
    def __init__(self, dim: int, size: int):
        super().__init__()
        self.code = nn.Parameter(torch.randn(size, dim))
        self.proj = nn.Linear(dim, dim, bias=False)
        self.act = nn.Sigmoid()
        self.mha = nn.TransformerDecoderLayer(
            dim,
            max(1, dim // 64),
            dim * 4,
            0,
            F.silu,
            batch_first=True,
            norm_first=True,
        )

    def forward(self, x: Tensor, meta: dict | None = None) -> Tensor:
        code = self.act(self.proj(self.code))
        qcode = torch.where(code > 0.5, 1, 0)
        qcode = code + (qcode - code).detach()
        q_loss = F.mse_loss(qcode.detach(), code)
        if meta is not None:
            meta["q_loss"] += q_loss

        x = (
            self.mha(
                x.flatten(-2).transpose(-1, -2),
                qcode.unsqueeze(0).repeat(x.shape[0], 1, 1),
            )
            .transpose(-1, -2)
            .reshape_as(x)
        )
        return x


class CodeConvDown(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, book_size: int, eps: float):
        super().__init__()
        self.codebook = BinaryCodebook(in_channels, book_size)
        self.down = ConvDown(in_channels, out_channels)

    def forward(self, x: Tensor, meta: dict | None = None) -> Tensor:
        x = self.codebook(x, meta)
        x = self.down(x)
        return x


class CodeConvUp(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, book_size: int, eps: float):
        super().__init__()
        self.codebook = BinaryCodebook(in_channels, book_size)
        self.up = ConvUp(in_channels, out_channels)

    def forward(self, x: Tensor, meta: dict | None = None) -> Tensor:
        x = self.codebook(x, meta)
        x = self.up(x)
        return x


class CodeUNetEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: List[int],
        latent_size: int,
        book_size: int = 64,
        eps: float = 1e-4,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.latent_size = latent_size
        self.layers = nn.ModuleList()
        self.in_conv = QuantConv2d(in_channels, channels[0], kernel_size=1)
        for i in range(len(channels) - 1):
            self.layers.append(
                CodeConvDown(channels[i], channels[i + 1], book_size, eps)
            )
        self.layers.append(ResidualBlock(channels[-1], channels[-1], eps))
        self.out_conv = QuantConv2d(channels[-1], latent_size, kernel_size=1)

    def forward(self, x: Tensor, meta: dict | None = None) -> Tensor:
        x = self.in_conv(x)
        for layer in self.layers:
            x = layer(x, meta)
        z = self.out_conv(x)
        return z


class CodeUNetDecoder(nn.Module):
    def __init__(
        self,
        out_channels: int,
        channels: List[int],
        latent_size: int,
        book_size: int = 64,
        eps: float = 1e-4,
    ):
        super().__init__()
        self.out_channels = out_channels
        channels = channels[::-1]
        self.in_conv = QuantConv2d(latent_size, channels[0], kernel_size=1)
        self.layers = nn.ModuleList()
        self.layers.append(ResidualBlock(channels[0], channels[0], eps))
        for i in range(len(channels) - 1):
            self.layers.append(CodeConvUp(channels[i], channels[i + 1], book_size, eps))
        self.out_norm = nn.InstanceNorm2d(channels[-1], eps=eps)
        self.out = QuantConv2d(channels[-1], out_channels, kernel_size=1)

    def forward(self, qz: Tensor, meta: dict | None = None) -> Tensor:
        x = self.in_conv(qz)
        for layer in self.layers:
            x = layer(x, meta)
        x = self.out(x)
        return x


class CodeUnet(UNet):
    ...
