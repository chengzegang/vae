from .flash_attention import MultiheadAttention, QuantMultiheadAttention
from .swiglu import SwiGLU, QuantSwiGLU
import torch
from torch import nn, Tensor


class AttentionLayer(nn.Module):
    def __init__(self, model_dim: int, head_size: int):
        super().__init__()
        self.attn = MultiheadAttention(model_dim, head_size)
        self.mlp = SwiGLU(model_dim, model_dim * 4)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(x)
        x = x + self.mlp(x)

        return x


class QuantAttentionLayer(nn.Module):
    def __init__(self, model_dim: int, head_size: int):
        super().__init__()
        self.attn = QuantMultiheadAttention(model_dim, head_size)
        self.mlp = QuantSwiGLU(model_dim, model_dim * 4)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(x)
        x = x + self.mlp(x)

        return x


class AttentionLayer2d(AttentionLayer):
    def forward(self, x: Tensor) -> Tensor:
        return (
            super().forward(x.flatten(2).transpose(-1, -2)).transpose(-1, -2).view_as(x)
        )


class QuantAttentionLayer2d(QuantAttentionLayer):
    def forward(self, x: Tensor) -> Tensor:
        return (
            super().forward(x.flatten(2).transpose(-1, -2)).transpose(-1, -2).view_as(x)
        )
