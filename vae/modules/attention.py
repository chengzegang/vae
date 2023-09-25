from .flash_attention import MultiheadAttention
from .swiglu import SwiGLU
import torch
from torch import nn, Tensor


class MLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.SiLU(True)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.dropout(self.act(self.fc1(x))))


class AttentionLayer(nn.Module):
    def __init__(self, model_dim: int, head_size: int):
        super().__init__()
        self.attn = MultiheadAttention(model_dim, head_size)
        self.mlp = SwiGLU(model_dim, model_dim * 4)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class AttentionLayer2d(AttentionLayer):
    def forward(self, x: Tensor) -> Tensor:
        return (
            super().forward(x.flatten(2).transpose(-1, -2)).transpose(-1, -2).view_as(x)
        )
