# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Union
from torch import Tensor, nn
import torch
from xformers.ops import memory_efficient_attention


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        head_size: int,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.head_size = head_size
        self.norm = nn.LayerNorm(embed_dim)
        self.in_proj = nn.Linear(embed_dim * 3, embed_dim * 3)

    def _split_heads(self, x: Tensor) -> Tensor:
        return x.view(x.shape[0], x.shape[1], self.embed_dim // self.head_size, -1)

    def _merge_heads(self, x: Tensor) -> Tensor:
        return x.view(x.shape[0], x.shape[1], -1)

    def _attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
    ) -> Tensor:
        if torch.jit.is_scripting() or torch.jit.is_tracing() or q.device.type == "cpu":
            return self._torch_attention(q, k, v)
        else:
            return self._flash_attention(q, k, v)

    @torch.jit.ignore
    def _flash_attention(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        return memory_efficient_attention(q, k, v)

    def _torch_attention(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        return (
            torch.nn.functional.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            )
            .transpose(1, 2)
            .contiguous()
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.dequant(x)
        x = self.norm(x)
        x = self.in_proj(x.repeat(1, 1, 3))
        qkv = self._split_heads(x)
        q, k, v = qkv.chunk(3, dim=-1)
        a_val = self._attention(
            q,
            k,
            v,
        )
        x = self._merge_heads(a_val)
        x = self.quant(x)
        return x
