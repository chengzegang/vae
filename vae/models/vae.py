import copy
from functools import partial
from typing import Any, Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint

from ..data.gaussian import Gaussian
from ..modules import UNetDecoder, UNetEncoder
import io


class VAE(nn.Module):
    def __init__(
        self,
        encoder: UNetEncoder,
        decoder: UNetDecoder,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

        for layer in self.encoder.layers + [self.encoder.in_conv]:
            layer._org_forward_impl = layer.forward
            layer.forward = partial(checkpoint, layer.forward, use_reentrant=False)
        for layer in self.decoder.layers + [self.decoder.in_conv]:
            layer._org_forward_impl = layer.forward
            layer.forward = partial(checkpoint, layer.forward, use_reentrant=False)
        self.encoder.to(memory_format=torch.channels_last)
        self.decoder.to(memory_format=torch.channels_last)
        self.encoder.compile(fullgraph=True, dynamic=False, backend="aot_ts_nvfuser")
        self.decoder.compile(fullgraph=True, dynamic=False, backend="aot_ts_nvfuser")

    @classmethod
    def from_meta(cls, meta: dict) -> "VAE":
        encoder_meta = meta["model"].copy()
        encoder_meta["latent_size"] = encoder_meta["latent_size"] * 2
        decoder_meta = meta["model"].copy()
        decoder_meta["out_channels"] = decoder_meta.pop("in_channels")
        obj = cls(
            encoder=UNetEncoder(**encoder_meta),
            decoder=UNetDecoder(**decoder_meta),
        )
        obj.to(meta["device"])
        if meta.get("ddp", False):
            obj.encoder = DDP(
                obj.encoder, gradient_as_bucket_view=True, static_graph=True
            )
            obj.decoder = DDP(
                obj.decoder, gradient_as_bucket_view=True, static_graph=True
            )
        return obj

    @torch.jit.export
    def encode(self, x: Tensor) -> Gaussian:
        z = self.encoder(x)
        return Gaussian.from_latent(z)

    @torch.jit.export
    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z)

    @torch.jit.export
    def forward(self, x: Tensor) -> Tensor:
        return self.sample(self.encode(x))

    @torch.jit.export
    def sample(self, z: Gaussian) -> Tensor:
        return self.decode(z.sample())

    @torch.jit.export
    def train_step(self, x: Tensor, kl_weight: float = 1.0) -> Tensor:
        z = self.encode(x)
        xh = self.decode(z.sample())
        loss = F.l1_loss(xh, x) + z.kl_loss() * kl_weight
        return loss

    @torch.jit.export
    @torch.no_grad()
    def eval_step(self, x: Tensor) -> Tensor:
        z = self.encode(x)
        xh = self.decode(z.sample())
        return xh
