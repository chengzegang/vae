from functools import partial
from typing import List

import torch
from torch import Tensor, nn
from torch.nn import functional as F


from ..data.gaussian import Gaussian
from ..modules import UNetDecoder, UNetEncoder, Discriminator
from torch.utils.checkpoint import checkpoint
from torch.nn.parallel import DistributedDataParallel as DDP


class VAE(nn.Module):
    def __init__(
        self,
        encoder: UNetEncoder,
        decoder: UNetDecoder,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        # self.D = Discriminator(
        #    encoder.in_channels, [64, 128, 256, 512, 1024, 1024, 1024], 1
        # )
        for layer in self.encoder.layers:
            layer.forward = partial(checkpoint, layer.forward, use_reentrant=False)
        for layer in self.decoder.layers:
            layer.forward = partial(checkpoint, layer.forward, use_reentrant=False)
        # for layer in self.D.layers:
        #    layer.forward = partial(checkpoint, layer.forward, use_reentrant=False)
        self.encoder.to(memory_format=torch.channels_last)
        self.decoder.to(memory_format=torch.channels_last)
        # self.D.to(memory_format=torch.channels_last)

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

    def encode(self, x: Tensor) -> Gaussian:
        z = self.encoder(x)
        return Gaussian.from_latent(z)

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z)

    def forward(self, x: Tensor) -> Tensor:
        return self.sample(self.encode(x))

    def sample(self, z: Gaussian) -> Tensor:
        return self.decode(z.sample())

    def D_step(self, x: Tensor, xh: Tensor) -> Tensor:
        self.D.requires_grad_(True)
        fake_pred = self.D(xh)
        real_pred = self.D(x)
        loss = F.binary_cross_entropy_with_logits(
            fake_pred, torch.zeros_like(fake_pred)
        ) + F.binary_cross_entropy_with_logits(real_pred, torch.ones_like(real_pred))
        return loss

    def G_step(self, x: Tensor, kl_weight: float = 1.0) -> Tensor:
        # self.D.requires_grad_(False)
        z = self.encode(x)
        xh = self.decode(z.sample())
        loss = F.l1_loss(xh, x) + z.kl_loss() * kl_weight
        # fake_pred = self.D(xh)
        # loss = loss + 1e-3 * F.binary_cross_entropy_with_logits(
        #    fake_pred, torch.ones_like(fake_pred)
        # )
        return loss

    @torch.no_grad()
    def eval_step(self, x: Tensor) -> Tensor:
        z = self.encode(x)
        xh = self.decode(z.sample())
        return xh
