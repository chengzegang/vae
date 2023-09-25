from functools import partial

import torch
from torch import Tensor, nn
from torch.nn import functional as F


from ..data.gaussian import Gaussian
from ..modules import UNetDecoder, UNetEncoder
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
        for layer in self.encoder.layers:
            layer.forward = partial(checkpoint, layer.forward, use_reentrant=False)
        for layer in self.decoder.layers:
            layer.forward = partial(checkpoint, layer.forward, use_reentrant=False)
        self.encoder.to(memory_format=torch.channels_last)
        self.decoder.to(memory_format=torch.channels_last)
        # self.encoder.requires_grad_(False)
        # self.decoder.requires_grad_(False)
        # self.decoder.layers[-1].requires_grad_(True)
        # self.decoder.out_norm.requires_grad_(True)
        # self.decoder.out.requires_grad_(True)

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

    def sample(self, z: Gaussian) -> Tensor:
        return self.decode(z.sample())

    def train_step(self, x: Tensor, kl_weight: float = 1.0) -> Tensor:
        z = self.encode(x)
        xh = self.decode(z.sample())
        loss = F.l1_loss(xh, x) + z.kl_loss() * kl_weight
        return loss

    @torch.no_grad()
    def eval_step(self, x: Tensor) -> Tensor:
        z = self.encode(x)
        xh = self.decode(z.sample())
        return xh
