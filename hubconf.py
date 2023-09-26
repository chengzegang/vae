from vae import models
import torch
import os


def vae(pretrained=True, qint8: bool = True, **kwargs):
    model = models.VAE(
        in_channels=3, out_channels=3, channels=[64, 128, 256, 512], latent_size=64
    )
    model.eval()
    if pretrained:
        ckpt = torch.load(
            os.path.join(
                os.path.dirname(__file__),
                "checkpoints",
                "imagenet1k_256_64_55100-32_0.015656.pt",
            )
        )
        ckpt.pop("step")
        model.load_state_dict(ckpt)

    if qint8:
        model.convert_quant()
    return model
