import torch
from torch import Tensor
from tensordict import tensorclass
import torch.nn.functional as F


@tensorclass
class Gaussian:
    z: Tensor
    qz: Tensor
    mean: Tensor
    logvar: Tensor
    min_logvar: float
    max_logvar: float

    @classmethod
    def from_latent(
        cls, z: Tensor, min_logvar: float = -15.0, max_logvar: float = 15.0
    ) -> "Gaussian":
        qz = z.clone().detach()
        qmean = qz[:, : z.shape[1] // 2]
        qvar = cls.clamp(z[:, z.shape[1] // 2 :], min_logvar, max_logvar)
        qz = torch.cat([qmean, qvar], dim=1)
        return cls(
            z,
            qz,
            qz[:, : z.shape[1] // 2],
            qz[:, z.shape[1] // 2 :],
            min_logvar=min_logvar,
            max_logvar=max_logvar,
            batch_size=(z.shape[0],),
        )

    @staticmethod
    def clamp(x: Tensor, min_val: float, max_val: float) -> Tensor:
        cx = x.clone().detach().clamp(min_val, max_val)
        x = cx - x.detach() + x
        return x

    def sample(self) -> Tensor:
        eps = torch.randn_like(self.mean)
        std = torch.exp(0.5 * self.logvar)
        return self.mean + eps * std

    def kl_loss(self) -> Tensor:
        return -0.5 * torch.mean(
            1 + self.logvar - self.mean.pow(2) - self.logvar.exp()
        ) + F.mse_loss(self.z, self.qz.detach())
