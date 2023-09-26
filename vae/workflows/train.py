from collections import OrderedDict
from dataclasses import dataclass, field

import torch
from torch import Tensor
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
import os
from .bases import EMA, Environment, Optimizer, PerParameterFinetuning as PPF
from ..models.vae import VAE
import matplotlib.pyplot as plt
import torchvision.transforms.v2.functional as TF


@dataclass
class Train:
    env: Environment
    model: VAE
    data: DataLoader
    optimizer: Optimizer
    ema: EMA
    save_every: int = 100
    total_epochs: int = 100
    step: int = 0
    epoch: int = 0
    kl_weight_min: float = 1e-3
    kl_weight_max: float = 1e-6
    kl_anneal_steps: int = 10000
    grad_norm_reg: float = 1e-4
    ppf_steps: int = 10
    ppf: PPF = field(init=False)

    def __post_init__(self):
        self.ppf = PPF(self.model, self.ppf_steps)

    def start(self) -> None:
        scaler = GradScaler()
        kl_weights = torch.linspace(
            self.kl_weight_min, self.kl_weight_max, self.kl_anneal_steps
        )
        self.load()
        for ep in range(self.epoch, self.total_epochs):
            self.epoch = ep
            for batch in self.data:
                batch = batch.to(self.env.device).contiguous(
                    memory_format=torch.channels_last
                )
                self.model.train()
                with torch.autocast("cuda", self.env.dtype):
                    loss = self.model.train_step(
                        batch, kl_weights[self.step % self.kl_anneal_steps]
                    )
                self.model.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                self.ema.step(self.step)
                scaler.update()

                print(
                    f"Epoch {self.epoch} Step {self.step} Loss {loss.item():.6f} ",
                    end="\r",
                )
                if self.step % self.save_every == 0:
                    self.model.eval()
                    with torch.no_grad():
                        self.save()
                        self.show(batch, self.model.eval_step(batch))
                self.step += 1

    @torch.no_grad()
    def save(self):
        self.model.eval()
        state = self.model.state_dict()
        org_state = OrderedDict()
        for k, v in state.items():
            org_state[k.replace("_org_mod.", "")] = v
        org_state["step"] = self.step
        torch.save(org_state, os.path.join(self.env.log_dir, "model.pt"))

    @torch.no_grad()
    def load(self):
        self.model.eval()
        try:
            state = torch.load(os.path.join(self.env.log_dir, "model.pt"))
            org_state = OrderedDict()
            for k, v in state.items():
                org_state[k.replace("_org_mod.", "")] = v
            self.model.load_state_dict(org_state, strict=False)
            self.step = state["step"]
        except Exception as e:
            print(e)
            print("No model loaded")

    @torch.no_grad()
    def show(self, input: Tensor, output: Tensor):
        fig = plt.figure()
        ax = fig.add_subplot(1, 2, 1)
        ax.imshow(TF.to_pil_image(input[0].clamp(0, 1)))
        ax.set_title("Input")
        ax.set_xticks([])
        ax.set_yticks([])
        ax = fig.add_subplot(1, 2, 2)
        ax.imshow(TF.to_pil_image(output[0].clamp(0, 1)))
        ax.set_title("Output")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(f"{self.env.log_dir}/result.png")
        plt.close(fig)
