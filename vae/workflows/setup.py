from functools import partial
import os
from dataclasses import dataclass, field

import torch
from torch import nn
from torch.optim import AdamW, Optimizer, swa_utils
from torch.utils.data import DataLoader, Dataset

from .. import data, models
from torch.utils.checkpoint import checkpoint


class EMA:
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        swa_lr: float = 1e-4,
        anneal_epochs: int = 10000,
        decay: float = 0.75,
        start_step: int = 100,
    ):
        self.model = model
        self.start_step = start_step
        self.avg_model = swa_utils.AveragedModel(
            model, avg_fn=swa_utils.get_ema_avg_fn(decay), use_buffers=True
        )
        self.ema_sched = swa_utils.SWALR(
            optimizer, swa_lr=swa_lr, anneal_epochs=anneal_epochs, anneal_strategy="cos"
        )

    @property
    def lr(self):
        return self.ema_sched.get_last_lr()[0]

    def step(self, step: int):
        if step < self.start_step:
            return
        self.avg_model.update_parameters(self.model)
        self.ema_sched.step()


@dataclass
class Setup:
    conf: dict
    log_dir: str = "logs"
    ddp: bool = False
    lr: float = 1e-4
    batch_size: int = 32
    num_workers: int = 0
    weight_decay: float = 1e-5
    betas: tuple = (0.9, 0.96)
    swa_lr: float = 1e-4
    anneal_epochs: int = 10000
    decay: float = 0.75
    start_step: int = 100
    save_every: int = 100
    total_epochs: int = 100
    step: int = 0
    epoch: int = 0
    kl_weight_min: float = 1.0
    kl_weight_max: float = 0.01
    kl_anneal_steps: int = 10000

    device: torch.device = field(init=False)
    dtype: torch.dtype = field(init=False)
    local_rank: int = field(init=False)
    world_size: int = field(init=False)
    model: nn.Module = field(init=False)
    dataset: Dataset = field(init=False)
    data: DataLoader = field(init=False)
    model: nn.Module = field(init=False)
    optimizer: Optimizer = field(init=False)
    ema: EMA = field(init=False)

    def __post_init__(self):
        self.device = torch.device(self.conf.pop("device", "cuda"))
        self.dtype = torch.dtypes(self.conf.pop("dtype", "float32"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.model = (
            models.VAE.from_meta(self.conf)
            .to(self.device)
            .to(memory_format=torch.channels_last)
        )
        self.apply_grad_checkpoint()
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.lr,
            betas=self.betas,
            weight_decay=self.weight_decay,
            fused=True,
        )
        self.dataset = data.Datasets[self.conf["dataset"]].value.from_meta(self.conf)
        self.data = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        self.log_dir = self.conf.get("log_dir", "logs")
        self.ddp = self.conf.get("ddp", False)
        self.ema = EMA(
            self.model,
            self.optimizer,
            self.swa_lr,
            self.anneal_epochs,
            self.decay,
            self.start_step,
        )

    def apply_grad_checkpoint(self):
        for layer in self.model.encoder.layers:
            layer._org_forward = layer.forward
            layer.forward = partial(checkpoint, layer._org_forward, use_reentrant=False)

        for layer in self.model.decoder.layers:
            layer._org_forward = layer.forward
            layer.forward = partial(checkpoint, layer._org_forward, use_reentrant=False)
