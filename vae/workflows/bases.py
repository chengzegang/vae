from dataclasses import dataclass
from functools import partial
from more_itertools import raise_

import torch
from torch import nn
from torch.optim import Optimizer as TorchOptimizer
import inspect
from torch.optim import swa_utils
import os


@dataclass
class Environment:
    device: torch.device
    dtype: torch.dtype
    local_rank: int
    world_size: int
    batch_size: int
    num_workers: int
    log_dir: str = "logs"
    ddp: bool = False

    def __post_init__(self):
        os.makedirs(self.log_dir, exist_ok=True)

    @classmethod
    def from_meta(cls, meta: dict) -> "Environment":
        return cls(
            device=torch.device(meta["device"]),
            dtype=torch.float32
            if meta["dtype"] == "float32"
            else torch.float16
            if meta["dtype"] == "float16"
            else torch.bfloat16
            if meta["dtype"] == "bfloat16"
            else raise_(ValueError(f"Unknown dtype: {meta['dtype']}")),
            local_rank=int(os.environ.get("LOCAL_RANK", 0)),
            world_size=int(os.environ.get("WORLD_SIZE", 1)),
            batch_size=meta.get("batch_size", 1),
            num_workers=meta.get("num_workers", 0),
            ddp=meta.get("ddp", False),
        )


class Optimizer(TorchOptimizer):
    @classmethod
    def from_meta(cls, meta: dict) -> "Optimizer":
        opt_cls = getattr(torch.optim, meta.get("optimizer", "Adam"))
        args = inspect.getfullargspec(opt_cls.__init__).args
        meta = {k: v for k, v in meta.items() if k in args}
        return partial(opt_cls, **meta)


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

    @classmethod
    def from_meta(cls, meta: dict) -> "EMA":
        return partial(
            cls,
            swa_lr=meta.get("ema_lr", 1e-4),
            anneal_epochs=meta.get("ema_anneal_steps", 10),
            decay=meta.get("ema_decay", 0.9),
            start_step=meta.get("ema_start_step", 100),
        )

    def step(self, step: int):
        if step < self.start_step:
            return
        self.avg_model.update_parameters(self.model)
        self.ema_sched.step()


class PerParameterFinetuning(object):
    def __init__(self, module: nn.Module, steps_per_layer: int = 10):
        self.module = module
        self.steps_per_layer = steps_per_layer
        self._step = 0
        self._params = list(self.module.parameters())
        self._last_p_idx = 0
        self._changed = True

    @property
    def _p_idx(self):
        return self._step // self.steps_per_layer % len(self._params)

    def step(self, steps: int | None = None):
        if steps is None:
            self._step += 1
        else:
            self._step += steps
        if self._p_idx != self._last_p_idx:
            self._changed = True
        self.set_params()

    def set_params(self):
        if self._changed:
            for p in self._params:
                p.requires_grad_(False)
            self._params[self._p_idx].requires_grad_(True)
            self._changed = False
            self._last_p_idx = self._p_idx
