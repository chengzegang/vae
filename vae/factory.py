from .workflows import Train
from .workflows.bases import Environment, Optimizer, EMA
from .models.vae import VAE
from .data import Datasets
import yaml
from torch.utils.data import DataLoader


def train_from_conf(path: str):
    conf = yaml.safe_load(open(path))
    env = Environment.from_meta(conf)
    model = VAE.from_meta(conf)
    optimizer = Optimizer.from_meta(conf)(model.parameters())
    data = Datasets[conf["dataset"]].value.from_meta(conf)
    data = DataLoader(
        data,
        batch_size=env.batch_size,
        shuffle=True,
        num_workers=env.num_workers,
    )
    workflow = Train(
        env=env,
        model=model,
        data=data,
        optimizer=optimizer,
        total_epochs=conf["total_epochs"],
    )
    workflow.start()
