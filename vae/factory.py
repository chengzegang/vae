from .workflows import Train
import yaml


def train_from_conf(path: str):
    workflow = Train(yaml.full_load(open(path)))
    workflow.start()
