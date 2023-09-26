import typer
from .workflows import Train
import yaml

app = typer.Typer()


@app.command()
def train(path: str = "conf.yml"):
    workflow = Train(yaml.full_load(open(path)))
    workflow.start()


app()
