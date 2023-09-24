import typer
from .factory import train_from_conf

app = typer.Typer()


@app.command()
def train(path: str = "conf.yml"):
    train_from_conf(path)


app()
