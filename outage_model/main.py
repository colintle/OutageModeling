import click

from . import preprocess_cli
from . import train_validate

cli = click.CommandCollection(sources=[
    preprocess_cli.PREPROCESS,
    train_validate.TRAIN_VALIDATE
])

if __name__ == "__main__":
    cli()