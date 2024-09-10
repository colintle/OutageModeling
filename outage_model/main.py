import click

from . import preprocess_cli
from . import train_validate_cli

cli = click.CommandCollection(sources=[
    preprocess_cli.PREPROCESS,
    train_validate_cli.TRAIN_VALIDATE
])

if __name__ == "__main__":
    cli()