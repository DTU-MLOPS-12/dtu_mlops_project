from typing import Annotated
import typer


app = typer.Typer()


@app.command()
def train(output: Annotated[str, typer.Option("--output", "-o")] = "model.ckpt"):
    """
    (Placeholder) Trains the model on the data.

    :param output: Destination to write the results to.
    """


if __name__ == '__main__':
    app()
