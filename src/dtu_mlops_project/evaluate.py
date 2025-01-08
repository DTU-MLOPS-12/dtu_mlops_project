import typer

app = typer.Typer()


@app.command()
def evaluate() -> None:
    """
    (Placeholder) Function for evaluating the model.
    """


if __name__ == '__main__':
    app()
