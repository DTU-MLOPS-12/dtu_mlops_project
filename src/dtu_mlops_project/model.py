import typer

app = typer.Typer()


@app.command()
def model() -> None:
    """
    (Placeholder) Function for running the model.
    """


if __name__ == '__main__':
    app()
