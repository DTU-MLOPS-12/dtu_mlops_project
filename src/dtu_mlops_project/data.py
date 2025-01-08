from pathlib import Path
from typing import Annotated

import typer
from torch.utils.data import Dataset

app = typer.Typer()


class MyDataset(Dataset):
    """My custom dataset."""

    def __init__(self, raw_data_path: Path) -> None:
        self.data_path = raw_data_path

    def __len__(self) -> int:
        """Return the length of the dataset."""

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""

    def preprocess(self, output_folder: Path) -> None:
        """Preprocess the raw data and save it to the output folder."""


@app.command()
def preprocess(
    raw_data_path: Annotated[Path, typer.Option("--raw-data")] = "data/raw/",
    output_folder: Annotated[Path, typer.Option(
        "--output-folder")] = "data/processed",
) -> None:
    """

    :param raw_data_path: Path-like object designating the location of the raw data
    :param output_folder: Path-lik object designating the location of the output of the preprocessing result.
    """
    print("Preprocessing data...")
    dataset = MyDataset(raw_data_path)
    dataset.preprocess(output_folder)


if __name__ == "__main__":
    app()
