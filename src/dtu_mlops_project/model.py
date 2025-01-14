import functools
import typing

import timm
import torch
import typer
from torch import nn

app = typer.Typer()


@functools.cache
def get_dummy_model() -> typing.Callable:
    """
    Initializes a 'dummy' model ('mobilenetv4') for temporary use.

    :returns: A callable which can run the initialized model.
    """
    model_name = "mobilenetv4_conv_small.e2400_r224_in1k"
    mobilenetv4_model = timm.create_model(model_name, pretrained=True)
    mobilenetv4_model = mobilenetv4_model.eval()

    data_config = timm.data.resolve_model_data_config(mobilenetv4_model)
    transforms = timm.data.create_transform(**data_config, is_training=False)

    def _run_model(image, k: int = 5) -> tuple:
        """
        Runs the model on an image object with and computes the top 'k'
        probabilities and their associated class indices.

        :param image: The image to run the model on.
        :param k: The number of classes to return.
        :returns: a tuple of top 'k' class indices and their probabilities.
        """
        output = mobilenetv4_model(transforms(image).unsqueeze(0))

        probabilities, class_indices = torch.topk(output.softmax(dim=1) * 100, k=k)
        return probabilities, class_indices

    return _run_model


class MyAwesomeModel(nn.Module):
    """My awesome model."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.relu(self.conv3(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc1(x)


@app.command()
def model_info() -> None:
    """
    Print model architecture and number of parameters.
    """
    model = MyAwesomeModel()
    print(f"Model architecture: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    dummy_input = torch.randn(1, 1, 28, 28)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")


if __name__ == "__main__":
    app()
