import functools
import typing

import timm
import torch
import typer

app = typer.Typer()


@app.command()
def model() -> None:
    """
    (Placeholder) Function for running the model.
    """


@functools.cache
def dummy_model(image, k: int = 5) -> typing.Tuple:
    """
    Initializes a 'dummy' model ('mobilenetv4')
    and runs it on an image (for temporary use).

    :param image: The image to run the model on.
    :param k: The number of classes to return.
    :returns: a tuple of top 'k' class indices and their probabilities.
    """
    model_name = 'mobilenetv4_conv_small.e2400_r224_in1k'
    mobilenetv4_model = timm.create_model(model_name, pretrained=True)
    mobilenetv4_model.eval()

    data_config = timm.data.resolve_model_data_config(mobilenetv4_model)
    transforms = timm.data.create_transform(**data_config, is_training=False)

    output = mobilenetv4_model(transforms(image).unsqueeze(0))

    probabilities, class_indices = torch.topk(output.softmax(dim=1) * 100, k=k)
    return probabilities, class_indices


if __name__ == '__main__':
    app()
