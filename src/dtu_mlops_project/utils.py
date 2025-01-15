import torch


def get_device() -> str:
    """
    Selects the appropriate available device for performing computations.

    :returns: The name of the device as a string
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"
