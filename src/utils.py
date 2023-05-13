import torch
import logging


def check_gpu_available() -> torch.device:
    """This function checks which hardware backend is available on your device
    (mps or cuda) and returns the torch device. If no accelerator is available
    it will return a "cpu" device.

    Args:
        - none

    Returns:
        torch.device: Pytorch specific device depending on the available backend
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logging.info("Running on MPS. M1 GPU is available")

    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info("Running on Nvidia. Cuda is available")

    else:
        device = torch.device("cpu")
        logging.info("Running on CPU")

    return device