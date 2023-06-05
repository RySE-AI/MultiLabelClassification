import torch
import logging
from .custom_datasets import MultiLabelImageFolder

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


def print_dataset_information(dataset_dir: str):
    image_dataset = MultiLabelImageFolder(dataset_dir)
    
    classes = ", ".join(image_dataset.classes)
    
    print("Dataset Summary")
    print(50*"-")
    print(f"\nThe dataset contains {len(image_dataset)} images")
    print(f"The possible classes are:\n{classes.title()}")
    print(f"\nThe folders are structured as followed:\n")
    for i, cls_combi in enumerate(image_dataset.cls_combinations):
        cls_combi = " and ".join(cls_combi)
        print(f"{i}: {cls_combi}")
        
    print("\nThese represents the possible class combinations in your dataset")
    print(50*"-")    