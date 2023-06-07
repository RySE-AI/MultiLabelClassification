import pathlib
import logging
from collections import Counter

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics.metric import Metric

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
    

def check_img_shapes(image_folder: str,
                     ext = ".jpg") -> Counter:
    image_paths = pathlib.Path(image_folder).rglob(f"**/*{ext}")
    shapes = [cv2.imread(str(img_path)).shape[:2] for img_path in image_paths]
    
    return Counter(shapes)


def calculate_mean_stds_per_channel(image_folder,
                                    ext = ".jpg"):
    channels_mean = np.zeros(3)
    channels_std = np.zeros(3)

    image_paths = pathlib.Path(image_folder).rglob(f"**/*{ext}")
    
    if not image_paths:
        raise ValueError("There no images in the given folder")
    
    for count ,image_path in enumerate(image_paths, 1):
        image = cv2.imread(str(image_path))
        image = image.astype(np.float32) / 255.0  # Normalize pixel values

        channels_mean += np.mean(image, axis=(0, 1))
        channels_std += np.std(image, axis=(0, 1))

    channels_mean /= count
    channels_std /= count

    return channels_mean[::-1], channels_std[::-1] #rearrange for RGB Order


def test_model(model: nn.Module,
               metric: Metric,
               test_loader: DataLoader,
               device: str = "auto"):
    
    if device == "auto":
        device = check_gpu_available()
        print(f"Model will run on {device}")
    
    test_metrics = metric.to(device)
    model.to(device)
    model.eval()

    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.to(device), targets.to(device)

            # forward pass to get outputs
            preds = model(images)
            test_metrics.update(preds, targets)

    return test_metrics.compute()
