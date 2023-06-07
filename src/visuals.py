import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import random
import pathlib
from typing import Dict

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from math import ceil

from .utils import image_prediction
from .model import MultiLabelClassifier
from .custom_datasets import MultiLabelImageFolder


def plot_random_images(
    image_folder,
    n_display=15,
    ext=".jpg",
    figsize=(12, 12),
    seed=None
):
    random.seed(seed)
    n_cols, n_rows = 5, ceil(n_display / 5)
    image_paths = list(pathlib.Path(image_folder).rglob(f"**/*{ext}"))

    if len(image_paths) <= 0:
        raise ValueError(f"No Files with extension {ext} found")

    _, axs = plt.subplots(n_rows, n_cols, layout="constrained", figsize=figsize)
    axs = axs.ravel()  # flattening axs

    # Settings for all axis
    for ax in axs:
        ax.axis("off")

    for n in range(n_display):
        random_img_path = random.choice(image_paths)
        img = mpimg.imread(random_img_path)
        axs[n].imshow(img)
        