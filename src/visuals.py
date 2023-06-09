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
     
     
def plot_multilabel_confusion_matrix(
    cfs_matrices: torch.Tensor,
    idx_to_class: Dict,
    class_names = ["Pos", "Neg"],
    fontsize=14
) -> None:
    fig = plt.figure(figsize=(15, 10))
    cfs_matrices = cfs_matrices.cpu().numpy()
    n_matrices = len(cfs_matrices)
    
    heatmap_settings = {"annot": True,
                        "cmap": "crest_r",
                        "fmt": "d",
                        "cbar": False,
                        "linewidths": 0.8}

    for idx, cfs_matrix in enumerate(cfs_matrices):
        ax = fig.add_subplot(ceil(n_matrices / 3), 3, idx + 1)

        df_cm = pd.DataFrame(
            np.flip(cfs_matrix),
            index=class_names,
            columns=class_names,
        )
        
        try:
            heatmap = sns.heatmap(df_cm, ax=ax, **heatmap_settings)
        except ValueError:
            raise ValueError("Confusion matrix values must be integers.")
        heatmap.yaxis.set_ticklabels(
            heatmap.yaxis.get_ticklabels(), rotation=0, ha="right", fontsize=fontsize
        )
        heatmap.xaxis.set_ticklabels(
            heatmap.xaxis.get_ticklabels(), rotation=0, ha="right", fontsize=fontsize
        )

        ax.xaxis.tick_top()
        ax.set_ylabel("True label")
        ax.set_xlabel("Predicted label")
        ax.set_title("Confusion Matrix for the class - " + idx_to_class[idx])

    plt.tight_layout()


def plot_multilabel_predictions(
    model: MultiLabelClassifier,
    dataset: MultiLabelImageFolder,
    indices=None,
    n_images=8,
    pred_threshold=0.5,
    seed=None,
    figsize=(15,10)
    ) -> None:
    title_settings = {"fontsize": 8}
    thresh = nn.Threshold(pred_threshold, 0)
    fig = plt.figure(figsize=figsize)
    idx_to_class = dataset.idx_to_classes
    
    if not indices:
        random.seed(seed)
        indices = random.sample(range(len(dataset)), n_images)
    
    for ax_id, idx in enumerate(indices, 1):
        original_image = mpimg.imread(dataset.imgs[idx][0])
        image = dataset[idx][0]
        prediction = torch.sigmoid(image_prediction(model=model, image=image))
        
        real_target = dataset[idx][1].to("cpu").numpy()
        prediction = thresh(prediction)[0].to("cpu").numpy()
        
        target_indices = np.where(real_target != 0)[0]
        pred_indices = np.where(prediction != 0)[0]
        
        real_labels = " & ".join([idx_to_class[real_i] for real_i in target_indices])
        pred_labels = " & ".join([idx_to_class[pred_i] for pred_i in pred_indices])

        ax = fig.add_subplot(ceil(n_images / 4), 4, ax_id)
        ax.axis("off")
        ax.set_title(
            "Prediction: {}\n(Real: {})".format(pred_labels, real_labels),
            color=("green" if pred_labels == real_labels else "red"),
            fontdict=title_settings,
            )
        ax.imshow(original_image)
        
    plt.tight_layout()    
    plt.show()   