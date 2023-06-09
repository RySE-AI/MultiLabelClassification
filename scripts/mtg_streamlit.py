import sys
from pathlib import Path
import json

sys.path.append(str(Path(__file__).parents[1]))

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
import seaborn as sns
from matplotlib import colors

from src.model import MultiLabelClassifier
from src.utils import image_prediction

MODEL_CHECKPOINT = "model/epoch=47-step=2016.ckpt"
IDX_TO_CLASSES = "model/idx_to_classes.json"
ORDER = [
    "black",
    "blue",
    "green",
    "red",
    "white",
    "artifact",
    "planeswalker",
    "creature",
    "special",
    "mana",
]


@st.cache_resource
def load_model():
    model = MultiLabelClassifier.load_from_checkpoint(
        MODEL_CHECKPOINT, map_location="cpu"
    )
    return model.eval()


@st.cache_data
def load_prediction_dict():
    with open(IDX_TO_CLASSES) as f:
        data = json.load(f)
    return {int(key): val for key, val in data.items()}


@st.cache_data
def preprocess_image(image):
    transform_img = transforms.Compose(
        [
            transforms.Resize((256, 320)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    input_tensor = transform_img(image).unsqueeze(0)

    return input_tensor


@st.cache_data
def make_prediction(_model, image):
    image = preprocess_image(image=image)
    logits = image_prediction(_model, image, "cpu")
    pred = torch.sigmoid(logits).squeeze().numpy()

    return pred


def process_model_prediction(model_pred, threshold, col, idx_to_class):
    df_pred = pd.DataFrame(
        model_pred, columns=["confidence"], index=list(idx_to_class.values())
    ).reindex(ORDER)
    df_pred.index.name = "Class"
    cmap = sns.light_palette("seagreen", as_cmap=True)
    df_pred = df_pred.style.background_gradient(cmap=cmap, vmin=0.0, vmax=1.0)

    col.dataframe(df_pred)

    pred_indices = np.where(model_pred >= threshold)[0]
    pred_labels = ", ".join(
        [idx_to_class[pred_i].title() for pred_i in pred_indices]
    )

    col.write(f"**Predicted label(s):**")
    col.write(pred_labels)


def load_image(col):
    my_upload = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if my_upload is not None:
        image_data = my_upload.getvalue()
        col.image(image_data)
        return Image.open(my_upload).convert("RGB")  # mode=RGBA
    else:
        return None


def streamlit_settings():
    st.set_page_config(page_title="MTG Multi Label Classification")
    st.title("MTG Multi-Label Classification")


def main():
    streamlit_settings()
    main_col1, main_col2 = st.columns([0.5, 0.5])

    model = load_model()
    idx_to_classes = load_prediction_dict()
    image = load_image(main_col1)
    threshold = st.sidebar.slider("Select Prediction Threshold", 0.0, 1.0, 0.5)

    if image is not None:
        prediction = make_prediction(model, image)
        process_model_prediction(prediction, threshold, main_col2, idx_to_classes)

        st.markdown(
            """<hr style="height:2px;border:none;color:#333;background-color:#333;" /> """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
