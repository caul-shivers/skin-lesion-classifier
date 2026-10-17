import os
import urllib.request

import streamlit as st
import torch
from PIL import Image, ImageOps
from torchvision import transforms

from model import GoogleNetTransferModel, FINAL_FC_LAYERS
from gradcam import gradcam, overlay

MODEL_PATH = "best_model.pth"
MODEL_URL = "" #GitHub Release URL if the .pth is too big to commit
LABELS = ["Benign", "Malignant / pre-malignant"]

st.set_page_config(page_title="Skin Lesion Classifier", page_icon="🔬")
st.title("Skin Lesion Classifier")
st.warning(
    "Research/educational demo only. This is NOT a medical device and is not a "
    "substitute for evaluation by a clinician. Uploaded images are processed in "
    "memory and are not stored."
)

#Resize to 450x600 (H x W) -> Normalize to [-1, 1]
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((450, 600)),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH) and MODEL_URL:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    model = GoogleNetTransferModel(FINAL_FC_LAYERS, num_classes=2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))  # strict
    model.eval()
    return model


model = load_model()


def classify(img: Image.Image):
    """Returns (probs, cam). Grad-CAM explains the class the model predicted."""
    img = ImageOps.exif_transpose(img).convert("RGB")
    x = preprocess(img).unsqueeze(0)
    probs, cam, _ = gradcam(model, x)
    return img, probs, cam


tab_upload, tab_camera = st.tabs(["Upload", "Camera"])
with tab_upload:
    up = st.file_uploader("Upload a lesion image", type=["jpg", "jpeg", "png"])
with tab_camera:
    cam = st.camera_input("Take a picture of the lesion")

show_cam = st.checkbox("Show Grad-CAM heatmap (where the model looked)", value=True)

file = cam or up
if file:
    img, probs, cam_map = classify(Image.open(file))
    col1, col2 = st.columns(2) if show_cam else (st.container(), None)
    col1.image(img, caption="Input image", use_container_width=True)
    if show_cam:
        col2.image(overlay(img, cam_map), caption="Grad-CAM (red = most influential)",
                   use_container_width=True)
    st.subheader("Result")
    for label, p in zip(LABELS, probs):
        st.progress(float(p), text=f"{label}: {p:.1%}")
    if show_cam:
        st.caption("The heatmap is coarse and only shows which regions influenced the "
                   "prediction. It does not prove the model is looking at the right thing, "
                   "and highlighted areas are not a diagnosis.")
    st.caption("Probabilities come from a model trained on dermoscopic images (ISIC 2019). "
               "Phone photos differ from that data, so results may be unreliable.")
