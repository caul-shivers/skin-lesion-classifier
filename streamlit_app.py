import os
import urllib.request

import streamlit as st
import torch
from PIL import Image, ImageOps
from torchvision import transforms

from model import GoogleNetTransferModel, FINAL_FC_LAYERS

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
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model


model = load_model()


def classify(img: Image.Image):
    img = ImageOps.exif_transpose(img).convert("RGB")
    x = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    return probs.tolist()


tab_upload, tab_camera = st.tabs(["Upload", "Camera"])
with tab_upload:
    up = st.file_uploader("Upload a lesion image", type=["jpg", "jpeg", "png"])
with tab_camera:
    cam = st.camera_input("Take a picture of the lesion")

file = cam or up
if file:
    img = Image.open(file)
    st.image(img, caption="Input image", use_container_width=True)
    with st.spinner("Classifying..."):
        probs = classify(img)
    st.subheader("Result")
    for label, p in zip(LABELS, probs):
        st.progress(float(p), text=f"{label}: {p:.1%}")
    st.caption("Probabilities come from a model trained on dermoscopic images (ISIC 2019). "
               "Phone photos differ from that data, so results may be unreliable.")
