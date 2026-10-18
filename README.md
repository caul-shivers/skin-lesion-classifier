# 🔬 Skin Lesion Classifier

A deep learning demo that classifies skin lesion images as **benign** or **malignant / pre-malignant**. Upload a photo or use your camera, and see a Grad-CAM heatmap of the regions that influenced the prediction.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://skin-lesion-classifier-nbfxrpsxnv3yeunftikr2x.streamlit.app)

> **Note:**⏳ First load may take a minute. The live demo runs on Streamlit Community Cloud’s free tier, which puts apps to sleep after a period of inactivity. If you see a “Yes, get this app back up!” button, click it, and give the app about 30–60 seconds to wake up. After that it responds normally.

> **Disclaimer:** Research/educational project only, not for diagnosis. Consult a dermatologist for any skin concern. Uploaded images are processed in memory and not stored.

## Table of Contents
1. [Model](#model)
2. [Performance](#performance)
3. [Limitations](#limitations)
4. [Comments](#comments)
5. [Future Improvements](#future-improvements)


## Model
- **Architecture:** Pre-trained GoogLeNet (frozen feature extractor) + custom head (64 → 32 → 8 → 2); ReLU + batch norm on the 64 and 8 layers
- **Task:** Binary classification (0 = benign: `bkl`, `df`, `nv`, `vasc`; 1 = malignant/pre-malignant: `ak`, `bcc`, `mel`, `scc`)
- **Input:** 450×600 RGB, normalized to [-1, 1]
- **Data:** ISIC 2019 (train/val/test, split by lesion); external check on MILK10k. ADD licenses and class balance

## Performance
![Confusion matrix](./figures/Confusion%20Matrix.png)
![Recall by lesion type](./figures/Binary%20Recall.png)
![F1 by lesion type](./figures/Binary%20F1%20Score.png)
![ROC Curve by lesion type](./figures/ROC%20Curve.png)

## Limitations
- Trained on dermoscopic images; phone photos may perform worse.
- Public skin lesion datasets skew toward lighter skin tones, so performance on darker skin is likely worse.
- Misses about 1 in 4 malignant lesions overall and about 1 in 3 melanomas, so a "benign" result must never be taken as true negative.
- Not clinically validated.

## Comments
- Too much time and resources spent training a fully-connected model for baseline comparison, which is not shown.
- Too much time and resources also spent training a convolutional feature extractor (not shown) before switching to transfer learning.
- Time spent here could have otherwise been spent fine-tuning the GoogLeNet feature extractor
- GoogLeNet is relatively outdated compared to modern architectures

## Future Improvements
- Fine-tune feature extractor: Train head, then unfreeze backbone layers and fine-tune with smaller learning rate.
- ResNet: replace GoogLeNet with ResNet backbone.
- Dropout: add dropout to force robust feature representation and avoid overfitting.
- Multi-task learning: instead of mapping final output directly to 2 binary classes (benign/malignant), train one head to predict the exact 8 lesion types and a parallel head to predict the 2 binary biopsy outcomes (benign/malignant). This would force backbone to learn granular lesion distinctions.
- Focal loss: replace cross entropy loss with focal loss. Due to the class imbalance (many more benign than malignant lesions), focal loss forces optimizer to skip easy negatives and concentrate gradient updates on harder, misclassified instances, which better matches the dataset due to the asymmetric risk of a false positive (uneccessary biopsy ordered on benign lesion) compared to false negative (malignant lesion goes untreated).
