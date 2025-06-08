---
title: CTScan-LungCancer-ResNet50
emoji: 🫁
colorFrom: blue
colorTo: red
sdk: gradio
app_file: app.py
pinned: true
license: mit
tags:
  - lung cancer
  - CT scan
  - medical imaging
  - resnet50
  - gradcam
  - image-classification
  - healthcare
  - tensorflow
---

[![HF Spaces](https://img.shields.io/badge/🤗%20HuggingFace-Space-blue?logo=huggingface&style=flat-square)](https://github.com/McKlay/lung-cancer-detection-resnet50)
[![Gradio](https://img.shields.io/badge/Built%20with-Gradio-orange?logo=gradio&style=flat-square)](https://www.gradio.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

![GitHub last commit](https://img.shields.io/github/last-commit/McKlay/lung-cancer-detection-resnet50)
![GitHub Repo stars](https://img.shields.io/github/stars/McKlay/lung-cancer-detection-resnet50?style=social)
![GitHub forks](https://img.shields.io/github/forks/McKlay/lung-cancer-detection-resnet50?style=social)
![MIT License](https://img.shields.io/github/license/McKlay/lung-cancer-detection-resnet50)

![Visitors](https://visitor-badge.laobi.icu/badge?page_id=McKlay.lung-cancer-detection-resnet50)

# 🫁 CT Scan Lung Cancer Detection (ResNet50 + Grad-CAM)

**CTScan-LungCancer-ResNet50** is an AI-powered tool that detects lung cancer from chest CT scan images using a fine-tuned ResNet50 model. It also visualizes model attention with Grad-CAM heatmaps to help interpret predictions.

> **Upload a CT scan (.JPG/.PNG)**  
> The model predicts **Cancer / No Cancer**  
> A Grad-CAM heatmap highlights regions that influenced the decision

---

## 🌐 Demo

**Deployed on Hugging Face Spaces:** [![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/raw/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/McKlay/CTScan-LungCancer-ResNet50)

---

## Model Details

- **Base Model:** ResNet50 (`imagenet` weights)
- **Custom Head:** GlobalAveragePooling → Dense(128, relu) → Dropout(0.3) → Dense(1, sigmoid)
- **Input Shape:** 224×224 RGB
- **Dataset:** Public CT scan dataset from Kaggle
- **Training:** 10 epochs, ~96% validation accuracy

---

## 📓 Training Notebook (Kaggle)

Model was fine-tuned on Kaggle using the notebook below:

[Fine-Tuning ResNet50 for Lung Cancer Detection](https://www.kaggle.com/code/claymarksarte/fine-tuning-resnet50-for-lung-cancer-detection)

Includes:
- Custom label preprocessing
- Data augmentation
- Model training history plots
- Confusion matrix & classification report

---

## Features

- Binary classification: `Cancer` or `No Cancer`
- Shows confidence score (0–100%)
- Grad-CAM heatmap overlay for explainability
- Real-time predictions via Gradio

---

## Folder Structure

```

CTScan-LungCancer-ResNet50/
├── app.py                 # Gradio UI logic
├── model/
│   └── resnet50_lung_model.h5
├── utils.py               # Preprocessing and Grad-CAM
├── requirements.txt
├── README.md
└── .gitignore

````

---

## 📷 Example Output

| CT Scan Image | Grad-CAM Heatmap |
|---------------|------------------|
| ![CT Scan](https://huggingface.co/datasets/McKlay/documentation-images/resolve/main/ctscan-resnet50/cancer-scan.jpg) | ![Grad-CAM](https://huggingface.co/datasets/McKlay/documentation-images/resolve/main/ctscan-resnet50/cancer-heatmap.jpg) |

---

## Installation

You can run this locally with:

```bash
pip install -r requirements.txt
python app.py
````

---

## Requirements

```
tensorflow
gradio
numpy
opencv-python
matplotlib
Pillow
```

---

## 👨‍💻 Author

Developed by [Clay Mark Sarte](https://github.com/McKlay)
Powered by Kaggle + TensorFlow + Gradio + Hugging Face

---

## ⚠️ Disclaimer

> **This tool is for educational and research use only.**
> It is **not a substitute for professional medical advice** or diagnosis. Always consult a certified medical expert.
