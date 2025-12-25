# DeepX: AI-Powered Chest X-Ray Diagnostic System

## Overview
DeepX is an advanced, deployable deep learning system for **chest X-ray image classification**.  
It predicts whether an X-ray shows **NORMAL** or **PNEUMONIA**, with **explainable visualizations using GradCAM**.  

This project uses **CNN + Transfer Learning (DenseNet121)** and a **Streamlit frontend**, making it fully interactive and portfolio-ready.

---

## Features
- CNN with **DenseNet121 backbone** (Transfer Learning)  
- GradCAM visualization highlights areas influencing the prediction  
- Streamlit interface for **interactive image upload and prediction**  
- Clean modular project structure for professional presentation  

---

## Dataset
- **Chest X-Ray Pneumonia Dataset** from Kaggle:  
  [https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)  
- Images are organized into `train/val/test` folders with subfolders `NORMAL` and `PNEUMONIA`.

---

## Usage

### 1️⃣ Train the model
```bash
python model/train_model.py
