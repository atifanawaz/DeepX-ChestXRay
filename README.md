# DeepX-ChestXray: AI-Powered Chest X-Ray Diagnostic System

## Project Overview
DeepX-ChestXray is an AI-powered web application designed for automated analysis of chest X-ray images. Using state-of-the-art convolutional neural networks (CNNs), the system performs classification of chest X-rays and provides explainable visualizations to help understand model predictions.

The platform is built with an interactive Streamlit interface, making it accessible for researchers, educators, and AI enthusiasts to explore AI-based medical imaging.  
You can access the live application here: [DeepX-ChestXray Web App](https://deepx-chestxray.streamlit.app/)

### Key Features
- Classification of chest X-rays into normal or abnormal/pathology-specific categories
- Explainable AI outputs using techniques such as GradCAM++ and feature attribution methods
- Real-time inference for immediate results
- Modular architecture allowing integration of new models or additional pathologies
- Visualization overlays highlighting regions of interest for interpretability

---

## Project Structure
````
DeepX-ChestXray/
├── app.py # Main Streamlit application script
├── models/ # Pre-trained CNN models
│ ├── cnn_model_final.keras
│ └── cnn_model_backup.keras
├── assets/ # Static frontend resources (styles, icons)
├── requirements.txt # Python dependencies
├── README.md # Project documentation
└── data/ # Optional folder for datasets

````

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/DeepX-ChestXray.git
cd DeepX-ChestXray
```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Run the application:
```
streamlit run app.py
```

## Dataset

Chest X-Ray Pneumonia Dataset from Kaggle:  
[https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)  

The dataset is organized into `train/`, `val/`, and `test/` folders with subfolders:  
- `NORMAL` — images of healthy patients  
- `PNEUMONIA` — images of patients diagnosed with pneumonia  

---

## Usage

Upload a chest X-ray image (JPEG/PNG). The app will display:
- Predicted classification (e.g., Normal, Pneumonia, TB)
- Explainability visualization (heatmap or GradCAM++ overlay)
- Probability scores for model predictions

---

## Methodology

- **Model Architecture:** Utilizes a CNN backbone (e.g., ResNet, DenseNet, or custom CNN) trained on chest X-ray datasets.
- **Preprocessing:** Input images are resized, normalized, and optionally converted from DICOM format.
- **Explainability:** GradCAM++ and feature attribution methods (like SHAP or Integrated Gradients) generate visual explanations of model focus areas.
- **Inference:** Real-time predictions are provided via Streamlit interface.
- **Deployment:** The web app allows interactive uploading, inference, and visualization without requiring code execution by the user.

---

## Intended Use

- Research and prototyping of AI in medical imaging
- Educational purposes to demonstrate AI interpretability
- Not intended for clinical diagnosis; all results should be validated by certified radiologists

---

## Limitations

- Performance depends heavily on training data diversity and quality
- Explainability overlays show areas the model focused on, not definitive diagnosis
- Privacy and regulatory compliance are critical when handling medical images; this app is for research/demo only

---

## Future Enhancements

- Multi-class pathology detection (e.g., pneumonia, tuberculosis, lung opacity)
- Support for DICOM medical imaging format
- External validation on independent datasets for generalization
- Confidence scoring / uncertainty estimation for predictions
- Structured report export in PDF or JSON summarizing predictions and visual explanations

---

## References

- Selvaraju, R.R., Cogswell, M., Das, A., et al., *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*, ICCV 2017.
- Lundberg, S.M., Lee, S.I., *A Unified Approach to Interpreting Model Predictions*, NIPS 2017.
