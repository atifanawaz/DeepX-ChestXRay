# import streamlit as st
# import tensorflow as tf
# import numpy as np
# from PIL import Image
# import cv2
# import os
# import shap

# from tf_keras_vis.gradcam import Gradcam
# from tf_keras_vis.saliency import Saliency
# from tf_keras_vis.utils.scores import BinaryScore
# from tensorflow.keras.models import load_model

# # -------------------------------------------------
# # Streamlit Configuration & Custom Styling
# # -------------------------------------------------


# st.set_page_config(
#     page_title="DeepX: Chest X-Ray Diagnostic",
#     page_icon="🩺",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# st.markdown("""
# <style>
#     /* Main background - soft cream with peachy gradient */
#     .stApp {
#         background: linear-gradient(135deg, #FFF5F3 0%, #FFE8E3 50%, #FFF0EB 100%);
#     }
    
#     /* Header styling - warm coral/salmon gradient */
#     .main-header {
#         color: #FFA07A;
#         font-size: 3rem;
#         font-weight: 800;
#         text-align: center;
#         margin-bottom: 0.5rem;
#         letter-spacing: -1px;
#     }
    
#     .sub-header {
#         color: #6B5B5B;
#         text-align: center;
#         font-size: 1.1rem;
#         margin-bottom: 2rem;
#         font-weight: 400;
#     }
    
#     /* Card styling - white cards with warm shadows */
#     .card {
#         background: #ffffff;
#         border: 1px solid #FFE0D6;
#         border-radius: 20px;
#         padding: 1.5rem;
#         margin-bottom: 1rem;
#         box-shadow: 0 4px 20px rgba(255, 107, 107, 0.08);
#     }
    
#     .card-title {
#         color: #3D3D3D;
#         font-size: 1.25rem;
#         font-weight: 600;
#         margin-bottom: 1rem;
#         display: flex;
#         align-items: center;
#         gap: 0.5rem;
#     }
    
#     /* Upload area styling - peachy dashed border */
#     .upload-area {
#         border: 2px dashed #FFB5A7;
#         border-radius: 20px;
#         padding: 3rem;
#         text-align: center;
#         background: rgba(255, 181, 167, 0.05);
#         transition: all 0.3s ease;
#     }
    
#     .upload-area:hover {
#         background: rgba(255, 181, 167, 0.12);
#         border-color: #FF8E72;
#     }
    
#     /* Result badges - soft green for normal, coral for pneumonia */
#     .result-normal {
#         background: linear-gradient(135deg, #7BC67E 0%, #5AB55E 100%);
#         color: white;
#         padding: 1rem 2rem;
#         border-radius: 16px;
#         font-size: 1.5rem;
#         font-weight: 700;
#         text-align: center;
#         box-shadow: 0 4px 15px rgba(123, 198, 126, 0.35);
#     }
    
#     .result-pneumonia {
#         background: linear-gradient(135deg, #FF6B6B 0%, #FF8E72 100%);
#         color: white;
#         padding: 1rem 2rem;
#         border-radius: 16px;
#         font-size: 1.5rem;
#         font-weight: 700;
#         text-align: center;
#         box-shadow: 0 4px 15px rgba(255, 107, 107, 0.35);
#     }
    
#     /* Confidence meter */
#     .confidence-container {
#         background: #FFF5F3;
#         border-radius: 16px;
#         padding: 1rem;
#         margin-top: 1rem;
#     }
    
#     .confidence-label {
#         color: #8B7575;
#         font-size: 0.875rem;
#         margin-bottom: 0.5rem;
#     }
    
#     .confidence-bar {
#         height: 10px;
#         border-radius: 5px;
#         background: #FFE0D6;
#         overflow: hidden;
#     }
    
#     .confidence-fill {
#         height: 100%;
#         border-radius: 5px;
#         transition: width 0.5s ease;
#     }
    
#     /* Section headers - coral accent */
#     .section-header {
#         color: #3D3D3D;
#         font-size: 1.5rem;
#         font-weight: 700;
#         margin: 2rem 0 1rem 0;
#         padding-bottom: 0.5rem;
#         border-bottom: 3px solid #FFB5A7;
#     }
    
#     /* Info box - soft peachy accent */
#     .info-box {
#         background: linear-gradient(135deg, rgba(255, 181, 167, 0.15) 0%, rgba(255, 142, 114, 0.08) 100%);
#         border-left: 4px solid #FF8E72;
#         padding: 1rem;
#         border-radius: 0 16px 16px 0;
#         color: #5D4E4E;
#         margin: 1rem 0;
#     }
    
#     /* Image container */
#     .image-container {
#         background: #ffffff;
#         border-radius: 16px;
#         padding: 1rem;
#         border: 1px solid #FFE0D6;
#         box-shadow: 0 2px 12px rgba(255, 107, 107, 0.06);
#     }
    
#     .image-label {
#         color: #8B7575;
#         font-size: 0.875rem;
#         text-align: center;
#         margin-top: 0.75rem;
#         font-weight: 500;
#     }
    
#     /* Sidebar styling - warm dark theme */
#     [data-testid="stSidebar"] {
#         background: linear-gradient(180deg, #3D3D3D 0%, #4A4040 100%);
#         border-right: 1px solid rgba(255, 181, 167, 0.1);
#     }
    
#     [data-testid="stSidebar"] .block-container {
#         padding-top: 2rem;
#     }
    
#     /* Feature list - coral accent */
#     .feature-item {
#         display: flex;
#         align-items: center;
#         gap: 0.75rem;
#         padding: 0.75rem;
#         background: rgba(255, 181, 167, 0.12);
#         border-radius: 12px;
#         margin-bottom: 0.5rem;
#         color: #F5E6E0;
#         border: 1px solid rgba(255, 181, 167, 0.2);
#     }
    
#     .feature-icon {
#         font-size: 1.25rem;
#     }
    
#     /* Metric cards */
#     .metric-card {
#         background: #ffffff;
#         border-radius: 16px;
#         padding: 1.25rem;
#         text-align: center;
#         border: 1px solid #FFE0D6;
#         box-shadow: 0 2px 12px rgba(255, 107, 107, 0.06);
#     }
    
#     .metric-value {
#         font-size: 2rem;
#         font-weight: 700;
#         color: #FF6B6B;
#     }
    
#     .metric-label {
#         color: #8B7575;
#         font-size: 0.875rem;
#         margin-top: 0.25rem;
#     }
    
#     /* Hide Streamlit branding */
#     #MainMenu {visibility: hidden;}
#     footer {visibility: hidden;}
    
#     /* File uploader styling - peachy accent */
#     [data-testid="stFileUploader"] {
#         background: #ffffff;
#         border-radius: 16px;
#         padding: 1rem;
#         border: 2px dashed #FFB5A7;
#     }
    
#     [data-testid="stFileUploader"]:hover {
#         border-color: #FF8E72;
#         background: rgba(255, 181, 167, 0.05);
#     }
    
#     /* Divider - soft peachy gradient */
#     .divider {
#         height: 1px;
#         background: linear-gradient(90deg, transparent, #FFD0C4, transparent);
#         margin: 2rem 0;
#     }
    
#     /* Spinner override */
#     .stSpinner > div {
#         border-top-color: #FF8E72 !important;
#     }
    
#     /* Button styling */
#     .stButton > button {
#         background: linear-gradient(135deg, #FF6B6B 0%, #FF8E72 100%);
#         color: white;
#         border: none;
#         border-radius: 12px;
#         padding: 0.5rem 1.5rem;
#         font-weight: 600;
#     }
    
#     .stButton > button:hover {
#         background: linear-gradient(135deg, #FF5252 0%, #FF7A5C 100%);
#         box-shadow: 0 4px 15px rgba(255, 107, 107, 0.3);
#     }
# </style>
# """, unsafe_allow_html=True)



# # -------------------------------------------------
# # Sidebar
# # -------------------------------------------------
# with st.sidebar:
#     st.markdown("""
#         <div style="text-align: center; padding: 1rem 0;">
#             <span style="font-size: 3rem;">💉</span>
#             <h2 style="color: #F5E6E0; margin-top: 0.5rem; font-weight: 700;">DeepX</h2>
#             <p style="color: #C4B0A8; font-size: 0.9rem;">AI-Powered Diagnostics</p>
#         </div>
#     """, unsafe_allow_html=True)
    
#     st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
#     st.markdown("""
#         <p style="color: #C4B0A8; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 1rem;">
#             Features
#         </p>
#     """, unsafe_allow_html=True)
    
#     features = [
#         ("🔬", "GradCAM++ Visualization"),
#         ("🧾", "Integrated Gradients"),
#         ("⚡", "Real-time Analysis"),
#         ("🛡️", "High Accuracy CNN")
#     ]
    
#     for icon, text in features:
#         st.markdown(f"""
#             <div class="feature-item">
#                 <span class="feature-icon">{icon}</span>
#                 <span>{text}</span>
#             </div>
#         """, unsafe_allow_html=True)
    
#     st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
#     st.markdown("""
#         <div style="background: rgba(255, 181, 167, 0.12); border-left: 4px solid #FF8E72; 
#                     padding: 1rem; border-radius: 0 12px 12px 0;">
#             <strong style="color: #FFB5A7;">About</strong><br><br>
#             <span style="color: #D9C9C3;">DeepX uses advanced deep learning to analyze chest X-rays and provide 
#             interpretable AI explanations for medical professionals.</span>
#         </div>
#     """, unsafe_allow_html=True)
    
#     st.markdown("""
#         <div style="margin-top: 2rem; padding: 1rem; background: rgba(255, 181, 167, 0.15); 
#                     border-radius: 12px; border: 1px solid rgba(255, 181, 167, 0.3);">
#             <p style="color: #E8C4B8; font-size: 0.8rem; margin: 0;">
#                 <strong>Disclaimer:</strong> This tool is for educational purposes only. 
#                 Always consult a qualified medical professional.
#             </p>
#         </div>
#     """, unsafe_allow_html=True)

# # -------------------------------------------------
# # Main Header
# # -------------------------------------------------
# st.markdown('<h1 class="main-header">💉 DeepX Diagnostic System</h1>', unsafe_allow_html=True)
# st.markdown("""
#   <p class="sub-header">
#       Advanced AI-powered chest X-ray analysis with explainable visualizations
#   </p>
# """, unsafe_allow_html=True)


# # -------------------------------------------------
# # Load Model
# # -------------------------------------------------
# @st.cache_resource(show_spinner=False)
# def load_cnn_model(path):
#     if not os.path.exists(path):
#         st.error(f"Model file NOT FOUND: {path}")
#         st.stop()
#     return load_model(path)
    

# model_path = "model/cnn_model_final_hdf5.h5"

# with st.spinner("🔄 Loading AI Model..."):
#     model = load_cnn_model(model_path)



# # -------------------------------------------------
# # Upload Section
# # -------------------------------------------------
# st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# col_upload, col_info = st.columns([2, 1])

# with col_upload:
#     st.markdown("""
#         <div class="card">
#             <div class="card-title">📤 Upload X-Ray Image</div>
#             <p style="color: #94a3b8; margin-bottom: 1rem;">
#                 Drag and drop or click to upload a chest X-ray image (PNG, JPG, JPEG)
#             </p>
#         </div>
#     """, unsafe_allow_html=True)
    
#     uploaded_file = st.file_uploader(
#         "Choose a chest X-ray image",
#         type=["png", "jpg", "jpeg"],
#         label_visibility="collapsed"
#     )

# with col_info:
#     st.markdown("""
#         <div class="card">
#             <div class="card-title">📋 Guidelines</div>
#             <ul style="color: #94a3b8; padding-left: 1.25rem; margin: 0;">
#                 <li style="margin-bottom: 0.5rem;">Use frontal chest X-rays</li>
#                 <li style="margin-bottom: 0.5rem;">Ensure good image quality</li>
#                 <li style="margin-bottom: 0.5rem;">Supported: PNG, JPG, JPEG</li>
#                 <li>Max recommended: 1024x1024px</li>
#             </ul>
#         </div>
#     """, unsafe_allow_html=True)

# # -------------------------------------------------
# # SAFE IMAGE PIPELINE (THIS IS WHAT FIXES YOUR CRASH)
# # -------------------------------------------------
# if uploaded_file is not None:
#     from PIL import Image
#     import numpy as np

#     try:
#         img = Image.open(uploaded_file).convert("RGB")
#         img_resized = img.resize((224, 224))

#         st.image(img_resized, width=400)

#         img_array = np.array(img_resized, dtype=np.float32) / 255.0
#         img_array = np.expand_dims(img_array, axis=0)

#     except Exception as e:
#         st.error(f"Image processing failed: {e}")
#         st.stop()

# # -------------------------------------------------
# # Analysis Section
# # -------------------------------------------------
# if uploaded_file:
#     st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
#     img = Image.open(uploaded_file).convert("RGB")
#     img_resized = img.resize((224, 224))
#     img_array = np.expand_dims(np.array(img_resized) / 255.0, axis=0)

#     # -------------------------------------------------
#     # Prediction with Progress
#     # -------------------------------------------------
#     with st.spinner("🔍 Analyzing X-ray..."):
#         pred = model.predict(img_array, verbose=0)[0][0]
    
#     if pred > 0.5:
#         label = "PNEUMONIA"
#         class_index = 1
#         confidence = pred
#         result_class = "result-pneumonia"
#         result_icon = "⚠️"
#     else:
#         label = "NORMAL"
#         class_index = 0
#         confidence = 1 - pred
#         result_class = "result-normal"
#         result_icon = "✔"

#     st.success(f"Prediction Confidence: {confidence*100:.2f}%")

#     # Results Display
#     col_result, col_image = st.columns([1, 1])
    
#     with col_result:
#         st.markdown(f"""
#             <div class="card" style="height: 100%;">
#                 <div class="card-title">🧾 Diagnosis Result</div>
#                 <div class="{result_class}">
#                     {result_icon} {label}
#                 </div>
#                 <div class="confidence-container">
#                     <div class="confidence-label">Confidence Score</div>
#                     <div style="display: flex; align-items: center; gap: 1rem;">
#                         <div class="confidence-bar" style="flex: 1;">
#                             <div class="confidence-fill" style="width: {confidence*100}%; 
#                                 background: {'#7BC67E' if label == 'NORMAL' else '#FF6B6B'};"></div>
#                         </div>
#                         <span style="color: #3D3D3D; font-weight: 700; font-size: 1.25rem;">
#                             {confidence*100:.1f}%
#                         </span>
#                     </div>
#                 </div>
#             </div>
#         """, unsafe_allow_html=True)
    
#     with col_image:
#         st.markdown("""
#             <div class="card">
#                 <div class="card-title">📥 Uploaded X-Ray </div>
#             </div>
#         """, unsafe_allow_html=True)
#         st.image(img_resized, width=400)

#     st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    
#     # -------------------------------------------------
#     # GradCAM++ & Integrated Gradients
#     # -------------------------------------------------
#     st.markdown('<div class="section-header">🎨 Explainability Maps</div>', unsafe_allow_html=True)

#     with st.spinner("🔥 Generating GradCAM++ and Integrated Gradients..."):
#         # GradCAM++
#         gradcam = Gradcam(model, clone=True)
#         score = BinaryScore(target_values=[class_index])
#         cam = gradcam(score, img_array)[0]

#         heatmap = cv2.resize(cam, (224, 224))
#         heatmap_uint8 = np.uint8(255 * heatmap)
#         heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
#         overlay_gradcam = cv2.addWeighted(np.array(img_resized), 0.6, heatmap_color, 0.4, 0)

#         # Integrated Gradients
#         saliency = Saliency(model)
#         saliency_map = saliency(score, img_array)[0]
#         heatmap_saliency = cv2.resize(saliency_map, (224, 224))
#         heatmap_saliency_uint8 = np.uint8(255 * heatmap_saliency)
#         heatmap_saliency_color = cv2.applyColorMap(heatmap_saliency_uint8, cv2.COLORMAP_HOT)
#         overlay_ig = cv2.addWeighted(np.array(img_resized), 0.6, heatmap_saliency_color, 0.4, 0)

#     col3, col4 = st.columns(2)
#     with col3:
#         st.markdown("""
#             <div class="image-container">
#         """, unsafe_allow_html=True)
#         st.image(overlay_gradcam, width=450)
#         st.markdown(f"""
#                 <div class="image-label">GradCAM++ • {label}</div>
#             </div>
#         """, unsafe_allow_html=True)
#         st.markdown("""
#             <div class="info-box" style="margin-top: 1rem;">
#                 <strong>🔥 GradCAM++:</strong> Uses gradient information to highlight important regions. 
#                 Red/yellow areas strongly influenced the prediction.
#             </div>
#         """, unsafe_allow_html=True)

#     with col4:
#         st.markdown("""
#             <div class="image-container">
#         """, unsafe_allow_html=True)
#         st.image(overlay_ig, width=450)
#         st.markdown(f"""
#                 <div class="image-label">Integrated Gradients • {label}</div>
#             </div>
#         """, unsafe_allow_html=True)
#         st.markdown("""
#             <div class="info-box" style="margin-top: 1rem;">
#                 <strong>⚡ Integrated Gradients:</strong> Attributes predictions to input features. 
#                 Brighter areas indicate higher pixel importance.
#             </div>
#         """, unsafe_allow_html=True)

#     # -------------------------------------------------
#     # Summary Metrics
#     # -------------------------------------------------
#     st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
#     st.markdown('<div class="section-header">📈 Analysis Summary</div>', unsafe_allow_html=True)
    
#     col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    
#     with col_m1:
#         st.markdown(f"""
#             <div class="metric-card">
#                 <div class="metric-value">{confidence*100:.1f}%</div>
#                 <div class="metric-label">Confidence</div>
#             </div>
#         """, unsafe_allow_html=True)
    
#     with col_m2:
#         st.markdown(f"""
#             <div class="metric-card">
#                 <div class="metric-value" style="color: {'#7BC67E' if label == 'NORMAL' else '#FF6B6B'};">
#                     {label}
#                 </div>
#                 <div class="metric-label">Diagnosis</div>
#             </div>
#         """, unsafe_allow_html=True)
    
#     with col_m3:
#         st.markdown("""
#             <div class="metric-card">
#                 <div class="metric-value">3</div>
#                 <div class="metric-label">XAI Methods</div>
#             </div>
#         """, unsafe_allow_html=True)
    
#     with col_m4:
#         st.markdown("""
#             <div class="metric-card">
#                 <div class="metric-value">224×224</div>
#                 <div class="metric-label">Resolution</div>
#             </div>
#         """, unsafe_allow_html=True)

# else:
#     st.markdown("""
#         <div style="text-align: center; padding: 4rem 2rem; 
#                     background: #ffffff; 
#                     border-radius: 16px; 
#                     border: 2px dashed #FFB5A7;
#                     margin-top: 2rem;
#                     box-shadow: 0 4px 6px -1px rgba(255, 107, 107, 0.05);">
#             <span style="font-size: 4rem; opacity: 0.6;">📷</span>
#             <h3 style="color: #3D3D3D; margin-top: 1rem; font-weight: 500;">
#                 No X-Ray Uploaded
#             </h3>
#             <p style="color: #8B7575; max-width: 400px; margin: 0 auto;">
#                 Upload a chest X-ray image to begin AI-powered diagnostic analysis
#             </p>
#         </div>
#     """, unsafe_allow_html=True)

# st.markdown("""
#     <div style="text-align: center; padding: 2rem; margin-top: 3rem; 
#                 border-top: 1px solid #FFE0D6;">
#         <p style="color: #8B7575; font-size: 0.875rem;">
#             DeepX Diagnostic System | Powered by TensorFlow & Streamlit<br>
#             <span style="font-size: 0.75rem; color: #C4B0A8;">For educational and research purposes only</span>
#         </p>
#     </div>
# """, unsafe_allow_html=True)

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import os
import shap

from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.saliency import Saliency
from tf_keras_vis.utils.scores import BinaryScore
from tensorflow.keras.models import load_model

# -------------------------------------------------
# Streamlit Configuration & Custom Styling
# -------------------------------------------------


st.set_page_config(
    page_title="DeepX: Chest X-Ray Diagnostic",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Main background - soft cream with peachy gradient */
    .stApp {
        background: linear-gradient(135deg, #FFF5F3 0%, #FFE8E3 50%, #FFF0EB 100%);
    }
    
    /* Header styling - warm coral/salmon gradient */
    .main-header {
        color: #FFA07A;
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
    }
    
    .sub-header {
        color: #6B5B5B;
        text-align: center;
        font-size: 1.1rem;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* Card styling - white cards with warm shadows */
    .card {
        background: #ffffff;
        border: 1px solid #FFE0D6;
        border-radius: 20px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 20px rgba(255, 107, 107, 0.08);
    }
    
    .card-title {
        color: #3D3D3D;
        font-size: 1.25rem;
        font-weight: 600;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Upload area styling - peachy dashed border */
    .upload-area {
        border: 2px dashed #FFB5A7;
        border-radius: 20px;
        padding: 3rem;
        text-align: center;
        background: rgba(255, 181, 167, 0.05);
        transition: all 0.3s ease;
    }
    
    .upload-area:hover {
        background: rgba(255, 181, 167, 0.12);
        border-color: #FF8E72;
    }
    
    /* Result badges - soft green for normal, coral for pneumonia */
    .result-normal {
        background: linear-gradient(135deg, #7BC67E 0%, #5AB55E 100%);
        color: white;
        padding: 1rem 2rem;
        border-radius: 16px;
        font-size: 1.5rem;
        font-weight: 700;
        text-align: center;
        box-shadow: 0 4px 15px rgba(123, 198, 126, 0.35);
    }
    
    .result-pneumonia {
        background: linear-gradient(135deg, #FF6B6B 0%, #FF8E72 100%);
        color: white;
        padding: 1rem 2rem;
        border-radius: 16px;
        font-size: 1.5rem;
        font-weight: 700;
        text-align: center;
        box-shadow: 0 4px 15px rgba(255, 107, 107, 0.35);
    }
    
    /* Confidence meter */
    .confidence-container {
        background: #FFF5F3;
        border-radius: 16px;
        padding: 1rem;
        margin-top: 1rem;
    }
    
    .confidence-label {
        color: #8B7575;
        font-size: 0.875rem;
        margin-bottom: 0.5rem;
    }
    
    .confidence-bar {
        height: 10px;
        border-radius: 5px;
        background: #FFE0D6;
        overflow: hidden;
    }
    
    .confidence-fill {
        height: 100%;
        border-radius: 5px;
        transition: width 0.5s ease;
    }
    
    /* Section headers - coral accent */
    .section-header {
        color: #3D3D3D;
        font-size: 1.5rem;
        font-weight: 700;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #FFB5A7;
    }
    
    /* Info box - soft peachy accent */
    .info-box {
        background: linear-gradient(135deg, rgba(255, 181, 167, 0.15) 0%, rgba(255, 142, 114, 0.08) 100%);
        border-left: 4px solid #FF8E72;
        padding: 1rem;
        border-radius: 0 16px 16px 0;
        color: #5D4E4E;
        margin: 1rem 0;
    }
    
    /* Image container */
    .image-container {
        background: #ffffff;
        border-radius: 16px;
        padding: 1rem;
        border: 1px solid #FFE0D6;
        box-shadow: 0 2px 12px rgba(255, 107, 107, 0.06);
    }
    
    .image-label {
        color: #8B7575;
        font-size: 0.875rem;
        text-align: center;
        margin-top: 0.75rem;
        font-weight: 500;
    }
    
    /* Sidebar styling - warm dark theme */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #3D3D3D 0%, #4A4040 100%);
        border-right: 1px solid rgba(255, 181, 167, 0.1);
    }
    
    [data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
    }
    
    /* Feature list - coral accent */
    .feature-item {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 0.75rem;
        background: rgba(255, 181, 167, 0.12);
        border-radius: 12px;
        margin-bottom: 0.5rem;
        color: #F5E6E0;
        border: 1px solid rgba(255, 181, 167, 0.2);
    }
    
    .feature-icon {
        font-size: 1.25rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: #ffffff;
        border-radius: 16px;
        padding: 1.25rem;
        text-align: center;
        border: 1px solid #FFE0D6;
        box-shadow: 0 2px 12px rgba(255, 107, 107, 0.06);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #FF6B6B;
    }
    
    .metric-label {
        color: #8B7575;
        font-size: 0.875rem;
        margin-top: 0.25rem;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* File uploader styling - peachy accent */
    [data-testid="stFileUploader"] {
        background: #ffffff;
        border-radius: 16px;
        padding: 1rem;
        border: 2px dashed #FFB5A7;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #FF8E72;
        background: rgba(255, 181, 167, 0.05);
    }
    
    /* Divider - soft peachy gradient */
    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, #FFD0C4, transparent);
        margin: 2rem 0;
    }
    
    /* Spinner override */
    .stSpinner > div {
        border-top-color: #FF8E72 !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #FF6B6B 0%, #FF8E72 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #FF5252 0%, #FF7A5C 100%);
        box-shadow: 0 4px 15px rgba(255, 107, 107, 0.3);
    }
    /* ================================================
       LANDING PAGE STYLES (Medicy-inspired, green theme)
       ================================================ */
    .lp-wrap { font-family: inherit; }

    .lp-navbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1rem 1.75rem;
        background: #ffffff;
        border-radius: 20px;
        border: 1px solid #E3F2E8;
        box-shadow: 0 4px 20px rgba(22, 163, 74, 0.06);
        margin-bottom: 1.5rem;
        flex-wrap: wrap;
        gap: 0.75rem;
    }
    .lp-logo { font-size: 1.4rem; font-weight: 800; color: #16241c; }
    .lp-logo span { color: #16a34a; }
    .lp-navlinks a {
        color: #4b5a52;
        text-decoration: none;
        margin: 0 0.9rem;
        font-weight: 500;
        font-size: 0.95rem;
    }
    .lp-navlinks a:hover { color: #16a34a; }

    .lp-hero {
        background: linear-gradient(135deg, #EAFBF1 0%, #F3FFF7 60%, #EFFDF4 100%);
        border-radius: 32px;
        padding: 3rem 2.5rem 1rem 2.5rem;
        text-align: center;
        border: 1px solid #E3F2E8;
    }
    .lp-pill {
        display: inline-block;
        background: #ffffff;
        color: #16a34a;
        padding: 0.4rem 1.1rem;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 700;
        border: 1px solid #CFF3DB;
        margin-bottom: 1.2rem;
    }
    .lp-hero-title {
        font-size: 2.75rem;
        font-weight: 800;
        color: #16241c;
        line-height: 1.15;
        margin-bottom: 1rem;
        letter-spacing: -1px;
    }
    .lp-hero-sub {
        color: #536259;
        font-size: 1.05rem;
        max-width: 640px;
        margin: 0 auto 1.5rem auto;
    }
    .lp-orbits {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 1.25rem;
        margin: 1.5rem 0;
        flex-wrap: wrap;
    }
    .lp-orb {
        background: #ffffff;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 8px 20px rgba(22, 163, 74, 0.15);
        border: 1px solid #E3F2E8;
    }
    .lp-orb-lg { width: 140px; height: 140px; font-size: 3.2rem; }
    .lp-orb-md { width: 92px; height: 92px; font-size: 2.1rem; }
    .lp-orb-sm { width: 68px; height: 68px; font-size: 1.6rem; }
    .lp-orb-play {
        background: #16a34a;
        color: #ffffff;
        box-shadow: 0 8px 24px rgba(22, 163, 74, 0.35);
    }

    .lp-stat-float {
        display: inline-flex;
        align-items: center;
        gap: 0.6rem;
        background: #ffffff;
        padding: 0.6rem 1.1rem;
        border-radius: 16px;
        box-shadow: 0 6px 16px rgba(22, 163, 74, 0.12);
        border: 1px solid #E3F2E8;
        font-weight: 700;
        color: #16241c;
        font-size: 0.85rem;
    }

    .lp-darkbar {
        background: linear-gradient(135deg, #0d1f13 0%, #16281c 100%);
        border-radius: 999px;
        padding: 1rem 1.75rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 1rem;
        margin: 2rem 0 1rem 0;
    }
    .lp-darkbar-text { color: #cfe3d5; font-size: 0.88rem; margin: 0; }
    .lp-darkbar-strong { color: #4ade80; font-weight: 700; }

    .lp-section-tag {
        display: block;
        text-align: center;
        color: #16a34a;
        font-weight: 700;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 2.5rem;
    }
    .lp-section-title {
        text-align: center;
        font-size: 2rem;
        font-weight: 800;
        color: #16241c;
        margin: 0.3rem 0 0.4rem 0;
    }
    .lp-section-sub {
        text-align: center;
        color: #6b7a71;
        margin-bottom: 1.5rem;
    }

    .lp-avatar-badge {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.75rem;
        margin-bottom: 1.5rem;
    }
    .lp-avatar-badge-icon {
        width: 46px; height: 46px; border-radius: 50%;
        background: #16a34a; color: white;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.2rem; font-weight: 700;
        box-shadow: 0 4px 12px rgba(22, 163, 74, 0.25);
    }
    .lp-avatar-badge-text { color: #16241c; font-weight: 700; font-size: 0.95rem; }
    .lp-avatar-badge-sub { color: #8a978f; font-size: 0.8rem; font-weight: 400; }

    .lp-feature-card {
        border-radius: 20px;
        padding: 1.5rem;
        border: 1px solid rgba(0,0,0,0.04);
        height: 100%;
        min-height: 190px;
    }
    .lp-feature-dot { width: 10px; height: 10px; border-radius: 50%; margin-bottom: 0.9rem; }
    .lp-feature-card h4 { color: #16241c; font-size: 1.05rem; font-weight: 700; margin-bottom: 0.5rem; }
    .lp-feature-card p { color: #5c6b62; font-size: 0.88rem; margin: 0 0 0.75rem 0; }
    .lp-feature-card a { color: #16a34a; font-size: 0.85rem; font-weight: 600; text-decoration: none; }

    .lp-quality-card {
        background: #ffffff;
        border-radius: 20px;
        padding: 1.75rem;
        border: 1px solid #E3F2E8;
        box-shadow: 0 4px 20px rgba(22, 163, 74, 0.06);
    }
    .lp-quality-card li { color: #536259; margin-bottom: 0.6rem; font-size: 0.95rem; }
    .lp-check { color: #16a34a; font-weight: 700; margin-right: 0.4rem; }

    .lp-stat-row {
        display: flex;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 1rem;
        margin-top: 1.5rem;
        padding-top: 1.25rem;
        border-top: 1px solid #E3F2E8;
    }
    .lp-stat-item { text-align: center; flex: 1; min-width: 90px; }
    .lp-stat-item .lp-stat-num { font-size: 1.4rem; font-weight: 800; color: #16241c; }
    .lp-stat-item .lp-stat-label { font-size: 0.78rem; color: #8a978f; }

    .lp-pillrow {
        display: flex;
        justify-content: center;
        gap: 0.75rem;
        flex-wrap: wrap;
        margin: 1.5rem 0 2rem 0;
    }
    .lp-pillrow span {
        background: #ffffff;
        border: 1px solid #E3F2E8;
        border-radius: 999px;
        padding: 0.45rem 1rem;
        font-size: 0.85rem;
        font-weight: 600;
        color: #16241c;
        box-shadow: 0 2px 10px rgba(22, 163, 74, 0.06);
    }
    .lp-pillrow span.active { background: #16a34a; color: #ffffff; border-color: #16a34a; }

    .lp-darkfooter {
        background: linear-gradient(135deg, #0d1f13 0%, #16281c 100%);
        border-radius: 28px;
        padding: 2.5rem 2rem;
        text-align: center;
        margin-top: 2.5rem;
    }
    .lp-darkfooter h3 { color: #ffffff; font-size: 1.4rem; font-weight: 800; margin-bottom: 0.3rem; }
    .lp-darkfooter p { color: #9db8a4; font-size: 0.9rem; margin-bottom: 1.5rem; }
    .lp-tech-row {
        display: flex;
        justify-content: center;
        gap: 1rem;
        flex-wrap: wrap;
    }
    .lp-tech-pill {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.12);
        color: #d7e8db;
        padding: 0.5rem 1.1rem;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 600;
    }

    .lp-footer-note {
        text-align: center;
        color: #9aa8a0;
        font-size: 0.8rem;
        margin-top: 1.5rem;
    }

    /* Primary (green) buttons used for Try It Now CTAs */
    button[kind="primary"] {
        background: linear-gradient(135deg, #16a34a 0%, #22c55e 100%) !important;
        border: none !important;
        border-radius: 999px !important;
        font-weight: 700 !important;
        box-shadow: 0 6px 16px rgba(22, 163, 74, 0.25) !important;
    }
    button[kind="primary"]:hover {
        background: linear-gradient(135deg, #128a3e 0%, #1ea750 100%) !important;
        box-shadow: 0 8px 20px rgba(22, 163, 74, 0.35) !important;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# Landing Page <-> App routing (session state)
# -------------------------------------------------
if "view" not in st.session_state:
    st.session_state.view = "landing"

def _go_to_app():
    st.session_state.view = "app"

def _go_to_landing():
    st.session_state.view = "landing"

def render_landing():
    # Top navbar
    nav_l, nav_m, nav_r = st.columns([2, 3, 1.4])
    with nav_l:
        st.markdown('<div class="lp-navbar" style="justify-content:flex-start;"><div class="lp-logo">🫁 Deep<span>X</span></div></div>', unsafe_allow_html=True)
    with nav_m:
        st.markdown("""
            <div class="lp-navbar">
                <div class="lp-navlinks">
                    <a href="#capabilities">Capabilities</a>
                    <a href="#quality">How It Works</a>
                    <a href="#footer-note">About</a>
                </div>
            </div>
        """, unsafe_allow_html=True)
    with nav_r:
        if st.button("Try It Now", key="cta_nav", type="primary", use_container_width=True):
            _go_to_app()
            st.rerun()

    # Hero
    st.markdown("""
        <div class="lp-hero">
            <span class="lp-pill">🩺 AI-Powered Chest X-Ray Diagnostics</span>
            <div class="lp-hero-title">Clear Diagnosis for Your<br>Chest X-Rays — Instantly</div>
            <p class="lp-hero-sub">
                Upload a chest X-ray and get an AI-powered diagnosis in seconds — backed by
                GradCAM++ and Integrated Gradients explainability, so you can see exactly
                what the model is looking at.
            </p>
            <div class="lp-orbits">
                <div class="lp-orb lp-orb-sm">🧠</div>
                <div class="lp-orb lp-orb-md">🩻</div>
                <div class="lp-orb lp-orb-lg lp-orb-play">▶</div>
                <div class="lp-orb lp-orb-md">📊</div>
                <div class="lp-orb lp-orb-sm">⚡</div>
            </div>
            <div style="display:flex; justify-content:center; gap:1rem; flex-wrap:wrap; margin-bottom:0.5rem;">
                <span class="lp-stat-float">⭐ High-Accuracy CNN</span>
                <span class="lp-stat-float">🧪 3 Explainability Methods</span>
                <span class="lp-stat-float">⚡ Real-Time Results</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="lp-darkbar">
            <p class="lp-darkbar-text">🔬 Model: <span class="lp-darkbar-strong">CNN</span> trained on chest X-ray data</p>
            <p class="lp-darkbar-text">📐 Input resolution: <span class="lp-darkbar-strong">224×224</span></p>
            <p class="lp-darkbar-text">🎓 <span class="lp-darkbar-strong">Educational &amp; research use only</span></p>
        </div>
    """, unsafe_allow_html=True)

    _, cta_col, _ = st.columns([1, 1, 1])
    with cta_col:
        if st.button("🚀 Ma Diagnostic Gratuit — Try It Now", key="cta_hero", type="primary", use_container_width=True):
            _go_to_app()
            st.rerun()

    # Capabilities section (feature cards, styled like the reference "Abonnements" cards)
    st.markdown('<div id="capabilities"></div>', unsafe_allow_html=True)
    st.markdown('<div class="lp-section-title">Our Capabilities</div>', unsafe_allow_html=True)
    st.markdown('<div class="lp-section-sub">Everything the model gives you from a single upload</div>', unsafe_allow_html=True)

    st.markdown("""
        <div class="lp-avatar-badge">
            <div class="lp-avatar-badge-icon">AI</div>
            <div class="lp-avatar-badge-text">Trusted by researchers &amp; learners<br><span class="lp-avatar-badge-sub">Built with TensorFlow &amp; Streamlit</span></div>
        </div>
    """, unsafe_allow_html=True)

    cards = [
        ("#EDE9FE", "#8B5CF6", "🧠", "CNN Classifier", "A deep learning model trained to detect pneumonia-related patterns in chest X-rays."),
        ("#DCFCE7", "#16A34A", "🔥", "GradCAM++", "Visual heatmaps that highlight exactly which regions of the X-ray influenced the result."),
        ("#FFE4D6", "#F2764A", "⚡", "Integrated Gradients", "Pixel-level attribution mapping that shows which details mattered most to the model."),
        ("#DBEAFE", "#3B82F6", "⏱️", "Real-Time Results", "Get a diagnosis, confidence score, and explainability maps back in seconds."),
    ]
    fcols = st.columns(4)
    for col, (bg, dot, icon, title, desc) in zip(fcols, cards):
        with col:
            st.markdown(f"""
                <div class="lp-feature-card" style="background:{bg};">
                    <div class="lp-feature-dot" style="background:{dot};"></div>
                    <h4>{icon} {title}</h4>
                    <p>{desc}</p>
                    <a href="#quality">En savoir plus →</a>
                </div>
            """, unsafe_allow_html=True)

    # Quality / trust section
    st.markdown('<div id="quality"></div>', unsafe_allow_html=True)
    st.markdown('<span class="lp-section-tag">Qualité</span>', unsafe_allow_html=True)
    st.markdown('<div class="lp-section-title">Built for Clinical-Grade Insight</div>', unsafe_allow_html=True)
    st.markdown('<div class="lp-section-sub">Not just a prediction — a transparent one</div>', unsafe_allow_html=True)

    q_left, q_right = st.columns([3, 2])
    with q_left:
        st.markdown("""
            <div class="lp-quality-card">
                <ul style="list-style:none; padding-left: 0; margin: 0;">
                    <li><span class="lp-check">✔</span>Every prediction is paired with GradCAM++ and Integrated Gradients maps</li>
                    <li><span class="lp-check">✔</span>Confidence scores are shown alongside every diagnosis</li>
                    <li><span class="lp-check">✔</span>Model trained and evaluated on chest X-ray imaging data</li>
                    <li><span class="lp-check">✔</span>Built with TensorFlow, Keras, and Streamlit for fast iteration</li>
                </ul>
                <div class="lp-stat-row">
                    <div class="lp-stat-item"><div class="lp-stat-num">224²</div><div class="lp-stat-label">Resolution</div></div>
                    <div class="lp-stat-item"><div class="lp-stat-num">3</div><div class="lp-stat-label">XAI Methods</div></div>
                    <div class="lp-stat-item"><div class="lp-stat-num">CNN</div><div class="lp-stat-label">Architecture</div></div>
                    <div class="lp-stat-item"><div class="lp-stat-num">⚡</div><div class="lp-stat-label">Real-Time</div></div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    with q_right:
        st.markdown("""
            <div class="lp-feature-card" style="background:#0d1f13; display:flex; flex-direction:column; justify-content:center; align-items:center; text-align:center; min-height: 100%;">
                <div style="font-size:3rem; margin-bottom: 0.75rem;">🫁</div>
                <h4 style="color:#ffffff;">See It In Action</h4>
                <p style="color:#9db8a4; margin-bottom:1rem;">Upload a chest X-ray and watch the model explain its own reasoning in real time.</p>
            </div>
        """, unsafe_allow_html=True)

    # Explainability toolkit pill row
    st.markdown('<div class="lp-section-title" style="margin-top:2.5rem;">Explainability Toolkit</div>', unsafe_allow_html=True)
    st.markdown('<div class="lp-section-sub">Notre expertise au service de votre diagnostic</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="lp-pillrow">
            <span class="active">✅ Confidence Score</span>
            <span>🔥 GradCAM++</span>
            <span>⚡ Integrated Gradients</span>
            <span>🧠 CNN Model</span>
            <span>📊 224×224 Input</span>
        </div>
    """, unsafe_allow_html=True)

    _, cta_col2, _ = st.columns([1, 1, 1])
    with cta_col2:
        if st.button("🚀 Try It Now", key="cta_bottom", type="primary", use_container_width=True):
            _go_to_app()
            st.rerun()

    # Closing dark section (mirrors the reference's dark team strip)
    st.markdown("""
        <div class="lp-darkfooter">
            <h3>Powered By</h3>
            <p>The open-source stack behind DeepX's diagnostics</p>
            <div class="lp-tech-row">
                <span class="lp-tech-pill">🐍 TensorFlow / Keras</span>
                <span class="lp-tech-pill">🎈 Streamlit</span>
                <span class="lp-tech-pill">🔥 tf-keras-vis</span>
                <span class="lp-tech-pill">🧩 SHAP</span>
                <span class="lp-tech-pill">🖼️ OpenCV</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <p id="footer-note" class="lp-footer-note">
            DeepX Diagnostic System | Powered by TensorFlow &amp; Streamlit<br>
            For educational and research purposes only — always consult a qualified medical professional.
        </p>
    """, unsafe_allow_html=True)


if st.session_state.view == "landing":
    render_landing()
    st.stop()

# -------------------------------------------------
# Sidebar
# -------------------------------------------------
with st.sidebar:
    if st.button("← Back to Home", key="back_to_home", use_container_width=True):
        _go_to_landing()
        st.rerun()
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <span style="font-size: 3rem;">💉</span>
            <h2 style="color: #F5E6E0; margin-top: 0.5rem; font-weight: 700;">DeepX</h2>
            <p style="color: #C4B0A8; font-size: 0.9rem;">AI-Powered Diagnostics</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    st.markdown("""
        <p style="color: #C4B0A8; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 1rem;">
            Features
        </p>
    """, unsafe_allow_html=True)
    
    features = [
        ("🔬", "GradCAM++ Visualization"),
        ("🧾", "Integrated Gradients"),
        ("⚡", "Real-time Analysis"),
        ("🛡️", "High Accuracy CNN")
    ]
    
    for icon, text in features:
        st.markdown(f"""
            <div class="feature-item">
                <span class="feature-icon">{icon}</span>
                <span>{text}</span>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    st.markdown("""
        <div style="background: rgba(255, 181, 167, 0.12); border-left: 4px solid #FF8E72; 
                    padding: 1rem; border-radius: 0 12px 12px 0;">
            <strong style="color: #FFB5A7;">About</strong><br><br>
            <span style="color: #D9C9C3;">DeepX uses advanced deep learning to analyze chest X-rays and provide 
            interpretable AI explanations for medical professionals.</span>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div style="margin-top: 2rem; padding: 1rem; background: rgba(255, 181, 167, 0.15); 
                    border-radius: 12px; border: 1px solid rgba(255, 181, 167, 0.3);">
            <p style="color: #E8C4B8; font-size: 0.8rem; margin: 0;">
                <strong>Disclaimer:</strong> This tool is for educational purposes only. 
                Always consult a qualified medical professional.
            </p>
        </div>
    """, unsafe_allow_html=True)

# -------------------------------------------------
# Main Header
# -------------------------------------------------
st.markdown('<h1 class="main-header">💉 DeepX Diagnostic System</h1>', unsafe_allow_html=True)
st.markdown("""
  <p class="sub-header">
      Advanced AI-powered chest X-ray analysis with explainable visualizations
  </p>
""", unsafe_allow_html=True)


# -------------------------------------------------
# Load Model
# -------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_cnn_model(path):
    if not os.path.exists(path):
        st.error(f"Model file NOT FOUND: {path}")
        st.stop()
    return load_model(path)
    

model_path = "model/cnn_model_final_hdf5.h5"

with st.spinner("🔄 Loading AI Model..."):
    model = load_cnn_model(model_path)



# -------------------------------------------------
# Upload Section
# -------------------------------------------------
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

col_upload, col_info = st.columns([2, 1])

with col_upload:
    st.markdown("""
        <div class="card">
            <div class="card-title">📤 Upload X-Ray Image</div>
            <p style="color: #94a3b8; margin-bottom: 1rem;">
                Drag and drop or click to upload a chest X-ray image (PNG, JPG, JPEG)
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose a chest X-ray image",
        type=["png", "jpg", "jpeg"],
        label_visibility="collapsed"
    )

with col_info:
    st.markdown("""
        <div class="card">
            <div class="card-title">📋 Guidelines</div>
            <ul style="color: #94a3b8; padding-left: 1.25rem; margin: 0;">
                <li style="margin-bottom: 0.5rem;">Use frontal chest X-rays</li>
                <li style="margin-bottom: 0.5rem;">Ensure good image quality</li>
                <li style="margin-bottom: 0.5rem;">Supported: PNG, JPG, JPEG</li>
                <li>Max recommended: 1024x1024px</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

# -------------------------------------------------
# SAFE IMAGE PIPELINE (THIS IS WHAT FIXES YOUR CRASH)
# -------------------------------------------------
if uploaded_file is not None:
    from PIL import Image
    import numpy as np

    try:
        img = Image.open(uploaded_file).convert("RGB")
        img_resized = img.resize((224, 224))

        st.image(img_resized, width=400)

        img_array = np.array(img_resized, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

    except Exception as e:
        st.error(f"Image processing failed: {e}")
        st.stop()

# -------------------------------------------------
# Analysis Section
# -------------------------------------------------
if uploaded_file:
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    img = Image.open(uploaded_file).convert("RGB")
    img_resized = img.resize((224, 224))
    img_array = np.expand_dims(np.array(img_resized) / 255.0, axis=0)

    # -------------------------------------------------
    # Prediction with Progress
    # -------------------------------------------------
    with st.spinner("🔍 Analyzing X-ray..."):
        pred = model.predict(img_array, verbose=0)[0][0]
    
    if pred > 0.5:
        label = "PNEUMONIA"
        class_index = 1
        confidence = pred
        result_class = "result-pneumonia"
        result_icon = "⚠️"
    else:
        label = "NORMAL"
        class_index = 0
        confidence = 1 - pred
        result_class = "result-normal"
        result_icon = "✔"

    st.success(f"Prediction Confidence: {confidence*100:.2f}%")

    # Results Display
    col_result, col_image = st.columns([1, 1])
    
    with col_result:
        st.markdown(f"""
            <div class="card" style="height: 100%;">
                <div class="card-title">🧾 Diagnosis Result</div>
                <div class="{result_class}">
                    {result_icon} {label}
                </div>
                <div class="confidence-container">
                    <div class="confidence-label">Confidence Score</div>
                    <div style="display: flex; align-items: center; gap: 1rem;">
                        <div class="confidence-bar" style="flex: 1;">
                            <div class="confidence-fill" style="width: {confidence*100}%; 
                                background: {'#7BC67E' if label == 'NORMAL' else '#FF6B6B'};"></div>
                        </div>
                        <span style="color: #3D3D3D; font-weight: 700; font-size: 1.25rem;">
                            {confidence*100:.1f}%
                        </span>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with col_image:
        st.markdown("""
            <div class="card">
                <div class="card-title">📥 Uploaded X-Ray </div>
            </div>
        """, unsafe_allow_html=True)
        st.image(img_resized, width=400)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    
    # -------------------------------------------------
    # GradCAM++ & Integrated Gradients
    # -------------------------------------------------
    st.markdown('<div class="section-header">🎨 Explainability Maps</div>', unsafe_allow_html=True)

    with st.spinner("🔥 Generating GradCAM++ and Integrated Gradients..."):
        # GradCAM++
        gradcam = Gradcam(model, clone=True)
        score = BinaryScore(target_values=[class_index])
        cam = gradcam(score, img_array)[0]

        heatmap = cv2.resize(cam, (224, 224))
        heatmap_uint8 = np.uint8(255 * heatmap)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        overlay_gradcam = cv2.addWeighted(np.array(img_resized), 0.6, heatmap_color, 0.4, 0)

        # Integrated Gradients
        saliency = Saliency(model)
        saliency_map = saliency(score, img_array)[0]
        heatmap_saliency = cv2.resize(saliency_map, (224, 224))
        heatmap_saliency_uint8 = np.uint8(255 * heatmap_saliency)
        heatmap_saliency_color = cv2.applyColorMap(heatmap_saliency_uint8, cv2.COLORMAP_HOT)
        overlay_ig = cv2.addWeighted(np.array(img_resized), 0.6, heatmap_saliency_color, 0.4, 0)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("""
            <div class="image-container">
        """, unsafe_allow_html=True)
        st.image(overlay_gradcam, width=450)
        st.markdown(f"""
                <div class="image-label">GradCAM++ • {label}</div>
            </div>
        """, unsafe_allow_html=True)
        st.markdown("""
            <div class="info-box" style="margin-top: 1rem;">
                <strong>🔥 GradCAM++:</strong> Uses gradient information to highlight important regions. 
                Red/yellow areas strongly influenced the prediction.
            </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
            <div class="image-container">
        """, unsafe_allow_html=True)
        st.image(overlay_ig, width=450)
        st.markdown(f"""
                <div class="image-label">Integrated Gradients • {label}</div>
            </div>
        """, unsafe_allow_html=True)
        st.markdown("""
            <div class="info-box" style="margin-top: 1rem;">
                <strong>⚡ Integrated Gradients:</strong> Attributes predictions to input features. 
                Brighter areas indicate higher pixel importance.
            </div>
        """, unsafe_allow_html=True)

    # -------------------------------------------------
    # Summary Metrics
    # -------------------------------------------------
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">📈 Analysis Summary</div>', unsafe_allow_html=True)
    
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    
    with col_m1:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{confidence*100:.1f}%</div>
                <div class="metric-label">Confidence</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col_m2:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color: {'#7BC67E' if label == 'NORMAL' else '#FF6B6B'};">
                    {label}
                </div>
                <div class="metric-label">Diagnosis</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col_m3:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-value">3</div>
                <div class="metric-label">XAI Methods</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col_m4:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-value">224×224</div>
                <div class="metric-label">Resolution</div>
            </div>
        """, unsafe_allow_html=True)

else:
    st.markdown("""
        <div style="text-align: center; padding: 4rem 2rem; 
                    background: #ffffff; 
                    border-radius: 16px; 
                    border: 2px dashed #FFB5A7;
                    margin-top: 2rem;
                    box-shadow: 0 4px 6px -1px rgba(255, 107, 107, 0.05);">
            <span style="font-size: 4rem; opacity: 0.6;">📷</span>
            <h3 style="color: #3D3D3D; margin-top: 1rem; font-weight: 500;">
                No X-Ray Uploaded
            </h3>
            <p style="color: #8B7575; max-width: 400px; margin: 0 auto;">
                Upload a chest X-ray image to begin AI-powered diagnostic analysis
            </p>
        </div>
    """, unsafe_allow_html=True)

st.markdown("""
    <div style="text-align: center; padding: 2rem; margin-top: 3rem; 
                border-top: 1px solid #FFE0D6;">
        <p style="color: #8B7575; font-size: 0.875rem;">
            DeepX Diagnostic System | Powered by TensorFlow & Streamlit<br>
            <span style="font-size: 0.75rem; color: #C4B0A8;">For educational and research purposes only</span>
        </p>
    </div>
""", unsafe_allow_html=True)
