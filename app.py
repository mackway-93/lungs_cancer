import os
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# ✅ Absolute path to the model
model_path = r"C:\Users\HP\OneDrive\Desktop\lungs_cancer\lung_cancer_model.h5"

# Check if model exists
if not os.path.exists(model_path):
    st.error(f"Model file not found at: {model_path}")
    st.stop()

# ✅ Load the model
model = load_model(model_path)

st.title("Lung Cancer Prediction From X-ray")

uploaded_file = st.file_uploader("Upload a chest X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).resize((128, 128))
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Predict
    prediction = model.predict(img_array)[0][0]

    if prediction > 0.5:
        st.success("Lung Cancer Detected 😷")
    else:
        st.success("No Lung Cancer Detected 😊")