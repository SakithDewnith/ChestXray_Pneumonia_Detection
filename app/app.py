import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import os

st.title("🫁 Pneumonia Detection AI")

MODEL_PATH = os.path.join(os.path.dirname(__file__), '../models/pneumonia_model.h5')

if not os.path.exists(MODEL_PATH):
    st.error("⚠️ Model not found! Please run src/train.py first.")
else:
    model = tf.keras.models.load_model(MODEL_PATH)
    
    file = st.file_uploader("Upload Chest X-Ray", type=["jpg", "png", "jpeg"])
    
    if file:
        image = Image.open(file)
        st.image(image, width=200)
        
        # Resize to 150x150 to match the model
        image = ImageOps.fit(image, (150, 150), Image.Resampling.LANCZOS)
        img_array = np.asarray(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        prediction = model.predict(img_array)
        confidence = prediction[0][0]
        
        if confidence > 0.5:
            st.error(f"PNEUMONIA Detected ({confidence:.2f})")
        else:
            st.success(f"NORMAL ({1-confidence:.2f})")
