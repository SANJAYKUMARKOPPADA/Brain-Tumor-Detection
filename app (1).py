import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page config
st.set_page_config(
    page_title="Brain Tumor Detection",
    layout="centered"
)

# Load model
model = tf.keras.models.load_model("saved_model/brain_tumor_model")

# Title
st.markdown("<h2 style='text-align: center;'>🧠 Brain Tumor Detection</h2>", unsafe_allow_html=True)
st.write("Upload a brain MRI image to detect tumor presence.")

# File uploader
uploaded_file = st.file_uploader(
    "Choose an MRI image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Show image
    image_pil = Image.open(uploaded_file).convert("RGB")
    st.image(image_pil, caption="Uploaded MRI Image", use_column_width=True)

    # Preprocess image
    image_resized = image_pil.resize((224, 224))
    image_array = np.array(image_resized) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Prediction
    prediction = model.predict(image_array)

    st.markdown("---")

    if prediction[0][0] > 0.5:
        st.error("🧠 **Tumor Detected**")
    else:
        st.success("✅ **No Tumor Detected**")

# Footer
st.markdown(
    "<p style='text-align: center; font-size: 12px;'>Deep Learning Project | TensorFlow + Streamlit</p>",
    unsafe_allow_html=True
)
