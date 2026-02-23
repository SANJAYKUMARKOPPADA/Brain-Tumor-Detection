import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="Brain Tumor Detection", layout="centered")

model = tf.keras.models.load_model("saved_model/brain_tumor_model")

st.title("🧠 Brain Tumor Detection System")
st.write("Upload an MRI image to check tumor presence")

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded MRI", use_column_width=True)

    img = img.resize((224,224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)

    if prediction[0][0] > 0.5:
        st.error("🧠 Tumor Detected")
    else:
        st.success("✅ No Tumor Detected")