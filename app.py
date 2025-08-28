import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model once
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("rotten_fruit_model.h5")

model = load_model()
class_names = ["0% rotten", "25% rotten", "50% rotten", "75% rotten", "100% rotten"]

def preprocess_image(image: Image.Image):
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    return np.expand_dims(img_array, axis=0)

st.title("🍎 Rotten Fruit Detector")
uploaded_file = st.file_uploader("Upload a fruit image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    img = Image.open(uploaded_file).convert("RGB")
    processed = preprocess_image(img)

    if st.button("Predict"):
        preds = model.predict(processed)
        class_idx = np.argmax(preds[0])
        confidence = float(np.max(preds[0]))
        st.success(f"Prediction: **{class_names[class_idx]}** ({confidence*100:.1f}% confidence)")
