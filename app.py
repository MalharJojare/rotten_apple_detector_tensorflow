import streamlit as st
import requests

st.title("🍎 Rotten Fruit Detector")
st.write("Upload an image of a fruit to check if it's rotten.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        files = {"file": uploaded_file.getvalue()}
        response = requests.post("http://127.0.0.1:8000/predict", files=files)

        if response.status_code == 200:
            result = response.json()
            st.success(f"Prediction: **{result['class']}** ({result['confidence']*100:.1f}% confidence)")
        else:
            st.error("Prediction failed. Check FastAPI server.")
