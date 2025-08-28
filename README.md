# 🍎 Detection of Rotten Fruits (DRF) using Image Processing & Deep Learning  

## 📌 Overview
This project demonstrates how deep learning can be applied to **fruit quality detection** by classifying fruit images into categories based on their rottenness percentage (**0%, 25%, 50%, 75%, 100% rotten**).  

We use:
- **MobileNetV2 (Transfer Learning)** for efficient feature extraction.  
- **FastAPI** as the backend REST API for model inference.  
- **Streamlit** as the interactive dashboard where users can upload fruit images and get predictions in real-time.  

---

## ⚙️ Features
- ✅ Train a CNN/transfer learning model for rotten fruit classification.  
- ✅ REST API (`/predict`) using **FastAPI** for serving predictions.  
- ✅ Interactive **Streamlit dashboard** for uploading images.  
- ✅ Supports drag-and-drop image upload and displays prediction with confidence score.  
- ✅ Easy to deploy locally or on cloud platforms (Heroku, Render, AWS, etc.).  

---

## 📂 Project Structure
├── dataset/ # Training & testing dataset (organized in subfolders)
│ ├── Train/
│ └── Test/
├── rotten_fruit_detection.py # Model training script
├── rotten_fruit_model.h5 # Saved trained model
├── main.py # FastAPI backend
├── app.py # Streamlit frontend
├── requirements.txt # Python dependencies
└── README.md # Project documentation

---

## 🚀 Getting Started

### 1️⃣ Clone the repo
```bash
git clone https://github.com/your-username/Detection-of-Rotten-Fruits-DRF.git
cd Detection-of-Rotten-Fruits-DRF
```
### 2️⃣ Create environment & install dependencies
```bash
conda create -n tf-gpu python=3.9
conda activate tf-gpu
pip install -r requirements.txt
```
### 3️⃣ Train the model (optional)
```bash
python rotten_fruit_detection.py
```
### 4️⃣ Run FastAPI server
```bash
python -m uvicorn main:app --reload
```
### 5️⃣ Run Streamlit dashboard
```bash
streamlit run app.py
