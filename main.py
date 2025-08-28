from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import uvicorn
import numpy as np
import tensorflow as tf
from PIL import Image

app = FastAPI()

# Load model once at startup
model = tf.keras.models.load_model("rotten_fruit_model.h5")
class_names = ["0% rotten", "25% rotten", "50% rotten", "75% rotten", "100% rotten"]

def preprocess_image(image: Image.Image):
    image = image.resize((224, 224))  # match training size
    img_array = np.array(image) / 255.0
    return np.expand_dims(img_array, axis=0)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img = Image.open(file.file).convert("RGB")
    processed = preprocess_image(img)
    preds = model.predict(processed)
    class_idx = np.argmax(preds[0])
    confidence = float(np.max(preds[0]))
    return JSONResponse({
        "class": class_names[class_idx],
        "confidence": round(confidence, 3)
    })

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
