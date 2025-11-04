from fastapi import FastAPI, File, UploadFile
import numpy as np
import onnxruntime as ort
from PIL import Image
import io

app = FastAPI()

THRESHOLD = 0.88  # 88%

# Load ONNX model
session = ort.InferenceSession("../JFL_Crate_Data.onnx")
input_name = session.get_inputs()[0].name

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read and preprocess image
    image = Image.open(io.BytesIO(await file.read()))
    image = image.resize((224, 224))  # depends on your model input
    input_data = np.array(image).astype(np.float32) / 255.0
    input_data = np.expand_dims(input_data.transpose(2, 0, 1), axis=0)

    # Inference
    pred = session.run(None, {input_name: input_data})[0][0]
    prob_crate = float(pred[0])

    # Apply threshold
    if prob_crate >= THRESHOLD:
        return {"prediction": "Crate", "confidence": prob_crate}
    else:
        return {"prediction": "Creat_not_Detected", "confidence": prob_crate}
