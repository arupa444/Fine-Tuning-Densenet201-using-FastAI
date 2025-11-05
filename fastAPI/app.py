from fastapi import FastAPI, File, UploadFile
import numpy as np
import onnxruntime as ort
from PIL import Image
import io

app = FastAPI()

# Load ONNX model
model_path = "../model/JFL_Crate_Data.onnx"  # Path to your ONNX model
session = ort.InferenceSession(model_path)

# Auto-detect model input and output
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name


def preprocess_image(image: Image.Image, size=(224, 224)) -> np.ndarray:
    """Preprocess the image for model input (you may customize this)."""
    image = image.resize(size)
    image = np.array(image).astype(np.float32) / 255.0  # Normalize between 0–1
    image = np.transpose(image, (2, 0, 1))  # HWC ➜ CHW
    image = np.expand_dims(image, axis=0)   # Add batch dimension
    return image


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Run inference with the ONNX model on an uploaded image."""
    # Read and load the uploaded image
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Preprocess
    input_data = preprocess_image(image)

    # Run inference
    outputs = session.run([output_name], {input_name: input_data})

    return {
        "model_input_name": input_name,
        "model_output_name": output_name,
        "output_shape": np.array(outputs[0]).shape,
        "output": np.array(outputs[0]).tolist()  # Convert to JSON-serializable format
    }
