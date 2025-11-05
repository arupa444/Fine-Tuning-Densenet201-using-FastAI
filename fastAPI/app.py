from fastapi import FastAPI, File, UploadFile
import numpy as np
import onnxruntime as ort
from PIL import Image
import io

app = FastAPI()

# Load ONNX model
def get_model_path(model_path):
    model_path = model_path  # Path to your ONNX model
    session = ort.InferenceSession(model_path)
    # Auto-detect model input and output
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    return input_name, output_name, session


def preprocess_image(image: Image.Image, size=(64, 64)) -> np.ndarray:
    """Preprocess the image for model input (you may customize this)."""
    image = image.resize(size)
    image = np.array(image).astype(np.float32) / 255.0  # Normalize between 0–1
    image = np.transpose(image, (2, 0, 1))  # HWC ➜ CHW
    image = np.expand_dims(image, axis=0)   # Add batch dimension
    return image


@app.post("/marker_classification_predict")
async def predict(file: UploadFile = File(...)):
    input_name, output_name, session = get_model_path("../models/marker_classification_efficientnet_21st_aug_2025_fp16.onnx")
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


@app.post("/color_classifier_predict")
async def predict(file: UploadFile = File(...)):
    input_name, output_name, session = get_model_path("../models/color_classifier.onnx")
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