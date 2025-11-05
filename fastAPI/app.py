from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from ultralytics import YOLO
from PIL import Image
import numpy as np
import io
import onnxruntime as ort

app = FastAPI()


# Load YOLO TFLite models for object detection
crate_model = YOLO(
    r"../model/jfl_crate_model_obb_01052025_with_nms_conf_0.5_yolo_v11.tflite",
    task="obb"
)

marker_model = YOLO(
    r"../model/jfl_marker_vertical_rotate_28_july_2025_float32_nms_false_yolo_v11.tflite",
    task="obb"
)

SUPPORTED_IMAGE_TYPES = [
    "image/jpeg", "image/png", "image/bmp", "image/tiff", "image/webp"
]

# --- Helper Functions for YOLO Models ---

def read_image_for_yolo(file_bytes: bytes) -> np.ndarray:
    """Load an image from bytes for YOLO prediction."""
    try:
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        return np.array(image)
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is not a valid image or file is corrupted.",
        )

def process_yolo_results(results):
    """Extract predicted oriented bounding boxes and render the annotated output image."""
    boxes_info = []
    obb = results[0].obb  # Oriented bounding boxes result handler

    for i in range(len(obb.conf)):
        cls = int(obb.cls[i])  # class index
        conf = float(obb.conf[i])  # confidence score
        boxes_info.append({
            "class_id": cls,
            "confidence": conf
        })

    annotated = results[0].plot()  # Visualized result in numpy format
    return boxes_info, annotated


def serve_annotated_image(np_image, headers):
    """Convert numpy image array into an HTTP image response."""
    img = Image.fromarray(np_image)
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG")
    img_bytes.seek(0)
    return StreamingResponse(img_bytes, media_type="image/png", headers=headers)


# --- Helper Functions for ONNX Models ---

def get_onnx_session(model_path):
    """Load an ONNX model and return the session and input/output names."""
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    return session, input_name, output_name

def preprocess_image_for_onnx(image: Image.Image, size) -> np.ndarray:
    """Preprocess the image for ONNX model input."""
    image = image.resize(size)
    image = np.array(image).astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))  # HWC -> CHW
    image = np.expand_dims(image, axis=0)
    return image

# --- YOLO Object Detection Endpoints ---

@app.post("/predict/crate/")
async def predict_crate(file: UploadFile = File(...)):
    if file.content_type not in SUPPORTED_IMAGE_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported image type: {file.content_type}. Supported: {SUPPORTED_IMAGE_TYPES}",
        )

    image_bytes = await file.read()
    image_np = read_image_for_yolo(image_bytes)

    results = crate_model.predict(source=image_np, conf=0.25, verbose=False)
    boxes, annotated_image = process_yolo_results(results)

    return serve_annotated_image(annotated_image, headers={"X-Predictions": str(boxes)})


@app.post("/predict/marker/")
async def predict_marker(file: UploadFile = File(...)):
    if file.content_type not in SUPPORTED_IMAGE_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported image type: {file.content_type}. Supported: {SUPPORTED_IMAGE_TYPES}",
        )

    image_bytes = await file.read()
    image_np = read_image_for_yolo(image_bytes)

    results = marker_model.predict(source=image_np, conf=0.25, verbose=False)
    boxes, annotated_image = process_yolo_results(results)

    return serve_annotated_image(annotated_image, headers={"X-Predictions": str(boxes)})


# --- ONNX Classification Endpoints ---

@app.post("/marker_classification_predict")
async def predict_marker_classification(file: UploadFile = File(...)):
    """Run inference with the marker classification ONNX model."""
    session, input_name, output_name = get_onnx_session("../model/marker_classification_efficientnet_21st_aug_2025_fp16.onnx")

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    input_data = preprocess_image_for_onnx(image, (64, 64))

    outputs = session.run([output_name], {input_name: input_data})

    return {
        "model_input_name": input_name,
        "model_output_name": output_name,
        "output_shape": np.array(outputs[0]).shape,
        "output": np.array(outputs[0]).tolist()
    }


@app.post("/color_classifier_predict")
async def predict_color_classification(file: UploadFile = File(...)):
    """Run inference with the color classification ONNX model."""
    session, input_name, output_name = get_onnx_session("../model/color_classifier.onnx")

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    input_data = preprocess_image_for_onnx(image, (244, 244))

    outputs = session.run([output_name], {input_name: input_data})

    return {
        "model_input_name": input_name,
        "model_output_name": output_name,
        "output_shape": np.array(outputs[0]).shape,
        "output": np.array(outputs[0]).tolist()
    }