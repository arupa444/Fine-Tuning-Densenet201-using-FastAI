import boto3
from botocore.exceptions import NoCredentialsError
from datetime import datetime
import uuid
import json
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
from functools import lru_cache
from ultralytics import YOLO
from PIL import Image, ImageOps
import onnxruntime as ort
import numpy as np
import io
import base64
import logging

# -------------------- CONFIG --------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
app = FastAPI()

BUCKET_NAME = "tracksure-jfl-dev"
s3 = boto3.client("s3")

SUPPORTED_IMAGE_TYPES = [
    "image/jpeg", "image/png", "image/bmp", "image/tiff", "image/webp"
]

# -------------------- HELPERS --------------------

def upload_to_s3(file_bytes, key, content_type="image/jpeg"):
    try:
        s3.put_object(Bucket=BUCKET_NAME, Key=key, Body=file_bytes, ContentType=content_type)
        return key
    except NoCredentialsError:
        raise HTTPException(status_code=500, detail="AWS credentials not found.")

def generate_s3_key(app_type: str, session_id: str, type_of_load: str, store_transfer_type: str, image_name: str) -> str:
    """Mimics Android logic for S3 key generation."""
    date = datetime.now().strftime("%d-%m-%Y")
    if app_type in ["DriverApp", "SCCApp"]:
        return f"{app_type}/{date}/{session_id}/{type_of_load}/{image_name}"
    else:
        if type_of_load == "store_crate_inventory":
            return f"{app_type}/{date}/{session_id}/{store_transfer_type}/{image_name}"
        else:
            return f"{app_type}/{date}/{session_id}/{store_transfer_type}/{type_of_load}/{image_name}"

def get_s3_url(key: str) -> str:
    return f"https://{BUCKET_NAME}.s3.amazonaws.com/{key}"

def read_image_for_yolo(file_bytes: bytes) -> np.ndarray:
    try:
        image = Image.open(io.BytesIO(file_bytes))
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")
        return np.array(image)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid or corrupted image file.")

def process_yolo_results(results):
    boxes_info = []
    obb = results[0].obb
    for i in range(len(obb.conf)):
        cls = int(obb.cls[i])
        conf = float(obb.conf[i])
        boxes_info.append({"class_id": cls, "confidence": conf})
    annotated = results[0].plot()
    return boxes_info, annotated

def get_annotated_image_json(scan_type, app_type, type_of_load, store_transfer_type,
                             android_session_id, np_image, boxes_info):
    """Convert numpy annotated image -> upload to S3 -> return response."""
    try:
        img = Image.fromarray(np_image)
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="JPEG")
        img_bytes.seek(0)

        # Prepare metadata JSON
        result_json = {
            "scan_type": scan_type,
            "app_type": app_type,
            "type_of_load": type_of_load,
            "store_transfer_type": store_transfer_type,
            "android_session_id": android_session_id,
            "predictions": boxes_info
        }

        # Generate unique keys for S3
        img_name = f"{uuid.uuid4()}.jpg"
        json_name = f"{uuid.uuid4()}.json"
        image_key = generate_s3_key(app_type, android_session_id, type_of_load, store_transfer_type, img_name)
        json_key = generate_s3_key(app_type, android_session_id, type_of_load, store_transfer_type, json_name)

        # Upload to S3
        upload_to_s3(img_bytes.getvalue(), image_key, content_type="image/jpeg")
        upload_to_s3(json.dumps(result_json).encode(), json_key, content_type="application/json")

        return JSONResponse(content={
            "status": "success",
            "image_url": get_s3_url(image_key),
            "json_url": get_s3_url(json_key),
            "predictions": boxes_info
        })

    except Exception as e:
        logging.error(f"S3 upload failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to generate annotated image JSON")

# -------------------- YOLO MODEL LOADING --------------------
@lru_cache(maxsize=1)
def get_crate_model():
    return YOLO("model/jfl_crate_model_obb_01052025_with_nms_conf_0.5_yolo_v11.tflite", task="obb")

@lru_cache(maxsize=1)
def get_marker_model():
    return YOLO("model/jfl_marker_vertical_rotate_28_july_2025_float32_nms_false_yolo_v11.tflite", task="obb")

# -------------------- ONNX MODEL --------------------
@lru_cache(maxsize=1)
def get_onnx_session(model_path):
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    return session, input_name, output_name

def preprocess_image_for_onnx(image: Image.Image, size) -> np.ndarray:
    image = image.resize(size)
    image = np.array(image).astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)
    return image

# -------------------- ENDPOINTS --------------------

@app.post("/predict/crate/")
async def predict_crate(scan_type: str = Form(None), app_type: str = Form(None),
                        type_of_load: str = Form(None), store_transfer_type: str = Form(None),
                        android_session_id: str = Form(None), file: UploadFile = File(...)):
    if file.content_type not in SUPPORTED_IMAGE_TYPES:
        raise HTTPException(status_code=415, detail="Unsupported image type.")
    image_bytes = await file.read()
    image_np = read_image_for_yolo(image_bytes)
    model = get_crate_model()
    results = model.predict(source=image_np, conf=0.25, verbose=False)
    boxes, annotated_image = process_yolo_results(results)
    return get_annotated_image_json(scan_type, app_type, type_of_load, store_transfer_type,
                                    android_session_id, annotated_image, boxes)


@app.post("/predict/marker/")
async def predict_marker(scan_type: str = Form(None), app_type: str = Form(None),
                        type_of_load: str = Form(None), store_transfer_type: str = Form(None),
                        android_session_id: str = Form(None), file: UploadFile = File(...)):
    if file.content_type not in SUPPORTED_IMAGE_TYPES:
        raise HTTPException(status_code=415, detail="Unsupported image type.")
    image_bytes = await file.read()
    image_np = read_image_for_yolo(image_bytes)
    model = get_marker_model()
    results = model.predict(source=image_np, conf=0.25, verbose=False)
    boxes, annotated_image = process_yolo_results(results)
    return get_annotated_image_json(scan_type, app_type, type_of_load, store_transfer_type,
                                    android_session_id, annotated_image, boxes)



@app.post("/marker_classification_predict")
async def predict_marker_classification(
        scan_type: str = Form(None),
        app_type: str = Form(None),
        type_of_load: str = Form(None),
        store_transfer_type: str = Form(None),
        android_session_id: str = Form(None),
        file: UploadFile = File(...)
):
    """Run inference with the marker classification ONNX model."""
    try:
        session, input_name, output_name = get_onnx_session("model/marker_classification_efficientnet_21st_aug_2025_fp16.onnx")

        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        # Apply EXIF orientation
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")
        input_data = preprocess_image_for_onnx(image, (64, 64))

        outputs = session.run([output_name], {input_name: input_data})

        return JSONResponse(content={
            "scan_type": scan_type,
            "app_type": app_type,
            "type_of_load": type_of_load,
            "store_transfer_type": store_transfer_type,
            "android_session_id": android_session_id,
            "model_input_name": input_name,
            "model_output_name": output_name,
            "output_shape": np.array(outputs[0]).shape,
            "output": np.array(outputs[0]).tolist()
        })
    except Exception as e:
        logging.error(f"marker classification failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to perform marker classification")


@app.post("/color_classifier_predict")
async def predict_color_classification(
        scan_type: str = Form(None),
        app_type: str = Form(None),
        type_of_load: str = Form(None),
        store_transfer_type: str = Form(None),
        android_session_id: str = Form(None),
        file: UploadFile = File(...)
):
    """Run inference with the color classification ONNX model."""
    try:
        session, input_name, output_name = get_onnx_session("model/color_classifier.onnx")

        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        # Apply EXIF orientation
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")
        input_data = preprocess_image_for_onnx(image, (244, 244))


        outputs = session.run([output_name], {input_name: input_data})

        return JSONResponse(content={
            "scan_type": scan_type,
            "app_type": app_type,
            "type_of_load": type_of_load,
            "store_transfer_type": store_transfer_type,
            "android_session_id": android_session_id,
            "model_input_name": input_name,
            "model_output_name": output_name,
            "output_shape": np.array(outputs[0]).shape,
            "output": np.array(outputs[0]).tolist()
        })
    except Exception as e:
        logging.error(f"color classifier failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to perform color classifier")



@app.post("/predict/crate_with_color/")
async def predict_crate_with_color(scan_type: str = Form(None), app_type: str = Form(None),
                                   type_of_load: str = Form(None), store_transfer_type: str = Form(None),
                                   android_session_id: str = Form(None), file: UploadFile = File(...)):
    try:
        if file.content_type not in SUPPORTED_IMAGE_TYPES:
            raise HTTPException(status_code=415, detail="Unsupported image type.")
        image_bytes = await file.read()
        image_np = read_image_for_yolo(image_bytes)
        pil_image = Image.fromarray(image_np)

        crate_model = get_crate_model()
        crate_results = crate_model.predict(source=image_np, conf=0.25, verbose=False)
        boxes_info, annotated_image = process_yolo_results(crate_results)

        color_counts = {"BLUE": 0, "RED": 0, "YELLOW": 0}
        color_labels = ["BLUE", "RED", "YELLOW"]

        session, input_name, output_name = get_onnx_session("model/color_classifier.onnx")

        for box in crate_results[0].obb.xyxy:
            x1, y1, x2, y2 = map(int, box)
            crop = pil_image.crop((x1, y1, x2, y2))
            input_data = preprocess_image_for_onnx(crop, (244, 244))
            outputs = session.run([output_name], {input_name: input_data})
            class_idx = int(np.argmax(outputs[0]))
            color_counts[color_labels[class_idx]] += 1

        # Prepare annotated image and JSON for S3 upload
        img = Image.fromarray(annotated_image)
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="JPEG")
        img_bytes.seek(0)

        data_json = {
            "scan_type": scan_type,
            "app_type": app_type,
            "type_of_load": type_of_load,
            "store_transfer_type": store_transfer_type,
            "android_session_id": android_session_id,
            "predictions": boxes_info,
            "color_counts": color_counts
        }

        img_name = f"{uuid.uuid4()}.jpg"
        json_name = f"{uuid.uuid4()}.json"
        image_key = generate_s3_key(app_type, android_session_id, type_of_load, store_transfer_type, img_name)
        json_key = generate_s3_key(app_type, android_session_id, type_of_load, store_transfer_type, json_name)

        upload_to_s3(img_bytes.getvalue(), image_key, "image/jpeg")
        upload_to_s3(json.dumps(data_json).encode(), json_key, "application/json")

        return JSONResponse(content={
            "status": "success",
            "image_url": get_s3_url(image_key),
            "json_url": get_s3_url(json_key),
            "blue_count": color_counts["BLUE"],
            "red_count": color_counts["RED"],
            "yellow_count": color_counts["YELLOW"],
        })
    except Exception as e:
        logging.error(f"Crate with color classification failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Crate detection failed")

