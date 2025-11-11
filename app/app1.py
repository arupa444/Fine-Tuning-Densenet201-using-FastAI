from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Request
from botocore.exceptions import NoCredentialsError
from fastapi.responses import JSONResponse
from PIL import Image, ImageOps
from functools import lru_cache
from datetime import datetime
from ultralytics import YOLO
import onnxruntime as ort
import numpy as np
import logging
import base64
import boto3
import uuid
import json
import cv2
import io



# -------------------- CONFIG --------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
app = FastAPI()

def standard_response(data, message, code, error=None):
    return JSONResponse(
        status_code=code,
        content= {
            "data": data,
            "message": message,
            "code": code,
            "error": error
        }
    )

# Handle FastAPI HTTP errors (400, 401, 404, etc)
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return standard_response(
        data=None,
        message="Crate detection failed",
        code=exc.status_code,
        error=str(exc.detail)
    )

# Handle all unhandled exceptions (500)
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    return standard_response(
        data=None,
        message="Crate detection failed",
        code=500,
        error=str(exc)
    )

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
    annotated = results[0].orig_img.copy()

    for i in range(len(obb.conf)):
        cx, cy, w, h, angle = map(float, obb.xywhr[i])
        confidence = float(obb.conf[i])
        clsId = int(obb.cls[i])

        label = results[0].names[clsId]

        x1, y1, x2, y2 = map(int, obb.xyxy[i])


        color = (0, 0, 255)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness = 7)

        boxes_info.append({
            "class_id": clsId,
            "class_name": label,
            "confidence": round(confidence, 4),
            "cx": round(cx, 4),
            "cy": round(cy, 4),
            "w": round(w, 4),
            "h": round(h, 4),
            "angle": round(angle, 4),
            "bbox": [x1, y1, x2, y2]
        })

    return boxes_info, annotated


# def standard_response(data, message, code, error=None):
#     return {
#         "data": data,
#         "message": message,
#         "code": code,
#         "error":error
#     }

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


        return JSONResponse(status_code=200, content={
            "status": "success",
            "data":{
                "predictions" : boxes_info
            },
            "message" : "Crate detection done with color classification",
            "code" : 200,
            "error" : None
        })

    except Exception as e:
        logging.error(f"Crate prediction failed: {e}", exc_info=True)
        return standard_response(
            None,
            "Crate prediction failed",
            400,
            str(e)
        )

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

# @app.post("/predict/crate/")
# async def predict_crate(scan_type: str = Form(...), app_type: str = Form(...),
#                         type_of_load: str = Form(...), store_transfer_type: str = Form(...),
#                         android_session_id: str = Form(...), file: UploadFile = File(...)):
#     try:
#         storeFieldData = []
#         if scan_type == "":
#             storeFieldData.append("scan_type")
#         if app_type == "":
#             storeFieldData.append("app_type")
#         if type_of_load == "":
#             storeFieldData.append("type_of_load")
#         if store_transfer_type == "":
#             storeFieldData.append("store_transfer_type")
#         if android_session_id == "":
#             storeFieldData.append("android_session_id")

#         if len(storeFieldData) == 1:
#             raise HTTPException(status_code=400, detail=f"{storeFieldData[0]} field is missing")

#         if len(storeFieldData):
#             raise HTTPException(status_code=400, detail=f"{', '.join(storeFieldData)} fields are missing")

#         if file.content_type not in SUPPORTED_IMAGE_TYPES:
#             raise HTTPException(status_code=415, detail="Unsupported image type.")
#         image_bytes = await file.read()
#         image_np = read_image_for_yolo(image_bytes)
#         model = get_crate_model()
#         results = model.predict(source=image_np, conf=0.25, verbose=False)
#         boxes, annotated_image = process_yolo_results(results)
#         return get_annotated_image_json(scan_type, app_type, type_of_load, store_transfer_type,
#                                         android_session_id, annotated_image, boxes)
#     except Exception as e:
#         logging.error(f"Crate detection classification failed: {e}", exc_info=True)
#         return standard_response(
#             None,
#             "Crate detection failed",
#             400,
#             str(e)
#         )


@app.post("/predict/marker/")
async def predict_marker(scan_type: str = Form(None), app_type: str = Form(None),
                         type_of_load: str = Form(None), store_transfer_type: str = Form(None),
                         android_session_id: str = Form(None), file: UploadFile = File(...)):
    try:
        storeFieldData = []
        if scan_type == "":
            storeFieldData.append("scan_type")
        if app_type == "":
            storeFieldData.append("app_type")
        if type_of_load == "":
            storeFieldData.append("type_of_load")
        if store_transfer_type == "":
            storeFieldData.append("store_transfer_type")
        if android_session_id == "":
            storeFieldData.append("android_session_id")

        if len(storeFieldData) == 1:
            raise HTTPException(status_code=400, detail=f"{storeFieldData[0]} field is missing")

        if len(storeFieldData):
            raise HTTPException(status_code=400, detail=f"{', '.join(storeFieldData)} fields are missing")


        if file.content_type not in SUPPORTED_IMAGE_TYPES:
            raise HTTPException(status_code=415, detail="Unsupported image type.")
        image_bytes = await file.read()
        image_np = read_image_for_yolo(image_bytes)
        model = get_marker_model()
        results = model.predict(source=image_np, conf=0.25, verbose=False)
        boxes, annotated_image = process_yolo_results(results)
        return get_annotated_image_json(scan_type, app_type, type_of_load, store_transfer_type,
                                        android_session_id, annotated_image, boxes)
    except Exception as e:
        logging.error(f"Color classification failed: {e}", exc_info=True)
        return standard_response(
            None,
            "Crate detection failed",
            400,
            str(e)
        )


@app.post("/marker_classification_predict")
async def predict_marker_classification(
        scan_type: str = Form(...),
        app_type: str = Form(...),
        type_of_load: str = Form(...),
        store_transfer_type: str = Form(...),
        android_session_id: str = Form(...),
        file: UploadFile = File(...)
):
    """Run inference with the marker classification ONNX model."""
    try:
        storeFieldData = []
        if scan_type == "":
            storeFieldData.append("scan_type")
        if app_type == "":
            storeFieldData.append("app_type")
        if type_of_load == "":
            storeFieldData.append("type_of_load")
        if store_transfer_type == "":
            storeFieldData.append("store_transfer_type")
        if android_session_id == "":
            storeFieldData.append("android_session_id")

        if len(storeFieldData) == 1:
            raise HTTPException(status_code=400, detail=f"{storeFieldData[0]} field is missing")

        if len(storeFieldData):
            raise HTTPException(status_code=400, detail=f"{', '.join(storeFieldData)} fields are missing")

        if file.content_type not in SUPPORTED_IMAGE_TYPES:
            raise HTTPException(status_code=415, detail="Unsupported image type.")

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


# @app.post("/color_classifier_predict")
# async def predict_color_classification(
#         scan_type: str = Form(...),
#         app_type: str = Form(...),
#         type_of_load: str = Form(...),
#         store_transfer_type: str = Form(...),
#         android_session_id: str = Form(...),
#         file: UploadFile = File(...)
# ):
#     """Run inference with the color classification ONNX model."""
#     try:
#         storeFieldData = []
#         if scan_type == "":
#             storeFieldData.append("scan_type")
#         if app_type == "":
#             storeFieldData.append("app_type")
#         if type_of_load == "":
#             storeFieldData.append("type_of_load")
#         if store_transfer_type == "":
#             storeFieldData.append("store_transfer_type")
#         if android_session_id == "":
#             storeFieldData.append("android_session_id")

#         if len(storeFieldData) == 1:
#             raise HTTPException(status_code=400, detail=f"{storeFieldData[0]} field is missing")

#         if len(storeFieldData):
#             raise HTTPException(status_code=400, detail=f"{', '.join(storeFieldData)} fields are missing")

#         if file.content_type not in SUPPORTED_IMAGE_TYPES:
#             raise HTTPException(status_code=415, detail="Unsupported image type.")

#         session, input_name, output_name = get_onnx_session("model/color_classifier.onnx")

#         image_bytes = await file.read()
#         image = Image.open(io.BytesIO(image_bytes))
#         # Apply EXIF orientation
#         image = ImageOps.exif_transpose(image)
#         image = image.convert("RGB")
#         input_data = preprocess_image_for_onnx(image, (244, 244))


#         outputs = session.run([output_name], {input_name: input_data})

#         return JSONResponse(content={
#             "scan_type": scan_type,
#             "app_type": app_type,
#             "type_of_load": type_of_load,
#             "store_transfer_type": store_transfer_type,
#             "android_session_id": android_session_id,
#             "model_input_name": input_name,
#             "model_output_name": output_name,
#             "output_shape": np.array(outputs[0]).shape,
#             "output": np.array(outputs[0]).tolist()
#         })
#     except Exception as e:
#         logging.error(f"color classifier failed: {e}", exc_info=True)
#         raise HTTPException(status_code=500, detail="Failed to perform color classifier")



@app.post("/predict/crate_with_color/")
async def predict_crate_with_color(scan_type: str = Form(None), app_type: str = Form(None),
                                   type_of_load: str = Form(None), store_transfer_type: str = Form(None),
                                   android_session_id: str = Form(None), file: UploadFile = File(...)):
    try:
        storeFieldData = []
        factorsToValidate = ['', None, "nil", 'none']
        if scan_type in factorsToValidate:
            storeFieldData.append("scan_type")
        if app_type in factorsToValidate:
            storeFieldData.append("app_type")
        if type_of_load in factorsToValidate:
            storeFieldData.append("type_of_load")
        if store_transfer_type in factorsToValidate:
            storeFieldData.append("store_transfer_type")
        if android_session_id  in factorsToValidate:
            storeFieldData.append("android_session_id")

        if len(storeFieldData) == 1:
            raise HTTPException(status_code=400, detail=f"{storeFieldData[0]} field is missing")

        if len(storeFieldData):
            raise HTTPException(status_code=400, detail=f"{', '.join(storeFieldData)} fields are missing {scan_type, app_type}")

        # if len(storeFieldData) == 1:
        #     raise HTTPException(status_code=400, detail=f"{storeFieldData[0]} field is missing")

        # if len(storeFieldData) > 1:
        #     raise HTTPException(status_code=400, detail=f"{', '.join(storeFieldData)} fields are missing")

        if file.content_type not in SUPPORTED_IMAGE_TYPES:
            raise HTTPException(status_code=415, detail="Unsupported image type.")


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




        return JSONResponse(status_code=200, content={
            "status": "success",
            "data":{
                "predictions" : data_json,
                "blue_count": color_counts["BLUE"],
                "yellow_count": color_counts["YELLOW"],
                "red_count": color_counts["RED"]
            },
            "message" : "Crate detection done with color classification",
            "code" : 200,
            "error" : None
        })
    except Exception as e:
        logging.error(f"Crate with color classification failed: {e}", exc_info=True)
        return standard_response(
            None,
            "Crate detection failed",
            400,
            str(e)
        )