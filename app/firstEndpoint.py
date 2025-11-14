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

    # Define font and visual settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_thickness = 2
    box_color = (255, 0, 0)      # 🔴 Red in BGR
    text_color = (255, 255, 255) # White

    for i in range(len(obb.conf)):
        cx, cy, w, h, angle = map(float, obb.xywhr[i])
        confidence = float(obb.conf[i])
        clsId = int(obb.cls[i])
        label = results[0].names[clsId]
        x1, y1, x2, y2 = map(int, obb.xyxy[i])

        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, thickness=4)

        # Prepare label text
        text = f"{label} ({confidence:.2f})"
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, font_thickness)

        # Draw background rectangle for readability
        cv2.rectangle(
            annotated,
            (x1, y1 - th - 8),
            (x1 + tw + 4, y1),
            box_color,
            thickness=-1
        )

        # Draw text label
        cv2.putText(
            annotated,
            text,
            (x1 + 2, y1 - 5),
            font,
            font_scale,
            text_color,
            font_thickness,
            cv2.LINE_AA
        )

        # Store box metadata
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



def process_yolo_results_crate_id(results):
    boxes_info = []
    obb = results[0].obb
    annotated = results[0].orig_img.copy()

    # Define font and visual settings
    box_color = (255, 0, 0)      # 🔴 Red in RGB... cause the microservice need's that...(matplotlib)

    for i in range(len(obb.conf)):
        cx, cy, w, h, angle = map(float, obb.xywhr[i])
        confidence = float(obb.conf[i])
        clsId = int(obb.cls[i])
        label = results[0].names[clsId]
        x1, y1, x2, y2 = map(int, obb.xyxy[i])

        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, thickness=2)


        # Store box metadata
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

        return JSONResponse(status_code=200, content={
            "status": "success",
            "data":{
                "predictions" : boxes_info,
                "image_key": image_key,
                "image_url": get_s3_url(image_key)
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

@app.post("/predict/crate_detection_classification/")
async def predict_crate_with_color(scan_type: str = Form(None), app_type: str = Form(None),
                                   type_of_load: str = Form(None), store_transfer_type: str = Form(None),
                                   android_session_id: str = Form(None), file: UploadFile = File(...)):
    try:
        # ------------------ VALIDATION ------------------
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
        if android_session_id in factorsToValidate:
            storeFieldData.append("android_session_id")

        if len(storeFieldData) == 1:
            raise HTTPException(status_code=400, detail=f"{storeFieldData[0]} field is missing")
        if len(storeFieldData) > 1:
            raise HTTPException(status_code=400, detail=f"{', '.join(storeFieldData)} fields are missing")

        if file.content_type not in SUPPORTED_IMAGE_TYPES:
            raise HTTPException(status_code=415, detail="Unsupported image type.")
        image_bytes = await file.read()
        image_np = read_image_for_yolo(image_bytes)
        pil_image = Image.fromarray(image_np)

        crate_model = get_crate_model()
        crate_results = crate_model.predict(source=image_np, conf=0.3, verbose=False)
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




        return JSONResponse(status_code=200, content={
            "status": "success",
            "data":{
                "predictions" : boxes_info,
                "image_key": image_key,
                "image_url": get_s3_url(image_key),
                "blue_count": color_counts["BLUE"],
                "yellow_count": color_counts["YELLOW"],
                "red_count": color_counts["RED"]
            },
            "message" : "Crate detection done with color classification",
            "code" : 200,
            "error" : None
        })
    except Exception as e:
        logging.error(f"Crate classification failed: {e}", exc_info=True)
        return standard_response(
            None,
            "Crate detection failed",
            400,
            str(e)
        )


@app.post("/predict/marker_detection_classification/")
async def predict_marker(
    scan_type: str = Form(None),
    app_type: str = Form(None),
    type_of_load: str = Form(None),
    store_transfer_type: str = Form(None),
    android_session_id: str = Form(None),
    file: UploadFile = File(...)
):
    try:
        # ------------------ VALIDATION ------------------
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
        if android_session_id in factorsToValidate:
            storeFieldData.append("android_session_id")

        if len(storeFieldData) == 1:
            raise HTTPException(status_code=400, detail=f"{storeFieldData[0]} field is missing")
        if len(storeFieldData) > 1:
            raise HTTPException(status_code=400, detail=f"{', '.join(storeFieldData)} fields are missing")

        if file.content_type not in SUPPORTED_IMAGE_TYPES:
            raise HTTPException(status_code=415, detail="Unsupported image type.")

        # ------------------ LOAD IMAGE ------------------
        image_bytes = await file.read()
        image_np = read_image_for_yolo(image_bytes)
        pil_image = Image.fromarray(image_np)

        # ------------------ CRATE DETECTION ------------------
        crate_model = get_crate_model()
        crate_results = crate_model.predict(source=image_np, conf=0.25, verbose=False)
        boxes_info, annotated_image = process_yolo_results_crate_id(crate_results)  # crates in RED

        # ------------------ COLOR CLASSIFICATION ------------------
        color_counts = {"BLUE": 0, "RED": 0, "YELLOW": 0}
        color_labels = ["BLUE", "RED", "YELLOW"]
        color_session, color_input, color_output = get_onnx_session("model/color_classifier.onnx")

        # ------------------ MARKER MODELS ------------------
        marker_model = get_marker_model()
        marker_cls_session, marker_cls_input, marker_cls_output = get_onnx_session(
            "model/marker_classification_efficientnet_21st_aug_2025_fp16.onnx"
        )

        # ------------------ LABEL MAP ------------------
        labelDecodeMap = {
            "filled_circle": "0",
            "filled_rectangle_vertical": "1",
            "filled_semicircle_arc_bottom": "2",
            "filled_semicircle_arc_left": "3",
            "filled_semicircle_arc_right": "4",
            "filled_semicircle_arc_top": "5",
            "filled_triangle_base_bottom": "7",
            "filled_triangle_base_left": "8",
            "filled_triangle_base_right": "9",
            "filled_triangle_base_top": "A",
            "hollow_circle": "B",
            "hollow_rectangle_vertical": "C",
            "hollow_semicircle_arc_bottom": "D",
            "hollow_semicircle_arc_left": "E",
            "hollow_semicircle_arc_right": "F",
            "hollow_semicircle_arc_top": "G",
            "hollow_triangle_base_bottom": "I",
            "hollow_triangle_base_left": "J",
            "hollow_triangle_base_right": "K",
            "hollow_triangle_base_top": "L",
            "other": "other"
        }

        # ------------------ PROCESS EACH CRATE ------------------
        final_crates = []

        for crate_box in crate_results[0].obb.xyxy:
            x1, y1, x2, y2 = map(int, crate_box)
            crate_crop = pil_image.crop((x1, y1, x2, y2))

            # ---- Step 1: Color Classification ----
            input_data = preprocess_image_for_onnx(crate_crop, (244, 244))
            color_out = color_session.run([color_output], {color_input: input_data})
            color_idx = int(np.argmax(color_out[0]))
            color_label = color_labels[color_idx]
            color_counts[color_label] += 1

            # ---- Step 2: Marker Detection ----
            marker_np = np.array(crate_crop)
            marker_results = marker_model.predict(source=marker_np, conf=0.5, verbose=False)
            marker_boxes_info, _ = process_yolo_results(marker_results)

            # ---- Step 3: Marker Classification ----
            classified_markers = []
            for i, m_box in enumerate(marker_results[0].obb.xyxy):
                mx1, my1, mx2, my2 = map(int, m_box)
                marker_crop = crate_crop.crop((mx1, my1, mx2, my2))
                marker_input = preprocess_image_for_onnx(marker_crop, (64, 64))
                cls_output = marker_cls_session.run([marker_cls_output], {marker_cls_input: marker_input})
                cls_idx = int(np.argmax(cls_output[0]))

                # Decode label
                if 0 <= cls_idx < len(labelDecodeMap.keys()):
                    decoded_label = list(labelDecodeMap.keys())[cls_idx]
                    encoded_value = labelDecodeMap[decoded_label]
                else:
                    decoded_label = "unknown"
                    encoded_value = "?"

                # Get YOLO OBB details
                m_confidence = float(marker_results[0].obb.conf[i])
                m_cx, m_cy, m_w, m_h, m_angle = map(float, marker_results[0].obb.xywhr[i])

                classified_markers.append({
                    "confidence": round(m_confidence, 4),
                    "cx": round(m_cx, 4),
                    "cy": round(m_cy, 4),
                    "w": round(m_w, 4),
                    "h": round(m_h, 4),
                    "angle": round(m_angle, 4),
                    "marker_bbox": [mx1, my1, mx2, my2],
                    "class_index": cls_idx,
                    "decoded_label": decoded_label,
                    "encoded_value": encoded_value
                })

            classified_markers.sort(key=lambda x: x['cx'])
            crate_id = ''.join(str(m["encoded_value"]) for m in classified_markers)
            final_crates.append({
                "crate_bbox": [x1, y1, x2, y2],
                "color": color_label,
                "crate_id": ''.join(str(m["encoded_value"]) for m in classified_markers),
                "markers": classified_markers
            })
            if crate_id:
                # --- Configuration for text and spacing ---
                font = cv2.FONT_HERSHEY_DUPLEX
                font_scale = 1 # font size
                font_thickness = 2 # font thickness
                text_color = (0, 255, 150)  # florocent gray
                char_spacing = 15  # <--- ADJUST SPACING HERE (in pixels)

                # --- Calculate total width and max height for the spaced-out text ---
                total_width = 0
                max_height = 0
                char_widths = []

                for char in crate_id:
                    (w, h), baseline = cv2.getTextSize(char, font, font_scale, font_thickness)
                    total_width += w
                    char_widths.append(w)
                    if h > max_height:
                        max_height = h

                # Add the total spacing between characters
                if len(crate_id) > 1:
                    total_width += char_spacing * (len(crate_id) - 1)

                # --- Calculate positioning ---
                # Center of the crate's bounding box
                center_x = x1 + (x2 - x1) // 2
                center_y = y1 + (y2 - y1) // 2

                # Starting X position to make the whole text block centered
                start_x = center_x - (total_width // 2)

                # Y position (consistent for all characters)
                # This positions the text vertically in the center
                text_y = center_y + (max_height // 2)



                # --- Draw each character individually ---
                current_x = start_x
                for i, char in enumerate(crate_id):
                    cv2.putText(
                        annotated_image,
                        char,
                        (current_x, text_y),
                        font,
                        font_scale,
                        text_color,
                        font_thickness,
                        cv2.LINE_AA
                    )
                    # Move to the next character's position
                    current_x += char_widths[i] + char_spacing


        final_crates.sort(key=lambda x: x['crate_bbox'][1])

        for crate in final_crates:
            crate.pop("crate_bbox", None)

        timestamp = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
        unique_id = str(uuid.uuid4())

        img_name = f"crate_{unique_id}_{timestamp}.jpg"
        json_name = f"crate_metadata_{unique_id}_{timestamp}.json"
        image_key = generate_s3_key(app_type, android_session_id, type_of_load, store_transfer_type, img_name)
        json_key = generate_s3_key(app_type, android_session_id, type_of_load, store_transfer_type, json_name)

        # Convert annotated image to bytes
        img = Image.fromarray(annotated_image)
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="JPEG")
        img_bytes.seek(0)
        img_size_kb = len(img_bytes.getvalue()) / 1024


        data_json = {
            "bucket": BUCKET_NAME,
            "scan_type": scan_type,
            "app_type": app_type,
            "type_of_load": type_of_load,
            "store_transfer_type": store_transfer_type,
            "android_session_id": android_session_id,
            "crates": final_crates,
            "color_counts": color_counts,
            "image_size_kb": round(img_size_kb, 2),
            "image_key": image_key,
            "json_key": json_key,
            "image_url": get_s3_url(image_key),
            "json_url": get_s3_url(json_key)
        }
        # ---------- Upload to S3 ----------
        upload_to_s3(img_bytes.getvalue(), image_key, "image/jpeg")
        upload_to_s3(json.dumps(data_json, indent=2).encode(), json_key, "application/json")


        # ------------------ RETURN RESPONSE ------------------
        return JSONResponse(status_code=200, content={
            "status": "success",
            "data": {
                "crates": final_crates,
                "image_key": image_key,
                "image_url": get_s3_url(image_key),
            },
            "message": "Crate detection, color classification, marker detection and classification completed",
            "code": 200,
            "error": None
        })

    except Exception as e:
        logging.error(f"Crate with color + marker classification failed: {e}", exc_info=True)
        return standard_response(
            None,
            "Crate detection failed",
            400,
            str(e)
        )
