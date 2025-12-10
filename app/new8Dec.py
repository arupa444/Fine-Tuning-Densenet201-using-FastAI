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
        image_np = np.array(image)
        return cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid or corrupted image file.")


def process_yolo_results(results):
    boxes_info = []
    obb = results[0].obb
    annotated = results[0].orig_img.copy()

    box_color = (0, 0, 255)

    for i in range(len(obb.conf)):
        cx, cy, w, h, angle = map(float, obb.xywhr[i])
        confidence = float(obb.conf[i])
        clsId = int(obb.cls[i])
        label = results[0].names[clsId]

        # Convert OBB to 4 corner points
        angle_deg = np.degrees(angle)
        rect = ((cx, cy), (w, h), angle_deg)
        box = cv2.boxPoints(rect)
        box = box.astype(np.int32)

        # Draw oriented bounding box
        cv2.polylines(annotated, [box], isClosed=True, color=box_color, thickness=2)

        # Compute axis-aligned bounding box from rotated box
        x1, y1 = int(box[:,0].min()), int(box[:,1].min())
        x2, y2 = int(box[:,0].max()), int(box[:,1].max())

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

    box_color = (0, 0, 255)

    for i in range(len(obb.conf)):
        cx, cy, w, h, angle = map(float, obb.xywhr[i])
        confidence = float(obb.conf[i])
        clsId = int(obb.cls[i])
        label = results[0].names[clsId]

        # Convert OBB to 4 corner points
        angle_deg = np.degrees(angle)
        rect = ((cx, cy), (w, h), angle_deg)
        box = cv2.boxPoints(rect)
        box = box.astype(np.int32)

        # Draw oriented bounding box
        cv2.polylines(annotated, [box], isClosed=True, color=box_color, thickness=2)

        # Compute axis-aligned bounding box from rotated box
        x1, y1 = int(box[:,0].min()), int(box[:,1].min())
        x2, y2 = int(box[:,0].max()), int(box[:,1].max())

        boxes_info.append({
            "class_id": clsId,
            "class_name": label,
            "confidence": round(confidence, 4),
            "cx": round(cx, 4),
            "cy": round(cy, 4),
            "w": round(w, 4),
            "h": round(h, 4),
            "angle": round(angle, 4),
            # "bbox": [x1, y1, x2, y2],
            # "obb_points": box.tolist()
        })

    return boxes_info, annotated


def draw_scaled_centered_text(annotated_image, crate_id, bbox):
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1

    font = cv2.FONT_HERSHEY_DUPLEX

    # Auto-scale text height relative to crate
    font_scale = max(0.5, min(2.2, h / 85))
    thickness = int(max(.5, int(font_scale * 2.2)))

    char_spacing = int(font_scale * 18)
    color = (0, 255, 150)  # neon green like your example

    # Compute total length
    char_widths = []
    total_width = 0
    max_h = 0

    for ch in crate_id:
        (cw, chh), _ = cv2.getTextSize(ch, font, font_scale, thickness)
        char_widths.append(cw)
        total_width += cw
        max_h = max(max_h, chh)

    if len(crate_id) > 1:
        total_width += char_spacing * (len(crate_id) - 1)

    # Position: horizontally centered, vertically centered
    start_x = x1 + (w - total_width) // 2
    text_y = y1 + (h // 2) + (max_h // 2)

    current_x = start_x

    # Draw each character
    for i, ch in enumerate(crate_id):
        cv2.putText(
            annotated_image,
            ch,
            (int(current_x), int(text_y)),
            font,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA
        )
        current_x += char_widths[i] + char_spacing


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


def preprocess_image_for_onnx_marker(image: Image.Image, size) -> np.ndarray:
    w, h = image.size
    max_dim = max(w, h)
    padded = Image.new('RGB', (max_dim, max_dim), (0, 0, 0))
    padded.paste(image, ((max_dim - w) // 2, (max_dim - h) // 2))

    resized = padded.resize(size)

    image_array = np.array(resized, dtype=np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image_array = (image_array - mean) / std

    image_array = np.transpose(image_array, (2, 0, 1))
    image_array = np.expand_dims(image_array, axis=0)

    return image_array.astype(np.float32)



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

        crate_model = get_crate_model()
        crate_results = crate_model.predict(source=image_np, conf=0.5, imgsz=640, verbose=False)
        boxes_info, annotated_image = process_yolo_results_crate_id(crate_results)

        color_counts = {"BLUE": 0, "RED": 0, "YELLOW": 0}
        color_labels = ["BLUE", "RED", "YELLOW"]


        session, input_name, output_name = get_onnx_session("model/color_classifier.onnx")

        for ind, box in enumerate(crate_results[0].obb.xyxy):
            x1, y1, x2, y2 = map(int, box)
            crop_np = image_np[y1:y2, x1:x2]
            crop = Image.fromarray(cv2.cvtColor(crop_np, cv2.COLOR_BGR2RGB))
            input_data = preprocess_image_for_onnx(crop, (224, 224))
            outputs = session.run([output_name], {input_name: input_data})
            class_idx = int(np.argmax(outputs[0]))
            boxes_info[ind]["class_id"] = class_idx
            color_counts[color_labels[class_idx]] += 1
        boxes_info.sort(key=lambda x: x['cy'])
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

        cv2.imshow("Annotated Image", annotated_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


        # upload_to_s3(img_bytes.getvalue(), image_key, "image/jpeg")
        # upload_to_s3(json.dumps(data_json).encode(), json_key, "application/json")




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
        image_np = read_image_for_yolo(image_bytes)  #input_tensor after applying the padding

        # ------------------ CRATE DETECTION ------------------
        crate_model = get_crate_model()
        crate_results = crate_model.predict(source=image_np, conf=0.5, iou=0.5, imgsz=640, verbose=False)
        _, annotated_image = process_yolo_results_crate_id(crate_results)  # crates in RED

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

        for ind, crate_box in enumerate(crate_results[0].obb.xyxy):
            x1, y1, x2, y2 = map(int, crate_box)
            crop_np = image_np[y1:y2, x1:x2]
            crate_crop = Image.fromarray(cv2.cvtColor(crop_np, cv2.COLOR_BGR2RGB))
            # apply the paading if needed

            # ---- Step 1: Color Classification ----
            input_data = preprocess_image_for_onnx(crate_crop, (224, 224))
            color_out = color_session.run([color_output], {color_input: input_data})
            color_idx = int(np.argmax(color_out[0]))
            color_label = color_labels[color_idx]
            if color_label in ["YELLOW", "RED"]:
                continue
            color_counts[color_label] += 1

            # ---- Step 2: Marker Detection ----
            marker_np = np.array(crate_crop)
            marker_results = marker_model.predict(source=marker_np, conf=0.5, verbose=False)
            marker_boxes_info, _ = process_yolo_results(marker_results)

            # ---- Step 3: Marker Classification ----
            classified_markers = []
            for i, m_box in enumerate(marker_results[0].obb.xyxy):
                mx1, my1, mx2, my2 = map(int, m_box)
                padding = 5
                mx1 = max(0, mx1 - padding)
                my1 = max(0, my1 - padding)
                mx2 = min(crate_crop.width, mx2 + padding)
                my2 = min(crate_crop.height, my2 + padding)
                marker_crop = crate_crop.crop((mx1, my1, mx2, my2))
                marker_input = preprocess_image_for_onnx_marker(marker_crop, (64, 64))
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
                    # "marker_bbox": [mx1, my1, mx2, my2],
                    "class_index": cls_idx,
                    "decoded_label": decoded_label,
                    "encoded_value": encoded_value
                })

            classified_markers.sort(key=lambda x: x['confidence'])

            classified_markers = classified_markers[:6]

            classified_markers.sort(key=lambda x: x['cx'])
            crate_id = ''.join(str(m["encoded_value"]) for m in classified_markers)
            c_confidence = float(crate_results[0].obb.conf[ind])
            c_cx, c_cy, c_w, c_h, c_angle = map(float, crate_results[0].obb.xywhr[ind])

            final_crates.append({
                "crate_bbox": {
                    "class_index": color_idx,
                    "confidence": round(c_confidence, 4),
                    "cx": round(c_cx, 4),
                    "cy": round(c_cy, 4),
                    "w": round(c_w, 4),
                    "h": round(c_h, 4),
                    "angle": round(c_angle, 4),
                    # "bbox": [x1, y1, x2, y2]  # Using "bbox" to avoid confusion with the parent key
                },
                "color": color_label,
                "crate_id": crate_id,
                "markers": classified_markers
            })
            if crate_id:
                draw_scaled_centered_text(annotated_image, crate_id, (x1, y1, x2, y2))

        final_crates.sort(key=lambda x: x['crate_bbox']['cy'])

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
        print(color_counts)


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

        cv2.imshow("Annotated Image", annotated_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # ---------- Upload to S3 ----------
        # upload_to_s3(img_bytes.getvalue(), image_key, "image/jpeg")
        # upload_to_s3(json.dumps(data_json, indent=2).encode(), json_key, "application/json")


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