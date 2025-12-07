import cv2
import base64
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO

app = Flask(__name__)
CORS(app)

# Load YOLO model
MODEL_PATH = "yolov8n.pt"
model = YOLO(MODEL_PATH)

CLASS_NAMES = [
    "ceramic-capacitor",
    "diode",
    "electrolytic-capacitor",
    "polyester-capacitor",
    "resistor",
    "transistor"
]

CLASS_MAP = {
    "ceramic-capacitor": 0,
    "diode": 1,
    "electrolytic-capacitor": 2,
    "polyester-capacitor": 3,
    "resistor": 4,
    "transistor": 5
}

@app.route("/")
def home():
    return "YOLOv8 Electronic Component Detector (Render Deployed)"


@app.route("/detect_frame", methods=["POST"])
def detect_frame():

    # -----------------------------
    # 1. Validate request  
    # -----------------------------
    if not request.is_json:
        return jsonify({"error": "Invalid request format"}), 400

    body = request.get_json(silent=True)
    if not body or "frame" not in body:
        return jsonify({"error": "No frame received"}), 400

    frame_data = body["frame"]

    try:
        encoded = frame_data.split(",")[1]
    except:
        return jsonify({"error": "Invalid base64 data"}), 400

    # -----------------------------
    # 2. Convert Base64 → Image
    # -----------------------------
    try:
        img_bytes = base64.b64decode(encoded)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"error": "Image decode failed"}), 400

    except Exception as e:
        return jsonify({"error": "Image decode error", "details": str(e)}), 400

    # -----------------------------
    # 3. YOLO Inference
    # -----------------------------
    try:
        results = model(frame, verbose=False)[0]
    except Exception as e:
        return jsonify({"error": "Model inference failed", "details": str(e)}), 500

    detected_value = -1
    box_list = []

    # -----------------------------
    # 4. Process detections safely
    # -----------------------------
    for box in results.boxes:
        cls_id = int(box.cls[0])

        if cls_id >= len(CLASS_NAMES):
            continue  # skip unknown YOLO classes

        name = CLASS_NAMES[cls_id]
        mapped_id = CLASS_MAP.get(name, -1)

        x1, y1, x2, y2 = map(float, box.xyxy[0])
        conf = float(box.conf[0])

        box_list.append({
            "class": name,
            "mapped": mapped_id,
            "conf": conf,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2
        })

        if detected_value == -1:
            detected_value = mapped_id

    # -----------------------------
    # 5. Return result safely
    # -----------------------------
    return jsonify({
        "detected": detected_value,
        "boxes": box_list
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
