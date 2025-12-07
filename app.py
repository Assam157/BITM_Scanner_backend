import os
import cv2
import base64
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO

# =========================
# Flask setup
# =========================
app = Flask(__name__)
CORS(app)

# =========================
# Load YOLO model
# =========================
MODEL_PATH = "best.pt"
model = YOLO(MODEL_PATH)
print("YOLO model loaded successfully.")

# Define class names and mapping
CLASS_NAMES = [
    "ceramic-capacitor",
    "diode",
    "electrolytic-capacitor",
    "polyester-capacitor",
    "resistor",
    "transistor"
]
CLASS_MAP = {name: i for i, name in enumerate(CLASS_NAMES)}

# =========================
# Flask endpoints
# =========================
@app.route("/")
def home():
    return "YOLOv8 Electronic Component Detector API is running."

@app.route("/detect_frame", methods=["POST"])
def detect_frame():
    if not request.is_json:
        return jsonify({"error": "Invalid request"}), 400

    body = request.get_json()
    if "frame" not in body:
        return jsonify({"error": "No frame received"}), 400

    frame_data = body["frame"]
    try:
        # Decode base64 image
        encoded = frame_data.split(",")[1] if "," in frame_data else frame_data
        img_bytes = base64.b64decode(encoded)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error": "Image decode failed"}), 400
    except Exception as e:
        return jsonify({"error": "Base64 decode error", "details": str(e)}), 400

    # YOLO Inference
    try:
        results = model(frame, verbose=False)[0]
    except Exception as e:
        return jsonify({"error": "Model inference failed", "details": str(e)}), 500

    detected_value = -1
    box_list = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id >= len(CLASS_NAMES):
            continue
        name = CLASS_NAMES[cls_id]
        mapped_id = CLASS_MAP[name]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
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

    return jsonify({"detected": detected_value, "boxes": box_list})

# =========================
# Run Flask app
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, use_reloader=False)
