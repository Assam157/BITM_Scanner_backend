import cv2
import base64
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO

# ============================
# Flask Setup
# ============================
app = Flask(__name__)
CORS(app)

# ============================
# Load your YOLOv8 model from GitHub repo
# Render will read best.pt directly.
# ============================
MODEL_PATH = "yolov8n.pt"   # file in your repo
model = YOLO(MODEL_PATH)

# ============================
# Component class names (same as training)
# ============================
CLASS_NAMES = [
    "ceramic-capacitor",
    "diode",
    "electrolytic-capacitor",
    "polyester-capacitor",
    "resistor",
    "transistor"
]

# ============================
# Mapping to ID
# ============================
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
    data = request.json["frame"]

    encoded = data.split(",")[1]
    img_bytes = base64.b64decode(encoded)

    np_arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Run YOLOv8 inference
    results = model(frame, verbose=False)[0]

    detected_value = -1
    box_list = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        name = CLASS_NAMES[cls_id]

        x1, y1, x2, y2 = map(float, box.xyxy[0])
        mapped_id = CLASS_MAP.get(name, -1)

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

    return jsonify({
        "detected": detected_value,
        "boxes": box_list
    })


# Gunicorn will run this
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
