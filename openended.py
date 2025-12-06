import cv2
import torch
import base64
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

from ultralytics import YOLO  # YOLOv8 import

# Flask
app = Flask(__name__)
CORS(app)

# Load YOLOv8 model from local path
MODEL_PATH = "yolov8n.pt"
model = YOLO(MODEL_PATH)

# Map YOLO classes → your categories
CLASS_MAP = {
    "cell phone": 1,  # Resistor
    "person": 0,      # Capacitor
    "mouse": 2        # Transducer
}

@app.route("/")
def home():
    return "Scanner YOLO backend (YOLOv8)"

@app.route("/detect_frame", methods=["POST"])
def detect_frame():
    data = request.json["frame"]

    # Remove data:image/png;base64,
    encoded = data.split(",")[1]

    # Decode base64 → image
    img_bytes = base64.b64decode(encoded)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # YOLOv8 detect
    results = model(frame)[0]  # YOLO outputs a list → take index 0

    detected_value = -1
    box_list = []

    # Loop detections
    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        name = model.names[cls_id]
        x1, y1, x2, y2 = map(float, box.xyxy[0])

        box_list.append({
            "class": name,
            "mapped": CLASS_MAP.get(name, -1),
            "conf": conf,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2
        })

        if name in CLASS_MAP and detected_value == -1:
            detected_value = CLASS_MAP[name]

    return jsonify({
        "detected": detected_value,
        "boxes": box_list
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)

