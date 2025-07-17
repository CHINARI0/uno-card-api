from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np

app = Flask(__name__)
model = YOLO("best.pt")  # Make sure best.pt is in the same folder

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "No image provided"}), 400

    img_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

    results = model.predict(source=img, conf=0.5)
    detected_classes = [model.names[int(cls)] for cls in results[0].boxes.cls]

    return jsonify({"detections": detected_classes})
