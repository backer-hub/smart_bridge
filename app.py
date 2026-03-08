"""
PlantCare AI - Flask Web Application
=====================================
Serves the plant disease detection model via a REST API and web UI.

Usage:
    python app.py

Endpoints:
    GET  /           → Web interface
    POST /predict    → Image upload + inference (JSON response)
    GET  /health     → Health check
"""

import os
import json
import uuid
import numpy as np
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename

# ─── Optional TensorFlow import (graceful demo fallback) ──────────────────────
try:
    import tensorflow as tf
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("⚠️  TensorFlow not installed — running in DEMO mode")

from PIL import Image

# ─── App Setup ────────────────────────────────────────────────────────────────

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB limit
app.config["UPLOAD_FOLDER"] = "static/uploads"

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
IMG_SIZE = 224
MODEL_DIR = Path("model")

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# ─── Model Loading ────────────────────────────────────────────────────────────

model = None
class_names = []

# Full PlantVillage class names (38 classes)
DEMO_CLASSES = [
"Pepperbell_Bacterial spot",
"Pepperbell_healthy",
"Potato_Early blight",
"Potato_Late blight",
"Potato_healthy",
"Tomato_Bacterial spot",
"Tomato_Early blight",
"Tomato_Late blight",
"Tomato_Leaf Mold",
"Tomato_Septoria leaf spot",
"Tomato_Spider mites Two spotted spider mite",
"Tomato_Target Spot",
"Tomato_ Tomato YellowLeaf Curl Virus",
"Tomato_Tomato mosaic virus",
"Tomato_healthy"
]

def load_model_and_classes():
    global model, class_names

    # Load class names
    class_file = MODEL_DIR / "class_names.json"
    if class_file.exists():
        with open(class_file) as f:
            class_names = json.load(f)
        print(f"✅ Loaded {len(class_names)} class names")
    else:
        class_names = DEMO_CLASSES
        print(f"ℹ️  Using default {len(class_names)} PlantVillage classes")

    # Load model
    if TF_AVAILABLE:
        model_path = MODEL_DIR / "plantcare_model.keras"
        if model_path.exists():
            model = tf.keras.models.load_model(str(model_path))
            print(f"✅ Model loaded from {model_path}")
        else:
            print("⚠️  No trained model found — running in DEMO mode")
    else:
        print("⚠️  TensorFlow not available — running in DEMO mode")

# ─── Helper Functions ─────────────────────────────────────────────────────────

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def format_class_name(raw_name):
    """Convert 'Tomato___Early_blight' → {'plant': 'Tomato', 'disease': 'Early Blight'}"""
    parts = raw_name.split("_")
    plant = parts[0].replace("_", " ").replace(",", "")
    disease = parts[1].replace("_", " ").title() if len(parts) > 1 else "Unknown"
    return {"plant": plant, "disease": disease, "raw": raw_name}

def preprocess_image(image_path):
    """Load and preprocess image for MobileNetV2."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    if TF_AVAILABLE:
        arr = preprocess_input(arr)
    return arr

def run_inference(image_path):
    """Run model inference and return top-5 predictions."""
    arr = preprocess_image(image_path)

    if model is not None:
        preds = model.predict(arr, verbose=0)[0]
    else:
        # Demo mode: generate plausible-looking random predictions
        np.random.seed(int(Path(image_path).stat().st_size) % 9999)
        raw = np.random.dirichlet(np.ones(len(class_names)) * 0.3)
        # Boost a few classes to look realistic
        top_idx = np.random.choice(len(class_names), 3, replace=False)
        raw[top_idx[0]] *= 8
        raw[top_idx[1]] *= 3
        raw[top_idx[2]] *= 2
        preds = raw / raw.sum()

    # Top-5
    top5_idx   = np.argsort(preds)[::-1][:5]
    top5_probs = preds[top5_idx]

    results = []
    for idx, prob in zip(top5_idx, top5_probs):
        info = format_class_name(class_names[idx])
        results.append({
            "rank":       len(results) + 1,
            "class_idx":  int(idx),
            "raw_name":   info["raw"],
            "plant":      info["plant"],
            "disease":    info["disease"],
            "confidence": round(float(prob) * 100, 2),
            "is_healthy": "healthy" in class_names[idx].lower(),
        })

    return results

# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html", demo_mode=(model is None))

@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "tf_available": TF_AVAILABLE,
        "num_classes": len(class_names),
        "demo_mode": model is None,
    })

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Use PNG, JPG, JPEG, or WEBP"}), 400

    # Save uploaded file
    ext      = secure_filename(file.filename).rsplit(".", 1)[-1].lower()
    filename = f"{uuid.uuid4().hex}.{ext}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    try:
        predictions = run_inference(filepath)
    except Exception as e:
        os.remove(filepath)
        return jsonify({"error": f"Inference failed: {str(e)}"}), 500

    return jsonify({
        "success":     True,
        "demo_mode":   model is None,
        "image_url":   f"/static/uploads/{filename}",
        "predictions": predictions,
        "top_result":  predictions[0] if predictions else None,
    })

@app.route("/static/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

# ─── Boot ─────────────────────────────────────────────────────────────────────

load_model_and_classes()

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
