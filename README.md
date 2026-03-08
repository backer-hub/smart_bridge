# 🌱 PlantCare AI — Plant Disease Detection System

An end-to-end AI system that detects plant diseases from leaf images using MobileNetV2
transfer learning, trained on the PlantVillage dataset.

---

## 🏗️ Project Structure

```
plantcare-ai/
├── train_model.py          # Full training pipeline (Phase 1 + Fine-tuning)
├── app.py                  # Flask web application
├── requirements.txt        # Python dependencies
├── templates/
│   └── index.html          # Responsive web UI
├── static/
│   ├── uploads/            # Uploaded user images (auto-created)
│   └── css/, js/           # Static assets
└── model/                  # Saved model files (after training)
    ├── plantcare_model.keras
    ├── class_names.json
    └── classification_report.txt
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download the PlantVillage dataset
```bash
# Option A: Kaggle
kaggle datasets download -d emmarex/plantdisease
unzip plantdisease.zip

# Option B: TensorFlow Datasets
python -c "import tensorflow_datasets as tfds; tfds.load('plant_village')"
```

### 3. Train the model
```bash
python train_model.py \
  --data_dir /path/to/PlantVillage \
  --output_dir model \
  --epochs 20 \
  --ft_epochs 10
```

The training script will:
- **Phase 1** (20 epochs): Train only the custom classification head (base frozen)
- **Phase 2** (10 epochs): Fine-tune the entire network with LR=1e-5
- Save best weights via `ModelCheckpoint`
- Stop early via `EarlyStopping` (patience=5)
- Output `classification_report.txt` and `training_history.png`

### 4. Run the web app
```bash
python app.py
```
Visit http://localhost:5000

---

## 🧠 Model Architecture

```
Input (224×224×3)
    │
MobileNetV2 (ImageNet weights, 2.2M params)
    │  [Phase 1: frozen  →  Phase 2: unfrozen]
GlobalAveragePooling2D
    │
Dense(256, relu)
    │
Dropout(0.3)
    │
Dense(38, softmax)   ← 38 PlantVillage classes
```

**Transfer Learning Strategy:**
| Phase | Base layers | Learning Rate | Epochs |
|-------|------------|---------------|--------|
| 1 — Head training | Frozen      | 1e-3 | 20 |
| 2 — Fine-tuning   | Unfrozen    | 1e-5 | 10 |

---

## 📊 Disease Classes (38 total)

| Plant | Diseases |
|-------|----------|
| Tomato | Bacterial spot, Early blight, Late blight, Leaf mold, Septoria, Spider mites, Target spot, YLC Virus, Mosaic virus, Healthy |
| Potato | Early blight, Late blight, Healthy |
| Apple | Scab, Black rot, Cedar rust, Healthy |
| Grape | Black rot, Esca, Leaf blight, Healthy |
| Corn | Cercospora, Common rust, Northern Leaf Blight, Healthy |
| + more | Pepper, Peach, Orange, Strawberry, Squash, Soybean, Cherry, Blueberry, Raspberry |

---

## 🌐 API Reference

### `POST /predict`
Upload an image and get disease predictions.

**Request:** `multipart/form-data` with field `file` (image/*)

**Response:**
```json
{
  "success": true,
  "demo_mode": false,
  "image_url": "/static/uploads/abc123.jpg",
  "predictions": [
    {
      "rank": 1,
      "plant": "Tomato",
      "disease": "Early Blight",
      "confidence": 87.3,
      "is_healthy": false
    }
  ],
  "top_result": { ... }
}
```

### `GET /health`
```json
{
  "status": "ok",
  "model_loaded": true,
  "tf_available": true,
  "num_classes": 38,
  "demo_mode": false
}
```

---

## 📈 Expected Performance

After full training on PlantVillage (54,309 images):

| Metric | Phase 1 | After Fine-tuning |
|--------|---------|-------------------|
| Val Accuracy | ~88% | ~96%+ |
| Val Loss | ~0.40 | ~0.15 |

Results vary by class — common diseases with many training samples typically
achieve F1 > 0.97, while rare classes may be lower.

---

## 🛠️ Production Deployment

```bash
# Using Gunicorn (recommended)
gunicorn -w 4 -b 0.0.0.0:8000 app:app

# With Docker
docker build -t plantcare-ai .
docker run -p 5000:5000 -v $(pwd)/model:/app/model plantcare-ai
```

---

## 📚 Dataset Citation

Hughes, D.P.; Salathé, M. (2015). *An open access repository of images on plant health
to enable the development of mobile disease diagnostics.* arXiv:1511.08060.

---

*Built with TensorFlow 2.x, Flask, and MobileNetV2 — PlantCare AI*
