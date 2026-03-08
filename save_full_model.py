"""
save_full_model.py
==================
Run this AFTER training is complete.
Loads the saved weights and exports a full plantcare_model.keras
that app.py can load for real inference.

Usage:
    python save_full_model.py
"""

import json
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models

MODEL_DIR  = "model"
IMG_SIZE   = 224
DROPOUT    = 0.3

# ── Load class names ──────────────────────────────────────────────────────────
class_file = os.path.join(MODEL_DIR, "class_names.json")
with open(class_file) as f:
    class_names = json.load(f)
num_classes = len(class_names)
print(f"✅ Loaded {num_classes} class names: {class_names}")

# ── Rebuild exact same architecture ───────────────────────────────────────────
base = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet",
)
base.trainable = True  # Must match phase 2 (fine-tuned) state

inputs  = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x       = base(inputs, training=False)
x       = layers.GlobalAveragePooling2D()(x)
x       = layers.Dense(256, activation="relu")(x)
x       = layers.Dropout(DROPOUT)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)
model   = models.Model(inputs, outputs)

# ── Load best weights (prefer phase 2, fall back to phase 1) ─────────────────
phase2_weights = os.path.join(MODEL_DIR, "best_phase2.weights.h5")
phase1_weights = os.path.join(MODEL_DIR, "best_phase1.weights.h5")

if os.path.exists(phase2_weights):
    model.load_weights(phase2_weights)
    print(f"✅ Loaded fine-tuned weights from {phase2_weights}")
elif os.path.exists(phase1_weights):
    model.load_weights(phase1_weights)
    print(f"✅ Loaded phase 1 weights from {phase1_weights}")
else:
    print("❌ No weights file found in model/ folder!")
    print("   Make sure training has completed first.")
    exit(1)

# ── Quick sanity check ────────────────────────────────────────────────────────
dummy = np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
preds = model.predict(dummy, verbose=0)
print(f"✅ Sanity check passed — output shape: {preds.shape}")

# ── Save full model ───────────────────────────────────────────────────────────
out_path = os.path.join(MODEL_DIR, "plantcare_model.keras")
model.save(out_path)
print(f"\n🎉 Full model saved to: {out_path}")
print(f"   Now run: python app.py")
print(f"   Then open: http://localhost:5000")
