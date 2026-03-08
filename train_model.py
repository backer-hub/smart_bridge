import os
import argparse
import numpy as np
import json
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras import layers, models, callbacks
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# ─── Configuration ────────────────────────────────────────────────────────────

IMG_SIZE    = 224
BATCH_SIZE  = 32
INIT_LR     = 1e-3
FINETUNE_LR = 1e-5
DROPOUT     = 0.3

# ─── Argument Parsing ─────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train PlantCare AI model")
    p.add_argument("--data_dir",  default="PlantVillage",  help="Path to dataset root")
    p.add_argument("--output_dir",default="model",          help="Where to save model")
    p.add_argument("--epochs",    type=int, default=20,     help="Initial training epochs")
    p.add_argument("--ft_epochs", type=int, default=10,     help="Fine-tuning epochs")
    return p.parse_args()

# ─── Dataset Loading ──────────────────────────────────────────────────────────

def load_datasets(data_dir):
    """Load and split dataset into train/val with 80/20 split."""
    print(f"\n📂 Loading dataset from: {data_dir}")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)
    print(f"✅ Found {num_classes} disease classes")
    print(f"   Classes: {class_names[:5]}{'...' if len(class_names) > 5 else ''}")

    # Save class names for inference
    return train_ds, val_ds, class_names

def preprocess_and_augment(train_ds, val_ds):
    """Apply MobileNetV2 preprocessing and data augmentation."""

    augment = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ])

    AUTOTUNE = tf.data.AUTOTUNE

    def preprocess(images, labels):
        return preprocess_input(images), labels

    train_ds = (
        train_ds
        .map(lambda x, y: (augment(x, training=True), y), num_parallel_calls=AUTOTUNE)
        .map(preprocess, num_parallel_calls=AUTOTUNE)
        .prefetch(AUTOTUNE)
    )
    val_ds = (
        val_ds
        .map(preprocess, num_parallel_calls=AUTOTUNE)
        .prefetch(AUTOTUNE)
    )

    return train_ds, val_ds

# ─── Model Architecture ───────────────────────────────────────────────────────

def build_model(num_classes):
    """Build MobileNetV2 transfer learning model."""
    print("\n🏗️  Building MobileNetV2 model...")

    base = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False  # Freeze for initial training

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(DROPOUT)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(INIT_LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    print(f"   Base params (frozen): {base.count_params():,}")
    print(f"   Trainable params:     {model.count_params() - base.count_params():,}")

    return model, base

# ─── Training ─────────────────────────────────────────────────────────────────

def train_phase1(model, train_ds, val_ds, epochs, output_dir):
    """Phase 1: Train only classification head."""
    print(f"\n🚀 Phase 1 — Training classification head ({epochs} epochs)")

    cbs = [
        callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
        callbacks.ModelCheckpoint(
            os.path.join(output_dir, "best_phase1.weights.h5"),
            monitor="val_accuracy", save_best_only=True, save_weights_only=True
        ),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
    ]

    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=cbs)
    return history

def train_phase2(model, base, train_ds, val_ds, epochs, output_dir):
    """Phase 2: Fine-tune entire network with low LR."""
    print(f"\n🔬 Phase 2 — Fine-tuning full network ({epochs} epochs)")

    base.trainable = True  # Unfreeze base

    model.compile(
        optimizer=tf.keras.optimizers.Adam(FINETUNE_LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    cbs = [
        callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
        callbacks.ModelCheckpoint(
            os.path.join(output_dir, "best_phase2.weights.h5"),
            monitor="val_accuracy", save_best_only=True, save_weights_only=True
        ),
    ]

    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=cbs)
    return history

# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_model(model, val_ds, class_names, output_dir):
    """Generate full classification report."""
    print("\n📊 Evaluating model...")

    all_labels, all_preds = [], []
    for images, labels in val_ds:
        preds = np.argmax(model.predict(images, verbose=0), axis=1)
        all_labels.extend(labels.numpy())
        all_preds.extend(preds)

    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    print("\nClassification Report:")
    print(report)

    report_path = os.path.join(output_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"✅ Report saved to {report_path}")

# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_history(h1, h2, output_dir):
    """Plot combined training curves."""
    acc  = h1.history["accuracy"]      + h2.history["accuracy"]
    val  = h1.history["val_accuracy"]  + h2.history["val_accuracy"]
    loss = h1.history["loss"]          + h2.history["loss"]
    vloss= h1.history["val_loss"]      + h2.history["val_loss"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ep = range(1, len(acc) + 1)
    ft_start = len(h1.history["accuracy"])

    ax1.plot(ep, acc,  label="Train"); ax1.plot(ep, val, label="Val")
    ax1.axvline(ft_start, color="red", linestyle="--", label="Fine-tune start")
    ax1.set_title("Accuracy"); ax1.legend()

    ax2.plot(ep, loss, label="Train"); ax2.plot(ep, vloss, label="Val")
    ax2.axvline(ft_start, color="red", linestyle="--", label="Fine-tune start")
    ax2.set_title("Loss"); ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_history.png"), dpi=150)
    print(f"✅ Training plot saved")

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    train_ds, val_ds, class_names = load_datasets(args.data_dir)
    train_ds, val_ds = preprocess_and_augment(train_ds, val_ds)

    # Save class names for Flask inference
    with open(os.path.join(args.output_dir, "class_names.json"), "w") as f:
        json.dump(class_names, f, indent=2)
    print(f"✅ Class names saved to {args.output_dir}/class_names.json")

    # Build & train
    model, base = build_model(len(class_names))

    h1 = train_phase1(model, train_ds, val_ds, args.epochs, args.output_dir)
    h2 = train_phase2(model, base, train_ds, val_ds, args.ft_epochs, args.output_dir)

    # Evaluate
    evaluate_model(model, val_ds, class_names, args.output_dir)
    plot_history(h1, h2, args.output_dir)

    # Save final model
    final_path = os.path.join(args.output_dir, "plantcare_model.keras")
    model.save(final_path)
    print(f"\n🎉 Training complete! Model saved to {final_path}")


if __name__ == "__main__":
    main()
