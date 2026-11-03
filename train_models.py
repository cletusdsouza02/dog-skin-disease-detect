"""
Dog Skin Disease Detection - Multi-Model Training & Benchmarking Pipeline
Configured for 6 Dataset Classes (2,554 Images):
1. Dermatitis (465 train, 81 val, 66 test)
2. Fungal_infections (319 train, 56 val, 54 test)
3. Healthy (419 train, 73 val, 69 test)
4. Hypersensitivity (102 train, 18 val, 29 test)
5. demodicosis (500 train, 88 val, 100 test)
6. ringworm (73 train, 12 val, 30 test)

Supported Architectures:
1. MobileNetV2 (Primary Production Architecture - Fast, lightweight, 2.6M Params)
2. EfficientNetB0 (Benchmark Architecture - Compound Scaling, 4.4M Params)
3. ResNet-50 (Baseline Deep Residual Network - 24.6M Params)
"""

import os
import sys
import json
import time
import argparse

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import (
    MobileNetV2,
    EfficientNetB0,
    ResNet50
)

# Configuration Constants
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 6
EPOCHS = 10
LEARNING_RATE = 1e-4

CLASS_NAMES = [
    "Dermatitis",
    "Fungal_infections",
    "Healthy",
    "Hypersensitivity",
    "demodicosis",
    "ringworm"
]


# ==============================================================================
# 1. DATA LOADERS & AUGMENTATION PIPELINE
# ==============================================================================
def load_datasets(dataset_dir="dataset", img_size=IMG_SIZE, batch_size=BATCH_SIZE):
    """Loads train, valid, and test datasets from directory splits."""
    train_dir = os.path.join(dataset_dir, "train")
    val_dir = os.path.join(dataset_dir, "valid")
    test_dir = os.path.join(dataset_dir, "test")

    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Training directory not found at: {train_dir}")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        class_names=CLASS_NAMES,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=True
    )

    val_ds = None
    if os.path.exists(val_dir) and any(os.scandir(val_dir)):
        try:
            val_ds = tf.keras.utils.image_dataset_from_directory(
                val_dir,
                class_names=CLASS_NAMES,
                image_size=img_size,
                batch_size=batch_size,
                label_mode="categorical",
                shuffle=False
            )
        except Exception as e:
            print(f"[!] Notice: Validation split could not be loaded ({e})")
            val_ds = None

    test_ds = None
    if os.path.exists(test_dir) and any(os.scandir(test_dir)):
        try:
            test_ds = tf.keras.utils.image_dataset_from_directory(
                test_dir,
                class_names=CLASS_NAMES,
                image_size=img_size,
                batch_size=batch_size,
                label_mode="categorical",
                shuffle=False
            )
        except Exception as e:
            print(f"[!] Notice: Test split could not be loaded ({e})")
            test_ds = None

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=autotune)
    if val_ds:
        val_ds = val_ds.prefetch(buffer_size=autotune)
    if test_ds:
        test_ds = test_ds.prefetch(buffer_size=autotune)

    return train_ds, val_ds, test_ds


def get_data_augmentation():
    """Returns data augmentation pipeline optimized for dermatological lesion images."""
    return tf.keras.Sequential([
        layers.RandomRotation(0.12),
        layers.RandomFlip("horizontal"),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ], name="data_augmentation")


# ==============================================================================
# 2. MODEL 1: MobileNetV2 (PRIMARY DEPLOYED MODEL)
# ==============================================================================
def build_mobilenet_v2(num_classes=NUM_CLASSES, input_shape=(224, 224, 3)):
    """
    Builds MobileNetV2 Transfer Learning Architecture.
    - Lightweight, Inverted Residuals & Linear Bottlenecks
    - Optimized for low-latency inference & deployment
    """
    inputs = layers.Input(shape=input_shape, name="input_image")
    augmented = get_data_augmentation()(inputs)
    preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(augmented)

    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False

    x = base_model(preprocessed)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3, name="dropout_1")(x)
    x = layers.Dense(256, activation="relu", name="dense_features")(x)
    x = layers.Dropout(0.2, name="dropout_2")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="classification_head")(x)

    model = models.Model(inputs, outputs, name="MobileNetV2_DogSkin")
    return model, base_model


# ==============================================================================
# 3. MODEL 2: EfficientNetB0 (BENCHMARK MODEL)
# ==============================================================================
def build_efficientnet_b0(num_classes=NUM_CLASSES, input_shape=(224, 224, 3)):
    """
    Builds EfficientNetB0 Transfer Learning Architecture.
    - Compound Scaling (Depth, Width, Resolution)
    """
    inputs = layers.Input(shape=input_shape, name="input_image")
    augmented = get_data_augmentation()(inputs)
    preprocessed = tf.keras.applications.efficientnet.preprocess_input(augmented)

    base_model = EfficientNetB0(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False

    x = base_model(preprocessed)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.35, name="dropout_1")(x)
    x = layers.Dense(256, activation="relu", name="dense_features")(x)
    x = layers.Dropout(0.2, name="dropout_2")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="classification_head")(x)

    model = models.Model(inputs, outputs, name="EfficientNetB0_DogSkin")
    return model, base_model


# ==============================================================================
# 4. MODEL 3: ResNet-50 (DEEP RESIDUAL BASELINE)
# ==============================================================================
def build_resnet50(num_classes=NUM_CLASSES, input_shape=(224, 224, 3)):
    """
    Builds ResNet-50 Transfer Learning Architecture.
    - 50-Layer Deep Residual Network with Identity Skip Connections
    """
    inputs = layers.Input(shape=input_shape, name="input_image")
    augmented = get_data_augmentation()(inputs)
    preprocessed = tf.keras.applications.resnet50.preprocess_input(augmented)

    base_model = ResNet50(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False

    x = base_model(preprocessed)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4, name="dropout_1")(x)
    x = layers.Dense(512, activation="relu", name="dense_features")(x)
    x = layers.Dropout(0.3, name="dropout_2")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="classification_head")(x)

    model = models.Model(inputs, outputs, name="ResNet50_DogSkin")
    return model, base_model


# ==============================================================================
# 5. COMPILATION, TRAINING & TFLITE EXPORT
# ==============================================================================
def compile_and_train(model, base_model, train_ds, val_ds=None, model_name="mobilenetv2", epochs=EPOCHS, fine_tune_epochs=3):
    """Compiles the model and executes two-phase transfer learning & fine-tuning."""
    os.makedirs("saved_models", exist_ok=True)
    best_model_path = f"saved_models/{model_name}_best.keras"

    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")]
    )

    monitor_loss = "val_loss" if val_ds is not None else "loss"
    monitor_acc = "val_accuracy" if val_ds is not None else "accuracy"

    phase1_callbacks = [
        callbacks.EarlyStopping(monitor=monitor_loss, patience=4, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor=monitor_loss, factor=0.5, patience=2, min_lr=1e-6),
        callbacks.ModelCheckpoint(best_model_path, save_best_only=True, monitor=monitor_acc, mode="max")
    ]

    print(f"\n=======================================================")
    print(f" [Phase 1] Training Classifier Head: {model_name}")
    print(f" Total Parameters: {model.count_params():,}")
    print(f" Epochs: {epochs} | Monitored Metric: {monitor_acc}")
    print(f"=======================================================")

    history_phase1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=phase1_callbacks
    )

    history_phase2 = None
    if fine_tune_epochs > 0:
        # Fine-Tuning Phase (Unfreeze top layers, keeping BatchNorm frozen for numerical stability)
        print(f"\n[Phase 2] Fine-Tuning Top Layers of Base Model ({fine_tune_epochs} Epochs)...")
        base_model.trainable = True
        num_layers_to_unfreeze = min(30, len(base_model.layers))
        for layer in base_model.layers[:-num_layers_to_unfreeze]:
            layer.trainable = False
        for layer in base_model.layers[-num_layers_to_unfreeze:]:
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False
            else:
                layer.trainable = True

        model.compile(
            optimizer=optimizers.Adam(learning_rate=1e-5),
            loss="categorical_crossentropy",
            metrics=["accuracy", tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")]
        )

        phase2_callbacks = [
            callbacks.EarlyStopping(monitor=monitor_loss, patience=3, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(monitor=monitor_loss, factor=0.5, patience=2, min_lr=1e-6),
            callbacks.ModelCheckpoint(best_model_path, save_best_only=True, monitor=monitor_acc, mode="max")
        ]

        history_phase2 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=fine_tune_epochs,
            callbacks=phase2_callbacks
        )

    # Save final model weights
    model.save(best_model_path)
    print(f"[✓] Best model weights successfully saved to: {best_model_path}")

    # Save class indices
    class_map = {idx: name for idx, name in enumerate(CLASS_NAMES)}
    class_indices_path = "saved_models/class_indices.json"
    with open(class_indices_path, "w", encoding="utf-8") as f:
        json.dump(class_map, f, indent=2)
    print(f"[✓] Saved class index dictionary: {class_indices_path}")

    return model, history_phase1, history_phase2


def export_to_tflite(model, output_path="saved_models/dog_skin_mobilenetv2.tflite"):
    """Exports trained model to optimized TensorFlow Lite format for mobile deployment."""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()

        with open(output_path, "wb") as f:
            f.write(tflite_model)
        size_mb = len(tflite_model) / (1024 * 1024)
        print(f"[✓] Successfully exported TFLite mobile model: {output_path} ({size_mb:.2f} MB)")
        return output_path
    except Exception as e:
        print(f"[!] Notice: TFLite export bypassed ({e}). Keras .keras model remains fully functional.")
        return None


# ==============================================================================
# 6. COMMAND-LINE PIPELINE INTERFACE
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Dog Skin Disease Model Training & Benchmarking Pipeline")
    parser.add_argument("--model", type=str, default="mobilenetv2", choices=["mobilenetv2", "efficientnetb0", "resnet50"],
                        help="Model architecture to train (default: mobilenetv2)")
    parser.add_argument("--epochs", type=int, default=6, help="Epochs for Phase 1 classifier head training (default: 6)")
    parser.add_argument("--fine-tune-epochs", type=int, default=3, help="Epochs for Phase 2 fine-tuning (default: 3)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for dataset loading (default: 32)")
    parser.add_argument("--verify-only", action="store_true", help="Only verify architectures and dataset splits without running training")
    parser.add_argument("--no-export", action="store_true", help="Skip exporting model to TFLite format")
    args = parser.parse_args()

    print("==================================================================")
    print(" 🐶 Dog Skin Disease Detection - Training Pipeline")
    print(f" Target Classes ({NUM_CLASSES}): {', '.join(CLASS_NAMES)}")
    print("==================================================================")

    # 1. Dataset Verification
    print("Loading dataset splits from 'dataset/'...")
    train_ds, val_ds, test_ds = load_datasets(dataset_dir="dataset", batch_size=args.batch_size)
    print("[✓] Datasets loaded successfully.")

    # 2. Architecture registry
    model_builders = {
        "mobilenetv2": build_mobilenet_v2,
        "efficientnetb0": build_efficientnet_b0,
        "resnet50": build_resnet50
    }

    if args.verify_only:
        print("\n[VERIFICATION MODE] Testing model architectures:")
        for name, builder in model_builders.items():
            m, _ = builder()
            print(f"  * {name:<16}: {m.count_params():,} parameters")
        print("\n[OK] Model architectures and dataset splits verified.")
        return

    builder = model_builders[args.model]
    model, base_model = builder()
    print(f"\n[✓] Built architecture: {args.model.upper()} ({model.count_params():,} parameters)")

    # 3. Train Model
    t0 = time.time()
    trained_model, h1, h2 = compile_and_train(
        model=model,
        base_model=base_model,
        train_ds=train_ds,
        val_ds=val_ds,
        model_name=args.model,
        epochs=args.epochs,
        fine_tune_epochs=args.fine_tune_epochs
    )
    duration = time.time() - t0
    print(f"\n[✓] Training pipeline finished in {duration:.1f}s!")

    # 4. Evaluation on unseen Test split
    if test_ds:
        print("\nEvaluating on unseen Test Dataset...")
        eval_results = trained_model.evaluate(test_ds, verbose=1)
        metrics_dict = dict(zip(trained_model.metrics_names, eval_results))
        acc = metrics_dict.get("accuracy", 0.0)
        loss = metrics_dict.get("loss", 0.0)
        print(f"[✓] Final Test Accuracy: {acc * 100:.2f}% | Test Loss: {loss:.4f}")

    # 5. TFLite Export
    if not args.no_export:
        tflite_path = f"saved_models/dog_skin_{args.model}.tflite"
        export_to_tflite(trained_model, tflite_path)

    print("\n==================================================================")
    print(" [OK] Pipeline finished! Model is ready for real-time inference.")
    print("==================================================================")


if __name__ == "__main__":
    main()
