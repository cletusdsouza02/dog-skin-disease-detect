import os
import sys
import time
import json

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from train_models import build_mobilenet_v2, load_datasets, export_to_tflite

def main():
    print("==================================================================")
    print(" [TRAINING] MobileNetV2 Transfer Learning for Dog Skin Diseases")
    print("==================================================================")
    
    os.makedirs("saved_models", exist_ok=True)
    best_model_path = os.path.join("saved_models", "mobilenetv2_best.keras")
    
    print("Loading Train (1878), Val (328), and Test (348) splits...")
    train_ds, val_ds, test_ds = load_datasets(dataset_dir="dataset", batch_size=32)
    
    model, base_model = build_mobilenet_v2(num_classes=6)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    callbacks_list = [
        tf.keras.callbacks.ModelCheckpoint(
            best_model_path,
            save_best_only=True,
            monitor="val_accuracy",
            mode="max"
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-5
        )
    ]
    
    print("\nStarting Training (6 Epochs Transfer Learning)...")
    t0 = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=6,
        callbacks=callbacks_list
    )
    total_time = time.time() - t0
    print(f"\n[OK] Training completed in {total_time:.2f}s!")
    
    # Evaluate on test set
    if test_ds:
        print("\nEvaluating on Test Dataset (348 unseen clinical images)...")
        test_loss, test_acc = model.evaluate(test_ds)
        print(f"[OK] Final Test Accuracy: {test_acc*100:.2f}% | Test Loss: {test_loss:.4f}")
    
    # Save the model
    model.save(best_model_path)
    print(f"[OK] Saved best model weights: {best_model_path}")
    
    # Save class indices
    class_map = {
        0: "Dermatitis",
        1: "Fungal_infections",
        2: "Healthy",
        3: "Hypersensitivity",
        4: "demodicosis",
        5: "ringworm"
    }
    with open("saved_models/class_indices.json", "w") as f:
        json.dump(class_map, f, indent=2)
    print("[OK] Saved saved_models/class_indices.json")
    
    # Try exporting TFLite
    try:
        export_to_tflite(model, "saved_models/dog_skin_mobilenetv2.tflite")
    except Exception as e:
        print(f"[!] TFLite export notice: {e}")
        
    print("\n==================================================================")
    print(" [OK] Pipeline finished! Model is ready for real-time inference.")
    print("==================================================================")

if __name__ == "__main__":
    main()
