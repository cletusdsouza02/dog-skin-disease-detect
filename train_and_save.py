import os
import sys
import time
import json

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from train_models import (
    build_mobilenet_v2,
    load_datasets,
    compute_balanced_class_weights,
    compile_and_train,
    export_to_tflite,
    CLASS_NAMES
)

def main():
    print("==================================================================")
    print(" [BALANCED RETRAINING] MobileNetV2 Transfer Learning Pipeline")
    print(" Resolving Class Imbalance & Boosting Ringworm Sensitivity")
    print("==================================================================")
    
    os.makedirs("saved_models", exist_ok=True)
    best_model_path = os.path.join("saved_models", "mobilenetv2_best.keras")
    
    print("\n1. Loading Train (2,549 balanced), Val (328), and Test (348) datasets...")
    train_ds, val_ds, test_ds = load_datasets(dataset_dir="dataset", batch_size=32)
    
    print("\n2. Computing Inversely-Proportional Class Weights...")
    class_weights = compute_balanced_class_weights(dataset_dir="dataset")
    
    print("\n3. Building MobileNetV2 Architecture with Enhanced Augmentation...")
    model, base_model = build_mobilenet_v2(num_classes=6)
    
    print("\n4. Executing Two-Phase Training (Head Training + Fine-Tuning)...")
    t0 = time.time()
    trained_model, h1, h2 = compile_and_train(
        model=model,
        base_model=base_model,
        train_ds=train_ds,
        val_ds=val_ds,
        model_name="mobilenetv2",
        epochs=6,
        fine_tune_epochs=3,
        class_weights=class_weights
    )
    total_time = time.time() - t0
    print(f"\n[OK] Training completed in {total_time:.2f}s!")
    
    # Evaluate on test set
    if test_ds:
        print("\n5. Evaluating on Test Dataset (348 unseen clinical images)...")
        results = trained_model.evaluate(test_ds, return_dict=True)
        print(f"[OK] Final Test Accuracy: {results.get('accuracy', 0)*100:.2f}% | Loss: {results.get('loss', 0):.4f}")
    
    # Save the model
    trained_model.save(best_model_path)
    print(f"[OK] Saved best model weights: {best_model_path}")
    
    # Save class indices
    class_map = {idx: name for idx, name in enumerate(CLASS_NAMES)}
    with open("saved_models/class_indices.json", "w", encoding="utf-8") as f:
        json.dump(class_map, f, indent=2)
    print("[OK] Saved saved_models/class_indices.json")
    
    # Export optimized TFLite for Vercel/mobile
    tflite_path = "saved_models/dog_skin_mobilenetv2.tflite"
    export_to_tflite(trained_model, tflite_path)
        
    print("\n==================================================================")
    print(" [OK] Retraining complete! Balanced model ready for inference.")
    print("==================================================================")

if __name__ == "__main__":
    main()
