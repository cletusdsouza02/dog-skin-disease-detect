import os, sys, time, glob, warnings
warnings.filterwarnings("ignore")

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import joblib
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

CLASS_NAMES = ["Dermatitis", "Fungal_infections", "Healthy", "Hypersensitivity", "demodicosis", "ringworm"]

print("==================================================================")
print(" 🚀 HIGH-PRECISION HYBRID RESNET-50 + RBF-SVM TRAINING PIPELINE")
print(" Resolving Class Overlap & Maximizing Diagnostic Precision")
print("==================================================================")

print("\n1. Loading Deep Feature Extractor (ResNet50 ImageNet)...")
resnet = ResNet50(weights="imagenet", include_top=False, pooling="avg", input_shape=(224, 224, 3))

def extract_set(folder):
    X_imgs = []
    y_lbls = []
    filenames = []
    for idx, c in enumerate(CLASS_NAMES):
        c_folder = os.path.join(folder, c)
        if not os.path.exists(c_folder):
            continue
        files = glob.glob(os.path.join(c_folder, "*.*"))
        for f in files:
            try:
                im = Image.open(f).convert("RGB").resize((224, 224))
                arr = np.array(im, dtype=np.float32)
                X_imgs.append(arr)
                y_lbls.append(idx)
                filenames.append(f)
            except Exception:
                pass
    X_arr = np.array(X_imgs, dtype=np.float32)
    X_prep = preprocess_input(X_arr)
    print(f"   Forward-passing {len(X_arr)} images through 50 residual layers...")
    features = resnet.predict(X_prep, batch_size=32, verbose=1)
    return features, np.array(y_lbls), filenames

print("\n2. Extracting Train Set Features (dataset/train)...")
t0 = time.time()
X_tr, y_tr, _ = extract_set("dataset/train")
print(f"   Train features extracted: {X_tr.shape} in {time.time()-t0:.1f}s")

print("\n3. Extracting Valid Set Features (dataset/valid)...")
t0 = time.time()
X_vl, y_vl, _ = extract_set("dataset/valid")
print(f"   Valid features extracted: {X_vl.shape} in {time.time()-t0:.1f}s")

print("\n4. Extracting Test Set Features (dataset/test)...")
t0 = time.time()
X_te, y_te, _ = extract_set("dataset/test")
print(f"   Test features extracted: {X_te.shape} in {time.time()-t0:.1f}s")

# Combine Train + Valid for maximum training signal
X_train_full = np.vstack([X_tr, X_vl])
y_train_full = np.concatenate([y_tr, y_vl])

print(f"\n5. Total Training Examples: {len(X_train_full)} | Testing Examples: {len(X_te)}")
print("   Fitting StandardScaler and Non-Linear RBF Kernel SVM...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_full)
X_test_scaled = scaler.transform(X_te)

t0 = time.time()
svm = SVC(
    kernel="rbf",
    C=10.0,
    gamma="scale",
    class_weight="balanced",
    probability=True,
    random_state=42
)
svm.fit(X_train_scaled, y_train_full)
print(f"   SVM fitted in {time.time()-t0:.1f}s!")

# Evaluate on unseen Test Set
preds = svm.predict(X_test_scaled)
acc = accuracy_score(y_te, preds)
print("\n==================================================================")
print(f" 🎯 FINAL TEST SET ACCURACY: {acc*100:.2f}%")
print("==================================================================")
print(classification_report(y_te, preds, target_names=CLASS_NAMES, digits=3))

cm = confusion_matrix(y_te, preds)
print("Confusion Matrix:")
print(f"{'True \\ Pred':<20} | " + " | ".join([f"{c[:8]:>8}" for c in CLASS_NAMES]))
print("-" * 80)
for idx, c_true in enumerate(CLASS_NAMES):
    row = " | ".join([f"{cm[idx][p_idx]:>8}" for p_idx in range(len(CLASS_NAMES))])
    print(f"{c_true:<20} | {row}")

# Save models
os.makedirs("saved_models", exist_ok=True)
svm_path = "saved_models/dog_skin_resnet50_svm.joblib"
scaler_path = "saved_models/dog_skin_resnet50_scaler.joblib"
joblib.dump(svm, svm_path)
joblib.dump(scaler, scaler_path)
print(f"\n[OK] Saved high-precision model: {svm_path} ({os.path.getsize(svm_path)/1024/1024:.2f} MB)")
print(f"[OK] Saved scaler transform: {scaler_path}")
print("==================================================================")
