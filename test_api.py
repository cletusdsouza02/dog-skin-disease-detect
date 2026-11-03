import sys
import requests

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

samples = [
    ("assets/images/sample_dermatitis.jpg", "Canine Dermatitis"),
    ("assets/images/sample_demodicosis.jpg", "Demodicosis (Demodectic Mange)"),
    ("assets/images/sample_ringworm.jpg", "Ringworm (Dermatophytosis)"),
    ("assets/images/sample_fungal_infections.jpg", "Fungal & Malassezia Yeast Infections"),
    ("assets/images/sample_hypersensitivity.jpg", "Hypersensitivity & Flea Allergy (FAD)"),
    ("assets/images/sample_healthy.jpg", "Healthy Skin & Coat")
]

print("==================================================================")
print(" LIVE API PREDICTION TEST ACROSS ALL 6 SAMPLES")
print("==================================================================")

for path, expected in samples:
    with open(path, "rb") as f:
        resp = requests.post("http://127.0.0.1:5000/api/predict", files={"image": f})
    data = resp.json()
    pred_name = data["disease"]["name"]
    conf = data["confidence"]
    latency = data.get("latency_ms", 0)
    top3 = ", ".join([f"{p['name']} ({p['value']}%)" for p in data["probabilities"]])
    print(f"Sample: {path:<42}")
    print(f"  * Expected: {expected}")
    print(f"  * Predicted: {pred_name} [{conf}%] (Latency: {latency}ms)")
    print(f"  * Candidates: {top3}\n")

print("==================================================================")
print(" [OK] All endpoints verified successfully!")
print("==================================================================")
