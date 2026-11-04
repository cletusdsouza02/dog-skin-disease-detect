"""
Dog Skin Disease Detection - Flask REST API Server
Provides health monitoring, image prediction REST endpoint, and disease catalog metadata.
"""

import os
import sys
import time

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from model_engine import DogSkinAIEngine, DISEASE_CATALOG

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder=BASE_DIR)

# Enable Cross-Origin Resource Sharing (CORS) for all frontend routes
CORS(app, resources={r"/*": {"origins": "*"}})

# Initialize AI Inference Engine
ai_engine = DogSkinAIEngine()

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.after_request
def add_header(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.route("/", methods=["GET"])
def home():
    """Serves the interactive web application interface."""
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/css/<path:filename>", methods=["GET"])
def serve_css(filename):
    return send_from_directory(os.path.join(BASE_DIR, "css"), filename)


@app.route("/js/<path:filename>", methods=["GET"])
def serve_js(filename):
    return send_from_directory(os.path.join(BASE_DIR, "js"), filename)


@app.route("/assets/<path:filename>", methods=["GET"])
def serve_assets(filename):
    return send_from_directory(os.path.join(BASE_DIR, "assets"), filename)


@app.route("/dataset/<path:filename>", methods=["GET"])
def serve_dataset(filename):
    return send_from_directory(os.path.join(BASE_DIR, "dataset"), filename)


@app.route("/api/health", methods=["GET"])
@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint to test server connectivity and API status."""
    return jsonify({
        "status": "online",
        "service": "Dog Skin Disease Detection API",
        "version": "2.0.0",
        "backend": "Python Flask",
        "model": ai_engine.model_name,
        "input_resolution": "224x224",
        "active_classes": len(DISEASE_CATALOG),
        "cors_enabled": True,
        "timestamp": time.time()
    }), 200


@app.route("/api/diseases", methods=["GET"])
@app.route("/diseases", methods=["GET"])
def get_diseases():
    """Returns the full knowledgebase catalog of canine skin conditions."""
    return jsonify({
        "status": "success",
        "count": len(DISEASE_CATALOG),
        "diseases": list(DISEASE_CATALOG.values())
    }), 200


@app.route("/api/predict", methods=["POST"])
@app.route("/predict", methods=["POST"])
def predict():
    """
    Main prediction endpoint. Receives uploaded image file via multipart/form-data,
    processes it through the AI engine, and returns predictions.
    """
    if "image" not in request.files:
        return jsonify({
            "status": "error",
            "message": "Missing 'image' key in form data upload."
        }), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({
            "status": "error",
            "message": "No selected file provided."
        }), 400

    if not allowed_file(file.filename):
        return jsonify({
            "status": "error",
            "message": f"Unsupported file type. Allowed formats: {', '.join(ALLOWED_EXTENSIONS)}"
        }), 400

    image_bytes = file.read()

    if len(image_bytes) > MAX_FILE_SIZE_BYTES:
        return jsonify({
            "status": "error",
            "message": "File size exceeds maximum 10MB limit."
        }), 400

    try:
        start_time = time.time()
        result = ai_engine.predict(image_bytes)
        latency_ms = round((time.time() - start_time) * 1000, 2)
        result["latency_ms"] = latency_ms

        return jsonify(result), 200

    except Exception as e:
        app.logger.error(f"Prediction processing error: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"AI model prediction failed: {str(e)}"
        }), 500


import webbrowser
from threading import Timer


def open_browser(target_url):
    """Automatically opens the default web browser after server starts."""
    try:
        webbrowser.open_new(target_url)
    except Exception:
        pass


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    local_url = f"http://localhost:{port}/"

    print("==================================================================")
    print("  🐶 Dog Skin Disease Detection Platform")
    print("==================================================================")
    print(f"  👉 Web App URL : {local_url}  (Ctrl + Click to open in VS Code)")
    print(f"  👉 Health Check: http://localhost:{port}/api/health")
    print(f"  👉 Predict API : http://localhost:{port}/api/predict")
    print("------------------------------------------------------------------")
    print("  🚀 Starting server and opening your browser...")
    print("  💡 Tip: Press Ctrl + C in this terminal anytime to stop.")
    print("==================================================================")

    # Automatically launch default browser in 1.5 seconds if not disabled
    if os.environ.get("NO_BROWSER") != "1":
        Timer(1.5, open_browser, args=[local_url]).start()

    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
