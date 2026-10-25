import os
import subprocess
import sys
import tempfile
from pathlib import Path

# These must be set before TensorFlow is imported by predict.py.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

from cnnclassifier.pipeline.predict import (
    MODEL_PATH,
    PredictPipeline,
    get_model,
    is_model_loaded,
)
from cnnclassifier.utils.common import decodeimage

MAX_BASE64_CHARS = 14 * 1024 * 1024


def _run(command, timeout):
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        print(f"command failed: {' '.join(command[:3])}: {error}", flush=True)
        return False
    if result.returncode != 0:
        print(
            f"command exited {result.returncode}: {' '.join(command[:3])}\n"
            f"{result.stderr[-2000:]}",
            flush=True,
        )
        return False
    return True


def _prepare_model():
    if os.environ.get("PULL_MODEL_ON_START", "1") != "1":
        return
    if Path(MODEL_PATH).is_file():
        return

    username = os.environ.get("DAGSHUB_USERNAME")
    token = os.environ.get("DAGSHUB_TOKEN")
    if not (username and token):
        print(
            "WARNING: model is missing and DAGSHUB_USERNAME / DAGSHUB_TOKEN "
            "are not set; /predict will return 503.",
            flush=True,
        )
        return

    _run(["dvc", "remote", "modify", "origin", "--local", "auth", "basic"], 20)
    _run(
        ["dvc", "remote", "modify", "origin", "--local", "user", username], 20
    )
    _run(
        ["dvc", "remote", "modify", "origin", "--local", "password", token],
        20,
    )
    print("pulling model from DagsHub...", flush=True)
    _run(["dvc", "pull", MODEL_PATH, "-r", "origin", "-v"], 90)


_prepare_model()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_BASE64_CHARS + 4096
CORS(app)


@app.errorhandler(413)
def request_too_large(_error):
    return jsonify({"error": "request body is too large"}), 413


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "ok",
            "model_loaded": is_model_loaded(),
            "model_path": MODEL_PATH,
        }
    )


@app.route("/predict", methods=["POST"])
def prediction_route():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "request body must be a JSON object"}), 400

    imgstring = payload.get("image")
    if not isinstance(imgstring, str) or not imgstring.strip():
        return jsonify({"error": "request body must contain an 'image' field"}), 400
    if len(imgstring) > MAX_BASE64_CHARS:
        return (
            jsonify({"error": "image is too large; maximum is 10 MB"}),
            413,
        )
    if not Path(MODEL_PATH).is_file():
        return jsonify({"error": "model is not available on this server"}), 503

    fd, tmp_path = tempfile.mkstemp(suffix=".jpeg")
    os.close(fd)
    try:
        decodeimage(imgstring, tmp_path)
        result = PredictPipeline(tmp_path).predict()
        return jsonify(result)
    except ValueError as error:
        return jsonify({"error": str(error)}), 400
    except FileNotFoundError as error:
        return jsonify({"error": str(error)}), 503
    except Exception as error:
        return jsonify({"error": str(error)}), 500
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


if __name__ == "__main__":
    try:
        get_model()
        print("model loaded for development server", flush=True)
    except Exception as error:
        print(f"WARNING: model not loaded: {error}", flush=True)
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8080")),
        debug=False,
    )
