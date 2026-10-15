import os
import sys
import tempfile

# Must be set before TensorFlow is imported (predict.py imports it).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("OMP_NUM_THREADS", "1")

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS, cross_origin
from cnnclassifier.utils.common import decodeimage
from cnnclassifier.pipeline.predict import PredictPipeline, get_model

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

dagshub_user = os.environ.get("DAGSHUB_USERNAME")
dagshub_token = os.environ.get("DAGSHUB_TOKEN")

if dagshub_user and dagshub_token:
    os.system('dvc remote modify origin --local auth basic')
    os.system(f'dvc remote modify origin --local user "{dagshub_user}"')
    os.system(f'dvc remote modify origin --local password "{dagshub_token}"')
    exit_code = os.system("timeout 60 dvc pull artifacts/training/model.h5 -r origin -v")
    print(f"dvc pull exit code = {exit_code}", flush=True)
else:
    print("WARNING: DAGSHUB_USERNAME / DAGSHUB_TOKEN not set, skipping dvc pull.", flush=True)

# Load the model once at boot so every request reuses the same instance.
# Tolerate a missing model so the container still starts and /predict returns
# a clear error instead of the whole app failing to import.
try:
    get_model()
    print("model preloaded successfully", flush=True)
except Exception as e:
    print(f"WARNING: could not preload model: {e}", flush=True)

app = Flask(__name__)
CORS(app)


@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template("index.html")


@app.route("/train", methods=['GET', 'POST'])
@cross_origin()
def trainingRoute():
    os.system("python main.py")
    return "Training done successfully"


@app.route("/predict", methods=['POST'])
@cross_origin()
def predictionRoute():
    payload = request.get_json(silent=True) or {}
    imgstring = payload.get('image')
    if not imgstring:
        return jsonify({"error": "request body must contain an 'image' field"}), 400

    # Write to a unique temp file: /tmp is the only reliably writable path on
    # Render, and a per-request name avoids concurrent requests clobbering one
    # another.
    fd, tmp_path = tempfile.mkstemp(suffix=".jpeg")
    os.close(fd)
    try:
        decodeimage(imgstring, tmp_path)
        result = PredictPipeline(tmp_path).predict()
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
