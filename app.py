import os
import sys
import urllib.request

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS, cross_origin
from cnnclassifier.utils.common import decodeimage
from cnnclassifier.pipeline.predict import PredictPipeline

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')


MODEL_PATH = "artifacts/training/model.h5"

if not os.path.exists(MODEL_PATH):
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    print("Downloading model from DAGsHub...")

    # لینک مستقیم فایل مدل شما در DAGsHub
    dagshub_model_url = "https://dagshub.com/TahaSalmani/image-classification-/raw/main/artifacts/training/model.h5"

    try:
        urllib.request.urlretrieve(dagshub_model_url, MODEL_PATH)
        print("Model downloaded successfully!")
    except Exception as e:
        print(f"Error downloading model: {e}")

app = Flask(__name__)
CORS(app)


class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"


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
    image = request.json['image']
    decodeimage(image, clApp.filename)

    pipeline = PredictPipeline(clApp.filename)
    result = pipeline.predict()

    return jsonify(result)


if __name__ == "__main__":
    clApp = ClientApp()
    app.run(host='0.0.0.0', port=8080)