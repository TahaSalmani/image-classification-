import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS, cross_origin
from cnnclassifier.utils.common import decodeimage
from cnnclassifier.pipeline.predict import PredictPipeline

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

dagshub_user = os.environ.get("DAGSHUB_USERNAME")
dagshub_token = os.environ.get("DAGSHUB_TOKEN")

if dagshub_user and dagshub_token:
    os.system(f'dvc remote modify origin --local user "{dagshub_user}"')
    os.system(f'dvc remote modify origin --local password "{dagshub_token}"')
    os.system("dvc pull artifacts/training/model.h5 -r origin")
else:
    print("WARNING: DAGSHUB_USERNAME / DAGSHUB_TOKEN not set, skipping dvc pull.")

app = Flask(__name__)
CORS(app)

class ClientApp:
    def __init__(self):
        self.filename = "inputimage.jpeg"

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