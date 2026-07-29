import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS, cross_origin
from cnnclassifier.utils.common import decodeimage
from cnnclassifier.pipeline.predict import PredictPipeline

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

os.system("dvc remote modify origin --local user $DAGSHUB_USERNAME")
os.system("dvc remote modify origin --local password $DAGSHUB_TOKEN")
os.system("dvc pull -r origin")

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
    os.system("python train.py")  # یا dvc repro
    return "training Done successfully"


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