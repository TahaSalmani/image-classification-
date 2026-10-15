import os

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

MODEL_PATH = os.path.join("artifacts", "training", "model.h5")
IMAGE_SIZE = (224, 224)
CLASS_LABELS = {0: "Normal", 1: "Pneumonia"}

_MODEL = None


def get_model():
    # Load the model once and reuse it. Reloading per request leaks memory
    # (TensorFlow never frees the previous graph), which OOM-kills Render's
    # 512 MB instance after a couple of predictions.
    global _MODEL
    if _MODEL is None:
        _MODEL = load_model(MODEL_PATH)
    return _MODEL


class PredictPipeline:
    def __init__(self, filename):
        self.filename = filename

    def predict(self):
        model = get_model()
        test_image = image.load_img(self.filename, target_size=IMAGE_SIZE)
        test_image = image.img_to_array(test_image)
        # training/evaluation rescale by 1/255; predict must do the same or the
        # model receives raw 0-255 pixels and its output is meaningless.
        test_image = test_image / 255.0
        test_image = np.expand_dims(test_image, axis=0)
        result = int(np.argmax(model.predict(test_image, verbose=0), axis=1)[0])
        return [{"image": CLASS_LABELS.get(result, "Normal")}]
