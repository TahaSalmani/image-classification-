import os
import threading

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

MODEL_PATH = os.environ.get(
    "MODEL_PATH", os.path.join("artifacts", "training", "model.h5")
)
IMAGE_SIZE = (224, 224)
CLASS_LABELS = {0: "Normal", 1: "Pneumonia"}
PREPROCESS = tf.keras.applications.vgg16.preprocess_input

_MODEL = None
_MODEL_LOCK = threading.Lock()
_PREDICT_LOCK = threading.Lock()


def get_model():
    global _MODEL
    if _MODEL is None:
        with _MODEL_LOCK:
            if _MODEL is None:
                _MODEL = load_model(MODEL_PATH, compile=False)
    return _MODEL


def is_model_loaded() -> bool:
    return _MODEL is not None


class PredictPipeline:
    def __init__(self, filename):
        self.filename = filename

    def predict(self):
        model = get_model()
        try:
            test_image = image.load_img(
                self.filename, target_size=IMAGE_SIZE, color_mode="rgb"
            )
        except (OSError, ValueError) as error:
            raise ValueError(f"invalid image: {error}") from error

        test_image = image.img_to_array(test_image)
        test_image = PREPROCESS(test_image)
        test_image = np.expand_dims(test_image, axis=0)

        with _PREDICT_LOCK:
            scores = np.asarray(
                model.predict(test_image, verbose=0)[0], dtype=float
            )

        result = int(np.argmax(scores))
        probabilities = {
            CLASS_LABELS[index]: float(scores[index])
            for index in CLASS_LABELS
        }
        return [
            {
                "image": CLASS_LABELS.get(result, "Normal"),
                "confidence": float(scores[result]),
                "probabilities": probabilities,
            }
        ]
