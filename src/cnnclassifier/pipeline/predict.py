import os
import threading

import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

from cnnclassifier.calibration import CALIBRATION

MODEL_PATH = os.environ.get(
    "MODEL_PATH", os.path.join("artifacts", "training", "model.h5")
)
IMAGE_SIZE = (224, 224)
CLASS_LABELS = {0: "Normal", 1: "Pneumonia"}
PREPROCESS = tf.keras.applications.vgg16.preprocess_input

_MODEL = None
_MODEL_LOCK = threading.Lock()
_PREDICT_LOCK = threading.Lock()


def _rebuild_architecture():
    """The head trained on top of VGG16 (see components/prepare_base_model.py)."""
    base = tf.keras.applications.VGG16(
        input_shape=(*IMAGE_SIZE, 3), weights=None, include_top=False
    )
    pooled = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    dropped = tf.keras.layers.Dropout(0.3)(pooled)
    prediction = tf.keras.layers.Dense(
        units=len(CLASS_LABELS), activation="softmax"
    )(dropped)
    return tf.keras.Model(inputs=base.input, outputs=prediction)


def _load_weights_file(model, path):
    """Copy weights straight out of the .h5 file, bypassing Keras config parsing.

    ``model.h5`` was written by Keras 3.15, which records initializer kwargs
    (``input_axes``) that older Keras releases reject, so ``load_model`` fails
    on them. The weights themselves are portable.
    """
    with h5py.File(path, "r") as handle:
        group = handle["model_weights"] if "model_weights" in handle else handle
        for layer in model.layers:
            if layer.name not in group:
                continue
            names = [
                name.decode() if isinstance(name, bytes) else str(name)
                for name in group[layer.name].attrs.get("weight_names", [])
            ]
            if not names:
                continue
            values = {
                name.split("/")[-1]: np.asarray(group[layer.name][name])
                for name in names
            }
            expected = [weight.name for weight in layer.weights]
            layer.set_weights([values[name] for name in expected])
    return model


def get_model():
    global _MODEL
    if _MODEL is None:
        with _MODEL_LOCK:
            if _MODEL is None:
                try:
                    _MODEL = load_model(MODEL_PATH, compile=False)
                except Exception as error:
                    print(
                        f"load_model failed ({type(error).__name__}); "
                        "loading weights directly from the .h5 file",
                        flush=True,
                    )
                    _MODEL = _load_weights_file(
                        _rebuild_architecture(), MODEL_PATH
                    )
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
            raw_scores = np.asarray(
                model.predict(test_image, verbose=0)[0], dtype=float
            )

        scores = CALIBRATION.apply(raw_scores)
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
                "thresholds": {
                    "high": CALIBRATION.high_threshold,
                    "low": CALIBRATION.low_threshold,
                },
            }
        ]
