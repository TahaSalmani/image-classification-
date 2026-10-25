from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from cnnclassifier.entity.config_entity import EvaluationConfig
from cnnclassifier.utils.common import save_json

PREPROCESS = tf.keras.applications.vgg16.preprocess_input


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    def _test_generator(self):
        datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=PREPROCESS
        )
        self.test_generator = datagenerator.flow_from_directory(
            directory=self.config.test_data,
            target_size=tuple(self.config.params_image_size[:-1]),
            batch_size=self.config.params_batch_size,
            class_mode="categorical",
            shuffle=False,
            interpolation="bilinear",
        )

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path, compile=False)

    def _compute_metrics(self, y_true, probabilities):
        y_pred = np.argmax(probabilities, axis=1)
        labels = list(range(probabilities.shape[1]))
        names = [
            name
            for name, index in sorted(
                self.test_generator.class_indices.items(),
                key=lambda item: item[1],
            )
        ]
        matrix = confusion_matrix(y_true, y_pred, labels=labels)
        report = classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=names,
            output_dict=True,
            zero_division=0,
        )
        try:
            if len(labels) == 2:
                auc = float(roc_auc_score(y_true, probabilities[:, 1]))
            else:
                auc = float(
                    roc_auc_score(
                        y_true, probabilities, multi_class="ovr"
                    )
                )
        except ValueError:
            auc = None

        per_class = {
            name: {
                key: float(report[name][key])
                for key in ("precision", "recall", "f1-score", "support")
            }
            for name in names
        }
        return {
            "auc": auc,
            "precision": float(report["weighted avg"]["precision"]),
            "recall": float(report["weighted avg"]["recall"]),
            "f1": float(report["weighted avg"]["f1-score"]),
            "confusion_matrix": matrix.tolist(),
            "class_labels": names,
            "per_class": per_class,
        }

    def evaluation(self):
        model = self.load_model(self.config.path_of_model)
        self._test_generator()
        y_true = np.asarray(self.test_generator.classes)
        probabilities = model.predict(self.test_generator, verbose=1)
        one_hot = tf.keras.utils.to_categorical(
            y_true, num_classes=probabilities.shape[1]
        )
        loss = float(
            tf.keras.losses.categorical_crossentropy(
                one_hot, probabilities
            ).numpy().mean()
        )
        accuracy = float(
            (np.argmax(probabilities, axis=1) == y_true).mean()
        )
        self.score = (loss, accuracy)
        self.metrics = self._compute_metrics(y_true, probabilities)

    def save_score(self):
        scores = {
            "split": "test",
            "n_samples": int(self.test_generator.samples),
            "loss": float(self.score[0]),
            "accuracy": float(self.score[1]),
            **self.metrics,
        }
        save_json(path=Path("scores.json"), data=scores)
