import tensorflow as tf

from cnnclassifier.entity.config_entity import PrepareBaseModelConfig


class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    def get_base_model(self):
        self.model = tf.keras.applications.VGG16(
            input_shape=tuple(self.config.params_image_size),
            weights=self.config.params_weight,
            include_top=self.config.params_include_top,
        )
        self.save_model(path=self.config.base_model_path, model=self.model)

    @staticmethod
    def prepare_ful_model(
        model, classes, freeze_all, freeze_till, learning_rate
    ):
        if freeze_all:
            for layer in model.layers:
                layer.trainable = False
        elif freeze_till is not None:
            for layer in model.layers[:freeze_till]:
                layer.trainable = False

        pooled = tf.keras.layers.GlobalAveragePooling2D()(model.output)
        dropped = tf.keras.layers.Dropout(0.3)(pooled)
        prediction = tf.keras.layers.Dense(
            units=classes, activation="softmax"
        )(dropped)

        full_model = tf.keras.Model(inputs=model.input, outputs=prediction)
        full_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"],
        )
        full_model.summary()
        return full_model

    def update_base_model(self):
        self.full_model = self.prepare_ful_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=False,
            freeze_till=self.config.params_freeze_till,
            learning_rate=self.config.params_learning_rate,
        )
        self.save_model(
            path=self.config.updated_base_model_path, model=self.full_model
        )

    @staticmethod
    def save_model(path, model: tf.keras.Model):
        model.save(path)
