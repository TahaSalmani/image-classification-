import  os
import  urllib.request as  request
from zipfile import  ZipFile
import  tensorflow as tf
from cnnclassifier.entity.config_entity import  PrepareBaseModelConfig
from pathlib import Path




class PrepareBaseModel :
    def __init__(self , config : PrepareBaseModelConfig) :
        self.config = config


    def get_base_model (self) :
        self.model = tf.keras.applications.vgg16.VGG16(

            input_shape = self.config.params_image_size,
            weights= self.config.params_weight ,
            include_top=self.config.params_include_top ,
            # output_shape = self.config.params_classes ,
        )

        self.save_model(path = self.config.base_model_path , model = self.model)

    @staticmethod
    def prepare_ful_model (model , classes , freeze_all , freeze_till , learning_rate) :
        if freeze_all :
            for layer in model.layers :
                layer.trainable = False
        elif (freeze_till is not None) and (freeze_till == 0) :
            for layer in model.layers :
                layer.trainable = False
        flatten_in  = tf.keras.layers.Flatten()(model.output)
        prediction = tf.keras.layers.Dense(units=classes , activation="softmax")(flatten_in)

        full_model = tf.keras.Model(
            inputs = model.input,
            outputs = prediction,
        )

        full_model.compile(
            optimizer =  tf.keras.optimizers.Adam(learning_rate=learning_rate) ,
            loss = tf.keras.losses.CategoricalCrossentropy(),
            metrics = ['accuracy']

        )

        full_model.summary()
        return full_model

    def update_base_model (self) :
        self.full_model = self.prepare_ful_model(
            model = self.model,
            classes = self.config.params_classes ,
            freeze_all= True ,
            learning_rate= self.config.params_learning_rate ,
            freeze_till= None ,

        )
        self.save_model(path = self.config.updated_base_model_path , model = self.full_model)

    @staticmethod
    def save_model(path = Path , model = tf.keras.Model ) :
        model.save(path)