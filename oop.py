from dataclasses import dataclass
from pathlib import Path
import  tensorflow as tf
model = tf.keras.applications.VGG16.VGG16(include_top=False, weights='imagenet')