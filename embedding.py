import tensorflow as tf
from tensorflow.keras.models import Model
from keras.applications.resnet50 import preprocess_input
from numpy.linalg import norm
import numpy as np
from PIL import Image
import requests

import warnings
warnings.filterwarnings(action="ignore")

class FeatureExtractor:
    def __init__(self, weights: str = "imagenet", color_type: str = "RGB"):
        base_model = tf.keras.applications.VGG16(weights=weights)
        self.model = Model(inputs=base_model.input, outputs=base_model.get_layer("fc1").output)
        self.color_type = color_type
        
    def _convert_url(self, url):
        return Image.open(requests.get(url, stream=True).raw)
        
    def get_feature(self, url):
        images = self._convert_url(url)
        
        return self.model.get_extract(images)

    def get_extract(self, img):
        img = img.resize((224, 224))
        img = img.convert(self.color_type) 

        x = np.array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        feature = self.model.predict(x)[0]

        return feature / norm(feature)