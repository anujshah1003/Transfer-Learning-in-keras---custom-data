# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 16:37:30 2018

@author: kusiwu
"""
import numpy as np
from vgg16 import VGG16
from resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from imagenet_utils import decode_predictions

model = VGG16(include_top=True, weights='imagenet')

img_path = './predict_this/elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
print('Input image shape:', x.shape)

preds = model.predict(x)
print('Predicted:', decode_predictions(preds))

model.summary()
model.layers[-1].get_config()

#%%

model = VGG16(weights='imagenet', include_top=False)

model.summary()
model.layers[-1].get_config()

img_path = './predict_this/elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

features = model.predict(x)