# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 16:37:30 2018

@author: kusiwu
"""
import numpy as np
from resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from imagenet_utils import decode_predictions

#%%

model = ResNet50(include_top=True,weights='imagenet')
model.summary()
model.layers[-1].get_config()
img_path = './predict_this/elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
#
preds = model.predict(x)
print('Predicted:', decode_predictions(preds))
## print: [[u'n02504458', u'African_elephant']]
#
##%%
model = ResNet50(include_top=False,weights='imagenet')
model.summary()
model.layers[-1].get_config()
img_path = './predict_this/elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
#preds=preds[0,0,0,:]
#print('Predicted:', decode_predictions(preds))