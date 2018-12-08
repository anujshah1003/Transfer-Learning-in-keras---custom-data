# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 01:09:08 2018

@author: kusiwu
@git: https://github.com/kusiwu
"""
# DEPENDENCIES
from tensorflow.python.client import device_lib
from keras.models import load_model
import numpy as np
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input

img_width, img_height = 224, 224
dimensionx=3
classnames = ['cats','dogs','horses','humans']

#try the images commented below.
#img_path = './predict_this/elephant.jpg' #actually elephant does not exist in our classes. Test it and see low accuracy as: Found class from prediction: horses  accuracy%: 0.002518
#img_path = './predict_this/cat.jpg'
img_path = './predict_this/cat2.jpg'
#img_path = './predict_this/dog.jpg'
#img_path = './predict_this/cat2.jpg'
print('Loading image: ',img_path)
img = image.load_img(img_path, target_size=(img_width, img_height))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
xTestPictures = preprocess_input(x)

model = load_model('./trained_models/resnet50model1.h5')

#model.summary()
#print(device_lib.list_local_devices())

yFit = model.predict(xTestPictures, batch_size=10, verbose=1)
y_classes = yFit.argmax(axis=-1)
print("Found class from prediction:",classnames[y_classes.flatten()[0]],' accuracy%:',yFit.flatten()[0]*100.0);
print();

print("percentages of classes:",yFit.flatten());
print("All of the classes:",classnames);

#del model