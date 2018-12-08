#Origina codes by anujshah1003 forked and developed by kusiwu: 12.08.2018
#git:   https://github.com/kusiwu/Transfer-Learning-in-keras---custom-data

import numpy as np
import os,sys
import time
from resnet50 import ResNet50
from keras.preprocessing import image
from keras.layers import GlobalAveragePooling2D, Dense, Dropout,Activation,Flatten

from imagenet_utils import preprocess_input
from keras.layers import Input
from keras.engine import Model
from keras.models import load_model
from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras import callbacks
from keras.utils.vis_utils import plot_model #for graphical demonstration of Network model #requires graphwiz. Not active for now...
from datetime import datetime


batch_trainsize=16 #decrease if you machine has low gpu or RAM
batch_testsize=16
nb_epoch = 1

# Loading the training data
PATH = os.getcwd()
# Define data path
data_path = PATH + '/data'
data_dir_list = os.listdir(data_path)

img_data_list=[]

for dataset in data_dir_list:
	img_list=os.listdir(data_path+'/'+ dataset)
	print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
	for img in img_list:
		img_path = data_path + '/'+ dataset + '/'+ img 
		img = image.load_img(img_path, target_size=(224, 224))
		x = image.img_to_array(img)
		x = np.expand_dims(x, axis=0)
		x = preprocess_input(x)
#		print('Input image shape:', x.shape)
		img_data_list.append(x)

img_data = np.array(img_data_list)
#img_data = img_data.astype('float32')
print (img_data.shape)
img_data=np.rollaxis(img_data,1,0)
print (img_data.shape)
img_data=img_data[0]
print (img_data.shape)


# Define the number of classes
num_classes = 4
num_of_samples = img_data.shape[0]
labels = np.ones((num_of_samples,),dtype='int64')

labels[0:202]=0
labels[202:404]=1
labels[404:606]=2
labels[606:]=3

names = ['cats','dogs','horses','humans']
# convert class labels to on-hot encoding
Y = np_utils.to_categorical(labels, num_classes)

#Shuffle the dataset
x,y = shuffle(img_data,Y, random_state=2)
# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

###########################################################################################################################
# Custom_resnet_model_1
#Training the classifier alone
image_input = Input(shape=(224, 224, 3))



previouslytrainedModelpath ='./trained_models/resnet50model1.h5'
if os.path.isfile(previouslytrainedModelpath):
    print('Loading previously trained model1...')
    model = load_model(previouslytrainedModelpath)
    print(previouslytrainedModelpath + ' successfully loaded!')
    custom_resnet_model=model
else :
    print('Initializing resnet50 model1')
    model = ResNet50(input_tensor=image_input, include_top=True,weights='imagenet')
    last_layer = model.get_layer('avg_pool').output
    x= Flatten(name='flatten')(last_layer)
    out = Dense(num_classes, activation='softmax', name='output_layer')(x)
    custom_resnet_model = Model(inputs=image_input,outputs= out)
#model.summary()

#custom_resnet_model.summary()

for layer in custom_resnet_model.layers[:-1]:
	layer.trainable = False

custom_resnet_model.layers[-1].trainable

custom_resnet_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

###### please install pydot with pip install pydot and download graphwiz from website :https://graphviz.gitlab.io/_pages/Download/Download_windows.html
####add graphwiz path to visualize model graph. No need for now.
#os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
#plot_model(custom_resnet_model, to_file='outputs/model1_plot.png', show_shapes=True, show_layer_names=True)


# callback for tensorboard integration
tb = callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)

# checkpoints. Save model if val_accuracy increases.
filepath="./trained_models/model1_-{epoch:02d}-{val_acc:.2f}_"
checkpoint = callbacks.ModelCheckpoint(filepath+f'{datetime.now():%Y-%m-%d_%H.%M.%S}'+'.h5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')

t=time.time()
hist = custom_resnet_model.fit(X_train, y_train, batch_size=batch_trainsize, epochs=nb_epoch, verbose=1, validation_data=(X_test, y_test),callbacks=[tb,checkpoint])
print('Training time: %s' % (time.time()-t))
(loss, accuracy) = custom_resnet_model.evaluate(X_test, y_test, batch_size=batch_testsize, verbose=1)

print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))

# serialize model to JSON
model_json = custom_resnet_model.to_json()
with open("./outputs/custom_resnet_model1.json", "w") as json_file:
    json_file.write(model_json)

#Save model
custom_resnet_model.save('./trained_models/resnet50model1.h5')
print('model1 resaved.')
del custom_resnet_model #prevent memory leak
###########################################################################################################################

# Fine tune the resnet 50
#image_input = Input(shape=(224, 224, 3))

previouslytrainedModelpath ='./trained_models/resnet50model2.h5'
if os.path.isfile(previouslytrainedModelpath):
    print('Loading previously trained model2...')
    model = load_model(previouslytrainedModelpath)
    print(previouslytrainedModelpath + ' successfully loaded!')
    custom_resnet_model2=model
else :
    print('Initializing resnet50 model2')
    model = ResNet50(weights='imagenet',include_top=False)
    last_layer = model.output
    # add a global spatial average pooling layer
    x = GlobalAveragePooling2D()(last_layer)
    # add fully-connected & dropout layers
    x = Dense(512, activation='relu',name='fc-1')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu',name='fc-2')(x)
    x = Dropout(0.5)(x)
    # a softmax layer for 4 classes
    out = Dense(num_classes, activation='softmax',name='output_layer')(x)
    # this is the model we will train
    custom_resnet_model2 = Model(inputs=model.input, outputs=out)

#model.summary()

#custom_resnet_model2.summary()

for layer in custom_resnet_model2.layers[:-6]:
	layer.trainable = False

custom_resnet_model2.layers[-1].trainable

custom_resnet_model2.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

###### please install pydot with pip install pydot and download graphwiz from website :https://graphviz.gitlab.io/_pages/Download/Download_windows.html
####add graphwiz path to visualize model graph. No need for now.
#os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
#plot_model(custom_resnet_model2, to_file='outputs/model2_plot.png', show_shapes=True, show_layer_names=True)


# callback for tensorboard integration
tb = callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)

# checkpoints. Save model if val_accuracy increases.
filepath="./trained_models/model2_-{epoch:02d}-{val_acc:.2f}_"
checkpoint = callbacks.ModelCheckpoint(filepath+f'{datetime.now():%Y-%m-%d_%H.%M.%S}'+'.h5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')


t=time.time()
hist = custom_resnet_model2.fit(X_train, y_train, batch_size=batch_trainsize, epochs=nb_epoch, verbose=1, validation_data=(X_test, y_test),callbacks=[tb,checkpoint])
print('Training time: %s' % (time.time()-t))
(loss, accuracy) = custom_resnet_model2.evaluate(X_test, y_test, batch_size=batch_testsize, verbose=1)

print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))
# serialize model to JSON
model_json2 = custom_resnet_model2.to_json()
with open("./outputs/custom_resnet_model2.json", "w") as json_file:
    json_file.write(model_json2)

#Save model
custom_resnet_model2.save('./trained_models/resnet50model2.h5')
print('model2 resaved.')
del custom_resnet_model2 #prevent memory leak
############################################################################################
import matplotlib.pyplot as plt
# visualizing losses and accuracy
train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
train_acc=hist.history['acc']
val_acc=hist.history['val_acc']
xc=range(12)

plt.figure(1,figsize=(7,5))
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train','val'])
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])

plt.figure(2,figsize=(7,5))
plt.plot(xc,train_acc)
plt.plot(xc,val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train','val'],loc=4)
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])