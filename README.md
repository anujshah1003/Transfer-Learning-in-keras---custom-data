# Transfer Learning in Python using custom-data and VGG16 & Resnet50 networks

The video tutorial for Transfer learning with VGG-16 : https://www.youtube.com/watch?v=L7qjQu2ry2Q&feature=youtu.be

The video tutorial for Transfer learning with Resnet-50 : https://youtu.be/m5RjXjvAAhQ

This repository shows how we can use transfer learning in keras with the example of training a 4 class classification model using VGG-16 and Resnet-50 pre-trained weights.The vgg-16 and resnet-50 are the CNN models trained on more than a million images of 1000 different categories.

Transfer learning refers to the technique of using knowledge of one domain to another domain.i.e. a NN model trained on one dataset can be used for other dataset by fine-tuning the former network.

Definition : Given a source domain Ds and a learning task Ts, a target domain Dt and learning task Tt, transfer learning aims to help improve the learning of the the target predictive function Ft(.) in Dt using the knowledge in Ds and Ts, where Ds ≠ Dt, or Ts ≠ Tt.

A good explanation of how to use transfer learning practically is explained in http://cs231n.github.io/transfer-learning/

## When and how to fine-tune?

How do you decide what type of transfer learning you should perform on a new dataset?
This is a function of several factors, but the two most important ones are the size of the new dataset (small or big), and its similarity
to the original dataset (e.g. ImageNet-like in terms of the content of images and the classes, or very different, such as microscope images).
Keeping in mind that ConvNet features are more generic in early layers and more original-dataset-specific in later layers, 
here are some common rules of thumb for navigating the 4 major scenarios:

	New dataset is small and similar to original dataset. Since the data is small, it is not a good idea to fine-tune the ConvNet 
due to overfitting concerns. Since the data is similar to the original data, we expect higher-level features in the ConvNet to be 
relevant to this dataset as well. Hence, the best idea might be to train a linear classifier on the CNN codes.

	New dataset is large and similar to the original dataset. Since we have more data, we can have more confidence that we won’t 
overfit if we were to try to fine-tune through the full network.

	New dataset is small but very different from the original dataset. Since the data is small, it is likely best to only train a 
linear classifier. Since the dataset is very different, it might not be best to train the classifier form the top of the network, 
which contains more dataset-specific features. Instead, it might work better to train the SVM classifier from activations somewhere 
earlier in the network.

	New dataset is large and very different from the original dataset. Since the dataset is very large, we may expect that we can 
afford to train a ConvNet from scratch. However, in practice it is very often still beneficial to initialize with weights from a 
pretrained model. In this case, we would have enough data and confidence to fine-tune through the entire network.

* MODEL1

Training results
```
Epoch 1/12
646/646 [==============================] - 51s 79ms/step - loss: 0.6835 - acc: 0.7446 - val_loss: 0.2666 - val_acc: 0.9012
Epoch 2/12
646/646 [==============================] - 45s 70ms/step - loss: 0.1500 - acc: 0.9505 - val_loss: 0.1744 - val_acc: 0.9506
Epoch 3/12
646/646 [==============================] - 45s 70ms/step - loss: 0.0726 - acc: 0.9814 - val_loss: 0.1513 - val_acc: 0.9444
Epoch 4/12
646/646 [==============================] - 45s 70ms/step - loss: 0.0549 - acc: 0.9938 - val_loss: 0.1335 - val_acc: 0.9568
Epoch 5/12
646/646 [==============================] - 45s 70ms/step - loss: 0.0511 - acc: 0.9892 - val_loss: 0.1186 - val_acc: 0.9691
Epoch 6/12
646/646 [==============================] - 45s 70ms/step - loss: 0.0443 - acc: 0.9876 - val_loss: 0.1146 - val_acc: 0.9630
Epoch 7/12
646/646 [==============================] - 46s 71ms/step - loss: 0.0327 - acc: 0.9969 - val_loss: 0.1096 - val_acc: 0.9691
Epoch 8/12
646/646 [==============================] - 46s 71ms/step - loss: 0.0244 - acc: 0.9985 - val_loss: 0.1204 - val_acc: 0.9691
Epoch 9/12
646/646 [==============================] - 46s 71ms/step - loss: 0.0287 - acc: 0.9985 - val_loss: 0.0902 - val_acc: 0.9691
Epoch 10/12
646/646 [==============================] - 46s 71ms/step - loss: 0.0192 - acc: 1.0000 - val_loss: 0.1027 - val_acc: 0.9630
Epoch 11/12
646/646 [==============================] - 47s 72ms/step - loss: 0.0191 - acc: 1.0000 - val_loss: 0.1018 - val_acc: 0.9691
Epoch 12/12
646/646 [==============================] - 46s 72ms/step - loss: 0.0201 - acc: 0.9969 - val_loss: 0.0975 - val_acc: 0.9630


- Evaluate results
[INFO] loss=0.0975, accuracy: 96.2963%
```

* MODEL2

Training results
```
Total params: 24,769,156
Trainable params: 24,716,036
Non-trainable params: 53,120
__________________________________________________________________________________________________
Train on 646 samples, validate on 162 samples
Epoch 1/12
646/646 [==============================] - 47s 73ms/step - loss: 0.7344 - acc: 0.7043 - val_loss: 0.4553 - val_acc: 0.8951
Epoch 2/12
646/646 [==============================] - 47s 73ms/step - loss: 0.2434 - acc: 0.9211 - val_loss: 0.2483 - val_acc: 0.9259
Epoch 3/12
646/646 [==============================] - 47s 73ms/step - loss: 0.1677 - acc: 0.9474 - val_loss: 0.2066 - val_acc: 0.9506
Epoch 4/12
646/646 [==============================] - 46s 72ms/step - loss: 0.1346 - acc: 0.9613 - val_loss: 0.0861 - val_acc: 0.9815
Epoch 5/12
646/646 [==============================] - 46s 72ms/step - loss: 0.1219 - acc: 0.9613 - val_loss: 0.1871 - val_acc: 0.9691
Epoch 6/12
646/646 [==============================] - 46s 71ms/step - loss: 0.1395 - acc: 0.9505 - val_loss: 0.1141 - val_acc: 0.9753
Epoch 7/12
646/646 [==============================] - 46s 71ms/step - loss: 0.0771 - acc: 0.9768 - val_loss: 0.1111 - val_acc: 0.9815
Epoch 8/12
646/646 [==============================] - 46s 72ms/step - loss: 0.1321 - acc: 0.9628 - val_loss: 0.1067 - val_acc: 0.9630
Epoch 9/12
646/646 [==============================] - 46s 72ms/step - loss: 0.1422 - acc: 0.9582 - val_loss: 0.0939 - val_acc: 0.9877
Epoch 10/12
646/646 [==============================] - 46s 72ms/step - loss: 0.0446 - acc: 0.9861 - val_loss: 0.0875 - val_acc: 0.9815
Epoch 11/12
646/646 [==============================] - 46s 72ms/step - loss: 0.0419 - acc: 0.9861 - val_loss: 0.1241 - val_acc: 0.9815
Epoch 12/12
646/646 [==============================] - 46s 72ms/step - loss: 0.0552 - acc: 0.9799 - val_loss: 0.0790 - val_acc: 0.9877
Training time: -557.5128879547119
162/162 [==============================] - 12s 71ms/step

[INFO] loss=0.0790, accuracy: 98.7654%
model2 resaved.
```

* Prediction Result for Resnet50 after 20 epoch:

```
Loading image: ./predict_this/cat2.jpg
1/1 [==============================] - 0s 140ms/step
Found class from prediction: cats  accuracy%: 99.99356269836426

percentages of classes: [9.9993563e-01 6.1858686e-05 5.1179603e-07 1.9930824e-06]
All of the classes: ['cats', 'dogs', 'horses', 'humans']
```