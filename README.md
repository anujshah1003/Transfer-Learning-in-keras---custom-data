# Transfer-Learning-in-keras---custom-data

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
