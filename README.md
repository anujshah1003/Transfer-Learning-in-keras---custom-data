# Transfer-Learning-in-keras---custom-data

This repository shows how we can use transfer learning in keras with the example of training a 4 class classification model using VGG-16 and Resnet-50 pre-trained weights.The vgg-16 and resnet-50 are the CNN models trained on more than a million images of 1000 different categories.

Transfer learning refers to the technique of using knowledge of one domain to another domain.i.e. a NN model trained on one dataset can be used for other dataset by fine-tuning the former network.

Definition : Given a source domain Ds and a learning task Ts, a target domain Dt and learning task Tt, transfer learning aims to help improve the learning of the the target predictive function Ft(.) in Dt using the knowledge in Ds and Ts, where Ds ≠ Dt, or Ts ≠ Tt.

A good explanation of how to use transfer learning practically is explained in http://cs231n.github.io/transfer-learning/

The vggface model weights is loaded as such without including the last layers by calling
