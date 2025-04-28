# MobileNetV2-in-depth

Sources:

- https://arxiv.org/pdf/1801.04381
- https://medium.com/@luis_gonzales/a-look-at-mobilenetv2-inverted-residuals-and-linear-bottlenecks-d49f85c12423
- https://medium.com/@alfred.weirich/finetuning-tensorflow-keras-networks-basics-using-mobilenetv2-as-an-example-8274859dc232
- https://www.tensorflow.org/tutorials/images/transfer_learning

## Overview of Standard 2D Convolution

Before diving into the mechanics of the depth-wise separable convolution, let’s review the standard 2D convolution. Suppose a convolution operation transforms an input volume of dimensions Dᵤ x Dᵤ x M to an output volume of dimensions Dᵥ x Dᵥ x N, as shown in Fig. 1(a). Specifically, we require N filters, each of dimension Dᵣ x Dᵣ x M, as shown in Fig. 1(b). 

Context: In standard 2D convolution, the input volume has dimensions Du×Du×M (width, height, channels), and each filter has dimensions Dr×Dr×M. The depth M of the filter must match the number of input channels, so that the filter can capture information from all channels (for example, all three color channels in an RGB image) at once. Each filter produces one 2D feature map. By using NN different filters, the output volume has NN channels — one for each filter. The number of filters (like 64) is chosen based on the complexity of the input, the depth of the network, available hardware, and experimentation. Early layers usually have fewer filters to capture simple features, while deeper layers have more filters to capture complex patterns. A common practice is to start with fewer filters (e.g., 32) and double them in deeper layers (e.g., 64, 128, 256). 

The two interpretations of convolution operations: 1. The spatial filtering interpretation: each input channel is convolved separately with the corresponding channel of the filter. For each position, the results from all channels are then summed together to produce a single output value. This process is repeated spatially across the whole input to build the output feature map. For NN filters, this operation is performed NN times, once for each filter, meaning that every input channel is filtered NN times, which can become quite computationally expensive; 2. The linear combinations interpretation: The filter is stepped spatially across the input, and at each location, an inner product (dot product) is taken between the overlapping regions of the input and the filter. Practically, both regions are flattened into vectors, and the dot product computes a single output value for each step.

## Depth-wise Separable Convolutions

Depth-wise separable convolutions were introduced in MobileNetV1 and are a type of factorized convolution that reduce the computational cost as compared to standard convolutions.

The depth-wise separable convolution is structured around the decomposition of the spatial filtering and linear combinations. As before, suppose an input volume of Dᵤ x Dᵤ x M is transformed to an output volume of Dᵥ x Dᵥ x N, as shown in Fig. 4(a). The first set of filters, shown in Fig. 4(b), are comprised of M single-channel filters, mapping the input volume to Dᵥ x Dᵥ x M on a per-channel basis. This stage, known as depth-wise convolutions, resembles the intermediate tensor shown in Fig. 3(c) and achieves the spatial filtering component. In order to construct new features from those already captured by the input volume, we require a linear combination. To do so, 1x1 kernels are used along the depth of the intermediate tensor; this step is referred to as point-wise convolution. N such 1x1 filters are used, resulting in the desired output volume of Dᵥ x Dᵥ x N.

The lowered computational cost of depth-wise separable convolutions comes predominantly from limiting the spatial filtering from M*N times in standard convolutions to M times.

## Inverted Residual and Linear Bottleneck Layer

In deep neural networks, the activations at each layer (the output tensors) can be thought of as a collection of "pixels" with multiple channels.
The important information across these activations forms a manifold of interest — a structured set of points — that often lies inside a low-dimensional subspace even if the full space has many channels. Earlier models like MobileNetV1 exploited this by reducing the number of channels (using a width multiplier) to make networks smaller and faster, assuming the information could still fit. However, this simple idea doesn't fully hold because real networks have non-linearities like ReLU, which:
- Keep the data linear where activations stay positive (alive),
- But destroy information when they zero out (kill) channels.

Thus, ReLU can preserve information only if the manifold already fits tightly inside a low-dimensional space. If not, ReLU might destroy critical information, especially in narrow layers where every channel matters. MobileNetV2 solves this by using linear bottlenecks:
- Narrow layers without ReLU, which compress features without risking information loss.
- This protects the manifold structure and keeps the network both efficient and accurate.

After compression, the network expands back and applies ReLU safely where losing a few channels isn’t as dangerous.
Experiments showed that adding ReLU inside bottlenecks actually hurts performance — confirming that the "safe compression" strategy is crucial for good results.

The layer takes in a low-dimensional tensor with k channels and performs three separate convolutions. First, a point-wise (1x1) convolution is used to expand the low-dimensional input feature map to a higher-dimensional space suited to non-linear activations, then ReLU6 is applied. The expansion factor is referred to as t throughout the paper, leading to tk channels in this first step. Next, a depth-wise convolution is performed using 3x3 kernels, followed by ReLU6 activation, achieving spatial filtering of the higher-dimensional tensor. Finally, the spatially-filtered feature map is projected back to a low-dimensional subspace using another point-wise convolution. The projection alone results in loss of information, so, intuitively, it’s important that the activation function in the last step be linear activation (see below for an empirical justification). When the initial and final feature maps are of the same dimensions (when the depth-wise convolution stride equals one and input and output channels are equal), a residual connection is added to aid gradient flow during backpropagation. Note that the final two steps are essentially a depth-wise separable convolution with the requirement that there be dimensionality reduction.

## Model architecture

Now we describe our architecture in detail. As discussed in the previous section the basic building block is a bottleneck depth-separable convolution with residuals. The architecture of MobileNetV2 contains the initial fully convolution layer with 32 filters, followed by 19 residual bottleneck layers. We use ReLU6 as the non-linearity because of its robustness when used with low-precision computation. We always use kernel size 3 3 as is standard for modern networks, and utilize dropout and batch normalization during training. 

With the exception of the first layer, we use constant expansion rate throughout the network. In our experiments we find that expansion rates between 5 and 10 result in nearly identical performance curves, with smaller networks being better off with slightly smaller expansion rates and larger networks having slightly better performance with larger expansion rates.

## Training the model

MobileNetV2 is trained on ImageNet data. ImageNet is an extensive image dataset project widely used in the field of machine learning. It contains a large collection of images across various categories, which can be used to teach networks to recognize and classify objects. MobileNetV2 is trained on 1,000 categories.

## Transfer learning and fine-tuning

The intuition behind transfer learning for image classification is that if a model is trained on a large and general enough dataset, this model will effectively serve as a generic model of the visual world. You can then take advantage of these learned feature maps without having to start from scratch by training a large model on a large dataset.

In this notebook, you will try two ways to customize a pretrained model:
- Feature Extraction: Use the representations learned by a previous network to extract meaningful features from new samples. You simply add a new classifier, which will be trained from scratch, on top of the pretrained model so that you can repurpose the feature maps learned previously for the dataset. You do not need to (re)train the entire model. The base convolutional network already contains features that are generically useful for classifying pictures. However, the final, classification part of the pretrained model is specific to the original classification task, and subsequently specific to the set of classes on which the model was trained.

In this step, you will freeze the convolutional base created from the previous step and to use as a feature extractor. Additionally, you add a classifier on top of it and train the top-level classifier. It is important to freeze the convolutional base before you compile and train the model. Freezing (by setting layer.trainable = False) prevents the weights in a given layer from being updated during training. MobileNet V2 has many layers, so setting the entire model's trainable flag to False will freeze all of them.

Important note about BatchNormalization-layers: Many models contain tf.keras.layers.BatchNormalization layers. This layer is a special case and precautions should be taken in the context of fine-tuning, as shown later in this tutorial. When you set layer.trainable = False, the BatchNormalization layer will run in inference mode, and will not update its mean and variance statistics. When you unfreeze a model that contains BatchNormalization layers in order to do fine-tuning, you should keep the BatchNormalization layers in inference mode by passing training = False when calling the base model. Otherwise, the updates applied to the non-trainable weights will destroy what the model has learned.

To generate predictions from the block of features, average over the spatial 5x5 spatial locations, using a tf.keras.layers.GlobalAveragePooling2D layer to convert the features to a single 1280-element vector per image. Apply a tf.keras.layers.Dense layer to convert these features into a single prediction per image. You don't need an activation function here because this prediction will be treated as a logit, or a raw prediction value. Positive numbers predict class 1, negative numbers predict class 0.

- Fine-Tuning: Unfreeze a few of the top layers of a frozen model base and jointly train both the newly-added classifier layers and the last layers of the base model. This allows us to "fine-tune" the higher-order feature representations in the base model in order to make them more relevant for the specific task.

In the feature extraction experiment, you were only training a few layers on top of an MobileNetV2 base model. The weights of the pre-trained network were not updated during training.

One way to increase performance even further is to train (or "fine-tune") the weights of the top layers of the pre-trained model alongside the training of the classifier you added. The training process will force the weights to be tuned from generic feature maps to features associated specifically with the dataset.

Also, you should try to fine-tune a small number of top layers rather than the whole MobileNet model. In most convolutional networks, the higher up a layer is, the more specialized it is. The first few layers learn very simple and generic features that generalize to almost all types of images. As you go higher up, the features are increasingly more specific to the dataset on which the model was trained. The goal of fine-tuning is to adapt these specialized features to work with the new dataset, rather than overwrite the generic learning.

  

## Data augmentation

When you don't have a large image dataset, it's a good practice to artificially introduce sample diversity by applying random, yet realistic, transformations to the training images, such as rotation and horizontal flipping. This helps expose the model to different aspects of the training data and reduce overfitting.

This model expects pixel values in [-1, 1], but at this point, the pixel values in your images are in [0, 255]



