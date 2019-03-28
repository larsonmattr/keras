# Keras Introduction and Guide #

## Author Matt Larson 2019

Keras is a high-level python API meant for ease of use and portability that can use Tensorflow, CNTK, and frameworks. Keras models can be saved and then brought into Tensorflow services, DL4J, Apple CoreML, etc. Keras allows for a lot of flexibility what actually does the training and then how the trained model can later be used.

## Sequential model ##

The Sequential model is a linear stack of layers, and you can create a model just by providing a list of layers to the constructor.

```
from keras.models import Sequential
from keras.layers import Dense, Activation

model = Sequential([
    Dense(32, input_shape=(784,)),
    Activation('relu'),
    Dense(10),
    Activation('softmax'),
	])
```

Also possible to add layers via .add() method.

## Functional API ##
- Functional model allows more complex architectures.
- You can reuse trained models and treat any model as if it were a layer by calling it on a tensor.
- When you call a model like this you are reuse the architecture and the weights.
- When creating layers, you tell how the inputs / outputs connect between the layers.
- Can share a layer across different inputs (reuse or improve training?)
- When calling a layer on an input your are creating a new tensor (output of the layer). You are also adding a “node” to a layer that links input tensor to output tensor.
- Shared vision model might be a useful unit for GAN models.

Example:
```
from keras.layers import Input, Dense
from keras.models import Model

# This returns a tensor
inputs = Input(shape=(784,))

# a layer instance is callable on a tensor, and returns a tensor
x = Dense(64, activation='relu')(inputs)
x = Dense(64, activation='relu')(x)
predictions = Dense(10, activation='softmax')(x)

# This creates a model that includes
# the Input layer and three Dense layers
model = Model(inputs=inputs, outputs=predictions)
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(data, labels)  # starts training
```

## Layers ##
Models are built up from layers. Keras layers have common methods `.get_weights()` that returns the weights of a layer as a list of Numpy arrays, `.set_weights(weights)` sets weights of a layer from a list of Numpy arrays with same shapes as the output of get_weights, `.get_config()` returns a dictionary containing the configuration of the layer. A layer can be reinstantiated from its config.

## Core Layers ##

### Dense ###
A densely-connected NN layer - [Documentation](https://keras.io/layers/core/#dense)
```
keras.layers.Dense(units, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
```

### Activation ###
Applies activation function to an output.

### Dropout ###
Dropout can help prevent overfitting and provides a way of approximately combining exponentially many different neural network architectures efficiently. [Dropout: A Simple Way to Prevent Neural Networks from Overfitting by N. Srivastava](http://www.jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf). "dropout" refers to dropping out units in a neural network, by temporarily removing it from the network along with all its incoming and out-going connections.

### Flatten ###
Flatten the input. (More information needed)

### Input ###
(More information needed)

### Reshape ###
Reshape the output to a certain shape. This could be useful to change shape from a 1D tensor layer to a 2D tensor layer.

### Permute ###
Permute the dimensions of the input to a given pattern. Useful for connecting RNNs and convnets.

## Other Layers ##

## Convolutional Layers ##
Convolutional layers apply a convolution operation on input and pass results to the next layer. (What is a convolution?)

### Conv1D ###
1D convolution layer is a temporal convolution.

### Conv2D ###
2D convolution layer is spatial convolution over images.

### Conv3D ###
3D convolution layer is spatial convolution over volumes.

### Conv2DTranspose and Conv3DTranspose ###
Transposed convolution layer is also called Deconvolution.


## Pooling ##
Pooling layers reduce dimensions of data by combining outputs of neuron clusters at one layer into a single neuron of the next layer. Local pooling will combine small clusters like 2x2. Global pooling acts on all neurons of the convolutional layer. Max pooling uses the maximum value from each of a cluster of neurons at the prior layer, average pooling uses the average value.

- MaxPooling1D, MaxPooling2D, and MaxPooling3D
- AveragePooling1D, AveragePooling2D, AveragePooling3D.

## Fully connected ##
Fully connected layers connect every neuron in one layer to every neuron in another layer.

### ReLU layers in Neural Networks ###
ReLU function is f(x) = max(0, x). ReLUs can speed up training where the gradient computation is 0 or 1 depending on the sign of x. It comes at a cost with the 0 gradient on the left-hand side creating "dead neurons".

More explanations about ReLU functions ["StackOverflow"](https://stats.stackexchange.com/questions/226923/why-do-we-use-relu-in-neural-networks-and-how-do-we-use-it)

# Architectures

## Image recognition

## Autoencoders for image generation from parameters
 "Autoencoding" is a data compression and decompression function that is lossy, data-specific, and can be learned automatically from examples.

 Autoencoders are often described in CNN tutorials but may not be the best way to generate images. Basically it may be training a network to encode the minimal number of parameters to recreate an image but is that useful?

## Inception
Using a CNN to understand image features and generate an image [Link to example in github](https://github.com/jcjohnson/cnn-vis). One way to understand a CNN is to try to find an image that causes a particular neuron to fire strongly. The trick here is to initialize an image with random noise, propagate the image forward through a network to compute activation of a target neuron, then propagate the activation of the neuron backward through the network to compute an update direction for the image. The process is repeated until it converges. This can be used to see activations of individual neurons, generate images of particular object classes, invert CNN features, and generate images to fool CNNs.

- Layer amplification (TODO)
- Multiscale, high-res images (TODO)
- Non-random initialization (TODO)

## Generative adversarial networks for image generation

# Dictionary
- convolutional neural network
- layer
- "latent image space" : when images are generated by a random set of parameters, we are exploring the space of generatable images.

# References
- ["How to do novelty detection in Keras"](https://www.dlology.com/blog/how-to-do-novelty-detection-in-keras-with-generative-adversarial-network/)
- ["GAN by example using Keras on Tensorflow"](https://towardsdatascience.com/gan-by-example-using-keras-on-tensorflow-backend-1a6d515a60d0)
- ["Creating AutoEncoders in Keras and generating images with VAE"](https://github.com/chaitanya100100/VAE-for-Image-Generation)
- ["Generative Adverserial Networks in Deblurring"](https://blog.sicara.com/keras-generative-adversarial-networks-image-deblurring-45e3ab6977b5)
- ["Building Autoencoders in Keras"](https://blog.keras.io/building-autoencoders-in-keras.html)
