import tensorflow as tf

# This is demonstration of how to set up convolution layer in tf 
# as presented in lessons

# Output depth
k_output = 64

# Image properties
image_width = 10
image_higth = 10
color_channels = 3

# Convolution filter
filter_size_width = 5
filter_size_hight = 5

# Input / Image
input = tf.placeholder(tf.float32, shape=[None, image_width, image_higth, color_channels])

# Weight and bias
weight = tf.Variable(tf.truncated_normal([filter_size_width, filter_size_hight, color_channels, k_output]))
bias = tf.Variable(tf.zeros(k_output))

# Apply convolution
conv_layer = tf.nn.conv2d(input, weight, strides=[1, 2, 2, 1], padding='SAME')
# Add bias
conv_layer = tf.nn.bias_add(conv_layer, bias)
# Apply activation
conv_layer = tf.nn.relu(conv_layer)

### Explanations
# conv2d function is tf convolution function moves arbitrary filters that mix channels together (https://www.tensorflow.org/api_guides/python/nn#Convolution)
# in our case weight specifies filter (size of it!). Strides are numbers representing stride for every dimension! 
# TensorFlow uses a stride for each input dimension, [batch, input_height, input_width, input_channels] 
# The tf.nn.bias_add() function adds a 1-d bias to the last dimension in a matrix.

# Apply max pooling
conv_layer = tf.nn.max_pool(conv_layer, ksize=[1, 2, 2, 1], stride=[1, 2, 2, 1], padding='SAME')

### Explanations
# The tf.nn.max_pool() function performs max pooling with the ksize parameter as the size of the filter 
# and the strides parameter as the length of the stride. 2x2 filters with a stride of 2x2 are common in practice.



