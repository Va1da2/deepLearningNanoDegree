# This is only walkthrough the code provided in Lesson 18.

import math
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# Getting data
mnist = input_data.read_data_sets(".", one_hot=True, reshape=False)

# Parameters
learning_rate = 0.001
training_epochs = 20
batch_size = 128
display_step = 1

n_inputs = 784	# MNIST data input (image shape 28 * 28)
n_classes = 10	# MNIST total classes (0-9 digits)

n_hidden_layer = 256	# layer number of features

# Store layers weights and biases in dictionaries
weights = {
	'hidden_layer': tf.Variable(tf.truncated_normal([n_inputs, n_hidden_layer], stddev=1./math.sqrt(n_inputs))),
	'out': tf.Variable(tf.truncated_normal([n_hidden_layer, n_classes], stddev=1./math.sqrt(n_hidden_layer)))
}

biases = {
	'hidden_layer': tf.Variable(tf.zeros([n_hidden_layer])),
	'out': tf.Variable(tf.zeros([n_classes]))
}

# tf Graph input 
x = tf.placeholder("float", [None, 28, 28, 1])
y = tf.placeholder("float", [None, n_classes])

x_flat = tf.reshape(x, [-1, n_inputs])

# Hidden layer with ReLU activation
layer_1 = tf.add(tf.matmul(x_flat, weights['hidden_layer']), biases['hidden_layer'])
layer_1 = tf.nn.relu(layer_1)

# Output layer with linear activation
logits = tf.add(tf.matmul(layer_1, weights['out']), biases['out'])

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializeing variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
	sess.run(init)

	# Training cycle
	for epoch in range(training_epochs):
		total_batch = int(mnist.train.num_examples/batch_size)
		# Loop over all batches
		for i in range(total_batch):
			batch_x, batch_y = mnist.train.next_batch(batch_size)
			# Run optimization and op (backprop) and cost op (to get loss value)
			sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
			batch_cost = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
		print('Cost for epoch {0} is {1}'.format(epoch, batch_cost))



