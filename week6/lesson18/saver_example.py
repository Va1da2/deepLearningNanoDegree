import tensorflow as tf

# The file path to save the data
save_file = 'model.ckpt'
load_file = '/Users/vaidasarmonas/Documents/Udacity/deepLearningNanoDegree/week6/lesson18/model.ckpt'

# Two tensor variables: weights and biases
weights = tf.Variable(tf.truncated_normal([2, 3]))
bias = tf.Variable(tf.truncated_normal([3]))

# Class used to save and/or restore Tensor variables
saver = tf.train.Saver()

with tf.Session() as sess:
	# Initialize all variables
	sess.run(tf.global_variables_initializer())

	# Show the values of weights and biases
	print('Weights: ')
	print(sess.run(weights))
	print('Bias:')
	print(sess.run(bias))

	saver.save(sess, save_file)

# Remove previous weights and biases
tf.reset_default_graph()

# Two variables - weights and biases
weights = tf.Variable(tf.truncated_normal([2, 3]))
bias = tf.Variable(tf.truncated_normal([3]))

# Class to save and load Tensor variables
saver = tf.train.Saver()

with tf.Session() as sess:
	# Loading saved variables
	saver.restore(sess, load_file)

	# Show values
	print('Weights: ')
	print(sess.run(weights))
	print('Bias:')
	print(sess.run(bias))


