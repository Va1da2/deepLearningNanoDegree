"""
We have to set name for the variables by hand, otherwise tensorflow will set them automatically
which means that order of specified variables will matter and that might cause issues.
"""
import tensorflow as tf

tf.reset_default_graph()

save_file = 'model_names.ckpt'

# Two Tensor Variables: weights and bias
weights = tf.Variable(tf.truncated_normal([2, 3]), name='weights_0')
bias = tf.Variable(tf.truncated_normal([3]), name='bias_0')

saver = tf.train.Saver()

# Print the name of Weights and Bias
print('Save Weights: {}'.format(weights.name))
print('Save Bias: {}'.format(bias.name))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.save(sess, save_file)

# Remove the previous weights and bias
tf.reset_default_graph()

# Two Variables: weights and bias
bias = tf.Variable(tf.truncated_normal([3]), name='bias_0')
weights = tf.Variable(tf.truncated_normal([2, 3]) ,name='weights_0')

saver = tf.train.Saver()

# Print the name of Weights and Bias
print('Load Weights: {}'.format(weights.name))
print('Load Bias: {}'.format(bias.name))

loading_name = '/Users/vaidasarmonas/Documents/Udacity/deepLearningNanoDegree/week6/lesson18/model_names.ckpt'

with tf.Session() as sess:
    # Load the weights and bias - No Error
    saver.restore(sess, loading_name)

print('Loaded Weights and Bias successfully.')