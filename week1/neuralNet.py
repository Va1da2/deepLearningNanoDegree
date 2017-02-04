import numpy as np

class NeuralNetwork():
	def __init__(self):
		# Seed the random number generator so it generates the same numbers
		# every time the program runs
		np.random.seed(1)

		# We model a single neuron with 3 input connections and 1 output connection.
		# We assigne a  random weights to a 3 x 1 matrix, with values in the range -1 to 1
		# and mean 0
		self.synaptic_weights = 2 * np.random.random((3,1)) - 1

	# The sigmoid function, which describes and 'S' shaped curve
	# We pass the weighted sum of the inputs through this function
	# to normalize them between 0 and 1
	def __sigmoid(self, x):
		return 1 / (1 + np.exp(-x))

	# Gradient of the sigmoid curve
	def __sigmoid_derivative(self, x):
		return x * (1 - x)

	def train(self, train_set_inputs, training_set_outputs, number_of_training_iterations):
		for iteration in range(number_of_training_iterations):
			# Pass the training set through our neural network
			output = self.predict(training_set_inputs)

			# Calculate the error
			error = training_set_outputs - output

			# Multiply the error by the input and again by the gradient of the sigmoid curve
			adjustment = np.dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))

			self.synaptic_weights += adjustment

	def predict(self, inputs):
		# Pass inputs through our neural network (our single neuron)
		return self.__sigmoid(np.dot(inputs, self.synaptic_weights))


if __name__ == '__main__':

	# initialize a single layer neural net
	neural_network = NeuralNetwork()

	print('random starting synaptic weights')
	print(neural_network.synaptic_weights)

	# The training set. We have 4 examples, each consisting of 3 input values
	# and 1 output value 
	training_set_inputs = np.array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
	training_set_outputs = np.array([[0, 1, 1, 0]]).T

	# Training the neural network using the training set.
	# Do it 10,000 times and make small adjustments each time
	neural_network.train(training_set_inputs, training_set_outputs, 10000)

	print('New synaptic weights after training: ')
	print(neural_network.synaptic_weights)

	# Test the neural network
	print('predicting:')
	print(neural_network.predict(np.array([1, 0, 0])))

