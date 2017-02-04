import numpy as np

def compute_error_for_line_given_points(b, m, points):
	# initialize error at 0
	totalError = 0

	for i in range(len(points)):
		# get x and y values
		x = points[i, 0]
		y = points[i, 1]
		# get the difference y - y_hat, square it, add to the total
		totalError += (y - (m * x + b)) ** 2

	return totalError / float(len(points))

def gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations):
	b = initial_b
	m = initial_m

	# gradient descent:
	for i in range(num_iterations):
		# update b and m with the new, more accurate b and m by performing 
		# this gradient step
		b, m = step_gradient(b, m, np.array(points), learning_rate)

	return [b, m]

def step_gradient(current_b, current_m, points, learning_rate):
	# starting points of gradients
	b_gradient = 0
	m_gradient = 0
	N = float(len(points))

	for i in range(len(points)):
		x = points[i, 0]
		y = points[i, 1] 
		# direction with respect to b and m
		# computing partial derivatives of our error function
		b_gradient += -(2/N) * (y - ((current_m * x) + current_b))
		m_gradient += -(2/N) * x * (y - ((current_m * x) + current_b))

	# update our b and m values using partial derivatives
	new_b = current_b - (learning_rate * b_gradient)
	new_m = current_m - (learning_rate * m_gradient)

	return [new_b, new_m]


def run():

	# Step 1: collect data:
	points = np.genfromtxt('data.csv', delimiter=',')

	# Step 2: define hyper parameters:
	learning_rate = 0.0001
	initial_b = 0
	initial_m = 0
	num_iterations = 1000

	# Step 3: train our model:
	print("starting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m, compute_error_for_line_given_points(initial_b, initial_m, points)))

	[b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)

	print("ending point at b = {0}, m = {1}, error = {2}".format(b, m, compute_error_for_line_given_points(b, m, points)))

if __name__ == '__main__':
	run()