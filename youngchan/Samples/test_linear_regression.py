"""
DrawML
Linear regression template file
Templates will be changed by appropriate value

There is some issues in input data
and maybe save model
"""
import tensorflow as tf
import numpy as np

training_epoch = 1024


def load_input():
	raw_data = np.loadtxt('linear_regression_data.txt', unpack=True, dtype='float32')
	raw_data = raw_data.T
	row = len(raw_data)
	col = len(raw_data[0])



	x_data = np.ones([row, col])
	data = np.ones([row, col+1])
	x_data[:, 1:col] = raw_data[:, 0:col-1]
	data[:, 1:col+1] = raw_data[:, 0:col]
	y_data = raw_data[:, col-1:col]

	np.savetxt('/Users/chan/test/data.txt', data)
	x_data = np.loadtxt('/Users/chan/test/x_data.txt', unpack=True)
	y_data = np.loadtxt('/Users/chan/test/y_data.txt', unpack=True)

	return x_data, y_data, x_data, y_data, x_data, y_data


def load_train_data():
	data = {X: x_train, Y: y_train}
	return data


def make_model(X, W):
	reg_enable = True
	reg_lambda = 0.001
	model = tf.matmul(X, tf.transpose(W))
	if reg_enable is True:
		model += reg_lambda * tf.reduce_sum(tf.square(W))
	return model


def cost_function(hypothesis, Y):
	return tf.reduce_mean(tf.square(hypothesis - Y))


def make_optimizer():
	optimizer_module = tf.train
	optimizer_name   = 'GradientDescentOptimizer'
	optimizer_params = {'learning_rate': 0.01}
	return getattr(optimizer_module, optimizer_name)(**optimizer_params)


def init_weights():
	weight_init_module = tf.random_uniform
	weight_params      = {'minval': -1.0, 'maxval': 1.0}

	weight_params['shape'] = [1, len(x_train[0])]

	weight = tf.Variable(weight_init_module(**weight_params))
	return weight


def save_model(sess, path):
	# save model
	saver = tf.train.Saver()
	saver.save(sess, path)


def restore_model(sess, path):
	saver = tf.train.Saver()
	saver.restore(sess, path)


x_train, y_train, x_valid, y_valid, x_test, y_test = load_input()

X = tf.placeholder(tf.float32)     # X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32)     # Y = tf.placeholder(tf.float32, [None, 10])

W = init_weights()
hypothesis = make_model(X, W)

cost = cost_function(hypothesis, Y)
optimizer = make_optimizer()
train = optimizer.minimize(cost)

with tf.Session() as sess:
	init = tf.initialize_all_variables()
	sess.run(init)
	train_data = load_train_data()
	for _ in range(training_epoch):
		sess.run(train, feed_dict=train_data)
		# some logging codes will be added...
	save_model(sess, "/Users/chan/test/trained")
	restore_model(sess, "/Users/chan/test/trained")

	print(sess.run(W))
