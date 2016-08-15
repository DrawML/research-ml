"""
DrawML
Logistic regression template file
Templates will be changed by appropriate value

There is some issues in input data
and maybe save model
"""

import tensorflow as tf
import numpy as np
import mnist_input
training_epoch = 1024


def load_input():
	mnist = mnist_input.read_data_sets("MNIST_data/", one_hot=True)

	raw_data, y_data, x_testt, y_test = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
	row = len(raw_data)
	col = len(raw_data[0])
	x_data = np.ones([row, col + 1])
	x_data[:, 1:col + 1] = raw_data[:, 0:col]

	row = len(x_testt)
	col = len(x_testt[0])
	x_test = np.ones([row, col + 1])
	x_test[:, 1:col + 1] = x_testt[:, 0:col]

	return x_data, y_data, x_data, y_data, x_test, y_test


def load_train_data():
	data = {X: x_train, Y: y_train}
	return data


def make_model(X, W):
	reg_enable = True
	reg_lambda = 0.0
	model = tf.matmul(X, tf.transpose(W))
	if reg_enable is True:
		model += (reg_lambda / 2) * tf.reduce_mean(tf.reduce_sum(tf.square(W)))
	return model


def cost_function(hypothesis, Y):
	return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(hypothesis, Y))

def make_optimizer():
	optimizer_module = tf.train
	optimizer_name   = 'GradientDescentOptimizer'
	optimizer_params = {'learning_rate': 0.01}
	return getattr(optimizer_module, optimizer_name)(**optimizer_params)


def init_weights():
	weight_init_module = tf.random_uniform
	weight_params      = {'minval': -1.0, 'maxval': 1.0}

	weight_params['shape'] = [len(y_train[0]), len(x_train[0])]

	weight = tf.Variable(weight_init_module(**weight_params))
	return weight


def save_model():
	# save model
	save_path = "/" """" *This will be filled at .exe step """
	saver = tf.train.Saver()
	saver.save(save_path=save_path)


x_train, y_train, x_valid, y_valid, x_test, y_test = load_input()

X = tf.placeholder(tf.float32, [None, len(x_train[0])])
Y = tf.placeholder(tf.float32, [None, len(y_train[0])])

W = init_weights()
hypothesis = make_model(X, W)

cost = cost_function(hypothesis, Y)
optimizer = make_optimizer()
train = optimizer.minimize(cost)
predict = tf.argmax(hypothesis, 1)


with tf.Session() as sess:
    # you need to initialize all variables
    tf.initialize_all_variables().run()

    for i in range(100):
        for start, end in zip(range(0, len(x_train), 128), range(128, len(x_train)+1, 128)):
            sess.run(train, feed_dict={X: x_train[start:end], Y: y_train[start:end]})
        print(i, np.mean(np.argmax(y_test, axis=1) ==
                         sess.run(predict, feed_dict={X: x_test, Y: y_test})))

"""
with tf.Session() as sess:
	init = tf.initialize_all_variables()
	sess.run(init)
	train_data = load_train_data()

	for step in range(60001):
		if step % 1000 == 0:
			print(step, sess.run(cost, feed_dict=train_data), sess.run(W))
		sess.run(train, feed_dict=train_data)
	print(sess.run(cost, feed_dict=train_data), sess.run(W))
"""