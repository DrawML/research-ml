#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import input_data

batch_size = 128
test_size = 256
training_epoch = 1024
dropout_conv   = 0.8
dropout_hidden = 0.8
x_vertical     = 28
x_horizontal   = 28
y_size         = 10


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
trX = trX.reshape(-1, 28, 28, 1)  # 28x28x1 input img
teX = teX.reshape(-1, 28, 28, 1)  # 28x28x1 input img



def load_train_data():
	data = {X: x_train, Y: y_train}
	return data


def no_module(param):
	return param

def make_module(src: str):
	module = no_module
	if src == 'none':
		return module

	src_split = src.split('.')
	for i in range(len(src_split)):
		if i == 0:
			module = globals()[src_split[i]]
		else:
			module = getattr(module, src_split[i])
	return module


def make_model(X, W):
	prev_layer = X;


	activ_func = tf.nn.relu
	pooling = tf.nn.max_pool

	l = tf.nn.conv2d(prev_layer, W['w1'],
	             strides=[1, 1, 1, 1],
	             padding='SAME')
	l = activ_func(l)
	l = pooling(l, ksize=[1, 2, 2, 1],
	            strides=[1, 2, 2, 1],
	            padding='SAME')
	l = tf.nn.dropout(l, p_keep_conv)
	prev_layer = l

	activ_func = tf.nn.relu
	pooling = tf.nn.max_pool

	l = tf.nn.conv2d(prev_layer, W['w2'],
	             strides=[1, 1, 1, 1],
	             padding='SAME')
	l = activ_func(l)
	l = pooling(l, ksize=[1, 2, 2, 1],
	            strides=[1, 2, 2, 1],
	            padding='SAME')
	l = tf.nn.dropout(l, p_keep_conv)
	prev_layer = l

	activ_func = tf.nn.relu
	pooling = tf.nn.max_pool

	l = tf.nn.conv2d(prev_layer, W['w3'],
	             strides=[1, 1, 1, 1],
	             padding='SAME')
	l = activ_func(l)
	l = pooling(l, ksize=[1, 2, 2, 1],
	            strides=[1, 2, 2, 1],
	            padding='SAME')

	l = tf.reshape(l, [-1, W['w4'].get_shape().as_list()[0]])
	l = tf.nn.dropout(l, p_keep_conv)
	prev_layer = l

	weight = W['w4']
	# prev_layer = tf.reshape(prev_layer, [-1, weight.get_shape().as_list()[0]])
	activ_func = tf.nn.relu
	l = activ_func(tf.matmul(prev_layer, weight))
	l = tf.nn.dropout(l, p_keep_hidden)
	prev_layer = l

	weight = W['w5']
	# prev_layer = tf.reshape(prev_layer, [-1, weight.get_shape().as_list()[0]])
	activ_func = no_module
	l = activ_func(tf.matmul(prev_layer, weight))
	prev_layer = l

	model = prev_layer
	return model


def cost_function(hypothesis, Y):
	return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(hypothesis, Y))

def make_optimizer():
	optimizer_module = tf.train
	optimizer_name   = 'RMSPropOptimizer'
	optimizer_params = {'learning_rate': 0.001}
	return getattr(optimizer_module, optimizer_name)(**optimizer_params)


def init_weights():
	W = {}
	weight_init_module = tf.random_uniform
	weight_params = {'minval': -1.0, 'maxval': 1.0}
	layers = [{'input_x': 3, 'type': 'conv', 'pooling': 'tf.nn.max_pool', 'activ_func': 'tf.nn.relu', 'input_y': 3, 'pooling_strides_h': 2, 'activ_padding': "'SAME'", 'output': 32, 'num': 1, 'input_z': 1, 'pooling_padding': "'SAME'", 'activ_strides_h': 1, 'pooling_strides_v': 2, 'activ_strides_v': 1}, {'input_x': 3, 'type': 'conv', 'pooling': 'tf.nn.max_pool', 'activ_func': 'tf.nn.relu', 'input_y': 3, 'pooling_strides_h': 2, 'activ_padding': "'SAME'", 'output': 64, 'num': 2, 'input_z': 32, 'pooling_padding': "'SAME'", 'activ_strides_h': 1, 'pooling_strides_v': 2, 'activ_strides_v': 1}, {'input_x': 3, 'type': 'conv', 'pooling': 'tf.nn.max_pool', 'activ_func': 'tf.nn.relu', 'input_y': 3, 'pooling_strides_h': 2, 'activ_padding': "'SAME'", 'output': 128, 'num': 3, 'input_z': 64, 'pooling_padding': "'SAME'", 'activ_strides_h': 1, 'pooling_strides_v': 2, 'activ_strides_v': 1}, {'output': 625, 'type': 'none', 'num': 4, 'input': 2048, 'activ_func': 'tf.nn.relu'}, {'output': 10, 'type': 'out', 'num': 5, 'input': 625, 'activ_func': 'no_module'}]

	shape = [3, 3, 1, 32]
	weight_params['shape'] = shape
	# weight_params['stddev'] = 0.01
	W['w1'] = tf.Variable(weight_init_module(**weight_params))

	shape = [3, 3, 32, 64]
	weight_params['shape'] = shape
	# weight_params['stddev'] = 0.01
	W['w2'] = tf.Variable(weight_init_module(**weight_params))

	shape = [3, 3, 64, 128]
	weight_params['shape'] = shape
	# weight_params['stddev'] = 0.01
	W['w3'] = tf.Variable(weight_init_module(**weight_params))

	shape = [2048, 625]
	weight_params['shape'] = shape
	# weight_params['stddev'] = 0.01
	W['w4'] = tf.Variable(weight_init_module(**weight_params))

	shape = [625, 10]
	weight_params['shape'] = shape
	# weight_params['stddev'] = 0.01
	W['w5'] = tf.Variable(weight_init_module(**weight_params))

	return W


def save_model():
	# save model
	save_path = "/" """" *This will be filled at .exe step """
	saver = tf.train.Saver()
	saver.save(save_path=save_path)

x_train = trX
y_train = trY

X = tf.placeholder(tf.float32, [None, x_vertical, x_horizontal, 1])
Y = tf.placeholder(tf.float32, [None, len(y_train[0])])
p_keep_conv = tf.placeholder(tf.float32)
p_keep_hidden = tf.placeholder(tf.float32)

W = init_weights()
hypothesis = make_model(X, W)

cost = cost_function(hypothesis, Y)
optimizer = make_optimizer()
train = optimizer.minimize(cost)
predict = tf.argmax(hypothesis, 1)

# Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize all variables
    tf.initialize_all_variables().run()

    for i in range(100):
        training_batch = zip(range(0, len(trX), batch_size),
                             range(batch_size, len(trX), batch_size))
        for start, end in training_batch:
            sess.run(train, feed_dict={X: trX[start:end], Y: trY[start:end],
                                          p_keep_conv: 0.8, p_keep_hidden: 0.5})

        test_indices = np.arange(len(teX)) # Get A Test Batch
        np.random.shuffle(test_indices)
        test_indices = test_indices[0:test_size]

        print(i, np.mean(np.argmax(teY[test_indices], axis=1) ==
                         sess.run(predict, feed_dict={X: teX[test_indices],
                                                         Y: teY[test_indices],
                                                         p_keep_conv: 1.0,
                                                         p_keep_hidden: 1.0})))
"""
0 0.26171875
1 0.09375
2 0.09375
3 0.0625
4 0.1328125
5 0.15234375
6 0.1875
7 0.12890625
8 0.125



"""