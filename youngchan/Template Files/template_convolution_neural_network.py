"""
DrawML
Neural Network template file

There is some issues in input data
and maybe save model
"""
import tensorflow as tf
import numpy as np

training_epoch = {{training_epoch}}
dropout_conv   = {{dropout_conv}}
dropout_hidden = {{dropout_hidden}}
x_vertical     = {{x_vertical}}
x_horizontal   = {{x_horizontal}}
y_size         = {{y_size}}


def load_input():
	raw_data = np.loadtxt('data.txt', unpack=True, dtype='float32')
	raw_data = raw_data.T
	row = len(raw_data)
	col = len(raw_data[0])

	x_data = np.ones([row, col])
	x_data[:, 1:col] = raw_data[:, 0:col - 1]
	y_data = raw_data[:, col - 1:col]

	return x_data, y_data, x_data, y_data, x_data, y_data


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
	prev_layer = X

	{% for layer in layers %}
	{% if layer.type == 'conv' %}

	activ_func = {{layer.activ_func}}
	pooling = {{layer.pooling}}

	l = tf.nn.conv2d(prev_layer, W['w{{layer.num}}'],
	             strides=[1, {{layer.activ_strides_v}}, {{layer.activ_strides_h}}, 1],
	             padding={{layer.activ_padding}})
	l = activ_func(l)
	l = pooling(l, ksize=[1, {{layer.pooling_strides_v}}, {{layer.pooling_strides_v}}, 1],
	            strides=[1, {{layer.pooling_strides_v}}, {{layer.pooling_strides_v}}, 1],
	            padding={{layer.pooling_padding}})
	l = tf.nn.dropout(l, p_keep_conv)
	prev_layer = l
	{% endif %}
	{% if layer.type == 'none' %}

	weight = W['w{{layer.num}}']
	prev_layer = tf.reshape(prev_layer, [-1, weight.get_shape().as_list()[0]])
	activ_func = {{layer.activ_func}}
	l = activ_func(tf.matmul(prev_layer, weight))
	l = tf.nn.dropout(l, p_keep_hidden)
	prev_layer = l
	{% endif %}
	{% if layer.type == 'out' %}

	weight = W['w{{layer.num}}']
	prev_layer = tf.reshape(prev_layer, [-1, weight.get_shape().as_list()[0]])
	activ_func = {{layer.activ_func}}
	l = activ_func(tf.matmul(prev_layer, weight))
	prev_layer = l
	{% endif %}
	{% endfor %}

	model = prev_layer
	return model


def cost_function(hypothesis, Y):
	return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(hypothesis, Y))

def make_optimizer():
	optimizer_module = {{optimizer_module}}
	optimizer_name   = {{optimizer_name}}
	optimizer_params = {{optimizer_params}}
	return getattr(optimizer_module, optimizer_name)(**optimizer_params)


def init_weights():
	W = {}
	weight_init_module = {{init_module}}
	weight_params = {{init_params}}
	layers = {{layers}}
	{% for layer in layers %}
	{% if layer.type == 'conv' %}

	shape = [{{layer.input_x}}, {{layer.input_y}}, {{layer.input_z}}, {{layer.output}}]
	weight_params['shape'] = shape
	W['w{{layer.num}}'] = tf.Variable(weight_init_module(**weight_params))
	{% endif %}
	{% if layer.type == 'none' or layer.type == 'out' %}

	shape = [{{layer.input}}, {{layer.output}}]
	weight_params['shape'] = shape
	W['w{{layer.num}}'] = tf.Variable(weight_init_module(**weight_params))
	{% endif %}
	{% endfor %}

	return W


def save_model():
	# save model
	save_path = "/" """" *This will be filled at .exe step """
	saver = tf.train.Saver()
	saver.save(save_path=save_path)


x_train, y_train, x_valid, y_valid, x_test, y_test = load_input()

x_train = x_train.reshape(-1, x_vertical, x_horizontal, 1)
x_test  = x_test.reshape(-1, x_vertical, x_horizontal, 1)
x_valid = x_valid.reshape(-1, x_vertical, x_horizontal, 1)

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

with tf.Session() as sess:
	init = tf.initialize_all_variables()
	sess.run(init)
	train_data = load_train_data()
	for _ in range(training_epoch):
		sess.run(train, feed_dict=train_data)
		# some logging codes will be added...
	save_model()
