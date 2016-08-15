#-*- coding: utf-8 -*-
import tensorflow as tf
import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

alpha = 0.001
training_epoch = 15
batch_size = 100
display_step = 1
"""
X = tf.placeholder("float", [None, 784])
Y = tf.placeholder("float", [None, 10])

W1 = tf.Variable(tf.random_normal([784, 256]))
W2 = tf.Variable(tf.random_normal([256, 256]))
W3 = tf.Variable(tf.random_normal([256, 10]))

B1 = tf.Variable(tf.random_normal([256]))
B2 = tf.Variable(tf.random_normal([256]))
B3 = tf.Variable(tf.random_normal([10]))

L1 = tf.nn.relu(tf.add(tf.matmul(X, W1), B1))
L2 = tf.nn.relu(tf.add(tf.matmul(L1, W2), B2))
h = tf.add(tf.matmul(L2, W3), B3)        #without softmax

# softmax_cross_entropy_((with_logits))는 h가 softmax를 취하지 않는 값이어도 값계산을 잘 해주는 api
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(h, Y))
optimizer = tf.train.AdamOptimizer(learning_rate=alpha).minimize(cost)  # Adam : 현존 최고의 옵티마이

init = tf.initialize_all_variables()

sess = tf.Session()
sess.as_default()
sess.run(init)
"""


def load_train_data():
	data = {X: x_train, Y: y_train}
	return data


def make_activation(src: str):
	src_split = src.split('.')
	module = None
	for i in range(len(src_split)):
		if i == 0:
			module = globals()[src_split[i]]
		else:
			module = getattr(module, src_split[i])
	return module


def make_model(X, W, B):
	layers = [X];
	activation_functions = ['tf.nn.relu', 'tf.nn.relu', 'tf.nn.relu']
	for i in range(len(W)-1):
		activ_func = make_activation(activation_functions[i])
		layer_temp = tf.add(tf.matmul(layers[i], W[i]), B[i])
		layer = activ_func(layer_temp)
		layers.append(layer)

	size = len(W)
	model = tf.add(tf.matmul(layers[size-1], W[size-1]), B[size-1])

	""" next 4 lines are for regularization.
		And They have to change
	reg_enable = True
	reg_lambda = 0.0
	if reg_enable is True:
		 model += (reg_lambda / 2) * tf.reduce_mean(tf.reduce_sum(tf.square(W)))
	"""
	return model


def cost_function(hypothesis, Y):
	return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(hypothesis, Y))

def make_optimizer():
	optimizer_module = tf.train
	optimizer_name   = 'GradientDescentOptimizer'
	optimizer_params = {'learning_rate': 0.01}
	return getattr(optimizer_module, optimizer_name)(**optimizer_params)


def init_weights():
	W = []
	B = []
	weight_init_module = tf.random_uniform
	weight_params = {'minval': -1.0, 'maxval': 1.0}
	layer_size = 3
	input_shape = [784, 256, 256]
	output_shape = [256, 256, 10]

	weight_params['shape'] = [input_shape[0], output_shape[0]]
	W.append(tf.Variable(weight_init_module(**weight_params)))
	weight_params['shape'] = [output_shape[0]]
	B.append(tf.Variable(weight_init_module(**weight_params)))

	weight_params['shape'] = [input_shape[1], output_shape[1]]
	W.append(tf.Variable(weight_init_module(**weight_params)))
	weight_params['shape'] = [output_shape[1]]
	B.append(tf.Variable(weight_init_module(**weight_params)))

	weight_params['shape'] = [input_shape[2], output_shape[2]]
	W.append(tf.Variable(weight_init_module(**weight_params)))
	weight_params['shape'] = [output_shape[2]]
	B.append(tf.Variable(weight_init_module(**weight_params)))

	return W, B


def save_model():
	# save model
	save_path = "/" """" *This will be filled at .exe step """
	saver = tf.train.Saver()
	saver.save(save_path=save_path)


x_train = trX
y_train = trY

X = tf.placeholder(tf.float32, [None, len(x_train[0])])
Y = tf.placeholder(tf.float32, [None, len(y_train[0])])

W, B = init_weights()
hypothesis = make_model(X, W, B)

cost = cost_function(hypothesis, Y)
optimizer = make_optimizer()
train = optimizer.minimize(cost)

with tf.Session() as sess:
	init = tf.initialize_all_variables()
	sess.run(init)
	train_data = load_train_data()

	# training cycle
	for epoch in range(training_epoch):
		avg_cost = 0.0
		total_batch = int(mnist.train.num_examples/batch_size)
		# loop over all batch
		for i in range(total_batch):
			batch_xs, batch_ys = mnist.train.next_batch(batch_size)
			# fit training using batch data
			sess.run(train, feed_dict={X : batch_xs, Y : batch_ys})
			# compute average loss
			avg_cost += sess.run(cost, feed_dict={X : batch_xs, Y : batch_ys}) / total_batch

		# display logs per epoch step
		if epoch % display_step == 0:
			print("Epoch : "," %04d" % (epoch+1), "cost = ", "{:.9f}".format(avg_cost))

	print(" Optimization Finished" )

	correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
	print("Accuracy : ", accuracy.eval(session=sess, feed_dict={X : mnist.test.images, Y : mnist.test.labels}))
