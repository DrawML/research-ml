"""
DrawML
Linear regression template file
Templates will be changed by appropriate value

There is some issues in input data
and maybe save model
"""
import tensorflow as tf
import numpy as np

training_epoch = {{training_epoch}}
x_data = {{x_data}}
y_data = {{y_data}}


def load_input():
	return x_data, y_data, x_data, y_data, x_data, y_data


def load_train_data():
    data = {X: x_train, Y: y_train}
    return data


def make_model(X, W, b):
    return tf.add(tf.matmul(X, W), b)


def cost_function():
	return tf.reduce_mean(tf.square(hypothesis - Y))


def make_optimizer():
    optimizer_module = {{optimizer_module}}
    optimizer_name   = {{optimizer_name}}
    optimizer_params = {{optimizer_params}}
    return getattr(optimizer_module, optimizer_name)(**optimizer_params)


def init_weights():
    weight_init_module = {{weight_init_module}}
    weight_params      = {{weight_params}}
    bias_init_module = {{bias_init_module}}
    bias_params      = {{bias_params}}

    weight = tf.Variable(weight_init_module(**weight_params))
    bias   = tf.Variable(bias_init_module(**bias_params))
    return weight, bias


def save_model():
    # save model
    save_path = "/" """" *This will be filled at .exe step """
    saver = tf.train.Saver()
    saver.save(save_path=save_path)

x_train, y_train, x_valid, y_valid, x_test, y_test = load_input()

X = tf.placeholder("float")
Y = tf.placeholder("float")

W, b = init_weights()
hypothesis = make_model(X, W, b)

cost = cost_function()
optimizer = make_optimizer()
train = optimizer.minimize(cost)

with tf.Session() as sess:
    init = tf.initialize_all_variables()
    sess.run(init)
    train_data = load_train_data()
    for _ in range(training_epoch):
        sess.run(train, feed_dict=train_data)
        # some logging codes will be added...
    save_model()