import tensorflow as tf
import numpy as np

xy=np.loadtxt('data.txt', unpack=True,dtype='float32')
x_data = xy[0:3]
y_data = xy[3:]

learning_rate = 0.1

W=tf.Variable(tf.random_uniform([len(y_data),len(y_data)]))

hypothesis = tf.nn.softmax(tf.matmul(W, x_data))
cost = tf.reduce_mean(-tf.reduce_sum(y_data*tf.log(hypothesis), reduction_indices=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)

    for step in range(2001):
        sess.run(optimizer)
        if step % 200 == 0:
            print(step, sess.run(cost), sess.run(W))
