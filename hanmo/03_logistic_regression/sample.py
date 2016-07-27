import tensorflow as tf
import numpy as np

xy=np.loadtxt('data.txt',unpack=True,dtype='float32')
x_data=xy[0:-1]
y_data=xy[-1]

learning_rate = 0.8
iteration = 20010

W=tf.Variable(tf.random_uniform([1,len(x_data)],-1.0,1.0))

h=tf.matmul(W,x_data)
hypothesis = tf.sigmoid(h,name='hypothesis')

cost = -tf.reduce_mean(y_data*tf.log(hypothesis)+(1-y_data)*tf.log(1-hypothesis))
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(cost)

init = tf.initialize_all_variables()

sess= tf.Session()
sess.run(init)

for step in range(iteration):
    sess.run(train)

print(sess.run(cost),sess.run(W))