import tensorflow as tf
import numpy as np

#데이터의 전처리
xy=np.loadtxt('data.txt', unpack=True, dtype='float32')
x_data = xy[0:-1]
y_data = xy[-1]

#필요한 변수들
running_rate =0.01
iteration =2001
init_min = -1.0
init_max = 1.0

#변수의 갯수 지정 필요 (x0,X1......Xn) -x_data의 길이로 대체
#초기화의 선택
W=tf.Variable(tf.random_uniform([1,len(x_data)],init_min,init_max))

hypothesis = tf.matmul(W,x_data)
cost = tf.reduce_mean(tf.square(hypothesis-y_data))

#Running rate 설정 필요
optimizer = tf.train.GradientDescentOptimizer(running_rate)
train = optimizer.minimize(cost)

init = tf.initialize_all_variables()

sess= tf.Session()
sess.run(init)

#반복 횟수 설정 필요
for step in range(iteration):
    sess.run(train)

print(sess.run(cost), sess.run(W))




