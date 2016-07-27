import tensorflow as tf
import numpy as np

#데이터의 전처리
xy=np.loadtxt('data.txt', unpack=True, dtype='float32')
pre_x_data = xy[0:-1]
y_data = xy[-1]

#필요한 변수들
polynomial=[[1],[1],[1],[1]] #b+x1+x1^3+x1^5+x1^7+x1^7+x2^1+.....

running_rate =0.01
iteration =2001
init_min = -1.0
init_max = 1.0
W_len=0

# X 데이터 처리
x_data=[]
for xIdx in range(len(polynomial)):
    for po in polynomial[xIdx]:
        x_data.append([(pow(x,po)) for x in pre_x_data[xIdx]])
        W_len+=1
x_data=tf.constant(x_data)



W=tf.Variable(tf.random_uniform([1,W_len],init_min,init_max))

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
