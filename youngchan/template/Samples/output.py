"""
DrawML
Linear regression template file
Templates will be changed by appropriate value

There is some issues in input data
and maybe save model
"""
import tensorflow as tf

learning_rate = 0.01
training_epoch = 1024
x_data = [1,2,3]
y_data = [1,2,3]

# this code has to be changed
x_train  = x_data
y_train  = y_data
x_valid  = x_data
y_valid  = y_data
x_test   = x_data
y_test   = y_data


def load_train_data():
    data = {X: x_train, Y: y_train}
    return data


def make_model(X, W, b):
    return tf.add(tf.matmul(X, W), b)


def init_weights():
    return tf.Variable(tf.random_uniform([1], -1.0, 1.0)), \
           tf.Variable(tf.random_uniform([1], 0, 0))


def make_optimizer():
    return tf.train.GradientDescentOptimizer(learning_rate)


def save_model():
    # save model
    save_path = "/" """" *This will be filled at .exe step """
    saver = tf.train.Saver()
    saver.save(save_path=save_path)

X = tf.placeholder("float")
Y = tf.placeholder("float")

W, b = init_weights()

hypothesis = make_model(X, W, b)

cost = tf.reduce_mean(tf.square(hypothesis - Y))
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