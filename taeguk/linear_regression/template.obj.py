"""
when model type == "linear_regression",
"""

import tensorflow as tf

def make_model(X, W, b):
    return tf.matmul(X, W) + b

def init_weights():
    """
    when model.initializer.type == "random",
    """
    return tf.Variable(tf.random_uniform(""" *This will be filled at .exe step """,
                                         """<model.initializer.low>""", """<model.initializer.high>""")), \
           tf.Variable(tf.random_uniform(""" *This will be filled at .exe step """,
                                         """<model.initializer.low>""", """<model.initializer.high>"""))

def make_optimizer():
    """
    when model.optimizer.type == "gradient_descent",
    """
    learning_rate = tf.Variable("""<model.optimizer.learning_rate>""")
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    return optimizer

def load_train_data():
    # load train data from train data files.
    train_xs = """ *This will be filled at .exe step """
    train_ys = """ *This will be filled at .exe step """
    data = {X: train_xs, Y: train_ys}
    return data

def save_model():
    # save model
    save_path = """ *This will be filled at .exe step """
    saver = tf.train.Saver()
    saver.save(save_path=save_path)

training_epochs = """<model.training_epochs>"""

X = tf.placeholder("float", """ *This will be filled at .exe step """)
Y = tf.placeholder("float", """ *This will be filled at .exe step """)

W, b = init_weights()

hypothesis = make_model(X, W, b)

cost = tf.reduce_mean(tf.square(hypothesis - Y))
optimizer = make_optimizer()
train = optimizer.minimize(cost)

with tf.Session() as sess:
    init = tf.initialize_all_variables()
    sess.run(init)
    train_data = load_train_data()
    for _ in range(training_epochs):
        sess.run(train, feed_dict=train_data)
        # some logging codes will be added...
    save_model()