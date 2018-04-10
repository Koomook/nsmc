import tensorflow as tf

def adam_learningrate(loss):
    """Produces trainer with decaying learning rate
    """
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    builder = tf.train.AdamOptimizer(learning_rate, 0.9, 0.98, 1e-9)
    optimizer = builder.minimize(loss)
    return optimizer, learning_rate