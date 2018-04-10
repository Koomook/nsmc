import tensorflow as tf
from tensorflow.python.layers.core import dense
import numpy as np
import math

def multi_head_attention(query, key=None, n_heads=8, causalty=True, keep_prob=0.8,reuse=None, is_training=True):
    """
    blah blah

        Args:
            query: current value
            key: memory from another computation or query as same -> self attention
    """
    def _split_concat(inputs, n_heads=n_heads, split_axis=2, concat_axis=0):
        return tf.concat(tf.split(inputs, n_heads, axis=split_axis), axis=concat_axis)

    def _scaling(inputs, embedding_size):
        return inputs / (embedding_size**0.5)

    def _dot_product_attention():
        # dot product
        matmul = tf.matmul(scaled_q, K_, transpose_b=True)
        # mask option
        # add bias here
        bias = tf.get_variable('bias', [matmul.get_shape().as_list()[-1]], initializer=tf.zeros_initializer())
        logits = matmul + bias

        if causalty:
            with tf.variable_scope('tril'):
                diagonal = tf.ones_like(logits[0,:,:])
                if tf.__version__ == '1.4.0':
                    tril_fn = tf.contrib.linalg.LinearOperatorTriL
                elif tf.__version__ == '1.5.0':
                    tril_fn = tf.contrib.linalg.LinearOperatorLowerTriangular
                tril = tril_fn(diag_vals).to_dense() # (T_q, T_k)
                masks = tf.tile(tf.expand_dims(tril,0), [shape_list(logits)[0],1,1]) # 직각삼각형을 쌓는다.
                paddings = tf.ones_like(masks) * (-2**32+1) # extreamly small value for softmax
                logits = tf.where(tf.equal(masks,0), paddings, logits)
        # get weights
        logits = tf.nn.softmax(logits) # 여기를 우리는 다른 걸로 바꾸면 어떨까?
        logits = tf.nn.dropout(logits, keep_prob)
        return tf.matmul(logits, V_)

    # checking self attention
    if key is None:
        key = query
    d = query.get_shape().as_list()[-1]
    n_units = d//n_heads

    Q = dense(query, d, use_bias=False)
    K = dense(key, d, use_bias=False)
    V = dense(key, d, use_bias=False) # (batch_size, n_target, embedding_size)

    Q_ = _split_concat(Q)
    K_ = _split_concat(K)
    V_ = _split_concat(V) # (batch_size*n_head, n_target, embedding_size/n_head)

    # pre scaling
    scaled_q = _scaling(Q_, d)
    # dot product attention
    with tf.variable_scope('dot_product_attention'):
        outputs = _dot_product_attention()
    # restore shape to beginning
    outputs = _split_concat(outputs, split_axis=0, concat_axis=2)
    # linear projection
    outputs = dense(outputs, d, use_bias=False, name='output_transform') # from google code
    return outputs

def feed_forward(inputs, filter_size=256*4, output_size=256, keep_prob=0.8, activation=tf.nn.relu):
    """
    dense_relu_dense
    """
    hidden = dense(inputs, filter_size, activation=activation, use_bias=True, name='conv1')
    hidden = tf.nn.dropout(hidden, keep_prob)
    outputs = dense(hidden, output_size, use_bias=True, name='conv2')

    return outputs

def layer_norm(inputs, epsilon=1e-6):
    """Layer norm raw computation"""
    with tf.variable_scope('layer_norm'):
        filters = inputs.get_shape()[-1]
        scale = tf.get_variable(
            "scale", [filters], initializer=tf.ones_initializer())
        bias = tf.get_variable(
            "bias", [filters], initializer=tf.zeros_initializer())
        if tf.__version__ == '1.4.0':
            mean = tf.reduce_mean(inputs, axis=-1, keep_dims=True)
            variance = tf.reduce_mean(tf.square(inputs-mean), axis=-1, keep_dims=True)
        elif tf.__version__ == '1.5.0':
            mean = tf.reduce_mean(inputs, axis=-1, keepdims=True)
            variance = tf.reduce_mean(tf.square(inputs-mean), axis=-1, keepdims=True)
        noise_sigma = tf.sqrt(variance + epsilon)
        return tf.divide(tf.subtract(inputs, mean), noise_sigma)

def layer_postprocess(previous_inputs, inputs, residual=True, normalize=True, dropout=True, keep_prob=0.8):
    """computation order must be d->r->n"""
    with tf.variable_scope('post_process'):
        outputs = inputs
        if dropout:
            outputs = tf.nn.dropout(outputs, keep_prob)
        if residual:
            outputs += previous_inputs
        if normalize:
            outputs = layer_norm(outputs)
        return outputs

def create_embedding_table(name, n_labels, embedding_size, initializer=tf.contrib.layers.xavier_initializer(), scaling=True):
    with tf.variable_scope('embedding_table'):
        table = tf.get_variable(name, dtype=tf.float32, shape=[n_labels, embedding_size], initializer=initializer)
        if scaling:
            return table * (embedding_size**0.5)
        return table

# from Tensor2Tensor
def get_timing_signal_1d(length, channels, min_timescale=1.0, max_timescale=1.0e4):
    """Gets a bunch of sinusoids of different frequencies.

    Args:
    length: scalar, length of timing signal sequence.
    channels: scalar, size of timing embeddings to create. The number of
        different timescales is equal to channels / 2.
    min_timescale: a float
    max_timescale: a float

    Returns:
    a Tensor of timing signals [1, length, channels]
    """
    position = tf.to_float(tf.range(length))
    num_timescales = channels // 2

    log_timescale_increment = (
        math.log(float(max_timescale) / float(min_timescale)) /
        (tf.to_float(num_timescales) - 1))
    inv_timescales = min_timescale * tf.exp(
        tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
    scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
    signal = tf.pad(signal, [[0, 0], [0, tf.mod(channels, 2)]])
    signal = tf.reshape(signal, [1, length, channels])
    return signal

def add_timing_signal_1d(x, min_timescale=1.0, max_timescale=1.0e4):
  length = shape_list(x)[1]
  channels = shape_list(x)[2]
  signal = get_timing_signal_1d(length, channels, min_timescale, max_timescale)
  return x + signal

def shape_list(x):
    """Return list of dims, statistically where possible"""
    x = tf.convert_to_tensor(x)

    # if unknown shape, return dynaic shape
    if x.get_shape().dims is None:
        return tf.shape(x)

    static = x.get_shape().as_list()
    shape = tf.shape(x)

    result = []
    for i in range(len(static)):
        dim = static[i]
        if dim is None:
            dim = shape[i]
        result.append(dim)
    return result
    