import tensorflow as tf
from tensorflow.python.layers.core import dense
import numpy as np
from .utils import *

class AttentionMLP(object):
    """This Model is based on self-attention frame work.
    as called transformer : https://arxiv.org/abs/1706.03762

    Attributes:
        inputs: placeholder for inputs
        targets: placeholder for targets
        keep_prob: the probability of dropout layer
    """

    def __init__(self,
                 batch_size=None,
                 n_input=30,
                 n_target=2,
                 n_blocks=4,
                 n_heads=8,
                 vocab_size=100000,
                 embedding_size=256,
                 ):
        """initialize fundamental variables
            
        """
        self.n_input = n_input
        self.n_target = n_target
        self.n_blocks = n_blocks
        self.n_heads = n_heads

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        
    def build_network(self, scope=None, default_name='SemanticClassifier'):
        with tf.variable_scope(scope, default_name=default_name):
            self.load_inputs()
            self.declare_variables()

            self.encoder_inputs = add_timing_signal_1d(self.inputs_embedded)
            self.outputs = self.self_attention(self.encoder_inputs)
            
            self.logits = dense(tf.reshape(self.outputs,[-1,1,self.n_input*self.embedding_size]), self.n_target, name='logits')

    def load_inputs(self):
        """Declare fundamental placeholder and weights to use for model"""
        
        self.inputs = tf.placeholder(tf.int32, shape=(None,self.n_input), name='inputs')
        self.inputs_mask = tf.placeholder(tf.float32, shape=(None, self.n_input), name='inputs_mask')
        self.targets = tf.placeholder(tf.int32, shape=(None,1), name='targets')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    def declare_variables(self):
        self.embedding_table = create_embedding_table('word_embedding', self.vocab_size, self.embedding_size, scaling=True)
        self.inputs_embedded = tf.nn.embedding_lookup(self.embedding_table, self.inputs)

    def self_attention(self, inputs):
        with tf.variable_scope('encoder'):
            # dropout
            x = tf.nn.dropout(inputs, self.keep_prob)
            # multihead attention
            for i in range(self.n_blocks):
                with tf.variable_scope('{}th_block'.format(i+1)):
                    with tf.variable_scope('self_attention'):
                        y = multi_head_attention(query=x, causalty=False, keep_prob=self.keep_prob, n_heads=self.n_heads)
                        x = layer_postprocess(x, y, keep_prob=self.keep_prob)
                    with tf.variable_scope('feed_forward'):
                        y = feed_forward(x, output_size=self.embedding_size, keep_prob=self.keep_prob)
                        x = layer_postprocess(x, y, keep_prob=self.keep_prob)
            return x