import tensorflow as tf

def calculate_loss(logits, labels, label_dim, label_smoothing=False):
    """Calculate loss

    Args:
        logits: logits calcuated from your model
        labels: ground truth labels
    
    Returns:
        mean_loss: degree of difference between logits and labels
    """
    def _label_smoothing(inputs, epsilon=0.1):
        '''Applies label smoothing. See https://arxiv.org/abs/1512.00567.

        Args:
          inputs: A 3d tensor with shape of [N, T, V], where V is the number of vocabulary.
          epsilon: Smoothing rate.
        '''
        K = shape_list(inputs)[-1] # number of channels
        return ((1.-epsilon) * inputs) + (epsilon / K)

    y = tf.one_hot(labels, depth=label_dim)

    if label_smoothing:
        y = _label_smoothing(y)
    loss = tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=y)
    mean_loss = tf.reduce_mean(loss)

    preds = tf.to_int32(tf.argmax(logits, axis=-1))        
    acc = tf.reduce_mean(tf.to_float(tf.equal(preds, labels)))
    return mean_loss, acc, preds