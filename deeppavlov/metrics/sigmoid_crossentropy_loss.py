import numpy as np

import tensorflow as tf

from deeppavlov.core.common.metrics_registry import register_metric


@register_metric('sigmoid_crossentropy_loss')
def sigmoid_crossentropy_loss(y_true, y_pred):
    n_classes = np.array(y_true).shape[1]

    sess = tf.InteractiveSession()
    p = tf.placeholder(tf.float32, shape=[None, n_classes])
    logit_q = tf.placeholder(tf.float32, shape=[None, n_classes])
    sess.run(tf.global_variables_initializer())

    feed_dict = {p: y_true,
                 logit_q: y_pred}

    x = tf.not_equal(p, -1)
    sigm = tf.nn.sigmoid_cross_entropy_with_logits(labels=p, logits=logit_q)
    loss3 = tf.reduce_mean(sigm[x]).eval(feed_dict)
    return float(loss3)
