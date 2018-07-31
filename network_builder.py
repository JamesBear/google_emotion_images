
from log_utils import *
import tensorflow as tf
import numpy as np

from tensorflow.contrib.layers import conv2d, max_pool2d, avg_pool2d, fully_connected, batch_norm, dropout


def build_lenet5(X_train_shape, n_outputs, use_batch_norm=False, use_dropout=False):
    lprint('Mimicing LeNet-5')
    print('X_train_shape', X_train_shape)
    X = tf.placeholder(tf.float32, shape=(None, X_train_shape[1], X_train_shape[2], X_train_shape[3]), name='X')
    y = tf.placeholder(tf.int64, shape=(None), name='y')


    #fake_is_training = tf.placeholder(tf.bool, shape=(), name='is_training')
    last_output = X
    layers = []
    he_init = tf.contrib.layers.variance_scaling_initializer()
    norm_fn = None
    norm_params = None
    is_training=tf.placeholder(tf.bool, shape=(), name='is_training')
    keep_prob = 0.5
    lprint('Use Batch Normalization:', use_batch_norm)
    lprint('Use Dropout:', use_dropout, ', keep_prob:', keep_prob)
    if use_batch_norm:
        norm_fn = batch_norm
        norm_params = {'is_training':is_training, 'decay':0.99, 'updates_collections': None}

    with tf.name_scope('cnn'):
        with tf.contrib.framework.arg_scope(
                [fully_connected, conv2d],
                activation_fn = tf.nn.relu,
                #normalizer_fn=norm_fn,
                #normalizer_params=norm_params,
                weights_initializer=he_init):
            C1 = conv2d(inputs=X, num_outputs=32, kernel_size=5, stride=1, padding='SAME', normalizer_fn=norm_fn, normalizer_params=norm_params)
            P1 = tf.nn.max_pool(C1, ksize=[1,2,2,1], strides=[1,2,2,1],padding='SAME')
            C2 = conv2d(inputs=P1, num_outputs=64, kernel_size=5, stride=1, padding='SAME', normalizer_fn=norm_fn, normalizer_params=norm_params)
            P2 = tf.nn.max_pool(C2, ksize=[1,2,2,1], strides=[1,2,2,1],padding='SAME')
            lprint('Pool2 shape:', P2)
            pool_shape = P2.get_shape().as_list()
            #shaped = tf.reshape(last_pool, [-1, 10])
            reshape = tf.reshape(
                P2,
                [-1, pool_shape[1] * pool_shape[2] * pool_shape[3]])
            F1 = fully_connected(reshape, 512)
            if use_dropout:
                hidden_drop = dropout(F1, keep_prob, is_training=is_training)
                last_output = hidden_drop
            else:
                last_output = F1
        logits = fully_connected(last_output, n_outputs, scope='outputs', activation_fn=None, weights_initializer=he_init)
    lprint(C1)
    lprint(last_output)
    lprint(logits)
    return X, y, logits, is_training

def build_larger_lenet5(X_train_shape, n_outputs, use_batch_norm=False, use_dropout=False):
    lprint('Mimicing LeNet-5')
    print('X_train_shape', X_train_shape)
    X = tf.placeholder(tf.float32, shape=(None, X_train_shape[1], X_train_shape[2], X_train_shape[3]), name='X')
    y = tf.placeholder(tf.int64, shape=(None), name='y')


    #fake_is_training = tf.placeholder(tf.bool, shape=(), name='is_training')
    last_output = X
    layers = []
    he_init = tf.contrib.layers.variance_scaling_initializer()
    norm_fn = None
    norm_params = None
    is_training=tf.placeholder(tf.bool, shape=(), name='is_training')
    keep_prob = 0.5
    lprint('Use Batch Normalization:', use_batch_norm)
    lprint('Use Dropout:', use_dropout, ', keep_prob:', keep_prob)
    if use_batch_norm:
        norm_fn = batch_norm
        norm_params = {'is_training':is_training, 'decay':0.99, 'updates_collections': None}

    with tf.name_scope('cnn'):
        with tf.contrib.framework.arg_scope(
                [fully_connected, conv2d],
                activation_fn = tf.nn.relu,
                #normalizer_fn=norm_fn,
                #normalizer_params=norm_params,
                weights_initializer=he_init):
            C1 = conv2d(inputs=X, num_outputs=64, kernel_size=5, stride=1, padding='SAME', normalizer_fn=norm_fn, normalizer_params=norm_params)
            P1 = tf.nn.max_pool(C1, ksize=[1,3,3,1], strides=[1,2,2,1],padding='SAME')
            C2 = conv2d(inputs=P1, num_outputs=64, kernel_size=5, stride=1, padding='SAME', normalizer_fn=norm_fn, normalizer_params=norm_params)
            P2 = tf.nn.max_pool(C2, ksize=[1,3,3,1], strides=[1,2,2,1],padding='SAME')
            C3 = conv2d(inputs=P1, num_outputs=128, kernel_size=4, stride=1, padding='SAME', normalizer_fn=norm_fn, normalizer_params=norm_params)
            P3 = tf.nn.max_pool(C2, ksize=[1,2,2,1], strides=[1,2,2,1],padding='SAME')
            lprint('Pool3 shape:', P3)
            pool_shape = P2.get_shape().as_list()
            #shaped = tf.reshape(last_pool, [-1, 10])
            reshape = tf.reshape(
                P3,
                [-1, pool_shape[1] * pool_shape[2] * pool_shape[3]])
            F1 = fully_connected(reshape, 2048)
            if use_dropout:
                hidden_drop = dropout(F1, keep_prob, is_training=is_training)
                last_output = hidden_drop
            else:
                last_output = F1
        logits = fully_connected(last_output, n_outputs, scope='outputs', activation_fn=None, weights_initializer=he_init)
    lprint(C1)
    lprint(last_output)
    lprint(logits)
    return X, y, logits, is_training
