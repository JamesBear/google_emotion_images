
"""
This program classifies images of people grabbed from image.google.com by different emotions using CNN.
"""
import numpy as np
from data_loader import load_data
from log_utils import *
import tensorflow as tf
import os
from process_original_images import get_first_face_image
import sys

FACE_SIZE=40
LABELS = np.array(['angry', 'disgusted', 'happy', 'sad', 'scared', 'surprised', 'emotionless'])
TRAINED_MODEL_PATH = './larger_lenet5_emotion_recognition_best.ckpt'
REAL_WORLD_PICS = ['katy_perry.jpg']


from sklearn.preprocessing import LabelEncoder
def load_dataset():
    global X, y
    X, y = load_data()
    le = LabelEncoder()
    le.fit(LABELS)
    y = le.transform(y)
    lprint(le.classes_)

from sklearn.model_selection import train_test_split
def split_dataset():
    global X, y, X_train, y_train, X_valid, y_valid, X_test, y_test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.13, stratify = y, shuffle=True)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, stratify = y_train)
    lprint('train, valid, test:', len(y_train), len(y_valid), len(y_test))

def init():
    global watch
    init_log_utils()
    watch = StopWatch()
    if len(sys.argv) > 1:
        target_image = sys.argv[1].strip('\'" ')
        REAL_WORLD_PICS[0] = target_image
        print('Image to classify:', target_image)
    np.random.seed(1024)

import matplotlib
import math
import matplotlib.pyplot as plt
def plot_images(instances, width, labels=None, title='emotion:'):
    if instances.shape[0] > 100:
        instances = instances[:100,]
    m = instances.shape[0]
    columns = int(math.sqrt(m*16/9))
    rows = m // columns
    if rows*columns < m:
        rows += 1
    #width = int(math.sqrt(instances.shape[1]))
    figsize = (max(int(columns/6*10), 8), max(int(rows/6*10), 4))
    fig = plt.figure(figsize=figsize)
    for i in range(m):
        plt.subplot(rows,columns,i+1)
        plt.tight_layout()
        #plt.imshow(instances[i].reshape(-1,width), cmap='gray', interpolation='none')
        plt.imshow(instances[i].reshape(-1,width, 3))
        if type(labels) != type(None):
            plt.title("{}{}".format(title, labels[i]))
        plt.xticks([])
        plt.yticks([])
    plt.show()

# Build network
from network_builder import build_lenet5, build_larger_lenet5

def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch

from AMSGrad import AMSGrad

# Adam optimization and early stopping.
# Save checkpoints at regular intervals and save the final model.
# Save learning curve.
# Then Try Batch Normalization and compare the learning curve.
# Is it overfitting? Then try dropout.
def build_optimizer(logits, y):

    with tf.name_scope('loss'):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=y, logits=logits)
        loss = tf.reduce_mean(xentropy, name='loss')

    with tf.name_scope('train'):
        learning_rate = 0.0001
        lprint('learning_rate =', learning_rate)
        #optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.99, beta2=0.999, amsgrad=True)
        #optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        optimizer = AMSGrad(learning_rate=learning_rate, beta1=0.9, beta2=0.99, epsilon=1e-8)
        lprint('optimizer = ', optimizer)
        training_op = optimizer.minimize(loss)
    return loss, training_op

def divide_into_batches(X, y, batch_size):
    m = X.shape[0]
    n_batches = m // batch_size
    if n_batches * batch_size < m:
        n_batches += 1
    for i in range(n_batches):
        start = i*batch_size
        end = min((i+1)*batch_size, m)
        yield i, X[start:end,:], y[start:end]

def test_real_world_pictures(sess, pred, X, is_training):
    pic = get_first_face_image(REAL_WORLD_PICS[0])
    if type(pic) == type(None):
        lprint('No face detected.')
        return None, None
    X_real = pic.reshape([1]+list(pic.shape))
    y_real = sess.run(pred, feed_dict={X:X_real, is_training:False})
    return X_real, y_real

def run_training(X, y, logits, is_training, loss, training_op, skip_learning=False):
    from datetime import datetime

    now = datetime.now().strftime("%Y%m%d%H%M%S")
    root_logdir = "tf_ann_logs"
    logdir = "{}/run-{}/".format(root_logdir, now)

    with tf.name_scope('eval'):
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    with tf.name_scope('summary'):
        acc_valid_summary = tf.summary.scalar('acc_valid', accuracy)
        acc_batch_summary = tf.summary.scalar('acc_batch', accuracy)
        loss_summary = tf.summary.scalar('loss_valid', loss)
        file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())


    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    pred = tf.argmax(logits, 1)

    n_epochs = 5500
    batch_size = 50
    n_batches = X_train.shape[0]//batch_size
    early_stopping_impatience=0
    max_early_stopping_impatience=100
    best_loss = np.infty
    best_val_acc = 0

    lprint('max early stopping impatience:', max_early_stopping_impatience)

    watch.start()
    if not skip_learning:
        with tf.Session() as sess:
            init.run()
            for epoch in range(n_epochs):
                for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
                    batch_feed = {X: X_batch, y: y_batch, is_training:True}
                    sess.run(training_op, feed_dict=batch_feed)
                acc_batch = accuracy.eval(feed_dict=batch_feed)
                #acc_val = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
                step = (epoch+1)*n_batches
                acc_val, loss_val, acc_str, loss_str = sess.run([accuracy, loss, acc_valid_summary, loss_summary], feed_dict = {X: X_valid, y: y_valid, is_training:False})
                file_writer.add_summary(acc_str, step)
                file_writer.add_summary(loss_str, step)
                lprint(epoch, 'batch accuracy:', acc_batch, ', Val accuracy:', acc_val, ', Val loss:', loss_val)
                #temp_path = saver.save(sess, './my_model_temp.ckpt')
                if loss_val < best_loss:
                    save_path = saver.save(sess, TRAINED_MODEL_PATH)
                    best_loss = loss_val
                    best_val_acc = acc_val
                    early_stopping_impatience = 0
                else:
                    early_stopping_impatience += 1
                    if early_stopping_impatience > max_early_stopping_impatience:
                        lprint('Early stopping!')
                        break

        #save_path = saver.save(sess, './my_model_final.ckpt')
    watch.stop('AdamOptimization')
    file_writer.close()
    with tf.Session() as sess:
        saver.restore(sess, TRAINED_MODEL_PATH)
        acc_test, loss_test, y_pred = sess.run([accuracy, loss, pred], feed_dict={X:X_test, y:y_test, is_training:False})
        sum_acc_train = 0
        sum_loss_train = 0
        num_train_batches = 0
        for i, X_train_batch, y_train_batch in divide_into_batches(X_train, y_train, 1000):
            acc_train, loss_train = sess.run([accuracy, loss], feed_dict={X:X_train_batch, y:y_train_batch, is_training:False})
            sum_acc_train += acc_train
            sum_loss_train += loss_train
            num_train_batches += 1
        acc_train = sum_acc_train / num_train_batches
        loss_train = sum_loss_train / num_train_batches
        real_world_X, real_world_y = test_real_world_pictures(sess, pred, X, is_training)

    lprint('Final classifier:')
    lprint('train acc={}, val acc={}, test acc={}'.format(acc_train, best_val_acc, acc_test))
    lprint('train loss={}, val loss={}, test loss={}'.format(loss_train, best_loss, loss_test))

    lprint('wrong prediction on test:', (y_pred != y_test).sum())
    #lprint('Plotting wrong cases:')
    #plot_images(X_test[(y_pred != y_test),:], FACE_SIZE, y_pred[(y_pred != y_test)], 'pred:')
    if type(real_world_X) != type(None):
        lprint('Plotting real world cases:')
        plot_images(real_world_X, FACE_SIZE, LABELS[real_world_y.astype('int')])

def main():
    init()
    load_dataset()
    split_dataset()
    #plot_images(X_train[1013:1016], FACE_SIZE, y_train[1013:1016])
    #X, y, logits, is_training = build_lenet5(X_train.shape, len(LABELS), use_batch_norm=True, use_dropout=True)
    X, y, logits, is_training = build_larger_lenet5(X_train.shape, len(LABELS), use_batch_norm=True, use_dropout=True)
    loss, training_op = build_optimizer(logits, y)
    skip_training = os.path.exists(TRAINED_MODEL_PATH+'.meta')
    run_training(X, y, logits, is_training, loss, training_op, skip_learning=skip_training)





if __name__ == '__main__':
    main()
