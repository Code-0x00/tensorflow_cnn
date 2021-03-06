# -*- coding: utf-8 -*-
import time
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data

import fc_inference
import fc_train

eval_interal_secs = 10


def evaluate(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, fc_inference.input_node], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, fc_inference.output_node], name='y-input')
        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}

        y = fc_inference.inference(x, None)

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        variable_averages = tf.train.ExponentialMovingAverage(fc_train.moving_average_decay)
        variable_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variable_to_restore)

        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(fc_train.model_save_path)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
                    print("Step:%s,accuracy:%g" % (global_step, accuracy_score))
                else:
                    print('error')
                    return
            time.sleep(eval_interal_secs)


def main(argv=None):
    mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
    evaluate(mnist)


if __name__ == '__main__':
    tf.app.run()
