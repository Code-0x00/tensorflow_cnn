import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os

import LeNet_5_inference

import numpy as np

batch_size = 100
learning_rate_base = 0.2
learning_rate_decay = 0.99
regularization_rate = 0.0001
training_steps = 30000
moving_average_decay = 0.99

model_save_path = './model/'
model_name = "model.ckpt"


def train(mnist):
    x = tf.placeholder(tf.float32, [
        batch_size,
        LeNet_5_inference.image_size,
        LeNet_5_inference.image_size,
        LeNet_5_inference.num_channels],
                       name='x-input')
    y_ = tf.placeholder(tf.float32, [None, LeNet_5_inference.output_node], name='y-input')

    regularizer = tf.contrib.layers.l2_regularizer(regularization_rate)
    y = LeNet_5_inference.inference(x, regularizer, train=True)

    global_step = tf.Variable(0, trainable=False)
    variable_averages = tf.train.ExponentialMovingAverage(moving_average_decay, global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

    learning_rate = tf.train.exponential_decay(learning_rate_base, global_step, mnist.train.num_examples / batch_size,
                                               learning_rate_decay)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name='train')

    saver = tf.train.Saver()
    with tf.Session() as sess:
        # tf.initialize_all_variables().run()
        tf.global_variables_initializer().run()

        for i in range(training_steps):
            xs, ys = mnist.train.next_batch(batch_size)
            reshaped_xs = np.reshape(xs, (batch_size,
                                          LeNet_5_inference.image_size,
                                          LeNet_5_inference.image_size,
                                          LeNet_5_inference.num_channels))
            _None, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: reshaped_xs, y_: ys})

            if i % 1000 == 0:
                print("Steps:%d,Loss:%g" % (step, loss_value))
                print(learning_rate.eval())
                saver.save(sess, os.path.join(model_save_path, model_name), global_step=global_step)


def main(argv=None):
    mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
    train(mnist)


if __name__ == '__main__':
    tf.app.run()
