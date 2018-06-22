import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from CNN import cnn, models

r"""
#data_size:batch_size, w, h, channel
#conv_size:k_size, k_size, channel, deep
#pool_size:channel, k_size.k_size, deep
#fcon_size:input, output
"""


def train(mnist, model_save_path):
    global_step = tf.Variable(0, trainable=False)
    fcnet = models.fcnet()

    x = tf.placeholder(tf.float32, fcnet[0]['size'], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, fcnet[-1]['size'][1]], name='y-input')

    train_step, loss = cnn.train_step_get(fcnet, mnist.train.num_examples / fcnet[0]['size'][0], global_step, x, y_)
    variable_averages_op = cnn.variable_averages_op_get(global_step=global_step)

    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name='train')
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        for i in range(10000):
            xs, ys = mnist.train.next_batch(fcnet[0]['size'][0])
            _None, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})

            if i % 1000 == 0:
                print("Steps:%d, Loss:%g" % (step, loss_value))
                saver.save(sess, model_save_path, global_step=global_step)


def evaluate(mnist, model_save_path):
    fcnet = models.fcnet()
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')
    y = cnn.inference(fcnet, x, None, train=False)

    validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver = tf.train.Saver()

    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(model_save_path)
        if ckpt and ckpt.model_checkpoint_path:
            print(ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
            accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
            print("Accuracy:%g" % (accuracy_score))
        else:
            print('error')
            return


def main(argv):
    model_save_path = './model/'
    model_name = "model.ckpt"
    if os.path.isfile(model_save_path):
        print(model_save_path, 'is a file')
        return -1
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)

    mnist = input_data.read_data_sets('../MNIST_data/', one_hot=True)
    # train(mnist, os.path.join(model_save_path, model_name))
    evaluate(mnist, model_save_path)


if __name__ == '__main__':
    tf.app.run()
