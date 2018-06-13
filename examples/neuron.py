import os
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data

from CNN import cnn

model_save_path = './model/'
model_name = "model"
moving_average_decay = 0.99


def train(mnist, net_model):
    learning_rate_base = 0.08
    learning_rate_decay = 0.99
    training_steps = 30000
    x = tf.placeholder(tf.float32, net_model[0]['size'], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, net_model[-1]['size'][1]], name='y-input')

    y = cnn.inference(net_model, x, None, train=True)

    global_step = tf.Variable(0, trainable=False)
    variable_averages = tf.train.ExponentialMovingAverage(moving_average_decay, global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    loss = tf.reduce_mean(cross_entropy)

    learning_rate = tf.train.exponential_decay(learning_rate_base, global_step,
                                               mnist.train.num_examples / net_model[0]['size'][0], learning_rate_decay)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name='train')
    saver = tf.train.Saver()
    with tf.Session() as sess:
        loss_base = 1.0
        tf.global_variables_initializer().run()

        for i in range(1, training_steps + 1):
            xs, ys = mnist.train.next_batch(net_model[0]['size'][0])
            _none, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})

            if i % 1000 == 0:
                print("Steps:%d, Loss:%g" % (step, loss_value))
                if loss_value < loss_base:
                    loss_base = loss_value
                    saver.save(sess, os.path.join(model_save_path, model_name), global_step=global_step)


def evaluate(mnist, net_model):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, 784], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')

        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}

        y = cnn.inference(net_model, x, None, train=False)

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        variable_averages = tf.train.ExponentialMovingAverage(moving_average_decay)
        variable_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variable_to_restore)

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(model_save_path)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
                print("Step:%s,accuracy:%g" % (global_step, accuracy_score))
            else:
                print('error')
                return


def main(argv):
    n_net = [
        {"name": "l0input", "type": "data", "size": [100, 784]},
        {"name": "l1fcon0", "type": "fcon", "size": [784, 10]}
    ]
    if os.path.isfile(model_save_path):
        print(model_save_path, 'is a file')
        return -1
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)
    mnist = input_data.read_data_sets('../MNIST_data/', one_hot=True)
    train(mnist, n_net)
    evaluate(mnist, n_net)


if __name__ == '__main__':
    tf.app.run()
