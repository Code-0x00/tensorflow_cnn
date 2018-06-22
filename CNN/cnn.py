import tensorflow as tf

r"""
Type:
    data
    conv
    fcon
    pool
    shape
    
"""


def inference(cnn_net, input_tensor, regularizer, train):
    if train:
        pass
    last_layer_output = input_tensor
    for layer in cnn_net:
        if layer['type'] == 'data':
            last_layer_output = input_tensor

        elif layer["type"] == "conv":
            with tf.variable_scope(layer['name']):
                weights = tf.get_variable("weight", layer['size'],
                                          initializer=tf.truncated_normal_initializer(stddev=0.1))
                biases = tf.get_variable("bias", layer['size'][3], initializer=tf.constant_initializer(0.0))
                conv = tf.nn.conv2d(last_layer_output, weights, strides=layer['strides'], padding=layer['padding'])
                relu = tf.nn.relu(tf.nn.bias_add(conv, biases))
                last_layer_output = relu

        elif layer['type'] == 'fcon':
            with tf.variable_scope(layer['name']):
                weights = tf.get_variable('weight', layer['size'],
                                          initializer=tf.truncated_normal_initializer(stddev=0.1))
                if regularizer is not None:
                    tf.add_to_collection('losses', regularizer(weights))
                biases = tf.get_variable('bias', layer['size'][1], initializer=tf.constant_initializer(0.1))
                relu = tf.nn.relu(tf.matmul(last_layer_output, weights) + biases)
                if train and 'dropout' in layer.keys() and layer['dropout'] < 1.0:
                    relu = tf.nn.dropout(relu, layer['dropout'])
                last_layer_output = relu

        elif layer['type'] == 'pool':
            with tf.name_scope(layer['name']):
                pool = tf.nn.max_pool(last_layer_output,
                                      ksize=layer['size'], strides=layer['strides'], padding=layer['padding'])
                last_layer_output = pool

        elif layer['type'] == 'shape':
            pool_shape = last_layer_output.get_shape().as_list()
            nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
            reshaped = tf.reshape(last_layer_output, [-1, nodes])
            last_layer_output = reshaped

        print(layer['name'], last_layer_output.shape)
    return last_layer_output


def train_step_get(net, train_onece_times, global_step, x, y_):
    learning_rate_base = 0.08
    learning_rate_decay = 0.99
    regularization_rate = 0.0001

    regularizer = tf.contrib.layers.l2_regularizer(regularization_rate)
    y = inference(net, x, regularizer, train=True)

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

    learning_rate = tf.train.exponential_decay(learning_rate_base, global_step,
                                               train_onece_times, learning_rate_decay)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    return train_step, loss


def variable_averages_op_get(global_step):
    moving_average_decay = 0.99

    variable_averages = tf.train.ExponentialMovingAverage(moving_average_decay, global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    return variable_averages_op
