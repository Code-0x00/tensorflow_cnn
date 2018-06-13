import tensorflow as tf
import random
from CNN import cnn
import pandas as pd
import numpy as np


def input_data(test=False):
    TRAIN_FILE = 'training.csv'
    TEST_FILE = 'test.csv'

    file_name = TEST_FILE if test else TRAIN_FILE
    df = pd.read_csv(file_name)
    cols = df.columns[:-1]

    # dropna()是丢弃有缺失数据的样本，这样最后7000多个样本只剩2140个可用的。
    df = df.dropna()
    df['Image'] = df['Image'].apply(lambda img: np.fromstring(img, sep=' ') / 255.0)

    x = np.vstack(df['Image'])
    x = x.reshape((-1, 96, 96, 1))

    if test:
        y = None
    else:
        y = df[cols].values / 96.0  # 将y值缩放到[0,1]区间

    return x, y


net = [
    {"name": "l0input", "type": "data", "size": [64, 96, 96, 1]},
    {"name": "l1conv1", "type": "conv", "size": [3, 3, 1, 32], "strides": [1, 1, 1, 1], "padding": "SAME"},
    {"name": "l2pool1", "type": "pool", "size": [1, 2, 2, 1], "strides": [1, 2, 2, 1], "padding": "SAME"},
    {"name": "l3conv2", "type": "conv", "size": [3, 3, 32, 64], "strides": [1, 1, 1, 1], "padding": "SAME"},
    {"name": "l4pool2", "type": "pool", "size": [1, 2, 2, 1], "strides": [1, 2, 2, 1], "padding": "SAME"},
    {"name": "l5conv3", "type": "conv", "size": [3, 3, 64, 128], "strides": [1, 1, 1, 1], "padding": "SAME"},
    {"name": "l6pool3", "type": "pool", "size": [1, 2, 2, 1], "strides": [1, 2, 2, 1], "padding": "SAME"},
    {"name": "l7conv4", "type": "conv", "size": [3, 3, 128, 256], "strides": [1, 1, 1, 1], "padding": "VALID"},
    {"name": "l8pool4", "type": "pool", "size": [1, 2, 2, 1], "strides": [1, 2, 2, 1], "padding": "SAME"},
    {"name": "reshape", "type": "shape"},
    {"name": "l9fcon1", "type": "fcon", "size": [6400, 500], 'dropout': 1.0},
    {"name": "l10fcon2", "type": "fcon", "size": [500, 500], 'dropout': 0.5},
    {"name": "l11fcon3", "type": "fcon", "size": [500, 30], 'dropout': 1.0}
]


def train():
    regularization_rate = 0.0001

    x = tf.placeholder("float", shape=[None, 96, 96, 1])
    y_ = tf.placeholder("float", shape=[None, 30])
    keep_prob = tf.placeholder("float")
    regularizer = tf.contrib.layers.l2_regularizer(regularization_rate)

    y_conv = cnn.inference(net, x, regularizer, train=True)
    rmse = tf.sqrt(tf.reduce_mean(tf.square(y_ - y_conv)))

    VALIDATION_SIZE = 100  # 验证集大小
    EPOCHS = 100  # 迭代次数
    BATCH_SIZE = 64  # 每个batch大小，稍微大一点的batch会更稳定
    EARLY_STOP_PATIENCE = 10  # 控制early stopping的参数

    train_step = tf.train.AdamOptimizer(1e-5).minimize(rmse)

    best_validation_loss = 1000000.0
    current_epoch = 0

    X, y = input_data()
    X_valid, y_valid = X[:VALIDATION_SIZE], y[:VALIDATION_SIZE]
    X_train, y_train = X[VALIDATION_SIZE:], y[VALIDATION_SIZE:]

    TRAIN_SIZE = X_train.shape[0]
    train_index = list(range(TRAIN_SIZE))
    random.shuffle(train_index)
    X_train, y_train = X_train[train_index], y_train[train_index]

    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        print('begin training..., train dataset size:{0}'.format(TRAIN_SIZE))
        for i in range(EPOCHS):
            random.shuffle(train_index)  # 每个epoch都shuffle一下效果更好
            X_train, y_train = X_train[train_index], y_train[train_index]

            for j in range(0, TRAIN_SIZE, BATCH_SIZE):
                # print('epoch {0}, train {1} samples done...'.format(i, j))

                train_step.run(feed_dict={x: X_train[j:j + BATCH_SIZE],
                                          y_: y_train[j:j + BATCH_SIZE], keep_prob: 0.5})

            # 电脑太渣，用所有训练样本计算train_loss居然死机，只好注释了。
            # train_loss = rmse.eval(feed_dict={x:X_train, y_:y_train, keep_prob: 1.0})
            validation_loss = rmse.eval(feed_dict={x: X_valid, y_: y_valid, keep_prob: 1.0})

            print('epoch {0} done! validation loss:{1}'.format(i, validation_loss * 96.0))
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                current_epoch = i
                saver.save(sess, 'model/model.ckpt')  # 即时保存最好的结果
            elif (i - current_epoch) >= EARLY_STOP_PATIENCE:
                print('early stopping')
                break


if __name__ == '__main__':
    train()
