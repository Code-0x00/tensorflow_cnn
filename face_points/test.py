import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from CNN import cnn

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
    {"name": "l9fcon1", "type": "fcon", "size": [6400, 500], 'droup': 1.0},
    {"name": "l10fcon2", "type": "fcon", "size": [500, 500], 'droup': 0.5},
    {"name": "l11fcon3", "type": "fcon", "size": [500, 30], 'droup': 1.0}
]


def main():
    SAVE_PATH = './model/model.ckpt'

    x = tf.placeholder("float", shape=[None, 96, 96, 1])
    keep_prob = tf.placeholder("float")

    img = cv2.imread('test.bmp')
    if img is None:
        print('Image Name Error')
        return
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (96, 96), interpolation=cv2.INTER_CUBIC)

    img_x = np.zeros((96, 96, 1))
    for i in range(96):
        for j in range(96):
            img_x[i][j] = [float(img[i][j]) / 255.0]
    X = [img_x]

    y_conv = cnn.inference(net, x, None, train=False)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, SAVE_PATH)
        y_batch = y_conv.eval(feed_dict={x: X, keep_prob: 1.0})

        test = y_batch[0]
        plt.imshow(img, cmap='gray')
        i = 0
        while i < 30:
            xx = 96 * test[i]
            i += 1
            yy = 96 * test[i]
            i += 1
            plt.plot(xx, yy, '*')

        plt.savefig('out.png')

        print('predict test image done!')


if __name__ == '__main__':
    main()
