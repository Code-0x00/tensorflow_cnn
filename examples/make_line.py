import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def make_line():
    # 使用 NumPy 生成假数据(phony data), 总共 100 个点.

    x_data = np.float32(np.random.rand(1, 100))  # 随机输入
    y_data = np.dot([0.100, 0.200], x_data) + 0.300

    # 构造一个线性模型
    #
    b = tf.Variable(tf.zeros([1]))
    w = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))
    y = tf.matmul(w, x_data) + b

    # 最小化方差
    loss = tf.reduce_mean(tf.square(y - y_data))
    optimizer = tf.train.GradientDescentOptimizer(0.5)
    train = optimizer.minimize(loss)

    # 拟合平面
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for step in range(201):
            sess.run(train)
            if step % 20 == 0:
                k = sess.run(w)[0]
                d = sess.run(b)[0]
                print(step, k, d)

                x_p = [[-1, -1], [1, 1]]
                k_label = [0.100, 0.200]
                y_label = [
                    np.dot(k_label, x_p[0]) + 0.300,
                    np.dot(k_label, x_p[1]) + 0.300
                ]
                y_p = [
                    np.dot(k, x_p[0]) + d,
                    np.dot(k, x_p[1]) + d
                ]
                plt.cla()
                plt.plot([-10, 10], y_label)
                plt.plot([-10, 10], y_p)
                plt.pause(0.9)


if __name__ == '__main__':
    make_line()
