import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def make_line():
    # 使用 NumPy 生成假数据(phony data), 总共 100 个点.
    x_data = np.float32(np.random.rand(2, 100))  # 随机输入
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

    # 初始化变量
    init = tf.global_variables_initializer()

    # 启动图 (graph)
    sess = tf.Session()
    sess.run(init)

    # 拟合平面
    for step in range(0, 201):
        sess.run(train)
        if step % 20 == 0:
            print(step, sess.run(w), sess.run(b))
            k = sess.run(w)[0]
            d = sess.run(b)[0]

            plt.cla()
            y_p = []
            x_p = [[-1, -1], [1, 1]]
            k_label = [0.100, 0.200]
            y_p.append(np.dot(k_label, x_p[0]) + 0.300)
            y_p.append(np.dot(k_label, x_p[1]) + 0.300)
            plt.plot([-10, 10], y_p)

            y_p = list()
            y_p.append(np.dot(k, x_p[0]) + d)
            y_p.append(np.dot(k, x_p[1]) + d)
            plt.plot([-10, 10], y_p)

            plt.pause(0.9)

    #  得到最佳拟合结果 W: [[0.100  0.200]], b: [0.300]


if __name__ == '__main__':
    make_line()
