# data_size:batch_size,w,h,channel
# conv_size:k_size,k_size,channel,deep
# pool_size:channel,k_size.k_size,deep
# fcon_size:input,output
def n_net():
    net = [
        {"name": "l0input", "type": "data", "size": [100, 784]},
        {"name": "l1fcon0", "type": "fcon", "size": [784, 10]}
    ]
    return net


def lenet5():
    net = [
        {"name": "l0input", "type": "data", "size": [100, 28, 28, 1]},
        {"name": "l1conv1", "type": "conv", "size": [5, 5, 1, 6], "strides": [1, 1, 1, 1], "padding": "SAME"},
        {"name": "l2pool1", "type": "pool", "size": [1, 2, 2, 1], "strides": [1, 2, 2, 1], "padding": "SAME"},
        {"name": "l3conv2", "type": "conv", "size": [5, 5, 6, 16], "strides": [1, 1, 1, 1], "padding": "VALID"},
        {"name": "l4pool2", "type": "pool", "size": [1, 2, 2, 1], "strides": [1, 2, 2, 1], "padding": "SAME"},
        {"name": "l5conv3", "type": "conv", "size": [5, 5, 16, 120], "strides": [1, 1, 1, 1], "padding": "VALID"},
        {"name": "reshape", "type": "shape"},
        {"name": "l6fcon1", "type": "fcon", "size": [120, 84]},
        {"name": "l7fcon2", "type": "fcon", "size": [84, 10]}
    ]
    return net


def facial_keypoint_net():
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
    return net


def fcnet():
    net = [
        {"name": "l0input", "type": "data", "size": [100, 784]},
        {"name": "l1fcon0", "type": "fcon", "size": [784, 500]},
        {"name": "l2fcon1", "type": "fcon", "size": [500, 10]}
    ]
    return net
