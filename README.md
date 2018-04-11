# Build CNNs with TensorFlow
# 用 TensorFlow 运行各种深度卷积神经网络
## CNN
一个通用的CNN搭建函数，目标是用户只需定义网络结构和训练方案，就能直接调用这里的函数搭建CNN网络。这里的代码会不断补充，后续的网络实现都会以这里的代码为基础。  
如[LeNet-5](https://github.com/Code-0x00/tensorflow_cnn/blob/master/LeNet-5/LeNet_5_train.py)只需用一个lenet列表定义整个网络结构。
## FC
Classify MNIST by 2 layers of *fully connected layers(FC)*.  
用2层全连接层分类手写字MNIST

|type|output|
|----|----|
|data|784|
|fc|500|
|fc|10|
## LeNet-5
Classify MNIST by 7 layers LeNet-5  
参考LeNet-5模型，第一层输入改为28x28（原论文32x32），由3层卷积层和2层全连接层组成，另外前两层卷积层都连接一层池化层。  
Paper:[Gradient-based learning applied to document recognition](http://xueshu.baidu.com/s?wd=paperuri%3A%2880fd293244903d8233327d0e5ba6de62%29).
* 激活函数采用ReLU
|layer|type|output|size|
|----|----|----|----|
|0|data|28x28x 1|1x1|
|1|conv|28x28x 6|5x5|
|2|pool|14x14x 6|2x2|
|3|conv|10x10x16|5x5|
|4|pool| 5x 5x16|2x2|
|5|conv| 1x1x120|5x5|
|6|fcon|84x 1|1x1|
|7|fcon|10x 1|1x1|