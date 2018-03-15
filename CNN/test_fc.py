import tensorflow as tf

import numpy as np
from functools import reduce

from tensorflow.examples.tutorials.mnist import input_data
import cnn
import os

VGG_MEAN = [103.939, 116.779, 123.68]

batch_size=100
learning_rate_base=0.02
learning_rate_decay=0.99
regularization_rate=0.0001

moving_average_decay=0.99

model_save_path='./model/'
model_name="model.ckpt"

class test_cnn(cnn.Cnn):

    def __init__(self):
    	self.data_dict = None
    	self.trainable = True
    	self.var_dict = {}
    	print("init...")


    def build(self, input_img):

        self.l1 = self.fc_layer(input_img, 784, 200, "fc_layer1")
        self.l2 = self.fc_layer(self.l1, 200, 10, "fc_layer2")

        self.prob = tf.nn.softmax(self.l2, name="prob")
        self.data_dict = None

def train(mnist):
	dnn = test_cnn()
    
	x=tf.placeholder(tf.float32,[None,784],name='x-input')
	y_=tf.placeholder(tf.float32,[None,10],name='y-input')
	dnn.build(x)
	
	y=dnn.l2

	global_step=tf.Variable(0,trainable=False)
	variable_averages=tf.train.ExponentialMovingAverage(moving_average_decay,global_step)
	variable_averages_op=variable_averages.apply(tf.trainable_variables())

	cost = tf.reduce_mean((dnn.l2 - y_) ** 2)

	learning_rate=tf.train.exponential_decay(learning_rate_base,global_step,mnist.train.num_examples/batch_size,learning_rate_decay)
	train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(cost,global_step=global_step)

	with tf.control_dependencies([train_step,variable_averages_op]):
		train_op=tf.no_op(name='train')

	saver=tf.train.Saver()

	training_steps=30000
	with tf.device('/gpu:0'):
		with tf.Session() as sess:
			tf.global_variables_initializer().run()

			for i in range(training_steps):
				xs,ys=mnist.train.next_batch(batch_size)
				_None,loss_value,step=sess.run([train_op,cost,global_step],feed_dict={x:xs,y_:ys})

				if i%1000==0:
					print("Steps:%d,Loss:%g"%(step,loss_value))
					saver.save(sess,os.path.join(model_save_path,model_name),global_step=global_step)

def main(argv=None):

	if os.path.isfile(model_save_path):
		print model_save_path,'is a file'
		return -1
	if not os.path.exists(model_save_path):
		os.mkdir(model_save_path)
	mnist=input_data.read_data_sets('../MNIST_data/',one_hot=True)
	train(mnist)

if __name__=='__main__':
	tf.app.run()
