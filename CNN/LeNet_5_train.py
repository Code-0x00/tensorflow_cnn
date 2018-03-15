import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os

import cnn

import numpy as np

#data_size:w,h,channel
#conv_size:k_size,k_size,channel,deep
#pool_size:channel,k_size.k_size,deep
#fcon_size:input,output

lenet=[
{"name":"l0input","type":"data","size":[28,28,1]},
{"name":"l1conv1","type":"conv","size":[ 5, 5, 1, 6],"strides":[1,1,1,1],"padding":"SAME"},
{"name":"l2pool1","type":"pool","size":[ 1, 2, 2, 1],"strides":[1,2,2,1],"padding":"SAME"},
{"name":"l3conv2","type":"conv","size":[ 5, 5, 6,16],"strides":[1,1,1,1],"padding":"VALID"},
{"name":"l4pool2","type":"pool","size":[ 1, 2, 2, 1],"strides":[1,2,2,1],"padding":"SAME"},
{"name":"l5conv3","type":"conv","size":[ 5,5,16,120],"strides":[1,1,1,1],"padding":"VALID"},
{"name":"reshape","type":"shape"},
{"name":"l6fcon1","type":"fcon","size":[120,84]},
{"name":"l7fcon2","type":"fcon","size":[84,10]}
]

batch_size=100
learning_rate_base=0.2
learning_rate_decay=0.99
regularization_rate=0.0001
training_steps=30000
moving_average_decay=0.99

model_save_path='./model/'
model_name="model.ckpt"

def train(mnist):
	x=tf.placeholder(tf.float32,
					[
					batch_size,
					lenet[0]['size'][0],
					lenet[0]['size'][1],
					lenet[0]['size'][2]
					],
					name='x-input')
	y_=tf.placeholder(tf.float32,[None,lenet[-1]['size'][1]],name='y-input')

	regularizer=tf.contrib.layers.l2_regularizer(regularization_rate)
	y=cnn.inference(lenet,x,regularizer,train=True)

	global_step=tf.Variable(0,trainable=False)
	variable_averages=tf.train.ExponentialMovingAverage(moving_average_decay,global_step)
	variable_averages_op=variable_averages.apply(tf.trainable_variables())

	cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
	cross_entropy_mean=tf.reduce_mean(cross_entropy)

	loss=cross_entropy_mean+tf.add_n(tf.get_collection('losses'))

	learning_rate=tf.train.exponential_decay(learning_rate_base,global_step,mnist.train.num_examples/batch_size,learning_rate_decay)
	train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)

	with tf.control_dependencies([train_step,variable_averages_op]):
		train_op=tf.no_op(name='train')

	saver=tf.train.Saver()
	with tf.Session() as sess:
		tf.global_variables_initializer().run()


		for i in range(training_steps):
			xs,ys=mnist.train.next_batch(batch_size)
			reshaped_xs=np.reshape(xs,(batch_size,
					lenet[0]['size'][0],
					lenet[0]['size'][1],
					lenet[0]['size'][2]))
			_None,loss_value,step=sess.run([train_op,loss,global_step],feed_dict={x:reshaped_xs,y_:ys})

			if i%1000==0:
				print("Steps:%d,Loss:%g"%(step,loss_value))
				saver.save(sess,os.path.join(model_save_path,model_name),global_step=global_step)

def main(argv=None):
	mnist=input_data.read_data_sets('../MNIST_data/',one_hot=True)
	train(mnist)

if __name__=='__main__':
	tf.app.run()
