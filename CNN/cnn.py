import tensorflow as tf

xNode=[784,500,10]
xLayer=['layer1','layer2']

input_node=xNode[0]
output_node=xNode[-1]

image_size=28
num_channels=1
num_labels=10

conv1_deep=32
conv1_size=5

conv2_deep=64
conv2_size=5

fc_size=512
lenet=[
{"name":"l1conv1","type":"conv","size":[ 5, 5, 1, 6],"strides":[1,1,1,1],"padding":"SAME"},#k_size,k_size,channel,deep
{"name":"l2pool1","type":"pool","size":[ 1, 2, 2, 1],"strides":[1,2,2,1],"padding":"SAME"},
{"name":"l3conv2","type":"conv","size":[ 5, 5, 6,16],"strides":[1,1,1,1],"padding":"VALID"},
{"name":"l4pool2","type":"pool","size":[ 1, 2, 2, 1],"strides":[1,2,2,1],"padding":"SAME"},
{"name":"l5conv3","type":"conv","size":[ 5,5,16,120],"strides":[1,1,1,1],"padding":"VALID"},
{"name":"reshape","type":"shape"},
{"name":"l6fcon1","type":"fcon","size":[120,84]},
{"name":"l7fcon2","type":"fcon","size":[84,10]}
]
def inference(input_tensor,regularizer,train):
	last_layer_output=input_tensor
	print 'l0data',last_layer_output.shape
	for layer in lenet:
		if layer["type"]=="conv":
			with tf.variable_scope(layer['name']):
				weights=tf.get_variable("weight",layer['size'],initializer=tf.truncated_normal_initializer(stddev=0.1))
				biases=tf.get_variable("bias",layer['size'][3],initializer=tf.constant_initializer(0.0))
				conv=tf.nn.conv2d(last_layer_output,weights,strides=layer['strides'],padding=layer['padding'])
				relu=tf.nn.relu(tf.nn.bias_add(conv,biases))
				last_layer_output=relu
		elif layer['type']=='fcon':
			with tf.variable_scope(layer['name']):
				weights=tf.get_variable('weight',layer['size'],initializer=tf.truncated_normal_initializer(stddev=0.1))
				if regularizer!=None:
					tf.add_to_collection('losses',regularizer(weights))
				biases=tf.get_variable('bias',layer['size'][1],initializer=tf.constant_initializer(0.1))
				relu=tf.nn.relu(tf.matmul(last_layer_output,weights)+biases)
				last_layer_output=relu
		elif layer['type']=='pool':
			with tf.name_scope(layer['name']):
				pool=tf.nn.max_pool(last_layer_output,ksize=layer['size'],strides=layer['strides'],padding=layer['padding'])
				last_layer_output=pool
		elif layer['type']=='shape':
			pool_shape=last_layer_output.get_shape().as_list()
			nodes=pool_shape[1]*pool_shape[2]*pool_shape[3]
			reshaped=tf.reshape(last_layer_output,[pool_shape[0],nodes])
			last_layer_output=reshaped

		print layer['name'],last_layer_output.shape
	return last_layer_output
