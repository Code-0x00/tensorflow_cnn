import tensorflow as tf

xNode=[784,500,10]
xLayer=['layer1','layer2']

input_node=xNode[0]
output_node=xNode[-1]

def get_weight_variable(shape,regularizer):
	weights=tf.get_variable("weights",shape,initializer=tf.truncated_normal_initializer(stddev=0.1))
	if regularizer!=None:
		tf.add_to_collection("losses",regularizer(weights))
	return weights

def inference(input_tensor,regularizer):
	layer_out=input_tensor
	for i in range(len(xLayer)):
		with tf.variable_scope(xLayer[i]):
			weights=get_weight_variable([xNode[i],xNode[i+1]],regularizer)
			biases=tf.get_variable("biases",[xNode[i+1]],initializer=tf.constant_initializer(0.0))
			layer_out=tf.nn.relu(tf.matmul(layer_out,weights)+biases)
	return layer_out