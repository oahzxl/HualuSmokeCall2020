
# MobileNetV2 的模型搭建脚本

import tensorflow as tf


class MobileNet(object):
	def __init__(self, train_status):
		print('Done creating MobileNet instance')
		self.train_status = train_status
	
	def mobile_block(self, x, filter1, filter2, name_postfix):
		net = tf.layers.separable_conv2d(x, filters = filter1, kernel_size = (3,3), strides = (1,1), padding = 'same', activation = tf.nn.relu, name = 'separable_conv2d_' + name_postfix + '_1')
		net = tf.layers.batch_normalization(net, training = self.train_status, name = 'bn_' + name_postfix + '_1')
		net = tf.layers.separable_conv2d(net, filters = filter1, kernel_size = (1,1), strides = (1,1), padding = 'same', activation = tf.nn.relu, name = 'separable_conv2d_' + name_postfix + '_2')
		net = tf.layers.batch_normalization(net, training = self.train_status, name = 'bn_' + name_postfix + '_2')
		net = tf.layers.separable_conv2d(net, filters = filter2, kernel_size = (3,3), strides = (2,2), padding = 'same', activation = tf.nn.relu, name = 'separable_conv2d_' + name_postfix + '_3')
		net = tf.layers.batch_normalization(net, training = self.train_status, name = 'bn_' + name_postfix + '_3')
		net = tf.layers.separable_conv2d(net, filters = filter2*2, kernel_size = (1,1), strides = (1,1), padding = 'same', activation = tf.nn.relu, name = 'separable_conv2d_' + name_postfix + '_4')
		net = tf.layers.batch_normalization(net, training = self.train_status, name = 'bn_' + name_postfix + '_4')
		return net
	
	def final_conv_block(self, x, name_postfix):
		#print('x shape: ', x.get_shape().as_list())
		net = tf.layers.separable_conv2d(x, filters = 512, kernel_size = (3,3), strides = (2,2), padding = 'same', activation = tf.nn.relu, depthwise_initializer = tf.truncated_normal_initializer(stddev=0.1), pointwise_initializer = tf.truncated_normal_initializer(stddev=0.1), name = 'seperable_conv2d_' + name_postfix + '_1')
		net = tf.layers.batch_normalization(net, training = self.train_status, name = 'bn_' + name_postfix + '_1')
		net = tf.layers.separable_conv2d(net, filters = 1024, kernel_size = (1,1), strides = (1,1), padding = 'same', activation = tf.nn.relu, depthwise_initializer = tf.truncated_normal_initializer(stddev=0.1), pointwise_initializer = tf.truncated_normal_initializer(stddev=0.1), name = 'seperable_conv2d_' + name_postfix + '_2')
		net = tf.layers.batch_normalization(net, training = self.train_status, name = 'bn_' + name_postfix + '_2')
		net = tf.layers.separable_conv2d(net, filters = 1024, kernel_size = (3,3), strides = (1,1), padding = 'same', activation = tf.nn.relu, depthwise_initializer = tf.truncated_normal_initializer(stddev=0.1), pointwise_initializer = tf.truncated_normal_initializer(stddev=0.1), name = 'seperable_conv2d_' + name_postfix + '_3')
		net = tf.layers.batch_normalization(net, training = self.train_status, name = 'bn_' + name_postfix + '_3')
		net = tf.layers.separable_conv2d(net, filters = 1024, kernel_size = (1,1), strides = (1,1), padding = 'same', activation = tf.nn.relu, depthwise_initializer = tf.truncated_normal_initializer(stddev=0.1), pointwise_initializer = tf.truncated_normal_initializer(stddev=0.1), name = 'seperable_conv2d_' + name_postfix + '_4')
		net = tf.layers.batch_normalization(net, training = self.train_status, name = 'bn_' + name_postfix + '_4')
		return net
	
	def separable_filters(self, net, name_postfix):
		for i in range(5):
			net = tf.layers.separable_conv2d(net, filters = 512, kernel_size = (3,3), strides = (1,1), padding = 'same', activation = tf.nn.relu, depthwise_initializer = tf.truncated_normal_initializer(stddev=0.1), pointwise_initializer = tf.truncated_normal_initializer(stddev=0.1), name = 'seperable_conv2d_' + name_postfix + '_' + str(i) + '_1')
			net = tf.layers.batch_normalization(net, training = self.train_status, name = 'bn_' + name_postfix + '_' + str(i) + '_1')
			net = tf.layers.separable_conv2d(net, filters = 512, kernel_size = (1,1), strides = (1,1), padding = 'same', activation = tf.nn.relu, depthwise_initializer = tf.truncated_normal_initializer(stddev=0.1), pointwise_initializer = tf.truncated_normal_initializer(stddev=0.1), name = 'seperable_conv2d_' + name_postfix + '_' + str(i) + '_2')
			net = tf.layers.batch_normalization(net, training = self.train_status, name = 'bn_' + name_postfix + '_' + str(i) + '_2')
		return net
	
	def pool_and_classify(self, x, n_classes, name_postfix):
		net = tf.layers.average_pooling2d(x, pool_size = (7,7), strides = (1,1), name = 'average_pooling2d_' + name_postfix)
		net = tf.contrib.layers.flatten(net)
		#net = tf.contrib.layers.flatten(x)
		print('flatten: ', net)
		weights = tf.get_variable(shape = [net.shape[-1], n_classes], dtype = tf.float32, initializer = tf.truncated_normal_initializer(stddev=0.1), name = 'fc_weights_' + name_postfix)
		biases = tf.get_variable(shape = [n_classes], dtype = tf.float32, initializer = tf.constant_initializer(0.0), name = 'fc_biases_' + name_postfix)
		output = tf.nn.bias_add(tf.matmul(net, weights), biases, name = 'logit_output_' + name_postfix)
		return output
	
	def get_model(self, input_tensor, n_classes):
		net = tf.layers.conv2d(input_tensor, filters = 32, kernel_size = (3,3), strides = (2,2), padding = 'same', name = 'first_conv2d_layer')
		net = self.mobile_block(net, filter1 = 32, filter2 = 64, name_postfix = 'first_mobile_block')
		net = self.mobile_block(net, filter1 = 128, filter2 = 128, name_postfix = 'second_mobile_block')
		net = self.mobile_block(net, filter1 = 256, filter2 = 256, name_postfix = 'third_mobile_block')
		print('mobile blocks returned: ', net)
		net = self.separable_filters(net, 'separable_filters')
		print('separable filters returned: ', net)
		net = self.final_conv_block(net, 'final_conv_block')
		net = self.pool_and_classify(net, n_classes, 'pool_and_classify')
		return net
