
# MobileNetV2的训练脚本

from my_model import MobileNet
from my_data import Data
import tensorflow as tf
import numpy as np
import scipy.misc
import random
import sys
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


if __name__ == '__main__':
	# 指定设备，使用哪块显存
	os.environ['CUDA_VISIBLE_DEVICES'] = '6'
	save_path = os.path.join('/notebooks/HUALU/MobileNetV2/checkpoints', 'model.ckpt')
	
	# 设置模型训练过程中的学习率、循环次数、类别数、每次训练放入模型的图片数量
	lr = 0.001
	train_loops = 10000
	n_classes = 4
	batch_size = 1
	#加载数据，调用my_data.py,实例化数据处理
	data = Data('/notebooks/HUALU/MobileNetV2/my_data/', 'class_to_index.log')
	#训练的次数、每步训练的轮次
	epochs = 5
	steps_per_epoch = int(len(data.train_data) / batch_size)
	# 输入图片resize为统一尺寸
	x = tf.placeholder(tf.float32, [None, 224*224*3])
	y = tf.placeholder(tf.float32, [None, n_classes])
	input_tensor = tf.reshape(x, [-1, 224, 224, 3])
	#实例化MobileNet模型
	MN = MobileNet(train_status=True)
	logit = MN.get_model(input_tensor, n_classes)
	#loss函数
	cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = logit))
	#预测的类别
	pred_label = tf.argmax(logit, 1)
	label = tf.argmax(y, 1)
	#准确率
	accuracy = tf.reduce_mean(tf.cast(tf.equal(pred_label, label), tf.float32))
	#优化器
	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	opt = tf.train.AdamOptimizer(learning_rate = lr, name = 'optimizer')
	#计算前向传播过程中的梯度值
	with tf.control_dependencies(update_ops):
		grads = opt.compute_gradients(cross_entropy_loss)
		train_op = opt.apply_gradients(grads)
	#配置，允许加速等
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	config.allow_soft_placement = True
	sess = tf.InteractiveSession(config = config)
	sess.run(tf.global_variables_initializer())
	
	#控制训练参数
	var_list = tf.trainable_variables()
	g_list = tf.global_variables()
	bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
	bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
	var_list += bn_moving_vars
	
	#开始训练
	generator = data.get_batch(batch_size)
	val_generator = data.get_validate_batch(batch_size)
	saver = tf.train.Saver(var_list = var_list, max_to_keep = 5)
	# 记录训练过程中的相关参数：训练集上的loss、准确度、验证集上的loss、准确度
	train_avg_loss_per_epoch = []
	train_avg_acc_per_epoch = []
	val_avg_loss_per_epoch = []
	val_avg_acc_per_epoch = []
	#每一步训练
	for epoch in range(epochs):
		print(epoch)
		train_loss_list = []
		train_acc_list = []
		val_loss_list = []
		val_acc_list = []
		#每个伦次
		for step in range(steps_per_epoch):
			#获取每个batch的训练数据、标签，验证数据、标签
			batch_images, batch_labels = generator.__next__()
			batch_val_images, batch_val_labels = val_generator.__next__()
			#计算当前伦次训练集上、验证集上的loss值及准确度
			_, loss, acc = sess.run([train_op, cross_entropy_loss, accuracy], feed_dict = {x: batch_images, y: batch_labels})
			val_loss, val_acc = sess.run([cross_entropy_loss, accuracy], feed_dict = {x: batch_val_images, y: batch_val_labels})
			#记录当前伦次的loss、准确度
			train_acc_list.append(acc)
			train_loss_list.append(loss)
			val_acc_list.append(val_acc)
			val_loss_list.append(val_loss)
			#输出，同时写入log
			print('loss:'+str(loss)+","+"acc:"+str(acc)+','+"val_loss:"+str(val_loss)+','+'val_acc:'+str(val_acc))
			with open('train_record.log', 'a') as out:
				out.write('Epoch: ' + str(epoch) + ', Step: ' + str(step) + '/' + str(steps_per_epoch) + ', Loss: ' + str(loss) + ', ' + 'accuracy: ' + str(acc) + ', ' + 'valid loss: ' + str(val_loss) + ', ' + 'valid accuracy: ' + str(val_acc))
				out.write('\n')
			#每训练100步，保存一次训练文件
			if (epoch*steps_per_epoch + step) % 100 == 0 and (epoch*steps_per_epoch + step) > 0:
				saver.save(sess, save_path, global_step = (epoch*steps_per_epoch + step))
			else:
				pass
		train_avg_loss_per_epoch.append(np.mean(train_loss_list))
		train_avg_acc_per_epoch.append(np.mean(train_acc_list))
		val_avg_loss_per_epoch.append(np.mean(val_loss_list))
		val_avg_acc_per_epoch.append(np.mean(val_acc_list))

		# 训练过程的准确度、loss画图保存
		fig1, ax1 = plt.subplots(figsize=(11, 8))
		ax1.plot(range(epoch + 1), train_avg_loss_per_epoch)
		# ax1.plot(range(1), avg_scores_per_epoch)
		ax1.set_title("Train: Average loss vs epochs")
		ax1.set_xlabel("Epoch")
		ax1.set_ylabel("Current loss")
		plt.savefig('train_loss_vs_epochs.png')
		plt.clf()

		fig2, ax2 = plt.subplots(figsize=(11, 8))
		ax2.plot(range(epoch + 1), train_avg_acc_per_epoch)
		ax2.set_title("Train: Average acc vs epochs")
		ax2.set_xlabel("Epoch")
		ax2.set_ylabel("Avg. train. accuracy")
		plt.savefig('train_acc_vs_epochs.png')
		plt.clf()

		fig3, ax3 = plt.subplots(figsize=(11, 8))
		ax3.plot(range(epoch + 1), val_avg_loss_per_epoch)
		# ax1.plot(range(1), avg_scores_per_epoch)
		ax3.set_title("Val: Average loss vs epochs")
		ax3.set_xlabel("Epoch")
		ax3.set_ylabel("Current loss")
		plt.savefig('val_loss_vs_epochs.png')
		plt.clf()

		fig4, ax4 = plt.subplots(figsize=(11, 8))
		ax4.plot(range(epoch + 1), val_avg_acc_per_epoch)
		ax4.set_title("val: Average acc vs epochs")
		ax4.set_xlabel("Epoch")
		ax4.set_ylabel("Avg. val. accuracy")
		plt.savefig('val_acc_vs_epochs.png')
		plt.clf()
	sess.close()














