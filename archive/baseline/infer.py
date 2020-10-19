
# MobileNetV2的推理脚本

import os
import tensorflow as tf
import numpy as np
import scipy.misc
import cv2
import sys
from my_model import MobileNet
import time

# 指定设备，使用哪块显存
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# 类别：索引字典
class_2_index = {0: 'smoking', 1: 'calling',2:'smoking_calling',3:'normal'}
#类别数量
n_classes = 4
## 输入图片resize为统一尺寸，与训练过程大小相同
x = tf.placeholder(tf.float32, [None, 224 * 224 * 3])
image_tensor = tf.reshape(x, [-1, 224, 224, 3])
#实例化MobileNet模型
MN = MobileNet(train_status=False)
pred_logit = MN.get_model(image_tensor, n_classes)
#配置，允许加速等、设置显存占用比
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.02  # 分配50%
sess = tf.InteractiveSession(config=config)
saver = tf.train.Saver()

# 加载权重文件
if tf.train.latest_checkpoint('/notebooks/HUALU/MobileNetV2/checkpoints') is not None:
    saver.restore(sess,
                  tf.train.latest_checkpoint('/notebooks/HUALU/MobileNetV2/checkpoints'))
    print(tf.train.latest_checkpoint('/notebooks/HUALU/MobileNetV2/checkpoints'))

else:
    assert 'can not find checkpoint folder path !'
    print('can not find checkpoints')

# 测试的图片路径
folder = '/notebooks/HUALU/MobileNetV2/my_data/test/'

#遍历测试文件夹，加载图片、图片预处理、预测类别、输出
result_list = []
for roots, dirs, files in os.walk(folder):
    for dir in dirs:
        dir_list = os.path.join(folder, dir)
        for root1, dirs1, files1 in os.walk(dir_list):
            for f in files1:
                
                # 加载图片
                image_path = os.path.join(dir_list, f)
                #图片预处理
                img = cv2.cvtColor(cv2.imread(image_path, -1), cv2.COLOR_BGR2RGB)
                
                img = np.resize(img, (224, 224, 3))
                img = img / 255.
                img = img.flatten()
                img = img[np.newaxis, :]

                # 识别写入list中
                res = sess.run(pred_logit, feed_dict={x: img})
                name = class_2_index[np.argmax(res[0])]
                result_data = {"image_name": str(f), "category": name, "score": 0.94135}
                result_list.append(result_data)
print(result_list)


