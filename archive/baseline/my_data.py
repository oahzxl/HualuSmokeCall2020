
# 数据预处理部分

from resnets_utils import *
import numpy as np
import scipy.misc
import random
import os
import cv2


class Data(object):
    def __init__(self, dataset_root, class_to_index_log):
        self.dataset_root = dataset_root
        self.classes = self.get_classes()
        self.name_to_index = dict(zip(self.classes, range(len(self.classes))))
        with open(class_to_index_log, 'a') as log:
            log.write(str(self.name_to_index))
            log.write('\n')
        self.train_data = self.collect_data(
            [os.path.join(self.dataset_root, 'train', cls) for cls in self.classes])
        self.validate_data = self.collect_data(
            [os.path.join(self.dataset_root, 'test', cls) for cls in self.classes])
        self.train_num, self.validate_num = len(
            self.train_data), len(self.validate_data)
        self.train_start = 0
        self.train_end = 0
        self.validate_start = 0
        self.validate_end = 0

    def get_classes(self):
        return os.listdir(os.path.join(self.dataset_root, 'train'))

    def collect_data(self, folder_list):
        data = []
        for ind in range(len(folder_list)):
            folder = folder_list[ind]
            for root, dirs, files in os.walk(folder):
                for f in files:
                    image_path = os.path.join(root, f)
                    data.append((image_path, ind))
        random.shuffle(data)
        return data

    def get_batch(self, batch_size):
        while True:
            if self.train_start + batch_size <= self.train_num:
                self.train_end = self.train_start + batch_size
                batch = self.train_data[self.train_start:self.train_end]
                self.train_start += batch_size
            else:
                self.train_start = 0
                self.train_end = batch_size
                batch = self.train_data[self.train_start:self.train_end]
                self.train_start += batch_size
            # batch_images = [cv2.imread(x[0]) for x in batch]

            batch_images = [cv2.cvtColor(cv2.imread(x[0], -1), cv2.COLOR_BGR2RGB) for x in batch]
            batch_images = [np.resize(img, (224, 224, 3)) for img in batch_images]
            batch_images = [img / 255. for img in batch_images]
            batch_images = [img.flatten() for img in batch_images]
            batch_images = np.array(batch_images)
            batch_labels = [x[1] for x in batch]
            batch_labels = np.array(batch_labels)
            batch_labels = batch_labels.reshape((1, batch_labels.shape[0]))
            batch_labels = convert_to_one_hot(
                batch_labels, len(self.classes)).T
            yield batch_images, batch_labels

    def get_validate_batch(self, batch_size):
        while True:
            if self.validate_start + batch_size <= self.validate_num:
                self.validate_end = self.validate_start + batch_size
                batch = self.validate_data[self.validate_start: self.validate_end]
                self.validate_start += batch_size
            else:
                self.validate_start = 0
                self.validate_end = batch_size
                batch = self.validate_data[self.validate_start: self.validate_end]
                self.validate_start += batch_size
           # batch_images = [cv2.imread(x[0]) for x in batch]
            batch_images = [cv2.cvtColor(cv2.imread(x[0], -1), cv2.COLOR_BGR2RGB) for x in batch]
            batch_images = [np.resize(img, (224, 224, 3)) for img in batch_images]
            batch_images = [img/255. for img in batch_images]
            batch_images = [img.flatten() for img in batch_images]
            batch_images = np.array(batch_images)
            batch_labels = [x[1] for x in batch]
            batch_labels = np.array(batch_labels)
            batch_labels = batch_labels.reshape((1, batch_labels.shape[0]))
            batch_labels = convert_to_one_hot(
                batch_labels, len(self.classes)).T
            yield batch_images, batch_labels
