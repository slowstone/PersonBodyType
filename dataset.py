import json
import os
import numpy as np
import cv2
import tensorflow as tf
import random
from config import Config

def imread(path,im_shape):
    im = cv2.imread(path)
    if im is None:
        return None

    im = im/255
    im_pad = np.zeros(im_shape,dtype=np.float64)
    h,w = im.shape[:2]
    if h/w > im_shape[0]/im_shape[1]:
        re_h = im_shape[0]
        re_w = int(w * (re_h / h))
    else:
        re_w = im_shape[1]
        re_h = int(h * (re_w / w))
    re_im = cv2.resize(im,(re_w,re_h))
    im_pad[:re_h,:re_w,:] = re_im.copy()
    return im_pad

class classify_sequence(tf.keras.utils.Sequence):
    def __init__(self, im_dir, json_path,config = Config(),class_weights=[1,1,1,1]):
        self.config = config
        self.class_weights = class_weights
        self.im_dir = im_dir
        f = open(json_path,'r')
        self.info = json.load(f)
        self.names = list(self.info.keys())
        self.names_num = len(self.names)
        self.offset = 0
        random.shuffle(self.names)
        f.close()
        self.test = {}

    def __len__(self):
        return int(len(self.names) / self.config.param['BATCH_SIZE']) # the length is the number of batches

    def on_epoch_end(self):
        self.offset = 0
        random.shuffle(self.names)

    def __getitem__(self, batch_id):
        images = []
        labels = []
        for i in range(batch_id * self.config.param['BATCH_SIZE'], (batch_id+1) * self.config.param['BATCH_SIZE']):
            while True:
                index = int((i+self.offset) % self.names_num)
                name = self.names[index]
                path = os.path.join(self.im_dir, name)

                im = imread(path,self.config.param['INPUT_SHAPE'])
                if im is None:
                    self.offset += 1
                    continue
                label = self.info[name]['label']
                if random.random() > self.class_weights[label]:
                    self.offset += 1
                    continue
                images.append(im)

                if index not in self.test.keys():
                    self.test[index] = 0
                self.test[index] += 1

                labels.append(label)
                break
        images = np.array(images)
        labels = np.array(labels)
        #labels = (np.arange(CLASS_NUMS) == labels[:, None]).astype(np.float32)
        labels = tf.keras.utils.to_categorical(labels,num_classes = self.config.param['CLASS_NUMS'])
        return images, labels

class regress_sequence(tf.keras.utils.Sequence):
    def __init__(self, im_dir, json_path,config = Config()):
        self.config = config
        self.im_dir = im_dir
        f = open(json_path,'r')
        self.info = json.load(f)
        self.names = list(self.info.keys())
        self.names_num = len(self.names)
        self.offset = 0
        random.shuffle(self.names)
        f.close()

    def __len__(self):
        return int(len(self.names) / self.config.param['BATCH_SIZE']) # the length is the number of batches

    def on_epoch_end(self):
        random.shuffle(self.names)
        self.offset = 0

    def __getitem__(self, batch_id):
        images = []
        labels = []
        for i in range(batch_id * self.config.param['BATCH_SIZE'], (batch_id+1) * self.config.param['BATCH_SIZE']):
            while True:
                index = int((i+self.offset) % self.names_num)
                name = self.names[index]
                path = os.path.join(self.im_dir, name)

                im = imread(path,self.config.param['INPUT_SHAPE'])
                if im is None:
                    self.offset += 1
                    continue
                if name not in self.info.keys():
                    self.offset += 1
                    continue
                label = self.info[name]['bmi']
                images.append(im)
                labels.append(label)
                break
        images = np.array(images)
        labels = np.array(labels)
        return images, labels

class surreal_sequence(tf.keras.utils.Sequence):
    def __init__(self,json_path,im_dir,label_dir,config = Config()):
        self.config = config
        self.im_dir = im_dir
        self.label_dir = label_dir
        f = open(json_path,'r')
        self.names = json.load(f)
        f.close()
        self.names_num = len(self.names)
        self.index = 0

    def __len__(self):
        return int(len(self.names) / self.config.param['BATCH_SIZE']) # the length is the number of batches

    def __getitem__(self, batch_id):
        images = []
        labels = []
        cur_size = 0
        while cur_size < self.config.param['BATCH_SIZE']:
            self.index = self.index % self.names_num
            if self.index >= self.names_num:
                self.index = 0
            base_name = self.names[self.index]
            im_path = os.path.join(self.im_dir,base_name+'.jpg')
            label_path = os.path.join(self.label_dir,base_name+'.json')
            if not os.path.exists(label_path):
                self.index += 1
                continue
            im = imread(im_path,self.config.param['INPUT_SHAPE'])
            if im is None:
                self.index += 1
                continue
            with open(label_path,'r') as f:
                info = json.load(f)
#             label = [ i * 100 for i in info['shape'] ]
            label = info['shape']
            images.append(im)
            labels.append(label)
            self.index += 1
            cur_size += 1
        images = np.array(images)
        labels = np.array(labels)
        return images, labels
