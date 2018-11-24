import json
import os
import numpy as np
import cv2
import tensorflow as tf
import random
from config import Config

class classify_sequence(tf.keras.utils.Sequence):
    def __init__(self, im_dir, json_path,config = Config(),class_weights=[1,1,1,1]):
        self.config = config
        self.class_weights = class_weights
        self.im_dir = im_dir
        f = open(json_path,'r')
        self.info = json.load(f)
        self.names = list(self.info.keys())
        self.offset = 0
        random.shuffle(self.names)
        f.close()
        self.test = {}
    
    def __len__(self):
        return int(len(self.names) / self.config.param['BATCH_SIZE']) # the length is the number of batches
    
    def on_epoch_end(self):
        self.offset = 0
        random.shuffle(self.names)
    
    def imread(self,path):
        im = cv2.imread(path)
        if im is None:
            return None

        im = im/255
        im_shape = self.config.param['INPUT_SHAPE']
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
    
    def __getitem__(self, batch_id):
        images = []
        labels = []
        for i in range(batch_id * self.config.param['BATCH_SIZE'], (batch_id+1) * self.config.param['BATCH_SIZE']):
            names_num = len(self.names)
            while True:
                index = int((i+self.offset) % names_num)
                name = self.names[index]
                path = os.path.join(self.im_dir, name)
                               
                im = self.imread(path)
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