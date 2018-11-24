# coding: utf-8
import json
import os
import numpy as np
import datetime

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # so the IDs match nvidia-smi
os.environ["CUDA_VISIBLE_DEVICES"] = "1" # "0, 1" for multiple

import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
tf.keras.backend.set_session(session)

from callbacks import *
from config import Config
from dataset import classify_sequence
from model import Model

model_config = Config()
IS_SAVE = True

model_name = 'classify'
im_name = 'split'
data_version = 'v5'
class_weights=[0.2,0.5,1,0.8]
# class_weights=[0.2,0.5,1,0.8]

base_dir = './logs'
# model_path = './logs/res50_softmax_momentum_20180722T0055/res50_softmax_momentum_0047.h5'
model_path = None

im_dir = os.path.join('../dataset/bodytype/',im_name)

train_json = '../dataset/bodytype/women_' + data_version + '_train.json'
val_json = '../dataset/bodytype/women_' + data_version + '_val.json'

f = open(train_json,'r')
train_f = json.load(f)
train_nums = len(train_f.keys())
f.close()

f = open(val_json,'r')
val_f = json.load(f)
val_nums = len(val_f.keys())
f.close()

train_steps = int(train_nums/model_config.param['TRAIN_STEPS'])
val_steps = int(val_nums/model_config.param['VALIDATION_STEPS'])

model_config.set_param(['TRAIN_STEPS','VALIDATION_STEPS'],[train_steps,val_steps])
model_config.show_config()

model = Model(model_path,model_name,model_config)

now = datetime.datetime.now()

log_dir = os.path.join(base_dir, "{}_{}_{:%Y%m%dT%H%M}".format(model_name,data_version,now))
checkpoint_path = os.path.join(log_dir, "ep_*epoch*.h5")
checkpoint_path = checkpoint_path.replace("*epoch*", "{epoch:04d}")

if IS_SAVE:
    callbacks = [
            LRTensorBoard(log_dir=log_dir,
                    histogram_freq=0, write_graph=True, write_images=False),
            # tf.keras.callbacks.TensorBoard(log_dir=log_dir,
            #          histogram_freq=0, write_graph=True, write_images=False),
            tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                    verbose=0, save_weights_only=True),
#             tf.keras.callbacks.LearningRateScheduler(lrdecay),
        ]
    model_config.save_config(out_dir=log_dir)
else:
    callbacks = [
#             tf.keras.callbacks.LearningRateScheduler(lrdecay),
        ]

train_generator = classify_sequence(im_dir,train_json,model_config,class_weights)
val_generator = classify_sequence(im_dir,val_json,model_config,class_weights)

model.train(train_generator,val_generator,callbacks)