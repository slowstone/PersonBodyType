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
from dataset import surreal_sequence,regress_sequence
from model import Model

model_config = Config()
IS_SAVE = True

model_name = model_config.param['MODEL_NAME']
data_version = model_config.param['DATA_VERSION']

im_dir = '../dataset/SURREAL/summary/human'
label_dir = '../dataset/SURREAL/summary/labels'
base_dir = './logs'

train_json_path = '../dataset/bodytype/SURREAL/train_names_sort.json'
train_generator = surreal_sequence(train_json_path,im_dir,label_dir,model_config)
# val_json_path = '../dataset/bodytype/dataset_up_val.json'
# val_generator = regress_sequence('../dataset/bodytype/up-3d-box/',val_json_path,model_config)
val_json_path = '../dataset/bodytype/SURREAL/val_names_sort.json'
val_generator = surreal_sequence(val_json_path,im_dir,label_dir,model_config)

model_config.set_param(['TRAIN_JSON_PATH','VAL_JSON_PATH'],[train_json_path,val_json_path])

train_nums = train_generator.names_num
val_nums = val_generator.names_num

print("====================dataset scale====================")
print("=========> Train dataset: {},{}".format(train_json_path,train_nums))
print("=========> Val dataset: {},{}".format(val_json_path,val_nums))
print("=====================================================\n")

# train_steps = int(train_nums/model_config.param['BATCH_SIZE'])
train_steps = 10000
val_steps = int(val_nums/model_config.param['BATCH_SIZE'])
model_config.set_param(['TRAIN_STEPS','VALIDATION_STEPS'],[train_steps,val_steps])

model_config.show_config()

model = Model(model_config)

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

model.train(train_generator,val_generator,callbacks)
