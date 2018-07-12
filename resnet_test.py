# coding: utf-8

import tensorflow as tf
import json
import os
import numpy as np
import datetime
import cv2
import random
import re
import multiprocessing

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # so the IDs match nvidia-smi
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # "0, 1" for multiple

# adam, momentum or nesterov
opt_string = "momentum"

LEARNING_RATE = 0.001
#using in lr decay
LR_DECAY = 0.95
#using in momentum and nesterov
MOMENTUM = 0.9
#using in adam
ADAM_BETA_1 = 0.9
ADAM_BETA_2 = 0.99

#base hyper-parameter
INPUT_SHAPE = (1024,512,3)
MEAN_PIXEL = np.array([93.2,104.6,116.6])
BATCH_SIZE = 40
VALIDATION_STEPS = 30
CLASS_NUMS = 4
IS_SAVE = True

im_dir = './dataset/image_women/'
train_json = './dataset/women_train_label.json'
val_json = './dataset/women_val_label.json'
base_dir = './logs'

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
tf.keras.backend.set_session(session)

def build_model():
    resnet_model = tf.keras.applications.resnet50.ResNet50(include_top=False,
                                weights='imagenet',input_shape=INPUT_SHAPE)
    resnet_output = resnet_model.output

    resnet_input = resnet_model.input

    fla_fm = tf.keras.layers.Flatten()(resnet_output)

    output = tf.keras.layers.Dense(CLASS_NUMS,activation='softmax',name='fc_softmax')(fla_fm)

    sm_model = tf.keras.models.Model(inputs = resnet_input,outputs = output, name = 'res50_softmax')

    return sm_model

def set_trainable(pattern, keras_model=None, verbose=1):
    """Sets model layers as trainable if their names match
    the given regular expression.
    """
    print("\nIn:",keras_model.name)
    print("The trainable layers:")
    # In multi-GPU training, we wrap the model. Get layers
    # of the inner model because they have the weights.
    layers = keras_model.layers
    for layer in layers:
        # Is the layer a model?
        if layer.__class__.__name__ == 'Model':
            set_trainable(
                pattern, keras_model=layer)
            continue

        if not layer.weights:
            continue
        # Is it trainable?
        trainable = bool(re.fullmatch(pattern, layer.name))
        # Update layer. If layer is a container, update inner layer.
        layer.trainable = trainable
        # Print trainble layer names
        if trainable and verbose > 0:
            print(" ",layer.name)

def imread(path):
    im = cv2.imread(path)
    if im is None:
        return None
    
    im = im/255
    
    im_pad = np.zeros(INPUT_SHAPE,dtype=np.float64)
    h,w = im.shape[:2]
    if h/w > INPUT_SHAPE[0]/INPUT_SHAPE[1]:
        re_h = INPUT_SHAPE[0]
        re_w = int(w * (re_h / h))
    else:
        re_w = INPUT_SHAPE[1]
        re_h = int(h * (re_w / w))
    re_im = cv2.resize(im,(re_w,re_h))
    im_pad[:re_h,:re_w,:] = re_im.copy()
    return im_pad
            
# class threadsafe_iter:
#     """Takes an iterator/generator and makes it thread-safe by
#     serializing call to the `next` method of given iterator/generator.
#     """
#     def __init__(self, it):
#         self.it = it
#         self.lock = threading.Lock()

#     def __iter__(self):
#         return self

#     def __next__(self): # Py3
#         with self.lock:
#             return next(self.it)

# def threadsafe_generator(f):
#     """A decorator that takes a generator function and makes it thread-safe.
#     """
#     def g(*a, **kw):
#         return threadsafe_iter(f(*a, **kw))
#     return g

# @threadsafe_generator
# def generator(im_dir, json_path, batch_size):
#     #the generator will full all cpu who can use, and stop the process
#
#     index = 0
#     f = open(json_path, 'r')
#     info = json.load(f)
#     f.close()
#     names = list(info.keys())
#     random.shuffle(names)
#     i = 0
#     while True:
#         try:
#             if i == 0:
#                 images = []
#                 labels = []
#             name = names[index]
#             path = os.path.join(im_dir, name)
#             im = imread(path)
#             if im is None:
#                 continue
#
#             label = info[name]['label']
#
#             images.append(im)
#             labels.append(label)
#             i += 1
#             index += 1
#             if index == len(names):
#                 index = 0
#                 random.shuffle(names)
#             if i >= batch_size:
#                 images = np.array(images)
#                 labels = np.array(labels)
#                 labels = (np.arange(4) == labels[:, None]).astype(np.integer)
#                 yield (images, labels)
#                 i = 0
#         except Exception as e:
#             raise

class mysequence(tf.keras.utils.Sequence):
    def __init__(self, im_dir, json_path, batch_size):
        self.batch_size = batch_size
        self.im_dir = im_dir
        f = open(json_path,'r')
        self.info = json.load(f)
        self.names = list(self.info.keys())
        random.shuffle(self.names)
        f.close()
    
    def __len__(self):
        return int(len(self.names) / self.batch_size) # the length is the number of batches
    
    def on_epoch_end(self):
        random.shuffle(self.names)
    
    def __getitem__(self, batch_id):
        images = []
        labels = []
        for i in range(batch_id * self.batch_size, (batch_id+1) * self.batch_size):
            names_num = len(self.names)
            index = int(i % names_num)
            name = self.names[index]
            path = os.path.join(self.im_dir, name)
            im = imread(path)
            if im is None:
                continue
            
            label = self.info[name]['label']
            
            images.append(im)
            labels.append(label)
        images = np.array(images)
        labels = np.array(labels)
        #labels = (np.arange(CLASS_NUMS) == labels[:, None]).astype(np.float32)
        labels = tf.keras.utils.to_categorical(labels,num_classes = CLASS_NUMS)
        return images, labels
            
class LRTensorBoard(tf.keras.callbacks.TensorBoard):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def on_epoch_end(self, epoch, logs=None):
        logs.update({'lr': tf.keras.backend.eval(self.model.optimizer.lr)})
        super().on_epoch_end(epoch, logs)
        

def lrdecay(epoch):
    lr_new = LEARNING_RATE * LR_DECAY ** epoch
    return lr_new        
        
layer_dict = {
            # all layers but the backbone
            "heads": r"(fc.*)",
            # From a specific Resnet stage and up
            "3+": r"(res3.*)|(bn3.*)|(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(fc.*)",
            "4+": r"(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(fc.*)",
            "5+": r"(res5.*)|(bn5.*)|(fc.*)",
            # All layers
            "all": ".*",
        }

model = build_model()
train_layer = layer_dict['heads']
set_trainable(train_layer,model)
if opt_string == "momentum":
    opt = tf.keras.optimizers.SGD(lr=LEARNING_RATE,momentum=MOMENTUM)
if opt_string == "nesterov":
    opt = tf.keras.optimizers.SGD(lr=LEARNING_RATE,momentum=MOMENTUM,nesterov=True)
if opt_string == "adam":
    opt = tf.keras.optimizers.Adam(lr=LEARNING_RATE,beta_1=ADAM_BETA_1,beta_2=ADAM_BETA_2)
model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['categorical_accuracy'])
#model.compile(optimizer=opt,loss='sparse_categorical_crossentropy',metrics=['sparse_categorical_accuracy'])

# train_generator = generator(im_dir,train_json,BATCH_SIZE)
train_generator = mysequence(im_dir, train_json, BATCH_SIZE)
# val_generator = generator(im_dir,val_json,BATCH_SIZE)
val_generator = mysequence(im_dir, val_json, BATCH_SIZE)

now = datetime.datetime.now()

base_name = 'res50_softmax_' + opt_string
log_dir = os.path.join(base_dir, base_name+"_{:%Y%m%dT%H%M}".format(now))
checkpoint_path = os.path.join(log_dir, base_name+"_*epoch*.h5")
checkpoint_path = checkpoint_path.replace("*epoch*", "{epoch:04d}")

if IS_SAVE:
    callbacks = [
            LRTensorBoard(log_dir=log_dir,
                    histogram_freq=0, write_graph=True, write_images=False),
            # tf.keras.callbacks.TensorBoard(log_dir=log_dir,
            #          histogram_freq=0, write_graph=True, write_images=False),
            tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                    verbose=0, save_weights_only=True),
            tf.keras.callbacks.LearningRateScheduler(lrdecay),
        ]
else:
    callbacks = [
            tf.keras.callbacks.LearningRateScheduler(lrdecay),
        ]



model.fit_generator(train_generator,
                   steps_per_epoch = 100,
                   epochs = 100,
                   validation_data=val_generator,
                   validation_steps=VALIDATION_STEPS,
                   callbacks=callbacks,
                   workers = multiprocessing.cpu_count(),
                   max_queue_size = 10,
                   shuffle = True,
                   use_multiprocessing = True)