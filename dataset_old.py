
from collections import Iterator
class datasetiter(Iterator):
    def __init__(self,arrs):
        self.index = 0
        self.arrs = arrs
    def __iter__(self):
        return self
    def __next__(self):
        output = self.arrs[self.index]
        self.index += 1
        if self.index == len(self.arrs):
            self.index = 0              
        return output
class Dataset:
    def __init__(self,im_dir=None,label_path=None,batch_size=16,input_shape=(512,512,3)):
        self.im_dir = im_dir
        self.label_path = label_path
        self.h = input_shape[0]
        self.w = input_shape[1]
        self.c = input_shape[2]
        if batch_size == 'full':
            batch_size = len(self.name_list)
        else:
            assert type(batch_size) is int,"batch_size should be int or 'full'"
            self.batch_size = batch_size
        import json
        f = open(self.label_path,'r')
        self.label = json.load(f)
        self.name_list = list(self.label.keys())
        self.iter = datasetiter(self.name_list)
    def read_im(self,path):
        import cv2
        im = cv2.imread(path)
        if im is None:
            return None
        re_im = cv2.resize(im,(self.h,self.w))
        assert re_im.shape == (self.h,self.w,self.c) , re_im.shape
        return re_im
    def next_batch(self):
        import os
        i = 0
        images = []
        labels = []
        while i < self.batch_size:
            name = next(self.iter)
            path = os.path.join(self.im_dir,name)
            im = self.read_im(path)
            if im is None:
                continue
            images.append(im)
            label = self.label[name]['label']
            labels.append(label)
            i += 1
        import numpy as np
        images = np.array(images)
        labels = np.array(labels)
        return images,labels

#####
#data generator
#####
class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self): # Py3
        with self.lock:
            return next(self.it)

def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g

@threadsafe_generator
def generator(im_dir, json_path, batch_size):
    #the generator will full all cpu who can use, and stop the process

    index = 0
    f = open(json_path, 'r')
    info = json.load(f)
    f.close()
    names = list(info.keys())
    random.shuffle(names)
    i = 0
    while True:
        try:
            if i == 0:
                images = []
                labels = []
            name = names[index]
            path = os.path.join(im_dir, name)
            im = imread(path)
            if im is None:
                continue

            label = info[name]['label']

            images.append(im)
            labels.append(label)
            i += 1
            index += 1
            if index == len(names):
                index = 0
                random.shuffle(names)
            if i >= batch_size:
                images = np.array(images)
                labels = np.array(labels)
                labels = (np.arange(4) == labels[:, None]).astype(np.integer)
                yield (images, labels)
                i = 0
        except Exception as e:
            raise
