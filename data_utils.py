import os
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from PIL import Image
from PIL import ImageDraw

import cv2

from skimage.io import imread

#利用pil.Image打开图片，判断是否有异常图片
def find_error_pil(image_dir):
    #读取图片shape是否都为3维度，判断是否有0元素
    
    #读取所有image文件名
    image_name_list = os.listdir(image_dir)
    
    cant_open = 0
    shape_len_not_3 = 0
    shape_error = 0
    for i,image_name in enumerate(image_name_list):
        try:
            image_path = os.path.join(image_dir,image_name)
            with Image.open(image_path) as itmp:
                #print(type(itmp))
                itmp = np.array(itmp)
                #print(type(itmp))
                #print(i,itmp.shape)
                if itmp is None:
                    cant_open = cant_open + 1
                    print(i,image_name)
                elif len(itmp.shape) != 3:
                    shape_len_not_3 = shape_len_not_3 + 1
                    print(i,image_name,itmp.shape)
                else:
                    h,w,c = itmp.shape
                    if w == 0 or h == 0 or c == 0 or c != 3:
                        shape_error = shape_error + 1
                        print(i,image_name,itmp.shape)
        except Exception as e:
            print(e)
    print("cant_open =",cant_open)
    print("shape_len_not_3 =",shape_len_not_3)
    print("shape_error =",shape_error)

#利用CV2打开图片，判断是否有异常图片
def find_error_cv(image_dir):
    
    #读取所有image文件名
    image_name_list = os.listdir(image_dir)
    
    cant_open = 0
    shape_len_not_3 = 0
    shape_error = 0
    for i,image_name in enumerate(image_name_list):
        try:
            image_path = os.path.join(image_dir,image_name)
            itmp = cv2.imread(image_path)
            #print(type(itmp))
            #print(i,type(itmp))
            if itmp is None:
                cant_open = cant_open + 1
                print(i,image_name)
            elif len(itmp.shape) != 3:
                shape_len_not_3 = shape_len_not_3 + 1
                print(i,image_name,im_arr.shape)
            else:
                h,w,c = itmp.shape
                if w == 0 or h == 0 or c == 0 or c != 3:
                    shape_error = shape_error + 1
                    print(i,image_name,itmp.shape)
        except Exception as e:
            print(e)
    print("cant_open =",cant_open)
    print("shape_len_not_3 =",shape_len_not_3)
    print("shape_error =",shape_error)

#利用skimage打开图片，判断是否有异常图片
def find_error_skimage(image_dir):
    
    #读取所有image文件名
    image_name_list = os.listdir(image_dir)
    
    cant_open = 0
    shape_len_not_3 = 0
    shape_error = 0
    for i,image_name in enumerate(image_name_list):
        try:
            image_path = os.path.join(image_dir,image_name)
            itmp = imread(image_path)
            #print(type(itmp))
            #print(i,type(itmp))
            if itmp is None:
                cant_open = cant_open + 1
                print(i,image_name)
            elif len(itmp.shape) != 3:
                shape_len_not_3 = shape_len_not_3 + 1
                print(i,image_name,itmp.shape)
            else:
                h,w,c = itmp.shape
                if w == 0 or h == 0 or c == 0 or c != 3:
                    shape_error = shape_error + 1
                    print(i,image_name,itmp.shape)
        except Exception as e:
            print(e)
    print("cant_open =",cant_open)
    print("shape_len_not_3 =",shape_len_not_3)
    print("shape_error =",shape_error)

def find_type_error(image_dir):
    
    #读取所有image文件名
    image_name_list = os.listdir(image_dir)
    
    type_error = []
    gif = []
    jpg = []
    jpeg = []
    png = []
    for i, image_name in enumerate(image_name_list):
        try:
            image_path = os.path.join(image_dir,image_name)
            with Image.open(image_path) as itmp:
                if itmp.format == 'PNG':
                    png.append((i,image_name,itmp.format))
                elif itmp.format == 'JPG':
                    jpg.append((i,image_name,itmp.format))
                elif itmp.format == 'JPEG':
                    jpeg.append((i,image_name,itmp.format))
                elif itmp.format == 'GIF':
                    gif.append((i,image_name,itmp.format))
                else:
                    type_error.append((i,image_name,itmp.format))
        except Exception as e:
            print(e)
    print("num_type_error:",len(type_error))
    print("gif:",len(gif))
    print("num_jpg:",len(jpg))
    print("num_jpeg:",len(jpeg))
    print("num_png:",len(png))