# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 18:07:57 2020

@author: zhou
"""
import glob
import os
import time
import numpy as np
import cv2


sample_dir="label"
dirname="test"

#divide phto into single number           
def divide(im):
    contours, hierarchy = cv2.findContours(im,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    result=[]
    if len(contours) == 4:
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            box = np.int0([[x,y], [x+w,y], [x+w,y+h], [x,y+h]])
            result.append(box)
    elif len(contours)==3:
        a=[]
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            a.append(w)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if(w!=max(a)):
                box = np.int0([[x,y], [x+w,y], [x+w,y+h], [x,y+h]])
                result.append(box)
            else: 
                box_left = np.int0([[x,y], [x+w/2,y], [x+w/2,y+h], [x,y+h]])
                box_right = np.int0([[x+w/2,y], [x+w,y], [x+w,y+h], [x+w/2,y+h]])
                result.append(box_left)
                result.append(box_right)
            
    elif len(contours)==1:
        x, y, w, h = cv2.boundingRect(contour)
        box0 = np.int0([[x,y], [x+w/4,y], [x+w/4,y+h], [x,y+h]])
        box1 = np.int0([[x+w/4,y], [x+w*2/4,y], [x+w*2/4,y+h], [x+w/4,y+h]])
        box2 = np.int0([[x+w*2/4,y], [x+w*3/4,y], [x+w*3/4,y+h], [x+w*2/4,y+h]])
        box3 = np.int0([[x+w*3/4,y], [x+w,y], [x+w,y+h], [x+w*3/4,y+h]])
        result.extend([box0, box1, box2, box3])
    elif len(contours) == 2:
        a=[]
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            a.append(w)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w == max(a) and max(a) >=min(a) * 2:
                box_left = np.int0([[x,y], [x+w/3,y], [x+w/3,y+h], [x,y+h]])
                box_mid = np.int0([[x+w/3,y], [x+w*2/3,y], [x+w*2/3,y+h], [x+w/3,y+h]])
                box_right = np.int0([[x+w*2/3,y], [x+w,y], [x+w,y+h], [x+w*2/3,y+h]])
                result.append(box_left)
                result.append(box_mid)
                result.append(box_right)
            elif max(a)< min(a) * 2:
                box_left = np.int0([[x,y], [x+w/2,y], [x+w/2,y+h], [x,y+h]])
                box_right = np.int0([[x+w/2,y], [x+w,y], [x+w,y+h], [x+w/2,y+h]])
                result.append(box_left)
                result.append(box_right)
            else:
                box = np.int0([[x,y], [x+w,y], [x+w,y+h], [x,y+h]])
                result.append(box)
    elif len(contours) > 4:
        print("you may have wrong when divided")
    result = sorted(result, key=lambda x: x[0][0])#这里多了一步排序，因为识别验证码和之前建立样本库不一样这里需要按照顺序，而这个sort排序原则就是（因为result是4个[x,y]一组对应一个矩形)取第一个[x,y]然后再取出里面的x,按照从小到大排序
    return result 
# process photo before divide it into single number
def photo_to_gray(im):
    im_gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    ret,im_bin=cv2.threshold(im_gray,127,255,cv2.THRESH_BINARY_INV)
    kernel=1/16*np.array([[1,2,1],[2,4,2],[1,2,1]])
    im_blur=cv2.filter2D(im_bin,-1,kernel)
    ret,im_fin=cv2.threshold(im_blur,127,255,cv2.THRESH_BINARY)
    return im_fin
# split captcha code into single characters
#load all data 
def load_data():
    filenames = os.listdir(sample_dir)
    samples = np.empty((0, 900))#创建一个空的np数组,这里900是因为我们的像素是30X30
    labels = []
    for filename in filenames:
        filepath = os.path.join(sample_dir, filename)
        label = filename.split(".")[0].split("_")[1]
        labels.append(label)
        im = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)#以灰度方式读入一张照片
        sample = im.reshape((1, 900)).astype(np.float32)#把原来30X30的图片变成一个新的1X900的矩阵，类型是浮点数，用来储存样本信息
        samples = np.append(samples, sample, 0)
    samples = samples.astype(np.float32)
    unique_labels = list(set(labels))#由于这里labels有重复的（肯定的因为样本不止一个)所以通过set去除重复样本，而这里需要list类型的samples所以再转化
    unique_ids = list(range(len(unique_labels)))#这个相当于获得了0-9的list
    label_id_map = dict(zip(unique_labels, unique_ids))
    id_label_map = dict(zip(unique_ids, unique_labels))
    label_ids = list(map(lambda x: label_id_map[x], labels))#形成对应的向量关系
    label_ids = np.array(label_ids).reshape((-1, 1)).astype(np.float32)#因为之后全部都要用到数组，所以还需要reshape
    return [samples, label_ids, id_label_map]
# Identification verification code
def get_code(im):
    [samples, label_ids, id_label_map] = load_data()
    
    model = cv2.ml.KNearest_create()#生成模型
    model.train(samples, cv2.ml.ROW_SAMPLE, label_ids)#训练数据,第一个参数是样本，第二个参数是样本类型，因为我们定义的1X900所以样本是row,当然你也可以选择列不过之前也要改,第三个参数是与训练样本相关的响应向量,也就是我们想知道的具体数字
    im_fin =photo_to_gray(im)
    boxes = divide(im_fin)
    result = []
    for box in boxes:
        im_divide= im_fin[box[0][1]:box[3][1],box[0][0]:box[1][0]]
        im_adjust= cv2.resize(im_divide, (30, 30))
        im_reshape=im_adjust.reshape((1, 900)).astype(np.float32)
        ret,results, neighbours, distances = model.findNearest( im_reshape, k = 3)#第一个参数就是我们的当前验证码数字图片，第二个参数是判断最近邻居的个数，这里用3，当然也可以适当调整,第二个输出参数就是我们的结果
        label_id = int(results[0,0])#获得向量的坐标
        label = id_label_map[label_id]#找到对应的数字
        result.append(label)
    e=result[0]+result[1]+result[2]+result[3]
    return e

files = os.listdir(dirname)
for file in files:
           filepath = os.path.join(dirname, file)
           im = cv2.imread(filepath)
           number=get_code(im)
           print(number)
