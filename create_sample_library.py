# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 16:43:17 2020

@author: zhou
"""
import cv2
import numpy as np
import time
import os

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
    return result 

# process photo before divide it into single number
def photo_to_gray(im):
    im_gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    ret,im_bin=cv2.threshold(im_gray,127,255,cv2.THRESH_BINARY_INV)
    kernel=1/16*np.array([[1,2,1],[2,4,2],[1,2,1]])
    im_blur=cv2.filter2D(im_bin,-1,kernel)
    ret,im_fin=cv2.threshold(im_blur,127,255,cv2.THRESH_BINARY)
    return im_fin

#save photo being divided
def label_number(im,boxs):
     for box in boxs:
               im_divide= im[box[0][1]:box[3][1],box[0][0]:box[1][0]]
               im_adjust= cv2.resize(im_divide, (30, 30))
               cv2.imshow("image", im_adjust)
               number=cv2.waitKey(0)
               char=chr(number)
               timestamp = int(time.time() * 1e6) 
               filename = "{a}_{b}.jpg".format(a=timestamp,b=char)
               filepath = os.path.join(dir_label, filename)
               cv2.imwrite(filepath, im_adjust)

dirname="test"#Change it to your own
dir_label="number"#Change it to your own
files = os.listdir(dirname)
for file in files:
           filepath = os.path.join(dirname, file)
           im = cv2.imread(filepath)
           im_fin=photo_to_gray(im)
           boxs=divide(im_fin)
           label_number(im_fin,boxs)
           cv2.destroyAllWindows()
