# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 10:03:21 2017

@author: soumil
"""
from keras.models import load_model
import cv2
import numpy as np
E_R=load_model('model.h5')
E_R.summary()
image=np.empty((1,3,100,100),dtype=np.int32)


cam = cv2.VideoCapture(0)
while input("press enter to continue or 'exit' to exit ")!="exit":
    _,img=cam.read()
    cv2.imshow('image',img)
    cv2.waitKey()
    image[0,:,:,:] = np.reshape(cv2.resize(img, (100,100)),(3,100,100))
    print(E_R.predict(image))
cam.release()