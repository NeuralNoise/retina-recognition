import cv2
import os
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation,Flatten
from keras.layers import Conv2D
neg_img = np.empty((30,3,100,100),dtype=np.int32)
pos_img = np.empty((50,3,100,100),dtype=np.int32)
print("keras imported")
E_R = Sequential()
#%%Reading the input data
os.chdir('C:/Users/soumil/Desktop/retina_recog/datasets')
file_names = os.listdir()
pos_count=0
neg_count=0
for file_name in file_names:
    if file_name[-3:]=="npy" and file_name[0:3]=="neg":
        temp = np.load(file_name)
        temp = np.reshape(cv2.resize(temp,(100,100)),(3,100,100))
        neg_img[neg_count,:,:,:]=temp
        neg_count+=1
    elif file_name[-3:]=="npy" and file_name[0] != "n":
        temp = np.load(file_name)
        temp = np.reshape(cv2.resize(temp,(100,100)),(3,100,100))
        pos_img[neg_count,:,:,:]=temp
        pos_count+=1
    else:
        temp = cv2.imread(file_name)
        temp = np.reshape(cv2.resize(temp,(100,100)),(3,100,100))
        neg_img[neg_count,:,:,:]=temp
        neg_count+=1
print("dataset loaded")
        

#%% Layering
E_R.add(Conv2D(25,3,3,input_shape = (3,100,100)))
E_R.add(Activation('relu'))
E_R.add(Dropout(0.1,seed = 12))
E_R.add(Conv2D(20,1,1))
E_R.add(Activation('relu'))
E_R.add(Dropout(0.1,seed=12))
E_R.add(Flatten())
E_R.add(Dense(1))
E_R.add(Activation('softmax'))
print("model made")
E_R.summary()

#%%Optimization
E_R.compile(loss='mean_squared_error', optimizer='Adam',metrics=['accuracy'])

#%%training
x_train=np.empty((63,3,100,100),dtype=np.int32)
x_test=np.empty((17,3,100,100),dtype=np.int32)
y_train=np.empty((63,1),dtype=np.int32)
y_test=np.empty((17,1),dtype=np.int32)
x_train[0:25,:,:,:]=neg_img[0:25,:,:,:]
x_train[25:63,:,:,:]=pos_img[0:38,:,:,:]
x_test[0:5,:,:,:]=neg_img[25:30,:,:,:]
x_test[5:17,:,:,:]=pos_img[25:37,:,:,:]
y_train[0:25,:]=np.zeros((25,1),dtype=np.int32)
y_train[25:63,:]=np.ones((38,1),dtype=np.int32)
y_test[0:5,:]=np.zeros((5,1),dtype=np.int32)
y_test[5:17,:]=np.ones((12,1),dtype=np.int32)
E_R.fit(x_train,y_train,epochs=50)
os.chdir('C:/Users/soumil/Desktop/retina_recog')
E_R.save('model.h5')
print(E_R.evaluate(x_test,y_test))

