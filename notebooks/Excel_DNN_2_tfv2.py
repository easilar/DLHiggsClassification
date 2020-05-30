#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras import layers, optimizers, regularizers
from keras.layers import Flatten , Activation
from keras.layers import Dense
from keras.utils import multi_gpu_model

import time

print(tf.__version__)
print(keras.__version__)

import keras.backend.tensorflow_backend as tfback

def _get_available_gpus():
    """Get a list of available gpu devices (formatted as strings).
    # Source of this function: https://github.com/keras-team/keras/issues/13684
    # Returns
        A list of available GPU devices.
    """
    #global _LOCAL_DEVICES
    if tfback._LOCAL_DEVICES is None:
        devices = tf.config.list_logical_devices()
        tfback._LOCAL_DEVICES = [x.name for x in devices]
    return [x for x in tfback._LOCAL_DEVICES if 'device:gpu' in x.lower()]



gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[1], True)
tf.config.set_visible_devices(gpus[1], 'GPU')

tfback._get_available_gpus = _get_available_gpus
print(_get_available_gpus())

df =pd.read_csv('../data/training.csv')

df = df.replace(-999, np.nan)
df = df.dropna()
print(df.shape)
df_droped = df.drop(['EventId', 'Weight'], axis=1)
df_droped['Label'] = df_droped['Label'].apply({'s':1, 'b':0}.get)

X_train, X_test, y_train, y_test = train_test_split(df_droped.drop(['Label'], axis=1), df_droped['Label'], test_size=0.2)
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()


scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


model = Sequential()
model.add(Dense(100))
model.add(Activation("relu"))
model.add(Dense(10))
model.add(Activation('relu'))          
# Add an output layer 
model.add(Dense(1))
model.add(Activation('sigmoid'))

print("model architecture is ready !")

#parallel_model = multi_gpu_model(model, gpus=1)
#print("model is parallel model from keras which will run on 2 GPUs")

model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
print("model is compiled !")
model.fit(X_train,y_train, epochs=10 ,batch_size=256)


          
          
val_loss, val_acc = model.evaluate(X_test, y_test) # evaluate the out of sample data with model
print("test loss=",val_loss) #model's loss
print("test acc =",val_acc)  #model's accuracy   


y_pred=model.predict(X_test)
y_pred=(y_pred>0.5)

print(y_pred.shape , y_test.shape)
print(y_pred)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred) #confusion matrix is not working with continious probability values
print(cm)

sensitivity=cm[0,0]/(cm[0,0]+cm[1,0]) 
specificity=cm[1,1]/(cm[1,0]+cm[1,1]) 
print("sensetivity =",sensitivity)
print("specificity =",specificity)








