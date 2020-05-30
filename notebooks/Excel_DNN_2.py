#!/usr/bin/env python
# coding: utf-8

import tensorflow.compat.v1 as tensorflow
tensorflow.disable_v2_behavior()

tf.config.experimental.set_visible_devices([], 'GPU')

#gpus = tensorflow.config.experimental.list_physical_devices('GPU')
#for gpu in gpus:
#  tensorflow.config.experimental.set_memory_growth(gpu, True)

#import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, optimizers, regularizers
from tensorflow.keras.layers import Flatten , Activation
from tensorflow.keras.layers import Dense
#from keras.wrappers.scikit_learn import KerasClassifier
import time
from tensorflow.keras.callbacks import TensorBoard

NAME ="Excel model-{}".format(int(time.time()))

tensorboard=TensorBoard(log_dir='logs/{}'.format(NAME))

df =pd.read_csv('../data/training.csv')

df = df.replace(-999, np.nan)
df = df.dropna()
df.shape
df_droped = df.drop(['EventId', 'Weight'], axis=1)
df_droped['Label'] = df_droped['Label'].apply({'s':1, 'b':0}.get)
X_train, X_test, y_train, y_test = train_test_split(df_droped.drop(['Label'], axis=1), df_droped['Label'], test_size=0.2)
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()
X_train = X_train.to_numpy()
X_test = X_test.to_numpy()

scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
print(y_train.shape , X_train.shape)
print(y_test.shape , X_test.shape)



model = Sequential()

model.add(Flatten())

model.add(Dense(1000))
model.add(Activation('relu')) 
'''
model.add(Dense(1000))
model.add(Activation('relu')) 

model.add(Dense(1000))
model.add(Activation('relu')) 

model.add(Dense(1000))
model.add(Activation('relu')) 

#model.add(Dense(500))
#model.add(Activation('relu')) 

#model.add(Dense(500))
#model.add(Activation("relu")) 

#model.add(Dense(500))
#model.add(Activation("relu"))

#model.add(Dense(500))
#model.add(Activation("relu"))

'''          
model.add(Dense(100))
model.add(Activation("relu"))
          
          
#model.add(Dense(10))
#model.add(Activation('relu'))          

# Add an output layer 
model.add(Dense(2,activation='sigmoid'))

# define Parameters for the training of the model
# Good default optimizer to start with , how will we calculate our "error." Neural network aims to minimize loss.


model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])

model.fit(X_train,y_train)


'''

          
          
val_loss, val_acc = model.evaluate(X_test, y_test) # evaluate the out of sample data with model
print("test loss=",val_loss) #model's loss
print("test acc =",val_acc)  #model's accuracy   


y_pred=model.predict(X_test)
y_pred= np.delete(y_pred, np.s_[-1:], axis=1)
y_pred=(y_pred>0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred) #confusion matrix is not working with continious probability values
print(cm)

sensitivity=cm[0,0]/(cm[0,0]+cm[1,0]) 
specificity=cm[1,1]/(cm[1,0]+cm[1,1]) 
print("sesnsetivity =",sensitivity)
print( "specificity =",specificity)

'''




