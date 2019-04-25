#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Originally Created on Sun Oct 30 20:11:19 2016
@author: stephen

Modified
Alexandre F M Souza (UFPR)
""" 
from __future__ import print_function

from keras.models import Model, Input
from keras.utils import np_utils
import math
import numpy as np
import os
from keras.callbacks import ModelCheckpoint
import pandas as pd
import sys
import csv
import keras 
import urllib3
from keras.callbacks import ReduceLROnPlateau

from minio import Minio
from minio.error import ResponseError

def readucr(filename):
    data = np.loadtxt(filename, delimiter = ',')
    Y = data[:,0]
    X = data[:,1:]
    return X, Y

def checkModel(model, X, Y):
    print("checkModel:")
    scores = model.evaluate(X, Y, verbose=1)
    return scores[1] 

def generateModel(x) -> Model: 
    #    drop_out = Dropout(0.2)(x)
    conv1 = keras.layers.Conv1D(filters=128, kernel_size=8, padding='same',input_shape=(64,1))(x)
    conv1 = keras.layers.normalization.BatchNormalization()(conv1)
    conv1 = keras.layers.Activation('relu')(conv1)

    #    drop_out = Dropout(0.2)(conv1)
    conv2 = keras.layers.Conv1D(filters=256, kernel_size=5, padding='same')(conv1)
    conv2 = keras.layers.normalization.BatchNormalization()(conv2)
    conv2 = keras.layers.Activation('relu')(conv2)

    #    drop_out = Dropout(0.2)(conv2)
    conv3 = keras.layers.Conv1D(filters=128, kernel_size=3, padding='same')(conv2)
    conv3 = keras.layers.normalization.BatchNormalization()(conv3)
    conv3 = keras.layers.Activation('relu')(conv3)

    full = keras.layers.pooling.GlobalAveragePooling1D()(conv3)    
    out = keras.layers.Dense(nb_classes, activation='softmax')(full)


    model = Model(input=x, output=out)
     
    optimizer = keras.optimizers.Adam()
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model
 


def emsemble(models, x_test, y_test):
    nb_classes = len(np.unique(y_test))
    i = 0
    vote1 = 0
    vote0 = 0
    acerto_n = 0
    erro_n = 0
    acerto_p = 0
    erro_p = 0
    with open('rejected.csv', mode='w') as rejected_file:
        rejected_writer = csv.writer(rejected_file, delimiter=',', escapechar='', quoting=csv.QUOTE_MINIMAL)
        while i < len(x_test):
            for model in models:
                q = model.predict(x_test[i:i+1])
                vote1+=q[0][0]
                vote0+=q[0][1]
            if (vote0) > (vote1):
                if (y_test[i][0] == 0):
                    acerto_n+=1
                else:
                    erro_n+=1
                    rejected_writer.writerow(map (lambda x: [x], np.insert(x_test[i:i+1],0,1)))
                    print("false 0: "+str(vote0)+" "+str(vote1))
            else:
                if (y_test[i][0] == 1):
                    acerto_p+=1
                else:
                    erro_p+=1
                    rejected_writer.writerow(map (lambda x: [x], np.insert(x_test[i:i+1],0,-1)))
                    print("false 1: "+str(vote0)+" "+str(vote1))
            i+=1
            vote1 = 0
            vote0 = 0
    print(""+str(acerto_n) + " "+ str(erro_n) + "\n " + str(erro_p) + " "+ str(acerto_p))
    print("\n"+str((acerto_n+acerto_p)/(acerto_n+acerto_p+erro_p+erro_n)))
numClassifiers = int(sys.argv[1])

fname = 'FordA'
x_train, y_train = readucr(fname+'/'+fname+'_TRAIN')
x_test, y_test = readucr(fname+'/'+fname+'_TEST')
nb_classes = len(np.unique(y_test))
batch_size = int(min(x_train.shape[0]/10, 16))
y_train = (y_train - y_train.min())/(y_train.max()-y_train.min())*(nb_classes-1)
y_test = (y_test - y_test.min())/(y_test.max()-y_test.min())*(nb_classes-1)
 
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)
print ("class:"+fname+", number of classes: "+str(nb_classes))

x_train_mean = x_train.mean()
x_train_std = x_train.std()
x_train = (x_train - x_train_mean)/(x_train_std)
 
x_test = (x_test - x_train_mean)/(x_train_std)
x_train = x_train.reshape(x_train.shape + (1,))
x_test = x_test.reshape(x_test.shape + (1,))
x = keras.layers.Input(x_train.shape[1:])
reduce_lr = ReduceLROnPlateau(monitor = 'loss', factor=0.5,
                  patience=50, min_lr=0.0001)

models = []
model = Model()
for num in range(numClassifiers+1):
    #filename="temp_weights-"+str(num)+".hdf5"
    filename="weights-best-"+str(numClassifiers)+"classifier-"+str(num)+".hdf5"
    if os.path.isfile(filename):
        model = generateModel(x)
        model.load_weights(filename)
        models.append(model)
        print("Pushed "+filename)

model = emsemble(models, x_test, Y_test)
