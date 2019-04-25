#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Originally Created on Sun Oct 30 20:11:19 2016
@author: stephen

Modified
Alexandre F M Souza (UFPR)
""" 
from __future__ import print_function

from keras.models import Model
from keras.utils import np_utils
import numpy as np
import os
from keras.callbacks import ModelCheckpoint
import pandas as pd
import sys
import keras 
import urllib3
import shutil
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

def connectMinio():
    httpClient = urllib3.PoolManager()
    return Minio('172.17.0.1:9000',
                  access_key='90WTMGQ1LA2VLTR635J3',
                  secret_key='BK2REoMkJ9KO2WZcpAjaUgRw9vfOrY+ANEOeOnUd',
                  secure=False,
                  http_client=httpClient)


myid = int(sys.argv[1])

for noden in range(1,myid+1):
    fname = str(noden)#'FordA'
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
     
    reduce_lr = ReduceLROnPlateau(monitor = 'loss', factor=0.5,
                      patience=50, min_lr=0.0001)

    minioClient = connectMinio()

    #List all object paths in bucket that begin with my-prefixname.
    objects = minioClient.list_objects_v2(sys.argv[2], prefix='weights',
                                  recursive=False)

    best_acc = 0
    best_filename = ""
    for obj in objects:
        filename=obj.object_name
        if not "-" + str(noden) + "-best" in filename:
            continue
        print(obj.bucket_name, filename, obj.last_modified, obj.etag, obj.size, obj.content_type)
        minioClient.fget_object(obj.bucket_name, filename, filename)
        if os.path.isfile(filename):
            model.load_weights(filename)
            model_acc = checkModel(model,x_train,Y_train)
            print("Tested "+filename+" Model accuracy: "+str(model_acc*100))
            if (model_acc > best_acc):
                best_acc=model_acc
                best_filename=filename
    if (len(best_filename) > 0):
        print("Best model number:"+best_filename)
        shutil.copy2(best_filename,"weights-best-"+str(myid)+"classifier-"+str(noden)+".hdf5")


