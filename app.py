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
    model.fit(x_train, Y_train, batch_size=batch_size, epochs=1, verbose=0)
    scores = model.evaluate(X, Y, verbose=1)
    return scores[1] 

def connectMinio():
    httpClient = urllib3.PoolManager()
    return Minio('192.168.100.8:9000',
                  access_key='90WTMGQ1LA2VLTR635J3',
                  secret_key='BK2REoMkJ9KO2WZcpAjaUgRw9vfOrY+ANEOeOnUd',
                  secure=False,
                  http_client=httpClient)


nb_epochs = 2
rounds=10
myid = int(sys.argv[1])
#flist = ['Adiac', 'Beef', 'CBF', 'ChlorineConcentration', 'CinC_ECG_torso', 'Coffee', 'Cricket_X', 'Cricket_Y', 'Cricket_Z', 
#'DiatomSizeReduction', 'ECGFiveDays', 'FaceAll', 'FaceFour', 'FacesUCR', '50words', 'FISH', 'Gun_Point', 'Haptics', 
#'InlineSkate', 'ItalyPowerDemand', 'Lighting2', 'Lighting7', 'MALLAT', 'MedicalImages', 'MoteStrain', 'NonInvasiveFatalECG_Thorax1', 
#'NonInvasiveFatalECG_Thorax2', 'OliveOil', 'OSULeaf', 'SonyAIBORobotSurface', 'SonyAIBORobotSurfaceII', 'StarLightCurves', 'SwedishLeaf', 'Symbols', 
#'synthetic_control', 'Trace', 'TwoLeadECG', 'Two_Patterns', 'uWaveGestureLibrary_X', 'uWaveGestureLibrary_Y', 'uWaveGestureLibrary_Z', 'wafer', 'WordsSynonyms', 'yoga']

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

for round_n in range(rounds):
    #List all object paths in bucket that begin with my-prefixname.
    objects = minioClient.list_objects('mybucket', prefix='weights',
                                  recursive=True)
    best_acc = 0
    best_filename = ""
    for obj in objects:
        filename=obj.object_name
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
        model.load_weights(best_filename)
     
    localFilePath="weights-best.hdf5"
    checkpointer = ModelCheckpoint(localFilePath,
            monitor = 'val_acc',
            verbose=1,
            save_best_only=True)

    hist = model.fit(x_train, Y_train, batch_size=batch_size, epochs=nb_epochs,
            verbose=1, callbacks=[checkpointer,reduce_lr], validation_data=(x_test, Y_test))

    #Print the testing results which has the lowest training loss.
    log = pd.DataFrame(hist.history)
    print (log.loc[log['loss'].idxmin]['loss'], log.loc[log['loss'].idxmin]['val_acc'])

    metadata =  {
        "Creator_id" : myid,
        "Creator_loss" : log.loc[log['loss'].idxmin]['loss'],
        "Creator_acc" :  log.loc[log['loss'].idxmin]['val_acc'],
        "Last_model" : best_filename
    }
    del minioClient
    minioClient = connectMinio()
    #filepath="weights-{epoch:02d}-{val_acc:.2f}.hdf5"
    try:
        if not minioClient.bucket_exists("mybucket"):
            minioClient.make_bucket("mybucket", location="us-east-1")
        minioClient.fput_object('mybucket', "weights-"+str(round_n)+"-"+str(myid)+"-best.hdf5", localFilePath)#, metadata=metadata)
    except ResponseError as err:
        print(err)
