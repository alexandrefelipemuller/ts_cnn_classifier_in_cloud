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
import random
import myClassifier
from keras.callbacks import ReduceLROnPlateau

from minio import Minio
from minio.error import ResponseError


nb_epochs = 25
first_round = 127
rounds=150

entropy=15

myid = int(sys.argv[1])
#flist = ['Adiac', 'Beef', 'CBF', 'ChlorineConcentration', 'CinC_ECG_torso', 'Coffee', 'Cricket_X', 'Cricket_Y', 'Cricket_Z', 
#'DiatomSizeReduction', 'ECGFiveDays', 'FaceAll', 'FaceFour', 'FacesUCR', '50words', 'FISH', 'Gun_Point', 'Haptics', 
#'InlineSkate', 'ItalyPowerDemand', 'Lighting2', 'Lighting7', 'MALLAT', 'MedicalImages', 'MoteStrain', 'NonInvasiveFatalECG_Thorax1', 
#'NonInvasiveFatalECG_Thorax2', 'OliveOil', 'OSULeaf', 'SonyAIBORobotSurface', 'SonyAIBORobotSurfaceII', 'StarLightCurves', 'SwedishLeaf', 'Symbols', 
#'synthetic_control', 'Trace', 'TwoLeadECG', 'Two_Patterns', 'uWaveGestureLibrary_X', 'uWaveGestureLibrary_Y', 'uWaveGestureLibrary_Z', 'wafer', 'WordsSynonyms', 'yoga']

fname = str(myid)
x_train, y_train = myClassifier.readucr(fname+'/'+fname+'_TRAIN')
x_test, y_test = myClassifier.readucr(fname+'/'+fname+'_TEST')
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

model = myClassifier.generateModel(x,nb_classes)

optimizer = keras.optimizers.Adam()
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])
 
reduce_lr = ReduceLROnPlateau(monitor = 'loss', factor=0.5,
                  patience=50, min_lr=0.0001)

minioClient = myClassifier.connectMinio()
if not minioClient.bucket_exists("cloudcl7"):
     minioClient.make_bucket("cloudcl7", location="us-east-1")

models_result = {}
for round_n in range(first_round, rounds):
    #List all object paths in bucket that begin with my-prefixname.
    objects = minioClient.list_objects('cloudcl7', prefix='weights',
                                  recursive=True)
    best_acc = 0
    best_filename = ""

    for obj in objects:
        filename=obj.object_name
        if str(myid) + "-best" in filename:
            continue
        print(obj.bucket_name, filename, obj.last_modified, obj.etag, obj.size, obj.content_type)
        try:
            minioClient.fget_object(obj.bucket_name, filename, filename)
        except:
            continue
        if os.path.isfile(filename):
            if filename in models_result:
                model_acc = models_result[filename]
            else:
                model.load_weights(filename)
                model_acc = myClassifier.checkSingleModel(model,x_train,Y_train,batch_size)
                models_result[filename] = model_acc
            print("Tested "+filename+" Model accuracy: "+str(model_acc*100))
            model_acc = (model_acc*10) * (random.randrange(1, entropy, 1) * ((rounds/round_n)-1) )
            if (model_acc > best_acc):
                best_acc=model_acc
                best_filename=filename
    if (len(best_filename) > 0):
        print("Best model number:"+best_filename)
        model.load_weights(best_filename)
    
    localFilePath="temp_weights-"+str(myid)+".hdf5"
    checkpointer = ModelCheckpoint(localFilePath,
            monitor = 'val_acc',
            verbose=1,
            save_best_only=True)

    hist = model.fit(x_train, Y_train, batch_size=batch_size, epochs=nb_epochs,
            verbose=1, callbacks=[checkpointer,reduce_lr], validation_data=(x_test, Y_test))

    #Print the testing results which has the lowest training loss.
    log = pd.DataFrame(hist.history)
    creator_loss = log.loc[log['loss'].idxmin]['loss']
    creator_acc = log.loc[log['loss'].idxmin]['val_acc']
    print (creator_loss, creator_acc)

    del minioClient
    minioClient = myClassifier.connectMinio()
    try:
        metadata =  {
        "Creator_id" : str(myid),
        "Creator_loss" : str(creator_loss),
        "Creator_acc" :  str(creator_acc),
        "Last_model" : best_filename
        }
        remoteFilePath = "weights-"+str(round_n)+"-"+str(myid)+"-best.hdf5"
        minioClient.fput_object('cloudcl7', remoteFilePath, localFilePath, metadata=metadata)
    except ResponseError as err:
        print(err)
    os.rename(localFilePath, remoteFilePath)
