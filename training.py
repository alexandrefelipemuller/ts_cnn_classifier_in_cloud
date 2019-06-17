#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Originally Created on Sun Oct 30 20:11:19 2016
@author: stephen
https://github.com/cauchyturing/UCR_Time_Series_Classification_Deep_Learning_Baseline

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
import random
import myClassifier
import re
from keras.callbacks import ReduceLROnPlateau

from minio import Minio
from minio.error import ResponseError


nb_epochs = 20
entropy=13

if len(sys.argv) < 3:
    print("usage: trainning.py <id> <target_bucket>")
    sys.exit()
myid = sys.argv[1]
target_bucket = sys.argv[2]
first_round = 1 #int(sys.argv[3])
rounds=60 #int(sys.argv[4])


#flist = ['Adiac', 'Beef', 'CBF', 'ChlorineConcentration', 'CinC_ECG_torso', 'Coffee', 'Cricket_X', 'Cricket_Y', 'Cricket_Z', 
#'DiatomSizeReduction', 'ECGFiveDays', 'FaceAll', 'FaceFour', 'FacesUCR', '50words', 'FISH', 'Gun_Point', 'Haptics', 
#'InlineSkate', 'ItalyPowerDemand', 'Lighting2', 'Lighting7', 'MALLAT', 'MedicalImages', 'MoteStrain', 'NonInvasiveFatalECG_Thorax1', 
#'NonInvasiveFatalECG_Thorax2', 'OliveOil', 'OSULeaf', 'SonyAIBORobotSurface', 'SonyAIBORobotSurfaceII', 'StarLightCurves', 'SwedishLeaf', 'Symbols', 
#'synthetic_control', 'Trace', 'TwoLeadECG', 'Two_Patterns', 'uWaveGestureLibrary_X', 'uWaveGestureLibrary_Y', 'uWaveGestureLibrary_Z', 'wafer', 'WordsSynonyms', 'yoga']

fname = myid
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
if not minioClient.bucket_exists(target_bucket):
     minioClient.make_bucket(target_bucket, location="us-east-1")

models_result = {}
for round_n in range(first_round, rounds):
    #List all object paths in bucket that begin with my-prefixname.
    objects = minioClient.list_objects(target_bucket, prefix='weights',
                                  recursive=True)
    best_acc = 0
    best_filename = ""

    for obj in objects:
        filename=obj.object_name
        m = re.search('weights-([0-9]*)-([A-Za-z_0-9]*)-best.hdf5', filename)
        if (myid == m.group(2) or int(m.group(1)) < round_n - 30): # if it the same node or older than 10 rounds, then skip it
            try:
                del models_result[filename]
            except NameError:
                pass
            except KeyError:
                pass
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
            model_acc = (model_acc*15) * (random.randrange(1, entropy, 1) * ((rounds/round_n)-1) )
            if (model_acc > best_acc):
                best_acc=model_acc
                best_filename=filename
    if (len(best_filename) > 0):
        print("Best model number:"+best_filename)
        model.load_weights(best_filename)
    
    localFilePath="temp_weights-"+myid+".hdf5"
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
        "Creator_id" : fname,
        "Creator_loss" : str(creator_loss),
        "Creator_acc" :  str(creator_acc),
        "Last_model" : best_filename
        }
        remoteFilePath = "weights-"+str(round_n)+"-"+myid+"-best.hdf5"
        minioClient.fput_object(target_bucket, remoteFilePath, localFilePath, metadata=metadata)
    except ResponseError as err:
        print(err)
    os.rename(localFilePath, remoteFilePath)
