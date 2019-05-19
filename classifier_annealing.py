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
import myClassifier
import math
import numpy as np
import os
from keras.callbacks import ModelCheckpoint
import pandas as pd
import sys
import keras
import random
from keras.callbacks import ReduceLROnPlateau
from minio import Minio
from minio.error import ResponseError


if len(sys.argv) < 2:
    print("usage: classifier_anneal.py <number of all classifiers>\n example if you have 5 nodes top 5: 25")
    sys.exit()

fname = 'FordA'
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
reduce_lr = ReduceLROnPlateau(monitor = 'loss', factor=0.5,
                  patience=50, min_lr=0.0001)

numClassifiers = int(sys.argv[1])

models = []
model = Model()
for num in range(numClassifiers+1):
    filename="weights-best-"+str(numClassifiers)+"classifier-"+str(num)+".hdf5"
    if os.path.isfile(filename):
        model = myClassifier.generateModel(x,nb_classes)
        model.load_weights(filename)
        models.append(model)
        print("Pushed "+filename)

from simanneal import Annealer
class OptimizeWeights(Annealer):
    def move(self):
        a = random.randint(0, len(self.state) - 1)
        if (random.randint(0,1) == 0):
            self.state[a] += 1
        else:
            if self.state[a] > 0:
                self.state[a] -= 1

    def energy(self):
        accuracy = myClassifier.ensemble_model(classifiers, Y_test, numClassifiers, self.state)
        e = 1 / accuracy
        return e
print("Checking models...")
classifiers = myClassifier.ensemble_store_results(models,x_test)
initialState = [ 1 for x in range(numClassifiers) ]
print("Models loaded")
del models, x_train, y_train
#accuracy = myClassifier.ensemble_model(classifiers, Y_test, numClassifiers, initialState)

optW = OptimizeWeights(initialState)

optW.Tmin = 1
optW.steps = 10000
optW.updates = 200

optimumWeights, optAccuracy = optW.anneal()
print("Annealing finished!")
print("Accuracy: "+str(1/optAccuracy))
print(optimumWeights)
