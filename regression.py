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
import myClassifier
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

def getParent(object, minioClient):
    print(object.object_name + " acc:"+object.metadata['X-Amz-Meta-Creator_acc'])
    parent = object.metadata['X-Amz-Meta-Last_model']
    if not parent:
        return
    else:
        object_stat = minioClient.stat_object(bucket, parent)
        getParent(object_stat,minioClient)


objectToCheck = sys.argv[1]
bucket = sys.argv[2]

minioClient = myClassifier.connectMinio()

#List all object paths in bucket that begin with my-prefixname.
object_stat = minioClient.stat_object(bucket, objectToCheck)
getParent(object_stat,minioClient)


