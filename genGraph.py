#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Originally Created on Sun Oct 30 20:11:19 2016
@author: stephen

Modified
Alexandre F M Souza (UFPR)
""" 
from __future__ import print_function

import numpy as np
import myClassifier
import os
import sys
import re

from minio import Minio
from minio.error import ResponseError

series = {}
series['Gun_Point'] = 1
series['FordA'] = 2
series['InlineSkate'] = 3
series['ShapeletSim'] = 4  

def nodeName2ID(nodeName):
    m = re.search('([0-9]*)-([a-zA-Z_0-9]*)', nodeName)
    roundname = m.group(1)
    nodename = m.group(2)
    return str((int(roundname)*1000)+int(nodename))

def getParent(object, minioClient):
    output=""
    parent = object.metadata['X-Amz-Meta-Last_model']
    if not parent:
        return ""
    else:
        m = re.search('weights-([0-9]*-[a-zA-Z_0-9]*)-best.hdf5', object.object_name)
        n = re.search('weights-([0-9]*-[a-zA-Z_0-9]*)-best.hdf5', parent)
        output+='<edge vertex1="'+nodeName2ID(n.group(1))+'" vertex2="'+nodeName2ID(m.group(1))+'" isDirect="true" model_width="4" model_type="0" model_curvedValue="0.1" ></edge>\n'
        object_stat = minioClient.stat_object(bucket, parent)
        #print(object.object_name + " parent:"+object.metadata['X-Amz-Meta-Last_model']+" grand parent:"+object_stat.metadata['X-Amz-Meta-Last_model'])
        if (object.object_name == object_stat.metadata['X-Amz-Meta-Last_model']): 
            return "" # Loop
        output+=getParent(object_stat,minioClient)
    return output


bucket = sys.argv[1]
nodes = ""
edges = ""
minioClient = myClassifier.connectMinio()

objects = minioClient.list_objects(bucket, prefix='weights',
                                  recursive=True)
for obj in objects:
    filename=obj.object_name
    m = re.search('weights-([0-9]*-[a-zA-Z_0-9]*)-best.hdf5', filename)
    if m:
        n = re.search('([0-9]*)-([a-zA-Z_0-9]*)',m.group(1))
        roundname = n.group(1)
        nodename = n.group(2)
        nodes += "<node id=\""+nodeName2ID(m.group(1))+"\" positionX=\""+str(int(roundname)*130)+"\" positionY=\""+str(int(nodename)*250)+"\" mainText=\""+roundname+"-"+nodename+"\" upText=\"\"></node>\n"
        #print(obj.bucket_name, filename, obj.last_modified, obj.etag, obj.size, obj.content_type)
        object_stat = minioClient.stat_object(bucket, filename)
        edges+=getParent(object_stat,minioClient)
print('<?xml version="1.0" encoding="UTF-8"?><graphml><graph id="Graph" uidGraph="21" uidEdge="10085"><nodes>'+nodes+'</nodes><edges>'+edges+'</edges></graph></graphml>')
