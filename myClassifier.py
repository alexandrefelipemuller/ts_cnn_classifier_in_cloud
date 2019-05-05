from keras.models import Model, Input
import numpy as np
import keras
import csv
import urllib3
from minio import Minio
from minio.error import ResponseError


def readucr(filename):
    data = np.loadtxt(filename, delimiter = ',')
    Y = data[:,0]
    X = data[:,1:]
    return X, Y

def connectMinio():
    httpClient = urllib3.PoolManager()
    return Minio('172.17.0.1:9000',
                  access_key='90WTMGQ1LA2VLTR635J3',
                  secret_key='BK2REoMkJ9KO2WZcpAjaUgRw9vfOrY+ANEOeOnUd',
                  secure=False,
                  http_client=httpClient)

def checkSingleModel(model, X, Y, batch_size):
    print("checkModel:")
    model.fit(X, Y, batch_size=batch_size, epochs=1, verbose=0)
    scores = model.evaluate(X, Y, verbose=1)
    return scores[1] 

def checkModel(model, X, Y):
    print("checkModel:")
    scores = model.evaluate(X, Y, verbose=1)
    return scores[1] 

def generateModel(x,nb_classes) -> Model: 
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

def ensemble(models, x_test, y_test, weights = [ 1 for x in range(999) ]):
    i, j, vote1, vote0, acerto_n, erro_n, acerto_p, erro_p = 0, 0, 0, 0, 0, 0, 0, 0
    with open('rejected.csv', mode='w') as rejected_file:
        rejected_writer = csv.writer(rejected_file, delimiter=',', escapechar='', quoting=csv.QUOTE_MINIMAL)
        while i < len(x_test):
            j=0
            for model in models: 
                q = model.predict(x_test[i:i+1])
                vote1+=(q[0][0]*weights[j])
                vote0+=(q[0][1]*weights[j])
                j+=1
            if (vote0) > (vote1):
                if (y_test[i][0] == 0):
                    acerto_n+=1
                else:
                    erro_n+=1
                    #print("false 0: "+str(vote0)+" "+str(vote1))      
                    rejected_writer.writerow(map (lambda x: [x], np.insert(x_test[i:i+1],0,-1)))
            else:
                if (y_test[i][0] == 1):
                    acerto_p+=1
                else:
                    erro_p+=1
                    #print("false 1: "+str(vote0)+" "+str(vote1))
                    rejected_writer.writerow(map (lambda x: [x], np.insert(x_test[i:i+1],0,1)))
            i+=1
            vote1 = 0
            vote0 = 0
        accuracy = ((acerto_n+acerto_p)/(acerto_n+acerto_p+erro_p+erro_n))
#   print(""+str(acerto_n) + " "+ str(erro_n) + "\n " + str(erro_p) + " "+ str(acerto_p))
    print("\n"+str(accuracy))
    return accuracy

def ensemble_model(models, y_test, classifiers, weights):
    i, vote1, vote0, acerto, erro = 0, 0, 0, 0, 0
    while i < len(y_test):
        for num in range(classifiers):
            q = models[(i*classifiers)+num]
            vote1+=(q[0]*weights[num])
            vote0+=(q[1]*weights[num])
        if (vote0 > vote1):
            if (y_test[i][0] == 0):
                acerto+=1
            else:
                erro+=1
        else:
            if (y_test[i][0] == 1):
                acerto+=1
            else:
                erro+=1
        i+=1
        vote1 = 0
        vote0 = 0
    return float(acerto/(acerto+erro))

def ensemble_store_results(models, x_test):
    output = []
    i = 0
    while i < len(x_test):
        for model in models: 
            q = model.predict(x_test[i:i+1])
            output.append(q[0])
        i+=1
    return output
