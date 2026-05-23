# IBIB Training
from sklearn.model_selection import train_test_split
import os
# import cv2
import numpy as np
import pandas as pd
import mne
import glob
import matplotlib.pyplot as plt
from utilities import train_test_splitter,get_rhythm


def read_data(sf,sec):
    # Read Healthy Data
    SZ_Data = []

    path='./SZ dataset'
    files = glob.glob(path + "/0/h*.edf")
    for filename in files:
        data = mne.io.read_raw_edf(filename)
        raw_data = data.get_data()
        SZ_Data.append(raw_data)
      # labels.append(1)

    print(len(SZ_Data))

    for i in range(len(SZ_Data)):
        print(np.array(SZ_Data[i]).shape)
    print( np.array(SZ_Data[1]).shape)

    """**normalise**"""

    SZ_Data[0]

    normalise_SZ_data=[]
    for i in  range(len(SZ_Data)):
        std=np.std(SZ_Data[i],axis=1)
        mean=np.mean(SZ_Data[i],axis=1)
        SZ_Data[i]=(SZ_Data[i].transpose()-mean.transpose()).transpose()
        SZ_Data[i]=(SZ_Data[i].transpose()/std.transpose()).transpose()
        normalise_SZ_data.append(SZ_Data[i])

    normalise_SZ_data[3]

    """**windowing**"""

    winSize= sf*sec   # Size of data point (data of 8 sec)
    stride= sf*sec  # sliding window with length winSize and stride 1 sec
    count=0
    windowing_SZ_data=[]
    for sample in normalise_SZ_data:
        for i in range(0,np.shape(sample)[1]-winSize,stride):
            count+=1
            windowing_SZ_data.append(sample[:,i:i+winSize])
    # for i in range(0,np.shape(normalise_SZ_data[3])[1]-winSize,stride):
    #   count+=1
    #   if len(np.shape(windowing_SZ_data))>1:
    #     windowing_SZ_data=np.dstack((windowing_SZ_data,normalise_SZ_data[3][:,i:i+winSize]))
    #   else:
    #     windowing_SZ_data=np.reshape(normalise_SZ_data[3][:,i:i+winSize],(19,np.shape(normalise_SZ_data[3][:,i:i+winSize])[1],1))

    print(count)
    windowing_SZ_data = np.array(windowing_SZ_data)
    the_shape = np.array(windowing_SZ_data).shape
    print(the_shape)
    
    print("Read Healthy Data")
    
    container = windowing_SZ_data
    labels = [0 for i in range(the_shape[0])]
    
    
    # Ready SZ Data
    SZ_Data = []

    path='SZ dataset'
    files = glob.glob(path + "/1/s*.edf")
    for filename in files:
        data = mne.io.read_raw_edf(filename)
        raw_data = data.get_data()
        SZ_Data.append(raw_data)
      # labels.append(1)

    print(len(SZ_Data))

    for i in range(len(SZ_Data)):
        print(np.array(SZ_Data[i]).shape)
    print( np.array(SZ_Data[1]).shape)

    """**normalise**"""

    SZ_Data[0]

    normalise_SZ_data=[]
    for i in  range(len(SZ_Data)):
        std=np.std(SZ_Data[i],axis=1)
        mean=np.mean(SZ_Data[i],axis=1)
        SZ_Data[i]=(SZ_Data[i].transpose()-mean.transpose()).transpose()
        SZ_Data[i]=(SZ_Data[i].transpose()/std.transpose()).transpose()
        normalise_SZ_data.append(SZ_Data[i])

    normalise_SZ_data[3]

    """**windowing**"""

    winSize= sf*sec # Size of data point (data of 8 sec)
    stride= sf*sec # sliding window with length winSize and stride 1 sec
    count=0
    windowing_SZ_data=[]
    for sample in normalise_SZ_data:
        for i in range(0,np.shape(sample)[1]-winSize,stride):
            count+=1
            windowing_SZ_data.append(sample[:,i:i+winSize])
    # for i in range(0,np.shape(normalise_SZ_data[3])[1]-winSize,stride):
    #   count+=1
    #   if len(np.shape(windowing_SZ_data))>1:
    #     windowing_SZ_data=np.dstack((windowing_SZ_data,normalise_SZ_data[3][:,i:i+winSize]))
    #   else:
    #     windowing_SZ_data=np.reshape(normalise_SZ_data[3][:,i:i+winSize],(19,np.shape(normalise_SZ_data[3][:,i:i+winSize])[1],1))

    print(count)
    windowing_SZ_data = np.array(windowing_SZ_data)
    the_shape = np.array(windowing_SZ_data).shape
    print(the_shape)
    
    labels = labels + [1 for i in range(the_shape[0])]
    container = np.concatenate((container, np.array(windowing_SZ_data)))
    return container, np.array(labels)


def ibib_training(sec):
    sf = 250
    print('--- Started Reading Data ---')
    data,labels = read_data(sf,sec)
    print('--- Got Data, Making Rhythms ---')
    data = get_rhythm(sf,sec,data,0)
    x_train,x_test,y_train,y_test = train_test_splitter(data,labels)
    return x_train,x_test,y_train,y_test