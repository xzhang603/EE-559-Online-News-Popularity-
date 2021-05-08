# Author: Xin Zhang

import csv
import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance
from sklearn.linear_model import LinearRegression, Perceptron
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler


def readData(new_file, ref):
    with open(new_file,'r')as file:
        read_file = csv.reader(file)
        data = []

        for ele in read_file:
            data.append(ele)
    ## ref=1 when we want to read feature data
    ## ref=2 when we want to read label data
    if ref==1:
        feature_data = [] ; feature_label = []
        for item in data[1:-1]:
            feature_data.append(np.array([float(i) for i in item]))
        feature_data = np.vstack(feature_data)
        for item in data[0]:
            feature_label.append(item)
        feature_label = np.vstack(feature_label)
        return feature_data, feature_label
    
    if ref==2:
        label_data = []
        for item in data[1:-1]:
            label_data.append(np.array([float(i) for i in item]))
        label_data = np.vstack(label_data)
        return label_data 

def splitData(train_data):
    feature_train = train_data[:,:-1]
    label_train = train_data[:, -1, None] # keep dim
    return feature_train, label_train


def preprocess(fil_tr_feature, 
               fil_tr_label, 
               largeVar_set,
               tr_feature_label):   
    mean_train, std_train = standardize(fil_tr_feature)
    binary_set = onehot_feature() 
    ## Reduce Large Variacne Data
    for i in range(len(tr_feature_label)):
        label = (tr_feature_label[i])[0]
        if label in largeVar_set:
            for j in range(len(fil_tr_label)):
                fil_tr_feature[j][i] = math.log(2+fil_tr_feature[j][i])
                
    ## Sandardize Data Except One-Hot Features
    for i in range(len(tr_feature_label)):
        label = (tr_feature_label[i])[0]
        if label in binary_set:
            continue
        avg = mean_train[i]
        std = std_train[i]
        for j in range(len(fil_tr_label)):
            fil_tr_feature[j][i] = (fil_tr_feature[j][i]-avg)/std
    
    return fil_tr_feature


def standardize(X_train):
        scaler = StandardScaler()
        scaler.fit(X_train)
        
        #Standardized Data
        X_train_standard = scaler.transform(X_train)
        
        #compute std and mean
        mean_train = scaler.mean_
        X1_train = X_train[0]
        X1_train_std = X_train_standard[0]
        std_train = (X1_train-mean_train)/X1_train_std

        return mean_train, std_train

def onehot_feature():
        binary_set = set()
        binary_set.add('n_non_stop_words'); 
        binary_set.add('data_channel_is_lifestyle')
        binary_set.add('data_channel_is_entertainment');
        binary_set.add('data_channel_is_bus'); 
        binary_set.add('data_channel_is_socmed'); 
        binary_set.add('data_channel_is_tech'); 
        binary_set.add('data_channel_is_world'); 
        binary_set.add('weekday_is_monday'); 
        binary_set.add('weekday_is_tuesday'); 
        binary_set.add('weekday_is_wednesday'); 
        binary_set.add('weekday_is_thursday'); 
        binary_set.add('weekday_is_friday'); 
        binary_set.add('weekday_is_saturday'); 
        binary_set.add('weekday_is_sunday'); 
        binary_set.add('is_weekend'); 
        return binary_set