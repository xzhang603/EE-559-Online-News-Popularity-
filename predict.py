### main entry
import os
import sys
import csv
import math
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pickle

from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import SVR
from sklearn import svm
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score


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


class TestDataset(object):
    def __init__(self, 
                 feat_dir,
                 standard=False,
                 PCA=False):

        self.feat_dir = feat_dir
        self.raw_feat, self.feat_lab = self.read_feature(self.feat_dir)
        self.raw_lab = np.ones(self.raw_feat.shape[0])

        self.feat = self.raw_feat
        self.lab = self.raw_lab

        if standard:
            self.feat = self.normlize_large_variance_feat(
                self.feat, self.lab, self.feat_lab)

        poly = PolynomialFeatures(1)
        self.feat = poly.fit_transform(self.feat)


    def read_feature(self, file_dir):
        # read feature data
        with open(file_dir,'r')as file:
            read_file = csv.reader(file)
            data = []

            for ele in read_file:
                data.append(ele)
    
        feature_data = [] ; feature_label = []
        for item in data[1:-1]:
            feature_data.append(np.array([float(i) for i in item]))
        feature_data = np.vstack(feature_data)
        for item in data[0]:
            feature_label.append(item)
        feature_label = np.vstack(feature_label)
        return feature_data, feature_label


    def normlize_large_variance_feat(self, feat, lab, feat_lab):
        mean_train, std_train = standardize(feat)
        idx = 0;  idx_set = []
        for i in std_train:
            if i > 1000:
                idx_set.append(idx)
            idx += 1
        print("The features that have large variance is: ")
        largeVar_set = set()
        for i in idx_set:
            largeVar_set.add((feat_lab[i])[0])
        print(largeVar_set)
        print()
        binary_set = onehot_feature()
        print("The features that is one-hot is: ")
        print(binary_set)

        norm_feat = preprocess(feat, lab, largeVar_set, feat_lab)
        return norm_feat


def write_csv(csv_path, data_list):
    outF = open(csv_path, "w")
    for x in data_list:
        x_str = str(x)
        outF.write(x_str)
        outF.write("\n")
    outF.close()


def main(input_dir, output_dir):
    standard=False

    # data
    data_te = TestDataset(feat_dir=input_dir,
                          standard=standard,)

    # model
    model = pickle.load(open('log/SVR/SVR_param_std_fil.pkl', 'rb'))

    # inference
    pred_te = model.predict(data_te.feat)

    write_csv(output_dir, pred_te)



if __name__ == '__main__':
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    main(input_dir, output_dir)
