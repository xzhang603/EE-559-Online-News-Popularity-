import csv
import numpy as np

from .utils import splitData, preprocess, standardize, onehot_feature

# Author: Xin Zhang, Xuan Shi

class Dataset(object):
    def __init__(self, feat_dir, label_dir):
        self.feat_dir = feat_dir
        self.label_dir = label_dir

        self.raw_feat, self.feat_lab = self.read_feature(self.feat_dir)
        self.raw_lab = self.read_label(self.label_dir)
        self.raw_data = np.concatenate((self.raw_feat, self.raw_lab), axis=1)

        """
        filtered_data = self.filter_outlier(self.data, self.lab)
        self.fil_data, self.fil_feat, self.fil_lab = filtered_data

        self.fil_norm_feat = self.normlize_large_variance_feat(self.fil_feat, self.fil_lab, self.feat_lab)
        self.norm_feat = self.normlize_large_variance_feat(self.feat, self.lab, self.feat_lab)
        # TODO fisher feature selection: i don't know how to implement that
        """

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
            

    def read_label(self, file_dir):
        # read label data
        with open(file_dir,'r')as file:
            read_file = csv.reader(file)
            data = []

            for ele in read_file:
                data.append(ele)

        label_data = []
        for item in data[1:-1]:
            label_data.append(np.array([float(i) for i in item]))
        label_data = np.vstack(label_data)
        return label_data
        

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
