import csv
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

from .utils import splitData, preprocess, standardize, onehot_feature
from .dataset import Dataset

# Author: Xin Zhang, Xuan Shi

class TrainDataset(Dataset):
    def __init__(self, 
                 feat_dir, 
                 label_dir, 
                 standard=False, 
                 filter_outlier=False,
                 PCA=False):

        super(TrainDataset, self).__init__(feat_dir, label_dir)

        self.raw_feat, self.feat_lab = self.read_feature(self.feat_dir)
        self.raw_lab = self.read_label(self.label_dir)
        self.raw_data = np.concatenate((self.raw_feat, self.raw_lab), axis=1)

        self.feat = self.raw_feat
        self.lab = self.raw_lab
        self.data = self.raw_data

        if filter_outlier:
            filtered_data = self.filter_outlier(self.data, self.lab)
            _, self.feat, self.lab = filtered_data

        if standard:
            self.feat = self.normlize_large_variance_feat(
                self.feat, self.lab, self.feat_lab)

        poly = PolynomialFeatures(1)
        self.feat = poly.fit_transform(self.feat)
        self.data = np.concatenate((self.feat, self.lab), axis=1)
        self.lab = np.squeeze(self.lab)


    def filter_outlier(self, data, lab):
        lab_std = np.std(lab)
        lab_mean = np.mean(lab)
        filter_data = []

        for item in data:
            if item[-1] < lab_mean + 2*lab_std:
                filter_data.append(item) 
                
        fil_data = np.vstack(filter_data)
        fil_feat, fil_lab = splitData(fil_data)
        return fil_data, fil_feat, fil_lab
