import csv
import numpy as np

from .utils import splitData, preprocess, standardize, onehot_feature
from .dataset import Dataset

# Author: Xin Zhang, Xuan Shi

class TrainDataset(Dataset):
    def __init__(self, feat_dir, label_dir, standard, filter_outlier):

        super(TrainDataset, self).__init__(feat_dir, label_dir)

        self.feat = self.raw_feat
        self.lab = self.raw_lab
        self.data = self.raw_data

        if filter_outlier:
            filtered_data = self.filter_outlier(self.data, self.lab)
            self.data, self.feat, self.lab = filtered_data

        if standard:
            self.feat = self.normlize_large_variance_feat(
                self.feat, self.lab, self.feat_lab)
            self.data = np.concatenate((self.feat, self.lab), axis=1)


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
