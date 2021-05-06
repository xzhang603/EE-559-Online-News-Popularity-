import csv
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

from .utils import splitData, preprocess, standardize, onehot_feature
from .dataset import Dataset

# Author: Xin Zhang, Xuan Shi

class TestDataset(Dataset):
    def __init__(self, feat_dir, label_dir, standard):

        super(TestDataset, self).__init__(feat_dir, label_dir)

        poly = PolynomialFeatures(1)
        self.feat = self.raw_feat
        self.lab = self.raw_lab
        self.data = self.raw_data

        if standard:
            self.feat = self.normlize_large_variance_feat(
                self.feat, self.lab, self.feat_lab)

        self.feat = poly.fit_transform(self.feat)
        self.data = np.concatenate((self.feat, self.lab), axis=1)

