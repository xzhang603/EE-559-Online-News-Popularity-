import csv
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

from .utils import splitData, preprocess, standardize, onehot_feature
from .dataset import Dataset

# Author: Xin Zhang, Xuan Shi

class TestDataset(Dataset):
    def __init__(self, 
                 feat_dir, 
                 label_dir, 
                 standard=False,
                 PCA=False):

        super(TestDataset, self).__init__(feat_dir, label_dir)

        self.raw_feat, self.feat_lab = self.read_feature(self.feat_dir)

        self.feat = self.raw_feat
        if standard:
            self.feat = self.normlize_large_variance_feat(
                self.feat, self.lab, self.feat_lab)

        poly = PolynomialFeatures(1)
        self.feat = poly.fit_transform(self.feat)
        self.lab = np.squeeze(self.lab)

