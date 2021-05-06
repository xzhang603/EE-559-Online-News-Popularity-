#!/usr/bin/env python
# coding: utf-8

import sklearn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import RFECV
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression

class RFECV_(object):
    def __init__(self, reg_model):
        self.model = reg_model
        self.min_features_to_select = 1
        self.selector = RFECV(self.model, step=1, cv=5,scoring='accuracy')
        
    def fit(self, X_tr, Y_tr):
        self.selector.fit(X_tr, Y_tr)
        
    def transform(self, X):
        return self.selector.transform(X)
        
    def selectedFeature(self):
        return self.selector.support_
    
    def selectedRank(self):
        return self.selector.ranking_
    
    def PlotSelection(self):
        plt.figure()
        plt.xlabel("Number of features selected")
        plt.ylabel("Cross validation score (nb of correct classifications)")
        plt.plot(range(self.min_features_to_select,
               len(self.selector.grid_scores_) + self.min_features_to_select),
         self.selector.grid_scores_)
        plt.show()






