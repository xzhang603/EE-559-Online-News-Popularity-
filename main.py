### main entry
import os
import math
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import SVR
from sklearn import svm
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans

from news_pop.datas import Dataset, TrainDataset, TestDataset
from news_pop.evaluation import (m_r_squared, mean_absolute_error,
                                 pMAE, pMSE, r_squared)
from news_pop.evaluation import plt_corr_matrix, plt_distribution, plt_eval_metrics
from news_pop.models import RBFModule, RFECV_


# TODO: hparams
standard = True
filter_outlier = False
select_feat = False
num_fold = 5
prefix = 'rbf_std'

# Ridge
# model_type = 'Ridge'
# model_sele_param = [math.exp(i-20) for i in range(24)]

# SVR 
# model_type = 'SVR'

# Linear Regression
# model_type = 'LinearRegression'

# RBF
model_type = 'RBF'
model_sele_param_hidden_size = [100, 300, 500, 1000, 3000, 5000, 10000]


def cross_val(k, data, model):
    kf = KFold(n_splits=k)
    kf.get_n_splits(data.feat)
    mae_set = []; r2_set = []; pmse_set = []; pmae_set = []; mr2_set = []
    feat = data.feat_reduced if select_feat else data.feat

    for idx_tr, idx_val in kf.split(feat):
        feat_tr, feat_val = feat[idx_tr], feat[idx_val]
        lab_tr, lab_val = data.lab[idx_tr], data.lab[idx_val]
        model.fit(feat_tr, lab_tr)
        pred_te = model.predict(feat_val)
        mae_set.append(mean_absolute_error(pred_te, lab_val))
        r2_set.append(r_squared(pred_te, lab_val))
        pmse_set.append(pMSE(pred_te, lab_val, r=10))
        pmae_set.append(pMAE(pred_te, lab_val, r=10))
        mr2_set.append(m_r_squared(pred_te, lab_val, r=10))
        
    return np.mean(mae_set), np.mean(r2_set), np.mean(pmse_set), np.mean(pmae_set), np.mean(mr2_set)



def main():
    ###########################################################################
    ### Database 
    # Initialization
    data_tr = TrainDataset(feat_dir='data/NEWS_Training_data.csv',
                           label_dir='data/NEWS_Training_label.csv',
                           standard=standard,
                           filter_outlier=filter_outlier,)

    data_te = TestDataset(feat_dir='data/NEWS_Test_data.csv',
                          label_dir='data/NEWS_Test_label.csv',
                          standard=standard,)

    """
    # Draw frequency histogram
    plt_distribution(data=data_tr.lab,
                     bins=300,
                     title='Frequency Histogram',
                     xlabel='Label (number of sharings)', 
                     ylabel='Frequency',
                     save_dir=os.path.join('log', 'freq_his.png'))


    # Draw frequency histogram for filtered data
    plt_distribution(data=data_tr.fil_lab, 
                     bins=100,
                     title='Frequency Histogram',
                     xlabel='Label (number of sharings)', 
                     ylabel='Frequency',
                     save_dir=os.path.join('log', 'fil_freq_his.png'))

    # Correlation matrix and Plot
    corr_mat = np.corrcoef(data_tr.fil_norm_feat.T)
    plt_corr_matrix(corr_mat, 
                    data_tr.feat_lab, 
                    os.path.join('log', 'corr_mat.png'))
    """


    ###########################################################################
    ### Model
    if model_type == 'RBF':
        model = []
        model_sele_param = []
        for i in model_sele_param_hidden_size:
            gamma = i / 32
            gamma_list = [gamma/1024, gamma/512, gamma/256, gamma/128]
            for j in gamma_list:
                model.append(RBFModule(hidden_shape=i, gamma=j))
                model_sele_param.append(i+j) # just for plot
    elif model_type == 'LinearRegression':
        model = LinearRegression()
    elif model_type == 'SVR':
        model = SVR(kernel="linear")
    elif model_type == 'Ridge':
        model = Ridge
        model = [model(alpha=la) for la in model_sele_param]
    else:
        raise NotImplementedError


    ###########################################################################
    ### Train (model selection and cross validation)

    # original feature
    # __import__('ipdb').set_trace()
    mae_set = []; r2_set = []; pmse_set = []; pmae_set = []; mr2_set = []
    for param, sub_model in zip(model_sele_param, model):

        if select_feat:
            selector = SelectFromModel(estimator=sub_model)
            selector.fit(data_tr.feat, data_tr.lab)
            data_tr.feat_reduced = selector.transform(data_tr.feat)

        mae, r2, pmse, pmae, mr2 = cross_val(num_fold, data_tr, sub_model)

        mae_set.append(mae)
        r2_set.append(r2)
        pmse_set.append(pmse)
        pmae_set.append(pmae)
        mr2_set.append(mr2)


    ###########################################################################
    ### Save Result
    eval_metr = {
        'mae': mae_set,
        'r2': r2_set,
        'pmse': pmse_set,
        'pmae': pmae_set,
        'mr2': mr2_set
    }

    plt_eval_metrics(
        x_data=model_sele_param,
        y_data=eval_metr,
        prefix=prefix,
        save_dir='log'
    )



if __name__ == '__main__':
    main()
