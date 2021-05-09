### main entry
import math
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import svm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFECV, SelectFromModel
from sklearn.linear_model import LassoCV, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR

from news_pop.datas import Dataset, TestDataset, TrainDataset
from news_pop.evaluation import (m_r_squared, plt_corr_matrix,
                                 plt_distribution, plt_eval_metrics, pMAE,
                                 pMSE)
from news_pop.models import RFECV_, RBFModule


###############################################################################
### parameter 

standard = True
filter_outlier = True
select_feat = False
pca = True
num_fold = 5
prefix = 'SVR'
x_label = 'Kernel (K) and Regularization parameter (C)'
save_dir = 'log/SVR'

# Linear Regression
# model_type = 'LinearRegression'

# SVR 
model_type = 'SVR'
model_sele_param_kernel = ['rbf', 'linear']

# Lasso
# model_type = 'Lasso'

# Ridge
# model_type = 'Ridge'
# model_sele_param_alpha = [math.exp(i-20) for i in range(24)]

# Perceptron
# model_type = 'Perceptron'

# RBF
# model_type = 'RBF'
# model_sele_param_hidden_size = [30, 50, 100, 150, 200]


def cross_val(k, feat, lab, model):
    kf = KFold(n_splits=k)
    kf.get_n_splits(feat)
    mae_set = []; r2_set = []; pmse_set = []; pmae_set = []; mr2_set = []

    for idx_tr, idx_val in kf.split(feat):
        feat_tr, feat_val = feat[idx_tr], feat[idx_val]
        lab_tr, lab_val = lab[idx_tr], lab[idx_val]
        model.fit(feat_tr, lab_tr)
        pred_te = model.predict(feat_val)
        mae_set.append(mean_absolute_error(lab_val, pred_te))
        r2_set.append(r2_score(lab_val, pred_te))
        pmse_set.append(pMSE(pred_te, lab_val, r=10))
        pmae_set.append(pMAE(pred_te, lab_val, r=10))
        mr2_set.append(m_r_squared(pred_te, lab_val, r=10))
        
    return np.mean(mae_set), np.mean(r2_set), np.mean(pmse_set), np.mean(pmae_set), np.mean(mr2_set)


def trainer(feat_tr, lab_tr, feat_te, lab_te, model):
    mae_set = []; r2_set = []; pmse_set = []; pmae_set = []; mr2_set = []
    model.fit(feat_tr, lab_tr)
    pred_te = model.predict(feat_te)
    mae_set.append(mean_absolute_error(lab_te, pred_te))
    r2_set.append(r2_score(lab_te, pred_te))
    pmse_set.append(pMSE(pred_te, lab_te, r=10))
    pmae_set.append(pMAE(pred_te, lab_te, r=10))
    mr2_set.append(m_r_squared(pred_te, lab_te, r=10))

    return np.mean(mae_set), np.mean(r2_set), np.mean(pmse_set), np.mean(pmae_set), np.mean(mr2_set)


def main():
    ###########################################################################
    ### Database 
    # Initialization
    data_tr = TrainDataset(feat_dir='data/NEWS_Training_data.csv',
                           label_dir='data/NEWS_Training_label.csv',
                           standard=standard,
                           filter_outlier=filter_outlier,)

    data_te = TrainDataset(feat_dir='data/NEWS_Test_data.csv',
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
    ### Model Setting
    if model_type == 'RBF':
        model = []
        model_sele_param = []
        for i in model_sele_param_hidden_size:
            kmeans = KMeans(n_clusters=i, init='random', random_state=0).fit(data_tr.feat)
            centers = kmeans.cluster_centers_
            gamma = (np.prod(np.ptp(data_tr.feat, axis=0)[1:]) / i) ** (1/58)
            gamma_list = [gamma/1024, gamma/512, gamma/256, gamma/128]
            for j in gamma_list:
                model.append(RBFModule(hidden_shape=i, centers=centers,gamma=j))
                model_sele_param.append('M:{}\n g:{:.2f}'.format(i, j))
    elif model_type == 'LinearRegression':
        model = LinearRegression()
    elif model_type == 'SVR':
        model = []
        model_sele_param = []
        for kernel in model_sele_param_kernel:
            if kernel == 'rbf':
                model.append(SVR(kernel=kernel))
                model_sele_param.append('K:{}\n'.format(kernel))
            else:
                C_list = [math.exp(i-2) for i in range(5)]
                for C in C_list:
                    model.append(SVR(kernel=kernel, C=C))
                    model_sele_param.append('K:{}\n C:{:.2f}'.format(kernel, C))
    elif model_type == 'Ridge':
        model = [Ridge(alpha=la) for la in model_sele_param_alpha]
        model_sele_param = model_sele_param_alpha
    elif model_type == 'Lasso':
        model = LassoCV()
    elif model_type == 'Trivial':
        model = None
    else:
        raise NotImplementedError


    ###########################################################################
    ### Model Selection (if necessary)

    if isinstance(model, (list, tuple)):
        # model selection
        mae_set = []; r2_set = []; pmse_set = []; pmae_set = []; mr2_set = []
        for param, sub_model in zip(model_sele_param, model):
            if select_feat:
                selector = SelectFromModel(estimator=sub_model)
                selector.fit(data_tr.feat, data_tr.lab)
                data_tr.feat_reduced = selector.transform(data_tr.feat)
                data_te.feat_reduced = selector.transform(data_te.feat)

                mae, r2, pmse, pmae, mr2 = cross_val(num_fold, 
                                                     data_tr.feat_reduced, 
                                                     data_tr.lab,
                                                     sub_model)
            mae, r2, pmse, pmae, mr2 = cross_val(num_fold, 
                                                 data_tr.feat, 
                                                 data_tr.lab,
                                                 sub_model)

            mae_set.append(mae)
            r2_set.append(r2)
            pmse_set.append(pmse)
            pmae_set.append(pmae)
            mr2_set.append(mr2)

        # save result
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
            x_label=x_label,
            prefix=prefix,
            save_dir=save_dir,
        )

        # get the optimal model
        model = model[np.argmax(np.array(r2_set))]
        print('optim parameter:{}'.format(model_sele_param[np.argmax(np.array(r2_set))]))


    ###########################################################################
    ### PCA feat selection 
    # cannot used together with model selection and feat selection from model
    if pca:
        D = data_tr.feat.shape[1] - 1
        n_comp = [int(i) for i in range(int(D / 10), int(D), 10)]
        mae_set = []; r2_set = []; pmse_set = []; pmae_set = []; mr2_set = []
        for item in n_comp:
            pca_module = PCA(n_components=item)
            pca_module.fit(data_tr.feat)
            data_tr.pca_feat = pca_module.fit_transform(data_tr.feat)
            data_te.pca_feat = pca_module.fit_transform(data_te.feat)

            mae, r2, pmse, pmae, mr2 = trainer(data_tr.feat, 
                                               data_tr.lab,
                                               data_te.feat,
                                               data_te.lab,
                                               model)

            mae_set.append(mae)
            r2_set.append(r2)
            pmse_set.append(pmse)
            pmae_set.append(pmae)
            mr2_set.append(mr2)

        eval_metr = {
            'mae': mae_set,
            'r2': r2_set,
            'pmse': pmse_set,
            'pmae': pmae_set,
            'mr2': mr2_set
        }

        plt_eval_metrics(
            x_data=n_comp,
            y_data=eval_metr,
            x_label='Number of components of PCA',
            prefix=prefix,
            save_dir=save_dir,
        )

        # get the optimal PCA_feat
        n_comp = n_comp[np.argmin(np.array(mae_set))]
        print('PCA: {}'.format(n_comp))


    ###########################################################################
    ### Inference and Save Result

    feat_tr = data_tr.feat
    feat_te = data_te.feat

    if select_feat:
        selector = SelectFromModel(estimator=model)
        selector.fit(data_tr.feat, data_tr.lab)
        data_tr.feat_reduced = selector.transform(data_tr.feat)
        data_te.feat_reduced = selector.transform(data_te.feat)
        feat_tr = data_tr.feat_reduced
        feat_te = data_te.feat_reduced

    if pca:
        pca_module = PCA(n_components=n_comp)
        pca_module.fit(data_tr.feat)
        data_tr.pca_feat = pca_module.fit_transform(data_tr.feat)
        data_te.pca_feat = pca_module.fit_transform(data_te.feat)
        feat_tr = data_tr.pca_feat
        feat_te = data_te.pca_feat


    model.fit(feat_tr, data_tr.lab)
    pickle.dump(model, open(os.path.join(save_dir, prefix + '.pkl'), 'wb'))
    pred_te = model.predict(feat_te)
    print('{} model measure on test set'.format(model_type))
    
    print('MAE: {}'.format(mean_absolute_error(data_te.lab, pred_te)))
    print('R2: {}'.format(r2_score(data_te.lab, pred_te)))
    print('pMSE: {}'.format(pMSE(pred_te, data_te.lab, r=10)))
    print('pMAE: {}'.format(pMAE(pred_te, data_te.lab, r=10)))
    print('mR2: {}'.format(m_r_squared(pred_te, data_te.lab, r=10)))



if __name__ == '__main__':
    main()
