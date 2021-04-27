### main entry
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from news_pop.datas import Dataset
from news_pop.evaluation import (m_r_squared, mean_absolute_error,
                                 pMAE, pMSE, r_squared)
from news_pop.evaluation import plt_corr_matrix, plt_distribution
from news_pop.models import RBFModule


# TODO: hparams
model_type = 'RBF'

def main():
    ###########################################################################
    ### Database 
    # Initialization
    # TODO 
    # train and test initialization should be different, I will do that later
    data_tr = Dataset(feat_dir='data/NEWS_Training_data.csv',
                      label_dir='data/NEWS_Training_label.csv')

    data_te = Dataset(feat_dir='data/NEWS_Test_data.csv',
                      label_dir='data/NEWS_Test_label.csv')

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


    ###########################################################################
    ### Model
    # TODO 
    # add other types model with "elif"
    # model must have "fit" & "predict" methods
    # TODO
    # think about how to add grid search for model hyperparameters
    if model_type == 'RBF':
        model = RBFModule(hidden_shape=50)
    else:
        raise NotImplementedError


    ###########################################################################
    ### Train and Inference
    model.fit(data_tr.fil_norm_feat, data_tr.fil_lab)
    pred_tr = model.predict(data_tr.fil_norm_feat)
    pred_te = model.predict(data_te.fil_norm_feat)


    ###########################################################################
    ### Evaluation
    mae = mean_absolute_error(pred_tr, data_tr.fil_lab)
    r2 = r_squared(pred_tr, data_tr.fil_lab)
    pmse = pMSE(pred_tr, data_tr.fil_lab, r=10)
    pmae = pMAE(pred_tr, data_tr.fil_lab, r=10)
    mr2 = m_r_squared(pred_tr, data_tr.fil_lab, r=10)


if __name__ == '__main__':
    main()
