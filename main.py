### main entry
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from news_pop.datas import readData, splitData, preprocess, standardize, onehot_feature
from news_pop.evaluation import heatmap

def main():
    ###########################################################################
    ### Database 
    # Initialization
    tr_feature, tr_feature_label = readData('data/NEWS_Training_data.csv', 1)
    tr_label = readData('data/NEWS_Training_label.csv', 2)
    tr_data = np.concatenate((tr_feature, tr_label), axis=1)

    # Draw frequency histogram
    fig, ax = plt.subplots()
    ax.hist(tr_label, bins=300)
    plt.gca().set(title='Frequency Histogram', 
                 xlabel='Label (number of sharings)', 
                 ylabel='Frequency')
    plt.savefig(os.path.join('log', 'freq_his.png'))

    # Filter outlier
    tr_label_std = np.std(tr_label)
    tr_label_mean = np.mean(tr_label)
    filter_train_data = []

    for item in tr_data:
        if item[-1] < tr_label_mean + 2*tr_label_std:
            filter_train_data.append(item) 
            
    filter_train_data = np.vstack(filter_train_data)
    fil_tr_feature, fil_tr_label = splitData(filter_train_data)

    # Draw frequency histogram for filtered data
    fig, ax = plt.subplots()
    ax.hist(fil_tr_label, bins=100)
    plt.gca().set(title='Filtered Frequency Histogram', 
                 xlabel='Label (number of sharings)',
                 ylabel='Frequency')
    plt.savefig(os.path.join('log', 'fil_freq_his.png'))
    
    # Find Large Variacne Features
    mean_train, std_train = standardize(fil_tr_feature)
    idx = 0;  idx_set = []
    for i in std_train:
        if i > 1000:
            idx_set.append(idx)
        idx += 1
    print("The features that have large variance is: ")
    largeVar_set = set()
    for i in idx_set:
        largeVar_set.add((tr_feature_label[i])[0])
    print(largeVar_set)
    print()
    binary_set = onehot_feature()
    print("The features that is one-hot is: ")
    print(binary_set)

    tr_feature = preprocess(fil_tr_feature, fil_tr_label, largeVar_set, tr_feature_label)

    # TODO: Fisher selection
    # placeholder here

    # Correlation matrix and Plot
    corr_mat = np.corrcoef(tr_feature.T)
    fig, ax = plt.subplots(figsize=(15, 15))
    im, cbar = heatmap(corr_mat, tr_feature_label, tr_feature_label, ax=ax,
                       cmap="YlGn")
    fig.tight_layout()
    plt.savefig(os.path.join('log', 'corr_mat.png'))
    


if __name__ == '__main__':
    main()