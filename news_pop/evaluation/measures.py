import numpy as np

"""
pred: the predicted value, shape: (N, D) 
gt_lab: the ground truth, shape: (N, D)
N: number of samples
D: number of candidate models
"""

def mean_absolute_error(pred, gt_lab):
    assert pred.shape == gt_lab.shape
    N = pred.shape[0]
    mae = np.sum(np.abs(pred - gt_lab), axis=0) / N
    return mae


def r_squared(pred, gt_lab):
    assert pred.shape == gt_lab.shape
    N = pred.shape[0]
    y_mean = np.ones_like(gt_lab) * np.mean(gt_lab)
    r_squared = 1 - np.sum((pred-gt_lab) ** 2, axis=0) / np.sum((pred-y_mean) ** 2, axis=0)
    return r_squared


def pMSE(pred, gt_lab, r=10):
    assert pred.shape == gt_lab.shape
    N = pred.shape[0]
    tmp = (gt_lab - pred) / (r + gt_lab)
    pmse = np.sum(tmp ** 2, axis=0) / N
    return pmse


def pMAE(pred, gt_lab, r=10):
    assert pred.shape == gt_lab.shape
    N = pred.shape[0]
    tmp = (gt_lab - pred) / (r + gt_lab)
    pmae = np.sum(np.abs(tmp), axis=0) / N
    return pmae


def m_r_squared(pred, gt_lab, r=10):
    assert pred.shape == gt_lab.shape
    N = pred.shape[0]
    y_mean = np.ones_like(gt_lab) * np.mean(gt_lab)
    m_r_squared = 1 - pMSE(pred, gt_lab, r=r) / pMSE(y_mean, gt_lab, r=r)
    return m_r_squared
