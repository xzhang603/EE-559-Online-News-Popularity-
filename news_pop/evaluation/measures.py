import numpy as np

"""
pred: the predicted value, shape (1-D) : (N,) 
gt_lab: the ground truth, shape (1-D) : (N,)
"""

def mean_absolute_error(pred, gt_lab):
    assert pred.shape == gt_lab.shape
    N = pred.shape[0]
    mae = np.sum(np.abs(pred - gt_lab)) / N
    return mae


def r_squared(pred, gt_lab):
    assert pred.shape == gt_lab.shape
    N = pred.shape[0]
    y_mean = np.ones_like(gt_lab) * np.mean(gt_lab)
    r_squared = 1 - np.dot((pred-gt_lab).T, (pred-gt_lab)) / np.dot((gt_lab-y_mean).T, (gt_lab-y_mean))
    return r_squared


def pMSE(pred, gt_lab, r=10):
    assert pred.shape == gt_lab.shape
    N = pred.shape[0]
    tmp = (gt_lab - pred) / (r + gt_lab)
    pmse = np.dot(tmp.T, tmp) / N
    return pmse


def pMAE(pred, gt_lab, r=10):
    assert pred.shape == gt_lab.shape
    N = pred.shape[0]
    tmp = (gt_lab - pred) / (r + gt_lab)
    pmae = np.sum(np.abs(tmp)) / N
    return pmae


def m_r_squared(pred, gt_lab, r=10):
    assert pred.shape == gt_lab.shape
    N = pred.shape[0]
    y_mean = np.ones_like(gt_lab) * np.mean(gt_lab)
    m_r_squared = 1 - pMSE(pred, gt_lab, r=r) / pMSE(y_mean, gt_lab, r=r)
    return m_r_squared
