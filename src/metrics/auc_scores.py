from sklearn import metrics
import numpy as np

def auc_roc(pred, true):
    # Tanh does not change the score. It allows auc score to support 'inf' scores
    # fpr, tpr, thresholds = metrics.roc_curve(true.reshape(-1), np.tanh(pred).reshape(-1))
    fpr, tpr, thresholds = metrics.roc_curve(true.reshape(-1), pred.reshape(-1))
    return metrics.auc(fpr, tpr)


def auc_apr(pred, true):
    # Tanh does not change the score. It allows auc score to support inf scores
    # return metrics.average_precision_score(true.reshape(-1), np.tanh(pred).reshape(-1))
    return metrics.average_precision_score(true.reshape(-1), pred.reshape(-1))
