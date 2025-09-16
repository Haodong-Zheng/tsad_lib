import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, precision_recall_curve

def calculate_metrics(labels, scores):

    precision, recall, _ = precision_recall_curve(y_true=labels, y_score=scores)
    f1 = 2 * precision * recall / (precision + recall+ 1e-5)
    best_f1 = f1[np.argmax(f1)]
    best_p = precision[np.argmax(f1)]
    best_r = recall[np.argmax(f1)]
    auc_roc = roc_auc_score(labels, scores)
    return {
        "precision": best_p,
        "recall": best_r,
        "f1": best_f1,
        "auc_roc": auc_roc
    }
