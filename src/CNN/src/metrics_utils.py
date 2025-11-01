import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

CLASS_NAMES = ("negative", "positive")

def ir_metrics(y_true_onehot, y_pred_indices):
    y_true = np.argmax(y_true_onehot, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred_indices, labels=[0,1], zero_division=0
    )
    try:
        auc = roc_auc_score(y_true, y_pred_indices)
    except Exception:
        auc = float("nan")
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": np.array([auc, auc]),
    }
