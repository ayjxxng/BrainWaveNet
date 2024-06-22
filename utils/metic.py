import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score


def get_pred(predictions: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculate predictions and probabilities from model outputs."""
    stack = torch.from_numpy(np.vstack(predictions))
    _, preds = torch.max(F.softmax(stack, -1).data, -1)
    proba = F.softmax(stack, -1).data[:, 1:].squeeze(-1)

    return preds, proba


def confusion(g_true: List[torch.Tensor], predictions: List[torch.Tensor]) -> Tuple[float, ...]:
    """Calculate confusion matrix metrics from ground truth and predictions."""
    pred, prob = get_pred(predictions)
    g_true = torch.from_numpy(np.vstack(g_true))
    _, lab = torch.max(g_true, -1)
    acc = accuracy_score(lab, pred)

    tn, fp, fn, tp = confusion_matrix(lab, pred).ravel()
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    auc = roc_auc_score(lab, prob)

    precision = tp / (tp + fp)
    recall = sens
    f1 = 2 * (precision * recall) / (precision + recall)

    return acc, sens, spec, auc, f1, recall, precision
    
