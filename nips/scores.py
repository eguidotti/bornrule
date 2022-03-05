import torch
from sklearn import metrics


def accuracy_score(output, target):
    y_true = torch.argmax(target, dim=1)
    y_pred = torch.argmax(output, dim=1)
    return metrics.accuracy_score(y_true=y_true, y_pred=y_pred)


def roc_auc_score(output, target):
    y_true = torch.argmax(target, dim=1)
    y_score = output[:, 1]
    return metrics.roc_auc_score(y_true=y_true, y_score=y_score)


def l1_loss(output, target):
    return torch.nn.functional.l1_loss(output, target)


def log_loss(output, target):
    return torch.nn.functional.nll_loss(torch.log(output), torch.argmax(target, dim=1))


def mse_loss(output, target):
    return torch.pow(output - target, 2).mean()
