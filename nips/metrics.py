import torch


def l1_loss(output, target):
    return torch.nn.functional.l1_loss(output, target)


def log_loss(output, target):
    return torch.nn.functional.nll_loss(torch.log(output), torch.argmax(target, dim=1))


def mse_loss(output, target):
    return torch.pow(output - target, 2).mean()
