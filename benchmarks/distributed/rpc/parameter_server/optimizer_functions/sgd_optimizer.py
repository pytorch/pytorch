import torch


def sgd_optimizer(parameters, lr):
    r"""
    A function that creates a SGD optimizer for training.
    Args:
        parameters (iterable): iterable of parameters to optimize
        lr (float): learning rate
    """
    return torch.optim.SGD(parameters, lr)
