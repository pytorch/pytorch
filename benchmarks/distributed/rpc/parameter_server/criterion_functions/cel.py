import torch.nn as nn


def cel(rank, model):
    r"""A function that creates a CrossEntropyLoss
    criterion for training.
    Args:
        rank (int): worker rank
        model (nn.Module): neural network model
    """
    return nn.CrossEntropyLoss().cuda(rank)
