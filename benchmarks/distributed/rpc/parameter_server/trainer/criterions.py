import torch.nn as nn


def cel(rank):
    r"""A function that creates a CrossEntropyLoss
    criterion for training.
    Args:
        rank (int): worker rank
    """
    return nn.CrossEntropyLoss().cuda(rank)
