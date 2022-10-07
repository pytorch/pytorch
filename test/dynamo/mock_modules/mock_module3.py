import torch


def method1(x, y):
    z = torch.ones(1, 1)  # noqa
    x.append(y)
    return x
