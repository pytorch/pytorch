import torch

def f(x):
    x.fill_diagonal_(True)

x = torch.zeros(4, 4)
f(x)