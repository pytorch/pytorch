import torch

global1 = torch.ones(3)


def reset_state():
    global global1
    global1 = torch.ones(3)


def fn(val, call):
    global global1
    global1 += 1
    val = val + global1
    val = call(val)
    val = val + 1
    return val
