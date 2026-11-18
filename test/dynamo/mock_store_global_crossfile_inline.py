import torch


global_flag = False


def set_flag_true():
    global global_flag
    global_flag = True


def set_flag_false():
    global global_flag
    global_flag = False


def shadow_list(x):
    global list
    list = x
    return torch.conj(list)


def builtin_only(x):
    size = len(range(len(x)))

    if isinstance(x, list):
        return x[0] + size

    return x + size


def read_global_flag(x):
    return x + global_flag
