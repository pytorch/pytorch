# common functions between the various counters
from functools import wraps

import torch
from torch.utils._pytree import tree_map


def get_shape(i):
    if isinstance(i, torch.Tensor):
        return i.shape
    return i


def shape_wrapper(f):
    @wraps(f)
    def nf(*args, out_val=None, **kwargs):
        args, kwargs, out_shape = tree_map(get_shape, (args, kwargs, out_val))
        return f(*args, out_shape=out_shape, **kwargs)

    return nf


def normalize_tuple(x):
    if not isinstance(x, tuple):
        return (x,)
    return x


# Define the suffixes for different orders of magnitude
suffixes = ["", "K", "M", "B", "T"]


# Thanks BingChat!
def get_suffix_str(number):
    # Find the index of the appropriate suffix based on the number of digits
    # with some additional overflow.
    # i.e. 1.01B should be displayed as 1001M, not 1.001B
    index = max(0, min(len(suffixes) - 1, (len(str(number)) - 2) // 3))
    return suffixes[index]


def convert_num_with_suffix(number, suffix):
    index = suffixes.index(suffix)
    # Divide the number by 1000^index and format it to two decimal places
    value = f"{number / 1000 ** index:.3f}"
    # Return the value and the suffix as a string
    return value + suffixes[index]


def convert_to_percent_str(num, denom):
    if denom == 0:
        return "0%"
    return f"{num / denom:.2%}"
