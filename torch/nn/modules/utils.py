import collections

def _pair(x):
    if isinstance(x, collections.Iterable):
        return x
    return x, x
