import torch


def _type(self, new_type=None, async=False):
    """Casts this object to the specified type.

    If this is already of the correct type, no copy is performed and the
    original object is returned.

    Args:
        new_type (type or string): The desired type
        async (bool): If True, and the source is in pinned memory and
                      destination is on the GPU or vice versa, the copy is
                      performed asynchronously with respect to the host.
                      Otherwise, the argument has no effect.
    """
    if new_type is None:
        return self.__module__ + '.' + self.__class__.__name__

    if isinstance(new_type, str):
        new_type = _import_dotted_name(new_type)
    if new_type == type(self):
        return self
    return new_type(self.size()).copy_(self, async)


def _cuda(self, device=None, async=False):
    """Returns a copy of this object in CUDA memory.

    If this object is already in CUDA memory and on the correct device, then
    no copy is performed and the original object is returned.

    Args:
        device (int): The destination GPU id. Defaults to the current device.
        async (bool): If True and the source is in pinned memory, the copy will
                      be asynchronous with respect to the host. Otherwise, the
                      argument has no effect.
    """
    if self.is_cuda:
        if device is None:
            device = torch.cuda.current_device()
        if self.get_device() != device:
            with torch.cuda.device(device):
                return type(self)(self.size()).copy_(self, async)
        else:
            return self
    else:
        if device is None:
            device = -1
        with torch.cuda.device(device):
            return self.type(getattr(torch.cuda, self.__class__.__name__), async)


def _range(*args, **kwargs):
    return __builtins__['range'](*args, **kwargs)


def _import_dotted_name(name):
    components = name.split('.')
    obj = __import__(components[0])
    for component in components[1:]:
        obj = getattr(obj, component)
    return obj


# Taken from python 3.5 docs
def _accumulate(iterable, fn=lambda x, y: x + y):
    'Return running totals'
    # _accumulate([1,2,3,4,5]) --> 1 3 6 10 15
    # _accumulate([1,2,3,4,5], operator.mul) --> 1 2 6 24 120
    it = iter(iterable)
    try:
        total = next(it)
    except StopIteration:
        return
    yield total
    for element in it:
        total = fn(total, element)
        yield total
