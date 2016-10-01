
def _type(self, new_type=None, async=False):
    if new_type is None:
        return self.__module__ + '.' + self.__class__.__name__

    if isinstance(new_type, str):
        new_type = _import_dotted_name(new_type)
    if new_type == type(self):
        return self
    return new_type(self.size()).copy_(self, async)

def _cuda(self, idx=None, async=False):
    import torch.cuda
    # This already is a CUDA tensor.
    # Let's check if it needs to be transfered to another GPU.
    if hasattr(self, 'get_device'):
        target_device = idx if idx else torch.cuda.current_device()
        if self.get_device() != target_device:
            with torch.cuda.device(target_device):
                return type(self)(self.size()).copy_(self, async)
        else:
            return self
    else:
        ctx = torch.cuda.device(idx) if idx else torch.cuda._dummy_ctx()
        with ctx:
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

