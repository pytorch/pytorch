import torch
import importlib


def _type(self, new_type=None, async=False):
    """Returns the type if `new_type` is not provided, else casts this object to
    the specified type.

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
    if self.is_sparse:
        if not new_type.is_sparse:
            raise RuntimeError("Cannot cast sparse tensor to dense tensor")
        new_type_name = new_type.__module__ + '.' + new_type.__name__
        new_values_type_name = new_type_name.replace('.sparse', '')
        new_values = self._values().type(new_values_type_name, async)
        return new_type(self._indices(), new_values, self.size())
    if new_type.is_sparse:
        raise RuntimeError("Cannot cast dense tensor to sparse tensor")
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
        if self.get_device() == device:
            return self
    else:
        if device is None:
            device = -1
    with torch.cuda.device(device):
        if self.is_sparse:
            new_type = getattr(torch.cuda.sparse, self.__class__.__name__)
            indices = self._indices().cuda(device, async)
            values = self._values().cuda(device, async)
            return new_type(indices, values, self.size())
        else:
            new_type = getattr(torch.cuda, self.__class__.__name__)
            return new_type(self.size()).copy_(self, async)


def _rebuild_tensor(storage, storage_offset, size, stride):
    class_name = storage.__class__.__name__.replace('Storage', 'Tensor')
    module = importlib.import_module(storage.__module__)
    tensor_class = getattr(module, class_name)
    return tensor_class().set_(storage, storage_offset, size, stride)


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


def _flatten_tensors(tensors):
    """Flatten tensors into a single contiguous 1D buffer"""
    if len(tensors) == 1:
        return tensors[0].contiguous().view(-1)
    numels = [tensor.numel() for tensor in tensors]
    size = sum(numels)
    offset = 0
    flat = tensors[0].new(size)
    for tensor, numel in zip(tensors, numels):
        flat.narrow(0, offset, numel).copy_(tensor, broadcast=False)
        offset += numel
    return flat


def _unflatten_tensors(flat, tensors):
    """View a flat buffer using the sizes of tensors"""
    outputs = []
    offset = 0
    for tensor in tensors:
        numel = tensor.numel()
        outputs.append(flat.narrow(0, offset, numel).view_as(tensor))
        offset += numel
    return tuple(outputs)


def _take_tensors(tensors, size_limit):
    """Groups tensors into lists of up to size_limit bytes"""
    buf = []
    size = 0
    last_type = type(tensors[0]) if len(tensors) > 0 else None
    for tensor in tensors:
        t = type(tensor)
        param_size = tensor.numel() * tensor.element_size()
        if t is not last_type or (size + param_size > size_limit and size > 0):
            yield buf
            last_type = t
            size = 0
            buf = []
        buf.append(tensor)
        size += param_size
    if len(buf) > 0:
        yield buf
