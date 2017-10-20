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
        new_indices_type_name = new_type.__module__ + '.LongTensor'
        new_indices = self._indices().type(new_indices_type_name, async)
        return new_type(new_indices, new_values, self.size())
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
    """Flatten tensors into a sequence of contiguous 1D buffers. Assume tensors
    are of same type.

    In case of dense tensors, the resulting tensor will be a concatenated 1D
    buffer. Element-wise operation on this buffer will be equivalent to
    operating separately.

    In case of sparse tensors, the result will be a tuple of two flat tensors,
    one for indices and one for values.
    """
    if tensors[0].is_sparse:
        flat_indices = _flatten_tensors([t._indices() for t in tensors])
        flat_values = _flatten_tensors([t._values() for t in tensors])
        return flat_indices, flat_values
    else:
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
    """View a flat buffer using the sizes of tensors. Assume that tensors are of
    same type, and that flat is given by _flatten_tensors.
    """
    if tensors[0].is_sparse:
        flat_indices, flat_values = flat
        indices = _unflatten_tensors(flat_indices, [t._indices() for t in tensors])
        values = _unflatten_tensors(flat_values, [t._values() for t in tensors])
        outputs = []
        for t, i, v in zip(tensors, indices, values):
            outputs.append(t.new(i, v, t.size()))
        return tuple(outputs)
    else:
        outputs = []
        offset = 0
        for tensor in tensors:
            numel = tensor.numel()
            outputs.append(flat.narrow(0, offset, numel).view_as(tensor))
            offset += numel
        return tuple(outputs)


def _reorder_tensors_as(tensors, ordered_tensors):
    """Assume that tensors are of same order as ordered_tensors within sparse
    and dense classes. Reordered them to be of same order as ordered_tensors.
    """
    sparse_tensors = iter(t for t in tensors if t.is_sparse)
    dense_tensors = iter(t for t in tensors if not t.is_sparse)
    outputs = []
    for t in ordered_tensors:
        if t.is_sparse:
            outputs.append(next(sparse_tensors))
        else:
            outputs.append(next(dense_tensors))
    return tuple(outputs)

def _take_tensors(tensors, size_limit, sparse_single_chunk):
    """Groups tensors into chunks. This generator yeilds a chunk at each call,
    each containing tensors of up to certain byte limit in total size.

    If sparse_single_chunk=True, yielded tensors are guaranteed to be only of
    same order within sparse and dense classes, i.e., sparse tensors are of same
    order with those in input tensors, and same for dense ones.

    Args:
        tensors (Sequence): A sequence of tensors to be separated into chunks.
        size_limit (int): The limit of each chunk in bytes.
        sparse_single_chunk (bool): Whether to put each sparse tensor in single
            chunks.
    """
    buf = []
    buf_size = 0
    last_type = None
    last_is_sparse = False
    for tensor in tensors:
        t = type(tensor)
        is_sparse = tensor.is_sparse
        if not is_sparse:
            param_size = tensor.numel() * tensor.element_size()
        elif sparse_single_chunk:
            yield [tensor]
            continue
        else:
            indices = tensor._indices()
            values = tensor._values()
            param_size = indices.numel() * indices.element_size() + values.numel() * values.element_size()
        if (len(buf) > 0 and t is not last_type) or (buf_size + param_size > size_limit and buf_size > 0):
            yield buf
            last_type = t
            buf_size = 0
            buf = []
        buf.append(tensor)
        last_is_sparse = is_sparse
        buf_size += param_size
    if len(buf) > 0:
        yield buf
