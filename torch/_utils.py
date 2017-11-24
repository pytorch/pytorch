import torch
import importlib
from collections import defaultdict


def _type(self, new_type=None, async=False):
    """Returns the type if `new_type` is not provided, else casts this object to
    the specified type.

    If this is already of the correct type, no copy is performed and the
    original object is returned.

    Args:
        new_type (type or string): The desired type
        async (bool): If ``True``, and the source is in pinned memory and
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
        new_module_name = new_type.__module__.replace('.sparse', '')
        new_values_type_name = new_module_name + '.' + new_type.__name__
        new_values = self._values().type(new_values_type_name, async)
        new_indices_type_name = new_module_name + '.LongTensor'
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
        async (bool): If ``True`` and the source is in pinned memory, the copy will
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


def _flatten_dense_tensors(tensors):
    """Flatten dense tensors into a contiguous 1D buffer. Assume tensors are of
    same dense type.

    Since inputs are dense, the resulting tensor will be a concatenated 1D
    buffer. Element-wise operation on this buffer will be equivalent to
    operating individually.

    Arguments:
        tensors (Iterable[Tensor]): dense tensors to flatten.

    Returns:
        A contiguous 1D buffer containing input tensors.
    """
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


def _flatten_sparse_tensors(tensors):
    """Flatten sparse tensors into two contiguous 1D buffers, one of indices and
    one of values. Assume tensors are of same sparse type.

    Arguments:
        tensors (Iterable[Tensor]): sparse tensors to flatten.

    Returns:
        A tuple of two contiguous 1D buffers, one containing input tensors'
        indices and the other containing the values.
    """
    flat_indices = _flatten_dense_tensors([t._indices() for t in tensors])
    flat_values = _flatten_dense_tensors([t._values() for t in tensors])
    return flat_indices, flat_values


def _unflatten_dense_tensors(flat, tensors):
    """View a flat buffer using the sizes of tensors. Assume that tensors are of
    same dense type, and that flat is given by _flatten_dense_tensors.

    Arguments:
        flat (Tensor): flattened dense tensors to unflatten.
        tensors (Iterable[Tensor]): dense tensors whose sizes will be used to
          unflatten flat.

    Returns:
        Unflattened dense tensors with sizes same as tensors and values from
        flat.
    """
    outputs = []
    offset = 0
    for tensor in tensors:
        numel = tensor.numel()
        outputs.append(flat.narrow(0, offset, numel).view_as(tensor))
        offset += numel
    return tuple(outputs)


def _unflatten_sparse_tensors(flat, tensors):
    """View flat buffer (containing indices and values) using the sizes of
    tensors. Assume that tensors are of same sparse type, and that flat is given
    by _flatten_sparse_tensors.

    Arguments:
        flat (tuple(Tensor, Tensor)): flattened indices and values of sparse
          tensors to unflatten.
        tensors (Iterable[Tensor]): sparse tensors whose sizes will be used to
          unflatten flat.

    Returns:
        Unflattened sparse tensors with sizes same as tensors and values from
        flat.
    """
    flat_indices, flat_values = flat
    indices = _unflatten_dense_tensors(flat_indices, [t._indices() for t in tensors])
    values = _unflatten_dense_tensors(flat_values, [t._values() for t in tensors])
    outputs = []
    for t, i, v in zip(tensors, indices, values):
        outputs.append(t.new(i, v, t.size()))
    return tuple(outputs)


def _reorder_tensors_as(tensors, ordered_tensors):
    """Assume that tensors are of same order as ordered_tensors within their
    types, e.g., from _take_tensors. Reorder them to be of same order as
    ordered_tensors.

    Arguments:
        tensors (Iterable[Tensor]): tensors to be reordered. They should be of
          the same order as ordered_tensors within their own types.
        ordered_tensors (Iterable[Tensor]): tensors whose order will be the
          reference.

    Returns:
        Ordered tuple of tensors with contents from tensors and order of
        ordered_tensors.
    """
    type_dict = defaultdict(list)
    for tensor in tensors:
        type_dict[type(tensor)].append(tensor)
    type_dict = {t: iter(coll) for t, coll in type_dict.items()}
    return tuple(next(type_dict[type(tensor)]) for tensor in ordered_tensors)


def _take_tensors(tensors, size_limit):
    """Group tensors into chunks. This generator yields a chunk at each time,
    each containing tensors of same type up to certain byte limit in total size.

    Args:
        tensors (Sequence): A sequence of tensors to be separated into chunks.
        size_limit (int): The limit of each chunk in bytes.

    Yields:
        Blocks of tensors of same type and within size_limit. The yielded
        tensors are only ordered as the original sequence within its types.
    """
    buf_dict = defaultdict(lambda: [[], 0])
    for tensor in tensors:
        t = type(tensor)
        if tensor.is_sparse:
            indices = tensor._indices()
            values = tensor._values()
            size = indices.numel() * indices.element_size() + values.numel() * values.element_size()
        else:
            size = tensor.numel() * tensor.element_size()
        buf_and_size = buf_dict[t]
        if buf_and_size[1] + size > size_limit and buf_and_size[1] > 0:
            yield buf_and_size[0]
            buf_and_size = buf_dict[t] = [[], 0]
        buf_and_size[0].append(tensor)
        buf_and_size[1] += size
    for buf, _ in buf_dict.values():
        if len(buf) > 0:
            yield buf
