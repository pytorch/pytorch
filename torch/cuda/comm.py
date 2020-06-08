import warnings

import torch

from . import nccl
from torch._utils import _take_tensors, _flatten_dense_tensors, \
    _unflatten_dense_tensors, _reorder_tensors_as


def broadcast(tensor, devices=None, *, out=None):
    r"""Broadcasts a tensor to a number of GPUs.

    Arguments:
        tensor (Tensor): tensor to broadcast.
        devices (Iterable): an iterable of devices among which to broadcast.
          Note that it should be like (src, dst1, dst2, ...), the first element
          of which is the source device to broadcast from.

    Returns:
        A tuple containing copies of the ``tensor``, placed on devices
        corresponding to indices from ``devices``.
    """
    if not ((devices is None) ^ (out is None)):
        raise RuntimeError(
            "Exactly one of 'devices' and 'out' must be specified, but got "
            "devices={} and out={}".format(devices, out))
    if devices is not None:
        devices = [torch.cuda._utils._get_device_index(d) for d in devices]
        return torch._C._broadcast(tensor, devices)
    else:
        return torch._C._broadcast_out(tensor, out)


def broadcast_coalesced(tensors, devices, buffer_size=10485760):
    r"""Broadcasts a sequence tensors to the specified GPUs.
    Small tensors are first coalesced into a buffer to reduce the number
    of synchronizations.

    Arguments:
        tensors (sequence): tensors to broadcast.
        devices (Iterable): an iterable of devices among which to broadcast.
          Note that it should be like (src, dst1, dst2, ...), the first element
          of which is the source device to broadcast from.
        buffer_size (int): maximum size of the buffer used for coalescing

    Returns:
        A tuple containing copies of the ``tensor``, placed on devices
        corresponding to indices from ``devices``.
    """
    devices = [torch.cuda._utils._get_device_index(d) for d in devices]
    return torch._C._broadcast_coalesced(tensors, devices, buffer_size)


def reduce_add(inputs, destination=None):
    """Sums tensors from multiple GPUs.

    All inputs should have matching shapes, dtype, and layout. The output tensor
    will be of the same shape, dtype, and layout.

    Arguments:
        inputs (Iterable[Tensor]): an iterable of tensors to add.
        destination (int, optional): a device on which the output will be
            placed (default: current device).

    Returns:
        A tensor containing an elementwise sum of all inputs, placed on the
        :attr:`destination` device.
    """
    # TODO: try to find an input on another gpu, copy it,
    #       and accumulate into the copy
    destination = torch.cuda._utils._get_device_index(destination, optional=True)
    input_size = inputs[0].size()
    root_index = None  # index of input tensor that already is on the correct device
    for i, inp in enumerate(inputs):
        assert inp.is_cuda, "reduce_add expects all inputs to be on GPUs"
        if inp.get_device() == destination:
            root_index = i
        if inp.size() != input_size:
            got = 'x'.join(str(x) for x in inp.size())
            expected = 'x'.join(str(x) for x in input_size)
            raise ValueError("input {} has invalid size: got {}, but expected "
                             "{}".format(i, got, expected))
    if root_index is None:
        raise RuntimeError("reduce_add expects destination to be on the same GPU with one of the tensors")

    # clone inputs[root_index] and accuimulate into the copy
    inputs = list(inputs)  # create a copy
    inputs[root_index] = inputs[root_index].clone()

    if nccl.is_available(inputs):
        nccl.reduce(inputs, root=root_index)
    else:
        for i in range(len(inputs)):
            if i != root_index:
                inputs[root_index].add_(inputs[i].cuda(destination, non_blocking=True))
    return inputs[root_index]


def reduce_add_coalesced(inputs, destination=None, buffer_size=10485760):
    """Sums tensors from multiple GPUs.

    Small tensors are first coalesced into a buffer to reduce the number
    of synchronizations.

    Arguments:
        inputs (Iterable[Iterable[Tensor]]): iterable of iterables that
            contain tensors from a single device.
        destination (int, optional): a device on which the output will be
            placed (default: current device).
        buffer_size (int): maximum size of the buffer used for coalescing

    Returns:
        A tuple of tensors containing an elementwise sum of each group of
        inputs, placed on the ``destination`` device.
    """
    # TODO: When `len(inputs) == 1` and all inputs are on `destination`, just
    #       return `inputs`.
    dense_tensors = [[] for _ in inputs]  # shape (num_gpus, num_tensors)
    output = []
    ref_order = []
    # process sparse ones first since they may have different sizes on different gpus
    for tensor_at_gpus in zip(*inputs):
        if all(t.is_sparse for t in tensor_at_gpus):
            result = reduce_add(tensor_at_gpus, destination)  # this will be sparse too
            output.append(result)
            ref_order.append(tensor_at_gpus[0])
        else:
            for coll, t in zip(dense_tensors, tensor_at_gpus):
                coll.append(t.to_dense() if t.is_sparse else t)
            ref_order.append(dense_tensors[0][-1])
    itrs = [_take_tensors(tensors, buffer_size) for tensors in dense_tensors]
    # now the dense ones, which have consistent sizes
    for chunks in zip(*itrs):
        flat_tensors = [_flatten_dense_tensors(chunk) for chunk in chunks]  # (num_gpus,)
        flat_result = reduce_add(flat_tensors, destination)
        for t in _unflatten_dense_tensors(flat_result, chunks[0]):
            # The unflattened tensors do not share storage, and we don't expose
            # base flat tensor anyways, so give them different version counters.
            # See NOTE [ Version Counter in comm.*_coalesced ]
            output.append(t.data)
    return tuple(_reorder_tensors_as(output, ref_order))


def scatter(tensor, devices=None, chunk_sizes=None, dim=0, streams=None, *, out=None):
    r"""Scatters tensor across multiple GPUs.

    Arguments:
        tensor (Tensor): tensor to scatter.
        devices (Iterable[int]): iterable of ints, specifying among which
            devices the tensor should be scattered.
        chunk_sizes (Iterable[int], optional): sizes of chunks to be placed on
            each device. It should match ``devices`` in length and sum to
            ``tensor.size(dim)``. If not specified, the tensor will be divided
            into equal chunks.
        dim (int, optional): A dimension along which to chunk the tensor.

    Returns:
        A tuple containing chunks of the ``tensor``, spread across given
        ``devices``.
    """
    if out is None:
        devices = [torch.cuda._utils._get_device_index(d) for d in devices]
        return tuple(torch._C._scatter(tensor, devices, chunk_sizes, dim, streams))
    else:
        if devices is not None:
            raise RuntimeError(
                "'devices' must not be specified when 'out' is specified, but "
                "got devices={}".format(devices))
        if chunk_sizes is not None:
            raise RuntimeError(
                "'chunk_sizes' must not be specified when 'out' is specified, "
                "but got chunk_sizes={}".format(chunk_sizes))
        return tuple(torch._C._scatter_out(tensor, out, dim, streams))

def gather(tensors, dim=0, destination=None, *, out=None):
    r"""Gathers tensors from multiple GPUs.

    Tensor sizes in all dimension different than ``dim`` have to match.

    Arguments:
        tensors (Iterable[Tensor]): iterable of tensors to gather.
        dim (int): a dimension along which the tensors will be concatenated.
        destination (int, optional): output device (-1 means CPU, default:
            current device)

    Returns:
        A tensor located on ``destination`` device, that is a result of
        concatenating ``tensors`` along ``dim``.
    """
    if out is None:
        if destination == -1:
            warnings.warn(
                'Using -1 to represent CPU tensor is deprecated. Please use a '
                'device object or string instead, e.g., "cpu".')
        destination = torch.cuda._utils._get_device_index(destination, allow_cpu=True, optional=True)
        return torch._C._gather(tensors, dim, destination)
    else:
        if destination is not None:
            raise RuntimeError(
                "'destination' must not be specified when 'out' is specified, but "
                "got destination={}".format(destination))
        return torch._C._gather_out(tensors, out, dim)
