# mypy: allow-untyped-defs
import warnings
import torch
from torch.cuda import nccl
from torch._utils import _take_tensors, _flatten_dense_tensors, \
    _unflatten_dense_tensors, _reorder_tensors_as, _get_device_index, _handle_complex
from typing import List

def broadcast(tensor, devices=None, *, out=None):
    r"""Broadcasts a tensor to specified GPU devices.

    Args:
        tensor (Tensor): tensor to broadcast. Can be on CPU or GPU.
        devices (Iterable[torch.device, str or int], optional): an iterable of
          GPU devices, among which to broadcast.
        out (Sequence[Tensor], optional, keyword-only): the GPU tensors to
          store output results.

    .. note::
        Exactly one of :attr:`devices` and :attr:`out` must be specified.

    Returns:
        - If :attr:`devices` is specified,
            a tuple containing copies of :attr:`tensor`, placed on
            :attr:`devices`.
        - If :attr:`out` is specified,
            a tuple containing :attr:`out` tensors, each containing a copy of
            :attr:`tensor`.
    """
    tensor = _handle_complex(tensor)
    if not ((devices is None) ^ (out is None)):
        raise RuntimeError(
            f"Exactly one of 'devices' and 'out' must be specified, but got devices={devices} and out={out}")
    if devices is not None:
        devices = [_get_device_index(d) for d in devices]
        return torch._C._broadcast(tensor, devices)
    else:
        return torch._C._broadcast_out(tensor, out)


def broadcast_coalesced(tensors, devices, buffer_size=10485760):
    """Broadcast a sequence of tensors to the specified GPUs.

    Small tensors are first coalesced into a buffer to reduce the number of synchronizations.

    Args:
        tensors (sequence): tensors to broadcast. Must be on the same device,
          either CPU or GPU.
        devices (Iterable[torch.device, str or int]): an iterable of GPU
          devices, among which to broadcast.
        buffer_size (int): maximum size of the buffer used for coalescing

    Returns:
        A tuple containing copies of :attr:`tensor`, placed on :attr:`devices`.
    """
    devices = [_get_device_index(d) for d in devices]
    tensors = [_handle_complex(t) for t in tensors]
    return torch._C._broadcast_coalesced(tensors, devices, buffer_size)


def reduce_add(inputs, destination=None):
    """Sum tensors from multiple GPUs.

    All inputs should have matching shapes, dtype, and layout. The output tensor
    will be of the same shape, dtype, and layout.

    Args:
        inputs (Iterable[Tensor]): an iterable of tensors to add.
        destination (int, optional): a device on which the output will be
            placed (default: current device).

    Returns:
        A tensor containing an elementwise sum of all inputs, placed on the
        :attr:`destination` device.
    """
    destination = _get_device_index(destination, optional=True)
    input_size = inputs[0].size()
    root_index = None  # index of input tensor that already is on the correct device
    for i, inp in enumerate(inputs):
        assert inp.device.type != "cpu", "reduce_add expects all inputs to be on GPUs"
        if inp.get_device() == destination:
            root_index = i
        if inp.size() != input_size:
            got = 'x'.join(str(x) for x in inp.size())
            expected = 'x'.join(str(x) for x in input_size)
            raise ValueError(f"input {i} has invalid size: got {got}, but expected {expected}")
    if root_index is None:
        raise RuntimeError("reduce_add expects destination to be on the same GPU with one of the tensors")

    if len(inputs) == 1:
        return inputs[0]

    if nccl.is_available(inputs):
        result = torch.empty_like(inputs[root_index])
        nccl.reduce(inputs, output=result, root=root_index)
    else:
        destination_device = torch.device(inputs[root_index].device.type, destination)
        nonroot = [t for i, t in enumerate(inputs) if i != root_index]
        # make a new tensor w/o clone
        result = inputs[root_index] + nonroot[0].to(device=destination_device, non_blocking=True)
        for other in nonroot[1:]:
            result.add_(other.to(device=destination_device, non_blocking=True))
    return result


def reduce_add_coalesced(inputs, destination=None, buffer_size=10485760):
    """Sum tensors from multiple GPUs.

    Small tensors are first coalesced into a buffer to reduce the number
    of synchronizations.

    Args:
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
    dense_tensors: List[List] = [[] for _ in inputs]  # shape (num_gpus, num_tensors)
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
    """Scatters tensor across multiple GPUs.

    Args:
        tensor (Tensor): tensor to scatter. Can be on CPU or GPU.
        devices (Iterable[torch.device, str or int], optional): an iterable of
          GPU devices, among which to scatter.
        chunk_sizes (Iterable[int], optional): sizes of chunks to be placed on
          each device. It should match :attr:`devices` in length and sums to
          ``tensor.size(dim)``. If not specified, :attr:`tensor` will be divided
          into equal chunks.
        dim (int, optional): A dimension along which to chunk :attr:`tensor`.
          Default: ``0``.
        streams (Iterable[torch.cuda.Stream], optional): an iterable of Streams, among
          which to execute the scatter. If not specified, the default stream will
          be utilized.
        out (Sequence[Tensor], optional, keyword-only): the GPU tensors to
          store output results. Sizes of these tensors must match that of
          :attr:`tensor`, except for :attr:`dim`, where the total size must
          sum to ``tensor.size(dim)``.

    .. note::
        Exactly one of :attr:`devices` and :attr:`out` must be specified. When
        :attr:`out` is specified, :attr:`chunk_sizes` must not be specified and
        will be inferred from sizes of :attr:`out`.

    Returns:
        - If :attr:`devices` is specified,
            a tuple containing chunks of :attr:`tensor`, placed on
            :attr:`devices`.
        - If :attr:`out` is specified,
            a tuple containing :attr:`out` tensors, each containing a chunk of
            :attr:`tensor`.
    """
    tensor = _handle_complex(tensor)
    if out is None:
        devices = [_get_device_index(d) for d in devices]
        return tuple(torch._C._scatter(tensor, devices, chunk_sizes, dim, streams))
    else:
        if devices is not None:
            raise RuntimeError(
                f"'devices' must not be specified when 'out' is specified, but got devices={devices}")
        if chunk_sizes is not None:
            raise RuntimeError(
                f"'chunk_sizes' must not be specified when 'out' is specified, but got chunk_sizes={chunk_sizes}")
        return tuple(torch._C._scatter_out(tensor, out, dim, streams))


def gather(tensors, dim=0, destination=None, *, out=None):
    r"""Gathers tensors from multiple GPU devices.

    Args:
        tensors (Iterable[Tensor]): an iterable of tensors to gather.
          Tensor sizes in all dimensions other than :attr:`dim` have to match.
        dim (int, optional): a dimension along which the tensors will be
          concatenated. Default: ``0``.
        destination (torch.device, str, or int, optional): the output device.
          Can be CPU or CUDA. Default: the current CUDA device.
        out (Tensor, optional, keyword-only): the tensor to store gather result.
          Its sizes must match those of :attr:`tensors`, except for :attr:`dim`,
          where the size must equal ``sum(tensor.size(dim) for tensor in tensors)``.
          Can be on CPU or CUDA.

    .. note::
        :attr:`destination` must not be specified when :attr:`out` is specified.

    Returns:
        - If :attr:`destination` is specified,
            a tensor located on :attr:`destination` device, that is a result of
            concatenating :attr:`tensors` along :attr:`dim`.
        - If :attr:`out` is specified,
            the :attr:`out` tensor, now containing results of concatenating
            :attr:`tensors` along :attr:`dim`.
    """
    tensors = [_handle_complex(t) for t in tensors]
    if out is None:
        if destination == -1:
            warnings.warn(
                'Using -1 to represent CPU tensor is deprecated. Please use a '
                'device object or string instead, e.g., "cpu".',
                FutureWarning,
                stacklevel=2,
            )
        destination = _get_device_index(destination, allow_cpu=True, optional=True)
        return torch._C._gather(tensors, dim, destination)
    else:
        if destination is not None:
            raise RuntimeError(
                f"'destination' must not be specified when 'out' is specified, but got destination={destination}")
        return torch._C._gather_out(tensors, out, dim)
