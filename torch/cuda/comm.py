import torch
from . import nccl
from torch._utils import _accumulate, _take_tensors, _flatten_tensors, _unflatten_tensors


def broadcast(tensor, devices):
    """Broadcasts a tensor to a number of GPUs.

    Arguments:
        tensor (Tensor): tensor to broadcast.
        devices (Iterable): an iterable of devices among which to broadcast.
          Note that it should be like (src, dst1, dst2, ...), the first element
          of which is the source device to broadcast from.

    Returns:
        A tuple containing copies of the ``tensor``, placed on devices
        corresponding to indices from ``devices``.
    """
    if nccl.is_available([tensor]) and len(set(devices)) == len(devices):
        tensors = [tensor]
        for device in devices[1:]:
            with torch.cuda.device(device):
                tensors.append(type(tensor)(tensor.size()))
        nccl.broadcast(tensors)
        return tuple(tensors)

    return tuple(tensor.cuda(gpu, async=True) for gpu in devices)


def broadcast_coalesced(tensors, devices, buffer_size=10485760):
    """Broadcasts a sequence tensors to the specified GPUs.

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
    for tensor in tensors:
        if tensor.get_device() != devices[0]:
            raise RuntimeError('all tensors must be on devices[0]')
    outputs = [[] for _ in devices]
    # use the original tensors for the first device
    outputs[0].extend(tensors)
    for chunk in _take_tensors(tensors, buffer_size):
        results = broadcast(_flatten_tensors(chunk), devices)
        # use the broadcasted tensors for the remaining devices
        for dst, res in zip(outputs[1:], results[1:]):
            dst.extend(_unflatten_tensors(res, chunk))
    return tuple(outputs)


def reduce_add(inputs, destination=None):
    """Sums tensors from multiple GPUs.

    All inputs should have matching shapes.

    Arguments:
        inputs (Iterable[Tensor]): an iterable of tensors to add.
        destination (int, optional): a device on which the output will be
            placed (default: current device).

    Returns:
        A tensor containing an elementwise sum of all inputs, placed on the
        ``destination`` device.
    """
    # TODO: try to find an input on another gpu, copy it,
    # and accumulate into the copy
    input_size = inputs[0].size()
    for i, inp in enumerate(inputs):
        assert inp.is_cuda, "reduce_add expects all inputs to be on GPUs"
        if inp.size() != input_size:
            got = 'x'.join(str(x) for x in inp.size())
            expected = 'x'.join(str(x) for x in input_size)
            raise ValueError("input {} has invalid size: got {}, but expected "
                             "{}".format(i, got, expected))
    if destination is None:
        destination = torch.cuda.current_device()
    with torch.cuda.device(destination):
        result = type(inp)(input_size).zero_()

    if nccl.is_available(inputs) and inputs[0].get_device() == destination:
        outputs = [result] + [t.new(t.size()) for t in inputs[1:]]
        nccl.reduce(inputs, outputs, root=destination)
        return result

    for inp in inputs:
        input_correct_gpu = inp.cuda(result.get_device())
        result.add_(input_correct_gpu)
    return result


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
    output = []
    itrs = [_take_tensors(tensors, buffer_size) for tensors in inputs]
    for chunks in zip(*itrs):
        flattened = [_flatten_tensors(chunk) for chunk in chunks]
        result = reduce_add(flattened, destination)
        output.extend(_unflatten_tensors(result, chunks[0]))
    return tuple(output)


def scatter(tensor, devices, chunk_sizes=None, dim=0, streams=None):
    """Scatters tensor across multiple GPUs.

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
    if chunk_sizes is None:
        chunks = tensor.chunk(len(devices), dim)
    else:
        assert sum(chunk_sizes) == tensor.size(dim), "given chunk sizes " \
            "don't sum up to the tensor's size (sum(chunk_sizes) == {}, but " \
            "expected {})".format(sum(chunk_sizes), tensor.size(dim))
        assert min(chunk_sizes) > 0, "got a negative chunk_size"
        chunks = [tensor.narrow(dim, start - size, size)
                  for start, size in zip(_accumulate(chunk_sizes), chunk_sizes)]
    chunks = tuple(chunk.contiguous() for chunk in chunks)
    # TODO: copy to a pinned buffer first (if copying from CPU)
    if streams is None:
        streams = [None] * len(devices)
    outputs = []
    for device, chunk, stream in zip(devices, chunks, streams):
        with torch.cuda.device(device), torch.cuda.stream(stream):
            outputs.append(chunk.cuda(device, async=True))
    return tuple(outputs)


def gather(tensors, dim=0, destination=None):
    """Gathers tensors from multiple GPUs.

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
    total_size = 0
    expected_size = list(tensors[0].size())
    for tensor in tensors:
        assert tensor.is_cuda, "gather expects all inputs to be on GPUs"
        expected_size[dim] = tensor.size(dim)
        if list(tensor.size()) != expected_size:
            got = 'x'.join(str(x) for x in tensor.size())
            expected = 'x'.join(str(x) for x in expected_size)
            raise ValueError("gather got an input of invalid size: got {}, "
                             "but expected {}".format(got, expected))
        total_size += tensor.size(dim)
    expected_size[dim] = total_size
    expected_size = torch.Size(expected_size)
    if destination is None:
        destination = torch.cuda.current_device()
    if destination == -1:
        result = getattr(torch, type(tensors[0]).__name__)(expected_size)
    else:
        with torch.cuda.device(destination):
            result = type(tensors[0])(expected_size)

    chunk_start = 0
    # TODO: if copying to CPU, allocate a pinned buffer, do async copies to it,
    # and copy it to regular memory
    for tensor in tensors:
        result.narrow(dim, chunk_start, tensor.size(dim)).copy_(tensor, True)
        chunk_start += tensor.size(dim)
    return result
