import warnings
import torch.cuda

__all__ = ['all_reduce', 'reduce', 'broadcast', 'all_gather', 'reduce_scatter']

SUM = 0  # ncclRedOp_t


def is_available(tensors):
    devices = set()
    for tensor in tensors:
        if tensor.is_sparse:
            return False
        if not tensor.is_contiguous():
            return False
        if not tensor.is_cuda:
            return False
        device = tensor.get_device()
        if device in devices:
            return False
        devices.add(device)

    if not hasattr(torch._C, '_nccl_all_reduce'):
        warnings.warn('PyTorch is not compiled with NCCL support')
        return False

    return True


def version():
    return torch._C._nccl_version()


def unique_id():
    return torch._C._nccl_unique_id()


def init_rank(num_ranks, uid, rank):
    return torch._C._nccl_init_rank(num_ranks, uid, rank)


def all_reduce(inputs, outputs=None, op=SUM, streams=None, comms=None):
    if outputs is None:
        outputs = inputs
    torch._C._nccl_all_reduce(inputs, outputs, op, streams, comms)


def reduce(inputs, outputs=None, root=0, op=SUM, streams=None, comms=None):
    if outputs is None:
        outputs = inputs
    torch._C._nccl_reduce(inputs, outputs, root, op, streams, comms)


def broadcast(inputs, root=0, streams=None, comms=None):
    torch._C._nccl_broadcast(inputs, root, streams, comms)


def all_gather(inputs, outputs, streams=None, comms=None):
    torch._C._nccl_all_gather(inputs, outputs, streams, comms)


def reduce_scatter(inputs, outputs, op=SUM, streams=None, comms=None):
    torch._C._nccl_reduce_scatter(inputs, outputs, op, streams, comms)
