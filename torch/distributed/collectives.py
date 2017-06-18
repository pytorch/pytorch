import torch
from . import _INITIALIZED_PG, _INITIALIZED_MW


class reduce_op(object):
    SUM = object()
    PRODUCT = object()
    MAX = object()
    MIN = object()


class group(object):
    WORLD = object()


class _DistributedRequest(object):
    def __init__(self, request):
        self.request = request

    def is_completed(self):
        return torch._C._dist_request_is_completed(self.request)

    def wait(self):
        torch._C._dist_request_wait(self.request)


def get_rank():
    assert torch.distributed._initialized
    return torch._C._dist_get_rank()


def get_world_size():
    assert torch.distributed._initialized
    return torch._C._dist_get_num_processes()


def isend(tensor, dst):
    assert torch.distributed._initialized == _INITIALIZED_PG, \
        "collective only supported in process-group mode"
    return _DistributedRequest(torch._C._dist_isend(tensor, dst))


def irecv(tensor, src):
    assert torch.distributed._initialized == _INITIALIZED_PG, \
        "collective only supported in process-group mode"
    return _DistributedRequest(torch._C._dist_irecv(tensor, src))


def send(tensor, dst):
    assert torch.distributed._initialized == _INITIALIZED_PG, \
        "collective only supported in process-group mode"
    return torch._C._dist_send(tensor, dst)


def recv(tensor, src=None):
    assert torch.distributed._initialized == _INITIALIZED_PG, \
        "collective only supported in process-group mode"
    if src is None:
        return torch._C._dist_recv_any_source(tensor)
    return torch._C._dist_recv(tensor, src)


def broadcast(tensor, src, group=group.WORLD):
    assert torch.distributed._initialized == _INITIALIZED_PG, \
        "collective only supported in process-group mode"
    return torch._C._dist_broadcast(tensor, src, group)


def all_reduce(tensor, op=reduce_op.SUM, group=group.WORLD):
    assert torch.distributed._initialized == _INITIALIZED_PG, \
        "collective only supported in process-group mode"
    return torch._C._dist_all_reduce(tensor, op, group)


def reduce(tensor, dst, op=reduce_op.SUM, group=group.WORLD):
    assert torch.distributed._initialized == _INITIALIZED_PG, \
        "collective only supported in process-group mode"
    return torch._C._dist_reduce(tensor, dst, op, group)


def all_gather(tensor_list, tensor, group=group.WORLD):
    assert torch.distributed._initialized == _INITIALIZED_PG, \
        "collective only supported in process-group mode"
    return torch._C._dist_all_gather(tensor_list, tensor, group)


def gather_send(tensor, dst, group=group.WORLD):
    assert torch.distributed._initialized == _INITIALIZED_PG, \
        "collective only supported in process-group mode"
    return torch._C._dist_gather_send(tensor, dst, group)


def gather_recv(tensor_list, tensor, group=group.WORLD):
    assert torch.distributed._initialized == _INITIALIZED_PG, \
        "collective only supported in process-group mode"
    return torch._C._dist_gather_recv(tensor_list, tensor, group)


def scatter_send(tensor_list, tensor, group=group.WORLD):
    assert torch.distributed._initialized == _INITIALIZED_PG, \
        "collective only supported in process-group mode"
    return torch._C._dist_scatter_send(tensor_list, tensor, group)


def scatter_recv(tensor, src, group=group.WORLD):
    assert torch.distributed._initialized == _INITIALIZED_PG, \
        "collective only supported in process-group mode"
    return torch._C._dist_scatter_recv(tensor, src, group)


def barrier(group=group.WORLD):
    assert torch.distributed._initialized == _INITIALIZED_PG, \
        "collective only supported in process-group mode"
    return torch._C._dist_barrier(group)


def new_group(ranks=None):
    assert torch.distributed._initialized == _INITIALIZED_PG, \
        "collective only supported in process-group mode"
    if ranks is None:
        ranks = list(range(get_world_size()))
    return torch._C._dist_new_group(ranks)
