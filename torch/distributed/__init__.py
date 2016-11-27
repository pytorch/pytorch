import torch


_initialized = False


def init_process_group(backend):
    global _initialized
    if _initialized:
        raise RuntimeError("trying to initialize torch.distributed twice!")
    torch._C._dist_init_process_group(backend)
    _initialized = True


class reduce_op(object):
    SUM = object()
    PRODUCT = object()
    MAX = object()
    MIN = object()

class group(object):
    WORLD = object()

def get_rank():
    return torch._C._dist_get_rank()


def get_num_processes():
    return torch._C._dist_get_num_processes()


def send(tensor, dst_rank):
    return torch._C._dist_send(tensor, dst_rank)


def recv(tensor, src_rank):
    return torch._C._dist_recv(tensor, src_rank)


def broadcast(tensor, src_rank, group=group.WORLD):
    return torch._C._dist_broadcast(tensor, src_rank, group)


def all_reduce(tensor, op=reduce_op.SUM, group=group.WORLD):
    return torch._C._dist_all_reduce(tensor, op, group)


def reduce(tensor, dst_rank, op=reduce_op.SUM, group=group.WORLD):
    return torch._C._dist_reduce(tensor, dst_rank, op, group)


def new_group(ranks):
    return torch._C._dist_new_group(ranks)

assert torch._C._dist_init_extension(reduce_op, group)
