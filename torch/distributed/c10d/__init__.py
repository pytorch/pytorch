import torch


def is_available():
    return hasattr(torch._C, "_c10d_init")


if is_available() and not torch._C._c10d_init():
    raise RuntimeError("c10d initialization failed")


if is_available():
    from .rendezvous import rendezvous, register_rendezvous_handler
    from . import BroadcastOptions, AllreduceOptions

    DEFAULT_REDUCE_OPTIONS = AllreduceOptions()

    def broadcast(tensor, src, process_group):
        opts = BroadcastOptions()
        opts.rootRank = src
        opts.rootTensor = 0
        return process_group.broadcast([tensor], opts)

    def all_reduce(tensor, process_group, opts=DEFAULT_REDUCE_OPTIONS):
        return process_group.allreduce([tensor], opts)
