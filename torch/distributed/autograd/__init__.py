
import sys
import torch

from typing import Dict
from enum import Enum

def is_available():
    return hasattr(torch._C, "_dist_autograd_init")


if is_available() and not torch._C._dist_autograd_init():
    raise RuntimeError("Failed to initialize torch.distributed.autograd")

if is_available():
    from torch._C._distributed_autograd import (
        DistAutogradContext,
        _init,
        _get_debug_info,
        _new_context,
        _release_context,
        get_gradients,
        _is_valid_context,
    )

class context(object):
    '''
    Context object to wrap forward and backward passes when using
    distributed autograd. The ``context_id`` generated in the ``with``
    statement  is required to uniquely identify a distributed backward pass
    on all workers. Each worker stores metadata associated with this
    ``context_id``, which is required to correctly execute a distributed
    autograd pass.

    Example::
        >>> import torch.distributed.autograd as dist_autograd
        >>> with dist_autograd.context() as context_id:
        >>>   t1 = torch.rand((3, 3), requires_grad=True)
        >>>   t2 = torch.rand((3, 3), requires_grad=True)
        >>>   loss = rpc.rpc_sync("worker1", torch.add, args=(t1, t2)).sum()
        >>>   dist_autograd.backward(context_id, [loss])
    '''
    def __enter__(self):
        self.autograd_context = _new_context()
        return self.autograd_context._context_id()

    def __exit__(self, type, value, traceback):
        _release_context(self.autograd_context._context_id())
