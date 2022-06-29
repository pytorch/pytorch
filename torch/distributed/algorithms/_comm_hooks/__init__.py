from enum import Enum
from functools import partial

from torch.distributed.fsdp import FullyShardedDataParallel

from . import default_hooks as default

def _fsdp_comm_hook_wrapper(comm_hook, model, state):
    model.register_comm_hook(state, comm_hook)

LOW_PRECISION_HOOKS = [
    default.fp16_compress_hook,
]

class FSDPCommHookType(Enum):
    """
    FSDPCommHookType enumerates the hooks of ``torch.distributed.algorithms._comm_hooks``
    as names and ``_fsdp_comm_hook_wrapper`` partials with hook specified.
    Here is an example how you can register an allreduce hook:
    ``FSDPCommHookType.ALLREDUCE.value(model=model, state=process_group)``.
    """

    ALLREDUCE = partial(_fsdp_comm_hook_wrapper, comm_hook=default.allreduce_hook)
    FP16_COMPRESS = partial(
        _fsdp_comm_hook_wrapper, comm_hook=default.fp16_compress_hook
    )

def register_fsdp_comm_hook(
    comm_hook_type: FSDPCommHookType, model: FullyShardedDataParallel, state=None
):
    """
    Registers hooks of ``torch.distributed.algorithms._comm_hooks``
    to the FSDP model. User can specify the type of a hook as an enum
    ``FSDPCommHookType`` type using ``comm_hook_type`` input. State input will
    be passed to the model.

    Example::
        >>> register_fsdp_comm_hook(FSDPCommHookType.FP16_COMPRESS, model, state)
    """
    comm_hook_type.value(model=model, state=state)
