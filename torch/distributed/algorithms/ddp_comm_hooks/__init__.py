from enum import Enum
from functools import partial

import torch.distributed.algorithms.ddp_comm_hooks.default_hooks as default
import torch.distributed.algorithms.ddp_comm_hooks.quantization_hooks as quantization

# Just a wrapper for DDPCommHookType Enum class to store hooks as partial functions.
def ddp_comm_hook_wrapper(comm_hook, model, state):
    model._register_comm_hook(state, comm_hook)

# DDPCommHookType enumerates the hooks of ``torch.distributed.algorithms.ddp_comm_hooks``
# as names and ``ddp_comm_hook_wrapper`` partials with hook specified. As an example, you can
# register allreduce hook by ``DDPCommHookType.ALLREDUCE.value(model=model, state=process_group)``.
class DDPCommHookType(Enum):
    ALLREDUCE = partial(ddp_comm_hook_wrapper, comm_hook=default.allreduce_hook)
    FP16_COMPRESS = partial(ddp_comm_hook_wrapper, comm_hook=default.fp16_compress_hook)
    QUANTIZE_PER_TENSOR = partial(
        ddp_comm_hook_wrapper, comm_hook=quantization.quantization_pertensor_hook
    )
    QUANTIZE_PER_CHANNEL = partial(
        ddp_comm_hook_wrapper, comm_hook=quantization.quantization_perchannel_hook
    )

# Registers the hooks of ``torch.distributed.algorithms.ddp_comm_hooks`` by taking ``comm_hook_name``
# input as a ``str```. Also, checks whether hook is in ``DDPCommHookType``.
def register_ddp_comm_hook(comm_hook_name, model, state=None):
    assert comm_hook_name in DDPCommHookType.__members__.keys(), (
        "%s is not in the supported DDP communication hook types: %s."
        % (comm_hook_name, list(DDPCommHookType.__members__.keys())),
    )
    getattr(DDPCommHookType, comm_hook_name).value(model=model, state=state)
