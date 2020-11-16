from enum import Enum
from functools import partial

import torch.distributed as dist
from . import default_hooks as default
from . import quantization_hooks as quantization
from torch.nn.parallel import DistributedDataParallel


def _ddp_comm_hook_wrapper(comm_hook, model, state):
    model.register_comm_hook(state, comm_hook)


class DDPCommHookType(Enum):
    """
    DDPCommHookType enumerates the hooks of ``torch.distributed.algorithms.ddp_comm_hooks``
    as names and ``ddp_comm_hook_wrapper`` partials with hook specified. As an example,
    you can register allreduce hook by
    ``DDPCommHookType.ALLREDUCE.value(model=model, state=process_group)``.
    """

    ALLREDUCE = partial(_ddp_comm_hook_wrapper, comm_hook=default.allreduce_hook)
    FP16_COMPRESS = partial(
        _ddp_comm_hook_wrapper, comm_hook=default.fp16_compress_hook
    )
    QUANTIZE_PER_TENSOR = partial(
        _ddp_comm_hook_wrapper, comm_hook=quantization.quantization_pertensor_hook
    )
    QUANTIZE_PER_CHANNEL = partial(
        _ddp_comm_hook_wrapper, comm_hook=quantization.quantization_perchannel_hook
    )


def register_ddp_comm_hook(
    comm_hook_type: DDPCommHookType, model: DistributedDataParallel, state=None
):
    """
        Registers the hooks of ``torch.distributed.algorithms.ddp_comm_hooks``
        to the DDP model. User can specify the type of hook as an enum
        ``DDPCommHookType`` type using ``comm_hook_type`` input. State input will
        be passed to the model.
        Uses Python comm hook implementations.

        Example::
            >>> register_ddp_comm_hook(DDPCommHookType.FP16_COMPRESS, model, state)
    """
    comm_hook_type.value(model=model, state=state)
