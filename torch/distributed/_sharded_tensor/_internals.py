from dataclasses import dataclass
from enum import Enum

import torch

class CreateOp(Enum):
    EMPTY = 0
    ONES = 1

@dataclass
class InitCommonParams(object):
    """ Container for list of common params to create new local tensor. """

    __slots__ = ['create_op', 'dtype', 'layout', 'requires_grad', 'pin_memory',
                 'memory_format']

    create_op: CreateOp
    dtype: torch.dtype
    layout: torch.layout
    requires_grad: bool
    pin_memory: bool
    memory_format: torch.memory_format

def create_tensor_from_params(*size, local_device, params: InitCommonParams):
    """ Helper to construct tensor from size, device and common params. """

    if params.create_op == CreateOp.ONES:
        return torch.ones(*size,
                          dtype=params.dtype,
                          layout=params.layout,
                          device=local_device,
                          pin_memory=params.pin_memory,
                          requires_grad=params.requires_grad,)
    elif params.create_op == CreateOp.EMPTY:
        return torch.empty(*size,
                           dtype=params.dtype,
                           layout=params.layout,
                           device=local_device,
                           requires_grad=params.requires_grad,
                           # Note memory_format param is not accepted by torch.ones
                           memory_format=params.memory_format,
                           pin_memory=params.pin_memory,)
    else:
        raise ValueError(f'Unsupported create_op: {params.create_op}')
