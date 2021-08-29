import collections
from typing import Callable

import torch

from .mappings import (
    functions_supported_by_quantization,
    module_types_supported_by_quantization,
    q_mod_to_float_mod_mapping,
)

def _raise_obs_not_found_error(func):
    raise RuntimeError(
        f'Encountered arithmetic operation {torch.typename(func)} but we have '
        f'encountered fewer arithmetic operations in previous calibration runs. '
        f'This likely indicates that the program contains dynamic control flow. '
        f' Quantization is not defined over dynamic control flow!')

def _raise_obs_op_mismatch(func, prev_op):
    raise RuntimeError(
        f'Encountered arithmetic operation {torch.typename(func)} but previously '
        f'recorded operation was {torch.typename(prev_op)}!. This likely indicates '
        f'that the program contains dynamic control flow. Quantization is not '
        f'defined over dynamic control flow!')


# TODO(future PR): figure out if there is a better option than namedtuple
SeenOp = collections.namedtuple(
    'SeenOp',
    [
        'idx',
        'type',
        # Note: FQN refers to the current module for modules and to the parent
        # module for functions
        'fqn',
        'input_tensor_ids',
        'output_tensor_ids',
    ],
)
def seen_op_repr(self) -> str:
    s = f"(type): {self.type}\n"
    s += f"     (fqn): {self.fqn}\n"
    s += f"     (input_tensor_ids): {self.input_tensor_ids}\n"
    s += f"     (output_tensor_ids): {self.output_tensor_ids}"
    return s

SeenOp.__repr__ = seen_op_repr  # type: ignore[assignment]

QTensorInfo = collections.namedtuple(
    'QTensorInfo',
    [
        'id',  # tensor ID
        'inf_dtype',  # dtype at inference
    ],
)

def op_needs_quantization(op: Callable) -> bool:
    if op in functions_supported_by_quantization:
        return True
    for module_type in module_types_supported_by_quantization:
        if isinstance(op, module_type):
            return True
    if op in q_mod_to_float_mod_mapping:
        return True
    return False
