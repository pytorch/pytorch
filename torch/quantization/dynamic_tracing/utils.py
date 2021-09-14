import collections
import enum
from typing import Callable, Tuple, Any, Dict

import torch

from .mappings import (
    functions_supported_by_quantization,
    module_types_supported_by_quantization,
    q_mod_to_float_mod_mapping,
)

from torch.quantization import (
    ObserverBase,
    FakeQuantizeBase,
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
        # string
        'idx',
        # Python type of the seen op. For modules, this is type(mod). For
        # functions, this is the target function.
        'type',
        # Note: FQN refers to the current module for modules and to the parent
        # module for functions
        'fqn',
        # Information about the input tensors, List[QTensorInfo].
        # Non-tensor inputs are represented with None.
        'input_tensor_infos',
        # Information about the output tensors, List[QTensorInfo].
        # Non-tensor outputs are represented with None.
        'output_tensor_infos',
    ],
)
def seen_op_repr(self) -> str:
    s = f"(type): {self.type}\n"
    s += f"     (fqn): {self.fqn}\n"
    s += f"     (input_tensor_infos): {self.input_tensor_infos}\n"
    s += f"     (output_tensor_infos): {self.output_tensor_infos}"
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

# TODO: fix lint
class ObserverWrapper(torch.nn.Identity):
    def __init__(self, child):
        super().__init__()
        self.child = child

def wrap_observers_in_placeholders(module: torch.nn.Module) -> None:
    """
    Wraps each child observer of `module` in a placeholder which prevents
    the execution of the observer during the forward. This is useful to prevent
    tracing the model with example inputs from contributing to calibration
    statistics.
    """
    for name, child in module.named_children():
        if isinstance(child, (ObserverBase, FakeQuantizeBase)):
            wrapper = ObserverWrapper(child)
            setattr(module, name, wrapper)
        else:
            wrap_observers_in_placeholders(child)

def unwrap_observers_from_placeholders(module: torch.nn.Module) -> None:
    """
    Restores observers back to their original state.
    """
    for name, child in module.named_children():
        if isinstance(child, ObserverWrapper):
            unwrapped = child.child
            setattr(module, name, unwrapped)
        else:
            unwrap_observers_from_placeholders(child)

def trace_with_inputs(
    model: torch.nn.Module,
    example_args: Tuple[Any],
) -> None:
    with torch.no_grad():
        old_training = model.training
        model.eval()
        wrap_observers_in_placeholders(model)
        model(*example_args)
        unwrap_observers_from_placeholders(model)
        if old_training:
            model.train()

# TODO(future PR): verify correctness of this for all
# quantizeable modules
def is_leaf(m: torch.nn.Module) -> bool:
    return (
        # allowlist everything in torch.nn except nn.Sequential
        (m.__module__.startswith('torch.nn') and (
            not isinstance(m, torch.nn.Sequential)
        )) or
        # allowlist nni modules, as they inherit from nn.Sequential
        m.__module__.startswith('torch.nn.intrinsic')
    )

class FuncOutputObsType(enum.Enum):
    NONE = 0
    NEW_OBS = 1
    REUSES_FIRST_INPUT_OBS = 2

def get_func_output_obs_type(
    op: Callable,
    args: Tuple[Any, ...],
) -> FuncOutputObsType:
    if isinstance(op, torch.nn.Module):
        return FuncOutputObsType.NONE
    if (
        op in (torch.add, torch.Tensor.add, torch.mul, torch.Tensor.mul) and
        len(args) > 1 and
        (not isinstance(args[1], torch.Tensor))
    ):
        return FuncOutputObsType.REUSES_FIRST_INPUT_OBS
    return FuncOutputObsType.NEW_OBS

def converted_func_needs_scale_zp(op: Callable, seen_op: SeenOp) -> bool:
    if isinstance(op, torch.nn.Module):
        return False
    if op in (torch.add, torch.Tensor.add, torch.mul, torch.Tensor.mul):
        # check if both arguments are tensors
        inputs = seen_op.input_tensor_infos
        both_args_tensors = len(inputs) == 2 and inputs[0] is not None and \
            inputs[1] is not None
        return both_args_tensors
    elif op in (torch.cat,):
        return True
    # TODO: add more ops
    # print('op', op)
    return False
