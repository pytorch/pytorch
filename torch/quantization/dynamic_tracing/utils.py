import collections
import enum
from typing import Callable, Tuple, Any, Dict

import torch

from .mappings import (
    functions_supported_by_quantization,
    module_types_supported_by_quantization,
    q_mod_to_float_mod_mapping,
    module_types_supported_by_quantization_preserves_dtype,
    functions_supported_by_quantization_preserves_dtype,
    fp32_to_int8_fun_mapping,
    add_and_mul_ops,
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
    # Note: we cannot use module.named_children() because we can
    # have two different names refer to the same module, for example
    # when we are reusing observers for torch.add scalar version.
    for name, child in module._modules.items():
        if child is None:
            continue
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
    if op in add_and_mul_ops:
        if len(args) > 0 and args[0].dtype in (torch.int32, torch.int64):
            # this is handling ops on dtypes such as torch.int
            return FuncOutputObsType.NONE
        elif (
            len(args) > 1 and
            (not isinstance(args[1], torch.Tensor))
        ):
            return FuncOutputObsType.REUSES_FIRST_INPUT_OBS
    elif op == torch.cat:
        if len(args[0]) > 0 and args[0][0].dtype in (torch.int32, torch.int64):
            return FuncOutputObsType.NONE
    return FuncOutputObsType.NEW_OBS

def converted_func_needs_scale_zp(op: Callable, seen_op: SeenOp) -> bool:
    if isinstance(op, torch.nn.Module):
        return False
    if op in add_and_mul_ops:
        # check if both arguments are tensors
        inputs = seen_op.input_tensor_infos
        both_args_tensors = len(inputs) == 2 and inputs[0] is not None and \
            inputs[1] is not None
        # disable quantization for torch.mul with int tensor arguments
        first_dtype_is_not_int = len(inputs) > 0 and \
            inputs[0].inf_dtype not in (torch.int32, torch.int64)
        return both_args_tensors and first_dtype_is_not_int
    elif op == torch.cat:
        inputs = seen_op.input_tensor_infos
        first_dtype_is_not_int = len(inputs) > 0 and \
            inputs[0].inf_dtype not in (torch.int32, torch.int64)
        return first_dtype_is_not_int
    # TODO: add more ops
    # print('op', op)
    return False

class FuncOutputDTypeType(enum.Enum):
    # for ops which are quantizeable and are configured by the qconfig,
    # for example F.conv2d
    DTYPE_DEPENDS_ON_QCONFIG = 0
    # for ops which are quantizeable and take the dtype of the previous
    # op, for example nn.Dropout
    DTYPE_EQUALS_INPUT_DTYPE = 1

def get_func_output_dtype_type(
    op: Callable,
    args: Tuple[Any, ...],
) -> FuncOutputDTypeType:
    if isinstance(op, torch.nn.Module):
        for target_mod_cls in module_types_supported_by_quantization_preserves_dtype:
            if isinstance(op, target_mod_cls):
                return FuncOutputDTypeType.DTYPE_EQUALS_INPUT_DTYPE
    elif op in functions_supported_by_quantization_preserves_dtype:
        return FuncOutputDTypeType.DTYPE_EQUALS_INPUT_DTYPE
    elif op in add_and_mul_ops and len(args) > 0 and \
            args[0].dtype in (torch.int32, torch.int64):
        # binary ops with torch.int arguments do not support quantization
        return FuncOutputDTypeType.DTYPE_EQUALS_INPUT_DTYPE
    elif op == torch.cat and len(args) > 0 and \
            args[0][0].dtype in (torch.int32, torch.int64):
        return FuncOutputDTypeType.DTYPE_EQUALS_INPUT_DTYPE

    return FuncOutputDTypeType.DTYPE_DEPENDS_ON_QCONFIG

def get_quantized_op(
    op: Callable,
    seen_op: SeenOp,
) -> Callable:
    new_op = op
    if not isinstance(op, torch.nn.Module):
        if (
            (op in add_and_mul_ops or op == torch.cat) and
            seen_op.input_tensor_infos[0].inf_dtype in (torch.int32, torch.int64)
        ):
            # handle torch.mul with int tensor arguments
            pass
        elif op in fp32_to_int8_fun_mapping:
            new_op = fp32_to_int8_fun_mapping[op]

    return new_op
