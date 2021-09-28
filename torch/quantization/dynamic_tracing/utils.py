import collections
import enum
from typing import Callable, Tuple, Any, Dict, List, Optional

import torch
import torch.nn.functional as F
toq = torch.ops.quantized

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
        # Information about tensors which will need to be packed,
        # Dict[int, str]
        # idx is the argument index in args
        # name is the name of this parameter in the parent module
        'packable_tensor_idx_to_name',
        # Information about non-tensors which will need to be packed,
        # Dict[int, Any]
        # idx is the argument index in args
        # arg is the argument value
        'packable_nontensor_idx_to_arg',
        # Information about tensors which will need to be packed from kwargs.
        # Dict[str, str]
        # kwarg_name is the kwarg name
        # name is the name of this parameter in the parent module
        'packable_tensor_kwarg_name_to_name',
        # This is True if all packable args are simple attributes, or there
        # are no packable args.
        # This is False if some packable args are results of other functions.
        # bool
        'op_packing_only_uses_module_attributes',
    ],
)
def seen_op_repr(self) -> str:
    s = f"(type): {self.type}\n"
    s += f"     (fqn): {self.fqn}\n"
    s += f"     (input_tensor_infos): {self.input_tensor_infos}\n"
    s += f"     (output_tensor_infos): {self.output_tensor_infos}"
    if len(self.packable_tensor_idx_to_name):
        s += f"\n     (packable_tensor_idx_to_name): {self.packable_tensor_idx_to_name}"
    if len(self.packable_nontensor_idx_to_arg):
        s += f"\n     (packable_nontensor_idx_to_arg): {self.packable_nontensor_idx_to_arg}"
    if len(self.packable_tensor_kwarg_name_to_name):
        s += f"\n     (packable_tensor_kwarg_name_to_name): {self.packable_tensor_kwarg_name_to_name}"
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
    if type(op) in module_types_supported_by_quantization:
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

def pack_weights_for_functionals(
    module: torch.nn.Module,
) -> None:
    """
    Packs weights for functionals seen while tracing.
    Note: weight packing for modules is handled by eager mode quantization
    flow.
    """
    if hasattr(module, '_auto_quant_state'):
        qstate = module._auto_quant_state
        # find any ops which need packing
        for idx, seen_op in qstate.idx_to_seen_ops.items():
            packable_args_len = len(seen_op.packable_tensor_idx_to_name) + \
                len(seen_op.packable_nontensor_idx_to_arg)
            if packable_args_len == 0:
                continue

            if seen_op.type == F.conv2d:
                # fetch all the info needed for packed params
                weight = getattr(module, seen_op.packable_tensor_idx_to_name[1])
                bias = getattr(module, seen_op.packable_tensor_idx_to_name[2])
                stride = seen_op.packable_nontensor_idx_to_arg[3]
                padding = seen_op.packable_nontensor_idx_to_arg[4]
                dilation = seen_op.packable_nontensor_idx_to_arg[5]
                groups = seen_op.packable_nontensor_idx_to_arg[6]

                # quantize the weight
                # TODO: create weight observers from qconfig.weight
                weight_tensor_id = seen_op.input_tensor_infos[1].id
                weight_obs = qstate.tensor_id_to_observer[str(weight_tensor_id)]
                scale, zp = weight_obs.calculate_qparams()
                qweight = torch.quantize_per_tensor(weight, scale, zp, torch.qint8)

                # create the packed params
                packed_params = toq.conv2d_prepack(
                    qweight, bias, stride, padding, dilation, groups)

                # attach to module
                name_idx = 0
                prefix = "_packed_params_"
                name_candidate = f"{prefix}{name_idx}"
                while hasattr(module, name_candidate):
                    name_idx += 1
                    name_candidate = f"{prefix}{name_idx}"
                setattr(module, name_candidate, packed_params)
                qstate.idx_to_packed_weight_name[str(idx)] = name_candidate
                # TODO: delete the original weights

            elif seen_op.type == F.linear:
                # fetch all the info needed for packed params
                weight = getattr(module, seen_op.packable_tensor_idx_to_name[1])
                bias = getattr(module, seen_op.packable_tensor_kwarg_name_to_name['bias'])

                # quantize the weight
                # TODO: create weight observers from qconfig.weight
                weight_tensor_id = seen_op.input_tensor_infos[1].id
                weight_obs = qstate.tensor_id_to_observer[str(weight_tensor_id)]
                scale, zp = weight_obs.calculate_qparams()
                qweight = torch.quantize_per_tensor(weight, scale, zp, torch.qint8)

                # create the packed params
                packed_params = toq.linear_prepack(qweight, bias)

                # attach to module
                name_idx = 0
                prefix = "_packed_params_"
                name_candidate = f"{prefix}{name_idx}"
                while hasattr(module, name_candidate):
                    name_idx += 1
                    name_candidate = f"{prefix}{name_idx}"
                setattr(module, name_candidate, packed_params)
                qstate.idx_to_packed_weight_name[str(idx)] = name_candidate
                # TODO: delete the original weights

    for _, child in module.named_children():
        pack_weights_for_functionals(child)

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
    op_packing_only_uses_module_attributes: bool,
) -> FuncOutputObsType:
    if isinstance(op, torch.nn.Module):
        return FuncOutputObsType.NONE

    # check for ops which need packed weights but the weights are
    # coming from another function
    if not op_packing_only_uses_module_attributes:
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
    elif op in (torch.relu, F.relu):
        return FuncOutputObsType.NONE
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
    elif op in (F.conv2d, F.linear):
        outputs = seen_op.output_tensor_infos
        is_int8 = outputs[0].inf_dtype == torch.quint8
        return is_int8
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
    # for ops which may be quantizeable in some cases but are not
    # quantizeable due to observed syntax (for example, F.conv2d with
    # weights coming from another function).
    DTYPE_DEFAULT_BC_UNSUPPORTED_SYNTAX = 2

def get_func_output_dtype_type(
    op: Callable,
    args: Tuple[Any, ...],
    op_packing_only_uses_module_attributes: bool,
) -> FuncOutputDTypeType:
    if isinstance(op, torch.nn.Module):
        if type(op) in module_types_supported_by_quantization_preserves_dtype:
            return FuncOutputDTypeType.DTYPE_EQUALS_INPUT_DTYPE

    # check for ops which need packed weights but the weights are
    # coming from another function
    if not op_packing_only_uses_module_attributes:
        return FuncOutputDTypeType.DTYPE_DEFAULT_BC_UNSUPPORTED_SYNTAX

    if op in functions_supported_by_quantization_preserves_dtype:
        return FuncOutputDTypeType.DTYPE_EQUALS_INPUT_DTYPE
    elif op in add_and_mul_ops and len(args) > 0 and \
            args[0].dtype in (torch.int32, torch.int64):
        # binary ops with torch.int arguments do not support quantization
        return FuncOutputDTypeType.DTYPE_EQUALS_INPUT_DTYPE
    elif op == torch.cat and len(args) > 0 and \
            args[0][0].dtype in (torch.int32, torch.int64):
        return FuncOutputDTypeType.DTYPE_EQUALS_INPUT_DTYPE

    return FuncOutputDTypeType.DTYPE_DEPENDS_ON_QCONFIG

def get_op_packing_only_uses_module_attributes(
    op: Callable,
    args: Tuple[Any, ...],
    module: torch.nn.Module,
) -> bool:
    # check for ops which need packed weights but the weights are
    # coming from another function
    packable_tensor_arg_idxs = get_packable_tensor_arg_idxs(op)
    if packable_tensor_arg_idxs is not None:
        for arg_idx in packable_tensor_arg_idxs:
            arg_name_in_root = get_param_name(module, args[arg_idx])
            if arg_name_in_root is None:
                return False
    return True

def get_quantized_op(
    op: Callable,
    seen_op: SeenOp,
) -> Callable:
    if seen_op.output_tensor_infos[0].inf_dtype != torch.quint8:
        return op

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

def get_input_observed_arg_idxs(
    op: Callable,
) -> Optional[List[int]]:
    if isinstance(op, torch.nn.Module):
        # TODO(future PR): handle RNNs
        return [0]
    if op == F.conv2d:
        return [0, 1]
    elif op == F.linear:
        return [0, 1]
    # None means "observe all Tensor args"
    return None

def get_packable_tensor_arg_idxs(op: Callable) -> Optional[List[int]]:
    """
    Returns tensor arg idxs which correspond to parameters which will need
    to be packed.
    """
    if op == F.conv2d:
        return [1, 2]
    elif op == F.linear:
        return [1]
    return None

def get_packable_tensor_kwarg_names(op: Callable) -> Optional[List[str]]:
    """
    Returns tensor kwarg names which correspond to parameters which will
    need to be packed.
    """
    if op == F.linear:
        return ['bias']
    return None

def get_param_name(module: torch.nn.Module, arg: Any) -> Optional[str]:
    """
    Returns the name of arg with respect to the current module.
    """
    for name, param in module.named_parameters():
        if arg is param:
            return name
    return None
    # raise AssertionError(f"arg {arg} not found in module {module}")

def get_packable_nontensor_arg_idxs(op: Callable) -> Optional[List[int]]:
    """
    Returns nontensor arg idxs which correspond to arguments which will need
    to be packed.
    """
    if op == F.conv2d:
        # stride, padding, dilation, groups
        return [3, 4, 5, 6]
    return None

def get_packable_arg_idxs(op: Callable) -> Optional[List[int]]:
    if op == F.conv2d:
        # weight, bias, stride, padding, dilation, groups
        return [1, 2, 3, 4, 5, 6]
    elif op == F.linear:
        # weight
        return [1]
    return None

def get_weight_arg_idx(op: Callable) -> Optional[int]:
    if op == F.conv2d:
        return 1
    elif op == F.linear:
        return 1
    return None

def iterate_and_apply(
    args: Any,
    flattened_tensor_infos: List[Optional[QTensorInfo]],
    func: Callable,
    flattened_tensor_infos_idx = None
) -> Any:
    """
    Inputs:
      `args`: arguments to a function, may contain nested types, for example:

        ([torch.Tensor, torch.Tensor], int, (int, int))

      `flattened_tensor_infos`: tensor information containers for each tensor
        in `args`, flattened, for example corresponding with above:

        ({...}, {...}, None, None, None)

      `func`: function to apply to each tensor in `args` to create `new_args`

    Returns `new_args`, where each tensor has been transformed by `func`.
    """
    arg_idx = 0
    if flattened_tensor_infos_idx is None:
        flattened_tensor_infos_idx = [0]

    if isinstance(args, tuple):
        new_args = []
        for arg in args:
            new_arg = iterate_and_apply(
                arg, flattened_tensor_infos, func, flattened_tensor_infos_idx)
            new_args.append(new_arg)
        return tuple(new_args)
    elif isinstance(args, list):
        for idx in range(len(args)):
            new_arg = iterate_and_apply(
                args[idx], flattened_tensor_infos, func, flattened_tensor_infos_idx)
            args[idx] = new_arg
        return args
    else:
        # individual element
        cur_flattened_tensor_info = \
            flattened_tensor_infos[flattened_tensor_infos_idx[0]]
        flattened_tensor_infos_idx[0] += 1

        if cur_flattened_tensor_info is not None:
            return func(args, cur_flattened_tensor_info)
        else:
            return args
