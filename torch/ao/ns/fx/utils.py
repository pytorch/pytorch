import enum
import operator

import torch
import torch.nn as nn
import torch.nn.intrinsic.quantized as nniq
import torch.ao.nn.quantized as nnq

toq = torch.ops.quantized
from typing import Tuple, Callable, Dict, Set, List, Optional, Union

from torch.fx import GraphModule
from torch.fx.graph import Node
from torch.ao.quantization import (
    ObserverBase,
    FakeQuantizeBase
)
from torch.ao.quantization.observer import _is_activation_post_process
from torch.ao.quantization.utils import _getattr_from_fqn

from .ns_types import NSNodeTargetType, NSResultsType

# TODO(future PR): consider deleting this enum and using the torch types
# directly.  This might be tricky because it is not a one to one mapping.
class NodeInputOrOutputType(enum.Enum):
    FP32 = enum.auto()  # torch.float
    INT8 = enum.auto()  # torch.qint8 or torch.quint8
    FP16 = enum.auto()  # torch.float16
    UNKNOWN = enum.auto()  # we cannot determine input/output dtype
    # TODO(future PR): while these functions can support multiple dtypes,
    #   for the purposes of numerical debugging we want to get the actual
    #   dtype used in the model. We will likely need some kind of dtype
    #   propagation to estimate this.
    FP32_OR_INT8 = enum.auto()  # either torch.float or torch.quint8 or torch.qint8
    # TODO(future PRs): dynamic quant, fake quant, etc


def get_node_first_input_and_output_type(
    node: Node,
    gm: GraphModule,
    logger_cls: Callable,
    node_type_to_io_type_map: Dict[str, Set[NSNodeTargetType]],
) -> Tuple[NodeInputOrOutputType, NodeInputOrOutputType]:

    # TODO(future PR): clean this up
    FUNS_IO_TYPE_FP32 = node_type_to_io_type_map["funs_io_type_fp32"]
    FUNS_IO_TYPE_FP16 = node_type_to_io_type_map["funs_io_type_fp16"]
    FUNS_IO_TYPE_INT8 = node_type_to_io_type_map["funs_io_type_int8"]
    FUNS_IO_TYPE_FP32_OR_INT8 = node_type_to_io_type_map["funs_io_type_fp32_or_int8"]
    MODS_IO_TYPE_FP32 = node_type_to_io_type_map["mods_io_type_fp32"]
    MODS_IO_TYPE_INT8 = node_type_to_io_type_map["mods_io_type_int8"]
    MODS_IO_TYPE_FP32_OR_INT8 = node_type_to_io_type_map["mods_io_type_fp32_or_int8"]
    METHS_IO_TYPE_FP32_OR_INT8 = node_type_to_io_type_map["meths_io_type_fp32_or_int8"]

    if node.op == "call_function":
        if node.target in FUNS_IO_TYPE_FP32:
            return (NodeInputOrOutputType.FP32, NodeInputOrOutputType.FP32)
        if node.target in FUNS_IO_TYPE_FP16:
            return (NodeInputOrOutputType.FP16, NodeInputOrOutputType.FP16)
        elif node.target in FUNS_IO_TYPE_INT8:
            return (NodeInputOrOutputType.INT8, NodeInputOrOutputType.INT8)
        elif node.target in FUNS_IO_TYPE_FP32_OR_INT8:
            first_arg = get_normalized_nth_input(node, gm, 0)
            assert isinstance(first_arg, Node)
            (
                _prev_node_input_type,
                prev_node_output_type,
            ) = get_node_first_input_and_output_type(
                first_arg, gm, logger_cls, node_type_to_io_type_map
            )
            return (prev_node_output_type, prev_node_output_type)
        else:
            return (NodeInputOrOutputType.UNKNOWN, NodeInputOrOutputType.UNKNOWN)

    elif node.op == "call_module":
        assert node.op == "call_module"
        assert isinstance(node.target, str)
        mod = _getattr_from_fqn(gm, node.target)
        is_known_fp32_or_int8_input_module = any(
            isinstance(mod, target_type) for target_type in MODS_IO_TYPE_FP32_OR_INT8  # type: ignore[arg-type]
        )
        if (
            isinstance(mod, (logger_cls, ObserverBase, FakeQuantizeBase))  # type: ignore[arg-type]
            or is_known_fp32_or_int8_input_module
        ):
            # A logger or observer's input and output type is the output
            # type of the preceding node.
            first_arg = get_normalized_nth_input(node, gm, 0)
            assert isinstance(first_arg, Node)
            (
                _prev_node_input_type,
                prev_node_output_type,
            ) = get_node_first_input_and_output_type(
                first_arg, gm, logger_cls, node_type_to_io_type_map
            )
            return (prev_node_output_type, prev_node_output_type)
        is_known_fp32_input_module = any(
            isinstance(mod, target_type) for target_type in MODS_IO_TYPE_FP32  # type: ignore[arg-type]
        )
        is_known_int8_input_module = any(
            isinstance(mod, target_type) for target_type in MODS_IO_TYPE_INT8  # type: ignore[arg-type]
        )
        if is_known_fp32_input_module:
            return (NodeInputOrOutputType.FP32, NodeInputOrOutputType.FP32)
        elif is_known_int8_input_module:
            return (NodeInputOrOutputType.INT8, NodeInputOrOutputType.INT8)
        else:
            return (NodeInputOrOutputType.UNKNOWN, NodeInputOrOutputType.UNKNOWN)

    elif node.op == "call_method":
        if node.target == "dequantize":
            # Dequantize is a special node because it allows multiple input types.
            # So, we look up the output type of the previous node and return that
            # as the input type of this node instance.
            prev_node = get_normalized_nth_input(node, gm, 0)
            assert isinstance(prev_node, Node)
            (
                _prev_node_input_type,
                prev_node_output_type,
            ) = get_node_first_input_and_output_type(
                prev_node, gm, logger_cls, node_type_to_io_type_map
            )
            return (prev_node_output_type, NodeInputOrOutputType.FP32)

        elif node.target == "to":
            # to is a special node because it allows multiple input types.
            # So, we look up the output type of the previous node and return that
            # as the input type of this node instance. We also look up the target
            # of to and return the correct output type.
            prev_node = get_normalized_nth_input(node, gm, 0)
            assert isinstance(prev_node, Node)
            (
                _prev_node_input_type,
                prev_node_output_type,
            ) = get_node_first_input_and_output_type(
                prev_node, gm, logger_cls, node_type_to_io_type_map
            )

            cur_node_dtype_target = get_normalized_nth_input(node, gm, 1)
            assert (
                cur_node_dtype_target is torch.float16
            ), f"{cur_node_dtype_target} handling needs to be added"

            return (prev_node_output_type, NodeInputOrOutputType.FP16)

        elif node.target in METHS_IO_TYPE_FP32_OR_INT8:
            first_arg = get_normalized_nth_input(node, gm, 0)
            assert isinstance(first_arg, Node)
            (
                _prev_node_input_type,
                prev_node_output_type,
            ) = get_node_first_input_and_output_type(
                first_arg, gm, logger_cls, node_type_to_io_type_map
            )
            return (prev_node_output_type, prev_node_output_type)

        return (NodeInputOrOutputType.UNKNOWN, NodeInputOrOutputType.UNKNOWN)
    else:
        return (NodeInputOrOutputType.UNKNOWN, NodeInputOrOutputType.UNKNOWN)


def get_node_input_qparams(
    node: Node,
    gm: GraphModule,
    node_type_to_io_type_map: Dict[str, Set[NSNodeTargetType]],
) -> Optional[Tuple[Union[torch.Tensor, float], Union[torch.Tensor, int]]]:
    """
    Returns the qparams (scale, zero_point) of the first input to `node`,
    if they can be inferred from the graph.
    """
    prev_node = get_normalized_nth_input(node, gm, 0)

    if not isinstance(prev_node, Node):
        return None

    MODS_IO_TYPE_FP32_OR_INT8 = node_type_to_io_type_map["mods_io_type_fp32_or_int8"]

    def _get_scale_zp_from_function_args(node, gm, scale_arg_idx, zp_arg_idx):
        scale_node = get_normalized_nth_input(node, gm, scale_arg_idx)
        zp_node = get_normalized_nth_input(node, gm, zp_arg_idx)
        assert isinstance(scale_node, Node) and isinstance(scale_node.target, str)
        assert isinstance(zp_node, Node) and isinstance(zp_node.target, str)
        scale_obj = _getattr_from_fqn(gm, scale_node.target)
        zp_obj = _getattr_from_fqn(gm, zp_node.target)
        return (scale_obj, zp_obj)

    if prev_node.op == "call_function":

        # quantize - read the args directly
        if prev_node.target == torch.quantize_per_tensor:
            return _get_scale_zp_from_function_args(prev_node, gm, 1, 2)
        elif prev_node.target in (toq.add, toq.add_relu, toq.mul, toq.mul_relu):
            return _get_scale_zp_from_function_args(prev_node, gm, 2, 3)

        return None
        # TODO(future PR): handle more functionals
        # TODO(future PR): handle functional ops which inherit qparams from input

    elif prev_node.op == "call_module":

        # get type of the module
        assert isinstance(prev_node.target, str)
        module_obj = _getattr_from_fqn(gm, prev_node.target)
        if isinstance(
            module_obj,
            (
                nnq.Linear,
                nnq.Conv1d,
                nnq.Conv2d,
                nniq.ConvReLU2d,
                nnq.Conv3d,
                nnq.BatchNorm2d,
                nnq.BatchNorm3d,
                nnq.ConvTranspose1d,
                nnq.ConvTranspose2d,
                nnq.ELU,
                nnq.GroupNorm,
                nnq.InstanceNorm1d,
                nnq.InstanceNorm2d,
                nnq.InstanceNorm3d,
                nnq.LayerNorm,
                nnq.Hardswish,
                nnq.LeakyReLU,
                nnq.ReLU6,
                nniq.BNReLU2d,
                nniq.BNReLU3d,
                nniq.ConvReLU1d,
                nniq.ConvReLU2d,
                nniq.ConvReLU3d,
                nniq.LinearReLU,
            ),
        ):
            return (module_obj.scale, module_obj.zero_point)  # type: ignore[return-value]

        is_known_fp32_or_int8_input_module = any(
            isinstance(module_obj, target_type) for target_type in MODS_IO_TYPE_FP32_OR_INT8  # type: ignore[arg-type]
        )
        if is_known_fp32_or_int8_input_module:
            return get_node_input_qparams(prev_node, gm, node_type_to_io_type_map)

    return None


def return_first_non_observer_node(
    node: Node,
    gm: GraphModule,
) -> Node:
    """
    If node is not an observer, returns it.  If node is an observer,
    navigates up the graph and returns the first parent which is not an
    observer.  For example,

    graph: (node_non_obs), node = node_non_obs : returns node_non_obs
    graph: (node_non_obs -> obs0), node = obs0 : returns node_non_obs
    graph: (node_non_obs -> obs0 -> fq0), node = fq0 : returns node_non_obs
    """
    if node.op == "call_module":
        node_obj = _getattr_from_fqn(gm, node.target)  # type: ignore[arg-type]
        if _is_activation_post_process(node_obj):
            assert len(node.args) == 1
            assert isinstance(node.args[0], Node)
            node = node.args[0]
            # code duplication intended, not worth refactoring
            assert isinstance(node.target, str)
            node_obj = _getattr_from_fqn(gm, node.target)
            if _is_activation_post_process(node_obj):
                assert len(node.args) == 1
                assert isinstance(node.args[0], Node)
                node = node.args[0]
    return node


def get_number_of_non_param_args(
    node: Node,
    gm: GraphModule,
) -> int:
    """
    Assumes that all non-param args occur first. Returns the number of
    non-param args expected for a node.  For example, for

      F.linear(x, weight, bias)

    Returns 1, because x is a non-param arg and weight and bias are params.
    For

      lstm_mod(x, hid)

    Returns 2, because both x and hid are non-param args.
    """
    if node.op == "call_module":
        node_obj = _getattr_from_fqn(gm, node.target)  # type: ignore[arg-type]
        if isinstance(node_obj, nn.LSTM):
            return 2

    # default is 1
    return 1


def get_arg_indices_of_inputs_to_log(node: Node) -> List[int]:
    """
    Returns the indices of args of the node which we should attach
    loggers to, if input logging is enabled.

    For example,
    * for (x + y), returns [0, 1]
    * for (1 + y), returns [1]
    * for (x + 1), returns [0]
    * for (linear(x, w, b)) returns [0]
    * by default, returns [0]
    """
    if len(node.args) == 0:
        return []
    if node.op == "call_function" and (
        # TODO(future PR): use relationship map instead of hardcoding
        node.target in (torch.add, torch.ops.quantized.add, operator.add)
        or node.target in (torch.mul, torch.ops.quantized.mul, operator.mul)
    ):
        result = []
        for i in range(2):
            if type(node.args[i]) == Node:
                result.append(i)
        return result
    return [0]


def get_target_type_str(node: Node, gm: GraphModule) -> str:
    """
    Returns a string representation of the type of the function or module
    pointed to by this node, or '' for other node types.
    """
    target_type = ""
    if node.op in ("call_function", "call_method"):
        target_type = torch.typename(node.target)
    elif node.op == "call_module":
        assert isinstance(node.target, str)
        target_mod = _getattr_from_fqn(gm, node.target)
        target_type = torch.typename(target_mod)
    return target_type


def rekey_logger_info_on_node_name_of_model(
    results: NSResultsType,
    model_name: str,
) -> NSResultsType:
    """
    Rekeys the layer name of a results dictionary to use node names
    from `model_name`.

    For example, transforms

        {'base_op_1_0': {'node_output': {'model_a':
          [{'ref_node_name': 'linear1', ...}]}}}

    into

        {'linear1': {'node_output': {'model_a':
          [{'ref_node_name': 'linear1', ...}]}}}

    Note: we cannot use these node names directly because they are not
    guaranteed to be consistent across models. This is why we extract
    the results first and rekey afterwards.
    """
    new_results = {}
    for old_layer_name, result_type_to_results in results.items():
        new_layer_name = None
        for _result_type, model_name_to_results in result_type_to_results.items():
            for cur_model_name, list_of_results in model_name_to_results.items():
                if cur_model_name == model_name:
                    assert len(list_of_results)
                    new_layer_name = list_of_results[0]["ref_node_name"]
                else:
                    continue
        if new_layer_name is not None:
            new_results[new_layer_name] = result_type_to_results
        else:
            new_results[old_layer_name] = result_type_to_results
    return new_results


def maybe_add_missing_fqns(results: NSResultsType) -> None:
    """
    If `fqn` entries are filled in for one of the models in `results`, copies
    them over to any models which do not have them filled out.

    A common use case benefitting from this is comparing a model prepared by
    quantization to a quantized model. In this case, the model prepared by
    quantization would have `fqn` entries, and the quantized model would not.
    """

    # Check in the first result to find any model with fqn entries defined.
    model_name_with_fqns = None
    for layer_name, result_type_to_results in results.items():
        for result_type, model_name_to_results in result_type_to_results.items():
            for model_name, model_results in model_name_to_results.items():
                if len(model_results) > 0:
                    if model_results[0]["fqn"] is not None:
                        model_name_with_fqns = model_name
                        break
            break
        break

    if model_name_with_fqns:
        for layer_name, result_type_to_results in results.items():
            for result_type, model_name_to_results in result_type_to_results.items():
                ref_model_results = model_name_to_results[model_name_with_fqns]
                for model_name, model_results in model_name_to_results.items():
                    if model_name == model_name_with_fqns:
                        continue
                    for i in range(len(model_results)):
                        fqn = ref_model_results[i]["fqn"]
                        model_results[i]["fqn"] = fqn


def maybe_dequantize_first_two_tensor_args_and_handle_tuples(f):
    def inner(*args, **kwargs):
        a0, a1, *a_other = args

        if (isinstance(a0, tuple) and isinstance(a1, tuple)) or (
            isinstance(a0, list) and isinstance(a1, list)
        ):
            results = []
            for el0, el1 in zip(a0, a1):
                new_args = (el0, el1, *a_other)
                results.append(inner(*new_args, **kwargs))
            return results

        elif isinstance(a0, torch.Tensor) and isinstance(a1, torch.Tensor):
            if a0.is_quantized:
                a0 = a0.dequantize()
            if a1.is_quantized:
                a1 = a1.dequantize()

        # for the purposes of this util, only handle floats
        if a0.dtype != torch.float or a1.dtype != torch.float:
            return None

        new_args = (a0, a1, *a_other)
        return f(*new_args, **kwargs)

    return inner


@maybe_dequantize_first_two_tensor_args_and_handle_tuples
def compute_sqnr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Computes the SQNR between `x` and `y`.

    Args:
        x: Tensor or tuple of tensors
        y: Tensor or tuple of tensors

    Return:
        float or tuple of floats
    """
    Ps = torch.norm(x)
    Pn = torch.norm(x - y)
    return 20 * torch.log10(Ps / Pn)


@maybe_dequantize_first_two_tensor_args_and_handle_tuples
def compute_normalized_l2_error(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Computes the normalized L2 error between `x` and `y`.

    Args:
        x: Tensor or tuple of tensors
        y: Tensor or tuple of tensors

    Return:
        float or tuple of floats
    """
    return torch.sqrt(((x - y) ** 2).sum() / (x ** 2).sum())


@maybe_dequantize_first_two_tensor_args_and_handle_tuples
def compute_cosine_similarity(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Computes the cosine similarity between `x` and `y`.

    Args:
        x: Tensor or tuple of tensors
        y: Tensor or tuple of tensors

    Return:
        float or tuple of floats
    """
    # For convolutions, the shape of the quantized weight has one additional
    # dimension compared to the shape of the fp32 weight. Match the shapes
    # to enable cosine similarity comparison.
    x = x.reshape(1, -1)
    y = y.reshape(1, -1)
    return torch.nn.functional.cosine_similarity(x, y)

def op_type_supports_shadowing(node: Node) -> bool:
    if node.op == 'call_function':
        if node.target in (torch.add, torch.mul, operator.add, operator.mul, torch.cat, torch.stack):
            # shadowing for ops with multiple tensor inputs is not implemented yet
            return False
    return True

def get_normalized_nth_input(node: Node, gm: GraphModule, idx: int) -> Node:
    """
    Given a node, gets the n'th input to that node, normalizing
    args and kwargs to the best of its ability.
    """
    try:
        norm_args_and_kwargs = node.normalized_arguments(
            gm, normalize_to_only_use_kwargs=True)
        if norm_args_and_kwargs is not None:
            norm_args, norm_kwargs = norm_args_and_kwargs
            assert len(norm_args) + len(norm_kwargs) > idx
            if idx < len(norm_args):
                return norm_args[idx]
            else:
                # note: in Python 3.7+ dicts are ordered
                return list(norm_kwargs.values())[idx]
        else:
            assert len(node.args) + len(node.kwargs) > idx
            if idx < len(node.args):
                return node.args[idx]  # type: ignore[return-value]
            else:
                kwargs_idx = idx + len(node.args)
                return list(node.kwargs.values())[kwargs_idx]  # type: ignore[return-value]
    except RuntimeError:
        # this RuntimeError happens when node argument normalization
        # requires typehints to proceed, such as for torch.add where
        # either the first, second or both arguments could be tensors
        assert len(node.args) + len(node.kwargs) > idx
        if idx < len(node.args):
            return node.args[idx]  # type: ignore[return-value]
        else:
            kwargs_idx = idx + len(node.args)
            return list(node.kwargs.values())[kwargs_idx]  # type: ignore[return-value]
