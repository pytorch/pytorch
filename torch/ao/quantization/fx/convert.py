from typing import Any, Dict, List, Optional, Set, Tuple, Union, Type, Callable
from torch.ao.quantization.quant_type import QuantType
import torch
import copy
import warnings
from torch.fx import (
    GraphModule,
)
from torch.fx.graph import (
    Graph,
    Node,
    Argument,
)
from ..utils import (
    activation_is_statically_quantized,
    weight_is_quantized,
    get_qparam_dict,
    _parent_name,
    get_swapped_custom_module_class,
)
from ..qconfig import (
    QConfigAny,
    qconfig_equals
)
from ..qconfig_mapping import QConfigMapping
from .qconfig_mapping_utils import (
    _generate_node_name_to_qconfig,
    _compare_prepare_convert_qconfig_mappings,
    _update_qconfig_for_fusion,
    _is_qconfig_supported_by_dtype_configs,
    _update_qconfig_for_qat,
)
from torch.ao.quantization.backend_config.utils import (
    get_root_module_to_quantized_reference_module,
    get_pattern_to_dtype_configs,
    get_fused_module_classes,
    get_qat_module_classes,
)
from torch.ao.quantization.backend_config import (
    BackendConfig,
    get_native_backend_config,
)
from torch.ao.quantization.observer import _is_activation_post_process
from .graph_module import (
    _is_observed_module,
    _is_observed_standalone_module,
)
from ._equalize import update_obs_for_equalization, convert_eq_obs
from torch.nn.utils.parametrize import type_before_parametrizations
from .utils import (
    _get_module,
    _is_custom_module_lstm,
    _is_custom_module_mha,
    get_custom_module_class_keys,
    create_getattr_from_value,
    collect_producer_nodes,
    graph_module_from_producer_nodes,
    node_arg_is_weight,
)
from torch.ao.quantization.utils import (
    is_per_channel,
    to_underlying_dtype,
)
from torch.ao.quantization.quantize import (
    _remove_qconfig,
)
from torch.ao.quantization.stubs import DeQuantStub
from .custom_config import (
    ConvertCustomConfig,
    PrepareCustomConfig,
)
from .lower_to_fbgemm import lower_to_fbgemm
# importing the lib so that the quantized_decomposed ops are registered
from ._decomposed import quantized_decomposed_lib  # noqa: F401
import operator

__all__ = [
    "convert",
    "convert_custom_module",
    "convert_standalone_module",
    "convert_weighted_module",
]

_QSCHEME_TO_CHOOSE_QPARAMS_OP = {
    torch.per_tensor_affine: torch.ops.quantized_decomposed.choose_qparams.tensor,
    torch.per_tensor_symmetric: torch.ops.quantized_decomposed.choose_qparams_symmetric.tensor,
}

def _replace_observer_with_quantize_dequantize_node_decomposed(
        model: torch.nn.Module,
        graph: Graph,
        node: Node,
        modules: Dict[str, torch.nn.Module],
        node_name_to_scope: Dict[str, Tuple[str, type]],
        node_name_to_qconfig: Dict[str, QConfigAny]) -> None:
    """ Replace activation_post_process module call node with quantize and
    dequantize node working with decomposed Tensor

    Before:
    ... -> observer_0(x) -> ...
    After:
    ... -> torch.ops.quantized_decomposed.quantize_per_tensor(x, ...) ->
    torch.ops.quantized_decomposed.dequantize_per_tensor() -> ...

    or quantize_per_channel and dequantize_per_channel
    """
    assert modules is not None
    assert isinstance(node.target, str)
    module_path, prefix = _get_module_path_and_prefix(node, node_name_to_scope, node_name_to_qconfig)
    activation_post_process = modules[node.target]
    # skip replacing observers to quant/dequant nodes if the qconfigs of all
    # consumers and producers of this observer are None
    skip_replacement = all(_has_none_qconfig(n, node_name_to_qconfig) for n in
                           list(node.args) + list(node.users.keys()))
    if skip_replacement or not _is_conversion_supported(activation_post_process):
        # didn't find corresponding quantize op and info for the activation_post_process
        # so we just remove the observer
        with graph.inserting_before(node):
            node.replace_all_uses_with(node.args[0])
            graph.erase_node(node)
        return

    # otherwise, we can convert the activation_post_process module call to quantize/dequantize node

    # 1. extract the information from activation_post_process module for generating
    # the quantize and dequantize operator
    dtype = activation_post_process.dtype  # type: ignore[attr-defined]

    is_dynamic = False
    if hasattr(activation_post_process, "is_dynamic"):
        is_dynamic = activation_post_process.is_dynamic  # type: ignore[assignment]

    if dtype in [torch.quint8, torch.qint8, torch.qint32] and \
            (not is_dynamic):
        # TODO: probably should cleanup this condition check, it's hard
        # to reason about this if and the following elif

        # uint8/int8/int32 static quantization branch

        # 1. extract information for inserting q/dq node from activation_post_process
        node_type = "call_function"
        quantize_op : Optional[Callable] = None
        scale, zero_point = activation_post_process.calculate_qparams()  # type: ignore[attr-defined, operator]
        if is_per_channel(activation_post_process.qscheme):  # type: ignore[attr-defined]
            ch_axis = int(activation_post_process.ch_axis)  # type: ignore[attr-defined, arg-type]
            quantize_op = torch.ops.quantized_decomposed.quantize_per_channel.default
            dequantize_op = torch.ops.quantized_decomposed.dequantize_per_channel.default
            quant_min = activation_post_process.quant_min
            quant_max = activation_post_process.quant_max
            dtype_ = to_underlying_dtype(dtype)
            qparams = {
                "_scale_": scale,
                "_zero_point_": zero_point,
                "_axis_": ch_axis,
                "_quant_min_": quant_min,
                "_quant_max_": quant_max,
                "_dtype_": dtype_
            }
        else:
            quantize_op = torch.ops.quantized_decomposed.quantize_per_tensor.default
            dequantize_op = torch.ops.quantized_decomposed.dequantize_per_tensor.default
            scale = float(scale)
            zero_point = int(zero_point)
            quant_min = activation_post_process.quant_min  # type: ignore[attr-defined]
            quant_max = activation_post_process.quant_max  # type: ignore[attr-defined]
            dtype_ = to_underlying_dtype(dtype)
            qparams = {
                "_scale_": scale,
                "_zero_point_": zero_point,
                "_quant_min_": quant_min,
                "_quant_max_": quant_max,
                "_dtype_": dtype_
            }

        # 2. replace activation_post_process node with quantize and dequantize
        with graph.inserting_before(node):
            input_node = node.args[0]
            quantize_op_inputs = [input_node]
            for key, value_or_node in qparams.items():
                # TODO: we can add the information of whether a value needs to
                # be registered as an attribute in qparams dict itself
                if key in ['_scale_', '_zero_point_'] and (not isinstance(value_or_node, (float, int))):
                    # For scale and zero_point values we register them as buffers in the root module.
                    # However, note that when the values are not tensors, as in the case of
                    # per_tensor quantization, they will be treated as literals.
                    # However, registring them as a node seems to cause issue with dynamo
                    # tracing where it may consider tensor overload as opposed to default.
                    # With extra check of scale and zero_point being scalar, it makes
                    # sure that the default overload can be used.
                    # TODO: maybe need more complex attr name here
                    qparam_node = create_getattr_from_value(
                        model, graph, module_path + prefix + key, value_or_node)
                    quantize_op_inputs.append(qparam_node)
                else:
                    # for qparams that are not scale/zero_point (like axis, dtype) we store them as literals in the graph.
                    quantize_op_inputs.append(value_or_node)

            quantized_node = graph.create_node(node_type, quantize_op, tuple(quantize_op_inputs), {})
            # use the same qparams from quantize op
            dq_inputs = [quantized_node] + quantize_op_inputs[1:]
            dequantized_node = graph.call_function(
                dequantize_op,
                tuple(dq_inputs),
                {}
            )
            node.replace_all_uses_with(dequantized_node)
            graph.erase_node(node)
    elif is_dynamic:

        # uint8/int8/fp16 dynamic quantization

        # 1. extract information for inserting q/dq node from activation_post_process
        node_type = "call_function"
        quantize_op = torch.ops.quantized_decomposed.quantize_per_tensor.tensor
        # we only use choose_qparams for is_decomposed now,
        # but we should probably align the non-decomposed path with this as well,
        # and that can be done after we remove reduce_range flag
        # 1. extract qparams from activation_post_process module
        dtype_ = to_underlying_dtype(dtype)
        assert dtype_ in [torch.uint8, torch.int8], \
            "only uint8 and int8 are supported in reference flow for " \
            "dynamic quantization right now"
        quant_min = activation_post_process.quant_min  # type: ignore[attr-defined]
        quant_max = activation_post_process.quant_max  # type: ignore[attr-defined]
        qscheme = getattr(activation_post_process, "qscheme", torch.per_tensor_affine)  # type: ignore[attr-defined]
        eps = getattr(activation_post_process, "eps", torch.finfo(torch.float32).eps)  # type: ignore[attr-defined]
        # note: scale and zero_point are missing for quantize_per_tensor op
        # we'll need to get this from choose_qparams op, which we'll add after
        # this step
        qparams = {
            "_quant_min_": quant_min,
            "_quant_max_": quant_max,
            "_eps_": eps,
            "_dtype_": dtype_
        }

        choose_qparams_op = _QSCHEME_TO_CHOOSE_QPARAMS_OP[qscheme]
        # 2. insert choose_qparams op and update the qparams list
        with graph.inserting_before(node):
            input_node = node.args[0]
            choose_qparams_op_inputs = [node.args[0]]
            for key, value in qparams.items():
                # we have quant_min, quant_max and dtype, all should be stored
                # as literals
                choose_qparams_op_inputs.append(value)
            choose_qparams_node = graph.create_node(
                "call_function",
                choose_qparams_op,
                tuple(choose_qparams_op_inputs),
                {}
            )
            # choose_qparms returns (scale, zero_point)
            scale_node = graph.create_node(
                "call_function",
                operator.getitem,
                (choose_qparams_node, 0),
                {}
            )
            zero_point_node = graph.create_node(
                "call_function",
                operator.getitem,
                (choose_qparams_node, 1),
                {}
            )
            quant_min = qparams["_quant_min_"]
            quant_max = qparams["_quant_max_"]
            dtype = qparams["_dtype_"]
            qparams = {
                "_scale_": scale_node,
                "_zero_point_": zero_point_node,
                "_quant_min_": quant_min,
                "_quant_max_": quant_max,
                "_dtype_": dtype
            }

        # 3. replace activation_post_process node to quantize and dequantize node
        with graph.inserting_before(node):
            input_node = node.args[0]
            quantize_op_inputs = [input_node]
            for key, value_or_node in qparams.items():
                # TODO: we can add the information of whether a value needs to
                # be registered as an attribute in qparams dict itself
                if key in ['_scale_', '_zero_point_']:
                    # in this case we have a node in the graph since it's dynamically
                    # computed from the input, with choose_qparams op
                    qparam_node = value_or_node
                    quantize_op_inputs.append(qparam_node)
                else:
                    # for qparams that are not scale/zero_point (like axis, dtype) we
                    # store them as literals in the graph.
                    quantize_op_inputs.append(value_or_node)

            quantized_node = graph.create_node(node_type, quantize_op, tuple(quantize_op_inputs), {})
            # use the same qparams from quantize op
            dq_inputs = [quantized_node] + quantize_op_inputs[1:]
            # need to use the tensor variant of this op, since scale and zero_point
            # from choose_qparam are Tensors, instead of float/int, this is to
            # prevent these nodes being traced away by downstream systems
            dequantize_op = torch.ops.quantized_decomposed.dequantize_per_tensor.tensor
            dequantized_node = graph.call_function(
                dequantize_op,
                tuple(dq_inputs),
                {}
            )
            node.replace_all_uses_with(dequantized_node)
            graph.erase_node(node)
    elif dtype == torch.float16:
        raise NotImplementedError("decomposed to float16 op not implemented yet")

    # should not reach since we have checks in the beginning to make sure the
    # activation_post_process is supported

def _replace_observer_with_quantize_dequantize_node(
        model: torch.nn.Module,
        graph: Graph,
        node: Node,
        modules: Dict[str, torch.nn.Module],
        node_name_to_scope: Dict[str, Tuple[str, type]],
        node_name_to_qconfig: Dict[str, QConfigAny]) -> None:
    """ Replace activation_post_process module call node with quantize and
    dequantize node

    Before:
    ... -> observer_0(x) -> ...
    After:
    ... -> torch.quantize_per_tensor(x, ...) -> x.dequantize() -> ...
    """
    assert modules is not None
    assert isinstance(node.target, str)
    module_path, prefix = _get_module_path_and_prefix(node, node_name_to_scope, node_name_to_qconfig)
    activation_post_process = modules[node.target]
    # skip replacing observers to quant/dequant nodes if the qconfigs of all
    # consumers and producers of this observer are None
    skip_replacement = all(_has_none_qconfig(n, node_name_to_qconfig) for n in
                           list(node.args) + list(node.users.keys()))
    if skip_replacement or not _is_conversion_supported(activation_post_process):
        # didn't find corresponding quantize op and info for the activation_post_process
        # so we just remove the observer
        with graph.inserting_before(node):
            node.replace_all_uses_with(node.args[0])
            graph.erase_node(node)
        return

    # otherwise, we can convert the activation_post_process module call to quantize/dequantize node
    dtype = activation_post_process.dtype  # type: ignore[attr-defined]

    is_dynamic = False
    if hasattr(activation_post_process, "is_dynamic"):
        is_dynamic = activation_post_process.is_dynamic  # type: ignore[attr-defined, assignment]

    if dtype in [torch.quint8, torch.qint8, torch.qint32] and \
            (not is_dynamic):
        # TODO: probably should cleanup this condition check, it's hard
        # to reason about this if and the following elif

        # uint8/int8/int32 static quantization branch

        # 1. extract the information from activation_post_process module for generating
        # the quantize and dequantize operator
        node_type = "call_function"
        quantize_op : Optional[Callable] = None
        scale, zero_point = activation_post_process.calculate_qparams()  # type: ignore[attr-defined, operator]
        if is_per_channel(activation_post_process.qscheme):  # type: ignore[attr-defined]
            ch_axis = int(activation_post_process.ch_axis)  # type: ignore[attr-defined, arg-type]
            qparams = {"_scale_": scale, "_zero_point_": zero_point, "_axis_": ch_axis, "_dtype_": dtype}
            quantize_op = torch.quantize_per_channel
        else:
            scale = float(scale)
            zero_point = int(zero_point)
            qparams = {"_scale_": scale, "_zero_point_": zero_point, "_dtype_": dtype}
            quantize_op = torch.quantize_per_tensor

        # 2. replace activation_post_process node with quantize and dequantize
        with graph.inserting_before(node):
            input_node = node.args[0]
            quantize_op_inputs = [input_node]
            for key, value_or_node in qparams.items():
                # TODO: we can add the information of whether a value needs to
                # be registered as an attribute in qparams dict itself
                if key in ['_scale_', '_zero_point_']:
                    # For scale and zero_point values we register them as buffers in the root module.
                    # TODO: maybe need more complex attr name here
                    qparam_node = create_getattr_from_value(
                        model, graph, module_path + prefix + key, value_or_node)
                    quantize_op_inputs.append(qparam_node)
                else:
                    # for qparams that are not scale/zero_point (like axis, dtype) we store them as literals in the graph.
                    quantize_op_inputs.append(value_or_node)

            quantized_node = graph.create_node(node_type, quantize_op, tuple(quantize_op_inputs), {})
            dequantized_node = graph.call_method("dequantize", args=(quantized_node,))
            node.replace_all_uses_with(dequantized_node)
            graph.erase_node(node)
    elif is_dynamic:

        # uint8/int8/fp16 dynamic quantization branch

        node_type = "call_function"
        quantize_op = torch.quantize_per_tensor_dynamic
        # TODO: get reduce range from observer
        # reduce_range = activation_post_process.reduce_range
        reduce_range = torch.backends.quantized.engine in ("fbgemm", "x86")
        qparams = {"_dtype_": dtype, "_reduce_range_": reduce_range}

        with graph.inserting_before(node):
            input_node = node.args[0]
            quantize_op_inputs = [input_node]
            for key, value in qparams.items():
                quantize_op_inputs.append(value)

            quantized_node = graph.create_node(node_type, quantize_op, tuple(quantize_op_inputs), {})
            dequantized_node = graph.call_method("dequantize", args=(quantized_node,))
            node.replace_all_uses_with(dequantized_node)
            graph.erase_node(node)
    elif dtype == torch.float16:
        node_type = "call_method"
        quantize_op = "to"  # type: ignore[assignment]
        qparams = {"_dtype_": dtype}
        with graph.inserting_before(node):
            input_node = node.args[0]
            quantize_op_inputs = [input_node]
            for key, value in qparams.items():
                # TODO: we can add the information of whether a value needs to
                # be registered as an attribute in qparams dict itself
                quantize_op_inputs.append(value)

            quantized_node = graph.create_node(node_type, quantize_op, tuple(quantize_op_inputs), {})
            dequantized_node = graph.call_method("dequantize", args=(quantized_node,))
            node.replace_all_uses_with(dequantized_node)
            graph.erase_node(node)

    # should not reach since we have checks in the beginning to make sure the
    # activation_post_process is supported

# this is a temporary hack for custom module, we may want to implement
# this properly after the custom module class design is finalized
# TODO: DeQuantStubs are currently inserted only after custom module LSTM, while observers are inserted
# after all other custom modules. In the future, we should simply insert QuantStubs before and DeQuantStubs
# after custom modules in general, and replace these with "quantize" and "dequantize" nodes respectively.
def _replace_observer_or_dequant_stub_with_dequantize_node(node: Node, graph: Graph) -> None:
    call_custom_module_node = node.args[0]
    assert isinstance(call_custom_module_node, Node), \
        f"Expecting the for call custom module node to be a Node, but got {call_custom_module_node}"
    node.replace_all_uses_with(call_custom_module_node)
    graph.erase_node(node)
    _insert_dequantize_node(call_custom_module_node, graph)

def _is_conversion_supported(activation_post_process: torch.nn.Module) -> bool:
    dtype = activation_post_process.dtype  # type: ignore[attr-defined]

    is_dynamic = False
    if hasattr(activation_post_process, "is_dynamic"):
        is_dynamic = activation_post_process.is_dynamic  # type: ignore[attr-defined, assignment]

    return (
        (dtype in [torch.quint8, torch.qint8, torch.qint32] and (not is_dynamic)) or  # type: ignore[return-value]
        is_dynamic or
        dtype == torch.float16
    )

def _has_none_qconfig(node: Argument, node_name_to_qconfig: Dict[str, QConfigAny]) -> bool:
    """ Check if a node has a qconfig of None, i.e. user requested to not quantize
    the node
    """
    return isinstance(node, Node) and node.name in node_name_to_qconfig and node_name_to_qconfig[node.name] is None

def _run_weight_observers(observed: GraphModule, backend_config: BackendConfig) -> None:
    """ Extract the subgraph that produces the weight for dynamic quant
    or weight only quant node and run the subgraph to observe the weight.
    Note that the observers of dynamic quant or weight only quant ops are
    run during the convert step.
    """
    for node in observed.graph.nodes:
        if node.op != "call_function":
            continue
        for node_arg in node.args:
            # node_arg is weight
            if node_arg and node_arg_is_weight(node, node_arg):
                weight_observer_nodes = collect_producer_nodes(node_arg)
                if weight_observer_nodes is None:
                    continue
                weight_observer_module = \
                    graph_module_from_producer_nodes(
                        observed, weight_observer_nodes)
                # run the weight observer
                weight_observer_module()

def _maybe_recursive_remove_dequantize(arg: Any, node: Node, graph: Graph) -> None:
    """ If the arg is a dequantize Node, or a list/tuple/dict of dequantize Node,
    we'll recursively remove the dequantize Node
    """
    if isinstance(arg, Node) and \
       arg.op == "call_method" and \
       arg.target == "dequantize":
        quantize_node = arg.args[0]
        # we only replace the specific use since dequantize could be used by other nodes
        # as well
        node.replace_input_with(arg, quantize_node)
    elif isinstance(arg, (list, tuple)):
        for arg_element in arg:
            _maybe_recursive_remove_dequantize(arg_element, node, graph)
    elif isinstance(arg, dict):
        for arg_element in arg.values():
            _maybe_recursive_remove_dequantize(arg_element, node, graph)
    else:
        warnings.warn(f"Unsupported node type in recursive remove dequantize: {type(arg)}")

def _get_module_path_and_prefix(
        obs_node: Node,
        node_name_to_scope: Dict[str, Tuple[str, type]],
        node_name_to_qconfig: Dict[str, QConfigAny]) -> Tuple[str, str]:
    """ Given and observer node, get the `Scope` or the fully qualified name for
    the submodule containing the observed node, also return a prefix of "_input"
    when the observed node is an input of a F.linear op, and not the output of another
    quantized op.
    TODO: this logic is hacky, we should think about how to remove it or make it more
    general
    """
    observed_node = obs_node.args[0]
    # an observer can be inserted for both input of the next operator or output of the previous
    # operator (they can be the same)
    # this flag identifies if the observer is inserted only because the observed node is
    # the input of the next operator
    assert isinstance(observed_node, Node), \
        f"Expecting observed node to be a Node, but got {observed_node}"
    is_input_observer_only = node_name_to_qconfig[observed_node.name] is None \
        if observed_node.name in node_name_to_qconfig else None
    if is_input_observer_only:
        # if the quantize function is at the input of op, then we find the first user of the observer_node
        # to get the path. If a linear call_function is in the user list, we return the first instance
        # of linear node to get the FQN.
        users = list(obs_node.users)
        first_linear_use_or_first_use = users[0] if users else None
        linear_node = None
        for n in users:
            if n.op == "call_function" and n.target == torch.nn.functional.linear:
                linear_node = n
                break
        if linear_node:
            first_linear_use_or_first_use = linear_node
        prefix = "_input"
    else:
        # if the quantize function is at the output of the op, we use the observer input node to get the path
        first_linear_use_or_first_use = observed_node
        prefix = ""

    if first_linear_use_or_first_use and first_linear_use_or_first_use.name in node_name_to_scope:
        module_path, _ = node_name_to_scope[first_linear_use_or_first_use.name]
    else:
        # TODO: it's not used, so actually we can skip quantization
        # but this requires changing return type of quantize_node
        # we can fix it later if needed
        module_path = ""
    return module_path, prefix

def _insert_dequantize_node(
        node: Node,
        graph: Graph) -> None:
    """ Inserts dequantize node for `node` in `graph`
    """
    with graph.inserting_after(node):
        dequantize_node = graph.call_method("dequantize", (node,))
        for user_node in dict(node.users):
            if user_node is not dequantize_node:
                user_node.replace_input_with(node, dequantize_node)

def _maybe_get_observer_for_node(
        node: Node,
        modules: Dict[str, torch.nn.Module]
) -> Optional[torch.nn.Module]:
    """
    If the node is observed, return the observer
    instance. Otherwise, return None.
    """
    for maybe_obs_node, _ in node.users.items():
        if maybe_obs_node.op == 'call_module':
            maybe_obs = modules[str(maybe_obs_node.target)]
            if _is_activation_post_process(maybe_obs):
                return maybe_obs
    return None

def convert_standalone_module(
        node: Node,
        modules: Dict[str, torch.nn.Module],
        model: torch.fx.GraphModule,
        is_reference: bool,
        backend_config: Optional[BackendConfig]) -> None:
    """ Converts a observed standalone module to a quantized standalone module by calling
    the fx convert api, currently using the same `is_reference` flag as parent, but we may
    changing this behavior in the future (e.g. separating quantization and lowering for
    standalone module as well)

    Args:
      - node: The call_module node of the observed standalone module
      - modules: named_module of original model
      - model: original model
      - is_reference: a flag from parent provided by user to decide if we want to
        produce a reference model or a fbgemm/qnnpack model
      - backend_config: backend configuration of the target backend of quantization
    """
    # TODO: remove is_reference flag
    if is_reference:
        convert_fn = torch.ao.quantization.quantize_fx.convert_to_reference_fx
    else:
        convert_fn = torch.ao.quantization.quantize_fx.convert_fx  # type: ignore[attr-defined]
    # We know that observed standalone module is a GraphModule since
    # it's produced by us
    observed_standalone_module : GraphModule = modules[str(node.target)]  # type: ignore[assignment]
    sm_input_quantized_idxs = \
        observed_standalone_module \
        .meta["_observed_graph_module_attrs"].standalone_module_input_quantized_idxs
    # remove the dequantize nodes for inputs
    args = list(node.args)
    for idx in range(len(args)):
        if idx in sm_input_quantized_idxs:
            arg = args[idx]
            if arg.op == "call_method" and arg.target == "dequantize":  # type: ignore[union-attr]
                quantize_node = arg.args[0]  # type: ignore[union-attr]
                node.replace_input_with(arg, quantize_node)
                if len(arg.users) == 0:  # type: ignore[union-attr]
                    model.graph.erase_node(arg)
    # add dequantize node for output
    sm_output_quantized_idxs = \
        observed_standalone_module \
        .meta["_observed_graph_module_attrs"].standalone_module_output_quantized_idxs
    if len(sm_output_quantized_idxs) > 0:
        assert sm_output_quantized_idxs[0] == 0, "Currently only quantized"
        "output idxs = [0] is supported"

        # if it's non-empty, then it means the output is kept in quantized form
        # we'll just add a dequantize node after this node
        _insert_dequantize_node(node, model.graph)

    # TODO: allow convert_custom_config to override backend_config
    # for standalone module
    quantized_standalone_module = convert_fn(
        observed_standalone_module,
        backend_config=backend_config)
    parent_name, name = _parent_name(node.target)
    # update the modules dict
    setattr(modules[parent_name], name, quantized_standalone_module)
    modules[str(node.target)] = quantized_standalone_module

def convert_weighted_module(
        node: Node,
        modules: Dict[str, torch.nn.Module],
        observed_node_names: Set[str],
        node_name_to_qconfig: Dict[str, QConfigAny],
        backend_config: BackendConfig,
        is_decomposed: bool = False,
        is_reference: bool = False,
) -> None:
    """ Convert a weighted module to reference quantized module in the model
    If the QConfig of a QAT module is not set, the module will still be converted to
    a float module.

    Args:
      - node: The call_module node of the observed standalone module
      - modules: named_module of original model
      - observed_node_names: names for the set of observed fx node, we can skip
        this conversion if the node is not observed
    """
    original_module = modules[str(node.target)]
    qconfig: QConfigAny = original_module.qconfig  # type: ignore[assignment]
    weight_post_process = None
    qat_module_classes = get_qat_module_classes(backend_config)

    if isinstance(
            original_module,
            qat_module_classes):
        # Converting qat module to a float module, we need to attach
        # weight fake_quant to the module, weight fake_quant is assumed to be run during
        # QAT so we don't need to run it again here
        weight_post_process = original_module.weight_fake_quant
        original_module = original_module.to_float()  # type: ignore[operator]
        # change qat module to float module
        parent_name, name = _parent_name(node.target)
        setattr(modules[parent_name], name, original_module)

    is_observed = node.name in observed_node_names
    # If a qconfig is not defined for this node, then skip converting to a reference module
    if qconfig is None or _has_none_qconfig(node, node_name_to_qconfig) or not is_observed:
        return

    # skip converting to reference quantized module if the qconfig is not supported
    pattern_to_dtype_configs = get_pattern_to_dtype_configs(backend_config)
    dtype_configs = pattern_to_dtype_configs.get(type(original_module), [])
    if not _is_qconfig_supported_by_dtype_configs(qconfig, dtype_configs):
        return

    # TODO: rename weight_is_statically_quantized to weight_is_int8_quantized
    is_weight_quantized = weight_is_quantized(qconfig)

    # the condition for swapping the module to reference quantized module is:
    # weights need to be quantized
    if not is_weight_quantized:
        return

    fused_module = None
    float_module = original_module
    # extract the individual float_module and fused module
    if isinstance(original_module, torch.ao.nn.intrinsic._FusedModule):
        fused_module = float_module
        float_module = fused_module[0]  # type: ignore[index]

    # TODO: move this to the reference quantized module
    # weight_qparams or weight_qparams dict
    wq_or_wq_dict = {"is_decomposed": is_decomposed}
    if isinstance(float_module, torch.nn.RNNCellBase):
        weight_post_process_ih = qconfig.weight()  # type: ignore[union-attr, operator]
        weight_post_process_hh = qconfig.weight()  # type: ignore[union-attr, operator]
        weight_post_process_ih(float_module.weight_ih)
        weight_post_process_hh(float_module.weight_hh)
        weight_qparams_ih = get_qparam_dict(weight_post_process_ih)
        weight_qparams_hh = get_qparam_dict(weight_post_process_hh)
        wq_or_wq_dict.update({
            "weight_ih": weight_qparams_ih,
            "weight_hh": weight_qparams_hh,
        })
    elif isinstance(float_module, (torch.nn.LSTM, torch.nn.GRU)):
        # format for wq_or_wq_dict (flattened attributes):
        # {"weight_ih_l0_scale": ..., "weight_ih_l0_qscheme": ..., ...}
        for wn in float_module._flat_weights_names:
            if hasattr(float_module, wn) and wn.startswith("weight"):
                weight = getattr(float_module, wn)
                weight_post_process = qconfig.weight()  # type: ignore[union-attr, operator]
                if weight_post_process.dtype == torch.qint8:  # type: ignore[union-attr]
                    weight_post_process(weight)  # type: ignore[operator, misc]
                wq_or_wq_dict[wn] = get_qparam_dict(weight_post_process)
    else:
        # weight_post_process is None means the original module is not a QAT module
        # we need to get weight_post_process from qconfig in this case
        is_ptq = weight_post_process is None
        if is_ptq:
            weight_post_process = qconfig.weight()  # type: ignore[union-attr, operator]

        # Call weight observer/fake_quant at least once to ensure the scales and zero points
        # have the right shapes. Note: there are two cases where we don't have to do this:
        #
        # (1) QAT: The model's forward method already calls the weight observer/fake_quant,
        #     and this typically happens during training, so we don't need to do it here.
        #
        # (2) Non-reference (lowered) case: The quantized module's from_float method already
        #     calls the weight observer/fake_quant, so we don't have to do it here.
        #
        # Currently we ignore both cases and call the weight observer/fake_quant here
        # regardless, which is technically incorrect. For (1), this is mainly to preserve BC
        # in test code, which may not always train before convert. In the future, we should
        # break BC for these two cases. See https://github.com/pytorch/pytorch/issues/73941.
        #
        # For PT2, however, we don't need to preserve BC here, so we can skip this hack
        # for QAT. We identify this case as (is_decomposed + is_reference + is_qat).
        # Note that we still need it for PTQ in the PT2 flow since the model's forward
        # method doesn't call the weight observer.
        is_qat = not is_ptq
        if not (is_decomposed and is_reference and is_qat):
            weight_post_process(float_module.weight)  # type: ignore[operator]

        wq_or_wq_dict.update(get_qparam_dict(weight_post_process))

    # We use the same reference module for all modes of quantization: static, dynamic, weight_only
    # root_module_to_quantized_reference_module: module mapping from root (floating point) module class
    # to quantized reference module class, e.g. nn.Conv2d to nn.quantized._reference.Conv2d
    root_module_to_quantized_reference_module = get_root_module_to_quantized_reference_module(backend_config)
    ref_qmodule_cls = root_module_to_quantized_reference_module.get(type_before_parametrizations(float_module), None)
    assert (
        ref_qmodule_cls is not None
    ), f"No reference quantized module class configured for {type_before_parametrizations(float_module)}"
    ref_qmodule = ref_qmodule_cls.from_float(float_module, wq_or_wq_dict)  # type: ignore[attr-defined]
    if fused_module is not None:
        fused_module[0] = ref_qmodule  # type: ignore[operator]
    else:
        parent_name, name = _parent_name(node.target)
        setattr(modules[parent_name], name, ref_qmodule)

def _remove_previous_dequantize_in_custom_module(node: Node, prev_node: Node, graph: Graph) -> None:
    """
    Given a custom module `node`, if the previous node is a dequantize, reroute the custom as follows:

    Before: quantize - dequantize - custom_module
    After: quantize - custom_module
                 \\ - dequantize
    """
    # expecting the input node for a custom module node to be a Node
    assert isinstance(prev_node, Node), \
        f"Expecting the argument for custom module node to be a Node, but got {prev_node}"
    if prev_node.op == "call_method" and prev_node.target == "dequantize":
        node.replace_input_with(prev_node, prev_node.args[0])
        # Remove the dequantize node if it doesn't have other users
        if len(prev_node.users) == 0:
            graph.erase_node(prev_node)

def convert_custom_module(
        node: Node,
        graph: Graph,
        modules: Dict[str, torch.nn.Module],
        custom_module_class_mapping: Dict[QuantType, Dict[Type, Type]],
        statically_quantized_custom_module_nodes: Set[Node]) -> None:
    """ Converts an observed custom module to a quantized custom module based on
    `custom_module_class_mapping`
    For static quantization, we'll also remove the previous `dequantize` node and
    attach the observer node for output to the module, the observer for the node
    will be converted to a dequantize node instead of quantize-dequantize pairs
    later in the graph. In the end we would have a quantized custom module that
    has the same interface as a default quantized module in nn.quantized namespace,
    i.e. quantized input and quantized output.

    Args:
      - node: The call_module node of the observed standalone module
      - graph: The graph containing the node
      - modules: named_module of original model
      - custom_module_class_mapping: mapping from observed custom module class to
        quantized custom module class, used to swap custom modules
      - statically_quantized_custom_module_nodes: we'll add the custom module node
        if we find it is statically quantized, this will be used later when converting
        observers to quant/dequant node pairs, if the observed node is a statically
        quantized custom module nodes, we'll convert the observer to a dequantize node,
        this is to keep the interface the same as the default quantized module.
        TODO: maybe we want to redesign this part to align with reference model design
        as well, but there has been some discussions around the interface, so we can do
        it later.
    """
    observed_custom_module = modules[str(node.target)]
    maybe_obs = _maybe_get_observer_for_node(node, modules)
    qconfig = observed_custom_module.qconfig
    if activation_is_statically_quantized(qconfig):
        statically_quantized_custom_module_nodes.add(node)
        if _is_custom_module_lstm(node, modules):
            # The inputs are tuples in the form (input, (hidden0, hidden1))
            # Ensure all three input nodes are quantized
            assert (
                len(node.args) == 2 and
                isinstance(node.args[1], tuple) and
                len(node.args[1]) == 2
            )
            (inputs, (hidden0, hidden1)) = node.args  # type: ignore[misc]
            assert isinstance(inputs, Node)
            assert isinstance(hidden0, Node)
            assert isinstance(hidden1, Node)
            _remove_previous_dequantize_in_custom_module(node, inputs, graph)
            _remove_previous_dequantize_in_custom_module(node, hidden0, graph)
            _remove_previous_dequantize_in_custom_module(node, hidden1, graph)
        elif _is_custom_module_mha(node, modules):
            # Inputs are in the form (query, key, value)
            # TODO: This is the first step in enabling the full fx custom module
            # quantization path for MultiheadAttention, and only covers the inputs
            # to the module.
            # Additional handling is yet to be implemented for the outputs, similar
            # to LSTM custom module
            assert len(node.args) == 3
            query, key, value = node.args
            assert isinstance(query, Node)
            assert isinstance(key, Node)
            assert isinstance(value, Node)
            _remove_previous_dequantize_in_custom_module(node, query, graph)
            _remove_previous_dequantize_in_custom_module(node, key, graph)
            _remove_previous_dequantize_in_custom_module(node, value, graph)
        else:
            # remove the previous dequant node to ensure the inputs are quantized
            arg = node.args[0]
            assert isinstance(arg, Node)
            _remove_previous_dequantize_in_custom_module(node, arg, graph)
            # absorb the following observer into the module conversion
            activation_post_process = _maybe_get_observer_for_node(node, modules)
            assert activation_post_process is not None
            observed_custom_module.activation_post_process = activation_post_process

    # swap the observed custom module to quantized custom module
    quantized_custom_module_class = get_swapped_custom_module_class(
        observed_custom_module, custom_module_class_mapping, qconfig)
    quantized_custom_module = \
        quantized_custom_module_class.from_observed(observed_custom_module)
    parent_name, name = _parent_name(node.target)
    setattr(modules[parent_name], name, quantized_custom_module)

def convert(
        model: GraphModule, is_reference: bool = False,
        convert_custom_config: Union[ConvertCustomConfig, Dict[str, Any], None] = None,
        is_standalone_module: bool = False,
        _remove_qconfig_flag: bool = True,
        qconfig_mapping: Union[QConfigMapping, Dict[str, Any], None] = None,
        backend_config: Union[BackendConfig, Dict[str, Any], None] = None,
        is_decomposed: bool = False) -> GraphModule:
    """
    We will convert an observed model (a module with observer calls) to a reference
    quantized model, the rule is simple:
    1. for each observer module call in the graph, we'll convert it to calls to
       quantize and dequantize functions based on the observer instance
    2. for weighted operations like linear/conv, we need to convert them to reference
       quantized module, this requires us to know whether the dtype configured for the
       weight is supported in the backend, this is done in prepare step and the result
       is stored in observed_node_names, we can decide whether we need to swap the
       module based on this set

    Args:
       * `is_standalone_module`: when this flag is True, it means we are quantizing
       a submodule that is not inlined in parent module, and will be quantized
       separately as one unit.

       * `is_decomposed`: a boolean flag to indicate whether we want to use the
        quantize operator for decomposed quantized tensor
        (torch.ops.quantized_decomposed.quantize_per_tensor) or default/standalone
        quantized tensor (torch.quantize_per_tensor)

    Returns:
         a quantized standalone module, whether input/output is quantized is
         specified by prepare_custom_config, with
         input_quantized_idxs, output_quantized_idxs, please
         see docs for :func:`~torch.ao.quantization.prepare_fx` for details
    """
    if convert_custom_config is None:
        convert_custom_config = ConvertCustomConfig()

    if isinstance(convert_custom_config, Dict):
        warnings.warn(
            "Passing a convert_custom_config_dict to convert is deprecated and will not be supported "
            "in a future version. Please pass in a ConvertCustomConfig instead.")
        convert_custom_config = ConvertCustomConfig.from_dict(convert_custom_config)

    if isinstance(qconfig_mapping, Dict):
        warnings.warn(
            "Passing a QConfig dictionary to convert is deprecated and will not be supported "
            "in a future version. Please pass in a QConfigMapping instead.")
        qconfig_mapping = QConfigMapping.from_dict(qconfig_mapping) if qconfig_mapping else None
    qconfig_mapping = copy.deepcopy(qconfig_mapping)
    assert(qconfig_mapping is None or isinstance(qconfig_mapping, QConfigMapping))

    if isinstance(backend_config, Dict):
        warnings.warn(
            "Passing a backend_config_dict to prepare is deprecated and will not be supported "
            "in a future version. Please pass in a BackendConfig instead.")
        backend_config = BackendConfig.from_dict(backend_config)

    if backend_config is None:
        backend_config = get_native_backend_config()

    assert _is_observed_module(model), \
        'incoming model must be produced by prepare_fx'
    observed_graph_module_attrs = model.meta["_observed_graph_module_attrs"]
    node_name_to_scope: Dict[str, Tuple[str, type]] = observed_graph_module_attrs.node_name_to_scope
    prepare_custom_config: PrepareCustomConfig = observed_graph_module_attrs.prepare_custom_config
    observed_node_names: Set[str] = observed_graph_module_attrs.observed_node_names
    node_name_to_qconfig: Dict[str, QConfigAny] = observed_graph_module_attrs.node_name_to_qconfig  # type: ignore[assignment]

    # mapping from fully qualified module name to module instance
    # for example,
    # {
    #   '': Model(...),
    #   'linear': Linear(...),
    #   'linear.weight_fake_quant': PerChannelMinMaxObserver(...),
    # }
    # We use remove_duplicate=False here because torch.cat uses
    # the same activation_post_process module instance but different names
    modules = dict(model.named_modules(remove_duplicate=False))

    # TODO refactor this code once we update the prepare logic to have additional information on
    # which graph nodes have been observed and share that with convert to decide which observers to ignore.
    if qconfig_mapping:
        prepare_qconfig_mapping: QConfigMapping = observed_graph_module_attrs.qconfig_mapping  # type: ignore[assignment]
        modules_copy = copy.deepcopy(modules)

        if observed_graph_module_attrs.is_qat:
            _update_qconfig_for_qat(qconfig_mapping, backend_config)
        _update_qconfig_for_fusion(model, qconfig_mapping)

        _compare_prepare_convert_qconfig_mappings(prepare_qconfig_mapping, qconfig_mapping)  # type: ignore[arg-type]
        convert_node_name_to_qconfig = _generate_node_name_to_qconfig(
            model, modules_copy, model.graph, qconfig_mapping, node_name_to_scope)
        # check the convert_node_name_to_qconfig generated and ensure that
        # all the values either match what was set in prepare node_name_to_qconfig
        # or are set to None in the convert_node_name_to_qconfig.
        for k, v in node_name_to_qconfig.items():
            assert k in convert_node_name_to_qconfig, 'Expected key {} in convert node_name_to_qconfig'.format(k)
            if convert_node_name_to_qconfig[k] is not None:
                assert qconfig_equals(v, convert_node_name_to_qconfig[k]), \
                    "Expected k {} to have the same value in prepare and convert QConfigMappings, " \
                    "but {} was updated to {}".format(k, v, convert_node_name_to_qconfig[k])
        node_name_to_qconfig = convert_node_name_to_qconfig

    custom_module_classes = get_custom_module_class_keys(convert_custom_config.observed_to_quantized_mapping)
    custom_module_class_mapping = convert_custom_config.observed_to_quantized_mapping

    if observed_graph_module_attrs.equalization_node_name_to_qconfig is not None:
        # If we want to do equalization then do the following:
        # Calculate the equalization scale, update the observers with the scaled
        # inputs, and scale the weight
        weight_eq_obs_dict = update_obs_for_equalization(model, modules)
        convert_eq_obs(model, modules, weight_eq_obs_dict)

    # always run weight observers in the top level forward method
    # for dynamic quant ops or weight only quant ops
    _run_weight_observers(model, backend_config)

    graph_inputs: List[str] = []
    for node in model.graph.nodes:
        if node.op == 'placeholder':
            graph_inputs.append(node.name)

    # additional state to override inputs to be quantized, if specified
    # by the user
    placeholder_node_seen_cnt = 0
    input_quantized_idxs: List[int] = prepare_custom_config.input_quantized_indexes
    output_quantized_idxs: List[int] = prepare_custom_config.output_quantized_indexes

    root_module_to_quantized_reference_module = get_root_module_to_quantized_reference_module(backend_config)
    # convert tuples so that it can work with isinstance(module, tuple_of_classes)
    root_module_classes = tuple(root_module_to_quantized_reference_module.keys())
    qat_module_classes = get_qat_module_classes(backend_config)
    fused_module_classes = get_fused_module_classes(backend_config)
    statically_quantized_custom_module_nodes: Set[Node] = set()

    for node in list(model.graph.nodes):
        if node.op == 'placeholder':
            cur_placeholder_node_idx = placeholder_node_seen_cnt
            placeholder_node_seen_cnt += 1
            if cur_placeholder_node_idx in input_quantized_idxs:
                # Inputs are assumed to be quantized if the user specified the
                # input_quantized_idxs override.
                # we need to dequantize the inputs since all operators took
                # floating point inputs in reference quantized models
                _insert_dequantize_node(node, model.graph)
        elif node.op == "output":
            # If the argument is empty we don't need to do anything
            if len(output_quantized_idxs) == 0:
                continue
            # Result are kept quantized if the user specified the
            # output_quantized_idxs override.
            # Remove the dequantize operator for the node in the end if any
            return_node = node
            output = node.args[0]
            # outputs can be Node, list, tuple, dict, other cases are not supported yet
            if isinstance(output, (list, tuple)):
                for idx in output_quantized_idxs:
                    _maybe_recursive_remove_dequantize(output[idx], return_node, model.graph)
            elif isinstance(output, (Node, dict)):
                # we treat dict as a single argument currently, but it can be extended
                # to support {"key": dtype} after we change output_quantized_idxs to
                # dict
                if 0 in output_quantized_idxs:
                    _maybe_recursive_remove_dequantize(output, return_node, model.graph)
            else:
                warnings.warn(f"Unsupported node type for output_quantized_idxs: {type(output)}")
        elif node.op == "call_module":
            mod = _get_module(node, modules)
            assert mod is not None
            if _is_activation_post_process(mod):
                observed_node = node.args[0]
                if observed_node in statically_quantized_custom_module_nodes:
                    _replace_observer_or_dequant_stub_with_dequantize_node(node, model.graph)
                else:
                    if is_decomposed:
                        _replace_observer_with_quantize_dequantize_node_decomposed(
                            model, model.graph, node, modules, node_name_to_scope,
                            node_name_to_qconfig)
                    else:
                        _replace_observer_with_quantize_dequantize_node(
                            model, model.graph, node, modules, node_name_to_scope,
                            node_name_to_qconfig)
            elif isinstance(mod, DeQuantStub):
                _replace_observer_or_dequant_stub_with_dequantize_node(node, model.graph)
            elif _is_observed_standalone_module(mod):
                convert_standalone_module(
                    node, modules, model, is_reference, backend_config)
            # below this point `type_before_parametrizations` is used
            # instead of `type` to handle situations with fx quant + sparsity
            elif type_before_parametrizations(mod) in set(
                    root_module_classes).union(qat_module_classes).union(fused_module_classes):
                # extra check for fused module classes to make sure they are fused module classes
                # of target modules
                if type_before_parametrizations(mod) in fused_module_classes and \
                   type_before_parametrizations(mod[0]) not in root_module_classes:  # type: ignore[index]
                    continue
                convert_weighted_module(
                    node, modules, observed_node_names, node_name_to_qconfig, backend_config,
                    is_decomposed, is_reference)
            elif type_before_parametrizations(mod) in custom_module_classes:
                convert_custom_module(
                    node, model.graph, modules, custom_module_class_mapping,
                    statically_quantized_custom_module_nodes)

    # remove deadcode after converting observers to quant/dequant ops
    model.graph.eliminate_dead_code()
    model = GraphModule(model, model.graph)

    # TODO: maybe move this to quantize_fx.py
    if not is_reference:
        model = lower_to_fbgemm(model, node_name_to_qconfig, node_name_to_scope)

    # TODO: this looks hacky, we want to check why we need this and see if we can
    # remove this
    # removes qconfig and activation_post_process modules
    if _remove_qconfig_flag:
        _remove_qconfig(model)
    model.delete_all_unused_submodules()
    model.meta.pop("_observed_graph_module_attrs", None)
    return model
