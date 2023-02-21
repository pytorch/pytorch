import copy
import torch
import warnings
from torch.fx import (
    GraphModule,
)
from torch.fx.graph import (
    Graph,
    Node,
)
from torch.fx.node import Argument

from ..quantize import (
    propagate_qconfig_,
)
from ..observer import (
    ObserverBase,
    _is_activation_post_process
)
from ..qconfig import (
    _is_reuse_input_qconfig,
    QConfigAny,
)
from ..qconfig_mapping import (
    QConfigMapping,
)
from .qconfig_mapping_utils import (
    _generate_node_name_to_qconfig,
    _update_qconfig_for_fusion,
    _get_flattened_qconfig_dict,
    _update_qconfig_for_qat,
)

from .quantize_handler import (
    _default_root_node_getter,
    _get_pattern_to_quantize_handlers,
    QuantizeHandler,
)

from torch.ao.quantization.utils import (
    Pattern,
    NodePattern,
)

from ._equalize import (
    is_equalization_observer,
    node_supports_equalization,
)

from .pattern_utils import (
    _sorted_patterns_dict,
)

from .match_utils import (
    _MatchResultWithQConfig,
    _find_matches,
)

from .utils import (
    _insert_dequant_stubs_for_custom_module_lstm_output,
    _is_custom_module_lstm,
    _maybe_get_custom_module_lstm_from_node_arg,
    _qconfig_satisfies_dtype_config_constraints,
    get_custom_module_class_keys,
    all_node_args_have_no_tensors,
    assert_and_get_unique_device,
    get_non_observable_arg_indexes_and_types,
    get_new_attr_name_with_prefix,
    node_arg_is_weight,
    node_arg_is_bias,
    NON_QUANTIZABLE_WEIGHT_OPS,
    ObservedGraphModuleAttrs,
)

from torch.ao.quantization import (
    PlaceholderObserver
)
from torch.ao.quantization.quantize import (
    convert
)

from ..utils import (
    _parent_name,
    get_qconfig_dtypes,
    get_swapped_custom_module_class,
    activation_is_statically_quantized,
)

from ..backend_config.utils import (
    get_pattern_to_dtype_configs,
    get_module_to_qat_module,
    get_fusion_pattern_to_root_node_getter,
)
from ..backend_config import (
    BackendConfig,
    DTypeConfig,
    get_native_backend_config,
)
from .custom_config import (
    PrepareCustomConfig,
    StandaloneModuleConfigEntry,
)

from torch._subclasses import FakeTensor

from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union, Callable


__all__ = [
    "insert_observers_for_model",
    "prepare",
    "propagate_dtypes_for_known_nodes",
]


# list of dtypes to not add observers to
_DO_NOT_OBS_DTYPE_LIST = [int, float, torch.bool, None]

# note: the following default target dtype info dicts are temporary,
# should be moved to the new programmable API class soon
_DEFAULT_FP32_QCONFIG_FOR_TARGET_DTYPE_INFO = {
    "input_act_obs_or_fq_ctr": torch.ao.quantization.qconfig._default_fp32_placeholder_qconfig.activation,
    "output_act_obs_or_fq_ctr": torch.ao.quantization.qconfig._default_fp32_placeholder_qconfig.activation
}

_DEFAULT_QUINT8_QCONFIG_FOR_TARGET_DTYPE_INFO = {
    "input_act_obs_or_fq_ctr": torch.ao.quantization.qconfig._default_quint8_placeholder_qconfig.activation,
    "output_act_obs_or_fq_ctr": torch.ao.quantization.qconfig._default_quint8_placeholder_qconfig.activation
}

def _is_activation_post_process_node(node: Node, named_modules: Dict[str, torch.nn.Module]) -> bool:
    return isinstance(node, torch.fx.Node) and node.op == "call_module" and \
        _is_activation_post_process(named_modules[str(node.target)])

def _get_dtype_and_is_dynamic(obs_or_fq_ctr: Optional[Callable]) -> Tuple[Optional[torch.dtype], bool]:
    """ Given a constructor for observer or fake quant module, returns
    a Tuple of dtype and is_dynamic
    """
    # TODO: instead of instantiating the instance, we can use inspect to get the default args
    if obs_or_fq_ctr is None:
        return None, False
    else:
        obs_or_fq = obs_or_fq_ctr()
        return obs_or_fq.dtype, getattr(obs_or_fq, "is_dynamic", False)

def _is_input_arg_dtype_supported_by_backend(
    arg: Argument,
    node: Node,
    qconfig: QConfigAny,
    dtype_config: DTypeConfig,
    backend_config: BackendConfig,
) -> bool:
    """ Check if the configured qconfig for the argument
    is supported by the backend or not
    """
    if isinstance(arg, (list, tuple)):
        return all(_is_input_arg_dtype_supported_by_backend(
            a, node, qconfig,
            dtype_config, backend_config) for a in arg)
    if not isinstance(arg, Node):
        return True
    # TODO: support check for standalone module
    is_weight = node_arg_is_weight(node, arg, backend_config)
    is_bias = node_arg_is_bias(node, arg, backend_config)
    is_activation = not is_weight and not is_bias
    if is_activation:
        input_act_obs_or_fq_ctr = node.meta["target_dtype_info"].get("input_act_obs_or_fq_ctr")
        qconfig_dtype, qconfig_is_dynamic = _get_dtype_and_is_dynamic(input_act_obs_or_fq_ctr)
        # TODO(future PR): remove the cast to bool below after figuring
        # out why backend_config has is_dynamic set to None in some cases.
        return (dtype_config.input_dtype is None) or (
            dtype_config.input_dtype == qconfig_dtype and
            bool(dtype_config.is_dynamic) == bool(qconfig_is_dynamic) and
            _qconfig_satisfies_dtype_config_constraints(qconfig, dtype_config.input_dtype_with_constraints)
        )
    elif is_weight:
        # TODO: move dtype check into `_qconfig_satisfies_dtype_config_constraints` as well
        weight_obs_or_fq_ctr = node.meta["target_dtype_info"].get("weight_obs_or_fq_ctr", None)
        qconfig_weight_dtype, _ = _get_dtype_and_is_dynamic(weight_obs_or_fq_ctr)
        backend_config_weight_dtype = dtype_config.weight_dtype
        dtype_matches = qconfig_weight_dtype == backend_config_weight_dtype
        qconfig_satisfies_constraints = _qconfig_satisfies_dtype_config_constraints(
            qconfig, dtype_config.weight_dtype_with_constraints, is_activation=False)
        return backend_config_weight_dtype is None or (dtype_matches and qconfig_satisfies_constraints)
    else:  # bias
        # TODO: move dtype check into `_qconfig_satisfies_dtype_config_constraints` as well
        bias_obs_or_fq_ctr = node.meta["target_dtype_info"].get("bias_obs_or_fq_ctr", None)
        qconfig_bias_dtype, _ = _get_dtype_and_is_dynamic(bias_obs_or_fq_ctr)
        backend_config_bias_dtype = dtype_config.bias_dtype
        return backend_config_bias_dtype is None or qconfig_bias_dtype == backend_config_bias_dtype

def _is_output_dtype_supported_by_backend(
    node: Node,
    qconfig: QConfigAny,
    dtype_config: DTypeConfig,
) -> bool:
    """ Check if the configured qconfig for the output
    is supported by the backend or not
    """
    # TODO: move dtype check into `_qconfig_satisfies_dtype_config_constraints` as well
    backend_config_output_dtype = dtype_config.output_dtype
    # TODO: we should check is_dynamic here as well, the code from _is_input_arg_dtype_supported_by_backend
    # from input activation check can be reused here
    qconfig_output_dtype = None
    output_act_obs_or_fq_ctr = node.meta["target_dtype_info"].get("output_act_obs_or_fq_ctr")
    qconfig_output_dtype, qconfig_output_is_dynamic = _get_dtype_and_is_dynamic(output_act_obs_or_fq_ctr)
    # TODO: this is a hack because we can only specify one activation_obs_or_fq for
    # qconfig (qconfig.activation), and we are only supporting dynamically quantized
    # linear op which has fp32 output dtype, this should be removed if we generalize
    # the structure of qconfig in the future
    if qconfig_output_is_dynamic:
        qconfig_output_dtype = torch.float32
    dtype_matches = qconfig_output_dtype == backend_config_output_dtype
    qconfig_satisfies_constraints = _qconfig_satisfies_dtype_config_constraints(
        qconfig, dtype_config.output_dtype_with_constraints)
    return backend_config_output_dtype is None or (dtype_matches and qconfig_satisfies_constraints)

def _is_observer_in_same_graph(node: Node, named_modules: Dict[str, torch.nn.Module]):
    """ Check if observer in same graph
    when the node output is not fp32 and input is 'placeholder'
    the input is assumed to be quantized, so it is observed
    in a different place rather than not observed.
    """
    node_output_dtype = _get_arg_target_dtype_as_output(node, named_modules)
    if len(node.args) > 0 and isinstance(node.args[0], Node):
        if node_output_dtype == torch.quint8 and node.args[0].op == 'placeholder':
            return False
    return True

def _is_pattern_dtype_config_and_qconfig_supported_by_backend(
    pattern: Optional[Pattern],
    matched_node_pattern: Optional[List[Node]],
    qconfig: QConfigAny,
    backend_config: BackendConfig,
) -> bool:
    """ Check if the dtype configuration of a pattern is supported by
    the backend or not, and whether the qconfig satisfies constraints
    specified in the corresponding dtype config.
    """
    if backend_config is None or pattern is None:
        return True
    assert matched_node_pattern is not None and len(matched_node_pattern) >= 1
    pattern_to_dtype_configs = get_pattern_to_dtype_configs(backend_config)
    dtype_configs: List[DTypeConfig] = pattern_to_dtype_configs.get(pattern, [])
    pattern_to_root_node_getter = get_fusion_pattern_to_root_node_getter(backend_config)

    root_node_getter = pattern_to_root_node_getter.get(pattern, _default_root_node_getter)
    root_node = root_node_getter(matched_node_pattern)
    input_node = root_node
    output_node = matched_node_pattern[0]
    for dtype_config in dtype_configs:
        # check if arg dtype are supported
        supported = True
        for arg in list(input_node.args) + list(input_node.kwargs.values()):
            supported = supported and _is_input_arg_dtype_supported_by_backend(
                arg, input_node, qconfig, dtype_config, backend_config)
        # check if output dtype is supported
        supported = supported and _is_output_dtype_supported_by_backend(
            output_node, qconfig, dtype_config)
        if supported:
            return True
    return False

def _get_standalone_module_configs(
    node: Node,
    named_modules: Dict[str, torch.nn.Module],
    prepare_custom_config: PrepareCustomConfig,
    parent_qconfig: QConfigAny,
    parent_backend_config: Optional[BackendConfig],
) -> Tuple[QConfigMapping, Tuple[Any, ...], PrepareCustomConfig, Optional[BackendConfig]]:
    """
    Returns the standalone module QConfigMapping and PrepareCustomConfig
    for `node`, assuming that the module pointed to by `node` is
    a standalone modules.
    """
    module_name = str(node.target)
    module_type = type(named_modules[module_name])  # type: ignore[index]
    # name config has precedence over type config
    config_entry = StandaloneModuleConfigEntry(None, (), None, None)
    config_entry = prepare_custom_config.standalone_module_classes.get(module_type, config_entry)
    config_entry = prepare_custom_config.standalone_module_names.get(module_name, config_entry)
    # fallback to use parent module's qconfig if user didn't specify qconfig dict
    qconfig_mapping = config_entry.qconfig_mapping or QConfigMapping().set_global(parent_qconfig)
    example_inputs = config_entry.example_inputs
    prepare_custom_config = config_entry.prepare_custom_config or PrepareCustomConfig()
    backend_config = config_entry.backend_config or parent_backend_config
    return (qconfig_mapping, example_inputs, prepare_custom_config, backend_config)

def _qat_swap_modules(
        root: torch.nn.Module,
        module_to_qat_module: Dict[Pattern, Type[torch.nn.Module]]) -> None:
    convert(root, mapping=module_to_qat_module, inplace=True, remove_qconfig=False)

def _add_matched_node_name_to_set(matched_node_pattern: NodePattern, s: Set[str]):
    if isinstance(matched_node_pattern, Node):
        s.add(matched_node_pattern.name)
    elif isinstance(matched_node_pattern, (list, tuple)):
        for maybe_node in matched_node_pattern:
            _add_matched_node_name_to_set(maybe_node, s)

def _insert_observer(
    node: Node,
    observer: ObserverBase,
    model: torch.nn.Module,
    named_modules: Dict[str, torch.nn.Module],
    graph: Graph,
) -> Node:
    """
    Attaches `observer` to `model`, and creates a node which calls
    `observer` on the output of `node`.
    """
    model_device = assert_and_get_unique_device(model)
    if model_device:
        observer.to(model_device)
    # add observer module as attribute
    if is_equalization_observer(observer):
        prefix = node.name + '_equalization_process_'
    else:
        prefix = 'activation_post_process_'
    get_new_observer_name = get_new_attr_name_with_prefix(prefix)
    observer_name = get_new_observer_name(model)
    setattr(model, observer_name, observer)
    named_modules[observer_name] = observer
    with graph.inserting_after(node):
        new_obs = graph.create_node(
            'call_module', observer_name, (node,), {})
    return new_obs

def _set_target_dtype_info_for_matched_node_pattern(
    matched_node_pattern: NodePattern,
    last_node: Node,
    qconfig: QConfigAny,
    backend_config: BackendConfig,
    named_modules: Dict[str, torch.nn.Module],
    cache_for_no_tensor_check: Dict[Node, bool],
    processed_nodes: Set[Node],
) -> None:
    """ Sets the target_dtype_info for each node in matched_node_pattern
    Note: processed_nodes is used to ensure we only process each node once
    """
    if isinstance(matched_node_pattern, (list, tuple)):
        for node_pattern in matched_node_pattern:
            _set_target_dtype_info_for_matched_node_pattern(
                node_pattern,
                last_node,
                qconfig,
                backend_config,
                named_modules,
                cache_for_no_tensor_check,
                processed_nodes
            )

    # set target_dtype_info if matched_node_pattern is a Node
    # other types of matched object, e.g. int, float literals, are ignored
    elif isinstance(matched_node_pattern, Node):
        # for pyre
        assert isinstance(matched_node_pattern, Node)
        node = matched_node_pattern
        if node in processed_nodes:
            return
        processed_nodes.add(node)

        if qconfig is None:
            return
        # TODO: refactor the following code in terms of apply a qconfig to a pattern
        # e.g. for a pattern with op1 -> op2 -> op3, and qconfig = QConfig(input_act=obs0, output_act=obs1)
        # we set the input_obs_or_fq_ctr for the arguments of op1 to based on qconfig.input_act,
        # and set output_obs_or_fq_ctr based on qconfig.output_act
        # this also requires we extend the structure of QConfig to support more fine
        # grained configurations
        target_dtype_info: Dict[str, Optional[Tuple[Union[torch.dtype, type], bool]]] = (
            _get_target_activation_dtype_for_node(
                node,
                qconfig,
                named_modules,
                cache_for_no_tensor_check,
            )
        )
        node.meta["target_dtype_info"] = target_dtype_info

def _get_target_activation_dtype_for_node(
    node: Node,
    qconfig: QConfigAny,
    named_modules: Dict[str, torch.nn.Module],
    cache_for_no_tensor_check: Dict[Node, bool],
) -> Dict[str, Optional[Tuple[Union[torch.dtype, type], bool]]]:
    """
    For each op attribute in the op's input activation, output activation,
    weight, bias - returns the settings of dtype and is_dynamic we expect
    for the `quantize` call in the reference model representation, or None
    if there is no `quantize` call needed.

    For example, if we have a node corresponding to `op0` in

      x0 -> op0 -> x1

    And we want a reference quantized representation to be

      x0 -> quant_static -> dequant -> op0 -> quant_dynamic -> dequant -> x1

    Then this function will return

      {
        "input_act_obs_or_fq_ctr": MinMaxObserver.with_args(dtype=torch.quint8, is_dynamic=False),
        "output_act_obs_or_fq_ctr": MinMaxObserver.with_args(dtype=torch.quint8, is_dynamic=False),
      }

    TODO(future PR, if needed): explicitly spell out the non-Tensor
    dtypes.
    """
    args_have_no_tensors = \
        all_node_args_have_no_tensors(
            node, named_modules, cache_for_no_tensor_check)
    if args_have_no_tensors:
        return {
            "input_act_obs_or_fq_ctr": None,
            "output_act_obs_or_fq_ctr": None,
        }
    # get qconfig to determine the eventual dtype of this node
    if qconfig is not None:
        act_dtype, weight_dtype, input_act_is_dynamic = \
            get_qconfig_dtypes(qconfig)

        # Currently `QConfig` only has one `activation` field.
        # For static quantization, it is reused for both input
        # and output activation. For dynamic quantization, this
        # field is currently only used for the input activation,
        # with the output activation being in fp32.
        # In the future this may change as we add more fields
        # to the `QConfig` object.
        output_act_dtype = act_dtype \
            if (not input_act_is_dynamic) else torch.float

        bias_dtype = torch.float16 \
            if (
                act_dtype == torch.float16
                and weight_dtype == torch.float16
                and (not input_act_is_dynamic)
            ) else torch.float
        return {
            "input_act_obs_or_fq_ctr": qconfig.activation,
            "weight_obs_or_fq_ctr": qconfig.weight,
            "bias_obs_or_fq_ctr": PlaceholderObserver.with_args(dtype=bias_dtype),
            "output_act_obs_or_fq_ctr": qconfig.activation,
        }
    return copy.copy(_DEFAULT_FP32_QCONFIG_FOR_TARGET_DTYPE_INFO)

def _get_arg_target_dtype_as_output(
    arg: Node,
    named_modules: Dict[str, torch.nn.Module],
) -> Optional[Union[torch.dtype, type]]:
    """ Get the target output activation dtype for
    the argument in the original graph, skipping inserted observers
    We are assuming that the observers are inserted correctly, and the dtype for
    argument in quantized graph will match what is specified by the qconfig
    """
    assert isinstance(arg, Node)
    # Custom module LSTM output is a tuple that we broke down into the internal nodes in order
    # to insert DeQuantStubs (see `_insert_dequant_stubs_for_custom_module_lstm_output`).
    # Since we modified the graph in this case, we must trace back from the args through
    # the specific nodes we added in order to reach the original LSTM node. Otherwise, we would
    # not be able to accurately detect whether this node is a consumer of custom module LSTM.
    custom_module_lstm_node = _maybe_get_custom_module_lstm_from_node_arg(arg, named_modules)
    output_act_obs_or_fq_ctr = None
    if custom_module_lstm_node is not None:
        output_act_obs_or_fq_ctr = custom_module_lstm_node.meta["target_dtype_info"]["output_act_obs_or_fq_ctr"]
    elif _is_activation_post_process_node(arg, named_modules):
        observed_arg = arg.args[0]
        assert isinstance(observed_arg, Node), "Currently we only support observing Node"
        output_act_obs_or_fq_ctr = observed_arg.meta["target_dtype_info"]["output_act_obs_or_fq_ctr"]
    else:
        output_act_obs_or_fq_ctr = \
            arg.meta["target_dtype_info"]["output_act_obs_or_fq_ctr"]
    output_act_dtype, _ = _get_dtype_and_is_dynamic(output_act_obs_or_fq_ctr)
    # TODO: should support is_dynamic here as well
    return output_act_dtype

def _get_arg_target_dtype_as_input_to_node(
    arg: Node,
    node: Node,
    named_modules: Dict[str, torch.nn.Module],
    backend_config: BackendConfig,
) -> Optional[Union[torch.dtype, type]]:
    """ Get the target argument dtype for the argument `arg`, as input
    to node `node`
    """
    assert isinstance(arg, Node)
    is_weight = node_arg_is_weight(node, arg, backend_config)
    is_bias = node_arg_is_bias(node, arg, backend_config)
    is_activation = not is_weight and not is_bias
    if is_activation:
        input_act_obs_or_fq_ctr = node.meta["target_dtype_info"].get("input_act_obs_or_fq_ctr")
        qconfig_dtype, _ = _get_dtype_and_is_dynamic(input_act_obs_or_fq_ctr)
        return qconfig_dtype
    elif is_weight:
        if node.target in NON_QUANTIZABLE_WEIGHT_OPS:
            return None
        else:
            weight_obs_or_fq_ctr = node.meta["target_dtype_info"].get("weight_obs_or_fq_ctr", None)
            qconfig_weight_dtype, _ = _get_dtype_and_is_dynamic(weight_obs_or_fq_ctr)
            return qconfig_weight_dtype
    else:
        bias_obs_or_fq_ctr = node.meta["target_dtype_info"].get("bias_obs_or_fq_ctr", None)
        qconfig_bias_dtype, _ = _get_dtype_and_is_dynamic(bias_obs_or_fq_ctr)
        return qconfig_bias_dtype

def _get_arg_target_is_dynamic_as_input_to_node(
    arg: Node,
    node: Node,
    named_modules: Dict[str, torch.nn.Module],
    backend_config: BackendConfig,
) -> bool:
    """ Get the target argument dtype for the argument `arg`, as input
    to node `node`
    """
    assert isinstance(arg, Node)
    is_weight = node_arg_is_weight(node, arg, backend_config)
    is_bias = node_arg_is_bias(node, arg, backend_config)
    is_activation = not is_weight and not is_bias
    if is_activation and "input_act_obs_or_fq_ctr" in node.meta["target_dtype_info"]:
        input_act_obs_or_fq_ctr = node.meta["target_dtype_info"].get("input_act_obs_or_fq_ctr")
        _, qconfig_is_dynamic = _get_dtype_and_is_dynamic(input_act_obs_or_fq_ctr)
        return qconfig_is_dynamic
    else:
        return False

def _maybe_insert_input_observer_for_arg_or_kwarg(
    node: Union[Node, Any],
    arg: Argument,
    qconfig: QConfigAny,
    model: torch.nn.Module,
    named_modules: Dict[str, torch.nn.Module],
    graph: Graph,
    qhandler: Optional[QuantizeHandler],
    prepare_custom_config: PrepareCustomConfig,
    backend_config: BackendConfig,
) -> Argument:
    """
    Given a `node` and an `arg`, inserts an input observer between
    `node` and `arg` if necessary.
    """
    # for ops such as torch.cat([x0, x1]),
    # traverse through the list
    if isinstance(arg, (list, tuple)):
        new_arg_to_return = []
        for inner_arg in arg:
            new_inner_arg = _maybe_insert_input_observer_for_arg_or_kwarg(
                node, inner_arg, qconfig, model, named_modules,
                graph,
                qhandler,
                prepare_custom_config,
                backend_config)
            new_arg_to_return.append(new_inner_arg)
        return type(arg)(new_arg_to_return)

    if not isinstance(arg, Node):
        return arg
    assert isinstance(arg, Node)
    # default (no observer)
    new_arg = arg

    is_standalone_module = qhandler is not None and qhandler.is_standalone_module()
    assert qconfig is not None
    if not is_standalone_module:
        # regular flow for most nodes, except standalone modules
        is_weight = node_arg_is_weight(node, arg, backend_config)

        _is_reuse_input_qconfig_ = _is_reuse_input_qconfig(qconfig)

        act_post_process_ctr = qconfig.weight if is_weight else \
            qconfig.activation

        arg_as_output_target_dtype = _get_arg_target_dtype_as_output(arg, named_modules)
        arg_as_input_target_dtype = _get_arg_target_dtype_as_input_to_node(
            arg, node, named_modules, backend_config)
        arg_as_input_target_is_dynamic = \
            _get_arg_target_is_dynamic_as_input_to_node(
                arg, node, named_modules, backend_config)  # type: ignore[arg-type]
        needs_obs = \
            (
                # the following code block is for static quantization
                (not arg_as_input_target_is_dynamic) and
                # if the dtypes are different, we need an observer
                (arg_as_output_target_dtype != arg_as_input_target_dtype) and
                # except if the second dtype is float, a dequant will be inserted
                # without an observer in convert
                # TODO(future PR): change this so a placeholder is inserted for
                # future dequants, to make the logic easier to understand
                (arg_as_input_target_dtype != torch.float) and
                # if arg output dtype is in _DO_NOT_OBS_DTYPE_LIST do not insert observer
                (arg_as_output_target_dtype not in _DO_NOT_OBS_DTYPE_LIST) and
                # if qconfig is reuse_input qconfig, we won't insert extra observer for input
                not _is_reuse_input_qconfig_
            ) or (
                # need to add input observer for dynamic quantization
                # only add observer for first input for now, we may need to extend
                # qconfig_dict and backend_config to support more general configurations
                # of dynamic quantization, e.g. dynamically quantizing second input, third
                # input etc.
                arg_as_input_target_is_dynamic and arg is node.args[0]
            )

    else:
        # custom flow for standalone modules
        _, _, sm_prepare_custom_config, _ = \
            _get_standalone_module_configs(
                node, named_modules, prepare_custom_config, qconfig, backend_config)
        sm_input_quantized_idxs = sm_prepare_custom_config.input_quantized_indexes

        # for args, this is set to the index of the current arg
        # for kwargs, this is left at None
        cur_input_idx = None
        for arg_idx, arg_to_check in enumerate(node.args):
            if arg_to_check is arg:
                cur_input_idx = arg_idx
                break

        if cur_input_idx is None:
            needs_obs = False
        else:
            arg_as_output_target_dtype = _get_arg_target_dtype_as_output(arg, named_modules)
            arg_as_input_target_dtype = torch.quint8 if cur_input_idx in sm_input_quantized_idxs \
                else torch.float
            needs_obs = (
                (arg_as_output_target_dtype != arg_as_input_target_dtype) and
                (arg_as_input_target_dtype != torch.float)
            )

        act_post_process_ctr = qconfig.activation

    if needs_obs:

        new_obs_mod = act_post_process_ctr()
        existing_obs_node = None

        # Before using the new observer, check if an observer
        # of the correct type already exists. If it does, use it.
        # This prevents duplicate observer insertions if a node is
        # used by multiple nodes.
        # TODO: this is looking into how the value is used in the future
        # we should remove this
        # removing this means we insert one observer for each use, even if they
        # have the same dtype, we can have an extra pass that removes the extra observers
        for maybe_obs_node, _ in arg.users.items():
            if maybe_obs_node.op == 'call_module':
                maybe_obs_mod = named_modules[maybe_obs_node.target]  # type: ignore[index]
                if (
                    type(maybe_obs_mod) == type(new_obs_mod) and
                    maybe_obs_mod.dtype == arg_as_input_target_dtype
                ):
                    existing_obs_node = maybe_obs_node
                    break

        if existing_obs_node is None:
            new_obs_node = _insert_observer(
                arg, new_obs_mod, model, named_modules, graph)
            # override this arg to be the observed arg
            new_arg = new_obs_node
        else:
            new_arg = existing_obs_node

    return new_arg


def _maybe_insert_input_observers_for_node(
    node: Node,
    qconfig: QConfigAny,
    model: torch.nn.Module,
    named_modules: Dict[str, torch.nn.Module],
    graph: Graph,
    qhandler: Optional[QuantizeHandler],
    prepare_custom_config: PrepareCustomConfig,
    backend_config: BackendConfig,
) -> None:
    """
    If needed, inserts observers to the input args and kwargs of `node`.
    Note: modifies `node` inplace.

    For example, if cur_node needs an observer after prev_node, we change from

      prev_node -> cur_node

    To

      prev_node -> obs -> cur_node
    """
    if qconfig is None:
        # if quantization is turned off for this node, we do not need
        # to insert input observers
        return
    assert qconfig is not None

    # Look through every input arg.  If that arg's target dtype does not
    # match the current node's target dtype, insert an observer.
    new_args = []
    for arg in node.args:
        new_arg = _maybe_insert_input_observer_for_arg_or_kwarg(
            node, arg, qconfig, model, named_modules, graph,
            qhandler,
            prepare_custom_config,
            backend_config)
        new_args.append(new_arg)

    new_kwargs = {}
    for k, kwarg in node.kwargs.items():
        new_kwarg = _maybe_insert_input_observer_for_arg_or_kwarg(
            node, kwarg, qconfig, model, named_modules, graph,
            qhandler,
            prepare_custom_config,
            backend_config)
        new_kwargs[k] = new_kwarg

    # assign the new args and kwargs to the node, inplace
    node.args = tuple(new_args)
    node.kwargs = new_kwargs

def _maybe_insert_input_equalization_observers_for_node(
    node: Node,
    equalization_qconfig: Any,
    model: torch.nn.Module,
    named_modules: Dict[str, torch.nn.Module],
    graph: Graph,
    is_branch: bool,
    backend_config: BackendConfig,
) -> None:
    """
    If `node` needs to be equalized, find the input/weight observers it needs in
    `equalization_qconfig`, creates them, and inserts it into `graph`.

    If `node` does not need an equalization observer, returns None.
    """
    if equalization_qconfig is None or not node_supports_equalization(node, named_modules):
        return

    if is_branch:
        warnings.warn(
            f"Cannot equalize {node} because it is part of a branch."
        )
        return

    new_args = []
    for arg in node.args:
        if not isinstance(arg, Node) or node_arg_is_bias(node, arg, backend_config):
            new_args.append(arg)
            continue

        is_weight = node_arg_is_weight(node, arg, backend_config)

        act_eq_process_ctr = equalization_qconfig.weight if is_weight else \
            equalization_qconfig.input_activation

        new_eq_obs_mod = act_eq_process_ctr()
        new_eq_obs_node = _insert_observer(
            arg, new_eq_obs_mod, model, named_modules, graph)

        new_args.append(new_eq_obs_node)

    # assign the new args and kwargs to the node, inplace
    node.args = tuple(new_args)

def _maybe_insert_output_observer_for_node(
    node: Node,
    model: torch.nn.Module,
    named_modules: Dict[str, torch.nn.Module],
    graph: Graph,
    node_name_to_match_result_with_qconfig: Dict[str, _MatchResultWithQConfig],
    matched_pattern: Any,
    qhandler: Optional[QuantizeHandler],
    is_qat: bool,
) -> Optional[Node]:
    """
    If `node` needs an output observer, creates it, inserts it into `graph`
    and returns it.

    If `node` does not need an output observer, returns None.
    """
    root_node, _, pattern, qhandler, qconfig = node_name_to_match_result_with_qconfig.get(
        node.name, (None, None, None, None, None))

    if qhandler is None:
        return None

    assert qconfig is not None
    assert node.op != 'output', 'observer insertion for outputs is handled elsewhere'

    is_standalone_module = qhandler is not None and qhandler.is_standalone_module()

    output_act_obs_or_fq_ctr = node.meta["target_dtype_info"].get("output_act_obs_or_fq_ctr")
    qconfig_dtype, _ = _get_dtype_and_is_dynamic(output_act_obs_or_fq_ctr)
    should_insert_observer = qconfig_dtype not in _DO_NOT_OBS_DTYPE_LIST + [torch.float]
    # TODO(future PR): move the following logic to
    # should_insert_observer_for_output
    should_insert_observer = should_insert_observer and \
        activation_is_statically_quantized(qconfig)

    # we never insert observers to output of standalone module, we assume
    # if needed, they are inserted inside the standalone module
    should_insert_observer = should_insert_observer and \
        (not is_standalone_module)

    if should_insert_observer:
        observer = qconfig.activation()
        return _insert_observer(node, observer, model, named_modules, graph)
    else:
        return None

def _maybe_insert_observers_before_graph_output(
    graph_output_node: Node,
    output_quantized_idxs: List[int],
    node_name_to_qconfig: Dict[str, QConfigAny],
    model: torch.nn.Module,
    named_modules: Dict[str, torch.nn.Module],
    graph: Graph,
) -> None:
    """
    If the output needs to be quantized and there are any nodes
    in the output which are not already observed, inserts observers
    for those nodes.
    """

    # TODO(future PR): update the output_quantized_idxs API to match
    # arbitrary data structures. There is always a single output, and
    # that output can have arbitrary nesting of values. List[int] is
    # not the right data type for this.
    assert output_quantized_idxs == [0] or output_quantized_idxs == [], \
        'unrecognized format of output_quantized_idxs'

    # Currently dequants are inserted in the convert step. So, we only
    # have to do anything if the output is hardcoded to be quantized
    if output_quantized_idxs == []:
        return
    # TODO(future PR): support more dtypes in model outputs, if necessary
    output_target_dtype = torch.quint8

    def _recursive_maybe_replace_node_with_obs(
        maybe_node: Argument,
        target_dtype: torch.dtype,
        node_name_to_qconfig: Dict[str, QConfigAny],
        model: torch.nn.Module,
        named_modules: Dict[str, torch.nn.Module],
        graph: Graph,
    ) -> Argument:
        """
        Navigate an arbitrary data structure of lists, tuples, dicts.
        For each container type, recurse on all inputs. Once any Node
        is found, insert an observer if needed and do not recurse further.

        For example, given a structure of

          {'foo1': [[bar1]], 'foo2': {'foo3': [[[bar3]]]}}

        we recurse down to bar1 and bar3, observe them if necessary,
        and if we inserted an observer then replace the original node
        with its observer.

        Returns the data structure with all nodes needing observation being
        replaced by their observers.
        """
        if isinstance(maybe_node, Node):
            # check dtype of this node
            this_node_dtype = _get_arg_target_dtype_as_output(
                maybe_node, named_modules)
            if this_node_dtype != target_dtype:
                # insert observer
                qconfig = node_name_to_qconfig.get(maybe_node.name)
                # TODO(future PR): see if we need to allow specifying qconfig
                #   on output nodes, to remove the restriction below.
                assert qconfig is not None, \
                    'Quantizing the output node without a qconfig is not supported'
                observer_mod = qconfig.activation()
                observer_node = _insert_observer(
                    maybe_node, observer_mod, model, named_modules, graph)
                return observer_node
            else:
                return maybe_node
        elif isinstance(maybe_node, (list, tuple)):
            results = []
            for inner_node in maybe_node:
                results.append(_recursive_maybe_replace_node_with_obs(
                    inner_node, target_dtype, node_name_to_qconfig, model, named_modules, graph))
            if isinstance(maybe_node, list):
                return results
            else:
                return tuple(results)
        elif isinstance(maybe_node, dict):
            results_dict = {}
            for k, inner_v in maybe_node.items():
                results_dict[k] = _recursive_maybe_replace_node_with_obs(
                    inner_v, target_dtype, node_name_to_qconfig, model, named_modules, graph)
            return results_dict
        else:
            return results

    new_args = []
    for old_arg in graph_output_node.args:
        new_args.append(
            _recursive_maybe_replace_node_with_obs(
                old_arg, output_target_dtype, node_name_to_qconfig, model, named_modules, graph))

    graph_output_node.args = tuple(new_args)  # type: ignore[assignment]


def _maybe_propagate_dtype_for_node(
    node: Node,
    target_dtype: Union[torch.dtype, type],
    node_name_to_match_result_with_qconfig: Dict[str, _MatchResultWithQConfig],
) -> None:
    """
    Assigns `target_dtype` to `node`, setting `is_dynamic` to False. If `node`
    is a general tensor shape op, also call this function recursively on
    the first argument, to propagate the dtype to the caller.
    """
    node.meta["target_dtype_info"]["input_act_obs_or_fq_ctr"] = None
    node.meta["target_dtype_info"]["output_act_obs_or_fq_ctr"] = None
    # if this is a copy node, propagate to first arg
    root_node, _, pattern, qhandler, qconfig = node_name_to_match_result_with_qconfig.get(
        node.name, (None, None, None, None, None))
    # TODO: probably need to remove `is_general_tensor_value_op`
    if qhandler is not None and qhandler.is_general_tensor_value_op():
        prev_node = node.args[0]
        if isinstance(prev_node, Node):
            _maybe_propagate_dtype_for_node(
                prev_node, target_dtype, node_name_to_match_result_with_qconfig)

def propagate_dtypes_for_known_nodes(
    graph: Graph,
    node_name_to_match_result_with_qconfig: Dict[str, _MatchResultWithQConfig],
) -> None:
    """
    Currently we assume that inputs to the graph are either `torch.float` or
    `torch.quint8`, which is not always correct. For ops such as
    `x.masked_fill(mask, value)`, we know that the dtype of  `mask` is a
    `BoolTensor`. Propagate this information throughout the graph.

    Note: not all dtypes in the graph will be correct after this pass, but a
    higher percentage of them will be correct. Hopefully in the future we can
    replace this with a better way to reason about dtypes of tensors.
    """
    for node in graph.nodes:
        non_observable_arg_dict = get_non_observable_arg_indexes_and_types(node)

        for arg_type in non_observable_arg_dict:
            non_observable_indices = non_observable_arg_dict[arg_type](node)

            for index in non_observable_indices:
                arg = node.args[index]

                # when an argument is a tuple, it does not show up as another node so we need to go through
                # all elements of the tuple manually
                if isinstance(arg, (tuple, list)):
                    arg_list = list(arg)
                else:
                    arg_list = [arg]

                for cur_arg in arg_list:
                    # hard coded arguments show up but aren't `Node` typed and do not need dtype propgated
                    if isinstance(cur_arg, torch.fx.node.Node):
                        _maybe_propagate_dtype_for_node(
                            cur_arg, arg_type, node_name_to_match_result_with_qconfig)

def _maybe_make_input_output_share_observers(
    node: Node,
    model: torch.nn.Module,
    named_modules: Dict[str, torch.nn.Module],
) -> bool:
    """
    Ensures that we share an observer
    for all input arguments as well as the output argument. In detail, given
    a graph of

      x0 -> obs0 -> op -> x2
                  /
      x1 -> obs1 /

    where node obs0 points to observer instance observer0,
    obs1 points to observer1 and obs2 points to observer2, we make nodes obs1
    and ob2 point to observer0.
    Returns: whether the operation succeeded or not
    """
    first_arg = None
    # find the first non-Tensor arg
    for i in range(len(node.args)):
        if isinstance(node.args[i], (Node, list, tuple)):
            first_arg = node.args[i]
            break

    # if there is no non-Tensor arg, return directly
    if first_arg is None:
        return False

    if isinstance(first_arg, (list, tuple)):
        first_arg_arg = first_arg[0]
    elif isinstance(first_arg, Node):
        first_arg_arg = first_arg
    else:
        return False

    # if we have a graph such as
    #   observed_node -> non_observed_node -> cat
    # we need to navigate up to the first observer
    iteration_guard = 0
    while not _is_activation_post_process_node(first_arg_arg, named_modules):
        if not isinstance(first_arg_arg, Node):
            return False
        # did not find an activation_post_process for the op
        if first_arg_arg.op == "placeholder":
            return False
        # trace back the args until we found the first Tensor/Node
        trace_back_node = None
        for i in range(len(first_arg_arg.args)):
            trace_back_node = first_arg_arg.args[i]
            if isinstance(trace_back_node, Node):
                break
        if trace_back_node is None:
            return False
        first_arg_arg = trace_back_node

        iteration_guard += 1
        if iteration_guard > 10000:
            raise AssertionError('Unable to find observer of previous node')

    assert isinstance(first_arg_arg, Node)
    target_to_use = first_arg_arg.target
    assert isinstance(target_to_use, str)
    obs_mod_to_use = named_modules[target_to_use]

    if isinstance(first_arg, (list, tuple)):
        # set all other input observer nodes to use that module
        for input_idx, input_arg in enumerate(first_arg):
            if input_idx == 0:
                continue
            iteration_guard = 0
            while not _is_activation_post_process_node(input_arg, named_modules):
                # failed to trace back since no input arg for the current node
                if len(input_arg.args) < 1:
                    return False
                input_arg = input_arg.args[0]
                iteration_guard += 1
                if iteration_guard > 10000:
                    raise AssertionError('Unable to find observer of previous node')

            parent_name, name = _parent_name(input_arg.target)
            setattr(named_modules[parent_name], name, obs_mod_to_use)

    # set the output observer node to use that module
    for output_obs_node, _ in node.users.items():
        assert _is_activation_post_process_node(output_obs_node, named_modules)
        parent_name, name = _parent_name(output_obs_node.target)
        setattr(named_modules[parent_name], name, obs_mod_to_use)

    # TODO(future PR): delete the orphaned observer modules
    return True

def _remove_output_observer(
        node: Node,
        model: torch.nn.Module,
        named_modules: Dict[str, torch.nn.Module]):
    items = list(node.users.items())
    for output_obs_node, _ in items:
        assert _is_activation_post_process_node(output_obs_node, named_modules)
        output_obs_node.replace_all_uses_with(node)
        model.graph.erase_node(output_obs_node)  # type: ignore[union-attr, operator]

def _swap_custom_module_to_observed(
        node: Node,
        qconfig: QConfigAny,
        named_modules: Dict[str, torch.nn.Module],
        prepare_custom_config: PrepareCustomConfig):
    custom_module = named_modules[node.target]  # type: ignore[index]
    custom_module_class_mapping = prepare_custom_config.float_to_observed_mapping
    observed_custom_module_class = \
        get_swapped_custom_module_class(
            custom_module, custom_module_class_mapping, qconfig)
    observed_custom_module = \
        observed_custom_module_class.from_float(custom_module)
    parent_name, name = _parent_name(node.target)
    setattr(named_modules[parent_name], name, observed_custom_module)

def insert_observers_for_model(
    model: GraphModule,
    node_name_to_match_result_with_qconfig: Dict[str, _MatchResultWithQConfig],
    node_name_to_qconfig: Dict[str, QConfigAny],
    prepare_custom_config: PrepareCustomConfig,
    equalization_config_map: Dict[str, Any],
    backend_config: BackendConfig,
    observed_node_names: Set[str],
    is_qat: bool,
) -> Optional[Node]:
    """
    Inserts observers, using the following high level algorithm:

    For each node in the graph:
      1. determine the target dtype of this node in the quantized graph, and save
           it for future steps
      2. determine the target dtype or all args and kwargs of this node
      3. if any arg or kwarg's target dtype does not match the current node's
           dtype, insert an observer
      4. if the current node needs an output observer, insert it

    For example:

    - starting graph:
        x0 -> linear -> x1

    - observed graph after processing x0:
        x0(fp32)

    - observed graph after processing linear:
        x0(fp32) -> x0_obs0(int8) -> linear(int8) -> linear_obs0(int8)

    - observed graph after processing x1:
        x0(fp32) -> x0_obs0(int8) -> linear(int8) -> linear_obs0(int8) -> x1

    After a node is processed, the naive observer placement is guaranteed to be
    complete for that node and all of its predecessors. There can be future
    passes which optimize the graph by deduplicating observers, etc.
    """

    # node.meta["target_dtype_info"] stores the target dtype information
    # that's derived from qconfig for the Node, for example, if we have
    # a conv2d node that has a qconfig
    # qconfig = QConfig(activation=..., weight=...)
    # # information for input and bias node omitted
    # # for getattr node
    # # weight = getattr(self, 'weight')
    # weight.meta["target_dtype_info"] = {
    #    'output_act_obs_or_fq_ctr': qconfig.weight,
    # }
    # # for conv2d node
    # # conv2d = call_function[target=torch.nn.functional.conv2d](
    # #            args=(input, weight, bias))
    # conv2d.meta["target_dtype_info"] = {
    #   'input_act_obs_or_fq_ctr': qconfig.activation
    #   'weight_obs_or_fq_ctr': qconfig.weight,
    #   'bias_obs_or_fq_ctr': PlaceholderObserver.with_args(dtype=torch.float32),
    #   'output_act_obs_or_fq_ctr': qconfig.activation,
    # }
    #
    cache_for_no_tensor_check: Dict[Node, bool] = {}

    # first, populate the dtype map based only on qconfig and qhandler
    # this assumes:
    # graph inputs are fp32 by default, and int8 where overriden
    # other nodes output dtype is specified by the qconfig
    named_modules = dict(model.named_modules(remove_duplicate=False))

    input_quantized_idxs: List[int] = prepare_custom_config.input_quantized_indexes
    output_quantized_idxs: List[int] = prepare_custom_config.output_quantized_indexes
    processed_nodes: Set[Node] = set()
    # initalize target_dtype_info
    for node in model.graph.nodes:
        node.meta["target_dtype_info"] = copy.copy(_DEFAULT_FP32_QCONFIG_FOR_TARGET_DTYPE_INFO)

    inputs_seen_counter = 0
    outputs_seen_counter = 0
    placeholder_node_to_input_index: Dict[Node, int] = {}
    # TODO: we probably don't need this counter since each graph will only have
    # one output node?
    output_node_to_output_index: Dict[Node, int] = {}
    for node in model.graph.nodes:
        if node.op == "placeholder":
            placeholder_node_to_input_index[node] = inputs_seen_counter
            inputs_seen_counter += 1
        if node.op == "output":
            output_node_to_output_index[node] = outputs_seen_counter
            outputs_seen_counter += 1

    # Step 1, set the observer or fake quantize module constructor for each node in the
    # matched_node_pattern

    for node_name, match_res_with_qconfig in node_name_to_match_result_with_qconfig.items():
        last_node, matched_node_pattern, pattern, qhandler, qconfig = match_res_with_qconfig
        assert qhandler is not None
        _set_target_dtype_info_for_matched_node_pattern(
            matched_node_pattern,
            last_node,
            qconfig,
            backend_config,
            named_modules,
            cache_for_no_tensor_check,
            processed_nodes
        )

    # Step 2. Special cases for some operators, we might be able to remove them
    # in the future if we know dtype information of each node better

    # Step 2.1. some settings are not based on patterns, we need to process each node
    # instead
    for node in model.graph.nodes:
        if node.op == "placeholder" and placeholder_node_to_input_index[node] in input_quantized_idxs:
            # users are not supposed to call calculate_qparams on PlaceholderObserver, and
            # this is OK because we are using this as a way to encode the dtypes of input
            # tensor, we won't actually insert these observers in the graph and won't
            # actually call calculate_qparams
            node.meta["target_dtype_info"] = copy.copy(_DEFAULT_QUINT8_QCONFIG_FOR_TARGET_DTYPE_INFO)
        elif node.op in ("call_module", "call_method", "call_function"):
            args_have_no_tensors = \
                all_node_args_have_no_tensors(
                    node, named_modules, cache_for_no_tensor_check)
            if args_have_no_tensors:
                node.meta["target_dtype_info"] = {
                    "input_act_obs_or_fq_ctr": None,
                    "output_act_obs_or_fq_ctr": None,
                }
        elif node.op == "output" and output_node_to_output_index[node] in output_quantized_idxs:
            node.meta["target_dtype_info"] = copy.copy(_DEFAULT_QUINT8_QCONFIG_FOR_TARGET_DTYPE_INFO)

    # Step 2.2, for nodes with known input dtypes, propagate them throughout the
    # graph. For example, if there is a call such as
    #   x1 = x0.masked_fill(mask, 1)
    # we propagate the type of mask to be torch.bool
    propagate_dtypes_for_known_nodes(model.graph, node_name_to_match_result_with_qconfig)

    # Step 3, check if the requested target_dtype_info is supported by backend or not
    # if not, we'll reset the target_dtye_info to use the default (float Tensor)

    # reset the counters and set of processed_nodes
    processed_nodes = set()
    for node_name, match_res_with_qconfig in node_name_to_match_result_with_qconfig.items():
        last_node, matched_node_pattern, pattern, qhandler, qconfig = match_res_with_qconfig
        is_supported_by_backend = _is_pattern_dtype_config_and_qconfig_supported_by_backend(
            pattern, matched_node_pattern, qconfig, backend_config)
        assert qhandler is not None

        # get output_act_dtype so that we don't also reset the special typed nodes
        # TODO: we might want to handle these more uniformly with the default path
        # this can be improved if we can use node.meta["val"]
        output_act_dtype, _ = _get_dtype_and_is_dynamic(node.meta["target_dtype_info"]["output_act_obs_or_fq_ctr"])
        if not is_supported_by_backend and output_act_dtype not in [None, int, float, torch.bool]:
            # restore target_dtype_info to default if it is not supported by backend
            _set_target_dtype_info_for_matched_node_pattern(
                matched_node_pattern,
                last_node,
                torch.ao.quantization.qconfig._default_fp32_placeholder_qconfig,
                backend_config,
                named_modules,
                cache_for_no_tensor_check,
                processed_nodes
            )

    # After this point, the current node and all of its arguments
    # have a target_dtype_info assigned. Now, we insert observers for inputs
    # of this node (if needed for this node), and the output of this node
    # (if needed for this node).

    # Since we are mutating the graph as we go, we iterate over the original
    # nodes before observer insertion, instead of model.graph.nodes.
    nodes_before_observation = list(model.graph.nodes)

    # Avoid duplicates custom module swaps for multiple nodes with same target.
    custom_module_names_already_swapped: Set[str] = set()

    # TODO: reuse placeholder_node_to_input_index and output_node_to_output_index
    # reset inputs/outputs counters
    inputs_seen_counter = 0
    outputs_seen_counter = 0
    results_node = None

    # TODO: change this to insert obs/fq by pattern instead of by node
    for node in nodes_before_observation:

        if node.op == 'placeholder':
            # if a graph input is in fp32, it does not need observation
            # if a graph input is in int8, we assume the observation happens
            #   outside of the graph, and no additional observation is needed
            pass

        elif node.op in ('call_module', 'call_method', 'call_function', 'output'):
            # check for matches
            last_node, matched_node_pattern, pattern, qhandler, qconfig = (
                node_name_to_match_result_with_qconfig.get(node.name, (None, None, None, None, None))  # type: ignore[assignment]
            )
            equalization_qconfig = equalization_config_map.get(node.name, None)

            this_node_dtype_info = node.meta["target_dtype_info"]
            if "val" in node.meta:
                output_is_a_tensor = (
                    this_node_dtype_info is not None and
                    isinstance(node.meta["val"], FakeTensor)
                )
            else:
                output_is_a_tensor = this_node_dtype_info is not None

            skip_inserting_observers = (
                (qconfig is None) or
                not output_is_a_tensor
            ) and (
                not node.op == 'output'
            )

            # TODO: take a closer look to see if we can remove this check
            # right now it is here because of `observed_node_names`, we are using
            # it as an indicator for swapping the modules to reference modules in
            # convert
            is_supported_by_backend = _is_pattern_dtype_config_and_qconfig_supported_by_backend(
                pattern, matched_node_pattern, qconfig, backend_config)

            if not skip_inserting_observers and is_supported_by_backend:
                named_modules = dict(model.named_modules(remove_duplicate=False))
                if node.op != 'output':
                    assert matched_node_pattern is not None
                    # add matched nodes to the observed node name set
                    _add_matched_node_name_to_set(matched_node_pattern, observed_node_names)

                    # This is currently only used for equalization.
                    # Checks if the current node is in a branch in which the two
                    # first layers are both being quantized.
                    #
                    # ex.       conv2
                    #         /
                    #      x -> conv1
                    #
                    # If this is the case, we will not apply equalization to the
                    # initial two layers.
                    is_quantized_branch = False
                    if (
                        len(node.args) > 0 and
                        isinstance(node.args[0], Node) and
                        len(node.args[0].users) > 1
                    ):
                        for user in node.args[0].users:
                            # Checks if there exists another user being quantized
                            is_user_quantized = (
                                node_name_to_qconfig.get(user.name, None) is not None or
                                (user.op == 'call_module' and isinstance(named_modules[str(user.target)], ObserverBase))
                            )
                            if user != node and is_user_quantized:
                                is_quantized_branch = True

                    pattern_to_root_node_getter = get_fusion_pattern_to_root_node_getter(backend_config)
                    root_node_getter = pattern_to_root_node_getter.get(pattern, _default_root_node_getter)
                    root_node = root_node_getter(matched_node_pattern)
                    is_input_node_of_the_pattern = node is root_node
                    if is_input_node_of_the_pattern:
                        # this modifies node inplace
                        _maybe_insert_input_observers_for_node(
                            node, qconfig, model, named_modules, model.graph,
                            qhandler,
                            prepare_custom_config,
                            backend_config)

                        # insert equalization input observers if needed
                        _maybe_insert_input_equalization_observers_for_node(
                            node, equalization_qconfig, model, named_modules, model.graph,
                            is_quantized_branch, backend_config)

                    is_last_node_of_pattern = node is last_node
                    is_general_tensor_value_op = \
                        (qhandler is not None and qhandler.is_general_tensor_value_op())
                    _is_reuse_input_qconfig_ = _is_reuse_input_qconfig(qconfig)

                    if is_last_node_of_pattern:
                        if _is_custom_module_lstm(node, named_modules, qconfig, qhandler):
                            # Currently custom module outputs are assumed to be already quantized,
                            # so we need to insert a DeQuantStub after the output. For custom module
                            # LSTM specifically, the outputs are also a nested tuple, so we must first
                            # break down the tuple to insert DeQuantStubs after the internal nodes.

                            # TODO: This currently diverges from how custom modules are handled today,
                            # where we insert observers after the output instead of DeQuantStubs, and
                            # replace these observers with "dequantize" nodes during convert. Conceptually,
                            # these output observers are the same as DeQuantStubs. In the future, we
                            # should resolve this inconsistency by inserting DeQuantStubs for all custom
                            # modules, not just for LSTM.
                            _insert_dequant_stubs_for_custom_module_lstm_output(node, model, named_modules, model.graph)
                            if(node.target not in custom_module_names_already_swapped):
                                custom_module_names_already_swapped.add(node.target)
                                _swap_custom_module_to_observed(node, qconfig, named_modules, prepare_custom_config)
                        else:
                            # this returns the new observer node if it was needed
                            maybe_output_obs_node = _maybe_insert_output_observer_for_node(
                                node, model, named_modules, model.graph, node_name_to_match_result_with_qconfig,
                                pattern, qhandler, is_qat)

                            if maybe_output_obs_node is not None:
                                # Update users of original node to use the output observer
                                # instead. For example, change
                                #
                                #           next_node
                                #          /
                                #   cur_node -> obs
                                #
                                # to
                                #
                                #                 next_node
                                #                 /
                                #   cur_node -> obs
                                #
                                # We need to save orig users before updating uses because
                                # the list of users will change as we update uses
                                orig_users = list(node.users.keys())
                                for user_node in orig_users:
                                    if user_node is maybe_output_obs_node:
                                        continue
                                    user_node.replace_input_with(node, maybe_output_obs_node)

                                _is_observer_in_same_graph_ = _is_observer_in_same_graph(
                                    node, named_modules)

                                # for general tensor value ops, we modify the graph
                                # to make all inputs and outputs use the first input's
                                # observer
                                if (is_general_tensor_value_op and _is_observer_in_same_graph_) or \
                                        _is_reuse_input_qconfig_:
                                    if not _maybe_make_input_output_share_observers(node, model, named_modules):
                                        _remove_output_observer(node, model, named_modules)

                                if qhandler is not None and qhandler.is_custom_module():
                                    if(node.target not in custom_module_names_already_swapped):
                                        custom_module_names_already_swapped.add(node.target)
                                        _swap_custom_module_to_observed(node, qconfig, named_modules, prepare_custom_config)

                else:  # output
                    _maybe_insert_observers_before_graph_output(
                        node, output_quantized_idxs,
                        node_name_to_qconfig,
                        model, named_modules, model.graph)

        #
        # After this point, the current node has input and output observers
        # that it needs for itself inserted.
        #

        # increment the counters, so future inputs and outputs are assigned
        # correct dtypes
        if node.op == 'placeholder':
            inputs_seen_counter += 1
        elif node.op == 'output':
            outputs_seen_counter += 1
            results_node = node

    return results_node

def _run_prepare_fx_on_standalone_modules(
    model: torch.nn.Module,
    is_qat: bool,
    named_modules: Dict[str, torch.nn.Module],
    node_name_to_match_result_with_qconfig: Any,
    prepare_custom_config: PrepareCustomConfig,
    backend_config: BackendConfig,
) -> None:
    """
    Runs prepare_fx on each standalone module. Note: this does
    not modify the graph, it just replaces the unobserved modules with
    their observed versions.
    """
    for (
        node_name,
        (root_node, _, pattern, qhandler, qconfig),
    ) in node_name_to_match_result_with_qconfig.items():
        if qhandler is None:
            continue
        elif not qhandler.is_standalone_module():
            continue

        sm_qconfig_mapping, sm_example_inputs, sm_prepare_custom_config, \
            sm_backend_config = _get_standalone_module_configs(
                root_node, named_modules, prepare_custom_config, qconfig, backend_config)

        standalone_module = named_modules[root_node.target]
        prepare = \
            torch.ao.quantization.quantize_fx._prepare_standalone_module_fx  # type: ignore[attr-defined]
        observed_standalone_module = \
            prepare(
                standalone_module,
                sm_qconfig_mapping,
                is_qat,
                example_inputs=sm_example_inputs,
                prepare_custom_config=sm_prepare_custom_config,
                backend_config=sm_backend_config)
        parent_name, name = _parent_name(root_node.target)
        setattr(named_modules[parent_name], name, observed_standalone_module)
        named_modules[root_node.target] = observed_standalone_module

def _save_state(
    observed: GraphModule,
    node_name_to_qconfig: Dict[str, QConfigAny],
    node_name_to_scope: Dict[str, Tuple[str, type]],
    prepare_custom_config: PrepareCustomConfig,
    equalization_node_name_to_qconfig: Dict[str, Any],
    qconfig_mapping: QConfigMapping,
    is_qat: bool,
    observed_node_names: Set[str],
) -> None:
    observed.meta["_observed_graph_module_attrs"] = (
        ObservedGraphModuleAttrs(
            node_name_to_qconfig=node_name_to_qconfig,
            node_name_to_scope=node_name_to_scope,
            prepare_custom_config=prepare_custom_config,
            equalization_node_name_to_qconfig=equalization_node_name_to_qconfig,
            qconfig_mapping=qconfig_mapping,
            is_qat=is_qat,
            observed_node_names=observed_node_names,
        )
    )

def prepare(
        model: GraphModule,
        qconfig_mapping: Union[QConfigMapping, Dict[str, Any]],
        is_qat: bool,
        node_name_to_scope: Dict[str, Tuple[str, type]],
        example_inputs: Tuple[Any, ...],
        prepare_custom_config: Union[PrepareCustomConfig, Dict[str, Any], None] = None,
        _equalization_config: Union[QConfigMapping, Dict[str, Any], None] = None,
        backend_config: Union[BackendConfig, Dict[str, Any], None] = None,
        is_standalone_module: bool = False) -> GraphModule:
    """ standalone_module means it a submodule that is not inlined in
    parent module, and will be quantized separately as one unit.

    How the standalone module is observed is specified by `input_quantized_idxs` and
    `output_quantized_idxs` in the prepare_custom_config for the standalone module
    Args:
        node_name_to_scope: mapping from node name to the scope of the module which contains the node.
        The scope is a tuple of fully qualified path of the module and the type of the module
    Returns:
        model(GraphModule): prepared standalone module
        attributes related to standalone module
        in model.meta["_observed_graph_module_attrs"]:
            is_observed_standalone_module (bool): boolean value that shows whether the
            current model is a observed standalone module or not
            standalone_module_input_quantized_idxs(List[Int]): a list of
                indexes for the graph input that is expected to be quantized,
                same as input_quantized_idxs configuration provided
                for the standalone module
            standalone_module_output_quantized_idxs(List[Int]): a list of
                indexs for the graph output that is quantized
                same as input_quantized_idxs configuration provided
                for the standalone module
    """
    if prepare_custom_config is None:
        prepare_custom_config = PrepareCustomConfig()
    if _equalization_config is None:
        _equalization_config = QConfigMapping()

    if isinstance(qconfig_mapping, Dict):
        warnings.warn(
            "Passing a QConfig dictionary to prepare is deprecated and will not be supported "
            "in a future version. Please pass in a QConfigMapping instead.")
        qconfig_mapping = QConfigMapping.from_dict(qconfig_mapping)

    if isinstance(_equalization_config, Dict):
        warnings.warn(
            "Passing a QConfig dictionary to prepare for equalization is deprecated and will not "
            "be supported in a future version. Please pass in a QConfigMapping instead.")
        _equalization_config = QConfigMapping.from_dict(_equalization_config)

    if isinstance(prepare_custom_config, Dict):
        warnings.warn(
            "Passing a prepare_custom_config_dict to prepare is deprecated and will not be supported "
            "in a future version. Please pass in a PrepareCustomConfig instead.")
        prepare_custom_config = PrepareCustomConfig.from_dict(prepare_custom_config)

    if isinstance(backend_config, Dict):
        warnings.warn(
            "Passing a backend_config_dict to prepare is deprecated and will not be supported "
            "in a future version. Please pass in a BackendConfig instead.")
        backend_config = BackendConfig.from_dict(backend_config)

    assert(isinstance(qconfig_mapping, QConfigMapping))
    assert(isinstance(_equalization_config, QConfigMapping))
    qconfig_mapping = copy.deepcopy(qconfig_mapping)
    _equalization_config = copy.deepcopy(_equalization_config)

    # mapping from a tuple of nodes in reverse order to uninitialized
    #   QuantizeHandler subclass. For example,
    # {
    #   # match a single node
    #   (<class 'torch.nn.modules.conv.Conv3d'>:
    #     <class 'torch.ao.quantization.fx.quantize.ConvRelu'>),
    #   # match multiple nodes in reverse order
    #   ((<function relu at 0x7f766a7360d0>, <built-in function add>):
    #     <class 'torch.ao.quantization.fx.quantize.Add'>),
    # }

    pattern_to_quantize_handler: Dict[Pattern, QuantizeHandler] = {}
    if backend_config is None:
        backend_config = get_native_backend_config()
    pattern_to_quantize_handler = _get_pattern_to_quantize_handlers(backend_config)
    pattern_to_quantize_handler = _sorted_patterns_dict(pattern_to_quantize_handler)

    root_node_getter_mapping = \
        get_fusion_pattern_to_root_node_getter(backend_config)

    _update_qconfig_for_fusion(model, qconfig_mapping)
    _update_qconfig_for_fusion(model, _equalization_config)
    flattened_qconfig_dict = _get_flattened_qconfig_dict(qconfig_mapping)
    # TODO: support regex as well
    propagate_qconfig_(model, flattened_qconfig_dict, prepare_custom_config.to_dict())

    if is_qat:
        module_to_qat_module = get_module_to_qat_module(backend_config)
        _qat_swap_modules(model, module_to_qat_module)
        _update_qconfig_for_qat(qconfig_mapping, backend_config)

    # mapping from fully qualified module name to module instance
    # for example,
    # {
    #   '': Model(...),
    #   'linear': Linear(...),
    #   'linear.weight_fake_quant': PerChannelMinMaxObserver(...),
    # }
    named_modules = dict(model.named_modules(remove_duplicate=False))

    # fill node_name_to_qconfig, a map from node name to qconfig, used in _find_matches
    equalization_node_name_to_qconfig = _generate_node_name_to_qconfig(
        model, named_modules, model.graph, _equalization_config, node_name_to_scope)
    node_name_to_qconfig = _generate_node_name_to_qconfig(model, named_modules, model.graph, qconfig_mapping, node_name_to_scope)

    # match the patterns that will get quantized
    standalone_module_names = list(prepare_custom_config.standalone_module_names.keys())
    standalone_module_classes = list(prepare_custom_config.standalone_module_classes.keys())

    custom_module_classes = get_custom_module_class_keys(prepare_custom_config.float_to_observed_mapping)
    matches_without_qconfig = _find_matches(
        model.graph, named_modules, pattern_to_quantize_handler, root_node_getter_mapping,
        standalone_module_names, standalone_module_classes, custom_module_classes)

    # map qconfig instances to matches
    node_name_to_match_result_with_qconfig = {}
    for node_name, match_without_qconfig in matches_without_qconfig.items():
        match_with_qconfig = (*match_without_qconfig, node_name_to_qconfig[node_name])
        node_name_to_match_result_with_qconfig[node_name] = match_with_qconfig

    _run_prepare_fx_on_standalone_modules(
        model, is_qat, named_modules, node_name_to_match_result_with_qconfig, prepare_custom_config, backend_config)

    # record names for the set of observed node, so that in convert step
    # we know whether we need to convert a floating point module to reference
    # quantized module or not
    observed_node_names: Set[str] = set()

    result_node = insert_observers_for_model(
        model,
        node_name_to_match_result_with_qconfig,
        node_name_to_qconfig,
        prepare_custom_config,
        equalization_node_name_to_qconfig,
        backend_config,
        observed_node_names,
        is_qat
    )
    model = GraphModule(model, model.graph)

    _save_state(model, node_name_to_qconfig, node_name_to_scope,
                prepare_custom_config, equalization_node_name_to_qconfig,
                qconfig_mapping, is_qat, observed_node_names)

    if is_standalone_module:
        assert result_node is not None
        assert isinstance(result_node.args[0], Node), \
            "standalone module only supports returning simple value currently"\
            "(not tuple, dict etc.)"
        # these inputs are observed in parent
        # converting List[int] to Tensor since module attribute is
        # Union[Tensor, Module]
        input_quantized_idxs: List[int] = prepare_custom_config.input_quantized_indexes
        output_quantized_idxs: List[int] = prepare_custom_config.output_quantized_indexes
        observed_graph_module_attrs = model.meta["_observed_graph_module_attrs"]
        # inplace modification
        observed_graph_module_attrs.is_observed_standalone_module = True
        observed_graph_module_attrs.standalone_module_input_quantized_idxs = \
            input_quantized_idxs
        observed_graph_module_attrs.standalone_module_output_quantized_idxs = \
            output_quantized_idxs
    return model
