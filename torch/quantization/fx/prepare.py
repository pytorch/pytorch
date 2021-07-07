import torch
import operator
from torch.fx import (
    GraphModule,
)

from torch.quantization import (
    propagate_qconfig_,
)
from torch.fx.graph import (
    Graph,
    Node,
)
from torch.fx.node import Argument

from .qconfig_utils import (
    convert_dict_to_ordered_dict,
    generate_qconfig_map,
    get_flattened_qconfig_dict,
    QConfigAny,
)

from .quantization_patterns import (
    QuantizeHandler,
    CatQuantizeHandler,
    CopyNodeQuantizeHandler,
    CustomModuleQuantizeHandler,
    StandaloneModuleQuantizeHandler,
)

from .quantization_types import Pattern

from ._equalize import (
    is_equalization_observer,
    node_supports_equalization,
)

from .graph_module import (
    ObservedGraphModule,
    ObservedStandaloneGraphModule,
)

from .pattern_utils import (
    MatchResult,
    get_default_quant_patterns,
    get_default_output_activation_post_process_map,
)

from .match_utils import (
    find_matches,
)

from .utils import (
    _parent_name,
    get_custom_module_class_keys,
    all_node_args_have_no_tensors,
    assert_and_get_unique_device,
    node_bool_tensor_arg_indexes,
    get_new_attr_name_with_prefix,
    NON_QUANTIZABLE_WEIGHT_OPS,
    WEIGHT_INDEX_DICT,
    FUNCTIONAL_OPS_WITH_BIAS,
)

from ..fuser_method_mappings import DEFAULT_OP_LIST_TO_FUSER_METHOD

from ..quantization_mappings import (
    get_default_qat_module_mappings,
)

from ..quantize import (
    is_activation_post_process,
    convert
)

from ..utils import (
    get_combined_dict,
    get_qconfig_dtypes,
    get_swapped_custom_module_class,
    weight_is_quantized,
    activation_is_statically_quantized,
    activation_is_int8_quantized,
    activation_dtype,
    weight_dtype,
)

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

def is_activation_post_process_node(node: Node, modules: Dict[str, torch.nn.Module]) -> bool:
    return node.op == "call_module" and \
        is_activation_post_process(modules[str(node.target)])

def node_arg_is_weight(node: Node, arg: Any) -> bool:
    if isinstance(node, Node) and node.op == 'call_function' and \
            node.target in WEIGHT_INDEX_DICT:
        for i, node_arg in enumerate(node.args):
            if arg is node_arg and i in \
                    WEIGHT_INDEX_DICT[node.target]:  # type: ignore[index]
                return True
        for kwarg_name, kwarg_value in node.kwargs.items():
            if kwarg_name == 'weight' and arg is kwarg_value:
                return True
    return False

CONV_OPS_WITH_BIAS = {
    torch.nn.functional.conv1d,
    torch.nn.functional.conv2d,
    torch.nn.functional.conv3d,
}
CONV_BIAS_ARG_INDEX = 2

def node_arg_is_bias(node: Node, arg: Any) -> bool:
    if isinstance(node, Node) and node.op == 'call_function':
        if node.target in CONV_OPS_WITH_BIAS:
            for i, node_arg in enumerate(node.args):
                if arg is node_arg and i == CONV_BIAS_ARG_INDEX:
                    return True
        elif node.target in FUNCTIONAL_OPS_WITH_BIAS:
            for kwarg_name, kwarg_value in node.kwargs.items():
                if kwarg_name == 'bias' and arg is kwarg_value:
                    return True
    return False

def get_standalone_module_configs(
    node: Node,
    modules: Dict[str, torch.nn.Module],
    prepare_custom_config_dict: Dict[str, Any],
    qconfig: QConfigAny,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Returns the standalone module qconfig_dict and prepare_config_dict
    for `node`, assuming that the module pointed to by `node` is
    a standalone modules.
    """
    standalone_module = modules[node.target]  # type: ignore[index]
    standalone_module_name_configs = \
        prepare_custom_config_dict.get("standalone_module_name", [])
    standalone_module_class_configs = \
        prepare_custom_config_dict.get("standalone_module_class", [])
    class_config_map = {x[0]: (x[1], x[2]) for x in standalone_module_class_configs}
    name_config_map = {x[0]: (x[1], x[2]) for x in standalone_module_name_configs}
    config = class_config_map.get(type(standalone_module), (None, None))
    config = name_config_map.get(node.target, config)
    sm_qconfig_dict = {"": qconfig} if config[0] is None else config[0]
    sm_prepare_config_dict = {} if config[1] is None else config[1]
    return sm_qconfig_dict, sm_prepare_config_dict

def qat_swap_modules(
        root: torch.nn.Module,
        additional_qat_module_mapping: Dict[Callable, Callable]) -> None:
    all_mappings = get_combined_dict(
        get_default_qat_module_mappings(), additional_qat_module_mapping)
    convert(root, mapping=all_mappings, inplace=True, remove_qconfig=False)

def update_qconfig_for_qat(
    qconfig_dict: Any,
    additional_qat_module_mapping: Dict[Callable, Callable]
) -> Any:
    """
    Update the qconfig_dict to account for module swaps during QAT.
    During QAT we perform a module swap on the nn.Module types to the corresponding nn.qat.modules types.
    """
    all_qat_mappings = get_combined_dict(
        get_default_qat_module_mappings(), additional_qat_module_mapping)
    object_type_dict = qconfig_dict.get("object_type", None)
    for k, v in object_type_dict.items():
        if k in all_qat_mappings:
            object_type_dict[all_qat_mappings[k]] = v
    return qconfig_dict

def update_qconfig_for_fusion(
    model: GraphModule,
    qconfig_dict: Any,
) -> Any:
    """
    Update the qconfig_dict to account for fused modules such as LinearReLU.
    """
    object_type_dict = qconfig_dict.get("object_type", None)
    if object_type_dict is None:
        return qconfig_dict

    modules = dict(model.named_modules())

    for node in model.graph.nodes:
        if node.op == 'call_module':
            module_type = type(modules[str(node.target)])
            if module_type not in list(DEFAULT_OP_LIST_TO_FUSER_METHOD.values()):
                continue

            for ops, fuser in DEFAULT_OP_LIST_TO_FUSER_METHOD.items():
                if module_type == fuser:
                    fused_qconfig = object_type_dict.get(ops[0], None)

                    # Raise an error if the modules in the fused module have
                    # different qconfigs specified in the qconfig_dict
                    for op in ops:
                        if object_type_dict.get(op, None) != fused_qconfig:
                            raise LookupError("During fusion, we need to specify the same " +
                                              f"qconfigs for both modules in {module_type}.")

                    if fused_qconfig is not None:
                        object_type_dict[module_type] = fused_qconfig

    return qconfig_dict

def insert_observer(
    node: Node,
    observer: torch.quantization.ObserverBase,
    model: torch.nn.Module,
    modules: Dict[str, torch.nn.Module],
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
        prefix = node.name + '_activation_post_process_'
    get_new_observer_name = get_new_attr_name_with_prefix(prefix)
    observer_name = get_new_observer_name(model)
    setattr(model, observer_name, observer)
    modules[observer_name] = observer
    with graph.inserting_after(node):
        new_obs = graph.create_node(
            'call_module', observer_name, (node,), {})
    return new_obs

def get_target_activation_dtype_for_node(
    node: Node,
    qconfig: QConfigAny,
    inputs_seen_counter: int,
    outputs_seen_counter: int,
    input_quantized_idxs: List[int],
    output_quantized_idxs: List[int],
    qhandler: Optional[QuantizeHandler],
    modules: Dict[str, torch.nn.Module],
    cache_for_no_tensor_check: Dict[Node, bool],
) -> Optional[torch.dtype]:
    """
    Returns the expected dtype of the input and output of this node after
    convert. If the value is not None, it represents the dtype of the
    Tensor. If the value is None, it means the value is not a Tensor.

    Note: this is for activations only, weight dtypes are not handled here.

    TODO(future PR, if needed): explicitly spell out the non-Tensor
    dtypes.
    """
    if node.op == 'placeholder':
        if inputs_seen_counter in input_quantized_idxs:
            return torch.quint8
        else:
            # if dtype is fp32 (default), do nothing
            # note: other dtypes are not supported
            return torch.float

    elif node.op in ('call_module', 'call_method', 'call_function'):
        args_have_no_tensors = \
            all_node_args_have_no_tensors(
                node, modules, cache_for_no_tensor_check)
        if args_have_no_tensors:
            return None

        # TODO(future PR): consider stopping matching getitem
        is_getitem = node.op == 'call_function' and \
            node.target == operator.getitem
        if is_getitem:
            return torch.float

        # get qconfig to determine the eventual dtype of this node
        if qconfig is not None:
            if qhandler is not None and qhandler.input_output_observed():
                act_dtype, weight_dtype, act_compute_dtype = \
                    get_qconfig_dtypes(qconfig)
                return act_dtype
            else:
                return torch.float
        else:
            return torch.float

    elif node.op == 'get_attr':
        return torch.float

    elif node.op == 'output':
        if outputs_seen_counter in output_quantized_idxs:
            return torch.quint8
        else:
            # if dtype is fp32 (default), do nothing
            # note: other dtypes are not supported
            return torch.float

    else:
        raise AssertionError(f'need to handle {node.format_node()}')

def maybe_insert_input_observer_for_arg_or_kwarg(
    node: Union[Node, Any],
    arg: Argument,
    qconfig: QConfigAny,
    model: torch.nn.Module,
    modules: Dict[str, torch.nn.Module],
    graph: Graph,
    node_name_to_target_dtype: Dict[str, Any],
    qhandler: Optional[QuantizeHandler],
    prepare_custom_config_dict: Dict[str, Any],
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
            new_inner_arg = maybe_insert_input_observer_for_arg_or_kwarg(
                node, inner_arg, qconfig, model, modules,
                graph, node_name_to_target_dtype,
                qhandler, prepare_custom_config_dict)
            new_arg_to_return.append(new_inner_arg)
        return new_arg_to_return

    if not isinstance(arg, Node):
        return arg
    assert isinstance(arg, Node)

    # default (no observer)
    new_arg = arg

    is_standalone_module = qhandler is not None and \
        isinstance(qhandler, StandaloneModuleQuantizeHandler)

    if not is_standalone_module:
        # regular flow for most nodes, except standalone modules
        is_weight = node_arg_is_weight(node, arg)
        assert qconfig is not None

        act_post_process_ctr = qconfig.weight if is_weight else \
            qconfig.activation

        is_bias = node_arg_is_bias(node, arg)
        is_activation = not (is_weight or is_bias)
        weight_needs_obs = is_weight and weight_is_quantized(qconfig) and node.target not in NON_QUANTIZABLE_WEIGHT_OPS
        bias_needs_obs = \
            (is_bias and activation_dtype(qconfig) == torch.float16) and \
            weight_dtype(qconfig) == torch.float16

        arg_dtype = node_name_to_target_dtype[arg.name]
        node_dtype = node_name_to_target_dtype[node.name]
        dtype_changes_and_second_dtype_not_float = (
            # if the dtypes are different, we need an observer
            (arg_dtype != node_dtype) and
            # except if the second dtype is float, a dequant will be inserted
            # without an observer in convert
            # TODO(future PR): change this so a placeholder is inserted for
            # future dequants, to make the logic easier to understand
            (node_dtype != torch.float) and
            # if arg is a bool tensor or not a tensor, do not insert observer
            (arg_dtype not in (torch.bool, None)) and
            (is_activation and activation_is_statically_quantized(qconfig))
        )

        needs_obs = (
            weight_needs_obs or
            bias_needs_obs or
            dtype_changes_and_second_dtype_not_float
        )

    else:
        # custom flow for standalone modules
        _sm_qconfig_dict, sm_prepare_config_dict = \
            get_standalone_module_configs(
                node, modules, prepare_custom_config_dict, qconfig)

        sm_input_quantized_idxs = \
            sm_prepare_config_dict.get('input_quantized_idxs', [])
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
            arg_dtype = node_name_to_target_dtype[arg.name]
            node_dtype = torch.quint8 if cur_input_idx in sm_input_quantized_idxs \
                else torch.float
            needs_obs = (
                (arg_dtype != node_dtype) and
                (node_dtype != torch.float)
            )

    if needs_obs:

        new_obs_mod = act_post_process_ctr()
        existing_obs_node = None

        # Before using the new observer, check if an observer
        # of the correct type already exists. If it does, use it.
        # This prevents duplicate observer insertions if a node is
        # used by multiple nodes.
        for maybe_obs_node, _ in arg.users.items():
            if maybe_obs_node.op == 'call_module':
                maybe_obs_mod = modules[maybe_obs_node.target]  # type: ignore[index]
                if (
                    type(maybe_obs_mod) == type(new_obs_mod) and
                    node_name_to_target_dtype[maybe_obs_node.name] == node_dtype
                ):
                    existing_obs_node = maybe_obs_node
                    break

        if existing_obs_node is None:
            new_obs_node = insert_observer(
                arg, new_obs_mod, model, modules, graph)
            # set the type, so the next node can read it
            node_name_to_target_dtype[new_obs_node.name] = node_dtype
            # override this arg to be the observed arg
            new_arg = new_obs_node
        else:
            new_arg = existing_obs_node

    return new_arg


def maybe_insert_input_observers_for_node(
    node: Node,
    qconfig: QConfigAny,
    model: torch.nn.Module,
    modules: Dict[str, torch.nn.Module],
    graph: Graph,
    node_name_to_target_dtype: Dict[str, Any],
    qhandler: Optional[QuantizeHandler],
    prepare_custom_config_dict: Dict[str, Any],
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
        new_arg = maybe_insert_input_observer_for_arg_or_kwarg(
            node, arg, qconfig, model, modules, graph,
            node_name_to_target_dtype,
            qhandler, prepare_custom_config_dict)
        new_args.append(new_arg)

    new_kwargs = {}
    for k, kwarg in node.kwargs.items():
        new_kwarg = maybe_insert_input_observer_for_arg_or_kwarg(
            node, kwarg, qconfig, model, modules, graph,
            node_name_to_target_dtype,
            qhandler, prepare_custom_config_dict)
        new_kwargs[k] = new_kwarg

    # assign the new args and kwargs to the node, inplace
    node.args = tuple(new_args)
    node.kwargs = new_kwargs

def maybe_insert_input_equalization_observers_for_node(
    node: Node,
    equalization_qconfig: Any,
    model: torch.nn.Module,
    modules: Dict[str, torch.nn.Module],
    graph: Graph,
    node_name_to_target_dtype: Dict[str, Any],
) -> None:
    """
    If `node` needs to be equalized, find the input/weight observers it needs in
    `equalization_qconfig`, creates them, and inserts it into `graph`.

    If `node` does not need an equalization observer, returns None.
    """
    if equalization_qconfig is None or not node_supports_equalization(node, modules):
        return

    new_args = []
    for arg in node.args:
        if not isinstance(arg, Node) or node_arg_is_bias(node, arg):
            new_args.append(arg)
            continue

        is_weight = node_arg_is_weight(node, arg)

        act_eq_process_ctr = equalization_qconfig.weight if is_weight else \
            equalization_qconfig.input_activation

        new_eq_obs_mod = act_eq_process_ctr()
        new_eq_obs_node = insert_observer(
            arg, new_eq_obs_mod, model, modules, graph)

        # set the type, so the next node can read it
        node_name_to_target_dtype[new_eq_obs_node.name] = node_name_to_target_dtype[arg.name]

        new_args.append(new_eq_obs_node)

    # assign the new args and kwargs to the node, inplace
    node.args = tuple(new_args)

def maybe_insert_output_observer_for_node(
    node: Node,
    model: torch.nn.Module,
    modules: Dict[str, torch.nn.Module],
    graph: Graph,
    matches: Dict[str, MatchResult],
    node_name_to_target_dtype: Dict[str, Any],
    matched_pattern: Any,
    qhandler: Optional[QuantizeHandler],
) -> Optional[Node]:
    """
    If `node` needs an output observer, creates it, inserts it into `graph`
    and returns it.

    If `node` does not need an output observer, returns None.
    """
    root_node, matched_nodes, pattern, qhandler, qconfig = matches.get(
        node.name, (None, None, None, None, None))

    if qhandler is None:
        return None

    assert qconfig is not None
    assert node.op != 'output', 'observer insertion for outputs is handled elsewhere'

    is_standalone_module = qhandler is not None and \
        isinstance(qhandler, StandaloneModuleQuantizeHandler)

    should_insert_observer = \
        qhandler.should_insert_observer_for_output(
            qconfig, model.training)
    # TODO(future PR): move the following logic to
    # should_insert_observer_for_output
    should_insert_observer = should_insert_observer and \
        activation_is_statically_quantized(qconfig)

    # we never insert observers to output of standalone module, we assume
    # if needed, they are inserted inside the standalone module
    should_insert_observer = should_insert_observer and \
        (not is_standalone_module)

    if should_insert_observer:
        act_post_process_ctr = qconfig.activation
        if activation_is_int8_quantized(qconfig):
            act_post_process_ctr = \
                get_default_output_activation_post_process_map().get(
                    matched_pattern,
                    act_post_process_ctr)
        observer = act_post_process_ctr()
        new_obs = insert_observer(node, observer, model, modules, graph)
        # set the type, so the next node can read it
        node_name_to_target_dtype[new_obs.name] = \
            node_name_to_target_dtype[node.name]
        return new_obs
    else:
        return None

def maybe_insert_observers_before_graph_output(
    graph_output_node: Node,
    output_quantized_idxs: List[int],
    node_name_to_target_dtype: Dict[str, torch.dtype],
    qconfig_map: Dict[str, QConfigAny],
    model: torch.nn.Module,
    modules: Dict[str, torch.nn.Module],
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
        node_name_to_target_dtype: Dict[str, torch.dtype],
        qconfig_map: Dict[str, QConfigAny],
        model: torch.nn.Module,
        modules: Dict[str, torch.nn.Module],
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
            this_node_dtype = node_name_to_target_dtype[maybe_node.name]
            if this_node_dtype != target_dtype:
                # insert observer
                qconfig = qconfig_map.get(maybe_node.name)
                # TODO(future PR): see if we need to allow specifying qconfig
                #   on output nodes, to remove the restriction below.
                assert qconfig is not None, \
                    'Quantizing the output node without a qconfig is not supported'
                observer_mod = qconfig.activation()
                observer_node = insert_observer(
                    maybe_node, observer_mod, model, modules, graph)
                return observer_node
            else:
                return maybe_node
        elif isinstance(maybe_node, (list, tuple)):
            results = []
            for inner_node in maybe_node:
                results.append(_recursive_maybe_replace_node_with_obs(
                    inner_node, target_dtype, node_name_to_target_dtype,
                    qconfig_map, model, modules, graph))
            if isinstance(maybe_node, list):
                return results
            else:
                return tuple(results)
        elif isinstance(maybe_node, dict):
            results_dict = {}
            for k, inner_v in maybe_node.items():
                results_dict[k] = _recursive_maybe_replace_node_with_obs(
                    inner_v, target_dtype, node_name_to_target_dtype,
                    qconfig_map, model, modules, graph)
            return results_dict
        else:
            return results

    new_args = []
    for old_arg in graph_output_node.args:
        new_args.append(
            _recursive_maybe_replace_node_with_obs(
                old_arg, output_target_dtype, node_name_to_target_dtype,
                qconfig_map, model, modules, graph))

    graph_output_node.args = new_args  # type: ignore[assignment]


def maybe_propagate_dtype_for_node(
    node: Node,
    target_dtype: torch.dtype,
    node_name_to_target_dtype: Dict[str, torch.dtype],
    matches: Dict[str, MatchResult],
) -> None:
    """
    Assigns `target_dtype` to `node`. If `node` is matched to an instance
    of `CopyNodeQuantizeHandler`, also call this function recursively on
    the first argument, to propagate the dtype to the caller.
    """
    node_name_to_target_dtype[node.name] = target_dtype
    # if this is a copy node, propagate to first arg
    root_node, matched_nodes, pattern, qhandler, qconfig = matches.get(
        node.name, (None, None, None, None, None))
    if isinstance(qhandler, CopyNodeQuantizeHandler):
        prev_node = node.args[0]
        if isinstance(prev_node, Node):
            maybe_propagate_dtype_for_node(
                prev_node, target_dtype, node_name_to_target_dtype, matches)

def propagate_dtypes_for_known_nodes(
    graph: Graph,
    node_name_to_target_dtype: Dict[str, torch.dtype],
    matches: Dict[str, MatchResult],
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
        bool_arg_idxs = node_bool_tensor_arg_indexes(node)
        for bool_arg_idx in bool_arg_idxs:
            cur_node = node.args[bool_arg_idx]
            maybe_propagate_dtype_for_node(
                cur_node, torch.bool, node_name_to_target_dtype, matches)

def adjust_observers_for_cat(
    node: Node,
    model: torch.nn.Module,
    modules: Dict[str, torch.nn.Module],
) -> None:
    """
    Ensures that for quantized `torch.cat` nodes, we share an observer
    for all input arguments as well as the output argument. In detail, given
    a graph of

      x0 -> obs0 -> cat -> x2
                  /
      x1 -> obs1 /

    where node obs0 points to observer instance observer0,
    obs1 points to observer1 and obs2 points to observer2, we make nodes obs1
    and ob2 point to observer0.
    """
    # find the observer module to use
    first_arg = node.args[0]
    assert isinstance(first_arg, (list, tuple))
    first_arg_arg = first_arg[0]

    # if we have a graph such as
    #   observed_node -> non_observed_node -> cat
    # we need to navigate up to the first observer
    iteration_guard = 0
    while not is_activation_post_process_node(first_arg_arg, modules):
        first_arg_arg = first_arg_arg.args[0]
        iteration_guard += 1
        if iteration_guard > 10000:
            raise AssertionError('Unable to find observer of previous node')

    assert isinstance(first_arg_arg, Node)
    target_to_use = first_arg_arg.target
    assert isinstance(target_to_use, str)
    obs_mod_to_use = modules[target_to_use]

    # set all other input observer nodes to use that module
    for input_idx, input_arg in enumerate(first_arg):
        if input_idx == 0:
            continue
        iteration_guard = 0
        while not is_activation_post_process_node(input_arg, modules):
            input_arg = input_arg.args[0]
            iteration_guard += 1
            if iteration_guard > 10000:
                raise AssertionError('Unable to find observer of previous node')

        parent_name, name = _parent_name(input_arg.target)
        setattr(modules[parent_name], name, obs_mod_to_use)

    # set the output observer node to use that module
    for output_obs_node, _ in node.users.items():
        assert is_activation_post_process_node(output_obs_node, modules)
        parent_name, name = _parent_name(output_obs_node.target)
        setattr(modules[parent_name], name, obs_mod_to_use)

    # TODO(future PR): delete the orphaned observer modules

def swap_custom_module_to_observed(
        node: Node,
        qconfig: QConfigAny,
        modules: Dict[str, torch.nn.Module],
        prepare_custom_config_dict: Dict[str, Any]):
    custom_module = modules[node.target]  # type: ignore[index]
    custom_module_class_mapping = prepare_custom_config_dict.get(
        "float_to_observed_custom_module_class", {})
    observed_custom_module_class = \
        get_swapped_custom_module_class(
            custom_module, custom_module_class_mapping, qconfig)
    observed_custom_module = \
        observed_custom_module_class.from_float(custom_module)
    parent_name, name = _parent_name(node.target)
    setattr(modules[parent_name], name, observed_custom_module)

def insert_observers_for_model(
    model: GraphModule,
    modules: Dict[str, torch.nn.Module],
    matches: Dict[str, MatchResult],
    qconfig_map: Dict[str, QConfigAny],
    graph: Graph,
    prepare_custom_config_dict: Dict[str, Any],
    equalization_config_map: Dict[str, Any],
    input_quantized_idxs: List[int],
    output_quantized_idxs: List[int],
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

    node_name_to_target_dtype: Dict[str, Any] = {}
    cache_for_no_tensor_check: Dict[Node, bool] = dict()

    inputs_seen_counter = 0
    outputs_seen_counter = 0
    results_node = None

    # first, populate the dtype map based only on qconfig and qhandler
    # this assumes:
    # graph inputs are fp32 by default, and int8 where overriden
    # other nodes output dtype is specified by the qconfig
    modules = dict(model.named_modules(remove_duplicate=False))
    for node in model.graph.nodes:
        root_node, matched_nodes, pattern, qhandler, qconfig = matches.get(
            node.name, (None, None, None, None, None))
        node_name_to_target_dtype[node.name] = get_target_activation_dtype_for_node(
            node, qconfig, inputs_seen_counter, outputs_seen_counter,
            input_quantized_idxs, output_quantized_idxs, qhandler,
            modules, cache_for_no_tensor_check)

    # Second, for nodes with known input dtypes, propagate them throughout the
    # graph. For example, if there is a call such as
    #   x1 = x0.masked_fill(mask, 1)
    # we propagate the type of mask to be torch.bool
    propagate_dtypes_for_known_nodes(
        model.graph, node_name_to_target_dtype, matches)

    # After this point, the current node and all of its arguments
    # have a dtype assigned. Now, we insert observers for inputs
    # of this node (if needed for this node), and the output of this node
    # (if needed for this node).

    # Since we are mutating the graph as we go, we iterate over the original
    # nodes before observer insertion, instead of model.graph.nodes.
    nodes_before_observation = list(model.graph.nodes)

    for node in nodes_before_observation:

        if node.op == 'placeholder':
            # if a graph input is in fp32, it does not need observation
            # if a graph input is in int8, we assume the observation happens
            #   outside of the graph, and no additional observation is needed
            pass

        elif node.op in ('call_module', 'call_method', 'call_function', 'output'):
            # check for matches
            root_node, matched_nodes, pattern, qhandler, qconfig = matches.get(
                node.name, (None, None, None, None, None))
            equalization_qconfig = equalization_config_map.get(node.name, None)

            this_node_dtype = node_name_to_target_dtype[node.name]
            output_not_a_tensor = this_node_dtype is None
            # TODO(future PR): consider stopping matching getitem
            is_getitem = node.op == 'call_function' and \
                node.target == operator.getitem

            skip_inserting_observers = (
                (qconfig is None) or
                output_not_a_tensor or
                is_getitem
            ) and (not node.op == 'output')

            if not skip_inserting_observers:
                modules = dict(model.named_modules(remove_duplicate=False))
                if node.op != 'output':
                    # this modifies node inplace
                    maybe_insert_input_observers_for_node(
                        node, qconfig, model, modules, graph,
                        node_name_to_target_dtype,
                        qhandler, prepare_custom_config_dict)

                    # Insert equalization input observers if needed
                    maybe_insert_input_equalization_observers_for_node(
                        node, equalization_qconfig, model, modules, graph,
                        node_name_to_target_dtype)

                    is_last_node_of_pattern = root_node is node
                    is_like_copy_node = \
                        (qhandler is not None and (
                            isinstance(qhandler, CopyNodeQuantizeHandler)
                        ))
                    if is_last_node_of_pattern and (not is_like_copy_node):
                        # this returns the new observer node if it was needed
                        maybe_output_obs_node = maybe_insert_output_observer_for_node(
                            node, model, modules, graph, matches,
                            node_name_to_target_dtype, pattern, qhandler)
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

                            # for quantized cat nodes only, we modify the graph
                            # to make all inputs and outputs use the first input's
                            # observer
                            if isinstance(qhandler, CatQuantizeHandler):
                                adjust_observers_for_cat(node, model, modules)

                            if isinstance(qhandler, CustomModuleQuantizeHandler):
                                swap_custom_module_to_observed(node, qconfig, modules, prepare_custom_config_dict)

                else:  # output
                    maybe_insert_observers_before_graph_output(
                        node, output_quantized_idxs,
                        node_name_to_target_dtype, qconfig_map,
                        model, modules, graph)

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

def run_prepare_fx_on_standalone_modules(
    model: torch.nn.Module,
    modules: Dict[str, torch.nn.Module],
    matches: Any,
    prepare_custom_config_dict: Dict[str, Any],
) -> None:
    """
    Runs prepare_fx on each standalone module. Note: this does
    not modify the graph, it just replaces the unobserved modules with
    their observed versions.
    """
    for (
        node_name,
        (root_node, matched_nodes, pattern, qhandler, qconfig),
    ) in matches.items():
        if qhandler is None:
            continue
        elif not isinstance(qhandler, StandaloneModuleQuantizeHandler):
            continue

        sm_qconfig_dict, sm_prepare_config_dict = \
            get_standalone_module_configs(
                root_node, modules, prepare_custom_config_dict, qconfig)

        standalone_module = modules[root_node.target]
        prepare = \
            torch.quantization.quantize_fx._prepare_standalone_module_fx  # type: ignore[attr-defined]
        observed_standalone_module = \
            prepare(standalone_module, sm_qconfig_dict, sm_prepare_config_dict)
        preserved_attributes = \
            set(sm_prepare_config_dict.get("preserved_attributes", []))
        observed_standalone_module = ObservedStandaloneGraphModule(
            observed_standalone_module, observed_standalone_module.graph,
            preserved_attributes)
        parent_name, name = _parent_name(root_node.target)
        setattr(modules[parent_name], name,
                observed_standalone_module)
        modules[root_node.target] = observed_standalone_module

def save_state(
    observed: GraphModule,
    qconfig_map: Dict[str, QConfigAny],
    node_name_to_scope: Dict[str, Tuple[str, type]],
    patterns: Dict[Pattern, QuantizeHandler],
    prepare_custom_config_dict: Dict[str, Any],
    equalization_qconfig_map: Dict[str, Any],
) -> None:
    observed._patterns = patterns  # type: ignore[assignment]
    observed._qconfig_map = qconfig_map  # type: ignore[assignment]
    observed._prepare_custom_config_dict = \
        prepare_custom_config_dict  # type: ignore[assignment]
    observed._node_name_to_scope = node_name_to_scope  # type: ignore[assignment]
    observed._equalization_qconfig_map = equalization_qconfig_map  # type: ignore[assignment]

def prepare(
        model: GraphModule,
        qconfig_dict: Any,
        node_name_to_scope: Dict[str, Tuple[str, type]],
        prepare_custom_config_dict: Optional[Dict[str, Any]] = None,
        equalization_qconfig_dict: Optional[Dict[str, Any]] = None,
        is_standalone_module: bool = False) -> ObservedGraphModule:
    """ standalone_module means it a submodule that is not inlined in
    parent module, and will be quantized separately as one unit.

    How the standalone module is observed is specified by `input_quantized_idxs` and
    `output_quantized_idxs` in the prepare_custom_config for the standalone module
    Args:
        node_name_to_scope: mapping from node name to the scope of the module which contains the node.
        The scope is a tuple of fully qualified path of the module and the type of the module
    Returns:
        model(GraphModule): prepared standalone module
        attributes:
            _standalone_module_input_quantized_idxs(List[Int]): a list of
                indexes for the graph input that is expected to be quantized,
                same as input_quantized_idxs configuration provided
                for the standalone module
            _standalone_module_output_quantized_idxs(List[Int]): a list of
                indexs for the graph output that is quantized
                same as input_quantized_idxs configuration provided
                for the standalone module
    """
    if prepare_custom_config_dict is None:
        prepare_custom_config_dict = {}
    if equalization_qconfig_dict is None:
        equalization_qconfig_dict = {}

    additional_quant_patterns = \
        prepare_custom_config_dict.get("additional_quant_pattern", {})
    # mapping from a tuple of nodes in reverse order to uninitialized
    #   QuantizeHandler subclass. For example,
    # {
    #   # match a single node
    #   (<class 'torch.nn.modules.conv.Conv3d'>:
    #     <class 'torch.quantization.fx.quantize.ConvRelu'>),
    #   # match multiple nodes in reverse order
    #   ((<function relu at 0x7f766a7360d0>, <built-in function add>):
    #     <class 'torch.quantization.fx.quantize.Add'>),
    # }
    patterns: Dict[Pattern, QuantizeHandler] = get_combined_dict(
        get_default_quant_patterns(), additional_quant_patterns)

    convert_dict_to_ordered_dict(qconfig_dict)
    convert_dict_to_ordered_dict(equalization_qconfig_dict)
    flattened_qconfig_dict = get_flattened_qconfig_dict(qconfig_dict)
    # TODO: support regex as well
    propagate_qconfig_(model, flattened_qconfig_dict)

    if model.training:
        additional_qat_module_mapping = prepare_custom_config_dict.get(
            "additional_qat_module_mapping", {})
        qat_swap_modules(model, additional_qat_module_mapping)
        qconfig_dict = update_qconfig_for_qat(qconfig_dict, additional_qat_module_mapping)

    qconfig_dict = update_qconfig_for_fusion(model, qconfig_dict)
    equalization_qconfig_dict = update_qconfig_for_fusion(model, equalization_qconfig_dict)

    # mapping from fully qualified module name to module instance
    # for example,
    # {
    #   '': Model(...),
    #   'linear': Linear(...),
    #   'linear.weight_fake_quant': PerChannelMinMaxObserver(...),
    # }
    modules = dict(model.named_modules())

    # fill qconfig_map, a map from node name to qconfig, used in find_matches
    equalization_qconfig_map = generate_qconfig_map(model, modules, model.graph, equalization_qconfig_dict, node_name_to_scope)
    qconfig_map = generate_qconfig_map(model, modules, model.graph, qconfig_dict, node_name_to_scope)

    # match the patterns that will get quantized
    standalone_module_name_configs = prepare_custom_config_dict.get(
        "standalone_module_name", [])
    standalone_module_class_configs = prepare_custom_config_dict.get(
        "standalone_module_class", [])

    standalone_module_names = [config[0] for config in standalone_module_name_configs]
    standalone_module_classes = [config[0] for config in standalone_module_class_configs]
    custom_module_classes = get_custom_module_class_keys(
        prepare_custom_config_dict, "float_to_observed_custom_module_class")
    matches = find_matches(
        model.graph, modules, patterns, qconfig_map, standalone_module_names,
        standalone_module_classes, custom_module_classes)

    input_quantized_idxs: List[int] = prepare_custom_config_dict.get(
        "input_quantized_idxs", [])
    output_quantized_idxs: List[int] = prepare_custom_config_dict.get(
        "output_quantized_idxs", [])

    run_prepare_fx_on_standalone_modules(
        model, modules, matches, prepare_custom_config_dict)

    result_node = insert_observers_for_model(
        model, modules, matches, qconfig_map,
        model.graph, prepare_custom_config_dict,
        equalization_qconfig_map,
        input_quantized_idxs, output_quantized_idxs)

    save_state(model, qconfig_map, node_name_to_scope, patterns,
               prepare_custom_config_dict, equalization_qconfig_map)
    preserved_attributes = set(prepare_custom_config_dict.get("preserved_attributes", []))
    model = ObservedGraphModule(model, model.graph, preserved_attributes)
    if is_standalone_module:
        assert result_node is not None
        assert isinstance(result_node.args[0], Node), \
            "standalone module only supports returning simple value currently"\
            "(not tuple, dict etc.)"
        # these inputs are observed in parent
        # converting List[int] to Tensor since module attribute is
        # Union[Tensor, Module]
        model._standalone_module_input_quantized_idxs = \
            torch.tensor(input_quantized_idxs)
        model._standalone_module_output_quantized_idxs = torch.tensor(output_quantized_idxs)
    return model
