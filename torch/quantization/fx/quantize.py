import torch
from torch.fx import (
    GraphModule,
    Proxy,
    map_arg
)

from torch.fx.graph import (
    Graph,
    Node,
)

from torch.fx.node import Argument

from torch.quantization import (
    propagate_qconfig_,
    convert,
)

from ..quantization_mappings import (
    get_default_qat_module_mappings,
)

from ..quantize import (
    _remove_qconfig,
    is_activation_post_process
)

from ..utils import (
    get_combined_dict,
    get_qconfig_dtypes,
    weight_is_quantized,
    activation_is_statically_quantized,
    activation_is_int8_quantized,
    activation_dtype,
    weight_dtype,
)

from .pattern_utils import (
    is_match,
    get_default_quant_patterns,
    get_default_output_activation_post_process_map,
    Pattern,
)

from .graph_module import (
    is_observed_module,
    is_observed_standalone_module,
    ObservedGraphModule,
    ObservedStandaloneGraphModule,
    QuantizedGraphModule,
)

from .quantization_patterns import (
    binary_op_supported_dtypes,
    binary_reference_op_supported_dtypes,
    BinaryOpQuantizeHandler,
    CatQuantizeHandler,
    CopyNodeQuantizeHandler,
    CustomModuleQuantizeHandler,
    QuantizeHandler,
    StandaloneModuleQuantizeHandler,
)

from .utils import (
    _parent_name,
    all_node_args_have_no_tensors,
    quantize_node,
    get_custom_module_class_keys,
    get_new_attr_name_with_prefix,
    collect_producer_nodes,
    graph_module_from_producer_nodes,
    assert_and_get_unique_device,
    node_return_type_is_int,
    node_bool_tensor_arg_indexes,
)

from .qconfig_utils import (
    convert_dict_to_ordered_dict,
    get_flattened_qconfig_dict,
    get_object_type_qconfig,
    get_qconfig,
    QConfigAny,
)

import operator

from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

# Define helper types
MatchResult = Tuple[Node, List[Node], Optional[Pattern], QuantizeHandler,
                    QConfigAny]

# ------------------------
# Helper Functions
# ------------------------

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
        weight_needs_obs = is_weight and weight_is_quantized(qconfig)
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
    node.kwargs = new_kwargs  # type: ignore[assignment]


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

    if qhandler is not None:
        assert qconfig is not None

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

    elif node.op == 'output':
        prev_node = node.args[0]
        assert isinstance(prev_node, Node)
        prev_node_dtype = node_name_to_target_dtype[prev_node.name]
        node_dtype = node_name_to_target_dtype[node.name]
        should_insert_observer = (
            prev_node_dtype == torch.float and
            node_dtype != torch.float
        )
        if should_insert_observer:
            assert qconfig is not None
            observer = qconfig.activation()
            new_obs = insert_observer(
                prev_node, observer, model, modules, graph)
            # set the type, so the next node can read it
            node_name_to_target_dtype[new_obs.name] = node_dtype
            return new_obs

    return None

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

def insert_observers_for_model(
    model: GraphModule,
    modules: Dict[str, torch.nn.Module],
    matches: Dict[str, MatchResult],
    qconfig_map: Dict[str, QConfigAny],
    graph: Graph,
    prepare_custom_config_dict: Dict[str, Any],
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

        # check for matches
        root_node, matched_nodes, pattern, qhandler, qconfig = matches.get(
            node.name, (None, None, None, None, None))

        if node.op == 'placeholder':
            # if a graph input is in fp32, it does not need observation
            # if a graph input is in int8, we assume the observation happens
            #   outside of the graph, and no additional observation is needed
            pass

        elif node.op in ('call_module', 'call_method', 'call_function', 'output'):
            modules = dict(model.named_modules(remove_duplicate=False))
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
                if node.op != 'output':
                    # this modifies node inplace
                    maybe_insert_input_observers_for_node(
                        node, qconfig, model, modules, graph,
                        node_name_to_target_dtype,
                        qhandler, prepare_custom_config_dict)

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

                else:  # output
                    prev_node = node.args[0]
                    if isinstance(prev_node, Node):
                        if is_activation_post_process_node(prev_node, modules):
                            prev_node = prev_node.args[0]
                    elif isinstance(prev_node, dict):
                        # get first value
                        prev_node = list(prev_node.items())[0][1]
                        assert isinstance(prev_node, Node)
                        if is_activation_post_process_node(prev_node, modules):
                            prev_node = prev_node.args[0]

                    # we check for node again because some graphs can return
                    # None
                    if isinstance(prev_node, Node):
                        prev_node_qconfig = qconfig_map.get(prev_node.name, None)
                        # this modifies node inplace
                        maybe_insert_input_observers_for_node(
                            node, prev_node_qconfig, model, modules, graph,
                            node_name_to_target_dtype,
                            qhandler, prepare_custom_config_dict)

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

        standalone_module = modules[root_node.target]  # type: ignore[index]
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
        modules[root_node.target] = observed_standalone_module  # type: ignore[index]


def is_activation_post_process_node(node: Node, modules: Dict[str, torch.nn.Module]) -> bool:
    return node.op == "call_module" and \
        is_activation_post_process(modules[str(node.target)])


# A dictionary for querying the weight index for a given op
WEIGHT_INDEX_DICT = {
    torch.nn.functional.conv1d : [1],
    torch.nn.functional.conv2d : [1],
    torch.nn.functional.conv3d : [1],
    torch.nn.functional.linear : [1],
}

def node_arg_is_weight(node: Node, arg: Any) -> bool:
    if isinstance(node, Node) and node.op == 'call_function' and \
            node.target in WEIGHT_INDEX_DICT:
        for i, node_arg in enumerate(node.args):
            if arg is node_arg and i in \
                    WEIGHT_INDEX_DICT[node.target]:  # type: ignore[index]
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
        elif node.target is torch.nn.functional.linear:
            for kwarg_name, kwarg_value in node.kwargs.items():
                if kwarg_name == 'bias' and arg is kwarg_value:
                    return True
    return False


# weight prepacking ops
WEIGHT_PREPACK_OPS = {
    torch._ops.ops.quantized.linear_prepack,
    torch._ops.ops.quantized.linear_prepack_fp16,
    torch._ops.ops.quantized.conv1d_prepack,
    torch._ops.ops.quantized.conv2d_prepack,
    torch._ops.ops.quantized.conv3d_prepack,
}

class Quantizer:
    def __init__(self):
        # mapping from node name to qconfig that should be used for that node
        # filled out for a model during _generate_qconfig_map
        self.qconfig_map: Dict[str, QConfigAny] = {}
        # mapping from fully qualified module name to module instance
        # for example,
        # {
        #   '': Model(...),
        #   'linear': Linear(...),
        #   'linear.weight_fake_quant': PerChannelMinMaxObserver(...),
        # }
        self.modules: Dict[str, torch.nn.Module] = {}
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
        self.patterns: Dict[Pattern, QuantizeHandler] = {}
        self.prepare_custom_config_dict: Dict[str, Any] = {}

        # mapping from node name to the scope of the module which contains the node.
        self.node_name_to_scope: Dict[str, Tuple[str, type]] = {}


    def _qat_swap_modules(
            self, root: torch.nn.Module,
            additional_qat_module_mapping: Dict[Callable, Callable]) -> None:
        all_mappings = get_combined_dict(
            get_default_qat_module_mappings(), additional_qat_module_mapping)
        convert(root, mapping=all_mappings, inplace=True, remove_qconfig=False)

    def _generate_qconfig_map(
            self,
            root: torch.nn.Module,
            input_graph: Graph,
            qconfig_dict: Any,
            node_name_to_scope: Dict[str, Tuple[str, type]]) -> None:
        global_qconfig = qconfig_dict.get("", None)
        self.node_name_to_scope = node_name_to_scope
        self.qconfig_map = dict()
        for node in input_graph.nodes:
            if node.op == "get_attr":
                module_name, _ = _parent_name(node.target)
                self.qconfig_map[node.name] = get_qconfig(
                    qconfig_dict, type(self.modules[module_name]), module_name, global_qconfig)
            elif node.op == "call_function":
                # precedence: [TODO] module_name_qconfig (need scope support
                # from fx)
                # > function_qconfig > global_qconfig
                # module_name takes precedence over function qconfig
                function_qconfig = get_object_type_qconfig(
                    qconfig_dict, node.target, global_qconfig)
                module_path, module_type = node_name_to_scope[node.name]
                qconfig = get_qconfig(
                    qconfig_dict, module_type, module_path, function_qconfig)
                self.qconfig_map[node.name] = qconfig
            elif node.op == "call_method":
                module_path, module_type = node_name_to_scope[node.name]
                # use the qconfig of the module that the node belongs to
                qconfig = get_qconfig(
                    qconfig_dict, module_type, module_path, global_qconfig)
                self.qconfig_map[node.name] = qconfig
            elif node.op == 'call_module':
                module_qconfig = get_qconfig(
                    qconfig_dict, type(self.modules[node.target]), node.target, global_qconfig)
                # regex is not supported eager mode propagate_qconfig_, we'll
                # need to set the qconfig explicitly here in case regex
                # is used
                self.modules[node.target].qconfig = module_qconfig
                self.qconfig_map[node.name] = module_qconfig

    def _prepare(
            self,
            model: GraphModule,
            qconfig_dict: Any,
            node_name_to_scope: Dict[str, Tuple[str, type]],
            prepare_custom_config_dict: Optional[Dict[str, Any]],
            is_standalone_module: bool) -> ObservedGraphModule:
        """ standalone_module means it a submodule that is not inlined in
        parent module, and will be quantized separately as one unit.

        How the standalone module is observed is specified by `input_quantized_idxs` and
        `output_quantized_idxs` in the prepare_custom_config for the standalone module
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
        self.prepare_custom_config_dict = prepare_custom_config_dict

        additional_quant_patterns = \
            prepare_custom_config_dict.get("additional_quant_pattern", {})
        self.patterns = get_combined_dict(
            get_default_quant_patterns(), additional_quant_patterns)

        convert_dict_to_ordered_dict(qconfig_dict)
        flattened_qconfig_dict = get_flattened_qconfig_dict(qconfig_dict)
        # TODO: support regex as well
        propagate_qconfig_(model, flattened_qconfig_dict)
        if model.training:
            additional_qat_module_mapping = prepare_custom_config_dict.get(
                "additional_qat_module_mapping", {})
            self._qat_swap_modules(model, additional_qat_module_mapping)

        self.modules = dict(model.named_modules())

        # fill self.qconfig_map, a map from node name to qconfig, used in _find_matches
        self._generate_qconfig_map(model, model.graph, qconfig_dict, node_name_to_scope)

        # match the patterns that will get quantized
        standalone_module_name_configs = prepare_custom_config_dict.get(
            "standalone_module_name", [])
        standalone_module_class_configs = prepare_custom_config_dict.get(
            "standalone_module_class", [])

        standalone_module_names = [config[0] for config in standalone_module_name_configs]
        standalone_module_classes = [config[0] for config in standalone_module_class_configs]
        custom_module_classes = get_custom_module_class_keys(
            prepare_custom_config_dict, "float_to_observed_custom_module_class")
        matches = self._find_matches(
            model.graph, self.modules, self.patterns, standalone_module_names,
            standalone_module_classes, custom_module_classes)

        input_quantized_idxs: List[int] = self.prepare_custom_config_dict.get(
            "input_quantized_idxs", [])
        output_quantized_idxs: List[int] = self.prepare_custom_config_dict.get(
            "output_quantized_idxs", [])

        run_prepare_fx_on_standalone_modules(
            model, self.modules, matches, prepare_custom_config_dict)

        result_node = insert_observers_for_model(
            model, self.modules, matches, self.qconfig_map,
            model.graph, prepare_custom_config_dict,
            input_quantized_idxs, output_quantized_idxs)

        self.save_state(model)
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

    def save_state(self, observed: GraphModule) -> None:
        observed._patterns = self.patterns  # type: ignore[assignment]
        observed._qconfig_map = self.qconfig_map  # type: ignore[assignment]
        observed._prepare_custom_config_dict = \
            self.prepare_custom_config_dict  # type: ignore[assignment]
        observed._node_name_to_scope = self.node_name_to_scope  # type: ignore[assignment]

    def restore_state(self, observed: GraphModule) -> None:
        assert is_observed_module(observed), \
            'incoming model must be produced by prepare_fx'
        self.patterns = observed._patterns  # type: ignore[assignment]
        self.qconfig_map = observed._qconfig_map  # type: ignore[assignment]
        self.prepare_custom_config_dict = \
            observed._prepare_custom_config_dict  # type: ignore[assignment]
        self.node_name_to_scope = observed._node_name_to_scope  # type: ignore[assignment]

    def prepare(
            self,
            model: GraphModule,
            qconfig_dict: Any,
            node_name_to_scope: Dict[str, Tuple[str, type]],
            prepare_custom_config_dict: Dict[str, Any] = None,
            is_standalone_module: bool = False) -> ObservedGraphModule:
        return self._prepare(
            model, qconfig_dict, node_name_to_scope, prepare_custom_config_dict,
            is_standalone_module)

    def _run_weight_observers(self, observed: GraphModule) -> None:
        r''' Extract the subgraph that produces the weight for dynamic quant
        or weight only quant node and run the subgraph to observe the weight.
        Note that the observers of dynamic quant or weight only quant ops are
        run during the convert step.
        '''
        for node in observed.graph.nodes:
            if node.op == 'call_function' and node.target in WEIGHT_INDEX_DICT:
                for i, node_arg in enumerate(node.args):
                    if i in WEIGHT_INDEX_DICT[node.target]:
                        # node_arg is weight
                        weight_observer_nodes = collect_producer_nodes(node_arg)
                        if weight_observer_nodes is not None:
                            weight_observer_module = \
                                graph_module_from_producer_nodes(
                                    observed, weight_observer_nodes)
                            # run the weight observer
                            weight_observer_module()
        return

    def _convert(self, model: GraphModule, is_reference: bool = False,
                 convert_custom_config_dict: Dict[str, Any] = None,
                 is_standalone_module: bool = False,
                 _remove_qconfig_flag: bool = True) -> QuantizedGraphModule:
        """ standalone_module means it a submodule that is not inlined in
        parent module, and will be quantized separately as one unit.

        Returns a quantized standalone module, whether input/output is quantized is
        specified by prepare_custom_config_dict, with
        input_quantized_idxs, output_quantized_idxs, please
        see docs for prepare_fx for details
        """
        if convert_custom_config_dict is None:
            convert_custom_config_dict = {}
        self.restore_state(model)
        # always run weight observers in the top level forward method
        # for dynamic quant ops or weight only quant ops
        self._run_weight_observers(model)

        # move to cpu since we only have quantized cpu kernels
        model.eval().cpu()
        self.modules = dict(model.named_modules(remove_duplicate=False))

        custom_module_classes = get_custom_module_class_keys(
            convert_custom_config_dict,
            "observed_to_quantized_custom_module_class")
        matches = self._find_matches(
            model.graph, self.modules, self.patterns,
            custom_module_classes=custom_module_classes)

        self.quantized_graph = Graph()
        env: Dict[str, Node] = {}
        # TODO: merge quant_env with env
        quant_env: Dict[str, Tuple[Node, torch.dtype]] = {}

        graph_inputs: List[str] = []
        for node in model.graph.nodes:
            if node.op == 'placeholder':
                graph_inputs.append(node.name)

        def load_non_quantized(n: Node) -> Node:
            if n.name not in env:
                assert n.name in quant_env, \
                    'trying to load float node but did not find ' + \
                    'node:' + n.name + \
                    ' in quantized or non quantized environment, env: ' + \
                    str(env) + ' quant_env:' + str(quant_env)
                quantized_node, _ = quant_env[n.name]
                env[n.name] = Proxy(quantized_node).dequantize().node
            return env[n.name]

        def load_quantized(n: Node) -> Node:
            assert n.name in quant_env, \
                'trying to load quantized node but did not find node:' + \
                n.name + ' in quant environment:' + str(quant_env)
            return quant_env[n.name][0]

        def load_x(n: Node) -> Node:
            assert n.name in env or n.name in quant_env, \
                'node ' + n.name + ' does not exist in either environment'
            if n.name in quant_env:
                return quant_env[n.name][0]
            else:
                return env[n.name]

        def load_arg(quantized: Optional[Union[List[int], bool, Tuple[int, ...]]]
                     ) -> Callable[[Node], Argument]:
            """
            Input: quantized, which can be None, list, boolean or tuple
              - if quantized is None, then we'll load the node as long as it
                exists
              - if quantized is a boolean, then all args will be
                quantized/not quantized
              - if quantized is an empty list or tuple, then it is the same as load_arg(quantized=False)
              - if quantized is a list or tuple, then arg should be a list and
                the args with corresponding indexes will be quantized


            Output: fn which takes arg_or_args, and loads them from the
                corresponding environment depending on the value of quantized.
            """
            assert quantized is None or \
                isinstance(quantized, (tuple, list, bool)), type(quantized)
            if isinstance(quantized, (tuple, list)) and len(quantized) == 0:
                # empty tuple or list means nothing is quantized
                quantized = False

            def load_arg_impl(arg_or_args):
                # we'll update the format of `quantized`
                # to better match arg_or_args
                updated_quantized: Optional[Union[List[int], bool, Tuple[int, ...]]] = quantized

                if isinstance(quantized, (tuple, list)) and \
                   len(quantized) == 1 and isinstance(arg_or_args, Node):
                    # when argument is one Node instead of tuple, we just need to check
                    # 0 is in the quantized list
                    updated_quantized = 0 in quantized

                if updated_quantized is None:
                    return map_arg(arg_or_args, load_x)
                if isinstance(updated_quantized, bool):
                    return map_arg(
                        arg_or_args,
                        load_quantized if updated_quantized else load_non_quantized)
                elif isinstance(updated_quantized, (tuple, list)):
                    assert isinstance(arg_or_args, (tuple, list)), arg_or_args
                    loaded_args = []
                    # for now, we only support quantizing positional arguments
                    for i, a in enumerate(arg_or_args):
                        if i in updated_quantized:
                            loaded_args.append(map_arg(a, load_quantized))
                        else:
                            loaded_args.append(map_arg(a, load_non_quantized))
                    return type(arg_or_args)(loaded_args)
            return load_arg_impl

        def node_arg_is_quantized(node_arg: Any) -> bool:
            if isinstance(node_arg, Node):
                assert node_arg.name in env or node_arg.name in quant_env, \
                    'Expecting node_arg to be in the environment'
                # there might be nodes appearing in both environemnts, but
                # quant_env will take precedence
                if node_arg.name in quant_env:
                    return True
                elif node_arg.name in env:
                    return False
                else:
                    return False
            elif isinstance(node_arg, list):
                quantized = map(node_arg_is_quantized, node_arg)
                if all(quantized):
                    return True
                elif not any(quantized):
                    return False
                else:
                    raise Exception(
                        "partially quantized inputs in list not handled yet")
            else:
                return False

        def is_output_quantized(node: Node, obj: QuantizeHandler) -> bool:
            """ Check if output node is quantized or not """
            assert self.modules is not None
            # by default the output for a quantizable node is expected to be quantized
            quantized = True

            # Need to get correct quantized/non-quantized state forn the output
            # of FixedQParamsQuantizeHandler
            # TODO: we may want to try to remove the special case here
            # as well
            if obj.should_mark_output_quantized_from_input_quantized_status():
                assert node.op in [
                    'call_module',
                    'call_function',
                    'call_method'], \
                    'FixedQParamsQuantizeHandler of type ' + node.op + ' is not handled'
                # TODO: need to extend this to consider all relevant args instead of just arg[0]
                quantized = node_arg_is_quantized(node.args[0])

            # the output is unquantized if the node is not a CopyNode
            # and activation is fp16 (since we will output fp32 currently for fp16
            # converter
            if not activation_is_int8_quantized(qconfig) or \
               not obj.input_output_observed():
                quantized = False
            if node_return_type_is_int(node):
                quantized = False

            return quantized

        def insert_quantize_node(node: Node) -> None:
            """ Given a activation_post_process module call node, insert a
            quantize node"""
            assert self.modules is not None
            assert isinstance(node.target, str)
            observer_module = self.modules[node.target]
            prev_node = node.args[0]
            if observer_module.dtype == torch.float32:
                # copy the observer for fp32 dtype
                env[node.name] = self.quantized_graph.node_copy(
                    node, load_non_quantized)
            elif isinstance(prev_node, Node) and prev_node.name in quant_env:
                # if previous node is already quantized, we'll just remove the
                # activation_post_process
                _, prev_dtype = quant_env[prev_node.name]
                current_dtype = observer_module.dtype
                if prev_dtype == current_dtype:
                    quant_env[node.name] = quant_env[prev_node.name]
                else:
                    root_module = self.modules[""]
                    assert isinstance(prev_node, Node)
                    observer_dtype: torch.dtype = observer_module.dtype  # type: ignore[assignment]
                    quant_env[node.name] = (
                        quantize_node(self, load_non_quantized(prev_node),
                                      observer_module, node, is_input=True),
                        observer_dtype)
            else:
                # replace activation post process with quantization ops
                root_module = self.modules[""]
                assert isinstance(node.args[0], Node)
                dtype: torch.dtype = observer_module.dtype  # type: ignore[assignment]
                quant_env[node.name] = (
                    quantize_node(self, load_non_quantized(node.args[0]),
                                  observer_module, node, is_input=True),
                    dtype)

        # additional state to override inputs to be quantized, if specified
        # by the user
        placeholder_node_seen_cnt = 0
        output_node_seen_cnt = 0
        input_quantized_idxs: List[int] = self.prepare_custom_config_dict.get(
            "input_quantized_idxs", [])
        output_quantized_idxs: List[int] = self.prepare_custom_config_dict.get(
            "output_quantized_idxs", [])

        for node in model.graph.nodes:
            if node.op == "output":
                cur_output_node_idx = output_node_seen_cnt
                output_node_seen_cnt += 1
                if cur_output_node_idx in output_quantized_idxs:
                    # Result are kept quantized if the user specified the
                    # output_quantized_idxs override.
                    graph_output = map_arg(node.args[0], load_x)
                else:
                    graph_output = map_arg(node.args[0], load_non_quantized)
                self.quantized_graph.output(graph_output)
                continue
            root_node, matched, matched_pattern, obj, qconfig = \
                matches.get(node.name, (None, None, None, None, None))
            if root_node is node:
                is_observed_standalone_module_node = (
                    node.op == 'call_module' and
                    is_observed_standalone_module(
                        self.modules[node.target])
                )
                if qconfig is None and not is_observed_standalone_module_node:
                    result = self.quantized_graph.node_copy(
                        node, load_non_quantized)
                    quantized = False
                else:
                    assert obj is not None
                    # We will get whether the output is quantized or not before
                    # convert for standalone module and after convert
                    # for non-standalone module, since _standalone_module_output_quantized_idxs
                    # is only available in observed standalone module
                    if is_observed_standalone_module_node:
                        out_quant_idxs = self.modules[node.target]._standalone_module_output_quantized_idxs.tolist()  # type: ignore[operator] # noqa: B950
                        assert len(out_quant_idxs) <= 1, "Currently standalone only support one output"
                        quantized = 0 in out_quant_idxs

                    result = obj.convert(
                        self, node, load_arg, is_reference=is_reference,
                        convert_custom_config_dict=convert_custom_config_dict)
                    if not is_observed_standalone_module_node:
                        quantized = is_output_quantized(node, obj)

                if quantized:
                    quant_env[node.name] = result, activation_dtype(qconfig)
                else:
                    env[node.name] = result
                continue
            elif root_node is not None:
                if qconfig is None:
                    # This branch is hit if all of these conditions are met:
                    # 1. we are in a fusion pattern of multiple nodes (i.e. add-relu)
                    # 2. the current node is not the "root_node" of the pattern
                    # 3. quantization for this pattern is disabled
                    #
                    # In this case, we need to make sure to populate the env with
                    # intermediate nodes manually, because the QuantizeHandler.convert
                    # function will not be called.
                    result = self.quantized_graph.node_copy(
                        node, load_non_quantized)
                    env[node.name] = result
                continue

            # handle activation post process calls
            if node.op == 'call_module' and \
                    is_activation_post_process(self.modules[node.target]):
                insert_quantize_node(node)
            elif node.op == 'placeholder':
                cur_placeholder_node_idx = placeholder_node_seen_cnt
                placeholder_node_seen_cnt += 1
                if cur_placeholder_node_idx in input_quantized_idxs:
                    quant_env[node.name] = \
                        self.quantized_graph.node_copy(node, load_non_quantized), activation_dtype(qconfig) if qconfig else None
                else:
                    env[node.name] = \
                        self.quantized_graph.node_copy(node, load_non_quantized)
            else:
                # copy quantized or non-quantized node
                env[node.name] = \
                    self.quantized_graph.node_copy(node, load_non_quantized)

        # remove activation post process
        act_post_process_removed_graph = Graph()
        env = {}

        def load_arg_simple(a: Argument) -> Argument:
            return map_arg(a, lambda node: env[node.name])
        for node in self.quantized_graph.nodes:
            if node.op == 'output':
                act_post_process_removed_graph.output(
                    map_arg(node.args[0], load_arg_simple))
                continue
            if node.op == 'call_module' and \
               is_activation_post_process(self.modules[node.target]):
                # remove activation post process node
                env[node.name] = env[node.args[0].name]
            else:
                env[node.name] = act_post_process_removed_graph.node_copy(
                    node, load_arg_simple)

        # removes qconfig and activation_post_process modules
        if _remove_qconfig_flag:
            _remove_qconfig(model)
        preserved_attributes = set(convert_custom_config_dict.get("preserved_attributes", []))
        model = QuantizedGraphModule(model, act_post_process_removed_graph, preserved_attributes)
        return model

    # Trace back from the weight node util we hit getattr, reconstruct the
    # graph module with the traced nodes and run the graph module to pack the
    # weight. then replace the original chain of ops with the packed weight.
    def _fold_weight(self, quantized: QuantizedGraphModule) -> QuantizedGraphModule:
        packed_weights = dict()
        # map from folded node name to the prepacked weight name
        folded_nodes = dict()
        # get packed weights
        for node in quantized.graph.nodes:
            if node.op == 'call_function' and node.target in WEIGHT_PREPACK_OPS:
                nodes_to_fold = collect_producer_nodes(node)
                if nodes_to_fold is not None:
                    for node_to_fold in nodes_to_fold:
                        folded_nodes[node_to_fold.name] = node

                    prepacking_module = graph_module_from_producer_nodes(
                        quantized, nodes_to_fold)
                    packed_weight = prepacking_module()
                    packed_weights[node.name] = packed_weight

        # remove folded nodes and replace the prepacking node with getattr
        folded_graph = Graph()
        env: Dict[Any, Any] = {}

        def load_arg(a):
            return map_arg(a, lambda node: env[node.name])
        quantized_root = quantized
        quantized_graph = quantized.graph

        for node in quantized_graph.nodes:
            prepack_node = folded_nodes.get(node.name, None)
            if prepack_node is node:
                packed_weight = packed_weights[node.name]
                # add a prepacked attribute to root
                op_node = list(prepack_node.users)[0]
                module_path, _ = self.node_name_to_scope[op_node.name]
                get_new_packed_weight_name = \
                    get_new_attr_name_with_prefix(module_path + '_packed_weight_')
                packed_weight_name = get_new_packed_weight_name(quantized_root)
                setattr(quantized_root, packed_weight_name, packed_weight)
                # replace prepack node with a getattr node
                env[node.name] = folded_graph.create_node(
                    'get_attr', packed_weight_name, (), {})
            elif prepack_node is not None:
                # remove the foled node
                continue
            else:
                # copy other nodes
                env[node.name] = folded_graph.node_copy(node, load_arg)
        quantized = QuantizedGraphModule(quantized_root, folded_graph, quantized_root.preserved_attr_names)
        return quantized

    def convert(self, model: GraphModule, is_reference: bool = False,
                convert_custom_config_dict: Dict[str, Any] = None,
                is_standalone_module: bool = False,
                _remove_qconfig: bool = True) -> QuantizedGraphModule:
        quantized = self._convert(
            model, is_reference, convert_custom_config_dict, is_standalone_module, _remove_qconfig_flag=_remove_qconfig)
        if not is_reference:
            quantized = self._fold_weight(quantized)
        return quantized

    def _find_matches(
            self, graph: Graph, modules: Dict[str, torch.nn.Module],
            patterns: Dict[Pattern, QuantizeHandler],
            standalone_module_names: List[str] = None,
            standalone_module_classes: List[Callable] = None,
            custom_module_classes: List[Any] = None) -> Dict[str, MatchResult]:
        """
        Matches the nodes in the input graph to quantization patterns, and
        outputs the information needed to quantize them in future steps.

        Inputs:
          - graph: an fx.Graph object
          - modules: a mapping of fully qualified module name to instance,
              for example, {'foo': ModuleFoo, ...}
          - patterns: a mapping from a tuple of nodes in reverse order to
              uninitialized QuantizeHandler subclass.

        Outputs a map of
          node_name ->
            (node, matched_values, matched_pattern, QuantizeHandler instance,
             qconfig)

        For example, {
          'relu_1': (relu_1, [relu_1], torch.nn.functional.relu,
                     <CopyNodeQuantizeHandler instance>, QConfig(...)),
          ...
        }
        """
        if custom_module_classes is None:
            custom_module_classes = []

        if standalone_module_classes is None:
            standalone_module_classes = []

        if standalone_module_names is None:
            standalone_module_names = []

        match_map: Dict[str, MatchResult] = {}
        all_matched : Set[str] = set()

        def record_match(pattern, node, matched):
            if isinstance(pattern, tuple):
                s, *args = pattern
                record_match(s, node, matched)
                if pattern[0] is not getattr:
                    for subpattern, arg in zip(args, node.args):
                        record_match(subpattern, arg, matched)
            else:
                matched.append(node)

        cache_for_no_tensor_check: Dict[Node, bool] = dict()
        for node in reversed(graph.nodes):
            if node.name not in match_map and node.name not in all_matched:
                for pattern, value in patterns.items():
                    if is_match(modules, node, pattern):
                        skip_this_match = False
                        if value is BinaryOpQuantizeHandler:

                            # to properly check for dtype support, we need to
                            # navigate to the base node of an add-relu or mul-relu
                            # pattern
                            base_node = node
                            if (
                                (node.op == 'call_function' and
                                 node.target is torch.nn.functional.relu) or
                                (node.op == 'call_module' and
                                 isinstance(modules[node.target], torch.nn.ReLU))
                            ):
                                base_node = node.args[0]

                            this_node_qconfig = \
                                self.qconfig_map[base_node.name]
                            if this_node_qconfig:
                                dtypes = get_qconfig_dtypes(this_node_qconfig)
                                # TODO(future PR): update the pattern to quantize
                                # handler logic to take this into account.


                                # This needs to handle 3 cases
                                # 1) op and dtype is in either [is_ref or non-ref] list -> don't skip
                                # 2) op is not in either list (i.e. relu) -> don't skip
                                # 3) op is in non-ref list, but not for dtype, and op+dtype not in is_ref list -> skip

                                # note: the value of is_reference is unknown at prepare, so we have to cover both cases
                                # handle is_reference = False
                                skip_match_not_is_reference = (
                                    (base_node.target in binary_op_supported_dtypes) and
                                    (dtypes not in binary_op_supported_dtypes[base_node.target])
                                )

                                # handle is_reference = True
                                supported_is_reference = (
                                    (base_node.target in binary_reference_op_supported_dtypes) and
                                    (dtypes in binary_reference_op_supported_dtypes[base_node.target])
                                )

                                # only skip if not reference says skip and is_reference doesn't support
                                skip_this_match = skip_match_not_is_reference and not supported_is_reference

                        if not skip_this_match:
                            matched: List[Any] = []
                            record_match(pattern, node, matched)
                            for n in matched:
                                match_map[n.name] = (
                                    node, matched, pattern, value(self, node),  # type: ignore[operator]
                                    self.qconfig_map[n.name])
                                all_matched.add(n.name)
                            # break after finding the first match
                            break

        # add custom module instances to the match result
        assert self.modules is not None
        for node in graph.nodes:
            if node.op == 'call_module' and \
               type(self.modules[node.target]) in custom_module_classes:
                custom_module_qconfig = self.qconfig_map[node.name]
                match_map[node.name] = (
                    node, [node], None, CustomModuleQuantizeHandler(self, node),
                    custom_module_qconfig)

        def is_standalone_module(node_target):
            assert self.modules is not None
            return (
                node_target in standalone_module_names or  # type: ignore[operator]
                type(self.modules[node_target]) in standalone_module_classes  # type: ignore[operator]
            )

        # add standalone modules to the match
        for node in graph.nodes:
            if node.op == 'call_module' and \
               (is_standalone_module(node.target) or
                    is_observed_standalone_module(self.modules[node.target])):
                # add node to matched nodes
                custom_module_qconfig = self.qconfig_map[node.name]
                match_map[node.name] = (
                    node, [node], None,
                    StandaloneModuleQuantizeHandler(self, node),
                    custom_module_qconfig)

        return match_map
