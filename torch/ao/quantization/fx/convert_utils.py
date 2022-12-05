from typing import Any, Dict, Optional, Set, Tuple, Type
import warnings
import torch
import torch.ao.quantization.quantize_fx
from torch.fx import GraphModule
from torch.fx.graph import (
    Argument,
    Graph,
    Node,
)
from ..backend_config import BackendConfig
from ..quant_type import QuantType
from ..quantize import is_activation_post_process
from ..qconfig import QConfigAny
from ..utils import (
    activation_is_statically_quantized,
    _parent_name,
    get_swapped_custom_module_class,
)
from .custom_config import PrepareCustomConfig
from .graph_module import is_observed_module
from .utils import (
    _is_custom_module_lstm,
    collect_producer_nodes,
    graph_module_from_producer_nodes,
    node_arg_is_weight,
)

# this is a temporary hack for custom module, we may want to implement
# this properly after the custom module class design is finalized
# TODO: DeQuantStubs are currently inserted only after custom module LSTM, while observers are inserted
# after all other custom modules. In the future, we should simply insert QuantStubs before and DeQuantStubs
# after custom modules in general, and replace these with "quantize" and "dequantize" nodes respectively.
def _replace_observer_or_dequant_stub_with_dequantize_node(node: Node, graph: Graph):
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

def _restore_state(
        observed: torch.nn.Module
) -> Tuple[Dict[str, Tuple[str, type]],
           PrepareCustomConfig,
           Set[str]]:
    assert is_observed_module(observed), \
        'incoming model must be produced by prepare_fx'
    prepare_custom_config: PrepareCustomConfig = observed._prepare_custom_config  # type: ignore[assignment]
    node_name_to_scope: Dict[str, Tuple[str, type]] = observed._node_name_to_scope  # type: ignore[assignment]
    observed_node_names: Set[str] = observed._observed_node_names  # type: ignore[assignment]
    return node_name_to_scope, prepare_custom_config, observed_node_names

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
            if node_arg and node_arg_is_weight(node, node_arg, backend_config):
                weight_observer_nodes = collect_producer_nodes(node_arg)
                if weight_observer_nodes is None:
                    continue
                weight_observer_module = \
                    graph_module_from_producer_nodes(
                        observed, weight_observer_nodes)
                # run the weight observer
                weight_observer_module()

def _maybe_recursive_remove_dequantize(arg: Any, node: Node, graph: Graph):
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
        node_name_to_qconfig: Dict[str, QConfigAny]):
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
        graph: Graph):
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
            if is_activation_post_process(maybe_obs):
                return maybe_obs
    return None

def _convert_standalone_module(
        node: Node,
        modules: Dict[str, torch.nn.Module],
        model: torch.fx.GraphModule,
        is_reference: bool,
        backend_config: Optional[BackendConfig]):
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
        ._standalone_module_input_quantized_idxs\
        .tolist()  # type: ignore[operator]
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
        ._standalone_module_output_quantized_idxs \
        .tolist()  # type: ignore[operator]
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

def _remove_previous_dequantize_in_custom_module(node: Node, prev_node: Node, graph: Graph):
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

def _convert_custom_module(
        node: Node,
        graph: Graph,
        modules: Dict[str, torch.nn.Module],
        custom_module_class_mapping: Dict[QuantType, Dict[Type, Type]],
        statically_quantized_custom_module_nodes: Set[Node]):
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
