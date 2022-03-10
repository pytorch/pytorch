from typing import Any, Dict, List, Optional, Set, Callable, Tuple
import torch
import copy
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
    get_quant_type,
)
from ..qconfig import (
    QConfigAny,
    qconfig_equals
)
from ..qconfig_dict_utils import (
    convert_dict_to_ordered_dict,
    update_qconfig_for_qat,
)
from .qconfig_utils import (
    generate_qconfig_map,
    compare_prepare_convert_qconfig_dict,
    update_qconfig_for_fusion,
)
from ..quantization_mappings import DEFAULT_REFERENCE_STATIC_QUANT_MODULE_MAPPINGS
from .backend_config.utils import get_quantized_reference_module_mapping
from .graph_module import (
    QuantizedGraphModule,
    is_observed_standalone_module,
)
from ._equalize import update_obs_for_equalization, convert_eq_obs
from .utils import (
    get_custom_module_class_keys,
    get_quantize_node_info,
    create_getattr_from_value,
    collect_producer_nodes,
    graph_module_from_producer_nodes,
    WEIGHT_INDEX_DICT,
)
from ..quant_type import QuantType

from torch.ao.quantization.quantize import (
    _remove_qconfig,
    is_activation_post_process,
)
from .lower_to_fbgemm import lower_to_fbgemm
from .convert import restore_state

# these are tuples so that they can work with isinstance(module, tuple_of_classes)
FUSED_MODULE_CLASSES = (
    torch.nn.intrinsic.LinearReLU,
    torch.nn.intrinsic.LinearBn1d,
    torch.nn.intrinsic.ConvReLU1d,
    torch.nn.intrinsic.ConvReLU2d,
    torch.nn.intrinsic.ConvReLU3d,
)

QAT_MODULE_CLASSES = (
    torch.nn.qat.Linear,
    torch.nn.qat.Conv2d,
    torch.nn.qat.Conv3d,
    torch.nn.intrinsic.qat.LinearReLU,
    torch.nn.intrinsic.qat.LinearBn1d,
    torch.nn.intrinsic.qat.ConvBn2d,
    torch.nn.intrinsic.qat.ConvBnReLU2d,
    torch.nn.intrinsic.qat.ConvReLU2d,
    torch.nn.intrinsic.qat.ConvBn3d,
    torch.nn.intrinsic.qat.ConvBnReLU3d,
    torch.nn.intrinsic.qat.ConvReLU3d
)

WEIGHT_ONLY_MODULE_CLASSES = (
    torch.nn.Embedding,
    torch.nn.EmbeddingBag,
)

DYNAMIC_MODULE_CLASSES = (
    torch.nn.GRUCell,
    torch.nn.LSTMCell,
    torch.nn.RNNCell,
    torch.nn.LSTM,
)

def has_none_qconfig(node: Argument, qconfig_map: Dict[str, QConfigAny]) -> bool:
    """ Check if a node has a qconfig of None, i.e. user requested to not quantize
    the node
    """
    return isinstance(node, Node) and node.name in qconfig_map and qconfig_map[node.name] is None

def run_weight_observers(observed: GraphModule) -> None:
    """ Extract the subgraph that produces the weight for dynamic quant
    or weight only quant node and run the subgraph to observe the weight.
    Note that the observers of dynamic quant or weight only quant ops are
    run during the convert step.
    """
    for node in observed.graph.nodes:
        if node.op != 'call_function' or node.target not in WEIGHT_INDEX_DICT:
            continue
        for i, node_arg in enumerate(node.args):
            if i not in WEIGHT_INDEX_DICT[node.target]:
                continue
            # node_arg is weight
            weight_observer_nodes = collect_producer_nodes(node_arg)
            if weight_observer_nodes is None:
                continue
            weight_observer_module = \
                graph_module_from_producer_nodes(
                    observed, weight_observer_nodes)
            # run the weight observer
            weight_observer_module()

def duplicate_dequantize_node(quantized: QuantizedGraphModule) -> QuantizedGraphModule:
    """
    If a dequantize node has multiple uses, duplicate it and create one dequantize node for each use.
    This is to enable the pattern matching to map from individual quant - dequant - ref_module to
    final quantized module.
    """
    quantized_root = quantized
    for node in quantized.graph.nodes:
        if (node.op == "call_method" and node.target == "dequantize" or
           (node.op == "call_function" and node.target == torch.dequantize)):
            users = list(node.users)
            if len(users) > 1:
                for user in users:
                    with quantized.graph.inserting_before(node):
                        new_node = quantized.graph.create_node("call_method", "dequantize", node.args, {})
                    user.replace_input_with(node, new_node)
                quantized.graph.erase_node(node)

    quantized = QuantizedGraphModule(quantized_root, quantized.graph, quantized_root.preserved_attr_names)
    return quantized

def remove_extra_dequantize(quantized: QuantizedGraphModule) -> QuantizedGraphModule:
    """
    Removes duplicate dequant nodes in the graph, for an operator that has multiple dequant nodes as a user,
    replace them with a single dequant node that can be shared across all the uses.
    """
    quantized_root = quantized
    for node in quantized.graph.nodes:
        users = list(node.users)
        dequant_users = [user for user in node.users if user.op == "call_method" and user.target == "dequantize" or
                         (user.op == "call_function" and user.target == torch.dequantize)]

        if len(dequant_users) > 1:
            with quantized.graph.inserting_after(node):
                unique_dq = quantized.graph.create_node("call_method", "dequantize", users[0].args, {})
            for dequant in dequant_users:
                dequant.replace_all_uses_with(unique_dq)
                quantized.graph.erase_node(dequant)

    quantized = QuantizedGraphModule(quantized_root, quantized.graph, quantized_root.preserved_attr_names)
    return quantized

def remove_quant_dequant_pairs(quantized: QuantizedGraphModule) -> QuantizedGraphModule:
    quantized_root = quantized
    for node in quantized.graph.nodes:
        if node.op == "call_function" and node.target in [torch.quantize_per_tensor, torch.quantize_per_channel]:
            users = list(node.users)
            user = users[0] if users else None
            if len(users) == 1 and user.op == "call_method" and user.target == "dequantize":
                user.replace_all_uses_with(node.args[0])
                quantized.graph.erase_node(user)
                orig_args = list(node.args)
                quantized.graph.erase_node(node)
                for arg in orig_args:
                    if isinstance(arg, Node) and len(list(arg.users)) == 0:
                        quantized.graph.erase_node(arg)

    quantized = QuantizedGraphModule(quantized_root, quantized.graph, quantized_root.preserved_attr_names)
    return quantized

def get_module_path_and_prefix(
        obs_node: Node,
        node_name_to_scope: Dict[str, Tuple[str, type]],
        qconfig_map: Dict[str, QConfigAny]):
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
    is_input_observer_only = qconfig_map[observed_node.name] is None if observed_node.name in qconfig_map else None
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

def insert_dequantize_node(
        node: Node,
        graph: Graph):
    """ Inserts dequantize node for `node` in `graph`
    """
    with graph.inserting_after(node):
        dequantize_node = graph.call_method("dequantize", (node,))
        for user_node in dict(node.users):
            if user_node is not dequantize_node:
                user_node.replace_input_with(node, dequantize_node)

def maybe_get_observer_for_node(
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

def convert_standalone_module(
        node: Node,
        modules: Dict[str, torch.nn.Module],
        model: torch.fx.GraphModule,
        is_reference: bool,
        backend_config_dict: Optional[Dict[str, Any]]):
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
      - backend_config_dict: backend configuration of the target backend of quantization
    """
    convert = torch.ao.quantization.quantize_fx.convert_fx  # type: ignore[attr-defined]
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
        insert_dequantize_node(node, model.graph)

    # TODO: allow convert_custom_config_dict to override backend_config_dict
    # for standalone module
    # TODO: think about how to handle `is_reference` here
    quantized_standalone_module = convert(
        observed_standalone_module,
        is_reference=is_reference,
        backend_config_dict=backend_config_dict)
    parent_name, name = _parent_name(node.target)
    # update the modules dict
    setattr(modules[parent_name], name, quantized_standalone_module)
    modules[str(node.target)] = quantized_standalone_module

def convert_weighted_module(
        node: Node,
        modules: Dict[str, torch.nn.Module],
        observed_node_names: Set[str],
        quantized_reference_module_mapping: Dict[Callable, Any],
        qconfig_map: Dict[str, QConfigAny]):
    """ Convert a weighted module to reference quantized module in the model
    If the QConfig of a QAT module is not set, the module will still be converted to
    a float module.

    Args:
      - node: The call_module node of the observed standalone module
      - modules: named_module of original model
      - observed_node_names: names for the set of observed fx node, we can skip
        this conversion if the node is not observed
      - quantized_reference_module_mapping: module mapping from floating point module class
        to quantized reference module class, e.g. nn.Conv2d to nn.quantized._reference.Conv2d
    """
    original_module = modules[str(node.target)]
    float_module = original_module
    weight_post_process = None

    if isinstance(
            original_module,
            QAT_MODULE_CLASSES):
        # Converting qat module to a float module, we need to attch
        # weight fake_quant to the module, weight fake_quant is assumed to be run during
        # QAT so we don't need to run it again here
        float_module = original_module.to_float()  # type: ignore[operator]
        # change qat module to float module
        parent_name, name = _parent_name(node.target)
        setattr(modules[parent_name], name, float_module)
        weight_post_process = original_module.weight_fake_quant

    qconfig = original_module.qconfig
    is_observed = node.name in observed_node_names
    # If a qconfig is not defined for this node, then skip converting to a reference module
    if qconfig is None or has_none_qconfig(node, qconfig_map) or not is_observed:
        return

    # TODO: rename weight_is_statically_quantized to weight_is_int8_quantized
    is_weight_quantized = weight_is_quantized(qconfig)
    quant_type = get_quant_type(qconfig)

    # skip reference module swapping for embedding when quantization mode does not
    # match
    # TODO: we need a more systematic way to handle this after we migrate to use
    # backend_config_dict everywhere
    if isinstance(original_module, WEIGHT_ONLY_MODULE_CLASSES) and \
       quant_type != QuantType.WEIGHT_ONLY:
        return

    if isinstance(original_module, DYNAMIC_MODULE_CLASSES) and \
       quant_type != QuantType.DYNAMIC:
        return

    # the condition for swapping the module to reference quantized module is:
    # weights need to be quantized
    if not is_weight_quantized:
        return

    fused_module = None
    # extract the inidividual float_module and fused module
    if isinstance(float_module, torch.nn.intrinsic._FusedModule):
        fused_module = float_module
        float_module = fused_module[0]  # type: ignore[index]

    # TODO: expose this through backend_config_dict
    # weight_qparams or weight_qparams dict
    wq_or_wq_dict = {}
    if isinstance(float_module, torch.nn.RNNCellBase):
        weight_post_process_ih = qconfig.weight()  # type: ignore[union-attr, operator]
        weight_post_process_hh = qconfig.weight()  # type: ignore[union-attr, operator]
        weight_post_process_ih(float_module.weight_ih)
        weight_post_process_hh(float_module.weight_hh)
        weight_qparams_ih = get_qparam_dict(weight_post_process_ih)
        weight_qparams_hh = get_qparam_dict(weight_post_process_hh)
        wq_or_wq_dict = {
            "weight_ih": weight_qparams_ih,
            "weight_hh": weight_qparams_hh,
        }
    elif isinstance(float_module, torch.nn.LSTM):
        # format for wq_or_wq_dict (flattened attributes):
        # {"weight_ih_l0_scale": ..., "weight_ih_l0_qscheme": ..., ...}
        for wn in float_module._flat_weights_names:
            if hasattr(float_module, wn) and wn.startswith("weight"):
                weight = getattr(float_module, wn)
                weight_post_process = qconfig.weight()  # type: ignore[union-attr, operator]
                if weight_post_process.dtype == torch.qint8:
                    weight_post_process(weight)
                wq_or_wq_dict[wn] = get_qparam_dict(weight_post_process)
    else:
        # weight_post_process is None means the original module is not a QAT module
        # we need to get weight_post_process from qconfig in this case
        if weight_post_process is None:
            weight_post_process = qconfig.weight()  # type: ignore[union-attr, operator]
        # run weight observer
        # TODO: This is currently a hack for QAT to get the right shapes for scale and zero point.
        # In the future, we should require the user to calibrate the model after calling prepare
        # Issue: https://github.com/pytorch/pytorch/issues/73941
        weight_post_process(float_module.weight)  # type: ignore[operator]
        wq_or_wq_dict = get_qparam_dict(weight_post_process)

    # We use the same reference module for all modes of quantization: static, dynamic, weight_only
    ref_qmodule_cls = quantized_reference_module_mapping.get(type(float_module), None)
    assert ref_qmodule_cls is not None, f"No reference quantized module class configured for {type(float_module)}"
    ref_qmodule = ref_qmodule_cls.from_float(float_module, wq_or_wq_dict)  # type: ignore[attr-defined]
    if fused_module is not None:
        fused_module[0] = ref_qmodule
    else:
        parent_name, name = _parent_name(node.target)
        setattr(modules[parent_name], name, ref_qmodule)

def convert_custom_module(
        node: Node,
        graph: Graph,
        modules: Dict[str, torch.nn.Module],
        custom_module_class_mapping: Dict[Callable, Callable],
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
    maybe_obs = maybe_get_observer_for_node(node, modules)
    qconfig = observed_custom_module.qconfig
    if activation_is_statically_quantized(qconfig):
        statically_quantized_custom_module_nodes.add(node)
        # remove the previous dequant node
        prev_node = node.args[0]
        # expecting the input node for a custom module node to be a Node
        assert isinstance(prev_node, Node), \
            f"Expecting the argument for custom module node to be a Node, but got {prev_node}"
        if prev_node.op == "call_method" and prev_node.target == "dequantize":
            assert len(prev_node.users) == 1, "dequantize node before custom module is used "
            "multiple times, this is currently not supported yet, but it can be "
            "supported by duplicating the dequantize nodes in these cases"
            prev_node.replace_all_uses_with(prev_node.args[0])
            graph.erase_node(prev_node)

        # absorb the following observer into the module conversion
        activation_post_process = maybe_get_observer_for_node(node, modules)
        assert activation_post_process is not None
        observed_custom_module.activation_post_process = activation_post_process

    # swap the observed custom module to quantized custom module
    quantized_custom_module_class = get_swapped_custom_module_class(
        observed_custom_module, custom_module_class_mapping, qconfig)
    quantized_custom_module = \
        quantized_custom_module_class.from_observed(observed_custom_module)
    parent_name, name = _parent_name(node.target)
    setattr(modules[parent_name], name, quantized_custom_module)

def _convert_do_not_use(
        model: GraphModule, is_reference: bool = False,
        convert_custom_config_dict: Dict[str, Any] = None,
        is_standalone_module: bool = False,
        _remove_qconfig_flag: bool = True,
        convert_qconfig_dict: Dict[str, Any] = None,
        backend_config_dict: Optional[Dict[str, Any]] = None) -> torch.nn.Module:
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

    standalone_module means it a submodule that is not inlined in
    parent module, and will be quantized separately as one unit.

    Returns a quantized standalone module, whether input/output is quantized is
    specified by prepare_custom_config_dict, with
    input_quantized_idxs, output_quantized_idxs, please
    see docs for prepare_fx for details
    """
    if convert_custom_config_dict is None:
        convert_custom_config_dict = {}
    patterns, node_name_to_scope, prepare_custom_config_dict, observed_node_names = restore_state(model)
    qconfig_map: Dict[str, QConfigAny] = model._qconfig_map  # type: ignore[assignment]

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
    if convert_qconfig_dict:
        prepare_qconfig_dict: Dict[str, Dict[Any, Any]] = model._qconfig_dict  # type: ignore[assignment]
        modules_copy = copy.deepcopy(modules)
        convert_dict_to_ordered_dict(convert_qconfig_dict)
        if model._is_qat:
            additional_qat_module_mapping = prepare_custom_config_dict.get(
                "additional_qat_module_mapping", {})
            convert_qconfig_dict = update_qconfig_for_qat(convert_qconfig_dict, additional_qat_module_mapping)
        convert_qconfig_dict = update_qconfig_for_fusion(model, convert_qconfig_dict)

        compare_prepare_convert_qconfig_dict(prepare_qconfig_dict, convert_qconfig_dict)  # type: ignore[arg-type]
        convert_qconfig_map = generate_qconfig_map(model, modules_copy, model.graph, convert_qconfig_dict, node_name_to_scope)
        # check the convert_qconfig_map generated and ensure that all the values either match what was set in prepare qconfig_map
        # or are set to None in the convert_qconfig_map.
        for k, v in qconfig_map.items():
            assert k in convert_qconfig_map, 'Expected key {} in convert qconfig_map'.format(k)
            if convert_qconfig_map[k] is not None:
                assert qconfig_equals(v, convert_qconfig_map[k]), 'Expected k {} to have the same value in prepare qconfig_dict \
                and convert qconfig_dict, found {} updated to {}.'.format(k, v, convert_qconfig_map[k])
        qconfig_map = convert_qconfig_map

    custom_module_classes = get_custom_module_class_keys(
        convert_custom_config_dict,
        "observed_to_quantized_custom_module_class")
    custom_module_class_mapping = convert_custom_config_dict.get("observed_to_quantized_custom_module_class", {})

    if model._equalization_qconfig_map is not None:
        # If we want to do equalization then do the following:
        # Calculate the equalization scale, update the observers with the scaled
        # inputs, and scale the weight
        weight_eq_obs_dict = update_obs_for_equalization(model, modules)
        convert_eq_obs(model, modules, weight_eq_obs_dict)

    # always run weight observers in the top level forward method
    # for dynamic quant ops or weight only quant ops
    run_weight_observers(model)

    graph_inputs: List[str] = []
    for node in model.graph.nodes:
        if node.op == 'placeholder':
            graph_inputs.append(node.name)

    # TODO: move this outside of this function
    def replace_observer_with_quantize_dequantize_node(
            model: torch.nn.Module,
            graph: Graph,
            node: Node,
            modules: Dict[str, torch.nn.Module],
            node_name_to_scope: Dict[str, Tuple[str, type]],
            qconfig_map: Dict[str, QConfigAny]) -> None:
        """ Replace activation_post_process module call node with quantize and
        dequantize node

        Before:
        ... -> observer_0(x) -> ...
        After:
        ... -> torch.quantize_per_tensor(x, ...) -> x.dequantize() -> ...
        """
        assert modules is not None
        assert isinstance(node.target, str)
        module_path, prefix = get_module_path_and_prefix(node, node_name_to_scope, qconfig_map)
        observer_module = modules[node.target]
        maybe_quantize_node_info = get_quantize_node_info(observer_module)
        # Skip replacing observers to quant/dequant nodes if the qconfigs of all
        # consumers and producers of this observer are None
        skip_replacement = all([
            has_none_qconfig(n, qconfig_map) for n in
            list(node.args) + list(node.users.keys())])
        if skip_replacement or maybe_quantize_node_info is None:
            # didn't find correponding quantize op and info for the observer_module
            # so we just remove the observer
            with graph.inserting_before(node):
                node.replace_all_uses_with(node.args[0])
                graph.erase_node(node)
        else:
            # otherwise, we can convert the observer moduel call to quantize/dequantize node
            node_type, quantize_op, qparams = maybe_quantize_node_info
            # replace observer node with quant - dequant node
            with graph.inserting_before(node):
                input_node = node.args[0]
                inputs = [input_node]
                for key, value in qparams.items():
                    # TODO: we can add the information of whether a value needs to
                    # be registered as an attribute in qparams dict itself
                    if key in ['_scale_', '_zero_point_']:
                        # For scale and zero_point values we register them as buffers in the root module.
                        # TODO: maybe need more complex attr name here
                        qparam_node = create_getattr_from_value(model, graph, module_path + prefix + key, value)
                        inputs.append(qparam_node)
                    else:
                        # for qparams that are not scale/zero_point (like axis, dtype) we store them as literals in the graph.
                        inputs.append(value)

                quantized_node = graph.create_node(node_type, quantize_op, tuple(inputs), {})
                dequantized_node = graph.call_method("dequantize", args=(quantized_node,))
                node.replace_all_uses_with(dequantized_node)
                graph.erase_node(node)

    # this is a temporary hack for custom module, we may want to implement
    # this properly after the custom module class design is finalized
    def replace_observer_with_dequantize_node(node: Node, graph: Graph):
        call_custom_module_node = node.args[0]
        assert isinstance(call_custom_module_node, Node), \
            f"Expecting the for call custom module node to be a Node, but got {call_custom_module_node}"
        node.replace_all_uses_with(call_custom_module_node)
        graph.erase_node(node)
        insert_dequantize_node(call_custom_module_node, graph)

    # additional state to override inputs to be quantized, if specified
    # by the user
    placeholder_node_seen_cnt = 0
    output_node_seen_cnt = 0
    input_quantized_idxs: List[int] = prepare_custom_config_dict.get(
        "input_quantized_idxs", [])
    output_quantized_idxs: List[int] = prepare_custom_config_dict.get(
        "output_quantized_idxs", [])

    if backend_config_dict is None:
        quantized_reference_module_mapping = copy.deepcopy(DEFAULT_REFERENCE_STATIC_QUANT_MODULE_MAPPINGS)
    else:
        quantized_reference_module_mapping = get_quantized_reference_module_mapping(backend_config_dict)
    # convert tuples so that it can work with isinstance(module, tuple_of_classes)
    weighted_module_classes = tuple(quantized_reference_module_mapping.keys())
    statically_quantized_custom_module_nodes: Set[Node] = set()

    for node in list(model.graph.nodes):
        if node.op == 'placeholder':
            cur_placeholder_node_idx = placeholder_node_seen_cnt
            placeholder_node_seen_cnt += 1
            if cur_placeholder_node_idx in input_quantized_idxs:
                # Inputs are assumed to be quantized if the user specifid the
                # input_quantized_idxs override.
                # we need to dequantize the inputs since all operators took
                # floating point inputs in reference quantized models
                insert_dequantize_node(node, model.graph)
        elif node.op == "output":
            cur_output_node_idx = output_node_seen_cnt
            output_node_seen_cnt += 1
            if cur_output_node_idx in output_quantized_idxs:
                # Result are kept quantized if the user specified the
                # output_quantized_idxs override.
                # Remove the dequantize operator in the end
                maybe_dequantize_node = node.args[0]
                if isinstance(maybe_dequantize_node, Node) and \
                   maybe_dequantize_node.op == "call_method" and \
                   maybe_dequantize_node.target == "dequantize":
                    quantize_node = maybe_dequantize_node.args[0]
                    maybe_dequantize_node.replace_all_uses_with(quantize_node)
                    model.graph.erase_node(maybe_dequantize_node)
        elif node.op == "call_module":
            if is_activation_post_process(modules[node.target]):
                observed_node = node.args[0]
                if observed_node in statically_quantized_custom_module_nodes:
                    replace_observer_with_dequantize_node(node, model.graph)
                else:
                    replace_observer_with_quantize_dequantize_node(
                        model, model.graph, node, modules, node_name_to_scope,
                        qconfig_map)
            elif is_observed_standalone_module(modules[node.target]):
                convert_standalone_module(
                    node, modules, model, is_reference, backend_config_dict)
            elif type(modules[node.target]) in set(
                    weighted_module_classes).union(QAT_MODULE_CLASSES).union(FUSED_MODULE_CLASSES):
                convert_weighted_module(
                    node, modules, observed_node_names, quantized_reference_module_mapping, qconfig_map)
            elif type(modules[node.target]) in custom_module_classes:
                convert_custom_module(
                    node, model.graph, modules, custom_module_class_mapping,
                    statically_quantized_custom_module_nodes)

    preserved_attributes = set(convert_custom_config_dict.get("preserved_attributes", []))
    model = QuantizedGraphModule(model, model.graph, preserved_attributes)
    # TODO: maybe move this to quantize_fx.py
    if not is_reference:
        model = duplicate_dequantize_node(model)
        model = lower_to_fbgemm(model, qconfig_map, node_name_to_scope)
        model = remove_quant_dequant_pairs(model)
        model = remove_extra_dequantize(model)
    # TODO: this looks hacky, we want to check why we need this and see if we can
    # remove this
    # removes qconfig and activation_post_process modules
    if _remove_qconfig_flag:
        _remove_qconfig(model)
    return model
