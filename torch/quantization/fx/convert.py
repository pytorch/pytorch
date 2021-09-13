from typing import Any, Dict, Tuple, List, Callable, Optional, Union
from collections import defaultdict
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
from .quantization_types import Pattern
from ..qconfig import QConfigAny
from .match_utils import (
    find_matches,
)
from .graph_module import (
    is_observed_module,
    is_observed_standalone_module,
    QuantizedGraphModule,
)
from .quantization_patterns import (
    QuantizeHandler,
)
from ._equalize import update_obs_for_equalization, convert_eq_obs
from .utils import (
    is_get_tensor_info_node,
    node_return_type_is_int,
    quantize_node,
    get_new_attr_name_with_prefix,
    collect_producer_nodes,
    graph_module_from_producer_nodes,
    get_custom_module_class_keys,
    WEIGHT_INDEX_DICT,
)

from torch.ao.quantization.quantize import (
    _remove_qconfig,
    is_activation_post_process,
)
from ..utils import (
    activation_is_statically_quantized,
    activation_dtype,
)

from .lower_to_fbgemm import lower_to_fbgemm

# weight prepacking ops
WEIGHT_PREPACK_OPS = {
    torch._ops.ops.quantized.linear_prepack,
    torch._ops.ops.quantized.linear_prepack_fp16,
    torch._ops.ops.quantized.conv1d_prepack,
    torch._ops.ops.quantized.conv2d_prepack,
    torch._ops.ops.quantized.conv3d_prepack,
}

def run_weight_observers(observed: GraphModule) -> None:
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

def fold_weight(
        quantized: QuantizedGraphModule,
        node_name_to_scope: Dict[str, Tuple[str, type]]) -> QuantizedGraphModule:
    """
    Trace back from the weight node util we hit getattr, reconstruct the
    graph module with the traced nodes and run the graph module to pack the
    weight. then replace the original chain of ops with the packed weight.
    """
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
            module_path, _ = node_name_to_scope[op_node.name]
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

def restore_state(
        observed: GraphModule
) -> Tuple[Dict[Pattern, QuantizeHandler], Dict[str, Tuple[str, type]], Dict[str, Any]]:
    assert is_observed_module(observed), \
        'incoming model must be produced by prepare_fx'
    prepare_custom_config_dict: Dict[str, Any] = \
        observed._prepare_custom_config_dict  # type: ignore[assignment]
    node_name_to_scope: Dict[str, Tuple[str, type]] = observed._node_name_to_scope  # type: ignore[assignment]
    patterns: Dict[Pattern, QuantizeHandler] = observed._patterns  # type: ignore[assignment]
    return patterns, node_name_to_scope, prepare_custom_config_dict

def convert(model: GraphModule, is_reference: bool = False,
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
    patterns, node_name_to_scope, prepare_custom_config_dict = restore_state(model)
    qconfig_map: Dict[str, QConfigAny] = model._qconfig_map  # type: ignore[assignment]

    # TODO this should be removed now that gpu support for quantization is being supported.
    # however in practice, as of 7/22/2021, certain functions that get called by convert expect
    # only cpu arguments.
    # As an example, in TestQuantizeFxModels.test_qat_functional_linear when device='cuda',
    # fold_weight will call quantized::linear_prepack which doesn't support QuantizedCuda backend.
    if not is_reference:
        model.cpu()

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

    custom_module_classes = get_custom_module_class_keys(
        convert_custom_config_dict,
        "observed_to_quantized_custom_module_class")
    matches = find_matches(
        model.graph, modules, patterns,
        qconfig_map,
        custom_module_classes=custom_module_classes)

    if model._equalization_qconfig_map is not None:
        # If we want to do equalization then do the following:
        # Calculate the equalization scale, update the observers with the scaled
        # inputs, and scale the weight
        weight_eq_obs_dict = update_obs_for_equalization(model, modules)
        convert_eq_obs(model, modules, weight_eq_obs_dict)

    # always run weight observers in the top level forward method
    # for dynamic quant ops or weight only quant ops
    run_weight_observers(model)

    quantized_graph = Graph()
    env: Dict[str, Dict[Optional[torch.dtype], Node]] = defaultdict(lambda: defaultdict(Node))  # type: ignore[arg-type]

    graph_inputs: List[str] = []
    for node in model.graph.nodes:
        if node.op == 'placeholder':
            graph_inputs.append(node.name)

    def load_non_quantized(n: Node) -> Node:
        assert n.name in env, \
            'trying to load float node but did not find ' + \
            'node:' + n.name + \
            ' in env: ' + \
            str(env)
        dtype_to_node = env[n.name]
        if torch.float in dtype_to_node:
            return dtype_to_node[torch.float]
        elif None in dtype_to_node:
            return dtype_to_node[None]
        else:
            quantized_node = None
            for dtype in [torch.quint8, torch.qint8, torch.float16]:
                if dtype in dtype_to_node:
                    quantized_node = dtype_to_node[dtype]
                    break
            assert quantized_node is not None, "Did not find a supported quantized dtype:{}".format(dtype_to_node)
            env[n.name][torch.float] = Proxy(quantized_node).dequantize().node
            return env[n.name][torch.float]

    def load_quantized(dtype: torch.dtype):
        def load_quantized_impl(n: Node):
            assert n.name in env, \
                'trying to load quantized node but did not find node:' + \
                n.name + ' in environment:' + str(env)
            dtype_to_node = env[n.name]
            local_dtype : Optional[torch.dtype] = dtype
            if local_dtype == torch.float and local_dtype not in dtype_to_node:
                local_dtype = None
            if local_dtype in [torch.float, None]:
                return load_non_quantized(n)
            assert local_dtype in dtype_to_node, f'Expecting {dtype} in {dtype_to_node}'
            return dtype_to_node[local_dtype]

        return load_quantized_impl

    def load_x(n: Node) -> Node:
        assert n.name in env, \
            'node ' + n.name + ' does not exist in environment'
        dtype_to_node = env[n.name]
        dtypes = [torch.quint8, torch.qint8, torch.float16, torch.float32, None]
        for dtype in dtypes:
            if dtype in dtype_to_node:
                return dtype_to_node[dtype]
        raise Exception(f'dtype {dtype} not found in environment: {dtype_to_node} for node {n.name}')

    def load_arg(
            quantized: Optional[Union[List[int], Dict[int, torch.dtype], torch.dtype, Tuple[int, ...]]]
    ) -> Callable[[Node], Argument]:
        """
        Input: quantized, which can be None, torch.dtype, list or tuple
          - if quantized is None, then we'll load the node as long as it
            exists
          - if quantized is a dtype, then all args will be
            quantized to the specific dtype
          - if quantized is an empty list or tuple, then it is the same as load_arg(quantized=torch.float)
          - if quantized is a list or tuple, then arg should be a list and
            the args with corresponding indexes will be quantized to torch.quint8


        Output: fn which takes arg_or_args, and loads them from the
            corresponding environment depending on the value of quantized.
        """
        assert quantized is None or \
            isinstance(quantized, (tuple, list, dict, torch.dtype)), type(quantized)
        if isinstance(quantized, (tuple, list, dict)) and len(quantized) == 0:
            # empty tuple or list means nothing is quantized
            quantized = torch.float

        def load_arg_impl(arg_or_args):
            # we'll update the format of `quantized`
            # to better match arg_or_args
            updated_quantized: Optional[Union[List[int], torch.dtype, Dict[int, torch.dtype], Tuple[int, ...]]] = quantized

            if isinstance(quantized, (tuple, list)) and \
               len(quantized) == 1 and isinstance(arg_or_args, Node):
                # when argument is one Node instead of tuple, we just need to check
                # 0 is in the quantized list
                if 0 in quantized:
                    updated_quantized = torch.quint8

            if updated_quantized is None:
                return map_arg(arg_or_args, load_x)
            if isinstance(updated_quantized, torch.dtype):
                return map_arg(
                    arg_or_args,
                    load_quantized(updated_quantized))
            elif isinstance(updated_quantized, (tuple, list)):
                assert isinstance(arg_or_args, (tuple, list)), arg_or_args
                loaded_args = []
                # for now, we only support quantizing positional arguments
                for i, a in enumerate(arg_or_args):
                    if i in updated_quantized:
                        # Currently it's hardcoded to torch.quint8, we can extend this
                        # in the future to support all quantized
                        # dtypes
                        loaded_args.append(map_arg(a, load_quantized(torch.quint8)))
                    else:
                        loaded_args.append(map_arg(a, load_non_quantized))
                return type(arg_or_args)(loaded_args)
            elif isinstance(updated_quantized, dict):
                loaded_args = []
                for i, a in enumerate(arg_or_args):
                    if i in updated_quantized:
                        loaded_args.append(map_arg(a, load_quantized(updated_quantized[i])))
                    else:
                        loaded_args.append(map_arg(a, load_non_quantized))
                return type(arg_or_args)(loaded_args)
        return load_arg_impl

    def node_arg_is_quantized(node_arg: Any) -> bool:
        if isinstance(node_arg, Node):
            assert node_arg.name in env, \
                'Expecting node_arg to be in the environment'
            if node_arg.name in env:
                dtype_to_node = env[node_arg.name]
                return any([x in dtype_to_node for x in [torch.quint8, torch.qint8, torch.float16]])
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

    def is_output_quantized(
            node: Node, obj: QuantizeHandler, qconfig: QConfigAny,
            modules: Dict[str, torch.nn.Module], is_reference=False) -> bool:
        """ Check if output node is quantized or not """
        assert modules is not None
        # for some ops the output is quantized only when `is_reference` is True
        # and when `is_reference` is False, it has limited qconfig
        # support, for example `add`
        # ideally this check should not happen here, it should happen either in
        # prepare or during lowering, we don't need this check
        # after the default path is changed to produce reference patterns
        quantized = obj.is_output_quantized(qconfig, is_reference)

        # Need to get correct quantized/non-quantized state forn the output
        # of FixedQParamsQuantizeHandler
        # TODO: we may want to try to remove the special case here
        # as well
        if obj.should_mark_output_quantized_from_input_quantized_status(qconfig):
            assert node.op in [
                'call_module',
                'call_function',
                'call_method'], \
                'FixedQParamsQuantizeHandler of type ' + node.op + ' is not handled'
            # TODO: need to extend this to consider all relevant args instead of just arg[0]
            quantized = node_arg_is_quantized(node.args[0])

        # the output is unquantized if the node is not a CopyNode
        # or the activation is not statically quantized
        if not activation_is_statically_quantized(qconfig) or \
           not obj.input_output_observed():
            quantized = False
        if node_return_type_is_int(node):
            quantized = False

        return quantized

    def insert_quantize_node(node: Node, modules: Dict[str, torch.nn.Module]) -> None:
        """ Given a activation_post_process module call node, insert a
        quantize node"""
        assert modules is not None
        assert isinstance(node.target, str)
        observer_module = modules[node.target]
        prev_node = node.args[0]
        if observer_module.dtype == torch.float32:
            # copy the observer for fp32 dtype
            env[node.name][torch.float] = quantized_graph.node_copy(
                node, load_non_quantized)
        elif isinstance(prev_node, Node) and prev_node.name in env:
            # if previous node is already quantized, we'll just remove the
            # activation_post_process
            prev_dtype_to_node: Dict[Optional[torch.dtype], Node] = env[prev_node.name]
            current_dtype: Optional[torch.dtype] = observer_module.dtype  # type: ignore[assignment]
            if current_dtype in prev_dtype_to_node:
                env[node.name][current_dtype] = prev_dtype_to_node[current_dtype]
            else:
                root_module = modules[""]
                assert isinstance(prev_node, Node)
                observer_dtype: torch.dtype = observer_module.dtype  # type: ignore[assignment]
                env[node.name][observer_dtype] = \
                    quantize_node(
                        load_non_quantized(prev_node),
                        observer_module, node, modules, quantized_graph,
                        node_name_to_scope, is_input=True)
        else:
            # replace activation post process with quantization ops
            root_module = modules[""]
            assert isinstance(node.args[0], Node)
            dtype: torch.dtype = observer_module.dtype  # type: ignore[assignment]
            env[node.name][dtype] = \
                quantize_node(
                    load_non_quantized(node.args[0]),
                    observer_module, node, modules,
                    quantized_graph,
                    node_name_to_scope, is_input=True)

    # additional state to override inputs to be quantized, if specified
    # by the user
    placeholder_node_seen_cnt = 0
    output_node_seen_cnt = 0
    input_quantized_idxs: List[int] = prepare_custom_config_dict.get(
        "input_quantized_idxs", [])
    output_quantized_idxs: List[int] = prepare_custom_config_dict.get(
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
            quantized_graph.output(graph_output)
            continue
        root_node, matched, matched_pattern, obj, qconfig = \
            matches.get(node.name, (None, None, None, None, None))
        if root_node is node:
            is_observed_standalone_module_node = (
                node.op == 'call_module' and
                is_observed_standalone_module(
                    modules[node.target])
            )
            if qconfig is None and not is_observed_standalone_module_node:
                result = quantized_graph.node_copy(
                    node, load_non_quantized)
                quantized = False
            else:
                assert obj is not None
                # We will get whether the output is quantized or not before
                # convert for standalone module and after convert
                # for non-standalone module, since _standalone_module_output_quantized_idxs
                # is only available in observed standalone module
                if is_observed_standalone_module_node:
                    out_quant_idxs = modules[node.target]._standalone_module_output_quantized_idxs.tolist()  # noqa: B950
                    assert len(out_quant_idxs) <= 1, "Currently standalone only support one output"
                    quantized = 0 in out_quant_idxs

                qconfig = qconfig_map[node.name]
                # Note: load_arg can be overwritten in the convert method when used to
                # create Node in graph
                result = obj.convert(
                    node, qconfig, modules, quantized_graph, node_name_to_scope, load_arg, is_reference=is_reference,
                    convert_custom_config_dict=convert_custom_config_dict)
                if not is_observed_standalone_module_node:
                    quantized = is_output_quantized(node, obj, qconfig, modules, is_reference)

            if quantized:
                env[node.name][activation_dtype(qconfig)] = result
            else:
                env[node.name][torch.float] = result
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
                result = quantized_graph.node_copy(
                    node, load_non_quantized)
                env[node.name][torch.float] = result
            continue

        # handle activation post process calls
        if node.op == 'call_module' and \
                is_activation_post_process(modules[node.target]):
            insert_quantize_node(node, modules)
        elif node.op == 'placeholder':
            cur_placeholder_node_idx = placeholder_node_seen_cnt
            placeholder_node_seen_cnt += 1
            if cur_placeholder_node_idx in input_quantized_idxs:
                env[node.name][torch.quint8] = quantized_graph.node_copy(
                    node, load_non_quantized)
            else:
                env[node.name][torch.float] = \
                    quantized_graph.node_copy(node, load_non_quantized)
        else:
            # copy quantized or non-quantized node
            # get_tensor_info_node like shape works for both
            # quantized and non-quantized input and output a non-Tensor
            # (we use None for dtype currently for non-Tensors)
            if is_get_tensor_info_node(node):
                env[node.name][None] = \
                    quantized_graph.node_copy(node, load_x)
            else:
                env[node.name][torch.float] = \
                    quantized_graph.node_copy(node, load_non_quantized)

    # remove activation post process
    act_post_process_removed_graph = Graph()
    remove_env: Dict[str, Node] = {}

    def load_arg_remove(a: Argument) -> Argument:
        return map_arg(a, lambda node: remove_env[node.name])

    for node in quantized_graph.nodes:
        if node.op == 'output':
            act_post_process_removed_graph.output(
                map_arg(node.args[0], load_arg_remove))
            continue
        if node.op == 'call_module' and \
           is_activation_post_process(modules[node.target]):
            # remove activation post process node
            remove_env[node.name] = remove_env[node.args[0].name]
        else:
            remove_env[node.name] = act_post_process_removed_graph.node_copy(
                node, load_arg_remove)

    # removes qconfig and activation_post_process modules
    if _remove_qconfig_flag:
        _remove_qconfig(model)
    preserved_attributes = set(convert_custom_config_dict.get("preserved_attributes", []))
    model = QuantizedGraphModule(model, act_post_process_removed_graph, preserved_attributes)
    if not is_reference:
        model = fold_weight(model, node_name_to_scope)
        model = lower_to_fbgemm(model)
    return model
