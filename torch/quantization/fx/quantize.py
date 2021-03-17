import torch
from torch.fx import (  # type: ignore
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
    get_swapped_custom_module_class,
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
    input_output_observed,
    Pattern,
)

from .graph_module import (
    is_observed_module,
    is_observed_standalone_module,
    ObservedGraphModule,
    ObservedStandaloneGraphModule,
    QuantizedGraphModule,
)

from .quantization_patterns import *

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
)

from .qconfig_utils import *

from typing import Optional, Dict, Any, List, Tuple, Set, Callable

# Define helper types
MatchResult = Tuple[Node, List[Node], Optional[Pattern], QuantizeHandler,
                    QConfigAny]

# ------------------------
# Helper Functions
# ------------------------

def insert_observer(
        node: Node, observer: torch.quantization.ObserverBase,
        model: torch.nn.Module,
        activation_post_process_map: Dict[str, torch.quantization.ObserverBase],
        env: Dict[Any, Any], observed_graph: Graph, load_arg: Callable,
        observed_node_names_set: Set[str]):
    """Insert observer for node by modifying the observed_graph and
       attach observer module to the model
       Args:
         node: Node
         observer: observer/fake_quantize module instance
    """
    # respect device affinity when adding observers
    model_device = assert_and_get_unique_device(model)
    if model_device:
        observer.to(model_device)
    # add observer module as attribute
    prefix = node.name + '_activation_post_process_'
    get_new_observer_name = get_new_attr_name_with_prefix(prefix)
    observer_name = get_new_observer_name(model)
    setattr(model, observer_name, observer)
    # put observer instance activation_post_process map
    assert activation_post_process_map is not None
    activation_post_process_map[node.name] = observer
    # insert observer call
    env[node.name] = observed_graph.create_node(
        'call_module', observer_name, (load_arg(node),), {})
    observed_node_names_set.add(node.name)

def maybe_insert_observer_for_special_module(
        quantize_handler: QuantizeHandler, modules: Dict[str, torch.nn.Module],
        prepare_custom_config_dict: Any, qconfig: Any, node: Node) -> Optional[List[int]]:
    """ Insert observer for custom module and standalone module
      Returns: standalone_module_input_idxs: the indexs for inputs that
      needs to be observed by parent module
    """
    assert modules is not None
    standalone_module_input_idxs = None
    if isinstance(quantize_handler, CustomModuleQuantizeHandler):
        custom_module = modules[node.target]  # type: ignore
        custom_module_class_mapping = prepare_custom_config_dict.get(
            "float_to_observed_custom_module_class", {})
        observed_custom_module_class = \
            get_swapped_custom_module_class(
                custom_module, custom_module_class_mapping, qconfig)
        observed_custom_module = \
            observed_custom_module_class.from_float(custom_module)
        parent_name, name = _parent_name(node.target)
        setattr(modules[parent_name], name, observed_custom_module)
    elif isinstance(quantize_handler, StandaloneModuleQuantizeHandler):
        # observe standalone module
        standalone_module = modules[node.target]  # type: ignore
        standalone_module_name_configs = prepare_custom_config_dict.get("standalone_module_name", [])
        standalone_module_class_configs = prepare_custom_config_dict.get("standalone_module_class", [])
        class_config_map = {x[0]: (x[1], x[2]) for x in standalone_module_class_configs}
        name_config_map = {x[0]: (x[1], x[2]) for x in standalone_module_name_configs}
        config = class_config_map.get(type(standalone_module), (None, None))
        config = name_config_map.get(node.target, config)
        sm_qconfig_dict = {"": qconfig} if config[0] is None else config[0]
        sm_prepare_config_dict = {} if config[1] is None else config[1]
        prepare = \
            torch.quantization.quantize_fx._prepare_standalone_module_fx  # type: ignore
        observed_standalone_module = \
            prepare(standalone_module, sm_qconfig_dict, sm_prepare_config_dict)
        standalone_module_input_idxs = observed_standalone_module.\
            _standalone_module_input_quantized_idxs.int().tolist()
        observed_standalone_module = ObservedStandaloneGraphModule(
            observed_standalone_module, observed_standalone_module.graph)
        parent_name, name = _parent_name(node.target)
        setattr(modules[parent_name], name,
                observed_standalone_module)
        modules[node.target] = observed_standalone_module  # type: ignore
    return standalone_module_input_idxs

def insert_observer_for_output_of_the_node(
        node: Node,
        quantize_handler: QuantizeHandler,
        qconfig: Any,
        modules: Dict[str, torch.nn.Module],
        model: torch.nn.Module,
        pattern: Any,
        activation_post_process_map: Dict[str, torch.quantization.ObserverBase],
        env: Dict[Any, Any],
        observed_graph: Graph,
        load_arg: Callable,
        observed_node_names_set: Set[str],
        matched_nodes: Optional[List[Node]],
        standalone_module_input_idxs: Optional[List[int]]):
    """ Insert observer/fake_quantize module for output of the observed
    module if needed
    """
    # don't need to insert observer for output if activation does not
    # need to be statically quantized
    assert modules is not None
    # TODO: Add warnings in the quantize handlers that does not support fp16 quantization
    if activation_is_statically_quantized(qconfig):
        if isinstance(quantize_handler, FixedQParamsOpQuantizeHandler) \
                and model.training:
            # we only insert fake quantize module in qat
            assert pattern is not None
            if activation_dtype(qconfig) == torch.float16:
                activation_post_process_ctr = qconfig.activation
            else:
                activation_post_process_ctr = \
                    get_default_output_activation_post_process_map().get(
                        pattern, None)
            assert activation_post_process_ctr is not None, \
                "activation_post_process constructor not provided " + \
                "for pattern:" + str(pattern)
            insert_observer(
                node, activation_post_process_ctr(),
                model, activation_post_process_map, env, observed_graph,
                load_arg, observed_node_names_set)
        elif (isinstance(quantize_handler,
                         FixedQParamsOpQuantizeHandler) and
              not model.training) or \
                isinstance(quantize_handler, CopyNodeQuantizeHandler):
            # inserting observers for output of observed module, or
            # mark the output as observed
            assert node.op in [
                'call_module',
                'call_function',
                'call_method'], \
                'CopyNodeQuantizeHandler of type ' + node.op + ' is not handled'

            def is_observed(input_arg):
                if isinstance(input_arg, Node):
                    return input_arg.name in observed_node_names_set
                elif isinstance(input_arg, list):
                    return all(map(is_observed, input_arg))

            # insert observers for fixedqparams ops like sigmoid, since
            # it supports fp16 static quantization
            if isinstance(quantize_handler, FixedQParamsOpQuantizeHandler) and \
               activation_dtype(qconfig) == torch.float16:
                insert_observer(
                    node, qconfig.activation(),
                    model, activation_post_process_map, env, observed_graph,
                    load_arg, observed_node_names_set)
            else:
                # propagate observed property from input
                if is_observed(node.args[0]):
                    observed_node_names_set.add(node.name)
        elif (isinstance(quantize_handler, BinaryOpQuantizeHandler) and
              quantize_handler.num_tensor_args == 1):
            assert matched_nodes is not None
            input_node = matched_nodes[-1]  # first node in the sequence

            def input_is_observed(arg):
                return (isinstance(arg, Node) and
                        arg.name in observed_node_names_set)
            # This is checking if one of the argument of add/mul
            # is an observed node
            # If both of the inputs are number,
            # we will not consider the output to be observed
            if (input_is_observed(input_node.args[0]) or
                    input_is_observed(input_node.args[1])):
                observed_node_names_set.add(node.name)

            if activation_dtype(qconfig) == torch.float16:
                # observer for outputs
                new_observer = qconfig.activation()
                insert_observer(
                    node, new_observer, model,
                    activation_post_process_map, env, observed_graph,
                    load_arg, observed_node_names_set)
        elif isinstance(quantize_handler,
                        StandaloneModuleQuantizeHandler):
            assert node.op == "call_module"
            assert isinstance(node.target, str)
            sm_out_qidxs = modules[node.target]._standalone_module_output_quantized_idxs.tolist()  # type: ignore
            output_is_quantized = 0 in sm_out_qidxs

            if output_is_quantized:
                observed_node_names_set.add(node.name)
        elif (quantize_handler.all_node_args_are_tensors and
              input_output_observed(quantize_handler)):
            # observer for outputs
            new_observer = qconfig.activation()
            insert_observer(
                node, new_observer, model,
                activation_post_process_map, env, observed_graph,
                load_arg, observed_node_names_set)

        # insert observer for input of standalone module
        if standalone_module_input_idxs is not None:
            for idx in standalone_module_input_idxs:
                if node.args[idx].name not in observed_node_names_set:  # type: ignore
                    new_observer = qconfig.activation()
                    insert_observer(
                        node, new_observer, model,
                        activation_post_process_map, env, observed_graph,
                        load_arg, observed_node_names_set)

def insert_observer_for_input_arg_of_observed_node(
        node: Node, observed_node_names_set: Set[str],
        quants: Dict[str, Tuple[DefaultQuantizeHandler, Callable]],
        model: torch.nn.Module,
        activation_post_process_map: Dict[str, torch.quantization.ObserverBase],
        env: Dict[str, str], observed_graph: Graph,
        load_arg: Callable):
    if node.name not in observed_node_names_set and node.name in quants:
        _, activation_post_process_ctr = quants[node.name]
        if activation_post_process_ctr is not None:
            insert_observer(
                node, activation_post_process_ctr(),
                model, activation_post_process_map,
                env, observed_graph, load_arg, observed_node_names_set)

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
                    WEIGHT_INDEX_DICT[node.target]:  # type: ignore
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
    torch._ops.ops.quantized.conv2d_prepack,
    torch._ops.ops.quantized.conv3d_prepack,
}

class Quantizer:
    def __init__(self):
        # mapping from matched node to activation_post_process
        # must be filled before convert
        self.activation_post_process_map: Optional[
            Dict[str, torch.quantization.observer.ObserverBase]] = None
        # mapping from node name to qconfig that should be used for that node
        # filled out for a model during _generate_qconfig_map
        self.qconfig_map: Optional[Dict[str, QConfigAny]] = None
        # mapping from fully qualified module name to module instance
        # for example,
        # {
        #   '': Model(...),
        #   'linear': Linear(...),
        #   'linear.weight_fake_quant': PerChannelMinMaxObserver(...),
        # }
        self.modules: Optional[Dict[str, torch.nn.Module]] = None
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
        self.patterns: Optional[Dict[Pattern, QuantizeHandler]] = None
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
                assert self.modules is not None
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
                assert self.modules is not None
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

        # map from node name to qconfig, used in _find_matches
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
        assert self.patterns is not None
        matches = self._find_matches(
            model.graph, self.modules, self.patterns, standalone_module_names,
            standalone_module_classes, custom_module_classes)

        # find _inputs_ to matched nodes that are not quantized, these
        # have to be quantized, which requires measuring stats,
        # initialize an DefaultQuantizeHandler object for each
        quants: Dict[str, Tuple[DefaultQuantizeHandler, Callable]] = \
            self._find_quants(model.graph, matches)

        self.activation_post_process_map = dict()
        env: Dict[Any, Any] = {}
        observed_graph = Graph()
        observed_node_names_set: Set[str] = set()

        def load_arg(a):
            return map_arg(a, lambda node: env[node.name])

        graph_inputs = []
        for node in model.graph.nodes:
            if node.op == 'placeholder':
                graph_inputs.append(node.name)

        get_new_observer_name = get_new_attr_name_with_prefix(
            'activation_post_process_')

        placeholder_node_seen_cnt = 0
        output_node_seen_cnt = 0
        input_quantized_idxs: List[int] = self.prepare_custom_config_dict.get(
            "input_quantized_idxs", [])
        output_quantized_idxs: List[int] = self.prepare_custom_config_dict.get(
            "output_quantized_idxs", [])

        result_node : Optional[Node] = None
        for node in model.graph.nodes:
            if node.op == 'output':
                # If this output is hardcoded to be quantized, insert an
                # observer on the previous node if it does not already
                # exist.
                cur_output_node_idx = output_node_seen_cnt
                output_node_seen_cnt += 1
                if cur_output_node_idx in output_quantized_idxs:
                    prev_node = node.args[0]
                    assert isinstance(prev_node, Node), \
                        ('hardcoding list/dict outputs to be quantized is ' +
                         'not supported')
                    if prev_node.name not in observed_node_names_set:
                        assert self.qconfig_map is not None
                        local_qconfig = self.qconfig_map[prev_node.name]
                        assert local_qconfig is not None, \
                            'qconfig of a node before a quantized output must exist'
                        insert_observer(
                            prev_node, local_qconfig.activation(),
                            model, self.activation_post_process_map,
                            env, observed_graph, load_arg, observed_node_names_set)

                observed_graph.output(load_arg(node.args[0]))
                result_node = node
                continue

            if node.name in observed_node_names_set:
                continue

            root_node, matched_nodes, pattern, obj, qconfig = matches.get(
                node.name, (None, None, None, None, None))
            if root_node is None:
                env[node.name] = observed_graph.node_copy(node, load_arg)
            elif root_node is node:
                env[node.name] = observed_graph.node_copy(node, load_arg)
                # index for input of custom module that needs to be observed in
                # parent
                if qconfig is not None:
                    assert obj is not None
                    standalone_module_input_idxs = \
                        maybe_insert_observer_for_special_module(
                            obj, self.modules, prepare_custom_config_dict, qconfig,
                            node)
                    insert_observer_for_output_of_the_node(
                        node, obj, qconfig, self.modules, model, pattern,
                        self.activation_post_process_map, env,
                        observed_graph, load_arg, observed_node_names_set,
                        matched_nodes, standalone_module_input_idxs)
            else:
                env[node.name] = observed_graph.node_copy(node, load_arg)

            if node.op == 'placeholder':
                # skip adding observers at the graph input if the input is
                # overriden to be quantized
                cur_placeholder_node_idx = placeholder_node_seen_cnt
                placeholder_node_seen_cnt += 1
                if cur_placeholder_node_idx in input_quantized_idxs:
                    observed_node_names_set.add(node.name)
                    continue

            insert_observer_for_input_arg_of_observed_node(
                node, observed_node_names_set, quants,
                model, self.activation_post_process_map, env,
                observed_graph, load_arg)


        self.save_state(model)
        model = ObservedGraphModule(model, observed_graph)
        if is_standalone_module:
            assert result_node is not None
            assert isinstance(result_node.args[0], Node), \
                "standalone module only supports returning simple value currently"\
                "(not tuple, dict etc.)"
            # indicator for whether output is observed or not.
            # This used for correctly quantize standalone modules
            output_is_observed = \
                result_node.args[0].name in observed_node_names_set
            # these inputs are observed in parent
            # converting List[int] to Tensor since module attribute is
            # Union[Tensor, Module]
            model._standalone_module_input_quantized_idxs = \
                torch.Tensor(input_quantized_idxs)
            model._standalone_module_output_quantized_idxs = torch.Tensor(output_quantized_idxs)
        return model

    def save_state(self, observed: GraphModule) -> None:
        observed._activation_post_process_map = \
            self.activation_post_process_map  # type: ignore
        observed._patterns = self.patterns  # type: ignore
        observed._qconfig_map = self.qconfig_map  # type: ignore
        observed._prepare_custom_config_dict = \
            self.prepare_custom_config_dict  # type: ignore
        observed._node_name_to_scope = self.node_name_to_scope  # type: ignore

    def restore_state(self, observed: GraphModule) -> None:
        assert is_observed_module(observed), \
            'incoming model must be produced by prepare_fx'
        self.activation_post_process_map = \
            observed._activation_post_process_map  # type: ignore
        self.patterns = observed._patterns  # type: ignore
        self.qconfig_map = observed._qconfig_map  # type: ignore
        self.prepare_custom_config_dict = \
            observed._prepare_custom_config_dict  # type: ignore
        self.node_name_to_scope = observed._node_name_to_scope  # type: ignore

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
        self.modules = dict(model.named_modules())

        custom_module_classes = get_custom_module_class_keys(
            convert_custom_config_dict,
            "observed_to_quantized_custom_module_class")
        assert self.patterns is not None
        matches = self._find_matches(
            model.graph, self.modules, self.patterns,
            custom_module_classes=custom_module_classes)

        quants: Dict[str, Tuple[DefaultQuantizeHandler, Callable]] = \
            self._find_quants(model.graph, matches)

        self.quantized_graph = Graph()
        env: Dict[str, Node] = {}
        quant_env: Dict[str, Node] = {}

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
                quant_env_node = quant_env[n.name]
                env[n.name] = Proxy(quant_env[n.name]).dequantize().node
            return env[n.name]

        def load_quantized(n: Node) -> Node:
            assert n.name in quant_env, \
                'trying to load quantized node but did not find node:' + \
                n.name + ' in quant environment:' + str(quant_env)
            return quant_env[n.name]

        def load_x(n: Node) -> Node:
            assert n.name in env or n.name in quant_env, \
                'node ' + n.name + ' does not exist in either environment'
            if n.name in quant_env:
                return quant_env[n.name]
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
            # of CopyNodeQuantizeHandler
            if type(obj) in [
                    CopyNodeQuantizeHandler,
                    FixedQParamsOpQuantizeHandler
            ]:
                assert node.op in [
                    'call_module',
                    'call_function',
                    'call_method'], \
                    'CopyNodeQuantizeHandler of type ' + node.op + ' is not handled'
                quantized = node_arg_is_quantized(node.args[0])

            if not activation_is_int8_quantized(qconfig) or \
               not input_output_observed(obj):
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
                quant_env[node.name] = quant_env[prev_node.name]
            else:
                # replace activation post process with quantization ops
                root_module = self.modules[""]
                assert isinstance(node.args[0], Node)
                quant_env[node.name] = quantize_node(
                    self, load_non_quantized(node.args[0]), observer_module, node, is_input=True)

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
                        self.modules[node.target])  # type: ignore
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
                        out_quant_idxs = self.modules[node.target]._standalone_module_output_quantized_idxs.tolist()  # type: ignore
                        assert len(out_quant_idxs) <= 1, "Currently standalone only support one output"
                        quantized = 0 in out_quant_idxs

                    result = obj.convert(
                        self, node, load_arg, is_reference=is_reference,
                        convert_custom_config_dict=convert_custom_config_dict)
                    if not is_observed_standalone_module_node:
                        quantized = is_output_quantized(node, obj)

                if quantized:
                    quant_env[node.name] = result
                else:
                    env[node.name] = result
                continue
            elif root_node is not None:
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
                        self.quantized_graph.node_copy(node, load_non_quantized)
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
        model = QuantizedGraphModule(model, act_post_process_removed_graph)
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
        quantized = QuantizedGraphModule(quantized_root, folded_graph)
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

        assert self.qconfig_map is not None
        for node in reversed(graph.nodes):
            if node.name not in match_map and node.name not in all_matched:
                for pattern, value in patterns.items():
                    if is_match(modules, node, pattern):
                        skip_this_match = False
                        if value is BinaryOpQuantizeHandler:
                            use_copy_node = all_node_args_have_no_tensors(node)
                            if use_copy_node:
                                # TODO(future PR): update the pattern to quantize
                                # handler logic to take this into account.
                                value = CopyNodeQuantizeHandler  # type: ignore

                            this_node_qconfig = self.qconfig_map[node.name]
                            if this_node_qconfig:
                                dtypes = get_qconfig_dtypes(this_node_qconfig)
                                # TODO(future PR): update the pattern to quantize
                                # handler logic to take this into account.
                                skip_this_match = (
                                    (node.target in binary_op_supported_dtypes) and
                                    (dtypes not in binary_op_supported_dtypes[node.target])
                                )

                        if not skip_this_match:
                            matched: List[Any] = []
                            record_match(pattern, node, matched)
                            for n in matched:
                                match_map[n.name] = (
                                    node, matched, pattern, value(self, node),  # type: ignore
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
                node_target in standalone_module_names or  # type: ignore
                type(self.modules[node_target]) in standalone_module_classes  # type: ignore
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

    def _find_quants(self, graph: Graph, matches: Dict[str, MatchResult],
                     ) -> Dict[str, Tuple[DefaultQuantizeHandler, Callable]]:
        """
        Takes the nodes in the input graph and pending matches, and finds and
        returns the input and output nodes which need to be quantized.

        Inputs:
          - graph: an fx.Graph object
          - matches: output of self._find_matches function

        Outputs a map of
         node_name -> (QuantizeHandler instance (always DefaultQuantizeHandler),
         activation_post_process (observer/fake_quantize module) constructor)
        """
        quants: Dict[str, Tuple[DefaultQuantizeHandler, Callable]] = {}

        def visit(node, matched_pattern, qconfig):
            def visit_arg(arg):
                is_weight = node_arg_is_weight(node, arg)
                is_bias = node_arg_is_bias(node, arg)
                is_activation = not (is_weight or is_bias)
                no_tensors = all_node_args_have_no_tensors(arg)
                # bias needs to be quantized if activation is fp16 and weight is fp16
                # this is the case for glow
                should_add_handler = qconfig is not None and (
                    (is_activation and
                     activation_is_statically_quantized(qconfig)) or
                    (is_weight and weight_is_quantized(qconfig)) or
                    (is_bias and activation_dtype(qconfig) == torch.float16)
                    and weight_dtype(qconfig) == torch.float16) and \
                    (not no_tensors)

                if should_add_handler:
                    act_post_process_ctr = qconfig.weight if is_weight else \
                        qconfig.activation
                    # overwrite the constructor from qconfig
                    act_post_process_ctr = \
                        get_default_output_activation_post_process_map().get(
                            matched_pattern,
                            act_post_process_ctr)
                    quants[arg.name] = (
                        DefaultQuantizeHandler(self, arg), act_post_process_ctr)
            return visit_arg

        for node in graph.nodes:
            if node.name in matches:
                root_node, matched_nodes, matched_pattern, quantize_handler, \
                    qconfig = matches[node.name]
                # don't attach observer/fake_quant for CopyNodeQuantizeHandler
                if isinstance(quantize_handler, CopyNodeQuantizeHandler):
                    qconfig = None
                if root_node is node and \
                        input_output_observed(quantize_handler):
                    # matched_nodes[-1] is the first op in the sequence and
                    # matched_nodes[0] is the last op in the sequence
                    # inputs
                    # matched_pattern is set to None for inputs because
                    # we only want to select QuantizeHandler object based
                    # on pattern for output, inputs will always use
                    # DefaultQuantizeHandler
                    map_arg(matched_nodes[-1].args, visit(matched_nodes[-1],
                            None, qconfig))
                    map_arg(matched_nodes[-1].kwargs, visit(matched_nodes[-1],
                            None, qconfig))

                    # output
                    # we don't insert observer for output of standalone module
                    if not isinstance(
                            quantize_handler, StandaloneModuleQuantizeHandler):
                        # passing in matched_pattern here so that we can
                        # customize activation_post_process constructor for
                        # output based on the pattern, e.g.
                        # for sigmoid op we'll use
                        # default_affine_fixed_qparam_fake_quant
                        map_arg(matched_nodes[0],
                                visit(None, matched_pattern, qconfig))
        return quants
