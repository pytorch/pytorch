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
    activation_is_statically_quantized,
)

from .pattern_utils import (
    is_match,
    get_default_quant_patterns,
    get_default_output_activation_post_process_map,
    input_output_observed,
    Pattern,
)

from .observed_module import (
    mark_observed_module,
    is_observed_module,
    mark_observed_standalone_module,
    is_observed_standalone_module,
)

from .quantization_patterns import *

from .utils import (
    _parent_name,
    quantize_node,
    get_custom_module_class_keys,
    get_new_attr_name_with_prefix,
    collect_producer_nodes,
    graph_module_from_producer_nodes,
    assert_and_get_unique_device,
)

from .qconfig_utils import *

import warnings

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

def insert_observer_for_special_module(
        quantize_handler: QuantizeHandler, modules: Dict[str, torch.nn.Module],
        prepare_custom_config_dict: Any, qconfig: Any, node: Node):
    """ Insert observer for custom module and standalone module
      Returns: standalone_module_input_idxs: the indexs for inputs that
      needs to be observed by parent module
    """
    assert modules is not None
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
        config = name_config_map.get(node.target, (None, None))
        standalone_module_qconfig_dict = {"": qconfig} if config[0] is None else config[0]
        standalone_prepare_config_dict = {} if config[1] is None else config[1]
        prepare = \
            torch.quantization.quantize_fx._prepare_standalone_module_fx  # type: ignore
        observed_standalone_module = \
            prepare(standalone_module, standalone_module_qconfig_dict, standalone_prepare_config_dict)
        observed_standalone_module = mark_observed_standalone_module(
            observed_standalone_module)
        parent_name, name = _parent_name(node.target)
        setattr(modules[parent_name], name,
                observed_standalone_module)
        modules[node.target] = observed_standalone_module  # type: ignore

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
        matched_nodes: Optional[List[Node]]):
    """ Insert observer/fake_quantize module for output of the observed
    module if needed
    """
    # don't need to insert observer for output if activation does not
    # need to be statically quantized
    assert modules is not None
    if activation_is_statically_quantized(qconfig):
        if isinstance(quantize_handler, FixedQParamsOpQuantizeHandler) \
                and model.training:
            # we only insert fake quantize module in qat
            assert pattern is not None
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
                isinstance(quantize_handler, CopyNode):
            # inserting observers for output of observed module, or
            # mark the output as observed
            assert node.op in [
                'call_module',
                'call_function',
                'call_method'], \
                'CopyNode of type ' + node.op + ' is not handled'

            def is_observed(input_arg):
                if isinstance(input_arg, Node):
                    return input_arg.name in observed_node_names_set
                elif isinstance(input_arg, list):
                    return all(map(is_observed, input_arg))
            # propagate observed property from input
            if is_observed(node.args[0]):
                observed_node_names_set.add(node.name)
        elif ((isinstance(quantize_handler, Add) or
                isinstance(quantize_handler, Mul)) and
              quantize_handler.num_node_args == 1):
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
        elif isinstance(quantize_handler,
                        StandaloneModuleQuantizeHandler):
            # output is observed in the standalone module
            return
        elif (quantize_handler.all_node_args and
              input_output_observed(quantize_handler)):
            # observer for outputs
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
            qconfig_dict: Any) -> None:
        global_qconfig = qconfig_dict.get('', None)

        self.qconfig_map = dict()
        for node in input_graph.nodes:
            if node.op == 'get_attr':
                module_name, _ = _parent_name(node.target)
                self.qconfig_map[node.name] = get_qconfig(
                    self.modules, qconfig_dict, module_name, global_qconfig)
            elif node.op == 'call_function':
                # precedence: [TODO] module_name_qconfig (need scope support
                # from fx)
                # > function_qconfig > global_qconfig
                function_qconfig = get_object_type_qconfig(
                    qconfig_dict, node.target, global_qconfig)
                self.qconfig_map[node.name] = function_qconfig
            elif node.op == 'call_method':
                self_obj = node.args[0]
                # qconfig for call_method should be the same as the `self`
                # object for the call
                if self_obj.name in self.qconfig_map:
                    qconfig = self.qconfig_map[self_obj.name]
                else:
                    # need scope info for each node to support this
                    warnings.warn(
                        "Scope info is not yet supported, taking default " +
                        "qconfig for value {}".format(node.name))
                    qconfig = get_qconfig(
                        self.modules, qconfig_dict, '', global_qconfig)
                qconfig = get_object_type_qconfig(qconfig_dict, node.target, qconfig)
                self.qconfig_map[node.name] = qconfig
            elif node.op == 'call_module':
                module_qconfig = get_qconfig(
                    self.modules, qconfig_dict, node.target, global_qconfig)
                # regex is not supported eager mode propagate_qconfig_, we'll
                # need to set the qconfig explicitly here in case regex
                # is used
                assert self.modules is not None
                self.modules[node.target].qconfig = module_qconfig
                self.qconfig_map[node.name] = module_qconfig

    def _prepare(self, model: GraphModule, qconfig_dict: Any,
                 prepare_custom_config_dict: Optional[Dict[str, Any]],
                 is_standalone_module: bool) -> GraphModule:
        """ standalone_module means it a submodule that is not inlined in
        parent module, and will be quantized separately as one unit.

        When we are preparing a standalone module:
        both input and output are observed in prepared standalone module
        Returns:
            model(GraphModule): prepared standalone module
        """
        if prepare_custom_config_dict is None:
            prepare_custom_config_dict = {}
        self.prepare_custom_config_dict = prepare_custom_config_dict

        additional_quant_patterns = \
            prepare_custom_config_dict.get("additional_quant_pattern", {})
        self.patterns = get_combined_dict(
            get_default_quant_patterns(), additional_quant_patterns)

        flattened_qconfig_dict = get_flattened_qconfig_dict(qconfig_dict)
        # TODO: support regex as well
        propagate_qconfig_(model, flattened_qconfig_dict)
        if model.training:
            additional_qat_module_mapping = prepare_custom_config_dict.get(
                "additional_qat_module_mapping", {})
            self._qat_swap_modules(model, additional_qat_module_mapping)

        self.modules = dict(model.named_modules())

        convert_dict_to_ordered_dict(qconfig_dict)
        # map from node name to qconfig, used in _find_matches
        self._generate_qconfig_map(model, model.graph, qconfig_dict)

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

        # indexes for the inputs that needs to be observed
        standalone_module_observed_input_idxs: List[int] = []
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
                    insert_observer_for_special_module(
                        obj, self.modules, prepare_custom_config_dict, qconfig,
                        node)
                    insert_observer_for_output_of_the_node(
                        node, obj, qconfig, self.modules, model, pattern,
                        self.activation_post_process_map, env,
                        observed_graph, load_arg, observed_node_names_set,
                        matched_nodes)
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


        model = GraphModule(model, observed_graph)
        self.save_state(model)
        model = mark_observed_module(model)
        return model

    def save_state(self, observed: GraphModule) -> None:
        observed._activation_post_process_map = \
            self.activation_post_process_map  # type: ignore
        observed._patterns = self.patterns  # type: ignore
        observed._qconfig_map = self.qconfig_map  # type: ignore
        observed._prepare_custom_config_dict = \
            self.prepare_custom_config_dict  # type: ignore

    def restore_state(self, observed: GraphModule) -> None:
        assert is_observed_module(observed), \
            'incoming model must be produced by prepare_fx'
        self.activation_post_process_map = \
            observed._activation_post_process_map  # type: ignore
        self.patterns = observed._patterns  # type: ignore
        self.qconfig_map = observed._qconfig_map  # type: ignore
        self.prepare_custom_config_dict = \
            observed._prepare_custom_config_dict  # type: ignore

    def prepare(self, model: GraphModule, qconfig_dict: Any,
                prepare_custom_config_dict: Dict[str, Any] = None,
                is_standalone_module: bool = False) -> GraphModule:
        return self._prepare(
            model, qconfig_dict, prepare_custom_config_dict,
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

    def _convert(self, model: GraphModule, debug: bool = False,
                 convert_custom_config_dict: Dict[str, Any] = None,
                 is_standalone_module: bool = False) -> GraphModule:
        """ standalone_module means it a submodule that is not inlined in
        parent module, and will be quantized separately as one unit.

        Returns a quantized standalone module which accepts float input
        and produces float output.
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

        def load_arg(quantized: Optional[Union[List[Any], bool, Tuple[Any, ...]]]
                     ) -> Callable[[Node], Argument]:
            """
            Input: quantized, which can be None, list, boolean or tuple
              - if quantized is a list or tuple, then arg should be a list and
                the args with corresponding indexes will be quantized
              - if quantized is a boolean, then all args will be
                quantized/not quantized
              - if quantized is None, then we'll load the node as long as it
                exists

            Output: fn which takes arg_or_args, and loads them from the
                corresponding environment depending on the value of quantized.
            """
            assert quantized is None or \
                isinstance(quantized, (tuple, list, bool)), type(quantized)

            def load_arg_impl(arg_or_args):
                if quantized is None:
                    return map_arg(arg_or_args, load_x)
                if isinstance(quantized, bool):
                    return map_arg(
                        arg_or_args,
                        load_quantized if quantized else load_non_quantized)
                elif isinstance(quantized, (tuple, list)):
                    assert isinstance(arg_or_args, (tuple, list)), arg_or_args
                    loaded_args = []
                    # for now, we only support quantizing positional arguments
                    for i, a in enumerate(arg_or_args):
                        if i in quantized:
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
            # by default the output is expected to be quantized
            quantized = True

            # Need to get correct quantized/non-quantized state for the output
            # of CopyNode
            if type(obj) in [
                    CopyNode,
                    FixedQParamsOpQuantizeHandler
            ]:
                assert node.op in [
                    'call_module',
                    'call_function',
                    'call_method'], \
                    'CopyNode of type ' + node.op + ' is not handled'
                quantized = node_arg_is_quantized(node.args[0])

            if not activation_is_statically_quantized(qconfig) or \
               not input_output_observed(obj):
                quantized = False

            return quantized

        def insert_quantize_node(node: Node) -> None:
            """ Given a activation_post_process module call node, insert a
            quantize node"""
            assert self.modules is not None
            assert isinstance(node.target, str)
            observer_module = self.modules[node.target]
            prev_node = node.args[0]
            if observer_module.dtype == torch.float16:
                # activations are not quantized for
                # fp16 dynamic quantization
                # copy the activaiton_post_process node here
                # since we may need it when we insert prepack
                # op for weight of linear, this will be removed
                # later in a separate pass
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
                    root_module, self.quantized_graph,
                    load_non_quantized(node.args[0]), observer_module)

        # additional state to override inputs to be quantized, if specified
        # by the user
        placeholder_node_seen_cnt = 0
        output_node_seen_cnt = 0
        input_quantized_idxs: List[int] = self.prepare_custom_config_dict.get(
            "input_quantized_idxs", [])
        output_quantized_idxs: List[int] = self.prepare_custom_config_dict.get(
            "output_quantized_idxs", [])

        for node in model.graph.nodes:
            if node.op == 'output':
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
                    result = obj.convert(
                        self, node, load_arg, debug=debug,
                        convert_custom_config_dict=convert_custom_config_dict)
                    if is_observed_standalone_module_node:
                        quantized = False
                    else:
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
        _remove_qconfig(model)
        model = GraphModule(model, act_post_process_removed_graph)
        return model

    # Trace back from the weight node util we hit getattr, reconstruct the
    # graph module with the traced nodes and run the graph module to pack the
    # weight. then replace the original chain of ops with the packed weight.
    def _fold_weight(self, quantized: GraphModule) -> GraphModule:
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
        get_new_packed_weight_name = \
            get_new_attr_name_with_prefix('_fx_pass_packed_weight_')
        quantized_root = quantized
        quantized_graph = quantized.graph
        for node in quantized_graph.nodes:
            prepack_node = folded_nodes.get(node.name, None)
            if prepack_node is node:
                packed_weight = packed_weights[node.name]
                # add a prepacked attribute to root
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
        quantized = GraphModule(quantized_root, folded_graph)
        return quantized

    def convert(self, model: GraphModule, debug: bool = False,
                convert_custom_config_dict: Dict[str, Any] = None,
                is_standalone_module: bool = False) -> GraphModule:
        quantized = self._convert(
            model, debug, convert_custom_config_dict, is_standalone_module)
        if not debug:
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
                     <CopyNode instance>, QConfig(...)),
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
        all_matched = set()

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
                should_add_handler = qconfig is not None and (
                    (is_activation and
                        activation_is_statically_quantized(qconfig)) or
                    (is_weight and weight_is_statically_quantized(qconfig))
                )

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
                # don't attach observer/fake_quant for CopyNode
                if isinstance(quantize_handler, CopyNode):
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
