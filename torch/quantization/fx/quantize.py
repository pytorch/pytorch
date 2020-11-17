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
    get_combined_dict
)

from .pattern_utils import (
    is_match,
    get_default_quant_patterns,
    get_default_output_activation_post_process_map,
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
    get_swapped_custom_module_class,
    activation_is_statically_quantized,
)

from collections import OrderedDict
import warnings
import re

from typing import Optional

# ------------------------
# Helper Functions
# ------------------------

# Returns a function that can get a new attribute name for module with given prefix
# for example,
# >> get_new_observer_name = get_new_attr_name_with_prefix('_observer')
# >> new_name = get_new_observer_name(module)
# new_name will be an unused attribute name on module, e.g. `_observer_1`
def get_new_attr_name_with_prefix(prefix):
    def get_new_attr_name(module):
        def get_attr_name(i):
            return prefix + str(i)
        i = 0
        attr_name = get_attr_name(i)
        while hasattr(module, attr_name):
            i += 1
            attr_name = get_attr_name(i)
        return attr_name
    return get_new_attr_name

def collect_producer_nodes(node):
    r''' Starting from a target node, trace back until we hit inpu or
    getattr node. This is used to extract the chain of operators
    starting from getattr to the target node, for example
    def forward(self, x):
      observed = self.observer(self.weight)
      return F.linear(x, observed)
    collect_producer_nodes(observed) will either return a list of nodes that produces
    the observed node or None if we can't extract a self contained graph without
    free variables(inputs of the forward function).
    '''
    nodes = [node]
    frontier = [node]
    while frontier:
        node = frontier.pop()
        all_args = list(node.args) + list(node.kwargs.values())
        for arg in all_args:
            if not isinstance(arg, Node):
                continue
            if arg.op == 'placeholder':
                # hit input, can't fold in this case
                return None
            nodes.append(arg)
            if not (arg.op == 'call_function' and arg.target == getattr):
                frontier.append(arg)
    return nodes

def graph_module_from_producer_nodes(root, producer_nodes):
    r''' Construct a graph module from extracted producer nodes
    from `collect_producer_nodes` function
    Args:
      root: the root module for the original graph
      producer_nodes: a list of nodes we use to construct the graph
    Return:
      A graph module constructed from the producer nodes
    '''
    assert len(producer_nodes) > 0, 'list of producer nodes can not be empty'
    # since we traced back from node to getattrr
    producer_nodes.reverse()
    graph = Graph()
    env = {}

    def load_arg(a):
        return map_arg(a, lambda node: env[node])
    for producer_node in producer_nodes:
        env[producer_node] = graph.node_copy(producer_node, load_arg)
    graph.output(load_arg(producer_nodes[-1]))
    graph_module = GraphModule(root, graph)
    return graph_module

def assert_and_get_unique_device(module):
    """
    Returns the unique device for a module, or None if no device is found.
    Throws an error if multiple devices are detected.
    """
    devices = {p.device for p in module.parameters()} | \
        {p.device for p in module.buffers()}
    assert len(devices) <= 1, (
        "prepare only works with cpu or single-device CUDA modules, "
        "but got devices {}".format(devices)
    )
    device = next(iter(devices)) if len(devices) > 0 else None
    return device

def is_submodule_of_fake_quant(name, module, named_modules):
    parent_name, _ = _parent_name(name)
    return is_activation_post_process(named_modules[parent_name])

def get_flattened_qconfig_dict(qconfig_dict):
    """ flatten the global, object_type and module_name qconfig
    to the same qconfig_dict so that it can be used by
    propagate_qconfig_ function.
    "module_name_regex" is ignored for now since it's not supported
    in propagate_qconfig_, but it can be fixed later.

    For example:
    Input: {
      "": qconfig,
      "object_type": [
        (torch.add, qconfig)
      ],
      "module_name": [
        ("conv", qconfig)
      ]
    }

    Output: {
      "": qconfig,
      torch.add: qconfig,
      "conv": qconfig
    }
    """
    flattened = dict()
    if '' in qconfig_dict:
        flattened[''] = qconfig_dict['']

    def flatten_key(key):
        if key in qconfig_dict:
            for obj, qconfig in qconfig_dict[key]:
                flattened[obj] = qconfig

    flatten_key('object_type')
    flatten_key('module_name')
    return flattened

def convert_dict_to_ordered_dict(qconfig_dict):
    """ Convert dict in qconfig_dict to ordered dict
    """
    # convert a qconfig list for a type to OrderedDict
    def _convert_to_ordered_dict(key, qconfig_dict):
        qconfig_dict[key] = OrderedDict(qconfig_dict.get(key, []))

    _convert_to_ordered_dict('object_type', qconfig_dict)
    _convert_to_ordered_dict('module_name_regex', qconfig_dict)
    _convert_to_ordered_dict('module_name', qconfig_dict)

# A dictionary for querying the weight index for a given op
WEIGHT_INDEX_DICT = {
    torch.nn.functional.conv2d : [1],
    torch.nn.functional.linear : [1],
}

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
        self.activation_post_process_map = None
        # mapping from node name to qconfig that should be used for that node
        # filled out for a model during _generate_qconfig_map
        self.qconfig_map = None
        # mapping from fully qualified module name to module instance
        # for example,
        # {
        #   '': Model(...),
        #   'linear': Linear(...),
        #   'linear.weight_fake_quant': PerChannelMinMaxObserver(...),
        # }
        self.modules = None
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
        self.patterns = None


    def _qat_swap_modules(self, root, additional_qat_module_mapping):
        all_mappings = get_combined_dict(get_default_qat_module_mappings(), additional_qat_module_mapping)
        convert(root, mapping=all_mappings, inplace=True, remove_qconfig=False)

    def _generate_qconfig_map(self,
                              root,
                              input_graph,
                              qconfig_dict):
        global_qconfig = qconfig_dict.get('', None)

        def get_module_type_qconfig(
                module_type, fallback_qconfig=global_qconfig):
            return qconfig_dict['object_type'].get(module_type, fallback_qconfig)

        def get_function_qconfig(
                function, fallback_qconfig=global_qconfig):
            return qconfig_dict['object_type'].get(function, fallback_qconfig)

        def get_module_name_regex_qconfig(
                module_name, fallback_qconfig=global_qconfig):
            for regex_pattern, qconfig in qconfig_dict['module_name_regex'].items():
                if re.match(regex_pattern, module_name):
                    # first match wins
                    return qconfig
            return fallback_qconfig

        def get_module_name_qconfig(
                module_name, fallback_qconfig=global_qconfig):
            if module_name == '':
                # module name qconfig not found
                return fallback_qconfig
            if module_name in qconfig_dict['module_name']:
                return qconfig_dict['module_name'][module_name]
            else:
                parent, _ = _parent_name(module_name)
                return get_module_name_qconfig(parent, fallback_qconfig)

        # get qconfig for module_name,
        # fallback to module_name_regex_qconfig, module_type_qconfig, global_qconfig
        # if necessary
        def get_qconfig(module_name):
            module_type_qconfig = \
                get_module_type_qconfig(type(self.modules[module_name]))
            module_name_regex_qconfig = \
                get_module_name_regex_qconfig(module_name, module_type_qconfig)
            module_name_qconfig = \
                get_module_name_qconfig(module_name, module_name_regex_qconfig)
            return module_name_qconfig

        self.qconfig_map = dict()
        for node in input_graph.nodes:
            if node.op == 'get_attr':
                module_name, _ = _parent_name(node.target)
                self.qconfig_map[node.name] = get_qconfig(module_name)
            elif node.op == 'call_function':
                # precedence: [TODO] module_name_qconfig (need scope support from fx)
                # > function_qconfig > global_qconfig
                function_qconfig = get_function_qconfig(node.target)
                self.qconfig_map[node.name] = function_qconfig
            elif node.op == 'call_method':
                self_obj = node.args[0]
                # qconfig for call_method should be the same as the `self` object for the call
                if self_obj.name in self.qconfig_map:
                    qconfig = self.qconfig_map[self_obj.name]
                else:
                    # need scope info for each node to support this
                    warnings.warn("Scope info is not yet supported, taking default qconfig for value {}".format(node.name))
                    qconfig = get_qconfig('')
                self.qconfig_map[node.name] = qconfig
            elif node.op == 'call_module':
                module_qconfig = get_qconfig(node.target)
                # regex is not supported eager mode propagate_qconfig_, we'll need to
                # set the qconfig explicitly here in case regex
                # is used
                self.modules[node.target].qconfig = module_qconfig
                self.qconfig_map[node.name] = module_qconfig

    def _prepare(self, model, qconfig_dict, prepare_custom_config_dict, is_standalone_module):
        """ standalone_module means it a submodule that is not inlined in parent module,
        and will be quantized separately as one unit.

        When we are preparing a standalone module:
        input of the module is observed in parent module, output of the module
        is observed in the standalone module.
        Returns:
            model(GraphModule): prepared standalone module with following attributes:
                _standalone_module_observed_input_idxs(List[Int]): a list of indexs for the graph inputs that
                                         needs to be observed in parent module
                _output_is_observed(Bool): a boolean variable indicate whether the output of the
                                   custom module is observed or not
        """
        if prepare_custom_config_dict is None:
            prepare_custom_config_dict = {}

        additional_quant_patterns = prepare_custom_config_dict.get("additional_quant_pattern", {})
        self.patterns = get_combined_dict(get_default_quant_patterns(), additional_quant_patterns)

        flattened_qconfig_dict = get_flattened_qconfig_dict(qconfig_dict)
        # TODO: support regex as well
        propagate_qconfig_(model, flattened_qconfig_dict)
        if model.training:
            additional_qat_module_mapping = prepare_custom_config_dict.get("additioanl_qat_module_mapping", {})
            self._qat_swap_modules(model, additional_qat_module_mapping)

        self.modules = dict(model.named_modules())

        convert_dict_to_ordered_dict(qconfig_dict)
        # map from node name to qconfig, used in _find_matches
        self._generate_qconfig_map(model, model.graph, qconfig_dict)

        # match the patterns that will get quantized
        standalone_module_names = prepare_custom_config_dict.get("standalone_module_name", None)
        standalone_module_classes = prepare_custom_config_dict.get("standalone_module_class", None)
        custom_module_classes = get_custom_module_class_keys(prepare_custom_config_dict, "float_to_observed_custom_module_class")
        matches = self._find_matches(
            model.graph, self.modules, self.patterns, standalone_module_names, standalone_module_classes, custom_module_classes)

        # find _inputs_ to matched nodes that are not quantized, these
        # have to be quantized, which requires measuring stats,
        # initialize an DefaultQuantizeHandler object for each
        quants = self._find_quants(model.graph, matches)

        self.activation_post_process_map = dict()
        env = {}
        observed_graph = Graph()
        observed_node_names_set = set()

        def load_arg(a):
            return map_arg(a, lambda node: env[node.name])

        # indexes for the inputs that needs to be observed
        standalone_module_observed_input_idxs = []
        graph_inputs = []
        for node in model.graph.nodes:
            if node.op == 'placeholder':
                graph_inputs.append(node.name)

        get_new_observer_name = get_new_attr_name_with_prefix('activation_post_process_')
        model_device = assert_and_get_unique_device(model)

        def insert_observer(node, observer):
            """Insert observer for node by modifying the observed_graph and
               attach observer module to the model
               Args:
                 node: Node
                 observer: observer/fake_quantize module instance
            """
            # respect device affinity when adding observers
            if model_device:
                observer.to(model_device)
            # add observer module as attribute
            prefix = node.name + '_activation_post_process_'
            get_new_observer_name = get_new_attr_name_with_prefix(prefix)
            observer_name = get_new_observer_name(model)
            setattr(model, observer_name, observer)
            # put observer instance activation_post_process map
            self.activation_post_process_map[node.name] = observer
            # insert observer call
            env[node.name] = observed_graph.create_node('call_module', observer_name, (load_arg(node),), {})
            observed_node_names_set.add(node.name)

        def insert_observer_for_special_module(quantize_handler):
            """ Insert observer for custom module and standalone module
              Returns: standalone_module_input_idxs: the indexs for inputs that needs
              to be observed by parent module
            """
            standalone_module_input_idxs = None
            if isinstance(quantize_handler, CustomModuleQuantizeHandler):
                custom_module = self.modules[node.target]
                custom_module_class_mapping = prepare_custom_config_dict.get("float_to_observed_custom_module_class", {})
                observed_custom_module_class = \
                    get_swapped_custom_module_class(custom_module, custom_module_class_mapping, qconfig)
                observed_custom_module = \
                    observed_custom_module_class.from_float(custom_module)
                parent_name, name = _parent_name(node.target)
                setattr(self.modules[parent_name], name, observed_custom_module)
            elif isinstance(quantize_handler, StandaloneModuleQuantizeHandler):
                # observe standalone module
                standalone_module = self.modules[node.target]
                prepare = torch.quantization.quantize_fx._prepare_standalone_module_fx
                observed_standalone_module = prepare(standalone_module, {"": qconfig})
                observed_standalone_module.qconfig = qconfig
                standalone_module_input_idxs = observed_standalone_module._standalone_module_observed_input_idxs
                observed_standalone_module = mark_observed_standalone_module(observed_standalone_module)
                parent_name, name = _parent_name(node.target)
                setattr(self.modules[parent_name], name, observed_standalone_module)
                self.modules[node.target] = observed_standalone_module
            return standalone_module_input_idxs

        def insert_observer_for_output_of_the_node(
                node,
                quantize_handler,
                qconfig,
                standalone_module_input_idxs):
            """ Insert observer/fake_quantize module for output of the observed module
            if needed
            """
            # don't need to insert observer for output if activation does not
            # need to be statically quantized
            if activation_is_statically_quantized(qconfig):
                if isinstance(quantize_handler, FixedQParamsOpQuantizeHandler) and model.training:
                    # we only insert fake quantize module in qat
                    activation_post_process_ctr = \
                        get_default_output_activation_post_process_map().get(pattern, None)
                    assert activation_post_process_ctr is not None, \
                        "activation_post_process constructor not provided for " + \
                        "pattern:" + str(pattern)
                    insert_observer(node, activation_post_process_ctr())
                elif (isinstance(quantize_handler, FixedQParamsOpQuantizeHandler) and
                      not model.training) or isinstance(quantize_handler, CopyNode):
                    # inserting observers for output of observed module, or mark the output
                    # as observed
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
                elif ((isinstance(quantize_handler, Add) or isinstance(quantize_handler, Mul)) and
                      quantize_handler.num_node_args == 1):
                    input_node = matched_nodes[-1]  # first node in the sequence

                    def input_is_observed(arg):
                        return isinstance(arg, Node) and arg.name in observed_node_names_set
                    # This is checking if one of the argument of add/mul
                    # is an observed node
                    # If both of the inputs are number,
                    # we will not consider the output to be observed
                    if input_is_observed(input_node.args[0]) or input_is_observed(input_node.args[1]):
                        observed_node_names_set.add(node.name)
                elif isinstance(quantize_handler, StandaloneModuleQuantizeHandler):
                    assert node.op == 'call_module'
                    output_is_observed = self.modules[node.target]._output_is_observed
                    if output_is_observed:
                        observed_node_names_set.add(node.name)
                elif quantize_handler.all_node_args:
                    # observer for outputs
                    new_observer = qconfig.activation()
                    insert_observer(node, new_observer)

            # insert observer for input of standalone module
            if standalone_module_input_idxs is not None:
                for idx in standalone_module_input_idxs:
                    if node.args[idx].name not in observed_node_names_set:
                        new_observer = qconfig.activation()
                        insert_observer(node.args[idx], new_observer)

        def insert_observer_for_input_arg_of_observed_node(arg):
            """
               Input:
                 arg: input arg node for another observed node, e.g.
                 input activaiton for functional linear node
            """
            if node.name not in observed_node_names_set and node.name in quants:
                if is_standalone_module and node.name in graph_inputs:
                    # we'll insert observer for input of standalone module
                    # in parent graph
                    standalone_module_observed_input_idxs.append(graph_inputs.index(node.name))
                    return
                _, activation_post_process_ctr = quants[node.name]
                if activation_post_process_ctr is not None:
                    insert_observer(node, activation_post_process_ctr())

        result_node : Optional[Node] = None
        for node in model.graph.nodes:
            if node.op == 'output':
                observed_graph.output(load_arg(node.args[0]))
                result_node = node
                continue
            if node.name in observed_node_names_set:
                continue

            root_node, matched_nodes, pattern, obj, qconfig = matches.get(node.name, (None, None, None, None, None))
            if root_node is None:
                env[node.name] = observed_graph.node_copy(node, load_arg)
            elif root_node is node:
                env[node.name] = observed_graph.node_copy(node, load_arg)
                # index for input of custom module that needs to be observed in parent
                if qconfig is not None:
                    standalone_module_input_idxs = insert_observer_for_special_module(obj)
                    insert_observer_for_output_of_the_node(
                        node, obj, qconfig, standalone_module_input_idxs)
            else:
                env[node.name] = observed_graph.node_copy(node, load_arg)
            insert_observer_for_input_arg_of_observed_node(node)


        model = GraphModule(model, observed_graph)
        self.save_state(model)
        model = mark_observed_module(model)
        if is_standalone_module:
            assert result_node is not None
            assert isinstance(result_node.args[0], Node), \
                'standalone module returning dict is not yet supported'
            # indicator for whether output is observed or not.
            # This used for correctly quantize standalone modules
            output_is_observed = result_node.args[0].name in observed_node_names_set
            model._standalone_module_observed_input_idxs = standalone_module_observed_input_idxs
            model._output_is_observed = output_is_observed
        return model

    def save_state(self, observed):
        observed._activation_post_process_map = self.activation_post_process_map
        observed._patterns = self.patterns
        observed._qconfig_map = self.qconfig_map

    def restore_state(self, observed):
        assert is_observed_module(observed), 'incoming model must be produced by prepare_fx'
        self.activation_post_process_map = observed._activation_post_process_map
        self.patterns = observed._patterns
        self.qconfig_map = observed._qconfig_map

    def prepare(self, model, qconfig_dict, prepare_custom_config_dict=None, is_standalone_module=False):
        return self._prepare(model, qconfig_dict, prepare_custom_config_dict, is_standalone_module)

    def _run_weight_observers(self, observed):
        r''' Extract the subgraph that produces the weight for dynamic quant
        or weight only quant node and run the subgraph to observe the weight.
        Note that the observers of dynamic quant or weight only quant ops are run during
        the convert step.
        '''
        for node in observed.graph.nodes:
            if node.op == 'call_function' and node.target in WEIGHT_INDEX_DICT:
                for i, node_arg in enumerate(node.args):
                    if i in WEIGHT_INDEX_DICT[node.target]:
                        # node_arg is weight
                        weight_observer_nodes = collect_producer_nodes(node_arg)
                        if weight_observer_nodes is not None:
                            weight_observer_module = graph_module_from_producer_nodes(
                                observed, weight_observer_nodes)
                            # run the weight observer
                            weight_observer_module()
        return

    def _convert(self, model, debug=False, convert_custom_config_dict=None, is_standalone_module=False):
        """ standalone_module means it a submodule that is not inlined in parent module,
        and will be quantized separately as one unit.
        For standalone module: the inputs will be quantized by parent module,
        checks `_standalone_module_observed_input_idxs` of
        input observed model and will treat these inputs as quantized
        also will not dequantize the final output.
        Returns a quantized standalone module which accepts quantized input(if needed)
        and produces quantized output (if needed).
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
            convert_custom_config_dict, "observed_to_quantized_custom_module_class")
        matches = self._find_matches(
            model.graph, self.modules, self.patterns,
            custom_module_classes=custom_module_classes)

        quants = self._find_quants(model.graph, matches)

        self.quantized_graph = Graph()
        env = {}
        quant_env = {}

        graph_inputs = []
        for node in model.graph.nodes:
            if node.op == 'placeholder':
                graph_inputs.append(node.name)

        def load_non_quantized(n):
            if n.name not in env:
                assert n.name in quant_env, \
                    'trying to load float node but did not find node:' + n.name + \
                    ' in quantized or non quantized environment, env: ' + str(env) + \
                    ' quant_env:' + str(quant_env)
                env[n.name] = Proxy(quant_env[n.name]).dequantize().node
            return env[n.name]

        def load_quantized(n):
            if n.name not in quant_env:
                assert n.name in env, \
                    'trying to load quantized node but did not find node:' + n.name + \
                    ' in float environment:' + str(env)
                assert n.name in quants, 'did not find quant object for node:' + n.name
                quant = quants[n.name][0]
                quant_env[n.name] = quant.convert(self, env[n.name])
            return quant_env[n.name]

        def load_x(n):
            assert n.name in env or n.name in quant_env, \
                'node ' + n.name + ' does not exist in either environment'
            if n.name in quant_env:
                return quant_env[n.name]
            else:
                return env[n.name]

        def load_arg(quantized):
            """
            Input: quantized, which can be None, list, boolean or tuple
              - if quantized is a list or tuple, then arg should be a list and the args with corresponding
                indexes will be quantized
              - if quantized is a boolean, then all args will be quantized/not quantized
              - if quantized is None, then we'll load the node as long as it exists

            Output: fn which takes arg_or_args, and loads them from the corresponding
              environment depending on the value of quantized.
            """
            assert quantized is None or isinstance(quantized, (tuple, list, bool)), type(quantized)

            def load_arg_impl(arg_or_args):
                if quantized is None:
                    return map_arg(arg_or_args, load_x)
                if isinstance(quantized, bool):
                    return map_arg(arg_or_args, load_quantized if quantized else load_non_quantized)
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

        def is_quantized(node):
            if isinstance(node, Node):
                assert node.name in env or node.name in quant_env, 'Expecting node to be in the environment'
                # there might be nodes appearing in both environemnts, but quant_env will take
                # precedence
                if node.name in quant_env:
                    return True
                elif node.name in env:
                    return False
            elif isinstance(node, list):
                quantized = map(is_quantized, node)
                if all(quantized):
                    return True
                elif not any(quantized):
                    return False
                else:
                    raise Exception("partially quantized inputs in list not handled yet")

        for node in model.graph.nodes:
            if node.op == 'output':
                if is_standalone_module:
                    # result are kept quantized in the quantized standalone module
                    graph_output = map_arg(node.args[0], load_x)
                else:
                    graph_output = map_arg(node.args[0], load_non_quantized)
                self.quantized_graph.output(graph_output)
                continue
            root_node, matched, matched_pattern, obj, qconfig = matches.get(node.name, (None, None, None, None, None))
            if root_node is node:
                if qconfig is None:
                    result = self.quantized_graph.node_copy(node, load_non_quantized)
                    quantized = False
                else:
                    result = obj.convert(self, node, load_arg, debug=debug, convert_custom_config_dict=convert_custom_config_dict)
                    if node.op == 'call_module' and is_observed_standalone_module(self.modules[node.target]):
                        quantized = self.modules[node.target]._output_is_observed
                    else:
                        quantized = True

                    # Need to get correct quantized/non-quantized state for the output of CopyNode
                    if type(obj) in [
                            CopyNode,
                            FixedQParamsOpQuantizeHandler
                    ]:
                        assert node.op in [
                            'call_module',
                            'call_function',
                            'call_method'], \
                            'CopyNode of type ' + node.op + ' is not handled'
                        quantized = is_quantized(node.args[0])

                    if not activation_is_statically_quantized(qconfig):
                        quantized = False

                if quantized:
                    quant_env[node.name] = result
                else:
                    env[node.name] = result
                continue
            elif root_node is not None:
                continue

            # handle activation post process calls
            if node.op == 'call_module':
                if is_activation_post_process(self.modules[node.target]):
                    observer_module = self.modules[node.target]
                    prev_node = node.args[0]
                    if observer_module.dtype == torch.float16:
                        # activations are not quantized for
                        # fp16 dynamic quantization
                        # copy the activaiton_post_process node here
                        # since we may need it when we insert prepack
                        # op for weight of linear, this will be removed
                        # later in a separate pass
                        env[node.name] = self.quantized_graph.node_copy(node, load_non_quantized)
                        continue
                    if prev_node.name in quant_env:
                        # if previous node is already quantized, we'll just remove the activation_post_process
                        quant_env[node.name] = quant_env[prev_node.name]
                        continue
                    # replace activation post process with quantization ops
                    root_module = self.modules['']
                    quant_env[node.name] = quantize_node(
                        root_module, self.quantized_graph,
                        load_non_quantized(node.args[0]), observer_module)
                    continue

            if is_standalone_module and node.op == 'placeholder' and \
               graph_inputs.index(node.name) in model._standalone_module_observed_input_idxs:
                # the node is quantized in parent module
                quant_env[node.name] = self.quantized_graph.node_copy(node, load_non_quantized)
            else:
                # copy quantized or non-quantized node
                env[node.name] = self.quantized_graph.node_copy(node, load_non_quantized)

        # remove activation post process
        act_post_process_removed_graph = Graph()
        env = {}

        def load_arg(a):
            return map_arg(a, lambda node: env[node.name])
        for node in self.quantized_graph.nodes:
            if node.op == 'output':
                act_post_process_removed_graph.output(map_arg(node.args[0], load_arg))
                continue
            if node.op == 'call_module' and \
               is_activation_post_process(self.modules[node.target]):
                # remove activation post process node
                env[node.name] = env[node.args[0].name]
            else:
                env[node.name] = act_post_process_removed_graph.node_copy(node, load_arg)

        # removes qconfig and activation_post_process modules
        _remove_qconfig(model)
        model = GraphModule(model, act_post_process_removed_graph)
        return model

    # Trace back from the weight node util we hit getattr, reconstruct the graph module
    # with the traced nodes and run the graph module to pack the weight. then replace
    # the original chain of ops with the packed weight.
    def _fold_weight(self, quantized):
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
        env = {}

        def load_arg(a):
            return map_arg(a, lambda node: env[node.name])
        get_new_packed_weight_name = get_new_attr_name_with_prefix('_fx_pass_packed_weight_')
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

    def convert(self, model, debug=False, convert_custom_config_dict=None, is_standalone_module=False):
        quantized = self._convert(model, debug, convert_custom_config_dict, is_standalone_module)
        if not debug:
            quantized = self._fold_weight(quantized)
        return quantized

    def _find_matches(
            self, graph, modules, patterns,
            standalone_module_names=None,
            standalone_module_classes=None,
            custom_module_classes=None):
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
            (node, matched_values, matched_pattern, QuantizeHandler instance, qconfig)

        For example, {
          'relu_1': (relu_1, [relu_1], torch.nn.functional.relu, <CopyNode instance>, QConfig(...)),
          ...
        }
        """
        if custom_module_classes is None:
            custom_module_classes = []

        if standalone_module_classes is None:
            standalone_module_classes = []

        if standalone_module_names is None:
            standalone_module_names = []

        match_map = {}
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

        for node in reversed(graph.nodes):
            if node.name not in match_map and node.name not in all_matched:
                for pattern, value in patterns.items():
                    if is_match(modules, node, pattern):
                        matched = []
                        record_match(pattern, node, matched)
                        for n in matched:
                            match_map[n.name] = (node, matched, pattern, value(self, node), self.qconfig_map[n.name])
                            all_matched.add(n.name)
                        # break after finding the first match
                        break

        # add custom module instances to the match result
        for node in graph.nodes:
            if node.op == 'call_module' and \
               type(self.modules[node.target]) in custom_module_classes:
                custom_module_qconfig = self.qconfig_map[node.name]
                match_map[node.name] = (
                    node, [node], None, CustomModuleQuantizeHandler(self, node), custom_module_qconfig)

        def is_standalone_module(node_target):
            return node_target in standalone_module_names or \
                type(self.modules[node_target]) in standalone_module_classes

        # add standalone modules to the match
        for node in graph.nodes:
            if node.op == 'call_module' and \
               (is_standalone_module(node.target) or
                    is_observed_standalone_module(self.modules[node.target])):
                # add node to matched nodes
                custom_module_qconfig = self.qconfig_map[node.name]
                match_map[node.name] = (
                    node, [node], None, StandaloneModuleQuantizeHandler(self, node), custom_module_qconfig)

        return match_map

    def _find_quants(self, graph, matches):
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
        quants = {}

        def visit(node, matched_pattern, qconfig):
            def visit_arg(arg):
                is_weight = False
                if isinstance(node, Node) and node.op == 'call_function' and node.target in WEIGHT_INDEX_DICT:
                    for i, node_arg in enumerate(node.args):
                        if arg is node_arg and i in WEIGHT_INDEX_DICT[node.target]:
                            is_weight = True
                if qconfig is not None and \
                   (activation_is_statically_quantized(qconfig) or is_weight):
                    act_post_process_ctr = qconfig.weight if is_weight else qconfig.activation
                    quants[arg.name] = (DefaultQuantizeHandler(self, arg), qconfig, is_weight)
                    # overwrite the constructor from qconfig
                    act_post_process_ctr = \
                        get_default_output_activation_post_process_map().get(
                            matched_pattern,
                            act_post_process_ctr)
                    # overwrite previous activation post process constructor if necessary
                    quants[arg.name] = (DefaultQuantizeHandler(self, arg), act_post_process_ctr)
            return visit_arg

        for node in graph.nodes:
            if node.name in matches:
                root_node, matched_nodes, matched_pattern, quantize_handler, qconfig = matches[node.name]
                # don't attach observer/fake_quant for CopyNode
                if isinstance(quantize_handler, CopyNode):
                    qconfig = None
                if root_node is node:
                    # matched_nodes[-1] is the first op in the sequence and
                    # matched_nodes[0] is the last op in the sequence
                    # inputs
                    # matched_pattern is set to None for inputs because
                    # we only want to select QuantizeHandler object based
                    # on pattern for output, inputs will always use
                    # DefaultQuantizeHandler
                    map_arg(matched_nodes[-1].args, visit(matched_nodes[-1], None, qconfig))
                    map_arg(matched_nodes[-1].kwargs, visit(matched_nodes[-1], None, qconfig))

                    # output
                    # we don't insert observer for output of standalone module
                    if not isinstance(quantize_handler, StandaloneModuleQuantizeHandler):
                        # passing in matched_pattern here so that we can customize
                        # activation_post_process constructor for output based on the pattern, e.g.
                        # for sigmoid op we'll use default_affine_fixed_qparam_fake_quant
                        map_arg(matched_nodes[0], visit(None, matched_pattern, qconfig))
        return quants
