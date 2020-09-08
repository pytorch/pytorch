import torch
from torch.quantization import (
    propagate_qconfig_,
    convert,
)

from ..quantization_mappings import (
    get_qat_module_mapping,
)

from torch.fx import (
    GraphModule,
    Proxy,
)

from torch.fx.graph import (
    Graph,
    Node,
    map_arg,
)

from .pattern_utils import (
    is_match,
    get_quant_patterns,
    get_dynamic_quant_patterns,
)

from .quantization_patterns import *

from .utils import _parent_name

import copy

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
        return map_arg(a, lambda node: env[node.name])
    for producer_node in producer_nodes:
        env[producer_node.name] = graph.node_copy(producer_node, load_arg)
    graph.output(load_arg(producer_nodes[-1].name))
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


# A dictionary for querying the weight index for a given op
WEIGHT_INDEX_DICT = {
    torch.nn.functional.conv2d : [1],
    torch.nn.functional.linear : [1],
}

# weight prepacking ops
WEIGHT_PREPACK_OPS = {
    torch._ops.ops.quantized.linear_prepack,
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


    def _qat_swap_modules(self, root):
        convert(root, mapping=get_qat_module_mapping(), inplace=True, remove_qconfig=False)

    def _generate_qconfig_map(self, root, input_graph):
        def get_qconfig(module):
            return module.qconfig if hasattr(module, 'qconfig') else None

        self.qconfig_map = dict()
        for node in input_graph.nodes:
            if node.op == 'get_param':
                parent, _ = _parent_name(node.target)
                self.qconfig_map[node.name] = get_qconfig(self.modules[parent])
            elif node.op == 'call_function':
                self.qconfig_map[node.name] = get_qconfig(root)
            elif node.op == 'call_method':
                self_obj = node.args[0]
                # qconfig for call_method should be the same as the `self` object for the call
                self.qconfig_map[node.name] = self.qconfig_map[self_obj.name]
            elif node.op == 'call_module':
                self.qconfig_map[node.name] = get_qconfig(self.modules[node.target])

    def _prepare(self, model, qconfig_dict, inplace, is_dynamic_quant):
        assert not inplace, 'inplace prepare is not supported yet'
        input_root = model
        if not inplace:
            input_root = copy.deepcopy(input_root)

        input_graph = model.graph
        self.is_dynamic_quant = is_dynamic_quant
        # TODO: allow user specified patterns
        if self.is_dynamic_quant:
            self.patterns = get_dynamic_quant_patterns()
        else:
            self.patterns = get_quant_patterns()

        propagate_qconfig_(input_root, qconfig_dict)
        if input_root.training:
            self._qat_swap_modules(input_root)

        self.modules = dict(input_root.named_modules())

        # map from node name to qconfig, used in _find_matches
        self._generate_qconfig_map(input_root, input_graph)

        # match the patterns that will get quantized
        matches = self._find_matches(input_graph, self.modules, self.patterns)

        # find _inputs_ to matched nodes that are not quantized, these
        # have to be quantized, which requires measuring stats,
        # initialize an DefaultQuant object for each
        quants = self._find_quants(input_graph, matches)

        self.activation_post_process_map = dict()

        env = {}
        observed_graph = Graph()
        observed_node_names_set = set()

        def load_arg(a):
            return map_arg(a, lambda node: env[node.name])

        for node in input_graph.nodes:
            if node.name in observed_node_names_set:
                continue

            get_new_observer_name = get_new_attr_name_with_prefix('activation_post_process_')
            root_node, _, obj, qconfig = matches.get(node.name, (None, None, None, None))
            if root_node is None:
                env[node.name] = observed_graph.node_copy(node, load_arg)
            elif root_node is node:
                env[node.name] = observed_graph.node_copy(node, load_arg)

                def insert_observer(node, observer, device):
                    observer_name = get_new_observer_name(input_root)
                    setattr(input_root, observer_name, observer)
                    self.activation_post_process_map[node.name] = observer
                    env[node.name] = observed_graph.create_node('call_module', observer_name, (load_arg(node),), {})
                    observed_node_names_set.add(node.name)
                    if device:
                        getattr(input_root, observer_name).to(device)

                # don't need to insert observer for output in dynamic quantization
                if self.is_dynamic_quant:
                    continue

                if isinstance(obj, CopyNode):
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
                elif (isinstance(obj, Add) or isinstance(obj, Mul)) and not obj.all_nodes:
                    if node.args[0].name in observed_node_names_set:
                        observed_node_names_set.add(node.name)
                elif qconfig is not None and obj.all_nodes:
                    # observer for outputs
                    new_observer = qconfig.activation()
                    # respect device affinity when adding observers
                    device = assert_and_get_unique_device(input_root)
                    insert_observer(node, new_observer, device)
            else:
                env[node.name] = observed_graph.node_copy(node, load_arg)

            if node.name not in observed_node_names_set and node.name in quants:
                observer_name = get_new_observer_name(input_root)
                _, qconfig, is_weight = quants[node.name]
                if qconfig is not None:
                    new_observer = \
                        qconfig.weight() if is_weight else qconfig.activation()
                    # respect device affinity when adding observers
                    device = assert_and_get_unique_device(input_root)
                    if device:
                        new_observer.to(device)
                    self.activation_post_process_map[node.name] = new_observer
                    setattr(input_root, observer_name, self.activation_post_process_map[node.name])
                    env[node.name] = observed_graph.create_node('call_module', observer_name, (load_arg(node),), {})
                    observed_node_names_set.add(node.name)
        observed_graph.output(load_arg(input_graph.result))

        observed_module = GraphModule(input_root, observed_graph)
        self.save_state(observed_module)
        return observed_module

    def save_state(self, observed):
        observed._activation_post_process_map = self.activation_post_process_map
        observed._patterns = self.patterns
        observed._qconfig_map = self.qconfig_map

    def restore_state(self, observed):
        err_msg = 'please make sure the model is produced by prepare'
        assert hasattr(observed, '_activation_post_process_map'), 'did not found ' + \
            '_activation_post_process attribute ' + err_msg
        assert hasattr(observed, '_patterns'), 'did not found ' + \
            '_patterns attribute ' + err_msg
        assert hasattr(observed, '_qconfig_map'), 'did not found ' + \
            '_qconfig_map attribute ' + err_msg
        self.activation_post_process_map = observed._activation_post_process_map
        self.patterns = observed._patterns
        self.qconfig_map = observed._qconfig_map

    def prepare(self, model, qconfig_dict, inplace=False):
        return self._prepare(model, qconfig_dict, inplace, is_dynamic_quant=False)

    def prepare_dynamic(self, model, qconfig_dict, inplace=False):
        return self._prepare(model, qconfig_dict, inplace, is_dynamic_quant=True)

    def _run_weight_observers(self, observed):
        r''' Extract the subgraph that produces the weight for dynamically quantized
        node and run the subgraph to observe the weight.
        Note that the observers of dynamically quantized modules are run during
        the conversion step.
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

    def _convert(self, observed, inplace=False, debug=False, is_dynamic_quant=False):
        assert not inplace, 'inplace convert is not supported yet'
        self.restore_state(observed)
        self.is_dynamic_quant = is_dynamic_quant
        # run weight observers before inserting quant dequant nodes
        # for dynamic quantization
        if self.is_dynamic_quant:
            self._run_weight_observers(observed)

        # move to cpu since we only have quantized cpu kernels
        observed.eval().cpu()
        observed_root = observed
        observed_graph = observed.graph
        if not inplace:
            observed_root = copy.deepcopy(observed_root)

        self.modules = dict(observed_root.named_modules())

        matches = self._find_matches(observed.graph, self.modules, self.patterns)
        quants = self._find_quants(observed.graph, matches)
        self.quantized_graph = Graph()
        env = {}
        quant_env = {}

        def load_non_quantized(n):
            if n.name not in env:
                assert n.name in quant_env, \
                    'trying to load float node but did not find node:' + n.name + \
                    ' in quantized environment:' + str(quant_env)
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
                'node ' + n.name + ' does not exist in either of the environment'
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

        for node in observed_graph.nodes:
            root_node, matched, obj, qconfig = matches.get(node.name, (None, None, None, None))
            if root_node is node:
                result = obj.convert(self, node, load_arg)
                quantized = True
                # Need to get correct quantized/non-quantized state for the output of CopyNode
                if isinstance(obj, CopyNode):
                    assert node.op in [
                        'call_module',
                        'call_function',
                        'call_method'], \
                        'CopyNode of type ' + node.op + ' is not handled'
                    quantized = is_quantized(node.args[0])

                # output of dynamic quantization is not quantized
                if self.is_dynamic_quant:
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
                if node.target.split('.')[-1].startswith('activation_post_process_'):
                    observer_module = self.modules[node.target]
                    prev_node = node.args[0]
                    if prev_node.name in quant_env:
                        # if previous node is already quantized, we'll just remove the activation_post_process
                        quant_env[node.name] = quant_env[prev_node.name]
                        continue
                    # replace activation post process with quantization ops
                    parent_name = ''

                    scale, zero_point = observer_module.calculate_qparams()
                    dtype = observer_module.dtype

                    def is_per_channel(qscheme):
                        return qscheme == torch.per_channel_affine or \
                            qscheme == torch.per_channel_symmetric

                    if is_per_channel(observer_module.qscheme):
                        ch_axis = int(observer_module.ch_axis)
                        qparams = {'_scale_': scale, '_zero_point_': zero_point, '_axis': ch_axis, '_dtype_': dtype}
                        quantize_op = torch.quantize_per_channel
                    else:
                        scale = float(scale)
                        zero_point = int(zero_point)
                        qparams = {'_scale_': scale, '_zero_point_': zero_point, '_dtype_': dtype}
                        quantize_op = torch.quantize_per_tensor
                    i = 0

                    def noattr(module, qparams, i):
                        for name in qparams.keys():
                            if hasattr(module, name + str(i)):
                                return False
                        return True

                    def get_next_i(module, qparams):
                        i = 0
                        while not noattr(module, qparams, i):
                            i += 1
                        return i

                    parent_module = self.modules[parent_name]
                    i = get_next_i(parent_module, qparams)
                    inputs = [load_non_quantized(node.args[0])]
                    for key, value in qparams.items():
                        setattr(parent_module, key + str(i), value)
                        qparam_full_path = key + str(i)
                        if parent_name:
                            qparam_full_path = parent_name + '.' + qparam_full_path
                        inputs.append(self.quantized_graph.create_node('get_param', qparam_full_path))
                    quant_env[node.name] = self.quantized_graph.create_node('call_function', quantize_op, tuple(inputs), {})
                    continue
            # dequantize inputs for the node that are not quantized
            env[node.name] = self.quantized_graph.node_copy(node, load_non_quantized)

        self.quantized_graph.output(load_non_quantized(observed_graph.result))

        to_be_removed = []
        for name, _ in observed_root.named_modules():
            if name.split('.')[-1].startswith('activation_post_process_'):
                to_be_removed.append(name)
        for n in to_be_removed:
            delattr(observed_root, n)
        return GraphModule(observed_root, self.quantized_graph)

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
                    'get_param', packed_weight_name, (), {})
            elif prepack_node is not None:
                # remove the foled node
                continue
            else:
                # copy other nodes
                env[node.name] = folded_graph.node_copy(node, load_arg)
        folded_graph.output(load_arg(quantized_graph.result))
        return GraphModule(quantized_root, folded_graph)

    def convert(self, observed, inplace=False, debug=False, is_dynamic=False):
        quantized = self._convert(observed, inplace, debug, is_dynamic)
        if not debug:
            quantized = self._fold_weight(quantized)
        return quantized

    def _find_matches(self, graph, modules, patterns):
        match_map = {}  # node name -> (root_node, match_value?)
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
                            match_map[n.name] = (node, matched, value(self, node), self.qconfig_map[n.name])
                            all_matched.add(n.name)
                        # break after finding the first match
                        break
        return match_map

    def _find_quants(self, graph, matches):
        quants = {}

        def visit(node, qconfig):
            def visit_arg(arg):
                # note: we have to measure quantization information
                # even for nodes where we might not use it because it is already
                # quantized. This is because each match has the option to
                # say NotImplemented (if for instance, it is an __add__ and the data type is not appropriate)
                is_weight = False
                if isinstance(node, Node) and node.op == 'call_function' and node.target in WEIGHT_INDEX_DICT:
                    for i, node_arg in enumerate(node.args):
                        if arg is node_arg and i in WEIGHT_INDEX_DICT[node.target]:
                            is_weight = True
                if (not self.is_dynamic_quant) or is_weight:
                    # overwrite previous quant config
                    quants[arg.name] = (DefaultQuant(self, arg), qconfig, is_weight)
            return visit_arg

        for node in graph.nodes:
            if node.name in matches:
                root_node, matched, obj, qconfig = matches[node.name]
                # don't attach observer/fake_quant for CopyNode
                if isinstance(obj, CopyNode):
                    qconfig = None
                if root_node is node:
                    # matched[-1] is the first op in the sequence and
                    # matched[0] is the last op in the sequence
                    # inputs
                    map_arg(matched[-1].args, visit(matched[-1], qconfig))
                    map_arg(matched[-1].kwargs, visit(matched[-1], qconfig))
                    # output
                    map_arg(matched[0], visit(None, qconfig))
        return quants
