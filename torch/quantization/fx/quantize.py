import torch

from torch.quantization import (
    propagate_qconfig_,
    convert,
)

from torch.quantization.default_mappings import (
    DEFAULT_QAT_MODULE_MAPPING,
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
    matches,
    get_quant_patterns,
    get_dynamic_quant_patterns,
)

from .utils import _parent_name

# TODO before land: discuss is we want to namespace this instead,
#   something like patterns.CopyNode instead of CopyNode
from .patterns import *

import copy
import enum

# Quantization type (dynamic quantization, static quantization).
# Should match the c++ enum in quantization_type.h
class QuantType(enum.IntEnum):
    DYNAMIC = 0
    STATIC = 1
    QAT = 2

# A dictionary for querying the weight index for a given op
WEIGHT_INDEX_DICT = {
    torch.nn.functional.conv2d : [1],
    torch.nn.functional.linear : [1],
}

class Quantizer:
    def __init__(self):
        # mapping from matched node to activation_post_process
        # must be filled before convert
        self.activation_post_process_map = None

    def _qat_swap_modules(self, root):
        convert(root, mapping=DEFAULT_QAT_MODULE_MAPPING, inplace=True, remove_qconfig=False)

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

    def _prepare(self, model, qconfig_dict, inplace, quant_type):
        input_root = model.root
        if not inplace:
            input_root = copy.deepcopy(input_root)

        input_graph = model.graph
        self.quant_type = quant_type
        # TODO: allow user specified patterns
        if self.quant_type == QuantType.DYNAMIC:
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
        observed = set()

        def load_arg(a):
            return map_arg(a, lambda node: env[node.name])

        for node in input_graph.nodes:
            if node.name in observed:
                continue

            def get_new_observer_name(parent_module):
                i = 0

                def get_observer_name(i):
                    return 'activation_post_process_' + str(i)
                observer_name = get_observer_name(i)
                while hasattr(parent_module, observer_name):
                    i += 1
                    observer_name = get_observer_name(i)
                return observer_name
            root_node, _, obj, qconfig = matches.get(node.name, (None, None, None, None))
            if root_node is None:
                env[node.name] = observed_graph.node_copy(node, load_arg)
            elif root_node is node:
                env[node.name] = observed_graph.node_copy(node, load_arg)

                def insert_observer(node, observer):
                    observer_name = get_new_observer_name(input_root)
                    setattr(input_root, observer_name, observer)
                    self.activation_post_process_map[node.name] = observer
                    env[node.name] = observed_graph.create_node('call_module', observer_name, [load_arg(node)], {})
                    observed.add(node.name)

                # don't need to insert observer for output in dynamic quantization
                if self.quant_type == QuantType.DYNAMIC:
                    continue

                if isinstance(obj, CopyNode):
                    assert node.op in [
                        'call_module',
                        'call_function',
                        'call_method'], \
                        'CopyNode of type ' + node.op + ' is not handled'

                    def is_observed(input_arg):
                        if isinstance(input_arg, Node):
                            return input_arg.name in observed
                        elif isinstance(input_arg, list):
                            return all(map(is_observed, input_arg))
                    # propagate observed property from input
                    if is_observed(node.args[0]):
                        observed.add(node.name)
                elif (isinstance(obj, Add) or isinstance(obj, Mul)) and not obj.all_nodes:
                    if node.args[0].name in observed:
                        observed.add(node.name)
                elif qconfig is not None and obj.all_nodes:
                    # observer for outputs
                    insert_observer(node, qconfig.activation())
            else:
                env[node.name] = observed_graph.node_copy(node, load_arg)

            if node.name not in observed and node.name in quants:
                observer_name = get_new_observer_name(input_root)
                _, qconfig, is_weight = quants[node.name]
                if qconfig is not None:
                    self.activation_post_process_map[node.name] = qconfig.weight() if is_weight else qconfig.activation()
                    setattr(input_root, observer_name, self.activation_post_process_map[node.name])
                    env[node.name] = observed_graph.create_node('call_module', observer_name, [load_arg(node)], {})
                    observed.add(node.name)
        observed_graph.output(load_arg(input_graph.result))

        return GraphModule(input_root, observed_graph)

    def prepare(self, model, qconfig_dict, inplace=False):
        return self._prepare(model, qconfig_dict, inplace, quant_type=QuantType.STATIC)

    def prepare_dynamic(self, model, qconfig_dict, inplace=False):
        return self._prepare(model, qconfig_dict, inplace, quant_type=QuantType.DYNAMIC)

    def convert(self, observed, inplace=False, debug=False):
        assert self.activation_post_process_map is not None
        # move to cpu since we only have quantized cpu kernels
        observed.eval().cpu()
        observed_root = observed.root
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
            if quantized is a list, then arg should be a list and the args with corresponding
            indexes will be quantized
            if quantized is a boolean, then all args will be quantized/not quantized
            if quantized is None, then we'll load the node as long as it exists
            """
            assert quantized is None or isinstance(quantized, (tuple, list, bool)), type(quantized)

            def load_arg_impl(arg):
                if quantized is None:
                    return map_arg(arg, load_x)
                if isinstance(quantized, bool):
                    return map_arg(arg, load_quantized if quantized else load_non_quantized)
                elif isinstance(quantized, (tuple, list)):
                    assert isinstance(arg, (tuple, list)), arg
                    loaded_arg = []
                    # for now, we only support quantizing positional arguments
                    for i, a in enumerate(arg):
                        if i in quantized:
                            loaded_arg.append(map_arg(a, load_quantized))
                        else:
                            loaded_arg.append(map_arg(a, load_non_quantized))
                    return type(arg)(loaded_arg)
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

                if self.quant_type == QuantType.DYNAMIC:
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
                    # TODO: per channel
                    scale = float(scale)
                    zero_point = int(zero_point)
                    dtype = observer_module.dtype
                    qparams = {'_scale_': scale, '_zero_point_': zero_point, '_dtype_': dtype}
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
                        inputs.append(self.quantized_graph.get_param(qparam_full_path))
                    quant_env[node.name] = self.quantized_graph.create_node('call_function', torch.quantize_per_tensor, inputs, {})
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
                    if matches(modules, node, pattern):
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
                if self.quant_type != QuantType.DYNAMIC or is_weight:
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
