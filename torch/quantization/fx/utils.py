import re
import torch
from ..utils import is_per_tensor, is_per_channel

from torch.fx import GraphModule, map_arg

from torch.fx.graph import (
    Graph,
    Node,
)

from typing import Callable, Optional, List, Dict, Any

# turn foo.bar -> ['foo', 'bar']
def _parent_name(target):
    r = target.rsplit('.', 1)
    if len(r) == 1:
        return '', r[0]
    else:
        return r[0], r[1]

def graph_pretty_str(g, shorten=True) -> str:
    """Returns a printable representation of the ops in the graph of g.
    If shorten is True, tries to abbreviate fields.
    """
    built_in_func_re = re.compile('<built-in function (.*)>')
    built_in_meth_re = re.compile('<built-in method (.*) of type.*>')
    op_dict = {
        'placeholder': 'plchdr',
        'get_attr': 'gt_prm',
        'call_function': 'cl_fun',
        'call_module': 'cl_mod',
        'call_method': 'cl_meth',
    }

    max_lens = {}
    col_names = ("name", "op", "target", "args", "kwargs")
    for s in col_names:
        max_lens[s] = len(s)

    results = []
    for n in g.nodes:

        # activation_post_process_0 -> obs_0
        name = str(n.name)
        if shorten:
            name = name.replace("activation_post_process", "obs")

        op = str(n.op)
        # placeholder -> plchdr, and so on
        if shorten and op in op_dict:
            op = op_dict[op]

        target = str(n.target)
        # <built-in function foo> -> <bi_fun foo>, and so on
        if shorten:
            built_in_func = built_in_func_re.search(target)
            if built_in_func:
                target = f"<bi_fun {built_in_func.group(1)}>"
            built_in_meth = built_in_meth_re.search(target)
            if built_in_meth:
                target = f"<bi_meth {built_in_meth.group(1)}>"
            target = target.replace("activation_post_process", "obs")

        args = str(n.args)
        if shorten:
            args = args.replace("activation_post_process", "obs")

        kwargs = str(n.kwargs)

        # calculate maximum length of each column, so we can tabulate properly
        for k, v in zip(col_names, (name, op, target, args, kwargs)):
            max_lens[k] = max(max_lens[k], len(v))
        results.append([name, op, target, args, kwargs])

    res_str = ""
    format_str = "{:<{name}} {:<{op}} {:<{target}} {:<{args}} {:<{kwargs}}\n"
    res_str += format_str.format(*col_names, **max_lens)
    for result in results:
        res_str += format_str.format(*result, **max_lens)

    # print an exra note on abbreviations which change attribute names,
    # since users will have to un-abbreviate for further debugging
    if shorten:
        res_str += "*obs_{n} = activation_post_process_{n}\n"
    return res_str

def get_per_tensor_qparams(activation_post_process):
    assert is_per_tensor(activation_post_process.qscheme), 'Only per tensor quantization is supported'
    scale, zero_point = activation_post_process.calculate_qparams()
    scale = float(scale)
    zero_point = int(zero_point)
    dtype = activation_post_process.dtype
    return scale, zero_point, dtype

def get_quantize_op_and_qparams(activation_post_process):
    ''' Given an activation_post_process module,
    return quantize op(e.g. quantize_per_tensor) and a dictionary
    of extracted qparams from the module
    '''
    scale, zero_point = activation_post_process.calculate_qparams()
    dtype = activation_post_process.dtype
    if is_per_channel(activation_post_process.qscheme):
        ch_axis = int(activation_post_process.ch_axis)
        qparams = {'_scale_': scale, '_zero_point_': zero_point, '_axis_': ch_axis, '_dtype_': dtype}
        quantize_op = torch.quantize_per_channel
    else:
        scale = float(scale)
        zero_point = int(zero_point)
        qparams = {'_scale_': scale, '_zero_point_': zero_point, '_dtype_': dtype}
        quantize_op = torch.quantize_per_tensor  # type: ignore
    return quantize_op, qparams

def quantize_node(root_module, graph, node, activation_post_process):
    ''' Add quantization nodes for given node to graph
    with the qparams calculated from activation_post_process module
    e.g. Given input `node` in `node = self.conv(x)`, insert node:
    `quantized_node = torch.quantize_per_tensor(x, self._scale_0, self._zer_point_0, self._dtype_0)`
    where self._scale_0, self._zero_point_0 and self._dtype_0 are
    calculated from `activation_post_process`
    '''
    def module_has_qparams_attr_with_index(module, qparams, i):
        for name in qparams.keys():
            if hasattr(module, name + str(i)):
                return True
        return False

    def get_next_qparams_idx(module, qparams):
        idx = 0
        while module_has_qparams_attr_with_index(module, qparams, idx):
            idx += 1
        return idx

    quantize_op, qparams = get_quantize_op_and_qparams(activation_post_process)
    idx = get_next_qparams_idx(root_module, qparams)
    inputs = [node]
    for key, value in qparams.items():
        setattr(root_module, key + str(idx), value)
        qparam_full_path = key + str(idx)
        inputs.append(graph.create_node('get_attr', qparam_full_path))
    return graph.create_node('call_function', quantize_op, tuple(inputs), {})

def get_custom_module_class_keys(custom_config_dict, custom_config_dict_key):
    r""" Get all the unique custom module keys in the custom config dict
    e.g.
    Input:
    custom_config_dict = {
        "float_to_observed_custom_module_class": {
           "static": {
               CustomModule1: ObservedCustomModule
           },
           "dynamic": {
               CustomModule2: DynamicObservedCustomModule
           },
           "weight_only": {
               CustomModule3: WeightOnlyObservedCustomModule
           },
        },
    }

    Output:
    # extract all the keys in "static", "dynamic" and "weight_only" dict
    [CustomModule1, CustomModule2, CustomModule3]
    """
    # using set to dedup
    float_custom_module_classes = set()
    custom_module_mapping = custom_config_dict.get(custom_config_dict_key, {})
    for quant_mode in ["static", "dynamic", "weight_only"]:
        quant_mode_custom_module_config = custom_module_mapping.get(quant_mode, {})
        quant_mode_custom_module_classes = set(quant_mode_custom_module_config.keys())
        float_custom_module_classes |= quant_mode_custom_module_classes
    return list(float_custom_module_classes)

def get_linear_prepack_op_for_dtype(dtype):
    if dtype == torch.float16:
        return torch.ops.quantized.linear_prepack_fp16
    elif dtype == torch.qint8:
        return torch.ops.quantized.linear_prepack
    else:
        raise Exception("can't get linear prepack op for dtype:", dtype)

# Returns a function that can get a new attribute name for module with given
# prefix, for example,
# >> get_new_observer_name = get_new_attr_name_with_prefix('_observer')
# >> new_name = get_new_observer_name(module)
# new_name will be an unused attribute name on module, e.g. `_observer_1`
def get_new_attr_name_with_prefix(prefix: str) -> Callable:
    def get_new_attr_name(module: torch.nn.Module):
        def get_attr_name(i: int):
            return prefix + str(i)
        i = 0
        attr_name = get_attr_name(i)
        while hasattr(module, attr_name):
            i += 1
            attr_name = get_attr_name(i)
        return attr_name
    return get_new_attr_name

def collect_producer_nodes(node: Node) -> Optional[List[Node]]:
    r''' Starting from a target node, trace back until we hit inpu or
    getattr node. This is used to extract the chain of operators
    starting from getattr to the target node, for example
    def forward(self, x):
      observed = self.observer(self.weight)
      return F.linear(x, observed)
    collect_producer_nodes(observed) will either return a list of nodes that
    produces the observed node or None if we can't extract a self contained
    graph without free variables(inputs of the forward function).
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

def graph_module_from_producer_nodes(
        root: GraphModule, producer_nodes: List[Node]) -> GraphModule:
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
    env: Dict[Any, Any] = {}

    def load_arg(a):
        return map_arg(a, lambda node: env[node])
    for producer_node in producer_nodes:
        env[producer_node] = graph.node_copy(producer_node, load_arg)
    graph.output(load_arg(producer_nodes[-1]))
    graph_module = GraphModule(root, graph)
    return graph_module

def assert_and_get_unique_device(module: torch.nn.Module) -> Any:
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
