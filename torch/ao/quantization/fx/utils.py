import copy
import re
import torch
import torch.nn as nn
from torch.ao.quantization import (
    QConfigAny,
    QuantType,
)
from torch.ao.quantization.backend_config import (
    BackendConfig,
    DTypeWithConstraints,
)
from torch.ao.quantization.fake_quantize import FakeQuantizeBase
from torch.ao.quantization.observer import ObserverBase
from torch.ao.quantization.stubs import DeQuantStub
from torch.ao.quantization.utils import (
    activation_is_statically_quantized,
    is_per_tensor,
    is_per_channel,
    to_underlying_dtype,
)
from torch.ao.quantization.observer import _is_activation_post_process

from torch.fx import GraphModule, map_arg

from torch.fx.graph import (
    Graph,
    Node,
)
from .custom_config import PrepareCustomConfig
# importing the lib so that the quantized_decomposed ops are registered
from ._decomposed import quantized_decomposed_lib  # noqa: F401

from typing import Callable, Optional, List, Dict, Any, Set, Tuple, Union, Type
from collections import namedtuple
import operator
import warnings

# TODO: revisit this list. Many helper methods shouldn't be public
__all__ = [
    "all_node_args_except_first",
    "all_node_args_have_no_tensors",
    "assert_and_get_unique_device",
    "collect_producer_nodes",
    "create_getattr_from_value",
    "create_node_from_old_node_preserve_meta",
    "create_qparam_nodes",
    "EMPTY_ARG_DICT",
    "get_custom_module_class_keys",
    "get_linear_prepack_op_for_dtype",
    "get_new_attr_name_with_prefix",
    "get_non_observable_arg_indexes_and_types",
    "get_per_tensor_qparams",
    "get_qconv_op",
    "get_qconv_prepack_op",
    "get_quantize_node_info",
    "get_skipped_module_name_and_classes",
    "graph_module_from_producer_nodes",
    "graph_pretty_str",
    "is_get_tensor_info_node",
    "maybe_get_next_module",
    "NodeInfo",
    "node_return_type_is_int",
    "node_arg_is_bias",
    "node_arg_is_weight",
    "NON_OBSERVABLE_ARG_DICT",
    "NON_QUANTIZABLE_WEIGHT_OPS",
    "return_arg_list",
]

NON_QUANTIZABLE_WEIGHT_OPS = {torch.nn.functional.layer_norm, torch.nn.functional.group_norm, torch.nn.functional.instance_norm}

def node_arg_is_weight(node: Node, arg: Any, backend_config: BackendConfig) -> bool:
    """Returns if node arg is weight"""
    if isinstance(node, Node) and node.op == "call_function" and node.target in backend_config.configs:
        weight_index = backend_config.configs[node.target]._input_type_to_index.get("weight")
        if weight_index is not None and weight_index < len(node.args) and node.args[weight_index] is arg:
            return True
        return node.kwargs.get("weight") is arg
    return False

def node_arg_is_bias(node: Node, arg: Any, backend_config: BackendConfig) -> bool:
    """Returns if node arg is bias"""
    if isinstance(node, Node) and node.op == "call_function" and node.target in backend_config.configs:
        bias_index = backend_config.configs[node.target]._input_type_to_index.get("bias")
        if bias_index is not None and bias_index < len(node.args) and node.args[bias_index] is arg:
            return True
        return node.kwargs.get("bias") is arg
    return False

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

def get_quantize_node_info(
    activation_post_process: Callable,
    is_decomposed: bool
) -> Optional[Tuple[str, Union[Callable[..., Any], str], Dict[str, Any]]]:
    """ Extract information about quantize op from activation_post_process module
    Args:
      * `activation_post_process`: observer module instance or fake quant module instance
        after calibration/QAT
      * `is_decomposed`: a boolean flag to indicate whether we want to use the
        quantize operator for decomposed quantized tensor (torch.ops.quantized_decomposed.quantize_per_tensor) or default/standalone
        quantized tensor (torch.quantize_per_tensor)

    Returns
        node_type(e.g. call_function), quantize op(e.g. quantize_per_tensor) and a dictionary
        of extracted qparams from the module
    """
    dtype = activation_post_process.dtype  # type: ignore[attr-defined]
    compute_dtype = None
    if hasattr(activation_post_process, "compute_dtype"):
        compute_dtype = activation_post_process.compute_dtype  # type: ignore[attr-defined]
    quantize_op : Optional[Union[Callable, str]] = None
    if dtype in [torch.quint8, torch.qint8] and \
            not hasattr(activation_post_process, 'compute_dtype'):
        node_type = "call_function"
        scale, zero_point = activation_post_process.calculate_qparams()  # type: ignore[attr-defined]
        if is_per_channel(activation_post_process.qscheme):  # type: ignore[attr-defined]
            ch_axis = int(activation_post_process.ch_axis)  # type: ignore[attr-defined]
            qparams = {"_scale_": scale, "_zero_point_": zero_point, "_axis_": ch_axis, "_dtype_": dtype}
            if is_decomposed:
                raise NotImplementedError("decomposed quantize_per_channel op not implemented yet")
            else:
                quantize_op = torch.quantize_per_channel
        else:
            scale = float(scale)
            zero_point = int(zero_point)
            if is_decomposed:
                quant_min = activation_post_process.quant_min  # type: ignore[attr-defined]
                quant_max = activation_post_process.quant_max  # type: ignore[attr-defined]
                dtype = to_underlying_dtype(dtype)
                qparams = {
                    "_scale_": scale,
                    "_zero_point_": zero_point,
                    "_quant_min": quant_min,
                    "_quant_max": quant_max,
                    "_dtype_": dtype
                }
                quantize_op = torch.ops.quantized_decomposed.quantize_per_tensor
            else:
                qparams = {"_scale_": scale, "_zero_point_": zero_point, "_dtype_": dtype}
                quantize_op = torch.quantize_per_tensor
    elif compute_dtype in [torch.quint8, torch.qint8, torch.float16]:
        # TODO(future PR): switch compute_dtype to is_dynamic
        # dynamic quantization
        node_type = "call_function"
        if is_decomposed:
            raise NotImplementedError("decomposed quantize_per_tensor_dynamic op not implemented yet")
        else:
            quantize_op = torch.quantize_per_tensor_dynamic
        # TODO: get reduce range from observer
        # reduce_range = activation_post_process.reduce_range
        reduce_range = torch.backends.quantized.engine in ("fbgemm", "x86")
        qparams = {"_dtype_": compute_dtype, "_reduce_range_": reduce_range}
    elif dtype == torch.float16:
        node_type = "call_method"
        quantize_op = "to"
        qparams = {"_dtype_": dtype}
    else:
        warnings.warn(f"Unsupported activation_post_process in get_quantize_node_info: {activation_post_process}")
        return None
    return node_type, quantize_op, qparams  # type: ignore[return-value]

# Keep it here for BC in torch.quantization namespace, we can remove it after
# we deprecate the torch.quantization namespace
quantize_node = NotImplemented

def get_custom_module_class_keys(custom_module_mapping: Dict[QuantType, Dict[Type, Type]]) -> List[Any]:
    r""" Get all the unique custom module keys in the custom config dict
    e.g.
    Input:
    {
        QuantType.STATIC: {
            CustomModule1: ObservedCustomModule
        },
        QuantType.DYNAMIC: {
            CustomModule2: DynamicObservedCustomModule
        },
        QuantType.WEIGHT_ONLY: {
            CustomModule3: WeightOnlyObservedCustomModule
        },
    }

    Output:
    # extract the keys across all inner STATIC, DYNAMIC, and WEIGHT_ONLY dicts
    [CustomModule1, CustomModule2, CustomModule3]
    """
    # using set to dedup
    float_custom_module_classes : Set[Any] = set()
    for quant_mode in [QuantType.STATIC, QuantType.DYNAMIC, QuantType.WEIGHT_ONLY]:
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

def get_qconv_prepack_op(conv_op: Callable) -> Callable:
    prepack_ops = {
        torch.nn.functional.conv1d: torch.ops.quantized.conv1d_prepack,
        torch.nn.functional.conv2d: torch.ops.quantized.conv2d_prepack,
        torch.nn.functional.conv3d: torch.ops.quantized.conv3d_prepack
    }
    prepack_op = prepack_ops.get(conv_op, None)
    assert prepack_op, "Didn't find prepack op for {}".format(conv_op)
    return prepack_op

def get_qconv_op(conv_op: Callable, has_relu: bool) -> Callable:
    qconv_op = {
        # has relu
        True: {
            torch.nn.functional.conv1d: torch.ops.quantized.conv1d_relu,
            torch.nn.functional.conv2d: torch.ops.quantized.conv2d_relu,
            torch.nn.functional.conv3d: torch.ops.quantized.conv3d_relu
        },
        False: {
            torch.nn.functional.conv1d: torch.ops.quantized.conv1d,
            torch.nn.functional.conv2d: torch.ops.quantized.conv2d,
            torch.nn.functional.conv3d: torch.ops.quantized.conv3d
        }
    }
    qconv = qconv_op[has_relu].get(conv_op)
    assert qconv, "Can't find corresponding quantized conv op for {} {}".format(conv_op, has_relu)
    return qconv

# Returns a function that can get a new attribute name for module with given
# prefix, for example,
# >> get_new_observer_name = get_new_attr_name_with_prefix('_observer')
# >> new_name = get_new_observer_name(module)
# new_name will be an unused attribute name on module, e.g. `_observer_1`
def get_new_attr_name_with_prefix(prefix: str) -> Callable:
    prefix = prefix.replace(".", "_")

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

def create_getattr_from_value(module: torch.nn.Module, graph: Graph, prefix: str, value: Any) -> Node:
    """
    Given a value of any type, creates a getattr node corresponding to the value and
    registers the value as a buffer to the module.
    """
    get_new_attr_name = get_new_attr_name_with_prefix(prefix)
    attr_name = get_new_attr_name(module)
    device = assert_and_get_unique_device(module)
    new_value = value.clone().detach() if isinstance(value, torch.Tensor) \
        else torch.tensor(value, device=device)
    module.register_buffer(attr_name, new_value)
    # Create get_attr with value
    attr_node = graph.create_node("get_attr", attr_name)
    return attr_node

def create_qparam_nodes(
        node_name: str,
        scale: Any,
        zero_point: Any,
        modules: Dict[str, torch.nn.Module],
        quantized_graph: Graph,
        node_name_to_scope: Dict[str, Tuple[str, type]]
) -> Tuple[Node, Node]:
    """
    Create getattr nodes in the quantized graph for scale and zero point values.
    The nodes are registered with the root_module of the model.
    """
    root_module = modules['']
    module_path, _ = node_name_to_scope[node_name]
    scale_node = create_getattr_from_value(root_module, quantized_graph, (module_path + "_scale_"), scale)
    zero_point_node = create_getattr_from_value(root_module, quantized_graph, (module_path + "_zero_point_"), zero_point)
    return (scale_node, zero_point_node)


def all_node_args_have_no_tensors(node: Node, modules: Dict[str, torch.nn.Module], cache: Dict[Node, bool]) -> bool:
    """
    If we know for sure that all of this node's args have no
    tensors (are primitives), return True.  If we either
    find a tensor or are not sure, return False. Note: this
    function is not exact.
    """
    if cache and node in cache:
        return cache[node]

    result = False  # will be overwritten
    if not isinstance(node, Node):
        result = True
    elif node.op == 'placeholder':
        result = False
    elif node.op == 'call_module':
        assert isinstance(node.target, str)
        if _is_activation_post_process(modules[node.target]):
            result = all_node_args_have_no_tensors(node.args[0], modules, cache)  # type: ignore[arg-type]
    elif node.op == 'call_module':
        result = False
    elif node.op == 'call_function' and node.target is operator.getitem:
        result = all_node_args_have_no_tensors(node.args[0], modules, cache)  # type: ignore[arg-type]
    elif node.op == 'get_attr':
        result = False
    elif node.target is getattr and node.args[1] in ['ndim', 'shape']:
        # x1 = x0.ndim
        result = True
    elif node.op == 'call_method' and node.target == 'size':
        # x1 = x0.size(0)
        result = True
    else:
        found_one_tensor = False
        for arg in node.args:
            if isinstance(arg, list):
                for list_el in arg:
                    if isinstance(list_el, Node):
                        this_list_el_args_have_no_tensors = \
                            all_node_args_have_no_tensors(list_el, modules, cache)
                        found_one_tensor = found_one_tensor or \
                            (not this_list_el_args_have_no_tensors)
                        # If found_one_tensor is True, there is no point in
                        # recursing further as the end result will always
                        # be True.
                        # TODO(future PR): remove this entire function  and
                        # change to dtype inference without recursion.
                        if found_one_tensor:
                            result = not found_one_tensor
                            if cache:
                                cache[node] = result
                            return result
            elif isinstance(arg, int):
                pass
            else:
                if isinstance(arg, Node):
                    this_arg_args_have_no_tensors = all_node_args_have_no_tensors(arg, modules, cache)
                    found_one_tensor = found_one_tensor or \
                        (not this_arg_args_have_no_tensors)
                    # If found_one_tensor is True, there is no point in
                    # recursing further as the end result will always
                    # be True.
                    # TODO(future PR): remove this entire function  and
                    # change to dtype inference without recursion.
                    if found_one_tensor:
                        result = not found_one_tensor
                        if cache:
                            cache[node] = result
                        return result
                else:
                    found_one_tensor = True
            result = not found_one_tensor
    if cache:
        cache[node] = result
    return result

def all_node_args_except_first(node: Node) -> List[int]:
    """
    Returns all node arg indices after first
    """
    return list(range(1, len(node.args)))

def return_arg_list(arg_indices: List[int]) -> Callable[[Node], List[int]]:
    """
    Constructs a function that takes a node as arg and returns the arg_indices
    that are valid for node.args
    """
    def arg_indices_func(node: Node) -> List[int]:
        return [i for i in arg_indices if i < len(node.args)]
    return arg_indices_func

NodeInfo = namedtuple("NodeInfo", "op target")

# this dict identifies which indices of a node are non tensors
# so that they can be propagated correctly since inserting observers
# for them would cause errors

NON_OBSERVABLE_ARG_DICT: Dict[NodeInfo, Dict[Union[type, torch.dtype], Callable[[Node], List[int]]]] = {
    NodeInfo("call_method", "masked_fill") : {
        torch.bool: return_arg_list([1]),
        float: return_arg_list([2])
    },
    NodeInfo("call_method", "permute") : {
        int: all_node_args_except_first
    },
    NodeInfo("call_method", "repeat") : {
        int: all_node_args_except_first
    },
    NodeInfo("call_method", "reshape") : {
        int: all_node_args_except_first
    },
    NodeInfo("call_method", "size") : {
        int: return_arg_list([1])
    },
    NodeInfo("call_method", "transpose") : {
        int: all_node_args_except_first
    },
    NodeInfo("call_method", torch.transpose) : {
        int: all_node_args_except_first
    },
    NodeInfo("call_method", "unsqueeze") : {
        int: return_arg_list([1])
    },
    NodeInfo("call_method", "unsqueeze_") : {
        int: return_arg_list([1])
    },
    NodeInfo("call_method", torch.unsqueeze) : {
        int: return_arg_list([1])
    },
    NodeInfo("call_method", "view") : {
        int: all_node_args_except_first
    },
}

EMPTY_ARG_DICT: Dict[Union[type, torch.dtype], Callable[[Node], List[int]]] = {}

def get_non_observable_arg_indexes_and_types(node: Node) -> Dict[Union[type, torch.dtype], Callable[[Node], List[int]]]:
    """
    Returns a dict with of non float tensor types as keys and values which correspond to a
    function to retrieve the list (which takes the node as an argument)
    """
    info = NodeInfo(node.op, node.target)

    return NON_OBSERVABLE_ARG_DICT.get(info, EMPTY_ARG_DICT)

def node_return_type_is_int(node: Node) -> bool:
    """
    Returns true if this node results in an integer, even if some of the args
    are Tensors.
    """
    return node.op == 'call_method' and node.target == 'size'


def is_get_tensor_info_node(node: Node) -> bool:
    """ Returns True if this node is a node that takes a Tensor as input and output some
    meta information about the Tensor, e.g. shape, size etc.
    """
    result: bool = \
        node.op == "call_function" and node.target == getattr and node.args[1] == "shape"  # type: ignore[assignment]
    return result

def maybe_get_next_module(
    node: Node,
    modules: Dict[str, nn.Module],
    target_module_type: Optional[Type[nn.Module]] = None,
    target_functional_type: Any = None,
) -> Optional[Node]:
    """ Gets the next module that matches what is needed in
    is_target_module_type if it exists

    Args:
        node: The node whose users we want to look at
        target_module_type: Module type that we want to check
        target_functional_type: Functional type that we want to check
    """

    for user, _ in node.users.items():
        if user.op == 'call_module' and target_module_type is not None and \
           isinstance(modules[str(user.target)], target_module_type):
            return user
        elif (user.op == 'call_function' and target_functional_type is not None and
              user.target == target_functional_type):
            return user

    return None

def create_node_from_old_node_preserve_meta(
    quantized_graph: Graph,
    create_node_args: Tuple[Any, ...],
    old_node: Node,
) -> Node:
    """
    Creates `new_node` and copies the necessary metadata to it from `old_node`.
    """
    new_node = quantized_graph.create_node(*create_node_args)
    new_node.stack_trace = old_node.stack_trace
    return new_node

def get_skipped_module_name_and_classes(
        prepare_custom_config: PrepareCustomConfig,
        is_standalone_module: bool) -> Tuple[List[str], List[Type[Any]]]:
    skipped_module_names = copy.copy(prepare_custom_config.non_traceable_module_names)
    skipped_module_classes = copy.copy(prepare_custom_config.non_traceable_module_classes)
    if not is_standalone_module:
        # standalone module and custom module config are applied in top level module
        skipped_module_names += list(prepare_custom_config.standalone_module_names.keys())
        skipped_module_classes += list(prepare_custom_config.standalone_module_classes.keys())
        skipped_module_classes += get_custom_module_class_keys(prepare_custom_config.float_to_observed_mapping)

    return skipped_module_names, skipped_module_classes

def _is_custom_module_lstm(
        node: Node,
        named_modules: Dict[str, torch.nn.Module],
        qconfig: QConfigAny = None,
        # QuantizeHandler, but we cannot include the type here due to circular imports
        qhandler: Optional[Any] = None,
) -> bool:
    """
    Return whether this refers to the custom module LSTM flow.
    """
    mod = _get_module(node, named_modules)
    if qconfig is not None and qhandler is not None:
        assert isinstance(qhandler, torch.ao.quantization.fx.quantization_patterns.QuantizeHandler)  # type: ignore[attr-defined]
        return isinstance(mod, torch.nn.LSTM) and \
            activation_is_statically_quantized(qconfig) and \
            qhandler.is_custom_module()
    else:
        return isinstance(mod, torch.ao.nn.quantizable.LSTM)

def _get_module(node: Node, named_modules: Dict[str, torch.nn.Module]) -> Optional[torch.nn.Module]:
    """
    If `node` refers to a call_module node, return the module, else None.
    """
    if node.op == "call_module" and str(node.target) in named_modules:
        return named_modules[str(node.target)]
    else:
        return None

def _insert_dequant_stub(
    node: Node,
    model: torch.nn.Module,
    named_modules: Dict[str, torch.nn.Module],
    graph: Graph,
) -> Node:
    """
    Attach a `DeQuantStub` to the model and create a node that calls this
    `DeQuantStub` on the output of `node`, similar to how observers are inserted.
    """
    prefix = "dequant_stub_"
    get_new_dequant_stub_name = get_new_attr_name_with_prefix(prefix)
    dequant_stub_name = get_new_dequant_stub_name(model)
    dequant_stub = DeQuantStub()
    setattr(model, dequant_stub_name, dequant_stub)
    named_modules[dequant_stub_name] = dequant_stub
    with graph.inserting_after(node):
        return graph.call_module(dequant_stub_name, (node,))

def _insert_dequant_stubs_for_custom_module_lstm_output(
    node: Node,
    model: torch.nn.Module,
    named_modules: Dict[str, torch.nn.Module],
    graph: Graph,
) -> Node:
    """
    Insert DeQuantStubs after each internal output node of custom module LSTM.

    Custom module LSTM outputs are nested tuples of the sturcture (output, (hidden0, hidden1)),
    Since we cannot dequantize a tuple as a whole, we must first break down the tuple into its
    components through `getitem`. This function transforms the graph as follows:

      (1) Split the LSTM node into (output, (hidden0, hidden1))
      (2) Insert a DeQuantStub after each internal node
      (3) Recombine the DeQuantStubs into the same structure as before
      (4) Reroute all consumers of the original LSTM node and its sub-nodes
          (e.g. lstm[0])

    Before:
                   lstm_output
                        |
                        v
                  original_user(s)
    After:
                   lstm_output
                  /           \\
                 /  (getitem)  \\
                /               \\
               v                 v
             output            hidden
               |               /   \\
         (DeQuantStub)        (getitem)
               |             /       \\
               v            v         v
           output_dq     hidden0    hidden1
               |            |         |
               |    (DeQuantStub) (DeQuantStub)
               |            |         |
               |            v         v
               |      hidden0_dq  hidden1_dq
               |            \\       /
               |              (tuple)
               |              \\   /
               |               v  v
               |             hidden_dq
               \\               /
                \\   (tuple)   /
                 v            v
                 lstm_output_dq
                       |
                       v
                original_user(s)

    For step (4), reroute all users of the original LSTM node(s) as follows:
      lstm_output -> lstm_output_dq
      lstm_output[0] -> output_dq
      lstm_output[1] -> hidden_dq
      lstm_output[1][0] -> hidden0_dq
      lstm_output[1][1] -> hidden1_dq

    Return the node `lstm_output_dq`.
    """
    # (1) Split the LSTM node into (output, (hidden0, hidden1))
    # (2) Insert a DeQuantStub after each internal node
    with graph.inserting_after(node):
        output = graph.call_function(operator.getitem, (node, 0))
        output_dq = _insert_dequant_stub(output, model, named_modules, graph)
    with graph.inserting_after(output_dq):
        hidden = graph.call_function(operator.getitem, (node, 1))
    with graph.inserting_after(hidden):
        hidden0 = graph.call_function(operator.getitem, (hidden, 0))
        hidden0_dq = _insert_dequant_stub(hidden0, model, named_modules, graph)
    with graph.inserting_after(hidden0_dq):
        hidden1 = graph.call_function(operator.getitem, (hidden, 1))
        hidden1_dq = _insert_dequant_stub(hidden1, model, named_modules, graph)

    # (3) Recombine the DeQuantStubs into the same structure as before
    with graph.inserting_after(hidden1_dq):
        hidden_dq = graph.call_function(tuple, ([hidden0_dq, hidden1_dq],))
    with graph.inserting_after(hidden_dq):
        lstm_output_dq = graph.call_function(tuple, ([output_dq, hidden_dq],))

    # (4) Reroute all consumers of the original LSTM node and its sub-nodes
    for user in list(node.users.keys()):
        if user != output and user != hidden:
            user.replace_input_with(node, lstm_output_dq)
    # The getitem and tuple nodes we added here may interfere with reference quantized
    # pattern matching, so we need to redirect the consumers of internal nodes to the
    # corresponding nodes with DeQuantStubs (e.g. lstm_output_dq[0] -> output_dq) attached,
    # in order to preserve reference patterns like "dequantize - consumer - quantize".
    _reroute_tuple_getitem_pattern(graph)
    return lstm_output_dq

def _maybe_get_custom_module_lstm_from_node_arg(
    arg: Node,
    named_modules: Dict[str, torch.nn.Module],
) -> Optional[Node]:
    """
    Given an argument of a node, if the argument refers to the path through which the node
    is a consumer of custom module LSTM, return the custom module LSTM node, or None otherwise.

    This is used to determine whether a node is a consumer of custom module LSTM, and, if so,
    skip inserting input observers for this node. This is because custom module LSTM produces
    quantized outputs, so inserting an input observer for the consumer of custom module LSTM
    would unnecessarily quantize the outputs again.

      lstm -> consumer

    In practice, however, custom module LSTM outputs a tuple (output, (hidden0, hidden1)) with
    DeQuantStubs attached to each internal node (see `_insert_dequant_stubs_for_custom_module_lstm_output`).
    This tuple can be consumed in one of four ways:

      lstm -> getitem -> DeQuantStub -> consumer                       # consume lstm[0]
      lstm -> getitem -> getitem -> DeQuantStub -> tuple -> consumer   # consume lstm[1]
      lstm -> getitem -> getitem -> DeQuantStub -> consumer            # consume lstm[1][0] or lstm[1][1]
      lstm -> getitem -> DeQuantStub -> tuple -> consumer              # consume lstm

    Thus, we must match against the above patterns instead of simply checking the parent node
    to determine whether this node is a consumer of a custom module LSTM.
    """
    def match_dq(a):
        return isinstance(_get_module(a, named_modules), DeQuantStub)

    def match_lstm(a):
        return _is_custom_module_lstm(a, named_modules)

    def match_getitem(a):
        return a.op == "call_function" and a.target == operator.getitem

    def match_tuple(a):
        return a.op == "call_function" and a.target == tuple

    def _match_pattern(match_pattern: List[Callable]) -> Optional[Node]:
        """
        Traverse up the graph and match the args one by one.
        If there is a match, return the last matched node, or None otherwise.
        """
        a = arg
        for i, match in enumerate(match_pattern):
            if not match(a):
                return None
            # Match next arg, for tuple the arg is a tuple of a list, e.g. ([dq_1, other_node],)
            if i < len(match_pattern) - 1:
                if match == match_tuple:
                    a = a.args[0][0]  # type: ignore[assignment,index]
                else:
                    a = a.args[0]  # type: ignore[assignment]
        return a

    all_match_patterns = [
        [match_dq, match_getitem, match_lstm],
        [match_tuple, match_dq, match_getitem, match_getitem, match_lstm],
        [match_dq, match_getitem, match_getitem, match_lstm],
        [match_tuple, match_dq, match_getitem, match_lstm],
    ]

    for p in all_match_patterns:
        matched_node = _match_pattern(p)
        if matched_node is not None:
            return matched_node
    return None

def _reroute_tuple_getitem_pattern(graph: Graph):
    """
    Search for patterns where N consecutive `tuple` call_function nodes are followed by
    N consecutive `getitem` call_function nodes that are "reverses" of the `tuple` nodes.
    If we find this pattern, reroute the consumers of the last `getitem` to skip these
    N `tuple` and `getitem` nodes.

    Before:

        a   b     c
        |   \\   /
        \\   tuple
         \\   /
          tuple
            |
        getitem(1)
            |
        getitem(0)
            |
            d

    After:

        b
        |
        d
    """
    def find_patterns(
            node: Node,
            index_stack: List[int],
            current_pattern: List[Node],
            matched_patterns: List[List[Node]],
            seen: Set[Tuple[Node, Tuple[int, ...]]]):
        """
        Traverse the graph recursively to match for the N-tuple - N-getitem patterns,
        starting at the given node.

        We use a stack to keep track of the expected `getitem` indices, since these are
        reversed from the `tuple` indices. In the above example, the stack after
        (b -> tuple -> tuple) will be [0, 1], which will be popped by getitem(1) first
        and then by getitem(0).

        TODO: traverse upwards from the output and handle the case when tuple is not a
        separate node, e.g. graph.call_function(operator.getitem, args=(a, (b, c)))
        """
        if len(index_stack) == 0 and len(current_pattern) > 0:
            matched_patterns.append(copy.copy(current_pattern))
            current_pattern.clear()

        # Avoid duplicating work
        state = (node, tuple(index_stack))
        if state in seen:
            return
        seen.add(state)

        # Iterate through users of this node to find tuple/getitem nodes to match
        for user in node.users:
            if user.op == "call_function" and user.target == tuple:
                for i, user_arg in enumerate(user.args[0]):  # type: ignore[arg-type]
                    if user_arg == node:
                        index_stack.append(i)
                        current_pattern.append(user)
                        find_patterns(user, index_stack, current_pattern, matched_patterns, seen)
            elif user.op == "call_function" and user.target == operator.getitem:
                if len(index_stack) > 0:
                    if user.args[1] == index_stack[-1]:
                        index_stack.pop()
                        current_pattern.append(user)
                        find_patterns(user, index_stack, current_pattern, matched_patterns, seen)
        return matched_patterns

    # Collect all matched patterns
    matched_patterns: List[List[Node]] = []
    seen: Set[Tuple[Node, Tuple[int, ...]]] = set()  # (node, index_stack)
    for node in graph.nodes:
        find_patterns(node, [], [], matched_patterns, seen)

    # For each pattern, redirect all consumers of the last getitem node to the correct input
    # of the first tuple node
    for pattern in matched_patterns:
        first_tuple = pattern[0]
        last_getitem = pattern[-1]
        assert first_tuple.op == "call_function" and first_tuple.target == tuple
        assert last_getitem.op == "call_function" and last_getitem.target == operator.getitem
        last_getitem_index = last_getitem.args[1]
        new_input = first_tuple.args[0][last_getitem_index]  # type: ignore[index]
        for user in list(last_getitem.users.keys()):
            user.replace_input_with(last_getitem, new_input)

def _get_observer_from_activation_post_process(
    activation_post_process: Union[ObserverBase, FakeQuantizeBase],
) -> ObserverBase:
    """
    If `activation_post_process` is an observer, return the observer.
    If `activation_post_process` is a fake quantize, return the internal observer.
    """
    if isinstance(activation_post_process, ObserverBase):
        return activation_post_process
    else:
        assert isinstance(activation_post_process, FakeQuantizeBase)
        return activation_post_process.activation_post_process  # type: ignore[return-value]

def _qconfig_satisfies_dtype_config_constraints(
        qconfig: QConfigAny,
        dtype_with_constraints: DTypeWithConstraints,
        is_activation: bool = True) -> bool:
    """
    Return whether `qconfig` satisfies the following constraints from the backend,
    specified through the activation and weight DTypeWithConstraints.

        1. QConfig specified a quantization range that falls within the backend's, if any
        2. QConfig specified a min scale value that is >= the backend's, if any

    If `is_activation` is True, we check `qconfig.activation`, else we check `qconfig.weight`.
    If `qconfig` or `dtype_with_constraints.dtype` is None, or the dtypes do not match, return True.
    """
    def _activation_post_process_satisfies_dtype_config_constraints(
            activation_post_process: Union[ObserverBase, FakeQuantizeBase],
            dtype_with_constraints: DTypeWithConstraints,
            debug_string: str) -> bool:
        observer = _get_observer_from_activation_post_process(activation_post_process)
        app_quant_min = getattr(observer, "quant_min", None)
        app_quant_max = getattr(observer, "quant_max", None)
        # TODO: for now, just use the existing eps value as scale_min. In the future, we should
        # resolve the differences between the two, either by renaming eps or some other way
        app_scale_min = getattr(observer, "eps", None)
        backend_quant_min = dtype_with_constraints.quant_min_lower_bound
        backend_quant_max = dtype_with_constraints.quant_max_upper_bound
        backend_scale_min = dtype_with_constraints.scale_min_lower_bound
        # check quantization ranges
        if backend_quant_min is not None and backend_quant_max is not None:
            if app_quant_min is None or app_quant_max is None:
                warnings.warn("QConfig %s must specify 'quant_min' and 'quant_max', ignoring %s" %
                              (debug_string, qconfig))
                return False
            elif app_quant_min < backend_quant_min or app_quant_max > backend_quant_max:
                warnings.warn(("QConfig %s quantization range must fall within the backend's:\n"
                              "QConfig range = (%s, %s), BackendConfig range = (%s, %s), ignoring %s") %
                              (debug_string, app_quant_min, app_quant_max,
                              backend_quant_min, backend_quant_max, qconfig))
                return False
        # check scale min
        if backend_scale_min is not None:
            if app_scale_min is None:
                warnings.warn("QConfig %s must specify 'eps', ignoring %s" % (debug_string, qconfig))
                return False
            elif app_scale_min < backend_scale_min:
                warnings.warn(("QConfig %s eps (%s) must be greater than or equal to "
                              "the backend's min scale value (%s), ignoring %s") %
                              (debug_string, app_scale_min, backend_scale_min, qconfig))
                return False
        return True

    if qconfig is None or dtype_with_constraints.dtype is None:
        return True

    activation_post_process_ctr = qconfig.activation if is_activation else qconfig.weight
    debug_string = "activation" if is_activation else "weight"
    satisfies_constraints = True
    if activation_post_process_ctr is not None:
        activation_post_process = activation_post_process_ctr()
        assert _is_activation_post_process(activation_post_process)
        # If dtypes don't match, don't check the activation_post_process and return True early
        if activation_post_process.dtype != dtype_with_constraints.dtype:
            return True
        satisfies_constraints = _activation_post_process_satisfies_dtype_config_constraints(
            activation_post_process, dtype_with_constraints, debug_string)
    return satisfies_constraints
