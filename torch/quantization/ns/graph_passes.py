import torch
from torch.fx import GraphModule, map_arg
from torch.fx.graph import Graph, Node
from torch.quantization.fx.quantize import is_activation_post_process
from torch.quantization.fx.utils import get_new_attr_name_with_prefix

from .utils import (
    get_node_io_type,
    getattr_from_fqn,
    print_node,
    NodeIOType,
    return_first_non_observer_node,
)

from typing import Dict, Tuple, Callable, List, Any, Optional

def _insert_logger_after_node(
    node: Node,
    gm: GraphModule,
    logger_cls: Callable,
    logger_node_name_suffix: str,
    model_name: str,
    other_node_name: Optional[str] = None,
) -> Node:
    """
    Given a starting graph of

    prev_node -> node -> next_node

    This function creates a new logger_cls obj and adds it
    after node, resulting in

    prev_node -> node -> logger_obj -> next_node
    """
    # create new name
    logger_node_name = \
        get_new_attr_name_with_prefix(node.name + logger_node_name_suffix)(gm)
    # create the logger object
    logger_obj = logger_cls(node.name, model_name, other_node_name)
    # attach the logger object to the parent module
    setattr(gm, logger_node_name, logger_obj)
    logger_node = node.graph.create_node(
        'call_module', logger_node_name, (node,), {})
    return logger_node

def remove_observers_add_loggers(
    gm: GraphModule,
    nodes_to_instrument: List[Node],
    logger_cls: Callable,
    model_name: str,
) -> GraphModule:
    """
    Takes the graph of gm, removes all observers, adds loggers to the output
    of each node in nodes_to_instrument. Returns a GraphModule with the new
    graph.
    """

    new_graph = Graph()
    env: Dict[str, Any] = {}
    modules = dict(gm.named_modules())

    def load_arg(a):
        return map_arg(a, lambda node: env[node.name])

    for node in gm.graph.nodes:
        if node.op == 'output':
            new_graph.output(map_arg(node.args[0], load_arg))
            continue

        if node.op == 'call_module' and is_activation_post_process(modules[node.target]):
            # remove activation post process node
            env[node.name] = env[node.args[0].name]

        elif node in nodes_to_instrument:
            # ensure env is populated with base node
            env[node.name] = new_graph.node_copy(node, load_arg)
            # add the logger after the base node
            env[node.name] = _insert_logger_after_node(
                env[node.name], gm, logger_cls, '_ns_logger_', model_name)

        else:
            env[node.name] = new_graph.node_copy(node, load_arg)

    new_gm = GraphModule(gm, new_graph)
    return new_gm

def _insert_dtype_cast_after_node(
    node_a: Node,
    node_c: Node,
    prev_node_c: Node,
    gm_a: GraphModule,
    gm_b: GraphModule,
    node_name_prefix: str,
) -> Node:
    """
    Given a starting graph C (derived from graph B) of

    ... -> prev_node_c -> node_c -> ...

    And a corresponding related node_a, inserts the correct dtype
    cast node after prev_node_c to cast into the dtype expected
    by node_a, resulting in:

                          dtype_cast
                        /
    ... -> prev_node_c -> node_c -> ...

    For example, if node_c is an int8 op and node_a is an fp32 op, this function
    will insert a dequant.
    """
    dtype_cast_op = None
    node_io_type_a = get_node_io_type(node_a, gm_a)
    node_io_type_c = get_node_io_type(node_c, gm_b)

    if node_io_type_a == NodeIOType.FP32 and node_io_type_c == NodeIOType.INT8:
        dtype_cast_op = torch.dequantize
    else:
        raise AssertionError(
            f"dtype cast from {node_io_type_c} to {node_io_type_a} needs to be implemented")

    new_dtype_cast_name = \
        get_new_attr_name_with_prefix(node_name_prefix)(gm_b)
    return prev_node_c.graph.create_node(
        'call_function', dtype_cast_op, (prev_node_c,), {},
        new_dtype_cast_name)

def _insert_copy_of_node_a_after_input_node_c(
    input_node_c: Node,
    node_a: Node,
    gm_a: GraphModule,
    gm_b: GraphModule,
    node_name_prefix: str,
) -> Node:
    """
    Assume that node_a from graph_a has
      args (input, arg1, ...), and
      kwargs {kw0: kwarg0, ...}

    Copies the underlying values of arg1..argn and kwarg0..kwargn into gm_b,
    and creates the corresponding nodes in graph_c. Note: observers are ignored,
    so if an arg is an observer we navigate up until we find a non-observer parent.

    If node_a is a call_module, points the module pointed to by node_a to gm_b.

    Creates the copy of node_a in graph_c, with input as the first arg,
    and all other args and kwargs pointing to the copies of the objects
    in gm_b created above.

    An example in pictures:

    graph A:
    ========

    input -------------> node_a
                         / /
    weight -> weight_obs  /
                         /
    bias ----------------

    graph C (derived from B):
    =========================

    input_node_c --> node_a_copy
                     / /
    weight_copy ----/ /
                     /
    bias_copy ------/
    """
    graph_c = input_node_c.graph

    # generically handle all args and kwargs except for the input
    # Note: this hasn't been tested with many ops, logic may change.
    new_args = []
    # assumes that the first arg is the input
    for node_a_arg in node_a.args[1:]:
        if isinstance(node_a_arg, Node):
            arg_a = return_first_non_observer_node(node_a_arg, gm_a)
            arg_a_copy_name = \
                get_new_attr_name_with_prefix(arg_a.name + '_shadow_copy_')(gm_b)  # type: ignore
            arg_a_obj = getattr_from_fqn(gm_a, arg_a.target)  # type: ignore
            setattr(gm_b, arg_a_copy_name, arg_a_obj.detach())
            node_a_arg_copy = graph_c.create_node(
                'get_attr', arg_a_copy_name, (), {}, arg_a_copy_name)
            new_args.append(node_a_arg_copy)
        else:
            raise AssertionError(
                f"handling for arg of type {type(node_a_arg)} is not implemented")

    new_kwargs = {}
    for node_a_k, node_a_kwarg in node_a.kwargs.items():
        kwarg_a_copy_name = \
            get_new_attr_name_with_prefix(node_a_kwarg.name + '_shadow_copy_')(gm_b)  # type: ignore
        kwarg_a_obj = getattr_from_fqn(gm_a, node_a_kwarg.target)  # type: ignore
        setattr(gm_b, kwarg_a_copy_name, kwarg_a_obj.detach())
        node_a_kwarg_copy = graph_c.create_node(
            'get_attr', kwarg_a_copy_name, (), {}, kwarg_a_copy_name)
        new_kwargs[node_a_k] = node_a_kwarg_copy

    node_a_shadows_c_name = \
        get_new_attr_name_with_prefix(node_name_prefix)(gm_b)

    if node_a.op == 'call_module':
        # if target is a module, we point to the module from gm_b
        new_mod_copy_name = \
            get_new_attr_name_with_prefix(node_name_prefix)(gm_b)
        # fetch the corresponding module from gm_a
        assert isinstance(node_a.target, str)
        mod_a = getattr_from_fqn(gm_a, node_a.target)
        setattr(gm_b, new_mod_copy_name, mod_a)
        node_a_shadows_c = graph_c.create_node(
            node_a.op, new_mod_copy_name, (input_node_c, *new_args),
            new_kwargs, node_a_shadows_c_name)  # type: ignore
        return node_a_shadows_c
    else:
        assert node_a.op == 'call_function'
        node_a_shadows_c = graph_c.create_node(
            node_a.op, node_a.target, (input_node_c, *new_args),
            new_kwargs, node_a_shadows_c_name)  # type: ignore
        return node_a_shadows_c

def create_a_shadows_b(
    name_a: str,
    gm_a: GraphModule,
    name_b: str,
    gm_b: GraphModule,
    matched_node_pairs: Dict[str, Tuple[Node, Node]],
    logger_cls: Callable,
) -> GraphModule:
    """
    Creates a new GraphModule consisting of the graph of C, with the meaningful
    nodes of A shadowing the corresponding nodes of B.  For example,

    Graph A:
    a0 -> op0_fp32 -> a1 -> op1_fp32 -> a2

    Graph B:
    b0 -> op0_int8 -> b1 -> op1_int8 -> b2

    matched_node_pairs: {'op0': (op0_fp32, op0_int8), 'op1': (op1_fp32, op1_int8)}

    Graph C (A shadows B):

        / dequant0 -> op0_fp32 -> logger_a_0  / dequant_1 -> op1_fp32 -> logger_a_1
       /                                     /
    b0 -------------> op0_int8 -> logger_b_0 --------------> op1_int8 -> logger_b_1

    In a nutshell, this function does the following for each node pair:
    * copies the necessary attributes and modules from gm_a to gm_b,
      keeping names unique
    * adds a dtype cast op (dequant, quant, etc)
    * adds a copy of node_a in gm_b's graph
    * adds loggers to the outputs of node_a and node_b
    """

    # graph_c is the graph created from copying the nodes of graph_b and inserting
    # the shadows with the nodes copied from graph_a
    graph_c = Graph()
    env_c: Dict[str, Any] = {}
    modules = dict(gm_b.named_modules())

    def load_arg(a):
        return map_arg(a, lambda node: env_c[node.name])

    nodes_to_instrument_b_to_a = {}
    for match_name, (node_a, node_b) in matched_node_pairs.items():
        nodes_to_instrument_b_to_a[node_b] = node_a

    for node_b in gm_b.graph.nodes:
        if node_b.op == 'output':
            graph_c.output(map_arg(node_b.args[0], load_arg))
            continue

        if node_b.op == 'call_module' and is_activation_post_process(modules[node_b.target]):
            # remove activation post process node
            env_c[node_b.name] = env_c[node_b.args[0].name]  # type: ignore

        elif node_b in nodes_to_instrument_b_to_a:
            node_a = nodes_to_instrument_b_to_a[node_b]
            if False:
                print('b')
                print_node(node_b)
                print('a')
                print_node(node_a)

            # ensure env_c is populated with base node
            env_c[node_b.name] = graph_c.node_copy(node_b, load_arg)
            node_c = env_c[node_b.name]

            # after this point,
            #
            # node_a is the original node from graph_a, with parent module gm_a
            # node_b is the original node from graph_b, with parent module gm_b
            # node_c is the copy of node_b in graph_c
            #
            # subgraph so far:
            #
            # node_c

            # cast dtype from the dtype of node_c's input to the dtype of
            # node_a's input (dequant, etc)
            dtype_cast_node = _insert_dtype_cast_after_node(
                node_a, node_c, node_c.args[0], gm_a, gm_b, node_b.name + '_dtype_cast_')
            env_c[dtype_cast_node.name] = dtype_cast_node
            # subgraph so far:
            #
            #       dtype_cast_node
            #      /
            # node_c

            # hook up the new mod_a copy to be in the graph, receiving the
            # same inputs as mod_b does, with dtype cast to match a
            node_a_shadows_c = _insert_copy_of_node_a_after_input_node_c(
                env_c[dtype_cast_node.name],
                node_a, gm_a, gm_b, node_c.name + '_shadow_copy_')
            env_c[node_a_shadows_c.name] = node_a_shadows_c
            # subgraph so far:
            #
            #       dtype_cast_node --> node_a_copy(args/kwargs not shown)
            #      /
            # node_c

            # hook up a logger to the mod_b copy
            env_c[node_b.name] = _insert_logger_after_node(
                env_c[node_b.name], gm_b, logger_cls, '_ns_logger_b_', name_b)
            # subgraph so far:
            #
            #       dtype_cast_node --> node_a_copy
            #      /
            # node_c --> logger_c

            # hook up a logger to the mod_a copy
            # Note: we pass node_b.name to this logger, for easy matching later
            env_c[node_a_shadows_c.name] = _insert_logger_after_node(
                env_c[node_a_shadows_c.name], gm_b, logger_cls, '_ns_logger_a_', name_a,
                node_b.name)
            # subgraph so far:
            #
            #       dtype_cast_node --> node_a_copy --> logger_a
            #      /
            # node_c --> logger_c

        else:
            env_c[node_b.name] = graph_c.node_copy(node_b, load_arg)

    gm_c = GraphModule(gm_b, graph_c)
    return gm_c
