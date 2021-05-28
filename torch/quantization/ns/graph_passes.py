import torch
from torch.fx import GraphModule, map_arg
from torch.fx.graph import Graph, Node
from torch.quantization.fx.quantize import is_activation_post_process
from torch.quantization.fx.utils import get_new_attr_name_with_prefix

from .utils import (
    get_node_first_input_and_output_type,
    getattr_from_fqn,
    NodeInputOrOutputType,
    return_first_non_observer_node,
    get_number_of_non_param_args,
    get_target_type_str,
    get_arg_indices_of_inputs_to_log,
)

from .ns_types import (
    NSSingleResultValuesType,
    NSSubgraph,
    NSNodeTargetType,
)
from torch.quantization.ns.mappings import (
    get_node_type_to_io_type_map,
)

from typing import Dict, Tuple, Callable, List, Any, Union, Optional, Set

def _insert_logger_after_node(
    node: Node,
    gm: GraphModule,
    logger_cls: Callable,
    logger_node_name_suffix: str,
    ref_node_name: str,
    model_name: str,
    ref_name: str,
    results_type: str,
    index_within_arg: int,
    index_of_arg: int,
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
    target_type = get_target_type_str(node, gm)
    # create the logger object
    logger_obj = logger_cls(
        ref_node_name, node.name, model_name, ref_name, target_type,
        results_type, index_within_arg, index_of_arg)
    # attach the logger object to the parent module
    setattr(gm, logger_node_name, logger_obj)
    logger_node = node.graph.create_node(
        'call_module', logger_node_name, (node,), {})
    return logger_node

def remove_observers_add_loggers(
    gm: GraphModule,
    node_to_instrument_inputs_to_ref_node_name: Dict[Node, str],
    node_to_instrument_outputs_to_ref_node_name: Dict[Node, str],
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

        elif (
            (node in node_to_instrument_inputs_to_ref_node_name) or
            (node in node_to_instrument_outputs_to_ref_node_name)
        ):

            if node in node_to_instrument_inputs_to_ref_node_name:
                ref_name = node_to_instrument_inputs_to_ref_node_name[node]
                # Ops such add and mul are special because either
                # one or two of the first two arguments can be tensors,
                # and if one argument is a tensor it can be first or
                # second (x + 1 versus 1 + x).
                arg_indices_to_log = get_arg_indices_of_inputs_to_log(node)
                for node_arg_idx in arg_indices_to_log:
                    node_arg = node.args[node_arg_idx]
                    if type(node_arg) == Node:
                        # create a single input logger
                        prev_node = env[node_arg.name]
                        env[node_arg.name] = _insert_logger_after_node(
                            prev_node, gm, logger_cls, '_ns_logger_', node.name,
                            model_name, ref_name,
                            NSSingleResultValuesType.NODE_INPUT.value,
                            index_within_arg=0, index_of_arg=node_arg_idx)
                    elif type(node_arg) == torch.fx.immutable_collections.immutable_list:
                        # create N input loggers, one for each node
                        for arg_idx, arg in enumerate(node_arg):
                            prev_node = env[arg.name]
                            env[prev_node.name] = _insert_logger_after_node(
                                prev_node, gm, logger_cls, '_ns_logger_', node.name,
                                model_name, ref_name,
                                NSSingleResultValuesType.NODE_INPUT.value,
                                index_within_arg=arg_idx, index_of_arg=node_arg_idx)
                    else:
                        pass

            # ensure env is populated with base node
            # Note: runs for both inputs and outputs
            env[node.name] = new_graph.node_copy(node, load_arg)

            if node in node_to_instrument_outputs_to_ref_node_name:
                ref_name = node_to_instrument_outputs_to_ref_node_name[node]
                # add the logger after the base node
                env[node.name] = _insert_logger_after_node(
                    env[node.name], gm, logger_cls, '_ns_logger_', node.name,
                    model_name, ref_name, NSSingleResultValuesType.NODE_OUTPUT.value,
                    index_within_arg=0, index_of_arg=0)

        else:
            env[node.name] = new_graph.node_copy(node, load_arg)

    new_gm = GraphModule(gm, new_graph)
    return new_gm

def _insert_dtype_cast_after_node(
    node_a: Node,
    node_c: Node,
    prev_node_c: Union[Node, List[Node]],
    gm_a: GraphModule,
    gm_b: GraphModule,
    graph_c: Graph,
    node_name_prefix: str,
    logger_cls: Callable,
    node_type_to_io_type_map: Dict[str, Set[NSNodeTargetType]],
) -> Union[Node, List[Node]]:
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
    dtype_cast_mod_cls = None
    node_input_type_a, _node_output_type_a = \
        get_node_first_input_and_output_type(
            node_a, gm_a, logger_cls, node_type_to_io_type_map)
    node_input_type_c, _node_output_type_c = \
        get_node_first_input_and_output_type(
            node_c, gm_b, logger_cls, node_type_to_io_type_map)

    if (
        (node_input_type_a == NodeInputOrOutputType.FP32 and
         node_input_type_c == NodeInputOrOutputType.INT8) or
        (node_input_type_a == NodeInputOrOutputType.FP32 and
         node_input_type_c == NodeInputOrOutputType.FP16) or
        # TODO(future PR): determine the actual dtype of node_c,
        # the current code only works because dequantize works with
        # multiple input dtypes.
        (node_input_type_a == NodeInputOrOutputType.FP32 and
         node_input_type_c == NodeInputOrOutputType.FP32_OR_INT8)
    ):
        dtype_cast_op = torch.dequantize
    elif (
        node_input_type_a == node_input_type_c and
        node_input_type_a != NodeInputOrOutputType.UNKNOWN
    ):
        dtype_cast_mod_cls = torch.nn.Identity
    else:
        raise AssertionError(
            f"dtype cast from {node_input_type_c} {node_c.format_node()} to " +
            f"{node_input_type_a} {node_a.format_node()} needs to be implemented")

    if isinstance(prev_node_c, Node):
        new_dtype_cast_name = \
            get_new_attr_name_with_prefix(node_name_prefix)(gm_b)
        if dtype_cast_op:
            return graph_c.create_node(
                'call_function', dtype_cast_op, (prev_node_c,), {},
                new_dtype_cast_name)
        else:
            assert dtype_cast_mod_cls
            dtype_cast_mod = dtype_cast_mod_cls()
            setattr(gm_b, new_dtype_cast_name, dtype_cast_mod)
            return graph_c.create_node(
                'call_module', new_dtype_cast_name, (prev_node_c,), {},
                new_dtype_cast_name)
    elif isinstance(prev_node_c, list):
        results = []
        for prev_node_c_inner in prev_node_c:
            new_dtype_cast_name = \
                get_new_attr_name_with_prefix(node_name_prefix)(gm_b)
            if dtype_cast_op:
                new_dtype_cast_node = graph_c.create_node(
                    'call_function', dtype_cast_op, (prev_node_c_inner,), {},
                    new_dtype_cast_name)
                results.append(new_dtype_cast_node)
            else:
                assert dtype_cast_mod_cls
                dtype_cast_mod = dtype_cast_mod_cls()
                setattr(gm_b, new_dtype_cast_name, dtype_cast_mod)
                new_dtype_cast_node = graph_c.create_node(
                    'call_module', new_dtype_cast_name, (prev_node_c_inner,), {},
                    new_dtype_cast_name)
                results.append(new_dtype_cast_node)
        return results
    else:
        raise AssertionError(f"type f{type(prev_node_c)} is not handled")

# TODO(future PR): look into using copy_node API instead
def _copy_node_from_a_to_c(
    node_a: Node,
    gm_a: GraphModule,
    gm_b: GraphModule,
    graph_c: Graph,
) -> Node:
    """
    Simple copy of node_a to graph_c.
    """
    if node_a.op == 'get_attr':
        node_a_copy_name = \
            get_new_attr_name_with_prefix(node_a.name + '_shadow_copy_')(gm_b)
        node_a_obj = getattr_from_fqn(gm_a, node_a.target)  # type: ignore[arg-type]
        if torch.is_tensor(node_a_obj):
            node_a_obj = node_a_obj.detach()
        setattr(gm_b, node_a_copy_name, node_a_obj)
        node_a_copy = graph_c.create_node(
            node_a.op, node_a_copy_name, (), {}, node_a_copy_name)
        return node_a_copy
    elif node_a.op == 'call_method':
        assert node_a.target in ('dequantize', 'to'), \
            f"target {node_a.target} is not implemented"
        if node_a.target == 'dequantize':
            arg_copy = _copy_node_from_a_to_c(node_a.args[0], gm_a, gm_b, graph_c)  # type: ignore[arg-type]
            node_a_copy_name = \
                get_new_attr_name_with_prefix(node_a.name + '_shadow_copy_')(gm_b)
            node_a_copy = graph_c.create_node(
                node_a.op, node_a.target, (arg_copy,), {}, node_a_copy_name)
            return node_a_copy
        else:  # to
            arg_copy = _copy_node_from_a_to_c(node_a.args[0], gm_a, gm_b, graph_c)  # type: ignore[arg-type]
            node_a_copy_name = \
                get_new_attr_name_with_prefix(node_a.name + '_shadow_copy_')(gm_b)
            node_a_copy = graph_c.create_node(
                node_a.op, node_a.target, (arg_copy, node_a.args[1]), {},
                node_a_copy_name)
            return node_a_copy

    else:
        raise AssertionError(
            f"handling of node with op {node_a.op} is not implemented")

def _insert_copy_of_subgraph_a_after_input_node_c(
    input_node_c: Union[Node, List[Node]],
    input_node_c_2: Optional[Union[Node, List[Node]]],
    subgraph_a: NSSubgraph,
    gm_a: GraphModule,
    gm_b: GraphModule,
    node_name_prefix: str,
) -> Node:
    """
    TODO(before land): real docblock
    """
    if isinstance(input_node_c, Node):
        graph_c = input_node_c.graph
    else:
        graph_c = input_node_c[0].graph

    # create a sequential list of the subgraphs' nodes from start to end,
    # because we need to add the nodes to graph C in non-reverse order
    nodes_of_a = [subgraph_a.end_node]
    cur_node = subgraph_a.end_node
    while cur_node != subgraph_a.start_node:
        cur_node = cur_node.args[0]  # type: ignore[assignment]
        nodes_of_a.insert(0, cur_node)

    # go through nodes of a in order, and insert them into the graph of c
    # sequentially
    cur_node_a = nodes_of_a[0]
    cur_node_c = _insert_copy_of_node_a_after_input_node_c(
        input_node_c,
        input_node_c_2,
        cur_node_a,
        gm_a,
        gm_b,
        node_name_prefix)
    for cur_idx_a in range(1, len(nodes_of_a)):
        cur_node_a = nodes_of_a[cur_idx_a]
        prev_node_c = cur_node_c  # previous added node is the input to next node
        cur_node_c = _insert_copy_of_node_a_after_input_node_c(
            prev_node_c,
            # TODO(future PR): enable multiple inputs for nodes which are not at start of subgraph
            None,
            cur_node_a,
            gm_a,
            gm_b,
            node_name_prefix)
    # return the last inserted node
    return cur_node_c


def _insert_copy_of_node_a_after_input_node_c(
    input_node_c: Union[Node, List[Node]],
    input_node_c_2: Optional[Union[Node, List[Node]]],
    node_a: Node,
    gm_a: GraphModule,
    gm_b: GraphModule,
    node_name_prefix: str,
) -> Node:
    """
    Assume that node_a from graph_a has
      args (input, (input2)?, arg1, ...), and
      kwargs {kw0: kwarg0, ...}

    Note: input2 is optional. If it equals to None, we assume that the op
    has a single non-param input.  If it is specified, we assume that the op
    has two non-param inputs.

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
                         / / /
    (input_2)?----------/ / /
                         / /
    weight -> weight_obs  /
                         /
    bias ----------------

    graph C (derived from B):
    =========================

    input_node_c --> node_a_copy
                     / / /
    (input_node_c_2)? / /
                     / /
    weight_copy ----/ /
                     /
    bias_copy ------/
    """
    if isinstance(input_node_c, Node):
        graph_c = input_node_c.graph
    else:
        graph_c = input_node_c[0].graph

    # generically handle all args and kwargs except for the input
    # Note: this hasn't been tested with many ops, logic may change.
    new_args = []
    # assumes that the first arg is the input
    num_non_param_args = 1 if input_node_c_2 is None else 2
    for node_a_arg in node_a.args[num_non_param_args:]:
        if isinstance(node_a_arg, Node):
            arg_a = return_first_non_observer_node(node_a_arg, gm_a)
            node_a_arg_copy = _copy_node_from_a_to_c(arg_a, gm_a, gm_b, graph_c)
            new_args.append(node_a_arg_copy)
        else:
            raise AssertionError(
                f"handling for arg of type {type(node_a_arg)} is not implemented")

    new_kwargs: Dict[str, Any] = {}
    for node_a_k, node_a_kwarg in node_a.kwargs.items():
        if isinstance(node_a_kwarg, Node):
            kwarg_a = return_first_non_observer_node(node_a_kwarg, gm_a)
            node_a_kwarg_copy = _copy_node_from_a_to_c(kwarg_a, gm_a, gm_b, graph_c)
            new_kwargs[node_a_k] = node_a_kwarg_copy
        else:
            new_kwargs[node_a_k] = node_a_kwarg

    node_a_shadows_c_name = \
        get_new_attr_name_with_prefix(node_name_prefix)(gm_b)

    if input_node_c_2:
        input_node_c_args = [input_node_c, input_node_c_2]
    else:
        input_node_c_args = [input_node_c]

    if node_a.op == 'call_module':
        # if target is a module, we point to the module from gm_b
        new_mod_copy_name = \
            get_new_attr_name_with_prefix(node_name_prefix)(gm_b)
        # fetch the corresponding module from gm_a
        assert isinstance(node_a.target, str)
        mod_a = getattr_from_fqn(gm_a, node_a.target)
        setattr(gm_b, new_mod_copy_name, mod_a)
        node_a_shadows_c = graph_c.create_node(
            node_a.op, new_mod_copy_name, (*input_node_c_args, *new_args),
            new_kwargs, node_a_shadows_c_name)
        return node_a_shadows_c
    else:
        assert node_a.op in ('call_function', 'call_method')
        node_a_shadows_c = graph_c.create_node(
            node_a.op, node_a.target, (*input_node_c_args, *new_args),
            new_kwargs, node_a_shadows_c_name)
        return node_a_shadows_c

def create_a_shadows_b(
    name_a: str,
    gm_a: GraphModule,
    name_b: str,
    gm_b: GraphModule,
    matched_subgraph_pairs: Dict[str, Tuple[NSSubgraph, NSSubgraph]],
    logger_cls: Callable,
    should_log_inputs: bool,
    node_type_to_io_type_map: Optional[Dict[str, Set[NSNodeTargetType]]] = None,
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

    if node_type_to_io_type_map is None:
        node_type_to_io_type_map = get_node_type_to_io_type_map()

    # graph_c is the graph created from copying the nodes of graph_b and inserting
    # the shadows with the nodes copied from graph_a
    graph_c = Graph()
    env_c: Dict[str, Any] = {}
    modules = dict(gm_b.named_modules())

    def load_arg(a):
        return map_arg(a, lambda node: env_c[node.name])

    start_node_b_to_matched_subgraph_a_and_name = {}
    end_node_b_to_matched_subgraph_a_and_name = {}
    for match_name, match in matched_subgraph_pairs.items():
        subgraph_a, subgraph_b = match
        start_node_b_to_matched_subgraph_a_and_name[subgraph_b.start_node] = \
            (subgraph_a, match_name)
        end_node_b_to_matched_subgraph_a_and_name[subgraph_b.end_node] = \
            (subgraph_a, match_name)

    for node_b in gm_b.graph.nodes:
        if node_b.op == 'output':
            graph_c.output(map_arg(node_b.args[0], load_arg))
            continue

        # calculate the flags to determine what to do with this node
        node_b_is_observer = \
            node_b.op == 'call_module' and is_activation_post_process(modules[node_b.target])
        node_b_is_start_node = node_b in start_node_b_to_matched_subgraph_a_and_name
        node_b_is_end_node = node_b in end_node_b_to_matched_subgraph_a_and_name

        if node_b_is_observer:
            # remove activation post process node
            env_c[node_b.name] = env_c[node_b.args[0].name]

        elif (node_b_is_start_node or node_b_is_end_node):

            if node_b_is_start_node:
                subgraph_a, ref_name = \
                    start_node_b_to_matched_subgraph_a_and_name[node_b]
            else:
                assert node_b_is_end_node
                subgraph_a, ref_name = \
                    end_node_b_to_matched_subgraph_a_and_name[node_b]

            # For both start_node and end_node verify that we know how to do
            # the dtype cast. If we do not, skip.
            node_input_type_a, node_output_type_a = \
                get_node_first_input_and_output_type(
                    subgraph_a.start_node, gm_a, logger_cls,
                    node_type_to_io_type_map)
            node_input_type_b, node_output_type_b = \
                get_node_first_input_and_output_type(
                    node_b, gm_b, logger_cls,
                    node_type_to_io_type_map)
            node_io_types_known_a_and_b = (
                node_input_type_a != NodeInputOrOutputType.UNKNOWN and
                node_output_type_a != NodeInputOrOutputType.UNKNOWN and
                node_input_type_b != NodeInputOrOutputType.UNKNOWN and
                node_output_type_b != NodeInputOrOutputType.UNKNOWN
            )
            if not node_io_types_known_a_and_b:
                print(
                    f'skipping shadow loggers for node_b: {get_target_type_str(node_b, gm_b)}' +
                    f', start_node_a: {get_target_type_str(subgraph_a.start_node, gm_a)}' +
                    ', unknown dtype cast')
                env_c[node_b.name] = graph_c.node_copy(node_b, load_arg)
                continue

            if node_b_is_start_node:

                # if necessary, log the input of node_c
                if should_log_inputs:
                    if isinstance(node_b.args[0], Node):
                        prev_node_c = env_c[node_b.args[0].name]
                        env_c[prev_node_c.name] = _insert_logger_after_node(
                            prev_node_c, gm_b, logger_cls, '_ns_logger_b_inp_',
                            node_b.name, name_b, ref_name,
                            NSSingleResultValuesType.NODE_INPUT.value,
                            index_within_arg=0, index_of_arg=0)
                    elif isinstance(node_b.args[0], list):
                        # first, save the prev_node instances, because they
                        # will be overwritten in the env after the first logger
                        # is added
                        prev_node_c_list = [env_c[arg.name] for arg in node_b.args[0]]

                        for arg_idx, arg in enumerate(node_b.args[0]):
                            prev_node_c = prev_node_c_list[arg_idx]
                            env_c[prev_node_c.name] = _insert_logger_after_node(
                                prev_node_c, gm_b, logger_cls, '_ns_logger_b_inp_',
                                node_b.name, name_b, ref_name,
                                NSSingleResultValuesType.NODE_INPUT.value,
                                index_within_arg=arg_idx, index_of_arg=0)
                    else:
                        # logging of inputs which are not lists is not supported yet
                        raise AssertionError(f"type {type(node_b.args[0])} is not handled yet")
                # subgraph so far:
                #
                # (prev_node_c)+ -> (logger_c_input)?

            # Note: this if statement is always True, spelling it out to clarify code
            # intent.
            if node_b_is_start_node or node_b_is_end_node:
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
                # (prev_node_c)+ -> (logger_c_input)? -> node_start_c

            if node_b_is_start_node:

                # cast dtype from the dtype of node_c's input to the dtype of
                # node_a's input (dequant, etc)
                prev_node_c = node_c.args[0]
                if should_log_inputs:
                    # skip the input logger when inserting a dtype cast
                    if isinstance(prev_node_c, Node):
                        prev_node_c = prev_node_c.args[0]
                    elif isinstance(prev_node_c, list):
                        prev_node_c = [arg.args[0] for arg in prev_node_c]
                dtype_cast_node = _insert_dtype_cast_after_node(
                    subgraph_a.start_node, node_c, prev_node_c, gm_a, gm_b, graph_c,
                    node_b.name + '_dtype_cast_', logger_cls,
                    node_type_to_io_type_map)
                # note: not inserting to env_c because all nodes which use the dtype
                #   casts are copied from graph_a
                #
                # subgraph so far:
                #
                #           (dtype_cast_node)+
                #                  /
                # (prev_node_c)+ -> (logger_c_input)? -> node_start_c

                # if input logging is enabled, log the input to the subgraph
                if should_log_inputs:
                    # TODO: explain this
                    ref_node_name = ''
                    if isinstance(dtype_cast_node, Node):
                        dtype_cast_node = _insert_logger_after_node(
                            dtype_cast_node, gm_b, logger_cls, '_ns_logger_a_inp_',
                            ref_node_name, name_a, ref_name,
                            NSSingleResultValuesType.NODE_INPUT.value,
                            index_within_arg=0, index_of_arg=0)
                        input_logger: Union[Node, List[Node]] = dtype_cast_node
                    else:
                        assert isinstance(dtype_cast_node, list)
                        new_loggers = []
                        for dtype_cast_idx, dtype_cast_node_inner in enumerate(dtype_cast_node):
                            dtype_cast_logger = _insert_logger_after_node(
                                dtype_cast_node_inner, gm_b, logger_cls, '_ns_logger_a_inp_',
                                ref_node_name, name_a, ref_name,
                                NSSingleResultValuesType.NODE_INPUT.value,
                                index_within_arg=dtype_cast_idx,
                                index_of_arg=0)
                            new_loggers.append(dtype_cast_logger)
                        dtype_cast_node = new_loggers
                        input_logger = dtype_cast_node
                    # subgraph so far:
                    #
                    #       (dtype_cast_node)+ -> (logger_a_input)?
                    #                  /
                    # prev_node_c -> (logger_c_input)? -> node_start_c

                # hook up the new mod_a copy to be in the graph, receiving the
                # same inputs as mod_b does, with dtype cast to match a
                # Some ops, such as LSTMs, have two non-param inputs. If we have
                # such an op, pass the second param as well. Note: dtype casting
                # for the second param is not implemented yet, it can be added
                # later if there is a use case.
                node_c_second_non_param_arg = None
                num_non_param_args_node_a = get_number_of_non_param_args(subgraph_a.start_node, gm_a)
                if num_non_param_args_node_a == 2:
                    node_c_second_non_param_arg = node_c.args[1]
                node_a_shadows_c = _insert_copy_of_subgraph_a_after_input_node_c(
                    dtype_cast_node, node_c_second_non_param_arg,
                    subgraph_a, gm_a, gm_b, node_c.name + '_shadow_copy_')
                env_c[node_a_shadows_c.name] = node_a_shadows_c
                # subgraph so far:
                #
                #       dtype_cast_node -> (logger_a_input)? -> subgraph_a_copy(args/kwargs not shown)
                #                  /
                # (prev_node_c)+ -> (logger_c_input)? -> node_start_c

                if should_log_inputs:
                    # When we created the input logger, we left the ref_node_name
                    # as an empty string, because the subgraph copy did not exist
                    # yet. Now that the subgraph copy exists, we modify this name
                    # to its true value.
                    # Note: the alternative to this is to create the input logger
                    # after creating the subgraph, which is slightly more
                    # complicated. This is the lesser of two evils.
                    # input_logger = env_c[dtype_cast_node.name]
                    # Find the first node in the subgraph
                    cur_node = node_a_shadows_c
                    while cur_node.args[0] != input_logger:
                        cur_node = cur_node.args[0]  # type: ignore[assignment]
                    if isinstance(input_logger, Node):
                        input_logger_mod = getattr(gm_b, input_logger.name)
                        input_logger_mod.ref_node_name = cur_node.name
                    else:
                        assert isinstance(input_logger, list)
                        for input_logger_inner in input_logger:
                            input_logger_mod = getattr(gm_b, input_logger_inner.name)
                            input_logger_mod.ref_node_name = cur_node.name

                # hook up a logger to the mod_a copy
                env_c[node_a_shadows_c.name] = _insert_logger_after_node(
                    env_c[node_a_shadows_c.name], gm_b, logger_cls, '_ns_logger_a_',
                    node_a_shadows_c.name, name_a, ref_name,
                    NSSingleResultValuesType.NODE_OUTPUT.value,
                    index_within_arg=0, index_of_arg=0)
                # subgraph so far:
                #
                #       dtype_cast_node -> (logger_a_input)? -> subgraph_a_copy -> logger_a
                #                  /
                # (prev_node_c)+ -> (logger_c_input)? -> node_start_c

            if node_b_is_end_node:

                # hook up a logger to the mod_b copy
                env_c[node_b.name] = _insert_logger_after_node(
                    env_c[node_b.name], gm_b, logger_cls, '_ns_logger_b_',
                    node_b.name, name_b, ref_name,
                    NSSingleResultValuesType.NODE_OUTPUT.value,
                    index_within_arg=0, index_of_arg=0)
                # subgraph so far:
                #
                #       dtype_cast_node -> (logger_a_input)? -> subgraph_a_copy -> logger_a
                #                  /
                # (prev_node_c+) -> (logger_c_input)? -> node_start_c -> ... -> node_end_c -> logger_c
                #
                # Note: node_start_c may be the same node as node_end_c, or they
                # may have nodes inbetween.

        else:
            env_c[node_b.name] = graph_c.node_copy(node_b, load_arg)

    gm_c = GraphModule(gm_b, graph_c)
    return gm_c
