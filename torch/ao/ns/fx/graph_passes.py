# mypy: allow-untyped-defs
from collections.abc import Callable
from typing import Any

import torch
from torch.ao.ns.fx.mappings import get_node_type_to_io_type_map
from torch.ao.quantization.fx.utils import get_new_attr_name_with_prefix
from torch.ao.quantization.observer import _is_activation_post_process
from torch.fx import GraphModule, map_arg
from torch.fx.graph import Graph, Node
from .ns_types import NSNodeTargetType, NSSingleResultValuesType, NSSubgraph
from .utils import (
    get_arg_indices_of_inputs_to_log,
    get_node_first_input_and_output_type,
    get_node_input_qparams,
    get_normalized_nth_input,
    get_number_of_non_param_args,
    get_target_type_str,
    getattr_from_fqn,
    NodeInputOrOutputType,
    op_type_supports_shadowing,
    return_first_non_observer_node,
)


def _maybe_get_fqn(node: Node, gm: GraphModule) -> str | None:
    fqn = None
    if hasattr(gm, "_node_name_to_scope"):
        # fqn on observers is not present, because they do not
        # exist when the fqns are created during tracing. If this is
        # an observer, get the fqn of the node being observed.
        node_to_use_for_fqn = node
        if node.op == "call_module":
            if not isinstance(node.target, str):
                raise AssertionError(f"Expected str, got {type(node.target)}")
            module = getattr_from_fqn(gm, node.target)
            if _is_activation_post_process(module):
                node_to_use_for_fqn = get_normalized_nth_input(node, gm, 0)
        fqn = gm._node_name_to_scope[node_to_use_for_fqn.name][0]  # type: ignore[index]
    return fqn  # type: ignore[return-value]


def _insert_logger_after_node(
    node: Node,
    gm: GraphModule,
    logger_cls: Callable,
    logger_node_name_suffix: str,
    ref_node_name: str,
    model_name: str,
    ref_name: str,
    ref_node_target_type: str,
    results_type: str,
    index_within_arg: int,
    index_of_arg: int,
    fqn: str | None,
) -> Node:
    """
    Given a starting graph of

    prev_node -> node -> next_node

    This function creates a new logger_cls obj and adds it
    after node, resulting in

    prev_node -> node -> logger_obj -> next_node
    """
    # create new name
    logger_node_name = get_new_attr_name_with_prefix(
        node.name + logger_node_name_suffix
    )(gm)
    target_type = get_target_type_str(node, gm)
    # create the logger object
    logger_obj = logger_cls(
        ref_node_name,
        node.name,
        model_name,
        ref_name,
        target_type,
        ref_node_target_type,
        results_type,
        index_within_arg,
        index_of_arg,
        fqn,
    )
    # attach the logger object to the parent module
    setattr(gm, logger_node_name, logger_obj)
    logger_node = node.graph.create_node("call_module", logger_node_name, (node,), {})
    return logger_node


def add_loggers_to_model(
    gm: GraphModule,
    node_to_instrument_inputs_to_ref_node_name: dict[Node, tuple[str, str]],
    node_to_instrument_outputs_to_ref_node_name: dict[Node, tuple[str, str]],
    logger_cls: Callable,
    model_name: str,
) -> GraphModule:
    """
    Takes the graph of gm, adds loggers to the output
    of each node in nodes_to_instrument. Returns a GraphModule with the new
    graph.
    """

    new_graph = Graph()
    env: dict[str, Any] = {}

    def load_arg(a):
        return map_arg(a, lambda node: env[node.name])

    for node in gm.graph.nodes:
        if node.op == "output":
            new_graph.output(map_arg(get_normalized_nth_input(node, gm, 0), load_arg))
            continue

        if (node in node_to_instrument_inputs_to_ref_node_name) or (
            node in node_to_instrument_outputs_to_ref_node_name
        ):
            fqn = _maybe_get_fqn(node, gm)

            if node in node_to_instrument_inputs_to_ref_node_name:
                ref_name, ref_node_type = node_to_instrument_inputs_to_ref_node_name[
                    node
                ]
                # Ops such add and mul are special because either
                # one or two of the first two arguments can be tensors,
                # and if one argument is a tensor it can be first or
                # second (x + 1 versus 1 + x).
                arg_indices_to_log = get_arg_indices_of_inputs_to_log(node)
                for node_arg_idx in arg_indices_to_log:
                    node_arg = get_normalized_nth_input(node, gm, node_arg_idx)
                    if type(node_arg) is Node:
                        # create a single input logger
                        prev_node = env[node_arg.name]
                        env[node_arg.name] = _insert_logger_after_node(
                            prev_node,
                            gm,
                            logger_cls,
                            "_ns_logger_",
                            node.name,
                            model_name,
                            ref_name,
                            ref_node_type,
                            NSSingleResultValuesType.NODE_INPUT.value,
                            index_within_arg=0,
                            index_of_arg=node_arg_idx,
                            fqn=fqn,
                        )
                    elif (
                        type(node_arg) is torch.fx.immutable_collections.immutable_list
                    ):
                        # create N input loggers, one for each node
                        for arg_idx, arg in enumerate(node_arg):  # type: ignore[var-annotated, arg-type]
                            prev_node = env[arg.name]
                            env[prev_node.name] = _insert_logger_after_node(
                                prev_node,
                                gm,
                                logger_cls,
                                "_ns_logger_",
                                node.name,
                                model_name,
                                ref_name,
                                ref_node_type,
                                NSSingleResultValuesType.NODE_INPUT.value,
                                index_within_arg=arg_idx,
                                index_of_arg=node_arg_idx,
                                fqn=fqn,
                            )

            # ensure env is populated with base node
            # Note: runs for both inputs and outputs
            env[node.name] = new_graph.node_copy(node, load_arg)

            if node in node_to_instrument_outputs_to_ref_node_name:
                ref_name, ref_node_type = node_to_instrument_outputs_to_ref_node_name[
                    node
                ]
                # add the logger after the base node
                env[node.name] = _insert_logger_after_node(
                    env[node.name],
                    gm,
                    logger_cls,
                    "_ns_logger_",
                    node.name,
                    model_name,
                    ref_name,
                    ref_node_type,
                    NSSingleResultValuesType.NODE_OUTPUT.value,
                    index_within_arg=0,
                    index_of_arg=0,
                    fqn=fqn,
                )

        else:
            env[node.name] = new_graph.node_copy(node, load_arg)

    new_gm = GraphModule(gm, new_graph)
    return new_gm


def _insert_quantize_per_tensor_node(
    prev_node_c: Node,
    node_a: Node,
    gm_b: GraphModule,
    graph_c: Graph,
    scale: torch.Tensor | float,
    zero_point: torch.Tensor | int,
    dtype_cast_name: str,
) -> Node:
    # copy scale
    scale_node_name = get_new_attr_name_with_prefix(node_a.name + "_input_scale_")(gm_b)
    setattr(gm_b, scale_node_name, scale)
    scale_node = graph_c.create_node(
        "get_attr", scale_node_name, (), {}, scale_node_name
    )
    # copy zero_point
    zero_point_node_name = get_new_attr_name_with_prefix(
        node_a.name + "_input_zero_point_"
    )(gm_b)
    setattr(gm_b, zero_point_node_name, zero_point)
    zero_point_node = graph_c.create_node(
        "get_attr", zero_point_node_name, (), {}, zero_point_node_name
    )
    # create the quantize_per_tensor call
    return graph_c.create_node(
        "call_function",
        torch.quantize_per_tensor,
        (prev_node_c, scale_node, zero_point_node, torch.quint8),
        {},
        dtype_cast_name,
    )


def _insert_dtype_cast_after_node(
    node_a: Node,
    node_c: Node,
    prev_node_c: Node | list[Node],
    gm_a: GraphModule,
    gm_b: GraphModule,
    graph_c: Graph,
    node_name_prefix: str,
    logger_cls: Callable,
    node_type_to_io_type_map: dict[str, set[NSNodeTargetType]],
) -> Node | list[Node]:
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
    dtype_cast_method = None
    dtype_cast_method_dtype = None
    dtype_cast_scale = None
    dtype_cast_zero_point = None
    node_input_type_a, _node_output_type_a = get_node_first_input_and_output_type(
        node_a, gm_a, logger_cls, node_type_to_io_type_map
    )
    node_input_type_c, _node_output_type_c = get_node_first_input_and_output_type(
        node_c, gm_b, logger_cls, node_type_to_io_type_map
    )

    if (
        (
            node_input_type_a == NodeInputOrOutputType.FP32
            and node_input_type_c == NodeInputOrOutputType.INT8
        )
        or (
            node_input_type_a == NodeInputOrOutputType.FP32
            and node_input_type_c == NodeInputOrOutputType.FP16
        )
        or
        # TODO(future PR): determine the actual dtype of node_c,
        # the current code only works because dequantize works with
        # multiple input dtypes.
        (
            node_input_type_a == NodeInputOrOutputType.FP32
            and node_input_type_c == NodeInputOrOutputType.FP32_OR_INT8
        )
    ):
        dtype_cast_op = torch.dequantize
    elif (
        node_input_type_a == node_input_type_c
        and node_input_type_a != NodeInputOrOutputType.UNKNOWN
    ):
        dtype_cast_mod_cls = torch.nn.Identity
    elif (
        node_input_type_a == NodeInputOrOutputType.INT8
        and node_input_type_c == NodeInputOrOutputType.FP32
    ):
        # int8 shadows fp32, the dtype cast needs to quantize to int8
        # with the right qparams.
        node_a_input_qparams = get_node_input_qparams(
            node_a, gm_a, node_type_to_io_type_map
        )
        if node_a_input_qparams is not None:
            dtype_cast_op = torch.quantize_per_tensor  # type: ignore[assignment]
            dtype_cast_scale, dtype_cast_zero_point = node_a_input_qparams
    elif (
        node_input_type_a == NodeInputOrOutputType.FP16
        and node_input_type_c == NodeInputOrOutputType.FP32
    ):
        dtype_cast_method = "to"
        dtype_cast_method_dtype = torch.float16
    else:
        raise AssertionError(
            f"dtype cast from {node_input_type_c} {node_c.format_node()} to "
            + f"{node_input_type_a} {node_a.format_node()} needs to be implemented"
        )

    if isinstance(prev_node_c, Node):
        new_dtype_cast_name = get_new_attr_name_with_prefix(node_name_prefix)(gm_b)
        if dtype_cast_op:
            if dtype_cast_scale is not None and dtype_cast_zero_point is not None:
                return _insert_quantize_per_tensor_node(
                    prev_node_c,
                    node_a,
                    gm_b,
                    graph_c,
                    dtype_cast_scale,
                    dtype_cast_zero_point,
                    new_dtype_cast_name,
                )
            else:
                return graph_c.create_node(
                    "call_function",
                    dtype_cast_op,
                    (prev_node_c,),
                    {},
                    new_dtype_cast_name,
                )
        elif dtype_cast_method:
            return graph_c.create_node(
                "call_method",
                dtype_cast_method,
                (prev_node_c, dtype_cast_method_dtype),
                {},
                new_dtype_cast_name,
            )
        else:
            if not dtype_cast_mod_cls:
                raise AssertionError("Expected dtype_cast_mod_cls to be not None")
            dtype_cast_mod = dtype_cast_mod_cls()
            setattr(gm_b, new_dtype_cast_name, dtype_cast_mod)
            return graph_c.create_node(
                "call_module",
                new_dtype_cast_name,
                (prev_node_c,),
                {},
                new_dtype_cast_name,
            )
    elif isinstance(prev_node_c, list):
        results = []
        for prev_node_c_inner in prev_node_c:
            new_dtype_cast_name = get_new_attr_name_with_prefix(node_name_prefix)(gm_b)
            if dtype_cast_op:
                # TODO(future PR): add handling for quantize_per_tensor
                new_dtype_cast_node = graph_c.create_node(
                    "call_function",
                    dtype_cast_op,
                    (prev_node_c_inner,),
                    {},
                    new_dtype_cast_name,
                )
                results.append(new_dtype_cast_node)
            else:
                if not dtype_cast_mod_cls:
                    raise AssertionError("Expected dtype_cast_mod_cls to be not None")
                dtype_cast_mod = dtype_cast_mod_cls()
                setattr(gm_b, new_dtype_cast_name, dtype_cast_mod)
                new_dtype_cast_node = graph_c.create_node(
                    "call_module",
                    new_dtype_cast_name,
                    (prev_node_c_inner,),
                    {},
                    new_dtype_cast_name,
                )
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
    if node_a.op == "get_attr":
        node_a_copy_name = get_new_attr_name_with_prefix(node_a.name + "_shadow_copy_")(
            gm_b
        )
        node_a_obj = getattr_from_fqn(gm_a, node_a.target)  # type: ignore[arg-type]
        if torch.is_tensor(node_a_obj):
            node_a_obj = node_a_obj.detach()
        setattr(gm_b, node_a_copy_name, node_a_obj)
        node_a_copy = graph_c.create_node(
            node_a.op, node_a_copy_name, (), {}, node_a_copy_name
        )
        return node_a_copy
    elif node_a.op == "call_method":
        if node_a.target not in ("dequantize", "to"):
            raise AssertionError(f"target {node_a.target} is not implemented")
        if node_a.target == "dequantize":
            arg_copy = _copy_node_from_a_to_c(
                get_normalized_nth_input(node_a, gm_a, 0), gm_a, gm_b, graph_c
            )  # type: ignore[arg-type]
            node_a_copy_name = get_new_attr_name_with_prefix(
                node_a.name + "_shadow_copy_"
            )(gm_b)
            node_a_copy = graph_c.create_node(
                node_a.op, node_a.target, (arg_copy,), {}, node_a_copy_name
            )
            return node_a_copy
        else:  # to
            arg_copy = _copy_node_from_a_to_c(
                get_normalized_nth_input(node_a, gm_a, 0), gm_a, gm_b, graph_c
            )  # type: ignore[arg-type]
            node_a_copy_name = get_new_attr_name_with_prefix(
                node_a.name + "_shadow_copy_"
            )(gm_b)
            node_a_copy = graph_c.create_node(
                node_a.op,
                node_a.target,
                (arg_copy, get_normalized_nth_input(node_a, gm_a, 1)),
                {},
                node_a_copy_name,
            )
            return node_a_copy

    else:
        raise AssertionError(
            f"handling of node {node_a.format_node()} with op {node_a.op} is not implemented"
        )


def _can_insert_copy_of_subgraph_a(
    subgraph_a: NSSubgraph,
    gm_a: GraphModule,
    num_non_param_args_node_a: int,
) -> bool:
    """
    This function returns `False` if the input subgraph cannot be copied by
    `_insert_copy_of_subgraph_a_after_input_node_c`. This usually means
    that there is a corner case logic for which copy is not yet implemented.
    """
    # populate the list of nodes we need to check
    nodes = []
    cur_node = subgraph_a.end_node
    while cur_node != subgraph_a.start_node:
        nodes.append(cur_node)
        cur_node = get_normalized_nth_input(cur_node, gm_a, 0)  # type: ignore[assignment]
    nodes.append(cur_node)
    nodes.reverse()

    def _can_insert(node_a_arg, gm_a):
        if isinstance(node_a_arg, Node):
            arg_a = return_first_non_observer_node(node_a_arg, gm_a)
            if arg_a.op == "call_method":
                return arg_a.target in ("dequantize", "to")
            elif arg_a.op == "get_attr":
                return True
            else:
                return False
        elif isinstance(node_a_arg, (list, tuple)):
            for el in node_a_arg:
                if not isinstance(el, Node):
                    return False
        return True

    # For each node, check if we handle the copy behavior. This follows the
    # logic in `_insert_copy_of_subgraph_a_after_input_node_c`.
    for node_a in nodes:
        local_num_non_param_args_node_a = (
            num_non_param_args_node_a if node_a is nodes[0] else 1
        )

        norm_args_kwargs = node_a.normalized_arguments(
            gm_a, normalize_to_only_use_kwargs=True
        )
        if norm_args_kwargs is not None:
            norm_args, norm_kwargs = norm_args_kwargs
        else:
            norm_args, norm_kwargs = node_a.args, node_a.kwargs

        cur_idx = 0

        while cur_idx < len(norm_args):
            if cur_idx == 0:
                pass
            elif cur_idx == 1 and local_num_non_param_args_node_a == 2:
                pass
            else:
                if not _can_insert(norm_args[cur_idx], gm_a):
                    return False
            cur_idx += 1

        for kwarg_val in norm_kwargs.values():
            # stitch the inputs from base graph
            if cur_idx == 0:
                pass
            elif cur_idx == 1 and local_num_non_param_args_node_a == 2:
                pass
            else:
                if not _can_insert(kwarg_val, gm_a):
                    return False
            cur_idx += 1

    return True


def _insert_copy_of_subgraph_a_after_input_node_c(
    input_node_c: Node | list[Node],
    input_node_c_2: Node | list[Node] | None,
    subgraph_a: NSSubgraph,
    gm_a: GraphModule,
    gm_b: GraphModule,
    node_name_prefix: str,
) -> Node:
    """
    TODO(before land): real docblock
    """
    if not isinstance(input_node_c, (Node, list)):
        raise AssertionError(f"Expected Node or list, got {type(input_node_c)}")

    # create a sequential list of the subgraphs' nodes from start to end,
    # because we need to add the nodes to graph C in non-reverse order
    nodes_of_a = [subgraph_a.end_node]
    cur_node = subgraph_a.end_node
    while cur_node != subgraph_a.start_node:
        cur_node = get_normalized_nth_input(cur_node, gm_a, 0)  # type: ignore[assignment]
        nodes_of_a.insert(0, cur_node)

    # go through nodes of a in order, and insert them into the graph of c
    # sequentially
    cur_node_a = nodes_of_a[0]
    cur_node_c = _insert_copy_of_node_a_after_input_node_c(
        input_node_c, input_node_c_2, cur_node_a, gm_a, gm_b, node_name_prefix
    )
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
            node_name_prefix,
        )
    # return the last inserted node
    return cur_node_c


def _insert_copy_of_node_a_after_input_node_c(
    input_node_c: Node | list[Node],
    input_node_c_2: Node | list[Node] | None,
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
        if not isinstance(input_node_c, list):
            raise AssertionError(f"Expected list, got {type(input_node_c)}")
        graph_c = input_node_c[0].graph

    norm_args_kwargs = node_a.normalized_arguments(
        gm_a, normalize_to_only_use_kwargs=True
    )
    if norm_args_kwargs is not None:
        norm_args, norm_kwargs = norm_args_kwargs
    else:
        norm_args, norm_kwargs = node_a.args, node_a.kwargs

    new_args = []
    new_kwargs = {}

    def _copy_arg(arg):
        # copy the other inputs from the other graph
        if isinstance(arg, Node):
            arg = return_first_non_observer_node(arg, gm_a)
            arg = _copy_node_from_a_to_c(arg, gm_a, gm_b, graph_c)
            return arg
        elif isinstance(arg, (int, float, torch.dtype)):
            return arg
        elif isinstance(kwarg_val, (list, tuple)):
            for el in kwarg_val:
                if isinstance(el, Node):
                    raise AssertionError(
                        "handling of Node inside list is not implemented"
                    )
            return arg
        else:
            raise AssertionError(
                f"handling for kwarg of type {type(kwarg_val)} is not implemented"
            )

    cur_idx = 0

    while cur_idx < len(norm_args):
        if cur_idx == 0:
            new_arg = input_node_c
        elif cur_idx == 1 and input_node_c_2 is not None:
            new_arg = input_node_c_2
        else:
            new_arg = _copy_arg(norm_args[cur_idx])
        new_args.append(new_arg)
        cur_idx += 1

    for kwarg_name, kwarg_val in norm_kwargs.items():
        # stitch the inputs from base graph
        if cur_idx == 0:
            new_kwargs[kwarg_name] = input_node_c
        elif cur_idx == 1 and input_node_c_2 is not None:
            new_kwargs[kwarg_name] = input_node_c_2
        else:
            new_kwargs[kwarg_name] = _copy_arg(kwarg_val)
        cur_idx += 1

    new_args = tuple(new_args)  # type: ignore[assignment]

    node_a_shadows_c_name = get_new_attr_name_with_prefix(node_name_prefix)(gm_b)

    if node_a.op == "call_module":
        # if target is a module, we point to the module from gm_b
        new_mod_copy_name = get_new_attr_name_with_prefix(node_name_prefix)(gm_b)
        # fetch the corresponding module from gm_a
        if not isinstance(node_a.target, str):
            raise AssertionError(f"Expected str, got {type(node_a.target)}")
        mod_a = getattr_from_fqn(gm_a, node_a.target)
        setattr(gm_b, new_mod_copy_name, mod_a)
        node_a_shadows_c = graph_c.create_node(
            node_a.op,
            new_mod_copy_name,
            new_args,  # type: ignore[arg-type]
            new_kwargs,  # type: ignore[arg-type]
            node_a_shadows_c_name,
        )
        return node_a_shadows_c
    else:
        if node_a.op not in ("call_function", "call_method"):
            raise AssertionError(f"Unexpected op: {node_a.op}")
        node_a_shadows_c = graph_c.create_node(
            node_a.op,
            node_a.target,
            new_args,  # type: ignore[arg-type]
            new_kwargs,  # type: ignore[arg-type]
            node_a_shadows_c_name,
        )
        return node_a_shadows_c


def create_a_shadows_b(
    name_a: str,
    gm_a: GraphModule,
    name_b: str,
    gm_b: GraphModule,
    matched_subgraph_pairs: dict[str, tuple[NSSubgraph, NSSubgraph]],
    logger_cls: Callable,
    should_log_inputs: bool,
    node_type_to_io_type_map: dict[str, set[NSNodeTargetType]] | None = None,
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
    env_c: dict[str, Any] = {}

    def load_arg(a):
        return map_arg(a, lambda node: env_c[node.name])

    start_node_b_to_matched_subgraph_a_and_name = {}
    end_node_b_to_matched_subgraph_a_and_name = {}
    for match_name, match in matched_subgraph_pairs.items():
        subgraph_a, subgraph_b = match
        ref_node_type_a = get_target_type_str(subgraph_a.base_op_node, gm_a)
        ref_node_type_b = get_target_type_str(subgraph_b.base_op_node, gm_b)
        start_node_b_to_matched_subgraph_a_and_name[subgraph_b.start_node] = (
            subgraph_a,
            match_name,
            ref_node_type_a,
            ref_node_type_b,
        )
        end_node_b_to_matched_subgraph_a_and_name[subgraph_b.end_node] = (
            subgraph_a,
            match_name,
            ref_node_type_a,
            ref_node_type_b,
        )

    for node_b in gm_b.graph.nodes:
        if node_b.op == "output":
            graph_c.output(map_arg(node_b.args[0], load_arg))
            continue

        # calculate the flags to determine what to do with this node
        node_b_is_start_node = node_b in start_node_b_to_matched_subgraph_a_and_name
        node_b_is_end_node = node_b in end_node_b_to_matched_subgraph_a_and_name

        if node_b_is_start_node or node_b_is_end_node:
            if node_b_is_start_node:
                (
                    subgraph_a,
                    ref_name,
                    ref_node_type_a,
                    ref_node_type_b,
                ) = start_node_b_to_matched_subgraph_a_and_name[node_b]
            else:
                if not node_b_is_end_node:
                    raise AssertionError("Expected node_b_is_end_node to be not false")
                (
                    subgraph_a,
                    ref_name,
                    ref_node_type_a,
                    ref_node_type_b,
                ) = end_node_b_to_matched_subgraph_a_and_name[node_b]

            all_op_types_support_shadowing = op_type_supports_shadowing(
                subgraph_a.start_node
            ) and op_type_supports_shadowing(node_b)
            if not all_op_types_support_shadowing:
                print(
                    f"skipping shadow loggers for node_b: {get_target_type_str(node_b, gm_b)}"
                    + f", start_node_a: {get_target_type_str(subgraph_a.start_node, gm_a)}"
                    + ", unsupported"
                )
                env_c[node_b.name] = graph_c.node_copy(node_b, load_arg)
                continue

            # For both start_node and end_node verify that we know how to do
            # the dtype cast. If we do not, skip.
            (
                node_input_type_a,
                node_output_type_a,
            ) = get_node_first_input_and_output_type(
                subgraph_a.start_node, gm_a, logger_cls, node_type_to_io_type_map
            )
            (
                node_input_type_b,
                node_output_type_b,
            ) = get_node_first_input_and_output_type(
                node_b, gm_b, logger_cls, node_type_to_io_type_map
            )
            node_io_types_known_a_and_b = (
                node_input_type_a != NodeInputOrOutputType.UNKNOWN
                and node_output_type_a != NodeInputOrOutputType.UNKNOWN
                and node_input_type_b != NodeInputOrOutputType.UNKNOWN
                and node_output_type_b != NodeInputOrOutputType.UNKNOWN
            )
            if not node_io_types_known_a_and_b:
                print(
                    f"skipping shadow loggers for node_b: {get_target_type_str(node_b, gm_b)}"
                    + f", start_node_a: {get_target_type_str(subgraph_a.start_node, gm_a)}"
                    + ", unknown dtype cast"
                )
                env_c[node_b.name] = graph_c.node_copy(node_b, load_arg)
                continue

            # If we are shadowing from fp32 to int8, we need to insert
            # quantize_per_tensor call with qparams from the previous node.
            # Only do this if we are able to infer these qparams from the graph.
            if (
                node_input_type_a == NodeInputOrOutputType.INT8
                and node_input_type_b == NodeInputOrOutputType.FP32
            ):
                node_a_input_qparams = get_node_input_qparams(
                    subgraph_a.start_node, gm_a, node_type_to_io_type_map
                )
                if not node_a_input_qparams:
                    print(
                        f"skipping shadow loggers for node_b: {get_target_type_str(node_b, gm_b)}"
                        + f", start_node_a: {get_target_type_str(subgraph_a.start_node, gm_a)}"
                        + ", unknown input qparams"
                    )
                    env_c[node_b.name] = graph_c.node_copy(node_b, load_arg)
                    continue

            num_non_param_args_node_a = get_number_of_non_param_args(
                subgraph_a.start_node, gm_a
            )
            if not _can_insert_copy_of_subgraph_a(
                subgraph_a, gm_a, num_non_param_args_node_a
            ):
                print(
                    f"skipping shadow loggers for node_b: {get_target_type_str(node_b, gm_b)}"
                    + f", start_node_a: {get_target_type_str(subgraph_a.start_node, gm_a)}"
                    + ", unhandled logic in subgraph copy"
                )
                env_c[node_b.name] = graph_c.node_copy(node_b, load_arg)
                continue

            fqn_base_a = _maybe_get_fqn(subgraph_a.base_op_node, gm_a)
            fqn_base_b = _maybe_get_fqn(subgraph_b.base_op_node, gm_b)  # type: ignore[possibly-undefined]

            if node_b_is_start_node:
                # if necessary, log the input of node_c
                if should_log_inputs:
                    prev_node_b = get_normalized_nth_input(node_b, gm_b, 0)
                    if isinstance(prev_node_b, Node):
                        prev_node_c = env_c[prev_node_b.name]
                        env_c[prev_node_c.name] = _insert_logger_after_node(
                            prev_node_c,
                            gm_b,
                            logger_cls,
                            "_ns_logger_b_inp_",
                            node_b.name,
                            name_b,
                            ref_name,
                            ref_node_type_b,
                            NSSingleResultValuesType.NODE_INPUT.value,
                            index_within_arg=0,
                            index_of_arg=0,
                            fqn=fqn_base_b,
                        )
                    elif isinstance(prev_node_b, list):
                        # first, save the prev_node instances, because they
                        # will be overwritten in the env after the first logger
                        # is added
                        prev_node_c_list = [env_c[arg.name] for arg in prev_node_b]

                        for arg_idx, prev_node_c in enumerate(prev_node_c_list):
                            env_c[prev_node_c.name] = _insert_logger_after_node(
                                prev_node_c,
                                gm_b,
                                logger_cls,
                                "_ns_logger_b_inp_",
                                node_b.name,
                                name_b,
                                ref_name,
                                ref_node_type_b,
                                NSSingleResultValuesType.NODE_INPUT.value,
                                index_within_arg=arg_idx,
                                index_of_arg=0,
                                fqn=fqn_base_b,
                            )
                    else:
                        # logging of inputs which are not lists is not supported yet
                        raise AssertionError(
                            f"type {type(prev_node_b)} is not handled yet"
                        )
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
                # prev_node_c = node_c.args[0]
                prev_node_c = get_normalized_nth_input(node_c, gm_b, 0)  # type: ignore[possibly-undefined]
                if should_log_inputs:
                    # skip the input logger when inserting a dtype cast
                    if isinstance(prev_node_c, Node):
                        # pyrefly: ignore [unbound-name]
                        prev_node_c = get_normalized_nth_input(node_c, gm_b, 0)
                    elif isinstance(prev_node_c, list):
                        prev_node_c = [
                            get_normalized_nth_input(arg, gm_b, 0)
                            for arg in prev_node_c
                        ]
                dtype_cast_node = _insert_dtype_cast_after_node(
                    subgraph_a.start_node,
                    # pyrefly: ignore [unbound-name]
                    node_c,
                    prev_node_c,
                    gm_a,
                    gm_b,
                    graph_c,
                    node_b.name + "_dtype_cast_",
                    logger_cls,
                    node_type_to_io_type_map,
                )
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
                    ref_node_name = ""
                    if isinstance(dtype_cast_node, Node):
                        dtype_cast_node = _insert_logger_after_node(
                            dtype_cast_node,
                            gm_b,
                            logger_cls,
                            "_ns_logger_a_inp_",
                            ref_node_name,
                            name_a,
                            ref_name,
                            ref_node_type_a,
                            NSSingleResultValuesType.NODE_INPUT.value,
                            index_within_arg=0,
                            index_of_arg=0,
                            fqn=fqn_base_a,
                        )
                        input_logger: Node | list[Node] = dtype_cast_node
                    else:
                        if not isinstance(dtype_cast_node, list):
                            raise AssertionError(
                                f"Expected list, got {type(dtype_cast_node)}"
                            )
                        new_loggers = []
                        for dtype_cast_idx, dtype_cast_node_inner in enumerate(
                            dtype_cast_node
                        ):
                            dtype_cast_logger = _insert_logger_after_node(
                                dtype_cast_node_inner,
                                gm_b,
                                logger_cls,
                                "_ns_logger_a_inp_",
                                ref_node_name,
                                name_a,
                                ref_name,
                                ref_node_type_a,
                                NSSingleResultValuesType.NODE_INPUT.value,
                                index_within_arg=dtype_cast_idx,
                                index_of_arg=0,
                                fqn=fqn_base_a,
                            )
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
                num_non_param_args_node_a = get_number_of_non_param_args(
                    subgraph_a.start_node, gm_a
                )
                if num_non_param_args_node_a == 2:
                    # node_c_second_non_param_arg = node_c.args[1]
                    node_c_second_non_param_arg = get_normalized_nth_input(
                        # pyrefly: ignore [unbound-name]
                        node_c,
                        gm_b,
                        1,
                    )
                node_a_shadows_c = _insert_copy_of_subgraph_a_after_input_node_c(
                    dtype_cast_node,
                    node_c_second_non_param_arg,
                    subgraph_a,
                    gm_a,
                    gm_b,
                    # pyrefly: ignore [unbound-name]
                    node_c.name + "_shadow_copy_",
                )
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
                    while get_normalized_nth_input(cur_node, gm_b, 0) != input_logger:  # type: ignore[possibly-undefined]
                        cur_node = get_normalized_nth_input(cur_node, gm_b, 0)  # type: ignore[assignment]
                    # pyrefly: ignore [unbound-name]
                    if isinstance(input_logger, Node):
                        # pyrefly: ignore [unbound-name]
                        input_logger_mod = getattr(gm_b, input_logger.name)
                        input_logger_mod.ref_node_name = cur_node.name
                    else:
                        # pyrefly: ignore [unbound-name]
                        if not isinstance(input_logger, list):
                            raise AssertionError(
                                # pyrefly: ignore [unbound-name]
                                f"Expected list, got {type(input_logger)}"
                            )
                        # pyrefly: ignore [unbound-name]
                        for input_logger_inner in input_logger:
                            input_logger_mod = getattr(gm_b, input_logger_inner.name)
                            input_logger_mod.ref_node_name = cur_node.name

                # hook up a logger to the mod_a copy
                env_c[node_a_shadows_c.name] = _insert_logger_after_node(
                    env_c[node_a_shadows_c.name],
                    gm_b,
                    logger_cls,
                    "_ns_logger_a_",
                    node_a_shadows_c.name,
                    name_a,
                    ref_name,
                    ref_node_type_a,
                    NSSingleResultValuesType.NODE_OUTPUT.value,
                    index_within_arg=0,
                    index_of_arg=0,
                    fqn=fqn_base_a,
                )
                # subgraph so far:
                #
                #       dtype_cast_node -> (logger_a_input)? -> subgraph_a_copy -> logger_a
                #                  /
                # (prev_node_c)+ -> (logger_c_input)? -> node_start_c

            if node_b_is_end_node:
                # hook up a logger to the mod_b copy
                env_c[node_b.name] = _insert_logger_after_node(
                    env_c[node_b.name],
                    gm_b,
                    logger_cls,
                    "_ns_logger_b_",
                    node_b.name,
                    name_b,
                    ref_name,
                    ref_node_type_b,
                    NSSingleResultValuesType.NODE_OUTPUT.value,
                    index_within_arg=0,
                    index_of_arg=0,
                    fqn=fqn_base_b,
                )
                # subgraph so far:
                #
                #       dtype_cast_node -> (logger_a_input)? -> subgraph_a_copy -> logger_a
                #                  /
                # (prev_node_c+) -> (logger_c_input)? -> node_start_c -> ... -> node_end_c -> logger_c
                #
                # Note: node_start_c may be the same node as node_end_c, or they
                # may have nodes in between.

        else:
            env_c[node_b.name] = graph_c.node_copy(node_b, load_arg)

    gm_c = GraphModule(gm_b, graph_c)
    return gm_c
