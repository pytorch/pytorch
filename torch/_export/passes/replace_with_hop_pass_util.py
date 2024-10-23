# mypy: allow-untyped-defs

import contextlib
import copy
import operator
from typing import Callable

import torch
from torch._ops import HigherOrderOperator

from ..utils import node_replace_, nodes_map


def _replace_with_hop_helper(
    node: torch.fx.Node,
    enter_block_node: torch.fx.Node,
    node_filter: Callable,
    wrap_hoo: HigherOrderOperator,
):
    graph: torch.fx.Graph = node.graph
    gm: torch.fx.GraphModule = graph.owning_module
    assert isinstance(node.target, str)
    sub_gm = getattr(gm, node.target)

    def set_hoo_node_meta(call_func_node):
        call_func_node.meta["nn_module_stack"] = copy.copy(
            enter_block_node.meta.get("nn_module_stack", {})
        )
        call_func_node.meta["torch_fn"] = (
            f"{wrap_hoo.__name__}",
            f"{wrap_hoo.__class__.__name__}.{wrap_hoo.__name__}",
        )
        if isinstance(output_args, (tuple, list)):
            call_func_node.meta["val"] = tuple(arg.meta["val"] for arg in output_args)
        elif isinstance(output_args, torch.fx.Node):
            call_func_node.meta["val"] = (output_args.meta["val"],)

    with graph.inserting_before(node):
        get_attr_node = graph.get_attr(node.target)
        get_attr_node.meta["nn_module_stack"] = copy.copy(
            enter_block_node.meta.get("nn_module_stack", {})
        )
        output_node = next(iter(reversed(sub_gm.graph.nodes)), None)
        # Split_module pass intentially doesn't add output node
        # if the graph doesn't return anything.
        # TODO (tmanlaibaatar) Figure out if this is right behaviour
        # for split_module
        if isinstance(output_node, torch.fx.Node) and output_node.op != "output":
            output_node = None
        if output_node is not None:
            assert len(output_node.args) == 1
            output_args = output_node.args[0]
            enter_block_node_args = enter_block_node.args
            if isinstance(output_args, (tuple, list)):
                call_func_node = graph.call_function(
                    wrap_hoo,
                    (*enter_block_node_args, get_attr_node, *node.args),
                    {},
                )
                # Create the metadata
                set_hoo_node_meta(call_func_node)
                node_replace_(node, call_func_node)

                # Rename the name of getitem nodes to the actual name of its contents
                # for passing verifier and better readability, also propagate metadata
                for get_item_node in call_func_node.users.keys():
                    idx: int = get_item_node.args[1]  # type: ignore[assignment]
                    output_node = output_args[idx]
                    get_item_node._rename(output_node.name)
                    get_item_node.meta = output_node.meta

            elif isinstance(output_args, torch.fx.Node):
                call_func_node = graph.create_node(
                    "call_function",
                    wrap_hoo,
                    (*enter_block_node_args, get_attr_node, *node.args),
                    {},
                    output_args.name,
                )
                # Modify the subgraph to output a singleton list.
                output_node.args = ((output_args,),)
                # Add in an extra `getitem(wrap_hoo, 0)` node to the toplevel graph.
                get_item_node = graph.create_node(
                    "call_function",
                    operator.getitem,
                    (call_func_node, 0),
                    {},
                )
                # Create the metadata
                get_item_node.meta = output_args.meta
                set_hoo_node_meta(call_func_node)
                node_replace_(node, get_item_node)
            else:
                raise NotImplementedError(
                    f"repalce_with_hop_pass doesnt' support output type {type(output_args)}"
                )
        else:
            # TODO (shangdiy): remove this line, since the export graph can be non-functional
            node.graph.erase_node(node)


def _sequential_split_and_maybe_inline_subgraphs_helper(
    new_gm: torch.fx.GraphModule,
    graph_signature,
    maybe_inline_or_replace_with_hop: Callable[[torch.fx.Node], None],
):
    """
    Helper function for replacing graph nodse with higher order nodes.
    For each subgraph in `new_gm`, decides whether to construct a HOO subgraph, or inline the calls
    back into the parent graph module, depending on `maybe_inline_or_replace_with_hop`.
    """
    # new_gm is a new graph module that could have different output args names.
    # We need to fix the graph signature.
    replace_ctx = contextlib.nullcontext()
    new_signature = None
    if graph_signature is not None:
        # Cannot deep copy a real ScriptObject, which is referenced
        # in the FakeScriptObject. Copy should be good enough to guard
        # against accidental mutation to original graph_signature.
        new_signature = copy.copy(graph_signature)
        new_gm_out_node = next(reversed(new_gm.graph.find_nodes(op="output")))
        assert new_gm_out_node.op == "output" and len(new_gm_out_node.args[0]) == len(
            new_signature.output_specs
        )
        for arg_node, out_spec in zip(
            new_gm_out_node.args[0], new_signature.output_specs
        ):
            if arg_node is None:
                assert out_spec.arg.value is None
            elif (
                isinstance(arg_node, torch.fx.Node)
                and out_spec.arg.name != arg_node.name
            ):
                out_spec.arg.name = arg_node.name

        replace_ctx = new_gm._set_replace_hook(new_signature.get_replace_hook())  # type: ignore[assignment]

    with replace_ctx:
        nodes_map(
            list(new_gm.graph.nodes),
            lambda node: (
                maybe_inline_or_replace_with_hop(node)
                if node.op == "call_module"
                else node
            ),
        )
    new_gm.recompile()
    return new_gm, new_signature


def _replace_with_hop_pass_helper(
    gm: torch.fx.GraphModule,
    graph_signature,
    sequential_split_and_maybe_inline_subgraphs: Callable,
):
    """
    Split gm into sub-graph-modules using `sequential_split_and_maybe_inline_subgraphs`, and
    then recursively call itself on each of the submodules.
    """
    new_gm, new_signature = sequential_split_and_maybe_inline_subgraphs(
        gm, graph_signature
    )
    # recursively call
    for node in new_gm.graph.nodes:
        if node.op == "get_attr":
            subgm = getattr(new_gm, node.target)
            if not isinstance(subgm, torch.fx.GraphModule):
                continue
            new_subgm, _ = _replace_with_hop_pass_helper(
                subgm,
                None,
                sequential_split_and_maybe_inline_subgraphs,
            )
            setattr(new_gm, node.target, new_subgm)

    new_gm.recompile()
    new_gm.graph.lint()
    return new_gm, new_signature
