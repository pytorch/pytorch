# This file is copied from Meta internal repo and is not synced with the
# internal version. Once the internal version is fully mature, we should
# upstream again and retire the internal version. @yifuwang

import logging
import operator
from typing import Callable, List, Optional, Set, Tuple

import torch
from functorch import make_fx
from torch._inductor.compile_fx import compile_fx_inner
from torch._inductor.decomposition import select_decomp_table


MIN_ATEN_OPS_TO_LOWER = 10

logger: logging.Logger = logging.getLogger(__name__)


def _create_subgraph_module(
    inputs: List[torch.fx.Node], body: List[torch.fx.Node], outputs: List[torch.fx.Node]
) -> torch.fx.GraphModule:
    subgraph: torch.fx.Graph = torch.fx.Graph()
    node_to_subgraph_node = {}
    for idx, inp in enumerate(inputs):
        subgraph_inp = subgraph.placeholder(name=f"arg_{idx}")
        subgraph_inp.meta = inp.meta
        node_to_subgraph_node[inp] = subgraph_inp

    for node in body:
        subgraph_node = subgraph.node_copy(
            node, arg_transform=lambda x: node_to_subgraph_node[x]
        )
        node_to_subgraph_node[node] = subgraph_node

    subgraph.output(result=tuple(node_to_subgraph_node[x] for x in outputs))
    subgraph.eliminate_dead_code()
    subgraph.lint()
    return torch.fx.GraphModule(root={}, graph=subgraph)


def _is_container_node(node: torch.fx.Node) -> bool:
    if any(user.target == operator.getitem for user in node.users):
        assert all(user.target == operator.getitem for user in node.users), (
            "Malformed graph: a container node is used as input for non-getitem nodes."
            "\nNode: {fmt_node}\nUsers: {fmt_users}".format(
                fmt_node=node.format_node(),
                fmt_users="\n".join(u.format_node() for u in node.users),  # type: ignore[misc]
            )
        )
        return True
    return False


def _lower_subgraph_nodes(
    gm: torch.fx.GraphModule,
    subgraph_name: str,
    subgraph_nodes: List[torch.fx.Node],
    dumper: Callable[[str], str],
) -> None:
    prologue: List[torch.fx.Node] = []
    inputs: List[torch.fx.Node] = []
    body: List[torch.fx.Node] = []
    visible: Set[torch.fx.Node] = set()

    # Inductor requires all graph input to be tensors. When adding a container
    # node as subgraph input, add its descendant getitem nodes to the subgraph
    # prologue and add its leaf getitem nodes to the subgraph input.
    def add_input(arg: torch.fx.Node) -> None:
        stack = [arg]
        while len(stack) != 0:
            node = stack.pop()
            if _is_container_node(node):
                # We should only prepone nodes within subgraph_nodes
                prologue.extend(user for user in node.users if user in subgraph_nodes)
                stack.extend(node.users)
            else:
                if node not in visible:
                    inputs.append(node)
                    visible.add(node)

    for node in subgraph_nodes:
        if node.op == "get_attr":
            # Prepone get_attr to avoid having to copy
            # the attribute to the subgraph module.
            inputs.append(node)
            visible.add(node)
            continue

        for arg in node.all_input_nodes:
            if arg not in visible:
                add_input(arg)

        if node not in prologue:
            body.append(node)
            visible.add(node)

    outputs: List[torch.fx.Node] = []

    # Inductor requires all graph output to be tensors. When adding a container
    # node as subgraph output, add its descendant getitem nodes to the subgraph
    # body and add its leaf getitem nodes to the subgraph output.
    def add_output(output: torch.fx.Node) -> None:
        stack = [output]
        while len(stack) != 0:
            node = stack.pop()
            if _is_container_node(node):
                body.extend(node.users)
                stack.extend(node.users)
            elif not all(user in visible for user in node.users):
                if node not in outputs:
                    outputs.append(node)

    for node in body:
        if not all(user in visible for user in node.users):
            add_output(node)

    assert len(inputs) == len(set(inputs))
    assert len(outputs) == len(set(outputs))

    subgraph_module = _create_subgraph_module(inputs, body, outputs)
    readable_tag = dumper(str(subgraph_module.graph))
    setattr(gm, subgraph_name, _InductorModule(subgraph_module))

    insertion_point = subgraph_nodes[-1].next
    for node in prologue:
        insertion_point.prepend(node)

    with gm.graph.inserting_before(insertion_point):
        # Insert subgraph call
        subgraph_call = gm.graph.create_node(
            op="call_module",
            target=subgraph_name,
            args=tuple(inputs),
            kwargs={"tag": readable_tag},
        )
        # Replace parent graph nodes with their corresponding subgraph outputs
        for idx, output in enumerate(outputs):
            new_output = gm.graph.create_node(
                op="call_function",
                target=operator.getitem,
                args=(subgraph_call, idx),
            )
            new_output.meta = output.meta
            output.replace_all_uses_with(new_output)

    # Erase lowered nodes from the parent graph
    for node in reversed(body + outputs):
        if len(node.users) == 0:
            gm.graph.erase_node(node)


class _InductorModule(torch.nn.Module):
    def __init__(self, gm: torch.fx.GraphModule) -> None:
        super().__init__()
        self.gm = gm
        self.compiled: Optional[
            Callable[[List[torch.Tensor]], List[torch.Tensor]]
        ] = None

    def forward(self, *args: torch.Tensor, tag: str) -> List[torch.Tensor]:
        if self.compiled is None:
            inductor_decompositions = select_decomp_table()
            # TODO: figure out why turning on cudagraphs cause exceptions.
            decomp_gm = make_fx(self.gm, decomposition_table=inductor_decompositions)(
                *args
            )
            logger.info("Lowering subgraph (%s) to Inductor...", tag)
            self.compiled = compile_fx_inner(
                decomp_gm,
                list(args),
                cudagraphs=False,
            )
            logger.info("Completed lowering subgraph (%s) to Inductor", tag)
        with torch.profiler.record_function(tag):
            assert self.compiled is not None
            return self.compiled(list(args))


def _is_inductor_compatible(node: torch.fx.Node) -> Tuple[bool, str]:
    # `has_tag` is not supported yet
    # if has_tag(node, "non_lowerable"):

    if node.target in (
        torch.ops.aten._fused_adam_.default,
        torch.ops.aten._fused_adam.default,
        torch.ops.aten._foreach_add_.Scalar,
        torch.ops.aten._foreach_add.Scalar,
    ):
        return False, "fused adam is not supported yet"

    # TODO(yifu): apparently having a meta kernel is not a necessary
    # condition for Inductor compatiblity. We should refine the check.
    # Sneaking this one in for now to support comm_fusion_with_cat.
    if node.target == torch.ops.aten.flatten.using_ints:
        return True, ""

    if isinstance(node.target, torch._ops.OpOverload):
        if not node.target.has_kernel_for_dispatch_key(torch._C.DispatchKey.Meta):
            return False, f"{node.target} doesn't have a meta kernel registered"
    return True, ""


def _subgraph_predicate(nodes: List[torch.fx.Node]) -> bool:
    num_aten_ops = len([n for n in nodes if str(n.target).startswith("aten.")])
    return num_aten_ops >= MIN_ATEN_OPS_TO_LOWER


def partial_lower(
    gm: torch.fx.GraphModule,
    node_predicate: Callable[[torch.fx.Node], bool] = lambda x: True,
    subgraph_predicate: Callable[[List[torch.fx.Node]], bool] = lambda x: True,
    dumper: Callable[[str], str] = lambda x: "subgraph",
) -> torch.fx.GraphModule:
    """
    Lower Inductor compatible portions of the graph module to Inductor.

    Args:
        node_predicate: user predicate for determining whether to consider a node for
            lowering.
        subgraph_predicate: user predicate for determining whether to consider a list of
            candidate nodes for lowering.
        dumper: a callback for dumping subgraphs for human digestion. For exmaple, it
            can be a function that writes to disk/blob storage and returns the
            path/handle. The returned path/handle for each subgraph will be made
            available in the subgraph call node in the parent graph, as well as the
            label of the profiler block for the subgraph.
    """
    nodes_per_subgraph: List[List[torch.fx.Node]] = [[]]
    ptr = next(iter(gm.graph.nodes))

    def _node_predicate(node: torch.fx.Node) -> Tuple[bool, str]:
        should_lower, reason = _is_inductor_compatible(node)
        if not should_lower:
            return should_lower, reason
        if not node_predicate(node):
            return False, "user predicate"
        return True, ""

    while ptr.op != "output":
        if ptr.op == "placeholder":
            ptr = ptr.next
            continue
        should_lower, reason = _node_predicate(ptr)
        if should_lower:
            nodes_per_subgraph[-1].append(ptr)
        else:
            if len(nodes_per_subgraph[-1]) > 0:
                logger.warning(
                    "partial_lower: graph break at %s. Reason: %s", str(ptr), reason
                )
            nodes_per_subgraph.append([])
        ptr = ptr.next

    nodes_per_subgraph = [
        nodes
        for nodes in nodes_per_subgraph
        if subgraph_predicate(nodes) and _subgraph_predicate(nodes)
    ]

    for idx, subgraph_nodes in enumerate(nodes_per_subgraph):
        subgraph_name = f"subgraph_{idx}"
        _lower_subgraph_nodes(gm, subgraph_name, subgraph_nodes, dumper)

    gm.graph.lint()
    gm.recompile()
    return gm
