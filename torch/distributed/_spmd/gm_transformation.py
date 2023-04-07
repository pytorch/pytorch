import operator
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn

from functorch import make_fx
from torch import fx
from torch._inductor.compile_fx import compile_fx_inner
from torch._inductor.decomposition import select_decomp_table
from torch.distributed._spmd.graph_utils import OP
from torch.distributed._spmd.iter_graph_module import IterGraphModule
from torch.utils._pytree import tree_flatten, tree_map_only


class InductorWrapper(nn.Module):
    def __init__(self, gm: fx.GraphModule, enable_cudagraphs: bool) -> None:
        super().__init__()
        self._gm = gm
        self._compiled: Optional[nn.Module] = None
        self._enable_cudagraphs = enable_cudagraphs

    def forward(self, *args: Any) -> Any:
        if self._compiled is None:
            gm = make_fx(self._gm, decomposition_table=select_decomp_table())(*args)
            self._compiled = compile_fx_inner(
                gm,
                list(args),
                cudagraphs=self._enable_cudagraphs,
            )
        list_args, _ = tree_flatten(args)
        return self._compiled(list_args)


def lower_to_inductor(
    gm: torch.fx.GraphModule, enable_cudagraphs: bool
) -> torch.fx.GraphModule:
    """
    This API lowers the entire `gm` to the Inductor
    """
    orig_placeholders: List[fx.Node] = []
    orig_output_args: List[Any] = []
    output: fx.Node = next(iter(gm.graph.nodes))
    move_nodes: List[fx.Node] = []

    for node in gm.graph.nodes:
        if node.op == OP.OUTPUT:
            output = node
            orig_output_args, _ = tree_flatten((node.args, node.kwargs))
        elif node.op == OP.PLACEHOLDER:
            orig_placeholders.append(node)
        else:
            move_nodes.append(node)

    subgraph: torch.fx.Graph = torch.fx.Graph()
    node_mapping: Dict[torch.fx.Node, torch.fx.Node] = {}
    attrs = {}

    # Map all the inputs/placeholders first.
    for p in orig_placeholders:
        node_mapping[p] = subgraph.node_copy(p)

    # Create all other non-placeholders nodes
    for node in move_nodes:
        if node.op == OP.GET_ATTR:
            attrs[node.target] = getattr(gm, node.target)
        node_mapping[node] = subgraph.node_copy(node, lambda n: node_mapping[n])

    output_args = tuple(node_mapping[n] for n in orig_output_args)
    subgraph.output(result=output_args)

    # Remove unused placeholders from the subgraph. This is required as the
    # `train_step()` has module and optimizer as the inputs which cannot be
    # lowered to Inductor.
    placeholders: List[torch.fx.Node] = []
    for placeholder in orig_placeholders:
        new_placeholder = node_mapping[placeholder]
        if len(new_placeholder.users) == 0:
            subgraph.erase_node(new_placeholder)
        else:
            placeholders.append(placeholder)

    # Create the subgraph node in the original graph.
    sub_gm = torch.fx.GraphModule(root=attrs, graph=subgraph)
    gm.subgraph = InductorWrapper(sub_gm, enable_cudagraphs)
    with gm.graph.inserting_after(move_nodes[-1]):
        subgraph_call = gm.graph.create_node(
            op=OP.CALL_MODULE, target="subgraph", args=tuple(placeholders)
        )

    # Redistribute the output from the subgraph to the original output.
    output_idx = 0
    for i, node in enumerate(orig_output_args):
        with gm.graph.inserting_after(subgraph_call):
            new_node = gm.graph.call_function(
                operator.getitem, (subgraph_call, output_idx)
            )
            output_idx += 1
        orig_output_args[i] = new_node
    assert output_idx == len(output_args)
    gm.graph.erase_node(output)
    gm.graph.output(result=orig_output_args)

    gm.graph.eliminate_dead_code()
    gm.recompile()

    return gm


class GraphModuleTransformation:
    def __init__(
        self,
        num_iters: int,
        enable_inductor: bool = False,
        enable_cudagraphs: bool = False,
    ) -> None:
        self.num_iters = num_iters
        self.enable_inductor = enable_inductor
        self.enable_cudagraphs = enable_cudagraphs

    def __call__(self, gm: fx.GraphModule) -> Callable:
        iter_gm = IterGraphModule(gm)
        iter_gm.freeze_cross_iter_movement()
        iter_gm.setup(self.num_iters)

        if self.enable_inductor:
            iter_gm.main_gm = lower_to_inductor(iter_gm.main_gm, self.enable_cudagraphs)

        return iter_gm
