import operator
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn

from functorch import make_fx
from torch import fx
from torch._inductor.compile_fx import compile_fx_inner
from torch._inductor.decomposition import select_decomp_table
from torch.distributed._spmd.graph_utils import dump_graphs_to_files, OP
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
    orig_placeholders = []
    orig_output_args: List[Any] = []
    output = next(iter(gm.graph.nodes))
    move_nodes = []

    for node in gm.graph.nodes:
        if node.op == OP.OUTPUT:
            output = node
            orig_output_args, _ = tree_flatten(node.args)
        elif node.op == OP.PLACEHOLDER:
            orig_placeholders.append(node)
        else:
            move_nodes.append(node)
    orig_placeholders_set = set(orig_placeholders)

    subgraph: torch.fx.Graph = torch.fx.Graph()
    node_mapping: Dict[torch.fx.Node, torch.fx.Node] = {}
    attrs = {}
    for i, p in enumerate(orig_placeholders):
        node_mapping[p] = subgraph.placeholder(name=f"placeholder_{i}")

    for node in move_nodes:
        if node.op == OP.GET_ATTR:
            attrs[node.target] = getattr(gm, node.target)
        args, kwargs = tree_map_only(
            fx.Node, lambda n: node_mapping[n], (node.args, node.kwargs)
        )
        node_mapping[node] = subgraph.create_node(
            op=node.op, target=node.target, args=args, kwargs=kwargs
        )
        node_mapping[node].meta = node.meta

    placeholders: List[torch.fx.Node] = []
    for placeholder in orig_placeholders:
        new_placeholder = node_mapping[placeholder]
        if len(new_placeholder.users) == 0:
            # Remove unused placeholders from the subgraph. This is required as
            # the `train_step()` has module and optimizer as the inputs which
            # cannot be lowered to Inductor.
            subgraph.erase_node(new_placeholder)
        else:
            placeholders.append(placeholder)
    output_args = tuple(
        node_mapping[n] for n in orig_output_args if n not in orig_placeholders_set
    )
    subgraph.output(result=output_args)
    sub_gm = torch.fx.GraphModule(root=attrs, graph=subgraph)
    gm.subgraph = InductorWrapper(sub_gm, enable_cudagraphs)
    with gm.graph.inserting_after(move_nodes[-1]):
        subgraph_call = gm.graph.create_node(
            op=OP.CALL_MODULE, target="subgraph", args=tuple(placeholders)
        )

    # Redistribute the output from the subgraph to the original output.
    output_idx = 0
    for i, node in enumerate(orig_output_args):
        if node in orig_placeholders_set:
            continue
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
        *,
        enable_inductor: bool = False,
        enable_cudagraphs: bool = False,
        dump_graphs: bool = False,
    ) -> None:
        self.num_iters = num_iters
        self.enable_inductor = enable_inductor
        self.enable_cudagraphs = enable_cudagraphs
        self.dump_graphs = dump_graphs

    def __call__(self, gm: fx.GraphModule) -> Callable:
        if self.dump_graphs:
            graph_folder = dump_graphs_to_files(
                {"before_transformation_gm": gm.print_readable(False)}
            )

        iter_gm = IterGraphModule(gm)
        iter_gm.freeze_cross_iter_movement()
        iter_gm.setup(self.num_iters)

        if self.dump_graphs:
            dump_graphs_to_files(
                {
                    "iter_graph_setup_gm": iter_gm.setup_gm.print_readable(False),
                    "iter_graph_main_gm": iter_gm.main_gm.print_readable(False),
                    "iter_graph_cleanup_gm": iter_gm.cleanup_gm.print_readable(False),
                },
                graph_folder,
            )

        if self.enable_inductor:
            iter_gm.main_gm = lower_to_inductor(iter_gm.main_gm, self.enable_cudagraphs)

        return iter_gm
