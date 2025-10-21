import dataclasses
from contextlib import ExitStack
from typing import Any, Callable, Optional
from unittest.mock import patch

import torch
from torch import fx
from torch._library.utils import lookup_op


# Part1: Split gm
def resolve_defined_ops(op_names: list[str]) -> list["torch._ops.OpOverload"]:
    """Resolve operator names to OpOverload objects.

    Skips operators that fail to resolve (e.g., operators not registered or
    model-specific operators not present in the current model).

    Note: Users should inspect the operator graph before lowering and ensure
    the specified operators are present in the final graph. Built-in PyTorch
    operators (aten::*, torch::*) may be decomposed, fused, or transformed
    during Inductor's compilation passes, so use them with caution.

    Args:
        op_names: List of operator names in PyTorch format
            (e.g., "namespace::op_name")

    Returns:
        List of successfully resolved operator overloads
    """
    resolved = []
    for op_name in op_names:
        try:
            resolved.append(lookup_op(op_name))
        except Exception:
            # Skip operators that don't exist (e.g., model-specific ops)
            continue

    return resolved


@dataclasses.dataclass
class SplitItem:
    submod_name: str
    graph_id: int
    is_cudagraph_unsafe: bool
    gm: fx.GraphModule


def split_graph(
    gm: fx.GraphModule, resolved_ops: list[torch._ops.OpOverload]
) -> tuple[fx.GraphModule, list[SplitItem]]:
    """
    Given a graph module and a list of ops, split the gm into subgraphs.

    @param gm: the graph module to be split
    @param resolved_ops: a list of ops to be split

    @return: a tuple of (split_gm, split_items)
    """
    # split graph by ops
    subgraph_id = 0
    node_to_subgraph_id = {}
    split_op_graphs = []
    for node in gm.graph.nodes:
        if node.op in ("output", "placeholder"):
            continue
        # Match node.target against resolved_ops
        # node.target can be OpOverloadPacket, need to check .default
        if node.op == "call_function" and (
            node.target in resolved_ops
            or (hasattr(node.target, "default") and node.target.default in resolved_ops)
        ):
            subgraph_id += 1
            node_to_subgraph_id[node] = subgraph_id
            split_op_graphs.append(subgraph_id)
            subgraph_id += 1
        else:
            node_to_subgraph_id[node] = subgraph_id

    # `keep_original_order` is important!
    # otherwise pytorch might reorder the nodes and
    # the semantics of the graph will change when we
    # have mutations in the graph
    split_gm = torch.fx.passes.split_module.split_module(
        gm, None, lambda node: node_to_subgraph_id[node], keep_original_order=True
    )

    split_items = []
    names = [name for (name, module) in split_gm.named_modules()]
    for name in names:
        if "." in name or name == "":
            # recursive child module or the root module
            continue

        module = getattr(split_gm, name)
        graph_id = int(name.replace("submod_", ""))
        split_items.append(
            SplitItem(name, graph_id, (graph_id in split_op_graphs), module)
        )

    # sort by integer graph_id, rather than string name
    split_items.sort(key=lambda x: x.graph_id)

    return split_gm, split_items


# Part 2: CG Wrapper
_global_graph_pool = torch.cuda.graph_pool_handle()


class CUDAGraphWrapper:
    def __init__(
        self,
        runnable: Callable,
        gc_disable: bool = False,
    ):
        self.runnable = runnable
        self.gc_disable = gc_disable
        self.graph_pool = _global_graph_pool
        self.cudagraph: Optional[torch.cuda.CUDAGraph] = None
        # self.input_addresses = None # TODO: add input_addresses for debugging.
        self.output = None

    def __call__(self, *args, **kwargs):
        # assume that args and kwargs have been copied to
        # static tensors

        if self.cudagraph is None:
            # self.input_addresses = [
            #     x.data_ptr() for x in args if isinstance(x, torch.Tensor)
            # ]
            self.cudagraph = torch.cuda.CUDAGraph()

            with ExitStack() as stack:
                if self.gc_disable:
                    # during every model forward for piecewise cudagraph
                    # mode, we will capture many pieces of cudagraphs
                    # (roughly one per layer). running gc again and again
                    # across layers will make the cudagraph capture very slow.
                    # therefore, we only run gc for the first graph,
                    # and disable gc for the rest of the graphs.
                    stack.enter_context(patch("gc.collect", lambda: None))
                    stack.enter_context(patch("torch.cuda.empty_cache", lambda: None))

                with torch.cuda.graph(self.cudagraph, pool=self.graph_pool):
                    # `output` is managed by pytorch's cudagraph pool
                    # TODO: use weak ref for output to reuse memory
                    self.output = self.runnable(*args, **kwargs)

        self.cudagraph.replay()
        return self.output


# Part 3: Patch gm with CG Wrapper
class PiecewiseCompileInterpreter(torch.fx.Interpreter):
    def __init__(
        self,
        module: torch.fx.GraphModule,
        submod_names_to_cudagraph: list[str],
    ):
        super().__init__(module)
        self.submod_names_to_cudagraph = submod_names_to_cudagraph

    def call_module(
        self,
        target: torch.fx.node.Target,
        args: tuple[torch.fx.node.Argument, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        assert isinstance(target, str)
        output = super().call_module(target, args, kwargs)

        if target in self.submod_names_to_cudagraph:
            index = self.submod_names_to_cudagraph.index(target)
            is_first_graph = index == 0
            submod = self.fetch_attr(target)
            self.module.__dict__[target] = CUDAGraphWrapper(
                submod, gc_disable=not is_first_graph
            )

        return output


def cudagraph_partition_pass(
    gm: fx.GraphModule,
    example_inputs: list[torch.Tensor],
    input_clone_indices: list[int],
    split_ops: list[str],
) -> fx.GraphModule:
    """
    Partition the graph into subgraphs and wrap them with CUDAGraphWrapper.

    @param gm: the graph to be partitioned
    @param split_ops: a list of ops to be partitioned
    @param input_clone_indices: a list of indices of the inputs that need to be cloned
        to static tensors. This is needed since CUDAGraph requires static tensors.
        This includes user inputs. This does not include parameters and buffers,
        which are already static tensors.

    @return: the partitioned graph
    """
    # 1. Split graph
    resolved_split_ops = resolve_defined_ops(split_ops)
    split_gm, split_items = split_graph(gm, resolved_split_ops)
    submod_names_to_cudagraph = [
        item.submod_name for item in split_items if not item.is_cudagraph_unsafe
    ]

    # split_gm.input_buffers refers to the tensors in the list, instead of the list itself
    split_gm.input_buffers = [t for t in example_inputs]

    # 2. Wrap submodules with CUDAGraphWrapper
    PiecewiseCompileInterpreter(split_gm, submod_names_to_cudagraph).run(
        *example_inputs
    )

    # 3. Copy inputs to static tensors

    # TODO: Add a config to say whether we want to copy inputs to static tensors

    # TODO: Add a config to say whether we copy for submods, if users do not want to write an out_variant.

    split_gm_forward = split_gm.forward

    def copy_and_call(*args):
        for i in input_clone_indices:
            split_gm.input_buffers[i].copy_(args[i])
        return split_gm_forward(*split_gm.input_buffers)

    split_gm.forward = copy_and_call

    return split_gm
