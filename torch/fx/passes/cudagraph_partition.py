import dataclasses
from contextlib import ExitStack
from typing import Callable, Optional
from unittest.mock import patch

import torch
from torch import fx
from torch._library.utils import lookup_op
from torch.fx.passes.split_module import split_module
from torch.fx._utils import _dummy_wrapper
# from torch._inductor.utils import CUDAGraphWrapperMetadata, CUDAGraphWrapperType

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
            or (getattr(node.target, "default", None) in resolved_ops)
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
    split_gm = split_module(
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


def _compile_submod(gm: fx.GraphModule, submod_names_to_cudagraph: list[str], cudagraph_wrapper) -> fx.GraphModule:
    for node in gm.graph.nodes:
        if node.op == "call_module":
            target = node.target
            assert isinstance(target, str), f"Expected string target, got {target}:{type(target)}"

            if target not in submod_names_to_cudagraph:
                continue

            partition_id = submod_names_to_cudagraph.index(target)
            num_partitions = len(submod_names_to_cudagraph)
            # cg_metadata = CUDAGraphWrapperMetadata(num_partitions, partition_id)

            submod = getattr(gm, target)
            assert isinstance(submod, fx.GraphModule), f"Expected fx.GraphModule, got {submod}:{type(submod)}"

            # _dummy_wrapper is to make call_function happy
            cudagraphed_callable = cudagraph_wrapper(
                    submod, num_partitions, partition_id
                )

            gm.__dict__[target] = cudagraphed_callable

            # TODO: replace with call_function node
            # with gm.graph.inserting_after(node):
            #     new_node = gm.graph.call_function(
            #         cudagraphed_callable, args=node.args, kwargs=node.kwargs
            #     )
            #     new_node.meta = node.meta
            #     node.replace_all_uses_with(new_node)
            #     gm.graph.erase_node(node)
            #     del gm._modules[target]

    # gm.recompile()
    return gm


def cudagraph_partition_pass(
    gm: fx.GraphModule,
    example_inputs: list[torch.Tensor],
    input_clone_indices: list[int],
    split_ops: list[str],
    cudagraph_wrapper,
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

    # input_buffers refers to the tensors in the list, instead of the list itself
    static_input_buffers = list(example_inputs)

    # 2. Wrap submodules with CUDAGraphWrapper
    compiled_gm = _compile_submod(split_gm, submod_names_to_cudagraph, cudagraph_wrapper)

    if len(input_clone_indices) == 0:
        return compiled_gm

    # 3. Copy inputs to static tensors
    compiled_gm_forward = compiled_gm.forward

    def copy_and_call(*args):
        for i in input_clone_indices:
            static_input_buffers[i].copy_(args[i])
        return compiled_gm_forward(*static_input_buffers)

    compiled_gm.forward = copy_and_call

    return compiled_gm


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
        self.output = None
        self.has_warmup = False

    def __call__(self, *args, **kwargs):
        # assume that args and kwargs have been copied to
        # static tensors
        if not self.has_warmup:
            self.has_warmup = True
            return self.runnable(*args, **kwargs)

        if self.cudagraph is None:
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


def cudagraph_wrapper(fn: Callable, num_partitions: int, partition_id: int) -> Callable:
    """
    Wrap a function with CUDAGraphWrapper.

    @param fn: the function to be wrapped
    @param metadata: the metadata of the function

    @return: the wrapped function
    """
    gc_disable = partition_id != 0
    return CUDAGraphWrapper(fn, gc_disable)
