"""
Toy example to show an end-to-end example of using Inductor to compile a region of aot_eager graph.

- Backend: aot_eager_hackable - Basically aot_eager with fw_ and bw_compiler set
to pass that finds flex region and compiles it.
- Uses CapabilityBasedPartitioner to find the flex region
- Calls compile_fx_inner to compile the flex region
- Replaces the flex region (call_module) with the compiled callable (call_function)


Things learnt or not tried out yet

1) I think aot_eager_hackable is wrong, we need fx_traceback.annotate. This will
require me to change the operator support object.
2) No symbolic shapes yet.
3) Need to run a model with multiple attention layers to see how partitioner
works.
4) Need to compose with AC (or maybe SAC)
5) Stretch - make it work with SimpleFSDP playground.
"""

import functools
from typing import Any

import torch
from torch._dynamo.backends.common import aot_autograd as auto_autograd_backend
from torch._guards import TracingContext
from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner
from torch.fx.passes.operator_support import OperatorSupport
from torch.fx.passes.utils.fuser_utils import fuse_by_partitions


def listify_inputs(fn):
    # Handles boxed arguments expectation from compile_fx_inner
    @functools.wraps(fn)
    def inner(*args):
        return fn(list(args))

    return inner


def partition_by_supported_nodes(gm, supported_ops, prefix):
    partitioner = CapabilityBasedPartitioner(
        gm, supported_ops, allows_single_node_partition=True
    )

    candidate_partitions = partitioner.propose_partitions()
    partitioned_gm = fuse_by_partitions(
        partitioner.graph_module,
        [candidate_partitions[0].nodes],
        prefix=prefix,
        always_return_tuple=True,
    )

    return partitioned_gm


def compile_submod(gm, prefix):
    for node in gm.graph.nodes:
        if node.op == "call_module" and node.target.startswith(prefix):
            fake_inputs = []
            for inp_node in node.all_input_nodes:
                if hasattr(inp_node, "meta") and "val" in inp_node.meta:
                    fake_inputs.append(inp_node.meta["val"])

            submod = getattr(gm, node.target)
            # Ensure that it runs in eager
            submod(*fake_inputs)

            from torch._inductor.compile_fx import compile_fx_inner

            # [inductor-stateless-issue] - Calling compile_fx_inner is changing
            # the reported output strides.
            with TracingContext.report_output_strides():
                compiled_submod = listify_inputs(compile_fx_inner(submod, fake_inputs))

            with gm.graph.inserting_after(node):
                new_node = gm.graph.call_function(
                    compiled_submod, args=node.args, kwargs=node.kwargs
                )
                new_node.meta = node.meta
                node.replace_all_uses_with(new_node)
                gm.graph.erase_node(node)
                del gm._modules[node.target]

    gm.recompile()
    return gm


def find_flex_nodes(gm):
    flex_nodes = set(
        gm.graph.find_nodes(
            op="call_function", target=torch.ops.higher_order.flex_attention
        )
    )

    flex_bwd_nodes = set(
        gm.graph.find_nodes(
            op="call_function", target=torch.ops.higher_order.flex_attention_backward
        )
    )

    flex_nodes = flex_nodes.union(flex_bwd_nodes)

    flex_subgraph_nodes = set()
    for flex_node in flex_nodes:
        for arg in flex_node.all_input_nodes:
            if arg.op == "get_attr":
                flex_subgraph_nodes.add(arg)

    supported_nodes = flex_nodes.union(flex_subgraph_nodes)

    class FlexOperatorSupport(OperatorSupport):
        def is_node_supported(self, submodules, node: torch.fx.Node) -> bool:
            return node in supported_nodes

    return FlexOperatorSupport()


def compile_flex_with_inductor(gm):
    flex_nodes = find_flex_nodes(gm)
    gm = partition_by_supported_nodes(gm, flex_nodes, "___flex_submod")
    gm = compile_submod(gm, "___flex_submod")
    return gm


def run_graph_passes(
    gm: torch.fx.GraphModule, example_inputs: Any
) -> torch.fx.GraphModule:
    gm = compile_flex_with_inductor(gm)
    return gm


aot_eager_hackable = auto_autograd_backend(
    fw_compiler=run_graph_passes,
    bw_compiler=run_graph_passes,
    keep_inference_input_mutations=True,
)


# Test code

from torch.nn.attention.flex_attention import create_block_mask, flex_attention


def _squared(score, b, h, m, n):
    """Joint graph needed for correctness"""
    return score * score


def mask_mod(b, h, q, k):
    return q >= 0


a = 12
b = 64
block_mask = create_block_mask(mask_mod, None, None, a * b, a * b)


def fn(x: torch.Tensor):
    a = torch.cos(x)
    b = flex_attention(a, a, a, block_mask=block_mask, score_mod=_squared)
    c = torch.cos(b)
    return c


opt_fn = torch.compile(fn, backend=aot_eager_hackable)


v = torch.randn(
    1,
    1,
    a * b,
    b,
    dtype=torch.bfloat16,
    device="cuda",
    requires_grad=True,
)

fn(v)
out = opt_fn(v)
out.sum().backward()
