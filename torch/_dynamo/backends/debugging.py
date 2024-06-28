# mypy: ignore-errors

import dataclasses
import functools
from importlib import import_module
from typing import Any, List, Optional

import torch
from functorch.compile import min_cut_rematerialization_partition
from torch import _guards
from torch._functorch import config as functorch_config
from torch._functorch.compilers import ts_compile

from .common import aot_autograd
from .registry import register_debug_backend as register_backend


"""
This file contains TorchDynamo backends intended for debugging uses.
"""


@register_backend
def eager(gm, fake_tensor_inputs):
    return gm.forward


@register_backend
def eager_noexcept(gm, fake_tensor_inputs):
    # This backend is intended to check that dynamo-generated GraphModules
    # do not cause errors.
    def inner(*args):
        try:
            return gm(*args)
        except Exception as e:
            raise torch._dynamo.exc.TorchDynamoException(
                "Unexpected exception when running generated GraphModule"
            ) from e

    return inner


@register_backend
def pre_dispatch_eager(gm, fake_tensor_inputs):
    from torch.fx.experimental.proxy_tensor import make_fx

    def runnable_gm(*args):
        return torch.fx.Interpreter(gm).run(*args)

    pre_dispatch_gm = make_fx(runnable_gm, pre_dispatch=True)(*fake_tensor_inputs)
    pre_dispatch_gm.print_readable()

    return pre_dispatch_gm


@register_backend
def eager_debug(gm, fake_tensor_inputs):
    from torch._subclasses.schema_check_mode import SchemaCheckMode

    # We could add more debugging bits here.
    # Right now, this backend can be used to check for and error on
    # custom dispatcher ops that have incorrect schemas.
    def inner(*args):
        with SchemaCheckMode():
            return torch.fx.Interpreter(gm).run(*args)

    return inner


@register_backend(name="ts")
def torchscript(gm, fake_tensor_inputs):
    return torch.jit.script(gm)


# used boxed call to discard inputs when they are no longer needed
def boxed_nop(fx_g, example_inputs):
    def run(args):
        return torch.fx.Interpreter(fx_g).boxed_run(args)

    run._boxed_call = True
    return run


# Useful for debugging purpose
# aot_eager uses AOT Autograd backend with nop compiler. It is helpful in debugging.
aot_eager = aot_autograd(
    fw_compiler=boxed_nop,
    partition_fn=min_cut_rematerialization_partition,
    keep_inference_input_mutations=True,
)
register_backend(name="aot_eager", compiler_fn=aot_eager)

aot_eager_default_partitioner = aot_autograd(
    fw_compiler=boxed_nop, keep_inference_input_mutations=True
)
register_backend(
    name="aot_eager_default_partitioner", compiler_fn=aot_eager_default_partitioner
)


# Uses TorchInductor AOT Autograd decomps and partitioner to isolate aot vs
# inductor problems.
# aot_eager_decomp_partition just replaces the inductor compiler with nop to help
# isolate inductor vs aot_eager errors
def aot_eager_decomp_partition(gm, fake_tensor_inputs):
    with functorch_config.patch(unlift_effect_tokens=True):
        return aot_autograd(
            # these are taken from memory_efficient_fusion()
            fw_compiler=boxed_nop,
            bw_compiler=boxed_nop,
            # NB: lambda here is to delay import of inductor
            decompositions=lambda: import_module(
                "torch._inductor.compile_fx"
            ).select_decomp_table(),
            partition_fn=functools.partial(
                min_cut_rematerialization_partition, compiler="inductor"
            ),
        )(gm, fake_tensor_inputs)


register_backend(
    name="aot_eager_decomp_partition", compiler_fn=aot_eager_decomp_partition
)


# AOT Autograd with torchscript backend. Default partitioner.
# aot_ts uses torchscript backend. We can use this with both nnc and nvfuser
# by using the relevant fuser with torch.jit.fuser(...)
aot_ts = aot_autograd(fw_compiler=ts_compile)
register_backend(name="aot_ts", compiler_fn=aot_ts)

# These buggy backends are used for inducing bugs so that we can test
# our repro extraction / minifier scripts


class ReluCompileError(Exception):
    pass


class TestingOnlyCompileError(Exception):
    pass


@register_backend
def relu_compile_error_TESTING_ONLY(gm: torch.fx.GraphModule, example_inputs):
    for node in gm.graph.nodes:
        if node.target == torch.relu:
            raise ReluCompileError
    return gm


@register_backend
def relu_runtime_error_TESTING_ONLY(gm: torch.fx.GraphModule, example_inputs):
    for node in gm.graph.nodes:
        if node.target == torch.relu:
            node.target = torch._assert
            node.args = (False, "ReluRuntimeError")
    gm.recompile()
    return gm


@register_backend
def relu_accuracy_error_TESTING_ONLY(gm: torch.fx.GraphModule, example_inputs):
    for node in gm.graph.nodes:
        if node.target == torch.relu:
            node.target = torch.add
            node.args = (node.args[0], 1)
    gm.recompile()

    return gm


@register_backend
def non_leaf_compile_error_TESTING_ONLY(gm: torch.fx.GraphModule, example_inputs):
    # Require at least one non-trivial thing in the graph,
    # see https://github.com/pytorch/pytorch/issues/102898
    for node in gm.graph.nodes:
        if node.op == "call_function":
            break
    else:
        return gm
    for t in example_inputs:
        if not t.is_leaf:
            raise TestingOnlyCompileError
    return gm


@dataclasses.dataclass
class ExplainOutput:
    """
    This is the output of :func:`torch._dynamo.explain()`
    There is no reason to create this class directly.
    """

    graphs: List[torch.fx.GraphModule]
    graph_count: int
    graph_break_count: int
    break_reasons: List[
        Any
    ]  # Type is GraphCompileReason but doesn't matter for this purpose
    op_count: int
    ops_per_graph: Optional[List[torch.fx.Node]] = None
    out_guards: Optional[List[_guards.Guard]] = None
    compile_times: Optional[str] = None

    def __str__(self):
        output = f"Graph Count: {self.graph_count}\n"
        output += f"Graph Break Count: {self.graph_break_count}\n"
        output += f"Op Count: {self.op_count}\n"

        output += "Break Reasons:\n"
        for idx, break_reason in enumerate(self.break_reasons):
            output += f"  Break Reason {idx+1}:\n"
            output += f"    Reason: {break_reason.reason}\n"
            output += "    User Stack:\n"
            for frame_summary in break_reason.user_stack:
                output += f"      {frame_summary}\n"

        if self.ops_per_graph is not None:
            output += "Ops per Graph:\n"
            for idx, ops in enumerate(self.ops_per_graph):
                output += f"  Ops {idx+1}:\n"
                for op in ops:
                    output += f"    {op}\n"

        if self.out_guards is not None:
            output += "Out Guards:\n"
            for i, guard in enumerate(self.out_guards):
                output += f"  Guard {i+1}:\n"
                output += f"    {str(guard)}"

        if self.compile_times is not None:
            output += f"Compile Times: {self.compile_times}\n"
        return output


def _explain_graph_detail(
    gm: torch.fx.GraphModule, graphs, op_count, ops_per_graph, break_reasons
):
    """
    This function is a utility which processes a torch.fx.GraphModule and
    accumulates information about its ops, graph breaks, and other details. It
    is intended to be used by the ExplainWithBackend class and
    `torch._dynamo.explain()` to provide details from Dynamo's graph capture.

    Parameters:
        gm (torch.fx.GraphModule): The GraphModule to be processed.
        graphs (list): A list that accumulates all the GraphModules processed.
        op_count (int): The total count of operations in all GraphModules processed so far.
        ops_per_graph (list): A list that accumulates the operations of each GraphModule.
        break_reasons (list): A list that accumulates the reasons for breaks in each GraphModule.

    Returns:
        tuple: A tuple containing the processed GraphModule, the updated lists of graphs,
               operations per graph, and break reasons, and the updated operation count.
    """
    graphs.append(gm)
    ops = [node.target for node in gm.graph.nodes if node.op == "call_function"]
    op_count += len(ops)
    ops_per_graph.append(ops)
    if gm.compile_subgraph_reason.graph_break:
        break_reasons.append(gm.compile_subgraph_reason)

    return gm, graphs, op_count, ops_per_graph, break_reasons


class ExplainWithBackend:
    """
    This class is intended to be used as a backend for `torch.compile`. It is
    composable with other backends. When used in this way, it accumulates
    information about graph breaks, ops, and other info and provides a string
    representation summarizing this information.

    Attributes:
        backend (str): The name of the backend to use for optimization.
        graphs (list): A list of the graphs captured by TorchDynamo.
        op_count (int): The total number of operations in all optimized graphs.
        break_reasons (list): A list of graph break reasons with stack traces.

    Example Usage:
        def fn(x):
            x = torch.sigmoid(x)
            return x

        torch._dynamo.reset()
        eb = ExplainWithBackend("inductor")
        optimized_fn = torch.compile(fn, backend=eb)
        result = optimized_fn(torch.randn(5))
        print(eb.output())
    """

    def __init__(self, backend):
        from .registry import lookup_backend

        self.backend = lookup_backend(backend)
        self.graphs = []
        self.op_count = 0
        self.break_reasons = []

    def __call__(self, gm: torch.fx.GraphModule, example_inputs):
        gm, self.graphs, self.op_count, _, self.break_reasons = _explain_graph_detail(
            gm, self.graphs, self.op_count, [], self.break_reasons
        )
        return self.backend(gm, example_inputs)

    def output(self) -> ExplainOutput:
        graph_count = len(self.graphs)
        output = ExplainOutput(
            self.graphs,
            graph_count,
            graph_count - 1,
            self.break_reasons,
            self.op_count,
        )

        return output
