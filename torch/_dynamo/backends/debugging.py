"""
This module provides debugging backends for TorchDynamo to help diagnose and troubleshoot
compilation and execution issues. It includes:

Key Debugging Backends:
- eager: Simple pass-through backend that runs models in eager mode
- eager_noexcept: Similar to eager but with additional exception handling
- eager_debug: Adds schema validation checks for custom operators
- aot_eager: Uses AOT Autograd with nop compiler for debugging
- aot_eager_decomp_partition: Uses TorchInductor decompositions for debugging
- torchscript: Compiles using TorchScript for debugging JIT-related issues

Testing and Development Tools:
- Backends for inducing specific errors (compile/runtime/accuracy)
- ExplainOutput class for detailed graph compilation analysis
- Utilities for cross-referencing and mode management
- Tools for graph detail inspection and break reason analysis

These backends are primarily used for:
1. Debugging graph breaks and compilation failures
2. Testing error handling and recovery mechanisms
3. Analyzing performance bottlenecks
4. Validating operator schemas and decompositions
"""

import dataclasses
import functools
import logging
from collections.abc import Iterable
from importlib import import_module
from typing import Any, Callable, Optional, TYPE_CHECKING, Union

import torch
from functorch.compile import min_cut_rematerialization_partition
from torch import _guards
from torch._dynamo.output_graph import GraphCompileReason
from torch._functorch import config as functorch_config
from torch._functorch.compilers import ts_compile

from .common import aot_autograd
from .registry import CompiledFn, CompilerFn, register_debug_backend as register_backend


if TYPE_CHECKING:
    from torch.fx.node import Target


log = logging.getLogger(__name__)


@register_backend
def eager(
    gm: torch.fx.GraphModule, fake_tensor_inputs: list[torch.Tensor], **kwargs: Any
) -> Callable[..., Any]:
    if kwargs:
        log.warning("eager backend ignoring extra kwargs %s", kwargs)
    return gm.forward


def make_eager_backend_with_torch_function_mode(
    mode: torch.overrides.TorchFunctionMode,
) -> Callable[..., Any]:
    return make_eager_backend_with_torch_function_modes([mode])


def make_eager_backend_with_torch_function_modes(
    modes: Iterable[torch.overrides.TorchFunctionMode],
) -> Callable[..., Any]:
    """Used to trace HOPs (cond and while) for eager execution, the metadata
    TF mode mutates vars outside of the scope of the HOP, and we can't have graph breaks
    in the HOP, so we need to externally run this mode and not trace it."""
    from contextlib import ExitStack

    def fn(
        gm: torch.fx.GraphModule, fake_tensor_inputs: list[torch.Tensor], **kwargs: Any
    ) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with ExitStack() as stack:
                for mode in modes:
                    stack.enter_context(mode)
                return gm.forward(*args, **kwargs)

        return wrapper

    return fn


@register_backend
def eager_noexcept(
    gm: torch.fx.GraphModule, fake_tensor_inputs: list[torch.Tensor], **kwargs: Any
) -> Callable[..., Any]:
    if kwargs:
        log.warning("eager_noexcept backend ignoring extra kwargs %s", kwargs)

    # This backend is intended to check that dynamo-generated GraphModules
    # do not cause errors.
    def inner(*args: Any) -> Any:
        try:
            return gm(*args)
        except Exception as e:
            raise torch._dynamo.exc.TorchDynamoException(
                "Unexpected exception when running generated GraphModule"
            ) from e

    return inner


@register_backend
def pre_dispatch_eager(
    gm: torch.fx.GraphModule, fake_tensor_inputs: list[torch.Tensor], **kwargs: Any
) -> torch.fx.GraphModule:
    if kwargs:
        log.warning("pre_dispatch_eager backend ignoring extra kwargs %s", kwargs)

    from torch.fx.experimental.proxy_tensor import make_fx

    def runnable_gm(*args: Any) -> Any:
        return torch.fx.Interpreter(gm).run(*args)

    pre_dispatch_gm = make_fx(runnable_gm, pre_dispatch=True)(*fake_tensor_inputs)
    pre_dispatch_gm.print_readable()

    return pre_dispatch_gm


@register_backend
def eager_debug(
    gm: torch.fx.GraphModule, fake_tensor_inputs: list[torch.Tensor], **kwargs: Any
) -> Callable[..., Any]:
    if kwargs:
        log.warning("eager_debug backend ignoring extra kwargs %s", kwargs)

    from torch._subclasses.schema_check_mode import SchemaCheckMode

    # We could add more debugging bits here.
    # Right now, this backend can be used to check for and error on
    # custom dispatcher ops that have incorrect schemas.
    def inner(*args: Any) -> Any:
        with SchemaCheckMode():
            return torch.fx.Interpreter(gm).run(*args)

    return inner


@register_backend(name="ts")  # type: ignore[misc]
def torchscript(
    gm: torch.fx.GraphModule, fake_tensor_inputs: list[torch.Tensor]
) -> torch.jit.ScriptModule:
    return torch.jit.script(gm)


# used boxed call to discard inputs when they are no longer needed
def boxed_nop(
    fx_g: torch.fx.GraphModule, example_inputs: list[torch.Tensor]
) -> Callable[..., Any]:
    from torch.fx.graph import _BoxedCodeGen

    # Set the graph to use boxed codegen
    fx_g.graph.set_codegen(_BoxedCodeGen())
    fx_g.recompile()

    # Wrap the forward method in a function so we can set _boxed_call attribute
    forward_fn = fx_g.forward

    def run(args: Any) -> Any:
        return forward_fn(args)

    run._boxed_call = True  # type: ignore[attr-defined]
    return run


def boxed_nop_with_mode(
    fx_g: torch.fx.GraphModule,
    example_inputs: list[torch.Tensor],
    *,
    mode: torch.overrides.TorchFunctionMode,
) -> Callable[..., Any]:
    from torch.fx.graph import _BoxedCodeGen

    # Set the graph to use boxed codegen
    fx_g.graph.set_codegen(_BoxedCodeGen())
    fx_g.recompile()

    # Create a wrapper that runs with the mode
    forward_fn = fx_g.forward

    def run(args: Any) -> Any:
        with mode:
            return forward_fn(args)

    run._boxed_call = True  # type: ignore[attr-defined]
    return run


def fake_crossref_boxed_nop(
    fx_g: torch.fx.GraphModule,
    example_inputs: list[torch.Tensor],
    ignore_op_fn: Optional[Callable[[torch._ops.OpOverload], bool]] = None,
) -> Callable[..., Any]:
    from torch.fx.graph import _BoxedCodeGen

    # Set the graph to use boxed codegen
    fx_g.graph.set_codegen(_BoxedCodeGen())
    fx_g.recompile()

    # Create a wrapper that runs with the mode
    forward_fn = fx_g.forward

    def run(args: Any) -> Any:
        with torch._subclasses.CrossRefFakeMode(ignore_op_fn):
            return forward_fn(args)

    run._boxed_call = True  # type: ignore[attr-defined]
    return run


def ignore_builtins(op: torch._ops.OpOverload) -> bool:
    return op.namespace in ("aten", "prims", "prim")


def get_nop_func() -> Callable[
    [torch.fx.GraphModule, list[torch.Tensor]], Callable[..., Any]
]:
    if not torch._functorch.config.fake_tensor_crossref:
        return boxed_nop
    elif torch._functorch.config.fake_tensor_crossref == "all":
        return fake_crossref_boxed_nop
    else:
        assert torch._functorch.config.fake_tensor_crossref == "custom_ops"
        return functools.partial(fake_crossref_boxed_nop, ignore_op_fn=ignore_builtins)


# Useful for debugging purpose
# aot_eager uses AOT Autograd backend with nop compiler. It is helpful in debugging.
def aot_eager(
    gm: torch.fx.GraphModule,
    fake_tensor_inputs: list[torch.Tensor],
    fw_compiler: Optional[Callable[..., Any]] = None,
    bw_compiler: Optional[Callable[..., Any]] = None,
    **kwargs: Any,
) -> Callable[..., Any]:
    return aot_autograd(
        fw_compiler=fw_compiler or boxed_nop,
        bw_compiler=bw_compiler or boxed_nop,
        partition_fn=min_cut_rematerialization_partition,
        keep_inference_input_mutations=True,
    )(gm, fake_tensor_inputs, **kwargs)


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
def aot_eager_decomp_partition(
    gm: torch.fx.GraphModule, fake_tensor_inputs: list[torch.Tensor], **kwargs: Any
) -> Callable[..., Any]:
    if kwargs:
        log.warning(
            "aot_eager_decomp_partition backend ignoring extra kwargs %s", kwargs
        )

    from torch._inductor.compiler_bisector import CompilerBisector

    config_patches = {"unlift_effect_tokens": True}
    if bisect_changes := CompilerBisector.get_config_change(
        "aot_eager_decomp_partition"
    ):
        config_patches.update(bisect_changes)  # type: ignore[arg-type]

    with functorch_config.patch(config_patches):
        return aot_autograd(
            # these are taken from memory_efficient_fusion()
            fw_compiler=get_nop_func(),
            bw_compiler=get_nop_func(),
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


# aot_eager_decomp_partition_with_mode is similar as aot_eager_decomp_partition,
# except that it takes a TorchDispatchMode mode and run the fw/bw in the mode
def aot_eager_decomp_partition_with_mode(
    gm: torch.fx.GraphModule,
    fake_tensor_inputs: list[torch.Tensor],
    mode: Any,
    **kwarg: Any,
) -> Callable[..., Any]:
    return aot_autograd(
        # these are taken from memory_efficient_fusion()
        fw_compiler=functools.partial(boxed_nop_with_mode, mode=mode),
        bw_compiler=functools.partial(boxed_nop_with_mode, mode=mode),
        # NB: lambda here is to delay import of inductor
        decompositions=lambda: import_module(
            "torch._inductor.compile_fx"
        ).select_decomp_table(),
        partition_fn=functools.partial(
            min_cut_rematerialization_partition, compiler="inductor"
        ),
    )(gm, fake_tensor_inputs)


register_backend(
    name="aot_eager_decomp_partition_with_mode",
    compiler_fn=aot_eager_decomp_partition_with_mode,  # type: ignore[arg-type]
)


def aot_eager_decomp_partition_crossref(
    gm: torch.fx.GraphModule, fake_tensor_inputs: list[torch.Tensor], **kwargs: Any
) -> Callable[..., Any]:
    # if the config is set, respect it, otherwise only test custom_ops.
    # custom_op bad metas always manifest as an error whereas aten will only sometimes.
    # by default, use the less noisy option
    config_val = (
        "custom_ops"
        if not functorch_config.fake_tensor_crossref
        else functorch_config.fake_tensor_crossref
    )
    with functorch_config.patch(fake_tensor_crossref=config_val):
        return aot_eager_decomp_partition(gm, fake_tensor_inputs, **kwargs)


register_backend(
    name="aot_eager_decomp_partition_crossref",
    compiler_fn=aot_eager_decomp_partition_crossref,
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
def relu_compile_error_TESTING_ONLY(
    gm: torch.fx.GraphModule, example_inputs: list[torch.Tensor]
) -> torch.fx.GraphModule:
    for node in gm.graph.nodes:
        if node.target == torch.relu:
            raise ReluCompileError
    return gm


@register_backend
def relu_runtime_error_TESTING_ONLY(
    gm: torch.fx.GraphModule, example_inputs: list[torch.Tensor]
) -> torch.fx.GraphModule:
    for node in gm.graph.nodes:
        if node.target == torch.relu:
            node.target = torch._assert
            node.args = (False, "ReluRuntimeError")
    gm.recompile()
    return gm


@register_backend
def relu_accuracy_error_TESTING_ONLY(
    gm: torch.fx.GraphModule, example_inputs: list[torch.Tensor]
) -> torch.fx.GraphModule:
    for node in gm.graph.nodes:
        if node.target == torch.relu:
            node.target = torch.add
            node.args = (node.args[0], 1)
    gm.recompile()

    return gm


@register_backend
def non_leaf_compile_error_TESTING_ONLY(
    gm: torch.fx.GraphModule, example_inputs: list[torch.Tensor]
) -> torch.fx.GraphModule:
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

    graphs: list[torch.fx.GraphModule]
    graph_count: int
    graph_break_count: int
    break_reasons: list[GraphCompileReason]
    op_count: int
    ops_per_graph: Optional[list[list["Target"]]] = None
    out_guards: Optional[list[_guards.Guard]] = None
    compile_times: Optional[str] = None

    def __str__(self) -> str:
        output = f"Graph Count: {self.graph_count}\n"
        output += f"Graph Break Count: {self.graph_break_count}\n"
        output += f"Op Count: {self.op_count}\n"

        output += "Break Reasons:\n"
        for idx, break_reason in enumerate(self.break_reasons):
            output += f"  Break Reason {idx + 1}:\n"
            output += f"    Reason: {break_reason.reason}\n"
            output += "    User Stack:\n"
            for frame_summary in break_reason.user_stack:
                output += f"      {frame_summary}\n"

        if self.ops_per_graph is not None:
            output += "Ops per Graph:\n"
            for idx, ops in enumerate(self.ops_per_graph):
                output += f"  Ops {idx + 1}:\n"
                for op in ops:
                    output += f"    {op}\n"

        if self.out_guards is not None:
            output += "Out Guards:\n"
            for i, guard in enumerate(self.out_guards):
                output += f"  Guard {i + 1}:\n"
                output += f"    {str(guard)}"

        if self.compile_times is not None:
            output += f"Compile Times: {self.compile_times}\n"
        return output


def _explain_graph_detail(
    gm: torch.fx.GraphModule,
    graphs: list[torch.fx.GraphModule],
    op_count: int,
    ops_per_graph: list[list["Target"]],
    break_reasons: list[GraphCompileReason],
) -> tuple[
    torch.fx.GraphModule,
    list[torch.fx.GraphModule],
    int,
    list[list["Target"]],
    list[GraphCompileReason],
]:
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
    if gm.compile_subgraph_reason.graph_break:  # type: ignore[union-attr]
        break_reasons.append(gm.compile_subgraph_reason)  # type: ignore[arg-type]

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

    def __init__(self, backend: Union[CompilerFn, str]) -> None:
        from .registry import lookup_backend

        self.backend = lookup_backend(backend)
        self.graphs: list[torch.fx.GraphModule] = []
        self.op_count = 0
        self.break_reasons: list[GraphCompileReason] = []

    def __call__(
        self, gm: torch.fx.GraphModule, example_inputs: list[torch.Tensor]
    ) -> CompiledFn:
        ops_per_graph: list[list[Target]] = []
        gm, self.graphs, self.op_count, _, self.break_reasons = _explain_graph_detail(
            gm, self.graphs, self.op_count, ops_per_graph, self.break_reasons
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
