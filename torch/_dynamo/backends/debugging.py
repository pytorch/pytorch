# mypy: ignore-errors

import dataclasses
import functools
import itertools
import logging
import os
import traceback
from importlib import import_module
from typing import Any, Callable, List, Optional, Type

import torch
import torch.utils._pytree as pytree
from functorch.compile import min_cut_rematerialization_partition
from torch import _guards, Tensor
from torch._dynamo.source import LocalSource
from torch._dynamo.variables.builder import GraphArg
from torch._functorch import config as functorch_config
from torch._functorch.compilers import ts_compile
from torch.utils.weak import TensorWeakRef

from .common import aot_autograd
from .registry import register_debug_backend as register_backend


log = logging.getLogger(__name__)


"""
This file contains TorchDynamo backends intended for debugging uses.
"""


@register_backend
def eager(gm, fake_tensor_inputs, **kwargs):
    if kwargs:
        log.warning("eager backend ignoring extra kwargs %s", kwargs)
    return gm.forward


def make_eager_backend_with_torch_function_mode(mode):
    return make_eager_backend_with_torch_function_modes([mode])


def make_eager_backend_with_torch_function_modes(modes):
    """Used to trace HOPs (cond and while) for eager exectution, the metadata
    TF mode mutates vars outside of the scope of the HOP, and we can't have graph breaks
    in the HOP, so we need to externally run this mode and not trace it."""
    from contextlib import ExitStack

    def fn(gm, fake_tensor_inputs, **kwargs):
        stack = ExitStack()
        for mode in modes:
            stack.enter_context(mode)

        result = gm.forward
        stack.close()
        return result

    return fn


@register_backend
def eager_noexcept(gm, fake_tensor_inputs, **kwargs):
    if kwargs:
        log.warning("eager_noexcept backend ignoring extra kwargs %s", kwargs)

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
def pre_dispatch_eager(gm, fake_tensor_inputs, **kwargs):
    if kwargs:
        log.warning("pre_dispatch_eager backend ignoring extra kwargs %s", kwargs)

    from torch.fx.experimental.proxy_tensor import make_fx

    def runnable_gm(*args):
        return torch.fx.Interpreter(gm).run(*args)

    pre_dispatch_gm = make_fx(runnable_gm, pre_dispatch=True)(*fake_tensor_inputs)
    pre_dispatch_gm.print_readable()

    return pre_dispatch_gm


@register_backend
def eager_debug(gm, fake_tensor_inputs, **kwargs):
    if kwargs:
        log.warning("eager_debug backend ignoring extra kwargs %s", kwargs)

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


def fake_crossref_boxed_nop(fx_g, example_inputs, ignore_op_fn=None):
    def run(args):
        with torch._subclasses.CrossRefFakeMode(ignore_op_fn):
            return torch.fx.Interpreter(fx_g).boxed_run(args)

    run._boxed_call = True
    return run


def ignore_builtins(op: torch._ops.OpOverload) -> bool:
    return op.namespace in ("aten", "prims", "prim")


def get_nop_func():
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
    gm,
    fake_tensor_inputs,
    fw_compiler=None,
    bw_compiler=None,
    **kwargs,
):
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
def aot_eager_decomp_partition(gm, fake_tensor_inputs, **kwargs):
    if kwargs:
        log.warning(
            "aot_eager_decomp_partition backend ignoring extra kwargs %s", kwargs
        )

    from torch._inductor.compiler_bisector import CompilerBisector

    config_patches = {"unlift_effect_tokens": True}
    if bisect_changes := CompilerBisector.get_config_change(
        "aot_eager_decomp_partition"
    ):
        config_patches.update(bisect_changes)

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


def aot_eager_decomp_partition_crossref(gm, fake_tensor_inputs, **kwargs):
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


def _try_lift_tensor_arguments(gm, inputs):
    placeholders = {}
    i = 0
    added_inputs = []
    for n in gm.graph.nodes:
        if n.op == "placeholder":
            placeholders[n] = inputs[i]
            i += 1
        elif n.op == "call_function":
            if len(n.kwargs) != 0:
                continue
            node_args = [a for a in n.args if isinstance(a, torch.fx.Node)]
            other_args = [a for a in n.args if not isinstance(a, torch.fx.Node)]

            is_output_tensor = isinstance(n.meta["example_value"], Tensor)

            if (
                is_output_tensor
                and all(
                    (a in placeholders and isinstance(a, (bool, int, float)))
                    for a in node_args
                )
                and all(isinstance(a, (bool, int, float)) for a in other_args)
            ):
                real_args = [placeholders.get(a, a) for a in n.args]
                new_real_arg = n.target(*real_args)

                with gm.graph.inserting_before(n):
                    name = f"lifted_arg_{len(added_inputs)}"
                    new_input_node = gm.graph.placeholder(name, type_expr=Tensor)
                    added_inputs.append(new_real_arg)
                    new_input_node.meta["grapharg"] = GraphArg(
                        source=LocalSource(name),
                        _example=TensorWeakRef(new_real_arg),
                        pass_arg_as_tensor=False,
                        fake_tensor=None,
                        is_tensor=True,
                        example_strong_ref=new_real_arg,
                    )
                    n.replace_all_uses_with(new_input_node)

    gm.graph.eliminate_dead_code()
    gm.recompile()

    new_inputs = [*inputs, *added_inputs]

    log.debug(
        "test_subclasses _try_lift_tensor_arguments graph after lifting: inputs:%s result:%s",
        new_inputs,
        gm.print_readable(False),
    )
    return gm, new_inputs


@dataclasses.dataclass
class TensorToSubclassTransform:
    factory_fn: Callable[[Tensor], Type]
    precondition: Optional[Callable[[Tensor], bool]] = None

    def check_precondition(self, t):
        return self.precondition is None or self.precondition(t)


@register_backend
@torch._functorch.config.patch("enable_autograd_cache", False)
def test_subclasses(gm, inputs, **kwargs):
    from torch._subclasses.functional_tensor import FunctionalTensor

    if any(isinstance(inp, FunctionalTensor) for inp in inputs):
        return gm

    if kwargs:
        log.warning("test_subclasses backend ignoring extra kwargs %s", kwargs)

    log.debug(
        "test_subclasses backend call inputs:%s gm:%s", inputs, gm.print_readable(False)
    )

    import copy

    test_gm = copy.deepcopy(gm)

    # Verify original inputs
    aot_eager(test_gm, inputs)

    from torch.testing._internal.subclasses import (
        F32_QI32QuantRWTensor,
        WrapperSubclass,
    )
    from torch.testing._internal.two_tensor import TwoTensor

    TRANSFORMATIONS: List[TensorToSubclassTransform] = [
        TensorToSubclassTransform(
            factory_fn=lambda t: WrapperSubclass(t),
        ),
        TensorToSubclassTransform(
            factory_fn=lambda t: TwoTensor(t, t),
        ),
    ]
    if bool(os.getenv("PYTORCH_TEST_WITH_SUBCLASSES_NONTRIVIAL", default=0)):
        TRANSFORMATIONS.extend(
            [
                TensorToSubclassTransform(
                    factory_fn=lambda t: F32_QI32QuantRWTensor.from_src(t),
                    precondition=lambda t: t.ndim <= 2,
                ),
                # TODO: NestedTensor transformation can have many false-positive failures
                # as NT does not support many of the operations
                TensorToSubclassTransform(
                    factory_fn=lambda t: torch.nested.nested_tensor_from_jagged(
                        t, offsets=torch.tensor([0, t.size(1)])
                    ),
                    precondition=lambda t: t.ndim >= 2,
                ),
                # TODO(ivankobzarev): DTensor
            ],
        )

    def _is_tensor(t):
        return isinstance(t, Tensor)

    if not any(_is_tensor(t) for t in inputs):
        # Try to find tensor creations with inputs and lift as arguments
        test_gm, test_inputs = _try_lift_tensor_arguments(test_gm, inputs)

        if not any(_is_tensor(t) for t in inputs):
            log.debug("No tensor inputs")
            return gm

    MAX_SUBCLASSES_NESTING: int = int(
        os.getenv("PYTORCH_TEST_WITH_SUBCLASSES_MAX_NESTING", default=1)
    )
    N: int = len(TRANSFORMATIONS)

    TRANSFORM_SEQS: List[Any] = [
        (),  # empty tuple means no transformation
    ]
    for k in range(1, MAX_SUBCLASSES_NESTING + 1):
        for p in itertools.product(list(range(N)), repeat=k):
            TRANSFORM_SEQS.append(p)  # noqa: PERF402

    TENSOR_INPUTS_IDXS: List[int] = [
        i for i, inp in enumerate(inputs) if _is_tensor(inp)
    ]
    NUM_TENSOR_INPUTS = len(TENSOR_INPUTS_IDXS)

    TENSOR_INPUTS_TRANSFORM_SEQS = list(
        itertools.product(TRANSFORM_SEQS, repeat=NUM_TENSOR_INPUTS)
    )
    NUM_TENSOR_INPUTS_TRANSFORM_SEQS = len(TENSOR_INPUTS_TRANSFORM_SEQS)
    log.debug(
        "test_subclasses backend TENSOR_INPUTS_TRANSFORM_SEQS:%s",
        TENSOR_INPUTS_TRANSFORM_SEQS,
    )

    def apply_transform_seq(transform_seq, inp):
        ret = inp
        for transform_idx in transform_seq:
            transform = TRANSFORMATIONS[transform_idx]
            if transform.check_precondition(ret):
                ret = transform.factory_fn(ret)

        return ret

    log.info(
        "test_subclasses backend testing %d transformed inputs gm:%s",
        NUM_TENSOR_INPUTS_TRANSFORM_SEQS,
        test_gm.print_readable(False),
    )
    for i, transform_seqs in enumerate(TENSOR_INPUTS_TRANSFORM_SEQS):
        test_inputs = pytree.tree_map(lambda x: x.detach(), copy.copy(inputs))
        # Have to copy GraphModule, as AOTD caches some info based on inputs in attrs
        _test_gm = copy.deepcopy(test_gm)

        for seq_idx, idx in enumerate(TENSOR_INPUTS_IDXS):
            test_inputs[idx] = apply_transform_seq(
                transform_seqs[seq_idx], test_inputs[idx]
            )

        try:
            aot_eager(_test_gm, test_inputs)
            log.info(
                "test_subclasses backend testing %d/%d transformed inputs:%s OK",
                i,
                NUM_TENSOR_INPUTS_TRANSFORM_SEQS,
                test_inputs,
            )
        except Exception as ex:
            # TODO: Print script with graph and inputs for repro
            repro_py = f"""
import torch
from torch import tensor
from torch._dynamo.backends.debugging import aot_eager
from torch.testing._internal.subclasses import (
    F32_QI32QuantRWTensor,
    WrapperSubclass,
)
from torch.testing._internal.two_tensor import TwoTensor
from torch.nested._internal.nested_tensor import NestedTensor

inputs = {[ti.detach() if _is_tensor(ti) else ti for ti in test_inputs]}
{gm.print_readable(False)}
gm = GraphModule()
aot_eager(gm, inputs)"""

            log.error(
                "test_subclasses error compiling\ninputs:%s\ntransform_seqs:%s\n---REPRO_BEGIN---%s\n---REPRO_END---\nexception:%s",
                test_inputs,
                transform_seqs,
                repro_py,
                "".join(traceback.format_exception(type(ex), ex, ex.__traceback__)),
            )
            raise ex

    return gm


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

    def __str__(self) -> str:
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

    def __init__(self, backend) -> None:
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
