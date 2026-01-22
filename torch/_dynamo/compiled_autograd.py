"""
Provides functionality for compiling PyTorch's autograd (automatic differentiation) system.

This module implements compiled autograd, which traces and optimizes backward pass
computations at runtime. The key components are:

- AutogradCompilerInstance: Traces and compiles autograd graphs using FX
- Context managers (_enable/_disable): Control when compiled autograd is active
- Utility functions: Support graph manipulation, tensor operations, and hooks

Compiled autograd can significantly improve backward pass performance by removing
Python overhead and enabling additional optimizations. It works by capturing
backward computations into an FX graph that can be compiled and optimized,
while maintaining the same semantics as eager mode autograd.
"""

import contextlib
import functools
import itertools
import operator
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Generator, Sequence
from typing import Any, Optional, TYPE_CHECKING, Union

import torch
import torch.utils._pytree as pytree
from torch._dispatch.python import enable_python_dispatcher
from torch._dynamo.external_utils import (
    call_accumulate_grad,
    call_backward,
    call_hook,
    FakeCompiledAutogradEngine,
    unwrap_maybe_dynamic_int,
)
from torch._dynamo.source import GetItemSource, LocalSource
from torch._dynamo.utils import (
    counters,
    get_chromium_event_logger,
    lazy_format_graph_code,
    set_locals_to_steal,
)
from torch._functorch._aot_autograd.runtime_wrappers import (
    AutogradLazyBackwardCompileInfo,
    CachedAutogradLazyBackwardCompileInfo,
)
from torch._guards import compile_context, CompileContext, CompileId, Source
from torch._logging import getArtifactLogger, trace_structured
from torch._prims_common import clone_preserve_strides
from torch._subclasses import FakeTensorMode
from torch._subclasses.fake_tensor import FakeTensor
from torch.fx import GraphModule
from torch.fx.experimental._backward_state import BackwardState
from torch.fx.experimental.proxy_tensor import (
    decompose,
    disable_autocast_cache,
    disable_proxy_modes_tracing,
    fetch_object_proxy,
    ProxyTorchDispatchMode,
    PythonKeyTracer,
    track_tensor_tree,
)
from torch.fx.experimental.symbolic_shapes import DimDynamic, ShapeEnv
from torch.fx.traceback import preserve_node_meta, set_stack_trace
from torch.types import FloatLikeType, IntLikeType
from torch.utils._ordered_set import OrderedSet
from torch.utils._traceback import CapturedTraceback


if TYPE_CHECKING:
    from torch.fx.proxy import Proxy


TURN_OFF_MSG = """You can turn off compiled autograd by either:
1. Moving the unsupported autograd call outside of the torch.compile'd region.
2. Wrapping the unsupported autograd call in the torch._dynamo.compiled_autograd._disable() context manager.
3. Setting torch._dynamo.config.compiled_autograd=False for the torch.compile call containing the unsupported autograd call.
4. Setting torch._dynamo.config.compiled_autograd=False at the start of the program."""

compiled_autograd_log = getArtifactLogger(__name__, "compiled_autograd")
verbose_log = getArtifactLogger(__name__, "compiled_autograd_verbose")


def snapshot_verbose_logging_enabled() -> bool:
    return torch._logging._internal.log_state.is_artifact_enabled(
        "compiled_autograd_verbose"
    )


def snapshot_cudagraph_enabled() -> bool:
    return torch._inductor.config.triton.cudagraphs


def maybe_clone(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if x is not None:
        return clone_preserve_strides(x)
    return x


def extract_bw_module(CompiledFunction: Any) -> Callable[..., Any]:
    if isinstance(
        CompiledFunction._lazy_backward_info, AutogradLazyBackwardCompileInfo
    ):
        return CompiledFunction._lazy_backward_info.bw_module
    elif isinstance(
        CompiledFunction._lazy_backward_info, CachedAutogradLazyBackwardCompileInfo
    ):
        with torch._subclasses.fake_tensor.unset_fake_temporarily():
            return CompiledFunction._lazy_backward_info.bw_module_fn()
    else:
        raise AssertionError(
            "Unexpected Lazy Backward Compilation Info Type. Please file an issue."
        )


# Note: [Anomaly Mode Semantics in Compiled Autograd]
# In the eager autograd engine, anomaly mode is able to detect NaNs
# after each node. This is useful, because the executed code with
# and without anomaly mode are the same. So assuming determinism,
# a NaN in regular mode should also happen in anomaly mode.
#
# With torch.compile, following eager semantics would require inserting
# runtime asserts to check for NaNs, which could prevent some fusions.
# This results in different code being run with and without anomaly mode.
# So different semantics are needed, this implementation below will check
# for NaNs at the end of the autograd call, instead of after each node
class NaNChecker:
    def __init__(self, accumulate_grad: bool) -> None:
        self.accumulate_grad = accumulate_grad
        self.params_indices: list[int] = []
        self.params_to_check: dict[str, torch.Tensor] = {}
        self.output_names: list[str] = []

    def prep_with_graph(self, graph: torch.fx.Graph) -> None:
        inputs_node = next(iter(graph.nodes))
        acc_grad_nodes = graph.find_nodes(
            op="call_function", target=call_accumulate_grad
        )
        output_nodes = graph.find_nodes(op="output")[0].args[0]
        assert self.accumulate_grad == bool(
            acc_grad_nodes
        ) and self.accumulate_grad == (not output_nodes)

        for node in acc_grad_nodes:
            param_node = node.args[0]
            # AccumulateGrad always saves a reference to the param
            # so Compiled Autograd will always lift the param and
            # this should always be true
            assert (
                param_node.target is operator.getitem
                and param_node.args[0] is inputs_node  # type: ignore[possibly-undefined]
                and isinstance(param_node.args[1], int)
            )
            self.params_indices.append(param_node.args[1])

        self.output_names = [node.name for node in output_nodes]

    def prep_with_inputs(self, inputs: tuple[torch.Tensor, ...]) -> None:
        if not self.accumulate_grad:
            # Using .grad, nothing to prep
            return

        # Using .backward, we must check existing grads on params if any
        for idx in self.params_indices:
            grad = inputs[idx].grad
            if grad is not None:
                assert not torch.isnan(grad).any(), (
                    f"Compiled autograd running under anomaly mode with inputs[{idx}] already "
                    f"having NaN gradient. This is not supported. {TURN_OFF_MSG}"
                )

            self.params_to_check[f"inputs[{idx}]"] = inputs[idx]

    def check(self, out: tuple[torch.Tensor, ...]) -> None:
        if self.accumulate_grad:
            # Using .backward, graph outputs are empty
            assert not out
            nan_params: list[str] = []
            for inputs_str, param in self.params_to_check.items():
                assert param.grad is not None  # not true for autograd.grad
                if torch.isnan(param.grad).any():
                    nan_params.append(inputs_str)

            if nan_params:
                raise RuntimeError(
                    f"Compiled Autograd returned NaN gradients for parameters: {','.join(nan_params)}."
                )
        else:
            # Using .grad, graph outputs are grads
            nan_grads: list[str] = []
            for i, grad in enumerate(out):
                if torch.isnan(grad).any():
                    nan_grads.append(self.output_names[i])

            if nan_grads:
                raise RuntimeError(
                    f"Compiled Autograd returned NaN gradients for output nodes: {','.join(nan_grads)}."
                )


# We lazily bind "functional backward" variants for PyTorch built-in autograd
# nodes to this class. Example: torch._dynamo.compiled_autograd.ops.MulBackward0
# Each "functional backward" is bound the first time the node's apply_with_saved
# function is called. It's possible to avoid lazy binding and instead bind
# all of this upfront (perhaps at import time) via codegen changes.
class OpNamespace:
    def __init__(self) -> None:
        self.custom_function_name_counter: Counter[str] = Counter()

    def add(
        self,
        name: str,
        fn: Callable[..., Any],
        is_custom_function: bool,
        is_traceable: bool,
    ) -> str:
        if is_custom_function:
            name = "CppNode" + name
            count = self.custom_function_name_counter[name]
            self.custom_function_name_counter[name] += 1
            name = f"{name}{count}"

        assert not hasattr(self, name)
        result = Op(name, fn, is_custom_function)
        if is_traceable:
            setattr(self, name, torch._dynamo.allow_in_graph(result))
        else:
            # C++ autograd function was not marked as traceable
            # Dynamo can't dry run it at compile time, so must fallback to eager
            @torch._dynamo.disable  # type: ignore[misc]
            def run_non_traceable_cpp_in_eager(*args: Any, **kwargs: Any) -> Any:
                return result(*args, **kwargs)

            setattr(self, name, run_non_traceable_cpp_in_eager)
        return name

    def get(self, name: str) -> Any:
        return getattr(self, name)


class Op:
    def __init__(
        self, name: str, fn: Callable[..., Any], is_custom_function: bool
    ) -> None:
        self.fn = fn
        self.is_custom_function = is_custom_function
        self.__name__ = name
        self.__module__ = "torch._dynamo.compiled_autograd.ops"

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.fn(*args, **kwargs)

    def __repr__(self) -> str:
        return self.__module__ + "." + self.__name__


ops = OpNamespace()


_graph_placeholders = ["inputs", "sizes", "scalars", "hooks", "packed_data"]
_impure_targets = OrderedSet(
    [
        call_hook,
        call_backward,
        FakeCompiledAutogradEngine._exec_final_callbacks_stub,
        call_accumulate_grad,
    ]
)

COMPILE_COUNTER = itertools.count()


def make_compile_context(compiled_autograd_id: int) -> Any:
    return compile_context(
        CompileContext(
            CompileId(
                compiled_autograd_id=compiled_autograd_id,
                frame_id=None,
                frame_compile_id=None,
            )
        )
    )


class AutogradCompilerInstance:
    def __init__(self, compiler_fn: Callable[..., Any]) -> None:
        self.compiler_fn = compiler_fn
        self.stack = contextlib.ExitStack()
        self.close = self.stack.close
        self.shape_env = ShapeEnv()
        self.fake_tensor_mode = FakeTensorMode(
            allow_fallback_kernels=True,
            allow_non_fake_inputs=True,
            shape_env=self.shape_env,
        )
        self.fx_tracer = PythonKeyTracer()
        self.proxy_mode = ProxyTorchDispatchMode(self.fx_tracer, "symbolic")
        self.hooks_proxy: Optional[Proxy] = None

    def wrap_fake(self, x: torch.Tensor, source: Optional[Source]) -> FakeTensor:
        assert isinstance(x, torch.Tensor)
        return self.fake_tensor_mode.from_tensor(x, source=source)

    @staticmethod
    def source(name: str, idx: Any) -> GetItemSource:
        return GetItemSource(LocalSource(name), idx)

    def begin_capture(
        self,
        inputs: list[torch.Tensor],
        sizes: list[int],
        scalars: list[Union[int, float]],
        origins: list[list[tuple[int, str]]],
        accumulate_grad: bool,
        check_nans: bool,
    ) -> tuple[str, list[torch.Tensor], list[IntLikeType], list[FloatLikeType]]:
        counters["compiled_autograd"]["captures"] += 1
        self.id = next(COMPILE_COUNTER)
        self.aot_id_counter: dict[int, int] = defaultdict(int)
        self.compile_context = make_compile_context(self.id)
        self.compile_context.__enter__()
        self.nan_checker = NaNChecker(accumulate_grad) if check_nans else None
        self.start_time_ns = time.time_ns()
        get_chromium_event_logger().log_event_start(
            "compiled_autograd",
            self.start_time_ns,
            {"graph_id": self.id},
            log_pt2_compile_event=True,
        )
        self.fx_tracer.root = torch.nn.Module()
        self.fx_tracer.graph = torch.fx.Graph(tracer_cls=PythonKeyTracer)
        self.fx_tracer.tensor_attrs = {}
        self.symnode_proxy_lookup = {}
        (
            args_proxy,
            self.sizes_proxy,
            self.scalars_proxy,
            self.hooks_proxy,
            self.packed_data_proxy,
        ) = (
            self.fx_tracer.create_proxy("placeholder", name, (), {})
            for name in _graph_placeholders
        )

        self.stack.enter_context(preserve_node_meta())
        inputs_origins, sizes_origins, scalars_origins = origins

        # Turn on PythonDispatcher during initial trace to make it identifiable
        # that tracing is happening, which is needed to prevent hashing symints
        self.stack.enter_context(enable_python_dispatcher())

        # tensor inputs to fake tensors
        x = inputs[0]  # mypy will complain about unbound x
        try:
            for idx, x in enumerate(inputs):
                inputs[idx] = self.wrap_fake(x, self.source("inputs", idx))
        except Exception as e:
            raise NotImplementedError(
                f"Found tensor of type {type(x)}, which is not supported by FakeTensorMode. {TURN_OFF_MSG}"
            ) from e
        self.bind_objects_to_proxies(inputs, args_proxy, inputs_origins)

        # size inputs to symints
        sym_sizes = [
            self.shape_env.create_unspecified_symint_and_symbol(
                val,
                self.source("sizes", idx),
                DimDynamic.DYNAMIC,
            )
            for idx, val in enumerate(sizes)
        ]

        # We want to mark every size as dynamic, but since there's no way to
        # mark a primitive `int` as dynamic, we need to wrap it in a tensor.
        # In the graph, we unwrap it with `unwrap_maybe_dynamic_int` back into a primitive.
        proxies = [self.sizes_proxy[i] for i in range(len(sym_sizes))]  # type: ignore[index]
        for i, symint in enumerate(sym_sizes):
            proxies[i] = self.fx_tracer.create_proxy(
                "call_function",
                unwrap_maybe_dynamic_int,
                (proxies[i],),
                {},
            )
            self.symnode_proxy_lookup[symint.node] = proxies[i]
        proxies = self.bind_objects_to_proxies(sym_sizes, proxies, sizes_origins)

        for idx, val in enumerate(scalars):
            source = self.source("scalars", idx)
            if isinstance(val, int):
                scalars[idx] = self.shape_env.create_unspecified_symint_and_symbol(
                    val,
                    source,
                    DimDynamic.DYNAMIC,
                )
            elif isinstance(val, float):
                scalars[idx] = self.shape_env.create_symfloatnode(
                    self.shape_env.create_unspecified_symbol(
                        val,
                        source=source,
                        dynamic_dim=DimDynamic.DYNAMIC,
                    ),
                    hint=val,
                    source=source,
                )
            else:
                raise AssertionError("Unexpected scalar type: ", type(val))
        self.bind_objects_to_proxies(scalars, self.scalars_proxy, scalars_origins)
        for i, symval in enumerate(scalars):
            self.symnode_proxy_lookup[symval.node] = self.scalars_proxy[i]  # type: ignore[union-attr]

        # TODO(jansel): are all these modes needed?
        self.stack.enter_context(decompose({}))
        self.stack.enter_context(self.fake_tensor_mode)
        self.stack.enter_context(self.proxy_mode)
        self.stack.enter_context(disable_autocast_cache())
        # Needed to make sure we don't accidentally specialize any symbols
        assert self.fake_tensor_mode.shape_env is not None
        env = self.fake_tensor_mode.shape_env
        self.stack.enter_context(
            torch.fx.experimental.symbolic_shapes._suppress_guards(env)
        )
        # pyrefly: ignore [bad-return]
        return (
            str(CompileContext.current_compile_id()),
            inputs,
            sym_sizes,
            scalars,  # type: ignore[return-value]
        )

    def log_compile_reasons(
        self,
        compile_reasons: list[str],
    ) -> None:
        assert compile_reasons
        trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": "compiled_autograd_compile_reasons",
                "encoding": "json",
            },
            payload_fn=lambda: compile_reasons,
        )

    def proxy_call_aot_backward(
        self,
        pinputs: Sequence[Any],
        psaved_tensors: Sequence[torch.Tensor],
        saved_tensors: Sequence[torch.Tensor],
        pctx: Any,
        ctx: Any,
        maybe_backward_state_idx: Optional[int],
        opaque_object_indices: list[int],
    ) -> Sequence[Any]:
        # The AOTBackward call consists of three things: the prologue, the
        # backward graph, and the epilogue.
        # Our strategy is:
        # - allow_in_graph the prologue (in the CA graph and Dynamo graph),
        # - copy-paste the backward graph into the CA graph so that CA passes and Dynamo can see it
        # - trace directly through the epilogue. Anything that gets baked in is
        #   constant metadata (for example, metadata about the number of outputs, or removing
        #   RNG arguments or effect tokens).
        # If Dynamo graph capture were better, then we could add a node for the prologue
        # into the CA graph and have Dynamo trace into it.

        psymints = [self.to_proxy(e) for e in ctx._get_compiled_autograd_symints()]

        popaque_objects = [
            self.hooks_proxy[idx]  # type: ignore[index]
            for idx in opaque_object_indices
        ]

        # NOTE: we should only close over constants
        CompiledFunction = ctx._forward_cls
        bw_module = extract_bw_module(CompiledFunction)
        metadata = CompiledFunction.metadata
        maybe_subclass_metadata = CompiledFunction.maybe_subclass_metadata
        aot_id = CompiledFunction._aot_id
        del CompiledFunction

        if torch.is_grad_enabled():
            for output_alias_info in metadata.output_info:
                if output_alias_info.requires_grad:
                    raise RuntimeError(
                        "torch.compile does not currently support higher order gradients."
                    )

        @torch._dynamo.allow_in_graph  # type: ignore[misc]
        def call_aot_bwd_prologue(
            ctx_saved_tensors: Sequence[torch.Tensor],
            ctx_symints: Sequence[IntLikeType],
            ctx_opaque_objs: Sequence[Any],
            *flat_args: Sequence[Any],
        ) -> Any:
            out = torch._functorch._aot_autograd.runtime_wrappers._backward_prologue_functional(
                ctx_saved_tensors,
                ctx_symints,
                ctx_opaque_objs,
                metadata,
                maybe_subclass_metadata,
                *flat_args,
            )
            return out

        pgrads = self.fx_tracer.create_proxy(
            kind="call_function",
            target=call_aot_bwd_prologue,
            args=(
                psaved_tensors,
                psymints,
                popaque_objects,
                *pinputs,
            ),
            kwargs={},
        )

        pbackward_state = None
        if maybe_backward_state_idx is not None:
            pbackward_state = self.hooks_proxy[maybe_backward_state_idx]  # type: ignore[index]

        # Copy-paste the AOT backward graph into the compiled autograd graph
        def copy_paste_aot_backward_graph() -> list[torch.Tensor]:
            def num_inputs(graph: torch.fx.Graph) -> int:
                num_args = 0
                for node in graph.nodes:
                    if node.op == "placeholder":
                        num_args += 1
                        continue
                    else:
                        break
                return num_args

            # set up the proxy inputs to bw_module
            # the calling convention is: [*symints, *args (primals and tangents), backward_state]
            num_args = num_inputs(bw_module.graph)  # type: ignore[attr-defined]
            pall_args = [
                pgrads[i] for i in range(num_args - int(pbackward_state is not None))
            ]
            # replace the symints with our symints
            symints = ctx._get_compiled_autograd_symints()
            assert len(symints) == len(ctx.symints)
            psymints = [self.to_proxy(e) for e in symints]
            pall_args[: len(symints)] = psymints
            # Add backward_state
            if pbackward_state is not None:
                pall_args.append(pbackward_state)

            # run over all nodes of the aot_backward graph.
            # copy and paste them all into the compiled autograd graph.
            args_idx = 0
            value_remap = {}
            poutputs: Optional[list[torch.fx.Proxy]] = None

            # names of nodes must appear only once in the fx.Graph
            # dedup AOT backwards that appear multiple times
            deduped_aot_id = str(aot_id)
            if self.aot_id_counter[aot_id]:
                deduped_aot_id += f"_{self.aot_id_counter[aot_id]}"
            self.aot_id_counter[aot_id] += 1

            def make_unique(node_name: str) -> str:
                # make it both informative and unique
                return f"aot{deduped_aot_id}_{node_name}"

            for node in bw_module.graph.nodes:  # type: ignore[attr-defined]
                if node.op == "placeholder":
                    ph = pall_args[args_idx].node
                    ph.name = make_unique(node.name)
                    value_remap[node] = ph
                    args_idx += 1
                elif node.op == "output":
                    assert len(node.args) == 1
                    poutputs = [
                        torch.fx.Proxy(value_remap[n], self.fx_tracer)
                        if isinstance(n, torch.fx.Node)
                        else n
                        for n in node.args[0]
                    ]
                elif node.op == "get_attr":
                    name = node.target
                    qualname = self.fx_tracer.get_fresh_qualname(name)
                    setattr(self.fx_tracer.root, qualname, getattr(bw_module, name))
                    result = self.fx_tracer.create_node("get_attr", qualname, (), {})
                    result.name = make_unique(node.name)
                    value_remap[node] = result
                elif node.op == "call_function":
                    if node.target is torch.ops.aten.view.default:
                        # this aot bwd graph is being lazily compiled
                        # we must manually apply the view_to_reshape post grad pass
                        # since it was already applied to the aot fwd, and baked into the gradients
                        node.target = torch.ops.aten.reshape.default
                    result = self.fx_tracer.graph.node_copy(
                        node, lambda n: value_remap[n]
                    )
                    result.name = make_unique(node.name)
                    value_remap[node] = result
                elif node.op == "call_module":
                    name = node.target
                    qualname = self.fx_tracer.get_fresh_qualname(name)
                    setattr(self.fx_tracer.root, qualname, getattr(bw_module, name))
                    result = self.fx_tracer.graph.node_copy(
                        node, lambda n: value_remap[n]
                    )
                    result.target = qualname
                    value_remap[node] = result
                else:
                    raise AssertionError("shouldn't get here")

            assert poutputs is not None

            # In general we don't know what the shapes of the outputs are, so allocate
            # some dummy sizes for them.
            def dummy() -> torch.Tensor:
                with disable_proxy_modes_tracing():
                    return torch.zeros(0, 0, 0, 0, 123)

            outputs = [
                dummy() if isinstance(o, torch.fx.Proxy) else o for o in poutputs
            ]
            self.bind_objects_to_proxies(outputs, poutputs)
            return outputs

        outputs = copy_paste_aot_backward_graph()

        def proxy_subclass_constructor(
            subclass_meta: Any, is_runtime: bool, unwrapped_args: Sequence[Any]
        ) -> torch.Tensor:
            @torch._dynamo.allow_in_graph  # type: ignore[misc]
            def make_subclass(*unwrapped_args: Any) -> Any:
                return subclass_meta.creation_fn(unwrapped_args, is_runtime=is_runtime)

            punwrapped_args = pytree.tree_map(self.to_proxy, unwrapped_args)

            poutput = self.fx_tracer.create_proxy(
                kind="call_function",
                target=make_subclass,
                args=tuple(punwrapped_args),
                kwargs={},
            )

            output = self.allocate_dummy()
            self.bind_objects_to_proxies([output], [poutput])
            return output

        results = torch._functorch._aot_autograd.runtime_wrappers._backward_epilogue_functional(
            metadata,
            maybe_subclass_metadata,
            outputs,
            make_subclass_override=proxy_subclass_constructor,
        )
        presults = pytree.tree_map(self.to_proxy, results)
        return presults

    def proxy_call_backward(
        self,
        inputs: Sequence[Any],
        output_metadatas: Sequence[Any],
        saved_tensors: Sequence[torch.Tensor],
        backward_idx: int,
        ctx: torch.autograd.function.BackwardCFunction,
        maybe_backward_state_idx: Optional[int],
        opaque_object_indices: list[int],
    ) -> tuple[Optional[torch.Tensor], ...]:
        assert self.hooks_proxy is not None
        pctx = self.hooks_proxy[backward_idx]  # type: ignore[index]
        pinputs = self.to_proxy(inputs)
        psaved_tensors = self.to_proxy(saved_tensors)
        if hasattr(ctx._forward_cls, "_aot_id"):  # type: ignore[attr-defined]
            # AOT backward
            proxies = self.proxy_call_aot_backward(
                pinputs,
                psaved_tensors,
                saved_tensors,
                pctx,
                ctx,
                maybe_backward_state_idx,
                opaque_object_indices,
            )
        else:
            proxies = self.fx_tracer.create_proxy(
                kind="call_function",
                target=call_backward,
                args=(
                    pctx,
                    psaved_tensors,
                    *pinputs,
                ),
                kwargs={},
            )
        assert proxies is not None

        with disable_proxy_modes_tracing():
            # create fake Tensors
            grad_ins: list[Optional[torch.Tensor]] = []
            for idx, output_metadata in enumerate(output_metadatas):
                if output_metadata is None or proxies[idx] is None:
                    grad_ins.append(None)
                    continue

                layout, device, dtype, size = output_metadata
                grad_ins.append(
                    torch.empty(size=size, dtype=dtype, layout=layout, device=device)
                )
            self.bind_objects_to_proxies(grad_ins, proxies)
        return tuple(grad_ins)

    def call_copy_slices_prologue(
        self,
        inputs: Sequence[Any],
        base_sizes: Sequence[Any],
        base_strides: Sequence[Any],
        base_storage_offset: Any,
        view_sizes: Sequence[Any],
        view_strides: Sequence[Any],
        view_storage_offset: Any,
    ) -> Sequence[torch.Tensor]:
        args = (
            inputs,
            self.to_proxy(base_sizes),
            self.to_proxy(base_strides),
            self.to_proxy(base_storage_offset),
            self.to_proxy(view_sizes),
            self.to_proxy(view_strides),
            self.to_proxy(view_storage_offset),
        )
        return self.proxy_call(copy_slices_prologue, args, [None] * 3)

    def call_copy_slices_epilogue(
        self,
        needs_input_grad: Sequence[bool],
        result: torch.Tensor,
        res: Sequence[Any],
        grad_slice: torch.Tensor,
    ) -> Sequence[torch.Tensor]:
        return self.proxy_call(
            copy_slices_epilogue,
            (needs_input_grad, result, res, grad_slice),
            [None] * len(needs_input_grad),
        )

    def allocate_dummy(self) -> torch.Tensor:
        with disable_proxy_modes_tracing():
            # Weird quantity so it's easy to grep
            return torch.zeros([0, 123456789])

    def bind_function(
        self,
        fn_name: str,
        fn: Callable[..., Any],
        is_custom_function: bool,
        is_traceable: bool,
    ) -> str:
        """Binds ops.fn_name = fn"""
        return ops.add(fn_name, fn, is_custom_function, is_traceable)

    def apply_functional(
        self,
        fn_name: str,
        grads: Sequence[Any],
        args: Any,
        output_metadata: Sequence[Any],
    ) -> Sequence[torch.Tensor]:
        """Proxies a call to ops.fn_name(grads, *args) into the graph"""
        op = ops.get(fn_name)
        return self.proxy_call(op, (grads, *args), output_metadata)

    def proxy_call(
        self, fn: Callable[..., Any], args: Any, output_metadata: Sequence[Any]
    ) -> Sequence[torch.Tensor]:
        """Proxies a call to fn(*args) into the graph"""
        proxy_args = pytree.tree_map(lambda e: self.to_proxy(e), args)
        proxy_out = self.fx_tracer.create_proxy(
            "call_function", fn, args=proxy_args, kwargs={}
        )
        result = [self.allocate_dummy() for _ in output_metadata]
        self.bind_objects_to_proxies(result, [proxy_out[i] for i in range(len(result))])
        return result

    def validate_outputs(
        self, _: Any, outputs: Sequence[Any], args: Any, output_metadata: Sequence[Any]
    ) -> Sequence[torch.Tensor]:
        """Proxies a call to ops.validate_outputs(outputs, *args) into the graph"""
        op = ops.get("validate_outputs")
        proxy_args = pytree.tree_map(self.to_proxy, (outputs, *args))
        new_proxy_outputs = self.fx_tracer.create_proxy(
            "call_function", op, args=proxy_args, kwargs={}
        )
        assert len(output_metadata) == len(outputs)
        self.bind_objects_to_proxies(outputs, new_proxy_outputs)
        return outputs

    def accumulate(self, old_var: Any, new_var: Any) -> torch.Tensor:
        old_var_proxy = self.to_proxy(old_var)
        new_var_proxy = self.to_proxy(new_var)
        proxy_out = self.fx_tracer.create_proxy(
            "call_function", torch.add, args=(old_var_proxy, new_var_proxy), kwargs={}
        )
        result = self.allocate_dummy()
        self.bind_objects_to_proxies([result], [proxy_out])
        return result

    def accumulate_grad(
        self, variable: torch.Tensor, grad: torch.Tensor, has_post_hooks: bool
    ) -> None:
        self.fx_tracer.create_proxy(
            "call_function",
            call_accumulate_grad,
            args=(
                self.to_proxy(variable),
                self.to_proxy(grad),
                has_post_hooks,
            ),
            kwargs={},
        )

    def proxy_call_hook(
        self, hook: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> torch.fx.Proxy:
        return self.fx_tracer.create_proxy(
            "call_function",
            call_hook,
            (
                hook,
                *[self.to_proxy(x) for x in args],
            ),
            kwargs,
        )

    def unpack_hook(self, hook_id: int, data_id: int) -> torch.Tensor:
        assert self.hooks_proxy is not None
        hook = self.hooks_proxy[hook_id]  # type: ignore[index]
        data = self.packed_data_proxy[data_id]  # type: ignore[index]
        proxy = self.proxy_call_hook(
            hook,
            data,
            hook_type="unpack_hook",
        )
        out = self.allocate_dummy()
        self.bind_objects_to_proxies([out], [proxy])
        return out

    def tensor_pre_hook(
        self, inputs: list[torch.Tensor], hook_id: int, i: int
    ) -> list[torch.Tensor]:
        assert self.hooks_proxy is not None
        hook = self.hooks_proxy[hook_id]  # type: ignore[index]
        proxy = self.proxy_call_hook(
            hook,
            inputs[i],
            hook_type="tensor_pre_hook",
        )
        with disable_proxy_modes_tracing():
            inputs[i] = maybe_clone(inputs[i])  # type: ignore[assignment]
            self.bind_objects_to_proxies([inputs[i]], [proxy])
        return inputs

    def cpp_tensor_pre_hook(
        self, inputs: list[torch.Tensor], hook_id: int, i: int
    ) -> list[torch.Tensor]:
        proxy = self.fx_tracer.create_proxy(
            "call_function",
            torch._C._dynamo.compiled_autograd.call_cpp_tensor_pre_hooks,
            (hook_id, self.to_proxy(inputs[i])),
            {},
        )
        with disable_proxy_modes_tracing():
            inputs[i] = maybe_clone(inputs[i])  # type: ignore[assignment]
            self.bind_objects_to_proxies([inputs[i]], [proxy])
        return inputs

    def pre_hook(self, inputs: Sequence[Any], hook_id: int) -> list[torch.Tensor]:
        assert self.hooks_proxy is not None
        hook = self.hooks_proxy[hook_id]  # type: ignore[index]
        proxies = self.proxy_call_hook(
            hook,
            inputs,
            hook_type="pre_hook",
        )
        with disable_proxy_modes_tracing():
            inputs = [maybe_clone(x) for x in inputs]
            self.bind_objects_to_proxies(inputs, proxies)
        return inputs

    def post_hook(
        self, outputs: list[torch.Tensor], inputs: Sequence[torch.Tensor], hook_id: int
    ) -> list[torch.Tensor]:
        assert self.hooks_proxy is not None
        hook = self.hooks_proxy[hook_id]  # type: ignore[index]
        proxies = self.proxy_call_hook(
            hook,
            outputs,
            inputs,
            hook_type="post_hook",
        )
        with disable_proxy_modes_tracing():
            outputs = [maybe_clone(x) for x in outputs]  # type: ignore[misc]
            self.bind_objects_to_proxies(outputs, proxies)
        return outputs

    def post_acc_grad_hook(
        self, input: torch.Tensor, hook_id: int
    ) -> list[torch.Tensor]:
        assert isinstance(input, torch.Tensor)
        assert self.hooks_proxy is not None
        hook = self.hooks_proxy[hook_id]  # type: ignore[index]
        proxy = self.proxy_call_hook(
            hook,
            input,
            hook_type="post_acc_grad_hook",
        )
        with disable_proxy_modes_tracing():
            res = [maybe_clone(input)]
            self.bind_objects_to_proxies(res, [proxy])
        return res  # type: ignore[return-value]

    # Note: [Compiled autograd and cudagraphs]
    # Eager autograd backward implements scalars as 0-dim tensors, see DivBackward0::other_.
    # When compiled autograd traces those nodes, it lifts the scalar tensors, resulting in a graph
    # with some cpu 0-dim tensor inputs. To prevent the entire graph from skipping cudagraph, we move the
    # scalars tensors to cuda. This works because ATen/prims ops will accept cuda 0-dim tensors too.
    def move_graph_nodes_to_cuda(self, graph: torch.fx.Graph) -> list[int]:
        to_move: dict[int, torch.fx.Node] = {}
        has_cuda_inputs = False
        nodes = list(graph.nodes)
        assert nodes[0].target == "inputs"
        inputs = nodes[0]
        inputs_users = list(inputs.users.keys())
        # input access nodes should immediately follow placeholder nodes
        first_getitem_idx = len(_graph_placeholders)
        assert nodes[first_getitem_idx] == inputs_users[0]
        last_getitem_idx = first_getitem_idx + len(inputs_users) - 1
        assert nodes[last_getitem_idx] == inputs_users[-1]
        # getitem nodes on inputs
        for i, node in enumerate(inputs_users):
            if not has_cuda_inputs and node.meta["val"].device.type == "cuda":
                has_cuda_inputs = True
                continue

            is_cpu = node.meta["val"].device.type == "cpu"
            is_scalar = len(node.meta["val"].size()) == 0
            if is_cpu and is_scalar:
                node_users = list(node.users.keys())
                # We can only move the cpu scalar if it is not exposed to user code.
                if all(
                    (
                        isinstance(user.target, torch._ops.OpOverload)
                        and user.target.namespace in ("prims", "aten")
                    )
                    or (
                        isinstance(user.target, Op)
                        and not user.target.is_custom_function
                    )
                    for user in node_users
                ):
                    # all users are prims/aten, can move safely
                    to_move[i] = node

        # only move cpu scalars to cuda if there were cuda activations in this graph,
        # this is to handle the case where cudagraphs is enabled on a cpu-only graph
        if has_cuda_inputs:
            for node in to_move.values():
                verbose_log.debug("Moving node %s from cpu to cuda", node)
                node.meta["val"] = node.meta["val"].cuda()

            # return runtime indices we need to move to cuda
            return list(to_move.keys())

        return []

    def is_sym_node(self, node: Any) -> bool:
        return (
            isinstance(node, torch.fx.Node)
            and node.op == "call_function"
            and node.target
            in [torch.ops.aten.sym_size.int, torch.ops.aten.sym_numel.default]
        )

    def dce(self) -> None:
        # Most of these removed nodes would have been removed during Dynamo and AOTDispatch
        # Remove some of these nodes earlier to improve compilation speed

        # Dynamo guards will error instead of creating aliasing guards unless we unpack them in the graph
        unpack_nodes: OrderedSet[torch.fx.Node] = OrderedSet()
        i: int | None = None
        for i, node in enumerate(self.fx_tracer.graph.find_nodes(op="placeholder")):  # noqa: B007
            unpack_nodes.update(node.users.keys())
        assert i == len(_graph_placeholders) - 1

        def is_impure(node: torch.fx.Node) -> bool:
            if node in unpack_nodes or (
                node.op == "call_function" and node.target in _impure_targets
            ):
                return True
            return node.is_impure()

        before = len(self.fx_tracer.graph.nodes)
        self.fx_tracer.graph.eliminate_dead_code(is_impure)
        after = len(self.fx_tracer.graph.nodes)
        verbose_log.debug("DCE removed %d nodes", before - after)

    def remove_unused_sizes(self) -> set[int]:
        used_sizes = []
        unused_sizes = []

        # seek placeholder, should be at nodes[1]
        it = iter(self.fx_tracer.graph.nodes)
        next(it)
        sizes_node = next(it)
        assert sizes_node.name == "sizes"

        for getitem_node in sizes_node.users:
            assert getitem_node.target is operator.getitem
            if getitem_node.users:
                used_sizes.append(getitem_node)
            else:
                # remove from the graph
                unused_sizes.append(getitem_node)

        used_sizes_idx: set[int] = set()
        for used in used_sizes:
            assert isinstance(used.args, tuple)
            assert used.args[0] == sizes_node
            assert isinstance(used.args[1], int)
            next_size_idx = len(used_sizes_idx)
            # used later reindex the runtime sizes arg
            used_sizes_idx.add(used.args[1])
            # reindex the graph
            used.args = (used.args[0], next_size_idx)

        for unused in unused_sizes:
            self.fx_tracer.graph.erase_node(unused)

        return used_sizes_idx

    def create_graph_module(self, id: str) -> GraphModule:
        return GraphModule(self.fx_tracer.root, self.fx_tracer.graph, id)

    def end_capture(self, outputs: Any) -> tuple[Callable[..., Any], Any]:
        self.fx_tracer.create_proxy(
            "call_function",
            FakeCompiledAutogradEngine._exec_final_callbacks_stub,
            (),
            {},
        )
        self.stack.close()
        self.fx_tracer.create_node(
            "output",
            "output",
            (self.fx_tracer.create_arg(self.to_proxy(outputs)),),
            {},
        )
        runtime_inputs_to_move: list[int] = []
        if snapshot_cudagraph_enabled():
            runtime_inputs_to_move = self.move_graph_nodes_to_cuda(self.fx_tracer.graph)

        # We traced using dummy tensors. Delete all the metadata of the dummy tensors.
        # It's probably better to refactor this class to use a different tracer
        # than the make_fx tracer, but that is a larger change.
        for node in self.fx_tracer.graph.nodes:
            for field in ["tensor_meta", "example_value", "val"]:
                if field in node.meta:
                    del node.meta[field]

        trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": "compiled_autograd_graph_pre_reordering",
                "encoding": "string",
            },
            payload_fn=lambda: GraphModule(
                self.fx_tracer.root,
                self.fx_tracer.graph,
                f"CompiledAutograd{self.id}PreReordering",
            ).print_readable(print_output=False),
        )
        self.delay_unpack_hook_nodes()
        self.reorder_tensor_pre_hook_nodes()
        self.reorder_pre_hook_nodes_to_schedule_asap()
        self.reorder_accumulate_grad_nodes()
        self.reorder_pre_hook_nodes_to_mimic_eager()
        self.reorder_post_acc_grad_hook_nodes()
        self.reorder_post_hook_nodes()
        # TODO(yf225): work around: remove dead codes like `sym_size` and `sym_numel` which are not used downstream. e.g.
        # ```
        # sym_numel_default = torch.ops.aten.sym_numel.default(sum_109);  sum_109 = None
        # eq_115 = 16 == sym_numel_default;  sym_numel_default = eq_115 = None
        # sym_size_int_39 = torch.ops.aten.sym_size.int(getitem_112, 1);  getitem_112 = None
        # eq_116 = 16 == sym_size_int_39;  eq_116 = None
        # eq_117 = 16 == sym_size_int_39;  sym_size_int_39 = eq_117 = None
        # ```
        # Proper fix is Richard's Python compiled autograd effort which will avoid calling make_fx and
        # should prevent these ops from going into the CA graph.
        self.dce()
        if self.nan_checker:
            self.nan_checker.prep_with_graph(self.fx_tracer.graph)

        # keep only sizes that are actually used in the graph
        used_sizes_idx = self.remove_unused_sizes()

        graph = self.create_graph_module(f"CompiledAutograd{self.id}")
        set_locals_to_steal(graph, ["inputs"])
        lazy_graph_code = lazy_format_graph_code(
            "Compiled autograd graph",
            graph,
            include_device=True,
            include_stride=True,
            colored=True,
        )
        compiled_autograd_log.info("%s", lazy_graph_code)
        verbose_log.debug("%s", lazy_graph_code)
        trace_structured(
            "compiled_autograd_graph",
            payload_fn=lambda: graph.print_readable(print_output=False),
        )

        def runtime_wrapper(
            compiled_fn: Callable[..., Any],
            inputs: Any,
            sizes: Any,
            scalars: Any,
            hooks: Any,
            packed_inputs: Any,
        ) -> tuple[Any, Any]:
            global in_compiled_autograd_region
            try:
                in_compiled_autograd_region = True

                if self.nan_checker:
                    self.nan_checker.prep_with_inputs(inputs)

                filtered_sizes = []
                for idx, integer in enumerate(sizes):
                    if idx in used_sizes_idx:
                        # can't create negative size
                        if integer > 0:
                            filtered_sizes.append(torch.empty(0, integer))
                            torch._dynamo.maybe_mark_dynamic(filtered_sizes[-1], 1)
                        else:
                            filtered_sizes.append(integer)

                for i in runtime_inputs_to_move:
                    inputs[i] = inputs[i].pin_memory().cuda(non_blocking=True)

                with _disable(), make_compile_context(self.id):
                    out = compiled_fn(
                        inputs, filtered_sizes, scalars, hooks, packed_inputs
                    )
                    if self.nan_checker:
                        self.nan_checker.check(out)
                    return out
            finally:
                in_compiled_autograd_region = False

        get_chromium_event_logger().log_event_end(
            "compiled_autograd",
            time.time_ns(),
            {"graph_id": self.id},
            self.start_time_ns,
            log_pt2_compile_event=True,
        )
        self.compile_context.__exit__(None, None, None)
        return runtime_wrapper, self.compiler_fn(graph)

    @staticmethod
    def get_all_nodes(args: Sequence[Any]) -> list[torch.fx.Node]:
        # filter out non-Node args, like None
        nodes = [n for n in args if type(n) is torch.fx.Node]
        return nodes

    @staticmethod
    def is_placeholder(node: torch.fx.Node) -> bool:
        if node.op == "placeholder" or (
            node.op == "call_function"
            and node.target is operator.getitem
            and node.args[0].op == "placeholder"  # type: ignore[union-attr, arg-type]
        ):
            return True
        return False

    def reorder_accumulate_grad_nodes(self) -> None:
        """
        Usage of AOTAutograd causes all the accumulate_grad_ nodes to get pushed to the end of
        the graph.  This differs from eager mode, which schedules them as soon as possible. This
        pass attempts to reorder the graph to mimic eager behavior.
        """
        for node in self.fx_tracer.graph.find_nodes(
            op="call_function", target=call_accumulate_grad
        ):
            param_node, grad_node = node.args[0], node.args[1]
            getitem_node = None
            if grad_node.target is operator.getitem:
                getitem_node = grad_node
                grad_node = getitem_node.args[0]

            arg = max([param_node, grad_node])  # last arg
            if arg is not node.prev and not self.is_placeholder(arg):
                arg.append(node)
                if getitem_node is not None:
                    arg.append(getitem_node)

    def delay_unpack_hook_nodes(self) -> None:
        """
        We can delay unpack hooks until they are needed, even later than in the eager autograd engine.
        """
        for node in self.fx_tracer.graph.find_nodes(
            op="call_function", target=call_hook
        ):
            if node.kwargs.get("hook_type", None) != "unpack_hook":
                continue

            first_user = min(node.users)
            first_user.prepend(node)

    def reorder_tensor_pre_hook_nodes(self) -> None:
        """
        Usage of AOTAutograd causes all the tensor_pre_hook nodes to get pushed
        to the end of the graph. This differs from eager mode, which schedules
        them as soon as possible. This pass attempts to reorder the graph to
        mimic eager behavior.
        """
        for node in self.fx_tracer.graph.find_nodes(
            op="call_function", target=call_hook
        ):
            if node.kwargs.get("hook_type", None) != "tensor_pre_hook":
                continue

            getitem_node = node.args[0]
            input_node = node.args[1]  # tensor_pre_hook handle only one grad tensor

            if input_node is not node.prev and not self.is_placeholder(input_node):
                input_node.append(getitem_node)
                getitem_node.append(node)

    def reorder_pre_hook_nodes_to_schedule_asap(self) -> None:
        """
        In this function, we schedule the pre hooks as soon as possible. This
        does not match eager behavior (schedule pre hook right before its
        registered node), but it can make acc grad be scheduled properly when
        the pre hooks are registered to them. After reordering acc grad node, we
        will reorder the pre hooks again to mimic eager behavior.
        """
        for node in self.fx_tracer.graph.find_nodes(
            op="call_function", target=call_hook
        ):
            if node.kwargs.get("hook_type", None) != "pre_hook":
                continue

            getitem_node = node.args[0]
            # pre_hook handle a tuple of grad tensors
            input_nodes = self.get_all_nodes(node.args[1])

            to_remove = []
            to_append = []
            hook_block = [node]  # contain the hook and hook args getitem
            for n in input_nodes:
                if n.op == "call_function" and n.target is operator.getitem:
                    to_append.append(n.args[0])
                    to_remove.append(n)
                    hook_block.append(n)
            for a, b in zip(to_remove, to_append):
                input_nodes.remove(a)
                input_nodes.append(b)  # type: ignore[arg-type]

            arg = max(input_nodes)  # last input
            if arg is not node.prev and not self.is_placeholder(arg):
                arg.append(getitem_node)
                for n in hook_block:
                    getitem_node.append(n)

    def reorder_pre_hook_nodes_to_mimic_eager(self) -> None:
        """
        Usage of AOTAutograd causes all the pre_hook nodes to get pushed to the
        end of the graph. This differs from eager mode, which schedules them
        right before their registered node execution. This pass attempts to
        reorder the graph to mimic eager behavior.
        """
        pre_hooks = []
        for node in self.fx_tracer.graph.find_nodes(
            op="call_function", target=call_hook
        ):
            if node.kwargs.get("hook_type", None) != "pre_hook":
                continue
            pre_hooks.append(node)

        for node in reversed(pre_hooks):
            hook_getitem_node = node.args[0]

            users = list(node.users.keys())
            if len(users) == 0:
                continue

            # users are all getitem ops and they are used by same registered node
            assert all(
                user.op == "call_function" and user.target is operator.getitem
                for user in users
            )
            registered_node = next(iter(users[0].users.keys()))

            if registered_node is not node.next:
                registered_node.prepend(hook_getitem_node)
                registered_node.prepend(node)
                for getitem in users:
                    registered_node.prepend(getitem)

    def reorder_post_acc_grad_hook_nodes(self) -> None:
        """
        Usage of AOTAutograd causes all the post_acc_grad_hook nodes to get
        pushed to the end of the graph. This differs from eager mode, which
        schedules them as soon as possible. This pass attempts to reorder the
        graph to mimic eager behavior.
        """
        post_acc_grad_hooks = []
        for node in self.fx_tracer.graph.find_nodes(
            op="call_function", target=call_hook
        ):
            if node.kwargs.get("hook_type", None) != "post_acc_grad_hook":
                continue
            post_acc_grad_hooks.append(node)

        # nodes in post_acc_grad_hooks are in topo order. For hooks registered
        # to same node, we should keep their relative order
        for node in reversed(post_acc_grad_hooks):
            getitem_node = node.args[0]
            param_node = node.args[1]  # post_acc_grad_hook handle one param

            # find the corresponding acc_grad node
            acc_grad_node = None
            for n in list(param_node.users.keys()):
                if n.op == "call_function" and n.target is call_accumulate_grad:
                    acc_grad_node = n
                    break

            assert acc_grad_node is not None, (
                "post_acc_grad_hook must have corresponding acc grad node"
            )

            # append post_acc_grad_hook after acc_grad node
            acc_grad_node.append(getitem_node)
            getitem_node.append(node)

    def reorder_post_hook_nodes(self) -> None:
        """
        Usage of AOTAutograd causes all the post_hook nodes to get pushed to the
        end of the graph. This differs from eager mode, which schedules them as
        soon as possible. This pass attempts to reorder the graph to mimic eager
        behavior.
        """
        post_hooks = []
        for node in self.fx_tracer.graph.find_nodes(
            op="call_function", target=call_hook
        ):
            if node.kwargs.get("hook_type", None) != "post_hook":
                continue
            post_hooks.append(node)

        for node in reversed(post_hooks):
            getitem_node = node.args[0]
            output_nodes = node.args[1]
            input_nodes = node.args[2]

            if len(output_nodes) > 0:
                continue

            input_nodes_and_users = []
            input_nodes_and_users.extend(list(input_nodes))
            for input_node in input_nodes:
                input_nodes_and_users.extend(
                    user
                    for user in list(input_node.users.keys())
                    if not (
                        user.op == "call_function"
                        and user.target is call_hook
                        and node.kwargs.get("hook_type", None) == "post_hook"
                    )
                )

            arg = max(input_nodes_and_users)  # last input users
            if arg.op == "call_function" and arg.target is call_accumulate_grad:
                param_node = arg.args[0]
                post_acc_grad_hook_node = None
                for n in list(param_node.users.keys()):
                    if (
                        n.op == "call_function"
                        and n.target is call_hook
                        and n.kwargs.get("hook_type", None) == "post_acc_grad_hook"
                    ):
                        post_acc_grad_hook_node = n

                if post_acc_grad_hook_node is not None:
                    post_acc_grad_hook_node.append(getitem_node)
                    getitem_node.append(node)
                    continue

            if arg is not node.prev and not self.is_placeholder(arg):
                arg.append(getitem_node)
                getitem_node.append(node)

    def to_proxy(self, t: Any) -> Any:
        if t is None:
            return None
        if isinstance(t, list):
            return [self.to_proxy(x) for x in t]
        if isinstance(t, tuple):
            return tuple(self.to_proxy(x) for x in t)
        if isinstance(t, (torch.SymInt, torch.SymFloat)):
            return self.symnode_proxy_lookup[t.node]
        if not isinstance(t, torch.Tensor):
            # constant types like device, dtype, str
            return t
        proxy_tensor = fetch_object_proxy(self.fx_tracer, t)
        assert isinstance(proxy_tensor, torch.fx.experimental.proxy_tensor._ProxyTensor)
        return proxy_tensor.proxy

    def bind_objects_to_proxies(
        self,
        objects: Sequence[Any],
        proxies: Any,
        origins: Optional[list[tuple[int, str]]] = None,
    ) -> Sequence[Any]:
        if isinstance(proxies, torch.fx.Proxy):
            if origins:
                assert len(origins) == len(objects)
                bound_proxies = []
                for i in range(len(objects)):
                    nodecall_index, node_name = origins[i]
                    self.set_node_origin(node_name, nodecall_index, None)
                    bound_proxies.append(proxies[i])  # type: ignore[index]
                proxies = bound_proxies
            else:
                proxies = [proxies[i] for i in range(len(objects))]  # type: ignore[index]

        assert len(objects) == len(proxies)
        track_tensor_tree(objects, proxies, constant=None, tracer=self.fx_tracer)
        return proxies

    def bind_backward_state(self, index: int) -> BackwardState:
        assert self.hooks_proxy is not None
        proxy = self.hooks_proxy[index]  # type: ignore[index]
        bw_state = BackwardState()
        track_tensor_tree(bw_state, proxy, constant=None, tracer=self.fx_tracer)
        return bw_state

    def set_node_origin(
        self,
        node_name: str,
        nodecall_index: int,
        pyobj: Optional[torch.autograd.Function],
    ) -> None:
        maybe_aot_id = ""
        if pyobj is not None:
            forward_cls = pyobj._forward_cls  # type: ignore[attr-defined]
            if hasattr(forward_cls, "_aot_id"):
                # backward was created by AOT Dispatcher
                if forward_cls._lazy_backward_info is None:
                    raise RuntimeError(
                        """This compiled backward function was saved by AOTAutogradCache, which does not support
                    compiled autograd. Please turn off AOTAutogradCache using `TORCHINDUCTOR_AUTOGRAD_CACHE=0`."""
                    )
                maybe_aot_id = forward_cls._aot_id
        new_code = f"{node_name}{maybe_aot_id} (NodeCall {nodecall_index})"
        raw_stack_trace = CapturedTraceback.extract().format()[-1]
        new_stack_trace = raw_stack_trace.replace(
            "raw_stack_trace = CapturedTraceback.extract().format()[-1]", new_code
        )
        set_stack_trace(new_stack_trace)


# state of the autograd engine dispatch, kept in sync by enable/disable context managers
compiled_autograd_enabled = False

# global flag to check if compiled autograd is enabled but Dynamo stance is "force_eager"
compiled_autograd_enabled_force_eager = False

# global flag to check if we are processing graphs produced from a compiled autograd graph
in_compiled_autograd_region = False

active_disable_ctx = False

depth = 0


@contextlib.contextmanager
def _enable(
    compiler_fn: Callable[..., Any],
    dynamic: bool = True,
    ignore_active_disable_ctx: bool = True,
) -> Generator[None, None, None]:
    # The entrypoint to enable CA.
    # It is recommended to enable via `torch._dynamo.config.compiled_autograd = True` rather
    # than using this context manager directly. If you are torch.compiling the corresponding
    # forward pass, make sure they are wrapped under this context as well.
    #
    # Example:
    #   def train(model, inputs, target):
    #     compiled_model = torch.compile(model)
    #     pred = compiled_model(data)
    #     loss = compute_loss(pred, target)
    #     loss.backward()
    #
    #   with _enable(compiler_fn):
    #      train(model, inputs, target)
    #
    # Inputs:
    # - compiler_fn: The wrapper that will consume the compiled autograd graph, e.g. `torch.compile`
    # - dynamic: Whether compiled autograd will treat tensors in the autograd graph (params, activations) as dynamic.
    #   This doesn't affect the dynamic configuration of the compilation wrapper.

    if not ignore_active_disable_ctx and active_disable_ctx:
        yield
    else:
        if dynamic:
            assert type(dynamic) is bool

        from torch._dynamo import eval_frame

        if eval_frame._stance.stance == "force_eager":
            # If user explicitly sets Dynamo stance to "force_eager", we want Compiled Autograd
            # to fall back to eager as well.
            global compiled_autograd_enabled_force_eager
            compiled_autograd_enabled_force_eager = True
            try:
                yield
            finally:
                compiled_autograd_enabled_force_eager = False
        else:
            # we need to import this, because user might not have imported it if they directly use this context manager
            # we need to lazily import it, because of circular dependencies
            if torch.cuda.is_available():
                from torch._inductor import cudagraph_trees  # noqa: F401

            (
                prior_compiler,
                prior_dynamic,
            ) = torch._C._dynamo.compiled_autograd.set_autograd_compiler(
                functools.partial(AutogradCompilerInstance, compiler_fn), dynamic
            )
            if snapshot_verbose_logging_enabled():
                torch._C._dynamo.compiled_autograd.set_verbose_logger(verbose_log)  # type:ignore[arg-type]
            global compiled_autograd_enabled
            compiled_autograd_enabled = True
            global depth
            prior_depth = depth
            depth += 1
            try:
                with torch.autograd.set_multithreading_enabled(False):
                    yield
            finally:
                if not prior_compiler:
                    compiled_autograd_enabled = False
                torch._C._dynamo.compiled_autograd.set_autograd_compiler(
                    prior_compiler, prior_dynamic
                )
                depth -= 1
                assert depth == prior_depth, (
                    "Nested Compiled Autograd Contexts must return before their parent context"
                )


@contextlib.contextmanager
def _disable() -> Generator[None, None, None]:
    (
        prior_compiler,
        prior_dynamic,
    ) = torch._C._dynamo.compiled_autograd.set_autograd_compiler(None, False)
    global compiled_autograd_enabled
    compiled_autograd_enabled = False
    global active_disable_ctx
    if not active_disable_ctx:
        active_disable_ctx = True
    try:
        yield
    finally:
        if prior_compiler:
            compiled_autograd_enabled = True
        active_disable_ctx = False
        torch._C._dynamo.compiled_autograd.set_autograd_compiler(
            prior_compiler, prior_dynamic
        )


# return to starting state of a new process
def reset() -> None:
    global compiled_autograd_enabled
    compiled_autograd_enabled = False
    assert not in_compiled_autograd_region
    torch._C._dynamo.compiled_autograd.set_autograd_compiler(None, False)
    torch._C._dynamo.compiled_autograd.set_verbose_logger(None)
    torch._C._dynamo.compiled_autograd.clear_cache()
    global COMPILE_COUNTER
    COMPILE_COUNTER = itertools.count()


# Reimplementation of part of CopySlices::apply in Python.
# The shared code is really similar so we're not going to try to deduplicate.
def copy_slices_prologue(
    inputs: Sequence[torch.Tensor],
    base_sizes: Sequence[IntLikeType],
    base_strides: Sequence[IntLikeType],
    base_storage_offset: IntLikeType,
    view_sizes: Sequence[IntLikeType],
    view_strides: Sequence[IntLikeType],
    view_storage_offset: IntLikeType,
) -> list[torch.Tensor]:
    grad = inputs[0]
    result = grad.new_empty_strided(base_sizes, base_strides)
    assert grad is not None
    result.copy_(grad)
    offset = view_storage_offset - base_storage_offset
    grad_slice = result.as_strided(view_sizes, view_strides, offset)
    return [result, grad_slice, grad_slice.clone(memory_format=torch.contiguous_format)]


# Reimplementation of part of CopySlices::apply in Python.
# The shared code is really similar so we're not going to try to deduplicate.
def copy_slices_epilogue(
    needs_input_grad: Sequence[bool],
    result: torch.Tensor,
    res: Sequence[Optional[torch.Tensor]],
    grad_slice: torch.Tensor,
) -> list[Optional[torch.Tensor]]:
    grad_inputs: list[Optional[torch.Tensor]] = [None] * len(needs_input_grad)
    for i in range(len(needs_input_grad)):
        if needs_input_grad[i]:
            if res[i] is None:
                continue
            if i == 0:
                to_copy = res[i]
                assert to_copy is not None
                grad_slice.copy_(to_copy)
                grad_inputs[i] = result
            else:
                grad_inputs[i] = res[i]
    return grad_inputs
