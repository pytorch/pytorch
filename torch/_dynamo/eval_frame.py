import contextlib
import functools
import inspect
import logging
import os
import sys
import textwrap
import threading
import traceback
import types
import warnings
from enum import Enum
from typing import Optional, Tuple, TYPE_CHECKING, Union
from unittest.mock import patch

import torch
import torch.utils._pytree as pytree
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.graph import _PyTreeCodeGen, _PyTreeInfo
from torch.nn.parallel.distributed import DistributedDataParallel
from .backends.registry import CompilerFn, lookup_backend

from .hooks import Hooks

if TYPE_CHECKING:
    from torch._C._dynamo.eval_frame import (  # noqa: F401
        reset_code,
        set_eval_frame,
        set_guard_error_hook,
        set_guard_fail_hook,
        skip_code,
        unsupported,
    )
else:
    for name in dir(torch._C._dynamo.eval_frame):
        if name.startswith("__"):
            continue
        globals()[name] = getattr(torch._C._dynamo.eval_frame, name)

from . import config, convert_frame, skipfiles, utils
from .exc import ResetRequired
from .mutation_guard import install_generation_tagging_init
from .types import DynamoCallback
from .utils import compile_times

log = logging.getLogger(__name__)

from torch.fx.experimental import proxy_tensor

always_optimize_code_objects = utils.ExactWeakKeyDictionary()
null_context = contextlib.nullcontext

# See https://github.com/python/typing/pull/240
class Unset(Enum):
    token = 0


unset = Unset.token

compile_lock = threading.RLock()
most_recent_backend: Optional[CompilerFn] = None


class OptimizedModule(torch.nn.Module):
    """
    Wraps the original nn.Module object and later patches its
    forward method to optimized self.forward method.
    """

    def __init__(self, mod, dynamo_ctx):
        super().__init__()
        # Installs the params/buffer
        self._orig_mod = mod
        self.dynamo_ctx = dynamo_ctx

    def __getattr__(self, name):
        if name == "_orig_mod":
            return self._modules["_orig_mod"]
        return getattr(self._orig_mod, name)

    def forward(self, *args, **kwargs):
        return self.dynamo_ctx(self._orig_mod.forward)(*args, **kwargs)


def remove_from_cache(f):
    """
    Make sure f.__code__ is not cached to force a recompile
    """
    if isinstance(f, types.CodeType):
        reset_code(f)
    elif hasattr(f, "__code__"):
        reset_code(f.__code__)
    elif hasattr(getattr(f, "forward", None), "__code__"):
        reset_code(f.forward.__code__)
    else:
        from . import reset

        reset()
        log.warning("could not determine __code__ for %s", f)


def nothing():
    pass


def innermost_fn(fn):
    """
    In case of nesting of _TorchDynamoContext calls, find the innermost
    function. TorchDynamo caches on fn.__code__ object, so its necessary to find
    the innermost function to pass on the optimize, run, disable etc.
    """
    unaltered_fn = fn
    while hasattr(unaltered_fn, "_torchdynamo_orig_callable"):
        unaltered_fn = unaltered_fn._torchdynamo_orig_callable
        assert callable(unaltered_fn)
    return unaltered_fn


@contextlib.contextmanager
def enable_dynamic(enable: bool = True):
    if not enable:
        yield
        return
    with config.patch(dynamic_shapes=True, specialize_int_float=False):
        yield


class _TorchDynamoContext:
    def __init__(
        self,
        callback: DynamoCallback,
        on_enter=nothing,
        backend_ctx_ctor=null_context,
        patch_fn=nothing,
        first_ctx=False,
        *,
        dynamic=False,
    ):
        super().__init__()
        assert callable(callback) or callback is False or callback is None
        self.callback: DynamoCallback = callback
        self.prior: Union[Unset, DynamoCallback] = unset
        self.on_enter = on_enter
        self.extra_ctx_ctor = backend_ctx_ctor
        self.first_ctx = first_ctx
        self.dynamic = dynamic
        patch_fn()

    def __enter__(self):
        if config.raise_on_ctx_manager_usage:
            raise RuntimeError(
                "torch._dynamo.optimize(...) is used with a context manager. "
                "Please refer to https://github.com/pytorch/torchdynamo#usage-example "
                "to use torch._dynamo.optimize(...) as an annotation/decorator. "
            )
        self.on_enter()
        self.prior = set_eval_frame(self.callback)
        self.backend_ctx = self.extra_ctx_ctor()
        self.backend_ctx.__enter__()
        self.dynamic_ctx = enable_dynamic(self.dynamic)
        self.dynamic_ctx.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        assert self.prior is not unset
        set_eval_frame(self.prior)
        self.prior = unset
        # TODO: This is totally not the right way to chain contexts manually
        self.dynamic_ctx.__exit__(exc_type, exc_val, exc_tb)
        self.backend_ctx.__exit__(exc_type, exc_val, exc_tb)

    def __call__(self, fn):
        fn = innermost_fn(fn)
        # Optimize the forward method of torch.nn.Module object
        if isinstance(fn, torch.nn.Module):
            mod = fn
            new_mod = OptimizedModule(mod, self)
            # Save the function pointer to find the original callable while nesting
            # of decorators.
            new_mod._torchdynamo_orig_callable = mod.forward
            return new_mod

        assert callable(fn)

        callback = self.callback
        on_enter = self.on_enter
        backend_ctx_ctor = self.extra_ctx_ctor

        @functools.wraps(fn)
        def _fn(*args, **kwargs):
            if (
                not isinstance(self, DisableContext)
                and torch.fx._symbolic_trace.is_fx_tracing()
            ):
                if config.error_on_nested_fx_trace:
                    raise RuntimeError(
                        "Detected that you are using FX to symbolically trace "
                        "a dynamo-optimized function. This is not supported at the moment."
                    )
                else:
                    return fn(*args, **kwargs)

            on_enter()
            prior = set_eval_frame(callback)
            backend_ctx = backend_ctx_ctor()
            backend_ctx.__enter__()
            dynamic_ctx = enable_dynamic(self.dynamic)
            dynamic_ctx.__enter__()
            try:
                return fn(*args, **kwargs)
            finally:
                set_eval_frame(prior)
                dynamic_ctx.__exit__(None, None, None)
                backend_ctx.__exit__(None, None, None)

        # hooks to properly handle inlining
        if isinstance(self, DisableContext):
            _fn._torchdynamo_disable = True  # type: ignore[attr-defined]
        else:
            _fn._torchdynamo_inline = fn  # type: ignore[attr-defined]

        # Save the function pointer to find the original callable while nesting
        # of decorators.
        _fn._torchdynamo_orig_callable = fn  # type: ignore[attr-defined]

        # If the function is called using torch._dynamo.optimize decorator, we
        # should prevent any type of skipping.
        if callback not in (None, False):
            if not hasattr(fn, "__code__"):
                raise RuntimeError(
                    textwrap.dedent(
                        """

                        torch._dynamo.optimize is called on a non function object.
                        If this is a callable class, please wrap the relevant code into a function and optimize the
                        wrapper function.

                        >> class CallableClass:
                        >>     def __init__(self):
                        >>         super().__init__()
                        >>         self.relu = torch.nn.ReLU()
                        >>
                        >>     def __call__(self, x):
                        >>         return self.relu(torch.sin(x))
                        >>
                        >>     def print_hello(self):
                        >>         print("Hello world")
                        >>
                        >> mod = CallableClass()

                        If you want to optimize the __call__ function and other code, wrap that up in a function

                        >> def wrapper_fn(x):
                        >>     y = mod(x)
                        >>     return y.sum()

                        and then optimize the wrapper_fn

                        >> opt_wrapper_fn = torch._dynamo.optimize(wrapper_fn)
                        """
                    )
                )
            always_optimize_code_objects[fn.__code__] = True

        return _fn


class OptimizeContext(_TorchDynamoContext):
    @staticmethod
    def _different_backend(old, new):
        return not (old == new or old is None)

    def __init__(self, callback, backend_ctx_ctor, first_ctx=False, *, dynamic=False):
        def on_enter():
            global most_recent_backend
            if OptimizeContext._different_backend(most_recent_backend, compiler_fn):
                if config.raise_on_backend_change:
                    raise ResetRequired()
                else:
                    warnings.warn(
                        "changing options to `torch.compile()` may require "
                        "calling `torch._dynamo.reset()` to take effect"
                    )
            most_recent_backend = compiler_fn
            install_generation_tagging_init()

        compiler_fn = innermost_fn(callback)
        super().__init__(
            callback=callback,
            on_enter=on_enter,
            backend_ctx_ctor=backend_ctx_ctor,
            patch_fn=TorchPatcher.patch,
            first_ctx=first_ctx,
            dynamic=dynamic,
        )


class RunOnlyContext(_TorchDynamoContext):
    def __init__(self):
        super().__init__(callback=False)


class DisableContext(_TorchDynamoContext):
    def __init__(self):
        super().__init__(callback=None)


def catch_errors_wrapper(callback, hooks: Hooks):
    @functools.wraps(callback)
    def catch_errors(frame, cache_size):
        if (
            frame.f_lasti >= 0
            or skipfiles.check(frame.f_code.co_filename)
            or config.disable
        ):
            log.debug(f"skipping {frame.f_code.co_name} {frame.f_code.co_filename}")
            return None
        if frame.f_code.co_filename == "<string>" and frame.f_code.co_name == "__new__":
            # nametuple constructor
            return None
        if config.optimize_ddp:
            ddp_module = DistributedDataParallel._get_active_ddp_module()
            if ddp_module:
                with compile_lock:
                    from torch._dynamo.backends.distributed import DDPOptimizer

                    ddp_optimizer = DDPOptimizer(
                        bucket_bytes_cap=ddp_module.bucket_bytes_cap,
                        backend_compile_fn=callback._torchdynamo_orig_callable,
                    )
                    hijacked_callback = convert_frame.convert_frame(
                        ddp_optimizer.compile_fn,
                        hooks=hooks,
                    )
                    return hijacked_callback(frame, cache_size, hooks)

        with compile_lock:
            return callback(frame, cache_size, hooks)

    catch_errors._torchdynamo_orig_callable = callback  # type: ignore[attr-defined]
    return catch_errors


def _optimize_catch_errors(
    compile_fn, hooks: Hooks, backend_ctx_ctor=null_context, dynamic=False
):
    return OptimizeContext(
        catch_errors_wrapper(compile_fn, hooks),
        backend_ctx_ctor=backend_ctx_ctor,
        first_ctx=True,
        dynamic=dynamic,
    )


def get_compiler_fn(compiler_fn):
    from .debug_utils import wrap_backend_debug

    if hasattr(compiler_fn, "compiler_name"):
        compiler_str = compiler_fn.compiler_name
    elif isinstance(compiler_fn, str):
        compiler_str = compiler_fn
    else:
        compiler_str = None
    compiler_fn = lookup_backend(compiler_fn)
    return wrap_backend_debug(compiler_fn, compiler_str)


class _NullDecorator(contextlib.nullcontext):  # type: ignore[type-arg]
    def __call__(self, fn):
        assert callable(fn)
        return fn


def check_if_dynamo_supported():
    if sys.platform == "win32":
        raise RuntimeError("Windows not yet supported for torch.compile")
    if sys.version_info >= (3, 11):
        raise RuntimeError("Python 3.11+ not yet supported for torch.compile")


def optimize(
    backend="inductor",
    *,
    nopython=False,
    guard_export_fn=None,
    guard_fail_fn=None,
    disable=False,
    dynamic=False,
):
    """
    The main entrypoint of TorchDynamo.  Do graph capture and call
    backend() to optimize extracted graphs.

    Args:
        backend: One of the two things:
            - Either, a function/callable taking a torch.fx.GraphModule and
            example_inputs and returning a python callable that runs the
            graph faster.
            One can also provide additional context for the backend, like
            torch.jit.fuser("fuser2"), by setting the backend_ctx_ctor attribute.
            See AOTAutogradMemoryEfficientFusionWithContext for the usage.
            - Or, a string backend name in `torch._dynamo.list_backends()`
        nopython: If True, graph breaks will be errors and there will
            be a single whole-program graph.
        disable: If True, turn this decorator into a no-op
        dynamic: If True, turn on dynamic shapes support

    Example Usage::

        @torch._dynamo.optimize()
        def toy_example(a, b):
            ...
    """
    check_if_dynamo_supported()
    # Note: The hooks object could be global instead of passed around, *however* that would make
    # for a confusing API usage and plumbing story wherein we nest multiple .optimize calls.
    # There is some prior art around this, w/r/t nesting backend calls are enforced to be the same
    # compiler, however, this feels onerous for callback and hooks, and it feels better to give our users an
    # easier to understand UX at the cost of a little more plumbing on our end.
    hooks = Hooks(guard_export_fn=guard_export_fn, guard_fail_fn=guard_fail_fn)
    torch._C._log_api_usage_once("torch._dynamo.optimize")
    if disable or os.environ.get("TORCHDYNAMO_DISABLE", "") == "1":
        return _NullDecorator()

    backend = get_compiler_fn(backend)

    # Find if backend has any extra context manager
    backend_ctx_ctor = getattr(backend, "backend_ctx_ctor", null_context)

    if nopython:
        return optimize_assert(
            backend,
            dynamic=dynamic,
            hooks=hooks,
        )
    return _optimize_catch_errors(
        convert_frame.convert_frame(backend, hooks=hooks),
        hooks,
        backend_ctx_ctor,
        dynamic=dynamic,
    )


# TODO(voz): Consider making "explain" output alongside a run / part of a run
@patch("torch._dynamo.symbolic_convert.explain", True)
def explain(f, *args, **kwargs):
    # TODO(voz): Do we want a decorator for this?
    from . import reset

    reset()

    out_guards = []
    graphs = []
    ops_per_graph = []
    op_count = 0
    break_reasons = []

    def dynamo_graph_accumulating_compiler(gm: torch.fx.GraphModule, example_inputs):
        nonlocal graphs
        nonlocal op_count
        nonlocal ops_per_graph

        graphs.append(gm)
        ops = []
        for node in gm.graph.nodes:
            if node.op == "call_function":
                ops.append(node.target)

        op_count += len(ops)
        ops_per_graph.append(ops)
        if gm.compile_subgraph_reason is not None:
            break_reasons.append(gm.compile_subgraph_reason)
        return gm.forward

    def guard_export_print(guards):
        nonlocal out_guards
        out_guards.append(guards)

    with patch(f"{__name__}.most_recent_backend", None):
        opt_f = optimize(
            dynamo_graph_accumulating_compiler,
            nopython=False,
            guard_export_fn=guard_export_print,
        )(f)
        # TODO(voz): We may have instances of `f` that mutate inputs, we should track sideffects and reject.
        opt_f(*args, **kwargs)

    graph_count = len(graphs)

    # For the explanation summary, dedupe reasons by the innermost stack frame and dedupe by it.
    deduped_reasons = {}
    for reason in break_reasons:
        innermost_frame = reason.user_stack[-1]
        # __repr__ uniquely identifies a FrameSummary so we can use it for deduping
        deduped_reasons[repr(innermost_frame)] = reason

    formatted_list = ""
    for idx, break_reason in enumerate(deduped_reasons.values()):
        formatted_stack = "".join(traceback.format_list(break_reason.user_stack))
        msg = f"{break_reason.reason}\n{formatted_stack}"
        formatted_list += f"{idx + 1}. {msg} \n"

    explanation = f"Dynamo produced {graph_count} graphs "
    explanation += f"with {graph_count - 1} graph break and {op_count} ops"
    explanation_verbose = explanation
    explanation_verbose += f"\n Break reasons: \n\n{formatted_list}"

    explanation_verbose += compile_times()

    # TODO(voz): Do we want a decorator for this?
    reset()
    return (
        explanation,
        out_guards,
        graphs,
        ops_per_graph,
        break_reasons,
        explanation_verbose,
    )


def export(
    f, *args, aten_graph=False, decomposition_table=None, tracing_mode="real", **kwargs
):
    check_if_dynamo_supported()
    torch._C._log_api_usage_once("torch._dynamo.export")
    if decomposition_table is not None or tracing_mode != "real":
        assert (
            aten_graph
        ), "Specifying a decomposition_table table or tracing mode is illegal without setting aten_graph=True"
    f = innermost_fn(f)

    graph = None
    out_guards = None
    graph_captured_input = None
    graph_captured_result: Optional[Tuple[torch.Tensor, ...]] = None

    def produce_matching(source_args, candidate_args):
        matched_elements_positions = []
        dict_of_source_args = dict()
        for i in range(0, len(source_args)):
            element_id = id(source_args[i])
            dict_of_source_args[element_id] = i

        for i in range(0, len(candidate_args)):
            arg = candidate_args[i]
            # 1-element tensor arg can be unspec int/float
            if isinstance(arg, torch.Tensor) and torch.numel(arg) == 1:
                if id(arg) in dict_of_source_args:
                    matched_elements_positions.append(dict_of_source_args[id(arg)])
                elif id(arg.item()) in dict_of_source_args:
                    matched_elements_positions.append(
                        dict_of_source_args[id(arg.item())]
                    )
                else:
                    raise AssertionError(
                        "Dynamo input/output is not consistent with traced input/output"
                    )
            else:
                assert (
                    id(arg) in dict_of_source_args
                ), "Dynamo input and output is a strict subset of traced input/output"
                matched_elements_positions.append(dict_of_source_args[id(arg)])

        return matched_elements_positions

    def guard_export_print(guards):
        nonlocal out_guards
        assert out_guards is None, "whole graph export entails exactly one guard export"
        out_guards = guards

    def dynamo_normalization_capturing_compiler(
        gm: torch.fx.GraphModule, example_inputs
    ):
        nonlocal graph

        assert graph is None, "whole graph export entails exactly one graph"
        graph = gm

        def result_capturing_wrapper(*graph_inputs):
            nonlocal graph_captured_result
            nonlocal graph_captured_input

            graph_captured_input = graph_inputs
            assert graph is not None
            graph_captured_result = graph(*graph_inputs)
            return graph_captured_result

        return result_capturing_wrapper

    flat_args, in_spec = pytree.tree_flatten((args, kwargs))

    remove_from_cache(f)
    with patch(f"{__name__}.most_recent_backend", None):
        opt_f = optimize_assert(
            dynamo_normalization_capturing_compiler,
            hooks=Hooks(guard_export_fn=guard_export_print, guard_fail_fn=None),
            export=True,
            dynamic=(tracing_mode == "symbolic"),
        )(f)
        # TODO(voz): We may have instances of `f` that mutate inputs, we should track sideffects and reject.
        result_traced = opt_f(*args, **kwargs)
    remove_from_cache(f)

    assert graph is not None, "whole graph export entails exactly one call"
    assert out_guards is not None, "whole graph export entails exactly one guard export"

    matched_input_elements_positions = produce_matching(flat_args, graph_captured_input)

    flat_results_traced, out_spec_traced = pytree.tree_flatten(result_traced)

    assert graph_captured_result is not None
    flat_both = list(graph_captured_result) + flat_args
    matched_output_elements_positions = produce_matching(flat_both, flat_results_traced)

    class ChangeInputOutputSignature(torch.fx.interpreter.Transformer):
        def __init__(
            self,
            m,
        ):
            super().__init__(m)
            arg_len = len(flat_args)
            self.new_args = [
                super(ChangeInputOutputSignature, self).placeholder(f"arg{i}", (), {})
                for i in range(0, arg_len)
            ]
            self.old_args_gen = (
                self.new_args[i] for i in matched_input_elements_positions
            )

        def placeholder(self, target, args, kwargs):
            arg = next(self.old_args_gen)
            if "val" in self.current_node.meta:
                arg.node.meta["val"] = self.current_node.meta["val"]
            if "tensor_dict" in self.current_node.meta:
                arg.node.meta["tensor_dict"] = self.current_node.meta["tensor_dict"]
            return arg

        def output(self, target, args, kwargs):
            dynamo_result_flat = args[0]
            lookup = [*dynamo_result_flat, *self.new_args]
            new_result_flat = [lookup[i] for i in matched_output_elements_positions]
            return super().output(target, (new_result_flat,), {})

        def run_node(self, n):
            self.current_node = n
            return super().run_node(n)

    if aten_graph:
        # Running graph with interpreter is needed for propagating the stack_trace
        def graph_with_interpreter(*args):
            with torch.fx.traceback.preserve_node_meta():
                return torch.fx.Interpreter(graph).run(*args)

        graph = make_fx(
            graph_with_interpreter,
            decomposition_table=decomposition_table,
            tracing_mode=tracing_mode,
            _allow_non_fake_inputs=True,
        )(*graph_captured_input)

    new_graph = ChangeInputOutputSignature(
        graph,
    ).transform()

    # Make dynamo graph to have same input/output spec as user code
    input_strs = [f"orig_arg_{i}" for i in range(len(args))] + list(kwargs.keys())
    new_graph.graph._codegen = _PyTreeCodeGen(
        _PyTreeInfo(
            input_strs,
            in_spec,
            out_spec_traced,
        )
    )

    new_graph.recompile()

    return (new_graph, out_guards)


def assume_constant_result(fn):
    fn._dynamo_marked_constant = True
    return fn


def optimize_assert(backend, *, hooks=Hooks(None, None), export=False, dynamic=False):
    """
    The same as `torch._dynamo.optimize(backend, nopython=True)`
    """
    backend = get_compiler_fn(backend)

    # Find if backend has any extra context manager
    backend_ctx_ctor = getattr(backend, "backend_ctx_ctor", null_context)

    return _optimize_catch_errors(
        convert_frame.convert_frame_assert(backend, export=export),
        hooks,
        backend_ctx_ctor,
        dynamic=dynamic,
    )


def run(fn=None):
    """Don't do any dynamic compiles, just use prior optimizations"""
    if fn is not None:
        fn = innermost_fn(fn)
        assert callable(fn)
        return RunOnlyContext()(fn)
    return RunOnlyContext()


def disable(fn=None):
    """Decorator and context manager to disable TorchDynamo"""
    if fn is not None:
        fn = innermost_fn(fn)
        assert callable(fn)
        return DisableContext()(fn)
    return DisableContext()


def skip(fn=None):
    """
    Skip frames associated with the function code, but still process recursively
    invoked frames
    """
    if fn is None:
        return skip
    fn = innermost_fn(fn)
    assert callable(fn)
    skip_code(fn.__code__)
    fn._torchdynamo_disable = True
    return fn


class TorchPatcher:
    @staticmethod
    @functools.lru_cache(None)
    def patch():
        # Disable TorchDynamo on some torch.* compilers generated frames
        torch.jit.trace = disable(torch.jit.trace)
        torch.jit.trace_module = disable(torch.jit.trace_module)
        torch.jit._get_trace_graph = disable(torch.jit._get_trace_graph)

        # symbolic_trace creates new frames. We disable Dynamo on such frames
        torch.fx._symbolic_trace.Tracer.trace = disable(
            torch.fx._symbolic_trace.Tracer.trace
        )

        torch.onnx.export_to_pretty_string = disable(torch.onnx.export_to_pretty_string)
        torch.distributions.Distribution.set_default_validate_args(False)

        proxy_tensor.dispatch_trace = disable(proxy_tensor.dispatch_trace)

        optimizers = [
            opt
            for opt in torch.optim.__dict__.values()
            if inspect.isclass(opt) and issubclass(opt, torch.optim.Optimizer)
        ]

        # disable dynamo for the wrapper that helps give dynamo hints about entering DDP
        if hasattr(DistributedDataParallel, "_inside_ddp_forward"):
            DistributedDataParallel._inside_ddp_forward = skip(
                DistributedDataParallel._inside_ddp_forward
            )

        from ..optim import adagrad, adam, adamax, adamw, asgd, nadam, sgd

        for opt_mod in adagrad, adam, adamax, adamw, asgd, nadam, sgd:
            multi_tensor_fn_name = f"_multi_tensor_{opt_mod.__name__.split('.')[-1]}"
            if hasattr(opt_mod, multi_tensor_fn_name):
                setattr(
                    opt_mod,
                    multi_tensor_fn_name,
                    disable(getattr(opt_mod, multi_tensor_fn_name)),
                )

        excluded_opts = {torch.optim.SparseAdam, torch.optim.RAdam, torch.optim.LBFGS}
        for opt in optimizers:
            if opt in excluded_opts:
                opt.step = disable(opt.step)

            opt._cuda_graph_capture_health_check = disable(
                opt._cuda_graph_capture_health_check
            )
            opt.zero_grad = disable(opt.zero_grad)

            if hasattr(opt, "_init_group"):
                opt._init_group = disable(opt._init_group)

            # disable any currently set hooks
            # Note: we only want to disable the profiling hook
            # which is the *last* hook applied, we want to keep the no_grad hook
            hooked = getattr(opt.step, "hooked", False)
            if hooked:
                unwrapped_step = getattr(opt.step, "__wrapped__", None)
                if unwrapped_step:
                    opt.step = unwrapped_step

            # disable future hooking
            opt.step.hooked = True

    @staticmethod
    def suppress_torch_distributed_warnings(fn):
        def inner_fn(*args, **kwargs):
            warnings.filterwarnings(
                "ignore", category=UserWarning, module="torch.distributed"
            )
            return fn(*args, **kwargs)

        return inner_fn
