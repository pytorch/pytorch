from __future__ import annotations

import contextlib
import dis
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
from os.path import dirname, join
from typing import Any, List, Optional, TYPE_CHECKING, Union
from unittest.mock import patch

import torch
import torch.fx
from torch import _guards
from torch.nn.parallel.distributed import DistributedDataParallel
from ..fx import GraphModule
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

from . import config, convert_frame, external_utils, skipfiles, utils
from .exc import ResetRequired
from .mutation_guard import install_generation_tagging_init
from .types import DynamoCallback
from .utils import compile_times

log = logging.getLogger(__name__)


always_optimize_code_objects = utils.ExactWeakKeyDictionary()
null_context = contextlib.nullcontext


# See https://github.com/python/typing/pull/240
class Unset(Enum):
    token = 0


unset = Unset.token

compile_lock = threading.RLock()
most_recent_backend: Optional[CompilerFn] = None
DONT_WRAP_FILES = {
    # For tracing into fx modules
    inspect.getsourcefile(GraphModule),
    join(dirname(dirname(__file__)), "onnx/_internal/fx/dynamo_graph_extractor.py"),
}


class OptimizedModule(torch.nn.Module):
    """
    Wraps the original nn.Module object and later patches its
    forward method to optimized self.forward method.
    """

    def __init__(self, mod: torch.nn.Module, dynamo_ctx):
        super().__init__()
        # Installs the params/buffer
        self._orig_mod = mod
        self.dynamo_ctx = dynamo_ctx
        self._initialize()

    def _initialize(self):
        # Do this stuff in constructor to lower overhead slightly
        if isinstance(self._orig_mod.forward, types.MethodType) and skipfiles.check(
            inspect.getsourcefile(self._orig_mod.forward)
        ):
            # This may be a torch.nn.* instance in skipfiles.py which
            # won't trigger a frame evaluation workaround to add an extra
            # frame we can capture
            self.forward = self.dynamo_ctx(external_utils.wrap_inline(self._orig_mod))
        else:
            # Invoke hooks outside of dynamo then pickup the inner frame
            self.forward = self.dynamo_ctx(self._orig_mod.__call__)

        if hasattr(self._orig_mod, "_initialize_hook"):
            self._forward = self.forward
            self.forward = self._call_lazy_check

    def __getstate__(self):
        state = dict(self.__dict__)
        state.pop("forward", None)
        state.pop("__call__", None)
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self._initialize()

    def __getattr__(self, name):
        if name == "_orig_mod":
            return self._modules["_orig_mod"]
        return getattr(self._orig_mod, name)

    def _call_lazy_check(self, *args, **kwargs):
        if hasattr(self._orig_mod, "_initialize_hook"):
            # In the case of a lazy module, we want to run
            # the pre-hooks which initialize it.
            # Afterwards, lazy module deletes its pre-hooks
            # to avoid treating it as lazy on subsequent recompile.
            assert len(kwargs) == 0
            self._orig_mod._infer_parameters(self._orig_mod, args)
        return self._forward(*args, **kwargs)

    def __dir__(self):
        orig_mod_attrs = self._orig_mod.__dir__()
        return orig_mod_attrs + [
            attr for attr in super().__dir__() if attr not in orig_mod_attrs
        ]


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
def enable_dynamic(enable: bool = True, export: bool = False):
    if not enable:
        yield
        return
    # dynamic=True used to mean fully dynamic. However, with automatic dynamic, the default flipped to
    # deriving dynamism. For back compat, and forward compat for when dynamic=True is default, we take
    # dynamic=True here to mean "fully dynamic from the start".
    with config.patch(assume_static_by_default=False):
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
        export=False,
        dynamic=False,
    ):
        super().__init__()
        assert callable(callback) or callback is False or callback is None
        self.callback: DynamoCallback = callback
        self.prior: Union[Unset, DynamoCallback] = unset
        self.on_enter = on_enter
        self.extra_ctx_ctor = backend_ctx_ctor
        self.first_ctx = first_ctx
        self.export = export
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
        self.dynamic_ctx = enable_dynamic(self.dynamic, self.export)
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

        try:
            filename = inspect.getsourcefile(fn)
        except TypeError:
            filename = None
        if (
            (filename is None or skipfiles.check(filename))
            and (
                getattr(fn, "__name__", "") not in ["_call_impl", "_wrapped_call_impl"]
            )
            and filename not in DONT_WRAP_FILES
        ):
            # call to a builtin without a frame for us to capture
            fn = external_utils.wrap_inline(fn)

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
            dynamic_ctx = enable_dynamic(self.dynamic, self.export)
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

    def __init__(
        self,
        callback,
        backend_ctx_ctor,
        first_ctx=False,
        *,
        export=False,
        dynamic=False,
    ):
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
            export=export,
            dynamic=dynamic,
        )


class RunOnlyContext(_TorchDynamoContext):
    def __init__(self):
        # cudagraph trees relies on generation increment
        def on_enter():
            torch._dynamo.mutation_guard.GenerationTracker.generation += 1

        super().__init__(callback=False, on_enter=on_enter)


class DisableContext(_TorchDynamoContext):
    def __init__(self):
        super().__init__(callback=None)


def first_real_inst_idx(code):
    if sys.version_info < (3, 11):
        return 0
    for inst in dis.get_instructions(code):
        if inst.opname == "RESUME":
            return inst.offset // 2
    raise RuntimeError("RESUME instruction not found in code")


def catch_errors_wrapper(callback, hooks: Hooks):
    @functools.wraps(callback)
    def catch_errors(frame, cache_size, frame_state):
        assert frame_state is not None

        if (
            # TODO: the first condition is not covered by any test
            frame.f_lasti >= first_real_inst_idx(frame.f_code)
            or skipfiles.check(frame.f_code.co_filename)
            or config.disable
        ):
            log.debug("skipping %s %s", frame.f_code.co_name, frame.f_code.co_filename)
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
                    return hijacked_callback(frame, cache_size, hooks, frame_state)

        with compile_lock:
            return callback(frame, cache_size, hooks, frame_state)

    catch_errors._torchdynamo_orig_callable = callback  # type: ignore[attr-defined]
    return catch_errors


def _optimize_catch_errors(
    compile_fn, hooks: Hooks, backend_ctx_ctor=null_context, export=False, dynamic=False
):
    return OptimizeContext(
        catch_errors_wrapper(compile_fn, hooks),
        backend_ctx_ctor=backend_ctx_ctor,
        first_ctx=True,
        export=export,
        dynamic=dynamic,
    )


def get_compiler_fn(compiler_fn):
    from .repro.after_dynamo import wrap_backend_debug

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
    if sys.version_info >= (3, 12):
        raise RuntimeError("Python 3.12+ not yet supported for torch.compile")


def is_dynamo_supported():
    try:
        check_if_dynamo_supported()
        return True
    except Exception:
        return False


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

    graphs: List[torch.fx.GraphModule] = []
    break_reasons: List[Any] = []
    op_count: int = 0
    ops_per_graph: List[torch.fx.Node] = []
    out_guards: List[_guards.Guard] = []

    def dynamo_graph_accumulating_compiler(gm: torch.fx.GraphModule, example_inputs):
        from .backends.debugging import _explain_graph_detail

        nonlocal graphs
        nonlocal op_count
        nonlocal ops_per_graph
        nonlocal break_reasons

        gm, graphs, op_count, ops_per_graph, break_reasons = _explain_graph_detail(
            gm, graphs, op_count, ops_per_graph, break_reasons
        )

        return gm.forward

    def guard_export_print(guards):
        nonlocal out_guards
        out_guards.extend(guards)

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
        msg = f"{idx + 1}. Reason: {break_reason.reason}\n   User Stack: {formatted_stack}\n"
        formatted_list += msg

    graph_break_count = graph_count - 1
    compile_time = compile_times(repr="str")

    # TODO(voz): Do we want a decorator for this?
    reset()
    from .backends.debugging import ExplainOutput

    return ExplainOutput(
        graphs,
        graph_count,
        graph_break_count,
        break_reasons,
        op_count,
        ops_per_graph,
        out_guards,
        compile_time,
    )


def optimize_assert(
    backend,
    *,
    hooks=Hooks(None, None),
    export=False,
    export_constraints=None,
    dynamic=False,
):
    """
    The same as `torch._dynamo.optimize(backend, nopython=True)`
    """
    backend = get_compiler_fn(backend)

    # Find if backend has any extra context manager
    backend_ctx_ctor = getattr(backend, "backend_ctx_ctor", null_context)

    return _optimize_catch_errors(
        convert_frame.convert_frame_assert(
            backend, export=export, export_constraints=export_constraints
        ),
        hooks,
        backend_ctx_ctor,
        export=export,
        dynamic=dynamic,
    )


class TorchPatcher:
    @staticmethod
    @functools.lru_cache(None)
    def patch():
        # A better way to disable the following would be decorate the source
        # functions with @torch._disable_dynamo. However, this causes issues
        # with torch.deploy internally.
        from .decorators import disable

        torch.jit.trace = disable(torch.jit.trace)
        torch.jit.trace_module = disable(torch.jit.trace_module)
        torch.jit._get_trace_graph = disable(torch.jit._get_trace_graph)
        torch.fx._symbolic_trace.Tracer.trace = disable(
            torch.fx._symbolic_trace.Tracer.trace
        )
        torch.distributions.Distribution.set_default_validate_args(False)

        optimizers = [
            opt
            for opt in torch.optim.__dict__.values()
            if inspect.isclass(opt) and issubclass(opt, torch.optim.Optimizer)
        ]

        # Note: this excludes the optimizers that are unsupported in excluded_opts below
        from ..optim import (
            adadelta,
            adagrad,
            adamax,
            adamw,
            asgd,
            nadam,
            rmsprop,
            rprop,
            sgd,
        )

        for opt_mod in (
            adadelta,
            adagrad,
            adamax,
            adamw,
            asgd,
            nadam,
            rmsprop,
            rprop,
            sgd,
        ):
            opt_name = opt_mod.__name__.split(".")[-1]
            multi_tensor_fn_name = f"_multi_tensor_{opt_name}"
            fused_fn_name = f"_fused_{opt_name}"
            if hasattr(opt_mod, multi_tensor_fn_name):
                setattr(
                    opt_mod,
                    multi_tensor_fn_name,
                    disable(getattr(opt_mod, multi_tensor_fn_name)),
                )

            if hasattr(opt_mod, fused_fn_name):
                setattr(
                    opt_mod, fused_fn_name, disable(getattr(opt_mod, fused_fn_name))
                )

        # Note: we don't support sparsity, data-dependent control, or tracing through backwards
        excluded_opts = {torch.optim.SparseAdam, torch.optim.RAdam, torch.optim.LBFGS}
        for opt in optimizers:
            if opt in excluded_opts:
                opt.step = disable(opt.step)

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

        torch._dynamo.variables.lists._register_dynamo_list_to_tree_spec()
        torch._dynamo.variables.lists._register_dynamo_tuple_to_tree_spec()
        torch._dynamo.variables.dicts._register_dynamo_dict_to_tree_spec()

    @staticmethod
    def suppress_torch_distributed_warnings(fn):
        def inner_fn(*args, **kwargs):
            warnings.filterwarnings(
                "ignore", category=UserWarning, module="torch.distributed"
            )
            return fn(*args, **kwargs)

        return inner_fn
