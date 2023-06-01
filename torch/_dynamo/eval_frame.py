from __future__ import annotations

import contextlib
import dataclasses
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
import weakref
from enum import Enum
from os.path import dirname, join
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TYPE_CHECKING, Union
from unittest.mock import patch

import torch
import torch.fx
import torch.utils._pytree as pytree
import torch.utils.checkpoint
from torch import _guards
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.graph import _PyTreeCodeGen, _PyTreeInfo
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
from .exc import CondOpArgsMismatchError, ResetRequired, UserError, UserErrorType
from .mutation_guard import install_generation_tagging_init
from .types import DynamoCallback
from .utils import compile_times

log = logging.getLogger(__name__)

from torch._dispatch.python import enable_python_dispatcher
from torch.fx.experimental import proxy_tensor

always_optimize_code_objects = utils.ExactWeakKeyDictionary()
null_context = contextlib.nullcontext


import sympy

from torch.fx.experimental.symbolic_shapes import (
    ConstraintViolationError,
    StrictMinMaxConstraint,
)
from torch.utils._sympy.value_ranges import ValueRanges


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
        return self._orig_mod.__dir__()


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
def enable_dynamic(enable: bool = True, export: bool = False):
    if not enable:
        yield
        return
    with config.patch(dynamic_shapes=True):
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
        if gm.compile_subgraph_reason.graph_break:
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


@dataclasses.dataclass
class ConstraintTarget:
    """
    This represents input tensor dimensions.  Don't create this
    class directly; instead, use :func:`torch._export.dynamic_dim`.
    """

    w_tensor: weakref.ReferenceType[torch.Tensor]
    # TODO: We don't need t_id; we can get it off of w_tensor
    t_id: int
    dim: int


@dataclasses.dataclass
class Constraint(ConstraintTarget):
    """
    This represents constraints on input tensor dimensions, e.g., requiring
    them to be fully polymorphic or within some range.  Don't create this
    class directly; instead, use :func:`torch._export.dynamic_dim`.
    """

    # NOTE(avik): In the future, this could be Union[StrictMinMaxConstraint, <other kinds>]
    constraint_range: StrictMinMaxConstraint
    # Represent that `constraint_range` is shared with another ConstraintTarget, which
    # typically arises because of a specified equality with another dynamic dimension.
    shared: Optional[ConstraintTarget] = None

    def _clone_with_range(self, lower=2, upper=sympy.oo):
        constraint_range = StrictMinMaxConstraint(
            vr=self.constraint_range.vr & ValueRanges(lower=lower, upper=upper),
            warn_only=False,
        )
        return Constraint(
            self.w_tensor, self.t_id, self.dim, constraint_range, self.shared
        )

    def __ge__(self, lower):
        return self._clone_with_range(lower=lower)

    def __gt__(self, lower):
        return self._clone_with_range(lower=lower + 1)

    def __le__(self, upper):
        return self._clone_with_range(upper=upper)

    def __lt__(self, upper):
        return self._clone_with_range(upper=upper - 1)

    def __bool__(self):
        # NOTE(avik): We do not support compound expressions like a <= x <= b.
        # This is because Python implicitly desugars them into bool(a <= x) and bool(x <= b),
        # and moreover, enforces that any overload of __bool__ must return True or False.
        # FWIW, sympy also raises TypeError in this case.
        raise TypeError(
            "Cannot determine truth value of Constraint. "
            "If you are trying to combine Constraints with logical connectives, "
            "you can specify them separately instead."
        )

    @property
    def serializable_spec(self):
        # We need a serialization compatible format of the constraint so that it
        # can be savedin the graph module w/o breaking the module serialization.
        # The saved constraints will be used directly for the post-exporting pass
        # that converts constraints to runtime assertion. The saved constraints
        # will not be saved in the serialized module.
        # TODO: A better way is needed. Currently we use 't_id' to map the constraint,
        # which is not reliable
        return {
            "t_id": self.t_id,
            "dim": self.dim,
            "min": self.constraint_range.vr.lower,
            "max": self.constraint_range.vr.upper,
            "shared": (
                None
                if self.shared is None
                else {
                    "t_id": self.shared.t_id,
                    "dim": self.shared.dim,
                }
            ),
        }

    def __eq__(self, other):
        constraint_range = StrictMinMaxConstraint(
            vr=self.constraint_range.vr & other.constraint_range.vr,
            warn_only=False,
        )
        return Constraint(
            self.w_tensor,
            self.t_id,
            self.dim,
            constraint_range,
            shared=ConstraintTarget(other.w_tensor, other.t_id, other.dim),
        )


class FlattenInputOutputSignature(torch.fx.interpreter.Transformer):
    def __init__(
        self,
        m: torch.fx.GraphModule,
        flat_args: Tuple[Any],
        matched_input_elements_positions: List[int],
        matched_output_elements_positions: List[int],
        example_fake_inputs: List[torch.Tensor],
    ):
        super().__init__(m)

        matched_input_elements_to_fake = {
            val: example_fake_inputs[ix]
            for ix, val in enumerate(matched_input_elements_positions)
        }
        fake_mode = _guards.detect_fake_mode(example_fake_inputs)

        self.new_args = []
        for i in range(0, len(flat_args)):
            arg = super(FlattenInputOutputSignature, self).placeholder(
                f"arg{i}", (), {}
            )
            if i in matched_input_elements_to_fake:
                arg.node.meta["val"] = matched_input_elements_to_fake[i]
            else:
                # Fill node.mata["val"] with faketensor from the input,
                # if it's not found in matched_input_elements_positions
                if fake_mode is not None and isinstance(flat_args[i], torch.Tensor):
                    arg.node.meta["val"] = fake_mode.from_tensor(flat_args[i])
            self.new_args.append(arg)
        self.old_args_gen = (self.new_args[i] for i in matched_input_elements_positions)
        self.matched_output_elements_positions = matched_output_elements_positions

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
        new_result_flat = [lookup[i] for i in self.matched_output_elements_positions]
        return super().output(target, (new_result_flat,), {})

    def run_node(self, n):
        self.current_node = n
        r = super().run_node(n)
        if "val" in self.current_node.meta:
            r.node.meta["val"] = self.current_node.meta["val"]
        return r


def export(
    f: Callable[..., Any],
    *args,
    aten_graph: bool = False,
    pre_autograd: bool = False,
    decomposition_table: Optional[
        Dict[torch._ops.OpOverload, Callable[..., Any]]
    ] = None,
    tracing_mode: str = "symbolic",
    constraints: Optional[List[Constraint]] = None,
    assume_static_by_default: bool = False,
    functionalize: bool = False,
    **kwargs,
) -> Tuple[torch.fx.GraphModule, Set[_guards.Guard]]:
    """
    Export an input function f to a format that can be executed outside of PyTorch using the FX graph.

    Args:
        f (callable): A PyTorch function to be exported.

        *args: Variable length argument list to be passed to the function f.

        aten_graph (bool): If True, exports a graph with ATen operators.
        If False, exports a graph with Python operators. Default is False.

        pre_autograd (bool): If True, exports a graph with ATen operators,
        but before autograd has run. This can be useful if you want to apply further tranformations
        on a graph before running it through autograd.
        This flag is only valid if aten_graph=True is set.
        Default is False.

        decomposition_table (dict): A dictionary that maps operators to their decomposition functions.
        Required if aten_graph or tracing_mode is specified. Default is None.

        tracing_mode (str): If "symbolic", turn on dynamic shapes support. Default is "symbolic".

        functionalize (bool): If True, the resulting aten graph module will be functional. You will need to
        set aten_graph=True to see the effect. By default, this flag will be false.

        **kwargs: Arbitrary keyword arguments to be passed to the function f.

    Returns:
        A tuple of (graph, guards)
        Graph: An FX graph representing the execution of the input PyTorch function with the provided arguments and options.
        Guards: The guards we accumulated during tracing f above

    Raises:
        AssertionError: If decomposition_table is specified without setting aten_graph=True,
        or if graph breaks during tracing in export.

        AssertionError: If Dynamo input and output is not consistent with traced input/output.

    Note - this headerdoc was authored by ChatGPT, with slight modifications by the author.
    """
    check_if_dynamo_supported()
    torch._C._log_api_usage_once("torch._dynamo.export")
    if decomposition_table is not None:
        assert (
            aten_graph
        ), "Specifying a decomposition_table table or tracing mode is illegal without setting aten_graph=True"
    if pre_autograd:
        assert aten_graph, "pre_autograd=True can only be used when aten_graph=True"
    f = innermost_fn(f)
    call_to_inspect = f.forward if isinstance(f, torch.nn.Module) else f
    original_signature = inspect.signature(call_to_inspect)

    if functionalize and not aten_graph:
        raise UserError(
            UserErrorType.ANTI_PATTERN,
            "TorchDynamo won't functionalize non-aten graphs. Please set `functionalize` to true",
        )

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

    def guard_export_print(guards: Set[_guards.Guard]):
        nonlocal out_guards
        assert out_guards is None, "whole graph export entails exactly one guard export"
        out_guards = guards

    fake_mode = None
    example_inputs = []

    def dynamo_normalization_capturing_compiler(
        gm: torch.fx.GraphModule, inner_example_inputs
    ):
        nonlocal graph
        assert (
            graph is None
        ), "Tried to emit a second graph during export. Tracing through 'f' must produce a single graph."
        graph = gm

        nonlocal fake_mode, example_inputs
        fake_mode = _guards.detect_fake_mode(inner_example_inputs)
        example_inputs = inner_example_inputs

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
    constraint_violation_error = None
    with patch(f"{__name__}.most_recent_backend", None), config.patch(
        summarize_dim_constraints=True,
        specialize_int=True,
        assume_static_by_default=assume_static_by_default,
    ):
        opt_f = optimize_assert(
            dynamo_normalization_capturing_compiler,
            hooks=Hooks(
                guard_export_fn=guard_export_print,
                guard_fail_fn=None,
            ),
            export=True,
            export_constraints=constraints,
            dynamic=(tracing_mode == "symbolic"),
        )(f)
        # TODO(voz): We may have instances of `f` that mutate inputs, we should track sideffects and reject.
        try:
            result_traced = opt_f(*args, **kwargs)
        except ConstraintViolationError as e:
            constraint_violation_error = e
    remove_from_cache(f)

    if (
        shape_env := getattr(fake_mode, "shape_env", None)
    ) is not None and not skipfiles.check(inspect.getsourcefile(call_to_inspect)):
        dim_constraints = shape_env.dim_constraints
        assert dim_constraints is not None
        dim_constraints.solve()
        msg = dim_constraints.prettify_results(original_signature)
        if constraint_violation_error:
            constraint_violation_error.args = (
                constraint_violation_error.args[0] + msg,
            )
        else:
            log.info(
                "Summary of dimension constraints:%s",
                msg,
            )

        # Error if we have any constraints on static values
        for k in shape_env.var_to_range.keys():
            if isinstance(k, sympy.Integer):
                constraint_violation_error = ConstraintViolationError(
                    f"{''.join(traceback.format_list(shape_env.var_to_stack[k]))}\n"
                    "It appears that you're trying to set a constraint on a "
                    f"value which we evaluated to have a static value of {k}. "
                    "Scroll up to see where this constraint was set."
                )
    if constraint_violation_error:
        raise constraint_violation_error

    assert (
        graph is not None
    ), "Failed to produce a graph during tracing. Tracing through 'f' must produce a single graph."
    assert out_guards is not None, "Failed to produce guards during tracing"
    assert fake_mode is not None

    matched_input_elements_positions = produce_matching(flat_args, graph_captured_input)

    # NB: This is mostly hitting the cache; Dynamo already converted these
    example_fake_inputs = [fake_mode.from_tensor(t) for t in example_inputs]
    flat_results_traced, out_spec_traced = pytree.tree_flatten(result_traced)

    assert graph_captured_result is not None
    flat_both = list(graph_captured_result) + flat_args
    matched_output_elements_positions = produce_matching(flat_both, flat_results_traced)

    if aten_graph:
        memo: Dict[torch.Tensor, torch.Tensor] = {}

        def to_fun(t):
            if isinstance(t, torch.Tensor):
                if t in memo:
                    return memo[t]
                r = torch._to_functional_tensor(t, mirror_autograd_meta=True)
                memo[t] = r
                return r
            else:
                return t

        def from_fun(t):
            if not isinstance(t, torch.Tensor) or not torch._is_functional_tensor(t):
                return t
            torch._sync(t)
            return torch._from_functional_tensor(t)

        # Running graph with interpreter is needed for propagating the stack_trace
        def graph_with_interpreter(*args):
            with torch.fx.traceback.preserve_node_meta():
                if functionalize:
                    torch._enable_functionalization(reapply_views=True)
                    try:
                        return pytree.tree_map(
                            from_fun,
                            torch.fx.Interpreter(graph).run(
                                *pytree.tree_map(to_fun, args)
                            ),
                        )
                    finally:
                        torch._disable_functionalization()
                else:
                    return torch.fx.Interpreter(graph).run(*args)

        with enable_python_dispatcher(), fake_mode:
            try:
                graph = make_fx(
                    graph_with_interpreter,
                    decomposition_table=decomposition_table,
                    tracing_mode="real",
                    _allow_non_fake_inputs=True,
                    pre_autograd=pre_autograd,
                )(*example_fake_inputs)
            except CondOpArgsMismatchError as e:
                # Wrap the internal error to the user-facing error
                raise UserError(UserErrorType.DYNAMIC_CONTROL_FLOW, str(e))

    new_graph = FlattenInputOutputSignature(
        graph,
        flat_args,
        matched_input_elements_positions,
        matched_output_elements_positions,
        example_fake_inputs,
    ).transform()

    # Store constraints and inputs as metadata for user passes, e.g. turn constraints to runtime check
    new_graph.meta["input_shape_constraints"] = (
        [constraint.serializable_spec for constraint in constraints]
        if constraints
        else []
    )

    def signature_to_fullargspec(sig: inspect.Signature):
        # Get a list of Parameter objects from the Signature object
        params = list(sig.parameters.values())
        # Separate positional arguments, keyword-only arguments and varargs/varkw
        args = [
            p.name for p in params if p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
        ]
        kwonlyargs = [
            p.name for p in params if p.kind == inspect.Parameter.KEYWORD_ONLY
        ]
        varargs = next(
            (p.name for p in params if p.kind == inspect.Parameter.VAR_POSITIONAL), None
        )
        varkw = next(
            (p.name for p in params if p.kind == inspect.Parameter.VAR_KEYWORD), None
        )
        # Get default values for positional arguments and keyword-only arguments
        defaults = tuple(
            p.default
            for p in params
            if p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
            and p.default is not inspect.Parameter.empty
        )
        kwonlydefaults = {
            p.name: p.default
            for p in params
            if p.kind == inspect.Parameter.KEYWORD_ONLY
            and p.default is not inspect.Parameter.empty
        }
        # Get annotations for parameters and return value
        annotations = {}
        if sig.return_annotation:
            annotations = {"return": sig.return_annotation}
        for parameter in params:
            annotations[parameter.name] = parameter.annotation
        # Return a FullArgSpec object with the extracted attributes
        return inspect.FullArgSpec(
            args, varargs, varkw, defaults, kwonlyargs, kwonlydefaults, annotations
        )

    # Make dynamo graph to have same input/output spec as user code
    def argument_names(f: Callable[..., Any], *args, **kwargs) -> List[str]:
        fullargspec = signature_to_fullargspec(original_signature)

        # 1. Map `args` 1-to-1 to positional arguments in original signature.
        input_strs = fullargspec.args[: len(args)]

        if len(args) > len(fullargspec.args):
            # 2. If there are more arguments left in `args`, they map to varargs in original
            # signature. Assign names as {varargs}_0, {varargs}_1, ...
            assert fullargspec.varargs is not None, "More arguments than expected"
            input_strs += [
                f"{fullargspec.varargs}_{i}"
                for i in range(0, len(args) - len(input_strs))
            ]
        elif len(args) < len(fullargspec.args):
            # 3. If there are fewer arguments in `args` than `fullargspec.args`,
            # it implies these are arguments either with default values, or provided in
            # `kwargs`. The former can be safely ignored. Because Dynamo.export does not
            # export them as part of the function signature. The latter will be handled
            # in the next step.
            for unprovided_arg in fullargspec.args[
                len(args) : -len(fullargspec.defaults or [])
            ]:
                assert unprovided_arg in kwargs, f"Missing argument {unprovided_arg}"

        # 4. Keyword arguments provided in `kwargs`.
        input_strs += list(kwargs.keys())

        # 5. Keyword-only arguments with default values if not provided are not exported
        # as part of the function signature.
        for kwonly_arg in fullargspec.kwonlyargs:
            kwonlydefaults = fullargspec.kwonlydefaults or {}
            assert (
                kwonly_arg in kwargs or kwonly_arg in kwonlydefaults
            ), f"Missing keyword only argument {kwonly_arg}"

        return input_strs

    new_graph.graph._codegen = _PyTreeCodeGen(
        _PyTreeInfo(
            argument_names(f, *args, **kwargs),
            in_spec,
            out_spec_traced,
        )
    )

    new_graph.recompile()
    return (new_graph, out_guards)


def assume_constant_result(fn):
    fn._dynamo_marked_constant = True
    return fn


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


def run(fn=None):
    """Don't do any dynamic compiles, just use prior optimizations"""
    if fn is not None:
        fn = innermost_fn(fn)
        assert callable(fn)
        return RunOnlyContext()(fn)
    return RunOnlyContext()


def disable(fn=None, recursive=True):
    """
    Decorator and context manager to disable TorchDynamo

    If recursive=True, Dynamo is completely skipped on the decorated function
    frame as well as the recursively invoked functions.

    If recursive=False, Dynamo skips frames associated with the function code,
    but still process recursively invoked frames.
    """
    if recursive:
        if fn is not None:
            fn = innermost_fn(fn)
            assert callable(fn)
            return DisableContext()(fn)
        return DisableContext()
    else:
        return skip(fn)


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
            DistributedDataParallel._inside_ddp_forward = disable(
                DistributedDataParallel._inside_ddp_forward, recursive=False
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

        # TorchDynamo does not step inside utils.checkpoint function.  The flow
        # looks likes this
        #  1) TorchDynamo tries to wrap utils.checkpoint in a HigherOrderOp by
        #     speculatively checking if the forward function is safe to trace.
        #  2) If yes, then Dynamo-generated Fx graph has the wrapped higher
        #     order op. As a result, TorchDynamo does not look inside utils.checkpoint.
        #  3) If not, then TorchDynamo falls back to eager by performing a graph
        #     break. And here, the following disable wrapper ensures that
        #     TorchDynamo does not trigger again on the frames created by
        #     utils.checkpoint innards.
        torch.utils.checkpoint.checkpoint = disable(torch.utils.checkpoint.checkpoint)

    @staticmethod
    def suppress_torch_distributed_warnings(fn):
        def inner_fn(*args, **kwargs):
            warnings.filterwarnings(
                "ignore", category=UserWarning, module="torch.distributed"
            )
            return fn(*args, **kwargs)

        return inner_fn
