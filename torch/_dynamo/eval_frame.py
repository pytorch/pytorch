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
from collections import namedtuple
from enum import Enum
from os.path import dirname, join
from typing import (
    Any,
    Callable,
    Dict,
    List,
    NamedTuple,
    Optional,
    Set,
    Tuple,
    TYPE_CHECKING,
    Union,
)
from unittest.mock import patch

import torch
import torch.fx
import torch.utils._pytree as pytree
import torch.utils.checkpoint
from torch import _guards
from torch._subclasses import fake_tensor
from torch.export import Constraint
from torch.fx.experimental.proxy_tensor import make_fx, maybe_disable_fake_tensor_mode
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
from torch.utils._python_dispatch import _disable_current_modes

always_optimize_code_objects = utils.ExactWeakKeyDictionary()
null_context = contextlib.nullcontext


import sympy

from torch.fx.experimental.symbolic_shapes import ConstraintViolationError


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


CacheEntry = namedtuple("CacheEntry", "check_fn, code")


def _debug_get_cache_entry_list(code: types.CodeType) -> List[CacheEntry]:
    """
    Given a code object, retrieve the cache entries stored in this code.
    """
    cache_list = torch._C._dynamo.eval_frame._debug_get_cache_entry_list(code)
    return list(map(CacheEntry._make, cache_list))


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
        from . import reset  # type: ignore[attr-defined]

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
def enable_dynamic(enable: Optional[bool] = None, export: bool = False):
    if enable is None:
        yield
    elif enable:
        # Assume everything is dynamic by deafult
        with config.patch(assume_static_by_default=False):
            yield
    else:
        with config.patch(
            automatic_dynamic_shapes=False, assume_static_by_default=True
        ):
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
        dynamic=None,
        compiler_config=None,
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
        self.compiler_config = compiler_config
        patch_fn()

    def __enter__(self):
        if config.raise_on_ctx_manager_usage:
            raise RuntimeError(
                "torch._dynamo.optimize(...) is used with a context manager. "
                "Please refer to https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html "
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
        # public api for compiler config/options
        def get_compiler_config():
            return self.compiler_config

        fn = innermost_fn(fn)
        # Optimize the forward method of torch.nn.Module object
        if isinstance(fn, torch.nn.Module):
            mod = fn
            new_mod = OptimizedModule(mod, self)
            # Save the function pointer to find the original callable while nesting
            # of decorators.
            new_mod._torchdynamo_orig_callable = mod.forward

            # when compiling torch.nn.Module,
            # provide public api OptimizedModule.get_compiler_config()
            assert not hasattr(new_mod, "get_compiler_config")
            new_mod.get_compiler_config = get_compiler_config  # type: ignore[attr-defined]

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

        # when compiling user function instead of nn.Module
        # provide public api _fn.get_compiler_config()
        assert not hasattr(_fn, "get_compiler_config")
        _fn.get_compiler_config = get_compiler_config  # type: ignore[attr-defined]

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
        dynamic=None,
        compiler_config=None,
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
            compiler_config=compiler_config,
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
    def catch_errors(frame, cache_entry, frame_state):
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
                    return hijacked_callback(frame, cache_entry, hooks, frame_state)

        with compile_lock, _disable_current_modes():
            return callback(frame, cache_entry, hooks, frame_state)

    catch_errors._torchdynamo_orig_callable = callback  # type: ignore[attr-defined]
    return catch_errors


def _optimize_catch_errors(
    compile_fn,
    hooks: Hooks,
    backend_ctx_ctor=null_context,
    export=False,
    dynamic=None,
    compiler_config=None,
):
    return OptimizeContext(
        catch_errors_wrapper(compile_fn, hooks),
        backend_ctx_ctor=backend_ctx_ctor,
        first_ctx=True,
        export=export,
        dynamic=dynamic,
        compiler_config=compiler_config,
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
    dynamic=None,
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
        dynamic: If True, upfront compile as dynamic a kernel as possible.  If False,
            disable all dynamic shapes support (always specialize).  If None, automatically
            detect when sizes vary and generate dynamic kernels upon recompile.

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
        compiler_config=backend.get_compiler_config()
        if hasattr(backend, "get_compiler_config")
        else None,
    )


# TODO(voz): Consider making "explain" output alongside a run / part of a run
@patch("torch._dynamo.symbolic_convert.explain", True)
def explain(f, *extra_args, **extra_kwargs):
    def inner(*args, **kwargs):
        # TODO(voz): Do we want a decorator for this?
        from . import reset  # type: ignore[attr-defined]

        reset()

        graphs: List[torch.fx.GraphModule] = []
        break_reasons: List[Any] = []
        op_count: int = 0
        ops_per_graph: List[torch.fx.Node] = []
        out_guards: List[_guards.Guard] = []

        def dynamo_graph_accumulating_compiler(
            gm: torch.fx.GraphModule, example_inputs
        ):
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

    if extra_args or extra_kwargs:
        warnings.warn(
            "explain(f, *args, **kwargs) is deprecated, use explain(f)(*args, **kwargs) instead.  "
            "If you don't migrate, we may break your explain call in the future if your user defined kwargs "
            "conflict with future kwargs added to explain(f)."
        )
        return inner(*extra_args, **extra_kwargs)
    else:
        return inner


class FlattenInputOutputSignature(torch.fx.interpreter.Transformer):
    def __init__(
        self,
        m: torch.fx.GraphModule,
        flat_args: Tuple[Any],
        matched_input_elements_positions: List[int],
        matched_output_elements_positions: List[int],
        example_fake_inputs: List[torch.Tensor],
        fake_mode: Optional[fake_tensor.FakeTensorMode] = None,
    ):
        super().__init__(m)

        matched_input_elements_to_fake = {
            val: example_fake_inputs[ix]
            for ix, val in enumerate(matched_input_elements_positions)
        }

        self.new_args = []
        for i in range(0, len(flat_args)):
            arg = super().placeholder(f"arg{i}", (), {})
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


class ExportResult(NamedTuple):
    graph_module: torch.fx.GraphModule
    guards: Set[_guards.Guard]
    # NB: Do not add new fields without overriding __iter__; people are
    # destructuring so it is BC-breaking


def check_signature_rewritable(graph):
    input_errors = []
    for node in graph.graph.nodes:
        if node.op == "placeholder":
            assert hasattr(node, "_dynamo_source")
            source = node._dynamo_source
            user_stacks = graph._source_to_user_stacks.get(source)
            if user_stacks is None:
                continue
            assert len(user_stacks) > 0
            # In some cases we may not have a useful stack.  Look for a
            # useful stack
            stack = None
            for s in user_stacks:
                if len(s) == 0:
                    continue
                stack = s
                break
            if stack is None:
                msg = f"{source.name()}, a closed over free variable"
            else:
                tb = "".join(traceback.format_list(stack))
                extra = ""
                if len(user_stacks) > 1:
                    extra = f"(elided {len(user_stacks)-1} more accesses)"
                msg = f"{source.name()}, accessed at:\n{tb}{extra}"
            # TODO: option to print ALL of the stack traces at once
            input_errors.append(msg)

    if input_errors:
        raise UserError(
            UserErrorType.INVALID_INPUT,
            "Cannot export model which references tensors that are neither "
            "buffers/parameters/constants nor are direct inputs.  For each tensor, if you'd "
            "like this tensor to be an explicit input, add it as a dummy argument "
            "to the top-level model definition you are exporting; if you would "
            "like its value to be embedded as an exported constant, wrap its access "
            "in a function marked with @assume_constant_result.\n\n"
            + "\n\n".join(input_errors),
        )


def rewrite_signature(
    f_sig,
    graph,
    fake_mode,
    flat_args,
    in_spec,
    example_fake_inputs,
    graph_captured_input,
    graph_captured_output,
    dynamo_traced_result,
):
    orig_args, orig_kwargs = pytree.tree_unflatten(flat_args, in_spec)

    def produce_matching(source_args, candidate_args, loc):
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
                        f"Dynamo {loc} is not consistent with traced {loc}"
                    )
            else:
                assert (
                    id(arg) in dict_of_source_args
                ), f"Dynamo {loc} is a strict subset of traced {loc}"
                matched_elements_positions.append(dict_of_source_args[id(arg)])

        return matched_elements_positions

    matched_input_elements_positions = produce_matching(
        flat_args, graph_captured_input, "input"
    )

    flat_results_traced, out_spec_traced = pytree.tree_flatten(dynamo_traced_result)

    assert graph_captured_output is not None
    flat_both = list(graph_captured_output) + flat_args
    matched_output_elements_positions = produce_matching(
        flat_both, flat_results_traced, "output"
    )

    new_graph = FlattenInputOutputSignature(
        graph,
        flat_args,
        matched_input_elements_positions,
        matched_output_elements_positions,
        example_fake_inputs,
        fake_mode,
    ).transform()

    # Make dynamo graph to have same input/output spec as user code
    def argument_names(f_sig, args, kwargs) -> List[str]:
        def signature_to_fullargspec(sig: inspect.Signature):
            # Get a list of Parameter objects from the Signature object
            params = list(sig.parameters.values())
            # Separate positional arguments, keyword-only arguments and varargs/varkw
            args = [
                p.name
                for p in params
                if p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
            ]
            kwonlyargs = [
                p.name for p in params if p.kind == inspect.Parameter.KEYWORD_ONLY
            ]
            varargs = next(
                (p.name for p in params if p.kind == inspect.Parameter.VAR_POSITIONAL),
                None,
            )
            varkw = next(
                (p.name for p in params if p.kind == inspect.Parameter.VAR_KEYWORD),
                None,
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

        fullargspec = signature_to_fullargspec(f_sig)

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
            argument_names(f_sig, orig_args, orig_kwargs),
            in_spec,
            out_spec_traced,
        )
    )
    new_graph.recompile()
    return new_graph


def export(
    f: Callable[..., Any],
    *extra_args,
    aten_graph: bool = False,
    pre_dispatch: bool = False,
    decomposition_table: Optional[
        Dict[torch._ops.OpOverload, Callable[..., Any]]
    ] = None,
    tracing_mode: str = "symbolic",
    constraints: Optional[List[Constraint]] = None,
    assume_static_by_default: bool = False,
    same_signature: bool = True,
    **extra_kwargs,
) -> Callable[..., ExportResult]:
    """
    Export an input function f to a format that can be executed outside of PyTorch using the FX graph.

    Args:
        f (callable): A PyTorch function to be exported.

        aten_graph (bool): If True, exports a graph with ATen operators.
        If False, exports a graph with Python operators. Default is False.

        pre_dispatch (bool): If True, exports a graph with ATen operators,
        but before any logic in the PyTorch dispatcher has run.
        This can be useful if you want to apply further transformations on a graph before running it
        through autograd, autocast, or any other functionalities that are integrated into the dispatcher.
        This flag is only valid if aten_graph=True is set.
        Default is False.

        decomposition_table (dict): A dictionary that maps operators to their decomposition functions.
        Required if aten_graph or tracing_mode is specified. Default is None.

        tracing_mode (str): If "symbolic", turn on dynamic shapes support. Default is "symbolic".

        same_signature (bool): If True, rewrite the returned graph's signature to be the same as f.

    Returns:
        A function that given args and kwargs, returns a tuple of (graph, guards)
        Graph: An FX graph representing the execution of the input PyTorch function with the provided arguments and options.
        Guards: The guards we accumulated during tracing f above

    Raises:
        AssertionError: If decomposition_table is specified without setting aten_graph=True,
        or if graph breaks during tracing in export.

        AssertionError: If Dynamo input and output is not consistent with traced input/output.

    Note - this headerdoc was authored by ChatGPT, with slight modifications by the author.
    """
    # Deal with "local variable referenced before assignment"
    _f = f
    _assume_static_by_default = assume_static_by_default

    def inner(*args, **kwargs):
        f = _f
        assume_static_by_default = _assume_static_by_default
        check_if_dynamo_supported()
        torch._C._log_api_usage_once("torch._dynamo.export")
        if decomposition_table is not None:
            assert (
                aten_graph
            ), "Specifying a decomposition_table table or tracing mode is illegal without setting aten_graph=True"
        if pre_dispatch:
            assert aten_graph, "pre_dispatch=True can only be used when aten_graph=True"
        f = innermost_fn(f)
        call_to_inspect = f.forward if isinstance(f, torch.nn.Module) else f
        original_signature = inspect.signature(call_to_inspect)
        graph = None
        out_guards = None
        graph_captured_input = None
        graph_captured_result: Optional[Tuple[torch.Tensor, ...]] = None
        fake_mode = None

        def guard_export_print(guards: Set[_guards.Guard]):
            nonlocal out_guards
            assert (
                out_guards is None
            ), "whole graph export entails exactly one guard export"
            out_guards = guards

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
            # NB: do NOT pass inner_example_inputs here, we are detecting the
            # Dynamo allocated fake mode, which should be DISTINCT from a
            # potential outer ambient fake mode which the user provided.
            # example_inputs is always the user specified inputs, so they
            # would have the wrong fake mode attached to them
            fake_mode = _guards.detect_fake_mode()
            example_inputs = inner_example_inputs

            def result_capturing_wrapper(*graph_inputs):
                nonlocal graph_captured_result
                nonlocal graph_captured_input

                graph_captured_input = graph_inputs
                assert graph is not None

                named_parameters = dict(graph.named_parameters(remove_duplicate=False))
                named_buffers = dict(graph.named_buffers(remove_duplicate=False))

                ambient_fake_mode = (
                    _guards.detect_fake_mode(graph_inputs)
                    if _guards.detect_fake_mode(graph_inputs) is not None
                    else fake_mode
                )

                with ambient_fake_mode, enable_python_dispatcher():
                    params_and_buffers = {
                        **dict(named_parameters),
                        **dict(named_buffers),
                    }
                    fake_params_buffers = dict()

                    for name, value in params_and_buffers.items():
                        fake_params_buffers[name] = ambient_fake_mode.from_tensor(
                            value, static_shapes=True
                        )

                    fake_graph_inputs = pytree.tree_map(
                        ambient_fake_mode.from_tensor, graph_inputs
                    )
                    graph_captured_result = torch.func.functional_call(
                        graph, fake_params_buffers, fake_graph_inputs
                    )

                return graph_captured_result

            return result_capturing_wrapper

        # Note: This is needed by rewrite_signature. We need to put it before
        # optimize_assert since user program may mutate the inputs.
        flat_args, in_spec = pytree.tree_flatten((args, kwargs))

        remove_from_cache(f)
        constraint_violation_error = None
        if tracing_mode != "symbolic":
            assume_static_by_default = True
        with patch(f"{__name__}.most_recent_backend", None), config.patch(
            specialize_int=True,
            assume_static_by_default=assume_static_by_default,
            automatic_dynamic_shapes=False,
            capture_dynamic_output_shape_ops=True,
            capture_scalar_outputs=True,
        ):
            opt_f = optimize_assert(
                dynamo_normalization_capturing_compiler,
                hooks=Hooks(
                    guard_export_fn=guard_export_print,
                    guard_fail_fn=None,
                ),
                export=True,
                export_constraints=constraints,
            )(f)
            # TODO(voz): We may have instances of `f` that mutate inputs, we should track sideffects and reject.
            try:
                result_traced = opt_f(*args, **kwargs)
            except ConstraintViolationError as e:
                constraint_violation_error = e
        remove_from_cache(f)

        if (
            (shape_env := getattr(fake_mode, "shape_env", None)) is not None
            and (dim_constraints := shape_env.dim_constraints) is not None
            and not skipfiles.check(inspect.getsourcefile(call_to_inspect))
        ):
            dim_constraints.solve()
            msg = dim_constraints.prettify_results(original_signature)
            forced_specializations = dim_constraints.forced_specializations()
            if forced_specializations:
                msg = (
                    "Some dynamic dimensions need to be specialized because "
                    "the constraints inferred for them are too complex to specify.\n"
                    f"{forced_specializations}\n{msg}"
                )
            if constraint_violation_error:
                constraint_violation_error.args = (
                    constraint_violation_error.args[0] + msg,
                )
            else:
                if forced_specializations:
                    constraint_violation_error = ConstraintViolationError(msg)
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
        assert hasattr(graph, "_source_to_user_stacks")
        assert out_guards is not None, "Failed to produce guards during tracing"
        assert fake_mode is not None

        # This check need to happend before aten_graph
        # because placeholder's _source_node attribute is not preserved by make_fx
        if same_signature:
            check_signature_rewritable(graph)

        # NB: This is mostly hitting the cache; Dynamo already converted these
        example_fake_inputs = [fake_mode.from_tensor(t) for t in example_inputs]

        if aten_graph:
            # Running graph with interpreter is needed for propagating the stack_trace
            def graph_with_interpreter(*args):
                with torch.fx.traceback.preserve_node_meta():
                    return torch.fx.Interpreter(graph).run(*args)

            with maybe_disable_fake_tensor_mode(), enable_python_dispatcher(), (
                fake_mode
            ):
                try:
                    graph = make_fx(
                        graph_with_interpreter,
                        decomposition_table=decomposition_table,
                        tracing_mode="real",
                        _allow_non_fake_inputs=True,
                        pre_dispatch=pre_dispatch,
                        _allow_fake_constant=False,
                    )(*example_fake_inputs)
                except CondOpArgsMismatchError as e:
                    # Wrap the internal error to the user-facing error
                    raise UserError(UserErrorType.DYNAMIC_CONTROL_FLOW, str(e))

        if same_signature:
            graph = rewrite_signature(
                original_signature,
                graph,
                fake_mode,
                flat_args,
                in_spec,
                example_fake_inputs,
                graph_captured_input,
                graph_captured_result,
                result_traced,
            )
        # Store constraints and inputs as metadata for user passes, e.g. turn constraints to runtime check
        graph.meta["input_shape_constraints"] = (
            [constraint.serializable_spec for constraint in constraints]
            if constraints
            else []
        )

        return ExportResult(graph, out_guards)

    if extra_args or extra_kwargs:
        warnings.warn(
            "export(f, *args, **kwargs) is deprecated, use export(f)(*args, **kwargs) instead.  "
            "If you don't migrate, we may break your export call in the future if your user defined kwargs "
            "conflict with future kwargs added to export(f)."
        )
        return inner(*extra_args, **extra_kwargs)
    else:
        return inner


def optimize_assert(
    backend,
    *,
    hooks=Hooks(None, None),
    export=False,
    export_constraints=None,
    dynamic=None,
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

        from ..optim import (
            adadelta,
            adagrad,
            adam,
            adamax,
            adamw,
            asgd,
            lbfgs,
            nadam,
            radam,
            rmsprop,
            rprop,
            sgd,
            sparse_adam,
        )

        optimizer_modules = {
            adadelta,
            adagrad,
            adam,
            adamax,
            adamw,
            asgd,
            lbfgs,
            nadam,
            radam,
            rmsprop,
            rprop,
            sgd,
            sparse_adam,
        }

        disabled_multi_tensor_opt_modules = {
            adamax,
            nadam,
            radam,  # data-dependent control flow
            sgd,  # for now, until we can speed up compilation (this affects the benchmarks)
        }

        for opt_mod in optimizer_modules:
            opt_name = opt_mod.__name__.split(".")[-1]
            multi_tensor_fn_name = f"_multi_tensor_{opt_name}"
            fused_fn_name = f"_fused_{opt_name}"
            if (
                hasattr(opt_mod, multi_tensor_fn_name)
                and opt_mod in disabled_multi_tensor_opt_modules
            ):
                setattr(
                    opt_mod,
                    multi_tensor_fn_name,
                    disable(getattr(opt_mod, multi_tensor_fn_name)),
                )

            if hasattr(opt_mod, fused_fn_name):
                setattr(
                    opt_mod, fused_fn_name, disable(getattr(opt_mod, fused_fn_name))
                )

        optimizer_classes = [
            opt
            for opt in torch.optim.__dict__.values()
            if inspect.isclass(opt) and issubclass(opt, torch.optim.Optimizer)
        ]

        # Note: we don't support sparsity, data-dependent control, or tracing through backwards
        excluded_optimizer_classes = {
            torch.optim.SparseAdam,
            torch.optim.RAdam,
            torch.optim.LBFGS,
        }
        for opt in optimizer_classes:
            # We disable `register_load_state_dict_pre_hook` to allow torch.compile to trace
            # through the optimizer init without failing. See #107789
            opt.register_load_state_dict_pre_hook = disable(
                opt.register_load_state_dict_pre_hook
            )
            if opt in excluded_optimizer_classes:
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
