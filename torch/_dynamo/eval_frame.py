# mypy: disable-error-code="method-assign"

"""
This module implements the core frame evaluation handler for TorchDynamo's compilation system.
The eval frame handler intercepts Python bytecode execution at runtime to enable dynamic
compilation and optimization of PyTorch code.

Key components defined here:
- Frame evaluation handlers that intercept and analyze Python execution frames
- Guards management for tracking dependencies and invalidating compiled code
- Optimization contexts and decorators (optimize, run_once, disable, etc.)
- Export functionality for saving optimized graphs
- Backend compiler integrations and callback management

Functions in this file are responsible for modifying the eval frame handler at RUNTIME.
Therefore, all functions in this file are hot and performance-critical. Functions that
only execute at compile time should be placed in torch._dynamo.convert_frame.

The eval frame handler is the core mechanism that enables TorchDynamo to dynamically
intercept, analyze and optimize PyTorch code during execution. It works by registering
a custom frame evaluation function that gets called for every Python frame, allowing
us to detect PyTorch operations and trigger compilation as needed.
"""

from __future__ import annotations

import atexit
import contextlib
import functools
import inspect
import logging
import os
import sys
import sysconfig
import textwrap
import threading
import traceback
import types
import unittest
import warnings
import weakref
from collections.abc import Sized
from dataclasses import dataclass
from enum import Enum
from os.path import dirname, join
from typing import Any, NamedTuple, Optional, TYPE_CHECKING, Union
from unittest.mock import patch

import sympy

import torch
import torch.fx
import torch.utils._pytree as pytree
import torch.utils.checkpoint
from torch import _guards

# see discussion at https://github.com/pytorch/pytorch/issues/120699
from torch._C._dynamo.eval_frame import (  # noqa: F401
    reset_code,
    set_code_exec_strategy,
    set_eval_frame,
    set_guard_complete_hook,
    set_guard_error_hook,
    set_skip_guard_eval_unsafe,
    unsupported,
)
from torch._dispatch.python import enable_python_dispatcher
from torch._dynamo.types import ConvertFrameReturn, FrameAction, FrameExecStrategy
from torch._export.utils import _compiling_state_context
from torch._subclasses.fake_tensor import unset_fake_temporarily
from torch._utils_internal import DISABLE_JUSTKNOBS, justknobs_check, log_export_usage
from torch.export.dynamic_shapes import (
    _combine_args,
    _DimHint,
    _DimHintType,
    _IntWrapper,
    _process_dynamic_shapes,
    _RelaxedConstraint,
    Constraint,
)
from torch.fx import GraphModule, traceback as fx_traceback
from torch.fx.experimental._dynamism import (
    clone_and_convert_to_meta,
    track_dynamism_across_examples,
)
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.experimental.symbolic_shapes import (
    ConstraintViolationError,
    DimDynamic,
    ShapeEnv,
    StatelessSymbolicContext,
)
from torch.fx.graph import _PyTreeCodeGen, _PyTreeInfo

from . import config, convert_frame, distributed, external_utils, trace_rules, utils
from .backends.registry import CompilerFn, lookup_backend
from .code_context import code_context
from .exc import (
    CondOpArgsMismatchError,
    ShortenTraceback,
    Unsupported,
    UserError,
    UserErrorType,
)
from .hooks import Hooks
from .mutation_guard import install_generation_tagging_init
from .utils import (
    _get_error_on_graph_break,
    _set_error_on_graph_break,
    common_constant_types,
    compile_times,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    from torch._dynamo.package import CompilePackage
    from torch._dynamo.repro.after_dynamo import WrapBackendDebug
    from torch._subclasses import fake_tensor
    from torch.fx.node import Argument, Node, Target

    from .types import (
        CacheEntry,
        DynamoCallback,
        DynamoFrameType,
        GuardFail,
        GuardFilterEntry,
    )


log = logging.getLogger(__name__)


always_optimize_code_objects = utils.ExactWeakKeyDictionary()
null_context = contextlib.nullcontext


# See https://github.com/python/typing/pull/240
class Unset(Enum):
    token = 0


cached_backends: dict[int, CompilerFn] = {}

unset = Unset.token


if DISABLE_JUSTKNOBS:
    _maybe_set_eval_frame = set_eval_frame
else:

    def _maybe_set_eval_frame(callback: DynamoCallback) -> DynamoCallback:
        # A wrapper on set_eval_frame that is guarded by a Justknob.
        # Users can disable torchDynamo by setting the JK to False.
        if not justknobs_check("pytorch/compiler:enable_compiler_set_eval_frame"):
            torch._dynamo.utils.warn_once(
                "Dynamo disabled by Justknob: enable_compiler_set_eval_frame, skipping set_eval_frame"
            )
            return callback
        else:
            return set_eval_frame(callback)


@dataclass
class DynamoStance:
    stance: str = "default"
    skip_guard_eval_unsafe: bool = False
    backend: Union[str, Callable[..., Any], None] = None


_stance = DynamoStance()


def _set_stance(stance: DynamoStance) -> DynamoStance:
    global _stance

    from torch._C._dynamo.eval_frame import get_eval_frame_callback

    callback = get_eval_frame_callback()

    if callback is not False and callback is not None:
        raise RuntimeError("attempted to set_stance in a torch.compile region")

    prior = _stance
    _stance = stance
    return prior


_set_stance._dynamo_forbidden = True  # type: ignore[attr-defined]

_EXAMPLE_INPUTS: Optional[dict[str, list[Any]]] = None


def get_example_inputs(key: str) -> list[Any]:
    global _EXAMPLE_INPUTS
    if _EXAMPLE_INPUTS is None:
        _EXAMPLE_INPUTS = {}

    if key not in _EXAMPLE_INPUTS:
        _EXAMPLE_INPUTS[key] = []

    return _EXAMPLE_INPUTS[key]


def _callback_from_stance(callback: DynamoCallback) -> DynamoCallback:
    if _stance.stance == "default":
        # force_backend
        if _stance.backend is not None and callback not in (False, None):
            callback = _create_wrapped_callback(get_compiler_fn(_stance.backend))

        return callback
    elif _stance.stance == "eager_then_compile":
        if callback not in (False, None):
            return _create_delayed_compile_callback(callback, _stance.stance)
        return callback
    elif _stance.stance == "aot_eager_then_compile":
        if callback not in (False, None):
            return _create_delayed_compile_callback(callback, _stance.stance)
        return callback
    elif _stance.stance == "force_eager":
        # disable
        return None
    elif _stance.stance == "eager_on_recompile":
        # run mode
        return False
    elif _stance.stance == "fail_on_recompile":
        if callback in (False, None):
            return callback

        def fail_callback(
            frame: DynamoFrameType, *args: Any, **kwargs: Any
        ) -> ConvertFrameReturn:
            if trace_rules.check(frame.f_code):
                return ConvertFrameReturn()
            if not convert_frame.has_tensor_in_frame(frame):
                return ConvertFrameReturn()

            from torch._C._dynamo.eval_frame import (
                _debug_get_cache_entry_list,
                _debug_get_precompile_entries,
            )
            from torch._dynamo.guards import get_and_maybe_log_recompilation_reasons

            message = (
                "Detected recompile when torch.compile stance is 'fail_on_recompile'. "
                + f"filename: '{frame.f_code.co_filename}', "
                + f"function name: '{frame.f_code.co_name}', "
                + f"line number: {frame.f_lineno}"
            )
            cache_entries = _debug_get_cache_entry_list(frame.f_code)
            if cache_entries:
                reasons = get_and_maybe_log_recompilation_reasons(
                    cache_entries[0], frame, innermost_fn(callback), skip_logging=True
                )
                if reasons:
                    failures = textwrap.indent("\n".join(reasons), "- ")
                    guard_failure_details = (
                        f"triggered by the following guard failure(s):\n{failures}"
                    )
                    message += f"\n{textwrap.indent(guard_failure_details, '    ')}"
            precompile_entries = _debug_get_precompile_entries(frame.f_code)
            if len(precompile_entries) > 0:
                message += "\nFailed on the following precompiled guards: "
                for entry in precompile_entries:
                    message += f"\n{entry.guard_manager}{entry.guard_manager.check_verbose(frame.f_locals)}"  # type: ignore[attr-defined]
            raise RuntimeError(message)

        # to prevent cache miss due to different backend
        fail_callback._torchdynamo_orig_backend = callback  # type: ignore[attr-defined]

        return fail_callback
    else:
        raise RuntimeError(f"invalid torch.compile stance '{_stance}'")


def _create_wrapped_callback(
    compiler_fn: CompilerFn,
) -> convert_frame.CatchErrorsWrapper:
    hooks = Hooks()
    return convert_frame.catch_errors_wrapper(
        convert_frame.convert_frame(  # type: ignore[arg-type]
            compiler_fn,
            hooks,
        ),
        hooks,
    )


def _get_or_add_example_inputs(frame: DynamoFrameType) -> list[Any]:
    key = frame.f_code.co_filename + str(frame.f_code.co_firstlineno)
    example_inputs = get_example_inputs(key)

    if len(example_inputs) < 2:
        example_inputs.append(clone_and_convert_to_meta(frame.f_locals))

    return example_inputs


def _create_delayed_compile_callback(
    callback: DynamoCallback, stance: str
) -> Callable[..., Any]:
    def callback_fn(*args: Any, **kwargs: Any) -> convert_frame.ConvertFrameReturn:
        frame = args[0]
        example_inputs = _get_or_add_example_inputs(frame)

        if len(example_inputs) == 1:
            if stance == "eager_then_compile":
                return ConvertFrameReturn(
                    frame_exec_strategy=FrameExecStrategy(
                        FrameAction.DEFAULT, FrameAction.DEFAULT
                    )
                )
            elif stance == "aot_eager_then_compile":
                aot_eager_fn = get_compiler_fn("aot_eager")
                return _create_wrapped_callback(aot_eager_fn)(*args, **kwargs)

        dynamism = track_dynamism_across_examples(example_inputs)
        code_context.get_context(frame.f_code)["dynamism"] = dynamism
        compiler_fn = callback._torchdynamo_orig_backend._torchdynamo_orig_backend  # type: ignore[union-attr]
        return _create_wrapped_callback(compiler_fn)(*args, **kwargs)

    # to prevent cache miss due to different backend
    callback_fn._torchdynamo_orig_backend = callback  # type: ignore[attr-defined]

    return callback_fn


def _is_skip_guard_eval_unsafe_stance() -> bool:
    return _stance.skip_guard_eval_unsafe


def _reset_guarded_backend_cache() -> None:
    global cached_backends
    for backend in cached_backends.values():
        if hasattr(backend, "reset"):
            backend.reset()
    cached_backends.clear()


DONT_WRAP_FILES = {
    # For tracing into fx modules
    inspect.getsourcefile(GraphModule),
    join(dirname(dirname(__file__)), "onnx/_internal/fx/dynamo_graph_extractor.py"),
}


def _debug_get_cache_entry_list(
    code: Union[types.CodeType, Callable[..., Any]],
) -> list[CacheEntry]:
    """
    Given a code object or a callable object, retrieve the cache entries
     stored in this code.
    """
    if callable(code):
        code = code.__code__
    return torch._C._dynamo.eval_frame._debug_get_cache_entry_list(code)


class OptimizedModule(torch.nn.Module):
    """
    Wraps the original nn.Module object and later patches its
    forward method to optimized self.forward method.
    """

    _torchdynamo_orig_callable: Callable[..., Any]
    get_compiler_config: Callable[[], Any]

    _opt_mod_attributes = {
        "_orig_mod",
        "dynamo_ctx",
        "_torchdynamo_orig_callable",
        "get_compiler_config",
        "forward",
        "_forward",
        "__dict__",
        "named_children_walk",
        "_super_module_initialized",
    }

    def __init__(self, mod: torch.nn.Module, dynamo_ctx: _TorchDynamoContext) -> None:
        # NOTE: this must go first, because attribute reads/writes of `self`
        # uses `_orig_mod`, and sometimes users override `Module.__init__` to
        # do attribute reads/writes on `self`.
        #
        # We also can't use regular setattr because `super().__setattr__` will
        # complain for module value before `super().__init__()`
        object.__setattr__(self, "_orig_mod", mod)
        self._super_module_initialized = False
        super().__init__()
        self._super_module_initialized = True

        # Installs the params/buffer
        self._orig_mod = mod  # `super().__setattr__` will register this module
        self.dynamo_ctx = dynamo_ctx
        self._initialize()
        self.training = self._orig_mod.training

    def __len__(self) -> int:
        # Proxy the len call to the original module
        if isinstance(self._orig_mod, Sized):
            return len(self._orig_mod)
        # Mimic python's default behavior for objects without a length
        raise TypeError(f"{type(self._orig_mod).__name__} does not support len()")

    def _initialize(self) -> None:
        # Do this stuff in constructor to lower overhead slightly
        if isinstance(self.dynamo_ctx, DisableContext):
            # No need to check trace rules
            self.forward = self.dynamo_ctx(self._orig_mod.__call__)
        elif config.wrap_top_frame or (
            isinstance(self._orig_mod.forward, types.MethodType)
            and (
                trace_rules.check(self._orig_mod.forward)
                or getattr(self._orig_mod, "_is_fsdp_managed_module", False)
            )
        ):
            # This may be a torch.nn.* instance in trace_rules.py which
            # won't trigger a frame evaluation workaround to add an extra
            # frame we can capture
            self.forward = self.dynamo_ctx(external_utils.wrap_inline(self._orig_mod))
        else:
            # Invoke hooks outside of dynamo then pickup the inner frame
            self.forward = self.dynamo_ctx(self._orig_mod.__call__)

        if hasattr(self._orig_mod, "_initialize_hook"):
            self._forward = self.forward
            self.forward = self._call_lazy_check

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if torch.nn.modules.module._has_any_global_hook():
            warnings.warn(
                "Using `torch.compile(module)` when there are global hooks on "
                "modules (e.g., from `register_module_forward_hook`); this will"
                " cause the hooks to fire an extra time for the "
                "`OptimizedModule` created by `torch.compile(module)`. If this "
                "causes undesired behavior, please try using `module.compile()`"
                ", or use the per-module hooks instead",
                stacklevel=2,
            )
        return super().__call__(*args, **kwargs)

    def _aot_compile(self, inputs: list[torch._dynamo.aot_compile.ModelInput]) -> None:
        """
        Experimental: AOT Compile a set of inputs and use that as the forward function
        """
        model = self._orig_mod
        hooks = self.dynamo_ctx._hooks
        assert hooks is not None
        if not config.enable_aot_compile:
            raise RuntimeError(
                "AOT Compile is not enabled, please set torch._dynamo.config.enable_aot_config=True"
            )
        if not self.dynamo_ctx.fullgraph:
            raise RuntimeError(
                "Graph breaks are not supported with aot compile. Please use torch.compile(fullgraph=True)."
            )

        if not callable(self.dynamo_ctx.callback):
            raise RuntimeError("aot compile requires a callable dynamo callback.")

        backend = innermost_fn(
            self.dynamo_ctx.callback, unaltered_fn_attr="_torchdynamo_orig_backend"
        )
        from torch._dynamo.aot_compile import aot_compile_module

        self.forward = aot_compile_module(model, inputs, hooks, backend)

    def _save_aot_compiled_module(self, path: Optional[str] = None) -> bytes:
        if not config.enable_aot_compile:
            raise RuntimeError(
                "AOT Compile is not enabled, please set torch._dynamo.config.enable_aot_config=True"
            )
        from torch._dynamo.aot_compile import AOTCompiledModel

        assert isinstance(self.forward, AOTCompiledModel)
        result: bytes = self.forward.serialize()
        if path is not None:
            with open(path, "wb") as f:
                f.write(result)
        return result

    def _load_aot_compiled_module(self, data: bytes) -> None:
        if not config.enable_aot_compile:
            raise RuntimeError(
                "AOT Compile is not enabled, please set torch._dynamo.config.enable_aot_config=True"
            )
        from torch._dynamo.aot_compile import AOTCompiledModel

        compiled_forward = AOTCompiledModel.deserialize(self._orig_mod, data)
        assert isinstance(compiled_forward, AOTCompiledModel)
        self.forward = compiled_forward

    def __reduce__(
        self,
    ) -> tuple[type[OptimizedModule], tuple[torch.nn.Module, _TorchDynamoContext]]:
        return (self.__class__, (self._orig_mod, self.dynamo_ctx))

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state.pop("forward", None)
        state.pop("__call__", None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__ = state
        self._initialize()

    @property
    # pyrefly: ignore [bad-override]
    def training(self) -> bool:
        return self._orig_mod.training

    @training.setter
    def training(self, value: bool) -> None:
        # Ignore the `training` mutation in `super().__init__()`, since that's
        # setting the default on `nn.Module`, but we are mirroring the
        # `training` attr in `self._orig_mod`.
        if self._super_module_initialized:
            self._orig_mod.training = value

    def __getattr__(self, name: str) -> Any:
        if name == "_orig_mod":
            return self._modules["_orig_mod"]
        return getattr(self._orig_mod, name)

    def __setattr__(self, name: str, val: Any) -> None:
        # Allow patching over class attributes
        if hasattr(type(self), name):
            return super().__setattr__(name, val)

        if name in OptimizedModule._opt_mod_attributes:
            return super().__setattr__(name, val)
        return setattr(self._orig_mod, name, val)

    def __delattr__(self, name: str) -> None:
        # This mirrors `__setattr__`
        if hasattr(type(self), name):
            return super().__delattr__(name)

        if name in OptimizedModule._opt_mod_attributes:
            return super().__delattr__(name)
        return delattr(self._orig_mod, name)

    def _call_lazy_check(self, *args: Any, **kwargs: Any) -> Any:
        if (
            hasattr(self._orig_mod, "_initialize_hook")
            and hasattr(self._orig_mod, "_infer_parameters")
            and callable(self._orig_mod._infer_parameters)
        ):
            # In the case of a lazy module, we want to run
            # the pre-hooks which initialize it.
            # Afterwards, lazy module deletes its pre-hooks
            # to avoid treating it as lazy on subsequent recompile.
            self._orig_mod._infer_parameters(self._orig_mod, args, kwargs)
        return self._forward(*args, **kwargs)

    def __dir__(self) -> list[str]:
        orig_mod_attrs = self._orig_mod.__dir__()
        return orig_mod_attrs + [
            attr for attr in super().__dir__() if attr not in orig_mod_attrs
        ]


def remove_from_cache(f: Any) -> None:
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


def nothing() -> None:
    pass


def always_false() -> bool:
    return False


def innermost_fn(
    fn: Callable[..., Any], unaltered_fn_attr: str = "_torchdynamo_orig_callable"
) -> Callable[..., Any]:
    """
    In case of nesting of _TorchDynamoContext calls, find the innermost
    function. TorchDynamo caches on fn.__code__ object, so its necessary to find
    the innermost function to pass on the optimize, run, disable etc.
    """
    unaltered_fn = fn
    while hasattr(unaltered_fn, unaltered_fn_attr):
        unaltered_fn = getattr(unaltered_fn, unaltered_fn_attr)
        assert callable(unaltered_fn), (
            f"A callable function is expected, but {type(unaltered_fn)} is provided."
        )
    return unaltered_fn


def make_set_enable_dynamic(enable: bool) -> Any:
    assert isinstance(enable, bool)
    if enable:
        # Assume everything is dynamic by default
        return config._make_closure_patcher(assume_static_by_default=False)
    else:
        return config._make_closure_patcher(
            automatic_dynamic_shapes=False, assume_static_by_default=True
        )


# A thread local storage that serves to store information as Dynamo traces
# through a user provided function.
class DynamoTLS(threading.local):
    # Each string is a summary of a frame Dynamo attempted to trace, stored in
    # temporal order.
    traced_frame_infos: list[str] = []


dynamo_tls = DynamoTLS()


def clear_dynamo_tls() -> None:
    dynamo_tls.traced_frame_infos.clear()


@atexit.register
def _log_traced_frames() -> None:
    """
    At program exit, log all of the frames Dynamo has attempted to trace from,
    excluding the continuation frames generated by Dynamo.
    """
    msg = "\n".join(dynamo_tls.traced_frame_infos)
    msg = textwrap.indent(msg, "  * ")
    msg = f"TorchDynamo attempted to trace the following frames: [\n{msg}\n]"
    log.info(msg)


def guard_collectives_hook(guard_eval_result: bool) -> bool:
    import torch.distributed as dist
    from torch._dynamo.utils import dynamo_timed

    # guard_eval_result == True  ==>  cache hit
    if pg := distributed.get_guard_pg():
        with dynamo_timed(
            "guard_collective", log_pt2_compile_event=False, log_waitcounter=True
        ):
            log.debug("guard_collective %s", guard_eval_result)
            # TODO: a bit awkward to time, this isn't inside of the dynamo compile region
            all_results = [None] * pg.size()
            dist.all_gather_object(all_results, guard_eval_result, group=pg)
            # True = everyone hit, OK to run
            # False = someone missed, force recompile everywhere
            res = all(all_results)
            log.debug("guard_collective %s -> %s", guard_eval_result, res)
            return res
    return guard_eval_result


_not_set = object()


class _TorchDynamoContext:
    def __init__(
        self,
        callback: DynamoCallback,
        on_enter: Callable[[], Any] = nothing,
        backend_ctx_ctor: Callable[
            [], contextlib.AbstractContextManager[Any]
        ] = null_context,
        patch_fn: Callable[[], Any] = nothing,
        first_ctx: bool = False,
        *,
        fullgraph: bool = False,
        error_on_graph_break: Optional[bool] = None,
        export: bool = False,
        dynamic: Optional[bool] = None,
        compiler_config: Optional[Any] = None,
        package: Optional[CompilePackage] = None,
        hooks: Optional[Hooks] = None,
    ) -> None:
        super().__init__()
        assert callable(callback) or callback is False or callback is None
        self.callback: DynamoCallback = callback
        self._backend_ctx_ctor = backend_ctx_ctor
        self.prior: Union[Unset, DynamoCallback] = unset
        self.first_ctx = first_ctx
        self.fullgraph = fullgraph
        self.error_on_graph_break = error_on_graph_break
        self.export = export
        self._dynamic = dynamic
        self.compiler_config = compiler_config
        self.cleanup_fns: list[Callable[[], Any]] = []
        self.enter_exit_hooks = []
        self._package = package
        self._hooks = hooks
        patch_fn()

        # Save the backends so that we can reset them during torch._dynamo.reset
        backend = innermost_fn(callback, unaltered_fn_attr="_torchdynamo_orig_backend")  # type: ignore[arg-type]
        cached_backends.setdefault(id(backend), backend)  # type: ignore[arg-type]

        if dynamic is not None:
            self.enter_exit_hooks.append(make_set_enable_dynamic(dynamic))

        if on_enter is not nothing:
            # this case is not common
            def call_on_enter() -> Callable[[], None]:
                on_enter()
                return nothing

            self.enter_exit_hooks.append(call_on_enter)

        if backend_ctx_ctor is not contextlib.nullcontext:
            # this case is not common
            def call_backend_ctx() -> functools.partial[Optional[bool]]:
                ctx = backend_ctx_ctor()
                ctx.__enter__()
                return functools.partial(ctx.__exit__, None, None, None)

            self.enter_exit_hooks.append(call_backend_ctx)

    def __enter__(self) -> None:
        if config.raise_on_ctx_manager_usage:
            raise RuntimeError(
                "torch._dynamo.optimize(...) is used with a context manager. "
                "Please refer to https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html "
                "to use torch._dynamo.optimize(...) as an annotation/decorator. "
            )
        self.prior = set_eval_frame(None)
        self.cleanup_fns = [enter() for enter in self.enter_exit_hooks]
        self.prior_skip_guard_eval_unsafe = set_skip_guard_eval_unsafe(
            _is_skip_guard_eval_unsafe_stance()
        )
        _maybe_set_eval_frame(_callback_from_stance(self.callback))

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[types.TracebackType],
    ) -> Optional[bool]:
        assert self.prior is not unset
        set_eval_frame(None)
        set_skip_guard_eval_unsafe(self.prior_skip_guard_eval_unsafe)
        for cleanup in self.cleanup_fns:
            cleanup()
        self.cleanup_fns.clear()
        _maybe_set_eval_frame(_callback_from_stance(self.prior))
        self.prior = unset
        return None

    def __call__(self, fn: Any) -> Any:
        # public api for compiler config/options
        def get_compiler_config() -> Any:
            return self.compiler_config

        from .package import DynamoCache

        # If self._package is lazily initialized, we should check the dynamo cache now
        if config.caching_precompile:
            if self._package is not None and not self._package.is_initialized():
                fn_key = fn.forward if isinstance(fn, torch.nn.Module) else fn
                result = DynamoCache.load(fn_key)
                if result is None:
                    # Create a fresh CompilePackage
                    self._package.initialize(fn_key, None, ignore_inlined_sources=False)
                else:
                    try:
                        self._package.initialize(
                            fn_key, result.dynamo, ignore_inlined_sources=False
                        )
                        self._package.install(result.backends)
                    except RuntimeError:
                        log.warning(
                            "Failed to load entry from dynamo cache", exc_info=True
                        )
                        self._package.initialize(
                            fn_key, None, ignore_inlined_sources=False
                        )

        fn = innermost_fn(fn)

        def aot_compile(example_inputs: tuple[tuple[Any, ...], dict[str, Any]]) -> Any:
            from torch._dynamo.aot_compile import aot_compile_fullgraph

            if torch._inductor.config.force_disable_caches:
                raise RuntimeError(
                    "Cannot precompile with torch._inductor.config.force_disable_caches=True; caching is required."
                )

            if not self.fullgraph:
                raise RuntimeError(
                    "Graph breaks are not supported with aot compile. Please use torch.compile(fullgraph=True)."
                )

            if not callable(self.callback):
                raise RuntimeError("aot compile requires a callable dynamo callback.")

            assert self._hooks is not None

            return aot_compile_fullgraph(
                fn,
                example_inputs,
                hooks=self._hooks,
                backend=innermost_fn(
                    self.callback, unaltered_fn_attr="_torchdynamo_orig_backend"
                ),
            )

        # add context containing GraphModule to any GraphModule forward functions
        if isinstance(fn, GraphModule):
            # add context containing GraphModule to any GraphModule forward functions
            code_context.get_context(fn.forward.__code__)["orig_graphmodule"] = (
                weakref.ref(fn)
            )

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
            new_mod.get_compiler_config = get_compiler_config

            return new_mod

        if inspect.isclass(fn):
            # User has wrapped the class with compile/disable decorator. Apply
            # disable to init/call method.
            cls_obj = fn
            cls_obj.__call__ = self(cls_obj.__call__)
            if issubclass(cls_obj, torch.nn.Module):
                # NN module variable tracker directly inlines the _call_impl.
                cls_obj._call_impl = self(cls_obj._call_impl)
            return cls_obj

        assert callable(fn), (
            f"A callable function is expected, but {type(fn)} is provided."
        )

        try:
            filename = inspect.getsourcefile(fn)
        except TypeError:
            filename = None
        if config.debug_force_nested_calls:
            fn = external_utils.wrap_inline(fn)
        elif config.wrap_top_frame or (
            (filename is None or trace_rules.check(fn))
            and (
                getattr(fn, "__name__", "")
                not in ["_call_impl", "_wrapped_call_impl", "_lazy_forward"]
            )
            and filename not in DONT_WRAP_FILES
        ):
            # call to a builtin without a frame for us to capture
            fn = external_utils.wrap_inline(fn)

        def do_nothing(*arg: Any, **kwargs: Any) -> None:
            pass

        callback: Callable[..., Any] = do_nothing
        if hasattr(self, "callback"):
            callback = self.callback  # type: ignore[assignment]

        is_jit_tracing = torch._C._is_tracing
        is_fx_symbolic_tracing = torch.fx._symbolic_trace.is_fx_symbolic_tracing

        @functools.wraps(fn)
        def compile_wrapper(*args: Any, **kwargs: Any) -> Any:
            prior = set_eval_frame(None)
            try:
                # We shouldn't compile inside kernel invocation.
                if tracing_context := torch._guards.TracingContext.try_get():
                    if (
                        tracing_context.fake_mode is not None
                        and tracing_context.fake_mode.in_kernel_invocation
                    ):
                        return fn(*args, **kwargs)
                # Skip nested compile - just inline the function
                if is_fx_symbolic_tracing():
                    if config.error_on_nested_fx_trace:
                        raise RuntimeError(
                            "Detected that you are using FX to symbolically trace "
                            "a dynamo-optimized function. This is not supported at the moment."
                        )
                    else:
                        return fn(*args, **kwargs)

                if is_jit_tracing():
                    raise RuntimeError(
                        "Detected that you are using FX to torch.jit.trace "
                        "a dynamo-optimized function. This is not supported at the moment."
                    )

                cleanups = [enter() for enter in self.enter_exit_hooks]
                prior_skip_guard_eval_unsafe = set_skip_guard_eval_unsafe(
                    _is_skip_guard_eval_unsafe_stance()
                )
                prior_error_on_graph_break = None
                if not self.fullgraph and self.error_on_graph_break is not None:
                    prior_error_on_graph_break = _get_error_on_graph_break()
                    _set_error_on_graph_break(self.error_on_graph_break)

                # Ensure that if an assertion occurs after graph pushes
                # something onto the DynamicLayerStack then we pop it off (the
                # constructed graph code isn't guarded with try/finally).
                #
                # This used to be a context but putting a `with` here is a noticeable
                # perf regression (#126293)
                saved_dynamic_layer_stack_depth = (
                    torch._C._functorch.get_dynamic_layer_stack_depth()
                )

                _maybe_set_eval_frame(_callback_from_stance(callback))

                try:
                    return fn(*args, **kwargs)
                except Unsupported as e:
                    if config.verbose:
                        raise
                    # strip internal tracebacks from causes
                    cur_exn: BaseException = e
                    while cur_exn.__cause__ is not None:
                        cur_exn.__cause__.with_traceback(None)
                        cur_exn = cur_exn.__cause__
                    # pyrefly: ignore [invalid-inheritance]
                    raise e.with_traceback(None) from e.__cause__  # User compiler error
                except ShortenTraceback as e:
                    # Failures in the backend likely don't have useful
                    # data in the TorchDynamo frames, so we strip them out.
                    raise e.remove_dynamo_frames() from None  # see TORCHDYNAMO_VERBOSE=1
                finally:
                    # Restore the dynamic layer stack depth if necessary.
                    set_eval_frame(None)
                    if prior_error_on_graph_break is not None:
                        _set_error_on_graph_break(prior_error_on_graph_break)
                    torch._C._functorch.pop_dynamic_layer_stack_and_undo_to_depth(
                        saved_dynamic_layer_stack_depth
                    )

                    set_skip_guard_eval_unsafe(prior_skip_guard_eval_unsafe)
                    for cleanup in cleanups:
                        cleanup()
            finally:
                _maybe_set_eval_frame(prior)

        # hooks to properly handle inlining
        if self.error_on_graph_break is not None:
            compile_wrapper._torchdynamo_inline = (  # type: ignore[attr-defined]
                external_utils.wrap_inline_with_error_on_graph_break(
                    fn, self.error_on_graph_break
                )
            )
        else:
            compile_wrapper._torchdynamo_inline = fn  # type: ignore[attr-defined]

        # Save the function pointer to find the original callable while nesting
        # of decorators.
        compile_wrapper._torchdynamo_orig_callable = fn  # type: ignore[attr-defined]

        # when compiling user function instead of nn.Module
        # provide public api _fn.get_compiler_config()
        assert not hasattr(compile_wrapper, "get_compiler_config")
        compile_wrapper.get_compiler_config = get_compiler_config  # type: ignore[attr-defined]
        if torch._dynamo.config.enable_aot_compile:
            compile_wrapper.aot_compile = aot_compile  # type: ignore[attr-defined]

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
                        >>     def __init__(self) -> None:
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

        return compile_wrapper


class OptimizeContext(_TorchDynamoContext):
    def __init__(
        self,
        callback: DynamoCallback,
        backend_ctx_ctor: Callable[[], contextlib.AbstractContextManager[Any]],
        first_ctx: bool = False,
        *,
        fullgraph: bool = False,
        error_on_graph_break: Optional[bool] = None,
        export: bool = False,
        dynamic: Optional[bool] = None,
        compiler_config: Optional[Any] = None,
        rebuild_ctx: Optional[
            Callable[[], Union[OptimizeContext, _NullDecorator]]
        ] = None,
        package: Optional[CompilePackage] = None,
        hooks: Optional[Hooks] = None,
    ) -> None:
        def on_enter() -> None:
            install_generation_tagging_init()

        super().__init__(
            callback=callback,
            on_enter=on_enter,
            backend_ctx_ctor=backend_ctx_ctor,
            patch_fn=TorchPatcher.patch,
            first_ctx=first_ctx,
            fullgraph=fullgraph,
            error_on_graph_break=error_on_graph_break,
            export=export,
            dynamic=dynamic,
            compiler_config=compiler_config,
            package=package,
            hooks=hooks,
        )

        if config.compiled_autograd:
            _dynamic = self._dynamic
            if _dynamic is None:
                _dynamic = not torch._dynamo.config.assume_static_by_default

            def call_compiled_autograd() -> functools.partial[Optional[bool]]:
                assert rebuild_ctx is not None
                compiler_fn = rebuild_ctx()
                ctx = torch._dynamo.compiled_autograd._enable(
                    compiler_fn,
                    # pyrefly: ignore [bad-argument-type]
                    dynamic=_dynamic,
                    ignore_active_disable_ctx=False,
                )
                ctx.__enter__()
                return functools.partial(ctx.__exit__, None, None, None)

            self.enter_exit_hooks.append(call_compiled_autograd)

    def __reduce__(
        self,
    ) -> tuple[type[OptimizeContext], tuple[Any, ...], dict[str, Any]]:
        return (
            self.__class__,
            (self.callback, self._backend_ctx_ctor, self.first_ctx),
            {
                "export": self.export,
                "dynamic": self._dynamic,
                "compiler_config": self.compiler_config,
            },
        )


class RunOnlyContext(_TorchDynamoContext):
    def __init__(self) -> None:
        # cudagraph trees relies on generation increment
        def on_enter() -> None:
            torch._dynamo.mutation_guard.GenerationTracker.generation += 1

        super().__init__(callback=False, on_enter=on_enter)

    def __reduce__(self) -> tuple[type[RunOnlyContext], tuple[Any, ...]]:
        return (self.__class__, ())


class DisableContext(_TorchDynamoContext):
    def __init__(self, msg: Optional[str] = None, wrapping: bool = True) -> None:
        super().__init__(callback=None)
        self.msg = msg
        self.wrapping = wrapping

    def __call__(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        # Earlier this code was in the base class _TorchDynamoContext. But we
        # moved it here to have better code organization. For disable, we just
        # want the callback to be None. We don't have to check trace_rules or
        # create any wrapper.
        fn = innermost_fn(fn)

        if isinstance(fn, torch.nn.Module):
            mod = fn
            new_mod = OptimizedModule(mod, self)
            new_mod._torchdynamo_orig_callable = mod.forward
            return new_mod

        if isinstance(fn, type):
            # User has wrapped the class with compile/disable decorator. Apply
            # disable to init/call method.
            cls_obj = fn
            # Disable on init is useful for reconstruction of bytecodes where we
            # want to prevent Dynamo from tracing into the init function. Check
            # test_reconstruction in test_model_output.py.
            cls_obj.__init__ = self(cls_obj.__init__)  # type: ignore[misc]
            cls_obj.__call__ = self(cls_obj.__call__)
            if issubclass(cls_obj, torch.nn.Module):
                # NN module variable tracker directly inlines the _call_impl. Disable it.
                # pyrefly: ignore [missing-attribute]
                cls_obj._call_impl = self(cls_obj._call_impl)
            return cls_obj

        assert callable(fn), (
            f"A callable function is expected, but {type(fn)} is provided."
        )

        def _fn(*args: Any, **kwargs: Any) -> Any:
            prior = set_eval_frame(None)
            try:
                _maybe_set_eval_frame(_callback_from_stance(self.callback))
                try:
                    if torch.compiler.is_exporting():
                        with fx_traceback.annotate(
                            {
                                "_torchdynamo_disable": True,
                                "_torchdynamo_disable_recursive": True,
                                "_torchdynamo_disable_method": getattr(
                                    fn, "__name__", type(fn).__name__
                                ),
                            }
                        ):
                            return fn(*args, **kwargs)
                    return fn(*args, **kwargs)
                finally:
                    set_eval_frame(None)
            finally:
                _maybe_set_eval_frame(prior)

        # Under some circumstances (e.g. precompile) we can end up calling @disable
        # decorator in generated bytecode and trigger recompile. This is due to the
        # fact that the old callback from torch.compile() is still active and under
        # this circumstance we will trigger a failure with set_stance("fail_on_recompile").
        # Therefore we want to skip calling into any frame in this case.
        if self.wrapping:
            _fn = functools.wraps(fn)(_fn)

        _fn._torchdynamo_disable = True  # type: ignore[attr-defined]
        _fn._torchdynamo_disable_msg = self.msg  # type: ignore[attr-defined]

        # Save the function pointer to find the original callable while nesting
        # of decorators.
        _fn._torchdynamo_orig_callable = fn  # type: ignore[attr-defined]

        _fn._torchdynamo_disable_recursive = True  # type: ignore[attr-defined]

        return _fn

    def __reduce__(self) -> tuple[type[DisableContext], tuple[Any, ...]]:
        return (self.__class__, ())


def _optimize_catch_errors(
    compile_fn: convert_frame.ConvertFrameProtocol,
    hooks: Hooks,
    backend_ctx_ctor: Callable[
        [], contextlib.AbstractContextManager[Any]
    ] = null_context,
    fullgraph: bool = False,
    error_on_graph_break: Optional[bool] = None,
    export: bool = False,
    dynamic: Optional[bool] = None,
    compiler_config: Optional[Any] = None,
    rebuild_ctx: Optional[Callable[[], Union[OptimizeContext, _NullDecorator]]] = None,
    package: Optional[CompilePackage] = None,
) -> OptimizeContext:
    return OptimizeContext(
        convert_frame.catch_errors_wrapper(compile_fn, hooks),
        backend_ctx_ctor=backend_ctx_ctor,
        first_ctx=True,
        fullgraph=fullgraph,
        error_on_graph_break=error_on_graph_break,
        export=export,
        dynamic=dynamic,
        compiler_config=compiler_config,
        rebuild_ctx=rebuild_ctx,
        package=package,
        hooks=hooks,
    )


def get_compiler_fn(
    compiler_fn: Union[str, Callable[..., Any], None],
) -> WrapBackendDebug:
    from .repro.after_dynamo import wrap_backend_debug

    if compiler_fn is None:
        # Special case None to avoid crashing in hasattr
        compiler_str = None
    elif hasattr(compiler_fn, "compiler_name"):
        compiler_str = compiler_fn.compiler_name  # type: ignore[union-attr]
        assert isinstance(compiler_str, str)
    elif isinstance(compiler_fn, str):
        compiler_str = compiler_fn
    else:
        compiler_str = None
    compiler_fn = lookup_backend(compiler_fn)  # type: ignore[arg-type]
    return wrap_backend_debug(compiler_fn, compiler_str)


class _NullDecorator(contextlib.nullcontext):  # type: ignore[type-arg]
    def __call__(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        assert callable(fn), (
            f"A callable function is expected, but {type(fn)} is provided."
        )
        return fn


# Make dynamo graph to have same input/output spec as user code
def argument_names(
    f_sig: inspect.Signature,
    args: Union[list[Any], tuple[Any, ...]],
    kwargs: dict[str, Any],
) -> list[str]:
    def signature_to_fullargspec(sig: inspect.Signature) -> inspect.FullArgSpec:
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
            f"{fullargspec.varargs}_{i}" for i in range(len(args) - len(input_strs))
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
        assert kwonly_arg in kwargs or kwonly_arg in kwonlydefaults, (
            f"Missing keyword only argument {kwonly_arg}"
        )

    return input_strs


def check_if_dynamo_supported() -> None:
    if sys.version_info >= (3, 15):
        raise RuntimeError("Python 3.15+ not yet supported for torch.compile")
    elif sysconfig.get_config_var("Py_GIL_DISABLED") == 1 and sys.version_info < (
        3,
        13,
        3,
    ):
        raise RuntimeError(
            "torch.compile is not supported on Python < 3.13.3 built with GIL disabled. "
            "Please use Python 3.13.3+."
        )


def is_dynamo_supported() -> bool:
    try:
        check_if_dynamo_supported()
        return True
    except Exception:
        return False


def check_if_inductor_supported() -> None:
    check_if_dynamo_supported()


def is_inductor_supported() -> bool:
    try:
        check_if_inductor_supported()
        return True
    except Exception:
        return False


def check_for_incompatible_configs() -> None:
    # Some of the configs should be mutually exclusive
    assert not (config.suppress_errors and config.fail_on_recompile_limit_hit), (
        "Dynamo configs suppress_error and fail_on_recompile_limit_hit can not both be active at the same time."
    )


def optimize(*args: Any, **kwargs: Any) -> Union[OptimizeContext, _NullDecorator]:
    def rebuild_ctx() -> Union[OptimizeContext, _NullDecorator]:
        ca_kwargs_override = config.compiled_autograd_kwargs_override
        if ca_kwargs_override:
            # NOTE: The process of translating other `torch.compile` kwargs to `torch._dynamo.optimize` kwargs
            # is more complicated, we will add it in the future when needed.
            assert set(ca_kwargs_override.keys()) == {"fullgraph"}, (
                f"Only `fullgraph` kwarg override is supported for now, but got {ca_kwargs_override.keys()}"
            )
            kwargs["nopython"] = ca_kwargs_override["fullgraph"]
        return optimize(*args, **kwargs)

    return _optimize(rebuild_ctx, *args, **kwargs)


def _optimize(
    rebuild_ctx: Callable[[], Union[OptimizeContext, _NullDecorator]],
    backend: Union[str, Callable[..., Any]] = "inductor",
    *,
    nopython: bool = False,
    error_on_graph_break: Optional[bool] = None,
    guard_export_fn: Optional[Callable[[_guards.GuardsSet], None]] = None,
    guard_fail_fn: Optional[Callable[[GuardFail], None]] = None,
    guard_filter_fn: Optional[Callable[[list[GuardFilterEntry]], list[bool]]] = None,
    disable: bool = False,
    dynamic: Optional[bool] = None,
    package: Optional[CompilePackage] = None,
) -> Union[OptimizeContext, _NullDecorator]:
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
        error_on_graph_break: If not None, the current `error_on_graph_break` setting is set to the given value.
            See `torch._dynamo.error_on_graph_break()` for more details on what `error_on_graph_break` means.

            Unlike `nopython=True` (i.e. `fullgraph=True`), there is no guarantee of a single whole-program graph.
            If `nopython` is True, `error_on_graph_break` does nothing.
        disable: If True, turn this decorator into a no-op
        dynamic: If True, upfront compile as dynamic a kernel as possible.  If False,
            disable all dynamic shapes support (always specialize).  If None, automatically
            detect when sizes vary and generate dynamic kernels upon recompile.

    Example Usage::

        @torch._dynamo.optimize()
        def toy_example(a, b): ...
    """
    check_if_dynamo_supported()
    check_for_incompatible_configs()
    # Note: The hooks object could be global instead of passed around, *however* that would make
    # for a confusing API usage and plumbing story wherein we nest multiple .optimize calls.
    # There is some prior art around this, w/r/t nesting backend calls are enforced to be the same
    # compiler, however, this feels onerous for callback and hooks, and it feels better to give our users an
    # easier to understand UX at the cost of a little more plumbing on our end.
    hooks = Hooks(
        guard_export_fn=guard_export_fn,
        guard_fail_fn=guard_fail_fn,
        guard_filter_fn=guard_filter_fn,
    )
    torch._C._log_api_usage_once("torch._dynamo.optimize")
    if (
        disable
        or os.environ.get("TORCHDYNAMO_DISABLE", "") == "1"
        or (not justknobs_check("pytorch/compiler:enable_dynamo"))
    ):
        return _NullDecorator()

    if nopython and not config.debug_force_graph_break_on_leaf_return:
        return optimize_assert(
            backend,
            dynamic=dynamic,
            hooks=hooks,
            rebuild_ctx=rebuild_ctx,
            package=package,
        )

    backend = get_compiler_fn(backend)

    # Find if backend has any extra context manager
    backend_ctx_ctor = getattr(backend, "backend_ctx_ctor", null_context)

    # The backend function is stashed in the callable returned by
    # _optimize_catch_errors in the field _torchdynamo_orig_backend. This can
    # be used by eval_frame.c to insert a guard on the backend.

    # With CachingPrecompile, instantiate an uninitialized CompilePackage
    # which gets initialized by _optimize_catch_errors.__call__ once we have a function
    if config.caching_precompile and package is None:
        from .package import CompilePackage

        package = CompilePackage(fn=None, dynamo=None, ignore_inlined_sources=False)

    return _optimize_catch_errors(
        convert_frame.convert_frame(
            backend,
            hooks,
            package=package,
        ),
        hooks,
        backend_ctx_ctor,
        fullgraph=False,
        error_on_graph_break=error_on_graph_break
        and not config.debug_force_graph_break_on_leaf_return,
        dynamic=dynamic,
        compiler_config=(
            backend.get_compiler_config()
            if hasattr(backend, "get_compiler_config")
            else None
        ),
        rebuild_ctx=rebuild_ctx,
        package=package,
    )


# TODO(voz): Consider making "explain" output alongside a run / part of a run
@patch("torch._dynamo.symbolic_convert.explain", True)
def explain(f: Callable[..., Any], *extra_args: Any, **extra_kwargs: Any) -> Any:
    from .backends.debugging import ExplainOutput

    def inner(*args: Any, **kwargs: Any) -> ExplainOutput:
        # TODO(voz): Do we want a decorator for this?
        from . import reset  # type: ignore[attr-defined]

        reset()

        graphs: list[torch.fx.GraphModule] = []
        break_reasons: list[Any] = []
        op_count: int = 0
        ops_per_graph: list[list[Target]] = []
        out_guards: list[_guards.Guard] = []

        def dynamo_graph_accumulating_compiler(
            gm: torch.fx.GraphModule, example_inputs: Any
        ) -> Callable[..., Any]:
            from .backends.debugging import _explain_graph_detail

            nonlocal graphs
            nonlocal op_count
            nonlocal ops_per_graph
            nonlocal break_reasons

            gm, graphs, op_count, ops_per_graph, break_reasons = _explain_graph_detail(
                gm, graphs, op_count, ops_per_graph, break_reasons
            )

            return gm.forward

        def guard_export_print(guards: Iterable[_guards.Guard]) -> None:
            nonlocal out_guards
            out_guards.extend(guards)

        opt_f = optimize(
            dynamo_graph_accumulating_compiler,
            nopython=False,
            guard_export_fn=guard_export_print,
        )(f)
        # TODO(voz): We may have instances of `f` that mutate inputs, we should track sideeffects and reject.
        opt_f(*args, **kwargs)

        graph_count = len(graphs)
        graph_break_count = graph_count - 1
        compile_time = compile_times(repr="str")

        # TODO(voz): Do we want a decorator for this?
        reset()

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
            "conflict with future kwargs added to explain(f).",
            FutureWarning,
            stacklevel=2,
        )
        return inner(*extra_args, **extra_kwargs)
    else:
        return inner


class FlattenInputOutputSignature(torch.fx.Transformer):
    def __init__(
        self,
        m: torch.fx.GraphModule,
        flat_args: list[Any],
        matched_input_elements_positions: list[int],
        flat_results: Sequence[Any],
        matched_output_elements_positions: list[int],
        example_fake_inputs: list[torch.Tensor],
        flat_args_dynamic_dims: list[set[int]],
        fake_mode: Optional[fake_tensor.FakeTensorMode] = None,
    ) -> None:
        super().__init__(m)

        assert len(flat_args_dynamic_dims) == len(flat_args)
        matched_input_elements_to_fake = {
            val: example_fake_inputs[ix]
            for ix, val in enumerate(matched_input_elements_positions)
        }

        self.new_args = []
        for i in range(len(flat_args)):
            arg = super().placeholder(f"arg{i}", (), {})
            if i in matched_input_elements_to_fake:
                arg.node.meta["val"] = matched_input_elements_to_fake[i]
            else:
                # Fill node.meta["val"] with faketensor from the input,
                # if it's not found in matched_input_elements_positions
                if fake_mode is not None and isinstance(flat_args[i], torch.Tensor):
                    # TODO(zhxchen17) Also preserve all the user constraints here.
                    arg.node.meta["val"] = fake_mode.from_tensor(
                        flat_args[i],
                        symbolic_context=StatelessSymbolicContext(
                            dynamic_sizes=[
                                (
                                    DimDynamic.DYNAMIC
                                    if d in flat_args_dynamic_dims[i]
                                    else DimDynamic.STATIC
                                )
                                for d in range(len(flat_args[i].shape))
                            ],
                            constraint_sizes=[None] * len(flat_args[i].shape),
                        ),
                    )
                elif isinstance(flat_args[i], _IntWrapper):
                    arg.node.meta["val"] = flat_args[i].val
                else:
                    arg.node.meta["val"] = flat_args[i]

            self.new_args.append(arg)
        self.old_args_gen = (self.new_args[i] for i in matched_input_elements_positions)
        self.matched_output_elements_positions = matched_output_elements_positions
        self.flat_results = flat_results

    def placeholder(
        self, target: Target, args: tuple[Argument, ...], kwargs: dict[str, Any]
    ) -> Any:
        arg = next(self.old_args_gen)
        if "val" in self.current_node.meta:
            arg.node.meta["val"] = self.current_node.meta["val"]
        if "tensor_dict" in self.current_node.meta:
            arg.node.meta["tensor_dict"] = self.current_node.meta["tensor_dict"]
        if "example_value" in self.current_node.meta:
            # NB: intentionally do not use set_example_value
            arg.node.meta["example_value"] = self.current_node.meta["example_value"]
        if "unbacked_bindings" in self.current_node.meta:
            arg.node.meta["unbacked_bindings"] = self.current_node.meta[
                "unbacked_bindings"
            ]
        return arg

    def output(
        self, target: Target, args: tuple[Argument, ...], kwargs: dict[str, Any]
    ) -> Any:
        dynamo_result_flat = args[0]
        lookup = [*dynamo_result_flat, *self.new_args]  # type: ignore[misc]
        new_results_flat = []
        for i in range(len(self.flat_results)):
            if self.matched_output_elements_positions[i] is not None:
                new_results_flat.append(
                    lookup[self.matched_output_elements_positions[i]]
                )
            else:
                const_val = self.flat_results[i]
                assert isinstance(const_val, tuple(common_constant_types))
                new_results_flat.append(const_val)
        return super().output(target, (new_results_flat,), {})

    def run_node(self, n: Node) -> Any:
        self.current_node = n
        result_proxy = super().run_node(n)
        if "val" in self.current_node.meta:
            result_proxy.node.meta["val"] = self.current_node.meta["val"]
        if "example_value" in self.current_node.meta:
            # NB: intentionally do not use set_example_value
            result_proxy.node.meta["example_value"] = self.current_node.meta[
                "example_value"
            ]
        if "unbacked_bindings" in self.current_node.meta:
            result_proxy.node.meta["unbacked_bindings"] = self.current_node.meta[
                "unbacked_bindings"
            ]
        if self.current_node.op != "output":
            result_proxy.node._rename(
                getattr(self.current_node, "name", result_proxy.node.name)
            )
        return result_proxy

    def transform(self) -> torch.fx.GraphModule:
        result_gm = super().transform()
        if "dynamo_flat_name_to_original_fqn" in self.module.meta:  # type: ignore[operator]
            result_gm.meta["dynamo_flat_name_to_original_fqn"] = self.module.meta[  # type: ignore[index]
                "dynamo_flat_name_to_original_fqn"  # type: ignore[index]
            ]
        if "dynamo_compile_id" in self.module.meta:  # type: ignore[operator]
            result_gm.meta["dynamo_compile_id"] = self.module.meta["dynamo_compile_id"]  # type: ignore[index]
        return result_gm


class ExportResult(NamedTuple):
    graph_module: torch.fx.GraphModule
    guards: _guards.GuardsSet
    # NB: Do not add new fields without overriding __iter__; people are
    # destructuring so it is BC-breaking


# NOTE: this function only supports graphs created by Dynamo's OutputGraph module
def check_signature_rewritable(graph: torch.fx.GraphModule) -> None:
    input_errors = []
    for node in graph.graph.find_nodes(op="placeholder"):
        # set in OutputGraph._call_user_compiler
        assert hasattr(node, "_dynamo_source")
        assert hasattr(graph, "_source_to_user_stacks")

        # NOTE: We can safely ignore these type warnings if and only if
        # the function is made from OutputGraph (checked in the assertions)
        source = node._dynamo_source  # type: ignore[attr-defined]
        user_stacks = graph._source_to_user_stacks.get(source)  # type: ignore[operator, union-attr]
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
            msg = f"{source.name}, a closed over free variable"
        else:
            tb = "".join(traceback.format_list(stack))
            extra = ""
            if len(user_stacks) > 1:
                extra = f"(elided {len(user_stacks) - 1} more accesses)"
            msg = f"{source.name}, accessed at:\n{tb}{extra}"
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


def check_user_input_output(flat_values: list[Any], error_type: UserErrorType) -> None:
    supported_types = [
        torch.Tensor,
        torch.SymInt,
        torch.SymFloat,
        torch.SymBool,
        torch._C.ScriptObject,
        _IntWrapper,
    ] + list(common_constant_types)

    def is_supported_type(val: Any) -> bool:
        return isinstance(val, tuple(supported_types))

    value_type = "input" if error_type == UserErrorType.INVALID_INPUT else "output"
    # We only check that the outputs are not None. Inputs can be None.
    for v in flat_values:
        if not is_supported_type(v):
            if error_type == UserErrorType.INVALID_INPUT and v is None:
                continue

            raise UserError(
                error_type,
                f"It looks like one of the {value_type}s with type `{type(v)}` "
                "is not supported or pytree-flattenable. \n"
                f"Exported graphs {value_type}s can only contain the "
                f"following supported types: {supported_types}. \n"
                "If you are using a custom class object, "
                "please register a pytree_flatten/unflatten function "
                "using `torch.utils._pytree.register_pytree_node` or "
                "`torch.export.register_dataclass`.",
            )


def rewrite_signature(
    f_sig: inspect.Signature,
    graph: torch.fx.GraphModule,
    fake_mode: Optional[fake_tensor.FakeTensorMode],
    flat_args: list[Any],
    in_spec: pytree.TreeSpec,
    example_fake_inputs: list[Any],
    graph_captured_input: Iterable[Any],
    graph_captured_output: Optional[Iterable[Any]],
    dynamo_traced_result: Any,
    flat_args_dynamic_dims: list[set[int]],
) -> torch.fx.GraphModule:
    orig_args, orig_kwargs = pytree.tree_unflatten(flat_args, in_spec)

    check_user_input_output(flat_args, UserErrorType.INVALID_INPUT)
    flat_results_traced, out_spec_traced = pytree.tree_flatten(dynamo_traced_result)
    check_user_input_output(flat_results_traced, UserErrorType.INVALID_OUTPUT)

    def check_optional_input_and_error(f_sig: inspect.Signature) -> None:
        # Check if function has optional input.
        for name, param in f_sig.parameters.items():
            if param.default is not inspect.Parameter.empty:
                import torch._dynamo.graph_break_hints as graph_break_hints
                from torch._dynamo.exc import unimplemented

                log.error(
                    "Parameter %s is optional with a default value of %s",
                    name,
                    param.default,
                )
                unimplemented(
                    gb_type="rewrite_signature: cannot trace optional function input",
                    context="",
                    explanation=f"Parameter {name} is optional with a default value of {param.default}. This is not supported yet.",
                    hints=[
                        *graph_break_hints.SUPPORTABLE,
                    ],
                )

    def produce_matching(
        debug_type: str, sources: Iterable[Any], candidates: Iterable[Any]
    ) -> list[Optional[int]]:
        matched_elements_positions: list[Optional[int]] = []
        dict_of_source_vals = {}
        for i, val in enumerate(sources):
            dict_of_source_vals[id(val)] = i

        for val in candidates:
            if isinstance(val, tuple(common_constant_types)):
                matched_elements_positions.append(None)
            elif id(val) not in dict_of_source_vals:
                if debug_type == "inputs":
                    check_optional_input_and_error(f_sig)
                raise AssertionError(
                    f"Unexpectedly found a {type(val)} in the {debug_type}.\n"
                    'Please file an issue along with a paste of the logs from TORCH_LOGS="+export"',
                )
            else:
                matched_elements_positions.append(dict_of_source_vals[id(val)])

        return matched_elements_positions

    matched_input_elements_positions = produce_matching(
        "inputs", flat_args, graph_captured_input
    )

    assert graph_captured_output is not None
    matched_output_elements_positions = produce_matching(
        "outputs", list(graph_captured_output) + flat_args, flat_results_traced
    )

    new_graph = FlattenInputOutputSignature(
        graph,
        flat_args,
        matched_input_elements_positions,  # type: ignore[arg-type]
        flat_results_traced,
        matched_output_elements_positions,  # type: ignore[arg-type]
        example_fake_inputs,
        flat_args_dynamic_dims,
        fake_mode,
    ).transform()

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
    *extra_args: Any,
    aten_graph: bool = False,
    pre_dispatch: bool = False,
    decomposition_table: Optional[
        dict[torch._ops.OpOverload, Callable[..., Any]]
    ] = None,
    tracing_mode: str = "symbolic",
    dynamic_shapes: Optional[Union[dict[str, Any], tuple[Any], list[Any]]] = None,
    specialize_float: bool = True,
    assume_static_by_default: bool = False,
    same_signature: bool = True,
    disable_constraint_solver: bool = False,
    prefer_deferred_runtime_asserts_over_guards: bool = False,
    _log_export_usage: bool = True,
    constraints: Optional[list[Constraint]] = None,
    **extra_kwargs: Any,
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

        dynamic_shapes:
         An optional argument where the type should either be:
         1) a dict from argument names of ``f`` to their dynamic shape specifications,
         2) a tuple that specifies dynamic shape specifications for each input in original order.
         If you are specifying dynamism on keyword args, you will need to pass them in the order that
         is defined in the original function signature.

         The dynamic shape of a tensor argument can be specified as either
         (1) a dict from dynamic dimension indices to :func:`Dim` types, where it is
         not required to include static dimension indices in this dict, but when they are,
         they should be mapped to None; or (2) a tuple / list of :func:`Dim` types or None,
         where the :func:`Dim` types correspond to dynamic dimensions, and static dimensions
         are denoted by None. Arguments that are dicts or tuples / lists of tensors are
         recursively specified by using mappings or sequences of contained specifications.

        same_signature (bool): If True, rewrite the returned graph's signature to be the same as f.

        disable_constraint_solver (bool): Whether the dim constraint solver must be disabled.

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
    if config.debug_force_graph_break_on_leaf_return:
        raise unittest.SkipTest("Cannot force graph break on export")

    if _log_export_usage:
        log_export_usage(event="export.private_api", flags={"_dynamo"})

    # Deal with "local variable referenced before assignment"
    _f = f
    _specialize_float = specialize_float
    _assume_static_by_default = assume_static_by_default
    _constraints = constraints

    def inner(*args: Any, **kwargs: Any) -> ExportResult:
        if not _constraints:
            combined_args = _combine_args(_f, args, kwargs)
            constraints = _process_dynamic_shapes(combined_args, dynamic_shapes)
        else:
            constraints = _constraints

        f = _f
        specialize_float = _specialize_float
        assume_static_by_default = _assume_static_by_default
        check_if_dynamo_supported()
        torch._C._log_api_usage_once("torch._dynamo.export")
        if decomposition_table is not None:
            assert aten_graph, (
                "Specifying a decomposition_table table or tracing mode is illegal without setting aten_graph=True"
            )
        if pre_dispatch:
            assert aten_graph, "pre_dispatch=True can only be used when aten_graph=True"
        f = innermost_fn(f)
        call_to_inspect = f.forward if isinstance(f, torch.nn.Module) else f
        original_signature = inspect.signature(call_to_inspect)  # type: ignore[arg-type]
        graph = None
        out_guards = None
        graph_captured_input = None
        graph_captured_result: Optional[tuple[torch.Tensor, ...]] = None
        fake_mode = None
        result_traced = None

        def guard_export_print(guards: _guards.GuardsSet) -> None:
            nonlocal out_guards
            assert out_guards is None, (
                "whole graph export entails exactly one guard export"
            )
            out_guards = guards

        example_inputs: list[Any] = []

        def dynamo_normalization_capturing_compiler(
            gm: torch.fx.GraphModule, inner_example_inputs: list[Any]
        ) -> Callable[..., Any]:
            nonlocal graph
            assert graph is None, (
                "Tried to emit a second graph during export. Tracing through 'f' must produce a single graph."
            )
            graph = gm

            nonlocal fake_mode, example_inputs
            # NB: do NOT pass inner_example_inputs here, we are detecting the
            # Dynamo allocated fake mode, which should be DISTINCT from a
            # potential outer ambient fake mode which the user provided.
            # example_inputs is always the user specified inputs, so they
            # would have the wrong fake mode attached to them
            fake_mode = _guards.detect_fake_mode()
            example_inputs = inner_example_inputs

            def result_capturing_wrapper(*graph_inputs: Any) -> Any:
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

                # We reran fake tensor propagation, but we didn't do
                # anything with the resulting unbacked SymInts.  Drop them
                # from the pending list.
                # NB: this is wrong if graph_captured_result has
                # data-dependent output size!
                ignore_fresh_unbacked = null_context()
                assert ambient_fake_mode is not None
                if shape_env := ambient_fake_mode.shape_env:
                    ignore_fresh_unbacked = shape_env.ignore_fresh_unbacked_symbols()  # type: ignore[assignment]

                with (
                    ambient_fake_mode,
                    enable_python_dispatcher(),
                    ignore_fresh_unbacked,
                ):
                    params_and_buffers = {
                        **named_parameters,
                        **named_buffers,
                    }
                    fake_params_buffers = {}

                    for name, value in params_and_buffers.items():
                        fake_params_buffers[name] = ambient_fake_mode.from_tensor(
                            value, static_shapes=True
                        )

                    from torch._export.non_strict_utils import (
                        key_path_to_source,
                        KeyPath,
                    )

                    def fakify_with_ambient(
                        path: KeyPath, t: Union[torch.Tensor, _IntWrapper, Any]
                    ) -> Any:
                        if isinstance(t, torch.Tensor):
                            # pyrefly: ignore [missing-attribute]
                            return ambient_fake_mode.from_tensor(t, static_shapes=True)
                        elif isinstance(t, _IntWrapper):
                            if (
                                t.dynamism is not None
                                and isinstance(t.dynamism, _DimHint)
                                and t.dynamism.type
                                in (
                                    _DimHintType.DYNAMIC,
                                    _DimHintType.AUTO,
                                )
                            ):  # type: ignore[union-attr]
                                source = key_path_to_source(path)
                                symint = ambient_fake_mode.shape_env.create_unspecified_symint_and_symbol(  # type: ignore[union-attr]
                                    t.val, source, DimDynamic.DYNAMIC
                                )
                                return symint
                            else:
                                return t.val
                        else:
                            return t

                    fake_graph_inputs = pytree.tree_map_with_path(
                        fakify_with_ambient, graph_inputs
                    )
                    graph_captured_result = torch.func.functional_call(
                        graph,
                        fake_params_buffers,  # type: ignore[arg-type]
                        fake_graph_inputs,  # type: ignore[arg-type]
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
        with (
            config.patch(
                specialize_int=True,
                specialize_float=specialize_float,
                assume_static_by_default=assume_static_by_default,
                automatic_dynamic_shapes=False,
                capture_dynamic_output_shape_ops=True,
                capture_scalar_outputs=True,
                constant_fold_autograd_profiler_enabled=True,
                prefer_deferred_runtime_asserts_over_guards=prefer_deferred_runtime_asserts_over_guards,
                # install_free_tensors ensures that params and buffers are still
                # added as graph attributes, and makes Dynamo emits graphs that
                # follow export pytree-able input requirements
                install_free_tensors=config.install_free_tensors_for_export,
            ),
            _compiling_state_context(),
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
            # TODO(voz): We may have instances of `f` that mutate inputs, we should track sideeffects and reject.
            try:
                result_traced = opt_f(*args, **kwargs)
            except ConstraintViolationError as e:
                constraint_violation_error = e
        remove_from_cache(f)

        if (
            not disable_constraint_solver
            and (shape_env := getattr(fake_mode, "shape_env", None)) is not None
            and (dim_constraints := shape_env.dim_constraints) is not None
            and not isinstance(
                call_to_inspect, (torch._ops.OpOverloadPacket, torch._ops.OpOverload)
            )
            and not trace_rules.check(call_to_inspect)
        ):
            dim_constraints.solve()

            forced_specializations = dim_constraints.forced_specializations()

            msg = dim_constraints.prettify_results(
                original_signature,
                dynamic_shapes,
                constraint_violation_error,
                forced_specializations,
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

            for k in shape_env.var_to_range:
                if isinstance(k, sympy.Integer):
                    constraint_violation_error = ConstraintViolationError(
                        f"{''.join(traceback.format_list(shape_env.var_to_stack[k]))}\n"
                        "It appears that you're trying to set a constraint on a "
                        f"value which we evaluated to have a static value of {k}. "
                        'Set TORCH_LOGS="+export" for more information.'
                    )
        if constraint_violation_error:
            raise constraint_violation_error

        if graph is None:
            assert same_signature, (
                "Failed to produce a graph during tracing as no tensor operations were found and same_signature is False."
            )
            # If the module does not contain any tensor computation, we would create a graph with inputs and outputs.
            # To be consistent with the graph traced by dynano, `graph` will have only tensor inputs as placeholders
            # and tensor outputs as output nodes. non-tensor inputs and outputs will be added when rewriting signature.
            # We will also construct the `example_inputs`, `graph_captured_input`, and `graph_captured_result` corresponding
            # to `graph`.
            example_inputs = []
            graph_captured_input = ()
            graph_captured_result = ()
            fake_mode = torch._subclasses.FakeTensorMode(
                shape_env=ShapeEnv(), export=True
            )
            if out_guards is None:
                out_guards = _guards.GuardsSet()
            assert out_guards is not None  # suppress mypy error
            parameter_names = list(original_signature.parameters.keys())
            fx_graph = torch.fx.Graph()
            for i, name in enumerate(parameter_names):
                if torch.is_tensor(flat_args[i]):
                    node = fx_graph.placeholder(name)
                    node.meta["val"] = fake_mode.from_tensor(
                        flat_args[i], static_shapes=True
                    )
                    graph_captured_input = graph_captured_input + (flat_args[i],)
                    example_inputs.append(flat_args[i])
            fx_graph.output(graph_captured_result)
            module = torch.nn.Module()
            graph = torch.fx.GraphModule(module, fx_graph)
            log.info(
                "Failed to capture a graph during tracing as no tensor operations were found.:\n\n%s",
                graph.print_readable(print_output=False, colored=True),
            )
        else:
            assert out_guards is not None, "Failed to produce guards during tracing"
            assert fake_mode is not None

            log.info(
                "Dynamo captured graph:\n\n%s",
                graph.print_readable(print_output=False, colored=True),
            )

            # This check need to happened before aten_graph
            # because placeholder's _source_node attribute is not preserved by make_fx
            if same_signature:
                check_signature_rewritable(graph)

        # NB: This is mostly hitting the cache; Dynamo already converted these
        example_fake_inputs = [
            fake_mode.from_tensor(t) if isinstance(t, torch.Tensor) else t
            for t in example_inputs
        ]

        if aten_graph:
            # Running graph with interpreter is needed for propagating the stack_trace
            def graph_with_interpreter(*args: Any) -> Any:
                with torch.fx.traceback.preserve_node_meta():
                    return torch.fx.Interpreter(graph).run(*args)  # type: ignore[arg-type]

            with unset_fake_temporarily(), enable_python_dispatcher(), fake_mode:
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
                    raise UserError(  # noqa: B904
                        UserErrorType.DYNAMIC_CONTROL_FLOW,
                        str(e),
                        case_name="cond_operands",
                    )

            assert graph is not None
            for node in graph.graph.find_nodes(op="get_attr"):
                if isinstance(getattr(graph, node.target), torch.Tensor):  # type: ignore[arg-type]
                    node.meta["val"] = fake_mode.from_tensor(
                        getattr(graph, node.target),  # type: ignore[arg-type]
                        static_shapes=True,
                    )

        if same_signature:
            flat_args_dynamic_dims = [
                {
                    c.dim
                    for c in (constraints or ())
                    if (
                        c.t_id == id(x)
                        and not isinstance(c, _RelaxedConstraint)
                        and c.constraint_range.vr.lower != c.constraint_range.vr.upper
                    )
                }
                for x in flat_args
            ]
            graph = rewrite_signature(
                original_signature,
                graph,
                fake_mode,
                flat_args,
                in_spec,
                example_fake_inputs,
                graph_captured_input,  # type: ignore[arg-type]
                graph_captured_result,
                result_traced,  # type: ignore[possibly-undefined]
                flat_args_dynamic_dims,
            )
        return ExportResult(graph, out_guards)

    if extra_args or extra_kwargs:
        warnings.warn(
            "export(f, *args, **kwargs) is deprecated, use export(f)(*args, **kwargs) instead.  "
            "If you don't migrate, we may break your export call in the future if your user defined kwargs "
            "conflict with future kwargs added to export(f).",
            FutureWarning,
            stacklevel=2,
        )
        return inner(*extra_args, **extra_kwargs)  # type: ignore[return-value]
    else:
        return inner


def optimize_assert(*args: Any, **kwargs: Any) -> OptimizeContext:
    if "rebuild_ctx" in kwargs and kwargs["rebuild_ctx"] is not None:
        # called from optimize
        rebuild_ctx = kwargs["rebuild_ctx"]
        del kwargs["rebuild_ctx"]
    else:

        def rebuild_ctx() -> OptimizeContext:
            return optimize_assert(*args, **kwargs)

    return _optimize_assert(rebuild_ctx, *args, **kwargs)


def _optimize_assert(
    rebuild_ctx: Callable[[], OptimizeContext],
    backend: Union[str, Callable[..., Any], None],
    *,
    hooks: Hooks = Hooks(None, None, None),
    export: bool = False,
    export_constraints: Optional[Any] = None,
    dynamic: Optional[bool] = None,
    package: Optional[CompilePackage] = None,
) -> OptimizeContext:
    """
    Guarantees single-graph capture.
    The same as `torch._dynamo.optimize(backend)` but ignores
    symbolic_convert.error_on_graph_break setting.

    Used for fullgraph=True and export, since we must always error on graph breaks and ignore
    symbolic_convert.error_on_graph_break. Can also be used for testing.
    """
    backend = get_compiler_fn(backend)

    # Find if backend has any extra context manager
    backend_ctx_ctor = getattr(backend, "backend_ctx_ctor", null_context)

    if config.caching_precompile and package is None:
        # Create an uninitialized package that will be set/filled by
        # _OptimizeContext.__call__
        # We need to instantiate the object here because the same CompilePackage
        # needs to be shared between convert_frame_assert
        # and OptimizeContext.
        from .package import CompilePackage

        package = CompilePackage(fn=None, dynamo=None, ignore_inlined_sources=False)

    return _optimize_catch_errors(
        convert_frame.convert_frame_assert(
            backend,
            export=export,
            export_constraints=export_constraints,
            package=package,
        ),
        hooks,
        backend_ctx_ctor,
        fullgraph=True,
        export=export,
        dynamic=dynamic,
        rebuild_ctx=rebuild_ctx,
        package=package,
    )


class TorchPatcher:
    @staticmethod
    @functools.cache
    def patch() -> None:
        # A better way to disable the following would be decorate the source
        # functions with @torch._disable_dynamo. However, this causes issues
        # with torch.deploy internally.
        from .decorators import disable

        torch.jit.trace = disable(
            torch.jit.trace, reason="tracing into TorchScript not fully supported"
        )
        torch.jit.trace_module = disable(
            torch.jit.trace_module,
            reason="tracing into TorchScript not fully supported",
        )
        torch.jit._get_trace_graph = disable(
            torch.jit._get_trace_graph,
            reason="tracing into TorchScript not fully supported",
        )
        torch.fx._symbolic_trace.Tracer.trace = disable(
            torch.fx._symbolic_trace.Tracer.trace,
            reason="tracing into FX not fully supported",
        )
        torch.distributions.Distribution.set_default_validate_args(False)

        from torch.optim import (
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

        for opt_mod in optimizer_modules:
            opt_name = opt_mod.__name__.split(".")[-1]
            fused_fn_name = f"_fused_{opt_name}"

            if hasattr(opt_mod, fused_fn_name):
                setattr(
                    opt_mod,
                    fused_fn_name,
                    disable(
                        getattr(opt_mod, fused_fn_name),
                        reason="don't trace into fused optimizer",
                    ),
                )

        optimizer_classes = [
            opt
            for opt in torch.optim.__dict__.values()
            if inspect.isclass(opt) and issubclass(opt, torch.optim.Optimizer)
        ]

        # Note: we don't support sparsity or tracing through backwards
        excluded_optimizer_classes = {
            torch.optim.SparseAdam,
            torch.optim.LBFGS,
        }

        for opt in optimizer_classes:
            if opt in excluded_optimizer_classes:
                opt.step = disable(
                    opt.step, reason=f"optimizer {opt} step not supported"
                )

            if hasattr(opt, "_init_group"):
                opt._init_group = disable(
                    opt._init_group, reason=f"optimizer {opt} _init_group not supported"
                )

    @staticmethod
    def suppress_torch_distributed_warnings(
        fn: Callable[..., Any],
    ) -> Callable[..., Any]:
        def inner_fn(*args: Any, **kwargs: Any) -> Any:
            with torch._logging.hide_warnings(
                torch._logging._internal.user_warning_filter
            ):
                return fn(*args, **kwargs)

        return inner_fn


def skip_code(code: types.CodeType) -> None:
    set_code_exec_strategy(
        code, FrameExecStrategy(FrameAction.SKIP, FrameAction.DEFAULT)
    )
