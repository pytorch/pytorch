"""
This module implements TorchDynamo's core frame conversion functionality, transforming Python
frames into FX graphs. It handles:

- Frame analysis and bytecode transformation
- Guard creation and management for dynamic behaviors
- Cache management for recompilation
- Error handling and fallback mechanisms

Key classes:
- ConvertFrame: Main entry point for frame conversion with error handling
- ConvertFrameAssert: Implements core frame to graph conversion logic
- Tracker: Tracks input/output code objects during conversion
- CatchErrorsWrapper: Provides error handling and suppression logic

The conversion process preserves program semantics while enabling optimizations
through torch.compile() and related systems.

NOTE: _torchdynamo_orig_backend is used for convert frame wrappers to identify the inner wrapped function.
By going down the _torchdynamo_orig_backend chain, one can recover the original unwrapped backend,
which is checked for during the Dynamo cache lookup.
"""

from __future__ import annotations

import collections
import contextlib
import cProfile
import dis
import functools
import gc
import inspect
import itertools
import logging
import os
import pstats
import random
import subprocess
import sys
import threading
import time
import traceback
import types
import typing
import weakref
from dataclasses import dataclass
from pathlib import Path
from types import CellType, CodeType, FunctionType, ModuleType
from typing import Any, Optional, TypeVar, Union
from typing_extensions import ParamSpec
from weakref import ReferenceType

import torch
import torch._logging
from torch._C._dynamo.guards import GlobalStateGuard
from torch._dynamo.callback import CallbackTrigger
from torch._dynamo.distributed import get_compile_pg
from torch._dynamo.symbolic_convert import TensorifyState
from torch._guards import compile_context, CompileContext, CompileId, tracing
from torch._logging import structured
from torch._utils_internal import (
    compile_time_strobelight_meta,
    justknobs_check,
    maybe_upload_prof_stats_to_manifold,
    signpost_event,
)
from torch.fx._lazy_graph_module import _use_lazy_graph_module
from torch.fx.experimental.symbolic_shapes import (
    ConstraintViolationError,
    GuardOnDataDependentSymNode,
)
from torch.fx.graph_module import _forward_from_src as original_forward_from_src
from torch.monitor import _WaitCounter
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.utils._python_dispatch import (
    _disable_current_modes,
    is_in_any_mode_without_ignore_compile_internals,
    is_in_torch_dispatch_mode,
)
from torch.utils._traceback import CapturedTraceback, format_traceback_short

from . import config, decorators, exc, graph_break_hints, trace_rules
from .bytecode_analysis import remove_dead_code, remove_pointless_jumps
from .bytecode_transformation import (
    check_inst_exn_tab_entries_valid,
    Instruction,
    is_generator,
    propagate_inst_exn_table_entries,
    transform_code_object,
)
from .cache_size import (
    CacheSizeRelevantForFrame,
    compute_cache_size,
    exceeds_recompile_limit,
    is_recompilation,
)
from .eval_frame import (
    always_optimize_code_objects,
    Constraint,
    dynamo_tls,
    skip_code,
    TorchPatcher,
)
from .exc import (
    augment_exc_message,
    BackendCompilerFailed,
    FailOnRecompileLimitHit,
    format_error_msg,
    InternalTorchDynamoError,
    PackageError,
    RecompileLimitExceeded,
    ResumePrologueTracingError,
    ShortenTraceback,
    SkipCodeRecursiveException,
    TorchRuntimeError,
    UncapturedHigherOrderOpError,
    unimplemented,
    Unsupported,
)
from .graph_bytecode_inputs import reset_user_object_tracking
from .guards import (
    CheckFunctionManager,
    get_and_maybe_log_recompilation_reasons,
    GuardedCode,
)
from .hooks import Hooks
from .output_graph import DynamoTracerOutput, OutputGraphCommon
from .pgo import (
    _log_size_mismatch_recompile,
    log_frame_dynamic_whitelist,
    put_code_state,
)
from .replay_record import ExecutionRecord
from .resume_execution import TORCH_DYNAMO_RESUME_IN_PREFIX
from .symbolic_convert import (
    DistributedState,
    ExceptionStack,
    InstructionTranslator,
    LocalState,
    SpeculationLog,
)
from .trace_rules import is_numpy
from .types import ConvertFrameReturn, FrameAction, FrameExecStrategy, wrap_guarded_code
from .utils import (
    _get_error_on_graph_break,
    chromium_event_timed,
    CleanupManager,
    CompileTimeInstructionCounter,
    counters,
    dynamo_timed,
    format_bytecode,
    gen_record_file_name,
    get_hook_for_recompile_user_context,
    get_metrics_context,
    increment_frame,
    is_namedtuple,
    istype,
    LazyString,
    maybe_disable_inference_mode,
    maybe_disable_inference_mode_for_fake_prop,
    orig_code_map,
    reset_graph_break_dup_checker,
    setup_compile_debug,
    to_int_us,
    troubleshooting_url,
    write_record_to_file,
)
from .variables.torch_function import torch_function_mode_stack_state_mgr


np: Optional[ModuleType]
try:
    import numpy as np
except ModuleNotFoundError:
    np = None


if typing.TYPE_CHECKING:
    from collections.abc import Callable

    from torch.utils.weak import WeakIdKeyDictionary

    from .backends.registry import CompilerFn
    from .package import CompilePackage
    from .repro.after_dynamo import WrapBackendDebug
    from .types import BytecodeHook, CacheEntry, DynamoFrameType
    from .variables.builder import FrameStateSizeEntry


log = logging.getLogger(__name__)
bytecode_log = torch._logging.getArtifactLogger(__name__, "bytecode")
graph_break_log = torch._logging.getArtifactLogger(__name__, "graph_breaks")


compile_lock = threading.RLock()

_T = TypeVar("_T")
_P = ParamSpec("_P")


class TODO_UNKNOWN:
    pass


class Tracker:
    def __init__(self) -> None:
        self.seen: list[ReferenceType[CodeType]] = []
        self.seen_ids: set[int] = set()

    def add(self, strong_obj: CodeType) -> None:
        idx = id(strong_obj)
        if idx not in self.seen_ids:
            obj = weakref.ref(strong_obj, lambda _: self.seen_ids.remove(idx))
            self.seen.append(obj)
            self.seen_ids.add(idx)

    def __contains__(self, item: CodeType) -> bool:
        return id(item) in self.seen_ids

    def clear(self) -> None:
        self.seen.clear()
        self.seen_ids.clear()


input_codes = Tracker()
output_codes = Tracker()

initial_global_state: Optional[GlobalStateGuard] = None


@functools.wraps(original_forward_from_src)
def fx_forward_from_src_skip_result(
    src: str, globals: dict[str, Any], co_fields: Optional[dict[str, str]] = None
) -> FunctionType:
    # we monkey patch FX to prevent infinite loop of trying to convert
    # our generated code
    result = original_forward_from_src(src, globals, co_fields)
    skip_code(result.__code__)
    return result


def log_dynamo_start(code: CodeType, skip: int = 0) -> list[str]:
    convert_frame_intern = structured.intern_string(__file__)
    captured_tb = CapturedTraceback.extract(skip=4 + skip).summary()
    frames_interned = structured.from_traceback(captured_tb)
    # Extract and filter the stack
    stack = list(
        itertools.takewhile(
            lambda f: f["filename"] != convert_frame_intern,
            frames_interned,
        )
    ) + [
        {
            "line": code.co_firstlineno,
            "name": code.co_name,
            "filename": structured.intern_string(code.co_filename),
        }
    ]
    # Initialize the ChromiumEventLogger on start
    torch._logging.trace_structured(
        "dynamo_start",
        lambda: {"stack": stack},
    )

    # Capture stack separately without using from_traceback to get the actual filenames
    stack_strings = [
        f"Line: {frame.lineno}, Name: {frame.name}, Filename: {frame.filename}"
        for frame in captured_tb
        if frame.filename != convert_frame_intern
    ] + [
        f"Line: {code.co_firstlineno}, Name: {code.co_name}, Filename: {code.co_filename}"
    ]
    return stack_strings


def preserve_global_state(fn: Callable[_P, _T]) -> Callable[_P, _T]:
    """
    Context manager to:
        1) Save/restore torch.is_grad_enabled() state
        2) Save/restore python random state
        3) Save/restore torch random state
        4) Monkey patch torch.fx.graph_module._forward_from_src
    """

    @functools.wraps(fn)
    def _fn(*args: _P.args, **kwargs: _P.kwargs) -> _T:
        guards = GlobalStateGuard()
        prior_grad_mode = torch.is_grad_enabled()

        # Just in case we get left in a bad dispatch state we want to restore
        # it. This can happen because the dispatch bits aren't a true
        # stack/counter - so we can't just increment/decrement them as we enter
        # and leave.
        with (
            torch._C._PreserveDispatchKeyGuard(),
            maybe_disable_inference_mode(),
            maybe_disable_inference_mode_for_fake_prop(),
        ):
            prior_inference_mode = torch.is_inference_mode_enabled()
            prior_deterministic = torch.are_deterministic_algorithms_enabled()
            prior_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
            prior_mobile_allocator_state = (
                torch._C._is_default_mobile_cpu_allocator_set()
            )
            py_rng_state = random.getstate()
            prior_dtype = torch.get_default_dtype()
            torch_rng_state = torch.random.get_rng_state()
            cuda_rng_state = None
            if torch.cuda.is_available():
                with torch._C.DisableTorchFunction():
                    cuda_rng_state = torch.cuda.get_rng_state()
            cuda_matmul_fp32_prec = torch._C._get_fp32_precision_getter(
                "cuda", "matmul"
            )
            prior_fwd_from_src = torch.fx.graph_module._forward_from_src
            torch.fx.graph_module._forward_from_src = fx_forward_from_src_skip_result
            cleanup = setup_compile_debug()
            exit_stack = contextlib.ExitStack()
            exit_stack.enter_context(
                torch.fx._symbolic_trace._maybe_revert_all_patches()
            )
            exit_stack.enter_context(torch_function_mode_stack_state_mgr)
            reset_user_object_tracking()
            try:
                return fn(*args, **kwargs)
            finally:
                cleanup.close()
                assert torch._C._len_torch_function_stack() == 0, (
                    "Torch function mode stack state changed while dynamo tracing, please report a bug"
                )
                exit_stack.close()
                torch._C._set_grad_enabled(prior_grad_mode)
                torch.autograd.grad_mode._enter_inference_mode(prior_inference_mode)
                torch.use_deterministic_algorithms(
                    prior_deterministic, warn_only=prior_warn_only
                )
                random.setstate(py_rng_state)
                torch.random.set_rng_state(torch_rng_state)
                torch.set_default_dtype(prior_dtype)
                curr_mobile_allocator_state = (
                    torch._C._is_default_mobile_cpu_allocator_set()
                )
                if prior_mobile_allocator_state != curr_mobile_allocator_state:
                    torch._C._unset_default_mobile_cpu_allocator()
                if cuda_rng_state is not None:
                    with torch._C.DisableTorchFunction():
                        torch.cuda.set_rng_state(cuda_rng_state)
                torch._C._set_fp32_precision_setter(
                    "cuda", "matmul", cuda_matmul_fp32_prec
                )
                torch.fx.graph_module._forward_from_src = prior_fwd_from_src
                assert guards.check(), (
                    f"Global {guards.reason()}state changed while dynamo tracing, please report a bug"
                )

    _fn._torchdynamo_orig_backend = fn  # type: ignore[attr-defined]
    return _fn


@TorchPatcher.suppress_torch_distributed_warnings
def has_tensor_in_frame(frame: DynamoFrameType) -> bool:
    """Check if the frame has torch.* related bits"""
    # Check if the function was decorated using torch._dynamo.optimize
    if frame.f_code in always_optimize_code_objects:
        return True

    # Check if there is global import of torch.*
    for co_name in frame.f_code.co_names:
        if co_name in frame.f_globals:
            obj = frame.f_globals[co_name]
            if isinstance(obj, ModuleType) and (
                obj.__name__.startswith("torch.") or obj is torch
            ):
                return True
            # ... or a global import of numpy.*
            if np and config.trace_numpy and (obj is np or is_numpy(obj)):
                return True

    seen_ids: dict[int, bool] = {}

    def has_tensor(obj: object) -> bool:
        """Recursively check if the obj has a tensor"""
        obj_id = id(obj)
        if obj_id in seen_ids:
            return seen_ids[obj_id]
        seen_ids[obj_id] = False

        if isinstance(obj, (torch.Tensor, torch.nn.Module)) or (
            istype(obj, type) and issubclass(obj, torch.nn.Module)
        ):
            seen_ids[obj_id] = True
            return seen_ids[obj_id]
        elif (
            config.trace_numpy
            and np
            and (istype(obj, np.ndarray) or isinstance(obj, np.generic))
        ):
            seen_ids[obj_id] = True
            return seen_ids[obj_id]
        elif istype(obj, (list, tuple)):
            seen_ids[obj_id] = any(has_tensor(v) for v in obj)
            return seen_ids[obj_id]
        elif istype(obj, dict):
            # Some packages like pytest can be updated during runtime. So, make a
            # copy of values to avoid issues like "RuntimeError: dictionary
            # changed size during iteration"
            values = list(obj.values())
            seen_ids[obj_id] = any(has_tensor(v) for v in values)
            return seen_ids[obj_id]
        elif istype(obj, (str, int, float, type(None), bool)):
            seen_ids[obj_id] = False
            return seen_ids[obj_id]
        elif is_namedtuple(obj) and hasattr(obj, "_fields"):
            seen_ids[obj_id] = any(has_tensor(getattr(obj, v)) for v in obj._fields)
            return seen_ids[obj_id]
        else:
            # if config.debug:
            #     print(
            #         f"Assuming that object of type {type(obj)} does not have a tensor"
            #     )
            return False

    # Check if the passed arguments are of type Tensor
    for value in frame.f_locals.values():
        if has_tensor(value):
            return True

    log.debug(
        "skipping because no torch.* %s \
            %s %s",
        frame.f_code.co_name,
        frame.f_code.co_filename,
        frame.f_code.co_firstlineno,
    )

    return False


def exception_handler(
    e: Exception,
    code: CodeType,
    frame: Optional[DynamoFrameType] = None,
    export: bool = False,
) -> None:
    record_filename = None
    if hasattr(e, "exec_record"):
        record_filename = gen_record_file_name(e, code)
        write_record_to_file(record_filename, e.exec_record)
        e.record_filename = record_filename  # type: ignore[attr-defined]

    augment_exc_message(e, export=export)


FRAME_COUNTER = 0
FRAME_COMPILE_COUNTER: typing.Counter[Union[int, FrameStateSizeEntry]] = (
    collections.Counter()
)


def maybe_cprofile(func: Callable[_P, _T]) -> Callable[_P, _T]:
    if config.cprofile:
        return cprofile_wrapper(func)
    return func


def cprofile_wrapper(func: Callable[_P, _T]) -> Callable[_P, _T]:
    @functools.wraps(func)
    def profile_wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _T:
        trace_id = CompileContext.current_trace_id()
        assert trace_id, "Trace id is None"
        profile_path = Path(
            f"/tmp/{func.__name__}_{str(trace_id).replace('/', '_')}.profile"
        )
        prof = cProfile.Profile()
        try:
            prof.enable()
            start_ts = time.time()
            # pyrefly: ignore [bad-argument-type]
            retval = prof.runcall(func, *args, **kwargs)
            profile_latency = time.time() - start_ts
            prof.disable()
        except ValueError:
            log.exception("failed to enable cProfile")
            profile_latency = 0
            retval = func(*args, **kwargs)
        log.warning(
            "### Cprofile for %s trace id [%s] took %.3f seconds ###",
            func.__name__,
            trace_id,
            profile_latency,
        )
        ps = pstats.Stats(prof)
        try:
            prof.dump_stats(profile_path)
        except OSError:
            log.exception("Cannot write to %s", profile_path)
        log.warning("Raw profile at %s", profile_path)
        svg_path = profile_path.with_suffix(".svg")
        try:
            gprof2dot_process = subprocess.Popen(
                [
                    "gprof2dot",
                    "-f",
                    "pstats",
                    "--node-label=total-time-percentage",
                    "--node-label=self-time-percentage",
                    "--node-label=total-time",
                    str(profile_path),
                ],
                stdout=subprocess.PIPE,
            )
            subprocess.check_call(
                ["dot", "-Tsvg", "-o", str(svg_path)],
                stdin=gprof2dot_process.stdout,
            )
            log.warning("Generated SVG from profile at %s", svg_path)
        except FileNotFoundError:
            log.warning(
                "Failed to generate SVG from profile -- dumping stats instead."
                "Try installing gprof2dot and dot for a better visualization"
            )
            ps.sort_stats(pstats.SortKey.TIME).print_stats(20)
            ps.sort_stats(pstats.SortKey.CUMULATIVE).print_stats(20)

        if manifold_link := maybe_upload_prof_stats_to_manifold(
            str(profile_path)
        ):  # fb-only
            torch._logging.trace_structured(
                "link",
                lambda: {"name": "cprofile_manifold_url", "url": manifold_link},
            )
        return retval

    return profile_wrapper


@dataclass
class ConvertFrameBox:
    error_on_graph_break: Optional[bool] = None


def get_compile_id(
    frame_state: dict[str, Union[int, FrameStateSizeEntry]],
) -> CompileId:
    global FRAME_COUNTER
    if "_id" not in frame_state:
        frame_state["_id"] = FRAME_COUNTER
        FRAME_COUNTER += 1
    frame_id = frame_state["_id"]
    assert isinstance(frame_id, int)

    frame_compile_id = FRAME_COMPILE_COUNTER[frame_id]
    FRAME_COMPILE_COUNTER[frame_id] += 1

    compiled_autograd_id = None
    if prior := CompileContext.current_compile_id():
        compiled_autograd_id = prior.compiled_autograd_id
    return CompileId(
        compiled_autograd_id=compiled_autograd_id,
        frame_id=frame_id,
        frame_compile_id=frame_compile_id,
    )


class ConvertFrameAssert:
    def __init__(
        self,
        compiler_fn: CompilerFn,
        one_graph: bool = True,
        export: bool = False,
        export_constraints: Optional[typing.Never] = None,
        package: Optional[CompilePackage] = None,
    ) -> None:
        # assert export_constraints is None
        reset_graph_break_dup_checker()
        self._torchdynamo_orig_backend = compiler_fn
        self._one_graph = one_graph
        self._export = export
        self._export_constraints = export_constraints
        self._package = package
        self._box = ConvertFrameBox()

    @property
    def _clone_with_backend(self) -> Callable[[CompilerFn], ConvertFrameAssert]:
        return lambda backend: convert_frame_assert(
            backend,
            self._one_graph,
            self._export,
            self._export_constraints,
        )

    def __call__(
        self,
        frame: DynamoFrameType,
        cache_entry: Optional[CacheEntry],
        hooks: Hooks,
        frame_state: dict[str, Union[int, FrameStateSizeEntry]],
        *,
        skip: int = 0,
    ) -> ConvertFrameReturn:
        increment_frame()
        code = frame.f_code

        cache_size = compute_cache_size(frame, cache_entry)
        input_codes.add(code)
        if code in output_codes:
            return ConvertFrameReturn()
        if (
            os.environ.get("TORCHDYNAMO_DEBUG_FUNCTION")
            and os.environ.get("TORCHDYNAMO_DEBUG_FUNCTION") != code.co_name
        ):
            return ConvertFrameReturn()
        if code.co_name == "<genexpr>" and code.co_filename.endswith(
            (
                "transformers/file_utils.py",
                "transformers/utils/generic.py",
                "diffusers/utils/outputs.py",
            )
        ):
            # not needed, but cleans up torchbench error stats
            return ConvertFrameReturn()
        if code.co_name == "__setattr__":
            # setattr could be tricky to handle generally,
            # but also not likely useful to compile- skip the whole frame
            return ConvertFrameReturn()
        if code.co_name == "__init__" and code.co_filename.startswith(
            os.path.dirname(torch.optim.__file__)
        ):
            # optimizer support is still incomplete see
            # test_state_dict in test/dynamo/test_optimizers.py
            return ConvertFrameReturn()

        # Check if the frame is generated by an exec builtin call
        # TODO - Running exec generated frame seems propagates f_globals to the
        # next frames.
        if code.co_name == "<module>" and code.co_filename == "<string>":
            return ConvertFrameReturn()

        if (
            code.co_name == "<lambda>"
            and code.co_filename == "<string>"
            and not bool(frame.f_builtins)
        ):
            # namedtuple subclass constructor. Empty builtins cause issue with
            # len keyword in LIST_LEN guard.
            return ConvertFrameReturn()

        if is_generator(code):
            unimplemented(
                gb_type="Attempt to trace generator",
                context="",
                explanation="Generators cannot be compiled directly with `torch.compile`.",
                hints=[
                    "Call a generator from inside of a non-generator Python function and "
                    "compile that function instead.",
                    *graph_break_hints.FUNDAMENTAL,
                ],
            )

        if not has_tensor_in_frame(frame):
            return ConvertFrameReturn()

        # skip tracing non-recursive disabled functions
        # detect if the previous frame (non-convert_frame) is a non-recursive disable wrapper
        prev_frame = sys._getframe()
        while (
            prev_frame
            and "torch/_dynamo/convert_frame.py" in prev_frame.f_code.co_filename
        ):
            prev_frame = prev_frame.f_back  # type: ignore[assignment]
        if (
            prev_frame
            and prev_frame.f_code is decorators._nonrecursive_disable_wrapper_code
        ):
            return ConvertFrameReturn(apply_to_code=False)

        global initial_global_state
        initial_global_state = GlobalStateGuard()

        compile_id = get_compile_id(frame_state)
        frame_id = compile_id.frame_id

        signpost_event(
            "dynamo",
            "_convert_frame_assert._compile",
            {
                "co_name": code.co_name,
                "frame_id": frame_id,
                "compile_id": str(compile_id),
                "co_filename": code.co_filename,
                "co_firstlineno": code.co_firstlineno,
                "cache_size": cache_size.num_cache_entries_with_same_id_matched_objs,
                "accumulated_cache_size": cache_size.num_cache_entries,
            },
        )

        # Record traced frames, skipping Dynamo generated ones.
        if not code.co_name.startswith(TORCH_DYNAMO_RESUME_IN_PREFIX):
            info = f"{code.co_name} {code.co_filename}:{code.co_firstlineno}"
            dynamo_tls.traced_frame_infos.append(info)

        with compile_context(CompileContext(compile_id)):
            result = _compile(
                frame.f_code,
                frame.f_globals,
                frame.f_locals,
                frame.f_builtins,
                frame.closure,
                self._torchdynamo_orig_backend,
                self._one_graph,
                self._export,
                self._export_constraints,
                hooks,
                cache_entry,
                cache_size,
                frame,
                frame_state=frame_state,
                compile_id=compile_id,
                skip=skip + 1,
                package=self._package,
                convert_frame_box=self._box,
            )

        if config.caching_precompile and self._package is not None:
            from .package import DynamoCache

            # Record that the dynamo package has changed
            DynamoCache.record_package(self._package)
        return result


def convert_frame_assert(
    compiler_fn: CompilerFn,
    one_graph: bool = True,
    export: bool = False,
    export_constraints: Optional[typing.Never] = None,
    package: Optional[CompilePackage] = None,
) -> ConvertFrameAssert:
    """Fully convert a frame into an FX graph, raising an exception if we fail."""
    return ConvertFrameAssert(
        compiler_fn, one_graph, export, export_constraints, package
    )


from collections import OrderedDict

from torch.utils.hooks import RemovableHandle


# we have to use `OrderedDict` to make `RemovableHandle` work.
_bytecode_hooks: dict[int, BytecodeHook] = OrderedDict()


def register_bytecode_hook(hook: BytecodeHook) -> RemovableHandle:
    """Register hooks for bytecode generated by Dynamo. The hook can do some
    logging, as well as return a new code object to be used. Please refer
    to `BytecodeHook` for the hook signature.
    """
    handle = RemovableHandle(_bytecode_hooks)
    _bytecode_hooks[handle.id] = hook
    return handle


# TODO - We want to run preserve_node_meta context manager here, but the CI
# fails (its unclear if the failures were flaky)
# @torch.fx.traceback.preserve_node_meta()
@preserve_global_state
def trace_frame(
    code: types.CodeType,
    globals: dict[str, object],
    locals: dict[str, object],
    builtins: dict[str, object],
    closure: tuple[CellType],
    compiler_fn: CompilerFn,
    tf_mode_stack: list[torch.overrides.TorchFunctionMode],
    one_graph: bool,
    speculation_log: SpeculationLog,
    instructions: list[Instruction],
    code_options: dict[str, object],
    *,
    export: bool = False,
    export_constraints: Optional[typing.Never] = None,
    frame_state: Optional[dict[str, Union[int, FrameStateSizeEntry]]] = None,
    distributed_state: Optional[DistributedState] = None,
    package: Optional[CompilePackage] = None,
) -> DynamoTracerOutput:
    from torch.fx.experimental.validator import bisect, translation_validation_enabled

    speculation_log.restart()  # type: ignore[has-type]
    exn_vt_stack = ExceptionStack()
    tracer = InstructionTranslator(
        instructions,
        code,
        locals,
        globals,
        builtins,
        closure,
        tf_mode_stack,
        code_options,
        compiler_fn,
        one_graph,
        export,
        export_constraints,
        frame_state=frame_state,
        speculation_log=speculation_log,  # type: ignore[has-type]
        exn_vt_stack=exn_vt_stack,
        distributed_state=distributed_state,  # type: ignore[has-type]
        package=package,
    )

    def run_tracer() -> None:
        try:
            tracer.output.mark_bytecode_tracing_start()
            with tracing(tracer.output.tracing_context), tracer.set_current_tx():
                tracer.run()
        except exc.UnspecializeRestartAnalysis:
            speculation_log.clear()  # type: ignore[has-type]
            raise
        except (
            exc.SpeculationRestartAnalysis,
            exc.TensorifyScalarRestartAnalysis,
            exc.SkipFrame,
        ):
            raise
        except Exception:
            if translation_validation_enabled():
                bisect(tracer.output.shape_env)
            raise
        finally:
            tracer.output.call_cleanup_hooks()
            tracer.f_locals = {}

    try:
        run_tracer()
        tracer_output = DynamoTracerOutput(tracer)
        output = tracer_output.output_graph
        assert output is not None
        assert output.output_instructions
        instructions[:] = output.output_instructions
        code_options.update(output.code_options)
        propagate_inst_exn_table_entries(instructions)
        check_inst_exn_tab_entries_valid(instructions)
        instructions[:] = remove_pointless_jumps(remove_dead_code(instructions))
    except Exception as e:
        e._torch_dynamo_tracer_output = DynamoTracerOutput(tracer, error=True)  # type: ignore[attr-defined]
        raise

    return tracer_output


@dataclass
class DynamoOutput:
    """
    Represents the core data returned from a single dynamo run, including:
      - Guards, wrapped inside tracer_output.output_graph.guards
      - Generated bytecode
      - Other information needed for compilation.
    This data structure should capture all the "interesting" information dynamo
    produces on the frontend side before it enters user backend.
    """

    tracer_output: DynamoTracerOutput
    bytecode: types.CodeType
    last_attempt_start_time: Optional[float]

    def build_guards(
        self,
        code: types.CodeType,
        hooks: Optional[Hooks] = None,
        save: bool = False,
        cache_entry: Optional[CacheEntry] = None,
        strict_error: bool = False,
    ) -> CheckFunctionManager:
        output_graph = self.tracer_output.output_graph
        assert output_graph is not None
        return CheckFunctionManager(
            code,
            output_graph,
            cache_entry,
            hooks.guard_fail_fn if hooks else None,
            hooks.guard_filter_fn if hooks else None,
            save_guards=save,
            strict_error=strict_error,
        )

    def graph_capture_output(
        self, argdefs: Optional[tuple[Any, ...]] = None
    ) -> GraphCaptureOutput:
        output_graph = self.tracer_output.output_graph
        assert output_graph is not None
        return GraphCaptureOutput(
            OutputGraphCommon(
                output_graph.dump_guards_state(),
                output_graph.import_sources,
                output_graph.shape_env,
                output_graph.export_metadata,
                output_graph.tracked_fakes_id_to_source,
            ),
            output_graph.import_sources,
            output_graph.traced_code,
            self.bytecode,
            self.tracer_output.closure,
            argdefs,
        )


@dataclass
class BackendInput:
    """
    Represents core data structure that dynamo will pass to a backend, including:
      - Graph module
      - Example inputs
      - The FakeTensorMode used for compiling graph.
    This data structure should capture all the information dynamo produces
    on for the user backend.
    """

    backend_id: str
    graph_module: torch.fx.GraphModule
    example_inputs: Any
    fake_mode: torch._subclasses.fake_tensor.FakeTensorMode
    tensor_to_context: WeakIdKeyDictionary


@dataclass
class GraphCaptureOutput:
    """
    Minimal version of DynamoOutput
    """

    output_graph: OutputGraphCommon
    import_sources: dict[str, str]
    traced_code: list[CodeType]
    bytecode: CodeType
    closure: Optional[tuple[Any, ...]]
    argdefs: Optional[tuple[Any, ...]]

    def build_guards(
        self,
        code: types.CodeType,
        hooks: Optional[Hooks] = None,
        save: bool = False,
        cache_entry: Optional[CacheEntry] = None,
        strict_error: bool = False,
    ) -> CheckFunctionManager:
        return CheckFunctionManager(
            code,
            self.output_graph,
            cache_entry,
            hooks.guard_fail_fn if hooks else None,
            hooks.guard_filter_fn if hooks else None,
            save_guards=save,
            strict_error=strict_error,
        )


@dataclass
class CaptureOutput:
    """
    CaptureOutput should represent all the information produced from torch
    compiler for a single graph capture. This intends to be consumed by
    various compiler frontends so that we can share as much compiler internals
    as possible and avoid great divergence between different stacks.
    This data structure should eventually contain all the information compiler
    produces as more refactors happens to converge different compiler
    frontends.
    """

    graph_capture_output: GraphCaptureOutput
    # BackendInput can be None when dynamo didn't compile any graph (no tensor op)
    backend_input: Optional[BackendInput]

    def forward_callable(self) -> Callable[..., Any]:
        import importlib

        # TODO code sharing
        import_sources = self.graph_capture_output.output_graph.import_sources
        assert self.backend_input is not None
        backend_id = self.backend_input.backend_id
        import_sources = {
            alias: importlib.import_module(module_name)
            for alias, module_name in import_sources.items()
        }
        f_globals = {
            **import_sources,
            backend_id: self.backend_input.graph_module,
        }
        return types.FunctionType(
            self.graph_capture_output.bytecode,
            f_globals,
            closure=self.graph_capture_output.closure,
            argdefs=self.graph_capture_output.argdefs,
        )


def get_traced_fn(mod: Any) -> tuple[FunctionType, Optional[object]]:
    """
    Utility function to get the function to trace, and optionally a bound self
    object, from a callable (nn.Module, function, or method).
    """
    import inspect

    if isinstance(mod, torch.nn.Module):
        # Mirrored from NNModuleVariable.call_function:
        # https://github.com/pytorch/pytorch/blob/main/torch/_dynamo/variables/nn_module.py#L1035
        if (
            len(mod._forward_pre_hooks) == 0
            and len(mod._forward_hooks) == 0
            and len(torch.nn.modules.module._global_forward_pre_hooks) == 0
            and len(torch.nn.modules.module._global_forward_hooks) == 0
            and len(mod._backward_pre_hooks) == 0
            and len(mod._backward_hooks) == 0
            and len(torch.nn.modules.module._global_backward_pre_hooks) == 0
            and len(torch.nn.modules.module._global_backward_hooks) == 0
        ):
            mod = mod.forward
        elif isinstance(mod, torch.fx.GraphModule):
            mod = mod._call_impl
        else:
            mod = mod.__call__

    if hasattr(mod, "__self__"):
        # pyrefly: ignore [missing-attribute]
        return mod.__func__, mod.__self__
    elif inspect.isfunction(mod):
        return mod, None
    else:
        raise RuntimeError(f"Unsupported model code type {mod}")


def _get_signature(fn: Any) -> inspect.Signature:
    return inspect.signature(fn, follow_wrapped=False)


def _get_frame(
    mod: Any,
    args: tuple[Any, ...],
    kwargs: Optional[dict[str, Any]] = None,
) -> FrameInfo:
    """
    Create a frame to trace, given a model, args, and optional kwargs.
    """
    import builtins

    fn, self_opt = get_traced_fn(mod)
    if self_opt is not None:
        args = (self_opt,) + args
    if kwargs is None:
        kwargs = {}

    signature = _get_signature(fn)
    bound_arguments = signature.bind(*args, **kwargs)
    bound_arguments.apply_defaults()
    f_locals = bound_arguments.arguments

    closure = fn.__closure__ or ()
    freevars = fn.__code__.co_freevars
    if freevars or closure:
        assert len(closure) == len(freevars)
        f_locals.update(
            {name: cell.cell_contents for name, cell in zip(freevars, closure)}
        )

    return FrameInfo(
        fn.__code__,
        fn.__globals__,
        f_locals,
        builtins.__dict__,
        closure=fn.__closure__ or (),  # type: ignore[arg-type]
        argdefs=fn.__defaults__,
    )


def fullgraph_capture(
    mod: Any,
    args: tuple[Any, ...],
    kwargs: Optional[dict[str, Any]] = None,
    *,
    constraints: Optional[list[Constraint]] = None,
    _is_export_deprecated_do_not_use: bool = False,
) -> CaptureOutput:
    """
    This API captures a full graph for a model, given example inputs to trace with.

    Specifically, it takes a callable (nn.Module, method, or function), args, and
    optional kwargs, and returns Dynamo-captured graph along with other important
    compile-time information. This serves as the common graph-capture mechanism
    for different torch compiler AOT frontends (e.g. AOT precompile, export).

    Note that this API doesn't apply context managers like metrics context,
    and the expectation is that the caller will apply them depending
    on the use case.

    The CaptureOutput is separated into two parts:
    1. Frontend specific information, which includes:
        - guards
        - generated bytecode
        - other information tracked by OutputGraphCommon.
    2. Backend specific information (indexed by unique backend id) such as:
        - fx graph
        - example inputs
    """
    frame = _get_frame(mod, args, kwargs)

    with compile_context(CompileContext(get_compile_id({}))):
        return _fullgraph_capture_frame(
            frame,
            constraints=constraints,
            _is_export_deprecated_do_not_use=_is_export_deprecated_do_not_use,
        )


@dataclass
class FrameInfo:
    code: types.CodeType
    globals: dict[str, object]
    locals: dict[str, object]
    builtins: dict[str, object]
    closure: tuple[CellType]
    argdefs: Optional[tuple[Any, ...]]


def _fullgraph_capture_frame(
    frame: FrameInfo,
    *,
    constraints: Optional[list[Constraint]] = None,
    _is_export_deprecated_do_not_use: bool = False,
) -> CaptureOutput:
    from torch._guards import TracingContext

    backend_input: Optional[BackendInput] = None

    def fullgraph_compiler(
        gm: torch.fx.GraphModule, example_inputs: list[torch.Tensor]
    ) -> torch.fx.GraphModule:
        nonlocal backend_input
        tracing_context = TracingContext.get()
        fake_mode = tracing_context.fake_mode
        tensor_to_context = tracing_context.tensor_to_context
        assert fake_mode is not None
        assert isinstance(gm.meta["backend_id"], str)
        backend_input = BackendInput(
            gm.meta["backend_id"], gm, example_inputs, fake_mode, tensor_to_context
        )
        return gm

    try:
        dynamo_output = compile_frame(
            frame.code,
            frame.globals,
            frame.locals,
            frame.builtins,
            frame.closure,
            compiler_fn=fullgraph_compiler,
            export=_is_export_deprecated_do_not_use,
            export_constraints=constraints,  # type: ignore[arg-type]
            one_graph=True,
            restart_reasons=set(),
        )
        # https://github.com/pytorch/pytorch/blob/main/torch/_dynamo/eval_frame.py#L831
    except Unsupported as e:
        augment_exc_message(e)
        if config.verbose:
            raise
        # strip internal tracebacks from causes
        cur_exn: BaseException = e
        while cur_exn.__cause__ is not None:
            cur_exn.__cause__.with_traceback(None)
            cur_exn = cur_exn.__cause__
        # pyrefly: ignore [invalid-inheritance]
        raise e.with_traceback(None) from e.__cause__  # User compiler error

    return CaptureOutput(
        dynamo_output.graph_capture_output(frame.argdefs),
        backend_input,
    )


def compile_frame(  # type: ignore[return]
    code: types.CodeType,
    globals: dict[str, object],
    locals: dict[str, object],
    builtins: dict[str, object],
    closure: tuple[CellType],
    compiler_fn: CompilerFn,
    one_graph: bool,
    restart_reasons: set[str],
    *,
    export: bool = False,
    export_constraints: Optional[typing.Never] = None,
    frame_state: Optional[dict[str, Union[int, FrameStateSizeEntry]]] = None,
    distributed_state: Optional[DistributedState] = None,
    package: Optional[CompilePackage] = None,
    # pyrefly: ignore [bad-return]
) -> DynamoOutput:
    """
    A helper function taking a frame and backend, then return the generated bytecode
    and guards as a common data structure.
    This is a shared interface for multiple compiler frontends (e.g. torch.compile,
    torch.export) that needs to capture a graph out of python code.
    """
    # This is shared across restarts
    speculation_log = SpeculationLog()

    def transform(
        instructions: list[Instruction], code_options: dict[str, object]
    ) -> DynamoTracerOutput:
        tf_mode_stack: list[torch.overrides.TorchFunctionMode] = (
            torch.overrides._get_current_function_mode_stack()
        )
        tracer_output = trace_frame(
            code,
            globals,
            locals,
            builtins,
            closure,
            compiler_fn,
            tf_mode_stack,
            one_graph,
            speculation_log,
            instructions,
            code_options,
            export=export,
            export_constraints=export_constraints,
            frame_state=frame_state,
            distributed_state=distributed_state,
            package=package,
        )

        assert tracer_output is not None
        return tracer_output

    last_attempt_start_time = None
    for attempt in itertools.count():
        CompileContext.get().attempt = attempt

        try:
            with dynamo_timed(f"compile_attempt_{attempt}", log_pt2_compile_event=True):
                bytecode, tracer_output = transform_code_object(code, transform)
                assert tracer_output is not None
                return DynamoOutput(
                    tracer_output=tracer_output,
                    bytecode=bytecode,
                    last_attempt_start_time=last_attempt_start_time,
                )
        except exc.RestartAnalysis as e:
            if not isinstance(e, exc.TensorifyScalarRestartAnalysis):
                TensorifyState.clear()
            log.info(
                "Restarting analysis due to %s",
                LazyString(format_traceback_short, e.__traceback__),
            )
            # If restart reason is None just log the type of the exception
            restart_reasons.add(e.restart_reason or str(type(e)))
            # We now have a new "last attempt", reset the clock
            last_attempt_start_time = time.time()
            if attempt > 100:
                unimplemented(
                    gb_type="Excessive RestartAnalysis() calls",
                    context="",
                    explanation="Dynamo attempted to trace the same frame 100+ times. "
                    "Giving up on compiling as the compile time tradeoff is likely not "
                    "worth the performance gain.",
                    hints=[],
                )
        except exc.SkipFrame as e:
            if not isinstance(e, exc.TensorifyScalarRestartAnalysis):
                TensorifyState.clear()
            log.debug(  # noqa: G200
                "Skipping frame %s %s \
                %s %s",
                e,
                code.co_name,
                code.co_filename,
                code.co_firstlineno,
            )
            raise


def _compile(
    code: CodeType,
    globals: dict[str, object],
    locals: dict[str, object],
    builtins: dict[str, object],
    closure: tuple[CellType],
    compiler_fn: CompilerFn,
    one_graph: bool,
    export: bool,
    export_constraints: Optional[typing.Never],
    hooks: Hooks,
    cache_entry: Optional[CacheEntry],
    cache_size: CacheSizeRelevantForFrame,
    frame: Optional[DynamoFrameType] = None,
    frame_state: Optional[dict[str, Union[int, FrameStateSizeEntry]]] = None,
    *,
    compile_id: CompileId,
    skip: int = 0,
    package: Optional[CompilePackage] = None,
    # Can be used to record things for the caller, both
    # in the case of normal and exception code paths
    convert_frame_box: Optional[ConvertFrameBox] = None,
) -> ConvertFrameReturn:
    from torch.fx.experimental.validator import (
        BisectValidationException,
        ValidationException,
    )

    # Only nonlocal defs here please!
    # Time spent compiling this frame before restarting or failing analysis
    dynamo_time_before_restart: float = 0.0

    @compile_time_strobelight_meta(phase_name="compile_inner")
    def compile_inner(
        code: CodeType, one_graph: bool, hooks: Hooks
    ) -> tuple[ConvertFrameReturn, Optional[DynamoTracerOutput]]:
        with contextlib.ExitStack() as stack:
            stack.enter_context(
                torch._dynamo.callback_handler.install_callbacks(
                    CallbackTrigger.DYNAMO, str(CompileContext.current_compile_id())
                )
            )
            stack.enter_context(CompileTimeInstructionCounter.record())
            return _compile_inner(code, one_graph, hooks)

        return (
            ConvertFrameReturn(),
            None,
        )  # dead, but see https://github.com/python/mypy/issues/7577

    @maybe_cprofile
    def _compile_inner(
        code: CodeType,
        one_graph: bool,
        hooks: Hooks,
    ) -> tuple[ConvertFrameReturn, DynamoTracerOutput]:
        nonlocal dynamo_time_before_restart
        last_attempt_start_time = start_time = time.time()

        def log_bytecode(
            prefix: str, name: str, filename: str, line_no: int, code: CodeType
        ) -> None:
            if bytecode_log.isEnabledFor(logging.DEBUG):
                bytecode_log.debug(
                    format_bytecode(prefix, name, filename, line_no, code)
                )

        log_bytecode(
            "ORIGINAL BYTECODE",
            code.co_name,
            code.co_filename,
            code.co_firstlineno,
            code,
        )

        out_code = None
        try:
            dynamo_output = compile_frame(
                code,
                globals,
                locals,
                builtins,
                closure,
                compiler_fn,
                one_graph,
                restart_reasons,
                export=export,
                export_constraints=export_constraints,
                frame_state=frame_state,
                distributed_state=distributed_state,
                package=package,
            )
        except exc.SkipFrame as e:
            if one_graph:
                log.debug("No graph captured with export/fullgraph=True")
            assert e._torch_dynamo_tracer_output is not None
            return ConvertFrameReturn(), e._torch_dynamo_tracer_output

        assert distributed_state is None or distributed_state.all_states is not None, (  # type: ignore[has-type]
            "compiler collective wasn't run before compilation completed"
        )
        out_code = dynamo_output.bytecode
        tracer_output = dynamo_output.tracer_output
        if dynamo_output.last_attempt_start_time is not None:
            last_attempt_start_time = dynamo_output.last_attempt_start_time

        assert out_code is not None
        log_bytecode(
            "MODIFIED BYTECODE",
            code.co_name,
            code.co_filename,
            code.co_firstlineno,
            out_code,
        )

        for idx, hook in enumerate(_bytecode_hooks.values()):
            with dynamo_timed(f"bytecode_hooks_{idx}", log_pt2_compile_event=True):
                hook_output = hook(code, out_code)
                if hook_output is not None:
                    out_code = hook_output

        orig_code_map[out_code] = code
        output_codes.add(out_code)
        dynamo_time_before_restart = last_attempt_start_time - start_time
        assert tracer_output.output_graph is not None
        output = tracer_output.output_graph

        # Tests for new code objects.
        # The rationale for these tests can be found in torch/csrc/dynamo/eval_frame.c
        # Only test once the code object is created.
        # They are not tested during runtime.

        def count_args(code: CodeType) -> int:
            import inspect

            return (
                code.co_argcount
                + code.co_kwonlyargcount
                + bool(code.co_flags & inspect.CO_VARARGS)
                + bool(code.co_flags & inspect.CO_VARKEYWORDS)
            )

        assert out_code is not None

        total_argcount_old = count_args(code)
        total_argcount_new = count_args(out_code)
        msg = "arg mismatch: "
        msg += f"old code object has args {code.co_varnames[:total_argcount_old]}, "
        msg += f"new code object has args {out_code.co_varnames[:total_argcount_new]}"
        assert (
            code.co_varnames[:total_argcount_old]
            == out_code.co_varnames[:total_argcount_new]
        ), msg

        msg = "free var mismatch: "
        msg += f"old code object has free var {code.co_freevars}, "
        msg += f"new code object has free var {out_code.co_freevars}"
        assert code.co_freevars == out_code.co_freevars, msg

        msg = "cell var mismatch: "
        msg += f"old code object has cell var {code.co_cellvars}, "
        msg += f"new code object has cell var {out_code.co_cellvars}"
        assert code.co_cellvars == out_code.co_cellvars, msg

        # Skipping Dynamo on a frame without any extracted graph.
        # This does not affect eager functionality. But this is necessary
        # for export for cases where Dynamo-reconstructed bytecode can create
        # new function frames, confusing export in thinking that there
        # are extra graphs now.

        if output.export and output.is_empty_graph():
            return ConvertFrameReturn(), tracer_output

        assert output.guards is not None
        CleanupManager.instance[out_code] = output.cleanups
        nonlocal cache_entry
        with dynamo_timed("build_guards", log_pt2_compile_event=True):
            check_fn = dynamo_output.build_guards(
                code,
                hooks=hooks,
                save=package is not None,
                cache_entry=cache_entry,
            )

        if package is not None:
            assert check_fn.guards_state is not None
            package.add_guarded_code(check_fn.guards_state, out_code)
            package.add_inlined_source(output.tracing_context.traced_code)
            package.update_device_type(output.current_tracer.graph)

        compile_id_str = str(compile_id) if compile_id is not None else "Unknown"
        annotation_str = "Torch-Compiled Region: " + compile_id_str
        guarded_code = GuardedCode(
            out_code,
            check_fn.guard_manager,  # type: ignore[arg-type]
            compile_id,
            annotation_str,
        )

        if not output.is_empty_graph() and hooks.guard_export_fn is not None:
            # We should not run the guard_export_fn when Dynamo does not
            # generate any graph. This can happen in export when TorchDynamo
            # generated bytecode has some reconstruction logic for mutated
            # variables which can trigger TorchDynamo on the children frames but
            # they are benign and do not generate any new graphs.
            hooks.guard_export_fn(output.guards)

        return wrap_guarded_code(guarded_code), tracer_output

    metrics_context = get_metrics_context()
    code_context = (
        package.code_context(code) if package is not None else contextlib.nullcontext()
    )
    with (
        _use_lazy_graph_module(config.use_lazy_graph_module),
        compile_context(CompileContext(compile_id)),
        chromium_event_timed(
            "dynamo", reset_event_log_on_exit=True, log_pt2_compile_event=True
        ),
        _WaitCounter("pytorch.wait_counter.entire_forward_compile").guard(),
        metrics_context,
        dynamo_timed(
            "_compile.compile_inner",
            phase_name="entire_frame_compile",
            dynamo_compile_column_us="dynamo_cumulative_compile_time_us",
        ),
        code_context,
    ):
        restart_reasons: set[str] = set()
        if compile_pg := get_compile_pg():
            distributed_state = DistributedState(compile_pg, LocalState())
        else:
            distributed_state = None

        # Check recompilations
        recompile_reason: Optional[str] = None
        if is_recompilation(cache_size) and frame:
            reasons = get_and_maybe_log_recompilation_reasons(cache_entry, frame)
            recompile_reason = (
                "Unable to find recompilation reasons" if not reasons else reasons[0]
            )
        # Recheck for recompilation, for when inline_inbuilt_nn_modules is set to False
        inline_inbuilt_nn_modules_candidate = False
        if not config.inline_inbuilt_nn_modules and frame:
            inbuilt_nn_reasons = get_and_maybe_log_recompilation_reasons(
                cache_entry, frame, skip_logging=True
            )
            inbuilt_nn_recompile_reason = (
                None if not inbuilt_nn_reasons else inbuilt_nn_reasons[0]
            )

            if (
                inbuilt_nn_recompile_reason is not None
                and "[inline-inbuilt-nn-modules-candidate]"
                in inbuilt_nn_recompile_reason
            ):
                inline_inbuilt_nn_modules_candidate = True

        # Set if the recompile is a candidate for inline_inbuilt_nn_modules
        # regardless of whether inline_inbuilt_nn_modules is set or not
        metrics_context.update_outer(
            {
                "recompile_reason": recompile_reason,
                "inline_inbuilt_nn_modules_candidate": inline_inbuilt_nn_modules_candidate,
            }
        )

        recompile_user_contexts = get_hook_for_recompile_user_context()
        if recompile_user_contexts:
            # cap each user context to N chars for data retention purposes. N=256
            # is chosen to be large enough to capture the most important info.
            user_contexts_msg = {
                user_context()[:256] for user_context in recompile_user_contexts
            }
            metrics_context.set("recompile_user_contexts", user_contexts_msg)

        exceeded, limit_type = exceeds_recompile_limit(cache_size, compile_id)
        if exceeded:

            def format_func_info(code: CodeType) -> str:
                return f"'{code.co_name}' ({code.co_filename}:{code.co_firstlineno})"

            # NS: Don't add period at the end of string, as it'll be added to URL
            # rendering it incorrect
            log.warning(
                "torch._dynamo hit config.%s (%s)\n"
                "   function: %s\n"
                "   last reason: %s\n"
                'To log all recompilation reasons, use TORCH_LOGS="recompiles".\n'
                "To diagnose recompilation issues, see %s",
                limit_type,
                getattr(config, limit_type),
                format_func_info(code),
                recompile_reason,
                troubleshooting_url,
            )
            if config.fail_on_recompile_limit_hit:
                raise FailOnRecompileLimitHit(
                    f"{limit_type} reached, because fail_on_recompile_limit_hit = True this is a HARD failure"
                )
            elif one_graph:
                raise FailOnRecompileLimitHit(
                    f"{limit_type} reached with fullgraph=True. Excessive recompilations can degrade "
                    "performance due to the compilation overhead of each recompilation. To monitor "
                    "recompilations, enable TORCH_LOGS=recompiles. If recompilations are expected, consider "
                    "increasing torch._dynamo.config.cache_size_limit to an appropriate value."
                )
            elif justknobs_check(
                "pytorch/compiler:skip_code_recursive_on_recompile_limit_hit"
            ):
                raise RecompileLimitExceeded(f"{limit_type} reached")
            else:
                # do not recursively skip frames
                unimplemented(
                    gb_type="Dynamo cache limit exceeded",
                    context=f"Limit type: {limit_type}",
                    explanation="Dynamo attempted to recompile the code object too many times, "
                    f"exceeding the {limit_type} cache size limit."
                    "Giving up on compiling as the compile time tradeoff is likely not "
                    "worth the performance gain.",
                    hints=[],
                )

        log.debug(
            "torchdynamo start compiling %s %s:%s, stack (elided %s frames):\n%s",
            code.co_name,
            code.co_filename,
            code.co_firstlineno,
            skip + 2,
            # -2: omit current frame, omit contextlib decorator
            "".join(CapturedTraceback.extract(skip=2 + skip).format()),
        )
        # -4: -2 as above, plus trace_structured frames
        #
        # NB: the frame looks like this:
        #
        # # handled by skip argument
        # torch/_dynamo/convert_frame.py:1069 in catch_errors
        # torch/_dynamo/convert_frame.py:910 in _convert_frame
        # torch/_dynamo/convert_frame.py:464 in _convert_frame_assert
        # torch/_utils_internal.py:70 in wrapper_function
        #
        # # 2 current frame and context lib
        # env/lib/python3.10/contextlib.py:79 in inner
        # torch/_dynamo/convert_frame.py:776 in _compile
        #
        # # 2 extra here
        # torch/_logging/_internal.py:1064 in trace_structured
        # torch/_dynamo/convert_frame.py:780 in <lambda>
        stack_trace = log_dynamo_start(code, skip)
        start_time_ns = time.time_ns()
        fail_type: Optional[str] = None
        fail_reason: Optional[str] = None
        exception_stack_trace: Optional[list[str]] = None
        fail_user_frame_filename: Optional[str] = None
        fail_user_frame_lineno: Optional[int] = None
        torch._dynamo.utils.ReinplaceCounters.clear()
        guarded_code = None
        tracer_output = None
        try:
            guarded_code, tracer_output = compile_inner(code, one_graph, hooks)

            # NB: We only put_code_state in success case.  Success case here
            # does include graph breaks; specifically, if a graph break still
            # resulted in a partially compiled graph, we WILL return here.  An
            # Unsupported exception will only bubble to the top level if we
            # are unable to compile the frame at all.  In this case, there's
            # no point in uploading the code state, because we will always
            # fail exactly the same way even without the update.  (It's useful
            # to upload for graph break though, because this can prevent
            # extra graph break compilations.)
            put_code_state()
            if (
                tracer_output
                and (output_graph := tracer_output.output_graph)
                and output_graph.has_outputs()
            ):
                log_frame_dynamic_whitelist(code)
                if recompile_reason and "size mismatch at index" in recompile_reason:
                    _log_size_mismatch_recompile()

            return guarded_code
        except Exception as e:
            # NB: e's msg is mutated here to add user stack, but we DON'T want
            # that stack in the Scuba logged fail_reason. So we grab the fail
            # info here and add it to the metrics context below.
            fail_type = type(e).__qualname__
            fail_reason = str(e)
            exception_stack_trace = [traceback.format_exc()]
            exception_handler(e, code, frame, export=export)
            # NB: this is the post-mutation exception
            torch._logging.trace_structured(
                "artifact",
                metadata_fn=lambda: {
                    "name": "dynamo_error",
                    "encoding": "string",
                },
                payload_fn=lambda: traceback.format_exc(),
            )
            fail_user_frame_filename, fail_user_frame_lineno = exc.get_exc_message(
                e, compile_id
            )
            tracer_output = getattr(e, "_torch_dynamo_tracer_output", None)
            if isinstance(
                e,
                (
                    Unsupported,
                    TorchRuntimeError,
                    BackendCompilerFailed,
                    AssertionError,
                    ConstraintViolationError,
                    GuardOnDataDependentSymNode,
                    ValidationException,
                    UncapturedHigherOrderOpError,
                    BisectValidationException,
                    ShortenTraceback,
                    PackageError,
                    ResumePrologueTracingError,
                ),
            ):
                raise
            else:
                # Rewrap for clarity
                raise InternalTorchDynamoError(
                    f"{type(e).__qualname__}: {str(e)}"
                ).with_traceback(e.__traceback__) from None
        finally:
            # === WARNING WARNING WARNING ===
            # If you commit a bug here, it will suppress writing to
            # dynamo_compile table, and we will not have telemetry.
            # Be extra careful when making changes here!

            if torch._dynamo.config.run_gc_after_compile:
                with dynamo_timed("gc", dynamo_compile_column_us="gc_time_us"):
                    log.info("run_gc_after_compile: running gc")
                    gc.collect(1)

            output = None
            if tracer_output:
                output = tracer_output.output_graph
            if output:
                output.local_scope = {}
                # tracer should already be None, keep an extra check here just in case.
                if tracer := output.root_tx:
                    tracer.f_locals = {}

            from .utils import curr_frame

            frame_key = str(curr_frame)
            if fail_reason is None and output is not None:
                guard_count = len(output.guards)
                shape_env_guard_count = len(output.shape_env.guards)
                graph_op_count = output.count_calls()
                graph_node_count = len(output.graph.nodes)
                graph_node_shapes = output.get_graph_sizes_structured()
                graph_input_count = len(output.placeholders)
                non_compliant_ops = {op.__qualname__ for op in output.non_compliant_ops}
                compliant_custom_ops = {
                    op.__qualname__ for op in output.compliant_custom_ops
                }
                torch._dynamo.utils.ReinplaceCounters.log()
            else:
                guard_count = None
                shape_env_guard_count = None
                graph_op_count = None
                graph_node_count = None
                graph_node_shapes = {}
                graph_input_count = None
                non_compliant_ops = set({})
                compliant_custom_ops = set({})
                restart_reasons = set()
                # If compilation failed, the entire time is wasted
                dynamo_time_before_restart = (time.time_ns() - start_time_ns) / 1e9

            metrics = {
                "frame_key": frame_key,
                "co_name": code.co_name,
                "co_filename": code.co_filename,
                "co_firstlineno": code.co_firstlineno,
                "cache_size": cache_size.num_cache_entries_with_same_id_matched_objs,
                "accumulated_cache_size": cache_size.num_cache_entries,
                "guard_count": guard_count,
                "shape_env_guard_count": shape_env_guard_count,
                "graph_op_count": graph_op_count,
                "graph_node_count": graph_node_count,
                "graph_input_count": graph_input_count,
                "fail_type": fail_type,
                "fail_reason": fail_reason,
                "fail_user_frame_filename": fail_user_frame_filename,
                "fail_user_frame_lineno": fail_user_frame_lineno,
                "non_compliant_ops": non_compliant_ops,
                "compliant_custom_ops": compliant_custom_ops,
                "restart_reasons": restart_reasons,
                "dynamo_time_before_restart_s": dynamo_time_before_restart,
                "has_guarded_code": guarded_code is not None,
                "specialize_float": config.specialize_float,
                "is_forward": True,
                "dynamo_compile_time_before_restart_us": to_int_us(
                    dynamo_time_before_restart
                ),
                "stack_trace": stack_trace,
                "graph_node_shapes": str(graph_node_shapes),
                "exception_stack_trace": exception_stack_trace,
            }
            # TODO: replace with CompileEventLogger.compilation_metrics
            # There are some columns here not in PT2 Compile Events
            # so we need to slightly change it
            metrics_context.update_outer(metrics)
            # === END WARNING WARNING WARNING ===

            # If tracer is available, then tracer.error_on_graph_break reflects value of
            # global symbolic_convert.error_on_graph_break at the time of the graph break -
            # symbolic_convert.error_on_graph_break may have been (correctly) changed during cleanup.
            # If tracer is unavailable, then fallback to symbolic_convert.error_on_graph_break.
            if convert_frame_box:
                convert_frame_box.error_on_graph_break = (
                    tracer_output.error_on_graph_break
                    if tracer_output
                    else _get_error_on_graph_break()
                )


class ConvertFrame:
    def __init__(
        self,
        compiler_fn: CompilerFn,
        hooks: Hooks,
        package: Optional[CompilePackage] = None,
    ) -> None:
        self._torchdynamo_orig_backend = compiler_fn
        self._inner_convert = convert_frame_assert(
            compiler_fn, one_graph=False, package=package
        )
        self._hooks = hooks

    @property
    def _clone_with_backend(self) -> Callable[[WrapBackendDebug], ConvertFrame]:
        return lambda backend: convert_frame(
            backend,
            self._hooks,
        )

    def __call__(
        self,
        frame: DynamoFrameType,
        cache_entry: Optional[CacheEntry],
        hooks: Hooks,
        frame_state: dict[str, Union[int, FrameStateSizeEntry]],
        skip: int = 0,
    ) -> ConvertFrameReturn:
        input_codes.add(frame.f_code)
        counters["frames"]["total"] += 1
        try:
            result = self._inner_convert(
                frame, cache_entry, hooks, frame_state, skip=skip + 1
            )
            counters["frames"]["ok"] += 1
            return result
        except Exception as e:
            # Do not allow errors to be suppressed if we're tracing a resume function prologue
            if isinstance(e, ResumePrologueTracingError):
                raise

            error_on_graph_break = (
                self._inner_convert._box.error_on_graph_break is not None
            )
            assert error_on_graph_break is not None
            if self._inner_convert._box.error_on_graph_break:
                # NOTE we _might_ have to wrap the current in a custom exception
                # in order to correctly bubble up to the top-level compile wrapper in
                # eval_frame.py. But re-raising seems to work for now because exceptions from tracing
                # a nested call that results in a top-level frame compile will be handled by the caller
                # as an observed exception - we don't expect that exception to be suppressed.
                raise

            # These two exception types are "soft" failure, in the sense that
            # we know this is due to something we didn't implement all the
            # way, scare the user less about it.  That being said, if you
            # are trying to understand why a graph break happened, it's still
            # important to have this information, so offer it.
            #
            # NB: NotImplementedError used to be on this list, but actually
            # it is impossible for it to reach here, as it is converted into
            # InternalTorchDynamoError.  This behavior seemed reasonable
            # to me (ezyang, Aug 2023) so I kept it, but maybe at some point
            # someone wanted these to also get suppressed.  If so, you'll
            # need to make these exceptions not get wrapped

            # We intentionally don't want to suppress error here.
            if isinstance(e, UncapturedHigherOrderOpError):
                raise

            soft_fail = isinstance(e, Unsupported)

            # This is a soft failure. In the sense, the code path reaches here
            # when we do not support graph breaks on bytecodes like LOAD_ATTR,
            # BUILD_SET etc. In such case, we can fallback to eager without
            # scaring users.
            if soft_fail and graph_break_log.isEnabledFor(logging.DEBUG):
                # Log this message in the graph break. Also use the string
                # "skip: " to tell that the whole frame is falling back to
                # eager.
                if hasattr(e, "compile_id") and hasattr(e, "real_stack"):
                    with compile_context(CompileContext(e.compile_id)):  # type: ignore[attr-defined]
                        user_stack = e.real_stack
                        user_stack_formatted = "".join(
                            traceback.format_list(user_stack)
                        )
                        user_stack_trace = f"Graph break: skip: from user code at:\n{user_stack_formatted}"
                        torch._logging.trace_structured(
                            "artifact",
                            metadata_fn=lambda: {
                                "name": "dynamo_graph_break_reason",
                                "encoding": "string",
                            },
                            payload_fn=lambda: f"{user_stack_trace}\n{traceback.format_exc()}",
                        )
                        graph_break_log.debug(
                            user_stack_trace,
                            exc_info=True,
                        )

            if not config.suppress_errors and not soft_fail:
                raise

            # Suppress the error.  NB: It's very important to do the
            # suppression logging HERE, where the actual suppression
            # happens. Previously it was somewhere else and so it was
            # possible to accidentally not log at all.
            record_filename = getattr(e, "record_filename", None)
            code = frame.f_code
            error_msg = format_error_msg(e, code, record_filename, frame)

            if soft_fail:
                log.info(error_msg, exc_info=True)
            else:
                log.warning(error_msg, exc_info=True)

            if isinstance(e, SkipCodeRecursiveException):
                return ConvertFrameReturn(
                    frame_exec_strategy=FrameExecStrategy(
                        FrameAction.SKIP, FrameAction.SKIP
                    )
                )
            elif isinstance(e, RecompileLimitExceeded):
                return ConvertFrameReturn(
                    frame_exec_strategy=FrameExecStrategy(
                        FrameAction.RUN_ONLY, FrameAction.RUN_ONLY
                    )
                )

        return ConvertFrameReturn()


def convert_frame(
    compiler_fn: CompilerFn,
    hooks: Hooks,
    package: Optional[CompilePackage] = None,
) -> ConvertFrame:
    """Try to convert a frame into an FX graph, if error leave frame unmodified"""
    return ConvertFrame(compiler_fn, hooks, package=package)


# TODO mlazos: add support for same args, or record them
def replay(filename: str) -> None:
    from .backends.debugging import eager

    original_replay_val = config.replay_record_enabled
    config.replay_record_enabled = False
    with open(filename, "rb") as in_file:
        record = ExecutionRecord.load(in_file)
    record.globals = dict(itertools.chain(record.globals.items(), globals().items()))

    with decorators.error_on_graph_break(False):
        try:
            _compile(
                record.code,
                record.globals,
                record.locals,
                record.builtins,
                record.closure,
                compiler_fn=eager,
                one_graph=False,
                export=False,
                export_constraints=None,
                hooks=Hooks(),
                cache_size=CacheSizeRelevantForFrame(0, 0),
                cache_entry=None,
                frame=None,
                frame_state={},
                compile_id=CompileId(frame_id=42, frame_compile_id=999),
            )
        finally:
            config.replay_record_enabled = original_replay_val


def first_real_inst_idx(code: CodeType) -> int:
    if sys.version_info < (3, 11):
        return 0
    for inst in dis.get_instructions(code):
        if inst.opname == "RESUME":
            return inst.offset // 2
    raise RuntimeError("RESUME instruction not found in code")


class ConvertFrameProtocol(typing.Protocol):
    def __call__(
        self,
        frame: DynamoFrameType,
        cache_entry: Optional[CacheEntry],
        hooks: Hooks,
        frame_state: dict[str, Union[int, FrameStateSizeEntry]],
        *,
        skip: int = 0,
    ) -> ConvertFrameReturn: ...


def should_skip_due_to_torch_dispatch_mode() -> bool:
    return is_in_any_mode_without_ignore_compile_internals()


class CatchErrorsWrapper:
    def __init__(self, callback: ConvertFrameProtocol, hooks: Hooks) -> None:
        functools.wraps(callback)(self)
        self._torchdynamo_orig_backend = callback
        self.hooks = hooks

    def __call__(
        self,
        frame: DynamoFrameType,
        cache_entry: Optional[CacheEntry],
        frame_state: dict[str, Union[int, FrameStateSizeEntry]],
    ) -> ConvertFrameReturn:
        assert frame_state is not None
        input_codes.add(frame.f_code)

        is_skipfile = trace_rules.check(frame.f_code)
        if sys.version_info >= (3, 13):
            has_started_execution = frame.f_lasti > first_real_inst_idx(frame.f_code)
        else:
            has_started_execution = frame.f_lasti >= first_real_inst_idx(frame.f_code)
        if (
            # TODO: the first condition is not covered by any test
            has_started_execution
            or is_skipfile
            or config.disable
            or (
                should_skip_due_to_torch_dispatch_mode()
                and not getattr(self._torchdynamo_orig_backend, "_export", False)
            )
        ):
            if log.isEnabledFor(logging.DEBUG):
                if has_started_execution:
                    skip_reason = "traced frame already"
                elif trace_rules.check(frame.f_code):
                    skip_reason = "in skipfiles"
                elif is_in_torch_dispatch_mode(include_infra_modes=False):
                    skip_reason = "non-infra torch dispatch mode present, this is not supported today in torch.compile"
                else:
                    skip_reason = "dynamo tracing is disabled"

                log.debug(
                    "skipping: %s (reason: %s, file: %s)",
                    frame.f_code.co_name,
                    skip_reason,
                    frame.f_code.co_filename,
                )
            return ConvertFrameReturn()

        if (
            frame.f_code.co_filename == "<string>" and frame.f_code.co_name == "__new__"
        ) or (
            frame.f_code.co_filename.endswith("collections/__init__.py")
            and frame.f_code.co_name == "_make"
        ):
            # nametuple constructor/_make
            return ConvertFrameReturn()
        if torch._dynamo.utils.get_optimize_ddp_mode() == "ddp_optimizer":
            ddp_module = DistributedDataParallel._get_active_ddp_module()
            if ddp_module:
                with compile_lock:
                    from torch._dynamo.backends.distributed import DDPOptimizer

                    ddp_optimizer = DDPOptimizer(
                        bucket_bytes_cap=ddp_module.bucket_bytes_cap,
                        backend_compile_fn=self._torchdynamo_orig_backend._torchdynamo_orig_backend,  # type: ignore[attr-defined]
                    )
                    assert hasattr(
                        self._torchdynamo_orig_backend, "_clone_with_backend"
                    ), (
                        "DDPOptimizer only supports callback fns that know how to clone themselves."
                    )
                    hijacked_callback = (
                        self._torchdynamo_orig_backend._clone_with_backend(
                            ddp_optimizer.compile_fn,
                        )
                    )
                    return hijacked_callback(
                        frame, cache_entry, self.hooks, frame_state
                    )

        with compile_lock, _disable_current_modes():
            # skip=1: skip this frame
            result = self._torchdynamo_orig_backend(
                frame, cache_entry, self.hooks, frame_state, skip=1
            )
            return result


def catch_errors_wrapper(
    callback: ConvertFrameProtocol, hooks: Hooks
) -> CatchErrorsWrapper:
    return CatchErrorsWrapper(callback, hooks)
