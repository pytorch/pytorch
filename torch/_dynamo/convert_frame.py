# mypy: allow-untyped-decorators
from __future__ import annotations

import collections
import contextlib
import cProfile
import dis
import functools
import gc
import itertools
import logging
import os
import pstats
import subprocess
import sys
import threading
import time
import traceback
import typing
import weakref
from pathlib import Path
from types import CellType, CodeType, FunctionType, ModuleType
from typing import Any, Callable, Optional, TypeVar, Union
from typing_extensions import ParamSpec
from weakref import ReferenceType

import torch
import torch._logging
from torch._C._dynamo.guards import GlobalStateGuard
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
    is_in_torch_dispatch_mode,
)
from torch.utils._traceback import CapturedTraceback, format_traceback_short

from . import config, exc, trace_rules
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
    RecompileLimitExceeded,
    ShortenTraceback,
    SkipCodeRecursiveException,
    TorchRuntimeError,
    UncapturedHigherOrderOpError,
    unimplemented,
    Unsupported,
)
from .guards import (
    CheckFunctionManager,
    get_and_maybe_log_recompilation_reasons,
    GuardedCode,
)
from .hooks import Hooks
from .pgo import put_code_state
from .replay_record import ExecutionRecord
from .resume_execution import TORCH_DYNAMO_RESUME_IN_PREFIX
from .symbolic_convert import (
    DistributedState,
    InstructionTranslator,
    LocalState,
    SpeculationLog,
)
from .trace_rules import is_numpy
from .utils import (
    chromium_event_timed,
    CleanupManager,
    CompileTimeInstructionCounter,
    counters,
    dynamo_timed,
    format_bytecode,
    gen_record_file_name,
    get_metrics_context,
    increment_frame,
    is_namedtuple,
    istype,
    LazyString,
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
    from .backends.registry import CompilerFn
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


# TODO it is possible to move more global state preservation to eval_frame.py/c.
# See how we preserve Python random state.
def preserve_global_state(fn: Callable[_P, _T]) -> Callable[_P, _T]:
    """
    Context manager to:
        1) Save/restore torch.is_grad_enabled() state
        2) Save/restore torch random state
        3) Monkey patch torch.fx.graph_module._forward_from_src
    """

    @functools.wraps(fn)
    def _fn(*args: _P.args, **kwargs: _P.kwargs) -> _T:
        guards = GlobalStateGuard()
        prior_grad_mode = torch.is_grad_enabled()
        # Just in case we get left in a bad dispatch state we want to restore
        # it. This can happen because the dispatch bits aren't a true
        # stack/counter - so we can't just increment/decrement them as we enter
        # and leave.
        with torch._C._PreserveDispatchKeyGuard():
            prior_inference_mode = torch.is_inference_mode_enabled()
            prior_deterministic = torch.are_deterministic_algorithms_enabled()
            prior_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
            prior_mobile_allocator_state = (
                torch._C._is_default_mobile_cpu_allocator_set()
            )
            prior_dtype = torch.get_default_dtype()
            torch_rng_state = torch.random.get_rng_state()
            cuda_rng_state = None
            if torch.cuda.is_available():
                cuda_rng_state = torch.cuda.get_rng_state()
            allow_tf32 = torch._C._get_cublas_allow_tf32()
            prior_fwd_from_src = torch.fx.graph_module._forward_from_src
            torch.fx.graph_module._forward_from_src = fx_forward_from_src_skip_result
            cleanup = setup_compile_debug()
            exit_stack = contextlib.ExitStack()
            exit_stack.enter_context(
                torch.fx._symbolic_trace._maybe_revert_all_patches()
            )
            exit_stack.enter_context(torch_function_mode_stack_state_mgr)
            try:
                return fn(*args, **kwargs)
            finally:
                cleanup.close()
                assert (
                    torch._C._len_torch_function_stack() == 0
                ), "Torch function mode stack state changed while dynamo tracing, please report a bug"
                exit_stack.close()
                torch._C._set_grad_enabled(prior_grad_mode)
                torch.autograd.grad_mode._enter_inference_mode(prior_inference_mode)
                torch.use_deterministic_algorithms(
                    prior_deterministic, warn_only=prior_warn_only
                )
                torch.random.set_rng_state(torch_rng_state)
                torch.set_default_dtype(prior_dtype)
                curr_mobile_allocator_state = (
                    torch._C._is_default_mobile_cpu_allocator_set()
                )
                if prior_mobile_allocator_state != curr_mobile_allocator_state:
                    torch._C._unset_default_mobile_cpu_allocator()
                if cuda_rng_state is not None:
                    torch.cuda.set_rng_state(cuda_rng_state)
                torch._C._set_cublas_allow_tf32(allow_tf32)
                torch.fx.graph_module._forward_from_src = prior_fwd_from_src
                assert (
                    guards.check()
                ), f"Global {guards.reason()}state changed while dynamo tracing, please report a bug"

    _fn._torchdynamo_orig_callable = fn  # type: ignore[attr-defined]
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
FRAME_COMPILE_COUNTER: typing.Counter[
    Union[int, FrameStateSizeEntry]
] = collections.Counter()


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
        prof.enable()
        start_ts = time.time()
        retval = prof.runcall(func, *args, **kwargs)
        profile_latency = time.time() - start_ts
        prof.disable()
        log.warning(
            "### Cprofile for %s trace id [%s] took %.3f seconds ###",
            func.__name__,
            trace_id,
            profile_latency,
        )
        ps = pstats.Stats(prof)
        try:
            prof.dump_stats(profile_path)
        except PermissionError:
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


class ConvertFrameAssert:
    def __init__(
        self,
        compiler_fn: CompilerFn,
        one_graph: bool = True,
        export: bool = False,
        export_constraints: Optional[typing.Never] = None,
    ) -> None:
        # assert export_constraints is None
        reset_graph_break_dup_checker()
        self._torchdynamo_orig_callable = compiler_fn
        self._one_graph = one_graph
        self._export = export
        self._export_constraints = export_constraints

    @property
    def _clone_with_backend(self) -> Callable[[CompilerFn], ConvertFrameAssert]:
        return lambda backend: convert_frame_assert(
            backend, self._one_graph, self._export, self._export_constraints
        )

    def __call__(
        self,
        frame: DynamoFrameType,
        cache_entry: Optional[CacheEntry],
        hooks: Hooks,
        frame_state: dict[str, Union[int, FrameStateSizeEntry]],
        *,
        skip: int = 0,
    ) -> Optional[GuardedCode]:
        increment_frame()

        code = frame.f_code

        cache_size = compute_cache_size(frame, cache_entry)
        input_codes.add(code)
        if code in output_codes:
            return None
        if (
            os.environ.get("TORCHDYNAMO_DEBUG_FUNCTION")
            and os.environ.get("TORCHDYNAMO_DEBUG_FUNCTION") != code.co_name
        ):
            return None
        if code.co_name == "<genexpr>" and code.co_filename.endswith(
            (
                "transformers/file_utils.py",
                "transformers/utils/generic.py",
                "diffusers/utils/outputs.py",
            )
        ):
            # not needed, but cleans up torchbench error stats
            return None
        if code.co_name == "__setattr__":
            # setattr could be tricky to handle generally,
            # but also not likely useful to compile- skip the whole frame
            return None
        if code.co_name == "__init__" and code.co_filename.startswith(
            os.path.dirname(torch.optim.__file__)
        ):
            # optimizer support is still incomplete see
            # test_state_dict in test/dynamo/test_optimizers.py
            return None

        # Check if the frame is generated by an exec builtin call
        # TODO - Running exec generated frame seems propagates f_globals to the
        # next frames.
        if code.co_name == "<module>" and code.co_filename == "<string>":
            return None

        if (
            code.co_name == "<lambda>"
            and code.co_filename == "<string>"
            and not bool(frame.f_builtins)
        ):
            # namedtuple subclass constructor. Empty builtins cause issue with
            # len keyword in LIST_LEN guard.
            return None

        if is_generator(code):
            unimplemented("generator")

        if not has_tensor_in_frame(frame):
            return None

        global initial_global_state
        initial_global_state = GlobalStateGuard()

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
        compile_id = CompileId(
            compiled_autograd_id=compiled_autograd_id,
            frame_id=frame_id,
            frame_compile_id=frame_compile_id,
        )

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
            return _compile(
                frame.f_code,
                frame.f_globals,
                frame.f_locals,
                frame.f_builtins,
                frame.closure,
                self._torchdynamo_orig_callable,
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
            )


def convert_frame_assert(
    compiler_fn: CompilerFn,
    one_graph: bool = True,
    export: bool = False,
    export_constraints: Optional[typing.Never] = None,
) -> ConvertFrameAssert:
    """Fully convert a frame into an FX graph"""
    return ConvertFrameAssert(compiler_fn, one_graph, export, export_constraints)


from collections import OrderedDict

from torch.utils.hooks import RemovableHandle


if typing.TYPE_CHECKING:
    from .output_graph import OutputGraph

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
) -> Optional[GuardedCode]:
    from torch.fx.experimental.validator import (
        bisect,
        BisectValidationException,
        translation_validation_enabled,
        ValidationException,
    )

    # Only nonlocal defs here please!
    # Time spent compiling this frame before restarting or failing analysis
    dynamo_time_before_restart: float = 0.0
    output: Optional[OutputGraph] = None
    tracer: Optional[InstructionTranslator] = None

    tf_mode_stack: list[
        torch.overrides.TorchFunctionMode
    ] = torch.overrides._get_current_function_mode_stack()

    @preserve_global_state
    def transform(
        instructions: list[Instruction], code_options: dict[str, object]
    ) -> None:
        nonlocal output
        nonlocal tracer
        speculation_log.restart()
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
            speculation_log=speculation_log,
            distributed_state=distributed_state,
        )

        try:
            with tracing(tracer.output.tracing_context), tracer.set_current_tx():
                tracer.run()
        except exc.UnspecializeRestartAnalysis:
            speculation_log.clear()
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

        output = tracer.output
        assert output is not None
        assert output.output_instructions
        instructions[:] = output.output_instructions
        code_options.update(output.code_options)
        propagate_inst_exn_table_entries(instructions)
        check_inst_exn_tab_entries_valid(instructions)
        instructions[:] = remove_pointless_jumps(remove_dead_code(instructions))

    @compile_time_strobelight_meta(phase_name="compile_inner")
    def compile_inner(
        code: CodeType,
        one_graph: bool,
        hooks: Hooks,
        transform: Callable[[list[Instruction], dict[str, Any]], Any],
    ) -> Optional[GuardedCode]:
        with contextlib.ExitStack() as stack:
            stack.enter_context(
                dynamo_timed(
                    "_compile.compile_inner",
                    phase_name="entire_frame_compile",
                    dynamo_compile_column_us="dynamo_cumulative_compile_time_us",
                )
            )
            stack.enter_context(
                _WaitCounter("pytorch.wait_counter.dynamo_compile").guard()
            )
            stack.enter_context(torch._dynamo.callback_handler.install_callbacks())
            stack.enter_context(CompileTimeInstructionCounter.record())
            return _compile_inner(code, one_graph, hooks, transform)

        return None  # dead, but see https://github.com/python/mypy/issues/7577

    @maybe_cprofile
    def _compile_inner(
        code: CodeType,
        one_graph: bool,
        hooks: Hooks,
        transform: Callable[[list[Instruction], dict[str, Any]], Any],
    ) -> Optional[GuardedCode]:
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
        for attempt in itertools.count():
            CompileContext.get().attempt = attempt
            try:
                out_code = transform_code_object(code, transform)
                break
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
                    unimplemented("100+ RestartAnalysis() calls")
            except exc.SkipFrame as e:
                if not isinstance(e, exc.TensorifyScalarRestartAnalysis):
                    TensorifyState.clear()
                log.debug(
                    "Skipping frame %s %s \
                    %s %s",
                    e,
                    code.co_name,
                    code.co_filename,
                    code.co_firstlineno,
                )
                if one_graph:
                    log.debug("No graph captured with one_graph=True")
                return None

        assert (
            distributed_state is None or distributed_state.all_states is not None
        ), "compiler collective wasn't run before compilation completed"

        assert out_code is not None
        log_bytecode(
            "MODIFIED BYTECODE",
            code.co_name,
            code.co_filename,
            code.co_firstlineno,
            out_code,
        )

        for hook in _bytecode_hooks.values():
            hook_output = hook(code, out_code)
            if hook_output is not None:
                out_code = hook_output

        orig_code_map[out_code] = code
        output_codes.add(out_code)
        dynamo_time_before_restart = last_attempt_start_time - start_time
        assert output is not None

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
            return None

        assert output.guards is not None
        CleanupManager.instance[out_code] = output.cleanups
        nonlocal cache_entry
        check_fn = CheckFunctionManager(
            code,
            output,
            cache_entry,
            hooks.guard_fail_fn if hooks else None,
        )

        compile_id_str = str(compile_id) if compile_id is not None else "Unknown"
        annotation_str = "Torch-Compiled Region: " + compile_id_str
        guarded_code = GuardedCode(
            out_code, check_fn.guard_manager, compile_id, annotation_str  # type: ignore[arg-type]
        )

        if not output.is_empty_graph() and hooks.guard_export_fn is not None:
            # We should not run the guard_export_fn when Dynamo does not
            # generate any graph. This can happen in export when TorchDynamo
            # generated bytecode has some reconstruction logic for mutated
            # variables which can trigger TorchDynamo on the children frames but
            # they are benign and do not generate any new graphs.
            hooks.guard_export_fn(output.guards)

        return guarded_code

    metrics_context = get_metrics_context()
    with _use_lazy_graph_module(config.use_lazy_graph_module), compile_context(
        CompileContext(compile_id)
    ), chromium_event_timed(
        "dynamo", reset_event_log_on_exit=True, log_pt2_compile_event=True
    ), metrics_context:
        restart_reasons: set[str] = set()
        # This is shared across restarts
        speculation_log = SpeculationLog()
        if compile_pg := get_compile_pg():
            distributed_state = DistributedState(compile_pg, LocalState())
        else:
            distributed_state = None

        # Check recompilations
        recompile_reason: Optional[str] = None
        if is_recompilation(cache_size) and frame:
            reasons = get_and_maybe_log_recompilation_reasons(cache_entry, frame)
            recompile_reason = (
                "Unable to find recompilation reasons" if not reasons else reasons[-1]
            )
        metrics_context.update_outer({"recompile_reason": recompile_reason})

        exceeded, limit_type = exceeds_recompile_limit(cache_size, compile_id)
        if exceeded:

            def format_func_info(code: CodeType) -> str:
                return f"'{code.co_name}' ({code.co_filename}:{code.co_firstlineno})"

            log.warning(
                "torch._dynamo hit config.%s (%s)\n"
                "   function: %s\n"
                "   last reason: %s\n"
                'To log all recompilation reasons, use TORCH_LOGS="recompiles".\n'
                "To diagnose recompilation issues, see %s.",
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
                    f"{limit_type} reached with one_graph=True. Excessive recompilations can degrade "
                    "performance due to the compilation overhead of each recompilation. To monitor "
                    "recompilations, enable TORCH_LOGS=recompiles. If recompilations are expected, consider "
                    "increasing torch._dynamo.config.cache_size_limit to an appropriate value."
                )
            elif config.skip_code_recursive_on_recompile_limit_hit and justknobs_check(
                "pytorch/compiler:skip_code_recursive_on_recompile_limit_hit"
            ):
                raise RecompileLimitExceeded(f"{limit_type} reached")
            else:
                # do not recursively skip frames
                unimplemented(f"{limit_type} reached")

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
        convert_frame_intern = structured.intern_string(__file__)
        # Initialize the ChromiumEventLogger on start
        torch._logging.trace_structured(
            "dynamo_start",
            lambda: {
                "stack": list(
                    itertools.takewhile(
                        lambda f: f["filename"] != convert_frame_intern,
                        structured.from_traceback(
                            CapturedTraceback.extract(skip=4 + skip).summary()
                        ),
                    )
                )
                + [
                    {
                        "line": code.co_firstlineno,
                        "name": code.co_name,
                        "filename": structured.intern_string(code.co_filename),
                    }
                ]
            },
        )
        start_time_ns = time.time_ns()
        fail_type: Optional[str] = None
        fail_reason: Optional[str] = None
        fail_user_frame_filename: Optional[str] = None
        fail_user_frame_lineno: Optional[int] = None
        torch._dynamo.utils.ReinplaceCounters.clear()
        guarded_code = None
        try:
            guarded_code = compile_inner(code, one_graph, hooks, transform)

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

            return guarded_code
        except Exception as e:
            # NB: e's msg is mutated here to add user stack, but we DON'T want
            # that stack in the Scuba logged fail_reason. So we grab the fail
            # info here and add it to the metrics context below.
            fail_type = type(e).__qualname__
            fail_reason = str(e)
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

            if tracer:
                tracer.output.local_scope = {}

            from .utils import curr_frame

            frame_key = str(curr_frame)
            if fail_reason is None and output is not None:
                guard_count = len(output.guards)
                shape_env_guard_count = len(output.shape_env.guards)
                graph_op_count = output.count_calls()
                graph_node_count = len(output.graph.nodes)
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
                "config_suppress_errors": config.suppress_errors,
                "config_inline_inbuilt_nn_modules": config.inline_inbuilt_nn_modules,
                "specialize_float": config.specialize_float,
                "is_forward": True,
                "dynamo_compile_time_before_restart_us": to_int_us(
                    dynamo_time_before_restart
                ),
            }
            # TODO: replace with CompileEventLogger.compilation_metrics
            # There are some columns here not in PT2 Compile Events
            # so we need to slightly change it
            metrics_context.update_outer(metrics)
            # === END WARNING WARNING WARNING ===


class ConvertFrame:
    def __init__(self, compiler_fn: CompilerFn, hooks: Hooks) -> None:
        self._torchdynamo_orig_callable = compiler_fn
        self._inner_convert = convert_frame_assert(compiler_fn, one_graph=False)
        self._hooks = hooks

    @property
    def _clone_with_backend(self) -> Callable[[WrapBackendDebug], ConvertFrame]:
        return lambda backend: convert_frame(backend, self._hooks)

    def __call__(
        self,
        frame: DynamoFrameType,
        cache_entry: Optional[CacheEntry],
        hooks: Hooks,
        frame_state: dict[str, Union[int, FrameStateSizeEntry]],
        skip: int = 0,
    ) -> Optional[
        Union[
            GuardedCode,
            torch._C._dynamo.eval_frame.SkipCodeRecursiveFlag,
            torch._C._dynamo.eval_frame.CacheLimitHitFlag,
        ]
    ]:
        counters["frames"]["total"] += 1
        try:
            result = self._inner_convert(
                frame, cache_entry, hooks, frame_state, skip=skip + 1
            )
            counters["frames"]["ok"] += 1
            return result
        except Exception as e:
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
            if isinstance(e, Unsupported) and graph_break_log.isEnabledFor(
                logging.DEBUG
            ):
                # Log this message in the graph break. Also use the string
                # "skip: " to tell that the whole frame is falling back to
                # eager.
                if hasattr(e, "compile_id"):
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

            # If we encounter SkipCodeRecursiveException, return skip_code_recursive_flag
            # to signal to Dynamo eval frame to skip the current frame and any recursive calls.
            if isinstance(e, SkipCodeRecursiveException):
                return torch._C._dynamo.eval_frame.skip_code_recursive_flag
            elif isinstance(e, RecompileLimitExceeded):
                # signal to Dynamo to run this frame on run-only mode, skipping recursively if
                # no valid cache entry is found.
                return torch._C._dynamo.eval_frame.cache_limit_hit_flag

        return None


def convert_frame(compiler_fn: CompilerFn, hooks: Hooks) -> ConvertFrame:
    """Try to convert a frame into an FX graph, if error leave frame unmodified"""
    return ConvertFrame(compiler_fn, hooks)


# TODO mlazos: add support for same args, or record them
def replay(filename: str) -> None:
    from .backends.debugging import eager

    original_replay_val = config.replay_record_enabled
    config.replay_record_enabled = False
    with open(filename, "rb") as in_file:
        record = ExecutionRecord.load(in_file)
    record.globals = dict(itertools.chain(record.globals.items(), globals().items()))

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
    ) -> Optional[GuardedCode]:
        ...


class CatchErrorsWrapper:
    def __init__(self, callback: ConvertFrameProtocol, hooks: Hooks) -> None:
        functools.wraps(callback)(self)
        self._torchdynamo_orig_callable = callback
        self.hooks = hooks

    def __call__(
        self,
        frame: DynamoFrameType,
        cache_entry: Optional[CacheEntry],
        frame_state: dict[str, Union[int, FrameStateSizeEntry]],
    ) -> Optional[GuardedCode]:
        assert frame_state is not None

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
                is_in_torch_dispatch_mode(include_infra_modes=False)
                and not getattr(self._torchdynamo_orig_callable, "_export", False)
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
            return None

        if frame.f_code.co_filename == "<string>" and frame.f_code.co_name == "__new__":
            # nametuple constructor
            return None
        if config._get_optimize_ddp_mode() == "ddp_optimizer":
            ddp_module = DistributedDataParallel._get_active_ddp_module()
            if ddp_module:
                with compile_lock:
                    from torch._dynamo.backends.distributed import DDPOptimizer

                    ddp_optimizer = DDPOptimizer(
                        bucket_bytes_cap=ddp_module.bucket_bytes_cap,
                        backend_compile_fn=self._torchdynamo_orig_callable._torchdynamo_orig_callable,  # type: ignore[attr-defined]
                    )
                    assert hasattr(
                        self._torchdynamo_orig_callable, "_clone_with_backend"
                    ), "DDPOptimizer only supports callback fns that know how to clone themselves."
                    hijacked_callback = (
                        self._torchdynamo_orig_callable._clone_with_backend(
                            ddp_optimizer.compile_fn,
                        )
                    )
                    return hijacked_callback(
                        frame, cache_entry, self.hooks, frame_state
                    )

        with compile_lock, _disable_current_modes():
            # skip=1: skip this frame
            return self._torchdynamo_orig_callable(
                frame, cache_entry, self.hooks, frame_state, skip=1
            )


def catch_errors_wrapper(
    callback: ConvertFrameProtocol, hooks: Hooks
) -> CatchErrorsWrapper:
    return CatchErrorsWrapper(callback, hooks)
