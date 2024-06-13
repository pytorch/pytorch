# mypy: allow-untyped-defs
import collections
import cProfile
import dis
import functools
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
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from torch._utils_internal import maybe_upload_prof_stats_to_manifold

from torch.fx._lazy_graph_module import (  # type: ignore[attr-defined]
    _use_lazy_graph_module,
)
from torch.utils._traceback import CapturedTraceback

try:
    import numpy as np
except ModuleNotFoundError:
    np = None  # type: ignore[assignment]

import torch
import torch._logging
from torch._guards import compile_context, CompileContext, CompileId, tracing
from torch._logging import structured
from torch._utils_internal import compile_time_strobelight_meta, signpost_event
from torch.fx.experimental.symbolic_shapes import (
    ConstraintViolationError,
    GuardOnDataDependentSymNode,
)
from torch.fx.graph_module import _forward_from_src as original_forward_from_src
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.utils._python_dispatch import _disable_current_modes
from torch.utils._traceback import format_traceback_short

from . import config, exc, trace_rules
from .backends.registry import CompilerFn
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
    exceeds_cache_size_limit,
    is_recompilation,
)
from .eval_frame import always_optimize_code_objects, skip_code, TorchPatcher
from .exc import (
    augment_exc_message,
    BackendCompilerFailed,
    format_error_msg,
    InternalTorchDynamoError,
    TorchRuntimeError,
    UncapturedHigherOrderOpError,
    unimplemented,
    Unsupported,
)
from .guards import (
    CheckFunctionManager,
    get_and_maybe_log_recompilation_reason,
    GuardedCode,
)
from .hooks import Hooks
from .replay_record import ExecutionRecord
from .symbolic_convert import InstructionTranslator, SpeculationLog
from .trace_rules import is_numpy
from .types import BytecodeHook
from .utils import (
    CleanupManager,
    CompilationMetrics,
    counters,
    dynamo_timed,
    format_bytecode,
    frame_phase_timing,
    gen_record_file_name,
    increment_frame,
    is_namedtuple,
    istype,
    LazyString,
    orig_code_map,
    record_compilation_metrics,
    reset_graph_break_dup_checker,
    setup_compile_debug,
    troubleshooting_url,
    write_record_to_file,
)

log = logging.getLogger(__name__)
bytecode_log = torch._logging.getArtifactLogger(__name__, "bytecode")
graph_break_log = torch._logging.getArtifactLogger(__name__, "graph_breaks")
GlobalStateGuard = torch._C._dynamo.guards.GlobalStateGuard

compile_lock = threading.RLock()


class Tracker:
    def __init__(self):
        self.seen = []
        self.seen_ids = set()

    def add(self, strong_obj):
        idx = id(strong_obj)
        if idx not in self.seen_ids:
            obj = weakref.ref(strong_obj, lambda _: self.seen_ids.remove(idx))
            self.seen.append(obj)
            self.seen_ids.add(idx)

    def __contains__(self, item):
        return id(item) in self.seen_ids

    def clear(self):
        self.seen.clear()
        self.seen_ids.clear()


input_codes = Tracker()
output_codes = Tracker()

initial_global_state: Optional[GlobalStateGuard] = None


@functools.wraps(original_forward_from_src)
def fx_forward_from_src_skip_result(*args, **kwargs):
    # we monkey patch FX to prevent infinite loop of trying to convert
    # our generated code
    result: types.FunctionType = original_forward_from_src(*args, **kwargs)
    skip_code(result.__code__)
    return result


def preserve_global_state(fn):
    """
    Context manager to:
        1) Save/restore torch.is_grad_enabled() state
        2) Save/restore python random state
        3) Save/restore torch random state
        4) Monkey patch torch.fx.graph_module._forward_from_src
    """

    @functools.wraps(fn)
    def _fn(*args, **kwargs):
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
            py_rng_state = random.getstate()
            torch_rng_state = torch.random.get_rng_state()
            if torch.cuda.is_available():
                cuda_rng_state = torch.cuda.get_rng_state()
            allow_tf32 = torch._C._get_cublas_allow_tf32()
            prior_fwd_from_src = torch.fx.graph_module._forward_from_src
            torch.fx.graph_module._forward_from_src = fx_forward_from_src_skip_result
            cleanup = setup_compile_debug()
            try:
                return fn(*args, **kwargs)
            finally:
                cleanup.close()
                torch._C._set_grad_enabled(prior_grad_mode)
                torch.autograd.grad_mode._enter_inference_mode(prior_inference_mode)
                torch.use_deterministic_algorithms(
                    prior_deterministic, warn_only=prior_warn_only
                )
                random.setstate(py_rng_state)
                torch.random.set_rng_state(torch_rng_state)
                if torch.cuda.is_available():
                    torch.cuda.set_rng_state(cuda_rng_state)  # type: ignore[possibly-undefined]
                torch._C._set_cublas_allow_tf32(allow_tf32)
                torch.fx.graph_module._forward_from_src = prior_fwd_from_src
                assert (
                    guards.check()
                ), f"Global {guards.reason()}state changed while dynamo tracing, please report a bug"

    _fn._torchdynamo_orig_callable = fn  # type: ignore[attr-defined]
    return _fn


@TorchPatcher.suppress_torch_distributed_warnings
def has_tensor_in_frame(frame):
    """Check if the frame has torch.* related bits"""
    # Check if the function was decorated using torch._dynamo.optimize
    if frame.f_code in always_optimize_code_objects:
        return True

    # Check if there is global import of torch.*
    for co_name in frame.f_code.co_names:
        if co_name in frame.f_globals:
            obj = frame.f_globals[co_name]
            if isinstance(obj, types.ModuleType) and (
                obj.__name__.startswith("torch.") or obj is torch
            ):
                return True
            # ... or a global import of numpy.*
            if np and config.trace_numpy and (obj is np or is_numpy(obj)):
                return True

    seen_ids: Dict[int, bool] = dict()

    def has_tensor(obj):
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


def exception_handler(e, code, frame=None, export=False):
    record_filename = None
    if hasattr(e, "exec_record"):
        record_filename = gen_record_file_name(e, code)
        write_record_to_file(record_filename, e.exec_record)
        e.record_filename = record_filename

    augment_exc_message(e, export=export)


FRAME_COUNTER = 0
FRAME_COMPILE_COUNTER: typing.Counter[int] = collections.Counter()


def maybe_cprofile(func):
    if config.cprofile:
        return cprofile_wrapper(func)
    return func


def cprofile_wrapper(func):
    @functools.wraps(func)
    def profile_wrapper(*args, **kwargs):
        trace_id = CompileContext.current_trace_id()
        assert trace_id, "Trace id is None"
        profile_path = Path(
            f"/tmp/{func.__name__}_{str(trace_id).replace('/','_')}.profile"
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
            log.warning("Cannot write to %s", str(profile_path))
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
            log.warning("Generated SVG from profile at %s", str(svg_path))
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
        export_constraints=None,
    ):
        reset_graph_break_dup_checker()
        self._torchdynamo_orig_callable = compiler_fn  # type: ignore[attr-defined]
        self._one_graph = one_graph
        self._export = export
        self._export_constraints = export_constraints

    @property
    def _clone_with_backend(self):
        return lambda backend: convert_frame_assert(
            backend, self._one_graph, self._export, self._export_constraints
        )

    def __call__(
        self,
        frame: types.FrameType,
        cache_entry,
        hooks: Hooks,
        frame_state,
        *,
        skip: int = 0,
    ):
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

        frame_compile_id = FRAME_COMPILE_COUNTER[frame_id]
        FRAME_COMPILE_COUNTER[frame_id] += 1

        compile_id = CompileId(frame_id, frame_compile_id)

        signpost_event(
            "dynamo",
            "_convert_frame_assert._compile",
            {
                "co_name": code.co_name,
                "co_filename": code.co_filename,
                "co_firstlineno": code.co_firstlineno,
                "cache_size": cache_size.num_cache_entries_with_same_id_matched_objs,
                "accumulated_cache_size": cache_size.num_cache_entries,
            },
        )

        return _compile(
            frame.f_code,
            frame.f_globals,
            frame.f_locals,
            frame.f_builtins,
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
    export_constraints=None,
):
    """Fully convert a frame into an FX graph"""
    return ConvertFrameAssert(compiler_fn, one_graph, export, export_constraints)


from collections import OrderedDict

from torch.utils.hooks import RemovableHandle

if typing.TYPE_CHECKING:
    from .output_graph import OutputGraph

# we have to use `OrderedDict` to make `RemovableHandle` work.
_bytecode_hooks: Dict[int, BytecodeHook] = OrderedDict()


def register_bytecode_hook(hook: BytecodeHook) -> RemovableHandle:
    """Register hooks for bytecode generated by Dynamo. The hook can do some
    logging, as well as return a new code object to be used. Please refer
    to `BytecodeHook` for the hook signature.
    """
    handle = RemovableHandle(_bytecode_hooks)
    _bytecode_hooks[handle.id] = hook
    return handle


@compile_time_strobelight_meta(phase_name="_compile")
@_use_lazy_graph_module(config.use_lazy_graph_module)
def _compile(
    code: types.CodeType,
    globals: Dict[str, object],
    locals: Dict[str, object],
    builtins: Dict[str, object],
    compiler_fn: CompilerFn,
    one_graph: bool,
    export: bool,
    export_constraints,
    hooks: Hooks,
    cache_entry,
    cache_size: CacheSizeRelevantForFrame,
    frame: Optional[types.FrameType] = None,
    frame_state=None,
    compile_id=None,
    *,
    skip: int = 0,
) -> Optional[GuardedCode]:
    from torch.fx.experimental.validator import (
        bisect,
        BisectValidationException,
        translation_validation_enabled,
        ValidationException,
    )

    # Time spent compiling this frame before restarting or failing analysis
    dynamo_time_before_restart: float = 0.0
    restart_reasons: set[str] = set()
    output: Optional[OutputGraph] = None
    tracer: Optional[InstructionTranslator] = None
    # This is shared across restarts
    mutated_closure_cell_contents: Set[str] = set()
    speculation_log = SpeculationLog()
    torch._dynamo.callback_handler.run_start_callbacks()

    @preserve_global_state
    def transform(instructions, code_options):
        nonlocal output
        nonlocal tracer
        speculation_log.restart()
        tracer = InstructionTranslator(
            instructions,
            code,
            locals,
            globals,
            builtins,
            code_options,
            compiler_fn,
            one_graph,
            export,
            export_constraints,
            mutated_closure_cell_contents,
            frame_state=frame_state,
            speculation_log=speculation_log,
        )

        try:
            with tracing(tracer.output.tracing_context), tracer.set_current_tx():
                tracer.run()
        except exc.UnspecializeRestartAnalysis:
            speculation_log.clear()
            raise
        except (exc.SpeculationRestartAnalysis, exc.SkipFrame):
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

        if config.dead_code_elimination:
            propagate_inst_exn_table_entries(instructions)
            check_inst_exn_tab_entries_valid(instructions)
            instructions[:] = remove_pointless_jumps(remove_dead_code(instructions))

    @dynamo_timed(phase_name="entire_frame_compile")
    @maybe_cprofile
    def compile_inner(
        code: types.CodeType,
        one_graph: bool,
        hooks: Hooks,
        transform: Callable[[List[Instruction], Dict[str, Any]], Any],
    ) -> Optional[GuardedCode]:
        nonlocal output
        nonlocal dynamo_time_before_restart
        nonlocal restart_reasons
        last_attempt_start_time = start_time = time.time()

        def log_bytecode(prefix, name, filename, line_no, code):
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

        for attempt in itertools.count():
            CompileContext.get().attempt = attempt
            try:
                out_code = transform_code_object(code, transform)
                break
            except exc.RestartAnalysis as e:
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

        log_bytecode(
            "MODIFIED BYTECODE",
            code.co_name,
            code.co_filename,
            code.co_firstlineno,
            out_code,  # type: ignore[possibly-undefined]
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

        def count_args(code):
            import inspect

            return (
                code.co_argcount
                + code.co_kwonlyargcount
                + bool(code.co_flags & inspect.CO_VARARGS)
                + bool(code.co_flags & inspect.CO_VARKEYWORDS)
            )

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
        check_fn = CheckFunctionManager(
            output,
            hooks.guard_fail_fn if hooks else None,
        )

        guarded_code = GuardedCode(out_code, check_fn.check_fn)

        if not output.is_empty_graph() and hooks.guard_export_fn is not None:
            # We should not run the guard_export_fn when Dynamo does not
            # generate any graph. This can happen in export when TorchDynamo
            # generated bytecode has some reconstruction logic for mutated
            # variables which can trigger TorchDynamo on the children frames but
            # they are benign and do not generate any new graphs.
            hooks.guard_export_fn(output.guards)

        return guarded_code

    with compile_context(CompileContext(compile_id)):
        # Check recompilations
        recompile_reasons = None
        if is_recompilation(cache_size) and frame:
            recompile_reasons = get_and_maybe_log_recompilation_reason(
                cache_entry, frame
            )

        exceeded, limit_type = exceeds_cache_size_limit(cache_size)
        if exceeded:

            def format_func_info(code):
                return f"'{code.co_name}' ({code.co_filename}:{code.co_firstlineno})"

            def format_guard_failures():
                if not recompile_reasons:
                    return "Unable to find recompilation reasons"
                return recompile_reasons[-1]

            log.warning(
                "torch._dynamo hit config.%s (%s)\n"
                "   function: %s\n"
                "   last reason: %s\n"
                'To log all recompilation reasons, use TORCH_LOGS="recompiles".\n'
                "To diagnose recompilation issues, see %s.",
                limit_type,
                getattr(config, limit_type),
                format_func_info(code),
                format_guard_failures(),
                troubleshooting_url,
            )
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
        torch._logging.trace_structured(
            "dynamo_start",
            lambda: {
                "stack": structured.from_traceback(
                    CapturedTraceback.extract(skip=4 + skip).summary()
                )
            },
        )
        start_time = time.time()
        fail_type: Optional[str] = None
        fail_reason: Optional[str] = None
        fail_user_frame_filename: Optional[str] = None
        fail_user_frame_lineno: Optional[int] = None
        guarded_code = None
        try:
            guarded_code = compile_inner(code, one_graph, hooks, transform)
            return guarded_code
        except (
            Unsupported,
            TorchRuntimeError,
            BackendCompilerFailed,
            AssertionError,
            ConstraintViolationError,
            GuardOnDataDependentSymNode,
            ValidationException,
            UncapturedHigherOrderOpError,
            BisectValidationException,
        ) as e:
            fail_type = str(type(e))
            fail_reason = str(e)
            exception_handler(e, code, frame, export=export)
            if e.innermost_user_frame_summary is not None:  # type: ignore[union-attr]
                fail_user_frame_filename = e.innermost_user_frame_summary.filename  # type: ignore[union-attr]
                fail_user_frame_lineno = e.innermost_user_frame_summary.lineno  # type: ignore[union-attr]
            e.compile_id = compile_id  # type: ignore[union-attr]
            raise
        except Exception as e:
            fail_type = str(type(e))
            fail_reason = str(e)
            exception_handler(e, code, frame, export=export)
            if e.innermost_user_frame_summary is not None:  # type: ignore[attr-defined]
                fail_user_frame_filename = e.innermost_user_frame_summary.filename  # type: ignore[attr-defined]
                fail_user_frame_lineno = e.innermost_user_frame_summary.lineno  # type: ignore[attr-defined]
            e.compile_id = compile_id  # type: ignore[attr-defined]
            raise InternalTorchDynamoError(str(e)).with_traceback(
                e.__traceback__
            ) from None
        finally:
            if tracer:
                tracer.output.local_scope = {}

            from .utils import curr_frame

            frame_key = str(curr_frame)
            if (
                fail_reason is None
                and output is not None
                and frame_key in frame_phase_timing
            ):
                guard_count = len(output.guards)
                shape_env_guard_count = len(output.shape_env.guards)
                graph_op_count = output.count_calls()
                graph_node_count = len(output.graph.nodes)
                graph_input_count = len(output.placeholders)
                entire_frame_compile_time = frame_phase_timing[frame_key].get(
                    "entire_frame_compile", None
                )
                backend_compile_time = frame_phase_timing[frame_key].get(
                    "backend_compile", None
                )
                inductor_compile_time = frame_phase_timing[frame_key].get(
                    "inductor_compile", None
                )
                code_gen_time = frame_phase_timing[frame_key].get("code_gen", None)
                non_compliant_ops = {op.__qualname__ for op in output.non_compliant_ops}
                compliant_custom_ops = {
                    op.__qualname__ for op in output.compliant_custom_ops
                }
            else:
                guard_count = None
                shape_env_guard_count = None
                graph_op_count = None
                graph_node_count = None
                graph_input_count = None
                entire_frame_compile_time = None
                backend_compile_time = None
                inductor_compile_time = None
                code_gen_time = None
                non_compliant_ops = set({})
                compliant_custom_ops = set({})
                restart_reasons = set()
                # If compilation failed, the entire time is wasted
                dynamo_time_before_restart = time.time() - start_time

            metrics = CompilationMetrics(
                str(compile_id),
                frame_key,
                code.co_name,
                code.co_filename,
                code.co_firstlineno,
                cache_size.num_cache_entries_with_same_id_matched_objs,
                cache_size.num_cache_entries,
                guard_count,
                shape_env_guard_count,
                graph_op_count,
                graph_node_count,
                graph_input_count,
                start_time,
                entire_frame_compile_time,
                backend_compile_time,
                inductor_compile_time,
                code_gen_time,
                fail_type,
                fail_reason,
                fail_user_frame_filename,
                fail_user_frame_lineno,
                non_compliant_ops,
                compliant_custom_ops,
                restart_reasons,
                dynamo_time_before_restart,
                guarded_code is not None,
            )
            record_compilation_metrics(metrics)
            torch._dynamo.callback_handler.run_end_callbacks()


class ConvertFrame:
    def __init__(self, compiler_fn: CompilerFn, hooks: Hooks):
        self._torchdynamo_orig_callable = compiler_fn
        self._inner_convert = convert_frame_assert(compiler_fn, one_graph=False)
        self._hooks = hooks

    @property
    def _clone_with_backend(self):
        return lambda backend: convert_frame(backend, self._hooks)

    def __call__(
        self,
        frame: types.FrameType,
        cache_entry,
        hooks: Hooks,
        frame_state,
        skip: int = 0,
    ):
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
                        graph_break_log.debug(
                            "Graph break: skip: from user code at:\n%s",
                            user_stack_formatted,
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
        return None


def convert_frame(compiler_fn: CompilerFn, hooks: Hooks):
    """Try to convert a frame into an FX graph, if error leave frame unmodified"""
    return ConvertFrame(compiler_fn, hooks)


# TODO mlazos: add support for same args, or record them
def replay(filename):
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
            compiler_fn=eager,
            one_graph=False,
            export=False,
            export_constraints=None,
            hooks=Hooks(),
            cache_size=CacheSizeRelevantForFrame(0, 0),
            frame=None,
            frame_state={},
        )
    finally:
        config.replay_record_enabled = original_replay_val


def first_real_inst_idx(code):
    if sys.version_info < (3, 11):
        return 0
    for inst in dis.get_instructions(code):
        if inst.opname == "RESUME":
            return inst.offset // 2
    raise RuntimeError("RESUME instruction not found in code")


class CatchErrorsWrapper:
    def __init__(self, callback, hooks):
        functools.wraps(callback)(self)
        self._torchdynamo_orig_callable = callback
        self.hooks = hooks

    def __call__(self, frame, cache_entry, frame_state):
        assert frame_state is not None

        is_skipfile = trace_rules.check(frame.f_code)
        if (
            # TODO: the first condition is not covered by any test
            frame.f_lasti >= first_real_inst_idx(frame.f_code)
            or is_skipfile
            or config.disable
        ):
            if log.isEnabledFor(logging.DEBUG):
                skip_reason = (
                    "traced frame already"
                    if frame.f_lasti >= first_real_inst_idx(frame.f_code)
                    else (
                        "in skipfiles"
                        if trace_rules.check(frame.f_code)
                        else "dynamo tracing is disabled"
                    )
                )
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
                        backend_compile_fn=self._torchdynamo_orig_callable._torchdynamo_orig_callable,
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


def catch_errors_wrapper(callback, hooks: Hooks):
    return CatchErrorsWrapper(callback, hooks)
