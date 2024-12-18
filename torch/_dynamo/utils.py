# mypy: allow-untyped-defs
from __future__ import annotations

import atexit
import collections
import contextlib
import copy
import dataclasses
import datetime
import dis
import enum
import functools
import gc
import importlib
import inspect
import itertools
import json
import linecache
import logging
import math
import operator
import os
import re
import sys
import textwrap
import threading
import time
import traceback
import types
import typing
import uuid
import warnings
import weakref
from contextlib import contextmanager
from dataclasses import is_dataclass
from functools import lru_cache
from types import MethodWrapperType
from typing import (
    Any,
    Callable,
    cast,
    ClassVar,
    Counter,
    DefaultDict,
    Deque,
    Dict,
    Generator,
    Generic,
    Iterable,
    Iterator,
    KeysView,
    List,
    Optional,
    overload,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    ValuesView,
)
from typing_extensions import Literal, TypeIs

import torch
import torch._functorch.config
import torch.fx.experimental.symbolic_shapes
import torch.utils._pytree as pytree
from torch import fx
from torch._C import (
    _instruction_counter,
    _len_torch_function_stack,
    _pop_torch_function_stack,
    _push_on_torch_function_stack,
)
from torch._dispatch.python import enable_python_dispatcher
from torch._dynamo.metrics_context import MetricsContext, RuntimeMetricsContext
from torch._guards import CompileId, Source, TracingContext
from torch._subclasses.meta_utils import is_sparse_compressed
from torch._utils_internal import (
    log_chromium_event_internal,
    log_compilation_event,
    record_chromium_event_internal,
    signpost_event,
)
from torch.fx._utils import _format_graph_code, lazy_format_graph_code
from torch.monitor import _WaitCounter
from torch.nn.modules.lazy import LazyModuleMixin
from torch.utils._triton import has_triton, has_triton_package
from torch.utils.hooks import RemovableHandle


try:
    import numpy as np
except ModuleNotFoundError:
    np = None  # type: ignore[assignment]

try:
    import torch._logging
    import torch._numpy as tnp
    from torch._guards import detect_fake_mode  # noqa: F401n
    from torch._logging import LazyString

    from . import config

    # NOTE: Make sure `NP_SUPPORTED_MODULES` and `NP_TO_TNP_MODULE` are in sync.
    if np:
        NP_SUPPORTED_MODULES: Tuple[types.ModuleType, ...] = (
            np,
            np.fft,
            np.linalg,
            np.random,
        )

        NP_TO_TNP_MODULE = {
            np: tnp,
            np.fft: tnp.fft,
            np.linalg: tnp.linalg,
            np.random: tnp.random,
        }
    else:
        NP_SUPPORTED_MODULES = ()

        NP_TO_TNP_MODULE = {}
    from torch._subclasses.fake_tensor import FakeTensor, is_fake, maybe_get_fake_mode
except ImportError:
    pass


T = TypeVar("T")

unpatched_nn_module_getattr = torch.nn.Module.__getattr__
unpatched_nn_module_call = torch.nn.Module.__call__
unpatched_nn_module_call_impl = torch.nn.Module._call_impl

counters: DefaultDict[str, Counter[str]] = collections.defaultdict(collections.Counter)
optimus_scuba_log: Dict[str, Any] = {}
troubleshooting_url = (
    "https://pytorch.org/docs/main/torch.compiler_troubleshooting.html"
)
nnmodule_doc_url = "https://pytorch.org/docs/main/torch.compiler_nn_module.html"
nnmodule_doc_url_msg = f"See {nnmodule_doc_url} for more information and limitations."
log = logging.getLogger(__name__)

# profiling compilation time by function
compilation_time_metrics: Dict[str, List[float]] = {}

# This supports calculate_time_spent(), which reports cumulative times
# across the process for any "phase" populated by dynamo_timed. Reset if
# reset_frame_count() is called.
cumulative_time_spent_ns: Dict[str, float] = collections.defaultdict(float)

timer_counter = itertools.count()


# Abstraction on top of counters.
class ReInplaceTrigger(enum.Enum):
    AUTO_FUNC_V1 = 1
    AUTO_FUNC_V2 = 2
    TRITON_OPS = 3


class ReinplaceCounters:
    _values: DefaultDict[str, int] = collections.defaultdict(int)

    # Track sizes of known not re-inplaced tensors (exclude dynamic shapes).
    @classmethod
    def add_missed_bytes(cls, trigger: ReInplaceTrigger, bytes: int):
        if bytes != 0:
            cls._values[f"missed_bytes_{trigger.name}"] += bytes

    # Track number of not re-inplaced tensors.
    @classmethod
    def add_missed_opportunities(cls, trigger: ReInplaceTrigger, count: int):
        if count != 0:
            cls._values[f"missed_tensors_{trigger}"] += count

    @classmethod
    def clear(cls):
        cls._values.clear()

    @classmethod
    def get_total_missed(cls):
        sum = 0
        for trigger in ReInplaceTrigger:
            sum += cls._values.get(f"missed_tensors_{trigger}", 0)
        return sum

    @classmethod
    def get_total_missed_bytes(cls):
        sum = 0
        for trigger in ReInplaceTrigger:
            sum += cls._values.get(f"missed_bytes_{trigger.name}", 0)
        return sum

    @classmethod
    def log(cls):
        # if not empty log.
        if cls._values:
            signpost_event("inductor", "reinplace_counters", cls._values)


def tabulate(
    rows: Union[List[Tuple[str, object]], List[List[object]]],
    headers: Union[Tuple[str, ...], List[str]],
) -> str:
    try:
        import tabulate

        return tabulate.tabulate(rows, headers=headers)
    except ImportError:
        return "\n".join(
            ", ".join(map(str, row)) for row in itertools.chain([headers], rows)
        )


curr_frame = 0


# Note: Called for you by dynamo - you almost never ever want to invoke this yourself.
def increment_frame() -> None:
    global curr_frame
    curr_frame = curr_frame + 1


# Note: Called for you by dynamo - you almost never ever want to invoke this yourself.
def reset_frame_count() -> None:
    global curr_frame
    cumulative_time_spent_ns.clear()
    compilation_time_metrics.clear()
    curr_frame = 0


op_count = 0


def increment_op_count(cnt: int) -> None:
    global op_count
    op_count += cnt


# Get the total time in seconds for each "phase"
# For example, {'entire_frame_compile':8.574629999999999, 'backend_compile':5.26806}
def calculate_time_spent() -> Dict[str, float]:
    total_by_key = {}
    for phase, timing in cumulative_time_spent_ns.items():
        total_by_key[phase] = timing / 1e9

    total_by_key["total_wall_time"] = total_by_key.get(
        "entire_frame_compile", 0
    ) + total_by_key.get("entire_backward_compile", 0)
    return total_by_key


# Print a report of time spent so far
# Ex:
# TIMING:
# entire_frame_compile:8.574629999999999
# backend_compile:5.26806
def print_time_report() -> None:
    total_by_key = calculate_time_spent()

    out = "TIMING:"
    for key, value in total_by_key.items():
        out = f"{out} {key}:{round(value, 5)}"

    print(out)


# Use the following singleton to capture and log CompilationMetrics. Entering the context
# manager allocates a new record to be logged when it exits. (You should not need to use
# this directly unless you introduce a new code path where compilation metrics would be
# gathered). While compiling, use the setters or timer in MetricsContext to update fields
# in the current context. For example:
#
# To set a single field once (use overwrite=True to overwrite):
#   get_metrics_context().set("metric_name", value)
#
# To set multiple fields at once (use overwrite=True to overwrite):
#   get_metrics_context().update({"name1": val1, "name2": val2})
#
# To increment an integer field:
#   get_metrics_context().increment("metric_name", value)
#
# To record execution time, MetricsContext works with dynamo_timed:
#    def foo(...):
#        # Updates the "metric_us" field.
#        with dynamo_timed("metric", dynamo_compile_column_us="metric_us")
#            ...
#
_METRICS_CONTEXT: MetricsContext
_RUNTIME_METRICS_CONTEXT: RuntimeMetricsContext


def get_metrics_context() -> MetricsContext:
    return _METRICS_CONTEXT


def get_runtime_metrics_context() -> RuntimeMetricsContext:
    return _RUNTIME_METRICS_CONTEXT


@contextmanager
def dynamo_timed(
    key: str,
    # TODO(masneral): Deprecate this param.
    phase_name: Optional[str] = None,
    log_pt2_compile_event: bool = False,
    metadata: Optional[Dict[str, object]] = None,
    dynamo_compile_column_us: Optional[str] = None,
    dynamo_compile_runtime_column_us: Optional[str] = None,
    compile_id: Optional[CompileId] = None,
    is_forward: Optional[bool] = None,
    log_waitcounter: bool = False,
) -> Generator[Any, None, None]:
    """
    dynamo_timed is a context manager
    By wrapping a function in dynamo_timed, we can get a few things:

    1) Optionally log timings to pt2_compile_events.
    2) Optionally log timings to CompilationMetrics (dynamo_compile).
    3) Optionally log chromium events.
    4) Optionally increment a WaitCounter.
    5) Store a record in compilation_time_metrics
       For example:

        def _foo(...):
            with dynamo_timed("_foo"):
                ...

        Would show up as an entry in our timing dict:
        OrderedDict([('_foo', [0.083690, 0.23949, 3.1425e-05])])
        This is extremely useful for granular debugging.

    Although it is tempting to use dynamo_timed as a decorator, please do not.
    In its decorator form it makes cProfile traces less useful as dynamo_timed
    suddenly becomes a bottleneck for lots of function calls (as only one parent
    pointer is recorded).

    Params:
    - key: key into compile_time_metrics. If phase_name is not provided, this is
      also the event name used for pt2_compile_events logs and chromium events.
    - phase_name: Optional override for the event name.
    - log_pt2_compile_event: Whether to log a pt2 compile event internally.
    - metadata: Extra metadata to put in pt2_compile_events.
    - dynamo_compile_column_us: If provided, updates the specified CompilationMetrics
      field to be logged to dyname_compile column. We expect all columns to be _us;
      therefore, the field name must end with "_us".
    - dynamo_compile_runtime_column_us: Like 'dynamo_compile_column_us', but should
      be used for those columns captured outside of a compile context, e.g.,
      runtime autotuning.
    - compile_id: In the typical case, this parameter should not be needed. Use to
      supply the compile_id for those cases where we want to log a compile_id where
      it's not naturally available, e.g., for runtime autotuning.
    - is_forward: Optionally set an is_forward field for those logging destinations
      that support it.
    - log_waitcounter: If set, we'll log a waitcounter of the form "pytorch.dynamo_timed.{key}"
    """
    # We're standardizing on microseconds for dynamo_compile timings.
    if dynamo_compile_column_us is not None:
        assert dynamo_compile_column_us.endswith("_us")

    # Only one of these should be set.
    assert dynamo_compile_column_us is None or dynamo_compile_runtime_column_us is None

    if phase_name:
        event_name = phase_name
        fn_name = key
    else:
        event_name = key
        fn_name = None

    if key not in compilation_time_metrics:
        compilation_time_metrics[key] = []

    event_metadata = {}
    if metadata:
        event_metadata.update(metadata)
    if fn_name:
        event_metadata.update({"fn_name": fn_name})
    if is_forward is not None:
        event_metadata.update({"is_backward": not is_forward})

    chromium_log: ChromiumEventLogger = get_chromium_event_logger()
    start_ns = time.time_ns()
    chromium_log.log_event_start(
        event_name, start_ns, event_metadata, log_pt2_compile_event, compile_id
    )

    try:
        with torch.profiler.record_function(f"{key} (dynamo_timed)"):
            if log_waitcounter:
                with _WaitCounter(f"pytorch.dynamo_timed.{key}").guard():
                    yield
            else:
                yield
    finally:
        end_ns = time.time_ns()
        time_spent_ns = end_ns - start_ns
        compilation_time_metrics[key].append(time_spent_ns / 1e9)
        chromium_log.log_event_end(
            event_name, end_ns, {}, start_ns, log_pt2_compile_event, compile_id
        )
        if dynamo_compile_column_us:
            metrics_context = get_metrics_context()
            if metrics_context.in_progress():
                metrics_context.increment(
                    dynamo_compile_column_us, time_spent_ns // 1000
                )
            # TODO: the events that we capture in calculate_time_spent() seem a little
            # arbitrary. Currently, it's only those fields that are present in
            # CompilationMetrics (but note that we accumulate by the associated event
            # name, not the field name in CompilationMetrics). Do we want to keep it
            # this way?
            cumulative_time_spent_ns[event_name] += time_spent_ns

        if dynamo_compile_runtime_column_us:
            get_runtime_metrics_context().increment(
                dynamo_compile_runtime_column_us,
                time_spent_ns // 1000,
                extra={
                    "compile_id": compile_id,
                    "is_runtime": True,
                    "is_forward": is_forward,
                },
            )
            cumulative_time_spent_ns[event_name] += time_spent_ns


@overload
def compile_times(repr: Literal["str"], aggregate: bool = False) -> str:
    ...


@overload
def compile_times(
    repr: Literal["csv"], aggregate: bool = False
) -> Tuple[List[str], List[object]]:
    ...


def compile_times(repr="str", aggregate: bool = False):
    """
    Get metrics about torchdynamo frontend/backend compilation times.

    Accumulates information from functions tagged with `dynamo_timed`.

    repr='str' returns a printable string for user interaction, and 'csv'
    returns headers, rows which can be logged for output

    aggregate causes values from multiple compilations (e.g. split graphs)
    to be accumulated into one value.  If false, expect more than one value
    per metric.
    """

    def fmt_fn(values, item_fn=lambda x: x):
        if aggregate:
            return item_fn(sum(values))
        return ", ".join(map(item_fn, values))

    if repr == "str":
        rows = [
            (k, fmt_fn(compilation_time_metrics[k], item_fn=lambda x: f"{x:.4f}"))
            for k in compilation_time_metrics
        ]
        out = "TorchDynamo compilation metrics:\n"
        out += tabulate(rows, headers=("Function", "Runtimes (s)"))
        return out
    elif repr == "csv":
        values = [
            fmt_fn(v, item_fn=lambda x: f"{x:.6f}")
            for v in compilation_time_metrics.values()
        ]
        headers = list(compilation_time_metrics.keys())
        return headers, values
    return None


@atexit.register
def dump_compile_times() -> None:
    log.info(compile_times(repr="str", aggregate=True))


tensortype_to_dtype = {
    torch.FloatTensor: (torch.float32, torch.float),
    torch.DoubleTensor: (torch.float64, torch.double),
    torch.HalfTensor: (torch.float16, torch.half),
    torch.BFloat16Tensor: (torch.bfloat16,),
    torch.ByteTensor: (torch.uint8,),
    torch.CharTensor: (torch.int8,),
    torch.LongTensor: (torch.int64, torch.long),
    torch.IntTensor: (torch.int32, torch.int),
    torch.ShortTensor: (torch.int16, torch.short),
    torch.BoolTensor: (torch.bool,),
}


class DuplicateWarningChecker:
    def __init__(self, maxsize: int = 4096) -> None:
        self.maxsize = maxsize
        self.reset()

    def reset(self):
        self.set = collections.OrderedDict()

    def add(self, key: Union[str, Tuple[object, object]]) -> bool:
        if key in self.set:
            self.set.move_to_end(key, last=True)
            if not config.verbose:
                return False
        else:
            self.set[key] = None
            while len(self.set) > self.maxsize:
                self.set.popitem(last=False)
        return True


graph_break_dup_warning_checker = DuplicateWarningChecker()


def setup_compile_debug():
    compile_debug = os.environ.get("TORCH_COMPILE_DEBUG", "0") == "1"

    if compile_debug:
        return add_file_handler()

    return contextlib.ExitStack()


def reset_graph_break_dup_checker() -> None:
    graph_break_dup_warning_checker.reset()


def add_file_handler():
    log_path = os.path.join(get_debug_dir(), "torchdynamo")
    os.makedirs(log_path, exist_ok=True)

    log_file_handler = logging.FileHandler(os.path.join(log_path, "debug.log"))
    logger = logging.getLogger("torch._dynamo")
    logger.addHandler(log_file_handler)

    exitstack = contextlib.ExitStack()
    exitstack.callback(lambda: logger.removeHandler(log_file_handler))
    return exitstack


def setup_log_file():
    exitstack = contextlib.ExitStack()
    if config.log_file_name is not None:
        log_file_handler = logging.FileHandler(config.log_file_name)
        for logger in torch._logging._internal.get_loggers():
            logger.addHandler(log_file_handler)
            exitstack.callback(lambda: logger.removeHandler(log_file_handler))
        return exitstack

    return exitstack


def gen_record_file_name(exc, code) -> str:
    return f"{get_debug_dir()}/error_recordings/\
{code.co_name}_{type(exc).__name__}_{code.co_firstlineno}.rec"


def write_record_to_file(filename: str, exec_record) -> None:
    try:
        if os.path.exists(filename):
            log.warning(
                "Unable to write execution record %s; file already exists.", filename
            )
        else:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, "wb") as f:
                exec_record.dump(f)
    except Exception:
        log.exception("Unable to write execution record %s", filename)


def count_calls(g: fx.Graph) -> int:
    c = 0
    for n in g.nodes:
        if "call" in n.op:
            c += 1
    return c


def identity(x: T) -> T:
    return x


def hashable(x):
    try:
        hash(x)
        return True
    except TypeError:
        return False
    # cannot hash writable memoryview object
    except ValueError:
        return False


def nothing(*args, **kwargs):
    pass


class ExactWeakKeyDictionary:
    """Similar to weakref.WeakKeyDictionary, but use `is`/`id` rather than `==` to compare equality"""

    def __init__(self):
        self.values = {}
        self.refs = {}

    def __getitem__(self, key):
        return self.values[id(key)]

    def get(self, key, default=None):
        return self.values.get(id(key), default)

    def __contains__(self, key):
        return id(key) in self.values

    def __setitem__(self, key, value):
        idx = id(key)
        if idx not in self.refs:
            self.refs[idx] = weakref.ref(key, lambda ref: self._remove_id(idx))
        self.values[idx] = value

    def _remove_id(self, idx):
        if idx in self.values:
            del self.values[idx]
        if idx in self.refs:
            del self.refs[idx]

    def clear(self):
        self.refs.clear()
        self.values.clear()


@overload
def istype(obj: object, allowed_types: Type[T]) -> TypeIs[T]:
    ...


@overload
def istype(
    obj: object, allowed_types: Tuple[Type[List[T]], Type[Tuple[T, ...]]]
) -> TypeIs[T]:
    ...


@overload
def istype(obj: object, allowed_types: Iterable[type]) -> bool:
    ...


def istype(obj, allowed_types):
    """isinstance() without subclasses"""
    if isinstance(allowed_types, (tuple, list, set)):
        return type(obj) in allowed_types
    return type(obj) is allowed_types


if sys.version_info >= (3, 12):
    # Some typing classes moved to C in 3.12,
    # which no longer have the _Final mixin.
    _builtin_final_typing_classes = (
        typing.ParamSpecArgs,
        typing.ParamSpecKwargs,
        typing.ParamSpec,
        typing.TypeVar,
        typing.TypeVarTuple,
        typing.TypeAliasType,
    )


def is_typing(value):
    # _Final catches most of typing classes:
    #   - Any
    #   - Callable
    #   - Union
    #   ...
    #
    # NB: we intentionally ignore classes that inherit from Generic, since they
    # can be used as both TypingVariable as well as UserDefinedClassVariable.
    if sys.version_info >= (3, 12) and isinstance(value, _builtin_final_typing_classes):
        return True
    return isinstance(value, typing._Final) or value is typing.Generic  # type: ignore[attr-defined]


def is_numpy_int_type(value):
    if not np:
        return False

    return istype(
        value,
        (
            np.int8,
            np.int16,
            np.int32,
            np.int64,
            np.uint8,
            np.uint16,
            np.uint32,
            np.uint64,
        ),
    )


def is_numpy_float_type(value):
    if not np:
        return False

    return istype(
        value,
        (
            np.float16,
            np.float32,
            np.float64,
        ),
    )


def is_lru_cache_wrapped_function(value):
    return isinstance(value, functools._lru_cache_wrapper) and is_function(
        inspect.getattr_static(value, "__wrapped__")
    )


def is_function_or_wrapper(value):
    return is_function(value) or isinstance(
        value, (torch._ops.OpOverloadPacket, torch._ops.OpOverload)
    )


def is_function(value):
    return isinstance(
        value,
        (
            types.FunctionType,
            types.BuiltinFunctionType,
            types.MethodDescriptorType,
            types.WrapperDescriptorType,
        ),
    )


def is_wrapper_or_member_descriptor(value):
    return isinstance(
        value,
        (
            # set up by PyGetSetDef
            types.GetSetDescriptorType,
            # set by PyMethodDef, e.g. list.append
            types.MethodDescriptorType,
            # slots - list.__add__
            types.WrapperDescriptorType,
            # set up by PyMemberDef
            types.MemberDescriptorType,
            # wrapper over C functions
            types.MethodWrapperType,
        ),
    )


def unwrap_if_wrapper(fn):
    return unwrap_with_attr_name_if_wrapper(fn)[0]


def unwrap_with_attr_name_if_wrapper(fn):
    # TODO(anijain2305) - Investigate if we can get rid of this function
    # unpack @torch._dynamo.optimize()(fn) wrapped function
    if is_function(fn) and inspect.getattr_static(fn, "_torchdynamo_inline", False):
        fn = inspect.getattr_static(fn, "_torchdynamo_inline", fn)
        attr_name = "_torchdynamo_inline"
    else:
        attr_name = None
    return fn, attr_name


def is_numpy_ndarray(value):
    if not np:
        return False

    return istype(value, np.ndarray)


def istensor(obj):
    """Check of obj is a tensor"""
    tensor_list: Tuple[type, ...] = (
        torch.Tensor,
        torch.nn.Parameter,
        *config.traceable_tensor_subclasses,
    )
    tensor_list = tensor_list + (torch._subclasses.FakeTensor,)
    return istype(obj, tensor_list)


def is_lazy_module(mod):
    return isinstance(mod, LazyModuleMixin)


@functools.lru_cache(4096)
def print_once(*args):
    print(*args)


def make_cell(val=None):
    """Some black magic to create a cell object that usually only exists in a closure"""
    x = val

    def f():
        return x

    assert f.__closure__ is not None and len(f.__closure__) == 1
    return f.__closure__[0]


def proxy_args_kwargs(args, kwargs):
    try:
        proxy_args = tuple(arg.as_proxy() for arg in args)
        proxy_kwargs = {key: arg.as_proxy() for key, arg in kwargs.items()}
        return proxy_args, proxy_kwargs
    except NotImplementedError as e:
        from .exc import unimplemented
        from .variables.base import typestr

        unimplemented(
            f"call_function args: {typestr(*args)} {typestr(*list(kwargs.values()))}",
            from_exc=e,
        )


def to_int_ms(v: Optional[float]) -> Optional[int]:
    return None if v is None else int(v * 1000)


# float64 timestamp has a quarter microsecond precision in 2024, so while
# this is suboptimal we shouldn't meaningfully lose precision
def to_int_us(v: Optional[float]) -> Optional[int]:
    return None if v is None else int(v * 1_000_000)


# Version field added to every log. Increment to make it easier to distinguish new
# vs. old entries when you make a substantive change to how the logs are populated.
LOG_FORMAT_VERSION = 3


@dataclasses.dataclass
class CompilationMetrics:
    compile_id: Optional[str] = None
    frame_key: Optional[str] = None
    co_name: Optional[str] = None
    co_filename: Optional[str] = None
    co_firstlineno: Optional[int] = None
    cache_size: Optional[int] = None
    accumulated_cache_size: Optional[int] = None
    guard_count: Optional[int] = None
    shape_env_guard_count: Optional[int] = None
    graph_op_count: Optional[int] = None
    graph_node_count: Optional[int] = None
    graph_input_count: Optional[int] = None
    start_time: Optional[float] = None
    entire_frame_compile_time_s: Optional[float] = None
    backend_compile_time_s: Optional[float] = None
    inductor_compile_time_s: Optional[float] = None
    code_gen_time_s: Optional[float] = None
    fail_type: Optional[str] = None
    fail_reason: Optional[str] = None
    fail_user_frame_filename: Optional[str] = None
    fail_user_frame_lineno: Optional[int] = None
    non_compliant_ops: Optional[Set[str]] = None
    compliant_custom_ops: Optional[Set[str]] = None
    restart_reasons: Optional[Set[str]] = None
    dynamo_time_before_restart_s: Optional[float] = None
    # Sometimes, we will finish analyzing a frame but conclude we don't want
    # to install any guarded code.  True means we actually decided to install
    # a compiled frame
    has_guarded_code: Optional[bool] = None
    remote_cache_time_saved_s: Optional[float] = None
    structured_logging_overhead_s: Optional[float] = None
    config_suppress_errors: Optional[bool] = None
    config_inline_inbuilt_nn_modules: Optional[bool] = None
    specialize_float: Optional[bool] = None
    dynamo_config: Optional[str] = None
    is_forward: Optional[bool] = None
    num_triton_bundles: Optional[int] = None
    remote_fx_graph_cache_get_time_ms: Optional[int] = None
    remote_fx_graph_cache_put_time_ms: Optional[int] = None
    start_time_us: Optional[int] = None
    duration_us: Optional[int] = None
    dynamo_cumulative_compile_time_us: Optional[int] = None
    aot_autograd_cumulative_compile_time_us: Optional[int] = None
    inductor_cumulative_compile_time_us: Optional[int] = None
    inductor_code_gen_cumulative_compile_time_us: Optional[int] = None
    triton_compile_time_us: Optional[int] = None
    runtime_cudagraphify_time_us: Optional[int] = None  # TODO: instrument
    runtime_triton_autotune_time_us: Optional[int] = None
    dynamo_compile_time_before_restart_us: Optional[int] = None
    cuda_synchronize_time_us: Optional[int] = None  # TODO: instrument
    distributed_ephemeral_timeout_us: Optional[int] = None
    structured_logging_overhead_us: Optional[int] = None
    remote_fx_graph_cache_get_time_us: Optional[int] = None
    remote_fx_graph_cache_put_time_us: Optional[int] = None
    backward_cumulative_compile_time_us: Optional[int] = None
    end_time_us: Optional[int] = None
    pre_grad_pass_time_us: Optional[int] = None
    post_grad_pass_time_us: Optional[int] = None
    joint_graph_pass_time_us: Optional[int] = None
    log_format_version: int = LOG_FORMAT_VERSION
    inductor_config: Optional[str] = None
    remote_cache_version: Optional[int] = None
    inductor_fx_remote_cache_hit_count: Optional[int] = None
    inductor_fx_remote_cache_miss_count: Optional[int] = None
    inductor_fx_remote_cache_backend_type: Optional[str] = None
    inductor_fx_remote_cache_hit_keys: Optional[str] = None
    inductor_fx_remote_cache_miss_keys: Optional[str] = None
    cuda_version: Optional[str] = None
    triton_version: Optional[str] = None
    feature_usage: Optional[dict[str, bool]] = None
    compile_time_autotune_time_us: Optional[int] = None
    is_runtime: Optional[bool] = False
    gc_time_us: Optional[int] = None


DEFAULT_COMPILATION_METRICS_LIMIT = 64


_compilation_metrics: Deque[CompilationMetrics] = collections.deque(
    maxlen=DEFAULT_COMPILATION_METRICS_LIMIT
)


def add_compilation_metrics_to_chromium(c: CompilationMetrics) -> None:
    event_logger = get_chromium_event_logger()
    event_name = event_logger.get_top()
    if not event_name:
        return
    event_logger.add_event_data(
        event_name=event_name,
        frame_key=c.frame_key,
        co_name=c.co_name,
        co_filename=c.co_filename,
        co_firstlineno=c.co_firstlineno,
        cache_size=c.cache_size,
        accumulated_cache_size=c.accumulated_cache_size,
        guard_count=c.guard_count,
        shape_env_guard_count=c.shape_env_guard_count,
        graph_op_count=c.graph_op_count,
        graph_node_count=c.graph_node_count,
        graph_input_count=c.graph_input_count,
        fail_type=c.fail_type,
        fail_reason=c.fail_reason,
        fail_user_frame_filename=c.fail_user_frame_filename,
        fail_user_frame_lineno=c.fail_user_frame_lineno,
        # Sets aren't JSON serializable
        non_compliant_ops=list(c.non_compliant_ops)
        if c.non_compliant_ops is not None
        else None,
        compliant_custom_ops=list(c.compliant_custom_ops)
        if c.compliant_custom_ops is not None
        else None,
        restart_reasons=list(c.restart_reasons)
        if c.restart_reasons is not None
        else None,
        dynamo_time_before_restart_s=c.dynamo_time_before_restart_s,
        has_guarded_code=c.has_guarded_code,
        dynamo_config=c.dynamo_config,
    )


def _scrubbed_inductor_config_for_logging() -> Optional[str]:
    """
    Method to parse and scrub uninteresting configs from inductor config
    """

    # TypeSafeSerializer for json.dumps()
    # Skips complex types as values in config dict
    class TypeSafeSerializer(json.JSONEncoder):
        def default(self, o):
            try:
                return super().default(o)
            except Exception:
                return "Value is not JSON serializable"

    configs_to_scrub_re = r"((^TYPE_CHECKING$)|(.*_progress$)|(.*TESTING.*)|(.*(rocm|halide).*)|(^trace\..*)|(^_))"
    keys_to_scrub = set()
    inductor_conf_str = None
    inductor_config_copy = (
        torch._inductor.config.get_config_copy() if torch._inductor.config else None
    )
    if inductor_config_copy is not None:
        try:
            for key, val in inductor_config_copy.items():
                if not isinstance(key, str) or re.search(configs_to_scrub_re, key):
                    keys_to_scrub.add(key)
                # Convert set() to list for json.dumps()
                if isinstance(val, set):
                    inductor_config_copy[key] = list(val)
            # Evict unwanted keys
            for key in keys_to_scrub:
                del inductor_config_copy[key]
            # Stringify Inductor config
            inductor_conf_str = json.dumps(
                inductor_config_copy,
                cls=TypeSafeSerializer,
                skipkeys=True,
                sort_keys=True,
            )
        except Exception:
            # Don't crash because of runtime logging errors
            inductor_conf_str = "Inductor Config is not JSON serializable"
    return inductor_conf_str


def record_compilation_metrics(
    start_time_ns: int,
    end_time_ns: int,
    metrics: Dict[str, Any],
    exc_type: Optional[Type[BaseException]],
    exc_value: Optional[BaseException],
):
    def us_to_s(field):
        metric = metrics.get(field, None)
        return metric / 1e6 if metric is not None else None

    def us_to_ms(field):
        metric = metrics.get(field, None)
        return metric // 1000 if metric is not None else None

    def _convert_collection_to_str(field: str) -> Optional[str]:
        def safe_str(item: Any) -> str:
            try:
                return str(item)
            except Exception:
                return str(None)

        metric = metrics.get(field, None)
        if metric is None:
            return None

        # Remove this field (list/set) from metrics to avoid clashes
        del metrics[field]
        if not isinstance(metric, set) and not isinstance(metric, list):
            return None
        return ",".join(safe_str(item) for item in metric)

    structured_logging_overhead_s = torch._logging.get_structured_logging_overhead()

    if torch._inductor.utils.should_use_remote_fx_graph_cache():
        try:
            from torch._inductor.fb.remote_cache import (
                FbRemoteFxGraphCache,
                REMOTE_CACHE_VERSION,
            )

            remote_cache_version = REMOTE_CACHE_VERSION
            backend = FbRemoteFxGraphCache.get_remote_backend()
            inductor_fx_remote_cache_backend_type = type(backend).__name__
        except ModuleNotFoundError:
            remote_cache_version = None
            inductor_fx_remote_cache_backend_type = None
    else:
        inductor_fx_remote_cache_backend_type = None
        remote_cache_version = None

    # Populate the compile_id from the metrics context if it's set. Otherwise
    # look for it in the compile context.
    compile_id = metrics.get("compile_id")
    if not compile_id:
        compile_id = torch._guards.CompileContext.current_compile_id()

    common_metrics = {
        "compile_id": str(compile_id) if compile_id else None,
        "start_time_us": start_time_ns // 1000,
        "end_time_us": end_time_ns // 1000,
        "duration_us": (end_time_ns - start_time_ns) // 1000,
        "fail_type": exc_type.__qualname__ if exc_type else None,
        "fail_reason": str(exc_value) if exc_value else None,
        "structured_logging_overhead_us": to_int_us(structured_logging_overhead_s),
        "inductor_config": _scrubbed_inductor_config_for_logging(),
        "cuda_version": torch.version.cuda,
        "triton_version": triton.__version__ if has_triton() else "",
        "inductor_fx_remote_cache_hit_keys": _convert_collection_to_str(
            "inductor_fx_remote_cache_hit_keys"
        ),
        "inductor_fx_remote_cache_miss_keys": _convert_collection_to_str(
            "inductor_fx_remote_cache_miss_keys"
        ),
        "remote_cache_version": remote_cache_version,
        "inductor_fx_remote_cache_backend_type": inductor_fx_remote_cache_backend_type,
    }

    # TODO: The following are legacy fields, populated from the fields that replace
    # them. Remove these when we decide we can really deprecate them.
    legacy_metrics = {
        "start_time": start_time_ns / 1e9,
        "entire_frame_compile_time_s": us_to_s("dynamo_cumulative_compile_time_us"),
        "backend_compile_time_s": us_to_s("aot_autograd_cumulative_compile_time_us"),
        "inductor_compile_time_s": us_to_s("inductor_cumulative_compile_time_us"),
        "code_gen_time_s": us_to_s("inductor_code_gen_cumulative_compile_time_us"),
        "remote_cache_time_saved_s": us_to_s("distributed_ephemeral_timeout_us"),
        "remote_fx_graph_cache_get_time_ms": us_to_ms(
            "remote_fx_graph_cache_get_time_us"
        ),
        "remote_fx_graph_cache_put_time_ms": us_to_ms(
            "remote_fx_graph_cache_put_time_us"
        ),
        "structured_logging_overhead_s": structured_logging_overhead_s,
    }

    compilation_metrics = CompilationMetrics(
        **{**legacy_metrics, **common_metrics, **metrics}
    )
    _compilation_metrics.append(compilation_metrics)

    name = "compilation_metrics"
    if compilation_metrics.is_forward is False:
        name = "bwd_" + name
    if compilation_metrics.is_runtime is True:
        name = name + "_runtime"

    torch._logging.trace_structured(
        name,
        lambda: {
            k: list(v) if isinstance(v, set) else v
            for k, v in dataclasses.asdict(compilation_metrics).items()
        },
        # NB: Because compilation metrics *includes* the logging overhead time,
        # we can't both *measure* the logging overhead of compilation metrics
        # without making it inconsistent with compilation metrics itself, so
        # we ignore the (hopefully small) time spent logging compilation metrics
        record_logging_overhead=False,
        # These may be runtime logs, e.g., runtime autotunning, so we provide
        # the CompileId from the compilation metrics in case it's not available
        # in the current trace.
        compile_id=compile_id,
    )

    # If there's a chromium event in flight, add the CompilationMetrics to it.
    add_compilation_metrics_to_chromium(compilation_metrics)

    # Finally log the compilation metrics.
    if config.log_compilation_metrics:
        log_compilation_event(compilation_metrics)


# record_compilation_metrics is called by the singleton MetricsContext exit handler.
_METRICS_CONTEXT = MetricsContext(on_exit=record_compilation_metrics)
_RUNTIME_METRICS_CONTEXT = RuntimeMetricsContext(on_exit=record_compilation_metrics)


def set_compilation_metrics_limit(new_size: int) -> None:
    global _compilation_metrics
    while len(_compilation_metrics) > new_size:
        _compilation_metrics.popleft()
    new_deque = collections.deque(_compilation_metrics, maxlen=new_size)
    _compilation_metrics = new_deque


def clear_compilation_metrics() -> None:
    global _compilation_metrics
    _compilation_metrics.clear()


def get_compilation_metrics() -> List[CompilationMetrics]:
    return list(_compilation_metrics)


class ChromiumEventLogger:
    """Logs chromium events to structured logs. tlparse will concatenate these into a perfetto UI link.

    See https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview#heading=h.yr4qxyxotyw for
    a specification of the Chromium Event JSON format.
    """

    def get_stack(self) -> List[str]:
        """
        The main event stack, with every chromium event.
        Logged to tlparse.
        """
        if hasattr(self.tls, "stack"):
            return self.tls.stack
        else:
            self.tls.stack = []
            return self.tls.stack

    def get_top(self) -> Optional[str]:
        """
        Get the top event name or None if the stack is empty.
        """
        stack = self.get_stack()
        return stack[-1] if stack else None

    def get_pt2_compile_substack(self):
        """
        A smaller subset of the main stack that gets used to log
        PT2 Compile Events internally.
        """
        if hasattr(self.tls, "pt2_compile_substack"):
            return self.tls.pt2_compile_substack
        else:
            self.tls.pt2_compile_substack = []
            return self.tls.pt2_compile_substack

    def get_event_data(self) -> Dict[str, Any]:
        if not hasattr(self.tls, "event_data"):
            self.tls.event_data = {}
        return self.tls.event_data

    def __init__(self):
        self.tls = threading.local()
        # Generate a unique id for this logger, which we can use in scuba to filter down
        # to a single python run.
        self.id_ = str(uuid.uuid4())

        # TODO: log to init/id tlparse after I add support for it
        log.info("ChromiumEventLogger initialized with id %s", self.id_)

    def try_add_event_data(self, event_name: str, **kwargs) -> None:
        """
        Same as add_event_data, but will silently not log if the event isn't in the stack.
        """
        if event_name not in self.get_stack():
            return
        self.add_event_data(event_name, **kwargs)

    def add_event_data(
        self,
        event_name: str,
        **kwargs,
    ) -> None:
        """
        Adds additional metadata info to an in-progress event
        This metadata is recorded in the END event
        """
        if event_name not in self.get_stack():
            raise RuntimeError(
                f"Event {repr(event_name)} not in {self.get_stack()}. "
                "Cannot add metadata to events that aren't in progress. "
                "Please make sure the event has started and hasn't ended."
            )
        event_data = self.get_event_data()
        if event_name not in event_data:
            event_data[event_name] = {}
        event_data[event_name].update(kwargs)

    def log_event_start(
        self,
        event_name: str,
        time_ns: int,
        metadata: Dict[str, Any],
        log_pt2_compile_event: bool = False,
        compile_id: Optional[CompileId] = None,
    ) -> None:
        """
        Logs the start of a single event.
        :param str event_name Name of event to appear in trace
        :param time_ns Timestamp in nanoseconds
        :param metadata: Any extra metadata associated with this event
        :param log_pt_compile_event: If True, log to pt2_compile_events
        :param compile_id: Explicit compile_id (rather than using the current context)
        """
        compile_id = compile_id or torch._guards.CompileContext.current_compile_id()
        metadata["compile_id"] = str(compile_id)
        self._log_timed_event(
            event_name,
            time_ns,
            "B",
            metadata,
        )
        self.get_stack().append(event_name)
        # Add metadata from start event
        self.add_event_data(event_name, **metadata)
        if log_pt2_compile_event:
            self.get_pt2_compile_substack().append(event_name)

    def reset(self) -> None:
        # We this on every compile in case a compile crashes or restarts and we haven't
        # cleared the stack.
        stack = self.get_stack()
        substack = self.get_pt2_compile_substack()
        stack.clear()
        substack.clear()
        event_data = self.get_event_data()
        event_data.clear()

    def log_event_end(
        self,
        event_name: str,
        time_ns: int,
        metadata: Dict[str, Any],
        start_time_ns: int,
        log_pt2_compile_event: bool,
        compile_id: Optional[CompileId] = None,
    ) -> None:
        """
        Logs the end of a single event. This function should only be
        called after log_event_start with the same event_name.
        :param event_name: Name of event to appear in trace
        :param time_ns: Timestamp in nanoseconds
        :param metadata: Any extra metadata associated with this event
        :param start_time_ns: The start time timestamp in nanoseconds
        :param log_pt_compile_event: If True, log to pt2_compile_events
        :param compile_id: Explicit compile_id (rather than using the current context)
        """
        compile_id = compile_id or torch._guards.CompileContext.current_compile_id()
        metadata["compile_id"] = str(compile_id)

        # Grab metadata collected during event span
        all_event_data = self.get_event_data()
        if event_name in all_event_data:
            event_metadata = all_event_data[event_name]
            del all_event_data[event_name]
        else:
            event_metadata = {}
        # Add the passed in metadata
        event_metadata.update(metadata)

        event = self._log_timed_event(
            event_name,
            time_ns,
            "E",
            event_metadata,
        )

        def pop_stack(stack):
            while event_name != stack[-1]:
                # If the event isn't the most recent one to end, pop
                # off the stack until it is.
                # Since event_name in self.stack, this pop is always safe
                log.warning(
                    "ChromiumEventLogger: Detected overlapping events, fixing stack"
                )
                stack.pop()

        event_stack = self.get_stack()
        # These stack health checks currently never happen,
        # but they're written this way to future proof any weird event
        # overlaps in the future.
        if event_name not in event_stack:
            # Something went wrong, we never called start on this event,
            # or it was skipped due to overlapping events below
            log.warning("ChromiumEventLogger: Start event not in stack, ignoring")
            return

        pop_stack(event_stack)

        if log_pt2_compile_event:
            pt2_compile_substack = self.get_pt2_compile_substack()
            pop_stack(pt2_compile_substack)
            log_chromium_event_internal(
                event, pt2_compile_substack, self.id_, start_time_ns
            )
            # Pop actual event off of stack
            pt2_compile_substack.pop()

        # Finally pop the actual event off the stack
        event_stack.pop()

    def _log_timed_event(
        self,
        event_name: str,
        time_ns: int,
        phase: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Logs a timed event in chromium format. See log_event_start, log_event_end, etc.
        """
        event = {
            "name": event_name,
            "ts": time_ns / 1000,  # Chromium events are in micro seconds
            "args": metadata,
            "ph": phase,
            # These categories are needed in all chromium traces
            "cat": "dynamo_timed",
            "tid": 0,
            "pid": 0,  # pid should be specified on all logs, we don't personally care about the actual process id
        }
        torch._logging.trace_structured(
            "chromium_event",
            payload_fn=lambda: event,
            suppress_context=False,
            expect_trace_id=False,  # Not every chromium event will have a trace_id
        )
        record_chromium_event_internal(event)
        return event

    def log_instant_event(
        self,
        event_name: str,
        time_ns: int,
        metadata: Optional[Dict[str, Any]] = None,
        # By default, an instant event isn't logged internally, only to structured logging.
        log_pt2_compile_event: bool = False,
    ) -> None:
        """
        Log an instant event with no associated duration.
        :param str event_name: Name of event to appear in trace
        :param int time_ns Timestamp in nanoseconds
        :param Optional[Dict[str, Any]] metadata: Any extra metadata associated with this event
        :param str cname optional color for the arrow in the trace
        """
        if metadata is None:
            metadata = {}
        compile_id = str(torch._guards.CompileContext.current_compile_id())
        metadata["compile_id"] = compile_id
        event = {
            "name": event_name,
            "ts": time_ns / 1000,
            "args": metadata,
            "ph": "i",
            # These categories are needed in all chromium traces
            "cat": "dynamo_timed",
            "tid": 0,
            "pid": 0,
            "s": "p",  # We use "process" level instant events so they all appear on the same row in the trace.
        }
        torch._logging.trace_structured(
            "chromium_event",
            payload_fn=lambda: event,
            suppress_context=False,
            expect_trace_id=True,
        )
        if log_pt2_compile_event:
            # Log an instant event with the same start and end time
            log_chromium_event_internal(
                event, self.get_pt2_compile_substack(), self.id_, time_ns
            )


CHROMIUM_EVENT_LOG: Optional[ChromiumEventLogger] = None


def get_chromium_event_logger() -> ChromiumEventLogger:
    global CHROMIUM_EVENT_LOG
    if CHROMIUM_EVENT_LOG is None:
        CHROMIUM_EVENT_LOG = ChromiumEventLogger()
    return CHROMIUM_EVENT_LOG


@contextmanager
def chromium_event_timed(
    event_name: str,
    reset_event_log: bool = False,
    log_pt2_compile_event: bool = False,
) -> Generator[Any, None, None]:
    """
    Context manager that creates a chromium start and end event. Chromium event
    logging is integrated with dynamo_timed, so you probably want to use that
    instead. Use this context manager only if you want to avoid dynamo_timed.
    """
    chromium_event_log = get_chromium_event_logger()
    if reset_event_log:
        chromium_event_log.reset()
    chromium_start_time = time.time_ns()
    chromium_event_log.log_event_start(
        event_name,
        chromium_start_time,
        {},
        log_pt2_compile_event,
    )
    try:
        yield
    finally:
        chromium_event_log.log_event_end(
            event_name,
            time.time_ns(),
            {},
            chromium_start_time,
            log_pt2_compile_event,
        )


@dataclasses.dataclass
class CleanupHook:
    """Remove a global variable when hook is called"""

    scope: Dict[str, Any]
    name: str

    def __call__(self, *args):
        # Make sure we're not shutting down
        if CleanupManager is not None:
            CleanupManager.count -= 1
        del self.scope[self.name]

    @staticmethod
    def create(scope, name, val):
        assert name not in scope
        CleanupManager.count += 1
        scope[name] = val
        return CleanupHook(scope, name)


class CleanupManager(ExactWeakKeyDictionary):
    count = 0
    instance: ClassVar[CleanupManager]

    def _remove_id(self, idx):
        for hook in self.values[idx]:
            hook()
        super()._remove_id(idx)


CleanupManager.instance = CleanupManager()


def clone_tensor(x):
    """Clone the tensor and its gradient"""
    y = x.clone().requires_grad_(x.requires_grad)
    if x.is_leaf and x.grad is not None:
        y.grad = x.grad.clone()
    return y


def clone_input(x, *, dtype=None):
    """copy while preserving strides"""
    # TODO: this is questionable
    if is_fake(x):
        # this func fails on fake tensors in __torch_dispatch__
        return x

    def torch_clone(x):
        y = torch.clone(x)
        if x.is_leaf:
            y.requires_grad_(x.requires_grad)
        if x.is_leaf and x.grad is not None:
            y.grad = clone_input(x.grad, dtype=dtype)
        if hasattr(x, "_dynamo_dynamic_indices"):
            y._dynamo_dynamic_indices = x._dynamo_dynamic_indices.copy()  # type: ignore[attr-defined]
        return y

    with torch.no_grad():
        if x.device.type == "xla":
            # Access data_ptr() for a xla tensor will cause crash
            return torch_clone(x)

        # Handle sparse storage (no stride).
        if x.layout is torch.sparse_coo:
            return torch.sparse_coo_tensor(
                torch_clone(x._indices()),
                torch_clone(x._values()),
                x.shape,
                is_coalesced=x.is_coalesced(),
            )
        elif is_sparse_compressed(x):
            if x.layout in {torch.sparse_csr, torch.sparse_bsr}:
                compressed_indices = x.crow_indices()
                plain_indices = x.col_indices()
            else:
                compressed_indices = x.ccol_indices()
                plain_indices = x.row_indices()
            return torch.sparse_compressed_tensor(
                torch_clone(compressed_indices),
                torch_clone(plain_indices),
                torch_clone(x.values()),
                x.shape,
                layout=x.layout,
            )

        needed_size = sum(
            (shape - 1) * stride for shape, stride in zip(x.size(), x.stride())
        )
        if x.is_quantized:
            result = torch.empty_quantized((needed_size + 32,), x)
        else:
            result = torch.empty(
                needed_size + 32, dtype=dtype or x.dtype, device=x.device
            )
        cache_line_offset = (
            (x.data_ptr() - result.data_ptr()) % 32
        ) // x.element_size()
        result.as_strided_(x.size(), x.stride(), cache_line_offset)
        try:
            result.copy_(x.clone())
            if x.is_leaf:
                result.requires_grad_(x.requires_grad)
            if x.is_leaf and x.grad is not None:
                result.grad = clone_input(x.grad, dtype=dtype)
        except RuntimeError:
            # RuntimeError: unsupported operation: more than one element of the written-to
            # tensor refers to a single memory location. Please clone() the tensor before
            # performing the operation.
            return torch_clone(x)
        if hasattr(x, "_dynamo_dynamic_indices"):
            result._dynamo_dynamic_indices = x._dynamo_dynamic_indices.copy()  # type: ignore[attr-defined]
        return result


def clone_inputs(example_inputs):
    res: Union[Dict[Any, Any], List[Any]]
    if type(example_inputs) is dict:
        res = dict(example_inputs)
        for key, value in res.items():
            if isinstance(value, tuple):
                res[key] = clone_inputs(value)
            else:
                assert isinstance(value, torch.Tensor), type(value)
                res[key] = clone_input(value)
        return res

    res = list(example_inputs)
    for i in range(len(res)):
        if isinstance(res[i], torch.Tensor):
            res[i] = clone_input(res[i])
    return res


def skip_frame_if_in_functorch_mode(val: torch.Tensor):
    try:
        val.data_ptr()  # will throw for functorch tensors
    except RuntimeError as e:
        from .exc import SkipFrame

        # This will be GradTrackingTensor/BatchedTensor/etc
        functorch_subclass_name = re.sub(r"\(.*", "", repr(val))
        raise SkipFrame(
            f"torch.compile cannot be run in context: {functorch_subclass_name}"
        ) from e


@contextmanager
def preserve_rng_state():
    disable_functorch = torch._C._DisableFuncTorch
    disable_current_modes = torch.utils._python_dispatch._disable_current_modes
    with disable_current_modes(), disable_functorch():
        rng_state = torch.clone(torch.random.get_rng_state())
        skip_frame_if_in_functorch_mode(rng_state)
        if torch.cuda.is_available():
            cuda_rng_state = torch.clone(torch.cuda.get_rng_state())
    try:
        yield
    finally:
        with torch.utils._python_dispatch._disable_current_modes():
            torch.random.set_rng_state(rng_state)
            if torch.cuda.is_available():
                torch.cuda.set_rng_state(cuda_rng_state)  # type: ignore[possibly-undefined]


def is_jit_model(model0):
    return isinstance(
        model0,
        (
            torch.jit._trace.TopLevelTracedModule,
            torch.jit._script.RecursiveScriptModule,
            torch.jit.ScriptFunction,
            torch.jit.ScriptModule,
        ),
    )


def torchscript(model, example_inputs, verbose=False):
    if is_jit_model(model):
        # already done?
        return model

    try:
        return torch.jit.trace(model, example_inputs)
    except Exception:
        try:
            return torch.jit.script(model)
        except Exception:
            if verbose:
                log.exception("jit error")
            else:
                log.error("Both torch.jit.trace and torch.jit.script failed")
    return None


def getfile(obj):
    try:
        return inspect.getfile(obj)
    except (TypeError, OSError):
        return None


def is_namedtuple(obj):
    """Test if an object is a namedtuple or a torch.return_types.* quasi-namedtuple"""
    return is_namedtuple_cls(type(obj))


def is_namedtuple_cls(cls):
    """Test if an object is a namedtuple or a (torch.return_types|torch.autograd.forward_ad).* quasi-namedtuple"""
    try:
        if issubclass(cls, tuple):
            module = getattr(cls, "__module__", None)
            if module in ("torch.return_types", "torch.autograd.forward_ad"):
                return True
            if isinstance(getattr(cls, "_fields", None), tuple) and callable(
                getattr(cls, "_make", None)
            ):
                # The subclassing style namedtuple can have an extra base `typing.Generic`
                bases = tuple(t for t in cls.__bases__ if t is not Generic)
                if bases == (tuple,):
                    # This is a namedtuple type directly created by `collections.namedtuple(...)`
                    return True
                if bases and any(
                    (
                        # Subclass of namedtuple
                        is_namedtuple_cls(t)
                        # For subclasses of namedtuple, the __new__ method should not be customized
                        and cls.__new__ is t.__new__
                    )
                    for t in bases
                ):
                    return True
    except TypeError:
        pass
    return False


@functools.lru_cache(1)
def namedtuple_fields(cls) -> Tuple[str, ...]:
    """Get the fields of a namedtuple or a torch.return_types.* quasi-namedtuple"""
    if cls is slice:
        return ("start", "stop", "step")

    assert issubclass(cls, tuple)
    if hasattr(cls, "_fields"):
        # normal namedtuples
        return cls._fields

    @dataclasses.dataclass
    class Marker:
        index: int

    # frustrating ones e.g. torch.return_types.max
    assert cls.__module__ == "torch.return_types"
    obj = cls(map(Marker, range(cls.n_fields)))
    fields: Dict[str, int] = {}
    for name in dir(obj):
        if name[0] != "_" and isinstance(getattr(obj, name), Marker):
            fields[name] = getattr(obj, name).index
    assert len(fields) == cls.n_fields
    return tuple(sorted(fields, key=fields.get))  # type: ignore[arg-type]


def checkpoint_params(gm):
    with torch.no_grad():
        rng_state = torch.clone(torch.random.get_rng_state())
        if torch.cuda.is_available():
            cuda_rng_state = torch.clone(torch.cuda.get_rng_state())
        saved_state = [
            (param, param._version, torch.clone(param))
            for param in itertools.chain(gm.parameters(), gm.buffers())
        ]

    def restore():
        with torch.no_grad():
            torch.random.set_rng_state(rng_state)
            if torch.cuda.is_available():
                torch.cuda.set_rng_state(cuda_rng_state)
            for param, version, original_value in saved_state:
                if param._version != version:
                    param.copy_(original_value)

    return restore


def timed(model, example_inputs, times=1):
    if torch.cuda.is_available():
        synchronize = torch.cuda.synchronize
    else:
        synchronize = nothing

    synchronize()
    gc.collect()
    torch.manual_seed(1337)
    t0 = time.perf_counter()
    for _ in range(times):
        result = model(*example_inputs)
        synchronize()
    t1 = time.perf_counter()
    return result, t1 - t0  # type: ignore[possibly-undefined]


def check_is_cuda(gm, example_inputs):
    return all(x.is_cuda for x in itertools.chain(example_inputs, gm.parameters(True)))


@lru_cache(32)
def rot_n_helper(n):
    assert n > 1
    vars = [f"v{i}" for i in range(n)]
    rotated = reversed(vars[-1:] + vars[:-1])
    fn = eval(f"lambda {','.join(vars)}: ({','.join(rotated)})")
    fn.__name__ = f"rot_{n}_helper"
    return fn


common_constant_types: Set[type] = {
    int,
    float,
    complex,
    bool,
    str,
    bytes,
    type(None),
    Ellipsis.__class__,
    NotImplemented.__class__,
    types.CodeType,
    # Commonly used immutable types from torch.
    torch.device,
    torch.dtype,
    torch.memory_format,
    torch.layout,
    torch.finfo,
    torch.iinfo,
    torch.nn.attention.SDPBackend,
    torch.cuda._CudaDeviceProperties,
}

if has_triton_package():
    import triton

    common_constant_types.add(triton.language.dtype)

"""
    Difference between is_safe_constant and common_constant_types.
    * common_constant_types: Constants would be wrapped by VariableBuilder.wrap_literal
                             as ConstantVariable.
    * is_safe_constant: Constants can be loaded by LOAD_CONST bytecode.
"""


def is_safe_constant(v):
    if istype(v, (tuple, frozenset)):
        return all(map(is_safe_constant, v))
    return isinstance(v, (enum.Enum, type, torch.Size)) or istype(
        v,
        common_constant_types | {slice},
    )


def specialize_symnode(arg):
    from .variables import ConstantVariable, SymNodeVariable

    # Guard and specialize
    if isinstance(arg, SymNodeVariable):
        return ConstantVariable.create(arg.evaluate_expr())

    return arg


def guard_if_dyn(arg):
    from .variables import ConstantVariable

    arg = specialize_symnode(arg)

    if isinstance(arg, ConstantVariable):
        return arg.as_python_constant()

    return arg


def check_constant_args(args, kwargs):
    return all(x.is_python_constant() for x in itertools.chain(args, kwargs.values()))


def check_unspec_python_args(args, kwargs):
    from .variables.constant import ConstantVariable
    from .variables.tensor import UnspecializedPythonVariable

    unspec_count = 0
    for x in itertools.chain(args, kwargs.values()):
        if isinstance(x, UnspecializedPythonVariable):
            unspec_count += 1
        elif not isinstance(x, ConstantVariable):
            return False
    return unspec_count > 0


def check_unspec_or_constant_args(args, kwargs):
    # A fused version of:
    # return check_constant_args(args, kwargs) or check_unspec_python_args(args, kwargs)
    from .variables.tensor import UnspecializedPythonVariable

    for x in itertools.chain(args, kwargs.values()):
        if not (x.is_python_constant() or isinstance(x, UnspecializedPythonVariable)):
            return False
    return True


def check_numpy_ndarray_args(args, kwargs):
    from .variables.tensor import NumpyNdarrayVariable

    return any(
        isinstance(x, NumpyNdarrayVariable)
        for x in itertools.chain(args, kwargs.values())
    )


dict_keys: Type[KeysView[Any]] = type({}.keys())
dict_values: Type[ValuesView[Any]] = type({}.values())
odict_values: Type[ValuesView[Any]] = type(collections.OrderedDict().values())
tuple_iterator: Type[Iterator[Any]] = type(iter(()))
range_iterator: Type[Iterator[Any]] = type(iter(range(0)))
tuple_iterator_len = tuple_iterator.__length_hint__  # type: ignore[attr-defined]
object_new = object.__new__


def nn_module_new(cls):
    obj = object_new(cls)
    torch.nn.Module.__init__(obj)
    return obj


def product(it):
    return functools.reduce(operator.mul, it, 1)


def tuple_iterator_getitem(it, index):
    _, (obj,), start = it.__reduce__()
    return obj[start + index]


iter_next = next


def normalize_range_iter(range_iter) -> Tuple[int, int, int]:
    _, (range_obj,), maybe_idx = range_iter.__reduce__()
    # In 3.12+, `maybe_idx` could be None, and `range_obj.start` would've been
    # already incremented by the current index.
    start = range_obj.start + (maybe_idx or 0)
    stop = range_obj.stop
    step = range_obj.step
    return (start, stop, step)


def to_subclass(t, cls):
    return t.as_subclass(cls)


def dict_keys_getitem(d, n):
    return next(itertools.islice(iter(d), n, n + 1))


def enum_repr(value, local):
    # enum class can override __str__ method. Use __class__ and name attribute
    # to extract the class name and key name.
    name = value.__class__.__name__
    val = value.name
    scope = "L" if local else "G"
    local_name = f'{scope}["{name}"].{val}'
    return local_name


def set_example_value(node, example_value):
    # NB: example_value is a bit of a misnomer, because this is always a fake
    # tensor of some sort.  Furthermore, these example values serve as the
    # runtime state of Dynamo tracing, which means if metadata mutation
    # occurs, the example_value gets directly updated (so you can't rely on
    # this to accurately reflect what the state of the value was at the time
    # the program was traced).
    node.meta["example_value"] = example_value
    shape_env = TracingContext.get().fake_mode.shape_env
    if symbol_to_path := torch.fx.experimental.symbolic_shapes.compute_unbacked_bindings(
        shape_env, example_value
    ):
        node.meta["unbacked_bindings"] = symbol_to_path


def _get_fake_tensor(vt):
    fake_tensor = vt.as_proxy().node.meta.get("example_value")
    if not is_fake(fake_tensor):
        from .exc import unimplemented

        unimplemented("Cannot check Tensor object identity without its fake value")
    return fake_tensor


def iter_contains(items, search, tx, check_tensor_identity=False):
    from .variables import (
        BuiltinVariable,
        ConstantVariable,
        TensorVariable,
        VariableTracker,
    )

    if search.is_python_constant():
        found_const = any(
            x.is_python_constant()
            and x.as_python_constant() == search.as_python_constant()
            for x in items
        )
        return ConstantVariable.create(found_const)

    must_check_tensor_id = False
    if check_tensor_identity and isinstance(search, TensorVariable):
        must_check_tensor_id = True
        # Match of Tensor means match of FakeTensor
        search = _get_fake_tensor(search)

    found: Optional[VariableTracker] = None
    for x in items:
        if must_check_tensor_id:
            if isinstance(x, TensorVariable):
                if search is _get_fake_tensor(x):  # Object equivalence
                    return ConstantVariable.create(True)
        else:
            check = BuiltinVariable(operator.eq).call_function(tx, [x, search], {})
            if found is None:
                found = check
            else:
                found = BuiltinVariable(operator.or_).call_function(
                    tx, [check, found], {}
                )
    if found is None:
        found = ConstantVariable.create(False)
    return found


def key_is_id(k):
    """Returns whether it indexes dictionaries using its id"""
    return isinstance(k, (torch.Tensor, torch.nn.Module, MethodWrapperType))


def key_to_id(value):
    return [id(k) if key_is_id(k) else k for k in value.keys()]


def const_repr(x, *, local) -> str:
    from .trace_rules import is_builtin_callable

    if isinstance(x, (list, tuple)):
        elems_repr = ",".join(const_repr(s, local=local) for s in x)
        if isinstance(x, list):
            return f"[{elems_repr}]"
        else:
            assert isinstance(x, tuple)
            if len(x) == 1:
                return f"({elems_repr},)"
            else:
                return f"({elems_repr})"
    elif isinstance(x, enum.Enum):
        # To workaround repr(Enum) returning invalid global reference before python 3.11
        # by calling enum_repr and removing quotes to render enum in guard code.
        return enum_repr(x, local=local).replace("'", "")
    elif is_builtin_callable(x):
        return x.__name__
    elif isinstance(x, type):

        def fullname(o):
            klass = o.__class__
            module = klass.__module__
            if module == "builtins":
                return klass.__qualname__  # avoid outputs like 'builtins.str'
            return module + "." + klass.__qualname__

        return fullname(x)
    else:
        return f"{x!r}"


def dict_keys_repr(const_keys, *, local) -> str:
    keys_str = ",".join(const_repr(s, local=local) for s in const_keys)
    return "[" + keys_str + "]"


GLOBAL_KEY_PREFIX = "__dict_key"


from torch._subclasses import UnsupportedFakeTensorException  # noqa: F401


def get_safe_global_name(tx, root, obj):
    # The global_mangled_class_name should be different for different
    # invocations of torch.compile. Otherwise, we can run into a situation
    # where multiple torch.compile invocations re-use the same global name,
    # but the global's lifetime is tied to the first invocation (and
    # may be deleted when the first torch.compile invocation is deleted)
    # We mangle it based off of the output_graph's id.
    return f"{root}_{id(obj)}_c{tx.output.compile_id}"


def wrap_fake_exception(fn):
    try:
        return fn()
    except UnsupportedFakeTensorException as e:
        from .exc import unimplemented

        msg = f"Unsupported: {e.reason} with fake tensor propagation."
        log.warning(msg)
        unimplemented(msg, from_exc=e)


def deepcopy_to_fake_tensor(obj, fake_mode):
    with torch._subclasses.fake_tensor.FakeCopyMode(fake_mode):
        return wrap_fake_exception(lambda: copy.deepcopy(obj))


def rmse(ref, res):
    """
    Calculate root mean squared error
    """
    return torch.sqrt(torch.mean(torch.square(ref - res)))


def same(
    ref,
    res,
    fp64_ref=None,
    cos_similarity=False,
    tol=1e-4,
    equal_nan=False,
    exact_dtype=True,
    relax_numpy_equality=False,
    ignore_non_fp=False,
    log_error=log.error,
    use_larger_multiplier_for_smaller_tensor=False,
):
    """Check correctness to see if ref and res match"""
    if fp64_ref is None:
        fp64_ref = ref
    if isinstance(
        ref, (list, tuple, collections.deque, torch.nn.ParameterList, torch.Size)
    ):
        assert isinstance(
            res, (list, tuple, collections.deque)
        ), f"type mismatch {type(ref)} {type(res)}"
        if len(ref) != len(res):
            log_error("Length mismatch")
            return False
        return len(ref) == len(res) and all(
            same(
                ai,
                bi,
                fp64_refi,
                cos_similarity,
                tol,
                equal_nan,
                exact_dtype,
                relax_numpy_equality,
                ignore_non_fp,
                log_error=log_error,
                use_larger_multiplier_for_smaller_tensor=use_larger_multiplier_for_smaller_tensor,
            )
            for ai, bi, fp64_refi in zip(ref, res, fp64_ref)
        )
    elif type(ref).__name__ == "QuestionAnsweringModelOutput":
        # This skips checking accuracy for start_logits/end_logits.
        # Tentatively, start_logits/end_logits appear to be very prone to
        # inaccuracies and is somewhat subsumed by checking the loss.
        return same(
            ref.loss,
            res.loss,
            fp64_ref.loss,
            cos_similarity,
            tol,
            equal_nan,
            exact_dtype,
            relax_numpy_equality,
            ignore_non_fp,
            log_error=log_error,
            use_larger_multiplier_for_smaller_tensor=use_larger_multiplier_for_smaller_tensor,
        )
    elif isinstance(ref, dict):
        assert isinstance(res, dict)
        assert set(ref.keys()) == set(
            res.keys()
        ), f"keys mismatch {set(ref.keys())} == {set(res.keys())}"
        for k in sorted(ref.keys()):
            if not (
                same(
                    ref[k],
                    res[k],
                    fp64_ref[k],
                    cos_similarity=cos_similarity,
                    tol=tol,
                    equal_nan=equal_nan,
                    exact_dtype=exact_dtype,
                    relax_numpy_equality=relax_numpy_equality,
                    ignore_non_fp=ignore_non_fp,
                    log_error=log_error,
                    use_larger_multiplier_for_smaller_tensor=use_larger_multiplier_for_smaller_tensor,
                )
            ):
                log_error("Accuracy failed for key name %s", k)
                return False
        return True
    elif isinstance(ref, set):
        assert isinstance(res, set)
        assert set(ref) == set(res), f"elements mismatch {set(ref)} == {set(res)}"
        return True
    elif isinstance(ref, (torch.Tensor, float)):
        assert not isinstance(ref, torch._subclasses.FakeTensor)
        assert not isinstance(res, torch._subclasses.FakeTensor)

        def to_tensor(t):
            return t if isinstance(t, torch.Tensor) else torch.tensor(t)

        ref, res, fp64_ref = (to_tensor(val) for val in (ref, res, fp64_ref))

        if ref.is_sparse:
            assert res.is_sparse
            ref = ref.to_dense()
            res = res.to_dense()
        assert isinstance(res, torch.Tensor), f"type mismatch {type(ref)} {type(res)}"
        if exact_dtype:
            if ref.dtype != res.dtype:
                log_error("dtype mismatch %s, %s", ref.dtype, res.dtype)
                return False
            if ref.dtype == torch.bool:
                if ignore_non_fp:
                    return True
                # triton stores bool as int8, so add this for more accurate checking
                r = torch.allclose(
                    ref.to(dtype=torch.uint8),
                    res.to(dtype=torch.uint8),
                    atol=tol,
                    rtol=tol,
                    equal_nan=equal_nan,
                )
                if not r:
                    log_error("Accuracy failed: uint8 tensor did not match")
                return r

        if cos_similarity:
            ref = ref.flatten().to(torch.float32)
            res = res.flatten().to(torch.float32)
            if torch.allclose(ref, res, atol=tol, rtol=tol, equal_nan=True):
                # early exit that handles zero/nan better
                # cosine_similarity(zeros(10), zeros(10), dim=0) is 0
                return True
            score = torch.nn.functional.cosine_similarity(ref, res, dim=0, eps=1e-6)
            if score < 0.99:
                log.warning("Similarity score=%s", score.cpu().detach().item())
            return score >= 0.99
        else:
            if not exact_dtype:
                ref = ref.to(res.dtype)

            # First try usual allclose
            if torch.allclose(ref, res, atol=tol, rtol=tol, equal_nan=equal_nan):
                return True

            # Check error from fp64 version
            if fp64_ref.dtype == torch.float64:
                # Fix a corner case that res and fp64_ref does not contains NaN and match (with loose tolerance)
                # while the ref contains NaN. In this case, RMSE should not match any ways.
                # But res is 'BETTER' than ref so we count it pass.
                #
                # This happens for Super_SloMo when loop ordering after fusion is enabled:
                # https://gist.github.com/shunting314/11f235c70f7db0d52718d26f4a701cab
                loose_tol = 1e-2 * 4
                if (
                    not fp64_ref.isnan().any()
                    and not res.isnan().any()
                    and ref.isnan().any()
                    and torch.allclose(
                        fp64_ref.to(dtype=res.dtype),
                        res,
                        atol=loose_tol,
                        rtol=loose_tol,
                        equal_nan=equal_nan,
                    )
                ):
                    return True
                ref_error = rmse(fp64_ref, ref).item()
                # ref unable to produce this with stable numerics in this precision, ignore
                if math.isnan(ref_error):
                    log.warning(
                        "Found nan in reference. Consider running in higher precision."
                    )

                res_error = rmse(fp64_ref, res).item()

                # In the case of using AMP (Automatic Mixed Precision), certain models have
                # failed the benchmark's correctness check. However, the end-to-end model's
                # accuracy when comparing AMP with FP32 is within a difference of less than 0.1%.
                # Thus, it's possible that the correctness check failures for these models are
                # false alarms. We use multiplier of 3 instead of 2 to avoid these false alarms.
                multiplier = (
                    3.0 if res.dtype in (torch.float16, torch.bfloat16) else 2.0
                )

                if use_larger_multiplier_for_smaller_tensor and (
                    fp64_ref.numel() <= 10 and tol >= 4 * 1e-2
                ):
                    multiplier = 10.0
                elif use_larger_multiplier_for_smaller_tensor and (
                    fp64_ref.numel() <= 500 and tol >= 4 * 1e-2
                ):
                    multiplier = 5.0
                elif (
                    fp64_ref.numel() < 1000
                    or (ref.ndim == 4 and ref.shape[-1] == ref.shape[-2] == 1)
                    # large tol means a benchmark has been specified as REQUIRE_HIGHER_TOLERANCE
                    or tol >= 2 * 1e-2
                ):
                    # In the presence of noise, noise might dominate our error
                    # metric for smaller tensors.
                    # Similary, for 1x1 kernels, there seems to be high noise with amp.
                    multiplier = 3.0

                passes_test = res_error <= (multiplier * ref_error + tol / 10.0)
                if (
                    not passes_test
                    and equal_nan
                    and math.isnan(ref_error)
                    and math.isnan(res_error)
                    # Some unit test for the accuracy minifier relies on
                    # returning false in this case.
                    and not torch._inductor.config.cpp.inject_relu_bug_TESTING_ONLY
                ):
                    passes_test = True
                if not passes_test:
                    log_error(
                        "RMSE (res-fp64): %.5f, (ref-fp64): %.5f and shape=%s. res.dtype: %s, multiplier: %f, tol: %f"
                        ", use_larger_multiplier_for_smaller_tensor: %d",
                        res_error,
                        ref_error,
                        res.size(),
                        res.dtype,
                        multiplier,
                        tol,
                        use_larger_multiplier_for_smaller_tensor,
                    )
                return passes_test

            if ignore_non_fp:
                return True

            log_error("Accuracy failed: allclose not within tol=%s", tol)
            return False
    elif isinstance(ref, (str, int, type(None), bool, torch.device)):
        if ignore_non_fp:
            return True
        r = ref == res
        if not r:
            log_error("Accuracy failed (%s): %s != %s", type(ref), ref, res)
        return r
    elif is_numpy_int_type(ref) or is_numpy_float_type(ref):
        if relax_numpy_equality and not (
            is_numpy_int_type(res) or is_numpy_float_type(res)
        ):
            ref = ref.item()
        r = (type(ref) is type(res)) and (ref == res)
        if not r:
            log_error("Accuracy failed (numpy): %s != %s", ref, res)
        return r
    elif is_numpy_ndarray(ref):
        return (type(ref) is type(res)) and same(
            torch.as_tensor(ref),
            torch.as_tensor(res),
            fp64_ref,
            cos_similarity=cos_similarity,
            tol=tol,
            equal_nan=equal_nan,
            exact_dtype=exact_dtype,
            relax_numpy_equality=relax_numpy_equality,
            ignore_non_fp=ignore_non_fp,
            log_error=log_error,
            use_larger_multiplier_for_smaller_tensor=use_larger_multiplier_for_smaller_tensor,
        )
    elif type(ref).__name__ in (
        "MaskedLMOutput",
        "Seq2SeqLMOutput",
        "CausalLMOutputWithCrossAttentions",
        "LongformerMaskedLMOutput",
        "Instances",
        "SquashedNormal",
        "Boxes",
        "Normal",
        "TanhTransform",
        "Foo",
        "Variable",
    ):
        assert type(ref) is type(res)
        return all(
            same(
                getattr(ref, key),
                getattr(res, key),
                getattr(fp64_ref, key),
                cos_similarity=cos_similarity,
                tol=tol,
                equal_nan=equal_nan,
                exact_dtype=exact_dtype,
                relax_numpy_equality=relax_numpy_equality,
                ignore_non_fp=ignore_non_fp,
                log_error=log_error,
                use_larger_multiplier_for_smaller_tensor=use_larger_multiplier_for_smaller_tensor,
            )
            for key in ref.__dict__.keys()
        )
    else:
        raise RuntimeError(f"unsupported type: {type(ref).__name__}")


def format_func_info(code):
    short_filename = code.co_filename.split("/")[-1]
    return f"'{code.co_name}' ({short_filename}:{code.co_firstlineno})"


@contextlib.contextmanager
def disable_cache_limit():
    prior = config.cache_size_limit
    config.cache_size_limit = sys.maxsize
    prior_acc_limit = config.accumulated_cache_size_limit
    config.accumulated_cache_size_limit = sys.maxsize

    try:
        yield
    finally:
        config.cache_size_limit = prior
        config.accumulated_cache_size_limit = prior_acc_limit


# map from transformed code back to original user code
orig_code_map = ExactWeakKeyDictionary()

# keep a record of code_obj -> list of guard failure reasons for logging
guard_failures: DefaultDict[Any, List[Any]] = collections.defaultdict(list)

# Keep a record of graph break reasons for logging
graph_break_reasons: List[torch._dynamo.output_graph.GraphCompileReason] = []

# keep record of compiled code, if we are in "error if recompile"
# to track code that dynamo has compiled previously
seen_code_map = ExactWeakKeyDictionary()


# return same dir unless user changes config between calls
@functools.lru_cache(None)
def _get_debug_dir(root_dir):
    dir_name = (
        "run_"
        + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
        # use pid to avoid conflicts among ranks
        + "-pid_"
        + str(os.getpid())
    )
    return os.path.join(root_dir, dir_name)


def get_debug_dir():
    debug_root = config.debug_dir_root
    return _get_debug_dir(debug_root)


def extract_fake_example_value(node, required=True):
    if "example_value" in node.meta and is_fake(node.meta["example_value"]):
        return node.meta["example_value"]
    elif required:
        from torch._dynamo.exc import unimplemented

        unimplemented("`FakeTensor` example value was required but not available")
    else:
        return None


def ensure_graph_fake(e, tx):
    assert maybe_get_fake_mode(e) is tx.fake_mode
    return e


def get_fake_values_from_nodes(tx, nodes, allow_non_graph_fake):
    def visit(n: torch.fx.Node):
        if n.op == "call_function" and "example_value" not in n.meta:
            # fake tensor validity is checked inside get_fake_value using
            # ensure_graph_fake
            return get_fake_value(n, tx, allow_non_graph_fake)

        out = n.meta["example_value"]
        if not allow_non_graph_fake and isinstance(out, torch.Tensor):
            return ensure_graph_fake(out, tx)
        return out

    return torch.fx.node.map_arg(nodes, visit)


def get_fake_value(node, tx, allow_non_graph_fake=False):
    """
    Run the computation represented by `node` using fake tensors and return the result.

    allow_non_graph_fake: whether to allow the return result to be:
        1. non-fake or 2. fake that is not created by this instance of Dynamo.
        If `True`, you must be prepared to deal with such return values, ideally
        by further wrapping them as this graph's fakes.
    """
    from torch.utils._sympy.value_ranges import ValueRangeError

    from .exc import (
        TorchRuntimeError,
        unimplemented,
        Unsupported,
        UserError,
        UserErrorType,
    )

    op = node.op

    # FX Node should always return the same fake value
    if "example_value" in node.meta and is_fake(node.meta["example_value"]):
        return node.meta["example_value"]

    args, kwargs = get_fake_values_from_nodes(
        tx, (node.args, node.kwargs), allow_non_graph_fake
    )

    nnmodule = None
    if op == "call_method" and len(args) > 0 and isinstance(args[0], torch.nn.Module):
        # If the first argument is nn.Module, should copy to fake mode.
        args = (deepcopy_to_fake_tensor(args[0], tx.fake_mode),) + tuple(args[1:])

    if op == "call_module":
        nnmodule = tx.output.nn_modules[node.target]

        if is_lazy_module(nnmodule) and hasattr(nnmodule, "_initialize_hook"):
            # In the case of a lazy module, we want to run
            # the pre-hooks which initialize it.
            # Afterwards, lazy module deletes its pre-hooks
            # to avoid treating it as lazy on subsequent recompile.
            nnmodule._infer_parameters(nnmodule, args)

        # no matter it's lazy module or not, we should copy to fake mode.
        nnmodule = deepcopy_to_fake_tensor(nnmodule, tx.fake_mode)

    if node.name in ["interpolate", "is_integer", "wrapped_gradient"] or any(
        isinstance(a, complex) for a in args
    ):
        # We need to specialize symfloats for now. Eventually we should do a tensorify pass in dynamo.
        args = tuple(
            float(arg)
            if isinstance(arg, torch.SymFloat) and arg.node.hint is not None
            else arg
            for arg in args
        )

    try:
        with tx.fake_mode, enable_python_dispatcher():
            ret_val = wrap_fake_exception(
                lambda: run_node(tx.output, node, args, kwargs, nnmodule)
            )
    except Unsupported:
        raise
    except RuntimeError as e:
        cause: BaseException = e
        if e.__cause__ is not None:
            cause = e.__cause__

        if isinstance(
            cause, torch._subclasses.fake_tensor.DataDependentOutputException
        ):
            unimplemented(
                f"data dependent operator: {cause.func}; "
                "to enable, set torch._dynamo.config.capture_scalar_outputs = True"
            )
        elif isinstance(
            cause, torch._subclasses.fake_tensor.DynamicOutputShapeException
        ):
            if not torch._dynamo.config.capture_dynamic_output_shape_ops:
                unimplemented(
                    f"dynamic shape operator: {cause.func}; "
                    "to enable, set torch._dynamo.config.capture_dynamic_output_shape_ops = True"
                )
            else:
                unimplemented(
                    f"dynamic shape operator: {cause.func}; "
                    "Operator does not have a meta kernel that supports dynamic output shapes, "
                    "please report an issue to PyTorch"
                )
        elif isinstance(
            cause, torch._subclasses.fake_tensor.UnsupportedOperatorException
        ):
            op = cause.func
            import_suggestion = ""
            if isinstance(op, torch._ops.OpOverload):
                maybe_pystub = torch._C._dispatch_pystub(
                    op._schema.name, op._schema.overload_name
                )
                if maybe_pystub is not None:
                    module, ctx = maybe_pystub
                    import_suggestion = (
                        f"It's possible that the support was implemented in "
                        f"module `{module}` and you may need to `import {module}`"
                        f"({ctx}), otherwise "
                    )
            unimplemented(
                f"unsupported operator: {cause.func} ({import_suggestion}see "
                "https://docs.google.com/document/d/1GgvOe7C8_NVOMLOCwDaYV1mXXyHMXY7ExoewHqooxrs/edit#heading=h.64r4npvq0w0"
                " for how to fix)"
            )
        elif isinstance(
            cause, torch.fx.experimental.symbolic_shapes.GuardOnDataDependentSymNode
        ):
            raise UserError(  # noqa: B904
                UserErrorType.CONSTRAINT_VIOLATION,
                str(cause),
                case_name="constrain_as_size_example",
            )
        elif isinstance(cause, ValueRangeError):
            raise UserError(UserErrorType.CONSTRAINT_VIOLATION, e.args[0]) from e
        elif isinstance(cause, TypeError) and "argument" in str(cause):
            unimplemented(f"TypeError {node.target}: {cause}")

        raise TorchRuntimeError(str(e)).with_traceback(e.__traceback__) from None

    if not allow_non_graph_fake:
        _ = pytree.tree_map_only(
            torch.Tensor, functools.partial(ensure_graph_fake, tx=tx), ret_val
        )
    return ret_val


_current_node = threading.local()


def get_current_node():
    return getattr(_current_node, "value", None)


@contextmanager
def set_current_node(node):
    old = get_current_node()
    _current_node.value = node
    try:
        yield
    finally:
        _current_node.value = old


def run_node(tracer, node, args, kwargs, nnmodule):
    """
    Runs a given node, with the given args and kwargs.

    Behavior is dictated by a node's op.

    run_node is useful for extracting real values out of nodes.
    See get_real_value for more info on common usage.

    Note: The tracer arg is only used for 'get_attr' ops
    Note: The nnmodule arg is only used for 'call_module' ops

    Nodes that are not call_function, call_method, call_module, or get_attr will
    raise an AssertionError.
    """
    op = node.op

    with set_current_node(node):

        def make_error_message(e):
            return f"Failed running {op} {node.target}(*{args}, **{kwargs}):\n" + str(e)

        try:
            if op == "call_function":
                return node.target(*args, **kwargs)
            elif op == "call_method":
                return getattr(args[0], node.target)(*args[1:], **kwargs)
            elif op == "call_module":
                assert nnmodule is not None
                return nnmodule(*args, **kwargs)
            elif op == "get_attr":
                return tracer.output_graph.get_submodule(node.target)
            elif op == "placeholder":
                assert "example_value" in node.meta
                return node.meta["example_value"]

        except (NotImplementedError, UnsupportedFakeTensorException) as e:
            # NB: mimic how wrap_fake_exception does it
            from .exc import unimplemented

            unimplemented(make_error_message(e), from_exc=e)
        except Exception as e:
            raise RuntimeError(make_error_message(e)).with_traceback(
                e.__traceback__
            ) from e

    raise AssertionError(op)


def get_real_value(node, tracer):
    """
    Run the actual computation represented by `node` and return the result.
    This will execute any dependent nodes in the graph as well.
    """
    from .exc import TorchRuntimeError

    cache = tracer.real_value_cache
    if node in cache:
        return cache[node]

    op = node.op
    args, kwargs = torch.fx.node.map_arg(  # type: ignore[misc]
        (node.args, node.kwargs),
        lambda n: get_real_value(n, tracer),
    )

    if op == "placeholder" and "grapharg" in node.meta:
        return node.meta["grapharg"].example

    if op == "call_module":
        nn_module = tracer.output_graph.nn_modules[node.target]
        if not is_lazy_module(nn_module):
            nn_module = copy.deepcopy(nn_module)
        else:
            # In the case of a lazy module, we want to run
            # the pre-hooks which initialize it
            nn_module(*args, **kwargs)
    else:
        nn_module = None

    try:
        real_value = run_node(tracer, node, args, kwargs, nn_module)
        cache[node] = real_value
    except RuntimeError as e:
        raise TorchRuntimeError(str(e)).with_traceback(e.__traceback__) from None
    return real_value


def assert_no_fake_params_or_buffers(gm):
    from torch._subclasses.fake_tensor import FakeTensorConfig, is_fake

    def stack_or_hint(t):
        if FakeTensorConfig.debug:
            import traceback

            return f"FAKE TENSOR CREATION TRACEBACK: \n {traceback.format_list(t._debug_trace)}"
        else:
            return "Enable TORCH_FAKE_TENSOR_DEBUG=1 to get creation stack traces on fake tensors."

    for name, buffer in gm.named_buffers():
        assert not is_fake(
            buffer
        ), f"Unexpected fake buffer {name} {stack_or_hint(buffer)}"
    for name, param in gm.named_parameters():
        assert not is_fake(
            param
        ), f"Unexpected fake param {name} {stack_or_hint(param)}"


def fqn(obj: Any):
    """
    Returns the fully qualified name of the object.
    """
    return f"{obj.__module__}.{obj.__qualname__}"


def ifdynstaticdefault(count1, count2):
    if torch._dynamo.config.assume_static_by_default:
        return count1
    else:
        return count2


def import_submodule(mod: types.ModuleType):
    """
    Ensure all the files in a given submodule are imported
    """
    for filename in sorted(os.listdir(os.path.dirname(cast(str, mod.__file__)))):
        if filename.endswith(".py") and filename[0] != "_":
            importlib.import_module(f"{mod.__name__}.{filename[:-3]}")


def object_has_getattribute(value: Any):
    return class_has_getattribute(type(value))


def class_has_getattribute(cls: type):
    try:
        if isinstance(
            inspect.getattr_static(cls, "__getattribute__"),
            types.FunctionType,
        ):
            return True
    except AttributeError:
        pass
    return False


def get_custom_getattr(value: Any, ignore_nn_module_getattr: bool = False):
    try:
        getattr_fn = inspect.getattr_static(type(value), "__getattr__")
    except AttributeError:
        getattr_fn = None
    if ignore_nn_module_getattr and getattr_fn is torch.nn.Module.__getattr__:
        # ignore this case of getattr
        getattr_fn = None
    return getattr_fn


class TensorStaticReason(enum.Enum):
    PARAMETER = 2
    NOT_TENSOR = 4
    NN_MODULE_PROPERTY = 5


def tensor_static_reason_to_message(reason: TensorStaticReason):
    if reason == TensorStaticReason.PARAMETER:
        return "mark_dynamic on parameter, parameters are always static today."
    if reason == TensorStaticReason.NOT_TENSOR:
        return "mark_dynamic on a non tensor, how did this happen?"
    if reason == TensorStaticReason.NN_MODULE_PROPERTY:
        return "tensor is static because it is nn module associated."
    raise AssertionError(f"Illegal reason {reason}")


def tensor_always_has_static_shape(
    tensor: Union[torch.Tensor, Any],
    is_tensor: bool,
    tensor_source: Source,
) -> Tuple[bool, Optional[TensorStaticReason]]:
    """
    Given a tensor, source, and is_tensor flag, determine if a shape should be static.

    Args:
    tensor - the real tensor to evaluate, parameters force a static shape.
    is_tensor - internal dynamo check, essentially "is_tensor": target_cls is TensorVariable,
    tensors not in a TensorVariable for whatever reason are forced static.

    Returns a tuple, where the first element is the bool of whether or not this tensor should have a static shape.
    The second element is a TensorStaticReason, useful for passing to tensor_static_reason_to_message if needed.
    """
    from .source import is_from_unspecialized_param_buffer_source

    if (
        tensor_source.guard_source().is_specialized_nn_module()
        or tensor_source.guard_source().is_unspecialized_builtin_nn_module()
    ) and config.force_nn_module_property_static_shapes:
        return True, TensorStaticReason.NN_MODULE_PROPERTY

    if (
        type(tensor) is torch.nn.Parameter
        or is_from_unspecialized_param_buffer_source(tensor_source)
    ) and config.force_parameter_static_shapes:
        return True, TensorStaticReason.PARAMETER
    if not is_tensor:
        return True, TensorStaticReason.NOT_TENSOR
    return False, None


def lazy_format_graph_tabular(fn_name, gm):
    def inner():
        try:
            from tabulate import tabulate  # TODO: Check that this is installed
        except ImportError:
            return (
                "Tabulate module missing, please install tabulate to log the graph in tabular format, logging code instead:\n"
                + str(lazy_format_graph_code(fn_name, gm))
            )

        node_specs = [
            [n.op, n.name, n.target, n.args, n.kwargs] for n in gm.graph.nodes
        ]
        graph_str = tabulate(
            node_specs, headers=["opcode", "name", "target", "args", "kwargs"]
        )
        return _format_graph_code(fn_name, gm.forward.__code__.co_filename, graph_str)

    return LazyString(inner)


def format_bytecode(prefix, name, filename, line_no, code):
    return f"{prefix} {name} {filename} line {line_no} \n{dis.Bytecode(code).dis()}\n"


forward_hook_names = ["_forward_pre_hooks", "_forward_hooks"]
backward_hook_names = ["_backward_pre_hooks", "_backward_hooks"]
state_dict_hook_names = [
    "_state_dict_pre_hooks",
    "_state_dict_hooks",
    "_load_state_dict_pre_hooks",
    "_load_state_dict_post_hooks",
]
all_hook_names = forward_hook_names + backward_hook_names + state_dict_hook_names


def nn_module_has_global_hooks():
    # This is limited to backward hooks for now because NNModuleVariable
    # supports fwd hooks underneath.
    return len(torch.nn.modules.module._global_backward_hooks) or len(
        torch.nn.modules.module._global_backward_pre_hooks
    )


def nn_module_get_all_hooks(
    mod,
    check_forward_hooks=False,
    check_backward_hooks=False,
    check_state_dict_hooks=False,
):
    """
    Sometimes its useful to differentiate between types of hooks such as forward/backward/pre
    hooks executed during module.__call__, and state_dict hooks which are executed separately.
    """
    hook_dicts_to_check = []
    check_all_hooks = (
        not check_forward_hooks
        and not check_backward_hooks
        and not check_state_dict_hooks
    )
    if check_forward_hooks or check_all_hooks:
        hook_dicts_to_check.extend(forward_hook_names)
    if check_backward_hooks or check_all_hooks:
        hook_dicts_to_check.extend(backward_hook_names)
    if check_state_dict_hooks:
        hook_dicts_to_check.extend(state_dict_hook_names)

    all_hooks = []
    for hook_dict_name in hook_dicts_to_check:
        hooks = getattr(mod, hook_dict_name, [])
        for hook_name in hooks:
            hook = hooks[hook_name]

            all_hooks.append(hook)
    return all_hooks


def nnmodule_has_hooks(
    mod,
    check_forward_hooks=False,
    check_backward_hooks=False,
    check_state_dict_hooks=False,
):
    """
    Helper function to check if a module has any hooks attached to it.
    """
    hooks = nn_module_get_all_hooks(
        mod,
        check_forward_hooks=check_forward_hooks,
        check_backward_hooks=check_backward_hooks,
        check_state_dict_hooks=check_state_dict_hooks,
    )
    return bool(hooks)


def to_numpy_helper(value):
    """Convert tensor and tnp.ndarray to numpy.ndarray."""
    if is_fake(value):
        return value
    if isinstance(value, tnp.ndarray):
        return to_numpy_helper(value.tensor)
    elif isinstance(value, torch.Tensor):
        return value.numpy(force=True)
    elif isinstance(value, (tuple, list)):
        return type(value)(to_numpy_helper(obj) for obj in value)
    else:
        return value


def numpy_to_tensor(value):
    """Convert tnp.ndarray to tensor, leave other types intact. If a list/tuple, loop through it to convert."""
    assert np is not None
    if isinstance(value, np.ndarray):
        return torch.as_tensor(value)
    if isinstance(value, tnp.ndarray):
        return value.tensor
    elif isinstance(value, (tuple, list)):
        return type(value)(numpy_to_tensor(obj) for obj in value)
    else:
        return value


class numpy_to_tensor_wrapper:
    def __init__(self, f):
        self.f = f
        self.__name__ = "wrapped_" + self.f.__name__

    def __repr__(self) -> str:
        return f"<Wrapped function <original {self.f.__name__}>>"

    def __call__(self, *args, **kwargs):
        out = self.f(*args, **kwargs)
        return numpy_to_tensor(out)


def numpy_attr_wrapper(obj, name):
    if isinstance(obj, tnp.ndarray):
        out = getattr(obj, name)
        return numpy_to_tensor(out)
    elif isinstance(obj, torch.Tensor):
        out = getattr(tnp.ndarray(obj), name)
        return numpy_to_tensor(out)


class numpy_method_wrapper:
    """Convert obj from torch.Tensor to tnp.ndarray and call method. Then convert result back to torch.Tensor."""

    def __init__(self, method: str):
        self.method = method
        self.__name__ = "wrapped_" + self.method

    def __repr__(self) -> str:
        return f"<Wrapped method <original {self.method}>>"

    def __call__(self, *args, **kwargs):
        obj = args[0]
        if isinstance(obj, torch.Tensor):
            obj = tnp.ndarray(obj)
        method_callable = getattr(obj, self.method)
        out = method_callable(*args[1:], **kwargs)
        return numpy_to_tensor(out)


class numpy_operator_wrapper:
    """Implements dunder methods for tnp.ndarray via functions from the operator library"""

    def __init__(self, op: Callable[..., Any]):
        self.op = op
        self.__name__ = f"wrapped_{op.__name__}"

    def __repr__(self) -> str:
        return f"<Wrapped operator <original {self.__name__}>>"

    def __call__(self, *args, **kwargs):
        assert not kwargs

        args = (
            tnp.ndarray(arg) if isinstance(arg, torch.Tensor) else arg for arg in args
        )
        out = self.op(*args)
        return numpy_to_tensor(out)


def defake(x):
    if not isinstance(x, FakeTensor):
        return x
    size: torch._prims_common.ShapeType
    stride: torch._prims_common.StrideType
    if x._has_symbolic_sizes_strides:
        size = []
        for s in x.size():
            if isinstance(s, torch.SymInt):
                size.append(s.node.shape_env.size_hint(s.node.expr))
            else:
                size.append(s)
        stride = []
        for s in x.stride():
            if isinstance(s, torch.SymInt):
                stride.append(s.node.shape_env.size_hint(s.node.expr))
            else:
                stride.append(s)
    else:
        size = x.size()
        stride = x.stride()
    y = torch.empty_strided(
        size,
        stride,
        dtype=x.dtype,
        device=x.device,
        requires_grad=x.requires_grad,
    )
    y.zero_()
    return y


def is_utils_checkpoint(obj):
    # Lazy import to avoid circular dependencies
    import torch.utils.checkpoint

    return obj is torch.utils.checkpoint.checkpoint


def is_invoke_subgraph(obj):
    from torch._higher_order_ops.invoke_subgraph import invoke_subgraph_placeholder

    return obj is invoke_subgraph_placeholder


def build_invoke_subgraph_variable(**options):
    from .variables.higher_order_ops import TorchHigherOrderOperatorVariable

    return TorchHigherOrderOperatorVariable.make(
        torch._higher_order_ops.invoke_subgraph,
        **options,
    )


def build_checkpoint_variable(**options):
    import torch._higher_order_ops.wrap as higher_order_ops

    from .variables.higher_order_ops import TorchHigherOrderOperatorVariable

    # TODO - This is a temporary situation where we have two versions of
    # checkpointing implementation. We will converge on one and remove the other.
    activation_checkpoint_op: torch._ops.HigherOrderOperator = (
        higher_order_ops.tag_activation_checkpoint
    )
    if torch._functorch.config.functionalize_rng_ops:
        activation_checkpoint_op = higher_order_ops.wrap_activation_checkpoint

    return TorchHigherOrderOperatorVariable.make(
        activation_checkpoint_op,
        **options,
    )


def is_compile_supported(device_type):
    from .eval_frame import is_dynamo_supported

    compile_supported = is_dynamo_supported()
    if device_type == "cpu":
        pass
    elif device_type == "cuda" and compile_supported:
        compile_supported = has_triton()
    else:
        compile_supported = False
    return compile_supported


# The following 3.11 source code functions are adapted from
# https://github.com/python/cpython/blob/v3.11.4/Lib/traceback.py
# in order to output source code corresponding to bytecode in 3.11+.
# We need our own versions since we want to support multiline expressions.
def _fix_offset(str: str, offset: int) -> int:
    """
    Convert byte offset `offset` of `str` into character offset.
    Byte offset is used for 3.11+ instruction column data.
    Takes things like unicode characters into consideration.

    Unchanged from CPython implementation.
    """
    as_utf8 = str.encode("utf-8")
    return len(as_utf8[:offset].decode("utf-8", errors="replace"))


@dataclasses.dataclass
class _Anchors:
    # inclusive
    left_end_lineno: int
    left_end_offset: int
    right_start_lineno: int
    # exclusive
    right_start_offset: int


def _extract_anchors_from_expr(segment: str) -> Optional[_Anchors]:
    """
    Given source code `segment` corresponding to a bytecode
    instruction, determine:
        - for binary ops, the location of the binary op
        - for indexing, the location of the brackets.
    `segment` is expected to be a valid Python expression
    """
    assert sys.version_info >= (3, 11)

    import ast

    try:
        # Without brackets, `segment` is parsed as a statement.
        # We expect an expression, so wrap `segment` in
        # brackets to handle multi-line expressions.
        tree = ast.parse("(\n" + segment + "\n)")
    except SyntaxError:
        return None

    if len(tree.body) != 1:
        return None

    lines = segment.split("\n")

    # get character index given byte offset
    def normalize(lineno, offset):
        return _fix_offset(lines[lineno], offset)

    # Gets the next valid character index in `lines`, if
    # the current location is not valid. Handles empty lines.
    def next_valid_char(lineno, col):
        while lineno < len(lines) and col >= len(lines[lineno]):
            col = 0
            lineno += 1
        assert lineno < len(lines) and col < len(lines[lineno])
        return lineno, col

    # Get the next valid character index in `lines`.
    def increment(lineno, col):
        col += 1
        lineno, col = next_valid_char(lineno, col)
        assert lineno < len(lines) and col < len(lines[lineno])
        return lineno, col

    # Get the next valid character at least on the next line
    def nextline(lineno, col):
        col = 0
        lineno += 1
        lineno, col = next_valid_char(lineno, col)
        assert lineno < len(lines) and col < len(lines[lineno])
        return lineno, col

    statement = tree.body[0]
    if isinstance(statement, ast.Expr):
        expr = statement.value
        if isinstance(expr, ast.BinOp):
            # ast gives locations for BinOp subexpressions, e.g.
            # ( left_expr ) + ( right_expr )
            #   left^^^^^       right^^^^^
            # -2 since end_lineno is 1-indexed and because we added an extra
            # bracket to `segment` when calling ast.parse
            cur_lineno = cast(int, expr.left.end_lineno) - 2
            cur_col = normalize(cur_lineno, expr.left.end_col_offset)
            cur_lineno, cur_col = next_valid_char(cur_lineno, cur_col)

            # Heuristic to find the operator character.
            # The original CPython implementation did not look for ), \, or #,
            # leading to incorrect anchor location, e.g.
            # (x) + (y)
            # ~~^~~~~~~
            while (ch := lines[cur_lineno][cur_col]).isspace() or ch in ")\\#":
                if ch in "\\#":
                    cur_lineno, cur_col = nextline(cur_lineno, cur_col)
                else:
                    cur_lineno, cur_col = increment(cur_lineno, cur_col)

            # binary op is 1 or 2 characters long, on the same line
            right_col = cur_col + 1
            if (
                right_col < len(lines[cur_lineno])
                and not (ch := lines[cur_lineno][right_col]).isspace()
                and ch not in "\\#"
            ):
                right_col += 1
            # right_col can be invalid since it is exclusive

            return _Anchors(cur_lineno, cur_col, cur_lineno, right_col)
        elif isinstance(expr, ast.Subscript):
            # ast gives locations for value and slice subexpressions, e.g.
            # ( value_expr ) [ slice_expr ]
            #   value^^^^^     slice^^^^^
            # subscript^^^^^^^^^^^^^^^^^^^^
            # find left bracket (first '[' after value)
            left_lineno = cast(int, expr.value.end_lineno) - 2
            left_col = normalize(left_lineno, expr.value.end_col_offset)
            left_lineno, left_col = next_valid_char(left_lineno, left_col)
            while lines[left_lineno][left_col] != "[":
                left_lineno, left_col = increment(left_lineno, left_col)
            # find right bracket (final character of expression)
            right_lineno = cast(int, expr.end_lineno) - 2
            right_col = normalize(right_lineno, expr.end_col_offset)
            return _Anchors(left_lineno, left_col, right_lineno, right_col)
        elif isinstance(expr, ast.Call):
            # ( func_expr ) (args, kwargs)
            #   func^^^^^
            # call^^^^^^^^^^^^^^^^^^^^^^^^
            # find left bracket (first '(' after func)
            left_lineno = cast(int, expr.func.end_lineno) - 2
            left_col = normalize(left_lineno, expr.func.end_col_offset)
            left_lineno, left_col = next_valid_char(left_lineno, left_col)
            while lines[left_lineno][left_col] != "(":
                left_lineno, left_col = increment(left_lineno, left_col)
            # find right bracket (final character of expression)
            right_lineno = cast(int, expr.end_lineno) - 2
            right_col = normalize(right_lineno, expr.end_col_offset)
            return _Anchors(left_lineno, left_col, right_lineno, right_col)

    return None


def get_instruction_source_311(code: types.CodeType, inst: dis.Instruction) -> str:
    """
    Python 3.11+ only. Returns lines of source code (from code object `code`)
    corresponding to `inst`'s location data, and underlines relevant code to `inst`.

    Example: CALL on `g`:
    f(g(
      ^^
        h(x)))
        ^^^^^

    We need our own implementation in < 3.13 since `format_frame_summary` in
    Python's `traceback` module doesn't handle multi-line expressions
    (and their anchor extraction code is not completely correct).
    """
    if sys.version_info >= (3, 13):
        # multiline traceback implemented in 3.13+
        frame_summary = traceback.FrameSummary(
            code.co_filename,
            inst.positions.lineno,
            code.co_name,
            end_lineno=inst.positions.end_lineno,
            colno=inst.positions.col_offset,
            end_colno=inst.positions.end_col_offset,
        )
        result = traceback.format_list([frame_summary])[0]
        # remove first line containing filename info
        result = "\n".join(result.splitlines()[1:])
        # indent lines with original indentation
        orig_lines = [
            linecache.getline(code.co_filename, lineno).rstrip()
            for lineno in range(inst.positions.lineno, inst.positions.end_lineno + 1)
        ]
        orig_lines_dedent = textwrap.dedent("\n".join(orig_lines)).splitlines()
        indent_len = len(orig_lines[0]) - len(orig_lines_dedent[0])
        indent = orig_lines[0][:indent_len]
        result = textwrap.indent(textwrap.dedent(result), indent)
        return result

    assert inst.positions is not None
    if inst.positions.lineno is None:
        return ""
    # The rstrip + "\n" pattern is used throughout this function to handle
    # linecache.getline errors. Error lines are treated as empty strings "", but we want
    # to treat them as blank lines "\n".
    first_line = linecache.getline(code.co_filename, inst.positions.lineno).rstrip()
    if inst.positions.end_lineno is None:
        return first_line
    if inst.positions.col_offset is None or inst.positions.end_col_offset is None:
        return first_line

    # character index of the start of the instruction
    start_offset = _fix_offset(first_line, inst.positions.col_offset)
    # character index of the end of the instruction
    # compute later since end may be a different line
    end_offset = None
    # expression corresponding to the instruction so we can get anchors
    segment = ""
    # underline markers to be printed - start with `~` marker and replace with `^` later
    markers = []

    # Compute segment and initial markers
    if inst.positions.end_lineno == inst.positions.lineno:
        end_offset = _fix_offset(first_line, inst.positions.end_col_offset)
        segment = first_line[start_offset:end_offset]
        markers.append(" " * start_offset + "~" * (end_offset - start_offset))
    else:
        segment = first_line[start_offset:] + "\n"
        markers.append(" " * start_offset + "~" * (len(first_line) - start_offset))
        last_line = linecache.getline(
            code.co_filename, inst.positions.end_lineno
        ).rstrip()
        end_offset = _fix_offset(last_line, inst.positions.end_col_offset)
        for lineno in range(inst.positions.lineno + 1, inst.positions.end_lineno):
            line = linecache.getline(code.co_filename, lineno).rstrip()
            segment += line + "\n"
            # don't underline leading spaces
            num_spaces = len(line) - len(line.lstrip())
            markers.append(" " * num_spaces + "~" * (len(line) - num_spaces))
        segment += last_line[:end_offset]
        num_spaces = len(last_line) - len(last_line.lstrip())
        markers.append(" " * num_spaces + "~" * (end_offset - num_spaces))

    anchors: Optional[_Anchors] = None
    try:
        anchors = _extract_anchors_from_expr(segment)
    except AssertionError:
        pass

    # replace `~` markers with `^` where necessary
    if anchors is None:
        markers = [marker.replace("~", "^") for marker in markers]
    else:
        # make markers mutable
        mutable_markers: List[List[str]] = [list(marker) for marker in markers]

        # anchor positions do not take start_offset into account
        if anchors.left_end_lineno == 0:
            anchors.left_end_offset += start_offset
        if anchors.right_start_lineno == 0:
            anchors.right_start_offset += start_offset

        # Turn `~`` markers between anchors to `^`
        for lineno in range(len(markers)):
            for col in range(len(mutable_markers[lineno])):
                if lineno < anchors.left_end_lineno:
                    continue
                if lineno == anchors.left_end_lineno and col < anchors.left_end_offset:
                    continue
                if (
                    lineno == anchors.right_start_lineno
                    and col >= anchors.right_start_offset
                ):
                    continue
                if lineno > anchors.right_start_lineno:
                    continue
                if mutable_markers[lineno][col] == "~":
                    mutable_markers[lineno][col] = "^"

        # make markers into strings again
        markers = ["".join(marker) for marker in mutable_markers]

    result = ""
    for i in range(len(markers)):
        result += (
            linecache.getline(code.co_filename, inst.positions.lineno + i).rstrip()
            + "\n"
        )
        result += markers[i] + "\n"
    return result


def get_static_address_type(t):
    if isinstance(t, torch.Tensor):
        return getattr(t, "_dynamo_static_input_type", None)

    return None


def is_rng_state_getter_or_setter(value):
    getters = (
        # The following two functions are not identical, so don't remove anyone!
        torch._C.Generator.get_state,
        torch.default_generator.get_state,
        torch.get_rng_state,
        torch.cuda.get_rng_state,
    )
    setters = (
        torch._C.Generator.set_state,
        torch.default_generator.set_state,
        torch.set_rng_state,
        torch.cuda.set_rng_state,
    )
    return value in (*setters, *getters)


def is_tensor_base_attr_getter(value):
    return (
        isinstance(value, types.MethodWrapperType)
        and value.__name__ == "__get__"
        and value.__self__.__objclass__ is torch._C._TensorBase  # type: ignore[attr-defined]
    )


def is_torch_function_object(value):
    return hasattr(value, "__torch_function__")


def has_torch_function(vt: torch._dynamo.variables.base.VariableTracker) -> bool:
    from torch._dynamo.variables import UserDefinedObjectVariable
    from torch._dynamo.variables.torch_function import TensorWithTFOverrideVariable

    # Note on lazy vars: The value will either be realized or not throughout the course of execution
    # if the value has a torch function, it will eventually be realized so we can realize it here
    # if the value does not have a torch function, it may or may not be realized
    # if it is realized it will be used and guards will be installed properly
    # if it is not used, guards won't be installed, and it doesn't matter
    # if the value has a torch function or not, so we should *not* realize it.
    # NB: We technically know that if is_realized is False, LazyVariableTracker has the peek_value method
    # but mypy does not unfortunately
    if vt.is_realized() or (
        hasattr(vt, "peek_value") and hasattr(vt.peek_value(), "__torch_function__")
    ):
        if isinstance(vt, TensorWithTFOverrideVariable):
            return True

        return isinstance(vt, UserDefinedObjectVariable) and hasattr(
            vt.value, "__torch_function__"
        )

    return False


# see note [Tensor Fakification and Symbol Caching]
def to_fake_tensor(t, fake_mode):
    symbolic_context = None
    source = None
    if tracing_context := torch._guards.TracingContext.try_get():
        if t in tracing_context.tensor_to_context:
            symbolic_context = tracing_context.tensor_to_context[t]
            source = symbolic_context.tensor_source

    return fake_mode.from_tensor(
        t, static_shapes=False, symbolic_context=symbolic_context, source=source
    )


# NB: this works for both classes and instances
def is_frozen_dataclass(value):
    return (
        not object_has_getattribute(value)
        and not class_has_getattribute(value)
        and is_dataclass(value)
        and hasattr(value, "__dataclass_params__")
        and hasattr(value.__dataclass_params__, "frozen")
        and value.__dataclass_params__.frozen
    )


def get_first_attr(obj, *attrs):
    """
    Return the first available attribute or throw an exception if none is present.
    """
    for attr in attrs:
        if hasattr(obj, attr):
            return getattr(obj, attr)

    raise AssertionError(f"{obj} does not has any of the attributes: {attrs}")


@contextlib.contextmanager
def maybe_enable_compiled_autograd(should_enable, fullgraph=True, dynamic=True):
    if not should_enable:
        yield
    else:

        def compiler_fn(gm):
            def inner_compiler(gm_, example_inputs_):
                torch._dynamo.utils.counters["compiled_autograd"]["compiles"] += 1
                return torch._inductor.compile(gm_, example_inputs_)

            return torch.compile(
                gm, backend=inner_compiler, fullgraph=fullgraph, dynamic=dynamic
            )

        with torch._dynamo.compiled_autograd._enable(compiler_fn) as ctx:
            yield ctx


def invalid_removeable_handle():
    # need a subclass so weakref works
    class Invalid(dict):  # type: ignore[type-arg]
        pass

    return RemovableHandle(Invalid())


# Returns a "proxy" (new object with the same class and dict) for (non-GraphModule) nn.Module's.
# Attribute changes to the original object/proxy will be reflected in the other.
# This is useful for cases where we want a keep-alive reference to a module without increasing
# its reference count.
def nn_module_proxy(mod):
    if not isinstance(mod, torch.nn.Module):
        return mod
    if isinstance(mod, torch.fx.GraphModule):
        # Dynamo-generated GM's shouldn't contain user-created GM's
        return mod
    proxy = mod.__class__.__new__(mod.__class__)
    proxy.__dict__ = mod.__dict__
    return proxy


class GmWrapper(torch.nn.Module):
    def __init__(self, gm, unflatten_fn):
        super().__init__()
        self.gm = gm
        self.unflatten_fn = unflatten_fn

    def forward(self, *args):
        args: List[Any] = list(args)
        return self.gm(*self.unflatten_fn(args))


def flatten_graph_inputs(gm: torch.fx.GraphModule, inputs, compile_gm):
    """
    Mutate inputs so that they are flat and wrap gm such that it
    accepts those inputs.  This is needed for graphs that take
    bumpy inputs.
    """
    inputs_idx_to_clear = [
        i
        for i, node in enumerate(gm.graph.nodes)
        if node.op == "placeholder" and node.meta.get("steal_arg", False)
    ]

    if torch._dynamo.compiled_autograd.in_compiled_autograd_region:
        # fast path, avoid pytree overhead
        # compiled autograd inputs are always a list of tensors, maybe followed by symints
        assert inputs_idx_to_clear == [0]
        assert isinstance(inputs[0], list)
        boxed_inputs_count = len(inputs[0])

        def flatten_fn(args):
            return args[0] + list(args[1:])

        def unflatten_fn(flat_args):
            return (flat_args[:boxed_inputs_count], *flat_args[boxed_inputs_count:])

        compiled_fn = compile_gm(GmWrapper(gm, unflatten_fn), flatten_fn(inputs))
    else:
        # slow path, don't know inputs structure
        flat_inputs, spec = pytree.tree_flatten(inputs)
        unflatten_fn = functools.partial(pytree.tree_unflatten, treespec=spec)
        compiled_fn = compile_gm(GmWrapper(gm, unflatten_fn), flat_inputs)
        # note this doesn't check the spec, assuming it is the same
        flatten_fn = pytree.arg_tree_leaves

    def wrapper(*args):
        flat_args = flatten_fn(args)

        # flat_args is a new list, so we need to clear references from the old list
        for i in inputs_idx_to_clear:
            args[i].clear()

        # this call is boxed to avoid increasing refcount until we reach aot_module_simplified forward
        return compiled_fn(flat_args)

    return wrapper


def get_locals_to_steal(maybe_gm):
    if not isinstance(maybe_gm, torch.fx.GraphModule) or not hasattr(maybe_gm, "meta"):
        return []
    return maybe_gm.meta.get("locals_to_steal", [])


def set_locals_to_steal(gm, locals_to_steal):
    gm.meta["locals_to_steal"] = locals_to_steal


class Lit:
    def __init__(self, s):
        self.s = s

    def __repr__(self) -> str:
        return self.s


warn_once_cache: Set[str] = set()


def warn_once(msg, stacklevel=1):
    # Dynamo causes all warnings.warn (in user code and in Dynamo code) to print all the time.
    # https://github.com/pytorch/pytorch/issues/128427.
    # warn_once is a workaround: if the msg has been warned on before, then we will not
    # warn again.
    # NB: it's totally ok to store a cache of all the strings: this is what warnings.warn does as well.
    if msg in warn_once_cache:
        return
    warn_once_cache.add(msg)
    warnings.warn(msg, stacklevel=stacklevel + 1)


def strip_color_from_string(text):
    # This regular expression matches ANSI escape codes
    ansi_escape = re.compile(r"\x1B[@-_][0-?]*[ -/]*[@-~]")
    return ansi_escape.sub("", text)


@contextlib.contextmanager
def _disable_saved_tensors_hooks_during_tracing():
    # See NOTE: [Deferring tensor pack/unpack hooks until runtime]
    try:
        prior = torch._C._autograd._saved_tensors_hooks_set_tracing(True)
        yield
    finally:
        torch._C._autograd._saved_tensors_hooks_set_tracing(prior)


def is_parameter_freezing():
    return torch._inductor.config.freezing and not torch.is_grad_enabled()


def get_torch_function_mode_stack():
    return [
        get_torch_function_mode_stack_at(i) for i in range(_len_torch_function_stack())
    ]


def get_torch_function_mode_stack_at(ind):
    assert ind < _len_torch_function_stack() and ind >= 0
    return torch._C._get_function_stack_at(ind)


def set_torch_function_mode_stack(stack):
    for _ in range(_len_torch_function_stack()):
        _pop_torch_function_stack()

    for mode in stack:
        _push_on_torch_function_stack(mode)


def clear_torch_function_mode_stack():
    for _ in range(_len_torch_function_stack()):
        _pop_torch_function_stack()


# call from C dynamo in order to inspect values in pdb
def _breakpoint_for_c_dynamo(*args):
    breakpoint()


def verify_guard_fn_signature(value):
    fn = value.__metadata_guard__
    sig = inspect.signature(fn)
    if len(sig.parameters) != 2:
        from .exc import InternalTorchDynamoError

        raise InternalTorchDynamoError(
            "Tensor subclass method __metadata_guard__ must take exactly two subclass metadata arguments"
        )
    if fn.__self__ != value.__class__:
        from .exc import InternalTorchDynamoError

        raise InternalTorchDynamoError(
            "Tensor subclass method __metadata_guard__ must be a classmethod"
        )


def does_not_override_dict_iter_methods(user_cls):
    return (
        user_cls.items in (dict.items, collections.OrderedDict.items)
        and user_cls.values in (dict.values, collections.OrderedDict.values)
        and user_cls.keys in (dict.keys, collections.OrderedDict.keys)
        and user_cls.__iter__ in (dict.__iter__, collections.OrderedDict.__iter__)
    )


# Helper functions below are to prevent __torch_function__
# calls from happening in the middle of __torch_function__
# compiled bytecode
# They will be skipped which is the desired result
def call_size(x, i):
    @torch._dynamo.disable(recursive=True)
    def fn(x, i):
        return x.size(i)

    return fn(x, i)


def call_stride(x, i):
    @torch._dynamo.disable(recursive=True)
    def fn(x, i):
        return x.stride(i)

    return fn(x, i)


def call_storage_offset(x):
    @torch._dynamo.disable(recursive=True)
    def fn(x):
        return x.storage_offset()

    return fn(x)


# Helper function to extract relevant parts of a tensor's __dict__ to store in node meta.
# To avoid ref cycles, it's important that no tensors are present here, so leave those out.
def _extract_tensor_dict(t):
    KEYS_TO_COPY = [
        "_dynamo_static_input_type",
        "tag",
    ]

    tensor_dict = {
        key: copy.copy(t.__dict__[key]) for key in KEYS_TO_COPY if key in t.__dict__
    }

    return tensor_dict


# This is useful for reconstructing within the Dynamo graph the non-graph-input objects
# whose lifetime is governed by the user.
# e.g. torch.cuda.Event is a prime example.
user_obj_id_to_weakref: Dict[int, weakref.ReferenceType[object]] = {}


def get_user_object_from_id(obj_id):
    obj = user_obj_id_to_weakref[obj_id]()
    assert obj is not None, "User object is no longer alive"
    return obj


def store_user_object_weakref(obj):
    obj_id = id(obj)
    user_obj_id_to_weakref[obj_id] = weakref.ref(obj)


class CompileTimeInstructionCounter:
    _counter: int = 0
    _id: int = -1
    _depth = 0

    @classmethod
    def start(cls) -> None:
        cls._depth = cls._depth + 1
        if cls._depth == 1:
            cls._id = _instruction_counter.start()

    @classmethod
    def end(cls) -> None:
        cls._depth = cls._depth - 1
        if cls._depth == 0:
            cls._counter += _instruction_counter.end(cls._id)
            cls._id = -1

    @classmethod
    def clear(cls) -> None:
        cls._counter = 0

    @classmethod
    def value(cls) -> int:
        return cls._counter

    @classmethod
    @contextmanager
    def record(cls):
        try:
            if config.record_compile_time_instruction_count:
                cls.start()
            yield
        finally:
            if config.record_compile_time_instruction_count:
                cls.end()


def set_feature_use(feature: str, usage: bool):
    """
    Records whether we are using a feature
    Generally a feature is a JK.
    """
    # Note that sometimes (tests etc...) we're not in a context which we can record into
    if get_metrics_context().in_progress():
        get_metrics_context().set_key_value("feature_usage", feature, usage)
