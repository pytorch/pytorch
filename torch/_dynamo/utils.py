import atexit
import collections
import contextlib
import copy
import cProfile
import dataclasses
import datetime
import dis
import enum
import functools
import gc
import inspect
import itertools
import linecache
import logging
import math
import operator
import os
import pstats
import re
import subprocess
import sys
import textwrap
import threading
import time
import types
import typing
import weakref
from contextlib import contextmanager
from functools import lru_cache, wraps
from pathlib import Path
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
    Iterator,
    KeysView,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
    ValuesView,
)

from ..utils.hooks import RemovableHandle

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
        NP_SUPPORTED_MODULES = tuple()

        NP_TO_TNP_MODULE = {}
    from torch._subclasses.fake_tensor import FakeTensor, is_fake, maybe_get_fake_mode
except ImportError:
    pass

import importlib

import torch
import torch._functorch.config
import torch.fx.experimental.symbolic_shapes
from torch import fx
from torch._dispatch.python import enable_python_dispatcher
from torch._utils_internal import log_compilation_event

from torch.nn.modules.lazy import LazyModuleMixin
from torch.utils._pytree import tree_map_only
from torch.utils._triton import has_triton, has_triton_package


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

# profiling compilation time by frame phase
frame_phase_timing: Dict[str, Dict[str, float]] = {}

timer_counter = itertools.count()


def tabulate(rows, headers):
    try:
        import tabulate

        return tabulate.tabulate(rows, headers=headers)
    except ImportError:
        return "\n".join(
            ", ".join(map(str, row)) for row in itertools.chain([headers], rows)
        )


def maybe_cprofile(func):
    if config.cprofile:
        return cprofile_wrapper(func)
    return func


def cprofile_wrapper(func):
    @wraps(func)
    def profile_wrapper(*args, **kwargs):
        global timer_counter
        profile_cnt = next(timer_counter)
        profile_path = Path(func.__name__ + f"{profile_cnt}.profile")
        prof = cProfile.Profile()
        prof.enable()
        start_ts = time.time()
        retval = prof.runcall(func, *args, **kwargs)
        profile_latency = time.time() - start_ts
        prof.disable()
        print(
            f"### Cprofile for {func.__name__} iter {profile_cnt} took {profile_latency:.3f} seconds ###"
        )
        ps = pstats.Stats(prof)
        prof.dump_stats(profile_path)
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
            print(f"Generated SVG from profile at {str(svg_path)}")
        except FileNotFoundError:
            print(
                "Failed to generate SVG from profile -- dumping stats instead."
                "Try installing gprof2dot and dot for a better visualization"
            )
            ps.sort_stats(pstats.SortKey.TIME).print_stats(20)
            ps.sort_stats(pstats.SortKey.CUMULATIVE).print_stats(20)
        return retval

    return profile_wrapper


curr_frame = 0


# Note: Called for you by dynamo - you almost never ever want to invoke this yourself.
def increment_frame():
    global curr_frame
    curr_frame = curr_frame + 1


# Note: Called for you by dynamo - you almost never ever want to invoke this yourself.
def reset_frame_count():
    global curr_frame
    frame_phase_timing.clear()
    compilation_time_metrics.clear()
    curr_frame = 0


op_count = 0


def increment_op_count(cnt):
    global op_count
    op_count += cnt


# Print a report of time spent so far
# Ex:
# TIMING:
# entire_frame_compile:8.574629999999999
# backend_compile:5.26806
def print_time_report():
    total = 0.0
    total_by_key = {}
    for timings in frame_phase_timing.values():
        for key, timing in timings.items():
            total += timing
            if key not in total_by_key:
                total_by_key[key] = timing
            else:
                total_by_key[key] += timing

    out = "TIMING:"
    for key, value in total_by_key.items():
        out = f"{out} {key}:{round(value, 5)}"

    print(out)


# dynamo_timed API works as a function decorator
# By wrapping a function in dynamo_timed, we can store a record in compilation_time_metrics
# where the key is the functions name.
# For example:
#
#  @dynamo_timed
#  def _foo(...):
#
# Would show up as an entry in our timing dict:
# OrderedDict([('bar.<locals>._foo', [0.083690, 0.23949, 3.1425e-05])])
# This is extremely useful for granular debugging.
#
# For a higher-level mode, pass a phase_name into dynamo_timed
# phase_names record an extra record into a separate compilation timing structure,
# one keyed on frame+name rather than function.
# The frame is incremented outside of this function, in def increment_frame() above.


def dynamo_timed(original_function=None, phase_name=None):
    def dynamo_timed_inner(func):
        if config.cprofile:
            return func

        @wraps(func)
        def time_wrapper(*args, **kwargs):
            key = func.__qualname__
            if key not in compilation_time_metrics:
                compilation_time_metrics[key] = []
            with torch.profiler.record_function(f"{key} (dynamo_timed)"):
                t0 = time.time()
                r = func(*args, **kwargs)
                time_spent = time.time() - t0
            compilation_time_metrics[key].append(time_spent)
            if phase_name:
                frame_key = str(curr_frame)
                if frame_key not in frame_phase_timing:
                    frame_phase_timing[frame_key] = {}
                if phase_name not in frame_phase_timing[frame_key]:
                    frame_phase_timing[frame_key][phase_name] = time_spent
                else:
                    frame_phase_timing[frame_key][phase_name] += time_spent
            return r

        return time_wrapper

    if original_function:
        return dynamo_timed_inner(original_function)
    return dynamo_timed_inner


def compile_times(repr="str", aggregate=False):
    """
    Get metrics about torchdynamo frontend/backend compilation times.

    Accumulates information from functions tagged with `@dynamo_timed`.

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


@atexit.register
def dump_compile_times():
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
    def __init__(self, maxsize=4096):
        self.maxsize = maxsize
        self.reset()

    def reset(self):
        self.set = collections.OrderedDict()

    def add(self, key):
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
        torch._logging.set_logs(
            dynamo=logging.DEBUG,
            aot=logging.DEBUG,
            inductor=logging.DEBUG,
            output_code=True,  # this is off by default
        )
        return add_file_handler()

    return contextlib.ExitStack()


def reset_graph_break_dup_checker():
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


def gen_record_file_name(exc, code):
    return f"{get_debug_dir()}/error_recordings/\
{code.co_name}_{type(exc).__name__}_{code.co_firstlineno}.rec"


def write_record_to_file(filename, exec_record):
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


def count_calls(g: fx.Graph):
    c = 0
    for n in g.nodes:
        if "call" in n.op:
            c += 1
    return c


def identity(x):
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
        self.values = dict()
        self.refs = dict()

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


def istype(obj, allowed_types):
    """isinstance() without subclasses"""
    if isinstance(allowed_types, (tuple, list, set)):
        return type(obj) in allowed_types
    return type(obj) is allowed_types


def is_typing(value):
    # _Final catches most of typing classes:
    #   - Any
    #   - Callable
    #   - Union
    #   ...
    #
    # NB: we intentionally ignore classes that inherit from Generic, since they
    # can be used as both TypingVariable as well as UserDefinedClassVariable.
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


def is_function_or_wrapper(value):
    return (
        is_function(value)
        or isinstance(value, functools._lru_cache_wrapper)
        and is_function(inspect.getattr_static(value, "__wrapped__"))
        or isinstance(value, (torch._ops.OpOverloadPacket, torch._ops.OpOverload))
    )


def is_function(value):
    return isinstance(
        value,
        (
            types.FunctionType,
            types.BuiltinFunctionType,
            types.MethodDescriptorType,
            types.WrapperDescriptorType,
            torch.jit.ScriptFunction,
        ),
    )


def unwrap_if_wrapper(fn):
    return unwrap_with_attr_name_if_wrapper(fn)[0]


def unwrap_with_attr_name_if_wrapper(fn):
    # unpack @functools.lru_cache wrapped function
    if isinstance(fn, functools._lru_cache_wrapper):
        fn = inspect.getattr_static(fn, "__wrapped__")
        attr_name = "__wrapped__"
    # unpack @torch._dynamo.optimize()(fn) wrapped function
    elif is_function(fn) and inspect.getattr_static(fn, "_torchdynamo_inline", False):
        fn = inspect.getattr_static(fn, "_torchdynamo_inline", fn)
        attr_name = "_torchdynamo_inline"
    # unpack torch.jit.script_if_tracing
    elif is_function(fn) and inspect.getattr_static(
        fn, "__script_if_tracing_wrapper", False
    ):
        fn = inspect.getattr_static(fn, "__original_fn", fn)
        attr_name = "__original_fn"
    else:
        attr_name = None
    return fn, attr_name


def is_numpy_ndarray(value):
    if not np:
        return False

    return istype(value, np.ndarray)


def istensor(obj):
    """Check of obj is a tensor"""
    tensor_list = (
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


@dataclasses.dataclass
class CompilationMetrics:
    frame_key: str
    co_name: str
    co_filename: str
    co_firstlineno: int
    cache_size: int
    accumulated_cache_size: int
    guard_count: Optional[int]
    shape_env_guard_count: Optional[int]
    graph_op_count: Optional[int]
    graph_node_count: Optional[int]
    graph_input_count: Optional[int]
    start_time: float
    entire_frame_compile_time_s: Optional[float]
    backend_compile_time_s: Optional[float]
    inductor_compile_time_s: Optional[float]
    code_gen_time_s: Optional[float]
    fail_type: Optional[str]
    fail_reason: Optional[str]
    fail_user_frame_filename: Optional[str]
    fail_user_frame_lineno: Optional[int]
    non_compliant_ops: Set[str]
    compliant_custom_ops: Set[str]
    restart_reasons: Set[str]
    dynamo_time_before_restart_s: float


DEFAULT_COMPILATION_METRICS_LIMIT = 64


_compilation_metrics: Deque[CompilationMetrics] = collections.deque(
    maxlen=DEFAULT_COMPILATION_METRICS_LIMIT
)


def record_compilation_metrics(compilation_metrics: CompilationMetrics):
    global _compilation_metrics
    _compilation_metrics.append(compilation_metrics)
    if config.log_compilation_metrics:
        log_compilation_event(compilation_metrics)


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
    instance: ClassVar["CleanupManager"]

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
            bases = getattr(cls, "__bases__", []) or [None]
            module = getattr(cls, "__module__", None)
            return module in ("torch.return_types", "torch.autograd.forward_ad") or (
                bases[0] is tuple and hasattr(cls, "_make") and hasattr(cls, "_fields")
            )
    except TypeError:
        pass
    return False


@functools.lru_cache(1)
def namedtuple_fields(cls):
    """Get the fields of a namedtuple or a torch.return_types.* quasi-namedtuple"""
    if cls is slice:
        return ["start", "stop", "step"]

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
    fields: List[Optional[str]] = [None] * cls.n_fields
    for name in dir(obj):
        if name[0] != "_" and isinstance(getattr(obj, name), Marker):
            fields[getattr(obj, name).index] = name
    return fields


def checkpoint_params(gm):
    with torch.no_grad():
        rng_state = torch.clone(torch.random.get_rng_state())
        if torch.cuda.is_available():
            cuda_rng_state = torch.clone(torch.cuda.get_rng_state())
        saved_state = []
        for param in itertools.chain(gm.parameters(), gm.buffers()):
            saved_state.append((param, param._version, torch.clone(param)))

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


common_constant_types = {
    int,
    float,
    complex,
    bool,
    str,
    bytes,
    type(None),
    Ellipsis.__class__,
    types.CodeType,
    torch.device,
    torch.dtype,
    torch.memory_format,
    torch.layout,
}

if has_triton_package():
    import triton

    common_constant_types.add(triton.language.dtype)


def is_safe_constant(v):
    if istype(v, (tuple, frozenset)):
        return all(map(is_safe_constant, v))
    return isinstance(v, (enum.Enum, type)) or istype(
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


dict_keys: Type[KeysView[Any]] = type(dict().keys())
dict_values: Type[ValuesView[Any]] = type(dict().values())
odict_values: Type[ValuesView[Any]] = type(collections.OrderedDict().values())
tuple_iterator: Type[Iterator[Any]] = type(iter(tuple()))
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
):
    """Check correctness to see if ref and res match"""
    if fp64_ref is None:
        fp64_ref = ref
    if isinstance(ref, (list, tuple, torch.nn.ParameterList, torch.Size)):
        assert isinstance(res, (list, tuple)), f"type mismatch {type(ref)} {type(res)}"
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
            )
            for ai, bi, fp64_refi in zip(ref, res, fp64_ref)
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
                )
            ):
                log_error("Accuracy failed for key name %s", k)
                return False
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
                multiplier = 3.0 if res.dtype == torch.bfloat16 else 2.0

                if (
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
                if not passes_test:
                    log_error(
                        "RMSE (res-fp64): %.5f, (ref-fp64): %.5f and shape=%s. res.dtype: %s, multiplier: %f, tol: %f",
                        res_error,
                        ref_error,
                        res.size(),
                        res.dtype,
                        multiplier,
                        tol,
                    )
                    # import pdb; pdb.set_trace()
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
graph_break_reasons: List["torch._dynamo.output_graph.GraphCompileReason"] = list()

# keep record of compiled code, if we are in "error if recompile"
# to track code that dynamo has compiled previously
seen_code_map = ExactWeakKeyDictionary()


class CompileProfiler:
    """Utility for profiling how and what dynamo would compile.

    Can be used for
     * diagnosing recompilation issues
     * determining an appropriate compile cache limit
     * (TODO)confirming which functions got compiled/skipped
    """

    def __init__(self):
        self.frame_count = 0
        self.op_count = 0
        self.backend_ctx_ctor = disable_cache_limit

    def __call__(self, gm: torch.fx.GraphModule, example_inputs):
        self.frame_count += 1
        for node in gm.graph.nodes:
            if "call" in node.op:
                self.op_count += 1
        return gm.forward

    # no-op __enter__ and __exit__ to preserve BC
    def __enter__(self):
        return self

    def __exit__(self, typ, val, traceback):
        pass

    def get_metrics(self):
        return {"guard_failures": guard_failures}

    def report(self):
        metrics = self.get_metrics()
        gf = metrics["guard_failures"]

        def num_recompiles(code):
            return len(gf[code])

        def recompile_reasons(code):
            return "\n".join([str(x) for x in gf[code]])

        summarized_gf = [
            [format_func_info(code), num_recompiles(code), recompile_reasons(code)]
            for code in gf
        ]

        def graph_break_report():
            if "graph_break" in counters:
                graph_breaks = counters["graph_break"]
                return tabulate(
                    [[msg, graph_breaks[msg]] for msg in graph_breaks],
                    headers=["Graph Break Reason", "Count"],
                )

        def recompilation_report():
            if len(gf):
                max_recompiles = max([num_recompiles(code) for code in gf])
                recomp_table = tabulate(
                    summarized_gf,
                    headers=["Function", "Recompiles", "Recompile Reasons"],
                )
                return recomp_table + textwrap.dedent(
                    f"""

                    Set torch._dynamo.config.cache_size_limit to {max_recompiles} to avoid being cache limited.
                """
                )

        report = textwrap.dedent(
            """
            Torchdynamo Profiler Report
            ===========================

            Graph Breaks
            ------------
            Graph breaks happen when torchdynamo encounters code it can't safely trace.
            If you want to find out why breaks are happening, check below for each break reason
            You may gain additional insight by passing `fullgraph=True` to torch.compile,
            to stop at the first break.

        """
        )
        report += graph_break_report() or "No graph breaks detected."
        report += textwrap.dedent(
            """

            Recompilation
            -------------
            These subgraphs were recompiled more than once due to guard failures
            Guard failures indicate some condition assumed to be static by the tracer changed,
            making it unsafe to reuse the compiled program.

        """
        )
        report += recompilation_report() or "No recompilation detected.\n"
        return report


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
            raise UserError(  # noqa: TRY200
                UserErrorType.CONSTRAINT_VIOLATION,
                "Tried to use data-dependent value in the subsequent computation. "
                "This can happen when we encounter unbounded dynamic value that is unknown during tracing time.  "
                "You will need to explicitly give hint to the compiler. Please take a look at "
                f"constrain_as_value OR constrain_as_size APIs.  {cause}",
                case_name="constrain_as_size_example",
            )
        elif isinstance(cause, ValueRangeError):
            raise UserError(UserErrorType.CONSTRAINT_VIOLATION, e.args[0]) from e
        elif isinstance(cause, TypeError) and "argument" in str(cause):
            unimplemented(f"TypeError {node.target}: {cause}")

        raise TorchRuntimeError(str(e)).with_traceback(e.__traceback__) from None

    if not allow_non_graph_fake:
        _ = tree_map_only(
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
                return tracer.get_submodule(node.target)
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
    args, kwargs = torch.fx.node.map_arg(
        (node.args, node.kwargs),
        lambda n: get_real_value(n, tracer),
    )

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
    from torch._subclasses.fake_tensor import FakeTensorConfig

    def stack_or_hint(t):
        if FakeTensorConfig.debug:
            import traceback

            return f"FAKE TENSOR CREATION TRACEBACK: \n {traceback.format_list(t._debug_trace)}"
        else:
            return "Enable TORCH_FAKE_TENSOR_DEBUG=1 to get creation stack traces on fake tensors."

    for name, buffer in gm.named_buffers():
        assert not isinstance(
            buffer, torch._subclasses.FakeTensor
        ), f"Unexpected fake buffer {name} {stack_or_hint(buffer)}"
    for name, param in gm.named_parameters():
        assert not isinstance(
            param, torch._subclasses.FakeTensor
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
    try:
        if isinstance(
            inspect.getattr_static(type(value), "__getattribute__"),
            types.FunctionType,
        ):
            return True
    except AttributeError:
        pass
    return False


def get_custom_getattr(value: Any):
    try:
        getattr_fn = inspect.getattr_static(type(value), "__getattr__")
    except AttributeError:
        getattr_fn = None
    if getattr_fn is torch.nn.Module.__getattr__:
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
    guard_source: "torch._guards.GuardSource",
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
    if guard_source.is_nn_module() and config.force_nn_module_property_static_shapes:
        return True, TensorStaticReason.NN_MODULE_PROPERTY
    if type(tensor) is torch.nn.Parameter and config.force_parameter_static_shapes:
        return True, TensorStaticReason.PARAMETER
    if not is_tensor:
        return True, TensorStaticReason.NOT_TENSOR
    return False, None


def lazy_format_graph_code(name, gm, maybe_id=None):
    def format_name():
        if maybe_id is not None:
            return f"{name} {maybe_id}"
        else:
            return name

    return LazyString(
        lambda: _format_graph_code(
            f"===== {format_name()} =====\n",
            gm.forward.__code__.co_filename,
            gm.print_readable(print_output=False),
        )
    )


def _format_graph_code(name, filename, graph_str):
    return f"TRACED GRAPH\n {name} {filename} {graph_str}\n"


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


def nn_module_get_all_hooks(
    mod,
    check_forward_hooks=False,
    check_backward_hooks=False,
    check_state_dict_hooks=False,
):
    reset_code = torch._C._dynamo.eval_frame.reset_code
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

    def __repr__(self):
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

    def __repr__(self):
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

    def __repr__(self):
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
    size: "torch._prims_common.ShapeType"
    stride: "torch._prims_common.StrideType"
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


def build_checkpoint_variable(**options):
    import torch._higher_order_ops.wrap as higher_order_ops
    from .variables.higher_order_ops import TorchHigherOrderOperatorVariable

    # TODO - This is a temporary situation where we have two versions of
    # checkpointing implementation. We will converge on one and remove the other.
    activation_checkpoint_op: "torch._ops.HigherOrderOperator" = (
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

    We need our own implementation since `format_frame_summary` in
    Python's `traceback` module doesn't handle multi-line expressions
    (and their anchor extraction code is not completely correct).
    """
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


def has_torch_function(vt: "torch._dynamo.variables.base.VariableTracker") -> bool:
    from torch._dynamo.variables import UserDefinedObjectVariable
    from torch._dynamo.variables.torch_function import TensorWithTFOverrideVariable

    return isinstance(vt, TensorWithTFOverrideVariable) or (
        isinstance(vt, UserDefinedObjectVariable)
        and hasattr(vt.value, "__torch_function__")
    )


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


def get_first_attr(obj, *attrs):
    """
    Return the first available attribute or throw an exception if none is present.
    """
    for attr in attrs:
        if hasattr(obj, attr):
            return getattr(obj, attr)

    raise AssertionError(f"{obj} does not has any of the attributes: {attrs}")


@contextlib.contextmanager
def maybe_enable_compiled_autograd(should_enable):
    def compiler_fn(gm):
        def inner_compiler(gm_, example_inputs_):
            torch._dynamo.utils.counters["compiled_autograd"]["compiles"] += 1
            return torch._inductor.compile(gm_, example_inputs_)

        return torch.compile(gm, backend=inner_compiler, fullgraph=True, dynamic=True)

    if should_enable:
        with torch._dynamo.compiled_autograd.enable(compiler_fn) as ctx:
            yield ctx
    else:
        yield


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
