from __future__ import annotations

import collections
import contextlib
import dataclasses
import enum
import functools
import inspect
import io
import itertools
import logging
import math
import operator
import os
import platform
import shutil
import sys
import tempfile
import textwrap
import time
import unittest
from datetime import datetime
from io import StringIO
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Protocol,
    Set,
    TypeVar,
    Union,
    ValuesView,
)
from unittest import mock

import sympy
from typing_extensions import Concatenate, ParamSpec

import torch
from torch._dynamo.device_interface import get_interface_for_device
from torch._dynamo.utils import detect_fake_mode
from torch.autograd import DeviceType
from torch.autograd.profiler_util import EventList
from torch.fx.passes.shape_prop import ShapeProp
from torch.utils._sympy.functions import CeilDiv, CleanDiv, FloorDiv, ModularIndexing
from . import config
from .runtime.runtime_utils import ceildiv as runtime_ceildiv

log = logging.getLogger(__name__)

_T = TypeVar("_T")
VarRanges = Dict[sympy.Expr, sympy.Expr]


def do_bench_using_profiling(fn: Callable[[], Any], warmup=25, rep=100) -> float:
    """
    Returns benchmark results by examining torch profiler events.
    This could be more accurate as it doesn't count CPU side overhead.
    However, this also requires manually excluding irrelevant event, e.g.
    vectorized_elementwise_kernel which is used to fill L2 cache,
    various CUDA events, etc, so could also be fragile.
    """

    fn()
    torch.cuda.synchronize()
    cache = torch.empty(int(256e6 // 4), dtype=torch.int, device="cuda")

    # Estimate the runtime of the function
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(5):
        cache.zero_()
        fn()
    end_event.record()
    torch.cuda.synchronize()
    estimate_ms = start_event.elapsed_time(end_event) / 5

    # compute number of warmup and repeat
    n_warmup = max(1, int(warmup / estimate_ms))
    n_repeat = max(1, int(rep / estimate_ms))

    # Warm-up
    for _ in range(n_warmup):
        fn()

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as p:
        # Benchmark
        for i in range(n_repeat):
            # we clear the L2 cache before each run
            cache.zero_()
            # record time of `fn`
            fn()
        # Record clocks
        torch.cuda.synchronize()

    log.debug("raw events")
    log.debug(p.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))

    filtered_events = EventList(
        [
            event
            for event in p.events()
            if event.device_type == DeviceType.CUDA and event.name != "Context Sync"
        ]
    )
    if len(filtered_events) % n_repeat != 0:
        raise RuntimeError(
            "Failed to divide all profiling events into #repeat groups. "
            "#CUDA events: %d, #repeats: %s",
            len(filtered_events),
            n_repeat,
        )
    num_event_per_group = len(filtered_events) / n_repeat
    actual_events = EventList(
        [
            event
            for i, event in enumerate(filtered_events)
            if i % num_event_per_group != 0
        ]
    )
    actual_events._build_tree()
    actual_events = actual_events.key_averages()

    log.debug("profiling time breakdown")
    log.debug(actual_events.table(row_limit=-1))

    res = sum(event.device_time_total for event in actual_events) / 1000.0 / n_repeat
    log.debug("profiling results: %s ms", res)
    return res


@functools.lru_cache(None)
def has_torchvision_roi_align() -> bool:
    try:
        from torchvision.ops import roi_align  # noqa: F401

        return roi_align is not None and hasattr(
            getattr(torch.ops, "torchvision", None), "roi_align"
        )
    except ImportError:
        return False


def decode_device(device: Union[Optional[torch.device], str]) -> torch.device:
    if device is None:
        return torch.tensor(0.0).device  # default device
    if isinstance(device, str):
        device = torch.device(device)
    if device.type not in ("cpu", "meta") and device.index is None:
        device_interface = get_interface_for_device(device.type)
        return torch.device(device.type, index=device_interface.Worker.current_device())
    return device


def sympy_product(it):
    return functools.reduce(operator.mul, it, sympy.Integer(1))


def sympy_dot(seq1, seq2):
    assert len(seq1) == len(seq2)
    return sympy.expand(sum(a * b for a, b in zip(seq1, seq2)))


def unique(it: Iterable[_T]) -> ValuesView[_T]:
    return {id(x): x for x in it}.values()


def ceildiv(
    numer: Union[int, sympy.Expr], denom: Union[int, sympy.Expr]
) -> Union[int, sympy.Expr]:
    if isinstance(numer, sympy.Expr) or isinstance(denom, sympy.Expr):
        return CeilDiv(numer, denom)
    # TODO: There is a bug in a call to this function, to repro:
    # python benchmarks/dynamo/huggingface.py --inductor -d cuda --accuracy
    # --amp --only YituTechConvBert --dynamic-shapes
    assert isinstance(numer, int) and isinstance(
        denom, int
    ), f"{numer}: {type(numer)}, {denom}: {type(denom)}"
    return runtime_ceildiv(numer, denom)


def _type_of(key):
    # Use the function here to get rid of dependencies on the Triton during the codegen.
    # Refer to Triton implementation here:
    # https://github.com/openai/triton/blob/98b5945d2aef679e00ebca8e07c35c3658ec76de/python/triton/runtime/jit.py#L238
    # `None` is nullptr.  Implicitly convert to *i8.
    if key is None:
        return "*i8"
    dtype_str = str(key).split(".")[-1]
    tys = {
        "bool": "i1",
        "float8e4nv": "fp8e4nv",
        "float8e5": "fp8e5",
        "float8e4b15": "fp8e4b15",
        "float8e4b15x4": "fp8e4b15x4",
        "float8_e4m3fn": "fp8e4nv",
        "float8_e5m2": "fp8e5",
        "float16": "fp16",
        "bfloat16": "bf16",
        "float32": "fp32",
        "float64": "fp64",
        "int8": "i8",
        "int16": "i16",
        "int32": "i32",
        "int64": "i64",
        "uint8": "u8",
        "uint16": "u16",
        "uint32": "u32",
        "uint64": "u64",
    }
    # reinterpret can create triton type
    for v in list(tys.values()):
        tys[v] = v
    return key if isinstance(key, str) else f"*{tys[dtype_str]}"


def convert_shape_to_inductor(
    lst: Iterable[Union[int, torch.SymInt]]
) -> List[sympy.Expr]:
    """
    Gets the shape and stride of a tensor. For non-symbolic tensors, this is
    trivial. But for symbolic tensors, we need to map from SymIntNode into
    sympy.Expr.
    """
    return [
        i.node.expr if isinstance(i, torch.SymInt) else sympy.Integer(i) for i in lst
    ]


def convert_shape_to_symint(
    lst: Iterable[Union[int, sympy.Expr]]
) -> List[Union[int, torch.SymInt]]:
    """
    Takes a list of shapes from Inductor and converts them into symints (or just
    ints if all shapes are static).
    """
    from .virtualized import V

    return [
        i
        if isinstance(i, int)
        else int(i)
        if isinstance(i, sympy.Integer)
        else V.graph.sizevars.shape_env.create_symintnode(i, hint=None)
        for i in lst
    ]


def is_view(op: torch._ops.OpOverload):
    """
    Does this op overload have aliasing
    """
    assert isinstance(op, torch._ops.OpOverload)
    return any(a.alias_info is not None for a in op._schema.arguments)


def is_pointwise_use(use):
    if not use.op == "call_function":
        return False

    if not (
        isinstance(use.target, torch._ops.OpOverload) or use.target is operator.getitem
    ):
        return False

    if use.target is operator.getitem or is_view(use.target):
        return all(is_pointwise_use(u) for u in use.users)

    return torch.Tag.pointwise in use.target.tags


def gen_gm_and_inputs(target, args, kwargs):
    g = torch.fx.Graph()
    g_args = []
    a_args = []
    for n, arg in enumerate(args):
        if isinstance(arg, torch.Tensor):
            g_args.append(g.placeholder(f"arg{n}"))
            a_args.append(arg)
        else:
            g_args.append(arg)
    assert all(not isinstance(x, torch.Tensor) for x in kwargs.values())
    node = g.call_function(target, tuple(g_args), kwargs)
    if (
        len(target._schema.returns) == 1
        and str(target._schema.returns[0].type) == "Tensor"
    ):
        node = (node,)
    g.output(node)

    gm = torch.fx.GraphModule({}, g)
    return gm, a_args


def synchronize(device: str = "cuda"):
    if device == "cpu":
        return
    device_interface = get_interface_for_device(device)
    if device_interface.is_available():
        device_interface.synchronize()


def timed(
    model: Callable[..., Any], example_inputs, times: int = 1, device: str = "cuda"
) -> float:
    synchronize(device)
    torch.manual_seed(1337)
    t0 = time.perf_counter()
    for _ in range(times):
        result = model(*example_inputs)
        synchronize(device)
    t1 = time.perf_counter()
    # GC the result after timing
    assert result is not None  # type: ignore[possibly-undefined]
    return t1 - t0


def print_performance(
    fn, args=(), times=10, repeat=10, baseline=1.0, device: str = "cuda"
):
    timings = torch.tensor([timed(fn, args, times, device) for _ in range(repeat)])
    took = torch.median(timings) / times
    print(f"{took/baseline:.6f}")
    return took


def precompute_method(obj: Any, method: str):
    """Replace obj.method() with a new method that returns a precomputed constant."""
    result = getattr(obj, method)()
    setattr(obj, method, lambda: result)


def precompute_methods(obj: Any, methods: List[str]):
    """Replace methods with new methods that returns a precomputed constants."""
    for method in methods:
        precompute_method(obj, method)


def cmp(a, b) -> int:
    return int(a > b) - int(a < b)


def pad_listlike(x, size):
    if len(x) == 1:
        return type(x)([x[0]]) * size
    else:
        return x


# Used to ensure that iterating over a set is deterministic
def tuple_sorted(x):
    if len(x) == 0:
        return []

    def sort_func(elem):
        if isinstance(elem, str):
            return elem
        else:
            # We expect `elem` to be `scheduler.BaseSchedulerNode` type here,
            # but we are not able to do isinstance assert because of circular dependency
            return elem.get_name()

    return sorted(x, key=sort_func)


P = ParamSpec("P")
RV = TypeVar("RV", covariant=True)


class CachedMethod(Generic[P, RV], Protocol):
    @staticmethod
    def clear_cache(self) -> None:
        ...

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> RV:
        ...


# See https://github.com/python/mypy/issues/13222#issuecomment-1193073470 to understand the type signature
def cache_on_self(fn: Callable[Concatenate[Any, P], RV]) -> CachedMethod[P, RV]:
    key = f"__{fn.__name__}_cache"

    @functools.wraps(fn)
    def wrapper(self):
        if not hasattr(self, key):
            setattr(self, key, fn(self))
        return getattr(self, key)

    def clear_cache(self):
        if hasattr(self, key):
            delattr(self, key)

    wrapper.clear_cache = clear_cache  # type: ignore[attr-defined]
    return wrapper  # type: ignore[return-value]


def aggregate_origins(node_schedule):
    from . import ir

    if isinstance(node_schedule, list):
        return functools.reduce(
            operator.or_,
            [
                node.node.origins
                for node in node_schedule
                if hasattr(node, "node") and node.node
            ],
            set(),
        )
    elif isinstance(node_schedule, ir.ExternKernel):
        return node_schedule.origins
    else:
        return set()


def get_fused_kernel_name(node_schedule, descriptive_names):
    all_origins = aggregate_origins(node_schedule)
    if descriptive_names == "original_aten":
        # Bases the kernel name off of the top-level aten operator (i.e. pre-decompositions)
        sources = [
            origin.meta["original_aten"]._overloadpacket.__name__
            for origin in all_origins
            if origin.op == "call_function"
            and "original_aten" in origin.meta
            and origin.meta["original_aten"] is not None
        ]
        sources = sorted(set(sources))
    elif descriptive_names == "torch":
        # Bases the kernel name off of the top-level "torch" operator (i.e. post-dynamo graph)
        sources = []
        for origin in all_origins:
            if origin.op == "call_function" and "source_fn_stack" in origin.meta:
                source_fn = origin.meta["source_fn_stack"][-1]
                if isinstance(source_fn[1], str):
                    sources.append(source_fn[1])
                else:
                    sources.append(source_fn[1].__name__)
        sources = sorted(set(sources))
    elif descriptive_names == "inductor_node":
        sources = [
            origin.name for origin in all_origins if origin.op == "call_function"
        ]
    else:
        raise NotImplementedError
    sources = sources
    return "_".join(["fused"] + sources)


def get_kernel_metadata(node_schedule, wrapper):
    all_origins = aggregate_origins(node_schedule)
    inductor_nodes = [origin for origin in all_origins if origin.op == "call_function"]

    from_node_dict = collections.defaultdict(list)
    original_aten_dict = collections.defaultdict(list)
    for node in inductor_nodes:
        if "original_aten" in node.meta and node.meta["original_aten"] is not None:
            key = str(node.meta["original_aten"]._overloadpacket)
            original_aten_dict[key].append(node.name)
        if "from_node" in node.meta:
            key = node.meta["from_node"][0][0]
            from_node_dict[key].append(node.name)
    metadata = (
        f"{wrapper.comment} Source Nodes: [{', '.join(sorted(from_node_dict.keys()))}], "
        f"Original ATen: [{', '.join(sorted(original_aten_dict.keys()))}]"
    )
    # trace back to original node here
    detailed_metadata = []
    for original_node, nodes in sorted(from_node_dict.items()):
        detailed_metadata.append(
            f"{wrapper.comment} {original_node} => {', '.join(sorted(nodes))}"
        )
    return metadata, "\n".join(detailed_metadata)


def dominated_nodes(
    initial_queue: Iterable[torch.fx.Node], skip_filter=None
) -> Set[torch.fx.Node]:
    """Returns the set of nodes whose values depend on those within initial_queue"""
    initial_queue = list(initial_queue)
    dominated_set = set(initial_queue)

    while initial_queue:
        node = initial_queue.pop()
        for user in node.users:
            if skip_filter and skip_filter(user):
                continue
            if user not in dominated_set:
                dominated_set.add(user)
                initial_queue.append(user)

    return dominated_set


def gather_origins(args, kwargs):
    import itertools

    from . import ir

    def is_unrealized_node(n):
        if isinstance(n, ir.TensorBox):
            return is_unrealized_node(n.data)
        if isinstance(n, ir.StorageBox):
            return is_unrealized_node(n.data)
        return isinstance(n, ir.IRNode) and isinstance(n, ir.Pointwise)

    kwarg_origins = [val.origins for val in kwargs.values() if is_unrealized_node(val)]
    arg_origins = [arg.origins for arg in args if is_unrealized_node(arg)]
    return set(itertools.chain(*arg_origins, *kwarg_origins))


def sympy_str(expr: sympy.Expr) -> str:
    """
    Normal sympy str is very slow, this is a lot faster.  The result are
    somewhat worse, as it doesn't do as much simplification.  So don't
    use this for final codegen.
    """
    if isinstance(expr, sympy.Symbol):
        return expr.name
    if isinstance(expr, sympy.Add):
        return " + ".join(map(sympy_str, expr.args))
    if isinstance(expr, sympy.Mul):
        return " * ".join(map(sympy_str, expr.args))

    if isinstance(expr, (ModularIndexing, CleanDiv, FloorDiv)):
        return f"{expr.func.__name__}({', '.join(map(sympy_str, expr.args))})"
    return str(expr)


def sympy_index_symbol(name: str) -> sympy.Symbol:
    """
    Used to generate an integer-nonnegative symbol.
    """
    # This should never be used for creating shape/stride symbols, as those
    # should all be allocated before Inductor.
    assert name[0] != "s"
    # NOTE: shape symbols are positive (> 0), but index variables are only
    # non-negative (>= 0).
    return sympy.Symbol(name, integer=True, nonnegative=True)


def sympy_subs(expr: sympy.Expr, replacements: Dict[sympy.Expr, Any]) -> sympy.Expr:
    """
    When the passed replacement symbol v is a string, it is converted to a symbol with name v that
    have the same replaced expression integer and nonnegative properties.
    """

    def to_symbol(replaced, replacement):
        assert isinstance(replaced, sympy.Expr)
        if isinstance(replacement, str):
            return sympy.Symbol(
                replacement,
                integer=replaced.is_integer,  # type: ignore[attr-defined]
                nonnegative=replaced.is_nonnegative,  # type: ignore[attr-defined]
            )
        else:
            return replacement

    # xreplace is faster than subs, but is way more picky
    return sympy.sympify(expr).xreplace(
        {k: to_symbol(k, v) for k, v in replacements.items()}
    )


def free_symbol_startswith(index: sympy.Expr, prefix: str):
    return any(v.name.startswith(prefix) for v in index.free_symbols)  # type: ignore[attr-defined]


def free_symbol_has(index: sympy.Expr, pattern: str):
    return any(pattern in v.name for v in index.free_symbols)  # type: ignore[attr-defined]


def is_symbolic(a: Any) -> bool:
    return isinstance(a, torch.SymInt) or (
        isinstance(a, torch.Tensor)
        and any(is_symbolic(x) for x in itertools.chain(a.size(), a.stride()))
    )


def any_is_symbolic(*args: Any) -> bool:
    return any(is_symbolic(a) for a in args)


def has_incompatible_cudagraph_ops(gm):
    from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols

    forbidden_set = {
        "aten._fused_moving_avg_obs_fq_helper.default",
        "aten._fused_moving_avg_obs_fq_helper_functional.default",
        "aten.multinomial.default",
        "fbgemm.dense_to_jagged.default",
        "fbgemm.jagged_to_padded_dense.default",
        "run_and_save_rng_state",
        "run_with_rng_state",
        "aten._local_scalar_dense",
        # Technically, it's not necessary to ban this, because an
        # assert_scalar with constant arguments can be validly run
        # with CUDA graphs, but the operator is also pointless with
        # constant arguments, so might as well ban
        "aten._assert_scalar",
    }
    if torch.are_deterministic_algorithms_enabled():
        forbidden_set.update(
            {
                "aten._unsafe_index_put.default",
                "aten.index_put.default",
                "aten.index_put_.default",
                "aten.scatter.src",
                "aten.scatter.reduce",
                "aten.scatter.value_reduce",
                "aten.scatter_add_",
                "aten.scatter_add.default",
                "aten.scatter_reduce.two",
                "aten.scatter_reduce_.two",
                "aten.scatter_reduce.two_out",
            }
        )
    for node in gm.graph.nodes:
        if str(node.target) in forbidden_set:
            return True
        if (val := node.meta.get("val")) is not None and free_unbacked_symbols(val):
            return True
    return False


def output_node(gm: torch.fx.GraphModule):
    """Get the output node from an FX graph"""
    last_node = next(iter(reversed(gm.graph.nodes)))
    assert last_node.op == "output"
    return last_node


_registered_caches: List[Any] = []


def clear_on_fresh_inductor_cache(obj: Any):
    """
    Use this decorator to register any caches that should be cache_clear'd
    with fresh_inductor_cache().
    """
    if not hasattr(obj, "cache_clear") or not callable(obj.cache_clear):
        raise AttributeError(f"{obj} does not have a cache_clear method")

    _registered_caches.append(obj)
    return obj


@contextlib.contextmanager
def fresh_inductor_cache(cache_entries=None):
    """
    Contextmanager that provides a clean tmp cachedir for inductor.

    Optionally, pass a dict as 'cache_entries' to get a list of filenames and sizes
    generated with this cache instance.
    """
    for obj in _registered_caches:
        obj.cache_clear()

    inductor_cache_dir = tempfile.mkdtemp()
    try:
        with mock.patch.dict(
            os.environ, {"TORCHINDUCTOR_CACHE_DIR": inductor_cache_dir}
        ):
            triton_cache_dir = os.path.join(inductor_cache_dir, "triton")
            with mock.patch.dict(os.environ, {"TRITON_CACHE_DIR": triton_cache_dir}):
                yield
                if isinstance(cache_entries, dict):
                    assert len(cache_entries) == 0, "expected empty cache_entries dict"
                    if os.path.exists(triton_cache_dir):
                        files = os.listdir(triton_cache_dir)
                        cache_entries.update(
                            {
                                f: os.path.getsize(os.path.join(triton_cache_dir, f))
                                for f in files
                                if ".lock" not in f
                            }
                        )
        shutil.rmtree(inductor_cache_dir)
    except Exception:
        log.warning("on error, temporary cache dir kept at %s", inductor_cache_dir)
        raise


def argsort(seq) -> List[int]:
    # preserve original order for equal strides
    getter = seq.__getitem__
    a_r = range(len(seq))
    return list(reversed(sorted(a_r, key=getter, reverse=True)))  # noqa: C413


@functools.lru_cache(8)
def get_dtype_size(dtype):
    return torch.empty((), dtype=dtype).element_size()


class LineContext(NamedTuple):
    context: Any


class IndentedBuffer:
    tabwidth = 4

    def __init__(self, initial_indent=0):
        self._lines = []
        self._indent = initial_indent

    def getvaluewithlinemap(self) -> tuple[str, list[tuple[int, LineContext]]]:
        buf = StringIO()
        p = 1
        linemap = []
        for line in self._lines:
            if isinstance(line, DeferredLineBase):
                line = line()
                if line is None:
                    continue
            elif isinstance(line, LineContext):
                linemap.append((p, line.context))
                continue
            assert isinstance(line, str)
            buf.write(line)
            buf.write("\n")
            p += 1 + line.count("\n")
        return buf.getvalue(), linemap

    def getvalue(self) -> str:
        v, _ = self.getvaluewithlinemap()
        return v

    def getrawvalue(self) -> str:
        buf = StringIO()
        for line in self._lines:
            if isinstance(line, DeferredLineBase):
                line = line()
                if line is None:
                    continue
            elif isinstance(line, LineContext):
                continue
            assert isinstance(line, str)
            # backslash implies line continuation
            if line.endswith("\\"):
                buf.write(line[:-1])
            else:
                buf.write(line)
                buf.write("\n")
        return buf.getvalue()

    def clear(self):
        self._lines.clear()

    def __bool__(self):
        return bool(self._lines)

    def prefix(self):
        return " " * (self._indent * self.tabwidth)

    def newline(self):
        self.writeline("\n")

    def writeline(self, line):
        if isinstance(line, LineContext):
            self._lines.append(line)
        elif isinstance(line, DeferredLineBase):
            self._lines.append(line.with_prefix(self.prefix()))
        elif line.strip():
            self._lines.append(f"{self.prefix()}{line}")
        else:
            self._lines.append("")

    def writelines(self, lines):
        for line in lines:
            self.writeline(line)

    def indent(self, offset=1):
        @contextlib.contextmanager
        def ctx():
            self._indent += offset
            try:
                yield
            finally:
                self._indent -= offset

        return ctx()

    def do_indent(self, offset=1):
        self._indent += offset

    def do_unindent(self, offset=1):
        self._indent -= offset

    def splice(self, other_code, strip=False):
        if isinstance(other_code, IndentedBuffer):
            dedent = float("inf")
            for line in other_code._lines:
                if not isinstance(line, LineContext) and line:
                    dedent = min(dedent, len(line) - len(line.lstrip()))
            if math.isinf(dedent):
                dedent = 0
            for line in other_code._lines:
                if isinstance(line, LineContext):
                    self._lines.append(line)
                else:
                    IndentedBuffer.writeline(self, line[int(dedent) :])
        else:
            other_code = textwrap.dedent(other_code)
            if strip:
                other_code = other_code.lstrip()
            if not other_code:
                return
            other_code = other_code.rstrip()
            for line in other_code.split("\n"):
                self.writeline(line)

    def map(self, func: Callable[[Any], Any]) -> IndentedBuffer:
        res = IndentedBuffer(initial_indent=self._indent)
        res._lines = [func(line) for line in self._lines]
        return res

    def __repr__(self):
        return f"{type(self)}({self.getvalue()})"

    def __add__(self, other):
        assert self._indent == other._indent
        res = IndentedBuffer(initial_indent=self._indent)
        res.writelines(self._lines)
        res.writelines(other._lines)
        return res


class DeferredLineBase:
    """A line that can be 'unwritten' at a later time"""

    def __init__(self, line):
        if not line.strip():
            line = ""
        self.line = line

    def __call__(self) -> Optional[str]:
        """Returns either self.line or None to indicate the line has been 'unwritten'"""
        raise NotImplementedError

    def _new_line(self, line: str) -> DeferredLineBase:
        """Returns a new deferred line with the same condition"""
        raise NotImplementedError

    def with_prefix(self, prefix):
        return self._new_line(f"{prefix}{self.line}")

    def lstrip(self):
        return self._new_line(self.line.lstrip())

    def __getitem__(self, index):
        return self._new_line(self.line[index])

    def __bool__(self):
        return bool(self.line)

    def __len__(self):
        return len(self.line)


@functools.lru_cache(None)
def is_big_gpu(index) -> bool:
    min_sms = 68  # 3080
    avail_sms = torch.cuda.get_device_properties(index).multi_processor_count
    if avail_sms < min_sms:
        log.warning(
            "Not enough SMs to use max_autotune_gemm mode",
            extra={"min_sms": min_sms, "avail_sms": avail_sms},
        )
        return False
    return True


def use_max_autotune() -> bool:
    return (
        config.max_autotune or config.max_autotune_gemm or config.search_autotune_cache
    )


def _use_template_for_cuda(layout, allowed_layout_dtypes: List[torch.dtype]) -> bool:
    return (
        use_max_autotune()
        and layout.device.type == "cuda"
        and layout.dtype in allowed_layout_dtypes
        and is_big_gpu(layout.device.index or 0)
    )


def _use_autotune_backend(backend: str) -> bool:
    return backend.upper() in [
        x.strip() for x in config.max_autotune_gemm_backends.upper().split(",")
    ]


def use_triton_template(layout, *, enable_int32=False):
    layout_dtypes = [torch.float16, torch.bfloat16, torch.float32]
    if enable_int32:
        layout_dtypes = [torch.float16, torch.bfloat16, torch.float32, torch.int32]
    return _use_template_for_cuda(layout, layout_dtypes) and _use_autotune_backend(
        "TRITON"
    )


def use_cutlass_template(layout, m, n, k):
    from .virtualized import V

    gemm_size = V.graph.sizevars.size_hint(m * n * k, fallback=-1)
    if gemm_size <= 0 or gemm_size < config.cuda.cutlass_backend_min_gemm_size:
        return False
    from .codegen.cuda.cutlass_utils import try_import_cutlass

    # Do not use cutlass template on ROCm
    if torch.version.hip:
        return False

    layout_dtypes = [torch.float16, torch.bfloat16, torch.float32, torch.int32]
    res = _use_template_for_cuda(layout, layout_dtypes) and _use_autotune_backend(
        "CUTLASS"
    )

    if res:
        if not try_import_cutlass():
            log.warning(
                "Failed to import CUTLASS lib. Please check whether "
                "_inductor.config.cuda.cutlass_dir is set correctly. "
                "Skipping CUTLASS backend for now."
            )
            return False
    return res


def use_aten_gemm_kernels():
    return not use_max_autotune() or _use_autotune_backend("ATEN")


class DebugDirManager:
    counter = itertools.count(0)
    prev_debug_name: str

    def __init__(self):
        self.id = next(DebugDirManager.counter)

    def __enter__(self):
        self.prev_debug_name = torch._dynamo.config.debug_dir_root
        self.new_name = f"{self.prev_debug_name}_tmp_{self.id}"
        torch._dynamo.config.debug_dir_root = self.new_name

    def __exit__(self, *args):
        shutil.rmtree(self.new_name)
        torch._dynamo.config.debug_dir_root = self.prev_debug_name


def run_and_get_code(fn, *args, **kwargs):
    from .graph import GraphLowering

    compile_to_module = GraphLowering.compile_to_module
    source_codes: List[str] = []

    def patched_compile_to_module(self):
        mod = compile_to_module(self)
        with open(mod.__file__) as f:
            source_codes.append(f.read())
        return mod

    # If FX code caching is enabled, a hit prevents getting the code.
    with config.patch({"fx_graph_cache": False}):
        with mock.patch.object(
            GraphLowering, "compile_to_module", patched_compile_to_module
        ):
            torch._dynamo.reset()
            result = fn(*args, **kwargs)
    return result, source_codes


def get_code(fn, *args, **kwargs):
    """Get the inductor-generated code, but skip any actual compilation or running."""
    from .graph import GraphLowering

    source_codes: List[str] = []

    def patched_compile_to_module(self: GraphLowering):
        class DummyModule:
            """This is empty to replace the generated triton module"""

            def __init__(self):
                pass

            def call(self, *args, **kwargs):
                # Don't do anything when called
                pass

        code, _ = (
            self.codegen_with_cpp_wrapper() if self.cpp_wrapper else self.codegen()
        )
        # Skip all the actual compiling.

        source_codes.append(code)
        return DummyModule()

    # If FX code caching is enabled, a hit prevents getting the code.
    with config.patch({"fx_graph_cache": False}):
        with mock.patch.object(
            GraphLowering, "compile_to_module", patched_compile_to_module
        ):
            torch._dynamo.reset()
            # Note the return here is None
            _ = fn(*args, **kwargs)

    return source_codes


def get_triton_code(fn, *args, **kwargs):
    source_codes = get_code(fn, *args, **kwargs)
    # Can have two outputs if backwards was eagerly compiled
    assert (
        1 <= len(source_codes) <= 2
    ), f"expected one or two code outputs got {len(source_codes)}"
    return source_codes[0]


def run_and_get_triton_code(fn, *args, **kwargs):
    _, source_codes = run_and_get_code(fn, *args, **kwargs)
    # Can have two outputs if backwards was eagerly compiled
    assert (
        1 <= len(source_codes) <= 2
    ), f"expected one or two code outputs got {len(source_codes)}"
    return source_codes[0]


@contextlib.contextmanager
def override_lowering(aten_op, override_fn):
    """
    Override the lowering of aten_op with override_fn.
    The first argument of override_fn is the original lowering fn.
    """
    from torch._inductor import lowering

    orig_fn = lowering.lowerings[aten_op]
    try:
        lowering.lowerings[aten_op] = functools.partial(override_fn, orig_fn)
        yield
    finally:
        lowering.lowerings[aten_op] = orig_fn


def add_scheduler_init_hook(pre_fn, post_fn=None):
    """
    Add hook functions to be called at the beginning and end of Scheduler.__init__.
    Used for unit tests.
    """
    from torch._inductor.scheduler import Scheduler

    orig_fn = Scheduler.__init__

    def wrapper(scheduler, nodes):
        pre_fn(scheduler, nodes)
        out = orig_fn(scheduler, nodes)
        if post_fn:
            post_fn(scheduler, nodes)
        return out

    return unittest.mock.patch.object(Scheduler, "__init__", wrapper)


def developer_warning(msg):
    """
    Warnings that will be actionable for PyTorch developers, but not
    end users.  Allows us to easily disable them in stable releases but
    keep them on for nightly builds.
    """
    if config.developer_warnings:
        log.warning(msg)
    else:
        log.info(msg)


def get_benchmark_name():
    """
    An experimental API used only when config.benchmark_kernel is true.

    The benchmark name is only available at codegen time. So we can not
    directly call it in benchmark_all_kernels which is run after codegen.

    The function assumes the argument after --only is the benchmark name.
    It works for torchbench.py/hugginface.py/timm_models.py. But for ad-hoc
    scripts, this function may return None.

    There are 2 flavors of --only argument we need handle:
    1. --only model_name
    2. --only=model_name
    """
    try:
        idx = sys.argv.index("--only")
        if (
            idx + 1 < len(sys.argv)
            and len(sys.argv[idx + 1]) > 0
            and sys.argv[idx + 1][0] != "-"
        ):
            return sys.argv[idx + 1]
    except ValueError:
        pass

    for arg in sys.argv:
        if arg.startswith("--only="):
            return arg[len("--only=") :]


def is_ones(items):
    return all(x == 1 for x in items)


def is_zeros(items):
    return all(x == 0 for x in items)


def is_cpu_device(inputs):
    return all(
        item.device == torch.device("cpu")
        for item in inputs
        if isinstance(item, torch.Tensor)
    )


def get_sympy_Expr_dtype(val: sympy.Expr) -> torch.dtype:
    assert isinstance(
        val, sympy.Expr
    ), "only support sympy.Expr as input to get_sympy_Expr_dtype"
    if val.is_integer:  # type: ignore[attr-defined]
        return torch.int64
    else:
        return torch.float64


@contextlib.contextmanager
def maybe_profile(should_profile, *args, **kwargs):
    if should_profile:
        with torch.profiler.profile(*args, **kwargs) as p:
            yield p
    else:
        yield


def parallel_num_threads():
    threads = config.cpp.threads
    if threads < 1:
        threads = torch.get_num_threads()
    return threads


@functools.lru_cache(None)
def get_device_tflops(dtype):
    from triton.testing import get_max_simd_tflops, get_max_tensorcore_tflops

    assert dtype in (torch.float16, torch.bfloat16, torch.float32)

    if inspect.signature(get_max_simd_tflops).parameters.get("clock_rate"):
        # Triton API change in https://github.com/openai/triton/pull/2293
        from torch._utils_internal import max_clock_rate

        sm_clock = max_clock_rate()
        if dtype in (torch.float16, torch.bfloat16):
            return get_max_tensorcore_tflops(dtype, sm_clock)

        if torch.backends.cuda.matmul.allow_tf32:
            return get_max_tensorcore_tflops(torch.float32, sm_clock)
        else:
            return get_max_simd_tflops(torch.float32, sm_clock)
    else:
        if dtype in (torch.float16, torch.bfloat16):
            return get_max_tensorcore_tflops(dtype)

        if torch.backends.cuda.matmul.allow_tf32:
            return get_max_tensorcore_tflops(torch.float32)
        else:
            return get_max_simd_tflops(torch.float32)


@functools.lru_cache(None)
def get_gpu_dram_gbps():
    from triton.testing import get_dram_gbps

    return get_dram_gbps()


def is_welford_reduction(reduction_type):
    return reduction_type.startswith("welford")


def reduction_num_outputs(reduction_type):
    return 3 if is_welford_reduction(reduction_type) else 1


def is_linux() -> bool:
    return platform.system() == "Linux"


def has_free_symbols(itr: Iterable[Any]):
    return any(isinstance(x, sympy.Expr) and not x.is_number for x in itr)


def is_dynamic(*args):
    from . import ir

    for t in args:
        if isinstance(t, ir.TensorBox):
            if has_free_symbols(t.data.get_size()) or (
                hasattr(t.data, "get_stride") and has_free_symbols(t.data.get_stride())
            ):
                return True
        elif isinstance(t, (ir.StorageBox, ir.BaseView, ir.ComputedBuffer)):
            assert hasattr(t, "get_size") and hasattr(t, "get_stride")
            if has_free_symbols(t.get_size()) or has_free_symbols(t.get_stride()):
                return True
        elif not isinstance(t, ir.IRNode):
            continue
        else:
            raise TypeError(f"unexpected type for is_dynamic {type(t)}")

    return False


# Placeholder strings used in triton codegen.
class Placeholder(enum.Enum):
    # The placeholder for the actual name of a triton kernel.
    # e.g. for "def triton_" it would be "triton_"
    KERNEL_NAME = "KERNEL_NAME"

    # The descriptive name of the triton kernel; when unique_kernel_names = False, this
    # placeholder will be replaced with a string with more information.
    DESCRIPTIVE_NAME = "DESCRIPTIVE_NAME"


def pass_execution_and_save(func, gm, inp, msg):
    from .pattern_matcher import stable_topological_sort

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        delete=False,
    ) as f:
        before_io = io.StringIO()
        after_io = io.StringIO()
        ShapeProp(gm=gm, fake_mode=detect_fake_mode(inp)).propagate(*inp)
        print(f"Before:\n{gm.graph}", file=f)
        print(gm.graph, file=before_io)
        start_time = datetime.now()
        func(gm.graph)
        time_elapsed = datetime.now() - start_time
        # recompile graph
        stable_topological_sort(gm.graph)
        gm.graph.lint()
        gm.recompile()

        print(f"After:\n{gm.graph}", file=f)
        print(gm.graph, file=after_io)
        t = before_io.getvalue() == after_io.getvalue()
        log.info(
            "%s, save before/after graph to %s, graph before/after are the same = %s, time elapsed = %s",
            msg,
            f.name,
            t,
            time_elapsed,
        )


def is_collective(node):
    from . import ir

    return isinstance(node, ir.CollectiveKernel) or type(node) == ir._CollectiveKernel


def is_wait(node):
    from . import ir

    return isinstance(node, ir.Wait) or type(node) == ir._WaitKernel


def num_fw_fixed_arguments(dynamo_gm_num_inputs: int, aot_fw_gm_num_inputs: int):
    "Computes the number of inputs to the aot fw graph which have fixed addresses (params and buffers)"
    num_rng_seed_offset_inputs = (
        2 if torch._functorch.config.functionalize_rng_ops else 0
    )
    return aot_fw_gm_num_inputs - dynamo_gm_num_inputs - num_rng_seed_offset_inputs


def count_tangents(fx_g: torch.fx.GraphModule):
    """
    Infers which inputs are static for a backwards graph
    """

    def is_saved_tensor(x):
        return (
            "tangents" not in x.name
            and "bwd_seed" not in x.name
            and "bwd_base_offset" not in x.name
        )

    arg_count = 0
    static_arg_idxs = []
    for n in fx_g.graph.nodes:
        if n.op == "placeholder":
            if is_saved_tensor(n):
                static_arg_idxs.append(arg_count)
            arg_count += 1

    assert static_arg_idxs == list(range(len(static_arg_idxs)))
    return len(static_arg_idxs)


@dataclasses.dataclass
class BoxedBool:
    value: bool

    def __bool__(self):
        return self.value

    @staticmethod
    def disable(obj):
        if isinstance(obj, BoxedBool):
            obj.value = False
            return obj
        return False


@contextlib.contextmanager
def collect_defined_kernels(kernel_list):
    from .codegen.wrapper import WrapperCodeGen

    orig_define_kernel = WrapperCodeGen.define_kernel

    def new_define_kernel(wrapper, name, kernel_code, metadata, *args, **kwargs):
        nonlocal kernel_list
        kernel_list.append(kernel_code)
        return orig_define_kernel(wrapper, name, kernel_code, metadata, *args, **kwargs)

    with unittest.mock.patch.object(WrapperCodeGen, "define_kernel", new_define_kernel):
        yield


def get_cloned_parameter_buffer_name(name: str):
    return name + "__original__"


def is_gpu(device: str):
    return device in ["cuda", "xpu"]


def device_need_guard(device: str):
    assert isinstance(device, str)
    return is_gpu(device)


def needs_fallback_due_to_atomic_add_limitations(dtype):
    # tl.atomic_add does NOT support the following types
    return dtype in {torch.int64, torch.bool, torch.bfloat16}


def use_scatter_fallback(
    op_overload: torch._ops.OpOverload,
    reduction_type,
    self_dtype,
    src_dtype,
    src_device_type,
    src_is_tensor,
):
    reduce_ty = (
        "add" if op_overload.overloadpacket == torch.ops.aten.scatter_ else "sum"
    )

    return (
        reduction_type not in {None, reduce_ty}
        or (
            src_is_tensor
            and is_gpu(src_device_type)
            and needs_fallback_due_to_atomic_add_limitations(src_dtype)
        )
        or (
            op_overload.overloadpacket == torch.ops.aten.scatter_reduce_
            and reduction_type == "sum"
            and src_is_tensor
            and src_device_type == "cpu"
            and config.cpp.fallback_scatter_reduce_sum
            and (config.cpp.dynamic_threads or parallel_num_threads() != 1)
        )
        or (reduction_type == reduce_ty and self_dtype in {torch.bool, torch.int64})
        or torch.are_deterministic_algorithms_enabled()
    )


def dump_node_schedule(node_schedule):
    """
    An API that can be used in pdb to dump a node_schedule.
    Right mainly dump the read/write dependencies but can add more as needed.
    """
    from torch._inductor.codegen.triton import DisableReduction, EnableReduction
    from torch._inductor.scheduler import SchedulerNode

    print(f"Node schedule with {len(node_schedule)} nodes")
    for idx, node in enumerate(node_schedule):
        print(f" {idx:3}:")
        if node is EnableReduction:
            print("enable reduction")
        elif node is DisableReduction:
            print("disable reduction")
        elif isinstance(node, SchedulerNode):
            is_red = node.is_reduction()
            print(f"{'red' if is_red else 'pw'} scheduler node")
            if is_red:
                print(f"original reduction hint {node.node.data.reduction_hint}")  # type: ignore[attr-defined]
            print("ReadDep:")
            for dep in node.read_writes.reads:
                print(dep)
            print("WriteDep:")
            for dep in node.read_writes.writes:
                print(dep)
        else:
            raise RuntimeError(f"Unrecognized node type: {type(node)}")
