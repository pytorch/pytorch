from __future__ import annotations

import collections
import contextlib
import dataclasses
import enum
import functools
import importlib
import inspect
import io
import itertools
import logging
import math
import operator
import os
import platform
import re
import shutil
import statistics
import sys
import tempfile
import textwrap
import time
import unittest
from collections.abc import Collection, Iterator, Mapping, MutableMapping, MutableSet
from datetime import datetime
from io import StringIO
from typing import (
    Any,
    Callable,
    cast,
    Generic,
    Literal,
    NamedTuple,
    Optional,
    Protocol,
    TYPE_CHECKING,
    TypeVar,
    Union,
)
from typing_extensions import (
    Concatenate,
    dataclass_transform,
    ParamSpec,
    Self,
    TypeAlias,
    TypeGuard,
)
from unittest import mock

import sympy

import torch
from torch._inductor.runtime.hints import DeviceProperties
from torch.utils._ordered_set import OrderedSet
from torch.utils._pytree import tree_map_only


if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence, ValuesView

    from torch import SymBool, SymFloat, SymInt
    from torch._prims_common import ELEMENTWISE_TYPE_PROMOTION_KIND
    from torch.fx import GraphModule
    from torch.fx.experimental.symbolic_shapes import ShapeEnv
    from torch.fx.node import Node

    from .codegen.common import WorkspaceArg
    from .codegen.wrapper import PythonWrapperCodegen
    from .graph import GraphLowering
    from .ir import (
        Buffer,
        ExternKernel,
        ExternKernelOut,
        IRNode,
        Layout,
        Operation,
        ReinterpretView,
    )
    from .output_code import CompiledFxGraph
    from .scheduler import BaseSchedulerNode, SchedulerBuffer


GPU_TYPES = ["cuda", "mps", "xpu"]
T = TypeVar("T")


# defines here before import torch._dynamo is for avoiding circular import
# when get_gpu_type is imported from dynamo
@functools.lru_cache(None)
def get_gpu_type() -> str:
    avail_gpus = [x for x in GPU_TYPES if getattr(torch, x).is_available()]
    assert len(avail_gpus) <= 1
    gpu_type = "cuda" if len(avail_gpus) == 0 else avail_gpus.pop()
    return gpu_type


from torch._dynamo.device_interface import get_interface_for_device
from torch._dynamo.utils import detect_fake_mode
from torch.autograd import DeviceType
from torch.autograd.profiler_util import EventList
from torch.fx.passes.graph_transform_observer import GraphTransformObserver
from torch.fx.passes.shape_prop import ShapeProp
from torch.utils._sympy.functions import (
    CeilDiv,
    CleanDiv,
    FloorDiv,
    Identity,
    ModularIndexing,
)
from torch.utils._sympy.symbol import make_symbol, SymT
from torch.utils._sympy.value_ranges import bound_sympy, ValueRanges

from . import config
from .runtime.runtime_utils import ceildiv as runtime_ceildiv


_IS_WINDOWS = sys.platform == "win32"

log = logging.getLogger(__name__)

_T = TypeVar("_T")
VarRanges = dict[sympy.Expr, sympy.Expr]
InputType = Optional[Union[torch.Tensor, int, torch.SymInt]]

GPU_KERNEL_BIN_EXTS = {"cuda": ".cubin", "xpu": ".spv"}

GPU_ALIGN_BYTES = 16
ALIGNMENT = 16

TMA_ALIGNMENT = 16
TMA_DESCRIPTOR_SIZE = 128

ALIGN_BYTES = 64
assert (ALIGN_BYTES & (ALIGN_BYTES - 1)) == 0 and ALIGN_BYTES >= 8, "must be power of 2"


def _align(nbytes: int) -> int:
    """Round up to the nearest multiple of ALIGN_BYTES"""
    return (nbytes + ALIGN_BYTES - 1) & -ALIGN_BYTES


def _is_aligned(v: sympy.Expr) -> bool:
    """v can be statically proven to be a multiple of ALIGN_BYTES"""
    if isinstance(v, (sympy.Add, sympy.Max)):
        return all(map(_is_aligned, v.args))
    return isinstance(v, align) or sympy.gcd(v, ALIGN_BYTES) == ALIGN_BYTES


class align(sympy.Function):
    """Symbolically round up to the nearest multiple of ALIGN_BYTES"""

    nargs = (1,)
    is_integer = True

    @classmethod
    def eval(cls, value: sympy.Expr) -> Optional[sympy.Expr]:
        if isinstance(value, (int, sympy.Integer)):
            return _align(int(value))
        if _is_aligned(value):
            return value


@dataclasses.dataclass(frozen=True)
class GraphPartitionMap:
    """
    Mapping from the partition info (e.g., input/output) to the graph info
    """

    # a unique id of graph partition
    id: int

    # map partition input/output indices to graph input/output indices. None indicates
    # a partition input/output is not a graph input/output.
    input_index_mapping: list[Optional[int]]
    output_index_mapping: list[Optional[int]]

    # name of constants read/written by the graph partition
    constant_names: list[str]


def fp8_bench(fn: Callable[[], Any], warmup: int = 25, rep: int = 100) -> float:
    """
    Returns benchmark results by examining torch profiler events.
    This could be more accurate as it doesn't count CPU side overhead.
    However, this also requires manually excluding irrelevant event, e.g.
    vectorized_elementwise_kernel which is used to fill L2 cache,
    various CUDA events, etc, so could also be fragile.
    """

    fn()
    torch.cuda.synchronize()
    cache = torch.empty(int(256e6 // 4), dtype=torch.float16, device="cuda")

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

    start_event = [torch.cuda.Event(enable_timing=True) for _ in range(n_repeat)]
    end_event = [torch.cuda.Event(enable_timing=True) for _ in range(n_repeat)]
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as p:
        torch.cuda.synchronize()
        for i in range(n_repeat):
            cache.zero_()
            start_event[i].record()
            with torch.cuda.nvtx.range("RunCudaModule"):
                fn()
            end_event[i].record()
        torch.cuda.synchronize()
        times = torch.tensor(
            [s.elapsed_time(e) for s, e in zip(start_event, end_event)]
        )

    res = torch.mean(times).item()
    log.debug("raw events")
    log.debug(p.key_averages().table(sort_by="self_device_time_total", row_limit=-1))
    filtered_events = EventList(
        [
            event
            for event in p.events()
            if event.device_type == DeviceType.CUDA and "fused_abs_max_0" in event.name
        ]
    )
    if filtered_events:
        res -= (
            statistics.mean(event.device_time_total for event in filtered_events)
            / 1000.0
        )

    log.debug("profiling results: %s ms", res)
    return res


def do_bench_using_profiling(
    fn: Callable[[], Any], warmup: int = 25, rep: int = 100
) -> float:
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
    log.debug(p.key_averages().table(sort_by="self_device_time_total", row_limit=-1))

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

        torch._C._dispatch_has_kernel_for_dispatch_key("torchvision::nms", "Meta")
        return roi_align is not None and hasattr(
            getattr(torch.ops, "torchvision", None), "roi_align"
        )
    except ImportError:
        return False
    except RuntimeError as e:
        assert "torchvision::nms does not exist" in str(e)
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


def sympy_product(it: Iterable[sympy.Expr]) -> sympy.Expr:
    return functools.reduce(operator.mul, it, sympy.S.One)


def sympy_dot(seq1: Sequence[sympy.Expr], seq2: Sequence[sympy.Expr]) -> sympy.Expr:
    assert len(seq1) == len(seq2)
    return sympy.expand(sum(a * b for a, b in zip(seq1, seq2)))


def unique(it: Iterable[_T]) -> ValuesView[_T]:
    return {id(x): x for x in it}.values()


def ceildiv(
    numer: Union[int, sympy.Expr], denom: Union[int, sympy.Expr]
) -> Union[int, sympy.Expr]:
    if isinstance(numer, sympy.Expr) or isinstance(denom, sympy.Expr):
        return CeilDiv(sympy.sympify(numer), sympy.sympify(denom))
    # TODO: There is a bug in a call to this function, to repro:
    # python benchmarks/dynamo/huggingface.py --inductor -d cuda --accuracy
    # --amp --only YituTechConvBert --dynamic-shapes
    assert isinstance(numer, int) and isinstance(denom, int), (
        f"{numer}: {type(numer)}, {denom}: {type(denom)}"
    )
    return runtime_ceildiv(numer, denom)


def _type_of(key: Optional[torch.dtype]) -> str:
    # Use the function here to get rid of dependencies on the Triton during the codegen.
    # Refer to Triton implementation here:
    # https://github.com/triton-lang/triton/blob/98b5945d2aef679e00ebca8e07c35c3658ec76de/python/triton/runtime/jit.py#L238
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
        # TODO: remove when support is added in triton
        # https://github.com/triton-lang/triton/issues/6054
        "float8_e8m0fnu": "u8",
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
    tys.update({v: v for v in list(tys.values())})
    return key if isinstance(key, str) else f"*{tys[dtype_str]}"


def convert_shape_to_inductor(
    lst: Iterable[Union[int, torch.SymInt]],
) -> list[sympy.Expr]:
    """
    Gets the shape and stride of a tensor. For non-symbolic tensors, this is
    trivial. But for symbolic tensors, we need to map from SymIntNode into
    sympy.Expr.
    """
    return [sympy.sympify(i) for i in lst]


def convert_to_symint(i: Union[int, sympy.Expr]) -> Union[int, torch.SymInt]:
    """
    Like convert_shape_to_symint, but operates on a single expression.
    """
    from .virtualized import V

    return (
        i
        if isinstance(i, int)
        else (
            int(i)
            if isinstance(i, sympy.Integer)
            else V.graph.sizevars.shape_env.create_symintnode(i, hint=None)
        )
    )


def convert_shape_to_symint(
    lst: Iterable[Union[int, sympy.Expr]],
) -> list[Union[int, torch.SymInt]]:
    """
    Takes a list of shapes from Inductor and converts them into symints (or just
    ints if all shapes are static).
    """
    return [convert_to_symint(i) for i in lst]


def is_view(op: torch._ops.OpOverload) -> bool:
    """
    Does this op overload have aliasing
    """
    return any(a.alias_info is not None for a in op._schema.arguments)


def is_pointwise_use(
    use: Node,
    is_pointwise_fn: Callable[[torch._ops.OpOverload], bool] = lambda _: False,
) -> bool:
    """
    Do all uses of this op have torch.Tag.pointwise or return True for optional `is_pointwise_fn`

    Uses in views ops will follow the views uses
    """

    if not use.op == "call_function":
        return False
    if not (
        isinstance(use.target, torch._ops.OpOverload) or use.target is operator.getitem
    ):
        return False

    target = cast(torch._ops.OpOverload, use.target)
    if target is operator.getitem or is_view(target):
        return all(is_pointwise_use(u, is_pointwise_fn) for u in use.users)

    return torch.Tag.pointwise in target.tags or is_pointwise_fn(target)


def gen_gm_and_inputs(
    target: Any, args: list[Any], kwargs: dict[str, Any]
) -> tuple[GraphModule, list[torch.Tensor]]:
    g = torch.fx.Graph()
    graph_args: list[torch.Tensor] = []

    def add_tensor_arg(arg: torch.Tensor) -> Node:
        graph_args.append(arg)
        return g.placeholder(f"arg{len(graph_args)}")

    node = g.call_function(
        target, *tree_map_only(torch.Tensor, add_tensor_arg, (args, kwargs))
    )
    if (
        len(target._schema.returns) == 1
        and str(target._schema.returns[0].type) == "Tensor"
    ):
        node = (node,)  # type: ignore[assignment]
    g.output(node)

    gm = torch.fx.GraphModule({}, g)
    return gm, graph_args


def synchronize(device: str = "cuda") -> None:
    if device == "cpu":
        return
    device_interface = get_interface_for_device(device)
    if device_interface.is_available():
        device_interface.synchronize()


def timed(
    model: Callable[..., Any],
    example_inputs: Sequence[Any],
    times: int = 1,
    device: str = "cuda",
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
    model: Callable[..., Any],
    example_inputs: Sequence[Any] = (),
    times: int = 10,
    repeat: int = 10,
    baseline: float = 1.0,
    device: str = "cuda",
) -> float:
    timings = torch.tensor(
        [timed(model, example_inputs, times, device) for _ in range(repeat)]
    )
    took = torch.median(timings) / times
    print(f"{took / baseline:.6f}")
    return took.item()


def precompute_method(obj: Any, method: str) -> None:
    """Replace obj.method() with a new method that returns a precomputed constant."""
    result = getattr(obj, method)()
    setattr(obj, method, lambda: result)


def precompute_methods(obj: Any, methods: list[str]) -> None:
    """Replace methods with new methods that returns a precomputed constants."""
    for method in methods:
        precompute_method(obj, method)


def cmp(a: int, b: int) -> int:
    return int(a > b) - int(a < b)


def pad_listlike(x: Union[int, Sequence[int]], size: int) -> Sequence[int]:
    if isinstance(x, int):
        return [x] * size
    if len(x) == 1:
        return type(x)([x[0]]) * size  # type: ignore[call-arg, operator, return-value]
    return x


# Used to ensure that iterating over a set is deterministic
def tuple_sorted(x: tuple[_T, ...]) -> list[_T]:
    if len(x) == 0:
        return []

    def sort_func(elem: _T) -> str:
        if isinstance(elem, str):
            return elem

        from .scheduler import BaseSchedulerNode

        assert isinstance(elem, BaseSchedulerNode)
        return elem.get_name()

    return sorted(x, key=sort_func)


P = ParamSpec("P")
RV = TypeVar("RV", covariant=True)


class CachedMethod(Protocol, Generic[P, RV]):
    @staticmethod
    def clear_cache(cache: Any) -> None: ...

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> RV: ...


# See https://github.com/python/mypy/issues/13222#issuecomment-1193073470 to understand the type signature
def cache_on_self(fn: Callable[Concatenate[Any, P], RV]) -> CachedMethod[P, RV]:
    name = fn.__name__
    key = f"__{name}_cache"

    # wrapper is likely on the hot path, compile a specialized version of it
    ctx = {"fn": fn}
    exec(
        f"""\
        def {name}_cache_on_self(self):
            try:
                return self.{key}
            except AttributeError:
                pass
            rv = fn(self)
            object.__setattr__(self, "{key}", rv)
            return rv
        """.lstrip(),
        ctx,
    )
    wrapper = functools.wraps(fn)(ctx[f"{name}_cache_on_self"])

    def clear_cache(self: Any) -> None:
        if hasattr(self, key):
            delattr(self, key)

    wrapper.clear_cache = clear_cache  # type: ignore[attr-defined]
    return wrapper  # type: ignore[return-value]


def aggregate_origins(
    node_schedule: Union[Sequence[BaseSchedulerNode], ExternKernel],
) -> OrderedSet[Node]:
    from . import ir

    if isinstance(node_schedule, list):
        return functools.reduce(
            operator.or_,
            [
                node.node.origins
                for node in node_schedule
                if hasattr(node, "node") and node.node
            ],
            OrderedSet(),
        )
    elif isinstance(node_schedule, ir.ExternKernel):
        return node_schedule.origins
    else:
        return OrderedSet()


def get_fused_kernel_name(
    node_schedule: Sequence[BaseSchedulerNode],
    descriptive_names: Literal[True, "torch", "original_aten", "inductor_node"],
) -> str:
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
        sources = sorted(OrderedSet(sources))
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
        sources = sorted(OrderedSet(sources))
    elif descriptive_names == "inductor_node":
        sources = [
            origin.name for origin in all_origins if origin.op == "call_function"
        ]
    else:
        raise NotImplementedError
    sources = sources
    return "_".join(["fused"] + sources)


def get_kernel_metadata(
    node_schedule: Union[Sequence[BaseSchedulerNode], ExternKernel],
    wrapper: PythonWrapperCodegen,
) -> tuple[str, str]:
    all_origins = aggregate_origins(node_schedule)
    inductor_nodes = [origin for origin in all_origins if origin.op == "call_function"]

    from_node_dict = collections.defaultdict(list)
    original_aten_dict = collections.defaultdict(list)

    # Attempt to sort `inductor_nodes` topologically. Note that the case
    # where `inductor_nodes` contains nodes from multiple graph instances
    # is not supported. An example of this is conditional statements.
    single_graph = None
    if len(inductor_nodes):
        unique_graphs = OrderedSet(n.graph for n in inductor_nodes)
        if len(unique_graphs) == 1:
            single_graph = inductor_nodes[0].graph
            # create a map of idx -> node and cache it
            if not hasattr(single_graph, "_inductor_kernel_metadata_node_to_idx_map"):
                node_to_idx_map = {n: idx for idx, n in enumerate(single_graph.nodes)}
                single_graph._inductor_kernel_metadata_node_to_idx_map = node_to_idx_map  # type: ignore[attr-defined]
            inductor_nodes.sort(
                key=lambda n: single_graph._inductor_kernel_metadata_node_to_idx_map[n]  # type: ignore[attr-defined]
            )

    for node in inductor_nodes:
        if "original_aten" in node.meta and node.meta["original_aten"] is not None:
            key = str(node.meta["original_aten"]._overloadpacket)
            original_aten_dict[key].append(node.name)
        if "from_node" in node.meta:
            key = node.meta["from_node"][0].name
            from_node_dict[key].append(node.name)
    sort_str = "Topologically Sorted" if single_graph is not None else "Unsorted"
    metadata = (
        f"{wrapper.comment} {sort_str} Source Nodes: [{', '.join(from_node_dict.keys())}], "
        f"Original ATen: [{', '.join(original_aten_dict.keys())}]"
    )

    # trace back to original node here
    detailed_metadata = [f"{wrapper.comment} Source node to ATen node mapping:"]
    for original_node, nodes in sorted(from_node_dict.items()):
        detailed_metadata.append(
            f"{wrapper.comment}   {original_node} => {', '.join(sorted(nodes))}"
        )

    # print the aot_autograd graph fragment
    if single_graph is not None:
        detailed_metadata.append(f"{wrapper.comment} Graph fragment:")
        for n in inductor_nodes:
            # TODO(future): maybe refactor torch/fx/graph.py to make it easy to
            # generate python code for graph fragments
            detailed_metadata.append(f"{wrapper.comment}   {n.format_node()}")

    return metadata, "\n".join(detailed_metadata)


def dominated_nodes(
    initial_queue: Iterable[torch.fx.Node],
    skip_filter: Optional[Callable[[Any], bool]] = None,
) -> OrderedSet[torch.fx.Node]:
    """Returns the set of nodes whose values depend on those within initial_queue"""
    initial_queue = list(initial_queue)
    dominated_set = OrderedSet(initial_queue)

    while initial_queue:
        node = initial_queue.pop()
        for user in node.users:
            if skip_filter and skip_filter(user):
                continue
            if user not in dominated_set:
                dominated_set.add(user)
                initial_queue.append(user)

    return dominated_set


def gather_origins(
    args: Sequence[IRNode], kwargs: dict[str, IRNode]
) -> OrderedSet[IRNode]:
    import itertools

    from . import ir

    def is_unrealized_node(n: IRNode) -> bool:
        if isinstance(n, ir.TensorBox):
            return is_unrealized_node(n.data)
        if isinstance(n, ir.StorageBox):
            return is_unrealized_node(n.data)
        return isinstance(n, ir.IRNode) and isinstance(n, ir.Pointwise)

    kwarg_origins = [val.origins for val in kwargs.values() if is_unrealized_node(val)]
    arg_origins = [arg.origins for arg in args if is_unrealized_node(arg)]
    return OrderedSet(itertools.chain(*arg_origins, *kwarg_origins))


def sympy_str(expr: sympy.Expr) -> str:
    """
    Normal sympy str is very slow, this is a lot faster.  The result are
    somewhat worse, as it doesn't do as much simplification.  So don't
    use this for final codegen.
    """

    def is_neg_lead(expr: sympy.Expr) -> bool:
        return (
            isinstance(expr, sympy.Mul) and len(expr.args) == 2 and expr.args[0] == -1
        )

    def sympy_str_add(expr: sympy.Expr) -> str:
        if isinstance(expr, sympy.Add):
            # Special case 'a - b'. Note that 'a - b - c' will still appear as
            # 'a + -1 * b + -1 * c'.
            if len(expr.args) == 2 and is_neg_lead(expr.args[1]):
                return f"{sympy_str_mul(expr.args[0])} - {sympy_str_mul(expr.args[1].args[1])}"
            else:
                return " + ".join(map(sympy_str_mul, expr.args))
        else:
            return sympy_str_mul(expr)

    def sympy_str_mul(expr: sympy.Expr) -> str:
        if isinstance(expr, sympy.Mul):
            if is_neg_lead(expr):
                # Special case '-a'. Note that 'a * -b' will still appear as
                # '-1 * a * b'.
                return f"-{sympy_str_atom(expr.args[1])}"
            else:
                return " * ".join(map(sympy_str_atom, expr.args))
        else:
            return sympy_str_atom(expr)

    def sympy_str_atom(expr: sympy.Expr) -> str:
        if isinstance(expr, sympy.Symbol):
            return expr.name
        elif isinstance(expr, (sympy.Add, sympy.Mul)):
            return f"({sympy_str_add(expr)})"
        elif isinstance(expr, (ModularIndexing, CleanDiv, FloorDiv, Identity)):
            return f"{expr.func.__name__}({', '.join(map(sympy_str, expr.args))})"
        else:
            return str(expr)

    return sympy_str_add(expr)


def get_bounds_index_expr(index: sympy.Expr) -> ValueRanges[Any]:
    from .virtualized import V

    # If this expression does not come from an FX node, we compute its bounds
    if (
        config.compute_all_bounds
        and (fx_node := getattr(V.interpreter, "current_node", None))
        and fx_node.target != "index_expr"
    ):
        return bound_sympy(index)
    else:
        return ValueRanges.unknown()


def prefix_is_reduction(prefix: str) -> bool:
    return prefix[0] == "r"


def sympy_index_symbol_with_prefix(prefix: SymT, idx: int) -> sympy.Symbol:
    """
    Used to generate an integer-nonnegative symbol.
    """
    # This should never be used for creating shape/stride symbols, as those
    # should all be allocated before Inductor.
    assert prefix != SymT.SIZE
    # NOTE: shape symbols are positive (> 0), but index variables are only
    # non-negative (>= 0).
    return make_symbol(prefix, idx, integer=True, nonnegative=True)


def generate_assert(check: bool) -> bool:
    return (check or config.debug_index_asserts) and config.assert_indirect_indexing


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


def sympy_subs(expr: sympy.Expr, replacements: dict[sympy.Expr, Any]) -> sympy.Expr:
    """
    When the passed replacement symbol v is a string, it is converted to a symbol with name v that
    have the same replaced expression integer and nonnegative properties.
    """

    def to_symbol(
        replaced: sympy.Expr, replacement: Union[sympy.Expr, str]
    ) -> sympy.Symbol:
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


def is_symbolic(a: Any) -> TypeGuard[Union[torch.SymInt, torch.Tensor]]:
    return isinstance(a, torch.SymInt) or (
        isinstance(a, torch.Tensor)
        and any(is_symbolic(x) for x in itertools.chain(a.size(), a.stride()))
    )


def any_is_symbolic(*args: Any) -> bool:
    return any(is_symbolic(a) for a in args)


def get_first_incompatible_cudagraph_node(
    gm: torch.fx.GraphModule,
) -> Optional[torch.fx.Node]:
    from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols

    forbidden_set = OrderedSet(
        [
            "aten._fused_moving_avg_obs_fq_helper.default",
            "aten._fused_moving_avg_obs_fq_helper_functional.default",
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
        ]
    )
    if torch.are_deterministic_algorithms_enabled():
        forbidden_set.update(
            (
                "aten._unsafe_index_put.default",
                "aten._unsafe_masked_index_put_accumulate.default",
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
            )
        )

    for node in gm.graph.nodes:
        if str(node.target) in forbidden_set:
            return node

        if (
            not torch._inductor.config.graph_partition
            and isinstance(node.target, torch._ops.OpOverload)
            and torch._C.Tag.cudagraph_unsafe in node.target.tags
        ):
            # skip cudagraph if a cudagraph_unsafe op is detected.
            # graph_partition helps by spliting on this cudagraph_unsafe
            # op and cudagraphifying the subgraphs.
            return node

        if (val := node.meta.get("val")) is not None and free_unbacked_symbols(val):
            return node

    return None


def output_node(gm: torch.fx.GraphModule) -> Node:
    """Get the output node from an FX graph"""
    last_node = next(iter(reversed(gm.graph.nodes)))
    assert last_node.op == "output"
    return last_node


_registered_caches: list[Any] = []


def clear_on_fresh_inductor_cache(obj: Any) -> Any:
    """
    Use this decorator to register any caches that should be cache_clear'd
    with fresh_inductor_cache().
    """
    if not hasattr(obj, "cache_clear") or not callable(obj.cache_clear):
        raise AttributeError(f"{obj} does not have a cache_clear method")

    _registered_caches.append(obj)
    return obj


def clear_inductor_caches() -> None:
    """
    Clear all registered caches.
    """
    for obj in _registered_caches:
        obj.cache_clear()


import gc


def unload_xpu_triton_pyds() -> None:
    # unload __triton_launcher.pyd
    for module_name in list(sys.modules.keys()):
        if not module_name.startswith("torch._inductor.runtime.compile_tasks."):
            continue
        m = sys.modules[module_name]
        for attr_name in m.__dict__.keys():
            if attr_name.startswith("triton_"):
                kernel = getattr(m, attr_name)
                if isinstance(
                    kernel, torch._inductor.runtime.triton_heuristics.CachingAutotuner
                ):
                    for result in kernel.compile_results:
                        if isinstance(
                            result,
                            torch._inductor.runtime.triton_heuristics.TritonCompileResult,
                        ):
                            result.kernel.run.mod.__del__()
        del sys.modules[module_name]

    # unload spirv_utils.pyd
    if "triton.runtime.driver" in sys.modules:
        mod = sys.modules["triton.runtime.driver"]
        del type(mod.driver.active.utils).instance
        del mod.driver.active.utils

    gc.collect()


@contextlib.contextmanager
def fresh_inductor_cache(
    cache_entries: Optional[dict[str, Any]] = None,
    dir: Optional[str] = None,
    delete: bool = True,
) -> Iterator[None]:
    """
    Contextmanager that provides a clean tmp cachedir for inductor.

    Optionally, pass a dict as 'cache_entries' to get a list of filenames and sizes
    generated with this cache instance.
    """
    clear_inductor_caches()

    inductor_cache_dir = tempfile.mkdtemp(dir=dir)
    try:
        with mock.patch.dict(
            os.environ, {"TORCHINDUCTOR_CACHE_DIR": inductor_cache_dir}
        ):
            log.debug("Using inductor cache dir %s", inductor_cache_dir)
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
        if delete:
            if is_windows() and torch.xpu.is_available():
                unload_xpu_triton_pyds()

            shutil.rmtree(
                inductor_cache_dir,
                # Let's not fail if we can't clean up the temp dir. Also note that for
                # Windows, we can't delete the loaded modules because the module binaries
                # are open.
                onerror=lambda func, path, exc_info: log.warning(
                    "Failed to remove temporary cache dir at %s",
                    inductor_cache_dir,
                    exc_info=exc_info,
                ),
            )
    except Exception:
        log.warning("on error, temporary cache dir kept at %s", inductor_cache_dir)
        raise
    finally:
        clear_inductor_caches()


def argsort(seq: Sequence[Any]) -> list[int]:
    # preserve original order for equal strides
    getter = seq.__getitem__
    a_r = range(len(seq))
    return list(reversed(sorted(a_r, key=getter, reverse=True)))  # noqa: C413


def argsort_sym(
    shape_env: ShapeEnv, seq: Sequence[Union[int, torch.SymInt, sympy.Expr]]
) -> list[int]:
    def cmp(a: tuple[int, sympy.Expr], b: tuple[int, sympy.Expr]) -> int:
        a_idx, a_val = a
        b_idx, b_val = b

        def evaluate(expr: Union[bool, torch.SymInt, sympy.Expr]) -> bool:
            if isinstance(expr, bool):
                return expr
            return shape_env.evaluate_expr(expr, size_oblivious=True)

        if evaluate(a_val < b_val):
            return -1
        if evaluate(a_val > b_val):
            return 1
        # If strides are the same, prefer the original order.
        # (this matches argsort's algorithm).
        # For strides = [2048, 2048, 16, 1], this is
        # [3, 2, 1, 0].
        if a_idx < b_idx:
            return 1
        if a_idx > b_idx:
            return -1
        return 0

    # Strategy: convert all symints to sympy.Expr, then use a custom comparator
    exprs = [
        (idx, s.node.expr if isinstance(s, torch.SymInt) else s)
        for idx, s in enumerate(seq)
    ]
    exprs = sorted(exprs, key=functools.cmp_to_key(cmp))
    result = [idx for idx, _ in exprs]
    return result


@functools.lru_cache(8)
def get_dtype_size(dtype: torch.dtype) -> int:
    # TODO: Investigate why uint64 tensor creation causes overflow error:
    # Workaround for RuntimeError in memory size calculation, but underlying cause unclear
    if dtype == torch.uint64:
        return 8
    return torch.empty((), dtype=dtype).element_size()


class LineContext(NamedTuple):
    context: Any


@dataclasses.dataclass
class ValueWithLineMap:
    value: str
    line_map: list[tuple[int, LineContext]]


class IndentedBuffer:
    tabwidth = 4

    def __init__(self, initial_indent: int = 0) -> None:
        self._lines: list[Union[DeferredLineBase, LineContext, str]] = []
        self._indent = initial_indent

    @contextlib.contextmanager
    def set_tabwidth(self, tabwidth: int) -> Iterator[None]:
        prev = self.tabwidth
        try:
            self.tabwidth = tabwidth
            yield
        finally:
            self.tabwidth = prev

    def getvaluewithlinemap(self) -> ValueWithLineMap:
        buf = StringIO()
        p = 1
        linemap: list[tuple[int, LineContext]] = []
        for li in self._lines:
            if isinstance(li, DeferredLineBase):
                line = li()
                if line is None:
                    continue
            elif isinstance(li, LineContext):
                linemap.append((p, li.context))
                continue
            else:
                line = li
            assert isinstance(line, str)
            buf.write(line)
            buf.write("\n")
            p += 1 + line.count("\n")
        return ValueWithLineMap(buf.getvalue(), linemap)

    def getvalue(self) -> str:
        return self.getvaluewithlinemap().value

    def getrawvalue(self) -> str:
        buf = StringIO()
        for li in self._lines:
            if isinstance(li, DeferredLineBase):
                line = li()
                if line is None:
                    continue
            elif isinstance(li, LineContext):
                continue
            else:
                line = li
            assert isinstance(line, str)
            # backslash implies line continuation
            if line.endswith("\\"):
                buf.write(line[:-1])
            else:
                buf.write(line)
                buf.write("\n")
        return buf.getvalue()

    def clear(self) -> None:
        self._lines.clear()

    def __bool__(self) -> bool:
        return bool(self._lines)

    def prefix(self) -> str:
        return " " * (self._indent * self.tabwidth)

    def newline(self) -> None:
        self.writeline("\n")

    def writeline(self, line: Union[LineContext, DeferredLineBase, str]) -> None:
        if isinstance(line, LineContext):
            self._lines.append(line)
        elif isinstance(line, DeferredLineBase):
            self._lines.append(line.with_prefix(self.prefix()))
        elif line.strip():
            self._lines.append(f"{self.prefix()}{line}")
        else:
            self._lines.append("")

    def writelines(
        self, lines: Sequence[Union[LineContext, DeferredLineBase, str]]
    ) -> None:
        for line in lines:
            self.writeline(line)

    def indent(self, offset: int = 1) -> contextlib.AbstractContextManager[None]:
        @contextlib.contextmanager
        def ctx() -> Iterator[None]:
            self._indent += offset
            try:
                yield
            finally:
                self._indent -= offset

        return ctx()

    def do_indent(self, offset: int = 1) -> None:
        self._indent += offset

    def do_unindent(self, offset: int = 1) -> None:
        self._indent -= offset

    def splice(
        self, other_code: Union[IndentedBuffer, str], strip: bool = False
    ) -> None:
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
            for s in other_code.split("\n"):
                self.writeline(s)

    def map(self, func: Callable[[Any], Any]) -> IndentedBuffer:
        res = IndentedBuffer(initial_indent=self._indent)
        res._lines = [func(line) for line in self._lines]
        return res

    def __repr__(self) -> str:
        return f"{type(self)}({self.getvalue()})"

    def __add__(self, other: Self) -> IndentedBuffer:
        assert self._indent == other._indent
        res = IndentedBuffer(initial_indent=self._indent)
        # TODO(rec): or should this be self.__class__(initial_indent=self._indent)?
        res.writelines(self._lines)
        res.writelines(other._lines)
        return res


class FakeIndentedBuffer(IndentedBuffer):
    def __init__(self) -> None:
        super().__init__()

    def __getattribute__(self, name: str) -> Any:
        if name == "__class__":  # Allow access to the class attribute
            return object.__getattribute__(self, name)
        raise RuntimeError(
            f"Tried to call self.{name} on FakeIndentedBuffer. This buffer"
            "is currently used on TritonTemplateKernel to prevent actual"
            "writes to the body without explicitly specifying the body with"
            "`TritonTemplateKernel.set_subgraph_body(name)`"
        )


@contextlib.contextmanager
def restore_stdout_stderr() -> Iterator[None]:
    initial_stdout, initial_stderr = sys.stdout, sys.stderr
    try:
        yield
    finally:
        sys.stdout, sys.stderr = initial_stdout, initial_stderr


class DeferredLineBase:
    """A line that can be 'unwritten' at a later time"""

    def __init__(self, line: str):
        if not line.strip():
            line = ""
        self.line = line

    def __call__(self) -> Union[str, None]:
        """Returns either self.line or None to indicate the line has been 'unwritten'"""
        raise NotImplementedError

    def _new_line(self, line: str) -> Self:
        """Returns a new deferred line with the same condition"""
        raise NotImplementedError

    def with_prefix(self, prefix: str) -> Self:
        return self._new_line(f"{prefix}{self.line}")

    def lstrip(self) -> Self:
        return self._new_line(self.line.lstrip())

    def __getitem__(self, index: Union[int, slice]) -> Self:
        return self._new_line(self.line[index])

    def __bool__(self) -> bool:
        return bool(self.line)

    def __len__(self) -> int:
        return len(self.line)


class DelayReplaceLine(DeferredLineBase):
    """At end of codegen call `line.replace(key, value_fn())`"""

    def __init__(self, key: str, value_fn: Callable[[], str], line: str):
        super().__init__(line)
        self.key = key
        self.value_fn = value_fn

    def __call__(self) -> str:
        return self.line.replace(self.key, self.value_fn())

    def _new_line(self, line: str) -> DelayReplaceLine:
        return DelayReplaceLine(self.key, self.value_fn, line)


@functools.lru_cache(None)
def is_big_gpu(index_or_device: Union[int, torch.device] = 0) -> bool:
    if isinstance(index_or_device, torch.device):
        device = index_or_device
    else:
        device = torch.device(get_gpu_type(), index_or_device)

    prop = DeviceProperties.create(device)

    # SM logic is not relevant to ROCm gpus
    # Arbitrarily skipping the older models
    if torch.version.hip:
        assert prop.major is not None
        if prop.major < 9 or prop.major == 10:
            log.warning("GPU arch does not support max_autotune_gemm mode usage")
            return False
        return True

    min_sms = 16 if device.type == "xpu" else 68  # 3080
    avail_sms = prop.multi_processor_count
    if avail_sms < min_sms:
        log.warning(
            "Not enough SMs to use max_autotune_gemm mode",
            extra={"min_sms": min_sms, "avail_sms": avail_sms},
        )
        return False
    return True


@functools.lru_cache
def get_max_num_sms() -> int:
    return torch.cuda.get_device_properties("cuda").multi_processor_count


def get_num_sms() -> int:
    """Handle experimental carveout if set otherwise return hardware SM count"""
    # TODO we need to properly guard on this global
    carveout = torch._C._get_sm_carveout_experimental()
    return get_max_num_sms() - (carveout if carveout is not None else 0)


def get_tma_workspace_arg(
    num_tma_descriptors: int,
    device: torch.device,
    num_programs: Optional[int] = None,
) -> WorkspaceArg:
    """Builds and returns a WorkspaceArg for the device side TMA workspace buffer."""
    from .codegen.common import WorkspaceArg, WorkspaceZeroMode

    if num_programs is None:
        num_programs = get_num_sms()
    zero_mode = WorkspaceZeroMode.from_bool(False)
    size = num_programs * num_tma_descriptors * TMA_DESCRIPTOR_SIZE
    return WorkspaceArg(
        count=size,
        zero_mode=zero_mode,
        device=device,
        outer_name=WorkspaceArg.unique_name(),
    )


def use_max_autotune() -> bool:
    return (
        config.max_autotune or config.max_autotune_gemm or config.search_autotune_cache
    )


def _use_template_for_gpu(
    layout: Layout, allowed_layout_dtypes: list[torch.dtype]
) -> bool:
    if layout.dtype not in allowed_layout_dtypes:
        log.debug(
            "Not using template since dtype %s is not in allowed layout dtypes %s",
            layout.dtype,
            allowed_layout_dtypes,
        )
    return (
        is_gpu(layout.device.type)
        and layout.dtype in allowed_layout_dtypes
        and is_big_gpu(layout.device)
    )


def _use_autotune_backend(backend: str) -> bool:
    return backend.upper() in [
        x.strip() for x in config.max_autotune_gemm_backends.upper().split(",")
    ]


def _use_conv_autotune_backend(backend: str) -> bool:
    return backend.upper() in [
        x.strip() for x in config.max_autotune_conv_backends.upper().split(",")
    ]


def use_triton_template(
    layout: Layout, *, enable_int32: bool = False, enable_float8: bool = False
) -> bool:
    from .codegen.common import BackendFeature, has_backend_feature

    layout_dtypes = [torch.float16, torch.bfloat16, torch.float32]
    if enable_int32:
        layout_dtypes = [torch.float16, torch.bfloat16, torch.float32, torch.int32]
    if enable_float8:
        layout_dtypes.extend([torch.float8_e4m3fn, torch.float8_e5m2])
    return (
        (
            (
                is_gpu(layout.device.type)
                and _use_template_for_gpu(layout, layout_dtypes)
            )
            or (layout.device.type == "cpu" and layout.dtype in layout_dtypes)
        )
        and use_max_autotune()
        and _use_autotune_backend("TRITON")
        and has_backend_feature(layout.device, BackendFeature.TRITON_TEMPLATES)
    )


def use_triton_tma_template(*matrices: IRNode) -> bool:
    from torch.utils._triton import has_triton_tma

    from .virtualized import V

    def _is_tma_compatible(x: IRNode) -> bool:
        if len(x.get_size()) != 2:
            return False

        dtype = x.get_dtype()
        if dtype not in (torch.float16, torch.bfloat16, torch.float8_e4m3fn):
            return False

        layout = x.get_layout()
        transposed = layout.is_transposed()
        if not (layout.is_contiguous() or transposed):
            return False

        inner_dim = layout.size[1]
        if transposed:
            inner_dim = layout.size[0]

        if dtype == torch.float8_e4m3fn and V.graph.sizevars.statically_known_lt(
            inner_dim, 32
        ):
            return False

        inner_bytes = inner_dim * dtype.itemsize
        return V.graph.sizevars.statically_known_multiple_of(inner_bytes, TMA_ALIGNMENT)

    return (
        config.triton.enable_persistent_tma_matmul
        and has_triton_tma()
        and all(_is_tma_compatible(m) for m in matrices)
    )


def use_cutlass_template(layout: Layout, m: int, n: int, k: int) -> bool:
    from .virtualized import V

    gemm_size = V.graph.sizevars.size_hint(m * n * k, fallback=-1)
    if gemm_size <= 0 or gemm_size < config.cuda.cutlass_backend_min_gemm_size:
        return False
    from .codegen.cuda.cutlass_utils import try_import_cutlass

    # Do not use cutlass template on ROCm
    if torch.version.hip:
        return False

    # output dtype
    # FP32 not supported: https://github.com/pytorch/pytorch/issues/145952
    layout_dtypes = [torch.float16, torch.bfloat16, torch.int32]
    res = (
        _use_template_for_gpu(layout, layout_dtypes)
        and use_max_autotune()
        and _use_autotune_backend("CUTLASS")
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


decompose_k_threshold = 32

# To limit compile time
k_splits_limit = 5

# Hand-tuned
default_k_splits = [16, 32, 64, 128, 256]

_IntLike: TypeAlias = Union[int, sympy.Expr]


def use_decompose_k_choice(m: _IntLike, n: _IntLike, k: _IntLike) -> bool:
    from torch._inductor.virtualized import V

    return (
        V.graph.sizevars.is_expr_static_and_true(
            sympy.And(
                sympy.Ge(k, decompose_k_threshold * m),
                sympy.Ge(k, decompose_k_threshold * n),
            )
        )
        and not V.graph.aot_mode  # TODO: Support AOTI for decomposeK
        and not V.graph.cpp_wrapper
    )


@functools.lru_cache(None)
def get_k_splits(m: _IntLike, n: _IntLike, k: _IntLike) -> list[int]:
    # If k is a sympy expression, we can't do any splitting
    if isinstance(k, sympy.Expr) and not k.is_number:
        return default_k_splits

    if (isinstance(m, sympy.Expr) and not m.is_number) or (
        isinstance(n, sympy.Expr) and not n.is_number
    ):
        max_k_split = 256
    else:
        max_k_split = min(k // m, k // n)

    min_k_split = 2
    # Get all divisors of k, k has to be divisible by kPart
    divisors = sympy.divisors(k)

    divisors = [
        divisor
        for divisor in divisors
        if divisor <= max_k_split and divisor >= min_k_split
    ]

    pow_of_2_divisors, mul_of_32_divisors, rest_of_splits = [], [], []

    for d in divisors:
        kPart = k // d

        # Smaller than 128 might not even fit in a single tile, BLOCK_K can be 128
        if kPart < 128:
            continue

        # Power of 2 divisors are best performing, conform to hardware
        if (kPart & kPart - 1) == 0 and kPart >= 128:
            pow_of_2_divisors.append(d)
        # Else check if creates a multiple of 32
        elif kPart % 32 == 0:
            mul_of_32_divisors.append(d)
        # otherwise, take the smallest values
        else:
            rest_of_splits.append(d)

    # If the # of power of 2 divisors are greater than k_splits_limit, return all
    # This should be ok for compile time, all perfect squares between 128 and min(k / m, k / n)
    # should never be a massive amount
    if len(pow_of_2_divisors) >= k_splits_limit:
        return pow_of_2_divisors
    else:
        best_splits = pow_of_2_divisors + mul_of_32_divisors + rest_of_splits
        # Otherwise, conform results to k_splits_limit
        return best_splits[:k_splits_limit]


@functools.lru_cache(None)
def _rocm_native_device_arch_name(device: str) -> str:
    return torch.cuda.get_device_properties(device).gcnArchName


@functools.lru_cache(None)
def try_import_ck_lib() -> tuple[
    Optional[str], Callable[[], list[Any]], Callable[[], list[Any]], type[Any]
]:
    try:
        import ck4inductor  # type: ignore[import]
        from ck4inductor.universal_gemm.gen_instances import (  # type: ignore[import]
            gen_ops_library,
            gen_ops_preselected,
        )
        from ck4inductor.universal_gemm.op import (  # type: ignore[import]
            CKGemmOperation,
        )

        package_dirname = os.path.dirname(ck4inductor.__file__)
    except ImportError:

        def gen_ops_library() -> list[Any]:
            return []

        def gen_ops_preselected() -> list[Any]:
            return []

        class CKGemmOperation:  # type: ignore[no-redef]
            pass

        package_dirname = None
    return package_dirname, gen_ops_library, gen_ops_preselected, CKGemmOperation


def use_ck_template(layout: Layout) -> bool:
    # config knobs check 1
    if not use_max_autotune():
        return False
    # platform check
    if not torch.version.hip:
        return False
    # tensors must be on GPU
    if not layout.device.type == "cuda":
        return False
    # hardware check
    # if config arch list is not specified, get the native arch from the device properties
    native_arch = _rocm_native_device_arch_name(layout.device)
    requested_archs = {k.split(":")[0]: k for k in config.rocm.arch} or {
        native_arch.split(":")[0]: native_arch
    }
    requested_supported_archs = [
        requested_archs[k]
        for k in requested_archs.keys() & config.rocm.ck_supported_arch
    ]
    if not requested_supported_archs:
        return False
    # supported input dtypes
    if layout.dtype not in [torch.float16, torch.bfloat16, torch.float32]:
        return False

    ck_package_dirname, _, _, _ = try_import_ck_lib()

    if not ck_package_dirname:
        log.warning("Please pip install Composable Kernel package")
        return False

    if config.is_fbcode():
        config.rocm.ck_dir = ck_package_dirname

    if not config.rocm.ck_dir:
        log.warning("Please set TORCHINDUCTOR_CK_DIR env variable")
        return False

    if ck_package_dirname != config.rocm.ck_dir:
        log.warning("Invalid path to CK library")
        return False

    return True


def use_ck_gemm_template(layout: Layout, m: int, n: int, k: int) -> bool:
    from .virtualized import V

    return (
        _use_autotune_backend("CK")
        and use_ck_template(layout)
        and V.graph.sizevars.size_hint(m * n * k, fallback=-1) > 0
    )


def use_ck_conv_template(layout: Layout) -> bool:
    return _use_conv_autotune_backend("CK") and use_ck_template(layout)


def _use_template_for_cpu(layout: Layout) -> bool:
    return use_max_autotune() and layout.device.type == "cpu"


def use_cpp_bmm_template(
    layout: Layout, mat1: Union[ReinterpretView, Buffer], mat2: IRNode
) -> bool:
    from .ir import Layout

    assert isinstance(mat1.layout, Layout)

    return (
        use_cpp_gemm_template(layout, mat1, mat2, require_constant_mat2=False)
        and mat1.layout.is_contiguous()
    )


def use_cpp_gemm_template(
    layout: Layout,
    mat1: IRNode,
    mat2: IRNode,
    mat2_transposed: bool = False,
    require_constant_mat2: bool = True,
    is_woq_int4: bool = False,
    q_group_size: Optional[int] = None,
) -> bool:
    from . import ir
    from .codegen.cpp_micro_gemm import create_micro_gemm
    from .codegen.cpp_utils import get_gemm_template_output_and_compute_dtype
    from .kernel.mm_common import mm_args

    if not _use_template_for_cpu(layout) or not _use_autotune_backend("CPP"):
        return False

    if not config.cpp.weight_prepack:
        return False

    int8_gemm = mat1.get_dtype() in [torch.uint8, torch.int8]
    layout_dtypes = [torch.float32, torch.bfloat16, torch.half, torch.uint8]
    m, n, k, layout, mat1, mat2 = mm_args(
        mat1,
        mat2,
        out_dtype=layout.dtype if int8_gemm else None,
        mat2_transposed=mat2_transposed,
        use_4x2_dim=is_woq_int4,
    )

    # TODO(jgong5): support dynamic shapes for n or k
    if has_free_symbols((n, k)):
        return False
    if isinstance(mat2, ir.BaseView):
        mat2 = mat2.unwrap_view()

    output_dtype, _ = get_gemm_template_output_and_compute_dtype(mat1.get_dtype())
    micro_gemm = create_micro_gemm(
        "micro_gemm",
        m,
        n,
        k,
        input_dtype=mat1.get_dtype(),
        input2_dtype=mat2.get_dtype(),
        output_dtype=output_dtype,
        num_threads=parallel_num_threads(),
        use_ref=not is_woq_int4,
        q_group_size=q_group_size,
    )

    def is_last_dim_stride1(x: IRNode) -> bool:
        x.freeze_layout()
        return x.get_stride()[-1] == 1

    return (
        layout.dtype in layout_dtypes
        and micro_gemm is not None
        and is_last_dim_stride1(mat1)  # TODO(jgong5): support transposed input
        and isinstance(mat2, ir.StorageBox)
        and (mat2.is_module_buffer() or not require_constant_mat2)
    )


def use_aten_gemm_kernels() -> bool:
    return not use_max_autotune() or _use_autotune_backend("ATEN")


class DebugDirManager:
    counter = itertools.count(0)
    prev_debug_name: str

    def __init__(self) -> None:
        self.id = next(DebugDirManager.counter)

    def __enter__(self) -> None:
        self.prev_debug_name = torch._dynamo.config.debug_dir_root
        self.new_name = f"{self.prev_debug_name}_tmp_{self.id}"
        torch._dynamo.config.debug_dir_root = self.new_name

    def __exit__(self, *args: Any) -> None:
        shutil.rmtree(self.new_name)
        torch._dynamo.config.debug_dir_root = self.prev_debug_name


def run_and_get_code(
    fn: Callable[P, _T],
    *args: P.args,
    **kwargs: P.kwargs,
) -> tuple[_T, list[str]]:
    from .graph import GraphLowering

    source_codes: list[str] = []

    def save_output_code(code: str) -> None:
        source_codes.append(code)

    with mock.patch.object(GraphLowering, "save_output_code", save_output_code):
        torch._dynamo.reset()
        result = fn(*args, **kwargs)
    return result, source_codes


def run_and_get_kernels(
    fn: Callable[P, _T], *args: P.args, **kwargs: P.kwargs
) -> tuple[_T, list[str]]:
    result, source_codes = run_and_get_code(fn, *args, **kwargs)
    kernels = []
    for code in source_codes:
        kernels.extend(re.findall(r"'''.*?'''", code, re.DOTALL))
    return result, kernels


def run_fw_bw_and_get_code(fn: Callable[..., Any]) -> tuple[Any, list[str]]:
    def run_with_backward() -> Any:
        result = fn()
        result.sum().backward()
        return result

    return run_and_get_code(run_with_backward)


def get_code(fn: Callable[P, _T], *args: P.args, **kwargs: P.kwargs) -> list[str]:
    """Get the inductor-generated code, but skip any actual compilation or running."""
    from .graph import GraphLowering

    source_codes: list[str] = []

    def save_output_code(code: str) -> None:
        source_codes.append(code)

    def patched_compile_to_module(self: GraphLowering) -> Any:
        class DummyModule:
            """This is empty to replace the generated triton module"""

            def __init__(self) -> None:
                pass

            def call(self, *args: Any, **kwargs: Any) -> None:
                # Don't do anything when called
                pass

        wrapper_code, kernel_code = (
            self.codegen_with_cpp_wrapper() if self.cpp_wrapper else self.codegen()
        )
        # Skip all the actual compiling.
        nonlocal save_output_code
        save_output_code(wrapper_code.value)
        if kernel_code:
            save_output_code(kernel_code.value)

        return DummyModule()

    with (
        mock.patch.object(
            GraphLowering, "compile_to_module", patched_compile_to_module
        ),
        mock.patch.object(GraphLowering, "save_output_code", save_output_code),
    ):
        torch._dynamo.reset()
        # Note the return here is None
        _ = fn(*args, **kwargs)

    return source_codes


def get_triton_code(fn: Callable[P, _T], *args: P.args, **kwargs: P.kwargs) -> str:
    source_codes = get_code(fn, *args, **kwargs)
    # Can have two outputs if backwards was eagerly compiled
    assert 1 <= len(source_codes) <= 2, (
        f"expected one or two code outputs got {len(source_codes)}"
    )
    return source_codes[0]


def run_and_get_triton_code(
    fn: Callable[P, _T], *args: P.args, **kwargs: P.kwargs
) -> str:
    _, source_codes = run_and_get_code(fn, *args, **kwargs)
    # Can have two outputs if backwards was eagerly compiled
    assert 1 <= len(source_codes) <= 2, (
        f"expected one or two code outputs got {len(source_codes)}"
    )
    return source_codes[0]


def run_and_get_graph_lowering(
    fn: Callable[P, _T], *args: P.args, **kwargs: P.kwargs
) -> tuple[Any, list[GraphLowering]]:
    from torch._inductor.graph import GraphLowering
    from torch._inductor.output_code import CompiledFxGraph

    real_init = CompiledFxGraph.__init__
    graph_lowerings = []

    def fake_init(*args: Any, **kwargs: Any) -> None:
        real_init(*args, **kwargs)
        graph = args[2]
        assert isinstance(graph, GraphLowering)
        graph_lowerings.append(graph)

    with mock.patch.object(CompiledFxGraph, "__init__", fake_init):
        result = fn(*args, **kwargs)

    return result, graph_lowerings


@contextlib.contextmanager
def override_lowering(
    aten_op: Callable[..., Any], override_fn: Callable[..., Any]
) -> Iterator[None]:
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


def add_scheduler_init_hook(
    pre_fn: Callable[..., Any], post_fn: Optional[Callable[..., Any]] = None
) -> Any:
    """
    Add hook functions to be called at the beginning and end of Scheduler.__init__.
    Used for unit tests.
    """
    from torch._inductor.scheduler import Scheduler

    orig_fn = Scheduler.__init__

    def wrapper(scheduler: Any, nodes: Any) -> Any:
        pre_fn(scheduler, nodes)
        out = orig_fn(scheduler, nodes)
        if post_fn:
            post_fn(scheduler, nodes)
        return out

    return unittest.mock.patch.object(Scheduler, "__init__", wrapper)


def developer_warning(msg: str) -> None:
    """
    Warnings that will be actionable for PyTorch developers, but not
    end users.  Allows us to easily disable them in stable releases but
    keep them on for nightly builds.
    """
    if config.developer_warnings:
        log.warning(msg)
    else:
        log.info(msg)


def get_benchmark_name() -> Optional[str]:
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

    return None


def is_ones(items: Sequence[Any]) -> bool:
    return all(x == 1 for x in items)


def is_zeros(items: Sequence[Any]) -> bool:
    return all(x == 0 for x in items)


def is_cpu_device(inputs: Sequence[torch.Tensor]) -> bool:
    return all(
        item.device == torch.device("cpu")
        for item in inputs
        if isinstance(item, torch.Tensor)
    )


def get_sympy_Expr_dtype(val: sympy.Expr) -> torch.dtype:
    assert isinstance(val, sympy.Expr), (
        "only support sympy.Expr as input to get_sympy_Expr_dtype"
    )
    if val.is_integer:  # type: ignore[attr-defined]
        return torch.int64
    else:
        return torch.float64


@contextlib.contextmanager
def maybe_profile(should_profile: bool, *args: Any, **kwargs: Any) -> Iterator[Any]:
    if should_profile:
        with torch.profiler.profile(*args, **kwargs) as p:
            yield p
    else:
        yield


def parallel_num_threads() -> int:
    threads = config.cpp.threads
    if threads < 1:
        threads = torch.get_num_threads()
    return threads


@functools.lru_cache(None)
def get_backend_num_stages() -> int:
    from .runtime.triton_helpers import get_backend_options

    options = get_backend_options()
    return options.get("num_stages", 2 if torch.version.hip else 3)


@functools.lru_cache(None)
def get_device_tflops(dtype: torch.dtype) -> int:
    from triton.testing import get_max_simd_tflops, get_max_tensorcore_tflops

    assert dtype in (torch.float16, torch.bfloat16, torch.float32)

    if inspect.signature(get_max_simd_tflops).parameters.get("clock_rate"):
        # Triton API change in https://github.com/triton-lang/triton/pull/2293
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
def get_gpu_dram_gbps() -> int:
    from triton.testing import get_dram_gbps

    return get_dram_gbps()


def get_gpu_shared_memory() -> int:
    from triton.runtime import driver

    return driver.active.utils.get_device_properties(0).get("max_shared_mem", 0)


def is_welford_reduction(reduction_type: str) -> bool:
    return reduction_type.startswith("welford")


def reduction_num_outputs(reduction_type: str) -> int:
    if is_welford_reduction(reduction_type):
        return 3
    elif reduction_type == "online_softmax_reduce":
        return 2
    else:
        return 1


def is_linux() -> bool:
    return platform.system() == "Linux"


def is_windows() -> bool:
    return sys.platform == "win32"


def has_free_symbols(itr: Iterable[Any]) -> bool:
    return any(isinstance(x, sympy.Expr) and not x.is_number for x in itr)


def is_dynamic(*args: Any) -> bool:
    from . import ir

    for t in args:
        if isinstance(
            t, (ir.TensorBox, ir.StorageBox, ir.BaseView, ir.ComputedBuffer, ir.Buffer)
        ):
            if has_free_symbols(t.maybe_get_size() or ()) or has_free_symbols(
                t.maybe_get_stride() or ()
            ):
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


def pass_execution_and_save(
    func: Callable[..., Any], gm: GraphModule, inp: Sequence[Any], msg: str
) -> None:
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
        with GraphTransformObserver(gm, msg):
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


def is_multi_outputs_template(input_buf: Optional[Union[Buffer, Operation]]) -> bool:
    """
    Check if input buffer is a multi-outputs template buffer
    """
    from . import ir

    return isinstance(input_buf, ir.CppTemplateBuffer) and isinstance(
        input_buf.layout, ir.MultiOutputLayout
    )


def is_output_of_multi_outputs_template(
    input_buf: Optional[Union[Buffer, Operation]],
) -> bool:
    """
    Check if input buffer is a output of multi-outputs template buffer
    """
    from . import ir

    return (
        isinstance(input_buf, ir.MultiOutput)
        and len(input_buf.inputs) == 1
        and is_multi_outputs_template(input_buf.inputs[0])
    )


def is_collective(
    node: Optional[Union[Node, Operation]],
    op: Optional[torch._ops.OperatorBase] = None,
) -> bool:
    if node is None:
        return False

    from . import ir

    return (
        type(node) == ir._CollectiveKernel and (op is None or node.op_overload is op)
    ) or (
        # TODO: this is a temporary solution to ensure that we can identify torchrec's
        # communication ops. But in order to allow better communication and computation
        # overlap, torchrec's communication ops should be not used.
        type(node) == ir.FallbackKernel
        and (
            # NOTE: the `hasattr()` check is to bypass errors such as the following:
            # AttributeError: '_OpNamespace' 'torchrec' object has no attribute 'all_to_all_single'
            (
                hasattr(torch.ops.torchrec, "all_to_all_single")
                and node.op_overload == torch.ops.torchrec.all_to_all_single.default
            )
            or (
                hasattr(torch.ops.torchrec, "all_gather_into_tensor")
                and node.op_overload
                == torch.ops.torchrec.all_gather_into_tensor.default
            )
            or (
                hasattr(torch.ops.torchrec, "reduce_scatter_tensor")
                and node.op_overload == torch.ops.torchrec.reduce_scatter_tensor.default
            )
        )
    )


def is_wait(node: Optional[Union[IRNode, Operation]]) -> bool:
    from . import ir

    return type(node) == ir._WaitKernel


def contains_collective(snode: BaseSchedulerNode) -> bool:
    from torch._inductor.scheduler import GroupedSchedulerNode

    if isinstance(snode, GroupedSchedulerNode):
        return any(contains_collective(x) for x in snode.snodes)

    return is_collective(snode.node)


def contains_wait(snode: BaseSchedulerNode) -> bool:
    from torch._inductor.scheduler import GroupedSchedulerNode

    if isinstance(snode, GroupedSchedulerNode):
        return any(contains_wait(x) for x in snode.snodes)
    else:
        return is_wait(snode.node)


def is_fallback_op(
    node: Optional[Operation],
    op: Union[torch._ops.OpOverload, Collection[torch._ops.OpOverload]],
) -> bool:
    from . import ir

    if isinstance(op, torch._ops.OpOverload):
        op = [op]
    return isinstance(node, ir.FallbackKernel) and node.op_overload in op


def buf_name_to_fused_snode(
    buf_name: str, name_to_buf: dict[str, Any], name_to_fused_node: dict[str, Any]
) -> Any:
    return name_to_fused_node[name_to_buf[buf_name].defining_op.get_name()]


def find_recursive_deps_of_node(
    snode: BaseSchedulerNode,
    collected_node_set: MutableSet[BaseSchedulerNode],
    name_to_buf: dict[str, SchedulerBuffer],
    name_to_fused_node: dict[str, BaseSchedulerNode],
    criteria_cb: Callable[[Any], bool] = lambda snode: False,
) -> None:
    if criteria_cb(snode):
        return
    collected_node_set.add(snode)
    for dep in snode.unmet_dependencies:
        defining_op_for_dep = buf_name_to_fused_snode(
            dep.name, name_to_buf, name_to_fused_node
        )
        if defining_op_for_dep in collected_node_set:
            continue
        find_recursive_deps_of_node(
            defining_op_for_dep,
            collected_node_set,
            name_to_buf,
            name_to_fused_node,
            criteria_cb=criteria_cb,
        )


def find_recursive_users_of_node(
    snode: BaseSchedulerNode,
    collected_node_set: MutableSet[BaseSchedulerNode],
    name_to_buf: dict[str, SchedulerBuffer],
    name_to_fused_node: dict[str, BaseSchedulerNode],
    criteria_cb: Callable[[Any], bool] = lambda snode: False,
) -> None:
    if criteria_cb(snode):
        return
    collected_node_set.add(snode)
    for o in snode.get_outputs():
        for user in o.users:
            assert user.node is not None
            if user.node.get_name() == "OUTPUT":
                continue
            if user.node.get_name() not in name_to_fused_node:
                continue
            user_op = name_to_fused_node[user.node.get_name()]
            if user_op in collected_node_set:
                continue
            find_recursive_users_of_node(
                user_op,
                collected_node_set,
                name_to_buf,
                name_to_fused_node,
                criteria_cb=criteria_cb,
            )


def num_fw_fixed_arguments(dynamo_gm_num_inputs: int, aot_fw_gm_num_inputs: int) -> int:
    "Computes the number of inputs to the aot fw graph which have fixed addresses (params and buffers)"
    num_rng_seed_offset_inputs = (
        2 if torch._functorch.config.functionalize_rng_ops else 0
    )
    # AOT won't lift any parameters if we're inlining NN Modules
    # however desugaring subclasses will still add arguments
    # resulted in extra fixed inputs https://github.com/pytorch/pytorch/issues/130502
    return aot_fw_gm_num_inputs - dynamo_gm_num_inputs - num_rng_seed_offset_inputs


def count_tangents(fx_g: torch.fx.GraphModule) -> int:
    """
    Infers which inputs are static for a backwards graph
    """

    def is_saved_tensor(x: Node) -> bool:
        return (
            "tangents" not in x.name
            and "bwd_seed" not in x.name
            and "bwd_base_offset" not in x.name
            and "bwd_rng_state" not in x.name
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

    def __bool__(self) -> bool:
        return self.value

    @staticmethod
    def disable(obj: Any) -> Union[BoxedBool, bool]:
        if isinstance(obj, BoxedBool):
            obj.value = False
            return obj
        return False


@contextlib.contextmanager
def collect_defined_kernels(kernel_list: list[str]) -> Iterator[None]:
    from .codegen.wrapper import PythonWrapperCodegen

    orig_define_kernel = PythonWrapperCodegen.define_kernel

    def define_kernel(
        self: PythonWrapperCodegen,
        kernel_name: str,
        kernel_code: str,
        metadata: Optional[str] = None,
        gpu: bool = True,
        cpp_definition: Optional[str] = None,
    ) -> Any:
        kernel_list.append(kernel_code)
        return orig_define_kernel(
            self, kernel_name, kernel_code, metadata, gpu, cpp_definition
        )

    with mock.patch.object(PythonWrapperCodegen, "define_kernel", define_kernel):
        yield


def get_cloned_parameter_buffer_name(name: str) -> str:
    return name + "__original__"


def is_gpu(device: Optional[str]) -> bool:
    return device in GPU_TYPES


def device_need_guard(device: str) -> bool:
    return is_gpu(device)


def needs_fallback_due_to_atomic_add_limitations(dtype: torch.dtype) -> bool:
    # tl.atomic add has bfloat16 support in fbcode
    # but not in OSS https://github.com/pytorch/pytorch/issues/97016
    # we will fallback until the code is upstreamed to OSS
    if (
        config.is_fbcode()
        and dtype == torch.bfloat16
        and torch.cuda.is_available()
        and torch.cuda.get_device_capability() >= (9, 0)
        and config.bfloat16_atomic_adds_enabled
    ):
        return False
    else:
        return dtype in OrderedSet([torch.int64, torch.bool, torch.bfloat16])


def use_scatter_fallback(
    op_overload: torch._ops.OpOverload,
    reduction_type: Optional[str],
    self_dtype: torch.dtype,
    src_dtype: torch.dtype,
    src_device_type: str,
    src_is_tensor: bool,
) -> bool:
    if (
        op_overload.overloadpacket
        in (torch.ops.aten.scatter_reduce_, torch.ops.aten.scatter_reduce)
        and reduction_type is None
    ):
        return False

    reduce_ty = (
        "add" if op_overload.overloadpacket == torch.ops.aten.scatter_ else "sum"
    )

    return (
        reduction_type not in (None, reduce_ty)
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
        or (reduction_type == reduce_ty and self_dtype in (torch.bool, torch.int64))
        or torch.are_deterministic_algorithms_enabled()
    )


def dump_node_schedule(node_schedule: Sequence[BaseSchedulerNode]) -> None:
    """
    An API that can be used in pdb to dump a node_schedule.
    Right mainly dump the read/write dependencies but can add more as needed.
    """
    from torch._inductor.codegen.simd import DisableReduction, EnableReduction
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
                assert node.node is not None
                print(f"original reduction hint {node.node.data.reduction_hint}")  # type: ignore[attr-defined]
            print("ReadDep:")
            for dep in node.read_writes.reads:
                print(dep)
            print("WriteDep:")
            for dep in node.read_writes.writes:
                print(dep)
        else:
            raise RuntimeError(f"Unrecognized node type: {type(node)}")


def tensor_is_aligned(tensor: torch.Tensor) -> bool:
    # See Note: [Input Alignment handling in Inductor]
    # Right now, we don't try to guard on the alignment of the storage offset.
    # When this comment was written, non-symbolic storage_offsets are not guarded on
    # but symbolic storage_offsets are. For consistency, we suppress guard creation
    # upon performing this check: that ensures that we don't add recompiles when we
    # add this logic.
    from torch.fx.experimental.symbolic_shapes import statically_known_true

    return statically_known_true(
        (tensor.storage_offset() * get_dtype_size(tensor.dtype)) % GPU_ALIGN_BYTES == 0
    )


def should_assume_input_aligned(example_input: torch.Tensor) -> bool:
    # See Note: [Input Alignment handling in Inductor]

    # right now, we only care about alignment for cuda tensors.
    if not is_gpu(example_input.device.type):
        return False
    return config.assume_aligned_inputs or tensor_is_aligned(example_input)


def maybe_get_suppress_shape_guards_ctx() -> contextlib.AbstractContextManager[None]:
    # Try to get TracingContext.try_get().fake_mode.shape_env.suppress_guards()
    # If it's not available, return a nullcontext.

    # If we're dealing with cudagraphs, we might not have a tracing_context
    tracing_context = torch._guards.TracingContext.try_get()
    if not tracing_context:
        return contextlib.nullcontext()

    # In standalone inductor compile mode, we might not have a shape_env attached to the fake mode
    shape_env = tracing_context.fake_mode.shape_env
    if not shape_env:
        return contextlib.nullcontext()

    return shape_env.suppress_guards()


def run_and_get_cpp_code(
    fn: Callable[P, _T], *args: P.args, **kwargs: P.kwargs
) -> tuple[_T, str]:
    # We use the patch context manager instead of using it as a decorator.
    # In this way, we can ensure that the attribute is patched and unpatched correctly
    # even if this run_and_get_cpp_code function is called multiple times.
    with unittest.mock.patch.object(config, "debug", True):
        torch._dynamo.reset()
        import io
        import logging

        log_capture_string = io.StringIO()
        ch = logging.StreamHandler(log_capture_string)
        from torch._inductor.codecache import output_code_log

        output_code_log.addHandler(ch)
        prev_level = output_code_log.level
        output_code_log.setLevel(logging.DEBUG)
        result = fn(*args, **kwargs)
        s = log_capture_string.getvalue()
        output_code_log.setLevel(prev_level)
        output_code_log.removeHandler(ch)
    return result, s


def shape_env_from_inputs(inputs: Sequence[InputType]) -> Optional[ShapeEnv]:
    fake_mode = detect_fake_mode(inputs)

    # TODO(voz): It would be nice to enable this assert, but there are lots of tests that
    # pass in real inputs for now.
    # if len(inputs) > 0:
    # assert fake_mode is not None, breakpoint()

    if fake_mode is not None:
        return fake_mode.shape_env

    # When there are no tensor inputs, get shape_env from the first SymInt.
    for input in inputs:
        if isinstance(input, torch.SymInt):
            return input.node.shape_env

    # TODO(voz): Should we always have one anyway?
    return None


def align_inputs_from_check_idxs(
    model: Callable[[list[InputType]], _T],
    inputs_to_check: Sequence[int],
) -> Callable[[list[InputType]], _T]:
    if len(inputs_to_check) == 0:
        return model

    def run(new_inputs: list[InputType]) -> Any:
        copy_misaligned_inputs(new_inputs, inputs_to_check)
        return model(new_inputs)

    return run


def clone_preserve_strides(x: torch.Tensor) -> torch.Tensor:
    if 0 in x.size():
        # Short-circuits if the shape has no elements
        needed_size = 0
    else:
        needed_size = (
            sum((shape - 1) * stride for shape, stride in zip(x.size(), x.stride())) + 1
        )
    buffer = torch.as_strided(x, (needed_size,), (1,)).clone()
    return torch.as_strided(buffer, x.size(), x.stride())


def copy_misaligned_inputs(
    new_inputs: list[InputType], check_inputs_idxs: Sequence[int]
) -> None:
    for i in check_inputs_idxs:
        _inp = new_inputs[i]
        assert isinstance(_inp, torch.Tensor)
        if _inp.data_ptr() % ALIGNMENT:
            new_inputs[i] = clone_preserve_strides(_inp)


def remove_unaligned_input_idxs(
    inputs: Sequence[InputType],
    static_input_idxs: Sequence[int],
) -> Sequence[int]:
    """
    We require all inputs to be aligned, so introduce a copy for any
    that aren't.
    """
    aligned_static_input_idxs = []
    for idx in static_input_idxs:
        input = inputs[idx]
        if isinstance(input, torch.Tensor) and (input.data_ptr() % ALIGNMENT) == 0:
            aligned_static_input_idxs.append(idx)
    if len(aligned_static_input_idxs) != len(static_input_idxs):
        return aligned_static_input_idxs
    return static_input_idxs


def expr_fits_within_32bit(e: sympy.Expr) -> bool:
    from .virtualized import V

    int_max = torch.iinfo(torch.int32).max
    size_hint = V.graph.sizevars.size_hint
    has_hint = V.graph.sizevars.shape_env.has_hint

    # Allow for unhinted e as long as we can still statically prove
    # (e.g., via ValueRanges) that it is still in bounds
    if V.graph.sizevars.is_expr_static_and_true(e <= int_max):
        return True
    # Otherwise, the hint MUST exist and be in range
    return has_hint(e) and size_hint(e) <= int_max


def set_tracing_context_output_strides(
    example_inputs: Sequence[Any], compiled_graph: CompiledFxGraph
) -> None:
    # Return the output strides to the caller via TracingContext
    context = torch._guards.TracingContext.try_get()
    if context is not None and context.output_strides is not None:
        assert len(context.output_strides) == 0
        shape_env = shape_env_from_inputs(example_inputs)
        assert compiled_graph.output_strides is not None
        for exprs in compiled_graph.output_strides:
            if exprs is None:
                context.output_strides.append(None)
            else:
                fakify_first_call = False
                if ctx := torch._guards.TracingContext.try_get():
                    fakify_first_call = ctx.fakify_first_call

                def map_expr(e: Any) -> Union[float, int, SymInt, SymFloat, SymBool]:
                    if shape_env is None:
                        return int(e)
                    if fakify_first_call:
                        return shape_env.deserialize_symexpr(e)
                    return shape_env.evaluate_symexpr(e)

                context.output_strides.append(
                    tuple(map_expr(e) for e in exprs)  # type: ignore[misc]
                )


def should_use_remote_fx_graph_cache() -> bool:
    if config.fx_graph_remote_cache is not None:
        return config.fx_graph_remote_cache
    if not config.is_fbcode():
        return False

    if torch._utils_internal.is_fb_unit_test():
        return False

    try:
        from torch._inductor.fb.remote_cache import REMOTE_CACHE_VERSION
    except ModuleNotFoundError:
        return False

    return REMOTE_CACHE_VERSION >= torch._utils_internal.justknobs_getval_int(
        "pytorch/remote_cache:fx_graph_memcache_version"
    )


def normalize_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]", "_", name)


# correct cases where Triton types names don't match PyTorch
_triton_type_mapping = {
    "tl.bool": "tl.int1",
    "tl.float8_e4m3fn": "tl.float8e4nv",
    "tl.float8_e5m2": "tl.float8e5",
    "tl.float8_e4m3fnuz": "tl.float8e4b8",
    "tl.float8_e5m2fnuz": "tl.float8e5b16",
    # TODO: remove when support is added in triton
    # https://github.com/triton-lang/triton/issues/6054
    "tl.float8_e8m0fnu": "tl.uint8",
}
_torch_triton_mapping = {v: k for k, v in _triton_type_mapping.items()}


_triton_type_re = re.compile(r"^.*[.]")


def triton_type(dtype: torch.dtype) -> str:
    """Convert torch.dtype to triton type"""
    triton_type_name = _triton_type_re.sub("tl.", str(dtype))
    return _triton_type_mapping.get(triton_type_name, triton_type_name)


def triton_type_to_torch(dtype: str) -> torch.dtype:
    adjusted_type = _torch_triton_mapping.get(dtype, dtype)
    type_name = adjusted_type.replace("tl.", "")
    out_dtype = getattr(torch, type_name)
    assert isinstance(out_dtype, torch.dtype)
    return out_dtype


def is_same_tensor(data: torch.Tensor, value: torch.Tensor) -> bool:
    return (
        not data.is_mkldnn
        and data.size() == value.size()
        and data.stride() == value.stride()
        and data.dtype == value.dtype
        and data.device == value.device
        and data.untyped_storage().data_ptr() == value.untyped_storage().data_ptr()
        and data.storage_offset() == value.storage_offset()
    )


def is_same_mkldnn_tensor(data: torch.Tensor, value: torch.Tensor) -> bool:
    return (
        data.is_mkldnn
        and data.size() == value.size()
        and data.dtype == value.dtype
        and data.device == value.device
        and torch.ops.mkldnn.data_ptr(data) == torch.ops.mkldnn.data_ptr(value)
    )


@functools.lru_cache(None)
def boolean_ops() -> tuple[str, ...]:
    return (
        "isinf",
        "isnan",
        "logical_not",
        "logical_and",
        "signbit",
        "and_",
        "le",
        "lt",
        "ge",
        "gt",
        "eq",
        "ne",
        "or_",  # TODO should remove this op
        "xor",
    )


@dataclasses.dataclass
class OpDtypeRule:
    type_promotion_kind: ELEMENTWISE_TYPE_PROMOTION_KIND
    override_return_dtype: Optional[torch.dtype]


op_dtype_propagation_rules: dict[str, OpDtypeRule] = {}


def register_op_dtype_propagation_rules(
    name: str,
    type_promotion_kind: ELEMENTWISE_TYPE_PROMOTION_KIND,
    override_return_dtype: Optional[torch.dtype],
) -> None:
    op_dtype_propagation_rules[name] = OpDtypeRule(
        type_promotion_kind, override_return_dtype
    )


op_requires_libdevice_fp64: OrderedSet[str] = OrderedSet()


def register_op_requires_libdevice_fp64(name: str) -> None:
    op_requires_libdevice_fp64.add(name)


def get_current_backend() -> str:
    from torch._inductor.virtualized import V

    device_str = V.graph.get_current_device_or_throw().type
    if device_str == "cpu":
        return config.cpu_backend
    elif device_str == "mps":
        return "mps"
    else:
        return config.cuda_backend


def upcast_compute_type(dtype: torch.dtype) -> torch.dtype:
    """Maybe upcast [b]float16 to float32"""
    if (
        dtype in (torch.float16, torch.bfloat16)
        and config.triton.codegen_upcast_to_fp32
        and get_current_backend() == "triton"
    ):
        return torch.float32
    return dtype


KeyType = TypeVar("KeyType")
ValType = TypeVar("ValType")


class ScopedDict(MutableMapping[KeyType, ValType]):
    """
    A dictionary-like object that allows for scoped updates. It maintains
    an original dictionary and a set of new items that can override
    the original items within the scope.  The original dictionary is
    unmodified.
    """

    def __init__(self, original_dict: Mapping[KeyType, ValType]):
        self.original_dict = original_dict
        self.new_items: dict[KeyType, ValType] = {}

    def __getitem__(self, key: KeyType) -> ValType:
        if key in self.new_items:
            return self.new_items[key]
        return self.original_dict[key]

    def __setitem__(self, key: KeyType, value: ValType) -> None:
        self.new_items[key] = value

    def __contains__(self, key: object) -> bool:
        return key in self.new_items or key in self.original_dict

    def get(self, key: KeyType, default: Optional[ValType] = None) -> Optional[ValType]:  # type: ignore[override]
        if key in self.new_items:
            return self.new_items[key]
        return self.original_dict.get(key, default)

    def __len__(self) -> int:
        n = len(self.original_dict)
        for k in self.new_items:
            if k not in self.original_dict:
                n += 1
        return n

    def __iter__(self) -> Iterator[KeyType]:
        yield from self.original_dict
        for k in self.new_items:
            if k not in self.original_dict:
                yield k

    def __bool__(self) -> bool:
        return bool(self.original_dict or self.new_items)

    def __delitem__(self, key: KeyType) -> None:
        raise NotImplementedError


@dataclass_transform(frozen_default=True)
def ir_dataclass(cls: Optional[type[Any]] = None, /, *, frozen: bool = True) -> Any:
    def wrap(cls: _T) -> _T:
        if sys.version_info >= (3, 10):
            return dataclasses.dataclass(cls, kw_only=True, frozen=frozen)  # type: ignore[call-overload]
        else:
            # Polyfill for python=3.9. kw_only simply introduces an extra check
            # that only kwargs are used (and is not available on 3.9)
            return dataclasses.dataclass(cls, frozen=frozen)

    if cls is None:
        return wrap
    return wrap(cls)


def get_donated_idxs() -> Optional[list[int]]:
    tracing_context = torch._guards.TracingContext.try_get()
    if tracing_context is not None and tracing_context.fw_metadata:
        return tracing_context.fw_metadata.bw_donated_idxs
    return None


def set_kernel_post_grad_provenance_tracing(
    node_schedule: Union[Sequence[BaseSchedulerNode], ExternKernelOut],
    kernel_name: str,
    is_extern: bool = False,
) -> None:
    from .codegen.simd_kernel_features import DisableReduction, EnableReduction
    from .ir import ExternKernelOut
    from .virtualized import V

    if is_extern:
        assert isinstance(node_schedule, ExternKernelOut)
        curr_node_info = (
            V.debug._inductor_triton_kernel_to_post_grad_node_info.setdefault(
                kernel_name, []
            )
        )
        curr_node_info.extend(
            origin.name
            for origin in node_schedule.origins
            if origin.name not in curr_node_info
        )
    else:
        assert isinstance(node_schedule, list)
        for snode in node_schedule:
            if snode not in (EnableReduction, DisableReduction):
                if snode.node is not None:
                    curr_node_info = V.debug._inductor_triton_kernel_to_post_grad_node_info.setdefault(
                        kernel_name, []
                    )
                    curr_node_info.extend(
                        origin.name
                        for origin in snode.node.origins
                        if origin.name not in curr_node_info
                    )


class TritonAttrsDescriptorVersion(enum.Enum):
    V0_NO_TRITON = 0
    V1_COMPILER = 1  # triton.compiler.compiler.AttrsDescriptor
    V2_BACKENDS = 2  # triton.backends.compiler.AttrsDescriptor
    V3_BACKENDS_TUPLE = (
        3  # triton.backends.compiler.AttrsDescriptor, but with tuple support
    )
    V4_DICT = 4  # a raw dict


@functools.lru_cache(None)
def get_triton_attrs_descriptor_version() -> TritonAttrsDescriptorVersion:
    if importlib.util.find_spec("triton") is None:
        return TritonAttrsDescriptorVersion.V0_NO_TRITON

    import triton.backends.compiler
    import triton.compiler.compiler

    if hasattr(triton.backends.compiler, "AttrsDescriptor"):
        # Triton 3.2.0
        # AttrsDescriptor was moved from triton.compiler.compiler to triton.backends.compiler.
        # AttrsDescriptor and its serialization format were also changed.

        # TODO: implement V3_BACKENDS_TUPLE
        # On Dec 9, 2024, tuple support (triton #5220) was implemented and breaks handling.
        # We don't have a way to detect this (and haven't implemented this version)
        return TritonAttrsDescriptorVersion.V2_BACKENDS
    elif hasattr(triton.compiler.compiler, "AttrsDescriptor"):
        # Triton 3.0.0
        return TritonAttrsDescriptorVersion.V1_COMPILER
    else:
        # After Jan 1, 2025
        # AttrsDescriptor was removed and replaced with a raw dict.
        return TritonAttrsDescriptorVersion.V4_DICT


def triton_version_uses_attrs_dict() -> bool:
    return get_triton_attrs_descriptor_version() == TritonAttrsDescriptorVersion.V4_DICT


def is_cudagraph_unsafe_op(node: Operation) -> bool:
    """
    Returns True if the node is an op that is not cudagraphable.
    Usually only custom ops have this tag.
    """
    from . import ir

    if not isinstance(node, ir.FallbackKernel):
        return False

    if (
        isinstance(node.op_overload, torch._ops.OpOverload)
        and torch._C.Tag.cudagraph_unsafe in node.op_overload.tags
    ):
        return True

    return False


def get_ld_library_path() -> str:
    path = os.environ.get("LD_LIBRARY_PATH", "")
    if config.is_fbcode():
        from libfb.py.parutil import get_runtime_path

        runtime_path = get_runtime_path()
        if runtime_path:
            lib_path = os.path.join(runtime_path, "runtime", "lib")
            path = os.pathsep.join([lib_path, path]) if path else lib_path

    return path
