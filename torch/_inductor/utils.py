import collections
import contextlib
import functools
import inspect
import itertools
import logging
import math
import operator
import os
import shutil
import sys
import tempfile
import textwrap
import time
import unittest
from io import StringIO
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Set,
    TypeVar,
    Union,
    ValuesView,
)
from unittest import mock

import sympy

import torch
from torch.fx.immutable_collections import immutable_dict, immutable_list
from torch.utils._sympy.functions import CleanDiv, FloorDiv, ModularIndexing

from . import config
from .cuda_properties import current_device, get_device_capability

log = logging.getLogger(__name__)

_T = TypeVar("_T")
VarRanges = Dict[sympy.Expr, sympy.Expr]


def do_bench(*args, **kwargs):
    @functools.lru_cache(None)
    def load_triton():
        try:
            # NB: Lazily load triton, as importing triton is slow
            # see https://github.com/openai/triton/issues/1599
            from triton.testing import do_bench as triton_do_bench
        except ImportError:
            raise NotImplementedError("requires Triton")

        # triton PR https://github.com/openai/triton/pull/1513 change the
        # quantile fields name from 'percentiles' to 'quantiles'
        # and change the default value from (0.5, 0.2, 0.8) to None.
        # This may break inductor since a caller expects a tuple may get a item.
        #
        # Add a wrapper to maintain the same behavior for inductor.
        # Maybe we should have own implementation of this function?
        return triton_do_bench, (
            "quantiles"
            if inspect.signature(triton_do_bench).parameters.get("quantiles")
            is not None
            else "percentiles"
        )

    triton_do_bench, quantile_field_name = load_triton()

    if quantile_field_name not in kwargs:
        kwargs[quantile_field_name] = (0.5, 0.2, 0.8)
    return triton_do_bench(*args, **kwargs)[0]


@functools.lru_cache(None)
def has_triton() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        import triton

        return triton is not None and get_device_capability() >= (7, 0)
    except ImportError:
        return False


@functools.lru_cache(None)
def has_torchvision_roi_align() -> bool:
    try:
        from torchvision.ops import roi_align  # noqa: F401

        return roi_align is not None and hasattr(
            getattr(torch.ops, "torchvision", None), "roi_align"
        )
    except ImportError:
        return False


def conditional_product(*args):
    return functools.reduce(operator.mul, [x for x in args if x])


def decode_device(device: Union[Optional[torch.device], str]) -> torch.device:
    if device is None:
        return torch.tensor(0.0).device  # default device
    if isinstance(device, str):
        device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda", index=current_device())
    return device


def sympy_product(it):
    return functools.reduce(operator.mul, it, sympy.Integer(1))


def sympy_dot(seq1, seq2):
    assert len(seq1) == len(seq2)
    return sympy.expand(sum(a * b for a, b in zip(seq1, seq2)))


def unique(it: Iterable[_T]) -> ValuesView[_T]:
    return {id(x): x for x in it}.values()


def ceildiv(numer: int, denom: int) -> int:
    # TODO: There is a bug in a call to this function, to repro:
    # python benchmarks/dynamo/huggingface.py --inductor -d cuda --accuracy
    # --amp --only YituTechConvBert --dynamic-shapes
    assert isinstance(numer, int) and isinstance(
        denom, int
    ), f"{numer}: {type(numer)}, {denom}: {type(denom)}"
    return -(numer // -denom)


def next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 greater than or equal to n"""
    assert n <= 2**32, "32-bit only"
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n += 1
    return n


def convert_shape_to_inductor(lst: List[Union[int, torch.SymInt]]) -> List[sympy.Expr]:
    """
    Gets the shape and stride of a tensor. For non-symbolic tensors, this is
    trivial. But for symbolic tensors, we need to map from SymIntNode into
    sympy.Expr.
    """
    return [
        i.node.expr if isinstance(i, torch.SymInt) else sympy.Integer(i) for i in lst
    ]


def convert_shape_to_symint(
    lst: List[Union[int, sympy.Expr]]
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


def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(model: Callable[..., Any], example_inputs, times: int = 1) -> float:
    synchronize()
    torch.manual_seed(1337)
    t0 = time.perf_counter()
    for _ in range(times):
        result = model(*example_inputs)
        synchronize()
    t1 = time.perf_counter()
    # GC the result after timing
    assert result is not None
    return t1 - t0


def print_performance(fn, args=(), times=10, repeat=10, baseline=1.0):
    timings = torch.tensor([timed(fn, args, times) for _ in range(repeat)])
    took = torch.median(timings)
    print(f"{took/baseline:.6f}")
    return took


def freeze_inputs(f):
    """
    Useful for wrapping lists in tuples for caching purposes
    """

    def freeze_value(x):
        if isinstance(x, (immutable_dict, immutable_list)):
            return x
        if isinstance(x, list):
            return immutable_list(x)
        if isinstance(x, dict):
            return immutable_dict(x)
        return x

    @functools.wraps(f)
    def wrapped(*args):
        args = [freeze_value(x) for x in args]
        return f(*args)

    wrapped.cache_info = f.cache_info  # type: ignore[attr-defined]
    return wrapped


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


def cache_on_self(fn):
    key = f"__{fn.__name__}_cache"

    @functools.wraps(fn)
    def wrapper(self):
        if not hasattr(self, key):
            setattr(self, key, fn(self))
        return getattr(self, key)

    return wrapper


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
            if origin.op == "call_function" and "original_aten" in origin.meta
        ]
        sources = sorted(set(sources))
    elif descriptive_names == "torch":
        # Bases the kernel name off of the top-level "torch" operator (i.e. post-dynamo graph)
        sources = []
        for origin in all_origins:
            if origin.op == "call_function" and "source_fn" in origin.meta:
                if isinstance(origin.meta["source_fn"][1], str):
                    sources.append(origin.meta["source_fn"][1])
                else:
                    sources.append(origin.meta["source_fn"][1].__name__)
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
        if "original_aten" in node.meta:
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


def sympy_symbol(name: str) -> sympy.Symbol:
    # This should never be used for creating shape/stride symbols, as those
    # should all be allocated before Inductor.
    assert name[0] != "s"
    # NOTE: shape symbols are positive (> 0), but index variables are only
    # non-negative (>= 0).
    return sympy.Symbol(name, integer=True, nonnegative=True)


def sympy_subs(expr: sympy.Expr, replacements: Dict[Any, Any]) -> sympy.Expr:
    """
    xreplace is faster than subs, but is way more picky
    """

    def promote_strings(key):
        if isinstance(key, str):
            return sympy_symbol(key)
        return key

    return expr.xreplace(
        {promote_strings(k): promote_strings(v) for k, v in replacements.items()}
    )


def free_symbol_startswith(index: sympy.Expr, prefix: str):
    return any(v.name.startswith(prefix) for v in index.free_symbols)


def free_symbol_has(index: sympy.Expr, pattern: str):
    return any(pattern in v.name for v in index.free_symbols)


def has_incompatible_cudagraph_ops(gm):
    forbidden_set = {
        "aten._fused_moving_avg_obs_fq_helper.default",
        "aten._fused_moving_avg_obs_fq_helper_functional.default",
        "fbgemm.dense_to_jagged.default",
        "fbgemm.jagged_to_padded_dense.default",
        "run_with_rng_state",
        "run_and_save_rng_state",
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
    return False


instance_descriptor = collections.namedtuple(
    "instance_descriptor", ["divisible_by_16", "equal_to_1"]
)


@contextlib.contextmanager
def fresh_inductor_cache(cache_entries=None):
    """
    Contextmanager that provides a clean tmp cachedir for inductor.

    Optionally, pass a dict as 'cache_entries' to get a list of filenames and sizes
    generated with this cache instance.
    """
    with tempfile.TemporaryDirectory() as inductor_cache_dir:
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

    def getvaluewithlinemap(self):
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

    def getvalue(self):
        v, _ = self.getvaluewithlinemap()
        return v

    def getrawvalue(self):
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


class DeferredLineBase:
    """A line that can be 'unwritten' at a later time"""

    def __init__(self, line):
        if not line.strip():
            line = ""
        self.line = line

    def __call__(self) -> Optional[str]:
        """Returns either self.line or None to indicate the line has been 'unwritten'"""
        raise NotImplementedError()

    def _new_line(self, line: str) -> "DeferredLineBase":
        """Returns a new deferred line with the same condition"""
        raise NotImplementedError()

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
def is_big_gpu(index):
    sms = torch.cuda.get_device_properties(index).multi_processor_count
    if sms < 80:  # V100
        log.warning("not enough SMs to use max_autotune_gemm mode")
        return False
    return True


def use_triton_template(layout, *, enable_int32=False):
    layout_dtypes = [torch.float16, torch.bfloat16, torch.float32]
    if enable_int32:
        layout_dtypes = [torch.float16, torch.bfloat16, torch.float32, torch.int32]
    return (
        (
            config.max_autotune
            or config.max_autotune_gemm
            or config.search_autotune_cache
        )
        and "TRITON" in config.max_autotune_gemm_backends.upper().split(",")
        and layout.device.type == "cuda"
        and layout.dtype in layout_dtypes
        and is_big_gpu(layout.device.index or 0)
    )


def use_aten_gemm_kernels():
    return "ATEN" in config.max_autotune_gemm_backends.upper().split(",")


class DebugDirManager:
    counter = itertools.count(0)

    def __init__(self):
        self.id = next(DebugDirManager.counter)
        self.prev_debug_name = None

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
    source_codes = []

    def patched_compile_to_module(self):
        mod = compile_to_module(self)
        with open(mod.__file__) as f:
            source_codes.append(f.read())
        return mod

    with mock.patch.object(
        GraphLowering, "compile_to_module", patched_compile_to_module
    ):
        torch._dynamo.reset()
        result = fn(*args, **kwargs)
    return result, source_codes


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
    Override the lowering of aten_op with overide_fn.
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


def get_num_bytes(*args: torch.Tensor, num_in_out_args: int = 0) -> int:
    """
    Return the total number of bytes the arguments of tensor type takes.

    For in/out args, tensor sizes are counted twice: once for reading and
    once for writing.

    The first num_in_out_args arguments are in out tensors.
    """
    return sum(
        arg.numel() * arg.element_size() * (1 + int(i < num_in_out_args))
        for i, arg in enumerate(args)
        if isinstance(arg, torch.Tensor)
    )


def create_bandwidth_info_str(ms, num_gb, gb_per_s, prefix="", suffix=""):
    info_str = f"{prefix}{ms:.3f}ms    \t{num_gb:.3f} GB \t {gb_per_s:7.2f}GB/s{suffix}"
    try:
        import colorama  # type: ignore[import]

        if ms > 0.012 and gb_per_s < 650:
            info_str = colorama.Fore.RED + info_str + colorama.Fore.RESET
    except ImportError:
        log.warning("Colorama is not installed. Install it if you want colored output")

    return info_str


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
    if val.is_integer:
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


def triton_config_to_hashable(cfg):
    """
    Convert triton config to a tuple that can uniquely identify it. We can use
    the return value as a dictionary key.
    """
    items = sorted(cfg.kwargs.items())
    items.append(("num_warps", cfg.num_warps))
    items.append(("num_stages", cfg.num_stages))
    return tuple(items)


HAS_COLORAMA = True
try:
    import colorama
except ImportError:
    HAS_COLORAMA = False


def _color_text(msg, color):
    if not HAS_COLORAMA:
        return msg

    return getattr(colorama.Fore, color.upper()) + msg + colorama.Fore.RESET


def green_text(msg):
    return _color_text(msg, "green")


def yellow_text(msg):
    return _color_text(msg, "yellow")


def red_text(msg):
    return _color_text(msg, "red")


def blue_text(msg):
    return _color_text(msg, "blue")


PYTHON_TYPE_TO_SCHEMA_TYPE = {
    torch.dtype: "int",
    torch.device: "Device",
    bool: "bool",
}


def may_get_optional_schema_type(schema_type, is_optional_arg):
    return f"Optional[{schema_type}]" if is_optional_arg else schema_type


def type_match(arg, arg_type, is_optional_arg):
    if isinstance(arg, immutable_list):
        if all(
            isinstance(x, int) or (isinstance(x, sympy.Symbol) and x.is_integer)
            for x in arg
        ):
            may_optional_schema_type = may_get_optional_schema_type(
                "List[int]", is_optional_arg
            )
            return may_optional_schema_type == str(arg_type)
        else:
            # TODO: add support here
            return False

    if arg.__class__ in PYTHON_TYPE_TO_SCHEMA_TYPE:
        schema_type = PYTHON_TYPE_TO_SCHEMA_TYPE[arg.__class__]
        may_optional_schema_type = may_get_optional_schema_type(
            schema_type, is_optional_arg
        )
        return may_optional_schema_type == str(arg_type)

    # TODO: add support here
    return False


# torch/csrc/utils/python_arg_parser.cpp:FunctionSignature::parse
def schema_match(schema, args, kwargs):
    min_args = 0
    max_pos_args = 0
    for argument in schema.arguments:
        if not argument.has_default_value():
            min_args += 1
        if not argument.kwarg_only:
            max_pos_args += 1

    nargs = len(args)
    remaining_kwargs = len(kwargs)
    arg_pos = 0

    def args_error_message(nargs, max_pos_args, min_args):
        if min_args != max_pos_args:
            return f"takes from {min_args} to {max_pos_args} positional arguments but {nargs} were given"
        else:
            return f"takes {max_pos_args} positional arguments but {nargs} were given"

    def is_optional(arg):
        return "Optional" in str(arg.type)

    assert len(args) <= max_pos_args, args_error_message(
        len(args), max_pos_args, min_args
    )

    for argument in schema.arguments:
        obj = None
        is_kwd = False
        if arg_pos < nargs:
            if argument.kwarg_only:
                return False
            obj = args[arg_pos]
        elif kwargs:
            if argument.name in kwargs:
                obj = kwargs[argument.name]
                is_kwd = True

        if obj is None and not is_optional(argument):
            return False

        if obj is not None:
            expected_type = argument.type
            if not type_match(obj, expected_type, is_optional(argument)):
                return False

        if not is_kwd:
            arg_pos += 1
        elif (obj is None and is_optional(argument)) or obj is not None:
            remaining_kwargs -= 1

    if remaining_kwargs > 0:
        return False

    return True


def try_find_schema(schemas, args, kwargs):
    for schema in schemas:
        if schema_match(schema, args, kwargs):
            return schema

    return None


def get_device_tflops(dtype):
    from triton.testing import get_max_simd_tflops, get_max_tensorcore_tflops

    assert dtype in (torch.float16, torch.bfloat16, torch.float32)
    if dtype in (torch.float16, torch.bfloat16):
        return get_max_tensorcore_tflops(dtype)

    if torch.backends.cuda.matmul.allow_tf32:
        return get_max_tensorcore_tflops(torch.float32)
    else:
        return get_max_simd_tflops(torch.float32)


def get_gpu_dram_gbps():
    from triton.testing import get_dram_gbps

    return get_dram_gbps()


def is_welford_reduction(reduction_type):
    return reduction_type.startswith("welford")


def reduction_num_outputs(reduction_type):
    return 3 if is_welford_reduction(reduction_type) else 1
