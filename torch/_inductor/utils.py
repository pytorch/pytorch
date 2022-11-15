import collections
import contextlib
import functools
import operator
import os
import tempfile
import time
from importlib import import_module
from typing import Any, Dict, List
from unittest import mock

import numpy as np
import sympy

import torch
from torch.fx.immutable_collections import immutable_dict, immutable_list

from . import config
from .cuda_properties import get_device_capability

VarRanges = Dict[sympy.Expr, sympy.Expr]

# We import torchdynamo modules indirectly to allow a future rename to torch.dynamo
dynamo_config = import_module(f"{config.dynamo_import}.config")
dynamo_debug_utils = import_module(f"{config.dynamo_import}.debug_utils")
dynamo_logging = import_module(f"{config.dynamo_import}.logging")
dynamo_optimizations = import_module(f"{config.dynamo_import}.optimizations")
dynamo_testing = import_module(f"{config.dynamo_import}.testing")
dynamo_utils = import_module(f"{config.dynamo_import}.utils")


@functools.lru_cache(None)
def has_triton():
    if not torch.cuda.is_available():
        return False
    try:
        import triton

        return triton is not None and get_device_capability() >= (7, 0)
    except ImportError:
        return False


@functools.lru_cache(None)
def has_torchvision_roi_align():
    try:
        from torchvision.ops import roi_align  # noqa: F401

        return roi_align is not None and hasattr(
            getattr(torch.ops, "torchvision", None), "roi_align"
        )
    except ImportError:
        return False


def conditional_product(*args):
    return functools.reduce(operator.mul, [x for x in args if x])


def do_bench(
    fn,
    warmup=25,
    rep=100,
    grad_to_none=None,
    percentiles=(0.5, 0.2, 0.8),
    record_clocks=False,
    fast_flush=False,
):
    """
    Benchmark the runtime of the provided function. By default, return the median runtime of :code:`fn` along with
    the 20-th and 80-th performance percentile.

    :param fn: Function to benchmark
    :type fn: Callable
    :param warmup: Warmup time (in ms)
    :type warmup: int
    :param rep: Repetition time (in ms)
    :type rep: int
    :param grad_to_none: Reset the gradient of the provided tensor to None
    :type grad_to_none: torch.tensor, optional
    :param percentiles: Performance percentile to return in addition to the median.
    :type percentiles: list[float]
    :param fast_flush: Use faster kernel to flush L2 between measurements
    :type fast_flush: bool
    """

    # Estimate the runtime of the function
    fn()
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(5):
        fn()
    end_event.record()
    torch.cuda.synchronize()
    estimate_ms = start_event.elapsed_time(end_event) / 5
    # compute number of warmup and repeat
    n_warmup = max(1, int(warmup / estimate_ms))
    n_repeat = max(1, int(rep / estimate_ms))
    # We maintain a buffer of 256 MB that we clear
    # before each kernel call to make sure that the L2
    # doesn't contain any input data before the run
    start_event = [torch.cuda.Event(enable_timing=True) for i in range(n_repeat)]
    end_event = [torch.cuda.Event(enable_timing=True) for i in range(n_repeat)]
    if fast_flush:
        cache = torch.empty(int(256e6 // 4), dtype=torch.int, device="cuda")
    else:
        cache = torch.empty(int(256e6), dtype=torch.int8, device="cuda")
    # Warm-up
    for _ in range(n_warmup):
        fn()
    # Benchmark
    for i in range(n_repeat):
        # we don't want `fn` to accumulate gradient values
        # if it contains a backward pass. So we clear the
        # provided gradients
        if grad_to_none is not None:
            for x in grad_to_none:
                x.grad = None
        # we clear the L2 cache before each run
        cache.zero_()
        # record time of `fn`
        start_event[i].record()
        fn()
        end_event[i].record()
    # Record clocks
    torch.cuda.synchronize()
    times = torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)])
    if percentiles:
        percentiles = torch.quantile(times, torch.tensor(percentiles)).tolist()
        return tuple(percentiles)
    else:
        return torch.mean(times).item()


def sympy_product(it):
    return functools.reduce(operator.mul, it, sympy.Integer(1))


def sympy_dot(seq1, seq2):
    assert len(seq1) == len(seq2)
    return sympy.expand(sum(a * b for a, b in zip(seq1, seq2)))


def unique(it):
    return {id(x): x for x in it}.values()


def ceildiv(numer: int, denom: int):
    assert isinstance(numer, int) and isinstance(denom, int)
    return -(numer // -denom)


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


def timed(model, example_inputs, times=1):
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
    timings = [timed(fn, args, times) for _ in range(repeat)]
    took = np.median(timings)
    print(f"{took/baseline:.6f}")
    return took


immutable_dict.__hash__ = lambda self: hash(tuple(self.items()))
immutable_list.__hash__ = lambda self: hash(tuple(self))


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

    wrapped.cache_info = f.cache_info
    return wrapped


def precompute_method(obj: Any, method: str):
    """Replace obj.method() with a new method that returns a precomputed constant."""
    result = getattr(obj, method)()
    setattr(obj, method, lambda: result)


def precompute_methods(obj: Any, methods: List[str]):
    """Replace methods with new methods that returns a precomputed constants."""
    for method in methods:
        precompute_method(obj, method)


def cmp(a, b):
    return int(a > b) - int(a < b)


def cache_on_self(fn):
    key = f"__{fn.__name__}_cache"

    @functools.wraps(fn)
    def wrapper(self):
        if not hasattr(self, key):
            setattr(self, key, fn(self))
        return getattr(self, key)

    return wrapper


def get_fused_kernel_name(node_schedule):
    return "_".join(
        ["fused"]
        + [
            str(origin.name)
            for origin in functools.reduce(
                operator.or_,
                [node.node.origins for node in node_schedule if hasattr(node, "node")],
            )
            if origin.op == "call_function"
        ][0 : config.kernel_name_max_ops]
    )


def gather_origins(args, kwargs):
    import itertools

    from .ir import ComputedBuffer, IRNode

    def is_unrealized_node(n):
        return isinstance(n, IRNode) and not isinstance(n, ComputedBuffer)

    kwarg_origins = [val.origins for val in kwargs.values() if is_unrealized_node(val)]
    arg_origins = [arg.origins for arg in args if is_unrealized_node(arg)]
    return set(itertools.chain(*arg_origins, *kwarg_origins))


def sympy_str(expr: sympy.Expr):
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

    from .ir import CleanDiv, IndexingDiv, ModularIndexing

    if isinstance(expr, (ModularIndexing, CleanDiv, IndexingDiv)):
        return f"{expr.func.__name__}({', '.join(map(sympy_str, expr.args))})"
    return str(expr)


def sympy_symbol(name):
    return sympy.Symbol(name, integer=True, positive=True)


def sympy_subs(expr: sympy.Expr, replacements: Dict[Any, Any]):
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


def has_incompatible_cudagraph_ops(gm):
    forbidden_list = set(
        [
            "aten._fused_moving_avg_obs_fq_helper.default",
            "aten._fused_moving_avg_obs_fq_helper_functional.default",
            "fbgemm.dense_to_jagged.default",
            "fbgemm.jagged_to_padded_dense.default",
        ]
    )
    for node in gm.graph.nodes:
        if str(node.target) in forbidden_list:
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


def argsort(seq):
    return sorted(range(len(seq)), key=seq.__getitem__)
