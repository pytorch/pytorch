import collections
import contextlib
import functools
import glob
import itertools
import logging
import math
import operator
import os
import shutil
import tempfile
import textwrap
import time
from io import StringIO
from typing import Any, Dict, List, Optional, Union
from unittest import mock

import sympy

import torch
from torch.fx.immutable_collections import immutable_dict, immutable_list

from . import config, config as inductor_config
from .cuda_properties import get_device_capability

log = logging.getLogger(__name__)

VarRanges = Dict[sympy.Expr, sympy.Expr]


try:
    from triton.testing import do_bench
except ImportError:

    def do_bench(*args, **kwargs):
        raise NotImplementedError("requires Triton")


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
    timings = torch.tensor([timed(fn, args, times) for _ in range(repeat)])
    took = torch.median(timings)
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
        + sorted(
            [
                str(origin.name)
                for origin in functools.reduce(
                    operator.or_,
                    [
                        node.node.origins
                        for node in node_schedule
                        if hasattr(node, "node")
                    ],
                )
                if origin.op == "call_function"
            ]
        )[0 : config.kernel_name_max_ops]
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

    from .ir import CleanDiv, FloorDiv, ModularIndexing

    if isinstance(expr, (ModularIndexing, CleanDiv, FloorDiv)):
        return f"{expr.func.__name__}({', '.join(map(sympy_str, expr.args))})"
    return str(expr)


def sympy_symbol(name):
    # This should never be used for creating shape/stride symbols, as those
    # should all be allocated before Inductor.
    assert name[0] != "s"
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
    forbidden_list = {
        "aten._fused_moving_avg_obs_fq_helper.default",
        "aten._fused_moving_avg_obs_fq_helper_functional.default",
        "fbgemm.dense_to_jagged.default",
        "fbgemm.jagged_to_padded_dense.default",
    }
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
    # preserve original order for equal strides
    getter = seq.__getitem__
    a_r = range(len(seq))
    return list(reversed(sorted(a_r, key=getter, reverse=True)))  # noqa: C413


@functools.lru_cache(8)
def get_dtype_size(dtype):
    return torch.empty((), dtype=dtype).element_size()


class IndentedBuffer:
    tabwidth = 4

    def __init__(self, initial_indent=0):
        self._lines = []
        self._indent = initial_indent

    def getvalue(
        self,
    ):
        buf = StringIO()
        for line in self._lines:
            if isinstance(line, DeferredLineBase):
                line = line()
                if line is None:
                    continue
            assert isinstance(line, str)
            buf.write(line)
            buf.write("\n")
        return buf.getvalue()

    def getrawvalue(self):
        buf = StringIO()
        for line in self._lines:
            if isinstance(line, DeferredLineBase):
                line = line()
                if line is None:
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
        if isinstance(line, DeferredLineBase):
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
            yield
            self._indent -= offset

        return ctx()

    def splice(self, other_code, strip=False):
        if isinstance(other_code, IndentedBuffer):
            dedent = float("inf")
            for line in other_code._lines:
                if line:
                    dedent = min(dedent, len(line) - len(line.lstrip()))
            if math.isinf(dedent):
                dedent = 0
            for line in other_code._lines:
                IndentedBuffer.writeline(self, line[dedent:])
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
    cores = torch.cuda.get_device_properties(index).multi_processor_count
    if cores < 80:  # V100
        log.warning("not enough cuda cores to use max_autotune mode")
        return False
    return True


def use_triton_template(layout):
    return (
        inductor_config.max_autotune
        and layout.device.type == "cuda"
        and layout.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and is_big_gpu(layout.device.index or 0)
    )


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


def run_and_get_triton_code(fn, *args, **kwargs):
    from torch._inductor.debug import DebugContext
    from torch._inductor.virtualized import V

    torch._dynamo.reset()

    context = DebugContext()

    with DebugDirManager(), mock.patch.object(
        config.trace, "enabled", True
    ), context, V.set_debug_handler(context):

        dir_name = "/".join(context._path.split("/")[:-1]) + "/"
        fil = dir_name + "*inference*"
        existing_dirs = glob.glob(fil)

        fn(*args, **kwargs)

        assert context._path is not None

        dir_dbg = [x for x in glob.glob(fil) if x not in existing_dirs]

        assert len(dir_dbg) == 1, f"{dir_dbg}, {context._path}"

        full_name = os.path.join(dir_dbg[0], "output_code.py")
        with open(full_name, "r") as f:
            return f.read()


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
