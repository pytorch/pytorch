import contextlib
import dataclasses
import functools
import itertools
import logging
import re
import textwrap
from collections import OrderedDict
from contextlib import nullcontext
from enum import Enum
from functools import partial
from inspect import signature
from typing import Any, Callable, ClassVar, Dict, List, Optional, Set, Tuple, Union
from unittest.mock import patch

import numpy
import sympy
from sympy import Expr, Integer

import torch.fx
import torch.utils._pytree as pytree
from torch._prims_common import (
    is_boolean_dtype,
    is_float_dtype,
    make_channels_last_strides_for,
    make_contiguous_strides_for,
)
from torch._subclasses.fake_tensor import FakeTensorMode

from . import config, dependencies
from .codegen.common import index_prevent_reordering
from .cuda_properties import get_device_properties
from .dependencies import extract_read_writes, var_builder
from .utils import (
    argsort,
    cache_on_self,
    sympy_dot,
    sympy_product,
    sympy_subs,
    sympy_symbol,
)
from .virtualized import ops, V

log = logging.getLogger(__name__)
indent = functools.partial(textwrap.indent, prefix="  ")
aten = torch.ops.aten


def inverse_reorder(order):
    inv_order = dict(zip(order, range(len(order))))

    def reindex(index):
        assert len(index) == len(inv_order)
        return [index[inv_order[i]] for i in range(len(index))]

    return reindex


def same_reorder(order):
    def reindex(index):
        assert len(index) == len(order)
        return [index[order[i]] for i in range(len(index))]

    return reindex


def fuse_reindexing(reindex1, reindex2):
    def reindex(index):
        return reindex1(reindex2(index))

    return reindex


def stride_order2fill_order(order):
    """
    Convert stride order to fill order
    For channel last format,
    stride order = [3, 0, 2, 1] and fill order = [1, 3, 2, 0]
    """
    lookup = {pos: idx for idx, pos in enumerate(order)}
    fill_order = [lookup[i] for i in range(len(order))]
    return fill_order


def get_stride_order(seq):
    """
    Convert strides to stride order
    """
    sorted_idx = argsort(seq)
    out = [None for _ in range(len(seq))]
    for i, elem in enumerate(sorted_idx):
        out[elem] = i
    return out


def reads_from_conv(buf, var_ranges):
    """
    return:
    if reads_from_conv: boolean
    the new memory_addr: Sympy Expression
    """
    if buf is None:
        return False, None
    if isinstance(buf, Convolution):
        indexer = buf.layout.as_fixed().make_indexer()
        index_vars = sorted(var_ranges, key=lambda var: var.name)
        index = indexer(index_vars)
        return True, index
    # for case like
    # buf0 = conv(x, w)
    # return torch.cat([buf0, buf1]), torch.cat([buf0, buf2])
    # Because of ConcatKernel, it will create two bufs buf3 and 4
    # buf3 has the AliasedLayout which reads from buf0(Convolution)
    # but buf4 is a copy of buf3 which reads from buf3
    # we want to know that buf4 also follows buf0 conv's layout
    if isinstance(buf.layout, AliasedLayout):
        reads = buf.get_read_writes().reads
        reads_bufs = [
            V.graph.name_to_buffer[r.name]
            if r.name in V.graph.name_to_buffer.keys()
            else None
            for r in reads
        ]
        for reads_buf in reads_bufs:
            read_from_conv, addr = reads_from_conv(reads_buf, var_ranges)
            if read_from_conv:
                return True, addr
    return False, None


def ir_node_to_tensor(x, guard_shape=True):
    shape_fn = (
        V.graph.sizevars.guard_static_shape
        if guard_shape
        else V.graph.sizevars.size_hint
    )
    size = [shape_fn(s) for s in x.get_size()]
    if is_storage_and_layout(x):
        stride = [shape_fn(s) for s in x.get_layout().stride]
    else:
        stride = make_contiguous_strides_for(size)
    dtype = x.get_dtype()
    device = x.get_device()
    t = torch.empty_strided(
        size=size, stride=stride, dtype=dtype, device=device
    ).zero_()
    return t


def layout_priority_idx(reads_bufs, memory_addrs, var_ranges):
    """
    if reads from conv that needs to use specific layout
    return:
    priority_idx regarding memory_addrs idx
    memory_addrs - update memory_addrs with the true addr if needed
    """

    priority_idx = []
    for i, reads_buf in enumerate(reads_bufs):
        read_from_conv, mem_addr = reads_from_conv(reads_buf, var_ranges)
        if read_from_conv:
            priority_idx.append(i)
            memory_addrs[i] = mem_addr
    return priority_idx, memory_addrs


class ModularIndexing(sympy.Function):
    """
    ModularIndexing(a, b, c) => (a // b) % c
    """

    nargs = (3,)

    @classmethod
    def eval(cls, base, divisor, modulus):
        if base == 0 or modulus == 1:
            return sympy.Integer(0)

        if (
            isinstance(base, sympy.Integer)
            and isinstance(divisor, sympy.Integer)
            and isinstance(modulus, sympy.Integer)
        ):
            return (base // divisor) % modulus

        if divisor != 1:
            gcd = sympy.gcd(base, divisor)
            if gcd != 1:
                return ModularIndexing(base / gcd, divisor / gcd, modulus)

        if isinstance(base, sympy.Add):
            new_terms = []
            for term in base.args:
                if sympy.gcd(term, modulus * divisor) != modulus * divisor:
                    new_terms.append(term)
            if len(new_terms) != len(base.args):
                return ModularIndexing(sum(new_terms), divisor, modulus)

        if isinstance(base, IndexingDiv):
            return ModularIndexing(base.args[0], base.args[1] * divisor, modulus)


class IndexingDiv(sympy.Function):
    """
    a // b used in indexing where we need to be careful about simplification.
    We don't use sympy.FloorDiv to bypass some simplification rules.
    """

    nargs = (2,)

    @classmethod
    def eval(cls, base, divisor):
        if base == 0:
            return sympy.Integer(0)
        if divisor == 1:
            return base
        if isinstance(base, sympy.Integer) and isinstance(divisor, sympy.Integer):
            return base // divisor
        if isinstance(base, IndexingDiv):
            return IndexingDiv(base.args[0], base.args[1] * divisor)

        if isinstance(base, sympy.Add):
            for a in base.args:
                gcd = sympy.gcd(a, divisor)
                if gcd == divisor:
                    return IndexingDiv(base - a, divisor) + a / gcd
        gcd = sympy.gcd(base, divisor)
        if gcd != 1:
            return IndexingDiv(
                sympy.simplify(base / gcd), sympy.simplify(divisor / gcd)
            )


class CleanDiv(IndexingDiv):
    """
    Div where we can assume no rounding.
    This is to enable future optimizations.
    """

    pass


class CeilDiv(sympy.Function):
    """
    Div used in indexing that rounds up.
    """

    def __new__(cls, base, divisor):
        if sympy.gcd(base, divisor) == divisor:
            return CleanDiv(base, divisor)
        else:
            return IndexingDiv(base + (divisor - 1), divisor)


def get_device_type(x):
    if getattr(x, "get_device", None):
        return get_device_type(x.get_device())
    if isinstance(x, torch.device):
        return x.type
    return None


def is_triton(x):
    return get_device_type(x) == "cuda"


def is_cpu(x):
    return get_device_type(x) == "cpu"


@dataclasses.dataclass
class IRNode(object):
    _current_origins: ClassVar[Set[Any]] = set()

    @staticmethod
    @contextlib.contextmanager
    def current_origins(origins: Set[torch.fx.Node]):
        old = IRNode._current_origins
        IRNode._current_origins = old | origins
        yield
        IRNode._current_origins = old

    def __post_init__(self):
        self.origins = set(self._current_origins)

    def common_repr(self):
        return (
            [f"origins={self.origins}"] if hasattr(self, "origins") else ["no origins?"]
        )

    def str_helper(self, lines):
        lines = lines + self.common_repr()
        lines = indent(",\n".join(map(str, lines)))
        return f"{type(self).__name__}(\n{lines}\n)"

    def is_user_of(self, name):
        return any(name == dep.name for dep in self.get_reads())

    def get_numel(self):
        return sympy_product(self.get_size())


@dataclasses.dataclass
class Loops(IRNode):
    device: torch.device
    dtype: torch.dtype
    inner_fn: Callable
    ranges: List[Expr]

    def __str__(self, names=("ranges",)):
        return self.str_helper(
            [
                f"'{self.device.type}'",
                str(self.dtype),
                self.inner_fn_str(),
            ]
            + [f"{name}={getattr(self, name)}" for name in names]
        )

    __repr__ = __str__

    def get_dtype(self):
        return self.dtype

    def get_device(self):
        return self.device

    def get_size(self):
        return self.ranges

    def is_extern(self):
        return False

    @classmethod
    def create(cls, *args, **kwargs):
        return TensorBox.create(cls(*args, **kwargs))

    @staticmethod
    def _index(ranges, prefix="i"):
        return [
            sympy.Integer(0) if s == 1 else sympy_symbol(f"{prefix}{n}")
            for n, s in enumerate(ranges)
        ]

    @cache_on_self
    def inner_fn_str(self):
        try:
            with V.set_ops_handler(V.MockHandler()), patch.object(
                FlexibleLayout, "allow_indexing", True
            ):
                return str(self.inner_fn(self._index(self.ranges)))
        except Exception as e:
            return f"inner_fn(): {e}"

    def is_zero_elements(self):
        return any(r == 0 for r in self.ranges)

    @cache_on_self
    def get_reads(self):
        with patch.object(FlexibleLayout, "allow_indexing", True):
            if self.get_reduction_type():
                return extract_read_writes(
                    self.make_loader(),
                    self.get_size(),
                    self.get_reduction_size(),
                ).reads
            else:
                return extract_read_writes(
                    self.make_loader(),
                    self.get_size(),
                ).reads


class Pointwise(Loops):
    def make_loader(self):
        return self.inner_fn

    def get_reduction_size(self):
        return []

    def get_reduction_type(self):
        return None

    def store_output(self, output_name, indexer, vars):
        return ops.store(output_name, indexer(vars), self.inner_fn(vars))

    def constant_to_device(self, device):
        """Move this to a given device. Requires that all reads are to constants."""
        loader = self.make_loader()
        loader = patch.object(ConstantBuffer, "override_device", device)(loader)
        return Pointwise(device, self.dtype, loader, self.ranges)


@dataclasses.dataclass
class Scatter(Pointwise):
    output_indexer: Callable[[List[Expr]], Expr]
    scatter_mode: Optional[str] = None

    def constant_to_device(self, device):
        """Move this to a given device. Requires that all reads are to constants."""
        loader = self.make_loader()
        loader = patch.object(ConstantBuffer, "override_device", device)(loader)
        return Scatter(
            device,
            self.dtype,
            loader,
            self.ranges,
            self.output_indexer,
            self.scatter_mode,
        )

    def store_output(self, output_name, indexer, vars):
        return ops.store(
            output_name,
            indexer(self.output_indexer(vars)),
            self.inner_fn(vars),
            mode=self.scatter_mode,
        )


class ReductionHint(Enum):
    INNER = 0
    OUTER = 1
    OUTER_TINY = 2
    DEFAULT = 3


class TileHint(Enum):
    SQUARE = 0
    DEFAULT = 1


@dataclasses.dataclass
class Reduction(Loops):
    reduction_ranges: List[Expr]
    reduction_type: str
    # self.dtype represents the dst dtype
    src_dtype: torch.dtype
    reduction_hint: ReductionHint

    def __str__(self):
        return Loops.__str__(
            self, names=("ranges", "reduction_ranges", "reduction_type")
        )

    __repr__ = __str__

    def get_reduction_size(self):
        return self.reduction_ranges

    def get_reduction_type(self):
        return self.reduction_type

    def store_reduction(self, output_name, indexer, vars, reduction_vars):
        return ops.reduction(
            output_name,
            self.dtype,
            self.src_dtype,
            self.reduction_type,
            indexer(vars),
            self.inner_fn(vars, reduction_vars),
        )

    def index_length(self):
        return len(self.ranges) + len(self.reduction_ranges)

    @cache_on_self
    def inner_fn_str(self):
        try:
            with V.set_ops_handler(V.MockHandler()), patch.object(
                FlexibleLayout, "allow_indexing", True
            ):
                return str(
                    self.inner_fn(
                        self._index(self.ranges),
                        self._index(self.reduction_ranges, "r"),
                    )
                )
        except Exception as e:
            return f"inner_fn(): {e}"

    def constant_to_device(self, device):
        """Move this to a given device. Requires that all reads are to constants."""
        loader = self.make_loader()
        loader = patch.object(ConstantBuffer, "override_device", device)(loader)
        return Reduction(
            device,
            self.dtype,
            loader,
            self.ranges,
            self.reduction_ranges,
            self.reduction_type,
            self.src_dtype,
            ReductionHint.DEFAULT,
        )

    @staticmethod
    def num_splits(
        device,
        dst_dtype,
        src_dtype,
        inner_fn,
        ranges,
        reduction_ranges,
        reduction_type,
        reduction_numel,
    ):
        num_sm = get_device_properties(device).multi_processor_count
        min_elements_per_thread = 32
        max_elements_per_thread = 512
        threads_per_sm = 2048
        min_elements_per_device = min_elements_per_thread * num_sm * threads_per_sm
        max_elements_per_device = max_elements_per_thread * num_sm * threads_per_sm

        def inner_reduction_splits(reduction_numel_hint, numel_hint):
            # do heuristics that's close to eager mode for split inner reduction
            # we leak reduction autotune configs here, and will need to refactor to avoid this later
            num_warps = 8
            num_threads = 32 * num_warps
            if numel_hint >= 2 * num_sm:  # don't split if there are enough outputs
                return 1
            if reduction_numel_hint <= 8192:
                return 1
            if reduction_numel_hint * numel_hint <= min_elements_per_device:
                split_size = min_elements_per_thread
            elif reduction_numel_hint * numel_hint < max_elements_per_device:
                target_blocks = num_sm * threads_per_sm // (2 * num_threads)
                blocks_per_output = (target_blocks + numel_hint - 1) // numel_hint
                tmp_split_size = (
                    reduction_numel_hint + num_threads * blocks_per_output - 1
                ) // (num_threads * blocks_per_output)
                divisors = sympy.divisors(reduction_numel_hint)
                closest = min(divisors, key=lambda x: abs(x - tmp_split_size))
                if abs(closest - tmp_split_size) < 30:
                    # prefer even splits, but never smalle than min_elements_per_thread
                    split_size = max(closest, min_elements_per_thread)
                else:
                    split_size = tmp_split_size
            else:
                divisors = sympy.divisors(reduction_numel_hint)
                closest = min(divisors, key=lambda x: abs(x - max_elements_per_thread))
                if abs(closest - max_elements_per_thread) < 50:
                    # prefer even splits
                    split_size = closest
                else:
                    split_size = max_elements_per_thread
            return (reduction_numel_hint + split_size * num_threads - 1) // (
                split_size * num_threads
            )

        def outer_reduction_splits(reduction_numel_hint, numel_hint):
            # TODO the best heuristic currently has XBLOCK (corresponding to numel_hint) 128
            # extend to even smaller number of outputs
            num_warps = 8
            num_threads = num_warps * 32
            rvals_per_thread = 4  # comes from heuristics, refactor to not leak here
            xvals_per_block = 128
            xblocks = (numel_hint + xvals_per_block - 1) // xvals_per_block
            if reduction_numel_hint * numel_hint < min_elements_per_device:
                split_size = min_elements_per_thread
            elif reduction_numel_hint * numel_hint < max_elements_per_device:
                target_blocks = num_sm * threads_per_sm // (num_threads)
                target_blocks = (target_blocks + xblocks - 1) // xblocks
                tmp_split_size = (
                    reduction_numel_hint + rvals_per_thread * target_blocks - 1
                ) // (rvals_per_thread * target_blocks)
                divisors = sympy.divisors(reduction_numel_hint)
                closest = min(divisors, key=lambda x: abs(x - tmp_split_size))
                if abs(tmp_split_size - closest) < 20:
                    split_size = max(closest, min_elements_per_thread)
                else:
                    split_size = tmp_split_size
            else:
                divisors = sympy.divisors(reduction_numel_hint)
                closest = min(divisors, key=lambda x: abs(x - max_elements_per_thread))
                if abs(closest - max_elements_per_thread) < 50:
                    # prefer even splits
                    split_size = closest
                else:
                    split_size = max_elements_per_thread

            return (reduction_numel_hint + rvals_per_thread * split_size - 1) // (
                rvals_per_thread * split_size
            )

        reduction_numel_hint = V.graph.sizevars.size_hint(reduction_numel)
        numel_hint = V.graph.sizevars.size_hint(sympy_product(ranges))
        # easy cases
        if numel_hint == 1:
            return ReductionHint.INNER, inner_reduction_splits(
                reduction_numel_hint, numel_hint
            )
        if (
            reduction_numel_hint <= min_elements_per_thread
            or numel_hint >= num_sm * 2 * 32
        ):
            return ReductionHint.DEFAULT, 1

        r = Reduction(
            device,
            dst_dtype,
            inner_fn,
            ranges,
            reduction_ranges,
            reduction_type,
            src_dtype,
            ReductionHint.DEFAULT,
        )
        read_writes = ComputedBuffer(
            name=None,
            layout=FlexibleLayout(
                device=r.get_device(),
                dtype=r.get_dtype(),
                size=r.get_size(),
            ),
            data=r,
        ).get_read_writes()
        # try finding the full size producer
        # TODO this will fail for something like ((1, N) * (N, 1)).sum()
        # this would also possibly be wrong for producers with the different contiguity but we hope those cases are rare
        # TODO maybe go over all full size producers and pick the most common one?
        range_vars = [
            r
            for r in read_writes.range_vars
            if isinstance(r, sympy.Expr) and not isinstance(r, sympy.Number)
        ]
        index = None
        for md in read_writes.reads:
            if all([r in md.index.free_symbols for r in range_vars]):
                index = md.index
                break
        if not index:
            # TODO determine splits when all inputs are broadcasted
            return ReductionHint.DEFAULT, 1
        reduction_vars = [
            rv for rv in range_vars if read_writes.var_ranges[rv] in reduction_ranges
        ]
        strides = V.graph.sizevars.stride_hints(index, reduction_vars)
        outer = all([s > 1 for s in strides])
        if not outer:
            return ReductionHint.INNER, inner_reduction_splits(
                reduction_numel_hint, numel_hint
            )
        else:  # outer reduction
            return ReductionHint.OUTER, outer_reduction_splits(
                reduction_numel_hint, numel_hint
            )

    @staticmethod
    def _unroll_reduction_fn(inner_fn, reduction_ranges, reduction_type):
        """Convert inner_fn from a reduction to an pointwise"""
        reduction_ranges = [
            V.graph.sizevars.guard_static_shape(x) for x in reduction_ranges
        ]

        if reduction_type == "sum":

            def combine_fn(a, b):
                return ops.add(a, b)

        elif reduction_type == "min":

            def combine_fn(a, b):
                return ops.minimum(a, b)

        elif reduction_type == "max":

            def combine_fn(a, b):
                return ops.maximum(a, b)

        elif reduction_type == "any":

            def combine_fn(a, b):
                return ops.logical_or(a, b)

        elif reduction_type == "argmin":

            def combine_fn(a, b):
                return ops.minimum(a[0], b[0]), ops.where(
                    ops.lt(b[0], a[0]), b[1], a[1]
                )

        elif reduction_type == "argmax":

            def combine_fn(a, b):
                return ops.maximum(a[0], b[0]), ops.where(
                    ops.gt(b[0], a[0]), b[1], a[1]
                )

        else:
            raise NotImplementedError(f"unknown reduction_type={reduction_type}")

        def fn(index):
            return functools.reduce(
                combine_fn,
                (
                    value_fn(index, rindex)
                    for rindex in itertools.product(
                        *[range(x) for x in reduction_ranges]
                    )
                ),
            )

        if reduction_type in ("argmin", "argmax"):
            flatten_index = FixedLayout(
                None,
                None,
                reduction_ranges,
                FlexibleLayout.contiguous_strides(reduction_ranges),
            ).make_indexer()

            def value_fn(index, rindex):
                rindex = [sympy.expand(i) for i in rindex]
                return (
                    inner_fn(index, rindex),
                    ops.index_expr(flatten_index(rindex), torch.int64),
                )

            return lambda index: fn(index)[1]
        else:
            value_fn = inner_fn
            return fn

    @classmethod
    def create(
        cls,
        device: torch.device,
        dst_dtype: torch.dtype,
        src_dtype: torch.dtype,
        inner_fn: Callable,
        ranges: List[Expr],
        reduction_ranges: List[Expr],
        reduction_type: str,
        reduction_hint: ReductionHint = ReductionHint.DEFAULT,
    ):
        reduction_numel = V.graph.sizevars.simplify(sympy_product(reduction_ranges))

        if reduction_numel == 0:

            # N.B. This is a hack to generate the literal of the given type
            # Ideally, we should be fixing `def constant` in triton.py
            # but it breaks due to hardcoded dtypes in other places
            def py_cnst(val):
                return (
                    bool(val)
                    if dst_dtype == torch.bool
                    else float(val)
                    if dst_dtype.is_floating_point
                    else int(val)
                )

            rtypes_to_inits = {
                "sum": py_cnst(0),
                "prod": py_cnst(1),
                "any": py_cnst(0),
                # "all" is desugared to `!any(!val)`
            }

            assert (
                reduction_type in rtypes_to_inits.keys()
            ), f"{reduction_type} not supported for zero-dimension tensors!"

            def const_fn(index):
                return ops.constant(rtypes_to_inits[reduction_type], dst_dtype)

            return Pointwise.create(
                device=device,
                dtype=src_dtype,
                inner_fn=const_fn,
                ranges=list(ranges),
            )

        if reduction_numel == 1:
            # this reduction is actually a pointwise op
            if reduction_type in ("argmin", "argmax"):

                def fn(index):
                    return 0

            else:

                def fn(index):
                    reduction_index = [sympy.Integer(0) for _ in reduction_ranges]
                    return inner_fn(index, reduction_index)

            return Pointwise.create(device, dst_dtype, fn, ranges)

        if (
            isinstance(reduction_numel, sympy.Integer)
            and V.graph.sizevars.size_hint(reduction_numel)
            < config.unroll_reductions_threshold
            and sympy_product(ranges) != 1
        ):
            return Pointwise.create(
                device,
                dst_dtype,
                cls._unroll_reduction_fn(inner_fn, reduction_ranges, reduction_type),
                ranges,
            )

        if is_triton(device) and reduction_type not in {"argmax", "argmin"}:
            # triton doesn't support reduce to single element well, so break it up
            hint, split = cls.num_splits(
                device,
                dst_dtype,
                src_dtype,
                inner_fn,
                ranges,
                reduction_ranges,
                reduction_type,
                reduction_numel,
            )
            # intermediate reduction in split can contain complex indexing,
            # and num_splits will fail to correctly set the hint
            # reuse the passed hint if available
            if reduction_hint == ReductionHint.DEFAULT:
                reduction_hint = hint
            if split > 1:
                # triton doesn't support reduce to single element well, so break it up
                return cls.create_multilayer(
                    device,
                    dst_dtype,
                    src_dtype,
                    inner_fn,
                    ranges,
                    reduction_ranges,
                    reduction_type,
                    split,
                    reduction_hint,
                )

        return TensorBox.create(
            Reduction(
                device,
                dst_dtype,
                inner_fn,
                ranges,
                reduction_ranges,
                reduction_type,
                src_dtype,
                reduction_hint,
            )
        )

    @staticmethod
    def default_value(reduction_type, dtype):
        if reduction_type in {"max", "argmax"}:
            if is_float_dtype(dtype):
                return float("-inf")
            elif is_boolean_dtype(dtype):
                return 0
            else:
                return torch.iinfo(dtype).min
        if reduction_type in {"min", "argmin"}:
            if is_float_dtype(dtype):
                return float("inf")
            elif is_boolean_dtype(dtype):
                return 1
            else:
                return torch.iinfo(dtype).max

        return {
            "sum": 0,
            "any": 0,
        }[reduction_type]

    @classmethod
    def create_multilayer(
        cls,
        device: torch.device,
        dst_dtype: torch.dtype,
        src_dtype: torch.dtype,
        inner_fn: Callable,
        ranges: List[Expr],
        reduction_ranges: List[Expr],
        reduction_type: str,
        split: int,
        reduction_hint: ReductionHint,
    ):
        """
        Break a large reduction up into multiple smaller reductions
        recursively
        """
        reduction_numel = sympy_product(reduction_ranges)

        # TODO(jansel): convert this to dynamic shapes
        # TODO(jansel): realize the reduction so we can do dynamic indexing
        reduction_ranges = [
            sympy.Integer(V.graph.sizevars.guard_static_shape(s))
            for s in reduction_ranges
        ]
        reduction_numel = sympy.Integer(
            V.graph.sizevars.guard_static_shape(reduction_numel)
        )

        if V.graph.sizevars.size_hint(reduction_numel) % split == 0:
            need_mask = False
        else:
            need_mask = True

        split = sympy.Integer(split)
        block_size = IndexingDiv(reduction_numel + (split - 1), split)

        reindex = View.dynamic_reshape_indexer(reduction_ranges, [reduction_numel])

        def wrapper_fn(index, reduction_index):
            (reduction_index,) = reduction_index
            *new_index, reduction_block = index
            indices = block_size * reduction_block + reduction_index

            def body():
                return inner_fn(new_index, reindex([indices]))

            if need_mask:
                mask = ops.lt(
                    ops.index_expr(indices, torch.int32),
                    ops.index_expr(reduction_numel, torch.int32),
                )
                return ops.masked(
                    mask, body, cls.default_value(reduction_type, dst_dtype)
                )
            else:
                return body()

        # triton will automatically compute reductions in fp32 if reducing over fp16/bf16
        # within the kernel. keep the intermediate in fp32 so as to keep the whole reduction
        # in fp32 and not reduce precision by breaking up the kernel into multiple layers
        intermediate_dtype = (
            dst_dtype
            if dst_dtype not in (torch.float16, torch.bfloat16)
            else torch.float
        )
        intermediate = Reduction.create(
            device,
            intermediate_dtype,
            src_dtype,
            wrapper_fn,
            [*ranges, split],
            [block_size],
            reduction_type,
            reduction_hint,
        )
        intermediate.realize()
        intermediate_loader = intermediate.make_loader()

        def intermediate_fn(index, reduction_index):
            return intermediate_loader([*index, *reduction_index])

        numel_hint = V.graph.sizevars.size_hint(sympy_product(ranges))
        if split <= 512 and numel_hint <= 512 and reduction_hint == ReductionHint.OUTER:
            reduction_hint = ReductionHint.OUTER_TINY
        return TensorBox.create(
            Reduction(
                device,
                dst_dtype,
                intermediate_fn,
                ranges,
                [split],
                reduction_type,
                src_dtype,
                reduction_hint,
            )
        )


def is_storage_and_layout(x):
    try:
        as_storage_and_layout(x, freeze=False)
        return True
    except NotImplementedError:
        return False


def is_contiguous_storage_and_layout(x):
    try:
        buffer, layout = as_storage_and_layout(x, freeze=False)
        return layout.is_contiguous()
    except NotImplementedError:
        return False


def as_storage_and_layout(x, freeze=True, want_contiguous=False, stride_order=None):
    """Try to simplify x into a StorageBox and a Layout"""
    if isinstance(x, TensorBox):
        return as_storage_and_layout(
            x.data,
            freeze=freeze,
            want_contiguous=want_contiguous,
            stride_order=stride_order,
        )
    if isinstance(x, StorageBox) and isinstance(x.data, Buffer):
        if freeze:
            if want_contiguous:
                x.data.freeze_layout()
            elif stride_order is not None:
                x.data.freeze_layout_with_stride_order(stride_order)
            else:
                x.data.decide_layout()
        return x, x.data.layout
    if isinstance(x, ReinterpretView):
        buffer, _ = as_storage_and_layout(
            x.data,
            freeze=freeze,
            want_contiguous=want_contiguous,
            stride_order=stride_order,
        )
        return buffer, x.layout
    raise NotImplementedError


as_contiguous_storage_and_layout = functools.partial(
    as_storage_and_layout, want_contiguous=True
)


def is_stride_order_storage_and_layout(x, stride_order):
    try:
        buffer, layout = as_storage_and_layout(x, freeze=False)
        return layout.is_stride_ordered(stride_order)
    except NotImplementedError:
        return False


@dataclasses.dataclass
class BaseView(IRNode):
    data: IRNode

    def get_dtype(self):
        return self.data.get_dtype()

    def get_device(self):
        return self.data.get_device()

    def get_name(self):
        return self.data.get_name()

    def mark_reuse(self, users):
        return self.data.mark_reuse(users)

    def has_exceeded_max_reads(self):
        return self.data.has_exceeded_max_reads()

    def realize(self):
        return self.data.realize()

    def realize_hint(self):
        return self.data.realize_hint()

    def get_storage_numel(self):
        return self.data.get_storage_numel()

    def is_extern(self):
        return self.data.is_extern()

    @cache_on_self
    def get_reads(self):
        with patch.object(FlexibleLayout, "allow_indexing", True):
            return extract_read_writes(
                self.make_loader(),
                self.get_size(),
            ).reads

    def unwrap_view(self):
        x = self
        while isinstance(x, BaseView):
            x = x.data
        return x

    def constant_to_device(self, device):
        """Move this to a given device. Requires that all reads are to constants."""
        loader = self.make_loader()
        loader = patch.object(ConstantBuffer, "override_device", device)(loader)
        return Pointwise(device, self.get_dtype(), loader, self.get_size())


@dataclasses.dataclass
class ExpandView(BaseView):
    size: List[Expr]

    @staticmethod
    def _normalize_size(x, new_size):
        """Replace `-1` with correct sizes"""
        new_size = list(map(sympy.expand, new_size))
        old_size = x.get_size()
        old_size = [None] * (len(new_size) - len(old_size)) + list(old_size)
        assert len(new_size) == len(old_size)
        for i in range(len(new_size)):
            if new_size[i] == -1:
                assert old_size[i] is not None
                new_size[i] = old_size[i]
        return new_size

    @classmethod
    def create(cls, x, new_size):
        new_size = cls._normalize_size(x, new_size)

        if is_storage_and_layout(x):
            storage, old_layout = as_storage_and_layout(x)
            skip = len(new_size) - len(old_layout.size)
            assert skip >= 0
            new_stride = [sympy.Integer(0)] * skip
            for stride, size in zip(old_layout.stride, old_layout.size):
                new_stride.append(stride if size != 1 else sympy.Integer(0))
            new_layout = FixedLayout(
                old_layout.device,
                old_layout.dtype,
                list(new_size),
                new_stride,
                old_layout.offset,
            )
            return ReinterpretView(storage, new_layout)

        return ExpandView(x, new_size)

    def get_size(self):
        return self.size

    def make_loader(self):
        target = self.get_size()
        actual = self.data.get_size()
        skip = len(target) - len(actual)
        inner = self.data.make_loader()

        def load(index):
            index = list(index[skip:])
            assert len(index) == len(actual)
            for i in range(len(actual)):
                if actual[i] == 1:
                    # zero out broadcast dimension
                    index[i] = sympy.Integer(0)
            return inner(index)

        return load


@dataclasses.dataclass
class PermuteView(BaseView):
    dims: List[Expr]

    @classmethod
    def create(cls, x, dims):
        assert set(cls._map_neg_dims(dims)) == set(range(len(dims)))

        if is_storage_and_layout(x):
            storage, old_layout = as_storage_and_layout(x)
            new_layout = FixedLayout(
                old_layout.device,
                old_layout.dtype,
                [old_layout.size[i] for i in dims],
                [old_layout.stride[i] for i in dims],
                old_layout.offset,
            )
            return ReinterpretView(storage, new_layout)

        return PermuteView(x, dims)

    @classmethod
    def _map_neg_dims(cls, dims):
        return [dim if dim >= 0 else len(dims) + dim for dim in dims]

    def get_size(self):
        assert set(self._map_neg_dims(self.dims)) == set(range(len(self.dims)))
        size = self.data.get_size()
        return [size[i] for i in self.dims]

    def make_loader(self):
        inner = self.data.make_loader()
        inv = {j: i for i, j in enumerate(self.dims)}
        inv = [inv[i] for i in range(len(self.dims))]
        assert set(inv) == set(range(len(self.dims)))

        def load(index):
            index = [index[i] for i in inv]
            return inner(index)

        return load


class SqueezeView(BaseView):
    @classmethod
    def create(cls, x, *, dim=None):

        if is_storage_and_layout(x):
            storage, old_layout = as_storage_and_layout(x)
            new_size = []
            new_stride = []
            if dim is not None:
                assert isinstance(dim, int), "expected integer dim argument"
                assert 0 <= dim and dim < len(old_layout.size)

            for i, (size, stride) in enumerate(zip(old_layout.size, old_layout.stride)):
                if dim is None:
                    if size != 1:
                        new_size.append(size)
                        new_stride.append(stride)
                else:
                    if i != dim:
                        new_size.append(size)
                        new_stride.append(stride)
                    else:
                        assert size == 1, "expected squeezed size to be 1"

            new_layout = FixedLayout(
                old_layout.device,
                old_layout.dtype,
                new_size,
                new_stride,
                old_layout.offset,
            )
            return ReinterpretView(storage, new_layout)

        if dim is None:
            # redirect to a generic view
            return View.create(x, [s for s in x.get_size() if s != 1])
        else:
            assert x.get_size()[dim] == 1
            return View.create(x, [s for i, s in enumerate(x.get_size()) if i != dim])

    @staticmethod
    def squeezer(size: Tuple[sympy.Expr, ...]):
        new_size = [s for s in size if s != 1]
        not_one = [i for i, s in enumerate(size) if s != 1]
        length = len(size)

        def reindex(index: List[sympy.Expr]) -> List[sympy.Expr]:
            assert len(index) == len(not_one), f"{index} {not_one}"
            new_index = [sympy.Integer(0)] * length
            for idx, s in zip(not_one, index):
                new_index[idx] = s
            return tuple(new_index)

        return new_size, reindex

    def __init__(self, data):
        raise AssertionError("use SqueezeView.create()")


@dataclasses.dataclass
class View(BaseView):
    size: List[Expr]
    reindex: Callable

    def make_indexer(self):
        base_indexer = self.data.make_indexer()

        def indexer(idx):
            return base_indexer(self.reindex(idx))

        return indexer

    @staticmethod
    def handle_negative_index(idx, size):
        idx = sympy.expand(idx)
        size = sympy.expand(size)
        sizevars = V.graph.sizevars
        if sizevars.size_hint(idx) < 0:
            sizevars.guard_lt(idx, 0)
            idx = idx + size
        return idx

    def reindex_str(self):
        index_old = [sympy_symbol(f"i{n}") for n in range(len(self.size))]
        index_new = list(self.reindex(index_old))
        return f"lambda {', '.join(map(str, index_old))}: {index_new}"

    def __str__(self):
        return self.str_helper(
            [self.data, f"size={self.size}", f"reindex={self.reindex_str()}"]
        )

    __repr__ = __str__

    @classmethod
    def create(cls, x, new_size):
        assert isinstance(new_size, (tuple, list))
        old_size, new_size = cls.resolve_negative_size(x.get_size(), new_size)

        if V.graph.sizevars.maybe_guard_list_equals(old_size, new_size):
            return x

        # TODO: a new class for FixedTransferLayout that output layout is constrained by input layout
        if is_contiguous_storage_and_layout(x) and not isinstance(
            x.data, ExternKernelAlloc
        ):
            storage, old_layout = as_contiguous_storage_and_layout(x)
            new_layout = FixedLayout(
                old_layout.device,
                old_layout.dtype,
                new_size,
                FlexibleLayout.contiguous_strides(new_size),
                old_layout.offset,
            )
            return ReinterpretView(storage, new_layout)

        reindex = cls.dynamic_reshape_indexer(old_size, new_size)
        return cls(x, tuple(new_size), reindex)

    @staticmethod
    def resolve_negative_size(old_size, new_size):
        new_size = [V.graph.sizevars.simplify(x) for x in new_size]
        old_size = [V.graph.sizevars.simplify(x) for x in old_size]

        new_size = list(new_size)
        for i in range(len(new_size)):
            if new_size[i] == -1:
                new_size[i] = sympy.Integer(1)
                new_size[i] = CleanDiv(sympy_product(old_size), sympy_product(new_size))
                break

        V.graph.sizevars.guard_equals(sympy_product(old_size), sympy_product(new_size))
        return old_size, new_size

    @classmethod
    def dynamic_reshape_indexer(cls, old_size, new_size):
        try:
            reindex = cls._dynamic_reshape_indexer(old_size, new_size)
        except (AssertionError, IndexError):
            # optimistic algorithm failed, lets do a fallback
            flat = [sympy_product(old_size)]
            reindex1 = cls._dynamic_reshape_indexer(old_size, flat)
            reindex2 = cls._dynamic_reshape_indexer(flat, new_size)
            reindex = fuse_reindexing(reindex1, reindex2)
        return reindex

    @staticmethod
    def _dynamic_reshape_indexer(old_size, new_size):
        """
        Perform a reshape entirely by modifying indexing math
        """
        size_hint = V.graph.sizevars.size_hint
        vars = [sympy_symbol(f"view{i}") for i in range(len(new_size))]

        stack_new = list(zip(vars, new_size))
        stack_old = list(old_size)

        view_expr = []
        while stack_new and stack_old:
            size_old = stack_old.pop()
            var, size_new = stack_new.pop()
            if size_old == 1:
                view_expr.append(sympy.Integer(0))
                stack_new.append((var, size_new))  # re-add
            elif size_new == 1:
                stack_old.append(size_old)  # re-add
            elif size_hint(size_new) == size_hint(size_old):
                view_expr.append(var)
                V.graph.sizevars.guard_equals(size_new, size_old)
            elif size_hint(size_new) < size_hint(size_old):
                while size_hint(size_new) < size_hint(size_old):
                    var2, size_new2 = stack_new.pop()
                    var = var2 * size_new + var
                    size_new = size_new * size_new2
                view_expr.append(var)
                V.graph.sizevars.guard_equals(size_new, size_old)
            elif size_hint(size_new) > size_hint(size_old):
                divisor = sympy.Integer(1)
                modulus = size_old
                view_expr.append(ModularIndexing(var, divisor, modulus))
                divisor = divisor * modulus
                while size_hint(size_new) > size_hint(size_old):
                    modulus = stack_old.pop()
                    view_expr.append(ModularIndexing(var, divisor, modulus))
                    divisor = divisor * modulus
                    size_old = size_old * modulus
                V.graph.sizevars.guard_equals(size_new, size_old)
            else:
                raise AssertionError()

        while stack_old:
            size_old = stack_old.pop()
            assert size_old == 1
            view_expr.append(sympy.Integer(0))

        while stack_new:
            var, size_new = stack_new.pop()
            assert size_new == 1

        view_expr = list(reversed(view_expr))
        assert len(view_expr) == len(old_size)

        def reindex(index):
            assert len(index) == len(vars), (len(index), len(vars))
            replacements = dict(zip(vars, index))
            return tuple(sympy_subs(x, replacements) for x in view_expr)

        return reindex

    def get_size(self):
        return self.size

    def make_loader(self):
        def load(index):
            return inner(self.reindex(index))

        inner = self.data.make_loader()
        return load


@dataclasses.dataclass
class ReinterpretView(BaseView):
    """Pretend our storage has a different layout"""

    layout: "Layout"

    def __str__(self):
        return self.str_helper(
            [
                self.data,
                self.layout,
            ]
        )

    __repr__ = __str__

    def get_name(self):
        return self.data.get_name()

    def get_device(self):
        return self.layout.device

    def get_dtype(self):
        return self.layout.dtype

    def get_size(self):
        return self.layout.size

    def get_stride(self):
        return self.layout.stride

    def make_loader(self):
        def loader(index):
            indexer = self.layout.make_indexer()
            return ops.load(self.get_name(), indexer(index))

        return loader

    def make_indexer(self):
        return self.layout.make_indexer()

    def get_layout(self):
        return self.layout

    def freeze_layout(self):
        pass

    def codegen_reference(self):
        size = V.graph.sizevars.codegen_shape_tuple(self.layout.size)
        stride = V.graph.sizevars.codegen_shape_tuple(self.layout.stride)
        offset = V.graph.sizevars.codegen_sizevar(self.layout.offset)
        if offset != "0":
            return f"as_strided({self.get_name()}, {size}, {stride}, {offset})"
        return f"as_strided({self.get_name()}, {size}, {stride})"


class SliceView(View):
    @classmethod
    def create(cls, x, dim, start, end, step=1):
        step = sympy.expand(step)
        assert step > 0
        try:
            if start == 0 and end >= 2**63 and step == 1:
                return x
        except TypeError:
            pass

        sizevars = V.graph.sizevars
        new_size = list(x.get_size())

        start = cls.handle_negative_index(start, new_size[dim])
        end = cls.handle_negative_index(end, new_size[dim])

        end = sizevars.guard_min(end, new_size[dim])
        start = sizevars.guard_min(sizevars.guard_min(start, new_size[dim]), end)
        if start == 0 and sizevars.size_hint(end - new_size[dim]) == 0 and step == 1:
            sizevars.guard_equals(end, new_size[dim])
            return x

        new_size[dim] = IndexingDiv(end - start + (step - 1), step)

        if is_storage_and_layout(x):
            # Fast path
            storage, old_layout = as_storage_and_layout(x)
            new_stride = list(old_layout.stride)
            new_stride[dim] = new_stride[dim] * step
            new_layout = FixedLayout(
                old_layout.device,
                old_layout.dtype,
                new_size,
                new_stride,
                old_layout.offset + old_layout.stride[dim] * start,
            )
            return ReinterpretView(storage, new_layout)

        def reindex(index):
            assert len(index) == len(new_size), f"wrong ndim {index} {new_size}"
            index = list(index)
            index[dim] = index[dim] * step + start
            return index

        # redirect to a generic view
        return SliceView(x, size=new_size, reindex=reindex)


class BaseConstant(IRNode):
    def get_size(self):
        return ()

    def get_dtype(self):
        return self.dtype

    def get_device(self):
        return self.device

    def mark_reuse(self, users):
        pass

    def has_exceeded_max_reads(self):
        return False

    def get_reads(self):
        return ()

    def is_extern(self):
        return False


@dataclasses.dataclass
class Constant(BaseConstant):
    value: Any
    dtype: torch.dtype
    device: torch.device

    def make_loader(self):
        def loader(index):
            return ops.constant(self.value, self.dtype)

        return loader


@dataclasses.dataclass
class IndexingConstant(BaseConstant):
    index: Any
    dtype: torch.dtype
    device: torch.device

    def make_loader(self):
        def loader(index):
            return ops.index_expr(self.index, self.dtype)

        return loader


@dataclasses.dataclass
class Layout(IRNode):
    def __init__(
        self,
        device: torch.device,
        dtype: torch.dtype,
        size: List[Expr],
        stride: List[Expr],
        offset: Expr = Integer(0),
    ):
        self.device = device
        self.dtype = dtype
        self.size = size
        self._stride = stride
        self.offset = offset

    @property
    def stride(self):
        return self._stride

    def __str__(self):
        offset = ""
        if self.offset != 0:
            offset = f", offset={self.offset}"
        return (
            f"{type(self).__name__}('{self.device.type}', {self.dtype}, "
            f"size={self.size}, stride={self.stride}{offset})"
        )

    __repr__ = __str__

    def is_contiguous(self):
        for left, right, size in zip(
            self.stride, FlexibleLayout.contiguous_strides(self.size), self.size
        ):
            if size != 1 and left != right:
                return False
        return True

    def is_transposed(self):
        for left, right, size in zip(
            self.stride,
            reversed(FlexibleLayout.contiguous_strides(self.size)),
            self.size,
        ):
            if size != 1 and left != right:
                return False
        return True

    def is_stride_ordered(self, order):
        assert len(self.stride) == len(order)
        # reorder the stride given order
        stride_ordered = [None] * len(order)
        for i in range(len(order)):
            stride_ordered[order[i]] = V.graph.sizevars.size_hint(self.stride[i])
        # check if it is in ascending order
        for i in range(len(order) - 1):
            if stride_ordered[i] > stride_ordered[i + 1]:
                return False
        return True

    def is_channels_last_stride_ordered(self):
        # create channels_last order(NCHW, NCDHW, the C is the first order).
        order = [0] + list(reversed(range(1, len(self.stride) - 1)))
        order = [len(order)] + order
        return self.is_stride_ordered(order)

    def as_fixed(self):
        return FixedLayout(
            self.device,
            self.dtype,
            self.size,
            self.stride,
            self.offset,
        )

    def make_indexer(self):
        assert (
            FlexibleLayout.allow_indexing
        ), f"convert {type(self).__name__} to FixedLayout first"
        return self.as_fixed().make_indexer()

    def __eq__(self, other) -> bool:
        return (
            self.device == other.device
            and self.dtype == other.dtype
            and self.size == other.size
            and self.stride == other.stride
            and self.offset == other.offset
        )


class FixedLayout(Layout):
    """A Tensor layout we cannot change"""

    def make_indexer(self):
        """A closure containing math to read a given element"""

        def indexer(index):
            assert len(index) == len(self.stride) == len(self.size)
            result = self.offset
            for idx, stride, sz in zip(index, self.stride, self.size):
                if sz != 1:
                    result = result + idx * stride
            return result

        return indexer


class FlexibleLayout(Layout):
    """A Tensor layout we are allowed to change"""

    allow_indexing = False

    @staticmethod
    def contiguous_strides(sizes):
        if len(sizes) == 0:
            return []
        reversed_strides = [sympy.Integer(1)]
        for size in reversed(sizes[1:]):
            reversed_strides.append(size * reversed_strides[-1])
        return list(reversed(reversed_strides))

    @staticmethod
    def fill_ordered(sizes, order):
        """
        Create a stride based on the order the dimensions should be filled in.

        In this format, channels last would be:
            [1, 3, 2, 0]
        """
        assert set(range(len(sizes))) == set(order)
        next_stride = sympy.Integer(1)
        strides = [None] * len(order)

        for i in order:
            strides[i] = next_stride
            next_stride = next_stride * sizes[i]
        return strides

    @staticmethod
    def stride_ordered(sizes, order):
        """
        Create a stride based on the sorted order of a permuted range.

        In this format, channels last would be:
            [3, 0, 2, 1]
        """
        assert set(range(len(sizes))) == set(order)
        fill_order = stride_order2fill_order(order)
        return FlexibleLayout.fill_ordered(sizes, fill_order)

    @staticmethod
    def same_ordered(sizes, stride):
        """
        Create a stride that has the same stride order as given stride

        For example, if given stride is [1000, 1, 100, 10],
        the fill order should be [1, 3, 2, 0]
        """
        assert len(sizes) == len(stride)
        stride = [V.graph.sizevars.size_hint(x) for x in stride]
        fill_order = sorted(range(len(stride)), key=stride.__getitem__)
        return FlexibleLayout.fill_ordered(sizes, fill_order)

    def as_stride_order(self, order):
        return FixedLayout(
            self.device,
            self.dtype,
            self.size,
            self.stride_ordered(self.size, order),
            self.offset,
        )

    def as_fill_order(self, order):
        return FixedLayout(
            self.device,
            self.dtype,
            self.size,
            self.fill_ordered(self.size, order),
            self.offset,
        )

    def as_same_order(self, stride):
        return FixedLayout(
            self.device,
            self.dtype,
            self.size,
            self.same_ordered(self.size, stride),
            self.offset,
        )

    def __init__(self, device, dtype, size, stride_order=None):
        super(FlexibleLayout, self).__init__(
            device, dtype, size, FlexibleLayout.contiguous_strides(size)
        )
        self.preferred_stride_order = stride_order


class AliasedLayout(Layout):
    """Shares the same storage as another tensor"""

    def __init__(self, view: "ReinterpretView"):
        layout = view.get_layout()
        super().__init__(
            layout.device,
            layout.dtype,
            layout.size,
            layout.stride,
        )
        self.view = view

    def make_indexer(self):
        return self.as_fixed().make_indexer()

    def maybe_guard_aligned(self):
        offset = self.view.get_layout().offset
        if offset == 0:
            return True
        from .compile_fx import ALIGNMENT

        return V.graph.sizevars.maybe_guard_multiple_of(offset, ALIGNMENT)


class MutationLayout(Layout):
    def __init__(self, target: IRNode):
        super().__init__(
            target.get_device(),
            target.get_dtype(),
            target.get_size(),
            None,  # type: ignore[arg-type]
        )
        self.target = target

    @Layout.stride.getter
    def stride(self):
        return self.real_layout().stride

    def real_layout(self):
        if isinstance(self.target, MutationLayout):
            return self.target.real_layout()
        return self.target.data.layout

    @classmethod
    def realize_into(cls, src, dst):
        dst.realize()
        V.graph.realize_users_of(dst.get_name())

        if isinstance(src, TensorBox):
            src = src.data

        if not isinstance(src, StorageBox) or src.is_user_of(dst.get_name()):
            need_copy = True
        else:
            src.realize()
            need_copy = not isinstance(src.data.layout, FlexibleLayout)

        if need_copy:
            src = Pointwise.create(
                device=src.get_device(),
                dtype=src.get_dtype(),
                inner_fn=src.make_loader(),
                ranges=[
                    V.graph.sizevars.guard_equals(a, b)
                    for a, b in zip(src.get_size(), dst.get_size())
                ],
            ).data
            src.realize()

        assert isinstance(src.data.layout, FlexibleLayout)
        src.data.layout = MutationLayout(dst)
        return src.data

    def as_fixed(self):
        return self

    def make_indexer(self):
        return self.target.make_indexer()


@dataclasses.dataclass
class Buffer(IRNode):
    name: str
    layout: Layout

    def make_indexer(self):
        return self.layout.make_indexer()

    def get_name(self):
        assert self.name
        return self.name

    def get_device(self):
        return self.layout.device

    def get_dtype(self):
        return getattr(self.layout, "dtype", None)

    def get_size(self):
        return self.layout.size

    def get_stride(self):
        return self.layout.stride

    def get_layout(self):
        return self.layout

    def get_storage_numel(self):
        return self.get_numel()

    def is_extern(self):
        return False

    def freeze_layout(self):
        if not isinstance(self.layout, MultiOutputLayout):
            self.layout = self.layout.as_fixed()

    def freeze_layout_with_stride_order(self, order):
        assert isinstance(self.layout, FlexibleLayout)
        self.layout = self.layout.as_stride_order(order)

    def freeze_layout_with_fill_order(self, order):
        assert isinstance(self.layout, FlexibleLayout)
        self.layout = self.layout.as_fill_order(order)

    def freeze_layout_with_same_order(self, stride):
        assert isinstance(self.layout, FlexibleLayout)
        self.layout = self.layout.as_same_order(stride)

    def make_loader(self):
        def loader(index):
            indexer = self.layout.make_indexer()
            return ops.load(self.name, indexer(index))

        return loader

    def is_no_op(self):
        return False

    def codegen_reference(self):
        return self.get_name()

    def decide_layout(self):
        pass

    def get_alias_names(self):
        if isinstance(self.layout, AliasedLayout):
            return [self.layout.view.get_name()]
        return ()

    def get_mutation_names(self):
        if isinstance(self.layout, MutationLayout):
            return [self.layout.target.get_name()]
        return ()

    @cache_on_self
    def get_read_writes(self):
        with patch.object(FlexibleLayout, "allow_indexing", True):
            return extract_read_writes(
                self.make_loader(),
                self.get_size(),
            )

    def get_reads(self):
        return self.get_read_writes().reads

    def realize(self):
        pass


class InputBuffer(Buffer):
    pass


class ConstantBuffer(InputBuffer):
    override_device = None

    def make_loader(self):
        def loader(index):
            indexer = self.layout.make_indexer()
            return ops.load(
                V.graph.constant_name(self.name, self.override_device), indexer(index)
            )

        return loader

    def constant_to_device(self, device):
        return ConstantBuffer(V.graph.constant_name(self.name, device), self.layout)


class RandSeedBuffer(ConstantBuffer):
    def codegen_reference(self):
        # Clone makes sure if we pass this from forwards to backwards
        # the value does not get clobbered by the time backwards is run.
        return self.get_name() + ".clone()"


class NoneAsConstantBuffer(IRNode):
    def codegen_reference(self):
        return "None"


class ShapeAsConstantBuffer(IRNode):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def codegen_reference(self):
        return str(self.shape)


@dataclasses.dataclass
class ComputedBuffer(Buffer):
    data: Loops

    @cache_on_self
    def get_read_writes(self):
        with patch.object(FlexibleLayout, "allow_indexing", True):
            if self.data.get_reduction_type():
                return extract_read_writes(
                    self.get_store_function(),
                    self.data.get_size(),
                    self.data.get_reduction_size(),
                )
            else:
                return extract_read_writes(
                    self.get_store_function(),
                    self.data.get_size(),
                )

    def get_store_function(self):
        indexer = self.layout.as_fixed().make_indexer()
        if self.data.get_reduction_type():
            return partial(self.data.store_reduction, self.name, indexer)
        else:
            return partial(self.data.store_output, self.name, indexer)

    def decide_layout(self):
        """
        If our layout is still flexible, try to set it based on stride orders of reads.

        TODO(jansel): A better algorithm here would look at downstream consumers of this
                      value and try to do global graph-level layout optimization.
                      This is also something just begging to be autotuned.
        """
        if isinstance(self.layout, FlexibleLayout):
            _, (index_vars, reduction_vars), _ = dependencies.index_vars_squeeze(
                self.data.get_size(), self.data.get_reduction_size()
            )
            reads = self.get_read_writes().reads
            reads_bufs = [
                V.graph.name_to_buffer[r.name]
                if r.name in V.graph.name_to_buffer.keys()
                else None
                for r in reads
            ]
            priority_idx = []
            for i, reads_buf in enumerate(reads_bufs):
                if (
                    isinstance(reads_buf, Convolution)
                    and reads_buf.kernel != "aten.convolution"
                ):
                    # prioritize Conv layout order
                    priority_idx.append(i)
            # only consider reads to buffer of same size
            reads = [
                sympy_subs(
                    r.index, {v: sympy.Integer(0) for v in reduction_vars if v != 0}
                )
                for r in reads
            ]

            if reads:
                stride_lengths = numpy.array(
                    [V.graph.sizevars.stride_hints(expr, index_vars) for expr in reads],
                    dtype=numpy.int64,
                )
                from .scheduler import pick_loop_order

                self.freeze_layout_with_fill_order(
                    pick_loop_order(stride_lengths, self.get_size(), priority_idx)
                )

        if isinstance(self.layout, FlexibleLayout):
            self.freeze_layout()

    def simplify_and_reorder(self):
        """
        This is a main place where we do loop transformations in a
        backend-agnostic way.

        Here we:
            1) Remove any 1 dimensions
            2) Fuse contiguous dimensions together
            3) Reorder dimensions based on stride orders
        """
        _, args, var_ranges = dependencies.index_vars_squeeze(
            self.data.get_size(), self.data.get_reduction_size(), prefix="q"
        )
        with patch.object(ConstantBuffer, "override_device", self.get_device()):
            body = LoopBody(
                self.get_store_function(),
                (args if self.get_reduction_type() else args[:1]),
                var_ranges,
            )
        index_formulas = [*body.indexing_exprs.values()]
        reads_bufs = [
            V.graph.name_to_buffer[reads_name]
            if reads_name in V.graph.name_to_buffer.keys()
            else None
            for reads_name in body.reads_name2expr.keys()
        ]
        priority_idx = []
        if config.triton.convolution == "aten":
            memory_addrs = [
                *body.reads_name2expr.values(),
                *body.writes_name2expr.values(),
            ]
        else:
            # prioritize reads layout/loop_ordering over writes
            if len(body.reads_name2expr.values()) > 0:
                memory_addrs = [*body.reads_name2expr.values()]
            else:
                memory_addrs = [*body.writes_name2expr.values()]
            for i, reads_buf in enumerate(reads_bufs):
                if isinstance(reads_buf, Convolution):
                    priority_idx.append(i)
        index_vars = []
        reduce_vars = []
        index_size = []
        reduce_size = []
        for v, s in var_ranges.items():
            if v in args[0]:
                assert not reduce_vars
                index_vars.append(v)
                index_size.append(s)
            else:
                assert v in args[1]
                reduce_vars.append(v)
                reduce_size.append(s)

        # the reordering_reindex in reads' simplify_reorder_and_tile
        reordering_reindex = [same_reorder(range(len(index_vars)))] * len(memory_addrs)
        for i, reads_buf in enumerate(reads_bufs):
            if isinstance(reads_buf, ComputedBuffer) and hasattr(
                reads_buf, "iter_reordering_reindex"
            ):
                reordering_reindex[i] = reads_buf.iter_reordering_reindex

        def simplify_and_reorder(x_vars, sizes, reordering_reindex=None):
            sizes, reindex0, reindex1 = self._apply_loop_reordering(
                x_vars, sizes, memory_addrs, reordering_reindex, priority_idx
            )
            # for NHWC: reindex0([0,1,2,3]) = [0,2,3,1], reindex1([0,1,2,3]) = [0,3,2,1]
            x_vars = reindex0(x_vars)
            sizes, reindex2, prune = V.graph.sizevars._simplify_loops(
                x_vars,
                sizes,
                index_prevent_reordering(index_formulas, x_vars, sizes),
            )
            x_vars = prune(x_vars)
            # sizes, reindex1, prune = _simplify_loops(x_vars, sizes, index_formulas)
            # x_vars = prune(x_vars)
            # sizes, reindex2 = self._apply_loop_reordering(x_vars, sizes, memory_addrs)
            reindex = fuse_reindexing(reindex1, reindex2)
            return sizes, reindex, reindex1

        iter_ranges, iter_reindex, iter_reordering_reindex = simplify_and_reorder(
            index_vars, index_size, reordering_reindex
        )
        reduce_ranges, reduce_reindex, _ = simplify_and_reorder(
            reduce_vars, reduce_size
        )

        # remember the reordering order
        self.iter_reordering_reindex = iter_reordering_reindex
        # retrace the loop body with simplification and reordering applied
        (iter_vars, reduce_vars), var_ranges = dependencies.index_vars_no_squeeze(
            iter_ranges, reduce_ranges, prefix="z"
        )
        body = LoopBody(
            body, [iter_reindex(iter_vars), reduce_reindex(reduce_vars)], var_ranges
        )
        return (iter_ranges, reduce_ranges), body

    @staticmethod
    def _apply_loop_reordering(
        index_vars, sizes, memory_addrs, reordering_reindex=None, priority_idx=None
    ):
        """
        Shuffle the order of loops around to hopefully improve performance.
        """
        from .scheduler import pick_loop_order

        if priority_idx is None:
            priority_idx = []

        try:
            strides = numpy.array(
                [
                    V.graph.sizevars.stride_hints(expr, index_vars)
                    for expr in memory_addrs
                ],
                dtype=numpy.int64,
            )
            assert strides.shape == (len(memory_addrs), len(index_vars))
            # consider both layout(strides) and reordering(reordering_reindex)
            if reordering_reindex is not None:
                for i in range(len(memory_addrs)):
                    try:
                        strides[i] = reordering_reindex[i](strides[i])
                    # if len(order) != len(strides), do not reorder
                    except AssertionError:
                        pass
            order = list(reversed(pick_loop_order(strides, sizes, priority_idx)))
        except Exception:
            if config.debug:
                log.warning(
                    f"Did not simplify complex index:\n{dict(zip(index_vars, sizes))}\n{memory_addrs}"
                )
            order = list(range(len(sizes)))
        sizes = [sizes[i] for i in order]
        return sizes, same_reorder(order), inverse_reorder(order)

    def get_reduction_size(self):
        return self.data.get_reduction_size()

    def get_reduction_type(self):
        return self.data.get_reduction_type()

    def is_no_op(self):
        return self.data.is_zero_elements()

    def should_allocate(self):
        return True

    def constant_to_device(self, device):
        """Move this to a given device. Requires that all reads are to constants."""
        return self.data.constant_to_device(device)


@dataclasses.dataclass
class InputsKernel(Buffer):
    inputs: List[Buffer]

    def get_read_writes(self):
        return dependencies.ReadWrites(
            {dependencies.StarDep(x.get_name()) for x in self.inputs},
            {dependencies.StarDep(self.get_name())},
            set(),
            [],
            None,
        )

    @staticmethod
    def unwrap_storage(inputs):
        inputs_new = []
        for x in inputs:
            if isinstance(x, TensorBox):
                x = x.data
            if isinstance(x, StorageBox):
                x = x.data
            if isinstance(x, BaseView) and not isinstance(x, ReinterpretView):
                x = ExternKernel.realize_input(x)
            assert isinstance(x, (Buffer, ReinterpretView)), x
            inputs_new.append(x)
        return inputs_new

    def is_extern(self):
        return True


class NopKernel(InputsKernel):
    def is_no_op(self):
        return True


class ConcatKernel(NopKernel):
    """
    There isn't actually a real kernel for concat, we just change the
    storage for the upstream data.
    """

    @classmethod
    def create(cls, inputs, dim):
        device = inputs[0].get_device()
        dtype = inputs[0].get_dtype()
        new_size = list(inputs[0].get_size())
        offsets_start = [0]
        offsets_end = [new_size[dim]]
        assert 0 <= dim < len(new_size)
        for i in range(1, len(inputs)):
            input_size = inputs[i].get_size()
            offsets_start.append(new_size[dim])
            assert len(input_size) == len(new_size)
            assert inputs[i].get_dtype() == dtype
            assert inputs[i].get_device() == device
            for j in range(len(new_size)):
                if j == dim:
                    new_size[j] = new_size[j] + input_size[j]
                else:
                    new_size[j] = V.graph.sizevars.guard_equals(
                        new_size[j], input_size[j]
                    )
            offsets_end.append(new_size[dim])

        kernel = ConcatKernel(
            name=None,
            layout=FixedLayout(
                device=device,
                dtype=dtype,
                size=new_size,
                stride=FlexibleLayout.contiguous_strides(new_size),
            ),
            inputs=[],
        )
        kernel = StorageBox(kernel)
        for i in range(len(inputs)):
            kernel.data.inputs.append(
                cls.realize_into(
                    inputs[i],
                    SliceView.create(kernel, dim, offsets_start[i], offsets_end[i]),
                )
            )
        kernel.data.name = V.graph.register_buffer(kernel.data)
        kernel.data.inputs = cls.unwrap_storage(kernel.data.inputs)

        return kernel

    @classmethod
    def realize_into(cls, src, dst):
        # Attempt to turn this into a ReinterpretView rather than assert.
        # This has concessions around layout, as as_storage_and_layout
        # can cause us to go from flexible to fixed layout.
        if not isinstance(dst, ReinterpretView):
            if is_storage_and_layout(dst):
                storage, layout = as_storage_and_layout(dst)
                dst = ReinterpretView(storage, layout)
        assert isinstance(dst, ReinterpretView), dst
        if isinstance(src, TensorBox):
            # unwrap a TensorBox
            return cls.realize_into(src.data, dst)
        if isinstance(src, StorageBox):
            src.realize()
            # ExternKernelAlloc has specific requirements for output layout, should create a copy
            if isinstance(src.data.layout, FlexibleLayout) and not isinstance(
                src.data, ExternKernelAlloc
            ):
                src.data.layout = AliasedLayout(dst)
                return src.data
        # introduce a copy
        pw = Pointwise.create(
            device=src.get_device(),
            dtype=src.get_dtype(),
            inner_fn=src.make_loader(),
            ranges=[
                V.graph.sizevars.guard_equals(a, b)
                for a, b in zip(src.get_size(), dst.get_size())
            ],
        )
        return cls.realize_into(pw, dst)

    def should_allocate(self):
        return True


@dataclasses.dataclass
class ExternKernel(InputsKernel):
    constant_args: Tuple[Any, ...] = ()
    kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)
    output_view: Optional[ReinterpretView] = None

    def decide_layout(self):
        if isinstance(self.layout, FlexibleLayout):
            self.apply_constraint()
            self.freeze_layout()

    def codegen(self, wrapper):
        raise NotImplementedError

    @staticmethod
    def copy_input(x):
        pw = Pointwise.create(
            device=x.get_device(),
            dtype=x.get_dtype(),
            inner_fn=x.make_loader(),
            ranges=x.get_size(),
        )
        pw.realize()
        return pw

    @classmethod
    def process_kernel(cls, kernel, *args, **kwargs):
        binded_args = signature(kernel).bind(*args, **kwargs).arguments
        args_flat, args_spec = pytree.tree_flatten(binded_args)

        is_arg_tensor = []
        tensor_args = []
        non_tensor_args = []
        for arg in args_flat:
            is_arg_tensor.append(isinstance(arg, IRNode))
            if is_arg_tensor[-1]:
                tensor_args.append(arg)
            else:
                non_tensor_args.append(arg)

        def unflatten_args(new_tensor_args, new_non_tensor_args):
            result = []
            it_tensors = iter(new_tensor_args)
            it_non_tensors = iter(new_non_tensor_args)
            for is_tensor in is_arg_tensor:
                if is_tensor:
                    result.append(next(it_tensors))
                else:
                    result.append(next(it_non_tensors))
            result = pytree.tree_unflatten(result, args_spec)
            return result.get("args", []), result.get("kwargs", {})

        tensor_args = [cls.realize_input(x) for x in tensor_args]

        # freeze layout otherwise our output stride calculation might
        # become incorrect
        for x in tensor_args:
            if is_storage_and_layout(x):
                as_storage_and_layout(x, freeze=True)

        # We don't have generic shape formulas, so just burn in the
        # shapes and run an example input.
        # TODO(jansel): replace this with dynamic shape formulas
        example_args = []

        for x in tensor_args:
            example_args.append(ir_node_to_tensor(x, guard_shape=True))

        new_args, new_kwargs = unflatten_args(example_args, non_tensor_args)
        example_output = kernel(*new_args, **new_kwargs)

        return example_output, tensor_args, non_tensor_args, unflatten_args

    @classmethod
    def convert_to_reinterpret_view(cls, x):
        """
        In order to pass this to an extern kernel we need a
        ReinterpretView not a View.  This allows us to avoid some
        uneeded copies.
        """
        assert isinstance(x, BaseView)
        if isinstance(x, ReinterpretView):
            return x

        x.unwrap_view().freeze_layout()
        rw = extract_read_writes(x.make_loader(), x.get_size(), normalize=False)
        assert len(rw.reads) == 1

        index = V.graph.sizevars.simplify_with_ranges(
            list(rw.reads)[0].index, rw.var_ranges
        )
        strides = V.graph.sizevars.stride_vars(index, rw.range_vars)
        offset = V.graph.sizevars.offset_var(index, rw.range_vars)
        expected = sympy_dot(rw.range_vars, strides) + offset

        if index != expected:
            log.debug(
                "convert_to_reinterpret_view failed: stride=%s offset=%s index=%s",
                strides,
                offset,
                index,
            )
            raise NotImplementedError()

        return ReinterpretView(
            data=x.data,
            layout=FixedLayout(
                device=x.get_device(),
                dtype=x.get_dtype(),
                size=x.get_size(),
                stride=strides,
                offset=offset,
            ),
        )

    @classmethod
    def realize_input(cls, x):
        if x is None:
            return NoneAsConstantBuffer()
        if isinstance(x, sympy.Expr):
            return ShapeAsConstantBuffer(x)
        if isinstance(x, Constant):
            return V.graph.add_tensor_constant(
                torch.tensor(x.value, dtype=x.get_dtype(), device=x.get_device())
            )
        if isinstance(x, ConstantBuffer):
            return x
        if isinstance(x, TensorBox):
            return cls.realize_input(x.data)
        if isinstance(x, ReinterpretView):
            return x
        if isinstance(x, BaseView):
            x.realize()
            if is_storage_and_layout(x.unwrap_view()) and not isinstance(
                x.unwrap_view().data, ExternKernelAlloc
            ):
                try:
                    return cls.convert_to_reinterpret_view(x)
                except NotImplementedError:
                    pass
        if isinstance(x, StorageBox):
            # TODO(jansel): impose layout preference on realized buffer
            x.realize()
            return x
        return cls.copy_input(x)

    @classmethod
    def require_stride1(cls, x):
        if is_storage_and_layout(x):
            if len(x.get_stride()) == 0:
                return x
            for stride in x.get_stride():
                if stride == 1:
                    return x
        return cls.copy_input(x)

    @classmethod
    def require_stride_order(cls, x, order):
        # require x to have the layout as strided_ordered as order
        if is_storage_and_layout(x):
            if isinstance(
                x.get_layout(), FlexibleLayout
            ) and is_stride_order_storage_and_layout(x, order):
                # fix flexiblelayout to be FixedLayout with stride_order
                as_storage_and_layout(
                    x, freeze=True, want_contiguous=False, stride_order=order
                )
                return x
            elif isinstance(
                x.get_layout(), FixedLayout
            ) and x.get_layout().is_stride_ordered(order):
                return x
            elif isinstance(x.get_layout(), MutationLayout):
                if isinstance(x.get_layout().real_layout(), FlexibleLayout):
                    raise AssertionError(
                        "the MutationLayout's real layout shouldn't be FlexibleLayout"
                    )
                elif isinstance(
                    x.get_layout().real_layout(), FixedLayout
                ) and x.get_layout().real_layout().is_stride_ordered(order):
                    return x

        # TODO - Storage to InputBuffer
        if isinstance(x, InputBuffer) and x.get_layout().is_stride_ordered(order):
            return x
        x = cls.copy_input(x)
        as_storage_and_layout(x, freeze=True, want_contiguous=False, stride_order=order)
        assert is_stride_order_storage_and_layout(x, order)
        return x

    @classmethod
    def require_contiguous(cls, x):
        return cls.require_stride_order(x, list(reversed(range(len(x.get_size())))))

    def apply_constraint(self):
        pass

    def codegen_args(self):
        args = [x.codegen_reference() for x in self.inputs]
        args.extend(map(repr, self.constant_args))
        return args

    def codegen_kwargs(self):
        kwargs = []
        if self.kwargs:
            kwargs = [f"{k}={repr(v)}" for k, v in self.kwargs.items()]
        return kwargs

    def codegen_size_asserts(self, wrapper):
        if config.size_asserts:
            size = V.graph.sizevars.codegen_shape_tuple(self.get_size())
            stride = V.graph.sizevars.codegen_shape_tuple(self.get_stride())
            wrapper.writeline(
                f"assert_size_stride({self.get_name()}, {size}, {stride})"
            )

    def get_group_stride(self):
        """
        get output sizes and strides, for template_codegen
        """
        _size = self.get_size()
        _stride = self.get_stride()
        # iter_ranges = _size of output tensor, reduce_range = [] because no reduction
        return [_size, []], _stride

    def canonicalize(self):
        """
        Manually get cononicalization of the output index
        """
        # manually generate index formula for conv
        sizevars = V.graph.sizevars
        sizes = self.get_size()
        strides = self.get_stride()
        strides = [sizevars.size_hint(x) for x in strides]
        index_vars = [sympy_symbol(f"d{i}") for i in range(len(sizes))]
        # reorder index vars according to stride
        index_order = sorted(range(len(strides)), key=strides.__getitem__, reverse=True)
        lookup = {pos: idx for idx, pos in enumerate(index_order)}
        order = [lookup[i] for i in range(len(lookup))]
        index_vars = [index_vars[i] for i in order]
        indexer = self.make_indexer()
        index = indexer(index_vars)

        new_sizes, reindex, prune = V.graph.sizevars._simplify_loops(
            index_vars, sizes, [index]
        )

        # assign new variables each dimension to deal with numbering mismatches
        # d0, d1, d2 could become d0, d2 -- which won't match d0, d1
        _, add_var = var_builder("c")
        replacement = dict(zip(index_vars, reindex([add_var(x) for x in new_sizes])))

        index = sympy_subs(sympy.expand(index), replacement)
        return index, tuple(new_sizes)

    def __str__(self):
        lines = [
            f"{field.name}={getattr(self, field.name)}"
            for field in dataclasses.fields(self)
        ]
        return self.str_helper(lines)


@dataclasses.dataclass
class ExternKernelOut(ExternKernel):
    output_view: Optional[ReinterpretView] = None

    def codegen(self, wrapper):
        args = self.codegen_args()

        kwargs = self.codegen_kwargs()
        if kwargs:
            args.extend(kwargs)

        if self.output_view:
            args.append(f"out={self.output_view.codegen_reference()}")
        else:
            args.append(f"out={self.codegen_reference()}")
        wrapper.writeline(f"{self.kernel}({', '.join(args)})")

    def __init__(self, layout, inputs, constant_args=(), kwargs=None, output_view=None):
        super().__init__(
            None, layout, self.unwrap_storage(inputs), constant_args, kwargs or {}
        )
        self.output_view = output_view
        self.name = V.graph.register_buffer(self)

    def should_allocate(self):
        return True


class ExternKernelAlloc(ExternKernel):
    def codegen(self, wrapper):
        wrapper.writeline(
            f"{self.get_name()} = {self.kernel}({', '.join(self.codegen_args())})"
        )
        if isinstance(self.layout, Layout):
            self.codegen_size_asserts(wrapper)

    def __init__(self, layout, inputs, constant_args=()):
        super().__init__(None, layout, self.unwrap_storage(inputs), constant_args)
        self.name = V.graph.register_buffer(self)

    def should_allocate(self):
        return False

    def apply_constraint(self):
        raise NotImplementedError


class InplaceBernoulliFallback(ExternKernel):
    """
    This needs to be a custom class to handle mutation properly
    """

    kernel = "aten.bernoulli_"

    def codegen(self, wrapper):
        (x,) = [t.codegen_reference() for t in self.inputs]
        wrapper.writeline(
            f"{self.kernel}({x}, {', '.join(map(repr, self.constant_args))})"
        )

    def should_allocate(self):
        return False

    def get_mutation_names(self):
        assert isinstance(self.layout, MutationLayout)
        return (self.layout.target.get_name(),)

    def __init__(self, x, *constant_args):
        super().__init__(
            None,
            MutationLayout(x),
            self.unwrap_storage([x]),
            constant_args,
        )
        self.name = V.graph.register_buffer(self)


class IndexPutFallback(ExternKernel):
    """
    This needs to be a custom class to handle mutation and indices properly
    """

    kernel = "aten.index_put_"

    def codegen(self, wrapper):
        (x, values, *valid_indices) = [t.codegen_reference() for t in self.inputs]
        indices = []
        iter_valid_indices = iter(valid_indices)
        for i, _ in enumerate(self.indices):
            if self.indices[i] is not None:
                indices.append(next(iter_valid_indices))
            else:
                indices.append("None")
        wrapper.writeline(
            f"{self.kernel}({x}, [{','.join(indices)}], {values}, {repr(self.constant_args[0])})"
        )

    def should_allocate(self):
        return False

    def __init__(self, x, indices, values, accumulate):
        self.indices = indices
        valid_indices = [i for i in indices if i is not None]
        tensors = [self.realize_input(x) for x in [x, values, *valid_indices]]
        super().__init__(
            None,
            MutationLayout(x),
            self.unwrap_storage(tensors),
            [accumulate],
        )
        self.name = V.graph.register_buffer(self)


class MatrixMultiply(ExternKernelOut):
    kernel = "aten.mm.out"

    def __init__(
        self, layout, inputs, constant_args=(), output_view=None, kernel="aten.mm.out"
    ):
        super().__init__(layout, inputs, constant_args, output_view)
        self.kernel = kernel

    @classmethod
    def create(cls, a, b):
        *m, k1 = a.get_size()
        k2, n = b.get_size()
        V.graph.sizevars.guard_equals(k1, k2)
        a = cls.realize_input(a)
        b = cls.realize_input(b)
        if len(m) != 1 and not a.get_layout().is_contiguous():
            a = cls.copy_input(a)
        else:
            a = cls.require_stride1(a)
        b = cls.require_stride1(b)

        # choose runtime kernel
        config_mm = config.triton.mm
        # default kernel is aten
        kernel = "aten.mm.out"
        if config_mm == "aten":
            kernel = "aten.mm.out"
        elif config_mm == "triton" and a.get_device().type == "cuda":
            kernel = "triton_ops.matmul_out"
        elif config_mm == "autotune":
            from .codegen.autotuner import tuned_mm

            kernel = tuned_mm(
                a.get_size(),
                b.get_size(),
                a.get_stride(),
                b.get_stride(),
                a.get_device(),
                a.get_dtype(),
            )

        return MatrixMultiply(
            layout=FlexibleLayout(
                device=a.get_device(),
                dtype=a.get_dtype(),
                size=list(m) + [n],
            ),
            inputs=[a, b],
            kernel=kernel,
        )

    def get_template_tiling(self):
        tile1, tile2 = self.get_size()
        return (
            tile1,
            tile2,
            sympy.Integer(1),
        )

    def map_args(self):
        # a, b
        in_args = [x.codegen_reference() for x in self.inputs]
        # const_args = self.constant_args
        inout_dict = OrderedDict(
            [
                ("A", f"{in_args[0]}"),
                ("B", f"{in_args[1]}"),
                ("C", f"{self.get_name()}"),
            ]
        )
        # batch==1 bmm->mm
        if len(self.get_stride()) == 3:
            assert self.get_size()[0] == 1
            stride_cm = self.get_stride()[1]
            stride_cn = self.get_stride()[2]
        else:
            stride_cm = self.get_stride()[0]
            stride_cn = self.get_stride()[1]
        args_dict = OrderedDict(
            [
                ("M", f"{self.inputs[0].get_size()[0]}"),
                ("N", f"{self.inputs[1].get_size()[1]}"),
                ("K", f"{self.inputs[0].get_size()[1]}"),
                ("stride_am", f"{self.inputs[0].get_stride()[0]}"),
                ("stride_ak", f"{self.inputs[0].get_stride()[1]}"),
                ("stride_bk", f"{self.inputs[1].get_stride()[0]}"),
                ("stride_bn", f"{self.inputs[1].get_stride()[1]}"),
                ("stride_cm", f"{stride_cm}"),
                ("stride_cn", f"{stride_cn}"),
            ]
        )
        # accumulator types
        ACC_TYPE = (
            "tl.float32"
            if self.inputs[0].get_dtype()
            in [torch.float16, torch.bfloat16, torch.float32]
            else "tl.int32"
        )
        # dict for tl.constexpr
        const_dict = OrderedDict(
            [
                ("GROUP_M", "8"),
                ("ACC_TYPE", ACC_TYPE),
                ("allow_tf32", f"{torch.backends.cuda.matmul.allow_tf32}"),
            ]
        )

        other_dict = OrderedDict()

        return inout_dict, args_dict, const_dict, other_dict


class MatrixMultiplyAdd(ExternKernelOut):
    def __init__(self, layout, inputs, constant_args=(), kwargs=None, output_view=None):
        super().__init__(layout, inputs, constant_args, kwargs or {}, output_view)
        self.kernel = "aten.addmm.out"

    @classmethod
    def create(cls, inp, a, b, beta, alpha):
        m, k1 = a.get_size()
        k2, n = b.get_size()
        V.graph.sizevars.guard_equals(k1, k2)
        inp = cls.realize_input(inp)
        a = cls.realize_input(a)
        b = cls.realize_input(b)
        a = cls.require_stride1(a)
        b = cls.require_stride1(b)
        return MatrixMultiplyAdd(
            layout=FlexibleLayout(
                device=a.get_device(),
                dtype=a.get_dtype(),
                size=[m] + [n],
            ),
            inputs=[inp, a, b],
            kwargs={"beta": beta, "alpha": alpha},
        )


class BatchMatrixMultiply(ExternKernelOut):
    kernel = "aten.bmm.out"

    def __init__(self, layout, inputs, constant_args=(), output_view=None):
        super().__init__(layout, inputs, constant_args, output_view)
        if (
            config.triton.use_bmm
            and len(inputs) > 0
            and inputs[0].get_device().type == "cuda"
        ):
            self.kernel = "triton_bmm_out"

    @classmethod
    def create(cls, a, b):
        b1, m, k1 = a.get_size()
        b2, k2, n = b.get_size()
        b3 = V.graph.sizevars.guard_equals(b1, b2)
        V.graph.sizevars.guard_equals(k1, k2)
        a = cls.require_stride1(cls.realize_input(a))
        b = cls.require_stride1(cls.realize_input(b))

        output_layout = FlexibleLayout(
            device=a.get_device(),
            dtype=a.get_dtype(),
            size=[b3, m, n],
        ).as_fixed()

        if b3 == 1:
            # convert to normal mm
            data = MatrixMultiply(
                layout=output_layout.as_fixed(),
                inputs=[SqueezeView.create(a, dim=0), SqueezeView.create(b, dim=0)],
            )
            data.output_view = ReinterpretView(
                data,
                FlexibleLayout(
                    device=a.get_device(),
                    dtype=a.get_dtype(),
                    size=[m, n],
                ).as_fixed(),
            )
        else:
            data = BatchMatrixMultiply(
                layout=output_layout,
                inputs=[a, b],
            )
        return data


class DeviceCopy(ExternKernelOut):
    @classmethod
    def create(cls, x, device):
        if not x.is_extern() and all(
            (r.name in V.graph.constants and hasattr(r, "index")) for r in x.get_reads()
        ):
            return x.constant_to_device(device)

        V.graph.device_types.add(device.type)
        V.graph.device_types.add(x.get_device().type)

        log.warning("DeviceCopy")
        return DeviceCopy(
            FlexibleLayout(
                device=device,
                dtype=x.get_dtype(),
                size=x.get_size(),
            ),
            [cls.realize_input(x)],
        )

    def codegen(self, wrapper):
        args = self.codegen_args()
        assert len(args) == 1
        if self.output_view:
            wrapper.writeline(
                f"{self.output_view.codegen_reference()}.copy_({args[0]})"
            )
        else:
            wrapper.writeline(f"{self.codegen_reference()}.copy_({args[0]})")


class DynamicScalar(IRNode):
    """
    The result of a call to aten._local_scalar_dense.

    This is not yet implemented.  The one model (so far) that calls this
    (fastNLP_Bert) does not actually use the result.  So we expect this
    node to get dead code eliminated.
    """

    def get_reads(self):
        return ()


@dataclasses.dataclass
class FallbackKernel(ExternKernelAlloc):
    def __init__(
        self,
        layout,
        kernel,
        tensor_args,
        nontensor_args,
        unflatten_args,
        kwargs=None,
    ):
        super(FallbackKernel, self).__init__(
            layout,
            tuple(tensor_args),
            tuple(nontensor_args),
        )
        if getattr(torch.ops.aten, kernel.__name__, None) is kernel:
            self.kernel = f"aten.{kernel.__name__}"
        else:
            self.kernel = (
                f"{kernel.__module__.replace('._ops.', '.ops.')}.{kernel.__name__}"
            )
        self.unflatten_args = unflatten_args
        self.kwargs = {} if kwargs is None else kwargs
        if self.kernel not in ("aten.convolution_backward",):
            log.warning(f"Using FallbackKernel: {self.kernel}")

    def codegen_args(self):
        @dataclasses.dataclass
        class Shim:
            ref: Any

            def __repr__(self):
                return self.ref

        def gen_kwarg(k, v):
            return f"{k}={repr(v)}"

        tensor_args = [Shim(x.codegen_reference()) for x in self.inputs]
        constant_args = [Shim(repr(x)) for x in self.constant_args]
        args, kwargs = self.unflatten_args(tensor_args, constant_args)
        return list(map(repr, args)) + list(gen_kwarg(k, v) for k, v in kwargs.items())

    @classmethod
    def create(cls, kernel, *args, **kwargs):
        fake_incorrect_kernels = (
            aten._fft_r2c.default,
            aten._fft_r2c.out,
            aten._fft_c2r.default,
            aten._fft_c2c.default,
            aten._fft_c2c.out,
            aten._linalg_svd.default,
            aten._linalg_svd.U,
        )
        context = (
            FakeTensorMode if kernel not in fake_incorrect_kernels else nullcontext
        )
        with context():
            (
                example_output,
                tensor_args,
                non_tensor_args,
                unflatten_args,
            ) = cls.process_kernel(kernel, *args, **kwargs)

        assert tensor_args or isinstance(
            example_output, torch.Tensor
        ), "Not sure where to find device info"
        packed = FallbackKernel(
            MultiOutputLayout(
                tensor_args[0].get_device() if tensor_args else example_output.device
            ),
            kernel,
            tensor_args,
            non_tensor_args,
            unflatten_args,
            kwargs,
        )

        def generate_output(output, index=""):
            if isinstance(output, (list, tuple)):
                return type(output)(
                    generate_output(output[i], f"{index}[{i}]")
                    for i in range(len(output))
                )
            elif isinstance(output, torch.Tensor):
                return MultiOutput(
                    FixedLayout(
                        output.device,
                        output.dtype,
                        [sympy.Integer(s) for s in output.size()],
                        [sympy.Integer(s) for s in output.stride()],
                    ),
                    packed,
                    index,
                )
            else:
                assert output is None, "FallbackKernel output type is not supported"
                return None

        return generate_output(example_output)

    def apply_constraint(self):
        return super().apply_constraint()


@dataclasses.dataclass
class MultiOutputLayout(IRNode):
    device: torch.device


class MultiOutput(ExternKernel):
    def codegen(self, wrapper):
        wrapper.writeline(
            f"{self.get_name()} = {self.inputs[0].get_name()}{self.index}"
        )
        self.codegen_size_asserts(wrapper)

    def __init__(self, layout, input, index: str):
        super().__init__(None, layout, [input], ())
        self.name = V.graph.register_buffer(self)
        self.index = index

    def should_allocate(self):
        return False


class Convolution(ExternKernelAlloc):
    kernel = "aten.convolution"

    def __init__(
        self,
        layout,
        inputs,
        constant_args=(),
        preferred_stride_order=None,
        kernel="aten.convolution",
    ):
        super().__init__(layout, inputs, constant_args)
        self.kernel = kernel
        self.preferred_stride_order = preferred_stride_order

    def codegen(self, wrapper):
        if self.kernel == "triton_ops.conv":
            wrapper.header.writeline(
                f"import {config.inductor_import}.triton_ops.conv as {self.kernel}"
            )
        wrapper.writeline(
            f"{self.get_name()} = {self.kernel}({', '.join(self.codegen_args())})"
        )
        if isinstance(self.layout, Layout):
            self.codegen_size_asserts(wrapper)

    @classmethod
    def create(
        cls,
        x: "TensorBox",
        weight: "TensorBox",
        bias: "TensorBox",
        stride_: List[int],
        padding_: List[int],
        dilation_: List[int],
        transposed: bool,
        output_padding_: List[int],
        groups: int,
    ):
        with torch._subclasses.FakeTensorMode():
            x_fake = ir_node_to_tensor(x, guard_shape=True)
            weight_fake = ir_node_to_tensor(weight, guard_shape=True)
            bias_fake = (
                ir_node_to_tensor(bias, guard_shape=True) if bias is not None else bias
            )
            output = torch.ops.aten.convolution(
                x_fake,
                weight_fake,
                bias_fake,
                stride_,
                padding_,
                dilation_,
                transposed,
                output_padding_,
                groups,
            )
            req_stride_order = get_stride_order(output.stride())

        if config.triton.convolution == "aten":
            weight = cls.require_stride_order(weight, req_stride_order)
            x = cls.require_stride_order(x, req_stride_order)
        else:
            x = cls.require_stride1(cls.realize_input(x))
            weight = cls.require_stride1(cls.realize_input(weight))

        stride = tuple(stride_)
        padding = tuple(padding_)
        dilation = tuple(dilation_)
        assert isinstance(transposed, bool)
        output_padding = tuple(output_padding_)
        assert isinstance(groups, int)

        output_size = output.shape

        weight_shape = [
            sympy.Integer(V.graph.sizevars.guard_static_shape(s))
            for s in weight.get_size()
        ]
        _, _, *kernel_size = weight_shape

        # choose runtime kernel
        config_conv = config.triton.convolution
        if (
            config_conv == "aten"
            or len(kernel_size) != 2  # triton conv only supports conv2d
            or not is_triton(x.get_device())
            or transposed
            or groups != 1
            # or x.get_dtype() == torch.float16
            # or x.get_dtype() == torch.bfloat16
        ):
            kernel = "aten.convolution"
        elif config_conv == "triton":
            kernel = "triton_ops.conv"
        else:
            assert config_conv == "autotune"
            from .codegen.autotuner import tuned_conv

            kernel = tuned_conv(
                x.get_size(),
                weight.get_size(),
                x.get_stride(),
                weight.get_stride(),
                stride,
                padding,
                dilation,
                transposed,
                output_padding,
                groups,
                x.get_device(),
                x.get_dtype(),
            )

        # for conv2d or conv3d, prefer channels last format
        if kernel == "triton_ops.conv":
            output_layout_str = "torch.channels_last"

        elif config.tune_layout and len(x.get_size()) == 4:
            from .codegen.autotuner import tuned_conv_layout

            output_layout_str = tuned_conv_layout(
                kernel,
                x.get_size(),
                weight.get_size(),
                stride,
                padding,
                dilation,
                transposed,
                output_padding,
                groups,
                x.get_device(),
                x.get_dtype(),
            )

        else:
            output_layout_str = (
                "torch.contiguous_format"
                if output.is_contiguous()
                else "torch.channels_last"
            )

        if output_layout_str == "torch.channels_last":
            stride_order = [0] + list(reversed(range(1, len(kernel_size) + 1)))
            if len(stride_order) < len(output_size):
                # add batch dim if it exists
                stride_order = [len(stride_order)] + stride_order
            strides = make_channels_last_strides_for(output_size)
        else:
            stride_order = list(reversed(range(len(output_size))))
            strides = make_contiguous_strides_for(output_size)

        if config.triton.convolution != "aten":
            x = cls.require_stride_order(x, stride_order)

        output_layout = FixedLayout(
            x.get_device(),
            x.get_dtype(),
            output_size,
            strides,
        )

        if bias is not None:
            return Convolution(
                output_layout,
                (x, weight, bias),
                (stride, padding, dilation, transposed, output_padding, groups),
                stride_order,
                kernel,
            )
        else:
            return Convolution(
                output_layout,
                (x, weight),
                (bias, stride, padding, dilation, transposed, output_padding, groups),
                stride_order,
                kernel,
            )

    def map_args(self):
        # x, w, bias
        in_args = [x.codegen_reference() for x in self.inputs]
        # stride, padding, dilation, transposed, output_padding, groups
        const_args = self.constant_args
        if len(in_args) < 3:
            # otherwise, bias=None is the first constant_args
            const_args = const_args[1:]

        inout_dict = OrderedDict(
            [
                ("x", f"{in_args[0]}"),
                ("w", f"{in_args[1]}"),
                ("y", f"{self.get_name()}"),
            ]
        )
        args_dict = OrderedDict(
            [
                ("stride_xn", f"{self.inputs[0].get_stride()[0]}"),
                ("stride_xc", f"{self.inputs[0].get_stride()[1]}"),
                ("stride_xh", f"{self.inputs[0].get_stride()[2]}"),
                ("stride_xw", f"{self.inputs[0].get_stride()[3]}"),
                ("stride_wn", f"{self.inputs[1].get_stride()[0]}"),
                ("stride_wc", f"{self.inputs[1].get_stride()[1]}"),
                ("stride_wh", f"{self.inputs[1].get_stride()[2]}"),
                ("stride_ww", f"{self.inputs[1].get_stride()[3]}"),
                ("stride_yn", f"{self.get_stride()[0]}"),
                ("stride_yc", f"{self.get_stride()[1]}"),
                ("stride_yh", f"{self.get_stride()[2]}"),
                ("stride_yw", f"{self.get_stride()[3]}"),
                (
                    "stride_biasn",
                    f"{self.inputs[0].get_stride()[0]}"
                    if len(in_args) >= 3
                    else "None",
                ),
                # ("delta_x_ptr", "None"),
                ("BATCH", f"{self.inputs[0].get_size()[0]}"),
                ("IN_C", f"{self.inputs[0].get_size()[1]}"),
                ("IN_H", f"{self.inputs[0].get_size()[2]}"),
                ("IN_W", f"{self.inputs[0].get_size()[3]}"),
                ("KERNEL_N", f"{self.inputs[1].get_size()[0]}"),
                ("KERNEL_H", f"{self.inputs[1].get_size()[2]}"),
                ("KERNEL_W", f"{self.inputs[1].get_size()[3]}"),
                ("OUT_H", f"{self.get_size()[2]}"),
                ("OUT_W", f"{self.get_size()[3]}"),
                ("stride_h", f"{const_args[0][0]}"),
                ("stride_w", f"{const_args[0][1]}"),
                ("padding_h", f"{const_args[1][0]}"),
                ("padding_w", f"{const_args[1][1]}"),
                ("dilation_h", f"{const_args[2][0]}"),
                ("dilation_w", f"{const_args[2][1]}"),
                # ("transposed", f"{const_args[3]}"),
                ("output_padding_h", f"{const_args[4][0]}"),
                ("output_padding_w", f"{const_args[4][1]}"),
                ("groups", f"{const_args[5]}"),
            ]
        )

        # accumulator type
        ACC_TYPE = (
            "tl.float32"
            if self.inputs[0].get_dtype()
            in [torch.float16, torch.bfloat16, torch.float32]
            else "tl.int32"
        )
        CONV1X1_NHWC = (
            "True"
            if self.inputs[0].get_stride()[1] == 1
            and self.inputs[1].get_size()[2] == 1
            and self.inputs[1].get_size()[3] == 1
            else "False"
        )
        # dict for tl.constexpr
        const_dict = OrderedDict(
            [
                ("ACC_TYPE", ACC_TYPE),
                ("CONV1X1_NHWC", CONV1X1_NHWC),
            ]
        )

        # dict for non-kernel args (e.g. delta_x_ptr)
        other_dict = OrderedDict(
            [
                ("device", f'"{self.inputs[0].get_device()}"'),
            ]
        )

        return inout_dict, args_dict, const_dict, other_dict

    def get_template_tiling(self):
        n, c, h, w = self.get_size()
        return (
            n * h * w,
            c,
            sympy.Integer(1),
        )


def _prepare_convolution_fusion_create(
    cls,
    x: "TensorBox",
    weight: "TensorBox",
    bias: "TensorBox",
    padding_: List[int],
    stride_: List[int],
    dilation_: List[int],
    groups: int,
):
    """
    This function is a helper function to prepare inputs, layout and constant args
    for convolution post-op fusion's create function, including deciding the output
    layout (channels first or channels last), realizing inputs and make them etc. The
    function only supports the CPU device since conv post-op fusion kernel is only
    supported on CPU right now.
    """

    x = cls.require_stride1(cls.realize_input(x))
    weight = cls.require_stride1(cls.realize_input(weight))
    assert x.get_device().type == "cpu" and weight.get_device().type == "cpu"
    inputs = [x, weight]
    stride = tuple(stride_)
    padding = tuple(padding_)
    dilation = tuple(dilation_)
    assert isinstance(groups, int)
    with FakeTensorMode():
        output, *_ = cls.process_kernel(
            torch.ops.aten.convolution,
            x,
            weight,
            bias,
            stride,
            padding,
            dilation,
            False,
            [0, 0],
            groups,
        )

    output_size = output.shape
    weight_shape = [
        sympy.Integer(V.graph.sizevars.guard_static_shape(s)) for s in weight.get_size()
    ]
    _, _, *kernel_size = weight_shape
    output_layout_str = (
        "torch.contiguous_format" if output.is_contiguous() else "torch.channels_last"
    )

    if output_layout_str == "torch.channels_last":
        stride_order = [0] + list(reversed(range(1, len(kernel_size) + 1)))
        if len(stride_order) < len(output_size):
            # add batch dim if it exists
            stride_order = [len(stride_order)] + stride_order
    else:
        stride_order = list(reversed(range(len(output_size))))

    kernel_layout = FlexibleLayout(
        device=inputs[0].get_device(),
        dtype=inputs[0].get_dtype(),
        size=output_size,
        stride_order=stride_order,
    )
    constant_args = [padding, stride, dilation, groups]

    if bias is not None:
        inputs.append(bias)
    else:
        constant_args.insert(0, bias)
    return inputs, constant_args, kernel_layout


class ConvolutionUnary(ExternKernelAlloc):
    kernel = "torch.ops.mkldnn._convolution_pointwise"

    def __init__(
        self,
        layout,
        inputs,
        constant_args=(),
        kernel="torch.ops.mkldnn._convolution_pointwise",
    ):
        super().__init__(layout, inputs, constant_args)
        self.kernel = kernel

    def codegen(self, wrapper):
        wrapper.writeline(
            f"{self.get_name()} = {self.kernel}({', '.join(self.codegen_args())})"
        )
        if isinstance(self.layout, Layout):
            self.codegen_size_asserts(wrapper)

    @classmethod
    def create(
        cls,
        x: "TensorBox",
        weight: "TensorBox",
        bias: "TensorBox",
        padding_: List[int],
        stride_: List[int],
        dilation_: List[int],
        groups: int,
        attr,
        scalars,
        algorithm,
    ):
        kernel = "torch.ops.mkldnn._convolution_pointwise"
        (inputs, constant_args, kernel_layout,) = _prepare_convolution_fusion_create(
            cls, x, weight, bias, padding_, stride_, dilation_, groups
        )
        constant_args = constant_args + [attr, scalars, algorithm]
        return ConvolutionUnary(
            layout=kernel_layout,
            inputs=inputs,
            constant_args=constant_args,
            kernel=kernel,
        )

    def apply_constraint(self):
        x = self.inputs[0]
        # FixedLayout of input
        x = self.require_stride_order(x, self.layout.preferred_stride_order)
        self.inputs[0] = x
        self.freeze_layout_with_stride_order(self.layout.preferred_stride_order)


class ConvolutionBinary(ExternKernelAlloc):
    kernel = "torch.ops.mkldnn._convolution_pointwise.binary"

    def __init__(
        self,
        layout,
        inputs,
        constant_args=(),
        kernel="torch.ops.mkldnn._convolution_pointwise.binary",
    ):
        super().__init__(layout, inputs, constant_args)
        self.kernel = kernel

    def codegen(self, wrapper):
        wrapper.writeline(
            f"{self.get_name()} = {self.kernel}({', '.join(self.codegen_args())})"
        )
        if isinstance(self.layout, Layout):
            self.codegen_size_asserts(wrapper)

    @classmethod
    def create(
        cls,
        x: "TensorBox",
        other: "TensorBox",
        weight: "TensorBox",
        bias: "TensorBox",
        padding_: List[int],
        stride_: List[int],
        dilation_: List[int],
        groups: int,
        binary_attr: str,
        binary_alpha: Optional[float],
        unary_attr: Optional[str],
        unary_scalars: Optional[List],
        unary_algorithm: Optional[str],
    ):
        kernel = "torch.ops.mkldnn._convolution_pointwise.binary"
        (inputs, constant_args, kernel_layout,) = _prepare_convolution_fusion_create(
            cls, x, weight, bias, padding_, stride_, dilation_, groups
        )
        other = cls.require_stride1(cls.realize_input(other))
        inputs.insert(1, other)
        constant_args = constant_args + [
            binary_attr,
            binary_alpha,
            unary_attr,
            unary_scalars,
            unary_algorithm,
        ]
        return ConvolutionBinary(
            layout=kernel_layout,
            inputs=inputs,
            constant_args=constant_args,
            kernel=kernel,
        )

    def apply_constraint(self):
        x = self.inputs[0]
        # FixedLayout of input
        x = self.require_stride_order(x, self.layout.preferred_stride_order)
        self.inputs[0] = x
        other = self.inputs[1]
        # FixedLayout of other
        other = self.require_stride_order(other, self.layout.preferred_stride_order)
        self.inputs[1] = other
        self.freeze_layout_with_stride_order(self.layout.preferred_stride_order)


class ConvolutionBinaryInplace(ExternKernelAlloc):
    kernel = "torch.ops.mkldnn._convolution_pointwise_.binary"

    def __init__(
        self,
        kernel_layout,
        inputs_layout,
        inputs,
        constant_args=(),
        kernel="torch.ops.mkldnn._convolution_pointwise_.binary",
    ):
        super().__init__(kernel_layout, inputs, constant_args)
        self.kernel = kernel
        self.inputs_layout = inputs_layout

    def codegen(self, wrapper):
        wrapper.writeline(
            f"{self.get_name()} = {self.kernel}({', '.join(self.codegen_args())})"
        )

    def get_mutation_names(self):
        assert isinstance(self.layout, MutationLayout)
        return (self.layout.target.get_name(),)

    @classmethod
    def create(
        cls,
        x: "TensorBox",
        other: "TensorBox",
        weight: "TensorBox",
        bias: "TensorBox",
        padding_: List[int],
        stride_: List[int],
        dilation_: List[int],
        groups: int,
        binary_attr: str,
        binary_alpha: Optional[float],
        unary_attr: Optional[str],
        unary_scalars: Optional[List],
        unary_algorithm: Optional[str],
    ):
        kernel = "torch.ops.mkldnn._convolution_pointwise_.binary"
        (inputs, constant_args, inputs_layout,) = _prepare_convolution_fusion_create(
            cls, x, weight, bias, padding_, stride_, dilation_, groups
        )
        other = cls.realize_input(other)
        V.graph.realize_users_of(other.get_name())
        inputs.insert(1, other)
        constant_args = constant_args + [
            binary_attr,
            binary_alpha,
            unary_attr,
            unary_scalars,
            unary_algorithm,
        ]
        return ConvolutionBinaryInplace(
            kernel_layout=MutationLayout(inputs[1]),
            inputs_layout=inputs_layout,
            inputs=inputs,
            constant_args=constant_args,
            kernel=kernel,
        )

    def apply_constraint(self):
        x = self.inputs[0]
        # FixedLayout of input
        x = self.require_stride_order(x, self.inputs_layout.preferred_stride_order)
        self.inputs[0] = x
        self.freeze_layout_with_stride_order(self.inputs_layout.preferred_stride_order)


class LinearUnary(ExternKernelAlloc):
    kernel = "torch.ops.mkldnn._linear_pointwise"

    def __init__(
        self,
        layout,
        inputs,
        constant_args=(),
        kernel="torch.ops.mkldnn._linear_pointwise",
    ):
        super().__init__(layout, inputs, constant_args)
        self.kernel = kernel

    def codegen(self, wrapper):
        wrapper.writeline(
            f"{self.get_name()} = {self.kernel}({', '.join(self.codegen_args())})"
        )

    @classmethod
    def create(cls, x, w, b, attr, scalars, algorithm):
        kernel = "torch.ops.mkldnn._linear_pointwise"
        x = cls.require_stride1(cls.realize_input(x))
        w = cls.require_stride1(cls.realize_input(w))

        *m, ic = x.get_size()
        oc, ic = w.get_size()

        inputs = [x, w]
        constant_args = [attr, scalars, algorithm]
        if b is not None:
            b = cls.require_stride1(cls.realize_input(b))
            inputs.append(b)
        else:
            constant_args.insert(0, b)

        return LinearUnary(
            layout=FlexibleLayout(
                device=x.get_device(),
                dtype=x.get_dtype(),
                size=list(m) + [oc],
            ),
            inputs=inputs,
            constant_args=constant_args,
            kernel=kernel,
        )

    def apply_constraint(self):
        pass


class LinearBinary(ExternKernelAlloc):
    kernel = "torch.ops.mkldnn._linear_pointwise.binary"

    def __init__(
        self,
        layout,
        inputs,
        constant_args=(),
        kernel="torch.ops.mkldnn._linear_pointwise.binary",
    ):
        super().__init__(layout, inputs, constant_args)
        self.kernel = kernel

    def codegen(self, wrapper):
        wrapper.writeline(
            f"{self.get_name()} = {self.kernel}({', '.join(self.codegen_args())})"
        )

    @classmethod
    def create(cls, x, y, w, b, attr):
        kernel = "torch.ops.mkldnn._linear_pointwise.binary"
        x = cls.require_stride1(cls.realize_input(x))
        y = cls.require_stride1(cls.realize_input(y))
        w = cls.require_stride1(cls.realize_input(w))

        *m, ic = x.get_size()
        oc, ic = w.get_size()

        inputs = [x, y, w]
        constant_args = [attr]
        if b is not None:
            b = cls.require_stride1(cls.realize_input(b))
            inputs.append(b)
        else:
            constant_args.insert(0, b)

        return LinearBinary(
            layout=FlexibleLayout(
                device=x.get_device(),
                dtype=x.get_dtype(),
                size=list(m) + [oc],
            ),
            inputs=inputs,
            constant_args=constant_args,
            kernel=kernel,
        )

    def apply_constraint(self):
        pass


@dataclasses.dataclass
class MutableBox(IRNode):
    """
    TensorBox / StorageBox allow in-place mutation of Tensors
    """

    data: IRNode

    def __getattr__(self, name):
        fn = getattr(self.data, name)
        if callable(fn):
            return fn
        raise AttributeError(f"{type(self.data).__name__}.{name} not callable")

    def __str__(self):
        if isinstance(self.data, MutableBox):
            line0 = f"{type(self).__name__}({type(self.data).__name__}("
            endl = "))"
            inner = self.data.data
        else:
            line0 = f"{type(self).__name__}("
            inner = self.data
            endl = ")"

        lines = [
            line0,
            indent(str(inner)),
            endl,
        ]
        return "\n".join(lines)

    __repr__ = __str__


class TensorBox(MutableBox):
    @staticmethod
    def create(data):
        return TensorBox(StorageBox(data))


class StorageBox(MutableBox):
    def is_input_buffer(self):
        if isinstance(self.data, (InputBuffer, ReinterpretView)):
            return self.data.get_name() in V.graph.graph_inputs
        return False

    def realize(self):
        if isinstance(
            self.data, (ComputedBuffer, InputsKernel, InputBuffer, ReinterpretView)
        ):
            return self.data.get_name()
        assert isinstance(self.data, (Pointwise, Reduction)), type(self.data)
        self.data = ComputedBuffer(
            name=None,
            layout=FlexibleLayout(
                device=self.data.get_device(),
                dtype=self.data.get_dtype(),
                size=self.data.get_size(),
            ),
            data=self.data,
        )
        self.data.name = V.graph.register_buffer(self.data)
        self.data.origins = self.origins
        return self.data.name

    def realize_hint(self):
        """
        Called on buffers we expect to be forced to realize later.
        """
        if isinstance(self.data, (Pointwise, Reduction)) and self.num_reads() > 1:
            self.realize()

    def has_exceeded_max_reads(self):
        return isinstance(self.data, Pointwise) and (
            self.num_reads() > config.realize_acc_reads_threshold
            or len(self.inner_fn_str()) > config.realize_bytes_threshold
        )

    def mark_reuse(self, users):
        """
        A heuristic to decide if we should realize a tensor
        that is used multiple times.
        """

        def should_realize_on_cpu(loops: Union[Pointwise, Reduction]):
            """
            The heuristic for realizing reused result of heavy ops on cpu
            """
            heavy_ops = ["exp"]  # a list of heavy ops
            fn_str = loops.inner_fn_str()
            return any([fn_str.startswith(op + "(") for op in heavy_ops])

        if (
            users > 1
            and isinstance(self.data, (Pointwise, Reduction))
            and (
                self.num_reads() > config.realize_reads_threshold
                or len(self.inner_fn_str()) > config.realize_bytes_threshold
                or (is_cpu(self.data) and should_realize_on_cpu(self.data))
            )
        ):
            self.realize()

    @cache_on_self
    def num_reads(self):
        data = self.data
        if isinstance(data, (InputsKernel, InputBuffer, ReinterpretView)):
            return 1
        if isinstance(data, ComputedBuffer):
            read_writes = data.get_read_writes()
        else:
            assert isinstance(data, (Pointwise, Reduction)), type(data)
            read_writes = ComputedBuffer(
                name=None,
                layout=FlexibleLayout(
                    device=data.get_device(),
                    dtype=data.get_dtype(),
                    size=data.get_size(),
                ),
                data=data,
            ).get_read_writes()
        return len(read_writes.reads)


class LoopBody:
    """
    Captures the body of a Loops subclass into an FX graph.  Persists any
    indexing simplifications and makes it easier to analyze loop bodies.
    """

    def __init__(self, fn, args, var_ranges):
        super().__init__()
        self.var_ranges = var_ranges
        self.indexing_exprs = {}
        self.indexing_exprs_name = {}
        self.reads = []
        self.writes = []
        self.reads_name2expr = {}
        self.writes_name2expr = {}
        self.other = []
        self.submodules = {"get_index": self.get_index}
        self.subblocks = {}
        self.indirect_vars = []
        self.root_block = LoopBodyBlock(self, fn, args)
        self.indexing = None

    def debug_str(self):
        lines = [f"var_ranges = {dict(self.var_ranges)}"]
        lines.extend([f"{name} = {val}" for name, val in self.indexing_exprs.items()])
        lines.extend(
            [
                block.debug_str(name)
                for name, block in itertools.chain(
                    [("body", self.root_block)], self.subblocks.items()
                )
            ]
        )
        return "\n".join(lines)

    def add_index_expr(self, expr: sympy.Expr, category, buf_name):
        getattr(self, category).append(expr)
        if buf_name is not None:
            getattr(self, f"{category}_name2expr")[buf_name] = expr
        if expr not in self.indexing_exprs_name:
            name = f"index{len(self.indexing_exprs)}"
            self.indexing_exprs_name[expr] = name
            self.indexing_exprs[name] = expr
        return self.indexing_exprs_name[expr]

    def add_submodule(self, block, prefix):
        """Not actually for nn.Modules, but subblocks in generated code are mapped to FX call_module opcodes"""
        if prefix[-1].isnumeric() and prefix not in self.submodules:
            name = prefix
        else:
            name = f"{prefix}{len(self.submodules)}"
        self.submodules[name] = block
        return name

    def add_indirect(self):
        name = f"indirect{len(self.indirect_vars)}"
        var = sympy_symbol(name)
        self.indirect_vars.append([var])
        return var

    def replace_indirect(self, old, new):
        """Swap in a variable used in indirect indexing"""
        if str(old) == str(new):
            return
        self.indexing = {k: sympy_subs(v, {old: new}) for k, v in self.indexing.items()}

    def get_index(self, name):
        return self.indexing[name]

    def __call__(self, *indices):
        index = list(itertools.chain(*indices))
        assert len(index) == len(self.var_ranges), (index, self.var_ranges)
        assert all(v not in self.var_ranges for v in index)
        replacements = dict(zip(self.var_ranges.keys(), index))
        self.indexing = {
            name: sympy_subs(expr, replacements)
            for name, expr in self.indexing_exprs.items()
        }
        result = self.root_block()
        self.indexing = None
        return result


class LoopBodyBlock:
    """
    Captures the body of a Loops subclass into an FX graph.
    In normal cases there will be a 1:1 mapping between LoopBody and
    LoopBodyBlock, hower in the case of ops.masked() the masked out
    operations will manifest as an extra LoopBodyBlock.
    """

    def __init__(self, body: LoopBody, fn: Callable, args: List[Any]):
        self.body = body

        def add_index(expr, category, buf_name=None):
            return tracer.create_proxy(
                "call_module",
                "get_index",
                (self.body.add_index_expr(expr, category, buf_name),),
                {},
            )

        class CaptureIndexing(V.WrapperHandler):
            def load(self, name: str, index: sympy.Expr):
                index = add_index(index, "reads", name)
                return self._inner.load(name, index)

            def store(self, name, index, value, mode=None):
                index = add_index(index, "writes", name)
                return self._inner.store(name, index, value, mode)

            def reduction(self, name, dtype, src_dtype, reduction_type, index, value):
                index = add_index(index, "writes", name)
                return self._inner.reduction(
                    name, dtype, src_dtype, reduction_type, index, value
                )

            def index_expr(self, index, dtype):
                if isinstance(index, (int, sympy.Integer)):
                    return ops.constant(int(index), dtype)
                index = add_index(index, "other")
                return self._inner.index_expr(index, dtype)

            @staticmethod
            def masked(mask_proxy, masked_body: Callable, other_proxy):
                """
                Recursively capture the masked out body in another LoopBodyBlock
                """

                def shim(mask, other):
                    return V.ops.masked(mask, subblock, other)

                name = self.body.add_submodule(shim, "masked_subblock")
                subblock = LoopBodyBlock(self.body, masked_body, [])
                self.body.subblocks[name] = subblock
                return tracer.create_proxy(
                    "call_module", name, (mask_proxy, other_proxy), {}
                )

            @staticmethod
            def indirect_indexing(index_proxy):
                """
                Flow data from tensors into indexing formulas.
                Introduce a call_module to update the indexing.
                """

                def set_indirect(new_var):
                    self.body.replace_indirect(var, V.ops.indirect_indexing(new_var))

                var = self.body.add_indirect()
                tracer.create_proxy(
                    "call_module",
                    self.body.add_submodule(set_indirect, f"set_{var}"),
                    (index_proxy,),
                    {},
                )
                return var

        tracer = torch.fx.Tracer()
        tracer.graph = torch.fx.Graph(tracer_cls=tracer.__class__)
        proxy_ops = tracer.create_proxy("placeholder", "ops", (), {})
        from .sizevars import SimplifyIndexing

        with V.set_ops_handler(
            SimplifyIndexing(CaptureIndexing(proxy_ops), self.body.var_ranges)
        ):
            tracer.create_proxy("output", "output", (fn(*args),), {})
        self.graph = tracer.graph

    def __call__(self):
        graph = self.graph
        submodules = self.body.submodules

        class InterpreterShim(torch.fx.Interpreter):
            def __init__(self):
                """
                We don't call super() here to avoid constructing a
                GraphModule which is very expensive (it does codegen).
                """
                self.module = self
                self.graph = graph
                self.submodules = submodules
                self.garbage_collect_values = False
                self.env = {}
                self.fetch_attr = submodules.__getitem__

        return InterpreterShim().run(V.get_ops_handler())

    def debug_str(self, name="block"):
        code = torch.fx.GraphModule(self.body.submodules, self.graph).code
        return re.sub(
            # strip `; del var0` suffixes to make output prettier
            r";[^\n]*",
            "",
            code.strip().replace("def forward(", f"def {name}("),
        )
