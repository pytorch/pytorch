from __future__ import annotations

import collections
import contextlib
import dataclasses
import functools
import itertools
import logging
import math
import operator
import os
import textwrap
from functools import lru_cache
from typing import (
    Any,
    Callable,
    cast,
    Counter,
    DefaultDict,
    Dict,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

import sympy

import torch
import torch._logging
import torch.utils._pytree as pytree
from torch._dynamo.utils import preserve_rng_state

from torch._inductor.metrics import is_metric_table_enabled, log_kernel_metadata
from torch._prims_common import is_integer_dtype
from torch.utils._sympy.functions import FloorDiv, ModularIndexing
from torch.utils._sympy.value_ranges import ValueRanges
from torch.utils._triton import has_triton_package

from ..._dynamo.utils import counters
from .. import config, ir, scheduler
from ..codecache import code_hash, get_path, PyCodeCache
from ..dependencies import Dep, MemoryDep, StarDep, WeakDep
from ..ir import IRNode, ReductionHint, TritonTemplateBuffer
from ..optimize_indexing import indexing_dtype_strength_reduction
from ..scheduler import BaseSchedulerNode, BaseScheduling, WhyNoFuse
from ..triton_heuristics import AutotuneHint
from ..utils import (
    cache_on_self,
    do_bench,
    get_dtype_size,
    get_fused_kernel_name,
    get_kernel_metadata,
    get_max_y_grid,
    green_text,
    is_welford_reduction,
    next_power_of_2,
    Placeholder,
    sympy_dot,
    sympy_index_symbol,
    sympy_product,
    sympy_subs,
    unique,
    yellow_text,
)
from ..virtualized import _ops as ops, OpsHandler, ReductionType, StoreMode, V
from ..wrapper_benchmark import get_kernel_category_by_source_code
from .common import (
    CSE,
    CSEVariable,
    DeferredLine,
    free_symbol_startswith,
    IndentedBuffer,
    index_prevent_reordering,
    Kernel,
    OpOverrides,
    PythonPrinter,
    SizeArg,
    TensorArg,
)
from .multi_kernel import MultiKernel
from .triton_utils import config_of, signature_of, signature_to_meta

log = logging.getLogger(__name__)
perf_hint_log = torch._logging.getArtifactLogger(__name__, "perf_hints")
schedule_log = torch._logging.getArtifactLogger(__name__, "schedule")
fusion_log = torch._logging.getArtifactLogger(__name__, "fusion")


@lru_cache(None)
def gen_attr_descriptor_import():
    """
    import AttrsDescriptor if the triton version is new enough to have this
    class defined.
    """
    if not has_triton_package():
        return ""

    import triton.compiler.compiler

    if hasattr(triton.compiler.compiler, "AttrsDescriptor"):
        return "from triton.compiler.compiler import AttrsDescriptor"
    else:
        return ""


@lru_cache(None)
def gen_common_triton_imports():
    imports = IndentedBuffer()
    imports.splice(
        """
        import triton
        import triton.language as tl
        """
    )
    if attr_desc := gen_attr_descriptor_import():
        imports.writeline(attr_desc)

    imports.splice(
        """
        from torch._inductor import triton_helpers, triton_heuristics
        from torch._inductor.ir import ReductionHint, TileHint
        from torch._inductor.triton_helpers import libdevice, math as tl_math
        from torch._inductor.triton_heuristics import AutotuneHint
        from torch._inductor.utils import instance_descriptor
        """
    )
    return imports.getvalue()


@dataclasses.dataclass
class IndexingOptions:
    index_str: str
    mask_vars: Set[sympy.Symbol]
    mask_str: str
    expand_str: Optional[str]
    _has_rindex: bool

    def has_mask(self):
        return bool(self.mask_vars)

    def has_rindex(self):
        return self._has_rindex

    def has_tmpmask(self):
        return "tmp" in self.mask_str

    def has_rmask(self):
        return "rmask" in self.mask_str


@dataclasses.dataclass
class BlockPtrOptions:
    constant_offset: sympy.Expr
    shape: List[sympy.Expr]
    strides: List[sympy.Expr]
    block_shape: List[str]
    order: List[int]
    offsets: List[str]
    mask_vars: Set[sympy.Symbol]
    reshape_suffix: List[str]

    @staticmethod
    def create(
        strides: List[sympy.Expr],
        constant_offset: sympy.Expr,
        range_trees: List[IterationRangesEntry],
        mask_vars: Set[sympy.Symbol],
    ) -> BlockPtrOptions:
        """Helper to create a  BlockPtrOptions instance"""
        block_shape = [f"{t.prefix.upper()}BLOCK" for t in range_trees]
        reshape_suffix = [*block_shape]

        broadcasting_dim = [s == 0 for s in strides]
        for i, is_broadcasting in enumerate(broadcasting_dim):
            if is_broadcasting:
                # drop any stride==0 dimensions for performance
                reshape_suffix[i] = "1"

        if V.kernel.no_x_dim:
            assert range_trees[0].prefix == "x"
            reshape_suffix.pop(0)

        if (
            not V.kernel.inside_reduction
            and len(strides) == len(V.kernel.numels) - 1
            and V.kernel.numels[-1] != 1
        ):
            # Need to expand rank by 1 to match rank when self.inside_reduction=True
            reshape_suffix.append("1")

        def filter(it):
            """Removes any broadcasting dims from a given sequence"""
            assert len(it) == len(broadcasting_dim)
            return [
                item
                for item, is_broadcasting in zip(it, broadcasting_dim)
                if not is_broadcasting
            ]

        return BlockPtrOptions(
            constant_offset=V.graph.sizevars.lookup_precomputed_size(constant_offset),
            shape=[
                V.graph.sizevars.lookup_precomputed_size(t.numel)
                for t in filter(range_trees)
            ],
            strides=[*map(V.graph.sizevars.lookup_precomputed_size, filter(strides))],
            block_shape=filter(block_shape),
            order=V.graph.sizevars.guarded_order(filter(strides)),
            offsets=filter([f"{t.prefix}offset" for t in range_trees]),
            mask_vars=mask_vars,
            reshape_suffix=reshape_suffix,
        )

    def format(self, name: str, roffset=True) -> str:
        """
        Codegen a call to tl.make_block_ptr()

        Args:
            name: variable name for pointer
            roffset: should roffset be included in offsets=..., for use with tl.advance()

        Returns:
            "tl.make_block_ptr(...)"
        """
        f = V.kernel.index_to_str
        offsets = [*self.offsets]
        if not roffset:
            offsets[offsets.index("roffset")] = "0"
        args = [
            f"{name} + ({f(self.constant_offset)})"
            if self.constant_offset != 0
            else name,
            f"shape={f(self.shape)}",
            f"strides={f(self.strides)}",
            f"block_shape={f(self.block_shape)}",
            f"order={f(self.order)}",
            f"offsets={f(offsets)}",
        ]
        return f"tl.make_block_ptr({', '.join(args)})"

    @cache_on_self
    def boundary_check(self) -> List[int]:
        """List of indices to pass to tl.load(boundary_check=...)"""
        check = []
        for i in range(len(self.shape)):
            if (
                self.block_shape[i] != "1"
                and not V.graph.sizevars.statically_known_equals(self.strides[i], 0)  # type: ignore[arg-type]
                and not V.graph.sizevars.statically_known_multiple_of(
                    self.shape[i],
                    config.triton.max_block[self.block_shape[i][0]],  # type: ignore[arg-type]
                )
                and not (V.kernel.no_x_dim and self.block_shape[i] == "XBLOCK")
            ):
                check.append(i)
        return check

    def advance_roffset(self):
        """Codegen string to pass to tl.advance(name, ...)"""
        advance = ["0"] * len(self.shape)
        advance[self.offsets.index("roffset")] = "RBLOCK"
        return V.kernel.index_to_str(advance)

    def has_rindex(self):
        return "RBLOCK" in self.block_shape

    def has_rmask(self):
        return self.has_rindex()

    def has_tmpmask(self):
        return False  # block_ptr can't do indirect indexing

    def has_mask(self):
        return bool(self.boundary_check())


def triton_reshape(value: str, old_shape: List[str], new_shape: List[str]):
    """Workaround https://github.com/openai/triton/issues/2836"""
    assert isinstance(old_shape, list) and isinstance(new_shape, list)
    if old_shape == new_shape:
        return value
    if [s for s in new_shape if s != "1"] != old_shape:
        return f"tl.reshape({value}, [{', '.join(new_shape)}])"
    # rewrite to [:, None] syntax, which is less buggy
    idx = 0
    expand = []
    for size in new_shape:
        if idx < len(old_shape) and size == old_shape[idx]:
            expand.append(":")
            idx += 1
        else:
            assert size == "1"
            expand.append("None")
    assert idx == len(old_shape)
    return f"{value}[{', '.join(expand)}]"


class TritonPrinter(PythonPrinter):
    def _print_floor(self, expr):
        assert len(expr.args) == 1
        return (
            f"libdevice.floor({self._print(expr.args[0])}).to({V.kernel.index_dtype})"
        )

    def _print_ceiling(self, expr):
        assert len(expr.args) == 1
        return f"libdevice.ceil({self._print(expr.args[0])}).to({V.kernel.index_dtype})"

    def _helper_sqrt(self, expr):
        return f"libdevice.sqrt({self._print(expr)}.to(tl.float32))"

    def _print_Where(self, expr):
        c = self.doprint(expr.args[0])
        p = self.doprint(expr.args[1])
        q = self.doprint(expr.args[2])
        return f"tl.where({c}, {p}, {q})"

    def _print_Min(self, expr):
        nargs = len(expr.args)
        if len(expr.args) == 1:
            return self._print(expr.args[0])

        mid = len(expr.args) // 2
        a = self._print(sympy.Min(*expr.args[:mid]))
        b = self._print(sympy.Min(*expr.args[mid:]))
        return f"tl.minimum({a}, {b})"

    def _print_Max(self, expr):
        nargs = len(expr.args)
        if len(expr.args) == 1:
            return self._print(expr.args[0])

        mid = len(expr.args) // 2
        a = self._print(sympy.Max(*expr.args[:mid]))
        b = self._print(sympy.Max(*expr.args[mid:]))

        return f"tl.maximum({a}, {b})"

    def _print_Abs(self, expr):
        assert len(expr.args) == 1
        return f"tl_math.abs({self._print(expr.args[0])})"

    def _print_cos(self, expr):
        assert len(expr.args) == 1
        return f"libdevice.cos(({self._print(expr.args[0])}).to(tl.float32))"

    def _print_cosh(self, expr):
        assert len(expr.args) == 1
        return f"libdevice.cosh(({self._print(expr.args[0])}).to(tl.float32))"

    def _print_acos(self, expr):
        assert len(expr.args) == 1
        return f"libdevice.acos(({self._print(expr.args[0])}).to(tl.float32))"

    def _print_sin(self, expr):
        assert len(expr.args) == 1
        return f"libdevice.sin(({self._print(expr.args[0])}).to(tl.float32))"

    def _print_sinh(self, expr):
        assert len(expr.args) == 1
        return f"libdevice.sinh(({self._print(expr.args[0])}).to(tl.float32))"

    def _print_asin(self, expr):
        assert len(expr.args) == 1
        return f"libdevice.asin(({self._print(expr.args[0])}).to(tl.float32))"

    def _print_tan(self, expr):
        assert len(expr.args) == 1
        return f"libdevice.tan(({self._print(expr.args[0])}).to(tl.float32))"

    def _print_tanh(self, expr):
        assert len(expr.args) == 1
        return f"libdevice.tanh(({self._print(expr.args[0])}).to(tl.float32))"

    def _print_atan(self, expr):
        assert len(expr.args) == 1
        return f"libdevice.atan(({self._print(expr.args[0])}).to(tl.float32))"

    def _print_FloorDiv(self, expr):
        if expr.is_integer:
            return super()._print_FloorDiv(expr)

        x, div = expr.args
        x = self.paren(self.doprint(x))
        div = self.paren(self.doprint(div))
        return f"libdevice.floor({x} / {div}).to({V.kernel.index_dtype})"

    def _print_Round(self, expr):
        assert len(expr.args) == 1
        return (
            f"libdevice.llrint({self._print(expr.args[0])}).to({V.kernel.index_dtype})"
        )

    def _print_RoundDecimal(self, expr):
        assert len(expr.args) == 2
        number, ndigits = expr.args
        if number.is_integer:
            # ndigits < 0 should have been filtered by the sympy function
            assert ndigits < 0
            raise ValueError(
                f"For integer inputs, only non-negative ndigits are currently supported, but got {ndigits}."
            )
        return f"libdevice.nearbyint(1e{ndigits} * {self.paren(self._print(number))}) * 1e{-ndigits}"


texpr = TritonPrinter().doprint
pexpr = PythonPrinter().doprint


def triton_compute_type(dtype):
    triton_type_name = str(dtype).split(".")[-1]
    if triton_type_name == "bool":
        triton_type_name = "int1"
    elif triton_type_name in ("float16", "bfloat16"):
        # float16 math is done in float32 inside the kernel
        triton_type_name = "float32"
    elif triton_type_name == "float8_e4m3fn":
        triton_type_name = "float8e4nv"
    elif triton_type_name == "float8_e5m2":
        triton_type_name = "float8e5"
    elif triton_type_name == "float8_e4m3fnuz":
        triton_type_name = "float8e4b8"
    elif triton_type_name == "float8_e5m2":
        triton_type_name = "float8e5b16"
    return f"tl.{triton_type_name}"


def triton_store_type(dtype):
    triton_type_name = str(dtype).split(".")[-1]
    if triton_type_name == "bool":
        triton_type_name = "int8"
    elif triton_type_name == "float8_e4m3fn":
        triton_type_name = "float8e4nv"
    elif triton_type_name == "float8_e5m2":
        triton_type_name = "float8e5"
    return f"tl.{triton_type_name}"


def triton_acc_type(dtype):
    if is_integer_dtype(dtype) and dtype.is_signed:
        nbits = 64 if dtype == torch.int64 else 32
        return f"tl.int{nbits}"
    return triton_compute_type(dtype)


def triton_constant(value):
    if value == float("inf"):
        return 'float("inf")'
    elif value == float("-inf"):
        return 'float("-inf")'
    elif math.isnan(value):
        return 'float("nan")'
    return repr(value)


class TritonCSEVariable(CSEVariable):
    def __init__(self, name, bounds: ValueRanges[Any]):
        super().__init__(name, bounds)
        # We'll use this to track which masks the variable needs when used for indirect indexing
        self.mask_vars: Set[str] = set()

    def update_on_args(self, name, args, kwargs):
        # When making a variable that is going to be used in indirect indexing
        # if a where clause is used it should mean that the result is always a
        # valid index, so you shouldn't include any of the dependent variables
        # in the resulting load mask
        if name == "where":
            return
        for arg in args:
            if isinstance(arg, TritonCSEVariable):
                self.mask_vars.update(arg.mask_vars)
            elif isinstance(arg, sympy.Symbol) and arg.name[0] in "xyr":
                # most of the time index vars don't need masks associated with them
                # however, when index vars are used to compute indices for indirect reads
                # those reads should subsequently be masked,
                self.mask_vars.update({f"{arg.name[0]}mask"})

    def __repr__(self):
        return f"TritonCSEVariable(name={self.name})"


class TritonOverrides(OpOverrides):
    """Map element-wise ops to Triton"""

    @staticmethod
    def to_dtype(x, dtype: torch.dtype, src_dtype: Optional[torch.dtype] = None):
        def _get_min_elements_per_thread(
            src_dtype: torch.dtype, dst_dtype: torch.dtype
        ) -> int:
            if src_dtype == dst_dtype:
                # No data type conversion is needed. No requirements on min_elem_per_thread.
                return 0

            # fp8 data type conversions has min_elem_per_thread requirements.
            # Refer to Triton implementations here:
            # https://github.com/openai/triton/blob/10f59d8ce04052521c1bc0cb3a3f8b98918fc7e3/lib/Conversion/TritonGPUToLLVM/ElementwiseOpToLLVM.cpp#L10.
            fp8_dtypes = {
                torch.float8_e4m3fn,
                torch.float8_e5m2,
            }
            # Triton doesn't support type conversions between fp8_e4m3 and fp8_e5m2.
            assert not (
                src_dtype in fp8_dtypes
                and dst_dtype in fp8_dtypes
                and src_dtype != dst_dtype
            ), "Conversions between float8_e5m2 and float8_e4m3fn is not supported!"
            if src_dtype == torch.float8_e5m2 or dst_dtype == torch.float8_e5m2:
                return 4
            if src_dtype == torch.float8_e4m3fn or dst_dtype == torch.float8_e4m3fn:
                return 2
            # No requirements on min_elem_per_thread.
            return 0

        if src_dtype is not None:
            # Both dtype and src_dtype are set. This is used by torch to(dtype=dtype).
            # It takes the maximum min_elem_per_thread if there are multiple fp8 conversions
            # in the same kernel.
            V.kernel.min_elem_per_thread = max(
                _get_min_elements_per_thread(src_dtype, dtype),
                V.kernel.min_elem_per_thread,
            )

        if dtype == torch.bool:
            return f"({x} != 0)"
        elif dtype == torch.uint8:
            # to work around llvm uint conversion semantics
            # that produces 0's for negative values
            return f"{x}.to(tl.int8).to(tl.uint8)"
        return f"{x}.to({triton_compute_type(dtype)})"

    @staticmethod
    def to_dtype_bitcast(x, dtype: torch.dtype, src_dtype: torch.dtype):
        triton_dtype = triton_compute_type(dtype)
        # We may promote float16 or bfloat16 to float32 and cause the
        # bitwidth of dtype to be different from the input tensor (i.e. float32).
        # In such as case, we will have to convert the input tensor to
        # its src_type, perform bitcast, and then convert the bit-casted
        # tensor back to float to ensure we use values with the right precision.
        if src_dtype in (torch.float16, torch.bfloat16):
            triton_src_dtype = str(src_dtype).split(".")[-1]
            cast_x = f"{x}.to(tl.{triton_src_dtype})"
            cast_x = f"{cast_x}.to({triton_dtype}, bitcast=True)"
            return f"{cast_x}.to(tl.float32)"
        else:
            return f"{x}.to({triton_dtype}, bitcast=True)"

    @staticmethod
    def _shaped_constant(value, dtype, shape):
        type_ = torch._prims_common.dtype_to_type(dtype)
        triton_val = triton_constant(type_(value))
        triton_type = triton_compute_type(dtype)

        if triton_type == "tl.float32":
            # Float constants are always f32 in triton
            return triton_val

        # NOTE: We use a tensor here in order to get the expected type.
        # Otherwise, e.g. float64 constants would be trunctated to float32.
        return f"tl.full({shape}, {triton_val}, {triton_type})"

    @classmethod
    def constant(cls, value, dtype):
        return cls._shaped_constant(value, dtype, shape=[])

    @staticmethod
    def abs(x):
        return f"tl_math.abs({x})"

    @staticmethod
    def libdevice_abs(x):
        return f"libdevice.abs({x})"

    @staticmethod
    def exp(x):
        return f"tl_math.exp({x})"

    @staticmethod
    def libdevice_exp(x):
        return f"libdevice.exp({x})"

    @staticmethod
    def exp2(x):
        return f"libdevice.exp2({x})"

    @staticmethod
    def expm1(x):
        return f"libdevice.expm1({x})"

    @staticmethod
    def sqrt(x):
        return f"libdevice.sqrt({x})"

    @staticmethod
    def libdevice_sqrt(x):
        return f"libdevice.sqrt({x})"

    @staticmethod
    def relu(x):
        bug = config.triton.inject_relu_bug_TESTING_ONLY
        if bug == "compile_error":
            return "compile error!"
        elif bug == "runtime_error":
            # NB: this only triggers runtime error as long as input
            # is not all zero
            return f'triton_helpers.device_assert_then({x} == 0, "injected assert fail", {x})'
        elif bug == "accuracy":
            return f"{x} + 1"
        elif bug is None:
            return ops.maximum("0", x)
        else:
            raise AssertionError(
                f"unrecognized config triton.inject_relu_bug_TESTING_ONLY = {bug!r}"
            )

    @staticmethod
    def minimum(a, b):
        return f"triton_helpers.minimum({a}, {b})"

    @staticmethod
    def maximum(a, b):
        return f"triton_helpers.maximum({a}, {b})"

    @staticmethod
    def where(a, b, c):
        return f"tl.where({a}, {b}, {c})"

    @staticmethod
    def cos(x):
        return f"tl_math.cos({x})"

    @staticmethod
    def libdevice_cos(x):
        return f"libdevice.cos({x})"

    @staticmethod
    def sin(x):
        return f"tl_math.sin({x})"

    @staticmethod
    def libdevice_sin(x):
        return f"libdevice.sin({x})"

    @classmethod
    def index_expr(cls, expr, dtype):
        raise NotImplementedError("ops.index_expr not implemented outside a kernel")

    @staticmethod
    def masked(mask, body, other):
        raise NotImplementedError("ops.masked not implemented outside a kernel")

    @staticmethod
    def lgamma(x):
        return f"libdevice.lgamma({x})"

    @staticmethod
    def erf(x):
        return f"libdevice.erf({x})"

    @staticmethod
    def cosh(x):
        return f"libdevice.cosh({x})"

    @staticmethod
    def sinh(x):
        return f"libdevice.sinh({x})"

    @staticmethod
    def acos(x):
        return f"libdevice.acos({x})"

    @staticmethod
    def acosh(x):
        return f"libdevice.acosh({x})"

    @staticmethod
    def asin(x):
        return f"libdevice.asin({x})"

    @staticmethod
    def asinh(x):
        return f"libdevice.asinh({x})"

    @staticmethod
    def atan2(x, y):
        return f"libdevice.atan2({x}, {y})"

    @staticmethod
    def atan(x):
        return f"libdevice.atan({x})"

    @staticmethod
    def atanh(x):
        return f"libdevice.atanh({x})"

    @staticmethod
    def copysign(x, y):
        return f"libdevice.copysign({x}, {y})"

    @staticmethod
    def erfc(x):
        return f"libdevice.erfc({x})"

    @staticmethod
    def erfinv(x):
        return f"libdevice.erfinv({x})"

    @staticmethod
    def hypot(x, y):
        return f"libdevice.hypot({x}, {y})"

    @staticmethod
    def log10(x):
        return f"libdevice.log10({x})"

    @staticmethod
    def nextafter(x, y):
        return f"libdevice.nextafter({x}, {y})"

    @staticmethod
    def logical_and(a, b):
        return f"{a} & {b}"

    @staticmethod
    def logical_not(a):
        return f"{a} == 0"

    @staticmethod
    def logical_or(a, b):
        return f"{a} | {b}"

    @staticmethod
    def logical_xor(a, b):
        return f"({a} ^ {b})"

    @staticmethod
    def bitwise_and(a, b):
        return f"{a} & {b}"

    @staticmethod
    def bitwise_not(a):
        return f"~{a}"

    @staticmethod
    def bitwise_or(a, b):
        return f"{a} | {b}"

    @staticmethod
    def bitwise_xor(a, b):
        return f"{a} ^ {b}"

    @staticmethod
    def bitwise_left_shift(a, b):
        return f"{a} << {b}"

    @staticmethod
    def bitwise_right_shift(a, b):
        return f"{a} >> {b}"

    @staticmethod
    def rand(seed, offset):
        offset = f"({offset}).to(tl.uint32)"
        return f"tl.rand({seed}, {offset})"

    @staticmethod
    def randn(seed, offset):
        offset = f"({offset}).to(tl.uint32)"
        return f"tl.randn({seed}, {offset})"

    @staticmethod
    def randint64(seed, offset, low, high):
        offset = f"({offset}).to(tl.uint32)"
        return f"triton_helpers.randint64({seed}, {offset}, {low}, {high})"

    @staticmethod
    def load_seed(name, offset):
        raise NotImplementedError("ops.load_seed not implemented outside a kernel")

    @staticmethod
    def rsqrt(x):
        return f"libdevice.rsqrt({x})"

    @staticmethod
    def log1p(x):
        return f"libdevice.log1p({x})"

    @staticmethod
    def tan(x):
        return f"libdevice.tan({x})"

    @staticmethod
    def tanh(x):
        return f"libdevice.tanh({x})"

    @staticmethod
    def sigmoid(x):
        return f"tl.sigmoid({x})"

    @staticmethod
    def libdevice_sigmoid(x):
        return f"1/(1 + libdevice.exp(-({x})))"

    @staticmethod
    def signbit(x):
        # XX: This is wrong for the value -0.0 in floating point
        return f"libdevice.signbit({x}) if ({x}).dtype is tl.float32 else {x} < 0"

    @staticmethod
    def fmod(a, b):
        return f"libdevice.fmod({a}, {b})"

    @staticmethod
    def pow(a, b):
        return f"libdevice.pow({a}, {b})"

    @staticmethod
    def log(x):
        return f"tl_math.log({x})"

    @staticmethod
    def libdevice_log(x):
        return f"libdevice.log({x})"

    @staticmethod
    def isinf(x):
        return f"libdevice.isinf({x}).to(tl.int1)"

    @staticmethod
    def isnan(x):
        return f"libdevice.isnan({x}).to(tl.int1)"

    @staticmethod
    def round(x):
        return f"libdevice.nearbyint({x})"

    @staticmethod
    def floor(x):
        return f"libdevice.floor({x})"

    @staticmethod
    def floordiv(a, b):
        # See the comment in lowering.div_mode. a and b are integer type.
        # Similar to div_floor_kernel_cuda in pytorch core.
        # Notice that // in triton behaves as truncdiv instead of floordiv
        quot = f"{a} // {b}"
        rem = f"{a} % {b}"
        return f"tl.where(({a} < 0) != ({b} < 0), tl.where({rem} != 0, {quot} - 1, {quot}), {quot})"

    @staticmethod
    def sign(x):
        def to_int(s):
            return f"{s}.to(tl.int8)"

        left = to_int(ops.lt("0", x))
        right = to_int(ops.lt(x, "0"))
        sub = ops.sub(left, right)
        return f"{sub}.to({x}.dtype)"

    @staticmethod
    def trunc(x):
        return f"libdevice.trunc({x})"

    @staticmethod
    def truncdiv(a, b):
        # See the comment in lowering.div_mode. a and b are integer type.
        # Notice that // in triton behaves as truncdiv instead of floordiv
        return f"{a} // {b}"

    @staticmethod
    def ceil(x):
        return f"libdevice.ceil({x})"


TritonOverrides._initialize_pointwise_overrides("triton")


# Use mypy to check protocol implemented correctly
def _typecheck_TritonOverrides(h: TritonOverrides) -> OpsHandler[str]:
    return h


class TritonKernelOverrides(TritonOverrides):
    """Map element-wise ops to Triton within a TritonKernel

    Unlike TritonOverrides, these assume the code is going to be inserted into
    the body of the main triton kernel and so it may use indexing and mask
    variables which are assumed to already be defined in the current scope.
    """

    @classmethod
    def constant(cls, value, dtype):
        # NOTE: Cannot use shape=[] as it's not supported by triton-rocm
        # We could use shape=[1] instead but starting with the correct
        # ndim avoids extra `tt.expand_dim` ops appearing in the triton IR.
        ndim = V.kernel.triton_tensor_ndim()
        shape = [1] * ndim
        return cls._shaped_constant(value, dtype, shape=shape)

    @classmethod
    def index_expr(cls, expr, dtype):
        indexing = V.kernel.indexing(expr, block_ptr=False)
        assert isinstance(indexing, IndexingOptions)
        # This is called from CSEProxy.__getattr__,  so we'll set the bounds there
        var = V.kernel.cse.generate(V.kernel.compute, indexing.index_str)

        if dtype not in {torch.int32, torch.int64}:
            var = V.kernel.cse.generate(V.kernel.compute, cls.to_dtype(var, dtype))
        var.mask_vars = indexing.mask_vars
        return var

    @staticmethod
    def masked(mask, body, other):
        with V.kernel.mask_loads(mask) as new_mask:
            result = body()

        # Take dtype from result to prevent accidental promotion
        other = V.kernel.cse.generate(
            V.kernel.compute,
            f"tl.full({result}.shape, {triton_constant(other)}, {result}.dtype)",
        )
        return ops.where(new_mask, result, other)

    @staticmethod
    def load_seed(name, offset):
        var = V.kernel.args.input(name)
        return (
            f"tl.load({var} + {V.kernel.args.seed_offset('load_seed_offset', offset)})"
        )

    @staticmethod
    def frexp(x):
        cache_key = f"frexp({x})"
        if cache_key in V.kernel.cse.cache:
            return V.kernel.cse.cache[cache_key]

        mantissa = V.kernel.cse.newvar()
        exponent = V.kernel.cse.newvar()
        V.kernel.compute.writeline(
            f"{mantissa}, {exponent} = triton_helpers.frexp({x})"
        )
        V.kernel.cse.cache[cache_key] = (mantissa, exponent)
        return (mantissa, exponent)


# Use mypy to check protocol implemented correctly
def _typecheck_TritonKernelOverrides(h: TritonKernelOverrides) -> OpsHandler[str]:
    return h


@dataclasses.dataclass
class IterationRanges:
    """
    Each range tree represents multiple sets of iteration indexing
    in a single tiled dimension in the output kernel.

    If you have two loops ranges one (4, 3, 2) and another (4, 6),
    then the range tree will be:
            4 (i0)
        3 (i1)  6 (i3)
        2 (i2)
    Where i0 is shared between both loops, but then the split into
    different indexing vars.  All loop ranges must iterate over
    the same number of elements.
    """

    def __init__(
        self,
        name: str,
        var_list: List[sympy.Symbol],
        var_ranges: Dict[sympy.Symbol, sympy.Expr],
        numel: sympy.Expr,
        prefix: str,
        *,
        kernel: TritonKernel,
        divisor=sympy.Integer(1),
        length=sympy.Integer(1),
        root: IterationRangesRoot,
    ):
        super().__init__()
        self.name = name
        self.var_list = var_list
        self.var_ranges = var_ranges
        self.numel = numel
        self.prefix = prefix
        self.divisor = divisor
        self.length = length
        self.kernel = kernel
        self.root = root

    def symbol(self):
        return sympy_index_symbol(self.name)


class IterationRangesRoot(IterationRanges):
    def __init__(
        self,
        name: str,
        numel: sympy.Expr,
        prefix: str,
        index: int,
        kernel: TritonKernel,
        pid_cache=None,
        *,
        is_loop: bool,
        tensor_dim: Optional[int],
        grid_dim: Optional[int],
    ):
        if pid_cache is None:
            pid_cache = {}
        super().__init__(
            name=name,
            var_list=[],
            var_ranges={},
            numel=numel,
            prefix=prefix,
            kernel=kernel,
            root=self,
        )
        self.index = index
        # Store all the nodes in one flat list
        self.nodes: Dict[sympy.Expr, IterationRangesEntry] = {}
        # This is for re-ordering program ID in triton mm template
        # pid_cache["tl.program_id(0)"] = pid_m
        self.pid_cache: Dict[str, str] = pid_cache

        # True if the dimension is implemented as a single program looping over
        # the full dimension (currently only used for non-persistent reduction)
        assert not is_loop or (prefix == "r" and grid_dim is None)
        self.is_loop = is_loop
        # Index of corresponding dimension on triton tensors
        self.tensor_dim = tensor_dim
        # Index of corresponding dimension in the triton grid
        self.grid_dim = grid_dim

    def __repr__(self):
        return f"IterationRangesRoot({self.name!r}, {self.numel}, ...)"

    def cache_clear(self):
        for node in self.nodes.values():
            node.cache_clear()

    def lookup(self, divisor, length):
        """
        Lookup a given RangeTreeEntry, creating it if needed
        """
        if V.graph.sizevars.statically_known_equals(divisor * length, self.numel):
            expr = FloorDiv(sympy_index_symbol(f"{self.prefix}index"), divisor)
        else:
            expr = ModularIndexing(
                sympy_index_symbol(f"{self.prefix}index"), divisor, length
            )

        if expr not in self.nodes:
            node = IterationRangesEntry(
                f"{self.prefix}{next(V.kernel.iter_vars_count)}",
                divisor,
                length,
                expr,
                self,
            )
            V.kernel.range_tree_nodes[node.symbol()] = node
            self.var_list.append(node.symbol())
            self.var_ranges[node.symbol()] = length
            self.nodes[expr] = node
        return self.nodes[expr]

    def construct_entries(self, lengths: List[sympy.Expr]):
        divisor = sympy.Integer(1)
        itervars = []
        for length in reversed(lengths):
            itervars.append(self.lookup(divisor, length))
            divisor = divisor * length
        return list(reversed(itervars))

    def construct(self, lengths: List[sympy.Expr]):
        return [e.symbol() for e in self.construct_entries(lengths)]

    def vars_and_sizes(self, index: sympy.Expr):
        """Figure out vars from this tree used in index"""
        nodes = [V.kernel.range_tree_nodes.get(s) for s in index.free_symbols]
        nodes = [n for n in nodes if n and n.prefix == self.prefix]
        nodes.sort(key=lambda x: V.graph.sizevars.size_hint(x.divisor))
        divisor = sympy.Integer(1)
        index_vars = []
        sizes = []

        def add(node):
            nonlocal divisor
            index_vars.append(node.symbol())
            sizes.append(node.length)
            divisor = divisor * node.length

        for node in nodes:
            if not V.graph.sizevars.statically_known_equals(node.divisor, divisor):
                # fill in unused index var
                add(self.lookup(divisor, FloorDiv(node.divisor, divisor)))
                divisor = node.divisor
            add(node)
        if not V.graph.sizevars.statically_known_equals(self.numel, divisor):
            # fill in unused index var
            add(self.lookup(divisor, FloorDiv(self.numel, divisor)))

        return list(reversed(index_vars)), list(reversed(sizes))

    def ranges_code(self):
        assert self.tensor_dim is not None
        size = self.kernel.indexing_size_str(self.tensor_dim)
        index_dtype = self.kernel.index_dtype
        convert = f".to({index_dtype})" if index_dtype != "tl.int32" else ""
        return f"tl.arange(0, {self.prefix.upper()}BLOCK){size}{convert}"

    def scalar_code(self, value):
        index_dtype = self.kernel.index_dtype
        ndim = self.kernel.triton_tensor_ndim()
        size = [1] * ndim
        return f"tl.full({size}, {value}, {index_dtype})"

    def get_pid(self):
        assert self.grid_dim is not None
        key = f"tl.program_id({self.grid_dim})"
        # y_grid has a limit, so express it in terms of y and z in case of overflow.
        # z grid is only exercised when max_tiles == 3 (off by default).
        if (
            self.grid_dim == 1
            and config.triton.max_tiles <= 2
            and not (isinstance(self.numel, int) and self.numel <= get_max_y_grid())
        ):
            key = f"{key} * (tl.program_id({self.grid_dim + 1}) + 1)"
        pid = self.pid_cache.get(key, key)
        if self.kernel.index_dtype != "tl.int32":
            return f"{pid}.to({self.kernel.index_dtype})"
        return pid

    def codegen_header(self, code):
        x = self.prefix
        if self.is_loop:
            code.writeline(f"{self.name} = {x}offset + {x}base")
        elif self.grid_dim is None:
            # no need to "{x}offset = "
            code.writeline(f"{self.name} = {self.ranges_code()}")
            code.writeline(f"{x}offset = 0")
        else:
            if self.tensor_dim is not None:
                line = f"{x}offset + {self.ranges_code()}"
            else:
                line = self.scalar_code(f"{x}offset")
            code.writelines(
                [
                    f"{x}offset = {self.get_pid()} * {x.upper()}BLOCK",
                    f"{self.name} = {line}",
                ]
            )
        code.writeline(f"{x}mask = {self.name} < {x}numel")


class IterationRangesEntry(IterationRanges):
    def __init__(
        self,
        name: str,
        divisor: sympy.Expr,
        length: sympy.Expr,
        expr: sympy.Expr,
        parent: IterationRanges,
    ):
        super().__init__(
            name=name,
            numel=parent.numel / length,
            var_list=parent.var_list,
            var_ranges=parent.var_ranges,
            prefix=parent.prefix,
            divisor=divisor,
            length=length,
            kernel=parent.kernel,
            root=parent.root,
        )
        self.parent = parent
        self.codegen = functools.lru_cache(None)(self._codegen)
        self.expr = expr

    def __repr__(self):
        return f"IterationRangesEntry({self.name}, {self.divisor}, {self.length}, {self.expr}, {self.var_ranges})"

    def set_name(self, name):
        self.codegen = lambda: name  # type: ignore[assignment]
        self.codegen.cache_clear = lambda: None  # type: ignore[method-assign]
        self.name = name

    def cache_clear(self):
        self.codegen.cache_clear()

    def writeline(self, line):
        if self.root.is_loop:
            V.kernel.indexing_code.writeline(line)
        else:
            # lift non-reduction stores outside loop
            V.kernel.body.writeline(line)

    def _codegen(self):
        self.writeline(f"{self.name} = " + texpr(V.kernel.rename_indexing(self.expr)))
        return self.name

    def precomputed_args(self):
        # for dynamic shapes, find parts of indexing expressions that have to be precomputed
        precomputed_args: List[sympy.Expr] = []
        if isinstance(self.expr, sympy.Symbol):
            return precomputed_args
        assert isinstance(self.expr, (FloorDiv, ModularIndexing)), type(self.expr)
        for arg in self.expr.args[1:]:
            if not isinstance(arg, (sympy.Integer, sympy.Symbol)):
                symbols = arg.free_symbols
                if len(symbols) > 0 and all(s.name.startswith("s") for s in symbols):
                    precomputed_args.append(arg)
        return precomputed_args

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other.name


class HelperFunctions:
    """An ordered set of helper functions."""

    _templates_seen: Dict[str, str]  # Template code to function name
    finalized_helpers: List[str]

    def __init__(self):
        self._templates_seen = {}
        self.finalized_helpers = []

    def add(self, template_code: str, *, base_name="_triton_helper_fn") -> str:
        """This accepts a function definition with the function name
        left as a format specifier e.g.

            @triton.jit
            def {name}(arg0, arg1):
                return arg0 + arg1

        We add the templated code to the function set and return the name
        assigned to that function.

        """
        existing_name = self._templates_seen.get(template_code)
        if existing_name is not None:
            # Don't duplicate existing helpers
            return existing_name

        name = f"{base_name}{len(self.finalized_helpers)}"
        self._templates_seen[template_code] = name
        self.finalized_helpers.append(template_code.format(name=name))
        return name

    def __iter__(self):
        return iter(self.finalized_helpers)

    def __getitem__(self, idx):
        return self.finalized_helpers[idx]


class TritonKernel(Kernel):
    overrides = TritonKernelOverrides  # type: ignore[assignment]
    sexpr = pexpr

    helper_functions: HelperFunctions

    def __init__(
        self,
        *groups,
        index_dtype: str,
        mutations: Optional[Set[str]] = None,
        pid_cache=None,
        reduction_hint=ReductionHint.DEFAULT,
        min_elem_per_thread=0,
        disable_persistent_reduction=False,
    ):
        if pid_cache is None:
            pid_cache = {}
        super().__init__()
        self.numels = [V.graph.sizevars.simplify(s) for s in groups]
        self.mutations: Set[str] = mutations if mutations is not None else set()
        self.range_trees: List[IterationRangesRoot] = []
        self.range_tree_nodes: Dict[sympy.Symbol, IterationRangesEntry] = {}
        self.iter_vars_count = itertools.count()
        self.inside_reduction = self.numels[-1] != 1
        self.body = IndentedBuffer()
        self.indexing_code = IndentedBuffer()
        self.suffix: IndentedBuffer = IndentedBuffer()  # type: ignore[assignment]
        self.outside_loop_vars: Set[Any] = set()
        self.reduction_hint = reduction_hint
        self.index_dtype: str = index_dtype
        self.min_elem_per_thread = min_elem_per_thread
        self.last_usage: Set[str] = set()
        self.block_ptr_id = itertools.count()
        # buffer accesses in the kernel
        self.buf_accesses: DefaultDict[str, List[Dep]] = collections.defaultdict(list)

        self.persistent_reduction: bool = (
            not disable_persistent_reduction
        ) and self.should_use_persistent_reduction()
        self.no_x_dim = (
            self.reduction_hint == ReductionHint.INNER
            and self.persistent_reduction
            and len(self.numels) == 2
            and self.numels[-1] >= 256
        )
        self.initialize_range_tree(pid_cache)

        self.helper_functions = HelperFunctions()

        # A set of autotuning hints to pass as part of triton_meta
        self.autotune_hints: Set[AutotuneHint] = set()

        # define this in a closure to make cache local to object
        @functools.lru_cache(None)
        def simplify_indexing(index: sympy.Expr):
            index = V.graph.sizevars.simplify_with_ranges(index, self.var_ranges())
            for tree in self.range_trees:
                index = self.combine_contiguous_dims(index, tree)
            return index

        self.simplify_indexing = simplify_indexing
        self.code_hash = None
        self.triton_meta: Optional[Dict[str, object]] = None

    def need_numel_args(self):
        r"""
        Indicate whether we need provide numel as arguments for the generated
        kernel calls in the benchmark.

        Should be true for pointwise/reduction kernels but false for triton
        matmul kernels.
        """
        return True

    def should_use_persistent_reduction(self) -> bool:
        """
        Heuristic to set self.persistent_reduction and add guards
        if needed.
        """
        if not (self.inside_reduction and config.triton.persistent_reductions):
            return False
        threshold = {
            ReductionHint.INNER: 1024,
        }.get(self.reduction_hint, 64)

        # If multi_kernel is enabled, we do more aggressive persistent reduction.
        # This may result in some persisent reductions slower than the
        # corresponding non-persistent reductions. MultiKernel will do benchmarking
        # to pick the faster one.
        if config.triton.multi_kernel:
            threshold *= 16
        last_numel = self.numels[-1]
        if not isinstance(last_numel, (int, sympy.Integer)):
            # Not static
            return False
        hint = V.graph.sizevars.size_hint(last_numel)
        if hint > threshold:
            return False
        # will need to recompile if we cross a larger power of 2 boundary
        V.graph.sizevars.guard_leq(self.numels[-1], next_power_of_2(hint))  # type: ignore[arg-type]
        return True

    def set_last_usage(self, nodes):
        if not self.inside_reduction or self.persistent_reduction:
            return
        self.last_usage = set(
            itertools.chain.from_iterable(
                n.last_usage for n in nodes if n is not EnableReduction
            )
        )

    def initialize_range_tree(self, pid_cache):
        no_r_dim = not self.inside_reduction or self.numels[-1] == 1

        prefixes = "zyxr"
        active_prefixes = prefixes[-len(self.numels) :]

        grid_dims = "xyz"
        if self.no_x_dim:
            tensor_dims = "r"
        elif no_r_dim:
            tensor_dims = "xyz"
        else:
            tensor_dims = "xyzr"

        tensor_dims = "".join(p for p in tensor_dims if p in active_prefixes)

        for i, prefix in enumerate(active_prefixes):
            is_reduction = prefix == "r"
            tensor_dim = tensor_dims.find(prefix) if prefix in tensor_dims else None
            grid_dim = None if is_reduction else grid_dims.find(prefix)
            index = i if grid_dim is None else grid_dim
            self.range_trees.append(
                IterationRangesRoot(
                    f"{prefix}index",
                    self.numels[i],
                    prefix,
                    index,
                    self,
                    pid_cache=pid_cache,
                    is_loop=is_reduction and not self.persistent_reduction,
                    tensor_dim=tensor_dim,
                    grid_dim=grid_dim,
                )
            )
        for tree in self.range_trees:
            # reduction indexing goes inside a loop
            if not tree.is_loop:
                tree.codegen_header(self.body)
        if self.inside_reduction and self.range_trees[-1].is_loop:
            # workaround for this issue:
            # https://gist.github.com/jansel/6527126f781559095c5531f98a4235a7
            self.body.writeline(f"rbase = {self.range_trees[-1].ranges_code()}")

    def disable_reduction(self):
        should_flush = self.range_trees[-1].is_loop

        @contextlib.contextmanager
        def ctx():
            if self.numels[-1] == 1:
                assert not self.inside_reduction
                yield
                return
            if should_flush:
                # calling codegen_body() will flush all the pending buffers
                # and write out a reduction loop
                self.codegen_body()
            self.inside_reduction = False
            try:
                yield
                if should_flush:
                    # flush out any code before opening the next loop
                    self.codegen_body()
            finally:
                self.inside_reduction = True

        return ctx()

    def set_ranges(self, *lengths):
        assert len(lengths) == len(self.range_trees)
        return [
            ranges.construct(length)
            for length, ranges in zip(lengths, self.range_trees)
        ]

    @staticmethod
    def _split_iteration_ranges(
        groups: Iterable[sympy.Expr], lengths: List[List[sympy.Expr]]
    ):
        sv = V.graph.sizevars
        new_ranges: List[List[sympy.Expr]] = [[] for _ in groups]
        remaining = [sv.simplify(g) for g in groups]
        var_count = itertools.count()

        def add_range(i, expr):
            expr = sv.simplify(expr)
            if not sv.statically_known_multiple_of(remaining[i], expr):
                raise CantSplit()
            # guard on the last item out
            remaining[i] = FloorDiv(remaining[i], expr)
            new_ranges[i].append(expr)
            return next(var_count)

        def make_combined(size, idx1, idx2):
            def getter(flat_vars):
                return size * flat_vars[idx1] + flat_vars[idx2]

            return getter

        return_getters_groups = []
        current_group = 0
        for length_group in lengths:
            return_getters = []
            for size in length_group:
                if sv.statically_known_equals(size, 1):  # type: ignore[arg-type]
                    return_getters.append(lambda _: sympy.Integer(0))
                    continue

                while (
                    current_group < len(remaining)
                    and sv.size_hint(remaining[current_group]) == 1
                ):
                    # scroll to next group with remaining elements
                    current_group += 1

                if sv.size_hint(size) > sv.size_hint(remaining[current_group]):
                    # need to break size in two
                    if not sv.statically_known_multiple_of(
                        size, remaining[current_group]
                    ):
                        raise CantSplit()
                    size1 = remaining[current_group]
                    size2 = FloorDiv(size, remaining[current_group])
                    return_getters.append(
                        make_combined(
                            size2,
                            add_range(current_group, size1),
                            add_range(current_group + 1, size2),
                        )
                    )
                else:
                    return_getters.append(
                        operator.itemgetter(add_range(current_group, size))
                    )
            return_getters_groups.append(return_getters)

        assert all(
            V.graph.sizevars.size_hint(s) == 1 for s in remaining
        ), f"failed to set ranges {remaining} {lengths}"

        return new_ranges, return_getters_groups

    @classmethod
    def is_compatible(
        cls, groups: Iterable[sympy.Expr], lengths: List[List[sympy.Expr]]
    ):
        try:
            cls._split_iteration_ranges(groups, lengths)
            return True
        except CantSplit:
            return False

    def split_and_set_ranges(self, lengths: List[List[sympy.Expr]]):
        """
        We may want to fuse `for i0 in s0*s1` into a tiled kernel with groups (s0, s1).

        To do this we need to split up the iteration space of i0 into something like:
            for i1 in s0:
              for i2 in s1:
                i0 = i1*s1 + i2
                ....

        This function matches and resplits lengths to the groups of
        this kernel to enable tiled + non-tiled fusions.
        """
        groups = [rt.numel for rt in self.range_trees]
        if not self.inside_reduction:
            groups[-1] = sympy.Integer(1)

        if len(lengths) == len(self.range_trees) and all(
            V.graph.sizevars.simplify(sympy_product(x) - g) == 0
            for x, g in zip(lengths, groups)
        ):
            return self.set_ranges(*lengths)

        new_ranges, return_getters_groups = self._split_iteration_ranges(
            groups, lengths
        )
        itervars = list(itertools.chain.from_iterable(self.set_ranges(*new_ranges)))
        return [[fn(itervars) for fn in fns] for fns in return_getters_groups]

    def is_indirect_indexing(self, index: sympy.Expr):
        # tmpX  means indirect indexing
        return free_symbol_startswith(index, "tmp")

    def is_broadcasted(self, index: sympy.Expr):
        # Note. This may not be correct when there is indirect indexing
        if self.is_indirect_indexing(index):
            return False

        index_numels = [1] * len(self.numels)
        for symbol in index.free_symbols:
            if symbol not in self.range_tree_nodes:
                # Non-iterated variables, e.g. strides
                continue
            entry = self.range_tree_nodes[symbol]  # type: ignore[index]
            assert isinstance(entry.parent, IterationRangesRoot)
            index_numels[entry.parent.index] *= entry.length

        # If the index variables only iterate over a subset of the kernel
        # numels, then it must be broadcasted.
        simplify = V.graph.sizevars.simplify
        return any(
            simplify(idx_range) != simplify(iter_range)  # type: ignore[arg-type]
            for idx_range, iter_range in zip(index_numels, self.numels)
        )

    def combine_contiguous_dims(self, index: sympy.Expr, tree: IterationRangesRoot):
        """
        More aggressive simplification to merge contiguous dims
        """
        if isinstance(index, (sympy.Integer, sympy.Symbol)):
            return index
        index_vars, sizes = tree.vars_and_sizes(index)
        if len(sizes) <= 1:
            return index
        new_sizes, reindex, prune = V.graph.sizevars._simplify_loops(
            index_vars, sizes, index_prevent_reordering([index], index_vars, sizes)
        )
        if new_sizes == sizes:
            return index
        new_index_vars = tree.construct(new_sizes)
        new_index = sympy_subs(index, dict(zip(index_vars, reindex(new_index_vars))))
        return new_index

    def index_to_str(self, index: sympy.Expr) -> str:
        """
        Convert an index expr to a string that can be used in triton code.
        e.g. a sympy expression "s2" may actually appear as "ks1" in the triton kernel.

        Index expressions often need to be passed in as arguments to the triton kernel.
        Rename_indexing and codegen_indexing keep track of the needed indices and add
        new parameters to the function signature.
        """
        if isinstance(index, list):
            return f"[{', '.join(map(self.index_to_str, index))}]"
        return texpr(self.rename_indexing(self.codegen_indexing(index)))

    def indexing(
        self,
        index: sympy.Expr,
        *,
        copy_shape=None,
        dense_indexing=False,
        override_mask=None,
        block_ptr=False,
    ) -> Union[IndexingOptions, BlockPtrOptions]:
        """
        Compute the index and mask to pass to tl.load() or tl.store()
        """
        index = self.simplify_indexing(index)
        index = sympy_subs(index, V.graph.sizevars.precomputed_replacements)
        # if simple replacements didn't get rid of floor/ceil, try full subs
        if len(index.atoms(sympy.floor)) or len(index.atoms(sympy.ceiling)):
            index = index.subs(V.graph.sizevars.precomputed_replacements)
        # last resort, if no range vars are in the expr, hoist it
        # TODO instead of trying to blindly find complicated exprs, we should hoist the
        # inputs/outputs sizes and strides, but at the time indexing is generated
        # kernel inputs and outputs are not set yet, we'd need a deeper refactor
        # to do it this way

        if len(index.atoms(sympy.ceiling)):
            for a in index.atoms(sympy.ceiling):
                # for nested exprs, atoms yields top level first (?)
                # so if everything goes fine, lower level replacements will come up empty
                symbols = a.free_symbols
                if len(symbols) > 0 and all(
                    s.name.startswith("s") or s.name.startswith("ps") for s in symbols
                ):
                    replacements = {a: V.graph.sizevars.lookup_precomputed_size(a)}
                    index = sympy_subs(index, replacements)

        index = self.simplify_indexing(index)
        index_vars = index.free_symbols
        has_rindex = False

        mask_vars: Set[str] = set()
        for var in index_vars:
            assert isinstance(var, sympy.Symbol)
            has_rindex = has_rindex or var.name.startswith("r")
            if override_mask:
                pass
            elif var.name.startswith("tmp"):
                # indirect indexing
                cse_var = self.cse.varname_map[var.name]
                mask_vars.update(cse_var.mask_vars)
            elif var.name.startswith(("s", "ps", "i", "u")):
                pass
            else:
                # var is one of xN, yN or rN
                assert var.name[0] in "xyr", var.name
                mask_vars.add(f"{var.name[0]}mask")

        need_dense = (
            config.triton.dense_indexing
            or dense_indexing
            or self._load_mask is not None
        ) and index != 0

        have_dense = True
        have_loop_vars = False
        dense_mask_vars = set()

        for tree in self.active_range_trees():
            if index_vars.intersection(tree.var_list):
                have_loop_vars = True
            else:
                have_dense = False
            dense_mask_vars.add(f"{tree.prefix}mask")

        if (
            block_ptr
            and config.triton.use_block_ptr
            and not override_mask
            and not self._load_mask
            and len(mask_vars - dense_mask_vars) == 0
            and not self.is_indirect_indexing(index)
            and have_loop_vars
            # workaround https://github.com/openai/triton/issues/2821
            and self.index_dtype == "tl.int32"
        ):
            index_relative_to_xyr_index = sympy_subs(
                index, {v: t.expr for v, t in self.range_tree_nodes.items()}
            )
            range_trees = self.active_range_trees(reorder=True)
            symbols = [t.symbol() for t in range_trees]
            strides = [sympy.Wild(f"stride_{s}", exclude=symbols) for s in symbols]
            offset = sympy.Wild("_offset", exclude=symbols)
            m = index_relative_to_xyr_index.match(sympy_dot(symbols, strides) + offset)
            # TODO(jansel): it is sometimes possible to do higher dimensional block_ptrs with
            #               a tl.reshape the correct block.  We will miss these cases today.
            if m:
                self.filter_masks(mask_vars)
                return BlockPtrOptions.create(
                    [m[s] for s in strides],
                    m[offset],
                    range_trees,
                    mask_vars,  # type: ignore[arg-type]
                )

        expand_str = None
        index_str = self.index_to_str(index)
        if isinstance(index, sympy.Integer):
            expand_str = f"{copy_shape}.shape" if copy_shape else self.dense_size_str()
            index_str = f"tl.full({expand_str}, {index_str}, tl.int32)"
            return IndexingOptions(index_str, set(), "None", expand_str, has_rindex)

        if need_dense and not have_dense:
            expand_str = f"{copy_shape}.shape" if copy_shape else self.dense_size_str()
            index_str = f"tl.broadcast_to({index_str}, {expand_str})"
            mask_vars = dense_mask_vars
        elif not have_loop_vars and copy_shape:
            index_str = f"tl.broadcast_to({index_str}, {copy_shape}.shape)"
            mask_vars = dense_mask_vars

        if override_mask:
            mask_vars = {override_mask}

        if self._load_mask:
            mask_vars.add(self._load_mask)

        self.filter_masks(mask_vars)

        mask_str = " & ".join(sorted(map(str, mask_vars))) if mask_vars else "None"
        return IndexingOptions(index_str, mask_vars, mask_str, expand_str, has_rindex)  # type: ignore[arg-type]

    def active_range_trees(self, reorder=False):
        trees = [
            t for t in self.range_trees if t.prefix != "r" or self.inside_reduction
        ]
        if reorder and len(trees) > 1:
            count = sum(t.prefix in "xyz" for t in trees)
            assert "".join(t.prefix for t in trees[:count]) == "zyx"[-count:], [
                t.prefix for t in trees[:count]
            ]
            trees[:count] = reversed(trees[:count])
        return trees

    def filter_masks(self, mask_vars):
        for tree in self.range_trees:
            # Masks are superfluous if we only have one element
            if V.graph.sizevars.statically_known_equals(tree.numel, 1):  # type: ignore[arg-type]
                mask_vars.discard(f"{tree.prefix}mask")
                continue
            # Masks are superfluous if numel is a multiple of BLOCK
            # (We use the fact that BLOCK is required by triton to be a power of 2)
            if tree.prefix.upper() not in config.triton.max_block:
                continue
            max_block = config.triton.max_block[tree.prefix.upper()]
            # Optional optimization: if block divides numel exactly, we will
            # never need to do a masked load to handle stragglers at the end.
            # It's faster to avoid masking at all.  But it is sound to always
            # mask.
            if V.graph.sizevars.statically_known_multiple_of(tree.numel, max_block):  # type: ignore[arg-type]
                mask_vars.discard(f"{tree.prefix}mask")

    def var_ranges(self):
        return dict(
            itertools.chain.from_iterable(
                tree.var_ranges.items() for tree in self.range_trees
            )
        )

    def codegen_indexing(self, expr: sympy.Expr):
        expr = V.graph.sizevars.simplify_with_ranges(expr, self.var_ranges())
        for sym in sorted(expr.free_symbols, key=str):
            if sym in self.range_tree_nodes:
                # if indexing expression is complicated, we precompute it on the host side
                # and send the result as a kernel argument
                replacements = {}
                for ps in self.range_tree_nodes[sym].precomputed_args():  # type: ignore[index]
                    replacements[ps] = V.graph.sizevars.lookup_precomputed_size(ps)
                if len(replacements) > 0:
                    self.range_tree_nodes[sym].expr = sympy_subs(  # type: ignore[index]
                        self.range_tree_nodes[sym].expr, replacements  # type: ignore[index]
                    )
                self.range_tree_nodes[sym].codegen()  # type: ignore[index]
        return expr

    @contextlib.contextmanager
    def mask_loads(self, mask):
        """Context manager to add an additional mask to tl.load/store"""
        prior = self._load_mask
        if prior:
            mask = self.cse.generate(self.compute, f"{mask} & {prior}")

        self._load_mask = mask
        try:
            # TODO(jansel): do we need a reshape here?
            yield mask
        finally:
            self._load_mask = prior

    def generate_assert(self, check):
        return torch.version.hip is None and super().generate_assert(check)

    def load_mask(self, var):
        mask = ""
        mask_vars = set(var.mask_vars)
        if self._load_mask:
            mask_vars.add(self._load_mask)

        if mask_vars:
            mask = (
                f"{next(iter(mask_vars))}"
                if len(mask_vars) == 1
                else f"({' & '.join(str(v) for v in mask_vars)})"
            )
        return mask

    @property
    def assert_function(self) -> str:
        return "tl.device_assert"

    def get_strides_of_load(self, index: sympy.Expr):
        """
        This gets the stride of the index for each of the tiling variables
        (technically, it does it at index 0)

        For example, if
        xindex = x0 + 512*x1 + 1024*r0
        x0 = (xindex//512)
        x1 = (xindex % 512)
        r0 = rindex // 1024

        this function would return
        {xindex: 512, rindex: 1024}
        """
        index_to_tile_indexes = {k: v.expr for k, v in self.range_tree_nodes.items()}
        index_in_tile_vars = sympy_subs(index, index_to_tile_indexes)  # type: ignore[arg-type]
        strides = {}
        for range_tree in self.range_trees:
            s = sympy_index_symbol(range_tree.name)
            strides[s] = sympy_subs(index_in_tile_vars, {s: 1}) - sympy_subs(
                index_in_tile_vars, {s: 0}
            )
        return strides

    def codegen_block_ptr(
        self, name: str, var: str, indexing: BlockPtrOptions, other=""
    ) -> Tuple[str, Optional[DeferredLine], str]:
        advance_block_ptr = None
        check = indexing.boundary_check()
        if not check:
            # workaround https://github.com/openai/triton/issues/2813
            other = ""
        elif other:
            assert other == ", other=0.0"
            other = f", boundary_check={check!r}, padding_option='zero'"
        else:
            other = f", boundary_check={check!r}"
        if (
            self.inside_reduction
            and self.range_trees[-1].is_loop
            and indexing.has_rindex()
        ):
            block_ptr = f"block_ptr{next(self.block_ptr_id)}"
            self.body.writeline(
                DeferredLine(
                    name, f"{block_ptr} = {indexing.format(var, roffset=False)}"
                )
            )
            advance_block_ptr = DeferredLine(
                name,
                f"{block_ptr} = tl.advance({block_ptr}, {indexing.advance_roffset()})",
            )
        else:
            block_ptr = indexing.format(var)
        return block_ptr, advance_block_ptr, other

    def codegen_block_ptr_store_line(self, name, indexing, block_ptr, value, other=""):
        # broadcasting is not implicit for block_ptrs
        value = (
            f"tl.broadcast_to({value}, {self.index_to_str(indexing.reshape_suffix)})"
        )
        # drop any extra size=1 dimensions
        value = triton_reshape(value, indexing.reshape_suffix, indexing.block_shape)
        # workaround https://github.com/openai/triton/issues/2814
        value = f"{value}.to({triton_store_type(V.graph.get_dtype(name))})"
        return f"tl.store({block_ptr}, {value}{other})"

    def load(self, name: str, index: sympy.Expr):
        var = self.args.input(name)
        indirect_indexing = self.is_indirect_indexing(index)
        original_index = index
        indexing = self.indexing(index, block_ptr=True)
        has_rindex = indexing.has_rindex()
        has_tmpmask = indexing.has_tmpmask()

        # Keep the variable in cache if were going to reuse it. Equiv., if any of the following hold
        #  1) We are doing broadcasting
        #  2) It is a non-coalesced load. The intuition is that if it's
        #  non-coalesced, we will likely load each element multiple times in
        #  practice.
        #  3) It will be used later and it won't be CSE'd. Equiv., if all the following hold
        #   3.1) We are in a reduction loop
        #   3.2) Its not its last use
        #   3.3) This load will not be lifted to the body
        #
        is_coalesced = any(
            i == 1 for i in self.get_strides_of_load(original_index).values()
        )
        if self.is_broadcasted(original_index):
            ep = ", eviction_policy='evict_last'"
        elif not is_coalesced:
            ep = ", eviction_policy='evict_last'"
        elif self.inside_reduction and self.range_trees[-1].is_loop:
            if name in self.args.inplace_buffers:
                names = set(self.args.inplace_buffers[name].other_names)
            else:
                names = {name}
            last_use = len(names & self.last_usage) > 0
            evict_last = not last_use and (has_rindex or indirect_indexing)
            if evict_last:
                ep = ", eviction_policy='evict_last'"
            else:
                ep = ", eviction_policy='evict_first'"
        else:
            ep = ""
        # "other" below is a workaround for https://github.com/openai/triton/issues/737
        # for bool, even though it's likely subject to the same bug, setting `other` leads
        # to LLVM errors so we are skipping it for now
        if (
            (has_tmpmask or has_rindex)
            and V.graph.get_dtype(name) != torch.bool
            and indexing.has_mask()
        ):
            other = ", other=0.0"
        else:
            other = ""

        advance_block_ptr = None
        append_broadcast = None
        if V.graph.is_unspec_arg(name):
            line = var
        else:
            if isinstance(indexing, BlockPtrOptions):
                block_ptr, advance_block_ptr, other = self.codegen_block_ptr(
                    name, var, indexing, other
                )
                line = f"tl.load({block_ptr}{other}{ep})"
                # add needed size=1 dimensions
                line = triton_reshape(
                    line, indexing.block_shape, indexing.reshape_suffix
                )
            elif isinstance(original_index, sympy.Integer):
                line = f"tl.load({var} + ({original_index}))"
                append_broadcast = indexing.expand_str
            else:
                line = f"tl.load({var} + ({indexing.index_str}), {indexing.mask_str}{ep}{other})"

            dtype = V.graph.get_dtype(name)
            if dtype in (torch.float16, torch.bfloat16):
                line += ".to(tl.float32)"
            if dtype == torch.bool and torch.version.hip is None:
                # Workaround for https://github.com/openai/triton/issues/2151
                # tl.load returns int8 when loading from pointer to int1
                # NOTE: Currently causes hangs on bool UTs for ROCm
                line += ".to(tl.int1)"

        if has_tmpmask:
            # Masked loads must come after the mask is computed
            load_buffer = self.compute
        elif (
            self.inside_reduction
            and self.range_trees[-1].is_loop
            and not indirect_indexing
            and not has_rindex
        ):
            # can lift a common load outside of reduction loop
            # One exception is when this is an indirect_load.
            load_buffer = self.body
        else:
            load_buffer = self.loads

        result_var = self.cse.generate(load_buffer, line)
        assert isinstance(result_var, TritonCSEVariable)
        result_var.mask_vars = indexing.mask_vars  # type: ignore[assignment]

        if append_broadcast:
            line = f"tl.broadcast_to({result_var}, {append_broadcast})"
            result_var = self.cse.generate(load_buffer, line)

        if advance_block_ptr:
            load_buffer.writeline(advance_block_ptr)

        if not self.inside_reduction or (not indexing.has_rmask() and not has_rindex):
            self.outside_loop_vars.add(result_var)

        return result_var

    def store(
        self, name: str, index: sympy.Expr, value: CSEVariable, mode: StoreMode = None
    ) -> None:
        var = self.args.output(name)
        original_index = index
        indexing = self.indexing(index, dense_indexing=True, block_ptr=mode is None)

        # Guard against write-after-read corruption in triton.
        # See # https://github.com/openai/triton/issues/1615
        # This triton bug means that a load which is broadcasted over multiple
        # warps may see the result of a store that happens later in the triton
        # program. The workaround is to add a barrier before storing, which
        # enforces that all warps have already read the data.
        is_inplace = name in self.args.inplace_buffers
        is_broadcasted = self.is_broadcasted(original_index)
        if is_inplace and is_broadcasted:
            self.stores.writeline(DeferredLine(name, "tl.debug_barrier()"))

        advance_block_ptr = None
        if isinstance(indexing, BlockPtrOptions):
            block_ptr, advance_block_ptr, other = self.codegen_block_ptr(
                name, var, indexing
            )
            # block_ptr stores don't do implicit casting
            line = self.codegen_block_ptr_store_line(
                name, indexing, block_ptr, value, other
            )
        elif mode is None:
            line = f"tl.store({var} + ({indexing.index_str}), {value}, {indexing.mask_str})"
        elif mode == "atomic_add":
            line = f"tl.atomic_add({var} + ({indexing.index_str}), {value}, {indexing.mask_str})"
        else:
            raise NotImplementedError(f"store mode={mode}")
        self.stores.writeline(DeferredLine(name, line))
        if advance_block_ptr:
            self.stores.writeline(advance_block_ptr)

        if not self.inside_reduction:
            self.outside_loop_vars.add(value)

    def bucketize(
        self,
        values: CSEVariable,
        offsets_name: str,
        offsets_size: sympy.Expr,
        indexing_dtype: torch.dtype,
        right: bool,
    ) -> CSEVariable:
        """
        See [Note: Inductor bucketize op]
        """

        # Triton performance for bucketize_binary_search is much better when the number
        # of threads equals the number of elements.
        # If we're trying to use a bucketize kernel, we should make sure that an
        # autotuning config with num_elements_per_warp=32 exists.
        self.autotune_hints.add(AutotuneHint.ELEMENTS_PER_WARP_32)

        offsets_ptr = self.args.input(offsets_name)
        block_size = self.dense_size_str()
        offsets_size_str = self.index_to_str(offsets_size)

        if indexing_dtype == torch.int32:
            triton_dtype = "tl.int32"
        elif indexing_dtype == torch.int64:
            triton_dtype = "tl.int64"
        else:
            raise NotImplementedError(
                "Bucketize only supports indexing with int32 and int64"
            )

        result = self.cse.generate(
            self.compute,
            f"triton_helpers.bucketize_binary_search({values}, {offsets_ptr}, {triton_dtype}, {right}, {offsets_size_str}, {block_size})",  # noqa: B950 line too long
        )

        return result

    def reduction_resize(self, value):
        ndims = self.triton_tensor_ndim()
        if ndims == 1:
            return f"triton_helpers.promote_to_tensor({value})"

        sizes = [":"] * ndims
        sizes[-1] = "None"
        return f"{value}[{', '.join(sizes)}]"

    @staticmethod
    def _map_tuple_or_scalar(fn, value):
        if isinstance(value, tuple):
            return tuple(map(fn, value))
        return fn(value)

    def reduction(
        self,
        dtype: torch.dtype,
        src_dtype: torch.dtype,
        reduction_type: ReductionType,
        value: Union[CSEVariable, Tuple[CSEVariable, ...]],
    ) -> Union[CSEVariable, Tuple[CSEVariable, ...]]:
        assert self.inside_reduction
        masks = {f"{tree.prefix}mask" for tree in self.range_trees}
        self.filter_masks(masks)
        masks = sorted(masks)
        if self._load_mask:
            masks.append(self._load_mask)
        reduction_range_prefix = self.range_trees[-1].prefix

        # Say we have
        #     tmp0 = ops.constant(1, torch.int64)
        #     tmp1 = ops.reduction(torch.int64, torch.int64, "sum", tmp0)
        # tmp0 in the triton code is either a scalar, or single-element tensor
        # so if we emit tl.sum directly, it will only give 1 instead of RBLOCK * 1
        # To avoid this, we broadcast to the expected shape first.
        dense_size_str = self.dense_size_str()
        value = self._map_tuple_or_scalar(
            lambda v: self.cse.generate(
                self.compute, f"tl.broadcast_to({v}, {dense_size_str})"
            ),
            value,
        )

        dim: int
        root_op: str

        def final_reduction(value):
            use_helper = reduction_type in {"any", "max", "min", "prod"}
            module = "triton_helpers" if use_helper else "tl"
            if reduction_type in {"max", "min"}:
                return self.reduction_resize(
                    f"{module}.{reduction_type}2({value}, {dim})"
                )
            return self.reduction_resize(f"{module}.{reduction_type}({value}, {dim})")

        def final_argreduce(buffer, result_var, value, index):
            buffer.splice(
                f"""\
                _, {result_var}_tmp = triton_helpers.{root_op}_with_index({value}, {index}, {dim})
                {result_var} = {self.reduction_resize(f'{result_var}_tmp')}
                """
            )

        cache_key = (src_dtype, reduction_type, value)
        if cache_key in self.cse.reduction_cache:
            return self.cse.reduction_cache[cache_key]

        dim = self.triton_tensor_ndim() - 1
        acc_type = triton_acc_type(src_dtype)
        result_var: Any = self.cse.newvar()
        result_var.mask_vars = {var for var in masks if var[0] != "r"}
        cond = " & ".join(masks)

        def where_cond(tval, fval):
            if not cond:
                return tval
            return TritonKernelOverrides.where(cond, tval, fval)

        if self.persistent_reduction:
            default = ir.Reduction.default_value(reduction_type, src_dtype)
            default = self._map_tuple_or_scalar(triton_constant, default)

            def _mask_value(value, default):
                return self.cse.generate(self.compute, where_cond(value, default))

            if isinstance(value, tuple):
                masked_value = [_mask_value(v, d) for v, d in zip(value, default)]
            else:
                masked_value = _mask_value(value, default)

            if reduction_type in {"argmax", "argmin"}:
                accumulator_index = str(
                    self.cse.generate(
                        self.compute,
                        f"tl.broadcast_to({reduction_range_prefix}index, {masked_value}.shape)",
                    )
                )
                root_op = {"argmax": "max", "argmin": "min"}[reduction_type]
                final_argreduce(
                    self.compute, result_var, masked_value, accumulator_index
                )
            elif reduction_type == "welford_reduce":
                # For persistent reductions, don't bother with
                # welford's algorithm since it uses more registers, and
                # taking two reductions doesn't increase memory usage.
                sum_ = ops.reduction(dtype, dtype, "sum", value)
                self.inside_reduction = False
                rnumel = ops.index_expr(self.numels[-1], dtype)
                mean = ops.truediv(sum_, rnumel)

                self.inside_reduction = True
                dx = ops.sub(value, mean)
                dx2 = ops.mul(dx, dx)
                m2 = ops.reduction(dtype, dtype, "sum", dx2)
                result_var = (mean, m2, rnumel)
            elif reduction_type == "welford_combine":
                mean, m2, weight = masked_value
                welford = f"triton_helpers.welford({mean}, {m2}, {weight}, {dim})"
                mean, m2, weight = (self.cse.newvar() for _ in range(3))
                self.compute.writeline(f"{mean}, {m2}, {weight} = {welford}")

                result_var = tuple(
                    self.cse.generate(self.compute, self.reduction_resize(var_name))
                    for var_name in (mean, m2, weight)
                )
            else:
                result_var = self.cse.generate(
                    self.compute, final_reduction(masked_value)
                )
        else:
            accumulator = f"_{result_var}"
            default = ir.Reduction.default_accumulator(reduction_type, src_dtype)
            default = self._map_tuple_or_scalar(triton_constant, default)
            if not isinstance(default, tuple):
                self.body.writeline(
                    f"{accumulator} = tl.full({self.dense_size_str()}, {default}, {acc_type})"
                )

            if reduction_type in {"argmax", "argmin"}:
                accumulator_index = f"_{result_var}_index"
                long_max = torch.iinfo(torch.int64).max
                self.body.writeline(
                    f"{accumulator_index} = tl.full({self.dense_size_str()}, {long_max}, tl.int64)"
                )
                root_op = {"argmax": "max", "argmin": "min"}[reduction_type]

                self.compute.splice(
                    f"""\
                {accumulator}_next, {accumulator_index}_next = triton_helpers.{root_op}imum_with_index(
                    {accumulator}, {accumulator_index}, {value}, {reduction_range_prefix}index
                )
                {accumulator} = {where_cond(f'{accumulator}_next', accumulator)}
                {accumulator_index} = {where_cond(f'{accumulator_index}_next', accumulator_index)}
                """
                )
                final_argreduce(self.suffix, result_var, accumulator, accumulator_index)
            elif is_welford_reduction(reduction_type):
                accumulator = f"{result_var}_mean"
                accumulator_m2 = f"{result_var}_m2"
                accumulator_weight = f"{result_var}_weight"
                self.body.writeline(
                    f"{accumulator} = tl.zeros({self.dense_size_str()}, {acc_type})"
                )
                self.body.writeline(
                    f"{accumulator_m2} = tl.zeros({self.dense_size_str()}, {acc_type})"
                )
                self.body.writeline(
                    f"{accumulator_weight} = tl.zeros({self.dense_size_str()}, {acc_type})"
                )

                if reduction_type == "welford_combine":
                    mean, m2, weight = value
                    self.compute.splice(
                        f"""\
                    {accumulator}_next, {accumulator_m2}_next, {accumulator_weight}_next = triton_helpers.welford_combine(
                        {accumulator}, {accumulator_m2}, {accumulator_weight},
                        {mean}, {m2}, {weight}
                    )
                    """
                    )
                else:
                    assert reduction_type == "welford_reduce"
                    self.compute.splice(
                        f"""\
                    {accumulator}_next, {accumulator_m2}_next, {accumulator_weight}_next = triton_helpers.welford_reduce(
                        {value}, {accumulator}, {accumulator_m2}, {accumulator_weight}, roffset == 0
                    )
                    """
                    )

                self.compute.splice(
                    f"""\
                {accumulator} = {where_cond(f'{accumulator}_next', accumulator)}
                {accumulator_m2} = {where_cond(f'{accumulator_m2}_next', accumulator_m2)}
                {accumulator_weight} = {where_cond(f'{accumulator_weight}_next', accumulator_weight)}
                """
                )

                result_mean = result_var
                result_m2 = self.cse.newvar()
                result_weight = self.cse.newvar()
                self.suffix.splice(
                    f"""\
                {result_mean}_tmp, {result_m2}_tmp, {result_weight}_tmp = triton_helpers.welford(
                    {accumulator}, {accumulator_m2}, {accumulator_weight}, {dim}
                )
                {result_mean} = {self.reduction_resize(f'{result_mean}_tmp')}
                {result_m2} = {self.reduction_resize(f'{result_m2}_tmp')}
                {result_weight} = {self.reduction_resize(f'{result_weight}_tmp')}
                """
                )
                result_var = result_mean, result_m2, result_weight
            else:
                combine_fn = ir.get_reduction_combine_fn(reduction_type, src_dtype)
                updated = combine_fn(accumulator, value)
                self.compute.writeline(
                    f"{accumulator} = {where_cond(updated, accumulator)}"
                )

                if src_dtype == torch.bool:
                    # This is only really used for aten.any. It changes the
                    # final reduction of a non-persistent reduction from
                    #     tmp5 = triton_helpers.max(_tmp5, 1)[:, None]
                    # to
                    #     tmp5 = triton_helpers.max(_tmp5.to(tl.int8), 1)[:, None].to(tl.int1)
                    # which is needed because tl.reduce doesn't support tl.int1
                    accumulator = f"{accumulator}.to(tl.int8)"
                    result_type = triton_compute_type(dtype)
                    self.suffix.writeline(
                        f"{result_var} = {final_reduction(accumulator)}.to({result_type})"
                    )
                else:
                    self.suffix.writeline(
                        f"{result_var} = {final_reduction(accumulator)}"
                    )

        self.cse.reduction_cache[cache_key] = result_var

        if isinstance(result_var, tuple):
            self.outside_loop_vars |= set(result_var)
        else:
            self.outside_loop_vars.add(result_var)

        return result_var

    def store_reduction(self, name: str, index: sympy.Expr, value: CSEVariable):
        assert self.inside_reduction
        self.inside_reduction = False
        indexing = self.indexing(index, block_ptr=True)
        self.inside_reduction = True
        var = self.args.output(name)

        if isinstance(indexing, BlockPtrOptions):
            self.suffix.writeline(
                DeferredLine(
                    name,
                    self.codegen_block_ptr_store_line(
                        name,
                        indexing,
                        indexing.format(var),
                        value,
                        f", boundary_check={indexing.boundary_check()!r}",
                    ),
                )
            )
        else:
            assert isinstance(indexing, IndexingOptions)
            self.suffix.writeline(
                DeferredLine(
                    name,
                    f"tl.store({var} + ({indexing.index_str}), {value}, {indexing.mask_str})",
                )
            )

    def _lift_helper(self, fn, num_args) -> str:
        # Lift IR function for scan operations into a triton function
        # in the global namespace
        helper = IndentedBuffer()
        helper.writeline("@triton.jit")
        args = [tuple(f"arg{i}_{n}" for n in range(num_args)) for i in range(2)]
        signature = ", ".join(itertools.chain.from_iterable(args))
        helper.writeline(f"def {{name}}({signature}):")

        cse = CSE(prefix="", suffix="")
        overrides = TritonOverrides(V.MockHandler())

        # Build a name that changes depending on fn to workaround a triton bug
        # where the combine_fn to reduce and scan is not hashed, and so different
        # scan ops may collide in the triton cache.
        # This is fixed with the latest triton pin, but not the triton-rocm pin.
        helper_name = "_triton_helper_fn"

        class CSEProxy:
            def __getattr__(self, name: str) -> Callable[..., CSEVariable]:
                def inner(*args, **kwargs):
                    nonlocal helper_name
                    helper_name += f"_{name}"
                    return cse.generate(
                        helper,
                        getattr(overrides, name)(*args, **kwargs),
                    )

                return inner

        with helper.indent(), V.set_ops_handler(CSEProxy()):
            outputs = fn(*args)
            outputs = ", ".join(str(output) for output in outputs)
            helper.writeline(f"return {outputs}")

        return self.helper_functions.add(helper.getvalue(), base_name=helper_name)

    def scan(
        self,
        dtypes: Tuple[torch.dtype, ...],
        combine_fn: Callable[
            [Tuple[CSEVariable, ...], Tuple[CSEVariable, ...]], Tuple[CSEVariable, ...]
        ],
        values: Tuple[CSEVariable, ...],
    ) -> Tuple[CSEVariable, ...]:
        assert self.inside_reduction
        masks = {f"{tree.prefix}mask" for tree in self.range_trees}
        self.filter_masks(masks)
        masks = sorted(masks)
        assert not self._load_mask, "ops.scan not supported inside ops.masked"
        reduction_range_prefix = self.range_trees[-1].prefix

        broadcasted_values = []
        accumulators = []

        cse_compute = functools.partial(self.cse.generate, self.compute)
        combine_helper_fn = self._lift_helper(combine_fn, len(values))
        dim = self.triton_tensor_ndim() - 1

        for value, dtype in zip(values, dtypes):
            acc_type = triton_acc_type(dtype)
            cond = " & ".join(masks)

            value_dtype = self.cse.generate(
                self.compute,
                f"{value}.to({triton_compute_type(dtype)})",
            )
            value = self.cse.generate(
                self.compute,
                f"tl.broadcast_to({value_dtype}, {self.dense_size_str()})",
            )
            broadcasted_values.append(value)

            acc_type = triton_acc_type(dtype)
            cond = " & ".join(masks)

            if not self.persistent_reduction:
                accumulator = self.cse.newvar()
                reduced_size = self.dense_size_list()
                reduced_size[-1] = "1"
                reduced_size = f"[{', '.join(reduced_size)}]"

                default = "float('nan')" if dtype.is_floating_point else "-1"
                self.body.writeline(
                    f"{accumulator} = tl.full({reduced_size}, {default}, {acc_type})"
                )

                accumulators.append(accumulator)

        def csv(values):
            return " ".join(f"{value}," for value in values)

        def cse_multiple(line, n, masks):
            cache_keys = [f"{line}, {i}, {masks}" for i in range(n)]
            if all(cache_key in self.cse.cache for cache_key in cache_keys):
                return [self.cse.cache[cache_key] for cache_key in cache_keys]
            result_vars = [self.cse.newvar() for _ in range(n)]
            self.compute.writeline(
                f"{csv(result_vars)} = {line}",
            )
            for result_var, cache_key in zip(result_vars, cache_keys):
                if masks:
                    result_var.mask_vars = masks  # type: ignore[attr-defined]
                self.cse.cache[cache_key] = result_var
            return tuple(result_vars)

        partial_scan_vars = cse_multiple(
            f"tl.associative_scan(({csv(broadcasted_values)}), {dim}, {combine_helper_fn})",
            len(values),
            masks,
        )

        if not self.persistent_reduction:
            partial_reduce_vars = pytree.tree_map(
                self.reduction_resize,
                cse_multiple(
                    f"tl.reduce(({csv(broadcasted_values)}), {dim}, {combine_helper_fn})",
                    len(values),
                    None,
                ),
            )
            accs_next = combine_fn(tuple(accumulators), partial_reduce_vars)
            full_scan_vars = combine_fn(tuple(accumulators), partial_scan_vars)
            result_vars = [
                cse_compute(f"tl.where(roffset > 0, {full_scan}, {partial_scan})")
                for full_scan, partial_scan in zip(full_scan_vars, partial_scan_vars)
            ]
            for acc_next, accumulator, partial_reduce in zip(
                accs_next, accumulators, partial_reduce_vars
            ):
                self.compute.writeline(
                    f"{accumulator} = tl.where(roffset > 0, {acc_next}, {partial_reduce})"
                )
        else:
            result_vars = partial_scan_vars

        for result_var in result_vars:
            result_var.mask_vars = masks  # type: ignore[attr-defined]

        return tuple(result_vars)

    def codegen_body(self):
        """
        Concat output code from index_code, loads, compute, stores,
        suffix into self.body.

        For pointwise kernels, this is called just once at the end.

        For reduction kernels, this generates a loop over the reduction
        axis.
        """
        if not (
            self.indexing_code
            or self.loads
            or self.stores
            or self.compute
            or self.suffix
        ):
            return

        if self.inside_reduction and self.range_trees[-1].is_loop:
            self.body.writeline("for roffset in range(0, rnumel, RBLOCK):")
            with self.body.indent():
                # last range tree is always reduction
                self.range_trees[-1].codegen_header(self.body)
                self.body.splice(self.indexing_code)
                self.body.splice(self.loads)
                self.body.splice(self.compute)
                self.body.splice(self.stores)

            # invalidate any caches that came from inside the reduction loop
            self.cse.invalidate(self.outside_loop_vars)
            self.range_trees[-1].cache_clear()
        else:
            self.body.splice(self.indexing_code)
            self.body.splice(self.loads)
            self.body.splice(self.compute)
            self.body.splice(self.stores)
        self.body.splice(self.suffix)
        self.indexing_code.clear()
        self.loads.clear()
        self.compute.clear()
        self.stores.clear()
        self.suffix.clear()

    def codegen_kernel_benchmark(self, num_gb, grid=None):
        result = IndentedBuffer()
        argdefs, call_args, signature = self.args.python_argdefs()

        result.writelines(["", "", "def get_args():"])
        with result.indent():
            name_cnt = itertools.count()
            var_names = []
            for arg_name, arg_sig in zip(call_args, signature):
                var_name = f"arg_{next(name_cnt)}"
                buf = V.graph.get_buffer(arg_name)
                if buf:
                    result.writeline(
                        f"{var_name} = rand_strided({V.graph.sizevars.size_hints(buf.get_size())}, {V.graph.sizevars.size_hints(buf.get_stride())}, device='{buf.get_device()}', dtype={buf.get_dtype()})"  # noqa: B950 line too long
                    )
                elif arg_name in V.graph.constants:
                    # note that random seed is put in V.graph.constants
                    const_tensor = V.graph.constants[arg_name]
                    result.writeline(
                        f"{var_name} = rand_strided({V.graph.sizevars.size_hints(const_tensor.size())}, {V.graph.sizevars.size_hints(const_tensor.stride())}, device='{const_tensor.device}', dtype={const_tensor.dtype})"  # type: ignore[arg-type]  # noqa: B950 line too long
                    )
                elif isinstance(arg_sig, SizeArg):
                    symval_hint = V.graph.sizevars.size_hint(arg_sig.expr)

                    # Force the seed_offset to be 0 so calls to the same kernel
                    # using different seed offset will have the same benchmark harness.
                    # We can dedup kernel definitions in this case.
                    if "seed_offset" in arg_sig.name:
                        symval_hint = 0
                    result.writeline(f"{var_name} = {symval_hint}")
                else:
                    raise KeyError(
                        f"Don't find the buffer or const tensor for {arg_name}"
                    )
                var_names.append(var_name)
            result.writeline(f"return {', '.join(var_names)},")

        result.writelines(["\n", "\n", "def call(args):"])
        if grid is None:
            grid = []
            extra_args = []
            extra_args_str = None
            for tree in self.active_range_trees():
                expr = pexpr(V.graph.sizevars.size_hint(tree.numel))
                extra_args.append(expr)
                if tree.prefix != "r":
                    grid.append(expr)
            if self.need_numel_args():
                extra_args_str = ", ".join(map(str, extra_args)) + ", "
            else:
                extra_args_str = ""
            grid_arg = f"{extra_args_str}grid=grid({', '.join(grid)})"
        else:
            grid_arg = f"grid={grid}"
        index = V.graph.scheduler.current_device.index
        with result.indent():
            result.writeline(f"with {V.graph.device_ops.device_guard(index)}:")
            with result.indent():
                result.writeline(
                    V.graph.device_ops.set_device(index)
                )  # no-op to ensure context
                stream_name = f"stream{index}"
                result.writeline(f"{stream_name} = get_raw_stream({index})")
                result.writeline(
                    f"{str(Placeholder.KERNEL_NAME)}.run(*args, {grid_arg}, stream={stream_name})"
                )

        # benchmark all configs
        result.writelines(["\n", "\n", "def benchmark_all_configs(args):"])
        with result.indent():
            result.writeline(f"with {V.graph.device_ops.device_guard(index)}:")
            with result.indent():
                result.writeline(
                    V.graph.device_ops.set_device(index)
                )  # no-op to ensure context
                result.writeline(
                    f"return {str(Placeholder.KERNEL_NAME)}.benchmark_all_configs(*args, {grid_arg})"
                )

        result.writelines(["\n", "\n", "if __name__ == '__main__':"])
        with result.indent():
            result.writeline("from triton.testing import do_bench")
            result.writeline("")

            result.writeline("args = get_args()")
            result.writeline(
                "ms = do_bench(lambda: call(args), rep=40, fast_flush=True)"
            )
            result.writeline(f"num_gb = {num_gb}")
            result.writeline("gb_per_s = num_gb / (ms / 1e3)")
            result.writeline(
                'print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")'
            )

        return result

    def imports_for_benchmark_kernel(self):
        return textwrap.dedent(
            """
            from torch._dynamo.testing import rand_strided
            {}
            import torch
            from torch._inductor.triton_heuristics import grid, split_scan_grid
        """.format(
                V.graph.device_ops.import_get_raw_stream_as("get_raw_stream")
            )
        )

    def estimate_kernel_num_bytes(self):
        """
        Try the best to estimate the total size (in bytes) of the
        kernel's inputs and outputs, which is used for estimating the memory
        throughput of this kernel. This information is used for checking how
        far we are from the peak memory bandwidth. It's important that
        we want to avoid overestimating the sizes of the inputs and outputs,
        because it can wrongfully give us a very large memory traffic value,
        which may be even larger than the theoretical bandwidth and thus
        become very misleading. This is particularly problematic for cases
        where we slice some inputs. In those cases, we should only count
        the size of the "slices" instead of the original inputs, because
        only the slices contribute to the real memory traffic.
        """
        nbytes = []
        ninplace_args = len(unique(self.args.inplace_buffers.values()))
        _, call_args, _ = self.args.python_argdefs()

        # For pointwise and reduction kernels, this is the upper-bound numels
        # for the output buffer.
        # FIXME: This is not exactly right for cases like below:
        #    def foo(tensor0, tensor1):
        #        x0 = narrow(tensor0)
        #        return cat(x0, tensor1)
        # For this example, we will end up overestimate the size for the
        # slice s0. Potentially, we could have precise inputs information
        # if we maintained the original inputs of the Pointwise kernel created
        # for the "cat". However, I think it might be a bit overwhelming that
        # we add such complexity only for handling some particular cases for
        # benchmarking.
        out_numel = V.graph.sizevars.size_hint(sympy_product(self.numels))
        for i, arg in enumerate(call_args):
            # "buf" may be narrowed. In this case, the number of memory accesses
            # should be estimated based on the reinterpreted layout.
            # On the other hand, buf may be broadcasted. In this case,
            # counting the size of the underline storage would give us
            # a better estimation in terms of memory accesses.
            if arg not in self.buf_accesses:
                nbytes.append(0)
                continue
            arg_numel = V.graph.get_numel(arg)
            buf_size = V.graph.sizevars.size_hint(arg_numel)
            if buf_size > out_numel:
                # This arg points to a buf that has been sliced.
                # We need to count each individual slice to have
                # a better estimation.
                indices: Set[Any] = set()
                no_index_dep_count = 0
                for dep in self.buf_accesses[arg]:
                    if isinstance(dep, (StarDep, WeakDep)):
                        indices.add(f"no_index_dep_{no_index_dep_count}")
                        no_index_dep_count += 1
                    else:
                        indices.add(dep.index)
                numel = len(indices) * out_numel
            else:
                numel = buf_size
            dtype = V.graph.get_dtype(arg)
            dtype_size = get_dtype_size(dtype)
            nbytes.append(numel * dtype_size * (1 + int(i < ninplace_args)))
        return sum(nbytes)

    def _get_heuristic(self):
        if self.persistent_reduction:
            assert self.inside_reduction
            return "persistent_reduction"
        elif self.inside_reduction:
            return "reduction"
        return "pointwise"

    def codegen_kernel(self, name=None):
        code = IndentedBuffer()

        size_hints = []
        for numel in self.numels:
            numel_hint = V.graph.sizevars.symbolic_hint(numel)
            if not isinstance(numel_hint, (int, sympy.Integer)):
                # This default heuristic hint was picked carefully: it is
                # large, to ensure that we don't shrink the block size (since
                # if you don't have many elements, it'd be wasteful to pick a
                # large block size).  Since we don't know how many elements we
                # might have, we should be OK with some inefficiency to make
                # sure we handle the large case well.  8192 is the largest
                # block size we support, so we pick that.
                #
                # If we have a better hint for unbacked SymInts (e.g., because
                # a user told us, or we are tracking upper bounds) we could
                # use that here.
                size_hint = 8192
            else:
                size_hint = next_power_of_2(int(numel_hint))
            size_hints.append(size_hint)

        if not self.inside_reduction:
            size_hints.pop()

        heuristics = self._get_heuristic()

        if name is None:
            code.splice(gen_common_triton_imports())

            if config.benchmark_kernel:
                code.splice(self.imports_for_benchmark_kernel())

        argdefs, _, signature = self.args.python_argdefs()
        # maps actual expression to SizeArg if it is in sizevars replacements
        for i, arg in enumerate(signature):
            if isinstance(arg, SizeArg):
                # mypy is unhappy about the sympy.Expr
                # type for the key of the dict below
                symbol = cast(sympy.Symbol, arg.expr)
                if symbol in V.graph.sizevars.inv_precomputed_replacements:
                    signature[i] = SizeArg(
                        arg.name, V.graph.sizevars.inv_precomputed_replacements[symbol]
                    )

        mutated_args = set()
        for mutation in self.mutations:
            if mutation in self.args.input_buffers:
                mutated_args.add(self.args.input_buffers[mutation])
            if (
                mutation in self.args.inplace_buffers
                and mutation not in V.graph.removed_buffers
                and mutation not in self.removed_buffers
            ):
                mutated_args.add(self.args.inplace_buffers[mutation].inner_name)
            if mutation in self.args.output_buffers:
                mutated_args.add(self.args.output_buffers[mutation])
        mutated_args = sorted(mutated_args)

        triton_meta_signature = signature_to_meta(
            signature, size_dtype=self.index_dtype
        )
        triton_meta = {
            "signature": triton_meta_signature,
            "device": V.graph.scheduler.current_device.index,
            "device_type": V.graph.scheduler.current_device.type,
            "constants": {},
        }

        inductor_meta = {
            "autotune_hints": set(self.autotune_hints),
            "kernel_name": str(Placeholder.DESCRIPTIVE_NAME),
            "mutated_arg_names": mutated_args,
            "no_x_dim": self.no_x_dim,
            "backend_hash": torch.utils._triton.triton_hash_with_backend(),
        }
        num_gb = None
        if config.benchmark_kernel or config.profile_bandwidth:
            num_gb = self.estimate_kernel_num_bytes() / 1e9
            inductor_meta["kernel_num_gb"] = num_gb

        for tree in self.active_range_trees():
            sizearg = SizeArg(f"{tree.prefix}numel", tree.numel)
            signature.append(sizearg)
            triton_meta_signature[len(argdefs)] = signature_of(
                sizearg, size_dtype=self.index_dtype
            )
            argdefs.append(f"{tree.prefix}numel")
            # constexpr version causes issues, see
            # https://github.com/pytorch/torchdynamo/pull/1362
            # triton_meta["constants"][len(argdefs)] = V.graph.sizevars.size_hint(
            #     tree.numel
            # )
            # argdefs.append(f"{tree.prefix}numel: tl.constexpr")
        triton_meta["configs"] = [config_of(signature)]

        # Triton compiler includes equal_to_1 args into constants even
        # when they are not constexpr. otherwise there may be a segfault
        # during launching the Inductor-compiled Triton kernel.
        # https://github.com/pytorch/pytorch/issues/120478#issuecomment-1962822307
        # https://github.com/openai/triton/blob/231efe9ed2d200be0f69a07c298e4342b08efe3d/python/triton/runtime/jit.py#L384
        for arg_num in triton_meta["configs"][0].equal_to_1:  # type: ignore[index]
            triton_meta["constants"][arg_num] = 1  # type: ignore[index]

        self.triton_meta = triton_meta

        for tree in self.range_trees:
            if tree.prefix == "r" and self.persistent_reduction:
                # RBLOCK for persistent_reduction is defined in codegen_static_numels
                continue
            if tree.tensor_dim is None:
                continue
            argdefs.append(f"{tree.prefix.upper()}BLOCK : tl.constexpr")

        self.codegen_body()

        for helper in self.helper_functions:
            code.writeline("")
            code.splice(helper)

        if self.inside_reduction:
            reduction_hint = self.reduction_hint
            heuristics_line = f"""
                @triton_heuristics.{heuristics}(
                    size_hints={size_hints!r},
                    reduction_hint={reduction_hint},
                    filename=__file__,
                    triton_meta={triton_meta!r},
                    inductor_meta={inductor_meta!r}
                )
                @triton.jit
            """
        else:
            tile_hint = ""
            if len(size_hints) == 2:
                if len(signature) == 4:  # input, output and 2 args
                    tile_hint = "tile_hint=TileHint.SQUARE,"
                else:
                    tile_hint = "tile_hint=TileHint.DEFAULT,"
            heuristics_line = f"""
                @triton_heuristics.{heuristics}(
                    size_hints={size_hints!r}, {tile_hint}
                    filename=__file__,
                    triton_meta={triton_meta!r},
                    inductor_meta={inductor_meta!r},
                    min_elem_per_thread={self.min_elem_per_thread}
                )
                @triton.jit
            """
        code.splice(heuristics_line)
        code.writeline(
            f"def {name or str(Placeholder.KERNEL_NAME)}({', '.join(argdefs)}):"
        )
        with code.indent():
            self.codegen_static_numels(code)
            for old, new in self.args.aliases():
                code.writeline(f"{old} = {new}")
            code.splice(self.body)

        if config.benchmark_kernel:
            code.splice(self.codegen_kernel_benchmark(num_gb))

        return code.getvalue()

    def codegen_static_numels(self, code):
        """
        We get a small speedup from hard coding numels if they are static.

        This code stomps on the passed-in values by writing an constant to the top of the kernel.

        In a kernel like:
        def KERNEL_NAME(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):

        We would add
        xnumel = 4096
        rnumel = 768

        After the signature, before the kernel code, if we decided to make these static. As its hardcoded, it becomes
        a better signal to triton on how to unroll and do some static indexing. So, it's not so much that downstream
        knows that its a static numel, as that you just plop a constant into the kernel.
        """
        for tree in self.range_trees:
            if tree.prefix != "r" or self.inside_reduction:
                simplified_tree_numel = V.graph.sizevars.simplify(tree.numel)
                if isinstance(simplified_tree_numel, (sympy.Integer, int)):
                    code.writeline(f"{tree.prefix}numel = {int(simplified_tree_numel)}")

            if tree.prefix == "r" and self.persistent_reduction:
                simplified_tree_numel = V.graph.sizevars.simplify(tree.numel)
                if isinstance(simplified_tree_numel, (sympy.Integer, int)):
                    val = int(simplified_tree_numel)
                else:
                    continue
                val = next_power_of_2(val)
                code.writeline(f"RBLOCK: tl.constexpr = {val}")

            if tree.prefix == "x" and self.no_x_dim:
                code.writeline("XBLOCK: tl.constexpr = 1")

    def triton_tensor_ndim(self):
        return sum(int(tree.tensor_dim is not None) for tree in self.range_trees)

    def indexing_size_str(self, i):
        sizes = ["None"] * self.triton_tensor_ndim()
        sizes[i] = ":"
        return f"[{', '.join(sizes)}]"

    def dense_size_list(self) -> List[str]:
        sizes = ["1"] * self.triton_tensor_ndim()
        for tree in self.range_trees:
            if tree.tensor_dim is None:
                continue

            if tree.prefix != "r" or self.inside_reduction:
                sizes[tree.tensor_dim] = f"{tree.prefix.upper()}BLOCK"
        return sizes

    def dense_size_str(self):
        sizes = self.dense_size_list()
        return f"[{', '.join(sizes)}]"

    def _get_grid_fn(self):
        return "grid"

    def add_numel_to_call_args_and_grid(self, name, call_args, grid):
        # TODO(jansel): if there are constants, we shouldn't bother passing them as args
        for tree in self.range_trees:
            if isinstance(tree.numel, (sympy.Integer, sympy.Symbol)):
                expr = tree.numel
            else:
                expr = V.graph.wrapper_code.generate_numel_expr(name, tree)

            if tree.prefix != "r" or self.inside_reduction:
                call_args.append(expr)
            if tree.grid_dim is not None:
                grid.append(expr)

    def get_call_args(self):
        _, call_args, _ = self.args.python_argdefs()
        # dynamo wraps unspec variable as 0d CPU tensor, need convert to scalar
        for i in range(len(call_args)):
            if V.graph.is_unspec_arg(call_args[i]):
                call_args[i] = call_args[i] + ".item()"

        return call_args

    def call_kernel(self, name: str, node: Optional[IRNode] = None):
        wrapper = V.graph.wrapper_code
        call_args = self.get_call_args()
        grid: List[Any] = []
        self.add_numel_to_call_args_and_grid(name, call_args, grid)
        current_device = V.graph.scheduler.current_device

        if self.args.workspace_arg is not None:
            ws = self.args.workspace_arg
            wrapper.generate_workspace_allocation(
                ws.nbytes, current_device, ws.zero_fill
            )

        grid = wrapper.generate_default_grid(name, grid)
        wrapper.generate_kernel_call(
            name,
            call_args,
            grid,
            current_device.index,
            cuda=True,
            triton=True,
            grid_fn=self._get_grid_fn(),
            triton_meta=self.triton_meta,
        )

        if self.args.workspace_arg is not None:
            wrapper.writeline(wrapper.make_free_by_names(["workspace"]))

    def codegen_nan_check(self):
        wrapper = V.graph.wrapper_code
        _, call_args, arg_types = self.args.python_argdefs()
        for arg, arg_type in zip(call_args, arg_types):
            if isinstance(arg_type, TensorArg):
                line = f"assert not {arg}.isnan().any().item()"
                wrapper.writeline(line)
                line = f"assert not {arg}.isinf().any().item()"
                wrapper.writeline(line)

    def warn_mix_layout(self, kernel_name):
        """
        Print message if the kernel have mixed layout inputs.
        Only care about 4D tensor for now.
        """
        if (
            len(self.args.input_buffers) == 1
            and len(self.args.output_buffers) == 1
            and len(self.args.inplace_buffers) == 0
        ):
            # even if input buffer and output buffer have different layout,
            # this can be a layout conversion kernel. No need to warn for
            # the mix layouts.
            return

        argdefs, call_args, signature = self.args.python_argdefs()
        uniform_stride_order = None
        for arg_name in call_args:
            buf = V.graph.get_buffer(arg_name)
            if buf and len(buf.layout.size) == 4:
                # ignore the tensor if only 1 dimension is non-zero
                if len([x for x in buf.layout.size if x == 1]) == 3:
                    continue
                stride_order = ir.get_stride_order(buf.layout.stride)
                if uniform_stride_order is None:
                    uniform_stride_order = stride_order
                elif uniform_stride_order != stride_order:
                    msg = yellow_text(
                        f"Expected stride order {uniform_stride_order}, but found stride order"
                        + f" {stride_order} for kernel {kernel_name}"
                    )
                    log.warning(msg)

                    stride_order_list = [
                        ir.get_stride_order(V.graph.get_buffer(name).layout.stride)
                        if V.graph.get_buffer(name)
                        else None
                        for name in call_args
                    ]
                    size_list = [
                        V.graph.get_buffer(name).layout.size
                        if V.graph.get_buffer(name)
                        else None
                        for name in call_args
                    ]
                    source_list = [
                        "GraphInput"
                        if name in V.graph.graph_inputs
                        else "IntermediateBuffer"
                        if name in V.graph.name_to_buffer
                        else None
                        for name in call_args
                    ]

                    msg = yellow_text(
                        f"  param names {argdefs}\n  buf names {call_args}\n  strides {stride_order_list}"
                        + f"\n  sizes {size_list}\n  sources {source_list}\n"
                    )
                    log.warning(msg)
                    return
        msg = green_text(
            f"All the inputs for the triton kernel {kernel_name} have uniform layout"
        )
        log.warning(msg)

    def create_cse_var(self, *args, **kwargs):
        return TritonCSEVariable(*args, **kwargs)


class TritonScheduling(BaseScheduling):
    def __init__(self, scheduler):
        self.scheduler = scheduler

    def group_fn(self, sizes):
        return tuple(V.graph.sizevars.simplify(sympy_product(s)) for s in sizes)

    def can_fuse(self, node1, node2):
        """
        Hook called by Scheduler to determine if the Triton backend
        can fuse node1 and node2.  These nodes might already be
        FusedSchedulerNodes.
        """
        if isinstance(node1, scheduler.ForeachKernelSchedulerNode) or isinstance(
            node2, scheduler.ForeachKernelSchedulerNode
        ):
            return scheduler.ForeachKernelSchedulerNode.can_fuse(node1, node2)

        _, (numel1, rnumel1) = node1.group
        _, (numel2, rnumel2) = node2.group
        why = WhyNoFuse(node1, node2)

        if node1.is_split_scan() and not node2.is_split_scan():
            if node2.is_reduction():
                why("Split scan cannot fuse with reductions")
        elif node2.is_split_scan() and not node1.is_split_scan():
            if node1.is_reduction():
                why("Split scan cannot fuse with reductions")

        if node1.is_reduction() and node2.is_reduction():
            reduction_can_fuse = numel1 == numel2 and rnumel1 == rnumel2
            if not reduction_can_fuse:
                why(
                    "numel/rnumel mismatch (reduce) (%s, %s), (%s, %s)",
                    numel1,
                    numel2,
                    rnumel1,
                    rnumel2,
                )
            return reduction_can_fuse

        if not node1.is_reduction() and not node2.is_reduction():
            if not (numel1 == numel2 and rnumel1 == rnumel2):
                why(
                    "numel/rnumel mismatch (non-reduce) (%s, %s), (%s, %s)",
                    numel1,
                    numel2,
                    rnumel1,
                    rnumel2,
                )
                return False

            if node1.is_template():
                # Only allow fusion for TritonTemplates for now.
                # Fusion for CUDATemplates are not supported.
                is_triton_template = isinstance(node1.node, TritonTemplateBuffer)
                if not is_triton_template:
                    why("node1 is not TritonTemplateBuffer")
                return is_triton_template

            # check for a bad combined tiling
            tiling1 = self.select_tiling(node1.get_nodes(), numel1, rnumel1)
            tiling2 = self.select_tiling(node2.get_nodes(), numel1, rnumel1)
            tiling3 = self.select_tiling(
                node1.get_nodes() + node2.get_nodes(), numel1, rnumel1
            )
            if config.triton.tiling_prevents_pointwise_fusion:
                cond = True
                if len(tiling1) > 2:
                    if len(tiling2) > 2:
                        cond = tiling1 == tiling2 == tiling3
                    else:
                        cond = tiling1 == tiling3
                elif len(tiling2) > 2:
                    cond = tiling2 == tiling3
                if not cond:
                    why(
                        "tiling mismatch (%s, %s, %s)",
                        tiling1,
                        tiling2,
                        tiling3,
                    )
                    return False

            return True

        if not node1.is_reduction() and node2.is_reduction():
            assert rnumel1 == 1 and rnumel2 != 1
            if numel1 == numel2 * rnumel2:
                if not all(
                    TritonKernel.is_compatible((numel2, rnumel2), n.get_ranges())
                    for n in node1.get_nodes()
                ):
                    why("nodes numel/rnumel incompatibility")
                    return False
                if (
                    config.triton.tiling_prevents_reduction_fusion
                    and not node1.is_template()
                ):
                    is_reduction_tiling_valid = self.select_tiling(
                        node1.get_nodes(), numel1
                    ) in (
                        (numel1, 1),
                        (numel2, rnumel2, 1),
                    )
                    if not is_reduction_tiling_valid:
                        why("invalid tiling for reduction")
                    return is_reduction_tiling_valid
                return True

            if numel1 != numel2:
                why("nodes numel incompatibility")
            return numel1 == numel2

        assert node1.is_reduction() and not node2.is_reduction()
        # swap args to hit the case above
        return self.can_fuse_horizontal(node2, node1)

    can_fuse_vertical = can_fuse
    can_fuse_horizontal = can_fuse

    def generate_node_schedule(self, nodes, numel, rnumel):
        node_schedule: List[Any] = []
        current_loop_writes: Set[str] = set()

        # Writes with a reduced shape, meaning they are only present once the
        # reduction loop has ended
        current_loop_reduced_writes = set()
        current_loop_has_writes = False
        done = set()

        def fits_in_main_body(n):
            _, (node_numel, node_rnumel) = n.group
            return (node_numel == numel and node_rnumel == rnumel) or (
                node_numel == numel * rnumel and node_rnumel == 1
            )

        def fits_outside_reduction(n):
            _, (node_numel, node_rnumel) = n.group
            return node_numel == numel and node_rnumel == 1 and rnumel != 1

        def schedule_node_in_loop(n):
            nonlocal current_loop_has_writes
            done.add(n)
            node_schedule.append(n)
            current_loop_has_writes = True
            # A scan is modelled as a reduction in the scheduler but has a
            # full sized output that can be used inside the loop body
            if (
                n.is_reduction()
                and isinstance(n, scheduler.SchedulerNode)
                and isinstance(n.node, ir.ComputedBuffer)
                and not isinstance(n.node.data, ir.Scan)
            ):
                current_loop_reduced_writes.add(n.get_name())

        @contextlib.contextmanager
        def end_current_reduction_loop():
            nonlocal current_loop_has_writes
            if current_loop_has_writes:
                # flush out any other runnable nodes to reduce number of loops
                for other_node in nodes[index + 1 :]:
                    if (
                        node not in done
                        and fits_in_main_body(other_node)
                        and not (current_loop_reduced_writes & other_node.ancestors)
                    ):
                        schedule_node_in_loop(node)

            if node_schedule and node_schedule[-1] is EnableReduction:
                node_schedule.pop()
            else:
                node_schedule.append(DisableReduction)
            yield
            node_schedule.append(EnableReduction)
            current_loop_reduced_writes.clear()
            current_loop_has_writes = False

        for index, node in enumerate(nodes):
            if node in done:
                continue
            done.add(node)

            def requires_closing_previous_reduction(node, node_schedule):
                if rnumel == 1:
                    return False
                if not current_loop_reduced_writes & node.ancestors:
                    return False
                assert node_schedule and not isinstance(
                    node_schedule[-1], (EnableReduction, DisableReduction)
                )
                return bool(current_loop_reduced_writes)

            if fits_in_main_body(node):
                if requires_closing_previous_reduction(node, node_schedule):
                    with end_current_reduction_loop():
                        pass  # need to start a new reduction loop

                schedule_node_in_loop(node)
            elif fits_outside_reduction(node):
                with end_current_reduction_loop():
                    node_schedule.append(node)
            else:
                raise NotImplementedError(
                    f"unexpected group: ({numel}, {rnumel}) != {node.group[1]}"
                )

        return node_schedule

    def codegen_nodes(self, nodes: List[scheduler.SchedulerNode]):
        """
        Given a set of pre-fused nodes, generate a Triton kernel.
        """
        _, (numel, rnumel) = max(nodes, key=lambda x: int(x.is_reduction())).group

        node_schedule = self.generate_node_schedule(nodes, numel, rnumel)
        buf_accesses = collections.defaultdict(list)
        for node in nodes:
            for access in node.read_writes.reads | node.read_writes.writes:
                buf_accesses[access.name].append(access)

        schedule_log.debug("Schedule:\n %s", node_schedule)

        return self.codegen_node_schedule(node_schedule, buf_accesses, numel, rnumel)

    @staticmethod
    def reduction_hint(node):
        assert node.is_reduction()
        if all(
            dep.is_contiguous()
            for dep in itertools.chain(node.read_writes.reads, node.read_writes.writes)
        ):
            return ReductionHint.INNER
        else:
            return node.node.data.reduction_hint

    @staticmethod
    def can_use_32bit_indexing(
        numel: sympy.Expr, buffers: Iterable[Union[ir.Buffer, ir.TensorBox]]
    ) -> bool:
        int_max = torch.iinfo(torch.int32).max
        size_hint = V.graph.sizevars.size_hint
        has_hint = V.graph.sizevars.shape_env.has_hint

        def within_32bit(e):
            # Allow for unhinted e as long as we can still statically prove
            # (e.g., via ValueRanges) that it is still in bounds
            if V.graph.sizevars.is_expr_static_and_true(e <= int_max):
                return True
            # Otherwise, the hint MUST exist and be in range
            return has_hint(e) and size_hint(e) <= int_max

        if not within_32bit(numel):
            return False

        # Any use of a MultiOutputLayout will create a buffer with a
        # Layout whose sizes are accounted for
        buf_sizes = [
            buf.get_layout().storage_size()
            for buf in buffers
            if not isinstance(buf.get_layout(), ir.MultiOutputLayout)
        ]

        if not all(within_32bit(size) for size in buf_sizes):
            return False

        # Only install guards for 32-bit indexing as there is no correctness
        # issue with using 64-bit for everything
        V.graph.sizevars.guard_leq(numel, int_max)  # type: ignore[arg-type]
        for size in buf_sizes:
            V.graph.sizevars.guard_leq(size, int_max)  # type: ignore[arg-type]
        return True

    @staticmethod
    def select_index_dtype(node_schedule, numel, reduction_numel):
        # Gather all used buffer names
        buffer_names = set()
        for node in node_schedule:
            if not isinstance(node, scheduler.BaseSchedulerNode):
                continue

            buffer_names.update(node.get_names())
            buffer_names.update(node.used_buffer_names())

        # Get buffers objects
        def _get_buffer(name: str) -> Union[ir.Buffer, ir.TensorBox]:
            if name in V.graph.name_to_buffer:
                return V.graph.name_to_buffer[name]
            elif name in V.graph.graph_inputs:
                return V.graph.graph_inputs[name]
            elif name in V.graph.constants:
                data = V.graph.constants[name]
                return ir.ConstantBuffer(
                    name,
                    ir.FixedLayout(
                        data.device, data.dtype, *V.graph.static_sizes_strides(data)
                    ),
                )
            raise RuntimeError(f"Failed to find buffer matching name {name}")

        buffers = [_get_buffer(name) for name in buffer_names]

        # In theory we can separately check xnumel and rnumel are <= int_max
        # but some indexers do use the full linear index so we need to be
        # conservative here.
        total_numel = numel * reduction_numel

        if TritonScheduling.can_use_32bit_indexing(total_numel, buffers):
            return "tl.int32"
        return "tl.int64"

    def get_kernel_args(self, node_schedule, numel, reduction_numel):
        reductions = list(
            filter(
                lambda n: n not in (EnableReduction, DisableReduction)
                and n.is_reduction(),
                node_schedule,
            )
        )
        if len(reductions) > 0:
            hints = [self.reduction_hint(n) for n in reductions]
            if hints.count(hints[0]) == len(hints):
                reduction_hint_val = hints[0]
            else:
                reduction_hint_val = ReductionHint.DEFAULT
        else:
            reduction_hint_val = ReductionHint.DEFAULT

        mutations = set()
        for node in node_schedule:
            if hasattr(node, "get_mutations"):
                mutations.update(node.get_mutations())

        index_dtype = self.select_index_dtype(node_schedule, numel, reduction_numel)

        return reduction_hint_val, mutations, index_dtype

    def codegen_comment(self, node_schedule):
        wrapper = V.graph.wrapper_code
        origins, detailed_origins = get_kernel_metadata(node_schedule, wrapper)
        if origins:
            wrapper.writeline(origins)

        if config.debug_fusion:
            from torch._inductor.scheduler import (
                BaseSchedulerNode,
                ForeachKernelSchedulerNode,
            )

            if not any(
                isinstance(n, ForeachKernelSchedulerNode) for n in node_schedule
            ):
                # We probably should look what are the nodes inside a foreach
                # schedule node
                node_names = [
                    n.get_name()
                    for n in node_schedule
                    if isinstance(n, BaseSchedulerNode)
                ]
                wrapper.writeline(
                    f"{wrapper.comment} Fused node name list: {', '.join(node_names)}"
                )

    def codegen_node_schedule(
        self, node_schedule, buf_accesses, numel, reduction_numel
    ):
        from torch._inductor.codegen.triton_split_scan import TritonSplitScanKernel

        tiled_groups = self.select_tiling(node_schedule, numel, reduction_numel)
        reduction_hint_val, mutations, index_dtype = self.get_kernel_args(
            node_schedule, numel, reduction_numel
        )

        is_split_scan = any(
            isinstance(node, BaseSchedulerNode) and node.is_split_scan()
            for node in node_schedule
        )
        kernel_type = TritonSplitScanKernel if is_split_scan else TritonKernel
        kernel_args = tiled_groups
        kernel_kwargs = {
            "reduction_hint": reduction_hint_val,
            "mutations": mutations,
            "index_dtype": index_dtype,
        }
        kernel = kernel_type(
            *kernel_args,
            **kernel_kwargs,
        )
        kernel.buf_accesses = buf_accesses

        self.codegen_node_schedule_with_kernel(node_schedule, kernel)

        with V.set_kernel_handler(kernel):
            src_code = kernel.codegen_kernel()

        kernel_name = self.define_kernel(src_code, node_schedule)
        log.debug("Generating kernel code with kernel_name: %s", kernel_name)
        kernel.kernel_name = kernel_name
        kernel.code_hash = code_hash(src_code)

        if kernel.persistent_reduction and config.triton.multi_kernel:
            kernel2 = TritonKernel(
                *kernel_args,
                **kernel_kwargs,
                disable_persistent_reduction=True,
            )
            self.codegen_node_schedule_with_kernel(node_schedule, kernel2)
            with V.set_kernel_handler(kernel2):
                src_code2 = kernel2.codegen_kernel()
            kernel_name2 = self.define_kernel(src_code2, node_schedule)
            kernel2.kernel_name = kernel_name2
            kernel2.code_hash = code_hash(src_code2)

            final_kernel = MultiKernel([kernel, kernel2])
        else:
            final_kernel = kernel  # type: ignore[assignment]

        with V.set_kernel_handler(final_kernel):
            for node in node_schedule:
                if node not in (EnableReduction, DisableReduction):
                    node.mark_run()

        self.codegen_comment(node_schedule)
        final_kernel.call_kernel(final_kernel.kernel_name)
        if config.nan_asserts:
            final_kernel.codegen_nan_check()
        if config.warn_mix_layout:
            final_kernel.warn_mix_layout(kernel_name)

        V.graph.removed_buffers |= final_kernel.removed_buffers
        V.graph.inplaced_to_remove |= final_kernel.inplaced_to_remove

        if (
            V.graph.wrapper_code.supports_intermediate_hooks
            and config.generate_intermediate_hooks
        ):
            # Not every node in the schedule will actually be live on output;
            # we can't check dead buffers.
            live_outs = kernel.args.live_output_buffers()
            for node in node_schedule:
                if not isinstance(node, scheduler.BaseSchedulerNode):
                    continue
                name = node.get_name()
                if name not in live_outs:
                    continue
                origin_node = node.node.get_origin_node()
                if origin_node is not None:
                    counters["inductor"]["intermediate_hooks"] += 1
                    V.graph.wrapper_code.writeline(
                        f"run_intermediate_hooks({origin_node.name!r}, {name})"
                    )

        self.scheduler.free_buffers()

    def codegen_node_schedule_with_kernel(self, node_schedule, kernel):
        def current_reduction_nodes(nodes):
            return itertools.takewhile(lambda n: n is not DisableReduction, nodes)

        with kernel:
            stack = contextlib.ExitStack()
            kernel.set_last_usage(current_reduction_nodes(node_schedule))

            for node in node_schedule:
                if node not in (EnableReduction, DisableReduction):
                    node.decide_inplace_update()
            for i, node in enumerate(node_schedule):
                if node is DisableReduction:
                    stack.enter_context(kernel.disable_reduction())
                elif node is EnableReduction:
                    stack.close()
                    kernel.set_last_usage(current_reduction_nodes(node_schedule[i:]))
                else:
                    # TODO - use split ranges ?
                    indexing_dtype_strength_reduction(node._body)
                    index_vars = kernel.split_and_set_ranges(node.get_ranges())
                    node.codegen(index_vars)

    def define_kernel(self, src_code, node_schedule):
        wrapper = V.graph.wrapper_code
        if src_code in wrapper.src_to_kernel:
            kernel_name = wrapper.src_to_kernel[src_code]
        else:
            fused_name = (
                get_fused_kernel_name(node_schedule, config.triton.descriptive_names)
                if config.triton.descriptive_names
                else ""
            )
            kernel_category = get_kernel_category_by_source_code(src_code)[:3]
            kernel_name = "_".join(
                ["triton", kernel_category, fused_name, wrapper.next_kernel_suffix()]
            )
            # use the original src_code as the key
            wrapper.src_to_kernel[src_code] = kernel_name
            subs_name = kernel_name if config.triton.unique_kernel_names else "triton_"

            # DESCRIPTIVE_NAME is used for profiling purposes; it shows the full kernel name
            # even when unique_kernel_names is turned off. Meanwhile, KERNEL_NAME is sometimes set
            # to "triton_" to maximize caching opportunities (when unique_kernel_names = False).
            src_code = src_code.replace(str(Placeholder.DESCRIPTIVE_NAME), kernel_name)
            src_code = src_code.replace(str(Placeholder.KERNEL_NAME), subs_name)

            # TODO(voz): Ostensibly, we should not need this. But there are cases where C++ codegen does
            # not use BracesBuffer, so we have no good indicator of a C++ buffer atm.
            src_code = src_code.replace("#pragma CMT", "#")

            basename, _, kernel_path = get_path(code_hash(src_code.strip()), "py")

            compile_wrapper = IndentedBuffer()
            compile_wrapper.writeline(f"async_compile.triton({subs_name!r}, '''")
            compile_wrapper.splice(src_code, strip=True)
            compile_wrapper.writeline(
                f"''', device_str='{V.graph.scheduler.current_device.type}')"
            )

            metadata_comment = f"# kernel path: {kernel_path}"
            origins, detailed_origins = get_kernel_metadata(node_schedule, wrapper)
            metadata_comment += "\n" + origins + "\n" + detailed_origins
            wrapper.define_kernel(
                kernel_name, compile_wrapper.getvalue(), metadata_comment
            )

            # log kernel metadata for offline analysis.
            # E.g. one can find all unaligned inner reduction and check if
            # padding helps with the perf kernel by kernel.
            if is_metric_table_enabled("kernel_metadata"):
                log_kernel_metadata(kernel_name, kernel_path, src_code)

        return kernel_name

    def codegen_template(
        self, template_node, epilogue_nodes, only_gen_src_code=False
    ) -> Optional[str]:
        """
        Codegen a triton template

        If `only_gen_src_code` the src code will be returned instead of codegen'd into the wrapper
        """
        _, (numel, rnumel) = template_node.group
        assert rnumel == 1
        kernel, render = template_node.node.make_kernel_render(template_node.node)
        with kernel:
            if not only_gen_src_code:
                for node in [template_node, *epilogue_nodes]:
                    node.mark_run()
            partial_code = render()
            for node in epilogue_nodes:
                node.codegen(kernel.split_and_set_ranges(node.get_ranges()))

        # finalize must be called after adding epilogue above
        with V.set_kernel_handler(kernel):
            # TODO: Maybe unify CUDATemplateKernel to also use PartialRender for flexible epilogue fusion.
            src_code = (
                partial_code
                if isinstance(partial_code, str)
                else partial_code.finalize()
            )
            node_schedule = [template_node, *epilogue_nodes]

            if config.benchmark_kernel:
                num_gb = kernel.estimate_kernel_num_bytes() / 1e9
                grid_args = V.graph.sizevars.size_hints(kernel.call_sizes)
                assert kernel.meta is not None, "meta is None"
                grid = kernel.grid_fn(*grid_args, kernel.meta)
                src_code = (
                    f"{kernel.imports_for_benchmark_kernel()}\n"
                    f"{src_code}\n"
                    f"{kernel.codegen_kernel_benchmark(num_gb, grid).getvalue()}"
                )

            if only_gen_src_code:
                return src_code

            kernel_name = self.define_kernel(src_code, node_schedule)

        self.codegen_comment(node_schedule)
        kernel.call_kernel(kernel_name, template_node.node)
        V.graph.removed_buffers |= kernel.removed_buffers
        V.graph.inplaced_to_remove |= kernel.inplaced_to_remove
        self.scheduler.free_buffers()
        return None

    def codegen_sync(self):
        V.graph.wrapper_code.writeline(V.graph.device_ops.synchronize())

    def codegen_foreach(self, foreach_node):
        from .triton_foreach import ForeachKernel

        for partitions_with_metadata in ForeachKernel.horizontal_partition(
            foreach_node.get_subkernel_nodes(), self
        ):
            kernel = ForeachKernel()
            for nodes, tiled_groups, numel, rnumel in partitions_with_metadata:
                node_schedule = self.generate_node_schedule(nodes, numel, rnumel)
                (
                    reduction_hint_val,
                    mutations,
                    index_dtype,
                ) = self.get_kernel_args(node_schedule, numel, rnumel)

                subkernel = kernel.create_sub_kernel(
                    *tiled_groups,
                    reduction_hint=reduction_hint_val,
                    mutations=mutations,
                    index_dtype=index_dtype,
                )

                self.codegen_node_schedule_with_kernel(
                    node_schedule,
                    subkernel,
                )

                with V.set_kernel_handler(subkernel):
                    for node in node_schedule:
                        if node not in (EnableReduction, DisableReduction):
                            node.mark_run()
                V.graph.removed_buffers |= subkernel.removed_buffers
                V.graph.inplaced_to_remove |= subkernel.inplaced_to_remove

            src_code = kernel.codegen_kernel()
            kernel_name = self.define_kernel(src_code, [foreach_node])
            self.codegen_comment([foreach_node])
            kernel.call_kernel(V.graph.wrapper_code, kernel_name)

        self.scheduler.free_buffers()

    @staticmethod
    @functools.lru_cache(32)
    def candidate_tilings(node):
        ranges, reduction_ranges = node.get_ranges()
        if len(ranges) <= 1:
            return ()

        rw = node.pointwise_read_writes()
        assert len(rw.range_vars) == len(ranges)

        # isinstance(dep, MemoryDep): this filters out StarDeps. StarDeps refer to reads
        # that need to access the entire tensor; they don't contribute read indexing
        # information (and practically, they don't have dep.index so they can't be used
        # for stride_hints below
        dep_sources = [rw.reads, rw.writes]
        assert all(
            isinstance(dep, (MemoryDep, StarDep))
            for dep in itertools.chain.from_iterable(dep_sources)
        )
        deps = [
            dep
            for dep in itertools.chain.from_iterable(dep_sources)
            if dep.name not in V.graph.removed_buffers and isinstance(dep, MemoryDep)
        ]
        write_names = {dep.name for dep in rw.writes}

        tilings: List[CandidateTiling] = []

        for dep in deps:
            strides = V.graph.sizevars.stride_hints(dep.index, rw.range_vars)
            assert len(strides) == len(ranges)
            try:
                split = strides.index(1) + 1
                if split == len(ranges):
                    continue
                if all(s == 0 for s in strides[split:]):
                    # if this is a broadcasted tensor and all dimensions after split are broadcast,
                    # this is not a real split
                    continue

            except ValueError:
                continue
            tiled_groups = (
                V.graph.sizevars.simplify(sympy_product(ranges[:split])),
                V.graph.sizevars.simplify(sympy_product(ranges[split:])),
            )
            # score by number of elements
            score = V.graph.sizevars.size_hint(
                sympy_product(
                    size for size, stride in zip(ranges, strides) if stride != 0
                )
            )
            if dep.name in write_names:
                # ngimel said contiguous writes is more important than reads
                score *= 2
            if CandidateTiling.is_good_size(tiled_groups[0]):
                score *= 2
            if CandidateTiling.is_good_size(tiled_groups[1]):
                score *= 2

            if (
                V.graph.sizevars.size_hint(
                    score - sympy_product(itertools.chain(ranges, reduction_ranges))
                )
                >= 0
            ):
                tilings.append(CandidateTiling(tiled_groups, score, dep.name))
        return tilings

    @classmethod
    def select_tiling(cls, node_schedule, numel, reduction_numel=sympy.Integer(1)):
        """
        Heuristics to decide how to tile kernels.
        Currently, we tile based on stride-1 dimensions.

        Returns:
            `(tile1, tile2, reduction_numel)` s.t. `tile1 * tile2 == numel`

        """
        if reduction_numel != 1 or config.triton.max_tiles <= 1:
            # TODO(jansel): should we tile reductions?
            # do perf hint here if stride-1 dim is not being reduced
            if perf_hint_log.level <= logging.WARNING:
                for node in EnableReduction.filter(node_schedule):
                    if len(cls.candidate_tilings(node)) > 0:
                        perf_hint_log.info("reduction over non-contiguous dims")
                        break
            return (numel, reduction_numel)

        seen_names = set()
        candidate_tiles: Counter[Any] = collections.Counter()
        for node in EnableReduction.filter(node_schedule):
            for tiling in cls.candidate_tilings(node):
                if tiling.name in seen_names:
                    continue
                seen_names.add(tiling.name)
                candidate_tiles[tiling.tiling] += tiling.score

        ranked_tilings = [tiling for tiling, score in candidate_tiles.most_common()]

        if config.triton.max_tiles >= 3:
            # Consider adding a third dimension of tiling, but only
            # when a1 is a multiple of b1; otherwise, you have a lot
            # of stragglers which is annoying to generate code for.
            #
            # NB: More than three max tiles is not enabled by default.

            # Add one 3D tiling choice
            for i in range(1, len(ranked_tilings)):
                a0, a1 = ranked_tilings[0]
                b0, b1 = ranked_tilings[i]
                if V.graph.sizevars.size_hint(a1 - b1) == 0:
                    continue
                if V.graph.sizevars.size_hint(a1 - b1) < 0:
                    # swap so a0 is bigger
                    a0, a1 = ranked_tilings[i]
                    b0, b1 = ranked_tilings[0]
                assert V.graph.sizevars.size_hint(a1 - b1) > 0
                if V.graph.sizevars.statically_known_multiple_of(a1, b1):
                    tiling = (a0, FloorDiv(a1, b1), b1)
                    ranked_tilings = [tiling] + ranked_tilings
                    break  # only 1 choice for now

        if len(ranked_tilings) > 1:
            perf_hint_log.info("possibly bad tiling: %s", ranked_tilings)

        for tiled_groups in ranked_tilings:
            new_groups = (*tiled_groups, reduction_numel)
            if all(
                TritonKernel.is_compatible(new_groups, node.get_ranges())
                for node in node_schedule
                if isinstance(node, scheduler.SchedulerNode)
            ):
                return new_groups

        return (numel, reduction_numel)

    def flush(self):
        pass

    def ready_to_flush(self) -> bool:
        return False

    @preserve_rng_state()
    def benchmark_fused_nodes(self, nodes):
        @dataclasses.dataclass
        class LastUsageHolder:
            n: Any
            last_usage: Any

            def __del__(self):
                self.n.last_usage = self.last_usage

        last_usage_holders = [LastUsageHolder(n, n.last_usage) for n in nodes]

        # empty last_usage. May cause more aggressive 'evict_last'. Should be fine.
        for n in nodes:
            n.last_usage = set()

        if not nodes[0].is_template():
            _, (numel, rnumel) = max(nodes, key=lambda x: int(x.is_reduction())).group
            node_schedule = self.generate_node_schedule(nodes, numel, rnumel)

            tiled_groups = self.select_tiling(node_schedule, numel, rnumel)
            reduction_hint_val, mutations, index_dtype = self.get_kernel_args(
                node_schedule, numel, rnumel
            )

            kernel = TritonKernel(
                *tiled_groups,
                reduction_hint=reduction_hint_val,
                mutations=mutations,
                index_dtype=index_dtype,
            )

            self.codegen_node_schedule_with_kernel(node_schedule, kernel)
            with config.patch("benchmark_kernel", True), V.set_kernel_handler(kernel):
                src_code = kernel.codegen_kernel()
        else:
            template_node = nodes[0]
            epilogue_nodes = nodes[1:]

            with config.patch("benchmark_kernel", True):
                src_code = self.codegen_template(
                    template_node, epilogue_nodes, only_gen_src_code=True
                )

        src_code = src_code.replace(str(Placeholder.KERNEL_NAME), "triton_")
        mod = PyCodeCache.load(src_code)

        def cache_file_path():
            assert mod.__file__ is not None
            return os.path.splitext(mod.__file__)[0] + ".kernel_perf"

        def load_cache():
            path = cache_file_path()
            if os.path.exists(path):
                with open(path) as fd:
                    return float(fd.read())
            return None

        def store_cache():
            path = cache_file_path()
            with open(path, "w") as fd:
                fd.write(str(ms))

        log.debug(
            "kernel src code for %s written to: %s",
            {n.get_name() for n in nodes},
            mod.__file__,
        )
        ms = load_cache()
        if ms is not None:
            return ms, mod.__file__

        args = mod.get_args()
        call = mod.call
        wrapped_jit_function = mod.triton_

        # call once to trigger the compilation
        call(wrapped_jit_function.clone_args(*args)[0])

        launchers = wrapped_jit_function.launchers
        assert len(launchers) == 1
        if launchers[0].n_spills > 0:
            # skip benchmarking the kernel if there are register spills
            ms = float("inf")
        else:
            # We have to clone the inplace updated arguments to avoid earlier calls
            # generating out of range indices for later calls.
            ms = do_bench(lambda: call(wrapped_jit_function.clone_args(*args)[0]))

            # overhead of cloning args gives bias for fusing the kernel
            # in the case of mutating/in-placeable second fusion
            # TODO - would be better as a hook in triton do_bench that reset
            # the input values between benchmarking
            ms = ms - do_bench(lambda: wrapped_jit_function.clone_args(*args))

        log.debug(
            "The fused kernel for %s took %.3f ms to run",
            {n.get_name() for n in nodes},
            ms,
        )
        store_cache()
        return ms, mod.__file__


@dataclasses.dataclass
class CandidateTiling:
    tiling: Tuple[sympy.Expr, sympy.Expr]
    score: int  # higher is better
    name: Optional[str] = None

    @staticmethod
    def is_good_size(s):
        """Somewhat arbitrary heuristic used to boost scores for some sizes"""
        s = V.graph.sizevars.size_hint(s)
        return s >= 32 and (s % 32 == 0)


class DisableReduction:
    """
    Marker to invoke `kernel.disable_reduction()`.  This closes a
    reduction loop and allows for pointwise ops to occur on the output
    of a reduction.
    """


class EnableReduction:
    """
    Marker to end a DisableReduction block.
    """

    @staticmethod
    def filter(node_schedule):
        """
        Get the nodes from node_schedule skipping those in a
        DisableReduction block.
        """
        disabled = False
        for node in node_schedule:
            if node in (EnableReduction, DisableReduction):
                # Don't tile stuff outside the main reduction loop
                disabled = node is DisableReduction
            elif disabled:
                pass
            else:
                yield node


class CantSplit(Exception):
    pass
