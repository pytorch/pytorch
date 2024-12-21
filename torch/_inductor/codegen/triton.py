# mypy: allow-untyped-defs
from __future__ import annotations

import collections
import contextlib
import dataclasses
import functools
import itertools
import logging
import os
import textwrap
from functools import lru_cache
from typing import (
    Any,
    Callable,
    cast,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    TYPE_CHECKING,
    Union,
)

import sympy
from sympy.printing.precedence import PRECEDENCE

import torch
import torch._logging
from torch._dynamo.device_interface import get_interface_for_device
from torch._dynamo.utils import identity, preserve_rng_state
from torch._prims_common import is_integer_dtype
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.functions import CeilDiv, FloorDiv, ModularIndexing
from torch.utils._triton import has_triton_package

from ...utils._sympy.symbol import free_symbol_is_type, prefix_str, symbol_is_type, SymT
from ...utils._sympy.value_ranges import ValueRanges
from .. import config, ir, metrics
from ..codecache import code_hash, get_path, PyCodeCache
from ..runtime.benchmarking import benchmarker
from ..runtime.hints import (
    AutotuneHint,
    DeviceProperties,
    TRITON_MAX_BLOCK,
    TRITON_MAX_RSPLIT,
)
from ..runtime.runtime_utils import get_max_y_grid, next_power_of_2
from ..runtime.triton_heuristics import (
    cooperative_reduction_grid,
    grid as default_grid_fn,
)
from ..scheduler import BaseSchedulerNode, FusedSchedulerNode, Scheduler, SchedulerNode
from ..utils import (
    cache_on_self,
    DelayReplaceLine,
    get_bounds_index_expr,
    get_fused_kernel_name,
    get_kernel_metadata,
    is_welford_reduction,
    Placeholder,
    prefix_is_reduction,
    sympy_dot,
    sympy_product,
    sympy_subs,
    triton_type,
    upcast_compute_type,
)
from ..virtualized import _ops as ops, OpsHandler, ReductionType, StoreMode, V
from ..wrapper_benchmark import get_kernel_category_by_source_code
from .block_analysis import BlockPatternMatcher
from .common import (
    BackendFeature,
    CSE,
    CSEVariable,
    DeferredLine,
    IndentedBuffer,
    OpOverrides,
    PythonPrinter,
    SizeArg,
    TensorArg,
    WorkspaceArg,
    WorkspaceZeroMode,
)
from .simd import (
    constant_repr,
    IterationRanges,
    IterationRangesEntry,
    IterationRangesRoot,
    pexpr,
    SIMDKernel,
    SIMDScheduling,
)
from .triton_utils import (
    config_of,
    should_unwrap_unspec_arg,
    signature_of,
    signature_to_meta,
)


if TYPE_CHECKING:
    from types import ModuleType

    from torch._inductor.dtype_propagation import DtypePropagationOpsHandler

    from ..ir import IRNode

log = logging.getLogger(__name__)
perf_hint_log = torch._logging.getArtifactLogger(__name__, "perf_hints")
schedule_log = torch._logging.getArtifactLogger(__name__, "schedule")
fusion_log = torch._logging.getArtifactLogger(__name__, "fusion")


class OpDtypeSupport:
    """
    Some Triton ops such as libdevice and tl.math only support float32 and float64.
    This class records which dtypes are supported by specific IR ops.
    """

    supported_dtypes: Dict[str, OrderedSet[torch.dtype]] = {}
    convert_outputs: Dict[str, bool] = {}

    @classmethod
    def register_upcast(cls, func: Callable[..., str], convert_output: bool):
        op_name = func.__name__
        cls.supported_dtypes[op_name] = OrderedSet([torch.float32, torch.float64])
        cls.convert_outputs[op_name] = convert_output


@lru_cache(None)
def gen_attr_descriptor_import():
    """
    import AttrsDescriptor if the triton version is new enough to have this
    class defined.
    """
    if not has_triton_package():
        return ""

    import triton.compiler.compiler

    # Note: this works because triton.compiler.compiler imports AttrsDescriptor from triton.backends.compiler
    # When support for the legacy AttrsDescriptor is removed then this import path should be changed.
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
        from torch._inductor.runtime import triton_helpers, triton_heuristics
        from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
        from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
        """
    )
    return imports.getvalue()


class TritonSymbols:
    """
    Stores sympy.Symbol instances and constants associated with triton codegen.
    """

    reduction_types = OrderedSet([SymT.R0_INDEX, SymT.R1_INDEX])
    block_types = OrderedSet([SymT.XBLOCK, SymT.YBLOCK, SymT.ZBLOCK, *reduction_types])

    block_offsets = {
        symt: sympy.Symbol(f"{prefix_str[symt]}offset", integer=True, nonnegative=True)
        for symt in block_types
    }

    block_sizes = {
        symt: sympy.Symbol(
            f"{prefix_str[symt].upper()}BLOCK", integer=True, positive=True
        )
        for symt in block_types
    }

    @classmethod
    def get_block_size(cls, tree: IterationRanges) -> sympy.Symbol:
        return cls.block_sizes[tree.symt]

    @classmethod
    def get_block_offset(cls, tree: IterationRanges) -> sympy.Symbol:
        return cls.block_offsets[tree.symt]


@dataclasses.dataclass
class IndexingOptions:
    index_str: str
    mask_vars: OrderedSet[str]
    mask_str: str
    expand_str: Optional[str]
    _has_rindex: bool
    index: sympy.Expr

    def has_mask(self):
        return bool(self.mask_vars)

    def has_indirect(self):
        return free_symbol_is_type(self.index, SymT.TMP)

    def has_rindex(self):
        return self._has_rindex

    def has_tmpmask(self):
        return "tmp" in self.mask_str

    def has_rmask(self):
        return any(str(mask).startswith("r") for mask in self.mask_vars)


@dataclasses.dataclass
class BlockPtrOptions:
    params: BlockParameters
    constant_offset: sympy.Expr
    order: List[int]
    mask_vars: OrderedSet[str]
    broadcast_shape: Sequence[sympy.Expr]
    broadcasting_dims: List[bool]
    final_shape: Sequence[sympy.Expr]
    _boundary_check: Optional[List[int]] = None

    @property
    def shape(self) -> List[sympy.Expr]:
        return self.params.shape

    @property
    def block_shape(self) -> List[sympy.Expr]:
        return self.params.block_shape

    @property
    def strides(self) -> List[sympy.Expr]:
        return self.params.strides

    @property
    def offsets(self) -> List[sympy.Expr]:
        return self.params.offsets

    def codegen_broadcast_and_reshape(
        self,
        value: str,
        initial_shape: Sequence[sympy.Expr],
        final_shape: Sequence[sympy.Expr],
        allow_implicit: bool,
    ) -> str:
        """
        Generate a broadcast and a reshape for the block pointer.
        This restores stride-0 dimensions which were removed from the block pointer.
        """

        # Reshape to add singletons.
        pre_broadcast_shape = [
            sympy.S.One if is_broadcasting else dim
            for dim, is_broadcasting in zip(
                self.broadcast_shape, self.broadcasting_dims
            )
        ]
        value = triton_reshape(value, initial_shape, pre_broadcast_shape)

        # Broadcast singletons.
        # For loads, we can often implicitly broadcast singleton dimensions.
        # We need an explicit broadcast for stores, or if the final reshape does more
        # than add singletons.
        sizevars = V.graph.sizevars
        require_broadcast = any(self.broadcasting_dims) and (
            len(pre_broadcast_shape) != len(final_shape)
            or any(
                not (
                    sizevars.statically_known_equals(pre_dim, 1)
                    or sizevars.statically_known_equals(pre_dim, post_dim)
                )
                for pre_dim, post_dim in zip(pre_broadcast_shape, final_shape)
            )
        )

        if not allow_implicit or require_broadcast:
            value = f"tl.broadcast_to({value}, {V.kernel.index_to_str(self.broadcast_shape)})"

        # Reshape to the final shape.
        value = triton_reshape(value, self.broadcast_shape, final_shape)

        return value

    @staticmethod
    def create(
        *,
        params: BlockParameters,
        constant_offset: sympy.Expr,
        range_trees: List[IterationRangesEntry],
        mask_vars: OrderedSet[str],
        get_max_block: Callable[[str], int],
    ) -> BlockPtrOptions:
        """Helper to create a  BlockPtrOptions instance"""

        sizevars = V.graph.sizevars

        def lookup_size(exprs: Iterable[sympy.Expr]) -> List[sympy.Expr]:
            return [sizevars.lookup_precomputed_size(expr) for expr in exprs]

        # Look up precomputed sizes
        params.shape = lookup_size(params.shape)
        params.strides = lookup_size(params.strides)

        # Strip out dimensions of stride 0.
        # These will be restored with tl.broadcast_to.
        broadcasting_dims = [
            sizevars.statically_known_equals(stride, 0) for stride in params.strides
        ]

        # Strip out dimensions of size 1.
        # These will be restored by tl.reshape.
        singleton_dims = [
            sizevars.statically_known_equals(dim, 1) for dim in params.block_shape
        ]
        if all(singleton_dims):
            # Handle a pure singletons, e.g. [1, 1]
            singleton_dims[-1] = False

        # Record the post-broadcast shape before broadcasting dims are removed.
        # The pre-broadcast shape is identical to this, except broadcasting dims are
        # replaced with 1.
        broadcast_shape = [
            dim
            for dim, is_singleton in zip(params.block_shape, singleton_dims)
            if not is_singleton
        ]

        # Combine all removable dims.
        removable_dims = [any(dims) for dims in zip(singleton_dims, broadcasting_dims)]

        def remove_dims(it):
            """Removes any broadcasting or singleton dims from a given sequence"""
            return [
                item
                for item, is_removable in zip(it, removable_dims)
                if not is_removable
            ]

        # Drop removable dimensions from the input.
        params = BlockParameters(
            **{key: remove_dims(val) for key, val in dataclasses.asdict(params).items()}
        )

        # Compute the final shape, adjusting for special kernel types.
        final_shape = [TritonSymbols.get_block_size(tree) for tree in range_trees]
        if V.kernel.no_x_dim:
            assert range_trees[0].prefix == "x"
            final_shape.pop(0)

        if (
            not V.kernel.inside_reduction
            and len(params.strides) == len(V.kernel.numels) - 1
            and V.kernel.features.is_reduction()
        ):
            # Need to expand rank by 1 to match rank when self.inside_reduction=True
            final_shape.append(sympy.S.One)

        result = BlockPtrOptions(
            params=params,
            constant_offset=V.graph.sizevars.lookup_precomputed_size(constant_offset),
            order=list(reversed(range(len(params.shape)))),
            mask_vars=mask_vars,
            final_shape=final_shape,
            broadcast_shape=broadcast_shape,
            broadcasting_dims=broadcasting_dims,
        )
        result.compute_boundary_check(get_max_block)
        return result

    def replace_roffset(self, expr: sympy.Expr, replacement: sympy.Expr) -> sympy.Expr:
        """
        Replaces instances of r0_offset with the new expression.
        """
        roffset = TritonSymbols.block_offsets[SymT.R0_INDEX]
        return sympy_subs(expr, {roffset: replacement})

    def format(self, name: str, roffset=True) -> str:
        """
        Codegen a call to tl.make_block_ptr()

        Args:
            name: variable name for pointer
            roffset: should rn_offset be included in offsets=..., for use with tl.advance()

        Returns:
            "tl.make_block_ptr(...)"
        """
        f = V.kernel.index_to_str
        offsets = [*self.offsets]
        if not roffset:
            offsets = [self.replace_roffset(offset, sympy.S.Zero) for offset in offsets]
        args = [
            (
                f"{name} + ({f(self.constant_offset)})"
                if self.constant_offset != 0
                else name
            ),
            f"shape={f(self.shape)}",
            f"strides={f(self.strides)}",
            f"block_shape={f(self.block_shape)}",
            f"order={f(self.order)}",
            f"offsets={f(offsets)}",
        ]
        return f"tl.make_block_ptr({', '.join(args)})"

    def compute_boundary_check(self, get_max_block: Callable[[str], int]) -> None:
        """List of indices to pass to tl.load(boundary_check=...)"""
        sizevars = V.graph.sizevars

        # Substitute maximum block sizes in shape expressions.
        # This works in multiple_of checks because block sizes are powers of 2.
        block_to_max: Dict[sympy.Expr, Any] = {
            block_size: get_max_block(prefix_str[symt])
            for symt, block_size in TritonSymbols.block_sizes.items()
        }

        self._boundary_check = [
            idx
            for idx in range(len(self.shape))
            if (
                not sizevars.statically_known_equals(self.strides[idx], sympy.S.Zero)
                and not sizevars.statically_known_multiple_of(
                    self.shape[idx], self.block_shape[idx]
                )
                and not sizevars.statically_known_multiple_of(
                    self.shape[idx], sympy_subs(self.block_shape[idx], block_to_max)
                )
                and not (
                    V.kernel.no_x_dim
                    and self.block_shape[idx] == TritonSymbols.block_sizes[SymT.XBLOCK]
                )
            )
        ]

    def boundary_check(self):
        assert self._boundary_check is not None
        return self._boundary_check

    def advance_roffset(self):
        """
        Codegen string to pass to tl.advance(name, ...).

        Advance is the difference between offsets in each loop iteration.
        To compute it, we replace roffset with multiples of R0_BLOCK.
        Since we expect roffset to vary in range(0, rnumel, R0_BLOCK), the first
        iteration has roffset=0, while the second has roffset=R0_BLOCK.
        """
        rblock = TritonSymbols.block_sizes[SymT.R0_INDEX]
        advance = [
            (
                self.replace_roffset(offset, rblock)
                - self.replace_roffset(offset, sympy.S.Zero)
            )
            for offset in self.offsets
        ]
        return V.kernel.index_to_str(advance)

    def has_indirect(self):
        return False  # block_ptr can't do indirect indexing

    def has_rindex(self) -> bool:
        return any(
            free_symbol_is_type(expr, TritonSymbols.reduction_types)
            for expr in self.block_shape
        )

    def has_rmask(self):
        return self.has_rindex()

    def has_tmpmask(self):
        return False  # block_ptr can't do indirect indexing

    def has_mask(self):
        return bool(self.boundary_check())


def triton_reshape(
    value: str, old_shape: Sequence[sympy.Expr], new_shape: Sequence[sympy.Expr]
):
    """Workaround https://github.com/openai/triton/issues/2836"""
    assert isinstance(old_shape, list) and isinstance(new_shape, list)

    old_shape_str = [V.kernel.index_to_str(shape) for shape in old_shape]
    new_shape_str = [V.kernel.index_to_str(shape) for shape in new_shape]

    if old_shape_str == new_shape_str:
        return value
    if [s for s in new_shape_str if s != "1"] != old_shape_str:
        return f"tl.reshape({value}, [{', '.join(new_shape_str)}])"
    # rewrite to [:, None] syntax, which is less buggy
    idx = 0
    expand = []
    for size in new_shape_str:
        if idx < len(old_shape_str) and size == old_shape_str[idx]:
            expand.append(":")
            idx += 1
        else:
            assert size == "1"
            expand.append("None")
    assert idx == len(old_shape_str)
    return f"{value}[{', '.join(expand)}]"


# NB: Inheriting from PythonPrinter is somewhat dangerous, because there are a
# number of operators which Triton "implements", but in a way that is
# inconsistent with Python semantics (and consistent with C semantics).  We
# must override all of these, or it is potential silent correctness problem
class TritonPrinter(PythonPrinter):
    def _print_TruncToInt(self, expr):
        assert len(expr.args) == 1
        return (
            f"libdevice.trunc({self._print(expr.args[0])}).to({V.kernel.index_dtype})"
        )

    def _print_Float(self, expr):
        if config.is_fbcode() and torch.version.hip:
            ret = f"{expr}"
        else:
            ret = f"tl.full([], {expr}, tl.float64)"
        return ret

    def _print_ToFloat(self, expr):
        assert len(expr.args) == 1
        s = self.parenthesize(expr.args[0], PRECEDENCE["Atom"] - 0.5)
        return f"{s}.to(tl.float64)"

    def _print_PythonMod(self, expr):
        quot, div = expr.args
        if quot.is_nonnegative and div.is_nonnegative:
            return self.stringify(expr.args, " % ", PRECEDENCE["Atom"] - 0.5)
        quot_s = self._print(quot)
        div_s = self._print(div)
        return f"triton_helpers.remainder_integer({quot_s}, {div_s})"

    def _print_FloorDiv(self, expr):
        assert expr.is_integer
        quot, div = expr.args
        if quot.is_nonnegative and div.is_nonnegative:
            return self.stringify(expr.args, " // ", PRECEDENCE["Atom"] - 0.5)
        quot_s = self._print(quot)
        div_s = self._print(div)
        return f"triton_helpers.div_floor_integer({quot_s},  {div_s})"

    # TODO: This is wrong, when lhs, rhs > 2**53, Python does a higher
    # precision algorithm, which we would need to replicate here
    def _print_IntTrueDiv(self, expr):
        return self.stringify(expr.args, " / ", PRECEDENCE["Atom"] - 0.5)

    # NB: sympy.floor/ceiling produce integers, so we have to do the
    # conversion to index dtype
    def _print_floor(self, expr):
        assert len(expr.args) == 1
        return (
            f"libdevice.floor({self._print(expr.args[0])}).to({V.kernel.index_dtype})"
        )

    def _print_FloorToInt(self, expr):
        assert len(expr.args) == 1
        return (
            f"libdevice.floor({self._print(expr.args[0])}).to({V.kernel.index_dtype})"
        )

    def _print_ceiling(self, expr):
        assert len(expr.args) == 1
        return f"libdevice.ceil({self._print(expr.args[0])}).to({V.kernel.index_dtype})"

    def _print_CeilToInt(self, expr):
        assert len(expr.args) == 1
        return f"libdevice.ceil({self._print(expr.args[0])}).to({V.kernel.index_dtype})"

    def _helper_sqrt(self, expr):
        return f"libdevice.sqrt(({self._print(expr)}).to(tl.float32))"

    def _print_FloatPow(self, expr):
        return (
            f"libdevice.pow({self._print(expr.args[0])}, {self._print(expr.args[1])})"
        )

    _print_PowByNatural = _print_FloatPow

    def _print_Where(self, expr):
        c = self.doprint(expr.args[0])
        p = self.doprint(expr.args[1])
        q = self.doprint(expr.args[2])
        return f"tl.where({c}, {p}, {q})"

    def _print_min_max_helper(self, expr: sympy.Expr, cmp: str) -> str:
        """
        Helper for max/min code genereration.
        cmp: > or <
        """
        if len(expr.args) == 1:
            return self._print(expr.args[0])

        mid = len(expr.args) // 2
        cls = type(expr)
        a = self._print(cls(*expr.args[:mid]))
        b = self._print(cls(*expr.args[mid:]))

        # Use a macro so we can propagate constexprs.
        # https://github.com/triton-lang/triton/issues/3815
        a, b = tuple(f"({x})" for x in (a, b))
        assert cmp in (">", "<"), f"Unexpected comparator: '{cmp}'"
        return f"({a} * ({a} {cmp}= {b}) + {b} * ({b} {cmp} {a}))"

    def _print_Min(self, expr):
        return self._print_min_max_helper(expr, "<")

    def _print_Max(self, expr):
        return self._print_min_max_helper(expr, ">")

    def _print_Abs(self, expr):
        assert len(expr.args) == 1
        return f"tl_math.abs({self._print(expr.args[0])})"

    def _print_OpaqueUnaryFn_cos(self, expr):
        assert len(expr.args) == 1
        return f"libdevice.cos(({self._print(expr.args[0])}).to(tl.float32))"

    def _print_OpaqueUnaryFn_cosh(self, expr):
        assert len(expr.args) == 1
        return f"libdevice.cosh(({self._print(expr.args[0])}).to(tl.float32))"

    def _print_OpaqueUnaryFn_acos(self, expr):
        assert len(expr.args) == 1
        return f"libdevice.acos(({self._print(expr.args[0])}).to(tl.float32))"

    def _print_OpaqueUnaryFn_sin(self, expr):
        assert len(expr.args) == 1
        return f"libdevice.sin(({self._print(expr.args[0])}).to(tl.float32))"

    def _print_OpaqueUnaryFn_sinh(self, expr):
        assert len(expr.args) == 1
        return f"libdevice.sinh(({self._print(expr.args[0])}).to(tl.float32))"

    def _print_OpaqueUnaryFn_asin(self, expr):
        assert len(expr.args) == 1
        return f"libdevice.asin(({self._print(expr.args[0])}).to(tl.float32))"

    def _print_OpaqueUnaryFn_tan(self, expr):
        assert len(expr.args) == 1
        return f"libdevice.tan(({self._print(expr.args[0])}).to(tl.float32))"

    def _print_OpaqueUnaryFn_tanh(self, expr):
        assert len(expr.args) == 1
        return f"libdevice.tanh(({self._print(expr.args[0])}).to(tl.float32))"

    def _print_OpaqueUnaryFn_atan(self, expr):
        assert len(expr.args) == 1
        return f"libdevice.atan(({self._print(expr.args[0])}).to(tl.float32))"

    def _print_RoundToInt(self, expr):
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

        number_str = self.parenthesize(number, PRECEDENCE["Mul"])
        return f"libdevice.nearbyint(1e{ndigits} * {number_str}) * 1e{-ndigits}"


texpr = TritonPrinter().doprint


def triton_compute_type(dtype: torch.dtype) -> str:
    """Convert torch.dtype to triton type and upcast [b]float16 to float32"""
    return triton_type(upcast_compute_type(dtype))


def _get_primitive_bitwidth(dtype: torch.dtype) -> int:
    """Number of bits of triton_compute_type()"""
    dtype = upcast_compute_type(dtype)
    itemsize = getattr(dtype, "itemsize", None)
    if itemsize:
        return itemsize * 8
    else:
        return -1


def triton_store_type(dtype: torch.dtype) -> str:
    """Convert torch.dtype to triton type, with fix for storing tl.bool"""
    if dtype == torch.bool:
        dtype = torch.int8
    return triton_type(dtype)


def upcast_acc_dtype(dtype: torch.dtype) -> torch.dtype:
    """Implicit upcasts used for Triton reduction types"""
    if is_integer_dtype(dtype) and dtype.is_signed and dtype.itemsize <= 4:
        return torch.int32
    return upcast_compute_type(dtype)


def triton_acc_type(dtype: torch.dtype) -> str:
    """Convert torch.dtype to triton type, with reduction upcasts"""
    return triton_compute_type(upcast_acc_dtype(dtype))


class TritonCSEVariable(CSEVariable):
    def __init__(self, name, bounds: ValueRanges[Any], dtype: torch.dtype) -> None:
        super().__init__(name, bounds, dtype)
        # We'll use this to track which masks the variable needs when used for indirect indexing
        self.mask_vars = OrderedSet[str]()
        assert dtype is not None, "TritonCSEVariable must have dtype"

    def update_on_args(self, name, args, kwargs):
        for arg in args:
            if isinstance(arg, TritonCSEVariable):
                self.mask_vars.update(arg.mask_vars)
            elif isinstance(arg, sympy.Symbol):
                # most of the time index vars don't need masks associated with them
                # however, when index vars are used to compute indices for indirect reads
                # those reads should subsequently be masked,
                for symt in TritonSymbols.block_types:
                    if symbol_is_type(arg, symt):
                        self.mask_vars.update([f"{prefix_str[symt]}mask"])
                        break


def get_dtype_handler() -> DtypePropagationOpsHandler:
    from torch._inductor.dtype_propagation import DtypePropagationOpsHandler

    return DtypePropagationOpsHandler()


def maybe_upcast_float32(convert_output: bool = True):
    """
    Codegen helper to upcast arguments to float32, depending on the config and dtype.
    This decorates tl.math/libdevice codegen functions.
    """

    def needs_upcast(var) -> bool:
        return (
            not config.triton.codegen_upcast_to_fp32
            and isinstance(var, CSEVariable)
            and var.dtype in (torch.float16, torch.bfloat16)
        )

    def maybe_upcast_arg(var) -> str:
        upcast_string = ".to(tl.float32)" if needs_upcast(var) else ""
        return f"{var}{upcast_string}"

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        # Record that this function only supports float32 and float64.
        OpDtypeSupport.register_upcast(func, convert_output)

        def wrapped(*args, **kwargs) -> str:
            # Optionally upcast args to float32.
            upcast_args = [maybe_upcast_arg(arg) for arg in args]
            upcast_kwargs = {key: maybe_upcast_arg(val) for key, val in kwargs.items()}

            # Call the decorated function, optionally downcasting the result.
            result = func(*upcast_args, **upcast_kwargs)
            any_needs_upcast = convert_output and any(
                needs_upcast(var) for var in itertools.chain(args, kwargs.values())
            )
            result_dtype = (
                None
                if not any_needs_upcast
                else getattr(get_dtype_handler(), func.__name__)(*args, **kwargs)
            )
            needs_downcast = result_dtype not in (torch.float32, None)
            downcast_string = (
                f".to({triton_type(result_dtype)})"
                if needs_downcast and result_dtype is not None
                else ""
            )
            return f"{result}{downcast_string}"

        return wrapped

    return decorator


class TritonOverrides(OpOverrides):
    """Map element-wise ops to Triton"""

    @staticmethod
    def to_dtype(
        x,
        dtype: torch.dtype,
        src_dtype: Optional[torch.dtype] = None,
        use_compute_types=True,
    ):
        def _get_min_elements_per_thread(
            src_dtype: torch.dtype, dst_dtype: torch.dtype
        ) -> int:
            if src_dtype == dst_dtype:
                # No data type conversion is needed. No requirements on min_elem_per_thread.
                return 0

            # fp8 data type conversions has min_elem_per_thread requirements.
            # Refer to Triton implementations here:
            # https://github.com/openai/triton/blob/10f59d8ce04052521c1bc0cb3a3f8b98918fc7e3/lib/Conversion/TritonGPUToLLVM/ElementwiseOpToLLVM.cpp#L10.
            fp8_dtypes = (
                torch.float8_e4m3fn,
                torch.float8_e5m2,
            )
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

        if use_compute_types:
            out_dtype = triton_compute_type(dtype)
        else:
            out_dtype = triton_store_type(dtype)

        return f"{x}.to({out_dtype})"

    @staticmethod
    def to_dtype_bitcast(x, dtype: torch.dtype, src_dtype: torch.dtype):
        triton_dtype = triton_compute_type(dtype)
        # We may promote float16 or bfloat16 to float32 and cause the
        # bitwidth of dtype to be different from the input tensor (i.e. float32).
        # In such as case, we will have to convert the input tensor to
        # its src_type, perform bitcast, and then convert the bit-casted
        # tensor back to float to ensure we use values with the right precision.
        if (
            src_dtype in (torch.float16, torch.bfloat16)
            and config.triton.codegen_upcast_to_fp32
        ):
            triton_src_dtype = str(src_dtype).split(".")[-1]
            cast_x = f"{x}.to(tl.{triton_src_dtype})"
            if dtype in (torch.float16, torch.bfloat16):
                triton_type_name = str(dtype).split(".")[-1]
                triton_dtype = f"tl.{triton_type_name}"
            cast_x = f"{cast_x}.to({triton_dtype}, bitcast=True)"
            if dtype in (torch.float16, torch.bfloat16):
                return f"{cast_x}.to(tl.float32)"
            return cast_x
        else:
            src_dtype_bitwidth = _get_primitive_bitwidth(src_dtype)
            target_dtype_bitwidth = _get_primitive_bitwidth(dtype)
            bitcast = "True" if src_dtype_bitwidth == target_dtype_bitwidth else "False"
            return f"{x}.to({triton_dtype}, bitcast={bitcast})"

    @staticmethod
    def _shaped_constant(value, dtype, shape):
        type_ = torch._prims_common.dtype_to_type(dtype)
        triton_val = constant_repr(type_(value))
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
    @maybe_upcast_float32()
    def abs(x):
        return f"tl_math.abs({x})"

    # TODO - register these ops as having divergent dtype
    # output if doing graph pass to remove consecutive casts

    @staticmethod
    def truediv(x, y):
        out = f"({x} / {y})"
        out_dtype = get_dtype_handler().truediv(x, y)
        if out_dtype in (torch.float16, torch.float32):
            out = f"{out}.to({triton_type(out_dtype)})"
        return out

    @staticmethod
    def mod(x, y):
        out = f"({x} % {y})"
        out_dtype = get_dtype_handler().mod(x, y)
        if out_dtype in (torch.float16, torch.float32):
            out = f"{out}.to({triton_type(out_dtype)})"
        return out

    @staticmethod
    @maybe_upcast_float32()
    def libdevice_abs(x):
        return f"libdevice.abs({x})"

    @staticmethod
    @maybe_upcast_float32()
    def exp(x):
        return f"tl_math.exp({x})"

    @staticmethod
    @maybe_upcast_float32()
    def libdevice_exp(x):
        return f"libdevice.exp({x})"

    @staticmethod
    @maybe_upcast_float32()
    def exp2(x):
        return f"libdevice.exp2({x})"

    @staticmethod
    @maybe_upcast_float32()
    def expm1(x):
        return f"libdevice.expm1({x})"

    @staticmethod
    @maybe_upcast_float32()
    def sqrt(x):
        return f"libdevice.sqrt({x})"

    @staticmethod
    @maybe_upcast_float32()
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
            return ops.maximum(ops.constant(0, torch.int32), x)
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
    def inline_asm_elementwise(
        *inputs, asm, constraints=None, dtype=torch.float32, is_pure=True, pack=1
    ):
        triton_type = triton_compute_type(dtype)
        input_refs = ", ".join([str(i) for i in inputs])
        if constraints is None:
            constraints = ", ".join(["=r"] + ["r" for _ in inputs])
        return f"tl.inline_asm_elementwise('{asm}', '{constraints}', [{input_refs}], dtype={triton_type}, is_pure={is_pure}, pack={pack})"  # noqa: B950

    @staticmethod
    @maybe_upcast_float32()
    def cos(x):
        return f"tl_math.cos({x})"

    @staticmethod
    @maybe_upcast_float32()
    def libdevice_cos(x):
        return f"libdevice.cos({x})"

    @staticmethod
    @maybe_upcast_float32()
    def sin(x):
        return f"tl_math.sin({x})"

    @staticmethod
    @maybe_upcast_float32()
    def libdevice_sin(x):
        return f"libdevice.sin({x})"

    @classmethod
    def index_expr(cls, expr, dtype):
        raise NotImplementedError("ops.index_expr not implemented outside a kernel")

    @staticmethod
    def masked(mask, body, other):
        raise NotImplementedError("ops.masked not implemented outside a kernel")

    @staticmethod
    @maybe_upcast_float32()
    def lgamma(x):
        return f"libdevice.lgamma({x})"

    @staticmethod
    @maybe_upcast_float32()
    def erf(x):
        return f"libdevice.erf({x})"

    @staticmethod
    @maybe_upcast_float32()
    def cosh(x):
        return f"libdevice.cosh({x})"

    @staticmethod
    @maybe_upcast_float32()
    def sinh(x):
        return f"libdevice.sinh({x})"

    @staticmethod
    @maybe_upcast_float32()
    def acos(x):
        return f"libdevice.acos({x})"

    @staticmethod
    @maybe_upcast_float32()
    def acosh(x):
        return f"libdevice.acosh({x})"

    @staticmethod
    @maybe_upcast_float32()
    def asin(x):
        return f"libdevice.asin({x})"

    @staticmethod
    @maybe_upcast_float32()
    def asinh(x):
        return f"libdevice.asinh({x})"

    @staticmethod
    @maybe_upcast_float32()
    def atan2(x, y):
        return f"libdevice.atan2({x}, {y})"

    @staticmethod
    @maybe_upcast_float32()
    def atan(x):
        return f"libdevice.atan({x})"

    @staticmethod
    @maybe_upcast_float32()
    def atanh(x):
        return f"libdevice.atanh({x})"

    @staticmethod
    @maybe_upcast_float32()
    def copysign(x, y):
        return f"libdevice.copysign({x}, {y})"

    @staticmethod
    @maybe_upcast_float32()
    def erfc(x):
        return f"libdevice.erfc({x})"

    @staticmethod
    @maybe_upcast_float32()
    def erfinv(x):
        return f"libdevice.erfinv({x})"

    @staticmethod
    @maybe_upcast_float32()
    def hypot(x, y):
        return f"libdevice.hypot({x}, {y})"

    @staticmethod
    @maybe_upcast_float32()
    def log10(x):
        return f"libdevice.log10({x})"

    @staticmethod
    @maybe_upcast_float32()
    def log2(x):
        return f"libdevice.log2({x})"

    @staticmethod
    @maybe_upcast_float32()
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
    @maybe_upcast_float32()
    def rsqrt(x):
        return f"libdevice.rsqrt({x})"

    @staticmethod
    @maybe_upcast_float32()
    def log1p(x):
        return f"libdevice.log1p({x})"

    @staticmethod
    @maybe_upcast_float32()
    def tan(x):
        return f"libdevice.tan({x})"

    @staticmethod
    @maybe_upcast_float32()
    def tanh(x):
        return f"libdevice.tanh({x})"

    @staticmethod
    @maybe_upcast_float32()
    def sigmoid(x):
        return f"tl.sigmoid({x})"

    @staticmethod
    def signbit(x):
        # XX: This is wrong for the value -0.0 in floating point
        return (
            f"(libdevice.signbit({x}) != 0) if ({x}).dtype is tl.float32 else {x} < 0"
        )

    @staticmethod
    @maybe_upcast_float32()
    def fmod(a, b):
        return f"libdevice.fmod({a}, {b})"

    @staticmethod
    @maybe_upcast_float32()
    def pow(a, b):
        return f"libdevice.pow({a}, {b})"

    @staticmethod
    @maybe_upcast_float32()
    def log(x):
        return f"tl_math.log({x})"

    @staticmethod
    @maybe_upcast_float32()
    def libdevice_log(x):
        return f"libdevice.log({x})"

    @staticmethod
    @maybe_upcast_float32(convert_output=False)
    def isinf(x):
        return f"libdevice.isinf({x}).to(tl.int1)"

    @staticmethod
    @maybe_upcast_float32(convert_output=False)
    def isnan(x):
        return f"libdevice.isnan({x}).to(tl.int1)"

    @staticmethod
    @maybe_upcast_float32()
    def round(x):
        return f"libdevice.nearbyint({x})"

    @staticmethod
    @maybe_upcast_float32()
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
        z = ops.constant(0, torch.int32)
        left = ops.to_dtype((ops.lt(z, x)), torch.int8)
        right = ops.to_dtype((ops.lt(x, z)), torch.int8)
        sub = ops.sub(left, right)
        return f"{sub}.to({x}.dtype)"

    @staticmethod
    @maybe_upcast_float32()
    def trunc(x):
        return f"libdevice.trunc({x})"

    @staticmethod
    def truncdiv(a, b):
        # See the comment in lowering.div_mode. a and b are integer type.
        # Notice that // in triton behaves as truncdiv instead of floordiv
        return f"{a} // {b}"

    @staticmethod
    @maybe_upcast_float32()
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

        # Our sympy expr printing casts to the current kernel index dtype.
        # we only respect non int32-int64 dtypes and otherwise use current kernel indexing dtype
        index_dtype = torch.int32 if V.kernel.index_dtype == "tl.int32" else torch.int64
        dtype = dtype if dtype not in (torch.int32, torch.int64) else index_dtype
        var = V.kernel.cse.generate(
            V.kernel.compute,
            indexing.index_str,
            bounds=get_bounds_index_expr(expr),
            dtype=dtype,
        )

        if dtype not in (torch.int32, torch.int64):
            var = V.kernel.cse.generate(
                V.kernel.compute,
                cls.to_dtype(var, dtype),
                dtype=upcast_compute_type(dtype),
            )
        else:
            # TODO: we are not always consistent in enforcing that the output of the index expr printing
            # results in the indexing dtype. So if we detect that we have an input which might type promote
            # to a dtype other than indexing dtype, add a cast.
            # Trying to avoid
            dtype = index_dtype
            for index_var in expr.free_symbols:
                if symbol_is_type(index_var, SymT.TMP):
                    dtype = torch.promote_types(
                        dtype, V.kernel.cse.varname_map[index_var.name].dtype
                    )

            if dtype != index_dtype:
                var = V.kernel.cse.generate(
                    V.kernel.compute,
                    cls.to_dtype(var, index_dtype),
                    dtype=index_dtype,
                )

        var.mask_vars = indexing.mask_vars
        return var

    @staticmethod
    def masked(mask, body, other):
        if mask is not None and torch.version.hip is not None:
            mask = V.kernel.cse.generate(
                V.kernel.compute,
                f"{mask}.to(tl.int1)",
                dtype=torch.bool,
            )

        nodes = body.graph.find_nodes(op="output")
        assert nodes, "graph for body does not contain an output"

        need_where = False
        for node in nodes:
            for arg in node.args:
                if arg.target != "load" or should_unwrap_unspec_arg(arg.args[0]):
                    need_where = True

        value = None if need_where else other

        with V.kernel.mask_loads(mask, value=value) as new_mask:
            result = body()

        if need_where:
            # Remove once CSEVariables track the dtype
            if result.bounds.is_bool:
                other = bool(other)
            # Take dtype from result to prevent accidental promotion
            other = V.kernel.cse.generate(
                V.kernel.compute,
                f"tl.full({result}.shape, {constant_repr(other)}, {result}.dtype)",
                bounds=ValueRanges.wrap(other),
                dtype=result.dtype,
            )
            ret = ops.where(new_mask, result, other)
        else:
            ret = result

        ret.mask_vars.discard(new_mask)
        return ret

    @staticmethod
    def load_seed(name, offset):
        var = V.kernel.args.input(name)
        return (
            f"tl.load({var} + {V.kernel.args.seed_offset('load_seed_offset', offset)})"
        )

    @staticmethod
    def frexp(x):
        cache_key = f"frexp({x})"
        if cse_val := V.kernel.cse.try_get(cache_key):
            return cse_val

        mantissa = V.kernel.cse.newvar(dtype=x.dtype)
        exponent = V.kernel.cse.newvar(dtype=torch.int32)
        V.kernel.compute.writeline(
            f"{mantissa}, {exponent} = triton_helpers.frexp({x})"
        )
        V.kernel.cse.put(cache_key, (mantissa, exponent))
        return (mantissa, exponent)


# Use mypy to check protocol implemented correctly
def _typecheck_TritonKernelOverrides(h: TritonKernelOverrides) -> OpsHandler[str]:
    return h


class HelperFunctions:
    """An ordered set of helper functions."""

    _templates_seen: Dict[str, str]  # Template code to function name
    finalized_helpers: List[str]

    def __init__(self) -> None:
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


@dataclasses.dataclass
class BlockParameters:
    """
    Class representing ND block dimensions, for block pointer analysis.
    """

    shape: List[sympy.Expr] = dataclasses.field(default_factory=list)
    block_shape: List[sympy.Expr] = dataclasses.field(default_factory=list)
    strides: List[sympy.Expr] = dataclasses.field(default_factory=list)
    offsets: List[sympy.Expr] = dataclasses.field(default_factory=list)

    def __add__(self, other: BlockParameters) -> BlockParameters:
        """
        Concatenates block parameters.
        """
        cls = type(self)
        a, b = tuple(dataclasses.asdict(x) for x in (self, other))
        return cls(**{key: a[key] + b[key] for key in a})


class CooperativeReductionWorkspaceCache:
    """
    The scratch space used for cooperative reductions can be reused
    after two reduction loops.  This keeps track of what can be reused.
    """

    def __init__(self, args):
        self.args = args
        self.current_loop = []
        self.prior_loop = []
        self.ready_for_reuse = collections.defaultdict(collections.deque)
        self.loop_count = 0
        self.store_count = 0

    def allocate(self, nbytes: sympy.Expr):
        cached = self.ready_for_reuse.get(nbytes)
        if cached:
            return cached.popleft()
        ws_name, ws_offset = self.args.workspace(nbytes, False)
        self.current_loop.append((nbytes, ws_name, ws_offset))
        return (ws_name, ws_offset)

    def on_loop_end(self):
        # Buffers can be reused after 2 loop ends
        for nbytes, ws_name, ws_offset in self.prior_loop:
            self.ready_for_reuse[nbytes].append((ws_name, ws_offset))
        self.prior_loop = self.current_loop
        self.current_loop = []
        self.loop_count += 1

    def increment_store_count(self):
        prior = self.store_count
        self.store_count += 1
        return prior


@dataclasses.dataclass
class FixedTritonConfig:
    config: Dict[str, int]

    def __getitem__(self, item):
        return self.config[item]


class TritonCSE(CSE):
    """
    Subclasses CSE to apply the current load mask to the cache key to avoid CSEing
    variables across separate masked blocks.
    """

    def augment_key(self, cache_key: object) -> object:
        if mask := V.kernel._load_mask:
            return (cache_key, mask.name)
        else:
            return cache_key


class TritonKernel(SIMDKernel):
    overrides = TritonKernelOverrides  # type: ignore[assignment]
    helper_functions: HelperFunctions
    kexpr: Callable[[sympy.Expr], str] = texpr
    allow_block_ptr = True

    def __init__(
        self,
        tiling: Dict[str, sympy.Expr],
        min_elem_per_thread=0,
        optimize_mask=True,
        fixed_config: Optional[FixedTritonConfig] = None,
        **kwargs,
    ) -> None:
        self.optimize_mask: bool = optimize_mask
        self.fixed_config = fixed_config
        super().__init__(tiling, **kwargs)
        self.cse = TritonCSE(self.newvar_prefix, self.suffix)
        self.post_loop_combine: IndentedBuffer = IndentedBuffer()
        self.post_loop_store: IndentedBuffer = IndentedBuffer()
        self.outside_loop_vars = OrderedSet[Any]()
        self.min_elem_per_thread = min_elem_per_thread
        self.block_ptr_id = itertools.count()
        self.helper_functions = HelperFunctions()
        self._load_counts: collections.Counter[str] = collections.Counter()

        # A set of autotuning hints to pass as part of triton_meta
        self.autotune_hints = OrderedSet[AutotuneHint]()
        self.triton_meta: Optional[Dict[str, Any]] = None

        if self.inside_reduction:
            self.codegen_reduction_numels(self.body)

        if self.cooperative_reduction:
            self.init_cooperative_reduction()

        self.codegen_range_tree()

    def dtype_to_str(self, dtype: torch.dtype) -> str:
        return triton_type(dtype)

    def should_use_cooperative_reduction(self) -> bool:
        return self.inside_reduction and V.choices.should_use_cooperative_reduction(
            self.features
        )

    def init_cooperative_reduction(self):
        """One time setup code for cooperative reductions."""
        assert self.cooperative_reduction

        # shift all the grids over since tl.program_id(0) is for rsplit
        for tree in self.range_trees:
            if tree.grid_dim is not None:
                tree.grid_dim += 1

        sem_count = self.numels["x"]
        if self.fixed_config:
            sem_count = CeilDiv(sem_count, self.fixed_config["XBLOCK"])
        self.semaphores_name = self.args.semaphores(sem_count)
        self.cooperative_reduction_workspace_cache = CooperativeReductionWorkspaceCache(
            self.args
        )
        self.body.splice(
            """
            rsplit_id = tl.program_id(0)
            num_rblocks = (rnumel + RBLOCK - 1) // RBLOCK
            rsplit_chunk = (num_rblocks + RSPLIT - 1) // RSPLIT * RBLOCK
            rsplit_start = rsplit_chunk * rsplit_id
            rsplit_end = rsplit_chunk * (rsplit_id + 1)
            """,
            strip=True,
        )
        if not self._has_constant_mask(self.range_trees[-1]):
            self.body.writeline(
                "rsplit_end = tl.where(rsplit_end < rnumel, rsplit_end, rnumel)"
            )

    def codegen_range_tree(self):
        for tree in self.range_trees:
            # reduction indexing goes inside a loop
            if not tree.is_loop:
                self.iteration_ranges_codegen_header(tree, self.body)
            elif self.inside_reduction:
                # workaround for this issue:
                # https://gist.github.com/jansel/6527126f781559095c5531f98a4235a7
                self.body.writeline(
                    f"{tree.prefix}base = {self.iteration_ranges_ranges_code(tree)}"
                )

        if self.inside_reduction:
            if any(tree.is_loop for tree in self.range_trees):
                # If the kernel contains loops, compute rbase.
                rn_bases = self._get_reduction_symbols(
                    "base", integer=True, nonnegative=True
                )
                rbase = self._flatten_reduction_indices(rn_bases)
                self.body.splice(f"rbase = {self.index_to_str(rbase)}")
            else:
                # For looped reductions, indexing is deferred to the innermost loop.
                self.codegen_reduction_indices(self.body)

    def need_numel_args(self):
        r"""
        Indicate whether we need provide numel as arguments for the generated
        kernel calls in the benchmark.

        Should be true for pointwise/reduction kernels but false for triton
        matmul kernels.
        """
        return True

    def should_use_persistent_reduction(self) -> bool:
        return self.inside_reduction and V.choices.should_use_persistent_reduction(
            self.features, self.cooperative_reduction
        )

    def want_no_x_dim(self):
        if self.persistent_reduction and len(self.numels) == 2:
            if self.fixed_config:
                return self.fixed_config["XBLOCK"] == 1
            return V.choices.want_no_x_dim(self.features)
        return False

    @property
    def assert_function(self) -> str:
        return "tl.device_assert"

    def indexing(
        self,
        index: sympy.Expr,
        *,
        copy_shape=None,
        dense_indexing=False,
        override_mask=None,
        block_ptr=False,
    ):
        """
        Compute the index and mask to pass to tl.load() or tl.store()
        """
        index = self.prepare_indexing(index)
        index_vars = index.free_symbols
        has_rindex = False

        mask_vars = OrderedSet[str]()
        for var in index_vars:
            assert isinstance(var, sympy.Symbol)
            has_rindex = has_rindex or symbol_is_type(
                var, TritonSymbols.reduction_types
            )
            if override_mask:
                pass
            elif symbol_is_type(var, SymT.TMP):
                # indirect indexing
                cse_var = self.cse.varname_map[var.name]
                mask_vars.update(cse_var.mask_vars)
            elif symbol_is_type(
                var,
                (
                    SymT.UNBACKED_INT,
                    SymT.SIZE,
                    SymT.PRECOMPUTED_SIZE,
                    SymT.INDEX,
                    SymT.FLOAT,
                    SymT.UNBACKED_FLOAT,
                ),
            ):
                pass
            else:
                # var is one of xN, yN, r0_N or r1_N
                prefix_matches = [
                    prefix_str[symt]
                    for symt in TritonSymbols.block_types
                    if symbol_is_type(var, symt)
                ]
                assert len(prefix_matches) == 1, f"Ambiguous type: {var.name}"
                mask_vars.add(f"{prefix_matches[0]}mask")

        need_dense = (
            config.triton.dense_indexing
            or dense_indexing
            or self._load_mask is not None
        ) and index != 0

        have_dense = True
        have_loop_vars = False
        dense_mask_vars = OrderedSet[str]()

        for tree in self.active_range_trees():
            if index_vars.intersection(tree.var_list):
                have_loop_vars = True
            else:
                have_dense = False
            dense_mask_vars.add(f"{tree.prefix}mask")

        if (
            block_ptr
            and self.allow_block_ptr
            and config.triton.use_block_ptr
            and not override_mask
            and not self._load_mask
            and len(mask_vars - dense_mask_vars) == 0
            and not self.is_indirect_indexing(index)
            and have_loop_vars
            # workaround https://github.com/openai/triton/issues/2821
            and self.index_dtype == "tl.int32"
        ):

            def match_strided_block(
                index: sympy.Expr, range_tree: IterationRangesEntry
            ) -> Optional[BlockParameters]:
                """
                Matches expressions of the form:
                    idx = s * xindex

                This implies stride (s,), and shape (XBLOCK,).
                """
                symbol = range_tree.symbol()
                stride = sympy.Wild("stride", exclude=[symbol])
                m = index.match(symbol * stride)
                if m is None:
                    return None

                return BlockParameters(
                    shape=[range_tree.numel],
                    block_shape=[TritonSymbols.get_block_size(range_tree)],
                    strides=[m[stride]],
                    offsets=[TritonSymbols.get_block_offset(range_tree)],
                )

            def match_mod_div_block(
                index: sympy.Expr, range_tree: IterationRangesEntry
            ) -> Optional[BlockParameters]:
                """
                Matches higher-dimensional blocks coming from FloorDiv and ModularIndexing.

                Example expression to match:
                   sN * ((rindex//(d1 * ... * d(N-1))))
                       + s1 * ModularIndexing(rindex, 1, d1)
                       + ...
                       + s(N-1) * ModularIndexing(rindex, d1 * ... * d(N-2), d(N-1))

                This iterates over a block of shape (dN, ..., d1) and stride
                (sN, ..., s1). (d1,...,d(N-1)) and (s1,...,sN) are
                wildcards that we match.

                Note that dN does not appear in the expression, but we solve for it
                using range tree numels and the other dims.
                """
                # Bound the possible number of dims. We use the following heuristics:
                # - At least one dim for each range tree node.
                # - At least one dim for every FloorDiv or ModularIndexing op.
                # - At least 2 dims to pattern match.
                num_dims = max(
                    2,
                    len(self.range_tree_nodes),
                    (index.count(FloorDiv) + index.count(ModularIndexing)),
                )

                # Pattern match to find the strides and offset.
                index_var = range_tree.symbol()
                match_result = BlockPatternMatcher.match_mod_div_block_expr(
                    index, index_var, range_tree.numel, num_dims
                )
                if match_result is None:
                    return None

                (
                    dims,
                    strides,
                    block_index_exprs,
                ) = match_result
                slice_numels = BlockPatternMatcher.get_slice_numels(dims)

                # Check for applicable iteration range sizes.
                # When mapping a 1D block into an ND one, we need to know that
                # the number of elements is not changed. This means the slice numels of
                # the ND iteration range must evenly divide the length of the 1D block.
                # There are two cases where we can guarantee this:
                #  1. Numels are powers of 2. If numel == 2 ** n, and we know XBLOCK == 2 ** m,
                #     with n and m integers, then either numel is a multiple of XBLOCK, or numel
                #     is less than XBLOCK. (If numel is less than XBLOCK, we round up to 1 below.)
                #  2. Numels are multiples of the maximum possible block size.
                sizevars = V.graph.sizevars
                max_block = self.max_block(range_tree.prefix)
                if any(
                    not sizevars.statically_known_multiple_of(numel, max_block)
                    and not sizevars.statically_known_power_of_2(numel)
                    for numel in slice_numels
                ):
                    return None

                # Compute the ND block shape from the linear block size.
                # Use CielDiv to round leading dimensions up to 1.
                # Non-leading dimensions are clamped to the size of the iteration range,
                # while the leading dimension can exceed this to accomodate a larger
                # block size.
                linear_block_size = TritonSymbols.get_block_size(range_tree)
                block_shape: List[sympy.Expr] = [
                    CeilDiv(linear_block_size, slice_numels[0])
                ] + [
                    sympy.Min(CeilDiv(linear_block_size, numel), dim)
                    for numel, dim in zip(slice_numels[1:], dims[1:])
                ]

                # Compute block offsets from {xyzr}offset and the matched expressions.
                block_offsets: List[sympy.Expr] = [
                    sympy_subs(
                        expr, {index_var: TritonSymbols.get_block_offset(range_tree)}
                    )
                    for expr in block_index_exprs
                ]

                return BlockParameters(
                    shape=dims,
                    block_shape=block_shape,
                    strides=strides,
                    offsets=block_offsets,
                )

            def match_block_pointer_subexpr(
                expr: sympy.Expr, range_tree: IterationRangesEntry
            ) -> Optional[BlockParameters]:
                """
                Match a block indexing subexpression involving a single range tree.
                """
                for match_func in (
                    match_strided_block,
                    match_mod_div_block,
                ):
                    match = match_func(expr, range_tree)
                    if match is not None:
                        return match

                return None

            def match_block_pointer() -> Optional[BlockPtrOptions]:
                index_relative_to_xyr_index = sympy_subs(
                    index, {v: t.expr for v, t in self.range_tree_nodes.items()}
                )
                range_trees = self.active_range_trees(reorder=True)

                # Partition the index into subexpressions pertaining to each range tree.
                # For example xindex * 5 + r0_index * 3 is partitioned to
                # (xindex * 5, r0_index * 3).
                index_subexprs = [
                    BlockPatternMatcher.get_subexpr_involving_symbol(
                        index_relative_to_xyr_index, tree.symbol()
                    )
                    for tree in range_trees
                ]

                # Match each range tree's subexpression separately.
                range_symbols = OrderedSet(tree.symbol() for tree in range_trees)
                block_params = BlockParameters()
                for tree, subexpr in zip(range_trees, index_subexprs):
                    # Reject mixed terms, e.g. xindex * r0_index.
                    # NB: the zero expression is allowed, for broadcasting.
                    if len(range_symbols.intersection(subexpr.free_symbols)) > 1:
                        return None

                    # Match the subexpression for this range tree.
                    params = match_block_pointer_subexpr(subexpr, tree)
                    if params is None:
                        return None
                    block_params += params

                # Collect leftover terms as a constant offset.
                offset = index_relative_to_xyr_index - sum(index_subexprs)

                # Form the block pointer.
                self.filter_masks(mask_vars)
                return BlockPtrOptions.create(
                    params=block_params,
                    constant_offset=offset,
                    range_trees=range_trees,
                    mask_vars=mask_vars,
                    get_max_block=self.max_block,
                )

            # Return a block pointer, if indexing matches the pattern.
            options = match_block_pointer()
            if options is not None:
                return options

        expand_str = None
        index_str = self.index_to_str(index)
        if isinstance(index, sympy.Integer):
            expand_str = f"{copy_shape}.shape" if copy_shape else self.dense_size_str()
            index_str = f"tl.full({expand_str}, {index_str}, tl.int32)"
            return IndexingOptions(
                index_str, OrderedSet(), "None", expand_str, has_rindex, index
            )

        if need_dense and not have_dense:
            expand_str = f"{copy_shape}.shape" if copy_shape else self.dense_size_str()
            index_str = f"tl.broadcast_to({index_str}, {expand_str})"
            mask_vars = dense_mask_vars
        elif not have_loop_vars and copy_shape:
            index_str = f"tl.broadcast_to({index_str}, {copy_shape}.shape)"
            mask_vars = dense_mask_vars

        if override_mask:
            mask_vars = OrderedSet([override_mask])

        if self._load_mask:
            mask_vars.add(self._load_mask)

        self.filter_masks(mask_vars)

        mask_str = " & ".join(sorted(map(str, mask_vars))) if mask_vars else "None"
        return IndexingOptions(index_str, mask_vars, mask_str, expand_str, has_rindex, index)  # type: ignore[arg-type]

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
        # Stores require an explicit broadcast.
        value = indexing.codegen_broadcast_and_reshape(
            value, indexing.final_shape, indexing.block_shape, False
        )

        # workaround https://github.com/openai/triton/issues/2814
        value = f"{value}.to({triton_store_type(V.graph.get_dtype(name))})"
        return f"tl.store({block_ptr}, {value}{other})"

    def check_bounds(
        self,
        expr: sympy.Expr,
        size: sympy.Expr,
        lower: bool,
        upper: bool,
    ):
        if not (lower or upper):
            return

        assert isinstance(expr, sympy.Expr)
        indexing = self.indexing(expr, block_ptr=False)
        assert isinstance(indexing, IndexingOptions)

        index_str = indexing.index_str
        mask_str = indexing.mask_str if indexing.has_mask() else None
        size_str = texpr(self.rename_indexing(size)) if upper else None

        # expr is already wrapped
        line = self.indirect_assert(
            index_str, "0" if lower else None, size_str, mask_str
        )

        buffer = self.get_load_buffer(indexing)
        self.cse.generate(buffer, line, assignment=False, dtype=torch.int32)

    def get_load_buffer(self, indexing):
        if indexing.has_indirect() or indexing.has_tmpmask():
            # Masked loads must come after the mask is computed
            return self.compute
        elif (
            self.inside_reduction
            and self.range_trees[-1].is_loop
            and not indexing.has_rindex()
        ):
            # can lift a common load outside of reduction loop
            # One exception is when this is an indirect_load.
            return self.body
        else:
            return self.loads

    def load(self, name: str, index: sympy.Expr):
        var = self.args.input(name)
        load_counts = self._load_counts
        load_counts[name] += 1
        make_line: Callable[[str], Union[str, DelayReplaceLine]] = identity
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

            def decide_later():
                if load_counts[name] > expected_count and (
                    has_rindex or indirect_indexing
                ):
                    return "evict_last"
                return "evict_first"

            expected_count = load_counts[name]
            ep = ", eviction_policy='<EP>'"
            make_line = functools.partial(DelayReplaceLine, "<EP>", decide_later)
        else:
            ep = ""

        if (has_tmpmask or has_rindex) and indexing.has_mask():
            if self._load_other:
                other = f", other={constant_repr(self._load_other)}"
            else:
                other = ", other=0.0"
        else:
            other = ""

        advance_block_ptr = None
        append_broadcast = None
        dtype = V.graph.get_dtype(name)

        if should_unwrap_unspec_arg(name):
            line = var
        else:
            if isinstance(indexing, BlockPtrOptions):
                block_ptr, advance_block_ptr, other = self.codegen_block_ptr(
                    name, var, indexing, other
                )
                line = f"tl.load({block_ptr}{other}{ep})"
                line = indexing.codegen_broadcast_and_reshape(
                    line, indexing.block_shape, indexing.final_shape, True
                )

            elif isinstance(original_index, sympy.Integer):
                line = f"tl.load({var} + ({original_index}))"
                append_broadcast = indexing.expand_str
            else:
                line = f"tl.load({var} + ({indexing.index_str}), {indexing.mask_str}{ep}{other})"

            if (
                dtype in (torch.float16, torch.bfloat16)
                and config.triton.codegen_upcast_to_fp32
            ):
                line += ".to(tl.float32)"
                dtype = torch.float32
            if dtype == torch.bool and torch.version.hip is None:
                # Workaround for https://github.com/openai/triton/issues/2151
                # tl.load returns int8 when loading from pointer to int1
                # NOTE: Currently causes hangs on bool UTs for ROCm
                line += ".to(tl.int1)"
                dtype = torch.bool

        load_buffer = self.get_load_buffer(indexing)
        result_var = self.cse.generate(load_buffer, make_line(line), dtype=dtype)
        if result_var.use_count > 1:
            load_counts[name] -= 1  # don't double count cache hit
        assert isinstance(result_var, TritonCSEVariable)
        result_var.mask_vars = indexing.mask_vars  # type: ignore[assignment]

        if append_broadcast:
            line = f"tl.broadcast_to({result_var}, {append_broadcast})"
            result_var = self.cse.generate(load_buffer, line, dtype=dtype)

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
            line = f"tl.atomic_add({var} + ({indexing.index_str}), {value}, {indexing.mask_str}, sem='relaxed')"
        else:
            raise NotImplementedError(f"store mode={mode}")

        exit_stack = contextlib.ExitStack()
        if not self.inside_reduction and self.cooperative_reduction:
            exit_stack.enter_context(self.guard_cooperative_store(name, self.stores))

        self.stores.writeline(DeferredLine(name, line))
        if advance_block_ptr:
            self.stores.writeline(advance_block_ptr)

        if not self.inside_reduction:
            self.outside_loop_vars.add(value)

        exit_stack.close()

    def guard_cooperative_store(self, name, buffer):
        """
        For cooperative reductions only one thread block should write out the result.
        We rotate which thread block does each write for better parallelism
        """
        idx = self.cooperative_reduction_workspace_cache.increment_store_count()
        buffer.writeline(DeferredLine(name, f"if rsplit_id == ({idx} % RSPLIT):"))
        return buffer.indent()

    def bucketize(
        self,
        values: CSEVariable,
        boundaries: Tuple[str, sympy.Expr, sympy.Expr, sympy.Expr],
        boundary_indices: CSEVariable,
        indexing_dtype: torch.dtype,
        right: bool,
        sorter: Optional[Tuple[str, sympy.Expr]] = None,
        sorter_indices: Optional[CSEVariable] = None,
    ) -> CSEVariable:
        """
        See [Note: Inductor bucketize op]
        """

        # Triton performance for bucketize_binary_search is much better when the number
        # of threads equals the number of elements.
        # If we're trying to use a bucketize kernel, we should make sure that an
        # autotuning config with num_elements_per_warp=(warp_size) exists.
        self.autotune_hints.add(AutotuneHint.ONE_ELEMENT_PER_THREAD)

        boundaries_ptr = self.args.input(boundaries[0])
        boundary_size = self.index_to_str(boundaries[1])
        boundaries_underlying_numel = self.index_to_str(boundaries[2])
        boundary_stride = self.index_to_str(boundaries[3])
        sorter_ptr = self.args.input(sorter[0]) if sorter else "None"
        sorter_stride = self.index_to_str(sorter[1]) if sorter else "None"
        block_size = self.dense_size_str()

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
            f"triton_helpers.bucketize_binary_search({values}, "
            f"{boundaries_ptr}, {boundary_size}, {boundaries_underlying_numel}, {boundary_stride}, "
            f"{boundary_indices}, "
            f"{triton_dtype}, "
            f"{right}, "
            f"{sorter_ptr}, {sorter_stride}, "
            f"{sorter_indices}, "
            f"{block_size}, "
            ")",
            dtype=indexing_dtype,  # type: ignore[attr-defined]
        )

        return result

    def reduction_resize(self, value):
        ndims = self.triton_tensor_ndim()
        if ndims == 1:
            return f"triton_helpers.promote_to_tensor({value})"

        sizes = [":"] * ndims
        sizes[-1] = "None"
        return f"{value}[{', '.join(sizes)}]"

    def reduction(
        self,
        dtype: torch.dtype,
        src_dtype: torch.dtype,
        reduction_type: ReductionType,
        value: Union[CSEVariable, Tuple[CSEVariable, ...]],
    ) -> Union[CSEVariable, Tuple[CSEVariable, ...]]:
        assert self.inside_reduction
        masks = OrderedSet(f"{tree.prefix}mask" for tree in self.range_trees)
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
                self.compute,
                f"tl.broadcast_to({v}, {dense_size_str})",
                dtype=v.dtype,
            ),
            value,
        )

        dim: int
        root_op: str

        def final_reduction(value):
            use_helper = reduction_type in ("any", "max", "min", "prod")
            module = "triton_helpers" if use_helper else "tl"
            if reduction_type in ("max", "min"):
                return self.reduction_resize(
                    f"{module}.{reduction_type}2({value}, {dim})"
                )
            return self.reduction_resize(f"{module}.{reduction_type}({value}, {dim})")

        def final_argreduce(buffer, result_var, value, index):
            buffer.splice(
                f"""\
                {result_var}_val, {result_var}_idx = triton_helpers.{root_op}_with_index({value}, {index}, {dim})
                {result_var} = {self.reduction_resize(f'{result_var}_idx')}
                """
            )

        cache_key = (src_dtype, reduction_type, value)
        if cache_key in self.cse.reduction_cache:
            return self.cse.reduction_cache[cache_key]

        dim = self.triton_tensor_ndim() - 1
        acc_type = triton_acc_type(src_dtype)
        torch_acc_type = upcast_acc_dtype(src_dtype)
        result_var: Any = self.cse.newvar(dtype=torch_acc_type)
        result_var.mask_vars = OrderedSet(
            var for var in masks if not prefix_is_reduction(var[0])
        )
        cond = " & ".join(masks)

        def where_cond(tval, fval):
            if not cond:
                return tval
            return TritonKernelOverrides.where(cond, tval, fval)

        if self.persistent_reduction:
            default = ir.Reduction.default_value(reduction_type, src_dtype)
            default = self._map_tuple_or_scalar(constant_repr, default)

            def _mask_value(value, default):
                return self.cse.generate(
                    self.compute, where_cond(value, default), dtype=value.dtype
                )

            if isinstance(value, tuple):
                masked_value = [_mask_value(v, d) for v, d in zip(value, default)]
            else:
                masked_value = _mask_value(value, default)

            if reduction_type in ("argmax", "argmin"):
                accumulator_index = str(
                    self.cse.generate(
                        self.compute,
                        f"tl.broadcast_to({reduction_range_prefix}index, {masked_value}.shape)",
                        dtype=torch.int64,
                    )
                )
                root_op = {"argmax": "max", "argmin": "min"}[reduction_type]
                final_argreduce(
                    self.compute, result_var, masked_value, accumulator_index
                )
            elif reduction_type == "welford_reduce":
                if self.cooperative_reduction:
                    # cooperative reductions require full welford for correctness
                    result_var = self.welford_reduce(
                        result_var, reduction_type, value, where_cond, acc_type, dtype
                    )
                else:
                    # For persistent reductions, don't bother with
                    # welford's algorithm since it uses more registers, and
                    # taking two reductions doesn't increase memory usage.
                    result_var = self.welford_reduce_fallback(dtype, value)
            elif reduction_type == "welford_combine":
                mean, m2, weight = masked_value
                welford = f"triton_helpers.welford({mean}, {m2}, {weight}, {dim})"
                mean, m2, weight = (self.cse.newvar(dtype=dtype) for _ in range(3))
                self.compute.writeline(f"{mean}, {m2}, {weight} = {welford}")

                result_var = tuple(
                    self.cse.generate(
                        self.compute, self.reduction_resize(var_name), dtype=dtype
                    )
                    for var_name in (mean, m2, weight)
                )
            else:
                assert isinstance(masked_value, CSEVariable)
                result_var = self.cse.generate(
                    self.compute,
                    final_reduction(masked_value),
                    dtype=masked_value.dtype,
                )
        else:
            accumulator = self.cse.namedvar(f"_{result_var}", dtype=torch_acc_type)
            default = ir.Reduction.default_accumulator(reduction_type, src_dtype)
            default = self._map_tuple_or_scalar(constant_repr, default)
            if not isinstance(default, tuple):
                self.body.writeline(
                    f"{accumulator} = tl.full({self.dense_size_str()}, {default}, {acc_type})"
                )

            if reduction_type in ("argmax", "argmin"):
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
                final_argreduce(
                    self.post_loop_combine, result_var, accumulator, accumulator_index
                )
            elif is_welford_reduction(reduction_type):
                result_var = self.welford_reduce(
                    result_var, reduction_type, value, where_cond, acc_type, dtype
                )
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
                    accumulator_casted_str = f"{accumulator}.to(tl.int8)"
                    result_type = triton_compute_type(dtype)
                    self.post_loop_combine.writeline(
                        f"{result_var} = {final_reduction(accumulator_casted_str)}.to({result_type})"
                    )
                else:
                    self.post_loop_combine.writeline(
                        f"{result_var} = {final_reduction(accumulator)}"
                    )

        if self.cooperative_reduction:
            exit_stack = contextlib.ExitStack()
            for buf in (self.post_loop_combine, self.post_loop_store):
                # only do cooperative reduction combines if we have more than one thread block
                buf.writeline("if RSPLIT > 1:")
                exit_stack.enter_context(buf.indent())

            if reduction_type in ("argmax", "argmin"):
                self.post_loop_combine.writeline(
                    f"{result_var}_bval = {self.reduction_resize(f'{result_var}_val')}"
                )
                peer_val = self.codegen_cooperative_reduction_peer_combine(
                    f"{result_var}_bval", src_dtype
                )
                peer_idx = self.codegen_cooperative_reduction_peer_combine(
                    result_var, dtype
                )
                final_argreduce(self.post_loop_store, result_var, peer_val, peer_idx)
            elif is_welford_reduction(reduction_type):
                assert reduction_type == "welford_reduce"
                result_mean, result_m2, result_weight = result_var
                peer_mean = self.codegen_cooperative_reduction_peer_combine(
                    result_mean, upcast_acc_dtype(src_dtype)
                )
                peer_m2 = self.codegen_cooperative_reduction_peer_combine(
                    result_m2, upcast_acc_dtype(src_dtype)
                )
                peer_weight = self.codegen_cooperative_reduction_peer_combine(
                    result_weight, upcast_acc_dtype(src_dtype)
                )
                self.welford_reduce_final_reduction(
                    self.post_loop_store,
                    result_mean,
                    result_m2,
                    result_weight,
                    peer_mean,
                    peer_m2,
                    peer_weight,
                    dim,
                )
            else:
                peers = self.codegen_cooperative_reduction_peer_combine(
                    result_var, upcast_acc_dtype(src_dtype)
                )
                self.post_loop_store.writeline(
                    f"{result_var} = {final_reduction(peers)}"
                )
            exit_stack.close()

        self.cse.reduction_cache[cache_key] = result_var

        if isinstance(result_var, tuple):
            assert all(isinstance(x, TritonCSEVariable) for x in result_var)
            self.outside_loop_vars.update(result_var)
        else:
            assert isinstance(result_var, TritonCSEVariable)
            self.outside_loop_vars.add(result_var)

        return result_var

    def welford_reduce(
        self, result_var, reduction_type, value, where_cond, acc_type, dtype
    ):
        """Helper to codegen a welford reduction"""
        dim = self.triton_tensor_ndim() - 1
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
        result_m2 = self.cse.newvar(dtype=dtype)
        result_weight = self.cse.newvar(dtype=dtype)
        return self.welford_reduce_final_reduction(
            self.post_loop_combine,
            result_mean,
            result_m2,
            result_weight,
            accumulator,
            accumulator_m2,
            accumulator_weight,
            dim,
        )

    def welford_reduce_final_reduction(
        self,
        buf,
        result_mean,
        result_m2,
        result_weight,
        accumulator,
        accumulator_m2,
        accumulator_weight,
        dim,
    ):
        """Helper to codegen call to triton_helpers.welford"""
        buf.splice(
            f"""\
            {result_mean}_tmp, {result_m2}_tmp, {result_weight}_tmp = triton_helpers.welford(
                {accumulator}, {accumulator_m2}, {accumulator_weight}, {dim}
            )
            {result_mean} = {self.reduction_resize(f'{result_mean}_tmp')}
            {result_m2} = {self.reduction_resize(f'{result_m2}_tmp')}
            {result_weight} = {self.reduction_resize(f'{result_weight}_tmp')}
            """
        )
        return result_mean, result_m2, result_weight

    def max_rsplit(self):
        if self.fixed_config:
            return self.fixed_config["RSPLIT"]
        return TRITON_MAX_RSPLIT

    def codegen_cooperative_reduction_peer_combine(self, result_var, dtype):
        """
        Generate code to save a [XBLOCK, RSPLIT] temporary workspace, where each thread block writes a different
        column.  After the barrier, every thread block loads the completed value so that it can compute the final
        value independently.
        """
        xnumel = self.numels["x"]
        mask = "xindex < xnumel" if xnumel != 1 and not self.no_x_dim else None
        expand = "" if self.no_x_dim else "[None,:]"

        nbytes = xnumel * dtype.itemsize * self.max_rsplit()
        ws_name, ws_offset = self.cooperative_reduction_workspace_cache.allocate(nbytes)

        self.post_loop_combine.splice(
            f"""
                {result_var}_ws = ({ws_name} + {self.index_to_str(ws_offset)}).to(tl.pointer_type({triton_type(dtype)}))
                tl.store({result_var}_ws + (xindex * RSPLIT + rsplit_id), {result_var}, {mask})
            """,
            strip=True,
        )
        self.post_loop_store.writeline(
            f"{result_var}_peers = tl.load({result_var}_ws + (xindex * RSPLIT + tl.arange(0, RSPLIT){expand}), "
            f"{mask}, eviction_policy='evict_first')"
        )
        return f"{result_var}_peers"

    def store_reduction(self, name: str, index: sympy.Expr, value: CSEVariable):
        assert self.inside_reduction
        self.inside_reduction = False
        indexing = self.indexing(index, block_ptr=True)
        self.inside_reduction = True
        var = self.args.output(name)

        exit_stack = contextlib.ExitStack()
        if self.cooperative_reduction:
            exit_stack.enter_context(
                self.guard_cooperative_store(name, self.post_loop_store)
            )

        if isinstance(indexing, BlockPtrOptions):
            self.post_loop_store.writeline(
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
            self.post_loop_store.writeline(
                DeferredLine(
                    name,
                    f"tl.store({var} + ({indexing.index_str}), {value}, {indexing.mask_str})",
                )
            )

        exit_stack.close()

    def _lift_helper(self, fn, num_args, dtypes: Tuple[torch.dtype, ...]) -> str:
        # Lift IR function for scan operations into a triton function
        # in the global namespace
        helper = IndentedBuffer()
        helper.writeline("@triton.jit")
        cse = CSE(prefix="", suffix="")

        args = [
            tuple(cse.namedvar(f"arg{i}_{n}", dtype=dtypes[n]) for n in range(num_args))
            for i in range(2)
        ]
        signature = ", ".join(str(x) for x in itertools.chain.from_iterable(args))
        helper.writeline(f"def {{name}}({signature}):")

        overrides = TritonOverrides(V.MockHandler())

        # Build a name that changes depending on fn to workaround a triton bug
        # where the combine_fn to reduce and scan is not hashed, and so different
        # scan ops may collide in the triton cache.
        # This is fixed with the latest triton pin, but not the triton-rocm pin.
        helper_name = "_triton_helper_fn"

        from torch._inductor.dtype_propagation import DtypePropagationOpsHandler

        dtype_handler = DtypePropagationOpsHandler()

        class CSEProxy:
            def __getattr__(self, name: str) -> Callable[..., CSEVariable]:
                def inner(*args, **kwargs):
                    nonlocal helper_name
                    helper_name += f"_{name}"

                    output_dtype = getattr(
                        dtype_handler,
                        name,
                    )(*args, **kwargs)

                    return cse.generate(
                        helper,
                        getattr(overrides, name)(*args, **kwargs),
                        dtype=output_dtype,
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
        assert not self.cooperative_reduction, "TODO"
        masks = OrderedSet(f"{tree.prefix}mask" for tree in self.range_trees)
        self.filter_masks(masks)
        masks = sorted(masks)
        assert not self._load_mask, "ops.scan not supported inside ops.masked"

        broadcasted_values = []
        accumulators = []

        cse_compute = functools.partial(self.cse.generate, self.compute)
        combine_helper_fn = self._lift_helper(combine_fn, len(values), dtypes)
        dim = self.triton_tensor_ndim() - 1

        for value, dtype in zip(values, dtypes):
            value_dtype = self.cse.generate(
                self.compute,
                f"{value}.to({triton_compute_type(dtype)})",
                dtype=upcast_compute_type(dtype),
            )
            value = self.cse.generate(
                self.compute,
                f"tl.broadcast_to({value_dtype}, {self.dense_size_str()})",
                dtype=upcast_compute_type(dtype),
            )
            broadcasted_values.append(value)

            acc_type = triton_acc_type(dtype)

            if not self.persistent_reduction:
                accumulator = self.cse.newvar(dtype=upcast_compute_type(dtype))
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

        def cse_multiple(line, values, masks, dtypes):
            n = len(values)
            cache_keys = [f"{line}, {i}, {masks}" for i in range(n)]
            if all(self.cse.contains(cache_key) for cache_key in cache_keys):
                return [self.cse.get(cache_key) for cache_key in cache_keys]
            result_vars = [self.cse.newvar(dtype=_dtype) for _dtype in dtypes]
            self.compute.writeline(
                f"{csv(result_vars)} = {line}",
            )
            for result_var, cache_key in zip(result_vars, cache_keys):
                if masks:
                    result_var.mask_vars = masks  # type: ignore[attr-defined]
                self.cse.put(cache_key, result_var)
            return tuple(result_vars)

        partial_scan_vars = cse_multiple(
            f"tl.associative_scan(({csv(broadcasted_values)}), {dim}, {combine_helper_fn})",
            values,
            masks,
            (upcast_compute_type(dtype) for dtype in dtypes),
        )

        if not self.persistent_reduction:
            # tl.reduce doesn't work for non-commutative operators, so instead
            # of repeating the scan op as a reduction, we use sum to select the
            # last scan value
            partial_reduce_vars = [
                cse_compute(
                    f"triton_helpers.select_one(({partial_scan_var}), rbase == (RBLOCK - 1), dim=-1, keep_dims=True)",
                    dtype=upcast_compute_type(partial_scan_var.dtype),
                )
                for partial_scan_var in partial_scan_vars
            ]
            accs_next = combine_fn(tuple(accumulators), tuple(partial_reduce_vars))
            full_scan_vars = combine_fn(tuple(accumulators), partial_scan_vars)
            result_vars = [
                cse_compute(
                    f"tl.where(roffset > 0, {full_scan}, {partial_scan})",
                    dtype=partial_scan.dtype,
                )
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

    def sort(
        self,
        dtypes: Tuple[torch.dtype, ...],
        values: Tuple[CSEVariable, ...],
        stable: bool,
        descending: bool,
    ) -> Tuple[CSEVariable, ...]:
        assert self.inside_reduction
        assert not self.cooperative_reduction, "TODO"
        masks = OrderedSet(f"{tree.prefix}mask" for tree in self.range_trees)
        self.filter_masks(masks)
        masks = sorted(masks)
        assert not self._load_mask, "ops.sort not supported inside ops.masked"
        assert (
            self.persistent_reduction
        ), "ops.sort is only supported in persistent reductions"

        cse_compute = functools.partial(self.cse.generate, self.compute)
        dim = self.triton_tensor_ndim() - 1

        assert len(dtypes) == len(values)
        broadcasted_values = [
            cse_compute(
                f"tl.broadcast_to({value}, {self.dense_size_str()})", dtype=dtypes[i]
            )
            for i, value in enumerate(values)
        ]

        def csv(values):
            return " ".join(f"{value}," for value in values)

        def cse_multiple(line, n, masks, dtypes):
            cache_keys = [f"{line}, {i}, {masks}" for i in range(n)]
            if all(self.cse.contains(cache_key) for cache_key in cache_keys):
                return [self.cse.get(cache_key) for cache_key in cache_keys]
            result_vars = [self.cse.newvar(dtype=dtypes[i]) for i in range(n)]  # type: ignore[attr-defined]
            self.compute.writeline(
                f"{csv(result_vars)} = {line}",
            )
            for result_var, cache_key in zip(result_vars, cache_keys):
                if masks:
                    result_var.mask_vars = masks  # type: ignore[attr-defined]
                self.cse.put(cache_key, result_var)
            return tuple(result_vars)

        assert self.range_trees[-1].is_reduction
        rnumel = "None" if self._has_constant_mask(self.range_trees[-1]) else "rnumel"

        if len(values) == 2:
            line = (
                f"triton_helpers.sort_with_index({broadcasted_values[0]}, {broadcasted_values[1]},"
                f" {rnumel}, {dim}, stable={stable}, descending={descending})"
            )
            result_vars = cse_multiple(line, len(values), masks, dtypes)
        else:
            raise AssertionError("Unhandled sort")

        for result_var, input_var in zip(result_vars, values):
            result_var.mask_vars = masks  # type: ignore[attr-defined]
            result_var.bounds = input_var.bounds

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
            or self.post_loop_combine
            or self.post_loop_store
        ):
            return

        innermost_tree = self.range_trees[-1]
        if self.inside_reduction and innermost_tree.is_loop:
            prefix = innermost_tree.prefix
            loop_start = "rsplit_start" if self.cooperative_reduction else "0"
            loop_end = "rsplit_end" if self.cooperative_reduction else f"{prefix}numel"
            self.body.writeline(
                f"for {prefix}offset in range({loop_start}, {loop_end}, {prefix.upper()}BLOCK):"
            )

            with self.body.indent():
                # last range tree is always reduction
                self.iteration_ranges_codegen_header(self.range_trees[-1], self.body)
                self.codegen_reduction_indices(self.body)
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
        self.body.splice(self.post_loop_combine)
        if self.cooperative_reduction and (
            self.post_loop_combine or self.post_loop_store
        ):
            sem_ptr = f"{self.semaphores_name} + tl.program_id(1)"
            self.body.splice(
                f"""
                if RSPLIT > 1:
                    triton_helpers.x_grid_barrier({sem_ptr})
                """,
                strip=True,
            )
            self.cooperative_reduction_workspace_cache.on_loop_end()
        self.body.splice(self.post_loop_store)
        self.indexing_code.clear()
        self.loads.clear()
        self.compute.clear()
        self.stores.clear()
        self.post_loop_combine.clear()
        self.post_loop_store.clear()

    def codegen_kernel_benchmark(self, num_gb, grid=None):
        result = IndentedBuffer()
        _argdefs, call_args, signature, _ = self.args.python_argdefs()

        result.writelines(["", "", "def get_args():"])
        with result.indent():
            name_cnt = itertools.count()
            var_names = []
            for arg_name, arg_sig in zip(call_args, signature):
                var_name = f"arg_{next(name_cnt)}"
                buf = V.graph.try_get_buffer(arg_name)
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
                elif isinstance(arg_sig, WorkspaceArg):
                    device = V.graph.get_current_device_or_throw()
                    count = V.graph.sizevars.size_hint(arg_sig.count)
                    result.writeline(
                        f"{var_name} = torch.zeros({count}, device='{device}', dtype={arg_sig.dtype})"
                    )
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
                if not tree.is_reduction:
                    grid.append(expr)
            if self.need_numel_args():
                extra_args_str = ", ".join(map(str, extra_args)) + ", "
            else:
                extra_args_str = ""
            grid_arg = f"{extra_args_str}grid=grid({', '.join(grid)})"
        else:
            grid_arg = f"grid={grid}"
        current_device = V.graph.get_current_device_or_throw()
        index = current_device.index
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
            result.writeline(
                "from torch._inductor.runtime.benchmarking import benchmarker"
            )
            result.writeline("")

            result.writeline("args = get_args()")
            result.writeline(
                "ms = benchmarker.benchmark_gpu(lambda: call(args), rep=40)"
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
            from torch._inductor.runtime.triton_heuristics import grid, split_scan_grid
        """.format(
                V.graph.device_ops.import_get_raw_stream_as("get_raw_stream")
            )
        )

    def _get_heuristic(self):
        if self.fixed_config:
            return "fixed_config"
        elif self.cooperative_reduction:
            return "cooperative_reduction"
        elif self.persistent_reduction:
            assert self.inside_reduction
            return "persistent_reduction"
        elif self.inside_reduction:
            return "reduction"
        return "pointwise"

    @staticmethod
    def inductor_meta_common():
        compile_id = torch._guards.CompileContext.current_compile_id()
        inductor_meta = {
            "backend_hash": torch.utils._triton.triton_hash_with_backend(),
            "are_deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
            "assert_indirect_indexing": config.assert_indirect_indexing,
            "autotune_local_cache": config.autotune_local_cache,
            "autotune_pointwise": config.triton.autotune_pointwise,
            "autotune_remote_cache": config.autotune_remote_cache,
            "force_disable_caches": config.force_disable_caches,
            "dynamic_scale_rblock": config.dynamic_scale_rblock,
            "max_autotune": config.max_autotune,
            "max_autotune_pointwise": config.max_autotune_pointwise,
            "min_split_scan_rblock": config.triton.min_split_scan_rblock,
            "spill_threshold": config.triton.spill_threshold,
            "store_cubin": config.triton.store_cubin,
            "compile_id": str(compile_id) if compile_id else None,
            "is_forward": not V.graph.is_backward,
        }
        if torch.version.hip is not None:
            inductor_meta["is_hip"] = True
        if config.is_fbcode():
            inductor_meta["is_fbcode"] = True
        if config.profile_bandwidth:
            inductor_meta["profile_bandwidth"] = config.profile_bandwidth
            inductor_meta["profile_bandwidth_regex"] = config.profile_bandwidth_regex
            inductor_meta["profile_bandwidth_output"] = config.profile_bandwidth_output
            inductor_meta[
                "profile_bandwidth_with_do_bench_using_profiling"
            ] = config.profile_bandwidth_with_do_bench_using_profiling
        if config.coordinate_descent_tuning:
            inductor_meta[
                "coordinate_descent_tuning"
            ] = config.coordinate_descent_tuning
            inductor_meta[
                "coordinate_descent_search_radius"
            ] = config.coordinate_descent_search_radius
            inductor_meta[
                "coordinate_descent_check_all_directions"
            ] = config.coordinate_descent_check_all_directions
        return inductor_meta

    def codegen_kernel(self, name=None):
        code = IndentedBuffer()

        size_hints = {}
        for prefix, numel in self.numels.items():
            if prefix_is_reduction(prefix) and not self.inside_reduction:
                continue

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
            size_hints[prefix] = size_hint

        if name is None:
            code.splice(gen_common_triton_imports())
            device_type = V.graph.get_current_device_or_throw().type
            if device_type == "cpu":
                code.splice("triton_helpers.set_driver_to_cpu()")
            else:
                code.splice("triton_helpers.set_driver_to_gpu()")

            if config.benchmark_kernel:
                code.splice(self.imports_for_benchmark_kernel())

        argdefs, _, signature, _ = self.args.python_argdefs()
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

        mutated_args = OrderedSet[str]()
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

        # Note: [Workspace Mutation]
        # workspace arguments are mutated, but are not marked as mutations in self.mutations
        # because their buffers are added during codegen, and aren't tracked during
        # lowering/scheduling. So we add them as mutated_args explicitly below.
        #
        # In the logic below, we only mark the workspaces a mutated if they are marked with
        # zero_fill: that's because, if we don't expect the buffer to be pre-filled with
        # zeros, then, although we still mutate the data, we don't care about those
        # mutations because we don't make any assumptions about the contents of the
        # workspace buffer.  Similarly, ZERO_PER_GRAPH requires the kernel to return
        # the buffer back to its original state.
        for argname, arg in zip(argdefs, signature):
            if (
                isinstance(arg, WorkspaceArg)
                and arg.zero_mode == WorkspaceZeroMode.ZERO_ON_CALL
            ):
                mutated_args.add(argname)

        mutated_args = sorted(mutated_args)

        triton_meta_signature = signature_to_meta(
            signature, size_dtype=self.index_dtype, argdefs=argdefs
        )
        triton_meta: Dict[str, Any] = {
            "signature": triton_meta_signature,
            "device": DeviceProperties.create(V.graph.get_current_device_or_throw()),
            "constants": {},
        }

        # Skip memory optimization for forward of the training loop where we expect
        # every new node will increase the peak memory and our greedy approach would
        # introduce a lot of unnecessary cpu copies.
        optimize_mem = V.graph.is_inference or V.graph.is_backward

        inductor_meta = {
            # Triton will not accept an OrderedSet for autotune_hints
            "autotune_hints": set(self.autotune_hints),  # noqa: set_linter
            "kernel_name": str(Placeholder.DESCRIPTIVE_NAME),
            "mutated_arg_names": mutated_args,
            "optimize_mem": optimize_mem,
            "no_x_dim": self.no_x_dim,
            "num_load": self.num_load,
            "num_reduction": self.num_reduction,
            **self.inductor_meta_common(),
        }
        if self.cooperative_reduction:
            inductor_meta["persistent_reduction"] = self.persistent_reduction

        num_gb = None
        if config.benchmark_kernel or config.profile_bandwidth:
            num_gb = self.estimate_kernel_num_bytes() / 1e9
            inductor_meta["kernel_num_gb"] = num_gb

        for tree in self.active_range_trees():
            sizearg = SizeArg(f"{tree.prefix}numel", tree.numel)
            signature.append(sizearg)
            triton_meta_signature[sizearg.name] = signature_of(
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
            triton_meta["constants"][signature[arg_num].name] = 1  # type: ignore[index]

        self.triton_meta = triton_meta

        for tree in self.range_trees:
            if tree.is_reduction and self.persistent_reduction:
                # Rn_BLOCK for persistent_reduction is defined in codegen_static_numels
                continue
            if tree.tensor_dim is None:
                continue
            argdefs.append(f"{tree.prefix.upper()}BLOCK : tl.constexpr")

        if self.cooperative_reduction:
            argdefs.append("RSPLIT : tl.constexpr")

        self.codegen_body()

        for helper in self.helper_functions:
            code.writeline("")
            code.splice(helper)

        if self.fixed_config:
            heuristics_line = f"""
                @triton_heuristics.{self._get_heuristic()}(
                    config={self.fixed_config.config!r},
                    filename=__file__,
                    triton_meta={triton_meta!r},
                    inductor_meta={inductor_meta!r}
                )
                @triton.jit
            """
        elif self.inside_reduction:
            reduction_hint = self.features.get_reduction_hint()
            heuristics_line = f"""
                @triton_heuristics.{self._get_heuristic()}(
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
                @triton_heuristics.{self._get_heuristic()}(
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

    @staticmethod
    def _get_persistent_RBLOCK(rnumel):
        rnumel = V.graph.sizevars.simplify(rnumel)
        if isinstance(rnumel, (sympy.Integer, int)):
            val = int(rnumel)
            val = next_power_of_2(val)
        else:
            val = 128
            while not V.graph.sizevars.statically_known_leq(rnumel, val):
                if val > 16 * 1024:
                    raise ValueError(f"Failed to find static RBLOCK for {rnumel}")
                val *= 2
        return val

    @staticmethod
    def has_persistent_RBLOCK(rnumel):
        try:
            TritonKernel._get_persistent_RBLOCK(rnumel)
            return True
        except ValueError:
            return False

    def codegen_static_numels(self, code):
        """
        We get a small speedup from hard coding numels if they are static.

        This code stomps on the passed-in values by writing an constant to the top of the kernel.

        In a kernel like:
        def KERNEL_NAME(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, Rn_BLOCK : tl.constexpr):

        We would add
        xnumel = 4096
        rnumel = 768

        After the signature, before the kernel code, if we decided to make these static. As its hardcoded, it becomes
        a better signal to triton on how to unroll and do some static indexing. So, it's not so much that downstream
        knows that its a static numel, as that you just plop a constant into the kernel.
        """
        for tree in self.range_trees:
            if not tree.is_reduction or self.inside_reduction:
                simplified_tree_numel = V.graph.sizevars.simplify(tree.numel)
                if isinstance(simplified_tree_numel, (sympy.Integer, int)):
                    code.writeline(f"{tree.prefix}numel = {int(simplified_tree_numel)}")

            if tree.is_reduction and self.persistent_reduction:
                val = self._get_persistent_RBLOCK(tree.numel)
                if self.cooperative_reduction:
                    val = f"{val} // RSPLIT"
                code.writeline(f"{tree.prefix.upper()}BLOCK: tl.constexpr = {val}")

            if tree.prefix == "x" and self.no_x_dim:
                code.writeline("XBLOCK: tl.constexpr = 1")

    def _get_grid_fn_str(self):
        return self._get_grid_fn().__name__

    def _get_grid_fn(self):
        if self.cooperative_reduction:
            return cooperative_reduction_grid
        return default_grid_fn

    def add_numel_to_call_args_and_grid(self, name, call_args, arg_types, grid):
        # TODO(jansel): if there are constants, we shouldn't bother passing them as args
        for tree in self.range_trees:
            if isinstance(tree.numel, (sympy.Integer, sympy.Symbol)):
                expr = tree.numel
            else:
                expr = V.graph.wrapper_code.generate_numel_expr(name, tree)

            if not tree.is_reduction or self.inside_reduction:
                call_args.append(expr)
                arg_types.append(type(expr))
            if tree.grid_dim is not None:
                grid.append(expr)

    def call_kernel(self, name: str, node: Optional[IRNode] = None):
        wrapper = V.graph.wrapper_code
        wrapper.write_triton_header_once()
        _, call_args, _, arg_types = self.args.python_argdefs()
        grid: List[Any] = []
        self.add_numel_to_call_args_and_grid(name, call_args, arg_types, grid)
        current_device = V.graph.get_current_device_or_throw()

        for ws in self.args.workspace_args:
            wrapper.generate_workspace_allocation(ws)

        grid = wrapper.generate_default_grid(
            name, grid, grid_callable=self._get_grid_fn()
        )
        wrapper.generate_kernel_call(
            name,
            call_args,
            grid,
            current_device.index,
            gpu=current_device.type != "cpu",
            triton=True,
            arg_types=arg_types,
            grid_fn=self._get_grid_fn_str(),
            triton_meta=self.triton_meta,
        )

        for ws in reversed(self.args.workspace_args):
            wrapper.generate_workspace_deallocation(ws)

    def codegen_nan_check(self):
        wrapper = V.graph.wrapper_code
        _, call_args, arg_signatures, _ = self.args.python_argdefs()
        for arg, arg_signature in zip(call_args, arg_signatures):
            if isinstance(arg_signature, TensorArg):
                if V.graph.cpp_wrapper:
                    wrapper.writeline(
                        f'AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_check_inf_and_nan("{arg}", {arg}));'
                    )
                else:
                    line = f"assert not {arg}.isnan().any().item()"
                    wrapper.writeline(line)
                    line = f"assert not {arg}.isinf().any().item()"
                    wrapper.writeline(line)

    def create_cse_var(self, *args, **kwargs):
        return TritonCSEVariable(*args, **kwargs)

    def codegen_iteration_ranges_entry(self, entry: IterationRangesEntry):
        line = f"{entry.name} = {self.kexpr(self.rename_indexing(entry.expr))}"
        if entry.root.is_loop:
            self.indexing_code.writeline(line)
        else:
            # lift non-reduction stores outside loop
            self.body.writeline(line)

    def iteration_ranges_ranges_code(self, entry):
        assert entry.tensor_dim is not None
        size = self.indexing_size_str(entry.tensor_dim)
        index_dtype = self.index_dtype
        suffix = f".to({index_dtype})" if index_dtype != "tl.int32" else ""
        if (
            self.cooperative_reduction
            and self.persistent_reduction
            and entry.is_reduction
        ):
            suffix = f"{suffix} + rsplit_start"
        return f"tl.arange(0, {entry.prefix.upper()}BLOCK){size}{suffix}"

    def iteration_ranges_scalar_code(self, entry, value):
        index_dtype = self.index_dtype
        ndim = self.triton_tensor_ndim()
        size = [1] * ndim
        return f"tl.full({size}, {value}, {index_dtype})"

    def iteration_ranges_get_pid(self, entry):
        assert entry.grid_dim is not None
        key = f"tl.program_id({entry.grid_dim})"
        # y_grid has a limit, so express it in terms of y and z in case of overflow.
        # z grid is only exercised when max_tiles == 3 (off by default).
        if (
            entry.grid_dim == 1
            and not entry.has_zdim
            and not self.cooperative_reduction
            and not V.graph.sizevars.statically_known_leq(entry.numel, get_max_y_grid())
        ):
            # For ynumel larger than max_ygrid, we need to use zdim.
            # For each z dimension, there are tl.num_programs(1) yblocks which is passed by grad(x,y,z).
            # So, we need to add tl.program_id(z) * tl.num_programs(y) *YBLOCK to get the correct yoffset.
            key = f"({key} + tl.program_id({entry.grid_dim + 1}) * tl.num_programs({entry.grid_dim}))"
        pid = entry.pid_cache.get(key, key)
        if self.index_dtype != "tl.int32":
            return f"{pid}.to({self.index_dtype})"
        return pid

    def max_block(self, prefix):
        if self.fixed_config:
            return self.fixed_config[f"{prefix.upper()}BLOCK"]
        return TRITON_MAX_BLOCK[prefix.upper()]

    def _has_constant_mask(self, tree: IterationRangesRoot):
        if not self.optimize_mask:
            return False
        if V.graph.sizevars.statically_known_equals(tree.numel, 1):  # type: ignore[arg-type]
            return True

        # Masks are superfluous if numel is a multiple of BLOCK
        # (We use the fact that BLOCK is required by triton to be a power of 2)
        if tree.is_reduction and self.persistent_reduction:
            max_block = self._get_persistent_RBLOCK(tree.numel)
        elif tree.prefix == "x" and self.no_x_dim:
            max_block = 1
        else:
            max_block = self.max_block(tree.prefix)

        if tree.is_reduction and self.cooperative_reduction:
            max_block = max_block * self.max_rsplit()

        # Optional optimization: if block divides numel exactly, we will
        # never need to do a masked load to handle stragglers at the end.
        # If this tree is for the y dimension, we should only use a constant
        # mask if it can be guaranteed that:
        # 1. (ynumel / YBLOCK) < max_ygrid or
        # 2. (ynumel / YBLOCK) % max_ygrid == 0
        # Because YBLOCK is not constant, use a conservative heuristic:
        # only use a constant mask if ynumel < max_ygrid.
        # It's faster to avoid masking at all.  But it is sound to always
        # mask.
        if V.graph.sizevars.statically_known_multiple_of(tree.numel, max_block):
            return (
                tree.grid_dim != 1
                or tree.has_zdim
                or V.graph.sizevars.statically_known_leq(tree.numel, get_max_y_grid())
            )

        return False

    def filter_masks(self, mask_vars):
        for tree in self.range_trees:
            if self._has_constant_mask(tree):
                mask_vars.discard(f"{tree.prefix}mask")

        # can be added as an override_mask
        mask_vars.discard("None")

    @cache_on_self
    def get_reduction_prefixes(self) -> List[str]:
        return [
            prefix_str[symt]
            for symt in list(TritonSymbols.reduction_types)[: self.num_reduction_dims]
        ]

    def codegen_reduction_numels(self, buffer) -> None:
        """
        Generates code that flattens ND reduction numels, block sizes, etc. into 1D.
        """
        # rnumel = r0_numel * ... * r(n-1)_numel
        reduction_trees = [tree for tree in self.range_trees if tree.is_reduction]
        rnumel = " * ".join(sorted(f"{tree.prefix}numel" for tree in reduction_trees))
        buffer.splice(f"rnumel = {self.kexpr(rnumel)}")

        # RBLOCK = R0_BLOCK * ... * R(N-1)_BLOCK
        rn_blocks = [
            TritonSymbols.block_sizes[tree.symt]
            for tree in self.range_trees
            if tree.is_reduction
        ]
        rblock = sympy_product(rn_blocks)
        buffer.splice(f"RBLOCK: tl.constexpr = {self.kexpr(rblock)}")

    def _get_reduction_symbols(self, suffix: str, **kwargs) -> List[sympy.Symbol]:
        """
        Helper to initialize symbols like rn_numel, rn_base, etc.
        """
        rn_prefixes = self.get_reduction_prefixes()
        return [sympy.Symbol(f"{prefix}{suffix}", **kwargs) for prefix in rn_prefixes]

    @cache_on_self
    def _get_reduction_index_coeffs(self) -> List[sympy.Expr]:
        """
        Compute coefficients to convert ND reduction indices to linear indices.
        For example:
          rindex = r0_index * r1_numel * ... * rn_numel + ... + rn_index.
        """
        rn_prefixes = self.get_reduction_prefixes()
        rn_numels = self._get_reduction_symbols("numel", integer=True, positive=True)
        return [
            sympy_product(rn_numels[idx + 1 :]) for idx in range(len(rn_prefixes) - 1)
        ] + [sympy.Integer(1)]

    def _flatten_reduction_indices(self, multi_inds: List[sympy.Expr]) -> sympy.Expr:
        """
        Compute linear reduction indices from N dimensional ones.
        """
        coeffs = self._get_reduction_index_coeffs()
        return sympy_dot(coeffs, multi_inds)

    def codegen_reduction_indices(self, buffer) -> None:
        """
        Generates code that converts ND reduction indices into linear indices.
        """
        # Gather relevant numels, indices, etc.
        rn_offsets = self._get_reduction_symbols(
            "offset", integer=True, nonnegative=True
        )
        rn_inds = self._get_reduction_symbols("index", integer=True, nonnegative=True)

        # Compute roffset and rindex.
        roffset = self._flatten_reduction_indices(rn_offsets)
        buffer.splice(f"roffset = {self.index_to_str(roffset)}")
        rindex = self._flatten_reduction_indices(rn_inds)
        buffer.splice(f"rindex = {self.index_to_str(rindex)}")

    def iteration_ranges_codegen_header(self, entry, code):
        x = entry.prefix
        if entry.is_loop:
            code.writeline(f"{entry.name} = {x}offset + {x}base")
        elif entry.grid_dim is None:
            # no need to "{x}offset = "
            code.writeline(f"{entry.name} = {self.iteration_ranges_ranges_code(entry)}")
            code.writeline(f"{x}offset = 0")
        else:
            if entry.tensor_dim is not None:
                line = f"{x}offset + {self.iteration_ranges_ranges_code(entry)}"
            else:
                line = self.iteration_ranges_scalar_code(entry, f"{x}offset")
            code.writelines(
                [
                    f"{x}offset = {self.iteration_ranges_get_pid(entry)} * {x.upper()}BLOCK",
                    f"{entry.name} = {line}",
                ]
            )

        if self._has_constant_mask(entry):
            sizes = self.dense_size_str()
            code.writeline(f"{x}mask = tl.full({sizes}, True, tl.int1)")
        else:
            code.writeline(f"{x}mask = {entry.name} < {x}numel")


class TritonScheduling(SIMDScheduling):
    kernel_type: Type[Any] = TritonKernel
    backend_features = dict.fromkeys(  # dict for deterministic order
        [
            BackendFeature.FOREACH,
            BackendFeature.BUCKETIZE,
            BackendFeature.INPLACE_BUFFERS,
            BackendFeature.MASKED_SCATTER_WITH_INDEX,
            BackendFeature.SCAN,
            BackendFeature.TRITON_TEMPLATES,
        ]
    )
    if torch.version.hip is None:
        backend_features.update(
            dict.fromkeys(
                [
                    # TODO: Move this above when ROCm triton adds support for multiple inputs
                    BackendFeature.TUPLE_REDUCTION,
                    BackendFeature.SORT,
                ]
            )
        )

    def __init__(self, scheduler: Scheduler) -> None:
        super().__init__(scheduler)
        if scheduler is None or not hasattr(scheduler, "nodes"):
            return
        for node in scheduler.nodes:
            if isinstance(node, (SchedulerNode, FusedSchedulerNode)):
                node.debug_device_str = debug_triton_code

    @classmethod
    def get_backend_features(cls, device: torch.device):
        if (
            config.triton.cooperative_reductions
            or config.triton.force_cooperative_reductions
        ):
            return {
                **cls.backend_features,
                BackendFeature.REDUCE_TO_SINGLE_ELEMENT: None,
            }
        return cls.backend_features

    def codegen_comment(self, node_schedule):
        wrapper = V.graph.wrapper_code
        origins, _detailed_origins = get_kernel_metadata(node_schedule, wrapper)
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

    def define_kernel(self, src_code, node_schedule, kernel):
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

            _basename, _, kernel_path = get_path(code_hash(src_code.strip()), "py")

            compile_wrapper = IndentedBuffer()
            compile_wrapper.writeline(f"async_compile.triton({subs_name!r}, '''")
            compile_wrapper.splice(src_code, strip=True)
            current_device = V.graph.get_current_device_or_throw()
            compile_wrapper.writeline(f"''', device_str='{current_device.type}')")

            metadata_comment = f"# kernel path: {kernel_path}"
            origins, detailed_origins = get_kernel_metadata(node_schedule, wrapper)
            metadata_comment += "\n" + origins + "\n" + detailed_origins
            wrapper.define_kernel(
                kernel_name, compile_wrapper.getvalue(), metadata_comment
            )

            # log kernel metadata for offline analysis.
            # E.g. one can find all unaligned inner reduction and check if
            # padding helps with the perf kernel by kernel.
            if metrics.is_metric_table_enabled("kernel_metadata"):
                metrics.log_kernel_metadata(kernel_name, kernel_path, src_code)

        return kernel_name

    def benchmark_fused_nodes(self, nodes, n_spills_threshold=8):
        device_interface = get_interface_for_device(V.graph.device_type)
        with preserve_rng_state(), device_interface.device(
            V.graph.get_current_device_or_throw()
        ):  # type: ignore[attr-defined]
            src_code = self.generate_kernel_code_from_nodes(
                nodes, benchmark_kernel=True
            )
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
                    fd.write(str(ms))  # type: ignore[has-type]

            log.debug(
                "kernel src code for %s written to: %s",
                OrderedSet(n.get_name() for n in nodes),
                mod.__file__,
            )
            ms = load_cache()
            if ms is not None:
                return ms, mod.__file__

            args = mod.get_args()
            call = mod.call
            wrapped_jit_function = mod.triton_

            # call once to trigger the compilation
            try:
                call(wrapped_jit_function.clone_args(*args)[0])
            except Exception as e:
                log.debug(
                    "Exception (%s) in compiling fused nodes %s",
                    e,
                    OrderedSet(n.get_name() for n in nodes),
                )
                ms = float("inf")
                store_cache()
                return ms, mod.__file__

            launchers = wrapped_jit_function.launchers
            assert len(launchers) == 1
            # n_spills does not necessarily mean it's not profitable to fuse,
            # and sometimes it can be inaccurate
            if launchers[0].n_spills > n_spills_threshold:
                # skip benchmarking the kernel if there are register spills
                ms = float("inf")
            else:
                # We have to clone the inplace updated arguments to avoid earlier calls
                # generating out of range indices for later calls.
                ms = benchmarker.benchmark_gpu(
                    lambda: call(wrapped_jit_function.clone_args(*args)[0])
                )

                # overhead of cloning args gives bias for fusing the kernel
                # in the case of mutating/in-placeable second fusion
                # TODO - would be better as a hook in triton do_bench that reset
                # the input values between benchmarking
                if len(wrapped_jit_function.mutated_arg_names) > 0:
                    ms = ms - benchmarker.benchmark_gpu(
                        lambda: wrapped_jit_function.clone_args(*args)
                    )

            log.debug(
                "The fused kernel for %s took %.3f ms to run",
                OrderedSet(n.get_name() for n in nodes),
                ms,
            )
            store_cache()
            return ms, mod.__file__

    def create_kernel_choices(
        self, kernel_features, kernel_args, kernel_kwargs
    ) -> List[SIMDKernel]:
        is_scan = kernel_features.contains_op("scan")
        is_split_scan = is_scan and any(
            node.is_split_scan() for node in kernel_features.scheduler_nodes()
        )
        kernel_type: Type[TritonKernel] = self.kernel_type
        if is_split_scan:
            from .triton_split_scan import TritonSplitScanKernel

            kernel_type = TritonSplitScanKernel

        if is_scan:
            # TODO(jansel): scan does not yet work with cooperative reductions
            kernel_kwargs["override_cooperative_reduction"] = False

        # ops.sort only works with persistent reduction, and is not bandwidth bound anyway
        # so taking the hit of non-coalesced loads is okay
        if kernel_features.contains_op("sort"):
            kernel_kwargs["override_persistent_reduction"] = True
            kernel_kwargs["override_cooperative_reduction"] = False

        if not TritonKernel.has_persistent_RBLOCK(kernel_features.reduction_numel):
            # Cannot use persistent reduction with unknown dynamic rnumel
            assert not kernel_kwargs.get("override_persistent_reduction")
            kernel_kwargs["override_persistent_reduction"] = False

        kernel_kwargs = V.choices.triton_kernel_kwargs(
            kernel_type, kernel_features, kernel_args, kernel_kwargs
        )
        kernel = kernel_type(*kernel_args, **kernel_kwargs)
        return self.add_multi_kernel_choices(kernel, kernel_args, kernel_kwargs)

    def add_multi_kernel_choices(
        self,
        kernel: SIMDKernel,
        kernel_args: List[Any],
        kernel_kwargs: Dict[str, Any],
    ) -> List[SIMDKernel]:
        kernels: List[SIMDKernel] = [kernel]
        if not config.triton.multi_kernel:
            return kernels

        optional_persistent = kernel.persistent_reduction and not kernel_kwargs.get(
            "override_persistent_reduction"
        )
        optional_cooperative = kernel.cooperative_reduction and not kernel_kwargs.get(
            "override_cooperative_reduction"
        )
        if optional_persistent:
            kernels.append(
                self.kernel_type(
                    *kernel_args,
                    **kernel_kwargs,
                    override_persistent_reduction=False,
                )
            )
        if optional_cooperative:
            rnumel = kernel.features.reduction_numel
            # for larger sizes non-cooperative gets very slow
            if V.graph.sizevars.statically_known_leq(rnumel, 65536):
                kernels.append(
                    other := self.kernel_type(
                        *kernel_args,
                        **kernel_kwargs,
                        override_cooperative_reduction=False,
                    )
                )
                if optional_persistent and other.persistent_reduction:
                    kernels.append(
                        self.kernel_type(
                            *kernel_args,
                            **kernel_kwargs,
                            override_cooperative_reduction=False,
                            override_persistent_reduction=False,
                        )
                    )

        if len(kernels) > 1:
            for kernel2 in kernels[1:]:
                # Keep buffers needed by the non-persistent reduction so both kernels have the same arguments
                kernel2.must_keep_buffers = kernel.must_keep_buffers
            # persistent kernels must be generated last so must_keep_buffers works right
            kernels.sort(key=lambda k: k.persistent_reduction)
        return kernels

    def benchmark_combo_kernel(self, node_list):
        mod: ModuleType
        ms: int
        ms_clone: int

        def cache_file_path():
            assert mod.__file__ is not None
            return os.path.splitext(mod.__file__)[0] + ".kernel_perf"

        def load_cache():
            path = cache_file_path()
            if os.path.exists(path):
                with open(path) as fd:
                    return tuple(float(e) for e in fd.read().split())
            return (None, None)

        def store_cache():
            path = cache_file_path()
            with open(path, "w") as fd:
                fd.write(str(ms) + " " + str(ms_clone))

        total_ms, file_list = 0, []
        total_clone_ms = 0
        removed_buffers_orig = V.graph.removed_buffers
        V.graph.removed_buffers = OrderedSet(removed_buffers_orig)
        inplaced_to_remove_orig = V.graph.inplaced_to_remove
        V.graph.inplaced_to_remove = OrderedSet(inplaced_to_remove_orig)
        enable_autotune = config.combo_kernels_autotune > 0
        mixed_sizes = config.combo_kernel_allow_mixed_sizes > 0
        kernel_code_list = self.generate_combo_kernel_code(
            subkernel_nodes=node_list,
            custom_part_algorithm=True,
            enable_autotune=enable_autotune,
            mixed_sizes=mixed_sizes,
            only_gen_src_code=True,
        )

        for src_code, _, node_group in kernel_code_list:
            fused_node_lists = [node.get_nodes() for node in node_group]
            names = [n.get_name() for nodes in fused_node_lists for n in nodes]

            src_code = src_code.replace(str(Placeholder.KERNEL_NAME), "triton_")
            mod = PyCodeCache.load(src_code)

            log.debug(
                "kernel src code for %s written to: %s",
                names,
                mod.__file__,
            )
            ms, ms_clone = load_cache()
            if ms is not None:
                total_ms += ms
                total_clone_ms += ms_clone
                file_list.append(mod.__file__)
                continue

            args = mod.get_args()
            call = mod.call
            wrapped_jit_function = mod.triton_

            # call once to trigger the compilation
            call(wrapped_jit_function.clone_args(*args)[0])

            launchers = wrapped_jit_function.launchers
            assert len(launchers) == 1
            if launchers[0].n_spills > 0:
                # skip benchmarking the kernel if there are register spills
                ms = ms_clone = float("inf")
            else:
                # We have to clone the inplace updated arguments to avoid earlier calls
                # generating out of range indices for later calls.
                ms = benchmarker.benchmark_gpu(
                    lambda: call(wrapped_jit_function.clone_args(*args)[0])
                )
                ms_clone = benchmarker.benchmark_gpu(
                    lambda: wrapped_jit_function.clone_args(*args)[0]
                )

            log.debug(
                "The fused kernel for %s took %.3f ms to run, %.3f ms to clone inputs",
                OrderedSet(n.get_name() for n in node_group),
                ms,
                ms_clone,
            )
            store_cache()
            total_ms += ms
            total_clone_ms += ms_clone
            file_list.append(mod.__file__)
        V.graph.removed_buffers = removed_buffers_orig
        V.graph.inplaced_to_remove = inplaced_to_remove_orig
        return total_ms, total_clone_ms, file_list


def debug_triton_code(node: BaseSchedulerNode) -> List[str]:
    lines = []
    multi_template = node.get_template_node()
    assert multi_template is None or isinstance(multi_template, ir.MultiTemplateBuffer)
    if multi_template and multi_template.make_kernel_render is None:
        lines.append(f"{node.get_name()} Unfinalized multi template buffer")
    else:
        from torch._inductor.codegen.cuda_combined_scheduling import (
            CUDACombinedScheduling,
        )

        device = node.get_device()
        assert device is not None
        backend = node.scheduler.get_backend(device)
        assert isinstance(
            backend, (SIMDScheduling, CUDACombinedScheduling)
        ), f"Scheduling backend should be SIMD or CUDACombined when generating debug Triton strings, got: {type(backend)}"

        with V.graph.set_current_device(device):
            # Don't increment kernel count when generating debug string.
            # This will confuse some unit tests that check the number of
            # generated kernels.
            old_generated_kernel_count = metrics.generated_kernel_count
            triton_code = backend.generate_kernel_code_from_nodes(
                node.get_nodes()
            ).strip()
            metrics.generated_kernel_count = old_generated_kernel_count

        lines.append(f"{node.get_name()} Triton code:")
        lines.append(textwrap.indent(triton_code, "    "))
    return lines
