# mypy: allow-untyped-defs
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
from abc import abstractmethod
from collections.abc import Callable, Iterable, Sequence
from functools import lru_cache
from typing import Any, cast, Optional, TYPE_CHECKING, TypeVar, Union

import sympy
from sympy.printing.precedence import PRECEDENCE

import torch
import torch._logging
import torch.utils._pytree as pytree
from torch._dynamo.device_interface import get_interface_for_device
from torch._dynamo.utils import identity, preserve_rng_state
from torch._prims_common import is_integer_dtype
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.functions import CeilDiv, FloorDiv, ModularIndexing
from torch.utils._triton import (
    get_triton_version,
    has_triton_package,
    has_triton_stable_tma_api,
)

from ...utils._sympy.symbol import free_symbol_is_type, prefix_str, symbol_is_type, SymT
from ...utils._sympy.value_ranges import ValueRanges
from .. import config, ir, metrics, utils
from ..async_compile import AsyncCompile
from ..codecache import code_hash, get_path, PyCodeCache, write_atomic
from ..debug import set_kernel_post_grad_provenance_tracing
from ..ops_handler import DefaultHandler
from ..runtime import triton_heuristics
from ..runtime.benchmarking import benchmarker
from ..runtime.hints import (
    AutotuneHint,
    DeviceProperties,
    ReductionHint,
    TRITON_MAX_BLOCK,
    TRITON_MAX_RSPLIT,
)
from ..runtime.runtime_utils import get_max_y_grid, next_power_of_2
from ..scheduler import BaseSchedulerNode, FusedSchedulerNode, Scheduler, SchedulerNode
from ..shape_propagation import get_broadcasted_shape
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
    triton_version_uses_attrs_dict,
    upcast_compute_type,
)
from ..virtualized import _ops as ops, ReductionType, StoreMode, V
from ..wrapper_benchmark import get_kernel_category_by_source_code
from .block_analysis import BlockPatternMatcher
from .common import (
    ArgName,
    BackendFeature,
    ConstexprArg,
    CSE,
    CSEVariable,
    DeferredLine,
    IndentedBuffer,
    InplacedBuffer,
    is_buffer_removed,
    OpOverrides,
    PythonPrinter,
    RemovedArg,
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
    PartialAccumulate,
    SIMDKernel,
    SIMDScheduling,
)
from .triton_utils import (
    config_of,
    equal_1_arg_indices,
    non_constexpr_signature,
    should_unwrap_unspec_arg,
    signature_to_meta,
)
from .wrapper import SymbolicCallArg


if TYPE_CHECKING:
    from types import ModuleType

    from torch._inductor.dtype_propagation import DtypePropagationOpsHandler
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    from ..ir import IRNode
    from .common import BlockShapeType
    from .simd_kernel_features import SIMDKernelFeatures

    _T = TypeVar("_T")

log = logging.getLogger(__name__)
perf_hint_log = torch._logging.getArtifactLogger(__name__, "perf_hints")
schedule_log = torch._logging.getArtifactLogger(__name__, "schedule")
fusion_log = torch._logging.getArtifactLogger(__name__, "fusion")
async_compile = AsyncCompile()

# Threshold for detecting inner reductions based on tiling score ratio.
# If r0_tiling_score / x_tiling_score >= this value, upgrade DEFAULT hint to INNER.
INNER_REDUCTION_RATIO_THRESHOLD = 8


def get_triton_reduction_function(reduction_type):
    use_helper = reduction_type in ("any", "max", "min", "prod")
    module = "triton_helpers" if use_helper else "tl"
    if reduction_type in ("max", "min"):
        return f"{module}.{reduction_type}2"
    else:
        return f"{module}.{reduction_type}"


def is_sympy_integer_like(expr: object):
    """ "
    Is this expression a Sympy Integer or is it an integer sympy Expr
    containing no free symbols. The latter case can happen with Identity expr.
    """
    if not isinstance(expr, sympy.Expr):
        return False
    return isinstance(expr, sympy.Integer) or (
        expr.is_integer and len(expr.free_symbols) == 0
    )


class OpDtypeSupport:
    """
    Some Triton ops such as libdevice and tl.math only support float32 and float64.
    This class records which dtypes are supported by specific IR ops.
    """

    supported_dtypes: dict[str, OrderedSet[torch.dtype]] = {}
    convert_outputs: dict[str, bool] = {}

    @classmethod
    def register_upcast(cls, func: Callable[..., str], convert_output: bool) -> None:
        op_name = func.__name__
        cls.supported_dtypes[op_name] = OrderedSet([torch.float32, torch.float64])
        cls.convert_outputs[op_name] = convert_output


@lru_cache(None)
def gen_attr_descriptor_import() -> str:
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
def gen_common_triton_imports() -> str:
    imports = IndentedBuffer()
    imports.splice(
        """
        import triton
        import triton.language as tl
        """
    )
    try:
        import triton.language.extra.tlx  # noqa: F401

        imports.splice(
            """
           import triton.language.extra.tlx as tlx  # noqa: F401
           """
        )
    except ImportError:
        pass
    if attr_desc := gen_attr_descriptor_import():
        imports.writeline(attr_desc)

    imports.splice(
        """
        from torch._inductor.runtime import triton_helpers, triton_heuristics
        from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
        from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
        """
    )
    if config.triton.proton_profiling:
        imports.splice(
            """
            import triton.profiler as proton
            import triton.profiler.language as pl
            pl.enable_semantic('triton')
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
    def get_block_shape(cls, expr: sympy.Expr) -> BlockShapeType:
        # return block shape of sympy Expression
        # e.g.,
        # tmp13 = y1
        # tmp14 = x0 - tmp13
        #
        # get_block_shape(y1) = (YBLOCK,1,1)
        # get_block_shape(x0-tmp13) = (YBLOCK,XBLOCK,1)

        expr_shape: BlockShapeType = ()
        expr_vars = expr.free_symbols
        for var in expr_vars:
            if symbol_is_type(var, SymT.TMP):
                cse_var = V.kernel.cse.varname_map[var.name]
                var_shape = cse_var.shape
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
                var_shape = ()
            else:
                symbol_matches = [
                    symt for symt in cls.block_types if symbol_is_type(var, symt)
                ]
                assert len(symbol_matches) == 1, f"Ambiguous type: {var.name}"

                sym = symbol_matches[0]
                ndim = V.kernel.triton_tensor_ndim()
                shape = ["1"] * ndim

                tree_match = [
                    tree
                    for tree in V.kernel.active_range_trees()
                    if prefix_str[sym] == tree.prefix
                ]
                assert len(tree_match) == 1, "# of Match expected to 1"

                shape[tree_match[0].tensor_dim] = str(cls.get_block_size(tree_match[0]))
                var_shape = tuple(shape)

            # Union current variable shape
            expr_shape = get_broadcasted_shape(expr_shape, var_shape)

        assert expr_shape is not None

        return expr_shape

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
    expand_str: Optional[str]
    _has_rindex: bool
    index: sympy.Expr
    expand_shape: Optional[Sequence[Union[int, str]]]

    def has_mask(self) -> bool:
        return bool(self.mask_vars)

    def has_indirect(self) -> bool:
        return free_symbol_is_type(self.index, SymT.TMP)

    def has_rindex(self) -> bool:
        return self._has_rindex

    def has_tmpmask(self) -> bool:
        return any(str(mask).startswith("tmp") for mask in self.mask_vars)

    def has_rmask(self) -> bool:
        return any(str(mask).startswith("r") for mask in self.mask_vars)

    @property
    def mask_str(self) -> str:
        # The sorted call is added to make sure the order is still
        # deterministic if self.mask_vars contains mix of string
        # and TritonCSEVariable
        return (
            " & ".join(sorted(map(str, self.mask_vars))) if self.mask_vars else "None"
        )


@dataclasses.dataclass
class BlockDescriptorOptions:
    """
    This is a base class that describes a block descriptor used in Triton kernels.
    It can be used to create either a tensor descriptor (with TensorDescriptorOptions)
    or a block pointer (with BlockPtrOptions).
    """

    params: BlockParameters
    constant_offset: sympy.Expr
    order: list[int]
    mask_vars: OrderedSet[str]
    broadcast_shape: Sequence[sympy.Expr]
    broadcasting_dims: list[bool]
    final_shape: Sequence[sympy.Expr]
    # If the BlockParameters have been sorted using a particular stride order
    # transpose load / store blocks at runtime using the information in
    # stride_sorter.
    stride_sorter: BlockParameters.StrideSorter
    _boundary_check: Optional[list[int]] = None
    # Can we safely lift the constructor
    # to the top of the kernel?
    can_lift: bool = False

    @property
    def shape(self) -> list[sympy.Expr]:
        return self.params.shape

    @property
    def block_shape(self) -> list[sympy.Expr]:
        return self.params.block_shape

    @property
    def strides(self) -> list[sympy.Expr]:
        return self.params.strides

    @property
    def offsets(self) -> list[sympy.Expr]:
        return self.params.offsets

    @classmethod
    def create(
        cls,
        *,
        params: BlockParameters,
        constant_offset: sympy.Expr,
        range_trees: list[IterationRangesRoot],
        mask_vars: OrderedSet[str],
        get_max_block: Callable[[str], int],
        stride_sorter_cls: type[BlockParameters.StrideSorter],
        can_lift: bool = False,
    ) -> BlockDescriptorOptions:
        """Helper to create a BlockDescriptorOptions instance"""

        sizevars = V.graph.sizevars

        def lookup_size(exprs: Iterable[sympy.Expr]) -> list[sympy.Expr]:
            return [sizevars.lookup_precomputed_size(expr) for expr in exprs]

        # Look up precomputed sizes
        params.shape = lookup_size(params.shape)
        params.strides = lookup_size(params.strides)

        # Strip out dimensions of size 1.
        # Size 1 dimensions are redundant since the triton kernel shape
        # will be e.g. [YBLOCK, XBLOCK], so tl.reshape would just remove these
        # dimensions anyway
        singleton_dims = [
            sizevars.statically_known_equals(dim, 1) for dim in params.block_shape
        ]
        if all(singleton_dims):
            # Handle a pure singletons, e.g. [1, 1]
            singleton_dims[-1] = False

        # Drop singleton dimensions from the block descriptor.
        params = params.remove_dims(singleton_dims)

        # Maybe reorder dimensions based on strides
        # with tl.trans applied at load / store time
        params, stride_sorter = params.maybe_sort_with_stride_order(
            stride_sorter_cls=stride_sorter_cls, shape_env=V.graph._shape_env
        )

        # Strip out dimensions of stride 0.
        # These will be restored with tl.broadcast_to.
        broadcasting_dims = [
            sizevars.statically_known_equals(stride, 0) for stride in params.strides
        ]

        # Record the post-broadcast shape before broadcasting dims are removed.
        # The pre-broadcast shape is identical to this, except broadcasting dims are
        # replaced with 1.
        broadcast_shape = params.block_shape

        # Drop broadcasting dims from the block descriptor.
        params = params.remove_dims(broadcasting_dims)

        # Compute the final shape, adjusting for special kernel types.
        final_shape = [TritonSymbols.get_block_size(tree) for tree in range_trees]
        if V.kernel.no_x_dim:
            assert range_trees[0].prefix == "x"
            final_shape.pop(0)

        reduction_ndim = V.kernel.num_reduction_dims
        if (
            not V.kernel.inside_reduction
            and len(params.strides) == len(V.kernel.numels) - reduction_ndim
            and V.kernel.features.is_reduction()
        ):
            # Need to expand rank to match the rank used inside the reduction loop
            final_shape += [sympy.S.One] * reduction_ndim

        try:
            # Get permutation to sort strides in ascending order.
            # This is used as the order argument in tl.make_block_ptr
            order = utils.argsort_sym(V.graph._shape_env, params.strides)
        except AssertionError:
            # Symbolic shapes, failed to evaluate comparison expression
            order = list(reversed(range(len(params.strides))))

        result = cls(
            params=params,
            constant_offset=V.graph.sizevars.lookup_precomputed_size(constant_offset),
            order=order,
            mask_vars=mask_vars,
            final_shape=final_shape,
            broadcast_shape=broadcast_shape,
            broadcasting_dims=broadcasting_dims,
            stride_sorter=stride_sorter,
            can_lift=can_lift,
        )
        result.compute_boundary_check(get_max_block, range_trees)
        return result

    def replace_offset(
        self, expr: sympy.Expr, replacement: sympy.Expr, symt: SymT
    ) -> sympy.Expr:
        """
        Replaces instances of {symt}_offset with the new expression.
        """
        roffset = TritonSymbols.block_offsets[symt]
        return sympy_subs(expr, {roffset: replacement})

    def remove_roffsets(self, expr: sympy.Expr) -> sympy.Expr:
        for symt in TritonSymbols.reduction_types:
            expr = self.replace_offset(expr, sympy.Integer(0), symt)
        return expr

    def compute_boundary_check(
        self,
        get_max_block: Callable[[str], int],
        range_trees: list[IterationRangesRoot],
    ) -> None:
        """List of indices to pass to tl.load(boundary_check=...)"""
        sizevars = V.graph.sizevars

        # Substitute maximum block sizes in shape expressions.
        # This works in multiple_of checks because block sizes are powers of 2.
        block_to_max: dict[sympy.Expr, Any] = {
            TritonSymbols.block_sizes[t.symt]: get_max_block(prefix_str[t.symt])
            for t in range_trees
        }

        # Also see Note: Constant mask optimisation
        # if ynumel / YBLOCK > max_ygrid, then the z dimension is used to handle
        # the remaining programs that cannot fit into the y dimension. This means
        # it's possible that more than the required number of programs are launched,
        # possibly leading to out-of-bounds accesses. So even if ynumel divides YBLOCK,
        # boundary checking is required in the dimensions that are based on YBLOCK
        # e.g. for [YBLOCK // 16, YBLOCK, XBLOCK] dimensions 0 and 1 need boundary
        # checks when max_ygrid is exceeded.
        needs_overflow_grid = any(map(V.kernel.needs_yz_grid_overflow, range_trees))
        self._boundary_check = [
            idx
            for idx in range(len(self.shape))
            if (
                not sizevars.statically_known_equals(self.strides[idx], sympy.S.Zero)
                and (
                    (
                        needs_overflow_grid
                        and TritonSymbols.block_sizes[SymT.YBLOCK]
                        in self.block_shape[idx].free_symbols
                    )
                    or (
                        not sizevars.statically_known_multiple_of(
                            self.shape[idx], self.block_shape[idx]
                        )
                        and not sizevars.statically_known_multiple_of(
                            self.shape[idx],
                            sympy_subs(self.block_shape[idx], block_to_max),
                        )
                    )
                )
                and not (
                    V.kernel.no_x_dim
                    and self.block_shape[idx] == TritonSymbols.block_sizes[SymT.XBLOCK]
                )
            )
        ]

    def boundary_check(self) -> list[int]:
        assert self._boundary_check is not None
        return self._boundary_check

    def has_indirect(self) -> bool:
        return False  # block_ptr can't do indirect indexing

    def has_rindex(self) -> bool:
        return any(
            free_symbol_is_type(expr, TritonSymbols.reduction_types)
            for expr in self.block_shape
        )

    def has_rmask(self) -> bool:
        return self.has_rindex()

    def has_tmpmask(self) -> bool:
        return False  # block_ptr can't do indirect indexing

    def has_mask(self) -> bool:
        return bool(self.boundary_check())

    def codegen_broadcast_and_reshape(
        self,
        value: str,
        initial_shape: Sequence[sympy.Expr],
        final_shape: Sequence[sympy.Expr],
        allow_implicit: bool,
        for_store: bool,
    ) -> str:
        """
        Generate a broadcast and a reshape for the block descriptor.
        This restores stride-0 dimensions which were removed from the block descriptor.

        Transposes are also applied to the input using self.stride_sorter:
        if for_store is True:
            - First Broadcast the value. Since self.broadcast_shape is stored in
            descending stride order, it must be reverted to the original order
            since the input value does not have dims with descending strides
            - After, transpose the broadcasted value so that dimensions are in
            descending stride order
            - Finally reshape to the block shape
        else (for load):
            - First broadcast the value to self.broadcast_shape (strides are descending)
            - Then transpose the value so that dimensions no longer have descending strides
            - Finally reshape the block to the final kernel tile shape
        """
        broadcast_shape = self.broadcast_shape
        broadcasting_dims = self.broadcasting_dims

        # If the block parameters have been sorted by descending strides,
        # permute the broadcasting parameters so that they are compatible
        # with the value being stored. This is because the dimensions
        # of the value being stored are not sorted in descending stride order,
        # but the broadcasting parameters are based on the dims in sorted order
        if for_store:
            broadcast_shape = self.stride_sorter.revert(self.broadcast_shape)
            broadcasting_dims = self.stride_sorter.revert(self.broadcasting_dims)

        # Reshape to add singletons.
        pre_broadcast_shape = [
            sympy.S.One if is_broadcasting else dim
            for dim, is_broadcasting in zip(broadcast_shape, broadcasting_dims)
        ]
        value = triton_reshape(value, initial_shape, pre_broadcast_shape)

        if (
            not self.stride_sorter.is_identity
            and not for_store
            and len(pre_broadcast_shape) == len(final_shape)
        ):
            # If all we need to do is transpose to match the final shape
            # with implicit broadcasting then we don't need an explicit broadcast
            # unless the caller requests it. So just test implicit broadcast support
            # with the transposed pre broadcast shape
            pre_broadcast_shape = self.stride_sorter.revert(pre_broadcast_shape)

        # Broadcast singletons.
        # For loads, we can often implicitly broadcast singleton dimensions.
        # We need an explicit broadcast for stores, or if the final reshape does more
        # than add singletons.
        sizevars = V.graph.sizevars
        supports_implicit_broadcast = allow_implicit and (
            len(pre_broadcast_shape) == len(final_shape)
            and all(
                sizevars.statically_known_equals(pre_dim, 1)
                or sizevars.statically_known_equals(pre_dim, post_dim)
                for pre_dim, post_dim in zip(pre_broadcast_shape, final_shape)
            )
        )

        if any(self.broadcasting_dims) and not supports_implicit_broadcast:
            value = (
                f"tl.broadcast_to({value}, {V.kernel.index_to_str(broadcast_shape)})"
            )

        old_shape = self.broadcast_shape
        if not self.stride_sorter.is_identity:
            # if for_store the transform is
            #   (non-descending strides) broadcasted kernel tile shape
            #       -> (descending strides) block descriptor shape
            # o/w if loading the transform is
            #   (descending strides) ((maybe implicitly) broadcasted block shape
            #       -> (non-descending) (maybe implicitly) broadcasted kernel tile shape
            permute_dims = (
                self.stride_sorter.sort_idx
                if for_store
                else self.stride_sorter.revert_sort_idx
            )
            value = f"tl.trans({value}, {permute_dims})"
            old_shape = (
                self.broadcast_shape
                if for_store
                else self.stride_sorter.revert(self.broadcast_shape)
            )

        # Reshape to the final shape.
        value = triton_reshape(value, old_shape, final_shape)

        return value


@dataclasses.dataclass
class TensorDescriptorOptions(BlockDescriptorOptions):
    def format(self, name: str, roffset=True) -> str:
        """
        Codegen a call to tl.make_tensor_descriptor()

        Args:
            name: variable name for pointer
            roffset: unused, but kept for compatibility with BlockPtrOptions.format()

        Returns:
            "tl.make_tensor_descriptor(...)"
        """

        f = V.kernel.index_to_str
        args = [
            (
                f"{name} + ({f(self.constant_offset)})"
                if self.constant_offset != 0
                else name
            ),
            f"shape={f(self.shape)}",
            f"strides={f(self.strides)}",
            f"block_shape={f(self.block_shape)}",
        ]

        return f"tl.make_tensor_descriptor({', '.join(args)})"


@dataclasses.dataclass
class BlockPtrOptions(BlockDescriptorOptions):
    def replace_offset(
        self, expr: sympy.Expr, replacement: sympy.Expr, symt: SymT
    ) -> sympy.Expr:
        """
        Replaces instances of {symt}_offset with the new expression.
        """
        roffset = TritonSymbols.block_offsets[symt]
        return sympy_subs(expr, {roffset: replacement})

    def remove_roffsets(self, expr: sympy.Expr) -> sympy.Expr:
        for symt in TritonSymbols.reduction_types:
            expr = self.replace_offset(expr, sympy.Integer(0), symt)
        return expr

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
            offsets = [self.remove_roffsets(offset) for offset in offsets]
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

    def advance_roffset(self, symt: SymT) -> sympy.Expr:
        """
        Codegen string to pass to tl.advance(name, ...).

        Advance is the difference between offsets in each loop iteration.
        To compute it, we replace rN_offset with multiples of RN_BLOCK.
        Since we expect rN_offset to vary in range(0, rN_numel, RN_BLOCK), the first
        iteration has rN_offset=0, while the second has rN_offset=RN_BLOCK.
        """
        rblock = TritonSymbols.block_sizes[symt]
        advance = [
            (
                self.replace_offset(offset, rblock, symt)
                - self.replace_offset(offset, sympy.S.Zero, symt)
            )
            for offset in self.offsets
        ]
        return advance


def triton_reshape(
    value: str, old_shape: Sequence[sympy.Expr], new_shape: Sequence[sympy.Expr]
) -> str:
    """Workaround https://github.com/triton-lang/triton/issues/2836"""
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
    def _print_TruncToInt(self, expr: sympy.Expr) -> str:
        assert len(expr.args) == 1
        return (
            f"libdevice.trunc({self._print(expr.args[0])}).to({V.kernel.index_dtype})"
        )

    def _print_Float(self, expr: sympy.Expr) -> str:
        if expr.is_integer:
            # sympy considers 0.0 to be integer, but triton doesn't.
            # this workaround prints the float as an integer
            # xref: https://github.com/sympy/sympy/issues/26620
            ret = str(int(expr))
        elif config.is_fbcode() and torch.version.hip:
            ret = f"{expr}"
        else:
            ret = f"tl.full([], {expr}, tl.float64)"
        return ret

    def _print_ToFloat(self, expr: sympy.Expr) -> str:
        assert len(expr.args) == 1
        # pyrefly: ignore [bad-argument-type]
        s = self.parenthesize(expr.args[0], PRECEDENCE["Atom"] - 0.5)
        return f"{s}.to(tl.float64)"

    def _print_PythonMod(self, expr: sympy.Expr) -> str:
        quot, div = expr.args
        if quot.is_nonnegative and div.is_nonnegative:
            return self.stringify(expr.args, " % ", PRECEDENCE["Atom"] - 0.5)
        quot_s = self._print(quot)
        div_s = self._print(div)
        return f"triton_helpers.remainder_integer({quot_s}, {div_s})"

    def _print_FloorDiv(self, expr: sympy.Expr) -> str:
        assert expr.is_integer
        quot, div = expr.args
        if quot.is_nonnegative and div.is_nonnegative:
            return self.stringify(expr.args, " // ", PRECEDENCE["Atom"] - 0.5)
        quot_s = self._print(quot)
        div_s = self._print(div)
        return f"triton_helpers.div_floor_integer({quot_s},  {div_s})"

    # TODO: This is wrong, when lhs, rhs > 2**53, Python does a higher
    # precision algorithm, which we would need to replicate here
    def _print_IntTrueDiv(self, expr: sympy.Expr) -> str:
        return self.stringify(expr.args, " / ", PRECEDENCE["Atom"] - 0.5)

    # NB: sympy.floor/ceiling produce integers, so we have to do the
    # conversion to index dtype
    def _print_floor(self, expr: sympy.Expr) -> str:
        assert len(expr.args) == 1
        return (
            f"libdevice.floor({self._print(expr.args[0])}).to({V.kernel.index_dtype})"
        )

    def _print_FloorToInt(self, expr: sympy.Expr) -> str:
        assert len(expr.args) == 1
        return (
            f"libdevice.floor({self._print(expr.args[0])}).to({V.kernel.index_dtype})"
        )

    def _print_ceiling(self, expr: sympy.Expr) -> str:
        assert len(expr.args) == 1
        return f"libdevice.ceil({self._print(expr.args[0])}).to({V.kernel.index_dtype})"

    def _print_CeilToInt(self, expr: sympy.Expr) -> str:
        assert len(expr.args) == 1
        return f"libdevice.ceil({self._print(expr.args[0])}).to({V.kernel.index_dtype})"

    def _helper_sqrt(self, expr: sympy.Expr) -> str:
        return f"tl.sqrt_rn(({self._print(expr)}).to(tl.float32))"

    def _print_FloatPow(self, expr: sympy.Expr) -> str:
        return (
            f"libdevice.pow({self._print(expr.args[0])}, {self._print(expr.args[1])})"
        )

    def _print_PowByNatural(self, expr: sympy.Expr) -> str:
        if expr.args[0].is_Integer:
            return f"libdevice.pow({float(expr.args[0])}, {self._print(expr.args[1])})"
        return (
            f"libdevice.pow({self._print(expr.args[0])}, {self._print(expr.args[1])})"
        )

    def _print_Where(self, expr: sympy.Expr) -> str:
        c = self.doprint(expr.args[0])
        p = self.doprint(expr.args[1])
        q = self.doprint(expr.args[2])
        return f"tl.where({c}, {p}, {q})"

    def _print_min_max_helper(self, expr: sympy.Expr, cmp: str) -> str:
        """
        Helper for max/min code generation.
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

    def _print_Min(self, expr: sympy.Expr) -> str:
        return self._print_min_max_helper(expr, "<")

    def _print_Max(self, expr: sympy.Expr) -> str:
        return self._print_min_max_helper(expr, ">")

    def _print_Abs(self, expr: sympy.Expr) -> str:
        assert len(expr.args) == 1
        return f"tl_math.abs({self._print(expr.args[0])})"

    def _print_OpaqueUnaryFn_cos(self, expr: sympy.Expr) -> str:
        assert len(expr.args) == 1
        return f"libdevice.cos(({self._print(expr.args[0])}).to(tl.float32))"

    def _print_OpaqueUnaryFn_cosh(self, expr: sympy.Expr) -> str:
        assert len(expr.args) == 1
        return f"libdevice.cosh(({self._print(expr.args[0])}).to(tl.float32))"

    def _print_OpaqueUnaryFn_acos(self, expr: sympy.Expr) -> str:
        assert len(expr.args) == 1
        return f"libdevice.acos(({self._print(expr.args[0])}).to(tl.float32))"

    def _print_OpaqueUnaryFn_sin(self, expr: sympy.Expr) -> str:
        assert len(expr.args) == 1
        return f"libdevice.sin(({self._print(expr.args[0])}).to(tl.float32))"

    def _print_OpaqueUnaryFn_sinh(self, expr: sympy.Expr) -> str:
        assert len(expr.args) == 1
        return f"libdevice.sinh(({self._print(expr.args[0])}).to(tl.float32))"

    def _print_OpaqueUnaryFn_asin(self, expr: sympy.Expr) -> str:
        assert len(expr.args) == 1
        return f"libdevice.asin(({self._print(expr.args[0])}).to(tl.float32))"

    def _print_OpaqueUnaryFn_tan(self, expr: sympy.Expr) -> str:
        assert len(expr.args) == 1
        return f"libdevice.tan(({self._print(expr.args[0])}).to(tl.float32))"

    def _print_OpaqueUnaryFn_tanh(self, expr: sympy.Expr) -> str:
        assert len(expr.args) == 1
        return f"libdevice.tanh(({self._print(expr.args[0])}).to(tl.float32))"

    def _print_OpaqueUnaryFn_atan(self, expr: sympy.Expr) -> str:
        assert len(expr.args) == 1
        return f"libdevice.atan(({self._print(expr.args[0])}).to(tl.float32))"

    def _print_OpaqueUnaryFn_log2(self, expr: sympy.Expr) -> str:
        assert len(expr.args) == 1
        return f"libdevice.log2(({self._print(expr.args[0])}).to(tl.float32))"

    def _print_RoundToInt(self, expr: sympy.Expr) -> str:
        assert len(expr.args) == 1
        return (
            f"libdevice.llrint({self._print(expr.args[0])}).to({V.kernel.index_dtype})"
        )

    def _print_RoundDecimal(self, expr: sympy.Expr) -> str:
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


def low_precision_fp(dtype: torch.dtype) -> bool:
    return dtype.itemsize <= 2 and dtype.is_floating_point


def low_precision_fp_var(var: Union[CSEVariable, Any]) -> bool:
    if not isinstance(var, CSEVariable):
        return False

    dtype = var.dtype
    return low_precision_fp(dtype) if isinstance(dtype, torch.dtype) else False


class TritonCSEVariable(CSEVariable):
    def __init__(
        self,
        name: str,
        bounds: ValueRanges[Any],
        dtype: torch.dtype,
        shape: BlockShapeType = None,
    ) -> None:
        super().__init__(name, bounds, dtype, shape=shape)
        # We'll use this to track which masks the variable needs when used for indirect indexing
        self.mask_vars: OrderedSet[str] = OrderedSet()
        assert dtype is not None, "TritonCSEVariable must have dtype"
        assert shape is not None, "TritonCSEVariable must have shape"

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


def maybe_upcast_float32(convert_output: bool = True) -> Callable[[_T], _T]:
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

    return decorator  # type: ignore[return-value]


class TritonOverrides(OpOverrides):
    """Map element-wise ops to Triton e.g., ops.to_dtype(x,...) -> x.to(...)"""

    _LOG_2_E = math.log2(math.e)

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
            # https://github.com/triton-lang/triton/blob/10f59d8ce04052521c1bc0cb3a3f8b98918fc7e3/lib/Conversion/TritonGPUToLLVM/ElementwiseOpToLLVM.cpp#L10.
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
        elif dtype == torch.uint8 and (
            src_dtype is not None and src_dtype.is_floating_point or src_dtype is None
        ):
            # to work around llvm uint conversion semantics that produces 0's for negative
            # values when converting from floating types.
            # optimization - if source type is known and it's not a floating type, then
            # do not apply conversion to the intermediate type.
            return f"{x}.to(tl.int16).to(tl.uint8)"

        if use_compute_types:
            out_dtype = triton_compute_type(dtype)
        else:
            out_dtype = triton_store_type(dtype)

        return f"{x}.to({out_dtype})"

    @staticmethod
    def to_dtype_bitcast(x, dtype: torch.dtype, src_dtype: torch.dtype):
        assert src_dtype.itemsize == dtype.itemsize
        # We may promote float16 or bfloat16 to float32 and cause the
        # bitwidth of dtype to be different from the input tensor (i.e. float32).
        # In such as case, we will have to convert the input tensor to
        # its src_type, perform bitcast, and then convert the bit-casted
        # tensor back to float to ensure we use values with the right precision.
        if x.dtype != src_dtype:
            x = f"{x}.to({triton_type(src_dtype)})"

        out = f"{x}.to({triton_type(dtype)}, bitcast=True)"
        if upcast_compute_type(dtype) != dtype:
            out = f"{out}.to({triton_type(upcast_compute_type(dtype))})"

        return out

    @staticmethod
    def _shaped_constant(value, dtype, shape):
        type_ = torch._prims_common.dtype_to_type(dtype)
        triton_val = constant_repr(type_(value))
        triton_type = triton_compute_type(dtype)

        # NOTE: We use tl.full here to get the expected type.
        # Otherwise, subnormal float32 values are treated as fp64
        # causing fp32 * fp64 promotion and different numerical results.
        if value < 0 and not dtype.is_signed:
            triton_signed_type = f"tl.{triton_type[4:]}"
            return f"tl.full({shape}, {triton_val}, {triton_signed_type}).to({triton_type})"
        else:
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
        x_dtype = getattr(x, "dtype", None)
        y_dtype = getattr(y, "dtype", None)

        if (
            x_dtype == torch.float32
            and y_dtype == torch.float32
            and config.emulate_divison_rounding
        ):
            # x / y in Triton is lowered to div.full which is approx
            # we want div_rn to adhere with eager
            out = f"triton.language.div_rn({x}, {y})"
        else:
            out = f"({x} / {y})"

        if low_precision_fp_var(x) or low_precision_fp_var(y):
            out_dtype = get_dtype_handler().truediv(x, y)
            if out_dtype in (torch.float16, torch.float32):
                out = f"{out}.to({triton_type(out_dtype)})"

        return out

    @staticmethod
    def mod(x, y):
        out = f"({x} % {y})"
        if low_precision_fp_var(x) or low_precision_fp_var(y):
            out_dtype = get_dtype_handler().mod(x, y)
            if out_dtype in (torch.float16, torch.float32):
                out = f"{out}.to({triton_type(out_dtype)})"
        return out

    @staticmethod
    @maybe_upcast_float32()
    def exp(x):
        """
        When use_fast_math, use the ftz (flushing to zero) variant
        of exponent computation.

        Check https://github.com/triton-lang/triton/issues/5735 for
        more details.
        """
        if config.use_fast_math:
            return f"tl_math.exp({x})"
        else:
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
        return f"tl.sqrt_rn({x})"

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
        if torch.version.hip:
            return f"tl.minimum({a}, {b}, tl.PropagateNan.ALL)"
        else:
            return f"triton_helpers.minimum({a}, {b})"

    @staticmethod
    def maximum(a, b):
        if torch.version.hip:
            return f"tl.maximum({a}, {b}, tl.PropagateNan.ALL)"
        else:
            return f"triton_helpers.maximum({a}, {b})"

    @staticmethod
    def where(a, b, c):
        return f"tl.where({a}, {b}, {c})"

    @staticmethod
    def dot(a, b):
        """
        Triton code generation for lowering ops.dot to tl.dot.

        The logic is as follows:

        1. Downcasting for performance
           If the data was previously upcasted to fp32, we downcast back to the
           original dtype (e.g., fp16 or bf16) for better performance. While
           surrounding operations may run in fp32, matmul itself is executed at the
           original precision to optimize throughput.

        2. Handling non-constant reduction masks
           If the reduction mask is not constant and there was any operation between
           tl.load and tl.dot, we zero out regions outside the mask using
           tl.where(r0_mask, val, 0).
           This ensures that values outside the mask do not contribute to the dot
           product, preventing incorrect results.

        3. Shape alignment for tl.dot
           We massage shapes to match the tl.dot requirement of (Y, R) x (R, X).
           Current codegen eagerly broadcasts tl.arange to create unique axes. We
           reshape, transpose, or broadcast to align with the (Y, R) x (R, X) shape.
           We avoid using 3D dot ((Z, Y, R) x (Z, R, X)) because 3D tl.dot has
           poor performance. During batched matmul (bmm), we keep ZBLOCK=1 and call
           the 2D dot kernel instead.
        """
        assert V.kernel.is_native_matmul
        orig_a, orig_b = a, b

        def is_where_needed(var):
            # Skip if the variable doesn't have a reduction mask
            if not any(map(prefix_is_reduction, var.mask_vars)):
                return False

            reduction_range = V.kernel.range_trees[-1]
            assert reduction_range.is_reduction

            # Skip if reduction mask was already constant
            if V.kernel._has_constant_mask(reduction_range):
                return False

            # Skip if the variable is already zeroed outside the mask
            # (e.g., from tl.load(..., other=0.0))
            # TODO : track the value of outside of mask region with cse
            for k, v in V.kernel.cse._cache.items():
                if v == var and "tl.load" in k and "other=0.0" in k:
                    return False

            return True

        def where_cond(var):
            default = ir.Reduction.default_value("dot", var.dtype)
            reduction_mask = [
                f"{tree.prefix}mask"
                for tree in V.kernel.range_trees
                if tree.is_reduction
            ]

            assert len(reduction_mask) == 1, "don't tile reduction when native matmul"

            where_var = TritonKernelOverrides.where(reduction_mask[0], var, default)
            return V.kernel.cse.generate(
                V.kernel.compute, where_var, dtype=var.dtype, shape=var.shape
            )

        # When computing expressions like ((A+1) @ (B+2)),
        # native codegen will do
        #
        # a = tl.load(..., r0_mask, other=0.0)
        # b = tl.load(..., r0_mask, other=0.0)
        # tmp0 = a+1
        # tmp1 = b+2
        # tmp2 = tl.dot(tmp0, tmp1)
        #
        # This produces incorrect results because outside of r0_mask is not zero.
        # So before calling tl.dot, apply tl.where to zero out values properly.
        # TODO: Optimize - We don't need both operands to be zeroed except NaN * 0
        if is_where_needed(orig_a):
            a = where_cond(a)
        if is_where_needed(orig_b):
            b = where_cond(b)

        def reshape_transpose_broadcast_for_dot(
            value,
            initial_shape: Sequence[sympy.Expr],
            final_shape: Sequence[sympy.Expr],
        ) -> str:
            """
            Generate a reshape, transpose, and broadcast for the tl.dot.
            tl.dot requires specific shape requirement : (Y,R) x (R,X)
            but the current triton codegen eagerly broadcast the tl.arange so
            it needs to be reshaped to meet the requirement.

            This is done by three steps.
            1. remove the empty dimension (dim with size 1) and make it 2d with tl.reshape
            2. permute the dimension if needed (e.g., (X,R) -> (R,X)) with tl.trans
            3. broadcast if needed with broadcast_to.
                - This shows up when matmul operand is broadcasted with torch.expand/repeat.
                - e.g., torch.rand((16,)).expand(16,16) @ B

            e.g., (Y,1,R), (Y,R) -> tl.reshape(var, (Y,R))
            e.g., (1,X,R), (R,X) -> tl.trans(tl.reshape(var, (X,R)))
            e.g., (1,X,1), (R,X) -> tl.broadcast_to(tl.trans(tl.reshape(var, (X,1))), (R,X))

            TODO : eventually we want to remove this function when lazy broadcasting arrives
            """

            # Triton 3d dot is slower than 2d dot, so we want to keep block shape in 2d
            # by fixing ZBLOCK=1 in the autotune config
            if ZBLOCK in initial_shape:
                initial_shape = ["1" if dim == ZBLOCK else dim for dim in initial_shape]

            if final_shape == [YBLOCK, RBLOCK]:
                assert XBLOCK not in initial_shape, (
                    "left tl.dot operand cannot depend on x"
                )

                shape_2d = ["1", "1"]
                if YBLOCK in initial_shape:
                    shape_2d[0] = YBLOCK
                if RBLOCK in initial_shape:
                    shape_2d[1] = RBLOCK

                # reshape it into 2d
                value = triton_reshape(value, initial_shape, shape_2d)

                # broadcast if needed
                broadcast_needed = shape_2d != [YBLOCK, RBLOCK]
                if broadcast_needed:
                    value = f"tl.broadcast_to({value}, ({YBLOCK}, {RBLOCK}))"

            elif final_shape == [RBLOCK, XBLOCK]:
                assert YBLOCK not in initial_shape, (
                    "right tl.dot operand cannot depend on y"
                )

                shape_2d = ["1", "1"]
                if XBLOCK in initial_shape:
                    shape_2d[0] = XBLOCK
                if RBLOCK in initial_shape:
                    shape_2d[1] = RBLOCK

                # reshape it into 2d (X,R)
                value = triton_reshape(value, initial_shape, shape_2d)

                # transpose to (R,X)
                value = f"tl.trans({value})"

                # broadcast if needed
                broadcast_needed = shape_2d != [XBLOCK, RBLOCK]
                if broadcast_needed:
                    value = f"tl.broadcast_to({value}, ({RBLOCK}, {XBLOCK}))"
            else:
                raise NotImplementedError

            return value

        assert len(V.kernel.dense_size_list()) >= 3, "tl.dot can only do mm and bmm"

        XBLOCK = str(TritonSymbols.block_sizes[SymT.XBLOCK])
        YBLOCK = str(TritonSymbols.block_sizes[SymT.YBLOCK])
        ZBLOCK = str(TritonSymbols.block_sizes[SymT.ZBLOCK])
        RBLOCK = str(TritonSymbols.block_sizes[SymT.R0_INDEX])

        a = V.kernel.cse.generate(
            V.kernel.compute,
            reshape_transpose_broadcast_for_dot(a, list(a.shape), [YBLOCK, RBLOCK]),
            dtype=a.dtype,
            shape=(YBLOCK, RBLOCK),
        )

        b = V.kernel.cse.generate(
            V.kernel.compute,
            reshape_transpose_broadcast_for_dot(b, list(b.shape), [RBLOCK, XBLOCK]),
            dtype=b.dtype,
            shape=(RBLOCK, XBLOCK),
        )

        if torch.backends.cuda.matmul.fp32_precision == "tf32":
            input_precision = "tf32"
        else:
            input_precision = "ieee"

        return f'tl.dot({a}, {b}, input_precision="{input_precision}")'

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
    def sin(x):
        return f"tl_math.sin({x})"

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
    def ldexp(x, n):
        return f"libdevice.ldexp({x}, {n}.to(tl.int32))"

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
        if torch.version.hip:
            return f"tl.rsqrt({x})"
        else:
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
        cse_var = V.kernel.cse.varname_map.get(x)
        if cse_var and hasattr(cse_var, "dtype"):
            dtype = cse_var.dtype
        else:
            dtype = None
        if (
            config.use_fast_math
            and torch.version.hip
            and get_triton_version() > (3, 5)
            and dtype != torch.float64
            and dtype is not None
        ):
            # Requires upstream Triton 3.6+ for latest fast_tanhf support
            # https://github.com/triton-lang/triton/pull/8551
            return f"libdevice.fast_tanhf({x})"
        else:
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


class TritonKernelOverrides(TritonOverrides):
    """Map element-wise ops to Triton within a TritonKernel

    Unlike TritonOverrides, these assume the code is going to be inserted into
    the body of the main triton kernel and so it may use indexing and mask
    variables which are assumed to already be defined in the current scope.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # happens in __init__ unlike _initialize_pointwise_overrides
        # because the libdevice registrations are populated during lowerings
        self._setup_libdevice_routing()

    @classmethod
    @functools.cache
    def _setup_libdevice_routing(cls):
        """Set up routing to libdevice implementations for fp64 inputs."""

        from torch._inductor.codegen.common import OpDecompositions

        for fn_name in torch._inductor.utils.op_requires_libdevice_fp64:
            assert hasattr(cls, fn_name)
            original_impl = getattr(cls, fn_name)

            def decomposition_router(x, _original_impl, _fn_name):
                if x.dtype != torch.float64:
                    return _original_impl(x)
                else:
                    return getattr(OpDecompositions, _fn_name)(x).value

            if fn_name == "sigmoid":
                assert hasattr(OpDecompositions, "sigmoid")
                fn = functools.partial(
                    decomposition_router, _original_impl=original_impl, _fn_name=fn_name
                )
                fn.__name__ = fn_name  # type: ignore[attr-defined]
                setattr(cls, fn_name, staticmethod(fn))
                continue

            def dtype_router(x, _original_impl, _fn_name):
                if x.dtype == torch.float64:
                    return f"libdevice.{_fn_name}({x})"
                else:
                    return _original_impl(x)

            fn = functools.partial(
                dtype_router, _original_impl=original_impl, _fn_name=fn_name
            )
            fn.__name__ = fn_name  # type: ignore[attr-defined]
            setattr(cls, fn_name, staticmethod(fn))

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
        indexing = V.kernel.indexing(
            expr, block_ptr=False, tma_compatibility_checker=None
        )
        assert isinstance(indexing, IndexingOptions)

        shape: BlockShapeType
        if indexing.expand_shape:
            shape = indexing.expand_shape
        else:
            shape = TritonSymbols.get_block_shape(indexing.index)

        # Our sympy expr printing casts to the current kernel index dtype.
        # we only respect non int32-int64 dtypes and otherwise use current kernel indexing dtype
        index_dtype = V.kernel.get_index_dtype_as_torch_dtype()
        dtype = dtype if dtype not in (torch.int32, torch.int64) else index_dtype

        # after we emit this var we cast it to the correct dtype
        orig = config.test_configs.runtime_triton_dtype_assert
        try:
            config.test_configs.runtime_triton_dtype_assert = False
            var = V.kernel.cse.generate(
                V.kernel.compute,
                indexing.index_str,
                bounds=get_bounds_index_expr(expr),
                dtype=dtype,
                shape=shape,
            )
        finally:
            config.test_configs.runtime_triton_dtype_assert = orig

        if dtype not in (torch.int32, torch.int64):
            var = V.kernel.cse.generate(
                V.kernel.compute,
                cls.to_dtype(var, dtype),
                dtype=upcast_compute_type(dtype),
                shape=var.shape,
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
                    shape=var.shape,
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
                shape=mask.shape,
            )

        nodes = body.graph.find_nodes(op="output")
        assert nodes, "graph for body does not contain an output"

        need_where = False
        # If we have a tl.load with a masking operator and no other value
        # we can add the mask here and the other value to the tl.load
        # operator to save the branching cost.
        for node in nodes:
            for arg in node.args:
                if arg.target != "load" or should_unwrap_unspec_arg(arg.args[1]):
                    need_where = True
                    break

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
                shape=result.shape,
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

        mantissa = V.kernel.cse.newvar(dtype=x.dtype, shape=x.shape)
        exponent = V.kernel.cse.newvar(dtype=torch.int32, shape=x.shape)
        V.kernel.compute.writeline(
            f"{mantissa}, {exponent} = triton_helpers.frexp({x})"
        )
        V.kernel.cse.put(cache_key, (mantissa, exponent))
        return (mantissa, exponent)

    @staticmethod
    # pyrefly: ignore [bad-override]
    def partial_accumulate(
        name: str,
        reduction_type: str,
        value: CSEVariable,
        extra_meta: dict[str, Any],
    ) -> None:
        raise NotImplementedError


class HelperFunctions:
    """An ordered set of helper functions."""

    _templates_seen: dict[str, str]  # Template code to function name
    finalized_helpers: list[str]

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

    shape: list[sympy.Expr] = dataclasses.field(default_factory=list)
    block_shape: list[sympy.Expr] = dataclasses.field(default_factory=list)
    strides: list[sympy.Expr] = dataclasses.field(default_factory=list)
    offsets: list[sympy.Expr] = dataclasses.field(default_factory=list)

    @dataclasses.dataclass
    class StrideSorter:
        original_strides: list[int]
        sort_idx: list[int]
        revert_sort_idx: list[int] = dataclasses.field(init=False)

        def __post_init__(self):
            assert len(self.original_strides) > 0
            assert len(self.sort_idx) == len(self.original_strides)

            identity_sort_idx = list(range(len(self.original_strides)))
            self._is_identity = self.sort_idx == identity_sort_idx

            # Set revert_sort_idx
            sorted_dims_by_strides_map = {k: i for i, k in enumerate(self.sort_idx)}
            self.revert_sort_idx = [
                sorted_dims_by_strides_map[i]
                for i in range(len(sorted_dims_by_strides_map))
            ]

        @property
        def is_identity(self):
            return self._is_identity

        @classmethod
        @abstractmethod
        def create(
            cls, original_strides: list[Union[int, sympy.Expr]], shape_env: ShapeEnv
        ) -> BlockParameters.StrideSorter:
            """Create a `StrideSorter` that can be used to sort block parameters."""

        def sort(self, attr):
            if not self.is_identity:
                return [attr[i] for i in self.sort_idx]
            return attr

        def revert(self, attr):
            if not self.is_identity:
                return [attr[i] for i in self.revert_sort_idx]
            return attr

    @dataclasses.dataclass
    class IdentityStrideSorter(StrideSorter):
        def __post_init__(self):
            super().__post_init__()

        @classmethod
        def create(
            cls, original_strides: list[Union[int, sympy.Expr]], shape_env: ShapeEnv
        ) -> BlockParameters.StrideSorter:
            return cls(
                original_strides=original_strides,
                sort_idx=list(range(len(original_strides))),
            )

    @dataclasses.dataclass
    class TensorDecriptorStrideSorter(StrideSorter):
        """
        Sorts BlockParameters dimensions with strides in descending order.
        """

        def __post_init__(self):
            super().__post_init__()

        @classmethod
        def create(
            cls, original_strides: list[Union[int, sympy.Expr]], shape_env: ShapeEnv
        ) -> BlockParameters.StrideSorter:
            """
            If the strides are not all known constants or if the strides are already
            sorted in descending order, return identity sort.

            For example if block_shape @ strides is [ZBLOCK, XBLOCK, YBLOCK] @ [8, 1, 16]
            The indices to sort the strides in descending order will be [2, 0, 1].
            The indices to revert back to the original order will be [1, 2, 0].
            """
            identity_sort = list(range(len(original_strides)))
            try:
                # TODO: even if the strides are not in descending order the strides
                # may be tensor descriptor compliant
                # i.e. innermost stride == 1 and outer strides 16 byte aligned
                # We should benchmark the effect of applying a transpose to these
                # cases vs leaving them unsorted.
                sort_idx = utils.argsort_sym(shape_env, original_strides, reverse=True)
            except AssertionError:
                # Symbolic shapes, failed to evaluate comparison expression
                sort_idx = identity_sort

            return cls(
                original_strides=original_strides,
                sort_idx=sort_idx,
            )

    def __add__(self, other: BlockParameters) -> BlockParameters:
        """
        Concatenates block parameters.
        """
        cls = type(self)
        a, b = tuple(dataclasses.asdict(x) for x in (self, other))
        return cls(**{key: a[key] + b[key] for key in a})

    def maybe_sort_with_stride_order(
        self, stride_sorter_cls: type[StrideSorter], shape_env: ShapeEnv
    ) -> tuple[BlockParameters, BlockParameters.StrideSorter]:
        """
        Sort `BlockParameter` with stride_sorter_cls. Returns block parameters
        as well as a `StrideSorter` which contains information on how the sort
        can be reverted.
        """
        stride_sorter = stride_sorter_cls.create(self.strides, shape_env=shape_env)
        params = BlockParameters(
            **{
                key: stride_sorter.sort(val)
                for key, val in dataclasses.asdict(self).items()
            }
        )
        return params, stride_sorter

    def remove_dims(self, removable_dims: list[bool]) -> BlockParameters:
        """
        Remove dimensions where removable_dims is True.
        """

        def filter_dims(it):
            return [
                item
                for item, is_removable in zip(it, removable_dims)
                if not is_removable
            ]

        return BlockParameters(
            **{key: filter_dims(val) for key, val in dataclasses.asdict(self).items()},
        )


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
        ws_name, _, ws_offset = self.args.workspace(nbytes, False)
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
    config: dict[str, int]

    def __getitem__(self, item):
        return self.config[item]

    def __contains__(self, item):
        return item in self.config


class TritonCSE(CSE[TritonCSEVariable, Union[str, tuple[str, str]]]):
    """
    Subclasses CSE to apply the current load mask to the cache key to avoid CSEing
    variables across separate masked blocks.
    """

    def augment_key(self, cache_key: str) -> Union[str, tuple[str, str]]:
        if mask := V.kernel._load_mask:
            return (cache_key, mask.name)
        else:
            return cache_key


@dataclasses.dataclass
class TMACompatibilityChecker:
    """
    Checks if the TMA API can be used for load / store triton operations.
    """

    kernel: TritonKernel
    dtype: torch.dtype
    for_store: bool
    force: bool

    def __post_init__(self):
        self.failed_debug_prefix = "Cannot use TMA descriptor for load / store since: "

    # Also see Note: TMA API Restrictions for the below
    def can_use_tma(
        self,
    ) -> bool:
        if self.force:
            return True
        if not (
            V.graph.get_current_device_or_throw().type == "cuda"
            and torch.cuda.get_device_capability()[0] >= 9
            and config.triton.use_tensor_descriptor
            and config.assume_aligned_inputs
            and has_triton_stable_tma_api()
            # For CUDA The base ptr needs to be aligned
        ):
            log.debug(
                (
                    "%s Requires triton>=3.4.0, a CUDA device with cc>=9.0 and"
                    " `use_tensor_descriptor` and `assume_aligned_inputs` options enabled"
                ),
                self.failed_debug_prefix,
            )
            return False

        # `no_x_dim` => XBLOCK=1, and for reductions this means only one element
        # is to be stored . However the TMA API requires that
        # the store will be 16 byte aligned, which is not attainable with a single
        # element
        if self.for_store and self.kernel.no_x_dim:
            log.debug(
                "%s stores with `no_x_dim` cannot load 16 bytes.",
                self.failed_debug_prefix,
            )
            return False

        return True

    def are_block_parameters_compatible(
        self,
        block_params: BlockParameters,
    ) -> bool:
        """
        Check if the block parameters are valid for TMA.
        If force, we allow relying on symbolic hints equivalent
        to what we check for Triton templates.
        """
        if self.force:
            strides = [
                V.graph.sizevars.symbolic_hint(st) for st in block_params.strides
            ]
        else:
            strides = block_params.strides

        # The TMA API requires that the innermost stride is 1
        # and that the outer strides are 16 byte aligned
        if not V.graph.sizevars.statically_known_equals(strides[-1], sympy.Integer(1)):
            log.debug(
                "%s TMA API requires innermost stride to be 1. Strides are: %s",
                self.failed_debug_prefix,
                strides,
            )
            return False

        element_size = self.dtype.itemsize
        for stride in strides[:-1]:
            if not V.graph.sizevars.statically_known_equals(
                ModularIndexing(stride * element_size, 1, sympy.Integer(16)),
                sympy.Integer(0),
            ):
                log.debug(
                    "%s TMA API requires outer strides to be 16 byte aligned. Dtype bytes: %d, strides: %s",
                    self.failed_debug_prefix,
                    element_size,
                    strides,
                )
                return False

        # Now compute the minimum value of the block type that is used
        # in the innermost block size that can guarantee that 16 bytes of data
        # can be loaded / stored.
        # Start with finding the innermost block type
        innermost_block_shape = block_params.block_shape[-1]

        # Pure singleton case
        if V.graph.sizevars.statically_known_equals(
            innermost_block_shape, sympy.Integer(1)
        ):
            log.debug(
                "%s innermost block shape cannot load 16 bytes. Block shape: %s",
                self.failed_debug_prefix,
                block_params.block_shape,
            )
            return False

        innermost_block_type = None
        innermost_block_symt = None
        for block_type_str in innermost_block_shape.free_symbols:
            for block_symt in TritonSymbols.block_types:
                if symbol_is_type(block_type_str, block_symt):
                    innermost_block_type = block_type_str
                    innermost_block_symt = block_symt
                    break

        assert innermost_block_type and innermost_block_symt, (
            f"{innermost_block_shape} expr must contain a single block type from {TritonSymbols.block_types}"
        )

        # For persistent reductions, the reduction block sizes are fixed at compile time.
        # Only apply this logic when the innermost block is a reduction block;
        # persistent reductions can still have pointwise-style loads where the innermost block is X/Y/Z,
        # and in that case we should fall back to the generic analysis below.
        if (
            self.kernel.persistent_reduction
            and not self.for_store
            and innermost_block_symt in TritonSymbols.reduction_types
        ):
            # For a discontiguous tensor, a 1D block will be split across several
            # dimensions, e.g. R0_BLOCK:
            # block_shape=[XBLOCK, ((R0_BLOCK + 31)//32), Min(1, ((R0_BLOCK + 31)//32)), Min(32, R0_BLOCK)]
            # The persistent R0_BLOCK will be a power of 2 that is at least r0_numel So it
            # should be guaranteed that Min(32, R0_BLOCK) * element_size >= 16
            innermost_tree_prefix = prefix_str[innermost_block_symt]
            tree_numel = None
            for t in self.kernel.range_trees:
                if t.is_reduction and t.prefix == innermost_tree_prefix:
                    tree_numel = t.numel
                    break
            if tree_numel is None:
                # If we can't map the innermost reduction block type to a reduction range tree,
                # we cannot determine the persistent RBLOCK value,
                # so we cannot validate the 16-byte innermost-dimension requirement for TMA.
                # Treat this as incompatible rather than asserting during compilation, fallback to non-TMA loads.
                log.debug(
                    "%s could not find reduction range tree for innermost prefix %s Block shape: %s",
                    self.failed_debug_prefix,
                    innermost_tree_prefix,
                    block_params.block_shape,
                )
                return False
            persistent_rblock = self.kernel._get_persistent_RBLOCK(tree_numel)
            innermost_block_bytes = (
                innermost_block_shape.subs({innermost_block_type: persistent_rblock})
                * element_size
            )
            if not V.graph.sizevars.statically_known_geq(
                innermost_block_bytes, sympy.Integer(16)
            ):
                log.debug(
                    "%s persistent reduction innermost block shape cannot load 16 bytes. Block shape: %s, persistent RBLOCK: %d",
                    self.failed_debug_prefix,
                    block_params.block_shape,
                    persistent_rblock,
                )
                return False

        else:
            # E.g. if the innermost block shape is Min(2, XBLOCK)
            # then the TMA API can only be used if the dtype has an 8 byte element
            # size so that 16 bytes of data can be loaded in the innermost dimension
            try:

                def indexing_div_rep(
                    x: sympy.Expr,
                    y: sympy.Expr,
                    z: Optional[sympy.Expr] = None,
                ) -> sympy.Expr:
                    div = x / y
                    if z:
                        div = div % z
                    return div

                solve_expr = innermost_block_shape * element_size - 16
                # Sympy cannot handle FloorDiv and ModularIndexing well, so simplify
                solve_expr_simplified = solve_expr.replace(
                    FloorDiv, indexing_div_rep
                ).replace(ModularIndexing, indexing_div_rep)
                min_block_size = next_power_of_2(
                    int(
                        sympy.nsolve(
                            solve_expr_simplified,
                            innermost_block_type,
                            1,
                        )
                    )
                )

                # TODO: min block size may be too large / introduce redundancy
                if min_block_size > self.kernel.max_block(
                    prefix_str[innermost_block_symt]
                ):
                    log.debug(
                        "%s the minimum block size to satisfy expression %s is too large: %d",
                        self.failed_debug_prefix,
                        solve_expr_simplified,
                        min_block_size,
                    )
                    return False

                block_type_str = self.kernel.index_to_str(innermost_block_type)
                # Check block sizes if the user has provided a fixed triton config
                if self.kernel.fixed_config:
                    if min_block_size > self.kernel.fixed_config[block_type_str]:
                        log.debug(
                            "%s For block %s, fixed config block size %d is smaller "
                            "than the minimum required: %d",
                            self.failed_debug_prefix,
                            block_type_str,
                            self.kernel.fixed_config[block_type_str],
                            min_block_size,
                        )
                        return False
                else:
                    # Update the minimum block sizes that are passed to triton
                    # heuristics
                    self.kernel.tma_min_block_sizes[block_type_str] = max(
                        min_block_size,
                        self.kernel.tma_min_block_sizes.get(block_type_str, 1),
                    )

            except ValueError:
                log.debug(
                    "%s innermost block shape cannot load 16 bytes. Block params: %s",
                    self.failed_debug_prefix,
                    block_params.block_shape,
                )
                return False

        return True

    def can_lift(self) -> bool:
        """
        Can you lift the make_tensor_descriptor
        call to the top of the kernel? This requires
        being certain that all of the shape, stride,
        and block_shape information is handled in arguments
        or top level definitions.

        Right now we assume this is always possible if you force TMA.
        """
        return self.force


class TritonKernel(SIMDKernel[TritonCSEVariable]):
    """A class to represent a triton kernel and helpers to generate
    triton kernel programmatically
    """

    overrides = TritonKernelOverrides  # type: ignore[assignment]
    helper_functions: HelperFunctions
    kexpr: Callable[[sympy.Expr], str] = texpr
    allow_block_ptr = True
    tma_compatibility_checker_cls = TMACompatibilityChecker
    transpose_discontiguous_tensor_descriptors_override: Optional[bool] = None

    def __init__(
        self,
        tiling: dict[str, sympy.Expr],
        min_elem_per_thread=0,
        optimize_mask=True,
        fixed_config: Optional[FixedTritonConfig] = None,
        hint_override: Optional[int] = None,
        is_combo_kernel: bool = False,
        **kwargs,
    ) -> None:
        self.optimize_mask: bool = optimize_mask
        self.fixed_config = fixed_config
        self.is_combo_kernel: bool = is_combo_kernel
        super().__init__(tiling, **kwargs)
        self.cse = TritonCSE(self.newvar_prefix, self.suffix)
        # Cache of values that can be reused for the prologue.
        self.prologue_cache: dict[str, str] = {}
        self.prologue: IndentedBuffer = IndentedBuffer()
        self.post_loop_combine: IndentedBuffer = IndentedBuffer()
        self.post_loop_store: IndentedBuffer = IndentedBuffer()
        self.outside_loop_vars = OrderedSet[Any]()
        self.min_elem_per_thread = min_elem_per_thread
        self.block_ptr_id = itertools.count()
        self.block_ptr_to_buffer = dict[str, str]()
        self.helper_functions = HelperFunctions()
        self.pointer_advancements: dict[SymT, dict[str, list[sympy.Expr]]] = (
            collections.defaultdict(dict)
        )
        self.tma_min_block_sizes = dict[str, int]()
        self.hint_override = hint_override
        self._load_counts: collections.Counter[str] = collections.Counter()
        self._pdl_load_index = 0
        self._pdl_has_wait = False

        # A set of autotuning hints to pass as part of triton_meta
        self.autotune_hints = OrderedSet[AutotuneHint]()
        self.triton_meta: Optional[dict[str, Any]] = None

        if self.inside_reduction:
            self.codegen_reduction_numels(self.body)

        if self.cooperative_reduction:
            self.init_cooperative_reduction()

        self.codegen_range_tree()

        if self.cooperative_reduction:
            self.init_cooperative_reduction_mask()

        self.has_load_with_contiguous_rdim = False
        # We track the store name since a store can be canceled later
        self.stores_with_contiguous_rdim: list[str] = []

    @staticmethod
    def _has_stride1_on_rdim(index) -> bool:
        # These analysis is only needed in deterministic mode so far
        # to filter triton configs. Return false immediately to avoid
        # increasing compilation time when the mode is off.
        if not (
            config.deterministic or config.test_configs.force_filter_reduction_configs
        ):
            return False
        support_vars = index.free_symbols
        reduce_vars = [
            var
            for var in support_vars
            if symbol_is_type(var, TritonSymbols.reduction_types)
        ]

        if len(reduce_vars) == 0:
            return False

        # for expression "x0 + 150528*((x1//(s27*s38))) + 3*(ModularIndexing(x1, 1, s38)) + 672*(ModularIndexing(x1, s38, s27))"
        # stride_vars will results in DivisionByZero error
        try:
            stride_vars = V.graph.sizevars.stride_vars(index, reduce_vars, support_vars)
        except ZeroDivisionError:
            return False

        return any(stride == 1 for stride in stride_vars)

    @property
    def has_store_with_contiguous_rdim(self) -> bool:
        return not all(
            is_buffer_removed(name) for name in self.stores_with_contiguous_rdim
        )

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
            """\
            RSPLIT_NEXT_POWER_OF_2: tl.constexpr = triton_helpers.constexpr_next_power_of_2(RSPLIT)
            RSPLIT_IS_POWER_OF_2: tl.constexpr = RSPLIT == RSPLIT_NEXT_POWER_OF_2
            HAS_RSPLIT: tl.constexpr = RSPLIT > 1
            rsplit_id = tl.program_id(0)
            num_rblocks = (rnumel + RBLOCK - 1) // RBLOCK
            rsplit_chunk = (num_rblocks + RSPLIT - 1) // RSPLIT * RBLOCK
            rsplit_start = rsplit_chunk * rsplit_id
            rsplit_end = rsplit_chunk * (rsplit_id + 1)
            """,
        )
        if any(
            not self._has_constant_mask(tree)
            for tree in self.range_trees
            if tree.is_reduction
        ):
            self.body.writeline(
                "rsplit_end = tl.where(rsplit_end < rnumel, rsplit_end, rnumel)"
            )

    def init_cooperative_reduction_mask(self):
        rsplit_arange = "tl.arange(0, RSPLIT_NEXT_POWER_OF_2)"
        if not self.no_x_dim:
            rsplit_arange = f"{rsplit_arange}[None, :]"
        self.body.writeline(f"rsplit_arange = {rsplit_arange}")

        if self._has_constant_xmask():
            self.body.splice(
                """\
                if RSPLIT_IS_POWER_OF_2:
                    rsplit_mask: tl.constexpr = None
                else:
                    rsplit_mask = rsplit_arange < RSPLIT
                """
            )
        else:
            assert not self.no_x_dim
            self.body.writeline(
                "rsplit_mask = xmask if RSPLIT_IS_POWER_OF_2 else ((rsplit_arange < RSPLIT) & xmask)"
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
        """
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
        return (
            self.persistent_reduction
            and len(self.numels) == self.num_reduction_dims + 1
            and self.fixed_config
            and self.fixed_config["XBLOCK"] == 1
        )

    @property
    def assert_function(self) -> str:
        return "tl.device_assert"

    def indexing(
        self,
        index: sympy.Expr,
        *,
        copy_shape: Optional[Union[str, tuple[str]]] = None,
        dense_indexing=False,
        override_mask=None,
        block_ptr=False,
        tma_compatibility_checker: Optional[TMACompatibilityChecker] = None,
    ):
        """
        Compute the index and mask to pass to tl.load() or tl.store()
        """
        index = self.prepare_indexing(index)
        index_vars = index.free_symbols
        has_rindex = False

        mask_vars: OrderedSet[str] = OrderedSet()
        for var in sorted(index_vars, key=operator.attrgetter("name")):
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
                if len(prefix_matches) == 0:
                    pass
                assert len(prefix_matches) == 1, f"Ambiguous type: {var.name}"
                mask_vars.add(f"{prefix_matches[0]}mask")

        need_dense = (
            config.triton.dense_indexing
            or dense_indexing
            or self._load_mask is not None
        ) and index != 0

        have_dense = True
        have_loop_vars = False
        dense_mask_vars: OrderedSet[str] = OrderedSet()

        for tree in self.active_range_trees():
            if index_vars.intersection(tree.var_list):
                have_loop_vars = True
            else:
                have_dense = False
            dense_mask_vars.add(f"{tree.prefix}mask")

        if (
            (
                (block_ptr and self.allow_block_ptr and config.triton.use_block_ptr)
                or (
                    tma_compatibility_checker
                    and tma_compatibility_checker.can_use_tma()
                )
            )
            and not override_mask
            and not self._load_mask
            and len(mask_vars - dense_mask_vars) == 0
            and not self.is_indirect_indexing(index)
            and have_loop_vars
            # workaround https://github.com/triton-lang/triton/issues/2821
            and self.index_dtype == "tl.int32"
        ):

            def match_affine_block(
                index: sympy.Expr, range_tree: IterationRangesRoot
            ) -> Optional[BlockParameters]:
                """
                Matches expressions of the form:
                    idx = s * xindex

                This implies stride (s,), and shape (XBLOCK,).
                """
                stride = BlockPatternMatcher.match_affine_block_expr(
                    index, range_tree.symbol()
                )
                if stride is None:
                    return None

                return BlockParameters(
                    shape=[range_tree.numel],
                    block_shape=[TritonSymbols.get_block_size(range_tree)],
                    strides=[stride],
                    offsets=[TritonSymbols.get_block_offset(range_tree)],
                )

            def match_mod_div_block(
                index: sympy.Expr, range_tree: IterationRangesRoot
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

                index_var = range_tree.symbol()

                # Bound the possible number of dims. We use the following heuristics:
                # - At least one dim for each range tree node.
                # - At least one dim for every FloorDiv or ModularIndexing op.
                # - At least 2 dims to pattern match.
                denom, modulo = sympy.symbols(
                    "denom modulo",
                    cls=functools.partial(sympy.Wild, exclude=[index_var]),
                )
                num_dims = max(
                    2,
                    # range_tree.nodes only includes the entries for the range tree
                    # len(range_tree.nodes) <= self.range_tree_nodes
                    len(range_tree.nodes),
                    (
                        index.count(FloorDiv(index_var, denom))
                        + index.count(ModularIndexing(index_var, denom, modulo))
                    ),
                )

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
                # while the leading dimension can exceed this to accommodate a larger
                # block size.
                linear_block_size = TritonSymbols.get_block_size(range_tree)
                block_shape: list[sympy.Expr] = [
                    CeilDiv(linear_block_size, slice_numels[0])
                ] + [
                    sympy.Min(CeilDiv(linear_block_size, numel), dim)
                    for numel, dim in zip(slice_numels[1:], dims[1:])
                ]

                # Compute block offsets from {xyzr}offset and the matched expressions.
                block_offsets: list[sympy.Expr] = [
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

            def match_block_subexpr(
                expr: sympy.Expr, range_tree: IterationRangesRoot
            ) -> Optional[BlockParameters]:
                """
                Match a block indexing subexpression involving a single range tree.
                """
                for match_func in (
                    match_affine_block,
                    match_mod_div_block,
                ):
                    match = match_func(expr, range_tree)
                    if match is not None:
                        return match

                return None

            def match_block_expr() -> Optional[BlockDescriptorOptions]:
                index_relative_to_xyr_index = sympy_subs(
                    index, {v: t.expr for v, t in self.range_tree_nodes.items()}
                )
                range_trees = self.active_range_trees()

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
                    params = match_block_subexpr(subexpr, tree)
                    if params is None:
                        return None
                    block_params += params

                # Collect leftover terms as a constant offset.
                offset = index_relative_to_xyr_index - sum(index_subexprs)

                # Form the block pointer or TMA descriptor.
                self.filter_masks(mask_vars)

                options_class = (
                    BlockPtrOptions
                    if config.triton.use_block_ptr
                    else TensorDescriptorOptions
                )
                nonlocal tma_compatibility_checker
                stride_sorter_cls: type[BlockParameters.StrideSorter]
                if config.triton.use_block_ptr:
                    can_lift = False
                    stride_sorter_cls = BlockParameters.IdentityStrideSorter
                else:
                    tma_compatibility_checker = cast(
                        TMACompatibilityChecker, tma_compatibility_checker
                    )
                    can_lift = tma_compatibility_checker.can_lift()

                    if (
                        self.transpose_discontiguous_tensor_descriptors_override
                        is not None
                    ):
                        transpose_contiguous = (
                            self.transpose_discontiguous_tensor_descriptors_override
                        )
                    else:
                        transpose_contiguous = (
                            config.triton.transpose_discontiguous_tensor_descriptor
                        )

                    # For templates:
                    # Only try transpose if we know the output shape
                    # in case we need to transpose the data.
                    if hasattr(self, "template_out_shape"):
                        transpose_contiguous &= copy_shape is not None

                    stride_sorter_cls = (
                        BlockParameters.TensorDecriptorStrideSorter
                        if transpose_contiguous
                        else BlockParameters.IdentityStrideSorter
                    )

                options = options_class.create(
                    params=block_params,
                    constant_offset=offset,
                    range_trees=range_trees,
                    mask_vars=mask_vars,
                    get_max_block=self.max_block,
                    can_lift=can_lift,
                    stride_sorter_cls=stride_sorter_cls,
                )
                if options_class == TensorDescriptorOptions:
                    tma_compatibility_checker = cast(
                        TMACompatibilityChecker, tma_compatibility_checker
                    )
                    if not tma_compatibility_checker.are_block_parameters_compatible(
                        options.params
                    ):
                        return None

                return options

            # Return a block pointer, if indexing matches the pattern.
            options = match_block_expr()
            if options is not None:
                return options
        expand_str = None
        expand_shape: BlockShapeType = None
        index_str = self.index_to_str(index)

        def _get_expand_str():
            if copy_shape:
                if isinstance(copy_shape, str):
                    return f"{copy_shape}.shape", None
                else:
                    return "[" + ", ".join(str(c) for c in copy_shape) + "]", copy_shape
            else:
                return self.dense_size_str(), tuple(self.dense_size_list())

        if is_sympy_integer_like(index):
            # Integer indexing produces a size-1 scalar tensor with the same shape
            # as the dense dimension. E.g, if dense_size = [YBLOCK, XBLOCK, R0_BLOCK],
            # then we create tl.full([1, 1, 1], int).
            #
            # Exceptions:
            # 1. If copy_shape is explicitly provided, use copy_shape expansion instead.
            # 2. If the dense tensor has only one dimension (e.g., [XBLOCK]),
            #    broadcasting does not apply. For example:
            #        tl.arange(0, XBLOCK) + tl.full([1], int)  # -> broadcasting error
            #    In this case, we fall back to dense indexing:
            #        tl.full([XBLOCK], int)
            if copy_shape or len(self.dense_size_list()) == 1:
                expand_str, expand_shape = _get_expand_str()
            else:
                expand_str = str([1] * len(self.dense_size_list()))
                expand_shape = tuple([1] * len(self.dense_size_list()))

            index_str = f"tl.full({expand_str}, {index_str}, tl.int32)"
            if self.fixed_config or self.is_combo_kernel:
                mask_vars = OrderedSet(
                    f"{tree.prefix}mask"
                    for tree in self.range_trees
                    if not tree.is_reduction and not self._has_constant_mask(tree)
                )
            else:
                mask_vars = OrderedSet()
            if self._load_mask:
                mask_vars.add(self._load_mask)
            return IndexingOptions(
                index_str,
                mask_vars,
                expand_str,
                has_rindex,
                index,
                expand_shape=expand_shape,
            )

        if need_dense and not have_dense:
            if self.inside_reduction and self.is_native_matmul:
                # This avoids full broadcasting (need_dense) when performing native matmul.
                # For example, self._load_mask previously required tl.broadcast_to() in index_str.
                # Due to the restrictions of tl.dot semantics, we only want to expand the block
                # shape for the necessary axes.
                #
                # Previously:
                #   tmp1 = tl.load(ptr + tl.broadcast_to(r0, [YBLOCK, XBLOCK, R0_BLOCK]),
                #                  r0_mask & tmp0 & xmask)
                #
                # Now:
                #   tmp1 = tl.load(ptr + tl.broadcast_to(r0, [1, 1, R0_BLOCK]),
                #                  r0_mask & tmp0 & xmask)
                #
                # We achieve this by determining the required block shape through mask inspection.
                # When a temporary variable appears in the mask (e.g., self._load_mask), we retrieve
                # its true shape by inspecting tmp.mask_vars tracked by TritonCSEVariable.
                #
                # Caution: it may miss the correct block shape if the specific mask was constant
                # and thus not tracked in TritonCSEVariable.mask_vars.
                #
                # TODO: Once the shape propagation PR lands, reimplement this logic:
                #       https://github.com/pytorch/pytorch/pull/152198
                mask_shape = mask_vars.copy()
                if self._load_mask:
                    mask_shape.add(self._load_mask)

                xyzr = OrderedSet(["xmask", "ymask", "zmask", "r0_mask"])
                while not mask_shape.issubset(xyzr):
                    tmp_masks = mask_shape.difference(xyzr)
                    tmp = tmp_masks.pop()
                    assert isinstance(tmp, TritonCSEVariable)
                    mask_shape.discard(tmp)
                    mask_shape.update(tmp.mask_vars)

                # e.g., expand_list becomes ['ZBLOCK', 1, 1, 'R0_BLOCK']
                expand_list = ["1"] * len(self.dense_size_list())
                for mask in mask_shape:
                    assert isinstance(mask, str)
                    for tree in self.active_range_trees():
                        if mask.startswith(tree.prefix):
                            dim = tree.tensor_dim
                            assert isinstance(dim, int)
                            expand_list[dim] = self.dense_size_list()[dim]

                expand_str = "[" + ",".join(map(str, expand_list)) + "]"
                expand_shape = tuple(expand_list)
                index_str = f"tl.broadcast_to({index_str}, {expand_str})"
            else:
                expand_str, expand_shape = _get_expand_str()
                index_str = f"tl.broadcast_to({index_str}, {expand_str})"
                mask_vars = dense_mask_vars
        elif not have_loop_vars and copy_shape:
            expand_shape_str, expand_shape = _get_expand_str()
            index_str = f"tl.broadcast_to({index_str}, {expand_shape_str})"
            mask_vars = dense_mask_vars

        if expand_shape is None:
            if need_dense or have_dense:
                _, expand_shape = _get_expand_str()
            else:
                expand_shape = ()

        if override_mask:
            mask_vars = OrderedSet([override_mask])

        if self._load_mask:
            mask_vars.add(self._load_mask)

        self.filter_masks(mask_vars)

        return IndexingOptions(
            index_str,
            mask_vars,
            expand_str,
            has_rindex,
            index,
            expand_shape=expand_shape,
        )

    def codegen_block_ptr(
        self,
        name: str,
        var: str,
        indexing: Union[BlockPtrOptions, TensorDescriptorOptions],
        other="",
    ) -> tuple[str, str]:
        """Generate a block pointer or tensor descriptor for Triton kernel operations.

        This method creates either a block pointer (for regular Triton operations) or
        a tensor descriptor (for TMA operations) based on the indexing type. It handles
        caching and reuse of descriptors for performance optimization.

        Args:
            name: The name of the buffer/tensor being accessed
            var: The variable name for the pointer
            indexing: Block pointer options or tensor descriptor options containing
                     indexing information and boundary check settings
            other: Additional parameters string (e.g., padding options)

        Returns:
            A tuple containing:
            - block_descriptor: The generated block pointer or tensor descriptor variable name
            - other: Modified additional parameters string with boundary check options
        """
        check = indexing.boundary_check()
        if isinstance(indexing, TensorDescriptorOptions):
            if check and other:
                # The TMA API currently does not support padding values
                # but the default is zero
                assert other == ", other=0.0"
                other = ""
        else:
            if not check:
                # workaround https://github.com/triton-lang/triton/issues/2813
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
        ) or indexing.can_lift:
            if indexing.can_lift and var in self.prologue_cache:
                # Check for epilogue subtiling to reuse the same
                # tensor descriptor.
                block_descriptor = self.prologue_cache[var]
            else:
                block_ptr_line = indexing.format(var, roffset=False)
                block_var = self.cse.try_get(block_ptr_line)

                # Early return if block descriptor already exists
                if block_var:
                    return str(block_var), other

                block_descriptor_id = next(self.block_ptr_id)
                if isinstance(indexing, BlockPtrOptions):
                    block_descriptor = f"block_ptr{block_descriptor_id}"
                else:
                    block_descriptor = f"tma_descriptor{block_descriptor_id}"
                named_var = self.cse.namedvar(
                    block_descriptor, dtype=torch.uint64, shape=[]
                )
                self.cse.put(block_ptr_line, named_var)

                line_body = DeferredLine(name, f"{block_descriptor} = {block_ptr_line}")
                if indexing.can_lift:
                    self.prologue.writeline(line_body)
                    # Cache the descriptor for epilogue subtiling
                    self.prologue_cache[var] = block_descriptor
                else:
                    self.body.writeline(line_body)

                if isinstance(indexing, BlockPtrOptions):
                    # Store for later use. If the buffer is removed the below advancements
                    # are no longer necessary
                    self.block_ptr_to_buffer[block_descriptor] = name

                    # Generate block pointer advancements, for later use.
                    for symt in TritonSymbols.reduction_types:
                        advance_offsets = indexing.advance_roffset(symt)

                        # Ignore identity advancements.
                        if all(
                            V.graph.sizevars.statically_known_equals(
                                offset, sympy.Integer(0)
                            )
                            for offset in advance_offsets
                        ):
                            continue

                        advancements = self.pointer_advancements[symt]
                        assert block_descriptor not in advancements, (
                            f"duplicate advancement for pointer '{block_descriptor}' at type '{symt}'"
                        )
                        advancements[block_descriptor] = advance_offsets
        else:
            block_descriptor = indexing.format(var)
        return block_descriptor, other

    def codegen_block_ptr_store_line(self, name, indexing, block_ptr, value, other=""):
        # Stores require an explicit broadcast. We do this in two phases:
        #  1. Broadcast the operand to the final shape of the range trees, e.g. [ZBLOCK,
        #     YBLOCK, XBLOCK]. This protects against implicit broadcasting from loads.
        #  2. In case the block pointer / tma descriptor has different dimensionality, broadcast/reshape the
        #     result to the shape of the pointer.
        value = f"tl.broadcast_to({value}, {indexing.final_shape})"

        # These dims no longer need broadcasting.
        for idx, (dim, broadcast_dim) in enumerate(
            zip(indexing.final_shape, indexing.broadcast_shape)
        ):
            if V.graph.sizevars.statically_known_equals(dim, broadcast_dim):
                indexing.broadcasting_dims[idx] = False

        value = indexing.codegen_broadcast_and_reshape(
            value,
            indexing.final_shape,
            indexing.block_shape,
            allow_implicit=False,
            for_store=True,
        )

        # workaround https://github.com/triton-lang/triton/issues/2814
        value = f"{value}.to({triton_store_type(V.graph.get_dtype(name))})"
        if isinstance(indexing, BlockPtrOptions):
            return f"tl.store({block_ptr}, {value}{other})"
        return f"{block_ptr}.store({V.kernel.index_to_str(indexing.offsets)}, {value})"

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
        indexing = self.indexing(expr, block_ptr=False, tma_compatibility_checker=None)
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

    GDC_WAIT = "tl.extra.cuda.gdc_wait()"
    GDC_LAUNCH = "tl.extra.cuda.gdc_launch_dependents()"

    def _enable_pdl_codegen(self):
        if not torch._inductor.config.triton.enable_pdl:
            return False
        if isinstance(V.kernel, torch._inductor.select_algorithm.TritonTemplateKernel):
            return False
        # PDL uses CUDA-specific intrinsics (gdc_wait/gdc_launch), not available on ROCm
        if torch.version.hip:
            return False
        return (
            V.graph.get_current_device_or_throw().type == "cuda"
            and torch.cuda.get_device_capability()[0] >= 9
        )

    def _handle_pdl_before_access(
        self, wait_buffer, *dependencies, consider_reads=False
    ):
        if not self._enable_pdl_codegen():
            return
        current_node = V.kernel.current_node
        prev_node = (
            V.graph.scheduler.previous_node if V.graph.scheduler is not None else None
        )

        def matching_dep(dep):
            assert prev_node is not None
            prev_deps = prev_node.read_writes.writes
            if consider_reads:
                prev_deps = itertools.chain(prev_deps, prev_node.read_writes.reads)
            return any(
                dep == current_node.mutation_renames.get(w.name, w.name)
                for w in prev_deps
            )

        assert dependencies
        need_wait = prev_node is None or any(matching_dep(d) for d in dependencies)
        if not need_wait:
            return
        # hoist before the loop
        if self.inside_reduction and self.range_trees[-1].is_loop:
            wait_buffer = self.body

        wait_buffer.writeline(self.GDC_WAIT)

    def _handle_pdl_after_load(self, launch_buffer, result_var):
        if not self._enable_pdl_codegen():
            return
        if result_var.use_count > 1:  # we already went through this
            return
        # hoist after the loop
        if self.inside_reduction and self.range_trees[-1].is_loop:
            launch_buffer = self.post_loop_combine

        # the issue is that we need to (a) make sure this happens
        # but (b) do not know if this is last yet
        # so we need to remember this (has_wait), which tells use
        # whether we would have needed it, and check if we are last
        launch_buffer.writeline(self.GDC_WAIT)
        launch_buffer.writeline(self.GDC_LAUNCH)

    def _filter_pdl(self, code: IndentedBuffer):
        new_lines = []
        has_wait = False
        previous_launch = None
        for l in code._lines:
            if type(l) is str and self.GDC_WAIT in l:
                if has_wait:
                    continue
                else:
                    has_wait = True
            if type(l) is str and self.GDC_LAUNCH in l:
                if previous_launch is not None:
                    new_lines.pop(previous_launch)
                previous_launch = len(new_lines)
            new_lines.append(l)
        code._lines = new_lines

    def partial_accumulate(
        self, name: str, reduction_type, val, extra_meta: dict[str, Any]
    ):
        self.saved_partial_accumulate.append(
            PartialAccumulate(name, reduction_type, val)
        )

    def load(self, name: str, index: sympy.Expr):
        """
        Load from the memory location 'name', offset by some indexing expression 'index'.
        """
        var = self.args.input(name)
        load_counts = self._load_counts
        load_counts[name] += 1
        make_line: Callable[[str], Union[str, DelayReplaceLine]] = identity
        indirect_indexing = self.is_indirect_indexing(index)
        original_index = index
        dtype = V.graph.get_dtype(name)
        indexing = self.indexing(
            index,
            block_ptr=True,
            tma_compatibility_checker=self.tma_compatibility_checker_cls(
                self,
                dtype,
                for_store=False,
                force=False,
            ),
        )

        if isinstance(indexing, IndexingOptions) and self._has_stride1_on_rdim(
            indexing.index
        ):
            self.has_load_with_contiguous_rdim = True

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

        """Check if the buffer we're about to load, has
        more than one read dependency
        NOTE: enabled with env variable TORCHINDUCTOR_SKIP_L1
        """
        has_read_deps = True
        if config.triton.skip_l1_cache:
            buffer_read_counts = self.features.buffer_read_counts()
            has_read_deps = buffer_read_counts[name] > 1
        """Skip L1 cache if we're (pretty?) sure the data is used only once
        """
        skip_l1_cache = (
            not self.is_broadcasted(original_index)
            and not self.inside_reduction
            and not has_read_deps
            and is_coalesced  # for indirect loads is_coalesced is False?
        )
        cachemod = ""
        if skip_l1_cache:
            cachemod = ", cache_modifier='.cg'"

        append_broadcast = None
        shape: BlockShapeType = None

        if should_unwrap_unspec_arg(name):
            line = var
            # unwrapped bf16/fp16 0d tensors are passed in as float32 scalars
            # see triton_utils.py:signature_of
            if dtype in (torch.float16, torch.bfloat16):
                if config.triton.codegen_upcast_to_fp32:
                    dtype = torch.float32
                else:
                    line += f".to({triton_type(dtype)})"
            shape = ()

        else:
            if isinstance(indexing, (BlockPtrOptions, TensorDescriptorOptions)):
                block_descriptor, other = self.codegen_block_ptr(
                    name, var, indexing, other
                )
                if isinstance(indexing, BlockPtrOptions):
                    line = f"tl.load({block_descriptor}{other}{ep}{cachemod})"
                else:
                    line = f"{block_descriptor}.load({V.kernel.index_to_str(indexing.offsets)})"
                line = indexing.codegen_broadcast_and_reshape(
                    line,
                    indexing.block_shape,
                    indexing.final_shape,
                    allow_implicit=True,
                    for_store=False,
                )
                shape = indexing.final_shape
            elif is_sympy_integer_like(original_index):
                line = f"tl.load({var} + ({original_index}))"
                append_broadcast = indexing.expand_str
                shape = ()
            else:
                line = f"tl.load({var} + ({indexing.index_str}), {indexing.mask_str}{ep}{other}{cachemod})"

                # The block shape of tl.load depends on the indexing expression.
                # Inferring shape solely from the mask may miss cases where the mask is constant.
                # Inferring from indexing.expand_shape alone may also fail when dense indexing is absent.
                # so, iterate over variables in the indexexpr to accurately infer the block shape.
                if indexing.expand_shape:
                    shape = indexing.expand_shape
                else:
                    shape = TritonSymbols.get_block_shape(indexing.index)

            if (
                dtype in (torch.float16, torch.bfloat16)
                and config.triton.codegen_upcast_to_fp32
            ):
                line += ".to(tl.float32)"
                dtype = torch.float32
            if dtype == torch.bool and torch.version.hip is None:
                # Workaround for https://github.com/triton-lang/triton/issues/2151
                # tl.load returns int8 when loading from pointer to int1
                # NOTE: Currently causes hangs on bool UTs for ROCm
                line += ".to(tl.int1)"
                dtype = torch.bool

        load_buffer = self.get_load_buffer(indexing)
        self._handle_pdl_before_access(load_buffer, name)
        result_var = self.cse.generate(
            load_buffer, make_line(line), dtype=dtype, shape=shape
        )
        self._handle_pdl_after_load(load_buffer, result_var)
        if result_var.use_count > 1:
            load_counts[name] -= 1  # don't double count cache hit
        assert isinstance(result_var, TritonCSEVariable)
        result_var.mask_vars = indexing.mask_vars  # type: ignore[assignment]

        if append_broadcast:
            line = f"tl.broadcast_to({result_var}, {append_broadcast})"
            result_var = self.cse.generate(
                load_buffer, line, dtype=dtype, shape=indexing.expand_shape
            )
            if indexing.mask_vars:
                if dtype.is_floating_point:
                    zero = "0.0"
                elif dtype == torch.bool:
                    zero = "True"
                else:
                    zero = "0"
                other_val = (
                    constant_repr(self._load_other) if self._load_other else zero
                )
                line = f"tl.where({indexing.mask_str}, {result_var}, {other_val})"
                result_var = self.cse.generate(
                    load_buffer, line, dtype=dtype, shape=result_var.shape
                )

        if not self.inside_reduction or (not indexing.has_rmask() and not has_rindex):
            self.outside_loop_vars.add(result_var)

        return result_var

    def store(
        self, name: str, index: sympy.Expr, value: CSEVariable, mode: StoreMode = None
    ) -> None:
        """
        store the 'value' to the memory location 'name', offset by some indexing expression 'index'.
        """

        var = self.args.output(name)
        original_index = index
        dtype = V.graph.get_dtype(name)

        tma_compatibility_checker = None
        if mode is None or mode == "tma":
            force = mode == "tma"
            tma_compatibility_checker = self.tma_compatibility_checker_cls(
                self,
                dtype,
                for_store=True,
                force=force,
            )
        indexing = self.indexing(
            index,
            dense_indexing=True,
            block_ptr=mode is None,
            tma_compatibility_checker=tma_compatibility_checker,
        )

        if isinstance(indexing, IndexingOptions) and self._has_stride1_on_rdim(
            indexing.index
        ):
            self.stores_with_contiguous_rdim.append(name)

        # Guard against write-after-read corruption in triton.
        # See # https://github.com/triton-lang/triton/issues/1615
        # This triton bug means that a load which is broadcasted over multiple
        # warps may see the result of a store that happens later in the triton
        # program. The workaround is to add a barrier before storing, which
        # enforces that all warps have already read the data.
        is_inplace = name in self.args.inplace_buffers
        is_broadcasted = self.is_broadcasted(original_index)
        if is_inplace and is_broadcasted:
            self.stores.writeline(DeferredLine(name, "tl.debug_barrier()"))

        if isinstance(indexing, (BlockPtrOptions, TensorDescriptorOptions)):
            block_descriptor, other = self.codegen_block_ptr(name, var, indexing)
            # block_ptr / tma descriptor stores don't do implicit casting
            line = self.codegen_block_ptr_store_line(
                name, indexing, block_descriptor, value, other
            )
        elif mode is None:
            # If indexing is an integer and value has block shape larger than one,
            # broadcasting fails. So, we manually broadcast indexing to the value shape.
            # Without broadcast :
            # tl.store(out_ptr0 + (tl.full([1, 1], 0, tl.int32)), tmp4, xmask) # Fail
            #
            # With broadcast:
            # tl.store(out_ptr0 + (tl.full([1, 1], 0, tl.int32).broadcast_to((XBLOCK,1)), tmp4, xmask)
            indexing_str = indexing.index_str
            if (
                is_sympy_integer_like(index)
                and value.shape is not None
                and not all(str(x) == "1" for x in value.shape)
            ):
                value_shape = ", ".join(map(str, value.shape))
                indexing_str += f".broadcast_to({value_shape})"
            line = f"tl.store({var} + ({indexing_str}), {value}, {indexing.mask_str})"
        elif mode == "atomic_add":
            self.atomic_add_found = True
            indexing_str = indexing.index_str
            if (
                is_sympy_integer_like(index)
                and value.shape is not None
                and not all(str(x) == "1" for x in value.shape)
            ):
                value_shape = ", ".join(map(str, value.shape))
                indexing_str += f".broadcast_to({value_shape})"
            line = f"tl.atomic_add({var} + ({indexing_str}), {value}, {indexing.mask_str}, sem='relaxed')"
        else:
            raise NotImplementedError(f"store mode={mode}")

        exit_stack = contextlib.ExitStack()
        if not self.inside_reduction and self.cooperative_reduction:
            exit_stack.enter_context(self.guard_cooperative_store(name, self.stores))

        self._handle_pdl_before_access(self.stores, name, consider_reads=True)
        self.stores.writeline(DeferredLine(name, line))

        if not self.inside_reduction:
            self.outside_loop_vars.add(value)

        exit_stack.close()

    def device_assert_async(self, cond, msg) -> None:
        self.compute.writeline(f"tl.device_assert({cond}, {repr(msg)})")

    def guard_cooperative_store(self, name, buffer):
        """
        For cooperative reductions only one thread block should write out the result.
        We rotate which thread block does each write for better parallelism
        """
        idx = self.cooperative_reduction_workspace_cache.increment_store_count()
        buffer.writeline(DeferredLine(name, f"if rsplit_id == ({idx} % RSPLIT):"))
        return buffer.indent()

    def _combine_masks(self, *variables: Optional[CSEVariable]):
        masks = None
        for elem in variables:
            if elem is None:
                continue
            if hasattr(elem, "mask_vars"):
                if masks is None:
                    masks = elem.mask_vars
                else:
                    masks = masks | elem.mask_vars
        return masks

    def bucketize(
        self,
        values: CSEVariable,
        boundaries: tuple[str, sympy.Expr, sympy.Expr, sympy.Expr],
        boundary_indices: CSEVariable,
        indexing_dtype: torch.dtype,
        right: bool,
        sorter: Optional[tuple[str, sympy.Expr]] = None,
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

        if indexing_dtype == torch.int32:
            triton_dtype = "tl.int32"
        elif indexing_dtype == torch.int64:
            triton_dtype = "tl.int64"
        else:
            raise NotImplementedError(
                "Bucketize only supports indexing with int32 and int64"
            )

        self._handle_pdl_before_access(
            self.compute, boundaries[0], *([sorter[0]] if sorter else [])
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
            ")",
            dtype=indexing_dtype,  # type: ignore[attr-defined]
            shape=values.shape,
        )
        self._handle_pdl_after_load(self.compute, result)

        masks = self._combine_masks(values, boundary_indices, sorter_indices)
        result.mask_vars = masks  # type: ignore[attr-defined]

        return result

    def reduction_resize(self, value) -> str:
        ndims = self.triton_tensor_ndim()
        if ndims == 1:
            return f"triton_helpers.promote_to_tensor({value})"

        nreduce = self.num_reduction_dims
        sizes = [":"] * (ndims - nreduce) + ["None"] * nreduce
        return f"{value}[{', '.join(sizes)}]"

    def reduction_resize_and_shape(self, value, shape) -> tuple[str, BlockShapeType]:
        ndims = self.triton_tensor_ndim()
        if ndims == 1:
            return f"triton_helpers.promote_to_tensor({value})", shape

        nreduce = self.num_reduction_dims
        sizes = [":"] * (ndims - nreduce) + ["None"] * nreduce
        new_shape = (
            (*shape[: (ndims - nreduce)], *[1] * nreduce) if shape is not None else None
        )
        return f"{value}[{', '.join(sizes)}]", new_shape

    def reduction_collapse_dims(
        self, buffer, value: CSEVariable, dtype: torch.dtype
    ) -> CSEVariable:
        """
        Reshape to RBLOCK, collapsing all reduction dims.
        """
        # This is not needed for 1D reductions.
        if self.num_reduction_dims == 1:
            return value

        target_ndim = self.triton_tensor_ndim() - self.num_reduction_dims
        initial_shape = self.dense_size_list()
        target_shape = initial_shape[:target_ndim] + ["RBLOCK"]
        return self.cse.generate(
            buffer,
            triton_reshape(str(value), initial_shape, target_shape),
            dtype=dtype,
            shape=tuple(target_shape),
        )

    def reduction(
        self,
        dtype: torch.dtype,
        src_dtype: torch.dtype,
        reduction_type: ReductionType,
        value: Union[CSEVariable, tuple[CSEVariable, ...]],
    ) -> Union[CSEVariable, tuple[CSEVariable, ...]]:
        """
        codegen reduction of value to Triton according the reduction_type
        """

        def maybe_upcast(value: CSEVariable) -> CSEVariable:
            # Math reductions in FP16/BF16 are less accurate because the Triton compiler does not
            # automatically promote to FP32 for accumulation. Additionally, max/min reductions
            # do not support FP16/BF16. We manually promote to FP32 here.
            return (
                ops.to_dtype(value, torch.float32)
                if value.dtype
                in [
                    torch.float16,
                    torch.bfloat16,
                ]
                else value
            )

        original_dtypes = [val.dtype for val in pytree.tree_leaves(value)]
        value = pytree.tree_map(maybe_upcast, value)
        if any(x in [torch.float16, torch.bfloat16] for x in original_dtypes):
            # Only promote FB16/BF16; do not promote other integer/boolean dtypes
            src_dtype = torch.promote_types(src_dtype, torch.float32)
            dtype = torch.promote_types(dtype, torch.float32)

        assert self.inside_reduction
        masks = OrderedSet(f"{tree.prefix}mask" for tree in self.range_trees)
        self.filter_masks(masks)
        masks = sorted(masks)
        if self._load_mask:
            masks.append(self._load_mask)
        reduction_range_prefix = self.range_trees[-1].prefix[0]

        # When we do native matmtul codegen,
        # we don't want to keep the R0_BLOCK/R1_BLOCK in the accumulator.
        # so instead of naively calling dense_size_str(), we filter out
        # reduction block from accumulator and only keep (Y,X).
        # In bmm (Z,Y,R)x(Z,R,X) case, we also remove z dimension from accumulator
        # because 3d (Z,Y,X) tl.dot is somehow slower than 2d tl.dot.
        # Instead, we force ZBLOCK to be always 1 during autotune.
        dense_size_str: str
        if self.is_native_matmul:
            dense_sizes = self.dense_size_list()
            assert len(dense_sizes) >= 3
            xy_sizes_only = [size for size in dense_sizes if "X" in size or "Y" in size]
            dense_size_str = f"[{', '.join(xy_sizes_only)}]"
            value_shape = tuple(xy_sizes_only)
        else:
            dense_size_str = self.dense_size_str()
            value_shape = tuple(self.dense_size_list())

        # Say we have
        #     tmp0 = ops.constant(1, torch.int64)
        #     tmp1 = ops.reduction(torch.int64, torch.int64, "sum", tmp0)
        # tmp0 in the triton code is either a scalar, or single-element tensor
        # so if we emit tl.sum directly, it will only give 1 instead of RBLOCK * 1
        # To avoid this, we broadcast to the expected shape first.
        value = self._map_tuple_or_scalar(
            lambda v: self.cse.generate(
                self.compute,
                f"tl.broadcast_to({v}, {dense_size_str})",
                dtype=v.dtype,
                shape=value_shape,
            ),
            value,
        )

        logical_index = None
        if reduction_type in ("argmin", "argmax"):
            if isinstance(value, tuple):
                value, logical_index = value

        dim = self.triton_tensor_ndim() - self.num_reduction_dims
        root_op: str

        def final_reduction(
            buffer,
            value: CSEVariable,
            result_type: Optional[torch.dtype],
        ) -> tuple[str, Optional[torch.dtype], BlockShapeType]:
            """
            Helper to generate a reduction call, e.g. tl.sum.
            """
            triton_reduction_fn = get_triton_reduction_function(reduction_type)

            value = self.reduction_collapse_dims(buffer, value, dtype)
            if reduction_type == "dot":
                # Native matmul is a special case because accumulator shape is fixed to (Y,X)
                is_bmm = len(self.dense_size_list()) == 4
                assert value.shape is not None
                if is_bmm:
                    result = f"{value}[None,:,:,None]"  # (Y,X) to (Z=1,Y,X,R=1)
                    shape = [1, *value.shape, 1]
                else:
                    result = f"{value}[:,:,None]"  # (Y,X) to (Y,X,R=1)
                    shape = [*value.shape, 1]
            else:
                result, shape = self.reduction_resize_and_shape(  # type: ignore[assignment]
                    f"{triton_reduction_fn}({value}, {dim})", value.shape
                )

            if result_type is not None:
                result = f"{result}.to({self.dtype_to_str(result_type)})"
            else:
                result_type = value.dtype

            return result, result_type, shape

        def final_reduction_define(
            buffer,
            result_var: CSEVariable,
            value: CSEVariable,
            result_type: Optional[torch.dtype],
        ) -> None:
            """
            Generate a reduction and assign it to an existing variable.
            """
            # pyrefly: ignore [bad-assignment]
            value, _, _ = final_reduction(buffer, value, result_type)
            buffer.splice(f"{result_var} = {value}")

        def final_argreduce(buffer, result_var, value, index):
            value = self.reduction_collapse_dims(buffer, value, dtype)
            index = self.reduction_collapse_dims(buffer, index, dtype)
            buffer.splice(
                f"""\
                {result_var}_val, {result_var}_idx = triton_helpers.{root_op}_with_index({value}, {index}, {dim})
                {result_var} = {self.reduction_resize(f"{result_var}_idx")}
                """
            )

        cache_key = (src_dtype, reduction_type, value)
        if cache_key in self.cse.reduction_cache:
            return self.cse.reduction_cache[cache_key]

        acc_type = triton_acc_type(src_dtype)
        torch_acc_type = upcast_acc_dtype(src_dtype)
        result_shape = list(self.dense_size_list())
        result_shape[dim] = "1"
        result_var: Any = self.cse.newvar(
            dtype=torch_acc_type, shape=tuple(result_shape)
        )
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

            def update_constant_dtype(constant, src_dtype, dst_dtype):
                "update reduction constant mask value to match dst_dtype"

                # int is the only mask which may not fit within lower bitwidth,
                # because float uses inf/-inf
                if src_dtype.is_floating_point or src_dtype == torch.bool:
                    return constant

                if src_dtype == dst_dtype or constant == 0:
                    return constant

                if constant == torch.iinfo(src_dtype).max:
                    return torch.iinfo(dst_dtype).max
                elif constant == torch.iinfo(src_dtype).min:
                    return torch.iinfo(dst_dtype).min
                else:
                    return constant

            def _mask_value(value, default) -> CSEVariable:
                default = update_constant_dtype(default, src_dtype, value.dtype)
                default_str = self._map_tuple_or_scalar(constant_repr, default)

                return self.cse.generate(
                    self.compute,
                    where_cond(value, default_str),
                    dtype=value.dtype,
                    shape=value.shape,
                )

            masked_value: Union[CSEVariable, Sequence[CSEVariable], None]
            if reduction_type == "online_softmax_reduce":
                # Don't generate mask value for online_softmax since we
                # will fallback below
                masked_value = None
            elif isinstance(value, tuple):
                masked_value = [_mask_value(v, d) for v, d in zip(value, default)]  # type: ignore[arg-type]
            elif reduction_type == "dot":
                # Here, we don't perform the masking.
                # Masking w/ where condition in native matmul is handled in ops.dot codegen.
                # Since tl.dot performs reduction within the triton block,
                # masking should happen before the tl.dot is called.
                masked_value = self.cse.generate(self.compute, value, dtype=value.dtype)
            else:
                masked_value = _mask_value(value, default)

            if reduction_type in ("argmax", "argmin"):
                assert isinstance(masked_value, CSEVariable)
                accumulator_dtype = V.kernel.get_index_dtype_as_torch_dtype()
                if logical_index:
                    accumulator_index = f"({str(logical_index)}).to({self.dtype_to_str(accumulator_dtype)})"
                else:
                    accumulator_index = str(
                        self.cse.generate(
                            self.compute,
                            f"tl.broadcast_to({reduction_range_prefix}index, {masked_value}.shape)",
                            dtype=accumulator_dtype,
                            shape=masked_value.shape,
                        )
                    )
                root_op = {"argmax": "max", "argmin": "min"}[reduction_type]
                final_argreduce(
                    self.compute, result_var, masked_value, accumulator_index
                )
                result_var.dtype = accumulator_dtype
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
                assert isinstance(masked_value, Sequence)
                (mean, m2, weight) = masked_value
                result_var = tuple(
                    self.cse.generate(self.compute, value, dtype=dtype, shape=shape)
                    for value, shape in self._welford(
                        self.compute, mean, m2, weight, dim, dtype
                    )
                )
            elif reduction_type == "online_softmax_reduce":
                # All data is loaded to register anyway, no need to do
                # online softmax
                result_var = self.prepare_softmax_twopass_fallback(dtype, value)
            else:
                assert isinstance(masked_value, CSEVariable)
                _result, _dtype, _shape = final_reduction(
                    self.compute, masked_value, masked_value.dtype
                )
                result_var = self.cse.generate(
                    self.compute, _result, dtype=_dtype, shape=_shape
                )
        else:
            accumulator = self.cse.namedvar(
                f"_{result_var}",
                dtype=torch_acc_type,
                shape=tuple(self.dense_size_list()),
            )
            default = ir.Reduction.default_accumulator(reduction_type, src_dtype)
            default = self._map_tuple_or_scalar(constant_repr, default)
            if not isinstance(default, tuple):
                if reduction_type == "dot":
                    dense_sizes = self.dense_size_list()
                    assert len(dense_sizes) >= 3
                    xy_sizes_only = [
                        size for size in dense_sizes if "X" in size or "Y" in size
                    ]
                    accumulator.shape = tuple(xy_sizes_only)
                    dense_size_str = f"[{', '.join(xy_sizes_only)}]"
                    self.body.writeline(
                        f"{accumulator} = tl.full({dense_size_str}, {default}, {acc_type})"
                    )
                else:
                    self.body.writeline(
                        f"{accumulator} = tl.full({self.dense_size_str()}, {default}, {acc_type})"
                    )

            if reduction_type in ("argmax", "argmin"):
                accumulator_index = f"_{result_var}_index"
                index_dtype = self.features.select_index_dtype()
                self.body.writeline(
                    f"{accumulator_index} = tl.full({self.dense_size_str()}, "
                    f"{torch.iinfo(index_dtype).max}, {self.dtype_to_str(index_dtype)})"
                )
                root_op = {"argmax": "max", "argmin": "min"}[reduction_type]
                # Use logical_index if it was unpacked, otherwise fall back to physical index
                index_var = (
                    f"({str(logical_index)}).to({self.dtype_to_str(index_dtype)})"
                    if logical_index is not None
                    else f"{reduction_range_prefix}index"
                )
                self.compute.splice(
                    f"""\
                {accumulator}_next, {accumulator_index}_next = triton_helpers.{root_op}imum_with_index(
                    {accumulator}, {accumulator_index}, {value}, {index_var}
                )
                {accumulator} = {where_cond(f"{accumulator}_next", accumulator)}
                {accumulator_index} = {where_cond(f"{accumulator_index}_next", accumulator_index)}
                """
                )
                final_argreduce(
                    self.post_loop_combine, result_var, accumulator, accumulator_index
                )
            elif is_welford_reduction(reduction_type):
                result_var = self.welford_reduce(
                    result_var, reduction_type, value, where_cond, acc_type, dtype
                )
            elif reduction_type == "online_softmax_reduce":
                accumulator_max = f"_{result_var}_max"
                accumulator_sum = f"_{result_var}_sum"

                # setup accumulator
                self.body.writeline(
                    f"{accumulator_max} = tl.full({self.dense_size_str()}, float('-inf'), {acc_type})"
                )
                self.body.writeline(
                    f"{accumulator_sum} = tl.zeros({self.dense_size_str()}, {acc_type})"
                )

                # combine
                # Note, we pass config.use_fast_math to the JITFunction
                # since a triton kernel can not access a config.
                self.compute.splice(
                    f"""
                    {accumulator_max}_next, {accumulator_sum}_next = triton_helpers.online_softmax_combine(
                        {accumulator_max}, {accumulator_sum}, {value}, {config.use_fast_math}
                    )
                    """
                )

                # mask
                self.compute.splice(
                    f"""
                    {accumulator_max} = {where_cond(f"{accumulator_max}_next", accumulator_max)}
                    {accumulator_sum} = {where_cond(f"{accumulator_sum}_next", accumulator_sum)}
                    """
                )

                # reduce. Similar to the final reduction for coopereative
                # reduction
                result_max = result_var
                result_sum = self.cse.newvar(dtype=dtype, shape=result_max.shape)

                result_var = self.online_softmax_reduce_final_reduction(
                    self.post_loop_combine,
                    result_max,
                    result_sum,
                    accumulator_max,
                    accumulator_sum,
                    dim,
                    dtype,
                )
            else:
                combine_fn = ir.get_reduction_combine_fn(reduction_type, src_dtype)
                updated = combine_fn(accumulator, value)
                if reduction_type == "dot":
                    self.compute.writeline(f"{accumulator} = {updated}")
                else:
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
                    accumulator = self.cse.generate(
                        self.post_loop_combine,
                        f"{accumulator}.to(tl.int8)",
                        dtype=torch.int8,
                        shape=accumulator.shape,
                    )

                final_reduction_define(
                    self.post_loop_combine, result_var, accumulator, None
                )

        if self.cooperative_reduction:
            default = ir.Reduction.default_accumulator(reduction_type, src_dtype)
            exit_stack = contextlib.ExitStack()
            for buf in (self.post_loop_combine, self.post_loop_store):
                # only do cooperative reduction combines if we have more than one thread block
                buf.writeline("if HAS_RSPLIT:")
                exit_stack.enter_context(buf.indent())

            if reduction_type in ("argmax", "argmin"):
                self.post_loop_combine.writeline(
                    f"{result_var}_bval = {self.reduction_resize(f'{result_var}_val')}"
                )
                peer_val = self.codegen_cooperative_reduction_peer_combine(
                    f"{result_var}_bval", src_dtype, default
                )
                index_dtype = self.features.select_index_dtype()
                peer_idx = self.codegen_cooperative_reduction_peer_combine(
                    result_var, index_dtype, torch.iinfo(index_dtype).max
                )
                final_argreduce(self.post_loop_store, result_var, peer_val, peer_idx)
            elif is_welford_reduction(reduction_type):
                assert reduction_type == "welford_reduce"
                result_mean, result_m2, result_weight = result_var
                peer_mean = self.codegen_cooperative_reduction_peer_combine(
                    result_mean,
                    upcast_acc_dtype(src_dtype),
                    default[0],  # type: ignore[index]
                )
                peer_m2 = self.codegen_cooperative_reduction_peer_combine(
                    result_m2,
                    upcast_acc_dtype(src_dtype),
                    default[1],  # type: ignore[index]
                )
                peer_weight = self.codegen_cooperative_reduction_peer_combine(
                    result_weight,
                    upcast_acc_dtype(src_dtype),
                    default[2],  # type: ignore[index]
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
                    dtype,
                )
            elif reduction_type == "online_softmax_reduce":
                result_max, result_sum = result_var
                assert isinstance(default, Sequence)
                peer_max = self.codegen_cooperative_reduction_peer_combine(
                    result_max, upcast_acc_dtype(src_dtype), default[0]
                )
                peer_sum = self.codegen_cooperative_reduction_peer_combine(
                    result_sum, upcast_acc_dtype(src_dtype), default[1]
                )
                self.online_softmax_reduce_final_reduction(
                    self.post_loop_store,
                    result_max,
                    result_sum,
                    peer_max,
                    peer_sum,
                    dim,
                    dtype,
                )
            else:
                peers = self.codegen_cooperative_reduction_peer_combine(
                    result_var, upcast_acc_dtype(src_dtype), default
                )
                final_reduction_define(self.post_loop_store, result_var, peers, None)
            exit_stack.close()

        self.cse.reduction_cache[cache_key] = result_var

        if isinstance(result_var, tuple):
            assert all(isinstance(x, TritonCSEVariable) for x in result_var)
            self.outside_loop_vars.update(result_var)

            # Match output dtype with input dtype
            if reduction_type in ("welford_reduce", "online_softmax_reduce"):
                assert len(original_dtypes) == 1
                original_dtypes = len(result_var) * original_dtypes

            assert len(result_var) == len(original_dtypes)
            for var, orig_dtype in zip(result_var, original_dtypes):
                assert orig_dtype is not None
                if var.dtype != orig_dtype:
                    self.post_loop_combine.writeline(
                        f"{var} = {var}.to({triton_compute_type(orig_dtype)})"
                    )
        else:
            assert isinstance(result_var, TritonCSEVariable)
            self.outside_loop_vars.add(result_var)

            # Match output dtype with input dtype
            if result_var.dtype != original_dtypes[0]:
                assert original_dtypes[0] is not None
                self.post_loop_combine.writeline(
                    f"{result_var} = {result_var}.to({triton_compute_type(original_dtypes[0])})"
                )

        return result_var

    def _online_softmax_reduce(
        self, buffer, accumulator_max, accumulator_sum, dim, dtype: torch.dtype
    ):
        accumulator_max = self.reduction_collapse_dims(buffer, accumulator_max, dtype)
        accumulator_sum = self.reduction_collapse_dims(buffer, accumulator_sum, dtype)
        result_max, result_sum = [str(self.cse.newvar(dtype=dtype)) for _ in range(2)]
        buffer.splice(
            f"""
            {result_max}, {result_sum} = triton_helpers.online_softmax_reduce(
                {accumulator_max}, {accumulator_sum}, {dim}, {config.use_fast_math})
            {result_max} = {self.reduction_resize(f"{result_max}")}
            {result_sum} = {self.reduction_resize(f"{result_sum}")}
            """
        )

        return result_max, result_sum

    def _welford(self, buffer, mean, m2, weight, dim, dtype: torch.dtype):
        """
        Helper to codegen triton_helpers.welford.
        """
        mean, m2, weight = (
            self.reduction_collapse_dims(buffer, value, dtype)
            for value in (mean, m2, weight)
        )
        welford = f"triton_helpers.welford({mean}, {m2}, {weight}, {dim})"

        def reduced_shape(shape):
            return tuple(shape[0:dim] + shape[dim + 1 :])

        welford_results = [
            self.cse.newvar(dtype=dtype, shape=reduced_shape(value.shape))
            for value in (mean, m2, weight)
        ]
        buffer.writeline(f"{', '.join([str(r) for r in welford_results])} = {welford}")

        return tuple(
            self.reduction_resize_and_shape(value, value.shape)
            for value in welford_results
        )

    def welford_reduce(
        self, result_var, reduction_type, value, where_cond, acc_type, dtype
    ):
        """Helper to codegen a welford reduction"""
        dim = self.triton_tensor_ndim() - self.num_reduction_dims

        accumulator = TritonCSEVariable(
            f"{result_var}_mean",
            shape=tuple(self.dense_size_list()),
            dtype=acc_type,
            bounds=ValueRanges.unknown(),
        )
        accumulator_m2 = TritonCSEVariable(
            f"{result_var}_m2",
            shape=tuple(self.dense_size_list()),
            dtype=acc_type,
            bounds=ValueRanges.unknown(),
        )
        accumulator_weight = TritonCSEVariable(
            f"{result_var}_weight",
            shape=tuple(self.dense_size_list()),
            dtype=acc_type,
            bounds=ValueRanges.unknown(),
        )
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
            {accumulator} = {where_cond(f"{accumulator}_next", accumulator)}
            {accumulator_m2} = {where_cond(f"{accumulator_m2}_next", accumulator_m2)}
            {accumulator_weight} = {where_cond(f"{accumulator_weight}_next", accumulator_weight)}
            """
        )
        result_mean = result_var
        return self.welford_reduce_final_reduction(
            self.post_loop_combine,
            result_mean,
            None,
            None,
            accumulator,
            accumulator_m2,
            accumulator_weight,
            dim,
            dtype,
        )

    def welford_reduce_final_reduction(
        self,
        buffer,
        result_mean,
        result_m2,
        result_weight,
        mean,
        m2,
        weight,
        dim,
        dtype,
    ):
        """Helper to codegen call to triton_helpers.welford"""
        values = list(self._welford(buffer, mean, m2, weight, dim, dtype))

        result_exprs = [result_mean, result_m2, result_weight]
        for i, (result_expr, (value, shape)) in enumerate(zip(result_exprs, values)):
            if result_expr is None:
                result_expr = self.cse.newvar(dtype=dtype, shape=shape)
                result_exprs[i] = result_expr
            buffer.splice(f"{result_expr} = {value}")

        return tuple(result_exprs)

    def online_softmax_reduce_final_reduction(
        self, buffer, result_max, result_sum, peer_max, peer_sum, dim, dtype
    ):
        accumulator_max = self.reduction_collapse_dims(buffer, peer_max, dtype)
        accumulator_sum = self.reduction_collapse_dims(buffer, peer_sum, dtype)
        buffer.splice(
            f"""
            {result_max}, {result_sum} = triton_helpers.online_softmax_reduce(
                {accumulator_max}, {accumulator_sum}, {dim}, {config.use_fast_math})
            {result_max} = {self.reduction_resize(f"{result_max}")}
            {result_sum} = {self.reduction_resize(f"{result_sum}")}
            """
        )
        return result_max, result_sum

    def max_rsplit(self):
        if self.fixed_config:
            return self.fixed_config["RSPLIT"]
        return TRITON_MAX_RSPLIT

    def codegen_cooperative_reduction_peer_combine(
        self, result_var, dtype, default_val
    ) -> CSEVariable:
        """
        Generate code to save a [XBLOCK, RSPLIT] temporary workspace, where each thread block writes a different
        column.  After the barrier, every thread block loads the completed value so that it can compute the final
        value independently.
        """
        xnumel = self.numels["x"]
        mask = "xindex < xnumel" if not self._has_constant_xmask() else None

        nbytes = xnumel * dtype.itemsize * self.max_rsplit()
        ws_name, ws_offset = self.cooperative_reduction_workspace_cache.allocate(nbytes)

        self.post_loop_combine.splice(
            f"""
                {result_var}_ws = ({ws_name} + {self.index_to_str(ws_offset)}).to(tl.pointer_type({triton_type(dtype)}))
                tl.store({result_var}_ws + (xindex * RSPLIT + rsplit_id), {result_var}, {mask})
            """,
            strip=True,
        )
        peers = self.create_cse_var(
            f"{result_var}_peers",
            shape=["XBLOCK", "RSPLIT"],
            dtype=dtype,
            bounds=ValueRanges.unknown(),
        )
        self.post_loop_store.writeline(
            f"{peers} = tl.load({result_var}_ws + (xindex * RSPLIT + rsplit_arange), "
            f"rsplit_mask, eviction_policy='evict_first', other=triton_helpers.if_mask(rsplit_mask, {constant_repr(default_val)}))"
        )
        return peers

    def store_reduction(
        self,
        name: str,
        index: sympy.Expr,
        value: CSEVariable,
    ):
        assert self.inside_reduction
        self.inside_reduction = False
        dtype = V.graph.get_dtype(name)
        indexing = self.indexing(
            index,
            block_ptr=True,
            tma_compatibility_checker=self.tma_compatibility_checker_cls(
                kernel=self,
                dtype=dtype,
                for_store=True,
                force=False,
            ),
        )
        self.inside_reduction = True
        var = self.args.output(name)

        exit_stack = contextlib.ExitStack()
        if self.cooperative_reduction:
            exit_stack.enter_context(
                self.guard_cooperative_store(name, self.post_loop_store)
            )

        self._handle_pdl_before_access(self.post_loop_store, var)

        if isinstance(indexing, (BlockPtrOptions, TensorDescriptorOptions)):
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

            indexing_str = indexing.index_str
            if (
                is_sympy_integer_like(index)
                and value.shape is not None
                and not all(str(x) == "1" for x in value.shape)
            ):
                value_shape = ", ".join(map(str, value.shape))
                indexing_str += f".broadcast_to({value_shape})"

            self.post_loop_store.writeline(
                DeferredLine(
                    name,
                    f"tl.store({var} + ({indexing_str}), {value}, {indexing.mask_str})",
                )
            )

        exit_stack.close()

    def _lift_helper(
        self, fn, values: tuple[CSEVariable, ...], dtypes: tuple[torch.dtype, ...]
    ) -> str:
        # Lift IR function for scan operations into a triton function
        # in the global namespace
        helper = IndentedBuffer()
        helper.writeline("@triton.jit")
        cse = CSE()

        args = [
            tuple(
                cse.namedvar(f"arg{i}_{n}", dtype=dtype, shape=value.shape)
                for n, (value, dtype) in enumerate(zip(values, dtypes))
            )
            for i in range(2)
        ]
        signature = ", ".join(str(x) for x in itertools.chain.from_iterable(args))
        helper.writeline(f"def {{name}}({signature}):")

        overrides = TritonOverrides()

        # Build a name that changes depending on fn to workaround a triton bug
        # where the combine_fn to reduce and scan is not hashed, and so different
        # scan ops may collide in the triton cache.
        # This is fixed with the latest triton pin, but not the triton-rocm pin.
        helper_name = "_triton_helper_fn"

        from torch._inductor.dtype_propagation import DtypePropagationOpsHandler
        from torch._inductor.shape_propagation import ShapePropagationOpsHandler

        shape_handler = ShapePropagationOpsHandler()
        dtype_handler = DtypePropagationOpsHandler()

        class CSEProxy(DefaultHandler):
            def _default(
                self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]
            ) -> Any:
                nonlocal helper_name
                helper_name += f"_{name}"

                output_dtype = getattr(
                    dtype_handler,
                    name,
                )(*args, **kwargs)

                output_shape = getattr(
                    shape_handler,
                    name,
                )(*args, **kwargs)

                return cse.generate(
                    helper,
                    getattr(overrides, name)(*args, **kwargs),
                    dtype=output_dtype,
                    shape=output_shape,
                )

        with helper.indent(), V.set_ops_handler(CSEProxy()):
            outputs = fn(*args)
            outputs = ", ".join(str(output) for output in outputs)
            helper.writeline(f"return {outputs}")

        return self.helper_functions.add(helper.getvalue(), base_name=helper_name)

    def scan(
        self,
        dtypes: tuple[torch.dtype, ...],
        combine_fn: Callable[
            [tuple[CSEVariable, ...], tuple[CSEVariable, ...]], tuple[CSEVariable, ...]
        ],
        values: tuple[CSEVariable, ...],
    ) -> tuple[CSEVariable, ...]:
        """
        Perform an associative scan on 'values'.
        """
        assert self.inside_reduction
        assert not self.cooperative_reduction, "TODO"
        masks = OrderedSet(f"{tree.prefix}mask" for tree in self.range_trees)
        self.filter_masks(masks)
        masks = sorted(masks)
        assert not self._load_mask, "ops.scan not supported inside ops.masked"

        broadcasted_values = []
        accumulators = []

        dtypes = tuple(upcast_compute_type(dtype) for dtype in dtypes)
        cse_compute = functools.partial(self.cse.generate, self.compute)
        combine_helper_fn = self._lift_helper(combine_fn, values, dtypes)
        dim = self.triton_tensor_ndim() - self.num_reduction_dims

        for value, dtype in zip(values, dtypes):
            value_dtype = self.cse.generate(
                self.compute,
                f"{value}.to({triton_compute_type(dtype)})",
                dtype=dtype,
                shape=value.shape,
            )
            value = self.cse.generate(
                self.compute,
                f"tl.broadcast_to({value_dtype}, {self.dense_size_str()})",
                dtype=dtype,
                shape=tuple(self.dense_size_list()),
            )
            broadcasted_values.append(value)

            acc_type = triton_acc_type(dtype)

            if not self.persistent_reduction:
                reduced_size = self.dense_size_list()
                reduced_size[-1] = "1"
                accumulator = self.cse.newvar(dtype=dtype, shape=reduced_size)
                reduced_size_str = f"[{', '.join(reduced_size)}]"

                default = "float('nan')" if dtype.is_floating_point else "-1"
                self.body.writeline(
                    f"{accumulator} = tl.full({reduced_size_str}, {default}, {acc_type})"
                )

                accumulators.append(accumulator)

        def csv(values):
            return " ".join(f"{value}," for value in values)

        def cse_multiple(line, values, masks, dtypes):
            n = len(values)
            cache_keys = [f"{line}, {i}, {masks}" for i in range(n)]
            if all(self.cse.contains(cache_key) for cache_key in cache_keys):
                return [self.cse.get(cache_key) for cache_key in cache_keys]
            result_vars = [
                self.cse.newvar(dtype=dtype, shape=value.shape)
                for (dtype, value) in zip(dtypes, values)
            ]
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
            broadcasted_values,
            masks,
            dtypes,
        )

        if not self.persistent_reduction:
            # tl.reduce doesn't work for non-commutative operators, so instead
            # of repeating the scan op as a reduction, we use sum to select the
            # last scan value
            def _partial_scan_shape(var):
                if var.shape is None:
                    return None
                else:
                    shape = list(var.shape)
                    shape[-1] = "1"
                    return shape

            partial_reduce_vars = [
                cse_compute(
                    f"triton_helpers.select_one(({partial_scan_var}), rbase == (RBLOCK - 1), dim=-1, keep_dims=True)",
                    dtype=upcast_compute_type(partial_scan_var.dtype),
                    shape=_partial_scan_shape(partial_scan_var),
                )
                for partial_scan_var in partial_scan_vars
            ]
            accs_next = combine_fn(tuple(accumulators), tuple(partial_reduce_vars))
            full_scan_vars = combine_fn(tuple(accumulators), partial_scan_vars)
            result_vars = [
                cse_compute(
                    f"tl.where(roffset > 0, {full_scan}, {partial_scan})",
                    dtype=partial_scan.dtype,
                    shape=partial_scan.shape,
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
            assert isinstance(result_var, TritonCSEVariable)
            result_var.mask_vars = OrderedSet(masks)

        return tuple(result_vars)

    def sort(
        self,
        dtypes: tuple[torch.dtype, ...],
        values: tuple[CSEVariable, ...],
        stable: bool,
        descending: bool,
    ) -> tuple[CSEVariable, ...]:
        assert self.inside_reduction
        assert not self.cooperative_reduction, "TODO"
        masks = OrderedSet(f"{tree.prefix}mask" for tree in self.range_trees)
        self.filter_masks(masks)
        masks = sorted(masks)
        assert not self._load_mask, "ops.sort not supported inside ops.masked"
        assert self.persistent_reduction, (
            "ops.sort is only supported in persistent reductions"
        )

        cse_compute = functools.partial(self.cse.generate, self.compute)
        dim = self.triton_tensor_ndim() - self.num_reduction_dims

        dtypes = tuple(upcast_compute_type(dtype) for dtype in dtypes)
        assert len(dtypes) == len(values)
        broadcasted_values = [
            cse_compute(
                f"tl.broadcast_to({value}, {self.dense_size_str()})",
                dtype=dtypes[i],
                shape=tuple(self.dense_size_list()),
            )
            for i, value in enumerate(values)
        ]

        def csv(values):
            return " ".join(f"{value}," for value in values)

        def cse_multiple(line, broadcasted_values, masks, dtypes):
            n = len(broadcasted_values)
            cache_keys = [f"{line}, {i}, {masks}" for i in range(n)]
            if all(self.cse.contains(cache_key) for cache_key in cache_keys):
                return [self.cse.get(cache_key) for cache_key in cache_keys]
            result_vars = [
                self.cse.newvar(dtype=dtype, shape=value.shape)
                for dtype, value in zip(dtypes, broadcasted_values)
            ]  # type: ignore[attr-defined]
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
            result_vars = cse_multiple(line, broadcasted_values, masks, dtypes)
        else:
            raise AssertionError("Unhandled sort")

        for result_var, input_var in zip(result_vars, values):
            result_var.mask_vars = masks  # type: ignore[attr-defined]
            result_var.bounds = input_var.bounds

        return tuple(result_vars)

    def codegen_prologue(self, code: IndentedBuffer):
        """
        Generate the output from prologue. This should be
        extracted from the subgraph, which is why this is
        partitioned from codegen_body.
        """
        if not self.prologue:
            return

        code.splice(self.prologue)
        self.prologue.clear()
        self.prologue_cache.clear()

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

        loop_trees = [tree for tree in self.range_trees if tree.is_loop]
        if self.mix_order_reduction:
            assert self.persistent_reduction, (
                "Mix order reduction requires persistent reduction"
            )
            accumname2var = {}
            for idx, partial_accum in enumerate(self.saved_partial_accumulate):
                reduction_type = partial_accum.reduction_type
                default = ir.Reduction.default_accumulator(reduction_type, torch.float)
                default = self._map_tuple_or_scalar(constant_repr, default)
                name = f"accum{idx}"
                self.body.writeline(
                    f"{name} = tl.full([R0_BLOCK], {default}, tl.float32)[None, :]"
                )
                accumname2var[name] = self.cse.namedvar(
                    name, dtype=torch.float, shape=("1", "R0_BLOCK")
                )
            self.body.writeline("split_size = min(RSPLIT_SIZE, xnumel - xoffset)")
            self.body.writeline(
                "for _ in tl.range(0, split_size, XBLOCK, num_stages=NUM_STAGES):"
            )
            with self.body.indent(offset=1):
                # generate xmask if it's not constant
                if not self._has_constant_xmask():
                    entry = self.range_trees[0]
                    assert entry.prefix == "x"
                    x = entry.prefix
                    self.body.writeline(f"{x}mask = {entry.name} < {x}numel")
                self.body.splice(self.indexing_code)
                self.body.writelines(
                    [
                        "xindex += XBLOCK",
                    ]
                )
                self.body.splice(self.loads)
                self.body.splice(self.compute)
                self.body.splice(self.stores)
                self.body.splice(self.post_loop_store)

                # no need to sum if XBLOCK == 1, or does that matter?
                for idx, partial_accum in enumerate(self.saved_partial_accumulate):
                    var = partial_accum.value
                    name = f"accum{idx}"
                    combine_fn = ir.get_reduction_combine_fn(
                        partial_accum.reduction_type, torch.float
                    )
                    triton_reduction_function = get_triton_reduction_function(
                        partial_accum.reduction_type,
                    )
                    newval = self.cse.generate(
                        self.body,
                        f"{triton_reduction_function}({var}, 0)",
                        dtype=var.dtype,
                        shape=("R0_BLOCK",),
                    )
                    import unittest

                    with unittest.mock.patch.object(self, "compute", self.body):
                        updated = combine_fn(
                            accumname2var[name],
                            newval,
                        )
                    self.body.writeline(f"{name} = {updated}")

            for idx in range(len(self.saved_partial_accumulate)):
                self.body.writeline(
                    f"tl.store(ws_ptr + (tl.program_id(0) + {idx} * tl.num_programs(0)) * r0_numel + r0_index, accum{idx}, r0_mask)"
                )

        elif self.inside_reduction and len(loop_trees) > 0:
            # Write the loop headers.
            for level, tree in enumerate(loop_trees):
                with self.body.indent(offset=level):
                    prefix = tree.prefix
                    loop_start = "rsplit_start" if self.cooperative_reduction else "0"
                    loop_end = (
                        "rsplit_end" if self.cooperative_reduction else f"{prefix}numel"
                    )
                    # Conditionalize pipelining on HIP for Triton due to
                    # reports of numerical inaccuracies on older Triton
                    if torch.version.hip and get_triton_version() > (3, 2):
                        num_stages = ", num_stages = 2"
                    else:
                        num_stages = ""
                    self.body.writeline(
                        f"for {prefix}offset in tl.range({loop_start}, {loop_end}, {prefix.upper()}BLOCK{num_stages}):"
                    )
                with self.body.indent(offset=level + 1):
                    self.iteration_ranges_codegen_header(tree, self.body)

            # The innermost loop performs the reduction.
            with self.body.indent(offset=len(loop_trees)):
                self.codegen_reduction_indices(self.body)
                self.body.splice(self.indexing_code)
                self.body.splice(self.loads)
                self.body.splice(self.compute)
                self.body.splice(self.stores)

            # Write loop suffixes.
            for level, tree in reversed([*enumerate(loop_trees)]):
                with self.body.indent(offset=level + 1):
                    # Advance pointers at the end of each loop.
                    for block_ptr, advancement in self.pointer_advancements[
                        tree.symt
                    ].items():
                        # Subtract any advancements made in the previous loop level.
                        if level < len(loop_trees) - 1:
                            prev_tree = loop_trees[level + 1]
                            prev_advancement = self.pointer_advancements[
                                prev_tree.symt
                            ][block_ptr]
                            prev_block = TritonSymbols.get_block_size(prev_tree)
                            prev_num_iter = CeilDiv(prev_tree.numel, prev_block)
                            advancement = [
                                cur - prev * prev_num_iter
                                for cur, prev in zip(advancement, prev_advancement)
                            ]

                        self.body.writeline(
                            DeferredLine(
                                self.block_ptr_to_buffer[block_ptr],
                                f"{block_ptr} = tl.advance({block_ptr}, {V.kernel.index_to_str(advancement)})",
                            )
                        )

                # Invalidate any cache entries that came from inside the loop.
                self.cse.invalidate(self.outside_loop_vars)
                tree.cache_clear()
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
                if HAS_RSPLIT:
                    triton_helpers.x_grid_barrier({sem_ptr})
                """,
                strip=True,
            )
            self.cooperative_reduction_workspace_cache.on_loop_end()
        if not self.mix_order_reduction:
            self.body.splice(self.post_loop_store)
        self.indexing_code.clear()
        self.loads.clear()
        self.compute.clear()
        self.stores.clear()
        self.post_loop_combine.clear()
        self.post_loop_store.clear()

    def kernel_benchmark_extra_args(self) -> list[str]:
        args = []
        if self.need_numel_args():
            numel_args: list[sympy.Expr] = []
            self.add_numel_to_call_args("", numel_args, [])
            for arg in numel_args:
                if isinstance(arg, int):
                    args.append(str(arg))
                elif isinstance(arg, SymbolicCallArg):
                    hint = V.graph.sizevars.size_hint(
                        arg.inner_expr,
                        hint_override=self.hint_override,
                        fallback=config.unbacked_symint_fallback,
                    )
                    args.append(str(hint))
                elif isinstance(arg, sympy.Expr):
                    hint = V.graph.sizevars.size_hint(
                        arg,
                        hint_override=self.hint_override,
                        fallback=config.unbacked_symint_fallback,
                    )
                    args.append(str(hint))
                else:
                    raise ValueError(f"Unsupported numel argument type: {type(arg)}")
        return args

    def codegen_kernel_benchmark(self, num_gb: Optional[float]) -> IndentedBuffer:
        """
        Generates Python code for benchmarking this Triton kernel.
        - Creates example inputs (random tensors, constants, sizes).
        - Runs the kernel on the current GPU/stream.
        - Prints runtime (ms) and throughput (GB/s) using `num_gb`.
        Args:
            num_gb (float): The number of gigabytes to use for throughput calculation.
        Returns:
            IndentedBuffer: A buffer containing the generated Python benchmark code.
        """
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
                    size = V.graph.sizevars.size_hints(
                        buf.get_size(),
                        hint_override=self.hint_override,
                        fallback=config.unbacked_symint_fallback,
                    )
                    stride = V.graph.sizevars.size_hints(
                        buf.get_stride(),
                        hint_override=self.hint_override,
                        fallback=config.unbacked_symint_fallback,
                    )
                    result.writeline(
                        f"{var_name} = rand_strided({size}, {stride}, device='{buf.get_device()}', dtype={buf.get_dtype()})"  # noqa: B950 line too long
                    )
                elif arg_name in V.graph.constants:
                    # note that random seed is put in V.graph.constants
                    const_tensor = V.graph.constants[arg_name]
                    size = V.graph.sizevars.size_hints(
                        const_tensor.size(),
                        hint_override=self.hint_override,
                        fallback=config.unbacked_symint_fallback,
                    )
                    stride = V.graph.sizevars.size_hints(
                        const_tensor.stride(),
                        hint_override=self.hint_override,
                        fallback=config.unbacked_symint_fallback,
                    )
                    result.writeline(
                        f"{var_name} = rand_strided({size}, {stride}, device='{const_tensor.device}', dtype={const_tensor.dtype})"  # type: ignore[arg-type]  # noqa: B950 line too long
                    )
                elif isinstance(arg_sig, SizeArg):
                    symval_hint = V.graph.sizevars.size_hint(
                        arg_sig.expr,
                        hint_override=self.hint_override,
                        fallback=config.unbacked_symint_fallback,
                    )

                    # Force the seed_offset to be 0 so calls to the same kernel
                    # using different seed offset will have the same benchmark harness.
                    # We can dedup kernel definitions in this case.
                    if "seed_offset" in arg_sig.name:
                        symval_hint = 0
                    result.writeline(f"{var_name} = {symval_hint}")
                elif isinstance(arg_sig, WorkspaceArg):
                    device = V.graph.get_current_device_or_throw()
                    count = V.graph.sizevars.size_hint(
                        arg_sig.count, hint_override=self.hint_override
                    )
                    result.writeline(
                        f"{var_name} = torch.zeros({count}, device='{device}', dtype={arg_sig.dtype})"
                    )
                else:
                    raise KeyError(
                        f"Don't find the buffer or const tensor for {arg_name}"
                    )
                var_names.append(var_name)
            var_names.extend(self.kernel_benchmark_extra_args())
            result.writeline(f"return {', '.join(var_names)},")

        result.writelines(["\n", "\n", "def call(args):"])
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
                    f"{str(Placeholder.KERNEL_NAME)}.run(*args, stream={stream_name})"
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
                    f"return {str(Placeholder.KERNEL_NAME)}.benchmark_all_configs(*args)"
                )

        result.writelines(["\n", "\n", "if __name__ == '__main__':"])
        with result.indent():
            result.writeline(
                "from torch._inductor.runtime.benchmarking import benchmarker"
            )
            result.writeline("")

            result.writeline("args = get_args()")
            result.writeline(
                f"ms = benchmarker.benchmark(lambda: call(args), device='{V.graph.get_current_device_or_throw().type}', rep=40)"  # noqa: B950 line too long
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
        """.format(V.graph.device_ops.import_get_raw_stream_as("get_raw_stream"))
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
        inductor_meta = {
            "backend_hash": torch.utils._triton.triton_hash_with_backend(),
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
            "deterministic": config.deterministic,
            "force_filter_reduction_configs": config.test_configs.force_filter_reduction_configs,
        }

        if config.write_are_deterministic_algorithms_enabled:
            inductor_meta["are_deterministic_algorithms_enabled"] = (
                torch.are_deterministic_algorithms_enabled()
            )

        if torch.version.hip is not None:
            inductor_meta["is_hip"] = True
        if config.is_fbcode():
            inductor_meta["is_fbcode"] = True
        if config.profile_bandwidth:
            inductor_meta["profile_bandwidth"] = config.profile_bandwidth
            inductor_meta["profile_bandwidth_regex"] = config.profile_bandwidth_regex
            inductor_meta["profile_bandwidth_output"] = config.profile_bandwidth_output
            inductor_meta["profile_bandwidth_with_do_bench_using_profiling"] = (
                config.profile_bandwidth_with_do_bench_using_profiling
            )
        if config.coordinate_descent_tuning:
            inductor_meta["coordinate_descent_tuning"] = (
                config.coordinate_descent_tuning
            )
            inductor_meta["coordinate_descent_search_radius"] = (
                config.coordinate_descent_search_radius
            )
            inductor_meta["coordinate_descent_check_all_directions"] = (
                config.coordinate_descent_check_all_directions
            )
        return inductor_meta

    def codegen_kernel(self, name=None) -> str:
        """
        Convert the TritonKernel from Inductor SIMD IR to triton code, including inductor triton heuristics, imports,
        metadata, and benchmarking infra.
        """

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

        mutated_args: OrderedSet[str] = OrderedSet()
        for mutation in self.mutations:
            if mutation in self.args.input_buffers:
                mutated_args.add(self.args.input_buffers[mutation])
            if (
                mutation in self.args.inplace_buffers
                and mutation not in V.graph.removed_buffers
                and mutation not in self.removed_buffers
            ):
                mutated_args.add(
                    cast(InplacedBuffer, self.args.inplace_buffers[mutation]).inner_name
                )
            if mutation in self.args.output_buffers:
                mutation_arg = self.args.output_buffers[mutation]
                assert not isinstance(mutation_arg, RemovedArg)
                mutated_args.add(mutation_arg)

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
                mutated_args.add(argname.name)

        # pyrefly: ignore [bad-assignment]
        mutated_args = sorted(mutated_args)

        for tree in self.active_range_trees():
            sizearg = SizeArg(f"{tree.prefix}numel", tree.numel)
            signature.append(sizearg)
            argdefs.append(ArgName(sizearg.name))
            # constexpr version causes issues, see
            # https://github.com/pytorch/torchdynamo/pull/1362
            # triton_meta["constants"][len(argdefs)] = V.graph.sizevars.size_hint(
            #     tree.numel
            # )
            # argdefs.append(f"{tree.prefix}numel: tl.constexpr")

        def add_constexpr_arg(arg_name):
            # new versions (but not old versions) of Triton need constexprs included in the signature
            if triton_version_uses_attrs_dict():
                signature.append(ConstexprArg(arg_name))
            argdefs.append(ArgName(arg_name, is_constexpr=True))

        for tree in self.range_trees:
            if tree.is_reduction and self.persistent_reduction:
                # Rn_BLOCK for persistent_reduction is defined in codegen_static_numels
                continue
            if tree.tensor_dim is None:
                continue

            add_constexpr_arg(f"{tree.prefix.upper()}BLOCK")

        if self.cooperative_reduction:
            add_constexpr_arg("RSPLIT")

        if self.mix_order_reduction:
            add_constexpr_arg("RSPLIT_SIZE")
            add_constexpr_arg("NUM_STAGES")

        triton_meta_signature = signature_to_meta(
            signature, size_dtype=self.index_dtype, argdefs=argdefs
        )
        triton_meta: dict[str, Any] = {
            "signature": triton_meta_signature,
            "device": DeviceProperties.create(V.graph.get_current_device_or_throw()),
            "constants": {},
            "native_matmul": (
                torch._inductor.config.triton.native_matmul
                and ("tl.dot" in str(self.body) or "tl.dot" in str(self.compute))
            ),
        }

        # Skip memory optimization for forward of the training loop where we expect
        # every new node will increase the peak memory and our greedy approach would
        # introduce a lot of unnecessary cpu copies.
        optimize_mem = V.graph.is_inference or V.graph.is_backward

        inductor_meta = {
            "grid_type": self._get_grid_type().__name__,
            # Triton will not accept an OrderedSet for autotune_hints
            "autotune_hints": set(self.autotune_hints),  # noqa: set_linter
            "kernel_name": str(Placeholder.DESCRIPTIVE_NAME),
            "mutated_arg_names": mutated_args,
            "optimize_mem": optimize_mem,
            "no_x_dim": self.no_x_dim,
            "atomic_add_found": self.atomic_add_found,
            "num_load": self.num_load,
            "num_store": self.num_store,
            "num_reduction": self.num_reduction,
            **self.inductor_meta_common(),
        }

        if self.mix_order_reduction:
            inductor_meta["RSPLIT_SIZE"] = self.rsplit_size

        if config.deterministic or config.test_configs.force_filter_reduction_configs:
            inductor_meta["has_loadstore_with_contiguous_rdim"] = (
                self.has_load_with_contiguous_rdim
                or self.has_store_with_contiguous_rdim
            )

        # Bail on 3d tiling, which has more complicated coalesce patterns
        looped_red = V.kernel.features.is_reduction() and not self.persistent_reduction
        tiling_scores = self.tiling_scores
        two_d_red = len(self.tiling) == 2
        if looped_red and two_d_red:
            memory_stats = self.features.memory_stats(self.tiling)
            dim_stats = memory_stats.persistent.memory.dim[0]
            mem_ops_per_thread = dim_stats.count_per_thread

            if (
                tiling_scores is not None
                and "x" in tiling_scores
                and "r0_" in tiling_scores
            ):
                # large rblock inhibits xblock size, dont attempt if there is a decent amount of
                # reads coalesced by xblock
                r_coalesce_ratio = tiling_scores["r0_"] / max(tiling_scores["x"], 1)
                contiguous_red = r_coalesce_ratio >= INNER_REDUCTION_RATIO_THRESHOLD
            else:
                contiguous_red = (
                    self.features.get_reduction_hint(tiling_scores)
                    == ReductionHint.INNER
                )

            looped_mem = memory_stats.looped.memory.bytes
            persistent_mem = memory_stats.persistent.memory.bytes
            # check that we save significant memory by doing persistent
            saved_bytes_ratio = V.graph.sizevars.size_hint(
                looped_mem, fallback=config.unbacked_symint_fallback
            ) / max(
                V.graph.sizevars.size_hint(
                    persistent_mem, fallback=config.unbacked_symint_fallback
                ),
                1,
            )

            # TODO - rnumel should be reasonably close to power of 2
            if (
                # significant memory bandwidth savings
                saved_bytes_ratio >= 1.3
                and contiguous_red
                # TODO - need more detailed register analysis
                and V.graph.sizevars.statically_known_leq(
                    self.features.reduction_numel, 32768
                )
                # We will already generate a persistent config in this case
                and V.graph.sizevars.statically_known_gt(
                    self.features.reduction_numel, 2048
                )
                and mem_ops_per_thread <= 10
            ):
                inductor_meta["add_persistent_rblock"] = True

        if self.tiling_scores:
            inductor_meta["tiling_scores"] = self.tiling_scores

        if self.tma_min_block_sizes:
            inductor_meta["tma_min_block_sizes"] = self.tma_min_block_sizes

        if self.cooperative_reduction:
            inductor_meta["persistent_reduction"] = self.persistent_reduction

        num_gb = None
        if config.benchmark_kernel or config.profile_bandwidth:
            num_gb = self.estimate_kernel_num_bytes() / 1e9
            if num_gb is not None:
                inductor_meta["kernel_num_gb"] = num_gb
        if config.benchmark_kernel:
            flops = self.estimate_flops()
            if flops is not None:
                inductor_meta["kernel_flop"] = flops

        triton_meta["configs"] = [config_of(signature)]

        triton_meta["launch_pdl"] = self._enable_pdl_codegen()

        # Triton compiler includes equal_to_1 args into constants even
        # when they are not constexpr. otherwise there may be a segfault
        # during launching the Inductor-compiled Triton kernel.
        # https://github.com/pytorch/pytorch/issues/120478#issuecomment-1962822307
        # https://github.com/triton-lang/triton/blob/231efe9ed2d200be0f69a07c298e4342b08efe3d/python/triton/runtime/jit.py#L384
        for arg_num in equal_1_arg_indices(signature):  # type: ignore[index]
            triton_meta["constants"][signature[arg_num].name] = 1  # type: ignore[index,union-attr]
        triton_meta["enable_fp_fusion"] = not config.emulate_precision_casts

        self.triton_meta = triton_meta

        self.codegen_prologue(self.body)
        self.codegen_body()
        self._filter_pdl(self.body)

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
            reduction_hint = self.features.get_reduction_hint(self.tiling_scores)
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
                if (
                    len(non_constexpr_signature(signature)) == 4
                ):  # input, output and 2 args
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
        kernel_name = name or str(Placeholder.KERNEL_NAME)
        code.writeline(
            f"def {kernel_name}({', '.join(x.full_name() for x in argdefs)}):"
        )
        with code.indent():
            if config.triton.proton_profiling:
                code.writeline(f'pl.enter_scope("{kernel_name}")')
            self.codegen_static_numels(code)
            for old, new in self.args.aliases():
                code.writeline(f"{old} = {new}")
            code.splice(self.body)
            if config.triton.proton_profiling:
                code.writeline(f'pl.exit_scope("{kernel_name}")')

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
            val = 2
            while not V.graph.sizevars.statically_known_leq(rnumel, val):
                if val > 16 * 1024:
                    raise ValueError(f"Failed to find static RBLOCK for {rnumel}")
                val *= 2

            return val

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
        def KERNEL_NAME(in_ptr0, in_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):

        We would add
        xnumel = 4096
        r0_numel = 768

        After the signature, before the kernel code, if we decided to make these static. As its hardcoded, it becomes
        a better signal to triton on how to unroll and do some static indexing. So, it's not so much that downstream
        knows that its a static numel, as that you just plop a constant into the kernel.
        """

        def is_static_integer(expr: sympy.Expr) -> bool:
            return isinstance(expr, (sympy.Integer, int))

        for tree in self.range_trees:
            if not tree.is_reduction or self.inside_reduction:
                simplified_tree_numel = V.graph.sizevars.simplify(tree.numel)
                if is_static_integer(simplified_tree_numel):
                    code.writeline(f"{tree.prefix}numel = {int(simplified_tree_numel)}")

            if tree.is_reduction and self.persistent_reduction:
                if self.cooperative_reduction:
                    numel = self.kexpr(self.rename_indexing(tree.numel))
                    val = f"triton_helpers.constexpr_next_power_of_2(({numel} + RSPLIT - 1) // RSPLIT)"
                else:
                    val = self._get_persistent_RBLOCK(tree.numel)
                    if self.is_native_matmul:
                        # tl.dot only supports shapes >= 16
                        val = max(val, 16)

                code.writeline(f"{tree.prefix.upper()}BLOCK: tl.constexpr = {val}")

            if tree.prefix == "x" and self.no_x_dim:
                code.writeline("XBLOCK: tl.constexpr = 1")

    def _get_grid_type(self) -> type[triton_heuristics.GridExpr]:
        n = sum([int(not tree.is_reduction) for tree in self.range_trees])
        if self.mix_order_reduction:
            assert n == 1
            return triton_heuristics.MixOrderReductionGrid
        elif self.cooperative_reduction:
            assert n == 1
            return triton_heuristics.CooperativeReductionGrid
        elif n == 1:
            return triton_heuristics.Grid1D
        elif n == 2:
            if any(map(self.needs_yz_grid_overflow, self.range_trees)):
                return triton_heuristics.Grid2DWithYZOverflow
            return triton_heuristics.Grid2D
        elif n == 3:
            return triton_heuristics.Grid3D
        raise ValueError(f"Unsupported number of dimensions: {n}")

    def add_numel_to_call_args(self, name, call_args, arg_types):
        # TODO(jansel): if there are constants, we shouldn't bother passing them as args
        for tree in self.range_trees:
            if isinstance(tree.numel, (sympy.Integer, sympy.Symbol)):
                expr = tree.numel
            else:
                expr = V.graph.wrapper_code.generate_numel_expr(name, tree)

            if not tree.is_reduction or self.inside_reduction:
                call_args.append(expr)
                arg_types.append(type(expr))

    def call_kernel(
        self, name: str, node: Optional[IRNode] = None, deallocate_ws: bool = True
    ):
        wrapper = V.graph.wrapper_code
        wrapper.write_triton_header_once()
        _, call_args, _, arg_types = self.args.python_argdefs()
        self.add_numel_to_call_args(name, call_args, arg_types)

        for ws in self.args.workspace_args:
            wrapper.generate_workspace_allocation(ws)

        wrapper.generate_kernel_call(
            name,
            call_args,
            triton=True,
            arg_types=arg_types,
            triton_meta=self.triton_meta,
        )

        if deallocate_ws:
            self.deallocate_workspaces()

    def codegen_nan_check(self) -> None:
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

    def create_cse_var(self, *args, **kwargs) -> TritonCSEVariable:
        return TritonCSEVariable(*args, **kwargs)

    def codegen_iteration_ranges_entry(self, entry: IterationRangesEntry):
        line = f"{entry.name} = {self.kexpr(self.rename_indexing(entry.expr))}"

        # mix order reduction introduces an extra loop across the x
        # dimension
        if entry.root.is_loop or (self.mix_order_reduction and entry.prefix == "x"):
            self.indexing_code.writeline(line)
        else:
            # lift non-reduction stores outside loop
            self.body.writeline(line)

    def iteration_ranges_ranges_code(self, entry: IterationRangesRoot) -> str:
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

    def iteration_ranges_scalar_code(
        self, entry: IterationRangesRoot, value: Any
    ) -> str:
        index_dtype = self.index_dtype
        ndim = self.triton_tensor_ndim()
        size = [1] * ndim
        return f"tl.full({size}, {value}, {index_dtype})"

    def iteration_ranges_get_pid(self, entry: IterationRangesRoot) -> str:
        assert entry.grid_dim is not None
        key = f"tl.program_id({entry.grid_dim})"
        # y_grid has a limit, so express it in terms of y and z in case of overflow.
        # z grid is only exercised when max_tiles == 3 (off by default).
        if self.needs_yz_grid_overflow(entry):
            # For ynumel larger than max_ygrid, we need to use zdim.
            # For each z dimension, there are tl.num_programs(1) yblocks which is passed by grad(x,y,z).
            # So, we need to add tl.program_id(z) * tl.num_programs(y) *YBLOCK to get the correct yoffset.
            key = f"({key} + tl.program_id({entry.grid_dim + 1}) * tl.num_programs({entry.grid_dim}))"
        pid = entry.pid_cache.get(key, key)
        if self.index_dtype != "tl.int32":
            return f"{pid}.to({self.index_dtype})"
        return pid

    def needs_yz_grid_overflow(self, entry: IterationRangesRoot) -> bool:
        return (
            entry.grid_dim == 1
            and not entry.has_zdim
            and not self.cooperative_reduction
            and not V.graph.sizevars.statically_known_leq(entry.numel, get_max_y_grid())
        )

    def max_block(self, prefix: str) -> int:
        if self.fixed_config:
            return self.fixed_config[f"{prefix.upper()}BLOCK"]
        return TRITON_MAX_BLOCK[prefix.upper()]

    def _has_constant_mask(self, tree: IterationRangesRoot) -> bool:
        if self.is_native_matmul:
            # tl.dot requires the shape to be >= 16,
            # so when matmul shape is smaller than 16, we always keep the mask.
            if V.graph.sizevars.statically_known_lt(tree.numel, 16):
                return False

        if not self.optimize_mask:
            return False

        if self.fixed_config and f"{tree.prefix.upper()}BLOCK" in self.fixed_config:
            if self.fixed_config[f"{tree.prefix.upper()}BLOCK"] == 1:
                return True
        elif not self.is_combo_kernel:
            if V.graph.sizevars.statically_known_equals(tree.numel, 1):
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

        # [Note: Constant mask optimisation]
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

    def _has_constant_xmask(self) -> bool:
        xtree = self.range_trees[0]
        assert xtree.prefix == "x"
        return self._has_constant_mask(xtree)

    def filter_masks(self, mask_vars: OrderedSet[str]) -> None:
        for tree in self.range_trees:
            if self._has_constant_mask(tree):
                mask_vars.discard(f"{tree.prefix}mask")

        # can be added as an override_mask
        mask_vars.discard("None")

    @cache_on_self
    def get_reduction_prefixes(self) -> list[str]:
        return [
            prefix_str[symt]
            for symt in list(TritonSymbols.reduction_types)[: self.num_reduction_dims]
        ]

    def codegen_reduction_numels(self, buffer: IndentedBuffer) -> None:
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

    def _get_reduction_symbols(self, suffix: str, **kwargs) -> list[sympy.Symbol]:
        """
        Helper to initialize symbols like rn_numel, rn_base, etc.
        """
        rn_prefixes = self.get_reduction_prefixes()
        return [sympy.Symbol(f"{prefix}{suffix}", **kwargs) for prefix in rn_prefixes]

    @cache_on_self
    def _get_reduction_index_coeffs(self) -> list[sympy.Expr]:
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

    def _flatten_reduction_indices(self, multi_inds: list[sympy.Expr]) -> sympy.Expr:
        """
        Compute linear reduction indices from N dimensional ones.
        """
        coeffs = self._get_reduction_index_coeffs()
        return sympy_dot(coeffs, multi_inds)

    def codegen_reduction_indices(self, buffer: IndentedBuffer) -> None:
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

    def iteration_ranges_codegen_header(
        self, entry: IterationRangesRoot, code: IndentedBuffer
    ) -> None:
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

            block_size = (
                f"{x.upper()}BLOCK" if not self.mix_order_reduction else "RSPLIT_SIZE"
            )
            code.writelines(
                [
                    f"{x}offset = {self.iteration_ranges_get_pid(entry)} * {block_size}",
                    f"{entry.name} = {line}",
                ]
            )
        if self._has_constant_mask(entry):
            code.writeline(self.create_constant_mask(entry))
        elif not (x == "x" and self.mix_order_reduction):
            # mix order reduction should generate xmask inside the loop
            code.writeline(f"{x}mask = {entry.name} < {x}numel")


class TritonScheduling(SIMDScheduling):
    kernel_type: type[Any] = TritonKernel
    backend_features = OrderedSet(
        [
            BackendFeature.FOREACH,
            BackendFeature.BUCKETIZE,
            BackendFeature.INPLACE_BUFFERS,
            BackendFeature.MASKED_SCATTER_WITH_INDEX,
            BackendFeature.SCAN,
            BackendFeature.SORT,
            BackendFeature.TRITON_TEMPLATES,
            BackendFeature.TUPLE_REDUCTION,
        ]
    )

    def __init__(self, scheduler: Optional[Scheduler]) -> None:
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
            return OrderedSet(
                [*cls.backend_features, BackendFeature.REDUCE_TO_SINGLE_ELEMENT]
            )
        return cls.backend_features

    def codegen_comment(self, node_schedule, kernel_name=None):
        wrapper = V.graph.wrapper_code
        origins, _detailed_origins = get_kernel_metadata(node_schedule, wrapper)
        if origins:
            wrapper.make_comment(origins)

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
                wrapper.make_comment(
                    f"{wrapper.comment} Fused node name list: {', '.join(node_names)}"
                )

        if kernel_name:
            debug_handle = set_kernel_post_grad_provenance_tracing(
                node_schedule,  # type: ignore[arg-type]
                kernel_name,
            )
            wrapper.write_provenance_debug_handle(kernel_name, debug_handle)

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
            if config.aot_inductor.model_name_for_generated_files:
                # When AOTI compiles multiple submodules, we need to use the model name to
                # distinguish kernel related symbols.
                kernel_name = f"{config.aot_inductor.model_name_for_generated_files}_{kernel_name}"

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

            if async_compile.use_process_pool():
                # The process pool is warm, we can shell out to workers right away. This
                # allows us to save the result in async_compile.CompiledTritonKernels,
                # so that the second time we call async_compile.triton, we do no work.
                async_compile.triton(subs_name, src_code)

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

    def benchmark_fused_nodes(self, nodes, n_spills_threshold=8) -> tuple[float, str]:
        """
        Benchmark fused list of nodes and return the execution time
        in milliseconds on randomly generated inputs.
        """
        src_code = self.generate_kernel_code_from_nodes(nodes, benchmark_kernel=True)
        mod = PyCodeCache.load(src_code)
        return self.benchmark_codegened_module(
            mod, n_spills_threshold, node_names=OrderedSet(n.get_name() for n in nodes)
        )

    def benchmark_codegened_module(
        self, mod, n_spills_threshold=8, node_names: Optional[OrderedSet[str]] = None
    ) -> tuple[float, str]:
        """Benchmark an already compiled module"""
        device_interface = get_interface_for_device(V.graph.device_type)
        with (
            preserve_rng_state(),
            device_interface.device(V.graph.get_current_device_or_throw()),  # type: ignore[attr-defined]
        ):
            ms = None

            def cache_file_path():
                assert mod.__file__ is not None
                return os.path.splitext(mod.__file__)[0] + ".kernel_perf"

            def store_cache():
                path = cache_file_path()
                write_atomic(path, str(ms))

            def load_cache():
                path = cache_file_path()
                if os.path.exists(path):
                    with open(path) as fd:
                        return float(fd.read())
                return None

            node_names = (
                node_names if node_names is not None else OrderedSet(["unknown"])
            )
            log.debug(
                "kernel src code for %s written to: %s",
                node_names,
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
                if config.triton.disallow_failing_autotune_kernels_TESTING_ONLY:
                    raise
                log.debug(  # noqa: G200
                    "Exception (%s) in compiling fused nodes %s",
                    e,
                    node_names,
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
                device = V.graph.get_current_device_or_throw()
                # We have to clone the inplace updated arguments to avoid earlier calls
                # generating out of range indices for later calls.
                ms = benchmarker.benchmark(
                    lambda: call(wrapped_jit_function.clone_args(*args)[0]),
                    device=device,
                )
                # overhead of cloning args gives bias for fusing the kernel
                # in the case of mutating/in-placeable second fusion
                # TODO - would be better as a hook in triton do_bench that reset
                # the input values between benchmarking
                if len(wrapped_jit_function.mutated_arg_names) > 0:
                    ms = ms - benchmarker.benchmark(
                        lambda: wrapped_jit_function.clone_args(*args),
                        device=str(device),
                    )

            log.debug(
                "The fused kernel for %s took %.3f ms to run",
                node_names,
                ms,
            )
            store_cache()
            return ms, mod.__file__

    def create_kernel_choices(  # type: ignore[override]
        self,
        kernel_features: SIMDKernelFeatures,
        kernel_args: list[Any],
        kernel_kwargs: dict[str, Any],
    ) -> list[TritonKernel]:
        is_scan = kernel_features.contains_op("scan")
        is_split_scan = is_scan and any(
            node.is_split_scan() for node in kernel_features.scheduler_nodes()
        )
        kernel_type: type[TritonKernel] = self.kernel_type
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
        kernel: TritonKernel,
        kernel_args: list[Any],
        kernel_kwargs: dict[str, Any],
    ) -> list[TritonKernel]:
        kernels: list[TritonKernel] = [kernel]
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
        ms: float
        ms_clone: float

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
            write_atomic(path, str(ms) + " " + str(ms_clone))

        total_ms, file_list = 0, []
        total_clone_ms: float = 0.0
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

        # pyrefly: ignore [bad-assignment]
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
                total_ms += ms  # type: ignore[assignment]
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
                device = V.graph.get_current_device_or_throw()
                # We have to clone the inplace updated arguments to avoid earlier calls
                # generating out of range indices for later calls.
                ms = benchmarker.benchmark(
                    lambda: call(wrapped_jit_function.clone_args(*args)[0]),
                    device=device,
                )
                ms_clone = benchmarker.benchmark(
                    lambda: wrapped_jit_function.clone_args(*args)[0],
                    device=device,
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


def debug_triton_code(node: BaseSchedulerNode) -> list[str]:
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
        assert isinstance(backend, (SIMDScheduling, CUDACombinedScheduling)), (
            f"Scheduling backend should be SIMD or CUDACombined when generating debug Triton strings, got: {type(backend)}"
        )

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
