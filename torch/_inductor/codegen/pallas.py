"""
Pallas Backend - Minimal 2D Canonical Form

This is a minimal Pallas codegen that uses a 2D canonical form:
- All data flows through (numel, rnumel) shape
- Load: buffer → reshape to (numel, rnumel)
- Compute: all ops in 2D
- Store: (numel, rnumel) → reshape to output buffer shape

NO GUESSING: This version does NOT try to detect permute/expand from
index expressions or buffer strides. It needs explicit upstream info.
"""
from __future__ import annotations

import hashlib
import math
from typing import Any, Callable, Optional, TYPE_CHECKING, Union

import sympy

import torch
from torch.utils._ordered_set import OrderedSet

from .. import config
from ..ir import PALLAS_EXPAND_STRIDE, PallasStride, PallasViewIdMarker
from ..utils import get_fused_kernel_name, get_kernel_metadata
from ..virtualized import V
from .common import (
    BackendFeature,
    CSEVariable,
    IndentedBuffer,
    OpOverrides,
    PythonPrinter,
)
from .simd import SIMDKernel, SIMDScheduling


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from ..ir import IRNode
    from ..ops_handler import ReductionType
    from ..scheduler import BaseSchedulerNode


kernel_code_log = torch._logging.getArtifactLogger(__name__, "kernel_code")


# Centralized dtype mapping for Pallas/JAX - used by constant(), to_dtype(), to_dtype_bitcast()
_PALLAS_DTYPE_MAP = {
    torch.float16: "jnp.float16", torch.bfloat16: "jnp.bfloat16",
    torch.float32: "jnp.float32", torch.float64: "jnp.float64",
    torch.int8: "jnp.int8", torch.int16: "jnp.int16",
    torch.int32: "jnp.int32", torch.int64: "jnp.int64",
    torch.uint8: "jnp.uint8", torch.bool: "jnp.bool_",
}


class PallasPrinter(PythonPrinter):
    """Sympy printer for Pallas/JAX expressions."""

    def _print_Abs(self, expr: sympy.Expr) -> str:
        if len(expr.args) != 1:
            raise AssertionError("Abs expects exactly one argument")
        return f"jnp.abs({self._print(expr.args[0])})"

    def _print_Min(self, expr: sympy.Expr) -> str:
        args = [self.doprint(arg) for arg in expr.args]
        result = args[0]
        for arg in args[1:]:
            result = f"jnp.minimum({result}, {arg})"
        return result

    def _print_Max(self, expr: sympy.Expr) -> str:
        args = [self.doprint(arg) for arg in expr.args]
        result = args[0]
        for arg in args[1:]:
            result = f"jnp.maximum({result}, {arg})"
        return result


pallas_pexpr = PallasPrinter().doprint


class PallasKernelOverrides(OpOverrides):
    """JAX/Pallas operation implementations for 2D tensors."""

    @staticmethod
    def constant(value: Any, dtype: torch.dtype) -> str:
        jax_dtype = _PALLAS_DTYPE_MAP.get(dtype, "jnp.float32")
        if dtype == torch.bool:
            return "True" if value else "False"
        if isinstance(value, float):
            import math
            if math.isnan(value):
                return f"jnp.array(jnp.nan, dtype={jax_dtype})"
            elif math.isinf(value):
                inf_val = "jnp.inf" if value > 0 else "-jnp.inf"
                return f"jnp.array({inf_val}, dtype={jax_dtype})"
        # Wrap all constants in jnp.array so they can be reshaped
        return f"jnp.array({repr(value)}, dtype={jax_dtype})"

    @staticmethod
    def index_expr(expr: sympy.Expr, dtype: torch.dtype) -> str:
        """Convert a sympy expression to a JAX expression.

        Uses rename_indexing to register symbolic sizes as kernel parameters.
        """
        from ..virtualized import V

        # Prepare and rename indexing to register size symbols as kernel args
        prepared = V.kernel.prepare_indexing(expr)
        renamed = V.kernel.rename_indexing(prepared)
        return V.kernel.kexpr(renamed)

    @staticmethod
    def to_dtype(x: str, dtype: torch.dtype, src_dtype: Optional[torch.dtype] = None) -> str:
        return f"({x}).astype({_PALLAS_DTYPE_MAP.get(dtype, 'jnp.float32')})"

    @staticmethod
    def to_dtype_bitcast(x: str, dtype: torch.dtype, src_dtype: torch.dtype) -> str:
        """Bitcast a value from one dtype to another with the same size.

        This reinterprets the raw bytes without conversion, used for tensor.view(dtype).
        """
        jax_dtype = _PALLAS_DTYPE_MAP.get(dtype, "jnp.float32")
        jax_src_dtype = _PALLAS_DTYPE_MAP.get(src_dtype, "jnp.float32")
        # First ensure the value is the correct source dtype (in case of internal promotion),
        # then bitcast to target dtype
        return f"jax.lax.bitcast_convert_type(jnp.asarray({x}).astype({jax_src_dtype}), {jax_dtype})"

    # Unary ops
    @staticmethod
    def abs(x: str) -> str: return f"jnp.abs({x})"
    @staticmethod
    def exp(x: str) -> str: return f"jnp.exp({x})"
    @staticmethod
    def log(x: str) -> str: return f"jnp.log({x})"
    @staticmethod
    def sqrt(x: str) -> str: return f"jnp.sqrt({x})"
    @staticmethod
    def rsqrt(x: str) -> str: return f"jax.lax.rsqrt({x})"
    @staticmethod
    def sin(x: str) -> str: return f"jnp.sin({x})"
    @staticmethod
    def cos(x: str) -> str: return f"jnp.cos({x})"
    @staticmethod
    def tanh(x: str) -> str: return f"jnp.tanh({x})"
    @staticmethod
    def sigmoid(x: str) -> str: return f"jax.nn.sigmoid({x})"
    @staticmethod
    def relu(x: str) -> str: return f"jax.nn.relu({x})"
    @staticmethod
    def neg(x: str) -> str: return f"-({x})"
    @staticmethod
    def floor(x: str) -> str: return f"jnp.floor({x})"
    @staticmethod
    def ceil(x: str) -> str: return f"jnp.ceil({x})"
    @staticmethod
    def round(x: str) -> str: return f"jnp.round({x})"
    @staticmethod
    def signbit(x: str) -> str: return f"jnp.signbit({x})"
    @staticmethod
    def trunc(x: str) -> str: return f"jnp.trunc({x})"
    @staticmethod
    def log1p(x: str) -> str: return f"jnp.log1p({x})"
    @staticmethod
    def expm1(x: str) -> str: return f"jnp.expm1({x})"

    # Binary ops
    @staticmethod
    def add(a: str, b: str) -> str: return f"({a}) + ({b})"
    @staticmethod
    def sub(a: str, b: str) -> str: return f"({a}) - ({b})"
    @staticmethod
    def mul(a: str, b: str) -> str: return f"({a}) * ({b})"
    @staticmethod
    def truediv(a: str, b: str) -> str: return f"({a}) / ({b})"
    @staticmethod
    def floordiv(a: str, b: str) -> str: return f"({a}) // ({b})"
    @staticmethod
    def mod(a: str, b: str) -> str: return f"({a}) % ({b})"
    @staticmethod
    def pow(a: str, b: str) -> str: return f"jnp.power({a}, {b})"
    @staticmethod
    def maximum(a: str, b: str) -> str: return f"jnp.maximum({a}, {b})"
    @staticmethod
    def minimum(a: str, b: str) -> str: return f"jnp.minimum({a}, {b})"
    @staticmethod
    def where(cond: str, a: str, b: str) -> str: return f"jnp.where({cond}, {a}, {b})"

    # Comparison ops
    @staticmethod
    def eq(a: str, b: str) -> str: return f"({a}) == ({b})"
    @staticmethod
    def ne(a: str, b: str) -> str: return f"({a}) != ({b})"
    @staticmethod
    def lt(a: str, b: str) -> str: return f"({a}) < ({b})"
    @staticmethod
    def le(a: str, b: str) -> str: return f"({a}) <= ({b})"
    @staticmethod
    def gt(a: str, b: str) -> str: return f"({a}) > ({b})"
    @staticmethod
    def ge(a: str, b: str) -> str: return f"({a}) >= ({b})"

    # Bitwise ops
    @staticmethod
    def bitwise_and(a: str, b: str) -> str: return f"({a}) & ({b})"
    @staticmethod
    def bitwise_or(a: str, b: str) -> str: return f"({a}) | ({b})"
    @staticmethod
    def bitwise_xor(a: str, b: str) -> str: return f"({a}) ^ ({b})"
    @staticmethod
    def bitwise_not(x: str) -> str: return f"~({x})"
    @staticmethod
    def logical_and(a: str, b: str) -> str: return f"({a}) & ({b})"
    @staticmethod
    def logical_or(a: str, b: str) -> str: return f"({a}) | ({b})"
    @staticmethod
    def logical_not(x: str) -> str: return f"~({x})"

    @staticmethod
    def masked(mask: str, body: Callable[[], str], other: float) -> str:
        """
        Computes body, but only uses the result where mask is true.
        Where mask is false, uses the 'other' value instead.
        """
        result = body()
        # Format the 'other' value properly for JAX
        if isinstance(other, float):
            if math.isnan(other):
                other_str = "jnp.nan"
            elif math.isinf(other):
                other_str = "jnp.inf" if other > 0 else "-jnp.inf"
            else:
                other_str = repr(other)
        else:
            other_str = repr(other)
        return f"jnp.where({mask}, {result}, {other_str})"


class PallasKernel(SIMDKernel):
    """
    Minimal Pallas kernel using 2D canonical form.

    NO GUESSING - just basic reshape to/from 2D.
    """

    overrides = PallasKernelOverrides  # type: ignore[assignment]
    kexpr: Callable[[sympy.Expr], str] = pallas_pexpr  # Use Pallas expression printer

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stores: list[tuple[str, str]] = []
        self._original_ranges: list[list[int]] = []  # Saved from split_and_set_ranges
        # Track shapes of CSE variables for indirect indexing and scalar detection
        # Shape () indicates a scalar constant that needs broadcast_to instead of reshape
        self.var_shapes: dict[str, Optional[tuple[int, ...]]] = {}
        # Track outputs that have scatter stores (need input_output_aliases)
        self.scatter_outputs: set[str] = set()
        # EXPLICIT iteration order tracking: maps (prefix, dim_index) -> flat_divisor
        # For C-order iteration, innermost dim has divisor=1, outer dims have product of inner dims
        # This is set in split_and_set_ranges and used in codegen_kernel
        self._iteration_divisors: dict[tuple[str, int], int] = {}
        # Track index source variables that need broadcasting in scatter operations
        # Maps tmp_var_name -> {"iter_dims": [dim_indices], "shape": [sizes]}
        self._index_source_vars: dict[str, dict] = {}
        # Track dim_index -> var_name mapping for generating index expressions
        self._dim_to_var: dict[int, str] = {}
        # Track reduction dim_index -> var_name mapping (for r vars)
        self._rdim_to_var: dict[int, str] = {}

    def dtype_to_str(self, dtype: torch.dtype) -> str:
        """Convert PyTorch dtype to JAX/Pallas dtype string."""
        if dtype in _PALLAS_DTYPE_MAP:
            return _PALLAS_DTYPE_MAP[dtype]
        raise NotImplementedError(f"Unsupported dtype for Pallas: {dtype}")

    def _safe_int(self, val):
        """Convert value to int, returning None for symbolic values.

        For dynamic shapes, returns None instead of raising.
        Callers should check for None and handle symbolic cases appropriately
        (e.g., using rename_indexing to pass symbols as kernel parameters).
        """
        try:
            if hasattr(val, 'is_number') and not val.is_number:
                return None
            return int(val)
        except (TypeError, ValueError):
            return None

    def split_and_set_ranges(self, lengths):
        """Override to save original ranges and compute iteration divisors."""
        # Save original ranges for Pallas expand/permute detection
        # Keep symbolic values as-is (None from _safe_int) for dynamic shape support
        self._original_ranges = [[self._safe_int(s) for s in group] for group in lengths]
        # Also keep raw symbolic expressions for rename_indexing
        self._original_ranges_raw = [[s for s in group] for group in lengths]

        # Call parent to get the var->dim mapping from set_ranges result
        result = super().split_and_set_ranges(lengths)

        # EXPLICIT iteration order tracking: compute flat divisors for C-order iteration
        # The result tells us which var corresponds to which dimension:
        # result = [[x1, x0], [r0]] means x1->dim0, x0->dim1, r0->rdim0
        # We compute divisors and store them keyed by var name for lookup in codegen
        prefixes = ["x", "r"] if len(lengths) > 1 else ["x"]
        # Use raw expressions (not _original_ranges which has None for symbolic dims)
        for prefix, dims_raw, vars_for_dims in zip(prefixes, self._original_ranges_raw, result):
            if not dims_raw:
                continue
            # C-order: rightmost dimension is innermost (divisor=1)
            # Compute divisors right-to-left and map to var names
            # Use sympy.Integer(1) to support symbolic dimensions
            divisor = sympy.Integer(1)
            for dim_idx in range(len(dims_raw) - 1, -1, -1):
                if dim_idx < len(vars_for_dims):
                    var = vars_for_dims[dim_idx]
                    var_name = str(var)
                    self._iteration_divisors[var_name] = divisor
                    # Track dim_index -> var_name for index_source broadcasting
                    if prefix == "x":
                        self._dim_to_var[dim_idx] = var_name
                    elif prefix == "r":
                        self._rdim_to_var[dim_idx] = var_name
                # Always multiply - dims_raw has symbolic expressions
                dim_val = dims_raw[dim_idx]
                if dim_val is not None:
                    divisor = divisor * dim_val

        return result

    def _get_numel_expr(self):
        """Product of non-reduction dimensions as sympy expression.

        Returns a sympy expression (possibly symbolic) that can be passed
        to kexpr() for code generation.
        """
        numel = sympy.Integer(1)
        for tree in self.range_trees:
            if not tree.is_reduction:
                numel = numel * tree.numel
        return V.graph.sizevars.simplify(numel)

    def _get_rnumel_expr(self):
        """Product of reduction dimensions as sympy expression.

        Returns a sympy expression (possibly symbolic) that can be passed
        to kexpr() for code generation.
        """
        rnumel = sympy.Integer(1)
        for tree in self.range_trees:
            if tree.is_reduction:
                rnumel = rnumel * tree.numel
        return V.graph.sizevars.simplify(rnumel)

    def _get_numel(self):
        """Product of non-reduction dimensions. Returns None if any dimension is symbolic."""
        expr = self._get_numel_expr()
        val = self._safe_int(expr)
        return max(val, 1) if val is not None else None

    def _get_rnumel(self):
        """Product of reduction dimensions. Returns None if any dimension is symbolic."""
        expr = self._get_rnumel_expr()
        val = self._safe_int(expr)
        return max(val, 1) if val is not None else None

    def _get_numel_rnumel_info(self) -> dict:
        """Get cached numel/rnumel information for codegen.

        Returns dict with keys:
            numel_expr: sympy expression for non-reduction dims product
            rnumel_expr: sympy expression for reduction dims product
            numel_val: concrete int or None if symbolic
            rnumel_val: concrete int or None if symbolic
            numel_str: string for codegen (via rename_indexing + kexpr)
            rnumel_str: string for codegen (via rename_indexing + kexpr)
        """
        if hasattr(self, '_numel_rnumel_cache'):
            return self._numel_rnumel_cache

        numel_expr = self._get_numel_expr()
        rnumel_expr = self._get_rnumel_expr()

        numel_renamed = self.rename_indexing(numel_expr)
        rnumel_renamed = self.rename_indexing(rnumel_expr)

        self._numel_rnumel_cache = {
            'numel_expr': numel_expr,
            'rnumel_expr': rnumel_expr,
            'numel_val': self._safe_int(numel_expr),
            'rnumel_val': self._safe_int(rnumel_expr),
            'numel_str': self.kexpr(numel_renamed),
            'rnumel_str': self.kexpr(rnumel_renamed),
        }
        return self._numel_rnumel_cache

    def _get_shape_str_for_reshape(self, inside_reduction: Optional[bool] = None) -> str:
        """Get consistent shape string for reshape operations.

        Args:
            inside_reduction: Override self.inside_reduction if provided

        Returns:
            Shape string like "numel_str, rnumel_str" or "numel_str"
        """
        info = self._get_numel_rnumel_info()
        in_red = inside_reduction if inside_reduction is not None else self.inside_reduction
        rnumel_val = info['rnumel_val']

        if rnumel_val is None or rnumel_val > 1:
            if in_red:
                return f"{info['numel_str']}, {info['rnumel_str']}"
            else:
                return f"{info['numel_str']}, 1"
        return info['numel_str']

    def _get_buffer_shape(self, name: str):
        """Get buffer shape. May contain None for symbolic dimensions."""
        buf = V.graph.get_buffer(name)
        return [self._safe_int(s) for s in buf.get_size()] if buf else []

    def _get_contiguous_strides(self, shape: list) -> list[int]:
        """Compute contiguous (row-major) strides from shape.

        Since buffers are made contiguous via .contiguous() before being passed
        to Pallas kernels, we need to use contiguous strides when computing
        flat indices for jnp.take operations.

        For shape [s0, s1, ..., sn], contiguous strides are:
        - stride[n] = 1
        - stride[i] = stride[i+1] * shape[i+1] for i < n

        Args:
            shape: List of dimension sizes (may contain None for symbolic dims)

        Returns:
            List of contiguous strides, or empty list if shape has None values
        """
        if not shape or any(s is None for s in shape):
            return []

        strides = [1] * len(shape)
        for i in range(len(shape) - 2, -1, -1):
            strides[i] = strides[i + 1] * shape[i + 1]
        return strides

    def _resolve_index_with_contiguous_strides(
        self, index: sympy.Expr, name: str
    ) -> sympy.Expr:
        """Resolve PallasStride markers using contiguous strides.

        When buffers are made contiguous via .contiguous() before being passed
        to Pallas kernels, the flat index must use contiguous strides instead
        of the original buffer strides.

        Args:
            index: Index expression with PallasStride markers
            name: Buffer name to get shape from

        Returns:
            Resolved index expression with contiguous strides
        """
        from ..ir import PallasStride

        buf_shape = self._get_buffer_shape(name)
        contig_strides = self._get_contiguous_strides(buf_shape)

        resolved_index = index
        for ps in index.atoms(PallasStride):
            dim_idx = int(ps.dim_index)
            # Use contiguous stride if available, otherwise fall back to original
            if contig_strides and 0 <= dim_idx < len(contig_strides):
                stride_val = contig_strides[dim_idx]
            else:
                stride_val = ps.stride_value
            # Replace the PallasStride with the stride value
            if isinstance(stride_val, sympy.Basic) and not stride_val.is_number:
                resolved_index = resolved_index.subs(ps, stride_val)
            else:
                resolved_index = resolved_index.subs(ps, sympy.Integer(int(stride_val)))

        return resolved_index

    def _get_iter_shape(self):
        """Get iteration shape - uses original ranges if available. May contain None for symbolic dims."""
        if self._original_ranges:
            # Use saved original ranges (non-reduction dims are first group)
            return self._original_ranges[0] if self._original_ranges else []
        # Fallback to merged numel
        return [self._safe_int(tree.numel) for tree in self.range_trees if not tree.is_reduction]

    def _get_nd_shape_for_compute(self) -> tuple[list[int], list[int]]:
        """
        Get the N-dimensional shape used by the arange iteration vars.

        Returns (x_dims, r_dims) where:
        - x_dims: lengths of non-reduction dimensions (e.g., [8, 8])
        - r_dims: lengths of reduction dimensions (e.g., [16])

        The total compute shape is x_dims + r_dims (e.g., [8, 8, 16]).
        """
        if not self.range_tree_nodes:
            return [self._get_numel()], [self._get_rnumel()] if self._get_rnumel() > 1 else []

        # ROBUST: Get ordered vars from range_trees structure instead of sorting by divisor
        # Each range_tree has a var_list that preserves creation order (outer to inner)
        x_vars = []
        r_vars = []

        for tree in self.range_trees:
            # Get entries from this tree in var_list order (preserved from creation)
            for var in tree.var_list:
                if var in self.range_tree_nodes:
                    entry = self.range_tree_nodes[var]
                    if tree.prefix.startswith("r"):
                        r_vars.append((var, entry))
                    else:
                        x_vars.append((var, entry))

        # NO SORTING NEEDED - var_list already has correct outer-to-inner order

        x_dims = [int(entry.length) for _, entry in x_vars] if x_vars else [self._get_numel()]
        r_dims = [int(entry.length) for _, entry in r_vars] if r_vars else []

        return x_dims, r_dims

    def _get_compute_shape_str(self) -> str:
        """
        Get the reshape target string for compute operations.

        ROBUST: Uses consistent 1D/2D shape logic:
        - When rnumel=1: 1D shape (numel,) - no reduction dimension
        - When rnumel > 1: 2D shape (numel, rnumel) or (numel, 1) based on inside_reduction
        """
        numel = self._get_numel()
        rnumel = self._get_rnumel()

        if rnumel > 1:
            if self.inside_reduction:
                return f"{numel}, {rnumel}"
            else:
                return f"{numel}, 1"
        return str(numel)

    def _extract_view_id_from_index(self, index: sympy.Expr) -> Optional[int]:
        """
        Extract view_id from PallasStride or PallasViewIdMarker in the index expression.

        PallasStride(stride_value, dim_idx, view_id, iter_var_pos) embeds the view_id
        that links to the transform entry stored in V.graph._pallas_view_entries.

        PallasViewIdMarker(view_id) is used for all-size-1 tensors where no PallasStride
        would be created (the marker evaluates to 0 but carries view_id).

        Returns:
        - view_id > 0: Links to a transform entry in the registry
        - view_id == -1: Direct buffer access (no transform chain needed)
        - None: No PallasStride/PallasViewIdMarker markers found
        """
        view_ids = set()

        # Check PallasStride markers
        for atom in index.atoms(PallasStride):
            if atom.view_id is not None:
                view_ids.add(int(atom.view_id))

        # Check PallasViewIdMarker (for all-size-1 tensors)
        for atom in index.atoms(PallasViewIdMarker):
            if atom.view_id is not None:
                view_ids.add(int(atom.view_id))

        if not view_ids:
            return None

        # If any view_id > 0, use that (prefer chain-based access)
        positive_ids = [v for v in view_ids if v > 0]
        if positive_ids:
            return positive_ids[0]

        # All view_ids are -1 (direct buffer access)
        if -1 in view_ids:
            return -1

        return None

    def _get_transform_chain_from_view_id(
        self, view_id: int, buf_shape: Optional[list] = None
    ) -> Optional[list[dict]]:
        """
        Retrieve transform chain from the view_id registry.

        Uses Option A design: reconstructs chain dynamically, stopping at materialization
        boundaries. If buf_shape matches a parent entry's output_shape, that parent was
        materialized and we skip its transforms.

        Args:
            view_id: The view_id to look up
            buf_shape: The actual shape of the buffer being loaded (for materialization detection)

        Returns:
            List of transforms like:
            [{"op": "permute", "dims": [2, 0, 1]}, {"op": "expand", "expand_dims": [0], "target_shape": [...]}]
        """
        from torch._inductor.ir import PallasViewTracker

        # Use entry-based lookup with materialization detection
        return PallasViewTracker.get_effective_chain(view_id, buf_shape) or []

    def _get_reduction_dim_indices(self) -> Optional[tuple[int, ...]]:
        """
        Get reduction_dim_indices from the Reduction IR node.

        Access path: self.features.reduction_nodes() → node.node.data.reduction_dim_indices
        """
        reduction_nodes = list(self.features.reduction_nodes())
        if not reduction_nodes:
            return None
        # Use first reduction node (typically there's only one per kernel)
        node = reduction_nodes[0]
        return getattr(node.node.data, 'reduction_dim_indices', None)

    def _get_reduction_positions_from_ir(self) -> Optional[list[int]]:
        """
        Get which positions are being reduced from the Reduction IR node.
        """
        dim_indices = self._get_reduction_dim_indices()
        if dim_indices is None:
            return None
        return list(dim_indices)

    def _get_dim_order_from_index(
        self,
        index: sympy.Expr,
        buf_shape: list[int],
    ) -> Optional[list[int]]:
        """Extract logical dimension order using iter_var_pos from PallasStride.

        ROBUST: Uses iter_var_pos (buffer position) stored directly in PallasStride,
        providing unambiguous mapping even when multiple dims have the same stride
        (e.g., trailing size-1 dims).

        For dimensions explicitly mapped via iter_var_pos, use that mapping.
        For unmapped dimensions with size 1, they don't affect the index calculation
        (stride * index where index is always 0), so they can be assigned their
        natural position in the dimension order.

        Returns list of dim_index values ordered by buffer position, or None if
        not determinable.
        """
        pallas_strides = list(index.atoms(PallasStride))
        if not pallas_strides:
            return None

        # Filter out expand markers and build direct mapping: buf_position -> dim_index
        pos_to_dim: dict[int, int] = {}
        for ps in pallas_strides:
            stride_val = ps.stride_value
            if stride_val == PALLAS_EXPAND_STRIDE or stride_val is PALLAS_EXPAND_STRIDE:
                continue  # Skip expand markers - they're not part of original buffer

            # Use iter_var_pos for direct buffer position mapping
            iter_pos = ps.iter_var_pos
            if iter_pos is not None:
                buf_pos = int(iter_pos)
                dim_idx = int(ps.dim_index)
                pos_to_dim[buf_pos] = dim_idx

        # Handle unmapped positions - they must be size-1 dimensions
        # Size-1 dims don't contribute to index calculation (stride * 0 = 0)
        # so they don't get PallasStride markers, but we need to assign them
        # unique dim indices to form a valid permutation
        unmapped_positions = [i for i in range(len(buf_shape)) if i not in pos_to_dim]

        # Find which dim_indices are already used by mapped positions
        used_dims = set(pos_to_dim.values())
        # Collect available dim indices that aren't used (for valid permutation)
        available_dims = [d for d in range(len(buf_shape)) if d not in used_dims]

        for pos in unmapped_positions:
            if buf_shape[pos] != 1:
                # Non-size-1 dimension is unmapped - cannot determine order
                return None
            if not available_dims:
                # No available dim indices left - shouldn't happen for valid buffers
                return None
            # Assign first available dim_index to this size-1 position
            # This ensures dim_order forms a valid permutation [0..n-1]
            pos_to_dim[pos] = available_dims.pop(0)

        # Build dim_order from buffer positions 0, 1, 2, ...
        dim_order = []
        for i in range(len(buf_shape)):
            if i not in pos_to_dim:
                return None
            dim_order.append(pos_to_dim[i])

        return dim_order

    def _compute_permutation_from_pallas_stride(
        self,
        index: sympy.Expr,
        buf_shape: list[int],
    ) -> Optional[list[int]]:
        """Compute permutation to move reduction dims to end, based on PallasStride info.

        This is more robust than using global reduced_positions because it uses
        the actual PallasStride dim_index values to determine the buffer's layout.

        Returns permutation list or None if no permutation needed / not determinable.
        """
        dim_order = self._get_dim_order_from_index(index, buf_shape)
        if dim_order is None:
            return None

        reduced_positions = self._get_reduction_positions_from_ir()
        if reduced_positions is None:
            return None

        # Determine which buffer positions have reduction dims
        red_buf_positions = [i for i, d in enumerate(dim_order) if d in reduced_positions]
        non_red_buf_positions = [i for i, d in enumerate(dim_order) if d not in reduced_positions]

        # Check if reduction dims are already at end
        n_red = len(red_buf_positions)
        if n_red == 0:
            return None

        expected_end = list(range(len(buf_shape) - n_red, len(buf_shape)))

        if red_buf_positions == expected_end:
            return None  # Already in correct order

        return non_red_buf_positions + red_buf_positions

    def _compute_full_permutation_for_iteration_order(
        self,
        index: sympy.Expr,
        buf_shape: list[int],
    ) -> Optional[list[int]]:
        """Compute permutation to align buffer with iteration variable order.

        For index expression like:
            x0*PallasStride(1, 3) + x1*PallasStride(4, 2) +
            (x2//4)*PallasStride(64, 0) + ModularIndexing(x2, 1, 4)*PallasStride(16, 1)

        Detects when a split iteration variable (x2 split as x2//4 and x2%4) accesses
        non-adjacent buffer dimensions, requiring a transpose.

        For output (8, 4, 4) from buffer (2, 4, 4, 4):
        - x2 iterates over 8 = 2*4 (merged batch*head)
        - If buffer is (batch, seq, head, head_dim), batch and head are NOT adjacent
        - Need transpose (0, 2, 1, 3) to make them adjacent

        Key insight: For a split var (x2 -> x2//4, x2%4), the outer part (x2//4) has
        a specific size, and the inner part (x2%4) has a specific size. These sizes
        MUST match specific buffer dimensions. If the matched dimensions are not
        consecutive, we need a permutation.

        Returns: permutation list or None if no permutation needed.
        """
        from ..utils import FloorDiv, ModularIndexing

        pallas_strides = list(index.atoms(PallasStride))
        if not pallas_strides:
            return None

        # Build a mapping from coefficient (iteration part) to (stride, dim_index)
        expanded = sympy.expand(index)
        terms = list(expanded.args) if expanded.is_Add else [expanded]

        # Parse each term to extract: (coeff, stride_val, dim_idx, iter_var_base, is_outer_split, is_inner_split, split_size)
        iter_parts = []  # [(coeff, stride, dim, var_base, is_outer, is_inner, size)]
        for term in terms:
            term_strides = list(term.atoms(PallasStride))
            if not term_strides:
                continue
            ps = term_strides[0]
            stride_val = ps.stride_value
            if stride_val == PALLAS_EXPAND_STRIDE or stride_val is PALLAS_EXPAND_STRIDE:
                continue  # Skip expand dimensions
            stride_val = int(stride_val)
            dim_idx = int(ps.dim_index)
            coeff = term / ps

            # Determine iteration variable base and split info
            var_base = None
            is_outer = False
            is_inner = False
            size = None

            # Check for FloorDiv pattern: x2//4 -> outer part of split
            # In torch/inductor, integer division x//n becomes FloorDiv(x, n)
            floordiv_atoms = list(coeff.atoms(FloorDiv))
            if floordiv_atoms:
                for fd in floordiv_atoms:
                    # FloorDiv(x2, 4) - args[0] is variable, args[1] is divisor
                    if len(fd.args) == 2:
                        var_base = str(fd.args[0])
                        is_outer = True
                        size = int(fd.args[1])  # The divisor is the inner size
                        break

            # Check for ModularIndexing pattern: x2 % 4 -> inner part of split
            mod_atoms = list(coeff.atoms(ModularIndexing))
            if mod_atoms:
                for mi in mod_atoms:
                    var_base = str(mi.args[0])
                    is_inner = True
                    size = int(mi.args[2])  # modulo value
                    break

            # Check for simple variable (no split)
            if var_base is None:
                for sym in coeff.free_symbols:
                    if str(sym).startswith('x') or str(sym).startswith('r'):
                        var_base = str(sym)
                        break

            iter_parts.append((coeff, stride_val, dim_idx, var_base, is_outer, is_inner, size))

        if len(iter_parts) != len(buf_shape):
            return None

        # Group by iteration variable base to find splits
        var_groups = {}  # var_base -> list of (coeff, stride, dim, is_outer, is_inner, size)
        for coeff, stride, dim, var_base, is_outer, is_inner, size in iter_parts:
            if var_base not in var_groups:
                var_groups[var_base] = []
            var_groups[var_base].append((coeff, stride, dim, is_outer, is_inner, size))

        # Find split variables (those with both outer and inner parts)
        split_vars = {}
        for var_base, parts in var_groups.items():
            if len(parts) == 2:
                has_outer = any(p[3] for p in parts)
                has_inner = any(p[4] for p in parts)
                if has_outer and has_inner:
                    outer_part = next(p for p in parts if p[3])
                    inner_part = next(p for p in parts if p[4])
                    split_vars[var_base] = {
                        'outer_dim': outer_part[2],
                        'inner_dim': inner_part[2],
                        'outer_stride': outer_part[1],
                        'inner_stride': inner_part[1],
                        'inner_size': inner_part[5],
                    }

        # Check if any split var accesses non-adjacent dims
        # This indicates a missing permutation between the write and read
        needs_perm = False

        for var_base, info in split_vars.items():
            outer_dim = info['outer_dim']
            inner_dim = info['inner_dim']

            # Only trigger when split var explicitly accesses non-adjacent dims
            # e.g., outer at dim 0 and inner at dim 2 means dims 0 and 2 are being
            # merged, requiring a permute to make them adjacent
            if inner_dim != outer_dim + 1:
                needs_perm = True
                break

        if not needs_perm:
            return None

        # Compute the permutation needed
        # For split var accessing dims (d_outer, d_inner) where d_inner != d_outer + 1,
        # we need to move d_inner to be right after d_outer.
        #
        # Example: buf (2, 4, 4, 4), split var accesses dims 0 and 2
        # We need perm (0, 2, 1, 3) to make dims 0 and 2 adjacent
        #
        # Build the expected dim order based on iteration variable order
        # ROBUST: Use iter_var_pos from PallasStride - NO sort by stride

        # Helper to extract iter_var_pos from part's coefficient
        def get_iter_var_pos(part):
            coeff = part[0]
            if hasattr(coeff, 'atoms'):
                for ps in coeff.atoms(PallasStride):
                    if ps.iter_var_pos is not None:
                        return int(ps.iter_var_pos)
            return None

        # Sort by iter_var_pos ONLY - NO stride sorting, NO fallback
        missing_pos = [p for p in iter_parts if get_iter_var_pos(p) is None]
        if missing_pos:
            # NO FALLBACK - require iter_var_pos for all parts
            raise NotImplementedError(
                f"Permutation computation requires iter_var_pos for all parts, "
                f"but {len(missing_pos)} of {len(iter_parts)} parts are missing it. "
                f"Ensure iter_var_pos is set in all PallasStride markers. "
                f"buf_shape={buf_shape}"
            )
        iter_parts_sorted = sorted(iter_parts, key=lambda p: get_iter_var_pos(p))

        # Now we need to figure out what permutation would make the current
        # stride-based access correct.
        #
        # The current index uses wrong strides. For example:
        # - x2%4 uses stride 16 (dim 1) but should use stride 4 (dim 2)
        # - x1 uses stride 4 (dim 2) but should use stride 16 (dim 1)
        #
        # We need to permute the buffer so that when we access with the
        # CURRENT strides, we get the RIGHT data.
        #
        # If we transpose(buf, (0, 2, 1, 3)), then:
        # - buf_transposed has shape (2, 4, 4, 4) but different data layout
        # - What was at buf[b, s, h, d] is now at buf_transposed[b, h, s, d]
        # - Accessing buf_transposed with original strides (64, 16, 4, 1) gives us
        #   data in the transposed order
        #
        # So if the iteration expects (batch, head, seq, hdim) order but the
        # buffer is (batch, seq, head, hdim), we need perm = (0, 2, 1, 3)

        # Determine expected dim order based on what each iteration coeff SHOULD access
        # For split vars, we know outer should access smaller dim, inner should access next dim
        expected_dims = []
        used_dims = set()

        for coeff, stride, dim, var_base, is_outer, is_inner, size in iter_parts_sorted:
            if var_base in split_vars:
                info = split_vars[var_base]
                if is_outer:
                    expected_dims.append(info['outer_dim'])
                    used_dims.add(info['outer_dim'])
                elif is_inner:
                    # ROBUST: Use dim_index from PallasStride directly
                    expected_dims.append(info['inner_dim'])
                    used_dims.add(info['inner_dim'])
            else:
                expected_dims.append(dim)
                used_dims.add(dim)

        # If expected_dims doesn't cover all dims, fill in the rest
        if len(expected_dims) != len(buf_shape):
            return None

        # The permutation is the expected dim order
        # Verify it's a valid permutation
        if sorted(expected_dims) != list(range(len(buf_shape))):
            return None

        perm = expected_dims

        # Check if permutation is identity
        if perm == list(range(len(buf_shape))):
            return None

        return perm

    def check_bounds(
        self,
        expr: sympy.Expr,
        size: sympy.Expr,
        lower: bool,
        upper: bool,
    ) -> None:
        """Check array bounds for indirect indexing.

        Let JAX/Pallas handle bounds checking internally.
        This is called by indirect_indexing() in common.py.
        """
        pass

    def _get_indirect_vars(self, index: sympy.Expr) -> list[sympy.Symbol]:
        """Get TMP symbols (indirect index variables) from index expression."""
        return [s for s in index.free_symbols if str(s).startswith("tmp")]

    def _has_indirect_vars(self, index: sympy.Expr) -> bool:
        """Check if index expression contains indirect variables."""
        return len(self._get_indirect_vars(index)) > 0

    def _store_scatter(
        self,
        name: str,
        index: sympy.Expr,
        value: CSEVariable,
        mode: Optional[str],
        indirect_vars: list[sympy.Symbol],
    ) -> None:
        """Handle scatter store operations where the store index contains indirect vars.

        For scatter operations like: output[i, index[i]] = src[i]
        The store index contains a tmp variable (the loaded index value) which
        determines WHERE to store the value in the output buffer.

        Uses JAX's .at[].set() for scattered writes.
        """
        from ..ir import PallasStride

        var = self.args.output(name)
        output_shape = self._get_buffer_shape(name)
        output_numel = 1
        for s in output_shape:
            output_numel *= s

        # Resolve the index expression to compute linear indices
        # Replace PallasStride markers with their stride values
        resolved_index = index
        for ps in index.atoms(PallasStride):
            stride_val = ps.stride_value
            if isinstance(stride_val, sympy.Basic) and not stride_val.is_number:
                resolved_index = resolved_index.subs(ps, stride_val)
            else:
                resolved_index = resolved_index.subs(ps, sympy.Integer(int(stride_val)))

        index_str = str(resolved_index)

        # NOTE: We do NOT add iteration variable indexing to indirect variables here.
        # The indirect variables (loaded via _load_index_source) are ALREADY indexed
        # by iteration variables at load time. For example:
        #   tmp0 = in_ptr0[...].reshape(-1)[x3]
        # So tmp0 has shape (iter_numel,) where each element is the correct index value.
        # Adding [x3] again would double-index and produce wrong results.

        # Get the value to store
        val_str = str(value)

        # For scatter, use functional update on flattened array
        # output = output.at[linear_indices].set(values)
        if mode == "atomic_add":
            # Use .at[].add() for atomic add
            store_line = f"{var}_flat = {var}[...].reshape(-1).at[jnp.array({index_str}).astype(jnp.int32).reshape(-1)].add({val_str}.reshape(-1)); {var}[...] = {var}_flat.reshape({', '.join(str(s) for s in output_shape)})"
        else:
            # Use .at[].set() for regular scatter
            store_line = f"{var}_flat = {var}[...].reshape(-1).at[jnp.array({index_str}).astype(jnp.int32).reshape(-1)].set({val_str}.reshape(-1)); {var}[...] = {var}_flat.reshape({', '.join(str(s) for s in output_shape)})"

        self.stores.append((var, store_line))

        # Register this output as needing input_output_aliases
        # so that pallas_call initializes output with existing values
        self.scatter_outputs.add(var)

    def _extract_var_dim_from_index(
        self,
        index: sympy.Expr,
        target_var: sympy.Symbol,
    ) -> Optional[int]:
        """Extract which dimension a variable indexes from PallasStride info.

        For index = tmp0 * PallasStride(64, 0) + i2 * PallasStride(1, 1):
        - _extract_var_dim_from_index(index, tmp0) returns 0
        - _extract_var_dim_from_index(index, i2) returns 1

        This is ROBUST because we use the explicit dim_index from PallasStride,
        not coefficient-to-dimension guessing.
        """
        expanded = sympy.expand(index)
        terms = expanded.args if expanded.is_Add else [expanded]

        for term in terms:
            if target_var not in term.free_symbols:
                continue

            for atom in term.atoms(PallasStride):
                # Check if this PallasStride multiplies with target_var
                coeff = term / atom
                if target_var in coeff.free_symbols:
                    return int(atom.dim_index)

        return None

    def _get_indirect_var_shape(self, var_name: str) -> Optional[tuple[int, ...]]:
        """Get shape of an indirect variable via ROBUST view_id chain lookup.

        Priority:
        1. Look up CSE var's index_source_view_id attribute → view chain → shape
        2. Fall back to var_shapes dict (for backwards compatibility)

        This is robust because:
        - The view_id attribute is propagated through indirect_indexing transformations
        - The view chain (stored at lowering time) contains the original index tensor shape
        - No reliance on ephemeral CSE variable names that change during transformations
        """
        from ..ir import PallasViewTracker

        # ROBUST: Try view_id entry lookup first
        if var_name in self.cse.varname_map:
            cse_var = self.cse.varname_map[var_name]
            view_id = getattr(cse_var, 'index_source_view_id', None)
            if view_id is not None:
                chain = PallasViewTracker.get_effective_chain(view_id, None)
                if chain:
                    for transform in chain:
                        if transform.get("op") == "index_source":
                            shape = transform.get("shape")
                            if shape is not None:
                                return tuple(shape)

        # Fall back to var_shapes dict
        if var_name in self.var_shapes:
            return self.var_shapes[var_name]

        return None

    def _load_gather(
        self,
        name: str,
        index: sympy.Expr,
        indirect_vars: list[sympy.Symbol],
    ) -> CSEVariable:
        """Generate JAX gather code for index operations.

        Uses the view chain "gather" op (if present) to determine the correct
        gather strategy. The view chain explicitly tracks:
        - source_shape: Original buffer shape
        - output_shape: Desired output shape
        - indexed_dims: Which dimensions are being indexed

        For simple cases (buffer dims <= 2 or all dims indexed), uses advanced indexing.
        For multi-dimensional cases with partial indexing, uses flat gather with
        computed indices based on the iteration order from the view chain.
        """
        from ..ir import PallasViewTracker

        var = self.args.input(name)
        dtype = V.graph.get_dtype(name)
        buf_shape = self._get_buffer_shape(name)
        # Use sympy expressions for shapes - handles dynamic shapes correctly
        numel_expr = self._get_numel_expr()
        rnumel_expr = self._get_rnumel_expr()
        # Also get concrete values for conditional logic (may be None for symbolic)
        numel = self._get_numel()
        rnumel = self._get_rnumel()

        # Check for view chain - this is the SOURCE OF TRUTH for gather info
        view_id = self._extract_view_id_from_index(index)
        chain = None
        if view_id is not None:
            # Use get_effective_chain to handle materialization boundaries
            chain = PallasViewTracker.get_effective_chain(view_id, buf_shape)

        # ROBUST: Use consistent shape logic with main load method
        numel_renamed = self.rename_indexing(numel_expr)
        rnumel_renamed = self.rename_indexing(rnumel_expr)
        numel_str = self.kexpr(numel_renamed)
        rnumel_str = self.kexpr(rnumel_renamed)

        # Check if rnumel > 1 (handles symbolic case by checking if it's definitely 1)
        is_rnumel_one = (rnumel is not None and rnumel == 1) or (rnumel is None and rnumel_expr.is_number and int(rnumel_expr) == 1)

        # For compute_shape (internal tracking), use concrete values or None for symbolic
        if not is_rnumel_one:
            compute_shape = (numel, rnumel) if self.inside_reduction else (numel, 1)
            shape_str = f"{numel_str}, {rnumel_str}" if self.inside_reduction else f"{numel_str}, 1"
        else:
            compute_shape = (numel,)
            shape_str = numel_str

        # Check for "gather" op in view chain - this tells us exactly how to gather
        gather_info = None
        pre_gather_chain = []
        if chain:
            for i, t in enumerate(chain):
                if t.get("op") == "gather":
                    gather_info = t
                    break
                # Collect transforms BEFORE the gather (like unsqueeze)
                pre_gather_chain.append(t)

        # If there are pre-gather transforms (like unsqueeze), apply them first
        # This creates an intermediate expression that has the expected shape for gather
        base_var = var
        intermediate_shape = list(buf_shape)
        if pre_gather_chain:
            base_expr, intermediate_shape = self._apply_view_chain_transforms(var, name, pre_gather_chain)
            # We'll use base_expr instead of var[...] below
        else:
            base_expr = f"{var}[...]"

        # If we have explicit gather info, handle multi-dimensional partial indexing
        if gather_info is not None:
            indexed_dims = set(gather_info.get("indexed_dims", []))
            source_shape = gather_info.get("source_shape", buf_shape)
            output_shape = gather_info.get("output_shape", [])
            # Use explicit mapping from view chain - tracked upstream in lowering.py
            source_to_output_dim = gather_info.get("non_indexed_to_output", {})

            # For multi-dimensional buffers with partial indexing (not all dims indexed),
            # we need multi-dimensional advanced indexing. For each dimension:
            # - Indexed dims: use the index tensor variable directly
            # - Non-indexed dims: use Pallas iteration variable for that output dim
            # This handles both 2D (e.g., x[indices, :]) and higher-dimensional cases.
            if len(source_shape) >= 2 and len(indexed_dims) < len(source_shape):
                # Resolve the flat index expression
                pallas_strides = list(index.atoms(PallasStride)) if hasattr(index, 'atoms') else []
                resolved_index = index
                for ps in pallas_strides:
                    stride_val = ps.stride_value
                    if isinstance(stride_val, sympy.Basic) and not stride_val.is_number:
                        resolved_index = resolved_index.subs(ps, stride_val)
                    else:
                        resolved_index = resolved_index.subs(ps, sympy.Integer(int(stride_val)))
                resolved_index = self.rename_indexing(resolved_index)
                flat_index_str = self.kexpr(resolved_index)

                # Compute iteration index array
                iter_numel_str = f"{numel_str}*{rnumel_str}" if not is_rnumel_one else numel_str
                iter_idx_expr = f"jnp.arange({iter_numel_str})"

                # Convert shapes to kernel expressions for dynamic shapes support
                def shape_to_kexpr(shape_list):
                    result = []
                    for s in shape_list:
                        if isinstance(s, sympy.Basic):
                            renamed = self.rename_indexing(s)
                            result.append(self.kexpr(renamed))
                        else:
                            result.append(str(s))
                    return result

                output_shape_strs = shape_to_kexpr(output_shape)
                source_shape_strs = shape_to_kexpr(source_shape)

                # Build mapping from output dim to Pallas iteration variable name
                # POSITION-BASED (not size-based) mapping:
                # - Kernel iteration order is: numel dims first, then rnumel dims
                # - _dim_to_var[i] gives the x variable for numel dim i
                # - _rdim_to_var[j] gives the r variable for rnumel dim j
                # - First len(_dim_to_var) non-trivial output dims → numel (x vars)
                # - Remaining non-trivial output dims → rnumel (r vars)
                non_one_output_dims = [(i, output_shape[i], output_shape_strs[i])
                                for i in range(len(output_shape))
                                if not (isinstance(output_shape[i], (int, sympy.Integer)) and int(output_shape[i]) == 1)]

                # POSITION-BASED split: first N dims are numel, rest are rnumel
                # where N = number of numel dimensions in the kernel
                num_kernel_numel_dims = len(self._dim_to_var)
                num_kernel_rnumel_dims = len(self._rdim_to_var)

                # Map: output_dim -> Pallas variable name
                output_dim_to_pallas_var = {}

                for pos, (orig_dim_idx, _, _) in enumerate(non_one_output_dims):
                    if pos < num_kernel_numel_dims:
                        # This output dim corresponds to a numel (x) dimension
                        # local_idx within numel dims = pos
                        if pos in self._dim_to_var:
                            output_dim_to_pallas_var[orig_dim_idx] = self._dim_to_var[pos]
                    else:
                        # This output dim corresponds to a rnumel (r) dimension
                        # local_idx within rnumel dims = pos - num_kernel_numel_dims
                        local_r_idx = pos - num_kernel_numel_dims
                        if local_r_idx in self._rdim_to_var:
                            output_dim_to_pallas_var[orig_dim_idx] = self._rdim_to_var[local_r_idx]

                # We still need dim_stride_exprs for compatibility with other code paths
                dim_stride_exprs = {}
                cumulative_stride_expr = "1"
                for i in range(len(non_one_output_dims) - 1, -1, -1):
                    orig_dim_idx, _, dim_size_str = non_one_output_dims[i]
                    dim_stride_exprs[orig_dim_idx] = cumulative_stride_expr
                    if cumulative_stride_expr == "1":
                        cumulative_stride_expr = dim_size_str
                    else:
                        cumulative_stride_expr = f"({cumulative_stride_expr})*({dim_size_str})"

                # EXPLICIT VIEW_ID-BASED TRACKING (no stride matching)
                # Get index_tensor_view_ids from gather_info: indexed_dim -> view_id
                index_tensor_view_ids = gather_info.get("index_tensor_view_ids", {})

                # Build reverse mapping: view_id -> indexed_dim
                view_id_to_indexed_dim = {vid: dim for dim, vid in index_tensor_view_ids.items()}

                # Extract tmp variables from flat_index_str
                import re
                tmp_var_pattern = re.compile(r'tmp\d+')
                tmp_vars_in_expr = tmp_var_pattern.findall(flat_index_str)

                # Map indexed dims to their tmp variables using explicit view_id tracking
                # Each tmp variable came from _load_index_source which stored its view_id
                indexed_dim_to_var = {}
                for tmp_var in tmp_vars_in_expr:
                    var_info = self._index_source_vars.get(tmp_var, {})
                    var_view_id = var_info.get("view_id")
                    if var_view_id is not None and var_view_id in view_id_to_indexed_dim:
                        indexed_dim = view_id_to_indexed_dim[var_view_id]
                        indexed_dim_to_var[indexed_dim] = tmp_var

                # Build index expression for each source dimension
                dim_index_exprs = []

                # Get index shapes from gather_info - tells us the shape of each index tensor
                index_shapes = gather_info.get("index_shapes", [])

                for dim_idx in range(len(source_shape)):
                    dim_size = source_shape[dim_idx]
                    dim_size_str = source_shape_strs[dim_idx]

                    if dim_idx in indexed_dims:
                        # Indexed dim: use the extracted index tensor variable
                        if dim_idx in indexed_dim_to_var:
                            var_name = indexed_dim_to_var[dim_idx]

                            # Get EXPLICIT index_to_output_dims mapping from gather_info
                            # This tells us which output dimensions this index tensor covers
                            # NO SIZE MATCHING - use explicit tracking from upstream
                            index_to_output_dims = gather_info.get("index_to_output_dims", {})
                            covered_output_dims = index_to_output_dims.get(dim_idx, [])

                            # Check if this variable (or a source variable with same view_id) was loaded
                            # from a tensor. This distinguishes:
                            # - Derived tensors (e.g., tmp4 from negative index wrapping of tmp0)
                            #   where tmp0 has is_loaded_tensor=True -> should subscript
                            # - Pure fused scalars (e.g., fill+clamp) where no source has
                            #   is_loaded_tensor=True -> should NOT subscript
                            var_info = self._index_source_vars.get(var_name, {})
                            var_view_id = var_info.get("view_id")
                            is_derived_from_loaded = var_info.get("is_loaded_tensor", False)

                            # If this var doesn't have is_loaded_tensor, check if any var
                            # with the same view_id has it (handles derived variables)
                            if not is_derived_from_loaded and var_view_id is not None:
                                for other_name, other_info in self._index_source_vars.items():
                                    if (other_info.get("view_id") == var_view_id and
                                        other_info.get("is_loaded_tensor", False)):
                                        is_derived_from_loaded = True
                                        break

                            if covered_output_dims and is_derived_from_loaded:
                                # The index tensor was loaded as 1D (flattened) by _load_index_source.
                                # We need to compute a FLAT index using the Pallas iteration vars.
                                # For covered_output_dims = [0, 1] with sizes [s0, s1]:
                                # flat_idx = x_dim0 * s1 + x_dim1
                                # This follows C-order (row-major) flattening.

                                if len(covered_output_dims) == 1:
                                    # Single dimension - just use the Pallas var directly
                                    out_dim = covered_output_dims[0]
                                    if out_dim in output_dim_to_pallas_var:
                                        dim_index_exprs.append(f"{var_name}[{output_dim_to_pallas_var[out_dim]}]")
                                    else:
                                        dim_index_exprs.append(var_name)
                                else:
                                    # Multiple dimensions - compute flat index
                                    # flat_idx = sum(x_i * stride_i) where stride_i = prod(sizes[i+1:])
                                    flat_idx_parts = []
                                    for i, out_dim in enumerate(covered_output_dims):
                                        if out_dim not in output_dim_to_pallas_var:
                                            continue
                                        pallas_var = output_dim_to_pallas_var[out_dim]
                                        # Compute stride: product of sizes of all dimensions after this one
                                        stride_parts = []
                                        for j in range(i + 1, len(covered_output_dims)):
                                            later_out_dim = covered_output_dims[j]
                                            later_size_str = output_shape_strs[later_out_dim]
                                            stride_parts.append(later_size_str)

                                        if stride_parts:
                                            stride_str = "*".join(stride_parts)
                                            flat_idx_parts.append(f"({pallas_var})*({stride_str})")
                                        else:
                                            # Last dimension, stride is 1
                                            flat_idx_parts.append(pallas_var)

                                    if flat_idx_parts:
                                        flat_idx_expr = " + ".join(flat_idx_parts)
                                        dim_index_exprs.append(f"{var_name}[{flat_idx_expr}]")
                                    else:
                                        dim_index_exprs.append(var_name)
                            else:
                                # Either:
                                # 1. No explicit output dims mapping (covered_output_dims is empty)
                                # 2. Variable is not derived from a loaded tensor (fused scalar)
                                # In both cases, use the variable directly without subscripting.
                                # Cast to int32 for JAX indexing compatibility.
                                dim_index_exprs.append(f"({var_name}).astype(jnp.int32)")
                        else:
                            # Fallback if we couldn't extract the variable
                            dim_index_exprs.append("0")
                    elif isinstance(dim_size, (int, sympy.Integer)) and int(dim_size) == 1:
                        # Size-1 dim: always index 0
                        dim_index_exprs.append("0")
                    else:
                        # Non-indexed (passthrough) dim: depends on if it's numel or rnumel
                        out_dim = source_to_output_dim.get(dim_idx)
                        pallas_var = output_dim_to_pallas_var.get(out_dim) if out_dim is not None else None

                        if pallas_var is not None and pallas_var.startswith("r"):
                            # Reduction dimension: use slice to take ALL elements
                            # The kernel will reduce over these, so we need the full dimension
                            dim_index_exprs.append(":")
                        elif pallas_var is not None:
                            # Numel dimension: use iteration variable for element-by-element access
                            dim_index_exprs.append(pallas_var)
                        else:
                            # Fallback: use slice for safety
                            dim_index_exprs.append(":")

                # Build the advanced indexing expression
                # Use base_expr which has pre-gather transforms applied (e.g., unsqueeze)
                index_parts = ", ".join(dim_index_exprs)
                load_expr = f"{base_expr}[{index_parts}].reshape({shape_str})"
                result = self.cse.generate(self.compute, load_expr, dtype=dtype, shape=compute_shape)
                self.var_shapes[str(result)] = compute_shape
                return result

        # For simpler cases, use advanced indexing
        # Apply view chain transforms first if present
        if chain:
            base_expr, current_shape = self._apply_view_chain_transforms(var, name, chain)
            n_dims = len(current_shape)
        else:
            base_expr = var
            current_shape = list(buf_shape)
            n_dims = len(buf_shape)

        # Map dimension -> indirect variable name
        dim_to_indirect: dict[int, str] = {}
        for indirect_var in indirect_vars:
            indexed_dim = self._extract_var_dim_from_index(index, indirect_var)
            if indexed_dim is None:
                raise NotImplementedError(
                    f"Could not determine indexed dimension for indirect var '{indirect_var}' "
                    f"in buffer '{name}' with shape {buf_shape}. "
                    f"PallasStride info missing or incomplete in index: {index}"
                )
            dim_to_indirect[indexed_dim] = str(indirect_var)

        # Validate that all indexed dimensions are within transformed shape bounds
        for indexed_dim in dim_to_indirect.keys():
            if indexed_dim >= n_dims:
                raise NotImplementedError(
                    f"Indirect variable indexes dimension {indexed_dim} "
                    f"but transformed tensor only has {n_dims} dimensions (shape={current_shape}). "
                    f"Original buffer shape={buf_shape}, view_id={view_id}, chain={chain}. "
                    f"index: {index}"
                )

        # Build slices for each dimension of the transformed shape
        #
        # For non-indexed dimensions, we need to decide between:
        # 1. ":" (take all) - when the full dimension is needed
        # 2. "jnp.arange(N)" - when we want element-by-element access to match iteration domain
        #
        # The key insight: when mixing ":" with indirect indexing, we get a cartesian product.
        # Example: nll_loss with (5,5) buffer, dim 1 indexed by 5 labels
        # - Using [:, labels] gives (5, 5) = 25 elements (5 rows * 5 indices)
        # - But iteration domain is 5 elements (one per sample)
        # - Correct: [jnp.arange(5), labels] gives 5 elements (diagonal-like access)
        #
        # Compute what shape we'd get if we use ":" for all non-indexed dims
        non_indexed_product = 1
        for dim in range(n_dims):
            if dim not in dim_to_indirect:
                dim_size = current_shape[dim]
                if isinstance(dim_size, (int, sympy.Integer)):
                    non_indexed_product *= int(dim_size)

        # Get expected iteration numel
        iter_numel = numel * rnumel if rnumel is not None else numel

        # Determine if we need element-by-element access for non-indexed dims
        # When using ":" with indirect indexing, we get cartesian product:
        # result_numel = non_indexed_product * index_count
        # If this exceeds iter_numel, we need element-wise access instead.
        #
        # Key insight: if non_indexed_product > 1 and we have indirect indexing,
        # the cartesian product will likely exceed iter_numel. The only exception
        # is when we truly want to broadcast (reduction over gathered elements).
        # For nll_loss-style operations, non_indexed_product == iter_numel means
        # we want diagonal access, not cartesian product.
        need_elementwise = (
            iter_numel is not None and
            non_indexed_product > 1 and
            len(dim_to_indirect) > 0  # We have indirect indexing
        )

        slices = []
        for dim in range(n_dims):
            if dim in dim_to_indirect:
                var_name = dim_to_indirect[dim]
                orig_shape = self._get_indirect_var_shape(var_name)

                if orig_shape is not None and len(orig_shape) > 1 and len(indirect_vars) > 1:
                    shape_str_slice = ", ".join(str(s) for s in orig_shape)
                    slices.append(f"{var_name}.reshape({shape_str_slice})")
                elif len(indirect_vars) > 1:
                    slices.append(f"{var_name}.reshape(-1, 1)")
                else:
                    slices.append(f"{var_name}.reshape(-1)")
            else:
                dim_size = current_shape[dim]
                if need_elementwise and isinstance(dim_size, (int, sympy.Integer)) and int(dim_size) > 1:
                    # Element-by-element access: use arange to match iteration indices
                    slices.append(f"jnp.arange({int(dim_size)})")
                else:
                    slices.append(":")

        slice_str = ", ".join(slices)
        load_expr = f"{base_expr}[{slice_str}].reshape({shape_str})"
        result = self.cse.generate(self.compute, load_expr, dtype=dtype, shape=compute_shape)
        self.var_shapes[str(result)] = compute_shape
        return result

    def _load_index_source(
        self,
        name: str,
        index: sympy.Expr,
    ) -> CSEVariable:
        """Load buffer that will be used as indirect index source.

        These are smaller than the full iteration space (e.g., token indices).
        We load and index them by the appropriate iteration variable for proper broadcasting.

        For example, if we have:
        - a[b, c] where a is (8, 8, 12), b and c are (4,)
        - Output is (4, 12) = 48 elements
        - x0 = column index (0-11), x1 = row index (0-3, repeated)
        - b and c have shape (4,) and correspond to rows
        - We load as: b[...].reshape(-1)[x1] to get shape (48,) via indexing

        ROBUST shape tracking:
        - Extract view_id from PallasStride markers in index
        - Apply view chain transforms (slice, squeeze, etc.) before flattening
        - Index by iteration variable matching the index source's dimension
        """
        var = self.args.input(name)
        dtype = V.graph.get_dtype(name)
        buf_shape = self._get_buffer_shape(name)

        # Extract view_id and get the view chain
        view_id = self._extract_view_id_from_index(index)
        chain = None
        if view_id is not None:
            chain = self._get_transform_chain_from_view_id(view_id, buf_shape)

        # IMPORTANT: Apply view chain transforms (slice, squeeze, permute, etc.) first
        # This is needed when the index tensor is a view (e.g., edge_index[0] is a slice of edge_index)
        # The chain contains transforms like [{op: slice, dim: 0, start: 0, end: 1}, {op: squeeze, dim: 0}]
        #
        # CRITICAL: Skip 'expand' transforms for index sources!
        # Expand/broadcast is handled by JAX's native advanced indexing at gather time.
        # If we apply expand here, the index tensor becomes (4,4)=16 elements,
        # but _get_indirect_var_shape returns the pre-expand shape (1,4)=4 elements,
        # causing reshape errors in _load_gather.
        if chain:
            # Filter out index_source markers (metadata) and expand transforms (handled at gather)
            transforms = [t for t in chain if t.get("op") not in ("index_source", "expand")]
            if transforms:
                # Apply transforms using _apply_view_chain_transforms
                load_expr, current_shape = self._apply_view_chain_transforms(var, name, transforms)
                # Now flatten the transformed result
                load_expr = f"{load_expr}.reshape(-1)"
                # Track the actual shape after transforms (for var_shapes tracking)
                actual_shape = tuple(current_shape)
            else:
                # No transforms (other than index_source/expand), just flatten
                load_expr = f"{var}[...].reshape(-1)"
                # Get shape from index_source marker in chain (more accurate than buf_shape
                # because it reflects the shape BEFORE broadcast but AFTER any views like unsqueeze)
                actual_shape = tuple(buf_shape)
                for t in chain:
                    if t.get("op") == "index_source" and t.get("shape"):
                        actual_shape = tuple(t["shape"])
                        break
        else:
            # No chain, just flatten the whole buffer
            load_expr = f"{var}[...].reshape(-1)"
            actual_shape = tuple(buf_shape)

        # CRITICAL: For multi-dimensional iteration with index sources that are smaller
        # than the iteration space, we need to index by the appropriate iteration variable.
        # This ensures proper broadcasting when the index source is used in expressions.
        #
        # Example: a[b, c] where output is (4, 12) and b, c are (4,)
        # - x0 is column index (shape 48,)
        # - x1 is row index (shape 48,)
        # - b has 4 elements, needs to be indexed by x1 to broadcast to 48 elements
        buf_numel = 1
        for s in actual_shape:
            if isinstance(s, (int, sympy.Integer)):
                buf_numel *= int(s)
            else:
                buf_numel = None
                break

        # ROBUST: Get iter_dims and broadcast_dims from view chain to determine indexing
        # This is the principled approach - use explicitly tracked info, not size matching
        iter_dims = None
        broadcast_dims = set()
        if chain:
            for transform in chain:
                if transform.get("op") == "index_source":
                    iter_dims = transform.get("iter_dims", None)
                    broadcast_dims = set(transform.get("broadcast_dims", []))
                    break

        if iter_dims is not None and len(self._dim_to_var) > 1:
            # Use iter_dims from view chain - this tells us exactly which iteration
            # dimensions this index source corresponds to
            # IMPORTANT: Skip broadcast dims (size 1) - they don't need iteration variable indexing
            # broadcast_dims is an index into the index tensor's shape, corresponding to iter_dims
            for i, dim_idx in enumerate(iter_dims):
                if i in broadcast_dims:
                    # This dim of the index tensor has size 1 (broadcast) - skip indexing
                    continue
                if dim_idx in self._dim_to_var:
                    var_name = self._dim_to_var[dim_idx]
                    load_expr = f"{load_expr}[{var_name}]"
            # Shape becomes the full iteration numel after indexing
            actual_shape = None

        elif buf_numel is not None and len(self._dim_to_var) > 1:
            # FALLBACK: Size-based matching (not robust, but kept for legacy)
            # Find which dimension this index source matches
            for dim_idx, var_name in sorted(self._dim_to_var.items()):
                if dim_idx < len(self._original_ranges[0]) if self._original_ranges else False:
                    dim_size = self._original_ranges[0][dim_idx]
                    if dim_size is not None and buf_numel == dim_size:
                        load_expr = f"{load_expr}[{var_name}]"
                        actual_shape = None
                        break

        # IMPORTANT: Use shape=None to bypass Pallas's index_expr shape override
        # during indirect_indexing. Shape is tracked via view_id chain.
        result = self.cse.generate(self.compute, load_expr, dtype=dtype, shape=None)

        # ROBUST: Set view_id attribute for propagation through indirect_indexing
        # This attribute will be propagated when CSEProxy.indirect_indexing creates new vars
        if view_id is not None:
            result.index_source_view_id = view_id

            # Store iter_dims and view_id info for use in _store_scatter and _load_gather
            # The view_id enables explicit tracking: tmp_var -> view_id -> indexed_dim
            #
            # CRITICAL: is_loaded_tensor=True distinguishes tensors loaded via _load_index_source
            # from fused scalars that go through indirect_indexing (which adds entries with only view_id).
            # This is EXPLICIT tracking, NOT inference from shape presence.
            self._index_source_vars[str(result)] = {
                "iter_dims": iter_dims if iter_dims else [],
                "shape": list(actual_shape) if actual_shape else None,
                "view_id": view_id,  # EXPLICIT tracking: look up indexed_dim via index_tensor_view_ids
                "is_loaded_tensor": True,  # EXPLICIT flag: this was loaded from a tensor buffer
            }

        # Also track in var_shapes as backup (for CSE vars that don't go through indirect_indexing)
        self.var_shapes[str(result)] = actual_shape
        return result

    def _load_gather_with_index_expr(
        self,
        name: str,
        index: sympy.Expr,
    ) -> CSEVariable:
        """Load buffer with explicit index computation for gather patterns.

        This handles cases like unfold where buf_numel < iter_numel due to
        element reuse (overlapping windows). The index expression encodes
        the mapping from output position to input position.

        For unfold(0, 2, 1) on (4, 4) -> (3, 4, 2):
          index = 4*x0 + x1 + 4*x2
        This means elements are reused, requiring explicit gather.
        """
        var = self.args.input(name)
        dtype = V.graph.get_dtype(name)
        shape_str = self._get_compute_shape_str()

        # Resolve PallasStride markers using contiguous strides.
        # Buffers are made contiguous via .contiguous() before being passed
        # to Pallas kernels, so flat indices must use contiguous strides.
        resolved_index = self._resolve_index_with_contiguous_strides(index, name)

        # Convert the resolved expression to a string
        index_str = str(resolved_index)

        # The resulting index array indexes into the flat input buffer
        # Use jnp.take to gather elements
        # Note: in Pallas, we need to read the data with [...] first before reshape
        load_expr = f"jnp.take({var}[...].reshape(-1), ({index_str}).astype(jnp.int32).reshape(-1), mode='wrap').reshape({shape_str})"

        # Compute shape for CSEVariable
        # ROBUST: In reduction kernels, use 2D shapes consistently
        numel = self._get_numel()
        rnumel = self._get_rnumel()
        if rnumel > 1:
            compute_shape = (numel, rnumel) if self.inside_reduction else (numel, 1)
        else:
            compute_shape = (numel,)
        result = self.cse.generate(self.compute, load_expr, dtype=dtype, shape=compute_shape)
        self.var_shapes[str(result)] = compute_shape
        return result

    def _infer_permutation_from_index(
        self,
        index: sympy.Expr,
        current_shape: list[int],
        iter_shape: list[int],
    ) -> Optional[list[int]]:
        """
        Infer the permutation needed to go from current_shape to iter_shape.

        ROBUST: Uses iter_var_pos from PallasStride for direct buffer position mapping,
        avoiding collisions when multiple dims have the same stride (e.g., size-1 dims).

        Returns a permutation list or None if no permutation is needed/detectable.
        """
        from ..ir import PallasStride

        # Only handle cases where shapes have the same number of dimensions
        if len(current_shape) != len(iter_shape):
            return None

        # Extract PallasStride info from index
        if not hasattr(index, 'atoms'):
            return None

        strides = list(index.atoms(PallasStride))
        if not strides:
            return None

        # Build direct mapping: buf_position -> dim_index using iter_var_pos
        pos_to_dim: dict[int, int] = {}
        for ps in strides:
            dim_idx = int(ps.dim_index)
            if dim_idx >= len(current_shape):
                continue

            # Use iter_var_pos for direct buffer position mapping
            iter_pos = ps.iter_var_pos
            if iter_pos is not None:
                buf_pos = int(iter_pos)
                pos_to_dim[buf_pos] = dim_idx

        if len(pos_to_dim) != len(current_shape):
            return None  # Not all positions mapped

        # Build dim_mapping from buffer positions 0, 1, 2, ...
        dim_mapping: list[int] = []
        for i in range(len(current_shape)):
            if i not in pos_to_dim:
                return None
            dim_mapping.append(pos_to_dim[i])

        # Check if dim_mapping is a valid permutation
        sorted_dims = sorted(dim_mapping)
        expected_dims = list(range(len(current_shape)))
        if sorted_dims != expected_dims:
            return None

        return dim_mapping

    def _compute_reduction_dim_permutation(
        self,
        buffer_name: str,
        target_shape: list[int],
        index: sympy.Expr,
    ) -> Optional[list[int]]:
        """
        Compute permutation to move reduction dims to the end of target_shape.

        For pallas, iter_shape is constructed from _original_ranges as:
            iter_shape = non_reduction_dims + reduction_dims
        So by construction, reduction dims are ALREADY at the end of iter_shape.

        This function is only needed when target_shape differs from iter_shape
        and the reduction dims are NOT at the end in target_shape.

        Args:
            buffer_name: The name of the buffer being loaded
            target_shape: The shape after transforms (e.g., [2, 4, 16, 8])
            index: The sympy index expression containing PallasStride terms

        Returns permutation or None if no reordering needed.
        """
        if len(self._original_ranges) < 2:
            return None  # No reduction dims

        red_dim_sizes = self._original_ranges[1]
        if not red_dim_sizes:
            return None

        # No permutation possible for 1D arrays
        if len(target_shape) <= 1:
            return None

        # In iter_shape, reduction dims are at the end by construction.
        # Check if target_shape matches iter_shape (meaning reduction dims are already at end)
        iter_shape = list(self._original_ranges[0]) + list(self._original_ranges[1])
        if target_shape == iter_shape:
            # Reduction dims are already at the end - no permutation needed
            return None

        # Get reduction positions from Reduction IR node
        reduced_positions = self._get_reduction_positions_from_ir()

        if reduced_positions is None:
            return None

        # Use explicit reduction positions from IR node
        # IMPORTANT: Filter to only include valid positions for current target_shape
        # The IR may store positions from the original higher-dim shape, but we've
        # reshaped to a lower-dim shape where those positions don't exist
        red_positions = [p for p in reduced_positions if p < len(target_shape)]

        # If no valid reduction positions remain, no permutation needed
        if not red_positions:
            return None

        non_red_positions = [i for i in range(len(target_shape)) if i not in red_positions]

        # Validate: the permutation should cover all positions exactly once
        perm = non_red_positions + red_positions
        if sorted(perm) != list(range(len(target_shape))):
            # Invalid permutation - some positions missing or duplicated
            return None

        # If all reduction dims are already at the end, no permutation needed
        expected_red_positions = list(range(len(target_shape) - len(red_positions), len(target_shape)))
        if red_positions == expected_red_positions:
            return None

        # Permutation: non-reduction positions first, then reduction positions
        return perm

    def _apply_transforms_core(
        self,
        var: str,
        name: str,
        chain: list[dict],
        use_kexpr: bool = False,
        track_dim_map: bool = False,
    ) -> tuple[str, list, Optional[list[Optional[int]]]]:
        """
        Core transform application logic used by multiple methods.

        Args:
            var: Input variable name
            name: Buffer name
            chain: List of transform dicts
            use_kexpr: If True, use rename_indexing/kexpr for sympy support
            track_dim_map: If True, track dimension mapping

        Returns:
            (expr, current_shape, dim_map or None)
        """
        buf_shape = self._get_buffer_shape(name)

        def shape_to_str(shape_list):
            if use_kexpr:
                parts = []
                for s in shape_list:
                    if isinstance(s, sympy.Basic):
                        renamed = self.rename_indexing(s)
                        parts.append(self.kexpr(renamed))
                    else:
                        parts.append(str(s))
                return ", ".join(parts)
            return ", ".join(str(s) for s in shape_list)

        expr = f"{var}[...]"
        current_shape = list(buf_shape)
        dim_map: Optional[list[Optional[int]]] = list(range(len(buf_shape))) if track_dim_map else None

        for transform in chain:
            op = transform["op"]

            if op == "permute":
                dims = transform["dims"]
                perm_str = ", ".join(str(d) for d in dims)
                expr = f"jnp.transpose({expr}, ({perm_str},))"
                current_shape = [current_shape[d] for d in dims]
                if dim_map is not None:
                    dim_map = [dim_map[d] for d in dims]

            elif op == "unsqueeze":
                dim = transform["dim"]
                current_shape.insert(dim, 1)
                if dim_map is not None:
                    dim_map.insert(dim, None)
                expr = f"{expr}.reshape({shape_to_str(current_shape)})"

            elif op == "reshape":
                to_shape = transform["to_shape"]
                expr = f"{expr}.reshape({shape_to_str(to_shape)})"
                current_shape = list(to_shape)
                if dim_map is not None:
                    dim_map = list(range(len(to_shape)))

            elif op == "expand":
                expand_dims = transform.get("expand_dims", [])
                target_shape = transform.get("target_shape", [])
                intermediate_shape = list(target_shape)
                for ed in expand_dims:
                    intermediate_shape[ed] = 1
                if len(current_shape) != len(intermediate_shape):
                    expr = f"{expr}.reshape({shape_to_str(intermediate_shape)})"
                    current_shape = list(intermediate_shape)
                    if dim_map is not None:
                        for ed in sorted(expand_dims, reverse=True):
                            dim_map.insert(ed, None)
                expr = f"jnp.broadcast_to({expr}, ({shape_to_str(target_shape)},))"
                current_shape = list(target_shape)

            elif op == "slice":
                dim = transform["dim"]
                start = transform.get("start", 0)
                end = transform.get("end", current_shape[dim])
                step = transform.get("step", 1)
                slices = []
                for i in range(len(current_shape)):
                    if i == dim:
                        slices.append(f"{start}:{end}" if step == 1 else f"{start}:{end}:{step}")
                    else:
                        slices.append(":")
                expr = f"{expr}[{', '.join(slices)}]"
                current_shape[dim] = (end - start + step - 1) // step

            elif op == "flip":
                dims = transform["dims"]
                if len(dims) == 1:
                    expr = f"jnp.flip({expr}, axis={dims[0]})"
                else:
                    expr = f"jnp.flip({expr}, axis=({', '.join(str(d) for d in dims)},))"

            elif op == "squeeze":
                dim = transform["dim"]
                if dim < len(current_shape):
                    current_shape.pop(dim)
                    if dim_map is not None and dim < len(dim_map):
                        dim_map.pop(dim)
                    expr = f"jnp.squeeze({expr}, axis={dim})"

            elif op in ("index_source", "gather_indices", "unfold"):
                pass  # Skip markers

        return expr, current_shape, dim_map

    def _apply_view_chain_transforms(
        self,
        var: str,
        name: str,
        chain: list[dict],
    ) -> tuple[str, list[int]]:
        """
        Apply view chain transforms to build an intermediate expression.

        Returns (expr, current_shape) - the expression after transforms and its shape.
        This is used by _load_gather to apply view transforms before advanced indexing.

        Unlike _build_load_from_view_chain, this does NOT do final reshape to iteration space.
        """
        expr, shape, _ = self._apply_transforms_core(var, name, chain, use_kexpr=True)
        return expr, shape

    def _build_load_from_view_chain(
        self,
        var: str,
        name: str,
        chain: list[dict],
        index: sympy.Expr,
        numel: Optional[int],
        rnumel: Optional[int],
        numel_expr: Optional[sympy.Basic] = None,
        rnumel_expr: Optional[sympy.Basic] = None,
    ) -> str:
        """
        Build load expression from an explicit transform chain.

        The chain is a list of transforms like:
        [{"op": "permute", "dims": [2, 0, 1]}, {"op": "unsqueeze", "dim": 1}, {"op": "permute", "dims": [0, 2, 1, 3]}]

        Returns the JAX expression string by applying transforms step by step.

        Args:
            numel: Concrete numel value (None if symbolic)
            rnumel: Concrete rnumel value (None if symbolic)
            numel_expr: Sympy expression for numel (for dynamic shapes)
            rnumel_expr: Sympy expression for rnumel (for dynamic shapes)
        """
        # Use unified transform helper (with dim_map tracking for reduction permutation)
        expr, current_shape, dim_map = self._apply_transforms_core(
            var, name, chain, use_kexpr=False, track_dim_map=True
        )

        # Check if we need to transpose for non-last-dim reduction
        # Get reduction positions and permute if needed
        target_shape = current_shape
        red_perm = self._compute_reduction_dim_permutation(name, target_shape, index)

        if red_perm is not None:
            perm_str = ", ".join(str(p) for p in red_perm)
            expr = f"jnp.transpose({expr}, ({perm_str},))"

        # Check if current shape matches iteration space
        current_numel = 1
        for s in current_shape:
            current_numel *= s

        # Handle symbolic shapes - use sympy expressions if concrete values not available
        if numel is not None and rnumel is not None:
            iter_numel = numel * rnumel
            numel_matches = current_numel == numel
            rnumel_gt_1 = rnumel > 1
        elif numel_expr is not None and rnumel_expr is not None:
            # Use symbolic comparison
            iter_numel_sym = numel_expr * rnumel_expr
            # Try to determine equality symbolically
            diff = sympy.simplify(current_numel - iter_numel_sym)
            if diff == 0:
                iter_numel = current_numel  # They're equal
            else:
                iter_numel = None  # Can't determine at compile time
            # For numel_matches: check if current_numel - numel_expr simplifies to 0
            numel_diff = sympy.simplify(current_numel - numel_expr)
            numel_matches = numel_diff == 0
            # For rnumel > 1: check if rnumel_expr - 1 is positive
            rnumel_gt_1 = self._safe_int(rnumel_expr) is None or self._safe_int(rnumel_expr) > 1
        else:
            # No shape info available - skip numel check
            iter_numel = None
            numel_matches = False
            rnumel_gt_1 = False

        # ROBUST: If numel doesn't match, handle specific patterns
        if iter_numel is not None and current_numel != iter_numel:
            # Check if view chain has explicit marker for index buffer
            # This is set by embedding/gather lowering via _set_pallas_outer_view_id()
            is_index_buffer = any(
                t.get("op") in ("index_source", "gather_indices", "unfold") or
                t.get("no_final_reshape", False)
                for t in chain
            )

            if is_index_buffer:
                return expr  # Explicitly marked - don't reshape

            # Post-reduction broadcast pattern: buf_numel == numel and rnumel > 1
            #
            # This is MATHEMATICALLY UNAMBIGUOUS, not a heuristic. Here's why:
            #
            # In Pallas 2D reduction model:
            #   - Iteration space is (numel, rnumel) where rnumel is the reduction axis
            #   - numel = product of non-reduction dimensions
            #   - rnumel = product of reduction dimensions
            #
            # When a buffer has exactly `numel` elements (not `numel * rnumel`):
            #   - It cannot participate in the full iteration space
            #   - It MUST broadcast along the reduction axis (rnumel)
            #   - The ONLY valid shape is (numel, 1) to broadcast to (numel, rnumel)
            #
            # Alternative interpretations are impossible:
            #   - (1, numel) would require buf_numel == rnumel, not buf_numel == numel
            #   - (sqrt(numel), sqrt(numel)) would require numel to be a perfect square
            #     AND explicit reshape info - we'd have view_id chain for that
            #
            # Common sources of this pattern:
            #   - Reduction outputs: y = x.sum(dim=-1) produces shape (numel,)
            #   - Broadcast inputs: bias vector added after reduction
            #
            # The NotImplementedError below catches any unexpected cases where
            # buf_numel doesn't match numel or iter_numel.
            if numel_matches and rnumel_gt_1:
                # Use concrete or symbolic numel for reshape
                if numel is not None:
                    return f"{expr}.reshape({numel}, 1)"
                elif numel_expr is not None:
                    numel_renamed = self.rename_indexing(numel_expr)
                    numel_str = self.kexpr(numel_renamed)
                    return f"{expr}.reshape({numel_str}, 1)"

            # Broadcast pattern: buf_numel < iter_numel
            #
            # This handles cases where a smaller buffer broadcasts to the iteration domain,
            # common in padding operations (e.g., pad a 1-element tensor to 21 elements).
            # Instead of trying to reshape, we return a shape that JAX can broadcast.
            if current_numel < iter_numel:
                # Scalar broadcast: single element broadcasts to any shape
                if current_numel == 1:
                    if rnumel is not None and rnumel > 1:
                        return f"{expr}.reshape(1, 1)"  # Broadcast to (numel, rnumel)
                    else:
                        return f"{expr}.reshape(1)"  # Broadcast to (numel,)

                # Divisible broadcast: buf can tile to fill iter domain
                if iter_numel % current_numel == 0:
                    # Return without reshape, let JAX broadcast
                    return expr

            # NO FALLBACK - require explicit marker for other patterns
            dtype = V.graph.get_dtype(name)
            raise NotImplementedError(
                f"Buffer '{name}' after view chain has numel={current_numel} != iter_numel={iter_numel}, "
                f"but chain does not indicate this is an index buffer. dtype={dtype}. "
                f"Chain: {chain}. Add explicit marker (index_source, no_final_reshape, etc.) if intentional."
            )

        # Final reshape - use consistent shape logic
        # ROBUST: When rnumel=1, use 1D shape; when rnumel > 1, use 2D shape
        if rnumel > 1:
            shape_str = f"{numel}, {rnumel}" if self.inside_reduction else f"{numel}, 1"
        else:
            shape_str = str(numel)
        return f"{expr}.reshape({shape_str})"

    def load(self, name: str, index: sympy.Expr) -> CSEVariable:
        """
        Load buffer and reshape to compute shape.

        SIMPLIFIED CHAIN-FIRST ARCHITECTURE:
        - PATH 1: Indirect vars → _load_gather()
        - PATH 2: Has view_id chain → _build_load_from_view_chain()
        - PATH 3: No view_id, no PallasStrides → simple reshape (fresh buffer)
        - PATH 4: Has PallasStrides but no chain → ERROR

        Special cases preserved:
        - Post-reduction broadcast (buf_numel == numel, rnumel > 1): mathematically unambiguous
        - Unfold in chain: needs gather for overlapping windows

        DYNAMIC SHAPES: Uses rename_indexing + kexpr to convert symbolic sizes to
        kernel parameters, allowing runtime-determined shapes.
        """
        var = self.args.input(name)
        dtype = V.graph.get_dtype(name)

        # Get numel/rnumel info via centralized helper (with caching)
        info = self._get_numel_rnumel_info()
        numel_raw = info['numel_expr']
        rnumel_raw = info['rnumel_expr']
        numel_str = info['numel_str']
        rnumel_str = info['rnumel_str']
        numel_val = info['numel_val']
        rnumel_val = info['rnumel_val']

        # Get buffer shape and compute buf_numel
        buf = V.graph.get_buffer(name)
        buf_shape_raw = list(buf.get_size()) if buf else []
        buf_shape_vals = [self._safe_int(s) for s in buf_shape_raw]

        # Compute buf_numel and iter_numel as sympy expressions (for symbolic comparison)
        buf_numel_expr = sympy.Integer(1)
        for s in buf_shape_raw:
            buf_numel_expr = buf_numel_expr * s
        buf_numel_expr = V.graph.sizevars.simplify(buf_numel_expr)
        iter_numel_expr = V.graph.sizevars.simplify(numel_raw * rnumel_raw)

        # Compute concrete values if all dims are concrete
        has_symbolic = numel_val is None or rnumel_val is None or None in buf_shape_vals
        if has_symbolic:
            buf_numel = None
            iter_numel = None
        else:
            buf_numel = 1
            for s in buf_shape_vals:
                buf_numel *= s
            iter_numel = numel_val * rnumel_val

        # Generate shape string for reshape operations
        shape_str = self._get_shape_str_for_reshape()

        # Compute shape for result tracking (use raw values for internal tracking)
        if rnumel_val is None or rnumel_val > 1:
            compute_shape = (numel_val, rnumel_val) if self.inside_reduction else (numel_val, 1)
        else:
            compute_shape = (numel_val,)

        # ========== PATH 1: Indirect variables (gather pattern) ==========
        indirect_vars = self._get_indirect_vars(index)
        if indirect_vars:
            return self._load_gather(name, index, indirect_vars)

        # Extract view_id and PallasStrides from index
        view_id = self._extract_view_id_from_index(index)
        pallas_strides = list(index.atoms(PallasStride)) if hasattr(index, 'atoms') else []

        # ========== PATH 2a: Has view_id chain (view_id > 0) - use it exclusively ==========
        if view_id is not None and view_id > 0:
            chain = self._get_transform_chain_from_view_id(view_id, buf_shape_vals)
            if chain is None:
                raise NotImplementedError(
                    f"Buffer '{name}' has view_id={view_id} but no chain in registry. "
                    f"Ensure _store_pallas_view_chain() was called. "
                    f"buf_shape={buf_shape_vals}, index={index}"
                )

            # Check for gather - explicit gather operation from index lowering
            # This is the ROBUST way to detect gather: via explicit view chain, not inference
            gather_ops = [t for t in chain if t.get("op") == "gather"]
            if gather_ops:
                gather_info = gather_ops[0]
                # Resolve PallasStride markers using contiguous strides.
                # Buffers are made contiguous via .contiguous() before being passed
                # to Pallas kernels, so flat indices must use contiguous strides.
                resolved_index = self._resolve_index_with_contiguous_strides(index, name)
                # Use gather with the resolved index - use rename_indexing for symbolic parts
                resolved_index = self.rename_indexing(resolved_index)
                index_str = self.kexpr(resolved_index)
                load_expr = f"jnp.take({var}[...].reshape(-1), jnp.array({index_str}).astype(jnp.int32).reshape(-1), mode='wrap').reshape({shape_str})"
                result = self.cse.generate(self.compute, load_expr, dtype=dtype, shape=compute_shape)
                self.var_shapes[str(result)] = compute_shape
                return result

            # Check for unfold - needs gather for overlapping windows
            if any(t.get("op") == "unfold" for t in chain):
                return self._load_gather_with_index_expr(name, index)

            # Check for index_source - special handling for index tensors
            if any(t.get("op") == "index_source" for t in chain):
                return self._load_index_source(name, index)

            # Apply transforms from chain
            load_expr = self._build_load_from_view_chain(
                var, name, chain, index, numel_val, rnumel_val,
                numel_expr=numel_raw, rnumel_expr=rnumel_raw
            )
            result = self.cse.generate(self.compute, load_expr, dtype=dtype, shape=compute_shape)
            self.var_shapes[str(result)] = compute_shape
            return result

        # ========== PATH 2b: Direct buffer access (view_id == -1) ==========
        if view_id == -1 and pallas_strides:
            # Check if we need gather (buf_numel != iter_numel)
            # Use symbolic comparison when concrete values are not available
            simplified_diff = V.graph.sizevars.simplify(buf_numel_expr - iter_numel_expr)
            numels_match = simplified_diff == 0

            if not numels_match:
                # Gather case: buffer doesn't match iteration domain
                # Resolve PallasStride markers using contiguous strides.
                # Buffers are made contiguous via .contiguous() before being passed
                # to Pallas kernels, so flat indices must use contiguous strides.
                resolved_index = self._resolve_index_with_contiguous_strides(index, name)
                # Use gather with the resolved index - use rename_indexing for symbolic parts
                resolved_index = self.rename_indexing(resolved_index)
                index_str = self.kexpr(resolved_index)
                load_expr = f"jnp.take({var}[...].reshape(-1), jnp.array({index_str}).astype(jnp.int32).reshape(-1), mode='wrap').reshape({shape_str})"
                result = self.cse.generate(self.compute, load_expr, dtype=dtype, shape=compute_shape)
                self.var_shapes[str(result)] = compute_shape
                return result

            # For multi-dimensional buffers with same numel, check if reduction dims need to be moved to end
            perm = self._compute_permutation_from_pallas_stride(index, buf_shape_vals)
            if perm is not None:
                perm_str = ", ".join(str(p) for p in perm)
                load_expr = f"jnp.transpose({var}[...], ({perm_str},)).reshape({shape_str})"
                result = self.cse.generate(self.compute, load_expr, dtype=dtype, shape=compute_shape)
                self.var_shapes[str(result)] = compute_shape
                return result

            # No permutation needed - simple reshape
            load_expr = f"{var}[...].reshape({shape_str})"
            result = self.cse.generate(self.compute, load_expr, dtype=dtype, shape=compute_shape)
            self.var_shapes[str(result)] = compute_shape
            return result

        # ========== SPECIAL CASE: Post-reduction broadcast ==========
        # Only apply if we have concrete values to compare
        if (buf_numel is not None and iter_numel is not None and numel_val is not None
                and rnumel_val is not None and buf_numel == numel_val
                and rnumel_val > 1 and buf_numel < iter_numel):
            load_expr = f"{var}[...].reshape({numel_str}, 1)"
            broadcast_shape = (numel_val, 1)
            result = self.cse.generate(self.compute, load_expr, dtype=dtype, shape=broadcast_shape)
            self.var_shapes[str(result)] = broadcast_shape
            return result

        # ========== PATH 3: No view_id, no PallasStrides → fresh buffer or gather ==========
        if not pallas_strides:
            # Use symbolic comparison to determine if buffer matches iteration domain
            # This handles both concrete and symbolic shapes correctly
            simplified_diff = V.graph.sizevars.simplify(buf_numel_expr - iter_numel_expr)
            numels_match = simplified_diff == 0

            if numels_match:
                # Standard case: buffer matches iteration domain (concrete or symbolic)
                load_expr = f"{var}[...].reshape({shape_str})"
            else:
                # Gather/broadcast case: buffer doesn't match iteration domain
                index_renamed = self.rename_indexing(index)
                index_str = self.kexpr(index_renamed)

                # Determine if this is a scalar broadcast or true gather
                # Scalar broadcast: buf_numel == 1, so any index always accesses the same element
                # In this case, jnp.take returns 1 element that should broadcast, not reshape
                is_scalar_broadcast = (buf_numel is not None and buf_numel == 1) or (
                    buf_numel_expr == sympy.Integer(1)
                )

                if is_scalar_broadcast:
                    # Scalar buffer: jnp.take returns single element, let JAX broadcast
                    load_expr = f"jnp.take({var}[...].reshape(-1), jnp.array({index_str}).astype(jnp.int32).reshape(-1), mode='wrap')"
                else:
                    # True gather: index varies, take returns iter_numel elements
                    load_expr = f"jnp.take({var}[...].reshape(-1), jnp.array({index_str}).astype(jnp.int32).reshape(-1), mode='wrap').reshape({shape_str})"
            result = self.cse.generate(self.compute, load_expr, dtype=dtype, shape=compute_shape)
            self.var_shapes[str(result)] = compute_shape
            return result

        # ========== PATH 4: Has PallasStrides but no view_id → MISSING CHAIN ==========
        raise NotImplementedError(
            f"Buffer '{name}' has PallasStride markers in index but no view_id chain. "
            f"This indicates a view operation did not set up pallas_view_id. "
            f"buf_shape={buf_shape_vals}, buf_numel={buf_numel}, iter_numel={iter_numel}, "
            f"pallas_strides={[str(ps) for ps in pallas_strides]}"
        )

    def _infer_permutation_from_store_strides(
        self,
        index: sympy.Expr,
        output_shape: list[int],
    ) -> Optional[list[int]]:
        """
        Infer permutation from store index using iter_var_pos from PallasStride.

        ROBUST: Uses ONLY iter_var_pos for explicit position mapping.
        NO sort-by-stride fallback - if iter_var_pos is missing, returns None.

        Returns permutation if output needs transposition, None otherwise.
        """
        from ..ir import PallasStride

        # No permutation for scalar (0D) or 1D outputs
        if len(output_shape) <= 1:
            return None

        if not hasattr(index, 'free_symbols'):
            return None

        expanded = sympy.expand(index)
        terms = expanded.args if expanded.is_Add else [expanded]

        # ROBUST: Use ONLY iter_var_pos - NO sort by stride
        pos_to_dim: dict[int, int] = {}
        for term in terms:
            for atom in term.atoms(PallasStride):
                iter_pos = atom.iter_var_pos
                if iter_pos is not None:
                    buf_pos = int(iter_pos)
                    dim_idx = int(atom.dim_index)
                    pos_to_dim[buf_pos] = dim_idx

        # If not all positions are mapped, return None (caller handles it)
        if len(pos_to_dim) != len(output_shape):
            return None

        # Build permutation from iter_var_pos mapping
        perm = [pos_to_dim[i] for i in range(len(output_shape))]
        if perm == list(range(len(output_shape))):
            return None  # Identity permutation - no transpose needed
        return perm

    def store(
        self,
        name: str,
        index: sympy.Expr,
        value: CSEVariable,
        mode: Optional[str] = None,
    ) -> None:
        """
        Store 2D value to output buffer.

        Handles output permutations via view_id tracking or store index strides.
        DYNAMIC SHAPES: Uses rename_indexing + kexpr for symbolic dimensions.
        """
        var = self.args.output(name)

        # Get output buffer shape - may contain symbolic dimensions
        # IMPORTANT: For MutationLayout, get the underlying buffer's shape, not the view's shape
        buf = V.graph.get_buffer(name)
        output_shape_raw = list(buf.get_size()) if buf else []

        # Check if this is a MutationLayout - if so, use the underlying buffer's shape
        # This explicitly tracks the actual buffer shape rather than guessing at runtime
        from torch._inductor.ir import MutationLayoutSHOULDREMOVE
        if buf and hasattr(buf, 'layout') and isinstance(buf.layout, MutationLayoutSHOULDREMOVE):
            underlying_layout = buf.layout.real_layout()
            output_shape_raw = list(underlying_layout.size)

        output_shape_vals = [self._safe_int(s) for s in output_shape_raw]

        # Check for indirect variables (scatter pattern)
        indirect_vars = self._get_indirect_vars(index)
        if indirect_vars:
            return self._store_scatter(name, index, value, mode, indirect_vars)

        # Handle scalar (0D) output
        if len(output_shape_raw) == 0:
            value_expr = f"{value}.squeeze()"
            if mode == "atomic_add":
                store_line = f"{var}[...] = {var}[...] + {value_expr}"
            else:
                store_line = f"{var}[...] = {value_expr}"
            self.stores.append((var, store_line))
            return

        # Convert symbolic dimensions to kernel parameters
        output_shape_renamed = [self.rename_indexing(s) for s in output_shape_raw]
        shape_str = ", ".join(self.kexpr(s) for s in output_shape_renamed)

        # Compute output_numel if all dims are concrete
        if None in output_shape_vals:
            output_numel = None
        else:
            output_numel = 1
            for s in output_shape_vals:
                output_numel *= s

        # Check if output has transforms via view_id
        view_id = self._extract_view_id_from_index(index)

        # Helper to reshape or broadcast value to output shape
        def to_output_shape(val_expr: str) -> str:
            return self._to_output_shape_expr(value, output_shape_vals, output_numel, shape_str)

        if view_id is not None:
            chain = self._get_transform_chain_from_view_id(view_id, output_shape_vals)
            if chain is not None:
                value_expr = self._build_store_with_transforms(value, output_shape_vals, chain, shape_str)
            else:
                # view_id but no chain - check strides in index
                perm = self._infer_permutation_from_store_strides(index, output_shape_vals)
                if perm is not None:
                    value_expr = self._build_store_with_permutation(value, output_shape_vals, perm, shape_str)
                else:
                    value_expr = to_output_shape(str(value))
        else:
            # No view_id - reshape or broadcast to output_shape
            # The output buffer's strides (if non-contiguous) are handled by copy_(),
            # not by the kernel. The Pallas kernel writes data contiguously to the
            # output array, and the PyTorch wrapper's copy_() handles the strided write.
            value_expr = to_output_shape(str(value))

        if mode == "atomic_add":
            store_line = f"{var}[...] = {var}[...] + {value_expr}"
        else:
            store_line = f"{var}[...] = {value_expr}"

        self.stores.append((var, store_line))

    def _to_output_shape_expr(
        self,
        value: CSEVariable,
        output_shape: list[Optional[int]],
        output_numel: Optional[int],
        shape_str: str
    ) -> str:
        """
        Build expression to convert value to output shape.

        ROBUST: Requires explicit shape tracking. NO silent fallback.
        Uses tracked information to decide between broadcast (for scalars) and reshape.
        Checks both var_shapes dict and CSEVariable.shape attribute.

        DYNAMIC SHAPES: shape_str is the pre-computed string using kernel parameters.
        output_numel may be None for symbolic shapes.
        """
        val_str = str(value)

        # Check if value is tracked with a known shape
        val_shape = self.var_shapes.get(val_str, None)

        if val_shape is None and hasattr(value, 'shape') and value.shape is not None:
            # Convert CSEVariable.shape (tuple of ints or sympy) to tuple of ints
            val_shape = tuple(self._safe_int(s) for s in value.shape)

        if val_shape is None:
            # NOT TRACKED - this is an error, NOT a fallback
            raise NotImplementedError(
                f"Value '{val_str}' has no tracked shape in var_shapes or CSEVariable.shape. "
                f"Cannot determine if broadcast or reshape is needed for output_shape={output_shape}. "
                f"Ensure all CSE variables have their shapes tracked."
            )

        # Tracked - determine correct operation
        if val_shape == ():
            # Empty tuple indicates scalar constant - must broadcast
            return f"jnp.broadcast_to({value}, ({shape_str},))"

        # Compute val_numel (may be None if symbolic)
        if None in val_shape:
            val_numel = None
        else:
            val_numel = 1
            for s in val_shape:
                val_numel *= s

        # Handle scalar broadcast
        if val_numel is not None and val_numel == 1:
            if output_numel is None or output_numel > 1:
                return f"jnp.broadcast_to({value}, ({shape_str},))"

        # Check numel mismatch only if both are concrete
        if val_numel is not None and output_numel is not None and val_numel != output_numel:
            raise NotImplementedError(
                f"Value '{val_str}' has numel={val_numel} but output requires numel={output_numel}. "
                f"Cannot reshape {val_shape} to {output_shape}. Check computation."
            )

        # Same numel - reshape is valid
        return f"{value}.reshape({shape_str})"

    def _build_store_with_permutation(
        self,
        value: CSEVariable,
        output_shape: list[Optional[int]],
        perm: list[int],
        shape_str: str,
    ) -> str:
        """
        Build store expression for non-contiguous output.

        The kernel iterates in OUTPUT shape order, so the value is already
        in the correct logical order. We just need to reshape to output_shape.
        The copy_() in the wrapper handles writing to strided buffers.

        NOTE: We do NOT apply permutation here because the kernel already
        iterates in the permuted (output) shape order.
        """
        # Compute output_numel if possible
        if None in output_shape:
            output_numel = None
        else:
            output_numel = 1
            for s in output_shape:
                output_numel *= s
        return self._to_output_shape_expr(value, output_shape, output_numel, shape_str)

    def _build_store_with_transforms(
        self,
        value: CSEVariable,
        output_shape: list[Optional[int]],
        chain: list[dict],
        shape_str: str,
    ) -> str:
        """
        Build store expression for output with transforms.

        The kernel iterates in the OUTPUT shape order (already permuted if applicable).
        The value is in flattened iteration order, so we just need to reshape to
        output_shape. The copy_() in the wrapper handles writing to strided buffers.

        NOTE: We do NOT apply permute transforms from the chain here because:
        1. The chain's permute describes how the INPUT was transformed
        2. The kernel already iterates in the permuted (output) shape
        3. copy_() handles the strided output buffer layout
        """
        # Compute output_numel if possible
        if None in output_shape:
            output_numel = None
        else:
            output_numel = 1
            for s in output_shape:
                output_numel *= s
        return self._to_output_shape_expr(value, output_shape, output_numel, shape_str)

    def reduction(
        self,
        dtype: torch.dtype,
        src_dtype: torch.dtype,
        reduction_type: ReductionType,
        value: Union[CSEVariable, tuple[CSEVariable, ...]],
    ) -> Union[CSEVariable, tuple[CSEVariable, ...]]:
        """Reduction along last axis (rnumel dimension)."""
        assert self.inside_reduction

        ops = {
            "sum": "jnp.sum", "prod": "jnp.prod",
            "max": "jnp.max", "min": "jnp.min",
            "any": "jnp.any", "argmax": "jnp.argmax", "argmin": "jnp.argmin",
        }

        numel = self._get_numel()
        reduction_shape = (numel, 1)

        # Handle welford_reduce specially - returns (mean, m2, weight)
        if reduction_type == "welford_reduce":
            # Get rnumel for computing mean
            rnumel = self._get_rnumel()

            # Compute mean = sum(value) / rnumel
            sum_expr = f"jnp.sum({value}, axis=-1, keepdims=True)"
            sum_var = self.cse.generate(
                self.compute, sum_expr, dtype=dtype, shape=reduction_shape
            )
            self.var_shapes[str(sum_var)] = reduction_shape

            mean_expr = f"({sum_var}) / {rnumel}"
            mean_var = self.cse.generate(
                self.compute, mean_expr, dtype=dtype, shape=reduction_shape
            )
            self.var_shapes[str(mean_var)] = reduction_shape

            # Compute m2 = sum((value - mean)^2)
            # Need to broadcast mean back to full shape for subtraction
            diff_expr = f"({value}) - ({mean_var})"
            diff_var = self.cse.generate(self.compute, diff_expr, dtype=dtype)

            sq_expr = f"({diff_var}) * ({diff_var})"
            sq_var = self.cse.generate(self.compute, sq_expr, dtype=dtype)

            m2_expr = f"jnp.sum({sq_var}, axis=-1, keepdims=True)"
            m2_var = self.cse.generate(
                self.compute, m2_expr, dtype=dtype, shape=reduction_shape
            )
            self.var_shapes[str(m2_var)] = reduction_shape

            # Weight is just rnumel broadcast to reduction shape
            weight_expr = f"jnp.full(({numel}, 1), {rnumel}, dtype=jnp.float32)"
            weight_var = self.cse.generate(
                self.compute, weight_expr, dtype=torch.float32, shape=reduction_shape
            )
            self.var_shapes[str(weight_var)] = reduction_shape

            return (mean_var, m2_var, weight_var)

        # Handle welford_combine - combines partial welford results
        if reduction_type == "welford_combine":
            assert isinstance(value, tuple) and len(value) == 3
            mean, m2, weight = value

            # For welford_combine, we need to combine partial results
            # combined_weight = sum(weight)
            # combined_mean = sum(mean * weight) / combined_weight
            # combined_m2 = sum(m2) + sum(weight * (mean - combined_mean)^2)

            # Sum weights
            weight_sum_expr = f"jnp.sum({weight}, axis=-1, keepdims=True)"
            weight_sum = self.cse.generate(
                self.compute, weight_sum_expr, dtype=torch.float32, shape=reduction_shape
            )
            self.var_shapes[str(weight_sum)] = reduction_shape

            # Weighted mean
            weighted_mean_expr = f"jnp.sum(({mean}) * ({weight}), axis=-1, keepdims=True) / ({weight_sum})"
            combined_mean = self.cse.generate(
                self.compute, weighted_mean_expr, dtype=dtype, shape=reduction_shape
            )
            self.var_shapes[str(combined_mean)] = reduction_shape

            # Combined m2
            delta_expr = f"({mean}) - ({combined_mean})"
            delta_var = self.cse.generate(self.compute, delta_expr, dtype=dtype)

            m2_sum_expr = f"jnp.sum({m2}, axis=-1, keepdims=True)"
            m2_sum = self.cse.generate(
                self.compute, m2_sum_expr, dtype=dtype, shape=reduction_shape
            )

            delta_sq_expr = f"jnp.sum(({weight}) * ({delta_var}) * ({delta_var}), axis=-1, keepdims=True)"
            delta_sq = self.cse.generate(
                self.compute, delta_sq_expr, dtype=dtype, shape=reduction_shape
            )

            combined_m2_expr = f"({m2_sum}) + ({delta_sq})"
            combined_m2 = self.cse.generate(
                self.compute, combined_m2_expr, dtype=dtype, shape=reduction_shape
            )
            self.var_shapes[str(combined_m2)] = reduction_shape

            return (combined_mean, combined_m2, weight_sum)

        op = ops.get(str(reduction_type))
        if op is None:
            raise RuntimeError(f"Unsupported reduction: {reduction_type}")

        if reduction_type in ("argmax", "argmin"):
            expr = f"{op}({value}, axis=-1).reshape({numel}, 1)"
        else:
            expr = f"{op}({value}, axis=-1, keepdims=True)"

        # Reduction along last axis produces (numel, 1)
        result = self.cse.generate(self.compute, expr, dtype=dtype, shape=reduction_shape)
        # Track shape in var_shapes dict as well for compatibility
        self.var_shapes[str(result)] = reduction_shape
        return result

    def codegen_kernel(self, name: Optional[str] = None) -> str:
        """Generate Pallas kernel code.

        For dynamic shapes: We use rename_indexing to convert symbolic sizes
        (like s14) to kernel parameters (like ks0). This allows the kernel to
        work with runtime-determined sizes.
        """
        kernel_name = name or "<KERNEL_NAME>"

        # Initial arg lookup (will be updated after body generation for size vars)
        arg_defs, call_args, _, _ = self.args.python_argdefs()
        params = [a.name for a in arg_defs]
        inputs = [p for p in params if p.startswith("in_ptr")]
        outputs = [p for p in params if p.startswith("out_ptr")]
        in_outs = [p for p in params if p.startswith("in_out_ptr")]
        all_inputs = in_outs + inputs

        # For scatter outputs, add them as inputs so pallas_call can alias them
        # Note: kernel_input_params order is: scatter_alias_params + in_outs + inputs
        # So scatter aliases are at indices 0, 1, 2, ... in the kernel input order
        input_output_aliases: dict[int, int] = {}
        scatter_alias_params: list[str] = []
        for i, out in enumerate(outputs):
            if out in self.scatter_outputs:
                # Alias index is position in scatter_alias_params (which come first in kernel_input_params)
                alias_idx = len(scatter_alias_params)
                alias_param = f"alias_in_{out}"
                all_inputs.append(out)
                scatter_alias_params.append(alias_param)
                input_output_aliases[alias_idx] = i

        # ============================================================
        # Generate kernel body FIRST - this discovers size variables
        # via rename_indexing calls for symbolic lengths
        # ============================================================
        kernel_body = IndentedBuffer()
        with kernel_body.indent():
            if self.range_tree_nodes:
                x_vars = []
                r_vars = []

                for tree in self.range_trees:
                    for var in tree.var_list:
                        if var in self.range_tree_nodes:
                            entry = self.range_tree_nodes[var]
                            if tree.prefix.startswith("r"):
                                r_vars.append((var, entry))
                            else:
                                x_vars.append((var, entry))

                # Get numel/rnumel info via centralized helper (with caching)
                info = self._get_numel_rnumel_info()
                numel_str = info['numel_str']
                rnumel_str = info['rnumel_str']
                rnumel_val = info['rnumel_val']
                has_reduction = rnumel_val is None or rnumel_val > 1

                if has_reduction:
                    # Has reduction - use (numel, rnumel) shape
                    for var, entry in x_vars:
                        var_name = str(var)
                        divisor = self._iteration_divisors.get(var_name, sympy.Integer(1))
                        length_renamed = self.rename_indexing(entry.length)
                        length_str = self.kexpr(length_renamed)
                        # Check if divisor is 1 (handles both int and sympy.Integer)
                        is_divisor_one = (isinstance(divisor, (int, sympy.Integer)) and int(divisor) == 1)
                        if is_divisor_one:
                            kernel_body.writeline(f"{var_name} = (jnp.arange({numel_str}) % ({length_str})).reshape({numel_str}, 1)")
                        else:
                            divisor_renamed = self.rename_indexing(divisor)
                            divisor_str = self.kexpr(divisor_renamed)
                            kernel_body.writeline(f"{var_name} = ((jnp.arange({numel_str}) // ({divisor_str})) % ({length_str})).reshape({numel_str}, 1)")

                    for var, entry in r_vars:
                        var_name = str(var)
                        divisor = self._iteration_divisors.get(var_name, sympy.Integer(1))
                        length_renamed = self.rename_indexing(entry.length)
                        length_str = self.kexpr(length_renamed)
                        is_divisor_one = (isinstance(divisor, (int, sympy.Integer)) and int(divisor) == 1)
                        if is_divisor_one:
                            kernel_body.writeline(f"{var_name} = (jnp.arange({rnumel_str}) % ({length_str})).reshape(1, {rnumel_str})")
                        else:
                            divisor_renamed = self.rename_indexing(divisor)
                            divisor_str = self.kexpr(divisor_renamed)
                            kernel_body.writeline(f"{var_name} = ((jnp.arange({rnumel_str}) // ({divisor_str})) % ({length_str})).reshape(1, {rnumel_str})")
                else:
                    # No reduction - use 1D (numel,) shape
                    for var, entry in x_vars:
                        var_name = str(var)
                        length_renamed = self.rename_indexing(entry.length)
                        length_str = self.kexpr(length_renamed)
                        if len(x_vars) == 1:
                            kernel_body.writeline(f"{var_name} = jnp.arange({length_str})")
                        else:
                            divisor = self._iteration_divisors.get(var_name, sympy.Integer(1))
                            is_divisor_one = (isinstance(divisor, (int, sympy.Integer)) and int(divisor) == 1)
                            if is_divisor_one:
                                kernel_body.writeline(f"{var_name} = jnp.arange({numel_str}) % ({length_str})")
                            else:
                                divisor_renamed = self.rename_indexing(divisor)
                                divisor_str = self.kexpr(divisor_renamed)
                                kernel_body.writeline(f"{var_name} = (jnp.arange({numel_str}) // ({divisor_str})) % ({length_str})")

            # Add compute lines
            for line in self.compute._lines:
                if isinstance(line, str):
                    kernel_body.writeline(line.lstrip())

        # ============================================================
        # Re-fetch args after body generation to capture size variables
        # ============================================================
        arg_defs, call_args, _, _ = self.args.python_argdefs()
        params = [a.name for a in arg_defs]
        size_var_names = OrderedSet(self.args.sizevars.values())
        size_var_params = [p for p in params if p in size_var_names]

        # Rebuild inputs/outputs lists from updated params
        inputs = [p for p in params if p.startswith("in_ptr")]
        outputs = [p for p in params if p.startswith("out_ptr")]
        in_outs = [p for p in params if p.startswith("in_out_ptr")]

        # Kernel input params for pallas_call
        kernel_input_params = scatter_alias_params + in_outs + inputs

        # ============================================================
        # Now generate the final code with correct signatures
        # ============================================================
        code = IndentedBuffer()

        # Imports
        code.writeline("import functools")
        code.writeline("import math")
        code.writeline("import torch")
        code.writeline("import jax")
        code.writeline("import jax.numpy as jnp")
        code.writeline("from jax.experimental import pallas as pl")
        code.writeline("from torch._inductor.runtime.runtime_utils import torch_dtype_to_jax_runtime")
        code.writeline("")

        # Kernel function signature:
        # 1. size_var_params (bound FIRST by functools.partial using positional binding)
        # 2. scatter_alias_params (aliases for in/out buffers, passed to pallas_call)
        # 3. in_outs + inputs (pointer_tail in pallas_call)
        # 4. outputs (created by pallas_call based on out_specs)
        # With positional partial binding, after size_vars are bound, remaining params are:
        # scatter_alias_params + in_outs + inputs + outputs
        # pallas_call passes: scatter_alias_params + in_outs + inputs, then adds outputs
        kernel_params = size_var_params + scatter_alias_params + in_outs + inputs + outputs
        code.writeline(f"def {kernel_name}_kernel({', '.join(kernel_params)}):")
        with code.indent():
            for line in kernel_body._lines:
                if isinstance(line, str):
                    code.writeline(line.lstrip())
                else:
                    code._lines.append(line)
            for var, store_line in self.stores:
                if var in params:
                    code.writeline(store_line)
        code.writeline("")

        # JIT wrapper - size vars are static args
        static_argnums = list(range(2 + len(size_var_params)))
        static_argnums_literal = "(" + ", ".join(str(x) for x in static_argnums) + ",)"
        wrapper_params = ["out_shapes", "out_dtypes"] + size_var_params + kernel_input_params
        code.writeline(f"@functools.partial(jax.jit, static_argnums={static_argnums_literal}, backend='cpu')")
        code.writeline(f"def {kernel_name}_jit_wrapper({', '.join(wrapper_params)}):")
        with code.indent():
            code.writeline("out_specs = tuple(jax.ShapeDtypeStruct(s, d) for s, d in zip(out_shapes, out_dtypes))")
            aliases_str = "{" + ", ".join(f"{k}: {v}" for k, v in input_output_aliases.items()) + "}"

            # Use functools.partial with POSITIONAL binding to pass size variables
            # This works because size_var_params are first in kernel signature
            # After partial binds them, pallas_call passes remaining args positionally
            if size_var_params:
                partial_args = ", ".join(size_var_params)  # Positional, not keyword
                kernel_ref = f"functools.partial({kernel_name}_kernel, {partial_args})"
            else:
                kernel_ref = f"{kernel_name}_kernel"

            code.writeline(f"return pl.pallas_call({kernel_ref}, out_shape=out_specs, interpret=True, grid=(1,), input_output_aliases={aliases_str})({', '.join(kernel_input_params)})")
        code.writeline("")

        # Main entry point
        code.writeline(f"def {kernel_name}_main({', '.join(params)}, stream=None):")
        with code.indent():
            code.writeline("jax.config.update('jax_enable_x64', True)")
            code.writeline("jax.clear_caches()")

            # Rebuild all_inputs for main
            all_inputs_main = in_outs + inputs
            for out in outputs:
                if out in self.scatter_outputs:
                    all_inputs_main.append(out)

            for p in all_inputs_main:
                code.writeline(f"{p}_jax = jax.device_put(jax.dlpack.from_dlpack({p}.detach().contiguous()), device=jax.devices('cpu')[0])")

            # Create _jax aliases for scatter_alias_params (they reference their original output buffers)
            for alias_param in scatter_alias_params:
                # alias_param is like "alias_in_out_ptr0", extract original output name "out_ptr0"
                original_out = alias_param.replace("alias_in_", "")
                code.writeline(f"{alias_param}_jax = {original_out}_jax")
            code.writeline(f"out_shapes = ({', '.join(f'tuple({p}.shape)' for p in outputs)},)")
            code.writeline(f"out_dtypes = ({', '.join(f'torch_dtype_to_jax_runtime({p}.dtype)' for p in outputs)},)")

            # Build call args: out_shapes, out_dtypes, size_vars, inputs
            call_arg_list = ["out_shapes", "out_dtypes"] + size_var_params + [f"{p}_jax" for p in kernel_input_params]
            code.writeline(f"res = {kernel_name}_jit_wrapper({', '.join(call_arg_list)})")
            code.writeline("result_values = res if isinstance(res, tuple) else (res,)")
            for i, p in enumerate(outputs):
                code.writeline(f"{p}.copy_(torch.from_dlpack(result_values[{i}]))")

        return code.getvalue()

    def call_kernel(self, name: str, node: Optional[IRNode] = None, deallocate_ws: bool = True) -> None:
        """Generate kernel call."""
        wrapper = V.graph.wrapper_code
        arg_defs, call_args, _, _ = self.args.python_argdefs()
        wrapper.writeline(f"{name}.run({', '.join(map(str, call_args))})")


class PallasScheduling(SIMDScheduling):
    """Scheduling for Pallas backend."""

    kernel_type = PallasKernel  # type: ignore[assignment]

    @classmethod
    def get_backend_features(cls, device: torch.device) -> OrderedSet[BackendFeature]:
        return OrderedSet([BackendFeature.REDUCE_TO_SINGLE_ELEMENT])

    def define_kernel(
        self,
        src_code: str,
        node_schedule: Sequence[BaseSchedulerNode],
        kernel: PallasKernel,
    ) -> str:
        wrapper = V.graph.wrapper_code
        if src_code in wrapper.src_to_kernel:
            return wrapper.src_to_kernel[src_code]

        fused_name = get_fused_kernel_name(node_schedule, config.triton.descriptive_names) if config.triton.descriptive_names else ""
        kernel_hash = hashlib.sha256(src_code.encode()).hexdigest()[:8]
        kernel_name = f"pallas_{fused_name}_{kernel_hash}" if fused_name != "fused" else f"pallas_{kernel_hash}"

        wrapper.src_to_kernel[src_code] = kernel_name
        src_code = src_code.replace("<KERNEL_NAME>", kernel_name)

        compile_wrapper = IndentedBuffer()
        compile_wrapper.writeline(f"async_compile.pallas({kernel_name!r}, r'''")
        compile_wrapper.splice(src_code, strip=True)
        compile_wrapper.writeline("''')")

        origins, detailed_origins = get_kernel_metadata(node_schedule, wrapper)
        wrapper.define_kernel(kernel_name, compile_wrapper.getvalue(), f"{origins}\n{detailed_origins}")

        return kernel_name


# Suffix for the main entry point function
MAIN_SUFFIX = "main"


class PallasKernelWrapper:
    """Wrapper for compiled Pallas kernels."""

    def __init__(self, main_func, kernel_path=None):
        self.main_func = main_func
        self.kernel_path = kernel_path

    def run(self, *args, **kwargs):
        """Execute the Pallas kernel."""
        return self.main_func(*args, **kwargs)
