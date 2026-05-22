from __future__ import annotations

import contextlib
import hashlib
import math
from typing import Any, TYPE_CHECKING

import torch
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.printers import PythonPrinter as _BasePrinter

from .. import config
from ..utils import get_fused_kernel_name, get_kernel_metadata, IndentedBuffer
from ..virtualized import V
from .common import BackendFeature, CSEVariable, DeferredLine, OpOverrides, RemovedArg
from .simd import SIMDKernel, SIMDScheduling

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import sympy

    from ..ir import IRNode
    from ..scheduler import BaseSchedulerNode
    from .simd_kernel_features import SIMDKernelFeatures


def _is_reduction_prefix(prefix: str) -> bool:
    from .simd import prefix_is_reduction

    return prefix_is_reduction(prefix)


class HelionKernelWrapper:
    def __init__(self, kernel_fn: Callable[..., Any]):
        self.kernel_fn = kernel_fn

    def run(self, *args: Any, **kwargs: Any) -> Any:
        return self.kernel_fn(*args)


# Simple unary ops: method_name maps to torch.{method_name}(x)
_UNARY_TORCH_OPS = [
    "sin", "cos", "tan", "sinh", "cosh", "tanh",
    "asin", "acos", "atan",
    "exp", "exp2", "expm1",
    "log", "log2", "log10", "log1p",
    "sqrt", "rsqrt", "abs",
    "sigmoid", "relu",
    "floor", "ceil", "trunc", "round", "sign",
    "reciprocal",
    "erf", "erfc", "erfinv", "lgamma",
    "isnan", "isinf", "isfinite",
    "signbit", "logical_not", "digamma",
]  # noqa: B950

# Simple binary ops: method_name maps to torch.{method_name}(a, b)
_BINARY_TORCH_OPS = [
    "maximum", "minimum", "pow", "atan2", "fmod", "remainder",
    "logical_and", "logical_or", "nextafter",
]

# Unary torch.special ops: method_name maps to torch.special.{method_name}(x)
_SPECIAL_UNARY_OPS = [
    "i0", "i0e", "i1", "i1e",
    "ndtr", "ndtri", "log_ndtr", "erfcx",
    "bessel_j0", "bessel_j1", "bessel_y0", "bessel_y1",
    "modified_bessel_i0", "modified_bessel_i1",
    "modified_bessel_k0", "modified_bessel_k1",
    "scaled_modified_bessel_k0", "scaled_modified_bessel_k1",
    "spherical_bessel_j0", "airy_ai",
]

# Binary torch.special ops: method_name maps to torch.special.{method_name}(a, b)
_SPECIAL_BINARY_OPS = [
    "gammainc", "gammaincc", "polygamma", "zeta",
    "chebyshev_polynomial_t", "chebyshev_polynomial_u",
    "chebyshev_polynomial_v", "chebyshev_polynomial_w",
    "legendre_polynomial_p",
    "shifted_chebyshev_polynomial_t", "shifted_chebyshev_polynomial_u",
    "shifted_chebyshev_polynomial_v", "shifted_chebyshev_polynomial_w",
    "hermite_polynomial_h", "hermite_polynomial_he",
    "laguerre_polynomial_l",
]

# Renamed ops: method_name -> torch_function_name (all under torch.special)
_RENAMED_BINARY_OPS = {
    "igamma": "gammainc",
    "igammac": "gammaincc",
}


class HelionKernelOverrides(OpOverrides):
    @staticmethod
    def where(cond, a, b):
        return f"torch.where({cond}, {a}, {b})"

    @staticmethod
    def masked(mask, body, other):
        result = body()
        if isinstance(other, float):
            if math.isnan(other):
                other_str = "float('nan')"
            elif math.isinf(other):
                other_str = "float('inf')" if other > 0 else "float('-inf')"
            else:
                other_str = repr(other)
        else:
            other_str = repr(other)
        return f"torch.where({mask}, {result}, {other_str})"

    @staticmethod
    def to_dtype(x, dtype, src_dtype=None, use_compute_types=True):
        if use_compute_types and dtype in (torch.float16, torch.bfloat16):
            dtype = torch.float32
        return f"({x}).to({_torch_dtype_str(dtype)})"

    @staticmethod
    def to_dtype_bitcast(x, dtype, src_dtype):
        return f"({x}).view({_torch_dtype_str(dtype)})"

    @staticmethod
    def constant(value, dtype):
        # Emit as 0-D scalar_tensor; store() expands to the output tile shape
        # via _is_scalar_constant when needed. Provenance for the resulting
        # CSE variable is recorded by HelionCSEVariable.update_on_args.
        if dtype in (torch.float16, torch.bfloat16):
            dtype = torch.float32
        dtype_str = _torch_dtype_str(dtype)
        if isinstance(value, float):
            if math.isnan(value):
                val_str = "float('nan')"
            elif math.isinf(value):
                val_str = "float('inf')" if value > 0 else "float('-inf')"
            else:
                val_str = repr(value)
        elif isinstance(value, (bool, int)):
            val_str = repr(value)
        else:
            return repr(value)
        return f"torch.scalar_tensor({val_str}, dtype={dtype_str})"

    @staticmethod
    def index_expr(expr, dtype):
        if expr.is_number:
            return repr(int(expr))
        # Try to evaluate to a concrete value using known shape bindings
        try:
            val = int(expr)
            return repr(val)
        except (TypeError, ValueError):
            pass
        return V.kernel.helion_index_expr(expr)

    @staticmethod
    def frexp(x):
        return (f"torch.frexp({x}).mantissa", f"torch.frexp({x}).exponent")

    @staticmethod
    def trunc_to_int(x, dtype):
        return f"({x}).to({_torch_dtype_str(dtype)})"

    @staticmethod
    def ceil_to_int(x, dtype):
        return f"torch.ceil({x}).to({_torch_dtype_str(dtype)})"

    @staticmethod
    def floor_to_int(x, dtype):
        return f"torch.floor({x}).to({_torch_dtype_str(dtype)})"

    @staticmethod
    def round_to_int(x, dtype):
        return f"torch.round({x}).to({_torch_dtype_str(dtype)})"

    @staticmethod
    def truncdiv(a, b):
        return f"torch.div({a}, {b}, rounding_mode='trunc')"

    @staticmethod
    def floordiv(a, b):
        return f"torch.div({a}, {b}, rounding_mode='floor')"

    @staticmethod
    def fma(a, b, c):
        return f"({a} * {b} + {c})"

    @staticmethod
    def mul_rn(a, b):
        return f"({a} * {b})"

    # Random number generation
    @staticmethod
    def load_seed(name, offset):
        var = V.kernel.args.input(name)
        return f"{var}[0]"

    @staticmethod
    def rand(seed, offset):
        shape_str = V.kernel._rand_shape_str()
        return f"hl.rand([{shape_str}], seed={seed})"

    @staticmethod
    def randn(seed, offset):
        shape_str = V.kernel._rand_shape_str()
        return (
            f"(torch.erfinv(2 * hl.rand([{shape_str}], seed={seed}) - 1)"
            f" * 1.4142135623730951)"
        )

    @staticmethod
    def randint64(seed, offset, low, high):
        kernel = V.kernel
        shape_str = kernel._rand_shape_str()
        low_int = _constant_int_str(low, kernel)
        high_int = _constant_int_str(high, kernel)
        return f"hl.randint([{shape_str}], {low_int}, {high_int}, seed={seed}).to(torch.int64)"


# Populate simple overrides from the tables above.
def _install_torch_overrides() -> None:
    def make_unary(prefix: str, torch_name: str):
        def fn(x):
            return f"{prefix}.{torch_name}({x})"
        return staticmethod(fn)

    def make_binary(prefix: str, torch_name: str):
        def fn(a, b):
            return f"{prefix}.{torch_name}({a}, {b})"
        return staticmethod(fn)

    for op in _UNARY_TORCH_OPS:
        setattr(HelionKernelOverrides, op, make_unary("torch", op))
    for op in _BINARY_TORCH_OPS:
        setattr(HelionKernelOverrides, op, make_binary("torch", op))
    for op in _SPECIAL_UNARY_OPS:
        setattr(HelionKernelOverrides, op, make_unary("torch.special", op))
    for op in _SPECIAL_BINARY_OPS:
        setattr(HelionKernelOverrides, op, make_binary("torch.special", op))
    for method, torch_name in _RENAMED_BINARY_OPS.items():
        setattr(HelionKernelOverrides, method, make_binary("torch.special", torch_name))


_install_torch_overrides()


def _constant_int_str(var: Any, kernel: Any) -> str:
    """Return the integer literal that produced a CSE variable.

    Uses ``HelionCSEVariable.constant_value`` populated by ``update_on_args``;
    falls back to ``str(var)`` for non-Helion variables (e.g. when the value
    came from indirect computation rather than a literal).
    """
    cse_var = kernel.cse.varname_map.get(str(var))
    if isinstance(cse_var, HelionCSEVariable) and cse_var.constant_value is not _NOT_CONSTANT:
        return repr(int(cse_var.constant_value))
    return str(var)


def _torch_dtype_str(dtype: torch.dtype) -> str:
    s = str(dtype)
    return s if s.startswith("torch.") else f"torch.{s}"


HELION_KERNEL_NAME = "_inductor_helion_kernel"

REDUCTION_TYPE_MAP = {
    "sum": "torch.sum",
    "max": "torch.amax",
    "amax": "torch.amax",
    "min": "torch.amin",
    "amin": "torch.amin",
    "prod": "torch.prod",
    "any": "torch.any",
    "argmax": "torch.argmax",
    "argmin": "torch.argmin",
}

ATOMIC_OP_FOR_REDUCTION = {
    "sum": "hl.atomic_add",
    "amax": "hl.atomic_max",
    "max": "hl.atomic_max",
    "amin": "hl.atomic_min",
    "min": "hl.atomic_min",
}

INIT_FOR_REDUCTION = {
    "sum": ".zero_()",
    "amax": ".fill_(float('-inf'))",
    "max": ".fill_(float('-inf'))",
    "amin": ".fill_(float('inf'))",
    "min": ".fill_(float('inf'))",
}


class _HelionPrinter(_BasePrinter):
    # Tile index expressions are tensors in Helion: emit torch.minimum/maximum
    # for the all-tensor case, torch.clamp for the mixed scalar-bound case, and
    # the Python builtins only when every operand is a literal. torch.round
    # replaces the builtin so RoundToInt yields tensor-friendly code.

    def _print_Min(self, expr):  # type: ignore[override]
        return self._helion_minmax(expr, is_min=True)

    def _print_Max(self, expr):  # type: ignore[override]
        return self._helion_minmax(expr, is_min=False)

    def _helion_minmax(self, expr, *, is_min: bool) -> str:
        import sympy as _sp

        const_args = [a for a in expr.args if a.is_number]
        other_args = [a for a in expr.args if not a.is_number]
        if other_args and const_args:
            if len(other_args) == 1:
                tensor_str = self._print(other_args[0])
            else:
                sub = _sp.Min(*other_args) if is_min else _sp.Max(*other_args)
                tensor_str = self._print(sub)
            bound = int(min(const_args) if is_min else max(const_args))
            kw = "max" if is_min else "min"
            return f"torch.clamp({tensor_str}, {kw}={bound})"
        if not other_args:
            args_str = ", ".join(self._print(a) for a in expr.args)
            return f"{'min' if is_min else 'max'}({args_str})"
        fn = "torch.minimum" if is_min else "torch.maximum"
        result = self._print(other_args[0])
        for arg in other_args[1:]:
            result = f"{fn}({result}, {self._print(arg)})"
        return result

    def _print_RoundToInt(self, expr):  # type: ignore[override]
        return f"torch.round({self._print(expr.args[0])})"


_helion_expr_printer = _HelionPrinter()


# Sentinel value used by HelionCSEVariable to mark vars that did NOT come from
# an ops.constant call (distinct from a constant whose value happens to be None).
_NOT_CONSTANT = object()


class HelionCSEVariable(CSEVariable):
    """CSEVariable that records provenance needed by the Helion backend.

    ``constant_value`` is set by ``update_on_args`` when the variable was
    produced by ``ops.constant`` (used by ``randint64`` for literal ints and by
    ``store()`` to expand to the tile shape). ``reduction_type`` is set
    manually in ``HelionKernel.reduction`` since reductions bypass CSEProxy.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.constant_value: Any = _NOT_CONSTANT
        self.reduction_type: str | None = None

    def update_on_args(self, name: str, args: Any, kwargs: Any) -> None:  # type: ignore[override]
        if name == "constant" and args:
            self.constant_value = args[0]


class HelionKernel(SIMDKernel):
    overrides = HelionKernelOverrides  # type: ignore[assignment]
    _REDUCTION_SLICE = ":"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.store_lines: list[DeferredLine] = []
        self._output_ndim: int = 0
        self._output_size: list[int] | None = None
        # Reduction type tracked per output buffer (graph name) so kernels with
        # multiple reductions (e.g. amax+sum) emit the correct init per output.
        self._reduction_type_per_out: dict[str, str] = {}
        self._scalar_as_1d: bool = False
        # Flat-multidim mode: tiling is 1-D over numel but the buffers are
        # multi-dim. The kernel reshapes each buffer to ``as_strided((n,),
        # (1,))`` so the load/store paths can use 1-D ``buf[tile]`` indexing.
        self._flat_multidim: bool = False
        # Per-buffer flat aliases needed by the N-D-tiling flat-decomp load
        # path; emitted as ``buf_flat = buf.as_strided(...)`` BEFORE the tile
        # loop in ``codegen_kernel``.
        self._flat_aliases: dict[str, str] = {}
        self._helion_backend: str = _get_helion_backend()

    def create_cse_var(self, *args: Any, **kwargs: Any) -> CSEVariable:
        return HelionCSEVariable(*args, **kwargs)

    def _has_reduction(self) -> bool:
        return any(
            tree.is_reduction and tree.numel != 1
            for tree in self.range_trees
        )

    def _first_output_buffer_name(self) -> str | None:
        # Returns the outer (graph) name; inplace_buffers may contain stale
        # RemovedArg entries which must be skipped.
        for bname, val in self.args.output_buffers.items():
            if not isinstance(val, RemovedArg):
                return bname
        for bname, val in self.args.inplace_buffers.items():
            if not isinstance(val, RemovedArg):
                return bname
        return None

    def _output_size_and_stride(self) -> tuple[list[int], list[int]] | None:
        """Get concrete output buffer size and stride, or None if unavailable."""
        hint = V.graph.sizevars.optimization_hint
        out_buf_name = self._first_output_buffer_name()
        if out_buf_name is None:
            return None
        buf = V.graph.try_get_buffer(out_buf_name)
        if buf is None:
            return None
        out_size = [hint(s, fallback=0) for s in buf.get_size()]
        out_stride = [hint(s, fallback=0) for s in buf.get_stride()]
        if len(out_size) != self._output_ndim:
            return None
        return out_size, out_stride

    def _find_buf_dim_alignment(self, name: str, buf_ndim: int | None = None) -> int | None:
        """Find the output dimension where a lower-rank buffer aligns.

        Returns the starting index in the output dimensions where the buffer's
        shape matches. Prefers trailing alignment (standard broadcast semantics).
        Returns None if no alignment is found.
        """
        out_ndim = self._output_ndim
        hint = V.graph.sizevars.optimization_hint
        if buf_ndim is None:
            buf_ndim = self._get_buffer_ndim(name)
        if buf_ndim >= out_ndim or buf_ndim == 0:
            return None
        out_size = self._output_size
        if out_size is None:
            result = self._output_size_and_stride()
            if result is None:
                return out_ndim - buf_ndim
            out_size, _ = result
        buf = V.graph.try_get_buffer(name)
        if buf is None:
            gi = V.graph.graph_inputs.get(name)
            if gi is None:
                return out_ndim - buf_ndim
            buf_size = [hint(s, fallback=0) for s in gi.get_size()]
        else:
            buf_size = [hint(s, fallback=0) for s in buf.get_size()]
        # Check trailing alignment first (standard broadcast semantics)
        trailing = out_ndim - buf_ndim
        if buf_size == out_size[trailing:]:
            return trailing
        # Then search for other alignments
        for start in range(out_ndim - buf_ndim + 1):
            if buf_size == out_size[start:start + buf_ndim]:
                return start
        return None

    def check_bounds(self, expr, size, lower, upper):
        # Helion handles bounds checking internally via tile masking
        pass

    def helion_index_expr(self, expr) -> str:
        """Convert a sympy index expression to Helion tile-based code.

        Maps iteration range variables (x0, y0, z0, etc.) to their
        corresponding tile variable's .index property which provides the
        actual integer positions within that tile.
        """
        import sympy as sp

        # Detect flat iteration in a multi-dim kernel: output_ndim > 1 but
        # only one pointwise range tree. In this case a single symbol
        # represents the flat linear index and must be expanded to the full
        # linearized expression over all tile dimensions.
        ndim = self._output_ndim
        pw_trees = [t for t in self.range_trees if not _is_reduction_prefix(t.prefix)]
        flat_iteration = ndim > 1 and len(pw_trees) == 1

        # Build placeholder symbols for range tree nodes so we can do
        # sympy-level substitution then print with _HelionExprPrinter.
        sym_to_placeholder: dict[sp.Symbol, sp.Symbol] = {}
        placeholder_to_tile: dict[str, str] = {}
        for sym, entry in self.range_tree_nodes.items():
            if sym not in expr.free_symbols:
                continue
            tree = entry.parent
            if _is_reduction_prefix(tree.prefix):
                continue
            if flat_iteration:
                # If the symbol is a decomposed dimension variable (its length
                # is less than the full tree numel), map it to the individual
                # tile dimension based on its divisor. Otherwise, the symbol
                # represents the full flat linear index and needs the complete
                # linearized expression.
                # NOTE (MA2): _flat_index_access uses the same substitution
                # skeleton; the two paths diverged once already (cf. BL1 in
                # code_review_round2_20260522.md). Keep their decomposed-sym
                # policies aligned, or factor into a shared helper.
                mapped = False
                sym_is_decomposed = entry.length != tree.numel
                if sym_is_decomposed:
                    divisor = entry.divisor
                    if divisor is not None:
                        try:
                            coeff = int(divisor)
                        except (TypeError, ValueError):
                            coeff = None
                        if coeff is not None and coeff != 0:
                            dim = self._coeff_to_dim(coeff)
                            if dim is not None:
                                placeholder_name = f"_ph_tile_{dim}_{sym}"
                                placeholder = sp.Symbol(placeholder_name)
                                sym_to_placeholder[sym] = placeholder
                                tile_expr = self._broadcast_tile_index_expr(
                                    ndim, dim
                                )
                                placeholder_to_tile[placeholder_name] = tile_expr
                                mapped = True
                    if not mapped:
                        # Decomposed sub-symbol with no dim assignment: the
                        # _flat_to_linear_tile_expr fallback below is the FULL
                        # linear flat index, which is only correct for the
                        # whole-tree sym -- substituting it for a sub-symbol
                        # silently produces (e.g.) LINEAR + 4*LINEAR == 5*LINEAR
                        # for ``x0 + 4*x1`` in shape (4, 4). Fail fast so the
                        # caller sees the unsupported case instead of wrong
                        # numerics.
                        raise NotImplementedError(
                            f"Helion backend: cannot map decomposed iter "
                            f"sub-symbol {sym!r} (divisor={entry.divisor}, "
                            f"length={entry.length}) to an output dim under "
                            f"flat iteration (output_size={self._output_size})"
                        )
                if not mapped:
                    linear_expr = self._flat_to_linear_tile_expr(ndim)
                    if linear_expr is not None:
                        placeholder_name = f"_ph_flat_linear_{sym}"
                        placeholder = sp.Symbol(placeholder_name)
                        sym_to_placeholder[sym] = placeholder
                        placeholder_to_tile[placeholder_name] = linear_expr
                        continue
                if mapped:
                    continue
            tile_var = self._prefix_to_tile_var(tree.prefix)
            if tile_var is not None:
                placeholder_name = f"_ph_{tile_var}"
                placeholder = sp.Symbol(placeholder_name)
                sym_to_placeholder[sym] = placeholder
                placeholder_to_tile[placeholder_name] = f"{tile_var}.index"

        # Resolve dynamic shape symbols (s0, s77, etc.) to concrete values.
        remaining = expr.free_symbols - set(sym_to_placeholder.keys())
        concrete_subs: dict[sp.Symbol, sp.Integer] = {}
        for sym in remaining:
            try:
                val = V.graph.sizevars.optimization_hint(sym)
                concrete_subs[sym] = sp.Integer(int(val))
            except (TypeError, ValueError, AttributeError):
                pass

        substituted = expr.subs(concrete_subs).subs(sym_to_placeholder)

        if substituted.is_number:
            try:
                return repr(int(substituted))
            except (TypeError, ValueError):
                pass

        # _helion_expr_printer converts Min->torch.minimum, FloorDiv->//, etc.
        result = _helion_expr_printer.doprint(substituted)

        for placeholder_name, tile_expr in sorted(
            placeholder_to_tile.items(), key=lambda x: -len(x[0])
        ):
            result = result.replace(placeholder_name, tile_expr)
        return result

    @staticmethod
    def _tile_var_name(ndim: int, idx: int = 0) -> str:
        """Return the tile loop variable name for a given dimensionality and index."""
        if ndim <= 1:
            return "tile"
        return f"tile_{idx}"

    @staticmethod
    def _tile_var_names(ndim: int) -> list[str]:
        """Return all tile loop variable names for a given dimensionality."""
        if ndim <= 1:
            return ["tile"]
        return [f"tile_{i}" for i in range(ndim)]

    def _rand_shape_str(self) -> str:
        """Return the shape list string for random op tensor creation."""
        tile_vars = self._tile_var_names(max(self._output_ndim, 1))
        return ", ".join(tile_vars)

    def _tile_loop_line(self, ndim: int, ref_param: str, ref_ndim: int | None = None) -> str:
        """Generate the for-loop line for tile iteration."""
        tile_vars = ", ".join(self._tile_var_names(ndim))
        if ndim <= 1:
            # When the reference buffer is multi-dim but the tiling is 1-D
            # (Inductor collapsed the body to a flat iteration), iterate over
            # the full numel and decompose per buffer in the load/store code.
            if ref_ndim is not None and ref_ndim > ndim:
                return f"for {tile_vars} in hl.tile({ref_param}.numel()):"
            return f"for {tile_vars} in hl.tile({ref_param}.size(0)):"
        if ref_ndim is not None and ref_ndim > ndim:
            return f"for {tile_vars} in hl.tile({ref_param}.size()[:{ndim}]):"
        return f"for {tile_vars} in hl.tile({ref_param}.size()):"

    def _prefix_to_tile_var(self, prefix: str) -> str | None:
        """Map a range tree prefix to its Helion tile variable name."""
        ndim = self._output_ndim
        if ndim <= 1:
            return self._tile_var_name(ndim)

        pointwise_prefixes = []
        for tree in self.range_trees:
            if not _is_reduction_prefix(tree.prefix):
                pointwise_prefixes.append(tree.prefix)

        if prefix in pointwise_prefixes:
            idx = pointwise_prefixes.index(prefix)
            return self._tile_var_name(ndim, idx)
        return None

    def _broadcast_tile_index_expr(self, ndim: int, dim: int) -> str:
        """Return tile_dim.index with broadcast reshaping for a multi-dim kernel.

        For ndim=2: tile_0.index.unsqueeze(1), tile_1.index.unsqueeze(0)
        For ndim=3: tile_i.index.reshape(1,..,-1,..,1) with -1 at position i.
        """
        tile_name = self._tile_var_name(ndim, dim)
        if ndim == 2:
            if dim == 0:
                return f"{tile_name}.index.unsqueeze(1)"
            else:
                return f"{tile_name}.index.unsqueeze(0)"
        shape = ", ".join("-1" if j == dim else "1" for j in range(ndim))
        return f"{tile_name}.index.reshape({shape})"

    def _coeff_to_dim(self, coeff: int) -> int | None:
        """Map an iter-tree entry's divisor to its output dimension index.

        Only meaningful in the flat-iter case (single pointwise tree whose
        entries decompose ``xindex`` into per-output-dim symbols via FloorDiv /
        ModularIndexing). For each entry, ``divisor`` is the running product of
        the inner iter sizes (innermost has divisor 1) and ``length`` is the
        entry's own iter size.

        For CUDA-style schedules the iter order matches the output's row-major
        contig stride order, so ``divisor`` happens to equal an output contig
        stride. For Pallas-style schedules the iter axes may be permuted (e.g.
        ``_sizes=(4, 12, 1024, 513)`` over an output of shape
        ``(4, 1024, 12, 513)``) and ``divisor`` no longer matches any output
        contig stride.

        We resolve both by building a (divisor -> output-dim) map from the iter
        entries themselves: sort entries by divisor (innermost first) and pair
        them with output dims by matching ``length`` to dim size. When the size
        match is ambiguous (two output dims share a size), refuse to map so
        the caller falls back to the full linearized expression.
        """
        if self._output_size is None:
            return None
        iter_to_dim = self._iter_divisor_to_output_dim()
        if iter_to_dim is None:
            return None
        return iter_to_dim.get(coeff)

    def _iter_divisor_to_output_dim(self) -> dict[int, int] | None:
        """Build a (iter-entry divisor -> output-dim) map for flat-iter kernels.

        Walks iter entries innermost-first (sorted by divisor) and matches
        each entry's ``length`` to an unused output dim with the same size.
        When two output dims share a size (e.g. ``(N, N)``), length alone is
        ambiguous; break the tie by matching the entry's ``divisor`` to the
        output's row-major stride for that dim. This is the natural mapping
        for the common case where the iter walks the output in row-major
        order: divisor (running product of inner iter sizes) and output
        row-major stride coincide. Entries that are still ambiguous after
        both checks, or whose length doesn't match any remaining dim, are
        left out; the caller treats a missing key per its own policy.
        """
        out_size = self._output_size
        if out_size is None:
            return None
        ndim = self._output_ndim
        if ndim <= 0:
            return None
        pw_entries: list[tuple[int, int]] = []
        for sym, entry in self.range_tree_nodes.items():
            if _is_reduction_prefix(entry.parent.prefix):
                continue
            try:
                d = int(entry.divisor)
                L = int(entry.length)
            except (TypeError, ValueError):
                return None
            pw_entries.append((d, L))
        if not pw_entries:
            return None
        # Row-major output strides: stride[i] = prod(out_size[i+1:]).
        row_major_stride = [1] * ndim
        for i in range(ndim - 2, -1, -1):
            row_major_stride[i] = row_major_stride[i + 1] * out_size[i + 1]
        pw_entries.sort(key=lambda x: x[0])
        unmatched: list[int] = list(range(ndim))
        result: dict[int, int] = {}
        for d, L in pw_entries:
            matches = [i for i in unmatched if out_size[i] == L]
            if len(matches) > 1:
                # Length is ambiguous; try to disambiguate by divisor matching
                # output row-major stride. Works for the symmetric (N, N) case
                # under row-major iter; harmless under non-row-major iter
                # (no stride match => still skip).
                stride_matches = [i for i in matches if row_major_stride[i] == d]
                if len(stride_matches) == 1:
                    matches = stride_matches
            if len(matches) == 1:
                result[d] = matches[0]
                unmatched.remove(matches[0])
        return result

    def _flat_to_linear_tile_expr(self, ndim: int) -> str | None:
        """Build the linearized index expression for flat iteration in a multi-dim kernel.

        Each tile_i.index must be reshaped to broadcast correctly across all
        dimensions. For ndim=2 with output size (s0, s1):
            tile_0.index.unsqueeze(1) * s1 + tile_1.index.unsqueeze(0)
        For ndim=3 with output size (s0, s1, s2):
            tile_0.index.reshape(-1,1,1) * (s1*s2) + tile_1.index.reshape(1,-1,1) * s2 + tile_2.index.reshape(1,1,-1)
        """
        out_size = self._output_size
        if out_size is None or len(out_size) != ndim:
            return None

        parts: list[str] = []
        for i in range(ndim):
            idx_expr = self._broadcast_tile_index_expr(ndim, i)
            # Compute the stride for this dimension: product of sizes of all
            # subsequent dimensions.
            stride = 1
            for j in range(i + 1, ndim):
                stride *= out_size[j]
            if stride == 1:
                parts.append(idx_expr)
            else:
                parts.append(f"{idx_expr} * {stride}")
        return "(" + " + ".join(parts) + ")"

    def _get_buffer_ndim(self, name: str) -> int:
        """Get the ndim of a buffer by its internal name."""
        buf = V.graph.try_get_buffer(name)
        if buf is not None:
            return len(buf.get_size())
        gi = V.graph.graph_inputs.get(name)
        if gi is not None:
            return len(gi.get_size())
        return self._output_ndim

    def _get_buffer_size_stride(
        self, name: str
    ) -> tuple[list[int], list[int]] | None:
        """Return concrete (size, stride) for a buffer by name, or None."""
        hint = V.graph.sizevars.optimization_hint
        buf = V.graph.try_get_buffer(name)
        obj = buf if buf is not None else V.graph.graph_inputs.get(name)
        if obj is None:
            return None
        size = [hint(s, fallback=0) for s in obj.get_size()]
        stride = [hint(s, fallback=0) for s in obj.get_stride()]
        return size, stride

    def _stride_decoded_load_access(
        self, name: str, index: sympy.Expr
    ) -> str | None:
        """Build a per-buffer-dim subscript by decoding stride coefficients.

        Inductor encodes view operations (transpose, permute) by changing the
        coefficients in the index expression while leaving the underlying
        buffer's shape/stride untouched. For an input buffer with stride
        ``[s_0, s_1, ...]`` and an index ``sum_i c_i * sym_i``, the symbol
        with coefficient ``s_k`` indexes buffer dimension ``k``. This is the
        key contrast with ``_flat_index_access`` (linearizes to a single
        offset) and the legacy positional fallback (assumes tile vars match
        output dims positionally, broken under transpose/permute).

        Returns the full RHS string to CSE-bind in ``load()`` -- including any
        ``.T`` / ``.permute(...)`` to match the output tile shape, broadcast
        ``.unsqueeze`` for missing output dims, and ``.to(torch.float32)`` for
        half-precision dtypes. Returns ``None`` when the index can't be
        interpreted as a stride-decoded access (e.g. contains reduction
        symbols, indirect indexing, or unmatched coefficients).

        Caller is expected to gate with ``_stride_decoded_load_access_applies``
        so ``_output_ndim > 1`` and reduction/flat-multidim/non-input-buffer
        cases are already excluded.
        """
        out_ndim = self._output_ndim

        # Bail on flat iteration: when the output is multi-dim but only one
        # pointwise range tree exists, symbols are decomposed segments of a
        # linear iteration, not direct tile coordinates. helion_index_expr
        # handles that path; here we only handle the simple case where each
        # pointwise tree corresponds to exactly one output dim.
        pw_trees = [
            t for t in self.range_trees if not _is_reduction_prefix(t.prefix)
        ]
        if len(pw_trees) != out_ndim:
            return None

        info = self._get_buffer_size_stride(name)
        if info is None:
            return None
        buf_size, buf_stride = info
        buf_ndim = len(buf_size)
        if buf_ndim == 0:
            return None

        # Map each free symbol to (output dim, tile var, coefficient).
        sym_to_out_dim: dict[sympy.Symbol, int] = {}
        for sym in index.free_symbols:
            entry = self.range_tree_nodes.get(sym)
            if entry is None:
                # Unknown symbol (e.g. dynamic shape) - bail.
                return None
            tree = entry.parent
            if _is_reduction_prefix(tree.prefix):
                return None
            # In the non-flat case each tree's symbol's length should equal
            # the tree's numel; a smaller length means decomposed iteration.
            if entry.length != tree.numel:
                return None
            tile_var = self._prefix_to_tile_var(tree.prefix)
            if tile_var is None:
                return None
            try:
                out_dim = self._tile_var_names(out_ndim).index(tile_var)
            except ValueError:
                return None
            sym_to_out_dim[sym] = out_dim

        # For each buffer dim, find the symbol whose coefficient matches the
        # buffer's stride at that dim. Stride 0 (broadcast / size-1 dim) is
        # allowed even with no matching symbol - the access is just a 0.
        buf_dim_to_sym: list[sympy.Symbol | None] = [None] * buf_ndim
        remaining_syms = set(sym_to_out_dim.keys())
        for k in range(buf_ndim):
            stride_k = buf_stride[k]
            if buf_size[k] == 1:
                # Size-1 dim, just use 0 as the index.
                continue
            match_sym = None
            for sym in remaining_syms:
                cf = index.coeff(sym)
                try:
                    cf_int = int(cf)
                except (TypeError, ValueError):
                    return None
                if cf_int == stride_k:
                    match_sym = sym
                    break
            if match_sym is None:
                # Could not match this buf dim to any symbol.
                return None
            buf_dim_to_sym[k] = match_sym
            remaining_syms.discard(match_sym)

        if remaining_syms:
            # Leftover symbols mean the index contains contributions we can't
            # represent as buf[sym0, sym1, ...]. Bail.
            return None

        # Build the tile-var list in buffer-dim order, using "0" for size-1
        # dims with no matching symbol.
        tile_var_per_buf_dim: list[str] = []
        used_out_dims: list[int] = []
        for k in range(buf_ndim):
            sym = buf_dim_to_sym[k]
            if sym is None:
                tile_var_per_buf_dim.append("0")
            else:
                out_dim = sym_to_out_dim[sym]
                tile_var_per_buf_dim.append(
                    self._tile_var_name(out_ndim, out_dim)
                )
                used_out_dims.append(out_dim)

        # Permutation maps from buffer-dim order to output-tile-dim order.
        # After loading buf[tile_vars_in_buf_dim_order], the result has dims
        # (size for each buf dim) with size-1 dims squeezed out by literal 0
        # indexing (or kept as a real dim when matched to a symbol).
        # The non-trivial dims appear in the order they were matched. We need
        # to permute them so the dim order matches the *output* tile dim order
        # (ascending).
        if len(used_out_dims) <= 1:
            perm: tuple[int, ...] = tuple(range(len(used_out_dims)))
        else:
            # used_out_dims gives the output-dim of each non-trivial buffer
            # dim, in buffer-dim order. We want to reorder these so they
            # match the natural output dim order (ascending out_dim).
            sorted_indices = sorted(
                range(len(used_out_dims)), key=lambda i: used_out_dims[i]
            )
            perm = tuple(sorted_indices)

        buf = self.args.input(name)
        dtype = V.graph.get_dtype(name)
        idx = ", ".join(tile_var_per_buf_dim)
        line = f"{buf}[{idx}]"
        if dtype in (torch.float16, torch.bfloat16):
            line += ".to(torch.float32)"
        # Apply permutation so the loaded tile has the output tile shape.
        if len(perm) >= 2 and tuple(perm) != tuple(range(len(perm))):
            if len(perm) == 2 and perm == (1, 0):
                line += ".T"
            else:
                line += f".permute({perm})"
        # Broadcast missing output dims by unsqueezing in the right places.
        if len(perm) < out_ndim:
            # tile_var_per_buf_dim contains tile_X names (or "0" for size-1
            # buf dims, which collapse out under literal-int indexing).
            tile_names = self._tile_var_names(out_ndim)
            present = set()
            for tv in tile_var_per_buf_dim:
                if tv in tile_names:
                    present.add(tile_names.index(tv))
            for d in range(out_ndim):
                if d in present:
                    continue
                # Insert a length-1 dim at position d. Use positive index
                # since dims to the left are already inserted.
                line += f".unsqueeze({d})"
                present.add(d)
        return line

    def _tile_index(self, name: str, index: sympy.Expr) -> str:
        """Return the appropriate tile indexing string for a buffer access.

        Uses right-aligned broadcasting: a buffer with ndim M uses the last M
        tile variables from the output's N tile variables.
        For reduction buffers, the last dim is accessed via ':' (the reduction
        dim), and the remaining dims use right-aligned tile variables.
        """
        out_ndim = self._output_ndim
        buf_ndim = self._get_buffer_ndim(name)
        has_reduction = self._has_reduction()

        # Flat-multidim mode: the buffer was reshaped to a flat (numel,)
        # view in codegen_kernel; use 1-D tile indexing regardless of the
        # buffer's logical ndim.
        if self._flat_multidim and buf_ndim > 0:
            return self._tile_var_name(1)

        if buf_ndim == 0:
            if self._scalar_as_1d:
                # Scalar buffers are reshaped to [1] at call site; index with tile.
                return self._tile_var_name(ndim=1)
            return "()"

        if out_ndim == 0:
            return self._tile_var_name(ndim=1)

        uses_reduction = False
        if has_reduction:
            for symbol in index.free_symbols:
                if symbol in self.range_tree_nodes:
                    entry = self.range_tree_nodes[symbol]
                    if _is_reduction_prefix(entry.parent.prefix):
                        uses_reduction = True
                        break

        if out_ndim <= 1 and not uses_reduction:
            if buf_ndim <= out_ndim or buf_ndim <= 1:
                return self._tile_var_name(out_ndim)
            # Buffer has extra dims (e.g. keepdim shape (8,1) with out_ndim=1)
            extra = buf_ndim - max(out_ndim, 1)
            parts = [self._tile_var_name(out_ndim)] + [self._REDUCTION_SLICE] * extra
            return ", ".join(parts)

        if out_ndim <= 1:
            if uses_reduction and self.inside_reduction:
                pw_dims = buf_ndim - 1  # last dim is reduction
                if pw_dims == 0:
                    return self._REDUCTION_SLICE
                n_reduction_slices = buf_ndim - max(out_ndim, 1)
                parts = [self._tile_var_name(out_ndim)]
                parts.extend([self._REDUCTION_SLICE] * n_reduction_slices)
                return ", ".join(parts)
            return self._tile_var_name(out_ndim)

        if uses_reduction and self.inside_reduction:
            pw_ndim = buf_ndim - 1
            start = max(0, out_ndim - pw_ndim)
            tile_parts = [self._tile_var_name(out_ndim, i) for i in range(start, out_ndim)]
            # Extra ':' slices for pointwise dims beyond the tile dimensions
            extra_pw = max(0, pw_ndim - out_ndim)
            tile_parts.extend([self._REDUCTION_SLICE] * extra_pw)
            tile_parts.append(self._REDUCTION_SLICE)
        else:
            if buf_ndim < out_ndim and name in self.args.input_buffers:
                # Buffer has fewer dims than output. Index with the tile
                # variables corresponding to the buffer's matched dimensions.
                matched_start = self._find_buf_dim_alignment(name, buf_ndim)
                if matched_start is not None:
                    tile_parts = [self._tile_var_name(out_ndim, i) for i in range(matched_start, matched_start + buf_ndim)]
                else:
                    start = max(0, out_ndim - buf_ndim)
                    tile_parts = [self._tile_var_name(out_ndim, i) for i in range(start, out_ndim)]
            else:
                start = max(0, out_ndim - buf_ndim)
                tile_parts = [self._tile_var_name(out_ndim, i) for i in range(start, out_ndim)]
            extra = max(0, buf_ndim - out_ndim)
            tile_parts.extend([self._REDUCTION_SLICE] * extra)

        if len(tile_parts) == 1:
            return tile_parts[0]
        return ", ".join(tile_parts)

    def _stride_decoded_load_access_applies(self, name: str) -> bool:
        """Whether ``load(name, ...)`` is a candidate for stride-decoded access.

        Paired with ``_stride_decoded_load_access`` (the doer): predicate +
        doer share the ``_stride_decoded_load_access`` stem. Also gates the
        flat-decode fallback and the legacy positional safety net so the
        three places in ``load()`` share the same precondition. The shape-
        mismatch check is intentionally NOT included here: it is the
        additional condition that further narrows path 2 (flat-decode) and
        path 3 (safety net).

        These checks are NOT redundant with the doer's internal bails:
        ``_flat_multidim`` forces 1-D buffer aliasing the doer's multi-dim
        subscript would violate; ``name in input_buffers`` keeps the doer
        from calling ``args.input(name)`` for an output/inplace buffer
        (which has the side effect of registering it as an input); and the
        ``inside_reduction`` guard is a safety policy broader than the
        doer's per-symbol reduction-prefix check.
        """
        return (
            self._output_ndim > 1
            and not self._flat_multidim
            and name in self.args.input_buffers
            and not (self._has_reduction() and self.inside_reduction)
        )

    def load(self, name: str, index: sympy.Expr) -> CSEVariable:
        buf = self.args.input(name)
        dtype = V.graph.get_dtype(name)
        out_ndim = self._output_ndim

        # First try the stride-decoded path: decode coefficients in the index
        # expression against the buffer's stride to produce a per-buffer-dim
        # subscript that handles transposed/permuted views (e.g. ``x + y.T``)
        # in-kernel.
        stride_decoded_applies = self._stride_decoded_load_access_applies(name)
        line = (
            self._stride_decoded_load_access(name, index)
            if stride_decoded_applies
            else None
        )
        if line is not None:
            return self.cse.generate(self.loads, line)

        # Detect dangerous N-D positional indexing: when the buffer's shape
        # doesn't match the output's tile-dim sizes, the legacy positional
        # ``buf[tile_0, tile_1, ...]`` walk indexes well past the buffer's
        # actual storage. This case caused the AllenaiLongformer CUDA crash
        # (see agents_notes/cuda_illegal_memory_access.md). Use the
        # stride-decoded flat indexing path instead.
        shape_mismatches = (
            stride_decoded_applies and self._buf_shape_mismatches_output(name)
        )
        flat_idx_expr = (
            self._flat_index_access(name, index) if shape_mismatches else None
        )
        if flat_idx_expr is not None:
            flat_name = f"{buf}_flat"
            self._flat_aliases[buf] = flat_name
            # Make correctness self-evident: wrap the clamped load in an
            # explicit bounds-check ``torch.where``. The clamp keeps the CUDA
            # address in-range; the ``where`` ensures OOB lanes return a
            # neutral default rather than whatever lives at offset 0, even if
            # a future Inductor change drops the downstream masking.
            # Bind ``flat_idx_expr`` to a CSE local so the 30+-op index
            # expression is emitted once and re-referenced for the two bounds
            # checks plus the clamp (instead of three textual copies).
            idx_var = self.cse.generate(self.loads, flat_idx_expr)
            size_expr = f"{flat_name}.size(0)"
            clamped = (
                f"{flat_name}[torch.clamp({idx_var}, "
                f"max={size_expr} - 1, min=0)]"
            )
            if dtype in (torch.float16, torch.bfloat16):
                clamped += ".to(torch.float32)"
            default = "False" if dtype is torch.bool else "0"
            line = (
                f"torch.where(({idx_var} >= 0) & "
                f"({idx_var} < {size_expr}), {clamped}, {default})"
            )
            return self.cse.generate(self.loads, line)

        # Fallback: legacy positional tile indexing + broadcast unsqueeze.
        # When the buffer's shape doesn't match the output's tile dim sizes,
        # positional ``buf[tile_0, tile_1, ...]`` indexing would walk past
        # the buffer's storage. This caused the AllenaiLongformer CUDA crash
        # (see ``cuda_illegal_memory_access.md``). Refuse to generate the
        # broken kernel so the compilation fails cleanly instead of crashing
        # CUDA at runtime and poisoning the device for subsequent tests.
        if shape_mismatches:
            raise NotImplementedError(
                f"Helion backend cannot generate safe N-D load for "
                f"buffer {name!r} (shape doesn't match output tile "
                f"sizes; index={index})"
            )
        idx = self._tile_index(name, index)
        line = f"{buf}[{idx}]"
        if dtype in (torch.float16, torch.bfloat16):
            line += ".to(torch.float32)"
        if name in self.args.input_buffers and self._output_ndim > 1:
            buf_ndim = self._get_buffer_ndim(name)
            if buf_ndim < out_ndim:
                matched_start = self._find_buf_dim_alignment(name, buf_ndim)
                if matched_start is not None:
                    for d in range(matched_start):
                        line += ".unsqueeze(0)"
                    trailing = out_ndim - (matched_start + buf_ndim)
                    for _ in range(trailing):
                        line += ".unsqueeze(-1)"
        return self.cse.generate(self.loads, line)

    def _buf_shape_mismatches_output(self, name: str) -> bool:
        """Return True if the buffer's positional shape can't match the output
        tile dims (i.e. ``buf[tile_0, tile_1, ...]`` would walk off the end).
        """
        buf_ndim = self._get_buffer_ndim(name)
        if buf_ndim != self._output_ndim:
            return True
        info = self._get_buffer_size_stride(name)
        if info is None or self._output_size is None:
            return False
        buf_size, _ = info
        # Buffer's positional sizes don't match the output tile dims sizes.
        return buf_size != self._output_size

    def _flat_index_access(
        self, name: str, index: sympy.Expr
    ) -> str | None:
        """Build a flat-storage-offset expression for ``buf[index]``.

        Substitutes each iter-tree symbol with its multi-dim tile-index
        broadcast expression, then renders the result as Helion DSL. The
        caller adds ``buf_flat = buf.as_strided((numel,), (1,))`` and uses
        ``buf_flat[expr]`` for the actual load. Handles both the natural
        N-D iteration case (one pw tree per output dim) and the flat-
        iteration case (single pw tree decomposed into per-dim subsymbols
        via ``range_tree_nodes``' divisor).
        """
        import sympy as sp

        out_ndim = self._output_ndim
        if out_ndim <= 1:
            return None
        pw_trees = [
            t for t in self.range_trees if not _is_reduction_prefix(t.prefix)
        ]
        flat_iteration = len(pw_trees) == 1
        sym_to_placeholder: dict[sp.Symbol, sp.Symbol] = {}
        placeholder_to_tile: dict[str, str] = {}
        for sym, entry in self.range_tree_nodes.items():
            if sym not in index.free_symbols:
                continue
            tree = entry.parent
            if _is_reduction_prefix(tree.prefix):
                return None
            placeholder_name = f"_flat_ph_{sym}"
            placeholder = sp.Symbol(placeholder_name)
            if flat_iteration:
                # The tree's symbol decomposes into sub-symbols; each maps to a
                # tile dim via its divisor in the output's row-major stride
                # arithmetic. See ``helion_index_expr`` for the same logic.
                if entry.length == tree.numel:
                    # Full flat symbol -- linearize across all tile dims.
                    linear_expr = self._flat_to_linear_tile_expr(out_ndim)
                    if linear_expr is None:
                        return None
                    sym_to_placeholder[sym] = placeholder
                    placeholder_to_tile[placeholder_name] = linear_expr
                    continue
                divisor = entry.divisor
                if divisor is None:
                    return None
                try:
                    coeff = int(divisor)
                except (TypeError, ValueError):
                    return None
                if coeff == 0:
                    return None
                dim = self._coeff_to_dim(coeff)
                if dim is None:
                    return None
                sym_to_placeholder[sym] = placeholder
                placeholder_to_tile[placeholder_name] = (
                    self._broadcast_tile_index_expr(out_ndim, dim)
                )
            else:
                if entry.length != tree.numel:
                    return None
                tile_var = self._prefix_to_tile_var(tree.prefix)
                if tile_var is None:
                    return None
                try:
                    dim = self._tile_var_names(out_ndim).index(tile_var)
                except ValueError:
                    return None
                sym_to_placeholder[sym] = placeholder
                placeholder_to_tile[placeholder_name] = (
                    self._broadcast_tile_index_expr(out_ndim, dim)
                )
        # Resolve dynamic shape symbols.
        remaining = index.free_symbols - set(sym_to_placeholder.keys())
        concrete_subs: dict[sp.Symbol, sp.Integer] = {}
        for sym in remaining:
            try:
                val = V.graph.sizevars.optimization_hint(sym)
                concrete_subs[sym] = sp.Integer(int(val))
            except (TypeError, ValueError, AttributeError):
                return None
        substituted = index.subs(concrete_subs).subs(sym_to_placeholder)
        try:
            result = _helion_expr_printer.doprint(substituted)
        except Exception:
            return None
        for placeholder_name, tile_expr in sorted(
            placeholder_to_tile.items(), key=lambda x: -len(x[0])
        ):
            result = result.replace(placeholder_name, tile_expr)
        return result

    def _is_scalar_constant(self, value: CSEVariable) -> bool:
        return (
            isinstance(value, HelionCSEVariable)
            and value.constant_value is not _NOT_CONSTANT
        )

    def store(
        self, name: str, index: sympy.Expr, value: CSEVariable, mode: Any = None
    ) -> None:
        out = self.args.output(name)
        self.store_buffer_names.add(name)
        idx = self._tile_index(name, index)
        # If this is a reduction kernel and the store doesn't include the
        # reduction dim (no ':'), squeeze the value to remove keepdim.
        # Skip for scalar outputs (ndim==0) which already reduce without keepdim.
        if self._has_reduction() and self._REDUCTION_SLICE not in idx and self._output_ndim > 0:
            value = self.cse.generate(
                self.compute, f"({value}).squeeze(-1)"
            )
        # Expand scalar constants to the output tile shape so Pallas doesn't
        # try to subscript a 0-dim value.
        if self._output_ndim > 0 and self._is_scalar_constant(value):
            value = self.cse.generate(
                self.compute, f"({value}).expand({out}[{idx}].shape)"
            )
        red_type = (
            value.reduction_type if isinstance(value, HelionCSEVariable) else None
        )
        if red_type is not None:
            self._reduction_type_per_out[name] = red_type
        if mode == "atomic_add":
            line = f"hl.atomic_add({out}, [{idx}], {value})"
        elif (
            self._output_ndim == 0
            and self._has_reduction()
            and red_type in ATOMIC_OP_FOR_REDUCTION
        ):
            atomic_fn = ATOMIC_OP_FOR_REDUCTION[red_type]
            idx_list = "[]" if idx == "()" else f"[{idx}]"
            line = f"{atomic_fn}({out}, {idx_list}, {value})"
        else:
            line = f"{out}[{idx}] = {value}"
        self.store_lines.append(DeferredLine(name, line))

    def reduction(self, dtype, src_dtype, reduction_type, value):
        """Handle reduction ops by emitting torch reduction calls."""
        if reduction_type == "welford_reduce":
            return self.welford_reduce_fallback(dtype, value)
        reduction_fn = REDUCTION_TYPE_MAP[reduction_type]
        if self._output_ndim == 0:
            expr = f"{reduction_fn}({value})"
        else:
            out_ndim = self._output_ndim
            expr = f"{reduction_fn}({value}.flatten({out_ndim}), dim=-1, keepdim=True)"
        result = self.cse.generate(self.compute, expr)
        if isinstance(result, HelionCSEVariable):
            result.reduction_type = reduction_type
        return result

    def store_reduction(self, name, index, value):
        prior = self.inside_reduction
        self.inside_reduction = False
        try:
            return self.store(name, index, value)
        finally:
            self.inside_reduction = prior

    def disable_reduction(self) -> contextlib.AbstractContextManager[None]:
        @contextlib.contextmanager
        def ctx():
            if not self._has_reduction():
                yield
                return
            prior = self.inside_reduction
            self.inside_reduction = False
            try:
                yield
            finally:
                self.inside_reduction = prior

        return ctx()

    def codegen_kernel(self) -> str:
        buf = IndentedBuffer()
        buf.writeline("import math")
        buf.writeline("import torch")
        buf.writeline("import helion")
        buf.writeline("import helion.language as hl")
        buf.writeline("from helion.runtime.settings import Settings")
        buf.writeline("")
        buf.writeline("")

        arg_defs, _, _, _ = self.args.python_argdefs()
        param_names = [a.name for a in arg_defs]
        param_strs = [f"{name}: torch.Tensor" for name in param_names]

        backend = self._helion_backend
        settings_parts = [f"backend='{backend}'"]
        if backend == "pallas":
            device = V.graph.get_current_device_or_throw()
            if device.type == "cpu":
                settings_parts.append("pallas_interpret=True")
        settings_parts.append(f"autotune_effort='{config.helion_autotune_effort}'")
        if config.helion_print_output_code:
            settings_parts.append("print_output_code=True")
        settings_str = ", ".join(settings_parts)
        buf.writeline(f"@helion.kernel(settings=Settings({settings_str}))")
        buf.writeline(
            f"def {HELION_KERNEL_NAME}({', '.join(param_strs)}) -> None:"
        )

        with buf.indent():
            out_params = [v for v in self.args.output_buffers.values() if not isinstance(v, RemovedArg)]
            in_params = list(self.args.input_buffers.values())
            inner_to_outer = {
                v: k for k, v in self.args.input_buffers.items()
            }
            ref_param = out_params[0] if out_params else param_names[0]
            ndim = self._output_ndim

            # All-scalar case (B1): every read/write is on a 0-D tensor.
            # Reshape each parameter to a 1-D view at the host level so the
            # device loop can use the usual tile indexing without mutating
            # caller tensors at the call site.
            if self._scalar_as_1d:
                for pname in param_names:
                    buf.writeline(f"{pname} = {pname}.view(1)")

            if ndim <= 1:
                # Check if the output buffer is scalar (0-dim); if so, tile
                # over an input buffer instead (full reduction case).
                out_buf_name = self._first_output_buffer_name()
                out_buf_ndim = (
                    self._get_buffer_ndim(out_buf_name) if out_buf_name else 1
                )
                if out_buf_ndim == 0 and not self._scalar_as_1d:
                    # Output is scalar but at least one input has ndim > 0
                    # (full reduction). Tile over the larger input.
                    for inp in in_params:
                        inp_buf_name = inner_to_outer.get(inp)
                        if inp_buf_name and self._get_buffer_ndim(inp_buf_name) > 0:
                            ref_param = inp
                            break
                # Pass ref_ndim so the tile loop iterates over .numel() when
                # the output is multi-dim but the tiling is 1-D.
                ref_buf_name_pw = self._first_output_buffer_name()
                ref_ndim_pw = (
                    self._get_buffer_ndim(ref_buf_name_pw)
                    if ref_buf_name_pw
                    else 1
                )
                # Flat-multidim mode: the tiling is 1-D but at least one
                # buffer is multi-dim. Reshape each multi-dim buffer to a
                # flat ``as_strided((numel,), (1,))`` view so the loads/
                # stores can use simple ``buf[tile]`` indexing.
                if self._flat_multidim:
                    for pname in param_names:
                        # Use the inner kernel arg name to look up the
                        # backing IR buffer's ndim.
                        outer = inner_to_outer.get(pname)
                        if outer is None:
                            # Output buffer: look up by output_buffers.
                            for bname, val in self.args.output_buffers.items():
                                if val == pname:
                                    outer = bname
                                    break
                        if outer is None:
                            continue
                        bndim = self._get_buffer_ndim(outer)
                        if bndim > 1:
                            buf.writeline(
                                f"{pname} = {pname}.as_strided(({pname}.numel(),), (1,))"
                            )
                buf.writeline(
                    self._tile_loop_line(1, ref_param, ref_ndim_pw)
                )
            else:
                # For reduction kernels, the output may have fewer dims than
                # ndim (e.g., var produces 1D from 2D input). Use an input
                # buffer as the tile reference when the output is too small.
                ref_buf_name = self._first_output_buffer_name()
                ref_ndim = (
                    self._get_buffer_ndim(ref_buf_name)
                    if ref_buf_name
                    else ndim
                )
                if ref_ndim != ndim and in_params:
                    for inp in in_params:
                        inp_buf_name = inner_to_outer.get(inp)
                        if inp_buf_name is None:
                            continue
                        if self._get_buffer_ndim(inp_buf_name) == ndim:
                            ref_param = inp
                            ref_ndim = ndim
                            break
                # Emit ``buf_flat = buf.as_strided((n,), (1,))`` for each
                # buffer the load path needs to flat-decode. Helion only
                # accepts ``as_strided`` calls at host level (before the tile
                # loop), so we hoist them here.
                for orig_name, flat_name in self._flat_aliases.items():
                    buf.writeline(
                        f"{flat_name} = {orig_name}.as_strided(({orig_name}.numel(),), (1,))"
                    )
                buf.writeline(self._tile_loop_line(ndim, ref_param, ref_ndim))
            with buf.indent():
                for line in self.loads._lines:
                    buf.writeline(str(line))
                for line in self.compute._lines:
                    buf.writeline(str(line))
                for deferred in self.store_lines:
                    resolved = deferred()
                    if resolved is not None:
                        buf.writeline(resolved)

        return buf.getvalue()

    def _is_pallas_backend(self) -> bool:
        return self._helion_backend == "pallas"

    def _noncontig_output_buffers(self) -> list[tuple[str, list[int]]]:
        """Detect output buffers with non-contiguous strides on the Pallas path.

        The Pallas runtime calls .contiguous() on all tensor args, which creates
        a copy for non-contiguous tensors.  For inplace outputs this means the
        kernel writes to the copy, not the original buffer.  Returns a list of
        (buf_name, size) for outputs that need a contiguous wrapper.
        """
        if not self._is_pallas_backend():
            return []
        hint = V.graph.sizevars.optimization_hint
        results = []
        for buf_name, buf_val in self.args.output_buffers.items():
            if isinstance(buf_val, RemovedArg):
                continue
            buf = V.graph.try_get_buffer(buf_name)
            if buf is None:
                continue
            size = [hint(s, fallback=0) for s in buf.get_size()]
            stride = [hint(s, fallback=0) for s in buf.get_stride()]
            expected = [0] * len(size)
            s = 1
            for i in range(len(size) - 1, -1, -1):
                expected[i] = s
                s *= size[i]
            if stride != expected:
                results.append((buf_name, size))
        return results

    def call_kernel(self, name: str, node: IRNode | None = None) -> None:
        wrapper = V.graph.wrapper_code
        _, call_args, _, _ = self.args.python_argdefs()
        call_arg_strs = [str(a) for a in call_args]

        # On the Pallas path, the runtime calls .contiguous() on all tensors.
        # Non-contiguous output buffers therefore receive a disconnected copy
        # that the kernel writes into, so results never reach the original
        # buffer.  Substitute a contiguous temporary and copy back. Seeding the
        # temporary with the current buffer contents preserves any value the
        # kernel only partially writes (atomic accumulations rely on this).
        # TODO(helion-followup): this contig-copy is a buffer-management
        # workaround for Pallas's runtime forcing .contiguous() on output
        # tensors. The proper fix is upstream in Helion/Pallas so the kernel
        # writes directly into the original (non-contig) storage. Tracked in
        # ~/dunfanlu_notes/inductor_to_helion/agents_notes/helion_followups_20260522.md.
        noncontig_outputs = self._noncontig_output_buffers()
        contig_for_buf: dict[str, str] = {}
        for buf_name, _size in noncontig_outputs:
            try:
                idx = call_arg_strs.index(buf_name)
            except ValueError:
                continue
            contig_name = f"{buf_name}_contig"
            wrapper.writeline(f"{contig_name} = {buf_name}.contiguous().clone()")
            call_arg_strs[idx] = contig_name
            contig_for_buf[buf_name] = contig_name

        # Initialize output buffers for scalar reductions using atomics.
        # Per-output map (M5) so kernels with multiple reductions emit the
        # right init for each output (e.g. amax => fill_(-inf), sum => zero_).
        if self._output_ndim == 0 and self._has_reduction():
            for buf_name, buf_val in self.args.output_buffers.items():
                if isinstance(buf_val, RemovedArg):
                    continue
                red_type = self._reduction_type_per_out.get(buf_name)
                init_call = INIT_FOR_REDUCTION.get(red_type) if red_type else None
                if init_call is None:
                    continue
                # Apply init to the contig temporary when one is in use so the
                # kernel accumulates into the buffer we actually read back.
                target = contig_for_buf.get(buf_name, buf_name)
                wrapper.writeline(f"{target}{init_call}")

        wrapper.writeline(f"{name}.run({', '.join(call_arg_strs)})")

        # Copy contiguous temporaries back to the original non-contiguous outputs.
        for buf_name, contig_name in contig_for_buf.items():
            wrapper.writeline(f"{buf_name}.copy_({contig_name})")


def _get_helion_backend() -> str:
    # tpu and cpu currently force pallas; xla is not registered as a backend
    # device (no entry in tpu_backends), so it can't reach here.
    device = V.graph.get_current_device_or_throw()
    if device.type in ("tpu", "cpu"):
        return "pallas"
    return config.helion_backend


class HelionScheduling(SIMDScheduling):
    kernel_type = HelionKernel  # type: ignore[assignment]

    @classmethod
    def get_backend_features(cls, device: torch.device) -> OrderedSet[BackendFeature]:
        return OrderedSet([BackendFeature.REDUCE_TO_SINGLE_ELEMENT])

    def _determine_output_shape(
        self,
        kernel_features: SIMDKernelFeatures,
        tiling: Any,
    ) -> tuple[int, list[int] | None, bool]:
        """Determine output ndim, concrete size, and scalar-kernel flag.

        Returns (output_ndim, output_size, all_scalar).
        """
        all_writes: set[str] = set()
        all_reads: set[str] = set()
        for node in kernel_features.scheduler_nodes():
            for dep in node.read_writes.writes:
                all_writes.add(dep.name)
            for dep in node.read_writes.reads:
                all_reads.add(dep.name)

        final_outputs = all_writes - all_reads
        output_ndim = 0
        for name in final_outputs:
            buf = V.graph.try_get_buffer(name)
            if buf is not None:
                output_ndim = max(output_ndim, len(buf.get_size()))
        if output_ndim == 0 and all_writes:
            for name in all_writes:
                buf = V.graph.try_get_buffer(name)
                if buf is not None:
                    output_ndim = max(output_ndim, len(buf.get_size()))

        # For reduction kernels where the output shape matches the input
        # (keepdim-style ops like softmax), subtract reduction dims.
        if kernel_features.is_reduction() and output_ndim > 1:
            n_reduction_dims = sum(
                1 for prefix in tiling if _is_reduction_prefix(prefix)
            )
            max_read_ndim = 0
            for name in all_reads:
                buf = V.graph.try_get_buffer(name)
                if buf is not None:
                    max_read_ndim = max(max_read_ndim, len(buf.get_size()))
                else:
                    gi = V.graph.graph_inputs.get(name)
                    if gi is not None:
                        max_read_ndim = max(max_read_ndim, len(gi.get_size()))
            if n_reduction_dims > 0 and output_ndim >= max_read_ndim:
                output_ndim -= n_reduction_dims

        # Detect scalar-to-scalar kernels
        all_scalar = output_ndim == 0
        if all_scalar:
            for name in all_reads:
                buf = V.graph.try_get_buffer(name)
                if buf is not None and len(buf.get_size()) > 0:
                    all_scalar = False
                    break
                gi = V.graph.graph_inputs.get(name) if buf is None else None
                if gi is not None and len(gi.get_size()) > 0:
                    all_scalar = False
                    break

        # Compute concrete output size for flat-iteration linearization
        output_size: list[int] | None = None
        if output_ndim > 1:
            hint = V.graph.sizevars.optimization_hint
            for name in (final_outputs or all_writes):
                buf = V.graph.try_get_buffer(name)
                if buf is not None and len(buf.get_size()) == output_ndim:
                    output_size = [hint(s, fallback=0) for s in buf.get_size()]
                    break

        return output_ndim, output_size, all_scalar

    @staticmethod
    def _dep_decomposes_under_nd_tiling(
        dep: Any,
        nd_sizes: list[int],
        buf_size: list[int],
        buf_stride: list[int],
    ) -> bool:
        """Check whether a dep's index can be expressed under N-D tiling.

        Called for both read and write deps. Substitute the dep's own iter
        symbols with their N-D-tiling decomposition (each dep dim splits into
        one or more output dims), then verify each output symbol's coefficient
        matches a buffer stride. Returns False whenever the decomposition
        would force the access to fall back to broken positional
        ``buf[tile_0, tile_1, ...]`` indexing.
        """
        import sympy as sp

        index = dep.index
        body_syms = list(dep.var_names)
        try:
            body_sizes = [int(s) for s in dep.size]
        except (TypeError, ValueError):
            return False
        if not body_sizes:
            return True  # 0-D access, trivially compatible.
        out_ndim = len(nd_sizes)
        out_syms = [sp.Symbol(f"_nd_o{i}") for i in range(out_ndim)]
        subs: dict[Any, Any] = {}
        out_idx = 0
        for b_idx, b_size in enumerate(body_sizes):
            if b_size == 1:
                subs[body_syms[b_idx]] = sp.Integer(0)
                continue
            prod = 1
            consumed: list[int] = []
            while prod < b_size and out_idx < out_ndim:
                prod *= int(nd_sizes[out_idx])
                consumed.append(out_idx)
                out_idx += 1
            if prod != b_size or not consumed:
                return False
            stride = 1
            terms: list[Any] = []
            for d in reversed(consumed):
                terms.append(out_syms[d] * stride)
                stride *= int(nd_sizes[d])
            subs[body_syms[b_idx]] = sum(terms)
        if out_idx != out_ndim:
            return False
        try:
            new_index = sp.expand(index.subs(subs))
        except Exception:
            return False
        used_buf_dims: set[int] = set()
        for out_sym in out_syms:
            cf = new_index.coeff(out_sym)
            try:
                c = int(cf)
            except (TypeError, ValueError):
                return False
            if c == 0:
                continue
            matched = False
            for k in range(len(buf_stride)):
                if k in used_buf_dims:
                    continue
                if int(buf_stride[k]) == c:
                    used_buf_dims.add(k)
                    matched = True
                    break
            if not matched:
                return False
        return True

    def _try_nd_tiling(self, kernel_features):
        """Try to produce N-D tiling matching the output buffer's natural shape.

        Returns (tiling, None) if successful, else None.
        """
        import sympy as sp

        from ..scheduler import SchedulerNode
        from .simd_kernel_features import EnableReduction

        node_schedule = kernel_features.node_schedule
        numel = kernel_features.numel
        reduction_numel = kernel_features.reduction_numel

        output_sizes = None
        input_sizes = None
        for node in EnableReduction.filter(node_schedule):
            if not isinstance(node, SchedulerNode):
                continue
            if not hasattr(node, "node"):
                continue
            for dep in node.read_writes.writes:
                buf = V.graph.try_get_buffer(dep.name)
                if buf is not None and len(buf.get_size()) >= 2:
                    output_sizes = list(buf.get_size())
                    break
            if output_sizes is not None:
                for dep in node.read_writes.reads:
                    buf = V.graph.try_get_buffer(dep.name)
                    if buf is not None and len(buf.get_size()) > len(output_sizes):
                        input_sizes = list(buf.get_size())
                        break
                break

        if output_sizes is None or not (2 <= len(output_sizes) <= 3):
            return None

        hint = V.graph.sizevars.optimization_hint
        out_ndim = len(output_sizes)

        # Only use N-D tiling when the output dims are a prefix of the
        # input dims (i.e. reduction is over trailing dimensions). When
        # reduction collapses inner dims, tile variables can't be used
        # as positional indices into the input.
        if input_sizes is not None:
            for i, out_s in enumerate(output_sizes):
                if hint(out_s) != hint(input_sizes[i]):
                    return None

        # Reject N-D tiling when any read buffer has different tiled dimensions
        # than the output. Operations like cat have inputs smaller than the
        # output, and tile indexing cannot express the offset adjustments.
        # Permuted inputs (same set of sizes in a different order) are allowed
        # only when their load index can be expressed in the N-D tiling space
        # via stride matching; otherwise fall back to flat tiling. Similarly,
        # an output buffer with non-canonical strides (e.g. a transposed view)
        # whose store index doesn't decompose in N-D forces a fallback.
        out_sorted = sorted(hint(s) for s in output_sizes)
        nd_sizes = [hint(s) for s in output_sizes]
        for node in EnableReduction.filter(node_schedule):
            if not isinstance(node, SchedulerNode):
                continue
            if not hasattr(node, "node"):
                continue
            for dep in node.read_writes.writes:
                buf = V.graph.try_get_buffer(dep.name)
                if buf is None:
                    continue
                buf_size = [hint(s) for s in buf.get_size()]
                buf_stride = [hint(s) for s in buf.get_stride()]
                if len(buf_size) != out_ndim:
                    continue
                if not self._dep_decomposes_under_nd_tiling(
                    dep, nd_sizes, buf_size, buf_stride
                ):
                    return None
            for dep in node.read_writes.reads:
                buf = V.graph.try_get_buffer(dep.name)
                if buf is None:
                    continue
                buf_size = [hint(s) for s in buf.get_size()]
                buf_stride = [hint(s) for s in buf.get_stride()]
                if len(buf_size) != out_ndim:
                    continue
                buf_sorted = sorted(buf_size)
                if buf_sorted == sorted(nd_sizes) and buf_size == nd_sizes:
                    # Positionally aligned with output; always safe.
                    continue
                if buf_sorted != out_sorted:
                    return None
                # Permuted input: verify its index decomposes under N-D
                # tiling by simulating the iter-var substitution and
                # matching coefficients to the buffer's strides.
                if not self._dep_decomposes_under_nd_tiling(
                    dep, nd_sizes, buf_size, buf_stride
                ):
                    return None

        reduction_dims = (
            [reduction_numel] if reduction_numel != sp.S.One else []
        )
        nd_tiling = self.create_tiling(output_sizes, reduction_dims)
        if self.tiling_is_compatible(
            node_schedule, numel, reduction_numel, nd_tiling
        ):
            # M4: empty dict instead of None so downstream consumers reading
            # self.tiling_scores via subscript don't fail.
            return nd_tiling, {}

        return None

    def codegen_node_schedule(self, kernel_features: SIMDKernelFeatures) -> None:
        node_schedule = kernel_features.node_schedule

        # Try N-D tiling first (only for codegen, not fusion decisions)
        nd_result = self._try_nd_tiling(kernel_features)
        if nd_result is not None:
            tiling, tiling_score = nd_result
        else:
            tiling, tiling_score = self.get_tiling_and_scores(
                node_schedule,
                kernel_features.numel,
                kernel_features.reduction_numel,
                kernel_features.coalesce_analysis,
            )
        kernels = self.create_kernel_choices(
            kernel_features,
            [tiling],
            {"features": kernel_features, "tiling_scores": tiling_score},
        )

        output_ndim, output_size, all_scalar = self._determine_output_shape(
            kernel_features, tiling
        )

        # Detect flat-multidim mode: tiling is 1-D pointwise, every touched
        # buffer is multi-dim with dense storage, and every buffer has the
        # same numel. This fires for cases like ``x.T + 1.0`` where Inductor
        # collapsed the body to a flat iteration over a non-contiguous output.
        # We reshape every buffer to ``as_strided((numel,), (1,))`` in the
        # kernel so the load/store paths can use 1-D indexing. Skip for
        # reductions, broadcasted buffers (different numel), or kernels with
        # any 1-D buffer (those use the legacy tile-index path).
        flat_multidim = False
        n_pw_tiles_chosen = (
            sum(1 for p in tiling if not _is_reduction_prefix(p))
            if isinstance(tiling, dict)
            else 0
        )
        if (
            not all_scalar
            and not kernel_features.is_reduction()
            and n_pw_tiles_chosen == 1
            and output_ndim > 1
        ):
            touched: set[str] = set()
            for n in kernel_features.scheduler_nodes():
                for dep in n.read_writes.writes:
                    touched.add(dep.name)
                for dep in n.read_writes.reads:
                    touched.add(dep.name)
            if touched:
                hint = V.graph.sizevars.optimization_hint
                ref_numel: int | None = None
                flat_multidim = True
                for nm in touched:
                    b = V.graph.try_get_buffer(nm)
                    if b is None:
                        b = V.graph.graph_inputs.get(nm)
                    if b is None or len(b.get_size()) < 2:
                        flat_multidim = False
                        break
                    nm_numel = 1
                    for s in b.get_size():
                        nm_numel *= int(hint(s, fallback=0))
                    if ref_numel is None:
                        ref_numel = nm_numel
                    elif nm_numel != ref_numel:
                        flat_multidim = False
                        break

        # In flat-multidim mode the kernel emits a single 1-D tile loop, so
        # the effective output_ndim is 1 even when the buffer's logical ndim
        # is higher. Without this, ``_tile_index`` would emit broken
        # ``buf[tile, :, :, ...]`` indexing.
        if flat_multidim:
            output_ndim = 1
            output_size = None

        for kernel in kernels:
            kernel._output_ndim = output_ndim
            kernel._output_size = output_size
            kernel._scalar_as_1d = all_scalar
            kernel._flat_multidim = flat_multidim

        for kernel in kernels:
            self.codegen_node_schedule_with_kernel(node_schedule, kernel)

        (final_kernel,) = kernels
        with V.set_kernel_handler(final_kernel):
            for node in kernel_features.scheduler_nodes():
                node.mark_run()

        for kernel in kernels:
            with V.set_kernel_handler(kernel):
                src_code = kernel.codegen_kernel()
            kernel_name = self.define_kernel(src_code, node_schedule, kernel)
            kernel.kernel_name = kernel_name

        final_kernel.call_kernel(final_kernel.kernel_name)

        V.graph.removed_buffers |= final_kernel.removed_buffers
        V.graph.inplaced_to_remove |= final_kernel.inplaced_to_remove
        self.free_buffers_in_scheduler()

    def define_kernel(
        self,
        src_code: str,
        node_schedule: Sequence[BaseSchedulerNode],
        kernel: Any = None,
    ) -> str:
        wrapper = V.graph.wrapper_code
        if src_code in wrapper.src_to_kernel:
            return wrapper.src_to_kernel[src_code]

        fused_name = (
            get_fused_kernel_name(node_schedule, config.triton.descriptive_names)
            if config.triton.descriptive_names
            else ""
        )
        kernel_hash = hashlib.sha256(src_code.encode("utf-8")).hexdigest()[:8]
        if fused_name == "fused":
            kernel_name = f"helion_{kernel_hash}"
        else:
            kernel_name = f"helion_{fused_name}_{kernel_hash}"
        wrapper.src_to_kernel[src_code] = kernel_name

        compile_wrapper = IndentedBuffer()
        compile_wrapper.writeline(f"async_compile.helion({kernel_name!r}, r'''")
        compile_wrapper.splice(src_code, strip=True)
        compile_wrapper.writeline("''')")

        origins, detailed_origins = get_kernel_metadata(node_schedule, wrapper)
        metadata_comment = f"{origins}\n{detailed_origins}"
        wrapper.define_kernel(kernel_name, compile_wrapper.getvalue(), metadata_comment)

        return kernel_name
