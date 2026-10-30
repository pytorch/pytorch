from __future__ import annotations

from typing import Any, TYPE_CHECKING

import torch
from torch.utils import _pytree as pytree
from torch.utils._python_dispatch import _disable_current_modes
from torch.utils._sympy.reference import TensorReferenceAnalysis

from .ops_handler import DefaultHandler
from .utils import upcast_compute_type
from .virtualized import V


if TYPE_CHECKING:
    import sympy


class MaskedValuePropagationOpsHandler(DefaultHandler):
    """Propagate the uniform value of lanes outside a logical mask.

    A known masked value is a typed, real CPU scalar (0-D tensor) shared
    by every lane outside the mask. ``None`` means that this uniform value
    is unknown; known values may be zero, nonzero, Inf, or NaN.

    Backend load code seeds zero only after proving its masking behavior.
    Pointwise operations evaluate known scalars with ``TensorReferenceAnalysis``.
    Constants and casts follow Inductor's compute dtype rules. Unknown
    inputs and unsupported operations produce ``None``.

    CSE stores these values on intermediate variables. Native dot uses a
    proven zero to omit redundant reduction masking.
    """

    name = "MaskedValuePropagationOpsHandler"

    def _default(
        self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> torch.Tensor | None:
        fn = getattr(TensorReferenceAnalysis, name, None)
        if fn is None or any(
            value is None
            or (
                isinstance(value, torch.Tensor)
                and (value.ndim != 0 or value.device.type != "cpu")
            )
            for value in pytree.arg_tree_leaves(*args, **kwargs)
        ):
            return None
        with _disable_current_modes():
            try:
                result = fn(*args, **kwargs)
            except (RuntimeError, TypeError, ValueError):
                return None
        if (
            isinstance(result, torch.Tensor)
            and result.ndim == 0
            and result.device.type == "cpu"
        ):
            return result
        return None

    @staticmethod
    def load(
        name: str,
        index: sympy.Expr,
        *,
        zero_is_proven: bool = False,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor | None:
        """Seed a typed zero only after the backend certifies the emitted load.

        Masking and zero fill must cover every lane outside the logical mask.
        TMA physical bounds alone do not prove this:
        repeat/modulo can bring valid source values into the logical tail.
        Use the emitted result dtype (including upcasts); an omitted dtype uses
        the source dtype. Unproven loads return ``None``.
        """
        if not zero_is_proven:
            return None
        with _disable_current_modes():
            try:
                return torch.tensor(
                    0,
                    dtype=V.graph.get_dtype(name) if dtype is None else dtype,
                    device="cpu",
                )
            except (RuntimeError, TypeError, ValueError, OverflowError):
                return None

    def constant(
        self, value: torch.types.Number, dtype: torch.dtype
    ) -> torch.Tensor | None:
        with _disable_current_modes():
            try:
                return torch.tensor(
                    value, dtype=upcast_compute_type(dtype), device="cpu"
                )
            except (RuntimeError, TypeError, ValueError, OverflowError):
                return None

    def to_dtype(
        self,
        value: torch.Tensor | None,
        dtype: torch.dtype,
        src_dtype: torch.dtype | None = None,
        use_compute_types: bool = True,
    ) -> torch.Tensor | None:
        if value is None or value.ndim != 0 or value.device.type != "cpu":
            return None
        dtype = upcast_compute_type(dtype) if use_compute_types else dtype
        with _disable_current_modes():
            try:
                return TensorReferenceAnalysis.to_dtype(value, dtype)
            except (RuntimeError, TypeError, ValueError):
                return None
