"""Native-AOT coverage for the Python JIT override router.

Ops under ``torch/_native/ops/<op>/`` opt into AOT with an ``aot.py`` declaration
module (contract: tools/native_aot/decl.py). The router checks coverage once per
call, ahead of its cond chain, so a covered call declines the Python route and
reaches the embedded kernel through the router's aten fallback. Anything uncovered
keeps its JIT override eligibility.

A call is covered only when the build embedded kernels for its op and device.
When available, the generated C++ predicate is authoritative: codegen conjoins
the declaration predicate with the artifact-target and ABI gates. The Python
fallback first checks the embedded device capabilities, then whether some point
of ``kernel_precompile_grid()`` matches every field ``covered_axes()`` returns.
Dtypes match by canonical torch dtype, grid-only fields like block sizes are
ignored, and an exception degrades to uncovered. The C++ dispatch chain is the
authority on what actually launches.
"""

import functools
import os
from collections.abc import Callable
from typing import Any

import torch
from torchgen.native_aot_decl import decl_id_for_op, load_by_path
from torchgen.native_aot_spec_grid import expand_specs


_OPS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ops")

# Grid dtype names (long form) -> torch dtypes.
_SPEC_DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float64": torch.float64,
}


class _Coverage:
    def __init__(
        self,
        op: str,
        covered_axes: Callable[..., dict[str, Any]],
        grid: list[dict[str, Any]],
        *,
        structured: bool = True,
        exact: bool = False,
    ) -> None:
        self._op: str = op
        self.structured: bool = structured
        self.exact: bool = exact
        self._covered_axes: Callable[..., dict[str, Any]] = covered_axes
        self._grid: list[dict[str, Any]] = grid
        # Resolve the authoritative C++ predicate after its library loads.
        self._cpp_covers: Callable[..., bool] | None = None
        self._cpp_probed: bool = False
        self._archs: tuple[int, ...] | None = None
        self._available: dict[int, bool] = {}

    def is_available(self, device: torch.device) -> bool:
        """Whether this build embedded this op for the given device."""
        if device.type != "cuda" or torch.version.hip is not None:
            return False
        if self._archs is None:
            try:
                name = f"archs_{decl_id_for_op(self._op)}"
                self._archs = tuple(getattr(torch.ops._native_aot, name)())
            except (AttributeError, RuntimeError):
                self._archs = ()
        if not self._archs:
            return False
        index = device.index
        if index is None:
            index = torch.cuda.current_device()
        if index not in self._available:
            major, minor = torch.cuda.get_device_capability(index)
            self._available[index] = major * 10 + minor in self._archs
        return self._available[index]

    def _resolve_cpp_covers(self) -> Callable[..., bool] | None:
        if not self._cpp_probed:
            self._cpp_probed = True
            ns = getattr(torch.ops, "_native_aot", None)
            if ns is not None:
                try:
                    # covers op name = decl_id (dots sanitized).
                    self._cpp_covers = getattr(ns, f"covers_{decl_id_for_op(self._op)}")
                except (AttributeError, RuntimeError):
                    pass
        return self._cpp_covers

    def covers(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        *,
        op_symbol: str | None = None,
    ) -> bool:
        # Keep JIT eligible when the C++ AOT consultation is disabled.
        if not torch._C._get_native_aot_enabled():
            return False
        if (
            self.structured
            and op_symbol
            and op_symbol.split(".")[0].endswith("_")
            and args
        ):
            kwargs = dict(kwargs, out=args[0])
        cpp = self._resolve_cpp_covers()
        if cpp is not None:
            try:
                return cpp(*args, **kwargs)
            except Exception:
                # Arguments the schema cannot bind: uncovered, so the cond decides.
                return False
        try:
            call_args = (*args, *kwargs.values())
            tensor = next(
                (arg for arg in call_args if isinstance(arg, torch.Tensor)),
                None,
            )
            if tensor is None or not self.is_available(tensor.device):
                return False
            values = self._covered_axes(*args, **kwargs)
        except Exception:
            # Failed availability or argument probes leave the call to the cond chain.
            return False
        for point in self._grid:
            if all(
                self._field_matches(values.get(f), v)
                for f, v in point.items()
                if f in values
            ):
                return True
        return False

    @staticmethod
    def _field_matches(got: Any, expected: Any) -> bool:
        if isinstance(got, torch.dtype) and isinstance(expected, str):
            expected = _SPEC_DTYPES.get(expected, expected)
        return got == expected


@functools.cache
def _load_coverage() -> dict[tuple[str, str], _Coverage]:
    coverage: dict[tuple[str, str], _Coverage] = {}
    if not os.path.isdir(_OPS_DIR):
        return coverage
    for entry in sorted(os.listdir(_OPS_DIR)):
        path = os.path.join(_OPS_DIR, entry, "aot.py")
        if not os.path.exists(path):
            continue
        # Codegen already validated the contract; load fixtures without package imports.
        mod = load_by_path(f"{entry}_aot", path)
        family = getattr(mod, "declarations", None)
        for d in family() if family is not None else [mod]:
            coverage[(d.ATEN_OP, d.DISPATCH_KEY)] = _Coverage(
                d.ATEN_OP,
                d.covered_axes,
                expand_specs(d.kernel_precompile_grid()),
                structured=getattr(d, "STRUCTURED", True),
                exact=getattr(d, "EXACT", False),
            )
    from .ops.pointwise.table import variant_aten_name

    for (op, key), c in list(coverage.items()):
        if c.structured:
            for variant in ("out", "inplace"):
                try:
                    name = variant_aten_name(op, variant)
                except AttributeError:
                    continue
                if name is not None:
                    coverage.setdefault((name, key), c)
    return coverage


def _base_name(op_symbol: str) -> str:
    # Overload-qualified and in-place symbols share the base op's declaration:
    # one structured wrapper serves all variants.
    base = op_symbol.split(".")[0]
    return base.removesuffix("_") if not base.endswith("__") else base


def get_coverage(op_symbol: str, dispatch_key: str) -> _Coverage | None:
    r"""Resolve exact coverage first, then the matching structured variants."""
    table = _load_coverage()
    c = table.get((op_symbol, dispatch_key))
    if c is not None:
        return c
    # Quantized copy_ uses the same Composite backend CUDA AOT consultation.
    if (op_symbol, dispatch_key) == ("copy_", "QuantizedCUDA"):
        return table.get(("copy_", "CUDA"))
    base = _base_name(op_symbol)
    candidate = table.get((base, dispatch_key))
    if candidate is not None and candidate.structured and not candidate.exact:
        return candidate
    return None


def covers(
    op_symbol: str,
    dispatch_key: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> bool:
    r"""True if an AOT-embedded kernel serves this call of aten::<op_symbol>."""
    c = get_coverage(op_symbol, dispatch_key)
    return c is not None and c.covers(args, kwargs, op_symbol=op_symbol)
