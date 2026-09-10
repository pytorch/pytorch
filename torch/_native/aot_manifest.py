"""Native-AOT coverage for the Python JIT override router.

Ops under ``torch/_native/ops/<op>/`` opt into AOT with an ``aot.py`` declaration
module (contract: tools/native_aot/decl.py). The router checks coverage once per
call, ahead of its cond chain, so a covered call declines the Python route and
reaches the embedded kernel through the router's aten fallback. Anything uncovered
keeps its JIT override eligibility.

A call is covered iff some point of the declaration's ``kernel_precompile_grid()``
matches every field ``covered_axes()`` returns; dtypes match by canonical torch
dtype, grid-only fields like block sizes are ignored, and an exception degrades to
uncovered. The C++ dispatch chain in the AOT library is the authority on what
actually launches, and drift is benign: a call both sides decline lands on stock
aten.
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
    def __init__(self, op: str, covered_axes: Callable[..., dict], grid: list[dict]):
        self._op = op
        self._covered_axes = covered_axes
        self._grid = grid
        # Declarations with cpp_covers() get a C++ predicate in the AOT library,
        # registered as torch.ops._native_aot.covers_<op>: the same answer as the
        # Python matching below for ~1.5us instead of ~7-10us. Resolved lazily,
        # because the library loads after coverage is built.
        self._cpp_covers: Callable[..., bool] | None = None
        self._cpp_probed = False

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

    def covers(self, args: tuple, kwargs: dict) -> bool:
        # Gate on the same Context switch the stub consultations read: with AOT
        # masked, a covered call must keep its JIT route rather than decline into a
        # stub that will not fire, which would lose both accelerated routes.
        if not torch._C._get_native_aot_enabled():
            return False
        cpp = self._resolve_cpp_covers()
        if cpp is not None:
            try:
                return cpp(*args, **kwargs)
            except Exception:
                # Arguments the schema cannot bind: uncovered, so the cond decides.
                return False
        try:
            values = self._covered_axes(*args, **kwargs)
        except Exception:
            # Underspecified call (e.g. a FakeTensor missing the queried attribute):
            # uncovered, so the cond decides.
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
        # Loaded by file path, not package import, so tests can point _OPS_DIR at
        # fixtures, and without the validating loader, since codegen already checked
        # the contract. A module is one declaration or a family exporting
        # declarations().
        mod = load_by_path(f"{entry}_aot", path)
        family = getattr(mod, "declarations", None)
        for d in family() if family is not None else [mod]:
            coverage[(d.ATEN_OP, d.DISPATCH_KEY)] = _Coverage(
                d.ATEN_OP, d.covered_axes, expand_specs(d.kernel_precompile_grid())
            )
    return coverage


def _base_name(op_symbol: str) -> str:
    # Overload-qualified ("topk.values") and in-place ("scatter_add_") symbols
    # share the base op's declaration: one structured wrapper serves all variants.
    base = op_symbol.split(".")[0]
    return base.removesuffix("_") if not base.endswith("__") else base


def get_coverage(op_symbol: str, dispatch_key: str) -> _Coverage | None:
    """The op's AOT coverage, or None if it has no declaration.

    The exact registration symbol wins over the base name, since a declaration may
    be overload-qualified ("gt.Tensor") while a base-named one serves every overload
    of its structured group. The router calls ``covers`` once per call rather than
    per cond, which would pay the check N times on a multi-path op for the same
    answer.
    """
    table = _load_coverage()
    c = table.get((op_symbol, dispatch_key))
    if c is not None:
        return c
    return table.get((_base_name(op_symbol), dispatch_key))


def covers(op_symbol: str, dispatch_key: str, args: tuple, kwargs: dict) -> bool:
    """True if an AOT-embedded kernel serves this call of aten::<op_symbol>."""
    c = get_coverage(op_symbol, dispatch_key)
    return c is not None and c.covers(args, kwargs)
