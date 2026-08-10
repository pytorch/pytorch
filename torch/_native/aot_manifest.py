"""Native-AOT coverage for the Python JIT override router.

Ops under ``torch/_native/ops/<op>/`` opt into AOT with an ``aot.py``
declaration module (see tools/native_aot/decl.py for the contract). The
router checks AOT coverage once per call, ahead of its cond chain, so
covered calls decline the Python route, fall through the router's aten
fallback, and reach the embedded kernel inside the aten implementation.
Everything not covered keeps JIT override eligibility.

A call is covered iff some point of the declaration's
``kernel_precompile_grid()`` matches every field ``covered_axes()``
returns (dtypes are matched by canonical torch dtype; grid-only fields
like block sizes are ignored). ``covered_axes`` exceptions degrade to
"uncovered". The C++ dispatch chain in the AOT library is the authority
on what actually launches; drift between it and covered_axes is benign
(a call both sides decline lands on stock aten).
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
        # Fast path: declarations with cpp_covers() get a C++ predicate
        # compiled into the AOT library and registered as
        # torch.ops._native_aot.covers_<op>. It answers the same
        # question as the Python matching below at custom-op call cost
        # (~1.5us) instead of covered_axes cost (~7-10us). Resolved
        # lazily: the library loads after coverage is built.
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
        # Gate on the Context switch the stub consultations check: with
        # AOT masked (set_aot_enabled(False)), a covered call must keep
        # its JIT route rather than decline into a stub that will not
        # fire -- otherwise masking silently loses BOTH accelerated
        # routes. ~0.1us per call.
        if not torch._C._get_native_aot_enabled():
            return False
        cpp = self._resolve_cpp_covers()
        if cpp is not None:
            try:
                return cpp(*args, **kwargs)
            except Exception:
                # Arguments the schema can't bind (SymInt sizes, exotic
                # kwargs): uncovered; the JIT cond decides.
                return False
        try:
            values = self._covered_axes(*args, **kwargs)
        except Exception:
            # Underspecified call (FakeTensor without the queried
            # attribute, or arguments that don't bind): uncovered; the
            # JIT cond decides.
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
        # Loaded by file path (not package import) so tests can point
        # _OPS_DIR at fixture directories. Deliberately skips the
        # validating loader (load_declarations): the contract is checked
        # at codegen time, and here we need only the coverage pieces. A
        # module is either one declaration or a family exporting
        # declarations().
        mod = load_by_path(f"{entry}_aot", path)
        family = getattr(mod, "declarations", None)
        for d in family() if family is not None else [mod]:
            coverage[(d.ATEN_OP, d.DISPATCH_KEY)] = _Coverage(
                d.ATEN_OP, d.covered_axes, expand_specs(d.kernel_precompile_grid())
            )
    return coverage


def _base_name(op_symbol: str) -> str:
    # Overload-qualified ("topk.values") and in-place ("scatter_add_")
    # symbols share the base op's declaration: all variants funnel
    # through the same structured wrapper.
    base = op_symbol.split(".")[0]
    return base.removesuffix("_") if not base.endswith("__") else base


def get_coverage(op_symbol: str, dispatch_key: str) -> _Coverage | None:
    """The op's AOT coverage, or None if it has no declaration.

    Lookup order: the exact registration symbol first (declarations may
    be overload-qualified, e.g. "gt.Tensor"), then the base name (base-
    named declarations serve every overload of their unique structured
    group).

    The router checks ``coverage.covers(args, kwargs)`` ONCE per call,
    ahead of the cond chain -- every override of the op would get the
    same answer for the same arguments, so per-cond wrapping would pay
    the check N times on multi-path ops.
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
