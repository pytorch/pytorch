"""Declaration contract and validating loader for native-AOT op modules.

An op opts into AOT by shipping ``torch/_native/ops/<op>/aot.py``, a
module whose scope imports with stdlib alone (torch lazily inside
function bodies) so torchgen can load it pre-build.

A module declares either ONE op (the module itself carries the exports
below) or a FAMILY: it exports ``declarations() -> list`` of objects,
each carrying the same exports as attributes/methods. Table-driven
families (e.g. pointwise) build their declaration objects from the same
table that drives their JIT registration.

Required exports (module or declaration object):

  ATEN_OP: str                name of a STRUCTURED op: a base name
                              ("topk") when the base resolves to exactly
                              one structured group, or overload-qualified
                              ("gt.Tensor", "all.dim") when overloads
                              have separate structured groups. decl_id()
                              (dots -> underscores) names the stub, the
                              generated kernel, and the covers op.
  DISPATCH_KEY: str           e.g. "CUDA"
  KERNEL_MODULE: str          sibling module exporting build(spec); the
                              export tool package-imports it with the
                              built torch available (two-stage build),
                              so it may share code with the JIT wrapper
  kernel_precompile_grid() -> list[dict]
                              the artifact grid; list-valued fields
                              cross-multiply; one precompiled kernel per
                              expanded point. Field values must survive
                              a JSON round-trip (sidecars store the
                              spec; tuples read back as lists and are
                              normalized for skip detection)
  covered_axes(*schema_args) -> dict
                              project a live call onto grid axes; a call
                              is covered (declines the JIT route) iff
                              some precompile point matches every
                              returned field; exceptions => uncovered
  cpp_dispatch(spec) -> str   one boolean C++ expr per precompile point:
                              given a call that passed the prelude, is
                              it served by THIS point? First match wins.
  cpp_launch(spec, launch_fn) -> str
                              C++ invoking this point's kernel via
                              launch_fn(...); no allocation, no fallback
                              (the chain's return false IS the fallback)

Optional exports:

  ARCHS: tuple[str, ...]      architectures the op's kernels are valid
                              on (sm strings). Defaults to all sm90+.
                              Export skips arches outside it; codegen
                              emits a runtime device gate from
                              ARCHS intersect shipped-arches, so
                              declarations never hand-write arch
                              checks.
  cpp_dispatch_prelude() -> str | None
                              shared front half of the dispatch chain:
                              cheap universal rejects and setup (locals,
                              classifier calls) every branch reads. May
                              also `return true` for degenerate calls
                              the op serves without a kernel (e.g. an
                              empty index -> copy only), bypassing the
                              chain entirely. Absent => every
                              cpp_dispatch(spec) must be self-contained.
  cpp_helpers() -> str | None C++ shared beyond one op (family
                              classifiers); emitted once per generated
                              file.
  cpp_covers() -> str | None  fast C++ port of covered_axes matching:
                              a bool-returning body over the op's
                              FUNCTIONAL schema arguments (outputs do
                              not exist yet -- this runs at router
                              time, pre-meta()). Registered by the AOT
                              library as torch.ops._native_aot
                              .covers_<op>; the runtime coverage layer
                              prefers it over the Python path when the
                              library is loaded. Must decide the SAME
                              covered set as covered_axes + grid
                              matching; like covered_axes it may be
                              narrower than the stub's dispatch chain
                              but never wider than intended coverage.

Emission cardinality: cpp_helpers once per file, cpp_dispatch_prelude
once per op, cpp_dispatch/cpp_launch once per precompile point. The
generated stub is::

    helpers | prelude -> [if (dispatch) { launch; return true; }]* -> return false

in the op's structured impl scope (outputs allocated by meta(), device
guard held). Dispatch conditions are evaluated ASSUMING the prelude
passed; locals the prelude declares are in scope for dispatch and
launch.

This module is stdlib-only: torchgen, the export tool, and the runtime
coverage layer all load it by file path.
"""

from __future__ import annotations

import importlib.util
import inspect
import os
from typing import Any, Protocol


class AotDeclaration(Protocol):
    ATEN_OP: str
    DISPATCH_KEY: str
    KERNEL_MODULE: str
    # Architectures this op's kernels are valid on (sm strings, e.g.
    # ("sm_90a", "sm_100a")). Optional in source declarations; the
    # validating loader defaults it to _DEFAULT_ARCHS and materializes
    # it, so loaded declarations always carry it (consumers read
    # d.ARCHS directly, never getattr). Export skips (declaration x
    # arch) pairs outside it; codegen emits a runtime gate from the
    # intersection of ARCHS with the arches actually shipped.
    ARCHS: tuple[str, ...]

    def kernel_precompile_grid(self) -> list[dict]: ...
    def covered_axes(self, *args: Any, **kwargs: Any) -> dict: ...
    def cpp_dispatch(self, spec: dict) -> str: ...
    def cpp_launch(self, spec: dict, launch_fn: str) -> str: ...


def decl_id(d: AotDeclaration) -> str:
    """C-identifier for a declaration: overload dots become underscores
    ("gt.Tensor" -> "gt_Tensor"). Names the DispatchStub
    (<id>_aot_stub), the generated kernel fn, the artifact directory,
    and the covers custom op (covers_<id>)."""
    return d.ATEN_OP.replace(".", "_")


_REQUIRED_CONSTS = ("ATEN_OP", "DISPATCH_KEY", "KERNEL_MODULE")
_REQUIRED_FNS = {
    # name -> takes_spec (the cardinality convention: spec-taking exports
    # render once per precompile point, no-arg exports once per op)
    "kernel_precompile_grid": False,
    "covered_axes": None,  # schema-shaped; arity not checked here
    "cpp_dispatch": True,
    "cpp_launch": True,
}
_OPTIONAL_FNS = {
    "cpp_dispatch_prelude": False,
    "cpp_helpers": False,
    "cpp_covers": False,
}

# Default ARCHS: every current kernel requires sm90+ features (TMA,
# clusters, cp.async.bulk); Blackwell variants included. Declarations
# override to narrow (e.g. a Blackwell-only kernel pins ("sm_100a",)).
_DEFAULT_ARCHS = ("sm_90", "sm_90a", "sm_100", "sm_100a", "sm_103", "sm_103a")

_SM_RE = r"sm_\d+a?"


def _load_by_path(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _check_arity(mod, name: str, takes_spec: bool, path: str) -> None:
    fn = getattr(mod, name)
    params = [
        p
        for p in inspect.signature(fn).parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    n = len(params)
    want = (1 if takes_spec else 0) + (1 if name == "cpp_launch" else 0)
    if n != want:
        kind = "per-point (spec-taking)" if takes_spec else "per-op (no-arg)"
        raise RuntimeError(
            f"{path}: {name} must be {kind}, expected {want} positional "
            f"parameter(s), got {n}"
        )


def _validate(d, path: str, label: str) -> None:
    for const in _REQUIRED_CONSTS:
        if not isinstance(getattr(d, const, None), str):
            raise RuntimeError(f"{path}: {label} missing or non-str constant {const}")

    for name, takes_spec in _REQUIRED_FNS.items():
        if not callable(getattr(d, name, None)):
            raise RuntimeError(f"{path}: {label} missing required function {name}()")
        if takes_spec is not None:
            _check_arity(d, name, takes_spec, path)
    for name, takes_spec in _OPTIONAL_FNS.items():
        if getattr(d, name, None) is not None:
            _check_arity(d, name, takes_spec, path)

    # Normalize ARCHS here, once: optional in the source module, always
    # present (as a tuple) on validated declarations. Everything
    # downstream reads d.ARCHS directly -- this is the only place
    # absence is legal.
    import re

    try:
        archs = d.ARCHS
    except AttributeError:
        archs = _DEFAULT_ARCHS
    if not archs or not all(
        isinstance(a, str) and re.fullmatch(_SM_RE, a) for a in archs
    ):
        raise RuntimeError(
            f"{path}: {label} ARCHS must be a non-empty sequence of sm "
            f"strings (e.g. ('sm_90a', 'sm_100a')), got {archs!r}"
        )
    d.ARCHS = tuple(archs)

    grid = d.kernel_precompile_grid()
    if not isinstance(grid, list) or not grid:
        raise RuntimeError(
            f"{path}: {label} kernel_precompile_grid() must return a non-empty list"
        )
    for point in grid:
        if not isinstance(point, dict):
            raise RuntimeError(
                f"{path}: {label} grid entries must be dicts, got {type(point)}"
            )


def load_declarations(path: str) -> list[AotDeclaration]:
    """Load and validate one aot.py: a single-op module (the module IS
    the declaration) or a family module (exports declarations() -> list
    of declaration objects). Raises RuntimeError naming the offending
    path/declaration on contract violations."""
    mod = _load_by_path(os.path.basename(os.path.dirname(path)) + "_aot", path)

    family = getattr(mod, "declarations", None)
    if family is not None:
        decls = family()
        if not isinstance(decls, list) or not decls:
            raise RuntimeError(f"{path}: declarations() must return a non-empty list")
        for i, d in enumerate(decls):
            _validate(d, path, f"declarations()[{i}] ({getattr(d, 'ATEN_OP', '?')}):")
        return decls

    _validate(mod, path, "")
    return [mod]


def load_declaration(path: str) -> AotDeclaration:
    """Single-declaration convenience: exactly one declaration expected."""
    decls = load_declarations(path)
    if len(decls) != 1:
        raise RuntimeError(f"{path}: expected a single declaration, got {len(decls)}")
    return decls[0]


def discover_declarations(ops_dir: str) -> dict[tuple[str, str], AotDeclaration]:
    """All (dispatch_key, op) -> declaration under ops_dir. Duplicate
    (op, key) pairs are an error."""
    decls: dict[tuple[str, str], AotDeclaration] = {}
    if not os.path.isdir(ops_dir):
        return decls
    for entry in sorted(os.listdir(ops_dir)):
        path = os.path.join(ops_dir, entry, "aot.py")
        if not os.path.exists(path):
            continue
        for d in load_declarations(path):
            key = (d.DISPATCH_KEY, d.ATEN_OP)
            if key in decls:
                raise RuntimeError(
                    f"{path}: duplicate declaration for {d.ATEN_OP}@{d.DISPATCH_KEY}"
                )
            decls[key] = d
    return decls
