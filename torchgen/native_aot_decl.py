"""Validating loader and identity helpers for native-AOT declarations.

The MECHANISM half of the declaration contract; the contract itself is
documented in tools/native_aot/decl.py, which re-exports this module.
Read that file to write a declaration, this one to change how they load.

Lives in torchgen, not tools/, because the wheel ships torchgen but not
tools/ -- and installed torchgen must load declarations out of tree
(`python -m torchgen.gen --source-path <site-packages>/torchgen/packaged
/ATen`), which a tools-side home broke with FileNotFoundError. It cannot
live under torch/ either: torchgen must not import torch, since stage-1
codegen runs before torch is built.

Torch-free for the same reason. Note that is the real constraint on
declaration modules too -- their module scope may import torchgen (pure
Python, no torch dependency), just not torch.
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
    # arch) pairs outside it; gen_aot_lib emits a runtime gate from the
    # intersection of ARCHS with the arches actually shipped.
    ARCHS: tuple[str, ...]

    def kernel_precompile_grid(self) -> list[dict]: ...
    def covered_axes(self, *args: Any, **kwargs: Any) -> dict: ...
    def cpp_dispatch(self, spec: dict) -> str: ...
    def cpp_launch(self, spec: dict, launch_fn: str) -> str: ...


def decl_id_for_op(aten_op: str) -> str:
    """C-identifier for an op name: overload dots become underscores
    ("gt.Tensor" -> "gt_Tensor"). Names the DispatchStub, the generated
    kernel fn, the artifact directory and the covers op.

    One rule, one home: the runtime resolving covers_<id>, the C++
    generator emitting its schema, and torchgen naming the stub must
    agree exactly or the coverage fast path silently misses.
    """
    return aten_op.replace(".", "_")


def decl_id(d: AotDeclaration) -> str:
    """decl_id_for_op for a loaded declaration."""
    return decl_id_for_op(d.ATEN_OP)


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


def load_by_path(name: str, path: str):
    """Import a module from an explicit file path.

    THE canonical copy for native-AOT code: torchgen ships in the wheel,
    so tools/ and torch/ can both reach it. (tools/shared/module_loader.py
    looks equivalent, but tools/ is not a wheel package, so neither
    installed torchgen nor torch can import it.)

    By path rather than by import because declaration modules must load
    where their package is not importable: torchgen reads
    torch/_native/ops/*/aot.py during stage 1, before torch is built.
    """
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

    archs = getattr(d, "ARCHS", _DEFAULT_ARCHS)
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
    mod = load_by_path(os.path.basename(os.path.dirname(path)) + "_aot", path)

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
