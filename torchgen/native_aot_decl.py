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
import re
from typing import Any, Protocol


class AotDeclaration(Protocol):
    ATEN_OP: str
    DISPATCH_KEY: str
    KERNEL_MODULE: str
    # Architectures this op's kernels are valid on (sm strings, e.g.
    # ("sm_90a", "sm_100a")). OPTIONAL in source declarations, so read it
    # through archs_of(d), never d.ARCHS. Export skips (declaration x
    # arch) pairs outside it; gen_aot_lib emits a runtime gate from the
    # intersection with the arches actually shipped.
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
    # name -> positional arity. That arity IS the cardinality convention:
    # spec-taking exports render once per precompile point, no-arg exports
    # once per op. cpp_launch also takes the launch_fn name.
    "kernel_precompile_grid": 0,
    "covered_axes": None,  # schema-shaped; arity not checked here
    "cpp_dispatch": 1,
    "cpp_launch": 2,
}
_OPTIONAL_FNS = {
    "cpp_dispatch_prelude": 0,
    "cpp_helpers": 0,
    "cpp_covers": 0,
}

# Every sm spelling this tooling can parse and target. export checks an explicit
# --arch against it before touching the disk, so a typo bails out naming the set
# instead of matching no declaration and exporting nothing at exit 0. Deliberately
# WIDER than EXPORTABLE_ARCHES below: an explicit --arch is how a hand run targets
# something the release wheels do not.
KNOWN_ARCHES = ("sm_90", "sm_90a", "sm_100", "sm_100a", "sm_103", "sm_103a")

# Which of them the STANDARD build ships: the TORCH_CUDA_ARCH_LIST entries eligible
# on the automatic export path, which an explicit --arch bypasses. Both spellings of
# a capability are listed because they are distinct nvcc targets for the same
# hardware -- "10.0a" (needed by tcgen05/wgmma) in b200-native-aot.yml, plain "10.0"
# in the manywheel lists -- and omitting either silently exports nothing there. Each
# entry also costs another full set of compiled kernels in every wheel naming it, and
# makes the DSL runtimes mandatory on a builder with that GPU. sm_103 stays out: no
# arch list we see names 10.3.
EXPORTABLE_ARCHES = ("sm_90", "sm_90a", "sm_100", "sm_100a")
if not set(EXPORTABLE_ARCHES) <= set(KNOWN_ARCHES):
    raise AssertionError(
        f"EXPORTABLE_ARCHES names arches this tooling cannot target: "
        f"{sorted(set(EXPORTABLE_ARCHES) - set(KNOWN_ARCHES))}"
    )

# Default ARCHS: every current kernel requires sm90+ features (TMA,
# clusters, cp.async.bulk); Blackwell variants included. Declarations
# override to narrow (e.g. a Blackwell-only kernel pins ("sm_100a",)).
# The same tuple as KNOWN_ARCHES today, named separately because "the tooling can
# target this arch" is not "every declaration's kernels work on it".
_DEFAULT_ARCHS = KNOWN_ARCHES

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


def archs_of(d: AotDeclaration) -> tuple[str, ...]:
    """The declaration's architectures, defaulted when it omits ARCHS.

    Read this instead of d.ARCHS: ARCHS is optional in a source
    declaration and the loader validates it without writing it back."""
    return tuple(getattr(d, "ARCHS", _DEFAULT_ARCHS))


# Majors a generated gate could plausibly match. Narrow on purpose: see cc_of.
_KNOWN_MAJORS = range(3, 13)

# The whole spelling in one pattern, digits included, rather than stripping the
# prefix and suffix and re-testing the middle. ASCII classes, not \d or
# str.isdigit(): both are Unicode-aware, and full-width or Arabic-Indic digits
# then read as an ordinary capability.
_SM_SPELLING = re.compile(r"sm_([1-9][0-9]{1,2})a?")


def cc_of(arch: str) -> tuple[int, int]:
    """sm string -> compute capability. "sm_90" -> (9, 0), "sm_103a" -> (10, 3).

    Shared, because the exporter (matching a detected arch against ARCHS) and the
    generator (grouping sidecars by capability) must agree what an sm string
    means: they disagreed while one compared capabilities and the other strings,
    and a declaration pinning ('sm_100a',) disowned the 'sm_100' its own on-device
    export produced.

    Refuses what it cannot parse rather than computing a capability: "sm_9" gives
    (0, 9) and "sm_1000" (100, 0), each a gate no device satisfies, so the op
    ships, links and declines every call unreported. Suffixes other than the
    arch-conditional "a" (CUDA 12.9+'s family-conditional "f") are refused too --
    they mean something the generator has not been taught.

    _KNOWN_MAJORS would reject "sm_9" and "sm_1000" anyway (as capability 0.9 and
    100.0), so the digit count in _SM_SPELLING is there for the DIAGNOSTIC: a
    malformed string should be reported as unreadable, not as hardware that does
    not exist."""
    m = _SM_SPELLING.fullmatch(arch)
    if m is None:
        raise RuntimeError(
            f"cannot read a compute capability from arch {arch!r}: expected "
            f"sm_<major><minor>[a], e.g. sm_90a or sm_100"
        )
    major, minor = divmod(int(m.group(1)), 10)
    if major not in _KNOWN_MAJORS:
        raise RuntimeError(
            f"arch {arch!r} parses as compute capability {major}.{minor}, "
            f"outside the known range {_KNOWN_MAJORS.start}-"
            f"{_KNOWN_MAJORS.stop - 1}; a gate for it would match no device"
        )
    return major, minor


def _check_arity(mod, name: str, want: int, path: str) -> None:
    fn = getattr(mod, name)
    params = [
        p
        for p in inspect.signature(fn).parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    n = len(params)
    if n != want:
        kind = "per-op (no-arg)" if want == 0 else "per-point (spec-taking)"
        raise RuntimeError(
            f"{path}: {name} must be {kind}, expected {want} positional "
            f"parameter(s), got {n}"
        )


def _validate(d, path: str, label: str) -> None:
    for const in _REQUIRED_CONSTS:
        if not isinstance(getattr(d, const, None), str):
            raise RuntimeError(f"{path}: {label} missing or non-str constant {const}")

    for name, arity in _REQUIRED_FNS.items():
        if not callable(getattr(d, name, None)):
            raise RuntimeError(f"{path}: {label} missing required function {name}()")
        if arity is not None:
            _check_arity(d, name, arity, path)
    for name, arity in _OPTIONAL_FNS.items():
        if getattr(d, name, None) is not None:
            _check_arity(d, name, arity, path)

    # Validate ARCHS but do NOT write it back: a validating loader that
    # mutates its input leaves the source module and the declaration
    # disagreeing. archs_of() applies the default at every read instead.
    archs = archs_of(d)
    if not archs or not all(
        isinstance(a, str) and re.fullmatch(_SM_RE, a) for a in archs
    ):
        raise RuntimeError(
            f"{path}: {label} ARCHS must be a non-empty sequence of sm "
            f"strings (e.g. ('sm_90a', 'sm_100a')), got {archs!r}"
        )
    # ...and each one must name a capability, not merely look like an sm string.
    # _SM_RE accepts "sm_9" and "sm_1000", which cc_of refuses -- and export
    # compares ARCHS entries by STRING, so a typo silently matched nothing: the
    # declaration exported no kernels, generation had no tree to complain about,
    # and the build shipped without that op, green. Refused here because this is
    # the only place that knows which file to name.
    for a in archs:
        try:
            cc_of(a)
        except RuntimeError as e:
            raise RuntimeError(f"{path}: {label} ARCHS entry {a!r}: {e}") from e

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
