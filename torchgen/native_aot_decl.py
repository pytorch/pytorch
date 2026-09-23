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
    # Canonical compile targets this op supports (sm strings, e.g.
    # ("sm_90", "sm_100f")). The exporter chooses the widest target compatible
    # with each device architecture supported by the containing build, preferring
    # an f target when coverage is equal; gen_aot_lib gates on shipped artifacts.
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


# An sm_XYf target covers known devices with major X and minor >= Y. Keep the
# target-to-device mapping explicit so runtime-only devices remain represented.
FAMILY_TARGET_DEVICES = {
    "sm_100f": ((10, 0), (10, 3), (10, 7)),
    "sm_103f": ((10, 3), (10, 7)),
    "sm_110f": ((11, 0),),
    "sm_120f": ((12, 0), (12, 1)),
    "sm_121f": ((12, 1),),
}

# Device capabilities understood by selection and generated routing. Some are
# runtime-only: they need not be compiler targets when an earlier family target
# covers them.
KNOWN_DEVICE_CAPABILITIES = (
    (9, 0),
    (10, 0),
    (10, 3),
    (10, 7),
    (11, 0),
    (12, 0),
    (12, 1),
)

# Compiler targets declarations may offer. Runtime-only family members need not
# appear here; a build architecture selects among an op's declared targets.
KNOWN_ARCHES = (
    "sm_90",
    "sm_90a",
    "sm_100",
    "sm_100f",
    "sm_100a",
    "sm_103",
    "sm_103f",
    "sm_103a",
    "sm_110",
    "sm_110f",
    "sm_110a",
    "sm_120",
    "sm_120f",
    "sm_120a",
    "sm_121",
    "sm_121f",
    "sm_121a",
)

_SM_RE = r"sm_\d+[af]?"


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
    """The declaration's validated compiler targets."""
    return tuple(d.ARCHS)


# Majors a generated gate could plausibly match. Narrow on purpose: see cc_of.
_KNOWN_MAJORS = range(3, 13)

# The whole spelling in one pattern, digits included, rather than stripping the
# prefix and suffix and re-testing the middle. ASCII classes, not \d or
# str.isdigit(): both are Unicode-aware, and full-width or Arabic-Indic digits
# then read as an ordinary capability.
_SM_SPELLING = re.compile(r"sm_([1-9][0-9]{1,2})([af]?)")


def _parse_arch(arch: str) -> tuple[tuple[int, int], str]:
    """Validated ``sm`` spelling -> (compute capability, feature suffix)."""
    m = _SM_SPELLING.fullmatch(arch)
    if m is None:
        raise RuntimeError(
            f"cannot read a compute capability from arch {arch!r}: expected "
            f"sm_<major><minor>[a|f], e.g. sm_90a, sm_100f or sm_100"
        )
    major, minor = divmod(int(m.group(1)), 10)
    if major not in _KNOWN_MAJORS:
        raise RuntimeError(
            f"arch {arch!r} parses as compute capability {major}.{minor}, "
            f"outside the known range {_KNOWN_MAJORS.start}-"
            f"{_KNOWN_MAJORS.stop - 1}; a gate for it would match no device"
        )
    return (major, minor), m.group(2)


def cc_of(arch: str) -> tuple[int, int]:
    """Return ``(major, minor)`` for a validated sm target."""
    return _parse_arch(arch)[0]


def suffix_of(arch: str) -> str:
    """The CUDA feature-set suffix ("a", "f", or "") of a validated target."""
    return _parse_arch(arch)[1]


def target_devices(target: str) -> tuple[tuple[int, int], ...]:
    """Known device capabilities on which ``target`` can run."""
    target_cc, suffix = _parse_arch(target)
    if suffix == "a":
        return (target_cc,)
    if suffix == "f":
        devices = FAMILY_TARGET_DEVICES.get(target)
        if devices is None:
            raise RuntimeError(
                f"family-specific target {target} has no entry in FAMILY_TARGET_DEVICES"
            )
        return devices
    major, minor = target_cc
    return tuple(
        cc for cc in KNOWN_DEVICE_CAPABILITIES if cc[0] == major and cc[1] >= minor
    )


def target_can_run_on(target: str, device_cc: tuple[int, int]) -> bool:
    """Whether ``target``'s cubin can run on a device compute capability."""
    return device_cc in target_devices(target)


_SUFFIX_STRENGTH = {"": 0, "f": 1, "a": 2}
_WIDEST_SUFFIX_ORDER = {"f": 0, "": 1, "a": 2}


def target_order_key(target: str) -> tuple[int, int, int, int]:
    """Static dispatch order: narrower coverage and stronger targets first."""
    major, minor = cc_of(target)
    return (
        len(target_devices(target)),
        major,
        -minor,
        -_SUFFIX_STRENGTH[suffix_of(target)],
    )


def compatible_targets(
    targets: tuple[str, ...] | list[str], device_cc: tuple[int, int]
) -> tuple[str, ...]:
    """Compatible targets in runtime dispatch preference order."""
    return tuple(
        sorted(
            (target for target in targets if target_can_run_on(target, device_cc)),
            key=target_order_key,
        )
    )


def widest_compatible_target(
    targets: tuple[str, ...] | list[str], device_cc: tuple[int, int]
) -> str | None:
    """The broadest declared target for ``device_cc``, preferring f on a tie."""
    compatible = [target for target in targets if target_can_run_on(target, device_cc)]
    if not compatible:
        return None
    return min(
        compatible,
        key=lambda target: (
            -len(target_devices(target)),
            _WIDEST_SUFFIX_ORDER[suffix_of(target)],
            *cc_of(target),
        ),
    )


def known_device_capabilities() -> frozenset[tuple[int, int]]:
    """Device capabilities understood by native-AOT selection and routing."""
    return frozenset(KNOWN_DEVICE_CAPABILITIES)


def _validate_family_targets() -> None:
    for target, devices in FAMILY_TARGET_DEVICES.items():
        if suffix_of(target) != "f":
            raise AssertionError(f"{target}: family target must have an f suffix")
        if not devices or len(set(devices)) != len(devices):
            raise AssertionError(f"{target}: devices must be non-empty and unique")
        if devices[0] != cc_of(target):
            raise AssertionError(
                f"{target}: first device must match the target's compute capability"
            )
        target_major, target_minor = cc_of(target)
        if any(
            major != target_major or minor < target_minor for major, minor in devices
        ):
            raise AssertionError(
                f"{target}: devices must have major {target_major} and minor >= "
                f"{target_minor}"
            )
        if any(device not in KNOWN_DEVICE_CAPABILITIES for device in devices):
            raise AssertionError(f"{target}: devices must all be known capabilities")
    for target in KNOWN_ARCHES:
        target_devices(target)


_validate_family_targets()


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
    if not hasattr(d, "ARCHS"):
        raise RuntimeError(f"{path}: {label} missing required constant ARCHS")

    for name, arity in _REQUIRED_FNS.items():
        if not callable(getattr(d, name, None)):
            raise RuntimeError(f"{path}: {label} missing required function {name}()")
        if arity is not None:
            _check_arity(d, name, arity, path)
    for name, arity in _OPTIONAL_FNS.items():
        if getattr(d, name, None) is not None:
            _check_arity(d, name, arity, path)

    archs = archs_of(d)
    if not archs or not all(
        isinstance(a, str) and re.fullmatch(_SM_RE, a) for a in archs
    ):
        raise RuntimeError(
            f"{path}: {label} ARCHS must be a non-empty sequence of sm "
            f"strings (e.g. ('sm_90', 'sm_100f')), got {archs!r}"
        )
    # ...and each one must name a target with known device coverage, not merely look
    # like an sm string. _SM_RE accepts "sm_9" and "sm_1000", which cc_of refuses.
    # Refused here because this is the only place that knows which file to name.
    for a in archs:
        try:
            target_devices(a)
        except RuntimeError as e:
            raise RuntimeError(f"{path}: {label} ARCHS entry {a!r}: {e}") from e
        if a not in KNOWN_ARCHES:
            raise RuntimeError(
                f"{path}: {label} ARCHS entry {a!r} is not a known compiler target"
            )

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
