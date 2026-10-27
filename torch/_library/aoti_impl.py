"""Prototype: ``torch.library.register_aot_impl``.

A builder registered here is called by AOTInductor at compile time with
FakeTensors (plus the op's scalar args) and an :class:`AOTTarget`. It returns
an artifact object -- currently only :class:`TVMFFIKernel` -- describing a
compiled kernel that AOTI embeds in the generated wrapper instead of falling
back to the proxy executor.

This module is the registry + artifact types only (no Inductor imports), so
``torch.library`` can expose it without creating an import cycle. The
compile-time resolution (shim codegen, linking) lives in
``torch._inductor.aoti_impl``.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union

from torch._ops import OpOverload


__all__ = [
    "AOTTarget",
    "TVMFFIKernel",
    "canonical_op_key",
    "register_aot_impl",
    "get_aot_impl",
]


@dataclass(frozen=True)
class AOTTarget:
    """Compile target handed to a ``register_aot_impl`` builder.

    The builder is called once per specialization (see
    ``torch._inductor.aoti_impl``). ``name`` is unique per specialization:
    export the kernel under it (CuTeDSL: ``export_to_c(...,
    function_name=target.name)``) so specializations don't collide at link
    time. ``workdir`` is a fresh directory per specialization.
    """

    arch: str  # e.g. "sm_90"
    workdir: Path
    name: str
    device: str = "cuda"


@dataclass(frozen=True)
class TVMFFIKernel:
    """A kernel compiled to the TVM-FFI ABI (what CuTeDSL ``export_to_c`` emits).

    ``objects`` are object files (``.o``) to link into the AOTI wrapper's
    companion library. ``symbol`` is the ``__tvm_ffi_<name>`` entry point.
    ``args`` is the packed-call order. Each entry is a schema argument name
    (a list arg expands to its elements, an absent ``Optional`` arg packs as
    None), ``out{i}`` for the i-th output (allocated by the generated shim
    from the fake outputs), or a literal ``None`` / bool / int / float for a
    slot the kernel expects but the op does not expose. If ``args`` is
    ``None``, the default is all tensor inputs, then outputs, then scalar
    inputs (the CuTeDSL ``export_to_c`` convention). A scalar schema argument
    left out of ``args`` is treated as a compile-time constant the builder
    baked into the kernel, so its value becomes part of the specialization.
    ``link_libs`` are extra shared libraries (full paths) the companion
    library links against, e.g. the DSL runtime.
    """

    objects: list[Union[str, Path]]
    symbol: str
    args: list[Union[str, None, bool, int, float]] | None = None
    link_libs: list[Union[str, Path]] = field(default_factory=list)

    def __post_init__(self):
        if not self.objects:
            raise ValueError("TVMFFIKernel requires at least one object file")
        for o in self.objects:
            if not isinstance(o, (str, Path)):
                raise TypeError(f"TVMFFIKernel objects must be str/Path, got {o!r}")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", self.symbol or ""):
            raise ValueError(
                f"TVMFFIKernel symbol must be a valid C identifier, got {self.symbol!r}"
            )
        if self.args is not None and not all(
            a is None or isinstance(a, (str, bool, int, float)) for a in self.args
        ):
            raise TypeError(
                "TVMFFIKernel args entries must be argument names (str) or "
                "None / bool / int / float literals"
            )


_AOT_IMPLS: dict[str, Callable[..., Any]] = {}


def canonical_op_key(op: Any) -> str:
    """Canonical registry key for an op: ``ns::op`` for the default overload,
    ``ns::op.<overload>`` otherwise.

    The same canonicalization is used for string registration
    (``"ns::op"``, ``"ns::op.default"``, ``"ns::op.foo"``), ``OpOverload``
    (whose ``name()`` keeps the overload suffix whenever it is non-empty),
    and ``CustomOpDef`` (always the default overload), so a registered
    builder can never silently miss because of an overload spelling.
    """
    if isinstance(op, str):
        name = op
        if "::" in name:
            ns, rest = name.split("::", 1)
            base, _, suffix = rest.partition(".")
            if suffix in ("", "default"):
                return f"{ns}::{base}"
            return f"{ns}::{base}.{suffix}"
        return name
    if isinstance(op, OpOverload):
        schema = op._schema
        overload = schema.overload_name
        if overload in ("", "default"):
            return schema.name
        return f"{schema.name}.{overload}"
    # CustomOpDef
    qualname = getattr(op, "_qualname", None)
    if isinstance(qualname, str):
        return qualname
    raise TypeError(f"Cannot identify op from {type(op)}")


def register_aot_impl(op: Any, func: Callable[..., Any] | None = None, /):
    """Register an AOTI builder for a custom op.

    Usage::

        @torch.library.register_aot_impl("mylib::scale")
        def scale_aot(x, alpha, *, target):
            ...
            return TVMFFIKernel(objects=[...], symbol="__tvm_ffi_scale")
    """

    def _register(fn: Callable[..., Any]) -> Callable[..., Any]:
        key = canonical_op_key(op)
        if key in _AOT_IMPLS:
            raise RuntimeError(f"register_aot_impl: {key} already has an AOT impl")
        _AOT_IMPLS[key] = fn
        return fn

    if func is not None:
        return _register(func)
    return _register


def get_aot_impl(op_name: str) -> Callable[..., Any] | None:
    return _AOT_IMPLS.get(op_name)
