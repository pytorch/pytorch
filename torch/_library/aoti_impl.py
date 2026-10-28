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
from torch.utils._exposed_in import exposed_in


__all__ = [
    "AOTTarget",
    "TVMFFIKernel",
    "canonical_op_key",
    "register_aot_impl",
    "get_aot_impl",
]


@exposed_in("torch.library")
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


@exposed_in("torch.library")
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
        # Without objects the link still succeeds (shared libraries may have
        # undefined symbols) and the failure only shows up at load time.
        if not self.objects:
            raise ValueError("TVMFFIKernel requires at least one object file")
        # The symbol is pasted into generated C++.
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", self.symbol or ""):
            raise ValueError(
                f"TVMFFIKernel symbol must be a valid C identifier, got {self.symbol!r}"
            )


_AOT_IMPLS: dict[str, Callable[..., Any]] = {}


def canonical_op_key(op: Any) -> str:
    """Registry key: ``ns::op`` for an op's default overload (however it is
    spelled: ``"ns::op"``, ``"ns::op.default"``, an ``OpOverload`` or a
    ``CustomOpDef``), ``ns::op.<overload>`` for any other overload."""
    if isinstance(op, str):
        name = op
    elif isinstance(op, OpOverload):
        name = op.name()
    elif isinstance(getattr(op, "_qualname", None), str):  # CustomOpDef
        name = op._qualname
    else:
        raise TypeError(f"Cannot identify op from {type(op)}")
    return name.removesuffix(".default")


@exposed_in("torch.library")
def register_aot_impl(op: Any, func: Callable[..., Any] | None = None, /):
    """Register an AOTI builder for a custom op (prototype).

    The builder receives every schema argument positionally (keyword-only
    ones included, defaults filled in), tensors as FakeTensors, plus
    ``target=`` (an :class:`AOTTarget`), and returns a :class:`TVMFFIKernel`.
    Only an op's default overload can be registered.

    Usage::

        @torch.library.register_aot_impl("mylib::scale")
        def scale_aot(x, alpha, *, target):
            ...
            return TVMFFIKernel(objects=[...], symbol="__tvm_ffi_scale")
    """

    def _register(fn: Callable[..., Any]) -> Callable[..., Any]:
        key = canonical_op_key(op)
        if "." in key.partition("::")[2]:
            raise NotImplementedError(
                f"register_aot_impl: {key} is not a default overload; the AOTI "
                "wrapper names C shims without an overload suffix, so only "
                "default overloads are supported"
            )
        if key in _AOT_IMPLS:
            raise RuntimeError(f"register_aot_impl: {key} already has an AOT impl")
        _AOT_IMPLS[key] = fn
        return fn

    if func is not None:
        return _register(func)
    return _register


def get_aot_impl(op_name: str) -> Callable[..., Any] | None:
    return _AOT_IMPLS.get(op_name)
