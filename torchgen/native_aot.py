"""Discovery and validation of native-AOT declarations.

Ops under torch/_native/ops/<op>/aot.py declare AOT-compiled DSL kernels embedded
into the op's ATen implementation; the contract is tools/native_aot/decl.py and the
validating loader torchgen/native_aot_decl.py. torchgen consumes the declarations to
generate NativeAotStubs.h -- one at::native DispatchStub per declared op,
signature-matched to the structured impl, with no kernel registered by default -- and
to emit a stub consultation between op.meta() and op.impl() in the generated wrapper.
Unstructured functional declarations instead return allocated tensors through an
output parameter, before the wrapper calls the ordinary backend implementation.

Only the identity torchgen needs is modeled here: the precompile grid belongs to the
export tool and to torch._native.aot_manifest, the C++-generating hooks to
gen_aot_lib.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

from torchgen.model import BaseTy, BaseType, DispatchKey
from torchgen.native_aot_decl import decl_id_for_op as _decl_id, discover_declarations


if TYPE_CHECKING:
    from collections.abc import Sequence

    from torchgen.api.types import Binding
    from torchgen.model import NativeFunction, NativeFunctionsGroup


@dataclass(frozen=True)
class NativeAotManifest:
    # Base name when it resolves to exactly one structured group, or
    # overload-qualified when overloads have separate structured groups.
    op: str
    dispatch_key: DispatchKey
    # Declared UNCONDITIONAL: these kernels are the implementation, so the
    # consultation gates on the private mask instead of the user-facing
    # switch (see gen_stub_consultation).
    unconditional: bool = False
    structured: bool = True
    exact: bool = False

    @property
    def decl_id(self) -> str:
        return _decl_id(self.op)

    def stub_name(self) -> str:
        return f"{self.decl_id}_aot_stub"

    def fn_type_name(self) -> str:
        return f"{self.decl_id}_aot_fn"

    def matches_group(self, g: NativeFunctionsGroup) -> bool:
        r"""Match an exact functional overload or a validated unique structured base."""
        if not self.structured or not g.structured:
            return False
        if "." in self.op or self.exact:
            return self.op == str(g.functional.func.name)
        return self.op == g.functional.func.name.name.base


def is_unconditional(d: object) -> bool:
    """Whether a declaration's kernels ARE the op's implementation rather than a
    faster route to the same answer (``UNCONDITIONAL``, default False). Such an op's
    gate reads the private mask instead of the user-facing switch, so nothing a caller
    can reach turns it off.

    Lives here because torchgen is the only consumer: export and gen_aot_lib build the
    same kernels either way. Read through this accessor, since the attribute is
    optional, and a non-bool is rejected rather than taken for its truthiness: the flag
    decides whether a user can switch the op off at all."""
    declared = getattr(d, "UNCONDITIONAL", False)
    if not isinstance(declared, bool):
        raise RuntimeError(
            f"{getattr(d, '__file__', getattr(d, 'ATEN_OP', '?'))}: "
            f"UNCONDITIONAL must be a bool, got {declared!r}"
        )
    return declared


def parse_native_aot_manifests(
    ops_dir: str,
) -> dict[tuple[DispatchKey, str], NativeAotManifest]:
    """Discover and validate all aot.py declarations under ops_dir
    (torch/_native/ops); see tools/native_aot/decl.py for the contract."""
    manifests: dict[tuple[DispatchKey, str], NativeAotManifest] = {}
    if not os.path.isdir(ops_dir):
        return manifests
    # STRUCTURED picks the wrapper ABI; UNCONDITIONAL picks its enablement gate.
    # The rest of each declaration is for the export tool and gen_aot_lib.
    for (key_str, op), d in discover_declarations(ops_dir).items():
        key = DispatchKey.parse(key_str)
        structured = getattr(d, "STRUCTURED", True)
        if not isinstance(structured, bool):
            raise RuntimeError(f"{op}: STRUCTURED must be a bool")
        exact = getattr(d, "EXACT", False)
        if not isinstance(exact, bool):
            raise RuntimeError(f"{op}: EXACT must be a bool")
        manifests[(key, op)] = NativeAotManifest(
            op=op,
            dispatch_key=key,
            unconditional=is_unconditional(d),
            structured=structured,
            exact=exact,
        )
    return manifests


def _impl_bindings(g: NativeFunctionsGroup) -> list[Binding]:
    import torchgen.api.structured as structured
    from torchgen.context import native_function_manager

    with native_function_manager(g):
        return structured.impl_arguments(g)


def functional_stub_params(f: NativeFunction) -> str:
    from torchgen.api.types import DispatcherSignature
    from torchgen.context import native_function_manager

    with native_function_manager(f):
        sig = DispatcherSignature.from_schema(f.func)
        params = [b.decl() for b in sig.arguments()]
        result_type = sig.returns_type().cpp_type(strip_ref=True)
        params.append(f"{result_type}& aot_result")
    return ", ".join(params)


def gen_stub_declaration(
    m: NativeAotManifest, g: NativeFunction | NativeFunctionsGroup
) -> str:
    from torchgen.model import NativeFunctionsGroup

    if isinstance(g, NativeFunctionsGroup):
        params = ", ".join(b.decl() for b in _impl_bindings(g))
    else:
        params = functional_stub_params(g)
    return f"""\
using {m.fn_type_name()} = bool (*)({params});
DECLARE_DISPATCH({m.fn_type_name()}, {m.stub_name()})
"""


def gen_stub_definition(m: NativeAotManifest) -> str:
    return f"""\
DEFINE_DISPATCH({m.stub_name()});
REGISTER_NO_CPU_DISPATCH({m.stub_name()})
"""


def gen_stub_consultation(
    m: NativeAotManifest,
    impl_exprs: str,
    *,
    returns_type: str | None = None,
    return_arg: str | None = None,
    tensor_arg: str | None = None,
    device_guarded: bool = False,
) -> str:
    r"""Emit a gated AOT launch before the backend fallback.
    Unconditional declarations use the private mask instead of allowNativeAot().
    """
    device_type = f"c10::DeviceType::{m.dispatch_key}"
    stub = f"at::native::{m.stub_name()}"
    gate = (
        "!at::globalContext().maskUnconditionalNativeAot()"
        if m.unconditional
        else "at::globalContext().allowNativeAot()"
    )
    if not m.structured:
        if returns_type is None:
            raise AssertionError("functional native-AOT hooks require a return type")
        result = return_arg or "aot_result"
        args = ", ".join(filter(None, (impl_exprs, result)))
        initialize = "" if return_arg else f"{returns_type} aot_result;"
        device = (
            f"{tensor_arg}.device().type() == {device_type} && " if tensor_arg else ""
        )
        guard = (
            f"const c10::OptionalDeviceGuard aot_guard({tensor_arg}.device());"
            if tensor_arg and not device_guarded
            else ""
        )
        return f"""
  if ({device}{gate} && {stub}.is_device_supported({device_type})) {{
    {guard}
    {initialize}
    if ({stub}({device_type}, {args})) {{
      return {result};
    }}
  }}
"""
    comment = (
        "// native-AOT: the last conjunct is the LAUNCH into meta outputs; false calls op.impl.\n"
        "// declared UNCONDITIONAL: private mask only; && short-circuits unsupported devices.\n"
        if m.unconditional
        else "// native-AOT: the last conjunct is the LAUNCH into meta outputs; false calls op.impl.\n"
        "// && short-circuits disabled AOT and unsupported devices.\n"
    )
    return (
        f"{comment}"
        f"if (!({gate} && "
        f"{stub}.is_device_supported({device_type}) && "
        f"{stub}({device_type}, {impl_exprs}))) {{ op.impl({impl_exprs}); }}"
    )


def first_tensor_arg(f: NativeFunction) -> str | None:
    """Choose a required Tensor for unstructured device checks and guards."""
    return next(
        (
            a.name
            for a in f.func.arguments.flat_non_out
            if a.type == BaseType(BaseTy.Tensor)
        ),
        None,
    )


def validate_native_aot_manifests(
    manifests: dict[tuple[DispatchKey, str], NativeAotManifest],
    grouped_native_functions: Sequence[NativeFunction | NativeFunctionsGroup],
) -> None:
    r"""Validate unique or EXACT structured names and exact unstructured names.
    Unstructured hooks require fresh Tensor returns or an inplace self alias.
    """
    from collections import defaultdict

    from torchgen.model import BaseTy, BaseType, NativeFunctionsGroup, SchemaKind

    structured_by_base: dict[str, list[str]] = defaultdict(list)
    functional_by_name: dict[str, NativeFunction] = {}
    for g in grouped_native_functions:
        if isinstance(g, NativeFunctionsGroup) and g.structured:
            structured_by_base[g.functional.func.name.name.base].append(
                str(g.functional.func.name)
            )
        else:
            for f in g.functions() if isinstance(g, NativeFunctionsGroup) else [g]:
                functional_by_name[str(f.func.name)] = f
    for (key, op), manifest in manifests.items():
        if not manifest.structured:
            f = functional_by_name.get(op)
            fresh = (
                f is not None
                and f.func.kind() == SchemaKind.functional
                and bool(f.func.returns)
                and all(
                    r.type == BaseType(BaseTy.Tensor) and r.annotation is None
                    for r in f.func.returns
                )
            )
            # FunctionSchema validates inplace self mutation and return aliasing.
            mutable = (
                f is not None
                and f.func.kind() == SchemaKind.inplace
                and bool(f.func.returns)
                and f.func.returns[0].type == BaseType(BaseTy.Tensor)
            )
            if not fresh and not mutable:
                raise RuntimeError(
                    f"native-aot declaration for {op}@{key}: STRUCTURED=False "
                    "requires fresh Tensor returns or an inplace return aliasing self"
                )
            if f is None or first_tensor_arg(f) is None:
                raise RuntimeError(
                    f"native-aot declaration for {op}@{key}: unstructured hooks "
                    "require a non-optional Tensor argument for the device guard"
                )
            continue
        base = op.split(".")[0]
        names = structured_by_base.get(base, [])
        if "." in op or manifest.exact:
            if op not in names:
                raise RuntimeError(
                    f"native-aot declaration for {op}@{key}: no structured "
                    f"group named {op!r} in native_functions.yaml "
                    f"(structured overloads of {base!r}: {names or 'none'})"
                )
        elif not names:
            raise RuntimeError(
                f"native-aot declaration for {op}@{key}: {op} is not a "
                f"structured op in native_functions.yaml. Use STRUCTURED=False "
                f"for an exact functional overload returning fresh tensors."
            )
        elif len(names) > 1:
            raise RuntimeError(
                f"native-aot declaration for {op}@{key}: base name {op!r} is "
                f"ambiguous across structured groups {names}; qualify "
                f"ATEN_OP with the overload (e.g. {names[0]!r})"
            )
