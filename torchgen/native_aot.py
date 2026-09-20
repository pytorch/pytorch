"""Discovery and validation of native-AOT declarations.

Ops under torch/_native/ops/<op>/aot.py declare AOT-compiled DSL kernels embedded
into the op's ATen implementation; the contract is tools/native_aot/decl.py and the
validating loader torchgen/native_aot_decl.py. torchgen consumes the declarations to
generate NativeAotStubs.h -- one at::native DispatchStub per declared op,
signature-matched to the structured impl, with no kernel registered by default -- and
to emit a stub consultation between op.meta() and op.impl() in the generated wrapper.

Only the identity torchgen needs is modeled here: the precompile grid belongs to the
export tool and to torch._native.aot_manifest, the C++-generating hooks to
gen_aot_lib.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

from torchgen.model import DispatchKey
from torchgen.native_aot_decl import decl_id_for_op as _decl_id, discover_declarations


if TYPE_CHECKING:
    from collections.abc import Sequence

    from torchgen.model import NativeFunction, NativeFunctionsGroup


@dataclass(frozen=True)
class NativeAotManifest:
    # Base name ("topk") when the base resolves to exactly one structured
    # group, or overload-qualified ("gt.Tensor") when overloads have
    # separate structured groups.
    op: str
    dispatch_key: DispatchKey
    # Declared UNCONDITIONAL: these kernels are the implementation, so the
    # consultation gates on the private mask instead of the user-facing
    # switch (see gen_stub_consultation).
    unconditional: bool = False

    @property
    def decl_id(self) -> str:
        return _decl_id(self.op)

    def stub_name(self) -> str:
        return f"{self.decl_id}_aot_stub"

    def fn_type_name(self) -> str:
        return f"{self.decl_id}_aot_fn"

    def matches_group(self, g: NativeFunctionsGroup) -> bool:
        """Does this manifest target group g? A qualified op matches the exact
        functional overload name; a base name matches the group's base, whose
        uniqueness validate_native_aot_manifests checks."""
        if "." in self.op:
            return self.op == str(g.functional.func.name)
        return self.op == g.functional.func.name.name.base


def is_unconditional(d) -> bool:
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
    # Beyond the (dispatch_key, op) keys, only UNCONDITIONAL is read here --
    # it picks the gate in the generated wrapper. The rest of each
    # declaration is for the export tool and gen_aot_lib.
    for (key_str, op), d in discover_declarations(ops_dir).items():
        key = DispatchKey.parse(key_str)
        manifests[(key, op)] = NativeAotManifest(
            op=op, dispatch_key=key, unconditional=is_unconditional(d)
        )
    return manifests


def _impl_bindings(g: NativeFunctionsGroup) -> list:
    import torchgen.api.structured as structured
    from torchgen.context import native_function_manager

    with native_function_manager(g):
        return structured.impl_arguments(g)


def gen_stub_declaration(m: NativeAotManifest, g: NativeFunctionsGroup) -> str:
    bindings = _impl_bindings(g)
    params = ", ".join(b.decl() for b in bindings)
    return f"""\
using {m.fn_type_name()} = bool (*)({params});
DECLARE_DISPATCH({m.fn_type_name()}, {m.stub_name()})
"""


def gen_stub_definition(m: NativeAotManifest) -> str:
    return f"""\
DEFINE_DISPATCH({m.stub_name()});
REGISTER_NO_CPU_DISPATCH({m.stub_name()})
"""


def gen_stub_consultation(m: NativeAotManifest, impl_exprs: str) -> str:
    """The structured-wrapper call site. The stub has no kernel unless the AOT
    library registered one, and a Context switch gates the whole path; a true return
    means the AOT kernel filled the meta()-allocated outputs and op.impl is skipped.

    Which switch depends on the declaration: an ordinary op reads allowNativeAot(),
    the user-facing off switch, while an UNCONDITIONAL op reads
    maskUnconditionalNativeAot(), since its kernels are the implementation and the
    user-facing switch must not reach them. A gate still has to exist for the private
    hatch that obtains stock aten values.

    The emitted comment states that the last conjunct LAUNCHES the kernel, which the
    call site alone does not show."""
    device_type = f"c10::DeviceType::{m.dispatch_key}"
    stub = f"at::native::{m.stub_name()}"
    if m.unconditional:
        gate = "!at::globalContext().maskUnconditionalNativeAot()"
        # The user-facing switch does NOT reach this op, so the shared comment's
        # "switched off" case would describe a route that does not exist here.
        gate_comment = (
            "// declared UNCONDITIONAL: these kernels are the implementation, so\n"
            "// torch._native.set_aot_enabled(False) does NOT mask them -- only the\n"
            "// private reference-computation hatch does. op.impl runs when that hatch\n"
            "// is set, when the device is unsupported, or when the kernels decline\n"
            "// this shape.\n"
        )
        cases = (
            "// the stub is never called when the device is unsupported. op.impl\n"
            "// below is the ordinary aten kernel.\n"
        )
    else:
        gate = "at::globalContext().allowNativeAot()"
        gate_comment = ""
        cases = (
            "// the stub is never called when AOT is switched off or the device is\n"
            "// unsupported. op.impl below is the ordinary aten kernel, and it runs in\n"
            "// exactly those three cases: switched off, unsupported device, or declined.\n"
        )
    return (
        f"// native-AOT: the last conjunct is the LAUNCH, not a query -- it runs the\n"
        f"// AOT kernel into the meta()-allocated outputs and returns true if it\n"
        f"// handled the call, false if it declined this shape. && short-circuits, so\n"
        f"{cases}"
        f"{gate_comment}"
        f"if (!({gate} && "
        f"{stub}.is_device_supported({device_type}) && "
        f"{stub}({device_type}, {impl_exprs}))) {{ op.impl({impl_exprs}); }}"
    )


def validate_native_aot_manifests(
    manifests: dict[tuple[DispatchKey, str], NativeAotManifest],
    grouped_native_functions: Sequence[NativeFunction | NativeFunctionsGroup],
) -> None:
    """Every manifest op must resolve to exactly one structured op group:
    the stub call site is emitted in the structured wrapper (between meta
    and impl), so an unstructured op has nowhere to put it, and an
    ambiguous base name would hook the wrong overload silently."""
    from collections import defaultdict

    from torchgen.model import NativeFunctionsGroup

    structured_by_base: dict[str, list[str]] = defaultdict(list)
    for g in grouped_native_functions:
        if isinstance(g, NativeFunctionsGroup) and g.structured:
            structured_by_base[g.functional.func.name.name.base].append(
                str(g.functional.func.name)
            )
    for key, op in manifests:
        base = op.split(".")[0]
        names = structured_by_base.get(base, [])
        if "." in op:
            if op not in names:
                raise RuntimeError(
                    f"native-aot declaration for {op}@{key}: no structured "
                    f"group named {op!r} in native_functions.yaml "
                    f"(structured overloads of {base!r}: {names or 'none'})"
                )
        elif not names:
            raise RuntimeError(
                f"native-aot declaration for {op}@{key}: {op} is not a "
                f"structured op in native_functions.yaml. The stub is "
                f"consulted between meta() and impl(), so unstructured ops "
                f"are served by the JIT layer only; to embed, structure the "
                f"op upstream first (preferred; e.g. var.correction is a "
                f"candidate)"
            )
        elif len(names) > 1:
            raise RuntimeError(
                f"native-aot declaration for {op}@{key}: base name {op!r} is "
                f"ambiguous across structured groups {names}; qualify "
                f"ATEN_OP with the overload (e.g. {names[0]!r})"
            )
