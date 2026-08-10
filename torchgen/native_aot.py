"""Discovery and validation of native-AOT declarations.

Ops under torch/_native/ops/<op>/aot.py declare AOT-compiled DSL kernels
embedded into the op's ATen implementation (contract + validating
loader: tools/native_aot/decl.py, which this module loads by file
path). torchgen consumes the declarations to (a) generate
NativeAotStubs.h -- one at::native DispatchStub per declared op,
signature-matched to the structured impl and with no kernel registered
by default -- and (b) emit a stub consultation between op.meta() and
op.impl() in the generated structured wrapper. The AOT kernel library
(built separately, from the same declarations) registers its kernels on
the stubs at load time via set_<device>_dispatch_ptr.

Only the identity torchgen needs is modeled here; the precompile grid
and C++-generating functions are consumed by the export tool and
gen_aot_lib, covered_axes by torch._native.aot_manifest at runtime.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

from torchgen.model import DispatchKey


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

    @property
    def decl_id(self) -> str:
        return self.op.replace(".", "_")

    def stub_name(self) -> str:
        return f"{self.decl_id}_aot_stub"

    def fn_type_name(self) -> str:
        return f"{self.decl_id}_aot_fn"

    def matches_group(self, g: NativeFunctionsGroup) -> bool:
        """Does this manifest target group g? Qualified ops match the
        exact functional overload name; base names match the group's
        base (uniqueness among structured groups is checked in
        validate_native_aot_manifests)."""
        if "." in self.op:
            return self.op == str(g.functional.func.name)
        return self.op == g.functional.func.name.name.base


def _decl_module():
    import importlib.util

    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "tools",
        "native_aot",
        "decl.py",
    )
    spec = importlib.util.spec_from_file_location("native_aot_decl", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_native_aot_manifests(
    ops_dir: str,
) -> dict[tuple[DispatchKey, str], NativeAotManifest]:
    """Discover and validate all aot.py declarations under ops_dir
    (torch/_native/ops); see tools/native_aot/decl.py for the contract."""
    manifests: dict[tuple[DispatchKey, str], NativeAotManifest] = {}
    if not os.path.isdir(ops_dir):
        return manifests
    for key_str, op in _decl_module().discover_declarations(ops_dir):
        key = DispatchKey.parse(key_str)
        manifests[(key, op)] = NativeAotManifest(op=op, dispatch_key=key)
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
    """The structured-wrapper call site. The stub has no kernel unless the
    AOT library registered one, and the Context switch gates the whole
    path; a true return means the AOT kernel filled the meta()-allocated
    outputs and op.impl is skipped.

    The emitted comment is not decoration: the last conjunct LAUNCHES the
    kernel, which no reader can infer from the call site alone (asked in
    review of the generated code)."""
    device_type = f"c10::DeviceType::{m.dispatch_key}"
    stub = f"at::native::{m.stub_name()}"
    return (
        f"// native-AOT: the last conjunct is the LAUNCH, not a query -- it runs the\n"
        f"// AOT kernel into the meta()-allocated outputs and returns true if it\n"
        f"// handled the call, false if it declined this shape. && short-circuits, so\n"
        f"// the stub is never called when AOT is switched off or the device is\n"
        f"// unsupported. op.impl below is the ordinary aten kernel, and it runs in\n"
        f"// exactly those three cases: switched off, unsupported device, or declined.\n"
        f"if (!(at::globalContext().allowNativeAot() && "
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
