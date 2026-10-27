"""Prototype: resolve ``torch.library.register_aot_impl`` builders at AOTI time.

For every op in the exported program that has an AOT impl, this calls the
builder with FakeTensors + an :class:`AOTTarget`, then turns the returned
:class:`TVMFFIKernel` into the existing ``custom_ops_to_c_shims`` /
``custom_op_libs`` machinery:

* the shim is *generated* from the op schema + fake outputs -- generic
  ``AtenTensorHandle -> DLTensor`` / scalar -> ``TVMFFIAny`` packing, output
  allocation from the fake outputs' symbolic sizes/strides, and stream
  setup via ``TVMFFIEnvSetStream``. No per-op hand-written C++.
* the generated shim + the builder's ``.o`` files are linked into a companion
  ``lib<op>_aot_impl.so``; its directory is added to ``LIBRARY_PATH`` so the
  wrapper link finds it (runtime still needs ``LD_LIBRARY_PATH``, exactly
  like today's hand-written ``custom_ops_to_c_shims`` path -- see
  ``torch/_inductor/cpp_builder.py``).

Prototype limits: CUDA only, no mutated args, Tensor/float/int/bool schema
args, Tensor (or tuple-of-Tensor) returns, output shapes must be expressible
in the input symbols, single device. One shim is generated per op, so every
call site of the same op in one program must have the same call-site
signature (input/output dtypes, ranks, and size/stride expression structure);
a mismatch raises instead of generating wrong code. Only the top-level graph
is scanned (ops inside ``cond``/``while`` subgraphs keep the proxy executor).
Non-default overloads are rejected: the wrapper derives the shim name from
``schema.name``, which carries no overload suffix, so overloads would collide.
The companion ``.so`` lives in a tempdir that is not cleaned up and is found
via a process-global ``LIBRARY_PATH`` entry at link time. At load time the
model ``.so`` needs it as a DT_NEEDED dependency: put its dir on
``LD_LIBRARY_PATH`` (exactly like today's hand-written
``custom_ops_to_c_shims`` path -- see ``torch/_inductor/cpp_builder.py``), or
preload it by path (it carries a SONAME, so ``ctypes.CDLL(path,
RTLD_GLOBAL)`` satisfies the dependency); it is not packaged into the ``.pt2``.
``aoti_compile_and_package`` is the only wired-up entry point. Anything else
raises a clear error at ``aoti_compile_and_package`` time instead of silently
generating a wrong shim.

(The registry lives in ``torch._library.aoti_impl``; this module is the
Inductor-side resolution. The duplicated module name is intentional for now.)
"""

import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import sympy
from sympy.printing.cxx import cxxcode

import torch
from torch._library.aoti_impl import (
    AOTTarget,
    canonical_op_key,
    get_aot_impl,
    TVMFFIKernel,
)
from torch._ops import OpOverload
from torch.utils._ordered_set import OrderedSet


# ---------------------------------------------------------------------------
# C++ codegen
# ---------------------------------------------------------------------------

_TORCH_DTYPE_TO_AOTI = {
    torch.float32: "float32",
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.float64: "float64",
    torch.int64: "int64",
    torch.int32: "int32",
    torch.int16: "int16",
    torch.int8: "int8",
    torch.uint8: "uint8",
    torch.bool: "bool",
}

# TVM-FFI DLDataType (kind, bits, lanes) per AOTI dtype name.
_DTYPE_TO_DL = {
    "float32": ("kDLFloat", 32),
    "float16": ("kDLFloat", 16),
    "bfloat16": ("kDLBfloat", 16),
    "float64": ("kDLFloat", 64),
    "int64": ("kDLInt", 64),
    "int32": ("kDLInt", 32),
    "int16": ("kDLInt", 16),
    "int8": ("kDLInt", 8),
    "uint8": ("kDLUInt", 8),
    "bool": ("kDLUInt", 1),
}

# torch.JitType -> C++ shim parameter type. (Only the covered subset.)
_SCHEMA_CPP_TYPE: dict[str, str] = {
    "Tensor": "AtenTensorHandle",
    "float": "double",
    "int": "int64_t",
    "bool": "bool",
}


def _schema_arg_cpp_type(arg) -> str:
    name = str(arg.type)
    if name in _SCHEMA_CPP_TYPE:
        return _SCHEMA_CPP_TYPE[name]
    raise NotImplementedError(
        f"register_aot_impl prototype: unsupported schema arg type {name!r} "
        f"for argument {arg.name!r}"
    )


def _sympy_expr(v):
    """FakeTensor size/stride element -> int or sympy expr (unwrap SymInt)."""
    if isinstance(v, (int, sympy.Basic)):
        return v
    node = getattr(v, "node", None)
    if node is not None and isinstance(node.expr, sympy.Basic):
        return node.expr
    raise TypeError(
        f"register_aot_impl prototype: cannot convert {v!r} to a sympy expr"
    )


def _cxx_expr(expr, sym_to_var: dict[sympy.Symbol, str]) -> str:
    """Render a sympy/int size/stride expression as C++ over input-size vars."""
    if isinstance(expr, (int, sympy.Integer)):
        return str(int(expr))
    if isinstance(expr, sympy.Symbol):
        if expr not in sym_to_var:
            raise NotImplementedError(
                f"register_aot_impl prototype: output size/stride uses symbol "
                f"{expr} which is not an input dimension"
            )
        return sym_to_var[expr]
    substituted = expr.subs({s: sympy.Symbol(v) for s, v in sym_to_var.items()})
    unknown = substituted.free_symbols
    if unknown:
        raise NotImplementedError(
            f"register_aot_impl prototype: output size/stride {expr} uses "
            f"symbols {unknown} which are not input dimensions"
        )
    return cxxcode(substituted)


_CXX_KEYWORDS = frozenset(
    [
        "alignas",
        "alignof",
        "and",
        "and_eq",
        "asm",
        "auto",
        "bitand",
        "bitor",
        "bool",
        "break",
        "case",
        "catch",
        "char",
        "char8_t",
        "char16_t",
        "char32_t",
        "class",
        "compl",
        "concept",
        "const",
        "consteval",
        "constexpr",
        "constinit",
        "continue",
        "co_await",
        "co_return",
        "co_yield",
        "decltype",
        "default",
        "delete",
        "do",
        "double",
        "dynamic_cast",
        "else",
        "enum",
        "explicit",
        "export",
        "extern",
        "false",
        "float",
        "for",
        "friend",
        "goto",
        "if",
        "inline",
        "int",
        "long",
        "mutable",
        "namespace",
        "new",
        "noexcept",
        "not",
        "not_eq",
        "nullptr",
        "operator",
        "or",
        "or_eq",
        "private",
        "protected",
        "public",
        "register",
        "reinterpret_cast",
        "requires",
        "return",
        "short",
        "signed",
        "sizeof",
        "static",
        "static_assert",
        "static_cast",
        "struct",
        "switch",
        "template",
        "this",
        "thread_local",
        "throw",
        "true",
        "try",
        "typedef",
        "typeid",
        "typename",
        "union",
        "unsigned",
        "using",
        "virtual",
        "void",
        "volatile",
        "wchar_t",
        "while",
        "xor",
        "xor_eq",
    ]
)


def _validate_cpp_identifier(name: str, what: str) -> None:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) or name in _CXX_KEYWORDS:
        raise ValueError(
            f"register_aot_impl prototype: {what} {name!r} is not a valid "
            "C++ identifier"
        )


def _expr_structure(v) -> str:
    """Size/stride expression reduced to structure: symbols -> S, ints -> N.

    Call-site signatures compare structure, not values: two sites with
    static sizes 4096 and 1000 (or dynamic s0 and s1) share one shim because
    sizes are runtime values, but ``s0`` vs ``s0 + 1`` must not.
    """
    expr = _sympy_expr(v)
    if isinstance(expr, int):
        return "N"
    s = str(expr)
    for sym in expr.free_symbols:
        s = re.sub(rf"\b{re.escape(str(sym))}\b", "S", s)
    return re.sub(r"\d+", "N", s)


def _call_site_signature(schema, fake_args: list, fake_outputs: list):
    parts = []
    for arg, v in zip(schema.arguments, fake_args):
        if isinstance(v, torch.Tensor):
            parts.append(
                (
                    "Tensor",
                    str(v.dtype),
                    v.dim(),
                    tuple(_expr_structure(s) for s in v.size()),
                    tuple(_expr_structure(s) for s in v.stride()),
                )
            )
        else:
            parts.append(("scalar", str(arg.type)))
    out_parts = [
        (
            "Tensor",
            str(o.dtype),
            o.dim(),
            tuple(_expr_structure(s) for s in o.size()),
            tuple(_expr_structure(s) for s in o.stride()),
        )
        for o in fake_outputs
    ]
    return (tuple(parts), tuple(out_parts))


def _shim_fn_name(op: OpOverload, device: str) -> str:
    # Must match the wrapper: FallbackKernel's cpp_kernel_name is
    # schema.name (no overload suffix), and get_c_shim_func_name keeps the
    # text after "::". Non-default overloads therefore collide in the
    # wrapper itself and are rejected in resolve_aot_impls.
    return f"aoti_torch_{device}_{op._schema.name.split('::', 1)[1]}"


def generate_tvm_ffi_shim(
    op: OpOverload,
    kernel: TVMFFIKernel,
    fake_inputs: list,
    fake_outputs: list,
    device: str,
) -> tuple[str, str]:
    """Return (shim C++ source, shim declaration) for a TVMFFIKernel.

    ``fake_inputs`` are the builder's argument values in schema order (FakeTensor
    for tensors, plain values for scalars); ``fake_outputs`` are FakeTensors.
    """
    schema = op._schema
    if any(a.alias_info is not None for a in schema.arguments):
        raise NotImplementedError(
            f"register_aot_impl prototype: {op.name()} mutates an argument; "
            "mutated args are not supported"
        )
    for ret in schema.returns:
        if str(ret.type) != "Tensor":
            raise NotImplementedError(
                f"register_aot_impl prototype: unsupported return type "
                f"{ret.type} for {op.name()}"
            )
    for a in schema.arguments:
        _validate_cpp_identifier(a.name, "schema argument name")

    # Tensor inputs, in schema order, and the symbol -> C var map for
    # sizes/strides. A symbol appearing only in a stride still needs a var.
    tensor_inputs = [
        (i, a.name, v)
        for i, (a, v) in enumerate(zip(schema.arguments, fake_inputs))
        if isinstance(v, torch.Tensor)
    ]
    if not tensor_inputs:
        raise NotImplementedError(
            f"register_aot_impl prototype: {op.name()} has no Tensor inputs; "
            "ops with only scalar inputs are not supported"
        )
    for _, _, val in tensor_inputs:
        if val.dtype not in _TORCH_DTYPE_TO_AOTI:
            raise NotImplementedError(
                f"register_aot_impl prototype: unsupported input dtype {val.dtype}"
            )
    sym_to_var: dict[sympy.Symbol, str] = {}
    for i, name, val in tensor_inputs:
        for d, size in enumerate(val.size()):
            size = _sympy_expr(size)
            if isinstance(size, sympy.Symbol) and size not in sym_to_var:
                sym_to_var[size] = f"in_{name}_size_{d}"
    for i, name, val in tensor_inputs:
        for d, stride in enumerate(val.stride()):
            stride = _sympy_expr(stride)
            if isinstance(stride, sympy.Symbol) and stride not in sym_to_var:
                sym_to_var[stride] = f"in_{name}_stride_{d}"

    scalar_inputs = [
        (a, v)
        for a, v in zip(schema.arguments, fake_inputs)
        if not isinstance(v, torch.Tensor)
    ]
    # Every schema arg must be accounted for exactly once.
    if len(tensor_inputs) + len(scalar_inputs) != len(schema.arguments):
        raise RuntimeError(
            f"register_aot_impl prototype: {op.name()} has "
            f"{len(schema.arguments)} schema args but got "
            f"{len(tensor_inputs) + len(scalar_inputs)} fake inputs"
        )

    shim_fn = _shim_fn_name(op, device)
    decl_params = [_schema_arg_cpp_type(a) + f" {a.name}" for a in schema.arguments]
    decl_params += [f"AtenTensorHandle* ret{i}" for i in range(len(fake_outputs))]
    decl = f"AOTITorchError {shim_fn}({', '.join(decl_params)})"

    lines = []
    lines.append("#include <cstdint>")
    lines.append("#include <dlpack/dlpack.h>")
    lines.append("#include <tvm/ffi/c_api.h>")
    lines.append("#include <tvm/ffi/extra/c_env_api.h>")
    lines.append("#include <torch/csrc/inductor/aoti_torch/c/shim.h>")
    lines.append("")
    lines.append(
        f'extern "C" int {kernel.symbol}(void*, const TVMFFIAny*, int32_t, TVMFFIAny*);'
    )
    lines.append("")
    lines.append("namespace {")
    lines.append("bool toDLDataType(int32_t dtype, DLDataType* out) {")
    for aoti_name, (kind, bits) in _DTYPE_TO_DL.items():
        lines.append(
            f"  if (dtype == aoti_torch_dtype_{aoti_name}()) "
            f"{{ *out = DLDataType{{{kind}, {bits}, 1}}; return true; }}"
        )
    lines.append("  return false;")
    lines.append("}")
    lines.append("bool toDLTensor(AtenTensorHandle t, DLTensor* dl) {")
    lines.append(
        "  void* data = nullptr; int64_t* sizes = nullptr; int64_t* strides = nullptr;"
    )
    lines.append("  int64_t ndim = 0; int32_t dtype = 0, device_type = 0, device = 0;")
    lines.append(
        "  if (aoti_torch_get_data_ptr(t, &data) || aoti_torch_get_sizes(t, &sizes) ||"
    )
    lines.append(
        "      aoti_torch_get_strides(t, &strides) || aoti_torch_get_dim(t, &ndim) ||"
    )
    lines.append(
        "      aoti_torch_get_dtype(t, &dtype) || aoti_torch_get_device_type(t, &device_type) ||"
    )
    lines.append("      aoti_torch_get_device_index(t, &device)) {")
    lines.append("    return false;")
    lines.append("  }")
    lines.append("  DLDataType dl_dtype;")
    lines.append("  if (!toDLDataType(dtype, &dl_dtype)) { return false; }")
    lines.append("  DLDevice dl_device;")
    lines.append(
        "  if (device_type == aoti_torch_device_type_cuda()) { dl_device = DLDevice{kDLCUDA, device}; }"
    )
    lines.append(
        "  else if (device_type == aoti_torch_device_type_cpu()) { dl_device = DLDevice{kDLCPU, device}; }"
    )
    lines.append("  else { return false; }")
    lines.append(
        "  *dl = DLTensor{data, dl_device, static_cast<int32_t>(ndim), dl_dtype, sizes, strides, 0};"
    )
    lines.append("  return true;")
    lines.append("}")
    lines.append("TVMFFIAny tensorArg(DLTensor* t) {")
    lines.append(
        "  TVMFFIAny a{}; a.type_index = kTVMFFIDLTensorPtr; a.v_ptr = t; return a;"
    )
    lines.append("}")
    lines.append("TVMFFIAny floatArg(double v) {")
    lines.append(
        "  TVMFFIAny a{}; a.type_index = kTVMFFIFloat; a.v_float64 = v; return a;"
    )
    lines.append("}")
    lines.append("TVMFFIAny intArg(int64_t v) {")
    lines.append("  TVMFFIAny a{}; a.type_index = kTVMFFIInt; a.v_int64 = v; return a;")
    lines.append("}")
    lines.append("TVMFFIAny boolArg(bool v) {")
    lines.append(
        "  TVMFFIAny a{}; a.type_index = kTVMFFIBool; a.v_int64 = v ? 1 : 0; return a;"
    )
    lines.append("}")
    lines.append("} // namespace")
    lines.append("")
    lines.append(f'extern "C" {decl} {{')

    # Input tensor handles, sizes/strides pointers, device check, size vars.
    for _, name, val in tensor_inputs:
        lines.append(
            f"  int64_t* in_{name}_sizes = nullptr; int64_t* in_{name}_strides = nullptr;"
        )
        lines.append(
            f"  int32_t in_{name}_device_type = 0; int32_t in_{name}_device = 0;"
        )
        lines.append(f"  if (aoti_torch_get_sizes({name}, &in_{name}_sizes) ||")
        lines.append(f"      aoti_torch_get_strides({name}, &in_{name}_strides) ||")
        lines.append(
            f"      aoti_torch_get_device_type({name}, &in_{name}_device_type) ||"
        )
        lines.append(
            f"      aoti_torch_get_device_index({name}, &in_{name}_device)) {{"
        )
        lines.append("    return AOTI_TORCH_FAILURE;")
        lines.append("  }")
        for d, size in enumerate(val.size()):
            sym = _sympy_expr(size)
            if (
                isinstance(sym, sympy.Symbol)
                and sym_to_var.get(sym) == f"in_{name}_size_{d}"
            ):
                lines.append(
                    f"  const int64_t in_{name}_size_{d} = in_{name}_sizes[{d}];"
                )
        for d, stride in enumerate(val.stride()):
            sym = _sympy_expr(stride)
            if (
                isinstance(sym, sympy.Symbol)
                and sym_to_var.get(sym) == f"in_{name}_stride_{d}"
            ):
                lines.append(
                    f"  const int64_t in_{name}_stride_{d} = in_{name}_strides[{d}];"
                )
    if tensor_inputs:
        first = tensor_inputs[0][1]
        lines.append(f"  const int32_t device_type = in_{first}_device_type;")
        lines.append(f"  const int32_t device = in_{first}_device;")
        lines.append("  if (device_type != aoti_torch_device_type_cuda()) {")
        lines.append("    return AOTI_TORCH_FAILURE;  // prototype: CUDA ops only")
        lines.append("  }")
        for _, name, _ in tensor_inputs[1:]:
            lines.append(
                f"  if (in_{name}_device_type != device_type || in_{name}_device != device) {{"
            )
            lines.append(
                "    return AOTI_TORCH_FAILURE;  // prototype: single-device ops only"
            )
            lines.append("  }")

    # Output allocation from the fake outputs' symbolic sizes/strides.
    for j, out in enumerate(fake_outputs):
        if out.dtype not in _TORCH_DTYPE_TO_AOTI:
            raise NotImplementedError(
                f"register_aot_impl prototype: unsupported output dtype {out.dtype}"
            )
        sizes = ", ".join(_cxx_expr(_sympy_expr(s), sym_to_var) for s in out.size())
        strides = ", ".join(_cxx_expr(_sympy_expr(s), sym_to_var) for s in out.stride())
        lines.append(f"  AtenTensorHandle out{j} = nullptr;")
        lines.append(
            f"  {{ int64_t sizes[] = {{{sizes}}}; int64_t strides[] = {{{strides}}};"
        )
        lines.append(
            f"    if (aoti_torch_empty_strided({out.dim()}, sizes, strides, "
            f"aoti_torch_dtype_{_TORCH_DTYPE_TO_AOTI[out.dtype]}(), device_type, device, &out{j})) {{"
        )
        lines.append("      return AOTI_TORCH_FAILURE;")
        lines.append("    } }")

    # Cleanup: delete allocated outputs, restore stream if it was set.
    lines.append("  TVMFFIStreamHandle prev = nullptr;")
    lines.append("  bool stream_set = false;")
    lines.append("  auto fail = [&]() {")
    for j in range(len(fake_outputs)):
        lines.append(
            f"    if (out{j} != nullptr) {{ aoti_torch_delete_tensor_object(out{j}); }}"
        )
    lines.append(
        "    if (stream_set) { TVMFFIEnvSetStream(kDLCUDA, device, prev, nullptr); }"
    )
    lines.append("    return AOTI_TORCH_FAILURE;")
    lines.append("  };")

    # Pack DLTensors for inputs + outputs.
    for _, name, _ in tensor_inputs:
        lines.append(f"  DLTensor dl_{name}{{}};")
        lines.append(f"  if (!toDLTensor({name}, &dl_{name})) {{ return fail(); }}")
    for j in range(len(fake_outputs)):
        lines.append(f"  DLTensor dl_out{j}{{}};")
        lines.append(f"  if (!toDLTensor(out{j}, &dl_out{j})) {{ return fail(); }}")

    # Stream: torch's current stream becomes the TVM-FFI env stream.
    lines.append("  void* stream = nullptr;")
    lines.append(
        "  if (aoti_torch_get_current_cuda_stream(device, &stream)) { return fail(); }"
    )
    lines.append("  TVMFFIEnvSetStream(kDLCUDA, device, stream, &prev);")
    lines.append("  stream_set = true;")

    # Packed call, in kernel.args order (default: tensor inputs, outputs, scalars).
    if kernel.args is None:
        packed = (
            [f"dl_{name}" for _, name, _ in tensor_inputs]
            + [f"dl_out{j}" for j in range(len(fake_outputs))]
            + [a.name for a, _ in scalar_inputs]
        )
        packed_kinds = ["tensor"] * (len(tensor_inputs) + len(fake_outputs)) + [
            str(a.type) for a, _ in scalar_inputs
        ]
    else:
        packed, packed_kinds = [], []
        for arg in kernel.args:
            m = re.fullmatch(r"out(\d+)", arg)
            if m:
                j = int(m.group(1))
                if j >= len(fake_outputs):
                    raise ValueError(
                        f"TVMFFIKernel.args references {arg} but {op.name()} "
                        f"has {len(fake_outputs)} outputs"
                    )
                packed.append(f"dl_out{j}")
                packed_kinds.append("tensor")
                continue
            matches = [a for a in schema.arguments if a.name == arg]
            if not matches:
                raise ValueError(
                    f"TVMFFIKernel.args references unknown argument {arg!r} "
                    f"for {op.name()}"
                )
            if str(matches[0].type) == "Tensor":
                packed.append(f"dl_{arg}")
                packed_kinds.append("tensor")
            else:
                packed.append(arg)
                packed_kinds.append(str(matches[0].type))
    arg_exprs = []
    for name, kind in zip(packed, packed_kinds):
        if kind == "tensor":
            arg_exprs.append(f"tensorArg(&{name})")
        elif kind in ("float", "double"):
            arg_exprs.append(f"floatArg({name})")
        elif kind in ("int",):
            arg_exprs.append(f"intArg({name})")
        elif kind == "bool":
            arg_exprs.append(f"boolArg({name})")
        else:
            raise NotImplementedError(
                f"register_aot_impl prototype: unsupported scalar kind {kind!r}"
            )
    lines.append(
        f"  const TVMFFIAny args[{len(arg_exprs)}] = {{{', '.join(arg_exprs)}}};"
    )
    lines.append("  TVMFFIAny result{}; result.type_index = kTVMFFINone;")
    lines.append(
        f"  const int rc = {kernel.symbol}(nullptr, args, {len(arg_exprs)}, &result);"
    )
    lines.append("  TVMFFIEnvSetStream(kDLCUDA, device, prev, nullptr);")
    lines.append("  stream_set = false;")
    lines.append("  if (rc != 0) { return fail(); }")
    for j in range(len(fake_outputs)):
        lines.append(f"  *ret{j} = out{j};")
    lines.append("  return AOTI_TORCH_SUCCESS;")
    lines.append("}")
    return "\n".join(lines) + "\n", decl


# ---------------------------------------------------------------------------
# Build + resolve
# ---------------------------------------------------------------------------


def _fake_value(v, node: Any = None):
    """FX node / constant -> value the builder sees (FakeTensor for tensors)."""
    if isinstance(v, torch.fx.Node):
        if "val" not in v.meta:
            raise RuntimeError(
                "register_aot_impl prototype: FX node "
                f"{v} (from {node}) has no FakeTensor meta['val']; "
                "the op cannot be resolved"
            )
        return v.meta["val"]
    if isinstance(v, (tuple, list)):
        return type(v)(_fake_value(x, node) for x in v)
    if isinstance(v, dict):
        return {k: _fake_value(x, node) for k, x in v.items()}
    return v


def _build_companion_lib(
    op: OpOverload, kernel: TVMFFIKernel, shim_src: str, workdir: Path
) -> str:
    """Compile shim + kernel objects into lib<op>_aot_impl.so; return lib name."""
    for o in kernel.objects:
        if not Path(o).exists():
            raise RuntimeError(
                f"register_aot_impl: builder for {op.name()} returned a "
                f"TVMFFIKernel with a missing object file: {o}"
            )
    try:
        import tvm_ffi
    except ImportError as e:
        raise RuntimeError(
            "register_aot_impl: TVM-FFI is required to build the companion "
            "library (pip install apache-tvm-ffi)"
        ) from e

    libname = f"{canonical_op_key(op).replace('::', '_').replace('.', '_')}_aot_impl"
    shim_path = workdir / f"{libname}_shim.cpp"
    shim_path.write_text(shim_src)
    so_path = workdir / f"lib{libname}.so"

    torch_include = str(Path(torch.__file__).parent / "include")
    # The shim calls the stable C ABI (aoti_torch_* in libtorch_cpu, plus
    # aoti_torch_get_current_cuda_stream in libtorch_cuda). Link them so the
    # companion lib carries proper DT_NEEDED entries and can be loaded on its
    # own (e.g. preloaded by path); the model .so links the same libs.
    torch_lib = str(Path(torch.__file__).parent / "lib")
    tvm_ffi_dir = Path(tvm_ffi.__file__).parent
    tvm_ffi_include = str(tvm_ffi_dir / "include")
    tvm_ffi_lib = str(tvm_ffi_dir / "lib")

    cmd = [
        "g++",
        "-shared",
        "-fPIC",
        "-DUSE_CUDA",
        f"-Wl,-soname,lib{libname}.so",
        f"-I{torch_include}",
        f"-I{tvm_ffi_include}",
        str(shim_path),
        *[str(o) for o in kernel.objects],
        "-o",
        str(so_path),
        f"-L{tvm_ffi_lib}",
        "-ltvm_ffi",
        f"-L{torch_lib}",
        "-ltorch_cpu",
        "-ltorch_cuda",
        "-lc10",
        *[str(p) for p in kernel.link_libs],
    ]
    rpaths = OrderedSet([tvm_ffi_lib, torch_lib])
    for lib in kernel.link_libs:
        rpaths.add(str(Path(lib).parent))
    for r in sorted(rpaths):
        cmd += [f"-Wl,-rpath,{r}"]
    cuda_stub = Path("/usr/local/cuda/lib64/stubs")
    if cuda_stub.is_dir():
        cmd += [f"-L{cuda_stub}", "-lcuda"]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError as e:
        raise RuntimeError(
            "register_aot_impl: failed to build companion lib for "
            f"{op.name()}: g++ not found on PATH"
        ) from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"register_aot_impl: failed to build companion lib for {op.name()}\n"
            f"command: {' '.join(cmd)}\nstdout:\n{e.stdout}\nstderr:\n{e.stderr}"
        ) from e
    return libname


def resolve_aot_impls(exported_program, inductor_configs: dict) -> None:
    """Fill ``inductor_configs`` with shims/libs for ops with an AOT impl.

    Idempotent per op: an op already present in
    ``aot_inductor.custom_ops_to_c_shims`` (e.g. a user-provided shim, or a
    previous resolve pass) is left alone. The caller's dict is untouched when
    no op in the program has an AOT impl.
    """
    graph = exported_program.graph_module.graph
    # Lazily created on the first op with an impl, so a program without any
    # does not mutate the caller's dict.
    shims = None
    libs = None
    seen: OrderedSet[str] = OrderedSet()
    signatures: dict[str, tuple] = {}
    shim_fns: dict[str, str] = {}
    libnames: dict[str, str] = {}

    for node in graph.nodes:
        if node.op != "call_function":
            continue
        target = node.target
        if not isinstance(target, OpOverload):
            continue
        key = canonical_op_key(target)
        if key in seen:
            # B1: every call site of one op shares one generated shim, so the
            # signatures must match; otherwise the later site would silently
            # get outputs allocated with the first site's metadata.
            sig = _call_site_signature(
                target._schema,
                _positional_args(target._schema, node),
                _fake_outputs_list(node),
            )
            if sig != signatures[key]:
                raise RuntimeError(
                    f"register_aot_impl prototype: {key} is called with "
                    "different call-site signatures in one program "
                    f"(first: {signatures[key]}, now: {sig}); one shim per op "
                    "cannot serve both"
                )
            continue
        if target in (inductor_configs.get("aot_inductor.custom_ops_to_c_shims") or {}):
            continue
        builder = get_aot_impl(key)
        if builder is None:
            continue
        if target._schema.overload_name not in ("", "default"):
            raise NotImplementedError(
                f"register_aot_impl prototype: non-default overload {key} is "
                "not supported; the wrapper derives the shim name from "
                "schema.name, which carries no overload suffix, so overloads "
                "would collide on one shim"
            )
        seen.add(key)

        fake_args = _positional_args(target._schema, node)
        fake_outputs = _fake_outputs_list(node)
        if not all(isinstance(o, torch.Tensor) for o in fake_outputs):
            raise NotImplementedError(
                f"register_aot_impl prototype: {key} has non-tensor outputs"
            )
        signatures[key] = _call_site_signature(target._schema, fake_args, fake_outputs)

        tensor_args = [v for v in fake_args if isinstance(v, torch.Tensor)]
        if not tensor_args or tensor_args[0].device.type != "cuda":
            raise NotImplementedError(
                f"register_aot_impl prototype: {key} needs a CUDA tensor input"
            )
        dev_idx = tensor_args[0].device.index
        if dev_idx is None:
            dev_idx = 0
        major, minor = torch.cuda.get_device_capability(dev_idx)
        workdir = Path(tempfile.mkdtemp(prefix=f"aoti_impl_{key.replace('::', '_')}_"))
        target_obj = AOTTarget(arch=f"sm_{major}{minor}", workdir=workdir)

        kernel = builder(*fake_args, target=target_obj)
        if not isinstance(kernel, TVMFFIKernel):
            raise TypeError(
                f"register_aot_impl builder for {key} returned "
                f"{type(kernel)}; prototype supports only TVMFFIKernel"
            )
        shim_src, decl = generate_tvm_ffi_shim(
            target, kernel, fake_args, fake_outputs, device="cuda"
        )
        # B2: the shim symbol carries no namespace (wrapper behavior), so two
        # ops with the same base name in different namespaces would collide.
        shim_fn = _shim_fn_name(target, "cuda")
        if shim_fn in shim_fns and shim_fns[shim_fn] != key:
            raise RuntimeError(
                f"register_aot_impl prototype: {key} and {shim_fns[shim_fn]} "
                f"map to the same shim symbol {shim_fn}; ops with the same "
                "base name in different namespaces collide"
            )
        shim_fns[shim_fn] = key
        libname = _build_companion_lib(target, kernel, shim_src, workdir)
        if libname in libnames and libnames[libname] != key:
            raise RuntimeError(
                f"register_aot_impl prototype: {key} and {libnames[libname]} "
                f"map to the same companion library {libname}"
            )
        libnames[libname] = key
        if shims is None:
            shims = dict(
                inductor_configs.get("aot_inductor.custom_ops_to_c_shims") or {}
            )
            inductor_configs["aot_inductor.custom_ops_to_c_shims"] = shims
        if libs is None:
            libs = list(inductor_configs.get("aot_inductor.custom_op_libs") or [])
            inductor_configs["aot_inductor.custom_op_libs"] = libs
        shims[target] = [decl]
        if libname not in libs:
            libs.append(libname)
        # The wrapper link resolves -l<libname> via LIBRARY_PATH.
        old = os.environ.get("LIBRARY_PATH", "")
        parts = [p for p in old.split(os.pathsep) if p]
        if str(workdir) not in parts:
            os.environ["LIBRARY_PATH"] = os.pathsep.join([str(workdir), *parts])


def _fake_outputs(node):
    if "val" not in node.meta:
        raise RuntimeError(
            "register_aot_impl prototype: FX node "
            f"{node} has no FakeTensor meta['val']; the op cannot be resolved"
        )
    return node.meta["val"]


def _fake_outputs_list(node) -> list:
    fake_out = _fake_outputs(node)
    return list(fake_out) if isinstance(fake_out, (tuple, list)) else [fake_out]


def _positional_args(schema, node) -> list:
    """Node args in schema order: kwargs mapped by name, omitted args filled
    with their schema defaults (mirrors Inductor's fill_non_provided_args)."""
    fake_args: list = [None] * len(schema.arguments)
    provided = [False] * len(schema.arguments)
    for i, a in enumerate(node.args):
        if i >= len(fake_args):
            raise RuntimeError(
                f"register_aot_impl prototype: {node.target} passes more "
                "positional args than the schema has"
            )
        fake_args[i] = _fake_value(a, node)
        provided[i] = True
    for k, v in node.kwargs.items():
        idx = next((i for i, a in enumerate(schema.arguments) if a.name == k), None)
        if idx is None:
            raise RuntimeError(
                f"register_aot_impl prototype: {node.target} called with "
                f"unknown kwarg {k!r}"
            )
        if provided[idx]:
            raise RuntimeError(
                f"register_aot_impl prototype: {node.target} passes {k!r} "
                "both positionally and by keyword"
            )
        fake_args[idx] = _fake_value(v, node)
        provided[idx] = True
    for i, a in enumerate(schema.arguments):
        if not provided[i]:
            if not a.has_default_value:
                raise RuntimeError(
                    f"register_aot_impl prototype: {node.target} is missing "
                    f"argument {a.name!r} and it has no schema default"
                )
            fake_args[i] = a.default_value
    return fake_args
