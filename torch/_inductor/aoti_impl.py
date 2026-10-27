"""Prototype: resolve ``torch.library.register_aot_impl`` builders at AOTI time.

For every op in the exported program that has an AOT impl, this calls the
builder with FakeTensors + an :class:`AOTTarget` once per *specialization*,
then turns the returned :class:`TVMFFIKernel` objects into the existing
``custom_ops_to_c_shims`` / ``custom_op_libs`` machinery:

* A specialization is what a call site's compiled kernel assumes about its
  inputs: tensor dtypes, ranks, static sizes/strides and equalities between
  dynamic dims, which ``Optional`` args are present, list lengths, and the
  values of scalars the builder baked in (scalars it left out of
  ``TVMFFIKernel.args``). Call sites that agree share a kernel; each new
  specialization calls the builder again.
* One shim per op is *generated* by rendering ``_TVM_FFI_SHIM_TEMPLATE``: it
  views every input as a DLTensor, then dispatches at runtime to the first
  specialization whose guard matches, allocates the outputs from the fake
  outputs' symbolic sizes/strides, packs ``TVMFFIAny`` args (absent
  ``Optional`` args as ``kTVMFFINone``, list args flattened), and calls the
  kernel on torch's current stream.
* The shim + every specialization's ``.o`` files are linked into a companion
  ``lib<op>_aot_impl.so``; its directory is added to ``LIBRARY_PATH`` so the
  wrapper link finds it.

Tuple-typed kernel parameters (e.g. a NamedTuple of tensors) cross the op as
a ``Tensor[]`` / ``float[]`` schema arg, which the shim flattens into
consecutive packed args; the builder exports a thin ``@cute.jit`` wrapper with
one parameter per element that rebuilds the tuple. This costs nothing per
call, unlike packing a ``kTVMFFIArray``.

Prototype limits: CUDA only, no mutated args, Tensor / float / int / bool
args (each optionally ``Optional`` or a list), Tensor returns, output shapes
must be expressible in input sizes/strides or int args, default overloads
only (the wrapper derives the shim name from ``schema.name``, which carries no
overload suffix). Only the top-level graph is scanned (ops inside
``cond``/``while`` subgraphs keep the proxy executor). The companion ``.so``
lives in a tempdir that is not cleaned up and is found via a process-global
``LIBRARY_PATH`` entry at link time. At load time the model ``.so`` needs it as
a DT_NEEDED dependency: put its dir on ``LD_LIBRARY_PATH`` (exactly like
today's hand-written ``custom_ops_to_c_shims`` path -- see
``torch/_inductor/cpp_builder.py``), or preload it by path (it carries a
SONAME, so ``ctypes.CDLL(path)`` satisfies the dependency; keep the default
``RTLD_LOCAL``, since a globally loaded shim would interpose on other
packages' shims for the same op); it is not packaged into the ``.pt2``.
``aoti_compile_and_package`` is the only wired-up entry point.

(The registry lives in ``torch._library.aoti_impl``; this module is the
Inductor-side resolution.)
"""

import difflib
import functools
import hashlib
import math
import os
import pprint
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import sympy

import torch
from torch._inductor.codegen.common import KernelTemplate
from torch._library.aoti_impl import (
    AOTTarget,
    canonical_op_key,
    get_aot_impl,
    TVMFFIKernel,
)
from torch._ops import OpOverload
from torch._subclasses.fake_tensor import FakeTensor
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.printers import CppPrinter


_cexpr = CppPrinter().doprint

# torch dtype -> (AOTI shim dtype name, DLPack type code, bits)
_DTYPES = {
    torch.float32: ("float32", "kDLFloat", 32),
    torch.float16: ("float16", "kDLFloat", 16),
    torch.bfloat16: ("bfloat16", "kDLBfloat", 16),
    torch.float64: ("float64", "kDLFloat", 64),
    torch.int64: ("int64", "kDLInt", 64),
    torch.int32: ("int32", "kDLInt", 32),
    torch.int16: ("int16", "kDLInt", 16),
    torch.int8: ("int8", "kDLInt", 8),
    torch.uint8: ("uint8", "kDLUInt", 8),
    torch.bool: ("bool", "kDLBool", 8),
}

# Schema scalar type -> (C type the AOTI wrapper passes, TVMFFIAny packer).
# Mirrors CppWrapperCpu.c_type_for_prim_type: bool travels as int32_t.
_SCALARS = {
    "float": ("double", "floatArg"),
    "int": ("int64_t", "intArg"),
    "bool": ("int32_t", "boolArg"),
}

# Shim parameters are the schema arg names prefixed with ``arg_``; every
# other name in the template is fixed or positional (``in{i}``, ``out{j}``),
# so none can collide with a parameter or a C++ keyword.
_TVM_FFI_SHIM_TEMPLATE = r"""
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <vector>
#include <c10/util/generic_math.h>
#include <dlpack/dlpack.h>
#include <tvm/ffi/c_api.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
{% for v in variants %}
extern "C" int {{v.symbol}}(void*, const TVMFFIAny*, int32_t, TVMFFIAny*);
{%- endfor %}

namespace {
using TVMFFIEntry = int (*)(void*, const TVMFFIAny*, int32_t, TVMFFIAny*);

bool toDLDataType(int32_t dtype, DLDataType* out) {
{%- for aoti_dtype, code, bits in dtypes %}
  if (dtype == aoti_torch_dtype_{{aoti_dtype}}()) { *out = DLDataType{ {{- code}}, {{bits}}, 1}; return true; }
{%- endfor %}
  return false;
}

// Views a CUDA tensor as a DLTensor (shape/strides alias the tensor's own).
// All tensors must share one device: the first one sets *device.
bool toDLTensor(AtenTensorHandle t, DLTensor* dl, int32_t* device) {
  void* data = nullptr;
  int64_t* sizes = nullptr;
  int64_t* strides = nullptr;
  int64_t ndim = 0;
  int32_t dtype = 0, device_type = 0, device_index = 0;
  DLDataType dl_dtype;
  if (aoti_torch_get_data_ptr(t, &data) || aoti_torch_get_sizes(t, &sizes) ||
      aoti_torch_get_strides(t, &strides) || aoti_torch_get_dim(t, &ndim) ||
      aoti_torch_get_dtype(t, &dtype) ||
      aoti_torch_get_device_type(t, &device_type) ||
      aoti_torch_get_device_index(t, &device_index) ||
      device_type != aoti_torch_device_type_cuda() ||
      !toDLDataType(dtype, &dl_dtype)) {
    return false;
  }
  if (*device < 0) {
    *device = device_index;
  } else if (device_index != *device) {
    return false;
  }
  *dl = DLTensor{data, DLDevice{kDLCUDA, device_index},
                 static_cast<int32_t>(ndim), dl_dtype, sizes, strides, 0};
  return true;
}

TVMFFIAny noneArg() {
  TVMFFIAny a{}; a.type_index = kTVMFFINone; return a;
}
TVMFFIAny tensorArg(DLTensor* t) {
  TVMFFIAny a{}; a.type_index = kTVMFFIDLTensorPtr; a.v_ptr = t; return a;
}
TVMFFIAny floatArg(double v) {
  TVMFFIAny a{}; a.type_index = kTVMFFIFloat; a.v_float64 = v; return a;
}
TVMFFIAny intArg(int64_t v) {
  TVMFFIAny a{}; a.type_index = kTVMFFIInt; a.v_int64 = v; return a;
}
TVMFFIAny boolArg(bool v) {
  TVMFFIAny a{}; a.type_index = kTVMFFIBool; a.v_int64 = v; return a;
}

// Calls a TVM-FFI entry on `stream` (as the TVM-FFI env stream) and prints
// the kernel's error, which the AOTI error code alone would drop.
int launch(TVMFFIEntry fn, const TVMFFIAny* args, int32_t num_args,
           int32_t device, void* stream, const char* where) {
  TVMFFIStreamHandle prev = nullptr;
  TVMFFIEnvSetStream(kDLCUDA, device, stream, &prev);
  TVMFFIAny result{};
  result.type_index = kTVMFFINone;
  const int rc = fn(nullptr, args, num_args, &result);
  TVMFFIEnvSetStream(kDLCUDA, device, prev, nullptr);
  if (rc == 0 && result.type_index >= kTVMFFIStaticObjectBegin) {
    TVMFFIObjectDecRef(result.v_obj);
  }
  if (rc != 0) {
    TVMFFIObjectHandle err = nullptr;
    TVMFFIErrorMoveFromRaised(&err);
    if (err != nullptr) {
      const TVMFFIErrorCell* cell = TVMFFIErrorGetCellPtr(err);
      fprintf(stderr, "%s: %.*s: %.*s\n", where,
              static_cast<int>(cell->kind.size), cell->kind.data,
              static_cast<int>(cell->message.size), cell->message.data);
      TVMFFIObjectDecRef(err);
    }
  }
  return rc;
}
} // namespace

extern "C" {{decl}} {
  int32_t device = -1;
{%- for t in tensor_inputs %}
{%- if t.is_list %}
  std::vector<DLTensor> {{t.var}}_dl({{t.param}}_len_);
  for (int64_t k = 0; k < {{t.param}}_len_; ++k) {
    if (!toDLTensor({{t.param}}[k], &{{t.var}}_dl[k], &device)) {
      return AOTI_TORCH_FAILURE;
    }
  }
{%- elif t.optional %}
  DLTensor {{t.var}}_dl{};
  if ({{t.param}} && !toDLTensor(*{{t.param}}, &{{t.var}}_dl, &device)) {
    return AOTI_TORCH_FAILURE;
  }
{%- else %}
  DLTensor {{t.var}}_dl;
  if (!toDLTensor({{t.param}}, &{{t.var}}_dl, &device)) {
    return AOTI_TORCH_FAILURE;
  }
{%- endif %}
{%- endfor %}
  void* stream = nullptr;
  if (aoti_torch_get_current_cuda_stream(device, &stream)) {
    return AOTI_TORCH_FAILURE;
  }
{%- for j in range(num_outputs) %}
  AtenTensorHandle out{{j}} = nullptr;
  DLTensor out{{j}}_dl;
{%- endfor %}
  auto fail = [&]() {
{%- for j in range(num_outputs) %}
    if (out{{j}}) {
      aoti_torch_delete_tensor_object(out{{j}});
    }
{%- endfor %}
    return AOTI_TORCH_FAILURE;
  };
{% for v in variants %}
  if ({{ v.guard | join(" &&\n      ") if v.guard else "true" }}) {
{%- for out in v.outputs %}
    {
      const std::array<int64_t, {{out.ndim}}> sizes{ {{- out.sizes -}} };
      const std::array<int64_t, {{out.ndim}}> strides{ {{- out.strides -}} };
      if (aoti_torch_empty_strided({{out.ndim}}, sizes.data(), strides.data(),
                                   aoti_torch_dtype_{{out.dtype}}(),
                                   aoti_torch_device_type_cuda(), device, &out{{loop.index0}}) ||
          !toDLTensor(out{{loop.index0}}, &out{{loop.index0}}_dl, &device)) {
        return fail();
      }
    }
{%- endfor %}
{%- if v.packed %}
    const TVMFFIAny args[] = { {{- v.packed | join(", ") -}} };
{%- else %}
    const TVMFFIAny* args = nullptr;
{%- endif %}
    if (launch({{v.symbol}}, args, {{v.packed | length}}, device, stream, "{{shim_fn}}")) {
      return fail();
    }
{%- for out in v.outputs %}
    *ret{{loop.index0}} = out{{loop.index0}};
{%- endfor %}
    return AOTI_TORCH_SUCCESS;
  }
{%- endfor %}
  fprintf(stderr, "{{shim_fn}}: no compiled specialization matches these inputs\n");
  return AOTI_TORCH_FAILURE;
}
"""


@functools.cache
def _shim_template() -> Any:
    return KernelTemplate._template_from_string(_TVM_FFI_SHIM_TEMPLATE)


@dataclass(frozen=True)
class _Arg:
    """A schema argument as the shim sees it."""

    name: str
    scalar: str | None  # "float" / "int" / "bool"; None for Tensor
    optional: bool
    is_list: bool

    @property
    def param(self) -> str:
        return f"arg_{self.name}"

    def decl(self) -> str:
        ctype = "AtenTensorHandle" if self.scalar is None else _SCALARS[self.scalar][0]
        if self.is_list:
            return f"const {ctype}* {self.param}, int64_t {self.param}_len_"
        return f"{ctype}{'*' if self.optional else ''} {self.param}"


def _schema_args(op: OpOverload) -> list[_Arg]:
    args = []
    for a in op._schema.arguments:
        if a.alias_info is not None:
            raise NotImplementedError(
                f"register_aot_impl prototype: {op.name()} mutates {a.name!r}; "
                "mutated args are not supported"
            )
        if re.fullmatch(r"out\d+", a.name):
            raise NotImplementedError(
                f"register_aot_impl prototype: {op.name()} has an argument "
                f"named {a.name!r}, which TVMFFIKernel.args reserves for outputs"
            )
        t = a.type
        optional = isinstance(t, torch.OptionalType)
        if optional:
            t = t.getElementType()
        is_list = isinstance(t, torch.ListType)
        if is_list:
            t = t.getElementType()
        if (optional and is_list) or not (
            isinstance(t, torch.TensorType) or str(t) in _SCALARS
        ):
            raise NotImplementedError(
                f"register_aot_impl prototype: unsupported schema arg type "
                f"{a.type} for argument {a.name!r}"
            )
        scalar = None if isinstance(t, torch.TensorType) else str(t)
        args.append(_Arg(a.name, scalar, optional, is_list))
    if not any(a.scalar is None and not a.optional and not a.is_list for a in args):
        raise NotImplementedError(
            f"register_aot_impl prototype: {op.name()} needs a plain Tensor "
            "input (it fixes the device)"
        )
    if any(str(r.type) != "Tensor" for r in op._schema.returns):
        raise NotImplementedError(
            f"register_aot_impl prototype: {op.name()} must return Tensors"
        )
    return args


def _expr(v) -> sympy.Expr:
    """Size/stride/scalar value (int or SymInt) -> sympy expr."""
    return v.node.expr if isinstance(v, torch.SymInt) else sympy.Integer(v)


def _aoti_dtype(dtype: torch.dtype) -> str:
    if dtype not in _DTYPES:
        raise NotImplementedError(
            f"register_aot_impl prototype: unsupported dtype {dtype}"
        )
    return _DTYPES[dtype][0]


def _literal(v, scalar: str) -> str:
    if isinstance(v, (torch.SymInt, torch.SymFloat, torch.SymBool)):
        raise NotImplementedError(
            "register_aot_impl prototype: a dynamic scalar cannot be baked "
            "into the kernel; list it in TVMFFIKernel.args"
        )
    if scalar == "bool":
        return "1" if v else "0"
    if scalar == "int":
        i = int(v)
        if not -(2**63) <= i < 2**63:
            raise NotImplementedError(
                f"register_aot_impl prototype: int {i} does not fit in int64"
            )
        return "(-9223372036854775807L - 1)" if i == -(2**63) else f"{i}L"
    if not math.isfinite(v):
        raise NotImplementedError(
            f"register_aot_impl prototype: cannot bake non-finite float {v}"
        )
    return repr(float(v))


def _shim_fn_name(op: OpOverload, device: str) -> str:
    # Must match the wrapper: FallbackKernel's cpp_kernel_name is
    # schema.name (no overload suffix), and get_c_shim_func_name keeps the
    # text after "::".
    return f"aoti_torch_{device}_{op._schema.name.split('::', 1)[1]}"


@dataclass(frozen=True)
class _Output:
    ndim: int
    dtype: str
    sizes: str
    strides: str


@dataclass(frozen=True)
class _VariantCode:
    """Everything one specialization contributes to the shim."""

    symbol: str
    guard: tuple[str, ...]
    outputs: tuple[_Output, ...]
    packed: tuple[str, ...]


def render_variant(
    op: OpOverload, kernel: TVMFFIKernel, fake_inputs: list, fake_outputs: list
) -> _VariantCode:
    """Guard, output allocation and packed call of one specialization.

    ``fake_inputs`` are the builder's argument values in schema order
    (FakeTensors, lists of them, scalars, or None for absent Optionals);
    ``fake_outputs`` are FakeTensors.
    """
    args = _schema_args(op)
    guard: list[str] = []
    # Every size/stride/int-arg expression readable at runtime (bare symbols,
    # and composite expressions like ``s0 - 1`` that first appear as a whole),
    # mapped to a symbol named after its C location. Output sizes and
    # dim-equality guards are printed over these.
    known: dict[sympy.Expr, sympy.Symbol] = {}
    packed: dict[str, list[str]] = {}  # TVMFFIKernel.args name -> TVMFFIAny exprs
    order = kernel.args

    def at(e: sympy.Expr) -> sympy.Expr | None:
        """``e`` over runtime locations, or None if it needs an unseen symbol."""
        r = e.xreplace(known)
        return r if r.free_symbols <= OrderedSet(known.values()) else None

    def bind(e: sympy.Expr, loc: str) -> None:
        """Guard that ``loc`` equals ``e``, or record ``loc`` as defining it."""
        if (r := at(e)) is not None:
            guard.append(f"{loc} == {_cexpr(r)}")
        else:
            known[e] = sympy.Symbol(loc, integer=True)

    def tensor_guard(c: str, t: torch.Tensor) -> None:
        _aoti_dtype(t.dtype)
        _, code, bits = _DTYPES[t.dtype]
        guard.extend(
            [
                f"{c}.ndim == {t.dim()}",
                f"{c}.dtype.code == {code}",
                f"{c}.dtype.bits == {bits}",
            ]
        )
        for d, size in enumerate(t.size()):
            bind(_expr(size), f"{c}.shape[{d}]")
        for d, (size, stride) in enumerate(zip(t.size(), t.stride())):
            # A size-0/1 dim's stride is never used to address memory.
            if not (isinstance(size, int) and size <= 1):
                bind(_expr(stride), f"{c}.strides[{d}]")
        # Kernels assume the data alignment of the fake they were compiled
        # for (CuTeDSL: assumed_align), so it is part of the specialization.
        offset = t.storage_offset()
        if isinstance(offset, int):
            nbytes = offset * t.element_size()
            align = 16 if nbytes % 16 == 0 else nbytes & -nbytes
            if align > 1:
                guard.append(f"reinterpret_cast<uintptr_t>({c}.data) % {align} == 0")

    def scalar_guard(loc: str, val, scalar: str, baked: bool) -> None:
        # Static ints are guarded even when passed at runtime: they can set
        # output sizes, which the shim bakes into the allocation.
        if baked or (scalar == "int" and not isinstance(val, torch.SymInt)):
            guard.append(f"{loc} == {_literal(val, scalar)}")
        elif scalar == "int":
            bind(_expr(val), loc)

    tensor_names, scalar_names = [], []
    tensor_index = 0
    for arg, val in zip(args, fake_inputs):
        p = arg.param
        if arg.scalar is None:
            var = f"in{tensor_index}"
            tensor_index += 1
            tensor_names.append(arg.name)
            if arg.is_list:
                guard.append(f"{p}_len_ == {len(val)}")
                for k, t in enumerate(val):
                    tensor_guard(f"{var}_dl[{k}]", t)
                packed[arg.name] = [
                    f"tensorArg(&{var}_dl[{k}])" for k in range(len(val))
                ]
            elif val is None:
                guard.append(f"{p} == nullptr")
                packed[arg.name] = ["noneArg()"]
            else:
                if arg.optional:
                    guard.append(f"{p} != nullptr")
                tensor_guard(f"{var}_dl", val)
                packed[arg.name] = [f"tensorArg(&{var}_dl)"]
            continue

        scalar_names.append(arg.name)
        packer = _SCALARS[arg.scalar][1]
        baked = order is not None and arg.name not in order
        if arg.is_list:
            guard.append(f"{p}_len_ == {len(val)}")
            for k, x in enumerate(val):
                scalar_guard(f"{p}[{k}]", x, arg.scalar, baked)
            packed[arg.name] = [f"{packer}({p}[{k}])" for k in range(len(val))]
        elif val is None:
            guard.append(f"{p} == nullptr")
            packed[arg.name] = ["noneArg()"]
        else:
            ref = f"(*{p})" if arg.optional else p
            if arg.optional:
                guard.append(f"{p} != nullptr")
            scalar_guard(ref, val, arg.scalar, baked)
            packed[arg.name] = [f"{packer}({ref})"]

    def render(values) -> str:
        exprs = []
        for v in values:
            if (r := at(_expr(v))) is None:
                raise NotImplementedError(
                    f"register_aot_impl prototype: output size/stride {_expr(v)} "
                    "is not expressible in input sizes/strides or int args"
                )
            exprs.append(_cexpr(r))
        return ", ".join(exprs)

    outputs = []
    for j, out in enumerate(fake_outputs):
        packed[f"out{j}"] = [f"tensorArg(&out{j}_dl)"]
        outputs.append(
            _Output(
                out.dim(),
                _aoti_dtype(out.dtype),
                render(out.size()),
                render(out.stride()),
            )
        )

    # Default packed-call order is the CuTeDSL export_to_c convention.
    if order is None:
        order = tensor_names + [f"out{j}" for j in range(len(outputs))] + scalar_names
    if unknown := [a for a in order if isinstance(a, str) and a not in packed]:
        raise ValueError(
            f"TVMFFIKernel.args references unknown argument(s) {unknown} for "
            f"{op.name()}; expected schema argument names or out<i>"
        )
    calls = []
    for a in order:
        if a is None:
            calls.append("noneArg()")
        elif isinstance(a, str):
            calls.extend(packed[a])
        elif isinstance(a, bool):
            calls.append(f"boolArg({_literal(a, 'bool')})")
        elif isinstance(a, int):
            calls.append(f"intArg({_literal(a, 'int')})")
        else:
            calls.append(f"floatArg({_literal(a, 'float')})")
    return _VariantCode(kernel.symbol, tuple(guard), tuple(outputs), tuple(calls))


def generate_tvm_ffi_shim(
    op: OpOverload, variants: list[_VariantCode], device: str = "cuda"
) -> tuple[str, str]:
    """Return (shim C++ source, shim declaration) dispatching over ``variants``.

    Variants are tried most-specific first (longest guard), so a kernel
    specialized for static sizes wins over a dynamic one it overlaps.
    """
    args = _schema_args(op)
    num_outputs = len(op._schema.returns)
    shim_fn = _shim_fn_name(op, device)
    params = [a.decl() for a in args] + [
        f"AtenTensorHandle* ret{j}" for j in range(num_outputs)
    ]
    decl = f"AOTITorchError {shim_fn}({', '.join(params)})"
    tensor_inputs = [
        {
            "var": f"in{i}",
            "param": a.param,
            "optional": a.optional,
            "is_list": a.is_list,
        }
        for i, a in enumerate(a for a in args if a.scalar is None)
    ]
    src = _shim_template().render(
        decl=decl,
        shim_fn=shim_fn,
        dtypes=_DTYPES.values(),
        tensor_inputs=tensor_inputs,
        num_outputs=num_outputs,
        variants=sorted(variants, key=lambda v: -len(v.guard)),
    )
    return src, decl


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
    return v


def _build_companion_lib(
    op: OpOverload,
    objects: list,
    link_libs: list,
    shim_src: str,
    workdir: Path,
) -> str:
    """Compile shim + kernel objects into lib<op>_aot_impl.so; return lib name."""
    for o in objects:
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

    # The name (and SONAME) carries a content hash: a process that loads two
    # packages using the same op must bind each to its own shim, and an
    # already-loaded SONAME would otherwise satisfy the second's DT_NEEDED.
    digest = hashlib.sha256(shim_src.encode())
    for o in objects:
        digest.update(Path(o).read_bytes())
    base = canonical_op_key(op).replace("::", "_").replace(".", "_")
    libname = f"{base}_aot_impl_{digest.hexdigest()[:12]}"
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
        "-std=c++17",
        "-DUSE_CUDA",
        f"-Wl,-soname,lib{libname}.so",
        f"-I{torch_include}",
        f"-I{tvm_ffi_include}",
        str(shim_path),
        *[str(o) for o in objects],
        "-o",
        str(so_path),
        f"-L{tvm_ffi_lib}",
        "-ltvm_ffi",
        f"-L{torch_lib}",
        "-ltorch_cpu",
        "-ltorch_cuda",
        "-lc10",
        *[str(p) for p in link_libs],
    ]
    rpaths = OrderedSet([tvm_ffi_lib, torch_lib])
    for lib in link_libs:
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


@dataclass
class _Variant:
    kernel: TVMFFIKernel
    code: _VariantCode


@dataclass
class _OpImpl:
    """One op's registered builder and the specializations built so far."""

    op: OpOverload
    builder: Any
    workdir: Path
    variants: list[_Variant] = field(default_factory=list)

    def add_call_site(self, fake_args: list, fake_outputs: list, symbols: dict) -> None:
        key = canonical_op_key(self.op)
        for v in self.variants:
            code = render_variant(self.op, v.kernel, fake_args, fake_outputs)
            if code.guard != v.code.guard:
                continue
            if code != v.code:
                diff = "\n".join(
                    difflib.unified_diff(
                        pprint.pformat(v.code).splitlines(),
                        pprint.pformat(code).splitlines(),
                        "existing specialization",
                        "this call site",
                        lineterm="",
                    )
                )
                raise RuntimeError(
                    f"register_aot_impl prototype: a call site of {key} has the "
                    "same runtime guard as an existing specialization but needs "
                    f"different code, so dispatch cannot tell them apart:\n{diff}"
                )
            return

        device = next(
            t.device
            for v in fake_args
            for t in (v if isinstance(v, (list, tuple)) else [v])
            if isinstance(t, torch.Tensor)
        )
        major, minor = torch.cuda.get_device_capability(
            0 if device.index is None else device.index
        )
        name = f"{key.replace('::', '_').replace('.', '_')}_v{len(self.variants)}"
        target = AOTTarget(
            arch=f"sm_{major}{minor}", workdir=self.workdir / name, name=name
        )
        target.workdir.mkdir()
        shape_env = next(
            (
                t.fake_mode.shape_env
                for v in fake_args
                for t in (v if isinstance(v, (list, tuple)) else [v])
                if isinstance(t, FakeTensor)
            ),
            None,
        )
        num_guards = len(shape_env.guards) if shape_env is not None else 0
        kernel = self.builder(*fake_args, target=target)
        if shape_env is not None and len(shape_env.guards) > num_guards:
            new_guards = [str(g.expr) for g in shape_env.guards[num_guards:]]
            raise RuntimeError(
                f"register_aot_impl prototype: the builder for {key} specialized "
                f"on symbolic sizes ({new_guards}), which the shim cannot check "
                "at runtime; treat SymInt sizes as dynamic in the builder (e.g. "
                "use statically_known_true / guard_or_false instead of bool())"
            )
        if not isinstance(kernel, TVMFFIKernel):
            raise TypeError(
                f"register_aot_impl builder for {key} returned "
                f"{type(kernel)}; prototype supports only TVMFFIKernel"
            )
        if kernel.symbol in symbols:
            raise RuntimeError(
                f"register_aot_impl prototype: {key} specialization {name} and "
                f"{symbols[kernel.symbol]} both export {kernel.symbol}; export each "
                "specialization under target.name (e.g. "
                "export_to_c(..., function_name=target.name))"
            )
        symbols[kernel.symbol] = name
        self.variants.append(
            _Variant(kernel, render_variant(self.op, kernel, fake_args, fake_outputs))
        )


def resolve_aot_impls(exported_program, inductor_configs: dict) -> None:
    """Fill ``inductor_configs`` with shims/libs for ops with an AOT impl.

    An op already present in ``aot_inductor.custom_ops_to_c_shims`` (e.g. a
    user-provided shim) is left alone. The caller's dict is untouched when no
    op in the program has an AOT impl.
    """
    user_shims = inductor_configs.get("aot_inductor.custom_ops_to_c_shims") or {}
    impls: dict[str, _OpImpl] = {}
    shim_fns: dict[str, str] = {}
    symbols: dict[str, str] = {}

    for node in exported_program.graph_module.graph.nodes:
        if node.op != "call_function" or not isinstance(node.target, OpOverload):
            continue
        op = node.target
        key = canonical_op_key(op)
        if key not in impls:
            builder = get_aot_impl(key)
            if builder is None or op in user_shims:
                continue
            if op._schema.overload_name not in ("", "default"):
                raise NotImplementedError(
                    f"register_aot_impl prototype: non-default overload {key} is "
                    "not supported; the wrapper derives the shim name from "
                    "schema.name, which carries no overload suffix, so overloads "
                    "would collide on one shim"
                )
            # The shim symbol carries no namespace (wrapper behavior), so two
            # ops with the same base name in different namespaces would collide.
            shim_fn = _shim_fn_name(op, "cuda")
            if shim_fn in shim_fns:
                raise RuntimeError(
                    f"register_aot_impl prototype: {key} and {shim_fns[shim_fn]} "
                    f"map to the same shim symbol {shim_fn}; ops with the same "
                    "base name in different namespaces collide"
                )
            shim_fns[shim_fn] = key
            _schema_args(op)  # fail fast on unsupported schemas
            workdir = Path(
                tempfile.mkdtemp(prefix=f"aoti_impl_{key.replace('::', '_')}_")
            )
            impls[key] = _OpImpl(op, builder, workdir)
        fake_outputs = _fake_outputs_list(node)
        if not all(isinstance(o, torch.Tensor) for o in fake_outputs):
            raise NotImplementedError(
                f"register_aot_impl prototype: {key} has non-tensor outputs"
            )
        impls[key].add_call_site(
            _positional_args(op._schema, node), fake_outputs, symbols
        )

    if not impls:
        return
    shims = dict(user_shims)
    libs = list(inductor_configs.get("aot_inductor.custom_op_libs") or [])
    for impl in impls.values():
        shim_src, decl = generate_tvm_ffi_shim(impl.op, [v.code for v in impl.variants])
        libname = _build_companion_lib(
            impl.op,
            [o for v in impl.variants for o in v.kernel.objects],
            list(OrderedSet(str(p) for v in impl.variants for p in v.kernel.link_libs)),
            shim_src,
            impl.workdir,
        )
        shims[impl.op] = [decl]
        libs.append(libname)
        # The wrapper link resolves -l<libname> via LIBRARY_PATH.
        parts = [p for p in os.environ.get("LIBRARY_PATH", "").split(os.pathsep) if p]
        if str(impl.workdir) not in parts:
            os.environ["LIBRARY_PATH"] = os.pathsep.join([str(impl.workdir), *parts])
    inductor_configs["aot_inductor.custom_ops_to_c_shims"] = shims
    inductor_configs["aot_inductor.custom_op_libs"] = libs


def _fake_outputs(node):
    if "val" not in node.meta:
        raise RuntimeError(
            "register_aot_impl prototype: FX node "
            f"{node} has no FakeTensor meta['val']; the op cannot be resolved"
        )
    return node.meta["val"]


def _fake_outputs_list(node) -> list:
    fake_out = _fake_outputs(node)
    if fake_out is None:
        return []
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
