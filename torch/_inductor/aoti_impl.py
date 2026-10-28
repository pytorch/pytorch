"""Prototype: resolve ``torch.library.register_aot_impl`` builders at AOTI time.

For each op in the exported program with an AOT impl, the builder is called
with FakeTensors and an :class:`AOTTarget` once per *specialization*: what a
call site's kernel assumes about its inputs (tensor dtypes, ranks, alignment,
static sizes/strides and equalities between dynamic dims, which ``Optional``
args are present, list lengths, and the values of scalars the builder baked
in by leaving them out of ``TVMFFIKernel.args``). Call sites that agree share
a kernel.

One C shim per op is rendered from ``_TVM_FFI_SHIM_TEMPLATE``. It views the
inputs as DLTensors, dispatches to the first specialization whose guard
matches, allocates outputs from the fake outputs' symbolic sizes/strides,
packs ``TVMFFIAny`` args, and calls the kernel on torch's current stream.
Absent ``Optional`` args pack as None and list args are flattened, which is
how a tuple-typed kernel parameter crosses: the builder exports a thin wrapper
taking one parameter per element. The shim and every specialization's objects
are linked into a companion ``.so`` and fed to the existing
``custom_ops_to_c_shims`` / ``custom_op_libs`` machinery.

Prototype limits: CUDA only; no mutated args; Tensor/float/int/bool args
(each optionally ``Optional`` or a list); Tensor returns; output shapes must
be expressible in input sizes/strides or int args; only the top-level graph is
scanned (ops in ``cond``/``while`` subgraphs keep the proxy executor); only
``aoti_compile_and_package`` is wired up. The companion ``.so`` is not
packaged into the ``.pt2``: the wrapper link finds it through a process-global
``LIBRARY_PATH`` entry, and at load time the model ``.so`` needs it as a
DT_NEEDED dependency, on ``LD_LIBRARY_PATH`` like today's hand-written
``custom_ops_to_c_shims`` path, or preloaded by path (it has a unique SONAME;
keep ``RTLD_LOCAL`` so one package's shim cannot interpose on another's). It
lives in a tempdir that is not cleaned up, and every compile prepends one
more directory to ``LIBRARY_PATH``.
"""

import functools
import hashlib
import math
import os
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
from torch.fx.node import map_arg
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
extern "C" int {{v.kernel.symbol}}(void*, const TVMFFIAny*, int32_t, TVMFFIAny*);
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
    if (launch({{v.kernel.symbol}}, args, {{v.packed | length}}, device, stream, "{{shim_fn}}")) {
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
    var: str | None  # Tensor args: shim local holding the DLTensor(s)

    @property
    def param(self) -> str:
        return f"arg_{self.name}"

    def decl(self) -> str:
        ctype = "AtenTensorHandle" if self.scalar is None else _SCALARS[self.scalar][0]
        if self.is_list:
            return f"const {ctype}* {self.param}, int64_t {self.param}_len_"
        return f"{ctype}{'*' if self.optional else ''} {self.param}"


def _schema_args(op: OpOverload) -> list[_Arg]:
    args: list[_Arg] = []
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
        if isinstance(t, torch.TensorType):
            var = f"in{sum(x.scalar is None for x in args)}"
            args.append(_Arg(a.name, None, optional, is_list, var))
        else:
            args.append(_Arg(a.name, str(t), optional, is_list, None))
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


def _shim_fn_name(op: OpOverload) -> str:
    # Must match the wrapper: FallbackKernel's cpp_kernel_name is
    # schema.name (no overload suffix), and get_c_shim_func_name keeps the
    # text after "::".
    return f"aoti_torch_cuda_{op._schema.name.split('::', 1)[1]}"


@dataclass(frozen=True)
class _Output:
    ndim: int
    dtype: str
    sizes: str
    strides: str


@dataclass(frozen=True)
class _Variant:
    """One specialization: its kernel and what it contributes to the shim."""

    kernel: TVMFFIKernel
    guard: tuple[str, ...]
    outputs: tuple[_Output, ...]
    packed: tuple[str, ...]


def render_variant(
    op: OpOverload, kernel: TVMFFIKernel, fake_inputs: list, fake_outputs: list
) -> _Variant:
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
        # A symbolic offset is required to keep 16-byte alignment.
        offset = t.storage_offset()
        nbytes = offset * t.element_size() if isinstance(offset, int) else 0
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
    for arg, val in zip(args, fake_inputs):
        p = arg.param
        if arg.scalar is None:
            tensor_names.append(arg.name)
            if arg.is_list:
                guard.append(f"{p}_len_ == {len(val)}")
                for k, t in enumerate(val):
                    tensor_guard(f"{arg.var}_dl[{k}]", t)
                packed[arg.name] = [
                    f"tensorArg(&{arg.var}_dl[{k}])" for k in range(len(val))
                ]
            elif val is None:
                guard.append(f"{p} == nullptr")
                packed[arg.name] = ["noneArg()"]
            else:
                if arg.optional:
                    guard.append(f"{p} != nullptr")
                tensor_guard(f"{arg.var}_dl", val)
                packed[arg.name] = [f"tensorArg(&{arg.var}_dl)"]
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
        elif isinstance(a, float):
            calls.append(f"floatArg({_literal(a, 'float')})")
        else:
            raise TypeError(
                f"TVMFFIKernel.args entry {a!r}: expected an argument name or a "
                "None / bool / int / float literal"
            )
    return _Variant(kernel, tuple(guard), tuple(outputs), tuple(calls))


def generate_tvm_ffi_shim(op: OpOverload, variants: list[_Variant]) -> tuple[str, str]:
    """Return (shim C++ source, shim declaration) dispatching over ``variants``.

    Variants are tried most-specific first (longest guard), so a kernel
    specialized for static sizes wins over a dynamic one it overlaps.
    """
    args = _schema_args(op)
    num_outputs = len(op._schema.returns)
    shim_fn = _shim_fn_name(op)
    params = [a.decl() for a in args] + [
        f"AtenTensorHandle* ret{j}" for j in range(num_outputs)
    ]
    decl = f"AOTITorchError {shim_fn}({', '.join(params)})"
    src = _shim_template().render(
        decl=decl,
        shim_fn=shim_fn,
        dtypes=_DTYPES.values(),
        tensor_inputs=[a for a in args if a.scalar is None],
        num_outputs=num_outputs,
        variants=sorted(variants, key=lambda v: -len(v.guard)),
    )
    return src, decl


# ---------------------------------------------------------------------------
# Build + resolve
# ---------------------------------------------------------------------------


def _call_site(node: torch.fx.Node) -> tuple[list, list]:
    """A call's fake args in schema order (defaults filled) and fake outputs."""
    args, kwargs = node.args, node.kwargs
    fake_args = [
        map_arg(
            args[i] if i < len(args) else kwargs.get(a.name, a.default_value),
            lambda n: n.meta["val"],
        )
        for i, a in enumerate(node.target._schema.arguments)
    ]
    out = node.meta["val"]
    if out is None:
        return fake_args, []
    return fake_args, list(out) if isinstance(out, (list, tuple)) else [out]


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
            try:
                new = render_variant(self.op, v.kernel, fake_args, fake_outputs)
            except NotImplementedError:
                continue  # e.g. v baked a scalar that is dynamic here
            if new.guard != v.guard:
                continue
            # Same kernel and guard, so only the output allocation can differ.
            if new.outputs != v.outputs:
                raise RuntimeError(
                    f"register_aot_impl prototype: two call sites of {key} match "
                    f"the same specialization but need different outputs "
                    f"({v.outputs} vs {new.outputs}); an output's shape, stride "
                    "or dtype depends on a runtime scalar the guard does not "
                    "check (e.g. a bool), so bake it into the kernel (leave it "
                    "out of TVMFFIKernel.args)"
                )
            return

        # _schema_args guarantees a plain Tensor input.
        fake = next(
            t
            for v in fake_args
            for t in (v if isinstance(v, (list, tuple)) else [v])
            if isinstance(t, FakeTensor)
        )
        major, minor = torch.cuda.get_device_capability(fake.device)
        name = f"{key.replace('::', '_')}_v{len(self.variants)}"
        target = AOTTarget(
            arch=f"sm_{major}{minor}", workdir=self.workdir / name, name=name
        )
        target.workdir.mkdir()
        shape_env = fake.fake_mode.shape_env
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
        self.variants.append(render_variant(self.op, kernel, fake_args, fake_outputs))

    def build(self) -> tuple[str, str]:
        """Generate the shim, link it with every specialization's objects into a
        companion library; return (shim declaration, library name)."""
        op, workdir = self.op, self.workdir
        shim_src, decl = generate_tvm_ffi_shim(op, self.variants)
        objects = [o for v in self.variants for o in v.kernel.objects]
        link_libs = list(
            OrderedSet(str(p) for v in self.variants for p in v.kernel.link_libs)
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
        libname = f"{canonical_op_key(op).replace('::', '_')}_aot_impl_{digest.hexdigest()[:12]}"
        shim_path = workdir / f"{libname}_shim.cpp"
        shim_path.write_text(shim_src)

        torch_include = str(Path(torch.__file__).parent / "include")
        # The shim calls the stable C ABI (aoti_torch_* in libtorch_cpu, plus
        # aoti_torch_get_current_cuda_stream in libtorch_cuda). Link them so the
        # companion lib carries proper DT_NEEDED entries and can be loaded on its
        # own (e.g. preloaded by path); the model .so links the same libs.
        torch_lib = str(Path(torch.__file__).parent / "lib")
        tvm_ffi_dir = Path(tvm_ffi.__file__).parent
        tvm_ffi_lib = str(tvm_ffi_dir / "lib")
        rpaths = OrderedSet(
            [tvm_ffi_lib, torch_lib, *(str(Path(p).parent) for p in link_libs)]
        )
        cmd = [
            "g++",
            "-shared",
            "-fPIC",
            "-std=c++17",
            "-DUSE_CUDA",
            f"-Wl,-soname,lib{libname}.so",
            f"-I{torch_include}",
            f"-I{tvm_ffi_dir / 'include'}",
            str(shim_path),
            *[str(o) for o in objects],
            "-o",
            str(workdir / f"lib{libname}.so"),
            f"-L{tvm_ffi_lib}",
            "-ltvm_ffi",
            f"-L{torch_lib}",
            "-ltorch_cpu",
            "-ltorch_cuda",
            "-lc10",
            *[str(p) for p in link_libs],
            *[f"-Wl,-rpath,{r}" for r in sorted(rpaths)],
        ]
        cuda_stub = Path("/usr/local/cuda/lib64/stubs")
        if cuda_stub.is_dir():
            cmd += [f"-L{cuda_stub}", "-lcuda"]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"register_aot_impl: failed to build companion lib for {op.name()}\n"
                f"command: {' '.join(cmd)}\nstdout:\n{e.stdout}\nstderr:\n{e.stderr}"
            ) from e
        return decl, libname


def resolve_aot_impls(exported_program, inductor_configs: dict) -> dict:
    """Return ``inductor_configs`` plus shims/libs for ops with an AOT impl.

    An op already present in ``aot_inductor.custom_ops_to_c_shims`` (e.g. a
    user-provided shim) is left alone. The caller's dict is never modified,
    so reusing it for another compile resolves the builders again instead of
    mistaking the generated shims for user ones.
    """
    user_shims = inductor_configs.get("aot_inductor.custom_ops_to_c_shims") or {}
    impls: dict[str, _OpImpl] = {}
    shim_fns: dict[str, str] = {}
    symbols: dict[str, str] = {}

    for node in exported_program.graph_module.graph.nodes:
        if node.op != "call_function" or not isinstance(node.target, OpOverload):
            continue
        op = node.target
        # Only default overloads can be registered, so another overload of a
        # registered op gets a different key and is not resolved.
        key = canonical_op_key(op)
        if key not in impls:
            builder = get_aot_impl(key)
            if builder is None or op in user_shims:
                continue
            # The shim symbol carries no namespace (wrapper behavior), so two
            # ops with the same base name in different namespaces would collide.
            shim_fn = _shim_fn_name(op)
            if shim_fn in shim_fns:
                raise RuntimeError(
                    f"register_aot_impl prototype: {key} and {shim_fns[shim_fn]} "
                    f"map to the same shim symbol {shim_fn}; ops with the same "
                    "base name in different namespaces collide"
                )
            shim_fns[shim_fn] = key
            _schema_args(op)  # fail fast, before calling the builder
            workdir = Path(
                tempfile.mkdtemp(prefix=f"aoti_impl_{key.replace('::', '_')}_")
            )
            impls[key] = _OpImpl(op, builder, workdir)
        impls[key].add_call_site(*_call_site(node), symbols)

    if not impls:
        return inductor_configs
    shims = dict(user_shims)
    libs = list(inductor_configs.get("aot_inductor.custom_op_libs") or [])
    for impl in impls.values():
        decl, libname = impl.build()
        shims[impl.op] = [decl]
        libs.append(libname)
        # The wrapper link resolves -l<libname> via LIBRARY_PATH.
        parts = [p for p in os.environ.get("LIBRARY_PATH", "").split(os.pathsep) if p]
        if str(impl.workdir) not in parts:
            os.environ["LIBRARY_PATH"] = os.pathsep.join([str(impl.workdir), *parts])
    return {
        **inductor_configs,
        "aot_inductor.custom_ops_to_c_shims": shims,
        "aot_inductor.custom_op_libs": libs,
    }
