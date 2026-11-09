"""Support for user-defined CuTeDSL kernels in the C++ wrapper (AOTInductor, and
torch.compile with cpp_wrapper).

A ``@cute.jit`` host function is exported with ``export_to_c``, which writes a C
header plus a PIC object. This module parses the header and generates the C++ that
loads and launches the kernel; it takes parameter specs, not IR nodes.

The DSL is imported lazily: importing this module never needs cutlass. Do not add
``from __future__ import annotations``: the DSL reads annotations at runtime and
string annotations break its Constexpr detection.
"""

import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch.utils._ordered_set import OrderedSet


__all__ = [
    "CPP_CUDA_ERROR_CHECK",
    "CPP_HEADERS",
    "ConstantParamSpec",
    "CuteDSLExportedKernel",
    "CuteDSLExportError",
    "ExportedABI",
    "ExportedTensorStruct",
    "ParamSpec",
    "ScalarParamSpec",
    "StreamParamSpec",
    "TensorParamSpec",
    "emit_cpp_declarations",
    "emit_cpp_launch",
    "parse_exported_header",
    "scalar_ctype",
]


class CuteDSLExportError(RuntimeError):
    """An exported CuTeDSL artifact does not match what PT2 asked for."""


@dataclass(frozen=True)
class TensorParamSpec:
    """A ``cute.Tensor`` parameter. ``None`` marks a dynamic size (an int32 slot in
    the exported struct) or a dynamic stride (an int64 slot)."""

    name: str
    dtype: torch.dtype
    sizes: tuple[int | None, ...]
    strides: tuple[int | None, ...]
    align: int

    def __post_init__(self) -> None:
        if len(self.sizes) != len(self.strides):
            raise ValueError(
                f"{self.name}: {len(self.sizes)} sizes but {len(self.strides)} strides"
            )
        if self.align <= 0 or self.align & (self.align - 1):
            raise ValueError(f"{self.name}: align={self.align} is not a power of two")


@dataclass(frozen=True)
class ScalarParamSpec:
    """A runtime scalar annotated with a cutlass Numeric type (``cutlass.Int32``...).
    ``hint`` is only the value used to compile; the export takes the scalar at run
    time, so the hint is not part of the kernel identity."""

    name: str
    annotation: type
    hint: int | float


@dataclass(frozen=True)
class ConstantParamSpec:
    """A value the DSL bakes into the kernel (Constexpr and other compile-time args)."""

    name: str
    value: object


@dataclass(frozen=True)
class StreamParamSpec:
    """The ``cuda.CUstream`` parameter; the launch passes the PT2 stream."""

    name: str


ParamSpec = TensorParamSpec | ScalarParamSpec | ConstantParamSpec | StreamParamSpec


# cutlass Numeric class name -> C type of the exported wrapper parameter. Mirrors
# ``numeric_to_c_type`` in cutlass/base_dsl/export/c_header_generator.py (4.4.2);
# BFloat16 and the float8 types are not supported by the DSL's C export.
_SCALAR_CTYPES = {
    "Int8": "int8_t",
    "Int16": "int16_t",
    "Int32": "int32_t",
    "Int64": "int64_t",
    "Uint8": "uint8_t",
    "Uint16": "uint16_t",
    "Uint32": "uint32_t",
    "Uint64": "uint64_t",
    "Boolean": "uint8_t",
    "Float32": "float",
    "TFloat32": "float",
    "Float64": "double",
    "Float16": "__half_raw",
}
# Integer C types narrower than the int64 values the wrapper computes with (a
# uint64_t scalar is passed as uint64_t: values from 2**63 do not fit an int64).
_INT_CTYPE_RANGES: dict[str, tuple[int, int]] = {
    "int8_t": (-(2**7), 2**7 - 1),
    "int16_t": (-(2**15), 2**15 - 1),
    "int32_t": (-(2**31), 2**31 - 1),
    "uint8_t": (0, 2**8 - 1),
    "uint16_t": (0, 2**16 - 1),
    "uint32_t": (0, 2**32 - 1),
}


def scalar_ctype(annotation: type) -> str:
    """C type the exported wrapper uses for a scalar with this cutlass annotation."""
    from torch._higher_order_ops.cutedsl_kernel_wrap import _is_cutlass_type

    name = getattr(annotation, "__name__", None)
    if not _is_cutlass_type(annotation) or name not in _SCALAR_CTYPES:
        raise CuteDSLExportError(
            f"scalar annotation {annotation!r} is not supported by CuTeDSL export_to_c "
            f"(supported: {', '.join(_SCALAR_CTYPES)})"
        )
    return _SCALAR_CTYPES[name]


@dataclass(frozen=True)
class ExportedTensorStruct:
    """``typedef struct {void *data; <shape_ctype> dynamic_shapes[k];
    <stride_ctype> dynamic_strides[m];} <ctype>;`` for one tensor parameter."""

    ctype: str
    param: str
    shape_ctype: str | None
    num_dynamic_shapes: int
    stride_ctype: str | None
    num_dynamic_strides: int


@dataclass(frozen=True)
class ExportedABI:
    """The C ABI of one ``export_to_c`` header."""

    prefix: str
    module_ctype: str
    unload_fn: str
    wrapper_fn: str
    wrapper_return_ctype: str
    tensor_structs: tuple[ExportedTensorStruct, ...]
    # (ctype, name) of every wrapper parameter after the leading module pointer.
    params: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class CuteDSLExportedKernel:
    """Header + object written by ``export_to_c`` for ``specs``."""

    prefix: str
    header_path: str
    object_path: str
    header_text: str
    abi: ExportedABI
    specs: tuple[ParamSpec, ...] = ()


# The export_to_c header's Module_Load (nvidia-cutlass-dsl 4.4.2 leaves ``ret``
# uninitialized), whose two steps emit_cpp_declarations calls itself: the object's
# cuda_init loads the CUDA library, and its cuda_load_to_device loads it onto one
# device. Compared with runs of white space collapsed.
_MODULE_LOAD = """
void _mlir_{p}_cuda_init(void **);
void _mlir_{p}_cuda_load_to_device(void **);
static inline void {p}_Kernel_Module_Load({p}_Kernel_Module_t *module) {{
    cudaLibrary_t *libraryPtr = &(module->module);
    cudaError_t ret{init};
    struct {{
        cudaLibrary_t **libraryPtr;
        cudaError_t *ret;
    }} initArgs = {{&libraryPtr, &ret}};
    _mlir_{p}_cuda_init((void **)(&initArgs));
    CUTE_DSL_CUDA_ERROR_CHECK(ret);
    int32_t device_id = 0;
    struct {{
        cudaLibrary_t **library;
        int32_t *device_id;
        cudaError_t *ret;
    }} loadArgs = {{&libraryPtr, &device_id, &ret}};
    int32_t device_count;
    CUTE_DSL_CUDA_ERROR_CHECK(cudaGetDeviceCount(&device_count));
    for (int32_t i = 0; i < device_count; i++) {{
        device_id = i;
        _mlir_{p}_cuda_load_to_device((void **)(&loadArgs));
        CUTE_DSL_CUDA_ERROR_CHECK(ret);
    }}
}}
"""

_STRUCT_FIELD = re.compile(
    r"(?P<ctype>int32_t|int64_t)\s+(?P<field>dynamic_shapes|dynamic_strides)"
    r"\s*\[\s*(?P<n>\d+)\s*\]"
)
_WRAPPER_PARAM = re.compile(
    r"(?P<ctype>[A-Za-z_][\w\s]*?)\s*(?P<ptr>\*?)\s*(?P<name>[A-Za-z_]\w*)"
)


def parse_exported_header(text: str, prefix: str) -> ExportedABI:
    """Parse the parts of an export_to_c header that the C++ launch depends on."""
    p = re.escape(prefix)

    def need(pattern: str, what: str) -> re.Match[str]:
        m = re.search(pattern, text)
        if m is None:
            raise CuteDSLExportError(f"export_to_c header for {prefix}: no {what}")
        return m

    module_ctype = f"{prefix}_Kernel_Module_t"
    need(
        rf"typedef\s+struct\s*\{{\s*cudaLibrary_t\s+module\s*;\s*\}}"
        rf"\s*{p}_Kernel_Module_t\s*;",
        "module struct",
    )
    words = " ".join(text.split())
    if not any(
        " ".join(_MODULE_LOAD.format(p=prefix, init=init).split()) in words
        for init in ("", " = cudaSuccess")
    ):
        raise CuteDSLExportError(
            f"export_to_c header for {prefix}: no module load function of the known "
            "form (see emit_cpp_declarations)"
        )
    need(
        rf"\b{p}_Kernel_Module_Unload\s*\(\s*{p}_Kernel_Module_t\s*\*",
        "module unload function",
    )

    structs = []
    for m in re.finditer(
        rf"typedef\s+struct\s*\{{(?P<body>[^}}]*)\}}"
        rf"\s*(?P<ctype>{p}_Tensor_(?P<param>\w+)_t)\s*;",
        text,
    ):
        decls = [d.strip() for d in m.group("body").split(";") if d.strip()]
        if not decls or re.fullmatch(r"void\s*\*\s*data", decls[0]) is None:
            raise CuteDSLExportError(
                f"{m.group('ctype')}: first field is not 'void *data'"
            )
        slots: dict[str, tuple[str, int]] = {}
        for decl in decls[1:]:
            f = _STRUCT_FIELD.fullmatch(decl)
            if f is None or f.group("field") in slots:
                raise CuteDSLExportError(
                    f"{m.group('ctype')}: unexpected field {decl!r}"
                )
            slots[f.group("field")] = (f.group("ctype"), int(f.group("n")))
        shape_ctype, n_shapes = slots.get("dynamic_shapes", (None, 0))
        stride_ctype, n_strides = slots.get("dynamic_strides", (None, 0))
        structs.append(
            ExportedTensorStruct(
                m.group("ctype"),
                m.group("param"),
                shape_ctype,
                n_shapes,
                stride_ctype,
                n_strides,
            )
        )

    wrapper_fn = f"cute_dsl_{prefix}_wrapper"
    w = need(
        rf"static\s+inline\s+(?P<ret>\w+)\s+{re.escape(wrapper_fn)}"
        r"\s*\((?P<params>[^)]*)\)",
        "wrapper function",
    )
    params = []
    for raw in w.group("params").split(","):
        pm = _WRAPPER_PARAM.fullmatch(raw.strip())
        if pm is None:
            raise CuteDSLExportError(
                f"{wrapper_fn}: cannot parse parameter {raw.strip()!r}"
            )
        ctype = " ".join(pm.group("ctype").split()) + pm.group("ptr")
        params.append((ctype, pm.group("name")))
    if not params or params[0][0] != f"{module_ctype}*":
        raise CuteDSLExportError(
            f"{wrapper_fn}: first parameter is not a {module_ctype} pointer"
        )
    return ExportedABI(
        prefix=prefix,
        module_ctype=module_ctype,
        unload_fn=f"{prefix}_Kernel_Module_Unload",
        wrapper_fn=wrapper_fn,
        wrapper_return_ctype=w.group("ret"),
        tensor_structs=tuple(structs),
        params=tuple(params[1:]),
    )


# Standard headers the declarations and launches use.
CPP_HEADERS = (
    "atomic",
    "cstdint",
    "exception",
    "iostream",
    "mutex",
    "string",
    "vector",
)

# Defined once per wrapper, before the first export_to_c header (whose own definition
# is #ifndef-guarded). Those headers print and carry on when a CUDA call fails; this
# makes the failure throw, so a failed module load becomes an AOTI runtime error
# instead of a later illegal launch.
CPP_CUDA_ERROR_CHECK = r"""
#define CUTE_DSL_CUDA_ERROR_CHECK(err)                                     \
  do {                                                                     \
    const cudaError_t cutedsl_err_ = (err);                                \
    AOTI_RUNTIME_CHECK(                                                    \
        cutedsl_err_ == cudaSuccess,                                       \
        std::string("CuTeDSL kernel module: ") +                           \
            cudaGetErrorName(cutedsl_err_) + ": " +                        \
            cudaGetErrorString(cutedsl_err_));                             \
  } while (0)
""".strip()


def emit_cpp_declarations(exported: CuteDSLExportedKernel, *, aot: bool) -> list[str]:
    """File-scope C++ for one exported kernel, to emit once per prefix before any
    launch (e.g. into the wrapper header), after CPP_HEADERS and CPP_CUDA_ERROR_CHECK:
    the export_to_c header and ``<prefix>_module_ref``, plus, without ``aot``,
    ``<prefix>_module(device)``.

    The kernel's CUDA library is loaded lazily, as AOTInductor loads a Triton kernel's
    module at its first launch. ``<prefix>_module_ref::get(device)``, which returns
    the module a launch passes, takes a reference to the library on its first call
    (the first reference loads it: the object's cuda_init, which loads it
    independently of any device) and loads it onto ``device`` on the first call for
    that device, by any reference, since it was loaded (cuda_load_to_device, which
    resolves the kernels and sets their attributes for one device); the last
    reference unloads it. These are the steps of the header's Module_Load
    (parse_exported_header checks that it runs them as here), which runs the second
    on every device cudaGetDeviceCount reports: the object's cubin is built for one
    architecture, so where a visible GPU could not run it, Module_Load would fail even
    for a model that runs on one that can. The steps pass the device to the CUDA calls
    they make: neither changes the current device nor creates a CUDA context.

    Once a reference has returned the library for a device (one of the first 64), it
    returns it for that device without locking, as a Triton launch takes no lock: the
    library stays loaded onto the device while the reference holds it.

    With ``aot``, no ``<prefix>_module`` is emitted: a caller is to hold one
    ``<prefix>_module_ref`` per model instance, so that the library is released with
    the last instance that launched the kernel, as their Triton modules are, even
    when dlclose leaves the model .so mapped. Otherwise (the JIT wrapper)
    ``<prefix>_module(device)`` holds one from its first launch until the process
    exits.

    A step that fails throws (CPP_CUDA_ERROR_CHECK) once the CUDA runtime's last error
    is cleared, as C10_CUDA_CHECK clears it (a failed unload clears it too): where the
    object's CUDA calls go to PyTorch's libcudart, PyTorch's next kernel launch check
    would report it otherwise. A failed cuda_init unloads what it loaded; a failed
    cuda_load_to_device leaves the library loaded for the devices it is loaded onto.
    The next launch tries again.
    """
    from torch._inductor.codegen.cpp_wrapper_gpu import cpp_string_literal

    p = exported.prefix
    abi = exported.abi
    header = "\n".join(
        line
        for line in exported.header_text.splitlines()
        if line.strip() != "#pragma once"
    )
    unload_error = cpp_string_literal(f"Failed to unload CuTeDSL kernel module {p}: ")
    header_name = os.path.basename(exported.header_path)
    return [
        f"// CuTeDSL user kernel {p}: export_to_c header {header_name}",
        f"#ifndef CUTEDSL_DECLARED_{p}",
        f"#define CUTEDSL_DECLARED_{p}",
        *header.splitlines(),
        "namespace {",
        f"std::mutex {p}_module_mutex;",
        f"int64_t {p}_module_refs = 0;",
        f"{abi.module_ctype} {p}_loaded_module{{}};",
        "// Whether the library is loaded onto each device, since it was loaded.",
        f"std::vector<bool> {p}_module_devices;",
        f"void {p}_cuda_init(cudaLibrary_t* library) {{",
        "  cudaError_t ret = cudaSuccess;",
        "  struct {",
        "    cudaLibrary_t** library;",
        "    cudaError_t* ret;",
        "  } args = {&library, &ret};",
        f"  _mlir_{p}_cuda_init((void**)(&args));",
        "  CUTE_DSL_CUDA_ERROR_CHECK(ret);",
        "}",
        f"void {p}_cuda_load_to_device(cudaLibrary_t* library, int32_t device) {{",
        "  cudaError_t ret = cudaSuccess;",
        "  struct {",
        "    cudaLibrary_t** library;",
        "    int32_t* device;",
        "    cudaError_t* ret;",
        "  } args = {&library, &device, &ret};",
        f"  _mlir_{p}_cuda_load_to_device((void**)(&args));",
        "  CUTE_DSL_CUDA_ERROR_CHECK(ret);",
        "}",
        f"class {p}_module_ref {{",
        " public:",
        f"  {p}_module_ref() = default;",
        f"  {p}_module_ref(const {p}_module_ref&) = delete;",
        f"  {p}_module_ref& operator=(const {p}_module_ref&) = delete;",
        f"  {abi.module_ctype}* get(int32_t device) {{",
        "    const uint64_t bit =",
        "        static_cast<uint32_t>(device) < 64 ? uint64_t{1} << device : 0;",
        "    if (devices_.load(std::memory_order_acquire) & bit) {",
        f"      return &{p}_loaded_module;",
        "    }",
        f"    std::lock_guard<std::mutex> lock({p}_module_mutex);",
        "    try {",
        "      if (!held_) {",
        f"        if ({p}_module_refs == 0) {{",
        f"          {p}_loaded_module = {{}};",
        f"          {p}_module_devices.clear();",
        "          try {",
        f"            {p}_cuda_init(&{p}_loaded_module.module);",
        "          } catch (...) {",
        f"            if ({p}_loaded_module.module != nullptr) {{",
        f"              cudaLibraryUnload({p}_loaded_module.module);",
        f"              {p}_loaded_module = {{}};",
        "            }",
        "            throw;",
        "          }",
        "        }",
        f"        ++{p}_module_refs;",
        "        held_ = true;",
        "      }",
        f"      if ({p}_module_devices.size() <= static_cast<size_t>(device)) {{",
        f"        {p}_module_devices.resize(device + 1);",
        "      }",
        f"      if (!{p}_module_devices[device]) {{",
        f"        {p}_cuda_load_to_device(&{p}_loaded_module.module, device);",
        f"        {p}_module_devices[device] = true;",
        "      }",
        "    } catch (...) {",
        "      (void)cudaGetLastError();",
        "      throw;",
        "    }",
        "    devices_.fetch_or(bit, std::memory_order_release);",
        f"    return &{p}_loaded_module;",
        "  }",
        f"  ~{p}_module_ref() {{",
        "    if (!held_) {",
        "      return;",
        "    }",
        "    try {",
        f"      std::lock_guard<std::mutex> lock({p}_module_mutex);",
        f"      if (--{p}_module_refs == 0) {{",
        f"        {abi.module_ctype} module = {p}_loaded_module;",
        f"        {p}_loaded_module = {{}};",
        f"        {abi.unload_fn}(&module);",
        "      }",
        "    } catch (const std::exception& e) {",
        f"      std::cerr << {unload_error} << e.what() << '\\n';",
        "      (void)cudaGetLastError();",
        "    }",
        "  }",
        "",
        " private:",
        "  bool held_ = false;",
        "  // Devices get() has returned the library for: it stays loaded onto them",
        "  // while this reference holds it.",
        "  std::atomic<uint64_t> devices_{0};",
        "};",
        *(
            []
            if aot
            else [
                f"{abi.module_ctype}* {p}_module(int32_t device) {{",
                f"  static {p}_module_ref ref;",
                "  return ref.get(device);",
                "}",
            ]
        ),
        "}  // namespace",
        f"#endif  // CUTEDSL_DECLARED_{p}",
    ]


def emit_cpp_launch(
    exported: CuteDSLExportedKernel,
    *,
    module_expr: str,
    tensor_args: Mapping[str, tuple[str, Sequence[str], Sequence[str]]],
    scalar_args: Mapping[str, str],
    stream_expr: str,
    check_align: Mapping[str, int],
) -> list[str]:
    """A C++ block that launches the kernel once; ``emit_cpp_declarations`` must be in
    scope. All arguments are C++ expressions:

    * ``module_expr``: the ``<prefix>_Kernel_Module_t*`` to launch with, which
      ``<prefix>_module_ref::get`` returns;
    * ``tensor_args[name] = (data_ptr, sizes, strides)``: a pointer convertible to
      ``void*``, then one int64 expression per dynamic size and per dynamic stride,
      in dimension order;
    * ``scalar_args[name]``: the value of each runtime scalar (a ``uint64_t`` one
      as an unsigned value: the caller checks its range);
    * ``stream_expr``: the ``cudaStream_t`` to launch on;
    * ``check_align[name]``: tensors whose data pointer must be checked against the
      alignment the kernel was compiled with (skip those PT2 already guarantees).

    The block checks alignment and that every value fits the narrower C type the ABI
    uses (int32 dynamic sizes, Int32 scalars...), fills the tensor structs, launches
    and turns a non-zero launch status into an exception, after clearing the CUDA
    runtime's last error (see emit_cpp_declarations).
    """
    from torch._inductor.codegen.cpp_wrapper_gpu import cpp_string_literal

    abi = exported.abi
    p = exported.prefix
    structs = {s.param: s for s in abi.tensor_structs}
    scalars = [n for c, n in abi.params if n not in structs and c != "cudaStream_t"]
    booleans = OrderedSet(
        s.name
        for s in exported.specs
        if isinstance(s, ScalarParamSpec)
        and getattr(s.annotation, "__name__", None) == "Boolean"
    )
    if OrderedSet(tensor_args) != OrderedSet(structs):
        raise ValueError(
            f"{p}: tensor_args {sorted(tensor_args)} != tensors {sorted(structs)}"
        )
    if OrderedSet(scalar_args) != OrderedSet(scalars):
        raise ValueError(
            f"{p}: scalar_args {sorted(scalar_args)} != scalars {sorted(scalars)}"
        )
    if not OrderedSet(check_align) <= OrderedSet(structs):
        extra = sorted(OrderedSet(check_align) - OrderedSet(structs))
        raise ValueError(f"{p}: check_align names {extra} are not tensors")

    lines = ["{", f"  // CuTeDSL user kernel {p}"]

    def check(cond: str, msg: str) -> None:
        lines.append(
            f"  AOTI_RUNTIME_CHECK({cond}, "
            f"{cpp_string_literal(f'CuTeDSL kernel {p}: {msg}')});"
        )

    call = [module_expr]
    for i, (ctype, name) in enumerate(abi.params):
        v = f"cutedsl_arg{i}"
        if name in structs:
            st = structs[name]
            data_ptr, sizes, strides = tensor_args[name]
            if (
                len(sizes) != st.num_dynamic_shapes
                or len(strides) != st.num_dynamic_strides
            ):
                raise ValueError(
                    f"{p}: {name} needs {st.num_dynamic_shapes} dynamic sizes and "
                    f"{st.num_dynamic_strides} dynamic strides, got {len(sizes)} and "
                    f"{len(strides)}"
                )
            lines.append(f"  void* {v}_data = {data_ptr};  // {name}")
            if name in check_align:
                align = check_align[name]
                if align <= 0 or align & (align - 1):
                    raise ValueError(
                        f"{p}: check_align[{name}]={align} is not a power of two"
                    )
                check(
                    f"reinterpret_cast<std::uintptr_t>({v}_data) % {align} == 0",
                    f"argument {name} is not {align}-byte aligned",
                )
            init = [f"{v}_data"]
            for j, size in enumerate(sizes):
                lines.append(
                    f"  const int64_t {v}_size{j} = static_cast<int64_t>({size});"
                )
                check(
                    f"{v}_size{j} >= 0 && {v}_size{j} <= 2147483647LL",
                    f"dynamic size {j} of argument {name} does not fit in int32",
                )
            if sizes:
                init.append(
                    "{"
                    + ", ".join(
                        f"static_cast<int32_t>({v}_size{j})" for j in range(len(sizes))
                    )
                    + "}"
                )
            if strides:
                init.append(
                    "{" + ", ".join(f"static_cast<int64_t>({s})" for s in strides) + "}"
                )
            lines.append(f"  {st.ctype} {v}{{{', '.join(init)}}};")
            call.append(f"&{v}")
        elif ctype == "cudaStream_t":
            call.append(f"static_cast<cudaStream_t>({stream_expr})")
        else:
            call.append(v)
            expr = scalar_args[name]
            if ctype in ("float", "double", "uint64_t"):
                lines.append(
                    f"  const {ctype} {v} = static_cast<{ctype}>({expr});  // {name}"
                )
                continue
            if ctype == "__half_raw":
                # Rounded once from double, as the DSL converts a Python float.
                lines.append(
                    f"  const __half_raw {v} = "
                    f"__double2half(static_cast<double>({expr}));  // {name}"
                )
                continue
            if name in booleans:
                # Truthiness, as the DSL converts a Python value for a Boolean (the
                # kernel reads only the low bit).
                lines.append(
                    f"  const uint8_t {v} = static_cast<uint8_t>(({expr}) != 0);"
                    f"  // {name}"
                )
                continue
            lines.append(
                f"  const int64_t {v}_value = static_cast<int64_t>({expr});  // {name}"
            )
            if ctype in _INT_CTYPE_RANGES:
                lo, hi = _INT_CTYPE_RANGES[ctype]
                check(
                    f"{v}_value >= {lo}LL && {v}_value <= {hi}LL",
                    f"scalar {name} does not fit in {ctype}",
                )
            elif ctype != "int64_t":
                raise CuteDSLExportError(
                    f"{p}: unsupported scalar C type {ctype} for {name}"
                )
            lines.append(f"  const {ctype} {v} = static_cast<{ctype}>({v}_value);")
    lines.append(
        f"  const int32_t cutedsl_status = {abi.wrapper_fn}({', '.join(call)});"
    )
    status = "static_cast<cudaError_t>(cutedsl_status)"
    message = cpp_string_literal(f"CuTeDSL kernel {p}: launch failed: ")
    lines += ["  if (cutedsl_status != 0) {", "    (void)cudaGetLastError();", "  }"]
    lines.append(
        f"  AOTI_RUNTIME_CHECK(cutedsl_status == 0, std::string({message}) + "
        f'cudaGetErrorName({status}) + ": " + cudaGetErrorString({status}));'
    )
    lines.append("}")
    return lines
