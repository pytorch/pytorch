"""Export and link support for user-defined CuTeDSL kernels in the C++ wrapper
(AOTInductor, and torch.compile with cpp_wrapper).

A ``@cute.jit`` host function is compiled against fake tensors that mirror the
layouts Inductor froze for it, and exported with ``export_to_c``, which writes a
C header plus a PIC object. The header is spliced into wrapper.cpp; the object,
the DSL's static CUDA-runtime shim archive and a few linker flags go into the
link of the model .so, or of the JIT wrapper's module. This module owns those
artifacts and the C++ that loads and launches them; it takes parameter specs, not
IR nodes (see cpp_wrapper_gpu.cutedsl_param_specs).

The DSL is imported lazily: importing this module never needs cutlass. Do not add
``from __future__ import annotations``: the DSL reads annotations at runtime and
string annotations break its Constexpr detection.
"""

import functools
import hashlib
import inspect
import logging
import os
import re
import shlex
import shutil
import struct
import tempfile
import threading
import types
import uuid
import weakref
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, TypeGuard

import torch
from torch.utils._ordered_set import OrderedSet


__all__ = [
    "CPP_CUDA_ERROR_CHECK",
    "CPP_HEADERS",
    "CPP_INTEGER_ARG",
    "ConstantParamSpec",
    "CuteDSLExportedKernel",
    "CuteDSLExportError",
    "ExportedABI",
    "ExportedTensorStruct",
    "ParamSpec",
    "ScalarParamSpec",
    "StreamParamSpec",
    "TensorParamSpec",
    "cutedsl_cache_dir",
    "cutedsl_kernel_prefix",
    "cutedsl_link_flags",
    "cutedsl_link_inputs",
    "cutedsl_runtime_archive",
    "emit_cpp_declarations",
    "emit_cpp_launch",
    "export_cutedsl_kernel",
    "export_for_link",
    "host_target_option",
    "parse_exported_header",
    "scalar_ctype",
    "validate_exported_abi",
]

log = logging.getLogger(__name__)

# Bump when the artifacts this module produces change for the same inputs.
_FORMAT_VERSION = 1
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_ARCH = re.compile(r"sm_[0-9]+[a-z]?\Z")
_OBJECT_ADDRESS = re.compile(r" at 0x[0-9A-Fa-f]+")


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

    @property
    def num_dynamic_sizes(self) -> int:
        return sum(s is None for s in self.sizes)

    @property
    def num_dynamic_strides(self) -> int:
        return sum(s is None for s in self.strides)


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

    def tensor_struct(self, param: str) -> ExportedTensorStruct | None:
        return next((s for s in self.tensor_structs if s.param == param), None)


@dataclass(frozen=True)
class CuteDSLExportedKernel:
    """Header + object written by ``export_cutedsl_kernel`` for ``specs``."""

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


def validate_exported_abi(abi: ExportedABI, specs: Sequence[ParamSpec]) -> None:
    """Raise CuteDSLExportError unless the header ABI is exactly what ``specs`` ask for:
    one wrapper parameter per non-constant spec, in order, with the C types and
    dynamic size/stride slot counts the C++ launch fills in."""
    where = abi.wrapper_fn
    if abi.wrapper_return_ctype != "int32_t":
        raise CuteDSLExportError(
            f"{where} returns {abi.wrapper_return_ctype}, expected the int32_t launch "
            "status; host functions with a return annotation are not supported"
        )
    runtime = [s for s in specs if not isinstance(s, ConstantParamSpec)]
    got = [name for _, name in abi.params]
    want = [s.name for s in runtime]
    if got != want:
        raise CuteDSLExportError(f"{where} parameters {got} != expected {want}")
    tensor_names = OrderedSet(
        [s.name for s in runtime if isinstance(s, TensorParamSpec)]
    )
    extra = [s.ctype for s in abi.tensor_structs if s.param not in tensor_names]
    if extra:
        raise CuteDSLExportError(f"{where}: unexpected tensor structs {extra}")
    for (ctype, name), spec in zip(abi.params, runtime):
        if isinstance(spec, TensorParamSpec):
            struct_ = abi.tensor_struct(name)
            if struct_ is None:
                raise CuteDSLExportError(f"{where}: no tensor struct for {name}")
            expected = (
                f"{struct_.ctype}*",
                "int32_t" if spec.num_dynamic_sizes else None,
                spec.num_dynamic_sizes,
                "int64_t" if spec.num_dynamic_strides else None,
                spec.num_dynamic_strides,
            )
            actual = (
                ctype,
                struct_.shape_ctype,
                struct_.num_dynamic_shapes,
                struct_.stride_ctype,
                struct_.num_dynamic_strides,
            )
            if actual != expected:
                raise CuteDSLExportError(
                    f"{where}: tensor {name} has (param ctype, shape ctype, #shapes, "
                    f"stride ctype, #strides) = {actual}, expected {expected} for "
                    f"sizes={spec.sizes} strides={spec.strides}"
                )
        elif isinstance(spec, ScalarParamSpec):
            if ctype != scalar_ctype(spec.annotation):
                raise CuteDSLExportError(
                    f"{where}: scalar {name} is {ctype}, expected "
                    f"{scalar_ctype(spec.annotation)} for {spec.annotation!r}"
                )
        elif ctype != "cudaStream_t":
            raise CuteDSLExportError(
                f"{where}: stream {name} is {ctype}, expected cudaStream_t"
            )


def _elf_global_symbols(path: str) -> list[str]:
    """Names of the global/weak symbols an ELF64 little-endian object defines."""
    data = Path(path).read_bytes()
    if data[:4] != b"\x7fELF" or data[4] != 2 or data[5] != 1:
        raise CuteDSLExportError(f"{path} is not an ELF64 little-endian object")
    (shoff,) = struct.unpack_from("<Q", data, 0x28)
    shentsize, shnum = struct.unpack_from("<HH", data, 0x3A)
    sections = [
        struct.unpack_from("<IIQQQQIIQQ", data, shoff + i * shentsize)
        for i in range(shnum)
    ]
    names = []
    for _, sh_type, _, _, offset, size, link, _, _, entsize in sections:
        if sh_type != 2:  # SHT_SYMTAB
            continue
        str_off = sections[link][4]
        for pos in range(offset, offset + size, entsize):
            st_name, st_info, _, st_shndx = struct.unpack_from("<IBBH", data, pos)
            if st_info >> 4 in (1, 2) and st_shndx != 0:  # STB_GLOBAL/STB_WEAK, defined
                end = data.index(b"\0", str_off + st_name)
                names.append(data[str_off + st_name : end].decode())
    return names


def _check_object_symbols(path: str, prefix: str) -> None:
    # cutedsl_link_flags hides "<prefix>_*" and "_mlir_<prefix>_*"; any other global
    # symbol would leak from the model .so and could clash between kernels.
    stray = [
        s
        for s in _elf_global_symbols(path)
        if not s.startswith((f"{prefix}_", f"_mlir_{prefix}_"))
    ]
    if stray:
        raise CuteDSLExportError(
            f"{path} defines global symbols outside {prefix}_*: {stray}"
        )


def _is_dsl_function(obj: object) -> TypeGuard[Callable[..., Any]]:
    # @cute.jit and @cute.kernel return functools.wraps wrappers; both the wrapper and
    # the user function carry ``_dsl_cls`` (a CuTeDSL subclass).
    return callable(obj) and getattr(obj, "_dsl_cls", None) is not None


def _code_objects(code: types.CodeType) -> Iterable[types.CodeType]:
    yield code
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            yield from _code_objects(const)


def _function_source(fn: Callable[..., Any]) -> str:
    try:
        source = inspect.getsource(fn)
    except (OSError, TypeError):
        code = getattr(fn, "__code__", None)
        return code.co_code.hex() if code is not None else repr(fn)
    # From the def line: the DSL's first compile of a function replaces its code with
    # one that starts there, so getsource stops returning the decorators.
    match = re.search(r"^[ \t]*(async[ \t]+)?def\b", source, re.MULTILINE)
    return source[match.start() :] if match else source


def _dsl_function_closure(host_fn: Callable[..., Any]) -> list[str]:
    """Source of the host function and of every @cute.jit/@cute.kernel function it
    references through globals, module attributes and closures, transitively, plus
    the values of the simple constants those functions read (both are baked into the
    generated code). DSL functions from the cutlass package itself are covered by
    cutlass.__version__."""
    parts: list[str] = []
    seen: OrderedSet[int] = OrderedSet()
    stack: list[Any] = [host_fn]
    while stack:
        fn = inspect.unwrap(stack.pop())
        if id(fn) in seen:
            continue
        seen.add(id(fn))
        parts.append(f"{fn.__module__}.{fn.__qualname__}\n{_function_source(fn)}")
        code = getattr(fn, "__code__", None)
        if code is None:
            continue
        names = sorted(OrderedSet([n for c in _code_objects(code) for n in c.co_names]))
        fn_globals = getattr(fn, "__globals__", {})
        refs = [(n, fn_globals[n]) for n in names if n in fn_globals]
        for name, cell in zip(code.co_freevars, fn.__closure__ or ()):
            try:
                refs.append((name, cell.cell_contents))
            except ValueError:  # empty cell
                continue
        # ``mod.kernel(...)``: the attribute name is in co_names too.
        for mod_name, mod in [r for r in refs if isinstance(r[1], types.ModuleType)]:
            attrs = vars(mod)
            refs.extend(
                (f"{mod_name}.{n}", attrs[n])
                for n in names
                if _is_dsl_function(attrs.get(n))
            )
        for name, value in refs:
            if _is_dsl_function(value):
                if not (inspect.unwrap(value).__module__ or "").startswith("cutlass."):
                    stack.append(value)
            elif isinstance(
                value, (bool, int, float, str, bytes, type(None), torch.dtype)
            ):
                parts.append(f"{name}={value!r}")
    return parts


def _stable_repr(value: object, inside: frozenset[int] = frozenset()) -> str:
    """repr() made independent of the process, so that the kernel prefix is the same
    in every process for equal constants: objects that constant_key compares by
    content (see user_kernel_hop._object_state) by type and contents, set items
    sorted, and object addresses removed from any other repr. A constant whose own
    repr differs between processes in another way still changes the prefix.

    A value inside itself (``inside`` holds the ids of the values being walked) is
    refused with a ValueError, as constant_key refuses one: constant_key does not
    walk a frozenset, so a cycle through one gets here."""
    from torch._higher_order_ops.cutedsl_kernel_wrap import CUTEDSL_CONSTANT_RULES
    from torch._higher_order_ops.user_kernel_hop import _object_state

    if value is None or isinstance(value, (bool, int, float, str, bytes, torch.dtype)):
        return repr(value)
    if id(value) in inside:
        what = type(value).__name__
        raise ValueError(f"a constant kernel argument of type {what} refers to itself")
    rec = functools.partial(_stable_repr, inside=inside | {id(value)})
    if isinstance(value, (tuple, list)):
        return f"{type(value).__name__}({', '.join(rec(v) for v in value)})"
    if isinstance(value, (set, frozenset)):
        items = sorted(rec(v) for v in value)
        return f"{type(value).__name__}({', '.join(items)})"
    if isinstance(value, dict):
        items = sorted((rec(k), rec(v)) for k, v in value.items())
        return "dict(" + ", ".join(f"{k}: {v}" for k, v in items) + ")"
    if isinstance(value, Enum):
        return f"{type(value).__module__}.{type(value).__qualname__}.{value.name}"
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    if isinstance(value, (types.FunctionType, types.MethodType)) or _is_dsl_function(
        value
    ):
        return "fn(" + "\n".join(_dsl_function_closure(value)) + ")"
    name = f"{type(value).__module__}.{type(value).__qualname__}"
    state = _object_state(value, CUTEDSL_CONSTANT_RULES)
    if state is not None:
        return f"{name}({rec(state)})"
    return f"{name}:{_OBJECT_ADDRESS.sub('', repr(value))}"


def _spec_key(spec: ParamSpec) -> str:
    if isinstance(spec, TensorParamSpec):
        return f"T:{spec.name}:{spec.dtype}:{spec.sizes}:{spec.strides}:{spec.align}"
    if isinstance(spec, ScalarParamSpec):  # the hint is not baked in
        return f"S:{spec.name}:{_stable_repr(spec.annotation)}"
    if isinstance(spec, ConstantParamSpec):
        return f"C:{spec.name}:{_stable_repr(spec.value)}"
    return f"R:{spec.name}"


def cutedsl_kernel_prefix(
    host_fn: Callable[..., Any], specs: Sequence[ParamSpec], arch: str
) -> str:
    """A readable, C-identifier-safe symbol prefix for one compiled specialization
    of ``host_fn``: a hash of the DSL version, the kernel sources (transitively), the
    plain constants they read, the parameter specs and the target architecture.

    It is not an identity: values it cannot see (a captured list, a helper's
    globals) can differ under the same prefix. export_for_link keys exports by host
    function and specs and keeps prefixes unique within a library."""
    import cutlass

    from torch._higher_order_ops.cutedsl_kernel_wrap import cutedsl_kernel_name

    if not _ARCH.match(arch):
        raise ValueError(f"arch must look like 'sm_90a', got {arch!r}")
    fn = inspect.unwrap(host_fn)
    spec_keys = []
    for spec in specs:
        try:
            spec_keys.append(_spec_key(spec))
        except ValueError as e:
            name = cutedsl_kernel_name(host_fn)
            raise ValueError(
                f"CuTeDSL kernel {name}: parameter {spec.name!r}: {e}"
            ) from None
    h = hashlib.sha256()
    for part in (
        f"format={_FORMAT_VERSION}",
        f"cutlass={cutlass.__version__}",
        f"arch={arch}",
        f"{fn.__module__}.{fn.__qualname__}",
        *_dsl_function_closure(host_fn),
        *spec_keys,
    ):
        h.update(part.encode())
        h.update(b"\0")
    name = re.sub(r"[^0-9A-Za-z]+", "_", fn.__name__).strip("_")[:32] or "kernel"
    return f"cutedsl_{name}_{arch.replace('_', '')}_{h.hexdigest()[:16]}"


def _cute_dtype(dtype: torch.dtype) -> Any:
    from torch._native.cutedsl.dtypes import torch2cute

    if dtype not in torch2cute:
        raise NotImplementedError(
            f"CuTeDSL user kernels: tensor dtype {dtype} is not supported"
        )
    return torch2cute[dtype]


def _fake_args(specs: Sequence[ParamSpec]) -> list[Any]:
    """Compile-time stand-ins: fake tensors with a fresh 32-bit symbol per dynamic size
    and a 64-bit symbol per dynamic stride, hint values for runtime scalars, the
    constants themselves and a fake stream."""
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_fake_stream, make_fake_tensor

    args: list[Any] = []
    for spec in specs:
        if isinstance(spec, TensorParamSpec):
            shape = tuple(cute.sym_int32() if s is None else s for s in spec.sizes)
            stride = tuple(cute.sym_int64() if s is None else s for s in spec.strides)
            args.append(
                make_fake_tensor(
                    _cute_dtype(spec.dtype), shape, stride, assumed_align=spec.align
                )
            )
        elif isinstance(spec, ScalarParamSpec):
            args.append(spec.annotation(spec.hint))
        elif isinstance(spec, ConstantParamSpec):
            args.append(spec.value)
        else:
            args.append(make_fake_stream())
    return args


# export_to_c gives its C wrapper's parameters the host function's names, next to the
# names it declares there itself (4.4.2: the module parameter, the ret and args
# locals). Those and the C++ keywords that are not Python keywords do not compile.
_UNEXPORTABLE_NAMES = frozenset(
    ["module", "ret", "args"]
    + ["alignas", "alignof", "and_eq", "asm", "auto", "bitand", "bitor", "bool", "case"]
    + ["catch", "char", "char8_t", "char16_t", "char32_t", "compl", "concept", "const"]
    + ["consteval", "constexpr", "constinit", "const_cast", "co_await", "co_return"]
    + ["co_yield", "decltype", "default", "delete", "do", "double", "dynamic_cast"]
    + ["enum", "explicit", "export", "extern", "false", "float", "friend", "goto"]
    + ["inline", "int", "long", "mutable", "namespace", "new", "noexcept", "not_eq"]
    + ["nullptr", "operator", "or_eq", "private", "protected", "public", "register"]
    + ["reinterpret_cast", "requires", "short", "signed", "sizeof", "static"]
    + ["static_assert", "static_cast", "struct", "switch", "template", "this"]
    + ["thread_local", "throw", "true", "typedef", "typeid", "typename", "union"]
    + ["unsigned", "using", "virtual", "void", "volatile", "wchar_t", "xor", "xor_eq"]
)


def _check_specs(host_fn: Callable[..., Any], specs: Sequence[ParamSpec]) -> None:
    params = list(inspect.signature(inspect.unwrap(host_fn)).parameters)
    names = [s.name for s in specs]
    if names != params:
        raise ValueError(
            f"specs {names} must name every parameter of "
            f"{host_fn.__name__}{params} in order"
        )
    if sum(isinstance(s, StreamParamSpec) for s in specs) > 1:
        raise ValueError(f"{host_fn.__name__}: more than one stream parameter")
    unexportable = [
        s.name
        for s in specs
        if not isinstance(s, ConstantParamSpec) and s.name in _UNEXPORTABLE_NAMES
    ]
    if unexportable:
        raise ValueError(
            f"{host_fn.__name__}: rename parameters {unexportable}: export_to_c gives "
            "its C wrapper's parameters the host function's names, which must not be "
            "C++ keywords or module, ret or args (names that wrapper declares itself)"
        )
    for s in specs:
        if isinstance(s, ScalarParamSpec):
            scalar_ctype(s.annotation)  # fail before a compile the export would reject


def host_target_option() -> str | None:
    """The cute.compile option that compiles an export's host code (its module load
    and launch wrapper) for the generic CPU of this machine's architecture
    (torch._native.cutedsl.export.host_target), so that it also runs on hosts with
    older CPUs; None for a DSL without --host-target (nvidia-cutlass-dsl 4.6.1 and
    later have it), which compiles it for this machine's CPU."""
    import cutlass.base_dsl.compiler

    from torch._native.cutedsl.export import host_target

    if getattr(cutlass.base_dsl.compiler, "HostTarget", None) is None:
        return None
    return f"--host-target {shlex.quote(host_target())}"


def export_cutedsl_kernel(
    host_fn: Callable[..., Any],
    specs: Sequence[ParamSpec],
    *,
    out_dir: str,
    prefix: str,
    arch: str,
) -> CuteDSLExportedKernel:
    """Compile ``host_fn`` for ``specs`` and ``arch`` and export it as
    ``out_dir/<prefix>.h`` + ``out_dir/<prefix>.o``.

    The object's host code is compiled for the generic CPU of this machine's
    architecture where the DSL can do so (see host_target_option).

    Files are written to a private temporary directory, validated (header ABI vs
    specs, object symbols vs prefix) and only then moved into ``out_dir``, the header
    last, so a present header means a complete pair.
    """
    specs = tuple(specs)
    _check_specs(host_fn, specs)
    if not _IDENTIFIER.match(prefix):
        raise ValueError(f"prefix {prefix!r} is not a C identifier")
    if not _ARCH.match(arch):
        raise ValueError(f"arch must look like 'sm_90a', got {arch!r}")
    header_path = os.path.join(out_dir, f"{prefix}.h")
    object_path = os.path.join(out_dir, f"{prefix}.o")

    import cutlass.cute as cute

    from torch._native.cutedsl.compile_lock import CUTEDSL_COMPILE_LOCK
    from torch._native.cutedsl.export import warm_up_exporter

    options = f"--gpu-arch {arch}"
    host_opt = host_target_option()
    if host_opt is not None:
        options += f" {host_opt}"
    warm_up_exporter()
    os.makedirs(out_dir, exist_ok=True)
    tmp_dir = tempfile.mkdtemp(prefix=f".{prefix}.", dir=out_dir)
    try:
        args = _fake_args(specs)
        with CUTEDSL_COMPILE_LOCK:
            compiled = cute.compile(host_fn, *args, options=options)
            compiled.export_to_c(
                file_path=tmp_dir, file_name=prefix, function_prefix=prefix
            )
        tmp_header = os.path.join(tmp_dir, f"{prefix}.h")
        tmp_object = os.path.join(tmp_dir, f"{prefix}.o")
        text = Path(tmp_header).read_text()
        abi = parse_exported_header(text, prefix)
        validate_exported_abi(abi, specs)
        _check_object_symbols(tmp_object, prefix)
        os.replace(tmp_object, object_path)
        os.replace(tmp_header, header_path)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return CuteDSLExportedKernel(prefix, header_path, object_path, text, abi, specs)


# host function -> {export key -> its exports, one per prefix it is linked under}
_exports: weakref.WeakKeyDictionary[
    Callable[..., Any], dict[tuple[Any, ...], list[CuteDSLExportedKernel]]
] = weakref.WeakKeyDictionary()
_exports_lock = threading.Lock()


def _export_key(specs: Sequence[ParamSpec], arch: str) -> tuple[Any, ...]:
    from torch._higher_order_ops.cutedsl_kernel_wrap import CUTEDSL_CONSTANT_RULES
    from torch._higher_order_ops.user_kernel_hop import constant_key

    return (
        arch,
        *(
            (s.name, constant_key(s.value, CUTEDSL_CONSTANT_RULES))
            if isinstance(s, ConstantParamSpec)
            else _spec_key(s)
            for s in specs
        ),
    )


def cutedsl_cache_dir() -> str:
    """Where the exports and the linker version scripts are stored, named by
    content."""
    from torch._inductor.runtime.cache_dir_utils import cache_dir

    return os.path.join(cache_dir(), "cutedsl_user_kernels")


def export_for_link(
    host_fn: Callable[..., Any],
    specs: Sequence[ParamSpec],
    arch: str,
    *,
    linked: Mapping[str, CuteDSLExportedKernel],
    out_root: str,
) -> CuteDSLExportedKernel:
    """The export of ``host_fn`` for ``specs`` and ``arch``, to link into a library
    that already links ``linked`` (prefix -> export), under a prefix no other kernel
    of the library uses.

    Exports are reused within this process only, keyed by the host function object
    and the specs: the prefix does not cover every value a kernel reads, so a new
    process exports again. The files go to ``out_root/<prefix>.<hash of their
    bytes>``, so exporting the same kernel again under the same prefix (see
    _stable_repr) yields the same paths (and with them the same JIT wrapper cache
    key) and no new files.
    """
    specs = tuple(specs)
    with _exports_lock:
        cached = _exports.setdefault(host_fn, {}).setdefault(
            _export_key(specs, arch), []
        )
        # An export whose cache directory was removed since cannot be linked.
        cached[:] = [e for e in cached if os.path.exists(e.object_path)]
        for exported in cached:
            if linked.get(exported.prefix, exported) is exported:
                return exported
    prefix = base = cutedsl_kernel_prefix(host_fn, specs, arch)
    n = 0
    while prefix in linked:
        n += 1
        prefix = f"{base}_{n}"
    os.makedirs(out_root, exist_ok=True)
    # The umask's mode, not mkdtemp's 0700: it becomes the cache entry, which other
    # users of a shared cache directory read.
    tmp_dir = os.path.join(out_root, f".{prefix}.{uuid.uuid4().hex}")
    os.mkdir(tmp_dir)
    try:
        exported = export_cutedsl_kernel(
            host_fn, specs, out_dir=tmp_dir, prefix=prefix, arch=arch
        )
        digest = hashlib.sha256()
        for path in (exported.header_path, exported.object_path):
            digest.update(Path(path).read_bytes())
        out_dir = os.path.join(out_root, f"{prefix}.{digest.hexdigest()[:16]}")
        header_path, object_path = (
            os.path.join(out_dir, f"{prefix}{ext}") for ext in (".h", ".o")
        )
        try:
            os.rename(tmp_dir, out_dir)
        except OSError:
            # Stored before, by this or another process: the same bytes. A damaged
            # entry (e.g. files deleted by a cache cleanup) is repaired in place:
            # other processes may be using it, and os.replace leaves no moment
            # where a path they were given is missing.
            if not (os.path.exists(header_path) and os.path.exists(object_path)):
                os.makedirs(out_dir, exist_ok=True)
                os.replace(exported.object_path, object_path)
                os.replace(exported.header_path, header_path)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    exported = replace(exported, header_path=header_path, object_path=object_path)
    with _exports_lock:
        cached.append(exported)
    return exported


# Standard headers the declarations, launches and CPP_INTEGER_ARG use.
CPP_HEADERS = (
    "atomic",
    "cstdint",
    "exception",
    "iostream",
    "mutex",
    "string",
    "type_traits",
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

# Defined once per wrapper. The C++ wrapper passes a symbolic value for an integer
# parameter through it, so that the C++ type of the expression picks the conversion,
# as it picks the dtype of a symbolic graph output (scalar_to_tensor_handle). A value
# the launcher's int64_t or uint64_t parameter cannot hold (a negative one for
# uint64_t; a floating one out of range, or NaN, whose conversion is undefined) is
# refused; a floating one is truncated toward zero, as the DSL converts a float.
CPP_INTEGER_ARG = r"""
namespace {
template <typename To, typename From>
To cutedsl_integer_arg(From value, const char* msg) {
  if constexpr (std::is_floating_point_v<From>) {
    AOTI_RUNTIME_CHECK(
        std::is_signed_v<To> ? value >= -0x1p63 && value < 0x1p63
                             : value > -1.0 && value < 0x1p64,
        msg);
  } else if constexpr (std::is_unsigned_v<To> && std::is_signed_v<From>) {
    AOTI_RUNTIME_CHECK(value >= 0, msg);
  }
  return static_cast<To>(value);
}
}  // namespace
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

    With ``aot``, each model instance holds a ``<prefix>_module_ref`` (a member of
    AOTInductorModelKernels named ``<prefix>_ref``), so the library is released with
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
    booleans = {
        s.name
        for s in exported.specs
        if isinstance(s, ScalarParamSpec)
        and getattr(s.annotation, "__name__", None) == "Boolean"
    }
    if OrderedSet(tensor_args) != OrderedSet(structs):
        raise ValueError(
            f"{p}: tensor_args {sorted(tensor_args)} != tensors {sorted(structs)}"
        )
    if OrderedSet(scalar_args) != OrderedSet(scalars):
        raise ValueError(
            f"{p}: scalar_args {sorted(scalar_args)} != scalars {sorted(scalars)}"
        )
    if not OrderedSet(check_align) <= OrderedSet(structs):
        raise ValueError(
            f"{p}: check_align names {sorted(set(check_align) - set(structs))} are "
            "not tensors"
        )

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


@functools.cache
def cutedsl_runtime_archive() -> str:
    """The DSL's static CUDA-runtime shim archive, whose ``_cudaLaunchKernelEx``-style
    shims the exported objects call, for the imported cutlass and torch's CUDA
    version: torch._native.cutedsl.export.runtime_archive, searched from the
    ancestors of the cutlass package, as found and with symlinks resolved."""
    import cutlass

    from torch._native.cutedsl.export import runtime_archive

    roots = [
        str(root)
        for start in (cutlass.__file__, os.path.realpath(cutlass.__file__))
        for root in Path(start).parents
    ]
    major = int(torch.version.cuda.split(".")[0]) if torch.version.cuda else None
    return runtime_archive(roots, major, log.warning)


def cutedsl_link_inputs(kernels: Iterable[CuteDSLExportedKernel]) -> list[str]:
    """Files to add to the model .so link after the wrapper objects: every exported
    object once, then the runtime archive (it must follow the objects using it).
    Like cutedsl_link_flags, returns shell fragments: each path is quoted."""
    objects = list(dict.fromkeys(k.object_path for k in kernels))
    paths = [*objects, cutedsl_runtime_archive()] if objects else []
    return [shlex.quote(path) for path in paths]


def cutedsl_link_flags(out_dir: str, prefixes: Iterable[str]) -> list[str]:
    """Linker flags that keep the CuTeDSL symbols private to the model .so.

    * ``--exclude-libs`` on the runtime archive: its ``_cuda*`` shims must bind
      inside the .so, never to a copy exported by another library in the process
      (e.g. the DSL's own runtime).
    * A version script (written to ``out_dir``) making ``<prefix>_*`` and
      ``_mlir_<prefix>_*`` local; every other symbol keeps its visibility.

    Each element is a shell fragment: CppBuilder joins passthrough args into one
    command line and shlex-splits it. The version script path is a token of its own
    (``-Xlinker <path>``) so fbcode remote builds stage it like other input files.
    """
    names = sorted(OrderedSet(prefixes))
    for p in names:
        if not _IDENTIFIER.match(p):
            raise ValueError(f"prefix {p!r} is not a C identifier")
    if not names:
        return []
    script = (
        "/* CuTeDSL user kernels: export_to_c symbols stay local to this library. */\n"
        "{\n  local:\n"
        + "".join(f"    {p}_*;\n    _mlir_{p}_*;\n" for p in names)
        + "};\n"
    )
    path = os.path.join(
        out_dir, f"cutedsl_{hashlib.sha256(script.encode()).hexdigest()[:16]}.ver"
    )
    if not os.path.exists(path) or Path(path).read_text() != script:
        from torch._inductor.codecache import write_atomic

        # The umask's mode, not mkstemp's 0600: other users of the cache read it.
        write_atomic(path, script, make_dirs=True)
    archive = os.path.basename(cutedsl_runtime_archive())
    return [
        f"-Wl,--exclude-libs,{archive}",
        f"-Xlinker --version-script -Xlinker {shlex.quote(path)}",
    ]
