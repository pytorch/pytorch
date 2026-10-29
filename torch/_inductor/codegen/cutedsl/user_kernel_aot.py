"""Export and link support for user-defined CuTeDSL kernels in the C++ wrapper
(AOTInductor, and torch.compile with cpp_wrapper).

A ``@cute.jit`` host function is compiled against fake tensors that mirror the
layouts Inductor froze for it, and exported with ``export_to_c``, which writes a
C header plus a PIC object. The header is spliced into wrapper.cpp; the object,
the DSL's static CUDA-runtime shim archive and a few linker flags go into the
link of the model .so, or of the JIT wrapper's module. This module owns those
artifacts and the C++ that loads and launches them. It does not depend on the IR
or on the wrapper codegen.

The DSL is imported lazily: importing this module never needs cutlass. Do not add
``from __future__ import annotations``: the DSL reads annotations at runtime and
string annotations break its Constexpr detection.
"""

import functools
import hashlib
import inspect
import os
import re
import shlex
import shutil
import struct
import tempfile
import threading
import types
import weakref
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any

import torch
from torch.utils._ordered_set import OrderedSet


__all__ = [
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
    "parse_exported_header",
    "scalar_ctype",
    "validate_exported_abi",
]

# Bump when the artifacts this module produces change for the same inputs.
_FORMAT_VERSION = 1
RUNTIME_ARCHIVE = "libcuda_dialect_runtime_static.a"
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
    name = getattr(annotation, "__name__", None)
    module = getattr(annotation, "__module__", "") or ""
    if not module.startswith("cutlass") or name not in _SCALAR_CTYPES:
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
    load_fn: str
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
    load_fn, unload_fn = (
        f"{prefix}_Kernel_Module_Load",
        f"{prefix}_Kernel_Module_Unload",
    )
    need(
        rf"\b{p}_Kernel_Module_Load\s*\(\s*{p}_Kernel_Module_t\s*\*",
        "module load function",
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
        load_fn=load_fn,
        unload_fn=unload_fn,
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


def _is_dsl_function(obj: object) -> bool:
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
        return inspect.getsource(fn)
    except (OSError, TypeError):
        code = getattr(fn, "__code__", None)
        return code.co_code.hex() if code is not None else repr(fn)


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


def _stable_repr(value: object) -> str:
    """repr() made independent of the process, so that the kernel prefix is the same
    in every process for equal constants: objects that constant_key compares by
    content (see user_kernel_hop._object_state) by type and contents, set items
    sorted, and object addresses removed from any other repr. A constant whose own
    repr differs between processes in another way still changes the prefix."""
    from torch._higher_order_ops.user_kernel_hop import _object_state

    if value is None or isinstance(value, (bool, int, float, str, bytes, torch.dtype)):
        return repr(value)
    if isinstance(value, (tuple, list)):
        return f"{type(value).__name__}({', '.join(_stable_repr(v) for v in value)})"
    if isinstance(value, (set, frozenset)):
        items = sorted(_stable_repr(v) for v in value)
        return f"{type(value).__name__}({', '.join(items)})"
    if isinstance(value, dict):
        items = sorted((_stable_repr(k), _stable_repr(v)) for k, v in value.items())
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
    state = _object_state(value)
    if state is not None:
        return f"{name}({_stable_repr(state)})"
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

    if not _ARCH.match(arch):
        raise ValueError(f"arch must look like 'sm_90a', got {arch!r}")
    fn = inspect.unwrap(host_fn)
    h = hashlib.sha256()
    for part in (
        f"format={_FORMAT_VERSION}",
        f"cutlass={cutlass.__version__}",
        f"arch={arch}",
        f"{fn.__module__}.{fn.__qualname__}",
        *_dsl_function_closure(host_fn),
        *(_spec_key(s) for s in specs),
    ):
        h.update(part.encode())
        h.update(b"\0")
    name = re.sub(r"[^0-9A-Za-z]+", "_", fn.__name__).strip("_")[:32] or "kernel"
    return f"cutedsl_{name}_{arch.replace('_', '')}_{h.hexdigest()[:16]}"


def _cute_dtype(dtype: torch.dtype) -> Any:
    import cutlass

    from torch._native.cutedsl.dtypes import torch2cute

    extra = {
        torch.int8: cutlass.Int8,
        torch.uint8: cutlass.Uint8,
        torch.int16: cutlass.Int16,
        torch.int64: cutlass.Int64,
        torch.float8_e4m3fn: cutlass.Float8E4M3FN,
        torch.float8_e5m2: cutlass.Float8E5M2,
    }
    cute_dtype = torch2cute.get(dtype, extra.get(dtype))
    if cute_dtype is None:
        raise NotImplementedError(
            f"CuTeDSL user kernels: tensor dtype {dtype} is not supported"
        )
    return cute_dtype


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
    for s in specs:
        if isinstance(s, ScalarParamSpec):
            scalar_ctype(s.annotation)  # fail before a compile the export would reject


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

    from torch._inductor.codegen.cutedsl.compile_lock import CUTEDSL_COMPILE_LOCK

    os.makedirs(out_dir, exist_ok=True)
    tmp_dir = tempfile.mkdtemp(prefix=f".{prefix}.", dir=out_dir)
    try:
        args = _fake_args(specs)
        with CUTEDSL_COMPILE_LOCK:
            compiled = cute.compile[cute.GPUArch(arch)](host_fn, *args)
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
    from torch._higher_order_ops.user_kernel_hop import constant_key

    return (
        arch,
        *(
            (s.name, constant_key(s.value))
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
    tmp_dir = tempfile.mkdtemp(prefix=f".{prefix}.", dir=out_root)
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
        for attempt in range(2):
            try:
                os.rename(tmp_dir, out_dir)
                break
            except OSError:
                # Stored before, by this or another process: the same bytes.
                if os.path.exists(header_path) and os.path.exists(object_path):
                    break
                if attempt:
                    raise
                # A damaged entry, e.g. files deleted by a cache cleanup.
                shutil.rmtree(out_dir, ignore_errors=True)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    exported = replace(exported, header_path=header_path, object_path=object_path)
    with _exports_lock:
        cached.append(exported)
    return exported


def _cpp_str(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


# Once per translation unit (repeating it is harmless). export_to_c headers print and
# carry on when a CUDA call fails; make those failures throw so a failed module load
# becomes an AOTI runtime error instead of a later illegal launch.
_CPP_PRELUDE = r"""#include <cstdint>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#ifndef AOTI_RUNTIME_CHECK
#define AOTI_RUNTIME_CHECK(EXPR, MSG) \
  do {                                \
    if (!(EXPR)) {                    \
      throw std::runtime_error(MSG);  \
    }                                 \
  } while (0)
#endif
#undef CUTE_DSL_CUDA_ERROR_CHECK
#define CUTE_DSL_CUDA_ERROR_CHECK(err)                                     \
  do {                                                                     \
    const cudaError_t cutedsl_err_ = (err);                                \
    AOTI_RUNTIME_CHECK(                                                    \
        cutedsl_err_ == cudaSuccess,                                       \
        std::string("CuTeDSL kernel module: ") +                           \
            cudaGetErrorName(cutedsl_err_) + ": " +                        \
            cudaGetErrorString(cutedsl_err_));                             \
  } while (0)"""


def emit_cpp_declarations(exported: CuteDSLExportedKernel, *, aot: bool) -> list[str]:
    """File-scope C++ for one exported kernel, to emit once per prefix before any
    launch (e.g. into the wrapper header): the CUTE_DSL_CUDA_ERROR_CHECK predefine,
    the export_to_c header, ``<prefix>_module_ref`` and ``<prefix>_module()``, which
    returns the module.

    Module_Load loads the library on every visible device and keeps the kernel
    handles in globals of the object, so every launch from the library the object
    is linked into uses one load. ``<prefix>_module_ref`` objects hold it: the first
    loads it and the last unloads it. With ``aot``, each model instance holds one
    (a member of AOTInductorModelKernels named ``<prefix>_ref``), so the library is
    released with the last instance, as their Triton modules are, even when dlclose
    leaves the model .so mapped. Otherwise (the JIT wrapper) ``<prefix>_module()``
    holds one from the first launch until the wrapper is unloaded or the process
    exits.
    """
    p = exported.prefix
    abi = exported.abi
    header = "\n".join(
        line
        for line in exported.header_text.splitlines()
        if line.strip() != "#pragma once"
    )
    unload_error = _cpp_str(f"Failed to unload CuTeDSL kernel module {p}: ")
    header_name = os.path.basename(exported.header_path)
    return [
        *_CPP_PRELUDE.splitlines(),
        f"// CuTeDSL user kernel {p}: export_to_c header {header_name}",
        f"#ifndef CUTEDSL_DECLARED_{p}",
        f"#define CUTEDSL_DECLARED_{p}",
        *header.splitlines(),
        "namespace {",
        f"std::mutex {p}_module_mutex;",
        f"int64_t {p}_module_refs = 0;",
        f"{abi.module_ctype} {p}_loaded_module{{}};",
        f"struct {p}_module_ref {{",
        f"  {p}_module_ref() {{",
        f"    std::lock_guard<std::mutex> lock({p}_module_mutex);",
        f"    if ({p}_module_refs == 0) {{",
        f"      {abi.load_fn}(&{p}_loaded_module);",
        "    }",
        f"    ++{p}_module_refs;",
        "  }",
        f"  ~{p}_module_ref() {{",
        "    try {",
        f"      std::lock_guard<std::mutex> lock({p}_module_mutex);",
        f"      if (--{p}_module_refs == 0) {{",
        f"        {abi.unload_fn}(&{p}_loaded_module);",
        "      }",
        "    } catch (const std::exception& e) {",
        f"      std::cerr << {unload_error} << e.what() << '\\n';",
        "    }",
        "  }",
        "};",
        f"{abi.module_ctype}* {p}_module() {{",
        *([] if aot else [f"  static {p}_module_ref ref;"]),
        f"  return &{p}_loaded_module;",
        "}",
        "}  // namespace",
        f"#endif  // CUTEDSL_DECLARED_{p}",
    ]


def emit_cpp_launch(
    exported: CuteDSLExportedKernel,
    *,
    tensor_args: Mapping[str, tuple[str, Sequence[str], Sequence[str]]],
    scalar_args: Mapping[str, str],
    stream_expr: str,
    check_align: Mapping[str, int],
) -> list[str]:
    """A C++ block that launches the kernel once; ``emit_cpp_declarations`` must be in
    scope. All arguments are C++ expressions:

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
    with ``<prefix>_module()`` and turns a non-zero launch status into an exception.
    """
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
            f"  AOTI_RUNTIME_CHECK({cond}, {_cpp_str(f'CuTeDSL kernel {p}: {msg}')});"
        )

    call = [f"{p}_module()"]
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
    message = _cpp_str(f"CuTeDSL kernel {p}: launch failed: ")
    lines.append(
        f"  AOTI_RUNTIME_CHECK(cutedsl_status == 0, std::string({message}) + "
        f'cudaGetErrorName({status}) + ": " + cudaGetErrorString({status}));'
    )
    lines.append("}")
    return lines


@functools.cache
def cutedsl_runtime_archive() -> str:
    """The DSL's static CUDA-runtime shim archive for the imported cutlass.

    Exported objects call ``_cudaLaunchKernelEx``-style shims defined in it; the
    shims call cudart. Searched in ``lib/`` and ``nvidia_cutlass_dsl/lib/`` of every
    ancestor of the cutlass package (as found and with symlinks resolved): pip wheels
    ship ``nvidia_cutlass_dsl/lib/libcuda_dialect_runtime_static.a``; fbcode pars put
    the libs-base archive in ``<root>/lib/`` and the libs-cu13 one, renamed
    ``libcuda_dialect_cu13_runtime_static.a``, in ``<root>/nvidia_cutlass_dsl/lib/``.
    The variant matching torch's CUDA major version is preferred.
    """
    import cutlass

    names = [RUNTIME_ARCHIVE]
    if torch.version.cuda:
        names.insert(
            0, f"libcuda_dialect_cu{torch.version.cuda.split('.')[0]}_runtime_static.a"
        )
    roots: list[Path] = []
    for start in (cutlass.__file__, os.path.realpath(cutlass.__file__)):
        roots.extend(a for a in Path(start).parents if a not in roots)
    dirs = [
        d for root in roots for d in (root / "lib", root / "nvidia_cutlass_dsl" / "lib")
    ]
    for name in names:
        for d in dirs:
            if (d / name).is_file():
                return str(d / name)
    raise RuntimeError(
        f"CuTeDSL runtime archive {' or '.join(names)} not found next to "
        f"{cutlass.__file__}; it ships with the nvidia-cutlass-dsl-libs wheel"
    )


def cutedsl_link_inputs(kernels: Iterable[CuteDSLExportedKernel]) -> list[str]:
    """Files to add to the model .so link after the wrapper objects: every exported
    object once, then the runtime archive (it must follow the objects using it)."""
    objects = list(dict.fromkeys(k.object_path for k in kernels))
    return [*objects, cutedsl_runtime_archive()] if objects else []


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
        os.makedirs(out_dir, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=out_dir, suffix=".ver.tmp")
        with os.fdopen(fd, "w") as f:
            f.write(script)
        os.replace(tmp, path)
    archive = os.path.basename(cutedsl_runtime_archive())
    return [
        f"-Wl,--exclude-libs,{archive}",
        f"-Xlinker --version-script -Xlinker {shlex.quote(path)}",
    ]
