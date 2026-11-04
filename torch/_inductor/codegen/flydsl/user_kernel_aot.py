"""AOTInductor export and link support for user-defined FlyDSL kernels.

A ``@flyc.jit`` host function is compiled from metadata-only placeholders that mirror
the layouts Inductor froze for its tensors (the FlyDSL compile helpers of
runtime/flydsl_user_kernel.py) and exported with CompiledFunction.export_to_c, which
writes a C header plus a PIC object. The object embeds the kernel's code object and a
private copy of FlyDSL's AOT runtime, and defines exactly five global symbols: the
packed entry point ``<prefix>``, ``<prefix>__module_init``, ``<prefix>__module_load``,
``<prefix>__module_unload`` and ``<prefix>__metadata``, a JSON description of the
exported C ABI (schema 2) that this module reads and checks against the specs.

The header is spliced into wrapper.cpp; the object, a version script that keeps those
symbols local and the link flags the metadata names go into the link of the model
.so. This module owns those artifacts, the C++ that loads and launches them and the
bodies of CppWrapperGpu's FlyDSL methods (flydsl_param_specs, define_flydsl_launcher,
flydsl_call_args, FlyDSLCallWrapper), which reach the IR and the wrapper only through
their arguments and lazy imports.

Sizes and strides that are symbolic in the graph compile as dynamic FlyDSL layouts
(eager and the Python wrapper compile one static layout per shape instead), so a
launch block derived from one is not a constant there: FlyDSL then knows no block size
for the kernel, whose maximum flat work-group size the ROCDL lowering would leave at
256 threads. Exports with dynamic layouts give such kernels HIP's default, 1024
(rocdl.flat_work_group_size "1,1024"), unless the kernel sets known_block_size or
that attribute (``value_attrs`` of the kernel call) itself.

Collaborators are called through module globals (``mock.patch.object`` on this module
reaches them). The DSL is imported lazily: importing this module never needs flydsl.
Do not add ``from __future__ import annotations``: the DSL reads annotations at
runtime.
"""

import contextlib
import hashlib
import inspect
import json
import os
import re
import shlex
import shutil
import struct
import tempfile
import threading
import types
import weakref
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, TYPE_CHECKING

import torch

# Its _constant_key keys a FlyDSL Numeric constant by value (FLYDSL_CONSTANT_RULES).
from torch._higher_order_ops import flydsl_kernel_wrap
from torch.utils._ordered_set import OrderedSet


if TYPE_CHECKING:
    from torch._higher_order_ops.flydsl_kernel_wrap import FlyDSLKernel
    from torch._inductor.codegen.cpp_wrapper_gpu import CppWrapperGpu


__all__ = [
    "ConstantParamSpec",
    "ExportedParam",
    "FlyDSLAbiSlot",
    "FlyDSLCallWrapper",
    "FlyDSLExportError",
    "FlyDSLExportedABI",
    "FlyDSLExportedKernel",
    "ParamSpec",
    "ScalarParamSpec",
    "StreamParamSpec",
    "TensorParamSpec",
    "cpp_uint64_arg",
    "define_flydsl_launcher",
    "emit_cpp_declarations",
    "emit_cpp_launch",
    "export_flydsl_kernel",
    "export_for_link",
    "flydsl_call_args",
    "flydsl_kernel_prefix",
    "flydsl_link_flags",
    "flydsl_link_inputs",
    "flydsl_param_specs",
    "launch_params",
    "parse_exported_abi",
    "scalar_ctype",
    "target_arch",
]

# Bump when the artifacts this module produces change for the same inputs.
_FORMAT_VERSION = 1
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_ARCH = re.compile(r"gfx[0-9a-f]+\Z")
# Module prefix of FlyDSL's own functions, which the version in kernel prefixes covers.
_DSL_PACKAGE = "flydsl."
# Directory under Inductor's cache dir that exports go to (a fresh one each).
_CACHE_SUBDIR = "flydsl_user_kernels"
# The metadata layout export_to_c writes (flydsl 0.4: schema 2).
_SCHEMA_VERSION = 2
_LIFECYCLE = ("__module_init", "__module_load", "__module_unload")

# FlyDSL Numeric class name -> C type of the exported parameter: the ``ctype`` of
# its metadata slot (flydsl/compiler/_aot.py _leaf_semantic_slots) as a C type.
_SCALAR_CTYPES = {
    "Boolean": "bool",
    "Int8": "int8_t",
    "Int16": "int16_t",
    "Int32": "int32_t",
    "Int64": "int64_t",
    "Uint8": "uint8_t",
    "Uint16": "uint16_t",
    "Uint32": "uint32_t",
    "Uint64": "uint64_t",
    "Float32": "float",
    "Float64": "double",
}
# Exported as the raw uint16 bits of the value, which the launcher would have to
# compute from the double it gets.
_RAW_BITS_SCALARS = ("Float16", "BFloat16")
# Integer C types narrower than the int64 values the launcher computes with.
_INT_CTYPE_RANGES: dict[str, tuple[int, int | None]] = {
    "int8_t": (-(2**7), 2**7 - 1),
    "int16_t": (-(2**15), 2**15 - 1),
    "int32_t": (-(2**31), 2**31 - 1),
    "uint8_t": (0, 2**8 - 1),
    "uint16_t": (0, 2**16 - 1),
    "uint32_t": (0, 2**32 - 1),
    "uint64_t": (0, None),
}


class FlyDSLExportError(RuntimeError):
    """An exported FlyDSL artifact does not match what PT2 asked for."""


###############################################################################
# Parameter specs: one per host-function parameter, in signature order


@dataclass(frozen=True)
class TensorParamSpec:
    """A tensor parameter, with the layout Inductor froze for it. ``None`` marks a
    dynamic size or stride: a slot of the exported ABI that the launch fills in."""

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
    """A runtime scalar of FlyDSL Numeric type ``annotation`` (``fx.Int32``...), which
    fixes its C type. ``hint`` is only the value used to compile; the export takes the
    scalar at run time, so the hint is not part of the kernel identity."""

    name: str
    annotation: type
    hint: int | float


@dataclass(frozen=True)
class ConstantParamSpec:
    """A value FlyDSL bakes into the kernel (Constexpr and other compile-time args)."""

    name: str
    value: object


@dataclass(frozen=True)
class StreamParamSpec:
    """The stream parameter (``fx.Stream``); the launch passes the PT2 stream."""

    name: str


ParamSpec = TensorParamSpec | ScalarParamSpec | ConstantParamSpec | StreamParamSpec


def scalar_ctype(annotation: type) -> str:
    """C type the exported entry point takes for a scalar with this FlyDSL Numeric
    annotation (``fx.Int32`` -> ``int32_t``)."""
    name = getattr(annotation, "__name__", None)
    module = getattr(annotation, "__module__", "") or ""
    if module.startswith("flydsl") and name in _RAW_BITS_SCALARS:
        raise FlyDSLExportError(
            f"scalar annotation {name} is not supported by AOTInductor: FlyDSL "
            "export_to_c passes Float16/BFloat16 scalars as raw bits (use "
            "fx.Float32, or fx.Constexpr for a compile-time value)"
        )
    if not module.startswith("flydsl") or name not in _SCALAR_CTYPES:
        raise FlyDSLExportError(
            f"scalar annotation {annotation!r} is not supported by FlyDSL export_to_c "
            f"(supported: {', '.join(_SCALAR_CTYPES)})"
        )
    return _SCALAR_CTYPES[name]


def _slot_ctype(ctype: str) -> str:
    # Metadata C type name -> the C type (int32 -> int32_t).
    return ctype if ctype in ("bool", "float", "double") else f"{ctype}_t"


###############################################################################
# Exported kernels


@dataclass(frozen=True)
class FlyDSLAbiSlot:
    """One slot of the exported entry point (``<prefix>(void **args)``), from the
    metadata: a tensor's data pointer ("tensor_data") or its packed dynamic layout
    ("tensor_layout": dynamic sizes of dims ``shape_dims`` as ``shape_bits``-bit ints,
    then dynamic strides of dims ``stride_dims`` as ``stride_bits``-bit ints), a
    "scalar" of C type ``ctype``, or the "stream". ``cpp_type``: the type of the
    matching parameter of the header's typed ``<prefix>_call``."""

    kind: str
    name: str
    ctype: str
    cpp_type: str
    shape_dims: tuple[int, ...] = ()
    stride_dims: tuple[int, ...] = ()
    shape_bits: int = 64
    stride_bits: int = 64


@dataclass(frozen=True)
class FlyDSLExportedABI:
    """The C ABI of an export (FlyDSLExportedKernel.abi): its slots in order, the
    target it was compiled for and the link flags its object needs."""

    slots: tuple[FlyDSLAbiSlot, ...]
    arch: str
    flydsl_version: str
    linker_flags: tuple[str, ...]

    def layout(self, param: str) -> FlyDSLAbiSlot | None:
        """The tensor_layout slot of tensor parameter ``param``, if it has one."""
        for slot in self.slots:
            if slot.kind == "tensor_layout" and slot.name == param:
                return slot
        return None


@dataclass(frozen=True)
class FlyDSLExportedKernel:
    """Header + object exported for ``specs`` under symbol prefix ``prefix``;
    ``abi`` is the exported C ABI, read from the object's metadata."""

    prefix: str
    header_path: str
    object_path: str
    header_text: str
    abi: FlyDSLExportedABI
    specs: tuple[ParamSpec, ...] = ()


@dataclass(frozen=True)
class ExportedParam:
    """One parameter of the C++ launcher PT2 generates for an exported kernel, which
    takes the non-constant parameters in host-function order, as the exported ABI
    does (launch_params).

    * kind "tensor": a data pointer, then ``num_dynamic_sizes`` sizes and
      ``num_dynamic_strides`` strides (int64 values the launch narrows to
      ``size_bits`` / ``stride_bits`` signed bits);
    * kind "scalar": a value of C type ``ctype``; ``floating``: the launcher takes it
      as a double (else as an int64); ``boolean``: converted by truthiness;
      ``uint64``: a ``uint64_t`` the launcher takes as a uint64_t, so values above
      INT64_MAX pass (the call site checks the values, cpp_uint64_arg);
    * kind "stream": the hipStream_t.
    """

    kind: str
    name: str
    ctype: str = ""
    num_dynamic_sizes: int = 0
    num_dynamic_strides: int = 0
    size_bits: int = 64
    stride_bits: int = 64
    floating: bool = False
    boolean: bool = False
    uint64: bool = False

    @property
    def launcher_ctype(self) -> str:
        """The C type the generated launcher takes a scalar as."""
        if self.floating:
            return "double"
        return "uint64_t" if self.uint64 else "int64_t"


###############################################################################
# The exported ABI: the object's <prefix>__metadata JSON and the header


def _read_metadata(object_path: str, prefix: str) -> dict[str, Any]:
    raw = elf_symbol_data(object_path, f"{prefix}__metadata")
    text = raw.split(b"\0", 1)[0]
    try:
        metadata = json.loads(text)
    except ValueError as e:
        raise FlyDSLExportError(
            f"{object_path}: {prefix}__metadata is not JSON: {e}"
        ) from None
    if not isinstance(metadata, dict):
        raise FlyDSLExportError(f"{object_path}: {prefix}__metadata is not an object")
    return metadata


def _call_param_types(header_text: str, prefix: str) -> list[str]:
    """Parameter types of the header's ``static inline int32_t <prefix>_call(...)``."""
    match = re.search(
        rf"^static inline int32_t {re.escape(prefix)}_call\(([^)]*)\)",
        header_text,
        re.MULTILINE,
    )
    if match is None:
        raise FlyDSLExportError(f"export_to_c header lacks {prefix}_call")
    types = []
    for param in match.group(1).split(","):
        # "<type> <name>" or "<type> *<name>": drop the name, keep the pointer.
        found = re.fullmatch(r"\s*(.*?[\s*])(\w+)\s*", param)
        if found is None:
            raise FlyDSLExportError(f"{prefix}_call: cannot parse parameter {param!r}")
        types.append(found.group(1).strip())
    return types


def _expected_slots(specs: Sequence[ParamSpec]) -> list[tuple[Any, ...]]:
    # (kind, name, metadata ctype, shape_dims, stride_dims) per slot the specs ask
    # for. Constexpr parameters are compiled in and have none; a tensor has a layout
    # slot only if it has dynamic sizes or strides.
    slots: list[tuple[Any, ...]] = []
    for s in specs:
        if isinstance(s, TensorParamSpec):
            slots.append(("tensor_data", s.name, "pointer", (), ()))
            shape_dims = tuple(i for i, v in enumerate(s.sizes) if v is None)
            stride_dims = tuple(i for i, v in enumerate(s.strides) if v is None)
            if shape_dims or stride_dims:
                slots.append(
                    ("tensor_layout", s.name, "bytes", shape_dims, stride_dims)
                )
        elif isinstance(s, ScalarParamSpec):
            ctype = scalar_ctype(s.annotation)
            slots.append(("scalar", s.name, ctype.removesuffix("_t"), (), ()))
        elif isinstance(s, StreamParamSpec):
            slots.append(("stream", s.name, "pointer", (), ()))
    return slots


def parse_exported_abi(
    metadata: dict[str, Any],
    header_text: str,
    specs: Sequence[ParamSpec],
    *,
    prefix: str,
    arch: str,
) -> FlyDSLExportedABI:
    """The ABI of an export from its ``<prefix>__metadata`` JSON and header text,
    checked against what ``specs`` and ``arch`` ask for: schema, symbol names,
    target, status convention, and one slot per runtime parameter, in order, with
    the dynamic dims of each tensor layout."""
    where = f"FlyDSL export {prefix}"

    def need(cond: bool, msg: str) -> None:
        if not cond:
            raise FlyDSLExportError(f"{where}: {msg}")

    version = metadata.get("schema_version")
    need(version == _SCHEMA_VERSION, f"metadata schema {version}, expected 2")
    need(metadata.get("symbol") == prefix, f"symbol {metadata.get('symbol')!r}")
    need(
        metadata.get("metadata_symbol") == f"{prefix}__metadata"
        and all(
            metadata.get(f"module_{n.removeprefix('__module_')}_symbol")
            == f"{prefix}{n}"
            for n in _LIFECYCLE
        ),
        "unexpected lifecycle symbol names",
    )
    got_arch = metadata.get("arch")
    need(
        got_arch == arch,
        f"compiled for {got_arch!r}, not the target {arch!r} (FlyDSL takes its "
        "target from the ARCH environment variable when a JitFunction first compiles)",
    )
    need(
        metadata.get("return_abi")
        == {"ctype": "int32", "kind": "status", "success": 0},
        f"unexpected return ABI {metadata.get('return_abi')!r}",
    )
    raw_slots = metadata.get("abi")
    if not isinstance(raw_slots, list):
        raise FlyDSLExportError(f"{where}: metadata has no abi slot list")
    expected = _expected_slots(specs)
    got = [
        (
            s.get("kind"),
            s.get("arg_name"),
            s.get("ctype"),
            tuple(s.get("shape_dims", ())),
            tuple(s.get("stride_dims", ())),
        )
        for s in raw_slots
    ]
    if got != expected:
        # A FlyDSL Numeric value for an unannotated parameter is a constant to PT2
        # but a runtime argument to FlyDSL.
        constants = OrderedSet(
            [s.name for s in specs if isinstance(s, ConstantParamSpec)]
        )
        runtime = [slot[1] for slot in got if slot[1] in constants]
        hint = (
            f"; FlyDSL passes {runtime} at run time: annotate them with "
            "fx.Constexpr or a FlyDSL Numeric type"
            if runtime
            else ""
        )
        need(False, f"ABI slots {got} do not match the specs {expected}{hint}")
    cpp_types = _call_param_types(header_text, prefix)
    need(
        len(cpp_types) == len(raw_slots),
        f"{prefix}_call takes {len(cpp_types)} parameters for {len(raw_slots)} slots",
    )
    slots = []
    for raw, cpp_type in zip(raw_slots, cpp_types):
        kind, name, ctype = raw["kind"], raw["arg_name"], raw["ctype"]
        need(
            "encoding" not in raw and "field" not in raw,
            f"slot {name!r} uses an encoding or struct field",
        )
        bits = (raw.get("shape_bits", 64), raw.get("stride_bits", 64))
        if kind == "tensor_layout":
            need(
                re.fullmatch(rf"const {re.escape(prefix)}_\w+_layout_t \*", cpp_type)
                is not None
                and all(b in (32, 64) for b in bits),
                f"unexpected layout slot of {name!r}: {cpp_type!r}, bits {bits}",
            )
        elif kind == "scalar":
            need(
                cpp_type == _slot_ctype(ctype),
                f"{prefix}_call takes {name!r} as {cpp_type!r}, not {ctype}",
            )
        else:
            need(cpp_type == "void *", f"{prefix}_call takes {name!r} as {cpp_type!r}")
        slots.append(
            FlyDSLAbiSlot(
                kind,
                name,
                ctype,
                cpp_type,
                tuple(raw.get("shape_dims", ())),
                tuple(raw.get("stride_dims", ())),
                *bits,
            )
        )
    flags = metadata.get("linker_flags", [])
    need(
        isinstance(flags, list) and all(isinstance(f, str) for f in flags),
        f"unexpected linker_flags {flags!r}",
    )
    return FlyDSLExportedABI(
        tuple(slots), arch, str(metadata.get("flydsl_version", "")), tuple(flags)
    )


###############################################################################
# ELF objects


@dataclass(frozen=True)
class ElfSymbol:
    """A symbol table entry of an ELF64 object."""

    name: str
    binding: int  # STB_LOCAL 0, STB_GLOBAL 1, STB_WEAK 2
    type: int  # STT_NOTYPE 0, STT_OBJECT 1, STT_FUNC 2, ...
    section: int  # 0: undefined
    value: int
    size: int

    @property
    def defined_global(self) -> bool:
        return self.binding in (1, 2) and self.section != 0


def _elf_sections(data: bytes, path: str) -> list[tuple[int, ...]]:
    if data[:4] != b"\x7fELF" or data[4] != 2 or data[5] != 1:
        raise FlyDSLExportError(f"{path} is not an ELF64 little-endian object")
    (shoff,) = struct.unpack_from("<Q", data, 0x28)
    shentsize, shnum = struct.unpack_from("<HH", data, 0x3A)
    return [
        struct.unpack_from("<IIQQQQIIQQ", data, shoff + i * shentsize)
        for i in range(shnum)
    ]


def elf_symbols(path: str) -> list[ElfSymbol]:
    """The symbol table (SHT_SYMTAB) entries of an ELF64 little-endian object."""
    data = Path(path).read_bytes()
    sections = _elf_sections(data, path)
    symbols = []
    for _, sh_type, _, _, offset, size, link, _, _, entsize in sections:
        if sh_type != 2:  # SHT_SYMTAB
            continue
        str_off = sections[link][4]
        for pos in range(offset, offset + size, entsize):
            st_name, st_info, _, st_shndx, st_value, st_size = struct.unpack_from(
                "<IBBHQQ", data, pos
            )
            end = data.index(b"\0", str_off + st_name)
            symbols.append(
                ElfSymbol(
                    data[str_off + st_name : end].decode(errors="surrogateescape"),
                    st_info >> 4,
                    st_info & 0xF,
                    st_shndx,
                    st_value,
                    st_size,
                )
            )
    return symbols


def elf_global_symbols(path: str) -> list[str]:
    """Names of the global/weak symbols an ELF64 little-endian object defines."""
    return [s.name for s in elf_symbols(path) if s.defined_global]


def elf_symbol_data(path: str, name: str) -> bytes:
    """The bytes of data symbol ``name`` defined in relocatable ELF64 object ``path``
    (e.g. a ``const char name[]``), as stored: relocations are not applied."""
    data = Path(path).read_bytes()
    sections = _elf_sections(data, path)
    found = [s for s in elf_symbols(path) if s.name == name and s.section]
    if len(found) != 1 or found[0].section >= len(sections):
        raise FlyDSLExportError(f"{path} does not define one data symbol {name}")
    sym = found[0]
    _, sh_type, _, _, offset, size, _, _, _, _ = sections[sym.section]
    if sh_type != 1 or sym.value + sym.size > size:  # SHT_PROGBITS
        raise FlyDSLExportError(f"{path}: symbol {name} has no data in the object")
    return data[offset + sym.value : offset + sym.value + sym.size]


def _export_symbols(prefix: str) -> list[str]:
    # The five global symbols of an export: the entry point, the lifecycle
    # functions and the metadata.
    return [prefix, f"{prefix}__metadata", *(prefix + n for n in _LIFECYCLE)]


def _local_symbol_patterns(prefix: str) -> list[str]:
    # Version-script patterns matching every global symbol of an export.
    return [prefix, f"{prefix}__*"]


def _check_object_symbols(path: str, prefix: str) -> None:
    """Raise FlyDSLExportError unless exported object ``path`` defines exactly the five
    global symbols of prefix ``prefix`` (export_to_c localizes everything else, the
    embedded runtime included): any other one would leak from the model .so and could
    clash between kernels."""
    names = _export_symbols(prefix)
    symbols = elf_global_symbols(path)
    stray = [s for s in symbols if s not in names]
    if stray:
        raise FlyDSLExportError(
            f"{path} defines global symbols outside {' '.join(names)}: {stray}"
        )
    missing = [s for s in names if s not in symbols]
    if missing:
        raise FlyDSLExportError(f"{path} does not define {missing}")


###############################################################################
# Kernel identity


def target_arch(device: torch.device | None) -> str:
    """The FlyDSL target to export a kernel launched on ``device`` for: its gfx
    architecture (runtime/flydsl_user_kernel.py device_arch)."""
    from torch._inductor.runtime.flydsl_user_kernel import device_arch

    return device_arch(device)


def _check_arch(arch: str) -> None:
    if not _ARCH.match(arch):
        raise ValueError(f"arch must look like 'gfx950', got {arch!r}")


def version_tag() -> str:
    """The FlyDSL version, as hashed into kernel prefixes: "flydsl=0.4.0"."""
    from torch._library.flydsl import require_flydsl

    require_flydsl()
    import flydsl

    return f"flydsl={flydsl.__version__}"


def _is_dsl_function(obj: object) -> bool:
    # @flyc.jit host functions (JitFunction) and @flyc.kernel kernels
    # (KernelFunction); both keep the user's function before FlyDSL's AST rewrite
    # as ``_original_func``.
    cls = type(obj)
    return cls.__name__ in ("JitFunction", "KernelFunction") and (
        cls.__module__ or ""
    ).startswith("flydsl.")


def _source_function(obj: Any) -> Any:
    # The user's function before FlyDSL's AST rewrite: its source, globals and
    # closure describe the code FlyDSL generates (the rewritten JitFunction.func has
    # the annotations, and the same parameters).
    if _is_dsl_function(obj):
        return obj._original_func
    return inspect.unwrap(obj)


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
    """Source of the host function and of every FlyDSL function it references
    through globals, module attributes and closures, transitively, plus the values
    of the simple constants those functions read (both are baked into the generated
    code). FlyDSL's own functions are covered by its version (version_tag)."""
    parts: list[str] = []
    seen: OrderedSet[int] = OrderedSet()
    stack: list[Any] = [host_fn]
    while stack:
        fn = _source_function(stack.pop())
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
                module = _source_function(value).__module__ or ""
                if not module.startswith(_DSL_PACKAGE):
                    stack.append(value)
            elif isinstance(
                value, (bool, int, float, str, bytes, type(None), torch.dtype)
            ):
                parts.append(f"{name}={value!r}")
    return parts


def _stable_repr(value: object) -> str:
    """repr() without object addresses, so the kernel prefix is stable across runs."""
    if value is None or isinstance(value, (bool, int, float, str, bytes, torch.dtype)):
        return repr(value)
    if isinstance(value, (tuple, list)):
        inner = ", ".join(_stable_repr(v) for v in value)
        return f"{type(value).__name__}({inner})"
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
    if is_dataclass(value):
        inner = ", ".join(
            f"{f.name}={_stable_repr(getattr(value, f.name))}" for f in fields(value)
        )
        return f"{type(value).__module__}.{type(value).__qualname__}({inner})"
    return f"{type(value).__module__}.{type(value).__qualname__}:{value!r}"


def _spec_key(spec: ParamSpec) -> str:
    if isinstance(spec, TensorParamSpec):
        return f"T:{spec.name}:{spec.dtype}:{spec.sizes}:{spec.strides}:{spec.align}"
    if isinstance(spec, ScalarParamSpec):  # the hint is not baked in
        return f"S:{spec.name}:{_stable_repr(spec.annotation)}"
    if isinstance(spec, ConstantParamSpec):
        return f"C:{spec.name}:{_stable_repr(spec.value)}"
    return f"R:{spec.name}"


def flydsl_kernel_prefix(
    host_fn: Callable[..., Any], specs: Sequence[ParamSpec], arch: str
) -> str:
    """A readable, C-identifier-safe symbol prefix for one compiled specialization
    of ``host_fn``, ``flydsl_<kernel>_<arch>_<16 hex digits>``: a hash of this
    module's format version, the FlyDSL version, the kernel sources (transitively),
    the plain constants they read, the parameter specs and the target architecture.

    It is not an identity: values it cannot see (a captured list, a tuple global, a
    helper's globals) can differ under the same prefix. export_for_link keys exports
    by host function, specs, prefix and _compile_key_extras (so an edited kernel or a
    rebound module global it reads is exported again) and keeps prefixes unique
    within a library."""
    version = version_tag()
    _check_arch(arch)
    fn = _source_function(host_fn)
    h = hashlib.sha256()
    for part in (
        f"format={_FORMAT_VERSION}",
        version,
        f"arch={arch}",
        f"{fn.__module__}.{fn.__qualname__}",
        *_dsl_function_closure(host_fn),
        *(_spec_key(s) for s in specs),
    ):
        h.update(part.encode())
        h.update(b"\0")
    name = re.sub(r"[^0-9A-Za-z]+", "_", fn.__name__).strip("_")[:32] or "kernel"
    return f"flydsl_{name}_{arch.replace('_', '')}_{h.hexdigest()[:16]}"


###############################################################################
# Export


def _compile_args(specs: Sequence[ParamSpec]) -> dict[str, Any]:
    """Compile-time stand-ins by parameter name: a meta placeholder per tensor with
    the frozen layout (dynamic sizes and strides marked dynamic, with nominal values:
    FlyDSL does not bake them in), the hint of each runtime scalar as a Python value
    of the annotation's kind, the constants themselves and the null stream."""
    import flydsl.expr as fx

    from torch._inductor.runtime.flydsl_user_kernel import meta_placeholder

    args: dict[str, Any] = {}
    for spec in specs:
        if isinstance(spec, TensorParamSpec):
            args[spec.name] = meta_placeholder(
                spec.dtype,
                [2 if s is None else s for s in spec.sizes],
                [1 if s is None else s for s in spec.strides],
                spec.align,
                dynamic_sizes=[i for i, s in enumerate(spec.sizes) if s is None],
                dynamic_strides=[i for i, s in enumerate(spec.strides) if s is None],
            )
        elif isinstance(spec, ScalarParamSpec):
            ctype = scalar_ctype(spec.annotation)
            kind = {"bool": bool, "float": float, "double": float}.get(ctype, int)
            args[spec.name] = kind(spec.hint)
        elif isinstance(spec, ConstantParamSpec):
            args[spec.name] = spec.value
        else:
            args[spec.name] = fx.Stream(None)
    return args


def _check_export_tools() -> None:
    # CompiledFunction.export_to_c runs these from PATH (flydsl_env(for_export=True)
    # puts the ROCm LLVM tools first) and fails with a bare FileNotFoundError.
    if not (shutil.which("ld.lld") or shutil.which("ld")):
        missing = "a relocatable linker (ld.lld or ld)"
    elif not (shutil.which("llvm-objcopy") or shutil.which("objcopy")):
        missing = "llvm-objcopy or objcopy"
    else:
        return
    raise RuntimeError(
        f"FlyDSL export_to_c needs {missing}, which is neither in "
        "$ROCM_PATH/llvm/bin (or $ROCM_HOME/llvm/bin) nor on PATH"
    )


# The maximum flat work-group size (threads per block) an export with dynamic layouts
# gives the kernels that set none (see the module docstring): HIP's default. A kernel
# compiled for up to 1024 threads may get fewer registers per thread than one compiled
# for 256; kernels that need them set known_block_size or a smaller
# rocdl.flat_work_group_size.
_DEFAULT_MAX_FLAT_WORK_GROUP_SIZE = 1024


def _set_default_flat_work_group_size(module: Any, max_size: int) -> list[str]:
    """Give every kernel (gpu.func) of FlyDSL's MLIR ``module`` that has neither a
    known_block_size nor a rocdl.flat_work_group_size the flat work-group size
    "1,<max_size>"; returns the names of the kernels it changed."""
    from flydsl._mlir import ir

    changed = []
    with module.context:
        for top in module.body.operations:
            if top.operation.name != "gpu.module":
                continue
            for op in top.regions[0].blocks[0].operations:
                attrs = op.operation.attributes
                if (
                    op.operation.name == "gpu.func"
                    and ("gpu.kernel" in attrs or "kernel" in attrs)
                    and "known_block_size" not in attrs
                    and "rocdl.flat_work_group_size" not in attrs
                ):
                    attrs["rocdl.flat_work_group_size"] = ir.StringAttr.get(
                        f"1,{max_size}"
                    )
                    changed.append(ir.StringAttr(attrs["sym_name"]).value)
    return changed


@contextlib.contextmanager
def _default_flat_work_group_size(max_size: int) -> Iterator[None]:
    """Inside the block, the kernels this thread compiles get
    _set_default_flat_work_group_size(module, max_size) before FlyDSL's pass
    pipeline runs: a thread-gated hook of RocmBackend.lower_compile_hints (where
    FlyDSL itself turns compile hints into kernel attributes). Other threads compile
    as usual meanwhile. Enter it under FLYDSL_COMPILE_LOCK."""
    import flydsl.compiler.backends.rocm as rocm

    me = threading.get_ident()
    cls = rocm.RocmBackend
    saved = cls.__dict__.get("lower_compile_hints")
    original = cls.lower_compile_hints

    def lower_compile_hints(
        self: Any, module: Any, *, compile_hints: dict[str, Any]
    ) -> None:
        original(self, module, compile_hints=compile_hints)
        if threading.get_ident() == me:
            _set_default_flat_work_group_size(module, max_size)

    cls.lower_compile_hints = lower_compile_hints
    try:
        yield
    finally:
        if saved is None:
            del cls.lower_compile_hints
        else:
            cls.lower_compile_hints = saved


def _check_specs(host_fn: Callable[..., Any], specs: Sequence[ParamSpec]) -> None:
    """Raise ValueError unless ``specs`` name every parameter of ``host_fn`` in order,
    with at most one stream, and FlyDSLExportError for a scalar the export cannot
    take."""
    fn = _source_function(host_fn)
    params = list(inspect.signature(fn).parameters)
    names = [s.name for s in specs]
    if names != params:
        raise ValueError(
            f"specs {names} must name every parameter of {fn.__name__}{params} in order"
        )
    if sum(isinstance(s, StreamParamSpec) for s in specs) > 1:
        raise ValueError(f"{fn.__name__}: more than one stream parameter")
    for s in specs:
        if isinstance(s, ScalarParamSpec):
            scalar_ctype(s.annotation)  # fail before a compile the export would reject


def export_flydsl_kernel(
    host_fn: Callable[..., Any],
    specs: Sequence[ParamSpec],
    *,
    out_dir: str,
    prefix: str,
    arch: str,
) -> FlyDSLExportedKernel:
    """Compile ``host_fn`` for ``specs`` and ``arch`` and export it as
    ``out_dir/<prefix>.h`` + ``out_dir/<prefix>.o``.

    The host function is compiled from the stand-ins of _compile_args through a fresh
    private JitFunction (flydsl_user_kernel.compile_host_fn: nothing is launched) and
    exported with CompiledFunction.export_to_c. Files are written to a private
    temporary directory, validated (the metadata's ABI vs specs and target, object
    symbols vs prefix) and only then moved into ``out_dir``, the header last, so a
    present header means a complete pair.
    """
    from torch._inductor.runtime import flydsl_user_kernel

    specs = tuple(specs)
    _check_specs(host_fn, specs)
    if not _IDENTIFIER.match(prefix):
        raise ValueError(f"prefix {prefix!r} is not a C identifier")
    _check_arch(arch)
    header_path = os.path.join(out_dir, f"{prefix}.h")
    object_path = os.path.join(out_dir, f"{prefix}.o")
    # A launch block derived from a dynamic size is not a constant (see the module
    # docstring).
    dynamic = any(
        None in (*s.sizes, *s.strides) for s in specs if isinstance(s, TensorParamSpec)
    )
    os.makedirs(out_dir, exist_ok=True)
    tmp_dir = tempfile.mkdtemp(prefix=f".{prefix}.", dir=out_dir)
    try:
        args = _compile_args(specs)
        with flydsl_user_kernel.FLYDSL_COMPILE_LOCK:
            with (
                _default_flat_work_group_size(_DEFAULT_MAX_FLAT_WORK_GROUP_SIZE)
                if dynamic
                else contextlib.nullcontext()
            ):
                compiled = flydsl_user_kernel.compile_host_fn(host_fn, args, arch)
            with flydsl_user_kernel.flydsl_env(arch, for_export=True):
                _check_export_tools()
                # function_prefix replaces the symbol name, so the five symbols
                # are <prefix>, <prefix>__module_init, ...
                compiled.export_to_c(tmp_dir, prefix, function_prefix=prefix)
        tmp_header = os.path.join(tmp_dir, f"{prefix}.h")
        tmp_object = os.path.join(tmp_dir, f"{prefix}.o")
        text = Path(tmp_header).read_text()
        abi = parse_exported_abi(
            _read_metadata(tmp_object, prefix), text, specs, prefix=prefix, arch=arch
        )
        _check_object_symbols(tmp_object, prefix)
        os.replace(tmp_object, object_path)
        os.replace(tmp_header, header_path)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return FlyDSLExportedKernel(prefix, header_path, object_path, text, abi, specs)


# host function -> {export key + kernel prefix + compile key extras -> its exports,
# one per prefix it is linked under}
_exports: weakref.WeakKeyDictionary[
    Callable[..., Any], dict[tuple[Any, ...], list[FlyDSLExportedKernel]]
] = weakref.WeakKeyDictionary()
_exports_lock = threading.Lock()


def _export_key(specs: Sequence[ParamSpec], arch: str) -> tuple[Any, ...]:
    # Constants by FlyDSL's own key (Numerics by value, Constexpr lambdas by code),
    # as the side table and the eager launch key take them.
    return (
        arch,
        *(
            (s.name, flydsl_kernel_wrap._constant_key(s.value))
            if isinstance(s, ConstantParamSpec)
            else _spec_key(s)
            for s in specs
        ),
    )


def _compile_key_extras(host_fn: Any) -> tuple[tuple[str, Any], ...]:
    """What a compile of user JitFunction ``host_fn`` depends on besides its
    arguments, target and sources: the entries launch_flydsl_kernel adds to an eager
    launch key (flydsl_user_kernel._launch_key_extras), i.e. the values of the module
    globals it reads, also through the helpers, kernels and host functions it calls
    (tuples by value, objects by identity), the environment variables that change a
    FlyDSL compile and the compile hints (the host function's merged with the calling
    thread's). The kernel prefix sees only the plain constants its FlyDSL functions
    read."""
    from torch._inductor.runtime import flydsl_user_kernel

    return tuple(flydsl_user_kernel._launch_key_extras(host_fn))


def export_for_link(
    host_fn: Callable[..., Any],
    specs: Sequence[ParamSpec],
    arch: str,
    *,
    linked: Mapping[str, FlyDSLExportedKernel],
    out_root: str,
) -> FlyDSLExportedKernel:
    """The export of ``host_fn`` for ``specs`` and ``arch``, to link into a library
    that already links ``linked`` (prefix -> export).

    Exports are reused within this process only (_exports), keyed by the host
    function object, the specs, the kernel prefix (which hashes the kernel sources
    and the plain constants they read) and what else the eager launch key holds
    (_compile_key_extras: the values of every module global the kernel reads, also
    through Python helpers, the FlyDSL environment and the compile hints), so a
    compile after any of them changed exports again. Nothing on disk is reused, since
    the prefix does not cover every value a kernel reads (captured objects, a
    helper's globals). Each export goes to a fresh directory under ``out_root``,
    under a prefix no other kernel of the library uses.
    """
    specs = tuple(specs)
    prefix = base = flydsl_kernel_prefix(host_fn, specs, arch)
    key = (*_export_key(specs, arch), base, *_compile_key_extras(host_fn))
    with _exports_lock:
        cached = _exports.setdefault(host_fn, {}).setdefault(key, [])
        # An export whose cache directory was removed since cannot be linked.
        cached[:] = [e for e in cached if os.path.exists(e.object_path)]
        for exported in cached:
            if linked.get(exported.prefix, exported) is exported:
                return exported
    n = 0
    while prefix in linked:
        n += 1
        prefix = f"{base}_{n}"
    os.makedirs(out_root, exist_ok=True)
    exported = export_flydsl_kernel(
        host_fn,
        specs,
        out_dir=tempfile.mkdtemp(prefix=f"{prefix}.", dir=out_root),
        prefix=prefix,
        arch=arch,
    )
    with _exports_lock:
        cached.append(exported)
    return exported


###############################################################################
# C++


def _cpp_str(text: str) -> str:
    """``text`` as a C++ string literal."""
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


# What the generated declarations and launchers use, once per translation unit
# (repeating it is harmless).
_CPP_PRELUDE = r"""#include <cstdint>
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
#endif"""

# Status check of the export_to_c entry points (part of the prelude of every
# declaration block): 0 is success, -1 FLYDSL_AOT_ERR_NOT_INITIALIZED, -2
# FLYDSL_AOT_ERR_NOT_LOADED (not loaded on the current device), anything else a
# hipError_t. FlyDSL's runtime leaves the HIP error of a failed module load or launch
# pending, so it is cleared before throwing: the next HIP call of the process would
# fail with it otherwise. ``launch_hint`` is appended to the HIP errors a launch block
# the kernel was not compiled for gives (_cpp_call).
_STATUS_HELPERS = r"""#ifndef FLYDSL_USER_KERNEL_CHECK_DEFINED
#define FLYDSL_USER_KERNEL_CHECK_DEFINED
#include <hip/hip_runtime_api.h>
static inline void flydsl_user_kernel_check(
    int32_t status, const char* what, const char* launch_hint = "") {
  if (status == 0) {
    return;
  }
  std::string reason;
  if (status == -1) {
    reason = "-1 (FLYDSL_AOT_ERR_NOT_INITIALIZED: module not initialized)";
  } else if (status == -2) {
    reason = "-2 (FLYDSL_AOT_ERR_NOT_LOADED: module not loaded on the current device)";
  } else {
    (void)hipGetLastError();
    const hipError_t err = static_cast<hipError_t>(status);
    reason = std::to_string(status) + " (" + hipGetErrorName(err) + ": " +
        hipGetErrorString(err) + ")";
    if (err == hipErrorLaunchFailure || err == hipErrorInvalidValue ||
        err == hipErrorInvalidConfiguration) {
      reason += launch_hint;
    }
  }
  AOTI_RUNTIME_CHECK(false, std::string(what) + ": status " + reason);
}
#endif"""


def emit_cpp_declarations(exported: FlyDSLExportedKernel) -> list[str]:
    """File-scope C++ for one exported kernel, to emit once per prefix before any
    launch (e.g. into the wrapper header): the prelude and the status check, then,
    under a per-prefix guard, the export_to_c header and ``<prefix>_init_module_once``.

    The object keeps its module state itself: __module_init (no HIP call) runs once
    per process, then each launch runs __module_load for the current device
    (emit_cpp_launch)."""
    p = exported.prefix
    guard = f"FLYDSL_DECLARED_{p}"
    header = "\n".join(
        line
        for line in exported.header_text.splitlines()
        if line.strip() != "#pragma once"
    )
    header_name = os.path.basename(exported.header_path)
    message = _cpp_str(f"FlyDSL kernel {p}: {p}__module_init failed")
    return [
        *_CPP_PRELUDE.splitlines(),
        *_STATUS_HELPERS.splitlines(),
        f"// FlyDSL user kernel {p}: export_to_c header {header_name}",
        f"#ifndef {guard}",
        f"#define {guard}",
        *header.splitlines(),
        f"static std::once_flag {p}_init_once;",
        f"static inline void {p}_init_module_once() {{",
        f"  std::call_once({p}_init_once, [] {{",
        f"    const int32_t status = {p}__module_init();",
        f"    flydsl_user_kernel_check(status, {message});",
        "  });",
        "}",
        f"#endif  // {guard}",
    ]


def launch_params(exported: FlyDSLExportedKernel) -> tuple[ExportedParam, ...]:
    """The parameters of the generated launcher, in exported ABI order: a tensor per
    tensor_data slot (with the dynamic dims and bit widths of its layout slot, if
    any), the scalars and the stream."""
    abi = exported.abi
    params = []
    for slot in abi.slots:
        if slot.kind == "tensor_data":
            layout = abi.layout(slot.name)
            params.append(
                ExportedParam(
                    "tensor",
                    slot.name,
                    "void*",
                    len(layout.shape_dims) if layout else 0,
                    len(layout.stride_dims) if layout else 0,
                    size_bits=layout.shape_bits if layout else 64,
                    stride_bits=layout.stride_bits if layout else 64,
                )
            )
        elif slot.kind == "scalar":
            ctype = _slot_ctype(slot.ctype)
            params.append(
                ExportedParam(
                    "scalar",
                    slot.name,
                    ctype,
                    floating=ctype in ("float", "double"),
                    boolean=ctype == "bool",
                    # Passed as it is: an int64 cannot hold its values >= 2**63.
                    uint64=ctype == "uint64_t",
                )
            )
        elif slot.kind == "stream":
            params.append(ExportedParam("stream", slot.name, "hipStream_t"))
    return tuple(params)


def _signed_range(bits: int) -> tuple[int, int]:
    return -(2 ** (bits - 1)), 2 ** (bits - 1) - 1


def _cpp_check(prefix: str, cond: str, msg: str) -> str:
    text = _cpp_str(f"FlyDSL kernel {prefix}: {msg}")
    return f"  AOTI_RUNTIME_CHECK({cond}, {text});"


def _cpp_scalar(
    exported: FlyDSLExportedKernel, param: ExportedParam, v: str, expr: str
) -> list[str]:
    """Statements defining ``v``: scalar ``param`` converted from C++ expression
    ``expr`` (of the param's launcher_ctype: a double for floating scalars, a uint64
    for uint64 ones, else an int64) to its C type."""
    name, ctype = param.name, param.ctype
    if ctype in ("float", "double"):
        return [f"  const {ctype} {v} = static_cast<{ctype}>({expr});  // {name}"]
    if param.boolean:
        # Truthiness, as FlyDSL converts a Python value for a Boolean (the kernel
        # reads only the low bit).
        return [
            f"  const {ctype} {v} = static_cast<{ctype}>(({expr}) != 0);  // {name}"
        ]
    if param.uint64:
        if ctype != "uint64_t":
            raise FlyDSLExportError(
                f"{exported.prefix}: {name} is passed as a uint64_t, not as {ctype}"
            )
        # Every value of the launcher's uint64_t fits; the call site checked it.
        return [f"  const uint64_t {v} = static_cast<uint64_t>({expr});  // {name}"]
    lines = [f"  const int64_t {v}_value = static_cast<int64_t>({expr});  // {name}"]
    if ctype in _INT_CTYPE_RANGES:
        lo, hi = _INT_CTYPE_RANGES[ctype]
        cond = f"{v}_value >= {lo}LL" + (
            "" if hi is None else f" && {v}_value <= {hi}LL"
        )
        lines.append(
            _cpp_check(exported.prefix, cond, f"scalar {name} does not fit in {ctype}")
        )
    elif ctype != "int64_t":
        raise FlyDSLExportError(
            f"{exported.prefix}: unsupported scalar C type {ctype} for {name}"
        )
    lines.append(f"  const {ctype} {v} = static_cast<{ctype}>({v}_value);")
    return lines


def cpp_uint64_arg(
    exported: FlyDSLExportedKernel, name: str, value: object, expr: str
) -> tuple[list[str], str]:
    """How a launch passes scalar ``name`` to a launcher parameter that takes it as a
    uint64_t (ExportedParam.uint64): statements to run before the call, and the call
    argument. A Python int ``value`` is checked now (ValueError unless it is in
    [0, 2**64)) and passed as a literal; any other value is the int64 C++ expression
    ``expr``, checked at run time not to be negative. One launcher serves every call
    site of an export, so the values are checked here rather than in the launcher."""
    if isinstance(value, int):
        if not 0 <= value < 2**64:
            raise ValueError(
                f"FlyDSL kernel {exported.prefix}: scalar {name}={value} does not fit "
                "in uint64_t"
            )
        return [], f"{int(value)}ULL"
    message = f"scalar {name} does not fit in uint64_t"
    check = _cpp_check(exported.prefix, f"({expr}) >= 0", message)
    return [check.strip()], expr


def _cpp_int_arg(
    exported: FlyDSLExportedKernel, param: ExportedParam, value: int
) -> str:
    """The call argument passing Python int ``value`` to the launcher parameter of
    scalar ``param``. If the export takes the scalar as an integer C type, ``value``
    is checked now, as launch_flydsl_kernel checks it in eager: ValueError unless the
    type holds it, since FlyDSL would wrap it (the launcher checks only values
    computed at run time, and none for an int64_t) and a literal beyond 64 bits would
    not compile."""
    match = re.fullmatch(r"(u?)int(8|16|32|64)_t", param.ctype)
    if match is not None:
        bits = int(match.group(2))
        lo, hi = (0, 2**bits - 1) if match.group(1) else _signed_range(bits)
        if not lo <= value <= hi:
            raise ValueError(
                f"FlyDSL kernel {exported.prefix}: scalar {param.name}={value} does "
                f"not fit in {param.ctype} [{lo}, {hi}], and FlyDSL would wrap it"
            )
    if value == -(2**63):
        # "-9223372036854775808" negates a literal that no int64 holds.
        return "(-9223372036854775807LL - 1)"
    return str(value)


def _cpp_tensor_arg(
    layout: FlyDSLAbiSlot | None, var: str, sizes: Sequence[str], strides: Sequence[str]
) -> tuple[list[str], list[str]]:
    # Statements and call arguments passing a tensor whose data pointer is
    # ``void* <var>_data`` and whose layout slot is ``layout`` (None: a static
    # layout, no slot). ``sizes`` / ``strides``: int64 expressions of its dynamic sizes
    # and strides, already checked to fit the slot's bit widths, which go into the
    # header's packed struct {int<shape_bits>_t shape[k]; int<stride_bits>_t
    # stride[m];} (fields of dims without dynamic values absent), passed by pointer.
    if layout is None:
        return [], [f"{var}_data"]
    init = []
    if sizes:
        ctype = f"int{layout.shape_bits}_t"
        init.append("{" + ", ".join(f"static_cast<{ctype}>({s})" for s in sizes) + "}")
    if strides:
        ctype = f"int{layout.stride_bits}_t"
        init.append(
            "{" + ", ".join(f"static_cast<{ctype}>({s})" for s in strides) + "}"
        )
    struct_ctype = layout.cpp_type.removeprefix("const ").removesuffix(" *")
    return [f"  const {struct_ctype} {var}_layout{{{', '.join(init)}}};"], [
        f"{var}_data",
        f"&{var}_layout",
    ]


def _cpp_call(exported: FlyDSLExportedKernel, call_args: Sequence[str]) -> list[str]:
    # The call of the header's typed entry point, a failed status turned into an
    # exception (flydsl_user_kernel_check).
    p = exported.prefix
    args = [
        f"{p}_call({', '.join(call_args)})",
        _cpp_str(f"FlyDSL kernel {p}: launch failed"),
    ]
    dynamic = [s.name for s in exported.abi.slots if s.kind == "tensor_layout"]
    if dynamic:
        # Compiled without a constant block size (see the module docstring).
        names = ", ".join(dynamic)
        args.append(
            _cpp_str(
                f"; this export has dynamic sizes or strides for {names}: a "
                "launch block derived from them must fit the kernel's maximum "
                "flat work-group size (1024 threads unless the kernel sets "
                "known_block_size or value_attrs={'rocdl.flat_work_group_size': "
                "'<min>,<max>'})"
            )
        )
    return [f"  flydsl_user_kernel_check({', '.join(args)});"]


def emit_cpp_launch(
    exported: FlyDSLExportedKernel,
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
    * ``scalar_args[name]``: the value of each runtime scalar (of its
      ExportedParam.launcher_ctype);
    * ``stream_expr``: the hipStream_t to launch on;
    * ``check_align[name]``: tensors whose data pointer must be checked against the
      alignment the kernel was compiled with (skip those PT2 already guarantees).

    The block initializes the module (once per process) and loads it on the current
    device (``__module_load(-1)``: AOTInductor's device guard selects the kernel's
    device; a cheap lookup after the first time), checks alignment and that every
    value fits the narrower C type the ABI uses (dynamic sizes and strides of fewer
    than 64 bits, Int32 scalars...), converts the scalars, packs the dynamic layouts
    and calls the typed ``<prefix>_call``, turning a failed status into an exception.
    """
    p = exported.prefix
    params = launch_params(exported)
    tensors = [q.name for q in params if q.kind == "tensor"]
    scalars = [q.name for q in params if q.kind == "scalar"]
    if OrderedSet(tensor_args) != OrderedSet(tensors):
        raise ValueError(
            f"{p}: tensor_args {sorted(tensor_args)} != tensors {sorted(tensors)}"
        )
    if OrderedSet(scalar_args) != OrderedSet(scalars):
        raise ValueError(
            f"{p}: scalar_args {sorted(scalar_args)} != scalars {sorted(scalars)}"
        )
    if not OrderedSet(check_align) <= OrderedSet(tensors):
        raise ValueError(
            f"{p}: check_align names {sorted(set(check_align) - set(tensors))} are not tensors"
        )

    load_message = _cpp_str(
        f"FlyDSL kernel {p}: {p}__module_load on the current device failed"
    )
    lines = [
        "{",
        f"  // FlyDSL user kernel {p}",
        f"  {p}_init_module_once();",
        f"  flydsl_user_kernel_check({p}__module_load(-1), {load_message});",
    ]

    def check(cond: str, msg: str) -> None:
        lines.append(_cpp_check(p, cond, msg))

    call: list[str] = []
    for i, param in enumerate(params):
        v = f"flydsl_arg{i}"
        name = param.name
        if param.kind == "tensor":
            data_ptr, sizes, strides = tensor_args[name]
            if (
                len(sizes) != param.num_dynamic_sizes
                or len(strides) != param.num_dynamic_strides
            ):
                raise ValueError(
                    f"{p}: {name} needs {param.num_dynamic_sizes} dynamic sizes and "
                    f"{param.num_dynamic_strides} dynamic strides, got {len(sizes)} and {len(strides)}"
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
            size_values = []
            for j, size in enumerate(sizes):
                lines.append(
                    f"  const int64_t {v}_size{j} = static_cast<int64_t>({size});"
                )
                if param.size_bits < 64:
                    _, hi = _signed_range(param.size_bits)
                    check(
                        f"{v}_size{j} >= 0 && {v}_size{j} <= {hi}LL",
                        f"dynamic size {j} of argument {name} does not fit in "
                        f"int{param.size_bits}",
                    )
                size_values.append(f"{v}_size{j}")
            # 64-bit strides go into the layout as they are.
            stride_values = list(strides)
            if param.stride_bits < 64:
                stride_values = []
                lo, hi = _signed_range(param.stride_bits)
                for j, stride in enumerate(strides):
                    lines.append(
                        f"  const int64_t {v}_stride{j} = static_cast<int64_t>({stride});"
                    )
                    check(
                        f"{v}_stride{j} >= {lo}LL && {v}_stride{j} <= {hi}LL",
                        f"dynamic stride {j} of argument {name} does not fit in "
                        f"int{param.stride_bits}",
                    )
                    stride_values.append(f"{v}_stride{j}")
            more, args = _cpp_tensor_arg(
                exported.abi.layout(name), v, size_values, stride_values
            )
            lines.extend(more)
            call.extend(args)
        elif param.kind == "stream":
            call.append(f"static_cast<void*>({stream_expr})")
        else:
            call.append(v)
            lines.extend(_cpp_scalar(exported, param, v, scalar_args[name]))
    lines.extend(_cpp_call(exported, call))
    lines.append("}")
    return lines


###############################################################################
# Link


def flydsl_link_inputs(kernels: Iterable[FlyDSLExportedKernel]) -> list[str]:
    """Files to add to the model .so link after the wrapper objects: every exported
    object once. Each embeds its own copy of FlyDSL's AOT runtime: there is no
    archive to link."""
    return list(dict.fromkeys(k.object_path for k in kernels))


def flydsl_link_flags(
    out_dir: str, kernels: Sequence[FlyDSLExportedKernel]
) -> list[str]:
    """Linker flags for a library calling ``kernels``: a version script (written to
    ``out_dir``) making the five symbols of each export local to the library (every
    other symbol keeps its visibility), then the link flags the exports' metadata ask
    for (-lamdhip64 -pthread -ldl).

    Each element is a shell fragment: CppBuilder joins passthrough args into one
    command line and shlex-splits it. The version script path is a token of its own
    (``-Xlinker <path>``) so fbcode remote builds stage it like other input files.
    """
    names = sorted(OrderedSet(k.prefix for k in kernels))
    for p in names:
        if not _IDENTIFIER.match(p):
            raise ValueError(f"prefix {p!r} is not a C identifier")
    if not names:
        return []
    script = (
        "/* FlyDSL user kernels: export_to_c symbols stay in this library. */\n"
        "{\n  local:\n"
        + "".join(f"    {pat};\n" for p in names for pat in _local_symbol_patterns(p))
        + "};\n"
    )
    path = os.path.join(
        out_dir, f"flydsl_{hashlib.sha256(script.encode()).hexdigest()[:16]}.ver"
    )
    if not os.path.exists(path) or Path(path).read_text() != script:
        os.makedirs(out_dir, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=out_dir, suffix=".ver.tmp")
        with os.fdopen(fd, "w") as f:
            f.write(script)
        os.replace(tmp, path)
    flags = dict.fromkeys(f for k in kernels for f in k.abi.linker_flags)
    return [f"-Xlinker --version-script -Xlinker {shlex.quote(path)}", *flags]


###############################################################################
# CppWrapperGpu's FlyDSL methods (cpp_wrapper_gpu.py calls these)


def flydsl_param_specs(
    kernel: "FlyDSLKernel", runtime_args: dict[str, Any], constant_args: dict[str, Any]
) -> tuple[list[ParamSpec], dict[str, int]]:
    """The export specs of a call to a user FlyDSL kernel: one per host-function
    parameter, from the IR layouts of the tensor arguments (a symbolic size or stride
    becomes a dynamic slot of the exported ABI) and the Numeric annotations of the
    runtime scalars. Also returns the tensors whose 16-byte alignment the kernel is
    compiled to assume, to check at launch."""
    import sympy

    from torch._inductor.codegen.triton_utils import is_unaligned_buffer_name
    from torch._inductor.ir import IRNode
    from torch._inductor.virtualized import V

    info = flydsl_kernel_wrap.get_host_fn_info(kernel.fn)
    sizevars = V.graph.sizevars

    def static(expr: Any) -> int | None:
        expr = sizevars.simplify(expr)
        return int(expr) if isinstance(expr, (int, sympy.Integer)) else None

    specs: list[ParamSpec] = []
    check_align: dict[str, int] = {}
    for name in info.params:
        if name == info.stream_param:
            specs.append(StreamParamSpec(name))
        elif name in runtime_args and isinstance(runtime_args[name], IRNode):
            value = runtime_args[name]
            layout = value.get_layout()
            itemsize = value.get_dtype().itemsize
            offset = static(layout.offset)
            align = itemsize
            if (
                offset is not None
                and (offset * itemsize) % 16 == 0
                and not is_unaligned_buffer_name(value.get_name())
            ):
                align = max(16, itemsize)
                check_align[name] = align
            specs.append(
                TensorParamSpec(
                    name,
                    value.get_dtype(),
                    tuple(static(e) for e in layout.size),
                    tuple(static(e) for e in layout.stride),
                    align,
                )
            )
        elif name in runtime_args:
            if name not in info.numeric_params:
                # export_to_c takes the C type of a scalar from its compile-time
                # type; FlyDSL types an unannotated one from each launch's Python
                # value, which a C launcher cannot follow.
                raise NotImplementedError(
                    f"FlyDSL kernel {kernel.name}: AOTInductor needs scalar parameter "
                    f"{name!r} annotated with a FlyDSL Numeric type (e.g. fx.Int32) or "
                    "fx.Constexpr"
                )
            value = runtime_args[name]
            hint = (
                value
                if isinstance(value, (int, float))
                else sizevars.optimization_hint(value)
            )
            specs.append(ScalarParamSpec(name, info.numeric_params[name], hint))
        else:
            specs.append(ConstantParamSpec(name, constant_args[name]))
    return specs, check_align


def define_flydsl_launcher(
    wrapper: "CppWrapperGpu",
    kernel: "FlyDSLKernel",
    constant_args: dict[str, Any],
    runtime_args: dict[str, Any],
    device: torch.device | None,
) -> str:
    """Export ``kernel`` for the layouts Inductor froze (flydsl_param_specs) and the
    target of ``device``, the device it runs on, and define its C++ launcher in
    ``wrapper``; returns the launcher's name, ``call_<prefix>``. Once per prefix, the
    export's declarations go into the wrapper's header and a FlyDSLCallWrapper, which
    finalize_prefix generates, into its ``_flydsl_call_wrappers``; the export is
    linked by AotCodeCompiler (codecache.flydsl_user_kernel_link_args). The body of
    CppWrapperGpu.define_user_defined_flydsl_kernel."""
    from torch._inductor.runtime.cache_dir_utils import cache_dir

    specs, check_align = flydsl_param_specs(kernel, runtime_args, constant_args)
    arch = target_arch(device)
    exported = export_for_link(
        kernel.fn,
        specs,
        arch,
        linked=wrapper.flydsl_exported_kernels,
        out_root=os.path.join(cache_dir(), _CACHE_SUBDIR),
    )
    prefix = exported.prefix
    wrapper_name = f"call_{prefix}"
    if prefix not in wrapper.flydsl_exported_kernels:
        wrapper.flydsl_exported_kernels[prefix] = exported
        wrapper.header.splice("\n".join(emit_cpp_declarations(exported)))
        wrapper._flydsl_call_wrappers[wrapper_name] = FlyDSLCallWrapper(
            wrapper_name, exported, check_align
        )
    return wrapper_name


def flydsl_call_args(
    wrapper: "CppWrapperGpu", kernel_name: str, runtime_args: dict[str, Any]
) -> list[str]:
    """The arguments of a call to launcher ``kernel_name`` (define_flydsl_launcher)
    in its parameter order, without the stream: per tensor its data pointer, then its
    sizes and strides that are not static; per scalar its value. Scalars the launcher
    takes as a uint64_t are checked here (cpp_uint64_arg): a Python int now, a
    symbolic value by a check written to ``wrapper`` before the call; a Python int
    for another integer scalar is checked now against its C type (_cpp_int_arg). The
    body of CppWrapperGpu.generate_user_defined_flydsl_kernel_call, which appends the
    stream."""
    import sympy

    from torch._inductor.ir import IRNode
    from torch._inductor.virtualized import V

    exported = wrapper._flydsl_call_wrappers[kernel_name].exported
    params = {p.name: p for p in launch_params(exported)}
    uint64_params = OrderedSet(p.name for p in params.values() if p.uint64)
    call_args = []
    for name, value in runtime_args.items():
        if name in uint64_params and not isinstance(value, (IRNode, float)):
            expr = (
                str(int(value))
                if isinstance(value, int)
                else wrapper.codegen_cpp_sizevar(value)
            )
            checks, arg = cpp_uint64_arg(exported, name, value, expr)
            for line in checks:
                wrapper.writeline(line)
            call_args.append(arg)
        elif isinstance(value, IRNode):
            tensor = wrapper.create_tmp_raii_handle_var_if_needed(
                value.codegen_reference()
            )
            call_args.append(f"{tensor}.data_ptr()")
            layout = value.get_layout()
            for expr in (*layout.size, *layout.stride):
                if not isinstance(
                    V.graph.sizevars.simplify(expr), (int, sympy.Integer)
                ):
                    call_args.append(wrapper.codegen_cpp_sizevar(expr))
        elif isinstance(value, bool):
            call_args.append(str(value).lower())
        elif isinstance(value, float):
            call_args.append(wrapper.generate_float_value(value))
        elif isinstance(value, int):
            call_args.append(_cpp_int_arg(exported, params[name], value))
        else:
            call_args.append(wrapper.codegen_cpp_sizevar(value))
    return call_args


@dataclass
class FlyDSLCallWrapper:
    """The C++ launcher of one exported user FlyDSL kernel: the counterpart of
    DeferredTritonCallWrapper. Nothing needs deferring (the kernel was exported at
    codegen time); CppWrapperGpu generates it with the Triton launchers in
    finalize_prefix.

    Parameters: per tensor a data pointer then its dynamic sizes and strides (int64,
    in dimension order), per scalar its value (its ExportedParam.launcher_ctype: a
    double, an int64 or a uint64), and the hipStream_t. They are named ``a<i>`` after
    the position in launch_params (a FlyDSL parameter may be named like a C++ keyword
    such as ``new`` or ``default``, a macro such as ``errno``, or like the stream);
    messages keep the host function's parameter names.
    """

    wrapper_name: str
    exported: FlyDSLExportedKernel
    check_align: dict[str, int]

    def generate(self, wrapper: "CppWrapperGpu") -> None:
        params: list[str] = []
        tensor_args: dict[str, tuple[str, list[str], list[str]]] = {}
        scalar_args: dict[str, str] = {}
        for i, param in enumerate(launch_params(self.exported)):
            name, var = param.name, f"a{i}"
            if param.kind == "tensor":
                sizes = [f"{var}_size{j}" for j in range(param.num_dynamic_sizes)]
                strides = [f"{var}_stride{j}" for j in range(param.num_dynamic_strides)]
                params.append(f"void* {var}")
                params.extend(f"int64_t {v}" for v in (*sizes, *strides))
                tensor_args[name] = (var, sizes, strides)
            elif param.kind == "scalar":
                params.append(f"{param.launcher_ctype} {var}")
                scalar_args[name] = var
        # The stream comes last whatever its position in the host function.
        params.append("hipStream_t stream")
        prefix = wrapper.prefix
        prefix.writeline(f"static inline void {self.wrapper_name}({', '.join(params)})")
        for line in emit_cpp_launch(
            self.exported,
            tensor_args=tensor_args,
            scalar_args=scalar_args,
            stream_expr="stream",
            check_align=self.check_align,
        ):
            prefix.writeline(line)
