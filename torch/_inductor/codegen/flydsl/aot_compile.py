# mypy: allow-untyped-defs
from __future__ import annotations

import ctypes
import filecmp
import inspect
import os
import re
import shutil
import subprocess
import tempfile
from contextlib import nullcontext
from pathlib import Path
from typing import Any, TYPE_CHECKING

from torch._inductor.codegen.flydsl.flydsl_utils import runtime_available
from torch.utils._ordered_set import OrderedSet


HAS_FLYDSL = runtime_available()
if TYPE_CHECKING or HAS_FLYDSL:
    import flydsl.utils as flydsl_utils
    from flydsl._mlir import execution_engine, ir
    from flydsl._mlir.dialects import func
    from flydsl.compiler import (
        backends,
        jit_argument,
        jit_executor,
        jit_function,
        kernel_function,
        protocol,
    )
    from flydsl.expr import meta, numeric, typing as flydsl_typing
    from flydsl.expr.utils import arith


_C_SYMBOL = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_GPU_MODULE_INIT = "flydsl_gpu_module_init"
_GPU_MODULE_LOAD_TO_DEVICE = "flydsl_gpu_module_load_to_device"
_ELF_SONAME = re.compile(r"\(SONAME\).*\[([^]]+)\]")
_LDD_LIBRARY = re.compile(r"^\s*(\S+)\s+=>\s+(\S+)\s+\(")


def _ctype_metadata(ctype: type, *, name: str | None = None) -> dict[str, Any]:
    if name is None:
        if ctype is ctypes.c_void_p:
            name = "pointer"
        elif issubclass(ctype, ctypes.Array):
            name = "bytes"
        elif ctype is ctypes.c_float:
            name = "float"
        elif ctype is ctypes.c_double:
            name = "double"
        elif ctype is ctypes.c_bool:
            name = "bool"
        else:
            name = ctype.__name__.removeprefix("c_")
    return {
        "ctype": name,
        "size": ctypes.sizeof(ctype),
        "alignment": ctypes.alignment(ctype),
    }


def _numeric_ctype_name(value: Any, ctype: type) -> str:
    if value.signed is not None:
        if value.width == 1:
            return "bool"
        return f"{'int' if value.signed else 'uint'}{value.width}"
    if isinstance(value, (numeric.Float16, numeric.BFloat16)):
        return "uint16"
    if isinstance(value, numeric.Float32):
        return "float"
    if isinstance(value, numeric.Float64):
        return "double"
    return ctype.__name__.removeprefix("c_")


def _argument_abi(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, jit_argument.MemRefJitArg):
        slots = [
            {
                "kind": "tensor_data",
                **_ctype_metadata(ctypes.c_void_p, name="pointer"),
            }
        ]
        if value.is_layout_dynamic:
            shape_dims = list(value.shape_dyn_indices)
            stride_dims = list(value.stride_dyn_indices)
            stride_ctype = ctypes.c_int32 if value.use_32bit_stride else ctypes.c_int64
            layout_ctype = ctypes.c_byte * (
                len(shape_dims) * ctypes.sizeof(ctypes.c_int32)
                + len(stride_dims) * ctypes.sizeof(stride_ctype)
            )
            slots.append(
                {
                    "kind": "tensor_layout",
                    **_ctype_metadata(layout_ctype, name="bytes"),
                    "shape_dims": shape_dims,
                    "stride_dims": stride_dims,
                    "stride_bits": 32 if value.use_32bit_stride else 64,
                }
            )
        return slots

    if isinstance(value, flydsl_typing.Stream):
        return [
            {
                "kind": "stream",
                **_ctype_metadata(ctypes.c_void_p, name="pointer"),
            }
        ]

    if isinstance(value, jit_argument.PointerJitArg):
        return [
            {
                "kind": "pointer",
                **_ctype_metadata(ctypes.c_void_p, name="pointer"),
            }
        ]

    if not isinstance(value, numeric.Numeric):
        raise NotImplementedError(
            "FlyDSL AOT supports tensor, pointer, scalar, and implicit stream "
            f"arguments; unsupported JIT argument type: {type(value).__name__}"
        )
    if value.signed is not None and value.width not in (1, 8, 16, 32, 64):
        raise NotImplementedError(
            f"FlyDSL AOT does not support {type(value).__name__} scalar arguments"
        )
    if value.signed is None and not isinstance(
        value,
        (
            numeric.Float16,
            numeric.BFloat16,
            numeric.Float32,
            numeric.Float64,
        ),
    ):
        raise NotImplementedError(
            f"FlyDSL AOT does not support {type(value).__name__} scalar arguments"
        )
    abi_spec = protocol.c_abi_spec(value)
    if len(abi_spec) != 1:
        raise NotImplementedError(
            "FlyDSL AOT scalar arguments must have exactly one C ABI slot"
        )
    ctype, _fill = abi_spec[0]
    slot = {
        "kind": "scalar",
        **_ctype_metadata(ctype, name=_numeric_ctype_name(value, ctype)),
    }
    if isinstance(value, numeric.Float16):
        slot["encoding"] = "float16_bits"
    elif isinstance(value, numeric.BFloat16):
        slot["encoding"] = "bfloat16_bits"
    return [slot]


def _launcher_abi(
    sig, bound, jit_args: list[Any], has_user_stream: bool
) -> tuple[dict[str, Any], ...]:
    abi = []
    jit_arg_index = 0
    for arg_index, (param_name, value) in enumerate(bound.arguments.items()):
        annotation = sig.parameters[param_name].annotation
        if annotation is not inspect.Parameter.empty and (
            flydsl_typing.Constexpr.is_constexpr_annotation(annotation)
            or jit_argument.is_type_param_annotation(annotation)
        ):
            continue
        jit_arg = jit_args[jit_arg_index]
        jit_arg_index += 1
        abi.extend(
            {"arg_index": arg_index, "arg_name": param_name, **slot}
            for slot in _argument_abi(jit_arg)
        )
    if not has_user_stream:
        abi.append(
            {
                "arg_index": None,
                "arg_name": None,
                "kind": "stream",
                **_ctype_metadata(ctypes.c_void_p, name="pointer"),
            }
        )
    return tuple(abi)


def _rename_symbol_ref(attr, symbol_map):
    components = list(ir.SymbolRefAttr(attr).value)
    if not components or components[0] not in symbol_map:
        return attr
    components[0] = symbol_map[components[0]]
    return ir.SymbolRefAttr.get(components)


def _rename_export_symbols(module: Any, entry: str, symbol: str) -> None:
    symbol_map = {}

    def collect(op):
        attrs = op.attributes
        if "sym_name" not in attrs:
            return ir.WalkResult.ADVANCE
        is_definition = op.name in ("llvm.mlir.global", "gpu.binary")
        if op.name == "llvm.func":
            is_definition = bool(op.opview.operation.regions)
        if not is_definition:
            return ir.WalkResult.ADVANCE
        old_name = attrs["sym_name"].value
        new_name = symbol if old_name == entry else f"{symbol}__{old_name}"
        symbol_map[old_name] = new_name
        attrs["sym_name"] = ir.StringAttr.get(new_name)
        return ir.WalkResult.ADVANCE

    def update_refs(op):
        attrs = op.attributes
        if op.name == "llvm.call" and "callee" in attrs:
            old_name = attrs["callee"].value
            if old_name in symbol_map:
                attrs["callee"] = ir.FlatSymbolRefAttr.get(symbol_map[old_name])
        elif op.name == "llvm.mlir.addressof" and "global_name" in attrs:
            old_name = attrs["global_name"].value
            if old_name in symbol_map:
                attrs["global_name"] = ir.FlatSymbolRefAttr.get(symbol_map[old_name])
        elif op.name in ("llvm.mlir.global_ctors", "llvm.mlir.global_dtors"):
            key = "ctors" if op.name.endswith("ctors") else "dtors"
            if key in attrs:
                attrs[key] = ir.ArrayAttr.get(
                    [
                        ir.FlatSymbolRefAttr.get(symbol_map.get(ref.value, ref.value))
                        for ref in attrs[key]
                    ]
                )
        elif op.name == "gpu.launch_func" and "kernel" in attrs:
            attrs["kernel"] = _rename_symbol_ref(attrs["kernel"], symbol_map)
        return ir.WalkResult.ADVANCE

    module.operation.walk(collect)
    if entry not in symbol_map:
        raise RuntimeError(f"FlyDSL AOT export could not find entry symbol {entry!r}")
    module.operation.walk(update_refs)


def _rename_loader_symbols(object_path: Path, symbol: str) -> tuple[str, str]:
    objcopy = shutil.which("llvm-objcopy") or shutil.which("objcopy")
    if objcopy is None:
        raise RuntimeError("FlyDSL AOT export requires llvm-objcopy or objcopy")

    init_symbol = f"{symbol}__{_GPU_MODULE_INIT}"
    load_symbol = f"{symbol}__{_GPU_MODULE_LOAD_TO_DEVICE}"
    symbol_map = {
        _GPU_MODULE_INIT: init_symbol,
        _GPU_MODULE_LOAD_TO_DEVICE: load_symbol,
        f"_mlir_{_GPU_MODULE_INIT}": f"_mlir_{init_symbol}",
        f"_mlir_{_GPU_MODULE_LOAD_TO_DEVICE}": f"_mlir_{load_symbol}",
    }
    file_descriptor, renamed_path = tempfile.mkstemp(
        prefix=f".{object_path.name}.",
        dir=object_path.parent,
    )
    os.close(file_descriptor)
    try:
        command = [objcopy]
        for old_name, new_name in symbol_map.items():
            command.extend(["--redefine-sym", f"{old_name}={new_name}"])
        command.extend(["--", str(object_path), renamed_path])
        subprocess.run(command, check=True, capture_output=True, text=True)
        Path(renamed_path).replace(object_path)
    finally:
        Path(renamed_path).unlink(missing_ok=True)
    return init_symbol, load_symbol


def _elf_soname(path: Path) -> str | None:
    readelf = shutil.which("llvm-readelf") or shutil.which("readelf")
    if readelf is None:
        raise RuntimeError("FlyDSL AOT export requires llvm-readelf or readelf")
    result = subprocess.run(
        [readelf, "-d", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    match = _ELF_SONAME.search(result.stdout)
    return match.group(1) if match is not None else None


def _runtime_library_dependencies(path: Path) -> dict[str, Path]:
    ldd = shutil.which("ldd")
    if ldd is None:
        raise RuntimeError("FlyDSL AOT export requires ldd")
    result = subprocess.run(
        [ldd, str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    dependencies = {}
    for line in result.stdout.splitlines():
        match = _LDD_LIBRARY.match(line)
        if match is not None and match.group(2) != "not":
            dependencies[match.group(1)] = Path(match.group(2)).resolve()
    return dependencies


def _flydsl_distribution_root(runtime_library: Path) -> Path:
    for parent in runtime_library.resolve().parents:
        if (parent / "flydsl").is_dir():
            return parent
    raise RuntimeError(
        f"could not locate the FlyDSL distribution for runtime library {runtime_library}"
    )


def _publish_runtime_library(source: Path, destination: Path) -> None:
    file_descriptor, temporary_path = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        dir=destination.parent,
    )
    os.close(file_descriptor)
    temporary = Path(temporary_path)
    try:
        shutil.copy2(source, temporary)
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if not filecmp.cmp(temporary, destination, shallow=False):
                raise RuntimeError(
                    "FlyDSL runtime library collision for "
                    f"{destination.name}: {source} differs from the cached file"
                ) from None
    finally:
        temporary.unlink(missing_ok=True)


def _bundle_runtime_libraries(
    runtime_libraries: list[str],
    output_dir: Path,
) -> list[str]:
    libraries: dict[str, Path] = {}
    distribution_roots = OrderedSet(
        _flydsl_distribution_root(Path(path)) for path in runtime_libraries
    )
    for path_string in runtime_libraries:
        path = Path(path_string).resolve()
        libraries[_elf_soname(path) or path.name] = path
        libraries.update(
            {
                name: dependency
                for name, dependency in _runtime_library_dependencies(path).items()
                if any(dependency.is_relative_to(root) for root in distribution_roots)
            }
        )

    bundled = []
    for name, source in libraries.items():
        destination = output_dir / name
        if source != destination:
            _publish_runtime_library(source, destination)
        bundled.append(str(destination))
    return bundled


class CompiledAOTLauncher:
    def __init__(self, module: Any, entry: str, abi: tuple[dict[str, Any], ...]):
        self._ir_text = str(module)
        self._entry = entry
        self.abi = abi

    def export_to_c(self, object_file_path: str, function_name: str) -> dict[str, Any]:
        if not _C_SYMBOL.fullmatch(function_name):
            raise ValueError(f"invalid C function name: {function_name!r}")
        object_path = Path(object_file_path)
        if not object_path.parent.exists():
            raise FileNotFoundError(
                f"object output directory does not exist: {object_path.parent}"
            )

        runtime_libraries = jit_executor._resolve_runtime_libs()
        ctx = jit_function._create_mlir_context()
        with ctx:
            module = ir.Module.parse(self._ir_text)
            _rename_export_symbols(module, self._entry, function_name)
            engine = execution_engine.ExecutionEngine(
                module,
                opt_level=3,
                shared_libs=runtime_libraries,
                enable_pic=True,
            )
            engine.dump_to_object_file(str(object_path))

        init_symbol, load_symbol = _rename_loader_symbols(object_path, function_name)
        bundled_runtime_libraries = _bundle_runtime_libraries(
            runtime_libraries,
            object_path.parent,
        )
        return {
            "object_file_path": str(object_path),
            "symbol": f"_mlir_{function_name}",
            "runtime_libraries": bundled_runtime_libraries,
            "abi": list(self.abi),
            "module_init_symbol": init_symbol,
            "module_load_symbol": load_symbol,
        }


def compile_aot(launcher: Any, *args, **kwargs) -> CompiledAOTLauncher:
    """Compile a specialized FlyDSL launcher into an AOT object and ABI."""
    if not HAS_FLYDSL:
        raise RuntimeError("FlyDSL AOT compilation requires the FlyDSL runtime")
    if not isinstance(launcher, jit_function.JitFunction):
        raise TypeError(
            f"flyc.compile_aot() expects a @flyc.jit function, got {type(launcher).__name__}"
        )

    launcher._ensure_sig()
    bound_self = None
    if launcher._has_self_param:
        if not args:
            raise TypeError(f"{launcher.func.__name__}() missing 'self' argument")
        bound_self, args = args[0], args[1:]
    sig = launcher._sig
    bound = sig.bind(*args, **kwargs)
    bound.apply_defaults()

    hints = (
        kernel_function.CompilationContext.compile_hints(launcher.compile_hints)
        if launcher.compile_hints
        else nullcontext()
    )
    with jit_function._create_mlir_context() as ctx, hints:
        param_names, jit_args, dsl_types, constexpr_values = (
            jit_argument.convert_to_jit_arguments(sig, bound)
        )
        has_user_stream = jit_function._ensure_stream_arg(jit_args)
        ir_types = protocol.get_ir_types(jit_args)
        loc = kernel_function.func_def_location(launcher.func, ctx)
        module = ir.Module.create(loc=loc)
        module.operation.attributes["gpu.container_module"] = ir.UnitAttr.get()

        with ir.InsertionPoint(module.body), loc:
            backend = backends.get_backend()
            gpu_module = kernel_function.create_gpu_module(
                "kernels",
                targets=backend.gpu_module_targets(),
                use_explicit_module=True,
            )
            func_op = func.FuncOp(launcher.func.__name__, (ir_types, []))
            func_op.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()
            entry_block = func_op.add_entry_block()

            with kernel_function.CompilationContext.create() as comp_ctx:
                comp_ctx.gpu_module_op = gpu_module
                comp_ctx.gpu_module_body = kernel_function.get_gpu_module_body(
                    gpu_module
                )
                with ir.InsertionPoint(entry_block):
                    ir_args = list(func_op.regions[0].blocks[0].arguments)
                    if not has_user_stream:
                        comp_ctx.stream_arg = ir_args[-1]
                    user_jit_args = jit_args[: len(param_names)]
                    dsl_args = protocol.construct_from_ir_values(
                        dsl_types, user_jit_args, ir_args
                    )
                    named_args = dict(zip(param_names, dsl_args))
                    named_args.update(constexpr_values)
                    fastmath_flag = kernel_function.effective_fastmath_hint(
                        kernel_function.CompilationContext.get_compile_hints()
                    )
                    fastmath_scope = (
                        arith.fastmath(fastmath_flag)
                        if fastmath_flag is not None
                        else nullcontext()
                    )
                    with meta.tracing_context(launcher.func), fastmath_scope:
                        if bound_self is not None:
                            launcher.func(bound_self, **named_args)
                        else:
                            launcher.func(**named_args)
                    func.ReturnOp([])

        link_libs = list(comp_ctx.link_libs) if comp_ctx.link_libs else None
        if comp_ctx.post_load_processors:
            raise RuntimeError(
                "FlyDSL AOT export does not support Python post-load processors"
            )
        if link_libs and jit_function._use_external_binary_codegen():
            raise RuntimeError(
                "FlyDSL external codegen does not support extern-linked AOT launchers"
            )
        if link_libs and "targets" in gpu_module.operation.attributes:
            del gpu_module.operation.attributes["targets"]

        flydsl_utils.log().info(f"FlyDSL AOT jit_args={jit_args}")
        compiled_module = jit_function.MlirCompiler.compile(
            module,
            arch=backend.target.arch,
            func_name=launcher.func.__name__,
            link_libs=link_libs,
        )
        abi = _launcher_abi(sig, bound, jit_args, has_user_stream)
        return CompiledAOTLauncher(compiled_module, launcher.func.__name__, abi)
