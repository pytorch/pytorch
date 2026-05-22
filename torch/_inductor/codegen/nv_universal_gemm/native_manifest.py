# mypy: allow-untyped-defs
from __future__ import annotations

import dataclasses
import importlib.util
import re
import sys
from pathlib import Path
from typing import Any


NVGEMM_NATIVE_C_ABI_VERSION = 1


@dataclasses.dataclass(frozen=True)
class NVGemmTensorArgSpec:
    name: str
    role: str
    dtype: str
    sizes: tuple[str, ...]
    strides: tuple[str, ...]
    offset: str

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class NVGemmTensorDescriptorABI:
    struct_fields: tuple[tuple[str, str], ...]
    descriptor_arg_kind: str

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class NVGemmModuleHooks:
    module_type: str
    load_hook: str
    unload_hook: str
    load_scope: str

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class NVGemmCHeaderField:
    name: str
    c_type: str
    array_size: int | None


@dataclasses.dataclass(frozen=True)
class NVGemmCHeaderTensorDescriptor:
    struct_name: str
    wrapper_arg_name: str | None
    fields: tuple[NVGemmCHeaderField, ...]


@dataclasses.dataclass(frozen=True)
class NVGemmCHeaderParam:
    name: str
    c_type: str
    kind: str
    tensor_struct_name: str | None = None


@dataclasses.dataclass(frozen=True)
class NVGemmCHeaderMetadata:
    module_struct_name: str | None
    module_load_hook: str | None
    module_unload_hook: str | None
    wrapper_name: str | None
    wrapper_return_type: str | None
    wrapper_params: tuple[NVGemmCHeaderParam, ...]
    tensor_descriptors: tuple[NVGemmCHeaderTensorDescriptor, ...]


@dataclasses.dataclass(frozen=True)
class NVGemmNativeArtifact:
    object_path: str | None
    header_path: str | None
    symbol_prefix: str
    wrapper_name: str
    link_flags: tuple[str, ...]
    header_metadata: NVGemmCHeaderMetadata | None

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class NVGemmNativeManifest:
    abi_version: int
    backend: str
    inductor_kernel_name: str
    python_entry: str
    cutlass_kernel_name: str
    variant: str
    accumulator_type: str
    workspace_size: int
    tensor_args: tuple[NVGemmTensorArgSpec, ...]
    tensor_descriptor_abi: NVGemmTensorDescriptorABI
    module_hooks: NVGemmModuleHooks
    stream_abi: str
    native_artifact: NVGemmNativeArtifact
    stable_cache_key_inputs: tuple[str, ...]
    export_options: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        result = dataclasses.asdict(self)
        result["tensor_args"] = [arg.to_dict() for arg in self.tensor_args]
        result["tensor_descriptor_abi"] = self.tensor_descriptor_abi.to_dict()
        result["module_hooks"] = self.module_hooks.to_dict()
        result["native_artifact"] = self.native_artifact.to_dict()
        result["stable_cache_key_inputs"] = list(self.stable_cache_key_inputs)
        return result


def default_tensor_descriptor_abi() -> NVGemmTensorDescriptorABI:
    return NVGemmTensorDescriptorABI(
        struct_fields=(
            ("data", "void *"),
            ("dynamic_shapes", "int32_t[]"),
            ("dynamic_strides", "int64_t[]"),
        ),
        descriptor_arg_kind="pointer_to_generated_descriptor_struct",
    )


def pending_native_artifact(symbol_prefix: str) -> NVGemmNativeArtifact:
    return NVGemmNativeArtifact(
        object_path=None,
        header_path=None,
        symbol_prefix=symbol_prefix,
        wrapper_name=f"cute_dsl_{symbol_prefix}_wrapper",
        link_flags=(),
        header_metadata=None,
    )


def module_hooks_for_symbol(symbol_prefix: str) -> NVGemmModuleHooks:
    return NVGemmModuleHooks(
        module_type=f"{symbol_prefix}_Kernel_Module_t",
        load_hook=f"{symbol_prefix}_Kernel_Module_Load",
        unload_hook=f"{symbol_prefix}_Kernel_Module_Unload",
        load_scope="once_per_wrapper_device",
    )


def export_compiled_artifact_to_c(
    compiled_obj: Any,
    output_dir: str | Path,
    file_name: str,
    function_prefix: str,
) -> NVGemmNativeArtifact:
    import cutlass_api.config

    compiled_exporter: Any = getattr(compiled_obj, "compiled_obj", None)
    if compiled_exporter is None:
        compiled_exporter = compiled_obj
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    global_options = cutlass_api.config.GlobalOptions()
    old_use_tvm_ffi = global_options.use_tvm_ffi
    global_options.use_tvm_ffi = False
    try:
        compiled_exporter.export_to_c(str(output), file_name, function_prefix)
    finally:
        global_options.use_tvm_ffi = old_use_tvm_ffi

    object_path = output / f"{file_name}.o"
    header_path = output / f"{file_name}.h"
    header_metadata = parse_nvgemm_header_metadata(header_path, function_prefix)
    return NVGemmNativeArtifact(
        object_path=str(object_path),
        header_path=str(header_path),
        symbol_prefix=function_prefix,
        wrapper_name=f"cute_dsl_{function_prefix}_wrapper",
        link_flags=(str(object_path), *nvgemm_runtime_link_flags()),
        header_metadata=header_metadata,
    )


def export_compiled_artifact_to_cache(
    compiled_obj: Any,
    stable_cache_key_inputs: tuple[str, ...],
) -> NVGemmNativeArtifact:
    from torch._inductor.codecache import code_hash, get_path

    cache_key = "\n".join(stable_cache_key_inputs)
    basename = code_hash(cache_key, extra=f"nvgemm-c-abi-{NVGEMM_NATIVE_C_ABI_VERSION}")
    _, output_dir, _ = get_path(basename, "nvgemm")
    symbol_prefix = f"nvgemm_{basename}"
    object_path = Path(output_dir) / f"{symbol_prefix}.o"
    header_path = Path(output_dir) / f"{symbol_prefix}.h"
    if object_path.is_file() and header_path.is_file():
        return NVGemmNativeArtifact(
            object_path=str(object_path),
            header_path=str(header_path),
            symbol_prefix=symbol_prefix,
            wrapper_name=f"cute_dsl_{symbol_prefix}_wrapper",
            link_flags=(str(object_path), *nvgemm_runtime_link_flags()),
            header_metadata=parse_nvgemm_header_metadata(
                header_path,
                symbol_prefix,
            ),
        )

    return export_compiled_artifact_to_c(
        compiled_obj,
        output_dir,
        symbol_prefix,
        symbol_prefix,
    )


def nvgemm_runtime_link_flags() -> tuple[str, ...]:
    nvidia_cutlass_dsl = importlib.util.find_spec("nvidia_cutlass_dsl")
    candidate_dirs = []
    if nvidia_cutlass_dsl is not None and nvidia_cutlass_dsl.origin is not None:
        candidate_dirs.append(Path(nvidia_cutlass_dsl.origin).parent / "lib")
    py_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    candidate_dirs.extend(
        [
            Path("/usr/local/lib")
            / py_version
            / "dist-packages/nvidia_cutlass_dsl/lib",
            Path("/usr/local/lib")
            / py_version
            / "site-packages/nvidia_cutlass_dsl/lib",
        ]
    )
    lib_dir = next((path for path in candidate_dirs if path.is_dir()), None)
    if lib_dir is None:
        return ()

    flags = [
        f"-L{lib_dir}",
        f"-Wl,-rpath,{lib_dir}",
        "-lcute_dsl_runtime",
    ]

    cuda_lib_dir = Path("/usr/local/cuda/lib64")
    if cuda_lib_dir.is_dir():
        flags.extend(
            [
                f"-L{cuda_lib_dir}",
                f"-Wl,-rpath,{cuda_lib_dir}",
                "-lcudart",
            ]
        )

    return tuple(flags)


def _parse_c_declaration(declaration: str) -> tuple[str, str, int | None]:
    declaration = declaration.strip().rstrip(";")
    array_size = None
    array_match = re.search(r"\[(\d+)\]$", declaration)
    if array_match is not None:
        array_size = int(array_match.group(1))
        declaration = declaration[: array_match.start()]

    name_match = re.search(r"([A-Za-z_]\w*)$", declaration)
    if name_match is None:
        raise RuntimeError(f"Could not parse C declaration: {declaration!r}")
    name = name_match.group(1)
    c_type = declaration[: name_match.start()].strip()
    if c_type.endswith("*"):
        c_type = c_type[:-1].rstrip() + " *"
    return c_type, name, array_size


def _parse_c_field(line: str) -> NVGemmCHeaderField | None:
    line = line.strip()
    if not line or line.startswith("//"):
        return None
    c_type, name, array_size = _parse_c_declaration(line)
    return NVGemmCHeaderField(name=name, c_type=c_type, array_size=array_size)


def parse_nvgemm_header_metadata(
    header_path: str | Path,
    symbol_prefix: str,
) -> NVGemmCHeaderMetadata | None:
    """Extract the generated C ABI shape needed by cpp_wrapper codegen."""
    path = Path(header_path)
    if not path.is_file():
        return None

    text = path.read_text()
    structs: dict[str, tuple[NVGemmCHeaderField, ...]] = {}
    for match in re.finditer(
        rf"typedef\s+struct\s*\{{(?P<body>.*?)\}}\s*"
        rf"(?P<name>{re.escape(symbol_prefix)}_[A-Za-z0-9_]+);",
        text,
        re.DOTALL,
    ):
        fields = tuple(
            field
            for line in match.group("body").splitlines()
            if (field := _parse_c_field(line)) is not None
        )
        structs[match.group("name")] = fields

    wrapper_name = f"cute_dsl_{symbol_prefix}_wrapper"
    wrapper_match = re.search(
        rf"static\s+inline\s+(?P<ret>\w+)\s+"
        rf"(?P<name>{re.escape(wrapper_name)})\((?P<params>.*?)\)\s*\{{",
        text,
        re.DOTALL,
    )
    wrapper_return_type = None
    wrapper_params: tuple[NVGemmCHeaderParam, ...] = ()
    if wrapper_match is not None:
        wrapper_return_type = wrapper_match.group("ret")
        params = []
        for param_decl in wrapper_match.group("params").split(","):
            c_type, name, _ = _parse_c_declaration(param_decl)
            tensor_struct_name = None
            kind = "scalar"
            c_type_without_ptr = c_type.removesuffix(" *")
            if c_type_without_ptr in structs and "_Tensor_" in c_type_without_ptr:
                kind = "tensor_descriptor"
                tensor_struct_name = c_type_without_ptr
            elif c_type_without_ptr.endswith("_Kernel_Module_t"):
                kind = "module"
            elif c_type_without_ptr == "cudaStream_t":
                kind = "stream"
            params.append(
                NVGemmCHeaderParam(
                    name=name,
                    c_type=c_type,
                    kind=kind,
                    tensor_struct_name=tensor_struct_name,
                )
            )
        wrapper_params = tuple(params)

    def wrapper_arg_for_struct(struct_name: str) -> str | None:
        for param in wrapper_params:
            if param.tensor_struct_name == struct_name:
                return param.name
        return None

    tensor_descriptors = tuple(
        NVGemmCHeaderTensorDescriptor(
            struct_name=struct_name,
            wrapper_arg_name=wrapper_arg_for_struct(struct_name),
            fields=fields,
        )
        for struct_name, fields in structs.items()
        if "_Tensor_" in struct_name
    )

    return NVGemmCHeaderMetadata(
        module_struct_name=(
            f"{symbol_prefix}_Kernel_Module_t"
            if f"{symbol_prefix}_Kernel_Module_t" in structs
            else None
        ),
        module_load_hook=f"{symbol_prefix}_Kernel_Module_Load",
        module_unload_hook=f"{symbol_prefix}_Kernel_Module_Unload",
        wrapper_name=wrapper_match.group("name") if wrapper_match is not None else None,
        wrapper_return_type=wrapper_return_type,
        wrapper_params=wrapper_params,
        tensor_descriptors=tensor_descriptors,
    )


def with_native_artifact(
    manifest: NVGemmNativeManifest,
    native_artifact: NVGemmNativeArtifact,
) -> NVGemmNativeManifest:
    return dataclasses.replace(
        manifest,
        native_artifact=native_artifact,
        module_hooks=module_hooks_for_symbol(native_artifact.symbol_prefix),
    )


def build_stable_cache_key_inputs(
    *,
    cutlass_kernel_name: str,
    variant: str,
    accumulator_type: str,
    workspace_size: int,
    tensor_args: tuple[NVGemmTensorArgSpec, ...],
    static_options: tuple[str, ...] = (),
) -> tuple[str, ...]:
    tensor_key = tuple(
        (
            f"{arg.role}:{arg.dtype}:sizes={arg.sizes}:"
            f"strides={arg.strides}:offset={arg.offset}"
        )
        for arg in tensor_args
    )
    result = (
        f"abi_version={NVGEMM_NATIVE_C_ABI_VERSION}",
        "use_tvm_ffi=False",
        f"cutlass_kernel_name={cutlass_kernel_name}",
        f"variant={variant}",
        f"accumulator_type={accumulator_type}",
        f"workspace_size={workspace_size}",
        *static_options,
        *tensor_key,
    )
    return result
