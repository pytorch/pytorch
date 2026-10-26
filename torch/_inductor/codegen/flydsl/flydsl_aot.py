# mypy: allow-untyped-defs
from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TYPE_CHECKING

from torch._inductor.codegen.cpp_utils import cexpr
from torch._inductor.runtime.cache_dir_utils import cache_dir
from torch._inductor.runtime.flydsl_cache import temporary_env
from torch._inductor.virtualized import V

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


_AOT_SCHEMA_VERSION = 2
_SUPPORTED_ARCH = "gfx950"
_aot_compile_lock = threading.Lock()


@dataclass(frozen=True)
class FlyDSLAOTArtifact:
    object_file_path: str
    symbol: str
    module_init_symbol: str
    module_load_symbol: str
    module_unload_symbol: str
    abi: tuple[dict[str, Any], ...]
    linker_flags: tuple[str, ...]
    arch: str
    return_success: int


def _validate_metadata(metadata: Mapping[str, Any]) -> FlyDSLAOTArtifact:
    schema_version = metadata.get("schema_version")
    if schema_version != _AOT_SCHEMA_VERSION:
        raise RuntimeError(
            "PyTorch FlyDSL AOT requires export metadata schema "
            f"{_AOT_SCHEMA_VERSION}, got {schema_version!r}"
        )
    backend = metadata.get("backend")
    if backend != "rocm":
        raise RuntimeError(
            f"PyTorch FlyDSL AOT requires the ROCm backend, got {backend!r}"
        )
    arch = metadata.get("arch")
    if arch != _SUPPORTED_ARCH:
        raise RuntimeError(
            f"PyTorch FlyDSL AOT currently supports {_SUPPORTED_ARCH}, got {arch!r}"
        )

    return_abi = metadata.get("return_abi")
    if not isinstance(return_abi, dict) or (
        return_abi.get("kind"),
        return_abi.get("ctype"),
    ) != ("status", "int32"):
        raise RuntimeError(f"unsupported FlyDSL AOT return ABI: {return_abi!r}")

    runtime_libraries = metadata.get("runtime_libraries")
    if runtime_libraries != [{"link_name": "libamdhip64.so"}]:
        raise RuntimeError(
            "unexpected FlyDSL AOT runtime-library contract: " f"{runtime_libraries!r}"
        )

    abi = metadata.get("abi")
    if not isinstance(abi, list) or not all(isinstance(slot, dict) for slot in abi):
        raise RuntimeError("FlyDSL AOT metadata must contain an ABI slot list")
    for slot in abi:
        if slot.get("field") is not None:
            raise NotImplementedError(
                "PyTorch FlyDSL AOT does not yet support struct ABI fields"
            )
        if slot.get("kind") not in {
            "tensor_data",
            "tensor_layout",
            "pointer",
            "scalar",
            "stream",
        }:
            raise NotImplementedError(f"unsupported FlyDSL AOT ABI slot: {slot!r}")

    return FlyDSLAOTArtifact(
        object_file_path=str(metadata["object_file_path"]),
        symbol=str(metadata["symbol"]),
        module_init_symbol=str(metadata["module_init_symbol"]),
        module_load_symbol=str(metadata["module_load_symbol"]),
        module_unload_symbol=str(metadata["module_unload_symbol"]),
        abi=tuple(dict(slot) for slot in abi),
        linker_flags=tuple(str(flag) for flag in metadata["linker_flags"]),
        arch=str(arch),
        return_success=int(return_abi["success"]),
    )


def _placeholder_tensors(
    tensor_names: Sequence[str],
    shapes: Mapping[str, Sequence[int]],
    strides: Mapping[str, Sequence[int]],
    dtypes: Mapping[str, str],
) -> tuple[Any, ...]:
    import torch

    return tuple(
        torch.empty_strided(
            tuple(shapes[name]),
            tuple(strides[name]),
            dtype=getattr(torch, dtypes[name]),
            device="cpu",
        )
        for name in (*tensor_names, "output")
    )


def compile_aot_source(
    kernel_name: str,
    source_code: str,
    precompile_metadata: Mapping[str, Any],
) -> FlyDSLAOTArtifact:
    """Compile one rendered gfx950 FlyDSL template through FlyDSL's public API."""
    from torch._inductor.codecache import PyCodeCache
    from torch._subclasses.fake_tensor import FakeTensorMode

    arch = precompile_metadata.get("flydsl_gpu_arch")
    if arch != _SUPPORTED_ARCH:
        raise RuntimeError(
            f"PyTorch FlyDSL AOT currently supports {_SUPPORTED_ARCH}, got {arch!r}"
        )

    tensor_names = tuple(precompile_metadata["tensor_names"])
    key, source_path = PyCodeCache.write(source_code)
    module = PyCodeCache.load_by_key_path(key, source_path)
    main = getattr(module, f"{kernel_name}_main")
    if not hasattr(module, "run_cached_flydsl"):
        raise NotImplementedError(
            "FlyDSL AOT currently supports templates using run_cached_flydsl"
        )

    compiled_functions = []

    def capture_compile(
        jit_func,
        *compile_args,
        compiler,
        **_kwargs,
    ):
        compiled = compiler(jit_func, *compile_args)
        if compiled is None:
            raise RuntimeError("FlyDSL AOT compilation did not return an artifact")
        compiled_functions.append(compiled)
        return compiled

    with _aot_compile_lock:
        original_run_cached = module.run_cached_flydsl
        module.run_cached_flydsl = capture_compile
        try:
            with (
                temporary_env({"FLYDSL_GPU_ARCH": _SUPPORTED_ARCH}),
                FakeTensorMode(),
            ):
                placeholders = _placeholder_tensors(
                    tensor_names,
                    precompile_metadata["precompile_shapes"],
                    precompile_metadata["precompile_strides"],
                    precompile_metadata["precompile_dtypes"],
                )
                input_tensors, output = placeholders[:-1], placeholders[-1]
                main(*input_tensors, output, stream=0)
        finally:
            module.run_cached_flydsl = original_run_cached

    if len(compiled_functions) != 1:
        raise RuntimeError(
            "expected one FlyDSL AOT specialization, got " f"{len(compiled_functions)}"
        )

    output_dir = Path(cache_dir()) / "flydsl_aot"
    os.makedirs(output_dir, exist_ok=True)
    file_name = f"{kernel_name}_aot"
    symbol = f"flydsl_aot_{kernel_name}"
    metadata = compiled_functions[0].export_to_c(
        str(output_dir),
        file_name,
        symbol,
        return_metadata=True,
    )
    if metadata is None:
        raise RuntimeError(
            "FlyDSL export_to_c(return_metadata=True) returned no metadata"
        )
    return _validate_metadata(metadata)


def register_aot_artifact(wrapper, kernel_name: str, artifact: FlyDSLAOTArtifact):
    from torch._inductor import config

    if config.aot_inductor.package_cpp_only:
        raise NotImplementedError(
            "FlyDSL AOT does not yet support aot_inductor.package_cpp_only"
        )

    wrapper.aot_kernel_objects.add(artifact.object_file_path)
    wrapper.external_kernel_libs.update(artifact.linker_flags)

    artifacts = getattr(wrapper, "_flydsl_aot_artifacts", None)
    if artifacts is None:
        artifacts = {}
        wrapper._flydsl_aot_artifacts = artifacts
    artifacts[kernel_name] = artifact

    if not getattr(wrapper, "_flydsl_aot_header_emitted", False):
        wrapper._flydsl_aot_header_emitted = True
        wrapper.header.writeline("#include <cstdint>")
        wrapper.header.writeline("#include <cstring>")
        wrapper.header.writeline("#include <stdexcept>")
    wrapper.header.writeline(f'extern "C" int32_t {artifact.symbol}(void**);')
    wrapper.header.writeline(f'extern "C" int32_t {artifact.module_init_symbol}(void);')
    wrapper.header.writeline(
        f'extern "C" int32_t {artifact.module_load_symbol}(int32_t);'
    )


def _aot_artifact(wrapper, kernel_name: str) -> FlyDSLAOTArtifact:
    artifacts = getattr(wrapper, "_flydsl_aot_artifacts", {})
    if kernel_name not in artifacts:
        raise AssertionError(f"missing FlyDSL AOT artifact for {kernel_name}")
    return artifacts[kernel_name]


def _status_check(wrapper, expression: str, name: str, success: int) -> None:
    wrapper.writeline(f"int32_t {name} = {expression};")
    wrapper.writeline(
        f"if ({name} != {success}) throw std::runtime_error("
        f'"FlyDSL AOT call failed: {expression}");'
    )


def generate_aot_kernel_call(
    wrapper,
    kernel_name: str,
    kernel_args: Sequence[Any],
    *,
    device,
) -> None:
    """Pack and invoke one exported FlyDSL template on Inductor's stream."""
    from torch._inductor.ir import IRNode
    from torch._inductor.stream_utils import DEFAULT_STREAM_IDX, get_stream_name

    artifact = _aot_artifact(wrapper, kernel_name)
    call_index = getattr(wrapper, "_flydsl_aot_call_counter", 0)
    wrapper._flydsl_aot_call_counter = call_index + 1

    status_name = f"flydsl_status_{call_index}"
    _status_check(
        wrapper,
        f"{artifact.module_init_symbol}()",
        status_name,
        artifact.return_success,
    )
    _status_check(
        wrapper,
        f"{artifact.module_load_symbol}(-1)",
        f"{status_name}_load",
        artifact.return_success,
    )

    current_stream_idx = V.graph.scheduler.current_stream_idx
    stream = (
        get_stream_name(current_stream_idx)
        if current_stream_idx is not None and current_stream_idx != DEFAULT_STREAM_IDX
        else wrapper.write_get_raw_stream(device.index, V.graph.name)
    )

    if any(slot.get("encoding") == "float16_bits" for slot in artifact.abi):
        wrapper.header.writeline("#include <c10/util/Half.h>")
    if any(slot.get("encoding") == "bfloat16_bits" for slot in artifact.abi):
        wrapper.header.writeline("#include <c10/util/BFloat16.h>")

    packed_entries = []
    for slot_index, slot in enumerate(artifact.abi):
        slot_name = f"flydsl_{call_index}_{slot_index}"
        kind = slot["kind"]
        arg_index = slot.get("arg_index")
        arg = (
            kernel_args[arg_index]
            if arg_index is not None and arg_index < len(kernel_args)
            else None
        )

        if kind == "tensor_data":
            if not isinstance(arg, IRNode):
                raise AssertionError("FlyDSL tensor ABI slot requires an IR node")
            wrapper.writeline(
                f"void* {slot_name} = {arg.codegen_reference()}.data_ptr();"
            )
            packed_entries.append(f"&{slot_name}")
        elif kind == "tensor_layout":
            if not isinstance(arg, IRNode):
                raise AssertionError("FlyDSL layout ABI slot requires an IR node")
            wrapper.writeline(
                f"alignas({slot['alignment']}) unsigned char "
                f"{slot_name}[{slot['size']}] = {{}};"
            )
            offset = 0
            shape_bits = int(slot["shape_bits"])
            shape_type = f"int{shape_bits}_t"
            shape_size = shape_bits // 8
            for dim in slot.get("shape_dims", ()):
                value_name = f"{slot_name}_shape_{dim}"
                value = cexpr(V.graph.sizevars.simplify(arg.get_size()[dim]))
                wrapper.writeline(
                    f"{shape_type} {value_name} = "
                    f"static_cast<{shape_type}>({value});"
                )
                wrapper.writeline(
                    f"std::memcpy({slot_name} + {offset}, &{value_name}, "
                    f"sizeof({value_name}));"
                )
                offset += shape_size
            stride_bits = int(slot["stride_bits"])
            stride_type = f"int{stride_bits}_t"
            stride_size = stride_bits // 8
            for dim in slot.get("stride_dims", ()):
                value_name = f"{slot_name}_stride_{dim}"
                value = cexpr(V.graph.sizevars.simplify(arg.get_stride()[dim]))
                wrapper.writeline(
                    f"{stride_type} {value_name} = "
                    f"static_cast<{stride_type}>({value});"
                )
                wrapper.writeline(
                    f"std::memcpy({slot_name} + {offset}, &{value_name}, "
                    f"sizeof({value_name}));"
                )
                offset += stride_size
            if offset != slot["size"]:
                raise AssertionError(
                    f"FlyDSL layout ABI packed {offset} bytes, "
                    f"expected {slot['size']}"
                )
            packed_entries.append(slot_name)
        elif kind == "scalar":
            if arg is None:
                raise NotImplementedError(
                    f"FlyDSL scalar ABI slot has no runtime argument: {slot!r}"
                )
            encoding = slot.get("encoding")
            value = cexpr(V.graph.sizevars.simplify(arg))
            if encoding == "float16_bits":
                wrapper.writeline(
                    f"uint16_t {slot_name} = "
                    "c10::detail::fp16_ieee_from_fp32_value("
                    f"static_cast<float>({value}));"
                )
            elif encoding == "bfloat16_bits":
                wrapper.writeline(
                    f"uint16_t {slot_name} = c10::detail::bits_from_f32("
                    f"static_cast<float>({value}));"
                )
            else:
                scalar_types = {
                    "bool": "bool",
                    "float": "float",
                    "double": "double",
                    **{f"int{bits}": f"int{bits}_t" for bits in (8, 16, 32, 64)},
                    **{f"uint{bits}": f"uint{bits}_t" for bits in (8, 16, 32, 64)},
                }
                scalar_type = scalar_types[slot["ctype"]]
                wrapper.writeline(
                    f"{scalar_type} {slot_name} = "
                    f"static_cast<{scalar_type}>({value});"
                )
            packed_entries.append(f"&{slot_name}")
        elif kind == "pointer":
            if arg is None:
                raise NotImplementedError(
                    f"FlyDSL pointer ABI slot has no runtime argument: {slot!r}"
                )
            value = (
                f"{arg.codegen_reference()}.data_ptr()"
                if isinstance(arg, IRNode)
                else cexpr(arg)
            )
            wrapper.writeline(f"void* {slot_name} = reinterpret_cast<void*>({value});")
            packed_entries.append(f"&{slot_name}")
        elif kind == "stream":
            wrapper.writeline(f"void* {slot_name} = reinterpret_cast<void*>({stream});")
            packed_entries.append(f"&{slot_name}")
        else:
            raise NotImplementedError(f"unsupported FlyDSL ABI slot: {slot!r}")

    packed_name = f"flydsl_packed_{call_index}"
    wrapper.writeline(f"void* {packed_name}[] = {{{', '.join(packed_entries)}}};")
    _status_check(
        wrapper,
        f"{artifact.symbol}({packed_name})",
        f"{status_name}_call",
        artifact.return_success,
    )
