# mypy: allow-untyped-defs

from __future__ import annotations

import os
import secrets
import shlex
import tempfile
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

from torch._inductor import config
from torch._inductor.codegen.cpp_utils import cexpr
from torch._inductor.runtime.runtime_utils import cache_dir
from torch._inductor.virtualized import V


if TYPE_CHECKING:
    import inspect


@dataclass(frozen=True)
class FlyDSLAOTArtifact:
    object_file_path: str
    symbol: str
    runtime_libraries: tuple[str, ...]
    abi: tuple[dict[str, Any], ...]
    module_init_symbol: str
    module_load_symbol: str


def argument_signature(arguments: tuple[Any, ...]) -> tuple[Any, ...]:
    signature: list[tuple[Any, ...]] = []
    for argument in arguments:
        if hasattr(argument, "shape") and hasattr(argument, "stride"):
            signature.append(
                (
                    "tensor",
                    str(argument.dtype),
                    tuple(argument.shape),
                    tuple(argument.stride()),
                )
            )
        else:
            signature.append((type(argument).__qualname__, repr(argument)))
    return tuple(signature)


def compile_launcher(
    launcher: Any,
    arguments: tuple[Any, ...],
    *,
    signature: inspect.Signature,
    bound_self: Any = None,
) -> FlyDSLAOTArtifact:
    from torch._higher_order_ops.flydsl_kernel_wrap import (
        split_flydsl_launcher_arguments,
    )
    from torch._inductor.codegen.flydsl.aot_compile import compile_aot

    positional_args, keyword_args = split_flydsl_launcher_arguments(
        signature, arguments
    )
    if bound_self is not None:
        positional_args = (bound_self, *positional_args)
    compiled = compile_aot(launcher, *positional_args, **keyword_args)
    output_dir = cache_dir()
    os.makedirs(output_dir, exist_ok=True)
    file_descriptor, object_path = tempfile.mkstemp(
        prefix="flydsl_launcher_",
        suffix=".o",
        dir=output_dir,
    )
    os.close(file_descriptor)
    symbol = f"flydsl_launcher_{secrets.token_hex(12)}"
    try:
        metadata = compiled.export_to_c(object_path, symbol)
    except Exception:
        os.unlink(object_path)
        raise
    return FlyDSLAOTArtifact(
        object_file_path=metadata["object_file_path"],
        symbol=metadata["symbol"],
        runtime_libraries=tuple(metadata["runtime_libraries"]),
        abi=tuple(metadata["abi"]),
        module_init_symbol=metadata["module_init_symbol"],
        module_load_symbol=metadata["module_load_symbol"],
    )


def _define_module_loader(wrapper, artifact: FlyDSLAOTArtifact) -> None:
    if not hasattr(wrapper, "_flydsl_module_loader_header_emitted"):
        wrapper._flydsl_module_loader_header_emitted = True
        wrapper.header.writeline("#include <cstdint>")
        wrapper.header.writeline("#include <mutex>")
        wrapper.header.writeline("#include <stdexcept>")
        wrapper.header.splice(
            """
            static bool flydsl_gpu_get_device(int* device_index) {
            #ifdef USE_ROCM
              return hipGetDevice(device_index) == hipSuccess;
            #else
              return cudaGetDevice(device_index) == cudaSuccess;
            #endif
            }

            static void flydsl_gpu_module_unload(void* module) noexcept {
            #ifdef USE_ROCM
              (void)hipModuleUnload(reinterpret_cast<hipModule_t>(module));
            #else
              (void)cuModuleUnload(reinterpret_cast<CUmodule>(module));
            #endif
            }
            """
        )
    ensure_symbol = f"{artifact.symbol}__ensure_module_loaded"
    holder_symbol = f"{artifact.symbol}__module_holder"
    wrapper.header.writeline(
        f'extern "C" void {artifact.module_init_symbol}(void**, int32_t*);'
    )
    wrapper.header.writeline(
        f'extern "C" void {artifact.module_load_symbol}(void**, int32_t*);'
    )
    wrapper.header.splice(
        f"""
        struct {holder_symbol} {{
          std::once_flag once;
          void* module = nullptr;
          int device_index = -1;

          ~{holder_symbol}() noexcept {{
            if (module != nullptr) {{
              flydsl_gpu_module_unload(module);
            }}
          }}
        }};

        static {holder_symbol} {holder_symbol}_instance;

        static void {ensure_symbol}() {{
          int device_index = -1;
          if (!flydsl_gpu_get_device(&device_index)) {{
            throw std::runtime_error("FlyDSL could not query the current GPU device");
          }}
          std::call_once({holder_symbol}_instance.once, [device_index]() {{
            int32_t error = 0;
            {artifact.module_init_symbol}(
                &{holder_symbol}_instance.module, &error);
            if (error != 0) {{
              throw std::runtime_error("FlyDSL GPU module initialization failed");
            }}
            {artifact.module_load_symbol}(
                &{holder_symbol}_instance.module, &error);
            if (error != 0 || {holder_symbol}_instance.module == nullptr) {{
              if ({holder_symbol}_instance.module != nullptr) {{
                flydsl_gpu_module_unload({holder_symbol}_instance.module);
                {holder_symbol}_instance.module = nullptr;
              }}
              throw std::runtime_error("FlyDSL GPU module load failed");
            }}
            {holder_symbol}_instance.device_index = device_index;
          }});
          if ({holder_symbol}_instance.device_index != device_index) {{
            throw std::runtime_error(
                "FlyDSL AOT modules cannot be shared across GPU devices");
          }}
        }}
        """
    )


def define_aot_kernel(wrapper, launcher_idx: int, call_spec_idx: int, example_args):
    from torch._higher_order_ops.flydsl_kernel_wrap import (
        flydsl_launcher_side_table,
        restore_flydsl_launcher_arguments,
    )
    from torch._inductor.codecache import ROCmCodeCache

    if not V.graph.aot_mode:
        raise NotImplementedError(
            "FlyDSL C++ wrapper launchers are currently supported in AOT mode"
        )
    if config.aot_inductor.package_cpp_only:
        raise NotImplementedError(
            "FlyDSL AOT launchers do not support aot_inductor.package_cpp_only; "
            "the packaged runtime libraries require linking before relocation"
        )
    full_example_args = restore_flydsl_launcher_arguments(
        example_args,
        call_spec_idx,
    )
    cache_key = (
        "flydsl_aot",
        launcher_idx,
        call_spec_idx,
        argument_signature(full_example_args),
    )
    cached = wrapper.user_defined_kernel_cache.get(cache_key)
    if cached is not None:
        return cached[2]["artifact"]

    registration = flydsl_launcher_side_table.get_registration(launcher_idx)
    artifact = compile_launcher(
        registration.launcher,
        full_example_args,
        signature=registration.signature,
        bound_self=registration.bound_self,
    )
    if artifact.object_file_path not in ROCmCodeCache.aot_kernels_o:
        ROCmCodeCache.aot_kernels_o.append(artifact.object_file_path)
    for path in artifact.runtime_libraries:
        wrapper.external_kernel_libs.add(f"-L{shlex.quote(os.path.dirname(path))}")
        wrapper.external_kernel_libs.add(f"-l:{shlex.quote(os.path.basename(path))}")
        if path not in wrapper.additional_files:
            wrapper.additional_files.append(path)
    if artifact.runtime_libraries:
        wrapper.external_kernel_libs.add("-Wl,-rpath,$ORIGIN")
    wrapper.header.writeline(f'extern "C" void {artifact.symbol}(void**);')
    _define_module_loader(wrapper, artifact)
    wrapper.user_defined_kernel_cache[cache_key] = (
        artifact.symbol,
        None,
        {"artifact": artifact},
    )
    return artifact


def generate_aot_kernel_call(
    wrapper,
    artifact,
    kernel_args,
    *,
    device,
    current_stream_idx,
) -> None:
    """Pack a FlyDSL ABI call and emit its AOTI launch on the scheduler stream."""
    from torch._inductor.ir import IRNode
    from torch._inductor.stream_utils import DEFAULT_STREAM_IDX, get_stream_name

    if not hasattr(wrapper, "_flydsl_call_counter"):
        wrapper._flydsl_call_counter = 0
        wrapper.header.writeline("#include <cstring>")
    call_index = wrapper._flydsl_call_counter
    wrapper._flydsl_call_counter += 1
    wrapper.writeline(f"{artifact.symbol}__ensure_module_loaded();")
    stream = (
        get_stream_name(current_stream_idx)
        if V.graph.aot_mode
        and current_stream_idx is not None
        and current_stream_idx != DEFAULT_STREAM_IDX
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
        arg_index = slot["arg_index"]
        arg = kernel_args[arg_index] if arg_index is not None else None
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
            for dim in slot["shape_dims"]:
                value_name = f"{slot_name}_shape_{dim}"
                value = cexpr(V.graph.sizevars.simplify(arg.get_size()[dim]))
                wrapper.writeline(
                    f"int32_t {value_name} = static_cast<int32_t>({value});"
                )
                wrapper.writeline(
                    f"std::memcpy({slot_name} + {offset}, &{value_name}, "
                    f"sizeof({value_name}));"
                )
                offset += 4
            stride_type = "int32_t" if slot["stride_bits"] == 32 else "int64_t"
            stride_size = 4 if slot["stride_bits"] == 32 else 8
            for dim in slot["stride_dims"]:
                value_name = f"{slot_name}_stride_{dim}"
                value = cexpr(V.graph.sizevars.simplify(arg.get_stride()[dim]))
                wrapper.writeline(
                    f"{stride_type} {value_name} = static_cast<{stride_type}>({value});"
                )
                wrapper.writeline(
                    f"std::memcpy({slot_name} + {offset}, &{value_name}, "
                    f"sizeof({value_name}));"
                )
                offset += stride_size
            packed_entries.append(slot_name)
        elif kind == "scalar":
            encoding = slot.get("encoding")
            value = cexpr(V.graph.sizevars.simplify(arg))
            if encoding == "float16_bits":
                wrapper.writeline(
                    f"uint16_t {slot_name} = "
                    "c10::detail::fp16_ieee_from_fp32_value("
                    f"static_cast<float>({value}));"
                )
                packed_entries.append(f"&{slot_name}")
                continue
            if encoding == "bfloat16_bits":
                wrapper.writeline(
                    f"uint16_t {slot_name} = c10::detail::bits_from_f32("
                    f"static_cast<float>({value}));"
                )
                packed_entries.append(f"&{slot_name}")
                continue
            scalar_types = {
                "bool": "bool",
                "float": "float",
                "double": "double",
                **{f"int{bits}": f"int{bits}_t" for bits in (8, 16, 32, 64)},
                **{f"uint{bits}": f"uint{bits}_t" for bits in (8, 16, 32, 64)},
            }
            scalar_type = scalar_types[slot["ctype"]]
            wrapper.writeline(
                f"{scalar_type} {slot_name} = static_cast<{scalar_type}>({value});"
            )
            packed_entries.append(f"&{slot_name}")
        elif kind in ("pointer", "stream"):
            if kind == "stream" and arg_index is not None:
                raise AssertionError(
                    "FlyDSL AOT launchers do not accept explicit stream arguments"
                )
            value = stream if kind == "stream" else cexpr(arg)
            wrapper.writeline(f"void* {slot_name} = reinterpret_cast<void*>({value});")
            packed_entries.append(f"&{slot_name}")
        else:
            raise NotImplementedError(f"Unsupported FlyDSL ABI slot kind: {kind}")
    packed_name = f"flydsl_packed_{call_index}"
    wrapper.writeline(f"void* {packed_name}[] = {{{', '.join(packed_entries)}}};")
    wrapper.writeline(f"{artifact.symbol}({packed_name});")
