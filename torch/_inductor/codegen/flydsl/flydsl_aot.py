# mypy: allow-untyped-defs
from __future__ import annotations

import hashlib
import inspect
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TYPE_CHECKING

from torch._inductor.codegen.cpp_utils import cexpr
from torch._inductor.runtime.cache_dir_utils import cache_dir
from torch._inductor.runtime.flydsl_cache import (
    capture_flydsl_compiles,
    flydsl_compile_lock,
    temporary_env,
)
from torch._inductor.virtualized import V

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


_AOT_SCHEMA_MAJOR = 3
_AOT_SCHEMA_MINOR = 0
_SUPPORTED_ARCH = "gfx950"


@dataclass(frozen=True)
class FlyDSLLayoutEntry:
    kind: str
    dim: int
    offset: int
    bits: int
    signed: bool


@dataclass(frozen=True)
class FlyDSLAbiSlot:
    arg_index: int | None
    arg_name: str | None
    kind: str
    ctype: str
    size: int
    alignment: int
    encoding: str | None
    field: str | None
    layout_entries: tuple[FlyDSLLayoutEntry, ...]


@dataclass(frozen=True)
class FlyDSLAOTManifest:
    symbol: str
    metadata_symbol: str
    module_init_symbol: str
    module_load_symbol: str
    module_unload_symbol: str
    abi: tuple[FlyDSLAbiSlot, ...]
    linker_flags: tuple[str, ...]
    runtime_libraries: tuple[str, ...]
    backend: str
    arch: str
    specialization_fingerprint: str
    capabilities: frozenset[str]
    return_success: int


@dataclass(frozen=True)
class FlyDSLAOTArtifact:
    object_file_path: str
    header_file_path: str
    manifest_file_path: str
    object_sha256: str
    manifest: FlyDSLAOTManifest

    @property
    def symbol(self) -> str:
        return self.manifest.symbol

    @property
    def module_init_symbol(self) -> str:
        return self.manifest.module_init_symbol

    @property
    def module_load_symbol(self) -> str:
        return self.manifest.module_load_symbol

    @property
    def module_unload_symbol(self) -> str:
        return self.manifest.module_unload_symbol

    @property
    def abi(self) -> tuple[FlyDSLAbiSlot, ...]:
        return self.manifest.abi

    @property
    def linker_flags(self) -> tuple[str, ...]:
        return self.manifest.linker_flags

    @property
    def arch(self) -> str:
        return self.manifest.arch

    @property
    def return_success(self) -> int:
        return self.manifest.return_success


@dataclass(frozen=True)
class FlyDSLAOTBinding:
    values: Mapping[str, Any]

    def resolve(self, slot: FlyDSLAbiSlot) -> Any:
        if slot.kind == "stream":
            return None
        if slot.arg_name is None:
            raise RuntimeError(f"FlyDSL ABI slot has no parameter name: {slot!r}")
        if slot.arg_name not in self.values:
            raise RuntimeError(
                f"FlyDSL AOT has no binding for parameter {slot.arg_name!r}"
            )
        value = self.values[slot.arg_name]
        if slot.field is not None:
            for component in slot.field.split("."):
                value = (
                    value[component]
                    if isinstance(value, dict)
                    else getattr(value, component)
                )
        return value


@dataclass(frozen=True)
class FlyDSLRuntimeScalar:
    cpp_expr: str


@dataclass(frozen=True)
class FlyDSLTensorBinding:
    node: Any
    sizes: tuple[Any, ...]
    strides: tuple[Any, ...]


def _flattened_tensor(node) -> FlyDSLTensorBinding:
    numel = 1
    for size in node.get_size():
        numel *= size
    return FlyDSLTensorBinding(node=node, sizes=(numel,), strides=(1,))


def bind_template_parameters(
    artifact: FlyDSLAOTArtifact,
    template_name: str,
    output,
    inputs: Mapping[str, Any],
    template_kwargs: Mapping[str, Any],
) -> FlyDSLAOTBinding:
    """Bind a PyTorch-owned template to a FlyDSL launcher by parameter name."""
    values = dict(inputs)
    values.update(
        {
            "out": output,
            "output": output,
            "grad_query": output,
            "a": inputs.get("mat1"),
            "b": inputs.get("mat2"),
        }
    )

    if template_name == "mm_flydsl":
        if "scale_a" in inputs:
            values["scale_a"] = inputs["scale_a"]
        if "scale_b" in inputs:
            values["scale_b"] = inputs["scale_b"]
        values["bias"] = inputs.get("bias", output)
    elif template_name == "grouped_gemm_flydsl":
        mat2 = inputs["mat2"]
        values.update(
            {
                "offs": inputs["offs"],
                "group_count": mat2.get_size()[0],
                "k": mat2.get_size()[1],
                "n": mat2.get_size()[2],
                "grid_size": FlyDSLRuntimeScalar(f"{artifact.symbol}__num_cus()"),
            }
        )
    elif template_name == "mxfp8_grouped_gemm_flydsl":
        values.update(
            {
                "A": _flattened_tensor(inputs["mat1"]),
                "B": _flattened_tensor(inputs["mat2"]),
                "OUT": _flattened_tensor(output),
                "A_scale": _flattened_tensor(inputs["scale_a"]),
                "B_scale": _flattened_tensor(inputs["scale_b"]),
                "OFFS": inputs["offs"],
                "n_blocks": FlyDSLRuntimeScalar(f"{artifact.symbol}__num_cus()"),
                "n_c_tiles": (
                    template_kwargs["GEMM_N"] + template_kwargs["BLOCK_C"] - 1
                )
                // template_kwargs["BLOCK_C"],
                "out_m": output.get_size()[0],
                "out_n": output.get_size()[1],
            }
        )

    kv_num_blocks = inputs.get("kv_num_blocks")
    if kv_num_blocks is not None:
        for index in range(4):
            values.setdefault(f"mask_buffer_{index}", kv_num_blocks)

    # Preserve explicit aliases declared by templates without teaching the ABI
    # emitter workload-specific positional conventions.
    for alias, target in template_kwargs.get("AOT_ALIASES", {}).items():
        if target == "output":
            values[alias] = output
        else:
            values[alias] = inputs[target]

    return FlyDSLAOTBinding(values)


def _parse_manifest(data: Mapping[str, Any]) -> FlyDSLAOTManifest:
    schema = data.get("schema_version")
    if (
        not isinstance(schema, dict)
        or int(schema.get("major", -1)) != _AOT_SCHEMA_MAJOR
    ):
        raise RuntimeError(
            "PyTorch FlyDSL AOT requires manifest schema major "
            f"{_AOT_SCHEMA_MAJOR}, got {schema!r}"
        )
    minimum = data.get("min_consumer_schema")
    if not isinstance(minimum, dict):
        raise RuntimeError("FlyDSL AOT manifest has no minimum consumer schema")
    minimum_version = (int(minimum["major"]), int(minimum["minor"]))
    if minimum_version > (_AOT_SCHEMA_MAJOR, _AOT_SCHEMA_MINOR):
        raise RuntimeError(
            "FlyDSL AOT manifest requires consumer schema "
            f"{minimum_version[0]}.{minimum_version[1]}"
        )

    producer = data.get("producer")
    if not isinstance(producer, dict):
        raise RuntimeError("FlyDSL AOT manifest has no producer")
    backend = producer.get("backend")
    if backend != "rocm":
        raise RuntimeError(
            f"PyTorch FlyDSL AOT requires the ROCm backend, got {backend!r}"
        )
    arch = producer.get("arch")
    if arch != _SUPPORTED_ARCH:
        raise RuntimeError(
            f"PyTorch FlyDSL AOT currently supports {_SUPPORTED_ARCH}, got {arch!r}"
        )

    symbols = data.get("symbols")
    entry = data.get("entry")
    link = data.get("link")
    if not all(isinstance(value, dict) for value in (symbols, entry, link)):
        raise RuntimeError("FlyDSL AOT manifest is missing symbols, entry, or link")
    if entry.get("calling_convention") != "packed_void_pp":
        raise RuntimeError(
            "unsupported FlyDSL AOT calling convention: "
            f"{entry.get('calling_convention')!r}"
        )
    lifecycle = symbols.get("lifecycle")
    if not isinstance(lifecycle, dict):
        raise RuntimeError("FlyDSL AOT manifest has no lifecycle symbols")

    return_abi = entry.get("return_abi")
    if not isinstance(return_abi, dict) or (
        return_abi.get("kind"),
        return_abi.get("ctype"),
    ) != ("status", "int32"):
        raise RuntimeError(f"unsupported FlyDSL AOT return ABI: {return_abi!r}")

    runtime_libraries = tuple(
        str(library["link_name"]) for library in link.get("libraries", ())
    )
    if "libamdhip64.so" not in runtime_libraries:
        raise RuntimeError(
            "FlyDSL AOT manifest does not link the HIP runtime: "
            f"{runtime_libraries!r}"
        )
    linker_flags = tuple(str(option) for option in link.get("options", ()))
    allowed_flags = {"-lamdhip64", "-pthread", "-ldl"}
    if not set(linker_flags).issubset(allowed_flags):
        raise RuntimeError(f"unsupported FlyDSL AOT linker options: {linker_flags!r}")

    abi = entry.get("abi")
    if not isinstance(abi, list) or not all(isinstance(slot, dict) for slot in abi):
        raise RuntimeError("FlyDSL AOT manifest must contain an ABI slot list")
    parsed_slots = []
    for raw_slot in abi:
        if raw_slot.get("kind") not in {
            "tensor_data",
            "tensor_layout",
            "pointer",
            "scalar",
            "stream",
        }:
            raise NotImplementedError(f"unsupported FlyDSL AOT ABI slot: {raw_slot!r}")
        parsed_slots.append(
            FlyDSLAbiSlot(
                arg_index=(
                    None
                    if raw_slot.get("arg_index") is None
                    else int(raw_slot["arg_index"])
                ),
                arg_name=(
                    None
                    if raw_slot.get("arg_name") is None
                    else str(raw_slot["arg_name"])
                ),
                kind=str(raw_slot["kind"]),
                ctype=str(raw_slot["ctype"]),
                size=int(raw_slot["size"]),
                alignment=int(raw_slot["alignment"]),
                encoding=(
                    str(raw_slot["encoding"]) if "encoding" in raw_slot else None
                ),
                field=str(raw_slot["field"]) if "field" in raw_slot else None,
                layout_entries=tuple(
                    FlyDSLLayoutEntry(
                        kind=str(layout["kind"]),
                        dim=int(layout["dim"]),
                        offset=int(layout["offset"]),
                        bits=int(layout["bits"]),
                        signed=bool(layout.get("signed", True)),
                    )
                    for layout in raw_slot.get("layout_entries", ())
                ),
            )
        )

    specialization = data.get("specialization", {})
    return FlyDSLAOTManifest(
        symbol=str(symbols["entry"]),
        metadata_symbol=str(symbols["metadata"]),
        module_init_symbol=str(lifecycle["init_symbol"]),
        module_load_symbol=str(lifecycle["load_symbol"]),
        module_unload_symbol=str(lifecycle["unload_symbol"]),
        abi=tuple(parsed_slots),
        linker_flags=linker_flags,
        runtime_libraries=runtime_libraries,
        backend=str(backend),
        arch=str(arch),
        specialization_fingerprint=str(specialization["fingerprint"]),
        capabilities=frozenset(str(item) for item in data.get("capabilities", ())),
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


def _artifact_from_export_result(result) -> FlyDSLAOTArtifact:
    manifest_data = result.manifest.to_dict()
    manifest = _parse_manifest(manifest_data)
    object_path = Path(result.object_path)
    header_path = Path(result.header_path)
    manifest_path = Path(result.manifest_path)
    if not all(path.is_file() for path in (object_path, header_path, manifest_path)):
        raise RuntimeError(
            "FlyDSL export_aot() did not publish all declared artifact files"
        )
    object_sha256 = hashlib.sha256(object_path.read_bytes()).hexdigest()
    if object_sha256 != result.object_sha256:
        raise RuntimeError("FlyDSL AOT object hash does not match export result")
    header_sha256 = hashlib.sha256(header_path.read_bytes()).hexdigest()
    if header_sha256 != result.header_sha256:
        raise RuntimeError("FlyDSL AOT header hash does not match export result")
    sidecar = json.loads(manifest_path.read_text())
    if sidecar.get("manifest") != manifest_data:
        raise RuntimeError("FlyDSL AOT sidecar manifest does not match export result")
    if sidecar.get("files", {}).get("object", {}).get("sha256") != object_sha256:
        raise RuntimeError("FlyDSL AOT sidecar object hash is stale")
    if sidecar.get("files", {}).get("header", {}).get("sha256") != header_sha256:
        raise RuntimeError("FlyDSL AOT sidecar header hash is stale")
    return FlyDSLAOTArtifact(
        object_file_path=str(object_path),
        header_file_path=str(header_path),
        manifest_file_path=str(manifest_path),
        object_sha256=object_sha256,
        manifest=manifest,
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
    compiled_functions = []

    def capture_compile(compiler, jit_func, compile_args, compile_kwargs):
        compiled = compiler(
            jit_func,
            *compile_args,
            **compile_kwargs,
        )
        if compiled is None:
            raise RuntimeError("FlyDSL AOT compilation did not return an artifact")
        compiled_functions.append(compiled)
        return compiled

    with flydsl_compile_lock():
        with (
            capture_flydsl_compiles(capture_compile),
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

    if not compiled_functions:
        raise RuntimeError("FlyDSL AOT template did not compile any specialization")
    template_name = precompile_metadata.get("template_name")
    if len(compiled_functions) > 1:
        if template_name != "mxfp8_grouped_gemm_flydsl":
            raise RuntimeError(
                "expected one FlyDSL AOT specialization, got "
                f"{len(compiled_functions)}"
            )
        # The last compile-only variant uses persistent traversal. It is
        # correct for every grid size and avoids a runtime artifact selector.
        compiled_functions = [compiled_functions[-1]]
        for name, shape in precompile_metadata["precompile_shapes"].items():
            numel = 1
            for size in shape:
                numel *= size
            if numel > 2**31 - 1:
                raise NotImplementedError(
                    "MXFP8 grouped AOT currently requires every operand to fit "
                    f"the int32 layout ABI; {name} has {numel} elements"
                )

    output_dir = Path(cache_dir()) / "flydsl_aot"
    os.makedirs(output_dir, exist_ok=True)
    file_name = f"{kernel_name}_aot"
    symbol = f"flydsl_aot_{kernel_name}"
    result = compiled_functions[0].export_aot(
        str(output_dir),
        file_name,
        symbol,
    )
    return _artifact_from_export_result(result)


def compile_portable_program(
    program_json: str,
    call_spec_json: str,
    example_args: tuple[Any, ...],
) -> FlyDSLAOTArtifact:
    """Compile a portable user launcher specialization to an AOT artifact."""
    import flydsl.compiler as flyc

    from torch._higher_order_ops.flydsl_program import (
        FlyDSLCallSpec,
        FlyDSLProgramSpec,
    )

    program = FlyDSLProgramSpec.from_json(program_json)
    launcher = program.resolve()
    constants = FlyDSLCallSpec.from_json(call_spec_json).decode()
    values = list(example_args)
    for index, value in constants.items():
        values[index] = value
    signature = inspect.signature(launcher.func)
    positional = []
    keyword = {}
    for parameter, value in zip(signature.parameters.values(), values):
        if parameter.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            positional.append(value)
        elif parameter.kind is inspect.Parameter.KEYWORD_ONLY:
            keyword[parameter.name] = value
        else:
            raise TypeError("portable FlyDSL launchers cannot be variadic")

    with flydsl_compile_lock(), temporary_env({"FLYDSL_GPU_ARCH": _SUPPORTED_ARCH}):
        compiled = flyc.compile(launcher, *positional, **keyword)
    if compiled is None:
        raise RuntimeError("FlyDSL portable AOT compilation returned no artifact")

    output_dir = Path(cache_dir()) / "flydsl_aot"
    os.makedirs(output_dir, exist_ok=True)
    argument_signature = []
    for value in example_args:
        if hasattr(value, "shape") and hasattr(value, "stride"):
            argument_signature.append(
                (
                    "tensor",
                    str(value.dtype),
                    tuple(value.shape),
                    tuple(value.stride()),
                )
            )
        else:
            argument_signature.append((type(value).__qualname__, repr(value)))
    argument_digest = hashlib.sha256(
        json.dumps(argument_signature, sort_keys=True).encode("utf-8")
    ).hexdigest()
    call_digest = FlyDSLCallSpec.from_json(call_spec_json).digest
    stem = (
        f"portable_{program.digest[:16]}_{call_digest[:12]}_" f"{argument_digest[:12]}"
    )
    result = compiled.export_aot(
        str(output_dir),
        stem,
        f"flydsl_aot_{stem}",
    )
    return _artifact_from_export_result(result)


@dataclass(frozen=True)
class DeferredFlyDSLCallWrapper:
    artifact: FlyDSLAOTArtifact

    @property
    def ensure_loaded_symbol(self) -> str:
        return f"{self.artifact.symbol}__ensure_loaded"

    def generate(self, wrapper) -> None:
        artifact = self.artifact
        wrapper.include_extra_header("cstdint")
        wrapper.include_extra_header("mutex")
        wrapper.include_extra_header("stdexcept")
        wrapper.prefix.splice(f"""
extern "C" int32_t {artifact.symbol}(void**);
extern "C" int32_t {artifact.module_init_symbol}(void);
extern "C" int32_t {artifact.module_load_symbol}(int32_t);

static std::once_flag {artifact.symbol}__init_once;
static int32_t {artifact.symbol}__init_status = {artifact.return_success};

static void {self.ensure_loaded_symbol}() {{
  std::call_once({artifact.symbol}__init_once, [] {{
    {artifact.symbol}__init_status = {artifact.module_init_symbol}();
  }});
  if ({artifact.symbol}__init_status != {artifact.return_success}) {{
    throw std::runtime_error("FlyDSL AOT module initialization failed");
  }}
  int32_t load_status = {artifact.module_load_symbol}(-1);
  if (load_status != {artifact.return_success}) {{
    throw std::runtime_error("FlyDSL AOT module load failed");
  }}
}}

static int32_t {artifact.symbol}__num_cus() {{
  int device = 0;
  hipDeviceProp_t properties;
  if (hipGetDevice(&device) != hipSuccess ||
      hipGetDeviceProperties(&properties, device) != hipSuccess) {{
    throw std::runtime_error("FlyDSL AOT could not query HIP device properties");
  }}
  return static_cast<int32_t>(properties.multiProcessorCount);
}}
""")


def register_aot_artifact(wrapper, kernel_name: str, artifact: FlyDSLAOTArtifact):
    cache_key = (
        artifact.manifest.specialization_fingerprint,
        artifact.arch,
        artifact.object_sha256,
    )
    artifact = wrapper.flydsl_aot_artifact_cache.setdefault(cache_key, artifact)
    wrapper.flydsl_aot_artifacts[kernel_name] = artifact
    wrapper.deferred_aot_call_wrappers[kernel_name] = DeferredFlyDSLCallWrapper(
        artifact
    )

    wrapper.aot_kernel_objects.add(artifact.object_file_path)
    wrapper.external_kernel_libs.update(artifact.linker_flags)
    wrapper.aot_link_libraries.update(artifact.manifest.runtime_libraries)
    wrapper.aot_link_options.update(artifact.linker_flags)
    for path in (artifact.header_file_path, artifact.manifest_file_path):
        if path not in wrapper.additional_files:
            wrapper.additional_files.append(path)


def _aot_artifact(wrapper, kernel_name: str) -> FlyDSLAOTArtifact:
    artifacts = wrapper.flydsl_aot_artifacts
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
    binding: FlyDSLAOTBinding,
    *,
    device,
) -> None:
    """Pack and invoke one exported FlyDSL template on Inductor's stream."""
    from torch._inductor.ir import IRNode
    from torch._inductor.stream_utils import DEFAULT_STREAM_IDX, get_stream_name

    artifact = _aot_artifact(wrapper, kernel_name)
    call_index = getattr(wrapper, "_flydsl_aot_call_counter", 0)
    wrapper._flydsl_aot_call_counter = call_index + 1
    deferred = wrapper.deferred_aot_call_wrappers[kernel_name]
    wrapper.writeline(f"{deferred.ensure_loaded_symbol}();")

    current_stream_idx = V.graph.scheduler.current_stream_idx
    stream = (
        get_stream_name(current_stream_idx)
        if current_stream_idx is not None and current_stream_idx != DEFAULT_STREAM_IDX
        else wrapper.write_get_raw_stream(device.index, V.graph.name)
    )

    if any(slot.encoding == "float16_bits" for slot in artifact.abi):
        wrapper.header.writeline("#include <c10/util/Half.h>")
    if any(slot.encoding == "bfloat16_bits" for slot in artifact.abi):
        wrapper.header.writeline("#include <c10/util/BFloat16.h>")

    packed_entries = []
    for slot_index, slot in enumerate(artifact.abi):
        slot_name = f"flydsl_{call_index}_{slot_index}"
        kind = slot.kind
        arg = binding.resolve(slot)
        tensor_node = arg.node if isinstance(arg, FlyDSLTensorBinding) else arg

        if kind == "tensor_data":
            if not isinstance(tensor_node, IRNode):
                raise AssertionError("FlyDSL tensor ABI slot requires an IR node")
            wrapper.writeline(
                f"void* {slot_name} = " f"{tensor_node.codegen_reference()}.data_ptr();"
            )
            packed_entries.append(f"&{slot_name}")
        elif kind == "tensor_layout":
            if not isinstance(tensor_node, IRNode):
                raise AssertionError("FlyDSL layout ABI slot requires an IR node")
            wrapper.writeline(
                f"alignas({slot.alignment}) unsigned char "
                f"{slot_name}[{slot.size}] = {{}};"
            )
            occupied_bytes = set()
            for layout_entry in slot.layout_entries:
                if layout_entry.bits not in (8, 16, 32, 64):
                    raise NotImplementedError(
                        f"unsupported FlyDSL layout width: {layout_entry.bits}"
                    )
                value_name = f"{slot_name}_{layout_entry.kind}_{layout_entry.dim}"
                if layout_entry.kind == "shape":
                    raw_value = (
                        arg.sizes[layout_entry.dim]
                        if isinstance(arg, FlyDSLTensorBinding)
                        else tensor_node.get_size()[layout_entry.dim]
                    )
                elif layout_entry.kind == "stride":
                    raw_value = (
                        arg.strides[layout_entry.dim]
                        if isinstance(arg, FlyDSLTensorBinding)
                        else tensor_node.get_stride()[layout_entry.dim]
                    )
                else:
                    raise NotImplementedError(
                        f"unsupported FlyDSL layout entry: {layout_entry!r}"
                    )
                value = cexpr(V.graph.sizevars.simplify(raw_value))
                value_type = (
                    f"{'int' if layout_entry.signed else 'uint'}"
                    f"{layout_entry.bits}_t"
                )
                wrapper.writeline(
                    f"{value_type} {value_name} = "
                    f"static_cast<{value_type}>({value});"
                )
                wrapper.writeline(
                    f"std::memcpy({slot_name} + {layout_entry.offset}, "
                    f"&{value_name}, "
                    f"sizeof({value_name}));"
                )
                width_bytes = layout_entry.bits // 8
                occupied_bytes.update(
                    range(
                        layout_entry.offset,
                        layout_entry.offset + width_bytes,
                    )
                )
            if occupied_bytes and max(occupied_bytes) >= slot.size:
                raise AssertionError(
                    f"FlyDSL layout ABI writes past its {slot.size}-byte slot"
                )
            packed_entries.append(slot_name)
        elif kind == "scalar":
            if arg is None:
                raise NotImplementedError(
                    f"FlyDSL scalar ABI slot has no runtime argument: {slot!r}"
                )
            encoding = slot.encoding
            value = (
                arg.cpp_expr
                if isinstance(arg, FlyDSLRuntimeScalar)
                else cexpr(V.graph.sizevars.simplify(arg))
            )
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
                scalar_type = scalar_types[slot.ctype]
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
                f"{tensor_node.codegen_reference()}.data_ptr()"
                if isinstance(tensor_node, IRNode)
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
        f"flydsl_status_{call_index}_call",
        artifact.return_success,
    )
