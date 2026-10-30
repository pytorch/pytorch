from __future__ import annotations

import importlib.metadata
import hashlib
import platform
import re
import shlex
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from torch._inductor.codegen.external_aot import (
    ExternalAOTArtifact,
    ExternalAOTBackend,
    ExternalAOTBuildSpec,
    ExternalAOTTensorArg,
    get_external_aot_backend,
    publish_external_aot_artifact,
    register_external_aot_backend,
)
from torch._inductor.runtime.cache_dir_utils import cache_dir

_compile_lock = threading.RLock()


@dataclass(frozen=True)
class CuteDSLAOTExportRequest:
    compiled: Any
    prefix: str


class CuteDSLAOTBackend(ExternalAOTBackend):
    kind = "cutedsl"

    _STRUCT = re.compile(
        r"typedef\s+struct\s*\{(?P<body>[^{}]*)\}\s*"
        r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*;",
        re.DOTALL,
    )
    _ARRAY = re.compile(
        r"(?P<ctype>int32_t|int64_t|std::int64_t)\s+"
        r"(?P<field>dynamic_shapes|dynamic_strides)"
        r"\s*\[\s*(?P<count>[0-9]+)\s*\]\s*;"
    )

    def export(
        self,
        request: CuteDSLAOTExportRequest,
        spec: ExternalAOTBuildSpec,
        output_dir: str,
    ) -> ExternalAOTArtifact:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        object_path = output / f"{request.prefix}.o"
        header_path = output / f"{request.prefix}.h"
        request.compiled.export_to_c(
            file_path=str(output),
            file_name=request.prefix,
            function_prefix=request.prefix,
        )
        if not object_path.is_file() or not header_path.is_file():
            raise RuntimeError(
                "CuteDSL export_to_c did not produce a C object and header"
            )
        artifact = ExternalAOTArtifact.from_files(
            spec=spec,
            object_path=object_path,
            header_path=header_path,
            link_libraries=(),
            link_options=(),
            payload={
                "prefix": request.prefix,
                "calling_convention": "typed_header",
                "tensor_args": [
                    {
                        "name": arg.name,
                        "source": arg.source,
                        "dynamic_sizes": list(arg.dynamic_sizes),
                        "dynamic_strides": list(arg.dynamic_strides),
                        "read_only": arg.read_only,
                    }
                    for arg in spec.tensor_args
                ],
                "scalar_args": [
                    {
                        "name": arg.name,
                        "ctype": arg.ctype,
                        "source": arg.source,
                    }
                    for arg in spec.scalar_args
                ],
            },
            runtime_versions={
                "nvidia-cutlass-dsl": _distribution_version("nvidia-cutlass-dsl"),
                "apache-tvm-ffi": _distribution_version("apache-tvm-ffi"),
            },
        )
        self.validate_abi(artifact, spec)
        return artifact

    def validate_abi(
        self,
        artifact: ExternalAOTArtifact,
        spec: ExternalAOTBuildSpec,
    ) -> None:
        header = Path(artifact.header_path).read_text(
            encoding="utf-8", errors="replace"
        )
        structs = {
            match.group("name"): match.group("body")
            for match in self._STRUCT.finditer(header)
        }
        prefix = str(artifact.payload["prefix"])
        for arg in spec.tensor_args:
            type_name = f"{prefix}_Tensor_{arg.name}_t"
            body = structs.get(type_name)
            if body is None:
                raise RuntimeError(f"{artifact.header_path}: missing {type_name}")
            arrays = {
                match.group("field"): (
                    match.group("ctype"),
                    int(match.group("count")),
                )
                for match in self._ARRAY.finditer(body)
            }
            _validate_slots(
                artifact.header_path,
                arg.name,
                "dynamic_shapes",
                arrays,
                len(arg.dynamic_sizes),
                shape=True,
            )
            _validate_slots(
                artifact.header_path,
                arg.name,
                "dynamic_strides",
                arrays,
                len(arg.dynamic_strides),
                shape=False,
            )

    def make_deferred_call(self, artifact: ExternalAOTArtifact) -> Any:
        return CuteDSLDeferredCall(artifact)

    def emit_call(self, wrapper, artifact, binding, *, device) -> None:
        from torch._inductor.stream_utils import DEFAULT_STREAM_IDX, get_stream_name
        from torch._inductor.virtualized import V

        current_stream_idx = V.graph.scheduler.current_stream_idx
        stream = (
            get_stream_name(current_stream_idx)
            if current_stream_idx is not None
            and current_stream_idx != DEFAULT_STREAM_IDX
            else wrapper.write_get_raw_stream(device.index, V.graph.name)
        )
        prefix = str(artifact.payload["prefix"])
        tensor_args = [
            binding[arg["source"]] for arg in artifact.payload["tensor_args"]
        ]
        scalar_args = [
            binding[arg["source"]] for arg in artifact.payload["scalar_args"]
        ]
        args = [
            *(arg.codegen_reference() for arg in tensor_args),
            *(str(arg) for arg in scalar_args),
            stream,
        ]
        wrapper.writeline(f"launch_{prefix}({', '.join(args)});")


@dataclass(frozen=True)
class CuteDSLDeferredCall:
    artifact: ExternalAOTArtifact

    def generate(self, wrapper) -> None:
        wrapper.include_extra_header("c10/util/CallOnce.h")
        wrapper.prefix.splice(Path(self.artifact.header_path).read_text())
        prefix = str(self.artifact.payload["prefix"])
        tensor_args = self.artifact.payload["tensor_args"]
        scalar_args = self.artifact.payload["scalar_args"]
        template_types = [f"typename T{index}" for index in range(len(tensor_args))]
        parameters = [
            f"const T{index}& {arg['name']}" for index, arg in enumerate(tensor_args)
        ]
        parameters.extend(f"{arg['ctype']} {arg['name']}" for arg in scalar_args)
        parameters.append("cudaStream_t stream")
        fills = []
        call_args = []
        for arg in tensor_args:
            name = arg["name"]
            fills.append(f"  {prefix}_Tensor_{name}_t {name}_s;")
            fills.append(f"  {name}_s.data = {name}.data_ptr();")
            for slot, dim in enumerate(arg["dynamic_sizes"]):
                fills.append(
                    f"  {name}_s.dynamic_shapes[{slot}] = "
                    f"static_cast<int32_t>({name}.size({dim}));"
                )
            for slot, dim in enumerate(arg["dynamic_strides"]):
                fills.append(
                    f"  {name}_s.dynamic_strides[{slot}] = " f"{name}.stride({dim});"
                )
            call_args.append(f"&{name}_s")
        call_args.extend(arg["name"] for arg in scalar_args)
        call_args.append("stream")
        fills_code = "\n".join(fills)
        wrapper.prefix.splice(f"""
{prefix}_Kernel_Module_t {prefix}_module;
c10::once_flag {prefix}_loaded;

template <{", ".join(template_types)}>
static void launch_{prefix}({", ".join(parameters)}) {{
  c10::call_once({prefix}_loaded, [] {{
    {prefix}_Kernel_Module_Load(&{prefix}_module);
  }});
{fills_code}
  int32_t rc = cute_dsl_{prefix}_wrapper(
      &{prefix}_module, {", ".join(call_args)});
  if (rc != 0) {{
    throw std::runtime_error("CuteDSL AOT launch failed");
  }}
}}
""")


def _validate_slots(
    header_path: str,
    arg_name: str,
    field: str,
    arrays: dict[str, tuple[str, int]],
    expected_count: int,
    *,
    shape: bool,
) -> None:
    declared = arrays.get(field)
    if expected_count == 0:
        if declared is not None and declared[1] != 0:
            raise RuntimeError(
                f"{header_path}: {arg_name} unexpectedly declares {field}"
            )
        return
    if declared is None or declared[1] != expected_count:
        raise RuntimeError(
            f"{header_path}: {arg_name} {field} count does not match "
            f"builder metadata ({expected_count})"
        )
    if not shape and declared[0] not in {
        "int64_t",
        "std::int64_t",
    }:
        raise RuntimeError(f"{header_path}: {arg_name} dynamic strides must be int64")


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


register_external_aot_backend(CuteDSLAOTBackend())


def export_cutedsl_compiled(
    wrapper,
    kernel_name: str,
    compiled: Any,
    spec: ExternalAOTBuildSpec,
    output_dir: str,
) -> ExternalAOTArtifact:
    backend = get_external_aot_backend("cutedsl")
    artifact = backend.export(
        CuteDSLAOTExportRequest(compiled=compiled, prefix=kernel_name),
        spec,
        output_dir,
    )
    deferred = backend.make_deferred_call(artifact)
    return publish_external_aot_artifact(
        wrapper,
        kernel_name,
        artifact,
        deferred,
    )


def compile_cutedsl_external_source(
    kernel_name: str,
    source_code: str,
    precompile_metadata: dict[str, Any],
) -> tuple[ExternalAOTArtifact, ExternalAOTBuildSpec]:
    import cutlass.cute as cute

    from torch._inductor.codecache import PyCodeCache

    key, source_path = PyCodeCache.write(source_code)
    module = PyCodeCache.load_by_key_path(key, source_path)
    precompile = getattr(module, f"{kernel_name}_precompile", None)
    if precompile is None:
        raise NotImplementedError(
            "CuteDSL AOT requires a generated precompile entry point"
        )
    compiled_functions = []
    original_compile = cute.compile

    def capture_compile(*args, **kwargs):
        options = [
            option
            for option in shlex.split(kwargs.get("options", ""))
            if option != "--enable-tvm-ffi"
        ]
        if not any(option.startswith("--host-target") for option in options):
            host_targets = {
                "x86_64": "llvm -mtriple=x86_64-unknown-linux-gnu -mcpu=x86-64",
                "aarch64": "llvm -mtriple=aarch64-unknown-linux-gnu",
            }
            machine = platform.machine()
            if machine not in host_targets:
                raise RuntimeError(
                    f"CuteDSL AOT has no generic host target for {machine!r}"
                )
            options.extend(["--host-target", host_targets[machine]])
        capability = precompile_metadata.get("device_capability")
        if capability is not None:
            options.extend(["--gpu-arch", f"sm_{capability[0]}{capability[1]}a"])
        kwargs["options"] = shlex.join(options)
        compiled = original_compile(*args, **kwargs)
        compiled_functions.append(compiled)
        return compiled

    with _compile_lock:
        cute.compile = capture_compile
        try:
            precompile(
                precompile_metadata["precompile_shapes"],
                precompile_metadata["precompile_strides"],
                precompile_metadata["precompile_dtypes"],
                device_index=precompile_metadata.get("device_index", 0),
                device_capability=precompile_metadata.get("device_capability"),
                hw_info=precompile_metadata.get("hw_info"),
            )
        finally:
            cute.compile = original_compile
    if len(compiled_functions) != 1:
        raise NotImplementedError(
            "CuteDSL AOT currently requires one compiled entry, got "
            f"{len(compiled_functions)}"
        )
    spec = _build_spec(kernel_name, source_code, precompile_metadata)
    output_dir = str(Path(cache_dir()) / "external_aot" / "cutedsl")
    backend = get_external_aot_backend("cutedsl")
    artifact = backend.export(
        CuteDSLAOTExportRequest(
            compiled=compiled_functions[0],
            prefix=f"cutedsl_{spec.digest[:24]}",
        ),
        spec,
        output_dir,
    )
    return artifact, spec


def _build_spec(
    kernel_name: str,
    source_code: str,
    metadata: dict[str, Any],
) -> ExternalAOTBuildSpec:
    shapes = metadata["precompile_shapes"]
    strides = metadata["precompile_strides"]
    sources = (*metadata["tensor_names"], "output")
    arg_names = metadata.get("aot_arg_names")
    if arg_names is None or len(arg_names) != len(sources):
        raise NotImplementedError(
            "CuteDSL AOT requires template AOT_ARG_NAMES matching "
            "the compiled function signature"
        )
    tensor_args = []
    for name, source in zip(arg_names, sources):
        shape = tuple(shapes[source])
        stride = tuple(strides[source])
        tensor_args.append(
            ExternalAOTTensorArg(
                name=name,
                source=source,
                dynamic_sizes=tuple(range(len(shape))),
                dynamic_strides=tuple(
                    index for index, value in enumerate(stride) if value != 1
                ),
                read_only=source != "output",
                mutated=source == "output",
            )
        )
    capability = metadata.get("device_capability")
    target = f"sm_{capability[0]}{capability[1]}" if capability is not None else "cuda"
    return ExternalAOTBuildSpec(
        kind="cutedsl",
        kernel_name=kernel_name,
        target=target,
        source_hash=hashlib.sha256(source_code.encode("utf-8")).hexdigest(),
        tensor_args=tuple(tensor_args),
    )
