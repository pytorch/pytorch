from __future__ import annotations

import hashlib
import importlib.metadata
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from torch._inductor.codegen.cpp_utils import cexpr
from torch._inductor.codegen.external_aot import (
    export_external_aot_files,
    ExternalAOTArtifact,
    ExternalAOTBuildSpec,
    ExternalAOTTensorArg,
    publish_external_aot_artifact,
)
from torch._inductor.runtime.cache_dir_utils import cache_dir
from torch._inductor.runtime.flydsl_cache import (
    capture_flydsl_compiles,
    flydsl_compile_lock,
    temporary_env,
)
from torch._inductor.stream_utils import DEFAULT_STREAM_IDX, get_stream_name
from torch._inductor.virtualized import V


@dataclass(frozen=True)
class FlyDSLAOTExportRequest:
    compiled: Any
    prefix: str


class FlyDSLAOTBackend:
    """Export FlyDSL objects and bridge their generated C ABI to AOTI."""

    kind = "flydsl"

    _LAYOUT = re.compile(
        r"typedef\s+struct\s+__attribute__\(\(packed\)\)\s*\{"
        r"(?P<body>[^{}]*)\}\s*(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*;",
        re.DOTALL,
    )
    _ARRAY = re.compile(
        r"(?P<ctype>u?int(?:8|16|32|64)_t)\s+"
        r"(?P<field>shape|stride)\[(?P<count>[0-9]+)\];"
    )
    _PARAM_NAME = re.compile(r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*$")

    def export(
        self,
        request: FlyDSLAOTExportRequest,
        spec: ExternalAOTBuildSpec,
        output_dir: str,
    ) -> ExternalAOTArtifact:
        output = Path(output_dir)
        object_path, header_path = export_external_aot_files(
            output,
            request.prefix,
            lambda directory: request.compiled.export_to_c(
                file_path=str(directory),
                file_name=request.prefix,
                function_prefix=request.prefix,
            ),
        )
        artifact = ExternalAOTArtifact.from_files(
            spec=spec,
            object_path=object_path,
            header_path=header_path,
            link_args=("-l:libamdhip64.so", "-pthread", "-ldl"),
            payload={
                "prefix": request.prefix,
                "calling_convention": "typed_header",
                "tensor_args": [
                    {
                        "name": arg.name,
                        "source": arg.source,
                        "dynamic_sizes": list(arg.dynamic_sizes),
                        "dynamic_strides": list(arg.dynamic_strides),
                    }
                    for arg in spec.tensor_args
                ],
            },
            runtime_versions={"flydsl": _distribution_version("flydsl")},
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
            for match in self._LAYOUT.finditer(header)
        }
        prefix = str(artifact.payload["prefix"])
        expected_params = []
        for arg in spec.tensor_args:
            expected_params.append(f"{arg.name}_data")
            if not arg.dynamic_sizes and not arg.dynamic_strides:
                continue
            type_name = f"{prefix}_{arg.name}_layout_t"
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
            _check_count(
                artifact.header_path,
                arg.name,
                "shape",
                arrays,
                len(arg.dynamic_sizes),
            )
            _check_count(
                artifact.header_path,
                arg.name,
                "stride",
                arrays,
                len(arg.dynamic_strides),
            )
            expected_params.append(f"{arg.name}_layout")
        expected_params.append("stream")
        call = re.search(
            rf"\b{re.escape(prefix)}_call_auto\s*\((?P<params>.*?)\)" r"\s*(?:\{|;)",
            header,
            re.DOTALL,
        )
        if call is None:
            raise RuntimeError(
                f"{artifact.header_path}: missing {prefix}_call_auto declaration"
            )
        actual_params = []
        for parameter in call.group("params").split(","):
            match = self._PARAM_NAME.search(parameter.strip())
            if match is None:
                raise RuntimeError(
                    f"{artifact.header_path}: cannot parse call_auto parameter "
                    f"{parameter!r}"
                )
            actual_params.append(match.group("name"))
        if actual_params != expected_params:
            raise RuntimeError(
                f"{artifact.header_path}: call_auto parameters {actual_params} "
                f"do not match builder order {expected_params}"
            )

    def make_deferred_call(self, artifact: ExternalAOTArtifact) -> Any:
        return FlyDSLDeferredCall(artifact)

    def emit_call(self, wrapper, artifact, binding, *, device) -> None:
        prefix = str(artifact.payload["prefix"])
        call_index = getattr(wrapper, "_flydsl_header_call_index", 0)
        wrapper._flydsl_header_call_index = call_index + 1
        args = []
        for tensor in artifact.payload["tensor_args"]:
            node = binding[tensor["source"]]
            args.append(f"{node.codegen_reference()}.data_ptr()")
            if tensor["dynamic_sizes"] or tensor["dynamic_strides"]:
                name = f"{prefix}_layout_{call_index}_{tensor['name']}"
                wrapper.writeline(f"{prefix}_{tensor['name']}_layout_t {name};")
                for slot, dim in enumerate(tensor["dynamic_sizes"]):
                    extent = node.get_size()[dim]
                    value = cexpr(V.graph.sizevars.simplify(extent))
                    wrapper.writeline(f"{name}.shape[{slot}] = {value};")
                for slot, dim in enumerate(tensor["dynamic_strides"]):
                    value = cexpr(V.graph.sizevars.simplify(node.get_stride()[dim]))
                    wrapper.writeline(f"{name}.stride[{slot}] = {value};")
                args.append(f"&{name}")
        current_stream_idx = V.graph.scheduler.current_stream_idx
        stream = (
            get_stream_name(current_stream_idx)
            if current_stream_idx is not None
            and current_stream_idx != DEFAULT_STREAM_IDX
            else wrapper.write_get_raw_stream(device.index, V.graph.name)
        )
        args.append(stream)
        status = f"flydsl_header_status_{call_index}"
        wrapper.writeline(f"int32_t {status} = {prefix}_call_auto({', '.join(args)});")
        wrapper.writeline(
            f'if ({status} != 0) throw std::runtime_error("FlyDSL AOT launch failed");'
        )


@dataclass(frozen=True)
class FlyDSLDeferredCall:
    artifact: ExternalAOTArtifact

    def generate(self, wrapper) -> None:
        wrapper.include_extra_header("stdexcept")
        wrapper.prefix.splice(Path(self.artifact.header_path).read_text())


def _check_count(
    header_path: str,
    arg_name: str,
    field: str,
    arrays: dict[str, tuple[str, int]],
    expected: int,
) -> None:
    declared = arrays.get(field)
    if expected == 0:
        if declared is not None and declared[1] != 0:
            raise RuntimeError(
                f"{header_path}: {arg_name} unexpectedly declares {field}"
            )
        return
    if declared is None or declared[1] != expected:
        raise RuntimeError(
            f"{header_path}: {arg_name} {field} count does not match "
            f"builder metadata ({expected})"
        )


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


_FLYDSL_AOT_BACKEND = FlyDSLAOTBackend()


def compile_flydsl_external_source(
    kernel_name: str,
    source_code: str,
    precompile_metadata: dict[str, Any],
    *,
    template_name: str,
) -> tuple[ExternalAOTArtifact, ExternalAOTBuildSpec]:
    import torch
    from torch._inductor.codecache import PyCodeCache
    from torch._subclasses.fake_tensor import FakeTensorMode

    metadata = dict(precompile_metadata)
    metadata["template_name"] = template_name
    key, source_path = PyCodeCache.write(source_code)
    module = PyCodeCache.load_by_key_path(key, source_path)
    main = getattr(module, f"{kernel_name}_main")
    compiled_functions = []

    def observe(compiler, jit_func, args, kwargs):
        compiled = compiler(jit_func, *args, **kwargs)
        if compiled is None:
            raise RuntimeError("FlyDSL compile returned no artifact")
        compiled_functions.append(compiled)
        return compiled

    names = tuple(metadata["tensor_names"])
    with (
        flydsl_compile_lock(),
        capture_flydsl_compiles(observe),
        temporary_env({"FLYDSL_GPU_ARCH": "gfx950"}),
        FakeTensorMode(),
    ):
        tensors = tuple(
            torch.empty_strided(
                tuple(metadata["precompile_shapes"][name]),
                tuple(metadata["precompile_strides"][name]),
                dtype=getattr(torch, metadata["precompile_dtypes"][name]),
                device="cpu",
            )
            for name in (*names, "output")
        )
        main(*tensors[:-1], tensors[-1], stream=0)
    if not compiled_functions:
        raise RuntimeError("FlyDSL template compiled no AOT specialization")
    if len(compiled_functions) != 1:
        raise RuntimeError(
            f"FlyDSL dense MM compiled {len(compiled_functions)} specializations"
        )

    spec = _build_spec(kernel_name, source_code, metadata)
    output_dir = os.path.join(cache_dir(), "external_aot", "flydsl")
    backend = _FLYDSL_AOT_BACKEND
    artifact = backend.export(
        FlyDSLAOTExportRequest(
            compiled=compiled_functions[0],
            prefix=f"flydsl_{spec.digest[:24]}",
        ),
        spec,
        output_dir,
    )
    return artifact, spec


def register_flydsl_external_artifact(
    wrapper,
    kernel_name: str,
    artifact: ExternalAOTArtifact,
) -> ExternalAOTArtifact:
    return publish_external_aot_artifact(
        wrapper,
        kernel_name,
        artifact,
        _FLYDSL_AOT_BACKEND.make_deferred_call(artifact),
    )


def emit_flydsl_external_call(wrapper, artifact, binding, *, device) -> None:
    _FLYDSL_AOT_BACKEND.emit_call(
        wrapper,
        artifact,
        binding,
        device=device,
    )


def _build_spec(
    kernel_name: str,
    source_code: str,
    metadata: dict[str, Any],
) -> ExternalAOTBuildSpec:
    template = str(metadata.get("template_name", ""))
    if template != "mm_flydsl" or "scale_a" in metadata["tensor_names"]:
        raise NotImplementedError("FlyDSL AOT initially supports dense MM only")
    shapes = metadata["precompile_shapes"]
    strides = metadata["precompile_strides"]
    input_names = tuple(metadata["tensor_names"])
    source_to_abi = {
        "output": "out",
        "mat1": "a",
        "mat2": "b",
    }
    ordered_sources = ("output", *input_names)
    tensor_args = []
    arg_pairs = tuple(
        (source_to_abi.get(source, source), source) for source in ordered_sources
    )
    for name, source in arg_pairs:
        shape = tuple(shapes[source])
        stride = tuple(strides[source])
        tensor_args.append(
            ExternalAOTTensorArg(
                name=name,
                source=source,
                dtype=str(metadata["precompile_dtypes"][source]),
                dynamic_sizes=tuple(range(len(shape))),
                dynamic_strides=tuple(
                    index for index, value in enumerate(stride) if value != 1
                ),
            )
        )
    return ExternalAOTBuildSpec(
        kind="flydsl",
        kernel_name=kernel_name,
        target="gfx950",
        source_hash=hashlib.sha256(source_code.encode("utf-8")).hexdigest(),
        tensor_args=tuple(tensor_args),
        options=(f"flydsl={_distribution_version('flydsl')}",),
    )
