from __future__ import annotations

import importlib.metadata
import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from torch._inductor.codegen.cpp_utils import cexpr
from torch._inductor.codegen.external_aot import (
    ExternalAOTArtifact,
    ExternalAOTBackend,
    ExternalAOTBuildSpec,
    ExternalAOTScalarArg,
    ExternalAOTTensorArg,
    get_external_aot_backend,
    publish_external_aot_artifact,
    register_external_aot_backend,
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


class FlyDSLAOTBackend(ExternalAOTBackend):
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

    def export(
        self,
        request: FlyDSLAOTExportRequest,
        spec: ExternalAOTBuildSpec,
        output_dir: str,
    ) -> ExternalAOTArtifact:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        request.compiled.export_to_c(
            file_path=str(output),
            file_name=request.prefix,
            function_prefix=request.prefix,
        )
        object_path = output / f"{request.prefix}.o"
        header_path = output / f"{request.prefix}.h"
        if not object_path.is_file() or not header_path.is_file():
            raise RuntimeError(
                "FlyDSL export_to_c did not produce a C object and header"
            )
        artifact = ExternalAOTArtifact.from_files(
            spec=spec,
            object_path=object_path,
            header_path=header_path,
            link_libraries=("libamdhip64.so",),
            link_options=("-lamdhip64", "-pthread", "-ldl"),
            payload={
                "prefix": request.prefix,
                "calling_convention": "typed_header",
                "tensor_args": [
                    {
                        "name": arg.name,
                        "source": arg.source,
                        "dynamic_sizes": list(arg.dynamic_sizes),
                        "dynamic_strides": list(arg.dynamic_strides),
                        "flattened": arg.flattened,
                    }
                    for arg in spec.tensor_args
                ],
                "scalar_args": [
                    {
                        "name": arg.name,
                        "source": arg.source,
                        "ctype": arg.ctype,
                    }
                    for arg in spec.scalar_args
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
        for arg in spec.tensor_args:
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
                    if tensor.get("flattened"):
                        extent = 1
                        for size in node.get_size():
                            extent *= size
                    else:
                        extent = node.get_size()[dim]
                    value = cexpr(V.graph.sizevars.simplify(extent))
                    wrapper.writeline(f"{name}.shape[{slot}] = {value};")
                for slot, dim in enumerate(tensor["dynamic_strides"]):
                    value = cexpr(V.graph.sizevars.simplify(node.get_stride()[dim]))
                    wrapper.writeline(f"{name}.stride[{slot}] = {value};")
                args.append(f"&{name}")
        for scalar in artifact.payload["scalar_args"]:
            args.append(cexpr(binding[scalar["source"]]))
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
            f"if ({status} != 0) throw std::runtime_error("
            f'"FlyDSL AOT launch failed");'
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


register_external_aot_backend(FlyDSLAOTBackend())


def compile_flydsl_external_source(
    kernel_name: str,
    source_code: str,
    precompile_metadata: dict[str, Any],
) -> tuple[ExternalAOTArtifact, ExternalAOTBuildSpec]:
    import torch

    from torch._inductor.codecache import PyCodeCache
    from torch._subclasses.fake_tensor import FakeTensorMode

    key, source_path = PyCodeCache.write(source_code)
    module = PyCodeCache.load_by_key_path(key, source_path)
    main = getattr(module, f"{kernel_name}_main")
    aot_arg_sources = getattr(module, f"{kernel_name}_aot_arg_sources", None)
    aot_arg_names = getattr(module, f"{kernel_name}_aot_arg_names", None)
    if aot_arg_sources is not None:
        precompile_metadata["aot_arg_sources"] = tuple(aot_arg_sources)
    if aot_arg_names is not None:
        precompile_metadata["aot_arg_names"] = tuple(aot_arg_names)
    compiled_functions = []

    def observe(compiler, jit_func, args, kwargs):
        compiled = compiler(jit_func, *args, **kwargs)
        if compiled is None:
            raise RuntimeError("FlyDSL compile returned no artifact")
        compiled_functions.append(compiled)
        return compiled

    names = tuple(precompile_metadata["tensor_names"])
    with (
        flydsl_compile_lock(),
        capture_flydsl_compiles(observe),
        temporary_env({"FLYDSL_GPU_ARCH": "gfx950"}),
        FakeTensorMode(),
    ):
        tensors = tuple(
            torch.empty_strided(
                tuple(precompile_metadata["precompile_shapes"][name]),
                tuple(precompile_metadata["precompile_strides"][name]),
                dtype=getattr(torch, precompile_metadata["precompile_dtypes"][name]),
                device="cpu",
            )
            for name in (*names, "output")
        )
        main(*tensors[:-1], tensors[-1], stream=0)
    if not compiled_functions:
        raise RuntimeError("FlyDSL template compiled no AOT specialization")
    if len(compiled_functions) > 1:
        if precompile_metadata.get("template_name") != ("mxfp8_grouped_gemm_flydsl"):
            raise RuntimeError(
                f"FlyDSL template compiled {len(compiled_functions)} specializations"
            )
        compiled_functions = [compiled_functions[-1]]

    spec = _build_spec(kernel_name, source_code, precompile_metadata)
    output_dir = os.path.join(cache_dir(), "external_aot", "flydsl")
    backend = get_external_aot_backend("flydsl")
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
    backend = get_external_aot_backend("flydsl")
    return publish_external_aot_artifact(
        wrapper,
        kernel_name,
        artifact,
        backend.make_deferred_call(artifact),
    )


def _build_spec(
    kernel_name: str,
    source_code: str,
    metadata: dict[str, Any],
) -> ExternalAOTBuildSpec:
    template = str(metadata.get("template_name", ""))
    shapes = metadata["precompile_shapes"]
    strides = metadata["precompile_strides"]
    input_names = tuple(metadata["tensor_names"])
    source_to_abi = {
        "output": "out",
        "mat1": "a",
        "mat2": "b",
    }
    ordered_sources = ("output", *input_names)
    explicit_sources = metadata.get("aot_arg_sources")
    explicit_names = metadata.get("aot_arg_names")
    if explicit_sources is not None:
        if explicit_names is None or len(explicit_names) != len(explicit_sources):
            raise RuntimeError(
                "FlyDSL AOT arg names and sources must have the same length"
            )
        ordered_sources = tuple(explicit_sources)
    if template == "mxfp8_grouped_gemm_flydsl" and explicit_sources is None:
        ordered_sources = (
            "mat1",
            "mat2",
            "output",
            "scale_a",
            "scale_b",
            "offs",
        )
        source_to_abi.update(
            {
                "mat1": "A",
                "mat2": "B",
                "output": "OUT",
                "scale_a": "A_scale",
                "scale_b": "B_scale",
                "offs": "OFFS",
            }
        )
    elif template.startswith("flex_flydsl") and explicit_sources is None:
        ordered_sources = (*input_names, "output")
    tensor_args = []
    static_layout = template.startswith("flex_flydsl")
    arg_pairs = (
        tuple(zip(explicit_names, explicit_sources))
        if explicit_sources is not None
        else tuple(
            (source_to_abi.get(source, source), source) for source in ordered_sources
        )
    )
    for name, source in arg_pairs:
        shape = tuple(shapes[source])
        stride = tuple(strides[source])
        flattened = template == "mxfp8_grouped_gemm_flydsl" and source != "offs"
        dynamic_sizes = (
            () if static_layout else ((0,) if flattened else tuple(range(len(shape))))
        )
        dynamic_strides = (
            ()
            if static_layout or flattened
            else tuple(index for index, value in enumerate(stride) if value != 1)
        )
        tensor_args.append(
            ExternalAOTTensorArg(
                name=name,
                source=source,
                dynamic_sizes=dynamic_sizes,
                dynamic_strides=dynamic_strides,
                read_only=source != "output",
                mutated=source == "output",
                flattened=flattened,
            )
        )
    if (
        template == "mm_flydsl"
        and "scale_a" in input_names
        and "bias" not in input_names
    ):
        tensor_args.append(
            ExternalAOTTensorArg(
                name="bias",
                source="output",
                dynamic_sizes=tuple(range(len(shapes["output"]))),
                dynamic_strides=tuple(
                    index for index, value in enumerate(strides["output"]) if value != 1
                ),
                read_only=True,
                alias_of="out",
            )
        )
    scalar_args = ()
    if template == "grouped_gemm_flydsl":
        scalar_args = (
            ExternalAOTScalarArg("group_count", "int32_t", "group_count"),
            ExternalAOTScalarArg("n", "int32_t", "n"),
            ExternalAOTScalarArg("k", "int32_t", "k"),
            ExternalAOTScalarArg("grid_size", "int32_t", "grid_size"),
        )
    elif template == "mxfp8_grouped_gemm_flydsl":
        scalar_args = (
            ExternalAOTScalarArg("n_blocks", "int32_t", "n_blocks"),
            ExternalAOTScalarArg("n_c_tiles", "int32_t", "n_c_tiles"),
            ExternalAOTScalarArg("out_m", "int32_t", "out_m"),
            ExternalAOTScalarArg("out_n", "int32_t", "out_n"),
        )
    return ExternalAOTBuildSpec(
        kind="flydsl",
        kernel_name=kernel_name,
        target="gfx950",
        source_hash=hashlib.sha256(source_code.encode("utf-8")).hexdigest(),
        tensor_args=tuple(tensor_args),
        scalar_args=scalar_args,
    )
