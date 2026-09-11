"""Repeatable torch.compile benchmarks for pretrained diffusion pipelines.

Each requested scenario runs in a fresh worker with an isolated compiler cache.
The parent process writes one CSV summary and a JSON-lines artifact compatible
with the PyTorch OSS benchmark database.

Examples:
    python compile_benchmark.py --model auraflow --mode full
    python compile_benchmark.py --model flux --mode full --backend eager
    python compile_benchmark.py --model tiny_stable_diffusion --mode all \
        --check-outputs --output results.csv

The ``auroflow`` spelling remains as an alias for compatibility.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import dataclasses
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import signal
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


MODES = ("eager", "full", "regional", "hierarchical")
OUTPUT_BOUNDARIES = ("latent", "decoded_tensor", "postprocessed_output")
STATUS_VALUES = ("success", "failed", "timed_out", "killed", "missing", "malformed")
DEFAULT_OUTPUT = "diffusion_benchmark.csv"
BENCHMARK_NAME = "PyTorch diffusion pipeline benchmark"


@dataclasses.dataclass(frozen=True)
class ModelArtifact:
    repo_id: str
    revision: str
    filename: str | None = None
    repo_type: str = "model"
    ignore_patterns: tuple[str, ...] = ()

    @property
    def key(self) -> str:
        return f"{self.repo_type}:{self.repo_id}:{self.filename or ''}"


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    name: str
    checkpoint: str
    revision: str
    component: str
    component_attr: str
    repeated_block_classes: tuple[str, ...]
    default_device: str
    default_dtype: str
    validation: str
    dependencies: tuple[str, ...]
    artifacts: tuple[ModelArtifact, ...]


@dataclasses.dataclass(frozen=True)
class WorkloadConfig:
    parameters: dict[str, Any]
    seed: int
    output_boundary: str


@dataclasses.dataclass(frozen=True)
class ExecutionConfig:
    modes: tuple[str, ...]
    backend: str
    cudagraphs: bool
    warmups: int
    repetitions: int
    timeout_s: float
    device: str
    dtype: str
    num_threads: int | None
    cache_policy: str = "isolated-per-scenario"


@dataclasses.dataclass(frozen=True)
class Recipe:
    model: ModelConfig
    workload: WorkloadConfig
    loader: str


@dataclasses.dataclass(frozen=True)
class Scenario:
    model: ModelConfig
    workload: WorkloadConfig
    execution: ExecutionConfig
    mode: str
    loader: str

    @property
    def scenario_id(self) -> str:
        cg = "cudagraphs" if self.execution.cudagraphs else "no-cudagraphs"
        return f"{self.model.name}:{self.mode}:{self.execution.backend}:{cg}"


@dataclasses.dataclass
class LoadedPipeline:
    pipeline: Any
    component: Any
    make_request: Callable[[], tuple[tuple[Any, ...], dict[str, Any]]]
    output_fields: tuple[str, ...]


def _artifact(
    repo_id: str,
    revision: str,
    filename: str | None = None,
    repo_type: str = "model",
    *,
    ignore_patterns: tuple[str, ...] = (),
) -> ModelArtifact:
    return ModelArtifact(repo_id, revision, filename, repo_type, ignore_patterns)


COMMON_DEPENDENCIES = (
    "torch",
    "diffusers",
    "transformers",
    "accelerate",
    "huggingface-hub",
    "safetensors",
    "numpy",
    "pillow",
)


def _recipe(
    name: str,
    checkpoint: str,
    revision: str,
    repeated_blocks: tuple[str, ...],
    parameters: dict[str, Any],
    output_boundary: str,
    loader: str,
    *,
    component_attr: str = "transformer",
    device: str = "cuda",
    dtype: str = "bfloat16",
    seed: int = 0,
    validation: str = "not executed in this change",
    dependencies: tuple[str, ...] = COMMON_DEPENDENCIES,
    artifacts: tuple[ModelArtifact, ...] | None = None,
) -> Recipe:
    return Recipe(
        ModelConfig(
            name=name,
            checkpoint=checkpoint,
            revision=revision,
            component=f"pipeline.{component_attr}",
            component_attr=component_attr,
            repeated_block_classes=repeated_blocks,
            default_device=device,
            default_dtype=dtype,
            validation=validation,
            dependencies=dependencies,
            artifacts=(
                (_artifact(checkpoint, revision),) if artifacts is None else artifacts
            ),
        ),
        WorkloadConfig(parameters, seed, output_boundary),
        loader,
    )


def _recipes() -> dict[str, Recipe]:
    pilot_repo = "diffusers/tiny-stable-diffusion-torch"
    pilot_revision = "c05d78754bd90a3ffea8fd3a0823ac17329cd1a6"
    recipes = {
        "tiny_stable_diffusion": _recipe(
            "tiny_stable_diffusion",
            pilot_repo,
            pilot_revision,
            ("BasicTransformerBlock",),
            {
                "prompt": "a small red cube",
                "height": 32,
                "width": 32,
                "num_inference_steps": 2,
                "guidance_scale": 1.0,
                "output_type": "latent",
            },
            "latent",
            "_load_tiny_stable_diffusion",
            component_attr="unet",
            device="cpu",
            dtype="float32",
            seed=1337,
            validation="validated pilot",
        ),
        "auraflow": _recipe(
            "auraflow",
            "fal/AuraFlow-v0.3",
            "2cd8588f04c886002be4571697d84654a50e3af3",
            ("AuraFlowSingleTransformerBlock", "AuraFlowJointTransformerBlock"),
            {
                "prompt": "A cute pony",
                "width": 512,
                "height": 512,
                "num_inference_steps": 50,
                "output_type": "pil",
            },
            "postprocessed_output",
            "_load_auraflow",
            dependencies=COMMON_DEPENDENCIES + ("gguf",),
            artifacts=(
                _artifact(
                    "fal/AuraFlow-v0.3",
                    "2cd8588f04c886002be4571697d84654a50e3af3",
                    ignore_patterns=(
                        "aura_flow_0.3.safetensors",
                        "transformer/*.safetensors",
                        "*.fp16.*",
                    ),
                ),
                _artifact(
                    "city96/AuraFlow-v0.3-gguf",
                    "f747743ef0dcc57229879fe54340dd20e907c456",
                    "aura_flow_0.3-Q2_K.gguf",
                ),
            ),
        ),
        "wan": _recipe(
            "wan",
            "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
            "b184e23a8a16b20f108f727c902e769e873ffc73",
            ("WanTransformerBlock",),
            {
                "prompt": "An astronaut hatching from an egg, on the surface of the moon, the darkness and depth of space realised in the background. High quality, ultrarealistic detail and breath-taking movie-like camera shot.",
                "negative_prompt": "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
                "num_frames": 33,
                "num_inference_steps": 50,
                "guidance_scale": 5.0,
                "output_type": "np",
            },
            "postprocessed_output",
            "_load_wan",
            artifacts=(
                _artifact(
                    "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
                    "b184e23a8a16b20f108f727c902e769e873ffc73",
                    ignore_patterns=("assets/*", "examples/*"),
                ),
                _artifact(
                    "huggingface/documentation-images",
                    "541575dc4c26c063abbd2a259c740835e88a3e6d",
                    "diffusers/astronaut.jpg",
                    "dataset",
                ),
            ),
        ),
        "ltx": _recipe(
            "ltx",
            "Lightricks/LTX-Video-0.9.7-dev",
            "2101082f5eb5540770a2df43747feadb6f69b889",
            ("LTXVideoTransformerBlock",),
            {
                "conditions": None,
                "prompt": "The video depicts a winding mountain road covered in snow, with a single vehicle traveling along it. The road is flanked by steep, rocky cliffs and sparse vegetation. The landscape is characterized by rugged terrain and a river visible in the distance. The scene captures the solitude and beauty of a winter drive through a mountainous region.",
                "negative_prompt": "worst quality, inconsistent motion, blurry, jittery, distorted",
                "width": 704,
                "height": 512,
                "num_frames": 121,
                "num_inference_steps": 50,
                "output_type": "latent",
            },
            "latent",
            "_load_ltx",
            artifacts=(
                _artifact(
                    "Lightricks/LTX-Video-0.9.7-dev",
                    "2101082f5eb5540770a2df43747feadb6f69b889",
                    ignore_patterns=("media/*",),
                ),
            ),
        ),
        "flux": _recipe(
            "flux",
            "black-forest-labs/FLUX.1-dev",
            "3de623fc3c33e44ffbe2bad470d0f45bccf2eb21",
            ("FluxTransformerBlock", "FluxSingleTransformerBlock"),
            {
                "prompt": "A cat holding a sign that says hello world",
                "height": 1024,
                "width": 1024,
                "guidance_scale": 3.5,
                "num_inference_steps": 50,
                "max_sequence_length": 512,
                "output_type": "pil",
            },
            "postprocessed_output",
            "_load_flux",
            validation="not executed in this change; checkpoint access is gated",
            artifacts=(
                _artifact(
                    "black-forest-labs/FLUX.1-dev",
                    "3de623fc3c33e44ffbe2bad470d0f45bccf2eb21",
                    ignore_patterns=("ae.safetensors", "flux1-dev.safetensors"),
                ),
            ),
        ),
        "toy": _recipe(
            "toy",
            "builtin://diffusion-toy",
            "1",
            ("ToyBlock",),
            {"num_inference_steps": 2, "output_type": "latent"},
            "latent",
            "_load_toy",
            component_attr="denoiser",
            device="cpu",
            dtype="float32",
            seed=11,
            validation="download-free test workload",
            dependencies=("torch",),
            artifacts=(),
        ),
    }
    recipes["auroflow"] = recipes["auraflow"]
    return recipes


BENCHMARKS = _recipes()


def _dtype_from_name(name: str) -> Any:
    import torch

    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def _resolve_device_and_dtype(scenario: Scenario) -> tuple[str, str, Any]:
    import torch

    device = scenario.execution.device
    if device == "auto":
        device = scenario.model.default_device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"{scenario.model.name} requires CUDA, but CUDA is unavailable"
        )
    dtype_name = scenario.execution.dtype
    if dtype_name == "auto":
        dtype_name = scenario.model.default_dtype
    return device, dtype_name, _dtype_from_name(dtype_name)


def _prefetch(artifacts: tuple[ModelArtifact, ...]) -> dict[str, str]:
    if not artifacts:
        return {}
    from huggingface_hub import hf_hub_download, snapshot_download

    paths = {}
    for artifact in artifacts:
        if artifact.filename is None:
            path = snapshot_download(
                repo_id=artifact.repo_id,
                revision=artifact.revision,
                repo_type=artifact.repo_type,
                ignore_patterns=list(artifact.ignore_patterns),
            )
        else:
            path = hf_hub_download(
                repo_id=artifact.repo_id,
                revision=artifact.revision,
                filename=artifact.filename,
                repo_type=artifact.repo_type,
            )
        paths[artifact.key] = path
    return paths


def _fresh_request(
    pipeline: Any,
    parameters: dict[str, Any],
    seed: int,
    device: str,
    mutable_parameters: dict[str, Callable[[], Any]] | None = None,
) -> Callable[[], tuple[tuple[Any, ...], dict[str, Any]]]:
    import torch

    scheduler_class = type(pipeline.scheduler)
    scheduler_config = dict(pipeline.scheduler.config)

    def make_request() -> tuple[tuple[Any, ...], dict[str, Any]]:
        pipeline.scheduler = scheduler_class.from_config(scheduler_config)
        torch.manual_seed(seed)
        kwargs = dict(parameters)
        kwargs["generator"] = torch.Generator(device=device).manual_seed(seed)
        if mutable_parameters:
            kwargs.update(
                {name: factory() for name, factory in mutable_parameters.items()}
            )
        return (), kwargs

    return make_request


def _finish_loading(
    pipeline: Any,
    scenario: Scenario,
    device: str,
    request_parameters: dict[str, Any] | None = None,
    mutable_parameters: dict[str, Callable[[], Any]] | None = None,
    output_fields: tuple[str, ...] = ("images", "frames"),
) -> LoadedPipeline:
    pipeline = pipeline.to(device)
    if hasattr(pipeline, "set_progress_bar_config"):
        pipeline.set_progress_bar_config(disable=True)
    for module in pipeline.components.values():
        if hasattr(module, "eval"):
            module.eval()
    parameters = request_parameters or scenario.workload.parameters
    return LoadedPipeline(
        pipeline=pipeline,
        component=getattr(pipeline, scenario.model.component_attr),
        make_request=_fresh_request(
            pipeline,
            parameters,
            scenario.workload.seed,
            device,
            mutable_parameters,
        ),
        output_fields=output_fields,
    )


def _load_tiny_stable_diffusion(
    scenario: Scenario, paths: dict[str, str], device: str, dtype: Any
) -> LoadedPipeline:
    from diffusers import StableDiffusionPipeline

    artifact = scenario.model.artifacts[0]
    pipeline = StableDiffusionPipeline.from_pretrained(
        paths[artifact.key], dtype=dtype, safety_checker=None
    )
    return _finish_loading(pipeline, scenario, device)


def _load_auraflow(
    scenario: Scenario, paths: dict[str, str], device: str, dtype: Any
) -> LoadedPipeline:
    from diffusers import (
        AuraFlowPipeline,
        AuraFlowTransformer2DModel,
        GGUFQuantizationConfig,
    )

    base, gguf = scenario.model.artifacts
    transformer = AuraFlowTransformer2DModel.from_single_file(
        paths[gguf.key],
        config=paths[base.key],
        subfolder="transformer",
        local_files_only=True,
        quantization_config=GGUFQuantizationConfig(compute_dtype=dtype),
        torch_dtype=dtype,
    )
    pipeline = AuraFlowPipeline.from_pretrained(
        paths[base.key], torch_dtype=dtype, transformer=transformer
    )
    return _finish_loading(pipeline, scenario, device)


def _load_wan(
    scenario: Scenario, paths: dict[str, str], device: str, dtype: Any
) -> LoadedPipeline:
    import numpy as np
    from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
    from PIL import Image
    from transformers import CLIPVisionModel

    import torch

    model, image_artifact = scenario.model.artifacts
    model_path = paths[model.key]
    with Image.open(paths[image_artifact.key]) as source:
        image = source.convert("RGB")
    image_encoder = CLIPVisionModel.from_pretrained(
        model_path, subfolder="image_encoder", torch_dtype=torch.float32
    )
    vae = AutoencoderKLWan.from_pretrained(
        model_path, subfolder="vae", torch_dtype=torch.float32
    )
    pipeline = WanImageToVideoPipeline.from_pretrained(
        model_path, vae=vae, image_encoder=image_encoder, torch_dtype=dtype
    )
    max_area = 480 * 832
    aspect_ratio = image.height / image.width
    mod_value = (
        pipeline.vae_scale_factor_spatial * pipeline.transformer.config.patch_size[1]
    )
    height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
    width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
    image = image.resize((width, height))
    parameters = dict(scenario.workload.parameters, height=height, width=width)
    return _finish_loading(
        pipeline,
        scenario,
        device,
        parameters,
        mutable_parameters={"image": image.copy},
        output_fields=("frames",),
    )


def _load_ltx(
    scenario: Scenario, paths: dict[str, str], device: str, dtype: Any
) -> LoadedPipeline:
    from diffusers import LTXConditionPipeline

    artifact = scenario.model.artifacts[0]
    pipeline = LTXConditionPipeline.from_pretrained(
        paths[artifact.key], torch_dtype=dtype
    )
    pipeline.vae.enable_tiling()
    height = 512 - (512 % pipeline.vae_spatial_compression_ratio)
    width = 704 - (704 % pipeline.vae_spatial_compression_ratio)
    parameters = dict(scenario.workload.parameters, height=height, width=width)
    return _finish_loading(
        pipeline, scenario, device, parameters, output_fields=("frames",)
    )


def _load_flux(
    scenario: Scenario, paths: dict[str, str], device: str, dtype: Any
) -> LoadedPipeline:
    from diffusers import FluxPipeline

    artifact = scenario.model.artifacts[0]
    pipeline = FluxPipeline.from_pretrained(paths[artifact.key], torch_dtype=dtype)
    return _finish_loading(pipeline, scenario, device)


def _load_toy(
    scenario: Scenario, paths: dict[str, str], device: str, dtype: Any
) -> LoadedPipeline:
    import torch

    class ToyBlock(torch.nn.Module):
        def __init__(self, width: int) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(width, width)

        def forward(self, value: Any, timestep: Any) -> Any:
            return torch.tanh(self.linear(value) + timestep)

    class ToyDenoiser(torch.nn.Module):
        _repeated_blocks = ("ToyBlock",)

        def __init__(self) -> None:
            super().__init__()
            self.blocks = torch.nn.ModuleList(ToyBlock(8) for _ in range(2))

        def forward(self, value: Any, timestep: Any) -> Any:
            for block in self.blocks:
                value = block(value, timestep)
            return value

        def compile_repeated_blocks(self, *args: Any, **kwargs: Any) -> None:
            for module in self.modules():
                if type(module).__name__ in self._repeated_blocks:
                    module.compile(*args, **kwargs)

    class ToyScheduler:
        config = {"step_scale": 0.01}

        def __init__(self, step_scale: float = 0.01) -> None:
            self.step_scale = step_scale
            self.step_index = 0

        @classmethod
        def from_config(cls, config: dict[str, Any]) -> ToyScheduler:
            return cls(**config)

        def step(self, value: Any) -> Any:
            self.step_index += 1
            return value - self.step_scale * self.step_index

    class ToyPipeline:
        def __init__(self) -> None:
            self.denoiser = ToyDenoiser()
            self.scheduler = ToyScheduler()
            self.components = {"denoiser": self.denoiser}

        def to(self, target: str) -> ToyPipeline:
            self.denoiser.to(device=target, dtype=dtype)
            return self

        def __call__(
            self,
            *,
            num_inference_steps: int,
            output_type: str,
            generator: Any,
        ) -> dict[str, Any]:
            value = torch.randn(1, 8, generator=generator, device=device, dtype=dtype)
            for step in range(num_inference_steps):
                timestep = torch.tensor(float(step), device=device, dtype=dtype)
                value = self.scheduler.step(self.denoiser(value, timestep))
            return {"images": value}

    torch.manual_seed(scenario.workload.seed)
    return _finish_loading(ToyPipeline(), scenario, device)


def _matched_region_classes(
    component: Any, expected: tuple[str, ...]
) -> tuple[tuple[type, ...], int]:
    matches = [
        module for module in component.modules() if type(module).__name__ in expected
    ]
    if not matches:
        raise ValueError(
            f"no repeated compilation targets matched {expected} in {type(component).__name__}"
        )
    classes = {type(module) for module in matches}
    missing = set(expected) - {cls.__name__ for cls in classes}
    if missing:
        raise ValueError(
            f"missing repeated compilation target classes: {sorted(missing)}"
        )
    ordered = tuple(sorted(classes, key=lambda cls: (cls.__module__, cls.__qualname__)))
    return ordered, len(matches)


def _compile_kwargs(backend: str, cudagraphs: bool) -> dict[str, Any]:
    if cudagraphs and backend != "inductor":
        raise ValueError("CUDA graphs require the inductor backend")
    kwargs: dict[str, Any] = {"backend": backend, "fullgraph": True}
    if backend == "inductor":
        kwargs["options"] = {"triton.cudagraphs": cudagraphs}
    return kwargs


@contextlib.contextmanager
def configure_compilation(
    component: Any,
    mode: str,
    expected_regions: tuple[str, ...],
    backend: str,
    cudagraphs: bool,
) -> Iterator[dict[str, Any]]:
    if mode not in MODES:
        raise ValueError(f"invalid compilation mode: {mode}")
    if mode == "eager":
        if cudagraphs:
            raise ValueError("CUDA graphs cannot be enabled for eager execution")
        yield {
            "compiled_targets": [],
            "matched_module_count": 0,
            "regional_entry_point": None,
        }
        return

    import torch

    kwargs = _compile_kwargs(backend, cudagraphs)
    details = {
        "compiled_targets": [],
        "matched_module_count": 0,
        "regional_entry_point": None,
    }
    if mode == "full":
        component.compile(**kwargs)
        details["compiled_targets"] = [type(component).__name__]
        details["matched_module_count"] = 1
        yield details
        return

    matched_classes, matched_module_count = _matched_region_classes(
        component, expected_regions
    )
    details["compiled_targets"] = [cls.__name__ for cls in matched_classes]
    details["matched_module_count"] = matched_module_count
    if mode == "regional":
        repeated = tuple(getattr(component, "_repeated_blocks", ()))
        if callable(getattr(component, "compile_repeated_blocks", None)) and repeated:
            if set(repeated) != set(expected_regions):
                raise ValueError(
                    f"model repeated blocks {repeated} do not match recipe {expected_regions}"
                )
            component.compile_repeated_blocks(**kwargs)
            details["regional_entry_point"] = "model.compile_repeated_blocks"
        else:
            for module in component.modules():
                if type(module) in matched_classes:
                    module.compile(**kwargs)
            details["regional_entry_point"] = "explicit class selection"
        yield details
        return

    originals: dict[type, Any] = {}
    try:
        for cls in matched_classes:
            forward = cls.forward
            if not hasattr(forward, "__marked_compile_region_fn__"):
                originals[cls] = forward
                cls.forward = torch.compiler.nested_compile_region(forward)
        component.compile(**kwargs)
        details["regional_entry_point"] = "torch.compiler.nested_compile_region"
        yield details
    finally:
        for cls, forward in originals.items():
            cls.forward = forward


def _synchronize(device: str) -> None:
    import torch

    device_type = torch.device(device).type
    if device_type == "cuda":
        torch.cuda.synchronize(device)
    elif device_type == "xpu":
        torch.xpu.synchronize(device)
    elif device_type == "mps":
        torch.mps.synchronize()


def _timed_request(
    loaded: LoadedPipeline, device: str, *, cudagraphs: bool
) -> tuple[float, Any]:
    import torch

    args, kwargs = loaded.make_request()
    _synchronize(device)
    if cudagraphs:
        torch.compiler.cudagraph_mark_step_begin()
    start = time.perf_counter()
    output = loaded.pipeline(*args, **kwargs)
    _synchronize(device)
    return time.perf_counter() - start, output


def _extract_tensor(value: Any, fields: tuple[str, ...]) -> Any:
    import numpy as np
    from PIL import Image

    import torch

    for field in fields:
        if isinstance(value, dict) and field in value:
            return _extract_tensor(value[field], fields)
        if hasattr(value, field):
            return _extract_tensor(getattr(value, field), fields)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value.copy())
    if isinstance(value, Image.Image):
        return torch.from_numpy(np.asarray(value).copy())
    if isinstance(value, (list, tuple)) and value:
        tensors = [_extract_tensor(item, fields) for item in value]
        return torch.stack(tensors)
    raise TypeError(f"cannot extract a tensor from output type {type(value).__name__}")


def _tensor_summary(tensor: Any) -> dict[str, Any]:
    import torch

    contiguous = tensor.contiguous()
    byte_view = contiguous.view(dtype=torch.uint8)
    digest = hashlib.sha256(byte_view.numpy().tobytes()).hexdigest()
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype), "sha256": digest}


def _device_memory(device: str) -> dict[str, int | None]:
    import torch

    if torch.device(device).type != "cuda":
        return {"allocated_bytes": None, "reserved_bytes": None}
    return {
        "allocated_bytes": torch.cuda.max_memory_allocated(device),
        "reserved_bytes": torch.cuda.max_memory_reserved(device),
    }


def _reset_device_memory(device: str) -> None:
    import torch

    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _compiler_diagnostics(mode: str, device: str, cudagraphs: bool) -> dict[str, Any]:
    import torch
    from torch._dynamo.utils import compilation_time_metrics, counters

    counter_values = {
        category: dict(values) for category, values in counters.items() if values
    }
    unique_graphs = counters["stats"]["unique_graphs"]
    diagnostics: dict[str, Any] = {
        "counters": counter_values,
        "unique_graphs": unique_graphs,
        "compiler_times_s": {
            name: list(values) for name, values in compilation_time_metrics.items()
        },
        "cudagraph_capture_count": None,
        "hierarchical_reuse_claimed": False,
        "steady_state_compilation_detected": False,
    }
    if mode != "eager" and unique_graphs == 0:
        raise RuntimeError("torch.compile produced no compiled graphs")
    if mode == "eager" and unique_graphs != 0:
        raise RuntimeError("eager scenario unexpectedly compiled a graph")
    if cudagraphs:
        from torch._inductor.cudagraph_trees import get_manager

        index = torch.device(device).index
        if index is None:
            index = torch.cuda.current_device()
        manager = get_manager(index, create_if_none_exists=False)
        nodes = [] if manager is None else list(manager.get_roots())
        capture_count = 0
        while nodes:
            node = nodes.pop()
            capture_count += node.graph is not None
            for children in node.children.values():
                nodes.extend(children)
        diagnostics["cudagraph_capture_count"] = capture_count
        if capture_count == 0:
            raise RuntimeError(
                "CUDA graphs were requested, but no graph capture was recorded"
            )
    return diagnostics


def _dependency_versions(names: tuple[str, ...]) -> dict[str, str]:
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def _hardware_info(device: str) -> dict[str, Any]:
    import torch

    cpu_model = platform.processor()
    system_memory = None
    try:
        cpu_model = next(
            line.split(":", 1)[1].strip()
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines()
            if line.startswith("model name")
        )
    except (OSError, StopIteration):
        pass
    try:
        system_memory = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        pass
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": cpu_model,
        "cpu_count": os.cpu_count(),
        "system_memory_bytes": system_memory,
        "torch_cpu_capability": torch.backends.cpu.get_cpu_capability(),
    }
    if torch.device(device).type == "cuda":
        props = torch.cuda.get_device_properties(device)
        info.update(
            {
                "accelerator": props.name,
                "accelerator_memory_bytes": props.total_memory,
                "compute_capability": f"{props.major}.{props.minor}",
                "driver_runtime": torch.version.cuda,
            }
        )
    else:
        info["accelerator"] = torch.device(device).type
    return info


def _relevant_environment() -> dict[str, str]:
    names = (
        "CPLUS_INCLUDE_PATH",
        "CXX",
        "LD_LIBRARY_PATH",
        "LIBRARY_PATH",
        "OMP_NUM_THREADS",
    )
    return {name: os.environ[name] for name in names if name in os.environ}


def _validate_worker_scenario(scenario: Scenario, device: str) -> None:
    if scenario.mode not in MODES:
        raise ValueError(f"invalid compilation mode: {scenario.mode}")
    if scenario.workload.output_boundary not in OUTPUT_BOUNDARIES:
        raise ValueError(
            f"invalid output boundary: {scenario.workload.output_boundary}"
        )
    if scenario.execution.warmups < 0:
        raise ValueError("warmups must be non-negative")
    if scenario.execution.repetitions < 1:
        raise ValueError("repetitions must be positive")
    if scenario.execution.timeout_s <= 0:
        raise ValueError("timeout must be positive")
    if (
        scenario.execution.num_threads is not None
        and scenario.execution.num_threads < 1
    ):
        raise ValueError("num_threads must be positive")
    if scenario.execution.cudagraphs:
        if scenario.mode == "eager":
            raise ValueError("CUDA graphs cannot be enabled for eager execution")
        if scenario.execution.backend != "inductor":
            raise ValueError("CUDA graphs require the inductor backend")
        if not device.startswith("cuda"):
            raise ValueError("CUDA graphs require a CUDA device")
        if scenario.execution.warmups < 1:
            raise ValueError("CUDA graphs require at least one warmup request")


def execute_scenario(
    scenario: Scenario, sample_path: str | None = None
) -> dict[str, Any]:
    import torch
    from torch._dynamo.utils import compilation_time_metrics, counters

    device, dtype_name, dtype = _resolve_device_and_dtype(scenario)
    _validate_worker_scenario(scenario, device)
    cudagraphs = scenario.execution.cudagraphs
    if scenario.execution.num_threads is not None:
        torch.set_num_threads(scenario.execution.num_threads)

    paths = _prefetch(scenario.model.artifacts)
    _synchronize(device)
    _reset_device_memory(device)
    setup_start = time.perf_counter()
    loader = globals()[scenario.loader]
    loaded = loader(scenario, paths, device, dtype)
    _synchronize(device)
    setup_s = time.perf_counter() - setup_start
    setup_memory = _device_memory(device)
    torch._dynamo.reset()
    counters.clear()
    compilation_time_metrics.clear()

    compile_start = time.perf_counter()
    with (
        torch._dynamo.config.patch(suppress_errors=False),
        configure_compilation(
            loaded.component,
            scenario.mode,
            scenario.model.repeated_block_classes,
            scenario.execution.backend,
            cudagraphs,
        ) as compile_details,
    ):
        compile_setup_s = time.perf_counter() - compile_start
        _reset_device_memory(device)
        with torch.inference_mode():
            first_request_s, output = _timed_request(
                loaded, device, cudagraphs=cudagraphs
            )
            if scenario.mode != "eager" and counters["stats"]["unique_graphs"] == 0:
                raise RuntimeError("first request completed without compiling a graph")
            del output
            for _ in range(scenario.execution.warmups):
                _, output = _timed_request(loaded, device, cudagraphs=cudagraphs)
                del output
            graphs_before_measurement = counters["stats"]["unique_graphs"]
            metric_counts = {
                name: len(values) for name, values in compilation_time_metrics.items()
            }
            samples = []
            for repetition in range(scenario.execution.repetitions):
                duration, output = _timed_request(loaded, device, cudagraphs=cudagraphs)
                samples.append(duration)
                if repetition + 1 < scenario.execution.repetitions:
                    del output
            new_compiler_metrics = {
                name: values[metric_counts.get(name, 0) :]
                for name, values in compilation_time_metrics.items()
                if len(values) > metric_counts.get(name, 0)
            }
            new_graphs = counters["stats"]["unique_graphs"] - graphs_before_measurement
            if new_graphs or new_compiler_metrics:
                raise RuntimeError(
                    "compilation occurred during steady-state measurement: "
                    f"new_graphs={new_graphs}, phases={sorted(new_compiler_metrics)}"
                )
        request_memory = _device_memory(device)
        diagnostics = _compiler_diagnostics(scenario.mode, device, cudagraphs)

    tensor = _extract_tensor(output, loaded.output_fields)
    output_summary = _tensor_summary(tensor)
    if sample_path is not None:
        torch.save(tensor, sample_path)
    median_s = statistics.median(samples)
    mad_s = statistics.median(abs(sample - median_s) for sample in samples)
    compile_options = (
        _compile_kwargs(scenario.execution.backend, cudagraphs)
        if scenario.mode != "eager"
        else {}
    )
    return {
        "scenario_id": scenario.scenario_id,
        "status": "success",
        "error": None,
        "model": dataclasses.asdict(scenario.model),
        "workload": dataclasses.asdict(scenario.workload),
        "execution": dataclasses.asdict(scenario.execution),
        "mode": scenario.mode,
        "pid": os.getpid(),
        "device": device,
        "dtype": dtype_name,
        "compiled_component": scenario.model.component
        if scenario.mode != "eager"
        else None,
        "compiled_targets": compile_details["compiled_targets"],
        "matched_module_count": compile_details["matched_module_count"],
        "regional_entry_point": compile_details["regional_entry_point"],
        "effective_compile_options": compile_options,
        "compiler_error_fallback": False,
        "prefetch_included_in_setup": False,
        "model_setup_s": setup_s,
        "compile_wrapper_setup_s": compile_setup_s,
        "first_request_s": first_request_s,
        "steady_state_samples_s": samples,
        "steady_state_median_s": median_s,
        "steady_state_mad_s": mad_s,
        "setup_device_peak": setup_memory,
        "request_device_peak": request_memory,
        "device_memory_definition": (
            "CUDA allocated/reserved process peaks; request peaks include resident model state"
            if device.startswith("cuda")
            else "unavailable for non-CUDA devices"
        ),
        "compiler_diagnostics": diagnostics,
        "output_summary": output_summary,
        "output_check": "not requested",
        "dependencies": _dependency_versions(scenario.model.dependencies),
        "torch": {
            "version": torch.__version__,
            "git_version": torch.version.git_version,
            "cuda": torch.version.cuda,
            "build_config": torch.__config__.show(),
        },
        "python_version": platform.python_version(),
        "hardware": _hardware_info(device),
        "environment": _relevant_environment(),
        "cache_policy": {
            "name": scenario.execution.cache_policy,
            "fx_graph_cache": False,
            "aot_autograd_cache": False,
            "remote_cache": False,
        },
    }


def _scenario_to_dict(scenario: Scenario) -> dict[str, Any]:
    return dataclasses.asdict(scenario)


def _scenario_from_dict(value: dict[str, Any]) -> Scenario:
    model_value = dict(value["model"])
    model_value["repeated_block_classes"] = tuple(model_value["repeated_block_classes"])
    model_value["dependencies"] = tuple(model_value["dependencies"])
    model_value["artifacts"] = tuple(
        ModelArtifact(
            **dict(artifact, ignore_patterns=tuple(artifact.get("ignore_patterns", ())))
        )
        for artifact in model_value["artifacts"]
    )
    execution_value = dict(value["execution"])
    execution_value["modes"] = tuple(execution_value["modes"])
    return Scenario(
        model=ModelConfig(**model_value),
        workload=WorkloadConfig(**value["workload"]),
        execution=ExecutionConfig(**execution_value),
        mode=value["mode"],
        loader=value["loader"],
    )


def _write_json(path: str | Path, value: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2)
    temporary.replace(output)


def _worker_main(config_path: str) -> int:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    scenario = _scenario_from_dict(config["scenario"])
    result_path = config["result_path"]
    try:
        result = execute_scenario(scenario, config.get("sample_path"))
    except BaseException as error:
        result = {
            "scenario_id": scenario.scenario_id,
            "status": "failed",
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
            "model": dataclasses.asdict(scenario.model),
            "workload": dataclasses.asdict(scenario.workload),
            "execution": dataclasses.asdict(scenario.execution),
            "mode": scenario.mode,
            "pid": os.getpid(),
        }
        _write_json(result_path, result)
        traceback.print_exc()
        return 1
    _write_json(result_path, result)
    return 0


def _failure_result(scenario_id: str, status: str, error: str) -> dict[str, Any]:
    return {"scenario_id": scenario_id, "status": status, "error": error}


def _validate_worker_result(result: Any, scenario_id: str) -> None:
    if not isinstance(result, dict) or result.get("scenario_id") != scenario_id:
        raise ValueError("worker result has the wrong scenario id")
    status = result.get("status")
    if status not in STATUS_VALUES:
        raise ValueError(f"malformed status: {status!r}")
    for name in (
        "model",
        "workload",
        "execution",
        "compiler_diagnostics",
        "setup_device_peak",
        "request_device_peak",
    ):
        if (status == "success" or name in result) and not isinstance(
            result.get(name), dict
        ):
            raise ValueError(f"malformed {name}: expected an object")
    if status != "success":
        if not isinstance(result.get("error"), str):
            raise ValueError("failed worker result is missing an error message")
        return

    if result.get("mode") not in MODES:
        raise ValueError(f"malformed mode: {result.get('mode')!r}")
    for name, value in (
        ("model.name", result["model"].get("name")),
        ("execution.backend", result["execution"].get("backend")),
        ("execution.dtype", result["execution"].get("dtype")),
        ("dtype", result.get("dtype", result["execution"].get("dtype"))),
    ):
        if not isinstance(value, str) or not value:
            raise ValueError(f"malformed {name}: expected a nonempty string")
    if result["workload"].get("output_boundary") not in OUTPUT_BOUNDARIES:
        raise ValueError("malformed output boundary")
    targets = result.get("compiled_targets", [])
    if not isinstance(targets, list) or any(
        not isinstance(target, str) for target in targets
    ):
        raise ValueError("malformed compiled targets")
    if result["mode"] != "eager" and not targets:
        raise ValueError("missing compiled targets")

    timings = {
        name: [result.get(name)]
        for name in (
            "model_setup_s",
            "compile_wrapper_setup_s",
            "first_request_s",
            "steady_state_median_s",
            "steady_state_mad_s",
        )
    }
    samples = result.get("steady_state_samples_s")
    repetitions = result["execution"].get("repetitions")
    if (
        type(repetitions) is not int
        or repetitions < 1
        or not isinstance(samples, list)
        or len(samples) != repetitions
    ):
        raise ValueError("missing or wrong number of steady-state samples")
    timings["steady_state_samples_s"] = samples
    compiler_times = result["compiler_diagnostics"].get("compiler_times_s")
    if not isinstance(compiler_times, dict):
        raise ValueError("malformed compiler_times_s: expected an object")
    timings.update(
        {f"compiler_times_s.{name}": values for name, values in compiler_times.items()}
    )
    for name, values in timings.items():
        if not isinstance(values, list) or any(
            type(value) not in (int, float)
            or value < 0
            or (isinstance(value, float) and not math.isfinite(value))
            for value in values
        ):
            raise ValueError(f"malformed {name}: expected finite non-negative values")
    for phase in ("setup", "request"):
        memory = result[f"{phase}_device_peak"]
        for kind in ("allocated_bytes", "reserved_bytes"):
            if kind not in memory:
                raise ValueError(f"missing {phase} {kind}")
            value = memory[kind]
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"malformed {phase} {kind}: {value!r}")


def run_worker_process(
    scenario_id: str,
    command: list[str],
    result_path: str | Path,
    log_path: str | Path,
    timeout_s: float,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    log = Path(log_path)
    log.parent.mkdir(parents=True, exist_ok=True)
    try:
        with log.open("w", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                command,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                start_new_session=sys.platform != "win32",
            )
            timed_out = False
            try:
                process.wait(timeout=timeout_s)
            except BaseException as error:
                with contextlib.suppress(ProcessLookupError):
                    if sys.platform == "win32":
                        process.kill()
                    else:
                        os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                if not isinstance(error, subprocess.TimeoutExpired):
                    raise
                timed_out = True
    except OSError as error:
        return _failure_result(
            scenario_id, "failed", f"could not start worker: {error}"
        )
    if timed_out:
        return _failure_result(
            scenario_id,
            "timed_out",
            f"worker exceeded {timeout_s:g}s; log: {log}",
        )
    if process.returncode is not None and process.returncode < 0:
        return _failure_result(
            scenario_id,
            "killed",
            f"worker terminated by signal {-process.returncode}; log: {log}",
        )
    result_file = Path(result_path)
    if not result_file.exists():
        return _failure_result(
            scenario_id,
            "missing",
            f"worker exited {process.returncode} without a result; log: {log}",
        )
    try:
        result = json.loads(result_file.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        return _failure_result(
            scenario_id,
            "malformed",
            f"could not read worker result: {error}; log: {log}",
        )
    try:
        _validate_worker_result(result, scenario_id)
    except ValueError as error:
        return _failure_result(
            scenario_id,
            "malformed",
            f"{error}; log: {log}",
        )
    if process.returncode and result.get("status") == "success":
        return _failure_result(
            scenario_id,
            "malformed",
            f"worker exited {process.returncode} but reported success; log: {log}",
        )
    return result


def validate_scenario_results(
    declared_scenario_ids: list[str], results: list[dict[str, Any]]
) -> list[str]:
    errors = []
    if len(declared_scenario_ids) != len(set(declared_scenario_ids)):
        errors.append("declared scenario ids contain duplicates")
    by_id: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        scenario_id = result.get("scenario_id") if isinstance(result, dict) else None
        if not isinstance(scenario_id, str):
            errors.append("malformed result without a string scenario_id")
            continue
        by_id.setdefault(scenario_id, []).append(result)
    for scenario_id in declared_scenario_ids:
        matches = by_id.get(scenario_id, [])
        if not matches:
            errors.append(f"missing result for {scenario_id}")
            continue
        if len(matches) > 1:
            errors.append(f"duplicate results for {scenario_id}")
            continue
        result = matches[0]
        try:
            _validate_worker_result(result, scenario_id)
        except ValueError as error:
            errors.append(f"malformed result for {scenario_id}: {error}")
            continue
        status = result["status"]
        if status != "success":
            errors.append(
                f"{scenario_id} {status}: {result.get('error', 'unknown error')}"
            )
            continue
    unexpected = set(by_id) - set(declared_scenario_ids)
    for scenario_id in sorted(unexpected):
        errors.append(f"unexpected result for {scenario_id}")
    return errors


def _compare_outputs(
    scenarios: list[Scenario],
    results: list[dict[str, Any]],
    sample_paths: dict[str, Path],
) -> None:
    import torch

    eager = next((scenario for scenario in scenarios if scenario.mode == "eager"), None)
    if eager is None:
        raise ValueError("--check-outputs requires an eager scenario")
    eager_result = next(
        result for result in results if result["scenario_id"] == eager.scenario_id
    )
    if eager_result.get("status") != "success":
        return
    try:
        reference = torch.load(sample_paths[eager.scenario_id], weights_only=True)
    except (OSError, RuntimeError) as error:
        eager_result["status"] = "failed"
        eager_result["error"] = f"could not load eager output for comparison: {error}"
        eager_result["output_check"] = "failed"
        return
    eager_result["output_check"] = "reference"
    for scenario in scenarios:
        if scenario.mode == "eager":
            continue
        result = next(
            result
            for result in results
            if result["scenario_id"] == scenario.scenario_id
        )
        if result.get("status") != "success":
            continue
        try:
            actual = torch.load(sample_paths[scenario.scenario_id], weights_only=True)
            torch.testing.assert_close(actual, reference)
        except (AssertionError, OSError, RuntimeError) as error:
            result["status"] = "failed"
            result["error"] = f"output sanity check failed: {error}"
            result["output_check"] = "failed"
        else:
            result["output_check"] = "passed with torch.testing.assert_close defaults"


def _record_extra_info(result: dict[str, Any]) -> dict[str, Any]:
    excluded = {
        "status",
        "error",
        "model_setup_s",
        "compile_wrapper_setup_s",
        "first_request_s",
        "steady_state_samples_s",
        "steady_state_median_s",
        "steady_state_mad_s",
        "setup_device_peak",
        "request_device_peak",
    }
    return {name: value for name, value in result.items() if name not in excluded}


def dashboard_records(result: dict[str, Any]) -> list[dict[str, Any]]:
    model = result.get("model", {})
    execution = result.get("execution", {})
    benchmark = {
        "name": BENCHMARK_NAME,
        "mode": "inference",
        "dtype": result.get("dtype", execution.get("dtype", "unknown")),
        "extra_info": _record_extra_info(result),
    }
    model_record = {
        "name": model.get("name", result["scenario_id"].split(":", 1)[0]),
        "type": "micro-benchmark" if model.get("name") == "toy" else "OSS model",
        "backend": execution.get("backend")
        if result.get("mode") != "eager"
        else "eager",
        "origins": ["pytorch"] if model.get("name") == "toy" else ["huggingface"],
    }
    if result.get("status") != "success":
        return [
            {
                "benchmark": benchmark,
                "model": model_record,
                "metric": {
                    "name": "scenario_status",
                    "extra_info": {
                        "benchmark_values": [result.get("status", "malformed")],
                        "error": result.get("error", "unknown error"),
                    },
                },
            }
        ]

    metrics: list[tuple[str, list[float]]] = [
        ("model_setup_s", [result["model_setup_s"]]),
        ("compile_wrapper_setup_s", [result["compile_wrapper_setup_s"]]),
        ("first_request_s", [result["first_request_s"]]),
        ("steady_state_latency_s", result["steady_state_samples_s"]),
        ("steady_state_median_s", [result["steady_state_median_s"]]),
        ("steady_state_mad_s", [result["steady_state_mad_s"]]),
    ]
    for phase, samples in result["compiler_diagnostics"]["compiler_times_s"].items():
        metrics.append((f"compiler_{phase}_s", samples))
    for phase in ("setup", "request"):
        memory = result[f"{phase}_device_peak"]
        for kind in ("allocated_bytes", "reserved_bytes"):
            if memory[kind] is not None:
                metrics.append((f"{phase}_peak_{kind}", [memory[kind]]))
    return [
        {
            "benchmark": benchmark,
            "model": model_record,
            "metric": {"name": name, "benchmark_values": values},
        }
        for name, values in metrics
    ]


def _csv_value(value: Any) -> Any:
    return f"{value:.6f}" if isinstance(value, float) else value


def _output_paths(csv_path: str | Path) -> tuple[Path, Path]:
    csv_output = Path(csv_path)
    json_output = csv_output.with_suffix(".json")
    if csv_output.suffix.lower() == ".json":
        raise ValueError("CSV output path must not end in .json")
    return csv_output, json_output


def write_outputs(csv_path: str | Path, results: list[dict[str, Any]]) -> Path:
    csv_output, json_output = _output_paths(csv_path)
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "scenario_id",
        "status",
        "error",
        "model",
        "mode",
        "backend",
        "device",
        "dtype",
        "output_boundary",
        "compiled_component",
        "compiled_targets",
        "model_setup_s",
        "compile_wrapper_setup_s",
        "first_request_s",
        "steady_state_median_s",
        "steady_state_mad_s",
        "steady_state_samples_s",
        "setup_peak_allocated_bytes",
        "setup_peak_reserved_bytes",
        "request_peak_allocated_bytes",
        "request_peak_reserved_bytes",
        "compiler_times_s",
        "provenance",
    ]
    with csv_output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=headers, lineterminator="\n")
        writer.writeheader()
        for result in results:
            model = result.get("model", {})
            execution = result.get("execution", {})
            workload = result.get("workload", {})
            setup_memory = result.get("setup_device_peak", {})
            request_memory = result.get("request_device_peak", {})
            compiler = result.get("compiler_diagnostics", {})
            writer.writerow(
                {
                    "scenario_id": result.get("scenario_id"),
                    "status": result.get("status"),
                    "error": result.get("error"),
                    "model": model.get("name"),
                    "mode": result.get("mode"),
                    "backend": execution.get("backend"),
                    "device": result.get("device"),
                    "dtype": result.get("dtype"),
                    "output_boundary": workload.get("output_boundary"),
                    "compiled_component": result.get("compiled_component"),
                    "compiled_targets": json.dumps(result.get("compiled_targets")),
                    "model_setup_s": _csv_value(result.get("model_setup_s")),
                    "compile_wrapper_setup_s": _csv_value(
                        result.get("compile_wrapper_setup_s")
                    ),
                    "first_request_s": _csv_value(result.get("first_request_s")),
                    "steady_state_median_s": _csv_value(
                        result.get("steady_state_median_s")
                    ),
                    "steady_state_mad_s": _csv_value(result.get("steady_state_mad_s")),
                    "steady_state_samples_s": json.dumps(
                        result.get("steady_state_samples_s")
                    ),
                    "setup_peak_allocated_bytes": setup_memory.get("allocated_bytes"),
                    "setup_peak_reserved_bytes": setup_memory.get("reserved_bytes"),
                    "request_peak_allocated_bytes": request_memory.get(
                        "allocated_bytes"
                    ),
                    "request_peak_reserved_bytes": request_memory.get("reserved_bytes"),
                    "compiler_times_s": json.dumps(compiler.get("compiler_times_s")),
                    "provenance": json.dumps(_record_extra_info(result)),
                }
            )
    with json_output.open("w", encoding="utf-8") as file:
        for result in results:
            for record in dashboard_records(result):
                print(json.dumps(record), file=file)
    return json_output


def _execution_from_args(
    args: argparse.Namespace, modes: tuple[str, ...]
) -> ExecutionConfig:
    if args.warmups < 0:
        raise ValueError("--warmups must be non-negative")
    if args.repetitions < 1:
        raise ValueError("--repetitions must be positive")
    if args.timeout <= 0:
        raise ValueError("--timeout must be positive")
    if args.num_threads is not None and args.num_threads < 1:
        raise ValueError("--num-threads must be positive")
    return ExecutionConfig(
        modes=modes,
        backend=args.backend,
        cudagraphs=args.cudagraphs,
        warmups=args.warmups,
        repetitions=args.repetitions,
        timeout_s=args.timeout,
        device=args.device,
        dtype=args.dtype,
        num_threads=args.num_threads,
    )


def _selected_modes(values: list[str]) -> tuple[str, ...]:
    if "all" in values and len(values) != 1:
        raise ValueError("--mode all cannot be combined with another --mode")
    modes = MODES if "all" in values else tuple(values)
    if len(modes) != len(set(modes)):
        raise ValueError("each mode may be requested only once")
    return modes


def _run_scenarios(
    scenarios: list[Scenario], output: Path, check_outputs: bool
) -> tuple[list[dict[str, Any]], list[str]]:
    log_dir = output.with_suffix("").with_name(output.stem + "_logs")
    results = []
    sample_paths: dict[str, Path] = {}
    with tempfile.TemporaryDirectory(prefix="diffusion-benchmark-") as temporary:
        root = Path(temporary)
        for index, scenario in enumerate(scenarios):
            scenario_dir = root / str(index)
            scenario_dir.mkdir()
            result_path = scenario_dir / "result.json"
            sample_path = scenario_dir / "output.pt"
            sample_paths[scenario.scenario_id] = sample_path
            config_path = scenario_dir / "config.json"
            _write_json(
                config_path,
                {
                    "scenario": _scenario_to_dict(scenario),
                    "result_path": str(result_path),
                    "sample_path": str(sample_path) if check_outputs else None,
                },
            )
            cache_dir = scenario_dir / "compiler-cache"
            env = os.environ.copy()
            env.update(
                {
                    "TORCHINDUCTOR_CACHE_DIR": str(cache_dir),
                    "TRITON_CACHE_DIR": str(cache_dir / "triton"),
                    "TORCHINDUCTOR_FX_GRAPH_CACHE": "0",
                    "TORCHINDUCTOR_AUTOGRAD_CACHE": "0",
                    "TORCHINDUCTOR_FX_GRAPH_REMOTE_CACHE": "0",
                    "TORCHINDUCTOR_AUTOGRAD_REMOTE_CACHE": "0",
                    "TORCHINDUCTOR_AUTOTUNE_REMOTE_CACHE": "0",
                    "TORCHINDUCTOR_BUNDLED_AUTOTUNE_REMOTE_CACHE": "0",
                }
            )
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                str(config_path),
            ]
            print(f"Running {scenario.scenario_id} in a fresh worker")
            result = run_worker_process(
                scenario.scenario_id,
                command,
                result_path,
                log_dir / f"{index}-{scenario.mode}.log",
                scenario.execution.timeout_s,
                env,
            )
            result.setdefault("model", dataclasses.asdict(scenario.model))
            result.setdefault("workload", dataclasses.asdict(scenario.workload))
            result.setdefault("execution", dataclasses.asdict(scenario.execution))
            result.setdefault("mode", scenario.mode)
            results.append(result)
            print(f"{scenario.scenario_id}: {result['status']}")
        if check_outputs:
            _compare_outputs(scenarios, results, sample_paths)
    errors = validate_scenario_results(
        [scenario.scenario_id for scenario in scenarios], results
    )
    return results, errors


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=BENCHMARKS)
    parser.add_argument("--mode", choices=(*MODES, "all"), action="append")
    parser.add_argument("--backend", default="inductor")
    parser.add_argument("--cudagraphs", action="store_true")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto"
    )
    parser.add_argument("--num-threads", type=int)
    parser.add_argument("--check-outputs", action="store_true")
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="CSV summary path; JSON records use the same stem with a .json suffix",
    )
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.worker:
        return _worker_main(args.worker)
    if args.model is None or args.mode is None:
        parser.error("--model and --mode are required")
    try:
        csv_output, _ = _output_paths(args.output)
        modes = _selected_modes(args.mode)
        if args.check_outputs and "eager" not in modes:
            raise ValueError("--check-outputs requires --mode eager or --mode all")
        if args.cudagraphs and modes == ("eager",):
            raise ValueError("CUDA graphs require at least one compiled mode")
        recipe = BENCHMARKS[args.model]
        execution = _execution_from_args(args, modes)
        eager_execution = dataclasses.replace(execution, cudagraphs=False)
        scenarios = [
            Scenario(
                recipe.model,
                recipe.workload,
                eager_execution if mode == "eager" else execution,
                mode,
                recipe.loader,
            )
            for mode in modes
        ]
        results, errors = _run_scenarios(scenarios, csv_output, args.check_outputs)
        json_output = write_outputs(csv_output, results)
    except ValueError as error:
        parser.error(str(error))
    print(f"CSV: {args.output}")
    print(f"JSON: {json_output}")
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
