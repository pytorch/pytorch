"""Diffusion benchmark components and the legacy pipeline command line."""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import importlib.metadata
import math
import os
import platform
import statistics
import time
from pathlib import Path
from typing import Any, TYPE_CHECKING

import torch


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


MODES = ("eager", "full", "regional", "hierarchical")
OUTPUT_BOUNDARIES = ("latent", "postprocessed_output")
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
    backend: str
    cudagraphs: bool
    warmups: int
    repetitions: int
    timeout_s: float
    device: str
    dtype: str
    num_threads: int | None


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
                # The diffusers Wan I2V example's area-preserving resize of the
                # pinned 1184x880 input to a multiple of 16.
                "height": 544,
                "width": 720,
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
            dependencies=("torch", "numpy"),
            artifacts=(),
        ),
    }
    return recipes


BENCHMARKS = _recipes()


def _dtype_from_name(name: str) -> Any:
    import torch

    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def _device_and_dtype_names(scenario: Scenario) -> tuple[str, str]:
    execution, model = scenario.execution, scenario.model
    device = model.default_device if execution.device == "auto" else execution.device
    dtype = model.default_dtype if execution.dtype == "auto" else execution.dtype
    return device, dtype


def _resolve_device_and_dtype(scenario: Scenario) -> tuple[str, str, Any]:
    import torch

    device, dtype_name = _device_and_dtype_names(scenario)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"{scenario.model.name} requires CUDA, but CUDA is unavailable"
        )
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
    mutable_parameters: dict[str, Callable[[], Any]] | None = None,
    output_fields: tuple[str, ...] = ("images", "frames"),
) -> LoadedPipeline:
    pipeline = pipeline.to(device)
    if hasattr(pipeline, "set_progress_bar_config"):
        pipeline.set_progress_bar_config(disable=True)
    for module in pipeline.components.values():
        if hasattr(module, "eval"):
            module.eval()
    return LoadedPipeline(
        pipeline=pipeline,
        component=getattr(pipeline, scenario.model.component_attr),
        make_request=_fresh_request(
            pipeline,
            scenario.workload.parameters,
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
        paths[artifact.key], torch_dtype=dtype, safety_checker=None
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
    parameters = scenario.workload.parameters
    image = image.resize((parameters["width"], parameters["height"]))
    return _finish_loading(
        pipeline,
        scenario,
        device,
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
    return _finish_loading(pipeline, scenario, device, output_fields=("frames",))


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


def _uses_accelerator(device: str) -> bool:
    import torch

    accelerator = torch.accelerator.current_accelerator()
    return accelerator is not None and torch.device(device).type == accelerator.type


def _synchronize(device: str) -> None:
    import torch

    if _uses_accelerator(device):
        torch.accelerator.synchronize(device)


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
    if isinstance(value, (list, tuple)) and value:
        tensors = [_extract_tensor(item, fields) for item in value]
        return torch.stack(tensors)
    # Only postprocessed pipeline outputs are images, so Pillow stays optional.
    from PIL import Image

    if isinstance(value, Image.Image):
        return torch.from_numpy(np.asarray(value).copy())
    raise TypeError(f"cannot extract a tensor from output type {type(value).__name__}")


def _tensor_summary(tensor: Any) -> dict[str, Any]:
    import torch

    contiguous = tensor.contiguous()
    byte_view = contiguous.view(dtype=torch.uint8)
    digest = hashlib.sha256(byte_view.numpy().tobytes()).hexdigest()
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype), "sha256": digest}


def _device_memory(device: str) -> dict[str, int | None]:
    import torch

    if not _uses_accelerator(device):
        return {"allocated_bytes": None, "reserved_bytes": None}
    return {
        "allocated_bytes": torch.accelerator.max_memory_allocated(device),
        "reserved_bytes": torch.accelerator.max_memory_reserved(device),
    }


def _reset_device_memory(device: str) -> None:
    import torch

    if _uses_accelerator(device):
        torch.accelerator.reset_peak_memory_stats(device)


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
    }
    if mode != "eager" and unique_graphs == 0:
        raise RuntimeError("torch.compile produced no compiled graphs")
    if mode == "eager" and unique_graphs != 0:
        raise RuntimeError("eager scenario unexpectedly compiled a graph")
    # Without a traced nested region, hierarchical mode is just full compilation.
    nested = "invoke_subgraph_trace" in compilation_time_metrics
    if mode == "hierarchical" and not nested:
        raise RuntimeError("hierarchical compilation produced no nested region")
    if cudagraphs:
        from torch._inductor.cudagraph_trees import get_manager

        # Inductor skips CUDA graphs per graph, at compile time or by falling back
        # to eager at runtime, so captures elsewhere do not prove full coverage.
        skips = counters["inductor"]["cudagraph_skips"]
        if skips:
            raise RuntimeError(
                f"CUDA graphs were requested, but Inductor skipped them {skips} "
                "time(s); the worker log has the reasons"
            )
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
    device_type = torch.device(device).type
    info["accelerator"] = device_type
    if device_type in ("cuda", "xpu"):
        props = getattr(torch, device_type).get_device_properties(device)
        info.update(accelerator=props.name, accelerator_memory_bytes=props.total_memory)
    if device_type == "cuda":
        info.update(
            {
                "compute_capability": f"{props.major}.{props.minor}",
                "cuda_compiled_version": torch.version.cuda,
                "hip_compiled_version": torch.version.hip,
            }
        )
    return info


def _relevant_environment() -> dict[str, str]:
    # Only record known tuning options; compiler-prefixed variables can contain
    # credentials (for example, a remote cache URL).
    names = (
        "CUDA_VISIBLE_DEVICES",
        "MKL_NUM_THREADS",
        "OMP_NUM_THREADS",
        "PYTORCH_ALLOC_CONF",
        "PYTORCH_CUDA_ALLOC_CONF",
        "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
        "TORCHINDUCTOR_COMPILE_THREADS",
        "TORCHINDUCTOR_FX_GRAPH_CACHE",
        "TORCHINDUCTOR_MAX_AUTOTUNE",
        "TORCHINDUCTOR_MAX_AUTOTUNE_GEMM",
        "TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS",
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
    timeout = scenario.execution.timeout_s
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be finite and positive")
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
    from torch._dynamo.utils import (
        calculate_time_spent,
        compilation_time_metrics,
        counters,
    )
    from torch._functorch import config as functorch_config
    from torch._inductor import config as inductor_config

    device, dtype_name, dtype = _resolve_device_and_dtype(scenario)
    _validate_worker_scenario(scenario, device)
    cudagraphs = scenario.execution.cudagraphs
    if scenario.execution.num_threads is not None:
        torch.set_num_threads(scenario.execution.num_threads)

    paths = _prefetch(scenario.model.artifacts)
    # Start the device runtime outside the setup timer; synchronizing and
    # resetting peaks are no-ops on an uninitialized indexed device ("cuda:0").
    torch.empty(0, device=device)
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
            # Exclude compilation, autotuning, and CUDA graph recording transients.
            _reset_device_memory(device)
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
        compile_time_s = calculate_time_spent()["total_wall_time"]

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
        "model_setup_s": setup_s,
        "compile_wrapper_setup_s": compile_setup_s,
        "compile_time_s": compile_time_s,
        "first_request_s": first_request_s,
        "steady_state_samples_s": samples,
        "steady_state_median_s": median_s,
        "steady_state_mad_s": mad_s,
        "setup_device_peak": setup_memory,
        "request_device_peak": request_memory,
        "device_memory_definition": (
            "allocator allocated/reserved process peaks; request peaks cover "
            "steady-state requests and include resident model state"
            if _uses_accelerator(device)
            else "unavailable on CPU"
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
        "compiler_caches": {
            "force_disabled": inductor_config.force_disable_caches,
            "fx_graph": inductor_config.fx_graph_cache,
            "fx_graph_remote": inductor_config.fx_graph_remote_cache,
            "autograd": functorch_config.enable_autograd_cache,
            "autograd_remote": functorch_config.enable_remote_autograd_cache,
            "autotune_local": inductor_config.autotune_local_cache,
            "autotune_remote": inductor_config.autotune_remote_cache,
            "bundled_autotune_remote": inductor_config.bundled_autotune_remote_cache,
            "dynamo_pgo_remote": torch._dynamo.config.automatic_dynamic_remote_pgo,
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
    return Scenario(
        model=ModelConfig(**model_value),
        workload=WorkloadConfig(**value["workload"]),
        execution=ExecutionConfig(**value["execution"]),
        mode=value["mode"],
        loader=value["loader"],
    )


def compile_model(model, mode, block_types, backend="inductor"):
    from torch._higher_order_ops.invoke_subgraph import mark_compile_region

    if mode == "full":
        model.compile(backend=backend, fullgraph=True, mode="reduce-overhead")
    elif mode == "regional":
        for submod in model.modules():
            if isinstance(submod, block_types):
                print("Compiling", submod.__class__)
                submod.compile(backend=backend, fullgraph=True)
    elif mode == "hierarchical":
        for submod in model.modules():
            if isinstance(submod, block_types):
                submod.__class__.forward = mark_compile_region(submod.__class__.forward)
        model.compile(backend=backend, fullgraph=True)


def bench(run_fn, warmup_steps=1, bench_steps=50):
    run_fn(warmup_steps)
    t0 = time.perf_counter()
    run_fn(bench_steps)
    t1 = time.perf_counter()
    print(f"{t1 - t0:.3f}s")


def auraflow_benchmark(mode, backend="inductor"):
    import diffusers
    from diffusers import (
        AuraFlowPipeline,
        AuraFlowTransformer2DModel,
        GGUFQuantizationConfig,
    )

    transformer = AuraFlowTransformer2DModel.from_single_file(
        "https://huggingface.co/city96/AuraFlow-v0.3-gguf/blob/main/aura_flow_0.3-Q2_K.gguf",
        quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
        torch_dtype=torch.bfloat16,
    )
    pipe = AuraFlowPipeline.from_pretrained(
        "fal/AuraFlow-v0.3",
        torch_dtype=torch.bfloat16,
        transformer=transformer,
    ).to("cuda")

    block_types = (
        diffusers.models.transformers.auraflow_transformer_2d.AuraFlowSingleTransformerBlock,
        diffusers.models.transformers.auraflow_transformer_2d.AuraFlowJointTransformerBlock,
    )
    compile_model(pipe.transformer, mode, block_types, backend)

    def run(steps):
        pipe("A cute pony", width=512, height=512, num_inference_steps=steps)

    bench(run)


def wan_benchmark(mode, backend="inductor"):
    import diffusers
    import numpy as np
    from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
    from diffusers.utils import load_image
    from transformers import CLIPVisionModel

    model_id = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
    image_encoder = CLIPVisionModel.from_pretrained(
        model_id, subfolder="image_encoder", torch_dtype=torch.float32
    )
    vae = AutoencoderKLWan.from_pretrained(
        model_id, subfolder="vae", torch_dtype=torch.float32
    )
    pipe = WanImageToVideoPipeline.from_pretrained(
        model_id, vae=vae, image_encoder=image_encoder, torch_dtype=torch.bfloat16
    ).to("cuda")

    image = load_image(
        "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg"
    )
    max_area = 480 * 832
    aspect_ratio = image.height / image.width
    mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
    height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
    width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
    image = image.resize((width, height))

    prompt = (
        "An astronaut hatching from an egg, on the surface of the moon, the darkness and depth of space realised in "
        "the background. High quality, ultrarealistic detail and breath-taking movie-like camera shot."
    )
    negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

    block_types = (diffusers.models.transformers.transformer_wan.WanTransformerBlock,)
    compile_model(pipe.transformer, mode, block_types, backend)

    def run(steps):
        pipe(
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=33,
            num_inference_steps=steps,
            guidance_scale=5.0,
        )

    bench(run)


def ltx_benchmark(mode, backend="inductor"):
    import diffusers
    from diffusers import LTXConditionPipeline

    pipe = LTXConditionPipeline.from_pretrained(
        "Lightricks/LTX-Video-0.9.7-dev", torch_dtype=torch.bfloat16
    ).to("cuda")
    pipe.vae.enable_tiling()

    height = 512 - (512 % pipe.vae_spatial_compression_ratio)
    width = 704 - (704 % pipe.vae_spatial_compression_ratio)

    prompt = "The video depicts a winding mountain road covered in snow, with a single vehicle traveling along it. The road is flanked by steep, rocky cliffs and sparse vegetation. The landscape is characterized by rugged terrain and a river visible in the distance. The scene captures the solitude and beauty of a winter drive through a mountainous region."
    negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"

    block_types = (
        diffusers.models.transformers.transformer_ltx.LTXVideoTransformerBlock,
    )
    compile_model(pipe.transformer, mode, block_types, backend)

    def run(steps):
        pipe(
            conditions=None,
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_frames=121,
            num_inference_steps=steps,
            generator=torch.Generator().manual_seed(0),
            output_type="latent",
        )

    bench(run)


def flux_benchmark(mode, backend="inductor"):
    import diffusers
    from diffusers import FluxPipeline

    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16,
    ).to("cuda")

    prompt = "A cat holding a sign that says hello world"

    block_types = (
        diffusers.models.transformers.transformer_flux.FluxTransformerBlock,
        diffusers.models.transformers.transformer_flux.FluxSingleTransformerBlock,
    )
    compile_model(pipe.transformer, mode, block_types, backend)

    def run(steps):
        pipe(
            prompt,
            height=1024,
            width=1024,
            guidance_scale=3.5,
            num_inference_steps=steps,
            max_sequence_length=512,
        )

    bench(run)


LEGACY_BENCHMARKS = {
    "auraflow": auraflow_benchmark,
    "wan": wan_benchmark,
    "ltx": ltx_benchmark,
    "flux": flux_benchmark,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=LEGACY_BENCHMARKS, required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--backend", default="inductor")
    args = parser.parse_args()
    LEGACY_BENCHMARKS[args.model](args.mode, args.backend)
