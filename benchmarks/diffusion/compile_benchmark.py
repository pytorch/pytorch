"""Diffusion benchmark components and the legacy pipeline command line."""

from __future__ import annotations

import argparse
import dataclasses
import time
from typing import Any, TYPE_CHECKING

import torch


if TYPE_CHECKING:
    from collections.abc import Callable


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
