"""
Compile benchmarks for diffusion model transformers.

These models have large numbers of repeated transformer blocks, leading to high
cold-start compile times under full-model compilation. This makes them good
candidates for regional/hierarchical compilation strategies that reduce compile
time by compiling individual blocks instead of the full model.

But if you are looking to reduce the cold start time for full models, these are
excellent candidates.

Example:
    python compile_benchmark.py --model auroflow --mode full
    python compile_benchmark.py --model flux --mode full --backend eager

If you see issues downloading models, try: pip uninstall hf_xet
"""

import argparse
import time

import diffusers
from diffusers import (
    AuraFlowPipeline,
    AuraFlowTransformer2DModel,
    GGUFQuantizationConfig,
)

import torch
from torch._higher_order_ops.invoke_subgraph import mark_compile_region


def compile_model(model, mode, block_types, backend="inductor"):
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


def auroflow_benchmark(mode, backend="inductor"):
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
    )  # noqa: B950
    negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"  # noqa: B950

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
    from diffusers import LTXConditionPipeline

    pipe = LTXConditionPipeline.from_pretrained(
        "Lightricks/LTX-Video-0.9.7-dev", torch_dtype=torch.bfloat16
    ).to("cuda")
    pipe.vae.enable_tiling()

    height = 512 - (512 % pipe.vae_spatial_compression_ratio)
    width = 704 - (704 % pipe.vae_spatial_compression_ratio)

    prompt = "The video depicts a winding mountain road covered in snow, with a single vehicle traveling along it. The road is flanked by steep, rocky cliffs and sparse vegetation. The landscape is characterized by rugged terrain and a river visible in the distance. The scene captures the solitude and beauty of a winter drive through a mountainous region."  # noqa: B950
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


BENCHMARKS = {
    "auroflow": auroflow_benchmark,
    "wan": wan_benchmark,
    "ltx": ltx_benchmark,
    "flux": flux_benchmark,
}

MODES = ("eager", "full", "regional", "hierarchical")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=BENCHMARKS, required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--backend", default="inductor")
    args = parser.parse_args()
    BENCHMARKS[args.model](args.mode, args.backend)
