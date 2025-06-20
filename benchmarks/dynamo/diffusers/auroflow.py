import sys
import time

import diffusers
from diffusers import (
    AuraFlowPipeline,
    AuraFlowTransformer2DModel,
    GGUFQuantizationConfig,
)

import torch
from torch._higher_order_ops.invoke_subgraph import mark_compile_region


def compile_full_model(model):
    model.compile(fullgraph=True, mode="reduce-overhead")


def compile_regions(model, nn_modules):
    model.compile_repeated_blocks(fullgraph=True)
    # for submod in model.modules():
    #     if isinstance(submod, nn_modules):
    #         print("Compiling", submod.__class__)
    #         submod.compile(fullgraph=True)


def compile_hierarchical(model, nn_modules):
    for submod in model.modules():
        if isinstance(submod, nn_modules):
            submod.__class__.forward = mark_compile_region(submod.__class__.forward)
    model.compile(fullgraph=True)


def auroflow_benchmark(mode):
    transformer = AuraFlowTransformer2DModel.from_single_file(
        "https://huggingface.co/city96/AuraFlow-v0.3-gguf/blob/main/aura_flow_0.3-Q2_K.gguf",
        quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
        torch_dtype=torch.bfloat16,
    )
    pipeline = AuraFlowPipeline.from_pretrained(
        "fal/AuraFlow-v0.3",
        torch_dtype=torch.bfloat16,
        transformer=transformer,
    ).to("cuda")

    if mode == "full":
        compile_full_model(pipeline.transformer)
    elif mode == "regional":
        compile_regions(
            pipeline.transformer,
            (
                diffusers.models.transformers.auraflow_transformer_2d.AuraFlowSingleTransformerBlock,
                diffusers.models.transformers.auraflow_transformer_2d.AuraFlowJointTransformerBlock,
            ),
        )
    elif mode == "hierarchical":
        compile_hierarchical(
            pipeline.transformer,
            (
                diffusers.models.transformers.auraflow_transformer_2d.AuraFlowSingleTransformerBlock,
                diffusers.models.transformers.auraflow_transformer_2d.AuraFlowJointTransformerBlock,
            ),
        )
    else:
        assert mode == "eager"

    pipeline("A cute pony", width=512, height=512, num_inference_steps=1)

    t0 = time.perf_counter()
    pipeline("A cute pony", width=512, height=512, num_inference_steps=50)
    t1 = time.perf_counter()
    print(t1 - t0)


def wan_benchmark(mode):
    import numpy as np
    from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
    from diffusers.utils import load_image
    from transformers import CLIPVisionModel

    # Available models: Wan-AI/Wan2.1-I2V-14B-480P-Diffusers, Wan-AI/Wan2.1-I2V-14B-720P-Diffusers
    model_id = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
    image_encoder = CLIPVisionModel.from_pretrained(
        model_id, subfolder="image_encoder", torch_dtype=torch.float32
    )
    vae = AutoencoderKLWan.from_pretrained(
        model_id, subfolder="vae", torch_dtype=torch.float32
    )
    pipe = WanImageToVideoPipeline.from_pretrained(
        model_id, vae=vae, image_encoder=image_encoder, torch_dtype=torch.bfloat16
    )

    # replace this with pipe.to("cuda") if you have sufficient VRAM
    # pipe.enable_model_cpu_offload()
    pipe.to("cuda")

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

    num_frames = 33

    if mode == "full":
        compile_full_model(pipe.transformer)
    elif mode == "regional":
        compile_regions(
            pipe.transformer,
            (diffusers.models.transformers.transformer_wan.WanTransformerBlock,),
        )
    elif mode == "hierarchical":
        compile_hierarchical(
            pipe.transformer,
            (diffusers.models.transformers.transformer_wan.WanTransformerBlock,),
        )
    else:
        assert mode == "eager"

    output = pipe(
        image=image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=1,
        guidance_scale=5.0,
    ).frames[0]

    t0 = time.perf_counter()
    output = pipe(
        image=image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=50,
        guidance_scale=5.0,
    ).frames[0]
    t1 = time.perf_counter()
    print(t1 - t0)


def ltx_benchmark(mode):
    from diffusers import LTXConditionPipeline

    import torch

    pipe = LTXConditionPipeline.from_pretrained(
        "Lightricks/LTX-Video-0.9.7-dev", torch_dtype=torch.bfloat16
    )
    pipe.to("cuda")
    pipe.vae.enable_tiling()

    def round_to_nearest_resolution_acceptable_by_vae(height, width):
        height = height - (height % pipe.vae_spatial_compression_ratio)
        width = width - (width % pipe.vae_spatial_compression_ratio)
        return height, width

    prompt = "The video depicts a winding mountain road covered in snow, with a single vehicle traveling along it. The road is flanked by steep, rocky cliffs and sparse vegetation. The landscape is characterized by rugged terrain and a river visible in the distance. The scene captures the solitude and beauty of a winter drive through a mountainous region."
    negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"
    expected_height, expected_width = 512, 704
    downscale_factor = 2 / 3
    num_frames = 121

    # Part 1. Generate video at smaller resolution
    downscaled_height, downscaled_width = (
        int(expected_height * downscale_factor),
        int(expected_width * downscale_factor),
    )
    downscaled_height, downscaled_width = round_to_nearest_resolution_acceptable_by_vae(
        downscaled_height, downscaled_width
    )

    if mode == "full":
        compile_full_model(pipe.transformer)
    elif mode == "regional":
        compile_regions(
            pipe.transformer,
            (diffusers.models.transformers.transformer_ltx.LTXVideoTransformerBlock,),
        )
    elif mode == "hierarchical":
        compile_hierarchical(
            pipe.transformer,
            (diffusers.models.transformers.transformer_ltx.LTXVideoTransformerBlock,),
        )

    latents = pipe(
        conditions=None,
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=downscaled_width,
        height=downscaled_height,
        num_frames=num_frames,
        num_inference_steps=1,
        generator=torch.Generator().manual_seed(0),
        output_type="latent",
    ).frames

    import time

    t0 = time.time()

    latents = pipe(
        conditions=None,
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=downscaled_width,
        height=downscaled_height,
        num_frames=num_frames,
        num_inference_steps=50,
        generator=torch.Generator().manual_seed(0),
        output_type="latent",
    ).frames
    t1 = time.time()
    print(t1 - t0)


def flux_benchmark(mode):
    import diffusers
    from diffusers import FluxPipeline

    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16,
    ).to("cuda")

    prompt = "A cat holding a sign that says hello world"

    if mode == "full":
        compile_full_model(pipe.transformer)
    elif mode == "regional":
        compile_regions(
            pipe.transformer,
            (
                diffusers.models.transformers.transformer_flux.FluxTransformerBlock,
                diffusers.models.transformers.transformer_flux.FluxSingleTransformerBlock,
            ),
        )
    elif mode == "hierarchical":
        compile_hierarchical(
            pipe.transformer,
            (
                diffusers.models.transformers.transformer_flux.FluxTransformerBlock,
                diffusers.models.transformers.transformer_flux.FluxSingleTransformerBlock,
            ),
        )

    pipe(
        prompt,
        height=1024,
        width=1024,
        guidance_scale=3.5,
        num_inference_steps=1,
        max_sequence_length=512,
    )

    t0 = time.perf_counter()
    pipe(
        prompt,
        height=1024,
        width=1024,
        guidance_scale=3.5,
        num_inference_steps=50,
        max_sequence_length=512,
    )
    t1 = time.perf_counter()
    print(t1 - t0)


model_name = sys.argv[1]
mode = sys.argv[2]

if model_name == "auroflow":
    auroflow_benchmark(mode)
elif model_name == "wan":
    wan_benchmark(mode)
elif model_name == "ltx":
    ltx_benchmark(mode)
elif model_name == "flux":
    flux_benchmark(mode)
