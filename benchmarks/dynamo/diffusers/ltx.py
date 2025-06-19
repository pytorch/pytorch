import diffusers
from diffusers import LTXConditionPipeline

import torch
from torch._higher_order_ops.invoke_subgraph import mark_compile_region


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


def compile_full_model(model):
    model.compile(fullgraph=True)


def compile_regions(model, nn_modules):
    for submod in model.modules():
        if isinstance(submod, nn_modules):
            submod.compile(fullgraph=True)


def compile_hierarchical(model, nn_modules):
    for submod in model.modules():
        if isinstance(submod, nn_modules):
            submod.__class__.forward = mark_compile_region(submod.__class__.forward)
    model.compile(fullgraph=True)


# full or regional or hierarchical
# For diffusers, it seems regional is the best option
compile_setting = None

if compile_setting == "full":
    compile_full_model(pipe.transformer)
elif compile_setting == "regional":
    compile_regions(
        pipe.transformer,
        (diffusers.models.transformers.transformer_ltx.LTXVideoTransformerBlock,),
    )
elif compile_setting == "hierarchical":
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
