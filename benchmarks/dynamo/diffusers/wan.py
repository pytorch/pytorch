import time

import diffusers
import numpy as np
from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
from diffusers.utils import load_image
from transformers import CLIPVisionModel

import torch
from torch._higher_order_ops.invoke_subgraph import mark_compile_region


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
compile_setting = "hierarchical"

if compile_setting == "full":
    compile_full_model(pipe.transformer)
elif compile_setting == "regional":
    compile_regions(
        pipe.transformer,
        (diffusers.models.transformers.transformer_wan.WanTransformerBlock,),
    )
elif compile_setting == "hierarchical":
    compile_hierarchical(
        pipe.transformer,
        (diffusers.models.transformers.transformer_wan.WanTransformerBlock,),
    )


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
