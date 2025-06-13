import time

import diffusers
from diffusers import FluxPipeline

import torch
from torch._higher_order_ops.invoke_subgraph import mark_compile_region


pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16,
).to("cuda")

prompt = "A cat holding a sign that says hello world"


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
compile_setting = "full"

if compile_setting == "full":
    compile_full_model(pipe.transformer)
elif compile_setting == "regional":
    compile_regions(
        pipe.transformer,
        (
            diffusers.models.transformers.transformer_flux.FluxTransformerBlock,
            diffusers.models.transformers.transformer_flux.FluxSingleTransformerBlock,
        ),
    )
elif compile_setting == "hierarchical":
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
