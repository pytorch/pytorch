import time

import diffusers
from diffusers import (
    AuraFlowPipeline,
    AuraFlowTransformer2DModel,
    GGUFQuantizationConfig,
)

import torch
from torch._higher_order_ops.invoke_subgraph import mark_compile_region


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

# pipeline.transformer = torch.compile(pipeline.transformer, fullgraph=True)


def compile_full_model(model):
    model.compile(fullgraph=True)


def compile_regions(model, nn_modules):
    for submod in model.modules():
        if isinstance(submod, nn_modules):
            print("Compiling", submod.__class__)
            submod.compile(fullgraph=True)


def compile_hierarchical(model, nn_modules):
    for submod in model.modules():
        if isinstance(submod, nn_modules):
            submod.__class__.forward = mark_compile_region(submod.__class__.forward)
    model.compile(fullgraph=True)


# compile_full_model(pipeline.transformer)

compile_regions(
    pipeline.transformer,
    (
        diffusers.models.transformers.auraflow_transformer_2d.AuraFlowSingleTransformerBlock,
        diffusers.models.transformers.auraflow_transformer_2d.AuraFlowJointTransformerBlock,
    ),
)

# compile_hierarchical(
#     pipeline.transformer,
#     (
#         diffusers.models.transformers.auraflow_transformer_2d.AuraFlowSingleTransformerBlock,
#         diffusers.models.transformers.auraflow_transformer_2d.AuraFlowJointTransformerBlock,
#     ),
# )

pipeline("A cute pony", width=512, height=512, num_inference_steps=1)

t0 = time.perf_counter()
pipeline("A cute pony", width=512, height=512, num_inference_steps=50)
t1 = time.perf_counter()
print(t1 - t0)
# breakpoint()
