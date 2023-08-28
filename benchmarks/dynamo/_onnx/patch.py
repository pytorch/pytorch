import torch
from torch.utils import _pytree as pytree


def patch_non_tensor_outputs(correct_result, new_result, fp64_outputs):
    """Patch non-tensor outputs to make them comparable with the correct result.

    ONNX model always returns a flat tuple of tensors, but the PyTorch model outputs
    `correct_result` and `fp64_outputs` can be arbitrary types. This function normalizes
    the outputs to make them comparable with the ONNX model output.
    """
    try:
        from transformers import modeling_outputs
    except ImportError:
        has_transformers = False
    else:
        has_transformers = True

    if has_transformers and isinstance(correct_result, modeling_outputs.ModelOutput):
        correct_result = correct_result.to_tuple()
        fp64_outputs = fp64_outputs.to_tuple() if fp64_outputs is not None else None
    elif type(correct_result).__name__ in (
        "MaskedLMOutput",
        "Seq2SeqLMOutput",
        "CausalLMOutputWithCrossAttentions",
        "LongformerMaskedLMOutput",
        "Instances",
        "SquashedNormal",
        "Boxes",
        "Normal",
        "TanhTransform",
        "Foo",
        "Variable",
    ):
        # Copied from `same` function in `torch._dynamo.utils`
        correct_result = [
            value
            for key in correct_result.__dict__.keys()
            if (value := getattr(correct_result, key)) is not None
        ]
        fp64_outputs = (
            [
                value
                for key in fp64_outputs.__dict__.keys()
                if (value := getattr(fp64_outputs, key)) is not None
            ]
            if fp64_outputs is not None
            else None
        )

    # Flatten nested tuple of tensors, i.e. past_key_values
    correct_result = pytree.tree_flatten(correct_result)[0]
    # Hack to put results from different runs on same device.
    # This is needed for ONNX CPU fallback benchmark, where PyTorch eager is run on GPU.
    # Assuming outputs from a single run are always on same device!
    devices = [x.device for x in correct_result if isinstance(x, torch.Tensor)]
    assert devices and all(x == devices[0] for x in devices), "All tensors must be on same device!"
    device = devices[0]
    new_result = pytree.tree_flatten(new_result)[0]
    new_result = pytree.tree_map(lambda x: x.to(device=device) if isinstance(x, torch.Tensor) else x, new_result)
    fp64_outputs = pytree.tree_flatten(fp64_outputs)[0]

    return correct_result, new_result, fp64_outputs
