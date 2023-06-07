from torch.utils import _pytree as pytree


def patch_non_tensor_outputs(correct_result, new_result, fp64_outputs):
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
    new_result = pytree.tree_flatten(new_result)[0]
    fp64_outputs = pytree.tree_flatten(fp64_outputs)[0]

    return correct_result, new_result, fp64_outputs
