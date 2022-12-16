from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import argparse

"""

# Key problems: Model too large

Not enough RAM to randomly initialize model, and load checkpoint.

- Solution:
    With MetaTensor, lazy load model.
    Is it really meta tensor? How is it done? (device_map or offload_state_dict?)

Not enough GPU memory to do all computation.

- Solution:
    Gradually load model during computation.
    How is it done? (device_map or offload_state_dict?)

Does it work with export (torch.jit.trace)?
"""

sentence = "Question: Can I run BLOOM on a single GPU? Answer:"

# Load model
def load_model(model_name: str = "bigscience/bloom-560m", large_model_support: bool = True):
    large_model_support_kwargs = {
        "device_map": "auto",  # requires `pip install accelerate`
        "offload_state_dict": True,
    } if large_model_support else {}

    print(f"large_model_support_kwargs: {large_model_support_kwargs}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        **large_model_support_kwargs,
    )
    if not large_model_support:
        # manually assign device
        model = model.to(0)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    inputs = tokenizer(sentence, return_tensors="pt").to(0)
    print(inputs.keys())
    return model, inputs, tokenizer


# Inference in PyTorch
def run_model(model, inputs, tokenizer):
    with torch.no_grad():
        outputs = model(**inputs, return_dict=False)

    token_id = outputs[0][0][-1].argmax()
    # token_id = outputs.logits[0][-1].argmax()
    answer = tokenizer.decode([token_id])

    print(f"{sentence}\n{answer}")


# Export to ONNX
def run_onnx_export(model, inputs, tokenizer):
    torch.onnx.export(
        model,
        (inputs["input_ids"], {"attention_mask": inputs["attention_mask"]}),
        "bloom.onnx",
        opset_version=14,
        do_constant_folding=True,
        input_names=["input_ids", "attention_mask"],
        output_names=["output"],
    )

    import onnxruntime

    ort_session = onnxruntime.InferenceSession("bloom.onnx")
    outs = ort_session.run(
        None,
        {
            "input_ids": inputs["input_ids"].cpu().numpy(),
            "attention_mask": inputs["attention_mask"].cpu().numpy(),
        },
    )

    token_id = outs[0][0][-1].argmax()
    answer = tokenizer.decode([token_id])
    print(f"{sentence}\n{answer}")

    # NOTE: The export is correct. Mismatch due to precision issue. More at 'mismatch.ipynb'.
    # outputs = model(**inputs, return_dict=False)
    # for pt_out, ort_out in zip(outputs, outs):
    #     torch.testing.assert_allclose(pt_out, torch.tensor(ort_out, device=pt_out.device))


# Inference in dynamo
def run_dynamo(model, inputs, tokenizer):
    from torch import _dynamo as torchdynamo

    opt_model = torchdynamo.optimize("eager")(model)

    run_model(opt_model, inputs, tokenizer)


# Export to ONNX via dynamo
def run_dynamo_onnx(model, inputs, tokenizer):
    raise NotImplementedError("TODO: Implement this.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="pytorch",
        choices=["pytorch", "onnx", "dynamo", "dynamo_onnx"],
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="bigscience/bloom-560m",
        choices=["bigscience/bloom-560m", "bigscience/bloom"],
    )
    parser.add_argument(
        "--no_large_model_support",
        action="store_true",
        help="Whether to use large model support (device_map or offload_state_dict).",
    )
    args = parser.parse_args()
    mode = args.mode
    model_name = args.model_name
    large_model_support = not args.no_large_model_support

    model, inputs, tokenizer = load_model(model_name, large_model_support)

    if mode == "pytorch":
        run_model(model, inputs, tokenizer)
    elif mode == "onnx":
        run_onnx_export(model, inputs, tokenizer)
    elif mode == "dynamo":
        run_dynamo(model, inputs, tokenizer)
    elif mode == "dynamo_onnx":
        run_dynamo_onnx(model, inputs, tokenizer)
    else:
        raise ValueError(f"Unknown mode: {mode}")
