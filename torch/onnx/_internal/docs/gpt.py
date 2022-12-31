# Import generic wrappers
import numpy as np
import onnx
import onnxruntime
import torch
import transformers
from torch import _dynamo as torchdynamo
from torch.utils._pytree import tree_flatten
from transformers import AutoModel, AutoTokenizer

try:
    from onnxruntime.capi import _pybind_state as ORTC
    from onnxruntime.training.torchdynamo.ort_backend import _get_onnx_devices

    HAS_ORT_TRAINING = True
except ImportError:
    HAS_ORT_TRAINING = False

# Define the model repo
model_name = "sshleifer/tiny-gpt2"


_NP_DTYPE = {
    torch.float16: np.float16,
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.uint8: np.uint8,
    torch.int8: np.int8,
    torch.int16: np.int16,
    torch.int32: np.int32,
    torch.int64: np.longlong,
    torch.bool: np.bool_,
}


def create_ort_tensors(pytorch_tensors):
    ort_tensors = ORTC.OrtValueVector()  # type: ignore
    ort_tensors.reserve(len(pytorch_tensors))

    dtypes = []
    shapes = []
    data_ptrs = []
    for value in pytorch_tensors:
        dtypes.append(_NP_DTYPE[value.dtype])
        shapes.append(value.size())
        data_ptrs.append(value.data_ptr())

    ort_devices = _get_onnx_devices(pytorch_tensors)
    ort_tensors.push_back_batch(pytorch_tensors, data_ptrs, dtypes, shapes, ort_devices)
    return ort_tensors


def create_ort_devices(pytorch_outputs):
    return _get_onnx_devices(pytorch_outputs)


def run_ort(onnx_model, onnx_model_text, pytorch_inputs, pytorch_outputs):
    if HAS_ORT_TRAINING:
        input_names = [v.name for v in onnx_model_text.graph.input]
        ort_inputs = create_ort_tensors(pytorch_inputs)

        run_options = onnxruntime.RunOptions()
        run_options.synchronize_execution_providers = True

        output_names = [v.name for v in onnx_model_text.graph.output]
        ort_outputs = ORTC.OrtValueVector()
        output_devices = create_ort_devices(pytorch_outputs)

        sess = onnxruntime.InferenceSession(
            onnx_model, providers=["CPUExecutionProvider"]
        )
        sess.run_with_ortvaluevector(
            run_options,
            input_names,
            ort_inputs,
            output_names,
            ort_outputs,
            output_devices,
        )
        return onnxruntime.training.ortmodule._utils._ortvalues_to_torch_tensor(
            ort_outputs
        )
    else:
        sess = onnxruntime.InferenceSession(
            onnx_model, providers=["CPUExecutionProvider"]
        )
        input_names = [v.name for v in onnx_model_text.graph.input]
        return sess.run(
            None, {k: v.cpu().numpy() for k, v in zip(input_names, pytorch_inputs)}
        )


def test_gpt2_one_shot(model_name):
    # Download pytorch model
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Transform input tokens
    inputs = tokenizer("Hello world!", return_tensors="pt")

    # Model apply
    outputs = model(**inputs)
    flat_outputs, _ = tree_flatten(outputs)

    # Export tiny GPT2.
    from torch.onnx._internal._fx import export, export_without_kwargs

    # onnx_model = export(model, **inputs)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    onnx_model_text = export_without_kwargs(model, **inputs, use_binary_format=False)
    onnx.save(onnx_model_text, "gpt2.onnx")
    onnx_model = export_without_kwargs(model, **inputs, use_binary_format=True)

    ref_outputs, _ = tree_flatten(model(**inputs, return_dict=False))
    pth_outputs = run_ort(
        onnx_model, onnx_model_text, (input_ids, attention_mask), ref_outputs
    )
    for _1, _2 in zip(ref_outputs, pth_outputs):
        print(_1 - _2)
        assert torch.allclose(_1, _2)


def test_gpt2_auto_regressive(model_name):
    # NOTE: auto regressive uses generation algorithms such as greedy search or beam
    # search that involves loops and control flows.

    model = transformers.GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    input_ids = inputs["input_ids"]

    # Transform input tokens
    inputs = tokenizer("Hello world!", return_tensors="pt")

    (
        explanation,
        out_guards,
        graphs,
        ops_per_graph,
        break_reasons,
        explanation_verbose,
    ) = torchdynamo.explain(model.generate, input_ids)

    print(explanation_verbose)


# test_gpt2_auto_regressive(model_name)
test_gpt2_one_shot(model_name)
