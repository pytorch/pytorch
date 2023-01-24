# Import generic wrappers
import torch
import torch._dynamo
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.onnx._internal import fx as fx_onnx
from transformers import AutoModel, AutoTokenizer


model_name = "sshleifer/tiny-gpt2"
ftm = FakeTensorMode(allow_non_fake_inputs=True, allow_fallback_kernels=False)
with ftm:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    inputs = tokenizer("Hello world!", return_tensors="pt")
    model = AutoModel.from_pretrained(model_name)
    outputs = model(**inputs)
    (
        onnx_model,
        graph_module,
        bound_args,
        replaced_attrs,
    ) = fx_onnx.export_without_parameters_and_buffers(
        model, use_binary_format=False, **inputs
    )


def test_one(graph_module, onnx_model, bound_args, replaced_attrs):
    import itertools

    pth_args = []
    for t in itertools.chain(bound_args, replaced_attrs):
        if t is not None:
            if t.dtype == torch.float32:
                real_t = torch.randn(t.shape, dtype=t.dtype, device="cpu")
            elif t.dtype in (torch.int64, torch.uint8):
                real_t = torch.randint(0, 3, t.shape, dtype=t.dtype, device="cpu")
            else:
                raise RuntimeError(f"Unsupported dtype {t.dtype}")
        else:
            real_t = None
        pth_args.append(real_t)
    pth_out = graph_module(*pth_args)

    import numpy as np
    import onnx
    import onnxruntime as ort

    onnx.save(onnx_model, "model_tiny_gpt.onnx")
    ort_sess = ort.InferenceSession(
        "model_tiny_gpt.onnx",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    ort_input_dict = {}
    for ort_input, t in zip(
        onnx_model.graph.input, [arg for arg in pth_args if arg is not None]
    ):
        ort_input_dict[ort_input.name] = t.numpy()
    ort_out = ort_sess.run(None, ort_input_dict)

    np.testing.assert_allclose(
        ort_out[0], pth_out["last_hidden_state"].numpy(), atol=1e-4, rtol=1e-3
    )
    for ort_value, pth_value in zip(
        ort_out[1:], pth_out["past_key_values"][0] + pth_out["past_key_values"][1]
    ):
        np.testing.assert_allclose(ort_value, pth_value.numpy(), atol=1e-4, rtol=1e-3)


for i in range(10):
    print(f"Test {i}")
    test_one(graph_module, onnx_model, bound_args, replaced_attrs)
