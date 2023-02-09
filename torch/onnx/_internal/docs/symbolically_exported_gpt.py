# Import generic wrappers
import numpy as np
import onnx
import onnxruntime  # type: ignore[import]
import torch
import torch._dynamo
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.onnx._internal import fx as fx_onnx, diagnostics
from transformers import AutoModel, AutoTokenizer  # type: ignore[import]

model_name = "sshleifer/tiny-gpt2"
ftm = FakeTensorMode(allow_non_fake_inputs=True, allow_fallback_kernels=False)
ctx = fx_onnx.FxToOnnxContext()
with ftm, ctx:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    inputs = tokenizer("Hello world!", return_tensors="pt")
    model = AutoModel.from_pretrained(model_name)

    outputs = model(**inputs)
    with diagnostics.engine.create_diagnostic_context(
        "fx-exporter", version=torch.__version__
    ) as diag_ctx:
        (
            onnx_model,
            graph_module,
            bound_args,
            replaced_attrs,
        ) = fx_onnx.export_without_parameters_and_buffers(
            model, use_binary_format=False, **inputs
        )

    diagnostics.engine.dump("report_symbolic_export_gpt2.sarif")

onnx.save(onnx_model, "gpt_stateless.onnx")
fx_onnx.save_model_with_external_data(
    ".", "gpt_external_data.onnx", "gpt_initializers", tuple(ctx.paths), onnx_model
)


def create_real_arguments(*args):
    real_args = []
    for arg in args:
        if arg is not None:
            assert isinstance(arg, torch.Tensor)
            if arg.dtype == torch.float32:
                real_t = torch.randn(arg.shape, dtype=arg.dtype, device="cpu")
            elif arg.dtype in (torch.int64, torch.uint8):
                real_arg = torch.randint(0, 3, arg.shape, dtype=arg.dtype, device="cpu")
            else:
                raise RuntimeError(f"Unsupported dtype {arg.dtype}")
        else:
            real_arg = None
        real_args.append(real_arg)
    return real_args


def test_external_data(baseline_model, onnx_model_path, example_inputs):
    # Generate real tensors from FakeTensors.
    real_inputs = create_real_arguments(*example_inputs)
    real_outputs = baseline_model(*real_inputs)

    # ort_sess = onnxruntime.InferenceSession(
    #     onnx_model_path,
    #     providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    # )
    import onnx.reference

    ort_sess = onnx.reference.ReferenceEvaluator(onnx_model_path, verbose=True)
    onnx_model = onnx.load(onnx_model_path)
    initializer_names = set([init.name for init in onnx_model.graph.initializer])
    ort_input_dict = {}
    for ort_input, t in zip(
        [
            input
            for input in onnx_model.graph.input
            if input.name not in initializer_names
        ],
        [arg for arg in real_inputs if arg is not None],
    ):
        ort_input_dict[ort_input.name] = t.numpy()
    ort_out = ort_sess.run(None, ort_input_dict)

    print("\nStart validation!")
    np.testing.assert_allclose(
        ort_out[0],
        real_outputs["last_hidden_state"].detach().numpy(),
        atol=1e-4,
        rtol=1e-3,
    )
    print("\nort_out[0]: ", ort_out[0])
    print(
        "\nreal_outputs['last_hidden_state']: ",
        real_outputs["last_hidden_state"].detach().numpy(),
    )

    print("\nDone validation on last_hidden_state!")
    for ort_value, pth_value in zip(
        ort_out[1:],
        real_outputs["past_key_values"][0] + real_outputs["past_key_values"][1],
    ):
        np.testing.assert_allclose(
            ort_value, pth_value.detach().numpy(), atol=1e-4, rtol=1e-3
        )
    print("\nDone validation on past_key_values!")


model = AutoModel.from_pretrained(model_name)
for i in range(10):
    test_external_data(model, "gpt_external_data.onnx", bound_args)
