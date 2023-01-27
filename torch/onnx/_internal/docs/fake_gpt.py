# Import generic wrappers
import numpy as np
import onnx
import onnxruntime  # type: ignore[import]
import torch
import torch._dynamo
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.onnx._internal import fx as fx_onnx
from transformers import AutoModel, AutoTokenizer  # type: ignore[import]
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import os

def create_tensor_proto_with_external_data(tensor: torch.Tensor, name: str, location: str, basepath: str):
    """ Create a TensorProto with external data from a PyTorch tensor.
    The external data is saved in os.path.join(basepath, location).

    Args:
        tensor: Tensor to be saved.
        name: Name of the tensor (i.e., initializer name in ONNX graph).
        location: Relative location of the external data file (e.g., "/tmp/initializers/weight_0" when model is "/tmp/model.onnx").
        basepath: Base path of the external data file (e.g., "/tmp/external_data").


    Reference:
        How to load? https://github.com/onnx/onnx/blob/5dac81ac0707bdf88f56c35c0a5e8855d3534673/onnx/external_data_helper.py#L187
        How to save? https://github.com/onnx/onnx/blob/5dac81ac0707bdf88f56c35c0a5e8855d3534673/onnx/external_data_helper.py#L43
        How to set ONNX fields? https://github.com/onnx/onnx/blob/5dac81ac0707bdf88f56c35c0a5e8855d3534673/onnx/external_data_helper.py#L88
    """
    tensor_proto = onnx.TensorProto()
    tensor_proto.name = name
    tensor_proto.data_type = torch.onnx._type_utils._SCALAR_TYPE_TO_ONNX[torch.onnx._type_utils._DTYPE_TO_SCALAR_TYPE[tensor.dtype]]
    tensor_proto.dims.extend(tensor.shape)
    tensor_proto.data_location = onnx.TensorProto.EXTERNAL

    # Settings for saving one tensor per file.
    # Offset is zero because there is no other tensor in the same file.
    key_value_pairs = {
        "location": location,
        "offset": 0,
        "length": tensor.untyped_storage().nbytes(),
    }
    for k, v in key_value_pairs.items():
        entry = tensor_proto.external_data.add()
        entry.key = k
        entry.value = str(v)

    # Actual path to write content of tensor.
    external_data_file_path = os.path.join(basepath, location)
    if os.path.exists(external_data_file_path):
        os.remove(external_data_file_path)

    # Create external data's folder if not exists.
    external_data_dir_path = os.path.dirname(external_data_file_path)
    if not os.path.exists(external_data_dir_path):
        # if the demo_folder directory is not present 
        # then create it.
        os.makedirs(external_data_dir_path)

    # Create a fresh file.
    with open(external_data_file_path, "xb") as data_file:
        # No need to call "seek" because offset is 0.
        # data_file.seek(0)
        # Write tensor content to the file.
        data_file.write(tensor.numpy().tobytes())

    return tensor_proto


original_load = torch.load

tensor_paths = []

def load_wrapper(
    f,
    *args,
    **kwargs,
):
    setattr(torch, "load", original_load)
    tensor_paths.append(f)
    result = torch.load(f, *args, **kwargs)
    setattr(torch, "load", load_wrapper)
    return result

setattr(torch, "load", load_wrapper)

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

onnx.save(onnx_model, "gpt_stateless.onnx")

setattr(torch, "load", original_load)

onnx_model_with_initializers = onnx.ModelProto()
onnx_model_with_initializers.CopyFrom(onnx_model)
onnx_input_names = [input.name for input in onnx_model.graph.input]
print(f"ONNX input names: {onnx_input_names}")
tensor_proto_dict = {}
for path in tensor_paths:
    for name, tensor in torch.load(path).items():
        # This name should match the name-generating code in FX-to-ONNX exporter.
        # See function _replace_get_attr_with_placeholder for details.
        refined_name = name.replace(".", "_")
        for i, onnx_input_name in enumerate(onnx_input_names):
            if onnx_input_name.endswith(refined_name) or refined_name.endswith(onnx_input_name):
                refined_name = onnx_input_name
                break
        print(f"Save {name} to {refined_name}")
        tensor_proto = create_tensor_proto_with_external_data(tensor, refined_name, f"initializer/{refined_name}", basepath=".")
        print(f"tensor_proto.data_location: {tensor_proto.data_location}")
        onnx_model_with_initializers.graph.initializer.append(tensor_proto)

print("Saving model with external data...")
onnx.save(onnx_model_with_initializers, "model_tiny_gpt_external.onnx")
ort_sess_with_initializers = onnxruntime.InferenceSession(
    "model_tiny_gpt_external.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)

def test_external_data(bound_args):
    model = AutoModel.from_pretrained(model_name)

    pth_args = []
    for t in bound_args:
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

    pth_out = model(*pth_args)
    ort_sess = onnxruntime.InferenceSession(
        "model_tiny_gpt_external.onnx",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    onnx_model = onnx.load("model_tiny_gpt_external.onnx")
    initializer_names = set([init.name for init in onnx_model.graph.initializer])
    ort_input_dict = {}
    for ort_input, t in zip(
        [input for input in onnx_model.graph.input if input.name not in initializer_names], [arg for arg in pth_args if arg is not None]
    ):
        ort_input_dict[ort_input.name] = t.numpy()
    ort_out = ort_sess.run(None, ort_input_dict)

    np.testing.assert_allclose(
        ort_out[0], pth_out["last_hidden_state"].detach().numpy(), atol=1e-4, rtol=1e-3
    )
    for ort_value, pth_value in zip(
        ort_out[1:], pth_out["past_key_values"][0] + pth_out["past_key_values"][1]
    ):
        np.testing.assert_allclose(ort_value, pth_value.detach().numpy(), atol=1e-4, rtol=1e-3)


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

    onnx.save(onnx_model, "model_tiny_gpt.onnx")
    ort_sess = onnxruntime.InferenceSession(
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
    test_one(graph_module, onnx_model, bound_args, replaced_attrs)
    test_external_data(bound_args)

