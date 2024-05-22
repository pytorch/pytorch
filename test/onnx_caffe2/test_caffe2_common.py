# Owner(s): ["module: onnx"]

import glob
import os

import caffe2.python.onnx.backend as c2

import numpy as np
import onnx.backend.test
from onnx import numpy_helper


def load_tensor_as_numpy_array(f):
    tensor = onnx.TensorProto()
    with open(f, "rb") as file:
        tensor.ParseFromString(file.read())
    return tensor


def assert_similar(ref, real):
    np.testing.assert_equal(len(ref), len(real))
    for i in range(len(ref)):
        np.testing.assert_allclose(ref[i], real[i], rtol=1e-3)


def run_generated_test(model_file, data_dir, device="CPU"):
    model = onnx.load(model_file)
    input_num = len(glob.glob(os.path.join(data_dir, "input_*.pb")))
    inputs = []
    for i in range(input_num):
        inputs.append(
            numpy_helper.to_array(
                load_tensor_as_numpy_array(os.path.join(data_dir, f"input_{i}.pb"))
            )
        )
    output_num = len(glob.glob(os.path.join(data_dir, "output_*.pb")))
    outputs = []
    for i in range(output_num):
        outputs.append(
            numpy_helper.to_array(
                load_tensor_as_numpy_array(os.path.join(data_dir, f"output_{i}.pb"))
            )
        )
    prepared = c2.prepare(model, device=device)
    c2_outputs = prepared.run(inputs)
    assert_similar(outputs, c2_outputs)
