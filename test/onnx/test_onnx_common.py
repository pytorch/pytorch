# Owner(s): ["module: onnx"]

from __future__ import annotations

import os
import unittest

import numpy as np
import onnxruntime

import torch
from torch.onnx import _constants, verification

onnx_model_dir = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    os.pardir,
    "repos",
    "onnx",
    "onnx",
    "backend",
    "test",
    "data",
)


pytorch_converted_dir = os.path.join(onnx_model_dir, "pytorch-converted")


pytorch_operator_dir = os.path.join(onnx_model_dir, "pytorch-operator")

_ORT_PROVIDERS = ("CPUExecutionProvider",)


def _run_model_test(test_suite: _TestONNXRuntime, *args, **kwargs):
    kwargs["ort_providers"] = _ORT_PROVIDERS
    kwargs["opset_version"] = test_suite.opset_version
    kwargs["keep_initializers_as_inputs"] = test_suite.keep_initializers_as_inputs
    return verification.verify(*args, **kwargs)


class _TestONNXRuntime(unittest.TestCase):
    opset_version = _constants.onnx_default_opset
    keep_initializers_as_inputs = True  # For IR version 3 type export.
    is_script = False

    def setUp(self):
        torch.manual_seed(0)
        onnxruntime.set_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)
        np.random.seed(seed=0)
        os.environ["ALLOW_RELEASED_ONNX_OPSET_ONLY"] = "0"
        self.is_script_test_enabled = True

    # The exported ONNX model may have less inputs than the pytorch model because of const folding.
    # This mostly happens in unit test, where we widely use torch.size or torch.shape.
    # So the output is only dependent on the input shape, not value.
    # remained_onnx_input_idx is used to indicate which pytorch model input idx is remained in ONNX model.
    def run_test(
        self,
        model,
        input_args,
        input_kwargs=None,
        rtol=1e-3,
        atol=1e-7,
        do_constant_folding=True,
        dynamic_axes=None,
        additional_test_inputs=None,
        input_names=None,
        output_names=None,
        fixed_batch_size=False,
        training=None,
        remained_onnx_input_idx=None,
        verbose=False,
    ):
        def _run_test(m, remained_onnx_input_idx, flatten=True):
            return _run_model_test(
                self,
                m,
                input_args=input_args,
                input_kwargs=input_kwargs,
                rtol=rtol,
                atol=atol,
                do_constant_folding=do_constant_folding,
                dynamic_axes=dynamic_axes,
                additional_test_inputs=additional_test_inputs,
                input_names=input_names,
                output_names=output_names,
                fixed_batch_size=fixed_batch_size,
                training=training,
                remained_onnx_input_idx=remained_onnx_input_idx,
                flatten=flatten,
                verbose=verbose,
            )

        if isinstance(remained_onnx_input_idx, dict):
            scripting_remained_onnx_input_idx = remained_onnx_input_idx["scripting"]
            tracing_remained_onnx_input_idx = remained_onnx_input_idx["tracing"]
        else:
            scripting_remained_onnx_input_idx = remained_onnx_input_idx
            tracing_remained_onnx_input_idx = remained_onnx_input_idx

        is_model_script = isinstance(
            model, (torch.jit.ScriptModule, torch.jit.ScriptFunction)
        )

        if self.is_script_test_enabled and self.is_script:
            script_model = model if is_model_script else torch.jit.script(model)
            _run_test(script_model, scripting_remained_onnx_input_idx, flatten=False)

        if not is_model_script and not self.is_script:
            _run_test(model, tracing_remained_onnx_input_idx)
