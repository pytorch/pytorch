# Owner(s): ["oncall: jit"]

import unittest

import torch
import torch._C

torch.ops.load_library("//caffe2:xnnpack_backend")

class TestXNNPackBackend(unittest.TestCase):
    def test_xnnpack_lowering(self):
        class Module(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x + x

        scripted_module = torch.jit.script(Module())

        faulty_compile_spec = {
            "backward": {
                "inputs" : [torch.zeros(1)],
                "outputs": [torch.zeros(1)],
            }
        }
        error_msg = (
            "method_compile_spec does not contain the \"forward\" key."
        )

        with self.assertRaisesRegex(
            RuntimeError,
            error_msg,
        ):
            _ = torch._C._jit_to_backend(
                "xnnpack",
                scripted_module,
                faulty_compile_spec,
            )

        mismatch_compile_spec = {
            "forward" : {
                "inputs" : [torch.zeros(1), torch.zeros(1)],
                "outputs" : [torch.zeros(1)]
            }
        }
        error_msg = ("method_compile_spec inputs do not match expected number of forward inputs")

        with self.assertRaisesRegex(
            RuntimeError,
            error_msg,
        ):
            _ = torch._C._jit_to_backend(
                "xnnpack",
                scripted_module,
                mismatch_compile_spec
            )

        lowered = torch._C._jit_to_backend(
            "xnnpack",
            scripted_module,
            {
                "forward": {
                    "inputs" : [torch.zeros(1)],
                    "outputs": [torch.zeros(1)],
                }
            }
        )
        lowered(torch.zeros(1))

    def test_xnnpack_unsupported(self):
        class AddSpliceModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                z = x + y[:, :, 1, :]
                return z

        sample_inputs = (torch.rand(1, 512, 512, 3), torch.rand(1, 512, 512, 3))
        sample_output = torch.zeros(1, 512, 512, 3)

        error_msg = (
            "the module contains the following unsupported ops:\n"
            "aten::select\n"
            "aten::slice\n"
        )

        add_module = torch.jit.script(AddSpliceModule())
        with self.assertRaisesRegex(
            RuntimeError,
            error_msg,
        ):
            _ = torch._C._jit_to_backend(
                "xnnpack",
                add_module,
                {
                    "forward": {
                        "inputs" : [sample_inputs[0], sample_inputs[1]],
                        "outputs": [sample_output]
                    }
                }
            )
