# Owner(s): ["oncall: jit"]

import unittest

import torch
import torch._C

torch.ops.load_library("//caffe2:xnnpack_backend")

class TestXNNPackBackend(unittest.TestCase):
    def test_xnnpack_constant_data(self):
        class Module(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self._constant = torch.ones(4, 4, 4)

            def forward(self, x):
                return x + self._constant

        scripted_module = torch.jit.script(Module())

        lowered_module = torch._C._jit_to_backend(
            "xnnpack",
            scripted_module,
            {
                "forward": {
                    "inputs" : [torch.randn(4, 4, 4)],
                    "outputs": [torch.randn(4, 4, 4)]
                }
            }
        )

        for i in range(0, 20):
            sample_input = torch.randn(4, 4, 4)
            actual_output = scripted_module(sample_input)
            expected_output = lowered_module(sample_input)
            self.assertTrue(torch.allclose(actual_output, expected_output, atol=1e-03, rtol=1e-03))

    def test_xnnpack_lowering(self):
        class Module(torch.nn.Module):
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

    def test_xnnpack_backend_add(self):
        class AddModule(torch.nn.Module):
            def forward(self, x, y):
                z = x + y
                z = z + x
                z = z + x
                return z

        add_module = AddModule()
        sample_inputs = (torch.rand(1, 512, 512, 3), torch.rand(1, 512, 512, 3))
        sample_output = torch.zeros(1, 512, 512, 3)

        add_module = torch.jit.script(add_module)
        expected_output = add_module(sample_inputs[0], sample_inputs[1])

        lowered_add_module = torch._C._jit_to_backend(
            "xnnpack",
            add_module,
            {
                "forward": {
                    "inputs" : [sample_inputs[0].clone(), sample_inputs[1].clone()],
                    "outputs": [sample_output]
                }
            }
        )

        actual_output = lowered_add_module.forward(sample_inputs[0], sample_inputs[1])
        self.assertTrue(torch.allclose(actual_output, expected_output, atol=1e-03, rtol=1e-03))

    def test_xnnpack_broadcasting(self):
        class AddModule(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        add_module = AddModule()
        sample_inputs = (torch.rand(5, 1, 4, 1), torch.rand(3, 1, 1))
        sample_output = torch.zeros(5, 3, 4, 1)

        add_module = torch.jit.script(add_module)
        expected_output = add_module(sample_inputs[0], sample_inputs[1])

        lowered_add_module = torch._C._jit_to_backend(
            "xnnpack",
            add_module,
            {
                "forward": {
                    "inputs" : [sample_inputs[0], sample_inputs[1]],
                    "outputs": [sample_output]
                }
            }
        )

        actual_output = lowered_add_module.forward(sample_inputs[0], sample_inputs[1])
        self.assertTrue(torch.allclose(actual_output, expected_output, atol=1e-03, rtol=1e-03))

    def test_xnnpack_unsupported(self):
        class AddSpliceModule(torch.nn.Module):
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
