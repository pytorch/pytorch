# Owner(s): ["module: onnx"]
"""Unit tests for the _tensors module."""

from __future__ import annotations
import contextlib

import torch
from torch.onnx._internal.exporter import _testing as onnx_testing
from torch.testing._internal import common_utils

@contextlib.contextmanager
def bypass_export_jit_isinstance():
    """
    Tries to bypass some functions torch.export.export does not
    support such as ``torch.jit.isinstance``.
    """
    import torch.jit

    f = torch.jit.isinstance
    torch.jit.isinstance = isinstance

    try:
        yield
    finally:
        torch.jit.isinstance = f


class SymbolicTensorTest(common_utils.TestCase):
    def test_jit_isinstance(self):
        class DummyModel(torch.nn.Module):
            def forward(self, a, b):
                if torch.jit.isinstance(a, torch.Tensor):
                    return a.cos()
                return b.sin()

        model = DummyModel()
        a = torch.rand(16, 16, dtype=torch.float16, device="cpu")
        b = torch.rand(16, 16, dtype=torch.float16, device="cpu")

        with bypass_export_jit_isinstance():
            onnx_program = torch.onnx.export(model, (a, b), dynamo=True)
        onnx_testing.assert_onnx_program(onnx_program)


if __name__ == "__main__":
    common_utils.run_tests()
