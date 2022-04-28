# Owner(s): ["module: onnx"]

import io
import os
import sys
from unittest.mock import patch

import onnx
import torch

from test_pytorch_common import TestCase
from torch.onnx.symbolic_helper import _onnx_unsupported
from torch.onnx import (OperatorExportTypes,
                        symbolic_registry)
from torch.testing._internal.common_utils import (skipIfCaffe2,
                                                  custom_op)


# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)

class TestONNXExport(TestCase):
    @skipIfCaffe2
    def test_clip_aten_fallback_due_exception(self):
        x = torch.randn(3, 4, requires_grad=True)

        def bad_clamp(g, self, min, max):
            return _onnx_unsupported("Bad boy!")

        class MyClip(torch.nn.Module):
            def forward(self, x):
                return torch.clamp(x, min=-0.5, max=0.5)

        f = io.BytesIO()
        with custom_op("aten::clamp", bad_clamp, 9):
            torch.onnx.export(MyClip(), x, f,
                              operator_export_type=OperatorExportTypes.ONNX_ATEN_FALLBACK)
        onnx_model = onnx.load_from_string(f.getvalue())
        self.assertAtenOp(onnx_model, "clamp", "Tensor")

    @skipIfCaffe2
    def test_clip_aten_fallback_explicit_request(self):
        class MyClip(torch.nn.Module):
            def forward(self, x):
                return torch.clamp(x, min=-0.5, max=0.5)

        def break_is_registered_op_api(opname, domain, version):
            fake_missing_symbolics = ('clamp',)
            if opname in fake_missing_symbolics:
                return False
            return (domain, version) in symbolic_registry._registry \
                and opname in symbolic_registry._registry[(domain, version)]

        f = io.BytesIO()
        with patch("torch.onnx.symbolic_registry.is_registered_op", side_effect=break_is_registered_op_api):
            # Force missing symbolic for well-known op
            x = torch.randn(3, 4, requires_grad=True)
            torch.onnx.export(MyClip(), x, f,
                              operator_export_type=OperatorExportTypes.ONNX_ATEN_FALLBACK)
        onnx_model = onnx.load_from_string(f.getvalue())
        self.assertAtenOp(onnx_model, "clamp", "Tensor")
