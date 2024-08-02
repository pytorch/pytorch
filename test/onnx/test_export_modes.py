# Owner(s): ["module: onnx"]

import io
import os
import shutil
import sys
import tempfile

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.onnx import OperatorExportTypes


# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
import pytorch_test_common

from torch.testing._internal import common_utils


# Smoke tests for export methods
class TestExportModes(pytorch_test_common.ExportTestCase):
    class MyModel(nn.Module):
        def __init__(self) -> None:
            super(TestExportModes.MyModel, self).__init__()

        def forward(self, x):
            return x.transpose(0, 1)

    def test_protobuf(self):
        torch_model = TestExportModes.MyModel()
        fake_input = Variable(torch.randn(1, 1, 224, 224), requires_grad=True)
        f = io.BytesIO()
        torch.onnx._export(
            torch_model,
            (fake_input),
            f,
            verbose=False,
            export_type=torch.onnx.ExportTypes.PROTOBUF_FILE,
        )

    def test_zipfile(self):
        torch_model = TestExportModes.MyModel()
        fake_input = Variable(torch.randn(1, 1, 224, 224), requires_grad=True)
        f = io.BytesIO()
        torch.onnx._export(
            torch_model,
            (fake_input),
            f,
            verbose=False,
            export_type=torch.onnx.ExportTypes.ZIP_ARCHIVE,
        )

    def test_compressed_zipfile(self):
        torch_model = TestExportModes.MyModel()
        fake_input = Variable(torch.randn(1, 1, 224, 224), requires_grad=True)
        f = io.BytesIO()
        torch.onnx._export(
            torch_model,
            (fake_input),
            f,
            verbose=False,
            export_type=torch.onnx.ExportTypes.COMPRESSED_ZIP_ARCHIVE,
        )

    def test_directory(self):
        torch_model = TestExportModes.MyModel()
        fake_input = Variable(torch.randn(1, 1, 224, 224), requires_grad=True)
        d = tempfile.mkdtemp()
        torch.onnx._export(
            torch_model,
            (fake_input),
            d,
            verbose=False,
            export_type=torch.onnx.ExportTypes.DIRECTORY,
        )
        shutil.rmtree(d)

    def test_onnx_multiple_return(self):
        @torch.jit.script
        def foo(a):
            return (a, a)

        f = io.BytesIO()
        x = torch.ones(3)
        torch.onnx.export(foo, (x,), f)

    @common_utils.skipIfNoLapack
    def test_aten_fallback(self):
        class ModelWithAtenNotONNXOp(nn.Module):
            def forward(self, x, y):
                abcd = x + y
                defg = torch.linalg.qr(abcd)
                return defg

        x = torch.rand(3, 4)
        y = torch.rand(3, 4)
        torch.onnx.export_to_pretty_string(
            ModelWithAtenNotONNXOp(),
            (x, y),
            add_node_names=False,
            do_constant_folding=False,
            operator_export_type=OperatorExportTypes.ONNX_ATEN_FALLBACK,
            # support for linalg.qr was added in later op set versions.
            opset_version=9,
        )

    # torch.fmod is using to test ONNX_ATEN.
    # If you plan to remove fmod from aten, or found this test failed.
    # please contact @Rui.
    def test_onnx_aten(self):
        class ModelWithAtenFmod(nn.Module):
            def forward(self, x, y):
                return torch.fmod(x, y)

        x = torch.randn(3, 4, dtype=torch.float32)
        y = torch.randn(3, 4, dtype=torch.float32)
        torch.onnx.export_to_pretty_string(
            ModelWithAtenFmod(),
            (x, y),
            add_node_names=False,
            do_constant_folding=False,
            operator_export_type=OperatorExportTypes.ONNX_ATEN,
        )


if __name__ == "__main__":
    common_utils.run_tests()
