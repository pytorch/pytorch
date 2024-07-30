# Owner(s): ["module: onnx"]
from __future__ import annotations

import onnx
import onnx.inliner
import pytorch_test_common

import torch
from torch.testing._internal import common_utils


def assert_op_in_onnx_model(model: onnx.ModelProto, op_type: str):
    inlined = onnx.inliner.inline_local_functions(model)
    for node in inlined.graph.node:
        if node.op_type == op_type:
            return
    raise AssertionError(f"Op {op_type} not found in model")


class TestDynamoExportDecompSkip(pytorch_test_common.ExportTestCase):
    def _test_exported_program_forces_decomposition(self, model, input, op_type):
        ep = torch.export.export(model, input)
        onnx_program = torch.onnx.dynamo_export(ep, *input)
        with self.assertRaises(AssertionError):
            assert_op_in_onnx_model(onnx_program.model_proto, op_type)

    def test_upsample_bilinear2d(self):
        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.upsample = torch.nn.Upsample(scale_factor=2, mode="bilinear")

            def forward(self, x):
                return self.upsample(x)

        onnx_program = torch.onnx.dynamo_export(TestModel(), torch.randn(1, 1, 2, 2))
        # If decomposition is skipped, the model will contain a Resize op instead of fine grained subgraph.
        assert_op_in_onnx_model(onnx_program.model_proto, "Resize")
        self._test_exported_program_forces_decomposition(
            TestModel(), (torch.randn(1, 1, 2, 2),), "Resize"
        )

    def test_upsample_bilinear2d_output_size(self):
        def func(x: torch.Tensor):
            return torch.nn.functional.interpolate(x, size=(4, 4), mode="bilinear")

        onnx_program = torch.onnx.dynamo_export(func, torch.randn(1, 1, 2, 2))
        # If decomposition is skipped, the model will contain a Resize op instead of fine grained subgraph.
        assert_op_in_onnx_model(onnx_program.model_proto, "Resize")

    def test_upsample_trilinear3d(self):
        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.upsample = torch.nn.Upsample(scale_factor=2, mode="trilinear")

            def forward(self, x):
                return self.upsample(x)

        onnx_program = torch.onnx.dynamo_export(TestModel(), torch.randn(1, 1, 2, 2, 3))
        # If decomposition is skipped, the model will contain a Resize op instead of fine grained subgraph.
        assert_op_in_onnx_model(onnx_program.model_proto, "Resize")
        self._test_exported_program_forces_decomposition(
            TestModel(), (torch.randn(1, 1, 2, 2, 3),), "Resize"
        )

    def test_upsample_trilinear3d_output_size(self):
        def func(x: torch.Tensor):
            return torch.nn.functional.interpolate(x, size=(4, 4, 4), mode="trilinear")

        onnx_program = torch.onnx.dynamo_export(func, torch.randn(1, 1, 2, 2, 3))
        # If decomposition is skipped, the model will contain a Resize op instead of fine grained subgraph.
        assert_op_in_onnx_model(onnx_program.model_proto, "Resize")

    def test_instance_norm(self):
        class TestModel(torch.nn.Module):
            def forward(self, x):
                return torch.nn.functional.instance_norm(x)

        onnx_program = torch.onnx.dynamo_export(TestModel(), torch.randn(1, 1, 2, 2))
        # If decomposition is skipped, the model will contain an InstanceNormalization op
        # instead of BatchNormalization op w/ training=True.
        assert_op_in_onnx_model(onnx_program.model_proto, "InstanceNormalization")
        self._test_exported_program_forces_decomposition(
            TestModel(), (torch.randn(1, 1, 2, 2),), "InstanceNormalization"
        )


if __name__ == "__main__":
    common_utils.run_tests()
