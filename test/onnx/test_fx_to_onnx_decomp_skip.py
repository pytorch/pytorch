# Owner(s): ["module: onnx"]
from __future__ import annotations

import torch
import pytorch_test_common

import onnx
import onnx.inliner

def assert_op_in_onnx_model(model: onnx.ModelProto, op_type: str):
    inlined = onnx.inliner.inline_local_functions(model)
    for node in inlined.graph.node:
        if node.op_type == op_type:
            return
    raise AssertionError(f"Op {op_type} not found in model")

class TestDynamoExportDecompSkip(pytorch_test_common.ExportTestCase):
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
