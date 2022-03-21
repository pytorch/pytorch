import torch
from torch._C import parse_ir
import io
import onnxruntime
import unittest
from test_pytorch_onnx_onnxruntime import run_ort, ort_compare_with_pytorch

class _TestJITIRToONNX:
    """Abstract base class for test cases.

    Intentionally not a sub-class of unittest.TestCase so that unittest / pytest
    don't run it directly. unitest.TestCase is mixed in as another base class when
    creating concrete sub-types. See MakeTestCase().
    """
    opset_version = -1  # Sub-classes must override
    ort_providers = ["CPUExecutionProvider"]

    def run_test(self, graph_ir, example_inputs):
        graph = parse_ir(graph_ir)
        jit_outs = torch._C._jit_interpret_graph(graph, example_inputs)

        f = io.BytesIO()
        # TODO: insert export api here to convert JIT IR to ONNX proto

        ort_sess = onnxruntime.InferenceSession(f.getvalue(), providers=self.ort_providers)
        ort_outs = run_ort(ort_sess, example_inputs)

        ort_compare_with_pytorch(ort_outs, jit_outs, rtol=1e-3, atol=1e-7)

    def test_example_ir(self):
        graph_ir = """
        graph(%1 : Float(2, 3),
              %2 : Float(2, 3)):
          %3 : int = prim::Constant[value=1]()
          %4 : Float(2, 3) = aten::add(%1, %2, %3)
          return (%4)
        """
        a = torch.randn(2, 3)
        b = torch.randn(2, 3)
        self.run_test(graph_ir, (a, b))

def MakeTestCase(opset_version: int) -> type:
    name = f"TestJITIRToONNX_opset{opset_version}"
    return type(str(name),
                (unittest.TestCase,),
                dict(_TestJITIRToONNX.__dict__,
                     opset_version=opset_version))

# opset 14 tests
TestJITIRToONNX_opset14 = MakeTestCase(14)
