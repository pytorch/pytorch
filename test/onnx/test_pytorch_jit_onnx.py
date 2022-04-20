# Owner(s): ["module: onnx"]
import torch
import onnxruntime
import unittest
from test_pytorch_onnx_onnxruntime import run_ort, ort_compare_with_pytorch
from torch._C import parse_ir

def _jit_graph_to_onnx_model(
        graph,
        operator_export_type,
        opset_version):
    r"""
    This function exports torch::jit::Graph object
    to serialized ONNX ModelProto.
    This function is for testing purpose.
    It only keeps the essential parts for IR graph conversions.
    It also does not interact with actual PyTorch modules nor
    PyTorch tensor inputs.
    """
    from torch.onnx.symbolic_helper import _set_onnx_shape_inference, _set_opset_version
    from torch.onnx.utils import _optimize_graph
    # Shape inference is required because some ops' symbolic functions
    # generate sub-graphs based on inputs' types.
    _set_onnx_shape_inference(True)
    _set_opset_version(opset_version)
    graph = _optimize_graph(graph, operator_export_type, params_dict={})
    proto, _, _, _ = graph._export_onnx(
        {}, opset_version, {}, False,
        operator_export_type, False, False,
        {}, True, "", {})
    return proto

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

        onnx_proto = _jit_graph_to_onnx_model(
            graph,
            torch.onnx.OperatorExportTypes.ONNX,
            self.opset_version)
        ort_sess = onnxruntime.InferenceSession(onnx_proto, providers=self.ort_providers)
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


TestJITIRToONNX_opset14 = MakeTestCase(14)

if __name__ == "__main__":
    unittest.main()
