# Owner(s): ["module: onnx"]
import onnxruntime

import torch
from torch.onnx import verification
from torch.testing._internal import common_utils


def _jit_graph_to_onnx_model(graph, operator_export_type, opset_version):
    r"""
    This function exports torch::jit::Graph object
    to serialized ONNX ModelProto.
    This function is for testing purpose.
    It only keeps the essential parts for IR graph conversions.
    It also does not interact with actual PyTorch modules nor
    PyTorch tensor inputs.
    """

    # Shape inference is required because some ops' symbolic functions
    # generate sub-graphs based on inputs' types.
    torch.onnx.symbolic_helper._set_onnx_shape_inference(True)
    torch.onnx.symbolic_helper._set_opset_version(opset_version)
    graph = torch.onnx.utils._optimize_graph(
        graph, operator_export_type, params_dict={}
    )
    proto, _, _, _ = graph._export_onnx(
        {},
        opset_version,
        {},
        False,
        operator_export_type,
        False,
        False,
        {},
        True,
        "",
        {},
    )
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
        graph = torch._C.parse_ir(graph_ir)
        jit_outs = torch._C._jit_interpret_graph(graph, example_inputs)

        onnx_proto = _jit_graph_to_onnx_model(
            graph, torch.onnx.OperatorExportTypes.ONNX, self.opset_version
        )
        ort_sess = onnxruntime.InferenceSession(
            onnx_proto, providers=self.ort_providers
        )
        ort_outs = verification._run_ort(ort_sess, example_inputs)

        verification._compare_ort_pytorch_outputs(
            ort_outs, jit_outs, rtol=1e-3, atol=1e-7
        )

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

    def test_convolution(self):
        graph_ir = """
        graph(%1 : Tensor,
              %2 : Tensor):
          %3 : NoneType = prim::Constant()
          %4 : int[] = prim::Constant[value=[1, 1]]()
          %5 : int[] = prim::Constant[value=[0, 0]]()
          %6 : bool = prim::Constant[value=0]()
          %7 : int = prim::Constant[value=1]()
          %8 : Tensor = aten::convolution(%1, %2, %3, %4, %5, %4, %6, %5, %7)
          return (%8)
        """
        x = torch.randn(8, 1, 5, 5)
        w = torch.randn(4, 1, 3, 3)
        self.run_test(graph_ir, (x, w))


def MakeTestCase(opset_version: int) -> type:
    name = f"TestJITIRToONNX_opset{opset_version}"
    return type(
        str(name),
        (common_utils.TestCase,),
        dict(_TestJITIRToONNX.__dict__, opset_version=opset_version),
    )


TestJITIRToONNX_opset14 = MakeTestCase(14)

if __name__ == "__main__":
    common_utils.run_tests()
