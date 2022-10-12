# Owner(s): ["module: onnx"]
import onnxruntime

import torch
from pytorch_test_common import skipIfNoCuda
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
    check_shape = True
    check_dtype = True
    ignore_none = True  # True for tracing, and Flase for scripting

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
            ort_outs,
            jit_outs,
            rtol=1e-3,
            atol=1e-7,
            check_shape=self.check_shape,
            check_dtype=self.check_dtype,
            ignore_none=self.ignore_none,
            acceptable_error_percentage=None,
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

    def test_add_sub_with_graph_inputs(self):
        for op in ["add", "sub", "rsub"]:
            graph_ir = f"""
            graph(%1 : Float(2, 3),
                  %2 : Float(2, 3),
                  %3 : int):
              %4 : Float(2, 3) = aten::{op}(%1, %2, %3)
              return (%4)
            """
            a = torch.randn(2, 3)
            b = torch.randn(2, 3)
            self.run_test(graph_ir, (a, b, 2))

    def test_native_layer_norm(self):
        graph_ir = """
        graph(%x : Float(2, 3, 2),
              %w : Float(3, 2),
              %b : Float(3, 2)):
          %5 : int = prim::Constant[value=3]()
          %6 : int = prim::Constant[value=2]()
          %7 : int[] = prim::ListConstruct(%5, %6)
          %10 : float = prim::Constant[value=1.0000000000000001e-05]()
          %11 : Float(2, 3, 2), %12 : Float(2, 1, 1), %13 : Float(2, 1, 1) = aten::native_layer_norm(%x, %7, %w, %b, %10)
          return (%11, %12, %13)
        """
        x = torch.randn(2, 3, 2)
        w = torch.randn(3, 2)
        b = torch.randn(3, 2)
        self.run_test(graph_ir, (x, w, b))

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

    def test_log_softmax(self):
        graph_ir = """
        graph(%x: Tensor):
          %half_to_float: bool = prim::Constant[value=0]()
          %dim: int = prim::Constant[value=1]()
          %y = aten::_log_softmax(%x, %dim, %half_to_float)
          return (%y)
        """
        x = torch.randn(5, 2)
        self.run_test(graph_ir, (x,))

    @skipIfNoCuda
    def test_log_softmax_half_to_float(self):
        graph_ir = """
        graph(%x: Tensor):
          %half_to_float: bool = prim::Constant[value=1]()
          %dim: int = prim::Constant[value=1]()
          %y = aten::_log_softmax(%x, %dim, %half_to_float)
          return (%y)
        """
        x = torch.randn(5, 2).half().to("cuda")
        self.run_test(graph_ir, (x,))

    def test_native_dropout(self):
        graph_ir = """
        graph(%1 : Float(2, 3)):
          %2 : float = prim::Constant[value=0.0]()
          %training : bool = prim::Constant[value=1]()
          %3 : Tensor, %4 : Tensor = aten::native_dropout(%1, %2, %training)
          return (%3, %4)
        """
        a = torch.randn(2, 3)
        self.run_test(graph_ir, (a,))


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
