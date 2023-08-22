# Owner(s): ["module: onnx"]

import onnx_test_common
import pytorch_test_common
import torch
import torch.utils.cpp_extension
from torch.onnx import symbolic_helper
from torch.testing._internal import common_utils


class TestCustomAutogradFunction(pytorch_test_common.ExportTestCase):
    opset_version = 9
    keep_initializers_as_inputs = False
    onnx_shape_inference = True

    def test_symbolic(self):
        class MyClip(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input, scalar):
                ctx.save_for_backward(input)
                return input.clamp(min=scalar)

            @staticmethod
            def symbolic(g, input, scalar):
                return g.op("Clip", input, min_f=scalar)

        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.clip = MyClip.apply

            def forward(self, x):
                h = self.clip(x, 2)
                return h

        x = torch.randn(2, 3, 4, requires_grad=True)
        model = MyModule()
        onnx_test_common.run_model_test(self, model, input_args=(x,))

    def test_register_op(self):
        class MyClip(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input, scalar):
                ctx.save_for_backward(input)
                return input.clamp(min=scalar)

        class MyRelu(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input):
                ctx.save_for_backward(input)
                return input.clamp(min=0)

        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.clip = MyClip.apply
                self.relu = MyRelu.apply

            def forward(self, x):
                h = self.clip(x, 2)
                h = self.relu(h)
                return h

        def symbolic_pythonop(ctx: torch.onnx.SymbolicContext, g, *args, **kwargs):
            n = ctx.cur_node
            name = kwargs["name"]
            if name == "MyClip":
                return g.op("Clip", args[0], min_f=args[1], outputs=n.outputsSize())
            elif name == "MyRelu":
                return g.op("Relu", args[0], outputs=n.outputsSize())
            else:
                return symbolic_helper._unimplemented(
                    "prim::PythonOp", "unknown node kind: " + name
                )

        from torch.onnx import register_custom_op_symbolic

        register_custom_op_symbolic("prim::PythonOp", symbolic_pythonop, 1)

        x = torch.randn(2, 3, 4, requires_grad=True)
        model = MyModule()
        onnx_test_common.run_model_test(self, model, input_args=(x,))


class TestExportAsContribOps(pytorch_test_common.ExportTestCase):
    opset_version = 14
    keep_initializers_as_inputs = False
    onnx_shape_inference = True

    def test_contrib_op_with_loop(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.gelu = torch.nn.GELU(approximate="none")

            def forward(self, x):
                res = []
                res2 = []
                for i in range(x.size(0)):
                    if len(res) > 0:
                        res2.append(res[0])
                    else:
                        res2.append(self.gelu(x[0]))
                    res.append(x[0])
                return torch.stack(res), torch.stack(res2)

        def symbolic_custom_gelu(g, input, approximate):
            return g.op("com.microsoft::Gelu", input).setType(input.type())

        from torch.onnx import register_custom_op_symbolic

        register_custom_op_symbolic("::gelu", symbolic_custom_gelu, 1)

        x = torch.randn(3, 3, 4, requires_grad=True)
        model = torch.jit.script(M())
        onnx_test_common.run_model_test(self, model, input_args=(x,))


if __name__ == "__main__":
    common_utils.run_tests()
