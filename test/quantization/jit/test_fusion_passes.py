# Owner(s): ["oncall: quantization"]

# torch
import torch
from torch.testing import FileCheck
from torch.testing._internal.common_quantization import QuantizationTestCase
from torch.testing._internal.common_utils import raise_on_run_directly


class TestFusionPasses(QuantizationTestCase):
    def test_quantized_add_relu_fusion(self):
        class MAdd(torch.nn.Module):
            def forward(self, x, y):
                a = torch.ops.quantized.add(x, y, 1.0, 0)
                relu_out = torch.relu(a)
                return relu_out

        A = torch.arange(-128, 130, dtype=torch.float)
        B = torch.arange(-128, 130, dtype=torch.float)
        scale = 2.0
        zero_point = 127
        qA = torch.quantize_per_tensor(
            A, scale=scale, zero_point=zero_point, dtype=torch.quint8
        )
        qB = torch.quantize_per_tensor(
            B, scale=scale, zero_point=zero_point, dtype=torch.quint8
        )

        # Check quantized add + relu fusion
        m = MAdd()
        scripted_m = torch.jit.script(m)
        ref_output = scripted_m(qA, qB)

        # Must inline the graph.
        # In this test case since we are directly calling ops
        # it does not matter, however if we are calling nn
        # modules we have to inline graph.
        torch._C._jit_pass_inline(scripted_m.graph)
        torch._C._jit_pass_fuse_quantized_add_relu(scripted_m.graph)
        FileCheck().check_not("aten::relu").check("quantized::add_relu").run(
            scripted_m.graph
        )
        output = scripted_m(qA, qB)
        self.assertEqual(ref_output, output)

        class MAddOut(torch.nn.Module):
            def forward(self, x, y, z):
                a = torch.ops.quantized.add_out(x, y, z)
                relu_out = torch.relu(a)
                return relu_out

        qC = torch._empty_affine_quantized(
            qA.shape, scale=scale, zero_point=zero_point, dtype=torch.quint8
        )
        # Check quantized add + relu fusion
        m = MAddOut()
        scripted_m = torch.jit.script(m)
        ref_output = scripted_m(qA, qB, qC)
        # Must inline the graph.
        # In this test case since we are directly calling ops
        # it does not matter, however if we are calling nn
        # modules we have to inline graph.
        torch._C._jit_pass_inline(scripted_m.graph)
        torch._C._jit_pass_fuse_quantized_add_relu(scripted_m.graph)
        FileCheck().check_not("aten::relu").check_not("quantized::add_out").check(
            "quantized::add_relu_out"
        ).run(scripted_m.graph)
        output = scripted_m(qA, qB, qC)
        self.assertEqual(ref_output, output)

        class MAddScalar(torch.nn.Module):
            def forward(self, x, y: float):
                a = torch.ops.quantized.add_scalar(x, y)
                relu_out = torch.relu(a)
                return relu_out

        # Check quantized add + relu fusion
        m = MAddScalar()
        scripted_m = torch.jit.script(m)
        ref_output = scripted_m(qA, 3.0)
        torch._C._jit_pass_inline(scripted_m.graph)
        torch._C._jit_pass_fuse_quantized_add_relu(scripted_m.graph)
        FileCheck().check_not("aten::relu").check_not("quantized::add_scalar(").check(
            "quantized::add_scalar_relu"
        ).run(scripted_m.graph)
        output = scripted_m(qA, 3.0)
        self.assertEqual(ref_output, output)

        class MAddScalarOut(torch.nn.Module):
            def forward(self, x, y: float, z):
                a = torch.ops.quantized.add_scalar_out(x, y, z)
                relu_out = torch.relu(a)
                return relu_out

        qC = torch._empty_affine_quantized(
            qA.shape, scale=scale, zero_point=zero_point, dtype=torch.quint8
        )
        m = MAddScalarOut()
        scripted_m = torch.jit.script(m)
        ref_output = scripted_m(qA, 3.0, qC)
        torch._C._jit_pass_inline(scripted_m.graph)
        torch._C._jit_pass_fuse_quantized_add_relu(scripted_m.graph)
        FileCheck().check_not("aten::relu").check_not(
            "quantized::add_scalar_out"
        ).check("quantized::add_scalar_relu_out").run(scripted_m.graph)
        output = scripted_m(qA, 3.0, qC)
        self.assertEqual(ref_output, output)


if __name__ == "__main__":
    raise_on_run_directly("test/test_quantization.py")
