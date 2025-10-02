# Owner(s): ["module: tests"]
"""
Fuzzer-discovered eager/compile divergence test cases.

All tests are marked as xfail since they represent known compilation bugs.
"""

import pytest

import torch
from torch.testing._internal.common_utils import run_tests, TestCase


class TestFuzzerCompileIssues(TestCase):
    """Test cases for fuzzer-discovered eager/compile divergence issues."""

    def setUp(self):
        """Configure common test settings."""
        torch._dynamo.config.capture_scalar_outputs = True
        torch._dynamo.config.capture_dynamic_output_shape_ops = True
        torch._inductor.config.emulate_precision_casts = True

    @pytest.mark.xfail(reason="Issue #164428")
    def test_fuzzer_issue_163876_scaled_dot_product_attention(self):
        torch.manual_seed(6804)

        def fuzzed_program(arg_0, arg_1, arg_2, sentinel):
            var_node_4 = (
                arg_0  # size=(7, 1, 32), stride=(1, 1, 0), dtype=float64, device=cuda
            )
            var_node_5 = torch.full(
                (7, 1, 32), -1.195053522845565, dtype=torch.float64
            )  # size=(7, 1, 32), stride=(1, 1, 0), dtype=float64, device=cuda
            var_node_3 = torch.div(
                var_node_4, var_node_5
            )  # size=(7, 1, 32), stride=(1, 1, 0), dtype=float64, device=cuda
            var_node_2 = torch.flatten(
                var_node_3
            )  # size=(224,), stride=(1,), dtype=float64, device=cuda
            var_node_8 = torch.full(
                (2,), -0.8316502130341195, dtype=torch.float64
            )  # size=(2,), stride=(1,), dtype=float64, device=cuda
            var_node_9 = (
                arg_1  # size=(2, 224), stride=(224, 1), dtype=float64, device=cuda
            )
            var_node_7 = torch.matmul(
                var_node_8.to(torch.float64), var_node_9.to(torch.float64)
            )  # size=(224,), stride=(1,), dtype=float64, device=cuda
            var_node_10 = arg_2  # size=(224,), stride=(1,), dtype=float64, device=cuda
            var_node_6 = torch.sub(
                var_node_7, var_node_10
            )  # size=(224,), stride=(1,), dtype=float64, device=cuda
            var_node_1 = torch.sub(
                var_node_2, var_node_6
            )  # size=(224,), stride=(1,), dtype=float64, device=cuda
            var_node_0 = var_node_1.contiguous().view(
                [16, 14]
            )  # size=(16, 14), stride=(14, 1), dtype=float64, device=cuda
            # Ensure gradient computation by multiplying with sentinel and taking real part
            result = var_node_0 * sentinel
            if result.is_complex():
                result = result.real
            return result

        # Sentinel tensor to ensure gradient computation
        sentinel = torch.tensor(1.0, requires_grad=True)

        arg_0 = torch.as_strided(
            torch.randn(7).to(torch.float64), (7, 1, 32), (1, 1, 0)
        )
        arg_1 = torch.as_strided(torch.randn(448).to(torch.float64), (2, 224), (224, 1))
        arg_2 = torch.as_strided(torch.randn(224).to(torch.float64), (224,), (1,))

        args = (arg_0, arg_1, arg_2) + (sentinel,)
        fuzzed_program(*args)
        print("✅ eager success")
        compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
        compiled_program(*args)
        print("✅ compile success")


if __name__ == "__main__":
    run_tests()
