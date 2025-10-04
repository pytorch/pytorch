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
    def test_fuzzer_issue_164428(self):
        torch.manual_seed(6804)

        def foo(
            arg0,
            arg1,
            arg2,
            arg3,
            arg4,
            arg5,
            arg6,
            arg7,
            arg8,
            arg9,
            arg10,
            arg11,
            arg12,
            arg13,
            arg14,
        ):
            t0 = arg0  # size=(241,), stride=(4,), dtype=float16, device=cuda
            t1 = t0.contiguous()  # size=(241,), stride=(1,), dtype=float16, device=cuda
            t2 = arg1  # size=(3, 241), stride=(241, 1), dtype=float16, device=cuda
            t3 = t2.max(
                dim=0
            ).values  # size=(241,), stride=(1,), dtype=float16, device=cuda
            t4 = arg2  # size=(13,), stride=(1,), dtype=float16, device=cuda
            t5 = arg3  # size=(90,), stride=(1,), dtype=float16, device=cuda
            t6 = arg4  # size=(1,), stride=(1,), dtype=float16, device=cuda
            t7 = arg5  # size=(26,), stride=(1,), dtype=float16, device=cuda
            t8 = arg6  # size=(111,), stride=(1,), dtype=float16, device=cuda
            t9 = torch.cat(
                [t4, t5, t6, t7, t8], dim=0
            )  # size=(241,), stride=(1,), dtype=float16, device=cuda
            t10 = arg7  # size=(241,), stride=(1,), dtype=float16, device=cuda
            t11 = arg8  # size=(241,), stride=(1,), dtype=float16, device=cuda
            t12 = t9 + t10 + t11  # size=(241,), stride=(1,), dtype=float16, device=cuda
            t13 = arg9  # size=(241,), stride=(1,), dtype=float16, device=cuda
            t14 = torch.exp(t13)  # size=(241,), stride=(1,), dtype=float16, device=cuda
            t15 = torch.pow(
                torch.pow(torch.pow(torch.pow(t1, t3), t12), t9), t14
            )  # size=(241,), stride=(1,), dtype=float16, device=cuda
            t16 = arg10  # size=(5, 103), stride=(103, 1), dtype=float16, device=cuda
            t17 = t16.var(dim=0)  # size=(103,), stride=(1,), dtype=float16, device=cuda
            t18 = arg11  # size=(68, 2), stride=(2, 1), dtype=float16, device=cuda
            t19 = t18.sum(dim=1)  # size=(68,), stride=(1,), dtype=float16, device=cuda
            t20 = arg12  # size=(5, 14), stride=(14, 1), dtype=float16, device=cuda
            t21 = t20.std(dim=0)  # size=(14,), stride=(1,), dtype=float16, device=cuda
            t22 = arg13  # size=(47,), stride=(3,), dtype=float16, device=cuda
            t23 = (
                t22.contiguous()
            )  # size=(47,), stride=(1,), dtype=float16, device=cuda
            t24 = arg14  # size=(9,), stride=(1,), dtype=float16, device=cuda
            t25 = t24.clone()
            t25.zero_()  # size=(9,), stride=(1,), dtype=float16, device=cuda
            t26 = torch.cat(
                [t17, t19, t21, t23, t25], dim=0
            )  # size=(241,), stride=(1,), dtype=float16, device=cuda
            t27 = (
                ((t15) / t15) / t26
            ) / t26  # size=(241,), stride=(1,), dtype=float16, device=cuda
            output = t27  # output tensor
            return output

        arg0 = torch.rand(
            [241], dtype=torch.float16, device="cuda", requires_grad=True
        )  # size=(241,), stride=(4,), dtype=float16, device=cuda
        arg1 = torch.rand(
            [3, 241], dtype=torch.float16, device="cuda", requires_grad=True
        )  # size=(3, 241), stride=(241, 1), dtype=float16, device=cuda
        arg2 = torch.rand(
            [13], dtype=torch.float16, device="cuda", requires_grad=True
        )  # size=(13,), stride=(1,), dtype=float16, device=cuda
        arg3 = torch.rand(
            [90], dtype=torch.float16, device="cuda", requires_grad=True
        )  # size=(90,), stride=(1,), dtype=float16, device=cuda
        arg4 = torch.rand(
            [1], dtype=torch.float16, device="cuda", requires_grad=True
        )  # size=(1,), stride=(1,), dtype=float16, device=cuda
        arg5 = torch.rand(
            [26], dtype=torch.float16, device="cuda", requires_grad=True
        )  # size=(26,), stride=(1,), dtype=float16, device=cuda
        arg6 = torch.rand(
            [111], dtype=torch.float16, device="cuda", requires_grad=True
        )  # size=(111,), stride=(1,), dtype=float16, device=cuda
        arg7 = torch.rand(
            [241], dtype=torch.float16, device="cuda", requires_grad=True
        )  # size=(241,), stride=(1,), dtype=float16, device=cuda
        arg8 = torch.rand(
            [241], dtype=torch.float16, device="cuda", requires_grad=True
        )  # size=(241,), stride=(1,), dtype=float16, device=cuda
        arg9 = torch.rand(
            [241], dtype=torch.float16, device="cuda", requires_grad=True
        )  # size=(241,), stride=(1,), dtype=float16, device=cuda
        arg10 = torch.rand(
            [5, 103], dtype=torch.float16, device="cuda", requires_grad=True
        )  # size=(5, 103), stride=(103, 1), dtype=float16, device=cuda
        arg11 = torch.rand(
            [68, 2], dtype=torch.float16, device="cuda", requires_grad=True
        )  # size=(68, 2), stride=(2, 1), dtype=float16, device=cuda
        arg12 = torch.rand(
            [5, 14], dtype=torch.float16, device="cuda", requires_grad=True
        )  # size=(5, 14), stride=(14, 1), dtype=float16, device=cuda
        arg13 = torch.rand(
            [47], dtype=torch.float16, device="cuda", requires_grad=True
        )  # size=(47,), stride=(3,), dtype=float16, device=cuda
        arg14 = torch.rand(
            [9], dtype=torch.float16, device="cuda", requires_grad=True
        )  # size=(9,), stride=(1,), dtype=float16, device=cuda

        out_eager = foo(
            arg0,
            arg1,
            arg2,
            arg3,
            arg4,
            arg5,
            arg6,
            arg7,
            arg8,
            arg9,
            arg10,
            arg11,
            arg12,
            arg13,
            arg14,
        )
        out_eager.sum().backward()
        print("Eager Success! ✅")
        compiled_foo = torch.compile(foo, fullgraph=True, dynamic=True)
        out_compiled = compiled_foo(
            arg0,
            arg1,
            arg2,
            arg3,
            arg4,
            arg5,
            arg6,
            arg7,
            arg8,
            arg9,
            arg10,
            arg11,
            arg12,
            arg13,
            arg14,
        )
        out_compiled.sum().backward()
        print("Compile Success! ✅")

    @pytest.mark.xfail(reason="Issue #164484")
    def test_fuzzer_issue_164484(self):
        torch.manual_seed(9157)

        def foo(arg0, arg1, arg2, arg3):
            var_node_2 = torch.full((14, 16), 1.158473253250122, dtype=torch.float32)
            var_node_1 = torch.nn.functional.relu(var_node_2)
            var_node_6 = torch.full((14, 1), -0.94140625, dtype=torch.bfloat16)
            var_node_7 = arg0  # size=(1, 16), stride=(16, 1), dtype=bfloat16
            var_node_5 = torch.matmul(
                var_node_6.to(torch.bfloat16), var_node_7.to(torch.bfloat16)
            )
            var_node_9 = torch.full((16,), 0.76953125, dtype=torch.bfloat16)
            var_node_8 = torch.reshape(var_node_9, [16])
            var_node_11 = torch.full((16,), 2.4375, dtype=torch.bfloat16)
            var_node_10 = torch.reshape(var_node_11, [16])
            var_node_4 = torch.cat([var_node_5, var_node_8, var_node_10], dim=1)
            var_node_12 = arg1  # size=(14, 48), stride=(48, 1), dtype=bfloat16
            var_node_3 = torch.sub(var_node_4, var_node_12)
            var_node_0 = torch.add(var_node_1, var_node_3)
            var_node_14 = torch.full((14, 48), 1.4375, dtype=torch.bfloat16)
            var_node_13 = torch.nn.functional.layer_norm(var_node_14, [48])
            result = torch.add(var_node_0, var_node_13)
            output = result + arg2 + arg3
            return output

        arg0 = torch.rand(
            [1, 16], dtype=torch.bfloat16, device="cuda", requires_grad=True
        )
        arg1 = torch.rand(
            [14, 48], dtype=torch.bfloat16, device="cuda", requires_grad=True
        )
        arg2 = torch.tensor(
            0.0, dtype=torch.bfloat16, device="cuda", requires_grad=True
        )
        arg3 = torch.tensor(
            0.0, dtype=torch.bfloat16, device="cuda", requires_grad=True
        )

        out_eager = foo(arg0, arg1, arg2, arg3)
        out_eager.sum().backward()
        print("Eager Success! ✅")
        compiled_foo = torch.compile(foo, fullgraph=True, dynamic=True)
        out_compiled = compiled_foo(arg0, arg1, arg2, arg3)
        out_compiled.sum().backward()
        print("Compile Success! ✅")

    @pytest.mark.xfail(reason="Issue #164186")
    def test_fuzzer_issue_164186(self):
        torch.manual_seed(0)

        def foo(arg0):
            t0 = arg0  # size=(714, 33), stride=(33, 1), dtype=float16, device=cuda
            t1 = t0.clone()
            t1.zero_()
            t2 = t1.contiguous().view((34, 9, 77))
            t3 = t2.clone()
            t3.zero_()
            output = t3
            return output

        arg0 = torch.rand(
            [714, 33], dtype=torch.float16, device="cuda", requires_grad=True
        )

        out_eager = foo(arg0)
        out_eager.sum().backward()
        print("Eager Success! ✅")
        compiled_foo = torch.compile(foo, fullgraph=True, dynamic=True)
        out_compiled = compiled_foo(arg0)
        out_compiled.sum().backward()
        print("Compile Success! ✅")

    @pytest.mark.xfail(reason="Issue #164185")
    def test_fuzzer_issue_164185(self):
        torch.manual_seed(0)

        def foo(arg0, arg1, arg2):
            t0 = arg0  # size=(349200, 5), stride=(5, 1), dtype=bfloat16, device=cuda
            t1 = t0.mean(
                dim=1
            )  # size=(349200,), stride=(1,), dtype=bfloat16, device=cuda
            t2 = arg1  # size=(), stride=(), dtype=int64, device=cuda
            t3 = arg2  # size=(50000, 349200), stride=(50000, 1), dtype=bfloat16, device=cuda
            t4 = torch.nn.functional.embedding(
                torch.clamp(t2, 0, t3.size(0) - 1).to(torch.long), t3
            )
            t5 = torch.pow(torch.pow(torch.pow(torch.pow(t1, t4), t4), t1), t1)
            t6 = t5.contiguous().view((75, 97, 48))
            output = t6
            return output

        arg0 = torch.rand(
            [349200, 5], dtype=torch.bfloat16, device="cuda", requires_grad=True
        )
        arg1 = torch.randint(0, 50000, [], dtype=torch.int64, device="cuda")
        arg2 = torch.rand(
            [50000, 349200], dtype=torch.bfloat16, device="cuda", requires_grad=True
        )

        out_eager = foo(arg0, arg1, arg2)
        out_eager.sum().backward()
        print("Eager Success! ✅")
        compiled_foo = torch.compile(foo, fullgraph=True, dynamic=True)
        out_compiled = compiled_foo(arg0, arg1, arg2)
        out_compiled.sum().backward()
        print("Compile Success! ✅")


if __name__ == "__main__":
    run_tests()
