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

    @pytest.mark.xfail(reason="Issue #164157")
    def test_fuzzer_issue_164157(self):
        torch.manual_seed(0)

        def foo(arg0, arg1, arg2, arg3, arg4, arg5):
            t0 = arg0  # size=(47,), stride=(1,), dtype=int64, device=cuda
            t1 = torch.tanh(t0)  # size=(47,), stride=(1,), dtype=int64, device=cuda
            t2 = arg1  # size=(), stride=(), dtype=int64, device=cuda
            t3 = arg2  # size=(), stride=(), dtype=int64, device=cuda
            t4 = t2 * t3  # size=(), stride=(), dtype=int64, device=cuda
            t5 = t1.clone()
            t5.fill_(t4.item())
            t6 = (
                arg3  # size=(256, 88, 1), stride=(88, 1, 1), dtype=float16, device=cuda
            )
            t7 = (
                arg4  # size=(256, 88, 1), stride=(88, 1, 1), dtype=float16, device=cuda
            )
            t8 = (
                arg5  # size=(256, 88, 1), stride=(88, 1, 1), dtype=float16, device=cuda
            )
            t9 = torch.cat([t6, t6, t7, t8], dim=2)
            t10 = t9.std(dim=2)
            t11 = torch.nn.functional.embedding(
                torch.clamp(t5, 0, t10.size(0) - 1), t10
            )
            output = t11
            return output

        arg0 = torch.randint(0, 100, [47], dtype=torch.int64, device="cuda")
        arg1 = torch.randint(0, 10, [], dtype=torch.int64, device="cuda")
        arg2 = torch.randint(0, 10, [], dtype=torch.int64, device="cuda")
        arg3 = torch.rand(
            [256, 88, 1], dtype=torch.float16, device="cuda", requires_grad=True
        )
        arg4 = torch.rand(
            [256, 88, 1], dtype=torch.float16, device="cuda", requires_grad=True
        )
        arg5 = torch.rand(
            [256, 88, 1], dtype=torch.float16, device="cuda", requires_grad=True
        )

        out_eager = foo(arg0, arg1, arg2, arg3, arg4, arg5)
        out_eager.sum().backward()
        print("Eager Success! ✅")
        compiled_foo = torch.compile(foo, fullgraph=True, dynamic=True)
        out_compiled = compiled_foo(arg0, arg1, arg2, arg3, arg4, arg5)
        out_compiled.sum().backward()
        print("Compile Success! ✅")

    @pytest.mark.xfail(reason="Issue #164428")
    def test_fuzzer_issue_164428_already_exists(self):
        torch.manual_seed(6804)

        def foo(arg0, arg1, arg2):
            var_node_4 = (
                arg0  # size=(7, 1, 32), stride=(1, 1, 0), dtype=float64, device=cuda
            )
            var_node_5 = torch.full((7, 1, 32), -1.195053522845565, dtype=torch.float64)
            var_node_3 = torch.div(var_node_4, var_node_5)
            var_node_2 = torch.flatten(var_node_3)
            var_node_8 = torch.full((2,), -0.8316502130341195, dtype=torch.float64)
            var_node_9 = arg1  # size=(2, 224), stride=(224, 1), dtype=float64
            var_node_7 = torch.matmul(
                var_node_8.to(torch.float64), var_node_9.to(torch.float64)
            )
            var_node_10 = arg2  # size=(224,), stride=(1,), dtype=float64
            var_node_6 = torch.sub(var_node_7, var_node_10)
            var_node_1 = torch.sub(var_node_2, var_node_6)
            output = var_node_1
            return output

        arg0 = torch.rand(
            [7, 1, 32], dtype=torch.float64, device="cuda", requires_grad=True
        )
        arg1 = torch.rand(
            [2, 224], dtype=torch.float64, device="cuda", requires_grad=True
        )
        arg2 = torch.rand([224], dtype=torch.float64, device="cuda", requires_grad=True)

        out_eager = foo(arg0, arg1, arg2)
        out_eager.sum().backward()
        print("Eager Success! ✅")
        compiled_foo = torch.compile(foo, fullgraph=True, dynamic=True)
        out_compiled = compiled_foo(arg0, arg1, arg2)
        out_compiled.sum().backward()
        print("Compile Success! ✅")

    @pytest.mark.xfail(reason="Issue #164086")
    def test_fuzzer_issue_164086(self):
        torch.manual_seed(0)

        def foo(arg0, arg1, arg2, arg3, arg4, arg5):
            t0 = arg0  # size=(42, 56), stride=(42, 1), dtype=int64, device=cuda
            t1 = torch.tanh(
                t0
            )  # size=(42, 56), stride=(42, 1), dtype=int64, device=cuda
            t2 = t1.clone()
            t2.zero_()  # size=(42, 56), stride=(42, 1), dtype=int64, device=cuda
            t3 = (
                arg1  # size=(50000, 128), stride=(50000, 1), dtype=float16, device=cuda
            )
            t4 = arg2  # size=(46, 128), stride=(46, 1), dtype=float16, device=cuda
            t5 = torch.nn.functional.linear(
                t3, t4
            )  # size=(50000, 46), stride=(50000, 1), dtype=float16, device=cuda
            t6 = arg3  # size=(50000, 4, 46), stride=(184, 46, 1), dtype=float16, device=cuda
            t7 = t6.max(
                dim=1
            ).values  # size=(50000, 46), stride=(50000, 1), dtype=float16, device=cuda
            t8 = arg4  # size=(25786, 46), stride=(46, 1), dtype=float16, device=cuda
            t9 = arg5  # size=(24214, 46), stride=(46, 1), dtype=float16, device=cuda
            t10 = torch.cat(
                [t8, t9], dim=0
            )  # size=(50000, 46), stride=(50000, 1), dtype=float16, device=cuda
            t11 = torch.pow(
                torch.pow(torch.pow(torch.pow(t5, t7), t10), t5), t7
            )  # size=(50000, 46), stride=(50000, 1), dtype=float16, device=cuda
            t12 = torch.nn.functional.embedding(
                torch.clamp(t2, 0, t11.size(0) - 1).to(torch.long), t11
            )  # size=(42, 56, 46), stride=(2576, 46, 1), dtype=float16, device=cuda
            output = t12
            return output

        arg0 = torch.randint(0, 1000, [42, 56], dtype=torch.int64, device="cuda")
        arg1 = torch.rand(
            [50000, 128], dtype=torch.float16, device="cuda", requires_grad=True
        )
        arg2 = torch.rand(
            [46, 128], dtype=torch.float16, device="cuda", requires_grad=True
        )
        arg3 = torch.rand(
            [50000, 4, 46], dtype=torch.float16, device="cuda", requires_grad=True
        )
        arg4 = torch.rand(
            [25786, 46], dtype=torch.float16, device="cuda", requires_grad=True
        )
        arg5 = torch.rand(
            [24214, 46], dtype=torch.float16, device="cuda", requires_grad=True
        )

        out_eager = foo(arg0, arg1, arg2, arg3, arg4, arg5)
        out_eager.sum().backward()
        print("Eager Success! ✅")
        compiled_foo = torch.compile(foo, fullgraph=True, dynamic=True)
        out_compiled = compiled_foo(arg0, arg1, arg2, arg3, arg4, arg5)
        out_compiled.sum().backward()
        print("Compile Success! ✅")

    @pytest.mark.xfail(reason="Issue #163876")
    def test_fuzzer_issue_163876(self):
        torch.manual_seed(0)

        def foo(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7):
            t0 = arg0  # size=(29, 50, 32, 5), stride=(46400, 1600, 32, 1), dtype=float16, device=cuda
            t1 = arg1  # size=(29, 50, 32, 5), stride=(46400, 1600, 32, 1), dtype=float16, device=cuda
            t2 = arg2  # size=(29, 50, 32, 5), stride=(46400, 1600, 32, 1), dtype=float16, device=cuda
            t3 = torch.nn.functional.scaled_dot_product_attention(
                t0, t1, t2
            )  # size=(29, 50, 32, 5), stride=(8000, 160, 5, 1), dtype=float16, device=cuda
            t4 = (
                t3.min(dim=3).values
            )  # size=(29, 50, 32), stride=(1600, 32, 1), dtype=float16, device=cuda
            t5 = arg3  # size=(3, 10, 4640), stride=(46400, 4640, 1), dtype=float16, device=cuda
            t6 = t5.var(
                dim=0
            )  # size=(10, 4640), stride=(4640, 1), dtype=float16, device=cuda
            t7 = t6.reshape(
                (29, 50, 32)
            )  # size=(29, 50, 32), stride=(1600, 32, 1), dtype=float16, device=cuda
            t8 = arg4  # size=(29, 50, 32), stride=(1600, 32, 1), dtype=float16, device=cuda
            t9 = arg5  # size=(32, 50, 29), stride=(1, 32, 1600), dtype=float16, device=cuda
            t10 = t9.clone()
            t10.zero_()  # size=(32, 50, 29), stride=(1, 32, 1600), dtype=float16, device=cuda
            t11 = t10.transpose(
                0, 2
            )  # size=(29, 50, 32), stride=(1600, 32, 1), dtype=float16, device=cuda
            t12 = torch.pow(
                torch.pow(t4, t8), t11
            )  # size=(29, 50, 32), stride=(1600, 32, 1), dtype=float16, device=cuda
            t13 = arg6  # size=(29, 50, 32), stride=(1450, 50, 1), dtype=float16, device=cuda
            t15 = torch.nn.functional.layer_norm(
                t13, (32,)
            )  # size=(29, 50, 32), stride=(1600, 32, 1), dtype=float16, device=cuda
            t16 = (
                (t12) / t15
            )  # size=(29, 50, 32), stride=(1600, 32, 1), dtype=float16, device=cuda
            t17 = (
                ((((t4) - t7) - t16) - t11) - t16
            )  # size=(29, 50, 32), stride=(1600, 32, 1), dtype=float16, device=cuda
            output = t17
            return output

        arg0 = torch.rand(
            [29, 50, 32, 5], dtype=torch.float16, device="cuda", requires_grad=True
        )
        arg1 = torch.rand(
            [29, 50, 32, 5], dtype=torch.float16, device="cuda", requires_grad=True
        )
        arg2 = torch.rand(
            [29, 50, 32, 5], dtype=torch.float16, device="cuda", requires_grad=True
        )
        arg3 = torch.rand(
            [3, 10, 4640], dtype=torch.float16, device="cuda", requires_grad=True
        )
        arg4 = torch.rand(
            [29, 50, 32], dtype=torch.float16, device="cuda", requires_grad=True
        )
        arg5 = torch.rand(
            [32, 50, 29], dtype=torch.float16, device="cuda", requires_grad=True
        )
        arg6 = torch.rand(
            [29, 50, 32], dtype=torch.float16, device="cuda", requires_grad=True
        )
        arg7 = torch.randint(0, 1000, [1], dtype=torch.int64, device="cuda")

        out_eager = foo(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7)
        out_eager.sum().backward()
        print("Eager Success! ✅")
        compiled_foo = torch.compile(foo, fullgraph=True, dynamic=True)
        out_compiled = compiled_foo(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7)
        out_compiled.sum().backward()
        print("Compile Success! ✅")

    @pytest.mark.xfail(reason="Issue #163877")
    def test_fuzzer_issue_163877(self):
        torch.manual_seed(0)

        def foo(arg0, arg1):
            t0 = arg0  # size=(401120, 3), stride=(3, 1), dtype=float32, device=cuda
            t1 = t0.clone()
            t1.zero_()  # size=(401120, 3), stride=(3, 1), dtype=float32, device=cuda
            t2 = t1.reshape(
                (109, 115, 96)
            )  # size=(109, 115, 96), stride=(11040, 96, 1), dtype=float32, device=cuda
            t3 = arg1  # size=(), stride=(), dtype=float32, device=cuda
            t4 = t3.contiguous()  # size=(), stride=(), dtype=float32, device=cuda
            t5 = torch.nn.functional.relu(
                t4
            )  # size=(), stride=(), dtype=float32, device=cuda
            t6 = t2.clone()
            t6.fill_(
                t5.item()
            )  # size=(109, 115, 96), stride=(11040, 96, 1), dtype=float32, device=cuda
            output = t6
            return output

        arg0 = torch.rand(
            [401120, 3], dtype=torch.float32, device="cuda", requires_grad=True
        )
        arg1 = torch.rand([], dtype=torch.float32, device="cuda", requires_grad=True)

        out_eager = foo(arg0, arg1)
        out_eager.sum().backward()
        print("Eager Success! ✅")
        compiled_foo = torch.compile(foo, fullgraph=True, dynamic=True)
        out_compiled = compiled_foo(arg0, arg1)
        out_compiled.sum().backward()
        print("Compile Success! ✅")

    @pytest.mark.xfail(reason="Issue #163971")
    def test_fuzzer_issue_163971(self):
        torch.manual_seed(0)

        def foo(arg0):
            t0 = arg0  # size=(), stride=(), dtype=bfloat16, device=cuda
            t1 = torch.softmax(
                t0, dim=0
            )  # size=(), stride=(), dtype=bfloat16, device=cuda
            t2 = torch.nn.functional.gelu(
                t1
            )  # size=(), stride=(), dtype=bfloat16, device=cuda
            t3 = torch.softmax(
                t2, dim=0
            )  # size=(), stride=(), dtype=bfloat16, device=cuda
            output = t3
            return output

        arg0 = torch.rand([], dtype=torch.bfloat16, device="cuda", requires_grad=True)

        out_eager = foo(arg0)
        out_eager.sum().backward()
        print("Eager Success! ✅")
        compiled_foo = torch.compile(foo, fullgraph=True, dynamic=True)
        out_compiled = compiled_foo(arg0)
        out_compiled.sum().backward()
        print("Compile Success! ✅")

    @pytest.mark.xfail(reason="Issue #164059")
    def test_fuzzer_issue_164059(self):
        torch.manual_seed(0)

        def foo(arg0, arg1, arg2):
            t0 = arg0  # size=(16, 38073, 1), stride=(38073, 1, 1), dtype=float32, device=cuda
            t1 = t0.clone()
            t1.zero_()  # size=(16, 38073, 1), stride=(38073, 1, 1), dtype=float32, device=cuda
            t2 = t1.contiguous().view(
                (49, 112, 111)
            )  # size=(49, 112, 111), stride=(5488, 112, 1), dtype=float32, device=cuda
            t3 = arg1  # size=(1,), stride=(1,), dtype=int64, device=cuda
            t4 = arg2  # size=(1,), stride=(1,), dtype=int64, device=cuda
            t5 = t3 + t3 + t4  # size=(1,), stride=(1,), dtype=int64, device=cuda
            t6 = torch.exp(t5)  # size=(1,), stride=(1,), dtype=int64, device=cuda
            t7 = torch.nn.functional.layer_norm(
                t2, (111,)
            )  # size=(49, 112, 111), stride=(12432, 111, 1), dtype=float32, device=cuda
            output = t7
            return output

        arg0 = torch.rand(
            [16, 38073, 1], dtype=torch.float32, device="cuda", requires_grad=True
        )
        arg1 = torch.randint(0, 1000, [1], dtype=torch.int64, device="cuda")
        arg2 = torch.randint(0, 1000, [1], dtype=torch.int64, device="cuda")

        out_eager = foo(arg0, arg1, arg2)
        out_eager.sum().backward()
        print("Eager Success! ✅")
        compiled_foo = torch.compile(foo, fullgraph=True, dynamic=True)
        out_compiled = compiled_foo(arg0, arg1, arg2)
        out_compiled.sum().backward()
        print("Compile Success! ✅")

    @pytest.mark.xfail(reason="Issue #164088")
    def test_fuzzer_issue_164088(self):
        torch.manual_seed(0)

        def foo(arg0, arg1, arg2, arg3, arg4):
            t0 = arg0  # size=(23, 4), stride=(4, 1), dtype=bfloat16, device=cuda
            t1 = t0.clone()
            t1.zero_()  # size=(23, 4), stride=(4, 1), dtype=bfloat16, device=cuda
            t2 = t1.contiguous().view(
                (92,)
            )  # size=(92,), stride=(1,), dtype=bfloat16, device=cuda
            t3 = arg1  # size=(5, 4, 5), stride=(20, 5, 1), dtype=bfloat16, device=cuda
            t4 = t3.min()  # size=(), stride=(), dtype=bfloat16, device=cuda
            t5 = arg2  # size=(), stride=(), dtype=bfloat16, device=cuda
            t6 = torch.nn.functional.silu(
                t5
            )  # size=(), stride=(), dtype=bfloat16, device=cuda
            t7 = arg3  # size=(3, 2, 3), stride=(6, 3, 1), dtype=bfloat16, device=cuda
            t8 = t7.min()  # size=(), stride=(), dtype=bfloat16, device=cuda
            t9 = arg4  # size=(), stride=(), dtype=bfloat16, device=cuda
            t10 = ((t8) / t9) / t9  # size=(), stride=(), dtype=bfloat16, device=cuda
            t11 = (
                t4 + t4 + t6 + t10 + t8
            )  # size=(), stride=(), dtype=bfloat16, device=cuda
            t12 = t2.clone()
            t12.fill_(
                t11.item()
            )  # size=(92,), stride=(1,), dtype=bfloat16, device=cuda
            output = t12
            return output

        arg0 = torch.rand(
            [23, 4], dtype=torch.bfloat16, device="cuda", requires_grad=True
        )
        arg1 = torch.rand(
            [5, 4, 5], dtype=torch.bfloat16, device="cuda", requires_grad=True
        )
        arg2 = torch.rand([], dtype=torch.bfloat16, device="cuda", requires_grad=True)
        arg3 = torch.rand(
            [3, 2, 3], dtype=torch.bfloat16, device="cuda", requires_grad=True
        )
        arg4 = torch.rand([], dtype=torch.bfloat16, device="cuda", requires_grad=True)

        out_eager = foo(arg0, arg1, arg2, arg3, arg4)
        out_eager.sum().backward()
        print("Eager Success! ✅")
        compiled_foo = torch.compile(foo, fullgraph=True, dynamic=True)
        out_compiled = compiled_foo(arg0, arg1, arg2, arg3, arg4)
        out_compiled.sum().backward()
        print("Compile Success! ✅")

    @pytest.mark.xfail(reason="Issue #163894")
    def test_fuzzer_issue_163894(self):
        torch.manual_seed(9)

        def foo(arg0):
            var_node_1 = arg0  # size=(1, 2), stride=(2, 1), dtype=int64, device=cuda
            var_node_5 = torch.full(
                (1, 2), -66, dtype=torch.int32
            )  # size=(1, 2), stride=(2, 1), dtype=int32, device=cuda
            var_node_6 = torch.full(
                (1, 2), 77, dtype=torch.int64
            )  # size=(1, 2), stride=(2, 1), dtype=int64, device=cuda
            var_node_4 = torch.ops.aten.add(
                var_node_5, var_node_6
            )  # size=(1, 2), stride=(2, 1), dtype=int32, device=cuda
            var_node_7 = torch.full(
                (1, 2), -64, dtype=torch.int32
            )  # size=(1, 2), stride=(2, 1), dtype=int32, device=cuda
            var_node_3 = torch.ops.aten.mul(
                var_node_4, var_node_7
            )  # size=(1, 2), stride=(2, 1), dtype=int32, device=cuda
            var_node_9 = torch.full(
                (3, 4), False, dtype=torch.bool
            )  # size=(3, 4), stride=(4, 1), dtype=bool, device=cuda
            var_node_8 = torch.nonzero(
                var_node_9
            )  # size=(0, 2), stride=(2, 1), dtype=int64, device=cuda
            if var_node_8.numel() == 0:
                var_node_8 = torch.zeros((1, 2), dtype=torch.int64, device="cuda")
            var_node_2 = torch.ops.aten.add(var_node_3, var_node_8)
            output = var_node_2.float()
            return output

        arg0 = torch.randint(0, 10, [1, 2], dtype=torch.int64, device="cuda")

        out_eager = foo(arg0)
        out_eager.sum().backward()
        print("Eager Success! ✅")
        compiled_foo = torch.compile(foo, fullgraph=True, dynamic=True)
        out_compiled = compiled_foo(arg0)
        out_compiled.sum().backward()
        print("Compile Success! ✅")

    @pytest.mark.xfail(reason="Issue #164486")
    def test_fuzzer_issue_164486(self):
        torch.manual_seed(238)

        def foo(arg0):
            var_node_2 = torch.full(
                (), 1, dtype=torch.int16
            )  # size=(), stride=(), dtype=int16, device=cuda
            var_node_3 = arg0  # size=(), stride=(), dtype=int16, device=cuda
            var_node_1 = torch.add(
                var_node_2, var_node_3
            )  # size=(), stride=(), dtype=int16, device=cuda
            var_node_5 = torch.full(
                (1,), 3, dtype=torch.int16
            )  # size=(1,), stride=(1,), dtype=int16, device=cuda
            var_node_4 = torch.squeeze(
                var_node_5
            )  # size=(), stride=(), dtype=int16, device=cuda
            var_node_0 = torch.div(
                var_node_1, var_node_4
            )  # size=(), stride=(), dtype=int16, device=cuda
            result = var_node_0.float()
            return result

        arg0 = torch.randint(0, 10, [], dtype=torch.int16, device="cuda")

        out_eager = foo(arg0)
        out_eager.sum().backward()
        print("Eager Success! ✅")
        compiled_foo = torch.compile(foo, fullgraph=True, dynamic=True)
        out_compiled = compiled_foo(arg0)
        out_compiled.sum().backward()
        print("Compile Success! ✅")

    @pytest.mark.xfail(reason="Issue #163674")
    def test_fuzzer_issue_163674(self):
        torch.manual_seed(0)

        def foo(arg0, arg1, arg2):
            t0 = arg0  # size=(79488, 1, 3, 1), stride=(3, 3, 1, 1), dtype=float16, device=cuda
            t1 = t0.clone()
            t1.zero_()  # size=(79488, 1, 3, 1), stride=(3, 3, 1, 1), dtype=float16, device=cuda
            t2 = arg1  # size=(79488, 1, 3, 1), stride=(3, 3, 1, 1), dtype=float32, device=cuda
            t3 = arg2  # size=(), stride=(), dtype=float32, device=cuda
            t4 = t2.clone()
            t4.fill_(
                t3.item()
            )  # size=(79488, 1, 3, 1), stride=(3, 3, 1, 1), dtype=float32, device=cuda
            t5 = torch.pow(
                t1, t4
            )  # size=(79488, 1, 3, 1), stride=(3, 3, 1, 1), dtype=float32, device=cuda
            t6 = t5.reshape(
                (96, 69, 36)
            )  # size=(96, 69, 36), stride=(2484, 36, 1), dtype=float32, device=cuda
            output = t6
            return output

        arg0 = torch.rand(
            [79488, 1, 3, 1], dtype=torch.float16, device="cuda", requires_grad=True
        )
        arg1 = torch.rand(
            [79488, 1, 3, 1], dtype=torch.float32, device="cuda", requires_grad=True
        )
        arg2 = torch.rand([], dtype=torch.float32, device="cuda", requires_grad=True)

        out_eager = foo(arg0, arg1, arg2)
        out_eager.sum().backward()
        print("Eager Success! ✅")
        compiled_foo = torch.compile(foo, fullgraph=True, dynamic=True)
        out_compiled = compiled_foo(arg0, arg1, arg2)
        out_compiled.sum().backward()
        print("Compile Success! ✅")


if __name__ == "__main__":
    run_tests()
