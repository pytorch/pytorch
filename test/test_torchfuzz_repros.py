# Owner(s): ["module: tests"]
"""
Fuzzer-discovered eager/compile divergence test cases.

All tests are marked as xfail since they represent known compilation bugs.

IF YOU ARE HERE YOU LIKELY DIDN'T DO ANYTHING WRONG. In fact, you probably did something right!
All of these tests are associated with bugs the fuzzer found. If one of these tests starts failing due to your PR,
it actually means your PR fixed the bug! Feel free to delete the test and close out the issue linked from the test.
"""

import pytest

import torch
from torch.testing._internal.common_utils import run_tests, TestCase


class TestFuzzerCompileIssues(TestCase):
    """Test cases for fuzzer-discovered eager/compile divergence issues."""

    def setUp(self):
        """Configure common test settings."""
        super().setUp()
        torch._dynamo.config.capture_scalar_outputs = True
        torch._dynamo.config.capture_dynamic_output_shape_ops = True
        torch._inductor.config.emulate_precision_casts = True

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
            t6 = torch.exp(  # noqa: F841
                t5
            )  # size=(1,), stride=(1,), dtype=int64, device=cuda  # noqa: F841
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
            var_node_1 = arg0  # size=(1, 2), stride=(2, 1), dtype=int64, device=cuda  # noqa: F841
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


if __name__ == "__main__":
    run_tests()
