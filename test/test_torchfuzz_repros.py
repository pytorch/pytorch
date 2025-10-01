"""
Fuzzer-discovered eager/compile divergence test cases.

This file contains reproduction cases for all 25 open fuzzer issues by bobrenjc93
that test eager vs compile divergence. All tests use `torch.compile()` with
`dynamic=True` and `fullgraph=True`.

Issue categories:
- InductorError (9 issues): Compilation failures in the inductor backend
- AssertionError (7 issues): Assertion failures in compiled code
- RuntimeError (4 issues): Runtime errors during compilation
- NameError (2 issues): Undefined variable errors in generated code
- Other (3 issues): Various other error types

All tests are marked as xfail since they represent known compilation bugs.
"""

import torch
import pytest
from torch.testing._internal.common_utils import run_tests, TestCase


class TestFuzzerCompileIssues(TestCase):
    """Test cases for fuzzer-discovered eager/compile divergence issues."""

    def setUp(self):
        """Configure common test settings."""
        torch._dynamo.config.capture_scalar_outputs = True
        torch._dynamo.config.capture_dynamic_output_shape_ops = True
        torch._inductor.config.emulate_precision_casts = True

    # ===== InductorError Issues =====

    @pytest.mark.xfail(reason="Issue #163876: AssertionError -991967080329279/1000000000000000")
    def test_fuzzer_issue_163876_scaled_dot_product_attention(self):
        """Fuzzer issue #163876: InductorError with scaled_dot_product_attention."""
        def foo(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7):
            t0 = arg0  # size=(29, 50, 32, 5), stride=(46400, 1600, 32, 1), dtype=float16, device=cuda
            t1 = arg1  # size=(29, 50, 32, 5), stride=(46400, 1600, 32, 1), dtype=float16, device=cuda
            t2 = arg2  # size=(29, 50, 32, 5), stride=(46400, 1600, 32, 1), dtype=float16, device=cuda
            t3 = torch.nn.functional.scaled_dot_product_attention(t0, t1, t2)  # size=(29, 50, 32, 5), stride=(8000, 160, 5, 1), dtype=float16, device=cuda
            t4 = t3.min(dim=3).values  # size=(29, 50, 32), stride=(1600, 32, 1), dtype=float16, device=cuda
            t5 = arg3  # size=(3, 10, 4640), stride=(46400, 4640, 1), dtype=float16, device=cuda
            t6 = t5.var(dim=0)  # size=(10, 4640), stride=(4640, 1), dtype=float16, device=cuda
            t7 = t6.reshape((29, 50, 32))  # size=(29, 50, 32), stride=(1600, 32, 1), dtype=float16, device=cuda
            t8 = arg4  # size=(29, 50, 32), stride=(1600, 32, 1), dtype=float16, device=cuda
            t9 = arg5  # size=(32, 50, 29), stride=(1, 32, 1600), dtype=float16, device=cuda
            t10 = t9.clone(); t10.zero_()  # size=(32, 50, 29), stride=(1, 32, 1600), dtype=float16, device=cuda
            t11 = t10.transpose(0, 2)  # size=(29, 50, 32), stride=(1600, 32, 1), dtype=float16, device=cuda
            t12 = torch.pow(torch.pow(t4, t8), t11)  # size=(29, 50, 32), stride=(1600, 32, 1), dtype=float16, device=cuda
            t13 = arg6  # size=(29, 50, 32), stride=(1450, 50, 1), dtype=float16, device=cuda
            t14 = arg7  # size=(1,), stride=(1,), dtype=int64, device=cuda
            t15 = torch.nn.functional.layer_norm(t13, (32,))  # size=(29, 50, 32), stride=(1600, 32, 1), dtype=float16, device=cuda
            t16 = (t12) / t15  # size=(29, 50, 32), stride=(1600, 32, 1), dtype=float16, device=cuda
            t17 = ((((t4) - t7) - t16) - t11) - t16  # size=(29, 50, 32), stride=(1600, 32, 1), dtype=float16, device=cuda
            return t17

        arg0 = torch.rand([29, 50, 32, 5], dtype=torch.float16, device='cuda', requires_grad=True)
        arg1 = torch.rand([29, 50, 32, 5], dtype=torch.float16, device='cuda', requires_grad=True)
        arg2 = torch.rand([29, 50, 32, 5], dtype=torch.float16, device='cuda', requires_grad=True)
        arg3 = torch.rand([3, 10, 4640], dtype=torch.float16, device='cuda', requires_grad=True)
        arg4 = torch.rand([29, 50, 32], dtype=torch.float16, device='cuda', requires_grad=True)
        arg5 = torch.rand([32, 50, 29], dtype=torch.float16, device='cuda', requires_grad=True)
        arg6 = torch.rand([29, 50, 32], dtype=torch.float16, device='cuda', requires_grad=True)
        arg7 = torch.randint(0, 1000, [1], dtype=torch.int64, device='cuda')

        out_eager = foo(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7)
        out_eager.sum().backward()

        compiled_foo = torch.compile(foo, fullgraph=True, dynamic=True)
        out_compiled = compiled_foo(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7)
        out_compiled.sum().backward()

    @pytest.mark.xfail(reason="Issue #163874: PassManager::run failed")
    def test_fuzzer_issue_163874_passmanager_fail(self):
        """Fuzzer issue #163874: InductorError PassManager::run failed."""
        def foo(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, arg12, arg13, arg14, arg15):
            t0 = arg0  # size=(2, 88, 94), stride=(8272, 94, 1), dtype=float16, device=cuda
            t1 = t0.norm(dim=0)  # size=(88, 94), stride=(94, 1), dtype=float16, device=cuda
            t2 = arg1  # size=(88, 128), stride=(128, 1), dtype=float16, device=cuda
            t3 = arg2  # size=(128, 94), stride=(94, 1), dtype=float16, device=cuda
            t4 = torch.addmm(t1, t2, t3)  # size=(88, 94), stride=(94, 1), dtype=float16, device=cuda
            t5 = arg3  # size=(88, 94, 4), stride=(376, 4, 1), dtype=float16, device=cuda
            t6 = t5.norm(dim=2)  # size=(88, 94), stride=(94, 1), dtype=float16, device=cuda
            t7 = arg4  # size=(88, 64), stride=(64, 1), dtype=float16, device=cuda
            t8 = arg5  # size=(64, 94), stride=(94, 1), dtype=float16, device=cuda
            t9 = torch.addmm(t6, t7, t8)  # size=(88, 94), stride=(94, 1), dtype=float16, device=cuda
            t10 = arg6  # size=(94, 88), stride=(1, 94), dtype=float16, device=cuda
            t11 = t10.transpose(0, 1)  # size=(88, 94), stride=(94, 1), dtype=float16, device=cuda
            t12 = torch.nn.functional.relu(t1)  # size=(88, 94), stride=(94, 1), dtype=float16, device=cuda
            t13 = ((((t9) / t11) / t1) / t12) / t1  # size=(88, 94), stride=(94, 1), dtype=float16, device=cuda
            t14 = arg7  # size=(88, 26), stride=(26, 1), dtype=float16, device=cuda
            t15 = arg8  # size=(88, 34), stride=(34, 1), dtype=float16, device=cuda
            t16 = arg9  # size=(88, 24), stride=(24, 1), dtype=float16, device=cuda
            t17 = arg10  # size=(88, 10), stride=(10, 1), dtype=float16, device=cuda
            t18 = torch.cat([t14, t15, t16, t17], dim=1)  # size=(88, 94), stride=(94, 1), dtype=float16, device=cuda
            t19 = arg11  # size=(88, 94), stride=(94, 1), dtype=float16, device=cuda
            t20 = arg12  # size=(88, 94), stride=(94, 1), dtype=float16, device=cuda
            t21 = arg13  # size=(88, 94), stride=(94, 1), dtype=float16, device=cuda
            t22 = torch.pow(torch.pow(torch.pow(t13, t19), t20), t21)  # size=(88, 94), stride=(94, 1), dtype=float16, device=cuda
            t23 = arg14  # size=(88, 256), stride=(88, 1), dtype=float16, device=cuda
            t24 = arg15  # size=(94, 256), stride=(94, 1), dtype=float16, device=cuda
            t25 = torch.nn.functional.linear(t23, t24)  # size=(88, 94), stride=(94, 1), dtype=float16, device=cuda
            t26 = ((((t18) - t22) - t6) - t1) - t25  # size=(88, 94), stride=(94, 1), dtype=float16, device=cuda
            t27 = t4 * t13 * t4 * t4 * t26  # size=(88, 94), stride=(94, 1), dtype=float16, device=cuda
            t28 = torch.pow(torch.pow(t27, t26), t13)  # size=(88, 94), stride=(94, 1), dtype=float16, device=cuda
            return t28

        arg0 = torch.rand([2, 88, 94], dtype=torch.float16, device='cuda', requires_grad=True)
        arg1 = torch.rand([88, 128], dtype=torch.float16, device='cuda', requires_grad=True)
        arg2 = torch.rand([128, 94], dtype=torch.float16, device='cuda', requires_grad=True)
        arg3 = torch.rand([88, 94, 4], dtype=torch.float16, device='cuda', requires_grad=True)
        arg4 = torch.rand([88, 64], dtype=torch.float16, device='cuda', requires_grad=True)
        arg5 = torch.rand([64, 94], dtype=torch.float16, device='cuda', requires_grad=True)
        arg6 = torch.rand([94, 88], dtype=torch.float16, device='cuda', requires_grad=True)
        arg7 = torch.rand([88, 26], dtype=torch.float16, device='cuda', requires_grad=True)
        arg8 = torch.rand([88, 34], dtype=torch.float16, device='cuda', requires_grad=True)
        arg9 = torch.rand([88, 24], dtype=torch.float16, device='cuda', requires_grad=True)
        arg10 = torch.rand([88, 10], dtype=torch.float16, device='cuda', requires_grad=True)
        arg11 = torch.rand([88, 94], dtype=torch.float16, device='cuda', requires_grad=True)
        arg12 = torch.rand([88, 94], dtype=torch.float16, device='cuda', requires_grad=True)
        arg13 = torch.rand([88, 94], dtype=torch.float16, device='cuda', requires_grad=True)
        arg14 = torch.rand([88, 256], dtype=torch.float16, device='cuda', requires_grad=True)
        arg15 = torch.rand([94, 256], dtype=torch.float16, device='cuda', requires_grad=True)

        out_eager = foo(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, arg12, arg13, arg14, arg15)
        out_eager.sum().backward()

        compiled_foo = torch.compile(foo, fullgraph=True, dynamic=True)
        out_compiled = compiled_foo(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, arg12, arg13, arg14, arg15)
        out_compiled.sum().backward()

    @pytest.mark.xfail(reason="Issue #164063: TypeError unexpected type fp32")
    def test_fuzzer_issue_164063_unexpected_fp32_type(self):
        """Fuzzer issue #164063: InductorError TypeError unexpected type fp32."""
        def foo(arg0, arg1, arg2, arg3, arg4, sentinel):
            t0 = arg0  # size=(36, 7112, 1, 1), stride=(7112, 1, 1, 1), dtype=bfloat16, device=cuda
            t1 = t0.reshape((28, 24, 3, 127))  # size=(28, 24, 3, 127), stride=(9144, 381, 127, 1), dtype=bfloat16, device=cuda
            t2 = t1.var(dim=2)  # size=(28, 24, 127), stride=(3048, 127, 1), dtype=bfloat16, device=cuda
            t3 = arg1  # size=(30, 24), stride=(30, 1), dtype=int64, device=cuda
            t4 = arg2  # size=(512, 127), stride=(512, 1), dtype=bfloat16, device=cuda
            t5 = torch.nn.functional.embedding(torch.clamp(t3, 0, t4.size(0) - 1).to(torch.long), t4)  # size=(30, 24, 127), stride=(3048, 127, 1), dtype=bfloat16, device=cuda
            t6 = arg3  # size=(30, 24, 15), stride=(720, 24, 1), dtype=bfloat16, device=cuda
            t7 = torch.nn.functional.pad(t6, [0, 1], mode='constant', value=0.0)  # size=(30, 24, 16), stride=(384, 16, 1), dtype=bfloat16, device=cuda
            t8 = arg4  # size=(30, 4, 16, 127), stride=(8128, 2032, 127, 1), dtype=bfloat16, device=cuda
            t9 = t8.sum(dim=1)  # size=(30, 16, 127), stride=(2032, 127, 1), dtype=bfloat16, device=cuda
            t10 = torch.baddbmm(t5, t7, t9)  # size=(30, 24, 127), stride=(3048, 127, 1), dtype=bfloat16, device=cuda
            t11 = torch.cat([t2, t10], dim=0)  # size=(58, 24, 127), stride=(3048, 127, 1), dtype=bfloat16, device=cuda
            return t11 + sentinel

        arg0 = torch.rand([36, 7112, 1, 1], dtype=torch.bfloat16, device='cuda', requires_grad=True)
        arg1 = torch.randint(0, 512, [30, 24], dtype=torch.int64, device='cuda')
        arg2 = torch.rand([512, 127], dtype=torch.bfloat16, device='cuda', requires_grad=True)
        arg3 = torch.rand([30, 24, 15], dtype=torch.bfloat16, device='cuda', requires_grad=True)
        arg4 = torch.rand([30, 4, 16, 127], dtype=torch.bfloat16, device='cuda', requires_grad=True)
        sentinel = torch.tensor(0.0, dtype=torch.bfloat16, device='cuda', requires_grad=True)

        out_eager = foo(arg0, arg1, arg2, arg3, arg4, sentinel)
        out_eager.sum().backward()

        compiled_foo = torch.compile(foo, fullgraph=True, dynamic=True)
        out_compiled = compiled_foo(arg0, arg1, arg2, arg3, arg4, sentinel)
        out_compiled.sum().backward()

    @pytest.mark.xfail(reason="Issue #163871: CantSplit InductorError")
    def test_fuzzer_issue_163871_cant_split(self):
        """Fuzzer issue #163871: InductorError CantSplit."""
        def foo(arg0, arg1, arg2, arg3):
            t0 = arg0  # size=(82, 72, 95), stride=(5904, 72, 1), dtype=bfloat16, device=cuda
            t1 = arg1  # size=(1,), stride=(1,), dtype=int64, device=cuda
            t2 = t1.permute(0)  # size=(1,), stride=(1,), dtype=int64, device=cuda
            t3 = torch.nn.functional.group_norm(t0, 4)  # size=(82, 72, 95), stride=(5904, 72, 1), dtype=bfloat16, device=cuda
            t4 = arg2  # size=(82, 72, 95), stride=(5904, 72, 1), dtype=bfloat16, device=cuda
            t5 = torch.nn.functional.gelu(t4)  # size=(82, 72, 95), stride=(5904, 72, 1), dtype=bfloat16, device=cuda
            t6 = arg3  # size=(72, 82, 95), stride=(72, 5904, 1), dtype=bfloat16, device=cuda
            t7 = t6.transpose(1, 0)  # size=(82, 72, 95), stride=(5904, 72, 1), dtype=bfloat16, device=cuda
            t8 = t3 * t5 * t5 * t7 * t5  # size=(82, 72, 95), stride=(5904, 72, 1), dtype=bfloat16, device=cuda
            t9 = t2.contiguous().view((1,))  # size=(1,), stride=(1,), dtype=int64, device=cuda
            t10 = torch.nn.functional.layer_norm(t8, (95,))  # size=(82, 72, 95), stride=(6840, 95, 1), dtype=bfloat16, device=cuda
            return t10

        arg0 = torch.rand([82, 72, 95], dtype=torch.bfloat16, device='cuda', requires_grad=True)
        arg1 = torch.randint(0, 1000, [1], dtype=torch.int64, device='cuda')
        arg2 = torch.rand([82, 72, 95], dtype=torch.bfloat16, device='cuda', requires_grad=True)
        arg3 = torch.rand([72, 82, 95], dtype=torch.bfloat16, device='cuda', requires_grad=True)

        out_eager = foo(arg0, arg1, arg2, arg3)
        out_eager.sum().backward()

        compiled_foo = torch.compile(foo, fullgraph=True, dynamic=True)
        out_compiled = compiled_foo(arg0, arg1, arg2, arg3)
        out_compiled.sum().backward()

    @pytest.mark.xfail(reason="Issue #164102: cannot determine truth value of Relational")
    def test_fuzzer_issue_164102_relational_truth_value(self):
        """Fuzzer issue #164102: InductorError cannot determine truth value of Relational."""
        def foo(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, sentinel):
            t0 = arg0  # size=(93, 62, 23), stride=(1426, 23, 1), dtype=bfloat16, device=cuda
            t1 = arg1  # size=(93, 62, 11), stride=(682, 11, 1), dtype=bfloat16, device=cuda
            t2 = arg2  # size=(93, 62, 10), stride=(620, 10, 1), dtype=bfloat16, device=cuda
            t3 = arg3  # size=(93, 62, 81), stride=(5022, 81, 1), dtype=bfloat16, device=cuda
            t4 = arg4  # size=(93, 62, 2), stride=(124, 2, 1), dtype=bfloat16, device=cuda
            t5 = torch.cat([t0, t1, t2, t3, t4], dim=2)  # size=(93, 62, 127), stride=(23622, 635, 4), dtype=bfloat16, device=cuda
            t6 = t5.contiguous()  # size=(93, 62, 127), stride=(7874, 127, 1), dtype=bfloat16, device=cuda
            t7 = arg5  # size=(93, 62, 8), stride=(5766, 62, 1), dtype=bfloat16, device=cuda
            t8 = torch.exp(t7)  # size=(93, 62, 8), stride=(5766, 62, 1), dtype=bfloat16, device=cuda
            t9 = torch.rms_norm(t8, (62, 8))  # size=(93, 62, 8), stride=(496, 8, 1), dtype=bfloat16, device=cuda
            t10 = arg6  # size=(77, 8, 127), stride=(1016, 127, 1), dtype=bfloat16, device=cuda
            t11 = torch.exp(t10)  # size=(77, 8, 127), stride=(1016, 127, 1), dtype=bfloat16, device=cuda
            t12 = arg7  # size=(16, 8, 15), stride=(128, 8, 1), dtype=bfloat16, device=cuda
            t13 = torch.nn.functional.interpolate(t12, size=(127,), mode='nearest')  # size=(16, 8, 127), stride=(1016, 127, 1), dtype=bfloat16, device=cuda
            t14 = torch.cat([t11, t13], dim=0)  # size=(93, 8, 127), stride=(1016, 127, 1), dtype=bfloat16, device=cuda
            t15 = torch.baddbmm(t6, t9, t14)  # size=(93, 62, 127), stride=(7874, 127, 1), dtype=bfloat16, device=cuda
            return t15 + sentinel

        arg0 = torch.rand([93, 62, 23], dtype=torch.bfloat16, device='cuda', requires_grad=True)
        arg1 = torch.rand([93, 62, 11], dtype=torch.bfloat16, device='cuda', requires_grad=True)
        arg2 = torch.rand([93, 62, 10], dtype=torch.bfloat16, device='cuda', requires_grad=True)
        arg3 = torch.rand([93, 62, 81], dtype=torch.bfloat16, device='cuda', requires_grad=True)
        arg4 = torch.rand([93, 62, 2], dtype=torch.bfloat16, device='cuda', requires_grad=True)
        arg5 = torch.rand([93, 62, 8], dtype=torch.bfloat16, device='cuda', requires_grad=True)
        arg6 = torch.rand([77, 8, 127], dtype=torch.bfloat16, device='cuda', requires_grad=True)
        arg7 = torch.rand([16, 8, 15], dtype=torch.bfloat16, device='cuda', requires_grad=True)
        sentinel = torch.tensor(0.0, dtype=torch.bfloat16, device='cuda', requires_grad=True)

        out_eager = foo(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, sentinel)
        out_eager.sum().backward()

        compiled_foo = torch.compile(foo, fullgraph=True, dynamic=True)
        out_compiled = compiled_foo(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, sentinel)
        out_compiled.sum().backward()

    @pytest.mark.xfail(reason="Issue #163875: AttributeError 'Infinity' object has no attribute '_mpf_'")
    def test_fuzzer_issue_163875_infinity_mpf_attr(self):
        """Fuzzer issue #163875: InductorError AttributeError Infinity object has no attribute '_mpf_'."""
        def foo(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8):
            t0 = arg0  # size=(65, 1024, 10), stride=(66560, 1024, 1), dtype=float32, device=cuda
            t1 = arg1  # size=(2,), stride=(1,), dtype=int64, device=cuda
            t2 = torch.nn.functional.layer_norm(t0, (1024, 10))  # size=(65, 1024, 10), stride=(66560, 1024, 1), dtype=float32, device=cuda
            t3 = arg2  # size=(134, 4, 640), stride=(2560, 640, 1), dtype=float32, device=cuda
            t4 = t3.contiguous().view((67, 1024, 5))  # size=(67, 1024, 5), stride=(68608, 1024, 1), dtype=float32, device=cuda
            t5 = torch.nn.functional.conv1d(t2, t4, stride=1, padding=0)  # size=(65, 67, 6), stride=(402, 6, 1), dtype=float32, device=cuda
            t6 = arg3  # size=(65, 67, 6), stride=(4355, 67, 1), dtype=bfloat16, device=cuda
            t7 = torch.sqrt(t6)  # size=(65, 67, 6), stride=(4355, 67, 1), dtype=bfloat16, device=cuda
            t8 = arg4  # size=(1,), stride=(1,), dtype=int64, device=cuda
            t9 = arg5  # size=(1,), stride=(1,), dtype=int64, device=cuda
            t10 = arg6  # size=(1,), stride=(1,), dtype=int64, device=cuda
            t11 = arg7  # size=(1,), stride=(1,), dtype=int64, device=cuda
            t12 = arg8  # size=(1,), stride=(1,), dtype=int64, device=cuda
            t13 = torch.pow(torch.pow(torch.pow(torch.pow(t8, t9), t10), t11), t12)  # size=(1,), stride=(1,), dtype=int64, device=cuda
            t14 = torch.nn.functional.group_norm(t7, 1)  # size=(65, 67, 6), stride=(402, 6, 1), dtype=bfloat16, device=cuda
            t15 = (((t5) - t14) - t5) - t5  # size=(65, 67, 6), stride=(402, 6, 1), dtype=float32, device=cuda
            return t15

        arg0 = torch.rand([65, 1024, 10], dtype=torch.float32, device='cuda', requires_grad=True)
        arg1 = torch.randint(0, 1000, [2], dtype=torch.int64, device='cuda')
        arg2 = torch.rand([134, 4, 640], dtype=torch.float32, device='cuda', requires_grad=True)
        arg3 = torch.rand([65, 67, 6], dtype=torch.bfloat16, device='cuda', requires_grad=True)
        arg4 = torch.randint(0, 1000, [1], dtype=torch.int64, device='cuda')
        arg5 = torch.randint(0, 1000, [1], dtype=torch.int64, device='cuda')
        arg6 = torch.randint(0, 1000, [1], dtype=torch.int64, device='cuda')
        arg7 = torch.randint(0, 1000, [1], dtype=torch.int64, device='cuda')
        arg8 = torch.randint(0, 1000, [1], dtype=torch.int64, device='cuda')

        out_eager = foo(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8)
        out_eager.sum().backward()

        compiled_foo = torch.compile(foo, fullgraph=True, dynamic=True)
        out_compiled = compiled_foo(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8)
        out_compiled.sum().backward()

    @pytest.mark.xfail(reason="Issue #164185: XBLOCK too large. Maximum: 4096. Actual: 8192")
    def test_fuzzer_issue_164185_xblock_too_large(self):
        """Fuzzer issue #164185: InductorError XBLOCK too large."""
        def foo(arg0, arg1, arg2, sentinel):
            t0 = arg0  # size=(349200, 5), stride=(5, 1), dtype=bfloat16, device=cuda
            t1 = t0.mean(dim=1)  # size=(349200,), stride=(1,), dtype=bfloat16, device=cuda
            t2 = arg1  # size=(), stride=(), dtype=int64, device=cuda
            t3 = arg2  # size=(50000, 349200), stride=(50000, 1), dtype=bfloat16, device=cuda
            t4 = torch.nn.functional.embedding(torch.clamp(t2, 0, t3.size(0) - 1).to(torch.long), t3)  # size=(349200,), stride=(1,), dtype=bfloat16, device=cuda
            t5 = torch.pow(torch.pow(torch.pow(torch.pow(t1, t4), t4), t1), t1)  # size=(349200,), stride=(1,), dtype=bfloat16, device=cuda
            t6 = t5.contiguous().view((75, 97, 48))  # size=(75, 97, 48), stride=(4656, 48, 1), dtype=bfloat16, device=cuda
            return t6 + sentinel

        arg0 = torch.rand([349200, 5], dtype=torch.bfloat16, device='cuda', requires_grad=True)
        arg1 = torch.randint(0, 50000, [], dtype=torch.int64, device='cuda')
        arg2 = torch.rand([50000, 349200], dtype=torch.bfloat16, device='cuda', requires_grad=True)
        sentinel = torch.tensor(0.0, dtype=torch.bfloat16, device='cuda', requires_grad=True)

        out_eager = foo(arg0, arg1, arg2, sentinel)
        out_eager.sum().backward()

        compiled_foo = torch.compile(foo, fullgraph=True, dynamic=True)
        out_compiled = compiled_foo(arg0, arg1, arg2, sentinel)
        out_compiled.sum().backward()

    @pytest.mark.xfail(reason="Issue #164157: IncompatibleTypeErrorImpl invalid operands pointer<fp16> and float64")
    def test_fuzzer_issue_164157_incompatible_type_fp16_float64(self):
        """Fuzzer issue #164157: InductorError IncompatibleTypeErrorImpl invalid operands pointer<fp16> and float64."""
        def foo(arg0, arg1, arg2, arg3, arg4, arg5, sentinel):
            t0 = arg0  # size=(47,), stride=(1,), dtype=int64, device=cuda
            t1 = torch.tanh(t0)  # size=(47,), stride=(1,), dtype=int64, device=cuda
            t2 = arg1  # size=(), stride=(), dtype=int64, device=cuda
            t3 = arg2  # size=(), stride=(), dtype=int64, device=cuda
            t4 = t2 * t3  # size=(), stride=(), dtype=int64, device=cuda
            t5 = t1.clone(); t5.fill_(t4.item())  # size=(47,), stride=(1,), dtype=int64, device=cuda
            t6 = arg3  # size=(256, 88, 1), stride=(88, 1, 1), dtype=float16, device=cuda
            t7 = arg4  # size=(256, 88, 1), stride=(88, 1, 1), dtype=float16, device=cuda
            t8 = arg5  # size=(256, 88, 1), stride=(88, 1, 1), dtype=float16, device=cuda
            t9 = torch.cat([t6, t6, t7, t8], dim=2)  # size=(256, 88, 4), stride=(352, 4, 1), dtype=float16, device=cuda
            t10 = t9.std(dim=2)  # size=(256, 88), stride=(256, 1), dtype=float16, device=cuda
            t11 = torch.nn.functional.embedding(torch.clamp(t5, 0, t10.size(0) - 1).to(torch.long), t10)  # size=(47, 88), stride=(88, 1), dtype=float16, device=cuda
            return t11 + sentinel

        arg0 = torch.randint(0, 1000, [47], dtype=torch.int64, device='cuda')
        arg1 = torch.randint(0, 1000, [], dtype=torch.int64, device='cuda')
        arg2 = torch.randint(0, 1000, [], dtype=torch.int64, device='cuda')
        arg3 = torch.rand([256, 88, 1], dtype=torch.float16, device='cuda', requires_grad=True)
        arg4 = torch.rand([256, 88, 1], dtype=torch.float16, device='cuda', requires_grad=True)
        arg5 = torch.rand([256, 88, 1], dtype=torch.float16, device='cuda', requires_grad=True)
        sentinel = torch.tensor(0.0, dtype=torch.float16, device='cuda', requires_grad=True)

        out_eager = foo(arg0, arg1, arg2, arg3, arg4, arg5, sentinel)
        out_eager.sum().backward()

        compiled_foo = torch.compile(foo, fullgraph=True, dynamic=True)
        out_compiled = compiled_foo(arg0, arg1, arg2, arg3, arg4, arg5, sentinel)
        out_compiled.sum().backward()

    @pytest.mark.xfail(reason="Issue #164086: IncompatibleTypeErrorImpl invalid operands pointer<fp16> and float64")
    def test_fuzzer_issue_164086_incompatible_type_fp16_float64(self):
        """Fuzzer issue #164086: InductorError IncompatibleTypeErrorImpl invalid operands pointer<fp16> and float64."""
        def foo(arg0, arg1, arg2, arg3, arg4, arg5, sentinel):
            t0 = arg0  # size=(42, 56), stride=(42, 1), dtype=int64, device=cuda
            t1 = torch.tanh(t0)  # size=(42, 56), stride=(42, 1), dtype=int64, device=cuda
            t2 = t1.clone(); t2.zero_()  # size=(42, 56), stride=(42, 1), dtype=int64, device=cuda
            t3 = arg1  # size=(50000, 128), stride=(50000, 1), dtype=float16, device=cuda
            t4 = arg2  # size=(46, 128), stride=(46, 1), dtype=float16, device=cuda
            t5 = torch.nn.functional.linear(t3, t4)  # size=(50000, 46), stride=(50000, 1), dtype=float16, device=cuda
            t6 = arg3  # size=(50000, 4, 46), stride=(184, 46, 1), dtype=float16, device=cuda
            t7 = t6.max(dim=1).values  # size=(50000, 46), stride=(50000, 1), dtype=float16, device=cuda
            t8 = arg4  # size=(25786, 46), stride=(46, 1), dtype=float16, device=cuda
            t9 = arg5  # size=(24214, 46), stride=(46, 1), dtype=float16, device=cuda
            t10 = torch.cat([t8, t9], dim=0)  # size=(50000, 46), stride=(50000, 1), dtype=float16, device=cuda
            t11 = torch.pow(torch.pow(torch.pow(torch.pow(t5, t7), t10), t5), t7)  # size=(50000, 46), stride=(50000, 1), dtype=float16, device=cuda
            t12 = torch.nn.functional.embedding(torch.clamp(t2, 0, t11.size(0) - 1).to(torch.long), t11)  # size=(42, 56, 46), stride=(2576, 46, 1), dtype=float16, device=cuda
            return t12 + sentinel

        arg0 = torch.randint(0, 1000, [42, 56], dtype=torch.int64, device='cuda')
        arg1 = torch.rand([50000, 128], dtype=torch.float16, device='cuda', requires_grad=True)
        arg2 = torch.rand([46, 128], dtype=torch.float16, device='cuda', requires_grad=True)
        arg3 = torch.rand([50000, 4, 46], dtype=torch.float16, device='cuda', requires_grad=True)
        arg4 = torch.rand([25786, 46], dtype=torch.float16, device='cuda', requires_grad=True)
        arg5 = torch.rand([24214, 46], dtype=torch.float16, device='cuda', requires_grad=True)
        sentinel = torch.tensor(0.0, dtype=torch.float16, device='cuda', requires_grad=True)

        out_eager = foo(arg0, arg1, arg2, arg3, arg4, arg5, sentinel)
        out_eager.sum().backward()

        compiled_foo = torch.compile(foo, fullgraph=True, dynamic=True)
        out_compiled = compiled_foo(arg0, arg1, arg2, arg3, arg4, arg5, sentinel)
        out_compiled.sum().backward()

    # ===== AssertionError Issues =====

    @pytest.mark.xfail(reason="Issue #163872: AssertionError expected stride 1==52")
    def test_fuzzer_issue_163872_stride_mismatch(self):
        """Fuzzer issue #163872: AssertionError expected stride 1==52."""
        def foo(arg0, arg1, arg2):
            t0 = arg0  # size=(91, 64, 52), stride=(5824, 64, 1), dtype=float32, device=cuda
            t1 = arg1  # size=(2,), stride=(1,), dtype=int64, device=cuda
            t2 = torch.nn.functional.layer_norm(t0, (64, 52))  # size=(91, 64, 52), stride=(5824, 64, 1), dtype=float32, device=cuda
            t3 = arg2  # size=(7, 5, 64), stride=(448, 1, 64), dtype=float32, device=cuda
            t4 = t3.permute(0, 2, 1)  # size=(7, 64, 5), stride=(448, 64, 1), dtype=float32, device=cuda
            t5 = torch.nn.functional.conv1d(t2, t4, stride=1, padding=0)  # size=(91, 7, 48), stride=(336, 48, 1), dtype=float32, device=cuda
            t6 = torch.tanh(t5)  # size=(91, 7, 48), stride=(336, 48, 1), dtype=float32, device=cuda
            return t6

        arg0 = torch.rand([91, 64, 52], dtype=torch.float32, device='cuda', requires_grad=True)
        arg1 = torch.randint(0, 1000, [2], dtype=torch.int64, device='cuda')
        arg2 = torch.rand([7, 5, 64], dtype=torch.float32, device='cuda', requires_grad=True)

        out_eager = foo(arg0, arg1, arg2)
        out_eager.sum().backward()

        compiled_foo = torch.compile(foo, fullgraph=True, dynamic=True)
        out_compiled = compiled_foo(arg0, arg1, arg2)
        out_compiled.sum().backward()

    @pytest.mark.xfail(reason="Issue #163878: AssertionError expected stride 1==94")
    def test_fuzzer_issue_163878_conv1d_stride_mismatch(self):
        """Fuzzer issue #163878: AssertionError expected stride 1==94."""
        def foo(arg0, arg1, arg2, arg3, arg4):
            t0 = arg0  # size=(40, 128, 1024), stride=(131072, 1024, 1), dtype=float32, device=cuda
            t1 = arg1  # size=(40, 1024, 94), stride=(96256, 94, 1), dtype=float32, device=cuda
            t2 = torch.bmm(t0, t1)  # size=(40, 128, 94), stride=(5120, 128, 1), dtype=float32, device=cuda
            t3 = arg2  # size=(21, 1, 128), stride=(2688, 1, 128), dtype=float32, device=cuda
            t4 = t3.permute(0, 2, 1)  # size=(21, 128, 1), stride=(2688, 128, 1), dtype=float32, device=cuda
            t5 = torch.nn.functional.conv1d(t2, t4, stride=1, padding=0)  # size=(40, 21, 94), stride=(840, 21, 1), dtype=float32, device=cuda
            t6 = arg3  # size=(5, 1), stride=(1, 1), dtype=int64, device=cuda
            t7 = arg4  # size=(5, 1), stride=(1, 1), dtype=int64, device=cuda
            t8 = ((t6) / t6) / t7  # size=(5, 1), stride=(1, 1), dtype=int64, device=cuda
            t9 = t8.min(dim=0).values  # size=(1,), stride=(1,), dtype=int64, device=cuda
            t10 = torch.nn.functional.group_norm(t5, 1)  # size=(40, 21, 94), stride=(1974, 94, 1), dtype=float32, device=cuda
            return t10

        arg0 = torch.rand([40, 128, 1024], dtype=torch.float32, device='cuda', requires_grad=True)
        arg1 = torch.rand([40, 1024, 94], dtype=torch.float32, device='cuda', requires_grad=True)
        arg2 = torch.rand([21, 1, 128], dtype=torch.float32, device='cuda', requires_grad=True)
        arg3 = torch.randint(0, 1000, [5, 1], dtype=torch.int64, device='cuda')
        arg4 = torch.randint(0, 1000, [5, 1], dtype=torch.int64, device='cuda')

        out_eager = foo(arg0, arg1, arg2, arg3, arg4)
        out_eager.sum().backward()

        compiled_foo = torch.compile(foo, fullgraph=True, dynamic=True)
        out_compiled = compiled_foo(arg0, arg1, arg2, arg3, arg4)
        out_compiled.sum().backward()

    @pytest.mark.xfail(reason="Issue #163877: Node full_1 was invalid, but is output")
    def test_fuzzer_issue_163877_invalid_node_output(self):
        """Fuzzer issue #163877: AssertionError Node full_1 was invalid, but is output."""
        def foo(arg0, arg1):
            t0 = arg0  # size=(401120, 3), stride=(3, 1), dtype=float32, device=cuda
            t1 = t0.clone(); t1.zero_()  # size=(401120, 3), stride=(3, 1), dtype=float32, device=cuda
            t2 = t1.reshape((109, 115, 96))  # size=(109, 115, 96), stride=(11040, 96, 1), dtype=float32, device=cuda
            t3 = arg1  # size=(), stride=(), dtype=float32, device=cuda
            t4 = t3.contiguous()  # size=(), stride=(), dtype=float32, device=cuda
            t5 = torch.nn.functional.relu(t4)  # size=(), stride=(), dtype=float32, device=cuda
            t6 = t2.clone(); t6.fill_(t5.item())  # size=(109, 115, 96), stride=(11040, 96, 1), dtype=float32, device=cuda
            return t6

        arg0 = torch.rand([401120, 3], dtype=torch.float32, device='cuda', requires_grad=True)
        arg1 = torch.rand([], dtype=torch.float32, device='cuda', requires_grad=True)

        out_eager = foo(arg0, arg1)
        out_eager.sum().backward()

        compiled_foo = torch.compile(foo, fullgraph=True, dynamic=True)
        out_compiled = compiled_foo(arg0, arg1)
        out_compiled.sum().backward()

    @pytest.mark.xfail(reason="Issue #164186: Node add_25 was invalid, but is output")
    def test_fuzzer_issue_164186_invalid_add_node(self):
        """Fuzzer issue #164186: AssertionError Node add_25 was invalid, but is output."""
        def foo(arg0, sentinel):
            t0 = arg0  # size=(714, 33), stride=(33, 1), dtype=float16, device=cuda
            t1 = t0.clone(); t1.zero_()  # size=(714, 33), stride=(33, 1), dtype=float16, device=cuda
            t2 = t1.contiguous().view((34, 9, 77))  # size=(34, 9, 77), stride=(693, 77, 1), dtype=float16, device=cuda
            t3 = t2.clone(); t3.zero_()  # size=(34, 9, 77), stride=(693, 77, 1), dtype=float16, device=cuda
            return t3 + sentinel

        arg0 = torch.rand([714, 33], dtype=torch.float16, device='cuda', requires_grad=True)
        sentinel = torch.tensor(0.0, dtype=torch.float16, device='cuda', requires_grad=True)

        out_eager = foo(arg0, sentinel)
        out_eager.sum().backward()

        compiled_foo = torch.compile(foo, fullgraph=True, dynamic=True)
        out_compiled = compiled_foo(arg0, sentinel)
        out_compiled.sum().backward()

    @pytest.mark.xfail(reason="Issue #164059: assert not waiting and len(ready) == len(graph.nodes)")
    def test_fuzzer_issue_164059_scheduler_assertion(self):
        """Fuzzer issue #164059: AssertionError scheduler not waiting and len(ready) == len(graph.nodes)."""
        def foo(arg0, arg1, arg2):
            t0 = arg0  # size=(16, 38073, 1), stride=(38073, 1, 1), dtype=float32, device=cuda
            t1 = t0.clone(); t1.zero_()  # size=(16, 38073, 1), stride=(38073, 1, 1), dtype=float32, device=cuda
            t2 = t1.contiguous().view((49, 112, 111))  # size=(49, 112, 111), stride=(5488, 112, 1), dtype=float32, device=cuda
            t3 = arg1  # size=(1,), stride=(1,), dtype=int64, device=cuda
            t4 = arg2  # size=(1,), stride=(1,), dtype=int64, device=cuda
            t5 = t3 + t3 + t4  # size=(1,), stride=(1,), dtype=int64, device=cuda
            t6 = torch.exp(t5)  # size=(1,), stride=(1,), dtype=int64, device=cuda
            t7 = torch.nn.functional.layer_norm(t2, (111,))  # size=(49, 112, 111), stride=(12432, 111, 1), dtype=float32, device=cuda
            return t7

        arg0 = torch.rand([16, 38073, 1], dtype=torch.float32, device='cuda', requires_grad=True)
        arg1 = torch.randint(0, 1000, [1], dtype=torch.int64, device='cuda')
        arg2 = torch.randint(0, 1000, [1], dtype=torch.int64, device='cuda')

        out_eager = foo(arg0, arg1, arg2)
        out_eager.sum().backward()

        compiled_foo = torch.compile(foo, fullgraph=True, dynamic=True)
        out_compiled = compiled_foo(arg0, arg1, arg2)
        out_compiled.sum().backward()

    @pytest.mark.xfail(reason="Issue #163572: Node full_default_1 was invalid, but is output")
    def test_fuzzer_issue_163572_invalid_full_default_node(self):
        """Fuzzer issue #163572: AssertionError Node full_default_1 was invalid, but is output."""
        def foo(arg0):
            t0 = arg0  # size=(11, 3, 3), stride=(9, 3, 1), dtype=float32, device=cuda
            t1 = t0.clone(); t1.zero_()  # size=(11, 3, 3), stride=(9, 3, 1), dtype=float32, device=cuda
            t2 = t1.contiguous().view((99,))  # size=(99,), stride=(1,), dtype=float32, device=cuda
            t3 = t2.clone(); t3.zero_()  # size=(99,), stride=(1,), dtype=float32, device=cuda
            return t3

        arg0 = torch.rand([11, 3, 3], dtype=torch.float32, device='cuda', requires_grad=True)

        out_eager = foo(arg0)
        out_eager.sum().backward()

        compiled_foo = torch.compile(foo, fullgraph=True, dynamic=True)
        out_compiled = compiled_foo(arg0)
        out_compiled.sum().backward()

    @pytest.mark.xfail(reason="Issue #163603: assertion failure not waiting and len(ready) == len(graph.nodes)")
    def test_fuzzer_issue_163603_scheduler_assertion_2(self):
        """Fuzzer issue #163603: AssertionError assertion failure not waiting and len(ready) == len(graph.nodes)."""
        def foo(arg0, arg1, arg2):
            t0 = arg0  # size=(3556, 60), stride=(60, 1), dtype=float16, device=cuda
            t1 = t0.clone(); t1.zero_()  # size=(3556, 60), stride=(60, 1), dtype=float16, device=cuda
            t2 = t1.contiguous().view((420, 508))  # size=(420, 508), stride=(420, 1), dtype=float16, device=cuda
            t3 = arg1  # size=(1,), stride=(1,), dtype=int64, device=cuda
            t4 = (((t3) / t3) / t3) / t3  # size=(1,), stride=(1,), dtype=int64, device=cuda
            t5 = arg2  # size=(), stride=(), dtype=int64, device=cuda
            t6 = t5 + t5  # size=(), stride=(), dtype=int64, device=cuda
            t7 = t4.clone(); t7.fill_(t6.item())  # size=(1,), stride=(1,), dtype=int64, device=cuda
            t8 = torch.nn.functional.layer_norm(t2, (508,))  # size=(420, 508), stride=(508, 1), dtype=float16, device=cuda
            return t8

        arg0 = torch.rand([3556, 60], dtype=torch.float16, device='cuda', requires_grad=True)
        arg1 = torch.randint(0, 1000, [1], dtype=torch.int64, device='cuda')
        arg2 = torch.randint(0, 1000, [], dtype=torch.int64, device='cuda')

        out_eager = foo(arg0, arg1, arg2)
        out_eager.sum().backward()

        compiled_foo = torch.compile(foo, fullgraph=True, dynamic=True)
        out_compiled = compiled_foo(arg0, arg1, arg2)
        out_compiled.sum().backward()

    # ===== RuntimeError Issues =====

    @pytest.mark.xfail(reason="Issue #163611: DTensor RuntimeError")
    def test_fuzzer_issue_163611_dtensor_runtime_error(self):
        """Fuzzer issue #163611: RuntimeError with DTensor."""
        def foo(arg0, arg1, arg2, arg3):
            t0 = arg0  # size=(5, 3), stride=(3, 1), dtype=float16, device=cuda
            t1 = arg1  # size=(5, 3), stride=(3, 1), dtype=float16, device=cuda
            t2 = arg2  # size=(5, 3), stride=(3, 1), dtype=float16, device=cuda
            t3 = arg3  # size=(5, 3), stride=(3, 1), dtype=float16, device=cuda
            t4 = t0 + t1 + t2 + t0 + t3  # size=(5, 3), stride=(3, 1), dtype=float16, device=cuda
            t5 = t4.sum()  # size=(), stride=(), dtype=float16, device=cuda
            t6 = torch.tanh(t5)  # size=(), stride=(), dtype=float16, device=cuda
            return t6

        # Skip DTensor setup for simplicity - test regular tensors
        arg0 = torch.rand([5, 3], dtype=torch.float16, device='cuda', requires_grad=True)
        arg1 = torch.rand([5, 3], dtype=torch.float16, device='cuda', requires_grad=True)
        arg2 = torch.rand([5, 3], dtype=torch.float16, device='cuda', requires_grad=True)
        arg3 = torch.rand([5, 3], dtype=torch.float16, device='cuda', requires_grad=True)

        out_eager = foo(arg0, arg1, arg2, arg3)
        out_eager.sum().backward()

        compiled_foo = torch.compile(foo, fullgraph=True, dynamic=True)
        out_compiled = compiled_foo(arg0, arg1, arg2, arg3)
        out_compiled.sum().backward()

    @pytest.mark.xfail(reason="Issue #163567: PassManager::run failed on backwards only")
    def test_fuzzer_issue_163567_passmanager_backwards_fail(self):
        """Fuzzer issue #163567: RuntimeError PassManager::run failed on backwards only."""
        def foo(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, arg12, arg13, arg14):
            t0 = arg0  # size=(241,), stride=(4,), dtype=float16, device=cuda
            t1 = t0.contiguous()  # size=(241,), stride=(1,), dtype=float16, device=cuda
            t2 = arg1  # size=(3, 241), stride=(241, 1), dtype=float16, device=cuda
            t3 = t2.max(dim=0).values  # size=(241,), stride=(1,), dtype=float16, device=cuda
            t4 = arg2  # size=(13,), stride=(1,), dtype=float16, device=cuda
            t5 = arg3  # size=(90,), stride=(1,), dtype=float16, device=cuda
            t6 = arg4  # size=(1,), stride=(1,), dtype=float16, device=cuda
            t7 = arg5  # size=(26,), stride=(1,), dtype=float16, device=cuda
            t8 = arg6  # size=(111,), stride=(1,), dtype=float16, device=cuda
            t9 = torch.cat([t4, t5, t6, t7, t8], dim=0)  # size=(241,), stride=(1,), dtype=float16, device=cuda
            t10 = arg7  # size=(241,), stride=(1,), dtype=float16, device=cuda
            t11 = arg8  # size=(241,), stride=(1,), dtype=float16, device=cuda
            t12 = t9 + t10 + t11  # size=(241,), stride=(1,), dtype=float16, device=cuda
            t13 = arg9  # size=(241,), stride=(1,), dtype=float16, device=cuda
            t14 = torch.exp(t13)  # size=(241,), stride=(1,), dtype=float16, device=cuda
            t15 = torch.pow(torch.pow(torch.pow(torch.pow(t1, t3), t12), t9), t14)  # size=(241,), stride=(1,), dtype=float16, device=cuda
            t16 = arg10  # size=(5, 103), stride=(103, 1), dtype=float16, device=cuda
            t17 = t16.var(dim=0)  # size=(103,), stride=(1,), dtype=float16, device=cuda
            t18 = arg11  # size=(68, 2), stride=(2, 1), dtype=float16, device=cuda
            t19 = t18.sum(dim=1)  # size=(68,), stride=(1,), dtype=float16, device=cuda
            t20 = arg12  # size=(5, 14), stride=(14, 1), dtype=float16, device=cuda
            t21 = t20.std(dim=0)  # size=(14,), stride=(1,), dtype=float16, device=cuda
            t22 = arg13  # size=(47,), stride=(3,), dtype=float16, device=cuda
            t23 = t22.contiguous()  # size=(47,), stride=(1,), dtype=float16, device=cuda
            t24 = arg14  # size=(9,), stride=(1,), dtype=float16, device=cuda
            t25 = t24.clone(); t25.zero_()  # size=(9,), stride=(1,), dtype=float16, device=cuda
            t26 = torch.cat([t17, t19, t21, t23, t25], dim=0)  # size=(241,), stride=(1,), dtype=float16, device=cuda
            t27 = (((t15) / t15) / t26) / t26  # size=(241,), stride=(1,), dtype=float16, device=cuda
            return t27

        arg0 = torch.rand([241], dtype=torch.float16, device='cuda', requires_grad=True)
        arg1 = torch.rand([3, 241], dtype=torch.float16, device='cuda', requires_grad=True)
        arg2 = torch.rand([13], dtype=torch.float16, device='cuda', requires_grad=True)
        arg3 = torch.rand([90], dtype=torch.float16, device='cuda', requires_grad=True)
        arg4 = torch.rand([1], dtype=torch.float16, device='cuda', requires_grad=True)
        arg5 = torch.rand([26], dtype=torch.float16, device='cuda', requires_grad=True)
        arg6 = torch.rand([111], dtype=torch.float16, device='cuda', requires_grad=True)
        arg7 = torch.rand([241], dtype=torch.float16, device='cuda', requires_grad=True)
        arg8 = torch.rand([241], dtype=torch.float16, device='cuda', requires_grad=True)
        arg9 = torch.rand([241], dtype=torch.float16, device='cuda', requires_grad=True)
        arg10 = torch.rand([5, 103], dtype=torch.float16, device='cuda', requires_grad=True)
        arg11 = torch.rand([68, 2], dtype=torch.float16, device='cuda', requires_grad=True)
        arg12 = torch.rand([5, 14], dtype=torch.float16, device='cuda', requires_grad=True)
        arg13 = torch.rand([47], dtype=torch.float16, device='cuda', requires_grad=True)
        arg14 = torch.rand([9], dtype=torch.float16, device='cuda', requires_grad=True)

        out_eager = foo(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, arg12, arg13, arg14)
        out_eager.sum().backward()

        compiled_foo = torch.compile(foo, fullgraph=True, dynamic=True)
        out_compiled = compiled_foo(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, arg12, arg13, arg14)
        out_compiled.sum().backward()

    @pytest.mark.xfail(reason="Issue #163451: operand #0 does not dominate this use")
    def test_fuzzer_issue_163451_operand_dominate(self):
        """Fuzzer issue #163451: RuntimeError operand #0 does not dominate this use."""
        def foo(arg0, arg1):
            t0 = arg0  # size=(5, 1, 1, 1), stride=(1, 1, 1, 1), dtype=bfloat16, device=cuda
            t1 = t0.sum(dim=0)  # size=(1, 1, 1), stride=(1, 1, 1), dtype=bfloat16, device=cuda
            t2 = arg1  # size=(1, 1, 1, 2), stride=(2, 2, 2, 1), dtype=bfloat16, device=cuda
            t3 = t2.sum(dim=3)  # size=(1, 1, 1), stride=(1, 1, 1), dtype=bfloat16, device=cuda
            t4 = torch.pow(torch.pow(t1, t1), t3)  # size=(1, 1, 1), stride=(1, 1, 1), dtype=bfloat16, device=cuda
            t5 = t4.view(())  # size=(), stride=(), dtype=bfloat16, device=cuda
            return t5

        arg0 = torch.rand([5, 1, 1, 1], dtype=torch.bfloat16, device='cuda', requires_grad=True)
        arg1 = torch.rand([1, 1, 1, 2], dtype=torch.bfloat16, device='cuda', requires_grad=True)

        out_eager = foo(arg0, arg1)
        out_eager.sum().backward()

        compiled_foo = torch.compile(foo, fullgraph=True, dynamic=True)
        out_compiled = compiled_foo(arg0, arg1)
        out_compiled.sum().backward()

    @pytest.mark.xfail(reason="Issue #164088: Argument 'sym_size_int_3' was used before it has been defined")
    def test_fuzzer_issue_164088_undefined_sym_size(self):
        """Fuzzer issue #164088: RuntimeError Argument 'sym_size_int_3' was used before it has been defined."""
        def foo(arg0, arg1, arg2, arg3, arg4, sentinel):
            t0 = arg0  # size=(23, 4), stride=(4, 1), dtype=bfloat16, device=cuda
            t1 = t0.clone(); t1.zero_()  # size=(23, 4), stride=(4, 1), dtype=bfloat16, device=cuda
            t2 = t1.contiguous().view((92,))  # size=(92,), stride=(1,), dtype=bfloat16, device=cuda
            t3 = arg1  # size=(5, 4, 5), stride=(20, 5, 1), dtype=bfloat16, device=cuda
            t4 = t3.min()  # size=(), stride=(), dtype=bfloat16, device=cuda
            t5 = arg2  # size=(), stride=(), dtype=bfloat16, device=cuda
            t6 = torch.nn.functional.silu(t5)  # size=(), stride=(), dtype=bfloat16, device=cuda
            t7 = arg3  # size=(3, 2, 3), stride=(6, 3, 1), dtype=bfloat16, device=cuda
            t8 = t7.min()  # size=(), stride=(), dtype=bfloat16, device=cuda
            t9 = arg4  # size=(), stride=(), dtype=bfloat16, device=cuda
            t10 = ((t8) / t9) / t9  # size=(), stride=(), dtype=bfloat16, device=cuda
            t11 = t4 + t4 + t6 + t10 + t8  # size=(), stride=(), dtype=bfloat16, device=cuda
            t12 = t2.clone(); t12.fill_(t11.item())  # size=(92,), stride=(1,), dtype=bfloat16, device=cuda
            return t12 + sentinel

        arg0 = torch.rand([23, 4], dtype=torch.bfloat16, device='cuda', requires_grad=True)
        arg1 = torch.rand([5, 4, 5], dtype=torch.bfloat16, device='cuda', requires_grad=True)
        arg2 = torch.rand([], dtype=torch.bfloat16, device='cuda', requires_grad=True)
        arg3 = torch.rand([3, 2, 3], dtype=torch.bfloat16, device='cuda', requires_grad=True)
        arg4 = torch.rand([], dtype=torch.bfloat16, device='cuda', requires_grad=True)
        sentinel = torch.tensor(0.0, dtype=torch.bfloat16, device='cuda', requires_grad=True)

        out_eager = foo(arg0, arg1, arg2, arg3, arg4, sentinel)
        out_eager.sum().backward()

        compiled_foo = torch.compile(foo, fullgraph=True, dynamic=True)
        out_compiled = compiled_foo(arg0, arg1, arg2, arg3, arg4, sentinel)
        out_compiled.sum().backward()

    # ===== NameError Issues =====

    @pytest.mark.xfail(reason="Issue #163674: NameError zuf0 is not defined")
    def test_fuzzer_issue_163674_nameerror_zuf0(self):
        """Fuzzer issue #163674: NameError zuf0 is not defined."""
        def foo(arg0, arg1, arg2):
            t0 = arg0  # size=(79488, 1, 3, 1), stride=(3, 3, 1, 1), dtype=float16, device=cuda
            t1 = t0.clone(); t1.zero_()  # size=(79488, 1, 3, 1), stride=(3, 3, 1, 1), dtype=float16, device=cuda
            t2 = arg1  # size=(79488, 1, 3, 1), stride=(3, 3, 1, 1), dtype=float32, device=cuda
            t3 = arg2  # size=(), stride=(), dtype=float32, device=cuda
            t4 = t2.clone(); t4.fill_(t3.item())  # size=(79488, 1, 3, 1), stride=(3, 3, 1, 1), dtype=float32, device=cuda
            t5 = torch.pow(t1, t4)  # size=(79488, 1, 3, 1), stride=(3, 3, 1, 1), dtype=float32, device=cuda
            t6 = t5.reshape((96, 69, 36))  # size=(96, 69, 36), stride=(2484, 36, 1), dtype=float32, device=cuda
            return t6

        arg0 = torch.rand([79488, 1, 3, 1], dtype=torch.float16, device='cuda', requires_grad=True)
        arg1 = torch.rand([79488, 1, 3, 1], dtype=torch.float32, device='cuda', requires_grad=True)
        arg2 = torch.rand([], dtype=torch.float32, device='cuda', requires_grad=True)

        out_eager = foo(arg0, arg1, arg2)
        out_eager.sum().backward()

        compiled_foo = torch.compile(foo, fullgraph=True, dynamic=True)
        out_compiled = compiled_foo(arg0, arg1, arg2)
        out_compiled.sum().backward()

    @pytest.mark.xfail(reason="Issue #163420: NameError zuf0 is not defined with fill_diagonal_")
    def test_fuzzer_issue_163420_nameerror_fill_diagonal(self):
        """Fuzzer issue #163420: NameError zuf0 is not defined with fill_diagonal_."""
        def foo(arg0, arg1):
            t0 = arg0  # size=(1, 1), stride=(1, 1), dtype=float32, device=cuda
            t1 = arg1  # size=(), stride=(), dtype=float32, device=cuda
            t2 = t0.clone(); t2.fill_diagonal_(t1.item())  # size=(1, 1), stride=(1, 1), dtype=float32, device=cuda
            return t2

        arg0 = torch.empty([1, 1], dtype=torch.float32, device='cuda', requires_grad=True)
        arg1 = torch.empty([], dtype=torch.float32, device='cuda', requires_grad=True)

        out_eager = foo(arg0, arg1)
        out_eager.sum().backward()

        compiled_foo = torch.compile(foo, fullgraph=True, dynamic=True)
        out_compiled = compiled_foo(arg0, arg1)
        out_compiled.sum().backward()

    # ===== Other Error Types =====

    @pytest.mark.xfail(reason="Issue #163971: list index out of range")
    def test_fuzzer_issue_163971_list_index_out_of_range(self):
        """Fuzzer issue #163971: Error list index out of range."""
        def foo(arg0):
            t0 = arg0  # size=(), stride=(), dtype=bfloat16, device=cuda
            t1 = torch.softmax(t0, dim=0)  # size=(), stride=(), dtype=bfloat16, device=cuda
            t2 = torch.nn.functional.gelu(t1)  # size=(), stride=(), dtype=bfloat16, device=cuda
            t3 = torch.softmax(t2, dim=0)  # size=(), stride=(), dtype=bfloat16, device=cuda
            return t3

        arg0 = torch.rand([], dtype=torch.bfloat16, device='cuda', requires_grad=True)

        out_eager = foo(arg0)
        out_eager.sum().backward()

        compiled_foo = torch.compile(foo, fullgraph=True, dynamic=True)
        out_compiled = compiled_foo(arg0)
        out_compiled.sum().backward()

    @pytest.mark.xfail(reason="Issue #163894: DDE _get_symint_hints(ph_arg.stride()) != real_stride")
    def test_fuzzer_issue_163894_dde_symint_hints(self):
        """Fuzzer issue #163894: DDE _get_symint_hints(ph_arg.stride()) != real_stride."""
        def fuzzed_program(arg_0, sentinel):
            var_node_1 = arg_0  # size=(1, 2), stride=(2, 1), dtype=int64, device=cuda
            var_node_5 = torch.full((1, 2), -66, dtype=torch.int32)  # size=(1, 2), stride=(2, 1), dtype=int32, device=cuda
            var_node_6 = torch.full((1, 2), 77, dtype=torch.int64)  # size=(1, 2), stride=(2, 1), dtype=int64, device=cuda
            var_node_4 = torch.ops.aten.add(var_node_5, var_node_6)  # size=(1, 2), stride=(2, 1), dtype=int32, device=cuda
            var_node_7 = torch.full((1, 2), -64, dtype=torch.int32)  # size=(1, 2), stride=(2, 1), dtype=int32, device=cuda
            var_node_3 = torch.ops.aten.mul(var_node_4, var_node_7)  # size=(1, 2), stride=(2, 1), dtype=int32, device=cuda
            var_node_9 = torch.full((3, 4), False, dtype=torch.bool)  # size=(3, 4), stride=(4, 1), dtype=bool, device=cuda
            var_node_8 = torch.nonzero(var_node_9)  # size=(1, 2), stride=(2, 1), dtype=int64, device=cuda
            var_node_2 = torch.ops.aten.add(var_node_3, var_node_8)  # size=(1, 2), stride=(2, 1), dtype=int32, device=cuda
            var_node_0 = torch.ops.aten.div(var_node_1, var_node_2)  # size=(1, 2), stride=(2, 1), dtype=int64, device=cuda
            result = var_node_0 * sentinel
            if result.is_complex():
                result = result.real
            return result

        torch.manual_seed(9)
        sentinel = torch.tensor(1.0, requires_grad=True)
        arg_0 = torch.randint(0, 3, (1, 2), dtype=torch.int64)

        result_original = fuzzed_program(arg_0, sentinel)

        compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
        result_compiled = compiled_program(arg_0, sentinel)

    @pytest.mark.xfail(reason="Issue #163435: a var subtract by itself should equal 0")
    def test_fuzzer_issue_163435_var_subtract_itself(self):
        """Fuzzer issue #163435: a var subtract by itself should equal 0?"""
        def foo(arg0, arg1, arg2, arg3):
            t0 = arg0  # size=(), stride=(), dtype=float16, device=cuda
            t1 = torch.tanh(t0)  # size=(), stride=(), dtype=float16, device=cuda
            t2 = arg1  # size=(), stride=(), dtype=float16, device=cuda
            t3 = arg2  # size=(), stride=(), dtype=float16, device=cuda
            t4 = arg3  # size=(), stride=(), dtype=float16, device=cuda
            t5 = t2 + t0 + t3 + t0 + t4  # size=(), stride=(), dtype=float16, device=cuda
            t6 = t1 * t1 * t5  # size=(), stride=(), dtype=float16, device=cuda
            t7 = (t6) - t6  # size=(), stride=(), dtype=float16, device=cuda
            return t7

        arg0 = torch.rand([], dtype=torch.float16, device='cuda', requires_grad=True)
        arg1 = torch.rand([], dtype=torch.float16, device='cuda', requires_grad=True)
        arg2 = torch.rand([], dtype=torch.float16, device='cuda', requires_grad=True)
        arg3 = torch.rand([], dtype=torch.float16, device='cuda', requires_grad=True)

        out_eager = foo(arg0, arg1, arg2, arg3)
        out_eager.sum().backward()

        compiled_foo = torch.compile(foo, fullgraph=True, dynamic=True)
        out_compiled = compiled_foo(arg0, arg1, arg2, arg3)
        out_compiled.sum().backward()


if __name__ == "__main__":
    run_tests()