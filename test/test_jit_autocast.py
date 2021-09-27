import torch
#from torch.cuda.amp import autocast
from torch import autocast
from typing import Optional

import unittest
from test_jit import JitTestCase
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_utils import run_tests


class TestAutocast(JitTestCase):
    def setUp(self):
        # common input tensors
        self.a_fp16 = torch.rand((2, 2), dtype=torch.float16, device='cuda')
        self.b_fp16 = torch.rand((2, 2), dtype=torch.float16, device='cuda')
        self.c_fp16 = torch.rand((2, 2), dtype=torch.float16, device='cuda')
        self.d_fp16 = torch.rand((2, 2), dtype=torch.float16, device='cuda')
        self.a_fp32 = torch.rand((2, 2), dtype=torch.float32, device='cuda')
        self.b_fp32 = torch.rand((2, 2), dtype=torch.float32, device='cuda')
        self.c_fp32 = torch.rand((2, 2), dtype=torch.float32, device='cuda')
        self.d_fp32 = torch.rand((2, 2), dtype=torch.float32, device='cuda')
        self.old_value = torch._C._jit_set_autocast_mode(True)
        super().setUp()

    def tearDown(self):
        torch._C._jit_set_autocast_mode(self.old_value)
        super().tearDown()

    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_minimal(self):
        @torch.jit.script
        def fn(a, b):
            with autocast():
                return torch.mm(a, b)
        result = fn(self.a_fp32, self.b_fp32)
        self.assertEqual(result.dtype, torch.float16)

    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_minimal_cpu(self):
        @torch.jit.script
        def fn(a, b):
            with autocast():
                return torch.mm(a, b)
        result = fn(self.a_fp32.to('cpu'), self.b_fp32.to('cpu'))
        self.assertEqual(result.dtype, torch.float16)

    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_minimal_off(self):
        @torch.jit.script
        def fn(a, b):
            with autocast(enabled=False):
                return torch.mm(a, b)
        result = fn(self.a_fp32, self.b_fp32)
        self.assertEqual(result.dtype, torch.float32)

    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_runtime_autocast_state(self):
        @torch.jit.script
        def fn(a, b, use_amp: bool):
            with autocast(enabled=use_amp):
                return torch.mm(a, b)
        # runtime values for autocast enable argument are not supported
        with self.assertRaises(RuntimeError):
            fn(self.a_fp32, self.b_fp32, True)

    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_runtime_autocast_state_expr(self):
        @torch.jit.script
        def fn(a, b):
            with autocast(enabled=True if a[0][0] > 0.5 else False):
                return torch.mm(a, b)
        # runtime values for autocast enable argument are not supported
        with self.assertRaises(RuntimeError):
            fn(self.a_fp32, self.b_fp32)

    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_explicit_casts(self):
        @torch.jit.script
        def fn(a, b, c, d):
            with autocast():
                e = torch.mm(a.double(), b.double()).float()
                f = torch.mm(c, d).double()
            g = torch.mm(c.double(), f)
            return e, f, g
        e, f, g = fn(self.a_fp32, self.b_fp32, self.c_fp32, self.d_fp32)
        self.assertEqual(e.dtype, torch.float32)
        self.assertEqual(f.dtype, torch.float64)
        self.assertEqual(g.dtype, torch.float64)

    # multiple uses of the same input value
    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_duplicate_inputs(self):
        @torch.jit.script
        def fn(a, b):
            with autocast():
                e = torch.mm(a, a)
                f = torch.mm(e, e)
            return e, f
        e, f = fn(self.a_fp32, self.b_fp32)
        self.assertEqual(e.dtype, torch.float16)
        self.assertEqual(f.dtype, torch.float16)

    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_fp32_policy(self):
        @torch.jit.script
        def fn(a):
            with autocast(enabled=True):
                return torch.log(a)
        result = fn(self.a_fp16)
        self.assertEqual(result.dtype, torch.float32)

    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_fp32_policy_with_fp64(self):
        @torch.jit.script
        def fn(a):
            with autocast(enabled=True):
                return torch.log(a)
        # fp32 policy should not narrow fp64 to fp32!
        result = fn(self.a_fp32.double())
        self.assertEqual(result.dtype, torch.float64)

    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_promote_policy(self):
        @torch.jit.script
        def fn(a, b, c, d):
            with autocast():
                e = torch.mm(a, b)
                f = torch.addcmul(e, c, d, value=0.1)
            return e, f
        e, f = fn(self.a_fp32, self.b_fp32, self.c_fp32, self.d_fp32)
        self.assertEqual(e.dtype, torch.float16)
        self.assertEqual(f.dtype, torch.float32)

    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_promote_policy_fp64(self):
        @torch.jit.script
        def fn(a, b):
            with autocast(enabled=True):
                return torch.addcmul(a, a, b, value=0.1)
        result = fn(self.a_fp32.double(), self.b_fp32.double())
        self.assertEqual(result.dtype, torch.float64)

    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_fp32_set_opt_dtype_policy(self):
        @torch.jit.script
        def fn(a, b, c, d, dtype: Optional[int]):
            with autocast(enabled=True):
                x = torch.softmax(a, 0)
                y = torch.softmax(b, 0, None)
                z = torch.softmax(c, 0, torch.float64)
                w = torch.softmax(d, 0, dtype)
            return x, y, z, w
        x, y, z, w = fn(self.a_fp16, self.b_fp16, self.c_fp16, self.d_fp16, None)
        self.assertEqual(x.dtype, torch.float32)
        self.assertEqual(y.dtype, torch.float32)
        self.assertEqual(z.dtype, torch.float64)
        self.assertEqual(w.dtype, torch.float16)

    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_fp32_set_opt_dtype_policy_fp64(self):
        @torch.jit.script
        def fn(a, b, c, d, dtype: Optional[int]):
            with autocast(enabled=True):
                x = torch.softmax(a, 0)
                y = torch.softmax(b, 0, None)
                z = torch.softmax(c, 0, torch.float64)
                w = torch.softmax(d, 0, dtype)
            return x, y, z, w
        x, y, z, w = fn(self.a_fp32.double(), self.b_fp32.double(), self.c_fp32.double(), self.d_fp32.double(), None)
        self.assertEqual(x.dtype, torch.float64)
        self.assertEqual(y.dtype, torch.float64)
        self.assertEqual(z.dtype, torch.float64)
        self.assertEqual(w.dtype, torch.float64)

    @unittest.skipIf(True, "broken")
    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_control_flow(self):
        @torch.jit.script
        def fn(a, b, c, d):
            with autocast():
                if a[0][0] > 0.5:
                    e = torch.mm(a, b)
                    x = 1
                else:
                    e = torch.mm(c, d)
                    x = 2
                f = torch.mm(d, e) * x
            return e, f
        e, f = fn(self.a_fp32, self.b_fp32, self.c_fp32, self.d_fp32)
        self.assertEqual(e.dtype, torch.float16)
        self.assertEqual(f.dtype, torch.float16)

    # this works find in regular Python, but it creates a delicate
    # situation in TorchScript where the types are not consistent across
    # the then/else branches
    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_divergent_types(self):
        @torch.jit.script
        def fn(a, b, c, d):
            with autocast():
                if a[0][0] > 0.5:
                    e = torch.mm(a, b)
                    f = torch.mm(a, b).float()
                else:
                    e = torch.mm(c, d).float()
                    f = torch.mm(a, b)
            return torch.mm(e.float(), f.float())
        result = fn(self.a_fp32, self.b_fp32, self.c_fp32, self.d_fp32)
        self.assertEqual(result.dtype, torch.float32)

    # another, more complex case of divergent types
    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_divergent_autocast(self):
        @torch.jit.script
        def fn(a, b, c, d):
            autocast_on = autocast(enabled=True)
            autocast_off = autocast(enabled=False)
            if a[0][0] > 0.5:
                with autocast_on:
                    e = torch.mm(a, b)
            else:
                with autocast_off:
                    e = torch.mm(c, d)
            return torch.mm(e, e)
        fn(self.a_fp32, self.b_fp32, self.c_fp32, self.d_fp32)

    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_conditional_autocast(self):
        @torch.jit.script
        def fn(a, b):
            autocast_on = autocast(enabled=True)
            autocast_off = autocast(enabled=False)
            with autocast_on if a[0][0] > 0.5 else autocast_off:
                return torch.mm(a, b)
        # conditional autocast expressions are not supported
        with self.assertRaises(RuntimeError):
            fn(self.a_fp32, self.b_fp32)

    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_nested_autocast(self):
        @torch.jit.script
        def fn(a, b, c, d):
            with autocast(enabled=False):
                e = torch.mm(a, b)
                with autocast(enabled=True):
                    f = torch.mm(e, c)
                    with autocast(enabled=False):
                        g = torch.mm(e, d)
            return e, f, g
        e, f, g = fn(self.a_fp32, self.b_fp32, self.c_fp32, self.d_fp32)
        self.assertEqual(e.dtype, torch.float32)
        self.assertEqual(f.dtype, torch.float16)
        self.assertEqual(g.dtype, torch.float32)

    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_implicitly_nested_autocast(self):
        @torch.jit.script
        def fn(a, b):
            with autocast(enabled=False), autocast(enabled=True):
                return torch.mm(a, b)
        result = fn(self.a_fp32, self.b_fp32)
        self.assertEqual(result.dtype, torch.float16)

    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_reused_autocast(self):
        @torch.jit.script
        def fn(a, b, c, d):
            autocast_instance = autocast(enabled=True)
            with autocast_instance:
                e = torch.mm(a, b)
                with autocast_instance:
                    e = torch.mm(c, d)
                    f = torch.mm(d, e)
            g = torch.mm(e, f)
            return e, f, g
        e, f, g = fn(self.a_fp32, self.b_fp32, self.c_fp32, self.d_fp32)
        self.assertEqual(e.dtype, torch.float16)
        self.assertEqual(f.dtype, torch.float16)
        self.assertEqual(g.dtype, torch.float16)

    # TODO: fix and enable this test?
    #   (we could technically fix this, but is it really worth it?)
    @unittest.skipIf(True, "unsuported autocast syntax")
    def test_reused_autocast_expr(self):
        @torch.jit.script
        def fn(a, b, c, d):
            with autocast(enabled=True) as autocast_instance:
                e = torch.mm(a, b)
                with autocast_instance:
                    e = torch.mm(c, d)
                    f = torch.mm(d, e)
            g = torch.mm(e, f)
            return e, f, g
        e, f, g = fn(self.a_fp32, self.b_fp32, self.c_fp32, self.d_fp32)
        self.assertEqual(e.dtype, torch.float16)
        self.assertEqual(f.dtype, torch.float16)
        self.assertEqual(g.dtype, torch.float16)

    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_callees(self):
        def helper(a, b):
            return torch.mm(a, b)

        @torch.jit.script
        def fn(a, b):
            with autocast(enabled=True):
                tmp = helper(a, b)
                tmp = helper(tmp, tmp)
                tmp = helper(tmp, tmp)
                tmp = helper(tmp, tmp)
                return helper(tmp, b)

        result = fn(self.a_fp32, self.b_fp32)
        self.assertEqual(result.dtype, torch.float16)

    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_callees_with_autocast_on(self):
        def helper(a, b):
            with autocast(enabled=True):
                return torch.mm(a, b)

        @torch.jit.script
        def fn(a, b):
            with autocast(enabled=False):
                return helper(a, b)

        result = fn(self.a_fp32, self.b_fp32)
        self.assertEqual(result.dtype, torch.float16)

    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_callees_with_autocast_off(self):
        def helper(a, b):
            with autocast(enabled=False):
                return torch.mm(a, b)

        @torch.jit.script
        def fn(a, b):
            with autocast(enabled=True):
                return helper(a, b)

        result = fn(self.a_fp32, self.b_fp32)
        self.assertEqual(result.dtype, torch.float32)

    # scripting inside eager autocast
    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_eager_and_script(self):
        @torch.jit.script
        def fn(a, b):
            return torch.mm(a, b)
        for i in range(8):
            use_autocast = (i % 2 == 0)
            expected_dtype = torch.float16 if use_autocast else torch.float32
            with autocast(enabled=use_autocast):
                result = fn(self.a_fp32, self.b_fp32)
            self.assertEqual(result.dtype, expected_dtype)

    # traced inside scripting
    @unittest.skipIf(True, "broken")
    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_script_and_tracing(self):
        def helper(a, b):
            return torch.mm(a, b) * 2.0

        traced = torch.jit.trace(helper, (self.a_fp32, self.a_fp32))

        @torch.jit.script
        def fn(a, b):
            with autocast(enabled=True):
                return traced(a, b)

        result = fn(self.a_fp32, self.b_fp32)
        self.assertEqual(result.dtype, torch.float16)

    # traced with autocast inside scripting
    @unittest.skipIf(True, "autocast(False) is ignored inside traced functions")
    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_script_and_tracing_with_autocast(self):
        def helper(a, b):
            with autocast(enabled=False):
                return torch.mm(a, b) * 2.0

        traced = torch.jit.trace(helper, (self.a_fp32, self.a_fp32))

        @torch.jit.script
        def fn(a, b):
            with autocast(enabled=True):
                return traced(a, b)

        result = fn(self.a_fp32, self.b_fp32)
        self.assertEqual(result.dtype, torch.float32)

    # scripted called from traced
    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_tracing_and_script(self):
        @torch.jit.script
        def fn(a, b):
            with autocast():
                return torch.mm(a, b)

        def traced(a, b):
            return fn(a, b)

        traced = torch.jit.trace(traced, (self.a_fp32, self.b_fp32))
        result = traced(self.a_fp32, self.b_fp32)
        self.assertEqual(result.dtype, torch.float16)

    # scripted called from traced with autocast
    @unittest.skipIf(True, "scripted called from traced TorchScript is not yet working")
    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_tracing_with_autocast_and_script(self):
        @torch.jit.script
        def fn(a, b):
            return torch.mm(a, b)

        def traced(a, b):
            with autocast(enabled=True):
                return fn(a, b)

        traced = torch.jit.trace(traced, (self.a_fp32, self.b_fp32))
        result = traced(self.a_fp32, self.b_fp32)
        self.assertEqual(result.dtype, torch.float16)

    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_script_module(self):
        class TestModule(torch.nn.Module):
            def __init__(self, N, M):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.rand((N, M), dtype=torch.float32))
                self.linear = torch.nn.Linear(N, M).float()

            def forward(self, input):
                with autocast(enabled=True):
                    output = self.weight.mv(input)
                    output = self.linear(output)
                    return output

        scripted_module = torch.jit.script(TestModule(2, 3)).cuda()
        input = torch.rand(3, dtype=torch.float32, device='cuda')
        result = scripted_module(input)
        self.assertEqual(result.dtype, torch.float16)

    @unittest.skipIf(True, "autocast decorators not supported")
    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_autocast_decorator(self):
        @torch.jit.script
        @autocast(enabled=True)
        def fn(a, b):
            return torch.mm(a, b)
        result = fn(self.a_fp32, self.b_fp32)
        self.assertEqual(result.dtype, torch.float16)

    # this is equivalent to running scripted functions inside autocast)
    # (see also test_eager_and_script)
    @unittest.skipIf(True, "script inside autocast not supported")
    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_autocast_decorator_outside_jit(self):
        @autocast(enabled=True)
        @torch.jit.script
        def fn(a, b):
            return torch.mm(a, b)
        result = fn(self.a_fp32, self.b_fp32)
        self.assertEqual(result.dtype, torch.float16)

    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_inplace(self):
        @torch.jit.script
        def fn(a, b, c):
            with autocast(enabled=True):
                x = torch.addmm(a, b, c)
                y = torch.addmm(a, b, c, out=a)
                z = a.addmm_(b, c)
                return x, y, z
        x, y, z = fn(self.a_fp32, self.b_fp32, self.c_fp32)
        self.assertEqual(x.dtype, torch.float16)
        self.assertEqual(y.dtype, torch.float32)
        self.assertEqual(z.dtype, torch.float32)


if __name__ == '__main__':
    run_tests()
