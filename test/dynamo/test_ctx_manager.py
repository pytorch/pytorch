# Owner(s): ["module: dynamo"]
import contextlib
import sys
import traceback
import unittest
from contextlib import contextmanager

import torch
import torch._dynamo.test_case
import torch._dynamo.testing
from torch._dynamo.testing import EagerAndRecordGraphs, normalize_gm, same
from torch._dynamo.utils import counters
from torch.nn import functional as F
from torch.testing._internal.common_cuda import PLATFORM_SUPPORTS_FLASH_ATTENTION
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    TEST_WITH_ROCM,
)


def _contextmanager(func):
    # Copied from CPython! Behave exactly as contextlib.contextmanager but do not
    # wraps func. It is used internally in a few tests to check if the context
    # manager support in dynamo is correct. We can't use the original implementation
    # as it can lead to graph breaks in some unrelated cases because it wraps(func)
    from contextlib import _GeneratorContextManager

    # @wraps(func)
    def helper(*args, **kwds):
        return _GeneratorContextManager(func, args, kwds)

    return helper


z_glb = 0
k_glb = 0


class CustomizedCtxManager:
    def __init__(self, mode):
        self.prev = torch.is_grad_enabled()
        self.mode = mode

    def __enter__(self):
        torch._C._set_grad_enabled(self.mode)

    def __exit__(self, exc_type, exc_value, traceback):
        torch._C._set_grad_enabled(self.prev)


@contextlib.contextmanager
def customized_ctx_manager(mode):
    prev = torch.is_grad_enabled()
    try:
        yield torch._C._set_grad_enabled(mode)
    finally:
        torch._C._set_grad_enabled(prev)


class CustomizedCtxManagerWithGraphBreak(CustomizedCtxManager):
    def __enter__(self):
        torch._dynamo.graph_break()
        super().__enter__()


@contextlib.contextmanager
def customized_ctx_manager_with_graph_break(mode):
    prev = torch.is_grad_enabled()
    try:
        torch._dynamo.graph_break()
        yield torch._C._set_grad_enabled(mode)
    finally:
        torch._C._set_grad_enabled(prev)


class CtxManagerTests(torch._dynamo.test_case.TestCase):
    def test_no_grad(self):
        def fn1(a, b):
            x = a + 1
            # redundant no_grad should get ignored
            with torch.no_grad():
                x = x + b
            x = x + 2
            return x

        def fn2(a, b):
            x = a + 1
            with torch.set_grad_enabled(False):
                x = x + b
            x = x + 2
            return x

        def fn3(a, b):
            x = a + 1
            with torch.enable_grad():
                x = x + b
            x = x + 2
            return x

        def fn4(a, b):
            x = a + 1
            with torch.set_grad_enabled(True):
                if torch.is_grad_enabled():
                    x = x + b
            x = x + 2
            return x

        with torch.no_grad():
            torch._dynamo.testing.standard_test(
                self, fn=fn1, nargs=2, expected_ops=3
            )  # coalesced noop
            torch._dynamo.testing.standard_test(
                self, fn=fn2, nargs=2, expected_ops=3
            )  # coalesced noop
            torch._dynamo.testing.standard_test(self, fn=fn3, nargs=2, expected_ops=5)
            torch._dynamo.testing.standard_test(self, fn=fn4, nargs=2, expected_ops=5)
        with torch.enable_grad():
            torch._dynamo.testing.standard_test(self, fn=fn1, nargs=2, expected_ops=5)
            torch._dynamo.testing.standard_test(self, fn=fn2, nargs=2, expected_ops=5)
            torch._dynamo.testing.standard_test(
                self, fn=fn3, nargs=2, expected_ops=3
            )  # coalesced noop
            torch._dynamo.testing.standard_test(
                self, fn=fn4, nargs=2, expected_ops=3
            )  # coalesced noop

    def test_grad_mode_guard(self):
        def fn(a, b):
            prev_grad = torch.is_grad_enabled()
            torch.set_grad_enabled(False)
            a = a + 1
            a.tolist()  # graph break
            ret = a + b
            torch.set_grad_enabled(prev_grad)
            return ret

        a = torch.randn([3, 4])
        b = torch.randn([3, 4])
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        for _ in range(10):
            opt_fn(a, b)
        self.assertEqual(cnts.frame_count, 2)

    def test_nested_grad_mode_graph_break(self):
        def fn(x):
            before = torch.is_grad_enabled()
            with torch.set_grad_enabled(False):
                torch._dynamo.graph_break()
                with torch.set_grad_enabled(True):
                    x = torch.mul(x, 5)
                    torch._dynamo.graph_break()
                    x = torch.sqrt(x)
                    assert torch.is_grad_enabled()
                assert not torch.is_grad_enabled()
            assert torch.is_grad_enabled() == before
            return x

        a = torch.randn([3, 4])
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts)(fn)

        for _ in range(10):
            opt_fn(a)
        self.assertEqual(cnts.frame_count, 2)

    def test_torch_profiler(self):
        # wrap torch.profiler.* as NullContextVariable and do nothing
        def fn(x):
            y = x**2
            with torch.profiler.profile():
                y = y + 2
                with torch.profiler.record_function("my_function"):
                    z = y**3
                    z.tolist()  # graph break
                    z = z + 1
            return z

        x = torch.randn((2, 2), requires_grad=True)
        ref = fn(x)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        res = opt_fn(x)
        self.assertTrue(same(ref, res))
        self.assertEqual(cnts.frame_count, 2)

    def test_autograd_profiler(self):
        # wrap torch.autograd.profiler.* as NullContextVariable and do nothing
        def fn(x):
            y = x**2
            with torch.autograd.profiler.profile():
                y = y + 2
                with torch.autograd.profiler.record_function("my_function"):
                    z = y**3
                    z.tolist()  # graph break
                    z = z + 1
            return z

        x = torch.randn((2, 2), requires_grad=True)
        ref = fn(x)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        res = opt_fn(x)
        self.assertTrue(same(ref, res))
        self.assertEqual(cnts.frame_count, 2)

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_cuda_stream_context_manager1(self):
        def fn(x):
            s = torch.cuda.Stream()
            x = torch.mul(x, 5)
            x = torch.add(x, 2)
            current_stream = torch.cuda.current_stream()
            s.wait_stream(current_stream)
            with torch.cuda.stream(s):
                x = torch.relu(x)
            current_stream.wait_stream(s)
            x = torch.add(x, 1)
            x = torch.cos(x)
            return x

        x = torch.randn((2, 2), device="cuda")
        ref = fn(x)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts, nopython=True)(fn)
        res = opt_fn(x)
        self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 12)

    @unittest.expectedFailure  # https://github.com/pytorch/pytorch/issues/118204
    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_cuda_stream_across_graph_break(self):
        def fn(x):
            s = torch.cuda.Stream()
            x = torch.mul(x, 5)
            x = torch.add(x, 2)

            print("foo")

            tcs = torch.cuda.stream(s)
            current_stream = torch.cuda.current_stream()
            s.wait_stream(current_stream)

            with tcs:
                x = torch.relu(x)

            current_stream.wait_stream(s)
            x = torch.add(x, 1)
            x = torch.cos(x)
            return x

        x = torch.randn((2, 2), device="cuda")
        ref = fn(x)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        res = opt_fn(x)
        self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(cnts.op_count, 9)

    @unittest.expectedFailure  # https://github.com/pytorch/pytorch/issues/118204
    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_cuda_stream_context_manager2(self):
        def fn(x, s):
            x = torch.mul(x, 5)
            x = torch.add(x, 2)

            current_stream = torch.cuda.current_stream()
            s.wait_stream(current_stream)

            with torch.cuda.stream(s):
                x = torch.relu(x)

            current_stream.wait_stream(s)
            with torch.cuda.stream(current_stream):
                x = torch.relu(x)

            s2 = torch.cuda.Stream()
            s2.wait_stream(current_stream)
            with torch.cuda.stream(s2):
                x = torch.relu(x)

            current_stream.wait_stream(s2)
            x = torch.add(x, 1)
            x = torch.cos(x)
            return x

        x = torch.randn((2, 2), device="cuda")
        s = torch.cuda.Stream()
        ref = fn(x, s)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts, nopython=True)(fn)
        res = opt_fn(x, s)
        self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 18)

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_cuda_stream_method(self):
        def fn(x):
            x = torch.mul(x, 1)
            x = torch.add(x, 2)

            new_stream = torch.cuda.Stream()
            cur_stream = torch.cuda.current_stream()
            new_stream.wait_stream(cur_stream)

            with torch.cuda.stream(new_stream):
                x = torch.sin(x)
                x = torch.add(x, 3)

            cur_stream.wait_stream(new_stream)

            x = torch.add(x, 4)
            is_idle = cur_stream.query()
            cur_stream.synchronize()

            with torch.cuda.stream(new_stream):
                x = torch.add(x, 5)
            new_stream.synchronize()

            is_equal = cur_stream == new_stream

            x = torch.relu(x)
            x = torch.cos(x)
            return x

        x = torch.randn((2, 2), device="cuda")
        ref = fn(x)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts, nopython=True)(fn)
        res = opt_fn(x)
        self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 21)

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_cuda_stream_compared_with_constant(self):
        def fn(x):
            x = torch.mul(x, 1)
            x = torch.add(x, 2)

            cur_stream = torch.cuda.current_stream()
            if cur_stream is not None:
                return x + 1
            return x - 1

        def fn2(x):
            x = torch.mul(x, 1)
            x = torch.add(x, 2)

            cur_stream = torch.cuda.current_stream()
            if cur_stream != "const_str":
                return x + 1
            return x - 1

        x = torch.randn((2, 2), device="cuda")
        ref = fn(x)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts, nopython=True)(fn)
        opt_fn2 = torch._dynamo.optimize(cnts, nopython=True)(fn2)
        res = opt_fn(x)
        res2 = opt_fn2(x)
        self.assertEqual(ref, res)
        self.assertEqual(ref, res2)

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_cuda_stream_compared_with_stream(self):
        def fn(x, s0, s1):
            if s0 == s1:
                return x + 1
            else:
                return x - 1

        s0 = torch.cuda.Stream()
        s1 = torch.cuda.Stream()
        x = torch.randn(2, 2)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts, nopython=True)(fn)

        ref0 = fn(x, s0, s1)
        res0 = opt_fn(x, s0, s1)
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(ref0, res0)

        ref1 = fn(x, s1, s1)
        res1 = opt_fn(x, s1, s1)
        # We have a re-compilation because of chaning inputs
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(ref1, res1)

        torch._dynamo.reset()
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts, nopython=True)(fn)

        ref1 = fn(x, s1, s1)
        res1 = opt_fn(x, s1, s1)
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(ref1, res1)

        ref0 = fn(x, s0, s1)
        res0 = opt_fn(x, s0, s1)
        # We have a re-compilation because of chaning inputs
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(ref0, res0)

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_cuda_event_reconstruct(self):
        def fn(x):
            e = torch.cuda.Event()
            x = torch.mul(x, 5)
            x = torch.add(x, 2)
            return x, e

        x = torch.randn((2, 2), device="cuda")
        ref = fn(x)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        res = opt_fn(x)
        self.assertEqual(ref[0], res[0])
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 3)

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_cuda_event_across_graph_break(self):
        def fn(x):
            e = torch.cuda.Event()
            e.record()
            x = torch.mul(x, 5)
            x = torch.add(x, 2)

            print("foo")

            torch.cuda.current_stream().wait_event(e)
            x = torch.add(x, 1)
            x = torch.cos(x)
            return x, e

        x = torch.randn((2, 2), device="cuda")
        ref = fn(x)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts)(fn)
        res = opt_fn(x)
        self.assertEqual(ref[0], res[0])
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(cnts.op_count, 9)

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_cuda_event_created_outside_of_graph(self):
        user_stream = torch.cuda.Stream()
        event = torch.cuda.Event()
        foo = torch.empty((2, 2), device="cuda")

        def func(foo):
            event.wait()
            return foo + 1, event

        x = torch.randn((1024, 1024), device="cuda")
        cnts = torch._dynamo.testing.CompileCounter()

        def run_iters(fn, compile=False):
            if compile:
                fn = torch._dynamo.optimize(cnts)(fn)
            for _ in range(10):
                with torch.cuda.stream(user_stream):
                    torch.mm(x, x, out=foo)
                    event.record()
                out = fn(foo)
            return out

        ref = run_iters(func, compile=False)
        res = run_iters(func, compile=True)
        self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 3)

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_cuda_event_method_create_stream_outside_of_compile(self):
        def fn(x, cur_stream, new_stream):
            x = torch.mul(x, 1)
            x = torch.add(x, 2)

            x = torch.add(x, 3)

            event = cur_stream.record_event()
            is_idle = event.query()

            new_stream.wait_event(event)
            with torch.cuda.stream(new_stream):
                x = torch.add(x, 4)

            new_event = torch.cuda.Event()
            new_event.record(new_stream)

            new_event.wait(cur_stream)
            x = torch.add(x, 5)

            # use new event to sync
            new_event.synchronize()

            x = torch.relu(x)
            x = torch.cos(x)
            return x

        x = torch.randn((2, 2), device="cuda")
        cur_stream = torch.cuda.current_stream()
        new_stream = torch.cuda.Stream()
        ref = fn(x, cur_stream, new_stream)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts, nopython=True)(fn)
        res = opt_fn(x, cur_stream, new_stream)
        self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 19)

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_cuda_event_method(self):
        def fn(x):
            x = torch.mul(x, 1)
            x = torch.add(x, 2)

            cur_stream = torch.cuda.current_stream()
            new_stream = torch.cuda.Stream()

            x = torch.add(x, 3)

            event = cur_stream.record_event()
            is_idle = event.query()

            new_stream.wait_event(event)
            with torch.cuda.stream(new_stream):
                x = torch.add(x, 4)

            new_event = torch.cuda.Event()
            new_event.record(new_stream)

            new_event.wait(cur_stream)
            x = torch.add(x, 5)

            # use new event to sync
            new_event.synchronize()

            x = torch.relu(x)
            x = torch.cos(x)
            return x

        x = torch.randn((2, 2), device="cuda")
        ref = fn(x)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts, nopython=True)(fn)
        res = opt_fn(x)
        self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 19)

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_cuda_device(self):
        def fn(x):
            with torch.cuda.device(x.device.index - 1):
                x = torch.sin(x + 1)
            return x

        x = torch.randn((2, 2), device="cuda")
        ref = fn(x)
        opt_fn = torch.compile(backend="eager", fullgraph=True)(fn)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    def test_autograd_profiler_enabled(self):
        def fn(x):
            if torch.autograd._profiler_enabled():
                return x + 1
            else:
                return x - 1

        x = torch.randn((2, 2), requires_grad=True)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts)(fn)

        if torch.autograd._profiler_enabled():
            torch.autograd._disable_profiler()
        assert not torch.autograd._profiler_enabled()
        ref = fn(x)
        res = opt_fn(x)
        self.assertTrue(same(ref, res))

        with torch.autograd.profiler.profile():
            assert torch.autograd._profiler_enabled()
            ref = fn(x)
            res = opt_fn(x)
            self.assertTrue(same(ref, res))

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_autocast(self):
        if not torch.cuda.is_bf16_supported():
            raise unittest.SkipTest("requires bf16")

        class MyModule(torch.nn.Module):
            def forward(self, x):
                a_float32 = torch.rand((8, 8), device="cuda")
                b_float32 = torch.rand((8, 8), device="cuda")
                d_float32 = torch.rand((8, 8), device="cuda")

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    e_float16 = torch.mm(a_float32, b_float32)
                    f_float16 = torch.mm(d_float32, e_float16)
                return f_float16

        module = MyModule()
        real = module(torch.tensor([0.5]))
        real_device = real.device
        real_dtype = real.dtype

        graph, guards = torch._dynamo.export(module)(torch.tensor([[0.0, 0], [0, 0]]))
        exported = graph(torch.tensor([0.5]))
        self.assertEqual(exported.device, real_device)
        self.assertEqual(exported.dtype, real_dtype)

        self.assertEqual(exported.device.type, "cuda")
        self.assertEqual(exported.device.index, 0)
        self.assertEqual(exported.dtype, torch.bfloat16)

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_cuda_amp_autocast(self):
        class MyModule(torch.nn.Module):
            def forward(self, x):
                a_float32 = torch.rand((8, 8), device="cuda")
                b_float32 = torch.rand((8, 8), device="cuda")

                with torch.autocast(device_type="cuda", dtype=torch.float64):
                    c_float64 = torch.mm(a_float32, b_float32)
                return c_float64

        module = MyModule()
        real = module(torch.tensor([0.5]))
        real_device = real.device
        real_dtype = real.dtype

        graph, _ = torch._dynamo.export(module)(torch.tensor([[0.0, 0], [0, 0]]))
        exported = graph(torch.tensor([0.5]))
        self.assertEqual(exported.device, real_device)
        self.assertEqual(exported.dtype, real_dtype)

        self.assertEqual(exported.device.type, "cuda")
        self.assertEqual(exported.device.index, 0)
        self.assertEqual(exported.dtype, torch.float64)

    def test_is_autocast_cpu_enabled(self):
        def fn(a_float32, b_float32):
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                c_float16 = torch.mm(a_float32, b_float32)
                if torch.is_autocast_cpu_enabled():
                    c_float16 = c_float16 + 1
            return c_float16

        a = torch.rand((8, 8))
        b = torch.rand((8, 8))
        ref = fn(a, b)
        opt_fn = torch._dynamo.optimize("eager", nopython=True)(fn)
        res = opt_fn(a, b)
        self.assertTrue(same(ref, res))

    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FLASH_ATTENTION or TEST_WITH_ROCM,
        "Can't run fused SDPA on this platform",
    )
    def test_autocast_sdpa(self):
        class MyModule(torch.nn.Module):
            def forward(self, query, key, value):
                with torch.autocast("cpu"):
                    with torch.autocast("cuda", dtype=torch.float32):
                        out = F.scaled_dot_product_attention(
                            query, key, value, None, 0.0, True
                        )
                return out

        dtype = torch.float32
        seq_len_q = 1
        seq_len_k = 1
        head_dim = 8
        query = torch.ones(
            1, 8, seq_len_q, head_dim, device="cuda", dtype=dtype, requires_grad=True
        )
        key = torch.ones(
            1, 8, seq_len_k, head_dim, device="cuda", dtype=dtype, requires_grad=True
        )
        value = torch.ones(
            1, 8, seq_len_k, head_dim, device="cuda", dtype=dtype, requires_grad=True
        )

        module = MyModule()
        real = module(query, key, value)
        real_device = real.device
        real_dtype = real.dtype

        opt_mod = torch._dynamo.optimize("inductor")(module)
        compiled = opt_mod(query, key, value)

        self.assertEqual(compiled.device, real_device)
        self.assertEqual(compiled.dtype, real_dtype)

        self.assertEqual(compiled.device.type, "cuda")
        self.assertEqual(compiled.device.index, 0)
        self.assertEqual(compiled.dtype, torch.float32)

    def test_autocast_cpu(self):
        class MyModule(torch.nn.Module):
            def forward(self, x):
                a_float32 = torch.rand((8, 8), device="cpu")
                b_float32 = torch.rand((8, 8), device="cpu")
                d_float32 = torch.rand((8, 8), device="cpu")

                with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                    e_float16 = torch.mm(a_float32, b_float32)
                    f_float16 = torch.mm(d_float32, e_float16)
                return f_float16

        module = MyModule()
        real = module(torch.tensor([0.5]))
        real_device = real.device
        real_dtype = real.dtype

        graph, guards = torch._dynamo.export(module)(torch.tensor([[0.0, 0], [0, 0]]))
        exported = graph(torch.tensor([0.5]))
        self.assertEqual(exported.device, real_device)
        self.assertEqual(exported.dtype, real_dtype)

        self.assertEqual(exported.device.type, "cpu")
        self.assertEqual(exported.dtype, torch.bfloat16)

    def test_autocast_cpu_graph_break(self):
        class MyModule(torch.nn.Module):
            def forward(self, x):
                a_float32 = torch.rand((8, 8), device="cpu")
                b_float32 = torch.rand((8, 8), device="cpu")
                torch._dynamo.graph_break()
                d_float32 = torch.rand((8, 8), device="cpu")

                with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                    e_float16 = torch.mm(a_float32, b_float32)
                    torch._dynamo.graph_break()
                    f_float16 = torch.mm(d_float32, e_float16)
                return f_float16

        module = MyModule()
        real = module(torch.tensor([0.5]))
        real_device = real.device
        real_dtype = real.dtype

        opt = torch._dynamo.optimize("eager")(module)
        res = opt(torch.tensor([0.5]))
        self.assertEqual(res.device, real_device)
        self.assertEqual(res.dtype, real_dtype)

        self.assertEqual(res.device.type, "cpu")
        self.assertEqual(res.dtype, torch.bfloat16)

    def test_autocast_cpu_graph_break_2(self):
        # Regression for: https://github.com/pytorch/pytorch/issues/93890
        def fn(x):
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                x = torch.mm(x, x)
                torch._dynamo.graph_break()
                x = torch.relu(x)
            return x

        x = torch.rand([4, 4])
        self.assertEqual(x.dtype, torch.float32)
        res = fn(x)
        opt_fn = torch._dynamo.optimize("eager")(fn)
        opt_res = opt_fn(x)
        self.assertTrue(torch.allclose(res, opt_res))
        self.assertEqual(res.dtype, torch.bfloat16)
        self.assertEqual(opt_res.dtype, torch.bfloat16)

    def test_autocast_cpu_graph_break_inner_fn(self):
        class MyModule(torch.nn.Module):
            @staticmethod
            def mm_breaks(x, y):
                torch._dynamo.graph_break()
                return torch.mm(x, y)

            def forward(self, x):
                a_float32 = torch.rand((8, 8), device="cpu")
                b_float32 = torch.rand((8, 8), device="cpu")

                with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                    torch._dynamo.graph_break()
                    with torch.autocast(
                        device_type="cpu", dtype=torch.bfloat16, enabled=False
                    ):
                        torch._dynamo.graph_break()
                        g_float32 = torch.mm(a_float32, b_float32)
                        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                            # Check that nested with non-inlineable function with graph break
                            torch._dynamo.graph_break()
                            f_float16_1 = self.mm_breaks(a_float32, b_float32)
                    # We remember to exit the inner autocast correctly to outer
                    # even after graph breaks
                    f_float16 = self.mm_breaks(a_float32, b_float32)
                    assert f_float16.dtype == f_float16_1.dtype
                return f_float16, g_float32

        module = MyModule()
        real_16, real_32 = module(torch.tensor([0.5]))
        real_device_16 = real_16.device
        real_dtype_16 = real_16.dtype
        real_device_32 = real_32.device
        real_dtype_32 = real_32.dtype

        graph = torch._dynamo.optimize("eager")(module)
        out_16, out_32 = graph(torch.tensor([0.5]))
        self.assertEqual(out_16.device, real_device_16)
        self.assertEqual(out_16.dtype, real_dtype_16)
        self.assertEqual(out_32.device, real_device_32)
        self.assertEqual(out_32.dtype, real_dtype_32)

        self.assertEqual(out_16.device.type, "cpu")
        self.assertEqual(out_16.dtype, torch.bfloat16)
        self.assertEqual(out_32.device.type, "cpu")
        self.assertEqual(out_32.dtype, torch.float32)

    def test_autocast_graph_break_method(self):
        class MyModule(torch.nn.Module):
            def __init__(self, bias):
                super().__init__()
                self.bias = bias

            def mm_not_break(self, x, y):
                return torch.mm(x, y) + self.bias

            def mm_breaks(self, x, y):
                torch._dynamo.graph_break()
                return torch.mm(x, y) + self.bias

            def forward(self, x):
                a_float32 = torch.rand((8, 8), device="cpu")
                b_float32 = torch.rand((8, 8), device="cpu")

                with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                    with torch.autocast(
                        device_type="cpu", dtype=torch.bfloat16, enabled=False
                    ):
                        g_float32 = torch.mm(a_float32, b_float32)
                    f_float16 = self.mm_breaks(a_float32, b_float32)

                    assert (
                        f_float16[0][0] == self.mm_not_break(a_float32, b_float32)[0][0]
                    )
                return f_float16, g_float32

        module = MyModule(bias=torch.rand((8, 8), device="cpu", dtype=torch.bfloat16))

        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            # Autocast doesn't work on addition, so we need the bias to be `bfloat16`
            res = torch.rand((8, 8), device="cpu", dtype=torch.float32) + torch.rand(
                (8, 8), device="cpu", dtype=torch.bfloat16
            )
            self.assertEqual(res.dtype, torch.float32)

        real_16, real_32 = module(torch.tensor([0.5]))
        real_device_16 = real_16.device
        real_dtype_16 = real_16.dtype
        real_device_32 = real_32.device
        real_dtype_32 = real_32.dtype

        graph = torch._dynamo.optimize("eager")(module)
        out_16, out_32 = graph(torch.tensor([0.5]))
        self.assertEqual(out_16.device, real_device_16)
        self.assertEqual(out_16.dtype, real_dtype_16)
        self.assertEqual(out_32.device, real_device_32)
        self.assertEqual(out_32.dtype, real_dtype_32)

        self.assertEqual(out_16.device.type, "cpu")
        self.assertEqual(out_16.dtype, torch.bfloat16)
        self.assertEqual(out_32.device.type, "cpu")
        self.assertEqual(out_32.dtype, torch.float32)

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_autocast_float64(self):
        class MyModule(torch.nn.Module):
            def forward(self, x):
                a_float32 = torch.rand((8, 8), device="cuda")
                b_float32 = torch.rand((8, 8), device="cuda")
                d_float32 = torch.rand((8, 8), device="cuda")

                with torch.autocast(device_type="cuda", dtype=torch.float64):
                    e_float64 = torch.mm(a_float32, b_float32)
                    f_float64 = torch.mm(d_float32, e_float64)
                return f_float64

        module = MyModule()
        real = module(torch.tensor([0.5]))
        real_device = real.device
        real_dtype = real.dtype

        graph, guards = torch._dynamo.export(module)(torch.tensor([[0.0, 0], [0, 0]]))
        exported = graph(torch.tensor([0.5]))
        self.assertEqual(exported.device, real_device)
        self.assertEqual(exported.dtype, real_dtype)

        self.assertEqual(exported.device.index, 0)
        self.assertEqual(exported.dtype, torch.float64)

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_autocast_device(self):
        class MyModule(torch.nn.Module):
            def forward(self, x):
                a_float32 = torch.rand((8, 8), device="cuda")
                b_float32 = torch.rand((8, 8), device="cuda")
                d_float32 = torch.rand((8, 8), device="cuda")

                with torch.autocast("cuda"):
                    e_float64 = torch.mm(a_float32, b_float32)
                    f_float64 = torch.mm(d_float32, e_float64)
                return f_float64

        module = MyModule()
        real = module(torch.tensor([0.5]))
        real_device = real.device
        real_dtype = real.dtype

        graph, guards = torch._dynamo.export(module)(torch.tensor([[0.0, 0], [0, 0]]))
        exported = graph(torch.tensor([0.5]))
        self.assertEqual(exported.device, real_device)
        self.assertEqual(exported.dtype, real_dtype)

        self.assertEqual(exported.device.index, 0)
        self.assertEqual(exported.dtype, torch.float16)

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_autocast_arguments_binding(self):
        def f1(x):
            with torch.autocast(device_type="cuda", enabled=False):
                x = torch.sin(x + 1)
            return x

        def f2(x):
            with torch.autocast(device_type="cpu", enabled=False):
                x = torch.cos(x + 1)
            return x

        x = torch.rand([2, 3])
        ref1 = f1(x)
        ref2 = f2(x)
        opt_f1 = torch.compile(backend="eager")(f1)
        opt_f2 = torch.compile(backend="eager")(f2)
        res1 = opt_f1(x)
        res2 = opt_f2(x)
        self.assertTrue(same(ref1, res1))
        self.assertTrue(same(ref2, res2))

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_autocast_decorator(self):
        def autocast_func(orig_func):
            @torch.amp.autocast(device_type="cuda", dtype=torch.float16)
            def new_fwd(*args, **kwargs):
                return orig_func(*args, **kwargs)

            return new_fwd

        def autocast_func_cuda(orig_func):
            @torch.autocast(device_type="cuda", dtype=torch.float16)
            def new_fwd(*args, **kwargs):
                return orig_func(*args, **kwargs)

            return new_fwd

        def autocast_func_cpu(orig_func):
            @torch.autocast(device_type="cpu", dtype=torch.float16)
            def new_fwd(*args, **kwargs):
                return orig_func(*args, **kwargs)

            return new_fwd

        def mm(a, b):
            return torch.mm(a, b)

        mm_float16 = autocast_func(mm)
        mm_float16_cuda = autocast_func_cuda(mm)
        mm_float16_cpu = autocast_func_cpu(mm)

        def fn(a, b):
            return mm_float16(a, b), mm_float16_cuda(a, b), mm_float16_cpu(a, b)

        a_float32 = torch.rand((8, 8), device="cuda")
        b_float32 = torch.rand((8, 8), device="cuda")

        ref = fn(a_float32, b_float32)
        opt_fn = torch.compile(backend="eager", fullgraph=True)(fn)
        res = opt_fn(a_float32, b_float32)
        self.assertTrue(same(ref, res))
        self.assertTrue(res[0].dtype == torch.float16)
        self.assertTrue(res[1].dtype == torch.float16)

    @parametrize(
        "Ctx",
        [CustomizedCtxManagerWithGraphBreak, customized_ctx_manager_with_graph_break],
        name_fn=lambda x: x.__name__,
    )
    def test_generic_ctx_manager_with_graph_break(self, Ctx):
        def fn(x):
            with Ctx(False):
                # body runs on eager
                y = x * 2
                z = y.sin() + 3
            return z

        x = torch.randn(2, 3)
        opt_fn = torch.compile(backend="eager", fullgraph=False)(fn)
        self.assertEqual(fn(x), opt_fn(x))

    def test_return_context_manager(self):
        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            cm = CustomizedCtxManager(False)
            with cm:
                pass
            return cm

        x = torch.randn(2, 3)
        cm = f(x)
        self.assertFalse(cm.mode)

    def test_return_context_manager_with_graph_break(self):
        @torch.compile(backend="eager", fullgraph=False)
        def f(x):
            cm = CustomizedCtxManager(False)
            torch._dynamo.graph_break()
            with cm:
                pass
            return cm

        x = torch.randn(2, 3)
        cm = f(x)
        self.assertFalse(cm.mode)

    @torch._dynamo.config.patch(enable_trace_contextlib_contextmanager=True)
    @parametrize(
        "Ctx",
        [CustomizedCtxManager, customized_ctx_manager],
        name_fn=lambda x: x.__name__,
    )
    def test_generic_context_manager(self, Ctx):
        def fn(x):
            with Ctx(True):
                x = x + 1
                if torch.is_grad_enabled():
                    x = x * 2
                x = torch.relu(x)
            return x - 1

        x = torch.rand(2, 3)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(backend=cnts, fullgraph=True)(fn)

        with torch.no_grad():
            ref = fn(x)
            res = opt_fn(x)
            self.assertTrue(same(ref, res))
            self.assertEqual(cnts.frame_count, 1)
            self.assertEqual(cnts.op_count, 6)

        with torch.enable_grad():
            ref = fn(x)
            res = opt_fn(x)
            self.assertTrue(same(ref, res))
            self.assertEqual(cnts.frame_count, 2)
            self.assertEqual(cnts.op_count, 12)

    @torch._dynamo.config.patch(enable_trace_contextlib_contextmanager=True)
    @parametrize(
        "Ctx",
        [CustomizedCtxManager, customized_ctx_manager],
        name_fn=lambda x: x.__name__,
    )
    def test_nested_generic_context_manager(self, Ctx):
        def fn(x):
            with Ctx(True):
                x = x + 1
                if torch.is_grad_enabled():
                    x = x * 2
                with Ctx(False):
                    if torch.is_grad_enabled():
                        x = x - 3
                    x = x * 1.5
                x = torch.relu(x)
            return x - 1

        x = torch.rand(2, 3)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(backend=cnts, fullgraph=True)(fn)

        with torch.no_grad():
            ref = fn(x)
            res = opt_fn(x)
            self.assertTrue(same(ref, res))
            self.assertEqual(cnts.frame_count, 1)
            self.assertEqual(cnts.op_count, 9)

        with torch.enable_grad():
            ref = fn(x)
            res = opt_fn(x)
            self.assertTrue(same(ref, res))
            self.assertEqual(cnts.frame_count, 2)
            self.assertEqual(cnts.op_count, 18)

    @torch._dynamo.config.patch(enable_trace_contextlib_contextmanager=True)
    @parametrize(
        "Ctx",
        [CustomizedCtxManager, customized_ctx_manager],
        name_fn=lambda x: x.__name__,
    )
    def test_generic_context_manager_with_graph_break(self, Ctx):
        def fn(x):
            with Ctx(True):
                x = x + 1
                if torch.is_grad_enabled():
                    x = x * 2
                torch._dynamo.graph_break()
                x = torch.relu(x)
            return x - 1

        x = torch.rand(2, 3)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(backend=cnts, fullgraph=False)(fn)

        with torch.no_grad():
            ref = fn(x)
            res = opt_fn(x)
            self.assertTrue(same(ref, res))
            if Ctx is CustomizedCtxManager:
                self.assertEqual(cnts.frame_count, 2)
                self.assertEqual(cnts.op_count, 2)

        with torch.enable_grad():
            ref = fn(x)
            res = opt_fn(x)
            self.assertTrue(same(ref, res))
            if Ctx is CustomizedCtxManager:
                self.assertEqual(cnts.frame_count, 4)
                self.assertEqual(cnts.op_count, 4)

    @torch._dynamo.config.patch(enable_trace_contextlib_contextmanager=True)
    @parametrize(
        "Ctx",
        [CustomizedCtxManager, customized_ctx_manager],
        name_fn=lambda x: x.__name__,
    )
    def test_nested_generic_context_manager_with_graph_break(self, Ctx):
        def fn(x):
            with Ctx(True):
                x = x + 1
                if torch.is_grad_enabled():
                    x = x * 2
                with Ctx(False):
                    if torch.is_grad_enabled():
                        x = x - 3
                    torch._dynamo.graph_break()
                    x = x * 1.5
                x = torch.relu(x)
            return x - 1

        x = torch.rand(2, 3)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(backend=cnts, fullgraph=False)(fn)

        with torch.no_grad():
            ref = fn(x)
            res = opt_fn(x)
            self.assertTrue(same(ref, res))
            if Ctx is CustomizedCtxManager:
                self.assertEqual(cnts.frame_count, 4)
                self.assertEqual(cnts.op_count, 4)

        torch._dynamo.reset()
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts, nopython=False)(fn)

        with torch.enable_grad():
            ref = fn(x)
            res = opt_fn(x)
            self.assertTrue(same(ref, res))
            if Ctx is CustomizedCtxManager:
                self.assertEqual(cnts.frame_count, 4)
                self.assertEqual(cnts.op_count, 4)

    def test_graph_break_inlining_grad(self):
        def gn(z):
            with torch.no_grad():
                torch._dynamo.graph_break()
                return torch.sin(z)

        def fn(x, y, z):
            a = torch.mm(x, y)
            z = gn(z)
            return a

        torch._dynamo.reset()
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(cnts, nopython=False)(fn)
        x = torch.randn(4, 4, requires_grad=True)
        y = torch.randn(4, 4, requires_grad=True)
        z = torch.randn(4)
        opt_fn(x, y, z).sum().backward()

        self.assertEqual(cnts.frame_count, 2)

    def _graph_break_inlining_autocast_test_helper(self, device):
        def gn(x, y):
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                z = torch.mm(x, y)
                torch._dynamo.graph_break()
                return torch.sin(z)

        def fn(x, y):
            z = torch.mm(x, y)
            z = z + gn(x, y)
            return z

        x = torch.rand(3, 3).to(device)
        y = torch.rand(3, 3).to(device)
        opt_fn = torch.compile(backend="eager")(fn)
        ref = fn(x, y)
        res = opt_fn(x, y)
        self.assertEqual(ref, res)

    def test_graph_break_inlining_autocast(self):
        for device in ["cuda", "cpu"]:
            if device == "cuda" and not (
                torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            ):
                continue
            self._graph_break_inlining_autocast_test_helper(device)

    def test_disable_saved_tensors_hooks(self):
        def fn(z):
            @torch.autograd.graph.disable_saved_tensors_hooks("This is not supported")
            def f(x, y):
                return x + y

            x, y = torch.ones(
                1,
            ), torch.zeros(
                1,
            )
            return f(x, y)

        eager = EagerAndRecordGraphs()
        torch.compile(fn, backend=eager, fullgraph=True)(torch.randn(()))

        graph = eager.graphs[0]
        actual = normalize_gm(graph.print_readable(False))

        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self):
        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable('This is not supported');  _saved_tensors_hooks_disable = None

        x: "f32[1]" = torch.ones(1)

        y: "f32[1]" = torch.zeros(1)

        add: "f32[1]" = x + y;  x = y = None

        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable();  _saved_tensors_hooks_enable = None
        return (add,)
""",  # NOQA: B950
        )

    def test_disable_saved_tensors_hooks_prev_disabled(self):
        def fn(z):
            @torch.autograd.graph.disable_saved_tensors_hooks("This is not supported")
            def f(x, y):
                return x + y

            x, y = torch.ones(
                1,
            ), torch.zeros(
                1,
            )
            return f(x, y)

        eager = EagerAndRecordGraphs()
        with torch.autograd.graph.disable_saved_tensors_hooks(
            "Previously disabled message"
        ):
            torch.compile(fn, backend=eager, fullgraph=True)(torch.randn(()))

        graph = eager.graphs[0]
        actual = normalize_gm(graph.print_readable(False))

        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self):
        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable('This is not supported');  _saved_tensors_hooks_disable = None

        x: "f32[1]" = torch.ones(1)

        y: "f32[1]" = torch.zeros(1)

        add: "f32[1]" = x + y;  x = y = None

        _saved_tensors_hooks_disable_1 = torch._C._autograd._saved_tensors_hooks_disable('Previously disabled message');  _saved_tensors_hooks_disable_1 = None
        return (add,)
""",  # NOQA: B950
        )

    def test_disable_saved_tensors_hooks_prev_disabled_nested(self):
        def fn(z):
            @torch.autograd.graph.disable_saved_tensors_hooks("This is not supported")
            def f(x, y):
                @torch.autograd.graph.disable_saved_tensors_hooks(
                    "This is not supported inner"
                )
                def inner_fn(x, y):
                    return x + y

                return inner_fn(x, y) + x

            x, y = torch.ones(
                1,
            ), torch.zeros(
                1,
            )
            return f(x, y)

        eager = EagerAndRecordGraphs()
        with torch.autograd.graph.disable_saved_tensors_hooks(
            "Previously disabled message"
        ):
            torch.compile(fn, backend=eager, fullgraph=True)(torch.randn(()))

        graph = eager.graphs[0]
        actual = normalize_gm(graph.print_readable(False))

        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self):
        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable('This is not supported');  _saved_tensors_hooks_disable = None

        x: "f32[1]" = torch.ones(1)

        y: "f32[1]" = torch.zeros(1)

        _saved_tensors_hooks_disable_1 = torch._C._autograd._saved_tensors_hooks_disable('This is not supported inner');  _saved_tensors_hooks_disable_1 = None

        add: "f32[1]" = x + y;  y = None

        _saved_tensors_hooks_disable_2 = torch._C._autograd._saved_tensors_hooks_disable('This is not supported');  _saved_tensors_hooks_disable_2 = None

        add_1: "f32[1]" = add + x;  add = x = None

        _saved_tensors_hooks_disable_3 = torch._C._autograd._saved_tensors_hooks_disable('Previously disabled message');  _saved_tensors_hooks_disable_3 = None
        return (add_1,)
""",  # NOQA: B950
        )

    def test_disable_saved_tensors_hooks_graph_break(self):
        def fn(x):
            with torch.autograd.graph.disable_saved_tensors_hooks(
                "This is not supported"
            ):
                y = x + 1
                torch._dynamo.graph_break()
                return y * 2

        eager = EagerAndRecordGraphs()
        torch.compile(fn, backend=eager, fullgraph=False)(torch.randn(()))

        def check_graph(actual, expected):
            self.assertExpectedInline(actual, expected)

        graph = eager.graphs[0]
        actual = normalize_gm(graph.print_readable(False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[]"):
        l_x_ = L_x_

        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable('This is not supported');  _saved_tensors_hooks_disable = None

        y: "f32[]" = l_x_ + 1;  l_x_ = None

        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable();  _saved_tensors_hooks_enable = None
        return (y,)
""",  # NOQA: B950
        )

        graph = eager.graphs[1]
        actual = normalize_gm(graph.print_readable(False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_y_: "f32[]"):
        l_y_ = L_y_

        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable('This is not supported');  _saved_tensors_hooks_disable = None

        mul: "f32[]" = l_y_ * 2;  l_y_ = None

        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable();  _saved_tensors_hooks_enable = None
        return (mul,)
""",  # NOQA: B950
        )

    def test_context_wrapping_grad_mode_decorator(self):
        ctx_wrappers = [(torch.enable_grad, True), (torch.no_grad, False)]
        for call in [True, False]:
            for i in range(2):
                torch._dynamo.reset()

                ctx_wrapper, mode = ctx_wrappers[i]
                ctx_wrapper_inverse, mode_inverse = ctx_wrappers[(i + 1) % 2]

                def fn(x):
                    def inner_func(x):
                        return x.sin()

                    with ctx_wrapper_inverse():
                        if call:
                            inner_func = ctx_wrapper()(inner_func)
                        else:
                            inner_func = ctx_wrapper(inner_func)

                        # Calling no_grad or enabled_grad should not mutate global state
                        assert torch.is_grad_enabled() == mode_inverse

                    with ctx_wrapper_inverse():
                        return inner_func(x)

                x = torch.zeros(10, requires_grad=True)
                opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
                self.assertEqual(fn(x), opt_fn(x))
                self.assertEqual(fn(x).requires_grad, opt_fn(x).requires_grad)

    def test_context_wrapping_grad_mode_nested_function_decorator(self):
        ctx_wrappers = [(torch.enable_grad, True), (torch.no_grad, False)]

        for call in [True, False]:
            for i in range(2):
                torch._dynamo.reset()

                ctx_wrapper, mode = ctx_wrappers[i]
                ctx_wrapper_inverse, mode_inverse = ctx_wrappers[(i + 1) % 2]

                def fn(x):
                    with ctx_wrapper_inverse():
                        if call:

                            @ctx_wrapper()
                            def inner_func(x):
                                return x.sin()

                        else:

                            @ctx_wrapper
                            def inner_func(x):
                                return x.sin()

                        # Calling no_grad or enabled_grad should not mutate global state
                        assert torch.is_grad_enabled() == mode_inverse

                    with ctx_wrapper_inverse():
                        return inner_func(x)

                x = torch.zeros(10, requires_grad=True)
                opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
                self.assertEqual(fn(x), opt_fn(x))
                self.assertEqual(fn(x).requires_grad, opt_fn(x).requires_grad)

    def test_context_wrapping_set_grad_enabled_nested_function(self):
        modes = [True, False]
        for decorator in [True, False]:
            for i in range(2):
                torch._dynamo.reset()

                mode = modes[i]
                mode_inverse = modes[(i + 1) % 2]

                def fn(x):
                    with torch.set_grad_enabled(mode_inverse):
                        if decorator:

                            @torch.set_grad_enabled(mode)
                            def inner_func(x):
                                return x.sin()

                        else:

                            def inner_func(x):
                                return x.sin()

                            inner_func = torch.set_grad_enabled(mode)(inner_func)

                        # Consuming set_grad_enabled by calling it on a function
                        # should not mutate global state
                        assert torch.is_grad_enabled() == mode_inverse

                    with torch.set_grad_enabled(mode_inverse):
                        return inner_func(x)

            x = torch.zeros(10, requires_grad=True)
            opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
            self.assertEqual(fn(x), opt_fn(x))
            self.assertEqual(fn(x).requires_grad, opt_fn(x).requires_grad)

    def test_inactive_context_graph_break_local(self):
        def fn(x):
            x = x + 1
            ctx = torch.set_grad_enabled(True)
            torch._dynamo.graph_break()
            with ctx:
                x = x + 1
            return x

        x = torch.zeros(10, requires_grad=False)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        self.assertEqual(fn(x), opt_fn(x))
        self.assertEqual(fn(x).requires_grad, opt_fn(x).requires_grad)
        self.assertEqual(cnts.frame_count, 2)

    def test_inactive_context_graph_break_local_nullctx(self):
        import contextlib

        # test with context manager that results in None target_values
        def fn(x):
            x = x + 1
            ctx = contextlib.nullcontext()
            torch._dynamo.graph_break()
            with ctx:
                x = x + 1
            return x

        x = torch.zeros(10, requires_grad=False)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        self.assertEqual(fn(x), opt_fn(x))
        self.assertEqual(fn(x).requires_grad, opt_fn(x).requires_grad)
        self.assertEqual(cnts.frame_count, 2)

    def test_inactive_context_graph_break_local_nullctx2(self):
        import contextlib

        # test with nullcontext where graph break happens
        # in an inlined function that returns something
        def gn():
            torch._dynamo.graph_break()
            return [0, 1, 2]

        def fn(x):
            x = x + 1
            ctx = contextlib.nullcontext()
            lst = gn()
            with ctx:
                x = x + lst[1]
            return x

        x = torch.zeros(10, requires_grad=False)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        self.assertEqual(fn(x), opt_fn(x))
        self.assertEqual(fn(x).requires_grad, opt_fn(x).requires_grad)
        self.assertEqual(cnts.frame_count, 2)

    def test_inactive_context_graph_break_stack(self):
        def gn(ctx):
            torch._dynamo.graph_break()
            return ctx

        def fn(x):
            x = x + 1
            ctx = gn(torch.set_grad_enabled(True))
            # we expect a graph break on next line as well
            with ctx:
                x = x + 1
            return x

        x = torch.zeros(10, requires_grad=False)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        self.assertEqual(fn(x), opt_fn(x))
        self.assertEqual(fn(x).requires_grad, opt_fn(x).requires_grad)

    def test_inactive_context_graph_break_stack2(self):
        def gn(x, ctx, y, z, dummy):
            with ctx:
                return x * y * z

        def fn(x):
            x = x + 1
            x = gn(x, torch.set_grad_enabled(True), 2, 3, torch._dynamo.graph_break())
            return x

        x = torch.zeros(10, requires_grad=False)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        self.assertEqual(fn(x), opt_fn(x))
        self.assertEqual(fn(x).requires_grad, opt_fn(x).requires_grad)
        self.assertEqual(cnts.frame_count, 2)

    def test_sdpa_kernel_ctx_manager1(self):
        modified_backend_state = [torch.nn.attention.SDPBackend.MATH]

        @torch._dynamo.allow_in_graph
        def check_backend_state_is_modified():
            self.assertEqual(
                torch.nn.attention._cur_sdpa_kernel_backends(), modified_backend_state
            )

        def f(x):
            with torch.nn.attention.sdpa_kernel(
                # pyre-fixme[16]: Module `torch.nn.attention` has no attribute `SDPBackend`.
                [torch.nn.attention.SDPBackend.MATH]
            ):
                output = torch.nn.functional.scaled_dot_product_attention(x, x, x).to(
                    torch.float32
                )
                check_backend_state_is_modified()

            return output

        opt_f = torch.compile(f, backend="eager", fullgraph=True)
        opt_f(torch.randn(2, 2, 2, 2).to(dtype=torch.float16))

    def test_sdpa_kernel_ctx_manager2(self):
        original_backend_state = set(torch.nn.attention._cur_sdpa_kernel_backends())
        modified_backend_state = [torch.nn.attention.SDPBackend.MATH]

        @torch._dynamo.allow_in_graph
        def check_backend_state_is_original():
            self.assertEqual(
                set(torch.nn.attention._cur_sdpa_kernel_backends()),
                original_backend_state,
            )

        @torch._dynamo.allow_in_graph
        def check_backend_state_is_modified():
            self.assertEqual(
                torch.nn.attention._cur_sdpa_kernel_backends(), modified_backend_state
            )

        def g(x):
            torch._dynamo.graph_break()
            output = torch.nn.functional.scaled_dot_product_attention(x, x, x).to(
                torch.float32
            )
            check_backend_state_is_modified()
            return output

        def f(x):
            check_backend_state_is_original()
            with torch.nn.attention.sdpa_kernel(
                # pyre-fixme[16]: Module `torch.nn.attention` has no attribute `SDPBackend`.
                [torch.nn.attention.SDPBackend.MATH]
            ):
                output1 = torch.nn.functional.scaled_dot_product_attention(x, x, x).to(
                    torch.float32
                )
                check_backend_state_is_modified()

                # graph break
                output2 = g(x)

                output3 = torch.nn.functional.scaled_dot_product_attention(x, x, x).to(
                    torch.float32
                )
                check_backend_state_is_modified()

            check_backend_state_is_original()

            return output1 + output2 + output3

        cnts = torch._dynamo.testing.CompileCounter()
        opt_f = torch.compile(f, backend=cnts)
        opt_f(torch.randn(2, 2, 2, 2).to(dtype=torch.float16))
        self.assertEqual(cnts.frame_count, 3)

    # test sdpa_kernel graph break with 2 arguments
    def test_sdpa_kernel_ctx_manager3(self):
        modified_backend_state = {
            torch.nn.attention.SDPBackend.MATH,
            torch.nn.attention.SDPBackend.FLASH_ATTENTION,
        }

        @torch._dynamo.allow_in_graph
        def check_backend_state_is_modified():
            self.assertEqual(
                set(torch.nn.attention._cur_sdpa_kernel_backends()),
                modified_backend_state,
            )

        def f(x):
            with torch.nn.attention.sdpa_kernel(
                # pyre-fixme[16]: Module `torch.nn.attention` has no attribute `SDPBackend`.
                [
                    torch.nn.attention.SDPBackend.MATH,
                    torch.nn.attention.SDPBackend.FLASH_ATTENTION,
                ]
            ):
                # FLASH_ATTENTION may not be supported, but we're not actually
                # doing any sdpa
                x = x + 1
                torch._dynamo.graph_break()
                check_backend_state_is_modified()
                x = x + 1

            return x

        opt_f = torch.compile(f, backend="eager")
        opt_f(torch.randn(2, 2))


class ContextlibContextManagerTests(torch._dynamo.test_case.TestCase):
    @torch._dynamo.config.patch(enable_trace_contextlib_contextmanager=True)
    def test_ctx_basic0(self):
        @contextlib.contextmanager
        def set_default_dtype(dtype):
            old_dtype = torch.get_default_dtype()
            try:
                torch.set_default_dtype(dtype)
                yield
            finally:
                torch.set_default_dtype(old_dtype)

        eager = EagerAndRecordGraphs()

        @torch.compile(backend=eager, fullgraph=True)
        def fn():
            with set_default_dtype(torch.float64):
                x = torch.tensor([3.0, 3.0 + 5.0j])
            return x

        y = fn()
        self.assertEqual(y.dtype, torch.complex128)
        graph = eager.graphs[0]
        actual = normalize_gm(graph.print_readable(False))

        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self):
        set_default_dtype = torch.set_default_dtype(torch.float64);  set_default_dtype = None

        x: "c128[2]" = torch.tensor([3.0, (3+5j)])

        set_default_dtype_1 = torch.set_default_dtype(torch.float32);  set_default_dtype_1 = None
        return (x,)
""",
        )

    @torch._dynamo.config.patch(enable_trace_contextlib_contextmanager=True)
    def test_ctx_basic1(self):
        @contextlib.contextmanager
        def compute_sin(x):
            try:
                yield x.sin()
            finally:
                pass

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            with compute_sin(x) as y:
                return y.cos()

        x = torch.tensor([1.0])
        y = fn(x)
        self.assertEqual(y, x.sin().cos())

    @torch._dynamo.config.patch(enable_trace_contextlib_contextmanager=True)
    def test_change_parent_nonlocal_0(self):
        # test if a nonlocal actually gets propagated
        z = 0
        k = 0

        def create_ctx():
            @contextmanager
            def ctx(x):
                nonlocal z
                nonlocal k
                try:
                    k = 100
                    yield x.sin()
                finally:
                    pass

            return ctx

        def run_ctx(ctx, x):
            nonlocal z
            with ctx(x) as y:
                z = k
                return y.cos()

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            ctx = create_ctx()
            return run_ctx(ctx, x)

        x = torch.tensor([1.0])
        y = fn(x)
        self.assertEqual(y, x.sin().cos())
        self.assertEqual(z, 100)
        self.assertEqual(k, 100)

    @torch._dynamo.config.patch(enable_trace_contextlib_contextmanager=True)
    def test_change_parent_nonlocal_1(self):
        # test if finally is executed and it is reading the correct variable
        z = 1
        k = 2

        def create_ctx():
            @contextmanager
            def ctx(x):
                nonlocal z
                nonlocal k
                try:
                    yield x.sin()
                finally:
                    k = z

            return ctx

        def run_ctx(ctx, x):
            nonlocal z
            z = 100
            with ctx(x) as y:
                return y.cos()

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            ctx = create_ctx()
            return run_ctx(ctx, x)

        x = torch.tensor([1.0])
        y = fn(x)
        self.assertEqual(y, x.sin().cos())
        self.assertEqual(z, 100)
        self.assertEqual(k, 100)

    @unittest.expectedFailure
    @torch._dynamo.config.patch(enable_trace_contextlib_contextmanager=True)
    def test_change_parent_global_0(self):
        # test if a global actually gets propagated
        global z_glb, k_glb
        z_glb, k_glb = 0, 0

        def create_ctx():
            @contextmanager
            def ctx(x):
                global z_glb, k_glb
                try:
                    k_glb = 100
                    yield x.sin()
                finally:
                    pass

            return ctx

        def run_ctx(ctx, x):
            global z_glb
            with ctx(x) as y:
                z_glb = k_glb
                return y.cos()

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            ctx = create_ctx()
            return run_ctx(ctx, x)

        x = torch.tensor([1.0])
        y = fn(x)
        self.assertEqual(y, x.sin().cos())
        self.assertEqual(z_glb, 100)
        self.assertEqual(k_glb, 100)

    @torch._dynamo.config.patch(enable_trace_contextlib_contextmanager=True)
    def test_change_parent_global_1(self):
        # test if finally is executed and it is reading the correct variable
        global z_glb, k_glb
        z_glb, k_glb = 0, 0

        def create_ctx():
            @contextmanager
            def ctx(x):
                global z_glb, k_glb
                try:
                    yield x.sin()
                finally:
                    k_glb = z_glb

            return ctx

        def run_ctx(ctx, x):
            global z_glb
            z_glb = 100
            with ctx(x) as y:
                return y.cos()

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            ctx = create_ctx()
            return run_ctx(ctx, x)

        x = torch.tensor([1.0])
        y = fn(x)
        self.assertEqual(y, x.sin().cos())
        self.assertEqual(z_glb, 100)
        self.assertEqual(k_glb, 100)

    @torch._dynamo.config.patch(enable_trace_contextlib_contextmanager=True)
    def test_change_parent_0(self):
        def create_ctx():
            @_contextmanager
            def ctx(x):
                try:
                    yield x.sin()
                finally:
                    pass

            return ctx

        def run_ctx(ctx, x):
            with ctx(x) as y:
                return y.cos()

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            ctx = create_ctx()
            return run_ctx(ctx, x)

        x = torch.tensor([1.0])
        y = fn(x)
        self.assertEqual(y, x.sin().cos())

    @torch._dynamo.config.patch(enable_trace_contextlib_contextmanager=True)
    def test_change_parent_1(self):
        def create_ctx(x):
            @_contextmanager
            def ctx():
                try:
                    yield x.sin()
                finally:
                    pass

            return ctx

        def run_ctx(ctx):
            with ctx() as y:
                return y.cos()

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            ctx = create_ctx(x)
            return run_ctx(ctx)

        x = torch.tensor([1.0])
        y = fn(x)
        self.assertEqual(y, x.sin().cos())

    @torch._dynamo.config.patch(enable_trace_contextlib_contextmanager=True)
    def test_graph_break_inside_ctx(self):
        @contextlib.contextmanager
        def whoo(x):
            y = x.tan()
            try:
                torch._dynamo.graph_break()
                yield y
            finally:
                pass

        def f(x):
            y = x.sin()
            with whoo(x) as z:
                y += z.neg()
            y += x.cos()

        x = torch.randn(2)
        expected = f(x)
        eager = EagerAndRecordGraphs()
        out = torch.compile(backend=eager, fullgraph=False)(f)(x)
        self.assertEqual(expected, out)
        # no graph will be generated as we will skip all frames due to the graph break
        self.assertEqual(len(eager.graphs), 0)

    @torch._dynamo.config.patch(enable_trace_contextlib_contextmanager=True)
    def test_graph_break_inside_ctx_1(self):
        @contextlib.contextmanager
        def whoo(x):
            y = x.tan()
            try:
                torch._dynamo.graph_break()
                yield y
            finally:
                pass

        def bar(x):
            with whoo(x) as z:
                return z.neg()

        def f(x):
            return x.sin() + bar(x) + x.cos()

        x = torch.randn(2)
        expected = f(x)
        eager = EagerAndRecordGraphs()
        out = torch.compile(backend=eager, fullgraph=False)(f)(x)
        self.assertEqual(expected, out)
        self.assertEqual(len(eager.graphs), 2)
        self.assertExpectedInline(
            normalize_gm(eager.graphs[0].print_readable(False)),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[2]"):
        l_x_ = L_x_

        sin: "f32[2]" = l_x_.sin();  l_x_ = None
        return (sin,)
""",
        )
        self.assertExpectedInline(
            normalize_gm(eager.graphs[1].print_readable(False)),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_stack0_: "f32[2]", L_stack1_: "f32[2]", L_x_: "f32[2]"):
        l_stack0_ = L_stack0_
        l_stack1_ = L_stack1_
        l_x_ = L_x_

        add: "f32[2]" = l_stack0_ + l_stack1_;  l_stack0_ = l_stack1_ = None
        cos: "f32[2]" = l_x_.cos();  l_x_ = None
        add_1: "f32[2]" = add + cos;  add = cos = None
        return (add_1,)
""",
        )

    @torch._dynamo.config.patch(enable_trace_contextlib_contextmanager=True)
    def test_graph_break_inside_ctx_2(self):
        @contextlib.contextmanager
        def whoo(x):
            try:
                torch._dynamo.graph_break()
                yield x.cos()
            finally:
                pass

        def g(x):
            return x.neg() + x.acos()

        def f(x):
            y = x.sin()
            with whoo(x) as z:
                y += g(z)
            y += y.tan()
            return y

        x = torch.randn(2)
        expected = f(x)
        eager = EagerAndRecordGraphs()
        out = torch.compile(backend=eager, fullgraph=False)(f)(x)
        self.assertEqual(expected, out)
        self.assertEqual(len(eager.graphs), 1)
        self.assertExpectedInline(
            normalize_gm(eager.graphs[0].print_readable(False)),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[2]"):
        l_x_ = L_x_

        neg: "f32[2]" = l_x_.neg()
        acos: "f32[2]" = l_x_.acos();  l_x_ = None
        add: "f32[2]" = neg + acos;  neg = acos = None
        return (add,)
""",
        )

    @torch._dynamo.config.patch(enable_trace_contextlib_contextmanager=False)
    def test_disable_trace_contextmanager(self):
        @contextlib.contextmanager
        def whoo(x):
            try:
                yield x.cos()
            finally:
                pass

        def g(x):
            return x.neg() + x.acos()

        def f(x):
            y = x.sin()
            with whoo(x) as z:
                y += g(z)
            y += y.tan()
            return y

        x = torch.randn(2)
        expected = f(x)
        eager = EagerAndRecordGraphs()
        out = torch.compile(backend=eager, fullgraph=False)(f)(x)
        self.assertEqual(expected, out)
        self.assertEqual(len(eager.graphs), 3)
        self.assertExpectedInline(
            normalize_gm(eager.graphs[0].print_readable(False)),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[2]"):
        l_x_ = L_x_

        y: "f32[2]" = l_x_.sin();  l_x_ = None
        return (y,)
""",
        )

        self.assertExpectedInline(
            normalize_gm(eager.graphs[1].print_readable(False)),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_args_0_: "f32[2]"):
        l_args_0_ = L_args_0_

        cos: "f32[2]" = l_args_0_.cos();  l_args_0_ = None
        return (cos,)
""",
        )

        self.assertExpectedInline(
            normalize_gm(eager.graphs[2].print_readable(False)),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[2]"):
        l_x_ = L_x_

        neg: "f32[2]" = l_x_.neg()
        acos: "f32[2]" = l_x_.acos();  l_x_ = None
        add: "f32[2]" = neg + acos;  neg = acos = None
        return (add,)
""",
        )

    @torch._dynamo.config.patch(enable_trace_contextlib_contextmanager=True)
    @parametrize("name", ("suppress", "stdout", "stderr"))
    def test_contextlib_suppress(self, name):
        counters.clear()
        eager = EagerAndRecordGraphs()

        def fn(t):
            y = t.sin()
            # ensure we graph break on the suppress call below
            if name == "suppress":
                ctx = contextlib.suppress(ValueError)
            elif name == "stdout":
                ctx = contextlib.redirect_stdout(sys.stderr)
            else:
                ctx = contextlib.redirect_stderr(sys.stdout)

            with ctx:
                y += t.cos()
            return y.tan()

        t = torch.randn(2)
        expected = fn(t)
        got = torch.compile(backend=eager, fullgraph=False)(fn)(t)
        self.assertEqual(expected, got)
        self.assertEqual(len(counters["graph_break"]), 1)
        name = f"redirect_{name}" if name in ("stdout", "stderr") else name
        self.assertEqual(
            {f"<class 'contextlib.{name}'> not supported": 1},
            dict(counters["graph_break"]),
        )

    @torch._dynamo.config.patch(enable_trace_contextlib_contextmanager=True)
    def test_contextlib_nullcontext(self):
        counters.clear()

        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            with contextlib.nullcontext():
                return t.sin()

        t = torch.randn(2)
        y = fn(t)
        # nullcontext is correctly handled in dynamo
        self.assertEqual(len(counters["graph_break"]), 0)
        self.assertEqual(y, t.sin())


class CPythonContextManagerTestCase(torch._dynamo.test_case.TestCase):
    # Tests taken from CPython source code in cpython/Lib/test/test_contextlib.py
    # https://github.com/python/cpython/blob/d48cc82ed25e26b02eb97c6263d95dcaa1e9111b/Lib/test/test_contextlib.py#L70

    @unittest.expectedFailure
    def test_contextmanager_plain(self):
        state = []

        @contextmanager
        def woohoo():
            state.append(1)
            yield 42
            state.append(999)

        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            y = t.sum()
            with woohoo() as x:
                assert state == [1]
                assert x == 42
                self.assertEqual(state, [1])
                self.assertEqual(x, 42)
                state.append(x)
                y += x
            return y

        t = torch.randn(2, 3)
        y = fn(t)
        self.assertEqual(state, [1, 42, 999])
        self.assertEqual(y, t.sum() + 42)

    @unittest.expectedFailure
    def test_contextmanager_finally(self):
        state = []

        @contextmanager
        def woohoo():
            state.append(1)
            try:
                yield 42
            finally:
                state.append(999)

        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            y = t.sum()
            with self.assertRaises(ZeroDivisionError):
                with woohoo() as x:
                    self.assertEqual(state, [1])
                    self.assertEqual(x, 42)
                    state.append(x)
                    raise ZeroDivisionError

        fn(torch.randn(2, 3))
        self.assertEqual(state, [1, 42, 999])

    @unittest.expectedFailure
    def test_contextmanager_traceback(self):
        @contextmanager
        def f():
            yield

        frames = []

        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            nonlocal frames
            y = t.sum()
            try:
                with f():
                    1 / 0
            except ZeroDivisionError as e:
                frames = traceback.extract_tb(e.__traceback__)

        fn(torch.randn(2, 3))
        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0].name, "test_contextmanager_traceback")
        self.assertEqual(frames[0].line, "1 / 0")

    @unittest.expectedFailure
    def test_contextmanager_traceback2(self):
        @contextmanager
        def f():
            yield

        # Repeat with RuntimeError (which goes through a different code path)
        class RuntimeErrorSubclass(RuntimeError):
            pass

        frames = []

        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            nonlocal frames
            y = t.sum()
            try:
                with f():
                    raise RuntimeErrorSubclass(42)
            except RuntimeErrorSubclass as e:
                frames = traceback.extract_tb(e.__traceback__)

        fn(torch.randn(2, 3))
        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0].name, "test_contextmanager_traceback")
        self.assertEqual(frames[0].line, "raise RuntimeErrorSubclass(42)")

    @unittest.expectedFailure
    def test_contextmanager_traceback3(self):
        @contextmanager
        def f():
            yield

        frames = []

        class StopIterationSubclass(StopIteration):
            pass

        for stop_exc in (
            StopIteration("spam"),
            StopIterationSubclass("spam"),
        ):
            with self.subTest(type=type(stop_exc)):

                @torch.compile(backend="eager", fullgraph=True)
                def fn(t):
                    nonlocal frames
                    y = t.sum()
                    try:
                        with f():
                            raise stop_exc
                    except type(stop_exc) as e:
                        self.assertIs(e, stop_exc)
                        frames = traceback.extract_tb(e.__traceback__)
                    else:
                        self.fail(f"{stop_exc} was suppressed")

                fn(torch.randn(2, 3))
                self.assertEqual(len(frames), 1)
                self.assertEqual(frames[0].name, "test_contextmanager_traceback")
                self.assertEqual(frames[0].line, "raise stop_exc")

    @unittest.expectedFailure
    def test_contextmanager_no_reraise(self):
        @contextmanager
        def whee():
            yield

        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            ctx = whee()
            ctx.__enter__()
            # Calling __exit__ should not result in an exception
            self.assertFalse(ctx.__exit__(TypeError, TypeError("foo"), None))
            return t.sum()

        fn(torch.randn(2, 3))

    @unittest.expectedFailure
    def test_contextmanager_trap_yield_after_throw(self):
        @contextmanager
        def whoo():
            try:
                yield
            except Exception:
                yield

        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            ctx = whoo()
            ctx.__enter__()
            with self.assertRaises(RuntimeError):
                ctx.__exit__(TypeError, TypeError("foo"), None)
            return t.sum()

        fn(torch.randn(2, 3))

    @unittest.expectedFailure
    def test_contextmanager_trap_no_yield(self):
        @contextmanager
        def whoo():
            if False:
                yield

        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            ctx = whoo()
            with self.assertRaises(RuntimeError):
                ctx.__enter__()
            return t.sum()

        fn(torch.randn(2, 3))

    @unittest.expectedFailure
    def test_contextmanager_trap_second_yield(self):
        @contextmanager
        def whoo():
            yield
            yield

        @torch.compile(backend="eager", fullgraph=True)
        def f(t):
            ctx = whoo()
            ctx.__enter__()
            with self.assertRaises(RuntimeError):
                ctx.__exit__(None, None, None)

        f(torch.randn(2))

    @unittest.expectedFailure
    def test_contextmanager_except(self):
        state = []

        @contextmanager
        def woohoo():
            state.append(1)
            try:
                yield 42
            except ZeroDivisionError as e:
                state.append(e.args[0])
                self.assertEqual(state, [1, 42, 999])

        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            with woohoo() as x:
                self.assertEqual(state, [1])
                self.assertEqual(x, 42)
                state.append(x)
                raise ZeroDivisionError(999)

        fn(torch.randn(2, 3))
        self.assertEqual(state, [1, 42, 999])

    @unittest.expectedFailure
    def test_contextmanager_do_not_unchain_non_stopiteration_exceptions(self):
        @contextmanager
        def test_issue29692():
            try:
                yield
            except Exception as exc:
                raise RuntimeError("issue29692:Chained") from exc

        @torch.compile(backend="eager", fullgraph=True)
        def f(t):
            try:
                with test_issue29692():
                    raise ZeroDivisionError
            except Exception as ex:
                self.assertIs(type(ex), RuntimeError)
                self.assertEqual(ex.args[0], "issue29692:Chained")
                self.assertIsInstance(ex.__cause__, ZeroDivisionError)

            try:
                with test_issue29692():
                    raise StopIteration("issue29692:Unchained")
            except Exception as ex:
                self.assertIs(type(ex), StopIteration)
                self.assertEqual(ex.args[0], "issue29692:Unchained")
                self.assertIsNone(ex.__cause__)

        f(torch.randn(2))

    @unittest.expectedFailure
    def test_contextmanager_wrap_runtimeerror(self):
        @contextmanager
        def woohoo():
            try:
                yield
            except Exception as exc:
                raise RuntimeError(f"caught {exc}") from exc

        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            with self.assertRaises(RuntimeError):
                with woohoo():
                    1 / 0

        fn(torch.randn(2, 3))

        # If the context manager wrapped StopIteration in a RuntimeError,
        # we also unwrap it, because we can't tell whether the wrapping was
        # done by the generator machinery or by the generator itself.
        with self.assertRaises(StopIteration):
            with woohoo():
                raise StopIteration

    @unittest.expectedFailure
    def test_keywords(self):
        # Ensure no keyword arguments are inhibited
        @contextmanager
        def woohoo(self, func, args, kwds):
            yield (self, func, args, kwds)

        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            with woohoo(self=11, func=22, args=33, kwds=44) as target:
                self.assertEqual(target, (11, 22, 33, 44))

        fn(torch.randn(2, 3))

    @unittest.expectedFailure
    def test_recursive(self):
        depth = 0
        ncols = 0

        @contextmanager
        def woohoo():
            nonlocal ncols
            ncols += 1
            nonlocal depth
            before = depth
            depth += 1
            yield
            depth -= 1
            self.assertEqual(depth, before)

        @woohoo()
        def recursive():
            if depth < 10:
                recursive()

        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            recursive()

        fn(torch.randn(2, 3))

        self.assertEqual(ncols, 10)
        self.assertEqual(depth, 0)


instantiate_parametrized_tests(CtxManagerTests)
instantiate_parametrized_tests(ContextlibContextManagerTests)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
