# Owner(s): ["module: dynamo"]
import contextlib

import torch._dynamo.test_case
import torch._dynamo.testing
from torch._dynamo.testing import CompileCounter
from torch.backends.cuda import SDPAParams
from torch.nn.attention import _cur_sdpa_kernel_backends, sdpa_kernel, SDPBackend


@contextlib.contextmanager
def allow_in_graph_sdpa_params():
    global SDPAParams
    try:
        old = SDPAParams
        SDPAParams = torch._dynamo.allow_in_graph(SDPAParams)
        yield
    finally:
        SDPAParams = old


class TestSDPA(torch._dynamo.test_case.TestCase):
    def assert_ref_equals_params(self, actual, expected):
        self.assertIs(actual.query, expected.query)
        self.assertIs(actual.key, expected.key)
        self.assertIs(actual.value, expected.value)
        self.assertIs(actual.attn_mask, expected.attn_mask)

    def test_returns_SDPAParams(self):
        with allow_in_graph_sdpa_params():
            counter = CompileCounter()

            @torch.compile(fullgraph=True, backend=counter)
            def fn(q, k, v, m):
                return SDPAParams(q, k, v, m, 0.1, True, False)

            q = torch.randn(10)
            k = torch.randn(10)
            v = torch.randn(10)
            m = torch.randn(10)
            o = fn(q, k, v, m)
            self.assertTrue(isinstance(o, SDPAParams))
            self.assert_ref_equals_params(o, SDPAParams(q, k, v, m, 0.1, True, False))
            self.assertEqual(counter.frame_count, 1)

    def test_graph_break_SDPAParams(self):
        with allow_in_graph_sdpa_params():
            counter = CompileCounter()

            @torch.compile(backend=counter)
            def fn(q, k, v, m):
                z = SDPAParams(q, k, v, m, 0.1, True, False)
                torch._dynamo.graph_break()
                return z, q + 1

            q = torch.randn(10)
            k = torch.randn(10)
            v = torch.randn(10)
            m = torch.randn(10)
            o, _ = fn(q, k, v, m)
            self.assertTrue(isinstance(o, SDPAParams))
            self.assert_ref_equals_params(o, SDPAParams(q, k, v, m, 0.1, True, False))
            self.assertEqual(counter.frame_count, 2)

    def test_input_SDPAParams(self):
        with allow_in_graph_sdpa_params():
            counter = CompileCounter()

            @torch.compile(backend=counter)
            def fn(sdpap, q):
                torch._dynamo.graph_break()
                return sdpap, sdpap.query + q

            q = torch.randn(10)
            k = torch.randn(10)
            v = torch.randn(10)
            m = torch.randn(10)
            s = SDPAParams(q, k, v, m, 0.1, True, False)
            o, _ = fn(s, q)
            self.assertIs(o, s)
            self.assertEqual(counter.frame_count, 1)

    def test_intermediate_attr_access_SDPAParams(self):
        with allow_in_graph_sdpa_params():
            counter = CompileCounter()

            @torch.compile(fullgraph=True, backend=counter)
            def fn(q, k, v, m):
                q += 1
                z = SDPAParams(q, k, v, m, 0.1, True, False)
                a = z.query
                return a + 1, z, q

            q = torch.randn(10)
            k = torch.randn(10)
            v = torch.randn(10)
            m = torch.randn(10)
            _, o, _ = fn(q, k, v, m)
            expected = SDPAParams(q, k, v, m, 0.1, True, False)
            self.assert_ref_equals_params(o, expected)
            self.assertEqual(counter.frame_count, 1)

    def test_sdpa_c_functions_no_graph_break(self):
        counter = CompileCounter()

        @torch.compile(fullgraph=True, backend=counter)
        def test_cur_sdpa_kernel_backends():
            return _cur_sdpa_kernel_backends()

        result = test_cur_sdpa_kernel_backends()

        self.assertIsInstance(result, list)
        self.assertEqual(counter.frame_count, 1)

    def test_sdpa_kernel_decorator_with_compile(self):
        SDPA_BACKEND_PRIORITY = [
            SDPBackend.MATH,
            SDPBackend.EFFICIENT_ATTENTION,
            SDPBackend.FLASH_ATTENTION,
        ]

        @sdpa_kernel(backends=SDPA_BACKEND_PRIORITY, set_priority=True)
        def scaled_dot_product_attention(q, k, v, *args, **kwargs):
            return torch.nn.functional.scaled_dot_product_attention(
                q, k, v, *args, **kwargs
            )

        counter = CompileCounter()

        @torch.compile(fullgraph=True, backend=counter)
        def f(x):
            return scaled_dot_product_attention(x, x, x)

        x = torch.rand(128, 64, 64, 256, dtype=torch.float16)
        result = f(x)

        self.assertEqual(result.shape, x.shape)
        self.assertEqual(counter.frame_count, 1)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
