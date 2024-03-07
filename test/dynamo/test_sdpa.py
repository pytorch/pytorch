# Owner(s): ["module: dynamo"]
import contextlib

import torch._dynamo.test_case
import torch._dynamo.testing
from torch._dynamo.testing import CompileCounter
from torch.backends.cuda import SDPAParams


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
                return SDPAParams(q, k, v, m, 0.1, True)

            q = torch.randn(10)
            k = torch.randn(10)
            v = torch.randn(10)
            m = torch.randn(10)
            o = fn(q, k, v, m)
            self.assertTrue(isinstance(o, SDPAParams))
            self.assert_ref_equals_params(o, SDPAParams(q, k, v, m, 0.1, True))
            self.assertEqual(counter.frame_count, 1)

    def test_graph_break_SDPAParams(self):
        with allow_in_graph_sdpa_params():
            counter = CompileCounter()

            @torch.compile(backend=counter)
            def fn(q, k, v, m):
                z = SDPAParams(q, k, v, m, 0.1, True)
                torch._dynamo.graph_break()
                return z, q + 1

            q = torch.randn(10)
            k = torch.randn(10)
            v = torch.randn(10)
            m = torch.randn(10)
            o, _ = fn(q, k, v, m)
            self.assertTrue(isinstance(o, SDPAParams))
            self.assert_ref_equals_params(o, SDPAParams(q, k, v, m, 0.1, True))
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
            s = SDPAParams(q, k, v, m, 0.1, True)
            o, _ = fn(s, q)
            self.assertIs(o, s)
            self.assertEqual(counter.frame_count, 1)

    def test_intermediate_attr_access_SDPAParams(self):
        with allow_in_graph_sdpa_params():
            counter = CompileCounter()

            @torch.compile(fullgraph=True, backend=counter)
            def fn(q, k, v, m):
                q += 1
                z = SDPAParams(q, k, v, m, 0.1, True)
                a = z.query
                return a + 1, z, q

            q = torch.randn(10)
            k = torch.randn(10)
            v = torch.randn(10)
            m = torch.randn(10)
            _, o, _ = fn(q, k, v, m)
            expected = SDPAParams(q, k, v, m, 0.1, True)
            self.assert_ref_equals_params(o, expected)
            self.assertEqual(counter.frame_count, 1)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
