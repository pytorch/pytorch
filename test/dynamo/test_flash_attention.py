# Owner(s): ["module: dynamo"]
import contextlib

import torch._dynamo.test_case
import torch._dynamo.testing
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


class TestFlashAttention(torch._dynamo.test_case.TestCase):
    def test_returns_SDPAParams(self):
        with allow_in_graph_sdpa_params():

            @torch.compile()
            def fn(q, k, v, m):
                return SDPAParams(q, k, v, m, 0.1, True)

            q = torch.randn(10)
            k = torch.randn(10)
            v = torch.randn(10)
            m = torch.randn(10)
            o = fn(q, k, v, m)
            self.assertTrue(isinstance(o, SDPAParams))
            self.assertIs(o.query, q)
            self.assertIs(o.key, k)
            self.assertIs(o.value, v)
            self.assertIs(o.attn_mask, m)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
