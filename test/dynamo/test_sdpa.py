# Owner(s): ["module: dynamo"]
import torch._dynamo.test_case
import torch._dynamo.testing
from torch._dynamo.testing import CompileCounter
from torch.backends.cuda import can_use_flash_attention, SDPAParams
from torch.nn.attention import _cur_sdpa_kernel_backends, sdpa_kernel, SDPBackend
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import HardwareClassification


def assert_ref_equals_params(test_case, actual, expected):
    test_case.assertIs(actual.query, expected.query)
    test_case.assertIs(actual.key, expected.key)
    test_case.assertIs(actual.value, expected.value)
    test_case.assertIs(actual.attn_mask, expected.attn_mask)


class TestSDPA(torch._dynamo.test_case.TestCase):
    hw_classification = HardwareClassification.GENERIC

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


class TestSDPACUDA(torch._dynamo.test_case.TestCase):
    hw_classification = HardwareClassification.CUDA

    def test_returns_SDPAParams(self, device):
        counter = CompileCounter()

        @torch.compile(fullgraph=True, backend=counter)
        def fn(q, k, v, m):
            return SDPAParams(q, k, v, m, 0.1, True, False)

        q = torch.randn(10, device=device)
        k = torch.randn(10, device=device)
        v = torch.randn(10, device=device)
        m = torch.randn(10, device=device)
        o = fn(q, k, v, m)
        self.assertTrue(isinstance(o, SDPAParams))
        assert_ref_equals_params(self, o, SDPAParams(q, k, v, m, 0.1, True, False))
        self.assertEqual(counter.frame_count, 1)

    def test_graph_break_SDPAParams(self, device):
        counter = CompileCounter()

        @torch.compile(backend=counter)
        def fn(q, k, v, m):
            z = SDPAParams(q, k, v, m, 0.1, True, False)
            torch._dynamo.graph_break()
            return z, q + 1

        q = torch.randn(10, device=device)
        k = torch.randn(10, device=device)
        v = torch.randn(10, device=device)
        m = torch.randn(10, device=device)
        o, _ = fn(q, k, v, m)
        self.assertTrue(isinstance(o, SDPAParams))
        assert_ref_equals_params(self, o, SDPAParams(q, k, v, m, 0.1, True, False))
        self.assertEqual(counter.frame_count, 2)

    def test_input_SDPAParams(self, device):
        counter = CompileCounter()

        @torch.compile(backend=counter)
        def fn(sdpap, q):
            torch._dynamo.graph_break()
            return sdpap, sdpap.query + q

        q = torch.randn(10, device=device)
        k = torch.randn(10, device=device)
        v = torch.randn(10, device=device)
        m = torch.randn(10, device=device)
        s = SDPAParams(q, k, v, m, 0.1, True, False)
        o, _ = fn(s, q)
        self.assertIs(o, s)
        self.assertEqual(counter.frame_count, 1)

    def test_intermediate_attr_access_SDPAParams(self, device):
        counter = CompileCounter()

        @torch.compile(fullgraph=True, backend=counter)
        def fn(q, k, v, m):
            q += 1
            z = SDPAParams(q, k, v, m, 0.1, True, False)
            a = z.query
            return a + 1, z, q

        q = torch.randn(10, device=device)
        k = torch.randn(10, device=device)
        v = torch.randn(10, device=device)
        m = torch.randn(10, device=device)
        _, o, _ = fn(q, k, v, m)
        expected = SDPAParams(q, k, v, m, 0.1, True, False)
        assert_ref_equals_params(self, o, expected)
        self.assertEqual(counter.frame_count, 1)

    def test_can_use_flash_attention_no_graph_break(self, device):
        counter = CompileCounter()

        @torch.compile(fullgraph=True, backend=counter)
        def fn(q, k, v):
            return can_use_flash_attention(SDPAParams(q, k, v, None, 0.0, True, False))

        q = torch.randn(2, 2, 8, 8, device=device)
        expected = can_use_flash_attention(SDPAParams(q, q, q, None, 0.0, True, False))
        self.assertEqual(fn(q, q, q), expected)
        self.assertEqual(counter.frame_count, 1)


instantiate_device_type_tests(TestSDPACUDA, globals(), only_for="cuda")


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
