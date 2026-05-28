# Owner(s): ["module: dynamo"]

import torch
import torch._dynamo.test_case
from torch.testing._internal.common_utils import instantiate_parametrized_tests


class ComplexDynamoTestCase(torch._dynamo.test_case.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        torch._functorch.config.enable_complex_wrapper = True

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        torch._functorch.config.enable_complex_wrapper = False


def sample_function(a: torch.Tensor, b: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    y = torch.mm(a, b)
    return y.abs() + x


class ComplexTests(ComplexDynamoTestCase):
    def test_simple(self):
        fn_c = torch.compile(sample_function, fullgraph=True)
        a = torch.randn(2, 2, dtype=torch.complex64)
        b = torch.randn(2, 2, dtype=torch.complex64)
        x = torch.randn(2, 2)

        self.assertEqual(fn_c(a, b, x), sample_function(a, b, x))

    def test_no_complex_in_out(self):
        def f(re, im):
            c = torch.complex(re, im)
            return c.abs()

        re = torch.randn(2, 2, dtype=torch.float32)
        im = torch.randn(2, 2, dtype=torch.float32)
        fn_c = torch.compile(f, fullgraph=True)
        self.assertEqual(fn_c(re, im), f(re, im))

    def test_no_complex_inputs(self):
        def f(x, y):
            a = torch.complex(torch.sin(x), torch.cos(y))
            b = torch.complex(torch.cos(x), torch.sin(y))
            return sample_function(a, b, x)

        fn_c = torch.compile(f, fullgraph=True)

        x = torch.randn(2, 2)
        y = torch.randn(2, 2)
        self.assertEqual(fn_c(x, y), f(x, y))

    def test_list_out(self):
        def f(a, b, x):
            return [sample_function(a, b, x), sample_function(b, a, x)]

        fn_c = torch.compile(f, fullgraph=True)

        a = torch.randn(2, 2, dtype=torch.complex64)
        b = torch.randn(2, 2, dtype=torch.complex64)
        x = torch.randn(2, 2)
        self.assertEqual(fn_c(a, b, x), f(a, b, x))

    def test_view_as_real_with_mutation(self):
        # view_as_real decomposes to stack (a copy) under ComplexTensor, so
        # the result does not alias the input. This is safe because we run
        # post-functionalization (no mutations in the graph) and Inductor
        # re-establishes any required output aliasing.
        def f(a):
            out = torch.view_as_real(a)
            a[...] = torch.zeros_like(a)
            return out

        a = torch.randn(2, 2, dtype=torch.complex64)
        fn_c = torch.compile(f, fullgraph=True)
        self.assertEqual(fn_c(a.clone()), f(a.clone()))

    def test_aliasing_semantics(self):
        def f(a):
            return a

        def mutate(fn):
            a = torch.ones(2, 2, dtype=torch.complex64)
            out = fn(a)
            a[...] = torch.zeros_like(a)
            return out

        fn_c = torch.compile(f, fullgraph=True)
        self.assertEqual(mutate(f), mutate(fn_c))

    def test_input_mutation(self):
        def f(a, b):
            a.mul_(b)
            return a.abs()

        a = torch.randn(2, 2, dtype=torch.complex64)
        b = torch.randn(2, 2, dtype=torch.complex64)

        a_ref = a.clone()
        ref = f(a_ref, b)

        fn_c = torch.compile(f, fullgraph=True)
        a_test = a.clone()
        result = fn_c(a_test, b)

        self.assertEqual(result, ref)
        self.assertEqual(a_test, a_ref)

    def test_view_as_real(self):
        def f():
            a = torch.ones(2, 2, dtype=torch.complex64)
            b = torch.zeros(2, 2, dtype=torch.complex64)
            return torch.view_as_real(a).clone(), b

        fn_c = torch.compile(f, fullgraph=True)
        self.assertEqual(f(), fn_c())

    def test_rope(self):
        """Test the RoPE function, taken from:
        https://github.com/meta-llama/llama/blob/689c7f261b9c5514636ecc3c5fefefcbb3e6eed7/llama/model.py#L80-L161"""

        def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
            freqs = 1.0 / (
                theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)
            )
            t = torch.arange(end, device=freqs.device)
            freqs = torch.outer(t, freqs).float()
            freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
            return freqs_cis

        def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
            ndim = x.ndim
            assert ndim >= 2  # noqa: S101
            assert freqs_cis.shape == (x.shape[1], x.shape[-1])  # noqa: S101
            shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
            return freqs_cis.view(*shape)

        def apply_rotary_emb(
            xq: torch.Tensor,
            xk: torch.Tensor,
            freqs_cis: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            # Clone to avoid aliasing errors
            xq_ = torch.view_as_complex(
                xq.clone().float().reshape(*xq.shape[:-1], -1, 2)
            )
            xk_ = torch.view_as_complex(
                xk.clone().float().reshape(*xk.shape[:-1], -1, 2)
            )
            freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
            xq_out = torch.view_as_real(xq_ * freqs_cis).clone().flatten(3)
            xk_out = torch.view_as_real(xk_ * freqs_cis).clone().flatten(3)
            return xq_out.type_as(xq), xk_out.type_as(xk)

        def f(xq: torch.Tensor, xk: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            freqs_cis = precompute_freqs_cis(xq.shape[-1], xq.shape[1])
            return apply_rotary_emb(xq, xk, freqs_cis)

        g = torch.Generator().manual_seed(42)
        xq = torch.randn(4, 32, 64, generator=g)
        xk = torch.randn(4, 32, 64, generator=g)

        fn_c = torch.compile(f, fullgraph=True)
        self.assertEqual(f(xq, xk), fn_c(xq, xk))


instantiate_parametrized_tests(ComplexTests)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
