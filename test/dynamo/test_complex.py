# Owner(s): ["module: dynamo"]
import itertools
import unittest

import torch
import torch._dynamo.test_case
from torch.testing._internal.common_utils import (
    HardwareClassification,
    instantiate_parametrized_tests,
    parametrize,
)


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
    hw_classification = HardwareClassification.GENERIC

    def test_simple(self):
        fn_c = torch.compile(sample_function, fullgraph=True)  # noqa: UNSPECIFIED_BACKEND
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
        fn_c = torch.compile(f, fullgraph=True)  # noqa: UNSPECIFIED_BACKEND
        self.assertEqual(fn_c(re, im), f(re, im))

    def test_no_complex_inputs(self):
        def f(x, y):
            a = torch.complex(torch.sin(x), torch.cos(y))
            b = torch.complex(torch.cos(x), torch.sin(y))
            return sample_function(a, b, x)

        fn_c = torch.compile(f, fullgraph=True)  # noqa: UNSPECIFIED_BACKEND

        x = torch.randn(2, 2)
        y = torch.randn(2, 2)
        self.assertEqual(fn_c(x, y), f(x, y))

    def test_list_out(self):
        def f(a, b, x):
            return [sample_function(a, b, x), sample_function(b, a, x)]

        fn_c = torch.compile(f, fullgraph=True)  # noqa: UNSPECIFIED_BACKEND

        a = torch.randn(2, 2, dtype=torch.complex64)
        b = torch.randn(2, 2, dtype=torch.complex64)
        x = torch.randn(2, 2)
        self.assertEqual(fn_c(a, b, x), f(a, b, x))

    def test_incorrect_aliasing_semantics_raises(self):
        # view_as_real decomposes to stack (a copy), so mutation of the
        # original complex tensor does not propagate through the view.
        def f(a):
            out = torch.view_as_real(a)
            a[...] = torch.zeros_like(a)
            return out

        a = torch.randn(2, 2, dtype=torch.complex64)
        fn_c = torch.compile(f, fullgraph=True)  # noqa: UNSPECIFIED_BACKEND
        self.assertRaises(RuntimeError, fn_c, a)

    @unittest.expectedFailure
    def test_incorrect_aliasing_semantics(self):
        # Once this succeeds, remove the xfail and the test above
        def f(a):
            out = torch.view_as_real(a)
            a[...] = torch.zeros_like(a)
            return out

        a = torch.randn(2, 2, dtype=torch.complex64)
        fn_c = torch.compile(f, fullgraph=True)  # noqa: UNSPECIFIED_BACKEND
        ref = f(a)
        test = fn_c(a)
        self.assertEqual(ref, test)

    def test_conj_alias(self):
        def f():
            a = torch.zeros(2, 2, dtype=torch.complex64)
            b = a.conj()
            b[...] = torch.ones_like(b)
            return a

        fn_c = torch.compile(f, fullgraph=True)  # noqa: UNSPECIFIED_BACKEND
        self.assertEqual(f(), fn_c())

    def test_neg_alias(self):
        def f():
            a = torch.zeros(2, 2, dtype=torch.complex64)
            b = a._neg_view()
            b[...] = torch.ones_like(b)
            return a

        fn_c = torch.compile(f, fullgraph=True)  # noqa: UNSPECIFIED_BACKEND
        self.assertEqual(f(), fn_c())

    def test_complex_output_aliases_input(self):
        # Once this succeeds, remove the xfail and the test above
        def f(a):
            return a[...]

        a = torch.randn(2, 2, dtype=torch.complex64)
        fn_c = torch.compile(f, fullgraph=True)  # noqa: UNSPECIFIED_BACKEND
        ref = f(a)
        test = fn_c(a)
        self.assertEqual(ref, test)
        a[...] = torch.zeros_like(a)
        self.assertEqual(ref, test)

    def test_complex_input_mutation_raises(self):
        def f(a, b):
            a.mul_(b)
            return a.abs()

        a = torch.randn(2, 2, dtype=torch.complex64)
        b = torch.randn(2, 2, dtype=torch.complex64)

        fn_c = torch.compile(f, fullgraph=True)  # noqa: UNSPECIFIED_BACKEND
        self.assertRaises(RuntimeError, fn_c, a, b)

    @unittest.expectedFailure
    def test_complex_input_mutation(self):
        # Once this succeeds, remove the xfail and the test above
        def f(a, b):
            a.mul_(b)
            return a.abs()

        a = torch.randn(2, 2, dtype=torch.complex64)
        b = torch.randn(2, 2, dtype=torch.complex64)

        fn_c = torch.compile(f, fullgraph=True)  # noqa: UNSPECIFIED_BACKEND

        a_ref = a.clone()
        a_test = a.clone()
        self.assertEqual(f(a_test, b), fn_c(a_ref, b))
        self.assertEqual(a_test, a_ref)

    def test_aliasing_semantics(self):
        def mutate():
            a = torch.ones(2, 2, dtype=torch.complex64)
            out = a[...]
            a[...] = torch.zeros_like(a)
            return out

        mutate_c = torch.compile(mutate, fullgraph=True)  # noqa: UNSPECIFIED_BACKEND
        self.assertEqual(mutate(), mutate_c())

    def test_view_as_real(self):
        def f():
            a = torch.ones(2, 2, dtype=torch.complex64)
            b = torch.zeros(2, 2, dtype=torch.complex64)
            return torch.view_as_real(a).clone(), b

        fn_c = torch.compile(f, fullgraph=True)  # noqa: UNSPECIFIED_BACKEND
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
            xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
            xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
            freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
            xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
            xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
            return xq_out.type_as(xq), xk_out.type_as(xk)

        def f(xq: torch.Tensor, xk: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            freqs_cis = precompute_freqs_cis(xq.shape[-1], xq.shape[1])
            return apply_rotary_emb(xq, xk, freqs_cis)

        g = torch.Generator().manual_seed(42)
        xq = torch.randn(4, 32, 64, generator=g)
        xk = torch.randn(4, 32, 64, generator=g)

        fn_c = torch.compile(f, fullgraph=True)  # noqa: UNSPECIFIED_BACKEND
        self.assertEqual(f(xq, xk), fn_c(xq, xk))

    def _assert_decomposed_no_complex(self, fn, *args):
        from functorch.compile import make_boxed_func
        from torch._dynamo.backends.common import aot_autograd
        from torch.utils._pytree import tree_leaves

        captured_graphs = []

        def fw_compiler(gm, example_inputs):
            captured_graphs.append(gm)
            return make_boxed_func(gm)

        ref = fn(*args)
        opt_aot = torch.compile(
            fn, backend=aot_autograd(fw_compiler=fw_compiler), fullgraph=True
        )
        self.assertEqual(ref, opt_aot(*args))
        self.assertEqual(len(captured_graphs), 1)
        for node in captured_graphs[0].graph.nodes:
            for val in tree_leaves(node.meta.get("val")):
                if isinstance(val, torch.Tensor):
                    self.assertFalse(
                        val.dtype.is_complex,
                        f"Unexpected complex node {node} ({val.dtype}) in decomposed graph",
                    )
            if node.op == "call_function":
                target_name = str(node.target)
                self.assertNotIn("view_as_complex", target_name)
                self.assertNotIn("view_as_real", target_name)
                self.assertNotIn("aten.complex", target_name)
                self.assertNotIn("aten.polar", target_name)

        opt_inductor = torch.compile(fn, fullgraph=True)  # noqa: UNSPECIFIED_BACKEND
        self.assertEqual(ref, opt_inductor(*args))

    def test_real_to_complex_constructors_decomposed(self):
        g = torch.Generator().manual_seed(0)
        x = torch.randn(4, 8, 2, generator=g)
        y = torch.randn(4, 8, 2, generator=g)
        re = torch.randn(4, 8, generator=g)
        im = torch.randn(4, 8, generator=g)
        abs_t = torch.rand(4, 8, generator=g)
        angle = torch.randn(4, 8, generator=g)

        cases = {
            "view_as_complex": (
                lambda a, b: torch.view_as_real(
                    torch.view_as_complex(a) * torch.view_as_complex(b)
                ),
                (x, y),
            ),
            "view_as_complex_copy": (
                lambda a, b: torch.view_as_real(
                    torch.view_as_complex_copy(a) * torch.view_as_complex_copy(b)
                ),
                (x, y),
            ),
            "complex": (
                lambda r1, i1, r2, i2: torch.view_as_real(
                    torch.complex(r1, i1) * torch.complex(r2, i2)
                ),
                (re, im, re + 0.5, im - 0.5),
            ),
            "polar_rope": (
                lambda q, mag, ang: torch.view_as_real(
                    torch.view_as_complex(q) * torch.polar(mag, ang)
                ),
                (x, abs_t, angle),
            ),
        }

        for name, (fn, inputs) in cases.items():
            with self.subTest(case=name):
                self._assert_decomposed_no_complex(fn, *inputs)

    def test_complex_and_polar_broadcast(self):
        g = torch.Generator().manual_seed(1)
        a = torch.randn(2, 1, generator=g)
        b = torch.randn(1, 3, generator=g)
        re2 = torch.randn(2, 3, generator=g)
        im2 = torch.randn(2, 3, generator=g)

        for name, op in [("complex", torch.complex), ("polar", torch.polar)]:
            with self.subTest(op=name):

                def fn(lhs, rhs, r2, i2, ctor=op):
                    return torch.view_as_real(ctor(lhs, rhs) * torch.complex(r2, i2))

                self._assert_decomposed_no_complex(fn, a, b, re2, im2)

    def test_polar_edge_cases(self):
        from torch._subclasses.complex_tensor import ComplexTensor, WrapComplexMode

        abs_t = torch.tensor([[1.5], [2.0]], dtype=torch.float32)
        angle = torch.tensor([[0.25, -0.5, 1.0]], dtype=torch.float32)
        expected = torch.polar(abs_t, angle)
        with WrapComplexMode():
            wrapped = torch.polar(abs_t, angle)
            scalar_res = torch.ops.aten.polar.float(2.0, 0.5)
        self.assertIsInstance(wrapped, ComplexTensor)
        self.assertEqual(wrapped.as_interleaved(), expected)
        self.assertEqual(scalar_res, torch.ops.aten.polar.float(2.0, 0.5))

        with WrapComplexMode():
            with self.assertRaises(RuntimeError):
                torch.polar(
                    torch.ones(2, dtype=torch.float32),
                    torch.ones(2, dtype=torch.float64),
                )

    @parametrize("dtype", [torch.float16, torch.float32, torch.float64])
    def test_polar_special_values(self, dtype):
        from torch._subclasses.complex_tensor import WrapComplexMode

        special_vals = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            float("inf"),
            float("-inf"),
            float("nan"),
        ]
        pairs = list(itertools.product(special_vals, special_vals)) + [
            (100.0, 0.25),
            (200.0, 1.5),
            (65504.0, 0.0),
        ]
        abs_t = torch.tensor([a for a, _ in pairs], dtype=dtype)
        angle = torch.tensor([b for _, b in pairs], dtype=dtype)
        expected = torch.polar(abs_t, angle)

        with WrapComplexMode():
            wrapped = torch.polar(abs_t, angle)
        tol = {"atol": 0, "rtol": 0} if dtype == torch.float16 else {}
        self.assertEqual(
            torch.view_as_real(wrapped.as_interleaved()),
            torch.view_as_real(expected),
            equal_nan=True,
            **tol,
        )

        # Chunk below AVX-512 float16 tiling width (32) so Inductor's
        # _load_or_store_non_contiguous get_result_size (num_elems * 2) stays
        # < 64 and does not trigger (1ULL << 64) UB in Vectorized16<Half>::store.
        for start in range(0, len(pairs), 13):
            self._assert_decomposed_no_complex(
                lambda a, b: torch.view_as_real(torch.polar(a, b)),
                abs_t[start : start + 13],
                angle[start : start + 13],
            )

    def test_real_to_complex_aliasing_and_mutation(self):
        def fn_passthrough(x):
            return torch.view_as_real(torch.view_as_complex(x))

        def fn_mutate(x):
            c = torch.view_as_complex(x)
            c.mul_(2.0)
            return x

        def fn_return_complex(x):
            return torch.view_as_complex(x)

        for name, fn in [
            ("passthrough", fn_passthrough),
            ("return_complex", fn_return_complex),
        ]:
            with self.subTest(case=name):
                x = torch.randn(4, 2)
                ref = fn(x)
                opt = torch.compile(fn, fullgraph=True)(x)  # noqa: UNSPECIFIED_BACKEND
                self.assertEqual(opt, ref)
                self.assertEqual(
                    opt.untyped_storage().data_ptr(), x.untyped_storage().data_ptr()
                )

        x_ref = torch.randn(4, 2)
        x_opt = x_ref.clone()
        out_ref = fn_mutate(x_ref)
        out_opt = torch.compile(fn_mutate, fullgraph=True)(  # noqa: UNSPECIFIED_BACKEND
            x_opt
        )
        self.assertEqual(out_opt, out_ref)
        self.assertEqual(x_opt, x_ref)

        def fn_complex_mutate_then_return(x, y):
            c = torch.complex(x, y)
            x.add_(1.0)
            return c

        def fn_copy_mutate_then_return(x):
            c = torch.view_as_complex_copy(x)
            x.add_(1.0)
            return c

        x_ref = torch.randn(4)
        y_ref = torch.randn(4)
        x_opt = x_ref.clone()
        out_ref = fn_complex_mutate_then_return(x_ref, y_ref)
        out_opt = torch.compile(
            fn_complex_mutate_then_return, backend="aot_eager", fullgraph=True
        )(x_opt, y_ref)
        self.assertEqual(out_opt, out_ref)
        self.assertEqual(x_opt, x_ref)

        x_ref = torch.randn(1, 2)
        x_opt = x_ref.clone()
        out_ref = fn_copy_mutate_then_return(x_ref)
        out_opt = torch.compile(
            fn_copy_mutate_then_return, backend="aot_eager", fullgraph=True
        )(x_opt)
        self.assertEqual(out_opt, out_ref)
        self.assertEqual(x_opt, x_ref)


instantiate_parametrized_tests(ComplexTests)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
