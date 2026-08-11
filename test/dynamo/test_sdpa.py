# Owner(s): ["module: dynamo"]
import unittest

import torch._dynamo.test_case
import torch._dynamo.testing
from torch._dynamo.graph_utils import remove_noop_sdpa_masks
from torch._dynamo.testing import CompileCounter
from torch.backends.cuda import can_use_flash_attention, SDPAParams
from torch.nn.attention import _cur_sdpa_kernel_backends, sdpa_kernel, SDPBackend
from torch.testing._internal.common_cuda import PLATFORM_SUPPORTS_FLASH_ATTENTION


class TestSDPA(torch._dynamo.test_case.TestCase):
    def _assert_mask_is_not_removed(self, fn, *inputs):
        expected = fn(*inputs)

        def backend(gm, example_inputs):
            self.assertFalse(remove_noop_sdpa_masks(gm))
            return gm.forward

        actual = torch.compile(fn, backend=backend, fullgraph=True)(*inputs)
        self.assertEqual(actual, expected)

        torch._dynamo.reset()
        eager_backend_result = torch.compile(fn, backend="eager", fullgraph=True)(
            *inputs
        )
        self.assertEqual(eager_backend_result, expected)

    def assert_ref_equals_params(self, actual, expected):
        self.assertIs(actual.query, expected.query)
        self.assertIs(actual.key, expected.key)
        self.assertIs(actual.value, expected.value)
        self.assertIs(actual.attn_mask, expected.attn_mask)

    def test_returns_SDPAParams(self):
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

    def test_can_use_flash_attention_no_graph_break(self):
        counter = CompileCounter()

        @torch.compile(fullgraph=True, backend=counter)
        def fn(q, k, v):
            return can_use_flash_attention(SDPAParams(q, k, v, None, 0.0, True, False))

        q = torch.randn(2, 2, 8, 8)
        expected = can_use_flash_attention(SDPAParams(q, q, q, None, 0.0, True, False))
        self.assertEqual(fn(q, q, q), expected)
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

    def test_remove_noop_sdpa_mask_from_graph(self):
        def fn(q, k, v):
            seq_len = q.size(2)
            q_indices = torch.arange(seq_len, device=q.device) + 0
            attention_mask = (q_indices[None, None, :, None] >= 0).expand(
                q.size(0), 1, seq_len, seq_len
            )
            return torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=attention_mask
            )

        seen_graphs = []

        def backend(gm, example_inputs):
            self.assertTrue(remove_noop_sdpa_masks(gm))
            seen_graphs.append(gm)
            return gm.forward

        q = torch.randn(2, 2, 4, 8)
        k = torch.randn(2, 2, 4, 8)
        v = torch.randn(2, 2, 4, 8)

        torch.compile(fn, backend=backend, fullgraph=True)(q, k, v)

        sdpa_nodes = [
            node
            for node in seen_graphs[0].graph.nodes
            if node.op == "call_function"
            and node.target is torch._C._nn.scaled_dot_product_attention
        ]
        self.assertEqual(len(sdpa_nodes), 1)
        self.assertIsNone(sdpa_nodes[0].kwargs["attn_mask"])

    def test_does_not_remove_wrapping_arange_sdpa_mask(self):
        def fn(q, k, v):
            seq_len = q.size(2)
            q_indices = torch.arange(seq_len, dtype=torch.int8, device=q.device)
            attention_mask = (q_indices[None, None, :, None] >= 0).expand(
                q.size(0), 1, seq_len, seq_len
            )
            return torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=attention_mask
            )

        def backend(gm, example_inputs):
            self.assertFalse(remove_noop_sdpa_masks(gm))
            return gm.forward

        q = torch.randn(1, 1, 130, 8)
        k = torch.randn(1, 1, 130, 8)
        v = torch.randn(1, 1, 130, 8)
        expected = fn(q, k, v)

        custom_backend_result = torch.compile(fn, backend=backend, fullgraph=True)(
            q, k, v
        )
        self.assertEqual(custom_backend_result, expected)

        torch._dynamo.reset()
        eager_backend_result = torch.compile(fn, backend="eager", fullgraph=True)(
            q, k, v
        )
        self.assertEqual(eager_backend_result, expected)

    def test_does_not_remove_wrapping_arange_add_sdpa_mask(self):
        def fn(q, k, v):
            seq_len = q.size(2)
            q_indices = torch.arange(seq_len, dtype=torch.int8, device=q.device) + 100
            attention_mask = (q_indices[None, None, :, None] >= 0).expand(
                q.size(0), 1, seq_len, seq_len
            )
            return torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=attention_mask
            )

        def backend(gm, example_inputs):
            self.assertFalse(remove_noop_sdpa_masks(gm))
            return gm.forward

        q = torch.randn(1, 1, 100, 8)
        k = torch.randn(1, 1, 100, 8)
        v = torch.randn(1, 1, 100, 8)
        expected = fn(q, k, v)

        custom_backend_result = torch.compile(fn, backend=backend, fullgraph=True)(
            q, k, v
        )
        self.assertEqual(custom_backend_result, expected)

        torch._dynamo.reset()
        eager_backend_result = torch.compile(fn, backend="eager", fullgraph=True)(
            q, k, v
        )
        self.assertEqual(eager_backend_result, expected)

    def test_does_not_remove_unsigned_arange_negative_threshold_sdpa_mask(self):
        def fn(q, k, v):
            seq_len = q.size(2)
            q_indices = torch.arange(seq_len, dtype=torch.uint8, device=q.device)
            attention_mask = (q_indices[None, None, :, None] >= -1).expand(
                q.size(0), 1, seq_len, seq_len
            )
            return torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=attention_mask
            )

        def backend(gm, example_inputs):
            self.assertFalse(remove_noop_sdpa_masks(gm))
            return gm.forward

        q = torch.randn(1, 1, 4, 8)
        k = torch.randn(1, 1, 4, 8)
        v = torch.randn(1, 1, 4, 8)
        expected = fn(q, k, v)

        custom_backend_result = torch.compile(fn, backend=backend, fullgraph=True)(
            q, k, v
        )
        self.assertEqual(custom_backend_result, expected)

        torch._dynamo.reset()
        eager_backend_result = torch.compile(fn, backend="eager", fullgraph=True)(
            q, k, v
        )
        self.assertEqual(eager_backend_result, expected)

    def test_does_not_remove_arange_with_keyword_negative_step(self):
        def fn(q, k, v):
            seq_len = q.size(2)
            q_indices = torch.arange(0, -seq_len, step=-1, device=q.device)
            attention_mask = (q_indices[None, None, :, None] >= 0).expand(
                q.size(0), 1, seq_len, seq_len
            )
            return torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=attention_mask
            )

        q = torch.randn(1, 1, 4, 8)
        self._assert_mask_is_not_removed(fn, q, q, q)

    def test_does_not_remove_dtype_reinterpreted_arange_mask(self):
        def fn(q, k, v):
            seq_len = q.size(2)
            q_indices = torch.arange(seq_len, dtype=torch.uint8, device=q.device).view(
                torch.int8
            )
            attention_mask = (q_indices[None, None, :, None] >= 0).expand(
                q.size(0), 1, seq_len, seq_len
            )
            return torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=attention_mask
            )

        q = torch.randn(1, 1, 130, 8)
        self._assert_mask_is_not_removed(fn, q, q, q)

    def test_does_not_remove_nonfinite_arange_add_mask(self):
        def fn(q, k, v):
            seq_len = q.size(2)
            q_indices = torch.add(
                0.0,
                torch.arange(seq_len, dtype=q.dtype, device=q.device),
                alpha=float("inf"),
            )
            attention_mask = (q_indices[None, None, :, None] >= 0).expand(
                q.size(0), 1, seq_len, seq_len
            )
            return torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=attention_mask
            )

        q = torch.randn(1, 1, 4, 8)
        self._assert_mask_is_not_removed(fn, q, q, q)

    def test_does_not_remove_mutated_bidirectional_mask(self):
        def fn(q, k, v):
            seq_len = q.size(2)
            q_indices = torch.arange(seq_len, device=q.device)
            attention_mask = (q_indices[None, None, :, None] >= 0).expand(
                q.size(0), 1, seq_len, seq_len
            )
            attention_mask.fill_(False)
            return torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=attention_mask
            )

        q = torch.randn(1, 1, 4, 8)
        self._assert_mask_is_not_removed(fn, q, q, q)

    def test_remove_noop_additive_sdpa_mask_from_graph(self):
        def fn(q, k, v):
            seq_len = q.size(2)
            attention_mask = torch.ones((q.size(0), seq_len), device=q.device)
            expanded_mask = attention_mask[:, None, None, :].expand(
                q.size(0), 1, seq_len, seq_len
            )
            expanded_mask = expanded_mask.to(dtype=q.dtype)
            inverted_mask = (
                torch.tensor(1.0, dtype=q.dtype, device=q.device) - expanded_mask
            )
            additive_mask = inverted_mask.masked_fill(
                inverted_mask.to(torch.bool), torch.finfo(q.dtype).min
            )
            return torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=additive_mask
            )

        seen_graphs = []

        def backend(gm, example_inputs):
            self.assertTrue(remove_noop_sdpa_masks(gm))
            seen_graphs.append(gm)
            return gm.forward

        q = torch.randn(2, 2, 4, 8)
        k = torch.randn(2, 2, 4, 8)
        v = torch.randn(2, 2, 4, 8)

        torch.compile(fn, backend=backend, fullgraph=True)(q, k, v)

        sdpa_nodes = [
            node
            for node in seen_graphs[0].graph.nodes
            if node.op == "call_function"
            and node.target is torch._C._nn.scaled_dot_product_attention
        ]
        self.assertEqual(len(sdpa_nodes), 1)
        self.assertIsNone(sdpa_nodes[0].kwargs["attn_mask"])

    def test_does_not_remove_nonzero_additive_sdpa_mask(self):
        def fn(q, k, v):
            attention_mask = torch.ones(
                (q.size(0), 1, q.size(2), q.size(2)),
                dtype=q.dtype,
                device=q.device,
            )
            return torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=attention_mask
            )

        def backend(gm, example_inputs):
            self.assertFalse(remove_noop_sdpa_masks(gm))
            return gm.forward

        q = torch.randn(2, 2, 4, 8)
        k = torch.randn(2, 2, 4, 8)
        v = torch.randn(2, 2, 4, 8)

        torch.compile(fn, backend=backend, fullgraph=True)(q, k, v)

    def test_does_not_remove_requires_grad_additive_sdpa_mask(self):
        def fn(q, k, v):
            seq_len = q.size(2)
            attention_mask = torch.ones((q.size(0), seq_len), device=q.device)
            expanded_mask = attention_mask[:, None, None, :].expand(
                q.size(0), 1, seq_len, seq_len
            )
            expanded_mask = expanded_mask.to(dtype=q.dtype)
            inverted_mask = (
                torch.tensor(1.0, dtype=q.dtype, device=q.device) - expanded_mask
            )
            additive_mask = inverted_mask.masked_fill(
                inverted_mask.to(torch.bool), torch.finfo(q.dtype).min
            )
            return torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=additive_mask
            )

        def backend(gm, example_inputs):
            sdpa_node = next(
                node
                for node in gm.graph.nodes
                if node.target is torch._C._nn.scaled_dot_product_attention
            )
            mask = sdpa_node.kwargs["attn_mask"]
            mask.meta["example_value"] = torch.empty_like(
                mask.meta["example_value"], requires_grad=True
            )
            self.assertFalse(remove_noop_sdpa_masks(gm))
            return gm.forward

        q = torch.randn(2, 2, 4, 8)
        k = torch.randn(2, 2, 4, 8)
        v = torch.randn(2, 2, 4, 8)

        torch.compile(fn, backend=backend, fullgraph=True)(q, k, v)

    def test_does_not_remove_additive_mask_with_out(self):
        def fn(q, k, v, attention_mask):
            seq_len = q.size(2)
            attention_mask = torch.ones((q.size(0), seq_len), out=attention_mask)
            expanded_mask = attention_mask[:, None, None, :].expand(
                q.size(0), 1, seq_len, seq_len
            )
            inverted_mask = torch.tensor(1.0, dtype=q.dtype) - expanded_mask
            additive_mask = inverted_mask.masked_fill(
                inverted_mask.to(torch.bool), torch.finfo(q.dtype).min
            )
            return torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=additive_mask
            )

        def backend(gm, example_inputs):
            self.assertFalse(remove_noop_sdpa_masks(gm))
            return gm.forward

        q = torch.randn(2, 2, 4, 8)
        attention_mask = torch.zeros(2, 4)
        torch.compile(fn, backend=backend, fullgraph=True)(q, q, q, attention_mask)
        self.assertEqual(attention_mask, torch.ones_like(attention_mask))

    def test_removal_preserves_unrelated_out_side_effect(self):
        def fn(q, k, v, out):
            torch.zeros(out.shape, out=out)
            seq_len = q.size(2)
            q_indices = torch.arange(seq_len, device=q.device)
            attention_mask = (q_indices[None, None, :, None] >= 0).expand(
                q.size(0), 1, seq_len, seq_len
            )
            return torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=attention_mask
            )

        def backend(gm, example_inputs):
            self.assertTrue(remove_noop_sdpa_masks(gm))
            return gm.forward

        q = torch.randn(1, 1, 4, 8)
        out = torch.full((2, 2), 9.0)
        torch.compile(fn, backend=backend, fullgraph=True)(q, q, q, out)
        self.assertEqual(out, torch.zeros_like(out))

    def test_does_not_remove_overflowing_zero_multiply_sdpa_mask(self):
        def fn(q, k, v):
            attention_mask = (
                torch.zeros(
                    (q.size(0), 1, q.size(2), q.size(2)),
                    dtype=q.dtype,
                    device=q.device,
                )
                * 1e100
            )
            return torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=attention_mask
            )

        def backend(gm, example_inputs):
            self.assertFalse(remove_noop_sdpa_masks(gm))
            return gm.forward

        q = torch.randn(1, 1, 4, 8)
        k = torch.randn(1, 1, 4, 8)
        v = torch.randn(1, 1, 4, 8)

        expected = fn(q, k, v)
        self.assertTrue(torch.isnan(expected).all())

        custom_backend_result = torch.compile(fn, backend=backend, fullgraph=True)(
            q, k, v
        )
        self.assertTrue(torch.isnan(custom_backend_result).all())

        torch._dynamo.reset()
        eager_backend_result = torch.compile(fn, backend="eager", fullgraph=True)(
            q, k, v
        )
        self.assertTrue(torch.isnan(eager_backend_result).all())

    def test_does_not_remove_mask_from_causal_sdpa(self):
        def fn(q, k, v):
            seq_len = q.size(2)
            q_indices = torch.arange(seq_len, device=q.device)
            attention_mask = (q_indices[None, None, :, None] >= 0).expand(
                q.size(0), 1, seq_len, seq_len
            )
            return torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=attention_mask
            )

        def backend(gm, example_inputs):
            sdpa_node = next(
                node
                for node in gm.graph.nodes
                if node.target is torch._C._nn.scaled_dot_product_attention
            )
            kwargs = dict(sdpa_node.kwargs)
            kwargs["is_causal"] = True
            sdpa_node.kwargs = kwargs
            self.assertFalse(remove_noop_sdpa_masks(gm))
            gm.recompile()
            return gm.forward

        q = torch.randn(1, 1, 4, 8)
        with self.assertRaisesRegex(
            RuntimeError, "Explicit attn_mask should not be set"
        ):
            torch.compile(fn, backend=backend, fullgraph=True)(q, q, q)

    def test_does_not_remove_mismatched_float_sdpa_mask(self):
        def fn(q, k, v):
            seq_len = q.size(2)
            attention_mask = torch.ones((q.size(0), seq_len), device=q.device)
            expanded_mask = attention_mask[:, None, None, :].expand(
                q.size(0), 1, seq_len, seq_len
            )
            expanded_mask = expanded_mask.to(dtype=q.dtype)
            inverted_mask = (
                torch.tensor(1.0, dtype=q.dtype, device=q.device) - expanded_mask
            )
            additive_mask = inverted_mask.masked_fill(
                inverted_mask.to(torch.bool), torch.finfo(q.dtype).min
            )
            return torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=additive_mask
            )

        def backend(gm, example_inputs):
            sdpa_node = next(
                node
                for node in gm.graph.nodes
                if node.target is torch._C._nn.scaled_dot_product_attention
            )
            mask = sdpa_node.kwargs["attn_mask"]
            inverted = mask.args[0]
            scalar_one, expanded_to = inverted.args[:2]

            for node in (expanded_to, scalar_one):
                kwargs = dict(node.kwargs)
                kwargs["dtype"] = torch.float64
                node.kwargs = kwargs
            for node in (expanded_to, scalar_one, inverted, mask):
                value = node.meta["example_value"]
                node.meta["example_value"] = torch.empty(
                    value.shape, dtype=torch.float64, device=value.device
                )

            self.assertFalse(remove_noop_sdpa_masks(gm))
            gm.recompile()
            return gm.forward

        q = torch.randn(1, 1, 4, 8)
        with self.assertRaisesRegex(RuntimeError, "Expected attn_mask dtype"):
            torch.compile(fn, backend=backend, fullgraph=True)(q, q, q)

    @unittest.skipIf(
        not torch.cuda.is_available() or not PLATFORM_SUPPORTS_FLASH_ATTENTION,
        "CUDA flash attention is not available",
    )
    def test_compiled_noop_sdpa_mask_uses_flash(self):
        def fn(q, k, v, attention_mask=None):
            if not torch.compiler.is_compiling() and attention_mask is None:
                mask = None
            else:
                seq_len = q.size(2)
                q_indices = torch.arange(seq_len, device=q.device)
                mask = (q_indices[None, None, :, None] >= 0).expand(
                    q.size(0), 1, seq_len, seq_len
                )
            return torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=mask
            )

        q = torch.randn(1, 4, 64, 64, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(1, 4, 64, 64, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(1, 4, 64, 64, device="cuda", dtype=torch.bfloat16)
        for backend in ("eager", "inductor"):
            with self.subTest(backend=backend):
                torch._dynamo.reset()
                opt_fn = torch.compile(fn, backend=backend, fullgraph=True)

                opt_fn(q, k, v)
                torch.cuda.synchronize()
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    with torch.profiler.profile(
                        activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA,
                        ]
                    ) as prof:
                        opt_fn(q, k, v)
                        torch.cuda.synchronize()

                op_names = {evt.key for evt in prof.key_averages()}
                self.assertIn("aten::_flash_attention_forward", op_names)
                self.assertNotIn("aten::_efficient_attention_forward", op_names)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
