# Owner(s): ["module: dynamo"]
import contextlib
import unittest

import torch
import torch._dynamo.test_case
from torch._dynamo.testing import CompileCounter
from torch.backends.cuda import SDPAParams
from torch.nn import functional as F
from torch.nn.attention import _cur_sdpa_kernel_backends, sdpa_kernel, SDPBackend
from torch.nn.attention.flex_attention import (
    AuxRequest,
    create_block_mask,
    flex_attention,
)
from torch.testing._internal.common_cuda import PLATFORM_SUPPORTS_FLASH_ATTENTION
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import skipIfHpu, TEST_WITH_ROCM


requires_cuda = unittest.skipUnless(torch.cuda.is_available(), "requires cuda")


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

    @requires_cuda
    @unittest.skipIf(
        TEST_WITH_ROCM or not PLATFORM_SUPPORTS_FLASH_ATTENTION,
        "flash attention not supported",
    )
    def test_flex_attention_guard_on_constant_func_defaults(self):
        """
        Dynamo must guard on mask_mod.__defaults__ so that when a
        compiled function is re-invoked with a new BlockMask whose
        mask_mod has the same __code__ but different __defaults__,
        Dynamo recompiles instead of reusing the stale first graph.
        """
        from torch.utils._triton import has_triton

        if not has_triton():
            self.skipTest("requires triton")

        @torch.compile(fullgraph=True)
        def flex_chunk(q, k, v, block_mask, scale):
            out, aux = flex_attention(
                q,
                k,
                v,
                block_mask=block_mask,
                scale=scale,
                return_aux=AuxRequest(lse=True),
            )
            return out, aux.lse

        def merge(out, lse, new_out, new_lse):
            lse, new_lse = lse.unsqueeze(-1), new_lse.unsqueeze(-1)
            mx = torch.maximum(lse, new_lse)
            e0, e1 = torch.exp(lse - mx), torch.exp(new_lse - mx)
            d = e0 + e1
            return (out * e0 + new_out * e1) / d, (mx + torch.log(d)).squeeze(-1)

        @torch.compile(fullgraph=True)
        def ref_attn(q, k, v, block_mask, scale):
            return flex_attention(q, k, v, block_mask=block_mask, scale=scale)

        torch.manual_seed(42)
        B, H, S, D = 1, 1, 512, 16
        device = "cuda"
        num_chunks = 4
        chunk_size = S // num_chunks

        q = torch.randn(B, H, S, D, device=device)
        k = torch.randn(B, H, S, D, device=device)
        v = torch.randn(B, H, S, D, device=device)
        scale = D**-0.5

        merged_out = merged_lse = None
        for step in range(num_chunks):
            kv_offset = step * chunk_size

            def mask_mod(b, h, q_idx, kv_idx, _offset=kv_offset):
                return q_idx >= kv_idx + _offset

            bm = create_block_mask(
                mask_mod, B=B, H=H, Q_LEN=S, KV_LEN=chunk_size, device=device
            )
            out, lse = flex_chunk(
                q,
                k[:, :, kv_offset : kv_offset + chunk_size],
                v[:, :, kv_offset : kv_offset + chunk_size],
                bm,
                scale,
            )
            if merged_out is None:
                merged_out, merged_lse = out, lse
            else:
                merged_out, merged_lse = merge(merged_out, merged_lse, out, lse)

        def causal(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx

        ref_bm = create_block_mask(causal, B=B, H=H, Q_LEN=S, KV_LEN=S, device=device)
        ref_out = ref_attn(q, k, v, ref_bm, scale)

        self.assertTrue(
            (merged_out - ref_out).abs().max().item() < 1e-3,
            "flex_attention mask_mod __defaults__ not properly guarded",
        )


class TestSDPADevice(torch._dynamo.test_case.TestCase):
    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FLASH_ATTENTION,
        "Can't run fused SDPA on this platform",
    )
    def test_parsing_sdpa(self, device):
        class MyModule(torch.nn.Module):
            def forward(self, query, key, value):
                out = F.scaled_dot_product_attention(query, key, value, None, 0, True)
                out = F.scaled_dot_product_attention(
                    query, key, value, None, 0, True, scale=8
                )
                out = F.scaled_dot_product_attention(
                    query=query,
                    key=key,
                    value=value,
                    attn_mask=None,
                    dropout_p=0,
                    is_causal=True,
                )
                out = F.scaled_dot_product_attention(
                    query,
                    key=key,
                    value=value,
                    attn_mask=None,
                    dropout_p=0,
                    is_causal=True,
                )
                out = F.scaled_dot_product_attention(
                    query, key, value, None, dropout_p=0, is_causal=True
                )
                out = F.scaled_dot_product_attention(query, key, value, None, scale=8)
                return out

        device = device  # noqa: PLW0127
        dtype = torch.float16
        seq_len_q = 1
        seq_len_k = 1
        head_dim = 8
        query = torch.ones(
            1, 8, seq_len_q, head_dim, device=device, dtype=dtype, requires_grad=True
        )
        key = torch.ones(
            1, 8, seq_len_k, head_dim, device=device, dtype=dtype, requires_grad=True
        )
        value = torch.ones(
            1, 8, seq_len_k, head_dim, device=device, dtype=dtype, requires_grad=True
        )
        module = MyModule()
        opt_mod = torch.compile(module, backend="inductor")
        opt_mod(query, key, value)

    @skipIfHpu
    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FLASH_ATTENTION,
        "flash attention not supported",
    )
    def test_flash_attn_backward_mixed_strides(self, device):
        def gen_inputs(device):
            return (
                torch.randn(
                    2, 513, 16, 64, dtype=torch.float16, device=device
                ).transpose(1, 2),
                torch.randn(2, 16, 513, 64, dtype=torch.float16, device=device),
                torch.randn(2, 16, 513, 64, dtype=torch.float16, device=device),
                torch.randn(
                    2, 513, 16, 64, dtype=torch.float16, device=device
                ).transpose(1, 2),
                torch.randn(2, 16, 513, 64, dtype=torch.float16, device=device),
                torch.randn(2, 16, 513, device=device),
                None,
                None,
                513,
                513,
                0.0,
                False,
                torch.tensor(1, dtype=torch.int64),
                torch.tensor(1, dtype=torch.int64),
            )

        inps_device = gen_inputs(device)
        inps_meta = gen_inputs("meta")
        (
            out1_ref,
            out2_ref,
            out3_ref,
        ) = torch.ops.aten._scaled_dot_product_flash_attention_backward(
            *inps_device, scale=0.125
        )
        from torch._meta_registrations import meta__scaled_dot_product_flash_backward

        out1_test, out2_test, out3_test = meta__scaled_dot_product_flash_backward(
            *inps_meta, scale=0.125
        )

        self.assertEqual(out1_ref.shape, out1_test.shape)
        self.assertEqual(out1_ref.stride(), out1_test.stride())
        self.assertEqual(out2_ref.shape, out2_test.shape)
        self.assertEqual(out2_ref.stride(), out2_test.stride())
        self.assertEqual(out3_ref.shape, out3_test.shape)
        self.assertEqual(out3_ref.stride(), out3_test.stride())

    @requires_cuda
    def test_sdpa_dynamic_shapes(self, device):
        def f(x, s0, s1, s2):
            q = x.view(2, s0, s2, s0)
            return torch._C._nn.scaled_dot_product_attention(
                q, q, q, attn_mask=None, dropout_p=0.0, is_causal=True
            )

        x = torch.randn(2, 32, 4096, dtype=torch.bfloat16, device=device)
        x_ref = x.clone().detach().requires_grad_()
        s0 = 32
        s1 = 64
        s2 = 128

        f_compiled = torch.compile(f, dynamic=True, backend="eager")

        with torch._dynamo.config.patch(assume_static_by_default=False):
            out_ref = f(x_ref, s0, s1, s2)
            out = f_compiled(x, s0, s1, s2)
            self.assertEqual(out_ref, out)


instantiate_device_type_tests(TestSDPADevice, globals(), only_for=("cuda", "hpu"))

if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
