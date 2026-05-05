# Owner(s): ["module: sdpa"]

import unittest

import torch
from torch.nn.attention.experimental._scaled_dot_product_attention_quantized import (
    _scaled_dot_product_attention_quantized,
    DescaleType,
)
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase


_FP8_MAX = torch.finfo(torch.float8_e4m3fn).max  # 448.0


def _sqnr(signal, noise):
    """Signal-to-quantization-noise ratio in dB."""
    noise = signal.float() - noise.float()
    return 20 * torch.log10(signal.float().norm() / noise.norm()).item()


def _quantize_fp8(t_bf16):
    """Quantize bf16 tensor to FP8 with amax-based scaling."""
    amax = t_bf16.abs().max().item()
    scale = _FP8_MAX / amax
    t_fp8 = (t_bf16 * scale).to(torch.float8_e4m3fn)
    descale = torch.tensor(
        [1.0 / scale], dtype=torch.float32, device=t_bf16.device
    ).view(1, 1, 1, 1)
    return t_fp8, descale


def _fp8_ref_forward(q_fp8, k_fp8, v_fp8, dq, dk, dv, is_causal, scale_s=256.0):
    """Python FP8 reference forward — simulates cuDNN's FP8 quantization pipeline."""
    q_f = q_fp8.float() * dq
    k_f = k_fp8.float() * dk
    v_f = v_fp8.float() * dv

    scale = 1.0 / (q_f.shape[-1] ** 0.5)
    S = torch.einsum("bhsd,bhkd->bhsk", q_f, k_f) * scale

    if is_causal:
        s_q, s_kv = S.shape[-2], S.shape[-1]
        mask = torch.triu(
            torch.ones(s_q, s_kv, device=S.device, dtype=torch.bool), diagonal=1
        )
        S = S.masked_fill(mask, float("-inf"))

    P = S.softmax(dim=-1)

    descale_s = 1.0 / scale_s
    P_fp8 = (P * scale_s).to(torch.float8_e4m3fn)

    O = torch.einsum("bhsk,bhkd->bhsd", P_fp8.float() * descale_s, v_f)
    O_bf16 = O.to(torch.bfloat16)

    return O_bf16


def _cudnn_fp8_available() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability(torch.cuda.current_device())
    if major < 9:  # Hopper+ required
        return False
    return (
        torch.backends.cudnn.is_available() and torch.backends.cudnn.version() >= 90100
    )


def _make_fp8_inputs(batch, heads, seq_len, head_dim, device):
    """Create FP8 Q, K, V and per-tensor descale tensors."""
    shape = (batch, heads, seq_len, head_dim)
    q = torch.randn(shape, dtype=torch.bfloat16, device=device).to(torch.float8_e4m3fn)
    k = torch.randn(shape, dtype=torch.bfloat16, device=device).to(torch.float8_e4m3fn)
    v = torch.randn(shape, dtype=torch.bfloat16, device=device).to(torch.float8_e4m3fn)
    descale_q = torch.ones(1, 1, 1, 1, dtype=torch.float32, device=device)
    descale_k = torch.ones(1, 1, 1, 1, dtype=torch.float32, device=device)
    descale_v = torch.ones(1, 1, 1, 1, dtype=torch.float32, device=device)
    return q, k, v, descale_q, descale_k, descale_v


class TestCuDNNQuantizedAttention(TestCase):
    @unittest.skipUnless(_cudnn_fp8_available(), "cuDNN FP8 SDPA unavailable")
    @parametrize("batch", [1, 2])
    @parametrize("seq_len", [512, 1024])
    @parametrize("heads", [4, 8])
    @parametrize("head_dim", [64, 128])
    @parametrize("is_causal", [False, True])
    def test_forward_runs(self, device, batch, seq_len, heads, head_dim, is_causal):
        """Forward pass smoke test: shape, dtype, no NaN."""
        q, k, v, dq, dk, dv = _make_fp8_inputs(batch, heads, seq_len, head_dim, device)
        with torch.no_grad():
            out = _scaled_dot_product_attention_quantized(
                q,
                k,
                v,
                is_causal=is_causal,
                q_descale=dq,
                k_descale=dk,
                v_descale=dv,
                q_descale_type=DescaleType.PER_TENSOR,
                k_descale_type=DescaleType.PER_TENSOR,
                v_descale_type=DescaleType.PER_TENSOR,
            )
        self.assertEqual(out.shape, (batch, heads, seq_len, head_dim))
        self.assertEqual(out.dtype, torch.bfloat16)
        self.assertFalse(torch.isnan(out).any())
        self.assertTrue(torch.isfinite(out).all())

    @unittest.skipUnless(_cudnn_fp8_available(), "cuDNN FP8 SDPA unavailable")
    @parametrize("is_causal", [False, True])
    def test_forward_compiled(self, device, is_causal):
        """Compiled FP8 forward matches eager."""
        torch.manual_seed(42)
        q, k, v, dq, dk, dv = _make_fp8_inputs(2, 8, 512, 64, device)

        def fp8_sdpa(q, k, v, dq, dk, dv):
            return _scaled_dot_product_attention_quantized(
                q,
                k,
                v,
                is_causal=is_causal,
                q_descale=dq,
                k_descale=dk,
                v_descale=dv,
                q_descale_type=DescaleType.PER_TENSOR,
                k_descale_type=DescaleType.PER_TENSOR,
                v_descale_type=DescaleType.PER_TENSOR,
            )

        compiled_fn = torch.compile(fp8_sdpa, fullgraph=True)

        with torch.no_grad():
            out_eager = fp8_sdpa(q, k, v, dq, dk, dv)
            out_compiled = compiled_fn(q, k, v, dq, dk, dv)

        self.assertEqual(out_eager.shape, out_compiled.shape)
        self.assertEqual(out_eager.stride(), out_compiled.stride())
        self.assertEqual(out_eager.dtype, out_compiled.dtype)
        self.assertTrue(
            torch.allclose(out_eager, out_compiled),
            f"Output max diff: {(out_eager - out_compiled).abs().max().item()}",
        )

    @unittest.skipUnless(_cudnn_fp8_available(), "cuDNN FP8 SDPA unavailable")
    @parametrize("is_causal", [False, True])
    @parametrize("head_dim", [64, 128])
    def test_forward_accuracy(self, device, is_causal, head_dim):
        """FP8 forward: cuDNN kernel vs Python FP8 reference (isolates kernel error)."""
        torch.manual_seed(42)
        B, H, S = 2, 8, 512
        shape = (B, H, S, head_dim)

        q_bf16 = torch.randn(shape, dtype=torch.bfloat16, device=device)
        k_bf16 = torch.randn(shape, dtype=torch.bfloat16, device=device)
        v_bf16 = torch.randn(shape, dtype=torch.bfloat16, device=device)

        q_fp8, dq = _quantize_fp8(q_bf16)
        k_fp8, dk = _quantize_fp8(k_bf16)
        v_fp8, dv = _quantize_fp8(v_bf16)

        with torch.no_grad():
            out_cudnn = _scaled_dot_product_attention_quantized(
                q_fp8,
                k_fp8,
                v_fp8,
                is_causal=is_causal,
                q_descale=dq,
                k_descale=dk,
                v_descale=dv,
                q_descale_type=DescaleType.PER_TENSOR,
                k_descale_type=DescaleType.PER_TENSOR,
                v_descale_type=DescaleType.PER_TENSOR,
            )

        out_pyref = _fp8_ref_forward(q_fp8, k_fp8, v_fp8, dq, dk, dv, is_causal)

        fwd_sqnr = _sqnr(out_pyref, out_cudnn)
        self.assertGreater(
            fwd_sqnr,
            25.0,
            f"Forward SQNR {fwd_sqnr:.1f} dB too low (cuDNN vs FP8 pyref)",
        )


instantiate_device_type_tests(TestCuDNNQuantizedAttention, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
