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

    return O_bf16, P, P_fp8, descale_s


def _fp8_ref_backward(
    q_fp8, k_fp8, v_fp8, dq, dk, dv, P, P_fp8, descale_s, out, grad_out, is_causal
):
    """Python FP8 reference backward — simulates cuDNN's FP8 backward pipeline."""
    q_f = q_fp8.float() * dq
    k_f = k_fp8.float() * dk
    v_f = v_fp8.float() * dv
    scale = 1.0 / (q_f.shape[-1] ** 0.5)

    dO_fp8 = grad_out.to(torch.float8_e4m3fn)
    dO_f = dO_fp8.float()

    dV = torch.einsum("bhsk,bhsd->bhkd", P_fp8.float() * descale_s, dO_f)

    dP = torch.einsum("bhsd,bhkd->bhsk", dO_f, v_f)

    O_f = out.float()
    D = (O_f * dO_f).sum(dim=-1, keepdim=True)
    dS = P * (dP - D) * scale

    dS_fp8 = dS.to(torch.float8_e4m3fn)
    dS_f = dS_fp8.float()

    dQ = torch.einsum("bhsk,bhkd->bhsd", dS_f, k_f)
    dK = torch.einsum("bhsk,bhsd->bhkd", dS_f, q_f)

    return (
        dQ.to(torch.float8_e4m3fn),
        dK.to(torch.float8_e4m3fn),
        dV.to(torch.float8_e4m3fn),
    )


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


def _requantize_output(output_bf16, amax_o, scale_s=256.0):
    """Requantize BF16 output to FP8 and compute descales for backward pass.

    This simulates what a training framework (e.g., torchao) would do between
    forward and backward: convert the BF16 output to FP8 using the amax from
    the forward pass, and compute the descale factors needed by the backward op.
    """
    scale_o = _FP8_MAX / amax_o.clamp(min=1e-12)
    fp8_o = (output_bf16.float() * scale_o).to(torch.float8_e4m3fn)
    descale_o = (amax_o / _FP8_MAX).view(1, 1, 1, 1).to(torch.float32)
    descale_s = torch.tensor(
        [1.0 / scale_s], dtype=torch.float32, device=output_bf16.device
    ).view(1, 1, 1, 1)
    return fp8_o, descale_o, descale_s


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
    @parametrize("batch", [1, 2])
    @parametrize("seq_len", [512, 1024])
    @parametrize("heads", [4, 8])
    @parametrize("head_dim", [64, 128])
    @parametrize("is_causal", [False, True])
    def test_backward_runs(self, device, batch, seq_len, heads, head_dim, is_causal):
        """Backward op smoke test: gradient shape, dtype, no NaN."""
        q, k, v, dq, dk, dv = _make_fp8_inputs(batch, heads, seq_len, head_dim, device)
        scale_s = 256.0

        # Forward pass via raw op
        output, softmax_stats, amax_s, amax_o = (
            torch.ops.aten._scaled_dot_product_cudnn_attention_quantized_per_tensor(
                q,
                k,
                v,
                dq,
                dk,
                dv,
                scale_s,
                is_causal,
            )
        )

        # Requantize output (simulates torchao custom autograd Function)
        fp8_o, descale_o, descale_s = _requantize_output(output, amax_o, scale_s)

        grad_out = torch.ones_like(output)

        # Backward pass via raw op
        dQ, dK, dV = (
            torch.ops.aten._scaled_dot_product_cudnn_attention_backward_quantized_per_tensor(
                grad_out,
                q,
                k,
                v,
                fp8_o,
                softmax_stats,
                dq,
                dk,
                dv,
                descale_o,
                descale_s,
                scale_s,
                is_causal,
            )
        )

        for name, grad in [("q", dQ), ("k", dK), ("v", dV)]:
            self.assertEqual(grad.shape, q.shape if name == "q" else k.shape)
            self.assertEqual(grad.dtype, torch.float8_e4m3fn)
            grad_f32 = grad.to(torch.float32)
            self.assertFalse(torch.isnan(grad_f32).any(), f"{name}.grad has NaN")
            self.assertTrue(torch.isfinite(grad_f32).all(), f"{name}.grad has Inf")

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

        out_pyref, _, _, _ = _fp8_ref_forward(
            q_fp8, k_fp8, v_fp8, dq, dk, dv, is_causal
        )

        fwd_sqnr = _sqnr(out_pyref, out_cudnn)
        self.assertGreater(
            fwd_sqnr,
            25.0,
            f"Forward SQNR {fwd_sqnr:.1f} dB too low (cuDNN vs FP8 pyref)",
        )

    @unittest.skipUnless(_cudnn_fp8_available(), "cuDNN FP8 SDPA unavailable")
    @parametrize("is_causal", [False, True])
    @parametrize("head_dim", [64, 128])
    def test_backward_accuracy(self, device, is_causal, head_dim):
        """FP8 backward: cuDNN kernel vs Python FP8 reference (isolates kernel error)."""
        torch.manual_seed(42)
        B, H, S = 2, 8, 512
        shape = (B, H, S, head_dim)

        q_bf16 = torch.randn(shape, dtype=torch.bfloat16, device=device)
        k_bf16 = torch.randn(shape, dtype=torch.bfloat16, device=device)
        v_bf16 = torch.randn(shape, dtype=torch.bfloat16, device=device)

        q_fp8, dq = _quantize_fp8(q_bf16)
        k_fp8, dk = _quantize_fp8(k_bf16)
        v_fp8, dv = _quantize_fp8(v_bf16)
        scale_s = 256.0

        # cuDNN forward + backward via raw ops
        output, softmax_stats, amax_s, amax_o = (
            torch.ops.aten._scaled_dot_product_cudnn_attention_quantized_per_tensor(
                q_fp8,
                k_fp8,
                v_fp8,
                dq,
                dk,
                dv,
                scale_s,
                is_causal,
            )
        )
        fp8_o, descale_o, descale_s = _requantize_output(output, amax_o, scale_s)
        grad_out = torch.ones_like(output)
        dQ_cudnn, dK_cudnn, dV_cudnn = (
            torch.ops.aten._scaled_dot_product_cudnn_attention_backward_quantized_per_tensor(
                grad_out,
                q_fp8,
                k_fp8,
                v_fp8,
                fp8_o,
                softmax_stats,
                dq,
                dk,
                dv,
                descale_o,
                descale_s,
                scale_s,
                is_causal,
            )
        )

        # Python FP8 reference
        out_pyref, P, P_fp8, pyref_descale_s = _fp8_ref_forward(
            q_fp8, k_fp8, v_fp8, dq, dk, dv, is_causal
        )
        dQ_ref, dK_ref, dV_ref = _fp8_ref_backward(
            q_fp8,
            k_fp8,
            v_fp8,
            dq,
            dk,
            dv,
            P,
            P_fp8,
            pyref_descale_s,
            out_pyref,
            grad_out,
            is_causal,
        )

        for name, g_cudnn, g_ref in zip(
            ["q", "k", "v"],
            [dQ_cudnn, dK_cudnn, dV_cudnn],
            [dQ_ref, dK_ref, dV_ref],
        ):
            bwd_sqnr = _sqnr(g_ref, g_cudnn)
            self.assertGreater(
                bwd_sqnr,
                25.0,
                f"{name}.grad SQNR {bwd_sqnr:.1f} dB too low (cuDNN vs FP8 pyref)",
            )


instantiate_device_type_tests(TestCuDNNQuantizedAttention, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
