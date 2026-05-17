# Owner(s): ["module: intel"]

from collections import namedtuple

import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_nn import NNTestCase
from torch.testing._internal.common_utils import parametrize, run_tests


Tolerances = namedtuple("Tolerances", ["atol", "rtol"])


class TestSDPAXpuOnly(NNTestCase):
    @parametrize("dtype", [torch.half, torch.bfloat16, torch.float32])
    @parametrize("train", [False])
    def test_onednn_attention_unaligned_input(self, device, dtype, train):
        """Verify that Q/K/V and attn_mask whose data pointers are not 64-byte
        aligned are handled correctly by the OneDNN SDPA forward and backward
        kernels.

        We force misalignment on all four tensors by allocating one extra element
        at the front and slicing it off, shifting the storage offset by one
        element (≤ 4 bytes) — always less than 64 bytes, so the base address is
        guaranteed to be non-64-byte-aligned.
        """
        tol = Tolerances(1e-5, 5e-6)
        if dtype is torch.bfloat16:
            tol = Tolerances(5e-2, 5e-2)
        if dtype is torch.float16:
            tol = Tolerances(1e-2, 1e-2)

        batch_size, n_head, q_size, head_dim = 2, 4, 64, 64
        qkv_numel = batch_size * n_head * q_size * head_dim

        def make_unaligned(numel):
            """Return a 1-D tensor of length `numel` whose data_ptr is not
            64-byte aligned, by slicing off a 1-element prefix."""
            t = torch.randn(numel + 1, device=device, dtype=dtype)
            return t[1:]

        # Build misaligned Q, K, V and reshape to (B, H, S, D)
        q = (
            make_unaligned(qkv_numel)
            .view(batch_size, q_size, n_head, head_dim)
            .transpose(1, 2)
        )
        k = (
            make_unaligned(qkv_numel)
            .view(batch_size, q_size, n_head, head_dim)
            .transpose(1, 2)
        )
        v = (
            make_unaligned(qkv_numel)
            .view(batch_size, q_size, n_head, head_dim)
            .transpose(1, 2)
        )

        # Build a misaligned attn_mask of shape (B, 1, 1, S)
        attn_mask = make_unaligned(batch_size * q_size).view(batch_size, 1, 1, q_size)

        # Confirm misalignment so the test is meaningful
        for name, t in [("q", q), ("k", k), ("v", v), ("attn_mask", attn_mask)]:
            self.assertNotEqual(
                t.data_ptr() % 64,
                0,
                msg=f"{name} is unexpectedly 64-byte aligned; "
                "the misalignment premise of this test is invalid",
            )

        q2 = q.clone().float()
        k2 = k.clone().float()
        v2 = v.clone().float()
        attn_mask2 = attn_mask.float()

        if train:
            q = q.detach().requires_grad_(True)
            k = k.detach().requires_grad_(True)
            v = v.detach().requires_grad_(True)
            q2 = q2.detach().requires_grad_(True)
            k2 = k2.detach().requires_grad_(True)
            v2 = v2.detach().requires_grad_(True)

        with sdpa_kernel(backends=[SDPBackend.OVERRIDEABLE]):
            actual = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False
            )

        with sdpa_kernel(backends=[SDPBackend.MATH]):
            math_ref = F.scaled_dot_product_attention(
                q2, k2, v2, attn_mask=attn_mask2, dropout_p=0.0, is_causal=False
            )

        if dtype in [torch.float16, torch.bfloat16]:
            math_ref = math_ref.to(dtype)

        self.assertEqual(actual, math_ref, atol=tol.atol, rtol=tol.rtol)

        if train:
            loss = torch.mean(actual)
            loss_ref = torch.mean(math_ref)
            loss.backward()
            loss_ref.backward()

            grad_q_actual, grad_k_actual, grad_v_actual = q.grad, k.grad, v.grad
            grad_q_ref, grad_k_ref, grad_v_ref = q2.grad, k2.grad, v2.grad
            if dtype in [torch.float16, torch.bfloat16]:
                grad_q_ref = grad_q_ref.to(dtype)
                grad_k_ref = grad_k_ref.to(dtype)
                grad_v_ref = grad_v_ref.to(dtype)

            self.assertEqual(grad_q_actual, grad_q_ref, atol=tol.atol, rtol=tol.rtol)
            self.assertEqual(grad_k_actual, grad_k_ref, atol=tol.atol, rtol=tol.rtol)
            self.assertEqual(grad_v_actual, grad_v_ref, atol=tol.atol, rtol=tol.rtol)


instantiate_device_type_tests(
    TestSDPAXpuOnly, globals(), only_for="xpu", allow_xpu=True
)

if __name__ == "__main__":
    run_tests()
