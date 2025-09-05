# Owner(s): ["module: sdpa"]

import unittest
from collections import namedtuple
from functools import partial

import torch_openreg  # noqa: F401

import torch
from torch.nn.attention import SDPBackend
from torch.testing._internal.common_nn import NNTestCase
from torch.testing._internal.common_utils import run_tests, skipIfTorchDynamo, TEST_XPU


SdpaShape = namedtuple("Sdpa_Shape", ["batch", "num_heads", "seq_len", "head_dim"])


@unittest.skipIf(TEST_XPU, "XPU does not support cppextension currently")
class TestSDPAPrivateUse1Only(NNTestCase):
    @skipIfTorchDynamo()
    def test_fused_sdp_choice_privateuseone(self):
        batch_size, seq_len, num_heads, head_dim = 4, 256, 2, 128
        make_tensor = partial(torch.rand, device="cpu", dtype=torch.float16)
        shape = SdpaShape(batch_size, num_heads, seq_len, head_dim)
        q_cpu, k_cpu, v_cpu = make_tensor(shape), make_tensor(shape), make_tensor(shape)
        q_privateuse1 = q_cpu.to("openreg")
        k_privateuse1 = k_cpu.to("openreg")
        v_privateuse1 = v_cpu.to("openreg")
        assert (
            torch._fused_sdp_choice(q_privateuse1, k_privateuse1, v_privateuse1)
            == SDPBackend.OVERRIDEABLE.value
        )

    def test_scaled_dot_product_fused_attention_overrideable(self):
        batch_size, seq_len, num_heads, head_dim = 4, 256, 2, 128
        make_tensor = partial(torch.rand, device="cpu", dtype=torch.float16)
        shape = SdpaShape(batch_size, num_heads, seq_len, head_dim)
        q_cpu, k_cpu, v_cpu = make_tensor(shape), make_tensor(shape), make_tensor(shape)
        q_privateuse1 = q_cpu.to("openreg")
        k_privateuse1 = k_cpu.to("openreg")
        v_privateuse1 = v_cpu.to("openreg")
        torch.nn.functional.scaled_dot_product_attention(
            q_privateuse1, k_privateuse1, v_privateuse1, attn_mask=None, dropout_p=0.0
        )

    def test_scaled_dot_product_fused_attention_overrideable_backward(self):
        batch_size, seq_len, num_heads, head_dim = 4, 256, 2, 128
        make_tensor = partial(
            torch.rand, device="cpu", dtype=torch.float16, requires_grad=True
        )
        shape = (batch_size, num_heads, seq_len, head_dim)
        q_cpu, k_cpu, v_cpu = make_tensor(shape), make_tensor(shape), make_tensor(shape)
        attn_mask = make_tensor((batch_size, num_heads, seq_len, seq_len))
        q_privateuse1 = q_cpu.to("openreg")
        k_privateuse1 = k_cpu.to("openreg")
        v_privateuse1 = v_cpu.to("openreg")
        attn_mask_privateuse1 = attn_mask.to("openreg")
        (
            output,
            logsumexp,
            cum_seq_q,
            cum_seq_k,
            max_q,
            max_k,
            philox_seed,
            philox_offset,
            debug_attn_mask,
        ) = torch.ops.aten._scaled_dot_product_fused_attention_overrideable(
            q_privateuse1, k_privateuse1, v_privateuse1, attn_bias=attn_mask_privateuse1
        )

        rand_upward = torch.rand(
            shape, device="cpu", dtype=torch.float16, requires_grad=False
        )
        rand_upward_privateuse1 = rand_upward.to("openreg")
        grad_input_mask = [True, True, True, True]
        grad_q, grad_k, grad_v, grad_attn_mask = (
            torch.ops.aten._scaled_dot_product_fused_attention_overrideable_backward(
                rand_upward_privateuse1,
                q_privateuse1,
                k_privateuse1,
                v_privateuse1,
                attn_mask_privateuse1,
                grad_input_mask,
                output,
                logsumexp,
                cum_seq_q,
                cum_seq_k,
                max_q,
                max_k,
                dropout_p=0.0,
                is_causal=False,
                philox_seed=philox_seed,
                philox_offset=philox_offset,
            )
        )


if __name__ == "__main__":
    run_tests()
