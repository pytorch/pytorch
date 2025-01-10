# Owner(s): ["module: intel"]
from collections import namedtuple
from functools import partial

import torch
import torch.nn.functional as F
import torch.utils.cpp_extension
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_nn import NNTestCase
from torch.testing._internal.common_utils import (
    parametrize,
    run_tests,
    skipIfTorchDynamo,
)

SdpaShape = namedtuple('Sdpa_Shape', ['batch', 'num_heads', 'seq_len', 'head_dim'])
Tolerances = namedtuple('Tolerances', ['atol', 'rtol'])


# Found in torch/testing/_comparison.py
default_atol = {torch.float16: 1e-3, torch.bfloat16: 1e-3, torch.float32: 1e-5}
default_rtol = {torch.float16: 1e-3, torch.bfloat16: 1.6e-2, torch.float32: 1.3e-6}


def rand_sdpa_tensor(shape: SdpaShape, device: str, dtype: torch.dtype, type: str,
                     requires_grad: bool = False, packed: bool = False) -> torch.Tensor:
    """Creates rand dense or nested tensor with given shape and type.

    Args:
        shape (Tuple[int]): Shape of Tensor to construct
        device (str): which device to create tensor on
        dtype (torch.dtype): Tensors' dtype
        type (str): Nested or Dense
        requires_grad (bool, optional): Tensors grad status. Defaults to False.
        packed (bool, optional): Whether to create a single QKV packed or not. Defaults to False.

    Returns:
        torch.Tensor: A new tensor
    """
    batch, num_heads, seq_len, head_dim = shape.batch, shape.num_heads, shape.seq_len, shape.head_dim
    if type == "nested":
        if isinstance(seq_len, list):
            def _size(i):
                return (seq_len[i], num_heads, head_dim) if not packed else (seq_len[i], 3 * num_heads * head_dim)

            return torch.nested.nested_tensor([
                torch.randn(_size(i), device=device, dtype=dtype, requires_grad=requires_grad)
                for i in range(batch)])
        else:
            size = (seq_len, num_heads, head_dim) if not packed else (seq_len, 3 * num_heads * head_dim)
            return torch.nested.nested_tensor([
                torch.randn(size, device=device, dtype=dtype, requires_grad=requires_grad)
                for _ in range(batch)])
    else:
        assert (isinstance(seq_len, int))
        size = (batch, seq_len, num_heads, head_dim) if not packed else (batch, seq_len, 3 * num_heads * head_dim)
        return torch.randn(size, device=device, dtype=dtype, requires_grad=requires_grad)


class TestSDPAXpuOnly(NNTestCase):
    """ Used to test XPU only functionality of scaled_dot_product_attention
    Mostly migrate from TestSDPACudaOnly in test/test_transformers.py

    Note that as SDPBackend.OVERRIDEABLE is not managed by sdpa_kernel so that
    math ref has to be called explicitly via torch.ops.aten._scaled_dot_product_attention_math.
    """

    @parametrize("type", ["dense"])
    @parametrize("dropout", [0.0, 0.7])
    @parametrize("dtype", [torch.float64, torch.float32, torch.bfloat16, torch.half])
    @skipIfTorchDynamo()
    def test_fused_sdp_choice_xpu(self, device, type: str, dropout: float, dtype: torch.dtype):
        # Migrate from test_fused_sdp_choice_cpu
        make_tensor = partial(rand_sdpa_tensor, type=type, device=device, dtype=dtype)
        size = SdpaShape(2, 8, 128, 64)
        q, k, v = make_tensor(size), make_tensor(size), make_tensor(size)
        if dropout > 0.0 or dtype not in [torch.float32, torch.bfloat16, torch.float16]:
            assert torch._fused_sdp_choice(q, k, v, dropout_p=dropout) == SDPBackend.MATH.value
        else:
            assert torch._fused_sdp_choice(q, k, v, dropout_p=dropout) == SDPBackend.OVERRIDEABLE.value

    def test_fused_attention_different_dk_dv(self, device):
        dtype = torch.bfloat16
        make_tensor = partial(torch.rand, device=device, dtype=dtype, requires_grad=True)
        batch, num_heads, head_dim_k, head_dim_v = 32, 16, 128, 64
        q_shape = SdpaShape(batch, num_heads, 1, head_dim_k)
        k_shape = SdpaShape(batch, num_heads, 2, head_dim_k)
        v_shape = SdpaShape(batch, num_heads, 2, head_dim_v)
        query, key, value = make_tensor(q_shape), make_tensor(k_shape), make_tensor(v_shape)

        # test that we do not dispatch to onednn for an unsupported case
        actual = F.scaled_dot_product_attention(
            query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False)

        math_ref = torch.ops.aten._scaled_dot_product_attention_math(
            query.contiguous().float(), key.contiguous().float(), value.contiguous().float(), attn_mask=None, dropout_p=0.0, is_causal=False)[0]

        self.assertEqual(actual.contiguous(), math_ref.contiguous().to(dtype), atol=1e-3, rtol=1e-2)

    def test_onednn_attention_fail_d256(self, device):
        # Test that onednn graph attention dispatching correctly bails out on d > 256
        b, h = 1, 2
        s_q, s_kv = 128, 128
        d_qk, d_v = 512, 512

        q = torch.randn(b, h, s_q, d_qk, device=device, dtype=torch.bfloat16)
        k = torch.randn(b, h, s_kv, d_qk, device=device, dtype=torch.bfloat16)
        v = torch.randn(b, h, s_kv, d_v, device=device, dtype=torch.bfloat16)

        with sdpa_kernel(backends=[SDPBackend.OVERRIDEABLE]):
            with self.assertRaisesRegex(RuntimeError, "No available kernel."):
                o = F.scaled_dot_product_attention(q, k, v)

    @parametrize("type", ["dense"])
    @parametrize("is_contiguous", [True, False])
    def test_scaled_dot_product_attention_fused_kernels_packed(self, device, type: str, is_contiguous: bool):
        make_tensor = partial(rand_sdpa_tensor, type=type, device=device, dtype=torch.float16, packed=True)

        batch_size, seq_len, num_heads, head_dim = 32, 64, 16, 64
        shape = SdpaShape(batch_size, num_heads, seq_len, head_dim)

        # Test Packed
        qkv = make_tensor(shape)
        query, key, value = qkv.chunk(3, dim=-1)

        query = query.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)

        if is_contiguous:
            query = query.contiguous()
            key = key.contiguous()
            value = value.contiguous()

        with sdpa_kernel(backends=[SDPBackend.OVERRIDEABLE]):
            actual = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False)
        math_ref = torch.ops.aten._scaled_dot_product_attention_math(
            query.contiguous(), key.contiguous(), value.contiguous(), attn_mask=None, dropout_p=0.0, is_causal=False)[0]

        self.assertEqual(actual.contiguous(), math_ref.contiguous(), atol=2e-3, rtol=1e-2)

    @parametrize("fused_kernel", [SDPBackend.MATH, SDPBackend.OVERRIDEABLE])
    @parametrize("dtype", [torch.half, torch.bfloat16, torch.float32])
    @parametrize("batch_size,n_head,q_size,kv_size,head_dim", [
        (2, 5, 9216, 9216, 64),
        (2, 5, 9216, 77, 64),
        (2, 10, 2304, 2304, 64),
        (2, 10, 2304, 77, 64),
        (2, 20, 576, 576, 64),
        (2, 20, 576, 77, 64),
        (2, 20, 144, 144, 64),
        (2, 20, 144, 77, 64),
        (1, 32, 1, 32, 128),
        (4, 32, 1, 32, 128),
        (1, 32, 32, 32, 128),
        (4, 32, 32, 32, 128),
        (1, 32, 2016, 2016, 128),
        (4, 32, 2016, 2016, 128),
    ])
    @parametrize("mask_type", ["float"])
    @parametrize("train", [False])
    def test_scaled_dot_product_fused_attention_mask_vs_math(
        self,
        device,
        fused_kernel,
        dtype,
        batch_size,
        q_size,
        kv_size,
        n_head,
        head_dim,
        mask_type,
        train,
    ):
        # Migrate from TestSDPACpuOnly
        tol = Tolerances(1e-5, 5e-6)
        if dtype is torch.bfloat16:
            tol = Tolerances(5e-2, 5e-2)
        if dtype is torch.float16:
            tol = Tolerances(1e-2, 1e-2)
        mask_shape = [batch_size, 1, 1, kv_size]
        make_tensor = partial(rand_sdpa_tensor, type="dense", device=device, dtype=dtype, requires_grad=False)
        q_shape = SdpaShape(batch_size, n_head, q_size, head_dim)
        kv_shape = SdpaShape(batch_size, n_head, kv_size, head_dim)
        q = make_tensor(q_shape)
        k = make_tensor(kv_shape)
        v = make_tensor(kv_shape)
        q2, k2, v2 = q.clone(), k.clone(), v.clone()

        if train:
            q.requires_grad_(True)
            k.requires_grad_(True)
            v.requires_grad_(True)
            q2.requires_grad_(True)
            k2.requires_grad_(True)
            v2.requires_grad_(True)

        q2, k2, v2 = q2.float(), k2.float(), v2.float()
        # (B, nh, T, hs)
        q = q.view(batch_size, q_size, n_head, head_dim).transpose(1, 2)
        k = k.view(batch_size, kv_size, n_head, head_dim).transpose(1, 2)
        v = v.view(batch_size, kv_size, n_head, head_dim).transpose(1, 2)
        attn_mask = None
        if mask_type == "bool":
            attn_mask = torch.randint(0, 2, size=mask_shape, dtype=torch.bool, device=device)
        elif mask_type == "float":
            attn_mask = torch.randn(mask_shape, dtype=dtype, device=device)

        q2 = q2.view(batch_size, q_size, n_head, head_dim).transpose(1, 2)
        k2 = k2.view(batch_size, kv_size, n_head, head_dim).transpose(1, 2)
        v2 = v2.view(batch_size, kv_size, n_head, head_dim).transpose(1, 2)

        if fused_kernel == SDPBackend.MATH:
            actual = torch.ops.aten._scaled_dot_product_attention_math(
                q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)[0]
        elif fused_kernel == SDPBackend.OVERRIDEABLE:
            actual = torch.ops.aten._scaled_dot_product_fused_attention_overrideable(
                q, k, v, attn_bias=attn_mask, dropout_p=0.0, is_causal=False)[0]

        math_ref = torch.ops.aten._scaled_dot_product_attention_math(
            q2, k2, v2, attn_mask=attn_mask.float(), dropout_p=0.0, is_causal=False)[0]

        self.assertEqual(actual.float(), math_ref, atol=tol.atol, rtol=tol.rtol)


instantiate_device_type_tests(
    TestSDPAXpuOnly, globals(), only_for="xpu", allow_xpu=True
)

if __name__ == "__main__":
    run_tests()
