# Owner(s): ["module: sdpa"]
import unittest
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.varlen import varlen_attn
from torch.testing._internal.common_cuda import PLATFORM_SUPPORTS_FLASH_ATTENTION
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_nn import NNTestCase
from torch.testing._internal.common_utils import parametrize, run_tests, skipIfRocm
from torch.utils._python_dispatch import TorchDispatchMode


VarlenShape = namedtuple(
    "VarlenShape", ["batch_size", "max_seq_len", "embed_dim", "num_heads"]
)


class OpLoggingMode(TorchDispatchMode):
    """Logging mode that captures all dispatched operations"""

    def __init__(self):
        self.called_ops = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        op_name = str(func)
        self.called_ops.append(op_name)
        return func(*args, **(kwargs or {}))


class AttentionBlock(nn.Module):
    def __init__(
        self, embed_dim: int, num_heads: int, device: torch.device, dtype: torch.dtype
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv_proj = nn.Linear(
            embed_dim, 3 * embed_dim, bias=False, device=device, dtype=dtype
        )
        self.out_proj = nn.Linear(
            embed_dim, embed_dim, bias=False, device=device, dtype=dtype
        )

    def get_varlen_qkv(
        self,
        x_packed: torch.Tensor,
    ):
        qkv = self.qkv_proj(x_packed)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_heads, self.head_dim)
        v = v.view(-1, self.num_heads, self.head_dim)

        return q, k, v

    def forward_varlen(
        self,
        x_packed: torch.Tensor,
        cu_seq: torch.Tensor,
        max_len: int,
        is_causal: bool = False,
    ):
        q, k, v = self.get_varlen_qkv(x_packed)

        attn_out = varlen_attn(q, k, v, cu_seq, cu_seq, max_len, max_len, is_causal)
        attn_out = attn_out.view(-1, self.embed_dim)

        return self.out_proj(attn_out)

    def forward_sdpa(
        self,
        x_padded: torch.Tensor,
        seq_lengths: torch.Tensor,
        is_causal: bool = False,
    ):
        batch_size, seq_len, _ = x_padded.shape

        qkv = self.qkv_proj(x_padded)
        q, k, v = qkv.chunk(3, dim=-1)

        mask = (
            torch.arange(seq_len, device=x_padded.device)[None, :]
            < seq_lengths[:, None]
        )

        attn_mask = mask[:, None, None, :].expand(
            batch_size, self.num_heads, seq_len, seq_len
        )

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=is_causal
        )

        attn_out = (
            attn_out.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.embed_dim)
        )

        return self.out_proj(attn_out)


def create_variable_length_batch(
    shape: VarlenShape, device: torch.device, dtype: torch.dtype
):
    seq_lengths = []
    for _ in range(shape.batch_size):
        length = torch.randint(1, shape.max_seq_len // 64 + 1, (1,)).item() * 64
        seq_lengths.append(min(length, shape.max_seq_len))

    seq_lengths = torch.tensor(seq_lengths, device=device)
    total_tokens = seq_lengths.sum().item()

    x_packed = torch.randn(
        total_tokens, shape.embed_dim, device=device, dtype=dtype, requires_grad=True
    )

    cu_seq = torch.zeros(shape.batch_size + 1, device=device, dtype=torch.int32)
    cu_seq[1:] = seq_lengths.cumsum(0)

    max_len = seq_lengths.max().item()
    x_padded = torch.zeros(
        shape.batch_size, max_len, shape.embed_dim, device=device, dtype=dtype
    )

    start_idx = 0
    for i, seq_len in enumerate(seq_lengths):
        end_idx = start_idx + seq_len
        x_padded[i, :seq_len] = x_packed[start_idx:end_idx]
        start_idx = end_idx
    x_padded = x_padded.clone().detach().requires_grad_()

    return {
        "seq_lengths": seq_lengths,
        "cu_seq": cu_seq,
        "x_packed": x_packed,
        "x_padded": x_padded,
        "max_len": max_len,
        "total_tokens": total_tokens,
    }


class TestVarlenAttention(NNTestCase):
    @skipIfRocm(msg="ROCM does not support variable length attention")
    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FLASH_ATTENTION, "Flash Attention not supported"
    )
    @parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_basic_functionality(self, device, dtype):
        torch.manual_seed(42)

        shape = VarlenShape(batch_size=2, max_seq_len=512, embed_dim=1024, num_heads=16)

        attention_block = AttentionBlock(
            shape.embed_dim, shape.num_heads, device, dtype
        )

        total_tokens = shape.batch_size * shape.max_seq_len
        x_packed = torch.randn(
            total_tokens,
            shape.embed_dim,
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        cu_seq = torch.tensor(
            [0, shape.max_seq_len, total_tokens], device=device, dtype=torch.int32
        )

        output = attention_block.forward_varlen(
            x_packed, cu_seq, shape.max_seq_len, is_causal=False
        )

        self.assertEqual(output.shape, (total_tokens, shape.embed_dim))
        self.assertEqual(output.device, torch.device(device))
        self.assertEqual(output.dtype, dtype)

        varlen_grad_out = torch.ones_like(output)

        varlen_grad = torch.autograd.grad(
            outputs=output,
            inputs=x_packed,
            grad_outputs=varlen_grad_out,
            retain_graph=True,
            create_graph=False,
            allow_unused=False,
        )[0]

        self.assertIsNotNone(varlen_grad)
        self.assertEqual(varlen_grad.shape, x_packed.shape)
        self.assertEqual(varlen_grad.dtype, x_packed.dtype)

    @skipIfRocm(msg="ROCM does not support variable length attention")
    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FLASH_ATTENTION, "Flash Attention not supported"
    )
    @parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_custom_op_compliance(self, device, dtype):
        torch.manual_seed(42)

        shape = VarlenShape(batch_size=2, max_seq_len=512, embed_dim=1024, num_heads=16)

        attention_block = AttentionBlock(
            shape.embed_dim, shape.num_heads, device, dtype
        )

        total_tokens = shape.batch_size * shape.max_seq_len
        x_packed = torch.randn(
            total_tokens,
            shape.embed_dim,
            device=device,
            dtype=dtype,
        )
        cu_seq = torch.tensor(
            [0, shape.max_seq_len, total_tokens], device=device, dtype=torch.int32
        )

        q, k, v = attention_block.get_varlen_qkv(x_packed)

        torch.library.opcheck(
            torch.ops.torch_attn._varlen_attn,
            (q, k, v, cu_seq, cu_seq, shape.max_seq_len, shape.max_seq_len, False),
        )

        out, lse, rng_state = torch.ops.torch_attn._varlen_attn(
            q, k, v, cu_seq, cu_seq, shape.max_seq_len, shape.max_seq_len, False
        )
        grad_out = torch.randn_like(out)

        # we don't support double backward
        # skipping test_autograd_registration, test_aot_dispatch_dynamic, test_aot_dispatch_static
        torch.library.opcheck(
            torch.ops.torch_attn._varlen_attn_backward,
            (
                grad_out,
                q,
                k,
                v,
                out,
                lse,
                cu_seq,
                cu_seq,
                shape.max_seq_len,
                shape.max_seq_len,
                False,
                rng_state,
            ),
            test_utils=["test_schema", "test_faketensor"],
        )

    @skipIfRocm(msg="ROCM does not support variable length attention")
    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FLASH_ATTENTION, "Flash Attention not supported"
    )
    @parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_custom_op_registration(self, device, dtype):
        torch.manual_seed(42)

        shape = VarlenShape(batch_size=2, max_seq_len=512, embed_dim=1024, num_heads=16)

        attention_block = AttentionBlock(
            shape.embed_dim, shape.num_heads, device, dtype
        )

        total_tokens = shape.batch_size * shape.max_seq_len
        x_packed = torch.randn(
            total_tokens,
            shape.embed_dim,
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        cu_seq = torch.tensor(
            [0, shape.max_seq_len, total_tokens], device=device, dtype=torch.int32
        )

        compiled_forward = torch.compile(
            attention_block.forward_varlen, backend="eager", fullgraph=True
        )
        with OpLoggingMode() as mode:
            output = compiled_forward(
                x_packed, cu_seq, shape.max_seq_len, is_causal=False
            )

            varlen_grad_out = torch.ones_like(output)
            _ = torch.autograd.grad(
                outputs=output,
                inputs=x_packed,
                grad_outputs=varlen_grad_out,
                retain_graph=True,
                create_graph=False,
                allow_unused=False,
            )[0]

        called_ops = mode.called_ops

        custom_ops_called = any(
            "torch_attn._varlen_attn" in op for op in called_ops
        ) and any("torch_attn._varlen_attn_backward" in op for op in called_ops)
        assert custom_ops_called

    @skipIfRocm(msg="ROCM does not support variable length attention")
    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FLASH_ATTENTION, "Flash Attention not supported"
    )
    @parametrize("dtype", [torch.bfloat16, torch.float16])
    @parametrize("is_causal", [False, True])
    def test_varlen_vs_sdpa(self, device, dtype, is_causal):
        torch.manual_seed(42)

        shape = VarlenShape(
            batch_size=8, max_seq_len=2048, embed_dim=1024, num_heads=16
        )

        attention_block = AttentionBlock(
            shape.embed_dim, shape.num_heads, device, dtype
        )

        golden_attention_block = AttentionBlock(
            shape.embed_dim, shape.num_heads, device, torch.float32
        )

        variable_length_batch_data = create_variable_length_batch(shape, device, dtype)
        golden_variable_length_batch_data = create_variable_length_batch(
            shape, device, torch.float32
        )

        varlen_output = attention_block.forward_varlen(
            variable_length_batch_data["x_packed"],
            variable_length_batch_data["cu_seq"],
            variable_length_batch_data["max_len"],
            is_causal=is_causal,
        )
        sdpa_output = attention_block.forward_sdpa(
            variable_length_batch_data["x_padded"],
            variable_length_batch_data["seq_lengths"],
            is_causal=is_causal,
        )

        golden_sdpa_output = golden_attention_block.forward_sdpa(
            golden_variable_length_batch_data["x_padded"],
            golden_variable_length_batch_data["seq_lengths"],
            is_causal=is_causal,
        )

        start_idx = 0
        for i, seq_len in enumerate(variable_length_batch_data["seq_lengths"]):
            end_idx = start_idx + seq_len

            varlen_seq = varlen_output[start_idx:end_idx]
            sdpa_seq = sdpa_output[i, :seq_len]
            golden_sdpa_seq = golden_sdpa_output[i, :seq_len]

            fwd_atol = (
                2 * (golden_sdpa_seq + 0.3 - 0.3 - golden_sdpa_seq).abs().max().item()
            )

            varlen_error = (varlen_seq - fwd_atol).abs().max().item()
            sdpa_error = (sdpa_seq - fwd_atol).abs().max().item()

            assert varlen_error <= 2 * sdpa_error + fwd_atol

            start_idx = end_idx

        varlen_grad_out = torch.ones_like(varlen_output)
        sdpa_grad_out = torch.ones_like(sdpa_output)
        golden_sdpa_grad_out = torch.ones_like(golden_sdpa_output)

        start_idx = 0
        for i, seq_len in enumerate(variable_length_batch_data["seq_lengths"]):
            end_idx = start_idx + seq_len
            sdpa_grad_out[i, :seq_len] = varlen_grad_out[start_idx:end_idx]
            start_idx = end_idx

        varlen_grad = torch.autograd.grad(
            outputs=varlen_output,
            inputs=variable_length_batch_data["x_packed"],
            grad_outputs=varlen_grad_out,
            retain_graph=True,
            create_graph=False,
            allow_unused=False,
        )[0]

        sdpa_grad = torch.autograd.grad(
            outputs=sdpa_output,
            inputs=variable_length_batch_data["x_padded"],
            grad_outputs=sdpa_grad_out,
            retain_graph=True,
            create_graph=False,
            allow_unused=False,
        )[0]

        golden_sdpa_grad = torch.autograd.grad(
            outputs=golden_sdpa_output,
            inputs=golden_variable_length_batch_data["x_padded"],
            grad_outputs=golden_sdpa_grad_out,
            retain_graph=True,
            create_graph=False,
            allow_unused=False,
        )[0]

        start_idx = 0
        for i, seq_len in enumerate(variable_length_batch_data["seq_lengths"]):
            end_idx = start_idx + seq_len

            varlen_grad_seq = varlen_grad[start_idx:end_idx]
            sdpa_grad_seq = sdpa_grad[i, :seq_len]
            golden_sdpa_seq = golden_sdpa_grad[i, :seq_len]

            fwd_atol = (
                2 * (golden_sdpa_seq + 0.3 - 0.3 - golden_sdpa_seq).abs().max().item()
            )

            varlen_error = (varlen_grad_seq - fwd_atol).abs().max().item()
            sdpa_error = (sdpa_grad_seq - fwd_atol).abs().max().item()

            assert varlen_error <= sdpa_error + fwd_atol

            start_idx = end_idx


device_types = ("cuda",)

instantiate_device_type_tests(TestVarlenAttention, globals(), only_for=device_types)

if __name__ == "__main__":
    run_tests()
