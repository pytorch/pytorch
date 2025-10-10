# Owner(s): ["module: nn"]
import unittest
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import varlen_attn
from torch.testing._internal.common_cuda import PLATFORM_SUPPORTS_FLASH_ATTENTION
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_nn import NNTestCase
from torch.testing._internal.common_utils import parametrize, run_tests


VarlenShape = namedtuple(
    "VarlenShape", ["batch_size", "max_seq_len", "embed_dim", "num_heads"]
)

default_tolerances = {
    torch.float16: {"atol": 1e-1, "rtol": 1e-1},
    torch.bfloat16: {"atol": 9e-2, "rtol": 5e-2},
    torch.float32: {"atol": 1e-5, "rtol": 1.3e-6},
}


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

    def forward_varlen(
        self,
        x_packed: torch.Tensor,
        cu_seq: torch.Tensor,
        max_len: int,
        is_causal: bool = False,
    ):
        qkv = self.qkv_proj(x_packed)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_heads, self.head_dim)
        v = v.view(-1, self.num_heads, self.head_dim)

        attn_out = varlen_attn(
            q, k, v, cu_seq, cu_seq, max_len, max_len, is_causal=is_causal
        )
        attn_out = attn_out.view(-1, self.embed_dim)

        return self.out_proj(attn_out)

    def forward_sdpa(self, x_padded: torch.Tensor, is_causal: bool = False):
        batch_size, seq_len, _ = x_padded.shape

        qkv = self.qkv_proj(x_padded)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
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

    x_packed = torch.randn(total_tokens, shape.embed_dim, device=device, dtype=dtype)

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

    return {
        "seq_lengths": seq_lengths,
        "cu_seq": cu_seq,
        "x_packed": x_packed,
        "x_padded": x_padded,
        "max_len": max_len,
        "total_tokens": total_tokens,
    }


class TestVarlenAttention(NNTestCase):
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
            total_tokens, shape.embed_dim, device=device, dtype=dtype
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

        variable_length_batch_data = create_variable_length_batch(shape, device, dtype)

        varlen_output = attention_block.forward_varlen(
            variable_length_batch_data["x_packed"],
            variable_length_batch_data["cu_seq"],
            variable_length_batch_data["max_len"],
            is_causal=is_causal,
        )
        sdpa_output = attention_block.forward_sdpa(
            variable_length_batch_data["x_padded"], is_causal=is_causal
        )

        tolerances = default_tolerances[dtype]
        start_idx = 0
        for i, seq_len in enumerate(variable_length_batch_data["seq_lengths"]):
            end_idx = start_idx + seq_len

            varlen_seq = varlen_output[start_idx:end_idx]
            sdpa_seq = sdpa_output[i, :seq_len]

            torch.testing.assert_close(varlen_seq, sdpa_seq, **tolerances)
            start_idx = end_idx


device_types = ("cuda",)

instantiate_device_type_tests(TestVarlenAttention, globals(), only_for=device_types)

if __name__ == "__main__":
    run_tests()
