# Owner(s): ["module: sdpa"]
import unittest
from collections import namedtuple
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import (
    activate_flash_attention_impl,
    list_flash_attention_impls,
    restore_flash_attention_impl,
)
from torch.nn.attention.varlen import varlen_attn
from torch.testing._internal.common_cuda import (
    PLATFORM_SUPPORTS_FLASH_ATTENTION,
    SM90OrLater,
)
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_nn import NNTestCase
from torch.testing._internal.common_utils import parametrize, run_tests, TEST_WITH_ROCM
from torch.utils._python_dispatch import TorchDispatchMode


@contextmanager
def use_fa3():
    try:
        activate_flash_attention_impl("FA3")
        yield
    finally:
        restore_flash_attention_impl()


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
        scale: float | None = None,
        window_size: tuple[int, int] = (-1, -1),
    ):
        q, k, v = self.get_varlen_qkv(x_packed)

        attn_out = varlen_attn(
            q,
            k,
            v,
            cu_seq,
            cu_seq,
            max_len,
            max_len,
            scale=scale,
            window_size=window_size,
        )
        attn_out = attn_out.view(-1, self.embed_dim)

        return self.out_proj(attn_out)

    def forward_sdpa(
        self,
        x_padded: torch.Tensor,
        seq_lengths: torch.Tensor,
        scale: float | None = None,
        window_size: tuple[int, int] = (-1, -1),
    ):
        batch_size, seq_len, _ = x_padded.shape

        qkv = self.qkv_proj(x_padded)
        q, k, v = qkv.chunk(3, dim=-1)

        padding_mask = (
            torch.arange(seq_len, device=x_padded.device)[None, :]
            < seq_lengths[:, None]
        )

        attn_mask = padding_mask[:, None, None, :].expand(
            batch_size, self.num_heads, seq_len, seq_len
        )

        if window_size == (-1, 0):
            causal_mask = torch.tril(
                torch.ones(seq_len, seq_len, device=x_padded.device, dtype=torch.bool)
            )
            # Combine: attention allowed where BOTH padding is valid AND causal constraint is met
            attn_mask = attn_mask & causal_mask[None, None, :, :]

        if window_size[0] >= 0 or window_size[1] >= 0:
            window_mask = torch.zeros(
                seq_len, seq_len, dtype=torch.bool, device=x_padded.device
            )
            for i in range(seq_len):
                start = i - window_size[0] if window_size[0] >= 0 else 0
                end = i + window_size[1] + 1 if window_size[1] >= 0 else seq_len
                start = max(start, 0)
                end = min(end, seq_len)
                window_mask[i, start:end] = True
            attn_mask = attn_mask & window_mask[None, None, :, :]

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Don't pass is_causal since we already incorporated it into attn_mask
        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, scale=scale
        )

        attn_out = (
            attn_out.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.embed_dim)
        )

        return self.out_proj(attn_out)


def pack_sequences(seqs, device):
    x_packed = torch.cat(seqs, dim=0)
    seq_lens = torch.tensor([len(s) for s in seqs], device=device)
    cu_seq = torch.zeros(len(seqs) + 1, device=device, dtype=torch.int32)
    cu_seq[1:] = seq_lens.cumsum(0)
    max_len = seq_lens.max().item()

    return x_packed, cu_seq, max_len


def create_variable_length_batch(
    shape: VarlenShape, device: torch.device, dtype: torch.dtype
):
    seq_lengths = []
    for _ in range(shape.batch_size):
        length = torch.randint(1, shape.max_seq_len // 64 + 1, (1,)).item() * 64
        seq_lengths.append(min(length, shape.max_seq_len))

    sequences_fp32 = [
        torch.randn(seq_len, shape.embed_dim, device=device, dtype=torch.float32)
        for seq_len in seq_lengths
    ]

    x_packed_fp32, cu_seq, max_len = pack_sequences(sequences_fp32, device)
    seq_lengths = torch.tensor(seq_lengths, device=device)

    x_padded_fp32 = torch.zeros(
        shape.batch_size, max_len, shape.embed_dim, device=device, dtype=torch.float32
    )
    start_idx = 0
    for i, seq_len in enumerate(seq_lengths):
        end_idx = start_idx + seq_len
        x_padded_fp32[i, :seq_len] = x_packed_fp32[start_idx:end_idx]
        start_idx = end_idx

    x_packed = x_packed_fp32.detach().clone().to(dtype).requires_grad_(True)
    x_padded = x_padded_fp32.detach().clone().to(dtype).requires_grad_(True)
    x_padded_ref = x_padded_fp32.detach().clone().requires_grad_(True)

    return {
        "seq_lengths": seq_lengths,
        "cu_seq": cu_seq,
        "x_packed": x_packed,
        "x_padded": x_padded,
        "x_padded_ref": x_padded_ref,
        "max_len": max_len,
    }


def combine_cache_and_new(
    batch_size,
    cache_batch_idx,
    use_cache_batch_idx,
    page_size,
    cache_seqlen,
    new_seqlen,
    k_packed,
    v_packed,
    k_cache,
    v_cache,
    page_table,
    device,
):
    complete_k = []
    complete_v = []
    for i in range(batch_size):
        # cache_idx is idx of cache for query batch idx i
        cache_idx = cache_batch_idx[i].item() if use_cache_batch_idx else i

        # getting chunk corresponding to this batch
        new_k = k_packed[i * new_seqlen : (i + 1) * new_seqlen]
        new_v = v_packed[i * new_seqlen : (i + 1) * new_seqlen]

        if cache_seqlen == 0:  # first token generation, no cache
            complete_k.append(new_k)
            complete_v.append(new_v)
        elif page_size > 0:
            # reconstructing cached sequence from paged memory
            cached_tokens_k, cached_tokens_v = [], []
            # iterate through each index to recover token value
            for token_idx in range(cache_seqlen):
                block_idx = token_idx // page_size
                offset = token_idx % page_size
                phys_block = page_table[cache_idx, block_idx].item()

                # cache[phys_block, offset] is the actual token
                cached_tokens_k.append(k_cache[phys_block, offset])
                cached_tokens_v.append(v_cache[phys_block, offset])

            cached_k = torch.stack(cached_tokens_k, dim=0)
            cached_v = torch.stack(cached_tokens_v, dim=0)

            complete_k.append(torch.cat([cached_k, new_k], dim=0))
            complete_v.append(torch.cat([cached_v, new_v], dim=0))
        else:
            cached_k = k_cache[
                cache_idx, :cache_seqlen
            ]  # sequence corresponding to this batch
            cached_v = v_cache[cache_idx, :cache_seqlen]

            complete_k.append(torch.cat([cached_k, new_k], dim=0))
            complete_v.append(torch.cat([cached_v, new_v], dim=0))

    complete_k_packed, complete_cu_seq_k, complete_max_k = pack_sequences(
        complete_k, device
    )
    complete_v_packed, _, _ = pack_sequences(complete_v, device)

    return (complete_k_packed, complete_v_packed, complete_cu_seq_k, complete_max_k)


class TestVarlenAttention(NNTestCase):
    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FLASH_ATTENTION, "Flash Attention not supported"
    )
    @unittest.skipIf(
        TEST_WITH_ROCM, "varlen attention w/ sliding window not supported on ROCm"
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

        output = attention_block.forward_varlen(x_packed, cu_seq, shape.max_seq_len)

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

    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FLASH_ATTENTION, "Flash Attention not supported"
    )
    @unittest.skipIf(
        TEST_WITH_ROCM, "varlen attention w/ sliding window not supported on ROCm"
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

    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FLASH_ATTENTION, "Flash Attention not supported"
    )
    @unittest.skipIf(
        TEST_WITH_ROCM, "varlen attention w/ sliding window not supported on ROCm"
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
            output = compiled_forward(x_packed, cu_seq, shape.max_seq_len)

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

    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FLASH_ATTENTION, "Flash Attention not supported"
    )
    @unittest.skipIf(
        TEST_WITH_ROCM, "varlen attention w/ sliding window not supported on ROCm"
    )
    @parametrize("dtype", [torch.bfloat16, torch.float16])
    @parametrize("scale", [None, 0.1])
    @parametrize(
        "window_size",
        [
            (-1, -1),
            (-1, 0),
            (4, 0),
            (-1, 4),
            (0, 0),
            (1, 0),
            (0, 1),
            (0, -1),
            (4, 4),
            (8, 2),
            (1025, 1025),
        ],
    )
    def test_varlen_vs_sdpa(self, device, dtype, scale, window_size):
        torch.manual_seed(42)

        shape = VarlenShape(
            batch_size=4, max_seq_len=1024, embed_dim=1024, num_heads=16
        )

        batch_data = create_variable_length_batch(shape, device, dtype)
        seq_lengths = batch_data["seq_lengths"]
        cu_seq = batch_data["cu_seq"]
        max_len = batch_data["max_len"]
        x_packed = batch_data["x_packed"]
        x_padded = batch_data["x_padded"]
        x_padded_ref = batch_data["x_padded_ref"]

        golden_attention_block = AttentionBlock(
            shape.embed_dim, shape.num_heads, device, torch.float32
        )
        attention_block = AttentionBlock(
            shape.embed_dim, shape.num_heads, device, dtype
        )
        with torch.no_grad():
            attention_block.qkv_proj.weight.copy_(
                golden_attention_block.qkv_proj.weight.to(dtype)
            )
            attention_block.out_proj.weight.copy_(
                golden_attention_block.out_proj.weight.to(dtype)
            )

        varlen_output = attention_block.forward_varlen(
            x_packed,
            cu_seq,
            max_len,
            scale=scale,
            window_size=window_size,
        )
        sdpa_output = attention_block.forward_sdpa(
            x_padded,
            seq_lengths,
            scale=scale,
            window_size=window_size,
        )
        golden_sdpa_output = golden_attention_block.forward_sdpa(
            x_padded_ref,
            seq_lengths,
            scale=scale,
            window_size=window_size,
        )

        start_idx = 0
        for i, seq_len in enumerate(seq_lengths):
            end_idx = start_idx + seq_len

            varlen_seq = varlen_output[start_idx:end_idx]
            sdpa_seq = sdpa_output[i, :seq_len]
            golden_seq = golden_sdpa_output[i, :seq_len]

            fwd_atol = 2 * (golden_seq + 0.3 - 0.3 - golden_seq).abs().max().item()

            varlen_error = (varlen_seq - golden_seq.to(dtype)).abs().max().item()
            sdpa_error = (sdpa_seq - golden_seq.to(dtype)).abs().max().item()

            self.assertLessEqual(
                varlen_error,
                2 * sdpa_error + fwd_atol,
            )

            start_idx = end_idx

        grad_out = torch.randn_like(varlen_output)
        sdpa_grad_out = torch.zeros_like(sdpa_output)
        golden_sdpa_grad_out = torch.zeros(
            shape.batch_size,
            max_len,
            shape.embed_dim,
            device=device,
            dtype=torch.float32,
        )
        start_idx = 0
        for i, seq_len in enumerate(seq_lengths):
            end_idx = start_idx + seq_len
            sdpa_grad_out[i, :seq_len] = grad_out[start_idx:end_idx]
            golden_sdpa_grad_out[i, :seq_len] = grad_out[start_idx:end_idx].to(
                torch.float32
            )
            start_idx = end_idx

        varlen_grad = torch.autograd.grad(
            outputs=varlen_output,
            inputs=x_packed,
            grad_outputs=grad_out,
        )[0]

        sdpa_grad = torch.autograd.grad(
            outputs=sdpa_output,
            inputs=x_padded,
            grad_outputs=sdpa_grad_out,
        )[0]

        golden_sdpa_grad = torch.autograd.grad(
            outputs=golden_sdpa_output,
            inputs=x_padded_ref,
            grad_outputs=golden_sdpa_grad_out,
        )[0]

        start_idx = 0
        for i, seq_len in enumerate(seq_lengths):
            end_idx = start_idx + seq_len

            varlen_grad_seq = varlen_grad[start_idx:end_idx]
            sdpa_grad_seq = sdpa_grad[i, :seq_len]
            golden_grad_seq = golden_sdpa_grad[i, :seq_len]

            bwd_atol = (
                2 * (golden_grad_seq + 0.3 - 0.3 - golden_grad_seq).abs().max().item()
            )

            varlen_error = (
                (varlen_grad_seq - golden_grad_seq.to(dtype)).abs().max().item()
            )
            sdpa_error = (sdpa_grad_seq - golden_grad_seq.to(dtype)).abs().max().item()

            self.assertLessEqual(
                varlen_error,
                2 * sdpa_error + bwd_atol,
            )

            start_idx = end_idx

    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FLASH_ATTENTION, "Flash Attention not supported"
    )
    @unittest.skipIf(
        TEST_WITH_ROCM, "varlen attention w/ sliding window not supported on ROCm"
    )
    @parametrize("dtype", [torch.bfloat16, torch.float16])
    @parametrize(
        "window_size",
        [
            (-1, -1),
            (-1, 0),
            (4, 0),
            (-1, 4),
            (0, 0),
            (1, 0),
            (0, 1),
            (0, -1),
            (4, 4),
            (8, 2),
            (1025, 1025),
        ],
    )
    @parametrize("num_perms", [1, 3, 5])
    def test_batch_invariance(self, device, dtype, window_size, num_perms):
        torch.manual_seed(42)

        batch_size, max_seq_len = 4, 128

        seq_lengths = []
        for _ in range(batch_size):
            length = torch.randint(1, max_seq_len // 64 + 1, (1,)).item() * 64
            seq_lengths.append(min(length, max_seq_len))

        sequences_qkv = [
            [
                torch.testing.make_tensor(
                    (seq_len, 2, 128), device=device, dtype=dtype, requires_grad=True
                )
                for _ in range(3)
            ]
            for seq_len in seq_lengths
        ]
        sequences_q, sequences_k, sequences_v = map(list, zip(*sequences_qkv))

        q_packed_orig = torch.cat(sequences_q, dim=0)
        k_packed_orig = torch.cat(sequences_k, dim=0)
        v_packed_orig = torch.cat(sequences_v, dim=0)

        seq_lens = torch.tensor(seq_lengths, device=device)
        cu_seq_orig = torch.zeros(batch_size + 1, device=device, dtype=torch.int32)
        cu_seq_orig[1:] = seq_lens.cumsum(0)

        original_output = varlen_attn(
            q_packed_orig,
            k_packed_orig,
            v_packed_orig,
            cu_seq_orig,
            cu_seq_orig,
            max_seq_len,
            max_seq_len,
            window_size=window_size,
        )

        original_grad_out = torch.randn_like(original_output)
        original_grads = torch.autograd.grad(
            outputs=original_output,
            inputs=[q_packed_orig, k_packed_orig, v_packed_orig],
            grad_outputs=original_grad_out,
        )

        for _ in range(num_perms):
            perm = torch.randperm(batch_size)
            permuted_sequences_q = [sequences_q[perm[i]] for i in range(batch_size)]
            permuted_sequences_k = [sequences_k[perm[i]] for i in range(batch_size)]
            permuted_sequences_v = [sequences_v[perm[i]] for i in range(batch_size)]

            q_packed_perm = torch.cat(permuted_sequences_q, dim=0)
            k_packed_perm = torch.cat(permuted_sequences_k, dim=0)
            v_packed_perm = torch.cat(permuted_sequences_v, dim=0)

            permuted_seq_lens = torch.tensor(
                [seq_lengths[perm[i]] for i in range(batch_size)], device=device
            )
            cu_seq_perm = torch.zeros(batch_size + 1, device=device, dtype=torch.int32)
            cu_seq_perm[1:] = permuted_seq_lens.cumsum(0)

            permuted_output = varlen_attn(
                q_packed_perm,
                k_packed_perm,
                v_packed_perm,
                cu_seq_perm,
                cu_seq_perm,
                max_seq_len,
                max_seq_len,
                window_size=window_size,
            )

            for i in range(batch_size):
                orig_idx = perm[i].item()

                orig_start = cu_seq_orig[orig_idx].item()
                orig_end = cu_seq_orig[orig_idx + 1].item()
                orig_seq_output = original_output[orig_start:orig_end]

                perm_start = cu_seq_perm[i].item()
                perm_end = cu_seq_perm[i + 1].item()
                perm_seq_output = permuted_output[perm_start:perm_end]

                self.assertEqual(orig_seq_output, perm_seq_output)

            permuted_grad_out = torch.zeros_like(permuted_output)
            for i in range(batch_size):
                orig_idx = perm[i].item()
                orig_start = cu_seq_orig[orig_idx].item()
                orig_end = cu_seq_orig[orig_idx + 1].item()

                perm_start = cu_seq_perm[i].item()
                perm_end = cu_seq_perm[i + 1].item()

                permuted_grad_out[perm_start:perm_end] = original_grad_out[
                    orig_start:orig_end
                ]

            permuted_grads = torch.autograd.grad(
                outputs=permuted_output,
                inputs=[q_packed_perm, k_packed_perm, v_packed_perm],
                grad_outputs=permuted_grad_out,
            )

            for original_grad, permuted_grad in zip(original_grads, permuted_grads):
                for i in range(batch_size):
                    orig_idx = perm[i].item()

                    orig_start = cu_seq_orig[orig_idx].item()
                    orig_end = cu_seq_orig[orig_idx + 1].item()
                    orig_seq_grad = original_grad[orig_start:orig_end]

                    perm_start = cu_seq_perm[i].item()
                    perm_end = cu_seq_perm[i + 1].item()
                    perm_seq_grad = permuted_grad[perm_start:perm_end]

                    self.assertEqual(orig_seq_grad, perm_seq_grad)

    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FLASH_ATTENTION, "Flash Attention not supported"
    )
    @unittest.skipIf(
        TEST_WITH_ROCM, "varlen attention w/ sliding window not supported on ROCm"
    )
    def test_kv_cache_requires_fa3(self, device):
        torch.manual_seed(42)
        dtype = torch.bfloat16

        shape = VarlenShape(batch_size=2, max_seq_len=128, embed_dim=256, num_heads=4)
        batch_data = create_variable_length_batch(shape, device, dtype)
        cu_seq = batch_data["cu_seq"]
        max_len = batch_data["max_len"]

        attention_block = AttentionBlock(
            shape.embed_dim, shape.num_heads, device, dtype
        )
        q, k, v = attention_block.get_varlen_qkv(batch_data["x_packed"])

        k_cache = torch.randn(
            shape.batch_size,
            64,
            shape.num_heads,
            shape.embed_dim // shape.num_heads,
            device=device,
            dtype=dtype,
        )
        v_cache = torch.randn_like(k_cache)
        cache_seqlens = torch.randint(
            1, 32, (shape.batch_size,), device=device, dtype=torch.int32
        )

        restore_flash_attention_impl()

        with self.assertRaisesRegex(
            RuntimeError, "KV cache parameters.*are not supported"
        ):
            varlen_attn(
                q,
                k,
                v,
                cu_seq,
                cu_seq,
                max_len,
                max_len,
                k_cache=k_cache,
                v_cache=v_cache,
                cache_seqlens=cache_seqlens,
            )

    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FLASH_ATTENTION, "Flash Attention not supported"
    )
    @unittest.skipIf(
        TEST_WITH_ROCM, "varlen attention w/ sliding window not supported on ROCm"
    )
    @unittest.skipIf(not SM90OrLater, "FA3 requires SM90+")
    @unittest.skipIf("FA3" not in list_flash_attention_impls(), "FA3 not available")
    @parametrize("dtype", [torch.bfloat16, torch.float16])
    @parametrize("cache_seqlen", [0, 32, 64])
    @parametrize("new_seqlen", [1, 16])
    @parametrize("use_cache_batch_idx", [False, True])
    @parametrize("page_size", [0, 4, 16, 32])
    def test_kv_cache_correctness(
        self, device, dtype, cache_seqlen, new_seqlen, use_cache_batch_idx, page_size
    ):
        torch.manual_seed(42)
        batch_size, num_heads, head_dim, max_cache_len = 2, 4, 64, 128
        cache_seqlens = torch.full(
            (batch_size,), cache_seqlen, device=device, dtype=torch.int32
        )

        if page_size > 0:
            num_blocks_per_seq = (max_cache_len + page_size - 1) // page_size
            total_blocks = num_blocks_per_seq * batch_size
            k_cache = torch.randn(
                total_blocks, page_size, num_heads, head_dim, device=device, dtype=dtype
            )
            v_cache = torch.randn_like(k_cache)
            page_table = torch.arange(
                total_blocks, device=device, dtype=torch.int32
            ).view(batch_size, num_blocks_per_seq)
        else:
            k_cache = torch.randn(
                batch_size,
                max_cache_len,
                num_heads,
                head_dim,
                device=device,
                dtype=dtype,
            )
            v_cache = torch.randn_like(k_cache)
            page_table = None

        cache_batch_idx = (
            torch.tensor([1, 0], device=device, dtype=torch.int32)
            if use_cache_batch_idx
            else None
        )
        total_new = new_seqlen * batch_size
        q_packed = torch.randn(
            total_new, num_heads, head_dim, device=device, dtype=dtype
        )
        k_packed = torch.randn(
            total_new, num_heads, head_dim, device=device, dtype=dtype
        )
        v_packed = torch.randn(
            total_new, num_heads, head_dim, device=device, dtype=dtype
        )
        cu_seq_new = torch.arange(
            0, total_new + 1, new_seqlen, device=device, dtype=torch.int32
        )

        complete_k_packed, complete_v_packed, complete_cu_seq_k, complete_max_k = (
            combine_cache_and_new(
                batch_size,
                cache_batch_idx,
                use_cache_batch_idx,
                page_size,
                cache_seqlen,
                new_seqlen,
                k_packed,
                v_packed,
                k_cache,
                v_cache,
                page_table,
                device,
            )
        )

        with use_fa3(), torch.no_grad():
            # full recompute without cache/pagetable
            expected = varlen_attn(
                query=q_packed,
                key=complete_k_packed,
                value=complete_v_packed,
                cu_seq_q=cu_seq_new,
                cu_seq_k=complete_cu_seq_k,
                max_q=new_seqlen,
                max_k=complete_max_k,
            )
            # using kv cache
            actual = varlen_attn(
                query=q_packed,
                key=k_packed,
                value=v_packed,
                cu_seq_q=cu_seq_new,
                cu_seq_k=cu_seq_new,
                max_q=new_seqlen,
                max_k=new_seqlen,
                k_cache=k_cache,
                v_cache=v_cache,
                cache_seqlens=cache_seqlens,
                cache_batch_idx=cache_batch_idx,
                page_table=page_table,
            )

        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual, expected)


device_types = ("cuda",)

instantiate_device_type_tests(TestVarlenAttention, globals(), only_for=device_types)

if __name__ == "__main__":
    run_tests()
