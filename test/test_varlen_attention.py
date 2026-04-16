# Owner(s): ["module: sdpa"]
import unittest
from collections import namedtuple
from contextlib import contextmanager, nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import (
    activate_flash_attention_impl,
    restore_flash_attention_impl,
)
from torch.nn.attention.varlen import varlen_attn, varlen_attn_out
from torch.testing._internal.common_cuda import (
    IS_SM90,
    PLATFORM_SUPPORTS_CK_SDPA,
    PLATFORM_SUPPORTS_FLASH_ATTENTION,
    SM100OrLater,
    SM120OrLater,
    SM90OrLater,
)
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_nn import NNTestCase
from torch.testing._internal.common_utils import (
    decorateIf,
    parametrize,
    run_tests,
    TEST_WITH_ROCM,
)
from torch.utils._python_dispatch import TorchDispatchMode


@contextmanager
def use_fa3():
    try:
        activate_flash_attention_impl("FA3")
    except (ModuleNotFoundError, RuntimeError) as err:
        raise unittest.SkipTest(
            "FA3 backend not available (flash_attn_interface missing)"
        ) from err
    try:
        yield
    finally:
        restore_flash_attention_impl()


@contextmanager
def use_fa4():
    try:
        activate_flash_attention_impl("FA4")
    except (ModuleNotFoundError, RuntimeError) as err:
        raise unittest.SkipTest("FA4 backend not available") from err
    try:
        yield
    finally:
        restore_flash_attention_impl()


def _use_backend(backend):
    return {"fa2": nullcontext, "fa3": use_fa3, "fa4": use_fa4}[backend]()


def _varlen_backends(*, include_fa4_paged_kv: bool) -> list[str]:
    fa4_supported = (
        SM100OrLater if include_fa4_paged_kv else SM90OrLater
    ) and not SM120OrLater
    return ["fa2"] + (["fa3"] if IS_SM90 else []) + (["fa4"] if fa4_supported else [])


VarlenShape = namedtuple(
    "VarlenShape",
    ["batch_size", "max_seq_len", "embed_dim", "num_heads", "num_kv_heads"],
    defaults=[None],
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
        self,
        embed_dim: int,
        num_heads: int,
        device: torch.device,
        dtype: torch.dtype,
        num_kv_heads: int | None = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(
            embed_dim,
            num_heads * self.head_dim,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.kv_proj = nn.Linear(
            embed_dim,
            2 * self.num_kv_heads * self.head_dim,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.out_proj = nn.Linear(
            embed_dim, embed_dim, bias=False, device=device, dtype=dtype
        )

    @property
    def enable_gqa(self):
        return self.num_kv_heads != self.num_heads

    def get_varlen_qkv(
        self,
        x_packed: torch.Tensor,
    ):
        q = self.q_proj(x_packed).view(-1, self.num_heads, self.head_dim)
        kv = self.kv_proj(x_packed).view(-1, 2, self.num_kv_heads, self.head_dim)
        k, v = kv[:, 0], kv[:, 1]
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
            enable_gqa=self.enable_gqa,
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

        q = self.q_proj(x_padded)
        kv = self.kv_proj(x_padded)
        k, v = kv.view(batch_size, seq_len, 2, self.num_kv_heads, self.head_dim).unbind(
            dim=2
        )

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
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(
            1, 2
        )
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(
            1, 2
        )

        # Don't pass is_causal since we already incorporated it into attn_mask.
        if self.enable_gqa:
            # Force math backend for GQA
            with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
                attn_out = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=attn_mask,
                    scale=scale,
                    enable_gqa=True,
                )
        else:
            attn_out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                scale=scale,
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


class TestVarlenAttention(NNTestCase):
    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FLASH_ATTENTION, "Flash Attention not supported"
    )
    @parametrize("dtype", [torch.bfloat16, torch.float16])
    @parametrize(
        "sdpa_backend",
        ["aotriton", "ck"] if PLATFORM_SUPPORTS_CK_SDPA else ["aotriton"],
    )
    @parametrize(
        "backend",
        _varlen_backends(include_fa4_paged_kv=False),
    )
    def test_basic_functionality(self, device, dtype, backend, sdpa_backend=None):
        if TEST_WITH_ROCM:
            torch.backends.cuda.preferred_rocm_fa_library(sdpa_backend)

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

        with _use_backend(backend):
            output = attention_block.forward_varlen(x_packed, cu_seq, shape.max_seq_len)

            self.assertEqual(output.shape, (total_tokens, shape.embed_dim))
            self.assertEqual(output.device, torch.device(device))
            self.assertEqual(output.dtype, dtype)

            # varlen_attn_out should produce the same result and write into the buffer
            with torch.no_grad():
                q, k, v = attention_block.get_varlen_qkv(x_packed)
                expected = varlen_attn(
                    q, k, v, cu_seq, cu_seq, shape.max_seq_len, shape.max_seq_len
                )
                out_buf = torch.empty_like(expected)
                actual = varlen_attn_out(
                    out_buf,
                    q,
                    k,
                    v,
                    cu_seq,
                    cu_seq,
                    shape.max_seq_len,
                    shape.max_seq_len,
                )
                self.assertEqual(actual.data_ptr(), out_buf.data_ptr())
                self.assertEqual(out_buf, expected)

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
    @parametrize(
        "sdpa_backend",
        ["aotriton", "ck"] if PLATFORM_SUPPORTS_CK_SDPA else ["aotriton"],
    )
    @parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_custom_op_compliance(self, device, dtype, sdpa_backend=None):
        if TEST_WITH_ROCM:
            torch.backends.cuda.preferred_rocm_fa_library(sdpa_backend)
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

        # opcheck for _varlen_attn_out (no backward)
        out_buf = torch.empty_like(q)
        torch.library.opcheck(
            torch.ops.torch_attn._varlen_attn_out,
            (
                out_buf,
                q,
                k,
                v,
                cu_seq,
                cu_seq,
                shape.max_seq_len,
                shape.max_seq_len,
                False,
            ),
            test_utils=["test_schema", "test_faketensor"],
        )

    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FLASH_ATTENTION, "Flash Attention not supported"
    )
    @parametrize(
        "sdpa_backend",
        ["aotriton", "ck"] if PLATFORM_SUPPORTS_CK_SDPA else ["aotriton"],
    )
    @parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_custom_op_registration(self, device, dtype, sdpa_backend=None):
        if TEST_WITH_ROCM:
            torch.backends.cuda.preferred_rocm_fa_library(sdpa_backend)
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
        if not custom_ops_called:
            raise AssertionError("custom varlen attention ops should have been called")

        # Also verify _varlen_attn_out dispatches correctly under compile
        q, k, v = attention_block.get_varlen_qkv(x_packed.detach())

        def run_varlen_out(q, k, v, cu_seq, max_len):
            out_buf = torch.empty_like(q)
            varlen_attn_out(out_buf, q, k, v, cu_seq, cu_seq, max_len, max_len)
            return out_buf

        compiled_out = torch.compile(run_varlen_out, backend="eager", fullgraph=True)
        with OpLoggingMode() as out_mode:
            compiled_out(q, k, v, cu_seq, shape.max_seq_len)

        if not any("torch_attn._varlen_attn_out" in op for op in out_mode.called_ops):
            raise AssertionError("custom _varlen_attn_out op should have been called")

    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FLASH_ATTENTION, "Flash Attention not supported"
    )
    @parametrize(
        "sdpa_backend",
        ["aotriton", "ck"] if PLATFORM_SUPPORTS_CK_SDPA else ["aotriton"],
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
    @parametrize(
        "backend",
        _varlen_backends(include_fa4_paged_kv=False),
    )
    @parametrize("enable_gqa", [False, True])
    def test_varlen_vs_sdpa(
        self, device, dtype, scale, window_size, backend, enable_gqa, sdpa_backend=None
    ):
        if TEST_WITH_ROCM:
            torch.backends.cuda.preferred_rocm_fa_library(sdpa_backend)

        torch.manual_seed(42)

        num_heads = 16
        num_kv_heads = 4 if enable_gqa else num_heads
        shape = VarlenShape(
            batch_size=4,
            max_seq_len=1024,
            embed_dim=1024,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
        )

        batch_data = create_variable_length_batch(shape, device, dtype)
        seq_lengths = batch_data["seq_lengths"]
        cu_seq = batch_data["cu_seq"]
        max_len = batch_data["max_len"]
        x_packed = batch_data["x_packed"]
        x_padded = batch_data["x_padded"]
        x_padded_ref = batch_data["x_padded_ref"]

        golden_attention_block = AttentionBlock(
            shape.embed_dim,
            shape.num_heads,
            device,
            torch.float32,
            num_kv_heads=num_kv_heads,
        )
        attention_block = AttentionBlock(
            shape.embed_dim,
            shape.num_heads,
            device,
            dtype,
            num_kv_heads=num_kv_heads,
        )
        with torch.no_grad():
            attention_block.q_proj.weight.copy_(
                golden_attention_block.q_proj.weight.to(dtype)
            )
            attention_block.kv_proj.weight.copy_(
                golden_attention_block.kv_proj.weight.to(dtype)
            )
            attention_block.out_proj.weight.copy_(
                golden_attention_block.out_proj.weight.to(dtype)
            )

        with _use_backend(backend):
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

        with _use_backend(backend):
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
    @parametrize("dtype", [torch.bfloat16, torch.float16])
    @parametrize("num_splits", [1, None])
    @parametrize(
        "window_size",
        [
            (-1, -1),
            (-1, 0),
            (1025, 1025),
            (384, 0),  # edge case
        ],
    )
    @parametrize(
        "backend",
        ["fa2"] + (["fa3"] if IS_SM90 else []) + (["fa4"] if SM100OrLater else []),
    )
    def test_batch_invariance(
        self, device, dtype, num_splits, window_size, backend, sdpa_backend=None
    ):
        if TEST_WITH_ROCM:
            torch.backends.cuda.preferred_rocm_fa_library(sdpa_backend)

        split_kwargs = {"num_splits": num_splits} if backend != "fa2" else {}

        torch.manual_seed(42)

        num_heads, head_dim = 2, 128
        target_seq_len = 512
        extra_seq_len = 1024

        target_q = torch.randn(
            target_seq_len, num_heads, head_dim, device=device, dtype=dtype
        )
        target_k = torch.randn(
            target_seq_len, num_heads, head_dim, device=device, dtype=dtype
        )
        target_v = torch.randn(
            target_seq_len, num_heads, head_dim, device=device, dtype=dtype
        )

        extra_q = torch.randn(
            extra_seq_len, num_heads, head_dim, device=device, dtype=dtype
        )
        extra_k = torch.randn(
            extra_seq_len, num_heads, head_dim, device=device, dtype=dtype
        )
        extra_v = torch.randn(
            extra_seq_len, num_heads, head_dim, device=device, dtype=dtype
        )

        cu_seq_solo = torch.tensor(
            [0, target_seq_len], device=device, dtype=torch.int32
        )
        cu_seq_batch = torch.tensor(
            [0, target_seq_len, target_seq_len + extra_seq_len],
            device=device,
            dtype=torch.int32,
        )

        all_q = torch.cat([target_q, extra_q], dim=0)
        all_k = torch.cat([target_k, extra_k], dim=0)
        all_v = torch.cat([target_v, extra_v], dim=0)

        # fa4 is batch invariant (num_splits=1) by default
        with _use_backend(backend), torch.no_grad():
            solo_output = varlen_attn(
                target_q,
                target_k,
                target_v,
                cu_seq_solo,
                cu_seq_solo,
                target_seq_len,
                target_seq_len,
                window_size=window_size,
                **split_kwargs,
            )

            batched_output = varlen_attn(
                all_q,
                all_k,
                all_v,
                cu_seq_batch,
                cu_seq_batch,
                extra_seq_len,
                extra_seq_len,
                window_size=window_size,
                **split_kwargs,
            )

            solo_out_buf = torch.empty_like(target_q)
            varlen_attn_out(
                solo_out_buf,
                target_q,
                target_k,
                target_v,
                cu_seq_solo,
                cu_seq_solo,
                target_seq_len,
                target_seq_len,
                window_size=window_size,
                **split_kwargs,
            )

            batched_out_buf = torch.empty_like(all_q)
            varlen_attn_out(
                batched_out_buf,
                all_q,
                all_k,
                all_v,
                cu_seq_batch,
                cu_seq_batch,
                extra_seq_len,
                extra_seq_len,
                window_size=window_size,
                **split_kwargs,
            )
            if num_splits == 1:
                self.assertEqual(solo_output, batched_output[:target_seq_len])
                self.assertEqual(solo_out_buf, batched_out_buf[:target_seq_len])
                self.assertEqual(solo_output, solo_out_buf)
            else:
                if backend == "fa3":
                    self.assertNotEqual(solo_output, batched_output[:target_seq_len])
                    self.assertNotEqual(solo_out_buf, batched_out_buf[:target_seq_len])

    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FLASH_ATTENTION, "Flash Attention not supported"
    )
    @unittest.skipIf(TEST_WITH_ROCM, "ROCm does not support seqused_k")
    @decorateIf(
        unittest.expectedFailure,
        lambda params: params["backend"] != "fa2"
        and any(kv_len < 128 for kv_len in params["actual_kv_lens"]),
    )
    @parametrize(
        "sdpa_backend",
        ["aotriton", "ck"] if PLATFORM_SUPPORTS_CK_SDPA else ["aotriton"],
    )
    @parametrize("dtype", [torch.bfloat16, torch.float16])
    @parametrize(
        "actual_kv_lens",
        [
            [32, 64, 96, 48],
            [1, 1, 1, 1],
            [128, 128, 128, 128],
            [1, 128, 1, 128],
            [127, 63, 33, 17],
        ],
    )
    @parametrize("backend", _varlen_backends(include_fa4_paged_kv=False))
    def test_seqused_k_kv_cache(
        self, device, dtype, actual_kv_lens, backend, sdpa_backend=None
    ):
        if TEST_WITH_ROCM:
            torch.backends.cuda.preferred_rocm_fa_library(sdpa_backend)

        torch.manual_seed(42)

        batch_size = 4
        num_heads = 8
        head_dim = 64
        cache_size = 128

        q_seqs = [
            torch.randn(1, num_heads, head_dim, device=device, dtype=dtype)
            for _ in range(batch_size)
        ]
        q_packed, cu_seq_q, max_q = pack_sequences(q_seqs, device)

        k_seqs = [
            torch.randn(kv_len, num_heads, head_dim, device=device, dtype=dtype)
            for kv_len in actual_kv_lens
        ]
        v_seqs = [
            torch.randn(kv_len, num_heads, head_dim, device=device, dtype=dtype)
            for kv_len in actual_kv_lens
        ]

        k_cache_slots = []
        v_cache_slots = []
        for i in range(batch_size):
            k_slot = torch.full(
                (cache_size, num_heads, head_dim),
                float("nan"),
                device=device,
                dtype=dtype,
            )
            v_slot = torch.full(
                (cache_size, num_heads, head_dim),
                float("nan"),
                device=device,
                dtype=dtype,
            )
            k_slot[: actual_kv_lens[i]] = k_seqs[i]
            v_slot[: actual_kv_lens[i]] = v_seqs[i]
            k_cache_slots.append(k_slot)
            v_cache_slots.append(v_slot)

        k_cache_packed = torch.cat(k_cache_slots, dim=0)
        v_cache_packed = torch.cat(v_cache_slots, dim=0)
        cu_seq_k_cache = torch.arange(
            0,
            (batch_size + 1) * cache_size,
            cache_size,
            device=device,
            dtype=torch.int32,
        )
        seqused_k = torch.tensor(actual_kv_lens, device=device, dtype=torch.int32)

        with _use_backend(backend), torch.no_grad():
            output_cached = varlen_attn(
                q_packed,
                k_cache_packed,
                v_cache_packed,
                cu_seq_q,
                cu_seq_k_cache,
                max_q,
                cache_size,
                seqused_k=seqused_k,
            )

        k_real_packed, cu_seq_k_real, max_k_real = pack_sequences(k_seqs, device)
        v_real_packed = torch.cat(v_seqs, dim=0)

        with _use_backend(backend), torch.no_grad():
            output_reference = varlen_attn(
                q_packed,
                k_real_packed,
                v_real_packed,
                cu_seq_q,
                cu_seq_k_real,
                max_q,
                max_k_real,
            )

        self.assertFalse(output_cached.isnan().any())
        self.assertEqual(output_cached, output_reference)

        # varlen_attn_out with seqused_k should match
        with _use_backend(backend), torch.no_grad():
            out_buf = torch.empty_like(q_packed)
            output_out = varlen_attn_out(
                out_buf,
                q_packed,
                k_cache_packed,
                v_cache_packed,
                cu_seq_q,
                cu_seq_k_cache,
                max_q,
                cache_size,
                seqused_k=seqused_k,
            )
            self.assertEqual(output_out.data_ptr(), out_buf.data_ptr())
            self.assertEqual(out_buf, output_cached)

    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FLASH_ATTENTION, "Flash Attention not supported"
    )
    @unittest.skipIf(TEST_WITH_ROCM, "ROCm does not support seqused_k")
    @parametrize("dtype", [torch.bfloat16, torch.float16])
    @parametrize("page_size", [32, 64, 128, 256])
    @parametrize("compile", [False, True])
    @parametrize(
        "actual_kv_lens",
        [
            [32, 64, 96, 48],
            [1, 1, 1, 1],
            [128, 128, 128, 128],
            [1, 128, 1, 128],
            [127, 63, 33, 17],
        ],
    )
    @parametrize(
        "backend",
        _varlen_backends(include_fa4_paged_kv=True),
    )
    def test_block_table_kv_cache(
        self, device, dtype, page_size, compile, actual_kv_lens, backend
    ):
        if backend == "fa2" and page_size % 256 != 0:
            self.skipTest("FA2 paged KV requires page_size divisible by 256")

        torch.manual_seed(42)

        batch_size = 4
        num_heads = 8
        head_dim = 64
        max_kv = max(actual_kv_lens)
        max_pages_per_seq = (max_kv + page_size - 1) // page_size
        cache_size = max_pages_per_seq * page_size
        total_pages = batch_size * max_pages_per_seq

        q_seqs = [
            torch.randn(1, num_heads, head_dim, device=device, dtype=dtype)
            for _ in range(batch_size)
        ]
        q_packed, cu_seq_q, max_q = pack_sequences(q_seqs, device)

        k_pages = torch.randn(
            total_pages, page_size, num_heads, head_dim, device=device, dtype=dtype
        )
        v_pages = torch.randn(
            total_pages, page_size, num_heads, head_dim, device=device, dtype=dtype
        )
        block_table = torch.randperm(
            total_pages, device=device, dtype=torch.int32
        ).view(batch_size, max_pages_per_seq)
        seqused_k = torch.tensor(actual_kv_lens, device=device, dtype=torch.int32)

        idx = (
            block_table.long()
            .view(-1, 1, 1, 1)
            .expand(-1, page_size, num_heads, head_dim)
        )
        k_gathered = k_pages.gather(0, idx).view(
            batch_size, cache_size, num_heads, head_dim
        )
        v_gathered = v_pages.gather(0, idx).view(
            batch_size, cache_size, num_heads, head_dim
        )
        k_seqs = [k_gathered[i, : actual_kv_lens[i]] for i in range(batch_size)]
        v_seqs = [v_gathered[i, : actual_kv_lens[i]] for i in range(batch_size)]

        k_real_packed, cu_seq_k_real, max_k_real = pack_sequences(k_seqs, device)
        v_real_packed = torch.cat(v_seqs, dim=0)

        attn_fn = torch.compile(varlen_attn, fullgraph=True) if compile else varlen_attn

        # Reference: no block_table
        with _use_backend(backend), torch.no_grad():
            output_reference = varlen_attn(
                q_packed,
                k_real_packed,
                v_real_packed,
                cu_seq_q,
                cu_seq_k_real,
                max_q,
                max_k_real,
            )

        cu_seq_k = torch.arange(
            0,
            (batch_size + 1) * cache_size,
            cache_size,
            device=device,
            dtype=torch.int32,
        )

        # FA2 requires cu_seq_k for paged KV; FA3/FA4 pass None
        cu_seq_k_paged = cu_seq_k if backend == "fa2" else None

        with _use_backend(backend), torch.no_grad():
            output_paged = attn_fn(
                q_packed,
                k_pages,
                v_pages,
                cu_seq_q,
                cu_seq_k_paged,
                max_q,
                cache_size,
                seqused_k=seqused_k,
                block_table=block_table,
            )

        self.assertEqual(output_paged, output_reference)

        # varlen_attn_out with paged KV cache should match
        with _use_backend(backend), torch.no_grad():
            out_buf = torch.empty_like(q_packed)
            output_out = varlen_attn_out(
                out_buf,
                q_packed,
                k_pages,
                v_pages,
                cu_seq_q,
                cu_seq_k_paged,
                max_q,
                cache_size,
                seqused_k=seqused_k,
                block_table=block_table,
            )
            self.assertEqual(output_out.data_ptr(), out_buf.data_ptr())
            self.assertEqual(out_buf, output_paged)

        # compile the lower level aten op (FA3 only, will cause graph break)
        if compile and backend != "fa2":
            compiled_aten_op = torch.compile(
                torch.ops.aten._flash_attention_forward_no_dropout_inplace
            )
            with _use_backend(backend), torch.no_grad():
                out_buf = torch.empty_like(q_packed)
                compiled_aten_op(
                    out_buf,
                    q_packed,
                    k_pages,
                    v_pages,
                    cu_seq_q,
                    None,
                    max_q,
                    cache_size,
                    0.0,
                    False,
                    False,
                    seqused_k=seqused_k,
                    block_table=block_table,
                )
            self.assertEqual(out_buf, output_reference)

    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FLASH_ATTENTION, "Flash Attention not supported"
    )
    @parametrize("dtype", [torch.bfloat16, torch.float16])
    @parametrize(
        "backend",
        ["fa2"] + (["fa3"] if IS_SM90 else []) + (["fa4"] if SM100OrLater else []),
    )
    def test_enable_gqa(self, device, dtype, backend):
        torch.manual_seed(42)

        head_dim = 64
        seq_len = 512
        num_heads_q, num_heads_k = 16, 4
        total_tokens = 2 * seq_len

        q = torch.randn(total_tokens, num_heads_q, head_dim, device=device, dtype=dtype)
        k = torch.randn(total_tokens, num_heads_k, head_dim, device=device, dtype=dtype)
        v = torch.randn(total_tokens, num_heads_k, head_dim, device=device, dtype=dtype)
        cu_seq = torch.tensor(
            [0, seq_len, total_tokens], device=device, dtype=torch.int32
        )

        with self.assertRaisesRegex(ValueError, "enable_gqa=True"):
            varlen_attn(q, k, v, cu_seq, cu_seq, seq_len, seq_len)

        with self.assertRaisesRegex(ValueError, "enable_gqa=True"):
            varlen_attn_out(
                torch.empty_like(q), q, k, v, cu_seq, cu_seq, seq_len, seq_len
            )

        k_bad = torch.randn(total_tokens, 3, head_dim, device=device, dtype=dtype)
        v_bad = torch.randn(total_tokens, 3, head_dim, device=device, dtype=dtype)
        with self.assertRaisesRegex(ValueError, "multiple of kv heads"):
            varlen_attn(
                q, k_bad, v_bad, cu_seq, cu_seq, seq_len, seq_len, enable_gqa=True
            )

        with _use_backend(backend), torch.no_grad():
            out = varlen_attn(
                q, k, v, cu_seq, cu_seq, seq_len, seq_len, enable_gqa=True
            )
            out_buf = torch.empty_like(q)
            varlen_attn_out(
                out_buf, q, k, v, cu_seq, cu_seq, seq_len, seq_len, enable_gqa=True
            )
            self.assertEqual(out_buf, out)


device_types = ("cuda",)

instantiate_device_type_tests(TestVarlenAttention, globals(), only_for=device_types)

if __name__ == "__main__":
    run_tests()
