# Owner(s): ["module: inductor"]
# flake8: noqa: B950

import functools
import random
import string
import unittest
from collections import namedtuple
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import product
from typing import Callable, Optional, Union
from unittest import expectedFailure, skip, skipUnless
from unittest.mock import patch

import torch
from torch._dynamo.testing import CompileCounterWithBackend, normalize_gm
from torch._inductor import metrics
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.utils import run_and_get_code
from torch.nn.attention.experimental._paged_attention import PagedAttention
from torch.nn.attention.flex_attention import (
    _create_empty_block_mask,
    _DEFAULT_SPARSE_BLOCK_SIZE,
    _identity,
    _mask_mod_signature,
    _score_mod_signature,
    and_masks,
    BlockMask,
    create_block_mask,
    flex_attention,
    noop_mask,
    or_masks,
)
from torch.testing import FileCheck
from torch.testing._internal import common_utils
from torch.testing._internal.common_cuda import PLATFORM_SUPPORTS_BF16, TEST_MULTIGPU
from torch.testing._internal.common_device_type import (
    flex_attention_supported_platform as supported_platform,
)
from torch.testing._internal.common_utils import IS_MACOS, TEST_WITH_ROCM
from torch.utils._triton import has_triton


# Use this decorator only when hitting Triton bugs on H100
running_on_a100_only = skipUnless(
    torch.cuda.is_available()
    and has_triton()
    and torch.cuda.get_device_capability() == (8, 0),
    "Requires A100 and Triton",
)

Tolerances = namedtuple("Tolerances", ["atol", "rtol"])
torch.set_float32_matmul_precision("high")

index = torch.ops.aten.index
Tensor = torch.Tensor


@contextmanager
def temp_float32_matmul_precision(precision: str):
    """
    Temporarily set the float32 matmul precision and restore it after the context is exited.

    Args:
    precision (str): The precision to set ('highest', 'high', or 'medium').
    """
    original_precision = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision(precision)
        yield
    finally:
        torch.set_float32_matmul_precision(original_precision)


def rmse(ref, res):
    """
    Calculate root mean squared error
    """
    return torch.sqrt(torch.mean(torch.square(ref - res)))


def create_attention(score_mod, block_mask, enable_gqa=False):
    return functools.partial(
        flex_attention,
        score_mod=score_mod,
        block_mask=block_mask,
        enable_gqa=enable_gqa,
    )


def create_block_mask_test(score_mod, query, key):
    block_mask = create_block_mask(
        score_mod,
        1,
        1,
        query.shape[-2],
        key.shape[-2],
        query.device,
    )
    return block_mask


TEST_ON_CUDA = (
    torch.cuda.is_available()
    and torch.utils._triton.has_triton()
    and torch.cuda.get_device_capability() >= (8, 0)
)

if TEST_ON_CUDA:
    test_device = "cuda"
    test_dtypes = (
        [torch.float32, torch.bfloat16, torch.float16]
        if PLATFORM_SUPPORTS_BF16
        else [torch.float16, torch.float32]
    )
    test_dtypes_fast = [torch.float16]
else:
    test_device = "cpu"
    torch_config_string = torch.__config__.show()
    LONG_COMPILATION_ON_CPU = False
    if "CLANG" in torch_config_string.upper():
        # if the compiler is clang, skip UT for CPU due to long compilation time found in CI
        # TODO: check reason of long compile time
        LONG_COMPILATION_ON_CPU = True

    import os

    # skip since currently flex attention requires at least `avx2` support on CPU.
    IS_PLATFORM_SUPPORTED = (
        not torch.xpu.is_available()
        and not IS_MACOS
        and torch.cpu._is_avx2_supported()
        and os.getenv("ATEN_CPU_CAPABILITY") != "default"
    )

    test_dtypes = (
        [torch.float32, torch.bfloat16]
        if torch.backends.mkldnn.is_available()
        and torch.ops.mkldnn._is_mkldnn_bf16_supported()
        else [torch.float32]
    )
    test_dtypes_fast = [torch.float32]


# --------- Useful score mod functions for testing ---------
def _causal(
    score: Tensor,
    batch: Tensor,
    head: Tensor,
    token_q: Tensor,
    token_kv: Tensor,
) -> Tensor:
    return torch.where(token_q >= token_kv, score, float("-inf"))


def _rel_bias(
    score: Tensor,
    batch: Tensor,
    head: Tensor,
    token_q: Tensor,
    token_kv: Tensor,
) -> Tensor:
    return score + (token_q - token_kv)


def _rel_causal(
    score: Tensor,
    batch: Tensor,
    head: Tensor,
    token_q: Tensor,
    token_kv: Tensor,
) -> Tensor:
    return torch.where(token_q >= token_kv, score + (token_q - token_kv), float("-inf"))


def _generate_alibi_bias(num_heads: int):
    def _alibi_bias(
        score: Tensor,
        batch: Tensor,
        head: Tensor,
        token_q: Tensor,
        token_kv: Tensor,
    ) -> Tensor:
        scale = torch.exp2(-((head + 1) * 8.0 / num_heads))
        return score + (token_kv - token_q) * scale

    return _alibi_bias


def _inverse_causal(score, b, h, m, n):
    return torch.where(m <= n, score, float("-inf"))


def _times_two(score, b, h, m, n):
    """Joint graph needed for correctness"""
    return score * 2


def _squared(score, b, h, m, n):
    """Joint graph needed for correctness"""
    return score * score


def _head_offset(dtype: torch.dtype):
    """Captured Buffer"""
    head_offset = torch.rand(H, device="cuda", dtype=dtype)

    def score_mod(score, b, h, m, n):
        return score * head_offset[h]

    return score_mod


def _trig(score, b, h, m, n):
    """Joint graph needed for correctness"""
    return torch.sin(torch.cos(score)) + torch.tan(b)


def _trig2(score, b, h, m, n):
    """Branching joint graph"""
    cos_score = torch.cos(score)
    sin_score = torch.sin(score)
    z = cos_score * sin_score + torch.tan(b)
    return z


# --------- Useful mask mod functions for testing ---------
def _causal_mask(
    batch: Tensor,
    head: Tensor,
    token_q: Tensor,
    token_kv: Tensor,
) -> Tensor:
    return token_q >= token_kv


def _inverse_causal_mask(
    batch: Tensor,
    head: Tensor,
    token_q: Tensor,
    token_kv: Tensor,
) -> Tensor:
    return token_q <= token_kv


test_score_mods = [
    _identity,
    _times_two,
    _squared,
    _causal,
    _inverse_causal,
    _rel_bias,
    _rel_causal,
    _generate_alibi_bias(8),
]

test_score_mask_mod_map = {
    _identity: noop_mask,
    _times_two: noop_mask,
    _squared: noop_mask,
    _causal: _causal_mask,
    _inverse_causal: _inverse_causal_mask,
    _rel_bias: noop_mask,
    _rel_causal: _causal_mask,
    _generate_alibi_bias(8): noop_mask,
}

captured_buffers_map = {
    "_head_offset": _head_offset,
}

B = 4
H = 8
S = 2048
D = 64

test_Hq_Hkv = [
    (4, 2),
    (4, 1),
]

test_Bq_Bkv = [
    (3, 1),
    (4, 1),
    (5, 1),
]

test_block_size = [
    128,
    256,
    (128, 256),
    (256, 128),
]


def query_key_value_clones(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    dtype: torch.dtype = None,
):
    """Clones the query, key, and value tensors and moves them to the specified dtype."""
    if dtype is None:
        dtype = query.dtype
    query_ref = query.detach().clone().to(dtype).requires_grad_(query.requires_grad)
    key_ref = key.detach().clone().to(dtype).requires_grad_(key.requires_grad)
    value_ref = value.detach().clone().to(dtype).requires_grad_(value.requires_grad)
    return query_ref, key_ref, value_ref


def batch_reserve(paged_attention: PagedAttention, target_seq_len: Tensor):
    (B,) = target_seq_len.shape
    for b in range(B):
        paged_attention.reserve(
            torch.tensor(b),
            target_seq_len[b],
        )


class TestFlexAttention(InductorTestCase):
    def setUp(self):
        super().setUp()
        self.device = test_device
        if self.device == "cpu":
            if LONG_COMPILATION_ON_CPU:
                self.skipTest(
                    "skip UT for CPU due to long compilation time found in CI"
                )
            if not IS_PLATFORM_SUPPORTED:
                self.skipTest("skip UT due to not support on those platforms")

    def _check_equal(
        self,
        golden_out: torch.Tensor,
        ref_out: torch.Tensor,
        compiled_out: torch.Tensor,
        fudge_factor: float,
        tensor_name: Optional[str] = None,
    ):
        compiled_error = (golden_out - compiled_out).abs().mean()
        ref_error = (golden_out - ref_out).abs().mean()
        if torch.isnan(compiled_error).any() or torch.isnan(ref_error).any():
            self.assertTrue(False, "Output/Grad with NaN")
        if compiled_error > ref_error * fudge_factor:
            name = tensor_name if tensor_name is not None else ""
            msg = f"{name} Compiled error {compiled_error} is greater than ref error {ref_error} by more than {fudge_factor}X."
            self.assertTrue(False, msg)

    def _check_out(
        self,
        golden_out: torch.Tensor,
        ref_out: torch.Tensor,
        compiled_out: torch.Tensor,
        is_paged_attention: bool = False,
    ):
        dtype = ref_out.dtype
        with torch.no_grad():
            # Note, it seems like we really are less accurate than the float32
            # computation, likely due to the online softmax
            if dtype == torch.float32:
                fudge_factor = 10.0
                if is_paged_attention:
                    # paged attention is less accurate since it may reorder
                    # the blocks from block mask
                    fudge_factor = 20.0
            else:
                fudge_factor = 1.1

            # Checkout output
            self._check_equal(golden_out, ref_out, compiled_out, fudge_factor, "Out")

    def _check_out_and_grad(
        self,
        golden_out: torch.Tensor,
        ref_out: torch.Tensor,
        compiled_out: torch.Tensor,
        q_gold: torch.Tensor,
        q_ref: torch.Tensor,
        q: torch.Tensor,
        k_gold: torch.Tensor,
        k_ref: torch.Tensor,
        k: torch.Tensor,
        v_gold: torch.Tensor,
        v_ref: torch.Tensor,
        v: torch.Tensor,
    ):
        dtype = ref_out.dtype
        with torch.no_grad():
            # Note, it seems like we really are less accurate than the float32
            # computation, likely due to the online softmax
            if dtype == torch.float32:
                fudge_factor = 10.0
            else:
                fudge_factor = 1.1

            # Checkout output
            self._check_equal(golden_out, ref_out, compiled_out, fudge_factor, "Out")

            # Check gradients
            q_fudge_factor = 1.0 * fudge_factor
            self._check_equal(
                q_gold.grad, q_ref.grad, q.grad, q_fudge_factor, "Grad_Query"
            )
            k_fudge_factor = 1.0 * fudge_factor
            self._check_equal(
                k_gold.grad, k_ref.grad, k.grad, k_fudge_factor, "Grad_Key"
            )
            v_fudge_factor = 1.0 * fudge_factor
            self._check_equal(
                v_gold.grad, v_ref.grad, v.grad, v_fudge_factor, "Grad_Value"
            )

    def run_test(
        self,
        score_mod: _score_mod_signature,
        dtype: torch.dtype = torch.float16,
        Q_B: int = B,
        Q_H: int = H,
        Q_S: int = S,
        Q_D: int = D,
        KV_B: Optional[int] = None,
        KV_H: Optional[int] = None,
        KV_S: Optional[int] = None,
        V_D: Optional[int] = None,
        block_mask: Optional[BlockMask] = None,
    ):
        if KV_B is None:
            KV_B = Q_B
        if KV_H is None:
            KV_H = Q_H
        if KV_S is None:
            KV_S = Q_S
        if V_D is None:
            V_D = Q_D

        if self.device == "cpu":
            test_inference_only = True
        else:
            test_inference_only = False

        q = torch.randn(
            (Q_B, Q_H, Q_S, Q_D),
            dtype=dtype,
            device=self.device,
            requires_grad=not test_inference_only,
        )
        k = torch.randn(
            (KV_B, KV_H, KV_S, Q_D),
            dtype=dtype,
            device=self.device,
            requires_grad=not test_inference_only,
        )
        v = torch.randn(
            (KV_B, KV_H, KV_S, V_D),
            dtype=dtype,
            device=self.device,
            requires_grad=not test_inference_only,
        )
        if block_mask is None:
            block_mask = create_block_mask(
                noop_mask, Q_B, Q_H, Q_S, KV_S, device=self.device
            )
        q_ref, k_ref, v_ref = query_key_value_clones(q, k, v)
        q_gold, k_gold, v_gold = query_key_value_clones(q, k, v, torch.float64)
        sdpa_partial = create_attention(
            score_mod, block_mask, enable_gqa=(not Q_H == KV_H)
        )

        compiled_sdpa = torch.compile(sdpa_partial)
        golden_out = sdpa_partial(q_gold, k_gold, v_gold)
        ref_out = sdpa_partial(q_ref, k_ref, v_ref)
        compiled_out = compiled_sdpa(q, k, v)
        if test_inference_only:
            self._check_out(
                golden_out,
                ref_out,
                compiled_out,
                is_paged_attention=False,
            )
        else:
            backward_grad = torch.randn(
                (Q_B, Q_H, Q_S, V_D), dtype=dtype, device=self.device
            )

            golden_out.backward(backward_grad.to(torch.float64))
            ref_out.backward(backward_grad)
            compiled_out.backward(backward_grad)

            self._check_out_and_grad(
                golden_out,
                ref_out,
                compiled_out,
                q_gold,
                q_ref,
                q,
                k_gold,
                k_ref,
                k,
                v_gold,
                v_ref,
                v,
            )

    def preprocess_paged_attention(
        self,
        score_mod: Optional[Callable],
        q: Tensor,
        k: Tensor,
        v: Tensor,
        block_mask,
        dtype: torch.dtype = torch.float16,
        page_size: int = 128,
    ) -> tuple[Tensor, Tensor, BlockMask, _score_mod_signature]:
        assert block_mask is not None, "Must provide block_mask"
        Q_B, Q_H, Q_S, _ = q.shape
        KV_B, KV_H, KV_S, QK_D = k.shape
        _, _, _, V_D = v.shape

        # test with different batch size
        max_batch_size = max(Q_B, KV_B) + 3

        n_pages = (KV_S + page_size - 1) // page_size * max_batch_size

        # allocate cache
        MAX_CACHED_SEQ_LEN = n_pages * page_size
        k_cache = torch.zeros(
            1,
            KV_H,
            MAX_CACHED_SEQ_LEN,
            QK_D,
            device=self.device,
            dtype=dtype,
        )
        v_cache = torch.zeros(
            1,
            KV_H,
            MAX_CACHED_SEQ_LEN,
            V_D,
            device=self.device,
            dtype=dtype,
        )

        # For testing purposes, we randomly initialize the page table, which maps
        # (batch_idx, logical_block_idx) to physical_block_idx. Specifically, PagedAttention
        # maintains a stack empty_pages of unused physical_block_idx. The `batch_reserve`
        # function grabs physical_block_idx from the top of empty_pages until there are enough
        # pages for each batch index (i.e., num pages for batch_idx >= target_seq_len[batch_idx]).
        # For example, at the first batch_reserve call, physical block indices (1,...,KV_S//4)
        # are allocated to batch index 0, and physical block indices
        # (KV_S//4+1, ..., KV_S//4 + KV_S//2) are allocated to batch index 1, etc.
        # Thus, kv tensors of batch index 1 will be scattered in the kv cache, simulating
        # a real use case of paged attention.
        paged_attention = PagedAttention(
            n_pages, page_size, max_batch_size, device=self.device
        )
        batch_reserve(
            paged_attention,
            torch.tensor(
                [KV_S // 4, KV_S // 2, KV_S // 4, KV_S // 3], device=self.device
            ),
        )
        batch_reserve(
            paged_attention,
            torch.tensor(
                [KV_S // 4, KV_S // 2, KV_S // 2, KV_S // 2], device=self.device
            ),
        )
        batch_reserve(
            paged_attention,
            torch.tensor([KV_S // 2, KV_S, KV_S // 2, KV_S], device=self.device),
        )
        batch_reserve(
            paged_attention, torch.tensor([KV_S, KV_S, KV_S, KV_S], device=self.device)
        )

        # update cache with k and v
        input_pos = (
            torch.arange(KV_S, device=self.device, dtype=torch.int32)
            .unsqueeze(0)
            .expand(KV_B, KV_S)
        )
        batch_idx = torch.arange(KV_B, device=self.device, dtype=torch.int32)
        paged_attention.assign(batch_idx, input_pos, k, v, k_cache, v_cache)

        # convert block mask and score mod
        converted_block_mask = paged_attention.convert_logical_block_mask(block_mask)
        converted_score_mod = paged_attention.get_score_mod(score_mod)
        return k_cache, v_cache, converted_block_mask, converted_score_mod

    def run_paged_attention(
        self,
        score_mod: Optional[Callable],
        q: Tensor,
        k: Tensor,
        v: Tensor,
        dtype: torch.dtype = torch.float16,
        block_mask: Optional[BlockMask] = None,
    ) -> tuple[Tensor, Tensor]:
        B, Q_H, Q_S, KV_H, KV_S = (
            q.shape[0],
            q.shape[1],
            q.shape[2],
            k.shape[1],
            k.shape[2],
        )
        if self.device == "cpu":
            test_inference_only = True
        else:
            test_inference_only = False
        if block_mask is None:
            block_mask = create_block_mask(
                noop_mask, B, 1, Q_S, KV_S, device=self.device
            )

        (
            k_cache,
            v_cache,
            converted_block_mask,
            converted_score_mod,
        ) = self.preprocess_paged_attention(
            score_mod,
            q,
            k,
            v,
            block_mask,
            dtype,
            block_mask.BLOCK_SIZE[1],
        )

        compiled_sdpa = torch.compile(flex_attention)

        # compute
        return_lse = True
        if test_inference_only:
            return_lse = False
            compiled_lse = None
            compiled_out = compiled_sdpa(
                q,
                k_cache,
                v_cache,
                return_lse=return_lse,
                block_mask=converted_block_mask,
                score_mod=converted_score_mod,
                enable_gqa=(not Q_H == KV_H),
            )

        else:
            compiled_out, compiled_lse = compiled_sdpa(
                q,
                k_cache,
                v_cache,
                return_lse=return_lse,
                block_mask=converted_block_mask,
                score_mod=converted_score_mod,
                enable_gqa=(not Q_H == KV_H),
            )
        return compiled_out, compiled_lse

    def run_test_with_paged_attention(
        self,
        score_mod: Optional[Callable] = _identity,
        dtype: torch.dtype = torch.float16,
        Q_B: int = B,
        Q_H: int = H,
        Q_S: int = S,
        QK_D: int = D,
        KV_B: int = B,
        KV_H: int = H,
        KV_S: int = S,
        V_D: int = D,
        block_mask: Optional[BlockMask] = None,
    ):
        assert Q_H % KV_H == 0
        if self.device == "cpu":
            test_inference_only = True
        else:
            test_inference_only = False
        q = torch.randn(
            (Q_B, Q_H, Q_S, QK_D), dtype=dtype, device=self.device, requires_grad=False
        )
        k = torch.randn(
            (KV_B, KV_H, KV_S, QK_D),
            dtype=dtype,
            device=self.device,
            requires_grad=False,
        )
        v = torch.randn(
            (KV_B, KV_H, KV_S, V_D),
            dtype=dtype,
            device=self.device,
            requires_grad=False,
        )
        q_ref, k_ref, v_ref = query_key_value_clones(q, k, v)
        q_gold, k_gold, v_gold = query_key_value_clones(q, k, v, torch.float64)

        if block_mask is None:
            block_mask = create_block_mask(
                noop_mask, Q_B, 1, Q_S, KV_S, device=self.device
            )

        sdpa_partial = create_attention(
            score_mod, block_mask, enable_gqa=(not Q_H == KV_H)
        )
        golden_out, golden_lse = sdpa_partial(q_gold, k_gold, v_gold, return_lse=True)
        ref_out, ref_lse = sdpa_partial(q_ref, k_ref, v_ref, return_lse=True)

        compiled_out, compiled_lse = self.run_paged_attention(
            score_mod, q, k, v, dtype, block_mask
        )
        self._check_out(
            golden_out,
            ref_out,
            compiled_out,
            is_paged_attention=True,
        )

        if not test_inference_only:
            self._check_out(
                golden_lse,
                ref_lse,
                compiled_lse,
                is_paged_attention=True,
            )

    def run_test_with_call(
        self,
        sdpa_call: Callable,
        dtype: torch.dtype = torch.float16,
        Q_B: int = B,
        Q_H: int = H,
        Q_S: int = S,
        Q_D: int = D,
        KV_B: int = B,
        KV_H: int = H,
        KV_S: int = S,
        V_D: int = D,
    ):
        if self.device == "cpu":
            test_inference_only = True
        else:
            test_inference_only = False
        q = torch.randn(
            (Q_B, Q_H, Q_S, Q_D),
            dtype=dtype,
            device=self.device,
            requires_grad=not test_inference_only,
        )
        k = torch.randn(
            (KV_B, KV_H, KV_S, Q_D),
            dtype=dtype,
            device=self.device,
            requires_grad=not test_inference_only,
        )
        v = torch.randn(
            (KV_B, KV_H, KV_S, V_D),
            dtype=dtype,
            device=self.device,
            requires_grad=not test_inference_only,
        )
        q_ref, k_ref, v_ref = query_key_value_clones(q, k, v)
        q_gold, k_gold, v_gold = query_key_value_clones(q, k, v, torch.float64)
        compiled_sdpa = torch.compile(sdpa_call)
        golden_out = sdpa_call(q_gold, k_gold, v_gold)
        ref_out = sdpa_call(q_ref, k_ref, v_ref)
        compiled_out = compiled_sdpa(q, k, v)
        if test_inference_only:
            self._check_out(
                golden_out,
                ref_out,
                compiled_out,
                is_paged_attention=False,
            )
        else:
            backward_grad = torch.randn(
                (Q_B, Q_H, Q_S, V_D), dtype=dtype, device=self.device
            )

            golden_out.backward(backward_grad.to(torch.float64))
            ref_out.backward(backward_grad)
            compiled_out.backward(backward_grad)

            self._check_out_and_grad(
                golden_out,
                ref_out,
                compiled_out,
                q_gold,
                q_ref,
                q,
                k_gold,
                k_ref,
                k,
                v_gold,
                v_ref,
                v,
            )

    def run_dynamic_test(
        self,
        score_mask_mod: tuple[Callable, Callable],
        dtype: torch.dtype = torch.float16,
        B: int = B,
        H: int = H,
        S: int = S,
        D: int = D,
    ):
        score_mod, mask_mod = score_mask_mod

        # First batch with original dimensions (B, H, S, D)
        block_mask1 = create_block_mask(mask_mod, 1, 1, S, S)
        sdpa_partial1 = create_attention(score_mod, block_mask=block_mask1)

        q1 = torch.randn((B, H, S, D), dtype=dtype, device="cuda", requires_grad=True)
        k1 = torch.randn((B, H, S, D), dtype=dtype, device="cuda", requires_grad=True)
        v1 = torch.randn((B, H, S, D), dtype=dtype, device="cuda", requires_grad=True)
        q1_ref, k1_ref, v1_ref = query_key_value_clones(q1, k1, v1)
        q1_gold, k1_gold, v1_gold = query_key_value_clones(q1, k1, v1, torch.float64)
        ref_out1 = sdpa_partial1(q1_ref, k1_ref, v1_ref)
        golden_out1 = sdpa_partial1(q1_gold, k1_gold, v1_gold)

        backward_grad1 = torch.randn((B, H, S, D), dtype=dtype, device="cuda")
        golden_out1.backward(backward_grad1.to(torch.float64))
        ref_out1.backward(backward_grad1)

        # Second batch with modified dimensions (B * 2, H, S / 2, D)
        B = int(B * 2)
        S = int(S / 2)
        block_mask2 = create_block_mask(mask_mod, 1, 1, S, S)
        sdpa_partial2 = create_attention(score_mod, block_mask=block_mask2)

        q2 = torch.randn((B, H, S, D), dtype=dtype, device="cuda", requires_grad=True)
        k2 = torch.randn((B, H, S, D), dtype=dtype, device="cuda", requires_grad=True)
        v2 = torch.randn((B, H, S, D), dtype=dtype, device="cuda", requires_grad=True)
        q2_ref, k2_ref, v2_ref = query_key_value_clones(q2, k2, v2)
        q2_gold, k2_gold, v2_gold = query_key_value_clones(q2, k2, v2, torch.float64)
        ref_out2 = sdpa_partial2(q2_ref, k2_ref, v2_ref)
        golden_out2 = sdpa_partial2(q2_gold, k2_gold, v2_gold)

        backward_grad2 = torch.randn((B, H, S, D), dtype=dtype, device="cuda")
        golden_out2.backward(backward_grad2.to(torch.float64))
        ref_out2.backward(backward_grad2)

        # Third batch with modified dimensions (B * 2, H, S / 4, D)
        S = int(S / 2)
        block_mask3 = create_block_mask(mask_mod, 1, 1, S, S)
        sdpa_partial3 = create_attention(score_mod, block_mask=block_mask3)

        q3 = torch.randn((B, H, S, D), dtype=dtype, device="cuda", requires_grad=True)
        k3 = torch.randn((B, H, S, D), dtype=dtype, device="cuda", requires_grad=True)
        v3 = torch.randn((B, H, S, D), dtype=dtype, device="cuda", requires_grad=True)
        q3_ref, k3_ref, v3_ref = query_key_value_clones(q3, k3, v3)
        q3_gold, k3_gold, v3_gold = query_key_value_clones(q3, k3, v3, torch.float64)
        ref_out3 = sdpa_partial3(q3_ref, k3_ref, v3_ref)
        golden_out3 = sdpa_partial3(q3_gold, k3_gold, v3_gold)

        backward_grad3 = torch.randn((B, H, S, D), dtype=dtype, device="cuda")
        golden_out3.backward(backward_grad3.to(torch.float64))
        ref_out3.backward(backward_grad3)

        # Clear dynamo counters
        torch._dynamo.reset()

        # First compilation with original dimensions
        backend = torch._dynamo.testing.CompileCounterWithBackend("inductor")
        compiled_sdpa1 = torch.compile(sdpa_partial1, backend=backend, dynamic=True)
        compiled_out1 = compiled_sdpa1(q1, k1, v1)
        compiled_out1.backward(backward_grad1)

        self._check_out_and_grad(
            golden_out1,
            ref_out1,
            compiled_out1,
            q1_gold,
            q1_ref,
            q1,
            k1_gold,
            k1_ref,
            k1,
            v1_gold,
            v1_ref,
            v1,
        )
        self.assertEqual(backend.frame_count, 1)

        # Second compilation with new dimensions
        compiled_sdpa2 = torch.compile(sdpa_partial2, backend=backend, dynamic=True)
        compiled_out2 = compiled_sdpa2(q2, k2, v2)
        compiled_out2.backward(backward_grad2)

        self._check_out_and_grad(
            golden_out2,
            ref_out2,
            compiled_out2,
            q2_gold,
            q2_ref,
            q2,
            k2_gold,
            k2_ref,
            k2,
            v2_gold,
            v2_ref,
            v2,
        )
        self.assertEqual(backend.frame_count, 1)

        # Third compilation with new dimensions
        compiled_sdpa3 = torch.compile(sdpa_partial3, backend=backend, dynamic=True)
        compiled_out3 = compiled_sdpa3(q3, k3, v3)
        compiled_out3.backward(backward_grad3)

        self._check_out_and_grad(
            golden_out3,
            ref_out3,
            compiled_out3,
            q3_gold,
            q3_ref,
            q3,
            k3_gold,
            k3_ref,
            k3,
            v3_gold,
            v3_ref,
            v3,
        )
        self.assertEqual(backend.frame_count, 1)

    def run_automatic_dynamic_test(
        self,
        score_mod: Callable,
        dtype: torch.dtype = torch.float16,
        B: int = B,
        H: int = H,
        S: int = S,
        D: int = D,
    ):
        if self.device == "cpu":
            test_inference_only = True
        else:
            test_inference_only = False
        block_mask1 = create_block_mask(noop_mask, 1, 1, S, S, device=self.device)
        sdpa_partial1 = create_attention(score_mod, block_mask=block_mask1)
        # The first eager batch, shape (B, H, S, D)
        q1 = torch.randn(
            (B, H, S, D),
            dtype=dtype,
            device=self.device,
            requires_grad=not test_inference_only,
        )
        k1 = torch.randn(
            (B, H, S, D),
            dtype=dtype,
            device=self.device,
            requires_grad=not test_inference_only,
        )
        v1 = torch.randn(
            (B, H, S, D),
            dtype=dtype,
            device=self.device,
            requires_grad=not test_inference_only,
        )
        golden_out1 = sdpa_partial1(
            q1.to(torch.float64), k1.to(torch.float64), v1.to(torch.float64)
        )
        ref_out1 = sdpa_partial1(q1, k1, v1)

        # The second eager batch, shape (B * 2, H, S / 2, D)
        B = int(B * 2)
        S = int(S / 2)
        block_mask2 = create_block_mask(noop_mask, 1, 1, S, S, device=self.device)
        sdpa_partial2 = create_attention(score_mod, block_mask=block_mask2)
        q2 = torch.randn(
            (B, H, S, D),
            dtype=dtype,
            device=self.device,
            requires_grad=not test_inference_only,
        )
        k2 = torch.randn(
            (B, H, S, D),
            dtype=dtype,
            device=self.device,
            requires_grad=not test_inference_only,
        )
        v2 = torch.randn(
            (B, H, S, D),
            dtype=dtype,
            device=self.device,
            requires_grad=not test_inference_only,
        )
        golden_out2 = sdpa_partial2(
            q2.to(torch.float64), k2.to(torch.float64), v2.to(torch.float64)
        )
        ref_out2 = sdpa_partial2(q2, k2, v2)

        # The third eager batch, shape (B * 4, H, S / 4, D)
        B = int(B * 2)
        S = int(S / 2)
        block_mask3 = create_block_mask(noop_mask, 1, 1, S, S, device=self.device)
        sdpa_partial3 = create_attention(score_mod, block_mask=block_mask3)
        q3 = torch.randn(
            (B, H, S, D),
            dtype=dtype,
            device=self.device,
            requires_grad=not test_inference_only,
        )
        k3 = torch.randn(
            (B, H, S, D),
            dtype=dtype,
            device=self.device,
            requires_grad=not test_inference_only,
        )
        v3 = torch.randn(
            (B, H, S, D),
            dtype=dtype,
            device=self.device,
            requires_grad=not test_inference_only,
        )
        golden_out3 = sdpa_partial3(
            q3.to(torch.float64), k3.to(torch.float64), v3.to(torch.float64)
        )
        ref_out3 = sdpa_partial3(q3, k3, v3)

        # Need to clear dynamo counters, since flex attention eager mode also uses dynamo tracing.
        # We check dynamo counters["frames"]["ok"] to ensure:
        # 1, the first batch is compiled with static shape
        # 2, the second batch is compiled with dynamic shape
        # 3, no re-compilation in the third batch
        torch._dynamo.reset()

        # Note, it seems like we really are less accurate than the float32
        # computation, likely due to the online softmax
        if dtype == torch.float32:
            fudge_factor = 10.0
        else:
            fudge_factor = 1.1

        # The first batch.
        backend = torch._dynamo.testing.CompileCounterWithBackend("inductor")
        compiled_out1 = torch.compile(sdpa_partial1, backend=backend)(q1, k1, v1)
        self._check_equal(golden_out1, ref_out1, compiled_out1, fudge_factor)
        self.assertEqual(backend.frame_count, 1)

        # The second batch (automatic dynamic).
        compiled_out2 = torch.compile(sdpa_partial2, backend=backend)(q2, k2, v2)
        self._check_equal(golden_out2, ref_out2, compiled_out2, fudge_factor)
        self.assertEqual(backend.frame_count, 2)

        # The third batch (no re-compilation).
        compiled_out3 = torch.compile(sdpa_partial3, backend=backend)(q3, k3, v3)
        self._check_equal(golden_out3, ref_out3, compiled_out3, fudge_factor)
        self.assertEqual(backend.frame_count, 2)

    @common_utils.parametrize("dtype", test_dtypes)
    @common_utils.parametrize("score_mod", test_score_mods)
    def test_builtin_score_mods(self, dtype: torch.dtype, score_mod: Callable):
        self.run_test(score_mod, dtype)
        self.run_test_with_paged_attention(score_mod, dtype)

    @running_on_a100_only
    @common_utils.parametrize("dtype", test_dtypes_fast)
    @common_utils.parametrize("score_mod", test_score_mods)
    def test_builtin_score_mods_seqlen_lt_default_sparse_block_size(
        self, dtype: torch.dtype, score_mod: Callable
    ):
        # _DEFAULT_SPARSE_BLOCK_SIZE is 128
        attention = functools.partial(
            flex_attention,
            score_mod=score_mod,
            kernel_options={"FORCE_USE_FLEX_ATTENTION": True},
        )
        self.run_test_with_call(attention, dtype, B, H, 64, D, B, H, 64, D)

    @running_on_a100_only
    @common_utils.parametrize("dtype", test_dtypes_fast)
    @common_utils.parametrize("score_mod", test_score_mods)
    def test_builtin_score_mods_seqlen_lt_custom_sparse_block_size(
        self, dtype: torch.dtype, score_mod: Callable
    ):
        def causal_mask(b, h, q, kv):
            return q >= kv

        block_mask = create_block_mask(causal_mask, 1, 1, 64, 64, BLOCK_SIZE=256)
        attention = functools.partial(
            flex_attention,
            score_mod=score_mod,
            block_mask=block_mask,
            kernel_options={"FORCE_USE_FLEX_ATTENTION": True},
        )
        self.run_test_with_call(attention, dtype, B, H, 64, D, B, H, 64, D)

    @supported_platform
    @common_utils.parametrize("dtype", test_dtypes_fast)
    @common_utils.parametrize("score_mask_mod", test_score_mask_mod_map.items())
    def test_builtin_score_mods_dynamic(
        self, dtype: torch.dtype, score_mask_mod: tuple[Callable, Callable]
    ):
        self.run_dynamic_test(score_mask_mod, dtype)

    @common_utils.parametrize("dtype", test_dtypes_fast)
    @common_utils.parametrize("score_mod", test_score_mods)
    def test_builtin_score_mods_automatic_dynamic(
        self, dtype: torch.dtype, score_mod: Callable
    ):
        self.run_automatic_dynamic_test(score_mod, dtype)

    @common_utils.parametrize("dtype", test_dtypes_fast)
    @common_utils.parametrize("score_mod", test_score_mods)
    def test_builtin_score_mods_different_seqlen(
        self, dtype: torch.dtype, score_mod: Callable
    ):
        inputs = (
            score_mod,
            dtype,
            B,
            H,
            S // 2,  # Seqlen of Q is different from seqlen of K/V
            D,
            B,
            H,
            S,
            D,
        )
        self.run_test(*inputs)
        self.run_test_with_paged_attention(*inputs)

    @common_utils.parametrize("dtype", test_dtypes)
    @common_utils.parametrize("score_mod", test_score_mods)
    @common_utils.parametrize("BLOCK_SIZE", test_block_size)
    def test_builtin_score_mods_different_block_size(
        self,
        dtype: torch.dtype,
        score_mod: Callable,
        BLOCK_SIZE: Union[int, tuple[int, int]],
    ):
        block_mask = create_block_mask(
            noop_mask, B, H, S, S, BLOCK_SIZE=BLOCK_SIZE, device=self.device
        )
        self.run_test(score_mod, dtype, block_mask=block_mask)
        self.run_test_with_paged_attention(score_mod, dtype, block_mask=block_mask)

    @common_utils.parametrize("dtype", test_dtypes_fast)
    @common_utils.parametrize("batch_dims", test_Bq_Bkv)
    @common_utils.parametrize("head_dims", test_Hq_Hkv)
    @common_utils.parametrize("score_mod", test_score_mods)
    def test_kv_batch_broadcast(
        self,
        dtype: torch.dtype,
        batch_dims: tuple[int, int],
        head_dims: tuple[int, int],
        score_mod: Callable,
    ):
        Hq, Hkv = head_dims
        assert Hq % Hkv == 0

        Bq, Bkv = batch_dims
        assert Bq > 1 and Bkv == 1

        block_mask = create_block_mask(noop_mask, Bq, 1, S, S, device=self.device)

        self.run_test(
            score_mod,
            dtype,
            Bq,
            Hq,
            S,
            D,
            Bkv,
            Hkv,
            S,
            D,
            block_mask,
        )

    @common_utils.parametrize("dtype", test_dtypes_fast)
    @common_utils.parametrize("batch_dims", test_Bq_Bkv)
    @common_utils.parametrize("head_dims", test_Hq_Hkv)
    @common_utils.parametrize("score_mod", test_score_mods)
    def test_kv_batch_broadcast_causal_mask(
        self,
        dtype: torch.dtype,
        batch_dims: tuple[int, int],
        head_dims: tuple[int, int],
        score_mod: Callable,
    ):
        Hq, Hkv = head_dims
        assert Hq % Hkv == 0

        Bq, Bkv = batch_dims
        assert Bq > 1 and Bkv == 1

        def mask_mod(b, h, q, kv):
            return q >= kv

        block_mask = create_block_mask(mask_mod, Bq, 1, S, S, device=self.device)
        attention = functools.partial(
            flex_attention, block_mask=block_mask, enable_gqa=(not Hq == Hkv)
        )

        self.run_test_with_call(attention, dtype, Bq, Hq, S, D, Bkv, Hkv, S, D)

    @common_utils.parametrize("dtype", test_dtypes_fast)
    @common_utils.parametrize("score_mod", test_score_mods)
    def test_GQA(self, dtype: torch.dtype, score_mod: Callable):
        inputs = (
            score_mod,
            dtype,
            B,
            H * 4,  # Hq = 4*Hkv.
            S // 8,
            D,
            B,
            H,
            S,
            D,
        )
        self.run_test(*inputs)
        self.run_test_with_paged_attention(*inputs)

    test_strides = [
        ((H * S * D, S * D, D, 1), 997),  # offset
        ((H * D, D, B * H * D, 1), 499),  # transposed dimensions
        ((H * S * D, D, H * D, 1), 0),  # heads/sequence transposed
        (
            (S * (D + 1), B * S * (D + 1), (D + 1), 1),
            293,
        ),  # additional buffer on one dim
        (
            (1, D, (B + 1) * (H + 1) * D, 1),
            97,
        ),  # additional buffer on multiple dim + shared dimension
    ]

    @supported_platform
    @common_utils.parametrize("dtype", test_dtypes_fast)
    @common_utils.parametrize(
        "q_s", test_strides[:2]
    )  # TODO: fix layout for query braodcasting
    @common_utils.parametrize(
        "k_s,v_s",
        [
            (test_strides[0], test_strides[0]),
            (test_strides[0], test_strides[1]),
            (test_strides[2], test_strides[3]),
            (test_strides[3], test_strides[1]),
            # (test_strides[2], test_strides[4]), # TODO: Doesn't work for
            # broadcasting reasons i think
        ],
    )
    @common_utils.parametrize("do_s", test_strides[:3])
    def test_strided_inputs(self, dtype: torch.dtype, q_s, k_s, v_s, do_s):
        q1 = torch.randn((B * H * S * D * 2), dtype=dtype, device="cuda")
        k1 = torch.randn((B * H * S * D * 2), dtype=dtype, device="cuda")
        v1 = torch.randn((B * H * S * D * 2), dtype=dtype, device="cuda")
        do1 = torch.randn((B * H * S * D * 2), dtype=dtype, device="cuda")

        q_shape = (B, H, S // 2, D)
        k_shape = (B, H, S, D)
        v_shape = (B, H, S, D)
        do_shape = (B, H, S // 2, D)

        def coerce_to_strides(val, shape, strides):
            strides, offset = strides
            val_max = [x * (y - 1) for x, y in zip(strides, shape)]
            assert sum(val_max) + offset < B * H * S * D * 2
            assert strides[-1] == 1
            return torch.as_strided(val, shape, strides, offset).requires_grad_(True)

        q = coerce_to_strides(q1, q_shape, q_s)
        k = coerce_to_strides(k1, k_shape, k_s)
        v = coerce_to_strides(v1, v_shape, v_s)
        do = coerce_to_strides(do1, do_shape, do_s)

        block_mask = _create_empty_block_mask(q, k)
        score_mod = _generate_alibi_bias(8)
        sdpa_partial = create_attention(score_mod=score_mod, block_mask=block_mask)
        compiled_sdpa = torch.compile(sdpa_partial)
        ref_out = sdpa_partial(q, k, v)
        compiled_out = compiled_sdpa(q, k, v)

        tolerance = Tolerances(atol=2e-1, rtol=2e-1)
        torch.testing.assert_close(
            ref_out, compiled_out, atol=tolerance.atol, rtol=tolerance.rtol
        )
        ref_out.backward(do)
        ref_grads = [q.grad, k.grad, v.grad]
        q.grad = None
        k.grad = None
        v.grad = None

        compiled_out.backward(do)
        compiled_grads = [q.grad, k.grad, v.grad]
        q.grad = None
        k.grad = None
        v.grad = None
        torch.testing.assert_close(
            compiled_grads[0], ref_grads[0], atol=tolerance.atol, rtol=tolerance.rtol
        )
        torch.testing.assert_close(
            compiled_grads[1], ref_grads[1], atol=tolerance.atol, rtol=tolerance.rtol
        )
        torch.testing.assert_close(
            compiled_grads[2], ref_grads[2], atol=tolerance.atol, rtol=tolerance.rtol
        )

        # test paged attention which does not support backward
        q.requires_grad, k.requires_grad, v.requires_grad = False, False, False
        paged_compiled_out, _ = self.run_paged_attention(score_mod, q, k, v, dtype)
        torch.testing.assert_close(
            ref_out, paged_compiled_out, atol=tolerance.atol, rtol=tolerance.rtol
        )

    @supported_platform
    def test_doc_mask_sparse(self):
        document_id = torch.zeros(S, dtype=torch.int, device="cuda")
        for i in range(0, S, 256):
            document_id[i : i + 256] = i // 256

        def document_masking_causal(score, b, h, q_idx, kv_idx):
            causal_mask = q_idx >= kv_idx
            document_mask = document_id[q_idx] == document_id[kv_idx]
            return torch.where(causal_mask & document_mask, score, -float("inf"))

        self.run_test(document_masking_causal, torch.float16)
        self.run_test_with_paged_attention(document_masking_causal, torch.float16)

    @supported_platform
    def test_index_multiple(self):
        bias = torch.randn(B, S, device="cuda")

        def index_multiple(score, b, h, q_idx, kv_idx):
            return score + bias[b][q_idx]

        self.run_test(index_multiple, torch.float16)
        self.run_test_with_paged_attention(index_multiple, torch.float16)

    @supported_platform
    def test_index_weird1(self):
        bias = torch.randn(4, B, H, S, device="cuda")

        def index_weird1(score, b, h, q_idx, kv_idx):
            return score + bias[0][b, h][q_idx]

        self.run_test(index_weird1, torch.float16)
        self.run_test_with_paged_attention(index_weird1, torch.float16)

    @supported_platform
    def test_index_weird2(self):
        bias = torch.randn(B, H, 4, S, device="cuda")
        which_bias = torch.tensor(0, device="cuda")

        def index_weird2(score, b, h, q_idx, kv_idx):
            return score + bias[b][h][which_bias, q_idx]

        self.run_test(index_weird2, torch.float16)
        self.run_test_with_paged_attention(index_weird2, torch.float16)

    @supported_platform
    @common_utils.parametrize("dtype", test_dtypes)
    def test_skip_odd_keys(self, dtype: torch.dtype):
        def score_mod(score, b, h, q, kv):
            return torch.where(kv % 2 == 0, score, float("-inf"))

        self.run_test(score_mod, dtype)
        self.run_test_with_paged_attention(score_mod, dtype)

    @supported_platform
    @common_utils.parametrize("dtype", test_dtypes)
    def test_function_composition(self, dtype: torch.dtype):
        def score_mod_1(score, b, h, m, n):
            return score + (m - n)

        def score_mod_2(score, b, h, m, n):
            return torch.where(m <= n, score, float("-inf"))

        def composed_score_mod(score, b, h, m, n):
            return score_mod_2(score_mod_1(score, b, h, m, n), b, h, m, n)

        self.run_test(composed_score_mod, dtype)
        self.run_test_with_paged_attention(composed_score_mod, dtype)

    @supported_platform
    @common_utils.parametrize("dtype", test_dtypes)
    def test_captured_buffers_all_dims(self, dtype: torch.dtype):
        head_scale = torch.randn(H, device="cuda")
        batch_scale = torch.randn(B, device="cuda")
        tok_scale = torch.randn(S, device="cuda")

        def all_bias(score, batch, head, token_q, token_kv):
            score = score + tok_scale[token_q]
            score = score + batch_scale[batch]
            score = score + head_scale[head]
            return score

        self.run_test(all_bias, dtype)
        self.run_test_with_paged_attention(all_bias, dtype)

    @supported_platform
    @common_utils.parametrize("dtype", test_dtypes_fast)
    def test_seq_masking(self, dtype):
        seq_idx = torch.zeros(S, device="cuda", dtype=torch.bool)
        seq_idx[S // 2 :] = 1

        def seq_mask_mod(score, b, h, q, kv):
            return torch.where(seq_idx[q] == seq_idx[kv], score, float("-inf"))

        self.run_test(seq_mask_mod, dtype)
        self.run_test_with_paged_attention(seq_mask_mod, dtype)

    @supported_platform
    @common_utils.parametrize("dtype", test_dtypes_fast)
    def test_load_from_bias_seq_only(self, dtype):
        bias = torch.randn(S, S, device="cuda", dtype=dtype)

        def bias_mod(score, b, h, q, kv):
            return score + bias[q, kv]

        self.run_test(bias_mod, dtype)
        self.run_test_with_paged_attention(bias_mod, dtype)

    @supported_platform
    @common_utils.parametrize("dtype", test_dtypes_fast)
    def test_load_from_bias_seq_batch(self, dtype):
        bias = torch.randn(B, S, S, device="cuda", dtype=dtype)

        def bias_mod(score, b, h, q, kv):
            return score + bias[b, q, kv]

        self.run_test(bias_mod, dtype)
        self.run_test_with_paged_attention(bias_mod, dtype)

    @supported_platform
    def test_load_from_view_buffer(self):
        dtype = torch.float16
        device = "cuda"
        W = 8

        class SimpleAttention(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.rel_pos_h = torch.randn(2 * H - 1, D, device=device, dtype=dtype)

            def forward(self, q, k, v):
                q = q.view(B * H, H * W, -1)
                score_mod = self.generate_score_mod(q)
                q = q.view(B, H, H * W, -1)
                return flex_attention(q, k, v, score_mod=score_mod)

            def generate_score_mod(self, q):
                rel_h = self.add_decomposed_rel_pos(q)
                rel_h = rel_h.view(
                    B, H, rel_h.size(1), rel_h.size(2), rel_h.size(3)
                ).squeeze(-1)

                def score_mod(score, batch, head, q_idx, k_idx):
                    h_idx = k_idx // W
                    return score + rel_h[batch, head, q_idx, h_idx]

                return score_mod

            @torch.no_grad()
            def add_decomposed_rel_pos(self, q):
                q_coords = torch.arange(H, device=self.rel_pos_h.device)[:, None]
                k_coords = torch.arange(H, device=self.rel_pos_h.device)[None, :]
                relative_coords = (q_coords - k_coords) + (H - 1)
                Rh = self.rel_pos_h[relative_coords.long()]
                r_q = q.reshape(B * H, H, W, D)
                rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
                return rel_h.reshape(B * H, H * W, H, 1)

        m = SimpleAttention().to(device).eval()
        m = torch.compile(m, mode="max-autotune", fullgraph=True)
        q = torch.randn(B, H, H * W, D, device=device, dtype=dtype, requires_grad=True)
        k = torch.randn(B, H, H * W, D, device=device, dtype=dtype, requires_grad=True)
        v = torch.randn(B, H, H * W, D, device=device, dtype=dtype, requires_grad=True)
        out = m(q, k, v)
        out.sum().backward()

    @supported_platform
    @common_utils.parametrize("dtype", test_dtypes_fast)
    def test_load_from_bias_head_seq_batch(self, dtype):
        bias = torch.randn(B, H, S, S, device="cuda", dtype=dtype)

        def bias_mod(score, b, h, q, kv):
            return score + bias[b, h, q, kv]

        self.run_test(bias_mod, dtype)
        self.run_test_with_paged_attention(bias_mod, dtype)

    @supported_platform
    @common_utils.parametrize("dtype", test_dtypes_fast)
    def test_load_rel_bias(self, dtype):
        rel_bias = torch.randn(2 * S, device="cuda", dtype=dtype)

        def bias_mod(score, b, h, q, kv):
            return score + rel_bias[(q - kv) + S]

        self.run_test(bias_mod, dtype)
        self.run_test_with_paged_attention(bias_mod, dtype)

    @supported_platform
    @common_utils.parametrize("dtype", test_dtypes_fast)
    def test_dependent_causal_bidirectional(self, dtype):
        num_bidirectional = torch.randint(0, S, (B,), device="cuda", dtype=torch.int32)

        def bias_mod(score, b, h, q, kv):
            causal_attention = q >= kv
            cur_num_bidirectional = num_bidirectional[b]
            bidirectional_attention_on_video = (q <= cur_num_bidirectional) & (
                kv <= cur_num_bidirectional
            )
            return torch.where(
                bidirectional_attention_on_video | causal_attention,
                score,
                -float("inf"),
            )

        self.run_test(bias_mod, dtype)
        self.run_test_with_paged_attention(bias_mod, dtype)

    @supported_platform
    @common_utils.parametrize("dtype", test_dtypes_fast)
    def test_natten_2d(self, dtype):
        H = 32
        W = S // H
        WINDOW = 3
        assert W * H == S

        def get_x_y(idx):
            # This should be a floor divide, but we don't support that properly
            return idx / W, idx % W

        def natten_mask(score, b, h, q, kv):
            q_x, q_y = get_x_y(q)
            kv_x, kv_y = get_x_y(kv)
            return torch.where(
                ((q_x - kv_x).abs() <= WINDOW) | ((q_y - kv_y).abs() <= WINDOW),
                score,
                float("-inf"),
            )

        self.run_test(natten_mask, dtype)
        self.run_test_with_paged_attention(natten_mask, dtype)

    @supported_platform
    @common_utils.parametrize("dtype", test_dtypes_fast)
    def test_subgraph_respect_decompostion(self, dtype):
        from torch._decomp import core_aten_decompositions
        from torch.fx.experimental.proxy_tensor import make_fx

        def score_mod_func(score, b, h, q, kv):
            return score - q // (1 + kv)

        make_tensor = functools.partial(
            torch.randn,
            (2, 2, 128, 4),
            device="cuda",
            dtype=torch.float64,
            requires_grad=True,
        )
        query, key, value = make_tensor(), make_tensor(), make_tensor()
        # floor_div is not decomposed in decompostion_table is empty
        attention = functools.partial(flex_attention, score_mod=score_mod_func)
        gm = make_fx(attention, decomposition_table={})(query, key, value)
        self.assertExpectedInline(
            gm.sdpa_score0.code.strip(),
            """\
def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1):
    add = torch.ops.aten.add.Tensor(arg4_1, 1);  arg4_1 = None
    floor_divide = torch.ops.aten.floor_divide.default(arg3_1, add);  arg3_1 = add = None
    sub = torch.ops.aten.sub.Tensor(arg0_1, floor_divide);  arg0_1 = floor_divide = None
    return sub""",
        )

        # floor_div is decomposed for core_aten_decompositions
        gm = make_fx(attention, decomposition_table=core_aten_decompositions())(
            query, key, value
        )
        self.assertExpectedInline(
            gm.sdpa_score0.code.strip(),
            """\
def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1):
    add = torch.ops.aten.add.Tensor(arg4_1, 1);  arg4_1 = None
    div = torch.ops.aten.div.Tensor_mode(arg3_1, add, rounding_mode = 'floor');  arg3_1 = add = None
    sub = torch.ops.aten.sub.Tensor(arg0_1, div);  arg0_1 = div = None
    return sub""",
        )

    @supported_platform
    @common_utils.parametrize("dtype", test_dtypes_fast)
    def test_silu_on_score(self, dtype):
        def silu_score(score, b, h, q, kv):
            return torch.nn.functional.silu(score)

        self.run_test(silu_score, dtype)
        self.run_test_with_paged_attention(silu_score, dtype)

    @supported_platform
    @common_utils.parametrize("dtype", test_dtypes_fast)
    def test_padded_dense_causal(self, dtype):
        seq_len = torch.arange(B, device="cuda", dtype=torch.int32) + 1

        def create_padded_dense_wrapper(orig_score_mod):
            def njt_score_mod(qk, b, h, q, kv):
                return torch.where(
                    qk <= seq_len[b], orig_score_mod(qk, b, h, q, kv), -float("inf")
                )

            return njt_score_mod

        causal_njt = create_padded_dense_wrapper(_causal)

        self.run_test(causal_njt, dtype)

    @supported_platform
    @common_utils.parametrize("dtype", test_dtypes_fast)
    def test_captured_scale(self, dtype):
        scale = torch.ones((), device="cuda", dtype=torch.int32)

        def score_mod_scale(qk, b, h, q, kv):
            return qk + scale

        self.run_test(score_mod_scale, dtype)
        self.run_test_with_paged_attention(score_mod_scale, dtype)

    @supported_platform
    @common_utils.parametrize("dtype", test_dtypes_fast)
    def test_recompile_changed_score_mod(self, dtype):
        scale = torch.ones((), device="cuda", dtype=torch.int32)
        ADD = True

        def score_mod_scale(qk, b, h, q, kv):
            if ADD:
                return qk + scale
            else:
                return qk * scale

        self.run_test(score_mod_scale, dtype)
        self.run_test_with_paged_attention(score_mod_scale, dtype)

        ADD = False
        self.run_test(score_mod_scale, dtype)
        self.run_test_with_paged_attention(score_mod_scale, dtype)

    @supported_platform
    @expectedFailure  # If we capture a tensor then we can perform a reduction on it, and that shouldn't be allowed
    @common_utils.parametrize("dtype", test_dtypes_fast)
    def test_captured_reduction(self, dtype):
        scale = torch.randn((B, 8), device="cuda")

        def score_mod_scale(qk, b, h, q, kv):
            return qk + scale[b].sum(dim=-1)

        self.run_test(score_mod_scale, dtype)

    @supported_platform
    def test_multiple_score_mod_calls(self):
        query = torch.randn((1, 8, 1024, 64), dtype=torch.float32, device="cuda")
        keys = [
            torch.randn((1, 8, 1024, 64), dtype=torch.float32, device="cuda")
            for _ in range(2)
        ]
        values = [
            torch.randn((1, 8, 1024, 64), dtype=torch.float32, device="cuda")
            for _ in range(2)
        ]

        def scoremod_1(qk, b, h, q, kv):
            return qk + (q - kv)

        def scoremod_2(qk, b, h, q, kv):
            return torch.where(q >= kv, qk, -float("inf"))

        def f(q, k1, k2, v1, v2):
            q2 = flex_attention(q, k1, v1, score_mod=scoremod_1)
            return flex_attention(q2, k2, v2, score_mod=scoremod_2)

        out = f(query, *keys, *values)
        out2 = torch.compile(f)(query, *keys, *values)
        tolerance = Tolerances(atol=2e-1, rtol=2e-1)
        torch.testing.assert_close(out, out2, atol=tolerance.atol, rtol=tolerance.rtol)

    @supported_platform
    def test_multiple_mask_calls(self):
        if TEST_WITH_ROCM:
            self.skipTest(
                "ROCM BUG SEE: https://github.com/pytorch/pytorch/issues/140855"
            )
        # Create inputs
        query = torch.randn(
            (1, 4, 512, 64), dtype=torch.float32, device="cuda", requires_grad=True
        )
        key = torch.randn(
            (1, 4, 512, 64), dtype=torch.float32, device="cuda", requires_grad=True
        )
        value = torch.randn(
            (1, 4, 512, 64), dtype=torch.float32, device="cuda", requires_grad=True
        )

        window_size = 32

        def causal_mask(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx

        def causal_mask_slidewindow_mod(b, h, q_idx, kv_idx):
            return (q_idx >= kv_idx) & (q_idx <= kv_idx + window_size)

        mask1 = create_block_mask(causal_mask, 1, None, 512, 512, _compile=False)
        mask2 = create_block_mask(
            causal_mask_slidewindow_mod, 1, None, 512, 512, _compile=False
        )

        def f(q, k, v):
            out1 = flex_attention(q, k, v, block_mask=mask1)
            out2 = flex_attention(q, k, v, block_mask=mask2)
            return out1 + out2

        f_compiled = torch.compile(f, fullgraph=True)

        out = f(query, key, value)
        out_compiled = f_compiled(query, key, value)

        grads = torch.autograd.grad((out,), (query, key, value), torch.ones_like(out))
        grads_compile = torch.autograd.grad(
            (out_compiled,), (query, key, value), torch.ones_like(out_compiled)
        )

        for grad, grad_compiled in zip(grads, grads_compile):
            torch.testing.assert_close(grad, grad_compiled, atol=3e-2, rtol=3e-2)

    @supported_platform
    def test_multiple_score_mod_calls2(self):
        query = torch.randn((1, 8, 1024, 64), dtype=torch.float32, device="cuda")
        keys = [
            torch.randn((1, 8, 1024, 64), dtype=torch.float32, device="cuda")
            for _ in range(3)
        ]
        values = [
            torch.randn((1, 8, 1024, 64), dtype=torch.float32, device="cuda")
            for _ in range(3)
        ]

        def scoremod_1(qk, b, h, q, kv):
            return qk + (q - kv)

        def scoremod_2(qk, b, h, q, kv):
            return torch.where(q >= kv, qk, -float("inf"))

        attention1 = functools.partial(flex_attention, score_mod=scoremod_1)

        def f(q, k1, k2, k3, v1, v2, v3):
            q2 = attention1(q, k1, v1)
            q3 = flex_attention(q2, k2, v2, score_mod=scoremod_2)
            return flex_attention(q3, k3, v3, score_mod=scoremod_1)

        out = f(query, *keys, *values)
        out2 = torch.compile(f)(query, *keys, *values)
        self.assertTrue((out - out2).abs().mean() < 1e-2)

    @supported_platform
    def test_multiple_score_mod_calls_paged_attention(self):
        query = torch.randn((1, 8, 1024, 64), dtype=torch.float32, device="cuda")
        keys = [
            torch.randn((1, 8, 1024, 64), dtype=torch.float32, device="cuda")
            for _ in range(2)
        ]
        values = [
            torch.randn((1, 8, 1024, 64), dtype=torch.float32, device="cuda")
            for _ in range(2)
        ]

        def scoremod_1(qk, b, h, q, kv):
            return qk + (q - kv)

        def scoremod_2(qk, b, h, q, kv):
            return torch.where(q >= kv, qk, -float("inf"))

        def f(q, k1, k2, v1, v2):
            q2 = flex_attention(q, k1, v1, score_mod=scoremod_1)
            return flex_attention(q2, k2, v2, score_mod=scoremod_2)

        eager_out = f(query, *keys, *values)

        block_mask = create_block_mask(noop_mask, 1, 1, 1024, 1024)

        (
            k_cache1,
            v_cache1,
            converted_block_mask1,
            converted_score_mod1,
        ) = self.preprocess_paged_attention(
            scoremod_1, query, keys[0], values[0], block_mask, torch.float32
        )
        (
            k_cache2,
            v_cache2,
            converted_block_mask2,
            converted_score_mod2,
        ) = self.preprocess_paged_attention(
            scoremod_2, query, keys[1], values[1], block_mask, torch.float32
        )

        def paged_f(q, k1, k2, v1, v2):
            q2 = flex_attention(
                q,
                k1,
                v1,
                score_mod=converted_score_mod1,
                block_mask=converted_block_mask1,
            )
            return flex_attention(
                q2,
                k2,
                v2,
                score_mod=converted_score_mod2,
                block_mask=converted_block_mask2,
            )

        compiled_out = torch.compile(paged_f)(
            query, k_cache1, k_cache2, v_cache1, v_cache2
        )
        tolerance = Tolerances(atol=2e-1, rtol=2e-1)
        torch.testing.assert_close(
            eager_out, compiled_out, atol=tolerance.atol, rtol=tolerance.rtol
        )

    @supported_platform
    def test_multiple_score_mod_calls2_paged_attention(self):
        query = torch.randn((1, 8, 1024, 64), dtype=torch.float32, device="cuda")
        keys = [
            torch.randn((1, 8, 1024, 64), dtype=torch.float32, device="cuda")
            for _ in range(3)
        ]
        values = [
            torch.randn((1, 8, 1024, 64), dtype=torch.float32, device="cuda")
            for _ in range(3)
        ]

        def scoremod_1(qk, b, h, q, kv):
            return qk + (q - kv)

        def scoremod_2(qk, b, h, q, kv):
            return torch.where(q >= kv, qk, -float("inf"))

        attention1 = functools.partial(flex_attention, score_mod=scoremod_1)

        def f(q, k1, k2, k3, v1, v2, v3):
            q2 = attention1(q, k1, v1)
            q3 = flex_attention(q2, k2, v2, score_mod=scoremod_2)
            return flex_attention(q3, k3, v3, score_mod=scoremod_1)

        eager_out = f(query, *keys, *values)

        block_mask = create_block_mask(noop_mask, 1, 1, 1024, 1024)
        (
            k_cache1,
            v_cache1,
            converted_block_mask1,
            converted_score_mod1,
        ) = self.preprocess_paged_attention(
            scoremod_1, query, keys[0], values[0], block_mask, torch.float32
        )
        (
            k_cache2,
            v_cache2,
            converted_block_mask2,
            converted_score_mod2,
        ) = self.preprocess_paged_attention(
            scoremod_2, query, keys[1], values[1], block_mask, torch.float32
        )
        (
            k_cache3,
            v_cache3,
            converted_block_mask3,
            converted_score_mod3,
        ) = self.preprocess_paged_attention(
            scoremod_1, query, keys[2], values[2], block_mask, torch.float32
        )

        paged_attention1 = functools.partial(
            flex_attention,
            score_mod=converted_score_mod1,
            block_mask=converted_block_mask1,
        )

        def paged_f(q, k1, k2, k3, v1, v2, v3):
            q2 = paged_attention1(q, k1, v1)
            q3 = flex_attention(
                q2,
                k2,
                v2,
                score_mod=converted_score_mod2,
                block_mask=converted_block_mask2,
            )
            return flex_attention(
                q3,
                k3,
                v3,
                score_mod=converted_score_mod3,
                block_mask=converted_block_mask3,
            )

        compiled_out = torch.compile(paged_f)(
            query, k_cache1, k_cache2, k_cache3, v_cache1, v_cache2, v_cache3
        )
        tolerance = Tolerances(atol=2e-1, rtol=2e-1)
        torch.testing.assert_close(
            eager_out, compiled_out, atol=tolerance.atol, rtol=tolerance.rtol
        )

    @supported_platform
    def test_inputs_are_realized(self):
        def f(q, k, v):
            x = torch.randn(1024, device="cuda")
            x = x * 2

            def func(qk, b, h, q, kv):
                return qk + x[q]

            return flex_attention(q.sin(), k, v, score_mod=func).cos()

        q, k, v = (
            torch.randn(1, 8, 1024, 64, device="cuda", requires_grad=True)
            for _ in range(3)
        )
        ref = f(q, k, v)
        out = torch.compile(f)(q, k, v)
        self.assertTrue((ref - out).abs().mean() < 1e-2)
        gradOut = torch.randn_like(q)

        ref_grads = torch.autograd.grad(ref, (q, k, v), gradOut)
        out_grads = torch.autograd.grad(out, (q, k, v), gradOut)
        for ref, out in zip(ref_grads, out_grads):
            self.assertTrue((ref - out).abs().mean() < 1e-2)

    @supported_platform
    def test_make_block_mask(self):
        def causal_mask(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx

        block_mask_a = torch.compile(create_block_mask)(causal_mask, 1, 1, 512, 512)
        block_mask_b = create_block_mask(causal_mask, 1, 1, 512, 512)
        self.assertEqual(block_mask_a.kv_num_blocks, block_mask_b.kv_num_blocks)
        self.assertEqual(block_mask_a.kv_indices, block_mask_b.kv_indices)
        self.assertEqual(block_mask_a.q_num_blocks, block_mask_b.q_num_blocks)

    @supported_platform
    def test_mask_mod_combiners(self):
        def causal_mask(b, h, q, kv):
            return q >= kv

        def neg_causal_mask(b, h, q, kv):
            return q < kv

        def sliding_window(b, h, q, kv):
            return (q - kv) <= 512

        block_mask = create_block_mask(
            and_masks(causal_mask, sliding_window), 1, 1, S, S
        )
        self.assertExpectedInline(block_mask.kv_num_blocks.sum().item(), """28""")
        attention = functools.partial(flex_attention, block_mask=block_mask)
        self.run_test_with_call(attention)

        block_mask = create_block_mask(
            and_masks(causal_mask, neg_causal_mask), 1, 1, S, S
        )
        self.assertEqual(block_mask.kv_num_blocks.sum(), 0)

        block_mask1 = create_block_mask(
            or_masks(causal_mask, neg_causal_mask), 1, 1, S, S
        )
        block_mask2 = create_block_mask(noop_mask, 1, 1, S, S)
        self.assertEqual(block_mask1.sparsity(), block_mask2.sparsity())

    @supported_platform
    def test_epilogue_fused(self):
        @torch.compile
        def f(q, k, v):
            out = flex_attention(q, k, v)
            return out.cos()

        q, k, v = (torch.randn(1, 8, 1024, 64, device="cuda") for _ in range(3))
        metrics.reset()
        _, code = run_and_get_code(f, q, k, v)
        fc = FileCheck()
        fc.check("triton_tem_fused")  # template call
        fc.check_not("poi_fused_cos")  # No cos pointwise operation
        fc.run(code[0])
        accessed_bytes = 1 * 8 * 1024 * 64 * torch.float32.itemsize
        num_accesses = 4  # q, k, v reads, one output.
        # TODO: Get rid of this fudge factor
        # We need this fudge factor for now as we write the extraneous logsumexp
        num_accesses += 1
        self.assertLess(metrics.num_bytes_accessed, accessed_bytes * num_accesses)

    @supported_platform
    @common_utils.parametrize("dtype", test_dtypes)
    def test_njt_causal(self, dtype):
        offsets = torch.tensor(
            [0, 1024, 1024 + 512, S], device="cuda", dtype=torch.int32
        )
        seq_idx = torch.zeros(S, device="cuda", dtype=torch.int32)
        for idx in range(len(offsets) - 1):
            seq_idx[offsets[idx] : offsets[idx + 1]] = idx

        def create_njt_wrapper(orig_score_mod, offsets, seq_idx):
            def njt_score_mod(qk, b, h, q, kv):
                q_nested = q - offsets[seq_idx[q]]
                kv_nested = kv - offsets[seq_idx[kv]]
                return orig_score_mod(qk, b, h, q_nested, kv_nested)

            return njt_score_mod

        causal_njt = create_njt_wrapper(_causal, offsets, seq_idx)

        self.run_test(causal_njt, dtype)
        self.run_test_with_paged_attention(causal_njt, dtype)

    @supported_platform
    def test_mixed_dtypes_fails(self):
        query = torch.randn((1, 1, 1024, 64), dtype=torch.float32, device="cuda")
        key = torch.randn((1, 1, 1024, 64), dtype=torch.float16, device="cuda")
        value = torch.randn((1, 1, 1024, 64), dtype=torch.float16, device="cuda")
        with self.assertRaisesRegex(
            ValueError, "Expected query, key, and value to have the same dtype"
        ):
            flex_attention(query, key, value, _identity)

    @supported_platform
    @patch.object(torch._inductor.config, "max_autotune", True)
    def test_max_autotune(self):
        def score_mod(score, b, h, m, n):
            return score * 2

        self.run_test(score_mod)
        self.run_test_with_paged_attention(score_mod)

    @supported_platform
    @skip("TODO: Figure out why this is erroring")
    @patch.object(torch._inductor.config, "max_autotune", True)
    def test_max_autotune_with_captured(self):
        head_scale = torch.randn(H, device="cuda")
        batch_scale = torch.randn(B, device="cuda")
        tok_scale = torch.randn(S, device="cuda")

        def bias_mod(score, batch, head, token_q, token_kv):
            score = score + tok_scale[token_q]
            score = score + batch_scale[batch]
            score = score + head_scale[head]
            return score

        self.run_test(bias_mod)

    @supported_platform
    @common_utils.parametrize("score_mod", test_score_mods)
    @common_utils.parametrize("dtype", test_dtypes)
    @common_utils.parametrize("head_dims", [(D, D // 2), (D // 2, D)])
    def test_non_equal_head_dims(self, dtype, score_mod, head_dims):
        qk_d, v_d = head_dims
        self.run_test(score_mod, dtype, B, H, S, qk_d, B, H, S, V_D=v_d)
        self.run_test_with_paged_attention(
            score_mod, dtype, B, H, S, qk_d, B, H, S, V_D=v_d
        )

    @supported_platform
    def test_autograd_function_in_score_mod(self):
        class ApplyMask(torch.autograd.Function):
            generate_vmap_rule = True

            @staticmethod
            def forward(a, mask):
                return torch.where(mask, a, -float("inf"))

            @staticmethod
            def setup_context(ctx, inputs, output):
                _, mask = inputs
                ctx.mark_non_differentiable(mask)

            @staticmethod
            def backward(ctx, i):
                return i, None

        def score_mod(score, b, h, q, kv):
            return ApplyMask.apply(score, q <= kv)

        func = torch.compile(flex_attention, fullgraph=True)

        q, k, v = (
            torch.randn(1, 8, 1024, 64, device="cuda", requires_grad=True)
            for _ in range(3)
        )

        # Just checking that it runs
        func(q, k, v)

        # expectedFailure
        # This doesn't work due to vmap + autograd.Function + torch.compile not composing
        # self.run_test(score_mod)

    @supported_platform
    def test_causal_block(self):
        def mask_mod(b, h, q, kv):
            return q >= kv

        block_mask = create_block_mask(mask_mod, 1, 1, S, S)
        attention = functools.partial(flex_attention, block_mask=block_mask)

        self.run_test_with_call(attention)

    @supported_platform
    def test_causal_block_paged_attention(self):
        def mask_mod(b, h, q, kv):
            return q >= kv

        block_mask = create_block_mask(mask_mod, B, 1, S, S)
        self.run_test_with_paged_attention(score_mod=_identity, block_mask=block_mask)

    @supported_platform
    def test_new_empty_mask_mod(self):
        S = 128
        q, k, v = (torch.randn(4, 1, S, 64, device="cuda") for _ in range(3))

        attn_mask = torch.ones(4, 1, S, S, dtype=torch.bool, device="cuda").tril()

        def score_mod(score, b, h, q_idx, kv_idx):
            h_ = h.new_zeros(h.shape)
            return score + attn_mask[b, h_, q_idx, kv_idx]

        def causal(b, h, q_idx, kv_idx):
            h_ = h.new_zeros(h.shape)
            return attn_mask[b, h_, q_idx, kv_idx]

        block_mask = create_block_mask(causal, B=4, H=None, Q_LEN=S, KV_LEN=S)
        torch.compile(flex_attention)(q, k, v, score_mod, block_mask=block_mask)

    @supported_platform
    @common_utils.parametrize("head_dim", [13, 24, 94, 121])
    @common_utils.parametrize("dtype", test_dtypes_fast)
    def test_non_pow_2_headdim(self, dtype, head_dim):
        self.run_test(_rel_bias, torch.float16, B, H, S, head_dim, B, H, S, head_dim)

    @supported_platform
    def test_GQA_causal_mask(self):
        def mask_mod(b, h, q, kv):
            return q >= kv

        block_mask = create_block_mask(mask_mod, B, 1, S // 8, S // 8)
        attention = functools.partial(
            flex_attention, block_mask=block_mask, enable_gqa=True
        )

        self.run_test_with_call(
            attention,
            torch.float16,
            B,
            H * 4,  # Hq = 4*Hkv.
            S // 8,
            D,
            B,
            H,
            S // 8,
            D,
        )

        self.run_test_with_paged_attention(
            Q_H=H * 4,
            Q_S=S // 8,
            KV_H=H,
            KV_S=S // 8,
            block_mask=block_mask,
        )

    @supported_platform
    def test_custom_block_mask_generator(self):
        def mask_mod(b, h, q, kv):
            return q >= kv

        auto_mask = create_block_mask(mask_mod, 1, 1, S, S)
        BLOCK_SIZE = 128

        def causal_constructor(S):
            num_blocks = torch.arange(S // BLOCK_SIZE, device="cuda") + 1
            indices = torch.arange(S // BLOCK_SIZE, device="cuda").expand(
                S // BLOCK_SIZE, S // BLOCK_SIZE
            )
            num_blocks = num_blocks[None, None, :]
            indices = indices[None, None, :]
            return BlockMask.from_kv_blocks(
                num_blocks, indices, BLOCK_SIZE=BLOCK_SIZE, mask_mod=mask_mod
            )

        manual_mask = causal_constructor(S)
        self.assertEqual(auto_mask.to_dense(), manual_mask.to_dense())

    @supported_platform
    @common_utils.parametrize("dtype", test_dtypes)
    @common_utils.parametrize("score_mod", [_identity, _causal])
    def test_logsumexp_correctness(self, dtype, score_mod):
        make_tensor = functools.partial(
            torch.randn,
            (B, H, S, D),
            dtype=dtype,
            device="cuda",
            requires_grad=True,
        )
        q, k, v = make_tensor(), make_tensor(), make_tensor()

        @torch.compile
        def sdpa_hop(q, k, v, score_mod):
            return flex_attention(q, k, v, score_mod, return_lse=True)

        @torch.compile(backend="aot_eager")
        def eager_sdpa_hop(q, k, v, score_mod):
            return flex_attention(q, k, v, score_mod, return_lse=True)

        ref_out, ref_lse = eager_sdpa_hop(
            q.to(torch.float64),
            k.to(torch.float64),
            v.to(torch.float64),
            score_mod,
        )
        compiled_out, compiled_lse = sdpa_hop(q, k, v, score_mod)

        self.assertTrue(ref_lse.dtype == torch.float64)
        self.assertTrue(compiled_lse.dtype == torch.float32)

        tolerance = Tolerances(atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(
            ref_out.to(dtype=torch.float32),
            compiled_out.to(dtype=torch.float32),
            atol=tolerance.atol,
            rtol=tolerance.rtol,
        )
        torch.testing.assert_close(
            ref_lse.to(dtype=torch.float32),
            compiled_lse.to(dtype=torch.float32),
            atol=tolerance.atol,
            rtol=tolerance.rtol,
        )

    @supported_platform
    def test_logsumexp_only_return(self):
        make_tensor = functools.partial(
            torch.randn,
            (B, H, S, D),
            dtype=torch.float32,
            device="cuda",
            requires_grad=True,
        )
        q, k, v = make_tensor(), make_tensor(), make_tensor()

        @torch.compile
        def func(q, k, v, score_mod):
            _, lse = flex_attention(q, k, v, score_mod, return_lse=True)
            lse_2 = lse * 2
            return lse_2

        _, code = run_and_get_code(func, q, k, v, _identity)
        # Ensure that we're still generating the flexattention kernel
        FileCheck().check_count(".run(primals_1, primals_2, primals_3", 1, True).run(
            code[0]
        )

    @supported_platform
    @common_utils.parametrize(
        "score_mod", [_identity, _causal, _times_two, _squared, _trig, _trig2]
    )
    def test_aot_eager_gradcheck(self, score_mod):
        make_tensor = functools.partial(
            torch.randn,
            (2, 2, 11, 4),
            device="cuda",
            dtype=torch.float64,
            requires_grad=True,
        )
        query, key, value = make_tensor(), make_tensor(), make_tensor()

        func = torch.compile(flex_attention, backend="aot_eager", fullgraph=True)

        self.assertTrue(
            torch.autograd.gradcheck(
                func, (query, key, value, score_mod), raise_exception=True
            )
        )

    @supported_platform
    def test_eager_backward_strides(self):
        class Repro(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.qkv_proj = torch.nn.Linear(256, 256 * 3)
                self.n_head = 256 // 64
                self.d_attn = 256

            def forward(self, x):
                n_batch, n_ctx, _ = x.shape
                q, k, v = self.qkv_proj(x).split(
                    [self.d_attn, self.d_attn, self.d_attn], dim=2
                )
                q = q.reshape(n_batch, n_ctx, self.n_head, -1)
                k = k.reshape(n_batch, n_ctx, self.n_head, -1)
                v = v.reshape(n_batch, n_ctx, self.n_head, -1)
                q = q.transpose(1, 2)
                k = k.transpose(1, 2)
                v = v.transpose(1, 2)
                x = torch.nn.attention.flex_attention.flex_attention(q, k, v)
                return x

        model = Repro().cuda()
        x = torch.randn((1, 512, 256), device="cuda", requires_grad=True)
        out = torch.compile(model, backend="aot_eager")(x)
        out.backward(torch.ones_like(out))

    @supported_platform
    def test_differentiable_logsumexp_gradcheck(self):
        make_tensor = functools.partial(
            torch.randn,
            (2, 2, 11, 4),
            device="cuda",
            dtype=torch.float64,
            requires_grad=True,
        )
        query, key, value = make_tensor(), make_tensor(), make_tensor()

        def flex_attention_lse_only(q, k, v):
            return flex_attention(q, k, v, return_lse=True)[1]

        func = torch.compile(
            flex_attention_lse_only, backend="aot_eager", fullgraph=True
        )

        self.assertTrue(
            torch.autograd.gradcheck(func, (query, key, value), raise_exception=True)
        )

    @supported_platform
    def test_differentiable_logsumexp_compiled(self):
        make_tensor = functools.partial(
            torch.randn,
            (2, 2, 128, 64),
            device="cuda",
            dtype=torch.float32,
            requires_grad=True,
        )
        q, k, v = make_tensor(), make_tensor(), make_tensor()
        lse_mask = torch.randn(2, 2, 128, device="cuda")

        out, lse = flex_attention(q, k, v, return_lse=True)
        (out.mean() + (lse * lse_mask).sum()).backward()
        q_grad, k_grad, v_grad = q.grad, k.grad, v.grad
        q.grad = None
        k.grad = None
        v.grad = None

        out2, lse2 = torch.compile(flex_attention)(q, k, v, return_lse=True)
        (out2.mean() + (lse2 * lse_mask).sum()).backward()
        q_grad2, k_grad2, v_grad2 = q.grad, k.grad, v.grad
        tolerance = Tolerances(atol=1e-1, rtol=1e-1)

        torch.testing.assert_close(out, out2, atol=tolerance.atol, rtol=tolerance.rtol)
        torch.testing.assert_close(lse, lse2, atol=tolerance.atol, rtol=tolerance.rtol)
        torch.testing.assert_close(
            q_grad, q_grad2, atol=tolerance.atol, rtol=tolerance.rtol
        )
        torch.testing.assert_close(
            k_grad, k_grad2, atol=tolerance.atol, rtol=tolerance.rtol
        )
        torch.testing.assert_close(
            v_grad, v_grad2, atol=tolerance.atol, rtol=tolerance.rtol
        )

    # Use weird mask to test reusing block_mask does work well.
    @supported_platform
    def _test_block_mask_reuse_with_weird_mask(self):
        def mask(b, h, q, kv):
            return (kv < 256) | (kv >= 2048)

        make_tensor = functools.partial(
            torch.randn,
            (4, 4, 4096, 64),
            device="cuda",
            dtype=torch.float32,
            requires_grad=True,
        )

        block_mask = create_block_mask(mask, None, None, 4096, 4096)
        # Compile 1st version with q/k/v(seqlen=4096) and block_mask(seqlen=4096)
        torch.compile(flex_attention, dynamic=True)(
            make_tensor(), make_tensor(), make_tensor(), block_mask=block_mask
        )

        make_tensor2 = functools.partial(
            torch.randn,
            (4, 4, 2048, 64),
            device="cuda",
            dtype=torch.float32,
            requires_grad=True,
        )
        q, k, v = make_tensor2(), make_tensor2(), make_tensor2()

        # Compile 2st version with q/k/v(seqlen=2048) and block_mask(seqlen=4096),
        # The graph includes the BlockMask._adjust part.
        out = torch.compile(flex_attention, dynamic=True)(
            q, k, v, block_mask=block_mask
        )
        out.sum().backward()
        q_grad, k_grad, v_grad = q.grad, k.grad, v.grad
        q.grad = None
        k.grad = None
        v.grad = None

        block_mask2 = create_block_mask(mask, None, None, 2048, 2048)
        # Reuse the 1st version with q/k/v(seqlen=2048) and block_mask(seqlen=2048)
        out2 = torch.compile(flex_attention, dynamic=True)(
            q, k, v, block_mask=block_mask2
        )
        out2.sum().backward()
        q_grad2, k_grad2, v_grad2 = q.grad, k.grad, v.grad
        tolerance = Tolerances(atol=1e-3, rtol=1e-3)

        torch.testing.assert_close(out, out2, atol=tolerance.atol, rtol=tolerance.rtol)
        torch.testing.assert_close(
            q_grad, q_grad2, atol=tolerance.atol, rtol=tolerance.rtol
        )
        torch.testing.assert_close(
            k_grad, k_grad2, atol=tolerance.atol, rtol=tolerance.rtol
        )
        torch.testing.assert_close(
            v_grad, v_grad2, atol=tolerance.atol, rtol=tolerance.rtol
        )

    @supported_platform
    def test_float32_matmul_precision(self):
        make_tensor = functools.partial(
            torch.zeros,
            (2, 2, 128, 32),
            device="cuda",
            dtype=torch.float32,
            requires_grad=False,
        )
        query, key, value = make_tensor(), make_tensor(), make_tensor()
        query.fill_(0.2)
        key.fill_(0.3)
        value.fill_(0.4)

        query.requires_grad = True
        key.requires_grad = True
        value.requires_grad = True

        def score_mod(score, b, h, q, kv):
            return score * 2

        with temp_float32_matmul_precision("highest"):
            out_eager = flex_attention(query, key, value, score_mod)
            flex_compiled = torch.compile(flex_attention, fullgraph=True)
            out_compiled = flex_compiled(query, key, value, score_mod)

            grads_eager = torch.autograd.grad(out_eager.sum(), (query, key, value))
            grads_compile = torch.autograd.grad(out_compiled.sum(), (query, key, value))

        torch.testing.assert_close(grads_eager, grads_compile)

    @supported_platform
    @common_utils.parametrize("score_mod_name", ["_head_offset"])
    @common_utils.parametrize("mode", ["eager", "aot_eager"])
    def test_captured_score_mod_aot_eager_gradcheck(
        self, score_mod_name: str, mode: str
    ):
        make_tensor = functools.partial(
            torch.randn,
            (2, 2, 11, 4),
            device="cuda",
            dtype=torch.float64,
            requires_grad=True,
        )
        query, key, value = make_tensor(), make_tensor(), make_tensor()

        func = torch.compile(flex_attention, backend=mode, fullgraph=True)
        score_mod = captured_buffers_map[score_mod_name](torch.float64)

        self.assertTrue(
            torch.autograd.gradcheck(
                func, (query, key, value, score_mod), raise_exception=True
            )
        )

    @supported_platform
    @common_utils.parametrize("mode", ["eager", "aot_eager"])
    def test_document_masking_edge_case(self, mode):
        document_masks = torch.full((2, 128), 0, dtype=torch.int32, device="cuda")
        document_masks[:, 64:] = 1

        def mask_mod(b, h, q, kv):
            same_doc = document_masks[b, q] == document_masks[b, kv]
            return same_doc

        make_tensor = functools.partial(
            torch.randn,
            (2, 1, 128, 4),
            device="cuda",
            dtype=torch.float64,
            requires_grad=True,
        )
        query, key, value = make_tensor(), make_tensor(), make_tensor()
        func = torch.compile(flex_attention, backend=mode, fullgraph=True)

        block_mask = create_block_mask(mask_mod, 2, 1, 128, 128)
        out = func(query, key, value, block_mask=block_mask)
        out.sum().backward()

    @supported_platform
    def test_strided_backwards(self):
        shape = (1, 2, 4096, 64)
        Q = torch.randn(shape, requires_grad=True, device="cuda")
        K = torch.randn(shape, requires_grad=True, device="cuda")
        V = torch.randn(shape, requires_grad=True, device="cuda")
        func = torch.compile(flex_attention, dynamic=True, fullgraph=True)

        K_sliced = K[:, :, :-128]
        V_sliced = V[:, :, :-128]

        out_eager = flex_attention(Q, K_sliced, V_sliced)
        out_compiled = func(Q, K_sliced, V_sliced)

        grad = torch.rand_like(out_eager)

        eager_grads = torch.autograd.grad(out_eager, (Q, K, V), grad)
        compiled_grads = torch.autograd.grad(out_compiled, (Q, K, V), grad)

        for eager, compiled in zip(eager_grads, compiled_grads):
            torch.testing.assert_close(eager, compiled, atol=9e-3, rtol=0)

    @supported_platform
    @common_utils.parametrize("mode", ["eager", "inductor", "paged_attention"])
    @common_utils.parametrize(
        "permute_order",
        [
            (0, 1, 2, 3),  # Default order
            (1, 0, 2, 3),  # Reverse order
            (0, 2, 1, 3),  # Mixed order
            (2, 0, 1, 3),  # Another mixed order
        ],
    )
    @common_utils.parametrize("shape", [(2, 1, 128, 16), (4, 2, 64, 16)])
    def test_flex_attention_stride_ordering(self, mode, permute_order, shape):
        if TEST_WITH_ROCM:
            self.skipTest(
                "ROCM BUG SEE: https://github.com/pytorch/pytorch/issues/140855"
            )
        from torch._inductor.ir import get_stride_order

        dtype = torch.float32
        # Setup
        make_tensor = functools.partial(
            torch.randn,
            shape,
            device="cuda",
            dtype=dtype,
            requires_grad=False if mode == "paged_attention" else True,
        )

        # Create and permute tensors
        query, key, value = make_tensor(), make_tensor(), make_tensor()
        query = query.permute(permute_order)
        key = key.permute(permute_order)
        value = value.permute(permute_order)

        if mode == "inductor":
            func = torch.compile(flex_attention, backend=mode, fullgraph=True)
            out = func(query, key, value)
        elif mode == "paged_attention":
            out, _ = self.run_paged_attention(_identity, query, key, value, dtype)
        else:
            func = flex_attention
            out = func(query, key, value)

        out_stride_order = get_stride_order(out.stride())
        query_stride_order = get_stride_order(query.stride())

        self.assertEqual(
            out_stride_order,
            query_stride_order,
            f"Stride order mismatch: out {out_stride_order}, query {query_stride_order}",
        )

    @supported_platform
    @common_utils.parametrize("mode", ["eager", "inductor"])
    @common_utils.parametrize(
        "permute_order",
        [
            (0, 1, 2, 3),
            (1, 0, 2, 3),
            (0, 2, 1, 3),
            (2, 0, 1, 3),
        ],
    )
    @common_utils.parametrize("shape", [(2, 5, 128, 16), (4, 2, 64, 16)])
    def test_flex_attention_backward_stride_ordering(self, mode, permute_order, shape):
        if TEST_WITH_ROCM:
            self.skipTest(
                "ROCM BUG SEE: https://github.com/pytorch/pytorch/issues/140855"
            )
        from torch._inductor.ir import get_stride_order

        dtype = torch.float32
        make_tensor = functools.partial(
            torch.randn, shape, device="cuda", dtype=dtype, requires_grad=False
        )

        query, key, value = make_tensor(), make_tensor(), make_tensor()
        query = query.permute(permute_order)
        key = key.permute(permute_order)
        value = value.permute(permute_order)

        query.requires_grad_()
        key.requires_grad_()
        value.requires_grad_()

        func = (
            torch.compile(flex_attention, backend=mode, fullgraph=True)
            if mode == "inductor"
            else flex_attention
        )
        out = func(query, key, value)
        grad_output = torch.randn_like(out)
        out.backward(grad_output)

        for leaf, grad, name in [
            (query, query.grad, "query"),
            (key, key.grad, "key"),
            (value, value.grad, "value"),
        ]:
            input_stride_order = get_stride_order(grad.stride())
            orig_stride_order = get_stride_order(leaf.stride())
            self.assertEqual(
                input_stride_order,
                orig_stride_order,
                f"Mode: {mode}, Stride order mismatch for {name}: grad {input_stride_order}, input {orig_stride_order}.",
            )

    @supported_platform
    @common_utils.parametrize("compile", [True, False])
    def test_fully_masked_out_rows_0_check(self, compile: bool):
        # Ensure fully masked out rows won't cause NaNs.
        query = torch.randn(
            (B, H, S, D), dtype=torch.float32, device="cuda", requires_grad=True
        )
        key = torch.randn(
            (B, H, S, D), dtype=torch.float32, device="cuda", requires_grad=True
        )
        value = torch.randn(
            (B, H, S, D), dtype=torch.float32, device="cuda", requires_grad=True
        )

        M = S // 2

        def mask_mod(b, h, q, kv):
            return q < M

        block_mask = create_block_mask(mask_mod, B, 1, S, S)

        flex = (
            torch.compile(flex_attention, dynamic=False) if compile else flex_attention
        )
        out, lse = flex(query, key, value, block_mask=block_mask, return_lse=True)
        self.assertEqual(out[:, :, M:, :].sum(), 0)
        self.assertTrue((lse[:, :, M:] == -float("inf")).all())

        loss = out.sum() + lse.sum()
        loss.backward()
        self.assertEqual(query.grad[:, :, M:, :].sum(), 0)

    @supported_platform
    @common_utils.parametrize("compile", [True, False])
    def test_fully_masked_out_rows(self, compile: bool):
        M = S // 2

        def mask_mod(b, h, q, kv):
            return q < M

        block_mask = create_block_mask(mask_mod, B, 1, S, S)

        def noop_mod(score, b, h, q_idx, kv_idx):
            return score

        self.run_test(noop_mod, torch.float32, B, H, S, D, B, H, S, D, block_mask)

    @supported_platform
    def test_kernel_options_argument_is_respected(self):
        make_tensor = functools.partial(
            torch.randn,
            (2, 2, 128, 64),
            device="cuda",
            dtype=torch.float32,
            requires_grad=True,
        )
        q, k, v = make_tensor(), make_tensor(), make_tensor()

        # Ensure we respect user's input kernel options.
        _, code = run_and_get_code(
            torch.compile(flex_attention), q, k, v, kernel_options={"BLOCK_M": 16}
        )
        FileCheck().check("BLOCK_M : tl.constexpr = 16").run(code[0])

    @supported_platform
    def test_comparison_vs_sdpa(self):
        def causal(score, b, h, q_idx, kv_idx):
            return torch.where(q_idx >= kv_idx, score, -float("inf"))

        def causal_mask(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx

        no_sparse_flex = functools.partial(flex_attention, score_mod=causal)
        score_mod_sparse_flex = functools.partial(
            flex_attention,
            score_mod=causal,
            block_mask=create_block_mask(causal_mask, 1, 1, 2048, 2048),
        )
        mask_mod_sparse_flex = functools.partial(
            flex_attention, block_mask=create_block_mask(causal_mask, 1, 1, 2048, 2048)
        )
        for attention_call in [
            no_sparse_flex,
            score_mod_sparse_flex,
            mask_mod_sparse_flex,
        ]:
            inputs = [
                torch.randn(
                    2,
                    2,
                    2048,
                    64,
                    device="cuda",
                    dtype=torch.float16,
                    requires_grad=True,
                )
                for _ in range(3)
            ]
            gradOut = torch.randn(2, 2, 2048, 64, device="cuda", dtype=torch.float16)
            out_ref = torch.nn.functional.scaled_dot_product_attention(
                *inputs, is_causal=True
            )
            out_ref.backward(gradOut)

            inputs_flex = [i.detach().clone().requires_grad_(True) for i in inputs]
            out_flex = torch.compile(attention_call)(*inputs_flex)
            out_flex.backward(gradOut)
            inputs_golden = [
                i.detach().clone().to(dtype=torch.float64).requires_grad_(True)
                for i in inputs
            ]
            out_golden = torch.nn.functional.scaled_dot_product_attention(
                *inputs_golden, is_causal=True
            )
            out_golden.backward(gradOut.to(dtype=torch.float64))

            for ref, flex, golden in [
                (out_ref, out_flex, out_golden),
                (inputs[0].grad, inputs_flex[0].grad, inputs_golden[0].grad),
                (inputs[1].grad, inputs_flex[1].grad, inputs_golden[1].grad),
                (inputs[2].grad, inputs_flex[2].grad, inputs_golden[2].grad),
            ]:
                ref_error = rmse(ref, golden)
                flex_error = rmse(flex, golden)
                # Note: This has been carefully tested that FlexAttention is within
                # 20% of the average error of SDPA! Do not bump this tolerance
                # unless you are absolutely sure you are not worsening the accuracy
                # of FlexAttention!
                self.assertTrue(
                    ref_error * 1.2 > flex_error,
                    f"Ref error: {ref_error}, Flex Error: {flex_error}",
                )

    @supported_platform
    def test_block_mask_non_divisible(self):
        seq = torch.arange(1023, device="cuda") // 128

        def mod(b, h, q, kv):
            return seq[q] == seq[kv]

        block_mask = create_block_mask(mod, None, None, 1023, 1023, device="cuda")
        torch.compile(create_block_mask)(mod, None, None, 1023, 1023, device="cuda")
        self.run_test_with_call(
            lambda q, k, v: flex_attention(q, k, v, block_mask=block_mask),
            Q_S=1023,
            KV_S=1023,
        )

    @supported_platform
    def test_head_bias_req_grad(self):
        B, H, S, D = 1, 4, 256, 64
        bias = torch.randn(H, device="cuda", dtype=torch.float16, requires_grad=True)

        bias_flex = bias.detach().clone().requires_grad_(True)

        def head_bias(score, b, h, q_idx, kv_idx):
            return score + bias_flex[h]

        bias_sdpa_ref = bias.detach().clone().requires_grad_(True)
        implicit_bias_sdpa_ref = bias_sdpa_ref
        implicit_bias_sdpa_ref = implicit_bias_sdpa_ref.view(H, 1, 1).expand(H, S, S)
        bias_sdpa_gold = (
            bias.detach().clone().to(dtype=torch.float64).requires_grad_(True)
        )
        implicit_bias_sdpa_gold = bias_sdpa_gold
        implicit_bias_sdpa_gold = implicit_bias_sdpa_gold.view(H, 1, 1).expand(H, S, S)

        self._test_learnable_bias_inner(
            B,
            H,
            S,
            D,
            head_bias,
            bias_flex,
            implicit_bias_sdpa_ref,
            bias_sdpa_ref,
            implicit_bias_sdpa_gold,
            bias_sdpa_gold,
        )

    @supported_platform
    def test_comparison_vs_sdpa_with_learnable_bias(self):
        # 1-dimensional bias:
        B, H, S, D = 1, 1, 256, 64
        bias = torch.randn(
            2 * S, device="cuda", dtype=torch.float16, requires_grad=True
        )

        bias_flex = bias.detach().clone().requires_grad_(True)

        def rel_pos_1d(score, b, h, q_idx, kv_idx):
            return score + bias_flex[q_idx + kv_idx]

        bias_indices = torch.arange(S)[:, None] + torch.arange(S)
        bias_sdpa_ref = bias.detach().clone().requires_grad_(True)
        implicit_bias_sdpa_ref = bias_sdpa_ref[bias_indices]
        bias_sdpa_gold = (
            bias.detach().clone().to(dtype=torch.float64).requires_grad_(True)
        )
        implicit_bias_sdpa_gold = bias_sdpa_gold[bias_indices]

        self._test_learnable_bias_inner(
            B,
            H,
            S,
            D,
            rel_pos_1d,
            bias_flex,
            implicit_bias_sdpa_ref,
            bias_sdpa_ref,
            implicit_bias_sdpa_gold,
            bias_sdpa_gold,
        )

        # 2-dimensional bias:
        B, H, S, D = 1, 1, 256, 64
        bias = torch.randn(S, S, device="cuda", dtype=torch.float16, requires_grad=True)

        bias_flex = bias.detach().clone().requires_grad_(True)

        def rel_pos_2d(score, b, h, q_idx, kv_idx):
            return score + bias_flex[q_idx, kv_idx]

        bias_sdpa_ref = bias.detach().clone().requires_grad_(True)
        implicit_bias_sdpa_ref = bias_sdpa_ref
        bias_sdpa_gold = (
            bias.detach().clone().to(dtype=torch.float64).requires_grad_(True)
        )
        implicit_bias_sdpa_gold = bias_sdpa_gold

        self._test_learnable_bias_inner(
            B,
            H,
            S,
            D,
            rel_pos_2d,
            bias_flex,
            implicit_bias_sdpa_ref,
            bias_sdpa_ref,
            implicit_bias_sdpa_gold,
            bias_sdpa_gold,
        )

        # 2-dimensional bias + index multiple
        B, H, S, D = 1, 1, 256, 64
        bias = torch.randn(S, S, device="cuda", dtype=torch.float16, requires_grad=True)

        bias_flex = bias.detach().clone().requires_grad_(True)

        def rel_pos_2d(score, b, h, q_idx, kv_idx):
            return score + bias_flex[q_idx][kv_idx]

        bias_sdpa_ref = bias.detach().clone().requires_grad_(True)
        implicit_bias_sdpa_ref = bias_sdpa_ref
        bias_sdpa_gold = (
            bias.detach().clone().to(dtype=torch.float64).requires_grad_(True)
        )
        implicit_bias_sdpa_gold = bias_sdpa_gold

        self._test_learnable_bias_inner(
            B,
            H,
            S,
            D,
            rel_pos_2d,
            bias_flex,
            implicit_bias_sdpa_ref,
            bias_sdpa_ref,
            implicit_bias_sdpa_gold,
            bias_sdpa_gold,
        )

        # 2-dimensional bias + transposed:
        B, H, S, D = 1, 1, 256, 64
        bias = torch.randn(S, S, device="cuda", dtype=torch.float16, requires_grad=True)

        bias_flex = bias.detach().clone().requires_grad_(True)

        def rel_pos_2d_transposed(score, b, h, q_idx, kv_idx):
            return score + bias_flex[kv_idx, q_idx]

        bias_sdpa_ref = bias.detach().clone().requires_grad_(True)
        implicit_bias_sdpa_ref = bias_sdpa_ref.transpose(-1, -2)
        bias_sdpa_gold = (
            bias.detach().clone().to(dtype=torch.float64).requires_grad_(True)
        )
        implicit_bias_sdpa_gold = bias_sdpa_gold.transpose(-1, -2)

        self._test_learnable_bias_inner(
            B,
            H,
            S,
            D,
            rel_pos_2d_transposed,
            bias_flex,
            implicit_bias_sdpa_ref,
            bias_sdpa_ref,
            implicit_bias_sdpa_gold,
            bias_sdpa_gold,
        )

        # 3-dimensional bias + transposed
        B, H, S, D = 4, 8, 256, 64
        bias = torch.randn(
            H, S, S, device="cuda", dtype=torch.float16, requires_grad=True
        )

        bias_flex = bias.detach().clone().requires_grad_(True)

        def rel_pos_3d_transposed(score, b, h, q_idx, kv_idx):
            return score + bias_flex[h, kv_idx, q_idx]

        bias_sdpa_ref = bias.detach().clone().requires_grad_(True)
        implicit_bias_sdpa_ref = bias_sdpa_ref.transpose(-1, -2)
        bias_sdpa_gold = (
            bias.detach().clone().to(dtype=torch.float64).requires_grad_(True)
        )
        implicit_bias_sdpa_gold = bias_sdpa_gold.transpose(-1, -2)

        self._test_learnable_bias_inner(
            B,
            H,
            S,
            D,
            rel_pos_3d_transposed,
            bias_flex,
            implicit_bias_sdpa_ref,
            bias_sdpa_ref,
            implicit_bias_sdpa_gold,
            bias_sdpa_gold,
        )

    def _test_learnable_bias_inner(
        self,
        B,
        H,
        S,
        D,
        score_mod,
        bias_flex,
        implicit_bias_sdpa_ref,
        bias_sdpa_ref,
        implicit_bias_sdpa_gold,
        bias_sdpa_gold,
    ):
        make_tensor = functools.partial(
            torch.ones,
            (B, H, S, D),
            device="cuda",
            dtype=torch.float16,
            requires_grad=True,
        )
        q_ref, k_ref, v_ref = make_tensor(), make_tensor(), make_tensor()
        q_gold, k_gold, v_gold = query_key_value_clones(
            q_ref, k_ref, v_ref, torch.float64
        )
        q_flex, k_flex, v_flex = query_key_value_clones(q_ref, k_ref, v_ref)

        out_ref = torch.nn.functional.scaled_dot_product_attention(
            q_ref, k_ref, v_ref, attn_mask=implicit_bias_sdpa_ref
        )
        out_ref.sum().backward()
        out_gold = torch.nn.functional.scaled_dot_product_attention(
            q_gold, k_gold, v_gold, attn_mask=implicit_bias_sdpa_gold
        )
        out_gold.sum().backward()
        out_flex = flex_attention(q_flex, k_flex, v_flex, score_mod=score_mod)
        out_flex.sum().backward()

        name = score_mod.__name__
        for ref, flex, gold in [
            (out_ref, out_flex, out_gold),
            (q_ref.grad, q_flex.grad, q_gold.grad),
            (k_ref.grad, k_flex.grad, k_gold.grad),
            (v_ref.grad, v_flex.grad, v_gold.grad),
            (bias_sdpa_ref.grad, bias_flex.grad, bias_sdpa_gold.grad),
        ]:
            ref_error = rmse(ref, gold)
            flex_error = rmse(flex, gold)
            self.assertTrue(
                ref_error * 1.2 >= flex_error,
                f"{name} -> Ref error: {ref_error}, Flex eager Error: {flex_error}",
            )

    @supported_platform
    def test_causal_block_non_divisible(self):
        def mask_mod(b, h, q, kv):
            return q >= kv

        block_mask = create_block_mask(mask_mod, B, 1, S - 1, S - 1)
        attention = functools.partial(flex_attention, block_mask=block_mask)

        self.run_test_with_call(attention, Q_S=S - 1, KV_S=S - 1)

    @supported_platform
    def test_modular_indexing(self):
        B, H, N, D = 100, 12, 128, 64
        dtype = torch.bfloat16
        device = torch.device("cuda")

        class Attention(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bias = torch.randn(B, N, N, H, device=device, dtype=dtype)

            def forward(
                self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
            ) -> torch.Tensor:
                score_mod = generate_score_mod(self.bias)
                o = flex_attention(q, k, v, score_mod=score_mod)
                return o

        def generate_score_mod(bias):
            bias = (2 * bias).view(B, H, N, N).contiguous()

            def score_mod(score, batch, head, q_idx, k_idx):
                attn_bias = bias[batch, head, q_idx, k_idx]
                return score + attn_bias

            return score_mod

        m = Attention().cuda().eval().to(dtype)
        m = torch.compile(m, mode="default", fullgraph=False)

        q = torch.randn(B, H, N, D, device=device, dtype=dtype)
        k = torch.randn(B, H, N, D, device=device, dtype=dtype)
        v = torch.randn(B, H, N, D, device=device, dtype=dtype)

        m(q, k, v)

    @supported_platform
    def test_force_write_lse(self):
        dtype = torch.float32
        make_tensor = functools.partial(
            torch.randn,
            (2, 2, 128, 16),
            device="cuda",
            dtype=dtype,
            requires_grad=False,
        )
        query, key, value = make_tensor(), make_tensor(), make_tensor()
        out_eager, lse_eager = flex_attention(query, key, value, return_lse=True)

        flex_compile = torch.compile(flex_attention, fullgraph=True)
        out_compiled, lse_compiled = flex_compile(query, key, value, return_lse=True)

        out_paged, lse_paged = self.run_paged_attention(
            score_mod=_identity, q=query, k=key, v=value, dtype=dtype
        )

        torch.testing.assert_close(lse_eager, lse_compiled, atol=3e-3, rtol=0)
        torch.testing.assert_close(lse_eager, lse_paged, atol=3e-3, rtol=0)

    @supported_platform
    @common_utils.parametrize("backend", ["flex_attention", "flex_decode", "eager"])
    def test_lse_masked_output(self, backend):
        if backend == "flex_decode":
            if TEST_WITH_ROCM:
                self.skipTest("backend=flex_decode is unsupported on ROCM, for now")
            kernel_options = {"FORCE_USE_FLEX_ATTENTION": False}
            flex_call = torch.compile(flex_attention, fullgraph=True)
            N_CTX = 96
        elif backend == "flex_attention":
            kernel_options = {"FORCE_USE_FLEX_ATTENTION": True}
            flex_call = torch.compile(flex_attention, fullgraph=True)
            N_CTX = 196
        else:
            kernel_options = {}
            flex_call = flex_attention
            N_CTX = 196

        SLIDING_WINDOW = 64
        make_tensor = functools.partial(
            torch.randn,
            (2, 2, N_CTX, 64),
            device="cuda",
            dtype=torch.float32,
            requires_grad=True,
        )

        def sliding_window_causal(b, h, q_idx, kv_idx):
            causal_mask = q_idx >= kv_idx
            window_mask = q_idx - kv_idx <= SLIDING_WINDOW
            return causal_mask & window_mask

        def global_causal(b, h, q_idx, kv_idx):
            causal_mask = q_idx >= kv_idx
            window_mask = q_idx - kv_idx > SLIDING_WINDOW
            return causal_mask & window_mask

        sliding_window_causal = torch.nn.attention.flex_attention.create_block_mask(
            sliding_window_causal, B=None, H=None, Q_LEN=N_CTX, KV_LEN=N_CTX
        )
        global_causal = torch.nn.attention.flex_attention.create_block_mask(
            global_causal, B=None, H=None, Q_LEN=N_CTX, KV_LEN=N_CTX
        )

        local_attn = functools.partial(
            flex_call,
            block_mask=sliding_window_causal,
            return_lse=True,
            kernel_options=kernel_options,
        )
        global_attn = functools.partial(
            flex_call,
            block_mask=global_causal,
            return_lse=True,
            kernel_options=kernel_options,
        )
        q, k, v = make_tensor(), make_tensor(), make_tensor()
        gradOut = make_tensor(requires_grad=False)

        x_local, lse_local = local_attn(q, k, v)
        x_global, lse_global = global_attn(q, k, v)

        max_lse = torch.maximum(lse_local, lse_global)
        lse_global = lse_global - max_lse
        lse_local = lse_local - max_lse
        lse_global = torch.exp(lse_global)
        lse_local = torch.exp(lse_local)
        x = ((x_local * lse_local[..., None]) + (x_global * lse_global[..., None])) / (
            lse_global[..., None] + lse_local[..., None]
        )
        x.backward(gradOut)
        flex_q_grad, flex_k_grad, flex_v_grad = q.grad, k.grad, v.grad
        q.grad = None
        k.grad = None
        v.grad = None

        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        out.backward(gradOut)

        torch.testing.assert_close(x, out, atol=3e-3, rtol=2e-3)
        torch.testing.assert_close(flex_q_grad, q.grad, atol=3e-3, rtol=2e-3)
        torch.testing.assert_close(flex_k_grad, k.grad, atol=3e-3, rtol=2e-3)
        torch.testing.assert_close(flex_v_grad, v.grad, atol=3e-3, rtol=2e-3)

    @supported_platform
    def test_mixed_device_error_message(self):
        # Create tensors on different devices
        cpu_tensor = torch.randn(2, 2, 128, 16, device="cpu")
        cuda_tensor = torch.randn(2, 2, 128, 16, device="cuda")

        # Use different devices for query, key, and value
        query, key, value = cpu_tensor, cuda_tensor, cpu_tensor

        expected_error_message = (
            "Expected query, key, and value to have the same device type, "
            f"but got query.device: {query.device}, key.device: {key.device}, "
            f"and value.device: {value.device} instead."
        )

        with self.assertRaisesRegex(ValueError, expected_error_message):
            flex_attention(query, key, value)

    @supported_platform
    def test_captured_wrong_device_error_message(self):
        means = torch.randn(64, 3).cuda()
        length_scales = torch.logspace(0.001, 0.1, 8)

        def euclidean_dist_pos_embed(score, b, h, q_idx, k_idx):
            q_pos = means[q_idx]
            k_pos = means[k_idx]
            dist = (q_pos - k_pos).pow(2).sum(-1).sqrt()
            scale = length_scales[h]
            inv_dist = torch.exp(-dist / scale)
            return inv_dist * score

        expected_error_message = "Buffers cannot be created"

        q, k, v = (torch.randn(1, 8, 64, 64, device="cuda") for _ in range(3))
        with self.assertRaisesRegex(RuntimeError, expected_error_message):
            torch.compile(flex_attention)(q, k, v, score_mod=euclidean_dist_pos_embed)

    @supported_platform
    def test_cant_lower_error_message(self):
        # We can't lower a 256-element reduction inside a pointwise reduction
        means = torch.randn(64, 256).cuda()
        length_scales = torch.logspace(0.001, 0.1, 8).cuda()

        def euclidean_dist_pos_embed(score, b, h, q_idx, k_idx):
            q_pos = means[q_idx]
            k_pos = means[k_idx]
            dist = (q_pos - k_pos).pow(2).sum(-1).sqrt()
            scale = length_scales[h]
            inv_dist = torch.exp(-dist / scale)
            return inv_dist * score

        expected_error_message = "Buffers cannot be created"

        q, k, v = (torch.randn(1, 8, 64, 64, device="cuda") for _ in range(3))
        with self.assertRaisesRegex(RuntimeError, expected_error_message):
            torch.compile(flex_attention)(q, k, v, score_mod=euclidean_dist_pos_embed)

    @supported_platform
    def test_reduction_unrolled(self):
        # We can't lower a 256-element reduction inside a pointwise reduction
        means = torch.randn(S, 3).cuda()
        length_scales = torch.logspace(0.001, 0.1, H).cuda()

        def euclidean_dist_pos_embed(score, b, h, q_idx, k_idx):
            q_pos = means[q_idx]
            k_pos = means[k_idx]
            dist = (q_pos - k_pos).pow(2).sum(-1).sqrt()
            scale = length_scales[h]
            inv_dist = torch.exp(-dist / scale)
            return inv_dist * score

        self.run_test(euclidean_dist_pos_embed, torch.bfloat16)

    @supported_platform
    def test_invalid_block_size(self):
        # Create tensors on different devices
        q, k, v = (torch.randn(1, 8, 128, 64, device="cuda") for _ in range(3))

        expected_error_message = (
            "ValueError: Q and KV block size must be divisible by BLOCK_M and BLOCK_N."
        )
        block_mask = create_block_mask(noop_mask, 1, 8, 128, 128, BLOCK_SIZE=96)

        with self.assertRaisesRegex(RuntimeError, expected_error_message):
            torch.compile(flex_attention)(q, k, v, block_mask=block_mask)

    @supported_platform
    def test_small_q_kv_len(self):
        make_tensor = functools.partial(
            torch.ones,
            (1, 1, 1, 16),
            device="cuda",
            dtype=torch.float32,
            requires_grad=True,
        )
        query, key, value = make_tensor(), make_tensor(), make_tensor()
        kernel_options = {"FORCE_USE_FLEX_ATTENTION": True}
        out_eager, lse_eager = flex_attention(
            query, key, value, return_lse=True, kernel_options=kernel_options
        )

        flex_compile = torch.compile(flex_attention, fullgraph=True)
        out_compiled, lse_compiled = flex_compile(
            query, key, value, return_lse=True, kernel_options=kernel_options
        )

        assert torch.equal(out_eager, out_compiled)
        assert torch.equal(lse_eager, lse_compiled)

        grads_eager = torch.autograd.grad(out_eager.sum(), (query, key, value))
        grads_compile = torch.autograd.grad(out_compiled.sum(), (query, key, value))

        torch.testing.assert_close(grads_eager, grads_compile)

    @supported_platform
    def test_dynamic_shapes_bug_dynamic_batch(self):
        def _flex_attention_mask(b, h, q_idx, kv_idx, input_lengths):
            padding_condition = (q_idx < input_lengths[b]) & (kv_idx < input_lengths[b])
            return padding_condition

        counter = CompileCounterWithBackend("inductor")

        class Model(torch.nn.Module):
            def __init__(self, dim=1024):
                super().__init__()
                self.subsampler = torch.nn.Conv1d(256, 256, 5)
                self.projector = torch.nn.Linear(256, dim)
                self.num_heads = 4

            def forward(self, x, input_lengths):
                x = self.subsampler(x.transpose(-1, -2)).transpose(-1, -2)
                x = self.projector(x).transpose(0, 1)
                head_dim = x.size(-1) // self.num_heads
                x = x.view(-1, x.size(1), self.num_heads, head_dim)
                x = x.permute(1, 2, 0, 3)

                max_time = x.size(-2)
                mask = torch.compile(create_block_mask, dynamic=True, fullgraph=False)(
                    functools.partial(
                        _flex_attention_mask,
                        input_lengths=input_lengths,
                    ),
                    B=input_lengths.size(0),
                    H=None,
                    Q_LEN=max_time,
                    KV_LEN=max_time,
                )

                x = torch.compile(
                    flex_attention, dynamic=True, fullgraph=True, backend=counter
                )(
                    query=x,
                    key=x,
                    value=x,
                    block_mask=mask,
                )
                return x

        model = Model(128).cuda()
        B, F, T = 16, 256, 12
        for _ in range(5):
            x = torch.randn(B, T, F, device="cuda")
            l = torch.randint(0, T, (B,), device="cuda")
            model(x, l)

        assert (
            counter.frame_count == 1
        ), f"Expected 1 graph, but got {counter.frame_count} graphs"

    @supported_platform
    def test_dynamic_shapes_with_custom_kernel_options(self):
        make_tensor = functools.partial(
            torch.ones,
            (8, 8, 1024, 64),
            device="cuda",
            dtype=torch.bfloat16,
        )
        query, key, value = make_tensor(), make_tensor(), make_tensor()
        kernel_options = {"BLOCK_M": 64, "BLOCK_N": 64}
        out_eager = flex_attention(query, key, value, kernel_options=kernel_options)

        flex_compile = torch.compile(flex_attention, fullgraph=True, dynamic=True)
        out_compiled = flex_compile(query, key, value, kernel_options=kernel_options)

        torch.testing.assert_close(out_eager, out_compiled, atol=3e-3, rtol=2e-3)

    @supported_platform
    def test_dynamic_shapes_with_max_autotune(self):
        make_tensor = functools.partial(
            torch.ones,
            (8, 8, 1024, 64),
            device="cuda",
            dtype=torch.bfloat16,
        )
        query, key, value = make_tensor(), make_tensor(), make_tensor()
        block_mask = create_block_mask(_causal_mask, None, None, 1024, 1024)

        out_eager = flex_attention(query, key, value, block_mask=block_mask)

        flex_compile = torch.compile(
            flex_attention, fullgraph=True, dynamic=True, mode="max-autotune"
        )
        out_compiled = flex_compile(query, key, value, block_mask=block_mask)

        torch.testing.assert_close(out_eager, out_compiled, atol=3e-3, rtol=2e-3)

    @supported_platform
    def test_zero_length_sequence_error(self):
        make_tensor = functools.partial(
            torch.ones,
            (8, 8, 0, 64),  # Zero in sequence dimension
            device="cuda",
            dtype=torch.bfloat16,
        )
        query, key, value = make_tensor(), make_tensor(), make_tensor()

        # Test compiled mode - should also raise assertion error
        flex_compile = torch.compile(flex_attention, fullgraph=True)
        with self.assertRaisesRegex(
            torch._inductor.exc.InductorError, "Query length must be greater than 0"
        ):
            flex_compile(query, key, value)

    @supported_platform
    def test_causal_block_non_divisible_with_captured_buffer(self):
        Q_S = S - 3
        KV_S = S - 3
        offset_q = torch.randn(Q_S, device="cuda", dtype=torch.bfloat16)
        offset_kv = torch.randn(KV_S, device="cuda", dtype=torch.bfloat16)

        def score_mod(score, b, h, q, kv):
            return score + offset_q[q] + offset_kv[kv]

        def mask_mod(b, h, q, kv):
            return q >= kv

        block_mask = create_block_mask(mask_mod, B, 1, Q_S, KV_S)

        attention = functools.partial(flex_attention, block_mask=block_mask)

        self.run_test_with_call(attention, Q_S=Q_S, KV_S=KV_S)

    @supported_platform
    def test_non_divisible_with_captured_buffer(self):
        Q_S = S + 3
        KV_S = S + 3

        multiplier = torch.randn(Q_S, device="cuda", dtype=torch.bfloat16)

        def apply_multiplicative_bias(score, b, h, q_idx, kv_idx):
            return score * multiplier[q_idx]

        attention = functools.partial(
            flex_attention, score_mod=apply_multiplicative_bias
        )

        self.run_test_with_call(attention, Q_S=Q_S, KV_S=KV_S)

    @supported_platform
    def test_num_warps_8_error(self):
        attention = functools.partial(flex_attention, score_mod=_identity)
        self.run_test_with_call(attention, Q_S=128, KV_S=128, Q_D=128, V_D=128)

    @unittest.skipIf(not TEST_MULTIGPU, "detected only one GPU")
    def test_qkv_and_block_mask_on_the_same_device(self):
        make_tensor = functools.partial(
            torch.ones,
            (2, 2, 256, 32),
            device="cuda:0",
            dtype=torch.float32,
            requires_grad=True,
        )
        query, key, value = make_tensor(), make_tensor(), make_tensor()

        def mask_mod(b, h, q, kv):
            return q >= kv

        block_mask = create_block_mask(mask_mod, 1, 1, 256, 256, device="cuda:1")
        with self.assertRaisesRegex(
            RuntimeError, "Expect q/k/v and block_mask to be on the same device"
        ):
            torch.compile(flex_attention)(query, key, value, block_mask=block_mask)

    @supported_platform
    def test_free_symbol_dynamic(self):
        def batch_flip_causal(b, h, q_idx, kv_idx):
            return (q_idx >= kv_idx) & (b % 2 == 0)

        class SimpleAttention(torch.nn.Module):
            def __init__(self, dim=512, n_head=8):
                super().__init__()
                self.qkv = torch.nn.Linear(dim, 3 * dim)
                self.n_head = n_head
                self.head_dim = dim // n_head

            def forward(self, x, block_mask=None):
                B, T, C = x.size()
                qkv = self.qkv(x).view(B, T, 3, self.n_head, self.head_dim)
                qkv = qkv.permute(2, 0, 3, 1, 4)
                q, k, v = qkv
                y = flex_attention(
                    q,
                    k,
                    v,
                    block_mask=block_mask,
                )
                return y.transpose(1, 2).contiguous().view(B, T, C)

        model = SimpleAttention().cuda()
        model.compile(mode="default", dynamic=True)
        sequence_len = 256

        # Test different batch shapes with dense masks
        torch._dynamo.reset()
        for batch_shape in [4, 16, 32]:
            # Create dense mask
            rand_mask = torch.randint(0, 2, (batch_shape, sequence_len)).cuda().bool()
            block_mask = torch.compile(create_block_mask, dynamic=True)(
                B=batch_shape,
                BLOCK_SIZE=128,
                mask_mod=lambda b, h, q_idx, kv_idx: ~rand_mask[b, q_idx],
                H=None,
                Q_LEN=sequence_len,
                KV_LEN=sequence_len,
                device="cuda",
            )

            # Run forward pass
            x = torch.randn(batch_shape, sequence_len, 512).cuda()
            model(x, block_mask=block_mask)

        self.assertEqual(torch._dynamo.utils.counters["aot_autograd"]["ok"], 2)

    @supported_platform
    def test_symbol_closure_in_score_mod(self):
        class SimpleAttention(torch.nn.Module):
            def __init__(self, dim=512, n_head=8):
                super().__init__()
                self.qkv = torch.nn.Linear(dim, 3 * dim)
                self.n_head = n_head
                self.head_dim = dim // n_head

            def forward(self, x, block_mask=None):
                B, T, C = x.size()
                qkv = self.qkv(x).view(B, T, 3, self.n_head, self.head_dim)
                qkv = qkv.permute(2, 0, 3, 1, 4)
                q, k, v = qkv
                return flex_attention(
                    q,
                    k,
                    v,
                    score_mod=lambda s, b, h, q, k: s + B,
                    block_mask=block_mask,
                )

        model = SimpleAttention().cuda()
        from torch._dynamo.testing import EagerAndRecordGraphs

        backend = EagerAndRecordGraphs()
        model.compile(mode="default", dynamic=True, backend=backend)
        sequence_len = 256

        torch._dynamo.reset()
        for batch_shape in [4, 16, 32]:
            x = torch.randn(batch_shape, sequence_len, 512).cuda()
            model(x)
        self.assertEqual(len(backend.graphs), 1)
        self.assertExpectedInline(
            backend.graphs[0].score_mod_0.code.strip(),
            """\
def forward(self, child : torch.Tensor, child_1 : torch.Tensor, child_2 : torch.Tensor, child_3 : torch.Tensor, child_4 : torch.Tensor, getitem : torch.SymInt):
    add = child + getitem;  child = getitem = None
    return add""",
        )

    @supported_platform
    def test_fw_bw_graph_correctness(self):
        cnt = CompileCounterWithBackend("aot_eager")
        make_tensor = functools.partial(
            torch.randn,
            (2, 2, 128, 4),
            device="cuda",
            dtype=torch.float64,
            requires_grad=True,
        )
        query, key, value = make_tensor(), make_tensor(), make_tensor()

        def causal_mask(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx

        block_mask = create_block_mask(causal_mask, 1, 1, 128, 128)

        func = torch.compile(flex_attention, backend=cnt, fullgraph=True)
        out = func(query, key, value, _squared, block_mask=block_mask)
        out.sum().backward()
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(len(cnt.graphs), 1)
        graph = cnt.graphs[0]
        norm_graph = normalize_gm(graph.print_readable(print_output=False))

        self.assertExpectedInline(
            norm_graph,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_query_: "f64[2, 2, 128, 4]", L_key_: "f64[2, 2, 128, 4]", L_value_: "f64[2, 2, 128, 4]", L_block_mask_kv_indices: "i32[1, 1, 1, 1]", L_block_mask_kv_num_blocks: "i32[1, 1, 1]", L_block_mask_full_kv_num_blocks: "i32[1, 1, 1]", L_block_mask_full_kv_indices: "i32[1, 1, 1, 1]", L_block_mask_q_num_blocks: "i32[1, 1, 1]", L_block_mask_q_indices: "i32[1, 1, 1, 1]", L_block_mask_full_q_num_blocks: "i32[1, 1, 1]", L_block_mask_full_q_indices: "i32[1, 1, 1, 1]"):
        l_query_ = L_query_
        l_key_ = L_key_
        l_value_ = L_value_
        l_block_mask_kv_indices = L_block_mask_kv_indices
        l_block_mask_kv_num_blocks = L_block_mask_kv_num_blocks
        l_block_mask_full_kv_num_blocks = L_block_mask_full_kv_num_blocks
        l_block_mask_full_kv_indices = L_block_mask_full_kv_indices
        l_block_mask_q_num_blocks = L_block_mask_q_num_blocks
        l_block_mask_q_indices = L_block_mask_q_indices
        l_block_mask_full_q_num_blocks = L_block_mask_full_q_num_blocks
        l_block_mask_full_q_indices = L_block_mask_full_q_indices

        score_mod_0 = self.score_mod_0
        mask_fn_0 = self.mask_fn_0
        flex_attention = torch.ops.higher_order.flex_attention(l_query_, l_key_, l_value_, score_mod_0, (128, 128, l_block_mask_kv_num_blocks, l_block_mask_kv_indices, l_block_mask_full_kv_num_blocks, l_block_mask_full_kv_indices, l_block_mask_q_num_blocks, l_block_mask_q_indices, l_block_mask_full_q_num_blocks, l_block_mask_full_q_indices, 128, 128, mask_fn_0), 0.5, {'PRESCALE_QK': False, 'ROWS_GUARANTEED_SAFE': False, 'BLOCKS_ARE_CONTIGUOUS': False, 'WRITE_DQ': True, 'OUTPUT_LOGSUMEXP': True}, (), ());  l_query_ = l_key_ = l_value_ = score_mod_0 = l_block_mask_kv_num_blocks = l_block_mask_kv_indices = l_block_mask_full_kv_num_blocks = l_block_mask_full_kv_indices = l_block_mask_q_num_blocks = l_block_mask_q_indices = l_block_mask_full_q_num_blocks = l_block_mask_full_q_indices = mask_fn_0 = None
        out: "f64[2, 2, 128, 4]" = flex_attention[0];  flex_attention = None
        return (out,)

    class score_mod_0(torch.nn.Module):
        def forward(self, child: "f64[]", child_1: "i32[]", child_2: "i32[]", child_3: "i32[]", child_4: "i32[]"):
            mul: "f64[]" = child * child;  child = None
            return mul

    class mask_fn_0(torch.nn.Module):
        def forward(self, child: "i32[]", child_1: "i32[]", child_2: "i32[]", child_3: "i32[]"):
            ge: "b8[]" = child_2 >= child_3;  child_2 = child_3 = None
            return ge
""",  # noqa: B950
        )
        # Save the AOT graphs
        aot_graphs = []
        from torch._inductor import compile_fx

        def debug_compile_fx_inner(graph, example_inputs, *args, **kwargs):
            aot_graphs.append(graph)
            return graph

        backend = functools.partial(
            compile_fx.compile_fx, inner_compile=debug_compile_fx_inner
        )
        func = torch.compile(func, backend=backend, fullgraph=True)
        out = func(query, key, value, _squared)
        out.sum().backward()

        joint_graph = normalize_gm(aot_graphs[1].print_readable(print_output=False))

        self.assertExpectedInline(
            joint_graph,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f64[2, 2, 128, 4]", primals_2: "f64[2, 2, 128, 4]", primals_3: "f64[2, 2, 128, 4]", full: "i32[1, 1, 1]", full_default: "i32[1, 1, 1, 1]", convert_element_type: "i32[1, 1, 1]", convert_element_type_1: "i32[1, 1, 1, 1]", getitem_2: "f64[2, 2, 128, 4]", getitem_3: "f32[2, 2, 128]", tangents_1: "f64[2, 2, 128, 4]"):
        full_default_4: "f32[2, 2, 128]" = torch.ops.aten.full.default([2, 2, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        fw_graph0 = self.fw_graph0
        joint_graph0 = self.joint_graph0
        mask_graph0 = self.mask_graph0
        flex_attention_backward = torch.ops.higher_order.flex_attention_backward(primals_1, primals_2, primals_3, getitem_2, getitem_3, tangents_1, full_default_4, fw_graph0, joint_graph0, (1, 1, full, full_default, None, None, convert_element_type, convert_element_type_1, None, None, 1073741824, 1073741824, mask_graph0), 0.5, {'PRESCALE_QK': False, 'ROWS_GUARANTEED_SAFE': False, 'BLOCKS_ARE_CONTIGUOUS': False, 'WRITE_DQ': True, 'OUTPUT_LOGSUMEXP': True}, (), ());  primals_1 = primals_2 = primals_3 = getitem_2 = getitem_3 = tangents_1 = full_default_4 = fw_graph0 = joint_graph0 = full = full_default = convert_element_type = convert_element_type_1 = mask_graph0 = None
        getitem_4: "f64[2, 2, 128, 4]" = flex_attention_backward[0]
        getitem_5: "f64[2, 2, 128, 4]" = flex_attention_backward[1]
        getitem_6: "f64[2, 2, 128, 4]" = flex_attention_backward[2];  flex_attention_backward = None
        return (getitem_4, getitem_5, getitem_6)

    class fw_graph0(torch.nn.Module):
        def forward(self, arg0_1: "f64[]", arg1_1: "i32[]", arg2_1: "i32[]", arg3_1: "i32[]", arg4_1: "i32[]"):
            mul: "f64[]" = torch.ops.aten.mul.Tensor(arg0_1, arg0_1);  arg0_1 = None
            return mul

    class joint_graph0(torch.nn.Module):
        def forward(self, arg0_1: "f64[]", arg1_1: "i32[]", arg2_1: "i32[]", arg3_1: "i32[]", arg4_1: "i32[]", arg5_1: "f64[]"):
            mul: "f64[]" = torch.ops.aten.mul.Tensor(arg0_1, arg0_1);  mul = None
            mul_1: "f64[]" = torch.ops.aten.mul.Tensor(arg5_1, arg0_1)
            mul_2: "f64[]" = torch.ops.aten.mul.Tensor(arg5_1, arg0_1);  arg5_1 = arg0_1 = None
            add: "f64[]" = torch.ops.aten.add.Tensor(mul_2, mul_1);  mul_2 = mul_1 = None
            return [add, None, None, None, None]

    class mask_graph0(torch.nn.Module):
        def forward(self, arg0_1: "i32[]", arg1_1: "i32[]", arg2_1: "i32[]", arg3_1: "i32[]"):
            full: "b8[]" = torch.ops.aten.full.default([], True, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
            return full
""",  # noqa: B950
        )

    @unittest.skipIf(TEST_ON_CUDA, "Testing CPU error message")
    def test_cpu_error_message_return_lse(self):
        make_tensor = functools.partial(
            torch.randn,
            (2, 2, 128, 16),
            device="cpu",
            dtype=torch.float32,
            requires_grad=False,
        )
        query, key, value = make_tensor(), make_tensor(), make_tensor()
        attention = torch.compile(flex_attention)
        with self.assertRaisesRegex(
            torch._inductor.exc.InductorError,
            r"NotImplementedError: torch.compile on CPU only supports inference and `return_lse` is not supported yet.",
        ):
            attention(query, key, value, return_lse=True)

    @unittest.skipIf(TEST_ON_CUDA, "Testing CPU error message")
    def test_validate_cpu_dtype_error_message(self):
        make_tensor = functools.partial(
            torch.randn,
            (2, 2, 128, 16),
            device="cpu",
            dtype=torch.half,
            requires_grad=False,
        )
        query, key, value = make_tensor(), make_tensor(), make_tensor()
        attention = torch.compile(flex_attention)
        with self.assertRaisesRegex(
            torch._inductor.exc.InductorError,
            r"`torch.float` and `torch.bfloat16` are supported in FlexAttention for CPU device. Found input tensors are `torch.float16`.",
        ):
            attention(query, key, value)

    @unittest.skipIf(not TEST_MULTIGPU, "detected only one GPU")
    def test_device_cuda_1(self):
        class TestModule(torch.nn.Module):
            def forward(self, q, k, v, block_mask):
                return flex_attention(q, k, v, block_mask=block_mask)

        q = torch.randn(1, 1, 256, 32, device="cuda:1", dtype=torch.bfloat16)
        k = torch.randn(1, 1, 256, 32, device="cuda:1", dtype=torch.bfloat16)
        v = torch.randn(1, 1, 256, 32, device="cuda:1", dtype=torch.bfloat16)
        mask = create_block_mask(
            lambda b, h, q_idx, kv_idx: q_idx >= kv_idx,
            B=None,
            H=None,
            Q_LEN=256,
            KV_LEN=256,
            device="cuda:1",
        )
        mod = torch.compile(TestModule())
        attn_output = mod(q, k, v, mask)
        self.assertEqual(attn_output.device, torch.device("cuda:1"))


class TestBlockMask(InductorTestCase):
    @supported_platform
    def test_block_mask_attributes(self):
        offset = torch.zeros(8, device="cuda")

        def causal_mask(b, h, q, kv):
            return (q + (offset[b] * 128)) >= kv

        block_mask = create_block_mask(causal_mask, 4, 2, 2048, 2048)
        self.assertEqual(block_mask.shape, (4, 2, 2048, 2048))
        self.assertEqual(block_mask[0].shape, (2, 2048, 2048))
        self.assertEqual(block_mask[0, 0].shape, (2048, 2048))
        self.assertEqual(block_mask.numel(), 4 * 2 * 2048 * 2048)
        self.assertEqual(block_mask.sparsity(), 46.875)
        self.assertEqual(block_mask[0].sparsity(), 46.875)
        self.assertEqual(block_mask[1, 0].sparsity(), 46.875)
        self.assertEqual(block_mask.sparsity(), block_mask[1].sparsity())

        offset = torch.arange(8, device="cuda")
        block_mask = create_block_mask(causal_mask, 8, 1, 2048, 2048)
        self.assertEqual(block_mask.sparsity(), 29.1015625)
        self.assertTrue(block_mask.sparsity() < block_mask[0].sparsity())
        self.assertTrue(block_mask[0].sparsity() > block_mask[1].sparsity())

    @supported_platform
    @common_utils.parametrize("BLOCK_SIZE", [32, 64, 128, 256, (32, 64), (64, 32)])
    def test_block_size_changes(self, BLOCK_SIZE: Union[int, tuple[int, int]]):
        B, H, Q_LEN, KV_LEN = 4, 2, 2048, 2048

        if isinstance(BLOCK_SIZE, int):
            Q_BLOCK_SIZE = BLOCK_SIZE
            KV_BLOCK_SIZE = BLOCK_SIZE
        else:
            Q_BLOCK_SIZE, KV_BLOCK_SIZE = BLOCK_SIZE

        block_mask = create_block_mask(
            noop_mask, B, H, Q_LEN, KV_LEN, BLOCK_SIZE=BLOCK_SIZE
        )

        self.assertEqual(block_mask.BLOCK_SIZE, (Q_BLOCK_SIZE, KV_BLOCK_SIZE))
        self.assertEqual(block_mask.shape, (B, H, Q_LEN, KV_LEN))

    @supported_platform
    def test_getitem(self):
        offset = torch.zeros(8, device="cuda")

        def causal_mask(b, h, q, kv):
            return (q + (offset[b] * 128)) >= kv

        block_mask = create_block_mask(causal_mask, 4, 2, 512, 512)
        assert block_mask.kv_num_blocks.shape == (4, 2, 4)
        assert block_mask.kv_indices.shape == (4, 2, 4, 4)

        # Index on batch dimension
        new_block_mask = block_mask[0]
        assert new_block_mask.kv_num_blocks.shape == (2, 4)
        assert new_block_mask.kv_indices.shape == (2, 4, 4)

        # Index on batch and head dimension
        new_block_mask = block_mask[0, 1]
        assert new_block_mask.kv_num_blocks.shape == (4,)
        assert new_block_mask.kv_indices.shape == (4, 4)

        # slicing on batch and head dimension
        new_block_mask = block_mask[0:2, 1:2]
        assert new_block_mask.kv_num_blocks.shape == (2, 1, 4)
        assert new_block_mask.kv_indices.shape == (2, 1, 4, 4)

        # slicing on batch, head, and query dimension
        new_block_mask = block_mask[0:2, 1:2, torch.tensor([1], dtype=torch.int32)]
        assert new_block_mask.kv_num_blocks.shape == (2, 1, 1)
        assert new_block_mask.kv_indices.shape == (2, 1, 1, 4)

        # slicing on batch, head, and query dimension
        q_index = torch.tensor([0], dtype=torch.int32)
        new_block_mask = block_mask[:, :, q_index]

        self.assertEqual(new_block_mask.kv_num_blocks.ndim, 3)
        self.assertEqual(new_block_mask.kv_indices.ndim, 4)
        torch.testing.assert_close(
            new_block_mask.kv_num_blocks,
            block_mask.kv_num_blocks[:, :, q_index],
        )
        torch.testing.assert_close(
            new_block_mask.kv_indices, block_mask.kv_indices[:, :, q_index, :]
        )

        if block_mask.full_kv_num_blocks is not None:
            assert new_block_mask.full_kv_num_blocks is not None
            assert new_block_mask.full_kv_indices is not None
            torch.testing.assert_close(
                new_block_mask.full_kv_num_blocks,
                block_mask.full_kv_num_blocks[:, :, q_index],
            )
            torch.testing.assert_close(
                new_block_mask.full_kv_indices,
                block_mask.full_kv_indices[:, :, q_index, :],
            )

    @supported_platform
    def test_block_mask_device_change(self):
        offset = torch.zeros(8, device="cuda")

        def causal_mask(b, h, q, kv):
            return (q + (offset[b] * 128)) >= kv

        block_mask = create_block_mask(causal_mask, 1, 1, 512, 512)
        assert block_mask.kv_indices.is_cuda
        assert block_mask.kv_num_blocks.is_cuda
        assert block_mask.q_indices.is_cuda
        assert block_mask.q_num_blocks.is_cuda

        block_mask = block_mask.to("cpu")
        assert block_mask.kv_indices.is_cpu
        assert block_mask.kv_num_blocks.is_cpu
        assert block_mask.q_indices.is_cpu
        assert block_mask.q_num_blocks.is_cpu

        block_mask = block_mask.to("cuda")
        assert block_mask.kv_indices.is_cuda
        assert block_mask.kv_num_blocks.is_cuda
        assert block_mask.q_indices.is_cuda
        assert block_mask.q_num_blocks.is_cuda

    @supported_platform
    def test_compiling_create_block_mask(self):
        seq = torch.arange(512, device="cuda") // 127

        def mask_mod(b, h, q, kv):
            return (q >= kv) & (seq[q] == seq[kv])

        block_mask = torch.compile(create_block_mask, fullgraph=True)(
            mask_mod, 1, 1, 512, 512
        )
        self.assertIsInstance(block_mask, BlockMask)
        self.assertEqual(block_mask.kv_num_blocks.shape, torch.Size((1, 1, 4)))
        self.assertEqual(block_mask.kv_indices.shape, torch.Size((1, 1, 4, 4)))

    @supported_platform
    def test_compiling_create_block_mask_no_recompile(self):
        def mask_mod(b, h, q, kv):
            return q >= kv

        torch._dynamo.reset()
        block_mask = torch.compile(create_block_mask)(mask_mod, 2, 4, 1024, 1024)
        self.assertIsInstance(block_mask, BlockMask)
        self.assertEqual(block_mask.kv_num_blocks.shape, torch.Size((2, 4, 8)))
        self.assertEqual(block_mask.kv_indices.shape, torch.Size((2, 4, 8, 8)))
        self.assertEqual(torch._dynamo.utils.counters["aot_autograd"]["ok"], 1)

        # automatic dynamic shapes triggered and recompilation.
        block_mask = torch.compile(create_block_mask)(mask_mod, 4, 8, 2048, 2048)
        self.assertIsInstance(block_mask, BlockMask)
        self.assertEqual(block_mask.kv_num_blocks.shape, torch.Size((4, 8, 16)))
        self.assertEqual(block_mask.kv_indices.shape, torch.Size((4, 8, 16, 16)))
        self.assertEqual(torch._dynamo.utils.counters["aot_autograd"]["ok"], 2)

        # no recompilation.
        block_mask = torch.compile(create_block_mask)(mask_mod, 6, 16, 3072, 3072)
        self.assertIsInstance(block_mask, BlockMask)
        self.assertEqual(block_mask.kv_num_blocks.shape, torch.Size((6, 16, 24)))
        self.assertEqual(block_mask.kv_indices.shape, torch.Size((6, 16, 24, 24)))
        self.assertEqual(torch._dynamo.utils.counters["aot_autograd"]["ok"], 2)

    @supported_platform
    def test_block_mask_viz(self):
        def causal_mask(b, h, q, kv):
            return q >= kv

        block_mask = create_block_mask(causal_mask, 1, 1, 2048, 2048)

        def replace_non_printable(s):
            def replace(c):
                if c not in string.printable:
                    return "@"
                elif c == " ":
                    return "s"
                return c

            return "".join(replace(c) for c in s)

        self.assertExpectedInline(
            replace_non_printable(str(block_mask)),
            """\
BlockMask(shape=(1,s1,s2048,s2048),ssparsity=46.88%,s
(0,s0)
@@ssssssssssssssssssssssssssssss
@@@@ssssssssssssssssssssssssssss
@@@@@@ssssssssssssssssssssssssss
@@@@@@@@ssssssssssssssssssssssss
@@@@@@@@@@ssssssssssssssssssssss
@@@@@@@@@@@@ssssssssssssssssssss
@@@@@@@@@@@@@@ssssssssssssssssss
@@@@@@@@@@@@@@@@ssssssssssssssss
@@@@@@@@@@@@@@@@@@ssssssssssssss
@@@@@@@@@@@@@@@@@@@@ssssssssssss
@@@@@@@@@@@@@@@@@@@@@@ssssssssss
@@@@@@@@@@@@@@@@@@@@@@@@ssssssss
@@@@@@@@@@@@@@@@@@@@@@@@@@ssssss
@@@@@@@@@@@@@@@@@@@@@@@@@@@@ssss
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ss
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
)""",
        )

        offset = torch.arange(8, device="cuda")

        def causal_offset_mask(b, h, q, kv):
            return (q + offset[b] * 128) >= kv

        block_mask = create_block_mask(causal_offset_mask, 8, 1, 2048, 2048)
        str_block_mask = str(block_mask)
        self.assertTrue("sparsity=29.10" in str_block_mask)

    def generate_test_inputs(self, full_seq_len: bool, device):
        if full_seq_len:
            kv_num_blocks = torch.tensor([1], dtype=torch.int32, device=device).view(
                1, 1, 1
            )
            kv_indices = torch.tensor([1, -1], dtype=torch.int32, device=device).view(
                1, 1, 1, 2
            )
            full_kv_num_blocks = torch.tensor(
                [1], dtype=torch.int32, device=device
            ).view(1, 1, 1)
            full_kv_indices = torch.tensor(
                [0, -1], dtype=torch.int32, device=device
            ).view(1, 1, 1, 2)
        else:
            kv_num_blocks = torch.tensor([2], dtype=torch.int32, device=device).view(
                1, 1, 1
            )
            kv_indices = torch.tensor([0, 1], dtype=torch.int32, device=device).view(
                1, 1, 1, 2
            )
            full_kv_indices = None
            full_kv_num_blocks = None
        return kv_num_blocks, kv_indices, full_kv_num_blocks, full_kv_indices

    @supported_platform
    @common_utils.parametrize("full_indices", [False, True])
    def test_from_kv_blocks(self, full_indices: bool):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        (
            kv_num_blocks,
            kv_indices,
            full_kv_num_blocks,
            full_kv_indices,
        ) = self.generate_test_inputs(full_indices, device=device)

        block_mask = BlockMask.from_kv_blocks(
            kv_num_blocks, kv_indices, full_kv_num_blocks, full_kv_indices
        )

        self.assertIsInstance(block_mask, BlockMask)
        torch.testing.assert_close(block_mask.kv_num_blocks, kv_num_blocks)
        torch.testing.assert_close(block_mask.kv_indices, kv_indices)

        if full_indices:
            torch.testing.assert_close(
                block_mask.full_kv_num_blocks, full_kv_num_blocks
            )
            torch.testing.assert_close(block_mask.full_kv_indices, full_kv_indices)
            torch.testing.assert_close(
                block_mask.q_num_blocks,
                torch.tensor([0, 1], dtype=torch.int32, device=device).view(1, 1, 2),
            )
            torch.testing.assert_close(
                block_mask.q_indices,
                torch.tensor([0, 0], dtype=torch.int32, device=device).view(1, 1, 2, 1),
            )
            torch.testing.assert_close(
                block_mask.full_q_num_blocks,
                torch.tensor([1, 0], dtype=torch.int32, device=device).view(1, 1, 2),
            )
            torch.testing.assert_close(
                block_mask.full_q_indices,
                torch.tensor([0, 0], dtype=torch.int32, device=device).view(1, 1, 2, 1),
            )

        else:
            torch.testing.assert_close(
                block_mask.q_num_blocks,
                torch.tensor([1, 1], dtype=torch.int32, device=device).view(1, 1, 2),
            )
            torch.testing.assert_close(
                block_mask.q_indices,
                torch.tensor([0, 0], dtype=torch.int32, device=device).view(1, 1, 2, 1),
            )
            self.assertIsNone(block_mask.full_kv_num_blocks)
            self.assertIsNone(block_mask.full_kv_indices)
            self.assertIsNone(block_mask.full_q_num_blocks)
            self.assertIsNone(block_mask.full_q_indices)

    @supported_platform
    def test_block_size(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        kv_num_blocks, kv_indices, _, _ = self.generate_test_inputs(False, device)
        block_mask = BlockMask.from_kv_blocks(kv_num_blocks, kv_indices)
        self.assertEqual(
            block_mask.BLOCK_SIZE,
            (_DEFAULT_SPARSE_BLOCK_SIZE, _DEFAULT_SPARSE_BLOCK_SIZE),
        )

        custom_block_size = (64, 64)
        block_mask_custom = BlockMask.from_kv_blocks(
            kv_num_blocks, kv_indices, BLOCK_SIZE=custom_block_size
        )
        self.assertEqual(block_mask_custom.BLOCK_SIZE, custom_block_size)

    @supported_platform
    def test_upcast_appropriately(self):
        q = torch.randn((1, 1, 128, 16), dtype=torch.float16, device="cuda")
        k = torch.randn((1, 1, 128, 16), dtype=torch.float16, device="cuda")
        v = torch.randn((1, 1, 128, 16), dtype=torch.float16, device="cuda")
        mass = torch.ones((1), dtype=torch.float16, device="cuda")

        def score_mod(score, b, h, q_idx, kv_idx):
            return score + torch.log(mass[0])

        torch.compile(flex_attention)(q, k, v, score_mod=score_mod)

    @supported_platform
    def test_init_mismatched_full_kv(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        kv_num_blocks, kv_indices, full_kv_num_blocks, _ = self.generate_test_inputs(
            True, device
        )

        with self.assertRaises(AssertionError):
            BlockMask(
                kv_num_blocks=kv_num_blocks,
                kv_indices=kv_indices,
                full_kv_num_blocks=full_kv_num_blocks,
                full_kv_indices=None,  # Mismatched, should raise error
                q_num_blocks=kv_num_blocks,
                q_indices=kv_indices,
                full_q_num_blocks=None,
                full_q_indices=None,
                BLOCK_SIZE=(64, 64),
                mask_mod=noop_mask,
                seq_lengths=(1, 1),
            )

    @supported_platform
    def test_init_mismatched_full_q(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        kv_num_blocks, kv_indices, _, _ = self.generate_test_inputs(False, device)

        with self.assertRaises(AssertionError):
            BlockMask(
                kv_num_blocks=kv_num_blocks,
                kv_indices=kv_indices,
                full_kv_num_blocks=None,
                full_kv_indices=None,
                q_num_blocks=kv_num_blocks,
                q_indices=kv_indices,
                full_q_num_blocks=kv_num_blocks,
                full_q_indices=None,  # Mismatched, should raise error
                BLOCK_SIZE=(64, 64),
                mask_mod=noop_mask,
                seq_lengths=(1, 1),
            )

    @supported_platform
    @common_utils.parametrize("compile", [False, True])
    def test_no_q_info(self, compile: bool):
        def causal_mask(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx

        block_mask = create_block_mask(causal_mask, 1, 1, 2048, 2048)
        # manually set q_num_blocks and q_indices to None
        block_mask.q_num_blocks = None
        block_mask.q_indices = None
        block_mask.full_q_num_blocks = None
        block_mask.full_q_indices = None

        mask_mod_sparse_flex = functools.partial(flex_attention, block_mask=block_mask)
        if compile:
            mask_mod_sparse_flex = torch.compile(
                mask_mod_sparse_flex, backend="inductor"
            )
        inputs = [
            torch.randn(
                2,
                2,
                2048,
                64,
                device="cuda",
                dtype=torch.float16,
                requires_grad=True,
            )
            for _ in range(3)
        ]

        causal_mask_out = mask_mod_sparse_flex(*inputs)
        sdpa_mask_out = torch.nn.functional.scaled_dot_product_attention(
            *inputs, is_causal=True
        )

        torch.testing.assert_close(causal_mask_out, sdpa_mask_out, atol=5e-3, rtol=0.0)

    @supported_platform
    def test_doc_mask_clamped_repro(self):
        def _offsets_to_doc_ids_tensor(offsets):
            device = offsets.device
            counts = offsets[1:] - offsets[:-1]
            return torch.repeat_interleave(
                torch.arange(len(counts), device=device, dtype=torch.int32), counts
            )

        def length_to_offsets(
            lengths: list[int], device: Union[str, torch.device]
        ) -> Tensor:
            offsets = [0]
            offsets.extend(lengths)
            offsets = torch.tensor(offsets, device=device, dtype=torch.int32)
            offsets = torch.cumsum(offsets, dim=-1)
            return offsets

        def generate_doc_mask_mod(offsets: Tensor) -> _mask_mod_signature:
            document_id = _offsets_to_doc_ids_tensor(offsets)

            def doc_mask_mod(b, h, q_idx, kv_idx):
                same_doc = document_id[q_idx] == document_id[kv_idx]
                return same_doc

            return doc_mask_mod

        random.seed(0)

        def generate_random_lengths(total_length, num_documents):
            lengths = [1] * num_documents
            remaining_length = total_length - num_documents
            for _ in range(remaining_length):
                index = random.randint(0, num_documents - 1)
                lengths[index] += 1
            return lengths

        device = "cuda"
        max_seq_len, doc_count = 128, 4
        SEQ_LEN = max_seq_len

        lengths = generate_random_lengths(max_seq_len, doc_count)
        offsets = length_to_offsets(lengths, device)

        document_causal_mask = generate_doc_mask_mod(offsets)
        block_mask_compiled = torch.compile(create_block_mask)(
            document_causal_mask,
            1,
            1,
            SEQ_LEN,
            SEQ_LEN,
            device=device,
        )
        block_mask = torch.compile(create_block_mask)(
            document_causal_mask,
            1,
            1,
            SEQ_LEN,
            SEQ_LEN,
            device=device,
        )
        self.assertEqual(block_mask_compiled.kv_indices, block_mask.kv_indices)
        self.assertEqual(
            block_mask_compiled.full_kv_indices, block_mask.full_kv_indices
        )
        for i in range(5):
            lengths = generate_random_lengths(1024 + i, 5)
            offsets = length_to_offsets(lengths, "cuda")
            doc_ids = _offsets_to_doc_ids_tensor(offsets)

            def doc_mask_mod(b, h, q_idx, kv_idx):
                return (
                    doc_ids[q_idx.clamp(0, doc_ids.shape[0] - 1)]
                    == doc_ids[kv_idx.clamp(0, doc_ids.shape[0] - 1)]
                )

            q, k, v = (
                torch.randn(1, 12, 1024 + i, 64, device=device) for _ in range(3)
            )
            block_mask = create_block_mask(doc_mask_mod, None, None, 1024 + i, 1024 + i)
            torch.compile(flex_attention)(q, k, v, block_mask=block_mask)

    @supported_platform
    def test_eager_tracing_correctness(self):
        qk_dims = 64
        v_dims = 128
        q_heads = 4
        kv_heads = 2
        seq_len = 256
        batch_size = 1

        make_tensor = functools.partial(torch.randn, device="cuda", dtype=torch.float16)
        q = make_tensor(*(batch_size, q_heads, seq_len, qk_dims))
        k = make_tensor(*(batch_size, kv_heads, seq_len, qk_dims))
        v = make_tensor(*(batch_size, kv_heads, seq_len, v_dims))

        def flex_attention_fn():
            out = flex_attention(q, k, v, enable_gqa=True)
            return out.view(batch_size, q_heads, seq_len, 2, 64)

        # Run with compilation
        compiled_fn = torch.compile(flex_attention_fn, fullgraph=True)
        result = compiled_fn()

        # Assert expected output shape
        expected_shape = (batch_size, q_heads, seq_len, 2, 64)
        self.assertEqual(
            result.shape,
            expected_shape,
            f"Expected output shape {expected_shape}, but got {result.shape}",
        )

    @supported_platform
    def test_create_is_cuda_graphable(self):
        def mask_mod(b, h, q, kv):
            return q >= kv

        g = torch.cuda.CUDAGraph()

        with torch.cuda.graph(g):
            create_block_mask(mask_mod, None, None, 256, 256)

        g.replay()

    @common_utils.parametrize("compile", [False, True])
    @supported_platform
    def test_block_mask_vs_sequence_lengths(self, compile):
        if compile:
            flex_attention_call = torch.compile(flex_attention)
        else:
            flex_attention_call = flex_attention

        def mask_mod(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx

        def create_inputs(S):
            q, k, v = (
                torch.randn(
                    1, 8, S, 64, dtype=torch.float16, requires_grad=True, device="cuda"
                )
                for _ in range(3)
            )
            return q, k, v

        block_mask = create_block_mask(mask_mod, None, None, 1024, 1024)
        flex_attention_call(*create_inputs(1024), block_mask=block_mask)
        with self.assertRaisesRegex(ValueError, "block_mask was created for"):
            flex_attention_call(*create_inputs(2048), block_mask=block_mask)

        block_mask = create_block_mask(mask_mod, None, None, 1023, 1023)
        with self.assertRaisesRegex(ValueError, "block_mask was created for"):
            flex_attention_call(*create_inputs(1024), block_mask=block_mask)


class TestPagedAttention(InductorTestCase):
    def setUp(self):
        super().setUp()
        self.device = test_device
        if self.device == "cpu":
            if LONG_COMPILATION_ON_CPU:
                self.skipTest(
                    "skip UT for CPU due to long compilation time found in CI"
                )
            if not IS_PLATFORM_SUPPORTED:
                self.skipTest("skip UT due to not support on those platforms")

    def _check_equal(
        self,
        golden_out: torch.Tensor,
        ref_out: torch.Tensor,
        compiled_out: torch.Tensor,
        fudge_factor: float,
        tensor_name: Optional[str] = None,
    ):
        compiled_error = (golden_out - compiled_out).abs().mean()
        ref_error = (golden_out - ref_out).abs().mean()
        if torch.isnan(compiled_error).any() or torch.isnan(ref_error).any():
            self.assertTrue(False, "Output/Grad with NaN")
        if compiled_error > ref_error * fudge_factor:
            name = tensor_name if tensor_name is not None else ""
            msg = f"{name} Compiled error {compiled_error} is greater than ref error {ref_error} by more than {fudge_factor}X."
            self.assertTrue(False, msg)

    def allocate_page_cache(self, n_pages: int, page_size: int):
        max_batch_size = 3
        paged_cache = PagedAttention(n_pages, page_size, max_batch_size)
        return paged_cache

    def cdiv(self, x, y):
        return (x + y - 1) // y

    def roundup(self, x, y):
        return (x + y - 1) // y * y

    @supported_platform
    def test_page_allocation(self):
        n_pages, page_size = 12, 4
        paged_cache = self.allocate_page_cache(n_pages, page_size)

        batch_reserve(paged_cache, torch.tensor([8, 24, 16]))

        with self.assertRaisesRegex(
            AssertionError, "requested 2 pages but there are only 0 empty pages"
        ):
            paged_cache.reserve(
                torch.tensor([0], device="cuda"), torch.tensor([16], device="cuda")
            )

        paged_cache.erase(torch.tensor([1], device="cuda"))
        paged_cache.reserve(
            torch.tensor([0], device="cuda"), torch.tensor([16], device="cuda")
        )

    @supported_platform
    def test_allocate(self):
        n_pages, page_size = 12, 4
        paged_cache = self.allocate_page_cache(n_pages, page_size)

        target_seq_len = torch.tensor([3, 11, 8])
        batch_reserve(paged_cache, target_seq_len)

        expected_allocated_pages = self.cdiv(target_seq_len, page_size).sum()
        self.assertEqual(paged_cache.capacity, self.roundup(target_seq_len, page_size))
        self.assertEqual(
            len(paged_cache.empty_pages), n_pages - expected_allocated_pages
        )

        # deallocate batch 1
        paged_cache.erase(torch.tensor([1], device="cuda"))
        target_seq_len = torch.tensor([3, 0, 8])
        expected_allocated_pages = self.cdiv(target_seq_len, page_size).sum()
        self.assertEqual(paged_cache.capacity, self.roundup(target_seq_len, page_size))
        self.assertEqual(
            len(paged_cache.empty_pages), n_pages - expected_allocated_pages
        )

        # re-allocate
        target_seq_len = torch.tensor([7, 2, 10])
        batch_reserve(paged_cache, target_seq_len)
        expected_allocated_pages = self.cdiv(target_seq_len, page_size).sum()
        self.assertEqual(paged_cache.capacity, self.roundup(target_seq_len, page_size))
        self.assertEqual(
            len(paged_cache.empty_pages), n_pages - expected_allocated_pages
        )

        # deallocate all batches
        paged_cache.erase(torch.tensor([0, 1, 2]))
        self.assertEqual(paged_cache.capacity, torch.tensor([0, 0, 0]))
        self.assertEqual(len(paged_cache.empty_pages), n_pages)

    @supported_platform
    def test_convert_logical_block_mask(self):
        n_pages, page_size, max_batch_size, max_seq_len = 8, 128, 2, 512
        paged_cache = PagedAttention(n_pages, page_size, max_batch_size)

        batch_reserve(paged_cache, torch.tensor([100, 200], device="cuda"))
        batch_reserve(paged_cache, torch.tensor([150, 300], device="cuda"))
        batch_reserve(paged_cache, torch.tensor([300, 512], device="cuda"))
        batch_reserve(paged_cache, torch.tensor([512, 512], device="cuda"))

        expected_page_table = torch.tensor(
            [[0, 3, 5, 7, -1, -1, -1, -1], [2, 1, 4, 6, -1, -1, -1, -1]],
            device="cuda",
        )
        self.assertEqual(
            paged_cache.capacity,
            torch.tensor([512, 512], device="cuda"),
        )
        self.assertEqual(paged_cache.page_table, expected_page_table)

        # Get a block mask
        def causal_mask(b, h, q, kv):
            return q >= kv

        block_mask = create_block_mask(
            causal_mask, max_batch_size, 1, max_seq_len, max_seq_len
        )
        new_block_mask = paged_cache.convert_logical_block_mask(block_mask)

        zeros = [0, 0, 0, 0]
        # Check that the new block mask is correct
        expected_kv_num_blocks = torch.tensor(
            [[[1, 1, 1, 1]], [[1, 1, 1, 1]]], device="cuda", dtype=torch.int32
        )
        expected_kv_indices = torch.tensor(
            [
                [
                    [
                        [0, 3, 5, 7, *zeros],
                        [3, 0, 5, 7, *zeros],
                        [5, 0, 3, 7, *zeros],
                        [7, 0, 3, 5, *zeros],
                    ]
                ],
                [
                    [
                        [2, 1, 4, 6, *zeros],
                        [1, 2, 4, 6, *zeros],
                        [4, 2, 1, 6, *zeros],
                        [6, 2, 1, 4, *zeros],
                    ]
                ],
            ],
            device="cuda",
            dtype=torch.int32,
        )
        expected_full_kv_num_blocks = torch.tensor(
            [[[0, 1, 2, 3]], [[0, 1, 2, 3]]], device="cuda:0", dtype=torch.int32
        )
        expected_full_kv_indices = torch.tensor(
            [
                [
                    [
                        [0, 3, 5, 7, *zeros],
                        [0, 3, 5, 7, *zeros],
                        [0, 3, 5, 7, *zeros],
                        [0, 3, 5, 7, *zeros],
                    ]
                ],
                [
                    [
                        [2, 1, 4, 6, *zeros],
                        [2, 1, 4, 6, *zeros],
                        [2, 1, 4, 6, *zeros],
                        [2, 1, 4, 6, *zeros],
                    ]
                ],
            ],
            device="cuda",
            dtype=torch.int32,
        )
        self.assertEqual(new_block_mask.kv_num_blocks, expected_kv_num_blocks)
        self.assertEqual(new_block_mask.kv_indices, expected_kv_indices)
        self.assertEqual(new_block_mask.full_kv_num_blocks, expected_full_kv_num_blocks)
        self.assertEqual(new_block_mask.full_kv_indices, expected_full_kv_indices)

    @supported_platform
    def test_convert_mask_mod(self):
        n_pages, page_size, max_batch_size = 8, 128, 2
        paged_cache = PagedAttention(n_pages, page_size, max_batch_size)

        batch_reserve(paged_cache, torch.tensor([100, 200], device="cuda"))
        batch_reserve(paged_cache, torch.tensor([150, 300], device="cuda"))
        batch_reserve(paged_cache, torch.tensor([300, 512], device="cuda"))
        batch_reserve(paged_cache, torch.tensor([512, 512], device="cuda"))

        expected_page_table = torch.tensor(
            [[0, 3, 5, 7, -1, -1, -1, -1], [2, 1, 4, 6, -1, -1, -1, -1]],
            device="cuda",
        )
        self.assertEqual(
            paged_cache.capacity,
            torch.tensor([512, 512], device="cuda"),
        )
        self.assertEqual(paged_cache.page_table, expected_page_table)

        expected_physical_to_logical = torch.tensor(
            [[0, -1, -1, 1, -1, 2, -1, 3], [-1, 1, 0, -1, 2, -1, 3, -1]],
            device="cuda",
        )
        self.assertEqual(paged_cache.physical_to_logical, expected_physical_to_logical)

        # Get a block mask
        def causal_mask(b, h, q, kv):
            return q >= kv

        converted_causal_mask = paged_cache.get_mask_mod(causal_mask)

        # Equivalent to: causal_mask(0, 0, 256, 128)
        self.assertEqual(converted_causal_mask(0, 0, 256, 384), True)
        # Equivalent to: causal_mask(0, 1, 256, 128)
        self.assertEqual(converted_causal_mask(0, 1, 256, 384), True)
        # Not found corresponding logical block
        self.assertEqual(converted_causal_mask(1, 0, 256, 384), False)
        # Equivalent to: causal_mask(1, 0, 64, 14)
        self.assertEqual(converted_causal_mask(1, 0, 64, 270), True)

    @supported_platform
    def test_update(self):
        dtype = torch.float32

        n_pages, page_size, max_batch_size, max_seq_len = 6, 2, 2, 6
        paged_cache = PagedAttention(n_pages, page_size, max_batch_size)

        n_heads, head_dim = 2, 3
        cache_shape = (1, n_heads, n_pages * page_size, head_dim)
        k_cache = torch.zeros(cache_shape, dtype=dtype, device="cuda")

        batch_reserve(paged_cache, torch.tensor([1, 3], device="cuda"))
        batch_reserve(paged_cache, torch.tensor([4, 5], device="cuda"))
        batch_reserve(paged_cache, torch.tensor([6, 6], device="cuda"))

        expected_page_table = torch.tensor(
            [[0, 3, 5, -1, -1, -1], [2, 1, 4, -1, -1, -1]],
            device="cuda",
        )
        self.assertEqual(paged_cache.page_table, expected_page_table)

        batch_idx = torch.arange(max_batch_size, device="cuda", dtype=torch.int32)
        input_pos = (
            torch.arange(max_seq_len, device="cuda", dtype=torch.int32)
            .unsqueeze(0)
            .expand(max_batch_size, max_seq_len)
        )
        k = torch.arange(
            max_batch_size * n_heads * max_seq_len * head_dim,
            device="cuda",
            dtype=dtype,
        ).view(max_batch_size, n_heads, max_seq_len, head_dim)

        v = k.detach().clone()
        v_cache = k_cache.detach().clone()

        paged_cache.assign(batch_idx, input_pos, k, v, k_cache, v_cache)

        expected_cache = torch.tensor(
            [
                [
                    # h = 0
                    [
                        # page = 0
                        [0.0, 1.0, 2.0],
                        [3.0, 4.0, 5.0],
                        # page = 1
                        [42.0, 43.0, 44.0],
                        [45.0, 46.0, 47.0],
                        # page = 2
                        [36.0, 37.0, 38.0],
                        [39.0, 40.0, 41.0],
                        # page = 3
                        [6.0, 7.0, 8.0],
                        [9.0, 10.0, 11.0],
                        # page = 4
                        [48.0, 49.0, 50.0],
                        [51.0, 52.0, 53.0],
                        # page = 5
                        [12.0, 13.0, 14.0],
                        [15.0, 16.0, 17.0],
                    ],
                    # h = 1
                    [
                        # page = 0
                        [18.0, 19.0, 20.0],
                        [21.0, 22.0, 23.0],
                        # page = 1
                        [60.0, 61.0, 62.0],
                        [63.0, 64.0, 65.0],
                        # page = 2
                        [54.0, 55.0, 56.0],
                        [57.0, 58.0, 59.0],
                        # page = 3
                        [24.0, 25.0, 26.0],
                        [27.0, 28.0, 29.0],
                        # page = 4
                        [66.0, 67.0, 68.0],
                        [69.0, 70.0, 71.0],
                        # page = 5
                        [30.0, 31.0, 32.0],
                        [33.0, 34.0, 35.0],
                    ],
                ]
            ],
            device="cuda",
            dtype=dtype,
        )
        self.assertEqual(k_cache, expected_cache)

    @common_utils.parametrize("dtype", test_dtypes_fast)
    @common_utils.parametrize("score_mod", test_score_mods)
    def test_paged_builtin_score_mods(self, dtype: torch.dtype, score_mod: Callable):
        n_pages, page_size, max_batch_size, max_seq_len = 32, 128, 4, 512
        n_heads, head_dim = 4, 16

        def causal_mask(b, h, q, kv):
            return q >= kv

        block_mask = create_block_mask(
            causal_mask, max_batch_size, 1, max_seq_len, max_seq_len, device=self.device
        )
        q = torch.randn(
            max_batch_size,
            n_heads,
            max_seq_len,
            head_dim,
            device=self.device,
            dtype=dtype,
            requires_grad=False,
        )
        k = torch.randn(
            max_batch_size,
            n_heads,
            max_seq_len,
            head_dim,
            device=self.device,
            dtype=dtype,
            requires_grad=False,
        )
        v = torch.randn(
            max_batch_size,
            n_heads,
            max_seq_len,
            head_dim,
            device=self.device,
            dtype=dtype,
            requires_grad=False,
        )

        q_ref, k_ref, v_ref = query_key_value_clones(q, k, v)
        q_gold, k_gold, v_gold = query_key_value_clones(q, k, v, torch.float64)

        sdpa_partial = create_attention(score_mod, block_mask, enable_gqa=False)

        golden_out = sdpa_partial(q_gold, k_gold, v_gold)
        ref_out = sdpa_partial(q_ref, k_ref, v_ref)

        MAX_CACHED_SEQ_LEN = n_pages * page_size
        k_cache = torch.zeros(
            1,
            n_heads,
            MAX_CACHED_SEQ_LEN,
            head_dim,
            device=self.device,
            dtype=dtype,
        )
        v_cache = torch.zeros(
            1,
            n_heads,
            MAX_CACHED_SEQ_LEN,
            head_dim,
            device=self.device,
            dtype=dtype,
        )

        paged_cache = PagedAttention(
            n_pages, page_size, max_batch_size, device=self.device
        )
        batch_reserve(
            paged_cache, torch.tensor([100, 200, 50, 300], device=self.device)
        )
        batch_reserve(
            paged_cache, torch.tensor([100, 512, 300, 300], device=self.device)
        )
        batch_reserve(
            paged_cache, torch.tensor([512, 512, 300, 300], device=self.device)
        )
        batch_reserve(
            paged_cache, torch.tensor([512, 512, 512, 300], device=self.device)
        )
        batch_reserve(
            paged_cache, torch.tensor([512, 512, 512, 512], device=self.device)
        )

        batch_idx = torch.arange(max_batch_size, device=self.device, dtype=torch.int32)
        input_pos = (
            torch.arange(max_seq_len, device=self.device, dtype=torch.int32)
            .unsqueeze(0)
            .expand(max_batch_size, max_seq_len)
        )
        paged_cache.assign(batch_idx, input_pos, k, v, k_cache, v_cache)

        new_block_mask = paged_cache.convert_logical_block_mask(block_mask)

        compiled_sdpa = torch.compile(
            create_attention(
                paged_cache.get_score_mod(score_mod), block_mask, enable_gqa=False
            )
        )
        paged_out = compiled_sdpa(q, k_cache, v_cache, block_mask=new_block_mask)

        with torch.no_grad():
            dtype = ref_out.dtype
            if dtype == torch.float32:
                fudge_factor = 10.0
            else:
                fudge_factor = 1.1

            # Checkout output
            self._check_equal(golden_out, ref_out, paged_out, fudge_factor, "Out")


@dataclass
class Params:
    batch_size: int
    num_heads: int
    seq_length: int
    head_dim: int
    dtype: torch.dtype
    config_str: Optional[str] = None

    def __str__(self):
        return f"batch:{self.batch_size}_head:{self.num_heads}_seq_len:{self.seq_length}_headdim:{self.head_dim}_dtype:{str(self.dtype).split('.')[-1]}"


def get_params(dtypes: list[torch.dtype]) -> list[Params]:
    params = []
    seq_lengths = [37, 256, 277]
    for seq_len, dtype in product(seq_lengths, dtypes):
        params.append(
            Params(
                batch_size=2, num_heads=4, seq_length=seq_len, head_dim=16, dtype=dtype
            )
        )
    return params


# ROCM BUG SEE: https://github.com/pytorch/pytorch/issues/140855
supports_learnable_bias = unittest.skipUnless(
    torch.cuda.is_available()
    and torch.utils._triton.has_triton()
    and torch.cuda.get_device_capability() >= (8, 0)
    and not TEST_WITH_ROCM,
    "Requires CUDA and Triton, and is not supported on ROCm",
)


@supports_learnable_bias
class TestLearnableBiases(InductorTestCase):
    def setUp(self):
        super().setUp()
        self.device = "cuda"
        self.dtype = torch.float32
        self.atol = 3e-2
        self.rtol = 3e-2

    def _init_tensors(self, params: Params):
        make_tensor = functools.partial(
            torch.randn,
            (params.batch_size, params.num_heads, params.seq_length, params.head_dim),
            device=self.device,
            dtype=params.dtype,
            requires_grad=True,
        )
        return (make_tensor(), make_tensor(), make_tensor())

    @torch.no_grad()
    def _gold_check(self, eager, compiled, gold, tensor_name, fudge_factor=1.35):
        ref_error = rmse(eager, gold)
        comp_error = rmse(compiled, gold)
        # Note: This has been carefully tested that FlexAttention is within
        # 20% of the average error of SDPA! Do not bump this tolerance
        # unless you are absolutely sure you are not worsening the accuracy
        # of FlexAttention!
        if eager.dtype == torch.float32:
            fudge_factor = 10.0 * fudge_factor

        comp_error = comp_error.item()
        ref_error = ref_error.item() * fudge_factor

        if (
            tensor_name == "out"
            and eager.dtype == torch.float32
            and comp_error > ref_error
        ):
            self.skipTest("Compiled FlexAttention is less accurate than eager in fp32")

        self.assertLessEqual(
            comp_error,
            (ref_error * fudge_factor),
            f"\nTensor: {tensor_name}\nCompiled error ({comp_error:.8f}) exceeds "
            f"reference error ({ref_error:.8f}) * fudge_factor ({fudge_factor})",
        )

    def _check_outputs_and_grads(
        self, out_eager, out_compiled, out_gold, tensors, names=None
    ):
        backwards_grad = torch.randn_like(out_eager)
        grads_eager = torch.autograd.grad((out_eager,), tensors, backwards_grad)
        grads_compiled = torch.autograd.grad((out_compiled,), tensors, backwards_grad)
        grads_gold = torch.autograd.grad((out_gold,), tensors, backwards_grad)

        tensor_names = (
            ["out", "grad_query", "grad_key", "grad_value", "grad_bias"]
            if names is None
            else names
        )

        eager_tensors = (out_eager, *grads_eager)
        compiled_tensors = (out_compiled, *grads_compiled)
        gold_tensors = (out_gold, *grads_gold)

        for eager, compiled, gold, name in zip(
            eager_tensors, compiled_tensors, gold_tensors, tensor_names, strict=True
        ):
            self._gold_check(eager, compiled, gold, name)

    @common_utils.parametrize(
        "params", get_params(test_dtypes), name_fn=lambda x: f"{x}"
    )
    @common_utils.parametrize("mode", ["default", "max-autotune-no-cudagraphs"])
    def test_relative_1d_bias(self, params, mode: str):
        query, key, value = self._init_tensors(params)
        bias = torch.randn(
            2 * params.seq_length,
            device=self.device,
            dtype=params.dtype,
            requires_grad=True,
        )

        def bias_func(score, b, h, q_idx, kv_idx):
            return score + bias[torch.abs(q_idx - kv_idx)]

        flex_compiled = torch.compile(flex_attention, mode=mode)
        out_eager = flex_attention(query, key, value, score_mod=bias_func)
        out_compiled = flex_compiled(query, key, value, score_mod=bias_func)
        out_gold = flex_attention(
            query.to(torch.float64),
            key.to(torch.float64),
            value.to(torch.float64),
            score_mod=bias_func,
        )

        self._check_outputs_and_grads(
            out_eager,
            out_compiled,
            out_gold,
            (query, key, value, bias),
        )

    @common_utils.parametrize(
        "params", get_params(test_dtypes), name_fn=lambda x: f"{x}"
    )
    def test_absolute_2d_bias(self, params):
        query, key, value = self._init_tensors(params)
        bias = torch.randn(
            params.seq_length,
            params.seq_length,
            device=self.device,
            dtype=params.dtype,
            requires_grad=True,
        )

        def bias_func(score, b, h, q_idx, kv_idx):
            return score + bias[q_idx, kv_idx]

        flex_compiled = torch.compile(flex_attention)
        out_eager = flex_attention(query, key, value, score_mod=bias_func)
        out_compiled = flex_compiled(query, key, value, score_mod=bias_func)
        out_gold = flex_attention(
            query.to(torch.float64),
            key.to(torch.float64),
            value.to(torch.float64),
            score_mod=bias_func,
        )

        self._check_outputs_and_grads(
            out_eager,
            out_compiled,
            out_gold,
            (query, key, value, bias),
        )

    @common_utils.parametrize(
        "params", get_params(test_dtypes), name_fn=lambda x: f"{x}"
    )
    def test_head_specific_bias(self, params):
        query, key, value = self._init_tensors(params)
        bias = torch.randn(
            params.num_heads,
            params.seq_length,
            params.seq_length,
            device=self.device,
            dtype=params.dtype,
            requires_grad=True,
        )

        def bias_func(score, b, h, q_idx, kv_idx):
            return score + bias[h, q_idx, kv_idx]

        flex_compiled = torch.compile(flex_attention)
        out_eager = flex_attention(query, key, value, score_mod=bias_func)
        out_compiled = flex_compiled(query, key, value, score_mod=bias_func)
        out_gold = flex_attention(
            query.to(torch.float64),
            key.to(torch.float64),
            value.to(torch.float64),
            score_mod=bias_func,
        )

        self._check_outputs_and_grads(
            out_eager,
            out_compiled,
            out_gold,
            (query, key, value, bias),
        )

    @common_utils.parametrize(
        "params", get_params(test_dtypes), name_fn=lambda x: f"{x}"
    )
    def test_batch_head_bias(self, params):
        query, key, value = self._init_tensors(params)
        bias = torch.randn(
            params.batch_size,
            params.num_heads,
            params.seq_length,
            params.seq_length,
            device=self.device,
            dtype=params.dtype,
            requires_grad=True,
        )

        def bias_func(score, b, h, q_idx, kv_idx):
            return score + bias[b, h, q_idx, kv_idx]

        flex_compiled = torch.compile(flex_attention)
        out_eager = flex_attention(query, key, value, score_mod=bias_func)
        out_compiled = flex_compiled(query, key, value, score_mod=bias_func)
        out_gold = flex_attention(
            query.to(torch.float64),
            key.to(torch.float64),
            value.to(torch.float64),
            score_mod=bias_func,
        )

        self._check_outputs_and_grads(
            out_eager,
            out_compiled,
            out_gold,
            (query, key, value, bias),
        )

    @common_utils.parametrize(
        "params", get_params(test_dtypes), name_fn=lambda x: f"{x}"
    )
    def test_multiplicative_bias(self, params):
        query, key, value = self._init_tensors(params)
        bias = torch.randn(
            params.seq_length,
            device=self.device,
            dtype=params.dtype,
            requires_grad=True,
        )

        def bias_func(score, b, h, q_idx, kv_idx):
            return score * bias[q_idx]

        flex_compiled = torch.compile(flex_attention)
        out_eager = flex_attention(query, key, value, score_mod=bias_func)
        out_compiled = flex_compiled(query, key, value, score_mod=bias_func)
        out_gold = flex_attention(
            query.to(torch.float64),
            key.to(torch.float64),
            value.to(torch.float64),
            score_mod=bias_func,
        )

        self._check_outputs_and_grads(
            out_eager,
            out_compiled,
            out_gold,
            (query, key, value, bias),
        )

    @common_utils.parametrize(
        "params", get_params(test_dtypes), name_fn=lambda x: f"{x}"
    )
    def test_local_window_bias(self, params):
        query, key, value = self._init_tensors(params)
        window_size = 8
        bias = torch.randn(
            2 * window_size + 1,
            device=self.device,
            dtype=params.dtype,
            requires_grad=True,
        )

        def bias_func(score, b, h, q_idx, kv_idx):
            window_idx = torch.clamp(q_idx - kv_idx + window_size, 0, 2 * window_size)
            return score + bias[window_idx]

        flex_compiled = torch.compile(flex_attention)
        out_eager = flex_attention(query, key, value, score_mod=bias_func)
        out_compiled = flex_compiled(query, key, value, score_mod=bias_func)
        out_gold = flex_attention(
            query.to(torch.float64),
            key.to(torch.float64),
            value.to(torch.float64),
            score_mod=bias_func,
        )

        self._check_outputs_and_grads(
            out_eager,
            out_compiled,
            out_gold,
            (query, key, value, bias),
        )

    @common_utils.parametrize(
        "params", get_params(test_dtypes), name_fn=lambda x: f"{x}"
    )
    def test_global_tokens_bias(self, params):
        query, key, value = self._init_tensors(params)
        bias = torch.randn(
            params.seq_length,
            device=self.device,
            dtype=params.dtype,
            requires_grad=True,
        )

        def bias_func(score, b, h, q_idx, kv_idx):
            return score + bias[kv_idx]

        flex_compiled = torch.compile(flex_attention)
        out_eager = flex_attention(query, key, value, score_mod=bias_func)
        out_compiled = flex_compiled(query, key, value, score_mod=bias_func)
        out_gold = flex_attention(
            query.to(torch.float64),
            key.to(torch.float64),
            value.to(torch.float64),
            score_mod=bias_func,
        )

        self._check_outputs_and_grads(
            out_eager,
            out_compiled,
            out_gold,
            (query, key, value, bias),
        )

    @common_utils.parametrize(
        "params", get_params(test_dtypes), name_fn=lambda x: f"{x}"
    )
    def test_weird_bias(self, params):
        query, key, value = self._init_tensors(params)
        bias = torch.randn(
            params.batch_size,
            params.num_heads,
            4,
            params.seq_length,
            device=self.device,
            dtype=params.dtype,
            requires_grad=True,
        )
        which_bias = torch.tensor(0, device=self.device)

        def bias_func(score, b, h, q_idx, kv_idx):
            return score + bias[b, h, which_bias, q_idx]

        flex_compiled = torch.compile(flex_attention)
        out_eager = flex_attention(query, key, value, score_mod=bias_func)
        out_compiled = flex_compiled(query, key, value, score_mod=bias_func)
        out_gold = flex_attention(
            query.to(torch.float64),
            key.to(torch.float64),
            value.to(torch.float64),
            score_mod=bias_func,
        )

        self._check_outputs_and_grads(
            out_eager,
            out_compiled,
            out_gold,
            (query, key, value, bias),
        )

    @common_utils.parametrize(
        "params", get_params(test_dtypes), name_fn=lambda x: f"{x}"
    )
    def test_indirect_bias(self, params):
        query, key, value = self._init_tensors(params)
        bias = torch.randn(
            params.seq_length,
            device=self.device,
            dtype=params.dtype,
            requires_grad=True,
        )

        offset = torch.randint(
            0,
            params.seq_length,
            (params.seq_length,),
            device=self.device,
        )

        def bias_func(score, b, h, q_idx, kv_idx):
            return score + bias[offset[q_idx]]

        flex_compiled = torch.compile(flex_attention)
        out_eager = flex_attention(query, key, value, score_mod=bias_func)
        out_compiled = flex_compiled(query, key, value, score_mod=bias_func)
        out_gold = flex_attention(
            query.to(torch.float64),
            key.to(torch.float64),
            value.to(torch.float64),
            score_mod=bias_func,
        )

        self._check_outputs_and_grads(
            out_eager,
            out_compiled,
            out_gold,
            (query, key, value, bias),
        )

    @common_utils.parametrize(
        "params", get_params([torch.float32]), name_fn=lambda x: f"{x}"
    )
    @common_utils.parametrize("mode", ["default", "max-autotune-no-cudagraphs"])
    def test_symmetric_bias(self, params, mode: str):
        query, key, value = self._init_tensors(params)
        bias = torch.randn(
            params.seq_length,
            device=self.device,
            dtype=params.dtype,
            requires_grad=True,
        )

        def bias_func(score, b, h, q_idx, kv_idx):
            return score + bias[q_idx] + bias[kv_idx]

        flex_compiled = torch.compile(flex_attention, mode=mode)
        out_eager = flex_attention(query, key, value, score_mod=bias_func)
        out_compiled = flex_compiled(query, key, value, score_mod=bias_func)
        out_gold = flex_attention(
            query.to(torch.float64),
            key.to(torch.float64),
            value.to(torch.float64),
            score_mod=bias_func,
        )
        # Error in backwards
        with self.assertRaisesRegex(
            torch._inductor.exc.LoweringException,
            "Using multiple indexing operations on the same tensor that requires gradients",
        ):
            self._check_outputs_and_grads(
                out_eager,
                out_compiled,
                out_gold,
                (query, key, value, bias),
            )

    @common_utils.parametrize(
        "params", get_params(test_dtypes), name_fn=lambda x: f"{x}"
    )
    def test_flipped_indexed_bias(self, params):
        query, key, value = self._init_tensors(params)
        bias = torch.randn(
            params.seq_length,
            params.seq_length,
            device=self.device,
            dtype=params.dtype,
            requires_grad=True,
        )

        def bias_func(score, b, h, q_idx, kv_idx):
            return score + bias[kv_idx, q_idx]

        flex_compiled = torch.compile(flex_attention)
        out_eager = flex_attention(query, key, value, score_mod=bias_func)
        out_compiled = flex_compiled(query, key, value, score_mod=bias_func)
        out_gold = flex_attention(
            query.to(torch.float64),
            key.to(torch.float64),
            value.to(torch.float64),
            score_mod=bias_func,
        )

        self._check_outputs_and_grads(
            out_eager,
            out_compiled,
            out_gold,
            (query, key, value, bias),
        )

    @common_utils.parametrize(
        "params", get_params(test_dtypes), name_fn=lambda x: f"{x}"
    )
    @common_utils.parametrize("mode", ["default", "max-autotune-no-cudagraphs"])
    def test_head_specific_gate(self, params, mode: str):
        query, key, value = self._init_tensors(params)
        gate_score = torch.randn(
            params.num_heads,
            device=self.device,
            dtype=params.dtype,
            requires_grad=True,
        )

        def bias_func(score, b, h, q_idx, kv_idx):
            return score * torch.sigmoid(gate_score[h].to(torch.float32))

        flex_compiled = torch.compile(flex_attention, mode=mode)
        out_eager = flex_attention(query, key, value, score_mod=bias_func)
        out_compiled = flex_compiled(query, key, value, score_mod=bias_func)
        out_gold = flex_attention(
            query.to(torch.float64),
            key.to(torch.float64),
            value.to(torch.float64),
            score_mod=bias_func,
        )

        self._check_outputs_and_grads(
            out_eager,
            out_compiled,
            out_gold,
            (query, key, value, gate_score),
        )

    @common_utils.parametrize(
        "params", get_params(test_dtypes), name_fn=lambda x: f"{x}"
    )
    def test_distinct_biases(self, params):
        query, key, value = self._init_tensors(params)
        # Create two separate bias tensors
        bias1 = torch.randn(
            params.seq_length,
            device=self.device,
            dtype=params.dtype,
            requires_grad=True,
        )
        bias2 = torch.randn(
            params.seq_length,
            device=self.device,
            dtype=params.dtype,
            requires_grad=True,
        )

        def bias_func(score, b, h, q_idx, kv_idx):
            return score + bias1[q_idx] + bias2[kv_idx]

        flex_compiled = torch.compile(flex_attention)
        out_eager = flex_attention(query, key, value, score_mod=bias_func)
        out_compiled = flex_compiled(query, key, value, score_mod=bias_func)
        out_gold = flex_attention(
            query.to(torch.float64),
            key.to(torch.float64),
            value.to(torch.float64),
            score_mod=bias_func,
        )

        # Include both bias tensors in the tuple for gradient checking
        self._check_outputs_and_grads(
            out_eager,
            out_compiled,
            out_gold,
            (query, key, value, bias1, bias2),
            names=[
                "out",
                "grad_query",
                "grad_key",
                "grad_value",
                "grad_bias1",
                "grad_bias2",
            ],
        )

    @common_utils.parametrize(
        "params", get_params(test_dtypes), name_fn=lambda x: f"{x}"
    )
    def test_relative_1d_bias_only_grad(self, params):
        query, key, value = self._init_tensors(params)
        query = query.detach().requires_grad_(False)
        key = key.detach().requires_grad_(False)
        value = value.detach().requires_grad_(False)

        # Only bias requires gradients
        bias = torch.randn(
            2 * params.seq_length,
            device=self.device,
            dtype=params.dtype,
            requires_grad=True,  # Only bias needs gradients
        )

        def bias_func(score, b, h, q_idx, kv_idx):
            return score + bias[torch.abs(q_idx - kv_idx)]

        flex_compiled = torch.compile(flex_attention)
        out_eager = flex_attention(query, key, value, score_mod=bias_func)
        out_compiled = flex_compiled(query, key, value, score_mod=bias_func)

        out_gold = flex_attention(
            query.to(torch.float64),
            key.to(torch.float64),
            value.to(torch.float64),
            score_mod=bias_func,
        )

        # For gradient checking, we only pass the bias tensor since it's the only one requiring gradients
        self._check_outputs_and_grads(
            out_eager, out_compiled, out_gold, (bias,), names=["out", "bias"]
        )


common_utils.instantiate_parametrized_tests(TestFlexAttention)
common_utils.instantiate_parametrized_tests(TestBlockMask)
common_utils.instantiate_parametrized_tests(TestPagedAttention)
common_utils.instantiate_parametrized_tests(TestLearnableBiases)

if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
