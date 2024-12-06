# Owner(s): ["module: inductor"]
# flake8: noqa: B950

import functools
from collections import namedtuple
from typing import Callable, Optional, Tuple, Union
from unittest import expectedFailure, skipUnless
from unittest.mock import patch

import torch
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.utils import run_and_get_code
from torch.nn.attention.experimental._paged_attention import PagedAttention
from torch.nn.attention.flex_attention import (
    _create_empty_block_mask,
    _identity,
    BlockMask,
    create_block_mask,
    flex_attention,
    noop_mask,
)
from torch.testing import FileCheck
from torch.testing._internal import common_utils
from torch.testing._internal.common_cuda import PLATFORM_SUPPORTS_BF16
from torch.testing._internal.common_utils import skipIfRocm, TEST_WITH_ROCM
from torch.utils._triton import has_triton


# Skip tests if Triton is not available
supported_platform = skipUnless(
    torch.cuda.is_available()
    and has_triton()
    and torch.cuda.get_device_capability() >= (8, 0),
    "Requires CUDA and Triton",
)

Tolerances = namedtuple("Tolerances", ["atol", "rtol"])
torch.set_float32_matmul_precision("high")

index = torch.ops.aten.index
Tensor = torch.Tensor


def create_attention(score_mod, block_mask, enable_gqa=False):
    return functools.partial(
        flex_attention,
        score_mod=score_mod,
        block_mask=block_mask,
        enable_gqa=enable_gqa,
    )


def create_block_mask_test(score_mod, query, key):
    block_mask = create_block_mask(
        score_mod, 1, 1, query.shape[-2], key.shape[-2], query.device
    )
    return block_mask


test_dtypes = (
    [torch.float16, torch.bfloat16, torch.float32]
    if PLATFORM_SUPPORTS_BF16
    else [torch.float16, torch.float32]
)

test_dtypes_fast = [torch.float16]

test_page_sizes = [64, 128, 256]


# --------- Useful score mod functions for testing ---------
def _causal(
    score: Tensor,
    batch: Tensor,
    head: Tensor,
    token_q: Tensor,
    token_kv: Tensor,
) -> Tensor:
    return torch.where(token_q >= token_kv, score, float("-inf"))


def _generate_windowed(offset):
    def _windowed(score, b, h, q, kv):
        return torch.where(q + offset >= kv, score, float("-inf"))

    return _windowed


def _get_windowed_sdpa_mask(Mq, Mkv, offset):
    return torch.tril(torch.ones(Mkv, Mkv, dtype=torch.bool, device="cuda"))[
        offset : offset + Mq
    ]


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
    head_offset = torch.rand(Hq, device="cuda", dtype=dtype)

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


test_score_mods = [
    _identity,
    _times_two,
    _squared,
    _causal,
    _inverse_causal,
    _rel_bias,
    _rel_causal,
    _generate_alibi_bias(8),
    _generate_windowed(1000),
]

captured_buffers_map = {
    "_head_offset": _head_offset,
}

B = 4
S = 2048
D = 64


test_Hq_Hkv = [
    (16, 1),
    (8, 2),
    (16, 16),
]

test_Bq_Bkv = [
    (3, 1),
    (5, 1),
    (8, 1),
    (16, 1),
]

test_block_size = [
    64,
    128,
    (1, 64),
    (128, 64),
]

(Hq, Hkv) = (16, 8)


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


class TestFlexDecoding(InductorTestCase):
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
        if torch.isnan(compiled_error).any() and not torch.isnan(ref_error).any():
            self.assertTrue(False, "Output/Grad with NaN")
        if ref_error < (1e-4) * golden_out.abs().mean():
            print(
                "very small ref error of ",
                (ref_error.to(torch.float64) * (1e5) / golden_out.abs().mean()),
            )
            tolerance = Tolerances(atol=2e-1, rtol=2e-1)
            torch.testing.assert_close(
                golden_out.to(dtype=compiled_out.dtype),
                compiled_out,
                atol=tolerance.atol,
                rtol=tolerance.rtol,
            )
        elif compiled_error > ref_error * fudge_factor:
            name = tensor_name if tensor_name is not None else ""
            msg = f"{name} Compiled error {compiled_error} is greater than ref error {ref_error} by more than {fudge_factor}X."
            self.assertTrue(False, msg)

    def _check_out(
        self,
        golden_out: torch.Tensor,
        ref_out: torch.Tensor,
        compiled_out: torch.Tensor,
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

    def run_test(
        self,
        score_mod: Optional[Callable],
        dtype: torch.dtype = torch.float16,
        Q_B: int = B,
        Q_H: int = Hq,
        Q_S: int = 1,
        Q_D: int = D,
        KV_B: int = B,
        KV_H: int = Hkv,
        KV_S: int = S,
        V_D: int = D,
        block_mask: Optional[BlockMask] = None,
    ):
        assert (
            score_mod is not None or block_mask is not None
        ), "Must provide score_mod or block_mask"
        assert Q_H % KV_H == 0
        q = torch.randn(
            (Q_B, Q_H, Q_S, Q_D),
            dtype=dtype,
            device="cuda",
            requires_grad=False,
        )
        k = torch.randn(
            (KV_B, KV_H, KV_S, Q_D), dtype=dtype, device="cuda", requires_grad=False
        )
        v = torch.randn(
            (KV_B, KV_H, KV_S, V_D), dtype=dtype, device="cuda", requires_grad=False
        )
        q_ref, k_ref, v_ref = query_key_value_clones(q, k, v)
        q_gold, k_gold, v_gold = query_key_value_clones(q, k, v, torch.float64)

        sdpa_partial = create_attention(
            score_mod, block_mask, enable_gqa=(not Q_H == KV_H)
        )
        compiled_sdpa = torch.compile(sdpa_partial)
        golden_out, gold_lse = sdpa_partial(q_gold, k_gold, v_gold, return_lse=True)
        ref_out, ref_lse = sdpa_partial(q_ref, k_ref, v_ref, return_lse=True)
        compiled_out, compiled_lse = compiled_sdpa(q, k, v, return_lse=True)

        self._check_out(
            golden_out,
            ref_out,
            compiled_out,
        )
        self._check_out(
            gold_lse,
            ref_lse,
            compiled_lse,
        )

    def run_test_with_call(
        self,
        sdpa_call: Callable,
        golden_call: Optional[Callable] = None,
        dtype: torch.dtype = torch.float16,
        Q_B: int = B,
        Q_H: int = Hq,
        Q_S: int = 1,
        Q_D: int = D,
        KV_B: int = B,
        KV_H: int = Hkv,
        KV_S: int = S,
        V_D: int = D,
    ):
        if not golden_call:
            golden_call = sdpa_call
        q = torch.randn(
            (Q_B, KV_H, Q_S * (Q_H // KV_H), Q_D),
            dtype=dtype,
            device="cuda",
            requires_grad=False,
        )
        k = torch.randn(
            (KV_B, KV_H, KV_S, Q_D), dtype=dtype, device="cuda", requires_grad=False
        )
        v = torch.randn(
            (KV_B, KV_H, KV_S, V_D), dtype=dtype, device="cuda", requires_grad=False
        )
        q_ref, k_ref, v_ref = query_key_value_clones(q, k, v)
        q_gold, k_gold, v_gold = query_key_value_clones(q, k, v, torch.float64)

        compiled_sdpa = torch.compile(sdpa_call)
        golden_out = golden_call(q_gold, k_gold, v_gold)
        ref_out = golden_call(q_ref, k_ref, v_ref)
        compiled_out = compiled_sdpa(q, k, v)

        self._check_out(
            golden_out,
            ref_out,
            compiled_out,
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
    ):
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
            device="cuda",
            dtype=dtype,
        )
        v_cache = torch.zeros(
            1,
            KV_H,
            MAX_CACHED_SEQ_LEN,
            V_D,
            device="cuda",
            dtype=dtype,
        )

        # "randomly" initialize the page table
        paged_attention = PagedAttention(n_pages, page_size, max_batch_size)
        batch_reserve(
            paged_attention,
            torch.tensor([KV_S // 4, KV_S // 2, KV_S // 4, KV_S // 3], device="cuda"),
        )
        batch_reserve(
            paged_attention,
            torch.tensor([KV_S // 4, KV_S // 2, KV_S // 2, KV_S // 2], device="cuda"),
        )
        batch_reserve(
            paged_attention,
            torch.tensor([KV_S // 2, KV_S, KV_S // 2, KV_S], device="cuda"),
        )
        batch_reserve(
            paged_attention, torch.tensor([KV_S, KV_S, KV_S, KV_S], device="cuda")
        )

        # update cache with k and v
        input_pos = torch.arange(KV_S, device="cuda", dtype=torch.int32)
        batch_idx = torch.arange(KV_B, device="cuda", dtype=torch.int32)
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
    ):
        Q_B, Q_H, KV_H = q.shape[0], q.shape[1], k.shape[1]

        if block_mask is None:
            block_mask = create_block_mask(noop_mask, Q_B, 1, 1, S)

        (
            k_cache,
            v_cache,
            converted_block_mask,
            converted_score_mod,
        ) = self.preprocess_paged_attention(
            score_mod, q, k, v, block_mask, dtype, block_mask.BLOCK_SIZE[1]
        )

        compiled_sdpa = torch.compile(flex_attention)

        # compute
        compiled_out, compiled_lse = compiled_sdpa(
            q,
            k_cache,
            v_cache,
            return_lse=True,
            block_mask=converted_block_mask,
            score_mod=converted_score_mod,
            enable_gqa=(not Q_H == KV_H),
        )
        return compiled_out, compiled_lse

    def run_test_with_paged_attention(
        self,
        score_mod: Optional[Callable],
        dtype: torch.dtype = torch.float16,
        Q_B: int = B,
        Q_H: int = Hq,
        Q_S: int = 1,
        QK_D: int = D,
        KV_B: int = B,
        KV_H: int = Hkv,
        KV_S: int = S,
        V_D: int = D,
        block_mask: Optional[BlockMask] = None,
    ):
        if TEST_WITH_ROCM and Q_H != KV_H:
            self.skipTest("enable_gqa=True is unsupported on ROCM, for now")

        assert Q_H % KV_H == 0

        q = torch.randn(
            (Q_B, Q_H, Q_S, QK_D),
            dtype=dtype,
            device="cuda",
            requires_grad=False,
        )
        k = torch.randn(
            (KV_B, KV_H, KV_S, QK_D),
            dtype=dtype,
            device="cuda",
            requires_grad=False,
        )
        v = torch.randn(
            (KV_B, KV_H, KV_S, V_D),
            dtype=dtype,
            device="cuda",
            requires_grad=False,
        )
        q_ref, k_ref, v_ref = query_key_value_clones(q, k, v)
        q_gold, k_gold, v_gold = query_key_value_clones(q, k, v, torch.float64)

        if block_mask is None:
            block_mask = create_block_mask(noop_mask, Q_B, 1, 1, KV_S)

        sdpa_partial = create_attention(
            score_mod, block_mask, enable_gqa=(not Q_H == KV_H)
        )
        golden_out, gold_lse = sdpa_partial(q_gold, k_gold, v_gold, return_lse=True)
        ref_out, ref_lse = sdpa_partial(q_ref, k_ref, v_ref, return_lse=True)

        compiled_out, compiled_lse = self.run_paged_attention(
            score_mod, q, k, v, dtype, block_mask
        )

        self._check_out(
            golden_out,
            ref_out,
            compiled_out,
        )
        self._check_out(
            gold_lse,
            ref_lse,
            compiled_lse,
        )

    def run_test_with_call_paged_attention(
        self,
        score_mod: Optional[Callable],
        mask_mod: Optional[Callable],
        sdpa_mask: Tensor,
        dtype: torch.dtype = torch.float16,
        Q_B: int = B,
        Q_H: int = Hq,
        Q_S: int = 1,
        Q_D: int = D,
        KV_B: int = B,
        KV_H: int = Hkv,
        KV_S: int = S,
        V_D: int = D,
    ):
        q = torch.randn(
            (Q_B, KV_H, Q_S * (Q_H // KV_H), Q_D),
            dtype=dtype,
            device="cuda",
            requires_grad=False,
        )
        k = torch.randn(
            (KV_B, KV_H, KV_S, Q_D), dtype=dtype, device="cuda", requires_grad=False
        )
        v = torch.randn(
            (KV_B, KV_H, KV_S, V_D), dtype=dtype, device="cuda", requires_grad=False
        )
        q_ref, k_ref, v_ref = query_key_value_clones(q, k, v)
        q_gold, k_gold, v_gold = query_key_value_clones(q, k, v, torch.float64)

        golden_call = functools.partial(
            torch.nn.functional.scaled_dot_product_attention, attn_mask=sdpa_mask
        )
        golden_out = golden_call(q_gold, k_gold, v_gold)
        ref_out = golden_call(q_ref, k_ref, v_ref)

        if mask_mod is not None:
            block_mask = create_block_mask(mask_mod, Q_B, 1, Q_S, KV_S)
        else:
            block_mask = create_block_mask(noop_mask, Q_B, 1, Q_S, KV_S)

        compiled_out, _ = self.run_paged_attention(
            score_mod, q, k, v, dtype, block_mask
        )

        self._check_out(
            golden_out,
            ref_out,
            compiled_out,
        )

    @supported_platform
    @expectedFailure
    @common_utils.parametrize("dtype", test_dtypes_fast)
    def test_bw_decoding_fails(self, dtype):
        make_kv = functools.partial(
            torch.randn,
            (2, 2, 128, 4),
            dtype=dtype,
            device="cuda",
            requires_grad=True,
        )
        make_q = functools.partial(
            torch.randn,
            (2, 2, 8, 4),
            dtype=dtype,
            device="cuda",
            requires_grad=True,
        )
        q, k, v, backward_grad = make_q(), make_kv(), make_kv(), make_q()

        block_mask = _create_empty_block_mask(q, k)

        @torch.compile
        def sdpa_hop(q, k, v, score_mod, block_mask):
            return flex_attention(q, k, v, score_mod)

        output = sdpa_hop(q, k, v, _identity, block_mask)

        output.backward(backward_grad)

    @supported_platform
    @common_utils.parametrize("dtype", test_dtypes)
    @common_utils.parametrize("score_mod", test_score_mods)
    @common_utils.parametrize("head_dims", test_Hq_Hkv)
    def test_builtin_score_mods(
        self, dtype: torch.dtype, score_mod: Callable, head_dims
    ):
        Hq, Hkv = head_dims
        assert Hq % Hkv == 0
        self.run_test(score_mod, dtype, Q_H=Hq, KV_H=Hkv)
        self.run_test_with_paged_attention(score_mod, dtype, Q_H=Hq, KV_H=Hkv)

    @supported_platform
    @common_utils.parametrize("dtype", test_dtypes_fast)
    @common_utils.parametrize("score_mod", test_score_mods)
    @common_utils.parametrize("head_dims", test_Hq_Hkv)
    @common_utils.parametrize("page_size", test_page_sizes)
    def test_paged_attention_page_size(
        self,
        dtype: torch.dtype,
        score_mod: Callable,
        head_dims: Tuple[int, int],
        page_size: int,
    ):
        Hq, Hkv = head_dims
        assert Hq % Hkv == 0

        def generate_causal_offset(offset: torch.Tensor):
            def causal_offset_mask(b, h, q_idx, kv_idx):
                return (offset + q_idx) >= kv_idx

            return causal_offset_mask

        mod = generate_causal_offset(
            torch.tensor(192, device="cuda", dtype=torch.int32)
        )
        block_mask = create_block_mask(mod, B, 1, 1, S, BLOCK_SIZE=page_size)

        self.run_test_with_paged_attention(
            score_mod,
            dtype,
            Q_B=B,
            Q_H=Hq,
            KV_B=B,
            KV_H=Hkv,
            KV_S=S,
            block_mask=block_mask,
        )

    @supported_platform
    @common_utils.parametrize("dtype", test_dtypes)
    @common_utils.parametrize("score_mod", test_score_mods)
    @common_utils.parametrize("BLOCK_SIZE", test_block_size)
    def test_builtin_score_mods_different_block_size(
        self,
        dtype: torch.dtype,
        score_mod: Callable,
        BLOCK_SIZE: Union[int, Tuple[int, int]],
    ):
        block_mask = create_block_mask(noop_mask, B, 1, 1, S, BLOCK_SIZE=BLOCK_SIZE)
        self.run_test(score_mod, dtype, block_mask=block_mask)

    def input_strides_1(B, H, S, D):
        return ((H * S * D, S * D, D, 1), 997)  # offset

    def input_strides_2(B, H, S, D):
        return ((H * D, D, B * H * D, 1), 499)  # transposed dimensions

    def input_strides_3(B, H, S, D):
        return ((S * (D + 1), B * S * (D + 1), (D + 1), 1), 293)  # additional buffer

    def input_strides_4(B, H, S, D):
        return ((1, D, (B + 1) * (H + 1) * D, 1), 97)  # shared dimension

    test_input_strides = [
        input_strides_1,
        input_strides_2,
        input_strides_3,
        input_strides_4,
    ]

    @supported_platform
    @common_utils.parametrize("dtype", test_dtypes_fast)
    @common_utils.parametrize("k_s", test_input_strides)
    @common_utils.parametrize("v_s", test_input_strides)
    @common_utils.parametrize("head_dims", test_Hq_Hkv)
    def test_strided_inputs(self, dtype: torch.dtype, k_s, v_s, head_dims):
        Hq, Hkv = head_dims
        assert Hq % Hkv == 0
        q1 = torch.randn((B * Hq * D), dtype=dtype, device="cuda")
        k1 = torch.randn((B * Hkv * S * D * 4), dtype=dtype, device="cuda")
        v1 = torch.randn((B * Hkv * S * D * 4), dtype=dtype, device="cuda")

        k_shape = (B, Hkv, S, D)
        v_shape = (B, Hkv, S, D)

        q = q1.view(1, Hq, B, D).transpose(0, 2)

        k_strides, k_offset = k_s(B, Hkv, S, D)
        k_max = [x * (y - 1) for x, y in zip(k_strides, k_shape)]
        assert sum(k_max) + k_offset < B * Hkv * S * D * 4
        assert k_strides[-1] == 1
        k = torch.as_strided(k1, k_shape, k_strides, k_offset)

        v_strides, v_offset = v_s(B, Hkv, S, D)
        v_max = [x * (y - 1) for x, y in zip(v_strides, v_shape)]
        assert sum(v_max) + v_offset < B * Hkv * S * D * 4
        assert v_strides[-1] == 1
        v = torch.as_strided(v1, v_shape, v_strides, v_offset)

        score_mod = _generate_alibi_bias(8)

        sdpa_partial = create_attention(
            score_mod=score_mod,
            block_mask=None,
            enable_gqa=(not Hq == Hkv),
        )
        compiled_sdpa = torch.compile(sdpa_partial)
        ref_out = sdpa_partial(q, k, v)
        compiled_out = compiled_sdpa(q, k, v)

        tolerance = Tolerances(atol=2e-1, rtol=2e-1)
        torch.testing.assert_close(
            ref_out, compiled_out, atol=tolerance.atol, rtol=tolerance.rtol
        )

        paged_compiled_out, _ = self.run_paged_attention(score_mod, q, k, v, dtype)
        torch.testing.assert_close(
            ref_out, paged_compiled_out, atol=tolerance.atol, rtol=tolerance.rtol
        )

    @supported_platform
    @common_utils.parametrize("dtype", test_dtypes_fast)
    @common_utils.parametrize("head_dims", test_Hq_Hkv)
    @common_utils.parametrize("batch_dims", test_Bq_Bkv)
    @common_utils.parametrize("score_mod", test_score_mods)
    def test_kv_batch_broadcast(
        self,
        dtype: torch.dtype,
        head_dims: Tuple[int, int],
        batch_dims: Tuple[int, int],
        score_mod: Callable,
    ):
        Hq, Hkv = head_dims
        assert Hq % Hkv == 0

        Bq, Bkv = batch_dims
        assert Bq > 1 and Bkv == 1

        block_mask = create_block_mask(noop_mask, Bq, 1, 1, S)

        self.run_test(
            score_mod,
            dtype,
            Bq,
            Hq,
            1,
            D,
            Bkv,
            Hkv,
            S,
            D,
            block_mask,
        )

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
    def test_captured_buffers(self, dtype: torch.dtype):
        head_offset = torch.rand(Hq, device="cuda", dtype=dtype)

        def score_mod(score, b, h, m, n):
            return score + head_offset[h]

        self.run_test(score_mod, dtype)
        self.run_test_with_paged_attention(score_mod, dtype)

    @supported_platform
    @common_utils.parametrize("dtype", test_dtypes)
    def test_captured_buffers_all_dims(self, dtype: torch.dtype):
        head_scale = torch.randn(Hq, device="cuda")
        batch_scale = torch.randn(B, device="cuda")
        kv_scale = torch.randn(S, device="cuda")
        q_scale = torch.randn(1, device="cuda")

        def all_bias(score, batch, head, token_q, token_kv):
            score = score + kv_scale[token_kv]
            score = score + q_scale[token_q]
            score = score + head_scale[head]
            score = score + batch_scale[batch]
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
        bias = torch.randn(1, S, device="cuda", dtype=dtype)

        def bias_mod(score, b, h, q, kv):
            return score + bias[q, kv]

        self.run_test(bias_mod, dtype)
        self.run_test_with_paged_attention(bias_mod, dtype)

    @supported_platform
    @common_utils.parametrize("dtype", test_dtypes_fast)
    def test_load_from_bias_seq_batch(self, dtype):
        bias = torch.randn(B, 1, S, device="cuda", dtype=dtype)

        def bias_mod(score, b, h, q, kv):
            return score + bias[b, q, kv]

        self.run_test(bias_mod, dtype)
        self.run_test_with_paged_attention(bias_mod, dtype)

    @skipIfRocm
    @supported_platform
    @common_utils.parametrize("dtype", test_dtypes_fast)
    def test_load_from_bias_head_seq_batch(self, dtype):
        bias = torch.randn(
            B,
            Hq,
            1,
            S,
            device="cuda",
            dtype=dtype,
        )

        def bias_mod(score, b, h, q, kv):
            return score + bias[b, h, q, kv]

        self.run_test(bias_mod, dtype)
        self.run_test_with_paged_attention(bias_mod, dtype)

    @supported_platform
    @common_utils.parametrize("score_mod", test_score_mods)
    @common_utils.parametrize("dtype", test_dtypes)
    @common_utils.parametrize("head_dims", [(D, D // 2), (D // 2, D)])
    def test_non_equal_head_dims(self, dtype, score_mod, head_dims):
        qk_d, v_d = head_dims
        self.run_test(score_mod, dtype, B, Hq, 1, qk_d, B, Hkv, S, V_D=v_d)
        self.run_test_with_paged_attention(
            score_mod, dtype, B, Hq, 1, qk_d, B, Hkv, S, V_D=v_d
        )

    @supported_platform
    @common_utils.parametrize("dtype", test_dtypes_fast)
    def test_subgraph_respect_decompostion(self, dtype):
        from torch._decomp import core_aten_decompositions
        from torch.fx.experimental.proxy_tensor import make_fx

        def score_mod_func(score, b, h, q, kv):
            return score - q // (1 + kv)

        make_kv = functools.partial(
            torch.randn,
            (2, 2, 128, 4),
            dtype=dtype,
            device="cuda",
            requires_grad=True,
        )
        make_q = functools.partial(
            torch.randn,
            (2, 2, 8, 4),
            dtype=dtype,
            device="cuda",
            requires_grad=True,
        )
        query, key, value = make_q(), make_kv(), make_kv()
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
        self.run_test_with_paged_attention(causal_njt, dtype)

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
        query = torch.randn((1, 8, 4, 64), dtype=torch.float32, device="cuda")
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
    def test_multiple_score_mod_calls2(self):
        query = torch.randn((1, 8, 4, 64), dtype=torch.float32, device="cuda")
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
        query = torch.randn((1, 8, 4, 64), dtype=torch.float32, device="cuda")
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

        block_mask = create_block_mask(noop_mask, 1, 1, 4, 1024)

        def f(q, k1, k2, v1, v2):
            q2 = flex_attention(q, k1, v1, score_mod=scoremod_1, block_mask=block_mask)
            return flex_attention(
                q2, k2, v2, score_mod=scoremod_2, block_mask=block_mask
            )

        eager_out = f(query, *keys, *values)

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
    def test_multiple_score_mod_calls_paged_attention2(self):
        query = torch.randn((1, 8, 4, 64), dtype=torch.float32, device="cuda")
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

        block_mask = create_block_mask(noop_mask, 1, 1, 4, 1024)

        attention1 = functools.partial(
            flex_attention, score_mod=scoremod_1, block_mask=block_mask
        )

        def f(q, k1, k2, k3, v1, v2, v3):
            q2 = attention1(q, k1, v1)
            q3 = flex_attention(q2, k2, v2, score_mod=scoremod_2, block_mask=block_mask)
            return flex_attention(
                q3, k3, v3, score_mod=scoremod_1, block_mask=block_mask
            )

        eager_out = f(query, *keys, *values)

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
        query = torch.randn((1, 1, 8, 64), dtype=torch.float32, device="cuda")
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
    @patch.object(torch._inductor.config, "max_autotune", True)
    def test_max_autotune_with_captured(self):
        head_scale = torch.randn(Hq, device="cuda")
        batch_scale = torch.randn(B, device="cuda")
        tok_scale = torch.randn(S, device="cuda")
        q_scale = torch.randn(1, device="cuda")

        def bias_mod(score, batch, head, token_q, token_kv):
            score = score + tok_scale[token_kv]
            score = score + q_scale[token_q]
            score = score + batch_scale[batch]
            score = score + head_scale[head]
            return score

        self.run_test(bias_mod)
        self.run_test_with_paged_attention(bias_mod)

    @skipIfRocm
    @supported_platform
    def test_fully_masked_out_rows_0_check_gqa(self):
        # Ensure fully masked out rows won't cause NaNs.
        query = torch.randn(
            (B, Hq, S, D), dtype=torch.float32, device="cuda", requires_grad=True
        )
        key = torch.randn(
            (B, Hkv, S, D), dtype=torch.float32, device="cuda", requires_grad=True
        )
        value = torch.randn(
            (B, Hkv, S, D), dtype=torch.float32, device="cuda", requires_grad=True
        )

        M = S // 2

        def mask_mod(b, h, q, kv):
            return q < M

        block_mask = create_block_mask(mask_mod, 1, 1, S, S)

        flex = torch.compile(flex_attention, dynamic=False)

        out, lse = flex(
            query, key, value, block_mask=block_mask, enable_gqa=True, return_lse=True
        )
        self.assertEqual(out[:, :, M:, :].sum(), 0)
        self.assertTrue((lse[:, :, M:] == -float("inf")).all())

        loss = out.sum() + lse.sum()
        loss.backward()
        self.assertEqual(query.grad[:, :, M:, :].sum(), 0)

    @supported_platform
    def test_windowed_no_mask_vs_sdpa(self):
        score_mod = _generate_windowed(1000)
        attention = functools.partial(flex_attention, score_mod=score_mod)

        sdpa_mask = _get_windowed_sdpa_mask(8, S, 1000)

        sdpa_attention = functools.partial(
            torch.nn.functional.scaled_dot_product_attention, attn_mask=sdpa_mask
        )

        self.run_test_with_call(attention, sdpa_attention, Q_H=16, KV_H=16, Q_S=8)

    @supported_platform
    def test_windowed_full_mask_vs_sdpa(self):
        def mask_mod(b, h, q, kv):
            return q + 1000 >= kv

        score_mod = _generate_windowed(1000)

        block_mask = create_block_mask(mask_mod, 1, 1, 8, S)
        attention = functools.partial(
            flex_attention, block_mask=block_mask, score_mod=score_mod
        )

        sdpa_mask = _get_windowed_sdpa_mask(8, S, 1000)
        sdpa_attention = functools.partial(
            torch.nn.functional.scaled_dot_product_attention, attn_mask=sdpa_mask
        )

        self.run_test_with_call(attention, sdpa_attention, Q_H=16, KV_H=16, Q_S=8)

    @supported_platform
    def test_windowed_partial_block_vs_sdpa(self):
        def mask_mod(b, h, q, kv):
            return q + 1000 >= kv

        block_mask = create_block_mask(mask_mod, 1, 1, 8, S)
        attention = functools.partial(flex_attention, block_mask=block_mask)

        sdpa_mask = _get_windowed_sdpa_mask(8, S, 1000)
        sdpa_attention = functools.partial(
            torch.nn.functional.scaled_dot_product_attention, attn_mask=sdpa_mask
        )

        self.run_test_with_call(attention, sdpa_attention, Q_H=16, KV_H=16, Q_S=8)

    @supported_platform
    def test_windowed_no_mask_vs_sdpa_paged_attention(self):
        score_mod = _generate_windowed(1000)

        sdpa_mask = _get_windowed_sdpa_mask(8, S, 1000)

        self.run_test_with_call_paged_attention(
            score_mod, None, sdpa_mask, Q_H=16, KV_H=16, Q_S=8
        )

    @supported_platform
    def test_windowed_full_mask_vs_sdpa_paged_attention(self):
        def mask_mod(b, h, q, kv):
            return q + 1000 >= kv

        score_mod = _generate_windowed(1000)
        sdpa_mask = _get_windowed_sdpa_mask(8, S, 1000)
        self.run_test_with_call_paged_attention(
            score_mod, mask_mod, sdpa_mask, Q_H=16, KV_H=16, Q_S=8
        )

    @supported_platform
    def test_windowed_partial_block_vs_sdpa_paged_attention(self):
        def mask_mod(b, h, q, kv):
            return q + 1000 >= kv

        sdpa_mask = _get_windowed_sdpa_mask(8, S, 1000)

        self.run_test_with_call_paged_attention(
            None, mask_mod, sdpa_mask, Q_H=16, KV_H=16, Q_S=8
        )

    @supported_platform
    @common_utils.parametrize("dtype", test_dtypes)
    @common_utils.parametrize("score_mod", [_identity, _causal])
    def test_logsumexp_correctness(self, dtype, score_mod):
        make_kv = functools.partial(
            torch.randn,
            (B, Hkv, S, D),
            dtype=dtype,
            device="cuda",
            requires_grad=True,
        )
        make_q = functools.partial(
            torch.randn,
            (B, Hkv, Hq // Hkv, D),
            dtype=dtype,
            device="cuda",
            requires_grad=True,
        )
        q, k, v = make_q(), make_kv(), make_kv()

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
        make_q = functools.partial(
            torch.randn,
            (B, Hkv, Hq // Hkv, D),
            dtype=torch.float32,
            device="cuda",
            requires_grad=True,
        )
        make_kv = functools.partial(
            torch.randn,
            (B, Hkv, S, D),
            dtype=torch.float32,
            device="cuda",
            requires_grad=True,
        )

        q, k, v = make_q(), make_kv(), make_kv()

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
    def test_non_sparse_mulitple_block_size(self):
        def generate_causal_offset(offset: torch.Tensor):
            def causal_offset_mask(b, h, q_idx, kv_idx):
                return (offset + q_idx) >= kv_idx

            return causal_offset_mask

        def noop(score, b, h, q_idx, kv_idx):
            return score

        mod = generate_causal_offset(
            torch.tensor(192, device="cuda", dtype=torch.int32)
        )
        block_mask = create_block_mask(mod, 1, 1, 1, 65)

        self.run_test(
            score_mod=None,
            dtype=torch.float32,
            block_mask=block_mask,
            Q_B=1,
            Q_H=1,
            Q_S=1,
            Q_D=16,
            KV_B=1,
            KV_H=1,
            KV_S=65,
            V_D=16,
        )
        self.run_test_with_paged_attention(
            score_mod=None,
            dtype=torch.float32,
            block_mask=block_mask,
            Q_B=1,
            Q_H=1,
            Q_S=1,
            QK_D=16,
            KV_B=1,
            KV_H=1,
            KV_S=65,
            V_D=16,
        )

    @supported_platform
    def test_do_not_trigger_dynamic_shapes_on_empty_block_mask(self):
        torch._dynamo.reset()
        H = Hq
        q = torch.randn(B, H, 1, D, device="cuda")
        for i in range(5):
            k = torch.randn(B, H, S + i, D, device="cuda")
            v = torch.randn(B, H, S + i, D, device="cuda")
            compiled_flex_attention = torch.compile(flex_attention)
            ref = flex_attention(q, k, v)
            res = compiled_flex_attention(q, k, v)
            tolerance = Tolerances(atol=2e-1, rtol=2e-1)
            torch.testing.assert_close(
                ref, res, atol=tolerance.atol, rtol=tolerance.rtol
            )
            # Ensure no more re-compilation after the second automatic dynamic shape version.
            if i == 0:
                self.assertEqual(torch._dynamo.utils.counters["frames"]["ok"], 1)
            else:
                self.assertEqual(torch._dynamo.utils.counters["frames"]["ok"], 2)

    @supported_platform
    def test_larger_block_mask_bug(self):
        def mask_mod(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx

        mask_2 = create_block_mask(
            mask_mod=mask_mod,
            B=2,
            H=None,
            Q_LEN=2,
            KV_LEN=2,
            device="cuda",
        )

        # Compile flex attention
        flex_attention_compiled = torch.compile(flex_attention, dynamic=False)

        # Create input tensors
        shape = (2, 1, 2, 16)
        q = torch.normal(0.0, 3.0, shape, device="cuda", dtype=torch.float16)
        k = torch.normal(0.0, 3.0, shape, device="cuda", dtype=torch.float16)
        v = torch.normal(0.0, 3.0, shape, device="cuda", dtype=torch.float16)
        eager = flex_attention(q, k, v, block_mask=mask_2)
        out = flex_attention_compiled(q, k, v, block_mask=mask_2)
        torch.testing.assert_close(eager, out, atol=5e-3, rtol=5e-3)


common_utils.instantiate_parametrized_tests(TestFlexDecoding)

if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
