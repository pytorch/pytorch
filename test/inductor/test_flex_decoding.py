# Owner(s): ["module: inductor"]
# flake8: noqa: B950

import functools
import sys
import unittest
from collections import namedtuple
from collections.abc import Callable
from typing import Optional, Union
from unittest import expectedFailure
from unittest.mock import patch

import torch
from torch._inductor.exc import InductorError
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
from torch.testing._internal.common_cuda import (
    PLATFORM_SUPPORTS_BF16,
    PLATFORM_SUPPORTS_FP8,
    with_tf32_off,
)
from torch.testing._internal.common_device_type import (
    E4M3_MAX_POS,
    e4m3_type,
    flex_attention_supported_platform as supported_platform,
    instantiate_device_type_tests,
    skipXPUIf,
)
from torch.testing._internal.common_quantized import _snr
from torch.testing._internal.common_utils import IS_CI, IS_WINDOWS
from torch.utils._triton import has_triton_tma_device


if IS_WINDOWS and IS_CI:
    # TODO(xuhancn) : Need track if it is a requirement on windows.
    sys.stderr.write("This UT is validated on windows, a lot of crash. Skip it.\n")
    if __name__ == "__main__":
        sys.exit(0)
    raise unittest.SkipTest("skip on Windows")


Tolerances = namedtuple("Tolerances", ["atol", "rtol"])
if torch.version.hip:
    torch.set_float32_matmul_precision("highest")
else:
    torch.set_float32_matmul_precision("high")

index = torch.ops.aten.index
Tensor = torch.Tensor

TEST_ON_CUDA = (
    torch.cuda.is_available()
    and torch.utils._triton.has_triton()
    and torch.cuda.get_device_capability() >= (8, 0)
)
TEST_ON_XPU = torch.xpu.is_available() and torch.utils._triton.has_triton()

if TEST_ON_CUDA:
    test_device = ("cuda",)
    test_dtypes = (
        [torch.float32, torch.bfloat16, torch.float16]
        if PLATFORM_SUPPORTS_BF16
        else [torch.float16, torch.float32]
    )
    test_dtypes_fast = [torch.float16]
    SKIP_UT_ON_CPU = False
elif TEST_ON_XPU:
    torch._C._set_onednn_allow_tf32(True)
    test_device = ("xpu",)
    test_dtypes = [torch.float32, torch.bfloat16, torch.float16]
    test_dtypes_fast = [torch.float16]
    SKIP_UT_ON_CPU = False
else:
    test_device = ("cpu",)
    torch_config_string = torch.__config__.show()
    SKIP_UT_ON_CPU = True
    LONG_COMPILATION_ON_CPU = False
    if "CLANG" in torch_config_string.upper():
        # if the compiler is clang, skip UT for CPU due to long compilation time found in CI
        # TODO: check reason of long compile time
        LONG_COMPILATION_ON_CPU = True

    test_dtypes = (
        [torch.float32, torch.bfloat16]
        if torch.backends.mkldnn.is_available()
        and torch.ops.mkldnn._is_mkldnn_bf16_supported()
        else [torch.float32]
    )
    test_dtypes_fast = [torch.float32]


def skip_on_xpu(test_func):
    """Decorator to skip tests that are not supported on Intel GPU."""
    decorated_func = skipXPUIf(True, "Not supported on Intel GPU")(test_func)
    return decorated_func


def create_attention(score_mod, block_mask, enable_gqa=False, kernel_options=None):
    return functools.partial(
        flex_attention,
        score_mod=score_mod,
        block_mask=block_mask,
        enable_gqa=enable_gqa,
        kernel_options=kernel_options,
    )


def create_block_mask_test(score_mod, query, key):
    block_mask = create_block_mask(
        score_mod, 1, 1, query.shape[-2], key.shape[-2], query.device
    )
    return block_mask


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
    return torch.tril(torch.ones(Mkv, Mkv, dtype=torch.bool, device=test_device[0]))[
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
    head_offset = torch.rand(Hq, device=test_device[0], dtype=dtype)

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
    (128, 64),
]

(Hq, Hkv) = (16, 8)


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
    def setUp(self):
        super().setUp()
        self.test_inference_only = False
        if test_device[0] == "cpu":
            if LONG_COMPILATION_ON_CPU:
                self.skipTest(
                    "skip UT for CPU due to long compilation time found in CI"
                )
            self.test_inference_only = True

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
        score_mod: Optional[Callable] = None,
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
        device="cuda",
        kernel_options=None,
    ):
        assert score_mod is not None or block_mask is not None, (
            "Must provide score_mod or block_mask"
        )
        assert Q_H % KV_H == 0
        if device == "cpu" and dtype is torch.float16:
            dtype = torch.float32

        q = torch.randn(
            (Q_B, Q_H, Q_S, Q_D),
            dtype=dtype,
            device=device,
            requires_grad=not self.test_inference_only,
        )
        k = torch.randn(
            (KV_B, KV_H, KV_S, Q_D),
            dtype=dtype,
            device=device,
            requires_grad=not self.test_inference_only,
        )
        v = torch.randn(
            (KV_B, KV_H, KV_S, V_D),
            dtype=dtype,
            device=device,
            requires_grad=not self.test_inference_only,
        )
        q_ref, k_ref, v_ref = query_key_value_clones(q, k, v)
        q_gold, k_gold, v_gold = query_key_value_clones(q, k, v, torch.float64)

        sdpa_partial = create_attention(
            score_mod,
            block_mask,
            enable_gqa=(Q_H != KV_H),
            kernel_options=kernel_options,
        )
        compiled_sdpa = torch.compile(sdpa_partial)
        if not self.test_inference_only:
            golden_out, gold_lse = sdpa_partial(q_gold, k_gold, v_gold, return_lse=True)
            ref_out, ref_lse = sdpa_partial(q_ref, k_ref, v_ref, return_lse=True)
            compiled_out, compiled_lse = compiled_sdpa(q, k, v, return_lse=True)
            self._check_out(
                gold_lse,
                ref_lse,
                compiled_lse,
            )
        else:
            golden_out = sdpa_partial(q_gold, k_gold, v_gold, return_lse=False)
            ref_out = sdpa_partial(q_ref, k_ref, v_ref, return_lse=False)
            compiled_out = compiled_sdpa(q, k, v, return_lse=False)
        self._check_out(
            golden_out,
            ref_out,
            compiled_out,
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
        device="cuda",
    ):
        if not golden_call:
            golden_call = sdpa_call

        if device == "cpu" and dtype is torch.float16:
            dtype = torch.float32

        q = torch.randn(
            (Q_B, KV_H, Q_S, Q_D),
            dtype=dtype,
            device=device,
            requires_grad=False,
        )
        k = torch.randn(
            (KV_B, KV_H, KV_S, Q_D),
            dtype=dtype,
            device=device,
            requires_grad=False,
        )
        v = torch.randn(
            (KV_B, KV_H, KV_S, V_D),
            dtype=dtype,
            device=device,
            requires_grad=False,
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
        device="cuda",
    ):
        assert block_mask is not None, "Must provide block_mask"
        if device == "cpu" and dtype is torch.float16:
            dtype = torch.float32
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
            device=device,
            dtype=dtype,
        )
        v_cache = torch.zeros(
            1,
            KV_H,
            MAX_CACHED_SEQ_LEN,
            V_D,
            device=device,
            dtype=dtype,
        )

        # "randomly" initialize the page table
        paged_attention = PagedAttention(
            n_pages, page_size, max_batch_size, device=device
        )
        batch_reserve(
            paged_attention,
            torch.tensor([KV_S // 4, KV_S // 2, KV_S // 4, KV_S // 3], device=device),
        )
        batch_reserve(
            paged_attention,
            torch.tensor([KV_S // 4, KV_S // 2, KV_S // 2, KV_S // 2], device=device),
        )
        batch_reserve(
            paged_attention,
            torch.tensor([KV_S // 2, KV_S, KV_S // 2, KV_S], device=device),
        )
        batch_reserve(
            paged_attention, torch.tensor([KV_S, KV_S, KV_S, KV_S], device=device)
        )

        # update cache with k and v
        input_pos = (
            torch.arange(KV_S, device=device, dtype=torch.int32)
            .unsqueeze(0)
            .expand(KV_B, KV_S)
        )
        batch_idx = torch.arange(KV_B, device=device, dtype=torch.int32)
        paged_attention.assign(batch_idx, input_pos, k, v, k_cache, v_cache)

        # convert block mask and score mod
        kv_len_tensor = torch.full((KV_B,), KV_S, device=device, dtype=torch.int64)
        converted_block_mask = paged_attention.convert_logical_block_mask(
            block_mask, kv_len=kv_len_tensor
        )
        converted_score_mod = paged_attention.get_score_mod(
            score_mod, kv_len=kv_len_tensor
        )

        return k_cache, v_cache, converted_block_mask, converted_score_mod

    def run_paged_attention(
        self,
        score_mod: Optional[Callable],
        q: Tensor,
        k: Tensor,
        v: Tensor,
        dtype: torch.dtype = torch.float16,
        block_mask: Optional[BlockMask] = None,
        device="cuda",
    ):
        Q_B, Q_H, KV_H = q.shape[0], q.shape[1], k.shape[1]
        if device == "cpu" and dtype is torch.float16:
            dtype = torch.float32

        if block_mask is None:
            block_mask = create_block_mask(noop_mask, Q_B, 1, 1, S, device=device)

        (
            k_cache,
            v_cache,
            converted_block_mask,
            converted_score_mod,
        ) = self.preprocess_paged_attention(
            score_mod, q, k, v, block_mask, dtype, block_mask.BLOCK_SIZE[1], device
        )

        compiled_sdpa = torch.compile(flex_attention)

        # compute
        if not self.test_inference_only:
            compiled_out, compiled_lse = compiled_sdpa(
                q,
                k_cache,
                v_cache,
                return_lse=True,
                block_mask=converted_block_mask,
                score_mod=converted_score_mod,
                enable_gqa=(Q_H != KV_H),
            )
        else:
            compiled_lse = None
            compiled_out = compiled_sdpa(
                q,
                k_cache,
                v_cache,
                return_lse=False,
                block_mask=converted_block_mask,
                score_mod=converted_score_mod,
                enable_gqa=(Q_H != KV_H),
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
        device="cuda",
    ):
        assert Q_H % KV_H == 0
        if device == "cpu" and dtype is torch.float16:
            dtype = torch.float32
        q = torch.randn(
            (Q_B, Q_H, Q_S, QK_D),
            dtype=dtype,
            device=device,
            requires_grad=False,
        )
        k = torch.randn(
            (KV_B, KV_H, KV_S, QK_D),
            dtype=dtype,
            device=device,
            requires_grad=False,
        )
        v = torch.randn(
            (KV_B, KV_H, KV_S, V_D),
            dtype=dtype,
            device=device,
            requires_grad=False,
        )
        q_ref, k_ref, v_ref = query_key_value_clones(q, k, v)
        q_gold, k_gold, v_gold = query_key_value_clones(q, k, v, torch.float64)

        if block_mask is None:
            block_mask = create_block_mask(noop_mask, Q_B, 1, 1, KV_S, device=device)

        sdpa_partial = create_attention(score_mod, block_mask, enable_gqa=(Q_H != KV_H))
        golden_out, gold_lse = sdpa_partial(q_gold, k_gold, v_gold, return_lse=True)
        ref_out, ref_lse = sdpa_partial(q_ref, k_ref, v_ref, return_lse=True)

        compiled_out, compiled_lse = self.run_paged_attention(
            score_mod, q, k, v, dtype, block_mask, device
        )

        self._check_out(
            golden_out,
            ref_out,
            compiled_out,
        )
        if not self.test_inference_only:
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
        device="cuda",
    ):
        if device == "cpu" and dtype is torch.float16:
            dtype = torch.float32

        q = torch.randn(
            (Q_B, KV_H, Q_S * (Q_H // KV_H), Q_D),
            dtype=dtype,
            device=device,
            requires_grad=False,
        )
        k = torch.randn(
            (KV_B, KV_H, KV_S, Q_D),
            dtype=dtype,
            device=device,
            requires_grad=False,
        )
        v = torch.randn(
            (KV_B, KV_H, KV_S, V_D),
            dtype=dtype,
            device=device,
            requires_grad=False,
        )
        q_ref, k_ref, v_ref = query_key_value_clones(q, k, v)
        q_gold, k_gold, v_gold = query_key_value_clones(q, k, v, torch.float64)

        golden_call = functools.partial(
            torch.nn.functional.scaled_dot_product_attention, attn_mask=sdpa_mask
        )
        golden_out = golden_call(q_gold, k_gold, v_gold)
        ref_out = golden_call(q_ref, k_ref, v_ref)

        if mask_mod is not None:
            block_mask = create_block_mask(mask_mod, Q_B, 1, Q_S, KV_S, device=device)
        else:
            block_mask = create_block_mask(noop_mask, Q_B, 1, Q_S, KV_S, device=device)

        compiled_out, _ = self.run_paged_attention(
            score_mod, q, k, v, dtype, block_mask, device
        )

        self._check_out(
            golden_out,
            ref_out,
            compiled_out,
        )

    @supported_platform
    @expectedFailure  # tl.dot does not support embedding size less than 16
    @unittest.skipIf(SKIP_UT_ON_CPU, "Skip on CPU as not supported")
    @common_utils.parametrize("dtype", test_dtypes_fast)
    def test_bw_decoding_fails(self, device, dtype):
        make_kv = functools.partial(
            torch.randn,
            (2, 2, 128, 4),
            dtype=dtype,
            device=device,
            requires_grad=True,
        )
        make_q = functools.partial(
            torch.randn,
            (2, 2, 8, 4),
            dtype=dtype,
            device=device,
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
    @with_tf32_off
    def test_builtin_score_mods(
        self, device, dtype: torch.dtype, score_mod: Callable, head_dims
    ):
        Hq, Hkv = head_dims
        assert Hq % Hkv == 0
        self.run_test(score_mod, dtype, Q_H=Hq, KV_H=Hkv, device=device)
        self.run_test_with_paged_attention(
            score_mod, dtype, Q_H=Hq, KV_H=Hkv, device=device
        )

    @supported_platform
    @common_utils.parametrize("dtype", test_dtypes_fast)
    @common_utils.parametrize("score_mod", test_score_mods)
    @common_utils.parametrize("head_dims", test_Hq_Hkv)
    @common_utils.parametrize("page_size", test_page_sizes)
    def test_paged_attention_page_size(
        self,
        device,
        dtype: torch.dtype,
        score_mod: Callable,
        head_dims: tuple[int, int],
        page_size: int,
    ):
        Hq, Hkv = head_dims
        assert Hq % Hkv == 0

        def generate_causal_offset(offset: torch.Tensor):
            def causal_offset_mask(b, h, q_idx, kv_idx):
                return (offset + q_idx) >= kv_idx

            return causal_offset_mask

        mod = generate_causal_offset(
            torch.tensor(192, device=device, dtype=torch.int32)
        )
        block_mask = create_block_mask(
            mod, B, 1, 1, S, BLOCK_SIZE=page_size, device=device
        )

        self.run_test_with_paged_attention(
            score_mod,
            dtype,
            Q_B=B,
            Q_H=Hq,
            KV_B=B,
            KV_H=Hkv,
            KV_S=S,
            block_mask=block_mask,
            device=device,
        )

    @supported_platform
    @common_utils.parametrize("dtype", test_dtypes)
    @common_utils.parametrize("score_mod", test_score_mods)
    @common_utils.parametrize("BLOCK_SIZE", test_block_size)
    def test_builtin_score_mods_different_block_size(
        self,
        device,
        dtype: torch.dtype,
        score_mod: Callable,
        BLOCK_SIZE: Union[int, tuple[int, int]],
    ):
        block_mask = create_block_mask(
            noop_mask, B, 1, 1, S, BLOCK_SIZE=BLOCK_SIZE, device=device
        )
        self.run_test(score_mod, dtype, block_mask=block_mask, device=device)

    @unittest.skipIf(not has_triton_tma_device(), "Skip when TMA is not available")
    @common_utils.parametrize("dtype", test_dtypes_fast)
    def test_tma_decoding(self, device, dtype: torch.dtype):
        n_heads, head_dim, seq_len = 4, 16, 128

        score_mod = _generate_alibi_bias(n_heads)
        kernel_options = {"USE_TMA": True}
        self.run_test(
            score_mod=score_mod,
            dtype=dtype,
            Q_B=1,
            Q_H=n_heads,
            Q_S=1,
            Q_D=head_dim,
            KV_B=1,
            KV_H=n_heads,
            KV_S=seq_len,
            V_D=head_dim,
            device=device,
            kernel_options=kernel_options,
        )

    @supported_platform
    @common_utils.parametrize("dtype", test_dtypes_fast)
    @common_utils.parametrize("k_s", test_input_strides)
    @common_utils.parametrize("v_s", test_input_strides)
    @common_utils.parametrize("head_dims", test_Hq_Hkv)
    def test_strided_inputs(self, device, dtype: torch.dtype, k_s, v_s, head_dims):
        Hq, Hkv = head_dims
        assert Hq % Hkv == 0
        q1 = torch.randn((B * Hq * D), dtype=dtype, device=device)
        k1 = torch.randn((B * Hkv * S * D * 4), dtype=dtype, device=device)
        v1 = torch.randn((B * Hkv * S * D * 4), dtype=dtype, device=device)

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
            enable_gqa=(Hq != Hkv),
        )
        compiled_sdpa = torch.compile(sdpa_partial)
        ref_out = sdpa_partial(q, k, v)
        compiled_out = compiled_sdpa(q, k, v)

        tolerance = Tolerances(atol=2e-1, rtol=2e-1)
        torch.testing.assert_close(
            ref_out, compiled_out, atol=tolerance.atol, rtol=tolerance.rtol
        )

        paged_compiled_out, _ = self.run_paged_attention(
            score_mod, q, k, v, dtype, device=device
        )
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
        device,
        dtype: torch.dtype,
        head_dims: tuple[int, int],
        batch_dims: tuple[int, int],
        score_mod: Callable,
    ):
        Hq, Hkv = head_dims
        assert Hq % Hkv == 0

        Bq, Bkv = batch_dims
        assert Bq > 1 and Bkv == 1

        block_mask = create_block_mask(noop_mask, Bq, 1, 1, S, device=device)

        self.run_test(
            score_mod, dtype, Bq, Hq, 1, D, Bkv, Hkv, S, D, block_mask, device=device
        )

    @supported_platform
    @common_utils.parametrize("dtype", test_dtypes)
    def test_skip_odd_keys(self, device, dtype: torch.dtype):
        def score_mod(score, b, h, q, kv):
            return torch.where(kv % 2 == 0, score, float("-inf"))

        self.run_test(score_mod, dtype, device=device)
        self.run_test_with_paged_attention(score_mod, dtype, device=device)

    @supported_platform
    @common_utils.parametrize("dtype", test_dtypes)
    def test_function_composition(self, device, dtype: torch.dtype):
        def score_mod_1(score, b, h, m, n):
            return score + (m - n)

        def score_mod_2(score, b, h, m, n):
            return torch.where(m <= n, score, float("-inf"))

        def composed_score_mod(score, b, h, m, n):
            return score_mod_2(score_mod_1(score, b, h, m, n), b, h, m, n)

        self.run_test(composed_score_mod, dtype, device=device)
        self.run_test_with_paged_attention(composed_score_mod, dtype, device=device)

    @supported_platform
    @common_utils.parametrize("dtype", test_dtypes)
    def test_captured_buffers(self, device, dtype: torch.dtype):
        head_offset = torch.rand(Hq, device=device, dtype=dtype)

        def score_mod(score, b, h, m, n):
            return score + head_offset[h]

        self.run_test(score_mod, dtype, device=device)
        self.run_test_with_paged_attention(score_mod, dtype, device=device)

    @supported_platform
    @common_utils.parametrize("dtype", test_dtypes)
    def test_captured_buffers_all_dims(self, device, dtype: torch.dtype):
        head_scale = torch.randn(Hq, device=device)
        batch_scale = torch.randn(B, device=device)
        kv_scale = torch.randn(S, device=device)
        q_scale = torch.randn(1, device=device)

        def all_bias(score, batch, head, token_q, token_kv):
            score = score + kv_scale[token_kv]
            score = score + q_scale[token_q]
            score = score + head_scale[head]
            score = score + batch_scale[batch]
            return score

        self.run_test(all_bias, dtype, device=device)
        self.run_test_with_paged_attention(all_bias, dtype, device=device)

    @supported_platform
    @common_utils.parametrize("dtype", test_dtypes_fast)
    def test_seq_masking(self, device, dtype):
        seq_idx = torch.zeros(S, device=device, dtype=torch.bool)
        seq_idx[S // 2 :] = 1

        def seq_mask_mod(score, b, h, q, kv):
            return torch.where(seq_idx[q] == seq_idx[kv], score, float("-inf"))

        self.run_test(seq_mask_mod, dtype, device=device)
        self.run_test_with_paged_attention(seq_mask_mod, dtype, device=device)

    @supported_platform
    def test_non_divisible_offset_mask(self, device):
        KV_S = S - 3
        offset_tensor = torch.tensor(S // 2 - 3, device=device, dtype=torch.int32)

        def mask_mod(b, h, q, kv):
            return kv >= q + offset_tensor

        block_mask = create_block_mask(mask_mod, B, 1, 1, KV_S, device=device)
        self.run_test(KV_S=KV_S, block_mask=block_mask, device=device)

    @supported_platform
    def test_non_divisible_offset_mask_with_captured_buffer(self, device):
        KV_S = S - 3
        offset_kv = torch.randn(KV_S, device=device, dtype=torch.bfloat16)
        offset_tensor = torch.tensor(S // 2 - 3, device=device, dtype=torch.int32)

        def score_mod(score, b, h, q, kv):
            return score + offset_kv[kv]

        def mask_mod(b, h, q, kv):
            return kv >= q + offset_tensor

        block_mask = create_block_mask(mask_mod, B, 1, 1, KV_S, device=device)
        self.run_test(
            KV_S=KV_S, block_mask=block_mask, score_mod=score_mod, device=device
        )

    @supported_platform
    def test_non_divisible_multi_token_offset_mask(self, device):
        KV_S = S - 3
        Q_S = 3
        offset_tensor = torch.tensor(S // 2 - 1, device=device, dtype=torch.int32)

        def mask_mod(b, h, q, kv):
            return kv >= q + offset_tensor

        block_mask = create_block_mask(mask_mod, B, 1, Q_S, KV_S, device=device)
        self.run_test(Q_S=Q_S, KV_S=KV_S, block_mask=block_mask, device=device)

    @supported_platform
    @unittest.skipIf(SKIP_UT_ON_CPU, "Skip on CPU as not supported")
    def test_non_divisible_multi_token_offset_mask_with_captured_buffer(self, device):
        KV_S = S - 3
        Q_S = 3
        offset_kv = torch.randn(KV_S, device=device, dtype=torch.bfloat16)
        offset_q = torch.randn(Q_S, device=device, dtype=torch.bfloat16)
        offset_tensor = torch.tensor(S // 2 - 3, device=device, dtype=torch.int32)

        def score_mod(score, b, h, q, kv):
            return score + offset_kv[kv] + offset_q[q]

        def mask_mod(b, h, q, kv):
            return kv >= q + offset_tensor

        block_mask = create_block_mask(mask_mod, B, 1, Q_S, KV_S, device=device)
        self.run_test(
            Q_S=Q_S,
            KV_S=KV_S,
            block_mask=block_mask,
            score_mod=score_mod,
            device=device,
        )

    @supported_platform
    @common_utils.parametrize("dtype", test_dtypes_fast)
    def test_load_from_bias_seq_only(self, device, dtype):
        bias = torch.randn(1, S, device=device, dtype=dtype)

        def bias_mod(score, b, h, q, kv):
            return score + bias[q, kv]

        self.run_test(bias_mod, dtype, device=device)
        self.run_test_with_paged_attention(bias_mod, dtype, device=device)

    @supported_platform
    @common_utils.parametrize("dtype", test_dtypes_fast)
    def test_load_from_bias_seq_batch(self, device, dtype):
        bias = torch.randn(B, 1, S, device=device, dtype=dtype)

        def bias_mod(score, b, h, q, kv):
            return score + bias[b, q, kv]

        self.run_test(bias_mod, dtype, device=device)
        self.run_test_with_paged_attention(bias_mod, dtype, device=device)

    @supported_platform
    @common_utils.parametrize("dtype", test_dtypes_fast)
    def test_load_from_bias_head_seq_batch(self, device, dtype):
        bias = torch.randn(
            B,
            Hq,
            1,
            S,
            device=device,
            dtype=dtype,
        )

        def bias_mod(score, b, h, q, kv):
            return score + bias[b, h, q, kv]

        self.run_test(bias_mod, dtype, device=device)
        self.run_test_with_paged_attention(bias_mod, dtype, device=device)

    @supported_platform
    @common_utils.parametrize("score_mod", test_score_mods)
    @common_utils.parametrize("dtype", test_dtypes)
    @common_utils.parametrize("head_dims", [(D, D // 2), (D // 2, D)])
    @with_tf32_off
    def test_non_equal_head_dims(self, device, dtype, score_mod, head_dims):
        qk_d, v_d = head_dims
        self.run_test(
            score_mod, dtype, B, Hq, 1, qk_d, B, Hkv, S, V_D=v_d, device=device
        )
        self.run_test_with_paged_attention(
            score_mod, dtype, B, Hq, 1, qk_d, B, Hkv, S, V_D=v_d, device=device
        )

    @supported_platform
    @common_utils.parametrize("dtype", test_dtypes_fast)
    @common_utils.parametrize("score_mod", test_score_mods)
    @common_utils.parametrize("head_dims", test_Hq_Hkv)
    def test_head_dependent_mask_mod(
        self, device, dtype: torch.dtype, score_mod, head_dims
    ):
        Hq, Hkv = head_dims
        assert Hq % Hkv == 0

        def head_attention_mod(kv_head_num):
            head_type = torch.tensor(
                [i % kv_head_num != 0 for i in range(kv_head_num)],
                dtype=torch.bool,
                device=device,
            )

            def mask_mod(b, h, q_idx, kv_idx):
                bi_mask = head_type[h]
                causal_mask = q_idx >= kv_idx

                return bi_mask & causal_mask

            return mask_mod

        mask_mod = head_attention_mod(Hq)
        mask = create_block_mask(mask_mod, 1, Hq, 1, S, device=device)
        self.run_test(
            score_mod, dtype, Q_H=Hq, KV_H=Hkv, block_mask=mask, device=device
        )
        self.run_test_with_paged_attention(
            score_mod, dtype, Q_H=Hq, KV_H=Hkv, device=device
        )

    @supported_platform
    @common_utils.parametrize("dtype", test_dtypes_fast)
    def test_subgraph_respect_decompostion(self, device, dtype):
        from torch._decomp import core_aten_decompositions
        from torch.fx.experimental.proxy_tensor import make_fx

        def score_mod_func(score, b, h, q, kv):
            return score - q // (1 + kv)

        make_kv = functools.partial(
            torch.randn,
            (2, 2, 128, 4),
            dtype=dtype,
            device=device,
            requires_grad=False,
        )
        make_q = functools.partial(
            torch.randn,
            (2, 2, 8, 4),
            dtype=dtype,
            device=device,
            requires_grad=False,
        )
        query, key, value = make_q(), make_kv(), make_kv()
        # floor_div is not decomposed in decomposition_table is empty
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
    def test_silu_on_score(self, device, dtype):
        def silu_score(score, b, h, q, kv):
            return torch.nn.functional.silu(score)

        self.run_test(silu_score, dtype, device=device)
        self.run_test_with_paged_attention(silu_score, dtype, device=device)

    @supported_platform
    @common_utils.parametrize("dtype", test_dtypes_fast)
    def test_padded_dense_causal(self, device, dtype):
        seq_len = torch.arange(B, device=device, dtype=torch.int32) + 1

        def create_padded_dense_wrapper(orig_score_mod):
            def njt_score_mod(qk, b, h, q, kv):
                return torch.where(
                    qk <= seq_len[b], orig_score_mod(qk, b, h, q, kv), -float("inf")
                )

            return njt_score_mod

        causal_njt = create_padded_dense_wrapper(_causal)

        self.run_test(causal_njt, dtype, device=device)
        self.run_test_with_paged_attention(causal_njt, dtype, device=device)

    @supported_platform
    @common_utils.parametrize("dtype", test_dtypes_fast)
    def test_captured_scale(self, device, dtype):
        scale = torch.ones((), device=device, dtype=torch.int32)

        def score_mod_scale(qk, b, h, q, kv):
            return qk + scale

        self.run_test(score_mod_scale, dtype, device=device)
        self.run_test_with_paged_attention(score_mod_scale, dtype, device=device)

    @supported_platform
    @common_utils.parametrize("dtype", test_dtypes_fast)
    def test_recompile_changed_score_mod(self, device, dtype):
        scale = torch.ones((), device=device, dtype=torch.int32)
        ADD = True

        def score_mod_scale(qk, b, h, q, kv):
            if ADD:
                return qk + scale
            else:
                return qk * scale

        self.run_test(score_mod_scale, dtype, device=device)
        self.run_test_with_paged_attention(score_mod_scale, dtype, device=device)

        ADD = False
        self.run_test(score_mod_scale, dtype, device=device)
        self.run_test_with_paged_attention(score_mod_scale, dtype, device=device)

    @supported_platform
    @common_utils.parametrize("head_dim", [17, 24, 94, 121])
    @common_utils.parametrize("dtype", test_dtypes_fast)
    @common_utils.serialTest()
    def test_non_pow_2_headdim(self, device, dtype, head_dim):
        self.run_test(
            _rel_bias, dtype, B, Hq, S, head_dim, B, Hkv, S, head_dim, device=device
        )

    @supported_platform
    @expectedFailure  # If we capture a tensor then we can perform a reduction on it, and that shouldn't be allowed
    @common_utils.parametrize("dtype", test_dtypes_fast)
    def test_captured_reduction(self, device, dtype):
        scale = torch.randn((B, 8), device=device)

        def score_mod_scale(qk, b, h, q, kv):
            return qk + scale[b].sum(dim=-1)

        self.run_test(score_mod_scale, dtype, device=device)

    @supported_platform
    def test_multiple_score_mod_calls(self, device):
        query = torch.randn((1, 8, 4, 64), dtype=torch.float32, device=device)
        keys = [
            torch.randn((1, 8, 1024, 64), dtype=torch.float32, device=device)
            for _ in range(2)
        ]
        values = [
            torch.randn((1, 8, 1024, 64), dtype=torch.float32, device=device)
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
    def test_multiple_score_mod_calls2(self, device):
        query = torch.randn((1, 8, 4, 64), dtype=torch.float32, device=device)
        keys = [
            torch.randn((1, 8, 1024, 64), dtype=torch.float32, device=device)
            for _ in range(3)
        ]
        values = [
            torch.randn((1, 8, 1024, 64), dtype=torch.float32, device=device)
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
    def test_multiple_score_mod_calls_paged_attention(self, device):
        query = torch.randn((1, 8, 4, 64), dtype=torch.float32, device=device)
        keys = [
            torch.randn((1, 8, 1024, 64), dtype=torch.float32, device=device)
            for _ in range(2)
        ]
        values = [
            torch.randn((1, 8, 1024, 64), dtype=torch.float32, device=device)
            for _ in range(2)
        ]

        def scoremod_1(qk, b, h, q, kv):
            return qk + (q - kv)

        def scoremod_2(qk, b, h, q, kv):
            return torch.where(q >= kv, qk, -float("inf"))

        block_mask = create_block_mask(noop_mask, 1, 1, 4, 1024, device=device)

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
            scoremod_1,
            query,
            keys[0],
            values[0],
            block_mask,
            torch.float32,
            device=device,
        )
        (
            k_cache2,
            v_cache2,
            converted_block_mask2,
            converted_score_mod2,
        ) = self.preprocess_paged_attention(
            scoremod_2,
            query,
            keys[1],
            values[1],
            block_mask,
            torch.float32,
            device=device,
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
    def test_multiple_score_mod_calls_paged_attention2(self, device):
        query = torch.randn((1, 8, 4, 64), dtype=torch.float32, device=device)
        keys = [
            torch.randn((1, 8, 1024, 64), dtype=torch.float32, device=device)
            for _ in range(3)
        ]
        values = [
            torch.randn((1, 8, 1024, 64), dtype=torch.float32, device=device)
            for _ in range(3)
        ]

        def scoremod_1(qk, b, h, q, kv):
            return qk + (q - kv)

        def scoremod_2(qk, b, h, q, kv):
            return torch.where(q >= kv, qk, -float("inf"))

        block_mask = create_block_mask(noop_mask, 1, 1, 4, 1024, device=device)

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
            scoremod_1,
            query,
            keys[0],
            values[0],
            block_mask,
            torch.float32,
            device=device,
        )
        (
            k_cache2,
            v_cache2,
            converted_block_mask2,
            converted_score_mod2,
        ) = self.preprocess_paged_attention(
            scoremod_2,
            query,
            keys[1],
            values[1],
            block_mask,
            torch.float32,
            device=device,
        )
        (
            k_cache3,
            v_cache3,
            converted_block_mask3,
            converted_score_mod3,
        ) = self.preprocess_paged_attention(
            scoremod_1,
            query,
            keys[2],
            values[2],
            block_mask,
            torch.float32,
            device=device,
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
    def test_njt_causal(self, device, dtype):
        offsets = torch.tensor(
            [0, 1024, 1024 + 512, S], device=device, dtype=torch.int32
        )
        seq_idx = torch.zeros(S, device=device, dtype=torch.int32)
        for idx in range(len(offsets) - 1):
            seq_idx[offsets[idx] : offsets[idx + 1]] = idx

        def create_njt_wrapper(orig_score_mod, offsets, seq_idx):
            def njt_score_mod(qk, b, h, q, kv):
                q_nested = q - offsets[seq_idx[q]]
                kv_nested = kv - offsets[seq_idx[kv]]
                return orig_score_mod(qk, b, h, q_nested, kv_nested)

            return njt_score_mod

        causal_njt = create_njt_wrapper(_causal, offsets, seq_idx)

        self.run_test(causal_njt, dtype, device=device)
        self.run_test_with_paged_attention(causal_njt, dtype, device=device)

    @supported_platform
    @unittest.skipIf(SKIP_UT_ON_CPU, "Skip on CPU as not supported")
    def test_mixed_dtypes(self, device):
        dtype_high = torch.float16 if PLATFORM_SUPPORTS_FP8 else torch.float32
        dtype_low = e4m3_type if PLATFORM_SUPPORTS_FP8 else torch.float16
        query = torch.randn((1, 1, 8, 64), dtype=dtype_high, device=device)
        key = torch.randn((1, 1, 1024, 64), dtype=dtype_high, device=device).to(
            dtype_low
        )
        value = torch.randn((1, 1, 1024, 64), dtype=dtype_high, device=device).to(
            dtype_low
        )
        kernel_options = {"BACKEND": "TRITON_DECODE"}
        out = torch.compile(flex_attention)(
            query, key, value, _identity, kernel_options=kernel_options
        )
        self.assertEqual(out.shape, query.shape)
        self.assertEqual(out.dtype, query.dtype)

    @supported_platform
    @unittest.skipIf(SKIP_UT_ON_CPU, "Skip on CPU as not supported")
    @unittest.skipUnless(PLATFORM_SUPPORTS_FP8, "FP8 is not supported on this platform")
    def test_mixed_dtypes_sqnr_per_tensor(self, device):
        query_ref = torch.testing.make_tensor(
            (1, 1, 8, 64), dtype=torch.bfloat16, device=device
        )
        key_ref = torch.testing.make_tensor(
            (1, 1, 1024, 64), dtype=torch.bfloat16, device=device
        )
        value_ref = torch.testing.make_tensor(
            (1, 1, 1024, 64), dtype=torch.bfloat16, device=device
        )

        key_scale = torch.max(torch.abs(key_ref)) / E4M3_MAX_POS
        value_scale = torch.max(torch.abs(value_ref)) / E4M3_MAX_POS

        key_fp8 = (key_ref / key_scale).to(e4m3_type)
        value_fp8 = (value_ref / value_scale).to(e4m3_type)

        def score_mod(score, b, h, m, n):
            # Dequantize keys inside the attention score computation
            return score * key_scale

        kernel_options = {"BACKEND": "TRITON_DECODE"}
        compiled_fn = torch.compile(flex_attention, fullgraph=True)
        out = (
            compiled_fn(
                query_ref, key_fp8, value_fp8, score_mod, kernel_options=kernel_options
            )
            * value_scale
        )
        out_ref = compiled_fn(
            query_ref, key_ref, value_ref, _identity, kernel_options=kernel_options
        )
        _, _, sqnr = _snr(out_ref, out)
        self.assertGreater(sqnr, 10)

    @supported_platform
    @unittest.skipIf(SKIP_UT_ON_CPU, "Skip on CPU as not supported")
    @unittest.skipUnless(PLATFORM_SUPPORTS_FP8, "FP8 is not supported on this platform")
    def test_mixed_dtypes_sqnr_per_head(self, device):
        query_ref = torch.testing.make_tensor(
            (1, 4, 8, 64), dtype=torch.bfloat16, device=device
        )
        key_ref = torch.testing.make_tensor(
            (1, 4, 1024, 64), dtype=torch.bfloat16, device=device
        )
        value_ref = torch.testing.make_tensor(
            (1, 4, 1024, 64), dtype=torch.bfloat16, device=device
        )

        fp8_max = E4M3_MAX_POS

        key_scale = torch.amax(torch.abs(key_ref), dim=(-2, -1)) / fp8_max  # (B, H)
        value_scale = torch.amax(torch.abs(value_ref), dim=(-2, -1)) / fp8_max  # (B, H)

        key_scale_b = key_scale[..., None, None]  # (B, H, 1, 1) for broadcasting
        value_scale_b = value_scale[..., None, None]

        key_fp8 = (key_ref / key_scale_b).to(e4m3_type)
        value_fp8 = (value_ref / value_scale_b).to(e4m3_type)

        def score_mod(score, b, h, m, n):
            # Dequantize keys inside the attention score computation
            return score * key_scale[b, h]

        kernel_options = {"BACKEND": "TRITON_DECODE"}
        compiled_fn = torch.compile(flex_attention, fullgraph=True)
        out = (
            compiled_fn(
                query_ref, key_fp8, value_fp8, score_mod, kernel_options=kernel_options
            )
            * value_scale_b
        )
        out_ref = compiled_fn(
            query_ref, key_ref, value_ref, _identity, kernel_options=kernel_options
        )
        _, _, sqnr = _snr(out_ref, out)
        self.assertGreater(sqnr, 10)

    @supported_platform
    @unittest.skipIf(SKIP_UT_ON_CPU, "Skip on CPU as not supported")
    def test_mixed_dtype_backwards(self, device):
        dtype_high = torch.float16 if PLATFORM_SUPPORTS_FP8 else torch.float32
        dtype_low = e4m3_type if PLATFORM_SUPPORTS_FP8 else torch.float16
        q = torch.testing.make_tensor(
            (1, 1, 8, 64),
            dtype=dtype_high,
            device=device,
            requires_grad=True,
        )
        k = torch.testing.make_tensor(
            (1, 1, 1024, 64),
            dtype=dtype_high,
            device=device,
            requires_grad=True,
        ).to(dtype_low)
        v = torch.testing.make_tensor(
            (1, 1, 1024, 64),
            dtype=dtype_high,
            device=device,
            requires_grad=True,
        ).to(dtype_low)

        kernel_options = {"BACKEND": "TRITON_DECODE"}
        compiled_fn = torch.compile(flex_attention, fullgraph=True)

        with self.assertRaisesRegex(
            InductorError,
            "Backward pass with mixed query, key, and value dtype is not supported",
        ):
            out_mixed = (
                compiled_fn(q, k, v, _identity, kernel_options=kernel_options)
            ).mean()
            out_mixed.backward()

    @supported_platform
    @patch.object(torch._inductor.config, "max_autotune", True)
    def test_max_autotune(self, device):
        def score_mod(score, b, h, m, n):
            return score * 2

        self.run_test(score_mod, device=device)
        self.run_test_with_paged_attention(score_mod, device=device)
        self.run_test_with_paged_attention(
            score_mod=score_mod,
            dtype=torch.bfloat16,
            Q_B=4,
            Q_H=1,
            Q_S=1,
            QK_D=16,
            KV_B=4,
            KV_H=1,
            KV_S=64,
            V_D=16,
            device=device,
        )

    @supported_platform
    @patch.object(torch._inductor.config, "max_autotune", True)
    def test_max_autotune_with_captured(self, device):
        head_scale = torch.randn(Hq, device=device)
        batch_scale = torch.randn(B, device=device)
        tok_scale = torch.randn(S, device=device)
        q_scale = torch.randn(1, device=device)

        def bias_mod(score, batch, head, token_q, token_kv):
            score = score + tok_scale[token_kv]
            score = score + q_scale[token_q]
            score = score + batch_scale[batch]
            score = score + head_scale[head]
            return score

        self.run_test(bias_mod, device=device)
        self.run_test_with_paged_attention(bias_mod, device=device)

    @supported_platform
    def test_fully_masked_out_rows_0_check_gqa(self, device):
        # Ensure fully masked out rows won't cause NaNs.
        query = torch.randn(
            (B, Hq, S, D),
            dtype=torch.float32,
            device=device,
            requires_grad=not self.test_inference_only,
        )
        key = torch.randn(
            (B, Hkv, S, D),
            dtype=torch.float32,
            device=device,
            requires_grad=not self.test_inference_only,
        )
        value = torch.randn(
            (B, Hkv, S, D),
            dtype=torch.float32,
            device=device,
            requires_grad=not self.test_inference_only,
        )

        M = S // 2

        def mask_mod(b, h, q, kv):
            return q < M

        block_mask = create_block_mask(mask_mod, 1, 1, S, S, device=device)

        flex = torch.compile(flex_attention, dynamic=False)
        if not self.test_inference_only:
            out, lse = flex(
                query,
                key,
                value,
                block_mask=block_mask,
                enable_gqa=True,
                return_lse=True,
            )
            self.assertTrue((lse[:, :, M:] == -float("inf")).all())

            loss = out.sum() + lse.sum()
            loss.backward()
            self.assertEqual(query.grad[:, :, M:, :].sum(), 0)
        else:
            out = flex(
                query,
                key,
                value,
                block_mask=block_mask,
                enable_gqa=True,
                return_lse=False,
            )
        self.assertEqual(out[:, :, M:, :].sum(), 0)

    @supported_platform
    def test_windowed_no_mask_vs_sdpa(self, device):
        score_mod = _generate_windowed(1000)
        attention = functools.partial(flex_attention, score_mod=score_mod)

        sdpa_mask = _get_windowed_sdpa_mask(8, S, 1000)

        sdpa_attention = functools.partial(
            torch.nn.functional.scaled_dot_product_attention, attn_mask=sdpa_mask
        )

        self.run_test_with_call(
            attention, sdpa_attention, Q_H=16, KV_H=16, Q_S=8, device=device
        )

    @supported_platform
    def test_windowed_full_mask_vs_sdpa(self, device):
        def mask_mod(b, h, q, kv):
            return q + 1000 >= kv

        score_mod = _generate_windowed(1000)

        block_mask = create_block_mask(mask_mod, 1, 1, 8, S, device=device)
        attention = functools.partial(
            flex_attention, block_mask=block_mask, score_mod=score_mod
        )

        sdpa_mask = _get_windowed_sdpa_mask(8, S, 1000)
        sdpa_attention = functools.partial(
            torch.nn.functional.scaled_dot_product_attention, attn_mask=sdpa_mask
        )

        self.run_test_with_call(
            attention, sdpa_attention, Q_H=16, KV_H=16, Q_S=8, device=device
        )

    @supported_platform
    def test_windowed_partial_block_vs_sdpa(self, device):
        def mask_mod(b, h, q, kv):
            return q + 1000 >= kv

        block_mask = create_block_mask(mask_mod, 1, 1, 8, S, device=device)
        attention = functools.partial(flex_attention, block_mask=block_mask)

        sdpa_mask = _get_windowed_sdpa_mask(8, S, 1000)
        sdpa_attention = functools.partial(
            torch.nn.functional.scaled_dot_product_attention, attn_mask=sdpa_mask
        )

        self.run_test_with_call(
            attention, sdpa_attention, Q_H=16, KV_H=16, Q_S=8, device=device
        )

    @supported_platform
    def test_windowed_no_mask_vs_sdpa_paged_attention(self, device):
        score_mod = _generate_windowed(1000)

        sdpa_mask = _get_windowed_sdpa_mask(8, S, 1000)

        self.run_test_with_call_paged_attention(
            score_mod, None, sdpa_mask, Q_H=16, KV_H=16, Q_S=8, device=device
        )

    @supported_platform
    def test_windowed_full_mask_vs_sdpa_paged_attention(self, device):
        def mask_mod(b, h, q, kv):
            return q + 1000 >= kv

        score_mod = _generate_windowed(1000)
        sdpa_mask = _get_windowed_sdpa_mask(8, S, 1000)
        self.run_test_with_call_paged_attention(
            score_mod, mask_mod, sdpa_mask, Q_H=16, KV_H=16, Q_S=8, device=device
        )

    @supported_platform
    def test_windowed_partial_block_vs_sdpa_paged_attention(self, device):
        def mask_mod(b, h, q, kv):
            return q + 1000 >= kv

        sdpa_mask = _get_windowed_sdpa_mask(8, S, 1000)

        self.run_test_with_call_paged_attention(
            None, mask_mod, sdpa_mask, Q_H=16, KV_H=16, Q_S=8, device=device
        )

    @supported_platform
    @unittest.skipIf(SKIP_UT_ON_CPU, "Skip on CPU as not supported")
    @common_utils.parametrize("dtype", test_dtypes)
    @common_utils.parametrize("score_mod", [_identity, _causal])
    def test_logsumexp_correctness(self, device, dtype, score_mod):
        make_kv = functools.partial(
            torch.randn,
            (B, Hkv, S, D),
            dtype=dtype,
            device=device,
            requires_grad=True,
        )
        make_q = functools.partial(
            torch.randn,
            (B, Hkv, Hq // Hkv, D),
            dtype=dtype,
            device=device,
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
    @unittest.skipIf(SKIP_UT_ON_CPU, "Skip on CPU as not supported")
    def test_not_pw_of_two(self, device):
        query = torch.randn(1, 12, 1, 16, device=device)
        key = torch.randn(1, 2, 128, 16, device=device)
        value = torch.randn(1, 2, 128, 16, device=device)

        flex_compiled = torch.compile(flex_attention)
        flex_compiled(query, key, value, enable_gqa=True)

    @supported_platform
    @unittest.skipIf(SKIP_UT_ON_CPU, "Skip on CPU as not supported")
    def test_logsumexp_only_return(self, device):
        make_q = functools.partial(
            torch.randn,
            (B, Hkv, Hq // Hkv, D),
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )
        make_kv = functools.partial(
            torch.randn,
            (B, Hkv, S, D),
            dtype=torch.float32,
            device=device,
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
    @skip_on_xpu  # TODO: SYCL acc issue
    def test_non_sparse_mulitple_block_size(self, device):
        def generate_causal_offset(offset: torch.Tensor):
            def causal_offset_mask(b, h, q_idx, kv_idx):
                return (offset + q_idx) >= kv_idx

            return causal_offset_mask

        def noop(score, b, h, q_idx, kv_idx):  # noqa: F841
            return score

        mod = generate_causal_offset(
            torch.tensor(192, device=device, dtype=torch.int32)
        )
        block_mask = create_block_mask(mod, 1, 1, 1, 65, device=device)

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
            device=device,
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
            device=device,
        )

    @supported_platform
    def test_do_not_trigger_dynamic_shapes_on_empty_block_mask(self, device):
        torch._dynamo.reset()
        H = Hq
        q = torch.randn(B, H, 1, D, device=device)
        for i in range(5):
            k = torch.randn(B, H, S + i, D, device=device)
            v = torch.randn(B, H, S + i, D, device=device)
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
    @common_utils.parametrize("q_seq_len", [4, 8, 16, 23, 64])
    def test_multi_block_m(self, q_seq_len, device):
        def mask_mod(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx

        B, KV_SEQ_LEN, NUM_Q_HEAD, NUM_KV_HEAD, HEAD_DIM = 1, 128, 32, 8, 128
        mask = create_block_mask(
            mask_mod=mask_mod,
            B=1,
            H=1,
            Q_LEN=q_seq_len,
            KV_LEN=KV_SEQ_LEN,
            device=device,
            BLOCK_SIZE=16,  # follow the vllm integration
        )

        flex_attention_compiled = torch.compile(flex_attention)

        dtype = torch.float
        q = torch.randn(B, NUM_Q_HEAD, q_seq_len, HEAD_DIM, device=device, dtype=dtype)
        k = torch.randn(
            B, NUM_KV_HEAD, KV_SEQ_LEN, HEAD_DIM, device=device, dtype=dtype
        )
        v = torch.randn(
            B, NUM_KV_HEAD, KV_SEQ_LEN, HEAD_DIM, device=device, dtype=dtype
        )

        # settings for vllm integration
        kernel_options = dict(BLOCK_M=16, BLOCK_N=16)
        kwargs = dict(
            block_mask=mask,
            kernel_options=kernel_options,
            enable_gqa=True,
        )
        eager = flex_attention(q, k, v, **kwargs)
        out = flex_attention_compiled(q, k, v, **kwargs)
        torch.testing.assert_close(eager, out, atol=5e-3, rtol=5e-3)

    @supported_platform
    @common_utils.parametrize("dtype", test_dtypes_fast)
    def test_larger_block_mask_bug(self, device, dtype):
        def mask_mod(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx

        mask_2 = create_block_mask(
            mask_mod=mask_mod,
            B=2,
            H=None,
            Q_LEN=2,
            KV_LEN=2,
            device=device,
        )

        # Compile flex attention
        flex_attention_compiled = torch.compile(flex_attention, dynamic=False)

        # Create input tensors
        shape = (2, 1, 2, 16)
        q = torch.normal(0.0, 3.0, shape, device=device, dtype=dtype)
        k = torch.normal(0.0, 3.0, shape, device=device, dtype=dtype)
        v = torch.normal(0.0, 3.0, shape, device=device, dtype=dtype)
        eager = flex_attention(q, k, v, block_mask=mask_2)
        out = flex_attention_compiled(q, k, v, block_mask=mask_2)
        torch.testing.assert_close(eager, out, atol=5e-3, rtol=5e-3)

    @common_utils.parametrize("dtype", test_dtypes_fast)
    @common_utils.parametrize("score_mod", test_score_mods)
    @supported_platform
    def test_decode_at_different_input_position(
        self, device, dtype: torch.dtype, score_mod: Callable
    ):
        n_pages, page_size, max_batch_size, max_seq_len = 32, 64, 4, 512
        n_heads, head_dim = 4, 16

        def causal_mask(b, h, q, kv):
            return q >= kv

        block_mask = create_block_mask(
            causal_mask,
            max_batch_size,
            1,
            max_seq_len,
            max_seq_len,
            device=device,
            BLOCK_SIZE=page_size,
        )

        # init 4 requests with different prefill length
        prefill_length = [5, 98, 47, 194]
        queries, keys, values = [], [], []
        for seq_len in prefill_length:
            q = torch.randn(
                1,
                n_heads,
                1,
                head_dim,
                device=device,
                dtype=dtype,
                requires_grad=False,
            )
            k = torch.randn(
                1,
                n_heads,
                seq_len,
                head_dim,
                device=device,
                dtype=dtype,
                requires_grad=False,
            )
            v = torch.randn(
                1,
                n_heads,
                seq_len,
                head_dim,
                device=device,
                dtype=dtype,
                requires_grad=False,
            )
            queries.append(q)
            keys.append(k)
            values.append(v)

        # get ground truth output
        ref_outs, golden_outs = [], []
        for q, k, v in zip(queries, keys, values):
            q_ref, k_ref, v_ref = query_key_value_clones(q, k, v)
            q_gold, k_gold, v_gold = query_key_value_clones(q, k, v, torch.float64)

            slice_block_mask = block_mask._adjust(1, k_ref.shape[2])
            slice_block_mask.seq_lengths = (1, k_ref.shape[2])
            ref_out = flex_attention(
                q_ref, k_ref, v_ref, score_mod, slice_block_mask, enable_gqa=False
            )
            golden_out = flex_attention(
                q_gold, k_gold, v_gold, score_mod, slice_block_mask, enable_gqa=False
            )

            ref_outs.append(ref_out)
            golden_outs.append(golden_out)
        ref_outs = torch.cat(ref_outs)
        golden_outs = torch.cat(golden_outs)

        # init paged attention
        paged_cache = PagedAttention(n_pages, page_size, max_batch_size, device=device)
        batch_reserve(paged_cache, torch.tensor([100, 200, 50, 300], device=device))
        batch_reserve(paged_cache, torch.tensor([100, 512, 300, 300], device=device))
        batch_reserve(paged_cache, torch.tensor([512, 512, 300, 300], device=device))
        batch_reserve(paged_cache, torch.tensor([512, 512, 512, 300], device=device))
        batch_reserve(paged_cache, torch.tensor([512, 512, 512, 512], device=device))

        # allocate paged kv cache
        MAX_CACHED_SEQ_LEN = n_pages * page_size
        k_cache = torch.zeros(
            1,
            n_heads,
            MAX_CACHED_SEQ_LEN,
            head_dim,
            device=device,
            dtype=dtype,
        )
        v_cache = torch.zeros(
            1,
            n_heads,
            MAX_CACHED_SEQ_LEN,
            head_dim,
            device=device,
            dtype=dtype,
        )

        # prefill paged kv cache
        for i, seq_len in enumerate(prefill_length):
            batch_idx = torch.tensor([i], device=device, dtype=torch.int32)
            input_pos = torch.arange(seq_len, device=device, dtype=torch.int32).view(
                1, seq_len
            )
            paged_cache.assign(
                batch_idx, input_pos, keys[i], values[i], k_cache, v_cache
            )

        # get paged out and check correctness
        batch_idx = torch.arange(max_batch_size, device=device, dtype=torch.int32)
        input_pos = torch.tensor(prefill_length, device=device, dtype=torch.int32).view(
            max_batch_size, 1
        )
        kv_len_tensor = torch.full(
            (max_batch_size,), max_seq_len, device=device, dtype=torch.int64
        )
        new_block_mask = paged_cache.convert_logical_block_mask(
            block_mask, kv_len=kv_len_tensor
        )
        new_block_mask.seq_lengths = (1, new_block_mask.seq_lengths[1])
        compiled_sdpa = torch.compile(
            create_attention(
                paged_cache.get_score_mod(score_mod, kv_len=kv_len_tensor),
                new_block_mask,
                enable_gqa=False,
            )
        )
        paged_out = compiled_sdpa(
            torch.cat(queries, 0), k_cache, v_cache, block_mask=new_block_mask
        )

        with torch.no_grad():
            dtype = paged_out.dtype
            if dtype == torch.float32:
                fudge_factor = 10.0
            else:
                fudge_factor = 1.1

            # Checkout output
            self._check_equal(golden_outs, ref_outs, paged_out, fudge_factor, "Out")


instantiate_device_type_tests(
    TestFlexDecoding, globals(), only_for=test_device, allow_xpu=True
)

if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
