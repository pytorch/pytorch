# Owner(s): ["module: inductor"]
# flake8: noqa: B950

import functools
from collections import namedtuple
from contextlib import nullcontext
from typing import Callable, Optional
from unittest import expectedFailure, skipUnless
from unittest.mock import patch

import torch
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.utils import run_and_get_code
from torch.nn.attention.flex_attention import (
    _create_empty_block_mask,
    _identity,
    BlockMask,
    create_block_mask,
    flex_attention,
)
from torch.testing import FileCheck
from torch.testing._internal import common_utils
from torch.testing._internal.common_cuda import PLATFORM_SUPPORTS_BF16
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
    query_ref = query.clone().detach().to(dtype).requires_grad_(query.requires_grad)
    key_ref = key.clone().detach().to(dtype).requires_grad_(key.requires_grad)
    value_ref = value.clone().detach().to(dtype).requires_grad_(value.requires_grad)
    return query_ref, key_ref, value_ref


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

        sdpa_partial = create_attention(
            score_mod=_generate_alibi_bias(8),
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

    @supported_platform
    @common_utils.parametrize("dtype", test_dtypes)
    def test_skip_odd_keys(self, dtype: torch.dtype):
        def score_mod(score, b, h, q, kv):
            return torch.where(kv % 2 == 0, score, float("-inf"))

        self.run_test(score_mod, dtype)

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

    @supported_platform
    @common_utils.parametrize("dtype", test_dtypes)
    def test_captured_buffers(self, dtype: torch.dtype):
        head_offset = torch.rand(Hq, device="cuda", dtype=dtype)

        def score_mod(score, b, h, m, n):
            return score + head_offset[h]

        self.run_test(score_mod, dtype)

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

    @supported_platform
    @common_utils.parametrize("dtype", test_dtypes_fast)
    def test_seq_masking(self, dtype):
        seq_idx = torch.zeros(S, device="cuda", dtype=torch.bool)
        seq_idx[S // 2 :] = 1

        def seq_mask_mod(score, b, h, q, kv):
            return torch.where(seq_idx[q] == seq_idx[kv], score, float("-inf"))

        self.run_test(seq_mask_mod, dtype)

    @supported_platform
    @common_utils.parametrize("dtype", test_dtypes_fast)
    def test_load_from_bias_seq_only(self, dtype):
        bias = torch.randn(1, S, device="cuda", dtype=dtype)

        def bias_mod(score, b, h, q, kv):
            return score + bias[q, kv]

        self.run_test(bias_mod, dtype)

    @supported_platform
    @common_utils.parametrize("dtype", test_dtypes_fast)
    def test_load_from_bias_seq_batch(self, dtype):
        bias = torch.randn(B, 1, S, device="cuda", dtype=dtype)

        def bias_mod(score, b, h, q, kv):
            return score + bias[b, q, kv]

        self.run_test(bias_mod, dtype)

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

    # TODO this config segfaults with Triton without:
    # https://github.com/triton-lang/triton/pull/4540
    @supported_platform
    @common_utils.parametrize("score_mod", test_score_mods)
    @common_utils.parametrize("dtype", test_dtypes)
    @common_utils.parametrize("head_dims", [(D, D // 2), (D // 2, D)])
    def test_non_equal_head_dims(self, dtype, score_mod, head_dims):
        qk_d, v_d = head_dims
        context = nullcontext() if qk_d > v_d else self.assertRaises(ValueError)
        with context:
            self.run_test(score_mod, dtype, B, Hq, 1, qk_d, B, Hkv, S, V_D=v_d)

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
        ADD = False
        self.run_test(score_mod_scale, dtype)

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


common_utils.instantiate_parametrized_tests(TestFlexDecoding)

if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
