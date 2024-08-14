# Owner(s): ["module: inductor"]
# flake8: noqa: B950

import functools
import string
from collections import namedtuple
from typing import Callable, Optional
from unittest import expectedFailure, skip, skipUnless
from unittest.mock import patch

import torch
from torch._dynamo.testing import CompileCounterWithBackend, normalize_gm
from torch._inductor import metrics
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.utils import run_and_get_code
from torch.nn.attention.flex_attention import (
    _create_empty_block_mask,
    _DEFAULT_SPARSE_BLOCK_SIZE,
    _identity,
    and_masks,
    BlockMask,
    create_block_mask,
    flex_attention,
    noop_mask,
    or_masks,
)
from torch.testing import FileCheck
from torch.testing._internal import common_utils
from torch.testing._internal.common_cuda import PLATFORM_SUPPORTS_BF16
from torch.utils._triton import has_triton


# Skip tests if Triton is not available
supported_platform = skipUnless(
    torch.cuda.is_available()
    and torch.version.hip is None
    and has_triton()
    and torch.cuda.get_device_capability() >= (8, 0),
    "Requires CUDA and Triton",
)

Tolerances = namedtuple("Tolerances", ["atol", "rtol"])
torch.set_float32_matmul_precision("high")

index = torch.ops.aten.index
Tensor = torch.Tensor


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

captured_buffers_map = {
    "_head_offset": _head_offset,
}

B = 4
H = 8
S = 2048
D = 64


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


class TestFlexAttention(InductorTestCase):
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
        # TODO: Make this check stricter after updating eager SDPA masked_softmax semantics
        if torch.isnan(compiled_error).any() and not torch.isnan(ref_error).any():
            self.assertTrue(False, "Output/Grad with NaN")
        if compiled_error > ref_error * fudge_factor:
            name = tensor_name if tensor_name is not None else ""
            msg = f"{name} Compiled error {compiled_error} is greater than ref error {ref_error} by more than {fudge_factor}X."
            self.assertTrue(False, msg)

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
        score_mod: Callable,
        dtype: torch.dtype = torch.float16,
        Q_B: int = B,
        Q_H: int = H,
        Q_S: int = S,
        Q_D: int = D,
        KV_B: int = B,
        KV_H: int = H,
        KV_S: int = S,
        KV_D: int = D,
    ):
        q = torch.randn(
            (Q_B, Q_H, Q_S, Q_D), dtype=dtype, device="cuda", requires_grad=True
        )
        k = torch.randn(
            (KV_B, KV_H, KV_S, KV_D), dtype=dtype, device="cuda", requires_grad=True
        )
        v = torch.randn(
            (KV_B, KV_H, KV_S, KV_D), dtype=dtype, device="cuda", requires_grad=True
        )
        q_ref, k_ref, v_ref = query_key_value_clones(q, k, v)
        q_gold, k_gold, v_gold = query_key_value_clones(q, k, v, torch.float64)
        block_mask = None
        sdpa_partial = create_attention(
            score_mod, block_mask, enable_gqa=(not Q_H == KV_H)
        )
        compiled_sdpa = torch.compile(sdpa_partial)
        golden_out = sdpa_partial(q_gold, k_gold, v_gold)
        ref_out = sdpa_partial(q_ref, k_ref, v_ref)
        compiled_out = compiled_sdpa(q, k, v)

        backward_grad = torch.randn((Q_B, Q_H, Q_S, Q_D), dtype=dtype, device="cuda")

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
        KV_D: int = D,
    ):
        q = torch.randn(
            (Q_B, Q_H, Q_S, Q_D), dtype=dtype, device="cuda", requires_grad=True
        )
        k = torch.randn(
            (KV_B, KV_H, KV_S, KV_D), dtype=dtype, device="cuda", requires_grad=True
        )
        v = torch.randn(
            (KV_B, KV_H, KV_S, KV_D), dtype=dtype, device="cuda", requires_grad=True
        )
        q_ref, k_ref, v_ref = query_key_value_clones(q, k, v)
        q_gold, k_gold, v_gold = query_key_value_clones(q, k, v, torch.float64)
        compiled_sdpa = torch.compile(sdpa_call)
        golden_out = sdpa_call(q_gold, k_gold, v_gold)
        ref_out = sdpa_call(q_ref, k_ref, v_ref)
        compiled_out = compiled_sdpa(q, k, v)

        backward_grad = torch.randn((Q_B, Q_H, Q_S, Q_D), dtype=dtype, device="cuda")

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
        score_mod: Callable,
        dtype: torch.dtype = torch.float16,
        B: int = B,
        H: int = H,
        S: int = S,
        D: int = D,
    ):
        sdpa_partial = create_attention(score_mod)
        # The first eager batch, shape (B, H, S, D)
        q1 = torch.randn((B, H, S, D), dtype=dtype, device="cuda", requires_grad=True)
        k1 = torch.randn((B, H, S, D), dtype=dtype, device="cuda", requires_grad=True)
        v1 = torch.randn((B, H, S, D), dtype=dtype, device="cuda", requires_grad=True)
        q1_ref, k1_ref, v1_ref = query_key_value_clones(q1, k1, v1)
        q1_gold, k1_gold, v1_gold = query_key_value_clones(q1, k1, v1, torch.float64)
        ref_out1 = sdpa_partial(q1_ref, k1_ref, v1_ref)
        golden_out1 = sdpa_partial(q1_gold, k1_gold, v1_gold)

        backward_grad1 = torch.randn((B, H, S, D), dtype=dtype, device="cuda")

        golden_out1.backward(backward_grad1.to(torch.float64))
        ref_out1.backward(backward_grad1)

        # The second eager batch, shape (B * 2, H, S / 2, D)
        B = int(B * 2)
        S = int(S / 2)
        q2 = torch.randn((B, H, S, D), dtype=dtype, device="cuda", requires_grad=True)
        k2 = torch.randn((B, H, S, D), dtype=dtype, device="cuda", requires_grad=True)
        v2 = torch.randn((B, H, S, D), dtype=dtype, device="cuda", requires_grad=True)
        q2_ref, k2_ref, v2_ref = query_key_value_clones(q2, k2, v2)
        q2_gold, k2_gold, v2_gold = query_key_value_clones(q2, k2, v2, torch.float64)
        ref_out2 = sdpa_partial(q2_ref, k2_ref, v2_ref)
        golden_out2 = sdpa_partial(q2_gold, k2_gold, v2_gold)

        backward_grad2 = torch.randn((B, H, S, D), dtype=dtype, device="cuda")

        golden_out2.backward(backward_grad2.to(torch.float64))
        ref_out2.backward(backward_grad2)

        # Need to clear dynamo counters, since flex attention eager mode also uses dynamo tracing.
        # We check dynamo counters["frames"]["ok"] to ensure there is no re-compilation.
        torch._dynamo.reset()
        # Compiling with dynamic shape in the first batch.
        compiled_sdpa = torch.compile(sdpa_partial, dynamic=True)
        compiled_out1 = compiled_sdpa(q1, k1, v1)
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
        self.assertEqual(torch._dynamo.utils.counters["frames"]["ok"], 1)

        # No re-compilation, use the compiled dynamic shape version.
        compiled_out2 = compiled_sdpa(q2, k2, v2)
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
        self.assertEqual(torch._dynamo.utils.counters["frames"]["ok"], 1)

    def run_automatic_dynamic_test(
        self,
        score_mod: Callable,
        dtype: torch.dtype = torch.float16,
        B: int = B,
        H: int = H,
        S: int = S,
        D: int = D,
    ):
        sdpa_partial = create_attention(score_mod)
        # The first eager batch, shape (B, H, S, D)
        q1 = torch.randn((B, H, S, D), dtype=dtype, device="cuda")
        k1 = torch.randn((B, H, S, D), dtype=dtype, device="cuda")
        v1 = torch.randn((B, H, S, D), dtype=dtype, device="cuda")
        golden_out1 = sdpa_partial(
            q1.to(torch.float64), k1.to(torch.float64), v1.to(torch.float64)
        )
        ref_out1 = sdpa_partial(q1, k1, v1)

        # The second eager batch, shape (B * 2, H, S / 2, D)
        B = int(B * 2)
        S = int(S / 2)
        q2 = torch.randn((B, H, S, D), dtype=dtype, device="cuda")
        k2 = torch.randn((B, H, S, D), dtype=dtype, device="cuda")
        v2 = torch.randn((B, H, S, D), dtype=dtype, device="cuda")
        golden_out2 = sdpa_partial(
            q2.to(torch.float64), k2.to(torch.float64), v2.to(torch.float64)
        )
        ref_out2 = sdpa_partial(q2, k2, v2)

        # The third eager batch, shape (B * 4, H, S / 4, D)
        B = int(B * 2)
        S = int(S / 2)
        q3 = torch.randn((B, H, S, D), dtype=dtype, device="cuda")
        k3 = torch.randn((B, H, S, D), dtype=dtype, device="cuda")
        v3 = torch.randn((B, H, S, D), dtype=dtype, device="cuda")
        golden_out3 = sdpa_partial(
            q3.to(torch.float64), k3.to(torch.float64), v3.to(torch.float64)
        )
        ref_out3 = sdpa_partial(q3, k3, v3)

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
        compiled_sdpa = torch.compile(sdpa_partial)
        compiled_out1 = compiled_sdpa(q1, k1, v1)
        self._check_equal(golden_out1, ref_out1, compiled_out1, fudge_factor)
        self.assertEqual(torch._dynamo.utils.counters["frames"]["ok"], 1)

        # The second batch (automatic dynamic).
        compiled_out2 = compiled_sdpa(q2, k2, v2)
        self._check_equal(golden_out2, ref_out2, compiled_out2, fudge_factor)
        self.assertEqual(torch._dynamo.utils.counters["frames"]["ok"], 2)

        # The third batch (no re-compilation).
        compiled_out3 = compiled_sdpa(q3, k3, v3)
        self._check_equal(golden_out3, ref_out3, compiled_out3, fudge_factor)
        self.assertEqual(torch._dynamo.utils.counters["frames"]["ok"], 2)

    @supported_platform
    @common_utils.parametrize("dtype", test_dtypes)
    @common_utils.parametrize("score_mod", test_score_mods)
    def test_builtin_score_mods(self, dtype: torch.dtype, score_mod: Callable):
        self.run_test(score_mod, dtype)

    @expectedFailure  # TODO: supports block sparsity with dynamic shapes
    @supported_platform
    @common_utils.parametrize("dtype", test_dtypes)
    @common_utils.parametrize("score_mod", test_score_mods)
    def test_builtin_score_mods_dynamic(self, dtype: torch.dtype, score_mod: Callable):
        self.run_dynamic_test(score_mod, dtype)

    @expectedFailure  # TODO: supports block sparsity with dynamic shapes
    @supported_platform
    @common_utils.parametrize("dtype", test_dtypes)
    @common_utils.parametrize("score_mod", test_score_mods)
    def test_builtin_score_mods_automatic_dynamic(
        self, dtype: torch.dtype, score_mod: Callable
    ):
        self.run_automatic_dynamic_test(score_mod, dtype)

    @supported_platform
    @common_utils.parametrize("dtype", test_dtypes_fast)
    @common_utils.parametrize("score_mod", test_score_mods)
    def test_builtin_score_mods_different_seqlen(
        self, dtype: torch.dtype, score_mod: Callable
    ):
        self.run_test(
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

    @supported_platform
    @common_utils.parametrize("dtype", test_dtypes_fast)
    @common_utils.parametrize("score_mod", test_score_mods)
    def test_GQA(self, dtype: torch.dtype, score_mod: Callable):
        self.run_test(
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
        sdpa_partial = create_attention(
            score_mod=_generate_alibi_bias(8), block_mask=block_mask
        )
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

    @supported_platform
    def test_index_multiple(self):
        bias = torch.randn(B, S, device="cuda")

        def index_multiple(score, b, h, q_idx, kv_idx):
            return score + bias[b][q_idx]

        self.run_test(index_multiple, torch.float16)

    @supported_platform
    def test_index_weird1(self):
        bias = torch.randn(4, B, H, S, device="cuda")

        def index_weird1(score, b, h, q_idx, kv_idx):
            return score + bias[0][b, h][q_idx]

        self.run_test(index_weird1, torch.float16)

    @supported_platform
    def test_index_weird2(self):
        bias = torch.randn(B, H, 4, S, device="cuda")
        which_bias = torch.tensor(0, device="cuda")

        def index_weird2(score, b, h, q_idx, kv_idx):
            return score + bias[b][h][which_bias, q_idx]

        self.run_test(index_weird2, torch.float16)

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
        head_offset = torch.rand(H, device="cuda", dtype=dtype)

        def score_mod(score, b, h, m, n):
            return score + head_offset[h]

        self.run_test(score_mod, dtype)

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
        bias = torch.randn(S, S, device="cuda", dtype=dtype)

        def bias_mod(score, b, h, q, kv):
            return score + bias[q, kv]

        self.run_test(bias_mod, dtype)

    @supported_platform
    @common_utils.parametrize("dtype", test_dtypes_fast)
    def test_load_from_bias_seq_batch(self, dtype):
        bias = torch.randn(B, S, S, device="cuda", dtype=dtype)

        def bias_mod(score, b, h, q, kv):
            return score + bias[b, q, kv]

        self.run_test(bias_mod, dtype)

    @supported_platform
    @common_utils.parametrize("dtype", test_dtypes_fast)
    def test_load_from_bias_head_seq_batch(self, dtype):
        bias = torch.randn(B, H, S, S, device="cuda", dtype=dtype)

        def bias_mod(score, b, h, q, kv):
            return score + bias[b, h, q, kv]

        self.run_test(bias_mod, dtype)

    @supported_platform
    @common_utils.parametrize("dtype", test_dtypes_fast)
    def test_load_rel_bias(self, dtype):
        rel_bias = torch.randn(2 * S, device="cuda", dtype=dtype)

        def bias_mod(score, b, h, q, kv):
            return score + rel_bias[(q - kv) + S]

        self.run_test(bias_mod, dtype)

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

        block_mask_a = create_block_mask(causal_mask, 1, 1, 512, 512, _compile=True)
        block_mask_b = create_block_mask(causal_mask, 1, 1, 512, 512, _compile=False)
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
                pass

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
    def test_GQA_causal_mask(self):
        def mask_mod(b, h, q, kv):
            return q >= kv

        block_mask = create_block_mask(mask_mod, 1, 1, S // 8, S // 8)
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
            (2, 2, 128, 4),
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
    def test_differentiable_logsumexp_gradcheck(self):
        make_tensor = functools.partial(
            torch.randn,
            (2, 2, 128, 4),
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

    @supported_platform
    @common_utils.parametrize("score_mod_name", ["_head_offset"])
    @common_utils.parametrize("mode", ["eager", "aot_eager"])
    def test_captured_score_mod_aot_eager_gradcheck(
        self, score_mod_name: str, mode: str
    ):
        make_tensor = functools.partial(
            torch.randn,
            (2, 2, 128, 4),
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
    def test_fully_masked_out_rows(self):
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
        do = torch.randn((B, H, S, D), dtype=torch.float32, device="cuda")

        M = S // 2

        def mask_mod(b, h, q, kv):
            return q < M

        block_mask = create_block_mask(mask_mod, 1, 1, S, S)
        out = torch.compile(flex_attention, dynamic=False)(
            query, key, value, block_mask=block_mask
        )
        # TODO: Switch to self.run_test_with_call after updating eager SDPA masked_softmax semantics
        self.assertEqual(out[:, :, M:, :].sum(), 0)

        out.backward(do)
        self.assertEqual(query.grad[:, :, M:, :].sum(), 0)

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
    def test_causal_block_non_divisible(self):
        def mask_mod(b, h, q, kv):
            return q >= kv

        block_mask = create_block_mask(mask_mod, 1, 1, S - 1, S - 1)
        attention = functools.partial(flex_attention, block_mask=block_mask)

        self.run_test_with_call(attention, Q_S=S - 1, KV_S=S - 1)

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

        block_mask = create_block_mask(mask_mod, 1, 1, Q_S, KV_S)
        # block_mask = None
        attention = functools.partial(flex_attention, block_mask=block_mask)

        self.run_test_with_call(attention, Q_S=Q_S, KV_S=KV_S)

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
    def forward(self, L_query_: "f64[2, 2, 128, 4]", L_key_: "f64[2, 2, 128, 4]", L_value_: "f64[2, 2, 128, 4]", L_block_mask_kv_num_blocks: "i32[1, 1, 1]", L_block_mask_kv_indices: "i32[1, 1, 1, 1]", L_block_mask_full_kv_num_blocks: "i32[1, 1, 1]", L_block_mask_full_kv_indices: "i32[1, 1, 1, 1]", L_block_mask_q_num_blocks: "i32[1, 1, 1]", L_block_mask_q_indices: "i32[1, 1, 1, 1]", L_block_mask_full_q_num_blocks: "i32[1, 1, 1]", L_block_mask_full_q_indices: "i32[1, 1, 1, 1]"):
        l_query_ = L_query_
        l_key_ = L_key_
        l_value_ = L_value_
        l_block_mask_kv_num_blocks = L_block_mask_kv_num_blocks
        l_block_mask_kv_indices = L_block_mask_kv_indices
        l_block_mask_full_kv_num_blocks = L_block_mask_full_kv_num_blocks
        l_block_mask_full_kv_indices = L_block_mask_full_kv_indices
        l_block_mask_q_num_blocks = L_block_mask_q_num_blocks
        l_block_mask_q_indices = L_block_mask_q_indices
        l_block_mask_full_q_num_blocks = L_block_mask_full_q_num_blocks
        l_block_mask_full_q_indices = L_block_mask_full_q_indices

        child_1: "i32[]" = l_query_.new_empty([], dtype = torch.int32);  child_1 = None
        child_2: "i32[]" = l_query_.new_empty([], dtype = torch.int32);  child_2 = None
        child_3: "i32[]" = l_query_.new_empty([], dtype = torch.int32);  child_3 = None
        child_4: "i32[]" = l_query_.new_empty([], dtype = torch.int32);  child_4 = None
        child: "f64[]" = l_query_.new_empty([], requires_grad = True);  child = None
        score_mod_0 = self.score_mod_0
        child_5: "i32[]" = l_query_.new_empty([], dtype = torch.int32);  child_5 = None
        child_6: "i32[]" = l_query_.new_empty([], dtype = torch.int32);  child_6 = None
        child_7: "i32[]" = l_query_.new_empty([], dtype = torch.int32);  child_7 = None
        child_8: "i32[]" = l_query_.new_empty([], dtype = torch.int32);  child_8 = None
        mask_fn_0 = self.mask_fn_0
        flex_attention = torch.ops.higher_order.flex_attention(l_query_, l_key_, l_value_, score_mod_0, (l_block_mask_kv_num_blocks, l_block_mask_kv_indices, l_block_mask_full_kv_num_blocks, l_block_mask_full_kv_indices, l_block_mask_q_num_blocks, l_block_mask_q_indices, l_block_mask_full_q_num_blocks, l_block_mask_full_q_indices, 128, 128, mask_fn_0), 0.5, {'ROWS_GUARANTEED_SAFE': False, 'PRESCALE_QK': False, 'OUTPUT_LOGSUMEXP': True, 'IS_DIVISIBLE': True}, (), ());  l_query_ = l_key_ = l_value_ = score_mod_0 = l_block_mask_kv_num_blocks = l_block_mask_kv_indices = l_block_mask_full_kv_num_blocks = l_block_mask_full_kv_indices = l_block_mask_q_num_blocks = l_block_mask_q_indices = l_block_mask_full_q_num_blocks = l_block_mask_full_q_indices = mask_fn_0 = None
        out: "f64[2, 2, 128, 4]" = flex_attention[0];  flex_attention = None
        return (out,)

    class score_mod_0(torch.nn.Module):
        def forward(self, child: "f64[]", child_1: "i32[]", child_2: "i32[]", child_3: "i32[]", child_4: "i32[]"):
            mul: "f64[]" = child * child;  child = None
            return mul

    class mask_fn_0(torch.nn.Module):
        def forward(self, child_5: "i32[]", child_6: "i32[]", child_7: "i32[]", child_8: "i32[]"):
            ge: "b8[]" = child_7 >= child_8;  child_7 = child_8 = None
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
        fw_graph = self.fw_graph
        joint_graph = self.joint_graph
        mask_graph = self.mask_graph
        flex_attention_backward = torch.ops.higher_order.flex_attention_backward(primals_1, primals_2, primals_3, getitem_2, getitem_3, tangents_1, full_default_4, fw_graph, joint_graph, (full, full_default, None, None, convert_element_type, convert_element_type_1, None, None, 128, 128, mask_graph), 0.5, {'ROWS_GUARANTEED_SAFE': False, 'PRESCALE_QK': False, 'OUTPUT_LOGSUMEXP': True, 'IS_DIVISIBLE': True}, (), ());  primals_1 = primals_2 = primals_3 = getitem_2 = getitem_3 = tangents_1 = full_default_4 = fw_graph = joint_graph = full = full_default = convert_element_type = convert_element_type_1 = mask_graph = None
        getitem_4: "f64[2, 2, 128, 4]" = flex_attention_backward[0]
        getitem_5: "f64[2, 2, 128, 4]" = flex_attention_backward[1]
        getitem_6: "f64[2, 2, 128, 4]" = flex_attention_backward[2];  flex_attention_backward = None
        return (getitem_4, getitem_5, getitem_6)

    class fw_graph(torch.nn.Module):
        def forward(self, arg0_1: "f64[]", arg1_1: "i32[]", arg2_1: "i32[]", arg3_1: "i32[]", arg4_1: "i32[]"):
            mul: "f64[]" = torch.ops.aten.mul.Tensor(arg0_1, arg0_1);  arg0_1 = None
            return mul

    class joint_graph(torch.nn.Module):
        def forward(self, arg0_1: "f64[]", arg1_1: "i32[]", arg2_1: "i32[]", arg3_1: "i32[]", arg4_1: "i32[]", arg5_1: "f64[]"):
            mul: "f64[]" = torch.ops.aten.mul.Tensor(arg0_1, arg0_1);  mul = None
            mul_1: "f64[]" = torch.ops.aten.mul.Tensor(arg5_1, arg0_1)
            mul_2: "f64[]" = torch.ops.aten.mul.Tensor(arg5_1, arg0_1);  arg5_1 = arg0_1 = None
            add: "f64[]" = torch.ops.aten.add.Tensor(mul_2, mul_1);  mul_2 = mul_1 = None
            return [add, None, None, None, None]

    class mask_graph(torch.nn.Module):
        def forward(self, arg0_1: "i32[]", arg1_1: "i32[]", arg2_1: "i32[]", arg3_1: "i32[]"):
            full: "b8[]" = torch.ops.aten.full.default([], True, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
            return full
""",  # noqa: B950
        )


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
            )

    @supported_platform
    @common_utils.parametrize("compile", [False, True])
    def test_no_q_info(self, compile: bool):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

        torch.testing.assert_close(causal_mask_out, sdpa_mask_out, atol=1e-3, rtol=0.0)


common_utils.instantiate_parametrized_tests(TestFlexAttention)
common_utils.instantiate_parametrized_tests(TestBlockMask)

if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
