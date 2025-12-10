# Owner(s): ["module: inductor"]
# flake8: noqa: B950

import functools
import json
import os
import random
import string
import tempfile
import unittest
import warnings
from collections import namedtuple
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import product
from typing import Optional, TypeVar, Union
from unittest import expectedFailure, mock, skip, skipUnless
from unittest.mock import patch

import torch
import torch.nn as nn
from torch._dynamo.testing import CompileCounterWithBackend, normalize_gm
from torch._inductor import config, metrics
from torch._inductor.runtime.triton_compat import HAS_WARP_SPEC
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.utils import run_and_get_code
from torch.nn.attention import SDPBackend
from torch.nn.attention.experimental._paged_attention import PagedAttention
from torch.nn.attention.flex_attention import (
    _apply_kernel_options,
    _create_empty_block_mask,
    _DEFAULT_SPARSE_BLOCK_SIZE,
    _identity,
    _mask_mod_signature,
    _score_mod_signature,
    _WARNINGS_SHOWN,
    and_masks,
    AuxOutput,
    AuxRequest,
    BlockMask,
    create_block_mask,
    flex_attention,
    flex_attention_hop,
    noop_mask,
    or_masks,
)
from torch.testing import FileCheck
from torch.testing._internal import common_utils
from torch.testing._internal.common_cuda import PLATFORM_SUPPORTS_BF16, TEST_MULTIGPU
from torch.testing._internal.common_device_type import (
    dtypes,
    dtypesIfCUDA,
    dtypesIfXPU,
    flex_attention_supported_platform as supported_platform,
    instantiate_device_type_tests,
    largeTensorTest,
    skipCPUIf,
    skipCUDAIf,
    skipXPUIf,
)
from torch.testing._internal.inductor_utils import HAS_GPU
from torch.utils._triton import has_triton, has_triton_tma_device


# Use this decorator only when hitting Triton bugs on H100
running_on_a100_only = skipUnless(
    (
        (torch.cuda.is_available() and has_triton())
        and (torch.cuda.get_device_capability() == (8, 0) or torch.version.hip)
    )
    or (torch.xpu.is_available() and has_triton()),
    "Requires Triton + A100 or Triton + ROCm or Triton + Intel GPU",
)

Tolerances = namedtuple("Tolerances", ["atol", "rtol"])
torch.set_float32_matmul_precision("high")

index = torch.ops.aten.index
Tensor = torch.Tensor


T = TypeVar("T")
M = TypeVar("M", bound=Callable)


def large_tensor_test_class(
    size: str, device: Optional[Union[torch.device, str]] = None
) -> Callable[[type[T]], type[T]]:
    def decorator(cls: type[T]) -> type[T]:
        for name, method in list(cls.__dict__.items()):
            if callable(method) and name.startswith("test_"):
                setattr(cls, name, largeTensorTest(size, device)(method))
        return cls

    return decorator


@contextmanager
def temp_float32_matmul_precision(precision: str):
    """
    Temporarily set the float32 matmul precision and restore it after the context is exited.

    Args:
    precision (str): The precision to set ('highest', 'high', or 'medium').
    """

    def set_float32_matmul_precision_xpu(precision: str):
        if precision == "highest":
            torch._C._set_onednn_allow_tf32(False)
        if precision == "high":
            torch._C._set_onednn_allow_tf32(True)

    original_precision = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision(precision)
        if TEST_ON_XPU:
            set_float32_matmul_precision_xpu(precision)
        yield
    finally:
        torch.set_float32_matmul_precision(original_precision)
        if TEST_ON_XPU:
            set_float32_matmul_precision_xpu(original_precision)


def skip_on_cpu(test_func):
    """Decorator to skip tests that are not supported on CPU."""
    decorated_func = skipCPUIf(True, "Not supported on CPU")(test_func)
    return decorated_func


def skip_on_cuda(test_func):
    """Decorator to skip tests that are not supported on CUDA."""
    decorated_func = skipCUDAIf(True, "Not supported on CUDA")(test_func)
    return decorated_func


def skip_on_rocm(test_func):
    """Decorator to skip tests that are not supported on CUDA."""
    IS_ROCM = torch.cuda.is_available() and torch.version.hip
    decorated_func = skipCUDAIf(IS_ROCM, "Not supported on ROCM")(test_func)
    return decorated_func


def skip_on_xpu(test_func):
    """Decorator to skip tests that are not supported on Intel GPU."""
    decorated_func = skipXPUIf(True, "Not supported on Intel GPU")(test_func)
    return decorated_func


def rmse(ref, res):
    """
    Calculate root mean squared error
    """
    ref = ref.to(torch.float64)
    res = res.to(torch.float64)
    return torch.sqrt(torch.mean(torch.square(ref - res)))


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
        score_mod,
        1,
        1,
        query.shape[-2],
        key.shape[-2],
        query.device,
    )
    return block_mask


@dataclass
class DeviceConfig:
    dtypes: list[torch.dtype]
    dtypes_fast: list[torch.dtype]


TEST_ON_CUDA = (
    torch.cuda.is_available()
    and torch.utils._triton.has_triton()
    and torch.cuda.get_device_capability() >= (8, 0)
)
TEST_ON_XPU = torch.xpu.is_available() and torch.utils._triton.has_triton()

device_configs = {}
if HAS_GPU:
    if TEST_ON_CUDA:
        test_device = (
            "cuda",
            "cpu",
        )
    elif TEST_ON_XPU:
        torch._C._set_onednn_allow_tf32(True)
        test_device = ("xpu",)
else:
    test_device = ("cpu",)


class SubstringSet:
    def __init__(self, items):
        self.items = set(items)

    def __contains__(self, item):
        if "cuda" in item:
            item = "cuda"
        if "xpu" in item:
            item = "xpu"
        return item in self.items


DEVICE_SUPPORTS_BACKWARDS = SubstringSet(
    [
        "cuda",
    ]
)

device_configs["cuda"] = DeviceConfig(
    dtypes=(
        [torch.float32, torch.bfloat16, torch.float16]
        if PLATFORM_SUPPORTS_BF16
        else [torch.float16, torch.float32]
    ),
    dtypes_fast=[torch.float16],
)
device_configs["xpu"] = DeviceConfig(
    dtypes=([torch.float32, torch.bfloat16, torch.float16]),
    dtypes_fast=[torch.float16],
)
device_configs["cpu"] = DeviceConfig(
    dtypes=(
        [torch.float32, torch.bfloat16, torch.float16]
        if torch.backends.mkldnn.is_available()
        and torch.ops.mkldnn._is_mkldnn_bf16_supported()
        else [torch.float32]
    ),
    dtypes_fast=[torch.float32],
)

torch_config_string = torch.__config__.show()
LONG_COMPILATION_ON_CPU = False

if "CLANG" in torch_config_string.upper():
    # if the compiler is clang, skip UT for CPU due to long compilation time found in CI
    # TODO: check reason of long compile time
    LONG_COMPILATION_ON_CPU = True


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


def _head_offset(dtype: torch.dtype, device: str):
    """Captured Buffer"""
    head_offset = torch.rand(H, device=device, dtype=dtype)

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

B = 2
H = 4
S = 256
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


def query_key_value_clones(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    dtype: Optional[torch.dtype] = None,
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


@large_tensor_test_class("2GB", device=test_device[0])
class TestFlexAttention(InductorTestCase):
    def setUp(self):
        super().setUp()
        skipCPUIf(
            LONG_COMPILATION_ON_CPU,
            "skip UT for CPU due to long compilation time found in CI",
        )

    def _check_equal(
        self,
        golden_out: torch.Tensor,
        ref_out: torch.Tensor,
        compiled_out: torch.Tensor,
        fudge_factor: float,
        tensor_name: Optional[str] = None,
        fudge_atol: float = 0,
    ):
        compiled_error = (golden_out - compiled_out).abs().mean()
        ref_error = (golden_out - ref_out).abs().mean()
        if torch.isnan(compiled_error).any() or torch.isnan(ref_error).any():
            self.fail("Output/Grad with NaN")
        name = tensor_name if tensor_name is not None else ""
        msg = f"{name} Compiled error {compiled_error} is greater than ref error {ref_error} by more than {fudge_factor}X."
        torch.testing.assert_close(
            compiled_error, ref_error, rtol=fudge_factor, atol=1e-7, msg=msg
        )

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
        dtype: torch.dtype,
        device: str,
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
        requires_grad = device in DEVICE_SUPPORTS_BACKWARDS
        if KV_B is None:
            KV_B = Q_B
        if KV_H is None:
            KV_H = Q_H
        if KV_S is None:
            KV_S = Q_S
        if V_D is None:
            V_D = Q_D

        if device == "cpu" and dtype is torch.float16:
            dtype = torch.float32

        requires_grad = device in DEVICE_SUPPORTS_BACKWARDS
        q = torch.randn(
            (Q_B, Q_H, Q_S, Q_D),
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
        k = torch.randn(
            (KV_B, KV_H, KV_S, Q_D),
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
        v = torch.randn(
            (KV_B, KV_H, KV_S, V_D),
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
        if block_mask is None:
            block_mask = create_block_mask(
                noop_mask, Q_B, Q_H, Q_S, KV_S, device=device
            )
        q_ref, k_ref, v_ref = query_key_value_clones(q, k, v)
        q_gold, k_gold, v_gold = query_key_value_clones(q, k, v, torch.float64)
        sdpa_partial = create_attention(score_mod, block_mask, enable_gqa=(Q_H != KV_H))

        compiled_sdpa = torch.compile(sdpa_partial)
        golden_out = sdpa_partial(q_gold, k_gold, v_gold)
        ref_out = sdpa_partial(q_ref, k_ref, v_ref)
        compiled_out = compiled_sdpa(q, k, v)

        assert isinstance(golden_out, torch.Tensor)
        assert isinstance(ref_out, torch.Tensor)
        assert isinstance(compiled_out, torch.Tensor)

        if not requires_grad:
            self._check_out(
                golden_out,
                ref_out,
                compiled_out,
                is_paged_attention=False,
            )
        else:
            backward_grad = torch.randn(
                (Q_B, Q_H, Q_S, V_D), dtype=dtype, device=device
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
        dtype: torch.dtype,
        device: str,
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
        dtype: torch.dtype,
        device: str,
        block_mask: Optional[BlockMask] = None,
        kernel_options: Optional[dict] = None,
    ) -> tuple[Tensor, Tensor]:
        B, Q_H, Q_S, KV_H, KV_S = (
            q.shape[0],
            q.shape[1],
            q.shape[2],
            k.shape[1],
            k.shape[2],
        )

        if block_mask is None:
            block_mask = create_block_mask(noop_mask, B, 1, Q_S, KV_S, device=device)

        (
            k_cache,
            v_cache,
            converted_block_mask,
            converted_score_mod,
        ) = self.preprocess_paged_attention(
            score_mod, q, k, v, block_mask, dtype, device, block_mask.BLOCK_SIZE[1]
        )

        compiled_sdpa = torch.compile(flex_attention)

        # compute
        return_lse = True
        requires_grad = device in DEVICE_SUPPORTS_BACKWARDS
        if requires_grad:
            compiled_out, compiled_lse = compiled_sdpa(
                q,
                k_cache,
                v_cache,
                return_lse=return_lse,
                block_mask=converted_block_mask,
                score_mod=converted_score_mod,
                enable_gqa=(Q_H != KV_H),
                kernel_options=kernel_options,
            )
        else:
            return_lse = False
            compiled_lse = None
            compiled_out = compiled_sdpa(
                q,
                k_cache,
                v_cache,
                return_lse=return_lse,
                block_mask=converted_block_mask,
                score_mod=converted_score_mod,
                enable_gqa=(Q_H != KV_H),
                kernel_options=kernel_options,
            )
        return compiled_out, compiled_lse

    def run_test_with_paged_attention(
        self,
        score_mod: Optional[Callable],
        dtype: torch.dtype,
        device,
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
        if device == "cpu" and dtype is torch.float16:
            dtype = torch.float32

        q = torch.randn(
            (Q_B, Q_H, Q_S, QK_D), dtype=dtype, device=device, requires_grad=False
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
            block_mask = create_block_mask(noop_mask, Q_B, 1, Q_S, KV_S, device=device)

        sdpa_partial = create_attention(score_mod, block_mask, enable_gqa=(Q_H != KV_H))
        golden_out, golden_lse = sdpa_partial(q_gold, k_gold, v_gold, return_lse=True)
        ref_out, ref_lse = sdpa_partial(q_ref, k_ref, v_ref, return_lse=True)

        compiled_out, compiled_lse = self.run_paged_attention(
            score_mod, q, k, v, dtype, device, block_mask
        )
        self._check_out(
            golden_out,
            ref_out,
            compiled_out,
            is_paged_attention=True,
        )
        requires_grad = device in DEVICE_SUPPORTS_BACKWARDS
        if requires_grad:
            self._check_out(
                golden_lse,
                ref_lse,
                compiled_lse,
                is_paged_attention=True,
            )

    def run_test_with_call(
        self,
        sdpa_call: Callable,
        dtype: torch.dtype,
        device: str,
        Q_B: int = B,
        Q_H: int = H,
        Q_S: int = S,
        Q_D: int = D,
        KV_B: int = B,
        KV_H: int = H,
        KV_S: int = S,
        V_D: int = D,
    ):
        if device == "cpu" and dtype is torch.float16:
            dtype = torch.float32

        requires_grad = device in DEVICE_SUPPORTS_BACKWARDS

        q = torch.randn(
            (Q_B, Q_H, Q_S, Q_D),
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
        k = torch.randn(
            (KV_B, KV_H, KV_S, Q_D),
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
        v = torch.randn(
            (KV_B, KV_H, KV_S, V_D),
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
        q_ref, k_ref, v_ref = query_key_value_clones(q, k, v)
        q_gold, k_gold, v_gold = query_key_value_clones(q, k, v, torch.float64)
        compiled_sdpa = torch.compile(sdpa_call)
        golden_out = sdpa_call(q_gold, k_gold, v_gold)
        ref_out = sdpa_call(q_ref, k_ref, v_ref)
        compiled_out = compiled_sdpa(q, k, v)
        if not requires_grad:
            self._check_out(
                golden_out,
                ref_out,
                compiled_out,
                is_paged_attention=False,
            )
        else:
            backward_grad = torch.randn(
                (Q_B, Q_H, Q_S, V_D), dtype=dtype, device=device
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
        dtype: torch.dtype,
        device,
        B: int = B,
        H: int = H,
        S: int = S,
        D: int = D,
    ):
        if device == "cpu" and dtype is torch.float16:
            dtype = torch.float32

        score_mod, mask_mod = score_mask_mod

        # First batch with original dimensions (B, H, S, D)
        block_mask1 = create_block_mask(mask_mod, 1, 1, S, S, device=device)
        sdpa_partial1 = create_attention(score_mod, block_mask=block_mask1)

        requires_grad = device in DEVICE_SUPPORTS_BACKWARDS

        q1 = torch.randn(
            (B, H, S, D),
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
        k1 = torch.randn(
            (B, H, S, D),
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
        v1 = torch.randn(
            (B, H, S, D),
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
        q1_ref, k1_ref, v1_ref = query_key_value_clones(q1, k1, v1)
        q1_gold, k1_gold, v1_gold = query_key_value_clones(q1, k1, v1, torch.float64)
        ref_out1 = sdpa_partial1(q1_ref, k1_ref, v1_ref)
        golden_out1 = sdpa_partial1(q1_gold, k1_gold, v1_gold)

        if requires_grad:
            backward_grad1 = torch.randn((B, H, S, D), dtype=dtype, device=device)
            golden_out1.backward(backward_grad1.to(torch.float64))
            ref_out1.backward(backward_grad1)

        # Second batch with modified dimensions (B * 2, H, S / 2, D)
        B = int(B * 2)
        S = int(S / 2)
        block_mask2 = create_block_mask(mask_mod, 1, 1, S, S, device=device)
        sdpa_partial2 = create_attention(score_mod, block_mask=block_mask2)

        q2 = torch.randn(
            (B, H, S, D),
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
        k2 = torch.randn(
            (B, H, S, D),
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
        v2 = torch.randn(
            (B, H, S, D),
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
        q2_ref, k2_ref, v2_ref = query_key_value_clones(q2, k2, v2)
        q2_gold, k2_gold, v2_gold = query_key_value_clones(q2, k2, v2, torch.float64)
        ref_out2 = sdpa_partial2(q2_ref, k2_ref, v2_ref)
        golden_out2 = sdpa_partial2(q2_gold, k2_gold, v2_gold)

        if requires_grad:
            backward_grad2 = torch.randn((B, H, S, D), dtype=dtype, device=device)
            golden_out2.backward(backward_grad2.to(torch.float64))
            ref_out2.backward(backward_grad2)

        # Third batch with modified dimensions (B * 2, H, S / 4, D)
        S = int(S / 2)
        block_mask3 = create_block_mask(mask_mod, 1, 1, S, S, device=device)
        sdpa_partial3 = create_attention(score_mod, block_mask=block_mask3)

        q3 = torch.randn(
            (B, H, S, D),
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
        k3 = torch.randn(
            (B, H, S, D),
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
        v3 = torch.randn(
            (B, H, S, D),
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
        q3_ref, k3_ref, v3_ref = query_key_value_clones(q3, k3, v3)
        q3_gold, k3_gold, v3_gold = query_key_value_clones(q3, k3, v3, torch.float64)
        ref_out3 = sdpa_partial3(q3_ref, k3_ref, v3_ref)
        golden_out3 = sdpa_partial3(q3_gold, k3_gold, v3_gold)

        if requires_grad:
            backward_grad3 = torch.randn((B, H, S, D), dtype=dtype, device=device)
            golden_out3.backward(backward_grad3.to(torch.float64))
            ref_out3.backward(backward_grad3)

        # Clear dynamo counters
        torch._dynamo.reset()

        # First compilation with original dimensions
        backend = torch._dynamo.testing.CompileCounterWithBackend("inductor")
        compiled_sdpa1 = torch.compile(sdpa_partial1, backend=backend, dynamic=True)
        compiled_out1 = compiled_sdpa1(q1, k1, v1)

        if requires_grad:
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
        else:
            self._check_out(golden_out1, ref_out1, compiled_out1)
        self.assertEqual(backend.frame_count, 1)

        # Second compilation with new dimensions
        compiled_sdpa2 = torch.compile(sdpa_partial2, backend=backend, dynamic=True)
        compiled_out2 = compiled_sdpa2(q2, k2, v2)

        if requires_grad:
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
        else:
            self._check_out(golden_out2, ref_out2, compiled_out2)
        self.assertEqual(backend.frame_count, 1)

        # Third compilation with new dimensions
        compiled_sdpa3 = torch.compile(sdpa_partial3, backend=backend, dynamic=True)
        compiled_out3 = compiled_sdpa3(q3, k3, v3)

        if requires_grad:
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
        else:
            self._check_out(golden_out3, ref_out3, compiled_out3)
        self.assertEqual(backend.frame_count, 1)

    def run_automatic_dynamic_test(
        self,
        score_mod: Callable,
        dtype: torch.dtype,
        device: str,
        B: int = B,
        H: int = H,
        S: int = S,
        D: int = D,
    ):
        if device == "cpu" and dtype is torch.float16:
            dtype = torch.float32

        block_mask1 = create_block_mask(noop_mask, 1, 1, S, S, device=device)
        sdpa_partial1 = create_attention(score_mod, block_mask=block_mask1)
        # The first eager batch, shape (B, H, S, D)
        requires_grad = device in DEVICE_SUPPORTS_BACKWARDS

        q1 = torch.randn(
            (B, H, S, D),
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
        k1 = torch.randn(
            (B, H, S, D),
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
        v1 = torch.randn(
            (B, H, S, D),
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
        golden_out1 = sdpa_partial1(
            q1.to(torch.float64), k1.to(torch.float64), v1.to(torch.float64)
        )
        ref_out1 = sdpa_partial1(q1, k1, v1)

        # The second eager batch, shape (B * 2, H, S / 2, D)
        B = int(B * 2)
        S = int(S / 2)
        block_mask2 = create_block_mask(noop_mask, 1, 1, S, S, device=device)
        sdpa_partial2 = create_attention(score_mod, block_mask=block_mask2)
        q2 = torch.randn(
            (B, H, S, D),
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
        k2 = torch.randn(
            (B, H, S, D),
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
        v2 = torch.randn(
            (B, H, S, D),
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
        golden_out2 = sdpa_partial2(
            q2.to(torch.float64), k2.to(torch.float64), v2.to(torch.float64)
        )
        ref_out2 = sdpa_partial2(q2, k2, v2)

        # The third eager batch, shape (B * 4, H, S / 4, D)
        B = int(B * 2)
        S = int(S / 2)
        block_mask3 = create_block_mask(noop_mask, 1, 1, S, S, device=device)
        sdpa_partial3 = create_attention(score_mod, block_mask=block_mask3)
        q3 = torch.randn(
            (B, H, S, D),
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
        k3 = torch.randn(
            (B, H, S, D),
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
        v3 = torch.randn(
            (B, H, S, D),
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
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
        compiled_out1 = torch.compile(sdpa_partial1, backend=backend, fullgraph=True)(
            q1, k1, v1
        )
        self._check_equal(golden_out1, ref_out1, compiled_out1, fudge_factor)
        self.assertEqual(backend.frame_count, 1)

        # The second batch (automatic dynamic).
        compiled_out2 = torch.compile(sdpa_partial2, backend=backend, fullgraph=True)(
            q2, k2, v2
        )
        self._check_equal(golden_out2, ref_out2, compiled_out2, fudge_factor)
        self.assertEqual(backend.frame_count, 2)

        # The third batch (no re-compilation).
        compiled_out3 = torch.compile(sdpa_partial3, backend=backend, fullgraph=True)(
            q3, k3, v3
        )
        self._check_equal(golden_out3, ref_out3, compiled_out3, fudge_factor)
        self.assertEqual(backend.frame_count, 2)

    @supported_platform
    @dtypes(*device_configs["cpu"].dtypes)
    @dtypesIfCUDA(*device_configs["cuda"].dtypes)
    @dtypesIfXPU(*device_configs["xpu"].dtypes)
    @common_utils.parametrize("score_mod", test_score_mods)
    def test_builtin_score_mods(self, device, dtype, score_mod: Callable):
        self.run_test(score_mod, dtype, device=device)
        self.run_test_with_paged_attention(score_mod, dtype, device=device)

    @running_on_a100_only
    @common_utils.parametrize("score_mod", test_score_mods)
    @dtypes(*device_configs["cpu"].dtypes_fast)
    @dtypesIfCUDA(*device_configs["cuda"].dtypes_fast)
    @dtypesIfXPU(*device_configs["xpu"].dtypes_fast)
    def test_builtin_score_mods_seqlen_lt_default_sparse_block_size(
        self, device, dtype, score_mod: Callable
    ):
        # _DEFAULT_SPARSE_BLOCK_SIZE is 128
        attention = functools.partial(
            flex_attention,
            score_mod=score_mod,
            kernel_options={"FORCE_USE_FLEX_ATTENTION": True},
        )
        self.run_test_with_call(attention, dtype, device, B, H, 64, D, B, H, 64, D)

    @running_on_a100_only
    @dtypes(*device_configs["cpu"].dtypes_fast)
    @dtypesIfCUDA(*device_configs["cuda"].dtypes_fast)
    @dtypesIfXPU(*device_configs["xpu"].dtypes_fast)
    @common_utils.parametrize("score_mod", test_score_mods)
    def test_builtin_score_mods_seqlen_lt_custom_sparse_block_size(
        self, device, dtype: torch.dtype, score_mod: Callable
    ):
        def causal_mask(b, h, q, kv):
            return q >= kv

        block_mask = create_block_mask(
            causal_mask, 1, 1, 64, 64, BLOCK_SIZE=256, device=device
        )
        attention = functools.partial(
            flex_attention,
            score_mod=score_mod,
            block_mask=block_mask,
            kernel_options={"FORCE_USE_FLEX_ATTENTION": True},
        )
        self.run_test_with_call(
            attention,
            dtype,
            device,
            B,
            H,
            64,
            D,
            B,
            H,
            64,
            D,
        )

    @supported_platform
    @dtypes(*device_configs["cpu"].dtypes_fast)
    @dtypesIfCUDA(*device_configs["cuda"].dtypes_fast)
    @dtypesIfXPU(*device_configs["xpu"].dtypes_fast)
    @common_utils.parametrize("score_mask_mod", test_score_mask_mod_map.items())
    def test_builtin_score_mods_dynamic(
        self, device, dtype: torch.dtype, score_mask_mod: tuple[Callable, Callable]
    ):
        self.run_dynamic_test(score_mask_mod, dtype, S=1024, device=device)

    @supported_platform
    @dtypes(*device_configs["cpu"].dtypes_fast)
    @dtypesIfCUDA(*device_configs["cuda"].dtypes_fast)
    @dtypesIfXPU(*device_configs["xpu"].dtypes_fast)
    @common_utils.parametrize("score_mod", test_score_mods)
    def test_builtin_score_mods_automatic_dynamic(
        self, device, dtype: torch.dtype, score_mod: Callable
    ):
        self.run_automatic_dynamic_test(score_mod, dtype, S=1024, device=device)

    @supported_platform
    @dtypes(*device_configs["cpu"].dtypes_fast)
    @dtypesIfCUDA(*device_configs["cuda"].dtypes_fast)
    @dtypesIfXPU(*device_configs["xpu"].dtypes_fast)
    @common_utils.parametrize("score_mod", test_score_mods)
    def test_builtin_score_mods_different_seqlen(
        self, device, dtype: torch.dtype, score_mod: Callable
    ):
        inputs = (
            score_mod,
            dtype,
            device,
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

    @supported_platform
    @dtypes(*device_configs["cpu"].dtypes)
    @dtypesIfCUDA(*device_configs["cuda"].dtypes)
    @dtypesIfXPU(*device_configs["xpu"].dtypes)
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
            noop_mask, B, H, S, S, BLOCK_SIZE=BLOCK_SIZE, device=device
        )
        self.run_test(score_mod, dtype, block_mask=block_mask, device=device)
        self.run_test_with_paged_attention(
            score_mod, dtype, block_mask=block_mask, device=device
        )

    @supported_platform
    @dtypes(*device_configs["cpu"].dtypes_fast)
    @dtypesIfCUDA(*device_configs["cuda"].dtypes_fast)
    @dtypesIfXPU(*device_configs["xpu"].dtypes_fast)
    @common_utils.parametrize("batch_dims", test_Bq_Bkv)
    @common_utils.parametrize("head_dims", test_Hq_Hkv)
    @common_utils.parametrize("score_mod", test_score_mods)
    def test_kv_batch_broadcast(
        self,
        device,
        dtype: torch.dtype,
        batch_dims: tuple[int, int],
        head_dims: tuple[int, int],
        score_mod: Callable,
    ):
        Hq, Hkv = head_dims
        assert Hq % Hkv == 0

        Bq, Bkv = batch_dims
        assert Bq > 1 and Bkv == 1

        block_mask = create_block_mask(noop_mask, Bq, 1, S, S, device=device)

        self.run_test(
            score_mod, dtype, device, Bq, Hq, S, D, Bkv, Hkv, S, D, block_mask
        )

    @supported_platform
    @skip_on_cpu
    def test_small_block_mask(self, device):
        compiled_create_block_mask = torch.compile(create_block_mask)

        def create_block_mask_from_seqlens(
            q_batch: torch.Tensor,
            kv_batch: torch.Tensor,
        ) -> BlockMask:
            B, H = None, None
            Q_LEN = q_batch.size(0)
            KV_LEN = kv_batch.size(0)

            def batch_mask_mod(
                b: torch.Tensor,
                h: torch.Tensor,
                q_idx: torch.Tensor,
                kv_idx: torch.Tensor,
            ):
                q_idx_batch = q_batch[q_idx]
                kv_idx_batch = kv_batch[kv_idx]
                batch_mask = (
                    (q_idx_batch == kv_idx_batch)
                    & (q_idx_batch != -1)
                    & (kv_idx_batch != -1)
                )

                return batch_mask

            return compiled_create_block_mask(
                batch_mask_mod,
                B=B,
                H=H,
                Q_LEN=Q_LEN,
                KV_LEN=KV_LEN,
                device=device,
            )

        a = torch.tensor([2, 42, 18, 21, 4, 2, 7, 1, 1], device=device)
        b = torch.tensor([57, 21, 16, 8], device=device)

        for seqlen in [a, b]:
            create_block_mask_from_seqlens(seqlen, seqlen)

    @supported_platform
    @dtypes(*device_configs["cpu"].dtypes_fast)
    @dtypesIfCUDA(*device_configs["cuda"].dtypes_fast)
    @dtypesIfXPU(*device_configs["xpu"].dtypes_fast)
    @common_utils.parametrize("batch_dims", test_Bq_Bkv)
    @common_utils.parametrize("head_dims", test_Hq_Hkv)
    @common_utils.parametrize("score_mod", test_score_mods)
    def test_kv_batch_broadcast_causal_mask(
        self,
        device,
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

        block_mask = create_block_mask(mask_mod, Bq, 1, S, S, device=device)
        attention = functools.partial(
            flex_attention, block_mask=block_mask, enable_gqa=(Hq != Hkv)
        )

        self.run_test_with_call(attention, dtype, device, Bq, Hq, S, D, Bkv, Hkv, S, D)

    @supported_platform
    @dtypes(*device_configs["cpu"].dtypes_fast)
    @dtypesIfCUDA(*device_configs["cuda"].dtypes_fast)
    @dtypesIfXPU(*device_configs["xpu"].dtypes_fast)
    @common_utils.parametrize("score_mod", test_score_mods)
    @skip_on_rocm  # TODO: NaNs on ROCM
    @skip_on_xpu  # TODO: NaNs on XPU like ROCM, need another PR to fix.
    def test_GQA(self, device, dtype: torch.dtype, score_mod: Callable):
        inputs = (
            score_mod,
            dtype,
            device,
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

    @supported_platform
    @dtypes(*device_configs["cpu"].dtypes_fast)
    @dtypesIfCUDA(*device_configs["cuda"].dtypes_fast)
    @dtypesIfXPU(*device_configs["xpu"].dtypes_fast)
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
    def test_strided_inputs(self, device, dtype: torch.dtype, q_s, k_s, v_s, do_s):
        q1 = torch.randn((B * H * S * D * 2), dtype=dtype, device=device)
        k1 = torch.randn((B * H * S * D * 2), dtype=dtype, device=device)
        v1 = torch.randn((B * H * S * D * 2), dtype=dtype, device=device)
        do1 = torch.randn((B * H * S * D * 2), dtype=dtype, device=device)

        q_shape = (B, H, S // 2, D)
        k_shape = (B, H, S, D)
        v_shape = (B, H, S, D)
        do_shape = (B, H, S // 2, D)

        requires_grad = device in DEVICE_SUPPORTS_BACKWARDS

        def coerce_to_strides(val, shape, strides):
            strides, offset = strides
            val_max = [x * (y - 1) for x, y in zip(strides, shape)]
            assert sum(val_max) + offset < B * H * S * D * 2
            assert strides[-1] == 1
            return torch.as_strided(val, shape, strides, offset).requires_grad_(
                requires_grad
            )

        q = coerce_to_strides(q1, q_shape, q_s)
        k = coerce_to_strides(k1, k_shape, k_s)
        v = coerce_to_strides(v1, v_shape, v_s)
        do = coerce_to_strides(do1, do_shape, do_s)

        kernel_options = {"USE_TMA": True}

        block_mask = _create_empty_block_mask(q, k)
        score_mod = _generate_alibi_bias(8)
        sdpa_partial = create_attention(
            score_mod=score_mod, block_mask=block_mask, kernel_options=kernel_options
        )
        compiled_sdpa = torch.compile(sdpa_partial, fullgraph=True)
        ref_out = sdpa_partial(q, k, v)
        compiled_out = compiled_sdpa(q, k, v)

        tolerance = Tolerances(atol=2e-1, rtol=2e-1)
        torch.testing.assert_close(
            ref_out, compiled_out, atol=tolerance.atol, rtol=tolerance.rtol
        )
        if requires_grad:
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
                compiled_grads[0],
                ref_grads[0],
                atol=tolerance.atol,
                rtol=tolerance.rtol,
            )
            torch.testing.assert_close(
                compiled_grads[1],
                ref_grads[1],
                atol=tolerance.atol,
                rtol=tolerance.rtol,
            )
            torch.testing.assert_close(
                compiled_grads[2],
                ref_grads[2],
                atol=tolerance.atol,
                rtol=tolerance.rtol,
            )

        # test paged attention which does not support backward
        q.requires_grad, k.requires_grad, v.requires_grad = False, False, False
        paged_compiled_out, _ = self.run_paged_attention(
            score_mod, q, k, v, dtype, device=device, kernel_options=kernel_options
        )
        torch.testing.assert_close(
            ref_out, paged_compiled_out, atol=tolerance.atol, rtol=tolerance.rtol
        )

    @supported_platform
    def test_doc_mask_sparse(self, device):
        document_id = torch.zeros(S, dtype=torch.int, device=device)
        for i in range(0, S, 256):
            document_id[i : i + 256] = i // 256

        def document_masking_causal(score, b, h, q_idx, kv_idx):
            causal_mask = q_idx >= kv_idx
            document_mask = document_id[q_idx] == document_id[kv_idx]
            return torch.where(causal_mask & document_mask, score, -float("inf"))

        self.run_test(document_masking_causal, torch.float16, device=device)
        self.run_test_with_paged_attention(
            document_masking_causal, torch.float16, device=device
        )

    @supported_platform
    def test_index_multiple(self, device):
        bias = torch.randn(B, S, device=device)

        def index_multiple(score, b, h, q_idx, kv_idx):
            return score + bias[b][q_idx]

        self.run_test(index_multiple, torch.float16, device=device)
        self.run_test_with_paged_attention(index_multiple, torch.float16, device=device)

    @supported_platform
    def test_index_weird1(self, device):
        bias = torch.randn(4, B, H, S, device=device)

        def index_weird1(score, b, h, q_idx, kv_idx):
            return score + bias[0][b, h][q_idx]

        self.run_test(index_weird1, torch.float16, device=device)
        self.run_test_with_paged_attention(index_weird1, torch.float16, device=device)

    @supported_platform
    def test_index_weird2(self, device):
        bias = torch.randn(B, H, 4, S, device=device)
        which_bias = torch.tensor(0, device=device)

        def index_weird2(score, b, h, q_idx, kv_idx):
            return score + bias[b][h][which_bias, q_idx]

        self.run_test(index_weird2, torch.float16, device=device)
        self.run_test_with_paged_attention(index_weird2, torch.float16, device=device)

    @supported_platform
    @dtypes(*device_configs["cpu"].dtypes)
    @dtypesIfCUDA(*device_configs["cuda"].dtypes)
    @dtypesIfXPU(*device_configs["xpu"].dtypes)
    def test_skip_odd_keys(self, device, dtype: torch.dtype):
        def score_mod(score, b, h, q, kv):
            return torch.where(kv % 2 == 0, score, float("-inf"))

        self.run_test(score_mod, dtype, device=device)
        self.run_test_with_paged_attention(score_mod, dtype, device=device)

    @supported_platform
    @dtypes(*device_configs["cpu"].dtypes)
    @dtypesIfCUDA(*device_configs["cuda"].dtypes)
    @dtypesIfXPU(*device_configs["xpu"].dtypes)
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
    @dtypes(*device_configs["cpu"].dtypes)
    @dtypesIfCUDA(*device_configs["cuda"].dtypes)
    @dtypesIfXPU(*device_configs["xpu"].dtypes)
    def test_captured_buffers_all_dims(self, device, dtype: torch.dtype):
        head_scale = torch.randn(H, device=device)
        batch_scale = torch.randn(B, device=device)
        tok_scale = torch.randn(S, device=device)

        def all_bias(score, batch, head, token_q, token_kv):
            score = score + tok_scale[token_q]
            score = score + batch_scale[batch]
            score = score + head_scale[head]
            return score

        self.run_test(all_bias, dtype, device=device)
        self.run_test_with_paged_attention(all_bias, dtype, device=device)

    @supported_platform
    @dtypes(*device_configs["cpu"].dtypes_fast)
    @dtypesIfCUDA(*device_configs["cuda"].dtypes_fast)
    @dtypesIfXPU(*device_configs["xpu"].dtypes_fast)
    def test_seq_masking(self, device, dtype):
        seq_idx = torch.zeros(S, device=device, dtype=torch.bool)
        seq_idx[S // 2 :] = 1

        def seq_mask_mod(score, b, h, q, kv):
            return torch.where(seq_idx[q] == seq_idx[kv], score, float("-inf"))

        self.run_test(seq_mask_mod, dtype, device=device)
        self.run_test_with_paged_attention(seq_mask_mod, dtype, device=device)

    @supported_platform
    @dtypes(*device_configs["cpu"].dtypes_fast)
    @dtypesIfCUDA(*device_configs["cuda"].dtypes_fast)
    @dtypesIfXPU(*device_configs["xpu"].dtypes_fast)
    def test_load_from_bias_seq_only(self, device, dtype):
        bias = torch.randn(S, S, device=device, dtype=dtype)

        def bias_mod(score, b, h, q, kv):
            return score + bias[q, kv]

        self.run_test(bias_mod, dtype, device=device)
        self.run_test_with_paged_attention(bias_mod, dtype, device=device)

    @supported_platform
    @dtypes(*device_configs["cpu"].dtypes_fast)
    @dtypesIfCUDA(*device_configs["cuda"].dtypes_fast)
    @dtypesIfXPU(*device_configs["xpu"].dtypes_fast)
    def test_load_from_bias_seq_batch(self, device, dtype):
        bias = torch.randn(B, S, S, device=device, dtype=dtype)

        def bias_mod(score, b, h, q, kv):
            return score + bias[b, q, kv]

        self.run_test(bias_mod, dtype, device=device)
        self.run_test_with_paged_attention(bias_mod, dtype, device=device)

    @supported_platform
    @skip_on_cpu
    def test_load_from_view_buffer(self, device):
        dtype = torch.float16
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
    @dtypes(*device_configs["cpu"].dtypes_fast)
    @dtypesIfCUDA(*device_configs["cuda"].dtypes_fast)
    @dtypesIfXPU(*device_configs["xpu"].dtypes_fast)
    def test_load_from_bias_head_seq_batch(self, device, dtype):
        bias = torch.randn(B, H, S, S, device=device, dtype=dtype)

        def bias_mod(score, b, h, q, kv):
            return score + bias[b, h, q, kv]

        self.run_test(bias_mod, dtype, device=device)
        self.run_test_with_paged_attention(bias_mod, dtype, device=device)

    @supported_platform
    @dtypes(*device_configs["cpu"].dtypes_fast)
    @dtypesIfCUDA(*device_configs["cuda"].dtypes_fast)
    @dtypesIfXPU(*device_configs["xpu"].dtypes_fast)
    def test_load_rel_bias(self, device, dtype):
        rel_bias = torch.randn(2 * S, device=device, dtype=dtype)

        def bias_mod(score, b, h, q, kv):
            return score + rel_bias[(q - kv) + S]

        self.run_test(bias_mod, dtype, device=device)
        self.run_test_with_paged_attention(bias_mod, dtype, device=device)

    @supported_platform
    @dtypes(*device_configs["cpu"].dtypes_fast)
    @dtypesIfCUDA(*device_configs["cuda"].dtypes_fast)
    @dtypesIfXPU(*device_configs["xpu"].dtypes_fast)
    def test_dependent_causal_bidirectional(self, device, dtype):
        num_bidirectional = torch.randint(0, S, (B,), device=device, dtype=torch.int32)

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

        self.run_test(bias_mod, dtype, device=device)
        self.run_test_with_paged_attention(bias_mod, dtype, device=device)

    @supported_platform
    @dtypes(*device_configs["cpu"].dtypes_fast)
    @dtypesIfCUDA(*device_configs["cuda"].dtypes_fast)
    @dtypesIfXPU(*device_configs["xpu"].dtypes_fast)
    def test_natten_2d(self, device, dtype):
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

        self.run_test(natten_mask, dtype, device=device)
        self.run_test_with_paged_attention(natten_mask, dtype, device=device)

    @supported_platform
    def test_subgraph_respect_decompostion(self, device):
        from torch._decomp import core_aten_decompositions
        from torch.fx.experimental.proxy_tensor import make_fx

        def score_mod_func(score, b, h, q, kv):
            return score - q // (1 + kv)

        make_tensor = functools.partial(
            torch.randn,
            (2, 2, 128, 4),
            device=device,
            dtype=torch.float64,
            requires_grad=True,
        )
        query, key, value = make_tensor(), make_tensor(), make_tensor()
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
    @dtypes(*device_configs["cpu"].dtypes_fast)
    @dtypesIfCUDA(*device_configs["cuda"].dtypes_fast)
    @dtypesIfXPU(*device_configs["xpu"].dtypes_fast)
    def test_silu_on_score(self, device, dtype):
        def silu_score(score, b, h, q, kv):
            return torch.nn.functional.silu(score)

        self.run_test(silu_score, dtype, device=device)
        self.run_test_with_paged_attention(silu_score, dtype, device=device)

    @supported_platform
    @dtypes(*device_configs["cpu"].dtypes_fast)
    @dtypesIfCUDA(*device_configs["cuda"].dtypes_fast)
    @dtypesIfXPU(*device_configs["xpu"].dtypes_fast)
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

    @supported_platform
    @dtypes(*device_configs["cpu"].dtypes_fast)
    @dtypesIfCUDA(*device_configs["cuda"].dtypes_fast)
    @dtypesIfXPU(*device_configs["xpu"].dtypes_fast)
    def test_captured_scale(self, device, dtype):
        scale = torch.ones((), device=device, dtype=torch.int32)

        def score_mod_scale(qk, b, h, q, kv):
            return qk + scale

        self.run_test(score_mod_scale, dtype, device=device)
        self.run_test_with_paged_attention(score_mod_scale, dtype, device=device)

    @supported_platform
    @dtypes(*device_configs["cpu"].dtypes_fast)
    @dtypesIfCUDA(*device_configs["cuda"].dtypes_fast)
    @dtypesIfXPU(*device_configs["xpu"].dtypes_fast)
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
    @expectedFailure  # If we capture a tensor then we can perform a reduction on it, and that shouldn't be allowed
    @dtypes(*device_configs["cpu"].dtypes_fast)
    @dtypesIfCUDA(*device_configs["cuda"].dtypes_fast)
    @dtypesIfXPU(*device_configs["xpu"].dtypes_fast)
    def test_captured_reduction(self, device, dtype):
        scale = torch.randn((B, 8), device=device)

        def score_mod_scale(qk, b, h, q, kv):
            return qk + scale[b].sum(dim=-1)

        self.run_test(score_mod_scale, dtype, device=device)

    @supported_platform
    @skip_on_cpu
    @dtypes(torch.float16)
    @dtypesIfCUDA(torch.float16)
    def test_dynamic_captured_buffer(self, device, dtype):
        def run_with_head_count(compiled_fa, head_count):
            head_scale = torch.randn(
                head_count, device=device, dtype=dtype, requires_grad=True
            )

            def score_mod(score, batch, head, token_q, token_kv):
                return score * head_scale[head]

            q = torch.randn(
                B, head_count, S, D, device=device, dtype=dtype, requires_grad=True
            )
            k = torch.randn_like(q, requires_grad=True)
            v = torch.randn_like(q, requires_grad=True)

            block_mask = create_block_mask(noop_mask, B, 1, S, S, device=device)

            out = compiled_fa(q, k, v, score_mod=score_mod, block_mask=block_mask)
            loss = out.sum()
            loss.backward()
            return out

        compiled_fa = torch.compile(flex_attention, fullgraph=True, dynamic=True)

        head_counts = [4, 8, 4, 16, 4]
        for head_count in head_counts:
            run_with_head_count(compiled_fa, head_count)

    @supported_platform
    @dtypes(*device_configs["cpu"].dtypes_fast)
    @dtypesIfCUDA(*device_configs["cuda"].dtypes_fast)
    @dtypesIfXPU(*device_configs["xpu"].dtypes_fast)
    @common_utils.parametrize(
        "score_mod", test_score_mods, name_fn=lambda score_mod: score_mod.__name__
    )
    @skip_on_cpu
    def test_return_max(self, device, dtype, score_mod):
        make_tensor = functools.partial(
            torch.randn,
            (2, 2, 243, 16),
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        query, key, value = make_tensor(), make_tensor(), make_tensor()

        out_only = flex_attention(query, key, value, score_mod)
        out_max, aux_max = flex_attention(
            query,
            key,
            value,
            score_mod,
            return_aux=AuxRequest(max_scores=True),
        )
        out_both, aux_both = flex_attention(
            query,
            key,
            value,
            score_mod,
            return_aux=AuxRequest(lse=True, max_scores=True),
        )

        flex_compile = torch.compile(flex_attention, fullgraph=True)
        out_compiled, aux_compiled = flex_compile(
            query,
            key,
            value,
            score_mod,
            return_aux=AuxRequest(max_scores=True),
        )

        torch.testing.assert_close(out_only, out_max, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(out_only, out_both, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(
            aux_max.max_scores, aux_both.max_scores, atol=1e-6, rtol=1e-6
        )

        # we are calculating slightly different scores so add a lil fudge
        # Extra tolerance for squared score_mod with float16 due to limited dynamic range
        if score_mod.__name__ == "_squared" and dtype == torch.float16:
            atol, rtol = 2e-2, 2e-2
        else:
            atol, rtol = 5e-3, 5e-3

        torch.testing.assert_close(out_max, out_compiled, atol=atol, rtol=rtol)
        torch.testing.assert_close(
            aux_max.max_scores, aux_compiled.max_scores, atol=atol, rtol=rtol
        )

        B, H, L = query.shape[:3]
        self.assertEqual(aux_max.max_scores.shape, (B, H, L))

        max_score_tensors = [
            aux_max.max_scores,
            aux_both.max_scores,
            aux_compiled.max_scores,
        ]
        for max_tensor in max_score_tensors:
            self.assertFalse(
                max_tensor.requires_grad, "max_scores should not require gradients"
            )
            self.assertEqual(
                max_tensor.dtype, torch.float32, "max_scores should be kept in fp32"
            )

        # Test gradient computation for both eager and compiled versions
        test_cases = [
            ("eager", out_max, "eager mode"),
            ("compiled", out_compiled, "compiled mode"),
        ]

        for mode_name, output, description in test_cases:
            loss = output.sum()
            grads = torch.autograd.grad(loss, (query, key, value))

            # Verify gradients are computed for all inputs
            input_names = ["query", "key", "value"]
            for grad, input_name in zip(grads, input_names):
                self.assertIsNotNone(
                    grad, f"{input_name} should receive gradients in {description}"
                )

    @supported_platform
    @dtypes(*device_configs["cpu"].dtypes_fast)
    @dtypesIfCUDA(*device_configs["cuda"].dtypes_fast)
    @dtypesIfXPU(*device_configs["xpu"].dtypes_fast)
    @common_utils.parametrize(
        "score_mod", test_score_mods, name_fn=lambda score_mod: score_mod.__name__
    )
    @skip_on_cpu
    def test_return_aux(self, device, dtype, score_mod):
        """Test the new return_aux API with AuxRequest/Output"""
        make_tensor = functools.partial(
            torch.randn,
            (2, 2, 243, 16),
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        query, key, value = make_tensor(), make_tensor(), make_tensor()

        flex_compile = torch.compile(flex_attention, fullgraph=True)
        flex_compile_partial = torch.compile(flex_attention, fullgraph=False)

        # Test 1: No auxiliary outputs (default behavior)
        out_only = flex_compile(query, key, value, score_mod)
        self.assertIsInstance(out_only, torch.Tensor)

        # Test 2: Request only LSE
        out, aux_lse = flex_compile(
            query, key, value, score_mod, return_aux=AuxRequest(lse=True)
        )
        self.assertIsInstance(aux_lse, AuxOutput)
        self.assertIsInstance(aux_lse.lse, torch.Tensor)
        self.assertIsNone(aux_lse.max_scores)
        self.assertEqual(aux_lse.lse.shape, (2, 2, 243))
        self.assertEqual(aux_lse.lse.dtype, torch.float32)

        # Test 3: Request only max_scores
        out, aux_max = flex_compile(
            query,
            key,
            value,
            score_mod,
            return_aux=AuxRequest(max_scores=True),
        )
        self.assertIsInstance(aux_max, AuxOutput)
        self.assertIsNone(aux_max.lse)
        self.assertIsInstance(aux_max.max_scores, torch.Tensor)
        self.assertEqual(aux_max.max_scores.shape, (2, 2, 243))
        self.assertEqual(aux_max.max_scores.dtype, torch.float32)

        # Test 4: Request both auxiliary outputs
        out, aux_both = flex_compile(
            query,
            key,
            value,
            score_mod,
            return_aux=AuxRequest(lse=True, max_scores=True),
        )
        self.assertIsInstance(aux_both, AuxOutput)
        self.assertIsInstance(aux_both.lse, torch.Tensor)
        self.assertIsInstance(aux_both.max_scores, torch.Tensor)
        self.assertEqual(aux_both.lse.shape, (2, 2, 243))
        self.assertEqual(aux_both.max_scores.shape, (2, 2, 243))

        # Test 5: Request no auxiliary outputs explicitly
        out, aux_none = flex_compile(
            query,
            key,
            value,
            score_mod,
            return_aux=AuxRequest(),  # Default is lse=False, max_scores=False
        )
        self.assertIsInstance(aux_none, AuxOutput)
        self.assertIsNone(aux_none.lse)
        self.assertIsNone(aux_none.max_scores)

        # Test 6: Verify outputs are consistent with legacy API, can't fullgraph through warnings
        out_legacy, lse_legacy = flex_compile_partial(
            query, key, value, score_mod, return_lse=True
        )
        torch.testing.assert_close(out_only, out_legacy, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(aux_lse.lse, lse_legacy, atol=1e-6, rtol=1e-6)

    @supported_platform
    @dtypes(*device_configs["cpu"].dtypes_fast)
    @dtypesIfCUDA(*device_configs["cuda"].dtypes_fast)
    @dtypesIfXPU(*device_configs["xpu"].dtypes_fast)
    @skip_on_cpu
    def test_return_aux_deprecation_warnings(self, device, dtype):
        """Test that deprecation warnings are issued for legacy parameters"""
        import warnings

        make_tensor = functools.partial(
            torch.randn,
            (2, 2, 64, 16),
            device=device,
            dtype=dtype,
        )
        query, key, value = make_tensor(), make_tensor(), make_tensor()

        # Clear shown warnings to ensure we can test them
        original_shown = _WARNINGS_SHOWN.copy()
        _WARNINGS_SHOWN.clear()

        try:
            # Test deprecation warning for return_lse
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                flex_attention(query, key, value, return_lse=True)
                self.assertTrue(
                    any(
                        "return_lse is deprecated" in str(warning.message)
                        for warning in w
                    )
                )

            # Clear for next test
            _WARNINGS_SHOWN.clear()

            # Test error when both old and new API are used
            with self.assertRaises(ValueError) as cm:
                flex_attention(
                    query,
                    key,
                    value,
                    return_lse=True,
                    return_aux=AuxRequest(lse=True),
                )
            self.assertIn(
                "Cannot specify both return_lse and return_aux", str(cm.exception)
            )

        finally:
            # Restore original warnings state
            _WARNINGS_SHOWN.clear()
            _WARNINGS_SHOWN.update(original_shown)

    @supported_platform
    @dtypes(*device_configs["cpu"].dtypes_fast)
    @dtypesIfCUDA(*device_configs["cuda"].dtypes_fast)
    @dtypesIfXPU(*device_configs["xpu"].dtypes_fast)
    @skip_on_cpu
    def test_dynamic_divisibility_guards(self, device, dtype):
        """Test guards for divisible/non-divisible shape transitions"""
        if device == "cpu" and dtype is torch.float16:
            dtype = torch.float32

        def score_mod(qk, b, h, q, kv):
            return torch.where(q >= kv, qk, -float("inf"))

        def test_shape(S, backend):
            """Test a single shape configuration"""
            block_mask = create_block_mask(noop_mask, 1, 1, S, S, device=device)
            sdpa_partial = create_attention(score_mod, block_mask=block_mask)

            tensors = [
                torch.randn(
                    2, 4, S, 64, dtype=dtype, device=device, requires_grad=False
                )
                for _ in range(3)
            ]

            compiled_sdpa = torch.compile(sdpa_partial, backend=backend)
            out, code = run_and_get_code(compiled_sdpa, *tensors)

            # Check divisibility flag
            is_divisible = S % 128 == 0
            expected_flag = f"IS_DIVISIBLE : tl.constexpr = {is_divisible}"
            self.assertIn(
                expected_flag, str(code), f"S={S} should have {expected_flag}"
            )

            self.assertEqual(out.shape, (2, 4, S, 64))
            return out, code

        torch._dynamo.reset()
        backend = CompileCounterWithBackend("inductor")

        # Test divisible and non-divisible shapes
        test_shapes = [256, 255, 383, 384]
        _ = [test_shape(S, backend) for S in test_shapes]

    @supported_platform
    @skip_on_cpu
    def test_mask_mod_handles_symint_addition(self, device):
        dtype = torch.float16

        def run(q, k, v):
            ql = q.size(-2)
            kl = k.size(-2)
            frame = 32

            def _opaque_mask(b, h, q_idx, kv_idx):
                ref = ql // frame
                mot = kl // frame  # codespell:ignore
                limit = (ref + mot) * frame  # codespell:ignore
                return q_idx < limit

            block_mask = create_block_mask(
                _opaque_mask,
                B=q.size(0),
                H=q.size(1),
                Q_LEN=ql,
                KV_LEN=kl,
                device=device,
            )
            return flex_attention(q, k, v, block_mask=block_mask)

        compiled_run = torch.compile(run, fullgraph=True, dynamic=True)

        q = torch.randn(1, 2, 192, 32, device=device, dtype=dtype)
        k = torch.randn(1, 2, 128, 32, device=device, dtype=dtype)
        v = torch.randn(1, 2, 128, 32, device=device, dtype=dtype)

        eager_out = run(q, k, v)
        compiled_out = compiled_run(q, k, v)
        torch.testing.assert_close(eager_out, compiled_out, atol=1e-3, rtol=1e-3)

        # Exercise different dynamic shapes to ensure SymInt sums remain well-formed.
        q2 = torch.randn(1, 2, 160, 32, device=device, dtype=dtype)
        k2 = torch.randn(1, 2, 96, 32, device=device, dtype=dtype)
        v2 = torch.randn(1, 2, 96, 32, device=device, dtype=dtype)

        eager_out2 = run(q2, k2, v2)
        compiled_out2 = compiled_run(q2, k2, v2)
        torch.testing.assert_close(eager_out2, compiled_out2, atol=1e-3, rtol=1e-3)

    @supported_platform
    def test_multiple_score_mod_calls(self, device):
        query = torch.randn((1, 8, 1024, 64), dtype=torch.float32, device=device)
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
    @skip_on_cpu
    @skip_on_rocm  # TODO: Investigate
    def test_multiple_mask_calls(self, device):
        make_tensor = functools.partial(
            torch.randn,
            (1, 4, 512, 64),
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )
        query, key, value = make_tensor(), make_tensor(), make_tensor()

        window_size = 32

        def causal_mask(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx

        def causal_mask_slidewindow_mod(b, h, q_idx, kv_idx):
            return (q_idx >= kv_idx) & (q_idx <= kv_idx + window_size)

        mask1 = create_block_mask(
            causal_mask, 1, None, 512, 512, _compile=False, device=device
        )
        mask2 = create_block_mask(
            causal_mask_slidewindow_mod,
            1,
            None,
            512,
            512,
            _compile=False,
            device=device,
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
    def test_multiple_score_mod_calls2(self, device):
        query = torch.randn((1, 8, 1024, 64), dtype=torch.float32, device=device)
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
        out2 = torch.compile(f, fullgraph=True)(query, *keys, *values)
        self.assertTrue((out - out2).abs().mean() < 1e-2)

    @supported_platform
    def test_multiple_score_mod_calls_paged_attention(self, device):
        query = torch.randn((1, 8, 1024, 64), dtype=torch.float32, device=device)
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

        eager_out = f(query, *keys, *values)

        block_mask = create_block_mask(noop_mask, 1, 1, 1024, 1024, device=device)

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

        compiled_out = torch.compile(paged_f, fullgraph=True)(
            query, k_cache1, k_cache2, v_cache1, v_cache2
        )
        tolerance = Tolerances(atol=2e-1, rtol=2e-1)
        torch.testing.assert_close(
            eager_out, compiled_out, atol=tolerance.atol, rtol=tolerance.rtol
        )

    @supported_platform
    def test_multiple_score_mod_calls2_paged_attention(self, device):
        query = torch.randn((1, 8, 1024, 64), dtype=torch.float32, device=device)
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

        eager_out = f(query, *keys, *values)

        block_mask = create_block_mask(noop_mask, 1, 1, 1024, 1024, device=device)
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

        compiled_out = torch.compile(paged_f, fullgraph=True)(
            query, k_cache1, k_cache2, k_cache3, v_cache1, v_cache2, v_cache3
        )
        tolerance = Tolerances(atol=2e-1, rtol=2e-1)
        torch.testing.assert_close(
            eager_out, compiled_out, atol=tolerance.atol, rtol=tolerance.rtol
        )

    @supported_platform
    @skip_on_cpu
    def test_inputs_are_realized(self, device):
        def f(q, k, v):
            x = torch.randn(1024, device=device)
            x = x * 2

            def func(qk, b, h, q, kv):
                return qk + x[q]

            return flex_attention(q.sin(), k, v, score_mod=func).cos()

        q, k, v = (
            torch.randn(1, 8, 1024, 64, device=device, requires_grad=True)
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
    @skip_on_cpu
    def test_make_block_mask(self, device):
        def causal_mask(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx

        block_mask_a = torch.compile(create_block_mask, fullgraph=True)(
            causal_mask, 1, 1, 512, 512, device=device
        )
        block_mask_b = create_block_mask(causal_mask, 1, 1, 512, 512, device=device)
        self.assertEqual(block_mask_a.kv_num_blocks, block_mask_b.kv_num_blocks)
        self.assertEqual(block_mask_a.kv_indices, block_mask_b.kv_indices)
        self.assertEqual(block_mask_a.q_num_blocks, block_mask_b.q_num_blocks)

    @supported_platform
    def test_mask_mod_combiners(self, device):
        def causal_mask(b, h, q, kv):
            return q >= kv

        def neg_causal_mask(b, h, q, kv):
            return q < kv

        def sliding_window(b, h, q, kv):
            return (q - kv) <= 512

        local_s = 2048
        block_mask = create_block_mask(
            and_masks(causal_mask, sliding_window),
            1,
            1,
            local_s,
            local_s,
            device=device,
        )
        self.assertExpectedInline(block_mask.kv_num_blocks.sum().item(), """28""")
        attention = functools.partial(flex_attention, block_mask=block_mask)
        self.run_test_with_call(
            attention, Q_S=local_s, KV_S=local_s, dtype=torch.float16, device=device
        )

        block_mask = create_block_mask(
            and_masks(causal_mask, neg_causal_mask),
            1,
            1,
            local_s,
            local_s,
            device=device,
        )
        self.assertEqual(block_mask.kv_num_blocks.sum(), 0)

        block_mask1 = create_block_mask(
            or_masks(causal_mask, neg_causal_mask),
            1,
            1,
            local_s,
            local_s,
            device=device,
        )
        block_mask2 = create_block_mask(
            noop_mask, 1, 1, local_s, local_s, device=device
        )
        self.assertEqual(block_mask1.sparsity(), block_mask2.sparsity())

    @supported_platform
    @skip_on_cpu
    def test_epilogue_fused(self, device):
        # set so that metrics appear
        torch._logging.set_logs(inductor_metrics=True)

        @torch.compile
        def f(q, k, v):
            out = flex_attention(q, k, v)
            return out.cos()

        q, k, v = (torch.randn(1, 8, 1024, 64, device=device) for _ in range(3))
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
        torch._logging.set_logs()

    @supported_platform
    @dtypes(*device_configs["cpu"].dtypes)
    @dtypesIfCUDA(*device_configs["cuda"].dtypes)
    @dtypesIfXPU(*device_configs["xpu"].dtypes)
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
    def test_mixed_dtypes_fails(self, device):
        query = torch.randn((1, 1, 1024, 64), dtype=torch.float32, device=device)
        key = torch.randn((1, 1, 1024, 64), dtype=torch.float16, device=device)
        value = torch.randn((1, 1, 1024, 64), dtype=torch.float16, device=device)
        with self.assertRaisesRegex(
            ValueError, "Expected query, key, and value to have the same dtype"
        ):
            flex_attention(query, key, value, _identity)

    @supported_platform
    @patch.object(torch._inductor.config, "max_autotune", True)
    def test_max_autotune(self, device):
        def score_mod(score, b, h, m, n):
            return score * 2

        self.run_test(score_mod, dtype=torch.float16, device=device)
        self.run_test_with_paged_attention(
            score_mod, dtype=torch.float16, device=device
        )
        self.run_test_with_paged_attention(
            score_mod=score_mod,
            dtype=torch.bfloat16,
            KV_S=64,
            device=device,
        )

    @supported_platform
    @skip("TODO: Figure out why this is erroring")
    @patch.object(torch._inductor.config, "max_autotune", True)
    def test_max_autotune_with_captured(self, device):
        head_scale = torch.randn(H, device=device)
        batch_scale = torch.randn(B, device=device)
        tok_scale = torch.randn(S, device=device)

        def bias_mod(score, batch, head, token_q, token_kv):
            score = score + tok_scale[token_q]
            score = score + batch_scale[batch]
            score = score + head_scale[head]
            return score

        self.run_test(bias_mod, dtype=torch.float32, device=device)

    @supported_platform
    @common_utils.parametrize("score_mod", test_score_mods)
    @dtypes(*device_configs["cpu"].dtypes)
    @dtypesIfCUDA(*device_configs["cuda"].dtypes)
    @dtypesIfXPU(*device_configs["xpu"].dtypes)
    @common_utils.parametrize("head_dims", [(D, D // 2), (D // 2, D)])
    def test_non_equal_head_dims(self, device, dtype, score_mod, head_dims):
        qk_d, v_d = head_dims
        self.run_test(score_mod, dtype, device, B, H, S, qk_d, B, H, S, V_D=v_d)
        self.run_test_with_paged_attention(
            score_mod, dtype, device, B, H, S, qk_d, B, H, S, V_D=v_d
        )

    @supported_platform
    @skip_on_cpu
    def test_autograd_function_in_score_mod(self, device):
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
            torch.randn(1, 8, 1024, 64, device=device, requires_grad=True)
            for _ in range(3)
        )

        # Just checking that it runs
        func(q, k, v)

        # expectedFailure
        # This doesn't work due to vmap + autograd.Function + torch.compile not composing
        # self.run_test(score_mod)

    @supported_platform
    def test_causal_block(self, device):
        def mask_mod(b, h, q, kv):
            return q >= kv

        block_mask = create_block_mask(mask_mod, 1, 1, S, S, device=device)
        attention = functools.partial(flex_attention, block_mask=block_mask)

        self.run_test_with_call(attention, dtype=torch.float16, device=device)

    @supported_platform
    def test_causal_block_paged_attention(self, device):
        def mask_mod(b, h, q, kv):
            return q >= kv

        block_mask = create_block_mask(mask_mod, B, 1, S, S, device=device)
        self.run_test_with_paged_attention(
            score_mod=_identity,
            dtype=torch.float16,
            device=device,
            block_mask=block_mask,
        )

    @supported_platform
    def test_new_empty_mask_mod(self, device):
        S = 128
        q, k, v = (torch.randn(4, 1, S, 64, device=device) for _ in range(3))

        attn_mask = torch.ones(4, 1, S, S, dtype=torch.bool, device=device).tril()

        def score_mod(score, b, h, q_idx, kv_idx):
            h_ = h.new_zeros(h.shape)
            return score + attn_mask[b, h_, q_idx, kv_idx]

        def causal(b, h, q_idx, kv_idx):
            h_ = h.new_zeros(h.shape)
            return attn_mask[b, h_, q_idx, kv_idx]

        block_mask = create_block_mask(
            causal, B=4, H=None, Q_LEN=S, KV_LEN=S, device=device
        )
        torch.compile(flex_attention, fullgraph=True)(
            q, k, v, score_mod, block_mask=block_mask
        )

    @supported_platform
    @common_utils.parametrize("head_dim", [17, 24, 94, 121])
    @dtypes(*device_configs["cpu"].dtypes_fast)
    @dtypesIfCUDA(*device_configs["cuda"].dtypes_fast)
    @dtypesIfXPU(*device_configs["xpu"].dtypes_fast)
    def test_non_pow_2_headdim(self, device, dtype, head_dim):
        self.run_test(_rel_bias, dtype, device, B, H, S, head_dim, B, H, S, head_dim)

    @supported_platform
    def test_GQA_causal_mask(self, device):
        def mask_mod(b, h, q, kv):
            return q >= kv

        block_mask = create_block_mask(mask_mod, B, 1, S // 8, S // 8, device=device)
        attention = functools.partial(
            flex_attention, block_mask=block_mask, enable_gqa=True
        )

        self.run_test_with_call(
            attention,
            torch.float16,
            device,
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
            _identity,
            dtype=torch.float16,
            device=device,
            Q_H=H * 4,
            Q_S=S // 8,
            KV_H=H,
            KV_S=S // 8,
            block_mask=block_mask,
        )

    @supported_platform
    def test_custom_block_mask_generator(self, device):
        def mask_mod(b, h, q, kv):
            return q >= kv

        auto_mask = create_block_mask(mask_mod, 1, 1, S, S, device=device)
        BLOCK_SIZE = 128

        def causal_constructor(S):
            num_blocks = torch.arange(S // BLOCK_SIZE, device=device) + 1
            indices = torch.arange(S // BLOCK_SIZE, device=device).expand(
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
    @skip_on_cpu
    @dtypes(*device_configs["cpu"].dtypes)
    @dtypesIfCUDA(*device_configs["cuda"].dtypes)
    @dtypesIfXPU(*device_configs["xpu"].dtypes)
    @common_utils.parametrize("score_mod", [_identity, _causal])
    def test_logsumexp_correctness(self, device, dtype, score_mod):
        make_tensor = functools.partial(
            torch.randn,
            (B, H, S, D),
            dtype=dtype,
            device=device,
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
    @skip_on_cpu
    def test_logsumexp_only_return(self, device):
        make_tensor = functools.partial(
            torch.randn,
            (B, H, S, D),
            dtype=torch.float32,
            device=device,
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
    @skip_on_cpu
    @common_utils.parametrize(
        "score_mod", [_identity, _causal, _times_two, _squared, _trig, _trig2]
    )
    def test_aot_eager_gradcheck(self, device, score_mod):
        make_tensor = functools.partial(
            torch.randn,
            (2, 2, 11, 4),
            device=device,
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
    @skip_on_cpu
    def test_eager_backward_strides(self, device):
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

        model = Repro().to(device)
        x = torch.randn((1, 512, 256), device=device, requires_grad=True)
        out = torch.compile(model, backend="aot_eager", fullgraph=True)(x)
        out.backward(torch.ones_like(out))

    @supported_platform
    @skip_on_cpu
    def test_differentiable_logsumexp_gradcheck(self, device):
        make_tensor = functools.partial(
            torch.randn,
            (2, 2, 11, 4),
            device=device,
            dtype=torch.float64,
            requires_grad=True,
        )
        query, key, value = make_tensor(), make_tensor(), make_tensor()

        def flex_attention_lse_only(q, k, v):
            return flex_attention(q, k, v, return_lse=True)[1]

        func = torch.compile(flex_attention_lse_only, backend="aot_eager")

        self.assertTrue(
            torch.autograd.gradcheck(func, (query, key, value), raise_exception=True)
        )

    @supported_platform
    @skip_on_cpu
    def test_differentiable_logsumexp_compiled(self, device):
        make_tensor = functools.partial(
            torch.randn,
            (2, 2, 128, 64),
            device=device,
            dtype=torch.float32,
            requires_grad=True,
        )
        q, k, v = make_tensor(), make_tensor(), make_tensor()
        lse_mask = torch.randn(2, 2, 128, device=device)

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
    @skip_on_cpu
    def _test_block_mask_reuse_with_weird_mask(self, device):
        def mask(b, h, q, kv):
            return (kv < 256) | (kv >= 2048)

        make_tensor = functools.partial(
            torch.randn,
            (4, 4, 4096, 64),
            device=device,
            dtype=torch.float32,
            requires_grad=True,
        )

        block_mask = create_block_mask(mask, None, None, 4096, 4096, device=device)
        # Compile 1st version with q/k/v(seqlen=4096) and block_mask(seqlen=4096)
        torch.compile(flex_attention, dynamic=True, fullgraph=True)(
            make_tensor(), make_tensor(), make_tensor(), block_mask=block_mask
        )

        make_tensor2 = functools.partial(
            torch.randn,
            (4, 4, 2048, 64),
            device=device,
            dtype=torch.float32,
            requires_grad=True,
        )
        q, k, v = make_tensor2(), make_tensor2(), make_tensor2()

        # Compile 2nd version with q/k/v(seqlen=2048) and block_mask(seqlen=4096),
        # The graph includes the BlockMask._adjust part.
        out = torch.compile(flex_attention, dynamic=True, fullgraph=True)(
            q, k, v, block_mask=block_mask
        )
        out.sum().backward()
        q_grad, k_grad, v_grad = q.grad, k.grad, v.grad
        q.grad = None
        k.grad = None
        v.grad = None

        block_mask2 = create_block_mask(mask, None, None, 2048, 2048, device=device)
        # Reuse the 1st version with q/k/v(seqlen=2048) and block_mask(seqlen=2048)
        out2 = torch.compile(flex_attention, dynamic=True, fullgraph=True)(
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
    @skip_on_cpu
    def test_float32_matmul_precision(self, device):
        make_tensor = functools.partial(
            torch.zeros,
            (2, 2, 128, 32),
            device=device,
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
    @skip_on_cpu
    @common_utils.parametrize("score_mod_name", ["_head_offset"])
    @common_utils.parametrize("mode", ["eager", "aot_eager"])
    def test_captured_score_mod_aot_eager_gradcheck(
        self, device, score_mod_name: str, mode: str
    ):
        make_tensor = functools.partial(
            torch.randn,
            (2, 2, 11, 4),
            device=device,
            dtype=torch.float64,
            requires_grad=True,
        )
        query, key, value = make_tensor(), make_tensor(), make_tensor()

        func = torch.compile(flex_attention, backend=mode, fullgraph=True)
        score_mod = captured_buffers_map[score_mod_name](torch.float64, device)

        self.assertTrue(
            torch.autograd.gradcheck(
                func, (query, key, value, score_mod), raise_exception=True
            )
        )

    @supported_platform
    @skip_on_cpu
    @common_utils.parametrize("mode", ["eager", "aot_eager"])
    def test_document_masking_edge_case(self, device, mode):
        requires_grad = device in DEVICE_SUPPORTS_BACKWARDS
        document_masks = torch.full((2, 128), 0, dtype=torch.int32, device=device)
        document_masks[:, 64:] = 1

        def mask_mod(b, h, q, kv):
            same_doc = document_masks[b, q] == document_masks[b, kv]
            return same_doc

        make_tensor = functools.partial(
            torch.randn,
            (2, 1, 128, 4),
            device=device,
            dtype=torch.float64,
            requires_grad=requires_grad,
        )
        query, key, value = make_tensor(), make_tensor(), make_tensor()
        func = torch.compile(flex_attention, backend=mode, fullgraph=True)

        block_mask = create_block_mask(mask_mod, 2, 1, 128, 128, device=device)
        out = func(query, key, value, block_mask=block_mask)
        if requires_grad:
            out.sum().backward()

    @supported_platform
    @skip_on_cpu
    def test_strided_backwards(self, device):
        shape = (1, 2, 4096, 64)
        Q = torch.randn(shape, requires_grad=True, device=device)
        K = torch.randn(shape, requires_grad=True, device=device)
        V = torch.randn(shape, requires_grad=True, device=device)
        func = torch.compile(flex_attention, dynamic=True, fullgraph=True)

        K_sliced = K[:, :, :-128]
        V_sliced = V[:, :, :-128]

        out_eager = flex_attention(Q, K_sliced, V_sliced)

        out_compiled, code = run_and_get_code(func, Q, K_sliced, V_sliced)

        # Make sure flex attention kernels have flex_attention in name
        FileCheck().check_regex("triton_tem_fused_flex_attention.*").run(code[0])
        FileCheck().check_regex("triton_tem_fused_flex_attention_backward.*").run(
            code[1]
        )

        grad = torch.rand_like(out_eager)

        eager_grads = torch.autograd.grad(out_eager, (Q, K, V), grad)
        compiled_grads = torch.autograd.grad(out_compiled, (Q, K, V), grad)

        for eager, compiled in zip(eager_grads, compiled_grads):
            torch.testing.assert_close(eager, compiled, atol=9e-3, rtol=0)

    @supported_platform
    @skip_on_cpu
    @common_utils.parametrize("mode", ["eager", "inductor", "paged_attention"])
    @common_utils.parametrize(
        "permute_order",
        [
            (0, 1, 2, 3),  # Default order
            (1, 0, 2, 3),  # Reverse order
            (0, 2, 1, 3),  # Mixed order
            (2, 0, 1, 3),  # Another mixed order
            (0, 1, 3, 2),  # Non contiguous last dim
        ],
    )
    @common_utils.parametrize("shape", [(2, 1, 128, 16), (4, 2, 64, 16)])
    def test_flex_attention_stride_ordering(self, device, mode, permute_order, shape):
        from torch._inductor.ir import get_stride_order

        if torch.version.hip and mode == "paged_attention":
            raise self.skipTest(
                "TODO: figure out why mode_paged_attention_permute_order3_shape0 on MI200 caused mem fault"
            )

        dtype = torch.float32
        # Setup
        requires_grad = device in DEVICE_SUPPORTS_BACKWARDS
        make_tensor = functools.partial(
            torch.randn,
            shape,
            device=device,
            dtype=dtype,
            requires_grad=False if mode == "paged_attention" else requires_grad,
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
            out, _ = self.run_paged_attention(
                _identity, query, key, value, dtype, device=device
            )
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
    @skip_on_cpu
    @common_utils.parametrize("mode", ["eager", "inductor"])
    @common_utils.parametrize(
        "permute_order",
        [(0, 1, 2, 3), (1, 0, 2, 3), (0, 2, 1, 3), (2, 0, 1, 3), (0, 1, 3, 2)],
    )
    @common_utils.parametrize("shape", [(2, 5, 128, 16), (4, 2, 64, 16)])
    def test_flex_attention_backward_stride_ordering(
        self, device, mode, permute_order, shape
    ):
        from torch._inductor.ir import get_stride_order

        dtype = torch.float32
        make_tensor = functools.partial(
            torch.randn, shape, device=device, dtype=dtype, requires_grad=False
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
    def test_non_contiguous_last_dim(self, device):
        """Test flex_attention with tensors having non contiguous last dimension."""
        B, H, D = 4, 8, 64
        dtype = torch.float16 if device in DEVICE_SUPPORTS_BACKWARDS else torch.float32
        for S in [16, 64]:

            def column_major_tensor():
                tensor = torch.randn(
                    (B, H, S, D),
                    dtype=dtype,
                    device=device,
                )
                # Column major in last 2 dims
                return tensor.transpose(-1, -2).contiguous().transpose(-1, -2)

            q = column_major_tensor()
            k = column_major_tensor()
            v = column_major_tensor()

            requires_grad = device in DEVICE_SUPPORTS_BACKWARDS
            if requires_grad:
                q.requires_grad_(True)
                k.requires_grad_(True)
                v.requires_grad_(True)

            self.assertNotEqual(q.stride()[-1], 1)
            self.assertNotEqual(k.stride()[-1], 1)
            self.assertNotEqual(v.stride()[-1], 1)

            q_ref, k_ref, v_ref = query_key_value_clones(q, k, v)
            q_gold, k_gold, v_gold = query_key_value_clones(q, k, v, torch.float64)

            golden_out = flex_attention(q_gold, k_gold, v_gold)
            ref_out = flex_attention(q_ref, k_ref, v_ref)

            flex_compiled = torch.compile(flex_attention, fullgraph=True, dynamic=True)
            compiled_out = flex_compiled(q, k, v)

            self._check_out(golden_out, ref_out, compiled_out)

            if requires_grad:
                backward_grad = torch.randn_like(ref_out)

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

    @supported_platform
    @common_utils.parametrize("compile", [True, False])
    def test_fully_masked_out_rows_0_check(self, device, compile: bool):
        # Ensure fully masked out rows won't cause NaNs.
        requires_grad = device in DEVICE_SUPPORTS_BACKWARDS
        query = torch.randn(
            (B, H, S, D),
            dtype=torch.float32,
            device=device,
            requires_grad=requires_grad,
        )
        key = torch.randn(
            (B, H, S, D),
            dtype=torch.float32,
            device=device,
            requires_grad=requires_grad,
        )
        value = torch.randn(
            (B, H, S, D),
            dtype=torch.float32,
            device=device,
            requires_grad=requires_grad,
        )

        M = S // 2

        def mask_mod(b, h, q, kv):
            return q < M

        block_mask = create_block_mask(mask_mod, B, 1, S, S, device=device)

        flex = (
            torch.compile(flex_attention, dynamic=False) if compile else flex_attention
        )
        if requires_grad:
            out, lse = flex(query, key, value, block_mask=block_mask, return_lse=True)
            self.assertEqual(out[:, :, M:, :].sum(), 0)
            self.assertTrue((lse[:, :, M:] == -float("inf")).all())

            loss = out.sum() + lse.sum()
            loss.backward()
            self.assertEqual(query.grad[:, :, M:, :].sum(), 0)
        else:
            out = flex(query, key, value, block_mask=block_mask, return_lse=False)

        self.assertEqual(out[:, :, M:, :].sum(), 0)

    @supported_platform
    def test_fully_masked_out_rows(self, device):
        M = S // 2

        def mask_mod(b, h, q, kv):
            return q < M

        block_mask = create_block_mask(mask_mod, B, 1, S, S, device=device)

        def noop_mod(score, b, h, q_idx, kv_idx):
            return score

        self.run_test(
            noop_mod, torch.float32, device, B, H, S, D, B, H, S, D, block_mask
        )

    @supported_platform
    @skip_on_cpu
    def test_kernel_options_argument_is_respected(self, device):
        make_tensor = functools.partial(
            torch.randn,
            (2, 2, 128, 64),
            device=device,
            dtype=torch.float32,
            requires_grad=True,
        )
        q, k, v = make_tensor(), make_tensor(), make_tensor()

        # Ensure we respect user's input kernel options.
        _, code = run_and_get_code(
            torch.compile(flex_attention, fullgraph=True),
            q,
            k,
            v,
            kernel_options={"BLOCK_M": 16},
        )
        FileCheck().check("BLOCK_M : tl.constexpr = 16").run(code[0])

    @supported_platform
    @skip_on_cpu
    def test_backend_auto_matches_triton_large(self, device):
        """BACKEND='AUTO' should follow Triton heuristics on large shapes."""
        make_tensor = functools.partial(
            torch.randn,
            (2, 2, 256, 64),
            device=device,
            dtype=torch.float16,
            requires_grad=False,
        )
        q, k, v = make_tensor(), make_tensor(), make_tensor()

        def compile_and_run(kernel_options):
            return run_and_get_code(
                torch.compile(flex_attention, fullgraph=True),
                q,
                k,
                v,
                kernel_options=kernel_options,
            )

        default_out, default_code = compile_and_run({"BACKEND": "AUTO"})
        triton_out, triton_code = compile_and_run({"BACKEND": "TRITON"})

        torch.testing.assert_close(default_out, triton_out, atol=0.0, rtol=0.0)

        default_src = "\n".join(default_code)
        FileCheck().check("flex_attention").check_not("flex_decoding").run(default_src)

        triton_src = "\n".join(triton_code)
        FileCheck().check("flex_attention").check_not("flex_decoding").run(triton_src)

    @supported_platform
    @skip_on_cpu
    def test_backend_triton_decode_matches_auto(self, device):
        """BACKEND='TRITON_DECODE' should match heuristics on decode-friendly shapes."""
        make_tensor = functools.partial(
            torch.randn,
            (1, 2, 64, 64),
            device=device,
            dtype=torch.float16,
            requires_grad=False,
        )
        q, k, v = make_tensor(), make_tensor(), make_tensor()

        def compile_and_run(kernel_options):
            return run_and_get_code(
                torch.compile(flex_attention, fullgraph=True),
                q,
                k,
                v,
                kernel_options=kernel_options,
            )

        from torch._inductor.kernel.flex import flex_attention as flex_kernel_mod

        with mock.patch.object(
            flex_kernel_mod,
            "create_flex_decoding_kernel",
            wraps=flex_kernel_mod.create_flex_decoding_kernel,
        ) as decode_kernel:
            default_out, _ = compile_and_run({"BACKEND": "AUTO"})
            self.assertTrue(
                decode_kernel.called,
                "Expected heuristics to dispatch to flex decoding kernel.",
            )

        with mock.patch.object(
            flex_kernel_mod,
            "create_flex_decoding_kernel",
            wraps=flex_kernel_mod.create_flex_decoding_kernel,
        ) as decode_kernel:
            decode_out, _ = compile_and_run({"BACKEND": "TRITON_DECODE"})
            self.assertTrue(
                decode_kernel.called,
                "Expected explicit BACKEND='TRITON_DECODE' to use flex decoding kernel.",
            )

        self.assertEqual(decode_out.shape, (1, 2, 64, 64))
        torch.testing.assert_close(default_out, decode_out, atol=3e-3, rtol=3e-3)

    @supported_platform
    @skip_on_cpu
    def test_backend_triton_decode_errors_when_not_supported(self, device):
        """Requesting decode on unsupported shapes should raise a helpful error."""
        make_tensor = functools.partial(
            torch.randn,
            (1, 2, 256, 64),
            device=device,
            dtype=torch.float16,
            requires_grad=False,
        )
        q, k, v = make_tensor(), make_tensor(), make_tensor()

        flex_compiled = torch.compile(flex_attention, fullgraph=True)
        with self.assertRaisesRegex(
            RuntimeError,
            r"BACKEND='TRITON_DECODE' was specified but flex_decoding cannot be used",
        ):
            flex_compiled(q, k, v, kernel_options={"BACKEND": "TRITON_DECODE"})

    @supported_platform
    @skip_on_cpu
    def test_backend_triton_decode_errors_with_non_power_of_two_gqa(self, device):
        """BACKEND='TRITON_DECODE' should fail when GQA ratio is not a power of two."""
        q = torch.randn(
            1, 3, 64, 64, device=device, dtype=torch.float16, requires_grad=False
        )
        k = torch.randn(
            1, 1, 64, 64, device=device, dtype=torch.float16, requires_grad=False
        )
        v = torch.randn(
            1, 1, 64, 64, device=device, dtype=torch.float16, requires_grad=False
        )

        flex_compiled = torch.compile(flex_attention, fullgraph=True)
        with self.assertRaisesRegex(
            RuntimeError,
            r"BACKEND='TRITON_DECODE' was specified but flex_decoding cannot be used",
        ):
            flex_compiled(
                q,
                k,
                v,
                enable_gqa=True,
                kernel_options={"BACKEND": "TRITON_DECODE"},
            )

    @supported_platform
    @skip_on_cpu
    def test_backend_rejects_legacy_force_use_flag(self, device):
        """Combining BACKEND with FORCE_USE_FLEX_ATTENTION should raise an error."""
        make_tensor = functools.partial(
            torch.randn,
            (2, 2, 128, 64),
            device=device,
            dtype=torch.float16,
            requires_grad=False,
        )
        q, k, v = make_tensor(), make_tensor(), make_tensor()

        flex_compiled = torch.compile(flex_attention, fullgraph=True)
        with self.assertRaisesRegex(
            RuntimeError,
            r"BACKEND cannot be combined with legacy FORCE_USE_FLEX_ATTENTION",
        ):
            flex_compiled(
                q,
                k,
                v,
                kernel_options={
                    "BACKEND": "TRITON",
                    "FORCE_USE_FLEX_ATTENTION": True,
                },
            )

    @supported_platform
    def test_backend_defaults_and_rejects_invalid(self, device):
        device = torch.device(device)
        query = torch.randn(1, 1, 4, 8, device=device, dtype=torch.float32)
        key = torch.randn(1, 1, 4, 8, device=device, dtype=torch.float32)
        value = torch.randn(1, 1, 4, 8, device=device, dtype=torch.float32)

        kernel_options = _apply_kernel_options(
            query, key, value, return_lse=True, kernel_options={}
        )
        self.assertEqual(kernel_options["BACKEND"], "AUTO")

        with self.assertRaisesRegex(ValueError, r"Invalid BACKEND value 'INVALID'"):
            _apply_kernel_options(
                query,
                key,
                value,
                return_lse=True,
                kernel_options={"BACKEND": "INVALID"},
            )

    @supported_platform
    def test_block_mask_non_divisible(self, device):
        seq = torch.arange(1023, device=device) // 128

        def mod(b, h, q, kv):
            return seq[q] == seq[kv]

        block_mask = create_block_mask(mod, None, None, 1023, 1023, device=device)
        torch.compile(create_block_mask)(mod, None, None, 1023, 1023, device=device)
        self.run_test_with_call(
            lambda q, k, v: flex_attention(q, k, v, block_mask=block_mask),
            torch.float16,
            device,
            Q_S=1023,
            KV_S=1023,
        )

    @supported_platform
    def test_causal_block_non_divisible(self, device):
        def mask_mod(b, h, q, kv):
            return q >= kv

        block_mask = create_block_mask(mask_mod, B, 1, S - 1, S - 1, device=device)
        attention = functools.partial(flex_attention, block_mask=block_mask)

        self.run_test_with_call(attention, torch.float16, device, Q_S=S - 1, KV_S=S - 1)

    @supported_platform
    @skip_on_cpu
    def test_modular_indexing(self, device):
        B, H, N, D = 100, 12, 128, 64
        dtype = torch.bfloat16
        device = torch.device(device)

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

        m = Attention().to(device).eval().to(dtype)
        m = torch.compile(m, mode="default", fullgraph=False)

        q = torch.randn(B, H, N, D, device=device, dtype=dtype)
        k = torch.randn(B, H, N, D, device=device, dtype=dtype)
        v = torch.randn(B, H, N, D, device=device, dtype=dtype)

        m(q, k, v)

    @supported_platform
    @skip_on_cpu
    def test_force_write_lse(self, device):
        dtype = torch.float32
        make_tensor = functools.partial(
            torch.randn,
            (2, 2, 128, 16),
            device=device,
            dtype=dtype,
            requires_grad=False,
        )
        query, key, value = make_tensor(), make_tensor(), make_tensor()
        out_eager, lse_eager = flex_attention(query, key, value, return_lse=True)

        flex_compile = torch.compile(flex_attention)
        out_compiled, lse_compiled = flex_compile(query, key, value, return_lse=True)

        out_paged, lse_paged = self.run_paged_attention(
            score_mod=_identity, q=query, k=key, v=value, dtype=dtype, device=device
        )

        torch.testing.assert_close(lse_eager, lse_compiled, atol=3e-3, rtol=0)
        requires_grad = device in DEVICE_SUPPORTS_BACKWARDS
        if requires_grad:
            torch.testing.assert_close(lse_eager, lse_paged, atol=3e-3, rtol=0)

    @supported_platform
    @skip_on_cpu
    @common_utils.parametrize("backend", ["flex_attention", "flex_decode", "eager"])
    def test_lse_masked_output(self, device, backend):
        if backend == "flex_decode":
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
            device=device,
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
            sliding_window_causal,
            B=None,
            H=None,
            Q_LEN=N_CTX,
            KV_LEN=N_CTX,
            device=device,
        )
        global_causal = torch.nn.attention.flex_attention.create_block_mask(
            global_causal, B=None, H=None, Q_LEN=N_CTX, KV_LEN=N_CTX, device=device
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
    @skip_on_cpu
    def test_mixed_device_error_message(self, device):
        # Create tensors on different devices
        cpu_tensor = torch.randn(2, 2, 128, 16, device="cpu")
        gpu_tensor = torch.randn(2, 2, 128, 16, device=device)

        # Use different devices for query, key, and value
        query, key, value = cpu_tensor, gpu_tensor, cpu_tensor

        expected_error_message = (
            "Expected query, key, and value to have the same device type, "
            f"but got query.device: {query.device}, key.device: {key.device}, "
            f"and value.device: {value.device} instead."
        )

        with self.assertRaisesRegex(ValueError, expected_error_message):
            flex_attention(query, key, value)

    @supported_platform
    @skip_on_cpu
    def test_captured_wrong_device_error_message(self, device):
        means = torch.randn(64, 3, device=device)
        length_scales = torch.logspace(0.001, 0.1, 8, device="cpu")

        def euclidean_dist_pos_embed(score, b, h, q_idx, k_idx):
            q_pos = means[q_idx]
            k_pos = means[k_idx]
            dist = (q_pos - k_pos).pow(2).sum(-1).sqrt()
            scale = length_scales[h]
            inv_dist = torch.exp(-dist / scale)
            return inv_dist * score

        expected_error_message = "Buffers cannot be created"

        q, k, v = (torch.randn(1, 8, 64, 64, device=device) for _ in range(3))
        with self.assertRaisesRegex(RuntimeError, expected_error_message):
            torch.compile(flex_attention)(q, k, v, score_mod=euclidean_dist_pos_embed)

    @supported_platform
    @skip_on_cpu
    def test_cant_lower_error_message(self, device):
        # We can't lower a 256-element reduction inside a pointwise reduction
        means = torch.randn(64, 256, device=device)
        length_scales = torch.logspace(0.001, 0.1, 8, device=device)

        def euclidean_dist_pos_embed(score, b, h, q_idx, k_idx):
            q_pos = means[q_idx]
            k_pos = means[k_idx]
            dist = (q_pos - k_pos).pow(2).sum(-1).sqrt()
            scale = length_scales[h]
            inv_dist = torch.exp(-dist / scale)
            return inv_dist * score

        expected_error_message = "Buffers cannot be created"

        q, k, v = (torch.randn(1, 8, 64, 64, device=device) for _ in range(3))
        with self.assertRaisesRegex(RuntimeError, expected_error_message):
            torch.compile(flex_attention)(q, k, v, score_mod=euclidean_dist_pos_embed)

    @supported_platform
    @skip_on_cpu
    def test_reduction_unrolled(self, device):
        # We can't lower a 256-element reduction inside a pointwise reduction
        means = torch.randn(S, 3, device=device)
        length_scales = torch.logspace(0.001, 0.1, H, device=device)

        def euclidean_dist_pos_embed(score, b, h, q_idx, k_idx):
            q_pos = means[q_idx]
            k_pos = means[k_idx]
            dist = (q_pos - k_pos).pow(2).sum(-1).sqrt()
            scale = length_scales[h]
            inv_dist = torch.exp(-dist / scale)
            return inv_dist * score

        self.run_test(euclidean_dist_pos_embed, torch.bfloat16, device=device)

    @supported_platform
    @skip_on_cpu
    def test_invalid_block_size(self, device):
        # Create tensors on different devices
        q, k, v = (torch.randn(1, 8, 128, 64, device=device) for _ in range(3))

        expected_error_message = (
            "ValueError: Q and KV block size must be divisible by BLOCK_M and BLOCK_N."
        )
        block_mask = create_block_mask(
            noop_mask, 1, 8, 128, 128, BLOCK_SIZE=96, device=device
        )

        with self.assertRaisesRegex(RuntimeError, expected_error_message):
            torch.compile(flex_attention)(q, k, v, block_mask=block_mask)

    @supported_platform
    @skip_on_cpu
    def test_small_q_kv_len(self, device):
        make_tensor = functools.partial(
            torch.ones,
            (1, 1, 1, 16),
            device=device,
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
    @skip_on_cpu
    def test_dynamic_shapes_bug_dynamic_batch(self, device):
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
                    device=device,
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

        model = Model(128).to(device)
        B, F, T = 16, 256, 12
        for _ in range(5):
            x = torch.randn(B, T, F, device=device)
            l = torch.randint(0, T, (B,), device=device)
            model(x, l)

        assert counter.frame_count == 1, (
            f"Expected 1 graph, but got {counter.frame_count} graphs"
        )

    @supported_platform
    @skip_on_cpu
    def test_dynamic_shapes_with_custom_kernel_options(self, device):
        make_tensor = functools.partial(
            torch.ones,
            (8, 8, 1024, 64),
            device=device,
            dtype=torch.bfloat16,
        )
        query, key, value = make_tensor(), make_tensor(), make_tensor()
        kernel_options = {"BLOCK_M": 64, "BLOCK_N": 64}
        out_eager = flex_attention(query, key, value, kernel_options=kernel_options)

        flex_compile = torch.compile(flex_attention, fullgraph=True, dynamic=True)
        out_compiled = flex_compile(query, key, value, kernel_options=kernel_options)

        torch.testing.assert_close(out_eager, out_compiled, atol=3e-3, rtol=2e-3)

    @supported_platform
    def test_dynamic_shapes_with_max_autotune(self, device):
        make_tensor = functools.partial(
            torch.ones,
            (8, 8, 1024, 64),
            device=device,
            dtype=torch.float if device == "cpu" else torch.bfloat16,
        )
        query, key, value = make_tensor(), make_tensor(), make_tensor()
        block_mask = create_block_mask(
            _causal_mask, None, None, 1024, 1024, device=device
        )

        out_eager = flex_attention(query, key, value, block_mask=block_mask)

        flex_compile = torch.compile(
            flex_attention, fullgraph=True, dynamic=True, mode="max-autotune"
        )
        out_compiled = flex_compile(query, key, value, block_mask=block_mask)

        torch.testing.assert_close(out_eager, out_compiled, atol=3e-3, rtol=2e-3)

    @supported_platform
    @skip_on_cpu
    def test_zero_length_sequence_error(self, device):
        make_tensor = functools.partial(
            torch.ones,
            (8, 8, 0, 64),  # Zero in sequence dimension
            device=device,
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
    def test_causal_block_non_divisible_with_captured_buffer(
        self,
        device,
    ):
        Q_S = S - 3
        KV_S = S - 3
        offset_q = torch.randn(Q_S, device=device, dtype=torch.bfloat16)
        offset_kv = torch.randn(KV_S, device=device, dtype=torch.bfloat16)

        def score_mod(score, b, h, q, kv):
            return score + offset_q[q] + offset_kv[kv]

        def mask_mod(b, h, q, kv):
            return q >= kv

        block_mask = create_block_mask(mask_mod, B, 1, Q_S, KV_S, device=device)

        attention = functools.partial(flex_attention, block_mask=block_mask)

        self.run_test_with_call(
            attention, Q_S=Q_S, KV_S=KV_S, dtype=torch.bfloat16, device=device
        )

    @supported_platform
    def test_non_divisible_with_captured_buffer(self, device):
        Q_S = S + 3
        KV_S = S + 3

        multiplier = torch.randn(Q_S, device=device, dtype=torch.bfloat16)

        def apply_multiplicative_bias(score, b, h, q_idx, kv_idx):
            return score * multiplier[q_idx]

        attention = functools.partial(
            flex_attention, score_mod=apply_multiplicative_bias
        )

        self.run_test_with_call(
            attention, Q_S=Q_S, KV_S=KV_S, dtype=torch.bfloat16, device=device
        )

    @supported_platform
    def test_num_warps_8_error(self, device):
        attention = functools.partial(flex_attention, score_mod=_identity)
        self.run_test_with_call(
            attention,
            dtype=torch.float16,
            device=device,
            Q_S=128,
            KV_S=128,
            Q_D=128,
            V_D=128,
        )

    @supported_platform
    @unittest.skipIf(not TEST_MULTIGPU, "detected only one GPU")
    def test_qkv_and_block_mask_on_the_same_device(self, device):
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
    @skip_on_cpu
    @unittest.skipIf(config.triton.native_matmul, "different dynamo counters")
    def test_free_symbol_dynamic(self, device):
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

        model = SimpleAttention().to(device)
        model.compile(mode="default", dynamic=True)
        sequence_len = 256

        # Test different batch shapes with dense masks
        torch._dynamo.reset()
        for batch_shape in [4, 16, 32]:
            # Create dense mask
            rand_mask = torch.randint(
                0, 2, (batch_shape, sequence_len), device=device
            ).bool()
            block_mask = torch.compile(create_block_mask, dynamic=True)(
                B=batch_shape,
                BLOCK_SIZE=128,
                mask_mod=lambda b, h, q_idx, kv_idx: ~rand_mask[b, q_idx],
                H=None,
                Q_LEN=sequence_len,
                KV_LEN=sequence_len,
                device=device,
            )

            # Run forward pass
            x = torch.randn(batch_shape, sequence_len, 512, device=device)
            model(x, block_mask=block_mask)

        self.assertEqual(torch._dynamo.utils.counters["aot_autograd"]["ok"], 2)

    @supported_platform
    @skip_on_cpu
    def test_symbol_closure_in_score_mod(self, device):
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

        model = SimpleAttention().to(device)
        from torch._dynamo.testing import EagerAndRecordGraphs

        backend = EagerAndRecordGraphs()
        model.compile(mode="default", dynamic=True, backend=backend)
        sequence_len = 256

        torch._dynamo.reset()
        for batch_shape in [4, 16, 32]:
            x = torch.randn(batch_shape, sequence_len, 512, device=device)
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
    @skip_on_cpu
    def test_fw_bw_graph_correctness(self, device):
        cnt = CompileCounterWithBackend("aot_eager")
        make_tensor = functools.partial(
            torch.randn,
            (2, 2, 128, 4),
            device=device,
            dtype=torch.float64,
            requires_grad=True,
        )
        query, key, value = make_tensor(), make_tensor(), make_tensor()

        def causal_mask(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx

        block_mask = create_block_mask(causal_mask, 1, 1, 128, 128, device=device)

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
        flex_attention = torch.ops.higher_order.flex_attention(l_query_, l_key_, l_value_, score_mod_0, (128, 128, l_block_mask_kv_num_blocks, l_block_mask_kv_indices, l_block_mask_full_kv_num_blocks, l_block_mask_full_kv_indices, l_block_mask_q_num_blocks, l_block_mask_q_indices, l_block_mask_full_q_num_blocks, l_block_mask_full_q_indices, 128, 128, mask_fn_0), 0.5, {'BACKEND': 'AUTO', 'PRESCALE_QK': False, 'ROWS_GUARANTEED_SAFE': False, 'BLOCKS_ARE_CONTIGUOUS': False, 'WRITE_DQ': True, 'OUTPUT_LOGSUMEXP': True, 'OUTPUT_MAX': False}, (), ());  l_query_ = l_key_ = l_value_ = score_mod_0 = l_block_mask_kv_num_blocks = l_block_mask_kv_indices = l_block_mask_full_kv_num_blocks = l_block_mask_full_kv_indices = l_block_mask_q_num_blocks = l_block_mask_q_indices = l_block_mask_full_q_num_blocks = l_block_mask_full_q_indices = mask_fn_0 = None
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
        full_default_4: "f32[2, 2, 128]" = torch.ops.aten.full.default([2, 2, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='GPU_TYPE', index=0), pin_memory = False)
        fw_graph0 = self.fw_graph0
        joint_graph0 = self.joint_graph0
        mask_graph0 = self.mask_graph0
        flex_attention_backward = torch.ops.higher_order.flex_attention_backward(primals_1, primals_2, primals_3, getitem_2, getitem_3, tangents_1, full_default_4, fw_graph0, joint_graph0, (1, 1, full, full_default, None, None, convert_element_type, convert_element_type_1, None, None, 1073741824, 1073741824, mask_graph0), 0.5, {'BACKEND': 'AUTO', 'PRESCALE_QK': False, 'ROWS_GUARANTEED_SAFE': False, 'BLOCKS_ARE_CONTIGUOUS': False, 'WRITE_DQ': True, 'OUTPUT_LOGSUMEXP': True, 'OUTPUT_MAX': False}, (), ());  primals_1 = primals_2 = primals_3 = getitem_2 = getitem_3 = tangents_1 = full_default_4 = fw_graph0 = joint_graph0 = full = full_default = convert_element_type = convert_element_type_1 = mask_graph0 = None
        getitem_5: "f64[2, 2, 128, 4]" = flex_attention_backward[0]
        getitem_6: "f64[2, 2, 128, 4]" = flex_attention_backward[1]
        getitem_7: "f64[2, 2, 128, 4]" = flex_attention_backward[2];  flex_attention_backward = None
        return (getitem_5, getitem_6, getitem_7)

    class fw_graph0(torch.nn.Module):
        def forward(self, arg0_1: "f64[]", arg1_1: "i32[]", arg2_1: "i32[]", arg3_1: "i32[]", arg4_1: "i32[]"):
            mul: "f64[]" = torch.ops.aten.mul.Tensor(arg0_1, arg0_1);  arg0_1 = None
            return mul

    class joint_graph0(torch.nn.Module):
        def forward(self, arg0_1: "f64[]", arg1_1: "i32[]", arg2_1: "i32[]", arg3_1: "i32[]", arg4_1: "i32[]", arg5_1: "f64[]"):
            mul_1: "f64[]" = torch.ops.aten.mul.Tensor(arg5_1, arg0_1)
            mul_2: "f64[]" = torch.ops.aten.mul.Tensor(arg5_1, arg0_1);  arg5_1 = arg0_1 = None
            add: "f64[]" = torch.ops.aten.add.Tensor(mul_2, mul_1);  mul_2 = mul_1 = None
            return [add, None, None, None, None]

    class mask_graph0(torch.nn.Module):
        def forward(self, arg0_1: "i32[]", arg1_1: "i32[]", arg2_1: "i32[]", arg3_1: "i32[]"):
            full_default: "b8[]" = torch.ops.aten.full.default([], True, dtype = torch.bool, layout = torch.strided, device = device(type='GPU_TYPE', index=0), pin_memory = False)
            return full_default
""".replace(  # noqa: B950
                "GPU_TYPE", torch.device(device).type
            ),
        )

    @supported_platform
    def test_tensor_subclass_dispatch_order(self, device):
        """Test that tensor subclasses get proper dispatch priority over modes.

        This test verifies the fix that allows tensor subclasses' pyimpl to run before
        FakeTensorMode/FunctionalTensorMode implementations, preventing issues
        where subclasses that error on as_strided would fail in flex_attention.
        """
        import torch.utils._pytree as pytree
        from torch.utils._python_dispatch import return_and_correct_aliasing

        class AsStridedErrorTensor(torch.Tensor):
            @staticmethod
            def __new__(cls, elem):
                assert isinstance(elem, torch.Tensor)
                return torch.Tensor._make_wrapper_subclass(
                    cls,
                    elem.shape,
                    strides=elem.stride(),
                    storage_offset=elem.storage_offset(),
                    dtype=elem.dtype,
                    layout=elem.layout,
                    device=elem.device,
                    requires_grad=elem.requires_grad,
                )

            def __init__(self, elem):
                self.elem = elem

            def __repr__(self):
                return f"AsStridedErrorTensor({self.elem})"

            def __tensor_flatten__(self):
                return ["elem"], None

            @staticmethod
            def __tensor_unflatten__(inner_tensors, meta, outer_size, outer_stride):
                assert meta is None
                elem = inner_tensors["elem"]
                return AsStridedErrorTensor(elem)

            @classmethod
            def __torch_dispatch__(cls, func, types, args, kwargs=None):
                # Error if as_strided is called
                if func is torch.ops.aten.as_strided.default:
                    raise RuntimeError("as_strided was called on AsStridedErrorTensor!")

                if kwargs is None:
                    kwargs = {}
                args_elem = pytree.tree_map_only(
                    AsStridedErrorTensor, lambda x: x.elem, args
                )
                kwargs_elem = pytree.tree_map_only(
                    AsStridedErrorTensor, lambda x: x.elem, kwargs
                )

                out = func(*args_elem, **kwargs_elem)

                def wrap_output(x):
                    if isinstance(x, torch.Tensor):
                        return AsStridedErrorTensor(x)
                    return x

                out_wrapped = pytree.tree_map(wrap_output, out)
                return return_and_correct_aliasing(func, args, kwargs, out_wrapped)

        from torch._higher_order_ops.flex_attention import (
            flex_attention as flex_attention_hop,
        )

        @flex_attention_hop.py_impl(AsStridedErrorTensor)
        def flex_attention_as_strided_error_tensor(
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            score_mod,
            block_mask,
            scale,
            kernel_options,
            score_mod_other_buffers=(),
            mask_mod_other_buffers=(),
        ):
            inner_q, inner_k, inner_v = query.elem, key.elem, value.elem
            out, lse, max_scores = flex_attention_hop(
                inner_q,
                inner_k,
                inner_v,
                score_mod,
                block_mask,
                scale,
                kernel_options,
                score_mod_other_buffers,
                mask_mod_other_buffers,
            )
            return (
                AsStridedErrorTensor(out),
                AsStridedErrorTensor(lse),
                AsStridedErrorTensor(max_scores),
            )

        # Test setup
        B, H, S, D = 2, 1, 128, 16
        dtype = torch.float32

        # Create regular tensors
        query_elem = torch.randn(B, H, S, D, device=device, dtype=dtype)
        key_elem = torch.randn(B, H, S, D, device=device, dtype=dtype)
        value_elem = torch.randn(B, H, S, D, device=device, dtype=dtype)

        # Test 1: Verify as_strided raises error when called directly on AsStridedErrorTensor
        test_tensor = AsStridedErrorTensor(query_elem)
        with self.assertRaisesRegex(
            RuntimeError, "as_strided was called on AsStridedErrorTensor!"
        ):
            torch.as_strided(
                test_tensor, size=(B, H, S, D), stride=test_tensor.stride()
            )

        # Test 2: Run flex_attention with normal tensors first
        compiled_fn = torch.compile(flex_attention, backend="aot_eager")
        normal_out, normal_lse = compiled_fn(
            query_elem, key_elem, value_elem, return_lse=True
        )

        # Test 3: Wrap in our subclass
        query = AsStridedErrorTensor(query_elem)
        key = AsStridedErrorTensor(key_elem)
        value = AsStridedErrorTensor(value_elem)

        # This should NOT error with as_strided after the fix
        # Before the fix, it would error because FakeTensorMode would directly
        # call flex_attention_fake_impl which uses as_strided
        out, lse = compiled_fn(query, key, value, return_lse=True)
        # Verify we got valid output
        self.assertIsInstance(out, AsStridedErrorTensor)
        self.assertIsInstance(lse, AsStridedErrorTensor)
        self.assertEqual(out.shape, (B, H, S, D))
        self.assertEqual(lse.shape, (B, H, S))

        # Test 4: Compare outputs between normal tensors and subclassed tensors
        torch.testing.assert_close(out.elem, normal_out, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(lse.elem, normal_lse, rtol=1e-5, atol=1e-5)

    @supported_platform
    @skip_on_cuda
    def test_cpu_error_message_return_lse(self, device):
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

    @unittest.skipIf(not TEST_MULTIGPU, "detected only one GPU")
    def test_device_cuda_1(self, device):
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

    @supported_platform
    @skip_on_cpu
    def test_custom_score_mod_layout_freeze(self, device):
        torch.manual_seed(0)

        class FlexAttentionCPB(nn.Module):
            def __init__(self, N: int, R: int, H: int = 4, hidden: int = 32):
                super().__init__()
                self.mlp = nn.Sequential(
                    nn.Linear(2, hidden),
                    nn.GELU(),
                    nn.Linear(hidden, H, bias=False),
                )
                self.gamma = nn.Parameter(torch.zeros(H))
                self.H = H
                self._init_tables(N, R)
                self.register_buffer(
                    "r_cutoff", torch.tensor(R, dtype=torch.long), persistent=False
                )

            def _init_tables(self, N: int, R: int) -> None:
                P = N - R
                S = int(P**0.5)
                assert S * S == P
                rng = torch.arange(-(S - 1), S, dtype=torch.float32)
                dY, dX = torch.meshgrid(rng, rng, indexing="ij")
                rel = torch.stack(
                    [dY / max(S - 1, 1), dX / max(S - 1, 1)], dim=-1
                ).reshape(-1, 2)
                rel_table = torch.sign(rel) * torch.log1p(rel.abs())
                self.register_buffer("rel_table", rel_table, persistent=False)

                yy, xx = torch.arange(S), torch.arange(S)
                Y, X = torch.meshgrid(yy, xx, indexing="ij")
                flat = torch.stack([Y, X], 0).flatten(1)
                d = flat[:, :, None] - flat[:, None, :]
                d = d.permute(1, 2, 0).contiguous()
                d[:, :, 0] += S - 1
                d[:, :, 1] += S - 1
                d[:, :, 0] *= 2 * S - 1
                l_idx = d.sum(-1).to(torch.long)
                idx = torch.full((N, N), 0, dtype=torch.long)
                idx[R:, R:] = l_idx
                self.register_buffer("idx_table", idx, persistent=False)

            def _score_mod(self, mu: torch.Tensor):
                bt = self.mlp(self.rel_table)
                idx = self.idx_table
                mu_q, mu_k = mu.unbind(2)
                gam_sig = torch.sigmoid(self.gamma)

                def score_mod(score, b, h, q, kv):
                    has_bias = (q >= self.r_cutoff) & (kv >= self.r_cutoff)
                    l2 = idx[q, kv]
                    bias = bt[l2, h]
                    w_gate = gam_sig[h] * (mu_q[b, h, q] + mu_k[b, h, kv])
                    return score + has_bias.to(score.dtype) * w_gate * bias

                return score_mod

            def forward(self, q, k, v, mu):
                return flex_attention(q, k, v, score_mod=self._score_mod(mu))

        dtype = torch.bfloat16 if PLATFORM_SUPPORTS_BF16 else torch.float16
        device_obj = torch.device(device)
        module = FlexAttentionCPB(N=18, R=2).to(device_obj)
        compiled_module = torch.compile(module, backend="inductor", dynamic=False)

        q = torch.randn(2, 4, 18, 32, device=device_obj, dtype=dtype)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        mu = torch.randn(2, 4, 2, 18, device=device_obj)

        with torch.no_grad():
            with torch.nn.attention.sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                eager_out = module(q, k, v, mu)
                compiled_out = compiled_module(q, k, v, mu)

        self.assertEqual(compiled_out.shape, eager_out.shape)
        torch.testing.assert_close(
            compiled_out.float(), eager_out.float(), atol=2e-2, rtol=2e-2
        )

    @supported_platform
    @skip_on_cpu
    @common_utils.parametrize(
        "ops_to_save",
        [
            [
                torch.ops.aten.mm.default,
            ],
            [
                flex_attention_hop,
            ],
            [torch.ops.aten.mm.default, flex_attention_hop],
        ],
    )
    def test_selective_ac(self, device, ops_to_save):
        class FlexAttentionModule(nn.Module):
            def __init__(self, hidden_size, num_heads):
                super().__init__()
                self.hidden_size = hidden_size
                self.num_heads = num_heads
                self.head_dim = hidden_size // num_heads

                # In-projections (query, key, value)
                self.q_proj = nn.Linear(hidden_size, hidden_size)
                self.k_proj = nn.Linear(hidden_size, hidden_size)
                self.v_proj = nn.Linear(hidden_size, hidden_size)

                # Out-projection
                self.out_proj = nn.Linear(hidden_size, hidden_size)

            def forward(self, x):
                batch_size, seq_len, _ = x.size()

                # Project queries, keys, and values
                q = (
                    self.q_proj(x)
                    .view(batch_size, seq_len, self.num_heads, self.head_dim)
                    .transpose(1, 2)
                )
                k = (
                    self.k_proj(x)
                    .view(batch_size, seq_len, self.num_heads, self.head_dim)
                    .transpose(1, 2)
                )
                v = (
                    self.v_proj(x)
                    .view(batch_size, seq_len, self.num_heads, self.head_dim)
                    .transpose(1, 2)
                )

                # Apply flex attention
                attn_output = flex_attention(
                    q,
                    k,
                    v,
                )

                # Reshape output
                attn_output = (
                    attn_output.transpose(1, 2)
                    .contiguous()
                    .view(batch_size, seq_len, self.hidden_size)
                )

                # Out projection
                output = self.out_proj(attn_output)

                return output

        from torch.utils.checkpoint import (
            checkpoint,
            create_selective_checkpoint_contexts,
        )

        context_fn = functools.partial(
            create_selective_checkpoint_contexts, ops_to_save
        )

        # Define a model that uses FlexAttention with selective activation checkpointing
        class SacModule(nn.Module):
            def __init__(self, hidden_size, num_heads, context_fn):
                super().__init__()
                self.flex_attn = FlexAttentionModule(hidden_size, num_heads)
                self.context_fn = context_fn

            def forward(self, x):
                def flex_attn_fn(x):
                    return self.flex_attn(x)

                output = checkpoint(
                    flex_attn_fn,
                    x,
                    use_reentrant=False,
                    context_fn=self.context_fn,
                )

                return output

        flex_module = SacModule(hidden_size=512, num_heads=8, context_fn=context_fn).to(
            device, dtype=torch.bfloat16
        )
        x = torch.ones(8, 1024, 512, device=device, dtype=torch.bfloat16)

        # Run without compilation
        output_module = flex_module(x)
        compiled_module = torch.compile(flex_module)
        output_compiled = compiled_module(x)

        torch.testing.assert_close(output_module, output_compiled, rtol=1e-2, atol=1e-2)

        # Calculate gradients and compare them
        x.requires_grad_(True)
        output_module = flex_module(x)
        output_compiled = compiled_module(x)
        grad_output = torch.ones_like(output_module)

        grad_module = torch.autograd.grad(
            outputs=output_module, inputs=x, grad_outputs=grad_output, retain_graph=True
        )[0]

        grad_compiled = torch.autograd.grad(
            outputs=output_compiled, inputs=x, grad_outputs=grad_output
        )[0]

        torch.testing.assert_close(grad_module, grad_compiled, rtol=1e-2, atol=1e-2)

    @supported_platform
    @skip_on_cpu
    def test_selective_ac_with_max_autotune_short_query(self, device):
        from functools import partial

        from torch.utils.checkpoint import (
            checkpoint,
            CheckpointPolicy,
            create_selective_checkpoint_contexts,
        )

        compute_intensive_ops = [
            torch.ops.aten.mm,
            torch.ops.aten.bmm,
        ]

        def policy_fn(ctx, op, *args, **kwargs):
            if op in compute_intensive_ops:
                return CheckpointPolicy.MUST_SAVE
            else:
                return CheckpointPolicy.PREFER_RECOMPUTE

        def causal_mask(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx

        class DummyAttentionModule(nn.Module):
            def __init__(self, dim=64, num_heads=4):
                super().__init__()
                self.dim = dim
                self.num_heads = num_heads
                self.head_dim = dim // num_heads

                self.q_proj = nn.Linear(dim, dim)
                self.k_proj = nn.Linear(dim, dim)
                self.v_proj = nn.Linear(dim, dim)
                self.out_proj = nn.Linear(dim, dim)

                self._activation_checkpoint_context_fn = partial(
                    create_selective_checkpoint_contexts, policy_fn
                )

                self._flex_attention = torch.compile(
                    partial(
                        checkpoint,
                        flex_attention,
                        use_reentrant=False,
                        context_fn=self._activation_checkpoint_context_fn,
                    ),
                    mode="max-autotune-no-cudagraphs",
                )

            def forward(self, x, block_mask):
                batch_size, seq_len, _ = x.shape

                q = self.q_proj(x)
                k = self.k_proj(x)
                v = self.v_proj(x)

                q = q.view(
                    batch_size, seq_len, self.num_heads, self.head_dim
                ).transpose(1, 2)
                k = k.view(
                    batch_size, seq_len, self.num_heads, self.head_dim
                ).transpose(1, 2)
                v = v.view(
                    batch_size, seq_len, self.num_heads, self.head_dim
                ).transpose(1, 2)

                attn_out = self._flex_attention(q, k, v, block_mask=block_mask)

                attn_out = (
                    attn_out.transpose(1, 2)
                    .contiguous()
                    .view(batch_size, seq_len, self.dim)
                )

                out = self.out_proj(attn_out)

                return out

        batch_size = 2
        seq_len = 64
        dim = 64
        num_heads = 4

        model = DummyAttentionModule(dim=dim, num_heads=num_heads).to(device)

        x = torch.randn(batch_size, seq_len, dim, device=device, requires_grad=True)

        block_mask = create_block_mask(
            causal_mask,
            B=batch_size,
            H=num_heads,
            Q_LEN=seq_len,
            KV_LEN=seq_len,
            device=device,
        )

        out = model(x, block_mask)

        loss = out.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)

    @supported_platform
    @skip_on_cpu
    def test_validate_small_embedding_size_error_message(self, device):
        # eager support for small embedding size
        q, k, v = [torch.randn(2, 2, 128, 8, device=device) for _ in range(3)]
        flex_attention(q, k, v)

        # compiled cpu support for small embedding size
        q, k, v = [torch.randn(2, 2, 128, 8, device=device) for _ in range(3)]
        flex_attention(q, k, v)

        # compiled gpu kernel does not support small embedding size
        q, k, v = [torch.randn(2, 2, 128, 8, device=device) for _ in range(3)]
        compiled_fa = torch.compile(flex_attention)

        with self.assertRaisesRegex(
            torch._inductor.exc.InductorError,
            "NYI: embedding dimension of the query, key, and value must be "
            "at least 16 but got E=8 and Ev=8",
        ):
            compiled_fa(q, k, v)

        # compiled gpu kernel supports large embedding size
        q, k, v = [torch.randn(2, 2, 128, 16, device=device) for _ in range(3)]
        compiled_fa = torch.compile(flex_attention)

    @unittest.skipIf(
        not has_triton() or not HAS_WARP_SPEC,
        reason="FBCODE Triton is required for this test",
    )
    def test_triton_template_warp_specialization(self, device):
        def make_tensor():
            return torch.rand(4, 16, 4096, 64, device=device, dtype=torch.bfloat16)

        q, k, v = make_tensor(), make_tensor(), make_tensor()
        flex_compiled = torch.compile(flex_attention, fullgraph=True)

        positional_args = (q, k, v)
        keyword_args = {
            "kernel_options": {
                "num_warps": 4,
                "num_consumer_groups": 2,
                "num_buffers_warp_spec": 3,
            }
        }

        # Check if kernel code contains warp specialization parameters
        _, kernel_code = run_and_get_code(
            flex_compiled,
            *positional_args,
            **keyword_args,
        )
        assert kernel_code is not None, "Failed to retrieve compiled kernel code"
        assert "num_consumer_groups" in kernel_code[0], (
            "num_consumer_groups missing in kernel definition"
        )
        assert "num_buffers_warp_spec" in kernel_code[0], (
            "num_buffers_warp_spec missing in kernel definition"
        )

        # Validate correctness
        C1 = flex_compiled(q, k, v)
        C2 = flex_attention(q, k, v)

        assert torch.allclose(C1, C2, atol=1e-2, rtol=1e-2), (
            "Warp specialized kernel result differs from reference"
        )

    @supported_platform
    @skip_on_cpu
    @skipCUDAIf(not has_triton_tma_device(), "Requires TMA enabled CUDA device")
    def test_tma_with_customer_kernel_options(self, device):
        make_tensor = functools.partial(
            torch.ones,
            (1, 1, 256, 128),
            device=device,
            dtype=torch.bfloat16,
        )
        query, key, value = make_tensor(), make_tensor(), make_tensor()

        kernel_options_1 = {
            "BLOCK_M": 128,
            "BLOCK_N": 128,
            "USE_TMA": False,
        }
        kernel_options_2 = {"BLOCK_M": 128, "BLOCK_N": 128, "USE_TMA": True}

        flex_compile = torch.compile(flex_attention, fullgraph=True, dynamic=True)
        out_compiled = flex_compile(query, key, value, kernel_options=kernel_options_1)
        out_tma_compiled = flex_compile(
            query, key, value, kernel_options=kernel_options_2
        )

        # vanilla compiled vs TMA compiled
        torch.testing.assert_close(out_tma_compiled, out_compiled, atol=2e-1, rtol=2e-1)

    @supported_platform
    @skip_on_cpu
    def test_large_batch_heads_grid_dimension(self, device):
        B, H, S, D = 22720, 3, 64, 32

        make_tensor = functools.partial(
            torch.randn,
            (B, H, S, D),
            device=device,
            dtype=torch.float16,
            requires_grad=True,
        )

        query, key, value = make_tensor(), make_tensor(), make_tensor()

        flex_compile = torch.compile(flex_attention, fullgraph=True, dynamic=True)
        out_compiled = flex_compile(query, key, value)

        self.assertEqual(out_compiled.shape, (B, H, S, D))

        grad_output = torch.randn_like(out_compiled)
        out_compiled.backward(grad_output)

        self.assertIsNotNone(query.grad)
        self.assertIsNotNone(key.grad)
        self.assertIsNotNone(value.grad)
        self.assertEqual(query.grad.shape, query.shape)
        self.assertEqual(key.grad.shape, key.shape)
        self.assertEqual(value.grad.shape, value.shape)

    @supported_platform
    def test_debug_flag_disables_internal_compilation(self, device):
        """Test that _FLEX_ATTENTION_DISABLE_COMPILE_DEBUG flag bypasses internal compilation."""
        import torch.nn.attention.flex_attention as fa

        original_flag = fa._FLEX_ATTENTION_DISABLE_COMPILE_DEBUG
        original_warnings_shown = fa._WARNINGS_SHOWN.copy()

        try:
            B, H, S, D = 1, 1, 128, 64
            query = torch.randn(B, H, S, D, device=device, dtype=torch.float32)
            key = torch.randn(B, H, S, D, device=device, dtype=torch.float32)
            value = torch.randn(B, H, S, D, device=device, dtype=torch.float32)

            def simple_score_mod(score, b, h, q_idx, kv_idx):
                return score

            # Test with debug flag False - should warn
            fa._FLEX_ATTENTION_DISABLE_COMPILE_DEBUG = False
            fa._WARNINGS_SHOWN.clear()

            with self.assertWarns(UserWarning) as cm:
                out_compiled = fa.flex_attention(
                    query, key, value, score_mod=simple_score_mod
                )

            self.assertIn(
                "flex_attention called without torch.compile", str(cm.warning)
            )

            # Test with debug flag True - should NOT warn
            fa._FLEX_ATTENTION_DISABLE_COMPILE_DEBUG = True

            # Should not error
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                out_debug = fa.flex_attention(
                    query, key, value, score_mod=simple_score_mod
                )

            torch.testing.assert_close(out_compiled, out_debug, rtol=1e-4, atol=1e-4)

        finally:
            fa._FLEX_ATTENTION_DISABLE_COMPILE_DEBUG = original_flag
            fa._WARNINGS_SHOWN = original_warnings_shown


class TestBlockMask(InductorTestCase):
    def setUp(self):
        super().setUp()

    @supported_platform
    def test_block_mask_attributes(self, device):
        offset = torch.zeros(8, device=device)

        def causal_mask(b, h, q, kv):
            return (q + (offset[b] * 128)) >= kv

        block_mask = create_block_mask(causal_mask, 4, 2, 2048, 2048, device=device)
        self.assertEqual(block_mask.shape, (4, 2, 2048, 2048))
        self.assertEqual(block_mask[0].shape, (1, 2, 2048, 2048))
        self.assertEqual(block_mask[0, 0].shape, (1, 1, 2048, 2048))
        self.assertEqual(block_mask.numel(), 4 * 2 * 2048 * 2048)
        self.assertEqual(block_mask.sparsity(), 46.875)
        self.assertEqual(block_mask[0].sparsity(), 46.875)
        self.assertEqual(block_mask[1, 0].sparsity(), 46.875)
        self.assertEqual(block_mask.sparsity(), block_mask[1].sparsity())

        offset = torch.arange(8, device=device)
        block_mask = create_block_mask(causal_mask, 8, 1, 2048, 2048, device=device)
        self.assertEqual(block_mask.sparsity(), 29.1015625)
        self.assertTrue(block_mask.sparsity() < block_mask[0].sparsity())
        self.assertTrue(block_mask[0].sparsity() > block_mask[1].sparsity())

    @supported_platform
    @common_utils.parametrize("BLOCK_SIZE", [32, 64, 128, 256, (32, 64), (64, 32)])
    def test_block_size_changes(self, device, BLOCK_SIZE: Union[int, tuple[int, int]]):
        B, H, Q_LEN, KV_LEN = 4, 2, 2048, 2048

        if isinstance(BLOCK_SIZE, int):
            Q_BLOCK_SIZE = BLOCK_SIZE
            KV_BLOCK_SIZE = BLOCK_SIZE
        else:
            Q_BLOCK_SIZE, KV_BLOCK_SIZE = BLOCK_SIZE

        block_mask = create_block_mask(
            noop_mask, B, H, Q_LEN, KV_LEN, BLOCK_SIZE=BLOCK_SIZE, device=device
        )

        self.assertEqual(block_mask.BLOCK_SIZE, (Q_BLOCK_SIZE, KV_BLOCK_SIZE))
        self.assertEqual(block_mask.shape, (B, H, Q_LEN, KV_LEN))

    @supported_platform
    def test_getitem(self, device):
        offset = torch.zeros(8, device=device)

        def causal_mask(b, h, q, kv):
            return (q + (offset[b] * 128)) >= kv

        block_mask = create_block_mask(causal_mask, 4, 2, 512, 512, device=device)
        assert block_mask.kv_num_blocks.shape == (4, 2, 4)
        assert block_mask.kv_indices.shape == (4, 2, 4, 4)

        # Index on batch dimension
        new_block_mask = block_mask[0]
        assert new_block_mask.kv_num_blocks.shape == (1, 2, 4)
        assert new_block_mask.kv_indices.shape == (1, 2, 4, 4)

        # Index on batch and head dimension
        new_block_mask = block_mask[0, 1]
        assert new_block_mask.kv_num_blocks.shape == (
            1,
            1,
            4,
        )
        assert new_block_mask.kv_indices.shape == (1, 1, 4, 4)

        # Index on batch and head dimension with -1 semantics
        new_block_mask = block_mask[-1, -2]
        assert new_block_mask.kv_num_blocks.shape == (
            1,
            1,
            4,
        )
        assert new_block_mask.kv_indices.shape == (1, 1, 4, 4)

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
    def test_sliced_blockmask_mask_mod_error(self, device):
        """Test that sliced BlockMask raises helpful error when used with flex_attention"""

        def causal_mask(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx

        base_mask = create_block_mask(
            causal_mask, B=1, H=1, Q_LEN=256, KV_LEN=256, device=device
        )
        sliced_mask = base_mask[:, :, 0]

        q = torch.randn(1, 1, 1, 64, device=device)
        k = torch.randn(1, 1, 256, 64, device=device)
        v = torch.randn(1, 1, 256, 64, device=device)

        compiled_fa = torch.compile(flex_attention)
        with self.assertRaisesRegex(
            RuntimeError, "Cannot use mask_mod from a sliced BlockMask"
        ):
            compiled_fa(q, k, v, block_mask=sliced_mask)

    @supported_platform
    def test_block_mask_device_change(self, device):
        device = torch.device(device)
        offset = torch.zeros(8, device=device)

        def causal_mask(b, h, q, kv):
            return (q + (offset[b] * 128)) >= kv

        block_mask = create_block_mask(causal_mask, 1, 1, 512, 512, device=device)
        assert block_mask.kv_indices.device.type == device.type
        assert block_mask.kv_num_blocks.device.type == device.type
        assert block_mask.q_indices.device.type == device.type
        assert block_mask.q_num_blocks.device.type == device.type

        block_mask = block_mask.to("cpu")
        assert block_mask.kv_indices.is_cpu
        assert block_mask.kv_num_blocks.is_cpu
        assert block_mask.q_indices.is_cpu
        assert block_mask.q_num_blocks.is_cpu

        block_mask = block_mask.to(device)
        assert block_mask.kv_indices.device.type == device.type
        assert block_mask.kv_num_blocks.device.type == device.type
        assert block_mask.q_indices.device.type == device.type
        assert block_mask.q_num_blocks.device.type == device.type

    @supported_platform
    def test_compiling_create_block_mask(self, device):
        seq = torch.arange(512, device=device) // 127

        def mask_mod(b, h, q, kv):
            return (q >= kv) & (seq[q] == seq[kv])

        block_mask = torch.compile(create_block_mask, fullgraph=True)(
            mask_mod, 1, 1, 512, 512, device=device
        )
        self.assertIsInstance(block_mask, BlockMask)
        self.assertEqual(block_mask.kv_num_blocks.shape, torch.Size((1, 1, 4)))
        self.assertEqual(block_mask.kv_indices.shape, torch.Size((1, 1, 4, 4)))

    @supported_platform
    def test_compiling_create_block_mask_no_recompile(self, device):
        def mask_mod(b, h, q, kv):
            return q >= kv

        torch._dynamo.reset()
        block_mask = torch.compile(create_block_mask)(
            mask_mod, 2, 4, 1024, 1024, device=device
        )
        self.assertIsInstance(block_mask, BlockMask)
        self.assertEqual(block_mask.kv_num_blocks.shape, torch.Size((2, 4, 8)))
        self.assertEqual(block_mask.kv_indices.shape, torch.Size((2, 4, 8, 8)))
        self.assertEqual(torch._dynamo.utils.counters["aot_autograd"]["ok"], 1)

        # automatic dynamic shapes triggered and recompilation.
        block_mask = torch.compile(create_block_mask)(
            mask_mod, 4, 8, 2048, 2048, device=device
        )
        self.assertIsInstance(block_mask, BlockMask)
        self.assertEqual(block_mask.kv_num_blocks.shape, torch.Size((4, 8, 16)))
        self.assertEqual(block_mask.kv_indices.shape, torch.Size((4, 8, 16, 16)))
        self.assertEqual(torch._dynamo.utils.counters["aot_autograd"]["ok"], 2)

        # no recompilation.
        block_mask = torch.compile(create_block_mask)(
            mask_mod, 6, 16, 3072, 3072, device=device
        )
        self.assertIsInstance(block_mask, BlockMask)
        self.assertEqual(block_mask.kv_num_blocks.shape, torch.Size((6, 16, 24)))
        self.assertEqual(block_mask.kv_indices.shape, torch.Size((6, 16, 24, 24)))
        self.assertEqual(torch._dynamo.utils.counters["aot_autograd"]["ok"], 2)

    @supported_platform
    def test_block_mask_viz(self, device):
        def causal_mask(b, h, q, kv):
            return q >= kv

        block_mask = create_block_mask(causal_mask, 1, 1, 2048, 2048, device=device)

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

        offset = torch.arange(8, device=device)

        def causal_offset_mask(b, h, q, kv):
            return (q + offset[b] * 128) >= kv

        block_mask = create_block_mask(
            causal_offset_mask, 8, 1, 2048, 2048, device=device
        )
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
    def test_from_kv_blocks(self, device, full_indices: bool):
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
    def test_block_size(self, device):
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
    def test_upcast_appropriately(self, device):
        q = torch.randn((1, 1, 128, 16), dtype=torch.float16, device=device)
        k = torch.randn((1, 1, 128, 16), dtype=torch.float16, device=device)
        v = torch.randn((1, 1, 128, 16), dtype=torch.float16, device=device)
        mass = torch.ones((1), dtype=torch.float16, device=device)

        def score_mod(score, b, h, q_idx, kv_idx):
            return score + torch.log(mass[0])

        torch.compile(flex_attention)(q, k, v, score_mod=score_mod)

    @supported_platform
    def test_init_mismatched_full_kv(self, device):
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
    def test_init_mismatched_full_q(self, device):
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
    def test_doc_mask_clamped_repro(self, device):
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
            offsets = length_to_offsets(lengths, device)
            doc_ids = _offsets_to_doc_ids_tensor(offsets)

            def doc_mask_mod(b, h, q_idx, kv_idx):
                return (
                    doc_ids[q_idx.clamp(0, doc_ids.shape[0] - 1)]
                    == doc_ids[kv_idx.clamp(0, doc_ids.shape[0] - 1)]
                )

            q, k, v = (
                torch.randn(1, 12, 1024 + i, 64, device=device) for _ in range(3)
            )
            block_mask = create_block_mask(
                doc_mask_mod, None, None, 1024 + i, 1024 + i, device=device
            )
            torch.compile(flex_attention)(q, k, v, block_mask=block_mask)

    @supported_platform
    def test_eager_tracing_correctness(self, device):
        qk_dims = 64
        v_dims = 128
        q_heads = 4
        kv_heads = 2
        seq_len = 256
        batch_size = 1

        make_tensor = functools.partial(torch.randn, device=device, dtype=torch.float16)
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
    @skip_on_xpu
    def test_create_is_cuda_graphable(self, device):
        def mask_mod(b, h, q, kv):
            return q >= kv

        g = torch.cuda.CUDAGraph()

        with torch.cuda.graph(g):
            create_block_mask(mask_mod, None, None, 256, 256)

        g.replay()

    @common_utils.parametrize("compile", [False, True])
    @supported_platform
    def test_block_mask_vs_sequence_lengths(self, device, compile):
        if compile:
            flex_attention_call = torch.compile(flex_attention)
        else:
            flex_attention_call = flex_attention

        def mask_mod(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx

        def create_inputs(S):
            q, k, v = (
                torch.randn(
                    1, 8, S, 64, dtype=torch.float16, requires_grad=True, device=device
                )
                for _ in range(3)
            )
            return q, k, v

        block_mask = create_block_mask(mask_mod, None, None, 1024, 1024, device=device)
        flex_attention_call(*create_inputs(1024), block_mask=block_mask)
        with self.assertRaisesRegex(ValueError, "block_mask was created for"):
            flex_attention_call(*create_inputs(2048), block_mask=block_mask)

        block_mask = create_block_mask(mask_mod, None, None, 1023, 1023, device=device)
        with self.assertRaisesRegex(ValueError, "block_mask was created for"):
            flex_attention_call(*create_inputs(1024), block_mask=block_mask)

    @supported_platform
    @common_utils.parametrize("full_indices", [False, True])
    def test_from_kv_blocks_without_q_computation(self, device, full_indices: bool):
        (
            kv_num_blocks,
            kv_indices,
            full_kv_num_blocks,
            full_kv_indices,
        ) = self.generate_test_inputs(full_indices, device=device)

        block_mask = BlockMask.from_kv_blocks(
            kv_num_blocks,
            kv_indices,
            full_kv_num_blocks,
            full_kv_indices,
            compute_q_blocks=False,
        )

        self.assertIsInstance(block_mask, BlockMask)
        self.assertEqual(block_mask.kv_num_blocks, kv_num_blocks)
        self.assertEqual(block_mask.kv_indices, kv_indices)

        self.assertIsNone(block_mask.q_num_blocks)
        self.assertIsNone(block_mask.q_indices)
        self.assertIsNone(block_mask.full_q_num_blocks)
        self.assertIsNone(block_mask.full_q_indices)

        if full_indices:
            self.assertEqual(block_mask.full_kv_num_blocks, full_kv_num_blocks)
            self.assertEqual(block_mask.full_kv_indices, full_kv_indices)
        else:
            self.assertIsNone(block_mask.full_kv_num_blocks)
            self.assertIsNone(block_mask.full_kv_indices)

    @supported_platform
    @skip_on_cpu
    def test_backward_error_with_none_q_indices(self, device):
        N_BLOCKS = 4
        B, H, S, D = 1, 1, 128, 64
        S_KV = N_BLOCKS * S

        kv_num_blocks = torch.tensor([[[N_BLOCKS]]], dtype=torch.int32, device=device)
        kv_indices = torch.tensor([[[[0, 1, 2, 3]]]], dtype=torch.int32, device=device)

        block_mask = BlockMask.from_kv_blocks(
            kv_num_blocks, kv_indices, compute_q_blocks=False
        )

        q = torch.randn(
            B, H, S, D, dtype=torch.float16, device=device, requires_grad=True
        )
        k = torch.randn(
            B, H, S_KV, D, dtype=torch.float16, device=device, requires_grad=True
        )
        v = torch.randn(
            B, H, S_KV, D, dtype=torch.float16, device=device, requires_grad=True
        )

        flex_compile = torch.compile(flex_attention, fullgraph=True)

        with torch.no_grad():
            out_no_grad = flex_compile(q, k, v, block_mask=block_mask)
            self.assertEqual(out_no_grad.shape, (B, H, S, D))

        # Forward pass with grad enabled should error immediately
        with self.assertRaisesRegex(
            RuntimeError,
            "BlockMask q_indices is None. Backward pass requires q_indices to be computed. "
            "Please create the BlockMask with compute_q_blocks=True",
        ):
            flex_compile(q, k, v, block_mask=block_mask)

    @supported_platform
    @skip_on_cpu
    def test_flex_attention_poisoned_rel_logits(self, device):
        B = 1
        H = 1
        S = 1025
        D = 64
        q, k, v = [
            torch.randn(B, H, S, D, requires_grad=True, device=device) for _ in range(3)
        ]
        rel_logits = torch.randn(2 * B, H, S, S, device=device)
        rel_logits[B:] = float("nan")

        def score_mod(score, b, h, q, kv):
            return score + rel_logits[b, h, q, kv]

        def causal(
            b: torch.Tensor, h: torch.Tensor, q: torch.Tensor, kv: torch.Tensor
        ) -> torch.Tensor:
            return q >= kv

        block_mask = create_block_mask(causal, B, H, S, S, device=device)
        out = torch.compile(flex_attention)(
            q, k, v, score_mod=score_mod, block_mask=block_mask
        )
        out.sum().backward()

        assert out.isfinite().all().item()
        assert q.grad.isfinite().all().item()
        assert k.grad.isfinite().all().item()
        assert v.grad.isfinite().all().item()

    @supported_platform
    @skip_on_cpu
    def test_flex_attention_poison_mod_fwd(self, device):
        """Div by score should cause our edge case handiling to NaN"""
        B = 1
        H = 1
        S = 257
        D = 16
        q, k, v = [
            torch.randn(B, H, S, D, requires_grad=True, device=device) for _ in range(3)
        ]

        def score_mod(score, b, h, q, kv):
            return 1 / score

        def causal(
            b: torch.Tensor, h: torch.Tensor, q: torch.Tensor, kv: torch.Tensor
        ) -> torch.Tensor:
            return q >= kv

        block_mask = create_block_mask(causal, B, H, S, S, device=device)
        out = torch.compile(flex_attention, backend="inductor")(
            q, k, v, score_mod=score_mod, block_mask=block_mask
        )
        out.sum().backward()
        assert out.isfinite().all().item()
        assert q.grad.isfinite().all().item()
        # assert k.grad.isfinite().all().item()
        assert v.grad.isfinite().all().item()

    @supported_platform
    @skip_on_cpu
    def test_flex_attention_poison_mod_bwd(self, device):
        """log score should cause our edge case handiling for NaN in grad score"""
        B = 1
        H = 1
        S = 257
        D = 16
        q, k, v = [
            torch.randn(B, H, S, D, requires_grad=True, device=device) for _ in range(3)
        ]

        def score_mod(score, b, h, q, kv):
            return torch.where(score > 0, torch.log(score), score)

        def causal(
            b: torch.Tensor, h: torch.Tensor, q: torch.Tensor, kv: torch.Tensor
        ) -> torch.Tensor:
            return q >= kv

        block_mask = create_block_mask(causal, B, H, S, S, device=device)
        out = torch.compile(flex_attention, backend="inductor")(
            q, k, v, score_mod=score_mod, block_mask=block_mask
        )
        out.sum().backward()
        assert out.isfinite().all().item()
        assert q.grad.isfinite().all().item()
        # assert k.grad.isfinite().all().item()
        assert v.grad.isfinite().all().item()

    @supported_platform
    @skip_on_cpu
    def test_forward_pass_with_none_q_indices(self, device):
        N_BLOCKS = 4
        B, H, S, D = 1, 1, 128, 64
        S_KV = N_BLOCKS * S

        kv_num_blocks = torch.tensor([[[N_BLOCKS]]], dtype=torch.int32, device=device)
        kv_indices = torch.tensor([[[[0, 1, 2, 3]]]], dtype=torch.int32, device=device)

        block_mask = BlockMask.from_kv_blocks(
            kv_num_blocks, kv_indices, compute_q_blocks=False
        )

        q = torch.randn(
            B,
            H,
            S,
            D,
            dtype=torch.float16,
            device=device,
        )
        k = torch.randn(
            B,
            H,
            S_KV,
            D,
            dtype=torch.float16,
            device=device,
        )
        v = torch.randn(
            B,
            H,
            S_KV,
            D,
            dtype=torch.float16,
            device=device,
        )

        flex_compile = torch.compile(flex_attention, fullgraph=True)
        out = flex_compile(q, k, v, block_mask=block_mask)

        self.assertEqual(out.shape, (B, H, S, D))
        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.dtype, torch.float16)

    @supported_platform
    def test_block_mask_operations_with_none_q_indices(self, device):
        kv_num_blocks = torch.tensor([[[4]]], dtype=torch.int32, device=device)
        kv_indices = torch.tensor([[[[0, 1, 2, 3]]]], dtype=torch.int32, device=device)

        block_mask = BlockMask.from_kv_blocks(
            kv_num_blocks, kv_indices, compute_q_blocks=False
        )
        self.assertEqual(block_mask.shape, (1, 1, 128, 512))
        self.assertEqual(block_mask.BLOCK_SIZE, (128, 128))

        sliced_mask = block_mask[0]
        self.assertEqual(sliced_mask.shape, (1, 1, 128, 512))
        self.assertIsNone(sliced_mask.q_indices)
        self.assertIsNone(sliced_mask.q_num_blocks)

        # Test device movement
        if device != "cpu":
            cpu_mask = block_mask.to("cpu")
            self.assertEqual(cpu_mask.kv_num_blocks.device.type, "cpu")
            self.assertIsNone(cpu_mask.q_indices)

    @supported_platform
    @skip_on_cpu
    def test_broadcasted_head_block_mask(self, device):
        torch.manual_seed(42)

        def causal_mask(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx

        def get_mask_mod_with_offset(mask_mod, offset_tensor):
            def _mask_mod(b, h, q, kv):
                return mask_mod(b, h, q + offset_tensor, kv)

            return _mask_mod

        B, T, H, D, current_pos = 4, 512, 8, 64, 128
        dtype = torch.float32

        q = torch.randn(B, H, 1, D, device=device, dtype=dtype)
        k_cache = torch.randn(B, H, T, D, device=device, dtype=dtype)
        v_cache = torch.randn(B, H, T, D, device=device, dtype=dtype)

        # Keep future tokens tiny to avoid numerical issues when using full caches
        k_cache[:, :, current_pos + 1 :, :] = (
            torch.randn_like(k_cache[:, :, current_pos + 1 :, :]) * 1e-10
        )
        v_cache[:, :, current_pos + 1 :, :] = (
            torch.randn_like(v_cache[:, :, current_pos + 1 :, :]) * 1e-10
        )

        k_cropped = k_cache[:, :, : current_pos + 1, :]
        v_cropped = v_cache[:, :, : current_pos + 1, :]
        sdpa_output = torch.nn.functional.scaled_dot_product_attention(
            q, k_cropped, v_cropped, attn_mask=None
        )

        base_mask = create_block_mask(
            causal_mask,
            B=B,
            H=None,  # broadcast across heads
            Q_LEN=T,
            KV_LEN=T,
            device=device,
            _compile=True,
        )

        q_block_size = base_mask.BLOCK_SIZE[0]
        block_offset = current_pos // q_block_size
        mask_slice = base_mask[:, :, block_offset]

        offset_tensor = torch.tensor(current_pos, device=device)
        mask_slice.mask_mod = get_mask_mod_with_offset(
            base_mask.mask_mod, offset_tensor
        )
        mask_slice.seq_lengths = (1, mask_slice.seq_lengths[1])

        fa = torch.compile(flex_attention, dynamic=True)
        flex_output = fa(q, k_cache, v_cache, block_mask=mask_slice)

        self.assertEqual(flex_output, sdpa_output, atol=1e-3, rtol=1e-3)

    @supported_platform
    def test_pytree_flatten_unflatten(self, device):
        """Test that BlockMask can be correctly flattened and unflattened using class methods."""

        def causal_mask(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx

        # Create a BlockMask with various attributes set
        block_mask = create_block_mask(
            causal_mask, B=2, H=4, Q_LEN=512, KV_LEN=512, device=device
        )

        # Flatten and unflatten using class methods
        tensors, context = block_mask._flatten()
        reconstructed_mask = BlockMask._unflatten(tensors, context)

        # Verify the reconstructed mask has the same attributes
        self.assertEqual(reconstructed_mask.shape, block_mask.shape)
        self.assertEqual(reconstructed_mask.sparsity(), block_mask.sparsity())

        # Verify all tensor attributes are equal (using _TENSOR_ATTRS)
        for attr_name in BlockMask._TENSOR_ATTRS:
            original_value = getattr(block_mask, attr_name)
            reconstructed_value = getattr(reconstructed_mask, attr_name)

            if original_value is None:
                self.assertIsNone(
                    reconstructed_value,
                    f"Tensor attribute {attr_name} should be None but got {reconstructed_value}",
                )
            else:
                self.assertIsInstance(
                    original_value,
                    torch.Tensor,
                    f"Expected {attr_name} to be a Tensor",
                )
                self.assertTrue(
                    torch.equal(original_value, reconstructed_value),
                    f"Tensor attribute {attr_name} not equal after reconstruction",
                )

        # Verify all context attributes are equal (using _CONTEXT_ATTRS)
        for attr_name in BlockMask._CONTEXT_ATTRS:
            original_value = getattr(block_mask, attr_name)
            reconstructed_value = getattr(reconstructed_mask, attr_name)

            self.assertEqual(
                original_value,
                reconstructed_value,
                f"Context attribute {attr_name} not equal after reconstruction",
            )

    @supported_platform
    def test_pytree_flatten_with_keys(self, device):
        """Test that BlockMask._flatten_with_keys works correctly for tracing."""

        def causal_mask(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx

        block_mask = create_block_mask(
            causal_mask, B=2, H=4, Q_LEN=512, KV_LEN=512, device=device
        )

        tensors_with_keys, context_with_keys = block_mask._flatten_with_keys()

        self.assertEqual(len(tensors_with_keys), len(BlockMask._TENSOR_ATTRS))
        self.assertEqual(len(context_with_keys), len(BlockMask._CONTEXT_ATTRS))

        from torch.utils._pytree import GetAttrKey

        for key, _tensor in tensors_with_keys:
            self.assertIsInstance(key, GetAttrKey)
            self.assertIsNotNone(key)

        for key, _value in context_with_keys:
            self.assertIsInstance(key, GetAttrKey)
            self.assertIsNotNone(key)

    @supported_platform
    def test_pytree_preserves_new_attributes(self, device):
        """
        Test that BlockMask._TENSOR_ATTRS and _CONTEXT_ATTRS are correctly defined
        and that flatten/unflatten preserves all attributes in these lists.

        """

        def causal_mask(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx

        block_mask = create_block_mask(
            causal_mask, B=2, H=4, Q_LEN=512, KV_LEN=512, device=device
        )

        # Flatten and unflatten using class methods
        tensors, context = block_mask._flatten()
        reconstructed_mask = BlockMask._unflatten(tensors, context)

        # Verify the number of tensors and context values matches the attribute lists
        self.assertEqual(
            len(tensors),
            len(BlockMask._TENSOR_ATTRS),
            "Number of tensors should match _TENSOR_ATTRS length",
        )
        self.assertEqual(
            len(context),
            len(BlockMask._CONTEXT_ATTRS),
            "Number of context values should match _CONTEXT_ATTRS length",
        )

        # Verify all attributes from the lists exist and are equal after reconstruction
        for attr_name in BlockMask._TENSOR_ATTRS + BlockMask._CONTEXT_ATTRS:
            self.assertTrue(
                hasattr(reconstructed_mask, attr_name),
                f"Reconstructed mask missing attribute: {attr_name}",
            )
            original_value = getattr(block_mask, attr_name)
            reconstructed_value = getattr(reconstructed_mask, attr_name)

            if isinstance(original_value, torch.Tensor):
                self.assertTrue(
                    torch.equal(original_value, reconstructed_value),
                    f"Tensor attribute {attr_name} not equal after reconstruction",
                )
            elif original_value is None:
                self.assertIsNone(
                    reconstructed_value,
                    f"Attribute {attr_name} should be None but got {reconstructed_value}",
                )
            else:
                self.assertEqual(
                    original_value,
                    reconstructed_value,
                    f"Attribute {attr_name} not equal after reconstruction",
                )


@large_tensor_test_class("2GB", device=test_device[0])
class TestPagedAttention(InductorTestCase):
    def setUp(self):
        super().setUp()
        skipCPUIf(
            LONG_COMPILATION_ON_CPU,
            "skip UT for CPU due to long compilation time found in CI",
        )

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

    def allocate_page_cache(self, n_pages: int, page_size: int, device: str):
        max_batch_size = 3
        paged_cache = PagedAttention(n_pages, page_size, max_batch_size, device=device)
        return paged_cache

    def cdiv(self, x, y):
        return (x + y - 1) // y

    def roundup(self, x, y):
        return (x + y - 1) // y * y

    @supported_platform
    def test_page_allocation(self, device):
        n_pages, page_size = 12, 4
        paged_cache = self.allocate_page_cache(n_pages, page_size, device=device)

        batch_reserve(paged_cache, torch.tensor([8, 24, 16]))

        with self.assertRaisesRegex(
            AssertionError, "requested 2 pages but there are only 0 empty pages"
        ):
            paged_cache.reserve(
                torch.tensor([0], device=device),
                torch.tensor([16], device=device),
            )

        paged_cache.erase(torch.tensor([1], device=device))
        paged_cache.reserve(
            torch.tensor([0], device=device),
            torch.tensor([16], device=device),
        )

    @supported_platform
    def test_allocate(self, device):
        n_pages, page_size = 12, 4
        paged_cache = self.allocate_page_cache(n_pages, page_size, device=device)

        target_seq_len = torch.tensor([3, 11, 8])
        batch_reserve(paged_cache, target_seq_len)

        expected_allocated_pages = self.cdiv(target_seq_len, page_size).sum()
        self.assertEqual(paged_cache.capacity, self.roundup(target_seq_len, page_size))
        self.assertEqual(
            len(paged_cache.empty_pages), n_pages - expected_allocated_pages
        )

        # deallocate batch 1
        paged_cache.erase(torch.tensor([1], device=device))
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
    def test_convert_logical_block_mask(self, device):
        n_pages, page_size, max_batch_size, max_seq_len = 8, 128, 2, 512
        paged_cache = PagedAttention(n_pages, page_size, max_batch_size, device=device)

        batch_reserve(paged_cache, torch.tensor([100, 200], device=device))
        batch_reserve(paged_cache, torch.tensor([150, 300], device=device))
        batch_reserve(paged_cache, torch.tensor([300, 512], device=device))
        batch_reserve(paged_cache, torch.tensor([512, 512], device=device))

        expected_page_table = torch.tensor(
            [[0, 3, 5, 7, -1, -1, -1, -1], [2, 1, 4, 6, -1, -1, -1, -1]],
            device=device,
        )
        self.assertEqual(
            paged_cache.capacity,
            torch.tensor([512, 512], device=device),
        )
        self.assertEqual(paged_cache.page_table, expected_page_table)

        # Get a block mask
        def causal_mask(b, h, q, kv):
            return q >= kv

        block_mask = create_block_mask(
            causal_mask, max_batch_size, 1, max_seq_len, max_seq_len, device=device
        )
        kv_len_tensor = torch.full(
            (max_batch_size,), max_seq_len, device=device, dtype=torch.int64
        )
        new_block_mask = paged_cache.convert_logical_block_mask(
            block_mask, kv_len=kv_len_tensor
        )

        zeros = [0, 0, 0, 0]
        # Check that the new block mask is correct
        expected_kv_num_blocks = torch.tensor(
            [[[1, 1, 1, 1]], [[1, 1, 1, 1]]], device=device, dtype=torch.int32
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
            device=device,
            dtype=torch.int32,
        )
        expected_full_kv_num_blocks = torch.tensor(
            [[[0, 1, 2, 3]], [[0, 1, 2, 3]]], device=device, dtype=torch.int32
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
            device=device,
            dtype=torch.int32,
        )
        self.assertEqual(new_block_mask.kv_num_blocks, expected_kv_num_blocks)
        self.assertEqual(new_block_mask.kv_indices, expected_kv_indices)
        self.assertEqual(new_block_mask.full_kv_num_blocks, expected_full_kv_num_blocks)
        self.assertEqual(new_block_mask.full_kv_indices, expected_full_kv_indices)

    @supported_platform
    def test_convert_mask_mod(self, device):
        n_pages, page_size, max_batch_size = 8, 128, 2
        paged_cache = PagedAttention(n_pages, page_size, max_batch_size, device=device)

        batch_reserve(paged_cache, torch.tensor([100, 200], device=device))
        batch_reserve(paged_cache, torch.tensor([150, 300], device=device))
        batch_reserve(paged_cache, torch.tensor([300, 512], device=device))
        batch_reserve(paged_cache, torch.tensor([512, 512], device=device))

        expected_page_table = torch.tensor(
            [[0, 3, 5, 7, -1, -1, -1, -1], [2, 1, 4, 6, -1, -1, -1, -1]],
            device=device,
        )
        self.assertEqual(
            paged_cache.capacity,
            torch.tensor([512, 512], device=device),
        )
        self.assertEqual(paged_cache.page_table, expected_page_table)

        expected_physical_to_logical = torch.tensor(
            [[0, -1, -1, 1, -1, 2, -1, 3], [-1, 1, 0, -1, 2, -1, 3, -1]],
            device=device,
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
    def test_update(self, device):
        dtype = torch.float32

        n_pages, page_size, max_batch_size, max_seq_len = 6, 2, 2, 6
        paged_cache = PagedAttention(n_pages, page_size, max_batch_size, device=device)

        n_heads, head_dim = 2, 3
        cache_shape = (1, n_heads, n_pages * page_size, head_dim)
        k_cache = torch.zeros(cache_shape, dtype=dtype, device=device)

        batch_reserve(paged_cache, torch.tensor([1, 3], device=device))
        batch_reserve(paged_cache, torch.tensor([4, 5], device=device))
        batch_reserve(paged_cache, torch.tensor([6, 6], device=device))

        expected_page_table = torch.tensor(
            [[0, 3, 5, -1, -1, -1], [2, 1, 4, -1, -1, -1]],
            device=device,
        )
        self.assertEqual(paged_cache.page_table, expected_page_table)

        batch_idx = torch.arange(max_batch_size, device=device, dtype=torch.int32)
        input_pos = (
            torch.arange(max_seq_len, device=device, dtype=torch.int32)
            .unsqueeze(0)
            .expand(max_batch_size, max_seq_len)
        )
        k = torch.arange(
            max_batch_size * n_heads * max_seq_len * head_dim,
            device=device,
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
            device=device,
            dtype=dtype,
        )
        self.assertEqual(k_cache, expected_cache)

    @supported_platform
    @dtypes(*device_configs["cpu"].dtypes)
    @dtypesIfCUDA(*device_configs["cuda"].dtypes)
    @dtypesIfXPU(*device_configs["xpu"].dtypes)
    @common_utils.parametrize("score_mod", test_score_mods)
    def test_paged_builtin_score_mods(
        self, device, dtype: torch.dtype, score_mod: Callable
    ):
        n_pages, page_size, max_batch_size, max_seq_len = 32, 128, 4, 512
        n_heads, head_dim = 4, 16

        def causal_mask(b, h, q, kv):
            return q >= kv

        block_mask = create_block_mask(
            causal_mask, max_batch_size, 1, max_seq_len, max_seq_len, device=device
        )
        q = torch.randn(
            max_batch_size,
            n_heads,
            max_seq_len,
            head_dim,
            device=device,
            dtype=dtype,
            requires_grad=False,
        )
        k = torch.randn(
            max_batch_size,
            n_heads,
            max_seq_len,
            head_dim,
            device=device,
            dtype=dtype,
            requires_grad=False,
        )
        v = torch.randn(
            max_batch_size,
            n_heads,
            max_seq_len,
            head_dim,
            device=device,
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

        paged_cache = PagedAttention(n_pages, page_size, max_batch_size, device=device)
        batch_reserve(paged_cache, torch.tensor([100, 200, 50, 300], device=device))
        batch_reserve(paged_cache, torch.tensor([100, 512, 300, 300], device=device))
        batch_reserve(paged_cache, torch.tensor([512, 512, 300, 300], device=device))
        batch_reserve(paged_cache, torch.tensor([512, 512, 512, 300], device=device))
        batch_reserve(paged_cache, torch.tensor([512, 512, 512, 512], device=device))

        batch_idx = torch.arange(max_batch_size, device=device, dtype=torch.int32)
        input_pos = (
            torch.arange(max_seq_len, device=device, dtype=torch.int32)
            .unsqueeze(0)
            .expand(max_batch_size, max_seq_len)
        )
        paged_cache.assign(batch_idx, input_pos, k, v, k_cache, v_cache)

        kv_len_tensor = torch.full(
            (max_batch_size,), max_seq_len, device=device, dtype=torch.int64
        )
        new_block_mask = paged_cache.convert_logical_block_mask(
            block_mask, kv_len=kv_len_tensor
        )

        compiled_sdpa = torch.compile(
            create_attention(
                paged_cache.get_score_mod(score_mod, kv_len=kv_len_tensor),
                block_mask,
                enable_gqa=False,
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


supports_learnable_bias = unittest.skipUnless(
    (
        (torch.cuda.is_available() and has_triton())
        and (torch.cuda.get_device_capability() >= (8, 0) or torch.version.hip)
    ),
    "Requires Triton + A100 or Triton + ROCm",
)


@supports_learnable_bias
@large_tensor_test_class("2GB", device=test_device[0])
class TestLearnableBiases(InductorTestCase):
    def setUp(self):
        super().setUp()
        skipCPUIf(
            LONG_COMPILATION_ON_CPU,
            "skip UT for CPU due to long compilation time found in CI",
        )
        self.dtype = torch.float32
        self.atol = 3e-2
        self.rtol = 3e-2

    def _init_tensors(self, params: Params, device: str):
        make_tensor = functools.partial(
            torch.randn,
            (params.batch_size, params.num_heads, params.seq_length, params.head_dim),
            device=device,
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
        backwards_grad = torch.randn_like(out_eager, device="cpu").to(out_eager.device)
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

    @skip_on_cpu
    @common_utils.parametrize(
        "params", get_params(device_configs["cuda"].dtypes), name_fn=lambda x: f"{x}"
    )
    @common_utils.parametrize("mode", ["default", "max-autotune-no-cudagraphs"])
    def test_relative_1d_bias(self, device, params, mode: str):
        query, key, value = self._init_tensors(params, device=device)
        bias = torch.randn(
            2 * params.seq_length,
            device=device,
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

    @skip_on_cpu
    @common_utils.parametrize(
        "params", get_params(device_configs["cuda"].dtypes), name_fn=lambda x: f"{x}"
    )
    def test_absolute_2d_bias(self, device, params):
        query, key, value = self._init_tensors(params, device=device)
        bias = torch.randn(
            params.seq_length,
            params.seq_length,
            device=device,
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

    @skip_on_cpu
    @common_utils.parametrize(
        "params", get_params(device_configs["cuda"].dtypes), name_fn=lambda x: f"{x}"
    )
    def test_head_specific_bias(self, device, params):
        query, key, value = self._init_tensors(params, device=device)
        bias = torch.randn(
            params.num_heads,
            params.seq_length,
            params.seq_length,
            device=device,
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

    @skip_on_cpu
    @common_utils.parametrize(
        "params", get_params(device_configs["cuda"].dtypes), name_fn=lambda x: f"{x}"
    )
    def test_batch_head_bias(self, device, params):
        query, key, value = self._init_tensors(params, device=device)
        bias = torch.randn(
            params.batch_size,
            params.num_heads,
            params.seq_length,
            params.seq_length,
            device=device,
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

    @skip_on_cpu
    @common_utils.parametrize(
        "params", get_params(device_configs["cuda"].dtypes), name_fn=lambda x: f"{x}"
    )
    def test_multiplicative_bias(self, device, params):
        query, key, value = self._init_tensors(params, device=device)
        bias = torch.randn(
            params.seq_length,
            device=device,
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

    @skip_on_cpu
    @common_utils.parametrize(
        "params", get_params(device_configs["cuda"].dtypes), name_fn=lambda x: f"{x}"
    )
    def test_local_window_bias(self, device, params):
        query, key, value = self._init_tensors(params, device=device)
        window_size = 8
        bias = torch.randn(
            2 * window_size + 1,
            device=device,
            dtype=torch.float32,
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

    @skip_on_cpu
    @common_utils.parametrize(
        "params", get_params(device_configs["cuda"].dtypes), name_fn=lambda x: f"{x}"
    )
    def test_global_tokens_bias(self, device, params):
        query, key, value = self._init_tensors(params, device=device)
        bias = torch.randn(
            params.seq_length,
            device=device,
            dtype=torch.float32,
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

    @skip_on_cpu
    @common_utils.parametrize(
        "params", get_params(device_configs["cuda"].dtypes), name_fn=lambda x: f"{x}"
    )
    def test_weird_bias(self, device, params):
        query, key, value = self._init_tensors(params, device=device)
        bias = torch.randn(
            params.batch_size,
            params.num_heads,
            4,
            params.seq_length,
            device=device,
            dtype=params.dtype,
            requires_grad=True,
        )
        which_bias = torch.tensor(0, device=device)

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

    @skip_on_cpu
    @common_utils.parametrize(
        "params", get_params(device_configs["cuda"].dtypes), name_fn=lambda x: f"{x}"
    )
    def test_indirect_bias(self, device, params):
        query, key, value = self._init_tensors(params, device=device)
        bias = torch.randn(
            params.seq_length,
            device=device,
            dtype=params.dtype,
            requires_grad=True,
        )

        offset = torch.randint(
            0,
            params.seq_length,
            (params.seq_length,),
            device=device,
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

    @skip_on_cpu
    @common_utils.parametrize(
        "params", get_params(device_configs["cuda"].dtypes), name_fn=lambda x: f"{x}"
    )
    @common_utils.parametrize("mode", ["default", "max-autotune-no-cudagraphs"])
    def test_symmetric_bias(self, device, params, mode: str):
        query, key, value = self._init_tensors(params, device=device)
        bias = torch.randn(
            params.seq_length,
            device=device,
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
            torch._inductor.exc.InductorError,
            "Using multiple indexing operations on the same tensor that requires gradients",
        ):
            self._check_outputs_and_grads(
                out_eager,
                out_compiled,
                out_gold,
                (query, key, value, bias),
            )

    @skip_on_cpu
    @common_utils.parametrize(
        "params", get_params(device_configs["cuda"].dtypes), name_fn=lambda x: f"{x}"
    )
    def test_flipped_indexed_bias(self, device, params):
        query, key, value = self._init_tensors(params, device=device)
        bias = torch.randn(
            params.seq_length,
            params.seq_length,
            device=device,
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

    @skip_on_cpu
    @common_utils.parametrize(
        "params", get_params(device_configs["cuda"].dtypes), name_fn=lambda x: f"{x}"
    )
    @common_utils.parametrize("mode", ["default", "max-autotune-no-cudagraphs"])
    def test_head_specific_gate(self, device, params, mode: str):
        query, key, value = self._init_tensors(params, device=device)
        gate_score = torch.randn(
            params.num_heads,
            device=device,
            dtype=torch.float32,
            requires_grad=True,
        )

        def bias_func(score, b, h, q_idx, kv_idx):
            return score * torch.sigmoid(gate_score[h])

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

    @skip_on_cpu
    @common_utils.parametrize(
        "params", get_params(device_configs["cuda"].dtypes), name_fn=lambda x: f"{x}"
    )
    def test_distinct_biases(self, device, params):
        query, key, value = self._init_tensors(params, device=device)
        # Create two separate bias tensors
        bias1 = torch.randn(
            params.seq_length,
            device=device,
            dtype=params.dtype,
            requires_grad=True,
        )
        bias2 = torch.randn(
            params.seq_length,
            device=device,
            dtype=torch.float32,
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

    @skip_on_cpu
    @common_utils.parametrize(
        "params", get_params(device_configs["cuda"].dtypes), name_fn=lambda x: f"{x}"
    )
    @torch.compile
    def test_learnable_bias_global_compiled(self, device, params):
        batch_size = 1
        num_heads = 1
        seq_len = 128
        head_dim = 16
        d_model = num_heads * head_dim

        query = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
        key = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
        value = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)

        out_proj = nn.Linear(d_model, d_model, device=device)

        query.requires_grad = True
        key.requires_grad = True
        value.requires_grad = True

        bias = torch.randn(
            batch_size,
            num_heads,
            seq_len,
            seq_len,
            device=device,
            requires_grad=True,
        )

        def bias_mod(score, b, h, q_idx, kv_idx):
            return score + bias[b, h, q_idx, kv_idx]

        out = flex_attention(
            query=query,
            key=key,
            value=value,
            score_mod=bias_mod,
        )
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)

        attn_output = out_proj(out)
        random_target = torch.randn(batch_size, seq_len, d_model, device=device)
        loss = torch.nn.functional.mse_loss(attn_output, random_target)
        loss.backward()

        assert bias.grad, "No gradient computed for bias"
        assert torch.any(bias.grad != 0), "Gradient for bias is 0"

    @skip_on_cpu
    def test_backprop_error_case(self, device):
        @torch.compile()
        def test(x, y):
            # Materialize a bias matrix
            B, L, device = x.shape[0], x.shape[1], x.device
            b = torch.arange(B, device=device, dtype=torch.long).view(B, 1, 1)
            q_idx = torch.arange(L, device=device, dtype=torch.long).view(1, L, 1)
            kv_idx = torch.arange(L, device=device, dtype=torch.long).view(1, 1, L)
            bias_mat = y[b, q_idx] + y[b, kv_idx]  # (B, L, L)

            # Dummy score_mod retrieving bias values
            def score_mod(score, b, h, q_idx, kv_idx):
                return score + bias_mat[b, q_idx, kv_idx]

            x_ = x[:, :, None].repeat(1, 1, 16, 1)
            # torch._dynamo.graph_break()
            return flex_attention(x_, x_, x_, score_mod=score_mod)

        B, L, D = 2, 16, 64

        x = torch.randn(B, L, D, device=device, requires_grad=True)
        y = torch.randn(B, L, device=device, requires_grad=True)

        _ = test(x, y).mean().backward()

        assert x.grad.norm() > 0
        assert y.grad.norm() > 0

    @skip_on_cpu
    @common_utils.parametrize(
        "params", get_params(device_configs["cuda"].dtypes), name_fn=lambda x: f"{x}"
    )
    def test_relative_1d_bias_only_grad(self, device, params):
        query, key, value = self._init_tensors(params, device=device)
        query = query.detach().requires_grad_(False)
        key = key.detach().requires_grad_(False)
        value = value.detach().requires_grad_(False)

        # Only bias requires gradients
        bias = torch.randn(
            2 * params.seq_length,
            device=device,
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

    def _test_flex_attention_with_dynamic_max_autotune(self, device):
        query = torch.randn(2, 16, 512, 64, device=device)
        key = torch.randn(2, 16, 512, 64, device=device)
        value = torch.randn(2, 16, 512, 64, device=device)
        query.requires_grad = True
        key.requires_grad = True
        value.requires_grad = True

        shape = (2, 16, 512, 16, 512, 64)
        B, Hq, M, Hkv, N, D = shape

        score_mod = _generate_alibi_bias(8)

        def causal(b, h, m, n):
            return m >= n

        mask_shape = (1, 1, M, N)
        block_mask = torch.compile(create_block_mask)(
            causal, *mask_shape, device=device
        )

        compiled_sdpa = torch.compile(
            flex_attention, dynamic=True, mode="max-autotune-no-cudagraphs"
        )

        out = compiled_sdpa(
            query=query,
            key=key,
            value=value,
            score_mod=score_mod,
            block_mask=block_mask,
            enable_gqa=True,
            kernel_options=None,
        )
        out.sum().backward()

        self.assertEqual(
            out.shape, query.shape, f"Expected shape {query.shape}, got {out.shape}"
        )

    @skip_on_cpu
    def test_flex_attention_with_dynamic_max_autotune(self, device):
        self._test_flex_attention_with_dynamic_max_autotune(device)

    @skip_on_cpu
    @torch._inductor.config.patch("graph_partition", True)
    def test_flex_attention_with_dynamic_max_autotune_graph_partition(self, device):
        self._test_flex_attention_with_dynamic_max_autotune(device)

    @skip_on_cpu
    def test_flex_attention_logging(self, device):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "flex_attention_configs")

            with patch.dict(
                os.environ, {"TORCHINDUCTOR_FLEX_ATTENTION_LOGGING_FILE": log_file}
            ):
                query = torch.randn(
                    1,
                    2,
                    128,
                    64,
                    device=device,
                    dtype=torch.float16,
                    requires_grad=True,
                )
                key = torch.randn(
                    1,
                    2,
                    128,
                    64,
                    device=device,
                    dtype=torch.float16,
                    requires_grad=True,
                )
                value = torch.randn(
                    1,
                    2,
                    128,
                    64,
                    device=device,
                    dtype=torch.float16,
                    requires_grad=True,
                )

                def score_mod(score, b, h, q_idx, kv_idx):
                    return score * 2

                def causal_mask(b, h, q_idx, kv_idx):
                    return q_idx >= kv_idx

                block_mask = torch.compile(create_block_mask)(
                    causal_mask, 1, 1, 128, 128, device=device
                )

                compiled_flex = torch.compile(
                    flex_attention, mode="max-autotune-no-cudagraphs"
                )

                out = compiled_flex(
                    query=query,
                    key=key,
                    value=value,
                    score_mod=score_mod,
                    block_mask=block_mask,
                )

                out.sum().backward()

                json_file = log_file + ".json"
                self.assertTrue(
                    os.path.exists(json_file), f"Log file {json_file} was not created"
                )

                with open(json_file) as f:
                    log_data = json.load(f)

                self.assertIsInstance(log_data, list)
                self.assertEqual(len(log_data), 2)

                keys_seen = [next(iter(entry.keys())) for entry in log_data]

                expected_fwd_key = "('forward', 1, 2, 2, 128, 128, 64, 64)"
                expected_bwd_key = "('backward', 1, 2, 2, 128, 128, 64, 64)"

                self.assertIn(expected_fwd_key, keys_seen)
                self.assertIn(expected_bwd_key, keys_seen)

                for entry in log_data:
                    self.assertIsInstance(entry, dict)
                    self.assertEqual(len(entry), 1)

                    dims_key = next(iter(entry.keys()))
                    choices = entry[dims_key]

                    kernel_type = eval(dims_key)[0]

                    self.assertIsInstance(choices, list)
                    self.assertGreater(len(choices), 0)

                    for i, choice in enumerate(choices):
                        self.assertIn("type", choice)
                        self.assertIn("time", choice)

                        if choice["type"] == "triton":
                            self.assertIn("num_warps", choice)
                            self.assertIn("num_stages", choice)

                            if kernel_type == "forward":
                                self.assertIn("BLOCK_M", choice)
                                self.assertIn("BLOCK_N", choice)
                                self.assertNotIn("BLOCK_M1", choice)
                            elif kernel_type == "backward":
                                self.assertIn("BLOCK_M1", choice)
                                self.assertIn("BLOCK_N1", choice)
                                self.assertIn("BLOCK_M2", choice)
                                self.assertIn("BLOCK_N2", choice)
                                self.assertNotIn("BLOCK_M", choice)
                                self.assertNotIn("BLOCK_N", choice)

                        if i > 0:
                            self.assertLessEqual(choices[0]["time"], choice["time"])

    @skip_on_cpu
    def test_inspect_bug(self, device):
        # https://github.com/pytorch/pytorch/issues/139374
        def sliding_window(b, h, q_idx, kv_idx, val):
            return (q_idx - kv_idx).abs() < val

        sliding_window2 = functools.partial(
            sliding_window, val=torch.randn((), device=device)
        )
        opt_fn = torch.compile(create_block_mask, fullgraph=True)
        create_block_mask(sliding_window2, None, None, 1024, 1024, device=device)
        # checks that the compile is working
        opt_fn(sliding_window2, None, None, 1024, 1024, device=device)

    @supported_platform
    @skip_on_cpu
    def test_head_bias_req_grad(self, device):
        B, H, S, D = 1, 4, 256, 64
        bias = torch.randn(H, device=device, dtype=torch.float16, requires_grad=True)

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
            device,
        )

    @supported_platform
    @skip_on_cpu
    def test_comparison_vs_sdpa_with_learnable_bias(self, device):
        # 1-dimensional bias:
        B, H, S, D = 1, 1, 256, 64
        bias = torch.randn(
            2 * S, device=device, dtype=torch.float16, requires_grad=True
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
            device,
        )

        # 2-dimensional bias:
        B, H, S, D = 1, 1, 256, 64
        bias = torch.randn(S, S, device=device, dtype=torch.float16, requires_grad=True)

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
            device,
        )

        # 2-dimensional bias + index multiple
        B, H, S, D = 1, 1, 256, 64
        bias = torch.randn(S, S, device=device, dtype=torch.float16, requires_grad=True)

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
            device,
        )

        # 2-dimensional bias + transposed:
        B, H, S, D = 1, 1, 256, 64
        bias = torch.randn(S, S, device=device, dtype=torch.float16, requires_grad=True)

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
            device,
        )

        # 3-dimensional bias + transposed
        B, H, S, D = 4, 8, 256, 64
        bias = torch.randn(
            H, S, S, device=device, dtype=torch.float16, requires_grad=True
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
            device,
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
        device,
    ):
        make_tensor = functools.partial(
            torch.ones,
            (B, H, S, D),
            device=device,
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


instantiate_device_type_tests(
    TestFlexAttention, globals(), only_for=test_device, allow_xpu=True
)
instantiate_device_type_tests(
    TestPagedAttention, globals(), only_for=test_device, allow_xpu=True
)
instantiate_device_type_tests(
    TestBlockMask,
    globals(),
    only_for=(test_device[0] if HAS_GPU else "cuda",),
    allow_xpu=True,
)
instantiate_device_type_tests(
    TestLearnableBiases, globals(), only_for=test_device, allow_xpu=True
)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
