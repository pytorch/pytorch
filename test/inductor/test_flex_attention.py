# Owner(s): ["module: inductor"]

import contextlib
import functools
import gc
import json
import math
import os
import random
import string
import tempfile
import types
import unittest
import warnings
from collections import namedtuple
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import product
from typing import TypeVar
from unittest import expectedFailure, mock, skip, skipUnless
from unittest.mock import patch

import sympy

import torch
import torch.nn as nn
from torch._dynamo.testing import CompileCounterWithBackend, normalize_gm
from torch._higher_order_ops.inline_asm_elementwise import inline_asm_elementwise
from torch._inductor import config, metrics
from torch._inductor.exc import InductorError
from torch._inductor.runtime.triton_compat import HAS_WARP_SPEC
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.utils import run_and_get_code
from torch.nn.attention import SDPBackend
from torch.nn.attention.experimental._paged_attention import PagedAttention
from torch.nn.attention.flex_attention import (
    _apply_kernel_options,
    _compute_dq_write_order_from_block_mask,
    _create_empty_block_mask,
    _DEFAULT_SPARSE_BLOCK_SIZE,
    _identity,
    _mask_mod_signature,
    _score_mod_signature,
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
from torch.testing._internal.common_cuda import (
    PLATFORM_SUPPORTS_BF16,
    PLATFORM_SUPPORTS_FP8,
)
from torch.testing._internal.common_device_type import (
    dtypes,
    dtypesIfCUDA,
    dtypesIfXPU,
    E4M3_MAX_POS,
    e4m3_type,
    flex_attention_supported_platform as supported_platform,
    instantiate_device_type_tests,
    IS_FLEX_ATTENTION_CPU_PLATFORM_SUPPORTED as TEST_ON_CPU,
    IS_FLEX_ATTENTION_CUDA_PLATFORM_SUPPORTED as TEST_ON_CUDA,
    IS_FLEX_ATTENTION_XPU_PLATFORM_SUPPORTED as TEST_ON_XPU,
    largeTensorTest,
    skipCPUIf,
    skipCUDAIf,
    skipMPSIf,
    skipXPUIf,
)
from torch.testing._internal.common_quantized import _snr
from torch.testing._internal.common_utils import (
    IS_LINUX,
    isRocmArchAnyOf,
    MI200_ARCH,
    serialTest,
    skipIfRocm,
    skipIfRocmArch,
    TEST_MULTIACCELERATOR,
    TEST_WITH_ROCM,
    TEST_WITH_SLOW,
)
from torch.testing._internal.inductor_utils import (
    HAS_GPU,
    HAS_MPS,
    running_on_tdm_device,
)
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


_PRIOR_FP32_MATMUL_PRECISION: str | None = None


def setUpModule():
    global _PRIOR_FP32_MATMUL_PRECISION
    _PRIOR_FP32_MATMUL_PRECISION = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high")


def tearDownModule():
    global _PRIOR_FP32_MATMUL_PRECISION
    if _PRIOR_FP32_MATMUL_PRECISION is not None:
        torch.set_float32_matmul_precision(_PRIOR_FP32_MATMUL_PRECISION)
        _PRIOR_FP32_MATMUL_PRECISION = None


index = torch.ops.aten.index
Tensor = torch.Tensor


T = TypeVar("T")
M = TypeVar("M", bound=Callable)


def large_tensor_test_class(
    size: str, device: torch.device | str | None = None
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


skip_on_mps = skipMPSIf(True, "Not supported on MPS")


def _is_not_implemented_exc(exc):
    """True if `exc` is or wraps a NotImplementedError (Dynamo/Inductor wrap it)."""
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if (
            isinstance(exc, NotImplementedError)
            or "NotImplemented" in type(exc).__name__
            or "NotImplementedError:" in str(exc)
        ):
            return True
        exc = getattr(exc, "inner_exception", None) or exc.__cause__ or exc.__context__
    return False


def expected_not_implemented_on_mps(test_func):
    """On MPS the test must raise NotImplementedError (possibly wrapped by
    Inductor/Dynamo); other exceptions or silent success fail the test."""

    @functools.wraps(test_func)
    def wrapper(self, *args, **kwargs):
        device = kwargs.get("device", args[0] if args else None)
        if not _is_mps_device(device):
            return test_func(self, *args, **kwargs)
        try:
            test_func(self, *args, **kwargs)
        except Exception as exc:
            if _is_not_implemented_exc(exc):
                return
            raise
        self.fail("expected NotImplementedError on MPS but test passed")

    return wrapper


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


def flex_attention_fwd(q, k, v, score_mod, block_mask, scale):
    # Uses the HOP directly because flex_attention_backward expects lse in log2
    # scale, but the public flex_attention API converts lse to natural log.
    out, lse, _ = flex_attention_hop(
        q,
        k,
        v,
        score_mod,
        block_mask.as_tuple(),
        scale,
        {},
    )
    return out, lse


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


device_configs = {}
# Tests are skipped when no device is supported, so CPU as default is safe
test_device = ("cpu",)
if HAS_GPU:
    if TEST_ON_CUDA:
        if TEST_ON_CPU:
            test_device = (
                "cuda",
                "cpu",
            )
        else:
            test_device = ("cuda",)
    elif TEST_ON_XPU:
        torch._C._set_onednn_allow_tf32(True)
        test_device = ("xpu",)
elif HAS_MPS:
    test_device = ("mps",)


class SubstringSet:
    def __init__(self, items):
        self.items = set(items)

    def __contains__(self, item):
        if "cuda" in item:
            item = "cuda"
        if "xpu" in item:
            item = "xpu"
        if "mps" in item:
            item = "mps"
        return item in self.items


DEVICE_SUPPORTS_BACKWARDS = SubstringSet(["cuda", "xpu"])

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
device_configs["mps"] = DeviceConfig(
    dtypes=[torch.float32, torch.float16, torch.bfloat16],
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


def _is_mps_device(device) -> bool:
    return str(device).startswith("mps") or (
        isinstance(device, torch.device) and device.type == "mps"
    )


def query_key_value_clones(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    dtype: torch.dtype | None = None,
):
    """Clones the query, key, and value tensors and moves them to the specified dtype."""
    if dtype is None:
        dtype = query.dtype
    if dtype == torch.float64 and query.device.type == "mps":
        # MPS lacks fp64, run the high-precision reference on CPU
        query_ref = (
            query.detach().cpu().clone().to(dtype).requires_grad_(query.requires_grad)
        )
        key_ref = key.detach().cpu().clone().to(dtype).requires_grad_(key.requires_grad)
        value_ref = (
            value.detach().cpu().clone().to(dtype).requires_grad_(value.requires_grad)
        )
    else:
        query_ref = query.detach().clone().to(dtype).requires_grad_(query.requires_grad)
        key_ref = key.detach().clone().to(dtype).requires_grad_(key.requires_grad)
        value_ref = value.detach().clone().to(dtype).requires_grad_(value.requires_grad)
    return query_ref, key_ref, value_ref


def _relocate_captures(mod, device):
    """Copy a score_mod / mask_mod with its captured tensors moved to `device`.

    A score_mod / mask_mod can reference a tensor from the surrounding scope:

        bias = torch.randn(S, S, device="mps")
        score_mod = lambda s, b, h, q, kv: s + bias[q, kv]   # captures `bias`

    The MPS tests check the compiled output against a reference run on CPU in
    float64 (MPS has no float64). That reference calls `score_mod` with CPU
    inputs, but `bias` is still on MPS, so `cpu_score + bias[q, kv]` mixes a CPU
    and an MPS tensor and raises a device-mismatch error.

    This returns a copy of `mod` with every captured tensor moved to `device`
    (and a captured BlockMask too, recursing into its mask_mod's captures). It
    builds a new function instead of editing `mod` in place, because the
    original is still used for the on-device run. A mod that captures nothing is
    returned unchanged.
    """
    cells = getattr(mod, "__closure__", None)
    if not cells:
        return mod
    relocated = []
    for cell in cells:
        val = cell.cell_contents
        if isinstance(val, BlockMask):
            val = val.to(device)
            val.mask_mod = _relocate_captures(val.mask_mod, device)
        elif isinstance(val, torch.Tensor):
            val = val.to(device)
        relocated.append(types.CellType(val))
    return types.FunctionType(
        mod.__code__, mod.__globals__, mod.__name__, mod.__defaults__, tuple(relocated)
    )


def batch_reserve(paged_attention: PagedAttention, target_seq_len: Tensor):
    (B,) = target_seq_len.shape
    for b in range(B):
        paged_attention.reserve(
            torch.tensor(b),
            target_seq_len[b],
        )


def _tdm_graph():
    """A real GraphLowering, so the gate runs against real sizevars."""
    from torch._inductor.graph import GraphLowering
    from torch.fx.experimental.proxy_tensor import make_fx

    return GraphLowering(make_fx(lambda: torch.zeros(2, 3))())


def _tdm_operand(name, stride, size=(2, 4, 128, 64), offset=0, dtype=torch.float16):
    """A real flex operand for the descriptor gate.

    Real IR rather than a Mock: a mock can satisfy a gate for the wrong reason
    and make an assertion pass vacuously.
    """
    from torch._inductor.ir import Buffer, FixedLayout

    return Buffer(
        name=name,
        layout=FixedLayout(
            torch.device("cuda"), dtype, size=size, stride=stride, offset=offset
        ),
    )


# Host-side gate logic. Device capability is mocked, so these are not ROCm-only:
# running them on CUDA and XPU builds is what keeps the gate honest there too.
class TestFlexAttentionROCmOptions(InductorTestCase):
    def _backward_auto_profile_inputs(
        self, dtype=torch.float16, batch=4, length=4096, dim=128, heads=64, kv_heads=8
    ):
        def operand(name, shape, dtype):
            strides, stride = [], 1
            for extent in reversed(shape):
                strides.insert(0, stride)
                stride *= extent
            return _tdm_operand(name, strides, size=shape, dtype=dtype)

        query = operand("auto_query", (batch, heads, length, dim), dtype)
        key = operand("auto_key", (batch, kv_heads, length, dim), dtype)
        value = operand("auto_value", (batch, kv_heads, length, dim), dtype)
        lse = operand("auto_lse", (batch, heads, length), torch.float32)
        delta = operand("auto_delta", (batch, heads, length), torch.float32)
        grad_out = operand("auto_grad_out", (batch, heads, length, dim), dtype)
        out = operand("auto_out", (batch, heads, length, dim), dtype)
        nodes = [query, key, value, lse, delta, grad_out, query, value]
        sparse_size = 128 if dim == 128 else 1 << 30
        blocks = length // 128 if dim == 128 else 1
        for index in range(8, 16):
            shape = (1, 1, blocks) if index % 2 == 0 else (1, 1, blocks, blocks)
            nodes.append(
                operand("auto_metadata_" + str(index), (0,), torch.float32)
                if dim != 128 and index >= 12
                else operand("auto_metadata_" + str(index), shape, torch.int32)
            )
        options = dict(
            SM_SCALE=1 / math.sqrt(dim),
            BLOCKS_ARE_CONTIGUOUS=False,
            USE_TMA=False,
            PRESCALE_QK=False,
            SPARSE_Q_BLOCK_SIZE=sparse_size,
            SPARSE_KV_BLOCK_SIZE=sparse_size,
            QK_HEAD_DIM=dim,
            V_HEAD_DIM=dim,
            QK_HEAD_DIM_ROUNDED=128 if dim == 96 else 256 if dim == 192 else dim,
            V_HEAD_DIM_ROUNDED=128 if dim == 96 else 256 if dim == 192 else dim,
            GQA_SHARED_HEADS=heads // kv_heads,
            HAS_FULL_BLOCKS=dim == 128,
            SAFE_HEAD_DIM=dim not in (96, 192),
            IS_DIVISIBLE=True,
            BLOCK_M1=32,
            BLOCK_N1=128,
            BLOCK_M2=128,
            BLOCK_N2=32,
            num_warps=4,
            num_stages=2,
            matrix_instr_nonkdim=0,
            waves_per_eu=0,
            kpack=2,
            FLOAT32_PRECISION="'ieee'",
            ROCM_SEPARATE_DQ=False,
            ROCM_EARLY_DP=False,
            XCD_REMAP=False,
        )
        return nodes, out, options

    def _backward_auto_profile(self, nodes, out, options, user_options=None, **flags):
        from torch._inductor.kernel.flex.flex_attention import (
            _get_rocm_backward_auto_profile,
            _is_rocm_scalar_true_mask_graph,
            _is_rocm_simple_causal_mask_graph,
        )
        from torch._inductor.virtualized import V

        dim = nodes[0].get_size()[-1]
        graph = torch.fx.Graph()
        indices = [graph.placeholder(name) for name in ("b", "h", "q", "k")]
        mask = (
            graph.call_function(torch.ops.aten.ge.Tensor, args=(indices[2], indices[3]))
            if dim == 128
            else graph.call_function(
                torch.ops.aten.full.default,
                args=([], True),
                kwargs={"dtype": torch.bool},
            )
        )
        graph.output(mask)
        mask_graph = torch.fx.GraphModule({}, graph)
        defaults = dict(
            backend="TRITON",
            is_gfx950=True,
            single_config=True,
            score_is_trivial=True,
            has_captures=False,
            mask_is_causal=_is_rocm_simple_causal_mask_graph(mask_graph),
            mask_is_scalar_true=_is_rocm_scalar_true_mask_graph(mask_graph),
            block_mask_is_default=dim != 128,
            grad_logsumexp_is_none=True,
        )
        defaults.update(flags)
        with V.set_graph_handler(_tdm_graph()):
            return _get_rocm_backward_auto_profile(
                {
                    "PRESCALE_QK": False,
                    "ROWS_GUARANTEED_SAFE": False,
                    "BLOCKS_ARE_CONTIGUOUS": False,
                    "WRITE_DQ": True,
                    "OUTPUT_LOGSUMEXP": True,
                    "OUTPUT_MAX": False,
                }
                if user_options is None
                else user_options,
                options,
                nodes,
                out,
                **defaults,
            )

    @common_utils.parametrize(
        "dtype,batch,length,dim,heads,kv_heads",
        (
            (torch.bfloat16, 4, 2048, 96, 21, 21),
            (torch.bfloat16, 4, 4096, 96, 21, 21),
            (torch.bfloat16, 2, 2048, 96, 8, 2),
            (torch.bfloat16, 2, 4096, 96, 8, 2),
            (torch.bfloat16, 4, 4096, 64, 64, 8),
            (torch.bfloat16, 4, 4096, 128, 64, 8),
            (torch.bfloat16, 4, 2048, 192, 10, 10),
            (torch.bfloat16, 4, 4096, 192, 10, 10),
            (torch.bfloat16, 2, 2048, 192, 8, 2),
            (torch.bfloat16, 4, 4096, 256, 32, 8),
            (torch.bfloat16, 4, 4096, 256, 32, 32),
            (torch.float16, 4, 2048, 128, 64, 8),
            (torch.float16, 4, 4096, 128, 64, 8),
            (torch.float16, 2, 2048, 128, 64, 8),
            (torch.float16, 2, 4096, 128, 64, 8),
            (torch.float16, 1, 4096, 128, 64, 8),
            (torch.float32, 4, 2048, 128, 64, 8),
            (torch.float32, 4, 4096, 128, 64, 8),
            (torch.float32, 2, 2048, 128, 64, 8),
            (torch.float32, 2, 4096, 128, 64, 8),
            (torch.float32, 1, 2048, 128, 64, 8),
            (torch.float32, 1, 4096, 128, 64, 8),
        ),
    )
    def test_backward_auto_profile_measured_shapes(
        self, dtype, batch, length, dim, heads, kv_heads
    ):
        from torch._inductor.kernel.flex.flex_attention import (
            _get_rocm_backward_config_options,
            _get_rocm_backward_phase_options,
            _normalize_rocm_backward_options,
        )
        from torch._inductor.virtualized import V

        nodes, out, options = self._backward_auto_profile_inputs(
            dtype, batch, length, dim, heads, kv_heads
        )
        original = dict(options)
        profile = self._backward_auto_profile(nodes, out, options)
        self.assertIsNotNone(profile)
        self.assertIs(profile["ROCM_SEPARATE_DQ"], dim not in (96, 192))
        self.assertIs(profile["ROCM_EARLY_DP"], dim not in (96, 192))
        self.assertIs(profile.get("bwd_ROCM_D96_PANEL_DOTS", False), dim == 96)
        self.assertIs(profile.get("ROCM_D256_PANEL_DOTS", False), dim == 256)
        self.assertIs(
            profile.get("bwd_ROCM_EARLY_PROBABILITY_COMPACT", False), dim == 256
        )
        self.assertIs(profile.get("bwd_ROCM_D128_REVERSE_DQ_ROWS", False), dim == 128)
        self.assertIs(
            profile.get("bwd_ROCM_D128_DV_VIEW_CARRY_SPLIT64", False), dim == 128
        )
        self.assertIs(
            profile.get("bwd_ROCM_D128_FP16_COVERAGE", False), dtype == torch.float16
        )
        self.assertIs(
            profile.get("bwd_ROCM_D128_FP32_BALANCED64_COVERAGE", False),
            dtype == torch.float32,
        )
        self.assertIs(profile["BLOCKS_ARE_CONTIGUOUS"], dim in (96, 192))
        self.assertEqual(
            tuple(
                profile[key] for key in ("BLOCK_M1", "BLOCK_N1", "BLOCK_M2", "BLOCK_N2")
            ),
            (16, 64, 128, 32)
            if dim in (192, 256)
            else (16, 64, 64, 16)
            if dtype == torch.float32
            else (32, 128, 128, 32),
        )
        self.assertEqual(options, original)
        self.assertIsNot(profile, options)
        requests = _normalize_rocm_backward_options(profile)
        current = _get_rocm_backward_config_options(profile, is_gfx950=True)
        with V.set_graph_handler(_tdm_graph()):
            dkdv, dq = _get_rocm_backward_phase_options(
                current,
                nodes,
                requests=requests,
                single_config=True,
                is_gfx950=True,
                query=nodes[0],
                key=nodes[1],
                value=nodes[2],
                grad_out=nodes[5],
                use_separate_dq=True,
                use_d192_panel_phase=dim == 192,
                use_d96_panel_phase=dim == 96,
                fp16_pilot=dtype == torch.float16,
                fp32_balanced64_pilot=dtype == torch.float32,
                score_is_trivial=True,
                has_captures=False,
                mask_is_causal=True if dim == 128 else None,
                mask_is_scalar_true=dim != 128,
                block_mask_is_default=dim != 128,
                grad_logsumexp_is_none=True,
                is_automatic=True,
            )
        self.assertEqual(
            dkdv["BLOCK_M1"],
            (32 if length == 2048 else 64)
            if dim == 96 and heads != kv_heads
            else 32
            if dim == 256 and heads != kv_heads
            else current["BLOCK_M1"],
        )
        self.assertEqual(dq["BLOCK_M1"], current["BLOCK_M1"])
        d96_gqa4 = dim == 96 and heads != kv_heads
        self.assertEqual(
            dkdv["BLOCK_N1"],
            (32 if length == 2048 else 64) if d96_gqa4 else current["BLOCK_N1"],
        )
        self.assertEqual(dkdv["num_stages"], 1 if d96_gqa4 else current["num_stages"])
        self.assertEqual((dkdv["ROCM_BWD_PHASE"], dq["ROCM_BWD_PHASE"]), (1, 2))
        self.assertIs(dkdv["ROCM_D192_PANEL_DOTS"], dim == 192)
        self.assertIs(dq["ROCM_D192_DQ_EARLY_DP"], dim == 192)
        self.assertIs(dkdv["ROCM_D96_PANEL_DOTS"], dim == 96 and heads == kv_heads)
        self.assertIs(dq["ROCM_D96_DQ_UNROLL2"], dim == 96 and heads == kv_heads)
        self.assertIs(dq["ROCM_D96_DQ_EARLY_DP"], dim == 96 and heads == kv_heads)
        self.assertIs(dkdv["ROCM_EARLY_PROBABILITY_COMPACT"], dim == 256)
        self.assertIs(dq["ROCM_EARLY_PROBABILITY_COMPACT"], False)
        self.assertIs(dkdv["ROCM_D128_DV_VIEW_CARRY_SPLIT64"], dim == 128)
        self.assertIs(dq["ROCM_D128_REVERSE_DQ_ROWS"], dim == 128)
        self.assertIs(
            dkdv.get("ROCM_D128_GQA_QUERY_TILE_OUTER", False),
            dim == 128
            and (
                dtype == torch.float16
                or dtype == torch.float32
                and (length == 4096 or (batch, length) == (4, 2048))
            ),
        )
        self.assertEqual(
            dkdv["matrix_instr_nonkdim"],
            16
            if (dtype, batch, heads, kv_heads, length, dim)
            == (torch.bfloat16, 4, 10, 10, 4096, 192)
            else 0,
        )
        self.assertEqual(
            dq["matrix_instr_nonkdim"],
            16
            if (dtype, batch, heads, kv_heads, length, dim)
            == (torch.bfloat16, 4, 10, 10, 4096, 192)
            else 0,
        )
        self.assertIs(
            dkdv["XCD_REMAP"], current["XCD_REMAP"] or dim == 256 and heads != kv_heads
        )
        self.assertIs(dq["XCD_REMAP"], current["XCD_REMAP"])
        profile["num_warps"] = 8
        self.assertEqual(options, original)

    def test_backward_auto_profile_d256_gqa4_uses_panel_profile(self):
        from torch._inductor.kernel.flex.flex_attention import (
            _get_rocm_backward_config_options,
            _normalize_rocm_backward_options,
            _use_rocm_separate_dq,
        )
        from torch._inductor.virtualized import V

        nodes, out, options = self._backward_auto_profile_inputs(
            torch.bfloat16, dim=256, heads=32, kv_heads=8
        )
        before = dict(options)
        profile = self._backward_auto_profile(nodes, out, options)
        self.assertIsNotNone(profile)
        self.assertEqual(options, before)
        self.assertIs(profile["ROCM_D256_PANEL_DOTS"], True)
        self.assertIs(profile["bwd_ROCM_EARLY_PROBABILITY_COMPACT"], True)
        _normalize_rocm_backward_options(profile)
        current = _get_rocm_backward_config_options(profile, is_gfx950=True)
        self.assertEqual(current["BLOCK_M1"], 16)
        with V.set_graph_handler(_tdm_graph()):
            self.assertTrue(
                _use_rocm_separate_dq(
                    True,
                    *nodes[:3],
                    current,
                    True,
                    False,
                    mask_is_scalar_true=True,
                    block_mask_is_default=True,
                )
            )

    @common_utils.parametrize("length", (2048, 4096))
    @common_utils.parametrize("automatic", (False, True))
    @common_utils.parametrize("panel", (False, True))
    @common_utils.parametrize("gqa", (1, 4))
    def test_backward_auto_profile_d96_gqa4_phase_options(
        self, length, automatic, panel, gqa
    ):
        from torch._inductor.kernel.flex.flex_attention import (
            _get_rocm_backward_config_options,
            _get_rocm_backward_phase_options,
            _normalize_rocm_backward_options,
        )
        from torch._inductor.virtualized import V

        batch, heads, kv_heads = (2, 8, 2) if gqa == 4 else (4, 21, 21)
        nodes, out, options = self._backward_auto_profile_inputs(
            torch.bfloat16, batch, length, 96, heads, kv_heads
        )
        profile = self._backward_auto_profile(nodes, out, options)
        self.assertIsNotNone(profile)
        requests = _normalize_rocm_backward_options(profile)
        current = _get_rocm_backward_config_options(profile, is_gfx950=True)
        before = current.copy()
        kwargs = dict(
            requests=requests,
            single_config=True,
            is_gfx950=True,
            query=nodes[0],
            key=nodes[1],
            value=nodes[2],
            grad_out=nodes[5],
            use_separate_dq=True,
            use_d192_panel_phase=False,
            use_d96_panel_phase=panel,
            fp16_pilot=False,
            fp32_balanced64_pilot=False,
            score_is_trivial=True,
            has_captures=False,
            mask_is_causal=False,
            mask_is_scalar_true=True,
            block_mask_is_default=True,
            grad_logsumexp_is_none=True,
        )
        with V.set_graph_handler(_tdm_graph()):
            explicit_dkdv, explicit_dq = _get_rocm_backward_phase_options(
                current, nodes, **kwargs
            )
            dkdv, dq = _get_rocm_backward_phase_options(
                current, nodes, is_automatic=automatic, **kwargs
            )
        expected = explicit_dkdv.copy()
        if automatic and panel and gqa == 4:
            expected.update(
                ROCM_D96_PANEL_DOTS=False,
                BLOCK_M1=32 if length == 2048 else 64,
                BLOCK_N1=32 if length == 2048 else 64,
                num_stages=1,
            )
        self.assertEqual(dkdv, expected)
        self.assertEqual(dq, explicit_dq)
        self.assertEqual(current, before)
        self.assertEqual(
            (explicit_dkdv["BLOCK_M1"], explicit_dkdv["BLOCK_N1"]), (32, 128)
        )
        self.assertEqual(
            (explicit_dkdv["num_warps"], explicit_dkdv["num_stages"]), (4, 2)
        )
        self.assertEqual((dq["num_warps"], dq["num_stages"]), (4, 2))

    @common_utils.parametrize("compact", (False, True))
    def test_backward_auto_profile_d256_gqa4_preserves_explicit_options(self, compact):
        from torch._inductor.kernel.flex.flex_attention import (
            _get_rocm_backward_config_options,
            _get_rocm_backward_phase_options,
            _normalize_rocm_backward_options,
            _use_rocm_separate_dq,
        )
        from torch._inductor.virtualized import V

        nodes, out, options = self._backward_auto_profile_inputs(
            torch.bfloat16, dim=256, heads=32, kv_heads=8
        )
        explicit = dict(
            bwd_BLOCK_M1=16,
            bwd_BLOCK_N1=64,
            bwd_BLOCK_M2=128,
            bwd_BLOCK_N2=32,
            bwd_num_warps=4,
            bwd_num_stages=2,
            bwd_matrix_instr_nonkdim=0,
            bwd_waves_per_eu=0,
            bwd_kpack=1,
            bwd_ROCM_SEPARATE_DQ=True,
            bwd_ROCM_EARLY_DP=True,
            bwd_XCD_REMAP=False,
            bwd_ROCM_D256_PANEL_DOTS=True,
            bwd_ROCM_EARLY_PROBABILITY_COMPACT=compact,
        )
        before = dict(explicit)
        self.assertIsNone(self._backward_auto_profile(nodes, out, options, explicit))
        self.assertEqual(explicit, before)
        current = dict(options, **explicit)
        requests = _normalize_rocm_backward_options(current)
        current = _get_rocm_backward_config_options(current, is_gfx950=True)
        with V.set_graph_handler(_tdm_graph()):
            separate = _use_rocm_separate_dq(
                True,
                *nodes[:3],
                current,
                True,
                False,
                mask_is_scalar_true=True,
                block_mask_is_default=True,
            )
            self.assertTrue(separate)
            dkdv, dq = _get_rocm_backward_phase_options(
                current,
                nodes,
                requests=requests,
                single_config=True,
                is_gfx950=True,
                query=nodes[0],
                key=nodes[1],
                value=nodes[2],
                grad_out=nodes[5],
                use_separate_dq=separate,
                use_d192_panel_phase=False,
                use_d96_panel_phase=False,
                fp16_pilot=False,
                fp32_balanced64_pilot=False,
                score_is_trivial=True,
                has_captures=False,
                mask_is_causal=None,
                mask_is_scalar_true=True,
                block_mask_is_default=True,
                grad_logsumexp_is_none=True,
                is_automatic=False,
            )
        self.assertEqual((dkdv["BLOCK_M1"], dq["BLOCK_M1"]), (16, 16))
        self.assertEqual((dkdv["ROCM_BWD_PHASE"], dq["ROCM_BWD_PHASE"]), (1, 2))
        self.assertEqual(
            (dkdv["matrix_instr_nonkdim"], dq["matrix_instr_nonkdim"]), (0, 0)
        )
        self.assertIs(dkdv["ROCM_EARLY_PROBABILITY_COMPACT"], compact)
        self.assertIs(dq["ROCM_EARLY_PROBABILITY_COMPACT"], False)

    @common_utils.parametrize("dim", (96, 128, 192, 256))
    @common_utils.parametrize(
        "case",
        (
            "architecture",
            "backend",
            "multi_config",
            "score",
            "capture",
            "mask",
            "grad_lse",
            "zero_scale",
            "negative_scale",
            "nan_scale",
            "boolean_scale",
            "shape",
            "batch",
            "heads",
            "query_dtype",
            "query_stride",
            "out_stride",
        ),
    )
    def test_backward_auto_profile_fallback(self, dim, case):
        nodes, out, options = self._backward_auto_profile_inputs(
            torch.bfloat16,
            dim=dim,
            heads=21 if dim == 96 else 10 if dim == 192 else 32 if dim == 256 else 64,
            kv_heads=21 if dim == 96 else 10 if dim == 192 else 32 if dim == 256 else 8,
        )
        flags = {
            "architecture": dict(is_gfx950=False),
            "backend": dict(backend="FLASH"),
            "multi_config": dict(single_config=False),
            "score": dict(score_is_trivial=False),
            "capture": dict(has_captures=True),
            "mask": dict(mask_is_causal=False, mask_is_scalar_true=False),
            "grad_lse": dict(grad_logsumexp_is_none=False),
        }
        scales = {
            "zero_scale": 0.0,
            "negative_scale": -0.125,
            "nan_scale": float("nan"),
            "boolean_scale": False,
        }
        if case in scales:
            options["SM_SCALE"] = scales[case]
        if case in ("shape", "batch", "heads", "query_dtype", "query_stride"):
            size, stride = list(nodes[0].get_size()), list(nodes[0].get_stride())
            if case in ("shape", "batch", "heads"):
                size[{"batch": 0, "heads": 1, "shape": 2}[case]] += 1
            if case == "query_stride":
                stride[-1] = 2
            nodes[0] = _tdm_operand(
                "invalid_auto_query",
                stride,
                size=size,
                dtype=torch.float16 if case == "query_dtype" else torch.bfloat16,
            )
        if case == "out_stride":
            stride = list(out.get_stride())
            stride[-1] = 2
            out = _tdm_operand(
                "invalid_auto_out", stride, size=out.get_size(), dtype=out.get_dtype()
            )
        self.assertIsNone(
            self._backward_auto_profile(nodes, out, options, **flags.get(case, {}))
        )

    @common_utils.parametrize("index", tuple(range(8, 16)))
    @common_utils.parametrize("field", ("shape", "stride", "dtype"))
    def test_backward_auto_profile_sparse_metadata(self, index, field):
        nodes, out, options = self._backward_auto_profile_inputs()
        size, stride = list(nodes[index].get_size()), list(nodes[index].get_stride())
        if field == "shape":
            size[0] = 4
        elif field == "stride":
            stride[-1] = 2
        nodes[index] = _tdm_operand(
            "invalid_auto_metadata",
            stride,
            size=size,
            dtype=torch.int64 if field == "dtype" else torch.int32,
        )
        self.assertIsNone(self._backward_auto_profile(nodes, out, options))

    @common_utils.parametrize("index", (0, 1, 2, 3, 4, 5, 16))
    @common_utils.parametrize("field", ("shape", "stride", "dtype"))
    def test_backward_auto_profile_operand_boundaries(self, index, field):
        nodes, out, options = self._backward_auto_profile_inputs()
        tensor = out if index == 16 else nodes[index]
        size, stride = list(tensor.get_size()), list(tensor.get_stride())
        if field == "shape":
            size[2] -= 1
        elif field == "stride":
            stride[-1] = 2
        replacement = _tdm_operand(
            "invalid_auto_operand",
            stride,
            size=size,
            dtype=torch.bfloat16 if field == "dtype" else tensor.get_dtype(),
        )
        if index == 16:
            out = replacement
        else:
            nodes[index] = replacement
        self.assertIsNone(self._backward_auto_profile(nodes, out, options))

    @common_utils.parametrize("prefix", ("", "bwd_", "fwd_"))
    @common_utils.parametrize(
        "option",
        (
            "ROCM_SEPARATE_DQ",
            "ROCM_EARLY_DP",
            "XCD_REMAP",
            "BLOCK_M1",
            "num_warps",
            "num_stages",
            "matrix_instr_nonkdim",
            "waves_per_eu",
            "kpack",
        ),
    )
    def test_backward_auto_profile_option_scope(self, prefix, option):
        nodes, out, options = self._backward_auto_profile_inputs()
        user_options = {
            prefix + option: False
            if option.startswith("ROCM_") or option == "XCD_REMAP"
            else 1
        }
        original = dict(user_options)
        profile = self._backward_auto_profile(nodes, out, options, user_options)
        self.assertEqual(user_options, original)
        if prefix == "fwd_":
            self.assertIsNotNone(profile)
        else:
            self.assertIsNone(profile)

    @common_utils.parametrize("dim", (96, 128, 256))
    def test_backward_auto_profile_indexed_blocks(self, dim):
        nodes, out, options = self._backward_auto_profile_inputs(
            torch.bfloat16,
            dim=dim,
            heads=21 if dim == 96 else 32 if dim == 256 else 64,
            kv_heads=21 if dim == 96 else 10 if dim == 192 else 32 if dim == 256 else 8,
        )
        profile = self._backward_auto_profile(
            nodes, out, options, {"bwd_BLOCKS_ARE_CONTIGUOUS": False}
        )
        if dim == 96:
            self.assertIsNone(profile)
        else:
            self.assertIsNotNone(profile)
            self.assertIs(profile["BLOCKS_ARE_CONTIGUOUS"], False)

    @common_utils.parametrize("split_available", (False, True))
    @common_utils.parametrize(
        "operation",
        ("unchanged", "copy", "append", "replace", "remove", "reverse", "clear"),
    )
    def test_backward_choices_hook_split_precedence(self, split_available, operation):
        from torch._inductor.choices import InductorChoices
        from torch._inductor.kernel.flex.flex_attention import (
            _append_flex_attention_backward_choices,
        )
        from torch._inductor.virtualized import V

        class CustomChoices(InductorChoices):
            def __init__(self):
                self.returned = None
                self.abi_sizes = None

            def append_flex_attention_backward_choices(
                self,
                choices,
                configs,
                input_nodes,
                subgraphs,
                layout,
                kernel_options,
                sparse_q_block_size,
                sparse_kv_block_size,
                *,
                mutated_inputs,
            ):
                self.abi_sizes = (len(input_nodes), len(subgraphs), len(mutated_inputs))
                input_nodes.clear()
                subgraphs.clear()
                mutated_inputs.clear()
                if operation == "copy":
                    choices = list(choices)
                elif operation == "append":
                    choices.append(custom)
                elif operation == "replace":
                    choices[:] = [custom]
                elif operation == "remove":
                    choices.pop()
                elif operation == "reverse":
                    choices.reverse()
                elif operation == "clear":
                    choices.clear()
                self.returned = choices
                return choices

        first, second, custom = object(), object(), object()
        inputs = [object() for _ in range(16)]
        subgraphs = [object() for _ in range(4)]
        mutations = [inputs[6], inputs[7]]
        handler = CustomChoices()
        with V.set_choices_handler(handler):
            choices, use_split = _append_flex_attention_backward_choices(
                [first, second],
                [],
                inputs,
                subgraphs,
                None,
                {},
                128,
                128,
                mutated_inputs=mutations,
                split_available=split_available,
            )
        expected = {
            "unchanged": [first, second],
            "copy": [first, second],
            "append": [first, second, custom],
            "replace": [custom],
            "remove": [first],
            "reverse": [second, first],
            "clear": [],
        }
        self.assertIs(choices, handler.returned)
        self.assertEqual(choices, expected[operation])
        self.assertEqual(
            use_split, split_available and operation in ("unchanged", "copy")
        )
        self.assertEqual(handler.abi_sizes, (16, 4, 2))
        self.assertEqual(len(inputs), 16)
        self.assertEqual(len(subgraphs), 4)
        self.assertEqual(mutations, [inputs[6], inputs[7]])

    @common_utils.parametrize(
        "case",
        (
            "enabled",
            "b2_n2048",
            "kpack2",
            "zero_scale",
            "negative_scale",
            "n4096",
            "n4096_mixed_key",
            "n4096_default",
            "n4096_numeric",
            "n4096_mi16",
            "n4096_mi32",
            "default",
            "numeric",
            "architecture",
            "score",
            "capture",
            "mask",
            "grad_lse",
            "query_dtype",
            "key_dtype",
            "value_dtype",
            "do_dtype",
            "out_dtype",
            "lse_dtype",
            "batch",
            "heads",
            "length",
            "key_shape",
            "out_shape",
            "lse_shape",
            "query_stride",
            "out_stride",
            "stages",
            "warps",
            "phase",
            "gqa",
            "prescale",
            "tma",
        ),
    )
    def test_d128_fp16_coverage_guard(self, case):
        from torch._inductor.kernel.flex.flex_attention import (
            _use_rocm_d128_fp16_coverage,
        )
        from torch._inductor.virtualized import V

        names = ("query", "key", "value", "do", "out", "lse")
        length = 4096 if case.startswith("n4096") else 2048
        shapes = (
            (4, 64, length, 128),
            (4, 8, length, 128),
            (4, 8, length, 128),
            (4, 64, length, 128),
            (4, 64, length, 128),
            (4, 64, length),
        )
        operands = []
        for name, shape in zip(names, shapes, strict=True):
            sizes = list(shape)
            if case == "batch":
                sizes[0] = 3
            if case == "b2_n2048":
                sizes[0] = 2
            if case == "heads" and name == "query":
                sizes[1] = 32
            if case == "length":
                sizes[2] = 8192
            if case == name + "_shape" or (case == "n4096_mixed_key" and name == "key"):
                sizes[2] = 1024
            strides, stride = [], 1
            for size in reversed(sizes):
                strides.insert(0, stride)
                stride *= size
            if case == name + "_stride":
                strides[-1] = 2
            dtype = torch.float32 if name == "lse" else torch.float16
            if case == name + "_dtype":
                dtype = torch.float16 if name == "lse" else torch.bfloat16
            operands.append(
                types.SimpleNamespace(
                    get_size=lambda sizes=sizes: sizes,
                    get_stride=lambda strides=strides: strides,
                    get_dtype=lambda dtype=dtype: dtype,
                    get_device=lambda: torch.device("cuda"),
                )
            )
        options = dict(
            ROCM_SEPARATE_DQ=True,
            ROCM_EARLY_DP=True,
            XCD_REMAP=True,
            BLOCKS_ARE_CONTIGUOUS=True,
            SAFE_HEAD_DIM=True,
            IS_DIVISIBLE=True,
            HAS_FULL_BLOCKS=True,
            USE_TMA=case == "tma",
            PRESCALE_QK=case == "prescale",
            SM_SCALE=0.0
            if case == "zero_scale"
            else -0.125
            if case == "negative_scale"
            else 128**-0.5,
            ROCM_BWD_PHASE=1 if case == "phase" else 0,
            ROCM_GQA_SPLITS=1,
            GQA_SHARED_HEADS=4 if case == "gqa" else 8,
            QK_HEAD_DIM=128,
            V_HEAD_DIM=128,
            QK_HEAD_DIM_ROUNDED=128,
            V_HEAD_DIM_ROUNDED=128,
            BLOCK_M1=32,
            BLOCK_N1=128,
            BLOCK_M2=128,
            BLOCK_N2=32,
            num_warps=8 if case == "warps" else 4,
            num_stages=1 if case == "stages" else 2,
            matrix_instr_nonkdim=16
            if case == "n4096_mi16"
            else 32
            if case == "n4096_mi32"
            else 0,
            waves_per_eu=0,
            kpack=2 if case == "kpack2" else 1,
            SPARSE_Q_BLOCK_SIZE=128,
            SPARSE_KV_BLOCK_SIZE=128,
        )
        sizevars = types.SimpleNamespace(
            statically_known_equals=lambda first, second: first == second
        )
        with V.set_graph_handler(types.SimpleNamespace(sizevars=sizevars)):
            actual = _use_rocm_d128_fp16_coverage(
                case != "architecture",
                *operands,
                options,
                case != "score",
                case == "capture",
                requested=1
                if case in ("numeric", "n4096_numeric")
                else case not in ("default", "n4096_default"),
                mask_is_causal=case != "mask",
                grad_logsumexp_is_none=case != "grad_lse",
            )
        self.assertIs(
            actual,
            case
            in (
                "enabled",
                "kpack2",
                "zero_scale",
                "negative_scale",
                "n4096",
                "b2_n2048",
            ),
        )

    @common_utils.parametrize(
        "batch,length,fp32,expected",
        (
            (4, 2048, False, True),
            (4, 4096, False, True),
            (2, 4096, False, True),
            (2, 2048, False, True),
            (2, 1024, False, False),
            (4, 1024, False, False),
            (4, 2048, True, True),
            (4, 4096, True, True),
            (2, 4096, True, True),
            (2, 2048, True, True),
            (2, 1024, True, False),
            (1, 4096, False, True),
            (1, 2048, False, False),
            (1, 4096, True, True),
            (1, 2048, True, True),
            (1, 1024, True, False),
            (3, 4096, False, False),
        ),
    )
    def test_d128_fp16_joint_batch_scope(self, batch, length, fp32, expected):
        from torch._inductor.virtualized import V

        values = self._d128_coverage_operands(batch, length, fp32)
        with V.set_graph_handler(_tdm_graph()):
            actual = self._d128_coverage_routes(values, fp32)
        self.assertEqual(actual, (expected,) * 4)

    @common_utils.parametrize("fp32", (False, True))
    @common_utils.parametrize("option", ("PRESCALE_QK", "USE_TMA"))
    @common_utils.parametrize(
        "case,expected",
        (
            ("forward", True),
            ("forward_backward", True),
            ("shared_backward", True),
            ("backward", False),
            ("shared", False),
            ("forward_shared", False),
        ),
    )
    def test_d128_coverage_option_scoping(self, fp32, option, case, expected):
        from torch._inductor.virtualized import V

        overrides = {
            "forward": {"fwd_" + option: True},
            "forward_backward": {"fwd_" + option: True, "bwd_" + option: False},
            "shared_backward": {option: True, "bwd_" + option: False},
            "backward": {"bwd_" + option: True},
            "shared": {option: True},
            "forward_shared": {"fwd_" + option: False, option: True},
        }
        values = self._d128_coverage_operands(4, 4096, fp32)
        with V.set_graph_handler(_tdm_graph()):
            actual = self._d128_coverage_routes(values, fp32, **overrides[case])
        self.assertEqual(actual[0], expected)

    @common_utils.parametrize("fp32", (False, True))
    @common_utils.parametrize("prefix", ("", "fwd_"))
    def test_d128_coverage_rejects_request_aliases(self, fp32, prefix):
        from torch._inductor.virtualized import V

        option = (
            "ROCM_D128_FP32_BALANCED64_COVERAGE" if fp32 else "ROCM_D128_FP16_COVERAGE"
        )
        values = self._d128_coverage_operands(4, 4096, fp32)
        with V.set_graph_handler(_tdm_graph()):
            actual = self._d128_coverage_routes(values, fp32, **{prefix + option: True})
        self.assertEqual(actual[0], False)

    @common_utils.parametrize("fp32", (False, True))
    @common_utils.parametrize(
        "case,value",
        (
            ("true", True),
            ("false", False),
            ("missing", None),
            ("numeric", 1),
            ("string", "True"),
        ),
    )
    def test_d128_coverage_indexed_blocks(self, fp32, case, value):
        from torch._inductor.virtualized import V

        values = self._d128_coverage_operands(4, 4096, fp32)
        with V.set_graph_handler(_tdm_graph()):
            actual = self._d128_coverage_routes(
                values, fp32, BLOCKS_ARE_CONTIGUOUS=value
            )
        admitted = type(value) is bool
        self.assertEqual(actual, (admitted, True, admitted, admitted))

    @common_utils.parametrize(
        "batch,length,fp32",
        (
            (1, 4096, False),
            (2, 4096, False),
            (2, 2048, False),
            (1, 4096, True),
            (2, 4096, True),
            (1, 2048, True),
            (2, 2048, True),
        ),
    )
    @common_utils.parametrize("field", ("batch", "length", "dtype", "stride"))
    @common_utils.parametrize("index", tuple(range(6)))
    def test_d128_coverage_joint_operand_boundaries(
        self, batch, length, fp32, field, index
    ):
        from torch._inductor.virtualized import V

        values = self._d128_coverage_operands(batch, length, fp32, field, index)
        if field == "length":
            other_length = 4096 if length == 2048 else 2048
            values[index] = self._d128_coverage_operands(batch, other_length, fp32)[
                index
            ]
        with V.set_graph_handler(_tdm_graph()):
            actual = self._d128_coverage_routes(values, fp32)
        self.assertEqual(actual, (False, index >= 3, index >= 4, index >= 4))

    def _d128_coverage_operands(self, batch, length, fp32, field=None, index=None):
        values = []
        for current, heads in enumerate((64, 8, 8, 64, 64, 64)):
            shape = [batch, heads, length]
            if current != 5:
                shape.append(128)
            if current == index and field == "batch":
                shape[0] = 4
            if current == index and field == "length":
                shape[2] = 2048
            strides, stride = [], 1
            for size in reversed(shape):
                strides.insert(0, stride)
                stride *= size
            if current == index and field == "stride":
                strides[-1] = 2
            dtype = torch.float32 if fp32 or current == 5 else torch.float16
            if current == index and field == "dtype":
                dtype = torch.float16 if current == 5 else torch.bfloat16
            values.append(
                _tdm_operand("joint_" + str(current), strides, size=shape, dtype=dtype)
            )
        return values

    def _d128_coverage_routes(self, values, fp32, **option_overrides):
        from torch._inductor.kernel.flex.flex_attention import (
            _get_rocm_backward_config_options,
            _normalize_rocm_backward_options,
            _use_rocm_d128_dv_view_carry_split64,
            _use_rocm_d128_fp16_coverage,
            _use_rocm_d128_fp32_balanced64_coverage,
            _use_rocm_d128_reverse_causal_dq_rows,
            _use_rocm_separate_dq,
        )

        options = dict(
            ROCM_SEPARATE_DQ=True,
            ROCM_EARLY_DP=True,
            XCD_REMAP=True,
            BLOCKS_ARE_CONTIGUOUS=True,
            SAFE_HEAD_DIM=True,
            IS_DIVISIBLE=True,
            HAS_FULL_BLOCKS=True,
            USE_TMA=False,
            PRESCALE_QK=False,
            SM_SCALE=128**-0.5,
            FLOAT32_PRECISION="'ieee'",
            ROCM_BWD_PHASE=0,
            ROCM_GQA_SPLITS=1,
            GQA_SHARED_HEADS=8,
            QK_HEAD_DIM=128,
            V_HEAD_DIM=128,
            QK_HEAD_DIM_ROUNDED=128,
            V_HEAD_DIM_ROUNDED=128,
            BLOCK_M1=16 if fp32 else 32,
            BLOCK_N1=64 if fp32 else 128,
            BLOCK_M2=64 if fp32 else 128,
            BLOCK_N2=16 if fp32 else 32,
            num_warps=4,
            num_stages=1 if fp32 else 2,
            matrix_instr_nonkdim=0,
            waves_per_eu=0,
            kpack=1,
            SPARSE_Q_BLOCK_SIZE=128,
            SPARSE_KV_BLOCK_SIZE=128,
        )
        coverage = (
            _use_rocm_d128_fp32_balanced64_coverage
            if fp32
            else _use_rocm_d128_fp16_coverage
        )
        flags = dict(requested=True, mask_is_causal=True, grad_logsumexp_is_none=True)
        if option_overrides:
            option = (
                "ROCM_D128_FP32_BALANCED64_COVERAGE"
                if fp32
                else "ROCM_D128_FP16_COVERAGE"
            )
            options["bwd_" + option] = True
            options.update(option_overrides)
            requests = _normalize_rocm_backward_options(options)
            options = _get_rocm_backward_config_options(options, is_gfx950=True)
            flags["requested"] = (
                requests.fp32_balanced64_coverage_requested
                if fp32
                else requests.fp16_coverage_requested
            )
            flags["other_requests"] = (
                requests.fp32_balanced64_coverage_other_requests
                if fp32
                else requests.fp16_coverage_other_requests
            )
        admitted = coverage(True, *values, options, True, False, **flags)
        pilot = dict(fp16_pilot=not fp32, fp32_balanced64_pilot=fp32)
        separate = _use_rocm_separate_dq(
            True, *values[:3], options, True, False, **pilot
        )
        reverse = _use_rocm_d128_reverse_causal_dq_rows(
            separate, True, *values[:4], options, True, False, **flags, **pilot
        )
        dv = _use_rocm_d128_dv_view_carry_split64(
            separate,
            True,
            *values[:4],
            options,
            True,
            False,
            reverse_requested=True,
            **flags,
            **pilot,
        )
        return admitted, separate, reverse, dv

    @common_utils.parametrize(
        "query_length,key_length,do_length,out_length,lse_length,expected",
        (
            (2048, 2048, 2048, 2048, 2048, True),
            (4096, 4096, 4096, 4096, 4096, True),
            (1024, 1024, 1024, 1024, 1024, False),
            (8192, 8192, 8192, 8192, 8192, False),
            (4096, 2048, 4096, 4096, 4096, False),
            (2048, 4096, 2048, 2048, 2048, False),
            (4096, 4096, 2048, 4096, 4096, False),
            (4096, 4096, 4096, 2048, 4096, False),
            (4096, 4096, 4096, 4096, 2048, False),
        ),
    )
    def test_d128_fp32_balanced64_joint_length_scope(
        self, query_length, key_length, do_length, out_length, lse_length, expected
    ):
        from torch._inductor.kernel.flex.flex_attention import (
            _use_rocm_d128_fp32_balanced64_coverage,
        )
        from torch._inductor.virtualized import V

        shapes = (
            (4, 64, query_length, 128),
            (4, 8, key_length, 128),
            (4, 8, key_length, 128),
            (4, 64, do_length, 128),
            (4, 64, out_length, 128),
            (4, 64, lse_length),
        )
        operands = []
        for shape in shapes:
            strides, stride = [], 1
            for size in reversed(shape):
                strides.insert(0, stride)
                stride *= size
            operands.append(
                types.SimpleNamespace(
                    get_size=lambda shape=shape: shape,
                    get_stride=lambda strides=strides: strides,
                    get_dtype=lambda: torch.float32,
                    get_device=lambda: torch.device("cuda"),
                )
            )
        options = dict(
            ROCM_SEPARATE_DQ=True,
            ROCM_EARLY_DP=True,
            XCD_REMAP=True,
            BLOCKS_ARE_CONTIGUOUS=True,
            SAFE_HEAD_DIM=True,
            IS_DIVISIBLE=True,
            HAS_FULL_BLOCKS=True,
            USE_TMA=False,
            PRESCALE_QK=False,
            SM_SCALE=1 / math.sqrt(128),
            ROCM_BWD_PHASE=0,
            ROCM_GQA_SPLITS=1,
            GQA_SHARED_HEADS=8,
            QK_HEAD_DIM=128,
            V_HEAD_DIM=128,
            QK_HEAD_DIM_ROUNDED=128,
            V_HEAD_DIM_ROUNDED=128,
            BLOCK_M1=16,
            BLOCK_N1=64,
            BLOCK_M2=64,
            BLOCK_N2=16,
            num_warps=4,
            num_stages=1,
            matrix_instr_nonkdim=0,
            waves_per_eu=0,
            kpack=1,
            SPARSE_Q_BLOCK_SIZE=128,
            SPARSE_KV_BLOCK_SIZE=128,
            FLOAT32_PRECISION="'ieee'",
        )
        sizevars = types.SimpleNamespace(
            statically_known_equals=lambda first, second: first == second
        )
        with V.set_graph_handler(types.SimpleNamespace(sizevars=sizevars)):
            actual = _use_rocm_d128_fp32_balanced64_coverage(
                True,
                *operands,
                options,
                True,
                False,
                requested=True,
                mask_is_causal=True,
                grad_logsumexp_is_none=True,
            )
        self.assertIs(actual, expected)

    @common_utils.parametrize(
        "case",
        (
            "enabled",
            "b2_n4096",
            "b1_n4096",
            "b1_n2048",
            "b2_n2048",
            "kpack2",
            "zero_scale",
            "negative_scale",
            "default",
            "numeric",
            "architecture",
            "score",
            "capture",
            "mask",
            "grad_lse",
            "query_dtype",
            "key_dtype",
            "value_dtype",
            "do_dtype",
            "out_dtype",
            "lse_dtype",
            "batch",
            "heads",
            "length",
            "key_shape",
            "out_shape",
            "lse_shape",
            "query_stride",
            "out_stride",
            "stages",
            "warps",
            "phase",
            "gqa",
            "prescale",
            "tma",
            "precision",
            "boolean_precision",
            "boolean_tiles",
            "boolean_stages",
            "boolean_scale",
            "scale_inf",
            "other_requests",
            "raw",
            "forward",
        ),
    )
    def test_d128_fp32_balanced64_coverage_guard(self, case):
        from torch._inductor.kernel.flex.flex_attention import (
            _use_rocm_d128_fp32_balanced64_coverage,
        )
        from torch._inductor.virtualized import V

        names = ("query", "key", "value", "do", "out", "lse")
        shapes = (
            (4, 64, 2048, 128),
            (4, 8, 2048, 128),
            (4, 8, 2048, 128),
            (4, 64, 2048, 128),
            (4, 64, 2048, 128),
            (4, 64, 2048),
        )
        operands = []
        for name, shape in zip(names, shapes, strict=True):
            sizes = list(shape)
            if case == "b2_n2048":
                sizes[0] = 2
            if case == "b1_n2048":
                sizes[0] = 1
            if case == "b1_n4096":
                sizes[0] = 1
                sizes[2] = 4096
            if case == "b2_n4096":
                sizes[0] = 2
                sizes[2] = 4096
            if case == "batch":
                sizes[0] = 3
            if case == "heads" and name == "query":
                sizes[1] = 32
            if case == "length":
                sizes[2] = 8192
            if case == name + "_shape":
                sizes[2] = 1024
            strides, stride = [], 1
            for size in reversed(sizes):
                strides.insert(0, stride)
                stride *= size
            if case == name + "_stride":
                strides[-1] = 2
            dtype = torch.float32
            if case == name + "_dtype":
                dtype = torch.bfloat16
            operands.append(
                types.SimpleNamespace(
                    get_size=lambda sizes=sizes: sizes,
                    get_stride=lambda strides=strides: strides,
                    get_dtype=lambda dtype=dtype: dtype,
                    get_device=lambda: torch.device("cuda"),
                )
            )
        options = dict(
            ROCM_SEPARATE_DQ=True,
            ROCM_EARLY_DP=True,
            XCD_REMAP=True,
            BLOCKS_ARE_CONTIGUOUS=True,
            SAFE_HEAD_DIM=True,
            IS_DIVISIBLE=True,
            HAS_FULL_BLOCKS=True,
            USE_TMA=case == "tma",
            PRESCALE_QK=case == "prescale",
            SM_SCALE=0.0
            if case == "zero_scale"
            else -0.125
            if case == "negative_scale"
            else 128**-0.5,
            ROCM_BWD_PHASE=1 if case == "phase" else 0,
            ROCM_GQA_SPLITS=1,
            GQA_SHARED_HEADS=4 if case == "gqa" else 8,
            QK_HEAD_DIM=128,
            V_HEAD_DIM=128,
            QK_HEAD_DIM_ROUNDED=128,
            V_HEAD_DIM_ROUNDED=128,
            BLOCK_M1=True if case == "boolean_tiles" else 16,
            BLOCK_N1=64,
            BLOCK_M2=64,
            BLOCK_N2=16,
            num_warps=8 if case == "warps" else 4,
            num_stages=True
            if case == "boolean_stages"
            else 2
            if case == "stages"
            else 1,
            matrix_instr_nonkdim=0,
            waves_per_eu=0,
            kpack=2 if case == "kpack2" else 1,
            SPARSE_Q_BLOCK_SIZE=128,
            SPARSE_KV_BLOCK_SIZE=128,
            FLOAT32_PRECISION=True
            if case == "boolean_precision"
            else "'tf32'"
            if case == "precision"
            else "'ieee'",
        )
        if case == "boolean_scale":
            options["SM_SCALE"] = True
        if case == "scale_inf":
            options["SM_SCALE"] = float("inf")
        sizevars = types.SimpleNamespace(
            statically_known_equals=lambda first, second: first == second
        )
        with V.set_graph_handler(types.SimpleNamespace(sizevars=sizevars)):
            actual = _use_rocm_d128_fp32_balanced64_coverage(
                case != "architecture",
                *operands,
                options,
                case != "score",
                case == "capture",
                requested=1
                if case == "numeric"
                else case not in ("default", "raw", "forward"),
                other_requests=case == "other_requests",
                mask_is_causal=case != "mask",
                grad_logsumexp_is_none=case != "grad_lse",
            )
        self.assertIs(
            actual,
            case
            in (
                "enabled",
                "zero_scale",
                "negative_scale",
                "b2_n4096",
                "b1_n4096",
                "b1_n2048",
                "b2_n2048",
            ),
        )

    @common_utils.parametrize("gqa", (1, 4))
    @common_utils.parametrize(
        "case",
        (
            "default",
            "disabled",
            "numeric_optin",
            "architecture",
            "fp16",
            "fp32",
            "key_dtype",
            "query_dim",
            "value_dim",
            "zero_len",
            "broadcast",
            "value_shape",
            "query_stride",
            "key_stride",
            "value_stride",
            "gqa",
            "gqa_metadata",
            "unknown_gqa",
            "unknown_len",
            "score",
            "mask",
            "score_capture",
            "mask_capture",
            "tma",
            "prescale",
            "consumers",
            "panels",
            "separate",
            "split",
            "early_dp",
            "xcd_remap",
            "m1",
            "n1",
            "m2",
            "n2",
            "warps",
            "stages",
            "phase",
            "owners",
            "length_bound",
        ),
    )
    def test_d256_xcd_snake_guard(self, gqa, case):
        from torch._inductor.kernel.flex.flex_attention import _use_rocm_xcd_snake
        from torch._inductor.virtualized import V

        dtype = {"fp16": torch.float16, "fp32": torch.float32}.get(case, torch.bfloat16)
        actual_gqa = 2 if case == "gqa" else gqa
        q_len = 0 if case == "zero_len" else 257
        if case == "unknown_len":
            q_len = sympy.Symbol("snake_q_len", integer=True, positive=True)
        if case == "length_bound":
            q_len = 2147483647 - 62

        def operand(name, size, dtype=dtype):
            strides, stride = [], 1
            for extent in reversed(size):
                strides.append(stride)
                stride *= extent
            if case == name + "_stride":
                strides[1] *= 2
            return _tdm_operand(name, list(reversed(strides)), size=size, dtype=dtype)

        query = operand(
            "query", (1, 2 * actual_gqa, q_len, 128 if case == "query_dim" else 256)
        )
        key = operand(
            "key",
            (2 if case == "broadcast" else 1, 2, 260, 256),
            torch.float16 if case == "key_dtype" else dtype,
        )
        value = operand(
            "value",
            (
                2 if case == "broadcast" else 1,
                2,
                259 if case == "value_shape" else 260,
                128 if case == "value_dim" else 256,
            ),
        )
        options = dict(
            ROCM_XCD_SNAKE=1 if case == "numeric_optin" else case != "disabled",
            USE_TMA=case == "tma",
            PRESCALE_QK=case == "prescale",
            num_consumer_groups=1 if case == "consumers" else 0,
            ROCM_D256_PANEL_DOTS=case == "panels",
            ROCM_SEPARATE_DQ=case == "separate",
            ROCM_GQA_SPLIT_DKDV=case == "split",
            ROCM_EARLY_DP=case == "early_dp",
            XCD_REMAP=case == "xcd_remap",
            BLOCK_M1=64 if case == "m1" else 32,
            BLOCK_N1=128 if case == "n1" else 64,
            BLOCK_M2=128 if case == "m2" else 64,
            BLOCK_N2=32 if case == "n2" else 64,
            num_warps=8 if case == "warps" else 4,
            num_stages=2 if case == "stages" else 1,
            GQA_SHARED_HEADS=4
            if case == "gqa_metadata" and gqa == 1
            else 1
            if case == "gqa_metadata"
            else actual_gqa,
            ROCM_BWD_PHASE=1 if case == "phase" else 0,
            ROCM_GQA_SPLITS=2 if case == "owners" else 1,
        )
        if case == "unknown_gqa":
            options["GQA_SHARED_HEADS"] = sympy.Symbol(
                "snake_gqa", integer=True, positive=True
            )
        with V.set_graph_handler(_tdm_graph()):
            actual = _use_rocm_xcd_snake(
                case != "architecture",
                query,
                key,
                value,
                options,
                case != "score",
                case != "mask",
                case in ("score_capture", "mask_capture"),
            )
        self.assertEqual(actual, case == "default")

    @common_utils.parametrize(
        "case,expected",
        (
            ("ge", True),
            ("le", True),
            ("reversed_ge", False),
            ("reversed_le", False),
            ("scalar_true", False),
            ("scalar_false", False),
            ("numeric_fill", False),
            ("extra_node", False),
            ("different_output", False),
            ("kwargs", False),
            ("extra_placeholder", False),
            ("unknown_target", False),
        ),
    )
    def test_d256_xcd_snake_causal_graph(self, case, expected):
        from torch._inductor.kernel.flex.flex_attention import (
            _is_rocm_simple_causal_mask_graph,
        )

        graph = torch.fx.Graph()
        indices = [graph.placeholder(name) for name in ("b", "h", "q_idx", "kv_idx")]
        if case == "extra_placeholder":
            graph.placeholder("captured")
        if case in ("scalar_true", "scalar_false", "numeric_fill"):
            result = graph.call_function(
                torch.ops.aten.full.default,
                ([], 1 if case == "numeric_fill" else case == "scalar_true"),
                {"dtype": torch.bool},
            )
        else:
            target = (
                torch.ops.aten.le.Tensor
                if case in ("le", "reversed_le")
                else torch.ops.aten.ge.Tensor
            )
            if case == "unknown_target":
                target = torch.ops.aten.eq.Tensor
            left, right = (
                (indices[3], indices[2])
                if case in ("le", "reversed_ge")
                else (indices[2], indices[3])
            )
            result = graph.call_function(
                target, (left, right), {"unknown": True} if case == "kwargs" else {}
            )
        if case == "extra_node":
            graph.call_function(torch.ops.aten.add.Tensor, (indices[2], indices[3]))
        graph.output(indices[2] if case == "different_output" else result)
        self.assertEqual(
            _is_rocm_simple_causal_mask_graph(types.SimpleNamespace(graph=graph)),
            expected,
        )

    @common_utils.parametrize(
        "gfx950,dtype,qk_dim,v_dim,use_tma,requested,expected",
        (
            (True, torch.bfloat16, 256, 256, False, True, True),
            (True, torch.bfloat16, 256, 256, False, None, False),
            (True, torch.bfloat16, 256, 256, False, False, False),
            (False, torch.bfloat16, 256, 256, False, True, False),
            (True, torch.float16, 256, 256, False, True, False),
            (True, torch.float32, 256, 256, False, True, False),
            (True, torch.bfloat16, 128, 256, False, True, False),
            (True, torch.bfloat16, 256, 128, False, True, False),
            (True, torch.bfloat16, 255, 256, False, True, False),
            (True, torch.bfloat16, 256, 255, False, True, False),
            (True, torch.bfloat16, 256, 256, True, True, False),
        ),
    )
    def test_d256_panel_guard(
        self, gfx950, dtype, qk_dim, v_dim, use_tma, requested, expected
    ):
        from torch._inductor.kernel.flex.flex_attention import _use_rocm_d256_panel_dots

        options = {"QK_HEAD_DIM": qk_dim, "V_HEAD_DIM": v_dim, "USE_TMA": use_tma}
        if requested is not None:
            options["ROCM_D256_PANEL_DOTS"] = requested
        self.assertEqual(
            _use_rocm_d256_panel_dots(gfx950, dtype, qk_dim, v_dim, options), expected
        )

    @common_utils.parametrize(
        "case",
        (
            "default",
            "disabled",
            "bf16",
            "fp32",
            "gqa2",
            "gqa8",
            "broadcast",
            "key_stride",
            "value_stride",
            "architecture",
            "captures",
            "score_mod",
            "custom_mask",
            "unknown_mask",
            "tma",
            "prescale",
            "consumers",
            "tiles",
        ),
    )
    def test_gqa_split_dkdv_guard(self, case):
        from torch._inductor.kernel.flex.flex_attention import _use_rocm_gqa_split_dkdv
        from torch._inductor.virtualized import V

        dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}.get(case, torch.float16)
        gqa = {"gqa2": 2, "gqa8": 8}.get(case, 4)
        batch_kv = 1 if case == "broadcast" else 2

        def operand(name, size):
            strides, stride = [], 1
            for extent in reversed(size):
                strides.append(stride)
                stride *= extent
            if case == name + "_stride":
                strides[1] *= 2
            return _tdm_operand(name, list(reversed(strides)), size=size, dtype=dtype)

        query = operand("query", (2, 4 * gqa, 257, 64))
        key = operand("key", (batch_kv, 4, 260, 64))
        value = operand("value", (batch_kv, 4, 260, 64))
        options = dict(
            ROCM_GQA_SPLIT_DKDV=case != "disabled",
            USE_TMA=case == "tma",
            PRESCALE_QK=case == "prescale",
            num_consumer_groups=2 if case == "consumers" else 0,
            BLOCK_M1=16 if case == "tiles" else 32,
            BLOCK_N1=128,
            BLOCK_M2=128,
            BLOCK_N2=32,
            num_warps=4,
            num_stages=2,
            GQA_SHARED_HEADS=gqa,
        )
        with V.set_graph_handler(_tdm_graph()):
            actual = _use_rocm_gqa_split_dkdv(
                case != "architecture",
                query,
                key,
                value,
                options,
                case != "score_mod",
                case != "custom_mask",
                case != "unknown_mask",
                case == "captures",
            )
        self.assertEqual(actual, case == "default")

    @common_utils.parametrize(
        "case",
        (
            "default",
            "length",
            "block_size",
            "full_blocks",
            "ordering",
            "counts",
            "indices",
            "missing_q",
        ),
    )
    def test_gqa_split_default_mask_structure(self, case):
        from torch._inductor.kernel.flex.flex_attention import (
            _is_rocm_default_block_mask,
        )
        from torch._inductor.virtualized import V
        from torch.nn.attention.flex_attention import _LARGE_SPARSE_BLOCK_SIZE

        def metadata(name, rank):
            return _tdm_operand(name, [1] * rank, size=[1] * rank, dtype=torch.int32)

        query = _tdm_operand("query", [131584, 16448, 64, 1], size=[1, 8, 257, 64])
        key = _tdm_operand("key", [33280, 16640, 64, 1], size=[1, 2, 260, 64])
        block_mask = [
            1,
            1,
            metadata("kv_num", 3),
            metadata("kv_idx", 4),
            None,
            None,
            metadata("q_num", 3),
            metadata("q_idx", 4),
            None,
            None,
            None,
            None,
            None,
            None,
            _LARGE_SPARSE_BLOCK_SIZE,
            _LARGE_SPARSE_BLOCK_SIZE,
            None,
        ]
        if case == "length":
            block_mask[0] = 257
        elif case == "block_size":
            block_mask[14] = 128
        elif case == "full_blocks":
            block_mask[4] = metadata("full_num", 3)
        elif case == "ordering":
            block_mask[10] = metadata("order", 3)
        elif case == "counts":
            block_mask[2] = _tdm_operand(
                "counts", [2, 2, 1], size=[1, 1, 2], dtype=torch.int32
            )
        elif case == "indices":
            block_mask[3] = _tdm_operand(
                "indices", [2, 2, 2, 1], size=[1, 1, 1, 2], dtype=torch.int32
            )
        elif case == "missing_q":
            block_mask[6] = None
        with V.set_graph_handler(_tdm_graph()):
            actual = _is_rocm_default_block_mask(tuple(block_mask), query, key)
        self.assertEqual(actual, case == "default")

    @common_utils.parametrize(
        "phase,splits,expected",
        ((0, 1, (15, 1, 2)), (1, 2, (3, 1, 4)), (2, 1, (12, 1, 2))),
    )
    def test_gqa_split_grid(self, phase, splits, expected):
        from torch._inductor.kernel.flex.flex_attention import (
            flex_attention_backward_grid,
        )

        meta = dict(
            BLOCK_M2=128, BLOCK_N1=128, ROCM_BWD_PHASE=phase, ROCM_GQA_SPLITS=splits
        )
        self.assertEqual(
            flex_attention_backward_grid(1, 8, 257, 64, 2, 260, meta), expected
        )

    @common_utils.parametrize(
        "case",
        (
            "true",
            "false",
            "integer",
            "float",
            "shape",
            "dtype",
            "tensor_fill",
            "output",
            "placeholders",
        ),
    )
    def test_d256_scalar_true_mask_graph(self, case):
        from torch._inductor.kernel.flex.flex_attention import (
            _is_rocm_scalar_true_mask_graph,
        )

        graph = torch.fx.Graph()
        placeholders = [graph.placeholder(name) for name in ("b", "h", "q", "kv")]
        if case == "placeholders":
            graph.placeholder("extra")
        fill = {"false": False, "integer": 1, "float": 1.0}.get(case, True)
        if case == "tensor_fill":
            fill = placeholders[0]
        output = graph.call_function(
            torch.ops.aten.full.default,
            ([1] if case == "shape" else [], fill),
            {"dtype": torch.float32 if case == "dtype" else torch.bool},
        )
        graph.output(placeholders[0] if case == "output" else output)
        self.assertEqual(
            _is_rocm_scalar_true_mask_graph(torch.fx.GraphModule({}, graph)),
            case == "true",
        )

    @common_utils.parametrize("gqa", (1, 4))
    @common_utils.parametrize("dq_m,dq_n", ((64, 16), (128, 32)))
    @common_utils.parametrize(
        "case",
        (
            "default",
            "disabled",
            "architecture",
            "fp16",
            "fp32",
            "key_dtype",
            "value_dim",
            "zero_len",
            "broadcast",
            "value_shape",
            "stride",
            "gqa",
            "gqa_metadata",
            "unknown_gqa_metadata",
            "score",
            "captures",
            "false_mask",
            "custom_block",
            "panels",
            "tma",
            "prescale",
            "consumers",
            "tiles",
            "warps",
            "stages",
        ),
    )
    def test_d256_dense_separate_dq_guard(self, gqa, dq_m, dq_n, case):
        from torch._inductor.kernel.flex.flex_attention import _use_rocm_separate_dq
        from torch._inductor.virtualized import V

        dtype = {"fp16": torch.float16, "fp32": torch.float32}.get(case, torch.bfloat16)
        actual_gqa = 2 if case == "gqa" else gqa

        def operand(name, size, dtype=dtype):
            strides, stride = [], 1
            for extent in reversed(size):
                strides.append(stride)
                stride *= extent
            if case == "stride" and name == "value":
                strides[1] *= 2
            return _tdm_operand(name, list(reversed(strides)), size=size, dtype=dtype)

        query = operand(
            "query", (2, 4 * actual_gqa, 0 if case == "zero_len" else 257, 256)
        )
        key = operand(
            "key",
            (1 if case == "broadcast" else 2, 4, 260, 256),
            torch.float16 if case == "key_dtype" else dtype,
        )
        value = operand(
            "value",
            (
                1 if case == "broadcast" else 2,
                4,
                259 if case == "value_shape" else 260,
                128 if case == "value_dim" else 256,
            ),
        )
        options = dict(
            ROCM_SEPARATE_DQ=case != "disabled",
            ROCM_D256_PANEL_DOTS=case != "panels",
            USE_TMA=case == "tma",
            PRESCALE_QK=case == "prescale",
            num_consumer_groups=1 if case == "consumers" else 0,
            BLOCK_M1=32 if case == "tiles" else 16,
            BLOCK_N1=64,
            BLOCK_M2=dq_m,
            BLOCK_N2=dq_n,
            num_warps=8 if case == "warps" else 4,
            num_stages=1 if case == "stages" else 2,
            GQA_SHARED_HEADS=(4 if gqa == 1 else 1)
            if case == "gqa_metadata"
            else actual_gqa,
            ROCM_BWD_PHASE=1,
            ROCM_GQA_SPLITS=2,
        )
        if case == "unknown_gqa_metadata":
            options["GQA_SHARED_HEADS"] = sympy.Symbol(
                "gqa_metadata", integer=True, positive=True
            )
        with V.set_graph_handler(_tdm_graph()):
            actual = _use_rocm_separate_dq(
                case != "architecture",
                query,
                key,
                value,
                options,
                case != "score",
                case == "captures",
                mask_is_scalar_true=case != "false_mask",
                block_mask_is_default=case != "custom_block",
            )
        self.assertEqual(actual, case == "default")

    @common_utils.parametrize(
        "mask_is_scalar_true,block_mask_is_default",
        ((False, False), (True, False), (False, True), (True, True)),
    )
    def test_d256_dense_guard_preserves_d128(
        self, mask_is_scalar_true, block_mask_is_default
    ):
        from torch._inductor.kernel.flex.flex_attention import _use_rocm_separate_dq
        from torch._inductor.virtualized import V

        query = _tdm_operand(
            "query",
            [526336, 32896, 128, 1],
            size=[1, 16, 257, 128],
            dtype=torch.bfloat16,
        )
        key = _tdm_operand(
            "key", [66560, 33280, 128, 1], size=[1, 2, 260, 128], dtype=torch.bfloat16
        )
        value = _tdm_operand(
            "value", [66560, 33280, 128, 1], size=[1, 2, 260, 128], dtype=torch.bfloat16
        )
        options = dict(
            ROCM_SEPARATE_DQ=True,
            ROCM_D256_PANEL_DOTS=False,
            USE_TMA=False,
            PRESCALE_QK=False,
            BLOCK_M1=32,
            BLOCK_N1=128,
            BLOCK_M2=128,
            BLOCK_N2=32,
            num_warps=4,
            num_stages=2,
            GQA_SHARED_HEADS=8,
        )
        with V.set_graph_handler(_tdm_graph()):
            actual = _use_rocm_separate_dq(
                True,
                query,
                key,
                value,
                options,
                True,
                False,
                mask_is_scalar_true=mask_is_scalar_true,
                block_mask_is_default=block_mask_is_default,
            )
        self.assertTrue(actual)

    def test_d256_dense_guard_preserves_fp16(self):
        from torch._inductor.kernel.flex.flex_attention import (
            _use_rocm_gqa_split_dkdv,
            _use_rocm_separate_dq,
        )
        from torch._inductor.virtualized import V

        query = _tdm_operand(
            "query", [131584, 16448, 64, 1], size=[1, 8, 257, 64], dtype=torch.float16
        )
        key = _tdm_operand(
            "key", [33280, 16640, 64, 1], size=[1, 2, 260, 64], dtype=torch.float16
        )
        value = _tdm_operand(
            "value", [33280, 16640, 64, 1], size=[1, 2, 260, 64], dtype=torch.float16
        )
        options = dict(
            ROCM_SEPARATE_DQ=True,
            ROCM_GQA_SPLIT_DKDV=True,
            ROCM_D256_PANEL_DOTS=False,
            USE_TMA=False,
            PRESCALE_QK=False,
            BLOCK_M1=32,
            BLOCK_N1=128,
            BLOCK_M2=128,
            BLOCK_N2=32,
            num_warps=4,
            num_stages=2,
            GQA_SHARED_HEADS=4,
        )
        with V.set_graph_handler(_tdm_graph()):
            separate = _use_rocm_separate_dq(
                True,
                query,
                key,
                value,
                options,
                True,
                False,
                mask_is_scalar_true=True,
                block_mask_is_default=True,
            )
            split = _use_rocm_gqa_split_dkdv(
                True, query, key, value, options, True, True, True, False
            )
        self.assertFalse(separate)
        self.assertTrue(split)

    @common_utils.parametrize("gqa", (1, 4))
    @common_utils.parametrize(
        "case",
        (
            "enabled",
            "missing",
            "disabled",
            "numeric_request",
            "raw",
            "architecture",
            "fp16",
            "fp32",
            "key_dtype",
            "value_dim",
            "broadcast",
            "value_shape",
            "zero_len",
            "stride",
            "gqa",
            "gqa_metadata",
            "unknown_gqa_metadata",
            "score",
            "captures",
            "false_mask",
            "custom_block",
            "panels",
            "separate",
            "early_dp",
            "numeric_early_dp",
            "tma",
            "prescale",
            "consumers",
            "m1",
            "n1",
            "m2",
            "n2",
            "warps",
            "stages",
        ),
    )
    def test_early_probability_compact_guard(self, gqa, case):
        from torch._inductor.kernel.flex.flex_attention import (
            _use_rocm_early_probability_compact,
        )
        from torch._inductor.virtualized import V

        dtype = {"fp16": torch.float16, "fp32": torch.float32}.get(case, torch.bfloat16)
        actual_gqa = 2 if case == "gqa" else gqa

        def operand(name, shape, dtype=dtype):
            strides, stride = [], 1
            for extent in reversed(shape):
                strides.insert(0, stride)
                stride *= extent
            if case == "stride" and name == "value":
                strides[-2] *= 2
            return _tdm_operand(name, strides, size=shape, dtype=dtype)

        query = operand(
            "query", (2, 4 * actual_gqa, 0 if case == "zero_len" else 257, 256)
        )
        key = operand(
            "key",
            (1 if case == "broadcast" else 2, 4, 260, 256),
            torch.float16 if case == "key_dtype" else dtype,
        )
        value = operand(
            "value",
            (
                1 if case == "broadcast" else 2,
                4,
                259 if case == "value_shape" else 260,
                128 if case == "value_dim" else 256,
            ),
        )
        options = dict(
            ROCM_SEPARATE_DQ=case != "separate",
            ROCM_D256_PANEL_DOTS=case != "panels",
            ROCM_EARLY_DP=1 if case == "numeric_early_dp" else case != "early_dp",
            ROCM_EARLY_PROBABILITY_COMPACT=True,
            USE_TMA=case == "tma",
            PRESCALE_QK=case == "prescale",
            num_consumer_groups=1 if case == "consumers" else 0,
            BLOCK_M1=32 if case == "m1" else 16,
            BLOCK_N1=128 if case == "n1" else 64,
            BLOCK_M2=64 if case == "m2" else 128,
            BLOCK_N2=16 if case == "n2" else 32,
            num_warps=8 if case == "warps" else 4,
            num_stages=1 if case == "stages" else 2,
            GQA_SHARED_HEADS=(4 if gqa == 1 else 1)
            if case == "gqa_metadata"
            else actual_gqa,
        )
        if case == "unknown_gqa_metadata":
            options["GQA_SHARED_HEADS"] = sympy.Symbol(
                "unknown_gqa", integer=True, positive=True
            )
        requested = 1 if case == "numeric_request" else case not in ("disabled", "raw")
        keywords = {} if case == "missing" else {"requested": requested}
        with V.set_graph_handler(_tdm_graph()):
            actual = _use_rocm_early_probability_compact(
                case != "architecture",
                query,
                key,
                value,
                options,
                case != "score",
                case == "captures",
                mask_is_scalar_true=case != "false_mask",
                block_mask_is_default=case != "custom_block",
                **keywords,
            )
        self.assertEqual(actual, case == "enabled")

    @common_utils.parametrize(
        "case",
        (
            "enabled",
            "gqa4_pilot_enabled",
            "gqa4_pilot_zero_scale",
            "gqa4_pilot_negative_scale",
            "gqa4_pilot_query_length",
            "gqa4_pilot_key_length",
            "gqa4_pilot_value_length",
            "gqa4_pilot_do_length",
            "gqa4_pilot_gqa_metadata",
            "gqa4_pilot_unknown_gqa",
            "gqa4_pilot_gqa2",
            "gqa4_pilot_small_batch",
            "gqa4_pilot_small_heads",
            "gqa4_pilot_sequence4096",
            "gqa4_pilot_missing",
            "gqa4_pilot_disabled",
            "gqa4_pilot_numeric_request",
            "gqa4_pilot_raw",
            "gqa4_pilot_forward",
            "gqa4_pilot_captures",
            "gqa4_pilot_grad_lse",
            "gqa4_pilot_score",
            "gqa4_pilot_false_mask",
            "gqa4_pilot_custom_block",
            "gqa4_pilot_do_dtype",
            "gqa4_pilot_do_stride",
            "gqa4_pilot_query_stride",
            "gqa4_pilot_key_stride",
            "gqa4_pilot_value_stride",
            "gqa4_pilot_boolean_warps",
            "gqa4_pilot_boolean_stages",
            "gqa4_pilot_boolean_kpack",
            "gqa4_pilot_string_stages",
            "sequence2048",
            "sequence1024",
            "mixed_lengths",
            "short_both",
            "excluded_tail",
            "small_batch1024",
            "small_batch2048",
            "zero_scale",
            "negative_scale",
            "missing",
            "disabled",
            "numeric_request",
            "raw",
            "architecture",
            "fp16",
            "fp32",
            "key_dtype",
            "value_dtype",
            "query_dim",
            "value_dim",
            "broadcast",
            "value_shape",
            "zero_len",
            "short_q",
            "short_kv",
            "small_batch",
            "small_heads",
            "length_bound",
            "query_stride",
            "key_stride",
            "value_stride",
            "gqa2",
            "gqa4",
            "gqa_metadata",
            "unknown_gqa",
            "unknown_len",
            "score",
            "captures",
            "false_mask",
            "custom_block",
            "grad_lse",
            "tma",
            "prescale",
            "panels",
            "early_dp",
            "compact",
            "xcd_remap",
            "snake",
            "owners",
            "consumers",
            "safe_head",
            "rounded_q",
            "rounded_v",
            "m1",
            "n1",
            "m2",
            "n2",
            "warps",
            "stages",
            "matrix",
            "waves",
            "kpack",
            "nonfinite_scale",
            "missing_do",
            "do_dtype",
            "do_shape",
            "do_rank",
            "do_stride",
        ),
    )
    def test_d192_panel_phase_guard(self, case):
        gqa4_pilot = case.startswith("gqa4_pilot_")
        if gqa4_pilot:
            case = case.removeprefix("gqa4_pilot_")
        from torch._inductor.kernel.flex.flex_attention import (
            _use_rocm_d192_panel_phase,
        )
        from torch._inductor.virtualized import V

        dtype = {"fp16": torch.float16, "fp32": torch.float32}.get(case, torch.bfloat16)
        batch = (
            1
            if case
            in ("small_batch", "short_both", "small_batch1024", "small_batch2048")
            else 4
        )
        heads = 2 if case == "small_heads" else 10
        actual_gqa = {"gqa2": 2, "gqa4": 4}.get(case, 1)
        q_len = {
            "zero_len": 0,
            "short_q": 257,
            "short_both": 257,
            "excluded_tail": 1025,
            "sequence2048": 2048,
            "small_batch2048": 2048,
            "sequence1024": 1024,
            "small_batch1024": 1024,
            "mixed_lengths": 2048,
        }.get(case, 4096)
        kv_len = {
            "short_kv": 260,
            "short_both": 260,
            "excluded_tail": 1030,
            "length_bound": 4097,
            "sequence2048": 2048,
            "small_batch2048": 2048,
            "sequence1024": 1024,
            "small_batch1024": 1024,
        }.get(case, 4096)
        if gqa4_pilot:
            batch = 1 if case == "small_batch" else 2
            heads = 1 if case == "small_heads" else 2
            actual_gqa = 2 if case == "gqa2" else 4
            q_len = kv_len = 4096 if case == "sequence4096" else 2048
        if case == "unknown_len":
            q_len = sympy.Symbol("d192_q_len", integer=True, positive=True)

        def operand(name, shape, dtype=dtype):
            if gqa4_pilot and case == name + "_length":
                shape = (*shape[:2], 4096, shape[3])
            strides, stride = [], 1
            for extent in reversed(shape):
                strides.insert(0, stride)
                stride *= extent
            if case == name + "_stride":
                strides[-2] *= 2
            return _tdm_operand(name, strides, size=shape, dtype=dtype)

        query = operand(
            "query",
            (batch, heads * actual_gqa, q_len, 128 if case == "query_dim" else 192),
        )
        key = operand(
            "key",
            (1 if case == "broadcast" else batch, heads, kv_len, 192),
            torch.float16 if case == "key_dtype" else dtype,
        )
        value = operand(
            "value",
            (
                1 if case == "broadcast" else batch,
                heads,
                kv_len - 1 if case == "value_shape" else kv_len,
                128 if case == "value_dim" else 192,
            ),
            torch.float16 if case == "value_dtype" else dtype,
        )
        do_shape = (
            batch,
            heads * actual_gqa,
            q_len - 1 if case == "do_shape" else q_len,
            192,
        )
        if case == "do_rank":
            do_shape = do_shape[:-1]
        grad_out = operand(
            "do", do_shape, torch.float32 if case == "do_dtype" else torch.bfloat16
        )
        options = dict(
            ROCM_D192_PANEL_DOTS=True,
            SAFE_HEAD_DIM=case == "safe_head",
            QK_HEAD_DIM=192,
            V_HEAD_DIM=192,
            QK_HEAD_DIM_ROUNDED=192 if case == "rounded_q" else 256,
            V_HEAD_DIM_ROUNDED=192 if case == "rounded_v" else 256,
            BLOCK_M1=32 if case == "m1" else 16,
            BLOCK_N1=128 if case == "n1" else 64,
            BLOCK_M2=64 if case == "m2" else 128,
            BLOCK_N2=16 if case == "n2" else 32,
            num_warps=8 if case == "warps" else 4,
            num_stages=1 if case == "stages" else 2,
            matrix_instr_nonkdim=16 if case == "matrix" else 0,
            waves_per_eu=1 if case == "waves" else 0,
            kpack=2 if case == "kpack" else 1,
            GQA_SHARED_HEADS=4 if case == "gqa_metadata" else actual_gqa,
            USE_TMA=case == "tma",
            PRESCALE_QK=case == "prescale",
            ROCM_D256_PANEL_DOTS=case == "panels",
            ROCM_EARLY_DP=case == "early_dp",
            ROCM_EARLY_PROBABILITY_COMPACT=case == "compact",
            XCD_REMAP=case == "xcd_remap",
            ROCM_XCD_SNAKE=case == "snake",
            ROCM_GQA_SPLIT_DKDV=case == "owners",
            num_consumer_groups=1 if case == "consumers" else 0,
            SM_SCALE=0.0
            if case == "zero_scale"
            else (-0.125 if gqa4_pilot else -0.07)
            if case == "negative_scale"
            else float("inf")
            if case == "nonfinite_scale"
            else (192**-0.5 if gqa4_pilot else 0.07),
        )
        if gqa4_pilot and case == "gqa_metadata":
            options["GQA_SHARED_HEADS"] = 8
        if gqa4_pilot and case in (
            "boolean_warps",
            "boolean_stages",
            "boolean_kpack",
            "string_stages",
        ):
            key, value = {
                "boolean_warps": ("num_warps", True),
                "boolean_stages": ("num_stages", True),
                "boolean_kpack": ("kpack", False),
                "string_stages": ("num_stages", "2"),
            }[case]
            options[key] = value
        if case == "unknown_gqa":
            options["GQA_SHARED_HEADS"] = sympy.Symbol(
                "d192_gqa", integer=True, positive=True
            )
        requested = (
            1
            if case == "numeric_request"
            else case not in ("disabled", "raw", "forward")
        )
        keywords = {} if case == "missing" else {"requested": requested}
        with V.set_graph_handler(_tdm_graph()):
            actual = _use_rocm_d192_panel_phase(
                case != "architecture",
                query,
                key,
                value,
                options,
                case != "score",
                case == "captures",
                mask_is_scalar_true=case != "false_mask",
                block_mask_is_default=case != "custom_block",
                grad_logsumexp_is_none=case != "grad_lse",
                grad_out=None if case == "missing_do" else grad_out,
                **keywords,
            )
        self.assertEqual(
            actual, case in ("enabled", "sequence2048", "zero_scale", "negative_scale")
        )

    @common_utils.parametrize("gqa", (1, 4))
    @common_utils.parametrize(
        "case",
        (
            "enabled",
            "zero_scale",
            "negative_scale",
            "missing",
            "disabled",
            "numeric_request",
            "raw",
            "forward_only",
            "architecture",
            "fp16",
            "fp32",
            "key_dtype",
            "value_dtype",
            "query_dim",
            "value_dim",
            "broadcast",
            "value_shape",
            "short",
            "sequence2048",
            "sequence1024",
            "mixed_lengths",
            "zero_len",
            "negative_len",
            "large_len",
            "batch",
            "heads",
            "key_heads",
            "unknown_batch",
            "unknown_heads",
            "unknown_len",
            "unknown_stride",
            "query_stride",
            "key_stride",
            "value_stride",
            "gqa2",
            "gqa8",
            "gqa_metadata",
            "unknown_gqa",
            "score",
            "captures",
            "false_mask",
            "custom_block",
            "grad_lse",
            "tma",
            "prescale",
            "panels",
            "d192",
            "early_dp",
            "compact",
            "xcd_remap",
            "snake",
            "owners",
            "consumers",
            "safe_head",
            "rounded_q",
            "rounded_v",
            "m1",
            "n1",
            "m2",
            "n2",
            "warps",
            "stages",
            "matrix",
            "waves",
            "kpack",
            "nonfinite_scale",
            "missing_do",
            "do_dtype",
            "do_shape",
            "do_rank",
            "do_stride",
        ),
    )
    def test_d96_panel_phase_guard(self, gqa, case):
        from torch._inductor.kernel.flex.flex_attention import _use_rocm_d96_panel_phase
        from torch._inductor.virtualized import V

        dtype = {"fp16": torch.float16, "fp32": torch.float32}.get(case, torch.bfloat16)
        batch, heads = (4, 21) if gqa == 1 else (2, 2)
        if case == "batch":
            batch = 1
        if case == "heads":
            heads += 1
        actual_gqa = {"gqa2": 2, "gqa8": 8}.get(case, gqa)
        q_len = {
            "short": 257,
            "sequence2048": 2048,
            "sequence1024": 1024,
            "mixed_lengths": 2048,
            "zero_len": 0,
            "negative_len": -1,
            "large_len": 4097,
        }.get(case, 4096)
        kv_len = {
            "short": 260,
            "sequence2048": 2048,
            "sequence1024": 1024,
            "large_len": 4097,
        }.get(case, 4096)
        if case == "unknown_batch":
            batch = sympy.Symbol("d96_batch", integer=True, positive=True)
        if case == "unknown_heads":
            heads = sympy.Symbol("d96_heads", integer=True, positive=True)
        if case == "unknown_len":
            q_len = sympy.Symbol("d96_q_len", integer=True, positive=True)

        def operand(name, shape, dtype=dtype):
            strides, stride = [], 1
            for extent in reversed(shape):
                strides.insert(0, stride)
                stride *= extent
            if case == name + "_stride":
                strides[-2] *= 2
            if case == "unknown_stride" and name == "query":
                strides[-2] = sympy.Symbol("d96_stride", integer=True, positive=True)
            return _tdm_operand(name, strides, size=shape, dtype=dtype)

        query = operand(
            "query",
            (batch, heads * actual_gqa, q_len, 128 if case == "query_dim" else 96),
        )
        key = operand(
            "key",
            (
                1 if case == "broadcast" else batch,
                heads + 1 if case == "key_heads" else heads,
                kv_len,
                96,
            ),
            torch.float16 if case == "key_dtype" else dtype,
        )
        value = operand(
            "value",
            (
                1 if case == "broadcast" else batch,
                heads,
                kv_len - 1 if case == "value_shape" else kv_len,
                128 if case == "value_dim" else 96,
            ),
            torch.float16 if case == "value_dtype" else dtype,
        )
        do_shape = (
            batch,
            heads * actual_gqa,
            q_len - 1 if case == "do_shape" else q_len,
            96,
        )
        if case == "do_rank":
            do_shape = do_shape[:-1]
        grad_out = operand(
            "do", do_shape, torch.float32 if case == "do_dtype" else torch.bfloat16
        )
        options = dict(
            ROCM_D96_PANEL_DOTS=True,
            fwd_ROCM_D96_PANEL_DOTS=case == "forward_only",
            SAFE_HEAD_DIM=case == "safe_head",
            QK_HEAD_DIM=96,
            V_HEAD_DIM=96,
            QK_HEAD_DIM_ROUNDED=96 if case == "rounded_q" else 128,
            V_HEAD_DIM_ROUNDED=96 if case == "rounded_v" else 128,
            BLOCK_M1=16 if case == "m1" else 32,
            BLOCK_N1=64 if case == "n1" else 128,
            BLOCK_M2=64 if case == "m2" else 128,
            BLOCK_N2=16 if case == "n2" else 32,
            num_warps=8 if case == "warps" else 4,
            num_stages=1 if case == "stages" else 2,
            matrix_instr_nonkdim=16 if case == "matrix" else 0,
            waves_per_eu=1 if case == "waves" else 0,
            kpack=2 if case == "kpack" else 1,
            GQA_SHARED_HEADS=(4 if gqa == 1 else 1)
            if case == "gqa_metadata"
            else actual_gqa,
            USE_TMA=case == "tma",
            PRESCALE_QK=case == "prescale",
            ROCM_D256_PANEL_DOTS=case == "panels",
            ROCM_D192_PANEL_DOTS=case == "d192",
            ROCM_EARLY_DP=case == "early_dp",
            ROCM_EARLY_PROBABILITY_COMPACT=case == "compact",
            XCD_REMAP=case == "xcd_remap",
            ROCM_XCD_SNAKE=case == "snake",
            ROCM_GQA_SPLIT_DKDV=case == "owners",
            num_consumer_groups=1 if case == "consumers" else 0,
            SM_SCALE=0.0
            if case == "zero_scale"
            else -0.07
            if case == "negative_scale"
            else float("inf")
            if case == "nonfinite_scale"
            else 0.07,
        )
        if case == "unknown_gqa":
            options["GQA_SHARED_HEADS"] = sympy.Symbol(
                "d96_gqa", integer=True, positive=True
            )
        requested = (
            1
            if case == "numeric_request"
            else case not in ("disabled", "raw", "forward_only")
        )
        keywords = {} if case == "missing" else {"requested": requested}
        with V.set_graph_handler(_tdm_graph()):
            actual = _use_rocm_d96_panel_phase(
                case != "architecture",
                query,
                key,
                value,
                options,
                case != "score",
                case == "captures",
                mask_is_scalar_true=case != "false_mask",
                block_mask_is_default=case != "custom_block",
                grad_logsumexp_is_none=case != "grad_lse",
                grad_out=None if case == "missing_do" else grad_out,
                **keywords,
            )
        self.assertEqual(
            actual, case in ("enabled", "sequence2048", "zero_scale", "negative_scale")
        )

    def _rocm_backward_phase_options(
        self,
        options,
        *,
        operands=None,
        single_config=True,
        use_separate_dq=True,
        use_d192_panel_phase=False,
        use_d96_panel_phase=False,
    ):
        from torch._inductor.kernel.flex.flex_attention import (
            _get_rocm_backward_config_options,
            _get_rocm_backward_phase_options,
            _normalize_rocm_backward_options,
        )
        from torch._inductor.virtualized import V

        if operands is None:
            operands = self._d128_coverage_operands(4, 2048, False)[:4]
        query, key, value, grad_out = operands
        requests = _normalize_rocm_backward_options(options)
        current = _get_rocm_backward_config_options(options, is_gfx950=True)
        with V.set_graph_handler(_tdm_graph()):
            dkdv, dq = _get_rocm_backward_phase_options(
                current,
                [query],
                requests=requests,
                single_config=single_config,
                is_gfx950=True,
                query=query,
                key=key,
                value=value,
                grad_out=grad_out,
                use_separate_dq=use_separate_dq,
                use_d192_panel_phase=use_d192_panel_phase,
                use_d96_panel_phase=use_d96_panel_phase,
                fp16_pilot=False,
                fp32_balanced64_pilot=False,
                score_is_trivial=True,
                has_captures=False,
                mask_is_causal=True,
                mask_is_scalar_true=True,
                block_mask_is_default=True,
                grad_logsumexp_is_none=True,
            )
        return current, dkdv, dq

    @common_utils.parametrize(
        "panel_request",
        ("missing", "disabled", "backward", "numeric", "raw", "forward"),
    )
    @common_utils.parametrize(
        "early_request",
        ("missing", "disabled", "backward", "numeric", "raw", "forward"),
    )
    def test_d192_gqa4_phase_options(self, panel_request, early_request):
        from torch._inductor.kernel.flex.flex_attention import (
            _get_rocm_backward_config_options,
            _normalize_rocm_backward_options,
            _use_rocm_d192_panel_phase,
        )
        from torch._inductor.virtualized import V

        requests = {
            "missing": {},
            "disabled": {"bwd_": False},
            "backward": {"bwd_": True},
            "numeric": {"bwd_": 1},
            "raw": {"": True},
            "forward": {"fwd_": True},
        }
        options = {
            prefix + "ROCM_D192_PANEL_DOTS": value
            for prefix, value in requests[panel_request].items()
        }
        options.update(
            {
                prefix + "ROCM_D192_DQ_EARLY_DP": value
                for prefix, value in requests[early_request].items()
            }
        )
        options.update(
            ROCM_BWD_PHASE=13,
            bwd_ROCM_BWD_PHASE=17,
            ROCM_GQA_SPLITS=7,
            bwd_ROCM_GQA_SPLITS=4,
            SAFE_HEAD_DIM=False,
            QK_HEAD_DIM=192,
            V_HEAD_DIM=192,
            QK_HEAD_DIM_ROUNDED=256,
            V_HEAD_DIM_ROUNDED=256,
            BLOCK_M1=16,
            BLOCK_N1=64,
            BLOCK_M2=128,
            BLOCK_N2=32,
            num_warps=4,
            num_stages=2,
            matrix_instr_nonkdim=0,
            waves_per_eu=0,
            kpack=1,
            GQA_SHARED_HEADS=4,
            SM_SCALE=192**-0.5,
        )
        operands = []
        for name, shape in zip(
            ("query", "key", "value", "do"),
            (
                (2, 8, 2048, 192),
                (2, 2, 2048, 192),
                (2, 2, 2048, 192),
                (2, 8, 2048, 192),
            ),
            strict=True,
        ):
            strides, stride = [], 1
            for extent in reversed(shape):
                strides.insert(0, stride)
                stride *= extent
            operands.append(
                _tdm_operand(name, strides, size=shape, dtype=torch.bfloat16)
            )
        probe = dict(options)
        _normalize_rocm_backward_options(probe)
        probe = _get_rocm_backward_config_options(probe, is_gfx950=True)
        with V.set_graph_handler(_tdm_graph()):
            parent = _use_rocm_d192_panel_phase(
                True,
                *operands[:3],
                probe,
                True,
                False,
                requested=options.get("bwd_ROCM_D192_PANEL_DOTS", False),
                mask_is_scalar_true=True,
                block_mask_is_default=True,
                grad_logsumexp_is_none=True,
                grad_out=operands[3],
            )
        self.assertIs(parent, panel_request == "backward")
        current, dkdv, dq = self._rocm_backward_phase_options(
            options, operands=operands, use_d192_panel_phase=parent
        )
        self.assertIs(dkdv["ROCM_D192_PANEL_DOTS"], parent)
        self.assertIs(dkdv["ROCM_D192_DQ_EARLY_DP"], False)
        self.assertIs(dq["ROCM_D192_PANEL_DOTS"], False)
        self.assertIs(
            dq["ROCM_D192_DQ_EARLY_DP"], parent and early_request == "backward"
        )
        self.assertEqual((dkdv["ROCM_BWD_PHASE"], dq["ROCM_BWD_PHASE"]), (1, 2))
        self.assertEqual((dkdv["ROCM_GQA_SPLITS"], dq["ROCM_GQA_SPLITS"]), (1, 1))
        self.assertIs(current["ROCM_D192_DQ_EARLY_DP"], False)
        self.assertFalse(any(key.startswith(("bwd_", "fwd_")) for key in dq))

    @common_utils.parametrize("parent_admitted", (False, True))
    @common_utils.parametrize(
        "case",
        (
            "enabled",
            "missing",
            "disabled",
            "numeric",
            "string",
            "phase1",
            "phase2",
            "splits",
            "early_dp",
        ),
    )
    def test_d192_dq_early_dp_guard(self, parent_admitted, case):
        from torch._inductor.kernel.flex.flex_attention import (
            _use_rocm_d192_dq_early_dp,
        )

        options = dict(
            ROCM_BWD_PHASE=1 if case == "phase1" else 2 if case == "phase2" else 0,
            ROCM_GQA_SPLITS=2 if case == "splits" else 1,
            ROCM_EARLY_DP=case == "early_dp",
        )
        requested = (
            1
            if case == "numeric"
            else "True"
            if case == "string"
            else case != "disabled"
        )
        keywords = {} if case == "missing" else dict(requested=requested)
        self.assertEqual(
            _use_rocm_d192_dq_early_dp(parent_admitted, options, **keywords),
            parent_admitted and case == "enabled",
        )

    @common_utils.parametrize("parent_admitted", (False, True))
    @common_utils.parametrize(
        "request", ("missing", "disabled", "backward", "numeric", "raw", "forward")
    )
    def test_d192_dq_early_dp_phase_options(self, parent_admitted, request):
        flag = "ROCM_D192_DQ_EARLY_DP"
        requests = {
            "missing": {},
            "disabled": {"bwd_" + flag: False},
            "backward": {"bwd_" + flag: True},
            "numeric": {"bwd_" + flag: 1},
            "raw": {flag: True},
            "forward": {"fwd_" + flag: True},
        }
        options = dict(
            requests[request],
            BLOCKS_ARE_CONTIGUOUS=True,
            ROCM_BWD_PHASE=13,
            bwd_ROCM_BWD_PHASE=17,
            ROCM_GQA_SPLITS=7,
            bwd_ROCM_GQA_SPLITS=4,
        )
        operands = None
        current, dkdv, dq = self._rocm_backward_phase_options(
            options, operands=operands, use_d192_panel_phase=parent_admitted
        )
        self.assertIs(dq[flag], parent_admitted and request == "backward")
        self.assertIs(dkdv[flag], False)
        self.assertIs(current[flag], False)
        self.assertEqual((dkdv["ROCM_BWD_PHASE"], dq["ROCM_BWD_PHASE"]), (1, 2))
        self.assertEqual((dkdv["ROCM_GQA_SPLITS"], dq["ROCM_GQA_SPLITS"]), (1, 1))
        self.assertFalse(any(key.startswith(("bwd_", "fwd_")) for key in dq))

    @common_utils.parametrize("parent_admitted", (False, True))
    @common_utils.parametrize(
        "case",
        (
            "enabled",
            "missing",
            "disabled",
            "numeric",
            "string",
            "phase1",
            "phase2",
            "splits",
            "early_dp",
            "contiguous_missing",
            "contiguous_false",
            "contiguous_numeric",
            "contiguous_string",
            "batch",
            "heads",
            "key_heads",
            "sequence2048",
            "sequence1024",
            "mixed_lengths",
            "gqa4",
            "unknown_batch",
            "unknown_heads",
            "unknown_length",
            "missing_query",
            "missing_key",
            "query_rank",
            "key_rank",
        ),
    )
    def test_d96_dq_early_dp_guard(self, parent_admitted, case):
        from torch._inductor.kernel.flex.flex_attention import _use_rocm_d96_dq_early_dp
        from torch._inductor.virtualized import V

        q_size, k_size = [4, 21, 4096, 96], [4, 21, 4096, 96]
        if case == "batch":
            q_size[0] = k_size[0] = 1
        elif case == "heads":
            q_size[1] = 20
        elif case == "key_heads":
            k_size[1] = 20
        elif case == "sequence2048":
            q_size[2] = k_size[2] = 2048
        elif case == "sequence1024":
            q_size[2] = k_size[2] = 1024
        elif case == "mixed_lengths":
            q_size[2], k_size[2] = 2048, 4096
        elif case == "gqa4":
            q_size, k_size = [2, 8, 4096, 96], [2, 2, 4096, 96]
        elif case == "unknown_batch":
            q_size[0] = sympy.Symbol("d96_early_batch", integer=True, positive=True)
        elif case == "unknown_heads":
            q_size[1] = sympy.Symbol("d96_early_heads", integer=True, positive=True)
        elif case == "unknown_length":
            q_size[2] = sympy.Symbol("d96_early_length", integer=True, positive=True)
        elif case == "query_rank":
            q_size = q_size[1:]
        elif case == "key_rank":
            k_size = k_size[1:]
        query = (
            None
            if case == "missing_query"
            else _tdm_operand(
                "query", [1] * len(q_size), size=q_size, dtype=torch.bfloat16
            )
        )
        key = (
            None
            if case == "missing_key"
            else _tdm_operand(
                "key", [1] * len(k_size), size=k_size, dtype=torch.bfloat16
            )
        )
        options = dict(
            ROCM_BWD_PHASE=1 if case == "phase1" else 2 if case == "phase2" else 0,
            ROCM_GQA_SPLITS=2 if case == "splits" else 1,
            ROCM_EARLY_DP=case == "early_dp",
        )
        if case != "contiguous_missing":
            options["BLOCKS_ARE_CONTIGUOUS"] = (
                False
                if case == "contiguous_false"
                else 1
                if case == "contiguous_numeric"
                else "True"
                if case == "contiguous_string"
                else True
            )
        requested = (
            1
            if case == "numeric"
            else "True"
            if case == "string"
            else case != "disabled"
        )
        keywords = {} if case == "missing" else dict(requested=requested)
        with V.set_graph_handler(_tdm_graph()):
            self.assertEqual(
                _use_rocm_d96_dq_early_dp(
                    parent_admitted, options, query=query, key=key, **keywords
                ),
                parent_admitted and case in ("enabled", "sequence2048"),
            )

    @common_utils.parametrize("parent_admitted", (False, True))
    @common_utils.parametrize(
        "request", ("missing", "disabled", "backward", "numeric", "raw", "forward")
    )
    def test_d96_dq_early_dp_phase_options(self, parent_admitted, request):
        flag = "ROCM_D96_DQ_EARLY_DP"
        requests = {
            "missing": {},
            "disabled": {"bwd_" + flag: False},
            "backward": {"bwd_" + flag: True},
            "numeric": {"bwd_" + flag: 1},
            "raw": {flag: True},
            "forward": {"fwd_" + flag: True},
        }
        options = dict(
            requests[request],
            BLOCKS_ARE_CONTIGUOUS=True,
            ROCM_BWD_PHASE=13,
            bwd_ROCM_BWD_PHASE=17,
            ROCM_GQA_SPLITS=7,
            bwd_ROCM_GQA_SPLITS=4,
        )
        query = _tdm_operand(
            "query", [1] * 4, size=[4, 21, 4096, 96], dtype=torch.bfloat16
        )
        key = _tdm_operand("key", [1] * 4, size=[4, 21, 4096, 96], dtype=torch.bfloat16)
        operands = [query, key, key, query]
        current, dkdv, dq = self._rocm_backward_phase_options(
            options, operands=operands, use_d96_panel_phase=parent_admitted
        )
        self.assertIs(dq[flag], parent_admitted and request == "backward")
        self.assertIs(dkdv[flag], False)
        self.assertIs(current[flag], False)
        self.assertEqual((dkdv["ROCM_BWD_PHASE"], dq["ROCM_BWD_PHASE"]), (1, 2))
        self.assertEqual((dkdv["ROCM_GQA_SPLITS"], dq["ROCM_GQA_SPLITS"]), (1, 1))
        self.assertFalse(any(key.startswith(("bwd_", "fwd_")) for key in dq))
        self.assertIs(dq["ROCM_D192_DQ_EARLY_DP"], False)

    @common_utils.parametrize(
        "case",
        (
            "enabled",
            "enabled_b2_n4096",
            "zero_scale_b2_n4096",
            "negative_scale_b2_n4096",
            "query_batch_b2_n4096",
            "key_batch_b2_n4096",
            "value_batch_b2_n4096",
            "do_batch_b2_n4096",
            "query_heads_b2_n4096",
            "key_heads_b2_n4096",
            "value_heads_b2_n4096",
            "do_heads_b2_n4096",
            "query_length_b2_n4096",
            "key_length_b2_n4096",
            "value_length_b2_n4096",
            "do_length_b2_n4096",
            "gqa_ratio_b2_n4096",
            "unsupported_b2_n1024",
            "unsupported_b2_n2048",
            "enabled_n1024",
            "zero_scale_n1024",
            "negative_scale_n1024",
            "query_length_n1024",
            "key_length_n1024",
            "value_length_n1024",
            "do_length_n1024",
            "gqa_ratio_n1024",
            "enabled_n2048",
            "zero_scale_n2048",
            "negative_scale_n2048",
            "query_length",
            "key_length",
            "value_length",
            "do_length",
            "gqa_ratio",
            "missing",
            "disabled",
            "numeric",
            "string",
            "separate",
            "architecture",
            "xcd_false",
            "xcd_missing",
            "xcd_numeric",
            "xcd_string",
            "early_dp",
            "contiguous",
            "captures",
            "score",
            "mask",
            "grad_lse",
            "batch",
            "heads",
            "length",
            "fp16",
            "do_dtype",
            "query_stride",
            "key_stride",
            "value_stride",
            "do_stride",
            "phase1",
            "phase2",
            "splits",
            "tile",
            "warps",
            "stages",
            "matrix",
            "waves",
            "kpack",
            "llvm_attrs",
            "nonfinite_scale",
            "zero_scale",
            "negative_scale",
        ),
    )
    def test_d128_xcd_reverse_dq_rows_guard(self, case):
        from torch._inductor.kernel.flex.flex_attention import (
            _use_rocm_d128_reverse_causal_dq_rows,
        )
        from torch._inductor.virtualized import V

        dtype = torch.float16 if case == "fp16" else torch.bfloat16
        batch, heads, length = (
            (1 if case == "batch" else 2 if "_b2_" in case else 4),
            (32 if case == "heads" else 64),
            (
                512
                if case == "length"
                else 1024
                if case.endswith("_n1024")
                else 2048
                if case.endswith("_n2048")
                else 4096
            ),
        )
        shapes = (
            (batch, heads, length, 128),
            (batch, 8, length, 128),
            (batch, 8, length, 128),
            (batch, heads, length, 128),
        )
        operands = []
        for name, shape in zip(("query", "key", "value", "do"), shapes, strict=True):
            if case == name + "_batch_b2_n4096":
                shape = (4, *shape[1:])
            if case == name + "_heads_b2_n4096":
                shape = (shape[0], 32 if name in ("query", "do") else 4, *shape[2:])
            if case in (
                name + "_length",
                name + "_length_n1024",
                name + "_length_b2_n4096",
            ):
                shape = (*shape[:2], 2048, shape[3])
            strides, stride = [], 1
            for extent in reversed(shape):
                strides.insert(0, stride)
                stride *= extent
            if case == name + "_stride":
                strides[-1] = 2
            operands.append(
                _tdm_operand(
                    name,
                    strides,
                    size=shape,
                    dtype=torch.float32
                    if name == "do" and case == "do_dtype"
                    else dtype,
                )
            )
        options = dict(
            ROCM_SEPARATE_DQ=True,
            ROCM_EARLY_DP=case != "early_dp",
            XCD_REMAP=False
            if case == "xcd_false"
            else 1
            if case == "xcd_numeric"
            else "True"
            if case == "xcd_string"
            else True,
            BLOCKS_ARE_CONTIGUOUS=case != "contiguous",
            SAFE_HEAD_DIM=True,
            IS_DIVISIBLE=True,
            HAS_FULL_BLOCKS=True,
            SM_SCALE=0.0
            if case
            in (
                "zero_scale",
                "zero_scale_n2048",
                "zero_scale_n1024",
                "zero_scale_b2_n4096",
            )
            else -0.125
            if case
            in (
                "negative_scale",
                "negative_scale_n2048",
                "negative_scale_n1024",
                "negative_scale_b2_n4096",
            )
            else float("inf")
            if case == "nonfinite_scale"
            else 128**-0.5,
            ROCM_BWD_PHASE=1 if case == "phase1" else 2 if case == "phase2" else 0,
            ROCM_GQA_SPLITS=2 if case == "splits" else 1,
            GQA_SHARED_HEADS=4
            if case in ("gqa_ratio", "gqa_ratio_n1024", "gqa_ratio_b2_n4096")
            else 8,
            QK_HEAD_DIM=128,
            V_HEAD_DIM=128,
            QK_HEAD_DIM_ROUNDED=128,
            V_HEAD_DIM_ROUNDED=128,
            BLOCK_M1=32,
            BLOCK_N1=128,
            BLOCK_M2=64 if case == "tile" else 128,
            BLOCK_N2=32,
            num_warps=8 if case == "warps" else 4,
            num_stages=1 if case == "stages" else 2,
            matrix_instr_nonkdim=16 if case == "matrix" else 0,
            waves_per_eu=1 if case == "waves" else 0,
            kpack=3 if case == "kpack" else 1,
            SPARSE_Q_BLOCK_SIZE=128,
            SPARSE_KV_BLOCK_SIZE=128,
            llvm_fn_attrs="amdgpu-sched-strategy=iterative-ilp"
            if case == "llvm_attrs"
            else "",
        )
        if case == "xcd_missing":
            options.pop("XCD_REMAP")
        requested = (
            1
            if case == "numeric"
            else "True"
            if case == "string"
            else case != "disabled"
        )
        keywords = {} if case == "missing" else dict(requested=requested)
        with V.set_graph_handler(_tdm_graph()):
            actual = _use_rocm_d128_reverse_causal_dq_rows(
                case != "separate",
                case != "architecture",
                *operands,
                options,
                case != "score",
                case == "captures",
                mask_is_causal=case != "mask",
                grad_logsumexp_is_none=case != "grad_lse",
                **keywords,
            )
        self.assertEqual(type(actual), bool)
        self.assertEqual(
            actual,
            case
            in (
                "enabled",
                "contiguous",
                "zero_scale",
                "negative_scale",
                "enabled_n2048",
                "zero_scale_n2048",
                "negative_scale_n2048",
                "enabled_n1024",
                "zero_scale_n1024",
                "negative_scale_n1024",
                "enabled_b2_n4096",
                "zero_scale_b2_n4096",
                "negative_scale_b2_n4096",
            ),
        )

    @common_utils.parametrize("separate", (False, True))
    @common_utils.parametrize("xcd", (False, True, 1))
    @common_utils.parametrize(
        "request",
        (
            "missing",
            "disabled",
            "backward",
            "numeric",
            "raw",
            "forward",
            "raw_backward",
            "forward_backward",
        ),
    )
    def test_d128_xcd_reverse_dq_rows_phase_options(self, separate, xcd, request):
        from torch._inductor.kernel.flex import flex_attention as flex_kernel_mod

        flag = "ROCM_D128_REVERSE_DQ_ROWS"
        requests = {
            "missing": {},
            "disabled": {"bwd_" + flag: False},
            "backward": {"bwd_" + flag: True},
            "numeric": {"bwd_" + flag: 1},
            "raw": {flag: True},
            "forward": {"fwd_" + flag: True},
            "raw_backward": {flag: True, "bwd_" + flag: True},
            "forward_backward": {"fwd_" + flag: True, "bwd_" + flag: True},
        }
        options = dict(
            requests[request],
            ROCM_BWD_PHASE=13,
            bwd_ROCM_BWD_PHASE=17,
            ROCM_GQA_SPLITS=7,
            bwd_ROCM_GQA_SPLITS=4,
            XCD_REMAP=xcd,
        )
        calls = []

        def guard(
            parent,
            architecture,
            query,
            key,
            value,
            grad_out,
            options,
            score,
            captures,
            **keywords,
        ):
            calls.append((parent, architecture, keywords))
            return (
                parent
                and architecture
                and keywords["requested"] is True
                and not keywords["other_requests"]
                and options["XCD_REMAP"] is True
            )

        with mock.patch.object(
            flex_kernel_mod, "_use_rocm_d128_reverse_causal_dq_rows", guard
        ):
            current, dkdv, dq = self._rocm_backward_phase_options(
                options, use_separate_dq=separate
            )
        self.assertEqual(len(calls), 1)
        self.assertIs(dkdv[flag], False)
        self.assertIs(dq[flag], separate and xcd is True and request == "backward")
        self.assertIs(current[flag], False)
        self.assertEqual((dkdv["ROCM_BWD_PHASE"], dq["ROCM_BWD_PHASE"]), (1, 2))
        self.assertEqual((dkdv["ROCM_GQA_SPLITS"], dq["ROCM_GQA_SPLITS"]), (1, 1))
        self.assertFalse(any(key.startswith(("fwd_", "bwd_")) for key in dq))

    @common_utils.parametrize("parent_admitted", (False, True))
    @common_utils.parametrize(
        "case",
        (
            "enabled",
            "missing",
            "disabled",
            "numeric",
            "string",
            "phase1",
            "phase2",
            "splits",
            "early_dp",
            "contiguous_missing",
            "contiguous_false",
            "contiguous_numeric",
            "contiguous_string",
            "batch",
            "heads",
            "key_heads",
            "sequence2048",
            "sequence1024",
            "mixed_lengths",
            "gqa4",
            "unknown_batch",
            "unknown_heads",
            "unknown_length",
            "missing_query",
            "missing_key",
            "query_rank",
            "key_rank",
            "missing_early",
            "early_numeric",
            "raw_conflict",
        ),
    )
    def test_d96_dq_unroll2_guard(self, parent_admitted, case):
        from torch._inductor.kernel.flex.flex_attention import _use_rocm_d96_dq_unroll2
        from torch._inductor.virtualized import V

        q_size, k_size = ([4, 21, 4096, 96], [4, 21, 4096, 96])
        if case == "batch":
            q_size[0] = k_size[0] = 1
        elif case == "heads":
            q_size[1] = 20
        elif case == "key_heads":
            k_size[1] = 20
        elif case == "sequence2048":
            q_size[2] = k_size[2] = 2048
        elif case == "sequence1024":
            q_size[2] = k_size[2] = 1024
        elif case == "mixed_lengths":
            q_size[2], k_size[2] = 2048, 4096
        elif case == "gqa4":
            q_size, k_size = ([2, 8, 4096, 96], [2, 2, 4096, 96])
        elif case == "unknown_batch":
            q_size[0] = sympy.Symbol("d96_early_batch", integer=True, positive=True)
        elif case == "unknown_heads":
            q_size[1] = sympy.Symbol("d96_early_heads", integer=True, positive=True)
        elif case == "unknown_length":
            q_size[2] = sympy.Symbol("d96_early_length", integer=True, positive=True)
        elif case == "query_rank":
            q_size = q_size[1:]
        elif case == "key_rank":
            k_size = k_size[1:]
        query = (
            None
            if case == "missing_query"
            else _tdm_operand(
                "query", [1] * len(q_size), size=q_size, dtype=torch.bfloat16
            )
        )
        key = (
            None
            if case == "missing_key"
            else _tdm_operand(
                "key", [1] * len(k_size), size=k_size, dtype=torch.bfloat16
            )
        )
        options = dict(
            ROCM_BWD_PHASE=1 if case == "phase1" else 2 if case == "phase2" else 0,
            ROCM_GQA_SPLITS=2 if case == "splits" else 1,
            ROCM_EARLY_DP=case == "early_dp",
        )
        if case != "contiguous_missing":
            options["BLOCKS_ARE_CONTIGUOUS"] = (
                False
                if case == "contiguous_false"
                else 1
                if case == "contiguous_numeric"
                else "True"
                if case == "contiguous_string"
                else True
            )
        requested = (
            1
            if case == "numeric"
            else "True"
            if case == "string"
            else case != "disabled"
        )
        keywords = {} if case == "missing" else dict(requested=requested)
        with V.set_graph_handler(_tdm_graph()):
            self.assertEqual(
                _use_rocm_d96_dq_unroll2(
                    parent_admitted,
                    options,
                    query=query,
                    key=key,
                    **keywords,
                    early_dq_dp_requested=1
                    if case == "early_numeric"
                    else case != "missing_early",
                    other_requests=case == "raw_conflict",
                ),
                parent_admitted and case in ("enabled", "sequence2048"),
            )

    @common_utils.parametrize(
        "request",
        (
            "missing",
            "backward",
            "false",
            "numeric",
            "string",
            "raw",
            "forward",
            "raw_conflict",
            "forward_conflict",
        ),
    )
    @common_utils.parametrize("panel", (False, True))
    @common_utils.parametrize("early", (False, True))
    @common_utils.parametrize("config_count", (1, 2))
    def test_d96_dq_unroll2_phase_options(self, request, panel, early, config_count):
        flag = "ROCM_D96_DQ_UNROLL2"
        requests = {
            "missing": {},
            "backward": {"bwd_" + flag: True},
            "false": {"bwd_" + flag: False},
            "numeric": {"bwd_" + flag: 1},
            "string": {"bwd_" + flag: "True"},
            "raw": {flag: True},
            "forward": {"fwd_" + flag: True},
            "raw_conflict": {"bwd_" + flag: True, flag: True},
            "forward_conflict": {"bwd_" + flag: True, "fwd_" + flag: True},
        }
        query = _tdm_operand(
            "query", [1] * 4, size=[4, 21, 4096, 96], dtype=torch.bfloat16
        )
        key = _tdm_operand("key", [1] * 4, size=[4, 21, 4096, 96], dtype=torch.bfloat16)
        options = dict(
            requests[request],
            bwd_ROCM_D96_DQ_EARLY_DP=early,
            BLOCKS_ARE_CONTIGUOUS=True,
        )
        current, dkdv, dq = self._rocm_backward_phase_options(
            options,
            operands=[query, key, key, query],
            use_d96_panel_phase=panel,
            single_config=config_count == 1,
        )
        self.assertIs(options[flag], False)
        self.assertIs(current[flag], False)
        self.assertIs(dkdv[flag], False)
        self.assertIs(
            dq[flag], request == "backward" and panel and early and config_count == 1
        )

    @common_utils.parametrize(
        "case",
        (
            "enabled",
            "enabled_b2_n4096",
            "zero_scale_b2_n4096",
            "negative_scale_b2_n4096",
            "query_batch_b2_n4096",
            "key_batch_b2_n4096",
            "value_batch_b2_n4096",
            "do_batch_b2_n4096",
            "query_heads_b2_n4096",
            "key_heads_b2_n4096",
            "value_heads_b2_n4096",
            "do_heads_b2_n4096",
            "query_length_b2_n4096",
            "key_length_b2_n4096",
            "value_length_b2_n4096",
            "do_length_b2_n4096",
            "gqa_ratio_b2_n4096",
            "unsupported_b2_n1024",
            "unsupported_b2_n2048",
            "enabled_n1024",
            "zero_scale_n1024",
            "negative_scale_n1024",
            "query_length_n1024",
            "key_length_n1024",
            "value_length_n1024",
            "do_length_n1024",
            "gqa_ratio_n1024",
            "enabled_n2048",
            "zero_scale_n2048",
            "negative_scale_n2048",
            "query_length",
            "key_length",
            "value_length",
            "do_length",
            "gqa_ratio",
            "missing",
            "disabled",
            "numeric",
            "string",
            "separate",
            "architecture",
            "xcd_false",
            "xcd_missing",
            "xcd_numeric",
            "xcd_string",
            "early_dp",
            "contiguous",
            "captures",
            "score",
            "mask",
            "grad_lse",
            "batch",
            "heads",
            "length",
            "fp16",
            "do_dtype",
            "query_stride",
            "key_stride",
            "value_stride",
            "do_stride",
            "phase1",
            "phase2",
            "splits",
            "tile",
            "warps",
            "stages",
            "matrix",
            "waves",
            "kpack",
            "llvm_attrs",
            "nonfinite_scale",
            "zero_scale",
            "negative_scale",
            "reverse_disabled",
            "reverse_numeric",
            "raw_conflict",
            "forward_conflict",
        ),
    )
    def test_d128_dv_view_carry_split64_guard(self, case):
        from torch._inductor.kernel.flex.flex_attention import (
            _use_rocm_d128_dv_view_carry_split64,
        )
        from torch._inductor.virtualized import V

        dtype = torch.float16 if case == "fp16" else torch.bfloat16
        batch, heads, length = (
            (1 if case == "batch" else 2 if "_b2_" in case else 4),
            (32 if case == "heads" else 64),
            (
                512
                if case == "length"
                else 1024
                if case.endswith("_n1024")
                else 2048
                if case.endswith("_n2048")
                else 4096
            ),
        )
        shapes = (
            (batch, heads, length, 128),
            (batch, 8, length, 128),
            (batch, 8, length, 128),
            (batch, heads, length, 128),
        )
        operands = []
        for name, shape in zip(("query", "key", "value", "do"), shapes, strict=True):
            if case == name + "_batch_b2_n4096":
                shape = (4, *shape[1:])
            if case == name + "_heads_b2_n4096":
                shape = (shape[0], 32 if name in ("query", "do") else 4, *shape[2:])
            if case in (
                name + "_length",
                name + "_length_n1024",
                name + "_length_b2_n4096",
            ):
                shape = (*shape[:2], 2048, shape[3])
            strides, stride = [], 1
            for extent in reversed(shape):
                strides.insert(0, stride)
                stride *= extent
            if case == name + "_stride":
                strides[-1] = 2
            operands.append(
                _tdm_operand(
                    name,
                    strides,
                    size=shape,
                    dtype=torch.float32
                    if name == "do" and case == "do_dtype"
                    else dtype,
                )
            )
        options = dict(
            ROCM_SEPARATE_DQ=True,
            ROCM_EARLY_DP=case != "early_dp",
            XCD_REMAP=False
            if case == "xcd_false"
            else 1
            if case == "xcd_numeric"
            else "True"
            if case == "xcd_string"
            else True,
            BLOCKS_ARE_CONTIGUOUS=case != "contiguous",
            SAFE_HEAD_DIM=True,
            IS_DIVISIBLE=True,
            HAS_FULL_BLOCKS=True,
            SM_SCALE=0.0
            if case
            in (
                "zero_scale",
                "zero_scale_n2048",
                "zero_scale_n1024",
                "zero_scale_b2_n4096",
            )
            else -0.125
            if case
            in (
                "negative_scale",
                "negative_scale_n2048",
                "negative_scale_n1024",
                "negative_scale_b2_n4096",
            )
            else float("inf")
            if case == "nonfinite_scale"
            else 128**-0.5,
            ROCM_BWD_PHASE=1 if case == "phase1" else 2 if case == "phase2" else 0,
            ROCM_GQA_SPLITS=2 if case == "splits" else 1,
            GQA_SHARED_HEADS=4
            if case in ("gqa_ratio", "gqa_ratio_n1024", "gqa_ratio_b2_n4096")
            else 8,
            QK_HEAD_DIM=128,
            V_HEAD_DIM=128,
            QK_HEAD_DIM_ROUNDED=128,
            V_HEAD_DIM_ROUNDED=128,
            BLOCK_M1=32,
            BLOCK_N1=128,
            BLOCK_M2=64 if case == "tile" else 128,
            BLOCK_N2=32,
            num_warps=8 if case == "warps" else 4,
            num_stages=1 if case == "stages" else 2,
            matrix_instr_nonkdim=16 if case == "matrix" else 0,
            waves_per_eu=1 if case == "waves" else 0,
            kpack=3 if case == "kpack" else 1,
            SPARSE_Q_BLOCK_SIZE=128,
            SPARSE_KV_BLOCK_SIZE=128,
            llvm_fn_attrs="amdgpu-sched-strategy=iterative-ilp"
            if case == "llvm_attrs"
            else "",
        )
        if case == "xcd_missing":
            options.pop("XCD_REMAP")
        requested = (
            1
            if case == "numeric"
            else "True"
            if case == "string"
            else case != "disabled"
        )
        keywords = {} if case == "missing" else dict(requested=requested)
        with V.set_graph_handler(_tdm_graph()):
            actual = _use_rocm_d128_dv_view_carry_split64(
                case != "separate",
                case != "architecture",
                *operands,
                options,
                case != "score",
                case == "captures",
                mask_is_causal=case != "mask",
                grad_logsumexp_is_none=case != "grad_lse",
                reverse_requested=False
                if case == "reverse_disabled"
                else 1
                if case == "reverse_numeric"
                else True,
                other_requests=case in ("raw_conflict", "forward_conflict"),
                **keywords,
            )
        self.assertEqual(type(actual), bool)
        self.assertEqual(
            actual,
            case
            in (
                "enabled",
                "contiguous",
                "zero_scale",
                "negative_scale",
                "enabled_n2048",
                "zero_scale_n2048",
                "negative_scale_n2048",
                "enabled_n1024",
                "zero_scale_n1024",
                "negative_scale_n1024",
                "enabled_b2_n4096",
                "zero_scale_b2_n4096",
                "negative_scale_b2_n4096",
            ),
        )

    @common_utils.parametrize(
        "request",
        (
            "missing",
            "disabled",
            "backward",
            "numeric",
            "raw",
            "forward",
            "raw_backward",
            "forward_backward",
            "unroll2",
        ),
    )
    @common_utils.parametrize("reverse", (False, True))
    @common_utils.parametrize("config_count", (1, 2))
    def test_d128_dv_view_carry_split64_phase_options(
        self, request, reverse, config_count
    ):
        from torch._inductor.kernel.flex import flex_attention as flex_kernel_mod

        flag = "ROCM_D128_DV_VIEW_CARRY_SPLIT64"
        options = {flag: False, "bwd_ROCM_D128_REVERSE_DQ_ROWS": reverse}
        if request in ("backward", "raw_backward", "forward_backward", "unroll2"):
            options["bwd_" + flag] = True
        elif request == "disabled":
            options["bwd_" + flag] = False
        elif request == "numeric":
            options["bwd_" + flag] = 1
        if request in ("raw", "raw_backward"):
            options[flag] = True
        if request in ("forward", "forward_backward"):
            options["fwd_" + flag] = True
        options["bwd_ROCM_D96_DQ_UNROLL2"] = request == "unroll2"
        calls = []

        def guard(
            *args,
            requested=False,
            reverse_requested=False,
            other_requests=False,
            **kwargs,
        ):
            calls.append((requested, reverse_requested, other_requests))
            return (
                requested is True and reverse_requested is True and not other_requests
            )

        with mock.patch.object(
            flex_kernel_mod, "_use_rocm_d128_dv_view_carry_split64", guard
        ):
            current, dkdv, dq = self._rocm_backward_phase_options(
                options, single_config=config_count == 1
            )
        self.assertIs(options[flag], False)
        self.assertNotIn("bwd_" + flag, options)
        self.assertIs(current[flag], False)
        self.assertIs(dq[flag], False)
        self.assertIs(
            dkdv[flag], request == "backward" and reverse and config_count == 1
        )
        self.assertEqual(len(calls), int(config_count == 1))

    def _d128_query_tile_outer_fp32_inputs(self, batch=4, length=4096):
        nodes, options = self._d128_query_tile_outer_inputs()
        values = self._d128_coverage_operands(batch, length, True)
        nodes[:8] = [
            values[0],
            values[1],
            values[2],
            values[5],
            values[5],
            values[3],
            values[0],
            values[2],
        ]
        if length != 4096:
            blocks = length // 128
            for index in range(8, 16):
                shape = (1, 1, blocks) if index % 2 == 0 else (1, 1, blocks, blocks)
                strides, stride = [], 1
                for size in reversed(shape):
                    strides.insert(0, stride)
                    stride *= size
                nodes[index] = _tdm_operand(
                    "fp32_query_outer_metadata_" + str(index),
                    strides,
                    size=shape,
                    dtype=torch.int32,
                )
        options.update(
            BLOCK_M1=16,
            BLOCK_N1=64,
            BLOCK_M2=64,
            BLOCK_N2=16,
            num_stages=1,
            matrix_instr_nonkdim=0,
            FLOAT32_PRECISION="'ieee'",
            ROCM_SEPARATE_DQ=True,
            ROCM_EARLY_DP=True,
            XCD_REMAP=True,
            BLOCKS_ARE_CONTIGUOUS=True,
            IS_DIVISIBLE=True,
            SAFE_HEAD_DIM=True,
            USE_TMA=False,
            PRESCALE_QK=False,
        )
        return nodes, options

    @common_utils.parametrize(
        "batch,case",
        (
            (4, "enabled"),
            (4, "zero"),
            (4, "negative"),
            (4, "common_device2"),
            (4, "option_off"),
            (4, "multi_config"),
            (4, "phase2"),
            (4, "matrix16"),
            (4, "stages2"),
            (4, "warps8"),
            (4, "waves1"),
            (4, "kpack2"),
            (4, "tile"),
            (4, "non_ieee"),
            (4, "prescale"),
            (4, "tma"),
            (4, "contiguous_false"),
            (4, "contiguous_numeric"),
            (4, "contiguous_string"),
            (4, "contiguous_missing"),
            (4, "other_scale"),
            (4, "bool_scale"),
            (4, "nan_scale"),
            (4, "pow_scale"),
            (4, "missing_option"),
            (4, "query_alias"),
            (4, "batch2"),
            (4, "length2048"),
            (4, "input_dtype"),
            (4, "lse_dtype"),
            (4, "dv_dtype"),
            (4, "metadata_dtype"),
            (4, "stride"),
            (4, "mixed_device"),
            (1, "enabled"),
            (1, "zero"),
            (1, "negative"),
            (1, "common_device2"),
            (1, "option_off"),
            (1, "matrix16"),
            (1, "stages2"),
            (1, "kpack2"),
            (1, "rank0"),
            (1, "batch0"),
            (1, "batch3"),
            (1, "mixed_key"),
            (1, "mixed_do"),
            (1, "mixed_lse"),
            (1, "mixed_dv"),
            (2, "enabled"),
            (2, "zero"),
            (2, "negative"),
            (2, "common_device2"),
            (2, "option_off"),
            (2, "matrix16"),
            (2, "stages2"),
            (2, "kpack2"),
            (2, "warps8"),
            (2, "waves1"),
            (2, "other_scale"),
            (2, "pow_scale"),
            (2, "rank0"),
            (2, "batch0"),
            (2, "batch3"),
            (2, "mixed_key"),
            (2, "mixed_do"),
            (2, "mixed_lse"),
            (2, "mixed_dv"),
        ),
    )
    def test_d128_query_tile_outer_fp32_guard(self, batch, case):
        from torch._inductor.ir import Buffer, FixedLayout
        from torch._inductor.kernel.flex.flex_attention import (
            _use_rocm_d128_gqa_query_tile_outer,
        )
        from torch._inductor.virtualized import V

        nodes, options = self._d128_query_tile_outer_fp32_inputs(batch=batch)
        changes = {
            "zero": ("SM_SCALE", 0.0),
            "negative": ("SM_SCALE", -0.125),
            "option_off": ("ROCM_D128_DV_VIEW_CARRY_SPLIT64", False),
            "phase2": ("ROCM_BWD_PHASE", 2),
            "matrix16": ("matrix_instr_nonkdim", 16),
            "stages2": ("num_stages", 2),
            "warps8": ("num_warps", 8),
            "waves1": ("waves_per_eu", 1),
            "kpack2": ("kpack", 2),
            "tile": ("BLOCK_N1", 32),
            "non_ieee": ("FLOAT32_PRECISION", "'tf32'"),
            "prescale": ("PRESCALE_QK", True),
            "tma": ("USE_TMA", True),
            "contiguous_false": ("BLOCKS_ARE_CONTIGUOUS", False),
            "contiguous_numeric": ("BLOCKS_ARE_CONTIGUOUS", 1),
            "contiguous_string": ("BLOCKS_ARE_CONTIGUOUS", "True"),
            "other_scale": ("SM_SCALE", 0.125),
            "bool_scale": ("SM_SCALE", False),
            "nan_scale": ("SM_SCALE", float("nan")),
            "pow_scale": ("SM_SCALE", 128**-0.5),
        }
        if case in changes:
            key, value = changes[case]
            options[key] = value
        if case == "contiguous_missing":
            options.pop("BLOCKS_ARE_CONTIGUOUS")
        if case == "missing_option":
            options.pop("num_stages")
        if case in ("batch0", "batch3"):
            nodes, options = self._d128_query_tile_outer_fp32_inputs(
                batch=0 if case == "batch0" else 3
            )
        if case == "query_alias":
            nodes[6] = _tdm_operand(
                "other_query",
                nodes[0].get_stride(),
                size=nodes[0].get_size(),
                dtype=torch.float32,
            )
        if case in ("batch2", "length2048"):
            values = self._d128_coverage_operands(
                (2 if batch == 4 else 4) if case == "batch2" else batch,
                2048 if case == "length2048" else 4096,
                True,
            )
            nodes[0] = nodes[6] = values[0]
        if case == "rank0":
            nodes[0] = nodes[6] = _tdm_operand(
                "scalar_query", (), size=(), dtype=torch.float32
            )
        if case in ("mixed_key", "mixed_do", "mixed_lse", "mixed_dv"):
            index = {"mixed_key": 1, "mixed_do": 5, "mixed_lse": 3, "mixed_dv": 7}[case]
            other, _ = self._d128_query_tile_outer_fp32_inputs(
                batch=2 if batch == 4 else 4
            )
            nodes[index] = other[index]
        if case in ("input_dtype", "lse_dtype", "dv_dtype", "metadata_dtype", "stride"):
            index = {
                "input_dtype": 5,
                "lse_dtype": 3,
                "dv_dtype": 7,
                "metadata_dtype": 10,
                "stride": 4,
            }[case]
            operand = nodes[index]
            strides = list(operand.get_stride())
            if case == "stride":
                strides[-1] = 2
            nodes[index] = _tdm_operand(
                "invalid_fp32_query_outer",
                strides,
                size=operand.get_size(),
                dtype=operand.get_dtype()
                if case == "stride"
                else torch.int64
                if case == "metadata_dtype"
                else torch.float16,
            )
        if case == "common_device2":
            nodes = [
                Buffer(
                    name="same_device_" + str(index),
                    layout=FixedLayout(
                        torch.device("cuda:2"),
                        node.get_dtype(),
                        size=node.get_size(),
                        stride=node.get_stride(),
                    ),
                )
                for index, node in enumerate(nodes)
            ]
            nodes[6] = nodes[0]
        if case == "mixed_device":
            operand = nodes[10]
            nodes[10] = Buffer(
                name="other_device",
                layout=FixedLayout(
                    torch.device("cuda:2"),
                    operand.get_dtype(),
                    size=operand.get_size(),
                    stride=operand.get_stride(),
                ),
            )
        before = dict(options)
        with V.set_graph_handler(_tdm_graph()):
            actual = _use_rocm_d128_gqa_query_tile_outer(
                nodes, options, single_config=case != "multi_config"
            )
        self.assertIs(
            actual,
            case
            in ("enabled", "zero", "negative", "common_device2", "contiguous_false"),
        )
        self.assertEqual(options.keys(), before.keys())
        for name in before:
            self.assertIs(options[name], before[name])

    @common_utils.parametrize(
        "batch,index",
        (
            (1, 8),
            (1, 10),
            (1, 14),
            (1, 15),
            (2, 8),
            (2, 9),
            (2, 10),
            (2, 11),
            (2, 12),
            (2, 13),
            (2, 14),
            (2, 15),
            (4, 8),
            (4, 9),
            (4, 10),
            (4, 11),
            (4, 12),
            (4, 13),
            (4, 14),
            (4, 15),
        ),
    )
    @common_utils.parametrize("field", ("batch", "head", "stride"))
    def test_d128_query_tile_outer_fp32_metadata(self, batch, index, field):
        from torch._inductor.kernel.flex.flex_attention import (
            _use_rocm_d128_gqa_query_tile_outer,
        )
        from torch._inductor.virtualized import V

        nodes, options = self._d128_query_tile_outer_fp32_inputs(batch=batch)
        shape = list(nodes[index].get_size())
        if field in ("batch", "head"):
            shape[0 if field == "batch" else 1] = (
                max(2, batch) if field == "batch" else 64
            )
        strides, stride = [], 1
        for size in reversed(shape):
            strides.insert(0, stride)
            stride *= size
        if field == "stride":
            strides[-1] = 2
        nodes[index] = _tdm_operand(
            "fp32_metadata", strides, size=shape, dtype=torch.int32
        )
        with V.set_graph_handler(_tdm_graph()):
            self.assertIs(
                _use_rocm_d128_gqa_query_tile_outer(nodes, options, single_config=True),
                False,
            )

    @common_utils.parametrize("batch", (1, 2, 4))
    @common_utils.parametrize(
        "case", ("enabled", "matrix16", "stages2", "expanded", "option_off")
    )
    def test_d128_query_tile_outer_fp32_selection(self, batch, case):
        from torch._inductor.kernel.flex.flex_attention import (
            _select_rocm_d128_query_tile_outer_options,
        )
        from torch._inductor.virtualized import V

        nodes, options = self._d128_query_tile_outer_fp32_inputs(batch=batch)
        if case in ("matrix16", "stages2", "option_off"):
            key, value = {
                "matrix16": ("matrix_instr_nonkdim", 16),
                "stages2": ("num_stages", 2),
                "option_off": ("ROCM_D128_DV_VIEW_CARRY_SPLIT64", False),
            }[case]
            options[key] = value
        if case == "expanded":
            nodes[10] = _tdm_operand(
                "expanded_fp32_metadata",
                (2048, 32, 1) if batch == 1 else (32, 32, 1),
                size=(1, 64, 32) if batch == 1 else (batch, 1, 32),
                dtype=torch.int32,
            )
        before = dict(options)
        with V.set_graph_handler(_tdm_graph()):
            _select_rocm_d128_query_tile_outer_options(
                nodes,
                options,
                single_config=True,
                fp16_pilot=False,
                fp32_balanced64_pilot=True,
            )
        self.assertIs(
            options.get("ROCM_D128_GQA_QUERY_TILE_OUTER", False), case == "enabled"
        )
        if case == "enabled":
            before["ROCM_D128_GQA_QUERY_TILE_OUTER"] = True
        self.assertEqual(options, before)

    @common_utils.parametrize("batch", (1, 2, 4))
    @common_utils.parametrize("length", (1024, 2048, 4096))
    @common_utils.parametrize("scale", (0.08838834764831843, 0.0, -0.125))
    @common_utils.parametrize("contiguous", (False, True))
    def test_d128_query_tile_outer_fp32_shape_scope(
        self, batch, length, scale, contiguous
    ):
        from torch._inductor.kernel.flex.flex_attention import (
            _use_rocm_d128_gqa_query_tile_outer,
        )
        from torch._inductor.virtualized import V

        nodes, options = self._d128_query_tile_outer_fp32_inputs(batch, length)
        options.update(SM_SCALE=scale, BLOCKS_ARE_CONTIGUOUS=contiguous)
        with V.set_graph_handler(_tdm_graph()):
            actual = _use_rocm_d128_gqa_query_tile_outer(
                nodes, options, single_config=True
            )
        self.assertIs(actual, length == 4096 or (batch, length) == (4, 2048))

    @common_utils.parametrize("index", tuple(range(16)))
    @common_utils.parametrize("field", ("size", "stride", "dtype", "missing"))
    def test_d128_query_tile_outer_fp32_n2048_operand_guard(self, index, field):
        from torch._inductor.kernel.flex.flex_attention import (
            _use_rocm_d128_gqa_query_tile_outer,
        )
        from torch._inductor.virtualized import V

        nodes, options = self._d128_query_tile_outer_fp32_inputs(length=2048)
        operand = nodes[index]
        shape, strides = list(operand.get_size()), list(operand.get_stride())
        if field == "size":
            shape[-1] += 1
        if field == "stride":
            strides[-1] = 2
        nodes[index] = (
            None
            if field == "missing"
            else _tdm_operand(
                "invalid_fp32_n2048",
                strides,
                size=shape,
                dtype=torch.float16 if field == "dtype" else operand.get_dtype(),
            )
        )
        with V.set_graph_handler(_tdm_graph()):
            self.assertIs(
                _use_rocm_d128_gqa_query_tile_outer(nodes, options, single_config=True),
                False,
            )

    def _d128_query_tile_outer_inputs(self):
        values = self._d128_coverage_operands(4, 4096, False)
        nodes = [
            values[0],
            values[1],
            values[2],
            values[5],
            values[5],
            values[3],
            values[0],
            values[2],
        ]
        for index in range(8, 16):
            shape = (1, 1, 32) if index % 2 == 0 else (1, 1, 32, 32)
            strides, stride = [], 1
            for size in reversed(shape):
                strides.insert(0, stride)
                stride *= size
            nodes.append(
                _tdm_operand(
                    "query_outer_" + str(index), strides, size=shape, dtype=torch.int32
                )
            )
        options = dict(
            ROCM_D128_DV_VIEW_CARRY_SPLIT64=True,
            HAS_FULL_BLOCKS=True,
            BLOCKS_ARE_CONTIGUOUS=True,
            SM_SCALE=0.08838834764831843,
            ROCM_BWD_PHASE=1,
            ROCM_GQA_SPLITS=1,
            GQA_SHARED_HEADS=8,
            QK_HEAD_DIM=128,
            V_HEAD_DIM=128,
            QK_HEAD_DIM_ROUNDED=128,
            V_HEAD_DIM_ROUNDED=128,
            BLOCK_M1=32,
            BLOCK_N1=128,
            BLOCK_M2=128,
            BLOCK_N2=32,
            num_warps=4,
            num_stages=2,
            matrix_instr_nonkdim=0,
            ROCM_SEPARATE_DQ=True,
            ROCM_EARLY_DP=True,
            XCD_REMAP=True,
            IS_DIVISIBLE=True,
            SAFE_HEAD_DIM=True,
            USE_TMA=False,
            PRESCALE_QK=False,
            FLOAT32_PRECISION="'ieee'",
            waves_per_eu=0,
            kpack=1,
            SPARSE_Q_BLOCK_SIZE=128,
            SPARSE_KV_BLOCK_SIZE=128,
        )
        return nodes, options

    @common_utils.parametrize(
        "case",
        (
            "enabled",
            "zero",
            "negative",
            "option_off",
            "numeric_option",
            "multi_config",
            "phase2",
            "kpack2",
            "matrix16",
            "stages1",
            "warps8",
            "waves1",
            "tile",
            "no_full",
            "contiguous_false",
            "contiguous_numeric",
            "contiguous_string",
            "contiguous_missing",
            "other_scale",
            "bool_scale",
            "nan_scale",
            "pow_scale",
            "missing_nodes",
            "missing_config",
            "batch2",
            "length2048",
            "dtype",
            "stride",
        ),
    )
    def test_d128_query_tile_outer_guard(self, case):
        from torch._inductor.kernel.flex.flex_attention import (
            _use_rocm_d128_gqa_query_tile_outer,
        )
        from torch._inductor.virtualized import V

        nodes, options = self._d128_query_tile_outer_inputs()
        changes = {
            "zero": ("SM_SCALE", 0.0),
            "negative": ("SM_SCALE", -0.125),
            "option_off": ("ROCM_D128_DV_VIEW_CARRY_SPLIT64", False),
            "numeric_option": ("ROCM_D128_DV_VIEW_CARRY_SPLIT64", 1),
            "phase2": ("ROCM_BWD_PHASE", 2),
            "kpack2": ("kpack", 2),
            "matrix16": ("matrix_instr_nonkdim", 16),
            "stages1": ("num_stages", 1),
            "warps8": ("num_warps", 8),
            "waves1": ("waves_per_eu", 1),
            "tile": ("BLOCK_N1", 64),
            "no_full": ("HAS_FULL_BLOCKS", False),
            "contiguous_false": ("BLOCKS_ARE_CONTIGUOUS", False),
            "contiguous_numeric": ("BLOCKS_ARE_CONTIGUOUS", 1),
            "contiguous_string": ("BLOCKS_ARE_CONTIGUOUS", "True"),
            "other_scale": ("SM_SCALE", 0.125),
            "bool_scale": ("SM_SCALE", False),
            "nan_scale": ("SM_SCALE", float("nan")),
            "pow_scale": ("SM_SCALE", 128**-0.5),
        }
        if case in changes:
            key, value = changes[case]
            options[key] = value
        if case == "contiguous_missing":
            options.pop("BLOCKS_ARE_CONTIGUOUS")
        if case == "missing_nodes":
            nodes.pop()
        if case == "missing_config":
            options.pop("num_stages")
        if case in ("batch2", "length2048", "dtype", "stride"):
            values = self._d128_coverage_operands(
                2 if case == "batch2" else 4,
                2048 if case == "length2048" else 4096,
                case == "dtype",
                "stride" if case == "stride" else None,
                0,
            )
            nodes[0] = values[0]
        with V.set_graph_handler(_tdm_graph()):
            actual = _use_rocm_d128_gqa_query_tile_outer(
                nodes, options, single_config=case != "multi_config"
            )
        self.assertIs(
            actual, case in ("enabled", "zero", "negative", "contiguous_false")
        )

    @common_utils.parametrize("index", tuple(range(8, 16)))
    @common_utils.parametrize("field", ("batch", "head", "shape", "dtype", "unknown"))
    def test_d128_query_tile_outer_metadata(self, index, field):
        from torch._inductor.kernel.flex.flex_attention import (
            _use_rocm_d128_gqa_query_tile_outer,
        )
        from torch._inductor.virtualized import V

        nodes, options = self._d128_query_tile_outer_inputs()
        shape = list(nodes[index].get_size())
        if field == "batch":
            shape[0] = 4
        elif field == "head":
            shape[1] = 64
        elif field == "shape":
            shape.pop()
        elif field == "unknown":
            shape[0] = sympy.Symbol("query_outer_batch", positive=True, integer=True)
        strides, stride = [], 1
        for size in reversed(shape):
            strides.insert(0, stride)
            stride *= size
        nodes[index] = _tdm_operand(
            "invalid_query_outer",
            strides,
            size=shape,
            dtype=torch.int64 if field == "dtype" else torch.int32,
        )
        with V.set_graph_handler(_tdm_graph()):
            self.assertIs(
                _use_rocm_d128_gqa_query_tile_outer(nodes, options, single_config=True),
                False,
            )

    @common_utils.parametrize("prefix", ("", "fwd_", "bwd_"))
    def test_d128_query_tile_outer_private_aliases(self, prefix):
        from torch._inductor.kernel.flex.flex_attention import (
            _strip_rocm_query_tile_outer_options,
        )

        options = {
            prefix + "ROCM_D128_GQA_QUERY_TILE_OUTER": True,
            "bwd_ROCM_D128_DV_VIEW_CARRY_SPLIT64": True,
        }
        _strip_rocm_query_tile_outer_options(options)
        self.assertEqual(options, {"bwd_ROCM_D128_DV_VIEW_CARRY_SPLIT64": True})

    @common_utils.parametrize(
        "case",
        ("broadcast", "expanded", "unknown", "kpack2", "option_off", "other_scale"),
    )
    def test_d128_query_tile_outer_selection(self, case):
        from torch._inductor.kernel.flex.flex_attention import (
            _select_rocm_d128_query_tile_outer_options,
        )
        from torch._inductor.virtualized import V

        nodes, options = self._d128_query_tile_outer_inputs()
        options["matrix_instr_nonkdim"] = 0
        if case in ("expanded", "unknown"):
            shape = (
                4
                if case == "expanded"
                else sympy.Symbol("unknown_batch", positive=True, integer=True),
                1,
                32,
            )
            nodes[10] = _tdm_operand(
                "selection_metadata", (32, 32, 1), size=shape, dtype=torch.int32
            )
        if case == "kpack2":
            options["kpack"] = 2
        if case == "option_off":
            options["ROCM_D128_DV_VIEW_CARRY_SPLIT64"] = False
        if case == "other_scale":
            options["SM_SCALE"] = 128**-0.5
        with V.set_graph_handler(_tdm_graph()):
            _select_rocm_d128_query_tile_outer_options(
                nodes,
                options,
                single_config=True,
                fp16_pilot=True,
                fp32_balanced64_pilot=False,
            )
        self.assertEqual(
            options["matrix_instr_nonkdim"],
            0,
        )
        self.assertIs(
            options.get("ROCM_D128_GQA_QUERY_TILE_OUTER", False), case == "broadcast"
        )

    def _d128_query_tile_outer_fp16_B2_inputs(self):
        nodes, options = self._d128_query_tile_outer_inputs()
        values = self._d128_coverage_operands(2, 4096, False)
        nodes[:8] = [
            values[0],
            values[1],
            values[2],
            values[5],
            values[5],
            values[3],
            values[0],
            values[2],
        ]
        options.update(
            matrix_instr_nonkdim=0,
            ROCM_SEPARATE_DQ=True,
            ROCM_EARLY_DP=True,
            XCD_REMAP=True,
            BLOCKS_ARE_CONTIGUOUS=True,
            IS_DIVISIBLE=True,
            SAFE_HEAD_DIM=True,
            USE_TMA=False,
            PRESCALE_QK=False,
            FLOAT32_PRECISION="'ieee'",
        )
        return nodes, options

    @common_utils.parametrize(
        "case",
        (
            "enabled",
            "zero",
            "negative",
            "common_device2",
            "option_off",
            "matrix16",
            "stages1",
            "warps8",
            "waves1",
            "kpack2",
            "pow_scale",
            "bool_scale",
            "missing_config",
            "multi_config",
            "contiguous_false",
            "contiguous_numeric",
            "contiguous_string",
            "contiguous_missing",
            "rank0",
            "mixed_lse",
            "mixed_delta",
            "mixed_dv",
            "alias",
            "different_device",
        ),
    )
    def test_d128_query_tile_outer_fp16_B2_guard(self, case):
        from torch._inductor.ir import Buffer, FixedLayout
        from torch._inductor.kernel.flex.flex_attention import (
            _use_rocm_d128_gqa_query_tile_outer,
        )
        from torch._inductor.virtualized import V

        nodes, options = self._d128_query_tile_outer_fp16_B2_inputs()
        changes = {
            "zero": ("SM_SCALE", 0.0),
            "negative": ("SM_SCALE", -0.125),
            "option_off": ("ROCM_D128_DV_VIEW_CARRY_SPLIT64", False),
            "matrix16": ("matrix_instr_nonkdim", 16),
            "stages1": ("num_stages", 1),
            "warps8": ("num_warps", 8),
            "waves1": ("waves_per_eu", 1),
            "kpack2": ("kpack", 2),
            "pow_scale": ("SM_SCALE", 128**-0.5),
            "bool_scale": ("SM_SCALE", False),
            "contiguous_false": ("BLOCKS_ARE_CONTIGUOUS", False),
            "contiguous_numeric": ("BLOCKS_ARE_CONTIGUOUS", 1),
            "contiguous_string": ("BLOCKS_ARE_CONTIGUOUS", "True"),
        }
        if case in changes:
            name, value = changes[case]
            options[name] = value
        if case == "contiguous_missing":
            options.pop("BLOCKS_ARE_CONTIGUOUS")
        if case == "missing_config":
            options.pop("num_stages")
        if case == "rank0":
            nodes[0] = nodes[6] = _tdm_operand(
                "fp16_scalar_query", (), size=(), dtype=torch.float16
            )
        if case.startswith("mixed_"):
            index = {"mixed_lse": 3, "mixed_delta": 4, "mixed_dv": 7}[case]
            other, _ = self._d128_query_tile_outer_inputs()
            nodes[index] = other[index]
        if case == "alias":
            nodes[6] = Buffer(
                name="fp16_wrong_alias",
                layout=FixedLayout(
                    nodes[0].get_device(),
                    nodes[0].get_dtype(),
                    size=nodes[0].get_size(),
                    stride=nodes[0].get_stride(),
                ),
            )
        if case == "common_device2":
            nodes = [
                Buffer(
                    name="fp16_common_device_" + str(index),
                    layout=FixedLayout(
                        torch.device("cuda:2"),
                        node.get_dtype(),
                        size=node.get_size(),
                        stride=node.get_stride(),
                    ),
                )
                for index, node in enumerate(nodes)
            ]
            nodes[6] = nodes[0]
        if case == "different_device":
            nodes[7] = Buffer(
                name="fp16_other_device",
                layout=FixedLayout(
                    torch.device("cuda:2"),
                    nodes[7].get_dtype(),
                    size=nodes[7].get_size(),
                    stride=nodes[7].get_stride(),
                ),
            )
        before = dict(options)
        with V.set_graph_handler(_tdm_graph()):
            actual = _use_rocm_d128_gqa_query_tile_outer(
                nodes, options, single_config=case != "multi_config"
            )
        self.assertIs(
            actual,
            case
            in ("enabled", "zero", "negative", "common_device2", "contiguous_false"),
        )
        self.assertEqual(options.keys(), before.keys())
        for name in before:
            self.assertIs(options[name], before[name])

    @common_utils.parametrize("index", tuple(range(16)))
    @common_utils.parametrize("field", ("dtype", "stride"))
    def test_d128_query_tile_outer_fp16_B2_operands(self, index, field):
        from torch._inductor.kernel.flex.flex_attention import (
            _use_rocm_d128_gqa_query_tile_outer,
        )
        from torch._inductor.virtualized import V

        nodes, options = self._d128_query_tile_outer_fp16_B2_inputs()
        if field == "dtype":
            name = "get_dtype"
            value = (
                torch.float16
                if index in (3, 4)
                else torch.bfloat16
                if index < 8
                else torch.int64
            )
        else:
            name = "get_stride"
            value = list(nodes[index].get_stride())
            value[-1] = 2
        with (
            mock.patch.object(nodes[index], name, return_value=value),
            V.set_graph_handler(_tdm_graph()),
        ):
            self.assertIs(
                _use_rocm_d128_gqa_query_tile_outer(nodes, options, single_config=True),
                False,
            )

    @common_utils.parametrize("index", (3, 4, 7, 8, 11, 14, 15))
    @common_utils.parametrize("field", ("rank", "stride_count"))
    def test_d128_query_tile_outer_fp16_B2_rank(self, index, field):
        from torch._inductor.kernel.flex.flex_attention import (
            _use_rocm_d128_gqa_query_tile_outer,
        )
        from torch._inductor.virtualized import V

        nodes, options = self._d128_query_tile_outer_fp16_B2_inputs()
        name = "get_size" if field == "rank" else "get_stride"
        value = list(getattr(nodes[index], name)())[:-1]
        with (
            mock.patch.object(nodes[index], name, return_value=value),
            V.set_graph_handler(_tdm_graph()),
        ):
            self.assertIs(
                _use_rocm_d128_gqa_query_tile_outer(nodes, options, single_config=True),
                False,
            )

    @common_utils.parametrize(
        "case",
        (
            "dimension_bool",
            "dimension_float",
            "dimension_string",
            "dimension_none",
            "stride_bool",
            "stride_float",
            "stride_string",
            "stride_none",
            "config_bool",
            "config_float",
            "config_string",
            "config_none",
        ),
    )
    def test_d128_query_tile_outer_fp16_B2_types(self, case):
        from torch._inductor.kernel.flex.flex_attention import (
            _use_rocm_d128_gqa_query_tile_outer,
        )
        from torch._inductor.virtualized import V

        nodes, options = self._d128_query_tile_outer_fp16_B2_inputs()
        field, kind = case.split("_")
        if field == "config":
            options["matrix_instr_nonkdim"] = {
                "bool": False,
                "float": 0.0,
                "string": "0",
                "none": None,
            }[kind]
            with V.set_graph_handler(_tdm_graph()):
                self.assertIs(
                    _use_rocm_d128_gqa_query_tile_outer(
                        nodes, options, single_config=True
                    ),
                    False,
                )
        else:
            index, name = (3, "get_size") if field == "dimension" else (7, "get_stride")
            value = list(getattr(nodes[index], name)())
            value[-1] = {
                "bool": True,
                "float": float(value[-1]),
                "string": str(value[-1]),
                "none": None,
            }[kind]
            with (
                mock.patch.object(nodes[index], name, return_value=value),
                V.set_graph_handler(_tdm_graph()),
            ):
                self.assertIs(
                    _use_rocm_d128_gqa_query_tile_outer(
                        nodes, options, single_config=True
                    ),
                    False,
                )

    @common_utils.parametrize(
        "case",
        (
            "enabled",
            "zero",
            "negative",
            "expanded",
            "option_off",
            "matrix16",
            "stages1",
            "multi_config",
            "indexed",
            "pow_scale",
            "kpack2",
        ),
    )
    def test_d128_query_tile_outer_fp16_B2_selection(self, case):
        from torch._inductor.kernel.flex.flex_attention import (
            _select_rocm_d128_query_tile_outer_options,
        )
        from torch._inductor.virtualized import V

        nodes, options = self._d128_query_tile_outer_fp16_B2_inputs()
        changes = {
            "zero": ("SM_SCALE", 0.0),
            "negative": ("SM_SCALE", -0.125),
            "option_off": ("ROCM_D128_DV_VIEW_CARRY_SPLIT64", False),
            "matrix16": ("matrix_instr_nonkdim", 16),
            "stages1": ("num_stages", 1),
            "indexed": ("BLOCKS_ARE_CONTIGUOUS", False),
            "pow_scale": ("SM_SCALE", 128**-0.5),
            "kpack2": ("kpack", 2),
        }
        if case in changes:
            name, value = changes[case]
            options[name] = value
        if case == "expanded":
            nodes[10] = _tdm_operand(
                "fp16_expanded_metadata",
                (32, 32, 1),
                size=(2, 1, 32),
                dtype=torch.int32,
            )
        before = dict(options)
        with V.set_graph_handler(_tdm_graph()):
            _select_rocm_d128_query_tile_outer_options(
                nodes,
                options,
                single_config=case != "multi_config",
                fp16_pilot=True,
                fp32_balanced64_pilot=False,
            )
        enabled = case in ("enabled", "zero", "negative", "indexed")
        self.assertIs(options.get("ROCM_D128_GQA_QUERY_TILE_OUTER", False), enabled)
        if enabled:
            before["ROCM_D128_GQA_QUERY_TILE_OUTER"] = True
        self.assertEqual(options, before)

    @common_utils.parametrize("contiguous", (False, True))
    def test_d128_query_tile_outer_fp16_B2_automatic_scope(self, contiguous):
        from torch._inductor.kernel.flex.flex_attention import (
            _get_rocm_backward_config_options,
            _get_rocm_backward_phase_options,
            _normalize_rocm_backward_options,
        )
        from torch._inductor.virtualized import V

        nodes, out, options = self._backward_auto_profile_inputs(
            dtype=torch.float16,
            batch=2,
            length=4096,
        )
        options["BLOCKS_ARE_CONTIGUOUS"] = contiguous
        profile = self._backward_auto_profile(
            nodes,
            out,
            options,
            {"bwd_BLOCKS_ARE_CONTIGUOUS": contiguous},
        )
        self.assertIsNotNone(profile)
        requests = _normalize_rocm_backward_options(profile)
        current = _get_rocm_backward_config_options(profile, is_gfx950=True)
        with V.set_graph_handler(_tdm_graph()):
            dkdv, dq = _get_rocm_backward_phase_options(
                current,
                nodes,
                requests=requests,
                single_config=True,
                is_gfx950=True,
                query=nodes[0],
                key=nodes[1],
                value=nodes[2],
                grad_out=nodes[5],
                use_separate_dq=True,
                use_d192_panel_phase=False,
                use_d96_panel_phase=False,
                fp16_pilot=True,
                fp32_balanced64_pilot=False,
                score_is_trivial=True,
                has_captures=False,
                mask_is_causal=True,
                mask_is_scalar_true=False,
                block_mask_is_default=False,
                grad_logsumexp_is_none=True,
            )
        self.assertIs(dkdv.get("ROCM_D128_GQA_QUERY_TILE_OUTER", False), True)
        self.assertIs(dq.get("ROCM_D128_GQA_QUERY_TILE_OUTER", False), False)
        self.assertEqual(dkdv["matrix_instr_nonkdim"], 0)
        self.assertEqual(dq["matrix_instr_nonkdim"], 0)
        self.assertEqual((dkdv["ROCM_BWD_PHASE"], dq["ROCM_BWD_PHASE"]), (1, 2))

    def _d64_four_owner_inputs(
        self,
        dtype=torch.float16,
        batch=1,
        heads=8,
        kv_heads=2,
        length=4096,
        dimension=64,
    ):
        nodes, out, options = self._backward_auto_profile_inputs(
            dtype, batch, length, dimension, heads, kv_heads
        )
        return (nodes, out, options)

    def test_backward_auto_profile_d64_default_profile_and_parent(self):
        from torch._inductor.kernel.flex import flex_attention as flex_kernel_mod
        from torch._inductor.virtualized import V

        nodes, out, options = self._d64_four_owner_inputs()
        original = dict(options)
        profile = self._backward_auto_profile(nodes, out, options)
        self.assertIsNotNone(profile)
        self.assertEqual(options, original)
        self.assertEqual(
            tuple(
                profile[name]
                for name in (
                    "BLOCK_M1",
                    "BLOCK_N1",
                    "BLOCK_M2",
                    "BLOCK_N2",
                    "num_warps",
                    "num_stages",
                    "matrix_instr_nonkdim",
                    "waves_per_eu",
                    "kpack",
                )
            ),
            (32, 128, 128, 32, 4, 2, 0, 0, 1),
        )
        self.assertIs(profile["ROCM_GQA_SPLIT_DKDV"], True)
        self.assertIs(profile["ROCM_SEPARATE_DQ"], False)
        self.assertIs(profile["ROCM_EARLY_DP"], False)
        self.assertIs(profile["XCD_REMAP"], False)
        current = dict(profile)
        requests = flex_kernel_mod._normalize_rocm_backward_options(current)
        current = flex_kernel_mod._get_rocm_backward_config_options(
            current, is_gfx950=True
        )
        with V.set_graph_handler(_tdm_graph()):
            self.assertTrue(
                flex_kernel_mod._use_rocm_gqa_split_dkdv(
                    True, *nodes[:3], current, True, True, True, False
                )
            )
            self.assertFalse(
                flex_kernel_mod._use_rocm_separate_dq(
                    True,
                    *nodes[:3],
                    current,
                    True,
                    False,
                    mask_is_scalar_true=True,
                    block_mask_is_default=True,
                )
            )
            phase1, phase2 = flex_kernel_mod._get_rocm_backward_phase_options(
                current,
                nodes,
                requests=requests,
                single_config=True,
                is_gfx950=True,
                query=nodes[0],
                key=nodes[1],
                value=nodes[2],
                grad_out=nodes[5],
                use_separate_dq=False,
                use_d192_panel_phase=False,
                use_d96_panel_phase=False,
                fp16_pilot=False,
                fp32_balanced64_pilot=False,
                score_is_trivial=True,
                has_captures=False,
                mask_is_causal=False,
                mask_is_scalar_true=True,
                block_mask_is_default=True,
                grad_logsumexp_is_none=True,
            )
        self.assertEqual((phase1["ROCM_BWD_PHASE"], phase2["ROCM_BWD_PHASE"]), (1, 2))
        for phase in (phase1, phase2):
            self.assertIs(phase.get("ROCM_D128_GQA_QUERY_TILE_OUTER", False), False)
            self.assertIs(phase["ROCM_EARLY_DP"], False)
            self.assertIs(phase["XCD_REMAP"], False)

    @common_utils.parametrize(
        "flag,value",
        (
            ("is_gfx950", False),
            ("single_config", False),
            ("score_is_trivial", False),
            ("has_captures", True),
            ("grad_logsumexp_is_none", False),
            ("mask_is_scalar_true", False),
            ("block_mask_is_default", False),
            ("mask_is_causal", True),
            ("backend", "FLASH"),
        ),
    )
    def test_backward_auto_profile_d64_semantic_fallback(self, flag, value):
        nodes, out, options = self._d64_four_owner_inputs()
        self.assertIsNone(
            self._backward_auto_profile(nodes, out, options, **{flag: value})
        )

    @common_utils.parametrize(
        "dtype,batch,heads,kv_heads,length,dimension",
        (
            (torch.bfloat16, 1, 8, 2, 4096, 64),
            (torch.float32, 1, 8, 2, 4096, 64),
            (torch.float16, 4, 32, 8, 4096, 64),
            (torch.float16, 1, 8, 2, 2048, 64),
            (torch.float16, 1, 8, 2, 8192, 64),
            (torch.float16, 2, 8, 2, 4096, 64),
            (torch.float16, 1, 16, 4, 4096, 64),
            (torch.float16, 1, 8, 1, 4096, 64),
            (torch.float16, 1, 8, 2, 4096, 128),
        ),
    )
    def test_backward_auto_profile_d64_shape_fallback(
        self, dtype, batch, heads, kv_heads, length, dimension
    ):
        nodes, out, options = self._d64_four_owner_inputs(
            dtype, batch, heads, kv_heads, length, dimension
        )
        self.assertIsNone(self._backward_auto_profile(nodes, out, options))

    @common_utils.parametrize(
        "scale", (0.0, -0.125, 0.25, True, float("nan"), float("inf"))
    )
    def test_backward_auto_profile_d64_scale_fallback(self, scale):
        nodes, out, options = self._d64_four_owner_inputs()
        options["SM_SCALE"] = scale
        self.assertIsNone(self._backward_auto_profile(nodes, out, options))

    @common_utils.parametrize("index", (0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 16))
    @common_utils.parametrize("field", ("dtype", "stride", "device", "shape", "rank"))
    def test_backward_auto_profile_d64_operand_fallback(self, index, field):
        nodes, out, options = self._d64_four_owner_inputs()
        tensor = out if index == 16 else nodes[index]
        shape, stride, dtype, device = (
            list(tensor.get_size()),
            list(tensor.get_stride()),
            tensor.get_dtype(),
            tensor.get_device(),
        )
        if field == "dtype":
            dtype = torch.float32 if dtype != torch.float32 else torch.float16
        elif field == "stride":
            stride[-1] = 2
        elif field == "device":
            device = torch.device("cpu")
        elif field == "shape":
            shape[0] += 1
        else:
            shape.append(1)
            stride.append(1)
        invalid = _tdm_operand("invalid_d64_owner", stride, size=shape, dtype=dtype)
        invalid.layout.device = device
        if index == 16:
            out = invalid
        else:
            nodes[index] = invalid
        self.assertIsNone(self._backward_auto_profile(nodes, out, options))

    @common_utils.parametrize("index", (12, 13, 14, 15))
    def test_backward_auto_profile_d64_full_metadata_fallback(self, index):
        nodes, out, options = self._d64_four_owner_inputs()
        nodes[index] = nodes[8 if index % 2 == 0 else 9]
        self.assertIsNone(self._backward_auto_profile(nodes, out, options))

    @common_utils.parametrize(
        "option,value",
        (
            ("BLOCKS_ARE_CONTIGUOUS", True),
            ("HAS_FULL_BLOCKS", True),
            ("IS_DIVISIBLE", False),
            ("SAFE_HEAD_DIM", False),
            ("GQA_SHARED_HEADS", 8),
            ("QK_HEAD_DIM", 128),
            ("V_HEAD_DIM", 128),
            ("QK_HEAD_DIM_ROUNDED", 128),
            ("V_HEAD_DIM_ROUNDED", 128),
            ("SPARSE_Q_BLOCK_SIZE", 128),
            ("SPARSE_KV_BLOCK_SIZE", 128),
            ("USE_TMA", True),
            ("PRESCALE_QK", True),
        ),
    )
    def test_backward_auto_profile_d64_metadata_and_option_fallback(
        self, option, value
    ):
        nodes, out, options = self._d64_four_owner_inputs()
        options[option] = value
        self.assertIsNone(self._backward_auto_profile(nodes, out, options))

    @common_utils.parametrize(
        "option",
        (
            "ROCM_GQA_SPLITS",
            "ROCM_BWD_PHASE",
            "ROCM_GQA_SPLIT_DKDV",
            "ROCM_GQA_FOUR_OWNER",
            "ROCM_SEPARATE_DQ",
            "ROCM_EARLY_DP",
            "XCD_REMAP",
            "BLOCK_M1",
            "BLOCK_N1",
            "BLOCK_M2",
            "BLOCK_N2",
            "num_warps",
            "num_stages",
            "matrix_instr_nonkdim",
            "waves_per_eu",
            "kpack",
            "BLOCKS_ARE_CONTIGUOUS",
        ),
    )
    @common_utils.parametrize("prefix", ("", "bwd_", "fwd_"))
    def test_backward_auto_profile_d64_explicit_options_reject_automatic_owner_selection(
        self, option, prefix
    ):
        nodes, out, options = self._d64_four_owner_inputs()
        selected = self._backward_auto_profile(
            nodes, out, options, {prefix + option: False}
        )
        if prefix == "fwd_" or (prefix == "" and option == "BLOCKS_ARE_CONTIGUOUS"):
            self.assertIsNotNone(selected)
        else:
            self.assertIsNone(selected)

    @common_utils.parametrize(
        "name,value",
        (
            ("fwd_num_warps", 4),
            ("fwd_num_warps", 8),
            ("fwd_num_stages", 1),
            ("fwd_BLOCK_M1", 64),
            ("fwd_BLOCKS_ARE_CONTIGUOUS", True),
        ),
    )
    def test_backward_auto_profile_d64_forward_options_preserve_defaults(
        self, name, value
    ):
        nodes, out, options = self._d64_four_owner_inputs()
        expected = self._backward_auto_profile(nodes, out, options)
        user = {name: value}
        before = dict(user)
        self.assertIsNotNone(expected)
        self.assertEqual(
            self._backward_auto_profile(nodes, out, options, user), expected
        )
        self.assertEqual(user, before)


class TestFlexAttentionTDMOptions(InductorTestCase):
    _PREREQS = "torch._inductor.utils._gfx1250_device_prereqs"

    def test_flex_tdm_gate_checks_layout_and_offset(self):
        from torch._inductor.utils import use_flex_tdm_descriptor
        from torch._inductor.virtualized import V

        good = _tdm_operand("good", [32768, 8192, 64, 1])
        blocks = [(128, 64)] * 3
        cases = {
            "good": (good, blocks, False),
            # Innermost stride must be one.
            "bad_head": (_tdm_operand("bad_head", [32768, 8192, 128, 2]), blocks, True),
            # 65 * 2 bytes is not 16-byte aligned.
            "bad_outer": (
                _tdm_operand("bad_outer", [32768, 8192, 65, 1]),
                blocks,
                True,
            ),
            "bad_offset": (
                _tdm_operand("bad_offset", [32768, 8192, 64, 1], offset=1),
                blocks,
                True,
            ),
            # 8 elements is a 16-byte offset, so this one is admissible.
            "semantic_offset": (
                _tdm_operand("semantic_offset", [32768, 8192, 64, 1], offset=8),
                blocks,
                False,
            ),
            # 96 * 2 bytes is not 128-byte aligned.
            "bad_block": (good, [(128, 64), (128, 64), (128, 96)], True),
            # head_dim 65 under a 128-wide block: the descriptor bounds-checks
            # and zero-fills the overhang, so a non-16-byte-multiple logical
            # extent is admissible on TDM even though can_use_tma refuses it.
            "padded_tail": (
                _tdm_operand(
                    "padded_tail", [65536, 16384, 128, 1], size=(2, 4, 128, 65)
                ),
                [(128, 128)] * 3,
                False,
            ),
        }

        graph = _tdm_graph()
        with (
            V.set_graph_handler(graph),
            mock.patch(self._PREREQS, return_value=True) as device_prereqs,
        ):
            for name, (operand, block_shapes, rejected) in cases.items():
                with self.subTest(case=name):
                    admitted = use_flex_tdm_descriptor(
                        good, good, operand, block_shapes=block_shapes
                    )
                    self.assertEqual(admitted, not rejected)

            # Capability is a hard boundary regardless of layout.
            device_prereqs.return_value = False
            self.assertFalse(
                use_flex_tdm_descriptor(good, good, good, block_shapes=blocks)
            )

    def test_flex_tdm_gate_honours_config_flag(self):
        """`enable_flex_tdm` must disable selection for operands that pass."""
        from torch._inductor import config
        from torch._inductor.utils import use_flex_tdm_descriptor
        from torch._inductor.virtualized import V

        good = _tdm_operand("good", [32768, 8192, 64, 1])
        blocks = [(128, 64)] * 3
        graph = _tdm_graph()
        with (
            V.set_graph_handler(graph),
            mock.patch(self._PREREQS, return_value=True),
        ):
            self.assertTrue(
                use_flex_tdm_descriptor(good, good, good, block_shapes=blocks)
            )
            with config.patch({"triton.enable_flex_tdm": False}):
                self.assertFalse(
                    use_flex_tdm_descriptor(good, good, good, block_shapes=blocks)
                )

    def test_flex_tdm_gate_requires_one_block_shape_per_operand(self):
        """A short list must raise, not silently leave operands unchecked.

        `zip` truncates, so a missing entry would skip an operand's rules.
        """
        from torch._inductor.utils import use_flex_tdm_descriptor
        from torch._inductor.virtualized import V

        good = _tdm_operand("good", [32768, 8192, 64, 1])
        bad = _tdm_operand("bad", [32768, 8192, 65, 1])
        graph = _tdm_graph()
        with (
            V.set_graph_handler(graph),
            mock.patch(self._PREREQS, return_value=True),
        ):
            with self.assertRaises(AssertionError):
                use_flex_tdm_descriptor(good, good, bad, block_shapes=[(128, 64)] * 2)

    def test_flex_tdm_gate_preserves_dynamic_sequence_lengths(self):
        from torch._dynamo.source import ConstantSource
        from torch._inductor.utils import use_flex_tdm_descriptor
        from torch._inductor.virtualized import V
        from torch.fx.experimental.symbolic_shapes import DimDynamic

        def make_qkv(name, seq_len):
            return _tdm_operand(
                name,
                stride=(256 * seq_len, 64 * seq_len, 64, 1),
                size=(2, 4, seq_len, 64),
            )

        graph = _tdm_graph()
        q_len = graph.sizevars.shape_env.create_symbol(
            128,
            source=ConstantSource("q_len"),
            dynamic_dim=DimDynamic.DYNAMIC,
        )
        kv_len = graph.sizevars.shape_env.create_symbol(
            256,
            source=ConstantSource("kv_len"),
            dynamic_dim=DimDynamic.DYNAMIC,
        )
        v_len = graph.sizevars.shape_env.create_symbol(
            384,
            source=ConstantSource("v_len"),
            dynamic_dim=DimDynamic.DYNAMIC,
        )
        query = make_qkv("query", q_len)
        key = make_qkv("key", kv_len)
        value = make_qkv("value", v_len)

        graph.unaligned_buffers.add("value")
        guards_before = len(graph.sizevars.shape_env.guards)
        with (
            V.set_graph_handler(graph),
            mock.patch(self._PREREQS, return_value=True),
        ):
            # Admission is list-wide, as it is for the MM templates: `value` is
            # rejected, so the earlier-checked `query` and `key` must not be
            # left carrying range guards for a descriptor that is never built.
            blocks = [(128, 64)] * 3
            self.assertFalse(
                use_flex_tdm_descriptor(query, key, value, block_shapes=blocks)
            )
            self.assertEqual(len(graph.sizevars.shape_env.guards), guards_before)
            graph.unaligned_buffers.remove("value")
            self.assertTrue(
                use_flex_tdm_descriptor(query, key, value, block_shapes=blocks)
            )

        # Only an admitted operand list installs bounds, and it installs exactly
        # one combined guard covering every operand -- that is what makes the
        # rejection above leave nothing behind.
        new_guards = graph.sizevars.shape_env.guards[guards_before:]
        int32_max = torch.iinfo(torch.int32).max
        self.assertEqual(len(new_guards), 1)
        self.assertEqual(
            set(new_guards[0][0].atoms(sympy.Le)),
            {
                sympy.Le(q_len, int32_max),
                sympy.Le(kv_len, int32_max),
                sympy.Le(v_len, int32_max),
            },
        )

    def test_flex_tdm_gate_rejects_unprovable_alignment_with_aligned_hint(self):
        """An aligned hint is not a proof, and must not be treated as one.

        The gate is guard-free by construction, so a dynamic stride whose hint
        happens to satisfy every alignment rule still has to be rejected rather
        than silently specialized on that hint.
        """
        from torch._dynamo.source import ConstantSource
        from torch._inductor.utils import use_flex_tdm_descriptor
        from torch._inductor.virtualized import V
        from torch.fx.experimental.symbolic_shapes import DimDynamic

        graph = _tdm_graph()
        # Hint 64 elements == 128 bytes, so the hint clears both the 16-byte and
        # the 128-byte rule while the symbol proves neither.
        row = graph.sizevars.shape_env.create_symbol(
            64,
            source=ConstantSource("row"),
            dynamic_dim=DimDynamic.DYNAMIC,
        )
        # The batch and head strides carry provable factors; only the row stride
        # is left unprovable, which is the realistic shape.
        mat = _tdm_operand("dynamic_row", stride=(512 * row, 128 * row, row, 1))

        guards_before = len(graph.sizevars.shape_env.guards)
        with (
            V.set_graph_handler(graph),
            mock.patch(self._PREREQS, return_value=True),
        ):
            self.assertFalse(
                use_flex_tdm_descriptor(mat, mat, mat, block_shapes=[(128, 64)] * 3)
            )
        self.assertEqual(len(graph.sizevars.shape_env.guards), guards_before)

    def test_flex_head_dim_32_is_legal_but_rejected_by_policy(self):
        """FP16 head_dim=32 is a 64-byte request: legal, and rejected anyway.

        The rejection reason is asserted, not just the boolean, so the 128-byte
        performance policy cannot be quietly restated as a legality rule.
        """
        from torch._inductor.utils import (
            _TDM_MIN_INNERMOST_REQUEST_BYTES,
            use_flex_tdm_descriptor,
        )
        from torch._inductor.virtualized import V

        # Row stride padded to 64 elements (128 bytes) so every outer stride
        # satisfies policy and the block width is the only thing left to fail.
        qkv = [
            _tdm_operand(n, [32768, 8192, 64, 1], size=(2, 4, 128, 32))
            for n in ("q", "k", "v")
        ]
        head_dim_bytes = 32 * torch.float16.itemsize
        self.assertEqual(head_dim_bytes, 64)
        # Legal: it clears the one alignment-shaped rule Triton enforces.
        self.assertGreaterEqual(head_dim_bytes, _TDM_MIN_INNERMOST_REQUEST_BYTES)

        graph = _tdm_graph()
        with (
            V.set_graph_handler(graph),
            mock.patch(self._PREREQS, return_value=True),
            self.assertLogs("torch._inductor.utils", level="DEBUG") as logs,
        ):
            self.assertFalse(
                use_flex_tdm_descriptor(*qkv, block_shapes=[(128, 32)] * 3)
            )

        reasons = "\n".join(logs.output)
        self.assertIn("block width is not 128-byte aligned", reasons)
        self.assertNotIn("innermost request extent", reasons)


# Forward builds descriptors for all three operands; decode builds them for K/V
# only, because its reshaped query uses ordinary tl.load.
_TDM_FORWARD_OPERANDS = ("q", "k", "v")
_TDM_DECODE_OPERANDS = ("k", "v")
_TDM_TOLERANCES = {
    torch.float16: Tolerances(atol=2e-2, rtol=2e-2),
    torch.bfloat16: Tolerances(atol=5e-2, rtol=5e-2),
    torch.float32: Tolerances(atol=2e-3, rtol=2e-3),
}


def _descriptor_load_pattern(operand):
    return rf"tl\.load_tensor_descriptor\(\s*desc_{operand}\s*,"


def _pointer_load_pattern(operand):
    # load_checked_2d is defined unconditionally in every generated kernel, so
    # only an operand-qualified call establishes the fallback.
    return rf"\bload_checked_2d\(\s*{operand.upper()}\s*,"


def _check_int64_decode_narrows_coordinates(
    test_case, query, key, value, *, kernel_options, tolerances
):
    """Assert decode keeps descriptor coordinates int32 under int64 indexing.

    Decode derives the K/V row offset from a program id cast to INDEX_DTYPE,
    and Triton rejects a 64-bit descriptor coordinate. The operands that would
    force INDEX_DTYPE=tl.int64 naturally are multi-gigabyte, so 32-bit indexing
    is denied directly instead; that reaches the same codegen on small inputs.

    Shared because the narrowing lives in common.py.jinja, which every
    descriptor backend renders: the gfx1250 caller reaches it by capability and
    the CUDA/XPU caller by an explicit USE_TMA request, but the generated
    coordinate expression is the same one.
    """
    from torch._inductor.codegen.triton import TritonScheduling

    def fn(q, k, v):
        return flex_attention(q, k, v, kernel_options=kernel_options)

    with mock.patch.object(
        TritonScheduling, "can_use_32bit_indexing", return_value=False
    ):
        result, code = run_and_get_code(torch.compile(fn), query, key, value)

    joined = "\n".join(code)
    test_case.assertIn("INDEX_DTYPE : tl.constexpr = tl.int64", joined)
    for name in _TDM_DECODE_OPERANDS:
        test_case.assertRegex(joined, _descriptor_load_pattern(name))
        # Per operand: one shared substring would accept a half-applied fix,
        # since K and V render the same expression.
        test_case.assertRegex(
            joined,
            _descriptor_load_pattern(name) + r"\s*\[\s*kv_base_offset\.to\(tl\.int32\)",
        )
    # Reference without kernel_options: the request selects an implementation,
    # it does not define the expected numbers.
    torch.testing.assert_close(
        result,
        flex_attention(query, key, value),
        atol=tolerances.atol,
        rtol=tolerances.rtol,
    )


@unittest.skipUnless(
    running_on_tdm_device(),
    "requires gfx1250 with ROCm 7.14+ and TDM-capable Triton",
)
class TestFlexAttentionTDMEndToEnd(InductorTestCase):
    # Neither length divides any default tile, so every descriptor block has a
    # masked tail and the bounds handling is always exercised.
    Q_LEN = 509
    KV_LEN = 513

    def _compile_and_get_code(self, fn, *args):
        # Flex TDM selection is capability- and layout-driven; no config opt-in.
        return run_and_get_code(torch.compile(fn), *args)

    def _assert_descriptor_path(self, code, operands):
        joined = "\n".join(code)
        for name in operands:
            self.assertRegex(joined, _descriptor_load_pattern(name))

    def _assert_pointer_path(self, code, operands):
        joined = "\n".join(code)
        for name in operands:
            self.assertRegex(joined, _pointer_load_pattern(name))
            self.assertNotRegex(joined, _descriptor_load_pattern(name))

    def _qkv(self, device, dtype, q_len=None, kv_len=None, head_dim=64, offset=0):
        # A nonzero offset is taken by slicing the sequence dim, which keeps the
        # strides and shifts the storage offset by offset * head_dim elements.
        def alloc(seq_len):
            base = torch.randn(
                2, 4, seq_len + offset, head_dim, device=device, dtype=dtype
            )
            return base[:, :, offset:, :] if offset else base

        q = alloc(self.Q_LEN if q_len is None else q_len)
        kv_len = self.KV_LEN if kv_len is None else kv_len
        return q, alloc(kv_len), alloc(kv_len)

    def _head_dim_tail(self, device, seq_len):
        # Slice a 128-wide buffer rather than allocating head_dim 65
        # contiguously: a contiguous tensor would have a 130-byte row stride,
        # which the descriptor gate rejects outright, so TDM would never be
        # selected and the test would pass without exercising the overhang.
        padded = torch.randn(2, 4, seq_len, 128, device=device, dtype=torch.float16)
        tail = padded[..., :65]
        self.assertEqual(tail.stride()[-2], 128)
        return tail

    @dtypes(*_TDM_TOLERANCES)
    def test_tdm_flex_forward_correctness_and_selection(self, device, dtype):
        def fn(q, k, v):
            return flex_attention(q, k, v)

        q, k, v = self._qkv(device, dtype)
        result, code = self._compile_and_get_code(fn, q, k, v)
        self._assert_descriptor_path(code, _TDM_FORWARD_OPERANDS)
        torch.testing.assert_close(
            result,
            fn(q, k, v),
            atol=_TDM_TOLERANCES[dtype].atol,
            rtol=_TDM_TOLERANCES[dtype].rtol,
        )

    @dtypes(*_TDM_TOLERANCES)
    def test_tdm_flex_decode_correctness_and_selection(self, device, dtype):
        def fn(q, k, v):
            return flex_attention(q, k, v)

        q, k, v = self._qkv(device, dtype, q_len=1)
        result, code = self._compile_and_get_code(fn, q, k, v)
        self._assert_descriptor_path(code, _TDM_DECODE_OPERANDS)
        torch.testing.assert_close(
            result,
            fn(q, k, v),
            atol=_TDM_TOLERANCES[dtype].atol,
            rtol=_TDM_TOLERANCES[dtype].rtol,
        )

    def test_tdm_flex_decode_narrows_descriptor_coordinates(self, device):
        """Descriptor coordinates stay int32 when indexing is int64.

        Selection is capability-driven here, so the request stays implicit; the
        empty option dict is what flex_attention() already builds when the
        argument is omitted. test_tma_with_customer_kernel_options_int64_decode
        runs the same checks on CUDA and XPU, where it must be explicit.
        """
        q, k, v = self._qkv(device, torch.float16, q_len=1)
        _check_int64_decode_narrows_coordinates(
            self,
            q,
            k,
            v,
            kernel_options={},
            tolerances=_TDM_TOLERANCES[torch.float16],
        )

    def test_tdm_flex_config_flag_disables_descriptors(self, device):
        """`enable_flex_tdm=False` forces the pointer path for eligible inputs."""
        from torch._inductor import config as inductor_config

        def fn(q, k, v):
            return flex_attention(q, k, v)

        q, k, v = self._qkv(device, torch.float16)
        with inductor_config.patch({"triton.enable_flex_tdm": False}):
            result, code = self._compile_and_get_code(fn, q, k, v)
        self._assert_pointer_path(code, _TDM_FORWARD_OPERANDS)
        torch.testing.assert_close(
            result,
            fn(q, k, v),
            atol=_TDM_TOLERANCES[torch.float16].atol,
            rtol=_TDM_TOLERANCES[torch.float16].rtol,
        )

    @common_utils.parametrize(
        "backend, operands",
        [("TRITON", _TDM_FORWARD_OPERANDS), ("TRITON_DECODE", _TDM_DECODE_OPERANDS)],
    )
    @common_utils.parametrize("rejected_by", ["layout", "capability", "config"])
    def test_tdm_flex_ignores_generic_tma_probe(
        self, device, backend, operands, rejected_by
    ):
        """A true can_use_tma must not route ROCm through NVIDIA's rules.

        has_triton_tma_device() excludes HIP only in its CUDA arm, so it can
        read true on a ROCm host. USE_TMA=True is required to reach the bypass:
        with the option omitted the old selector left it false and short-
        circuited before ever consulting the probe, so the pointer path held
        for an unrelated reason and this test would pass unfixed.
        """
        from torch._inductor import config as inductor_config

        options = {"USE_TMA": True, "BACKEND": backend}

        def fn(q, k, v):
            return flex_attention(q, k, v, kernel_options=options)

        # Each arm is rejected by a different AMD-side rule, all of which the
        # generic probe would skip.
        head_dim = 33 if rejected_by == "layout" else 64
        q, k, v = self._qkv(
            device,
            torch.float16,
            q_len=1 if backend == "TRITON_DECODE" else None,
            head_dim=head_dim,
        )

        with contextlib.ExitStack() as stack:
            stack.enter_context(
                mock.patch(
                    "torch._inductor.kernel.flex.flex_attention.can_use_tma",
                    return_value=True,
                )
            )
            stack.enter_context(
                mock.patch(
                    "torch._inductor.kernel.flex.flex_decoding.can_use_tma",
                    return_value=True,
                )
            )
            if rejected_by == "capability":
                stack.enter_context(
                    mock.patch(
                        "torch._inductor.utils._gfx1250_device_prereqs",
                        return_value=False,
                    )
                )
            if rejected_by == "config":
                stack.enter_context(
                    inductor_config.patch({"triton.enable_flex_tdm": False})
                )
            _, code = self._compile_and_get_code(fn, q, k, v)

        self._assert_pointer_path(code, operands)

    def test_tdm_flex_forward_nonzero_storage_offset(self, device):
        # 8 rows of head_dim 64 in fp16 is a 1024-byte offset, so the operand
        # stays admissible and the descriptor base is not the allocation base.
        def fn(q, k, v):
            return flex_attention(q, k, v)

        q, k, v = self._qkv(device, torch.float16, offset=8)
        self.assertNotEqual(q.storage_offset(), 0)
        result, code = self._compile_and_get_code(fn, q, k, v)
        self._assert_descriptor_path(code, _TDM_FORWARD_OPERANDS)
        torch.testing.assert_close(
            result,
            fn(q, k, v),
            atol=_TDM_TOLERANCES[torch.float16].atol,
            rtol=_TDM_TOLERANCES[torch.float16].rtol,
        )

    def test_tdm_flex_padded_head_dim_correctness_and_selection(self, device):
        # head_dim 65 rounds to a descriptor block width of 128, so the block
        # overhangs the logical head dimension. Check that the out-of-bounds
        # region does not corrupt the attention result.
        def fn(q, k, v):
            return flex_attention(q, k, v)

        q = self._head_dim_tail(device, self.Q_LEN)
        k = self._head_dim_tail(device, self.KV_LEN)
        v = self._head_dim_tail(device, self.KV_LEN)

        result, code = self._compile_and_get_code(fn, q, k, v)
        self._assert_descriptor_path(code, _TDM_FORWARD_OPERANDS)
        torch.testing.assert_close(
            result,
            fn(q, k, v),
            atol=_TDM_TOLERANCES[torch.float16].atol,
            rtol=_TDM_TOLERANCES[torch.float16].rtol,
        )

    def test_tdm_flex_decode_padded_head_dim_correctness_and_selection(self, device):
        # Decode composes flex_decode.py.jinja and builds its own descriptors,
        # so the forward padded-head test does not cover this codegen. Only
        # K/V are gated here; q_len=1 just selects the decode lowering.
        def fn(q, k, v):
            return flex_attention(q, k, v)

        q = self._head_dim_tail(device, 1)
        k = self._head_dim_tail(device, self.KV_LEN)
        v = self._head_dim_tail(device, self.KV_LEN)

        result, code = self._compile_and_get_code(fn, q, k, v)
        self._assert_descriptor_path(code, _TDM_DECODE_OPERANDS)
        torch.testing.assert_close(
            result,
            fn(q, k, v),
            atol=_TDM_TOLERANCES[torch.float16].atol,
            rtol=_TDM_TOLERANCES[torch.float16].rtol,
        )

    @common_utils.parametrize(
        "backend, operands",
        [("TRITON", _TDM_FORWARD_OPERANDS), ("TRITON_DECODE", _TDM_DECODE_OPERANDS)],
    )
    @common_utils.parametrize(
        "options, descriptors",
        [
            ({}, True),
            ({"USE_TMA": True}, True),
            ({"USE_TMA": False}, False),
            # USE_TMA is typed bool but not validated at runtime, and every
            # falsy value took the pointer path before this feature existed.
            # An identity test against False would let these through.
            ({"USE_TMA": 0}, False),
            ({"USE_TMA": None}, False),
            # fwd_ wins over the unprefixed key for the forward kernels.
            ({"USE_TMA": True, "fwd_USE_TMA": False}, False),
            ({"USE_TMA": True, "fwd_USE_TMA": 0}, False),
        ],
    )
    def test_tdm_flex_use_tma_option(
        self, device, backend, operands, options, descriptors
    ):
        def fn(q, k, v):
            return flex_attention(
                q, k, v, kernel_options=dict(options, BACKEND=backend)
            )

        q, k, v = self._qkv(
            device, torch.float16, q_len=1 if backend == "TRITON_DECODE" else None
        )
        result, code = self._compile_and_get_code(fn, q, k, v)
        if descriptors:
            self._assert_descriptor_path(code, operands)
        else:
            self._assert_pointer_path(code, operands)
        torch.testing.assert_close(
            result,
            fn(q, k, v),
            atol=_TDM_TOLERANCES[torch.float16].atol,
            rtol=_TDM_TOLERANCES[torch.float16].rtol,
        )

    @common_utils.parametrize(
        "backend, operands",
        [("TRITON", _TDM_FORWARD_OPERANDS), ("TRITON_DECODE", _TDM_DECODE_OPERANDS)],
    )
    def test_tdm_flex_rejected_layout_falls_back(self, device, backend, operands):
        # head_dim 33 in fp16 gives a 66-byte row stride, which fails the
        # operand alignment rule, so every operand must take the pointer path.
        def fn(q, k, v):
            return flex_attention(q, k, v, kernel_options={"BACKEND": backend})

        q, k, v = self._qkv(
            device,
            torch.float16,
            q_len=1 if backend == "TRITON_DECODE" else None,
            head_dim=33,
        )
        result, code = self._compile_and_get_code(fn, q, k, v)
        self._assert_pointer_path(code, operands)
        torch.testing.assert_close(
            result,
            fn(q, k, v),
            atol=_TDM_TOLERANCES[torch.float16].atol,
            rtol=_TDM_TOLERANCES[torch.float16].rtol,
        )

    def test_tdm_flex_dynamic_sequence_reuses_descriptor_graph(self, device):
        def fn(q, k, v):
            return flex_attention(q, k, v)

        cnt = CompileCounterWithBackend("inductor")
        compiled = torch.compile(fn, backend=cnt, dynamic=True)

        q, k, v = self._qkv(device, torch.float16)
        result, code = run_and_get_code(compiled, q, k, v)
        self.assertEqual(cnt.frame_count, 1)
        self._assert_descriptor_path(code, _TDM_FORWARD_OPERANDS)
        torch.testing.assert_close(
            result,
            fn(q, k, v),
            atol=_TDM_TOLERANCES[torch.float16].atol,
            rtol=_TDM_TOLERANCES[torch.float16].rtol,
        )

        # A second length must reuse the same graph. run_and_get_code is not
        # used here: with no recompilation there is no generated code to
        # capture, so the reuse itself is what carries the descriptor claim.
        q2, k2, v2 = self._qkv(device, torch.float16, q_len=637, kv_len=641)
        result2 = compiled(q2, k2, v2)
        self.assertEqual(cnt.frame_count, 1)
        torch.testing.assert_close(
            result2,
            fn(q2, k2, v2),
            atol=_TDM_TOLERANCES[torch.float16].atol,
            rtol=_TDM_TOLERANCES[torch.float16].rtol,
        )

    def test_tdm_flex_training_and_gqa_keep_backward_correct(self, device):
        # The skip-odd-keys score_mod plus GQA, run through the existing
        # backward: PR3 changes forward and decode only, so the gradients and
        # LSE have to stay correct while the forward takes the descriptor path.
        def score_mod(score, b, h, q, kv):
            return torch.where(kv % 2 == 0, score, float("-inf"))

        def fn(q, k, v):
            return flex_attention(
                q, k, v, score_mod=score_mod, enable_gqa=True, return_lse=True
            )

        def make(n_heads, seq_len, requires_grad):
            return torch.randn(
                2,
                n_heads,
                seq_len,
                64,
                device=device,
                dtype=torch.float16,
                requires_grad=requires_grad,
            )

        q, k, v = (
            make(8, self.Q_LEN, True),
            make(2, self.KV_LEN, True),
            make(2, self.KV_LEN, True),
        )
        ref_q, ref_k, ref_v = (
            t.detach().clone().requires_grad_(True) for t in (q, k, v)
        )

        (out, lse), code = self._compile_and_get_code(fn, q, k, v)
        self._assert_descriptor_path(code, _TDM_FORWARD_OPERANDS)

        ref_out, ref_lse = fn(ref_q, ref_k, ref_v)
        tol = _TDM_TOLERANCES[torch.float16]
        torch.testing.assert_close(out, ref_out, atol=tol.atol, rtol=tol.rtol)
        torch.testing.assert_close(lse, ref_lse, atol=tol.atol, rtol=tol.rtol)

        out.sum().backward()
        ref_out.sum().backward()
        for actual, expected in ((q, ref_q), (k, ref_k), (v, ref_v)):
            torch.testing.assert_close(actual.grad, expected.grad, atol=5e-2, rtol=5e-2)


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
        tensor_name: str | None = None,
        fudge_atol: float = 0,
    ):
        if (
            golden_out.device != ref_out.device
            or golden_out.device != compiled_out.device
        ):
            golden_out = golden_out.cpu()
            ref_out = ref_out.cpu()
            compiled_out = compiled_out.cpu()
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
        KV_B: int | None = None,
        KV_H: int | None = None,
        KV_S: int | None = None,
        V_D: int | None = None,
        block_mask: BlockMask | None = None,
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
        if _is_mps_device(device):
            sdpa_partial_gold = create_attention(
                _relocate_captures(score_mod, "cpu"),
                block_mask.to("cpu"),
                enable_gqa=(Q_H != KV_H),
            )
        else:
            sdpa_partial_gold = sdpa_partial

        compiled_sdpa = torch.compile(sdpa_partial)
        compiled_out = compiled_sdpa(q, k, v)
        golden_out = sdpa_partial_gold(q_gold, k_gold, v_gold)
        ref_out = sdpa_partial(q_ref, k_ref, v_ref)

        if not isinstance(golden_out, torch.Tensor):
            raise AssertionError(f"Expected torch.Tensor, got {type(golden_out)}")
        if not isinstance(ref_out, torch.Tensor):
            raise AssertionError(f"Expected torch.Tensor, got {type(ref_out)}")
        if not isinstance(compiled_out, torch.Tensor):
            raise AssertionError(f"Expected torch.Tensor, got {type(compiled_out)}")

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
        score_mod: Callable | None,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        block_mask,
        dtype: torch.dtype,
        device: str,
        page_size: int = 128,
    ) -> tuple[Tensor, Tensor, BlockMask, _score_mod_signature]:
        if block_mask is None:
            raise AssertionError("Must provide block_mask")
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

        def make_reserve_tensor(fractions):
            pattern = [KV_S * f for f in fractions]
            return torch.tensor(
                [pattern[i % len(pattern)] for i in range(KV_B)],
                device=device,
                dtype=torch.int64,
            )

        batch_reserve(
            paged_attention, make_reserve_tensor([1 / 4, 1 / 2, 1 / 4, 1 / 3])
        )
        batch_reserve(
            paged_attention, make_reserve_tensor([1 / 4, 1 / 2, 1 / 2, 1 / 2])
        )
        batch_reserve(paged_attention, make_reserve_tensor([1 / 2, 1, 1 / 2, 1]))
        batch_reserve(paged_attention, make_reserve_tensor([1, 1, 1, 1]))

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
        score_mod: Callable | None,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        dtype: torch.dtype,
        device: str,
        block_mask: BlockMask | None = None,
        kernel_options: dict | None = None,
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
        score_mod: Callable | None,
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
        block_mask: BlockMask | None = None,
    ):
        if Q_H % KV_H != 0:
            raise AssertionError(
                f"Expected Q_H % KV_H == 0, got {Q_H} % {KV_H} = {Q_H % KV_H}"
            )
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
        compiled_out, compiled_lse = self.run_paged_attention(
            score_mod, q, k, v, dtype, device, block_mask
        )
        if _is_mps_device(device):
            # MPS has no float64; the golden runs on CPU, so its score_mod
            # captures and block_mask must be relocated off-device too.
            sdpa_partial_gold = create_attention(
                _relocate_captures(score_mod, "cpu"),
                block_mask.to("cpu"),
                enable_gqa=(Q_H != KV_H),
            )
        else:
            sdpa_partial_gold = sdpa_partial
        golden_out, golden_lse = sdpa_partial_gold(
            q_gold, k_gold, v_gold, return_lse=True
        )
        ref_out, ref_lse = sdpa_partial(q_ref, k_ref, v_ref, return_lse=True)
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
        compiled_out = compiled_sdpa(q, k, v)
        # On MPS the gold path runs on CPU at fp64; relocate any device-resident
        # captures (block_mask, score_mod buffers, or a closed-over BlockMask) to
        # CPU so the reference stays single-device.
        sdpa_call_gold = sdpa_call
        if _is_mps_device(device):
            if isinstance(sdpa_call, functools.partial):
                cpu_kwargs = dict(sdpa_call.keywords)
                if cpu_kwargs.get("block_mask") is not None:
                    bm = cpu_kwargs["block_mask"].to("cpu")
                    bm.mask_mod = _relocate_captures(bm.mask_mod, "cpu")
                    cpu_kwargs["block_mask"] = bm
                if cpu_kwargs.get("score_mod") is not None:
                    cpu_kwargs["score_mod"] = _relocate_captures(
                        cpu_kwargs["score_mod"], "cpu"
                    )
                sdpa_call_gold = functools.partial(
                    sdpa_call.func, *sdpa_call.args, **cpu_kwargs
                )
            else:
                sdpa_call_gold = _relocate_captures(sdpa_call, "cpu")
        golden_out = sdpa_call_gold(q_gold, k_gold, v_gold)
        ref_out = sdpa_call(q_ref, k_ref, v_ref)
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
        # MPS gold runs on CPU at fp64, rebuild partial with CPU block_mask.
        sdpa_partial1_gold = (
            create_attention(score_mod, block_mask=block_mask1.to("cpu"))
            if _is_mps_device(device)
            else sdpa_partial1
        )
        ref_out1 = sdpa_partial1(q1_ref, k1_ref, v1_ref)
        golden_out1 = sdpa_partial1_gold(q1_gold, k1_gold, v1_gold)
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
        sdpa_partial2_gold = (
            create_attention(score_mod, block_mask=block_mask2.to("cpu"))
            if _is_mps_device(device)
            else sdpa_partial2
        )
        ref_out2 = sdpa_partial2(q2_ref, k2_ref, v2_ref)
        golden_out2 = sdpa_partial2_gold(q2_gold, k2_gold, v2_gold)

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
        sdpa_partial3_gold = (
            create_attention(score_mod, block_mask=block_mask3.to("cpu"))
            if _is_mps_device(device)
            else sdpa_partial3
        )
        ref_out3 = sdpa_partial3(q3_ref, k3_ref, v3_ref)
        golden_out3 = sdpa_partial3_gold(q3_gold, k3_gold, v3_gold)

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
        # MPS lacks fp64; gold path runs on CPU at fp64 with a CPU block_mask.
        sdpa_partial1_gold = (
            create_attention(score_mod, block_mask=block_mask1.to("cpu"))
            if _is_mps_device(device)
            else sdpa_partial1
        )
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
        q1_gold, k1_gold, v1_gold = query_key_value_clones(q1, k1, v1, torch.float64)
        golden_out1 = sdpa_partial1_gold(q1_gold, k1_gold, v1_gold)
        ref_out1 = sdpa_partial1(q1, k1, v1)

        # The second eager batch, shape (B * 2, H, S / 2, D)
        B = int(B * 2)
        S = int(S / 2)
        block_mask2 = create_block_mask(noop_mask, 1, 1, S, S, device=device)
        sdpa_partial2 = create_attention(score_mod, block_mask=block_mask2)
        sdpa_partial2_gold = (
            create_attention(score_mod, block_mask=block_mask2.to("cpu"))
            if _is_mps_device(device)
            else sdpa_partial2
        )
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
        q2_gold, k2_gold, v2_gold = query_key_value_clones(q2, k2, v2, torch.float64)
        golden_out2 = sdpa_partial2_gold(q2_gold, k2_gold, v2_gold)
        ref_out2 = sdpa_partial2(q2, k2, v2)

        # The third eager batch, shape (B * 4, H, S / 4, D)
        B = int(B * 2)
        S = int(S / 2)
        block_mask3 = create_block_mask(noop_mask, 1, 1, S, S, device=device)
        sdpa_partial3 = create_attention(score_mod, block_mask=block_mask3)
        sdpa_partial3_gold = (
            create_attention(score_mod, block_mask=block_mask3.to("cpu"))
            if _is_mps_device(device)
            else sdpa_partial3
        )
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
        q3_gold, k3_gold, v3_gold = query_key_value_clones(q3, k3, v3, torch.float64)
        golden_out3 = sdpa_partial3_gold(q3_gold, k3_gold, v3_gold)
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
        BLOCK_SIZE: int | tuple[int, int],
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
        if Hq % Hkv != 0:
            raise AssertionError(
                f"Expected Hq % Hkv == 0, got {Hq} % {Hkv} = {Hq % Hkv}"
            )

        Bq, Bkv = batch_dims
        if not (Bq > 1 and Bkv == 1):
            raise AssertionError(
                f"Expected Bq > 1 and Bkv == 1, got Bq={Bq}, Bkv={Bkv}"
            )

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
    @skip_on_cpu
    def test_create_block_mask_recompile_persistent_reduction_oob(self, device):
        """Recompiling create_block_mask with a much smaller Q_LEN must not OOB.

        When Q_LEN shrinks enough that (Q_LEN+15)//16 == 1 for all guarded
        values, _get_persistent_RBLOCK must return R0_BLOCK=1 (not 2) so
        the constant all-True mask does not read out of bounds.
        """
        compiled_create_block_mask = torch.compile(create_block_mask)

        def causal(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx

        KV_LEN = 2048

        compiled_create_block_mask(
            causal,
            B=None,
            H=None,
            Q_LEN=7896,
            KV_LEN=KV_LEN,
            device=device,
            BLOCK_SIZE=(16, 16),
        )

        compiled_bm = compiled_create_block_mask(
            causal,
            B=None,
            H=None,
            Q_LEN=2,
            KV_LEN=KV_LEN,
            device=device,
            BLOCK_SIZE=(16, 16),
        )
        eager_bm = create_block_mask(
            causal,
            B=None,
            H=None,
            Q_LEN=2,
            KV_LEN=KV_LEN,
            device=device,
            BLOCK_SIZE=(16, 16),
        )

        torch.testing.assert_close(
            compiled_bm.full_q_num_blocks,
            eager_bm.full_q_num_blocks,
        )
        torch.testing.assert_close(
            compiled_bm.q_num_blocks,
            eager_bm.q_num_blocks,
        )

    @supported_platform
    @skip_on_cpu
    @skip_on_mps  # hardcodes kernel_options={"BACKEND": "TRITON"}
    def test_kv_order_invariance_padded_causal(self, device):
        if device == "cuda" and not PLATFORM_SUPPORTS_BF16:
            self.skipTest("bf16 is required for this regression test")

        batch_size = 1
        q_heads = 16
        kv_heads = 4
        seq_len = 256
        head_dim = 128
        shared_prefix = 201
        dtype = torch.bfloat16

        def make_block_mask(real_q_len: int, separate_full_blocks: bool):
            def padded_causal_mask(b, h, q, kv):
                return (q < real_q_len) & (q >= kv)

            return create_block_mask(
                padded_causal_mask,
                B=batch_size,
                H=1,
                Q_LEN=seq_len,
                KV_LEN=seq_len,
                device=device,
                separate_full_blocks=separate_full_blocks,
            )

        q = torch.randn(
            batch_size, q_heads, seq_len, head_dim, device=device, dtype=dtype
        )
        k = torch.randn(
            batch_size, kv_heads, seq_len, head_dim, device=device, dtype=dtype
        )
        v = torch.randn(
            batch_size, kv_heads, seq_len, head_dim, device=device, dtype=dtype
        )

        compiled_attention = torch.compile(
            flex_attention,
            fullgraph=True,
        )
        default_partial_block_mask = make_block_mask(shared_prefix, True)
        default_full_block_mask = make_block_mask(seq_len, True)
        self.assertEqual(default_partial_block_mask.kv_indices[0, 0, 1, :2], [0, 1])
        self.assertEqual(default_partial_block_mask.full_kv_num_blocks[0, 0, 1], 0)
        self.assertEqual(default_full_block_mask.kv_indices[0, 0, 1, :1], [1])
        self.assertEqual(default_full_block_mask.full_kv_indices[0, 0, 1, :1], [0])

        partial_block_mask = make_block_mask(shared_prefix, False)
        full_block_mask = make_block_mask(seq_len, False)
        self.assertEqual(partial_block_mask.kv_indices[0, 0, 1, :2], [0, 1])
        self.assertIsNone(partial_block_mask.full_kv_num_blocks)
        self.assertEqual(full_block_mask.kv_indices[0, 0, 1, :2], [0, 1])
        self.assertIsNone(full_block_mask.full_kv_num_blocks)

        out_partial = compiled_attention(
            q,
            k,
            v,
            score_mod=_identity,
            block_mask=partial_block_mask,
            enable_gqa=True,
            kernel_options={"BACKEND": "TRITON"},
        )
        out_full = compiled_attention(
            q,
            k,
            v,
            score_mod=_identity,
            block_mask=full_block_mask,
            enable_gqa=True,
            kernel_options={"BACKEND": "TRITON"},
        )
        self.assertTrue(
            torch.equal(
                out_partial[:, :, :shared_prefix], out_full[:, :, :shared_prefix]
            )
        )

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
        if Hq % Hkv != 0:
            raise AssertionError(
                f"Expected Hq % Hkv == 0, got {Hq} % {Hkv} = {Hq % Hkv}"
            )

        Bq, Bkv = batch_dims
        if not (Bq > 1 and Bkv == 1):
            raise AssertionError(
                f"Expected Bq > 1 and Bkv == 1, got Bq={Bq}, Bkv={Bkv}"
            )

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
            if sum(val_max) + offset >= B * H * S * D * 2:
                raise AssertionError(
                    f"Expected sum(val_max) + offset < B * H * S * D * 2, "
                    f"got {sum(val_max) + offset} >= {B * H * S * D * 2}"
                )
            if strides[-1] != 1:
                raise AssertionError(f"Expected strides[-1] == 1, got {strides[-1]}")
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
    @skip_on_cpu
    @common_utils.parametrize(
        "trailing_shape,trailing_indices",
        [((1,), (0,)), ((4,), (2,)), ((2, 3), (1, 2))],
        name_fn=lambda trailing_shape, trailing_indices: (
            f"shape_{'x'.join(map(str, trailing_shape))}_"
            f"index_{'x'.join(map(str, trailing_indices))}"
        ),
    )
    @expected_not_implemented_on_mps
    @temp_float32_matmul_precision("highest")
    def test_captured_score_mod_nested_index_backward(
        self, device, trailing_shape, trailing_indices
    ):
        max_len = 4
        embedding_table = nn.Parameter(
            torch.randn(2 * max_len, *trailing_shape, device=device)
        )
        embedding_table_ref = nn.Parameter(embedding_table.detach().clone())
        query = torch.randn(1, 1, 3, 32, device=device)
        key = torch.randn(1, 1, 3, 32, device=device)
        value = torch.randn(1, 1, 3, 32, device=device)

        def rpe(score, _b, _h, q_idx, kv_idx):
            delta = q_idx - kv_idx
            delta = torch.clamp(delta, -max_len, max_len - 1)
            delta += max_len
            lookup = embedding_table[delta.int()]
            for index in trailing_indices:
                lookup = lookup[index]
            return score + lookup

        def rpe_ref(score, _b, _h, q_idx, kv_idx):
            delta = q_idx - kv_idx
            delta = torch.clamp(delta, -max_len, max_len - 1)
            delta += max_len
            return score + embedding_table_ref[(delta.int(), *trailing_indices)]

        compiled_flex_attention = torch.compile(flex_attention, fullgraph=True)
        out = compiled_flex_attention(query, key, value, score_mod=rpe)
        out_ref = flex_attention(query, key, value, score_mod=rpe_ref)

        self.assertEqual(out, out_ref)
        out.sum().backward()
        out_ref.sum().backward()
        self.assertEqual(embedding_table.grad, embedding_table_ref.grad)

    @supported_platform
    @skip_on_cpu
    @expected_not_implemented_on_mps
    @temp_float32_matmul_precision("highest")
    def test_captured_score_mod_nested_index_backward_dynamic(self, device):
        def attention(query, key, value, table):
            def rpe(score, _b, _h, q_idx, kv_idx):
                return score + table[q_idx - kv_idx + 2][2]

            return flex_attention(query, key, value, score_mod=rpe)

        def attention_ref(query, key, value, table):
            def rpe_ref(score, _b, _h, q_idx, kv_idx):
                return score + table[q_idx - kv_idx + 2, 2]

            return flex_attention(query, key, value, score_mod=rpe_ref)

        counter = CompileCounterWithBackend("inductor")
        compiled_attention = torch.compile(
            attention, backend=counter, fullgraph=True, dynamic=True
        )
        query = torch.randn(1, 1, 2, 32, device=device)
        key = torch.randn(1, 1, 2, 32, device=device)
        value = torch.randn(1, 1, 2, 32, device=device)
        # A plain tensor rather than an nn.Parameter, which would stay static.
        for width in (4, 6):
            table = torch.randn(5, width, device=device, requires_grad=True)
            table_ref = table.detach().clone().requires_grad_()
            out = compiled_attention(query, key, value, table)
            out_ref = attention_ref(query, key, value, table_ref)

            self.assertEqual(out, out_ref)
            out.sum().backward()
            out_ref.sum().backward()
            self.assertEqual(table.grad, table_ref.grad)
        self.assertEqual(counter.frame_count, 1)

    @supported_platform
    @skip_on_cpu
    @skip_on_mps  # MPS backward raises NotImplementedError before validation
    def test_captured_score_mod_nonterminal_index_backward_error(self, device):
        max_len = 4
        embedding_table = nn.Parameter(torch.randn(2 * max_len, 4, device=device))
        query = torch.randn(1, 1, 3, 32, device=device)
        key = torch.randn(1, 1, 3, 32, device=device)
        value = torch.randn(1, 1, 3, 32, device=device)

        def rpe(score, _b, _h, q_idx, kv_idx):
            delta = q_idx - kv_idx
            delta = torch.clamp(delta, -max_len, max_len - 1)
            delta += max_len
            return score + (embedding_table[delta.int()] * 2)[2]

        compiled_flex_attention = torch.compile(flex_attention, fullgraph=True)
        out = compiled_flex_attention(query, key, value, score_mod=rpe)
        with self.assertRaisesRegex(
            torch._inductor.exc.InductorError,
            "apply all indices in one indexing expression",
        ):
            out.sum().backward()

    @supported_platform
    @skip_on_cpu
    @skip_on_xpu
    @skip_on_mps
    @skip_on_rocm
    def test_inline_asm_score_mod(self, device):
        """score_mod using the inline_asm_elementwise HOP lowers to
        tl.inline_asm_elementwise inside the flex kernel. Forward only:
        the HOP has no autograd formula."""
        bias = torch.randn(S, device=device)

        def asm_score_mod(score, b, h, q, kv):
            return inline_asm_elementwise(
                score,
                bias[kv],
                asm_str="fma.rn.f32 $0, $1, 0f40000000, $2;",
                constraints="=f,f,f",
                dtype=torch.float32,
            )

        def ref_score_mod(score, b, h, q, kv):
            return score * 2.0 + bias[kv]

        q, k, v = [
            torch.randn(B, H, S, D, device=device, dtype=torch.float16)
            for _ in range(3)
        ]
        compiled = torch.compile(flex_attention)
        out = compiled(q, k, v, score_mod=asm_score_mod)
        ref = compiled(q, k, v, score_mod=ref_score_mod)
        self.assertEqual(out, ref, atol=1e-4, rtol=1e-3)

        # The eager (unfused) path applies score_mod under vmap and runs the
        # asm via the Jiterator.
        eager_out = flex_attention(q, k, v, score_mod=asm_score_mod)
        eager_ref = flex_attention(q, k, v, score_mod=ref_score_mod)
        self.assertEqual(eager_out, eager_ref)

    @supported_platform
    @skip_on_cpu
    @skip_on_xpu
    @skip_on_mps
    @skip_on_rocm
    def test_inline_asm_score_mod_pack2(self, device):
        def asm_score_mod(score, b, h, q, kv):
            return inline_asm_elementwise(
                score,
                asm_str="add.f32 $0, $2, $2; add.f32 $1, $3, $3;",
                constraints="=f,=f,f,f",
                dtype=torch.float32,
                pack=2,
            )

        def ref_score_mod(score, b, h, q, kv):
            return score * 2.0

        q, k, v = [
            torch.randn(B, H, S, D, device=device, dtype=torch.float16)
            for _ in range(3)
        ]
        compiled = torch.compile(flex_attention)
        out = compiled(q, k, v, score_mod=asm_score_mod)
        ref = compiled(q, k, v, score_mod=ref_score_mod)
        self.assertEqual(out, ref, atol=1e-4, rtol=1e-3)

    @supported_platform
    @skip_on_cpu
    @skip_on_xpu
    @skip_on_mps
    @skip_on_rocm
    def test_inline_asm_mask_mod(self, device):
        """mask_mod using the inline_asm_elementwise HOP: the multi-line PTX
        predicate is exercised both when building the block mask (eager vmap +
        Jiterator, and compiled) and inside the kernel on partial blocks."""
        asm_str = "{\n.reg .pred p;\nsetp.ge.s32 p, $1, $2;\nselp.u32 $0, 1, 0, p;\n}"

        def asm_causal_mask(b, h, q_idx, kv_idx):
            pred = inline_asm_elementwise(
                q_idx.to(torch.int32),
                kv_idx.to(torch.int32),
                asm_str=asm_str,
                constraints="=r,r,r",
                dtype=torch.int32,
            )
            return pred != 0

        def ref_causal_mask(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx

        bm = create_block_mask(asm_causal_mask, B, H, S, S, device=device)
        bm_ref = create_block_mask(ref_causal_mask, B, H, S, S, device=device)
        self.assertEqual(bm.kv_num_blocks, bm_ref.kv_num_blocks)
        self.assertEqual(bm.kv_indices, bm_ref.kv_indices)

        bm_compiled = torch.compile(create_block_mask)(
            asm_causal_mask, B, H, S, S, device=device
        )
        self.assertEqual(bm_compiled.kv_num_blocks, bm_ref.kv_num_blocks)

        q, k, v = [
            torch.randn(B, H, S, D, device=device, dtype=torch.float16)
            for _ in range(3)
        ]
        compiled = torch.compile(flex_attention)
        out = compiled(q, k, v, block_mask=bm)
        ref = compiled(q, k, v, block_mask=bm_ref)
        self.assertEqual(out, ref)

    @supported_platform
    @skip_on_cpu
    @expected_not_implemented_on_mps
    def test_bf16_score_mod_captured_grad_dtype(self, device):
        """
        Tests with tensors that require gradients with bf16 dtype in the score_mod
        """
        if not PLATFORM_SUPPORTS_BF16:
            self.skipTest("Platform does not support bf16")

        B_local, H_local, S_local, D_local = 2, 4, 32, 64
        dtype = torch.bfloat16
        x = torch.randn(
            (B_local, S_local, D_local),
            device=device,
            dtype=dtype,
        )

        def forward(bias_proj, x):
            q = x.view(B_local, S_local, H_local, D_local // H_local).permute(
                0, 2, 1, 3
            )
            k = x.view(B_local, S_local, H_local, D_local // H_local).permute(
                0, 2, 1, 3
            )
            v = x.view(B_local, S_local, H_local, D_local // H_local).permute(
                0, 2, 1, 3
            )
            attn_bias = (
                bias_proj(x)
                .view(B_local, S_local, H_local, S_local)
                .permute(0, 2, 1, 3)
            )

            def bias_mod(score, b, h, q_idx, kv_idx):
                return score + attn_bias[b, h, q_idx, kv_idx]

            return flex_attention(q, k, v, score_mod=bias_mod)

        def run_and_get_weight_grad(bias_proj, x, compiled):
            bias_proj.zero_grad(set_to_none=True)
            fn = torch.compile(forward) if compiled else forward
            out = fn(bias_proj, x)
            grad = torch.randn_like(out)
            out.backward(grad)
            return bias_proj.weight.grad

        bias_proj_eager = nn.Linear(
            D_local, H_local * S_local, device=device, dtype=dtype, bias=False
        )
        bias_proj_compiled = nn.Linear(
            D_local, H_local * S_local, device=device, dtype=dtype, bias=False
        )
        bias_proj_compiled.load_state_dict(bias_proj_eager.state_dict())

        torch.manual_seed(0)
        eager_weight_grad = run_and_get_weight_grad(bias_proj_eager, x, compiled=False)
        self.assertEqual(eager_weight_grad.dtype, bias_proj_eager.weight.dtype)

        torch.manual_seed(0)
        compiled_weight_grad = run_and_get_weight_grad(
            bias_proj_compiled, x, compiled=True
        )
        self.assertEqual(compiled_weight_grad.dtype, bias_proj_compiled.weight.dtype)
        self.assertEqual(eager_weight_grad, compiled_weight_grad, atol=1e-1, rtol=1e-1)

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
    @expected_not_implemented_on_mps
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
        if W * H != S:
            raise AssertionError(f"Expected W * H == S, got {W * H} != {S}")

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

        # MPS doesn't support fp64; the test only inspects FX graph code,
        # which does not embed the input dtype in this assertion.
        dtype = torch.float32 if _is_mps_device(device) else torch.float64
        make_tensor = functools.partial(
            torch.randn,
            (2, 2, 128, 4),
            device=device,
            dtype=dtype,
            requires_grad=False,
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
    @skip_on_cpu
    @dtypes(*device_configs["cuda"].dtypes_fast)
    @dtypesIfCUDA(*device_configs["cuda"].dtypes_fast)
    @dtypesIfXPU(*device_configs["xpu"].dtypes_fast)
    def test_captured_scalar_grad(self, device, dtype):
        """Test learnable scalar parameter with shape (1,) using literal index."""
        self._run_test_captured_scalar_grad(device, dtype)

    @supported_platform
    @common_utils.parametrize(
        "dtype,qk_dim,v_dim",
        (
            (torch.bfloat16, 64, 64),
            (torch.bfloat16, 128, 128),
            (torch.float16, 64, 128),
            (torch.float32, 128, 64),
        ),
    )
    @common_utils.parametrize("mask", ("dense", "causal", "sliding"))
    @common_utils.parametrize(
        "early_dp,xcd_remap",
        ((False, False), (True, False), (False, True), (True, True)),
    )
    def test_gfx950_backward_options(
        self, device, dtype, qk_dim, v_dim, mask, early_dp, xcd_remap
    ):
        self._run_gfx950_backward_options(
            device, dtype, qk_dim, v_dim, mask, early_dp, xcd_remap
        )

    @supported_platform
    def test_gfx950_backward_options_arch_guard(self, device):
        self._run_gfx950_backward_options(
            device, torch.bfloat16, 64, 64, "dense", True, True, arch="gfx942"
        )

    def _run_gfx950_backward_options(
        self, device, dtype, qk_dim, v_dim, mask, early_dp, xcd_remap, *, arch="gfx950"
    ):
        if (
            not torch.version.hip
            or torch.device(device).type != "cuda"
            or torch.cuda.get_device_properties(device).gcnArchName.split(":", 1)[0]
            != "gfx950"
        ):
            self.skipTest("Requires gfx950")
        q_len, kv_len = 257, 260
        q = torch.randn(3, 10, q_len, qk_dim, device=device, dtype=dtype) * 0.5
        k = torch.randn(3, 5, kv_len, qk_dim, device=device, dtype=dtype) * 0.5
        v = torch.randn(3, 5, kv_len, v_dim, device=device, dtype=dtype)
        scale = torch.linspace(0.75, 1.2, 10, device=device, requires_grad=True)
        inputs = tuple(tensor.requires_grad_() for tensor in (q, k, v, scale))
        grad_out = torch.randn(3, 10, q_len, v_dim, device=device, dtype=dtype) * 0.25

        def score_mod(score, b, h, q_idx, kv_idx):
            coefficient = scale[h]
            return score * coefficient + (coefficient - 1) * ((kv_idx + 1) / kv_len)

        def mask_mod(b, h, q_idx, kv_idx):
            valid = q_idx >= kv_idx
            if mask == "sliding":
                valid = valid & (q_idx - kv_idx < 67)
            return valid

        block_mask = None
        if mask != "dense":
            block_mask = create_block_mask(
                mask_mod, None, None, q_len, kv_len, device=device
            )

        def attention(query, key, value):
            return flex_attention(
                query,
                key,
                value,
                score_mod=score_mod,
                block_mask=block_mask,
                enable_gqa=True,
                kernel_options={
                    "BACKEND": "TRITON",
                    "bwd_ROCM_EARLY_DP": early_dp,
                    "bwd_XCD_REMAP": xcd_remap,
                },
            )

        from torch._inductor.kernel.flex import flex_attention as flex_kernel_mod

        architecture_patch = contextlib.nullcontext()
        if arch != "gfx950":
            # Keep the compiler and runtime on the real device.
            lowering_torch = types.SimpleNamespace(**vars(torch))
            lowering_torch.cuda = types.SimpleNamespace(**vars(torch.cuda))
            lowering_torch.cuda.get_device_properties = mock.Mock(
                return_value=types.SimpleNamespace(gcnArchName=arch)
            )
            architecture_patch = mock.patch.object(
                flex_kernel_mod, "torch", lowering_torch
            )
        with (
            architecture_patch,
            mock.patch.object(
                flex_kernel_mod.flex_attention_backward_template,
                "maybe_append_choice",
                wraps=flex_kernel_mod.flex_attention_backward_template.maybe_append_choice,
            ) as choices,
        ):
            out = torch.compile(attention, fullgraph=True)(q, k, v)
            gradients = torch.autograd.grad(out, inputs, grad_out)
        self.assertTrue(choices.call_args_list)
        for call in choices.call_args_list:
            self.assertEqual(
                call.kwargs["ROCM_EARLY_DP"], arch == "gfx950" and early_dp
            )
            self.assertEqual(call.kwargs["XCD_REMAP"], arch == "gfx950" and xcd_remap)
        ref_q, ref_k, ref_v, ref_scale = tuple(
            tensor.detach().float().requires_grad_() for tensor in inputs
        )
        scores = ref_q @ ref_k.repeat_interleave(2, dim=1).transpose(-1, -2)
        coefficient = ref_scale[None, :, None, None]
        scores = scores * (qk_dim**-0.5) * coefficient
        kv_idx = torch.arange(kv_len, device=device)
        scores = scores + (coefficient - 1) * ((kv_idx + 1) / kv_len)
        if mask != "dense":
            q_idx = torch.arange(q_len, device=device)[:, None]
            valid = q_idx >= kv_idx[None, :]
            if mask == "sliding":
                valid = valid & (q_idx - kv_idx[None, :] < 67)
            scores = scores.masked_fill(~valid, float("-inf"))
        ref_out = scores.softmax(-1) @ ref_v.repeat_interleave(2, dim=1)
        ref_gradients = torch.autograd.grad(
            ref_out, (ref_q, ref_k, ref_v, ref_scale), grad_out.float()
        )
        for actual, expected in zip(
            (out, *gradients), (ref_out, *ref_gradients), strict=True
        ):
            self.assertTrue(actual.isfinite().all().item())
            self.assertTrue(expected.isfinite().all().item())
            self.assertEqual(actual.float(), expected, rtol=0.08, atol=0.02)
            difference = actual.float() - expected
            error = difference.norm() / expected.norm().clamp_min(1e-12)
            self.assertLessEqual(error.item(), 0.01)

    @supported_platform
    @config.patch(
        fx_graph_cache=False,
        max_autotune=False,
        max_autotune_gemm=False,
        compile_threads=1,
        emulate_precision_casts=False,
    )
    @common_utils.parametrize("gqa,q_len,kv_len", ((1, 257, 260), (4, 129, 129)))
    @common_utils.parametrize("scale", (0.0625, 0.0, -0.0625))
    @common_utils.parametrize("capture", ("none", "score", "mask"))
    def test_gfx950_d256_xcd_snake(self, device, gqa, q_len, kv_len, scale, capture):
        self._run_gfx950_d256_xcd_snake(device, gqa, q_len, kv_len, scale, capture)

    def _check_gfx950_xcd_snake_values(self, actual, expected, *, strict):
        self.assertEqual(len(actual), len(expected))
        for index, (value, target) in enumerate(zip(actual, expected, strict=True)):
            self.assertTrue(value.isfinite().all().item())
            self.assertTrue(target.isfinite().all().item())
            if strict:
                atol, rtol = (0, 0) if index == 0 else (0.0003, 0.005)
                self.assertEqual(value, target, atol=atol, rtol=rtol)
                if index:
                    relative_l2 = (
                        value.float() - target.float()
                    ).norm() / target.float().norm().clamp_min(1e-12)
                    self.assertLessEqual(relative_l2.item(), 0.001)
            else:
                atol, rtol = (0.02, 0.02) if index == 0 else (0.03, 0.03)
                self.assertEqual(value.float(), target.float(), atol=atol, rtol=rtol)
        return len(actual)

    def _run_gfx950_d256_xcd_snake(self, device, gqa, q_len, kv_len, scale, capture):
        import re

        from torch._inductor.kernel.flex import flex_attention as flex_kernel_mod

        if (
            not torch.version.hip
            or torch.device(device).type != "cuda"
            or torch.cuda.get_device_properties(device).gcnArchName.split(":", 1)[0]
            != "gfx950"
        ):
            self.skipTest("Requires gfx950")
        q_heads, kv_heads = 2 * gqa, 2
        generator = torch.Generator(device=device).manual_seed(20261008)
        shapes = (
            (1, q_heads, q_len, 256),
            (1, kv_heads, kv_len, 256),
            (1, kv_heads, kv_len, 256),
        )
        q, k, v = tuple(
            torch.randn(
                shape, device=device, dtype=torch.bfloat16, generator=generator
            ).requires_grad_()
            for shape in shapes
        )
        bias = (
            torch.randn(kv_len, device=device, dtype=torch.float32, generator=generator)
            * 0.01
        ).requires_grad_()
        limit = torch.tensor([min(q_len, kv_len) - 3], device=device, dtype=torch.int32)
        inputs = (q, k, v, bias) if capture == "score" else (q, k, v)
        do = torch.randn(
            shapes[0], device=device, dtype=torch.bfloat16, generator=generator
        )
        original_do = do.clone()

        def score_mod(score, b, h, q_idx, kv_idx):
            return score + bias[kv_idx]

        def mask_mod(b, h, q_idx, kv_idx):
            if capture == "mask":
                return (q_idx >= kv_idx) & (kv_idx < limit[0])
            return q_idx >= kv_idx

        block_mask = create_block_mask(
            mask_mod, None, None, q_len, kv_len, device=device
        )

        def reference():
            reference_inputs = tuple(
                value.detach().float().requires_grad_() for value in inputs
            )
            ref_q, ref_k, ref_v = reference_inputs[:3]
            scores = (
                ref_q @ ref_k.repeat_interleave(gqa, dim=1).transpose(-1, -2)
            ) * scale
            if capture == "score":
                scores = scores + reference_inputs[3]
            q_idx = torch.arange(q_len, device=device)[:, None]
            kv_idx = torch.arange(kv_len, device=device)[None, :]
            keep = q_idx >= kv_idx
            if capture == "mask":
                keep = keep & (kv_idx < limit[0])
            probabilities = scores.masked_fill(~keep, float("-inf")).softmax(-1)
            out = probabilities @ ref_v.repeat_interleave(gqa, dim=1)
            gradients = torch.autograd.grad(out, reference_inputs, do.float())
            return (out.detach(), *(gradient.detach() for gradient in gradients))

        providers, comparisons = {}, 0
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        try:
            with torch.cuda.stream(stream), temp_float32_matmul_precision("highest"):
                for requested in (False, True):
                    torch._dynamo.reset()
                    options = dict(
                        BACKEND="TRITON",
                        bwd_ROCM_XCD_SNAKE=requested,
                        bwd_BLOCK_M1=32,
                        bwd_BLOCK_N1=64,
                        bwd_BLOCK_M2=64,
                        bwd_BLOCK_N2=64,
                        bwd_num_warps=4,
                        bwd_num_stages=1,
                        bwd_USE_TMA=False,
                        bwd_PRESCALE_QK=False,
                        bwd_ROCM_D256_PANEL_DOTS=False,
                        bwd_ROCM_SEPARATE_DQ=False,
                        bwd_ROCM_GQA_SPLIT_DKDV=False,
                        bwd_ROCM_EARLY_DP=False,
                        bwd_XCD_REMAP=False,
                    )

                    def forward(query, key, value, options=options):
                        return flex_attention(
                            query,
                            key,
                            value,
                            score_mod=score_mod if capture == "score" else None,
                            block_mask=block_mask,
                            enable_gqa=True,
                            scale=scale,
                            kernel_options=options,
                        )

                    compiled = torch.compile(forward, fullgraph=True, dynamic=False)
                    choices = []
                    original_append = flex_kernel_mod.flex_attention_backward_template.maybe_append_choice

                    def append_choice(*args, **kwargs):
                        choices.append(
                            (
                                kwargs["ROCM_XCD_SNAKE"],
                                kwargs["ROCM_BWD_PHASE"],
                                kwargs["ROCM_GQA_SPLITS"],
                                tuple(
                                    kwargs[name]
                                    for name in (
                                        "BLOCK_M1",
                                        "BLOCK_N1",
                                        "BLOCK_M2",
                                        "BLOCK_N2",
                                    )
                                ),
                                kwargs["num_warps"],
                                kwargs["num_stages"],
                            )
                        )
                        return original_append(*args, **kwargs)

                    def invoke():
                        out = compiled(q, k, v)
                        return out, torch.autograd.grad(
                            out, inputs, do, retain_graph=True
                        )

                    with mock.patch.object(
                        flex_kernel_mod.flex_attention_backward_template,
                        "maybe_append_choice",
                        side_effect=append_choice,
                    ):
                        (out, gradients), code = run_and_get_code(invoke)
                    torch.cuda.synchronize(device)
                    active = requested and capture == "none"
                    self.assertEqual(choices, [(active, 0, 1, (32, 64, 64, 64), 4, 1)])
                    backward_source = "\n".join(
                        part for part in code if "ROCM_XCD_SNAKE" in part
                    )
                    flags = re.findall(
                        r"ROCM_XCD_SNAKE\s*:\s*tl\.constexpr\s*=\s*(True|False)",
                        backward_source,
                    )
                    self.assertTrue(flags)
                    self.assertEqual(set(flags), {str(active)})
                    self.assertEqual(
                        "snake_group_size = tl.minimum(8, snake_work_per_head - snake_group_base)"
                        in backward_source,
                        active,
                    )
                    for name, expected in (
                        ("BLOCK_M1", 32),
                        ("BLOCK_N1", 64),
                        ("BLOCK_M2", 64),
                        ("BLOCK_N2", 64),
                        ("ROCM_BWD_PHASE", 0),
                        ("ROCM_GQA_SPLITS", 1),
                        ("GQA_SHARED_HEADS", gqa),
                    ):
                        self.assertRegex(
                            backward_source,
                            rf"{name}\s*:\s*tl\.constexpr\s*=\s*{expected}\b",
                        )
                    if scale == 0:
                        self.assertEqual(
                            gradients[0], torch.zeros_like(q), atol=0, rtol=0
                        )
                        self.assertEqual(
                            gradients[1], torch.zeros_like(k), atol=0, rtol=0
                        )

                    def backward(out=out):
                        return torch.autograd.grad(out, inputs, do, retain_graph=True)

                    providers[requested] = dict(
                        compiled=compiled, out=out, backward=backward, initial=gradients
                    )
                for sign in (1, -1, 1):
                    do.copy_(original_do * sign)
                    target = reference()
                    actual = {
                        name: (
                            problem["out"].detach().clone(),
                            *(
                                gradient.detach().clone()
                                for gradient in problem["backward"]()
                            ),
                        )
                        for name, problem in providers.items()
                    }
                    for value in actual.values():
                        comparisons += self._check_gfx950_xcd_snake_values(
                            value, target, strict=False
                        )
                    comparisons += self._check_gfx950_xcd_snake_values(
                        actual[True], actual[False], strict=True
                    )
                do.copy_(original_do)
                for problem in providers.values():
                    for _ in range(5):
                        problem["backward"]()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        retained = [problem["backward"]() for _ in range(10)]
                    problem.update(graph=graph, retained=retained)
                for sign in (1, -1, 1):
                    do.copy_(original_do * sign)
                    for problem in providers.values():
                        problem["graph"].replay()
                    torch.cuda.synchronize(device)
                    target = reference()
                    for index in range(10):
                        actual = {
                            name: (
                                problem["out"].detach().clone(),
                                *(
                                    gradient.detach().clone()
                                    for gradient in problem["retained"][index]
                                ),
                            )
                            for name, problem in providers.items()
                        }
                        for value in actual.values():
                            comparisons += self._check_gfx950_xcd_snake_values(
                                value, target, strict=False
                            )
                        comparisons += self._check_gfx950_xcd_snake_values(
                            actual[True], actual[False], strict=True
                        )
                self.assertEqual(comparisons, 9 * (len(inputs) + 1) * 11)
        finally:
            torch.cuda.current_stream(device).wait_stream(stream)

    @supported_platform
    @common_utils.parametrize("scale", (None, 0.0, 0.125, -0.125))
    def test_gfx950_gqa_split_dkdv(self, device, scale):
        self._run_gfx950_gqa_split_dkdv(device, scale=scale)

    @supported_platform
    @common_utils.parametrize("fwd_num_warps", (None, 4))
    def test_gfx950_gqa_split_dkdv_automatic(self, device, fwd_num_warps):
        self._run_gfx950_gqa_split_dkdv(
            device, automatic=True, fwd_num_warps=fwd_num_warps, graph_replay=True
        )

    @supported_platform
    def test_gfx950_gqa_split_dkdv_options(self, device):
        self._run_gfx950_gqa_split_dkdv(
            device, batch_size=2, early_dp=True, xcd_remap=True
        )

    @supported_platform
    def test_gfx950_gqa_split_dkdv_graph(self, device):
        self._run_gfx950_gqa_split_dkdv(device, graph_replay=True)

    @supported_platform
    @common_utils.parametrize("enabled", (False, True))
    @common_utils.parametrize("override_prefix", ("", "bwd_"))
    def test_gfx950_gqa_split_host_options(self, device, enabled, override_prefix):
        self._run_gfx950_gqa_split_dkdv(
            device, enabled=enabled, override_prefix=override_prefix
        )

    @supported_platform
    @common_utils.parametrize(
        "fallback",
        ("bf16", "causal", "sliding", "dense_mask", "score_capture", "mask_capture"),
    )
    def test_gfx950_gqa_split_dkdv_fallback(self, device, fallback):
        self._run_gfx950_gqa_split_dkdv(device, fallback=fallback)

    @supported_platform
    @common_utils.parametrize("fp16_fallback", (False, True))
    def test_gfx950_gqa_split_d128_optins(self, device, fp16_fallback):
        self._run_gfx950_separate_dq(
            device,
            "dense",
            0.125,
            fallback="dtype" if fp16_fallback else None,
            gqa_split=True,
        )

    def _run_gfx950_gqa_split_dkdv(
        self,
        device,
        *,
        scale=0.125,
        batch_size=1,
        early_dp=False,
        xcd_remap=False,
        enabled=True,
        override_prefix=None,
        fallback=None,
        graph_replay=False,
        automatic=False,
        fwd_num_warps=None,
    ):
        if (
            not torch.version.hip
            or torch.device(device).type != "cuda"
            or torch.cuda.get_device_properties(device).gcnArchName.split(":", 1)[0]
            != "gfx950"
        ):
            self.skipTest("Requires gfx950")
        q_len, kv_len = (4096, 4096) if automatic else (257, 260)
        q_heads, kv_heads = 8 * batch_size, 2 * batch_size
        dtype = torch.bfloat16 if fallback == "bf16" else torch.float16
        generator = torch.Generator(device=device).manual_seed(20261008)
        q = (
            torch.randn(
                batch_size,
                q_heads,
                q_len,
                64,
                device=device,
                dtype=dtype,
                generator=generator,
            )
            * 0.5
        )
        k = (
            torch.randn(
                batch_size,
                kv_heads,
                kv_len,
                64,
                device=device,
                dtype=dtype,
                generator=generator,
            )
            * 0.5
        )
        v = torch.randn_like(k)
        q, k, v = tuple(tensor.requires_grad_() for tensor in (q, k, v))
        inputs = (q, k, v)
        bias = (
            torch.randn(
                q_heads, kv_len, device=device, dtype=torch.float32, requires_grad=True
            )
            * 0.01
        )
        if fallback == "score_capture":
            inputs += (bias,)
        limit = torch.tensor([251], device=device, dtype=torch.int32)

        def score_mod(score, b, h, q_idx, kv_idx):
            return score + bias[h, kv_idx]

        def mask_mod(b, h, q_idx, kv_idx):
            if fallback == "mask_capture":
                return kv_idx < limit[0]
            if fallback == "dense_mask":
                return kv_idx >= 0
            valid = q_idx >= kv_idx
            return valid & (q_idx - kv_idx < 67) if fallback == "sliding" else valid

        block_mask = None
        if fallback in ("causal", "sliding", "dense_mask", "mask_capture"):
            block_mask = create_block_mask(
                mask_mod, None, None, q_len, kv_len, device=device
            )

        def attention(query, key, value, requested):
            options = {
                "BACKEND": "TRITON",
                "bwd_ROCM_GQA_SPLIT_DKDV": requested and enabled,
                "bwd_ROCM_EARLY_DP": requested and early_dp,
                "bwd_XCD_REMAP": requested and xcd_remap,
            }
            if automatic:
                options = {"BACKEND": "TRITON"}
                if not requested:
                    options["bwd_ROCM_GQA_SPLIT_DKDV"] = False
                if fwd_num_warps is not None:
                    options["fwd_num_warps"] = fwd_num_warps
            if requested and override_prefix is not None:
                options[override_prefix + "ROCM_BWD_PHASE"] = 13
                options[override_prefix + "ROCM_GQA_SPLITS"] = 17
            return flex_attention(
                query,
                key,
                value,
                block_mask=block_mask,
                score_mod=score_mod if fallback == "score_capture" else None,
                scale=scale,
                enable_gqa=True,
                kernel_options=options,
            )

        grad = torch.randn_like(q) * 0.25
        grad_outputs = (grad, -grad, grad)
        compiled = torch.compile(attention, fullgraph=True)
        stock_out = compiled(q, k, v, False)
        stock_gradients = tuple(
            torch.autograd.grad(stock_out, inputs, do, retain_graph=True)
            for do in grad_outputs
        )
        from torch._inductor.kernel.flex import flex_attention as flex_kernel_mod

        template = flex_kernel_mod.flex_attention_backward_template
        original_append, original_select = (
            template.maybe_append_choice,
            flex_kernel_mod.autotune_select_algorithm,
        )
        choice_metadata, selected = {}, []

        def append_choice(*args, **kwargs):
            choices = kwargs["choices"]
            count = len(choices)
            result = original_append(*args, **kwargs)
            for choice in choices[count:]:
                choice_metadata[id(choice)] = (
                    kwargs["ROCM_BWD_PHASE"],
                    kwargs["ROCM_GQA_SPLITS"],
                )
            return result

        def select_algorithm(*args, **kwargs):
            result = original_select(*args, **kwargs)
            if args[0] in ("flex_attention_backward", "flex_attention_backward_dq"):
                self.assertIn(id(result[1]), choice_metadata)
                selected.append(choice_metadata[id(result[1])])
            return result

        with (
            mock.patch.object(
                template, "maybe_append_choice", wraps=append_choice
            ) as choices,
            mock.patch.object(
                flex_kernel_mod, "autotune_select_algorithm", wraps=select_algorithm
            ),
        ):
            out = compiled(q, k, v, True)
            gradients = tuple(
                torch.autograd.grad(out, inputs, do, retain_graph=True)
                for do in grad_outputs
            )
        admitted = enabled and fallback is None
        splits = 4 if automatic else 2
        self.assertTrue(choices.call_args_list)
        self.assertEqual(selected, [(1, splits), (2, 1)] if admitted else [(0, 1)])
        self.assertEqual(
            [
                (call.kwargs["ROCM_BWD_PHASE"], call.kwargs["ROCM_GQA_SPLITS"])
                for call in choices.call_args_list
            ],
            [(0, 1), (1, splits), (2, 1)] if admitted else [(0, 1)],
        )
        for call in choices.call_args_list:
            self.assertEqual(call.kwargs["num_warps"], 4)
            self.assertEqual(call.kwargs["num_stages"], 2)
            self.assertEqual(
                tuple(
                    call.kwargs[name]
                    for name in ("BLOCK_M1", "BLOCK_N1", "BLOCK_M2", "BLOCK_N2")
                ),
                (32, 128, 128, 32),
            )
            if call.kwargs["ROCM_BWD_PHASE"] == 1:
                self.assertIs(
                    call.kwargs["input_nodes"][6], call.kwargs["input_nodes"][0]
                )
                self.assertEqual(call.kwargs["layout"].dtype, torch.float32)
                self.assertEqual(
                    list(call.kwargs["layout"].size),
                    [batch_size, kv_heads * splits, kv_len, 64],
                )
                mutations = call.kwargs["mutated_inputs"]
                self.assertEqual(len(mutations), 1)
                self.assertIs(mutations[0], call.kwargs["input_nodes"][7])
                self.assertEqual(mutations[0].get_dtype(), torch.float32)
                self.assertEqual(
                    list(mutations[0].get_size()),
                    [batch_size, kv_heads * splits, kv_len, 64],
                )
            elif call.kwargs["ROCM_BWD_PHASE"] == 2:
                self.assertIs(
                    call.kwargs["input_nodes"][6], call.kwargs["input_nodes"][0]
                )
                self.assertIs(
                    call.kwargs["input_nodes"][7], call.kwargs["input_nodes"][2]
                )
                self.assertEqual(call.kwargs["layout"].dtype, torch.float16)
                self.assertEqual(
                    list(call.kwargs["layout"].size), [batch_size, q_heads, q_len, 64]
                )
                self.assertEqual(call.kwargs["mutated_inputs"], [])

        reference_inputs = tuple(
            tensor.detach().float().requires_grad_() for tensor in inputs
        )
        ref_q, ref_k, ref_v = reference_inputs[:3]
        scores = (ref_q @ ref_k.repeat_interleave(4, dim=1).transpose(-1, -2)) * (
            0.125 if scale is None else scale
        )
        if fallback == "score_capture":
            scores = scores + reference_inputs[3][None, :, None, :]
        if fallback == "mask_capture":
            scores = scores.masked_fill(
                torch.arange(kv_len, device=device) >= limit[0], float("-inf")
            )
        elif fallback in ("causal", "sliding"):
            rows = torch.arange(q_len, device=device)[:, None]
            columns = torch.arange(kv_len, device=device)[None, :]
            valid = rows >= columns
            if fallback == "sliding":
                valid = valid & (rows - columns < 67)
            scores = scores.masked_fill(~valid, float("-inf"))
        ref_out = scores.softmax(-1) @ ref_v.repeat_interleave(4, dim=1)
        ref_gradients = tuple(
            torch.autograd.grad(
                ref_out, reference_inputs, do.float(), retain_graph=True
            )
            for do in grad_outputs
        )
        if graph_replay:
            mutable_grad = grad.clone()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    torch.autograd.grad(out, inputs, mutable_grad, retain_graph=True)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                captured_gradients = torch.autograd.grad(
                    out, inputs, mutable_grad, retain_graph=True
                )
            replay_gradients = []
            for do in grad_outputs:
                mutable_grad.copy_(do)
                graph.replay()
                replay_gradients.append(
                    tuple(tensor.clone() for tensor in captured_gradients)
                )
            torch.cuda.synchronize()
            gradients += tuple(replay_gradients)
            stock_gradients += stock_gradients
            ref_gradients += ref_gradients
            grad_outputs += grad_outputs
        for tensor in (out, stock_out, ref_out):
            self.assertTrue(tensor.isfinite().all().item())
        self.assertEqual(out, stock_out, atol=0, rtol=0)
        self.assertEqual(out.float(), ref_out, atol=0.02, rtol=0.02)
        for actual_grads, stock_grads, expected_grads, do in zip(
            gradients, stock_gradients, ref_gradients, grad_outputs, strict=True
        ):
            for actual, stock, expected in zip(
                actual_grads, stock_grads, expected_grads, strict=True
            ):
                self.assertTrue(actual.isfinite().all().item())
                self.assertTrue(stock.isfinite().all().item())
                self.assertTrue(expected.isfinite().all().item())
                self.assertEqual(actual, stock, atol=0.0003, rtol=0.005)
                error = (
                    actual.float() - stock.float()
                ).norm() / stock.float().norm().clamp_min(1e-12)
                self.assertLessEqual(error.item(), 0.001)
                self.assertEqual(actual.float(), expected, atol=0.03, rtol=0.03)
            if scale == 0:
                self.assertEqual(actual_grads[0], torch.zeros_like(q), atol=0, rtol=0)
                self.assertEqual(actual_grads[1], torch.zeros_like(k), atol=0, rtol=0)
                uniform = (
                    do.float().reshape(batch_size, kv_heads, 4, q_len, 64).sum((2, 3))
                    / kv_len
                )
                uniform = uniform[:, :, None, :].expand_as(k)
                self.assertEqual(actual_grads[2].float(), uniform, atol=0.03, rtol=0.03)

    @supported_platform
    @common_utils.parametrize("gqa", (1, 4))
    @common_utils.parametrize("dq_m,dq_n", ((64, 16), (128, 32)))
    @common_utils.parametrize("scale", (0.0625, 0.0, -0.0625))
    def test_gfx950_d256_dense_separate_dq(self, device, gqa, dq_m, dq_n, scale):
        self._run_gfx950_d256_dense_separate_dq(device, gqa, dq_m, dq_n, scale)

    @supported_platform
    @largeTensorTest("64GB", inductor=True)
    def test_gfx950_d256_dense_separate_dq_automatic(self, device):
        self._run_gfx950_d256_dense_separate_dq(
            device, 4, 128, 32, 0.0625, automatic=True
        )

    @supported_platform
    @common_utils.parametrize("fallback", ("causal", "mask_capture", "score_capture"))
    def test_gfx950_d256_dense_separate_dq_fallback(self, device, fallback):
        self._run_gfx950_d256_dense_separate_dq(
            device, 4, 128, 32, 0.0625, fallback=fallback
        )

    def _run_gfx950_d256_dense_separate_dq(
        self, device, gqa, dq_m, dq_n, scale, *, fallback=None, automatic=False
    ):
        if (
            not torch.version.hip
            or torch.device(device).type != "cuda"
            or torch.cuda.get_device_properties(device).gcnArchName.split(":", 1)[0]
            != "gfx950"
        ):
            self.skipTest("Requires gfx950")
        import ast

        from torch._inductor.codecache import PyCodeCache
        from torch._inductor.kernel.flex import flex_attention as flex_kernel_mod
        from torch._inductor.runtime.triton_heuristics import CachingAutotuner

        torch.manual_seed(20261008)
        batch_size = 4 if automatic else 1
        q_len, kv_len, q_heads = (4096, 4096, 32) if automatic else (257, 260, 4)
        inputs = tuple(
            tensor.requires_grad_()
            for tensor in (
                torch.randn(
                    batch_size, q_heads, q_len, 256, dtype=torch.bfloat16, device=device
                )
                * 0.5,
                torch.randn(
                    batch_size,
                    q_heads // gqa,
                    kv_len,
                    256,
                    dtype=torch.bfloat16,
                    device=device,
                )
                * 0.5,
                torch.randn(
                    batch_size,
                    q_heads // gqa,
                    kv_len,
                    256,
                    dtype=torch.bfloat16,
                    device=device,
                )
                * 0.5,
            )
        )
        q, k, v = inputs
        grad = torch.randn_like(q) * 0.25
        grad_outputs = (grad, -grad, grad)
        limit = torch.tensor([251], dtype=torch.int32, device=device)
        bias = torch.linspace(-0.1, 0.1, kv_len, device=device)

        def mask_mod(b, h, q_idx, kv_idx):
            return kv_idx < limit[0] if fallback == "mask_capture" else q_idx >= kv_idx

        def score_mod(score, b, h, q_idx, kv_idx):
            return score + bias[kv_idx]

        block_mask = (
            create_block_mask(mask_mod, None, None, q_len, kv_len, device=device)
            if fallback in ("causal", "mask_capture")
            else None
        )

        def attention(query, key, value, separate_dq, current_default):
            options = {"BACKEND": "TRITON"}
            if automatic:
                if not separate_dq:
                    options["bwd_ROCM_SEPARATE_DQ"] = False
            elif not current_default:
                options.update(
                    bwd_BLOCK_M1=16,
                    bwd_BLOCK_N1=64,
                    bwd_BLOCK_M2=dq_m,
                    bwd_BLOCK_N2=dq_n,
                    bwd_num_warps=4,
                    bwd_num_stages=2,
                    bwd_USE_TMA=False,
                    bwd_PRESCALE_QK=False,
                    bwd_ROCM_D256_PANEL_DOTS=True,
                    bwd_ROCM_EARLY_DP=True,
                    bwd_XCD_REMAP=False,
                    bwd_ROCM_SEPARATE_DQ=separate_dq,
                    bwd_ROCM_GQA_SPLIT_DKDV=separate_dq,
                    ROCM_BWD_PHASE=13,
                    bwd_ROCM_BWD_PHASE=2,
                    ROCM_GQA_SPLITS=7,
                    bwd_ROCM_GQA_SPLITS=2,
                )
            return flex_attention(
                query,
                key,
                value,
                block_mask=block_mask,
                score_mod=score_mod if fallback == "score_capture" else None,
                scale=scale,
                enable_gqa=True,
                kernel_options=options,
            )

        compiled = torch.compile(attention, fullgraph=True)
        matched_out = compiled(q, k, v, False, False)
        matched_gradients = tuple(
            torch.autograd.grad(matched_out, inputs, do, retain_graph=True)
            for do in grad_outputs
        )
        controls = [(matched_out, matched_gradients)]
        if fallback is None and not automatic:
            default_out = compiled(q, k, v, False, True)
            default_gradients = tuple(
                torch.autograd.grad(default_out, inputs, do, retain_graph=True)
                for do in grad_outputs
            )
            controls.append((default_out, default_gradients))
        template = flex_kernel_mod.flex_attention_backward_template
        original_append, original_select = (
            template.maybe_append_choice,
            flex_kernel_mod.autotune_select_algorithm,
        )
        choice_metadata, selected = {}, []

        def append_choice(*args, **kwargs):
            choices, count = kwargs["choices"], len(kwargs["choices"])
            result = original_append(*args, **kwargs)
            for choice in choices[count:]:
                choice_metadata[id(choice)] = (
                    kwargs["ROCM_BWD_PHASE"],
                    kwargs["ROCM_GQA_SPLITS"],
                )
            return result

        def select_algorithm(*args, **kwargs):
            result = original_select(*args, **kwargs)
            if args[0] in ("flex_attention_backward", "flex_attention_backward_dq"):
                selected.append(choice_metadata.get(id(result[1])))
            return result

        with (
            mock.patch.object(
                template, "maybe_append_choice", wraps=append_choice
            ) as choices,
            mock.patch.object(
                flex_kernel_mod, "autotune_select_algorithm", wraps=select_algorithm
            ),
        ):
            out = compiled(q, k, v, True, False)
            gradients = tuple(
                torch.autograd.grad(out, inputs, do, retain_graph=True)
                for do in grad_outputs
            )
        expected_phases = [0] if fallback else [1, 2]
        self.assertEqual(selected, [(phase, 1) for phase in expected_phases])
        self.assertEqual(
            [call.kwargs["ROCM_BWD_PHASE"] for call in choices.call_args_list],
            [0] if fallback else [0, 1, 2],
        )
        for call in choices.call_args_list:
            options = call.kwargs
            self.assertEqual(options["ROCM_GQA_SPLITS"], 1)
            self.assertEqual(options["GQA_SHARED_HEADS"], gqa)
            if automatic and options["ROCM_BWD_PHASE"] == 0:
                self.assertEqual(
                    tuple(
                        options[name]
                        for name in ("BLOCK_M1", "BLOCK_N1", "BLOCK_M2", "BLOCK_N2")
                    ),
                    (64, 64, 64, 64),
                )
                self.assertEqual((options["num_warps"], options["num_stages"]), (4, 1))
                self.assertFalse(options["ROCM_D256_PANEL_DOTS"])
                self.assertFalse(options["ROCM_EARLY_DP"])
                self.assertFalse(options["XCD_REMAP"])
                continue
            self.assertEqual(
                tuple(
                    options[name]
                    for name in ("BLOCK_M1", "BLOCK_N1", "BLOCK_M2", "BLOCK_N2")
                ),
                (
                    32 if automatic and options["ROCM_BWD_PHASE"] == 1 else 16,
                    64,
                    dq_m,
                    dq_n,
                ),
            )
            self.assertEqual((options["num_warps"], options["num_stages"]), (4, 2))
            self.assertTrue(options["ROCM_D256_PANEL_DOTS"])
            self.assertTrue(options["ROCM_EARLY_DP"])
            self.assertIs(
                options["XCD_REMAP"],
                automatic and gqa == 4 and options["ROCM_BWD_PHASE"] == 1,
            )
        modules = tuple(PyCodeCache.modules)
        actual_phases = set()
        expected_constants = dict(
            BLOCK_M1=16,
            BLOCK_N1=64,
            BLOCK_M2=dq_m,
            BLOCK_N2=dq_n,
            QK_HEAD_DIM=256,
            V_HEAD_DIM=256,
            GQA_SHARED_HEADS=gqa,
            ROCM_GQA_SPLITS=1,
            ROCM_D256_PANEL_DOTS=True,
            ROCM_EARLY_DP=True,
            XCD_REMAP=False,
        )
        for module in modules:
            for tuner in vars(module).values():
                if not isinstance(tuner, CachingAutotuner):
                    continue
                source = getattr(getattr(tuner, "fn", None), "src", "")
                constants = {
                    node.target.id: node.value.value
                    for node in ast.walk(ast.parse(source))
                    if isinstance(node, ast.AnnAssign)
                    and isinstance(node.target, ast.Name)
                    and isinstance(node.value, ast.Constant)
                }
                phase = constants.get("ROCM_BWD_PHASE")
                phase_constants = expected_constants.copy()
                if automatic and phase == 1:
                    phase_constants["BLOCK_M1"] = 32
                    phase_constants["XCD_REMAP"] = gqa == 4
                if all(
                    constants.get(name) == value
                    for name, value in phase_constants.items()
                ):
                    actual_phases.add(phase)
        self.assertTrue(set(expected_phases).issubset(actual_phases))
        reference_inputs = tuple(
            tensor.detach().float().requires_grad_() for tensor in inputs
        )
        ref_q, ref_k, ref_v = reference_inputs
        scores = (ref_q @ ref_k.repeat_interleave(gqa, dim=1).transpose(-1, -2)) * scale
        if fallback == "score_capture":
            scores = scores + bias.float()[None, None, None, :]
        if fallback in ("causal", "mask_capture"):
            rows, columns = (
                torch.arange(q_len, device=device)[:, None],
                torch.arange(kv_len, device=device)[None, :],
            )
            valid = (
                columns < limit[0] if fallback == "mask_capture" else rows >= columns
            )
            scores = scores.masked_fill(~valid, float("-inf"))
        ref_out = scores.softmax(-1) @ ref_v.repeat_interleave(gqa, dim=1)
        ref_gradients = tuple(
            torch.autograd.grad(
                ref_out, reference_inputs, do.float(), retain_graph=True
            )
            for do in grad_outputs
        )
        self.assertTrue(out.isfinite().all().item())
        self.assertEqual(out.dtype, torch.bfloat16)
        self.assertEqual(out.float(), ref_out, atol=0.02, rtol=0.02)
        for stock_out, stock_gradients in controls:
            self.assertEqual(out, stock_out, atol=0, rtol=0)
            for actual_grads, stock_grads, expected_grads in zip(
                gradients, stock_gradients, ref_gradients, strict=True
            ):
                for actual, stock, expected in zip(
                    actual_grads, stock_grads, expected_grads, strict=True
                ):
                    self.assertEqual(actual.dtype, torch.bfloat16)
                    self.assertTrue(actual.isfinite().all().item())
                    self.assertTrue(stock.isfinite().all().item())
                    self.assertTrue(expected.isfinite().all().item())
                    self.assertEqual(actual, stock, atol=0.0003, rtol=0.005)
                    error = (
                        actual.float() - stock.float()
                    ).norm() / stock.float().norm().clamp_min(1e-12)
                    self.assertLessEqual(error.item(), 0.001)
                    self.assertEqual(actual.float(), expected, atol=0.03, rtol=0.03)
        if scale == 0:
            for actual_grads in gradients:
                self.assertEqual(actual_grads[0], torch.zeros_like(q), atol=0, rtol=0)
                self.assertEqual(actual_grads[1], torch.zeros_like(k), atol=0, rtol=0)

    @supported_platform
    @common_utils.parametrize("mask", ("dense", "causal", "sliding"))
    @common_utils.parametrize("scale", (None, 0.125, -0.125))
    def test_gfx950_separate_dq(self, device, mask, scale):
        self._run_gfx950_separate_dq(device, mask, scale)

    @supported_platform
    def test_gfx950_separate_dq_options(self, device):
        self._run_gfx950_separate_dq(
            device,
            "dense",
            0.125,
            batch_size=2,
            q_heads=16,
            early_dp=True,
            xcd_remap=True,
        )

    @supported_platform
    @common_utils.parametrize(
        "fallback",
        (
            "architecture",
            "dtype",
            "head_dim",
            "gqa",
            "captures",
            "score_mod",
            "stride",
            "tile",
            "prescale",
        ),
    )
    def test_gfx950_separate_dq_fallback(self, device, fallback):
        self._run_gfx950_separate_dq(device, "dense", 0.125, fallback=fallback)

    def _run_gfx950_separate_dq(
        self,
        device,
        mask,
        scale,
        *,
        fallback=None,
        batch_size=1,
        q_heads=8,
        early_dp=False,
        xcd_remap=False,
        gqa_split=False,
    ):
        if (
            not torch.version.hip
            or torch.device(device).type != "cuda"
            or torch.cuda.get_device_properties(device).gcnArchName.split(":", 1)[0]
            != "gfx950"
        ):
            self.skipTest("Requires gfx950")
        q_len, kv_len = 257, 260
        dtype = torch.float16 if fallback == "dtype" else torch.bfloat16
        head_dim = 64 if fallback == "head_dim" else 128
        gqa = 2 if fallback == "gqa" else 8
        q = (
            torch.randn(
                batch_size, q_heads, q_len, head_dim, device=device, dtype=dtype
            )
            * 0.5
        )
        k = (
            torch.randn(
                batch_size, q_heads // gqa, kv_len, head_dim, device=device, dtype=dtype
            )
            * 0.5
        )
        v = torch.randn_like(k)
        if fallback == "stride":
            q = q.transpose(1, 2).contiguous().transpose(1, 2)
            self.assertFalse(q.is_contiguous())
            self.assertEqual(q.stride(-1), 1)
        inputs = tuple(tensor.requires_grad_() for tensor in (q, k, v))
        q, k, v = inputs
        grad_out = torch.randn_like(q) * 0.25
        grad_outputs = (grad_out, grad_out.flip(-2), grad_out)
        kv_limit = torch.tensor([251], device=device, dtype=torch.int32)

        def mask_mod(b, h, q_idx, kv_idx):
            if fallback == "captures":
                return kv_idx < kv_limit[0]
            valid = q_idx >= kv_idx
            if mask == "sliding":
                valid = valid & (q_idx - kv_idx < 67)
            return valid

        def score_mod(score, b, h, q_idx, kv_idx):
            return score * 1.1

        block_mask = None
        if mask != "dense" or fallback == "captures":
            block_mask = create_block_mask(
                mask_mod, None, None, q_len, kv_len, device=device
            )

        def attention(query, key, value, separate_dq):
            return flex_attention(
                query,
                key,
                value,
                block_mask=block_mask,
                score_mod=score_mod if fallback == "score_mod" else None,
                scale=scale,
                enable_gqa=True,
                kernel_options={
                    "BACKEND": "TRITON",
                    "bwd_ROCM_SEPARATE_DQ": separate_dq,
                    "bwd_ROCM_GQA_SPLIT_DKDV": separate_dq and gqa_split,
                    "bwd_ROCM_EARLY_DP": separate_dq and early_dp,
                    "bwd_XCD_REMAP": separate_dq and xcd_remap,
                    "bwd_BLOCK_M1": 32,
                    "bwd_BLOCK_N1": 64 if fallback == "tile" else 128,
                    "bwd_BLOCK_M2": 128,
                    "bwd_BLOCK_N2": 32,
                    "bwd_num_warps": 4,
                    "bwd_num_stages": 2,
                    "bwd_PRESCALE_QK": fallback == "prescale",
                },
            )

        compiled = torch.compile(attention, fullgraph=True)
        stock_out = compiled(q, k, v, False)
        stock_gradients = tuple(
            torch.autograd.grad(stock_out, inputs, grad, retain_graph=True)
            for grad in grad_outputs
        )
        from torch._inductor.kernel.flex import flex_attention as flex_kernel_mod

        architecture_patch = contextlib.nullcontext()
        if fallback == "architecture":
            lowering_torch = types.SimpleNamespace(**vars(torch))
            lowering_torch.cuda = types.SimpleNamespace(**vars(torch.cuda))
            lowering_torch.cuda.get_device_properties = mock.Mock(
                return_value=types.SimpleNamespace(gcnArchName="gfx942")
            )
            architecture_patch = mock.patch.object(
                flex_kernel_mod, "torch", lowering_torch
            )
        template = flex_kernel_mod.flex_attention_backward_template
        original_append = template.maybe_append_choice
        original_select = flex_kernel_mod.autotune_select_algorithm
        choice_phases = {}
        selected_phases = []

        def append_choice(*args, **kwargs):
            choices = kwargs["choices"]
            original_count = len(choices)
            result = original_append(*args, **kwargs)
            for choice in choices[original_count:]:
                choice_phases[id(choice)] = kwargs["ROCM_BWD_PHASE"]
            return result

        def select_algorithm(*args, **kwargs):
            result = original_select(*args, **kwargs)
            if args[0] in ("flex_attention_backward", "flex_attention_backward_dq"):
                self.assertIn(id(result[1]), choice_phases)
                selected_phases.append(choice_phases[id(result[1])])
            return result

        with (
            architecture_patch,
            mock.patch.object(
                template, "maybe_append_choice", wraps=append_choice
            ) as choices,
            mock.patch.object(
                flex_kernel_mod, "autotune_select_algorithm", wraps=select_algorithm
            ) as selections,
        ):
            out = compiled(q, k, v, True)
            gradients = tuple(
                torch.autograd.grad(out, inputs, grad, retain_graph=True)
                for grad in grad_outputs
            )
        self.assertTrue(selections.call_args_list)
        self.assertEqual(
            [call.kwargs["ROCM_BWD_PHASE"] for call in choices.call_args_list],
            [0] if fallback else [0, 1, 2],
        )
        self.assertEqual(selected_phases, [0] if fallback else [1, 2])
        for call in choices.call_args_list:
            self.assertEqual(call.kwargs["ROCM_GQA_SPLITS"], 1)
            self.assertEqual(
                call.kwargs["ROCM_EARLY_DP"], fallback != "architecture" and early_dp
            )
            self.assertEqual(
                call.kwargs["XCD_REMAP"], fallback != "architecture" and xcd_remap
            )

        ref_q, ref_k, ref_v = tuple(
            tensor.detach().float().requires_grad_() for tensor in inputs
        )
        scores = ref_q @ ref_k.repeat_interleave(gqa, dim=1).transpose(-1, -2)
        scores = scores * (head_dim**-0.5 if scale is None else scale)
        if fallback == "score_mod":
            scores = scores * 1.1
        kv_idx = torch.arange(kv_len, device=device)
        if fallback == "captures":
            scores = scores.masked_fill(kv_idx >= kv_limit[0], float("-inf"))
        elif mask != "dense":
            q_idx = torch.arange(q_len, device=device)[:, None]
            valid = q_idx >= kv_idx[None, :]
            if mask == "sliding":
                valid = valid & (q_idx - kv_idx[None, :] < 67)
            scores = scores.masked_fill(~valid, float("-inf"))
        ref_out = scores.softmax(-1) @ ref_v.repeat_interleave(gqa, dim=1)
        ref_gradients = tuple(
            torch.autograd.grad(
                ref_out, (ref_q, ref_k, ref_v), grad.float(), retain_graph=True
            )
            for grad in grad_outputs
        )
        comparisons = [(out, stock_out, ref_out, 0.02)]
        for actual_grads, stock_grads, expected_grads in zip(
            gradients, stock_gradients, ref_gradients, strict=True
        ):
            comparisons.extend(
                (actual, stock, expected, 0.03)
                for actual, stock, expected in zip(
                    actual_grads, stock_grads, expected_grads, strict=True
                )
            )
        for actual, stock, expected, tolerance in comparisons:
            for tensor in (actual, stock, expected):
                self.assertTrue(tensor.isfinite().all().item())
            self.assertEqual(actual, stock, atol=0.0003, rtol=0.005)
            error = (actual.float() - stock.float()).norm()
            error = error / stock.float().norm().clamp_min(1e-12)
            self.assertLessEqual(error.item(), 0.001)
            self.assertEqual(actual.float(), expected, atol=tolerance, rtol=tolerance)

    @supported_platform
    @common_utils.parametrize("gqa,write_dq", ((1, True), (2, False)))
    @common_utils.parametrize("mask", ("dense", "causal", "sliding"))
    @common_utils.parametrize(
        "early_dp,xcd_remap",
        ((False, False), (True, False), (False, True), (True, True)),
    )
    def test_gfx950_d256_panel_dots(
        self, device, gqa, write_dq, mask, early_dp, xcd_remap
    ):
        if (
            not torch.version.hip
            or torch.device(device).type != "cuda"
            or torch.cuda.get_device_properties(device).gcnArchName.split(":", 1)[0]
            != "gfx950"
        ):
            self.skipTest("Requires gfx950")
        head_dim = 256
        q_len, kv_len = 257, 260
        dtype = torch.bfloat16
        q = torch.randn(1, 4, q_len, head_dim, device=device, dtype=dtype) * 0.5
        k = torch.randn(1, 4 // gqa, kv_len, head_dim, device=device, dtype=dtype) * 0.5
        v = torch.randn_like(k)
        scale = torch.linspace(0.75, 1.2, 4, device=device, requires_grad=True)
        inputs = tuple(tensor.requires_grad_() for tensor in (q, k, v, scale))
        grad_out = torch.randn_like(q) * 0.25

        def score_mod(score, b, h, q_idx, kv_idx):
            coefficient = scale[h]
            return score * coefficient + (coefficient - 1) * ((kv_idx + 1) / kv_len)

        def mask_mod(b, h, q_idx, kv_idx):
            valid = q_idx >= kv_idx
            if mask == "sliding":
                valid = valid & (q_idx - kv_idx < 67)
            return valid

        block_mask = None
        if mask != "dense":
            block_mask = create_block_mask(
                mask_mod, None, None, q_len, kv_len, device=device
            )

        def attention(query, key, value):
            return flex_attention(
                query,
                key,
                value,
                score_mod=score_mod,
                block_mask=block_mask,
                enable_gqa=True,
                kernel_options={
                    "BACKEND": "TRITON",
                    "bwd_ROCM_EARLY_DP": early_dp,
                    "bwd_XCD_REMAP": xcd_remap,
                    "bwd_ROCM_D256_PANEL_DOTS": True,
                    "bwd_WRITE_DQ": write_dq,
                    "bwd_BLOCK_M1": 16,
                    "bwd_BLOCK_N1": 64,
                    "bwd_BLOCK_M2": 64,
                    "bwd_BLOCK_N2": 16,
                    "bwd_num_stages": 2,
                    "bwd_num_warps": 4,
                },
            )

        from torch._inductor.kernel.flex.flex_attention import (
            flex_attention_backward_template,
        )

        with mock.patch.object(
            flex_attention_backward_template,
            "maybe_append_choice",
            wraps=flex_attention_backward_template.maybe_append_choice,
        ) as choices:
            out = torch.compile(attention, fullgraph=True)(q, k, v)
            gradients = torch.autograd.grad(out, inputs, grad_out)
        self.assertTrue(choices.call_args_list)
        for call in choices.call_args_list:
            self.assertTrue(call.kwargs["ROCM_D256_PANEL_DOTS"])
        ref_q, ref_k, ref_v, ref_scale = tuple(
            tensor.detach().float().requires_grad_() for tensor in inputs
        )
        scores = ref_q @ ref_k.repeat_interleave(gqa, dim=1).transpose(-1, -2)
        coefficient = ref_scale[None, :, None, None]
        scores = scores * (head_dim**-0.5) * coefficient
        kv_idx = torch.arange(kv_len, device=device)
        scores = scores + (coefficient - 1) * ((kv_idx + 1) / kv_len)
        if mask != "dense":
            q_idx = torch.arange(q_len, device=device)[:, None]
            valid = q_idx >= kv_idx[None, :]
            if mask == "sliding":
                valid = valid & (q_idx - kv_idx[None, :] < 67)
            scores = scores.masked_fill(~valid, float("-inf"))
        ref_out = scores.softmax(-1) @ ref_v.repeat_interleave(gqa, dim=1)
        ref_gradients = torch.autograd.grad(
            ref_out, (ref_q, ref_k, ref_v, ref_scale), grad_out.float()
        )
        for actual, expected in zip(
            (out, *gradients), (ref_out, *ref_gradients), strict=True
        ):
            self.assertTrue(actual.isfinite().all().item())
            self.assertTrue(expected.isfinite().all().item())
            self.assertEqual(actual.float(), expected, rtol=0.08, atol=0.02)
            difference = actual.float() - expected
            error = difference.norm() / expected.norm().clamp_min(1e-12)
            self.assertLessEqual(error.item(), 0.01)

    @supported_platform
    @common_utils.parametrize("prescale", (False, True))
    @common_utils.parametrize("feature_stride", (1, 2))
    @common_utils.parametrize("mask", ("dense", "causal"))
    def test_gfx950_d256_panel_boundaries(self, device, prescale, feature_stride, mask):
        if (
            not torch.version.hip
            or torch.device(device).type != "cuda"
            or torch.cuda.get_device_properties(device).gcnArchName.split(":", 1)[0]
            != "gfx950"
        ):
            self.skipTest("Requires gfx950")
        q_len, kv_len, head_dim = 129, 133, 256
        q = (
            torch.randn(
                1,
                4,
                q_len,
                head_dim * feature_stride,
                device=device,
                dtype=torch.bfloat16,
            )
            * 0.5
        )
        k = (
            torch.randn(
                1,
                2,
                kv_len,
                head_dim * feature_stride,
                device=device,
                dtype=torch.bfloat16,
            )
            * 0.5
        )
        v = torch.randn_like(k)
        inputs = tuple(
            tensor[..., ::feature_stride].requires_grad_() for tensor in (q, k, v)
        )
        grad_out = torch.randn_like(q) * 0.25
        grad_out = grad_out[..., ::feature_stride]
        self.assertEqual(
            tuple(tensor.stride(-1) for tensor in inputs), (feature_stride,) * 3
        )
        self.assertEqual(grad_out.stride(-1), feature_stride)
        block_mask = None
        if mask == "causal":

            def mask_mod(b, h, q_idx, kv_idx):
                return q_idx >= kv_idx

            block_mask = create_block_mask(
                mask_mod, None, None, q_len, kv_len, device=device
            )

        def attention(query, key, value, panels):
            return flex_attention(
                query,
                key,
                value,
                block_mask=block_mask,
                enable_gqa=True,
                kernel_options={
                    "BACKEND": "TRITON",
                    "bwd_ROCM_D256_PANEL_DOTS": panels,
                    "bwd_ROCM_EARLY_DP": False,
                    "bwd_XCD_REMAP": False,
                    "bwd_PRESCALE_QK": prescale,
                    "bwd_BLOCK_M1": 16,
                    "bwd_BLOCK_N1": 64,
                    "bwd_BLOCK_M2": 64,
                    "bwd_BLOCK_N2": 16,
                    "bwd_num_stages": 2,
                    "bwd_num_warps": 4,
                },
            )

        compiled = torch.compile(attention, fullgraph=True)
        baseline = compiled(*inputs, False)
        baseline_grads = torch.autograd.grad(baseline, inputs, grad_out)
        out = compiled(*inputs, True)
        gradients = torch.autograd.grad(out, inputs, grad_out)
        for actual, expected in zip(
            (out, *gradients), (baseline, *baseline_grads), strict=True
        ):
            self.assertEqual(actual, expected, rtol=0.005, atol=0.0003)
            error = (actual.float() - expected.float()).norm()
            error = error / expected.float().norm().clamp_min(1e-12)
            self.assertLessEqual(error.item(), 0.001)

        ref_q, ref_k, ref_v = tuple(
            tensor.detach().float().requires_grad_() for tensor in inputs
        )
        scores = ref_q @ ref_k.repeat_interleave(2, dim=1).transpose(-1, -2)
        scores = scores * (head_dim**-0.5)
        if mask == "causal":
            q_idx = torch.arange(q_len, device=device)[:, None]
            kv_idx = torch.arange(kv_len, device=device)[None, :]
            scores = scores.masked_fill(q_idx < kv_idx, float("-inf"))
        ref_out = scores.softmax(-1) @ ref_v.repeat_interleave(2, dim=1)
        ref_grads = torch.autograd.grad(
            ref_out, (ref_q, ref_k, ref_v), grad_out.float()
        )
        for actual, expected in zip(
            (out, *gradients), (ref_out, *ref_grads), strict=True
        ):
            self.assertTrue(actual.isfinite().all().item())
            self.assertTrue(expected.isfinite().all().item())
            self.assertEqual(actual.float(), expected, rtol=0.08, atol=0.02)
            error = (actual.float() - expected).norm()
            error = error / expected.norm().clamp_min(1e-12)
            self.assertLessEqual(error.item(), 0.01)

    @supported_platform
    def test_gfx950_d256_panel_empty_rows(self, device):
        if (
            not torch.version.hip
            or torch.device(device).type != "cuda"
            or torch.cuda.get_device_properties(device).gcnArchName.split(":", 1)[0]
            != "gfx950"
        ):
            self.skipTest("Requires gfx950")
        q_len, kv_len, head_dim = 257, 260, 256
        q_shape, kv_shape = (1, 4, q_len, head_dim), (1, 2, kv_len, head_dim)
        q = torch.randn(q_shape, device=device, dtype=torch.bfloat16) * 0.5
        k = torch.randn(kv_shape, device=device, dtype=torch.bfloat16) * 0.5
        v = torch.randn_like(k)
        gain = torch.linspace(0.75, 1.2, 4, device=device, requires_grad=True)
        inputs = tuple(tensor.requires_grad_() for tensor in (q, k, v, gain))
        grad_out = torch.randn_like(q) * 0.25

        def mask_mod(b, h, q_idx, kv_idx):
            active = ((q_idx < 128) | (q_idx >= 256)) & (q_idx % 17 != 0)
            return active & (kv_idx < 113) & (q_idx >= kv_idx)

        def score_mod(score, b, h, q_idx, kv_idx):
            return score * gain[h]

        block_mask = create_block_mask(
            mask_mod, None, None, q_len, kv_len, device=device
        )

        def attention(query, key, value, panels):
            return flex_attention(
                query,
                key,
                value,
                score_mod=score_mod,
                block_mask=block_mask,
                enable_gqa=True,
                kernel_options={
                    "BACKEND": "TRITON",
                    "bwd_ROCM_D256_PANEL_DOTS": panels,
                    "bwd_ROCM_EARLY_DP": True,
                    "bwd_XCD_REMAP": True,
                    "bwd_WRITE_DQ": False,
                    "bwd_BLOCK_M1": 16,
                    "bwd_BLOCK_N1": 64,
                    "bwd_BLOCK_M2": 64,
                    "bwd_BLOCK_N2": 16,
                    "bwd_num_stages": 2,
                    "bwd_num_warps": 4,
                },
            )

        compiled = torch.compile(attention, fullgraph=True)
        baseline = compiled(q, k, v, False)
        baseline_grads = torch.autograd.grad(baseline, inputs, grad_out)
        out = compiled(q, k, v, True)
        gradients = torch.autograd.grad(out, inputs, grad_out)
        q_idx = torch.arange(q_len, device=device)[:, None]
        kv_idx = torch.arange(kv_len, device=device)[None, :]
        valid = mask_mod(None, None, q_idx, kv_idx)
        empty_rows = ~valid.any(-1)
        empty_keys = ~valid.any(-2)
        self.assertTrue(empty_rows[128:256].all().item())
        self.assertTrue(empty_keys[128:].all().item())
        self.assertEqual(valid[256].sum().item(), 113)
        for values, grads in ((baseline, baseline_grads), (out, gradients)):
            for actual in (values[:, :, empty_rows], grads[0][:, :, empty_rows]):
                self.assertEqual(actual, torch.zeros_like(actual))
            for tensor in grads[1:3]:
                actual = tensor[:, :, empty_keys]
                self.assertEqual(actual, torch.zeros_like(actual))
            for tensor in (values, *grads):
                self.assertTrue(tensor.isfinite().all().item())
        for actual, expected in zip(
            (out, *gradients), (baseline, *baseline_grads), strict=True
        ):
            self.assertEqual(actual, expected, rtol=0.005, atol=0.0003)
            error = (actual.float() - expected.float()).norm()
            error = error / expected.float().norm().clamp_min(1e-12)
            self.assertLessEqual(error.item(), 0.001)

        ref_q, ref_k, ref_v, ref_gain = tuple(
            tensor.detach().float().requires_grad_() for tensor in inputs
        )
        scores = ref_q @ ref_k.repeat_interleave(2, dim=1).transpose(-1, -2)
        scores = scores * (head_dim**-0.5) * ref_gain[None, :, None, None]
        scores = scores.masked_fill(~valid, float("-inf"))
        scores = torch.where(empty_rows[:, None], 0.0, scores)
        probabilities = scores.softmax(-1).masked_fill(~valid, 0.0)
        ref_out = probabilities @ ref_v.repeat_interleave(2, dim=1)
        ref_grads = torch.autograd.grad(
            ref_out, (ref_q, ref_k, ref_v, ref_gain), grad_out.float()
        )
        for actual, expected in zip(
            (out, *gradients), (ref_out, *ref_grads), strict=True
        ):
            self.assertTrue(expected.isfinite().all().item())
            self.assertEqual(actual.float(), expected, rtol=0.08, atol=0.02)
            error = (actual.float() - expected).norm()
            error = error / expected.norm().clamp_min(1e-12)
            self.assertLessEqual(error.item(), 0.01)

    def _run_test_captured_scalar_grad(self, device, dtype):
        def run():
            scale = torch.ones((1,), device=device, dtype=dtype, requires_grad=True)

            def score_mod_scale(qk, b, h, q, kv):
                return qk + scale[0]

            self.run_test(score_mod_scale, dtype, device=device)
            self.run_test_with_paged_attention(score_mod_scale, dtype, device=device)

        run()
        torch._dynamo.reset()
        gc.collect()
        if torch.cuda.is_available():
            torch._C._cuda_clearCublasWorkspaces()
        torch.accelerator.empty_cache()
        torch._dynamo.reset()
        gc.collect()
        if torch.cuda.is_available():
            torch._C._cuda_clearCublasWorkspaces()
        torch.accelerator.empty_cache()

    @supported_platform
    @skip_on_cpu
    @skip_on_mps
    @dtypes(*device_configs["cuda"].dtypes_fast)
    @dtypesIfCUDA(*device_configs["cuda"].dtypes_fast)
    @dtypesIfXPU(*device_configs["xpu"].dtypes_fast)
    @common_utils.parametrize("compiled", [False, True])
    def test_unused_captured_score_mod_grad(self, device, dtype, compiled):
        """Unused captured grad slots should propagate None through backward."""
        B_local, H_local, S_local, D_local = 1, 4, 16, 64
        make_input = functools.partial(
            torch.randn,
            (B_local, H_local, S_local, D_local),
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        q, k, v = make_input(), make_input(), make_input()
        unused = torch.zeros(
            (H_local, 2), device=device, dtype=dtype, requires_grad=True
        )

        def score_mod(score, b, h, q_idx, kv_idx):
            _ = unused[h]
            return score

        attention = torch.compile(flex_attention) if compiled else flex_attention
        attention(q, k, v, score_mod=score_mod).sum().backward()

        self.assertIsNotNone(q.grad)
        self.assertIsNotNone(k.grad)
        self.assertIsNotNone(v.grad)
        self.assertIsNone(unused.grad)
        torch._dynamo.reset()

    @supported_platform
    @skip_on_cpu
    @skip_on_xpu
    @dtypes(torch.float16)
    @dtypesIfCUDA(torch.float16)
    @common_utils.parametrize("detach_temp", [False, True])
    @expected_not_implemented_on_mps
    def test_captured_0d_scalar_grad(self, device, dtype, detach_temp):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.temp = nn.Parameter(
                    torch.tensor(0.7, device=device, dtype=torch.float32)
                )

            def forward(self, q, k, v):
                temp = self.temp

                def score_mod(score, b, h, q_idx, kv_idx):
                    if detach_temp:
                        return score + temp.detach()
                    return score * temp + temp

                return flex_attention(q, k, v, score_mod=score_mod)

        torch.manual_seed(123)
        shape = (1, 2, 16, 16)
        q = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
        k = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
        v = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
        grad = torch.randn(shape, device=device, dtype=dtype)
        q2 = q.detach().clone().requires_grad_()
        k2 = k.detach().clone().requires_grad_()
        v2 = v.detach().clone().requires_grad_()
        m1 = M()
        m2 = M()
        m2.load_state_dict(m1.state_dict())

        out1 = m1(q, k, v)
        out1.backward(grad)
        out2 = torch.compile(m2)(q2, k2, v2)
        out2.backward(grad)
        if device == "cuda":
            torch.cuda.synchronize()

        pairs = [
            (out1, out2),
            (q.grad, q2.grad),
            (k.grad, k2.grad),
            (v.grad, v2.grad),
        ]
        if detach_temp:
            self.assertIsNone(m1.temp.grad)
            self.assertIsNone(m2.temp.grad)
        else:
            self.assertIsNotNone(m1.temp.grad)
            self.assertIsNotNone(m2.temp.grad)
            pairs.append((m1.temp.grad, m2.temp.grad))
        for a, b in pairs:
            self.assertEqual(a, b, atol=1e-2, rtol=1e-2)

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
    @expected_not_implemented_on_mps
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
    @skip_on_cpu
    @dtypes(torch.bfloat16)
    @dtypesIfCUDA(torch.bfloat16)
    @expected_not_implemented_on_mps  # uses requires_grad+autograd.grad; backward NIE on MPS
    def test_compacted_block_mask_matches_reference_after_recompile(
        self, device, dtype
    ):
        block_size = 128
        seq_len = 512
        head_dim = 64
        batch_size = 1

        def mask_fn(b, h, q_idx, kv_idx, document_id):
            causal = q_idx >= kv_idx
            same_document = document_id[b, q_idx] == document_id[b, kv_idx]
            return causal | same_document

        def make_block_mask(document_id, compact):
            ref = create_block_mask(
                functools.partial(mask_fn, document_id=document_id),
                B=batch_size,
                H=None,
                Q_LEN=seq_len,
                KV_LEN=seq_len,
                device=device,
                BLOCK_SIZE=block_size,
            )
            if not compact:
                return ref

            nums = [
                ref.kv_num_blocks,
                ref.full_kv_num_blocks,
                ref.q_num_blocks,
                ref.full_q_num_blocks,
            ]
            indices = [
                ref.kv_indices,
                ref.full_kv_indices,
                ref.q_indices,
                ref.full_q_indices,
            ]
            compact_indices = [
                index[..., : max(int(num.max()), 1)].contiguous()
                for num, index in zip(nums, indices, strict=True)
            ]
            return BlockMask(
                seq_lengths=ref.seq_lengths,
                kv_num_blocks=ref.kv_num_blocks,
                kv_indices=compact_indices[0],
                full_kv_num_blocks=ref.full_kv_num_blocks,
                full_kv_indices=compact_indices[1],
                q_num_blocks=ref.q_num_blocks,
                q_indices=compact_indices[2],
                full_q_num_blocks=ref.full_q_num_blocks,
                full_q_indices=compact_indices[3],
                BLOCK_SIZE=ref.BLOCK_SIZE,
                mask_mod=ref.mask_mod,
            )

        document_id_1 = torch.arange(seq_len, dtype=torch.int64, device=device).repeat(
            batch_size, 1
        )
        document_id_1[:, :400] = 0
        document_id_2 = torch.arange(seq_len, dtype=torch.int64, device=device).repeat(
            batch_size, 1
        )

        query = torch.randn(
            (batch_size, 1, seq_len, head_dim), device=device, dtype=dtype
        )
        key = torch.randn_like(query)
        value = torch.randn_like(query)

        def run(mask, attention):
            q, k, v = [
                x.detach().clone().requires_grad_(True) for x in (query, key, value)
            ]
            out = attention(q, k, v, block_mask=mask)
            return out, torch.autograd.grad(out.sum(), (q, k, v))

        ref_mask_1 = make_block_mask(document_id_1, compact=False)
        ref_mask_2 = make_block_mask(document_id_2, compact=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ref_out_1, ref_grads_1 = run(ref_mask_1, flex_attention)
            ref_out_2, ref_grads_2 = run(ref_mask_2, flex_attention)

        compact_mask_1 = make_block_mask(document_id_1, compact=True)
        compact_mask_2 = make_block_mask(document_id_2, compact=True)

        torch._dynamo.reset()
        compiled_fa = torch.compile(flex_attention, fullgraph=True, dynamic=False)
        out_1, grads_1 = run(compact_mask_1, compiled_fa)
        out_2, grads_2 = run(compact_mask_2, compiled_fa)

        torch.testing.assert_close(ref_out_1, out_1, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(ref_out_2, out_2, atol=2e-2, rtol=2e-2)
        for ref_grad, grad in zip(ref_grads_1, grads_1):
            torch.testing.assert_close(ref_grad, grad, atol=3e-2, rtol=3e-2)
        for ref_grad, grad in zip(ref_grads_2, grads_2):
            torch.testing.assert_close(ref_grad, grad, atol=3e-2, rtol=3e-2)

    @supported_platform
    @dtypes(*device_configs["cpu"].dtypes_fast)
    @dtypesIfCUDA(*device_configs["cuda"].dtypes_fast)
    @dtypesIfXPU(*device_configs["xpu"].dtypes_fast)
    @common_utils.parametrize(
        "score_mod", test_score_mods, name_fn=lambda score_mod: score_mod.__name__
    )
    @skip_on_cpu
    def test_return_max(self, device, dtype, score_mod):
        # MPS has forward-only flex_attention
        requires_grad = device in DEVICE_SUPPORTS_BACKWARDS
        make_tensor = functools.partial(
            torch.randn,
            (2, 2, 243, 16),
            device=device,
            dtype=dtype,
            requires_grad=requires_grad,
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
        if requires_grad:
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
                        grad,
                        lambda msg: f"{msg}\n{input_name} should receive gradients in {description}",
                    )

    @supported_platform
    @dtypes(*device_configs["cpu"].dtypes_fast)
    @dtypesIfCUDA(*device_configs["cuda"].dtypes_fast)
    @dtypesIfXPU(*device_configs["xpu"].dtypes_fast)
    @common_utils.parametrize(
        "score_mod", test_score_mods, name_fn=lambda score_mod: score_mod.__name__
    )
    @skip_on_cpu
    @expected_not_implemented_on_mps
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

    @unittest.skipIf(
        IS_LINUX or TEST_WITH_ROCM or TEST_WITH_SLOW,
        "https://github.com/pytorch/pytorch/issues/162464",
    )
    @supported_platform
    @dtypes(*device_configs["cpu"].dtypes_fast)
    @dtypesIfCUDA(*device_configs["cuda"].dtypes_fast)
    @dtypesIfXPU(*device_configs["xpu"].dtypes_fast)
    @skip_on_cpu
    @skip_on_mps  # tests deprecation-warning emission; flaky under test-suite ordering
    def test_return_aux_deprecation_warnings(self, device, dtype):
        """Test that deprecation warnings are issued for legacy parameters"""
        import warnings

        import torch.nn.attention.flex_attention as fa

        make_tensor = functools.partial(
            torch.randn,
            (2, 2, 64, 16),
            device=device,
            dtype=dtype,
        )
        query, key, value = make_tensor(), make_tensor(), make_tensor()

        # Clear shown warnings to ensure we can test them
        original_shown = fa._WARNINGS_SHOWN.copy()
        fa._WARNINGS_SHOWN.clear()

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
            fa._WARNINGS_SHOWN.clear()

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
            fa._WARNINGS_SHOWN.clear()
            fa._WARNINGS_SHOWN.update(original_shown)

    @supported_platform
    @dtypes(*device_configs["cpu"].dtypes)
    @dtypesIfCUDA(*device_configs["cuda"].dtypes)
    @dtypesIfXPU(*device_configs["xpu"].dtypes)
    @expected_not_implemented_on_mps
    def test_autocast(self, device, dtype):
        """Test torch autocast functionality"""
        q = torch.randn(1, 1, 1024, 64, device=device, dtype=dtype).to(torch.float16)
        k = torch.randn(1, 1, 1024, 64, device=device, dtype=dtype)
        v = torch.randn(1, 1, 1024, 64, device=device, dtype=dtype)

        atol = 1e-3
        rtol = 1e-3

        if isRocmArchAnyOf(MI200_ARCH):
            # Every dtype variant compares the same fp16 kernels: q is cast to
            # fp16 above and autocast narrows k/v. On MI200 the sdpa and flex
            # fp16 outputs disagree by up to 1.21e-3 abs on a handful of
            # normal-magnitude elements (5/65536 measured), so atol needs
            # headroom; rtol does not (no subnormals are involved).
            atol = 2e-3

        with torch.autocast(dtype=torch.float16, enabled=True, device_type=device):
            sdpa_output = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            flex_output = flex_attention(q, k, v)
            torch.testing.assert_close(sdpa_output, flex_output, atol=atol, rtol=rtol)
            self.assertEqual(flex_output.dtype, torch.float16)

    @supported_platform
    @dtypes(*device_configs["cpu"].dtypes_fast)
    @dtypesIfCUDA(*device_configs["cuda"].dtypes_fast)
    @dtypesIfXPU(*device_configs["xpu"].dtypes_fast)
    @skip_on_cpu
    @skip_on_mps  # asserts Triton "IS_DIVISIBLE : tl.constexpr" in generated code
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
                expected_flag,
                str(code),
                lambda msg: f"{msg}\nS={S} should have {expected_flag}",
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
    @skip_on_cpu
    def test_mask_mod_handles_derived_symint_closure(self, device):
        dtype = torch.float16

        def run(q, k, v, current_pos):
            p_plus = current_pos + 1

            def _opaque_mask(b, h, q_idx, kv_idx):
                return kv_idx <= p_plus

            block_mask = create_block_mask(
                _opaque_mask,
                B=None,
                H=None,
                Q_LEN=q.size(-2),
                KV_LEN=k.size(-2),
                device=device,
            )
            return flex_attention(q, k, v, block_mask=block_mask)

        compiled_run = torch.compile(run, fullgraph=True, dynamic=True)

        q = torch.randn(1, 2, 192, 32, device=device, dtype=dtype)
        k = torch.randn(1, 2, 128, 32, device=device, dtype=dtype)
        v = torch.randn(1, 2, 128, 32, device=device, dtype=dtype)

        eager_out = run(q, k, v, 5)
        compiled_out = compiled_run(q, k, v, 5)
        torch.testing.assert_close(eager_out, compiled_out, atol=1e-3, rtol=1e-3)

        # Exercise a different captured value to ensure derived SymInt captures remain well-formed.
        eager_out2 = run(q, k, v, 9)
        compiled_out2 = compiled_run(q, k, v, 9)
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
    @expected_not_implemented_on_mps
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
    @expected_not_implemented_on_mps
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
    @skip_on_mps  # asserts "triton_tem_fused" in generated code
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
    @skip_on_mps  # eager fp8 mixed-dtype path; MPS lacks the dispatch
    def test_mixed_dtypes_eager(self, device):
        dtype_high = torch.float16 if PLATFORM_SUPPORTS_FP8 else torch.float32
        dtype_low = e4m3_type if PLATFORM_SUPPORTS_FP8 else torch.float16
        query = torch.randn((1, 1, 1024, 64), dtype=dtype_high, device=device)
        key = torch.randn((1, 1, 1024, 64), dtype=dtype_high, device=device).to(
            dtype_low
        )
        value = torch.randn((1, 1, 1024, 64), dtype=dtype_high, device=device).to(
            dtype_low
        )
        out = flex_attention(query, key, value, _identity)
        self.assertEqual(out.shape, query.shape)
        self.assertEqual(out.dtype, query.dtype)

    @supported_platform
    @expected_not_implemented_on_mps
    def test_mixed_dtypes_compiled(self, device):
        # MPS doesn't support fp8 dtype conversion; force the fp16/fp32 mix on MPS.
        use_fp8 = PLATFORM_SUPPORTS_FP8 and not _is_mps_device(device)
        dtype_high = torch.float16 if use_fp8 else torch.float32
        dtype_low = e4m3_type if use_fp8 else torch.float16
        query = torch.randn((1, 1, 1024, 64), dtype=dtype_high, device=device)
        key = torch.randn((1, 1, 1024, 64), dtype=dtype_high, device=device).to(
            dtype_low
        )
        value = torch.randn((1, 1, 1024, 64), dtype=dtype_high, device=device).to(
            dtype_low
        )
        compiled_fn = torch.compile(flex_attention, fullgraph=True)
        if device == "cpu":
            with self.assertRaisesRegex(
                InductorError,
                "Mixed query, key, and value dtype is not supported on this platform",
            ):
                compiled_fn(query, key, value, _identity)
        else:
            out = compiled_fn(query, key, value, _identity)
            self.assertEqual(out.shape, query.shape)
            self.assertEqual(out.dtype, query.dtype)

    @skip_on_cpu
    @supported_platform
    @skipUnless(PLATFORM_SUPPORTS_FP8, "FP8 is not supported on this platform")
    @skip_on_mps  # requires PLATFORM_SUPPORTS_FP8
    def test_mixed_dtypes_sqnr_per_tensor(self, device):
        query_ref = torch.testing.make_tensor(
            (1, 1, 1024, 64), dtype=torch.bfloat16, device=device
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

        compiled_fn = torch.compile(flex_attention, fullgraph=True)
        out = compiled_fn(query_ref, key_fp8, value_fp8, score_mod) * value_scale
        out_ref = compiled_fn(query_ref, key_ref, value_ref, _identity)
        _, _, sqnr = _snr(out_ref, out)
        self.assertGreater(sqnr, 10)

    @skip_on_cpu
    @supported_platform
    @skipUnless(PLATFORM_SUPPORTS_FP8, "FP8 is not supported on this platform")
    @skip_on_mps  # requires PLATFORM_SUPPORTS_FP8
    def test_mixed_dtypes_sqnr_per_head(self, device):
        query_ref = torch.testing.make_tensor(
            (1, 4, 1024, 64), dtype=torch.bfloat16, device=device
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

        compiled_fn = torch.compile(flex_attention, fullgraph=True)
        out = compiled_fn(query_ref, key_fp8, value_fp8, score_mod) * value_scale_b
        out_ref = compiled_fn(query_ref, key_ref, value_ref, _identity)
        _, _, sqnr = _snr(out_ref, out)
        self.assertGreater(sqnr, 10)

    @supported_platform
    @skip_on_cpu
    @skip_on_mps  # uses .to(fp8); MPS lacks fp8 dtype
    def test_mixed_dtype_backwards_eager(self, device):
        dtype_high = torch.float16 if PLATFORM_SUPPORTS_FP8 else torch.float32
        dtype_low = e4m3_type if PLATFORM_SUPPORTS_FP8 else torch.float16
        q = torch.testing.make_tensor(
            (1, 1, 1024, 64),
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
        out = flex_attention(q, k, v, _identity).mean()
        with self.assertRaisesRegex(
            ValueError,
            "Backward pass with mixed query, key, and value dtype is not supported",
        ):
            out.backward()

    @supported_platform
    @skip_on_cpu
    @skip_on_mps  # assertRaisesRegex on Triton-specific backward error
    def test_mixed_dtype_backwards_compiled(self, device):
        dtype_high = torch.float16 if PLATFORM_SUPPORTS_FP8 else torch.float32
        dtype_low = e4m3_type if PLATFORM_SUPPORTS_FP8 else torch.float16
        q = torch.testing.make_tensor(
            (1, 1, 1024, 64),
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

        compiled_fn = torch.compile(flex_attention, fullgraph=True)

        with self.assertRaisesRegex(
            InductorError,
            "Backward pass with mixed query, key, and value dtype is not supported",
        ):
            out_mixed = (compiled_fn(q, k, v, _identity)).mean()
            out_mixed.backward()

    @supported_platform
    @patch.object(torch._inductor.config, "max_autotune", True)
    @skip_on_mps  # uses Triton max_autotune (no autotuning on MPS)
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
    @skip_on_mps  # uses Triton max_autotune
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
    @skip_on_cpu
    @skip_on_mps  # uses Triton max_autotune
    @common_utils.parametrize("backend", ["TRITON", "TRITON_DECODE"])
    def test_max_autotune_with_gathered_captured_buffer(self, device, backend):
        B, H, N, D = 2, 2, 49, 32
        query, key, value = (torch.randn(B, H, N, D, device=device) for _ in range(3))
        table = torch.randn(2 * N, H, device=device)
        indices = torch.randint(0, 2 * N, (N, N), device=device)

        def attention(query, key, value):
            bias = table[indices.flatten()].view(N, N, H).permute(2, 0, 1).contiguous()

            def score_mod(score, batch, head, query_idx, kv_idx):
                return score + bias[head, query_idx, kv_idx]

            return flex_attention(
                query,
                key,
                value,
                score_mod=score_mod,
                kernel_options={"BACKEND": backend},
            )

        expected = attention(query, key, value)
        actual = torch.compile(
            attention, fullgraph=True, mode="max-autotune-no-cudagraphs"
        )(query, key, value)

        self.assertEqual(actual, expected, atol=5e-3, rtol=0)

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
    @expected_not_implemented_on_mps
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
    @expected_not_implemented_on_mps  # backward path (return_lse / requires_grad); NIE on MPS
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

        q_ref, k_ref, v_ref = query_key_value_clones(q, k, v, torch.float64)
        ref_out, ref_lse = eager_sdpa_hop(q_ref, k_ref, v_ref, score_mod)
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
    @expected_not_implemented_on_mps
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
        FileCheck().check_count(".run(primals_2, primals_1, primals_3", 1, True).run(
            code[0]
        )

    @supported_platform
    @skip_on_cpu
    @common_utils.parametrize(
        "score_mod", [_identity, _causal, _times_two, _squared, _trig, _trig2]
    )
    @expected_not_implemented_on_mps  # backward path; NIE on MPS via _validate_device
    def test_aot_eager_gradcheck(self, device, score_mod):
        # MPS lacks fp64; on MPS we just need _validate_device to raise NIE,
        # so use fp32 — the call never gets far enough for precision to matter.
        dtype = torch.float32 if _is_mps_device(device) else torch.float64
        make_tensor = functools.partial(
            torch.randn,
            (2, 2, 11, 4),
            device=device,
            dtype=dtype,
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
    @expected_not_implemented_on_mps
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
    @expected_not_implemented_on_mps  # backward path; NIE on MPS via _validate_device
    def test_differentiable_logsumexp_gradcheck(self, device):
        # See test_aot_eager_gradcheck for the per-device dtype rationale.
        dtype = torch.float32 if _is_mps_device(device) else torch.float64
        make_tensor = functools.partial(
            torch.randn,
            (2, 2, 11, 4),
            device=device,
            dtype=dtype,
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
    @expected_not_implemented_on_mps
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

    @supported_platform
    @skip_on_cpu
    @expected_not_implemented_on_mps
    def test_score_mod_without_score_gradient_errors(self, device):
        def score_mod(score, b, h, q_idx, kv_idx):
            return q_idx >= kv_idx

        dtype = torch.float16 if torch.device(device).type == "cuda" else torch.float32
        make_tensor = functools.partial(
            torch.randn,
            (1, 1, 128, 16),
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        q, k, v = make_tensor(), make_tensor(), make_tensor()

        with self.assertRaisesRegex(
            RuntimeError,
            "flex_attention backward requires the output of score_mod to depend on score",
        ):
            torch.compile(flex_attention, dynamic=False)(q, k, v, score_mod=score_mod)

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
    @expected_not_implemented_on_mps
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
    @expected_not_implemented_on_mps  # backward path; NIE on MPS via _validate_device
    def test_captured_score_mod_aot_eager_gradcheck(
        self, device, score_mod_name: str, mode: str
    ):
        # See test_aot_eager_gradcheck for the per-device dtype rationale.
        dtype = torch.float32 if _is_mps_device(device) else torch.float64
        make_tensor = functools.partial(
            torch.randn,
            (2, 2, 11, 4),
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        query, key, value = make_tensor(), make_tensor(), make_tensor()

        func = torch.compile(flex_attention, backend=mode, fullgraph=True)
        score_mod = captured_buffers_map[score_mod_name](dtype, device)

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

        # MPS doesn't support fp64; this test has no numerical assertions.
        dtype = torch.float32 if _is_mps_device(device) else torch.float64
        make_tensor = functools.partial(
            torch.randn,
            (2, 1, 128, 4),
            device=device,
            dtype=dtype,
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
    @expected_not_implemented_on_mps
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
            (1, 0, 2, 3),  # Swap first two dims
            (0, 2, 1, 3),  # Mixed order
            (2, 0, 1, 3),  # Another mixed order
            (0, 1, 3, 2),  # Swap last two dims (non-contiguous last dim)
            (1, 0, 3, 2),  # Swap first two AND last two dims
        ],
    )
    @common_utils.parametrize("shape", [(2, 1, 128, 16), (4, 2, 64, 16)])
    def test_flex_attention_stride_ordering(self, device, mode, permute_order, shape):
        from torch._inductor.ir import get_stride_order

        if mode == "paged_attention" and _is_mps_device(device):
            self.skipTest(
                "paged attention captures the page table; not supported on MPS"
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

        # We always enforce stride[-1]=1 for attention kernels
        self.assertEqual(out.stride()[-1], 1, "Output must have contiguous last dim")

        # When input already has stride[-1]=1, verify full stride order is preserved
        if query.stride()[-1] == 1:
            out_stride_order = get_stride_order(out.stride())
            query_stride_order = get_stride_order(query.stride())
            self.assertEqual(
                out_stride_order,
                query_stride_order,
                lambda msg: f"{msg}\nStride order mismatch: out {out_stride_order}, query {query_stride_order}",
            )

    @supported_platform
    @skip_on_cpu
    @common_utils.parametrize("mode", ["eager", "inductor"])
    @common_utils.parametrize(
        "permute_order",
        [
            (0, 1, 2, 3),
            (1, 0, 2, 3),
            (0, 2, 1, 3),
            (2, 0, 1, 3),
            (0, 1, 3, 2),
            (1, 0, 3, 2),
        ],
    )
    @common_utils.parametrize("shape", [(2, 5, 128, 16), (4, 2, 64, 16)])
    @expected_not_implemented_on_mps  # backward path; NIE on MPS via _validate_device
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
            self.assertIsNotNone(
                grad, lambda msg: f"{msg}\nGrad {name} should be computed"
            )
            self.assertFalse(
                torch.isnan(grad).any(), lambda msg: f"{msg}\nGrad {name} contains NaN"
            )

            # When input has stride[-1]=1, verify stride order is preserved
            if leaf.stride()[-1] == 1:
                input_stride_order = get_stride_order(grad.stride())
                orig_stride_order = get_stride_order(leaf.stride())
                self.assertEqual(
                    input_stride_order,
                    orig_stride_order,
                    lambda msg: f"{msg}\nMode: {mode}, Stride order mismatch for {name}: grad {input_stride_order}, input {orig_stride_order}.",
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
    @skip_on_mps  # asserts Triton "BLOCK_M : tl.constexpr" string in generated code
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
    @skip_on_mps
    @skip_on_xpu
    def test_backward_kernel_options_filtering(self, device):
        if not PLATFORM_SUPPORTS_BF16:
            self.skipTest("bf16 is required for this regression test")

        b, h, n, c = 1, 8, 234, 16
        block_size = 32

        def mask_mod(b, h, q, kv):
            return q > kv

        block_mask = create_block_mask(
            mask_mod,
            B=None,
            H=None,
            Q_LEN=n,
            KV_LEN=n,
            device=device,
            BLOCK_SIZE=block_size,
        )
        kernel_options = {
            "BLOCK_M": block_size,
            "BLOCK_M1": block_size,
            "BLOCK_M2": block_size,
            "BLOCK_N": block_size,
            "BLOCK_N1": block_size,
            "BLOCK_N2": block_size,
        }
        make_tensor = functools.partial(
            torch.randn,
            (b, h, n, c),
            device=device,
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        query, key, value = make_tensor(), make_tensor(), make_tensor()

        out = torch.compile(flex_attention)(
            query, key, value, block_mask=block_mask, kernel_options=kernel_options
        )
        out.mean().backward()

        self.assertEqual(query.grad.shape, query.shape)

    @supported_platform
    @skip_on_cpu
    @skip_on_mps
    @skip_on_xpu
    def test_invalid_forward_kernel_options_error(self, device):
        def mask_mod(b, h, q, kv):
            return q >= kv

        block_mask = create_block_mask(
            mask_mod,
            B=None,
            H=None,
            Q_LEN=64,
            KV_LEN=64,
            device=device,
            BLOCK_SIZE=32,
        )
        make_tensor = functools.partial(
            torch.randn,
            (1, 1, 64, 16),
            device=device,
            dtype=torch.float16,
        )
        query, key, value = make_tensor(), make_tensor(), make_tensor()

        with self.assertRaisesRegex(
            (InductorError, ValueError),
            "Invalid FlexAttention forward kernel options.*fwd_BLOCK_M.*fwd_num_stages",
        ):
            torch.compile(flex_attention)(
                query,
                key,
                value,
                block_mask=block_mask,
                kernel_options={"BACKEND": "TRITON", "BLOCK_M": 64, "BLOCK_N": 32},
            )

    @supported_platform
    @skip_on_cpu
    @skip_on_mps
    @skip_on_xpu
    def test_invalid_backward_kernel_options_error(self, device):
        def mask_mod(b, h, q, kv):
            return q >= kv

        block_mask = create_block_mask(
            mask_mod,
            B=None,
            H=None,
            Q_LEN=64,
            KV_LEN=64,
            device=device,
            BLOCK_SIZE=32,
        )
        make_tensor = functools.partial(
            torch.randn,
            (1, 1, 64, 16),
            device=device,
            dtype=torch.float16,
            requires_grad=True,
        )
        query, key, value = make_tensor(), make_tensor(), make_tensor()

        out = torch.compile(flex_attention)(
            query,
            key,
            value,
            block_mask=block_mask,
            kernel_options={
                "BLOCK_M": 32,
                "BLOCK_N": 32,
                "bwd_BLOCK_M1": 64,
                "bwd_BLOCK_N1": 64,
                "bwd_BLOCK_M2": 64,
                "bwd_BLOCK_N2": 64,
            },
        )
        with self.assertRaisesRegex(
            (InductorError, ValueError),
            "Invalid FlexAttention backward kernel options.*"
            "bwd_BLOCK_M1.*bwd_num_stages",
        ):
            out.mean().backward()

    @supported_platform
    @skip_on_cpu
    @skip_on_mps
    @skip_on_xpu
    def test_invalid_decode_kernel_options_error(self, device):
        def mask_mod(b, h, q, kv):
            return q >= kv

        block_mask = create_block_mask(
            mask_mod,
            B=None,
            H=None,
            Q_LEN=1,
            KV_LEN=256,
            device=device,
            BLOCK_SIZE=128,
        )
        query = torch.randn(1, 2, 1, 64, device=device, dtype=torch.float16)
        key = torch.randn(1, 2, 256, 64, device=device, dtype=torch.float16)
        value = torch.randn(1, 2, 256, 64, device=device, dtype=torch.float16)

        with self.assertRaisesRegex(
            (InductorError, ValueError),
            "Invalid FlexAttention decode kernel options.*fwd_BLOCK_M.*fwd_num_stages",
        ):
            torch.compile(flex_attention)(
                query,
                key,
                value,
                block_mask=block_mask,
                kernel_options={
                    "BACKEND": "TRITON_DECODE",
                    "fwd_BLOCK_M": 96,
                    "fwd_BLOCK_N": 64,
                },
            )

    @supported_platform
    @skip_on_cpu
    @skip_on_mps
    @skip_on_xpu
    def test_flex_flash_honors_sparse_metadata_with_trivial_mask(self, device):
        """FLASH should honor BlockMask metadata without relying on mask_mod."""
        from torch._inductor.kernel.flex.flex_flash_attention import (
            ensure_flash_available,
        )

        if not ensure_flash_available():
            self.skipTest("flash-attn-4 is not available")
        if torch.cuda.get_device_capability(device)[0] not in (10, 11):
            self.skipTest("FA4 sparse block path in this test targets SM100+")

        torch.manual_seed(0)
        batch = 1
        heads = 1
        q_blocks = 2
        kv_blocks = 2
        q_block_size = 256
        kv_block_size = 128
        head_dim = 128
        top_k = 1
        q = torch.randn(
            batch,
            heads,
            q_blocks * q_block_size,
            head_dim,
            device=device,
            dtype=torch.bfloat16,
        )
        k = torch.randn(
            batch,
            heads,
            kv_blocks * kv_block_size,
            head_dim,
            device=device,
            dtype=torch.bfloat16,
        )
        v = torch.randn(
            batch,
            heads,
            kv_blocks * kv_block_size,
            head_dim,
            device=device,
            dtype=torch.bfloat16,
        )

        selected = torch.tensor(
            [[[[0], [1]]]],
            device=device,
            dtype=torch.int32,
        )
        num_blocks = torch.full(
            selected.shape[:-1], top_k, device=device, dtype=torch.int32
        )
        zero_num_blocks = torch.zeros_like(num_blocks)
        zero_indices = torch.zeros(
            *selected.shape[:-1], kv_blocks, device=device, dtype=torch.int32
        )
        sparse_indices = torch.zeros_like(zero_indices)
        sparse_indices[..., :top_k] = selected

        block_map = torch.zeros(
            batch, heads, q_blocks, kv_blocks, device=device, dtype=torch.bool
        )
        block_map.scatter_(-1, selected.long(), True)
        dense_mask = block_map.repeat_interleave(q_block_size, -2).repeat_interleave(
            kv_block_size, -1
        )
        ref = (
            torch.softmax(
                (q @ k.transpose(-2, -1) / math.sqrt(head_dim)).masked_fill(
                    ~dense_mask, -float("inf")
                ),
                dim=-1,
            )
            @ v
        )

        block_masks = {
            "full": BlockMask.from_kv_blocks(
                kv_num_blocks=zero_num_blocks,
                kv_indices=zero_indices,
                full_kv_num_blocks=num_blocks,
                full_kv_indices=sparse_indices,
                BLOCK_SIZE=(q_block_size, kv_block_size),
                seq_lengths=(q_blocks * q_block_size, kv_blocks * kv_block_size),
            ),
            "partial": BlockMask.from_kv_blocks(
                kv_num_blocks=num_blocks,
                kv_indices=sparse_indices,
                BLOCK_SIZE=(q_block_size, kv_block_size),
                seq_lengths=(q_blocks * q_block_size, kv_blocks * kv_block_size),
            ),
        }
        flash = torch.compile(
            functools.partial(flex_attention, kernel_options={"BACKEND": "FLASH"}),
            dynamic=False,
        )
        for name, block_mask in block_masks.items():
            with self.subTest(name=name):
                out = flash(q, k, v, block_mask=block_mask)
                torch.cuda.synchronize()
                self.assertEqual(out, ref, atol=5e-2, rtol=5e-2)

    @supported_platform
    @skip_on_cpu
    @skip_on_mps
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
    @skip_on_mps  # exercises Triton flex_decoding kernel; not present on MPS
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

    def test_unbacked_flex_decoding_eligibility_falls_back(self, device):
        from torch._inductor.kernel.flex.flex_decoding import _use_flex_decoding
        from torch._inductor.sizevars import SizeVarAllocator
        from torch._inductor.virtualized import V
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        class FakeBuffer:
            def __init__(self, size):
                self.size = size

            def get_size(self):
                return self.size

        shape_env = ShapeEnv()
        seq_len = shape_env.create_unbacked_symint().node.expr
        shape_env.var_to_hint_override[seq_len] = 64
        graph = mock.Mock(sizevars=SizeVarAllocator(shape_env))

        with V.set_graph_handler(graph):
            self.assertFalse(
                _use_flex_decoding(
                    FakeBuffer([1, 2, seq_len, 64]),
                    FakeBuffer([1, 1, 1, 1]),
                    FakeBuffer([1, 2, seq_len, 64]),
                    {},
                    False,
                )
            )

    def test_backward_fake_symbolic_query_key_batch_metadata(self, device):
        from torch._dynamo.source import LocalSource
        from torch._higher_order_ops.flex_attention import (
            _permute_strides,
            flex_attention_backward_fake_tensor_mode,
        )
        from torch._subclasses.fake_tensor import FakeTensorMode
        from torch.fx.experimental.symbolic_shapes import (
            DimDynamic,
            ShapeEnv,
            StatelessSymbolicContext,
        )

        shape_env = ShapeEnv()
        fake_mode = FakeTensorMode(allow_non_fake_inputs=True, shape_env=shape_env)

        def fakeify(t, name, shape_id=None):
            dynamic_sizes = [DimDynamic.STATIC] * t.dim()
            shape_ids = {}
            if shape_id is not None:
                dynamic_sizes[0] = DimDynamic.UNBACKED
                shape_ids[0] = shape_id
            return fake_mode.from_tensor(
                t,
                source=LocalSource(name, is_input=True),
                symbolic_context=StatelessSymbolicContext(
                    dynamic_sizes=dynamic_sizes,
                    shape_ids=shape_ids,
                ),
            )

        B, H, S, D = 4, 2, 8, 16

        def run(query, key, value, out, logsumexp, grad_out, grad_logsumexp):
            with fake_mode:
                _, grad_key, grad_value, _ = flex_attention_backward_fake_tensor_mode(
                    query,
                    key,
                    value,
                    out,
                    logsumexp,
                    grad_out,
                    grad_logsumexp,
                    None,
                    None,
                    (),
                    1.0,
                    {},
                )
            return grad_key, grad_value

        def make_fake_inputs(Bq, Bkv, *, shared_batch=False, key_value_factory=None):
            shape_id = "batch" if shared_batch else None
            query = fakeify(torch.empty(Bq, H, S, D, device=device), "query", shape_id)
            if key_value_factory is None:
                key_real = torch.empty(Bkv, H, S, D, device=device)
                value_real = torch.empty(Bkv, H, S, D, device=device)
            else:
                key_real = key_value_factory(Bkv)
                value_real = key_value_factory(Bkv)
            key = fakeify(key_real, "key", shape_id)
            value = fakeify(value_real, "value", shape_id)
            out = fakeify(torch.empty(Bq, H, S, D, device=device), "out", shape_id)
            grad_out = fakeify(
                torch.empty(Bq, H, S, D, device=device), "grad_out", shape_id
            )
            logsumexp = fakeify(
                torch.empty(Bq, H, S, device=device), "logsumexp", shape_id
            )
            grad_logsumexp = fakeify(
                torch.empty(Bq, H, S, device=device), "grad_logsumexp", shape_id
            )
            return (
                query,
                key,
                value,
                out,
                logsumexp,
                grad_out,
                grad_logsumexp,
                key_real,
                value_real,
            )

        args = make_fake_inputs(B, B, shared_batch=True)
        grad_key, grad_value = run(*args[:7])

        self.assertEqual(grad_key.shape, args[1].shape)
        self.assertEqual(grad_value.shape, args[2].shape)

        def transposed_batch_tensor(B):
            return torch.empty(H, S, D, B, device=device).permute(3, 0, 1, 2)

        args = make_fake_inputs(1, 1, key_value_factory=transposed_batch_tensor)
        grad_key, grad_value = run(*args[:7])
        key_real, value_real = args[7:]
        expected_key = _permute_strides(
            key_real.new_empty((1, H, S, D)), key_real.stride()
        )
        expected_value = _permute_strides(
            value_real.new_empty((1, H, S, D)), value_real.stride()
        )
        self.assertEqual(grad_key.shape, args[1].shape)
        self.assertEqual(grad_value.shape, args[2].shape)
        self.assertEqual(grad_key.stride(), expected_key.stride())
        self.assertEqual(grad_value.stride(), expected_value.stride())

        args = make_fake_inputs(B, 1, key_value_factory=transposed_batch_tensor)
        grad_key, grad_value = run(*args[:7])
        key_real, value_real = args[7:]
        expected_key = torch.sum(
            _permute_strides(key_real.new_empty((B, H, S, D)), key_real.stride()),
            dim=0,
            keepdim=True,
        )
        expected_value = torch.sum(
            _permute_strides(value_real.new_empty((B, H, S, D)), value_real.stride()),
            dim=0,
            keepdim=True,
        )
        self.assertEqual(grad_key.shape, args[1].shape)
        self.assertEqual(grad_value.shape, args[2].shape)
        self.assertEqual(grad_key.stride(), expected_key.stride())
        self.assertEqual(grad_value.stride(), expected_value.stride())

        args = make_fake_inputs(B, 2)
        with self.assertRaisesRegex(RuntimeError, "batch must match"):
            run(*args[:7])

    @supported_platform
    @skip_on_cpu
    @skip_on_mps  # asserts Triton-specific BACKEND='TRITON_DECODE' error
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
    @skip_on_mps  # asserts Triton-specific BACKEND='TRITON_DECODE' error
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
    @skip_on_cpu
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
    @expected_not_implemented_on_mps
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
    @skip_on_mps  # asserts Triton-specific "Buffers cannot be created" message
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
    @skip_on_mps  # asserts Triton-specific "Buffers cannot be created" message
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
    @expected_not_implemented_on_mps  # logspace out not implemented
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
    @skip_on_mps  # exercises the Triton default-config tile clamping path
    def test_narrow_kv_block_size_uses_clamped_default_config(self, device):
        """The single default config must clamp its tiles to fit a narrower KV
        sparse block size instead of erroring, in both forward and backward."""
        q, k, v = (
            torch.randn(
                2, 8, 2048, 128, device=device, dtype=torch.bfloat16, requires_grad=True
            )
            for _ in range(3)
        )
        block_mask = create_block_mask(
            _causal_mask, None, None, 2048, 2048, BLOCK_SIZE=(128, 64), device=device
        )
        out = torch.compile(flex_attention)(q, k, v, block_mask=block_mask)
        ref = flex_attention(q, k, v, block_mask=block_mask)
        torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)

        grad_out = torch.randn_like(out)
        grads = torch.autograd.grad(out, (q, k, v), grad_out, retain_graph=True)
        ref_grads = torch.autograd.grad(ref, (q, k, v), grad_out, retain_graph=True)
        for g, rg in zip(grads, ref_grads):
            torch.testing.assert_close(g, rg, atol=2e-2, rtol=2e-2)

    @supported_platform
    @skip_on_cpu
    @skip_on_mps  # exercises the Triton IS_DIVISIBLE sparse-block guard
    def test_sparse_block_not_dividing_seqlen(self, device):
        """Sparse blocks that overhang a 128-divisible seq len must not be
        walked unmasked past KV_LEN (silent OOB corruption)."""

        def tail_only(b, h, m, n):
            return n >= 256

        S = 384
        q, k, v = (
            torch.randn(2, 4, S, 64, device=device, requires_grad=True)
            for _ in range(3)
        )
        block_mask = create_block_mask(
            tail_only, None, None, S, S, BLOCK_SIZE=256, device=device
        )
        out = torch.compile(flex_attention)(q, k, v, block_mask=block_mask)
        ref = flex_attention(q, k, v, block_mask=block_mask)
        torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)

        grad_out = torch.randn_like(out)
        grads = torch.autograd.grad(out, (q, k, v), grad_out, retain_graph=True)
        ref_grads = torch.autograd.grad(ref, (q, k, v), grad_out, retain_graph=True)
        for g, rg in zip(grads, ref_grads):
            torch.testing.assert_close(g, rg, atol=2e-2, rtol=2e-2)

    @supported_platform
    @skip_on_cpu
    @skip_on_mps  # asserts Triton-specific BLOCK_M/BLOCK_N divisibility error
    def test_invalid_block_size(self, device):
        # Create tensors on different devices
        q, k, v = (torch.randn(1, 8, 128, 64, device=device) for _ in range(3))

        expected_error_message = (
            "Invalid FlexAttention forward kernel options: Q and KV block sizes "
            "must be divisible by the selected tile sizes.*"
            "SPARSE_Q_BLOCK_SIZE=96.*SPARSE_KV_BLOCK_SIZE=96.*"
            "BLOCK_M=\\d+.*BLOCK_N=\\d+"
        )
        block_mask = create_block_mask(
            noop_mask, 1, 8, 128, 128, BLOCK_SIZE=96, device=device
        )

        with self.assertRaisesRegex(RuntimeError, expected_error_message):
            torch.compile(flex_attention)(q, k, v, block_mask=block_mask)

    @supported_platform
    @skip_on_cpu
    @expected_not_implemented_on_mps  # no backward
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

        if not torch.equal(out_eager, out_compiled):
            raise AssertionError("out_eager and out_compiled are not equal")
        if not torch.equal(lse_eager, lse_compiled):
            raise AssertionError("lse_eager and lse_compiled are not equal")

        grads_eager = torch.autograd.grad(out_eager.sum(), (query, key, value))
        grads_compile = torch.autograd.grad(out_compiled.sum(), (query, key, value))

        torch.testing.assert_close(grads_eager, grads_compile)

    @supported_platform
    @skip_on_cpu
    @expected_not_implemented_on_mps  # no backward
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

        if counter.frame_count != 1:
            raise AssertionError(
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
    @skip_on_mps  # asserts Triton-specific "Query length must be greater than 0"
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
    @unittest.skipUnless(TEST_MULTIACCELERATOR, "detected only one GPU")
    def test_qkv_and_block_mask_on_the_same_device(self, device):
        make_tensor = functools.partial(
            torch.ones,
            (2, 2, 256, 32),
            device=0,
            dtype=torch.float32,
            requires_grad=True,
        )
        query, key, value = make_tensor(), make_tensor(), make_tensor()

        def mask_mod(b, h, q, kv):
            return q >= kv

        block_mask = create_block_mask(mask_mod, 1, 1, 256, 256, device=1)
        with self.assertRaisesRegex(
            RuntimeError, "Expect q/k/v and block_mask to be on the same device"
        ):
            torch.compile(flex_attention)(query, key, value, block_mask=block_mask)

    @supported_platform
    @skip_on_cpu
    @config.patch(
        {
            "triton.native_matmul": False,
        }
    )
    @expected_not_implemented_on_mps  # no backward yet
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
    @expected_not_implemented_on_mps
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
    @expected_not_implemented_on_mps  # backward path; NIE on MPS via _validate_device
    def test_fw_bw_graph_correctness(self, device):
        cnt = CompileCounterWithBackend("aot_eager")
        # MPS lacks fp64; the NIE fires before any assertion would care.
        dtype = torch.float32 if _is_mps_device(device) else torch.float64
        make_tensor = functools.partial(
            torch.randn,
            (2, 2, 128, 4),
            device=device,
            dtype=dtype,
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
    def forward(self, L_block_mask_full_kv_indices: "i32[1, 1, 1, 1]", L_block_mask_full_kv_num_blocks: "i32[1, 1, 1]", L_block_mask_full_q_indices: "i32[1, 1, 1, 1]", L_block_mask_full_q_num_blocks: "i32[1, 1, 1]", L_block_mask_kv_indices: "i32[1, 1, 1, 1]", L_block_mask_kv_num_blocks: "i32[1, 1, 1]", L_block_mask_q_indices: "i32[1, 1, 1, 1]", L_block_mask_q_num_blocks: "i32[1, 1, 1]", L_key_: "f64[2, 2, 128, 4]", L_query_: "f64[2, 2, 128, 4]", L_value_: "f64[2, 2, 128, 4]"):
        l_block_mask_full_kv_indices = L_block_mask_full_kv_indices
        l_block_mask_full_kv_num_blocks = L_block_mask_full_kv_num_blocks
        l_block_mask_full_q_indices = L_block_mask_full_q_indices
        l_block_mask_full_q_num_blocks = L_block_mask_full_q_num_blocks
        l_block_mask_kv_indices = L_block_mask_kv_indices
        l_block_mask_kv_num_blocks = L_block_mask_kv_num_blocks
        l_block_mask_q_indices = L_block_mask_q_indices
        l_block_mask_q_num_blocks = L_block_mask_q_num_blocks
        l_key_ = L_key_
        l_query_ = L_query_
        l_value_ = L_value_

        mask_fn_0 = self.mask_fn_0
        score_mod_0 = self.score_mod_0
        flex_attention = torch.ops.higher_order.flex_attention(l_query_, l_key_, l_value_, score_mod_0, (128, 128, l_block_mask_kv_num_blocks, l_block_mask_kv_indices, l_block_mask_full_kv_num_blocks, l_block_mask_full_kv_indices, l_block_mask_q_num_blocks, l_block_mask_q_indices, l_block_mask_full_q_num_blocks, l_block_mask_full_q_indices, None, None, None, None, 128, 128, mask_fn_0), 0.5, {'BACKEND': 'AUTO', 'PRESCALE_QK': False, 'ROWS_GUARANTEED_SAFE': False, 'BLOCKS_ARE_CONTIGUOUS': False, 'WRITE_DQ': True, 'OUTPUT_LOGSUMEXP': True, 'OUTPUT_MAX': False}, (), ());  l_query_ = l_key_ = l_value_ = score_mod_0 = l_block_mask_kv_num_blocks = l_block_mask_kv_indices = l_block_mask_full_kv_num_blocks = l_block_mask_full_kv_indices = l_block_mask_q_num_blocks = l_block_mask_q_indices = l_block_mask_full_q_num_blocks = l_block_mask_full_q_indices = mask_fn_0 = None
        getitem: "f64[2, 2, 128, 4]" = flex_attention[0];  flex_attention = None
        return (getitem,)

    class mask_fn_0(torch.nn.Module):
        def forward(self, child: "i32[]", child_1: "i32[]", child_2: "i32[]", child_3: "i32[]"):
            ge: "b8[]" = child_2 >= child_3;  child_2 = child_3 = None
            return ge

    class score_mod_0(torch.nn.Module):
        def forward(self, child: "f64[]", child_1: "i32[]", child_2: "i32[]", child_3: "i32[]", child_4: "i32[]"):
            mul: "f64[]" = child * child;  child = None
            return mul
""",
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
    def forward(self, primals_1: "f64[2, 2, 128, 4]", primals_2: "f64[2, 2, 128, 4]", primals_3: "f64[2, 2, 128, 4]", full: "i32[1, 1, 1]", full_default: "i32[1, 1, 1, 1]", convert_element_type: "i32[1, 1, 1, 1]", convert_element_type_1: "i32[1, 1, 1]", getitem_2: "f64[2, 2, 128, 4]", getitem_3: "f32[2, 2, 128]", tangents_1: "f64[2, 2, 128, 4]"):
        fw_graph0 = self.fw_graph0
        joint_graph0 = self.joint_graph0
        mask_graph0 = self.mask_graph0
        flex_attention_backward = torch.ops.higher_order.flex_attention_backward(primals_2, primals_1, primals_3, getitem_2, getitem_3, tangents_1, None, fw_graph0, joint_graph0, (1, 1, full, full_default, None, None, convert_element_type_1, convert_element_type, None, None, None, None, None, None, 1073741824, 1073741824, mask_graph0), 0.5, {'BACKEND': 'AUTO', 'PRESCALE_QK': False, 'ROWS_GUARANTEED_SAFE': False, 'BLOCKS_ARE_CONTIGUOUS': False, 'WRITE_DQ': True, 'OUTPUT_LOGSUMEXP': True, 'OUTPUT_MAX': False}, (), ());  primals_2 = primals_1 = primals_3 = getitem_2 = getitem_3 = tangents_1 = fw_graph0 = joint_graph0 = full = full_default = convert_element_type_1 = convert_element_type = mask_graph0 = None
        getitem_5: "f64[2, 2, 128, 4]" = flex_attention_backward[0]
        getitem_6: "f64[2, 2, 128, 4]" = flex_attention_backward[1]
        getitem_7: "f64[2, 2, 128, 4]" = flex_attention_backward[2];  flex_attention_backward = None
        return (getitem_6, getitem_5, getitem_7)

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
""".replace("GPU_TYPE", torch.device(device).type),
        )

    @supported_platform
    @skip_on_cpu
    @expected_not_implemented_on_mps  # backward path; NIE on MPS via _validate_device
    def test_backward_calls_flex_attention_backward_choices_hook(self, device):
        from torch._inductor.choices import InductorChoices
        from torch._inductor.virtualized import V

        class ForwardChoices(InductorChoices):
            def __init__(self):
                self.calls = 0

            def append_flex_attention_choices(
                self,
                choices,
                configs,
                input_nodes,
                subgraphs,
                layout,
                kernel_options,
                sparse_q_block_size,
                sparse_kv_block_size,
            ):
                self.calls += 1
                return choices

        class SelectedChoice:
            def __init__(self, choice, owner):
                self.choice = choice
                self.owner = owner

            def __getattr__(self, name):
                return getattr(self.choice, name)

            def output_node(self):
                self.owner.selected = True
                return self.choice.output_node()

        class RecordingChoices(InductorChoices):
            def __init__(self):
                self.calls = []
                self.selected = False
                self.stock_input_nodes = None
                self.stock_mutated_inputs = None

            def append_flex_attention_backward_choices(
                self,
                choices,
                configs,
                input_nodes,
                subgraphs,
                layout,
                kernel_options,
                sparse_q_block_size,
                sparse_kv_block_size,
                *,
                mutated_inputs,
            ):
                self.calls.append((input_nodes, subgraphs, mutated_inputs))
                stock_choice = choices[0]
                self.stock_input_nodes = stock_choice.input_nodes
                self.stock_mutated_inputs = stock_choice.mutated_inputs
                choices.clear()
                choices.append(SelectedChoice(stock_choice, self))
                return choices

        class MutatingChoices(InductorChoices):
            def __init__(self):
                self.stock_input_node_count = None
                self.stock_mutated_input_count = None

            def append_flex_attention_backward_choices(
                self,
                choices,
                configs,
                input_nodes,
                subgraphs,
                layout,
                kernel_options,
                sparse_q_block_size,
                sparse_kv_block_size,
                *,
                mutated_inputs,
            ):
                input_nodes.clear()
                subgraphs.clear()
                mutated_inputs.clear()
                self.stock_input_node_count = len(choices[0].input_nodes)
                self.stock_mutated_input_count = len(choices[0].mutated_inputs)
                return choices

        shape = (1, 1, 128, 16)
        query = torch.randn(shape, device=device)
        key = torch.randn(shape, device=device)
        value = torch.randn(shape, device=device)
        block_mask = _create_empty_block_mask(query, key)
        scale = 1.0 / shape[-1] ** 0.5
        out, logsumexp = flex_attention_fwd(
            query,
            key,
            value,
            _identity,
            block_mask,
            scale,
        )
        grad_out = torch.randn_like(out)

        def compiled_bw(query, key, value, out, logsumexp, grad_out):
            return torch.ops.higher_order.flex_attention_backward(
                query,
                key,
                value,
                out,
                logsumexp,
                grad_out,
                None,
                _identity,
                None,
                block_mask.as_tuple(),
                scale,
                {"BACKEND": "TRITON"},
                (),
                (),
            )

        def run_with_choices(choices):
            torch._dynamo.reset()
            with (
                config.patch(force_disable_caches=True),
                V.set_choices_handler(choices),
                torch.no_grad(),
            ):
                return torch.compile(compiled_bw, fullgraph=True)(
                    query, key, value, out, logsumexp, grad_out
                )

        forward_choices = ForwardChoices()
        stock_result = run_with_choices(forward_choices)
        self.assertEqual(forward_choices.calls, 0)

        recording_choices = RecordingChoices()
        recording_result = run_with_choices(recording_choices)

        self.assertEqual(len(recording_choices.calls), 1)
        self.assertTrue(recording_choices.selected)
        self.assertEqual(recording_result, stock_result)
        input_nodes, subgraphs, mutated_inputs = recording_choices.calls[0]
        self.assertEqual(len(input_nodes), 16)
        self.assertEqual(len(subgraphs), 4)
        self.assertEqual(len(mutated_inputs), 2)
        self.assertEqual(len(recording_choices.stock_input_nodes), len(input_nodes))
        for stock_node, hook_node in zip(
            recording_choices.stock_input_nodes, input_nodes
        ):
            self.assertIs(hook_node, stock_node)
        self.assertEqual(
            len(recording_choices.stock_mutated_inputs), len(mutated_inputs)
        )
        for stock_node, hook_node in zip(
            recording_choices.stock_mutated_inputs, mutated_inputs
        ):
            self.assertIs(hook_node, stock_node)
        self.assertIs(mutated_inputs[0], input_nodes[6])
        self.assertIs(mutated_inputs[1], input_nodes[7])

        mutating_choices = MutatingChoices()
        mutating_result = run_with_choices(mutating_choices)
        self.assertEqual(mutating_result, stock_result)
        self.assertEqual(mutating_choices.stock_input_node_count, 16)
        self.assertEqual(mutating_choices.stock_mutated_input_count, 2)

    @supported_platform
    @skip_on_cpu
    @expected_not_implemented_on_mps  # backward path; NIE on MPS via _validate_device
    def test_backward_choices_hook_preserves_captured_grad_mutations(self, device):
        from torch._inductor.choices import InductorChoices
        from torch._inductor.virtualized import V

        class RecordingChoices(InductorChoices):
            def __init__(self):
                self.calls = []

            def uuid(self):
                return "captured_grad_mutation_test"

            def append_flex_attention_backward_choices(
                self,
                choices,
                configs,
                input_nodes,
                subgraphs,
                layout,
                kernel_options,
                sparse_q_block_size,
                sparse_kv_block_size,
                *,
                mutated_inputs,
            ):
                self.calls.append((input_nodes, subgraphs, mutated_inputs))
                return choices

        shape = (1, 1, 128, 16)
        query = torch.randn(shape, device=device, requires_grad=True)
        key = torch.randn(shape, device=device, requires_grad=True)
        value = torch.randn(shape, device=device, requires_grad=True)
        score_bias = torch.randn((1,), device=device, requires_grad=True)

        def compiled_attention(query, key, value, bias):
            def score_mod(score, b, h, m, n):
                return score * bias[0]

            return flex_attention(
                query,
                key,
                value,
                score_mod=score_mod,
                kernel_options={"BACKEND": "TRITON"},
            )

        choices = RecordingChoices()
        torch._dynamo.reset()
        with (
            config.patch(
                force_disable_caches=True,
                inductor_choices_class=lambda: choices,
            ),
            V.set_choices_handler(choices),
        ):
            torch.compile(compiled_attention, fullgraph=True)(
                query, key, value, score_bias
            ).sum().backward()

        backward_calls = [call for call in choices.calls if len(call[0]) == 16]
        self.assertEqual(len(backward_calls), 1)
        input_nodes, subgraphs, mutated_inputs = backward_calls[0]
        self.assertEqual(len(subgraphs), 4)
        self.assertEqual(len(mutated_inputs), 3)
        self.assertIs(mutated_inputs[0], input_nodes[6])
        self.assertIs(mutated_inputs[1], input_nodes[7])
        self.assertEqual(mutated_inputs[2].get_size(), [1])

    @supported_platform
    @skip_on_cpu
    @expected_not_implemented_on_mps  # backward path; NIE on MPS via _validate_device
    def test_direct_backward_preserves_explicit_buffers(self, device):
        mask_buffer = torch.full((), 128, device=device, dtype=torch.int32)

        def score_mod(score, b, h, m, n):
            return score

        def mask_mod(b, h, m, n):
            return m + mask_buffer >= n

        block_mask = create_block_mask(
            mask_mod,
            B=2,
            H=2,
            Q_LEN=128,
            KV_LEN=128,
            device=device,
        )
        scale = 1.0 / 16**0.5

        dtype = torch.float32
        q = torch.randn(
            (2, 2, 128, 16),
            dtype=dtype,
            device=device,
            requires_grad=True,
        )
        k = torch.randn(
            (2, 2, 128, 16),
            dtype=dtype,
            device=device,
            requires_grad=True,
        )
        v = torch.randn(
            (2, 2, 128, 16),
            dtype=dtype,
            device=device,
            requires_grad=True,
        )
        q_ref, k_ref, v_ref = query_key_value_clones(q, k, v)
        q_gold, k_gold, v_gold = query_key_value_clones(q, k, v, torch.float64)

        sdpa_partial = create_attention(score_mod, block_mask)
        golden_out = sdpa_partial(q_gold, k_gold, v_gold)
        ref_out = sdpa_partial(q_ref, k_ref, v_ref)

        backward_grad = torch.randn((2, 2, 128, 16), dtype=dtype, device=device)
        golden_out.backward(backward_grad.to(torch.float64))
        ref_out.backward(backward_grad)

        out, logsumexp = flex_attention_fwd(
            q,
            k,
            v,
            score_mod,
            block_mask,
            scale,
        )

        @torch.compile(fullgraph=True)
        def compiled_bw(query, key, value, fwd_out, lse, grad_out):
            return torch.ops.higher_order.flex_attention_backward(
                query,
                key,
                value,
                fwd_out,
                lse,
                grad_out,
                None,
                score_mod,
                None,
                block_mask.as_tuple(),
                scale,
                {},
                (),
                (),
            )

        with torch.no_grad():
            dq, dk, dv, _ = compiled_bw(
                q,
                k,
                v,
                out.detach(),
                logsumexp.detach(),
                backward_grad,
            )

        fudge_factor = 10.0
        self._check_equal(q_gold.grad, q_ref.grad, dq, fudge_factor, "Grad_Query")
        self._check_equal(k_gold.grad, k_ref.grad, dk, fudge_factor, "Grad_Key")
        self._check_equal(v_gold.grad, v_ref.grad, dv, fudge_factor, "Grad_Value")

    @supported_platform
    @skip_on_cpu
    def test_direct_backward_supports_symint_score_mod_buffers(self, device):
        def score_mod(score, b, h, m, n, head_dim):
            return score

        def mask_mod(b, h, m, n):
            return m >= n

        block_mask = create_block_mask(
            mask_mod,
            B=2,
            H=2,
            Q_LEN=128,
            KV_LEN=128,
            device=device,
        )
        scale = 1.0 / 16**0.5
        dtype = torch.float32
        q = torch.randn((2, 2, 128, 16), dtype=dtype, device=device)
        k = torch.randn((2, 2, 128, 16), dtype=dtype, device=device)
        v = torch.randn((2, 2, 128, 16), dtype=dtype, device=device)
        backward_grad = torch.randn((2, 2, 128, 16), dtype=dtype, device=device)

        out, logsumexp = flex_attention_fwd(
            q,
            k,
            v,
            _identity,
            block_mask,
            scale,
        )
        static_head_dim = q.shape[-1]

        @torch.compile(backend="aot_eager", fullgraph=True)
        def compiled_literal_bw(query, key, value, fwd_out, lse, grad_out):
            return torch.ops.higher_order.flex_attention_backward(
                query,
                key,
                value,
                fwd_out,
                lse,
                grad_out,
                None,
                score_mod,
                None,
                block_mask.as_tuple(),
                scale,
                {},
                (static_head_dim,),
                (),
            )

        @torch.compile(backend="aot_eager", fullgraph=True, dynamic=True)
        def compiled_bw(query, key, value, fwd_out, lse, grad_out):
            return torch.ops.higher_order.flex_attention_backward(
                query,
                key,
                value,
                fwd_out,
                lse,
                grad_out,
                None,
                score_mod,
                None,
                block_mask.as_tuple(),
                scale,
                {},
                (query.shape[-1],),
                (),
            )

        with torch.no_grad():
            ref_dq, ref_dk, ref_dv, ref_buffer_grads = compiled_literal_bw(
                q,
                k,
                v,
                out.detach(),
                logsumexp.detach(),
                backward_grad,
            )
            compiled_dq, compiled_dk, compiled_dv, compiled_buffer_grads = compiled_bw(
                q,
                k,
                v,
                out.detach(),
                logsumexp.detach(),
                backward_grad,
            )

        torch.testing.assert_close(compiled_dq, ref_dq)
        torch.testing.assert_close(compiled_dk, ref_dk)
        torch.testing.assert_close(compiled_dv, ref_dv)
        self.assertEqual(compiled_buffer_grads, ref_buffer_grads)

    @supported_platform
    @skip_on_cpu
    @expected_not_implemented_on_mps  # backward Triton lowering
    def test_direct_backward_inductor_batch_broadcast_matches_eager(self, device):
        def score_mod(score, b, h, m, n):
            return score

        def mask_mod(b, h, m, n):
            return m >= n

        block_mask = create_block_mask(
            mask_mod,
            B=4,
            H=None,
            Q_LEN=128,
            KV_LEN=128,
            device=device,
        )
        scale = 1.0 / 16**0.5
        dtype = torch.float32
        q = torch.randn((4, 2, 128, 16), dtype=dtype, device=device, requires_grad=True)
        k = torch.randn((1, 2, 128, 16), dtype=dtype, device=device, requires_grad=True)
        v = torch.randn((1, 2, 128, 16), dtype=dtype, device=device, requires_grad=True)
        q_ref, k_ref, v_ref = query_key_value_clones(q, k, v)
        q_gold, k_gold, v_gold = query_key_value_clones(q, k, v, torch.float64)

        sdpa_partial = create_attention(score_mod, block_mask)
        golden_out = sdpa_partial(q_gold, k_gold, v_gold)
        ref_out = sdpa_partial(q_ref, k_ref, v_ref)
        backward_grad = torch.randn_like(q)
        golden_out.backward(backward_grad.to(torch.float64))
        ref_out.backward(backward_grad)

        out, logsumexp = flex_attention_fwd(
            q,
            k,
            v,
            _identity,
            block_mask,
            scale,
        )

        def eager_bw(query, key, value, fwd_out, lse, grad_out):
            return torch.ops.higher_order.flex_attention_backward(
                query,
                key,
                value,
                fwd_out,
                lse,
                grad_out,
                None,
                score_mod,
                None,
                block_mask.as_tuple(),
                scale,
                {"BACKEND": "TRITON"},
                (),
                (),
            )

        @torch.compile(fullgraph=True)
        def compiled_bw(query, key, value, fwd_out, lse, grad_out):
            return eager_bw(query, key, value, fwd_out, lse, grad_out)

        with torch.no_grad():
            _, ref_dk, ref_dv, _ = eager_bw(
                q,
                k,
                v,
                out.detach(),
                logsumexp.detach(),
                backward_grad,
            )
            dq, dk, dv, _ = compiled_bw(
                q,
                k,
                v,
                out.detach(),
                logsumexp.detach(),
                backward_grad,
            )

        fudge_factor = 10.0
        self._check_equal(q_gold.grad, q_ref.grad, dq, fudge_factor, "Grad_Query")
        self._check_equal(k_gold.grad, k_ref.grad, dk, fudge_factor, "Grad_Key")
        self._check_equal(v_gold.grad, v_ref.grad, dv, fudge_factor, "Grad_Value")
        self.assertEqual(dk.shape, k.shape)
        self.assertEqual(dv.shape, v.shape)
        self.assertEqual(dk.stride(), ref_dk.stride())
        self.assertEqual(dv.stride(), ref_dv.stride())

    @supported_platform
    @skip_on_cpu
    @expected_not_implemented_on_mps  # backward Triton lowering
    def test_direct_backward_inductor_supports_scalar_mask_mod_buffers(self, device):
        def score_mod(score, b, h, m, n):
            return score

        def base_mask_mod(b, h, m, n):
            return m >= n

        def mask_mod(b, h, m, n, shift):
            return (m + shift) >= n

        base_block_mask = create_block_mask(
            base_mask_mod,
            B=2,
            H=2,
            Q_LEN=128,
            KV_LEN=128,
            device=device,
        )
        block_mask = BlockMask.from_kv_blocks(
            base_block_mask.kv_num_blocks,
            base_block_mask.kv_indices,
            base_block_mask.full_kv_num_blocks,
            base_block_mask.full_kv_indices,
            BLOCK_SIZE=base_block_mask.BLOCK_SIZE,
            mask_mod=mask_mod,
            seq_lengths=base_block_mask.seq_lengths,
        )
        scale = 1.0 / 16**0.5
        dtype = torch.float32
        q = torch.randn((2, 2, 128, 16), dtype=dtype, device=device, requires_grad=True)
        k = torch.randn((2, 2, 128, 16), dtype=dtype, device=device, requires_grad=True)
        v = torch.randn((2, 2, 128, 16), dtype=dtype, device=device, requires_grad=True)
        q_ref, k_ref, v_ref = query_key_value_clones(q, k, v)
        q_gold, k_gold, v_gold = query_key_value_clones(q, k, v, torch.float64)

        ref_attention = create_attention(score_mod, base_block_mask)
        golden_out = ref_attention(q_gold, k_gold, v_gold)
        ref_out = ref_attention(q_ref, k_ref, v_ref)
        backward_grad = torch.randn((2, 2, 128, 16), dtype=dtype, device=device)
        golden_out.backward(backward_grad.to(torch.float64))
        ref_out.backward(backward_grad)
        static_shift = 0

        out, logsumexp, _ = flex_attention_hop(
            q,
            k,
            v,
            _identity,
            block_mask.as_tuple(),
            scale,
            {},
            (),
            (static_shift,),
        )

        def eager_bw(query, key, value, fwd_out, lse, grad_out):
            return torch.ops.higher_order.flex_attention_backward(
                query,
                key,
                value,
                fwd_out,
                lse,
                grad_out,
                None,
                score_mod,
                None,
                block_mask.as_tuple(),
                scale,
                {"BACKEND": "TRITON"},
                (),
                (static_shift,),
            )

        @torch.compile(fullgraph=True)
        def compiled_bw(query, key, value, fwd_out, lse, grad_out):
            return eager_bw(query, key, value, fwd_out, lse, grad_out)

        with torch.no_grad():
            dq, dk, dv, buffer_grads = compiled_bw(
                q,
                k,
                v,
                out.detach(),
                logsumexp.detach(),
                backward_grad,
            )

        fudge_factor = 10.0
        self._check_equal(q_gold.grad, q_ref.grad, dq, fudge_factor, "Grad_Query")
        self._check_equal(k_gold.grad, k_ref.grad, dk, fudge_factor, "Grad_Key")
        self._check_equal(v_gold.grad, v_ref.grad, dv, fudge_factor, "Grad_Value")
        self.assertEqual(buffer_grads, ())

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
                if not isinstance(elem, torch.Tensor):
                    raise AssertionError(f"Expected torch.Tensor, got {type(elem)}")
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
                if meta is not None:
                    raise AssertionError(f"Expected meta to be None, got {meta}")
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
    @skip_on_xpu
    def test_cpu_qk_chunk_same_addmm_buffer(self, device):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.H = 2
                self.D = 16
                self.S = 128
                self.project_qk = nn.Linear(self.H * self.D, self.H * self.D * 2)
                self.project_v = nn.Linear(self.H * self.D, self.H * self.D)

            def forward(self, hidden_states):
                B = hidden_states.size(0)
                qk = self.project_qk(hidden_states)
                query, key = qk.chunk(2, dim=-1)

                query = query.view(B, self.S, self.H, self.D).permute(0, 2, 1, 3)
                key = key.view(B, self.S, self.H, self.D).permute(0, 2, 1, 3)
                value = (
                    self.project_v(hidden_states)
                    .view(B, self.S, self.H, self.D)
                    .permute(0, 2, 1, 3)
                )

                return flex_attention(query, key, value, score_mod=_identity)

        torch.manual_seed(0)
        x = torch.randn(1, 128, 32, device=device)
        model = Model().to(device).eval()

        with torch.no_grad():
            eager_out = model(x)
            compiled_out = torch.compile(model)(x)

        torch.testing.assert_close(compiled_out, eager_out, rtol=1e-4, atol=1e-4)

    @supported_platform
    @skip_on_cuda
    @skip_on_xpu
    @skip_on_mps  # asserts on the CPU C++ template's generated code
    @common_utils.parametrize("qkv_source", ["input", "bshd_buffer", "linear"])
    @common_utils.parametrize("q_len", [1, 256])
    def test_cpu_flex_template_choice_qkv_views(self, device, qkv_source, q_len):
        # q/k/v may be views over buffers whose shape differs from the logical
        # (B, H, S, D); template choice must use the logical sizes.
        B, H, S_kv, D = 1, 4, 1024, 64
        proj = nn.ModuleList(nn.Linear(H * D, H * D) for _ in range(3)).eval()

        def to_bhsd(x, i):
            if qkv_source == "input":
                return x
            if qkv_source == "bshd_buffer":
                return x.sin().transpose(1, 2)
            return proj[i](x).view(B, x.size(1), H, D).transpose(1, 2)

        def fn(q, k, v, block_mask):
            q, k, v = (to_bhsd(t, i) for i, t in enumerate((q, k, v)))
            return flex_attention(q, k, v, block_mask=block_mask)

        def make(S):
            if qkv_source == "input":
                return torch.randn(B, H, S, D, device=device)
            if qkv_source == "bshd_buffer":
                return torch.randn(B, S, H, D, device=device)
            return torch.randn(B, S, H * D, device=device)

        q, k, v = make(q_len), make(S_kv), make(S_kv)
        block_mask = create_block_mask(noop_mask, B, 1, q_len, S_kv, device=device)

        with torch.no_grad():
            eager_out = fn(q, k, v, block_mask)
            compiled_out, code = run_and_get_code(
                torch.compile(fn), q, k, v, block_mask
            )

        torch.testing.assert_close(compiled_out, eager_out, rtol=1e-4, atol=1e-4)
        # PARTITION_SIZE only appears in the FLEX_DECODING template.
        if q_len == 1:
            FileCheck().check("PARTITION_SIZE").run(code[0])
        else:
            FileCheck().check_not("PARTITION_SIZE").run(code[0])

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

    @supported_platform
    @skip_on_cpu
    @skip_on_xpu
    def test_nested_tensor_inputs_error(self, device):
        def make_nt(lengths, heads=2, dim=8):
            elems = [
                torch.randn(s, heads, dim, device=device, dtype=torch.float32)
                for s in lengths
            ]
            return torch.nested.nested_tensor(elems, layout=torch.jagged).transpose(
                1, 2
            )

        q = make_nt([3, 2])
        k = make_nt([3, 2])
        v = make_nt([3, 2])
        msg = "flex_attention does not support NestedTensor inputs"

        with self.assertRaisesRegex(NotImplementedError, msg):
            flex_attention(q, k, v)

        compiled_flex = torch.compile(flex_attention, fullgraph=True)
        with self.assertRaisesRegex(torch._dynamo.exc.Unsupported, msg):
            compiled_flex(q, k, v)

    @unittest.skipUnless(TEST_MULTIACCELERATOR, "detected only one GPU")
    def test_device_cuda_1(self, device):
        class TestModule(torch.nn.Module):
            def forward(self, q, k, v, block_mask):
                return flex_attention(q, k, v, block_mask=block_mask)

        q = torch.randn(1, 1, 256, 32, device=1, dtype=torch.bfloat16)
        k = torch.randn(1, 1, 256, 32, device=1, dtype=torch.bfloat16)
        v = torch.randn(1, 1, 256, 32, device=1, dtype=torch.bfloat16)
        mask = create_block_mask(
            lambda b, h, q_idx, kv_idx: q_idx >= kv_idx,
            B=None,
            H=None,
            Q_LEN=256,
            KV_LEN=256,
            device=1,
        )
        mod = torch.compile(TestModule())
        attn_output = mod(q, k, v, mask)
        self.assertEqual(attn_output.device, torch.device(1))

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
                if S * S != P:
                    raise AssertionError(f"Expected S * S == P, got {S * S} != {P}")
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
    @expected_not_implemented_on_mps  # no backward yet
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
    @skip_on_mps  # uses mode="max-autotune-no-cudagraphs"; Triton/CUDA-only
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
    @skip_on_mps  # asserts Triton-specific NYI "embedding dimension" message
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
        if kernel_code is None:
            raise AssertionError("Failed to retrieve compiled kernel code")
        if "num_consumer_groups" not in kernel_code[0]:
            raise AssertionError("num_consumer_groups missing in kernel definition")
        if "num_buffers_warp_spec" not in kernel_code[0]:
            raise AssertionError("num_buffers_warp_spec missing in kernel definition")

        # Validate correctness
        C1 = flex_compiled(q, k, v)
        C2 = flex_attention(q, k, v)

        if not torch.allclose(C1, C2, atol=1e-2, rtol=1e-2):
            raise AssertionError(
                "Warp specialized kernel result differs from reference"
            )

    @supported_platform
    @skip_on_cpu
    @skip_on_mps  # pins kernel_options={"BACKEND": ...}
    # Scope is NVIDIA CUDA and XPU. ROCm is excluded explicitly rather than by
    # has_triton_tma_device(), whose HIP exclusion sits only in its CUDA arm:
    # its XPU and Triton-CPU-backend alternatives can make the aggregate true on
    # a ROCm host, and gfx1250 also reports device type "cuda" while defaulting
    # the option on. TestFlexAttentionTDMEndToEnd covers that backend.
    @skipIfRocm
    @skipCUDAIf(not has_triton_tma_device(), "Requires TMA enabled CUDA device")
    @common_utils.parametrize("backend", ["TRITON", "TRITON_DECODE"])
    def test_tma_with_customer_kernel_options_selection(self, device, backend):
        """Descriptor selection follows USE_TMA on the TMA backends.

        test_tma_with_customer_kernel_options only compares numbers, so it still
        passes when both requests land on the same implementation. This asserts
        the generated code instead, and lives outside the gfx1250-gated class so
        the shared USE_TMA contract keeps CI coverage on CUDA and XPU. The name
        shares that test's prefix so the H100 smoke suite's -k filter selects
        both; keep them together if either is renamed.
        """
        # Operands are deliberately TMA-eligible: contiguous bf16 with a
        # 256-byte innermost row, so only the option decides the path.
        operands = ("q", "k", "v") if backend == "TRITON" else ("k", "v")
        make_tensor = functools.partial(
            torch.ones, (1, 1, 256, 128), device=device, dtype=torch.bfloat16
        )
        query = (
            make_tensor()
            if backend == "TRITON"
            else torch.ones((1, 1, 1, 128), device=device, dtype=torch.bfloat16)
        )
        key, value = make_tensor(), make_tensor()

        # Omission is backend-dependent: Intel GPUs default the option on, and
        # CUDA leaves it off. Everything else is explicit. This keys on the
        # device rather than has_triton_tma_device(), which is a single flag
        # covering both backends and so cannot express the differing defaults.
        xpu_defaults_on = device.split(":")[0] == "xpu"
        cases = [
            ({}, xpu_defaults_on),
            ({"USE_TMA": False}, False),
            ({"USE_TMA": True}, True),
            # Falsy non-False values are not runtime-validated and have always
            # taken the pointer path.
            ({"USE_TMA": 0}, False),
            ({"USE_TMA": None}, False),
            # fwd_ wins over the unprefixed key for forward and decode alike.
            ({"USE_TMA": True, "fwd_USE_TMA": False}, False),
        ]

        for options, expect_descriptors in cases:
            with self.subTest(options=options):
                kernel_options = dict(options, BACKEND=backend)

                def fn(q, k, v, kernel_options=kernel_options):
                    return flex_attention(q, k, v, kernel_options=kernel_options)

                torch._dynamo.reset()
                _, code = run_and_get_code(torch.compile(fn), query, key, value)
                joined = "\n".join(code)
                for name in operands:
                    if expect_descriptors:
                        self.assertRegex(joined, _descriptor_load_pattern(name))
                    else:
                        self.assertRegex(joined, _pointer_load_pattern(name))
                        self.assertNotRegex(joined, _descriptor_load_pattern(name))

    @supported_platform
    @skip_on_cpu
    @skip_on_mps  # pins kernel_options={"BACKEND": ...}
    # Same scope as the selection test above, and ROCm is excluded for the same
    # reason: TestFlexAttentionTDMEndToEnd holds the gfx1250 copy of this check.
    @skipIfRocm
    @skipCUDAIf(not has_triton_tma_device(), "Requires TMA enabled CUDA device")
    def test_tma_with_customer_kernel_options_int64_decode(self, device):
        """Decode descriptor coordinates stay int32 when indexing is int64.

        The narrowing lives in the shared common.py.jinja, so the bug this
        covers is reachable on any descriptor backend, but the gfx1250 copy of
        the check sits behind a hardware skip and never runs in CI. The name
        shares test_tma_with_customer_kernel_options' prefix so the H100 smoke
        suite's -k filter selects it; keep them together if either is renamed.
        """
        # Random rather than the neighbours' ones(): with a constant fill every
        # row reads alike, so a wrongly narrowed coordinate still produces the
        # reference numbers and only the codegen assertions would catch it.
        make_tensor = functools.partial(
            torch.randn, (1, 1, 256, 128), device=device, dtype=torch.bfloat16
        )
        _check_int64_decode_narrows_coordinates(
            self,
            torch.randn((1, 1, 1, 128), device=device, dtype=torch.bfloat16),
            make_tensor(),
            make_tensor(),
            # Decode is requested explicitly: CUDA defaults descriptors off,
            # and a query length of 1 alone does not pin the decode kernel.
            kernel_options={"USE_TMA": True, "BACKEND": "TRITON_DECODE"},
            tolerances=_TDM_TOLERANCES[torch.bfloat16],
        )

    @supported_platform
    @skip_on_cpu
    @skipCUDAIf(not has_triton_tma_device(), "Requires TMA enabled CUDA device")
    def test_tma_with_customer_kernel_options(self, device):
        requires_grad = device in DEVICE_SUPPORTS_BACKWARDS
        make_tensor = functools.partial(
            torch.ones,
            (1, 1, 256, 128),
            device=device,
            dtype=torch.bfloat16,
            requires_grad=requires_grad,
        )
        query, key, value = make_tensor(), make_tensor(), make_tensor()

        kernel_options_1 = {
            "BLOCK_M": 32,
            "BLOCK_N": 32,
            "USE_TMA": False,
        }
        kernel_options_2 = {"BLOCK_M": 32, "BLOCK_N": 32, "USE_TMA": True}

        flex_compile = torch.compile(flex_attention, fullgraph=True, dynamic=True)
        out_compiled = flex_compile(query, key, value, kernel_options=kernel_options_1)
        out_tma_compiled = flex_compile(
            query, key, value, kernel_options=kernel_options_2
        )

        # vanilla compiled vs TMA compiled
        torch.testing.assert_close(out_tma_compiled, out_compiled, atol=2e-1, rtol=2e-1)

        if requires_grad:
            grad_output = torch.randn_like(out_compiled)

            out_compiled.backward(grad_output)
            compiled_grads = [query.grad, key.grad, value.grad]
            query.grad = None
            key.grad = None
            value.grad = None

            out_tma_compiled.backward(grad_output)
            tma_grads = [query.grad, key.grad, value.grad]
            query.grad = None
            key.grad = None
            value.grad = None

            torch.testing.assert_close(
                compiled_grads[0],
                tma_grads[0],
                atol=2e-1,
                rtol=2e-1,
            )
            torch.testing.assert_close(
                compiled_grads[1],
                tma_grads[1],
                atol=2e-1,
                rtol=2e-1,
            )
            torch.testing.assert_close(
                compiled_grads[2],
                tma_grads[2],
                atol=2e-1,
                rtol=2e-1,
            )

    @supported_platform
    @skip_on_cpu
    @expected_not_implemented_on_mps  # no backward yet
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
            fa._WARNINGS_SHOWN.clear()
            fa._WARNINGS_SHOWN.update(original_warnings_shown)

    @supported_platform
    def test_mask_mod_functools_partial(self, device):
        """Test that functools.partial wrapped mask_mod works with flex_attention.

        Regression test for https://github.com/pytorch/pytorch/issues/170489
        """
        B, H, S, D = 1, 8, 64, 64

        def causal_mask_with_offset(b, h, q_idx, kv_idx, offset):
            return (q_idx + offset) >= kv_idx

        offset = 31
        mask_mod = functools.partial(causal_mask_with_offset, offset=offset)

        query = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
        key = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
        value = torch.randn(B, H, S, D, device=device, dtype=torch.float16)

        block_mask = create_block_mask(mask_mod, B, H, S, S, device=device)

        flex_compiled = torch.compile(flex_attention)
        out = flex_compiled(query, key, value, block_mask=block_mask)

        self.assertEqual(out.shape, (B, H, S, D))

    @supported_platform
    def test_callable_class_mask_mod(self, device):
        # Test that callable class instances work as mask_mod with flex_attention

        B, H, S, D = 1, 8, 64, 64

        class CausalOrWindowMask:
            __name__ = "causal_or_window"

            def __init__(self, window_size):
                self.window_size = window_size

            def __call__(self, b, h, q_idx, kv_idx):
                causal = q_idx >= kv_idx
                in_window = (kv_idx - q_idx) < self.window_size
                return causal | in_window

            def __eq__(self, other):
                if not isinstance(other, CausalOrWindowMask):
                    return NotImplemented
                return self.window_size == other.window_size

            def __hash__(self):
                return hash(self.window_size)

        mask_mod = CausalOrWindowMask(window_size=64)

        query = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
        key = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
        value = torch.randn(B, H, S, D, device=device, dtype=torch.float16)

        @torch.compile(fullgraph=True)
        def f(q, k, v):
            block_mask = create_block_mask(mask_mod, B, H, S, S, device=q.device)
            return flex_attention(q, k, v, block_mask=block_mask)

        _ = f(query, key, value)

    @supported_platform
    @skip_on_cpu
    @skip_on_mps  # asserts TritonTemplateKernel layout-freezing internal
    def test_flex_attention_always_freezes_layout(self, device):
        """Test that flex attention always freezes FlexibleLayout inputs.

        When always_freeze_layout=True on flex attention templates,
        get_stride_and_maybe_freeze_layout should freeze FlexibleLayout
        immediately rather than using layout constraints.
        """
        from torch._inductor import ir
        from torch._inductor.select_algorithm import TritonTemplateKernel

        B, H, S, D = 2, 4, 128, 64
        dtype = torch.float16

        query = torch.randn(B, H, S, D, device=device, dtype=dtype)
        key = torch.randn(B, H, S, D, device=device, dtype=dtype)
        value = torch.randn(B, H, S, D, device=device, dtype=dtype)

        flexible_layout_called = False
        orig_stride_call = TritonTemplateKernel.get_stride_and_maybe_freeze_layout

        def tracking_get_stride(self, node):
            nonlocal flexible_layout_called
            flexible_layout = isinstance(node.data.layout, ir.FlexibleLayout)
            result = orig_stride_call(self, node)
            if flexible_layout:
                flexible_layout_called = True
                if not isinstance(node.data.layout, ir.FixedLayout):
                    raise AssertionError(
                        f"Expected FixedLayout, got {type(node.data.layout)}"
                    )
            return result

        with patch.object(
            TritonTemplateKernel,
            "get_stride_and_maybe_freeze_layout",
            tracking_get_stride,
        ):
            compiled_flex = torch.compile(flex_attention, fullgraph=True)
            compiled_flex(query, key, value)

        self.assertTrue(
            flexible_layout_called,
            "get_stride_and_maybe_freeze_layout should be called with FlexibleLayout nodes",
        )

    @unittest.skip(
        "Too large to run reliably in CI. Kept as a manual repro for #185262"
    )
    @supported_platform
    @skip_on_cpu
    @largeTensorTest("12GB", inductor=True)
    @serialTest()
    def test_large_kv_int64_pointer_math(self, device):
        # Regression test for #185262: load_checked_2d (and a few inline
        # pointer-arithmetic sites in the backwards template) computed
        # `offs * stride` in int32 even when Inductor had already promoted
        # INDEX_DTYPE to int64 because some input buffer exceeded 2**31
        # elements. The product silently overflowed and caused an IMA.
        #
        # We construct K/V so that (a) each tensor's storage exceeds 2**31
        # elements (so Inductor uses int64 indexing), and (b) the last KV
        # index times its KV-dim stride also exceeds 2**31 (so the
        # in-kernel pointer math would overflow without the .to(INDEX_DTYPE)
        # cast). Without the fix this raises a CUDA illegal memory access.
        H, D = 1, 128
        KV_S = 2**24 + 1024
        Q_S = 128
        dtype = torch.bfloat16

        q = torch.randn(1, H, Q_S, D, device=device, dtype=dtype)
        k = torch.randn(1, H, KV_S, D, device=device, dtype=dtype)
        v = torch.randn(1, H, KV_S, D, device=device, dtype=dtype)

        def mask_mod(b, h, q_idx, kv_idx):
            return kv_idx >= (KV_S - tail)

        BLOCK = _DEFAULT_SPARSE_BLOCK_SIZE
        tail = BLOCK
        last_kv_block = KV_S // BLOCK - 1
        kv_num_blocks = torch.ones(1, 1, 1, dtype=torch.int32, device=device)
        kv_indices = torch.full(
            (1, 1, 1, 1), last_kv_block, dtype=torch.int32, device=device
        )
        block_mask = BlockMask.from_kv_blocks(
            kv_num_blocks,
            kv_indices,
            BLOCK_SIZE=BLOCK,
            mask_mod=mask_mod,
            seq_lengths=(Q_S, KV_S),
        )

        compiled = torch.compile(flex_attention, fullgraph=True)
        out = compiled(q, k, v, block_mask=block_mask)

        # Reference: only the last `tail` KV positions are unmasked.
        k_tail = k[:, :, -tail:, :].to(torch.float32)
        v_tail = v[:, :, -tail:, :].to(torch.float32)
        q_f32 = q.to(torch.float32)
        scale = D**-0.5
        scores = (q_f32 @ k_tail.transpose(-1, -2)) * scale
        ref = (torch.softmax(scores, dim=-1) @ v_tail).to(dtype)

        torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)

    @supported_platform
    @common_utils.parametrize("gqa", (1, 4))
    @common_utils.parametrize("scale", (0.0625, 0.0, -0.07))
    def test_gfx950_d256_early_probability_compact(self, device, gqa, scale):
        self._run_gfx950_d256_early_probability_compact(device, gqa, scale)

    @supported_platform
    @common_utils.parametrize(
        "fallback",
        (
            "causal",
            "explicit_dense",
            "score_capture",
            "mask_capture",
            "early_dp",
            "panels",
            "separate",
            "tiles",
            "raw",
            "forward_only",
        ),
    )
    def test_gfx950_d256_early_probability_compact_fallback(self, device, fallback):
        self._run_gfx950_d256_early_probability_compact(
            device, 4, 0.0625, fallback=fallback
        )

    @supported_platform
    @common_utils.parametrize("scale", (192**-0.5, 0.0, -0.125))
    def test_gfx950_d192_gqa4_panel_phase(self, device, scale):
        self._run_gfx950_d192_panel_phase(device, 2048, scale, gqa4_pilot=True)

    @supported_platform
    @common_utils.parametrize("scale", (192**-0.5, 0.0, -0.125))
    def test_gfx950_d192_gqa4_dq_early_dp(self, device, scale):
        self._run_gfx950_d192_dq_early_dp(device, 2048, scale, gqa4_pilot=True)

    @supported_platform
    @common_utils.parametrize(
        "fallback",
        (
            "short",
            "missing_panel",
            "raw",
            "forward_only",
            "score_capture",
            "mask_capture",
        ),
    )
    def test_gfx950_d192_gqa4_dq_early_dp_fallback(self, device, fallback):
        self._run_gfx950_d192_dq_early_dp(
            device, 2048, 192**-0.5, fallback=fallback, gqa4_pilot=True
        )

    @supported_platform
    @common_utils.parametrize("length", (2048, 4096))
    @common_utils.parametrize("scale", (192**-0.5, 0.0, -0.125))
    def test_gfx950_d192_panel_phase(self, device, length, scale):
        self._run_gfx950_d192_panel_phase(device, length, scale)

    @supported_platform
    @common_utils.parametrize("fallback", ("short", "gqa4"))
    def test_gfx950_d192_panel_phase_fallback(self, device, fallback):
        self._run_gfx950_d192_panel_phase(device, 2048, 192**-0.5, fallback=fallback)

    @supported_platform
    @common_utils.parametrize("length", (2048, 4096))
    @common_utils.parametrize("scale", (96**-0.5, 0.0, -0.125))
    def test_gfx950_d96_dq_early_dp(self, device, length, scale):
        self._run_gfx950_d96_dq_early_dp(device, length, scale)

    @supported_platform
    @common_utils.parametrize(
        "fallback",
        (
            "sequence1024",
            "gqa4",
            "short",
            "causal",
            "contiguous",
            "missing_panel",
            "raw",
            "forward_only",
            "score_capture",
            "mask_capture",
        ),
    )
    def test_gfx950_d96_dq_early_dp_fallback(self, device, fallback):
        self._run_gfx950_d96_dq_early_dp(device, 4096, 96**-0.5, fallback=fallback)

    def _check_gfx950_backward_gradients_and_graphs(
        self, device, stream, providers, inputs, do, original_do, scale, reference
    ):
        q, k, _ = inputs
        comparisons = 0
        for sign in (1, -1, 1):
            do.copy_(original_do * sign)
            target = reference()
            actual = {
                name: (
                    problem["out"].detach().clone(),
                    *(gradient.detach().clone() for gradient in problem["backward"]()),
                )
                for name, problem in providers.items()
            }
            for value in actual.values():
                comparisons += self._check_gfx950_xcd_snake_values(
                    value, target, strict=False
                )
            comparisons += self._check_gfx950_xcd_snake_values(
                actual[True], actual[False], strict=True
            )
            if scale == 0:
                self.assertEqual(actual[True][1], torch.zeros_like(q), atol=0, rtol=0)
                self.assertEqual(actual[True][2], torch.zeros_like(k), atol=0, rtol=0)
        do.copy_(original_do)
        for problem in providers.values():
            for _ in range(5):
                problem["backward"]()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                retained = [problem["backward"]() for _ in range(10)]
            problem.update(graph=graph, retained=retained)
        for sign in (1, -1, 1):
            do.copy_(original_do * sign)
            for problem in providers.values():
                problem["graph"].replay()
            torch.cuda.synchronize(device)
            target = reference()
            for index in range(10):
                actual = {
                    name: (
                        problem["out"].detach().clone(),
                        *(
                            gradient.detach().clone()
                            for gradient in problem["retained"][index]
                        ),
                    )
                    for name, problem in providers.items()
                }
                for value in actual.values():
                    comparisons += self._check_gfx950_xcd_snake_values(
                        value, target, strict=False
                    )
                comparisons += self._check_gfx950_xcd_snake_values(
                    actual[True], actual[False], strict=True
                )
        self.assertEqual(comparisons, 9 * (len(inputs) + 1) * 11)

    def _run_gfx950_d96_dq_early_dp(self, device, length, scale, *, fallback=None):
        import ast

        from torch._functorch import config as aot_config
        from torch._inductor.codecache import PyCodeCache
        from torch._inductor.kernel.flex import flex_attention as flex_kernel_mod

        if (
            not torch.version.hip
            or torch.device(device).type != "cuda"
            or torch.cuda.get_device_properties(device).gcnArchName.split(":", 1)[0]
            != "gfx950"
        ):
            self.skipTest("Requires gfx950")
        batch, q_heads, kv_heads, q_len, kv_len = 4, 21, 21, length, length
        if fallback == "short":
            batch, q_len, kv_len = 1, 257, 260
        elif fallback == "gqa4":
            batch, q_heads, kv_heads, q_len, kv_len = 2, 8, 2, 4096, 4096
        elif fallback == "sequence1024":
            q_len = kv_len = 1024
        gqa = q_heads // kv_heads
        bias = (
            torch.linspace(-0.1, 0.1, kv_len, device=device)
            if fallback == "score_capture"
            else None
        )
        limit = (
            torch.tensor(kv_len // 2, device=device)
            if fallback == "mask_capture"
            else None
        )
        score_mod = (
            (lambda score, b, h, q_idx, kv_idx: score + bias[kv_idx])
            if bias is not None
            else None
        )
        mask_mod = (
            (lambda b, h, q_idx, kv_idx: q_idx >= kv_idx)
            if fallback == "causal"
            else (lambda b, h, q_idx, kv_idx: kv_idx < limit)
            if limit is not None
            else None
        )
        block_mask = (
            create_block_mask(mask_mod, batch, q_heads, q_len, kv_len, device=device)
            if mask_mod is not None
            else None
        )
        generator = torch.Generator(device=device).manual_seed(20261008)
        shapes = (
            (batch, q_heads, q_len, 96),
            (batch, kv_heads, kv_len, 96),
            (batch, kv_heads, kv_len, 96),
        )
        inputs = tuple(
            (
                torch.randn(
                    shape, device=device, dtype=torch.bfloat16, generator=generator
                )
                * 0.5
            ).requires_grad_()
            for shape in shapes
        )
        q, k, v = inputs
        do = (
            torch.randn(
                shapes[0], device=device, dtype=torch.bfloat16, generator=generator
            )
            * 0.25
        )
        original_do = do.clone()

        def reference():
            reference_inputs = tuple(
                value.detach().float().requires_grad_() for value in inputs
            )
            ref_q, ref_k, ref_v = reference_inputs
            scores = (
                ref_q @ ref_k.repeat_interleave(gqa, dim=1).transpose(-1, -2)
            ) * scale
            if bias is not None:
                scores = scores + bias.float()[None, None, None, :]
            if limit is not None:
                cols = torch.arange(kv_len, device=device)
                scores = scores.masked_fill(
                    cols[None, None, None, :] >= limit, float("-inf")
                )
            if fallback == "causal":
                rows = torch.arange(q_len, device=device)
                cols = torch.arange(kv_len, device=device)
                scores = scores.masked_fill(
                    rows[None, None, :, None] < cols[None, None, None, :],
                    float("-inf"),
                )
            out = scores.softmax(-1) @ ref_v.repeat_interleave(gqa, dim=1)
            gradients = torch.autograd.grad(out, reference_inputs, do.float())
            return (out.detach(), *(gradient.detach() for gradient in gradients))

        providers = {}
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        try:
            with (
                torch.cuda.stream(stream),
                temp_float32_matmul_precision("highest"),
                config.patch(
                    fx_graph_cache=False,
                    max_autotune=False,
                    compile_threads=1,
                    emulate_precision_casts=False,
                ),
                aot_config.patch(
                    enable_autograd_cache=False, enable_remote_autograd_cache=False
                ),
            ):
                for requested in (False, True):
                    torch._dynamo.reset()
                    options = dict(
                        BACKEND="TRITON",
                        bwd_BLOCK_M1=32,
                        bwd_BLOCK_N1=128,
                        bwd_BLOCK_M2=128,
                        bwd_BLOCK_N2=32,
                        bwd_num_warps=4,
                        bwd_num_stages=2,
                        bwd_USE_TMA=False,
                        bwd_PRESCALE_QK=False,
                        bwd_ROCM_D256_PANEL_DOTS=False,
                        bwd_ROCM_SEPARATE_DQ=False,
                        bwd_ROCM_EARLY_DP=False,
                        bwd_ROCM_D96_PANEL_DOTS=fallback != "missing_panel",
                        bwd_ROCM_D96_DQ_EARLY_DP=requested
                        if fallback not in ("raw", "forward_only")
                        else False,
                        ROCM_D96_DQ_EARLY_DP=requested if fallback == "raw" else False,
                        fwd_ROCM_D96_DQ_EARLY_DP=requested
                        if fallback == "forward_only"
                        else False,
                        bwd_BLOCKS_ARE_CONTIGUOUS=fallback != "contiguous",
                        bwd_ROCM_D192_PANEL_DOTS=False,
                        bwd_ROCM_D192_DQ_EARLY_DP=False,
                        bwd_matrix_instr_nonkdim=0,
                        bwd_waves_per_eu=0,
                        bwd_kpack=1,
                        fwd_USE_TMA=False,
                        fwd_PRESCALE_QK=False,
                        fwd_BLOCK_M=128,
                        fwd_BLOCK_N=32,
                        fwd_num_warps=4,
                        fwd_num_stages=1,
                        bwd_XCD_REMAP=False,
                        ROCM_D96_PANEL_DOTS=True,
                        ROCM_BWD_PHASE=13,
                        bwd_ROCM_BWD_PHASE=17,
                        ROCM_GQA_SPLITS=7,
                        bwd_ROCM_GQA_SPLITS=4,
                    )

                    def forward(query, key, value, options=options):
                        return flex_attention(
                            query,
                            key,
                            value,
                            enable_gqa=True,
                            score_mod=score_mod,
                            block_mask=block_mask,
                            scale=scale,
                            kernel_options=options,
                        )

                    compiled = torch.compile(forward, fullgraph=True, dynamic=False)
                    template = flex_kernel_mod.flex_attention_backward_template
                    original_append, original_select = (
                        template.maybe_append_choice,
                        flex_kernel_mod.autotune_select_algorithm,
                    )
                    choices, choice_metadata, selected, modules = [], {}, [], []
                    original_load = PyCodeCache.load_by_key_path

                    def retain_module(*args, **kwargs):
                        module = original_load(*args, **kwargs)
                        modules.append(module)
                        return module

                    def append_choice(*args, **kwargs):
                        choice_list, count = kwargs["choices"], len(kwargs["choices"])
                        metadata = (
                            kwargs["ROCM_BWD_PHASE"],
                            kwargs["ROCM_GQA_SPLITS"],
                            kwargs["ROCM_D96_PANEL_DOTS"],
                            kwargs["ROCM_D96_DQ_EARLY_DP"],
                        )
                        choices.append(
                            (
                                metadata,
                                len(kwargs["input_nodes"]),
                                [
                                    index
                                    for index, value in enumerate(kwargs["input_nodes"])
                                    if any(
                                        value is target
                                        for target in kwargs["mutated_inputs"]
                                    )
                                ],
                                {
                                    key: kwargs[key]
                                    for key in (
                                        "num_warps",
                                        "num_stages",
                                        "matrix_instr_nonkdim",
                                        "waves_per_eu",
                                        "kpack",
                                    )
                                },
                            )
                        )
                        result = original_append(*args, **kwargs)
                        for choice in choice_list[count:]:
                            choice_metadata[id(choice)] = metadata
                        return result

                    def select_algorithm(*args, **kwargs):
                        result = original_select(*args, **kwargs)
                        if args[0] in (
                            "flex_attention_backward",
                            "flex_attention_backward_dq",
                        ):
                            selected.append(choice_metadata.get(id(result[1])))
                        return result

                    def invoke():
                        out = compiled(q, k, v)
                        return out, torch.autograd.grad(
                            out, inputs, do, retain_graph=True
                        )

                    with (
                        mock.patch.object(
                            PyCodeCache, "load_by_key_path", side_effect=retain_module
                        ),
                        mock.patch.object(
                            template, "maybe_append_choice", side_effect=append_choice
                        ),
                        mock.patch.object(
                            flex_kernel_mod,
                            "autotune_select_algorithm",
                            side_effect=select_algorithm,
                        ),
                    ):
                        (out, initial), code = run_and_get_code(invoke)
                    torch.cuda.synchronize(device)
                    panel_active = fallback not in (
                        "sequence1024",
                        "short",
                        "causal",
                        "missing_panel",
                        "score_capture",
                        "mask_capture",
                    )
                    active = (
                        requested
                        and panel_active
                        and fallback
                        not in (
                            "raw",
                            "forward_only",
                            "gqa4",
                            "sequence1024",
                            "contiguous",
                        )
                    )
                    phases = [1, 2] if panel_active else [0]
                    self.assertEqual(
                        selected,
                        [
                            (
                                phase,
                                1,
                                panel_active and phase == 1,
                                active and phase == 2,
                            )
                            for phase in phases
                        ],
                    )
                    self.assertEqual(
                        [item[0][0] for item in choices],
                        [0, 1, 2] if panel_active else [0],
                    )
                    for (
                        phase,
                        splits,
                        panels,
                        early_dp,
                    ), count, mutated, compiler in choices:
                        self.assertEqual(splits, 1)
                        self.assertEqual(panels, panel_active and phase == 1)
                        self.assertEqual(early_dp, active and phase == 2)
                        self.assertEqual(count, 16)
                        self.assertEqual(
                            mutated, [7] if phase == 1 else [] if phase == 2 else [6, 7]
                        )
                        self.assertEqual(
                            {
                                key: value
                                for key, value in compiler.items()
                                if key != "kpack"
                            },
                            dict(
                                num_warps=4,
                                num_stages=2,
                                matrix_instr_nonkdim=0,
                                waves_per_eu=0,
                            ),
                        )
                    self.assertTrue(
                        all(item[-1]["kpack"] in (1, 2) for item in choices)
                    )
                    for value, shape in zip(
                        (out, *initial), (shapes[0], *shapes), strict=True
                    ):
                        self.assertEqual(value.dtype, torch.bfloat16)
                        self.assertEqual(tuple(value.shape), shape)
                    actual_phases = set()
                    for part in code:
                        wrapper = ast.parse(part)
                        bodies = [
                            part,
                            *(
                                node.value
                                for node in ast.walk(wrapper)
                                if isinstance(node, ast.Constant)
                                and isinstance(node.value, str)
                                and "flex_attention_backward" in node.value
                            ),
                        ]
                        for body in bodies:
                            parsed = ast.parse(body)
                            if not any(
                                isinstance(node, ast.FunctionDef)
                                and "flex_attention_backward" in node.name
                                for node in parsed.body
                            ):
                                continue
                            constants = {
                                node.target.id: node.value.value
                                for node in ast.walk(parsed)
                                if isinstance(node, ast.AnnAssign)
                                and isinstance(node.target, ast.Name)
                                and isinstance(node.value, ast.Constant)
                            }
                            phase = constants.get("ROCM_BWD_PHASE")
                            if phase not in (0, 1, 2):
                                continue
                            actual_phases.add(phase)
                            self.assertEqual(constants.get("ROCM_GQA_SPLITS"), 1)
                            self.assertEqual(
                                constants.get("ROCM_D96_PANEL_DOTS"),
                                panel_active and phase == 1,
                            )
                            self.assertEqual(
                                constants.get("ROCM_D96_DQ_EARLY_DP"),
                                active and phase == 2,
                            )
                            self.assertEqual(
                                constants.get("ROCM_D192_PANEL_DOTS"), False
                            )
                            self.assertEqual(
                                constants.get("BLOCKS_ARE_CONTIGUOUS"),
                                fallback != "contiguous",
                            )
                            self.assertEqual(constants.get("ROCM_XCD_SNAKE"), False)
                            self.assertEqual(
                                constants.get("ROCM_EARLY_PROBABILITY_COMPACT"), False
                            )
                            self.assertEqual(
                                constants.get("ROCM_D256_PANEL_DOTS"), False
                            )
                            for key, value in dict(
                                BLOCK_M1=32,
                                BLOCK_N1=128,
                                BLOCK_M2=128,
                                BLOCK_N2=32,
                                QK_HEAD_DIM=96,
                                V_HEAD_DIM=96,
                                QK_HEAD_DIM_ROUNDED=128,
                                V_HEAD_DIM_ROUNDED=128,
                                SAFE_HEAD_DIM=False,
                                GQA_SHARED_HEADS=gqa,
                            ).items():
                                self.assertEqual(constants.get(key), value)
                    self.assertTrue(set(phases).issubset(actual_phases))

                    def backward(out=out):
                        return torch.autograd.grad(out, inputs, do, retain_graph=True)

                    providers[requested] = dict(
                        compiled=compiled,
                        out=out,
                        backward=backward,
                        initial=initial,
                        code=code,
                        modules=tuple(modules),
                    )
                self._check_gfx950_backward_gradients_and_graphs(
                    device, stream, providers, inputs, do, original_do, scale, reference
                )
        finally:
            torch.cuda.current_stream(device).wait_stream(stream)

    @supported_platform
    @common_utils.parametrize("length", (2048, 4096))
    @common_utils.parametrize("scale", (192**-0.5, 0.0, -0.125))
    def test_gfx950_d192_dq_early_dp(self, device, length, scale):
        self._run_gfx950_d192_dq_early_dp(device, length, scale)

    @supported_platform
    @common_utils.parametrize(
        "fallback",
        (
            "short",
            "missing_panel",
            "raw",
            "forward_only",
            "score_capture",
            "mask_capture",
        ),
    )
    def test_gfx950_d192_dq_early_dp_fallback(self, device, fallback):
        self._run_gfx950_d192_dq_early_dp(device, 2048, 192**-0.5, fallback=fallback)

    def _run_gfx950_d192_dq_early_dp(
        self, device, length, scale, *, fallback=None, gqa4_pilot=False
    ):
        import ast

        from torch._functorch import config as aot_config
        from torch._inductor.codecache import PyCodeCache
        from torch._inductor.kernel.flex import flex_attention as flex_kernel_mod

        if (
            not torch.version.hip
            or torch.device(device).type != "cuda"
            or torch.cuda.get_device_properties(device).gcnArchName.split(":", 1)[0]
            != "gfx950"
        ):
            self.skipTest("Requires gfx950")
        batch, q_heads, kv_heads, q_len, kv_len = 4, 10, 10, length, length
        if gqa4_pilot:
            batch, q_heads, kv_heads = 2, 8, 2
        if fallback == "short":
            batch, q_len, kv_len = 1, 257, 260
        elif fallback == "gqa4":
            batch, q_heads, kv_heads, q_len, kv_len = 1, 8, 2, 257, 260
        gqa = q_heads // kv_heads
        bias = (
            torch.linspace(-0.1, 0.1, kv_len, device=device)
            if fallback == "score_capture"
            else None
        )
        limit = (
            torch.tensor(kv_len // 2, device=device)
            if fallback == "mask_capture"
            else None
        )
        score_mod = (
            (lambda score, b, h, q_idx, kv_idx: score + bias[kv_idx])
            if bias is not None
            else None
        )
        mask_mod = (
            (lambda b, h, q_idx, kv_idx: kv_idx < limit) if limit is not None else None
        )
        block_mask = (
            create_block_mask(mask_mod, batch, q_heads, q_len, kv_len, device=device)
            if mask_mod is not None
            else None
        )
        generator = torch.Generator(device=device).manual_seed(20261008)
        shapes = (
            (batch, q_heads, q_len, 192),
            (batch, kv_heads, kv_len, 192),
            (batch, kv_heads, kv_len, 192),
        )
        inputs = tuple(
            (
                torch.randn(
                    shape, device=device, dtype=torch.bfloat16, generator=generator
                )
                * 0.5
            ).requires_grad_()
            for shape in shapes
        )
        q, k, v = inputs
        do = (
            torch.randn(
                shapes[0], device=device, dtype=torch.bfloat16, generator=generator
            )
            * 0.25
        )
        original_do = do.clone()

        def reference():
            reference_inputs = tuple(
                value.detach().float().requires_grad_() for value in inputs
            )
            ref_q, ref_k, ref_v = reference_inputs
            scores = (
                ref_q @ ref_k.repeat_interleave(gqa, dim=1).transpose(-1, -2)
            ) * scale
            if bias is not None:
                scores = scores + bias.float()[None, None, None, :]
            if limit is not None:
                cols = torch.arange(kv_len, device=device)
                scores = scores.masked_fill(
                    cols[None, None, None, :] >= limit, float("-inf")
                )
            out = scores.softmax(-1) @ ref_v.repeat_interleave(gqa, dim=1)
            gradients = torch.autograd.grad(out, reference_inputs, do.float())
            return (out.detach(), *(gradient.detach() for gradient in gradients))

        providers = {}
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        try:
            with (
                torch.cuda.stream(stream),
                temp_float32_matmul_precision("highest"),
                config.patch(
                    fx_graph_cache=False,
                    max_autotune=False,
                    compile_threads=1,
                    emulate_precision_casts=False,
                ),
                aot_config.patch(
                    enable_autograd_cache=False, enable_remote_autograd_cache=False
                ),
            ):
                for requested in (False, True):
                    torch._dynamo.reset()
                    options = dict(
                        BACKEND="TRITON",
                        bwd_BLOCK_M1=16,
                        bwd_BLOCK_N1=64,
                        bwd_BLOCK_M2=128,
                        bwd_BLOCK_N2=32,
                        bwd_num_warps=4,
                        bwd_num_stages=2,
                        bwd_USE_TMA=False,
                        bwd_PRESCALE_QK=False,
                        bwd_ROCM_D256_PANEL_DOTS=False,
                        bwd_ROCM_SEPARATE_DQ=False,
                        bwd_ROCM_EARLY_DP=False,
                        bwd_ROCM_D192_PANEL_DOTS=fallback != "missing_panel",
                        bwd_ROCM_D192_DQ_EARLY_DP=requested
                        if fallback not in ("raw", "forward_only")
                        else False,
                        ROCM_D192_DQ_EARLY_DP=requested if fallback == "raw" else False,
                        fwd_ROCM_D192_DQ_EARLY_DP=requested
                        if fallback == "forward_only"
                        else False,
                        bwd_BLOCKS_ARE_CONTIGUOUS=True,
                        bwd_ROCM_D96_PANEL_DOTS=False,
                        bwd_matrix_instr_nonkdim=0,
                        bwd_waves_per_eu=0,
                        bwd_kpack=1,
                        fwd_USE_TMA=False,
                        fwd_PRESCALE_QK=False,
                        fwd_BLOCK_M=128,
                        fwd_BLOCK_N=32,
                        fwd_num_warps=4,
                        fwd_num_stages=1,
                        bwd_XCD_REMAP=False,
                        ROCM_D192_PANEL_DOTS=True,
                        ROCM_BWD_PHASE=13,
                        bwd_ROCM_BWD_PHASE=17,
                        ROCM_GQA_SPLITS=7,
                        bwd_ROCM_GQA_SPLITS=4,
                    )

                    def forward(query, key, value, options=options):
                        return flex_attention(
                            query,
                            key,
                            value,
                            enable_gqa=True,
                            score_mod=score_mod,
                            block_mask=block_mask,
                            scale=scale,
                            kernel_options=options,
                        )

                    compiled = torch.compile(forward, fullgraph=True, dynamic=False)
                    template = flex_kernel_mod.flex_attention_backward_template
                    original_append, original_select = (
                        template.maybe_append_choice,
                        flex_kernel_mod.autotune_select_algorithm,
                    )
                    choices, choice_metadata, selected, modules = [], {}, [], []
                    original_load = PyCodeCache.load_by_key_path

                    def retain_module(*args, **kwargs):
                        module = original_load(*args, **kwargs)
                        modules.append(module)
                        return module

                    def append_choice(*args, **kwargs):
                        choice_list, count = kwargs["choices"], len(kwargs["choices"])
                        metadata = (
                            kwargs["ROCM_BWD_PHASE"],
                            kwargs["ROCM_GQA_SPLITS"],
                            kwargs["ROCM_D192_PANEL_DOTS"],
                            kwargs["ROCM_D192_DQ_EARLY_DP"],
                        )
                        choices.append(
                            (
                                metadata,
                                len(kwargs["input_nodes"]),
                                [
                                    index
                                    for index, value in enumerate(kwargs["input_nodes"])
                                    if any(
                                        value is target
                                        for target in kwargs["mutated_inputs"]
                                    )
                                ],
                                {
                                    key: kwargs[key]
                                    for key in (
                                        "num_warps",
                                        "num_stages",
                                        "matrix_instr_nonkdim",
                                        "waves_per_eu",
                                        "kpack",
                                    )
                                },
                            )
                        )
                        result = original_append(*args, **kwargs)
                        for choice in choice_list[count:]:
                            choice_metadata[id(choice)] = metadata
                        return result

                    def select_algorithm(*args, **kwargs):
                        result = original_select(*args, **kwargs)
                        if args[0] in (
                            "flex_attention_backward",
                            "flex_attention_backward_dq",
                        ):
                            selected.append(choice_metadata.get(id(result[1])))
                        return result

                    def invoke():
                        out = compiled(q, k, v)
                        return out, torch.autograd.grad(
                            out, inputs, do, retain_graph=True
                        )

                    with (
                        mock.patch.object(
                            PyCodeCache, "load_by_key_path", side_effect=retain_module
                        ),
                        mock.patch.object(
                            template, "maybe_append_choice", side_effect=append_choice
                        ),
                        mock.patch.object(
                            flex_kernel_mod,
                            "autotune_select_algorithm",
                            side_effect=select_algorithm,
                        ),
                    ):
                        (out, initial), code = run_and_get_code(invoke)
                    torch.cuda.synchronize(device)
                    panel_active = fallback not in (
                        "short",
                        "gqa4",
                        "missing_panel",
                        "score_capture",
                        "mask_capture",
                    )
                    active = (
                        requested
                        and panel_active
                        and fallback not in ("raw", "forward_only")
                    )
                    phases = [1, 2] if panel_active else [0]
                    self.assertEqual(
                        selected,
                        [
                            (
                                phase,
                                1,
                                panel_active and phase == 1,
                                active and phase == 2,
                            )
                            for phase in phases
                        ],
                    )
                    self.assertEqual(
                        [item[0][0] for item in choices],
                        [0, 1, 2] if panel_active else [0],
                    )
                    for (
                        phase,
                        splits,
                        panels,
                        early_dp,
                    ), count, mutated, compiler in choices:
                        self.assertEqual(splits, 1)
                        self.assertEqual(panels, panel_active and phase == 1)
                        self.assertEqual(early_dp, active and phase == 2)
                        self.assertEqual(count, 16)
                        self.assertEqual(
                            mutated, [7] if phase == 1 else [] if phase == 2 else [6, 7]
                        )
                        self.assertEqual(
                            {
                                key: value
                                for key, value in compiler.items()
                                if key != "kpack"
                            },
                            dict(
                                num_warps=4,
                                num_stages=2,
                                matrix_instr_nonkdim=0,
                                waves_per_eu=0,
                            ),
                        )
                    self.assertTrue(
                        all(item[-1]["kpack"] in (1, 2) for item in choices)
                    )
                    for value, shape in zip(
                        (out, *initial), (shapes[0], *shapes), strict=True
                    ):
                        self.assertEqual(value.dtype, torch.bfloat16)
                        self.assertEqual(tuple(value.shape), shape)
                    actual_phases = set()
                    for part in code:
                        wrapper = ast.parse(part)
                        bodies = [
                            part,
                            *(
                                node.value
                                for node in ast.walk(wrapper)
                                if isinstance(node, ast.Constant)
                                and isinstance(node.value, str)
                                and "flex_attention_backward" in node.value
                            ),
                        ]
                        for body in bodies:
                            parsed = ast.parse(body)
                            if not any(
                                isinstance(node, ast.FunctionDef)
                                and "flex_attention_backward" in node.name
                                for node in parsed.body
                            ):
                                continue
                            constants = {
                                node.target.id: node.value.value
                                for node in ast.walk(parsed)
                                if isinstance(node, ast.AnnAssign)
                                and isinstance(node.target, ast.Name)
                                and isinstance(node.value, ast.Constant)
                            }
                            phase = constants.get("ROCM_BWD_PHASE")
                            if phase not in (0, 1, 2):
                                continue
                            actual_phases.add(phase)
                            self.assertEqual(constants.get("ROCM_GQA_SPLITS"), 1)
                            self.assertEqual(
                                constants.get("ROCM_D192_PANEL_DOTS"),
                                panel_active and phase == 1,
                            )
                            self.assertEqual(
                                constants.get("ROCM_D192_DQ_EARLY_DP"),
                                active and phase == 2,
                            )
                            self.assertEqual(
                                constants.get("ROCM_D96_PANEL_DOTS"), False
                            )
                            self.assertEqual(
                                constants.get("BLOCKS_ARE_CONTIGUOUS"), True
                            )
                            self.assertEqual(constants.get("ROCM_XCD_SNAKE"), False)
                            self.assertEqual(
                                constants.get("ROCM_EARLY_PROBABILITY_COMPACT"), False
                            )
                            self.assertEqual(
                                constants.get("ROCM_D256_PANEL_DOTS"), False
                            )
                            for key, value in dict(
                                BLOCK_M1=16,
                                BLOCK_N1=64,
                                BLOCK_M2=128,
                                BLOCK_N2=32,
                                QK_HEAD_DIM=192,
                                V_HEAD_DIM=192,
                                QK_HEAD_DIM_ROUNDED=256,
                                V_HEAD_DIM_ROUNDED=256,
                                SAFE_HEAD_DIM=False,
                                GQA_SHARED_HEADS=gqa,
                            ).items():
                                self.assertEqual(constants.get(key), value)
                    self.assertTrue(set(phases).issubset(actual_phases))

                    def backward(out=out):
                        return torch.autograd.grad(out, inputs, do, retain_graph=True)

                    providers[requested] = dict(
                        compiled=compiled,
                        out=out,
                        backward=backward,
                        initial=initial,
                        code=code,
                        modules=tuple(modules),
                    )
                self._check_gfx950_backward_gradients_and_graphs(
                    device, stream, providers, inputs, do, original_do, scale, reference
                )
        finally:
            torch.cuda.current_stream(device).wait_stream(stream)

    @supported_platform
    @common_utils.parametrize("scale", (128**-0.5, 0.0, -0.125))
    def test_gfx950_d128_xcd_reverse_dq_rows(self, device, scale):
        self._run_gfx950_d128_xcd_reverse_dq_rows(device, 4096, scale)

    @supported_platform
    @common_utils.parametrize(
        "fallback",
        ("short", "xcd_false", "raw", "forward_only", "score_capture", "mask_capture"),
    )
    def test_gfx950_d128_xcd_reverse_dq_rows_fallback(self, device, fallback):
        self._run_gfx950_d128_xcd_reverse_dq_rows(
            device, 4096, 128**-0.5, fallback=fallback
        )

    def _run_gfx950_d128_xcd_reverse_dq_rows(
        self, device, length, scale, *, fallback=None
    ):
        import ast

        from torch._functorch import config as aot_config
        from torch._inductor.codecache import PyCodeCache
        from torch._inductor.kernel.flex import flex_attention as flex_kernel_mod

        if (
            not torch.version.hip
            or torch.device(device).type != "cuda"
            or torch.cuda.get_device_properties(device).gcnArchName.split(":", 1)[0]
            != "gfx950"
        ):
            self.skipTest("Requires gfx950")
        batch, q_heads, kv_heads, q_len, kv_len = 4, 64, 8, length, length
        if fallback == "short":
            batch, q_len, kv_len = 1, 257, 260
        gqa = q_heads // kv_heads
        bias = (
            torch.linspace(-0.1, 0.1, kv_len, device=device)
            if fallback == "score_capture"
            else None
        )
        limit = (
            torch.tensor(kv_len // 2, device=device)
            if fallback == "mask_capture"
            else None
        )
        score_mod = (
            (lambda score, b, h, q_idx, kv_idx: score + bias[kv_idx])
            if bias is not None
            else None
        )
        mask_mod = (
            (lambda b, h, q_idx, kv_idx: (q_idx >= kv_idx) & (kv_idx < limit))
            if limit is not None
            else (lambda b, h, q_idx, kv_idx: q_idx >= kv_idx)
        )
        block_mask = create_block_mask(
            mask_mod, None, None, q_len, kv_len, device=device, BLOCK_SIZE=128
        )
        generator = torch.Generator(device=device).manual_seed(20261008)
        shapes = (
            (batch, q_heads, q_len, 128),
            (batch, kv_heads, kv_len, 128),
            (batch, kv_heads, kv_len, 128),
        )
        inputs = tuple(
            (
                torch.randn(
                    shape, device=device, dtype=torch.bfloat16, generator=generator
                )
                * 0.5
            ).requires_grad_()
            for shape in shapes
        )
        q, k, v = inputs
        do = (
            torch.randn(
                shapes[0], device=device, dtype=torch.bfloat16, generator=generator
            )
            * 0.25
        )
        original_do = do.clone()

        def reference():
            reference_inputs = tuple(
                value.detach().float().requires_grad_() for value in inputs
            )
            ref_q, ref_k, ref_v = reference_inputs
            scores = (
                ref_q @ ref_k.repeat_interleave(gqa, dim=1).transpose(-1, -2)
            ) * scale
            if bias is not None:
                scores = scores + bias.float()[None, None, None, :]
            rows = torch.arange(q_len, device=device)
            cols = torch.arange(kv_len, device=device)
            scores = scores.masked_fill(
                rows[None, None, :, None] < cols[None, None, None, :], float("-inf")
            )
            if limit is not None:
                cols = torch.arange(kv_len, device=device)
                scores = scores.masked_fill(
                    cols[None, None, None, :] >= limit, float("-inf")
                )
            out = scores.softmax(-1) @ ref_v.repeat_interleave(gqa, dim=1)
            gradients = torch.autograd.grad(out, reference_inputs, do.float())
            return (out.detach(), *(gradient.detach() for gradient in gradients))

        providers = {}
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        try:
            with (
                torch.cuda.stream(stream),
                temp_float32_matmul_precision("highest"),
                config.patch(
                    fx_graph_cache=False,
                    max_autotune=False,
                    compile_threads=1,
                    emulate_precision_casts=False,
                ),
                aot_config.patch(
                    enable_autograd_cache=False, enable_remote_autograd_cache=False
                ),
            ):
                for requested in (False, True):
                    torch._dynamo.reset()
                    options = dict(
                        BACKEND="TRITON",
                        bwd_BLOCK_M1=32,
                        bwd_BLOCK_N1=128,
                        bwd_BLOCK_M2=128,
                        bwd_BLOCK_N2=32,
                        bwd_num_warps=4,
                        bwd_num_stages=2,
                        bwd_USE_TMA=False,
                        bwd_PRESCALE_QK=False,
                        bwd_ROCM_D256_PANEL_DOTS=False,
                        bwd_ROCM_SEPARATE_DQ=True,
                        bwd_ROCM_EARLY_DP=True,
                        bwd_ROCM_D192_PANEL_DOTS=False,
                        bwd_ROCM_D192_DQ_EARLY_DP=False,
                        bwd_ROCM_D128_REVERSE_DQ_ROWS=requested
                        if fallback not in ("raw", "forward_only")
                        else False,
                        ROCM_D128_REVERSE_DQ_ROWS=requested
                        if fallback == "raw"
                        else False,
                        fwd_ROCM_D128_REVERSE_DQ_ROWS=requested
                        if fallback == "forward_only"
                        else False,
                        bwd_BLOCKS_ARE_CONTIGUOUS=fallback != "short",
                        bwd_ROCM_D96_PANEL_DOTS=False,
                        bwd_matrix_instr_nonkdim=0,
                        bwd_waves_per_eu=0,
                        bwd_kpack=1,
                        fwd_USE_TMA=False,
                        fwd_PRESCALE_QK=False,
                        fwd_BLOCK_M=128,
                        fwd_BLOCK_N=32,
                        fwd_num_warps=4,
                        fwd_num_stages=1,
                        bwd_XCD_REMAP=fallback != "xcd_false",
                        ROCM_D192_PANEL_DOTS=False,
                        ROCM_BWD_PHASE=13,
                        bwd_ROCM_BWD_PHASE=17,
                        ROCM_GQA_SPLITS=7,
                        bwd_ROCM_GQA_SPLITS=4,
                    )

                    def forward(query, key, value, options=options):
                        return flex_attention(
                            query,
                            key,
                            value,
                            enable_gqa=True,
                            score_mod=score_mod,
                            block_mask=block_mask,
                            scale=scale,
                            kernel_options=options,
                        )

                    compiled = torch.compile(forward, fullgraph=True, dynamic=False)
                    template = flex_kernel_mod.flex_attention_backward_template
                    original_append, original_select = (
                        template.maybe_append_choice,
                        flex_kernel_mod.autotune_select_algorithm,
                    )
                    choices, choice_metadata, selected, modules = [], {}, [], []
                    original_load = PyCodeCache.load_by_key_path

                    def retain_module(*args, **kwargs):
                        module = original_load(*args, **kwargs)
                        modules.append(module)
                        return module

                    def append_choice(*args, **kwargs):
                        choice_list, count = kwargs["choices"], len(kwargs["choices"])
                        metadata = (
                            kwargs["ROCM_BWD_PHASE"],
                            kwargs["ROCM_GQA_SPLITS"],
                            kwargs["XCD_REMAP"],
                            kwargs["ROCM_D128_REVERSE_DQ_ROWS"],
                        )
                        choices.append(
                            (
                                metadata,
                                len(kwargs["input_nodes"]),
                                [
                                    index
                                    for index, value in enumerate(kwargs["input_nodes"])
                                    if any(
                                        value is target
                                        for target in kwargs["mutated_inputs"]
                                    )
                                ],
                                {
                                    key: kwargs[key]
                                    for key in (
                                        "num_warps",
                                        "num_stages",
                                        "matrix_instr_nonkdim",
                                        "waves_per_eu",
                                        "kpack",
                                    )
                                },
                            )
                        )
                        result = original_append(*args, **kwargs)
                        for choice in choice_list[count:]:
                            choice_metadata[id(choice)] = metadata
                        return result

                    def select_algorithm(*args, **kwargs):
                        result = original_select(*args, **kwargs)
                        if args[0] in (
                            "flex_attention_backward",
                            "flex_attention_backward_dq",
                        ):
                            selected.append(choice_metadata.get(id(result[1])))
                        return result

                    def invoke():
                        out = compiled(q, k, v)
                        return out, torch.autograd.grad(
                            out, inputs, do, retain_graph=True
                        )

                    with (
                        mock.patch.object(
                            PyCodeCache, "load_by_key_path", side_effect=retain_module
                        ),
                        mock.patch.object(
                            template, "maybe_append_choice", side_effect=append_choice
                        ),
                        mock.patch.object(
                            flex_kernel_mod,
                            "autotune_select_algorithm",
                            side_effect=select_algorithm,
                        ),
                    ):
                        (out, initial), code = run_and_get_code(invoke)
                    torch.cuda.synchronize(device)
                    separate_active = fallback not in ("score_capture", "mask_capture")
                    active = requested and fallback is None
                    xcd_active = fallback != "xcd_false"
                    phases = [1, 2] if separate_active else [0]
                    self.assertEqual(
                        selected,
                        [
                            (
                                phase,
                                1,
                                xcd_active,
                                active and phase == 2,
                            )
                            for phase in phases
                        ],
                    )
                    self.assertEqual(
                        [item[0][0] for item in choices],
                        [0, 1, 2] if separate_active else [0],
                    )
                    for (
                        phase,
                        splits,
                        xcd,
                        reverse_rows,
                    ), count, mutated, compiler in choices:
                        self.assertEqual(splits, 1)
                        self.assertEqual(xcd, xcd_active)
                        self.assertEqual(reverse_rows, active and phase == 2)
                        self.assertEqual(count, 16)
                        self.assertEqual(
                            mutated, [7] if phase == 1 else [] if phase == 2 else [6, 7]
                        )
                        self.assertEqual(
                            {
                                key: value
                                for key, value in compiler.items()
                                if key != "kpack"
                            },
                            dict(
                                num_warps=4,
                                num_stages=2,
                                matrix_instr_nonkdim=0,
                                waves_per_eu=0,
                            ),
                        )
                    self.assertTrue(
                        all(item[-1]["kpack"] in (1, 2) for item in choices)
                    )
                    for value, shape in zip(
                        (out, *initial), (shapes[0], *shapes), strict=True
                    ):
                        self.assertEqual(value.dtype, torch.bfloat16)
                        self.assertEqual(tuple(value.shape), shape)
                    actual_phases = set()
                    for part in code:
                        wrapper = ast.parse(part)
                        bodies = [
                            part,
                            *(
                                node.value
                                for node in ast.walk(wrapper)
                                if isinstance(node, ast.Constant)
                                and isinstance(node.value, str)
                                and "flex_attention_backward" in node.value
                            ),
                        ]
                        for body in bodies:
                            parsed = ast.parse(body)
                            if not any(
                                isinstance(node, ast.FunctionDef)
                                and "flex_attention_backward" in node.name
                                for node in parsed.body
                            ):
                                continue
                            constants = {
                                node.target.id: node.value.value
                                for node in ast.walk(parsed)
                                if isinstance(node, ast.AnnAssign)
                                and isinstance(node.target, ast.Name)
                                and isinstance(node.value, ast.Constant)
                            }
                            phase = constants.get("ROCM_BWD_PHASE")
                            if phase not in (0, 1, 2):
                                continue
                            actual_phases.add(phase)
                            self.assertEqual(constants.get("ROCM_GQA_SPLITS"), 1)
                            self.assertEqual(
                                constants.get("XCD_REMAP"),
                                xcd_active,
                            )
                            self.assertEqual(
                                constants.get("ROCM_D128_REVERSE_DQ_ROWS"),
                                active and phase == 2,
                            )
                            self.assertEqual(
                                constants.get("ROCM_D96_PANEL_DOTS"), False
                            )
                            self.assertEqual(
                                constants.get("BLOCKS_ARE_CONTIGUOUS"),
                                fallback != "short",
                            )
                            self.assertEqual(constants.get("ROCM_XCD_SNAKE"), False)
                            self.assertEqual(constants.get("ROCM_EARLY_DP"), True)
                            self.assertEqual(
                                constants.get("ROCM_D192_PANEL_DOTS"), False
                            )
                            self.assertEqual(
                                constants.get("ROCM_D192_DQ_EARLY_DP"), False
                            )
                            self.assertEqual(
                                constants.get("ROCM_D96_DQ_EARLY_DP"), False
                            )
                            self.assertEqual(
                                constants.get("ROCM_EARLY_PROBABILITY_COMPACT"), False
                            )
                            self.assertEqual(
                                constants.get("ROCM_D256_PANEL_DOTS"), False
                            )
                            for key, value in dict(
                                BLOCK_M1=32,
                                BLOCK_N1=128,
                                BLOCK_M2=128,
                                BLOCK_N2=32,
                                QK_HEAD_DIM=128,
                                V_HEAD_DIM=128,
                                QK_HEAD_DIM_ROUNDED=128,
                                V_HEAD_DIM_ROUNDED=128,
                                SAFE_HEAD_DIM=True,
                                GQA_SHARED_HEADS=gqa,
                            ).items():
                                self.assertEqual(constants.get(key), value)
                    self.assertTrue(set(phases).issubset(actual_phases))

                    def backward(out=out):
                        return torch.autograd.grad(out, inputs, do, retain_graph=True)

                    providers[requested] = dict(
                        compiled=compiled,
                        out=out,
                        backward=backward,
                        initial=initial,
                        code=code,
                        modules=tuple(modules),
                    )
                self._check_gfx950_backward_gradients_and_graphs(
                    device, stream, providers, inputs, do, original_do, scale, reference
                )
        finally:
            torch.cuda.current_stream(device).wait_stream(stream)

    @supported_platform
    @largeTensorTest("16GB", inductor=True)
    @common_utils.parametrize("length", (2048, 4096))
    def test_gfx950_d192_panel_phase_automatic(self, device, length):
        import math

        self._run_gfx950_d192_panel_phase(
            device, length, 1 / math.sqrt(192), automatic=True
        )

    def _run_gfx950_d192_panel_phase(
        self, device, length, scale, *, fallback=None, gqa4_pilot=False, automatic=False
    ):
        import ast

        from torch._functorch import config as aot_config
        from torch._inductor.codecache import PyCodeCache
        from torch._inductor.kernel.flex import flex_attention as flex_kernel_mod

        if (
            not torch.version.hip
            or torch.device(device).type != "cuda"
            or torch.cuda.get_device_properties(device).gcnArchName.split(":", 1)[0]
            != "gfx950"
        ):
            self.skipTest("Requires gfx950")
        batch, q_heads, kv_heads, q_len, kv_len = 4, 10, 10, length, length
        if gqa4_pilot:
            batch, q_heads, kv_heads = 2, 8, 2
        if fallback == "short":
            batch, q_len, kv_len = 1, 257, 260
        elif fallback == "gqa4":
            batch, q_heads, kv_heads, q_len, kv_len = 1, 8, 2, 257, 260
        gqa = q_heads // kv_heads
        generator = torch.Generator(device=device).manual_seed(20261008)
        shapes = (
            (batch, q_heads, q_len, 192),
            (batch, kv_heads, kv_len, 192),
            (batch, kv_heads, kv_len, 192),
        )
        inputs = tuple(
            (
                torch.randn(
                    shape, device=device, dtype=torch.bfloat16, generator=generator
                )
                * 0.5
            ).requires_grad_()
            for shape in shapes
        )
        q, k, v = inputs
        do = (
            torch.randn(
                shapes[0], device=device, dtype=torch.bfloat16, generator=generator
            )
            * 0.25
        )
        original_do = do.clone()

        def reference():
            reference_inputs = tuple(
                value.detach().float().requires_grad_() for value in inputs
            )
            ref_q, ref_k, ref_v = reference_inputs
            scores = (
                ref_q @ ref_k.repeat_interleave(gqa, dim=1).transpose(-1, -2)
            ) * scale
            out = scores.softmax(-1) @ ref_v.repeat_interleave(gqa, dim=1)
            gradients = torch.autograd.grad(out, reference_inputs, do.float())
            return (out.detach(), *(gradient.detach() for gradient in gradients))

        providers = {}
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        try:
            with (
                torch.cuda.stream(stream),
                temp_float32_matmul_precision("highest"),
                config.patch(
                    fx_graph_cache=False,
                    max_autotune=False,
                    compile_threads=1,
                    emulate_precision_casts=False,
                ),
                aot_config.patch(
                    enable_autograd_cache=False, enable_remote_autograd_cache=False
                ),
            ):
                for requested in (False, True):
                    torch._dynamo.reset()
                    options = dict(
                        BACKEND="TRITON",
                        bwd_BLOCK_M1=16,
                        bwd_BLOCK_N1=64,
                        bwd_BLOCK_M2=128,
                        bwd_BLOCK_N2=32,
                        bwd_num_warps=4,
                        bwd_num_stages=2,
                        bwd_USE_TMA=False,
                        bwd_PRESCALE_QK=False,
                        bwd_ROCM_D256_PANEL_DOTS=False,
                        bwd_ROCM_SEPARATE_DQ=False,
                        bwd_ROCM_EARLY_DP=False,
                        bwd_ROCM_D192_PANEL_DOTS=requested,
                        bwd_matrix_instr_nonkdim=0,
                        bwd_waves_per_eu=0,
                        bwd_kpack=1,
                        fwd_USE_TMA=False,
                        fwd_PRESCALE_QK=False,
                        fwd_BLOCK_M=128,
                        fwd_BLOCK_N=32,
                        fwd_num_warps=4,
                        fwd_num_stages=1,
                        bwd_XCD_REMAP=False,
                        ROCM_D192_PANEL_DOTS=True,
                        ROCM_BWD_PHASE=13,
                        bwd_ROCM_BWD_PHASE=17,
                        ROCM_GQA_SPLITS=7,
                        bwd_ROCM_GQA_SPLITS=4,
                    )

                    if automatic:
                        options = {"BACKEND": "TRITON"}
                        if not requested:
                            options["bwd_ROCM_SEPARATE_DQ"] = False

                    def forward(query, key, value, options=options):
                        return flex_attention(
                            query,
                            key,
                            value,
                            enable_gqa=True,
                            scale=scale,
                            kernel_options=options,
                        )

                    compiled = torch.compile(forward, fullgraph=True, dynamic=False)
                    template = flex_kernel_mod.flex_attention_backward_template
                    original_append, original_select = (
                        template.maybe_append_choice,
                        flex_kernel_mod.autotune_select_algorithm,
                    )
                    choices, choice_metadata, selected, modules = [], {}, [], []
                    automatic_options_by_choice, selected_automatic_options = {}, []
                    original_load = PyCodeCache.load_by_key_path

                    def retain_module(*args, **kwargs):
                        module = original_load(*args, **kwargs)
                        modules.append(module)
                        return module

                    def append_choice(*args, **kwargs):
                        choice_list, count = kwargs["choices"], len(kwargs["choices"])
                        metadata = (
                            kwargs["ROCM_BWD_PHASE"],
                            kwargs["ROCM_GQA_SPLITS"],
                            kwargs["ROCM_D192_PANEL_DOTS"],
                        )
                        choices.append(
                            (
                                metadata,
                                len(kwargs["input_nodes"]),
                                [
                                    index
                                    for index, value in enumerate(kwargs["input_nodes"])
                                    if any(
                                        value is target
                                        for target in kwargs["mutated_inputs"]
                                    )
                                ],
                                {
                                    key: kwargs[key]
                                    for key in (
                                        "num_warps",
                                        "num_stages",
                                        "matrix_instr_nonkdim",
                                        "waves_per_eu",
                                        "kpack",
                                    )
                                },
                            )
                        )
                        result = original_append(*args, **kwargs)
                        for choice in choice_list[count:]:
                            choice_metadata[id(choice)] = metadata
                            if automatic:
                                automatic_options_by_choice[id(choice)] = {
                                    key: kwargs[key]
                                    for key in (
                                        "ROCM_BWD_PHASE",
                                        "BLOCK_M1",
                                        "BLOCK_N1",
                                        "BLOCK_M2",
                                        "BLOCK_N2",
                                        "num_warps",
                                        "num_stages",
                                        "matrix_instr_nonkdim",
                                        "waves_per_eu",
                                        "kpack",
                                        "ROCM_D192_PANEL_DOTS",
                                        "ROCM_D192_DQ_EARLY_DP",
                                    )
                                }
                        return result

                    def select_algorithm(*args, **kwargs):
                        result = original_select(*args, **kwargs)
                        if args[0] in (
                            "flex_attention_backward",
                            "flex_attention_backward_dq",
                        ):
                            selected.append(choice_metadata.get(id(result[1])))
                            if automatic:
                                selected_automatic_options.append(
                                    automatic_options_by_choice.get(id(result[1]))
                                )
                        return result

                    def invoke():
                        out = compiled(q, k, v)
                        return out, torch.autograd.grad(
                            out, inputs, do, retain_graph=True
                        )

                    with (
                        mock.patch.object(
                            PyCodeCache, "load_by_key_path", side_effect=retain_module
                        ),
                        mock.patch.object(
                            template, "maybe_append_choice", side_effect=append_choice
                        ),
                        mock.patch.object(
                            flex_kernel_mod,
                            "autotune_select_algorithm",
                            side_effect=select_algorithm,
                        ),
                    ):
                        (out, initial), code = run_and_get_code(invoke)
                    torch.cuda.synchronize(device)
                    active = requested and fallback is None
                    phases = [1, 2] if active else [0]
                    self.assertEqual(
                        selected,
                        [(phase, 1, active and phase == 1) for phase in phases],
                    )
                    self.assertEqual(
                        [item[0][0] for item in choices], [0, 1, 2] if active else [0]
                    )
                    if automatic:
                        self.assertEqual(len(selected_automatic_options), len(phases))
                        for phase, observed in zip(
                            phases, selected_automatic_options, strict=True
                        ):
                            self.assertIsNotNone(observed)
                            self.assertEqual(observed["ROCM_BWD_PHASE"], phase)
                            self.assertEqual(
                                tuple(
                                    observed[key]
                                    for key in (
                                        "BLOCK_M1",
                                        "BLOCK_N1",
                                        "BLOCK_M2",
                                        "BLOCK_N2",
                                    )
                                ),
                                (64, 64, 64, 64) if phase == 0 else (16, 64, 128, 32),
                            )
                            self.assertEqual(
                                (
                                    observed["num_warps"],
                                    observed["num_stages"],
                                    observed["matrix_instr_nonkdim"],
                                ),
                                (
                                    4,
                                    1 if phase == 0 else 2,
                                    16 if phase in (1, 2) and length == 4096 else 0,
                                ),
                            )
                            self.assertEqual(
                                (observed["waves_per_eu"], observed["kpack"]), (0, 1)
                            )
                            self.assertIs(observed["ROCM_D192_PANEL_DOTS"], phase == 1)
                            self.assertIs(observed["ROCM_D192_DQ_EARLY_DP"], phase == 2)
                    for (phase, splits, panels), count, mutated, compiler in choices:
                        self.assertEqual(splits, 1)
                        self.assertEqual(panels, active and phase == 1)
                        self.assertEqual(count, 16)
                        self.assertEqual(
                            mutated, [7] if phase == 1 else [] if phase == 2 else [6, 7]
                        )
                        self.assertEqual(
                            compiler,
                            dict(
                                num_warps=4,
                                num_stages=1 if automatic and phase == 0 else 2,
                                matrix_instr_nonkdim=16
                                if automatic and phase in (1, 2) and length == 4096
                                else 0,
                                waves_per_eu=0,
                                kpack=1,
                            ),
                        )
                    for value, shape in zip(
                        (out, *initial), (shapes[0], *shapes), strict=True
                    ):
                        self.assertEqual(value.dtype, torch.bfloat16)
                        self.assertEqual(tuple(value.shape), shape)
                    actual_phases = set()
                    for part in code:
                        wrapper = ast.parse(part)
                        bodies = [
                            part,
                            *(
                                node.value
                                for node in ast.walk(wrapper)
                                if isinstance(node, ast.Constant)
                                and isinstance(node.value, str)
                                and "flex_attention_backward" in node.value
                            ),
                        ]
                        for body in bodies:
                            parsed = ast.parse(body)
                            if not any(
                                isinstance(node, ast.FunctionDef)
                                and "flex_attention_backward" in node.name
                                for node in parsed.body
                            ):
                                continue
                            constants = {
                                node.target.id: node.value.value
                                for node in ast.walk(parsed)
                                if isinstance(node, ast.AnnAssign)
                                and isinstance(node.target, ast.Name)
                                and isinstance(node.value, ast.Constant)
                            }
                            phase = constants.get("ROCM_BWD_PHASE")
                            if phase not in (0, 1, 2):
                                continue
                            actual_phases.add(phase)
                            self.assertEqual(constants.get("ROCM_GQA_SPLITS"), 1)
                            self.assertEqual(
                                constants.get("ROCM_D192_PANEL_DOTS"),
                                active and phase == 1,
                            )
                            self.assertEqual(constants.get("ROCM_XCD_SNAKE"), False)
                            self.assertEqual(
                                constants.get("ROCM_EARLY_PROBABILITY_COMPACT"), False
                            )
                            self.assertEqual(
                                constants.get("ROCM_D256_PANEL_DOTS"), False
                            )
                            if automatic:
                                self.assertIs(
                                    constants.get("ROCM_D192_DQ_EARLY_DP"), phase == 2
                                )
                            for key, value in dict(
                                BLOCK_M1=64 if automatic and phase == 0 else 16,
                                BLOCK_N1=64,
                                BLOCK_M2=64 if automatic and phase == 0 else 128,
                                BLOCK_N2=64 if automatic and phase == 0 else 32,
                                QK_HEAD_DIM=192,
                                V_HEAD_DIM=192,
                                QK_HEAD_DIM_ROUNDED=256,
                                V_HEAD_DIM_ROUNDED=256,
                                SAFE_HEAD_DIM=False,
                                GQA_SHARED_HEADS=gqa,
                            ).items():
                                self.assertEqual(constants.get(key), value)
                    self.assertTrue(set(phases).issubset(actual_phases))

                    def backward(out=out):
                        return torch.autograd.grad(out, inputs, do, retain_graph=True)

                    providers[requested] = dict(
                        compiled=compiled,
                        out=out,
                        backward=backward,
                        initial=initial,
                        code=code,
                        modules=tuple(modules),
                    )
                self._check_gfx950_backward_gradients_and_graphs(
                    device, stream, providers, inputs, do, original_do, scale, reference
                )
        finally:
            torch.cuda.current_stream(device).wait_stream(stream)

    @supported_platform
    @common_utils.parametrize("gqa", (1, 4))
    @common_utils.parametrize("length", (2048, 4096))
    @common_utils.parametrize("scale", (96**-0.5, 0.0, -0.125))
    def test_gfx950_d96_panel_phase(self, device, gqa, length, scale):
        self._run_gfx950_d96_panel_phase(device, gqa, length, scale)

    @supported_platform
    @largeTensorTest("16GB", inductor=True)
    @common_utils.parametrize("length", (2048, 4096))
    def test_gfx950_d96_panel_phase_automatic(self, device, length):
        import math

        self._run_gfx950_d96_panel_phase(
            device, 4, length, 1 / math.sqrt(96), automatic=True
        )

    @supported_platform
    @common_utils.parametrize("gqa", (1, 4))
    @common_utils.parametrize("fallback", ("short", "causal"))
    def test_gfx950_d96_panel_phase_fallback(self, device, gqa, fallback):
        self._run_gfx950_d96_panel_phase(device, gqa, 4096, 96**-0.5, fallback=fallback)

    def _run_gfx950_d96_panel_phase(
        self, device, gqa, length, scale, *, fallback=None, automatic=False
    ):
        import ast

        from torch._functorch import config as aot_config
        from torch._inductor.codecache import PyCodeCache
        from torch._inductor.kernel.flex import flex_attention as flex_kernel_mod

        if (
            not torch.version.hip
            or torch.device(device).type != "cuda"
            or torch.cuda.get_device_properties(device).gcnArchName.split(":", 1)[0]
            != "gfx950"
        ):
            self.skipTest("Requires gfx950")
        batch, q_heads, kv_heads = (4, 21, 21) if gqa == 1 else (2, 8, 2)
        q_len = kv_len = length
        if fallback == "short":
            batch, q_len, kv_len = 1, 257, 260
        gqa = q_heads // kv_heads
        generator = torch.Generator(device=device).manual_seed(
            20261016 if automatic else 20261008
        )
        shapes = (
            (batch, q_heads, q_len, 96),
            (batch, kv_heads, kv_len, 96),
            (batch, kv_heads, kv_len, 96),
        )
        inputs = tuple(
            (
                torch.randn(
                    shape, device=device, dtype=torch.bfloat16, generator=generator
                )
                * (1.0 if automatic else 0.5)
            ).requires_grad_()
            for shape in shapes
        )
        q, k, v = inputs
        do = torch.randn(
            shapes[0], device=device, dtype=torch.bfloat16, generator=generator
        ) * (1.0 if automatic else 0.25)
        original_do = do.clone()
        block_mask = None
        if fallback == "causal":
            block_mask = create_block_mask(
                lambda b, h, q_idx, kv_idx: q_idx >= kv_idx,
                None,
                None,
                q_len,
                kv_len,
                device=device,
            )

        def reference():
            reference_inputs = tuple(
                value.detach().float().requires_grad_() for value in inputs
            )
            ref_q, ref_k, ref_v = reference_inputs
            scores = (
                ref_q @ ref_k.repeat_interleave(gqa, dim=1).transpose(-1, -2)
            ) * scale
            if fallback == "causal":
                q_idx = torch.arange(q_len, device=device)[:, None]
                kv_idx = torch.arange(kv_len, device=device)[None, :]
                scores = scores.masked_fill(q_idx < kv_idx, float("-inf"))
            out = scores.softmax(-1) @ ref_v.repeat_interleave(gqa, dim=1)
            gradients = torch.autograd.grad(out, reference_inputs, do.float())
            return (out.detach(), *(gradient.detach() for gradient in gradients))

        providers = {}
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        try:
            with (
                torch.cuda.stream(stream),
                temp_float32_matmul_precision("highest"),
                config.patch(
                    fx_graph_cache=False,
                    max_autotune=False,
                    compile_threads=1,
                    emulate_precision_casts=False,
                ),
                aot_config.patch(
                    enable_autograd_cache=False, enable_remote_autograd_cache=False
                ),
            ):
                for requested in (False, True):
                    torch._dynamo.reset()
                    options = dict(
                        BACKEND="TRITON",
                        bwd_BLOCK_M1=32,
                        bwd_BLOCK_N1=128,
                        bwd_BLOCK_M2=128,
                        bwd_BLOCK_N2=32,
                        bwd_num_warps=4,
                        bwd_num_stages=2,
                        bwd_USE_TMA=False,
                        bwd_PRESCALE_QK=False,
                        bwd_ROCM_D256_PANEL_DOTS=False,
                        bwd_ROCM_SEPARATE_DQ=False,
                        bwd_ROCM_EARLY_DP=False,
                        bwd_ROCM_D96_PANEL_DOTS=requested,
                        bwd_ROCM_D192_PANEL_DOTS=False,
                        bwd_matrix_instr_nonkdim=0,
                        bwd_waves_per_eu=0,
                        bwd_kpack=1,
                        fwd_USE_TMA=False,
                        fwd_PRESCALE_QK=False,
                        fwd_BLOCK_M=128,
                        fwd_BLOCK_N=32,
                        fwd_num_warps=4,
                        fwd_num_stages=1,
                        bwd_XCD_REMAP=False,
                        ROCM_D96_PANEL_DOTS=True,
                        ROCM_BWD_PHASE=13,
                        bwd_ROCM_BWD_PHASE=17,
                        ROCM_GQA_SPLITS=7,
                        bwd_ROCM_GQA_SPLITS=4,
                    )

                    if automatic:
                        options = {"BACKEND": "TRITON"}
                        if not requested:
                            options["bwd_ROCM_D96_PANEL_DOTS"] = False

                    def forward(query, key, value, options=options):
                        return flex_attention(
                            query,
                            key,
                            value,
                            block_mask=block_mask,
                            enable_gqa=True,
                            scale=scale,
                            kernel_options=options,
                        )

                    compiled = torch.compile(forward, fullgraph=True, dynamic=False)
                    template = flex_kernel_mod.flex_attention_backward_template
                    original_append, original_select = (
                        template.maybe_append_choice,
                        flex_kernel_mod.autotune_select_algorithm,
                    )
                    choices, choice_metadata, selected, modules = [], {}, [], []
                    original_load = PyCodeCache.load_by_key_path

                    def retain_module(*args, **kwargs):
                        module = original_load(*args, **kwargs)
                        modules.append(module)
                        return module

                    def append_choice(*args, **kwargs):
                        choice_list, count = kwargs["choices"], len(kwargs["choices"])
                        metadata = (
                            kwargs["ROCM_BWD_PHASE"],
                            kwargs["ROCM_GQA_SPLITS"],
                            kwargs["ROCM_D96_PANEL_DOTS"],
                        )
                        choices.append(
                            (
                                metadata,
                                len(kwargs["input_nodes"]),
                                [
                                    index
                                    for index, value in enumerate(kwargs["input_nodes"])
                                    if any(
                                        value is target
                                        for target in kwargs["mutated_inputs"]
                                    )
                                ],
                                {
                                    key: kwargs[key]
                                    for key in (
                                        "num_warps",
                                        "num_stages",
                                        "matrix_instr_nonkdim",
                                        "waves_per_eu",
                                        "kpack",
                                        "ROCM_D192_PANEL_DOTS",
                                        "ROCM_D256_PANEL_DOTS",
                                    )
                                },
                                {
                                    key: kwargs[key]
                                    for key in (
                                        "ROCM_BWD_PHASE",
                                        "BLOCK_M1",
                                        "BLOCK_N1",
                                        "BLOCK_M2",
                                        "BLOCK_N2",
                                        "num_warps",
                                        "num_stages",
                                        "ROCM_D96_PANEL_DOTS",
                                        "ROCM_EARLY_DP",
                                        "XCD_REMAP",
                                    )
                                }
                                if automatic
                                else {},
                            )
                        )
                        result = original_append(*args, **kwargs)
                        for choice in choice_list[count:]:
                            choice_metadata[id(choice)] = metadata
                        return result

                    def select_algorithm(*args, **kwargs):
                        result = original_select(*args, **kwargs)
                        if args[0] in (
                            "flex_attention_backward",
                            "flex_attention_backward_dq",
                        ):
                            selected.append(choice_metadata.get(id(result[1])))
                        return result

                    def invoke():
                        out = compiled(q, k, v)
                        return out, torch.autograd.grad(
                            out, inputs, do, retain_graph=True
                        )

                    with (
                        mock.patch.object(
                            PyCodeCache, "load_by_key_path", side_effect=retain_module
                        ),
                        mock.patch.object(
                            template, "maybe_append_choice", side_effect=append_choice
                        ),
                        mock.patch.object(
                            flex_kernel_mod,
                            "autotune_select_algorithm",
                            side_effect=select_algorithm,
                        ),
                    ):
                        (out, initial), code = run_and_get_code(invoke)
                    torch.cuda.synchronize(device)
                    active = requested and fallback is None
                    phases = [1, 2] if active else [0]
                    self.assertEqual(
                        selected,
                        [
                            (phase, 1, active and phase == 1 and not automatic)
                            for phase in phases
                        ],
                    )
                    self.assertEqual(
                        [item[0][0] for item in choices], [0, 1, 2] if active else [0]
                    )
                    for (
                        phase,
                        splits,
                        panels,
                    ), count, mutated, compiler, automatic_options in choices:
                        if automatic:
                            phase = automatic_options["ROCM_BWD_PHASE"]
                            expected_tiles = (
                                (64, 64, 64, 64)
                                if phase == 0
                                else (
                                    (
                                        (32, 32, 128, 32)
                                        if length == 2048
                                        else (64, 64, 128, 32)
                                    )
                                    if phase == 1
                                    else (32, 128, 128, 32)
                                )
                            )
                            self.assertEqual(
                                tuple(
                                    automatic_options[name]
                                    for name in (
                                        "BLOCK_M1",
                                        "BLOCK_N1",
                                        "BLOCK_M2",
                                        "BLOCK_N2",
                                    )
                                ),
                                expected_tiles,
                            )
                            self.assertEqual(
                                (
                                    automatic_options["num_warps"],
                                    automatic_options["num_stages"],
                                ),
                                (4, 2 if phase == 2 else 1),
                            )
                            self.assertFalse(automatic_options["ROCM_D96_PANEL_DOTS"])
                            self.assertFalse(automatic_options["ROCM_EARLY_DP"])
                            self.assertFalse(automatic_options["XCD_REMAP"])
                        self.assertEqual(splits, 1)
                        self.assertEqual(
                            panels, active and phase == 1 and not automatic
                        )
                        self.assertEqual(count, 16)
                        self.assertEqual(
                            mutated, [7] if phase == 1 else [] if phase == 2 else [6, 7]
                        )
                        self.assertEqual(
                            compiler,
                            dict(
                                num_warps=4,
                                num_stages=1 if automatic and phase != 2 else 2,
                                matrix_instr_nonkdim=0,
                                waves_per_eu=0,
                                kpack=1,
                                ROCM_D192_PANEL_DOTS=False,
                                ROCM_D256_PANEL_DOTS=False,
                            ),
                        )
                    for value, shape in zip(
                        (out, *initial), (shapes[0], *shapes), strict=True
                    ):
                        self.assertEqual(value.dtype, torch.bfloat16)
                        self.assertEqual(tuple(value.shape), shape)
                    actual_phases = set()
                    for part in code:
                        wrapper = ast.parse(part)
                        bodies = [
                            part,
                            *(
                                node.value
                                for node in ast.walk(wrapper)
                                if isinstance(node, ast.Constant)
                                and isinstance(node.value, str)
                                and "flex_attention_backward" in node.value
                            ),
                        ]
                        for body in bodies:
                            parsed = ast.parse(body)
                            if not any(
                                isinstance(node, ast.FunctionDef)
                                and "flex_attention_backward" in node.name
                                for node in parsed.body
                            ):
                                continue
                            constants = {
                                node.target.id: node.value.value
                                for node in ast.walk(parsed)
                                if isinstance(node, ast.AnnAssign)
                                and isinstance(node.target, ast.Name)
                                and isinstance(node.value, ast.Constant)
                            }
                            phase = constants.get("ROCM_BWD_PHASE")
                            if phase not in (0, 1, 2):
                                continue
                            actual_phases.add(phase)
                            self.assertEqual(constants.get("ROCM_GQA_SPLITS"), 1)
                            self.assertEqual(
                                constants.get("ROCM_D96_PANEL_DOTS"),
                                active and phase == 1 and not automatic,
                            )
                            self.assertEqual(
                                constants.get("ROCM_D192_PANEL_DOTS"), False
                            )
                            self.assertEqual(constants.get("ROCM_XCD_SNAKE"), False)
                            self.assertEqual(
                                constants.get("ROCM_EARLY_PROBABILITY_COMPACT"), False
                            )
                            self.assertEqual(
                                constants.get("ROCM_D256_PANEL_DOTS"), False
                            )
                            for key, value in dict(
                                BLOCK_M1=(
                                    (32 if length == 2048 else 64)
                                    if automatic and phase == 1
                                    else 64
                                    if automatic and phase == 0
                                    else 32
                                ),
                                BLOCK_N1=(
                                    (32 if length == 2048 else 64)
                                    if automatic and phase == 1
                                    else 64
                                    if automatic and phase == 0
                                    else 128
                                ),
                                BLOCK_M2=64 if automatic and phase == 0 else 128,
                                BLOCK_N2=64 if automatic and phase == 0 else 32,
                                QK_HEAD_DIM=96,
                                V_HEAD_DIM=96,
                                QK_HEAD_DIM_ROUNDED=128,
                                V_HEAD_DIM_ROUNDED=128,
                                SAFE_HEAD_DIM=False,
                                GQA_SHARED_HEADS=gqa,
                            ).items():
                                self.assertEqual(constants.get(key), value)
                    self.assertTrue(set(phases).issubset(actual_phases))

                    def backward(out=out):
                        return torch.autograd.grad(out, inputs, do, retain_graph=True)

                    providers[requested] = dict(
                        compiled=compiled,
                        out=out,
                        backward=backward,
                        initial=initial,
                        code=code,
                        modules=tuple(modules),
                    )
                self._check_gfx950_backward_gradients_and_graphs(
                    device, stream, providers, inputs, do, original_do, scale, reference
                )
        finally:
            torch.cuda.current_stream(device).wait_stream(stream)

    def _run_gfx950_d256_early_probability_compact(
        self, device, gqa, scale, *, fallback=None
    ):
        import ast

        from torch._inductor.kernel.flex import flex_attention as flex_kernel_mod

        if (
            not torch.version.hip
            or torch.device(device).type != "cuda"
            or torch.cuda.get_device_properties(device).gcnArchName.split(":", 1)[0]
            != "gfx950"
        ):
            self.skipTest("Requires gfx950")
        q_len, kv_len, q_heads = 257, 260, 4
        generator = torch.Generator(device=device).manual_seed(20261008)
        shapes = (
            (1, q_heads, q_len, 256),
            (1, q_heads // gqa, kv_len, 256),
            (1, q_heads // gqa, kv_len, 256),
        )
        inputs = tuple(
            (
                torch.randn(
                    shape, device=device, dtype=torch.bfloat16, generator=generator
                )
                * 0.5
            ).requires_grad_()
            for shape in shapes
        )
        q, k, v = inputs
        do = (
            torch.randn(
                shapes[0], device=device, dtype=torch.bfloat16, generator=generator
            )
            * 0.25
        )
        original_do = do.clone()
        bias = torch.linspace(-0.1, 0.1, kv_len, device=device)
        limit = torch.tensor([251], device=device, dtype=torch.int32)

        def mask_mod(b, h, q_idx, kv_idx):
            if fallback == "explicit_dense":
                return torch.ones((), dtype=torch.bool, device=device)
            if fallback == "mask_capture":
                return kv_idx < limit[0]
            return q_idx >= kv_idx

        def score_mod(score, b, h, q_idx, kv_idx):
            return score + bias[kv_idx]

        block_mask = (
            create_block_mask(mask_mod, None, None, q_len, kv_len, device=device)
            if fallback in ("causal", "explicit_dense", "mask_capture")
            else None
        )

        def reference():
            reference_inputs = tuple(
                value.detach().float().requires_grad_() for value in inputs
            )
            ref_q, ref_k, ref_v = reference_inputs
            scores = (
                ref_q @ ref_k.repeat_interleave(gqa, dim=1).transpose(-1, -2)
            ) * scale
            if fallback == "score_capture":
                scores = scores + bias.float()[None, None, None, :]
            if fallback in ("causal", "mask_capture"):
                rows = torch.arange(q_len, device=device)[:, None]
                columns = torch.arange(kv_len, device=device)[None, :]
                valid = (
                    columns < limit[0]
                    if fallback == "mask_capture"
                    else rows >= columns
                )
                scores = scores.masked_fill(~valid, float("-inf"))
            out = scores.softmax(-1) @ ref_v.repeat_interleave(gqa, dim=1)
            gradients = torch.autograd.grad(out, reference_inputs, do.float())
            return (out.detach(), *(gradient.detach() for gradient in gradients))

        providers = {}
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        try:
            with torch.cuda.stream(stream), temp_float32_matmul_precision("highest"):
                for requested in (False, True):
                    torch._dynamo.reset()
                    options = dict(
                        BACKEND="TRITON",
                        bwd_BLOCK_M1=16,
                        bwd_BLOCK_N1=64,
                        bwd_BLOCK_M2=64 if fallback == "tiles" else 128,
                        bwd_BLOCK_N2=16 if fallback == "tiles" else 32,
                        bwd_num_warps=4,
                        bwd_num_stages=2,
                        bwd_USE_TMA=False,
                        bwd_PRESCALE_QK=False,
                        bwd_ROCM_D256_PANEL_DOTS=fallback != "panels",
                        bwd_ROCM_SEPARATE_DQ=fallback != "separate",
                        bwd_ROCM_EARLY_DP=fallback != "early_dp",
                        bwd_XCD_REMAP=False,
                        ROCM_EARLY_PROBABILITY_COMPACT=requested
                        if fallback == "raw"
                        else True,
                        ROCM_BWD_PHASE=13,
                        bwd_ROCM_BWD_PHASE=17,
                        ROCM_GQA_SPLITS=7,
                        bwd_ROCM_GQA_SPLITS=4,
                    )
                    if fallback == "forward_only":
                        options["fwd_ROCM_EARLY_PROBABILITY_COMPACT"] = requested
                    if requested and fallback not in ("raw", "forward_only"):
                        options["bwd_ROCM_EARLY_PROBABILITY_COMPACT"] = True

                    def forward(query, key, value, options=options):
                        return flex_attention(
                            query,
                            key,
                            value,
                            block_mask=block_mask,
                            score_mod=score_mod
                            if fallback == "score_capture"
                            else None,
                            enable_gqa=True,
                            scale=scale,
                            kernel_options=options,
                        )

                    compiled = torch.compile(forward, fullgraph=True, dynamic=False)
                    template = flex_kernel_mod.flex_attention_backward_template
                    original_append, original_select = (
                        template.maybe_append_choice,
                        flex_kernel_mod.autotune_select_algorithm,
                    )
                    choices, choice_metadata, selected = [], {}, []

                    def append_choice(*args, **kwargs):
                        choice_list, count = kwargs["choices"], len(kwargs["choices"])
                        metadata = (
                            kwargs["ROCM_BWD_PHASE"],
                            kwargs["ROCM_GQA_SPLITS"],
                            kwargs["ROCM_EARLY_PROBABILITY_COMPACT"],
                        )
                        choices.append(
                            (
                                metadata,
                                len(kwargs["input_nodes"]),
                                [
                                    index
                                    for index, value in enumerate(kwargs["input_nodes"])
                                    if any(
                                        value is target
                                        for target in kwargs["mutated_inputs"]
                                    )
                                ],
                            )
                        )
                        result = original_append(*args, **kwargs)
                        for choice in choice_list[count:]:
                            choice_metadata[id(choice)] = metadata
                        return result

                    def select_algorithm(*args, **kwargs):
                        result = original_select(*args, **kwargs)
                        if args[0] in (
                            "flex_attention_backward",
                            "flex_attention_backward_dq",
                        ):
                            selected.append(choice_metadata.get(id(result[1])))
                        return result

                    def invoke():
                        out = compiled(q, k, v)
                        return out, torch.autograd.grad(
                            out, inputs, do, retain_graph=True
                        )

                    with (
                        mock.patch.object(
                            template, "maybe_append_choice", side_effect=append_choice
                        ),
                        mock.patch.object(
                            flex_kernel_mod,
                            "autotune_select_algorithm",
                            side_effect=select_algorithm,
                        ),
                    ):
                        (out, initial), code = run_and_get_code(invoke)
                    torch.cuda.synchronize(device)
                    separate = fallback not in (
                        "causal",
                        "explicit_dense",
                        "score_capture",
                        "mask_capture",
                        "panels",
                        "separate",
                    )
                    active = requested and fallback is None
                    phases = [1, 2] if separate else [0]
                    self.assertEqual(
                        selected,
                        [(phase, 1, active and phase == 1) for phase in phases],
                    )
                    self.assertEqual(
                        [item[0][0] for item in choices], [0, 1, 2] if separate else [0]
                    )
                    for (phase, splits, compact), count, mutated in choices:
                        self.assertEqual(splits, 1)
                        self.assertEqual(compact, active and phase == 1)
                        self.assertEqual(count, 16)
                        self.assertEqual(
                            mutated, [7] if phase == 1 else [] if phase == 2 else [6, 7]
                        )
                    actual_phases = set()
                    for part in code:
                        wrapper = ast.parse(part)
                        bodies = [
                            part,
                            *(
                                node.value
                                for node in ast.walk(wrapper)
                                if isinstance(node, ast.Constant)
                                and isinstance(node.value, str)
                                and "flex_attention_backward" in node.value
                            ),
                        ]
                        for body in bodies:
                            parsed = ast.parse(body)
                            if not any(
                                isinstance(node, ast.FunctionDef)
                                and "flex_attention_backward" in node.name
                                for node in parsed.body
                            ):
                                continue
                            constants = {
                                node.target.id: node.value.value
                                for node in ast.walk(parsed)
                                if isinstance(node, ast.AnnAssign)
                                and isinstance(node.target, ast.Name)
                                and isinstance(node.value, ast.Constant)
                            }
                            phase = constants.get("ROCM_BWD_PHASE")
                            if phase not in (0, 1, 2):
                                continue
                            actual_phases.add(phase)
                            self.assertEqual(constants.get("ROCM_GQA_SPLITS"), 1)
                            self.assertEqual(
                                constants.get("ROCM_EARLY_PROBABILITY_COMPACT"),
                                active and phase == 1,
                            )
                            self.assertEqual(
                                "ppT = pT.to(MATMUL_PRECISION)" in body,
                                active and phase == 1,
                            )
                    self.assertTrue(set(phases).issubset(actual_phases))

                    def backward(out=out):
                        return torch.autograd.grad(out, inputs, do, retain_graph=True)

                    providers[requested] = dict(
                        compiled=compiled,
                        out=out,
                        backward=backward,
                        initial=initial,
                        code=code,
                    )
                self._check_gfx950_backward_gradients_and_graphs(
                    device, stream, providers, inputs, do, original_do, scale, reference
                )
        finally:
            torch.cuda.current_stream(device).wait_stream(stream)

    @supported_platform
    @common_utils.parametrize("length", (2048, 4096))
    @common_utils.parametrize("scale", (96**-0.5, 0.0, -0.125))
    def test_gfx950_d96_dq_unroll2(self, device, length, scale):
        self._run_gfx950_d96_dq_unroll2(device, length, scale)

    @supported_platform
    @common_utils.parametrize(
        "fallback",
        (
            "sequence1024",
            "gqa4",
            "contiguous",
            "raw",
            "forward_only",
            "score_capture",
            "mask_capture",
            "early_disabled",
        ),
    )
    def test_gfx950_d96_dq_unroll2_fallback(self, device, fallback):
        self._run_gfx950_d96_dq_unroll2(device, 4096, 96**-0.5, fallback=fallback)

    def _run_gfx950_d96_dq_unroll2(self, device, length, scale, *, fallback=None):
        import ast

        from torch._functorch import config as aot_config
        from torch._inductor.codecache import PyCodeCache
        from torch._inductor.kernel.flex import flex_attention as flex_kernel_mod

        if (
            not torch.version.hip
            or torch.device(device).type != "cuda"
            or torch.cuda.get_device_properties(device).gcnArchName.split(":", 1)[0]
            != "gfx950"
        ):
            self.skipTest("Requires gfx950")
        batch, q_heads, kv_heads, q_len, kv_len = 4, 21, 21, length, length
        if fallback == "short":
            batch, q_len, kv_len = 1, 257, 260
        elif fallback == "gqa4":
            batch, q_heads, kv_heads, q_len, kv_len = 2, 8, 2, 4096, 4096
        elif fallback == "sequence1024":
            q_len = kv_len = 1024
        gqa = q_heads // kv_heads
        bias = (
            torch.linspace(-0.1, 0.1, kv_len, device=device)
            if fallback == "score_capture"
            else None
        )
        limit = (
            torch.tensor(kv_len // 2, device=device)
            if fallback == "mask_capture"
            else None
        )
        score_mod = (
            (lambda score, b, h, q_idx, kv_idx: score + bias[kv_idx])
            if bias is not None
            else None
        )
        mask_mod = (
            (lambda b, h, q_idx, kv_idx: q_idx >= kv_idx)
            if fallback == "causal"
            else (lambda b, h, q_idx, kv_idx: kv_idx < limit)
            if limit is not None
            else None
        )
        block_mask = (
            create_block_mask(mask_mod, batch, q_heads, q_len, kv_len, device=device)
            if mask_mod is not None
            else None
        )
        generator = torch.Generator(device=device).manual_seed(20261008)
        shapes = (
            (batch, q_heads, q_len, 96),
            (batch, kv_heads, kv_len, 96),
            (batch, kv_heads, kv_len, 96),
        )
        inputs = tuple(
            (
                torch.randn(
                    shape, device=device, dtype=torch.bfloat16, generator=generator
                )
                * 0.5
            ).requires_grad_()
            for shape in shapes
        )
        q, k, v = inputs
        do = (
            torch.randn(
                shapes[0], device=device, dtype=torch.bfloat16, generator=generator
            )
            * 0.25
        )
        original_do = do.clone()

        def reference():
            reference_inputs = tuple(
                value.detach().float().requires_grad_() for value in inputs
            )
            ref_q, ref_k, ref_v = reference_inputs
            scores = (
                ref_q @ ref_k.repeat_interleave(gqa, dim=1).transpose(-1, -2)
            ) * scale
            if bias is not None:
                scores = scores + bias.float()[None, None, None, :]
            if limit is not None:
                cols = torch.arange(kv_len, device=device)
                scores = scores.masked_fill(
                    cols[None, None, None, :] >= limit, float("-inf")
                )
            if fallback == "causal":
                rows = torch.arange(q_len, device=device)
                cols = torch.arange(kv_len, device=device)
                scores = scores.masked_fill(
                    rows[None, None, :, None] < cols[None, None, None, :],
                    float("-inf"),
                )
            out = scores.softmax(-1) @ ref_v.repeat_interleave(gqa, dim=1)
            gradients = torch.autograd.grad(out, reference_inputs, do.float())
            return (out.detach(), *(gradient.detach() for gradient in gradients))

        providers = {}
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        try:
            with (
                torch.cuda.stream(stream),
                temp_float32_matmul_precision("highest"),
                config.patch(
                    fx_graph_cache=False,
                    max_autotune=False,
                    compile_threads=1,
                    emulate_precision_casts=False,
                ),
                aot_config.patch(
                    enable_autograd_cache=False, enable_remote_autograd_cache=False
                ),
            ):
                for requested in (False, True):
                    torch._dynamo.reset()
                    options = dict(
                        BACKEND="TRITON",
                        bwd_BLOCK_M1=32,
                        bwd_BLOCK_N1=128,
                        bwd_BLOCK_M2=128,
                        bwd_BLOCK_N2=32,
                        bwd_num_warps=4,
                        bwd_num_stages=2,
                        bwd_USE_TMA=False,
                        bwd_PRESCALE_QK=False,
                        bwd_ROCM_D256_PANEL_DOTS=False,
                        bwd_ROCM_SEPARATE_DQ=False,
                        bwd_ROCM_EARLY_DP=False,
                        bwd_ROCM_D96_PANEL_DOTS=fallback != "missing_panel",
                        bwd_ROCM_D96_DQ_EARLY_DP=fallback != "early_disabled",
                        bwd_ROCM_D96_DQ_UNROLL2=requested
                        if fallback not in ("raw", "forward_only")
                        else False,
                        ROCM_D96_DQ_UNROLL2=requested if fallback == "raw" else False,
                        fwd_ROCM_D96_DQ_UNROLL2=requested
                        if fallback == "forward_only"
                        else False,
                        bwd_BLOCKS_ARE_CONTIGUOUS=fallback != "contiguous",
                        bwd_ROCM_D192_PANEL_DOTS=False,
                        bwd_ROCM_D192_DQ_EARLY_DP=False,
                        bwd_matrix_instr_nonkdim=0,
                        bwd_waves_per_eu=0,
                        bwd_kpack=1,
                        fwd_USE_TMA=False,
                        fwd_PRESCALE_QK=False,
                        fwd_BLOCK_M=128,
                        fwd_BLOCK_N=32,
                        fwd_num_warps=4,
                        fwd_num_stages=1,
                        bwd_XCD_REMAP=False,
                        ROCM_D96_PANEL_DOTS=True,
                        ROCM_BWD_PHASE=13,
                        bwd_ROCM_BWD_PHASE=17,
                        ROCM_GQA_SPLITS=7,
                        bwd_ROCM_GQA_SPLITS=4,
                    )

                    def forward(query, key, value, options=options):
                        return flex_attention(
                            query,
                            key,
                            value,
                            enable_gqa=True,
                            score_mod=score_mod,
                            block_mask=block_mask,
                            scale=scale,
                            kernel_options=options,
                        )

                    compiled = torch.compile(forward, fullgraph=True, dynamic=False)
                    template = flex_kernel_mod.flex_attention_backward_template
                    original_append, original_select = (
                        template.maybe_append_choice,
                        flex_kernel_mod.autotune_select_algorithm,
                    )
                    choices, choice_metadata, selected, modules = [], {}, [], []
                    original_load = PyCodeCache.load_by_key_path

                    def retain_module(*args, **kwargs):
                        module = original_load(*args, **kwargs)
                        modules.append(module)
                        return module

                    def append_choice(*args, **kwargs):
                        choice_list, count = kwargs["choices"], len(kwargs["choices"])
                        metadata = (
                            kwargs["ROCM_BWD_PHASE"],
                            kwargs["ROCM_GQA_SPLITS"],
                            kwargs["ROCM_D96_PANEL_DOTS"],
                            kwargs["ROCM_D96_DQ_UNROLL2"],
                        )
                        choices.append(
                            (
                                metadata,
                                len(kwargs["input_nodes"]),
                                [
                                    index
                                    for index, value in enumerate(kwargs["input_nodes"])
                                    if any(
                                        value is target
                                        for target in kwargs["mutated_inputs"]
                                    )
                                ],
                                {
                                    key: kwargs[key]
                                    for key in (
                                        "num_warps",
                                        "num_stages",
                                        "matrix_instr_nonkdim",
                                        "waves_per_eu",
                                        "kpack",
                                    )
                                },
                            )
                        )
                        result = original_append(*args, **kwargs)
                        for choice in choice_list[count:]:
                            choice_metadata[id(choice)] = metadata
                        return result

                    def select_algorithm(*args, **kwargs):
                        result = original_select(*args, **kwargs)
                        if args[0] in (
                            "flex_attention_backward",
                            "flex_attention_backward_dq",
                        ):
                            selected.append(choice_metadata.get(id(result[1])))
                        return result

                    def invoke():
                        out = compiled(q, k, v)
                        return out, torch.autograd.grad(
                            out, inputs, do, retain_graph=True
                        )

                    with (
                        mock.patch.object(
                            PyCodeCache, "load_by_key_path", side_effect=retain_module
                        ),
                        mock.patch.object(
                            template, "maybe_append_choice", side_effect=append_choice
                        ),
                        mock.patch.object(
                            flex_kernel_mod,
                            "autotune_select_algorithm",
                            side_effect=select_algorithm,
                        ),
                    ):
                        (out, initial), code = run_and_get_code(invoke)
                    torch.cuda.synchronize(device)
                    panel_active = fallback not in (
                        "sequence1024",
                        "short",
                        "causal",
                        "missing_panel",
                        "score_capture",
                        "mask_capture",
                    )
                    active = (
                        requested
                        and panel_active
                        and fallback
                        not in (
                            "raw",
                            "forward_only",
                            "gqa4",
                            "sequence1024",
                            "contiguous",
                            "early_disabled",
                        )
                    )
                    common_early = panel_active and fallback not in (
                        "gqa4",
                        "sequence1024",
                        "contiguous",
                        "early_disabled",
                    )
                    phases = [1, 2] if panel_active else [0]
                    self.assertEqual(
                        selected,
                        [
                            (
                                phase,
                                1,
                                panel_active and phase == 1,
                                active and phase == 2,
                            )
                            for phase in phases
                        ],
                    )
                    self.assertEqual(
                        [item[0][0] for item in choices],
                        [0, 1, 2] if panel_active else [0],
                    )
                    for (
                        phase,
                        splits,
                        panels,
                        early_dp,
                    ), count, mutated, compiler in choices:
                        self.assertEqual(splits, 1)
                        self.assertEqual(panels, panel_active and phase == 1)
                        self.assertEqual(early_dp, active and phase == 2)
                        self.assertEqual(count, 16)
                        self.assertEqual(
                            mutated, [7] if phase == 1 else [] if phase == 2 else [6, 7]
                        )
                        self.assertEqual(
                            {
                                key: value
                                for key, value in compiler.items()
                                if key != "kpack"
                            },
                            dict(
                                num_warps=4,
                                num_stages=2,
                                matrix_instr_nonkdim=0,
                                waves_per_eu=0,
                            ),
                        )
                    self.assertTrue(
                        all(item[-1]["kpack"] in (1, 2) for item in choices)
                    )
                    for value, shape in zip(
                        (out, *initial), (shapes[0], *shapes), strict=True
                    ):
                        self.assertEqual(value.dtype, torch.bfloat16)
                        self.assertEqual(tuple(value.shape), shape)
                    actual_phases = set()
                    for part in code:
                        wrapper = ast.parse(part)
                        bodies = [
                            part,
                            *(
                                node.value
                                for node in ast.walk(wrapper)
                                if isinstance(node, ast.Constant)
                                and isinstance(node.value, str)
                                and "flex_attention_backward" in node.value
                            ),
                        ]
                        for body in bodies:
                            parsed = ast.parse(body)
                            if not any(
                                isinstance(node, ast.FunctionDef)
                                and "flex_attention_backward" in node.name
                                for node in parsed.body
                            ):
                                continue
                            constants = {
                                node.target.id: node.value.value
                                for node in ast.walk(parsed)
                                if isinstance(node, ast.AnnAssign)
                                and isinstance(node.target, ast.Name)
                                and isinstance(node.value, ast.Constant)
                            }
                            phase = constants.get("ROCM_BWD_PHASE")
                            if phase not in (0, 1, 2):
                                continue
                            actual_phases.add(phase)
                            self.assertEqual(constants.get("ROCM_GQA_SPLITS"), 1)
                            self.assertEqual(
                                constants.get("ROCM_D96_PANEL_DOTS"),
                                panel_active and phase == 1,
                            )
                            self.assertEqual(
                                constants.get("ROCM_D96_DQ_UNROLL2"),
                                active and phase == 2,
                            )
                            self.assertEqual(
                                constants.get("ROCM_D96_DQ_EARLY_DP"),
                                common_early and phase == 2,
                            )
                            self.assertEqual(
                                constants.get("ROCM_D192_PANEL_DOTS"), False
                            )
                            self.assertEqual(
                                constants.get("BLOCKS_ARE_CONTIGUOUS"),
                                fallback != "contiguous",
                            )
                            self.assertEqual(constants.get("ROCM_XCD_SNAKE"), False)
                            self.assertEqual(
                                constants.get("ROCM_EARLY_PROBABILITY_COMPACT"), False
                            )
                            self.assertEqual(
                                constants.get("ROCM_D256_PANEL_DOTS"), False
                            )
                            for key, value in dict(
                                BLOCK_M1=32,
                                BLOCK_N1=128,
                                BLOCK_M2=128,
                                BLOCK_N2=32,
                                QK_HEAD_DIM=96,
                                V_HEAD_DIM=96,
                                QK_HEAD_DIM_ROUNDED=128,
                                V_HEAD_DIM_ROUNDED=128,
                                SAFE_HEAD_DIM=False,
                                GQA_SHARED_HEADS=gqa,
                            ).items():
                                self.assertEqual(constants.get(key), value)
                    self.assertTrue(set(phases).issubset(actual_phases))

                    def backward(out=out):
                        return torch.autograd.grad(out, inputs, do, retain_graph=True)

                    providers[requested] = dict(
                        compiled=compiled,
                        out=out,
                        backward=backward,
                        initial=initial,
                        code=code,
                        modules=tuple(modules),
                    )
                self._check_gfx950_backward_gradients_and_graphs(
                    device, stream, providers, inputs, do, original_do, scale, reference
                )
        finally:
            torch.cuda.current_stream(device).wait_stream(stream)

    @supported_platform
    @common_utils.parametrize(
        "batch,length", ((4, 1024), (4, 2048), (4, 4096), (2, 4096))
    )
    @common_utils.parametrize("scale", (128**-0.5, 0.0, -0.125))
    def test_gfx950_d128_dv_view_carry_split64(self, device, batch, length, scale):
        self._run_gfx950_d128_dv_view_carry_split64(device, length, scale, batch=batch)

    @supported_platform
    @common_utils.parametrize(
        "fallback",
        (
            "short",
            "gqa1",
            "xcd_false",
            "reverse_false",
            "raw",
            "forward_only",
            "score_capture",
            "mask_capture",
        ),
    )
    def test_gfx950_d128_dv_view_carry_split64_fallback(self, device, fallback):
        self._run_gfx950_d128_dv_view_carry_split64(
            device, 4096, 128**-0.5, fallback=fallback
        )

    @supported_platform
    @common_utils.parametrize(
        "dtype,batch,length,default_scale",
        (
            (torch.float16, 4, 2048, 128**-0.5),
            (torch.float16, 4, 4096, 128**-0.5),
            (torch.float16, 2, 4096, 128**-0.5),
            (torch.float16, 2, 2048, 128**-0.5),
            (torch.float16, 1, 4096, 128**-0.5),
            (torch.float32, 4, 2048, 128**-0.5),
            (torch.float32, 4, 4096, 1 / math.sqrt(128)),
            (torch.float32, 2, 4096, 1 / math.sqrt(128)),
            (torch.float32, 1, 4096, 1 / math.sqrt(128)),
            (torch.float32, 1, 2048, 1 / math.sqrt(128)),
            (torch.float32, 2, 2048, 1 / math.sqrt(128)),
        ),
    )
    @common_utils.parametrize("scale", (None, 0.0, -0.125))
    def test_gfx950_d128_dv_view_carry_split64_dtype_coverage(
        self, device, dtype, batch, length, default_scale, scale
    ):
        self._run_gfx950_d128_dv_view_carry_split64(
            device,
            length,
            default_scale if scale is None else scale,
            batch=batch,
            dtype=dtype,
            fp16_scope=dtype == torch.float16,
            fp32_balanced64_scope=dtype == torch.float32,
        )

    @supported_platform
    @common_utils.parametrize("dtype", (torch.float16, torch.float32))
    @common_utils.parametrize("scale", (0.08838834764831843,))
    def test_gfx950_d128_query_tile_outer_qualified_default(self, device, dtype, scale):
        self._run_gfx950_d128_dv_view_carry_split64(
            device,
            4096,
            scale,
            dtype=dtype,
            fp16_scope=dtype == torch.float16,
            fp32_balanced64_scope=dtype == torch.float32,
        )

    @supported_platform
    @common_utils.parametrize(
        "dtype,batch,default_scale",
        (
            (torch.float16, 4, 0.08838834764831843),
            (torch.float32, 4, 0.08838834764831843),
            (torch.float32, 2, 0.08838834764831843),
            (torch.float32, 1, 0.08838834764831843),
        ),
    )
    @common_utils.parametrize("scale", (None, 0.0, -0.125))
    def test_gfx950_d128_query_tile_outer_expanded_metadata(
        self, device, dtype, batch, default_scale, scale
    ):
        self._run_gfx950_d128_dv_view_carry_split64(
            device,
            4096,
            default_scale if scale is None else scale,
            batch=batch,
            dtype=dtype,
            fp16_scope=dtype == torch.float16,
            fp32_balanced64_scope=dtype == torch.float32,
            expanded_metadata=True,
        )

    @supported_platform
    @common_utils.parametrize("scale", (0.08838834764831843, 0.0, -0.125))
    @common_utils.parametrize("expanded_metadata", (False, True))
    def test_gfx950_d128_query_tile_outer_fp32_n2048(
        self, device, scale, expanded_metadata
    ):
        self._run_gfx950_d128_dv_view_carry_split64(
            device,
            2048,
            scale,
            dtype=torch.float32,
            fp32_balanced64_scope=True,
            expanded_metadata=expanded_metadata,
        )

    @supported_platform
    @common_utils.parametrize("dtype", (torch.bfloat16, torch.float16, torch.float32))
    def test_gfx950_d128_automatic_backward_profile(self, device, dtype):
        self._run_gfx950_d128_dv_view_carry_split64(
            device,
            4096,
            1 / math.sqrt(128),
            dtype=dtype,
            fp16_scope=dtype == torch.float16,
            fp32_balanced64_scope=dtype == torch.float32,
            automatic=True,
        )

    @supported_platform
    @common_utils.parametrize("dtype", (torch.bfloat16, torch.float16, torch.float32))
    def test_gfx950_d128_automatic_backward_profile_sparse_gap(self, device, dtype):
        self._run_gfx950_d128_dv_view_carry_split64(
            device,
            4096,
            1 / math.sqrt(128),
            dtype=dtype,
            fp16_scope=dtype == torch.float16,
            fp32_balanced64_scope=dtype == torch.float32,
            automatic=True,
            indexed_gap=True,
        )

    def _run_gfx950_d128_dv_view_carry_split64(
        self,
        device,
        length,
        scale,
        *,
        fallback=None,
        batch=4,
        dtype=torch.bfloat16,
        fp16_scope=False,
        fp32_balanced64_scope=False,
        expanded_metadata=False,
        automatic=False,
        indexed_gap=False,
    ):
        import ast

        from torch._functorch import config as aot_config
        from torch._inductor.codecache import PyCodeCache
        from torch._inductor.kernel.flex import flex_attention as flex_kernel_mod

        if (
            not torch.version.hip
            or torch.device(device).type != "cuda"
            or torch.cuda.get_device_properties(device).gcnArchName.split(":", 1)[0]
            != "gfx950"
        ):
            self.skipTest("Requires gfx950")
        q_heads, kv_heads, q_len, kv_len = 64, 8, length, length
        if fallback == "short":
            batch, q_len, kv_len = 1, 257, 260
        if fallback == "gqa1":
            q_heads = kv_heads
        gqa = q_heads // kv_heads
        bias = (
            torch.linspace(-0.1, 0.1, kv_len, device=device)
            if fallback == "score_capture"
            else None
        )
        limit = (
            torch.tensor(kv_len // 2, device=device)
            if fallback == "mask_capture"
            else None
        )
        score_mod = (
            (lambda score, b, h, q_idx, kv_idx: score + bias[kv_idx])
            if bias is not None
            else None
        )
        mask_mod = (
            (lambda b, h, q_idx, kv_idx: (q_idx >= kv_idx) & (kv_idx < limit))
            if limit is not None
            else (lambda b, h, q_idx, kv_idx: q_idx >= kv_idx)
        )
        block_mask = create_block_mask(
            mask_mod,
            batch if expanded_metadata else None,
            q_heads if expanded_metadata else None,
            q_len,
            kv_len,
            device=device,
            BLOCK_SIZE=128,
        )
        if indexed_gap:
            full_num_blocks = block_mask.full_kv_num_blocks.clone()
            full_indices = block_mask.full_kv_indices.clone()
            full_indices[..., 4, 1:3] = full_indices[..., 4, 2:4].clone()
            full_num_blocks[..., 4] -= 1
            block_mask = BlockMask.from_kv_blocks(
                block_mask.kv_num_blocks,
                block_mask.kv_indices,
                full_num_blocks,
                full_indices,
                BLOCK_SIZE=128,
                mask_mod=mask_mod,
                seq_lengths=(q_len, kv_len),
            )
            query_count = int(block_mask.full_q_num_blocks[0, 0, 1])
            query_blocks = block_mask.full_q_indices[0, 0, 1, :query_count].tolist()
            self.assertEqual(query_blocks, [2, 3, *range(5, q_len // 128)])
        generator = torch.Generator(device=device).manual_seed(20261008)
        shapes = (
            (batch, q_heads, q_len, 128),
            (batch, kv_heads, kv_len, 128),
            (batch, kv_heads, kv_len, 128),
        )
        inputs = tuple(
            (
                torch.randn(shape, device=device, dtype=dtype, generator=generator)
                * 0.5
            ).requires_grad_()
            for shape in shapes
        )
        q, k, v = inputs
        do = (
            torch.randn(shapes[0], device=device, dtype=dtype, generator=generator)
            * 0.25
        )
        original_do = do.clone()

        def reference():
            reference_inputs = tuple(
                value.detach().float().requires_grad_() for value in inputs
            )
            ref_q, ref_k, ref_v = reference_inputs
            scores = (
                ref_q @ ref_k.repeat_interleave(gqa, dim=1).transpose(-1, -2)
            ) * scale
            if bias is not None:
                scores = scores + bias.float()[None, None, None, :]
            rows = torch.arange(q_len, device=device)
            cols = torch.arange(kv_len, device=device)
            scores = scores.masked_fill(
                rows[None, None, :, None] < cols[None, None, None, :], float("-inf")
            )
            if indexed_gap:
                scores = scores.masked_fill(
                    (rows[None, None, :, None] // 128 == 4)
                    & (cols[None, None, None, :] // 128 == 1),
                    float("-inf"),
                )
            if limit is not None:
                cols = torch.arange(kv_len, device=device)
                scores = scores.masked_fill(
                    cols[None, None, None, :] >= limit, float("-inf")
                )
            out = scores.softmax(-1) @ ref_v.repeat_interleave(gqa, dim=1)
            gradients = torch.autograd.grad(out, reference_inputs, do.float())
            return (out.detach(), *(gradient.detach() for gradient in gradients))

        providers = {}
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        try:
            with (
                torch.cuda.stream(stream),
                temp_float32_matmul_precision("highest"),
                config.patch(
                    fx_graph_cache=False,
                    max_autotune=False,
                    compile_threads=1,
                    emulate_precision_casts=False,
                ),
                aot_config.patch(
                    enable_autograd_cache=False, enable_remote_autograd_cache=False
                ),
            ):
                for requested in (False, True):
                    torch._dynamo.reset()
                    options = dict(
                        BACKEND="TRITON",
                        bwd_BLOCK_M1=32,
                        bwd_BLOCK_N1=128,
                        bwd_BLOCK_M2=128,
                        bwd_BLOCK_N2=32,
                        bwd_num_warps=4,
                        bwd_num_stages=2,
                        bwd_USE_TMA=False,
                        bwd_PRESCALE_QK=False,
                        bwd_ROCM_D256_PANEL_DOTS=False,
                        bwd_ROCM_SEPARATE_DQ=True,
                        bwd_ROCM_EARLY_DP=True,
                        bwd_ROCM_D192_PANEL_DOTS=False,
                        bwd_ROCM_D192_DQ_EARLY_DP=False,
                        bwd_ROCM_D128_REVERSE_DQ_ROWS=fallback != "reverse_false",
                        bwd_ROCM_D128_DV_VIEW_CARRY_SPLIT64=requested
                        if fallback not in ("raw", "forward_only")
                        else False,
                        ROCM_D128_DV_VIEW_CARRY_SPLIT64=requested
                        if fallback == "raw"
                        else False,
                        fwd_ROCM_D128_DV_VIEW_CARRY_SPLIT64=requested
                        if fallback == "forward_only"
                        else False,
                        bwd_BLOCKS_ARE_CONTIGUOUS=not indexed_gap
                        and fallback != "short",
                        bwd_ROCM_D96_PANEL_DOTS=False,
                        bwd_matrix_instr_nonkdim=0,
                        bwd_waves_per_eu=0,
                        bwd_kpack=1,
                        fwd_USE_TMA=False,
                        fwd_PRESCALE_QK=False,
                        fwd_BLOCK_M=128,
                        fwd_BLOCK_N=32,
                        fwd_num_warps=4,
                        fwd_num_stages=1,
                        bwd_XCD_REMAP=fallback != "xcd_false",
                        ROCM_D192_PANEL_DOTS=False,
                        ROCM_BWD_PHASE=13,
                        bwd_ROCM_BWD_PHASE=17,
                        ROCM_GQA_SPLITS=7,
                        bwd_ROCM_GQA_SPLITS=4,
                    )

                    if fp16_scope:
                        options["bwd_ROCM_D128_FP16_COVERAGE"] = True
                    if fp32_balanced64_scope:
                        options.update(
                            bwd_ROCM_D128_FP32_BALANCED64_COVERAGE=True,
                            bwd_BLOCK_M1=16,
                            bwd_BLOCK_N1=64,
                            bwd_BLOCK_M2=64,
                            bwd_BLOCK_N2=16,
                            bwd_num_stages=1,
                        )
                    if automatic:
                        options = {"BACKEND": "TRITON"}
                        if not requested:
                            options["bwd_ROCM_SEPARATE_DQ"] = False

                    def forward(query, key, value, options=options):
                        return flex_attention(
                            query,
                            key,
                            value,
                            enable_gqa=True,
                            score_mod=score_mod,
                            block_mask=block_mask,
                            scale=scale,
                            kernel_options=options,
                        )

                    compiled = torch.compile(forward, fullgraph=True, dynamic=False)
                    template = flex_kernel_mod.flex_attention_backward_template
                    original_append, original_select = (
                        template.maybe_append_choice,
                        flex_kernel_mod.autotune_select_algorithm,
                    )
                    choices, choice_metadata, selected, modules = [], {}, [], []
                    original_load = PyCodeCache.load_by_key_path

                    def retain_module(*args, **kwargs):
                        module = original_load(*args, **kwargs)
                        modules.append(module)
                        return module

                    def append_choice(*args, **kwargs):
                        choice_list, count = kwargs["choices"], len(kwargs["choices"])
                        metadata = (
                            kwargs["ROCM_BWD_PHASE"],
                            kwargs["ROCM_GQA_SPLITS"],
                            kwargs["XCD_REMAP"],
                            kwargs["ROCM_D128_REVERSE_DQ_ROWS"],
                            kwargs["ROCM_D128_DV_VIEW_CARRY_SPLIT64"],
                            kwargs.get("ROCM_D128_GQA_QUERY_TILE_OUTER", False),
                        )
                        choices.append(
                            (
                                metadata,
                                len(kwargs["input_nodes"]),
                                [
                                    index
                                    for index, value in enumerate(kwargs["input_nodes"])
                                    if any(
                                        value is target
                                        for target in kwargs["mutated_inputs"]
                                    )
                                ],
                                {
                                    key: kwargs[key]
                                    for key in (
                                        "num_warps",
                                        "num_stages",
                                        "matrix_instr_nonkdim",
                                        "waves_per_eu",
                                        "kpack",
                                    )
                                },
                            )
                        )
                        result = original_append(*args, **kwargs)
                        for choice in choice_list[count:]:
                            choice_metadata[id(choice)] = metadata
                        return result

                    def select_algorithm(*args, **kwargs):
                        result = original_select(*args, **kwargs)
                        if args[0] in (
                            "flex_attention_backward",
                            "flex_attention_backward_dq",
                        ):
                            selected.append(choice_metadata.get(id(result[1])))
                        return result

                    def invoke():
                        out = compiled(q, k, v)
                        return out, torch.autograd.grad(
                            out, inputs, do, retain_graph=True
                        )

                    with (
                        mock.patch.object(
                            PyCodeCache, "load_by_key_path", side_effect=retain_module
                        ),
                        mock.patch.object(
                            template, "maybe_append_choice", side_effect=append_choice
                        ),
                        mock.patch.object(
                            flex_kernel_mod,
                            "autotune_select_algorithm",
                            side_effect=select_algorithm,
                        ),
                    ):
                        (out, initial), code = run_and_get_code(invoke)
                    torch.cuda.synchronize(device)
                    separate_active = (
                        requested
                        if automatic
                        else fallback
                        not in (
                            "score_capture",
                            "mask_capture",
                            "gqa1",
                        )
                    )
                    active = requested and fallback is None
                    reverse_active = (
                        requested
                        if automatic
                        else fallback
                        not in (
                            "short",
                            "xcd_false",
                            "score_capture",
                            "mask_capture",
                            "reverse_false",
                            "gqa1",
                        )
                    )
                    xcd_active = requested if automatic else fallback != "xcd_false"
                    phases = [1, 2] if separate_active else [0]
                    query_tile_outer_eligible = (
                        active
                        and not expanded_metadata
                        and scale in (0.08838834764831843, 0.0, -0.125)
                        and (
                            fp16_scope
                            and (batch, length)
                            in ((4, 2048), (4, 4096), (2, 2048), (2, 4096), (1, 4096))
                            or fp32_balanced64_scope
                            and (batch, length)
                            in ((4, 4096), (2, 4096), (1, 4096), (4, 2048))
                        )
                    )
                    query_tile_outer_active = query_tile_outer_eligible and any(
                        item[0][0] == 1 and item[-1]["kpack"] == 1 for item in choices
                    )
                    self.assertEqual(
                        selected,
                        [
                            (
                                phase,
                                1,
                                xcd_active,
                                reverse_active and phase == 2,
                                active and phase == 1,
                                query_tile_outer_active and phase == 1,
                            )
                            for phase in phases
                        ],
                    )
                    self.assertEqual(
                        [item[0][0] for item in choices],
                        [0, 1, 2] if separate_active else [0],
                    )
                    for (
                        phase,
                        splits,
                        xcd,
                        reverse_rows,
                        view_carry,
                        query_tile_outer,
                    ), count, mutated, compiler in choices:
                        self.assertEqual(splits, 1)
                        self.assertEqual(
                            xcd, xcd_active and (not automatic or phase != 0)
                        )
                        self.assertEqual(reverse_rows, reverse_active and phase == 2)
                        self.assertIs(view_carry, active and phase == 1)
                        self.assertIs(
                            query_tile_outer,
                            query_tile_outer_eligible
                            and phase == 1
                            and compiler["kpack"] == 1,
                        )
                        self.assertEqual(count, 16)
                        self.assertEqual(
                            mutated, [7] if phase == 1 else [] if phase == 2 else [6, 7]
                        )
                        if not automatic or phase != 0:
                            self.assertEqual(
                                {
                                    key: value
                                    for key, value in compiler.items()
                                    if key != "kpack"
                                },
                                dict(
                                    num_warps=4,
                                    num_stages=1 if fp32_balanced64_scope else 2,
                                    matrix_instr_nonkdim=0,
                                    waves_per_eu=0,
                                ),
                            )
                    if not automatic or requested:
                        self.assertTrue(
                            all(
                                item[-1]["kpack"]
                                in ((1,) if fp32_balanced64_scope else (1, 2))
                                for item in choices
                                if not automatic or item[0][0] != 0
                            )
                        )
                    for value, shape in zip(
                        (out, *initial), (shapes[0], *shapes), strict=True
                    ):
                        self.assertEqual(value.dtype, dtype)
                        self.assertEqual(tuple(value.shape), shape)
                    actual_phases = set()
                    for part in code:
                        wrapper = ast.parse(part)
                        bodies = [
                            part,
                            *(
                                node.value
                                for node in ast.walk(wrapper)
                                if isinstance(node, ast.Constant)
                                and isinstance(node.value, str)
                                and "flex_attention_backward" in node.value
                            ),
                        ]
                        for body in bodies:
                            parsed = ast.parse(body)
                            if not any(
                                isinstance(node, ast.FunctionDef)
                                and "flex_attention_backward" in node.name
                                for node in parsed.body
                            ):
                                continue
                            constants = {
                                node.target.id: node.value.value
                                for node in ast.walk(parsed)
                                if isinstance(node, ast.AnnAssign)
                                and isinstance(node.target, ast.Name)
                                and isinstance(node.value, ast.Constant)
                            }
                            phase = constants.get("ROCM_BWD_PHASE")
                            if fp32_balanced64_scope and phase in (0, 1, 2):
                                self.assertEqual(
                                    constants.get("FLOAT32_PRECISION"), "ieee"
                                )
                                self.assertIs(constants.get("PRESCALE_QK"), False)
                            if phase not in (0, 1, 2):
                                continue
                            actual_phases.add(phase)
                            self.assertEqual(constants.get("ROCM_GQA_SPLITS"), 1)
                            self.assertEqual(
                                constants.get("XCD_REMAP"),
                                xcd_active and (not automatic or phase != 0),
                            )
                            self.assertEqual(
                                constants.get("ROCM_D128_REVERSE_DQ_ROWS"),
                                reverse_active and phase == 2,
                            )
                            self.assertIs(
                                constants.get("ROCM_D128_DV_VIEW_CARRY_SPLIT64"),
                                active and phase == 1,
                            )
                            self.assertIs(
                                constants.get("ROCM_D128_GQA_QUERY_TILE_OUTER", False),
                                query_tile_outer_active and phase == 1,
                            )
                            self.assertEqual(
                                constants.get("ROCM_D96_PANEL_DOTS"), False
                            )
                            self.assertEqual(
                                constants.get("BLOCKS_ARE_CONTIGUOUS"),
                                False
                                if automatic or indexed_gap
                                else fallback != "short",
                            )
                            self.assertEqual(constants.get("ROCM_XCD_SNAKE"), False)
                            self.assertEqual(
                                constants.get("ROCM_EARLY_DP"),
                                requested and phase != 0 if automatic else True,
                            )
                            self.assertEqual(
                                constants.get("ROCM_D192_PANEL_DOTS"), False
                            )
                            self.assertEqual(
                                constants.get("ROCM_D192_DQ_EARLY_DP"), False
                            )
                            self.assertEqual(
                                constants.get("ROCM_D96_DQ_EARLY_DP"), False
                            )
                            self.assertEqual(
                                constants.get("ROCM_EARLY_PROBABILITY_COMPACT"), False
                            )
                            self.assertEqual(
                                constants.get("ROCM_D256_PANEL_DOTS"), False
                            )
                            for key, value in dict(
                                QK_HEAD_DIM=128,
                                V_HEAD_DIM=128,
                                QK_HEAD_DIM_ROUNDED=128,
                                V_HEAD_DIM_ROUNDED=128,
                                SAFE_HEAD_DIM=True,
                                GQA_SHARED_HEADS=gqa,
                            ).items():
                                self.assertEqual(constants.get(key), value)
                            if not automatic or phase != 0:
                                for key, value in dict(
                                    BLOCK_M1=16 if fp32_balanced64_scope else 32,
                                    BLOCK_N1=64 if fp32_balanced64_scope else 128,
                                    BLOCK_M2=64 if fp32_balanced64_scope else 128,
                                    BLOCK_N2=16 if fp32_balanced64_scope else 32,
                                ).items():
                                    self.assertEqual(constants.get(key), value)
                    self.assertTrue(set(phases).issubset(actual_phases))

                    def backward(out=out):
                        return torch.autograd.grad(out, inputs, do, retain_graph=True)

                    providers[requested] = dict(
                        compiled=compiled,
                        out=out,
                        backward=backward,
                        initial=initial,
                        code=code,
                        modules=tuple(modules),
                    )
                self._check_gfx950_backward_gradients_and_graphs(
                    device, stream, providers, inputs, do, original_do, scale, reference
                )
        finally:
            torch.cuda.current_stream(device).wait_stream(stream)

    @supported_platform
    @common_utils.parametrize("scale", (0.08838834764831843,))
    def test_gfx950_d128_query_tile_outer_fp16_B2_qualified_default(
        self, device, scale
    ):
        self._run_gfx950_d128_dv_view_carry_split64(
            device,
            4096,
            scale,
            batch=2,
            dtype=torch.float16,
            fp16_scope=True,
        )

    @supported_platform
    @common_utils.parametrize("scale", (0.08838834764831843, 0.0, -0.125))
    def test_gfx950_d128_query_tile_outer_fp16_B2_expanded_metadata(
        self, device, scale
    ):
        self._run_gfx950_d128_dv_view_carry_split64(
            device,
            4096,
            scale,
            batch=2,
            dtype=torch.float16,
            fp16_scope=True,
            expanded_metadata=True,
        )

    @supported_platform
    def test_gfx950_d128_query_tile_outer_fp16_B2_automatic_default(self, device):
        self._run_gfx950_d128_dv_view_carry_split64(
            device,
            4096,
            1 / math.sqrt(128),
            batch=2,
            dtype=torch.float16,
            fp16_scope=True,
            automatic=True,
        )


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
    def test_block_mask_sparsity_with_partial_block(self, device):
        document_id = torch.zeros(100, dtype=torch.int, device=device)
        document_id[10:20] = 1
        for i in range(20, 100, 20):
            document_id[i : i + 20] = i // 20 + 1

        def document_causal_mask(b, h, q_idx, kv_idx):
            causal_mask = q_idx >= kv_idx
            document_mask = document_id[q_idx] == document_id[kv_idx]
            return causal_mask & document_mask

        block_mask = create_block_mask(document_causal_mask, 1, 1, 100, 100, device)
        self.assertEqual(block_mask.sparsity(), 0.0)
        self.assertTrue("sparsity=0.00%" in str(block_mask))

    @supported_platform
    def test_adjust_block_mask_ignores_entries_past_num_blocks(self, device):
        def mask_mod(b, h, q, kv):
            return (kv < 128) | ((kv >= 384) & (kv < 512))

        block_mask = create_block_mask(mask_mod, 1, 1, 128, 640, device=device)
        adjusted = block_mask._adjust(128, 384)
        expected = create_block_mask(mask_mod, 1, 1, 128, 384, device=device)

        self.assertEqual(adjusted.full_kv_num_blocks, expected.full_kv_num_blocks)
        self.assertEqual(adjusted.to_dense(), expected.to_dense())

    @supported_platform
    @common_utils.parametrize("BLOCK_SIZE", [32, 64, 128, 256, (32, 64), (64, 32)])
    def test_block_size_changes(self, device, BLOCK_SIZE: int | tuple[int, int]):
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
    def test_getitem_query_slice_updates_shape(self, device):
        def causal_mask(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx

        block_mask = create_block_mask(
            causal_mask,
            B=None,
            H=None,
            Q_LEN=8,
            KV_LEN=9,
            BLOCK_SIZE=4,
            device=device,
        )

        first_q_block = block_mask[:, :, :1]
        self.assertEqual(first_q_block.shape, (1, 1, 4, 9))
        self.assertEqual(first_q_block.seq_lengths, (4, 9))
        self.assertEqual(block_mask[:, :, 0].shape, (1, 1, 4, 9))

        second_q_block = block_mask[:, :, 1]
        self.assertEqual(second_q_block.shape, (1, 1, 4, 9))
        self.assertEqual(second_q_block.to_dense(), block_mask.to_dense()[:, :, 1:2, :])

        q = torch.empty(1, 1, 8, 2, device=device)
        self.assertEqual(q[:, :, :4, :].shape[-2], first_q_block.shape[-2])
        self.assertEqual(q[:, :, 4:8, :].shape[-2], second_q_block.shape[-2])

        with self.assertRaisesRegex(IndexError, "batch, head, and Q-block"):
            block_mask[:, :, :, :1]

        block_mask = create_block_mask(
            causal_mask,
            B=None,
            H=None,
            Q_LEN=9,
            KV_LEN=10,
            BLOCK_SIZE=4,
            device=device,
        )

        self.assertEqual(block_mask[:, :, :].shape, (1, 1, 9, 10))
        self.assertEqual(block_mask[:, :, 1].shape, (1, 1, 4, 10))
        self.assertEqual(block_mask[:, :, 1:].shape, (1, 1, 5, 10))
        self.assertEqual(block_mask[:, :, :0].shape, (1, 1, 0, 10))
        self.assertEqual(block_mask[:, :, ::2].shape, (1, 1, 5, 10))
        self.assertEqual(
            block_mask[:, :, ::2].to_dense(), block_mask.to_dense()[:, :, ::2, :]
        )

        tail_q_block = block_mask[:, :, 2]
        self.assertEqual(tail_q_block.shape, (1, 1, 1, 10))
        self.assertEqual(block_mask[:, :, -1].shape, tail_q_block.shape)
        self.assertEqual(tail_q_block.seq_lengths, (1, 10))
        self.assertEqual(tail_q_block.to_dense(), block_mask.to_dense()[:, :, 2:3, :])
        if tail_q_block.q_indices is None:
            raise AssertionError("Expected q_indices to be recomputed")
        self.assertEqual(
            tail_q_block.q_indices, torch.zeros_like(tail_q_block.q_indices)
        )

        tail_q_block_tensor_index = block_mask[:, :, torch.tensor([2], device=device)]
        self.assertEqual(tail_q_block_tensor_index.shape, (1, 1, 4, 10))

    @supported_platform
    def test_getitem(self, device):
        offset = torch.zeros(8, device=device)

        def causal_mask(b, h, q, kv):
            return (q + (offset[b] * 128)) >= kv

        block_mask = create_block_mask(causal_mask, 4, 2, 512, 512, device=device)
        if block_mask.kv_num_blocks.shape != (4, 2, 4):
            raise AssertionError(
                f"Expected shape (4, 2, 4), got {block_mask.kv_num_blocks.shape}"
            )
        if block_mask.kv_indices.shape != (4, 2, 4, 4):
            raise AssertionError(
                f"Expected shape (4, 2, 4, 4), got {block_mask.kv_indices.shape}"
            )

        # Index on batch dimension
        new_block_mask = block_mask[0]
        if new_block_mask.kv_num_blocks.shape != (1, 2, 4):
            raise AssertionError(
                f"Expected shape (1, 2, 4), got {new_block_mask.kv_num_blocks.shape}"
            )
        if new_block_mask.kv_indices.shape != (1, 2, 4, 4):
            raise AssertionError(
                f"Expected shape (1, 2, 4, 4), got {new_block_mask.kv_indices.shape}"
            )

        # Index on batch and head dimension
        new_block_mask = block_mask[0, 1]
        if new_block_mask.kv_num_blocks.shape != (
            1,
            1,
            4,
        ):
            raise AssertionError(
                f"Expected shape (1, 1, 4), got {new_block_mask.kv_num_blocks.shape}"
            )
        if new_block_mask.kv_indices.shape != (1, 1, 4, 4):
            raise AssertionError(
                f"Expected shape (1, 1, 4, 4), got {new_block_mask.kv_indices.shape}"
            )

        # Index on batch and head dimension with -1 semantics
        new_block_mask = block_mask[-1, -2]
        if new_block_mask.kv_num_blocks.shape != (
            1,
            1,
            4,
        ):
            raise AssertionError(
                f"Expected shape (1, 1, 4), got {new_block_mask.kv_num_blocks.shape}"
            )
        if new_block_mask.kv_indices.shape != (1, 1, 4, 4):
            raise AssertionError(
                f"Expected shape (1, 1, 4, 4), got {new_block_mask.kv_indices.shape}"
            )

        # slicing on batch and head dimension
        new_block_mask = block_mask[0:2, 1:2]
        if new_block_mask.kv_num_blocks.shape != (2, 1, 4):
            raise AssertionError(
                f"Expected shape (2, 1, 4), got {new_block_mask.kv_num_blocks.shape}"
            )
        if new_block_mask.kv_indices.shape != (2, 1, 4, 4):
            raise AssertionError(
                f"Expected shape (2, 1, 4, 4), got {new_block_mask.kv_indices.shape}"
            )

        # slicing on batch, head, and query dimension
        new_block_mask = block_mask[0:2, 1:2, torch.tensor([1], dtype=torch.int32)]
        if new_block_mask.kv_num_blocks.shape != (2, 1, 1):
            raise AssertionError(
                f"Expected shape (2, 1, 1), got {new_block_mask.kv_num_blocks.shape}"
            )
        if new_block_mask.kv_indices.shape != (2, 1, 1, 4):
            raise AssertionError(
                f"Expected shape (2, 1, 1, 4), got {new_block_mask.kv_indices.shape}"
            )

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
            if new_block_mask.full_kv_num_blocks is None:
                raise AssertionError("Expected full_kv_num_blocks to not be None")
            if new_block_mask.full_kv_indices is None:
                raise AssertionError("Expected full_kv_indices to not be None")
            torch.testing.assert_close(
                new_block_mask.full_kv_num_blocks,
                block_mask.full_kv_num_blocks[:, :, q_index],
            )
            torch.testing.assert_close(
                new_block_mask.full_kv_indices,
                block_mask.full_kv_indices[:, :, q_index, :],
            )

    @supported_platform
    def test_block_mask_positional_constructor_preserves_block_size(self, device):
        def causal_mask(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx

        ref = create_block_mask(
            causal_mask, B=1, H=1, Q_LEN=256, KV_LEN=256, BLOCK_SIZE=64, device=device
        )
        block_mask = BlockMask(
            ref.seq_lengths,
            ref.kv_num_blocks,
            ref.kv_indices,
            ref.full_kv_num_blocks,
            ref.full_kv_indices,
            ref.q_num_blocks,
            ref.q_indices,
            ref.full_q_num_blocks,
            ref.full_q_indices,
            ref.BLOCK_SIZE,
            ref.mask_mod,
        )

        self.assertEqual(block_mask.BLOCK_SIZE, ref.BLOCK_SIZE)
        self.assertIs(block_mask.mask_mod, ref.mask_mod)
        self.assertIsNone(block_mask.dq_write_order)

    @supported_platform
    def test_block_mask_transform_preserves_dq_write_order(self, device):
        def causal_mask(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx

        block_mask = create_block_mask(
            causal_mask,
            B=2,
            H=2,
            Q_LEN=256,
            KV_LEN=256,
            BLOCK_SIZE=128,
            device=device,
            compute_dq_write_order=True,
            dq_kv_order=True,
        )

        sliced = block_mask[0]
        self.assertIsNotNone(sliced.dq_write_order)
        self.assertEqual(sliced._dq_kv_order(), True)

        adjusted = block_mask._adjust(128, 128)
        self.assertIsNotNone(adjusted.dq_write_order)
        self.assertEqual(adjusted._dq_kv_order(), True)

        moved = block_mask.to(device)
        self.assertIsNotNone(moved.dq_write_order)
        self.assertEqual(moved._dq_kv_order(), True)

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
        if block_mask.kv_indices.device.type != device.type:
            raise AssertionError(
                f"Expected device type {device.type}, got {block_mask.kv_indices.device.type}"
            )
        if block_mask.kv_num_blocks.device.type != device.type:
            raise AssertionError(
                f"Expected device type {device.type}, got {block_mask.kv_num_blocks.device.type}"
            )
        if block_mask.q_indices.device.type != device.type:
            raise AssertionError(
                f"Expected device type {device.type}, got {block_mask.q_indices.device.type}"
            )
        if block_mask.q_num_blocks.device.type != device.type:
            raise AssertionError(
                f"Expected device type {device.type}, got {block_mask.q_num_blocks.device.type}"
            )

        block_mask = block_mask.to("cpu")
        if not block_mask.kv_indices.is_cpu:
            raise AssertionError("Expected kv_indices to be on CPU")
        if not block_mask.kv_num_blocks.is_cpu:
            raise AssertionError("Expected kv_num_blocks to be on CPU")
        if not block_mask.q_indices.is_cpu:
            raise AssertionError("Expected q_indices to be on CPU")
        if not block_mask.q_num_blocks.is_cpu:
            raise AssertionError("Expected q_num_blocks to be on CPU")

        block_mask = block_mask.to(device)
        if block_mask.kv_indices.device.type != device.type:
            raise AssertionError(
                f"Expected device type {device.type}, got {block_mask.kv_indices.device.type}"
            )
        if block_mask.kv_num_blocks.device.type != device.type:
            raise AssertionError(
                f"Expected device type {device.type}, got {block_mask.kv_num_blocks.device.type}"
            )
        if block_mask.q_indices.device.type != device.type:
            raise AssertionError(
                f"Expected device type {device.type}, got {block_mask.q_indices.device.type}"
            )
        if block_mask.q_num_blocks.device.type != device.type:
            raise AssertionError(
                f"Expected device type {device.type}, got {block_mask.q_num_blocks.device.type}"
            )

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
    def test_from_kv_blocks_unpadded_kv_indices_with_seq_lengths(self, device):
        kv_num_blocks = torch.ones((1, 1, 3), dtype=torch.int32, device=device)
        kv_indices = torch.arange(3, dtype=torch.int32, device=device).view(1, 1, 3, 1)

        block_mask = BlockMask.from_kv_blocks(
            kv_num_blocks,
            kv_indices,
            BLOCK_SIZE=32,
            seq_lengths=(96, 96),
        )

        self.assertEqual(block_mask.shape, (1, 1, 96, 96))
        torch.testing.assert_close(block_mask.kv_num_blocks, kv_num_blocks)
        torch.testing.assert_close(block_mask.kv_indices, kv_indices)
        torch.testing.assert_close(block_mask.q_num_blocks, kv_num_blocks)
        torch.testing.assert_close(block_mask.q_indices[..., :1], kv_indices)

    @supported_platform
    def test_from_kv_blocks_fullgraph_compile_with_inferred_seq_lengths(self, device):
        def from_kv_blocks_q_metadata(kv_num_blocks, kv_indices):
            block_mask = BlockMask.from_kv_blocks(kv_num_blocks, kv_indices)
            return block_mask.q_num_blocks, block_mask.q_indices

        kv_num_blocks, kv_indices, _, _ = self.generate_test_inputs(False, device)
        counter = CompileCounterWithBackend("aot_eager")
        q_num_blocks, q_indices = torch.compile(
            from_kv_blocks_q_metadata, backend=counter, fullgraph=True
        )(kv_num_blocks, kv_indices)

        self.assertEqual(counter.frame_count, 1)
        self.assertEqual(len(counter.graphs), 1)
        torch.testing.assert_close(
            q_num_blocks,
            torch.tensor([1, 1], dtype=torch.int32, device=device).view(1, 1, 2),
        )
        torch.testing.assert_close(
            q_indices,
            torch.tensor([0, 0], dtype=torch.int32, device=device).view(1, 1, 2, 1),
        )

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
    @expected_not_implemented_on_mps  # no captured buffers yet
    def test_doc_mask_clamped_repro(self, device):
        def _offsets_to_doc_ids_tensor(offsets):
            device = offsets.device
            counts = offsets[1:] - offsets[:-1]
            return torch.repeat_interleave(
                torch.arange(len(counts), device=device, dtype=torch.int32), counts
            )

        def length_to_offsets(lengths: list[int], device: str | torch.device) -> Tensor:
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
            lambda msg: f"{msg}\nExpected output shape {expected_shape}, but got {result.shape}",
        )

    @supported_platform
    @skip_on_xpu
    @skip_on_mps  # uses torch.cuda.CUDAGraph; CUDA-only
    def test_create_is_cuda_graphable(self, device):
        def mask_mod(b, h, q, kv):
            return q >= kv

        g = torch.cuda.CUDAGraph()

        with torch.cuda.graph(g):
            create_block_mask(mask_mod, None, None, 256, 256)

        g.replay()

    @common_utils.parametrize("compile", [False, True])
    @supported_platform
    @expected_not_implemented_on_mps  # no backward yet
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
        with self.assertRaisesRegex(RuntimeError, "block_mask was created for"):
            flex_attention_call(*create_inputs(2048), block_mask=block_mask)

        block_mask = create_block_mask(mask_mod, None, None, 1023, 1023, device=device)
        with self.assertRaisesRegex(RuntimeError, "block_mask was created for"):
            flex_attention_call(*create_inputs(1024), block_mask=block_mask)

    @supported_platform
    @skip_on_cpu
    @skip_on_mps
    def test_block_mask_check_does_not_specialize_backed_dynamic_length(self, device):
        def mask_mod(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx

        dtype = device_configs[torch.device(device).type].dtypes_fast[0]

        def create_inputs(S):
            q, k, v = (
                torch.randn(1, 1, S, 64, dtype=dtype, device=device) for _ in range(3)
            )
            block_mask = create_block_mask(mask_mod, None, None, S, S, device=device)
            return q, k, v, block_mask

        counter = CompileCounterWithBackend("eager")
        guard_code_parts = []

        @torch.compile(fullgraph=True, backend=counter)
        def flex_attention_call(q, k, v, block_mask):
            return flex_attention(q, k, v, block_mask=block_mask)

        def collect_guard_code_parts(guard_wrapper, f_locals, builder):
            parts = []
            for guard in guard_wrapper.root.get_epilogue_lambda_guards():
                parts.extend(guard.verbose_code_parts())
            guard_code_parts.append(parts)

        old_hook = torch._dynamo.guards.guard_manager_testing_hook_fn
        torch._dynamo.guards.guard_manager_testing_hook_fn = collect_guard_code_parts
        try:
            with torch.no_grad():
                for S in (320, 256, 192):
                    self.assertEqual(
                        flex_attention_call(*create_inputs(S)).shape, (1, 1, S, 64)
                    )
        finally:
            torch._dynamo.guards.guard_manager_testing_hook_fn = old_hook

        self.assertEqual(counter.frame_count, 2)
        dynamic_guard_code = "\n".join(guard_code_parts[-1])
        self.assertIn(
            "L['block_mask'].seq_lengths[0] == L['q'].size()[2]",
            dynamic_guard_code,
        )
        self.assertIn(
            "L['block_mask'].seq_lengths[1] == L['k'].size()[2]",
            dynamic_guard_code,
        )

        stale_block_mask = create_block_mask(
            mask_mod, None, None, 320, 320, device=device
        )
        with self.assertRaisesRegex(
            Exception, "block_mask was created for a smaller length"
        ):
            flex_attention_call(*create_inputs(512)[:3], stale_block_mask)

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
    @skip_on_mps  # asserts Triton-specific "BlockMask q_indices is None" runtime error
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
    @expected_not_implemented_on_mps  # no backwards yet
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

        if not out.isfinite().all().item():
            raise AssertionError("out contains non-finite values")
        if not q.grad.isfinite().all().item():
            raise AssertionError("q.grad contains non-finite values")
        if not k.grad.isfinite().all().item():
            raise AssertionError("k.grad contains non-finite values")
        if not v.grad.isfinite().all().item():
            raise AssertionError("v.grad contains non-finite values")

    @supported_platform
    @skip_on_cpu
    @expected_not_implemented_on_mps  # no backwards yet
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
        if not out.isfinite().all().item():
            raise AssertionError("out contains non-finite values")
        if not q.grad.isfinite().all().item():
            raise AssertionError("q.grad contains non-finite values")
        # assert k.grad.isfinite().all().item()
        if not v.grad.isfinite().all().item():
            raise AssertionError("v.grad contains non-finite values")

    @supported_platform
    @skip_on_cpu
    @expected_not_implemented_on_mps
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
        if not out.isfinite().all().item():
            raise AssertionError("out contains non-finite values")
        if not q.grad.isfinite().all().item():
            raise AssertionError("q.grad contains non-finite values")
        # assert k.grad.isfinite().all().item()
        if not v.grad.isfinite().all().item():
            raise AssertionError("v.grad contains non-finite values")

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
    def test_compute_dq_write_order_with_expanded_kv_indices(self, device):
        def causal_mask(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx

        block_mask = create_block_mask(
            causal_mask, B=1, H=1, Q_LEN=256, KV_LEN=256, BLOCK_SIZE=128, device=device
        )
        expanded_kv_block_mask = BlockMask(
            seq_lengths=block_mask.seq_lengths,
            kv_num_blocks=block_mask.kv_num_blocks.expand(2, 1, -1),
            kv_indices=block_mask.kv_indices.expand(2, 1, -1, -1),
            full_kv_num_blocks=(
                block_mask.full_kv_num_blocks.expand(2, 1, -1)
                if block_mask.full_kv_num_blocks is not None
                else None
            ),
            full_kv_indices=(
                block_mask.full_kv_indices.expand(2, 1, -1, -1)
                if block_mask.full_kv_indices is not None
                else None
            ),
            q_num_blocks=block_mask.q_num_blocks,
            q_indices=block_mask.q_indices,
            full_q_num_blocks=block_mask.full_q_num_blocks,
            full_q_indices=block_mask.full_q_indices,
            BLOCK_SIZE=block_mask.BLOCK_SIZE,
            mask_mod=block_mask.mask_mod,
        )
        dq_write_order, dq_write_order_full = _compute_dq_write_order_from_block_mask(
            expanded_kv_block_mask
        )

        self.assertEqual(dq_write_order.shape[:2], (2, 1))
        self.assertEqual(dq_write_order[0], dq_write_order[1])
        if dq_write_order_full is not None:
            self.assertEqual(dq_write_order_full.shape[:2], (2, 1))
            self.assertEqual(dq_write_order_full[0], dq_write_order_full[1])

    @supported_platform
    def test_create_block_mask_defaults_to_spt_dq_write_order(self, device):
        def causal_mask(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx

        block_mask = create_block_mask(
            causal_mask,
            B=1,
            H=1,
            Q_LEN=256,
            KV_LEN=256,
            BLOCK_SIZE=128,
            device=device,
            compute_dq_write_order=True,
        )

        self.assertIsNotNone(block_mask.dq_write_order)
        self.assertEqual(block_mask._dq_kv_order(), True)

    @supported_platform
    def test_create_block_mask_rejects_non_bool_dq_kv_order(self, device):
        def causal_mask(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx

        dq_kv_order = torch.tensor([[[1, 0]]], dtype=torch.int32, device=device)
        with self.assertRaisesRegex(
            ValueError,
            "dq_kv_order must be a bool",
        ):
            create_block_mask(
                causal_mask,
                B=1,
                H=1,
                Q_LEN=256,
                KV_LEN=256,
                BLOCK_SIZE=128,
                device=device,
                compute_dq_write_order=True,
                dq_kv_order=dq_kv_order,
            )

    @supported_platform
    def test_from_kv_blocks_accepts_dq_metadata(self, device):
        kv_num_blocks = torch.tensor([[[1, 2]]], dtype=torch.int32, device=device)
        kv_indices = torch.tensor(
            [[[[0, 0], [0, 1]]]], dtype=torch.int32, device=device
        )
        dq_write_order = torch.zeros((1, 1, 2, 2), dtype=torch.int32, device=device)
        dq_kv_order = torch.tensor([[[1, 0]]], dtype=torch.int32, device=device)

        block_mask = BlockMask.from_kv_blocks(
            kv_num_blocks,
            kv_indices,
            dq_write_order=dq_write_order,
            dq_kv_order=dq_kv_order,
        )
        spt_block_mask = BlockMask.from_kv_blocks(
            kv_num_blocks,
            kv_indices,
            dq_write_order=dq_write_order,
            dq_kv_order=True,
        )

        self.assertEqual(block_mask.dq_write_order, dq_write_order)
        self.assertEqual(block_mask.dq_kv_order, dq_kv_order)
        self.assertIsNone(block_mask.dq_write_order_full)
        self.assertEqual(spt_block_mask.dq_write_order, dq_write_order)
        self.assertEqual(spt_block_mask._dq_kv_order(), True)

        with self.assertRaisesRegex(ValueError, "dq_kv_order must be"):
            BlockMask.from_kv_blocks(
                kv_num_blocks,
                kv_indices,
                dq_write_order=dq_write_order,
                dq_kv_order=1,
            )
        with self.assertRaisesRegex(ValueError, "dq_write_order_full requires"):
            BlockMask.from_kv_blocks(
                kv_num_blocks,
                kv_indices,
                dq_write_order_full=dq_write_order,
            )

        with self.assertRaisesRegex(NotImplementedError, "tensor dq_kv_order"):
            block_mask[0]
        with self.assertRaisesRegex(NotImplementedError, "tensor dq_kv_order"):
            block_mask._adjust(128, 128)

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
                    lambda msg: f"{msg}\nTensor attribute {attr_name} should be None but got {reconstructed_value}",
                )
            else:
                self.assertIsInstance(
                    original_value,
                    torch.Tensor,
                    lambda msg: f"{msg}\nExpected {attr_name} to be a Tensor",
                )
                self.assertTrue(
                    torch.equal(original_value, reconstructed_value),
                    lambda msg: f"{msg}\nTensor attribute {attr_name} not equal after reconstruction",
                )

        # Verify all context attributes are equal (using _CONTEXT_ATTRS)
        for attr_name in BlockMask._CONTEXT_ATTRS:
            original_value = getattr(block_mask, attr_name)
            reconstructed_value = getattr(reconstructed_mask, attr_name)

            self.assertEqual(
                original_value,
                reconstructed_value,
                lambda msg: f"{msg}\nContext attribute {attr_name} not equal after reconstruction",
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

        n_regular = sum(
            getattr(block_mask, attr) is not None for attr in BlockMask._TENSOR_ATTRS
        )
        self.assertEqual(len(tensors_with_keys), n_regular)
        self.assertEqual(len(context_with_keys), len(BlockMask._CONTEXT_ATTRS) + 1)

        from torch.utils._pytree import GetAttrKey

        for key, _tensor in tensors_with_keys:
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

        # Verify the number of tensors and context values matches the present attributes
        n_regular = sum(
            getattr(block_mask, attr) is not None for attr in BlockMask._TENSOR_ATTRS
        )
        self.assertEqual(
            len(tensors),
            n_regular,
            "Number of tensors should match present tensor attributes",
        )
        self.assertEqual(
            len(context),
            len(BlockMask._CONTEXT_ATTRS) + 1,
            "Context values should include optional_tensor_attrs",
        )

        # Verify all attributes from the lists exist and are equal after reconstruction
        for attr_name in BlockMask._TENSOR_ATTRS + BlockMask._CONTEXT_ATTRS:
            self.assertTrue(
                hasattr(reconstructed_mask, attr_name),
                lambda msg: f"{msg}\nReconstructed mask missing attribute: {attr_name}",
            )
            original_value = getattr(block_mask, attr_name)
            reconstructed_value = getattr(reconstructed_mask, attr_name)

            if isinstance(original_value, torch.Tensor):
                self.assertTrue(
                    torch.equal(original_value, reconstructed_value),
                    lambda msg: f"{msg}\nTensor attribute {attr_name} not equal after reconstruction",
                )
            elif original_value is None:
                self.assertIsNone(
                    reconstructed_value,
                    lambda msg: f"{msg}\nAttribute {attr_name} should be None but got {reconstructed_value}",
                )
            else:
                self.assertEqual(
                    original_value,
                    reconstructed_value,
                    lambda msg: f"{msg}\nAttribute {attr_name} not equal after reconstruction",
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
        tensor_name: str | None = None,
    ):
        if (
            golden_out.device != ref_out.device
            or golden_out.device != compiled_out.device
        ):
            # MPS golden runs on CPU (no float64); compare everything on CPU.
            golden_out = golden_out.cpu()
            ref_out = ref_out.cpu()
            compiled_out = compiled_out.cpu()
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

        if _is_mps_device(device):
            # MPS has no float64; the golden runs on CPU, so its score_mod
            # captures and block_mask must be relocated off-device too.
            sdpa_partial_gold = create_attention(
                _relocate_captures(score_mod, "cpu"),
                block_mask.to("cpu"),
                enable_gqa=False,
            )
        else:
            sdpa_partial_gold = sdpa_partial

        golden_out = sdpa_partial_gold(q_gold, k_gold, v_gold)
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
    config_str: str | None = None

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


ROCM_FLAKY_LEARNABLE_BIAS_CASES = {
    "head_specific_gate": {
        (2, 4, 37, 16, torch.float16, "max-autotune-no-cudagraphs"),
        (2, 4, 37, 16, torch.float32, "max-autotune-no-cudagraphs"),
        (2, 4, 256, 16, torch.bfloat16, "max-autotune-no-cudagraphs"),
        (2, 4, 256, 16, torch.float16, "max-autotune-no-cudagraphs"),
        (2, 4, 256, 16, torch.float32, "max-autotune-no-cudagraphs"),
        (2, 4, 277, 16, torch.bfloat16, "max-autotune-no-cudagraphs"),
        (2, 4, 277, 16, torch.float32, "max-autotune-no-cudagraphs"),
    },
    "relative_1d_bias": {
        (2, 4, 37, 16, torch.bfloat16, "max-autotune-no-cudagraphs"),
        (2, 4, 256, 16, torch.bfloat16, "max-autotune-no-cudagraphs"),
        (2, 4, 256, 16, torch.float16, "max-autotune-no-cudagraphs"),
        (2, 4, 277, 16, torch.bfloat16, "max-autotune-no-cudagraphs"),
        (2, 4, 277, 16, torch.float16, "max-autotune-no-cudagraphs"),
        (2, 4, 277, 16, torch.float32, "max-autotune-no-cudagraphs"),
    },
    "symmetric_bias": {
        (2, 4, 37, 16, torch.bfloat16, "max-autotune-no-cudagraphs"),
        (2, 4, 37, 16, torch.float16, "max-autotune-no-cudagraphs"),
        (2, 4, 37, 16, torch.float32, "max-autotune-no-cudagraphs"),
        (2, 4, 256, 16, torch.float16, "max-autotune-no-cudagraphs"),
        (2, 4, 277, 16, torch.bfloat16, "max-autotune-no-cudagraphs"),
        (2, 4, 277, 16, torch.float16, "max-autotune-no-cudagraphs"),
        (2, 4, 277, 16, torch.float32, "max-autotune-no-cudagraphs"),
    },
}


supports_learnable_bias = unittest.skipUnless(
    (
        (torch.xpu.is_available() and has_triton())
        or (torch.cuda.is_available() and has_triton())
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

    def skip_rocm_flaky_learnable_bias_case(
        self, test_name: str, params: Params, mode: str
    ):
        """Skips parameterized ROCm flakes tracked by disabled-test issues."""
        case = (
            params.batch_size,
            params.num_heads,
            params.seq_length,
            params.head_dim,
            params.dtype,
            mode,
        )
        if TEST_WITH_ROCM and case in ROCM_FLAKY_LEARNABLE_BIAS_CASES[test_name]:
            self.skipTest(
                "Flaky on ROCm: https://github.com/pytorch/pytorch/issues/167271 and https://github.com/pytorch/pytorch/issues/167296"
            )

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
            lambda msg: f"{msg}\nTensor: {tensor_name}\nCompiled error ({comp_error:.8f}) exceeds "
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
        self.skip_rocm_flaky_learnable_bias_case("relative_1d_bias", params, mode)
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
        self.skip_rocm_flaky_learnable_bias_case("symmetric_bias", params, mode)
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
        self.skip_rocm_flaky_learnable_bias_case("head_specific_gate", params, mode)
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

        if bias.grad is None:
            raise AssertionError("No gradient computed for bias")
        if not torch.any(bias.grad != 0):
            raise AssertionError("Gradient for bias is 0")

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

        if not (x.grad.norm() > 0):
            raise AssertionError(f"Expected x.grad.norm() > 0, got {x.grad.norm()}")
        if not (y.grad.norm() > 0):
            raise AssertionError(f"Expected y.grad.norm() > 0, got {y.grad.norm()}")

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
            out.shape,
            query.shape,
            lambda msg: f"{msg}\nExpected shape {query.shape}, got {out.shape}",
        )

    @skip_on_cpu
    @skip_on_mps  # uses mode="max-autotune-no-cudagraphs"; Triton/CUDA-only
    def test_flex_attention_with_dynamic_max_autotune(self, device):
        self._test_flex_attention_with_dynamic_max_autotune(device)

    @skip_on_cpu
    @torch._inductor.config.patch("graph_partition", True)
    def test_flex_attention_with_dynamic_max_autotune_graph_partition(self, device):
        self._test_flex_attention_with_dynamic_max_autotune(device)

    @skip_on_cpu
    def test_flex_attention_logging(self, device):
        from torch._inductor.select_algorithm import get_flex_attention_log_filename

        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "flex_attention_configs")

            with patch.dict(
                os.environ, {"TORCHINDUCTOR_FLEX_ATTENTION_LOGGING_FILE": log_file}
            ):
                get_flex_attention_log_filename.cache_clear()
                self.addCleanup(get_flex_attention_log_filename.cache_clear)

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
                    flex_attention, dynamic=True, mode="max-autotune-no-cudagraphs"
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
                    os.path.exists(json_file),
                    lambda msg: f"{msg}\nLog file {json_file} was not created",
                )

                with open(json_file) as f:
                    log_data = json.load(f)

                self.assertIsInstance(log_data, list)
                self.assertEqual(len(log_data), 2)

                # Check that we have both forward and backward entries
                kernel_types_seen = [entry["kernel_type"] for entry in log_data]
                self.assertIn("forward", kernel_types_seen)
                self.assertIn("backward", kernel_types_seen)

                # Expected values from the test inputs
                expected_query_shape = "[1, 2, 128, 64]"
                expected_key_shape = "[1, 2, 128, 64]"
                expected_value_shape = "[1, 2, 128, 64]"
                expected_B = 1
                expected_Hq = 2
                expected_Hkv = 2
                expected_seq_len_q = 128
                expected_seq_len_kv = 128
                expected_qk_head_dim = 64
                expected_v_head_dim = 64

                for entry in log_data:
                    self.assertIsInstance(entry, dict)
                    # New format has: query_shape, key_shape, value_shape, kernel_type, choices
                    self.assertIn("kernel_type", entry)
                    self.assertIn("choices", entry)
                    self.assertIn("query_shape", entry)
                    self.assertIn("key_shape", entry)
                    self.assertIn("value_shape", entry)

                    # Check shape values
                    self.assertEqual(entry["query_shape"], expected_query_shape)
                    self.assertEqual(entry["key_shape"], expected_key_shape)
                    self.assertEqual(entry["value_shape"], expected_value_shape)

                    kernel_type = entry["kernel_type"]
                    choices = entry["choices"]

                    self.assertIsInstance(choices, list)
                    self.assertGreater(len(choices), 0)

                    for i, choice in enumerate(choices):
                        self.assertIn("type", choice)
                        self.assertIn("time", choice)

                        if choice["type"] == "triton":
                            self.assertIn("num_warps", choice)
                            self.assertIn("num_stages", choice)

                            # Check numerical values in each choice
                            self.assertIn("B", choice)
                            self.assertIn("Hq", choice)
                            self.assertIn("Hkv", choice)
                            self.assertIn("seq_len_q", choice)
                            self.assertIn("seq_len_kv", choice)
                            self.assertIn("qk_head_dim", choice)
                            self.assertIn("v_head_dim", choice)

                            self.assertEqual(choice["B"], expected_B)
                            self.assertEqual(choice["Hq"], expected_Hq)
                            self.assertEqual(choice["Hkv"], expected_Hkv)
                            self.assertEqual(choice["seq_len_q"], expected_seq_len_q)
                            self.assertEqual(choice["seq_len_kv"], expected_seq_len_kv)
                            self.assertEqual(
                                choice["qk_head_dim"], expected_qk_head_dim
                            )
                            self.assertEqual(choice["v_head_dim"], expected_v_head_dim)

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
    # Skipped on MI200 rather than loosened: flex's compiled backward dV is
    # genuinely less accurate there (v.grad rmse vs the fp64 gold is ~2.5e-3
    # vs ~3e-4 for the sdpa reference, an 8-11x ratio; no subnormals are
    # involved). Covering that with a larger max error ratio would also mask
    # regressions in the out/q.grad/k.grad/bias.grad comparisons, which all
    # match the reference exactly on MI200.
    @skipIfRocmArch(MI200_ARCH)
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
                lambda msg: f"{msg}\n{name} -> Ref error: {ref_error}, Flex eager Error: {flex_error}",
            )


common_utils.instantiate_parametrized_tests(TestFlexAttentionROCmOptions)

instantiate_device_type_tests(
    TestFlexAttention,
    globals(),
    only_for=test_device,
    allow_xpu=True,
    allow_mps=True,
)
instantiate_device_type_tests(
    TestPagedAttention,
    globals(),
    only_for=test_device,
    allow_xpu=True,
    allow_mps=True,
)
instantiate_device_type_tests(
    TestBlockMask,
    globals(),
    only_for=(test_device[0] if (HAS_GPU or HAS_MPS) else "cuda",),
    allow_xpu=True,
    allow_mps=True,
)
instantiate_device_type_tests(
    TestLearnableBiases, globals(), only_for=test_device, allow_xpu=True
)
# Register on the primary local device only. Hardcoding ("cuda",) generates no
# class at all on a CPU/XPU/MPS-only build, so the suite reports zero tests
# rather than a skip, and coverage becomes indistinguishable from absence.
# Passing all of test_device is also wrong: it includes "cpu" alongside "cuda",
# and running_on_tdm_device() is host-global, so the CPU class would clear the
# class-level skip on a gfx1250 host and then run these tests on CPU.
instantiate_device_type_tests(
    TestFlexAttentionTDMEndToEnd,
    globals(),
    only_for=(test_device[0],),
    allow_xpu=True,
    allow_mps=True,
)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
