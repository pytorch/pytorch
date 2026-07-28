"""Quack-backed RMSNorm overrides for aten fused RMSNorm operators.

Uses the vendored quack subset at ``torch._vendor.quack``.
"""
# mypy: allow-untyped-defs

from __future__ import annotations

import math

import torch

from ... import cutedsl_utils as cu
from .norms import _required_align_bytes


def _is_supported(input: torch.Tensor) -> bool:
    if input.device.type != "cuda":
        return False
    if torch.version.hip is not None:
        return False
    if input.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    major, _ = torch.cuda.get_device_capability(input.device)
    return major in (9, 10)


# quack splits each row across a CTA cluster (at most 16 on SM90/SM100, the
# only archs _is_supported accepts) and stages the per-CTA row tile in shared
# memory. Rows whose tile exceeds the smem budget even at the max cluster size
# cannot launch, and far beyond that (e.g. N=2^28) the CuTe DSL compiler hangs
# or crashes before any smem check could fire (gh-186800). Bound N here so
# such rows fall back to aten. The reserve covers the reduction buffer,
# mbarriers, and smem alignment (mirrors quack's _BWD_SMEM_RESERVED_BYTES).
_SMEM_RESERVED_BYTES = 4 * 1024
_MAX_CLUSTER_N = 16
# The kernel rounds the per-CTA tile up to vecsize * threads_per_row elements
# (reduction_base._get_tiled_copy). Both are powers of 2 with vecsize <= 8 and
# threads_per_row <= 256, so rounding up to 2048 never under-estimates.
_TILE_ROUND_ELEMS = 2048
# RmsNormBwdConfig.smem_stages default; both analytical heuristics use it.
_BWD_SMEM_STAGES = 2


def _smem_budget_bytes(device: torch.device) -> int:
    props = torch.cuda.get_device_properties(device)
    smem = getattr(
        props, "shared_memory_per_block_optin", props.shared_memory_per_block
    )
    return smem - _SMEM_RESERVED_BYTES


def _row_tile_elems(n: int) -> int:
    per_cta = -(-n // _MAX_CLUSTER_N)
    return -(-per_cta // _TILE_ROUND_ELEMS) * _TILE_ROUND_ELEMS


def _fwd_fits_smem(input: torch.Tensor, n: int) -> bool:
    # Fwd smem holds one row tile of x (rmsnorm.py sX).
    return _row_tile_elems(n) * input.element_size() <= _smem_budget_bytes(input.device)


def _bwd_fits_smem(input: torch.Tensor, grad_out: torch.Tensor, n: int) -> bool:
    # Quack's RMSNormBackward.__init__ raises for N > 128K with fp32 x; fall
    # back instead of surfacing its ValueError.
    if input.element_size() >= 4 and n > 128 * 1024:
        return False
    # Bwd smem holds smem_stages buffers of both x and dout (rmsnorm.py
    # sX/sdO).
    tile_bytes = (
        _row_tile_elems(n)
        * _BWD_SMEM_STAGES
        * (input.element_size() + grad_out.element_size())
    )
    return tile_bytes <= _smem_budget_bytes(input.device)


# A contiguous input with a misaligned base pointer must be cloned before the
# kernel can run (see norms._reshape_2d). The clone's extra read+write only
# pays off once quack's bandwidth advantage over aten can absorb it: measured
# on B200 (fwd, bf16/fp32), clone+quack wins 1.8-2.9x at 2^24 elements and
# loses below 2^23. Misaligned inputs smaller than this fall back to aten.
# The threshold is a perf heuristic, not a correctness bound; it has not been
# measured on H100 (SM90), where the crossover may differ. Make it per-arch
# if H100 measurements show a meaningfully different break-even.
_MISALIGNED_MIN_NUMEL = 1 << 24


def _misaligned_clone_unprofitable(t: torch.Tensor, n: int) -> bool:
    # Only contiguous tensors hit the clone path in norms._reshape_2d;
    # non-contiguous ones pay the reshape+contiguous materialization either
    # way, which always lands on an aligned fresh buffer.
    if not t.is_contiguous():
        return False
    if t.data_ptr() % _required_align_bytes(t, n) == 0:
        return False
    return t.numel() < _MISALIGNED_MIN_NUMEL


def _n_yields_valid_cp_size(n: int, dtype: torch.dtype) -> bool:
    # quack picks vecsize = gcd(N, 128 // dtype_bits) and lowers each thread's
    # gmem->smem copy to cp.async, whose PTX cp_size only accepts 32, 64, or
    # 128 bits. Narrow dtypes with an unfriendly N (e.g. odd N for bf16/fp16)
    # produce a 16-bit vector copy that fails CuTe IR verification at compile
    # time; fall through to aten in that case.
    dtype_bits = torch.finfo(dtype).bits
    vecsize = math.gcd(n, 128 // dtype_bits)
    return vecsize * dtype_bits in (32, 64, 128)


def _shape_is_valid(
    input: torch.Tensor,
    normalized_shape: list[int],
    weight: torch.Tensor | None,
) -> bool:
    # Mirror aten's _check_layer_norm_inputs: on any mismatch the aten path
    # raises, so we fall through and let it produce the proper error rather
    # than silently running the override on ill-shaped inputs.
    n = len(normalized_shape)
    if n < 1 or input.ndim < n:
        return False
    if list(input.shape[-n:]) != list(normalized_shape):
        return False
    if weight is not None and list(weight.shape) != list(normalized_shape):
        return False
    return True


def _fused_rms_norm_cond(
    input: torch.Tensor,
    normalized_shape: list[int],
    weight: torch.Tensor | None,
    eps: float | None,
) -> bool:
    if not _is_supported(input):
        return False
    if not _shape_is_valid(input, normalized_shape, weight):
        return False
    # Weight must match input dtype and device for quack's kernel; mismatches
    # are legal under aten (it casts), so fall through.
    if weight is not None and (
        weight.dtype != input.dtype or weight.device != input.device
    ):
        return False
    # Empty inputs crash quack with cudaErrorInvalidConfiguration -- the bad
    # launch config poisons the CUDA context for subsequent calls. Quack's own
    # rmsnorm_bwd guards against this with `if x.numel() > 0` (rmsnorm.py:1111)
    # but the fwd path doesn't.
    if input.numel() == 0:
        return False
    if not _n_yields_valid_cp_size(math.prod(normalized_shape), input.dtype):
        return False
    if not _fwd_fits_smem(input, math.prod(normalized_shape)):
        return False
    if _misaligned_clone_unprofitable(input, math.prod(normalized_shape)):
        return False
    # Non-contiguous weight would require a reshape+copy that we haven't
    # measured; fall through to aten until we do.
    if weight is not None and not weight.is_contiguous():
        return False
    # The override reshapes + makes contiguous, which materializes a COW input.
    # Match the bmm_outer_product cond (triton_impl.py:46) and fall through to
    # aten so composite-compliance tests don't flag spurious materialization.
    is_cow = torch._C._is_cow_tensor  # pyrefly: ignore[missing-attribute]
    if is_cow(input):
        return False
    if weight is not None and is_cow(weight):
        return False
    return True


def _fused_rms_norm_impl(
    input: torch.Tensor,
    normalized_shape: list[int],
    weight: torch.Tensor | None,
    eps: float | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if eps is None:
        # Match aten/src/ATen/native/cuda/layer_norm_kernel.cu:1841-1847:
        # aten picks eps from the *accumulator* dtype, which is float32 for
        # fp16/bf16/fp32 inputs (the only dtypes our cond accepts).
        eps = torch.finfo(torch.float32).eps

    from .norms import quack_rmsnorm_fwd

    return quack_rmsnorm_fwd(input, weight, normalized_shape, eps)


def _fused_rms_norm_backward_cond(
    grad_out: torch.Tensor,
    input: torch.Tensor,
    normalized_shape: list[int],
    rstd: torch.Tensor,
    weight: torch.Tensor | None,
    output_mask: list[bool],
) -> bool:
    if not _is_supported(input):
        return False
    if not _shape_is_valid(input, normalized_shape, weight):
        return False
    if weight is not None and (
        weight.dtype != input.dtype or weight.device != input.device
    ):
        return False
    if input.numel() == 0:
        return False
    if not _n_yields_valid_cp_size(math.prod(normalized_shape), input.dtype):
        return False
    if not _bwd_fits_smem(input, grad_out, math.prod(normalized_shape)):
        return False
    n = math.prod(normalized_shape)
    if _misaligned_clone_unprofitable(input, n) or _misaligned_clone_unprofitable(
        grad_out, n
    ):
        return False
    if weight is not None and not weight.is_contiguous():
        return False
    is_cow = torch._C._is_cow_tensor  # pyrefly: ignore[missing-attribute]
    for t in (grad_out, input, rstd, weight):
        if t is not None and is_cow(t):
            return False
    return True


def _fused_rms_norm_backward_impl(
    grad_out: torch.Tensor,
    input: torch.Tensor,
    normalized_shape: list[int],
    rstd: torch.Tensor,
    weight: torch.Tensor | None,
    output_mask: list[bool],
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    from .norms import quack_rmsnorm_bwd

    grad_input, grad_weight = quack_rmsnorm_bwd(
        grad_out,
        input,
        rstd,
        weight,
        normalized_shape,
        dw_mask=output_mask[1],
    )

    if not output_mask[0]:
        grad_input = None
    return grad_input, grad_weight


def register_rmsnorm_overrides() -> None:
    # Don't gate on torch.cuda.is_available() here: it calls cuInit and
    # poisons fork (vLLM EngineCore relies on fork). cu.register_op_override
    # already short-circuits on _cuda.is_built() via _check_runtime_available.
    cu.register_op_override(
        "aten",
        "_fused_rms_norm",
        "CUDA",
        cond=_fused_rms_norm_cond,
        impl=_fused_rms_norm_impl,
    )
    cu.register_op_override(
        "aten",
        "_fused_rms_norm_backward",
        "CUDA",
        cond=_fused_rms_norm_backward_cond,
        impl=_fused_rms_norm_backward_impl,
    )
