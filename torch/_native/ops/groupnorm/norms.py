"""CuteDSL GroupNorm adapter.

Adapts the CuTE DSL kernel interface to match the ATen op signatures
for ``native_group_norm`` and ``native_group_norm_backward``.
"""

from __future__ import annotations

import torch
from torch import Tensor


def _empty_zeroed(size: int, device: torch.device) -> Tensor:
    """Allocate a float32 tensor and zero it with cudaMemsetAsync.

    Cheaper than torch.zeros which goes through the full tensor-factory path
    (dtype dispatch, fill kernel launch). cudaMemsetAsync is a single async
    DMA command with negligible launch overhead.
    """
    import cuda.bindings.runtime as cudart

    t = torch.empty(size, device=device, dtype=torch.float32)
    stream = torch.cuda.current_stream(device).cuda_stream
    (err,) = cudart.cudaMemsetAsync(t.data_ptr(), 0, t.nbytes, stream)
    assert err == cudart.cudaError_t.cudaSuccess, f"cudaMemsetAsync failed: {err}"
    return t


def cutedsl_groupnorm_fwd(
    input: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    N: int,
    C: int,
    HxW: int,
    group: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from ._groupnorm_kernels import _groupnorm_fwd

    cpg = C // group
    M = N * group
    K = cpg * HxW

    x = input.view(M, K)
    if not x.is_contiguous():
        x = x.contiguous()

    out = torch.empty(M, K, device=x.device, dtype=x.dtype)
    mean = torch.empty(M, device=x.device, dtype=torch.float32)
    rstd = torch.empty(M, device=x.device, dtype=torch.float32)

    if M > 0:
        _groupnorm_fwd(x, weight, bias, out, mean, rstd, cpg, HxW, group, eps)

    return out.view_as(input), mean.view(N, group), rstd.view(N, group)


def cutedsl_groupnorm_bwd(
    grad_out: torch.Tensor,
    input: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    weight: torch.Tensor | None,
    N: int,
    C: int,
    HxW: int,
    group: int,
    output_mask: list[bool],
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    from ._groupnorm_kernels import _get_sm_count, _groupnorm_bwd

    cpg = C // group
    M = N * group
    K = cpg * HxW

    if M == 0:
        grad_input = input.new_empty(input.shape) if output_mask[0] else None
        grad_weight = (
            weight.new_zeros(weight.shape)
            if output_mask[1] and weight is not None
            else None
        )
        grad_bias = input.new_zeros(C) if output_mask[2] else None
        return grad_input, grad_weight, grad_bias

    x = input.view(M, K)
    if not x.is_contiguous():
        x = x.contiguous()
    dout = grad_out.view(M, K)
    if not dout.is_contiguous():
        dout = dout.contiguous()
    mean_flat = mean.view(M).contiguous()
    rstd_flat = rstd.view(M).contiguous()

    dx = torch.empty_like(x)
    sm_count = _get_sm_count(K, x.device)

    dw = (
        _empty_zeroed(C, x.device)
        if output_mask[1] and weight is not None
        else None
    )
    db = (
        _empty_zeroed(C, x.device)
        if output_mask[2]
        else None
    )

    _groupnorm_bwd(
        x, weight, dout, mean_flat, rstd_flat, dx, dw, db, sm_count,
        cpg, HxW, group,
    )

    grad_input = dx.view_as(input) if output_mask[0] else None
    grad_weight = dw if dw is not None else None
    grad_bias = db if db is not None else None
    return grad_input, grad_weight, grad_bias
