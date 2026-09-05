from custom_op import custom_op

import torch
from torch import Tensor


@custom_op("tmp::rmsnorm_partial_forward")
def rmsnorm_partial_forward(
    x: Tensor,
    weight: Tensor,
    block_size: int,
) -> tuple[Tensor, Tensor]:
    m, n = x.shape
    partial_squares = x.pow(2).reshape(m, n // block_size, block_size).mean(dim=-1)
    weighted = x * weight
    return partial_squares, weighted


@custom_op("tmp::rmsnorm_finalize_forward")
def rmsnorm_finalize_forward(
    weighted: Tensor,
    rstd: Tensor,
) -> Tensor:
    return weighted * rstd


@custom_op("tmp::rmsnorm_bwd_zdz")
def rmsnorm_bwd_zdz(
    grad_out: Tensor,
    x: Tensor,
    weight: Tensor,
    rstd: Tensor,
) -> Tensor:
    return (grad_out * x * rstd * weight).mean(dim=-1, keepdim=True)


@custom_op("tmp::rmsnorm_bwd_input")
def rmsnorm_bwd_input(
    grad_out: Tensor,
    x: Tensor,
    rms: dict[str, Tensor],
    zdz: Tensor,
) -> Tensor:
    return (grad_out * rms["weight"] - x * rms["rstd"] * zdz) * rms["rstd"]


@custom_op("tmp::rmsnorm_bwd_weight")
def rmsnorm_bwd_weight(
    grad_out: Tensor,
    x: Tensor,
    rstd: Tensor,
    block_size: int,
) -> Tensor:
    grad_weight = grad_out * x * rstd
    if block_size == 1:
        return grad_weight.T
    return grad_weight.reshape(-1, block_size, grad_weight.shape[-1]).sum(dim=1).T


@custom_op("tmp::rmsnorm_bwd_weight_reduce")
def rmsnorm_bwd_weight_reduce(partial_grad_weight: Tensor) -> Tensor:
    return partial_grad_weight.sum(dim=-1)


@custom_op("coda::gemm_residual_partial_rmsnorm")
def gemm_residual_partial_rmsnorm(
    x: Tensor,
    gemm_weight: Tensor,
    residual: Tensor,
    rms_weight: Tensor,
    block_size: int,
) -> tuple[Tensor, Tensor, Tensor]:
    d = residual + x @ gemm_weight.T
    m, n = d.shape
    partial_squares = d.pow(2).reshape(m, n // block_size, block_size).mean(dim=-1)
    weighted = d * rms_weight
    return d, partial_squares, weighted


@custom_op("coda::rms_final_reduce")
def rms_final_reduce(
    partial_squares: Tensor,
    eps: float,
) -> Tensor:
    return torch.rsqrt(partial_squares.mean(dim=-1, keepdim=True) + eps)


@custom_op("coda::gemm_rmsnorm")
def gemm_rmsnorm(
    weighted: Tensor,
    gemm_weight: Tensor,
    rstd: Tensor,
) -> Tensor:
    return (weighted @ gemm_weight.T) * rstd


@custom_op("coda::gemm_rmsnorm_swiglu")
def gemm_rmsnorm_swiglu(
    weighted: Tensor,
    params: dict[str, Tensor],
    rstd: Tensor,
) -> tuple[Tensor, Tensor]:
    gemm_weight = params["weight"]
    preactivation = (weighted @ gemm_weight.T) * rstd
    gate, value = preactivation.chunk(2, dim=-1)
    return preactivation, torch.nn.functional.silu(gate) * value


@custom_op("coda::gemm_rmsnorm_partial_cross_entropy")
def gemm_rmsnorm_partial_cross_entropy(
    weighted: Tensor,
    gemm_weight: Tensor,
    rstd: Tensor,
    targets: Tensor,
    block_size: int,
) -> tuple[Tensor, Tensor, Tensor]:
    logits = (weighted @ gemm_weight.T) * rstd
    logits_tgt = logits[torch.arange(logits.shape[0], device=logits.device), targets]
    logits_lse = logits.reshape(
        logits.shape[0], logits.shape[1] // block_size, block_size
    )
    return logits, logits_tgt, torch.logsumexp(logits_lse, dim=-1)


@custom_op("coda::cross_entropy")
def cross_entropy(
    logits: Tensor,
    targets: Tensor,
    logits_tgt: Tensor,
    logits_lse: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    logits_lse = torch.logsumexp(logits_lse, dim=-1)
    loss = logits_lse - logits_tgt
    grad_logits = torch.exp(logits - logits_lse.unsqueeze(-1))
    target_mask = torch.zeros_like(logits).scatter(1, targets.unsqueeze(-1), 1.0)
    grad_logits = grad_logits - target_mask
    zdz = (logits * grad_logits).sum(dim=-1, keepdim=True)
    return loss, grad_logits, zdz


@custom_op("coda::gemm_partial_swiglu_bwd")
def gemm_partial_swiglu_bwd(
    grad_out: Tensor,
    down_weight: Tensor,
    preactivation: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    grad_swiglu = grad_out @ down_weight
    gate, value = preactivation.chunk(2, dim=-1)
    sigmoid = torch.sigmoid(gate)
    silu = torch.nn.functional.silu(gate)
    swiglu_out = silu * value
    grad_gate = grad_swiglu * value * (sigmoid + silu * (1.0 - sigmoid))
    grad_value = grad_swiglu * silu
    grad_preactivation = torch.cat([grad_gate, grad_value], dim=-1)
    zdz = (preactivation * grad_preactivation).sum(dim=-1, keepdim=True)
    return grad_preactivation, zdz, swiglu_out


@custom_op("coda::gemm_residual_partial_rmsnorm_bwd")
def gemm_residual_partial_rmsnorm_bwd(
    grad_out: Tensor,
    next_weight: Tensor,
    norm_input: Tensor,
    rms_weight: Tensor,
    rstd: Tensor,
    zdz: Tensor,
    residual_grad: Tensor,
    block_size: int,
) -> tuple[Tensor, Tensor, Tensor]:
    grad_norm = grad_out @ next_weight
    norm_input_scaled = norm_input * rstd
    zdz = zdz / norm_input.shape[-1]
    residual_grad = (
        residual_grad + (grad_norm * rms_weight - norm_input_scaled * zdz) * rstd
    )
    normalized = norm_input_scaled * rms_weight
    grad_rms_weight = grad_norm * norm_input_scaled
    if block_size == 1:
        return residual_grad, normalized, grad_rms_weight.T
    return (
        residual_grad,
        normalized,
        grad_rms_weight.reshape(-1, block_size, grad_rms_weight.shape[-1]).sum(dim=1).T,
    )
