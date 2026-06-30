# ruff: noqa: S101

from collections import Counter

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.fx.experimental.proxy_tensor import make_fx

import coda_infra as coda

EPS = 1e-6


def rmsnorm(x: Tensor, weight: Tensor) -> Tensor:
    rstd = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + EPS)
    return x * rstd * weight


def swiglu(x: Tensor) -> Tensor:
    gate, value = x.chunk(2, dim=-1)
    return F.silu(gate) * value


def attention_context_from_qkv(qkv: Tensor, like: Tensor) -> Tensor:
    num_heads = 2
    q, k, v = qkv.view(like.shape[0], 3, num_heads, -1).unbind(dim=1)
    q = q.transpose(0, 1).unsqueeze(0)
    k = k.transpose(0, 1).unsqueeze(0)
    v = v.transpose(0, 1).unsqueeze(0)
    return F.scaled_dot_product_attention(q, k, v).squeeze(0).transpose(0, 1).reshape(like.shape)


def attention_context(x: Tensor, qkv_weight: Tensor) -> Tensor:
    return attention_context_from_qkv(F.linear(x, qkv_weight), x)


def attention(x: Tensor, qkv_weight: Tensor, out_weight: Tensor) -> Tensor:
    return F.linear(attention_context(x, qkv_weight), out_weight)


class CodaFirstHalf(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        attn_context: Tensor,
        residual: Tensor,
        attn_out: Tensor,
        norm_w: Tensor,
        up_w: Tensor,
        down_w: Tensor,
        next_norm_w: Tensor,
        qkv_w: Tensor,
    ) -> tuple[Tensor, Tensor]:
        norm0, partial0, weighted0 = coda.gemm_residual_partial_rmsnorm(
            attn_context, attn_out, residual, norm_w, residual.size(-1)
        )
        rstd0 = coda.rms_final_reduce(partial0, EPS)
        pre0, hidden0 = coda.gemm_rmsnorm_swiglu(weighted0, {"weight": up_w}, rstd0)
        norm1, partial1, weighted1 = coda.gemm_residual_partial_rmsnorm(
            hidden0, down_w, norm0, next_norm_w, norm0.size(-1)
        )
        rstd1 = coda.rms_final_reduce(partial1, EPS)
        qkv = coda.gemm_rmsnorm(weighted1, qkv_w, rstd1)
        ctx.save_for_backward(
            attn_context, attn_out, norm_w, up_w, down_w, next_norm_w, qkv_w,
            norm0, rstd0, pre0, hidden0, norm1, rstd1, qkv,
        )
        return norm1, qkv

    @staticmethod
    def backward(ctx, grad_norm1: Tensor, grad_qkv: Tensor):
        (
            attn_context, attn_out, norm_w, up_w, down_w, next_norm_w, qkv_w,
            norm0, rstd0, pre0, hidden0, norm1, rstd1, qkv,
        ) = ctx.saved_tensors
        zdz1 = (qkv * grad_qkv).sum(dim=-1, keepdim=True)
        grad_norm0, normalized1, partial_grad_next_norm = coda.gemm_residual_partial_rmsnorm_bwd(
            grad_qkv, qkv_w, norm1, next_norm_w, rstd1, zdz1, grad_norm1, 1
        )
        grad_pre0, zdz0, hidden0 = coda.gemm_partial_swiglu_bwd(grad_norm0, down_w, pre0)
        grad_residual, normalized0, partial_grad_norm = coda.gemm_residual_partial_rmsnorm_bwd(
            grad_pre0, up_w, norm0, norm_w, rstd0, zdz0, grad_norm0, 1
        )
        return (
            grad_residual @ attn_out,
            grad_residual,
            grad_residual.T @ attn_context,
            partial_grad_norm.sum(dim=-1),
            grad_pre0.T @ normalized0,
            grad_norm0.T @ hidden0,
            partial_grad_next_norm.sum(dim=-1),
            grad_qkv.T @ normalized1,
        )


class CodaSecondHalf(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        attn_context: Tensor,
        residual: Tensor,
        attn_out: Tensor,
        norm_w: Tensor,
        up_w: Tensor,
        down_w: Tensor,
        final_norm_w: Tensor,
        lm_head: Tensor,
        targets: Tensor,
    ) -> Tensor:
        norm0, partial0, weighted0 = coda.gemm_residual_partial_rmsnorm(
            attn_context, attn_out, residual, norm_w, residual.size(-1)
        )
        rstd0 = coda.rms_final_reduce(partial0, EPS)
        pre0, hidden0 = coda.gemm_rmsnorm_swiglu(weighted0, {"weight": up_w}, rstd0)
        norm1, partial1, weighted1 = coda.gemm_residual_partial_rmsnorm(
            hidden0, down_w, norm0, final_norm_w, norm0.size(-1)
        )
        rstd1 = coda.rms_final_reduce(partial1, EPS)
        logits, logits_tgt, logits_lse = coda.gemm_rmsnorm_partial_cross_entropy(
            weighted1, lm_head, rstd1, targets, lm_head.size(0)
        )
        loss, grad_logits, zdz1 = coda.cross_entropy(logits, targets, logits_tgt, logits_lse)
        ctx.save_for_backward(
            attn_context, attn_out, norm_w, up_w, down_w, final_norm_w, lm_head,
            norm0, rstd0, pre0, hidden0, norm1, rstd1, grad_logits, zdz1,
        )
        return loss

    @staticmethod
    def backward(ctx, grad_loss: Tensor):
        (
            attn_context, attn_out, norm_w, up_w, down_w, final_norm_w, lm_head,
            norm0, rstd0, pre0, hidden0, norm1, rstd1, grad_logits, zdz1,
        ) = ctx.saved_tensors
        grad_logits = grad_logits * grad_loss.unsqueeze(-1)
        zdz1 = zdz1 * grad_loss.unsqueeze(-1)
        grad_norm0, normalized1, partial_grad_final_norm = (
            coda.gemm_residual_partial_rmsnorm_bwd(
                grad_logits,
                lm_head,
                norm1,
                final_norm_w,
                rstd1,
                zdz1,
                torch.zeros_like(norm1),
                1,
            )
        )
        grad_pre0, zdz0, hidden0 = coda.gemm_partial_swiglu_bwd(grad_norm0, down_w, pre0)
        grad_residual, normalized0, partial_grad_norm = coda.gemm_residual_partial_rmsnorm_bwd(
            grad_pre0, up_w, norm0, norm_w, rstd0, zdz0, grad_norm0, 1
        )
        return (
            grad_residual @ attn_out,
            grad_residual,
            grad_residual.T @ attn_context,
            partial_grad_norm.sum(dim=-1),
            grad_pre0.T @ normalized0,
            grad_norm0.T @ hidden0,
            partial_grad_final_norm.sum(dim=-1),
            grad_logits.T @ normalized1,
            None,
        )


def coda_autograd_model_forward(
    x: Tensor,
    targets: Tensor,
    attn0_norm_weight: Tensor,
    attn0_qkv: Tensor,
    attn0_out: Tensor,
    mlp0_norm_weight: Tensor,
    mlp0_up: Tensor,
    mlp0_down: Tensor,
    attn1_norm_weight: Tensor,
    attn1_qkv: Tensor,
    attn1_out: Tensor,
    mlp1_norm_weight: Tensor,
    mlp1_up: Tensor,
    mlp1_down: Tensor,
    final_norm_weight: Tensor,
    lm_head: Tensor,
) -> Tensor:
    attn0_context = attention_context(rmsnorm(x, attn0_norm_weight), attn0_qkv)
    x, qkv = CodaFirstHalf.apply(
        attn0_context,
        x,
        attn0_out,
        mlp0_norm_weight,
        mlp0_up,
        mlp0_down,
        attn1_norm_weight,
        attn1_qkv,
    )
    attn1_context = attention_context_from_qkv(qkv, x)
    return CodaSecondHalf.apply(
        attn1_context,
        x,
        attn1_out,
        mlp1_norm_weight,
        mlp1_up,
        mlp1_down,
        final_norm_weight,
        lm_head,
        targets,
    )


def natural_model_forward(
    x: Tensor,
    targets: Tensor,
    attn0_norm_weight: Tensor,
    attn0_qkv: Tensor,
    attn0_out: Tensor,
    mlp0_norm_weight: Tensor,
    mlp0_up: Tensor,
    mlp0_down: Tensor,
    attn1_norm_weight: Tensor,
    attn1_qkv: Tensor,
    attn1_out: Tensor,
    mlp1_norm_weight: Tensor,
    mlp1_up: Tensor,
    mlp1_down: Tensor,
    final_norm_weight: Tensor,
    lm_head: Tensor,
) -> Tensor:
    residual = x
    x = rmsnorm(x, attn0_norm_weight)
    x = residual + attention(x, attn0_qkv, attn0_out)
    residual = x
    x = rmsnorm(x, mlp0_norm_weight)
    x = residual + F.linear(swiglu(F.linear(x, mlp0_up)), mlp0_down)
    residual = x
    x = rmsnorm(x, attn1_norm_weight)
    x = residual + attention(x, attn1_qkv, attn1_out)
    residual = x
    x = rmsnorm(x, mlp1_norm_weight)
    x = residual + F.linear(swiglu(F.linear(x, mlp1_up)), mlp1_down)
    logits = F.linear(rmsnorm(x, final_norm_weight), lm_head)
    return F.cross_entropy(logits, targets, reduction="none")


def make_inputs() -> tuple[Tensor, ...]:
    d_model = 8
    hidden = 16
    vocab = 5
    shapes = [
        (4, d_model),
        (4,),
        (d_model,), (d_model * 3, d_model), (d_model, d_model),
        (d_model,), (hidden * 2, d_model), (d_model, hidden),
        (d_model,), (d_model * 3, d_model), (d_model, d_model),
        (d_model,), (hidden * 2, d_model), (d_model, hidden),
        (d_model,), (vocab, d_model),
    ]
    return (torch.randn(*shapes[0]), torch.randint(vocab, shapes[1])) + tuple(
        torch.randn(*shape) for shape in shapes[2:]
    )


def clone_args(args: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
    return tuple(arg.detach().clone().requires_grad_(arg.is_floating_point()) for arg in args)


def full_forward_backward(fn, args):
    y = fn(*args).sum()
    grads = torch.autograd.grad(y, tuple(arg for arg in args if arg.requires_grad))
    return (y, *grads)


def call_targets(gm) -> list[str]:
    return [target_base_name(node.target) for node in gm.graph.nodes if node.op == "call_function"]


def target_base_name(target) -> str:
    return str(target)


def assert_close_all(actual, expected) -> None:
    for a, e in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, e, rtol=2e-5, atol=2e-5)


def assert_coda_counts(targets: list[str]) -> None:
    counts = Counter(target for target in targets if target.startswith("coda."))
    assert counts["coda.gemm_residual_partial_rmsnorm"] == 4
    assert counts["coda.rms_final_reduce"] == 4
    assert counts["coda.gemm_rmsnorm_swiglu"] == 2
    assert counts["coda.gemm_rmsnorm"] == 1
    assert counts["coda.gemm_rmsnorm_partial_cross_entropy"] == 1
    assert counts["coda.cross_entropy"] == 1
    assert counts["coda.gemm_partial_swiglu_bwd"] == 2
    assert counts["coda.gemm_residual_partial_rmsnorm_bwd"] == 4


def main() -> None:
    torch.manual_seed(0)
    args = make_inputs()
    coda_args = clone_args(args)
    natural_args = clone_args(args)
    gm = make_fx(
        lambda *xs: full_forward_backward(coda_autograd_model_forward, xs),
        tracing_mode="real",
    )(*coda_args)
    assert_coda_counts(call_targets(gm))
    assert_close_all(gm(*coda_args), full_forward_backward(natural_model_forward, natural_args))
    print("coda_autograd_function_example.py tests passed")


if __name__ == "__main__":
    main()
