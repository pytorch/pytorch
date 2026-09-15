# ruff: noqa: S101

import operator
from types import SimpleNamespace

import coda_infra
from custom_op import custom_op
from fx_pattern_utils import (
    find_pattern_matches,
    fx_pattern,
    fx_replacement,
    FxGraphEditor,
    replace_pattern,
)

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.fx import GraphModule
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.node import Node


EPS = 1e-6


def rms_norm(x: Tensor, weight: Tensor, eps: float = EPS) -> Tensor:
    rstd = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return x * rstd * weight


@custom_op("usrlib::rmsnorm_fwd")
def rmsnorm_fwd(
    x: Tensor,
    weight: Tensor,
    eps: float,
    coda: bool,
) -> Tensor:
    return rms_norm(x, weight, eps)


@custom_op("usrlib::rmsnorm_bwd")
def rmsnorm_bwd(
    grad_out: Tensor,
    x: Tensor,
    weight: Tensor,
    eps: float,
    coda: bool,
) -> tuple[Tensor, Tensor]:
    rstd = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    grad_unweighted = grad_out * weight
    correction = (grad_unweighted * x).mean(dim=-1, keepdim=True)
    grad_x = grad_unweighted * rstd - x * rstd.pow(3) * correction
    grad_weight = (grad_out * x * rstd).sum_to_size(weight.shape)
    return grad_x, grad_weight


class RmsNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps, coda):
        ctx.save_for_backward(x, weight)
        ctx.eps = eps
        ctx.coda = coda
        return rmsnorm_fwd(x, weight, eps, coda)

    @staticmethod
    def backward(ctx, grad_out):
        x, weight = ctx.saved_tensors
        grad_x, grad_weight = rmsnorm_bwd(grad_out, x, weight, ctx.eps, ctx.coda)
        return grad_x, grad_weight, None, None


def rmsnorm(x, weight, eps, coda=False):
    return RmsNorm.apply(x, weight, eps, coda)


def swiglu_math(x: Tensor) -> Tensor:
    gate, value = x.chunk(2, dim=-1)
    return torch.nn.functional.silu(gate) * value


@custom_op("usrlib::swiglu_fwd")
def swiglu_fwd(x: Tensor) -> Tensor:
    return swiglu_math(x)


@custom_op("usrlib::swiglu_bwd")
def swiglu_bwd(grad_out: Tensor, x: Tensor) -> Tensor:
    gate, value = x.chunk(2, dim=-1)
    sigmoid = torch.sigmoid(gate)
    silu = torch.nn.functional.silu(gate)
    grad_gate = grad_out * value * (sigmoid + silu * (1.0 - sigmoid))
    grad_value = grad_out * silu
    return torch.cat([grad_gate, grad_value], dim=-1)


class SwiGLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return swiglu_fwd(x)

    @staticmethod
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        return swiglu_bwd(grad_out, x)


def swiglu(x):
    return SwiGLU.apply(x)


def attention(
    x: Tensor,
    qkv_weight: Tensor,
    out_weight: Tensor,
) -> Tensor:
    num_heads = 2
    q, k, v = F.linear(x, qkv_weight).view(x.shape[0], 3, num_heads, -1).unbind(dim=1)
    q = q.transpose(0, 1).unsqueeze(0)
    k = k.transpose(0, 1).unsqueeze(0)
    v = v.transpose(0, 1).unsqueeze(0)
    x = (
        F.scaled_dot_product_attention(q, k, v)
        .squeeze(0)
        .transpose(0, 1)
        .reshape(x.shape)
    )
    return F.linear(x, out_weight)


def mlp(
    x: Tensor,
    up_weight: Tensor,
    down_weight: Tensor,
) -> Tensor:
    x = swiglu(F.linear(x, up_weight))
    return F.linear(x, down_weight)


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
    x = rmsnorm(x, attn0_norm_weight, EPS)
    x = residual + attention(x, attn0_qkv, attn0_out)
    residual = x
    x = rmsnorm(x, mlp0_norm_weight, EPS, coda=True)
    x = residual + mlp(x, mlp0_up, mlp0_down)
    residual = x
    x = rmsnorm(x, attn1_norm_weight, EPS, coda=True)
    x = residual + attention(x, attn1_qkv, attn1_out)
    residual = x
    x = rmsnorm(x, mlp1_norm_weight, EPS, coda=True)
    x = residual + mlp(x, mlp1_up, mlp1_down)
    x = rmsnorm(x, final_norm_weight, EPS, coda=True)
    logits = F.linear(x, lm_head)
    return F.cross_entropy(logits, targets, reduction="none")


usrlib = SimpleNamespace(
    rmsnorm_fwd=rmsnorm_fwd,
    rmsnorm_bwd=rmsnorm_bwd,
    swiglu_fwd=swiglu_fwd,
    swiglu_bwd=swiglu_bwd,
)
tmp = SimpleNamespace(
    rmsnorm_partial_forward=coda_infra.rmsnorm_partial_forward,
    rmsnorm_finalize_forward=coda_infra.rmsnorm_finalize_forward,
    rmsnorm_bwd_zdz=coda_infra.rmsnorm_bwd_zdz,
    rmsnorm_bwd_input=coda_infra.rmsnorm_bwd_input,
    rmsnorm_bwd_weight=coda_infra.rmsnorm_bwd_weight,
    rmsnorm_bwd_weight_reduce=coda_infra.rmsnorm_bwd_weight_reduce,
)
coda = SimpleNamespace(
    gemm_residual_partial_rmsnorm=coda_infra.gemm_residual_partial_rmsnorm,
    rms_final_reduce=coda_infra.rms_final_reduce,
    gemm_rmsnorm=coda_infra.gemm_rmsnorm,
    gemm_rmsnorm_swiglu=coda_infra.gemm_rmsnorm_swiglu,
    gemm_rmsnorm_partial_cross_entropy=coda_infra.gemm_rmsnorm_partial_cross_entropy,
    cross_entropy=coda_infra.cross_entropy,
    gemm_partial_swiglu_bwd=coda_infra.gemm_partial_swiglu_bwd,
    gemm_residual_partial_rmsnorm_bwd=coda_infra.gemm_residual_partial_rmsnorm_bwd,
)
aten = torch.ops.aten


def phase1(gm):
    """
    Phase 1: lower coda=True usrlib RMSNorm ops to decomposed tmp RMSNorm ops.
    """

    @fx_pattern(torch.empty(2, 8), torch.empty(2, 8), torch.empty(8))
    def phase1_pattern(m, grad_out, x, w):
        m.y = usrlib.rmsnorm_fwd(x, w, EPS, True)
        m.grads = usrlib.rmsnorm_bwd(grad_out, x, w, EPS, True)
        return m.y, m.grads

    graph = gm.graph
    editor = FxGraphEditor(gm)

    for m in find_pattern_matches(gm, phase1_pattern):
        old_y = m.y
        old_grads = m.grads
        _, _, eps, _ = old_y.args

        with editor.before(old_y):
            x_last_dim_size = editor.emit(aten.size, m.x, -1)
            partial_squares, weighted = editor.emit(
                tmp.rmsnorm_partial_forward, m.x, m.w, x_last_dim_size
            )
            saved_rstd = editor.emit(coda.rms_final_reduce, partial_squares, eps)
            new_y = editor.emit(tmp.rmsnorm_finalize_forward, weighted, saved_rstd)

        old_y.replace_all_uses_with(new_y)

        with editor.before(old_grads):
            zdz = editor.emit(tmp.rmsnorm_bwd_zdz, m.grad_out, m.x, m.w, saved_rstd)
            grad_x = editor.emit(
                tmp.rmsnorm_bwd_input,
                m.grad_out,
                m.x,
                {"weight": m.w, "rstd": saved_rstd},
                zdz,
            )
            partial_grad_weight = editor.emit(
                tmp.rmsnorm_bwd_weight, m.grad_out, m.x, saved_rstd, 1
            )
            grad_weight = editor.emit(
                tmp.rmsnorm_bwd_weight_reduce, partial_grad_weight
            )

        for user in list(old_grads.users):
            assert user.target is operator.getitem
            user.replace_all_uses_with(grad_x if user.args[1] == 0 else grad_weight)
            graph.erase_node(user)

        graph.erase_node(old_grads)
        graph.erase_node(old_y)

    graph.lint()
    gm.recompile()
    return gm


def apply_coda_transform(gm: GraphModule) -> GraphModule:
    """
    Rewrite a post-autograd graph into CODA GEMM+RMSNorm fused ops.

    Phase 1 lowers each coda=True usrlib RMSNorm forward/backward pair into
    decomposed RMSNorm ops: partial forward reduction, CODA final rstd reduce,
    forward finalize, ZdZ, input-gradient epilogue, and partial weight-gradient
    reduction. This makes Phase 2 a sequence of local find+replace patterns.

    Phase 2a is a set of independent replacements:
        GEMM -> residual add -> tmp.rmsnorm_partial_forward
            becomes coda.gemm_residual_partial_rmsnorm
        tmp.rmsnorm_finalize_forward -> GEMM -> SwiGLU
            becomes coda.gemm_rmsnorm_swiglu
        tmp.rmsnorm_finalize_forward -> LM head -> cross entropy
            becomes coda.gemm_rmsnorm_partial_cross_entropy
            plus coda.cross_entropy
        tmp.rmsnorm_finalize_forward -> plain GEMM
            becomes coda.gemm_rmsnorm

    Phase 2b rewrites the matching backward graph into:
        coda.gemm_partial_swiglu_bwd
        coda.gemm_residual_partial_rmsnorm_bwd
        pure GEMMs and reductions for the gradients around it

    This example marks all RMSNorms with a preceding residual-producing GEMM
    as coda=True. The initial attention RMSNorm remains ordinary usrlib because
    it starts the graph and has no preceding GEMM in this function.
    """
    phase1(gm)
    act = lambda: torch.empty(2, 8)
    mat = lambda: torch.empty(8, 8)
    vec = lambda: torch.empty(8)
    rstd = lambda: torch.empty(2, 1)
    mlp = lambda: torch.empty(2, 16)
    mlp_pre = lambda: torch.empty(2, 32)
    up = lambda: torch.empty(32, 8)
    down = lambda: torch.empty(8, 16)
    ce_act = lambda: torch.empty(4, 8)
    ce_rstd = lambda: torch.empty(4, 1)
    target = lambda: torch.zeros(4, dtype=torch.long)
    lm_head = lambda: torch.empty(5, 8)

    @fx_pattern(act(), mat(), act(), vec())
    def phase2a_pattern(m, gemm_x, gemm_weight, residual, rms_weight):
        m.gemm_rhs = aten.t.default(gemm_weight)
        m.gemm_out = aten.mm.default(gemm_x, m.gemm_rhs)
        m.norm_input = aten.add.Tensor(residual, m.gemm_out)
        m.rms_partial = tmp.rmsnorm_partial_forward(m.norm_input, rms_weight, 8)
        m.partial_squares = operator.getitem(m.rms_partial, 0)
        m.weighted = operator.getitem(m.rms_partial, 1)
        return m.norm_input, m.partial_squares, m.weighted

    @fx_replacement(act(), mat(), act(), vec())
    def phase2a_replacement(gemm_x, gemm_weight, residual, rms_weight):
        return coda.gemm_residual_partial_rmsnorm(
            gemm_x, gemm_weight, residual, rms_weight, residual.size(-1)
        )

    replace_pattern(gm, phase2a_pattern, phase2a_replacement)

    @fx_pattern(act(), rstd(), up())
    def phase2a_swiglu_pattern(m, weighted, saved_rstd, next_weight):
        m.normalized = tmp.rmsnorm_finalize_forward(weighted, saved_rstd)
        m.next_rhs = aten.t.default(next_weight)
        m.preactivation = aten.mm.default(m.normalized, m.next_rhs)
        m.activation = usrlib.swiglu_fwd(m.preactivation)
        return m.preactivation, m.activation

    @fx_replacement(act(), rstd(), up())
    def phase2a_swiglu_replacement(weighted, saved_rstd, next_weight):
        return coda.gemm_rmsnorm_swiglu(weighted, {"weight": next_weight}, saved_rstd)

    replace_pattern(gm, phase2a_swiglu_pattern, phase2a_swiglu_replacement)

    @fx_pattern(act(), mlp_pre(), down(), act(), vec(), rstd(), up())
    def phase2b_swiglu_pattern(
        m,
        grad_out,
        preactivation,
        down_weight,
        norm_input,
        rms_weight,
        saved_rstd,
        up_weight,
    ):
        m.down_rhs = aten.t.default(down_weight)
        m.down_rhs_t = aten.t.default(m.down_rhs)
        m.grad_swiglu = aten.mm.default(grad_out, m.down_rhs_t)
        m.grad_preactivation = usrlib.swiglu_bwd(m.grad_swiglu, preactivation)
        m.up_rhs = aten.t.default(up_weight)
        m.up_rhs_t = aten.t.default(m.up_rhs)
        m.grad_to_norm = aten.mm.default(m.grad_preactivation, m.up_rhs_t)
        m.zdz = tmp.rmsnorm_bwd_zdz(m.grad_to_norm, norm_input, rms_weight, saved_rstd)
        return m.grad_preactivation, m.grad_to_norm, m.zdz

    @fx_replacement(act(), down(), mlp_pre(), up())
    def phase2b_swiglu_replacement(grad_out, down_weight, preactivation, up_weight):
        grad_preactivation, zdz, _ = coda.gemm_partial_swiglu_bwd(
            grad_out, down_weight, preactivation
        )
        return grad_preactivation, grad_preactivation @ up_weight, zdz

    replace_pattern(gm, phase2b_swiglu_pattern, phase2b_swiglu_replacement)

    @fx_pattern(ce_act(), ce_rstd(), lm_head(), target(), ce_act(), vec())
    def phase2b_cross_entropy_pattern(
        m,
        weighted,
        saved_rstd,
        head_weight,
        targets,
        norm_input,
        rms_weight,
    ):
        m.normalized = tmp.rmsnorm_finalize_forward(weighted, saved_rstd)
        m.head_rhs = aten.t.default(head_weight)
        m.logits = aten.mm.default(m.normalized, m.head_rhs)
        m.log_probs = aten._log_softmax.default(m.logits, 1, False)
        m.loss_tuple = aten.nll_loss_forward.default(
            m.log_probs, targets, None, 0, -100
        )
        m.loss = operator.getitem(m.loss_tuple, 0)
        m.total_weight = operator.getitem(m.loss_tuple, 1)
        m.loss_sum = aten.sum.default(m.loss)
        m.ones = torch.ones_like(m.loss_sum)
        m.grad_loss = aten.expand.default(m.ones, [4])
        m.nll_grad = aten.nll_loss_backward.default(
            m.grad_loss, m.log_probs, targets, None, 0, -100, m.total_weight
        )
        m.log_probs_detached = aten.detach.default(m.log_probs)
        m.log_probs_detached_again = aten.detach.default(m.log_probs_detached)
        m.grad_logits = aten._log_softmax_backward_data.default(
            m.nll_grad, m.log_probs_detached_again, 1, torch.float32
        )
        m.grad_logits_t = aten.t.default(m.grad_logits)
        m.grad_head_mm = aten.mm.default(m.grad_logits_t, m.normalized)
        m.grad_head_t = aten.t.default(m.grad_head_mm)
        m.grad_head = aten.t.default(m.grad_head_t)
        m.head_rhs_t = aten.t.default(m.head_rhs)
        m.grad_to_norm = aten.mm.default(m.grad_logits, m.head_rhs_t)
        m.zdz = tmp.rmsnorm_bwd_zdz(m.grad_to_norm, norm_input, rms_weight, saved_rstd)
        m.grad_x = tmp.rmsnorm_bwd_input(
            m.grad_to_norm,
            norm_input,
            {"weight": rms_weight, "rstd": saved_rstd},
            m.zdz,
        )
        m.partial_grad_weight = tmp.rmsnorm_bwd_weight(
            m.grad_to_norm, norm_input, saved_rstd, 1
        )
        m.grad_weight = tmp.rmsnorm_bwd_weight_reduce(m.partial_grad_weight)
        return m.loss_sum, m.grad_logits, m.zdz, m.grad_x, m.grad_weight, m.grad_head

    @fx_replacement(ce_act(), ce_rstd(), lm_head(), target(), ce_act(), vec())
    def phase2b_cross_entropy_replacement(
        weighted,
        saved_rstd,
        head_weight,
        targets,
        norm_input,
        rms_weight,
    ):
        logits, logits_tgt, logits_lse = coda.gemm_rmsnorm_partial_cross_entropy(
            weighted, head_weight, saved_rstd, targets, head_weight.size(0)
        )
        loss, grad_logits, zdz = coda.cross_entropy(
            logits, targets, logits_tgt, logits_lse
        )
        grad_x, normalized, partial_grad_weight = (
            coda.gemm_residual_partial_rmsnorm_bwd(
                grad_logits,
                head_weight,
                norm_input,
                rms_weight,
                saved_rstd,
                zdz,
                torch.zeros_like(norm_input),
                1,
            )
        )
        return (
            loss.sum(),
            grad_logits,
            zdz,
            grad_x,
            partial_grad_weight.sum(dim=-1),
            grad_logits.T @ normalized,
        )

    replace_pattern(
        gm, phase2b_cross_entropy_pattern, phase2b_cross_entropy_replacement
    )

    @fx_pattern(mlp_pre(), mlp(), down(), act(), vec(), up(), act(), rstd())
    def phase2b_pattern1(
        m,
        grad_out,
        gemm_x,
        gemm_weight,
        residual,
        rms_weight,
        next_weight,
        incoming_residual_grad,
        zdz,
    ):
        m.gemm_rhs = aten.t.default(gemm_weight)
        m.gemm_fwd = coda.gemm_residual_partial_rmsnorm(
            gemm_x, gemm_weight, residual, rms_weight, 8
        )
        m.norm_input = operator.getitem(m.gemm_fwd, 0)
        m.partial_squares = operator.getitem(m.gemm_fwd, 1)
        m.weighted = operator.getitem(m.gemm_fwd, 2)
        m.rstd = coda.rms_final_reduce(m.partial_squares, EPS)
        m.normalized = tmp.rmsnorm_finalize_forward(m.weighted, m.rstd)
        m.grad_out_t = aten.t.default(grad_out)
        m.grad_next_weight_mm = aten.mm.default(m.grad_out_t, m.normalized)
        m.grad_next_weight_t = aten.t.default(m.grad_next_weight_mm)
        m.grad_next_weight = aten.t.default(m.grad_next_weight_t)
        m.grad_to_norm = aten.mm.default(grad_out, next_weight)
        m.grad_from_norm = tmp.rmsnorm_bwd_input(
            m.grad_to_norm, m.norm_input, {"weight": rms_weight, "rstd": m.rstd}, zdz
        )
        m.partial_grad_weight = tmp.rmsnorm_bwd_weight(
            m.grad_to_norm, m.norm_input, m.rstd, 1
        )
        m.grad_weight = tmp.rmsnorm_bwd_weight_reduce(m.partial_grad_weight)
        m.total_grad = aten.add.Tensor(incoming_residual_grad, m.grad_from_norm)
        m.total_grad_t = aten.t.default(m.total_grad)
        m.grad_weight_mm = aten.mm.default(m.total_grad_t, gemm_x)
        m.grad_weight_t = aten.t.default(m.grad_weight_mm)
        m.gemm_rhs_t = aten.t.default(m.gemm_rhs)
        m.grad_gemm_x = aten.mm.default(m.total_grad, m.gemm_rhs_t)
        return (
            m.grad_next_weight,
            m.total_grad,
            m.grad_gemm_x,
            m.grad_weight_t,
            m.grad_weight,
        )

    @fx_replacement(
        mlp_pre(), mlp(), down(), act(), vec(), up(), act(), act(), rstd(), rstd()
    )
    def phase2b_replacement1(
        grad_out,
        gemm_x,
        gemm_weight,
        residual,
        rms_weight,
        next_weight,
        incoming_residual_grad,
        norm_input,
        rstd,
        zdz,
    ):
        grad_bwd = coda.gemm_residual_partial_rmsnorm_bwd(
            grad_out,
            next_weight,
            norm_input,
            rms_weight,
            rstd,
            zdz,
            incoming_residual_grad,
            1,
        )
        grad_residual, normalized, partial_grad_rms_weight = grad_bwd
        return (
            grad_out.T @ normalized,
            grad_residual,
            grad_residual @ gemm_weight,
            (grad_residual.T @ gemm_x).T,
            partial_grad_rms_weight.sum(dim=-1),
        )

    replace_pattern(gm, phase2b_pattern1, phase2b_replacement1)

    @fx_pattern(act(), rstd(), mat())
    def phase2a_gemm_pattern(m, weighted, saved_rstd, next_weight):
        m.normalized = tmp.rmsnorm_finalize_forward(weighted, saved_rstd)
        m.next_rhs = aten.t.default(next_weight)
        m.next_gemm = aten.mm.default(m.normalized, m.next_rhs)
        return m.next_gemm

    @fx_replacement(act(), rstd(), mat())
    def phase2a_gemm_replacement(weighted, saved_rstd, next_weight):
        return coda.gemm_rmsnorm(weighted, next_weight, saved_rstd)

    replace_pattern(gm, phase2a_gemm_pattern, phase2a_gemm_replacement)

    graph = gm.graph
    for node in list(graph.nodes):
        if (
            node.op != "call_function"
            or (
                node.target is not tmp.rmsnorm_bwd_zdz
                and node.target not in tmp.rmsnorm_bwd_zdz._cache.values()
            )
            or not node.users
        ):
            continue
        grad_out, norm_input, rms_weight, saved_rstd = node.args
        with graph.inserting_before(node):
            zdz = graph.call_function(aten.mul.Tensor, (grad_out, norm_input))
            zdz = graph.call_function(aten.mul.Tensor, (zdz, saved_rstd))
            zdz = graph.call_function(aten.mul.Tensor, (zdz, rms_weight))
            zdz = graph.call_function(aten.sum.dim_IntList, (zdz, [-1], True))
        node.replace_all_uses_with(zdz)
        graph.erase_node(node)

    gm.graph.eliminate_dead_code()
    gm.graph.lint()
    gm.recompile()
    return gm


def clone_graph(gm: GraphModule) -> GraphModule:
    new_graph = torch.fx.Graph()
    env: dict[Node, Node] = {}
    for node in gm.graph.nodes:
        env[node] = new_graph.node_copy(node, lambda n: env[n])
        env[node].meta.update(node.meta)
    return GraphModule(gm, new_graph)


def make_inputs() -> tuple[Tensor, ...]:
    d_model = 8
    hidden = 16
    vocab = 5
    shapes = [
        (4, d_model),
        (4,),
        (d_model,),
        (d_model * 3, d_model),
        (d_model, d_model),
        (d_model,),
        (hidden * 2, d_model),
        (d_model, hidden),
        (d_model,),
        (d_model * 3, d_model),
        (d_model, d_model),
        (d_model,),
        (hidden * 2, d_model),
        (d_model, hidden),
        (d_model,),
        (vocab, d_model),
    ]
    return (torch.randn(*shapes[0]), torch.randint(vocab, shapes[1])) + tuple(
        torch.randn(*shape) for shape in shapes[2:]
    )


def call_targets(gm: GraphModule) -> list[str]:
    return [
        target_base_name(n.target) for n in gm.graph.nodes if n.op == "call_function"
    ]


def target_base_name(target) -> str:
    return str(target)


def assert_has(targets: list[str], *names: str) -> None:
    assert all(name in targets for name in names)


def assert_target_count(targets: list[str], name: str, count: int) -> None:
    assert targets.count(name) == count


def assert_only_expected_coda_targets(targets: list[str]) -> None:
    expected = {
        "coda.cross_entropy",
        "coda.gemm_partial_swiglu_bwd",
        "coda.gemm_residual_partial_rmsnorm",
        "coda.gemm_residual_partial_rmsnorm_bwd",
        "coda.gemm_rmsnorm",
        "coda.gemm_rmsnorm_partial_cross_entropy",
        "coda.gemm_rmsnorm_swiglu",
        "coda.rms_final_reduce",
    }
    assert {target for target in targets if target.startswith("coda.")} <= expected


def assert_close_all(actual, expected) -> None:
    for a, e in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, e, rtol=1e-5, atol=1e-5)


def full_forward_backward(fn, args):
    y = fn(*args).sum()
    grads = torch.autograd.grad(y, tuple(arg for arg in args if arg.requires_grad))
    return (y, *grads)


def main() -> None:
    torch.manual_seed(0)
    args = make_inputs()

    full_args = tuple(
        arg.detach().requires_grad_(arg.is_floating_point()) for arg in args
    )
    pre_coda_fw_bw_gm = make_fx(
        lambda *xs: full_forward_backward(natural_model_forward, xs),
        tracing_mode="real",
    )(*full_args)
    fused_post_autograd_gm = apply_coda_transform(clone_graph(pre_coda_fw_bw_gm))
    pre_coda_targets = call_targets(pre_coda_fw_bw_gm)
    assert_has(
        pre_coda_targets,
        "usrlib.rmsnorm_fwd",
        "usrlib.rmsnorm_bwd",
    )
    fused_post_autograd_targets = call_targets(fused_post_autograd_gm)
    assert_target_count(fused_post_autograd_targets, "usrlib.rmsnorm_fwd", 1)
    assert_target_count(fused_post_autograd_targets, "usrlib.rmsnorm_bwd", 1)
    assert all(not target.startswith("tmp.") for target in fused_post_autograd_targets)
    assert "usrlib.swiglu_fwd" not in fused_post_autograd_targets
    assert "usrlib.swiglu_bwd" not in fused_post_autograd_targets
    assert "aten._log_softmax.default" not in fused_post_autograd_targets
    assert "aten.nll_loss_forward.default" not in fused_post_autograd_targets
    assert_only_expected_coda_targets(fused_post_autograd_targets)
    for node in fused_post_autograd_gm.graph.nodes:
        if target_base_name(node.target) == "usrlib.rmsnorm_fwd":
            assert node.args[3] is False
        if target_base_name(node.target) == "usrlib.rmsnorm_bwd":
            assert node.args[4] is False
    assert_has(
        fused_post_autograd_targets,
        "coda.gemm_residual_partial_rmsnorm",
        "coda.rms_final_reduce",
        "coda.gemm_rmsnorm_swiglu",
        "coda.gemm_rmsnorm",
        "coda.gemm_rmsnorm_partial_cross_entropy",
        "coda.cross_entropy",
        "coda.gemm_partial_swiglu_bwd",
        "coda.gemm_residual_partial_rmsnorm_bwd",
    )
    assert_target_count(
        fused_post_autograd_targets,
        "coda.gemm_residual_partial_rmsnorm",
        4,
    )
    assert_target_count(fused_post_autograd_targets, "coda.rms_final_reduce", 4)
    assert_target_count(fused_post_autograd_targets, "coda.gemm_rmsnorm_swiglu", 2)
    assert_target_count(fused_post_autograd_targets, "coda.gemm_rmsnorm", 1)
    assert_target_count(
        fused_post_autograd_targets,
        "coda.gemm_rmsnorm_partial_cross_entropy",
        1,
    )
    assert_target_count(fused_post_autograd_targets, "coda.cross_entropy", 1)
    assert_target_count(fused_post_autograd_targets, "coda.gemm_partial_swiglu_bwd", 2)
    assert_target_count(
        fused_post_autograd_targets,
        "coda.gemm_residual_partial_rmsnorm_bwd",
        4,
    )
    pre_coda_outputs = pre_coda_fw_bw_gm(*full_args)
    assert_close_all(fused_post_autograd_gm(*full_args), pre_coda_outputs)
    print("coda_custom_op_example.py tests passed")


if __name__ == "__main__":
    main()
