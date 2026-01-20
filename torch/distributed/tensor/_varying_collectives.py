# mypy: allow-untyped-defs
"""Functional collectives with variant/invariant tensor/gradient semantics."""

import torch


def all_reduce_invariant(
    input: torch.Tensor,
    reduce_op: str,
    group_name: str,
) -> torch.Tensor:
    """Forward: all_reduce, Backward: identity (no grad aggregation)."""
    return torch.ops._c10d_functional.all_reduce_invariant(input, reduce_op, group_name)


def mark_varying(
    input: torch.Tensor,
    group_name: str,
) -> torch.Tensor:
    """Forward: identity (no-op), Backward: all_reduce(sum) gradients."""
    return torch.ops._c10d_functional.mark_varying(input, group_name)


@torch.library.custom_op("_c10d_functional::all_reduce_invariant", mutates_args=())
def all_reduce_invariant_op(
    input: torch.Tensor, reduce_op: str, group_name: str
) -> torch.Tensor:
    output = torch.ops._c10d_functional.all_reduce(input, reduce_op, group_name)
    return torch.ops._c10d_functional.wait_tensor(output)


@all_reduce_invariant_op.register_fake
def _(input: torch.Tensor, reduce_op: str, group_name: str) -> torch.Tensor:
    return torch.empty_like(input)


def all_reduce_invariant_backward(ctx, grad_output: torch.Tensor):
    if ctx.reduce_op != "sum":
        raise RuntimeError(
            f"all_reduce_invariant backward only supports 'sum' reduction, got '{ctx.reduce_op}'"
        )
    return grad_output, None, None


def all_reduce_invariant_setup_context(ctx, inputs, output):
    input, reduce_op, group_name = inputs
    ctx.reduce_op = reduce_op
    ctx.group_name = group_name
    return


all_reduce_invariant_op.register_autograd(
    all_reduce_invariant_backward,
    setup_context=all_reduce_invariant_setup_context,
)


@torch.library.custom_op(
    "_c10d_functional::mark_varying",
    mutates_args=(),
    #    schema="(Tensor(a) input, str group) -> Tensor(a)" # FIXME: compilation
)
def mark_varying_op(input: torch.Tensor, group_name: str) -> torch.Tensor:
    # return input.view_as(input) # FIXME: compilation
    return input.clone()


@mark_varying_op.register_fake
def _(input: torch.Tensor, group_name: str) -> torch.Tensor:
    # return input.view_as(input) # FIXME: compilation
    return torch.empty_like(input)


def mark_varying_backward(ctx, grad_output: torch.Tensor):
    group_name = ctx.group_name
    output = torch.ops._c10d_functional.all_reduce(
        grad_output.contiguous(), "sum", group_name
    )
    return torch.ops._c10d_functional.wait_tensor(output), None


def mark_varying_setup_context(ctx, inputs, output):
    input, group_name = inputs
    ctx.group_name = group_name
    return


mark_varying_op.register_autograd(
    mark_varying_backward,
    setup_context=mark_varying_setup_context,
)
