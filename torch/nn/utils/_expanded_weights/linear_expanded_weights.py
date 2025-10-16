# mypy: allow-untyped-defs
from typing import Optional

import torch
import torch.nn.functional as F

from .expanded_weights_impl import implements_per_sample_grads
from .expanded_weights_utils import (
    forward_helper,
    is_batch_first,
    set_grad_sample_if_exists,
    unpack_expanded_weight_or_tensor,
)


@implements_per_sample_grads(F.linear)
class LinearPerSampleGrad(torch.autograd.Function):
    @staticmethod
    # pyrefly: ignore  # bad-override
    def forward(ctx, _, __, *expanded_args_and_kwargs):
        if len(expanded_args_and_kwargs[0].shape) <= 1:
            raise RuntimeError(
                "Input does not have a batch dimension. Expanded Weights expected input "
                f"of at least rank 2, got of rank {len(expanded_args_and_kwargs[0].shape)}"
            )
        expanded_kwargs = {
            "bias": expanded_args_and_kwargs[2]
            if len(expanded_args_and_kwargs) == 3
            else None
        }
        expanded_args = expanded_args_and_kwargs[:2]
        ctx.batch_first = is_batch_first(expanded_args_and_kwargs)
        output = forward_helper(F.linear, expanded_args, expanded_kwargs)
        ctx.args = expanded_args
        ctx.kwargs = expanded_kwargs
        return output

    @staticmethod
    # pyrefly: ignore  # bad-override
    def backward(ctx, grad_output):
        input, weight = ctx.args
        bias = ctx.kwargs["bias"]
        results: list[Optional[torch.Tensor]] = []
        results.append(None)  # for kwarg_names
        results.append(None)  # for op reference

        if input.requires_grad:
            results.append(grad_output.matmul(unpack_expanded_weight_or_tensor(weight)))
        else:
            results.append(None)
        results.extend([None] * 2)  # weight and bias don't compute batched gradients

        if not ctx.batch_first:
            grad_output = grad_output.transpose(0, 1)
            input = input.transpose(0, 1)

        # weight and bias get their grad_sample fields set directly if they exist
        set_grad_sample_if_exists(
            weight, lambda _: torch.einsum("n...i,n...j->nij", grad_output, input)
        )
        set_grad_sample_if_exists(
            bias, lambda _: torch.einsum("n...k->nk", grad_output)
        )
        return tuple(results)
