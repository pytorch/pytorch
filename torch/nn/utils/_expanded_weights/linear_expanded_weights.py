import torch
import torch.nn.functional as F
from .expanded_weights_impl import implements_per_sample_grads
from .expanded_weights_utils import \
    forward_helper, set_grad_sample_if_exists, grad_if_exists_for_input, unpack_expanded_weight_or_tensor

@implements_per_sample_grads(F.linear)
class LinearPerSampleGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *expanded_args):
        if len(expanded_args[0].shape) <= 1:
            raise RuntimeError("Input does not have a batch dimension. Expanded Weights "
                               f"expected input of at least rank 2, got of rank {len(expanded_args[0].shape)}")
        expanded_args = expanded_args[:-1]
        expanded_kwargs = {'bias': expanded_args[2] if len(expanded_args) == 3 else None}
        expanded_args_without_kwargs = expanded_args[:2]
        output, aux_outputs = forward_helper(F.linear, expanded_args_without_kwargs, expanded_kwargs, 1)
        ctx.args = expanded_args_without_kwargs
        ctx.kwargs = expanded_kwargs
        ctx.aux_outputs = aux_outputs
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.args
        bias = ctx.kwargs['bias']
        results = []

        results.append(grad_if_exists_for_input(input, lambda: grad_output.matmul(unpack_expanded_weight_or_tensor(weight))))
        results.extend([None] * 3)  # weight and bias don't compute batched gradients

        # weight and bias have their grad_sample fields set directly
        set_grad_sample_if_exists(weight, lambda _: torch.einsum("n...i,n...j->nij", grad_output, input))
        set_grad_sample_if_exists(bias, lambda _: torch.einsum("n...k->nk", grad_output))
        return tuple(results)
