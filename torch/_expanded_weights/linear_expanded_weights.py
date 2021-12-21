import torch
import torch.nn.functional as F
from .expanded_weights_impl import forward_helper, implements_per_sample_grads
from .expanded_weights_utils import grad_if_exists, grad_if_exists_for_input, unpack_expanded_weight_or_tensor

@implements_per_sample_grads(F.linear)
class LinearPerSampleGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *expanded_args):
        if len(expanded_args[0].shape) <= 1:
            raise RuntimeError("Input does not have a batch dimension. Expanded Weights "
                               f"expected input of at least rank 2, got of rank {len(expanded_args[0].shape)}")
        return forward_helper(F.linear, ctx, expanded_args, 1)

    @staticmethod
    def backward(ctx, grad_output):
        (input, weight, bias) = ctx.args
        results = []

        results.append(grad_if_exists_for_input(input, lambda: grad_output.matmul(unpack_expanded_weight_or_tensor(weight))))
        results.append(grad_if_exists(weight, lambda _: torch.einsum("n...i,n...j->nij", grad_output, input)))
        results.append(grad_if_exists(bias, lambda _: torch.einsum("n...k->nk", grad_output)))
        return tuple(results)
