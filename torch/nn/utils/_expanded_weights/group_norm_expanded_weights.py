import torch
import torch.nn.functional as F
from .expanded_weights_impl import implements_per_sample_grads
from .expanded_weights_utils import standard_kwargs, \
    forward_helper, set_grad_sample_if_exists, grad_if_exists_for_input, unpack_expanded_weight_or_tensor

@implements_per_sample_grads(F.group_norm)
class GroupNormPerSampleGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *expanded_args_and_kwargs):
        expanded_args, expanded_kwargs = standard_kwargs(expanded_args_and_kwargs)
        output, aux_outputs = forward_helper(torch._group_norm_all_outputs, expanded_args, expanded_kwargs, 1)
        ctx.args = expanded_args
        ctx.kwargs = expanded_kwargs
        ctx.aux_outputs = aux_outputs
        return output[0]  # original function returns a single element, forward_helper returns a tuple

    @staticmethod
    def backward(ctx, grad_output):
        def input_grad():
            weight_c = unpack_expanded_weight_or_tensor(weight, lambda t: t.contiguous())
            input_c = input.contiguous()
            grad_output_c = grad_output.contiguous() if grad_output is not None else None
            N = input.shape[0]
            C = input.shape[1]
            HxW = 1
            for s in input.shape[2:]:
                HxW *= s
            input_grad_fn = torch.ops.aten.native_group_norm_backward
            return input_grad_fn(grad_output_c, input_c, mean, rstd, weight_c, N, C, HxW, num_groups, (True, False, False))[0]

        input, num_groups = ctx.args
        weight, bias, eps = ctx.kwargs['weight'], ctx.kwargs['bias'], ctx.kwargs['eps']
        mean, rstd = ctx.aux_outputs

        results = []
        results.append(grad_if_exists_for_input(input, input_grad))
        # weight and bias don't compute batched gradients; no other arguments are differentiable
        results = results + [None] * (len(ctx.args) + len(ctx.kwargs))

        # set grad_sample field for weight and bias with per sample gradients
        set_grad_sample_if_exists(weight,
                                  lambda _: torch.einsum("ni...->ni", F.group_norm(input, num_groups, eps=eps) * grad_output))
        set_grad_sample_if_exists(bias, lambda _: torch.einsum("ni...->ni", grad_output))
        return tuple(results)
