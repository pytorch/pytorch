
import torch
import torch.nn.functional as F
from .expanded_weights_impl import implements_per_sample_grads
from .expanded_weights_utils import forward_helper, set_grad_sample_if_exists, \
    grad_if_exists_for_input, standard_kwargs, sum_over_all_but_batch_and_last_n, unpack_expanded_weight_or_tensor

@implements_per_sample_grads(F.layer_norm)
class LayerNormPerSampleGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *expanded_args_and_kwargs):
        expanded_args, expanded_kwargs = standard_kwargs(expanded_args_and_kwargs)
        input = expanded_args[0]
        normalized_shape = expanded_args[1]
        if len(input.shape) <= len(normalized_shape):
            raise RuntimeError("Expanded Weights: Layer norm should not normalize over batch dimension for per sample gradient"
                               f"computations but got that normalized shape, {normalized_shape}, matched input shape.")
        output, aux_outputs = forward_helper(torch.native_layer_norm, expanded_args, expanded_kwargs, 1)
        ctx.args = expanded_args
        ctx.kwargs = expanded_kwargs
        ctx.aux_outputs = aux_outputs
        return output[0]  # original function returns a single element, forward_helper returns a tuple


    @staticmethod
    def backward(ctx, grad_output):
        def input_grad():
            weight_ = unpack_expanded_weight_or_tensor(weight)
            bias_ = unpack_expanded_weight_or_tensor(bias)
            return torch.ops.aten.native_layer_norm_backward(
                grad_output, input, normalized_shape, mean, rstd, weight_, bias_, (True, False, False))[0]

        def weight_per_sample_grad(weight):
            return sum_over_all_but_batch_and_last_n(F.layer_norm(input, normalized_shape, eps=eps) * grad_output, weight.dim())

        input, normalized_shape = ctx.args
        weight, bias, eps = ctx.kwargs['weight'], ctx.kwargs['bias'], ctx.kwargs['eps']
        mean, rstd = ctx.aux_outputs

        results = []
        results.append(grad_if_exists_for_input(input, input_grad))
        # weight and bias don't compute batched gradients; no other arguments are differentiable
        results = results + [None] * (len(ctx.args) + len(ctx.kwargs))

        # set grad_sample field for weight and bias with per sample gradients
        results.append(set_grad_sample_if_exists(weight, weight_per_sample_grad))
        results.append(set_grad_sample_if_exists(bias, lambda bias: sum_over_all_but_batch_and_last_n(grad_output, bias.dim())))
        return tuple(results)
