from functools import partial
import torch
import torch.nn.functional as F
from .expanded_weights_impl import implements_per_sample_grads
from .expanded_weights_utils import \
    forward_helper, set_grad_sample_if_exists, standard_kwargs, unpack_expanded_weight_or_tensor
from typing import List, Optional

@implements_per_sample_grads(F.instance_norm)
class InstanceNormPerSampleGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, kwarg_names, _, *expanded_args_and_kwargs):
        instance_norm = partial(torch.instance_norm, cudnn_enabled=True)
        expanded_args, expanded_kwargs = standard_kwargs(kwarg_names, expanded_args_and_kwargs)
        output = forward_helper(instance_norm, expanded_args, expanded_kwargs)
        ctx.input = expanded_args[0]
        ctx.running_mean, ctx.running_var = expanded_kwargs['running_mean'], expanded_kwargs['running_var']
        ctx.weight, ctx.bias, ctx.eps = expanded_kwargs['weight'], expanded_kwargs['bias'], expanded_kwargs['eps']
        return output


    @staticmethod
    def backward(ctx, grad_output):
        input, running_mean, running_var = ctx.input, ctx.running_mean, ctx.running_var
        weight, bias, eps = ctx.weight, ctx.bias, ctx.eps

        results: List[Optional[torch.Tensor]] = []
        results.append(None)  # for kwarg names
        results.append(None)  # for op reference
        if input.requires_grad:
            b = input.shape[0]
            c = input.shape[1]
            new_shape = (1, b * c, *input.shape[2:])

            weight_ = unpack_expanded_weight_or_tensor(weight, lambda orig_weight: orig_weight.repeat(b))
            running_mean_ = running_mean.repeat(b) if running_mean is not None else None
            running_var_ = running_var.repeat(b) if running_var is not None else None
            input_reshaped = input.contiguous().view(new_shape)
            grad_output_reshaped = grad_output.contiguous().view(new_shape)
            mean = torch.mean(input_reshaped, (0,) + tuple(range(2, input.dim())), False)
            var = torch.var(input_reshaped, (0,) + tuple(range(2, input.dim())), keepdim=False, unbiased=False)
            rstd = 1 / torch.sqrt(var + eps)

            # must use native batch norm since it supports all inputs. This may have used cuda or openmi during the forward but
            # it didn't save the metadata, so we don't know during the backward
            res = torch.ops.aten.native_batch_norm_backward(
                grad_output_reshaped, input_reshaped, weight_, running_mean_, running_var_,
                mean, rstd, True, eps, (True, False, False))
            results.append(res[0].reshape(input.shape))
        else:
            results.append(None)

        # weight and bias don't compute batched gradients; no other arguments are differentiable (2 are not saved from the forward)
        results = results + [None] * 7

        # set grad_sample field for weight and bias with per sample gradients
        set_grad_sample_if_exists(weight,
                                  lambda _: torch.einsum("ni...->ni", F.instance_norm(input, eps=eps) * grad_output))
        set_grad_sample_if_exists(bias, lambda _: torch.einsum("ni...->ni", grad_output))
        return tuple(results)
