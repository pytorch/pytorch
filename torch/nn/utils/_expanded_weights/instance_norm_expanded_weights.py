
from functools import partial
import torch
import torch.nn.functional as F
from .expanded_weights_impl import implements_per_sample_grads
from .expanded_weights_utils import forward_helper, grad_if_exists, grad_if_exists_for_input, unpack_expanded_weight_or_tensor

@implements_per_sample_grads(F.instance_norm)
class InstanceNormPerSampleGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *expanded_args):
        instance_norm = partial(torch._instance_norm_all_outputs, cudnn_enabled=True)
        output, expanded_args, aux_outputs = forward_helper(instance_norm, expanded_args, 1)
        ctx.args = expanded_args
        ctx.aux_outputs = aux_outputs
        return output


    @staticmethod
    def backward(ctx, grad_output):
        (input, running_mean, running_var, weight, bias, _, _, eps) = ctx.args
        (mean, rstd, reserve, idx) = ctx.aux_outputs

        def input_grad():
            b = input.shape[0]
            c = input.shape[1]
            new_shape = (1, b * c, *input.shape[2:])

            weight_ = unpack_expanded_weight_or_tensor(weight, lambda orig_weight: orig_weight.repeat(b))
            running_mean_ = running_mean.repeat(b) if running_mean is not None else None
            running_var_ = running_var.repeat(b) if running_var is not None else None
            input_reshaped = input.contiguous().view(new_shape)
            grad_output_reshaped = grad_output.contiguous().view(new_shape)
            res = torch.ops.aten._batch_norm_impl_index_backward(
                idx, input_reshaped, grad_output_reshaped, weight_, running_mean_, running_var_,
                mean, rstd, True, eps, (True, False, False), reserve)
            return res[0].reshape(input.shape)

        results = []
        results.append(grad_if_exists_for_input(input, input_grad))
        results.append(None)  # for running_mean
        results.append(None)  # for running_var
        results.append(grad_if_exists(weight,
                                      lambda _: torch.einsum("ni...->ni", F.instance_norm(input, eps=eps) * grad_output)))
        results.append(grad_if_exists(bias, lambda _: torch.einsum("ni...->ni", grad_output)))

        # no other arguments nor was_expanded are differentiable
        results = results + [None] * (len(ctx.args) - 3)
        return tuple(results)
