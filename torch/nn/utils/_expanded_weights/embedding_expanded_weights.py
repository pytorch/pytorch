import torch
import torch.nn.functional as F
from .expanded_weights_impl import implements_per_sample_grads
from .expanded_weights_utils import standard_kwargs, forward_helper, set_grad_sample_if_exists, grad_if_exists_for_input
from functools import partial

@implements_per_sample_grads(F.embedding)
class EmbeddingPerSampleGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *expanded_args_and_kwargs):
        expanded_args, expanded_kwargs = standard_kwargs(expanded_args_and_kwargs)
        if len(expanded_args[0].shape) == 1:
            raise RuntimeError(f"Expanded Weights needs an input with a batch size, got a 1D tensor, {expanded_args[0]}")
        output, aux_outputs = forward_helper(F.embedding, expanded_args, expanded_kwargs, 1)
        ctx.args = expanded_args
        ctx.kwargs = expanded_kwargs
        ctx.aux_outputs = aux_outputs
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.args
        padding_idx, scale_grad_by_freq, sparse = ctx.kwargs['padding_idx'], ctx.kwargs['scale_grad_by_freq'], ctx.kwargs['sparse']

        def input_grad(padding_idx):
            if padding_idx is not None:
                if padding_idx >= weight.shape[0]:
                    raise RuntimeError("Padding_idx must be within num_embeddings, "
                                       f"was ${padding_idx} but expected less than ${weight.shape[0]}")
                elif padding_idx < -weight.shape[0]:
                    raise RuntimeError("Padding_idx must be within num_embeddings, "
                                       f"was ${padding_idx} but expected more than -${weight.shape[0]}")
                elif padding_idx < 0:
                    padding_idx = weight.shape[0] + padding_idx
            else:
                padding_idx = -1
            return torch.ops.aten.embedding_backward(grad_output, input, weight.shape[0], padding_idx, scale_grad_by_freq, sparse)

        def weight_per_sample_grad(weight):
            batch_size = input.shape[0]
            embedding_dim = weight.shape[1]
            index = (
                input.unsqueeze(-1)
                .expand(*input.shape, embedding_dim)
                .reshape(batch_size, -1, embedding_dim)
            )
            grad_sample = torch.zeros(
                batch_size, *weight.shape, device=weight.device, dtype=grad_output.dtype
            )
            return grad_sample.scatter_add_(1, index, grad_output.reshape(batch_size, -1, embedding_dim))

        results = []
        results.append(grad_if_exists_for_input(input, partial(input_grad, padding_idx)))
        # weight doesn't compute batched gradients; no other arguments nor was_expanded are differentiable
        results = results + [None] * (len(ctx.args) + len(ctx.kwargs))

        # set grad_sample field for weight with per sample gradients
        set_grad_sample_if_exists(weight, weight_per_sample_grad)
        return tuple(results)
