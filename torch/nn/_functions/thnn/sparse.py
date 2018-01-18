import torch
from torch.autograd.function import Function
from torch._thnn import type2backend
from torch.autograd.function import once_differentiable

from . import _all_functions


MODE_SUM = 0
MODE_MEAN = 1


class EmbeddingBag(Function):

    @staticmethod
    def _renorm(ctx, indices, weight, max_norm, norm_type):
        # clone indices since LookupTable_renorm modifies it in-place
        ctx._backend.LookupTable_renorm(
            ctx._backend.library_state,
            indices.clone().view(-1),
            weight,
            max_norm,
            norm_type
        )

    @classmethod
    def forward(cls, ctx, weight, indices, offsets,
                max_norm, norm_type, scale_grad_by_freq, mode):

        ctx.max_norm = max_norm
        ctx.norm_type = norm_type
        ctx.scale_grad_by_freq = scale_grad_by_freq

        if mode == 'sum':
            ctx.mode = MODE_SUM
        elif mode == 'mean':
            ctx.mode = MODE_MEAN
        else:
            raise ValueError("mode needs to be 'sum' or 'mean', but got {}"
                             .format(mode))

        assert not ctx.needs_input_grad[1], "EmbeddingBag doesn't " \
            "compute the gradient w.r.t. the indices"

        assert not ctx.needs_input_grad[2], "EmbeddingBag doesn't " \
            "compute the gradient w.r.t. the offsets"

        assert indices.dim() == 1
        if offsets.dim() != 1:
            raise ValueError("offsets has to be a 1D Tensor")

        if offsets[0] != 0:
            raise ValueError("offsets[0] has to be 0, i.e. the first sequence"
                             " in the mini-batch has to start from position 0."
                             "However, got {}".format(offsets[0]))
        if offsets[-1] > indices.size(0):
            raise ValueError("offsets[-1] has to be smaller than indices's length"
                             " ({}), but got offsets[-1] of {}"
                             .format(indices.size(0), offsets[-1]))

        ctx._backend = type2backend[weight.type()]
        ctx._weight_size = weight.size()
        ctx._offset2bag = offsets.new()

        ctx.save_for_backward(indices)

        indices = indices.contiguous().view(-1)
        output = weight.new()

        if ctx.max_norm is not None:
            cls._renorm(ctx, indices, weight, max_norm=max_norm, norm_type=norm_type)

        if weight.is_cuda:
            if ctx.mode == MODE_MEAN:
                ctx.bag_size = offsets.new().resize_(offsets.size())
            else:
                ctx.bag_size = None

            ctx._backend.LookupTableBag_updateOutput(
                ctx._backend.library_state,
                indices,
                offsets,
                weight,
                output,
                ctx._offset2bag,
                ctx.mode,
                ctx.bag_size
            )
        else:
            # slow CPU implementation
            index_output = torch.index_select(weight, 0, indices)
            # indices = [1, 2, 30, 100, 12], offsets = [0, 2, 3]
            ctx._offset2bag.resize_(indices.size(0)).zero_()  # offset2bag = [0 0 0 0 0]
            ctx._offset2bag.index_fill_(0, offsets, 1)  # offset2bag = [1 0 1 0 1]
            ctx._offset2bag[0] = 0  # offset2bag = [0 0 1 0 1]
            ctx._offset2bag = ctx._offset2bag.cumsum(0)  # offset2bag = [0 0 1 1 2]
            output.resize_(offsets.size(0), weight.size(1)).zero_()
            output.index_add_(0, ctx._offset2bag, index_output)
            if ctx.mode == MODE_MEAN:
                if offsets.size(0) == 1:
                    ctx.bag_size = indices.size(0)
                else:
                    ctx.bag_size = weight.new().resize_(offsets.size())
                    ctx.bag_size[:-1] = offsets[1:] - offsets[:-1]
                    ctx.bag_size[-1] = indices.size(0) - offsets[-1]
                    ctx.bag_size = ctx.bag_size[:, None].expand_as(output)
                output /= ctx.bag_size

        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        indices, = ctx.saved_tensors
        indices = indices.contiguous().view(-1)
        grad_output = grad_output.contiguous()

        with torch.cuda.device_of(grad_output):
            if grad_output.is_cuda:
                _sorted = torch.cuda.LongTensor()
                _indices = torch.cuda.LongTensor()
                _count = torch.cuda.LongTensor()
            else:
                _count = torch.IntTensor()
                _sorted = _indices = None

        grad_weight = grad_output.new(ctx._weight_size).zero_()

        if grad_output.is_cuda:
            ctx._backend.LookupTableBag_accGradParameters(
                ctx._backend.library_state,
                indices,
                grad_output,
                grad_weight,
                ctx._offset2bag,
                _count,
                _sorted,
                _indices,
                ctx.scale_grad_by_freq,
                ctx.mode,
                ctx.bag_size,
                1
            )
        else:
            # slow CPU implementation
            if ctx.mode == MODE_MEAN:
                # divide by average count
                grad_output = grad_output / ctx.bag_size

            index_grad_output = grad_output.index_select(0, ctx._offset2bag)
            ctx._backend.LookupTable_accGradParameters(
                ctx._backend.library_state,
                indices,
                index_grad_output,
                grad_weight,
                _count,
                _sorted,
                _indices,
                ctx.scale_grad_by_freq,
                -1,
                1
            )

        return grad_weight, None, None, None, None, None, None


_all_functions.append(EmbeddingBag)
