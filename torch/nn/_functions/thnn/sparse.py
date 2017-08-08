import torch
from torch.autograd.function import Function
from torch._thnn import type2backend
from torch.autograd.function import once_differentiable

from . import _all_functions


class Embedding(Function):

    @staticmethod
    def _renorm(ctx, indices, weight, max_norm, norm_type):
        if indices.dim() == 2:
            indices = indices.clone().view(-1)

        ctx._backend.LookupTable_renorm(
            ctx._backend.library_state,
            indices,
            weight,
            max_norm,
            norm_type
        )

    @classmethod
    def forward(cls, ctx, indices, weight, padding_idx, max_norm, norm_type, scale_grad_by_freq,
                sparse=False):

        ctx.padding_idx = padding_idx
        ctx.scale_grad_by_freq = scale_grad_by_freq
        ctx._indices = None
        ctx.sparse = sparse

        assert indices.dim() <= 2
        assert not ctx.needs_input_grad[0], "Embedding doesn't " \
            "compute the gradient w.r.t. the indices"

        ctx._backend = type2backend[type(weight)]
        ctx._weight_size = weight.size()

        if not indices.is_contiguous():
            ctx._indices = indices.contiguous()
            indices = ctx._indices
        else:
            ctx.save_for_backward(indices)

        output = weight.new()
        if max_norm is not None:
            cls._renorm(ctx, indices, weight, max_norm, norm_type)

        if indices.dim() == 1:
            output = torch.index_select(weight, 0, indices)
        else:
            output = torch.index_select(weight, 0, indices.view(-1))
            output = output.view(indices.size(0), indices.size(1), weight.size(1))

        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        if ctx._indices is not None:
            indices = ctx._indices
        else:
            indices, = ctx.saved_tensors

        grad_output = grad_output.contiguous()
        if not ctx.sparse:
            if indices.dim() == 2:
                indices = indices.view(-1)

            with torch.cuda.device_of(grad_output):
                if grad_output.is_cuda:
                    _sorted = torch.cuda.LongTensor()
                    _indices = torch.cuda.LongTensor()
                    _count = torch.cuda.LongTensor()
                else:
                    _count = torch.IntTensor()
                    _sorted = _indices = None

            grad_weight = grad_output.new(ctx._weight_size).zero_()
            # Doesn't support Variable grad_output
            ctx._backend.LookupTable_accGradParameters(
                ctx._backend.library_state,
                indices,
                grad_output,
                grad_weight,
                _count,
                _sorted,
                _indices,
                ctx.scale_grad_by_freq,
                ctx.padding_idx,
                1
            )
        else:
            tensor_type = type(grad_output).__name__
            if grad_output.is_cuda:
                SparseTensor = getattr(torch.cuda.sparse, tensor_type)
            else:
                SparseTensor = getattr(torch.sparse, tensor_type)
            grad_weight = SparseTensor(
                indices.view(1, -1),
                grad_output.view(-1, ctx._weight_size[1]),
                ctx._weight_size,
            )
        return None, grad_weight, None, None, None, None, None


_all_functions.append(Embedding)

MODE_SUM = 0
MODE_MEAN = 1


class EmbeddingBag(Function):

    def __init__(self, max_norm, norm_type, scale_grad_by_freq, mode):
        super(EmbeddingBag, self).__init__()
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self._indices = None
        assert mode is not None
        if mode == 'sum':
            self.mode = MODE_SUM
        elif mode == 'mean':
            self.mode = MODE_MEAN
        else:
            raise ValueError("mode needs to be 'sum' or 'mean', but got {}"
                             .format(mode))

    def _renorm(self, indices, weight):
        self._backend.LookupTable_renorm(
            self._backend.library_state,
            indices,
            weight,
            self.max_norm,
            self.norm_type
        )

    def forward(self, weight, indices, offsets):
        assert not self.needs_input_grad[1], "EmbeddingBag doesn't " \
            "compute the gradient w.r.t. the indices"

        assert not self.needs_input_grad[2], "EmbeddingBag doesn't " \
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

        self._backend = type2backend[type(weight)]
        self._weight_size = weight.size()
        self._offset2bag = offsets.new()

        self.save_for_backward(indices)

        indices = indices.contiguous().view(-1)
        output = weight.new()
        if self.max_norm is not None:
            self._renorm(indices, weight)

        if weight.is_cuda:
            if self.mode == MODE_MEAN:
                self.bag_size = offsets.new().resize_(offsets.size())
            else:
                self.bag_size = None

            self._backend.LookupTableBag_updateOutput(
                self._backend.library_state,
                indices,
                offsets,
                weight,
                output,
                self._offset2bag,
                self.mode,
                self.bag_size
            )
        else:
            # slow CPU implementation
            index_output = torch.index_select(weight, 0, indices)
            # indices = [1, 2, 30, 100, 12], offsets = [0, 2, 3]
            self._offset2bag.resize_(indices.size(0)).zero_()  # offset2bag = [0 0 0 0 0]
            self._offset2bag.index_fill_(0, offsets, 1)  # offset2bag = [1 0 1 0 1]
            self._offset2bag[0] = 0  # offset2bag = [0 0 1 0 1]
            self._offset2bag = self._offset2bag.cumsum(0)  # offset2bag = [0 0 1 1 2]
            output.resize_(offsets.size(0), weight.size(1)).zero_()
            output.index_add_(0, self._offset2bag, index_output)
            if self.mode == MODE_MEAN:
                if offsets.size(0) == 1:
                    self.bag_size = indices.size(0)
                else:
                    self.bag_size = weight.new().resize_(offsets.size())
                    self.bag_size[:-1] = offsets[1:] - offsets[:-1]
                    self.bag_size[-1] = indices.size(0) - offsets[-1]
                    self.bag_size = self.bag_size[:, None].expand_as(output)
                output /= self.bag_size

        return output

    def backward(self, grad_output):
        indices, = self.saved_tensors
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

        grad_weight = grad_output.new(self._weight_size).zero_()

        if grad_output.is_cuda:
            self._backend.LookupTableBag_accGradParameters(
                self._backend.library_state,
                indices,
                grad_output,
                grad_weight,
                self._offset2bag,
                _count,
                _sorted,
                _indices,
                self.scale_grad_by_freq,
                self.mode,
                self.bag_size,
                1
            )
        else:
            # slow CPU implementation
            if self.mode == MODE_MEAN:
                # divide by average count
                grad_output = grad_output / self.bag_size

            index_grad_output = grad_output.index_select(0, self._offset2bag)
            self._backend.LookupTable_accGradParameters(
                self._backend.library_state,
                indices,
                index_grad_output,
                grad_weight,
                _count,
                _sorted,
                _indices,
                self.scale_grad_by_freq,
                -1,
                1
            )

        return grad_weight, None, None


_all_functions.append(EmbeddingBag)
