#include <torch/csrc/jit/symbolic_script.h>

namespace torch {
namespace jit {
namespace {
std::mutex lock;
const std::vector<std::string> functions = {
    R"(

        ####     HELPER FUNCTIONS           ###
        ####     PREFIX: AD_                ###
        ####     SCHEMA NOT SAVED IN CACHE  ###

        def AD_unsqueeze_multiple(t,
                                  dims: List[int],
                                  n_dims: int):
            seen = [False] * n_dims
            for i in range(len(dims)):
                seen[dims[i]] = True

            for d in range(n_dims):
                if seen[d]:
                    t = t.unsqueeze(d)
            return t

        def AD_sum_backward(grad,
                            sizes: List[int],
                            dims: List[int],
                            keepdim: bool):
            if not keepdim and len(sizes) > 0:
                if len(dims) == 1:
                    return grad.unsqueeze(dims[0]).expand(sizes)
                else:
                    res = AD_unsqueeze_multiple(grad, dims, len(sizes))
                    return res.expand(sizes)
            else:
                return grad.expand(sizes)

        def AD_logsumexp_backward(grad, self, result,
                                  dim: List[int],
                                  keepdim: bool):
            if not keepdim and self.dim() != 0:
                n_dims = len(self.size())
                grad = AD_unsqueeze_multiple(grad, dim, n_dims)
                result = AD_unsqueeze_multiple(result, dim, n_dims)
            return grad * (self - result).exp()

        def mean_0(self):
            self_size = self.size()
            self_numel = self.numel()
            def backward(grad_output):
                grad_self = grad_output.expand(self_size) / self_numel
                return grad_self

            return torch.mean(self), backward

        def mean_1(self,
                   dim: List[int],
                   keepdim: bool):
            self_size = self.size()
            def backward(grad_output):
                grad_self = AD_sum_backward(grad_output, self_size, dim, keepdim) / AD_safe_size(self_size, dim)
                return grad_self, None, None

            return torch.mean(self, dim, keepdim), backward

        def logsumexp(self,
                      dim: List[int],
                      keepdim: bool):
            result = torch.logsumexp(self, dim, keepdim)
            self_dim = self.dim()
            def backward(grad_output):
                grad_self = AD_logsumexp_backward(grad_output, self, result, dim, keepdim)
                return grad_self, None, None

            return result, backward

        def AD_bool_to_int(b: bool):
            # FIXME: torchscript: int - bool
            if b:
                i = 1
            else:
                i = 0
            return i

        def AD_var_backward_0(grad, self, unbiased: bool):
            b = AD_bool_to_int(unbiased)

            # FIXME: torchscript: div(float, float)
            return  grad * (self - self.mean()) * 2.0 / (self.numel() - b)

        def AD_safe_size(sizes: List[int],
                         dims: List[int]):
            if len(sizes) == 0:
                return 1

            size = 1
            for i in range(len(dims)):
                d = dims[i]
                size *= sizes[d]

            return size

        def AD_var_backward_1(grad,
                              self,
                              dim: List[int],
                              unbiased: bool,
                              keepdim: bool):
            if self.dim() == 0:
                return AD_var_backward_0(grad, self, unbiased)
            self_size = self.size()
            b = AD_bool_to_int(unbiased)
            if not keepdim and self.dim() > 1:
                grad = AD_unsqueeze_multiple(grad, dim, len(self_size))

            # FIXME: torchscript: div(float, float)
            return grad * (self - self.mean(dim, True)) * 2.0 / (AD_safe_size(self_size, dim) - b)

        def std_0(self,
                  unbiased: bool=True):
            std_out = torch.std(self, unbiased)
            def backward(grad_output):
                grad_self = AD_var_backward_0(grad_output / (std_out * 2), self, unbiased)
                return grad_self, None

            return std_out, backward

        def std_1(self,
                  dim: List[int],
                  unbiased: bool,
                  keepdim: bool):
            std_out = torch.std(self, dim, unbiased, keepdim)
            def backward(grad_output):
                grad_self = AD_var_backward_1(grad_output / (std_out * 2), self, dim, unbiased, keepdim)
                return grad_self, None, None, None

            return std_out, backward

        def var_0(self,
                  unbiased: bool=True):
            def backward(grad_output):
                grad_self = AD_var_backward_0(grad_output, self, unbiased)
                return grad_self, None

            return torch.var(self, unbiased), backward

        def var_1(self,
                  dim: List[int],
                  unbiased: bool,
                  keepdim: bool):
            def backward(grad_output):
                grad_self = AD_var_backward_1(grad_output, self, dim, unbiased, keepdim)
                return grad_self, None, None, None

            return torch.var(self, dim, unbiased, keepdim), backward

        def tanh(self):
            output = torch.tanh(self)
            def backward(grad_output):
                return grad_output * (1 - output * output)

            return output, backward

        def AD_index_select_backward(grad,
                                     dim: int,
                                     indices,
                                     sizes: List[int],
                                     keepdim: bool):
            if not keepdim and len(sizes) > 0:
                grad = grad.unsqueeze(dim)
                indices = indices.unsqueeze(dim)

            # FIXME: torchscript: torch.zeros(sizes, grad.options())
            return torch.zeros(sizes).to(grad).scatter_(dim, indices, grad)

        # def topk(self,
        #          k: int,
        #          dim: int = -1,
        #          largest: bool = True,
        #          sorted: bool = True):
        #     result0, result1 = torch.topk(self, k, dim, largest, sorted)
        #     self_size = self.size()
        #     def backward(grad_output):
        #         grad_self = AD_index_select_backward(grad_output, dim, result1, self_size, True)
        #         return grad_self, None, None, None, None

        #     return result0, result1, backward

        # def kthvalue(self,
        #              k: int,
        #              dim: int,
        #              keepdim: bool):
        #     result0, result1 = torch.kthvalue(self, k, dim, keepdim)
        #     self_size = self.size()
        #     def backward(grad_output):
        #         grad_self = AD_index_select_backward(grad_output, dim, result1, self_size, keepdim)
        #         return grad_self, None, None, None

        #     return result0, result1, backward

        def AD_mm_backward_self(grad, mat2):
            return grad.mm(mat2.t())

        def AD_mm_backward_mat2(grad, self):
            return self.t().mm(grad)

        def mm(self, mat2):
            def backward(grad_output):
                grad_self = AD_mm_backward_self(grad_output, mat2)
                grad_mat2 = AD_mm_backward_mat2(grad_output, self)
                return grad_self, grad_mat2

            return torch.mm(self, mat2), backward

        def AD_permute_backward(grad,
                                fwd_dims: List[int]):
            ndims = len(fwd_dims)
            dims = [0] * ndims

            for i in range(ndims):
                dims[fwd_dims[i]] = i

            return grad.permute(dims)

        def permute(self,
                    dims: List[int]):
            def backward(grad_output):
                grad_self = AD_permute_backward(grad_output, dims)
                return grad_self, None

            return torch.permute(self, dims), backward

        def AD_select_backward(grad,
                               input_sizes: List[int],
                               dim: int,
                               index: int):
            # FIXME: torchscript: torch.zeros(sizes, grad.options())
            grad_input = torch.zeros(input_sizes).to(grad)
            grad_input.select(dim, index).copy_(grad)
            return grad_input

        # TODO: fix torch.zeros(sizes, grad.options()) before enabling select, topk, kthvalue
        # def select(self,
        #            dim: int,
        #            index: int):
        #     self_size = self.size()
        #     def backward(grad_output):
        #         grad_self = AD_select_backward(grad_output, self_size, dim, index)
        #         return grad_self, None, None

        #     return torch.select(self, dim, index), backward

        def AD_slice_backward(grad,
                              input_sizes: List[int],
                              dim: int,
                              start: int,
                              end: int,
                              step: int):
            # FIXME: torchscript: torch.zeros(sizes, grad.options())
            grad_input = torch.zeros(input_sizes).to(grad)
            grad_input.slice(dim, start, end, step).copy_(grad)
            return grad_input

        # DON'T enable slice unless we can correctly handle view ops in graph executor.
        # It triggers failure of TestJit.test_sample in test_distributions.py.
        # def slice(self,
        #           dim: int=0,
        #           start: int=0,
        #           end: int=9223372036854775807,
        #           step: int=1):
        #     def backward(grad_output):
        #         grad_self = AD_slice_backward(grad_output, self.size(), dim, start, end, step)
        #         return grad_self, None, None, None, None

        #     return torch.slice(self, dim, start, end, step), backward

        def AD_unsqueeze_to_0(self,
                              sizes: List[int]):
            ndims = len(sizes)
            for i in range(ndims):
                if sizes[i] == 1:
                    self = self.unsqueeze(i)

            return self

        def AD_unsqueeze_to_1(self,
                              dim: int,
                              sizes: List[int]):
            if len(sizes) > 0 and sizes[dim] == 1:
                return self.unsqueeze(dim)
            return self

        def squeeze_0(self):
            self_size = self.size()
            def backward(grad_output):
                grad_self = AD_unsqueeze_to_0(grad_output, self_size)
                return grad_self

            return torch.squeeze(self), backward

        def squeeze_1(self,
                      dim: int):
            self_size = self.size()
            def backward(grad_output):
                grad_self = AD_unsqueeze_to_1(grad_output, dim, self_size)
                return grad_self, None

            return torch.squeeze(self, dim), backward

        def AD_infer_size(a: List[int],
                          b: List[int]):
            dimsA = len(a)
            dimsB = len(b)

            ndim = dimsA if dimsA > dimsB else dimsB
            expand_sizes = [0] * ndim

            for i in range(ndim):
                idx = - i + ndim - 1
                sizeA = a[i] if dimsA + i >= 0 else 1
                sizeB = b[i] if dimsB + i >= 0 else 1

                # Assert sizeA == sizeB or sizeA == 1 or sizeB == 1
                expand_sizes[i] = sizeB if sizeA == 1 else sizeA

            return expand_sizes

        def AD_bmm_backward_self(grad, mat2):
            return grad.bmm(mat2.transpose(1, 2))

        def AD_bmm_backward_mat2(grad, self):
            return self.transpose(1, 2).bmm(grad)

        def bmm(self, mat2):
            def backward(grad_output):
                grad_self = AD_bmm_backward_self(grad_output, mat2)
                grad_mat2 = AD_bmm_backward_mat2(grad_output, self)
                return grad_self, grad_mat2
            return torch.bmm(self, mat2), backward

        def AD_mat_transpose(mat):
            dim = mat.dim()
            if dim == 1:
                out = mat
            elif dim == 2:
                out = mat.t()
            else:
                dims = rangelist(dim)
                dims[-1] = dim - 2
                dims[-2] = dim - 1
                out = mat.permute(dims)
            return out

        def AD_matmul_size(mat1, mat2,
                           out_size: List[int]):
            dim1 = mat1.dim()
            dim2 = mat2.dim()
            dim_out = len(out_size)
            if dim1 == 0 or dim2 == 0:
                out = mat1 * mat2
            elif dim1 + dim2 == dim_out:
                if dim2 == 1:
                    target_dim2 = 0
                else:
                    target_dim2 = -2
                out = torch.matmul(mat1.unsqueeze(dim1), mat2.unsqueeze(target_dim2))
            elif dim_out == dim1 - dim2:
                out = torch.matmul(mat1, mat2.unsqueeze(dim2)).squeeze(-1)
            elif dim_out == dim2 - dim1:
                out = torch.matmul(mat1.unsqueeze(-2), mat2).squeeze(-2)
            else:
                out = torch.matmul(mat1, mat2)
            return out

        def matmul(self, other):
            def backward(grad_output):
                self_size = self.size()
                other_size = other.size()
                grad_self = AD_matmul_size(grad_output, AD_mat_transpose(other), self_size)._grad_sum_to_size(self_size)
                grad_other = AD_matmul_size(AD_mat_transpose(self), grad_output, other_size)._grad_sum_to_size(other_size)
                return grad_self, grad_other

            return torch.matmul(self, other), backward
    )",
    R"(
        def addcmul(self,
                    tensor1,
                    tensor2,
                    *,
                    value: float = 1.0):
            def backward(grad_output):
                grad = grad_output * value
                grad_tensor1 = (grad * tensor2)._grad_sum_to_size(tensor1.size())
                grad_tensor2 = (grad * tensor1)._grad_sum_to_size(tensor2.size())
                return grad_output._grad_sum_to_size(self.size()), grad_tensor1, grad_tensor2, None
            return torch.addcmul(self, tensor1, tensor2, value=value), backward

        def _dim_arange(like,
                        dim: int):
            def backward(grad_output):
                return None, None

            return torch._dim_arange(like, dim), backward

        def contiguous(self):
            def backward(grad_output):
                return None

            return self.contiguous(), backward

        def dot(self, tensor):
            def backward(grad_output):
                grad_self = grad_output * tensor
                grad_tensor = grad_output * self
                return grad_self, grad_tensor

            return torch.dot(self, tensor), backward

        def erf(self):
            def backward(grad_output):
                # Precomputed constant C = 2.0 / math.sqrt(math.pi)
                C = 1.1283791670955126
                grad_self =  C * torch.exp(- self * self) * grad_output
                return grad_self

            return torch.erf(self), backward

        def expand(self,
                   size: List[int],
                   *,
                   implicit: bool=False):
            self_size = self.size()
            def backward(grad_output):
                grad_self = torch._grad_sum_to_size(grad_output, self_size)
                return grad_self, None, None

            return torch.expand(self, size, implicit=implicit), backward

        def expand_as(self, other):
            self_size = self.size()
            def backward(grad_output):
                grad_self = grad_output._grad_sum_to_size(self_size)
                return grad_self, None

            return torch.expand_as(self, other), backward

        def full_like(self,
                      fill_value: float):
            def backward(grad_output):
                return None, None

            return torch.full_like(self, fill_value), backward

        def lerp_0(self,
                   end,
                   weight: float):
            def backward(grad_output):
                grad_self = (grad_output * (1 - weight))._grad_sum_to_size(self.size())
                grad_end = (grad_output * weight)._grad_sum_to_size(end.size())
                return grad_self, grad_end, None
            return torch.lerp(self, end, weight), backward

        def lerp_1(self,
                   end,
                   weight):
            def backward(grad_output):
                grad_self = (grad_output * (1 - weight))._grad_sum_to_size(self.size())
                grad_end = (grad_output * weight)._grad_sum_to_size(end.size())
                return grad_self, grad_end, None
            return torch.lerp(self, end, weight), backward

        def mul(self, other):
            def backward(grad_output):
                # self & other are used in backward. No need to pass in their size
                # from forward pass
                grad_self = (grad_output * other)._grad_sum_to_size(self.size())
                grad_other = (grad_output * self)._grad_sum_to_size(other.size())
                return grad_self, grad_other

            return self * other, backward

        def mv(self, vec):
            def backward(grad_output):
                grad_self = grad_output.ger(vec)
                grad_vec = self.t().mv(grad_output)
                return grad_self, grad_vec

            return torch.mv(self, vec), backward

        def nonzero(self):
            def backward(grad_output):
                return None

            return torch.nonzero(self), backward

        def ones_like(self):
            def backward(grad_output):
                return None

            return torch.ones_like(self), backward

        def pow_0(self,
                  exponent: float):
            def backward(grad_output):
                grad_self = torch.where(torch.tensor(exponent == 0.0), torch.zeros_like(self), grad_output * exponent * torch.pow(self, exponent - 1))
                return grad_self, None

            return torch.pow(self, exponent), backward

        def pow_1(self, exponent):
            def backward(grad_output):
                # self & exponent are used in backward, no need to pass in its size explicitly
                grad_self = torch.where(exponent == 0.0, torch.zeros_like(self), grad_output * exponent * torch.pow(self, exponent - 1))._grad_sum_to_size(self.size())
                grad_exponent = (grad_output * torch.pow(self, exponent) * torch.log(self))._grad_sum_to_size(exponent.size())
                return grad_self, grad_exponent

            return torch.pow(self, exponent), backward

        def pow_2(self: float,
                  exponent):
            def backward(grad_output):
                grad_exponent = grad_output * torch.pow(self, exponent) * torch.log(torch.tensor(self))
                return None, grad_exponent

            return torch.pow(self, exponent), backward

        def rsub_0(self, other,
                   alpha: float = 1.0):
            self_size = self.size()
            other_size = other.size()
            def backward(grad_output):
                grad_self = (- grad_output * alpha)._grad_sum_to_size(self_size)
                grad_other = (grad_output)._grad_sum_to_size(other_size)
                return grad_self, grad_other, None

            return torch.rsub(self, other, alpha), backward

        def rsub_1(self,
                   other: float,
                   alpha: float = 1.0):
            self_size = self.size()
            def backward(grad_output):
                grad_self = (- grad_output * alpha)._grad_sum_to_size(self_size)
                return grad_self, None, None

            return torch.rsub(self, other, alpha), backward

        def sqrt(self):
            result = torch.sqrt(self)
            def backward(grad_output):
                grad_self = grad_output / (2 * result)
                return grad_self

            return result, backward

        def t(self):
            def backward(grad_output):
                grad_self = torch.t(grad_output)
                return grad_self

            return torch.t(self), backward

        def to_0(self,
                 device: Optional[Device],
                 dtype: Optional[int],
                 non_blocking: bool=False,
                 copy: bool=False):
            self_device = self.device
            self_dtype = self.dtype
            if device is not None:
                result = self.to(device, dtype=dtype, non_blocking=non_blocking, copy=copy)
            else:
                result = self.to(dtype, non_blocking=non_blocking, copy=copy)
            def backward(grad_output):
                grad_self = grad_output.to(self_device, dtype=self_dtype, non_blocking=non_blocking, copy=copy)
                return grad_self, None, None, None, None

            return result, backward


        def to_1(self,
                 dtype: int,
                 non_blocking: bool=False,
                 copy: bool=False):
            self_dtype = self.dtype
            def backward(grad_output):
                grad_self = grad_output.to(self_dtype, non_blocking, copy)
                return grad_self, None, None, None

            return self.to(dtype=dtype, non_blocking=non_blocking, copy=copy), backward

        def to_2(self,
                 other,
                 non_blocking: bool=False,
                 copy: bool=False):
            def backward(grad_output):
                grad_self = grad_output.to(self, non_blocking, copy)
                return grad_self, None, None, None

            return self.to(other, non_blocking=non_blocking, copy=copy), backward

        def transpose(self,
                      dim0: int,
                      dim1: int):
            def backward(grad_output):
                grad_self = torch.transpose(grad_output, dim0, dim1)
                return grad_self, None, None

            return torch.transpose(self, dim0, dim1), backward

        def view(self,
                 size: List[int]):
            self_size = self.size()
            def backward(grad_output):
                grad_self = grad_output.reshape(self_size)
                return grad_self, None

            return torch.view(self, size), backward
    )",
    R"(
        def AD_adaptive_avg_pool2d_backward(grad,
                                            self,
                                            output_size: List[int]):
            if output_size[0] == 1 and output_size[1] == 1:
                self_size = self.size()
                grad_self = grad.expand(self.size()) / (self_size[-1] * self_size[-2])
            else:
                grad_self = torch._adaptive_avg_pool2d_backward(grad, self)

            return grad_self

        def AD_adaptive_avg_pool1d_backward(grad,
                                            input,
                                            output_size: List[int]):
            output_size_2d = [1, output_size[0]]
            grad_input = AD_adaptive_avg_pool2d_backward(grad.unsqueeze(2), input.unsqueeze(2), output_size_2d).squeeze(2)
            return grad_input

        def adaptive_avg_pool1d(self,
                                output_size: List[int]):
            def backward(grad_output):
                grad_self = AD_adaptive_avg_pool1d_backward(grad_output, self, output_size)
                return grad_self, None

            return torch.adaptive_avg_pool1d(self, output_size), backward

        def adaptive_avg_pool2d(self,
                                output_size: List[int]):
            def backward(grad_output):
                # self is used in backward, no need to pass in its size explicitly
                grad_self = AD_adaptive_avg_pool2d_backward(grad_output, self, output_size)
                return grad_self, None
            return torch.adaptive_avg_pool2d(self, output_size), backward

        def adaptive_avg_pool3d(self,
                                output_size: List[int]):
            def backward(grad_output):
                grad_self = torch.adaptive_avg_pool3d_backward(grad_output, self)
                return grad_self, None

            return torch.adaptive_avg_pool3d(self, output_size), backward

        def batch_norm(input : Tensor,
                       weight : Optional[Tensor],
                       bias : Optional[Tensor],
                       running_mean : Optional[Tensor],
                       running_var : Optional[Tensor],
                       training : bool,
                       momentum : float,
                       eps : float,
                       cudnn_enabled : bool):

            output, save1, save2, impl_idx = torch._batch_norm_impl_index(
                input, weight, bias, running_mean, running_var, training,
                momentum, eps, cudnn_enabled)
            has_weight = weight is not None
            has_bias = bias is not None

            def backward(grad_output):
                dinput, dweight, dbias = torch._batch_norm_impl_index_backward(
                    impl_idx, input, grad_output, weight, running_mean, running_var,
                    save1, save2, training, eps, [True, has_weight, has_bias])
                return dinput, dweight, dbias, None, None, None, None, None, None

            return output, backward

        def layer_norm(input : Tensor,
                       normalied_shape : List[int],
                       weight : Optional[Tensor],
                       bias : Optional[Tensor],
                       eps : float,
                       cudnn_enable : bool):

            bn_out, save1, save2, impl_idx = torch._batch_norm_impl_index(
                input, weight, bias, None, None, True,
                0.0, eps, cudnn_enable)
            has_weight = weight is not None
            has_bias = bias is not None

            bn_out = bn_out.view(input.sizes())
            if weight is not None and bias is not None:
                output = bias.addcmul(bn_out, weight)
            elif weight is not None:
                output = bn_out.mul(weight)
            elif bias is not None:
                output = bn_out.add(bias)
            else:
                output = bn_out

            def backward(grad_output):
                if weight is not None:
                    grad_output = grad_output * torch.t(weight)
                    weight = grad_output * torch.t(bn_out)

                grad_output = grad_output.reshape(input.sizes())

                dinput, dweight, dbias = torch._batch_norm_impl_index_backward(
                    impl_idx, input, grad_output, weight, None, None,
                    save1, save2, True, eps, [True, has_weight, has_bias])
                return dinput, None, dweight, dbias, None, None

            return output, backward

        def AD_fused_dropout_backward(grad,
                                      mask,
                                      p1m: float):
            p1r = 1. / p1m
            grad_input = grad * (mask.type_as(grad) * p1r)
            return grad_input

        def dropout(input,
                    p: float,
                    train: bool):
            use_cuda = input.is_cuda
            # lowering is specialized for cuda because cuda fuser can efficiently fuse those operations
            # for cpu backend, where fusions are disabled, a different lowering that is more efficient
            # in the absence of fusion is used
            p1m = 1. - p
            if use_cuda:
                mask = torch.rand_like(input) < p1m
                res = mask.type_as(input) * input * (1./p1m)
            else:
                mask = torch.empty_like(input)
                mask.bernoulli_(p1m)
                res = mask * input / p1m

            def backward(grad_output):
                if use_cuda:
                    grad_input = AD_fused_dropout_backward(grad_output, mask, p1m)
                else:
                    grad_input = grad_output * mask / p1m
                return grad_input, None, None
            return res, backward

        def embedding(weight,
                      indices,
                      padding_idx: int,
                      scale_grad_by_freq: bool,
                      sparse: bool):
            weight_size_0 = weight.size()[0]
            def backward(grad_output):
                grad_weight = torch.embedding_backward(grad_output, indices, weight_size_0, padding_idx, scale_grad_by_freq, sparse)
                return grad_weight, None, None, None, None

            return torch.embedding(weight, indices, padding_idx, scale_grad_by_freq, sparse), backward

        def nll_loss(self, target, weight: Optional[Tensor], reduction: int, ignore_index: int):
            result, total_weight = torch.nll_loss_forward(self, target, weight, reduction, ignore_index)
            def backward(grad):
                return torch.nll_loss_backward(grad, self, target, weight, reduction, ignore_index, total_weight), None, None, None, None
            return result, backward

        def softmax_0(self, dim: int):
            result = torch.softmax(self, dim)
            def backward(grad_output):
                grad_self = torch._softmax_backward_data(grad_output, result, dim, self)
                return grad_self, None

            return result, backward

        def softmax_1(self, dim: int, dtype: int):
            result = torch.softmax(self, dim, dtype)
            def backward(grad_output):
                grad_self = torch._softmax_backward_data(grad_output, result, dim, self)
                return grad_self, None, None

            return torch.softmax(self, dim, dtype), backward

        def AD_interpolate_backward(grad,
                                    input,
                                    mode: str,
                                    align_corners: bool):
            output_size = grad.size()[2:]
            input_size = input.size()
            input_dim = len(input_size)
            if input_dim == 3 and mode == 'nearest':
                grad_input = torch.upsample_nearest1d_backward(grad, output_size, input_size)
            elif input_dim == 4 and mode == 'nearest':
                grad_input = torch.upsample_nearest2d_backward(grad, output_size, input_size)
            elif input_dim == 5 and mode == 'nearest':
                grad_input = torch.upsample_nearest3d_backward(grad, output_size, input_size)
            elif input_dim == 3 and mode == 'linear':
                grad_input = torch.upsample_linear1d_backward(grad, output_size, input_size, align_corners)
            elif input_dim == 4 and mode == 'bilinear':
                grad_input = torch.upsample_bilinear2d_backward(grad, output_size, input_size, align_corners)
            elif input_dim == 5 and mode == 'trilinear':
                grad_input = torch.upsample_trilinear3d_backward(grad, output_size, input_size, align_corners)
            elif input_dim == 4 and mode == 'bicubic':
                grad_input = torch.upsample_bicubic2d_backward(grad, output_size, input_size, align_corners)
            elif input_dim == 3 and mode == 'area':
                grad_input = AD_adaptive_avg_pool1d_backward(grad, input, output_size)
            elif input_dim == 4 and mode == 'area':
                grad_input = AD_adaptive_avg_pool2d_backward(grad, input, output_size)
            elif input_dim == 5 and mode == 'area':
                grad_input = torch.adaptive_avg_pool3d_backward(grad, input)
            else:
                # NEVER REACH HERE
                grad_input = torch.zeros_like(input)
                raise RuntimeError('Input Error: Only 3D, 4D and 5D input Tensors supported')

            return grad_input

        def __interpolate_0(input,
                            size: Optional[int],
                            scale_factor: Optional[List[float]],
                            mode: str='nearest',
                            align_corners: Optional[bool]):
            def backward(grad_output):
                if align_corners is None:
                    align_corners = False
                grad_self = AD_interpolate_backward(grad_output, input, mode, align_corners)
                return grad_self, None, None, None, None

            return torch.__interpolate(input, size, scale_factor, mode, align_corners), backward

        def __interpolate_1(input,
                            size: Optional[List[int]],
                            scale_factor: Optional[List[float]],
                            mode: str='nearest',
                            align_corners: Optional[bool]):
            def backward(grad_output):
                if align_corners is None:
                    align_corners = False
                grad_self = AD_interpolate_backward(grad_output, input, mode, align_corners)
                return grad_self, None, None, None, None

            return torch.__interpolate(input, size, scale_factor, mode, align_corners), backward

        def __interpolate_2(input,
                            size: Optional[int],
                            scale_factor: Optional[float],
                            mode: str='nearest',
                            align_corners: Optional[bool]):
            def backward(grad_output):
                if align_corners is None:
                    align_corners = False
                grad_self = AD_interpolate_backward(grad_output, input, mode, align_corners)
                return grad_self, None, None, None, None

            return torch.__interpolate(input, size, scale_factor, mode, align_corners), backward

        def __interpolate_3(input,
                            size: Optional[List[int]],
                            scale_factor: Optional[float],
                            mode: str='nearest',
                            align_corners: Optional[bool]):
            def backward(grad_output):
                if align_corners is None:
                    align_corners = False
                grad_self = AD_interpolate_backward(grad_output, input, mode, align_corners)
                return grad_self, None, None, None, None

            return torch.__interpolate(input, size, scale_factor, mode, align_corners), backward

      )"};
std::unordered_map<std::string, GradientPair> schema_to_graphs;

// This map is a workaround to cache compiled gradient_pairs. Ideally this graph
// should be compiled only once and saved in Operator structure.
// This should be done along with merging into native_functions.yaml.
std::unordered_map<const FunctionSchema*, GradientPair> cached_gradient_pairs;
} // anonymous namespace

std::pair<std::shared_ptr<Graph>, Value*> extractClosure(Value* closure) {
  AT_CHECK(
      closure->node()->kind() == prim::TupleConstruct,
      "closure must be a literal tuple construct");
  Value* fn = closure->node()->inputs().at(0);
  Value* context = closure->node()->inputs().at(1);

  AT_CHECK(
      fn->node()->kind() == prim::Function,
      "closure tuple must contain a prim::Function");
  return std::make_pair(fn->node()->g(attr::Subgraph), context);
}

Argument originalReturnType(const TupleTypePtr& tup) {
  AT_CHECK(tup->elements().size() > 1);
  if (tup->elements().size() == 2)
    return Argument("", tup->elements().at(0));
  std::vector<TypePtr> types = tup->elements().vec();
  types.pop_back();
  return Argument("", TupleType::create(std::move(types)));
}

// In torchscript AD formulas, we define {func_0, func_1, ...} as
// overloaded functions of `func`.
// Remove the suffix before adding the schema string to map
// schema_to_graphs.
std::string overloadedSchemaString(const FunctionSchema& schema) {
  const auto& schema_name = schema.name();
  auto pos = schema_name.find_last_of('_');
  auto schema_name_suffix = schema_name.substr(pos + 1);
  std::string schema_string = canonicalSchemaString(schema);
  if (!schema_name_suffix.empty() &&
      schema_name_suffix.find_first_not_of("0123456789") == std::string::npos) {
    schema_string.replace(
        schema_string.find(schema_name),
        schema_name.length(),
        schema_name.substr(0, pos));
  }

  return schema_string;
}

bool isHelperFunction(const std::string& method_name) {
  std::string helper_prefix = "AD_";
  return method_name.compare(0, helper_prefix.length(), helper_prefix) == 0;
}

void loadModule(const std::shared_ptr<script::Module>& module) {
  for (const auto& method_ : module->get_methods()) {
    if (isHelperFunction(method_.key()))
      continue;

    const auto& method = method_.value();
    GradientPair pair;
    pair.forward = method->graph();

    // lookup the backward function
    Node* forward_tuple = pair.forward->outputs().at(0)->node();

    if (forward_tuple->kind() != prim::TupleConstruct) {
      throw script::ErrorReport(forward_tuple->getSourceLocation())
          << "gradient must return literal a tuple";
    }

    Value* context;
    std::tie(pair.backward, context) =
        extractClosure(forward_tuple->inputs().back());

    // do surgery on the forward function to remove the closure tuple and
    // replace it with the context variable:
    //  backward = (<lambda>, context_tuple)
    //  return original, backward
    //  -----
    //  return original, context_tuple
    std::vector<Value*> new_inputs = forward_tuple->inputs().vec();
    new_inputs.back() = context;
    Value* new_tuple =
        pair.forward->appendNode(pair.forward->createTuple(new_inputs))
            ->output();
    pair.forward->eraseOutput(0);
    pair.forward->registerOutput(new_tuple);
    forward_tuple->destroy();

    // derive schema from original function's schema:
    const FunctionSchema& loaded_schema = method->getSchema();
    FunctionSchema actual_schema(
        Symbol::aten(loaded_schema.name()),
        loaded_schema.overload_name(),
        loaded_schema.arguments(),
        {originalReturnType(new_tuple->type()->expect<TupleType>())});

    // modify canonical string for function overloading
    // prefer not to modify the schema name
    auto schema_string = overloadedSchemaString(actual_schema);

    schema_to_graphs[schema_string] = std::move(pair);
  }
}

void loadFunctions() {
  for (const std::string& str : functions) {
    auto cu = std::make_shared<script::Module>();
    script::defineMethodsInModule(
        cu, str, script::nativeResolver, c10::nullopt);
    loadModule(cu);
  }
}

c10::optional<GradientPair> gradientInfoForSchema(
    const FunctionSchema& schema) {
  std::lock_guard<std::mutex> guard(lock);
  if (schema_to_graphs.size() == 0) {
    loadFunctions();
  }
  auto cache_it = cached_gradient_pairs.find(&schema);
  if (cache_it != cached_gradient_pairs.end()) {
    return cache_it->second;
  } else {
    auto schema_str = canonicalSchemaString(schema);
    // Specialize Scalar to float for the arg type of the node schema
    // this is used to:
    // 1. define scalar type as float in TorchScript autodiff formula
    // 2. to make sure the input of any graph node does not contain scalar type
    //    in its argument, all scalar arg should already be passed with float
    //    value since scalar/int aren't differentiable either way.
    //
    c10::ReplaceAll(schema_str, "Scalar", "float");

    auto sym_script_it = schema_to_graphs.find(schema_str);

    if (sym_script_it != schema_to_graphs.end()) {
      cached_gradient_pairs.emplace_hint(
          cache_it, &schema, sym_script_it->second);
      return sym_script_it->second;
    }
  }
  return c10::nullopt;
}

bool hasGradientInfoForSchema(const FunctionSchema& schema) {
  return gradientInfoForSchema(schema).has_value();
}

} // namespace jit
} // namespace torch
