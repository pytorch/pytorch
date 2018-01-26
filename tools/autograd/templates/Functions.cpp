#include "Functions.h"
#include <ATen/WrapDimUtils.h>
#include <iostream>

// define constants like M_PI and C keywords for MSVC
#ifdef _MSC_VER
#define _USE_MATH_DEFINES
#include <ciso646>
#endif
#include <math.h>
#include <algorithm>

// ${generated_comment}

using at::Tensor;
using at::Scalar;
using at::IntList;
using at::TensorList;

namespace torch { namespace autograd { namespace generated {

namespace {

Tensor not_implemented(const char* name) {
  throw std::runtime_error(
      std::string("the derivative for '") + name + "' is not implemented");
}

Tensor maybe_multiply(const Tensor & t, const Scalar & s) {
  bool is_one = false;
  if (s.isFloatingPoint()) {
    is_one = s.toDouble() == 1;
  } else if(s.isIntegral()) {
    is_one = s.toLong() == 1;
  }

  if (is_one) {
    return t;
  } else {
    return t * s;
  }
}

Tensor norm_backward(const Tensor & grad, const Tensor & self, const Scalar & p_, const Tensor & norm) {
  double p = p_.toDouble();
  Tensor self_scaled;
  Tensor scale_v;
  if (p == 0.0) {
    return zeros_like(self);
  } else if (p == 1.0) {
    return self.sign() * grad;
  } else if (p < 2.0) {
    self_scaled = self.sign() * self.abs().pow(p - 1);
    scale_v = grad / norm.pow(p - 1);
  } else if (p == 2.0) {
    self_scaled = self;
    scale_v = grad / norm;
  } else {
    self_scaled = self * self.abs().pow(p - 2);
    scale_v = grad / norm.pow(p - 1);
  }
  // handle case at 0 where we return a subgradient containing 0
  scale_v.masked_fill_(norm == 0, 0);
  return self_scaled * scale_v;
}

Tensor norm_backward(Tensor grad, const Tensor & self, const Scalar & p_, Tensor norm, int64_t dim, bool keepdim) {
#ifdef WITH_SCALARS
  if (!keepdim) {
#else
  if (!keepdim && self.dim() > 1) {
#endif
    grad = grad.unsqueeze(dim);
    norm = norm.unsqueeze(dim);
  }
  return norm_backward(grad, self, p_, norm);
}

Tensor reduce_to(const Tensor & grad, IntList sizes) {
  if (sizes.size() == 0) {
    return grad.sum();
  }
  Tensor result = grad;
  while (result.dim() > (int64_t)sizes.size()) {
    result = result.sum(0, false);
  }
  for (int64_t i = 0; i < result.dim(); ++i) {
    if (sizes[i] == 1 && result.sizes()[i] > 1) {
      result = result.sum(i, true);
    }
  }
  return result;
}

Tensor permute_backwards(const Tensor & grad, IntList fwd_dims) {
  // invert the permutation
  std::vector<int64_t> dims(fwd_dims.size());
  for (size_t i = 0; i < fwd_dims.size(); i++) {
    dims[fwd_dims[i]] = i;
  }
  return grad.permute(dims);
}

Tensor sum_backward(const Tensor & grad, IntList sizes, int64_t dim, bool keepdim) {
#ifdef WITH_SCALARS
  if (!keepdim) {
#else
   if (!keepdim && sizes.size() > 1) {
#endif
    return grad.unsqueeze(dim).expand(sizes);
  } else {
    return grad.expand(sizes);
  }
}

Tensor reverse_dim(const Tensor& t, int64_t dim) {
  Tensor index = t.type().toScalarType(at::ScalarType::Long).arange(t.size(dim) - 1, -1, -1);
  return t.index_select(dim, index);
}

Tensor prod_safe_zeros_backward(const Tensor &grad, const Tensor& inp, int64_t dim) {
  if (inp.size(dim) == 1) {
    return grad;
  }

  std::vector<int64_t> ones_size(inp.sizes());
  ones_size[dim] = 1;
  Tensor ones = grad.type().ones(ones_size);
  Tensor exclusive_normal_nocp = at::cat({ones, inp.narrow(dim, 0, inp.size(dim) - 1)}, dim);
  Tensor exclusive_normal = exclusive_normal_nocp.cumprod(dim);

  Tensor narrow_reverse = reverse_dim(inp.narrow(dim, 1, inp.size(dim) - 1), dim);
  Tensor exclusive_reverse_nocp = at::cat({ones, narrow_reverse}, dim);
  Tensor exclusive_reverse = reverse_dim(exclusive_reverse_nocp.cumprod(dim), dim);

  return grad * (exclusive_normal * exclusive_reverse);
}

// note that the gradient for prod is equivalent to:
// cumprod(exclusive, normal) * cumprod(exclusive, reverse), e.g.:
// input:                        [    a,     b,     c]
// cumprod(exclusive, normal):   [1    ,     a, a * b]
// cumprod(exclusive, reverse):  [b * c,     c,     1]
// product:                      [b * c, a * c, a * b]
// and this is safe under input with 0s.
Tensor prod_backward(const Tensor& grad, const Tensor& input, const Tensor& result) {
  Tensor zero_idx = (input == 0).nonzero();
  if (zero_idx.numel() == 0) {
    return (grad * result) / input;
  } else if (zero_idx.size(0) > 1) {
    return zeros_like(input);
  } else {
    return prod_safe_zeros_backward(grad, input.contiguous().view(-1), 0).view_as(input);
  }
}

Tensor prod_backward(Tensor grad, const Tensor& input, Tensor result, int64_t dim, bool keepdim) {
  dim = at::maybe_wrap_dim(dim, input.sizes().size());
  if (!keepdim && input.dim() != 1) {
    grad = grad.unsqueeze(dim);
    result = result.unsqueeze(dim);
  }

  Tensor zero_mask = (input == 0);
  Tensor slice_zero_count = zero_mask.sum(dim, true);
  int64_t total_zeros = slice_zero_count.sum().toCLong();
  if (total_zeros == 0) {
    return (grad * result) / input;
  } else {
    return prod_safe_zeros_backward(grad, input, dim);
  }
}

Tensor sum_scan_exclusive(const Tensor& x, int64_t dim) {
  Tensor ret = at::cumsum(-x, dim);

  int64_t end_idx = ret.size(dim) - 1;
  Tensor ret_sum = ret.narrow(dim, end_idx, 1).clone();
  ret -= ret_sum.expand_as(ret);
  ret += x;
  return ret;
}

Tensor cumprod_backward(const Tensor &grad, const Tensor &input, int64_t dim) {
  /*
    There are two algorithms to do this. The first one
    is very efficient, but works only when there are no
    nonzero elements in the input.

    The second one is much more complex, but it doesn't
    assume anything on the input. The main downside is
    that it takes time O(n^2), where n = input.size(self.dim)
    (i.e. the length of the cumulative product). This is in
    contrast to the forward pass and the efficient algorithm,
    which are both O(n).

    The second algorithm is a simple application of the chain
    rule. If x is an n-dimensional vector, and y = cumprod(x),
    and F is the final cost, then

    dF / dx_k = sum_j (dF / dy_j) * (dy_j / dx_k)   (1)

    The term dF / dy_j is just grad_output[j] (assuming again
    everything is one-dimensional).

    The term (dy_j / dx_k) is easilly seen to be

    if j >= k
      dy_j / dx_k = prod_{1 <= i <= j, i != k} x_i
    else:
      dy_j / dx_k = 0

    Note that the indicator (j>=k) can be taken out
    by replacing the sum in (1) with a sum from
    j = k to n.

    Thus,
    df / dx_k = sum_{k <= j <= n} grad_output[j] * (dy_j / dx_k)

    with
    dy_j / dx_k = prod_{1 <= i <= j, i != k} x_i     (2)

    Note that this last term is just the cumulative product
    with k omitted. Thus, if x_k (the input) is nonzero, we can
    just express this as

    dy_j / dx_k = (prod_{1 <= i <= j} x_i) / x_k
                = y_j / x_k

    So therefore,

    df / dx_k = sum_{k <= j <= n} grad_output[j] * y_j / x_k

    so

    grad_output = sum_scan_exclusiv(grad_output * output) / input

    If the input is nonzero, we need to calculate the dy_j / dx_k
    by using the formula (2), called in the code omitted_products.

    The way the code calculates it is simply by noting that

    prod_{1 <= i <= j, i != k} x_i
        = (prod_{1 <= i <= k} x_i) * (prod_{k + 1 <= i <= j} x_i)

    the first term is calculated as prods_until_k, which since
    doesn't depend in j is easy to vectorize.

    The second term (indexed by j) is the cumulative product of
    x_{k+1}, x_{k+2}, ..., x_n, and it's named in the code
    prods_from_k_pkus_1, and it's calculated as a cumprod.

    In order to vectorize this properly, we need to add to
    omitted_products the dimensions where k > j, and therefore
    dy_j / dx_k = 0, which is done right after the assert.
  */

  dim = at::maybe_wrap_dim(dim, input.sizes().size());
  int64_t dim_size = input.size(dim);
  if (dim_size == 1) {
    return grad;
  }

  // Simple case with nonzero elements in the input
  if ((input != 0).all().toCByte()) {
    Tensor result = at::cumprod(input, dim);
    return sum_scan_exclusive(result * grad, dim) / input;
  }

  std::vector<int64_t> ones_size(input.sizes());
  ones_size[dim] = 1;
  Tensor ones = grad.type().ones({1}).expand(ones_size);
  Tensor grad_input = grad.type().zeros(input.sizes());
  Tensor prods_from_k_plus_1;
  Tensor omitted_products;
  for (int k = 0; k < dim_size; ++k) {
    if (k == 0) {
      prods_from_k_plus_1 = at::cumprod(input.slice(dim, k + 1), dim);
      omitted_products = at::cat({ones, prods_from_k_plus_1}, dim);
    } else if (k == dim_size - 1) {
      Tensor prods_until_k = at::prod(input.slice(dim, 0, k), dim, true);
      omitted_products = prods_until_k;
    } else {
      Tensor prods_until_k = at::prod(input.slice(dim, 0, k), dim, true);
      prods_from_k_plus_1 = at::cumprod(input.slice(dim, k+1), dim);
      omitted_products = prods_until_k.expand_as(prods_from_k_plus_1) * prods_from_k_plus_1;
      omitted_products = at::cat({prods_until_k, omitted_products}, dim);
    }

    // At this point omitted_products is the same size
    // as input, except on the dimension dim where it's
    // dim_size - k
    TORCH_ASSERT(omitted_products.size(dim) == dim_size - k);

    grad_input.select(dim, k).copy_(
        at::sum(grad.slice(dim, k) * omitted_products,dim));
  }

  return grad_input;
}

Tensor cumsum_backward(const Tensor & x, int64_t dim) {
  auto ret = at::cumsum(-x, dim);
  auto ret_sum = ret.narrow(dim, ret.size(dim) - 1, 1).clone();
  ret -= ret_sum.expand(ret.sizes());
  ret += x;
  return ret;
}

Tensor unsqueeze_to(const Tensor & self, IntList sizes) {
  auto result = self;

#ifndef WITH_SCALARS
  // Let's say the input had size (1, 1). input.squeeze(), with scalars
  // disabled, produces a result of size (1,). This needs some
  // special handling because for all other cases we unsqueeze every
  // dimension that has size 1; doing that here would lead to one extra
  // unsqueezed dimension
  if (self.sizes().equals({1})) {
    return result.view(sizes);
  }
#endif

  int64_t nDims = sizes.size();
  for (int64_t dim = 0; dim < nDims; dim++) {
    if (sizes[dim] == 1) {
      result = result.unsqueeze(dim);
    }
  }
  return result;
}

Tensor unsqueeze_to(const Tensor & self, int64_t dim, IntList sizes) {
  dim = at::maybe_wrap_dim(dim, sizes.size());
#ifdef WITH_SCALARS
  if (sizes[dim] == 1) {
#else
  if (sizes[dim] == 1 && sizes.size() != 1) {
#endif
    return self.unsqueeze(dim);
  }
  return self;
}

variable_list cat_tensors_backward(const Tensor & grad, const std::vector<int64_t> &sizes, int64_t dim) {
  variable_list grad_inputs(sizes.size());
  int64_t accumulate = 0;
  for (size_t i = 0; i < sizes.size(); ++i) {
    auto size = sizes[i];
    accumulate += size;
    if (size == 0) {
      grad_inputs[i] = grad.type().zeros({0});
    } else {
      grad_inputs[i] = grad.narrow(dim, accumulate - size, size);
    }
  }
  return grad_inputs;
}

Tensor mm_mat1_backward(const Tensor & grad, const Tensor & mat2, IntList sizes, IntList strides, const Scalar & alpha) {
  // if input was column-major, return grad as column-order for efficiency
  if (strides[0] == 1 && strides[1] == sizes[0]) {
    return maybe_multiply(mat2.mm(grad.t()).t(), alpha);
  } else {
    return maybe_multiply(grad.mm(mat2.t()), alpha);
  }
}

Tensor mm_mat2_backward(const Tensor & grad, const Tensor & mat1, IntList sizes, IntList strides, const Scalar & alpha) {
  // if input was column-major, return grad as column-order for efficiency
  if (strides[0] == 1 && strides[1] == sizes[0]) {
    return maybe_multiply(grad.t().mm(mat1).t(), alpha);
  } else {
    return maybe_multiply(mat1.t().mm(grad), alpha);
  }
}

Tensor renorm_backward(const Tensor & grad, const Tensor & self, Scalar p, int64_t dim, Scalar maxnorm) {
  auto transposed_sizes = std::vector<int64_t>(self.transpose(dim, 0).sizes());
  auto flatten = [&](const Tensor & t) {
    return t.transpose(dim, 0).contiguous().view({t.size(dim), -1});
  };
  auto unflatten = [&](const Tensor & t) {
    return t.contiguous().view(transposed_sizes).transpose(dim, 0);
  };

  // renorm computes the norm over all dimensions except `dim`, which is why
  // we need the flatten and unflatten business. TODO: simplify this when we
  // add support for norm over multiple dimensions.
  auto self_flat = flatten(self);
  auto grad_flat = flatten(grad);
  auto norm_flat = self_flat.norm(p, 1, true);
  auto grad_output = (self_flat * grad_flat).sum(1, true);
  auto nb = norm_backward(grad_output, self_flat, p, norm_flat, 1, true);
  auto invnorm = (norm_flat + 1e-7).reciprocal();
  auto grad_norm = unflatten(maxnorm * invnorm * (grad_flat - invnorm * nb));
  auto norm = unflatten(norm_flat.expand_as(self_flat));

  // TODO: remove the detach once comparison ops no longer require grad
  auto mask = Variable(norm < maxnorm).detach();
  return at::where(mask, grad, grad_norm);
}

Tensor select_backward_scalar(Tensor grad, const Tensor & input, const Tensor & value) {
  auto grad_input = zeros_like(input);
#ifdef WITH_SCALARS
  grad_input.masked_fill_(input == value, grad);
#else
  auto grad_data = static_cast<Variable&>(grad).data();
  grad_input.masked_fill_(input == value, Scalar(grad_data[0]));
#endif
  return grad_input;
}

Tensor select_backward(Tensor grad, int64_t dim, Tensor indices, IntList sizes, bool keepdim) {
#ifdef WITH_SCALARS
  if (!keepdim) {
#else
  if (!keepdim && sizes.size() > 1) {
#endif
    grad = grad.unsqueeze(dim);
    indices = indices.unsqueeze(dim);
  }
  return grad.type().zeros(sizes).scatter_(dim, indices, grad);
}

Tensor trace_backward(const Tensor & grad, IntList sizes) {
  if (sizes.size() != 2) {
    throw std::runtime_error("expected matrix input");
  }

  auto& long_type = grad.type().toScalarType(at::kLong);

  auto grad_input = grad.type().zeros(sizes[0] * sizes[1]);
  auto indices = long_type.arange(0, grad_input.numel(), sizes[1] + 1);
#ifdef WITH_SCALARS
  grad_input.index_fill_(0, indices, grad);
#else
  auto grad_data = static_cast<const Variable&>(grad).data();
  grad_input.index_fill_(0, indices, Scalar(grad_data[0]));
#endif
  return grad_input.view(sizes);
}

Tensor unfold_backward(const Tensor & grad, IntList input_sizes, int64_t dim, int64_t size, int64_t step) {
  auto& long_type = grad.type().toScalarType(at::kLong);

  int64_t numel = 1;
  for (auto size : input_sizes) {
    numel *= size;
  }

  auto idx = long_type.arange(0, numel).view(input_sizes);
  auto idx_unfolded = idx.unfold(dim, size, step).contiguous().view(-1);
  auto grad_input = grad.type().zeros({numel});
  grad_input.index_add_(0, idx_unfolded, grad.contiguous().view(-1));
  return grad_input.view(input_sizes);
}

Tensor var_backward(const Tensor & grad, const Tensor & self, bool unbiased) {
  return (2.0 / (self.numel() - unbiased)) * grad * (self - self.mean());
}

Tensor var_backward(Tensor grad, const Tensor & self, int64_t dim, bool unbiased, bool keepdim) {
  if (!keepdim && self.dim() > 1) {
    grad = grad.unsqueeze(dim);
  }
  return (2.0 / (self.size(dim) - unbiased)) * grad * (self - self.mean(dim, true));
}

Tensor masked_scatter_backward(const Tensor & grad, const Tensor & mask, IntList sizes) {
  int64_t numel = 1;
  for (auto size : sizes) {
    numel *= size;
  }
  auto mask_selected = grad.masked_select(mask);
  auto diff_nelem = numel - mask_selected.numel();
  if (diff_nelem > 0) {
    // because mask_selected returns a 1-d tensor with size of masked elements that are 1,
    // we need to fill out the rest with zeros then reshape back to tensor2's size.
    auto zeros_fillin = grad.type().zeros({diff_nelem});
    mask_selected = at::cat({mask_selected, zeros_fillin}, 0);
  }
  return mask_selected.view(sizes);
}

Tensor potrf_backward(Tensor grad, bool upper, Tensor L) {
  // cf. Iain Murray (2016); arXiv 1602.07527
  if (upper) {
    L = L.t();
    grad = grad.t();
  }

  auto phi = [](const Tensor & A) -> Tensor {
    auto B = A.tril();
    B = B - 0.5 * at::diag(at::diag(B));
    return B;
  };

  // make sure not to double-count variation, since
  // only half of output matrix is unique
  auto Lbar = grad.tril();

  auto P = phi(at::mm(L.t(), Lbar));
  Tensor S;
  std::tie(S, std::ignore) = at::gesv(P + P.t(), L.t());
  std::tie(S, std::ignore) = at::gesv(S.t(), L.t());
  S = phi(S);
  if (upper) {
    S = S.t();
  }
  return S;
}

Tensor split_backward(const std::vector<torch::autograd::Variable> &grads, int64_t split_size, int64_t dim, IntList sizes, const Type &type) {
  dim = at::maybe_wrap_dim(dim, sizes.size());
  int64_t dim_size = sizes[dim];
  int64_t num_splits = (dim_size + split_size - 1) / split_size;

  // it's possible some of the grads are not defined (represents tensors of all 0s).
  // Since at::cat can't handle those, let's define them
  std::vector<Tensor> grads_all_defined(grads.size());
  for (size_t j = 0; j < grads.size(); ++j) {
    if (grads[j].defined()) {
      grads_all_defined[ j ] = grads[ j ];
    } else {
      auto length = (int64_t)j < (num_splits - 1) ? split_size : split_size - (split_size * num_splits - dim_size);
      std::vector<int64_t> grad_size(sizes);
      grad_size[ dim ] = length;
      grads_all_defined[ j ] = type.zeros(grad_size);
    }
  }

  auto ret =  at::cat(grads_all_defined, dim);
  return ret;
}

Tensor adaptive_max_pool_double_backward(const Tensor & grad, const Tensor & self, const Tensor & indices, int dim) {
  TORCH_ASSERT(indices.dim() >= dim);
  auto size = std::vector<int64_t>(indices.sizes().slice(0, indices.dim() - dim));
  size.push_back(-1);
  auto indices_view = indices.view(size);
  return grad.contiguous().view(size).gather(-1, indices_view).view(indices.sizes());
}

Tensor glu_double_backward(const Tensor & grad, const Tensor & grad_output, const Tensor & input, int64_t dim) {
  auto& gO = grad_output;
  auto input_size = input.size(dim) / 2;
  auto first_half = input.narrow(dim, 0, input_size);
  auto second_half = input.narrow(dim, input_size, input_size);
  auto sig_second_half = second_half.sigmoid();
  auto one_sub_sig_second_half = 1 - sig_second_half;
  auto sig_one_sub_sig = sig_second_half * one_sub_sig_second_half;

  auto ggI_first_half = grad.narrow(dim, 0, input_size);
  auto ggI_second_half = grad.narrow(dim, input_size, input_size);
  auto ggI_second_half_times_first_half = ggI_second_half * first_half;

  auto gI_first_half = ggI_second_half * gO * sig_one_sub_sig;
  auto second_order_sh = sig_one_sub_sig * one_sub_sig_second_half - sig_second_half * sig_one_sub_sig;
  auto gI_second_half = ggI_second_half_times_first_half * gO * second_order_sh + ggI_first_half * gO * sig_one_sub_sig;
  return at::cat({gI_first_half, gI_second_half}, dim);
}

Tensor glu_double_backward_grad_output(const Tensor & grad, const Tensor & input, int64_t dim) {
  if (dim < 0) dim += input.dim();
  std::vector<int64_t> sizes = input.sizes();
  sizes[dim] /= 2;
  auto tmp = grad * glu_backward(input.type().ones(sizes), input, dim);
  return tmp.narrow(dim, 0, sizes[dim]) + tmp.narrow(dim, sizes[dim], sizes[dim]);
}

Tensor kl_div_double_backward_grad_output(const Tensor & grad, const Tensor & input, const Tensor & target, bool size_average, bool reduce) {
  auto result = kl_div_backward(grad, input, target, size_average, false);
  if (reduce && size_average) {
    return result.mean();
  } else if (reduce) {
    return result.sum();
  }
  return result;
}

Tensor log_sigmoid_double_backward(const Tensor & grad, const Tensor & input) {
  auto z = input.sigmoid();
  return grad * (z - 1) * z;
}

Tensor softmax_double_backward(const Tensor & grad, const Tensor & grad_output, int dim, const Tensor & output) {
  auto gO = grad_output;
  auto ggI = grad;

  auto ggI_output = ggI * output;
  auto ggI_out_sum = ggI_output.sum(dim, true);
  auto ggI_out_sum_output = ggI_out_sum * output;
  auto gO_out_sum = (gO * output).sum(dim, true);

  // gI calculation
  auto gI_t0 = ggI_output * (gO - gO_out_sum);
  auto gI_t1 = output * ((ggI_output * gO).sum(dim, true).sub_(gO_out_sum * ggI_out_sum));
  auto gI_t2 = ggI_out_sum_output * gO;
  auto gI_t3 = ggI_out_sum_output * gO_out_sum;
  return gI_t0 - gI_t1 - gI_t2 + gI_t3;
}

Tensor log_softmax_double_backward(const Tensor & grad, const Tensor & grad_output, int dim, const Tensor & output) {
  auto z = output.exp();
  return z * grad_output.sum(dim, true) * ((grad * z).sum(dim, true) - grad);
}

Tensor l1_loss_double_backward_grad_output(const Tensor & grad, const Tensor & input, const Tensor & target, bool size_average, bool reduce) {
  auto output = l1_loss_backward(grad, input, target, size_average, false);
  if (reduce and size_average) {
    return output.mean();
  } else if (reduce) {
    return output.sum();
  }
  return output;
}

Tensor smooth_l1_loss_double_backward(const Tensor & grad, const Tensor & input, const Tensor & target, bool size_average, bool reduce) {
  auto d = (input - target).abs();
  auto grad_input = grad * (d < 1).toType(grad.type());
  if (size_average && reduce) {
    grad_input /= input.numel();
  }
  return grad_input;
}

Tensor smooth_l1_loss_double_backward_grad_output(const Tensor & grad, const Tensor & grad_output, const Tensor & input, const Tensor & target, bool size_average, bool reduce) {
  if (!reduce) {
    return smooth_l1_loss_backward(grad, input, target, size_average, reduce);
  }
  auto r = smooth_l1_loss_backward(ones_like(grad_output), input, target, size_average, true);
  return (r * grad).sum().view({1});
}

Tensor max_pool2d_double_backward(const Tensor & grad, const Tensor & indices) {
  // fold the first two dims together and the last two together
  auto fold = [](const Tensor & t) -> Tensor {
    auto sizes = t.sizes();
    return t.contiguous().view({sizes[0] * sizes[1], sizes[2] * sizes[3]});
  };
  return fold(grad).gather(1, fold(indices)).view(indices.sizes());
}

Tensor mse_loss_double_backward(const Tensor & grad, const Tensor & input, bool size_average, bool reduce) {
  auto grad_input = 2 * grad;
  if (size_average && reduce) {
    grad_input /= input.numel();
  }
  return grad_input;
}

Tensor mse_loss_double_backward_grad_output(const Tensor & grad, const Tensor & grad_output, const Tensor & input, const Tensor & target, bool size_average, bool reduce) {
  if (!reduce) {
    return mse_loss_backward(grad, input, target, size_average, reduce);
  }
  auto r = mse_loss_backward(ones_like(grad_output), input, target, size_average, true);
  return (r * grad).sum().view({1});
}

Tensor soft_margin_loss_double_backward(const Tensor & grad, const Tensor & input, const Tensor & target, bool size_average) {
  auto z = (input * -target).exp();
  auto zplus1 = z + 1;
  auto grad_input = grad * (target * target) * z / (zplus1 * zplus1);
  if (size_average) {
    grad_input /= input.numel();
  }
  return grad_input;
}

Tensor softplus_double_backward(const Tensor & grad, const Tensor & input, Scalar beta, Scalar threshold) {
  auto x = (input * beta);
  return _sigmoid_backward(grad, x.sigmoid()) * (x < threshold).toType(grad.type()) * beta;
}

Tensor as_strided_backward(const Tensor & grad, TensorGeometry base, IntList sizes, IntList strides, int64_t storage_offset) {
  auto src = base.zeros_with_stride(grad.type());
  src.as_strided(sizes, strides, storage_offset - base.storage_offset()).copy_(grad);
  return src;
}

std::tuple<Tensor, Tensor> atan2_backward(const Tensor& grad, const Tensor& self, const Tensor& other, std::array<bool, 2> output_mask) {
  auto recip = (self * self + other * other).reciprocal();
  return std::tuple<Tensor,Tensor>{
            output_mask[0] ? grad * other * recip : Tensor(),
            output_mask[1] ? grad * -self * recip : Tensor() };
}

// TODO: Seriously consider writing the derivative formulas for
// each output separately; there is not all that much sharing
// of computation going on here.
std::tuple<Tensor, Tensor, Tensor> prelu_double_backward(
    const Tensor & mb_ggI,
    const Tensor & mb_ggW,
    const Tensor & mb_gO,
    const Tensor & input,
    const Tensor & weight,
    std::array<bool, 3> output_mask) {

  // Zero-fill undefined grads (TODO: do this more efficiently)
  auto ggI = mb_ggI.defined() ? mb_ggI : input.type().zeros_like(input);
  auto ggW = mb_ggW.defined() ? mb_ggW : weight.type().zeros_like(weight);
  auto gO = mb_gO.defined() ? mb_gO : input.type().zeros_like(input);

  auto positive_mask = (input > 0).type_as(ggI);
  auto nonpositive_mask = (input <= 0).type_as(ggW);

  // Explanation: Let input be i, weight be w, grad_output be gO.
  // f(i, w) = i  if i > 0
  //         = wi if i <= 0
  // df/di * gO  = gO      if i > 0      df/dw * g0 = 0      if i > 0
  //             = g0 * w  if i <= 0                = g0 * i  if i <= 0
  // The rest is taking derivatives of these wrt i, w, gO and summing/expanding properly.

  if (weight.numel() == 1) {
      // from PReLU.forward: num_parameters == 0 is used indicate that a
      // single weight is shared among all input channels.
      auto mask = positive_mask + nonpositive_mask * weight.expand_as(input);
      auto ggO = ggI * mask + ggW.expand_as(gO) * (nonpositive_mask * input);
      return std::tuple<Tensor, Tensor, Tensor>(
                ggO,
                ggW.expand_as(gO) * gO * nonpositive_mask,
                (ggI * gO * nonpositive_mask).sum()
          );
  } else {
      // Expand ggW to match size of ggI; a simple expand doesn't work because
      // ggW is the size of the input channel (dim==1 unless there is only 1 dimension).  For example,
      // let ggI be size (3,4,5,6,7) and ggW be size (4).  Then we unsqueeze ggW to be size (4,1,1,1)
      // so the expand succeeds.
      auto dims_to_unsqueeze = std::max<int64_t>(input.dim() - 2, 0);
      auto ggW_expanded = ggW;
      for (int64_t i = 0; i < dims_to_unsqueeze; i++) {
          ggW_expanded = ggW_expanded.unsqueeze(1);
      }
      ggW_expanded = ggW_expanded.expand_as(ggI);

      auto gI = ggW_expanded * gO * nonpositive_mask;

      auto gW = ggI * gO * nonpositive_mask;
      if (input.dim() > 1) {
          gW = gW.sum(0);
      }
      while (gW.dim() > 1) {
          gW = gW.sum(1);
      }

      Tensor ggO;
      if (output_mask[0]) {
          // expand weight as input as in ggW/ggI above
          auto weight_expanded = weight;
          for (int64_t i = 0; i < dims_to_unsqueeze; i++) {
              weight_expanded = weight_expanded.unsqueeze(1);
          }
          weight_expanded = weight_expanded.expand_as(input);

          auto mask = positive_mask + nonpositive_mask * weight_expanded;
          ggO = ggI * mask + ggW_expanded * nonpositive_mask * input;
      }
      return std::tuple<Tensor,Tensor,Tensor>{ggO, gI, gW};
  }
}

// https://j-towns.github.io/papers/svd-derivative.pdf
Tensor svd_backward(const std::vector<torch::autograd::Variable> &grads, const Tensor& self,
          bool some, const Tensor& raw_u, const Tensor& sigma, const Tensor& raw_v) {
  auto m = self.size(0);
  auto n = self.size(1);
  auto k = sigma.size(0);

  Tensor u, v;
  if (!some) {
    // ignore the free subspace
    u = raw_u.narrow(1, 0, k);
    v = raw_v.narrow(1, 0, k);
  } else {
    u = raw_u;
    v = raw_v;
  }

  auto gu = grads[0];
  auto gsigma = grads[1];
  auto gv = grads[2];
  auto im = self.type().eye(m);
  auto in = self.type().eye(n);
  auto ut = u.t();
  auto vt = v.t();
  auto sigma_mat = sigma.diag();
  auto sigma_mat_inv = sigma.pow(-1).diag();
  auto sigma_expanded_sq = sigma.pow(2).expand_as(sigma_mat);
  auto F = (sigma_expanded_sq - sigma_expanded_sq.t()).pow(-1);
  auto& long_type = sigma.type().toScalarType(at::kLong);
  auto diag_indices = long_type.arange(0, F.numel(), k + 1);
  F.view({-1}).index_fill_(0, diag_indices, 0);

  Tensor u_term, sigma_term, v_term;

  if (gu.defined()) {
    u_term = u.mm(F.mul(ut.mm(gu) - gu.t().mm(u))).mm(sigma_mat);
    if (m > k) {
      u_term = u_term + (im - u.mm(ut)).mm(gu).mm(sigma_mat_inv);
    }
    u_term = u_term.mm(vt);
  } else {
    u_term = self.type().zeros({1}).expand_as(self);
  }

  if (gsigma.defined()) {
    sigma_term = u.mm(gsigma.diag()).mm(vt);
  } else {
    sigma_term = self.type().zeros({1}).expand_as(self);
  }

  if (gv.defined()) {
    auto gvt = gv.t();
    v_term = sigma_mat.mm(F.mul(vt.mm(gv) - gvt.mm(v))).mm(vt);
    if (n > k) {
      v_term = v_term + sigma_mat_inv.mm(gvt.mm(in - v.mm(vt)));
    }
    v_term = u.mm(v_term);
  } else {
    v_term = self.type().zeros({1}).expand_as(self);
  }

  return u_term + sigma_term + v_term;
}

// Formula:
//   d det / d A_ij = \sum_k (\prod_{l neq k} Sigma_l) U_ik V_jk
// that is, if det != 0
//   d det / d A = U * (Sigma / det) * V^T
Tensor _det_with_svd_backward(const std::vector<torch::autograd::Variable> &grads, const Tensor& self,
          const Tensor& det, const Tensor& u, const Tensor& sigma, const Tensor& v) {
  std::vector<torch::autograd::Variable> svd_grads(grads.begin() + 1, grads.end());
  auto svd_term = svd_backward(svd_grads, self, true, u, sigma, v);

  auto det_grad = grads[0];
  auto size = self.size(0);
  auto null_dim = size - sigma.nonzero().size(0);
  if (null_dim >= 2) {
    // \prod_{l neq k} Sigma_l is zero every where
    return svd_term;
  }
  if (null_dim == 1) {
    // only last sigma is 0
    // \prod_{l neq k} Sigma_l is zero at all but last dim
    // at last dim, it is:
    auto scale = sigma.narrow(0, 0, size - 1).prod();
    auto last_u = u.narrow(1, size - 1, 1);
    auto last_v = v.narrow(1, size - 1, 1);
    return svd_term + last_u.mm(last_v.transpose(0, 1)).mul_(scale.mul_(det_grad));
  }
  // no zero singular values
  return svd_term + u.mm(sigma.pow(-1).mul_(det.mul(det_grad)).diag()).mm(v.transpose(0, 1));
}

// Reference:
// https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf
// Sec. 2.3.1 Matrix inverse product
std::tuple<Tensor, Tensor> trtrs_backward(
    const Tensor & grad_x, const Tensor & grad_m,
    const Tensor & b, const Tensor & a, const Tensor & x,
    const bool upper, const bool transpose, const bool unitriangular,
    std::array<bool, 2> output_mask) {
  Tensor grad_b, grad_a;
  if (grad_x.defined()) {
    grad_b = std::get<0>(grad_x.trtrs(a, upper, !transpose, unitriangular));
    if (output_mask[1]) {
      grad_a = transpose ? -x.mm(grad_b.t()) : -grad_b.mm(x.t());
      if (upper) {
        grad_a = grad_a.triu((int) unitriangular);
      } else {
        grad_a = grad_a.tril(-((int) unitriangular));
      }
    }
  }
  if (!grad_a.defined()) {
    grad_a = a.type().zeros({1}).expand_as(a);
  }
  if (!grad_b.defined()) {
    grad_b = b.type().zeros({1}).expand_as(b);
  }
  if (output_mask[1] && grad_m.defined()) {
    grad_a = grad_a.add(grad_m);
  }
  return std::tuple<Tensor, Tensor>{grad_b, grad_a};
}

Tensor sum_exclude_dim1(const Tensor& to_sum, bool keepdim=true) {
  auto r = to_sum.sum(0, keepdim);
  int64_t start_point_exclusive = keepdim ? 1 : 0;
  for (int64_t dim = r.dim() - 1; dim > start_point_exclusive; dim--) {
    r = r.sum(dim, keepdim);
  }
  return r;
}

// similar to expand_as below, but doesn't do the expand_as; operates as if
// reductions were done with keepdim=True
Tensor unsqueeze_dim1(const Tensor& src, const Tensor& target) {
  auto src_expanded = src;
  while (src_expanded.sizes().size() < target.sizes().size() - 1) {
    src_expanded = src_expanded.unsqueeze(1);
  }
  if (src_expanded.sizes().size() == target.sizes().size() - 1) {
    src_expanded = src_expanded.unsqueeze(0);
  }
  return src_expanded;
}

// Helper for batchnorm_double_backward
// because gamma/ggG/ggB are 1-dimensional and represent dim==1, we can't
// do a straight expansion because it won't follow the broadcasting rules.
Tensor expand_as_dim1(const Tensor& src, const Tensor& target) {
  auto src_expanded = src;
  while (src_expanded.sizes().size() < target.sizes().size() - 1) {
    src_expanded = src_expanded.unsqueeze(1);
  }
  return src_expanded.expand_as(target);
}

// NB: This currently is PURPOSELY outside of the anonymous namespace, because
// we are manually calling it from some legacy batchnorm invocation code.  Once
// that code moves into derivatives.yaml, this can be moved into the anonymous
// namespace, BUT NOT BEFORE THEN.
std::tuple<Tensor, Tensor, Tensor> batchnorm_double_backward(
    const Tensor & input,
    const Tensor & gamma,
    const Tensor & ggI,
    const Tensor & ggG,
    const Tensor & ggB,
    const Tensor & gO,
    double eps,
    const Tensor & save_mean_v,
    const Tensor & save_std_v,
    const Tensor & running_mean_v,
    const Tensor & running_var_v,
    bool training) {

  // NB: In the original design of BatchNorm, save_mean, save_std, running_mean
  // and running_var are unconditionally tensor "buffers", and never get wrapped
  // in variables.  However, when ATen happened, we never designed the API
  // to allow mixed passing of tensors and variables (and this would be very
  // confusing, because we always write "Tensor" in the signatures no matter if
  // it's a Variable or a Tensor).  So, when a user calls
  // batchnorm_double_backward from Python (which still thinks that these are
  // plain tensors), it goes ahead and wraps them in variables to appease
  // the interface that only understand variables.  Consequently, we have to
  // unwrap them again.
  const Tensor& save_mean = static_cast<const Variable&>(save_mean_v).data();
  const Tensor& save_std = static_cast<const Variable&>(save_std_v).data();
  const Tensor& running_mean = static_cast<const Variable&>(running_mean_v).data();
  const Tensor& running_var = static_cast<const Variable&>(running_var_v).data();

  bool affine = gamma.defined();
  // TODO: Do we have a ScalarOrTensor type?  Would such a thing exist?
  Tensor gamma_expanded;
  Tensor ggG_expanded, ggB_expanded;
  if (affine) {
    gamma_expanded = expand_as_dim1(gamma, input);
    if (ggG.defined()) {
      ggG_expanded = expand_as_dim1(ggG, input);
    }
    if (ggB.defined()) {
      ggB_expanded = expand_as_dim1(ggB, input);
    }
  } else {
    gamma_expanded = input.type().tensor({}).fill_(1);
  }

  // define some terms we will reuse
  auto M = input.size(0);
  for (auto s : input.sizes().slice(2)) {
    M *= s;
  }
  auto mu = unsqueeze_dim1(make_variable(training ? save_mean : running_mean), input);
  auto input_sub_mu = input - mu;
  auto sigma2_eps_neg_1_2 = unsqueeze_dim1(make_variable(training ? save_std : running_var.add(Scalar(eps)).pow(-0.5)), input);
  auto sigma2_eps_neg_1 = sigma2_eps_neg_1_2.pow(2);
  auto sigma2_eps_neg_3_2 = sigma2_eps_neg_1_2.pow(3);

  // calculate gI
  auto input_mu_sigma2_neg_3_2 = input_sub_mu * sigma2_eps_neg_3_2;
  auto gOinmu_sum = sum_exclude_dim1(gO * input_sub_mu);
  auto gO_sum = sum_exclude_dim1(gO);

  Tensor gI;
  if (ggI.defined() && training) {
    auto ggI_sum = sum_exclude_dim1(ggI);
    auto ggIinmu_sum = sum_exclude_dim1(ggI * input_sub_mu);
    auto all_sub = ((ggI_sum * gO_sum).div_(M)).sub_(sum_exclude_dim1(gO * ggI)).add_(
                    (sigma2_eps_neg_1 * gOinmu_sum * ggIinmu_sum).mul_(3. / M));
    auto gI_0t = (input_mu_sigma2_neg_3_2 * all_sub).div_(M);
    auto gI_1t = (ggIinmu_sum * sigma2_eps_neg_3_2).div_(M) * (gO_sum.div(M) - gO);
    auto gI_2t = (gOinmu_sum * sigma2_eps_neg_3_2).div_(M) * (ggI_sum.div(M) - ggI);
    gI = gamma_expanded * (gI_0t.add_(gI_1t).add_(gI_2t));
  }

  // add contribution of gamma term to gI
  Tensor gI_G_term;
  if (affine && ggG.defined()) {
    if (training) {
      auto t0 = gO * sigma2_eps_neg_1_2;
      auto t1 = (sigma2_eps_neg_1_2 * gO_sum).div_(-M);
      auto t2 = (input_mu_sigma2_neg_3_2 * sum_exclude_dim1(gO * input_sub_mu)).div_(-M);
      gI_G_term = ggG_expanded * (t0.add_(t1).add_(t2));
      gI = gI.defined() ? gI.add_(gI_G_term) : gI_G_term;
    } else {
      gI_G_term = ggG_expanded * sigma2_eps_neg_1_2 * gO;
      gI = gI.defined() ? gI.add_(gI_G_term) : gI_G_term;
    }
  }

  // this is the first backward's grad_input
  auto first_back_grad_input = [&](const Tensor& gO, const Tensor& gamma) -> Tensor {
    auto h0 = (gamma * sigma2_eps_neg_1_2).div_(M);
    auto h1 = (M * gO).sub_(sum_exclude_dim1(gO)).sub_(
                input_sub_mu.mul(sigma2_eps_neg_1) * sum_exclude_dim1(gO * input_sub_mu));
    return h0 * h1;
  };

  // calculate gG
  Tensor gG;
  if (affine && ggI.defined()) {
    if (training) {
      // gG is just the first backwards with the gamma term removed (then shaped properly)
      gG = ggI * first_back_grad_input(gO, sigma2_eps_neg_1_2.type().tensor({}).fill_(1));
      gG = sum_exclude_dim1(gG, false);
    } else {
      gG = sum_exclude_dim1(ggI * gO * sigma2_eps_neg_1_2, false);
    }
  }

  // calculate ggO
  Tensor ggO;
  // contribution of input term
  if (ggI.defined()) {
    if (training) {
      ggO = first_back_grad_input(ggI, gamma_expanded);
    } else {
      ggO = ggI * sigma2_eps_neg_1_2 * gamma_expanded;
    }
  }
  if (ggG.defined()) {
    auto ggO_G_term = ggG_expanded * input_sub_mu * sigma2_eps_neg_1_2;
    ggO = ggO.defined() ? ggO.add_(ggO_G_term) : ggO_G_term;
  }
  if (ggB.defined()) {
    auto ggO_B_term = ggB_expanded;
    ggO = ggO.defined() ? ggO.add_(ggO_B_term) : ggO_B_term;
  }

  return std::tuple<Tensor, Tensor, Tensor>{gI, gG, ggO};

}

} // anonymous namespace

${autograd_function_definitions}

}}} // namespace torch::autograd::generated
