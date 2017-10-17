#include "Functions.h"

// ${generated_comment}

using at::Tensor;
using at::Scalar;
using at::IntList;

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

Tensor norm_backward(const Tensor & grad, const Tensor & self, const Scalar & p_) {
  auto p = p_.toDouble();
  auto norm = self.norm(p_);

  if (norm.toDouble() == 0.0) {
    // handle case at 0 where we return a subgradient containing 0
    return zeros_like(self);
  }

  if (p == 2.0) {
    return self * (grad / norm);
  } else {
    auto pow_ = self.abs().pow(p - 2);
    auto scale_v = grad / norm.toTensor().pow(p - 1);
    return self * pow_ * scale_v;
  }
}

Tensor norm_backward(Tensor grad, const Tensor & self, const Scalar & p_, int64_t dim, bool keepdim) {
  if (!keepdim && self.dim() > 1) {
    grad = grad.unsqueeze(dim);
  }
  auto p = p_.toDouble();
  auto norm = self.norm(p, dim, true);
  Tensor grad_input;
  if (p == 2.0) {
    grad_input = self * (grad / norm);
  } else {
    auto pow_ = self.abs().pow(p - 2);
    auto scale_v = grad / norm.pow(p - 1);
    grad_input = self * pow_ * scale_v;
  }
  // handle case at 0 where we return a subgradient containing 0
  grad_input.masked_fill_(norm == 0, 0);
  return grad_input;
}

Tensor reduce_to(const Tensor & grad, IntList sizes) {
  if (sizes.size() == 0) {
    return grad.sum().toTensor();
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

Tensor sum_backward(const Tensor & grad, IntList sizes, int64_t dim, bool keepdim) {
  if (!keepdim && sizes.size() > 1) {
    return grad.unsqueeze(dim).expand(sizes);
  } else {
    return grad.expand(sizes);
  }
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
  int64_t nDims = sizes.size();
  for (int64_t dim = 0; dim < nDims; dim++) {
    if (sizes[dim] == 1) {
      result = result.unsqueeze(dim);
    }
  }
  return result;
}

Tensor maybe_unsqueeze(const Tensor & self, int64_t dim, int64_t prev_size) {
  if (prev_size == 1) {
    return self.unsqueeze(dim);
  }
  return self;
}

Tensor addmm_self_backward(const Tensor & grad, const Scalar &beta) {
  return maybe_multiply(grad, beta);
}

Tensor addmm_mat1_backward(const Tensor & grad, const Tensor & mat1, const Tensor & mat2, const Scalar & alpha) {
  auto mat1Strides = mat1.strides();
  auto mat1Sizes = mat1.sizes();
  if (mat1Strides[0] == 1 && mat1Strides[1] == mat1Sizes[0]) {
    return maybe_multiply(mat2.mm(grad.t()).t(), alpha);
  } else {
    return maybe_multiply(grad.mm(mat2.t()), alpha);
  }
}

Tensor addmm_mat2_backward(const Tensor & grad, const Tensor & mat1, const Tensor & mat2, const Scalar & alpha) {
  auto mat2Strides = mat2.strides();
  auto mat2Sizes = mat2.sizes();

  if (mat2Strides[0] == 1 && mat2Strides[1] == mat2Sizes[0]) {
    return maybe_multiply(grad.t().mm(mat1).t(), alpha);
  } else {
    return maybe_multiply(mat1.t().mm(grad), alpha);
  }
}

variable_list cat_tensors_backward(const Tensor & grad, const std::vector<int64_t> &sizes, int64_t dim) {
  variable_list grad_inputs(sizes.size());
  int64_t accumulate = 0;
  for (size_t i = 0; i < sizes.size(); ++i) {
    auto size = sizes[i];
    accumulate += size;
    grad_inputs[i] = grad.narrow(dim, accumulate - size, size);
  }
  return grad_inputs;
}

Tensor select_backward_scalar(Tensor grad, const Tensor & input, const Tensor & value) {
  if (grad.dim() == 1) {
    // TODO: remove this once zero-dim tensor work properly in PyTorch
    grad = grad.view({});
  }
  auto grad_input = zeros_like(input);
  grad_input.masked_fill_(input == value, Scalar(grad));
  return grad_input;
}

Tensor select_backward(Tensor grad, int64_t dim, Tensor indices, IntList sizes, bool keepdim) {
  if (!keepdim && sizes.size() > 1) {
    grad = grad.unsqueeze(dim);
    indices = indices.unsqueeze(dim);
  }
  return grad.type().zeros(sizes).scatter_(dim, indices, grad);
}

Tensor trace_backward(const Tensor & grad, IntList sizes) {
  if (sizes.size() != 2) {
    throw std::runtime_error("expected matrix input");
  }

  // TODO: simplify once toScalarType is virtual
  auto& long_type = *VariableImpl::getType(
      Variable(grad).data().type().toScalarType(at::kLong));

  auto grad_input = grad.type().zeros(sizes[0] * sizes[1]);
  auto indices = long_type.arange(0, grad_input.numel(), sizes[1] + 1);
  grad_input.index_fill_(0, indices, Scalar(grad.view({})));
  return grad_input.view(sizes);
}

Tensor unfold_backward(const Tensor & grad, IntList input_sizes, int64_t dim, int64_t size, int64_t step) {
  // TODO: simplify once toScalarType is virtual
  auto& long_type = *VariableImpl::getType(
      Variable(grad).data().type().toScalarType(at::kLong));

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
  return S;
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

Tensor smooth_l1_loss_double_backward(const Tensor & grad, const Tensor & input, const Tensor & target, bool size_average) {
  auto d = (input - target).abs();
  auto grad_input = grad * (d < 1).toType(grad.type());
  if (size_average) {
    grad_input /= input.numel();
  }
  return grad_input;
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
  return (r * grad).sum().toTensor().view({1});
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

}

${autograd_function_definitions}

}}} // namespace torch::autograd::generated
