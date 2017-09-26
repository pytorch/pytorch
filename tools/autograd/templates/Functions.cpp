#include "Functions.h"

// ${generated_comment}

using namespace at;

namespace torch { namespace autograd {

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
  if (p == 2.0) {
    return self * (grad / self.norm(2));
  } else {
    auto pow_ = self.abs().pow(p - 2);
    auto scale_v = grad / self.norm(p).toTensor().pow(p - 1);
    return self * pow_ * scale_v;
  }
}

Tensor norm_backward(const Tensor & grad, const Tensor & self, const Scalar & p, int64_t dim, bool keepdim) {
  throw std::runtime_error("norm_backward(dim): NYI");
}

Tensor reduce_to(const Tensor & grad, IntList sizes) {
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
  if (!keepdim) {
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

Tensor unnarrow(const Tensor & self, IntList sizes, int64_t dimension, int64_t offset) {
  throw std::runtime_error("unnarrow: not yet implemented");
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

${autograd_function_definitions}

}} // namespace torch::autograd
