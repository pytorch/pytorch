#include "Functions.h"

// ${generated_comment}

using namespace at;

namespace torch { namespace autograd {

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
  auto result = self.type().zeros(sizes);
  // TODO: implement unnarrow
  return result;
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

${autograd_function_definitions}

}} // namespace torch::autograd
