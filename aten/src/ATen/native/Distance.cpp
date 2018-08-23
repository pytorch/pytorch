#include "ATen/ATen.h"
#include "ATen/Dispatch.h"
#include "ATen/NativeFunctions.h"
#include "DistanceOpsKernel.h"
#include "cuda/DistanceKernel.cuh"

namespace at { namespace native {

Tensor pairwise_distance(const Tensor& x1, const Tensor& x2, double p, double eps, bool keepdim) {
  return at::norm(x1 - x2 + eps, p, 1, keepdim);
}

// This is to guarantee that the contiguous memory is passed to the backward pass
Tensor pdist(const Tensor& self, const double p) {
  AT_CHECK(self.dim() == 2,
      "pdist only supports 2D tensors, got: ", self.dim(), "D");
  AT_CHECK(at::isFloatingType(self.type().scalarType()), "pdist only supports floating-point dtypes");
  AT_CHECK(p >= 0, "pdist only supports non-negative p values");
  return at::_pdist_forward(self.contiguous(), p);
}

Tensor _pdist_forward(const Tensor& self, const double p) {
  AT_CHECK(self.is_contiguous(), "_pdist_forward requires contiguous input");
  Tensor result = self.type().tensor();
  if (self.size(0) <= 1) {
    result.resize_({0});
  } else {
    int64_t n = self.size(0);
    int64_t c = n * (n - 1) / 2;
    result.resize_({c});
    if (self.size(1) == 0) {
      result.fill_(0);
    } else if (self.type().backend() == Backend::CPU) {
      pdist_kernel_cpu(result, self, p);
    } else if (self.type().backend() == Backend::CUDA) {
      pdist_kernel_cuda(result, self, p);
    } else {
      AT_ERROR("pdist only supports CPU and CUDA backends, got: ", at::toString(self.type().backend()));
    }
  }
  return result;
}

Tensor _pdist_backward(const Tensor& grad, const Tensor& self, const double p, const Tensor& pdist) {
  AT_CHECK(self.is_contiguous(), "_pdist_backward requires self to be contiguous");
  AT_CHECK(pdist.is_contiguous(), "_pdist_backward requires pdist to be contiguous");
  Tensor result = at::empty_like(self);
  if (self.type().backend() == Backend::CPU) {
    pdist_backward_kernel_cpu(result, grad, self, p, pdist);
  } else if (self.type().backend() == Backend::CUDA) {
    pdist_backward_kernel_cuda(result, grad, self, p, pdist);
  } else {
    AT_ERROR("pdist_backward only supports CPU backend, got: ", at::toString(self.type().backend()));
  }
  return result;
}

}}  // namespace at::native
