#include "ATen/ATen.h"
#include "ATen/Dispatch.h"
#include "ATen/NativeFunctions.h"
#include "cpu/DistanceOpsKernel.h"

namespace at { namespace native {

DEFINE_DISPATCH(pdist_kernel);
DEFINE_DISPATCH(pdist_backward_kernel);

Tensor pairwise_distance(const Tensor& x1, const Tensor& x2, double p, double eps, bool keepdim) {
  return at::norm(x1 - x2 + eps, p, 1, keepdim);
}

Tensor& _pdist_out_cpu(Tensor& result, const Tensor& self, const double p) {
  AT_CHECK(self.dim() == 2,
      "pdist only supports 2D tensors, got: ", self.dim(), "D");
  AT_CHECK(at::isFloatingType(self.type().scalarType()), "pdist only supports floating-point dtypes");
  if (self.size(0) <= 1) {
    result.resize_({0});
  } else {
    int64_t n = self.size(0);
    int64_t c = n * (n - 1) / 2;
    result.resize_({c});
    if (self.size(1) == 0) {
      result.fill_(0);
    } else {
      pdist_kernel(kCPU, result, self, p);
    }
  }
  return result;
}

Tensor _pdist_cpu(const Tensor& self, const double p) {
  Tensor result = self.type().tensor();
  return _pdist_out_cpu(result, self, p);
}

Tensor _pdist_backward_cpu(const Tensor& grad, const Tensor& self, const double p, const Tensor& pdist) {
  Tensor result = at::native::zeros_like(self);
  pdist_backward_kernel(kCPU, result, grad, self, p, pdist);
  return result;
}

}}  // namespace at::native
