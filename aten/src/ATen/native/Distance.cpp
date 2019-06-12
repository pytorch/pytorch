#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/NativeFunctions.h>

#include <ATen/native/Distance.h>

namespace at { namespace native {

DEFINE_DISPATCH(pdist_forward_stub);
DEFINE_DISPATCH(pdist_backward_stub);

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
  auto device = self.type().device_type();
  AT_CHECK(device == kCPU || device == kCUDA, "_pdist_forward only supports CPU and CUDA devices, got: ", device);
  Tensor result = at::empty({0}, self.options());
  if (self.size(0) <= 1) {
    result.resize_({0});
  } else {
    int64_t n = self.size(0);
    int64_t c = n * (n - 1) / 2;
    result.resize_({c});
    if (self.size(1) == 0) {
      result.fill_(0);
    } else {
      pdist_forward_stub(device, result, self, p);
    }
  }
  return result;
}

Tensor _pdist_backward(const Tensor& grad, const Tensor& self, const double p, const Tensor& pdist) {
  AT_CHECK(self.is_contiguous(), "_pdist_backward requires self to be contiguous");
  AT_CHECK(pdist.is_contiguous(), "_pdist_backward requires pdist to be contiguous");
  auto device = self.type().device_type();
  AT_CHECK(device == kCPU || device == kCUDA, "_pdist_backward only supports CPU and CUDA devices, got: ", device);
  Tensor result = at::empty_like(self);
  pdist_backward_stub(device, result, grad, self, p, pdist);
  return result;
}

}}  // namespace at::native
