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
// TODO This currently enforces that the entire array is contiguous, but the
// batches don't really have to be for efficiency sake, meaning there may be a
// better way to only force the last two dimensions as contiguous.
Tensor pdist(const Tensor& self, const double p) {
  AT_CHECK(self.dim() >= 2,
      "pdist only supports at least 2D tensors, got: ", self.dim(), "D");
  AT_CHECK(at::isFloatingType(self.type().scalarType()), "pdist only supports floating-point dtypes");
  AT_CHECK(p >= 0, "pdist only supports non-negative p values");
  return at::_pdist_forward(self.contiguous(), p);
}

Tensor _pdist_forward(const Tensor& self, const double p) {
  AT_CHECK(self.is_contiguous(), "_pdist_forward requires contiguous input");
  auto device = self.type().device_type();
  AT_CHECK(device == kCPU || device == kCUDA, "_pdist_forward only supports CPU and CUDA devices, got: ", device);

  const auto batches = self.sizes().slice(0, self.dim() - 2);
  int64_t b = at::tensor(batches).prod().item<int64_t>();
  int64_t n = self.size(-2);
  int64_t m = self.size(-1);
  int64_t c = n * (n - 1) / 2;

  std::vector<int64_t> result_sizes(batches.begin(), batches.end());
  result_sizes.push_back(c);
  Tensor result = at::empty(result_sizes, self.options());

  if (n > 1) {
    if (m == 0) {
      result.fill_(0);
    } else {
      Tensor result_view = result.view({b, c});
      pdist_forward_stub(device, result_view, self.view({b, n, m}), p);
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

  int64_t b = at::tensor(self.sizes().slice(0, self.dim() - 2)).prod().item<int64_t>();
  int64_t n = self.size(-2);
  int64_t m = self.size(-1);
  int64_t c = pdist.size(-1);

  Tensor result_view = result.view({b, n, m});
  pdist_backward_stub(device, result_view, grad.contiguous().view({b, c}), self.view({b, n, m}), p, pdist.view({b, c}));
  return result;
}

Tensor cosine_similarity(const Tensor& x1, const Tensor& x2, int64_t dim, double eps) {
  Tensor w12 = at::sum(x1 * x2, dim);
  Tensor w1 = at::norm(x1, 2, dim);
  Tensor w2 = at::norm(x2, 2, dim);
  return w12.div_((w1 * w2).clamp_min_(eps));
}

}}  // namespace at::native
