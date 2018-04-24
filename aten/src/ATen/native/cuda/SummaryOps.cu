#include "ATen/ATen.h"
#include "ATen/cuda/CUDAApplyUtils.cuh"

#include <type_traits>

namespace at { namespace native {

///////////////// bincount /////////////////
namespace {
template <typename weights_t, typename integral_t>
Tensor _bincount_cuda_template(
    const Tensor& self,
    const Tensor& weights,
    int64_t minlength) {
  if (minlength < 0) {
    AT_ERROR("minlength should be >= 0");
  }
  if (self.dim() != 1 || self.numel() == 0 ||
      !isIntegralType(self.type().scalarType()) ||
      (!std::is_same<integral_t, uint8_t>::value &&
       *self.min().toBackend(kCPU).data<integral_t>() < 0)) {
    AT_ERROR("bincount only supports 1-d non-negative integral inputs.");
  }

  bool has_weights = weights.defined();
  if (has_weights && weights.numel() != self.numel()) {
    AT_ERROR("input and weights should have the same length");
  }

  auto maxScalarGpu = Scalar(self.max());
  auto nbins = maxScalarGpu.local().to<int64_t>() + 1L;
  nbins = std::max(nbins, minlength);
  // alloc output counter on GPU
  Tensor output;
  if (has_weights) {
    output = weights.type().zeros({nbins});
    auto ret = at::cuda::CUDA_tensor_histogram<weights_t, integral_t>(
        output, self, weights, nbins, 1);
  } else {
    output = zeros(CUDA(kLong), {nbins});
    auto ret = at::cuda::CUDA_tensor_histogram<int64_t, integral_t>(
        output, self, weights, nbins, 1);
  }
  return output;
}
} // namespace

Tensor
_bincount_cuda(const Tensor& self, const Tensor& weights, int64_t minlength) {
  return AT_DISPATCH_INTEGRAL_TYPES(self.type(), "bincount", [&] {
    if (weights.type().scalarType() == ScalarType::Float)
      return _bincount_cuda_template<float, scalar_t>(self, weights, minlength);
    return _bincount_cuda_template<double, scalar_t>(self, weights, minlength);
  });
}

}} // namespace at::native
