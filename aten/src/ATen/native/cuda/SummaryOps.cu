#include "ATen/ATen.h"
#include "ATen/cuda/CUDAApplyUtils.cuh"

namespace at { namespace native {

///////////////// bincount /////////////////
namespace {
template <typename input_t, typename weights_t>
Tensor _bincount_cuda_template(
    const Tensor& self,
    const Tensor& weights,
    int64_t minlength) {
  if (minlength < 0) {
    AT_ERROR("minlength should be >= 0");
  }
  if (self.dim() != 1 || self.numel() == 0 ||
      (!std::is_same<input_t, uint8_t>::value &&
       *self.min().toBackend(kCPU).data<input_t>() < 0)) {
    AT_ERROR("bincount only supports 1-d non-negative integral inputs.");
  }

  bool has_weights = weights.defined();
  if (has_weights && weights.size(0) != self.size(0)) {
    AT_ERROR("input and weights should have the same length");
  }

  auto maxScalarGpu = Scalar(self.max());
  auto nbins = maxScalarGpu.local().to<int64_t>() + 1L;
  nbins = std::max(nbins, minlength);
  // alloc output counter on GPU
  Tensor output;
  if (has_weights) {
    output = zeros(weights.type(), {nbins});
    auto ret = at::cuda::CUDA_tensor_histogram<weights_t, input_t, true>(
        output, self, weights, nbins, 1);
  } else {
    output = zeros(CUDA(kLong), {nbins});
    auto ret = at::cuda::CUDA_tensor_histogram<int64_t, input_t, false>(
        output, self, weights, nbins, 1);
  }
  return output;
}
} // namespace

Tensor
_bincount_cuda(const Tensor& self, const Tensor& weights, int64_t minlength) {
  return AT_DISPATCH_INTEGRAL_TYPES(self.type(), "bincount", [&] {
    const auto scalar = weights.type().scalarType();
    if (scalar == ScalarType::Undefined || scalar == ScalarType::Float)
      return _bincount_cuda_template<scalar_t, float>(self, weights, minlength);
    return _bincount_cuda_template<scalar_t, double>(
        self, weights.toType(CUDA(kDouble)), minlength);
  });
}

}} // namespace at::native
