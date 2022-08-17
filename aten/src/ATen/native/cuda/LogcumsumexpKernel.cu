#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/core/TensorBase.h>
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>

#include <ATen/native/cuda/ScanKernels.h>
#include <ATen/native/cuda/ScanUtils.cuh>

#include <cmath>
#include <limits>

namespace at { namespace native {

void launch_logcumsumexp_cuda_kernel(const TensorBase& result, const TensorBase& self, int64_t dim) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      ScalarType::Half, ScalarType::BFloat16,
      self.scalar_type(), "logcumsumexp_cuda",
      [&]() {
        using accscalar_t = acc_type<scalar_t, true>;
        scalar_t init = -std::numeric_limits<scalar_t>::infinity();
        auto log_add_exp = [] C10_HOST_DEVICE (const scalar_t x, const scalar_t y) -> scalar_t {
          scalar_t min = at::_isnan(y) ? y : std::min<scalar_t>(x,y); //std::min returns first arg if one of the args is nan
          scalar_t max = at::_isnan(y) ? y : std::max<scalar_t>(x,y); //std::max returns first arg if one of the args is nan
          if (min != max || ::isfinite(static_cast<accscalar_t>(min))) {
          // nan will be propagated here
              return ::log1p(std::exp(min - max)) + max;
          } else {
          // special case to correctly handle infinite inputs
             return x;
          }
        };
        scan_dim<scalar_t>(self, result, dim, init, log_add_exp);
      });
}

}} // namespace at::native
