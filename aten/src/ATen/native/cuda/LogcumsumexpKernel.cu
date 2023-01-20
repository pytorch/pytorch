#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/core/TensorBase.h>
#include <ATen/OpMathType.h>
#include <ATen/Dispatch.h>

#include <ATen/native/cuda/ScanKernels.h>
#include <ATen/native/cuda/ScanUtils.cuh>

#include <cmath>
#include <limits>

namespace at::native {

// custom min and max to be used in logcumsumexp for complex arguments
template <typename scalar_t>
__host__ __device__ c10::complex<scalar_t> _logcumsumexp_minmax(c10::complex<scalar_t> x, c10::complex<scalar_t> y, bool min) {
  scalar_t xr = std::real(x);
  scalar_t yr = std::real(y);
  if (::isnan(yr) || (::isnan(std::imag(y)))) {
    return y;
  } else if (::isnan(xr) || (::isnan(std::imag(x)))) {
    return x;
  } else if (min) { // min
    return (xr < yr) ? x : y;
  } else { // max
    return (xr >= yr) ? x : y;
  }
}

template <typename scalar_t>
__host__ __device__ scalar_t _log_add_exp_helper(scalar_t x, scalar_t y) {
  // Reference : https://www.tensorflow.org/api_docs/python/tf/math/cumulative_logsumexp
  // Using the original expression: `at::_isnan(y) ? y : std::min(x, y)` causes an error in ROCM
  auto isnan_x = at::_isnan(x);
  auto isnan_y = at::_isnan(y);
  scalar_t min = isnan_y ? y : (isnan_x ? x : std::min(x, y));
  scalar_t max = isnan_y ? y : (isnan_x ? x : std::max(x, y));
  if (min != max || ::isfinite(min)) {
    // nan will be propagated here
    return ::log1p(std::exp(min - max)) + max;
  } else {
    // special case to correctly handle infinite cases
    return x;
  }
}

template <typename scalar_t>
__host__ __device__ c10::complex<scalar_t> _log_add_exp_helper(c10::complex<scalar_t> x, c10::complex<scalar_t> y) {
  c10::complex<scalar_t> min = _logcumsumexp_minmax(x, y, /*min=*/true);
  c10::complex<scalar_t> max = _logcumsumexp_minmax(x, y, /*min=*/false);
  scalar_t min_real = std::real(min);
  scalar_t max_real = std::real(max);

  if (::isnan(min_real) || ::isnan(std::imag(min))) {
    // handling the "infectious" NaNs
    return {std::numeric_limits<scalar_t>::quiet_NaN(), std::numeric_limits<scalar_t>::quiet_NaN()};
  } else if ((!::isfinite(min_real)) && (min_real == max_real)) {
    if (min_real < 0) {
      // handle the -inf case, the imaginary part here does not really matter as the exp(value)
      // will be around 0.0 and the angle (i.e. the imaginary part) cannot be determined.
      // It does not matter if we're taking the exp of this value
      return min;
    } else {
      // handle the +inf case, we don't need the special precision for log1p for small values
      // and to avoid producing nan in case of real(max) == real(min) == +inf
      return ::log(::exp(min) + ::exp(max));
    }
  } else {
    return ::log1p(::exp(min - max)) + max;
  }
}

void launch_logcumsumexp_cuda_kernel(const TensorBase& result, const TensorBase& self, int64_t dim) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      ScalarType::Half, ScalarType::BFloat16,
      self.scalar_type(), "logcumsumexp_cuda",
      [&]() {
        using opmath_t = at::opmath_type<scalar_t>;
        scalar_t init = -std::numeric_limits<scalar_t>::infinity();
        auto log_add_exp = [] C10_HOST_DEVICE (const scalar_t x_, const scalar_t y_) -> scalar_t {
          const opmath_t x{x_}, y{y_};
          return _log_add_exp_helper(x, y);
        };
        scan_dim<scalar_t>(self, result, dim, init, log_add_exp);
      });
}

} // namespace at::native
