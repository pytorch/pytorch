#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/cuda/ScanUtils.cuh>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/OpMathType.h>
#include <c10/util/MathConstants.h>

#include <cmath>
#include <limits>

// NOTE: CUDA on Windows requires that the enclosing function
// of a __device__ lambda not have internal linkage.

namespace at::native {

// custom min and max to be used in logaddexp for  complex arguments
template <typename scalar_t, bool min>
__host__ __device__ c10::complex<scalar_t> _logaddexp_minmax(const c10::complex<scalar_t>& x, const c10::complex<scalar_t>& y) {
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
__host__ __device__ scalar_t _log_add_exp_helper(const scalar_t& x, const scalar_t& y) {
  // Reference : https://www.tensorflow.org/api_docs/python/tf/math/cumulative_logsumexp
  // Using the original expression: `at::_isnan(y) ? y : std::min(x, y)` causes an error in ROCM
  const auto isnan_x = at::_isnan(x);
  const auto isnan_y = at::_isnan(y);
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
__host__ __device__ c10::complex<scalar_t> _fast_build_exp(const c10::complex<scalar_t>& x) {
  // complex exponential function, but implemented manually to get fast compilation time
  // this function only handles the case where the x is finite (not inf nor nan)
  const auto xreal = std::real(x);
  const auto ximag = std::imag(x);
  const auto exp_x_abs = std::exp(xreal);
  auto exp_x_real = exp_x_abs * std::cos(ximag);
  auto exp_x_imag = exp_x_abs * std::sin(ximag);
  return {exp_x_real, exp_x_imag};
}

template <typename scalar_t>
__host__ __device__ c10::complex<scalar_t> _fast_build_exp_inf(const c10::complex<scalar_t>& x) {
  // complex exponential function, but implemented manually to get fast compilation time
  // this function only handles the case where the real part of x is infinite
  const auto ximag = std::imag(x);
  constexpr auto exp_x_abs = std::numeric_limits<scalar_t>::infinity();
  if (!::isfinite(ximag)) {  // add this to make consitent with std::exp(x+yi)
    return {exp_x_abs, std::numeric_limits<scalar_t>::quiet_NaN()};
  }
  const auto sin = std::sin(ximag);
  const auto cos = std::cos(ximag);
  // special case if the angle is exactly the multiple of pi/2
  auto exp_x_real = (cos == 0) ? (scalar_t)0.0 : exp_x_abs * cos;
  auto exp_x_imag = (sin == 0) ? (scalar_t)0.0 : exp_x_abs * sin;
  return {exp_x_real, exp_x_imag};
}

template <typename scalar_t>
__host__ __device__ c10::complex<scalar_t> _log_add_exp_helper(const c10::complex<scalar_t>& x, const c10::complex<scalar_t>& y) {
  c10::complex<scalar_t> min = _logaddexp_minmax<scalar_t, /*min=*/true>(x, y);
  c10::complex<scalar_t> max = _logaddexp_minmax<scalar_t, /*min=*/false>(x, y);
  scalar_t min_real = std::real(min);
  scalar_t max_real = std::real(max);

  if (::isnan(min_real) || ::isnan(std::imag(min))) {
    // handling the "infectious" NaNs
    return {std::numeric_limits<scalar_t>::quiet_NaN(), std::numeric_limits<scalar_t>::quiet_NaN()};
  }
  else if ((!::isfinite(min_real)) && (min_real == max_real)) {
    if (min_real < 0) {
      // handle the -inf case, the imaginary part here does not really matter as the exp(value)
      // will be around 0.0 and the angle (i.e. the imaginary part) cannot be determined.
      // It does not matter if we're taking the exp of this value
      return min;
    } else {
      // handle the +inf case, we don't need the special precision for log1p for small values
      // and to avoid producing nan in case of real(max) == real(min) == +inf
      const auto exp_min = _fast_build_exp_inf(min);
      const auto exp_max = _fast_build_exp_inf(max);
      return ::log1p(exp_min + exp_max - 1);  // log1p(x - 1) builds faster than log
    }
  } else {
    const auto minmax = min - max;
    c10::complex<scalar_t> exp_minmax;
    if (!::isfinite(minmax.real())) {
        exp_minmax = minmax.real() < 0 ? c10::complex<scalar_t>{0.0, 0.0} : _fast_build_exp_inf(minmax);
    } else {
        exp_minmax = _fast_build_exp(minmax);
    }
    return ::log1p(exp_minmax) + max;
  }
}

void logaddexp_kernel_cuda(TensorIteratorBase& iter) {
  if (at::isComplexType(iter.dtype())) {
    AT_DISPATCH_COMPLEX_TYPES_AND(at::ScalarType::ComplexHalf, iter.dtype(), "logaddexp_cuda", [&]() {
      using opmath_t = at::opmath_type<scalar_t>;
      gpu_kernel(iter, [] GPU_LAMBDA (scalar_t a_, scalar_t b_) -> scalar_t {
        const auto a = static_cast<opmath_t>(a_);
        const auto b = static_cast<opmath_t>(b_);
        return static_cast<scalar_t>(_log_add_exp_helper(a, b));
      });
    });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(
      ScalarType::BFloat16, ScalarType::Half,
      iter.dtype(), "logaddexp_cuda",
      [&]() {
        using opmath_t = at::opmath_type<scalar_t>;
        gpu_kernel(iter, [] GPU_LAMBDA (scalar_t a_, scalar_t b_) -> scalar_t {
          const auto a = static_cast<opmath_t>(a_);
          const auto b = static_cast<opmath_t>(b_);
          if (::isinf(a) && a == b) {
            return a;
          } else {
            const auto m = ::max(a, b);
            return m + ::log1p(::exp(-::abs(a - b)));
          }
        });
      });
  }
}

void logaddexp2_kernel_cuda(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      ScalarType::BFloat16, ScalarType::Half,
      iter.dtype(), "logaddexp2_cuda",
      [&]() {
        using opmath_t = at::opmath_type<scalar_t>;
        const auto inv_log_2 = static_cast<opmath_t>(1.0 / c10::ln_2<double>);
        gpu_kernel(iter, [inv_log_2] GPU_LAMBDA (scalar_t a_, scalar_t b_) -> scalar_t {
          const auto a = static_cast<opmath_t>(a_);
          const auto b = static_cast<opmath_t>(b_);
          if (::isinf(a) && a == b) {
            return a;
          } else {
            const auto m = ::max(a, b);
            return m + ::log1p(::exp2(-::abs(a - b))) * inv_log_2;
          }
        });
      });
}

REGISTER_DISPATCH(logaddexp_stub, &logaddexp_kernel_cuda)
REGISTER_DISPATCH(logaddexp2_stub, &logaddexp2_kernel_cuda)

} // namespace at::native
