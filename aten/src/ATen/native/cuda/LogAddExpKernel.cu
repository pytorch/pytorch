#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/cuda/JitLoops.cuh>
#include <ATen/native/cuda/jit_utils.h>
#include <ATen/native/cuda/ScanUtils.cuh>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/OpMathType.h>
#include <c10/util/MathConstants.h>
#include <c10/util/complex.h>

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
  if (!::isfinite(ximag)) {  // add this to make consistent with std::exp(x+yi)
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

// Complex logaddexp jiterator string
const auto logaddexp_complex_string = jiterator_stringify(
    template<typename T>
    std::complex<T> log1p(const std::complex<T>& z)
    {
      using complex_t = std::complex<T>;
      T x = z.real();
      T y = z.imag();
      T zabs = abs(z);
      T theta = atan2(y, x + T(1));
      if (zabs < 0.5) {
          T r = x * (T(2) + x) + y * y;
          if (r == 0) { // handle underflow
              return complex_t(x, theta);
          }
          return complex_t(T(0.5) * std::log1p(r), theta);
      } else {
          T z0 = std::hypot(x + 1, y);
          return complex_t(log(z0), theta);
      }
    }

    // separated _logaddexp_minmax into 2 different functions for jiterator_string
    template <typename T>
    std::complex<T> logaddexp_min(const std::complex<T>& x, const std::complex<T>& y) {
        T xr = x.real();
        T yr = y.real();
        if (isnan(yr) || isnan(y.imag())) {
            return y;
        } else if (isnan(xr) || isnan(x.imag())) {
            return x;
        } else {
            return (xr < yr) ? x : y;
        }
    }

    template <typename T>
    std::complex<T> logaddexp_max(const std::complex<T>& x, const std::complex<T>& y) {
        T xr = x.real();
        T yr = y.real();
        if (isnan(yr) || isnan(y.imag())) {
            return y;
        } else if (isnan(xr) || isnan(x.imag())) {
            return x;
        } else {
            return (xr >= yr) ? x : y;
        }
    }

    template <typename T>
    std::complex<T> fast_build_exp(const std::complex<T>& x) {
        const auto xreal = x.real();
        const auto ximag = x.imag();
        const auto exp_x_abs = exp(xreal);
        auto exp_x_real = exp_x_abs * cos(ximag);
        auto exp_x_imag = exp_x_abs * sin(ximag);
        return std::complex<T>(exp_x_real, exp_x_imag);
    }

    template <typename T>
    std::complex<T> fast_build_exp_inf(const std::complex<T>& x) {
        using complex_t = std::complex<T>;
        const auto ximag = x.imag();
        const T exp_x_abs = INFINITY;
        if (!isfinite(ximag)) {
            return complex_t(exp_x_abs, NAN);
        }
        const auto sin_val = sin(ximag);
        const auto cos_val = cos(ximag);
        auto exp_x_real = (cos_val == T(0)) ? T(0) : exp_x_abs * cos_val;
        auto exp_x_imag = (sin_val == T(0)) ? T(0) : exp_x_abs * sin_val;
        return complex_t(exp_x_real, exp_x_imag);
    }

    template <typename complex_t>
    complex_t logaddexp_complex(complex_t x, complex_t y) {
        using T = typename complex_t::value_type;
        complex_t min_val = logaddexp_min(x, y);
        complex_t max_val = logaddexp_max(x, y);
        T min_real = min_val.real();
        T max_real = max_val.real();

        if (isnan(min_real) || isnan(min_val.imag())) {
            return complex_t(NAN, NAN);
        }
        else if ((!isfinite(min_real)) && (min_real == max_real)) {
            if (min_real < T(0)) {
                return min_val;
            } else {
                const auto exp_min = fast_build_exp_inf<T>(min_val);
                const auto exp_max = fast_build_exp_inf<T>(max_val);
                return log1p(exp_min + exp_max - complex_t(1, 0));
            }
        } else {
            const auto minmax = min_val - max_val;
            complex_t exp_minmax;
            if (!isfinite(minmax.real())) {
                exp_minmax = (minmax.real() < T(0)) ? complex_t(0, 0) : fast_build_exp_inf<T>(minmax);
            } else {
                exp_minmax = fast_build_exp<T>(minmax);
            }
            return log1p(exp_minmax) + max_val;
        }
    }
);

constexpr char logaddexp_complex_name[] = "logaddexp_complex";
void logaddexp_kernel_cuda(TensorIteratorBase& iter) {
  if (at::isComplexType(iter.dtype())) {
#if AT_USE_JITERATOR()
    AT_DISPATCH_COMPLEX_TYPES_AND(at::ScalarType::ComplexHalf, iter.dtype(), "logaddexp_cuda", [&]() {
      jitted_gpu_kernel<
          /*name=*/logaddexp_complex_name,
          /*return_dtype=*/scalar_t,
          /*common_dtype=*/scalar_t,
          /*arity=*/2>(iter, logaddexp_complex_string);
    });
#else
    AT_DISPATCH_COMPLEX_TYPES_AND(at::ScalarType::ComplexHalf, iter.dtype(), "logaddexp_cuda", [&]() {
      using opmath_t = at::opmath_type<scalar_t>;
      gpu_kernel(iter, [] GPU_LAMBDA (scalar_t a_, scalar_t b_) -> scalar_t {
        const auto a = static_cast<opmath_t>(a_);
        const auto b = static_cast<opmath_t>(b_);
        return static_cast<scalar_t>(_log_add_exp_helper(a, b));
      });
    });
#endif
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
