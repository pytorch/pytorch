#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/OpMathType.h>
#include <ATen/native/cuda/jit_utils.h>
#include <c10/util/MathConstants.h>
#include <ATen/native/cuda/JitLoops.cuh>

// NOTE: CUDA on Windows requires that the enclosing function
// of a __device__ lambda not have internal linkage.

namespace at::native {

#if AT_USE_JITERATOR()

const auto logaddexp_string = jiterator_stringify(
  // custom min and max to be used in logcumsumexp for complex arguments
  template <typename scalar_t>
  __host__ __device__ std::pair<c10::complex<scalar_t>, c10::complex<scalar_t>> _logcumsumexp_minmax(
      const c10::complex<scalar_t>& x, const c10::complex<scalar_t>& y) {
    scalar_t xr = std::real(x);
    scalar_t yr = std::real(y);
    if (::isnan(yr) || (::isnan(std::imag(y)))) {
      return std::make_pair(y, y);
    } else if (::isnan(xr) || (::isnan(std::imag(x)))) {
      return std::make_pair(x, x);
    } else {
      return (xr < yr) ? std::make_pair(x, y) : std::make_pair(y, x);
    }
  }

  template <typename scalar_t>
  __host__ __device__ c10::complex<scalar_t> _log_add_exp_helper(const c10::complex<scalar_t>& x, const c10::complex<scalar_t>& y) {
    auto [min, max] = _logcumsumexp_minmax<scalar_t>(x, y);
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
        auto exp_min = std::exp(min);
        auto exp_max = std::exp(max);
        return std::log(exp_min + exp_max);
      }
    } else {
      auto minmax = min - max;
      auto exp_minmax = std::exp(minmax);
      return ::log1p(exp_minmax) + max;
    }
  }
)
CONSTEXPR_EXCEPT_WIN_CUDA char logaddexp_name[] = "logaddexp_kernel";
#else
// happy to remove this if I don't need to provide the non-jiterator implementation
#include <ATen/native/cuda/LogAddExp.cuh>
#endif

void logaddexp_kernel_cuda(TensorIteratorBase& iter) {
  auto dtype = iter.dtype();
  if (isComplexType(dtype)) {
#if AT_USE_JITERATOR()
    AT_DISPATCH_COMPLEX_TYPES(dtype, "logaddexp_cuda", [&]() {
      jitted_gpu_kernel<
        /*name=*/ logaddexp_name,
        /*return_dtype=*/ scalar_t,
        /*common_dtype=*/ scalar_t,
        /*arity=*/ 2>(iter, logaddexp_string);
    });
#else
  AT_DISPATCH_COMPLEX_TYPES(iter.dtype(), "logaddexp_cuda",
      [&]() {
        using opmath_t = at::opmath_type<scalar_t>;
        gpu_kernel(iter, [] GPU_LAMBDA (scalar_t a_, scalar_t b_) -> scalar_t {
          const auto a = static_cast<opmath_t>(a_);
          const auto b = static_cast<opmath_t>(b_);
          return _log_add_exp_helper(a, b);
        });
      });
#endif // AT_USE_JITERATOR()
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
  AT_DISPATCH_FLOATING_TYPES_AND(
      ScalarType::BFloat16,
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

REGISTER_DISPATCH(logaddexp_stub, &logaddexp_kernel_cuda);
REGISTER_DISPATCH(logaddexp2_stub, &logaddexp2_kernel_cuda);

} // namespace at::native
