#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/native/UnaryOps.h>

#include <cmath>
#include <limits>
#include <type_traits>

#include <ATen/Config.h>
#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/cpu/vml.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/CopyKernel.h>
#include <ATen/native/cpu/Loops.h>
#include <ATen/native/cpu/zmath.h>
#include <ATen/OpMathType.h>

#include <c10/util/math_compat.h>
#include <c10/util/MathConstants.h>
#include <c10/core/Scalar.h>
#include <c10/util/TypeSafeSignMath.h>
#include <c10/util/irange.h>

#if AT_MKL_ENABLED()
#include <mkl.h>
#endif

namespace at::native {

inline namespace CPU_CAPABILITY {

using namespace vec;

static void sigmoid_kernel(TensorIteratorBase& iter) {
  const auto dtype = iter.common_dtype();
  if (at::isReducedFloatingType(dtype)) {
    AT_DISPATCH_REDUCED_FLOATING_TYPES(dtype, "sigmoid_cpu_reduced_float", [&]() {
      cpu_kernel_vec(
          iter,
          [=](scalar_t a) -> scalar_t {
            float a0 = static_cast<float>(a);
            return static_cast<float>(1) / (static_cast<float>(1) + std::exp((-a0)));
          },
          [=](Vectorized<scalar_t> a) {
            Vectorized<float> a0, a1;
            std::tie(a0, a1) = convert_to_float<scalar_t>(a);
            a0 = (Vectorized<float>(static_cast<float>(1)) + a0.neg().exp()).reciprocal();
            a1 = (Vectorized<float>(static_cast<float>(1)) + a1.neg().exp()).reciprocal();
            return convert_from_float<scalar_t>(a0, a1);
          });
    });
  } else {
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(dtype, "sigmoid_cpu", [&]() {
      cpu_kernel_vec(
          iter,
          [=](scalar_t a) -> scalar_t {
            return (static_cast<scalar_t>(1) / (static_cast<scalar_t>(1) + std::exp((-a))));
          },
          [=](Vectorized<scalar_t> a) {
            a = Vectorized<scalar_t>(static_cast<scalar_t>(0)) - a;
            a = a.exp();
            a = Vectorized<scalar_t>(static_cast<scalar_t>(1)) + a;
            a = a.reciprocal();
            return a;
          });
    });
  }
}

#if AT_MKL_ENABLED()

template <typename T>
void VmlLog(int64_t N, const T* X, T* Y) {
  constexpr int64_t K = Vectorized<T>::size();
  at::parallel_for(0, N, K, [=](int64_t begin, int64_t end) {
    using VT = at::opmath_type<T>;
    vec::map(
        [](Vectorized<VT> x_vec) { return x_vec.log(); },
        Y + begin,
        X + begin,
        end - begin);
  });
}

template <>
void VmlLog<float>(int64_t N, const float* X, float* Y) {
  vsLn(N, X, Y);
}

template <>
void VmlLog<double>(int64_t N, const double* X, double* Y) {
  vdLn(N, X, Y);
}

template <typename T>
void LogitMKLKernel(T eps, TensorIteratorBase* it) {
  if (!it->can_use_32bit_indexing()) {
    for (auto& sub_it : it->with_32bit_indexing()) {
      LogitMKLKernel<T>(eps, &sub_it);
    }
    return;
  }

  constexpr int64_t K = Vectorized<T>::size();
  const int64_t N = it->numel();
  const T* X_data = static_cast<T*>(it->data_ptr(1));
  T* Y_data = static_cast<T*>(it->data_ptr(0));
  if (eps < T(0)) {
    at::parallel_for(0, N, K, [=](int64_t begin, int64_t end) {
      for (const auto i : c10::irange(begin, end)) {
        Y_data[i] = X_data[i] == T(1) ? std::numeric_limits<T>::infinity()
                                      : X_data[i] / (T(1) - X_data[i]);
      }
      VmlLog<T>(end - begin, Y_data + begin, Y_data + begin);
    });
  } else {
    const T lo = eps;
    const T hi = T(1) - eps;
    at::parallel_for(0, N, K, [=](int64_t begin, int64_t end) {
      for (const auto i : c10::irange(begin, end)) {
        const T x = X_data[i] < lo ? lo : (X_data[i] > hi ? hi : X_data[i]);
        Y_data[i] =
            x == T(1) ? std::numeric_limits<T>::infinity() : (x / (T(1) - x));
      }
      VmlLog<T>(end - begin, Y_data + begin, Y_data + begin);
    });
  }
}

#else

template <typename T>
void LogitMKLKernel(T eps, TensorIteratorBase* it) {
  TORCH_CHECK(false, "ATen not compiled with MKL");
}

#endif // AT_MKL_ENABLED

static void logit_kernel(TensorIteratorBase& iter, const Scalar& eps_scalar) {
  AT_DISPATCH_FLOATING_TYPES_AND(
      kBFloat16, iter.common_dtype(), "logit_cpu", [&]() {
        const scalar_t eps = eps_scalar.to<scalar_t>();
        if (at::hasMKL() && iter.is_contiguous()) {
          LogitMKLKernel<scalar_t>(eps, &iter);
        } else if (eps < scalar_t(0)) {
          const Vectorized<scalar_t> kOneVec(scalar_t(1));
          cpu_kernel_vec(
              iter,
              [](scalar_t x) {
                return x == scalar_t(1)
                    ? std::numeric_limits<scalar_t>::infinity()
                    : std::log(x / (scalar_t(1) - x));
              },
              [kOneVec](Vectorized<scalar_t> x_vec) {
                return (x_vec / (kOneVec - x_vec)).log();
              });
        } else {
          const scalar_t lo = eps;
          const scalar_t hi = scalar_t(1) - eps;
          const Vectorized<scalar_t> kOneVec(scalar_t(1));
          const Vectorized<scalar_t> lo_vec(lo);
          const Vectorized<scalar_t> hi_vec(hi);
          cpu_kernel_vec(
              iter,
              [lo, hi](scalar_t x) {
                x = x < lo ? lo : (x > hi ? hi : x);
                return x == scalar_t(1)
                    ? std::numeric_limits<scalar_t>::infinity()
                    : std::log(x / (scalar_t(1) - x));
              },
              [kOneVec, lo_vec, hi_vec](Vectorized<scalar_t> x_vec) {
                x_vec = vec::clamp(x_vec, lo_vec, hi_vec);
                return (x_vec / (kOneVec - x_vec)).log();
              });
        }
      });
}

#if !defined(C10_MOBILE)
#define _AT_DISPATCH_ABS_TYPES(TYPE, NAME, ...)                   \
        AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(                   \
            kHalf, kBFloat16, kFloat8_e5m2, kFloat8_e4m3fn,       \
            TYPE, NAME, __VA_ARGS__)
#else
#define _AT_DISPATCH_ABS_TYPES(TYPE, NAME, ...)          \
        AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(          \
            kHalf, kBFloat16,                            \
            TYPE, NAME, __VA_ARGS__)
#endif

static void abs_kernel(TensorIteratorBase& iter) {
  auto dtype = iter.dtype();
  if (dtype == kComplexHalf) {
    using scalar_t = c10::complex<Half>;
    using opmath_t = at::opmath_type<scalar_t>;
    cpu_kernel(iter, [=](scalar_t a) -> scalar_t { return abs_impl(opmath_t{a}); });
  } else {
    _AT_DISPATCH_ABS_TYPES(iter.dtype(), "abs_cpu", [&]() {
      cpu_kernel_vec(
          iter,
          [=](scalar_t a) -> scalar_t { return abs_impl(a); },
          [=](Vectorized<scalar_t> a) { return a.abs(); });
    });
  }
}

static void angle_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(kBFloat16, kHalf, iter.common_dtype(), "angle_cpu", [&]() {
    cpu_kernel_vec(
        iter,
        [=](scalar_t a) -> scalar_t { return angle_impl(a); },
        [=](Vectorized<scalar_t> a) { return a.angle(); });
  });
}

// NB: Ignores the negative bit on tensors
void conj_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_SWITCH(iter.common_dtype(), "conj_cpu",
    AT_DISPATCH_CASE_ALL_TYPES_AND3(kBool, kBFloat16, kHalf, [&] {
      // conj is a no-op for non-complex types
      direct_copy_kernel(iter);
    })
    AT_DISPATCH_CASE_COMPLEX_TYPES_AND(kComplexHalf, [&] {
      cpu_kernel_vec(
          iter,
          [=](scalar_t a) -> scalar_t { return conj_impl(a); },
          [=](Vectorized<scalar_t> a) { return a.conj(); });
    })
  );
}

static void bitwise_not_kernel(TensorIteratorBase& iter) {
  if (iter.dtype() == ScalarType::Bool) {
    // Boolean type does not work with ~ (bitwise NOT) in C++. bitwise_not wraps this operation for both Boolean and
    // integral types.
    cpu_kernel(
          iter,
          [](bool a) {
            return !a;
          });
  } else {
    AT_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "bitwise_not_cpu", [&]() {
      cpu_kernel_vec(
          iter,
          [](scalar_t a) -> scalar_t {
            return ~a;
          },
          [](Vectorized<scalar_t> a) -> Vectorized<scalar_t> {
            return ~a;
          });
    });
  }
}

static void frac_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, iter.dtype(), "frac_cpu", [&]() {
    cpu_kernel_vec(
        iter,
        [=](scalar_t a) -> scalar_t { return a - std::trunc(a); },
        [=](Vectorized<scalar_t> a) { return a.frac(); });
  });
}

static void logical_not_kernel(TensorIteratorBase& iter) {
  // NOTE: this implementation differs from the CUDA implementation which only does single dispatch
  // (to avoid expensive compilation) because CPU kernels don't handle dynamic_casting
  // (see needs_dynamic_casting).
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(kBool, kHalf, kBFloat16, iter.dtype(1), "logical_not_cpu", [&]() {
    using self_t = scalar_t;
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(kBool, kHalf, kBFloat16, iter.dtype(0), "logical_not_cpu", [&]() {
      cpu_kernel(iter, [](self_t a) -> scalar_t { return static_cast<scalar_t>(!a); });
    });
  });
}

void reciprocal_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(kBFloat16, kHalf, iter.common_dtype(), "reciprocal_cpu", [&]() {
    cpu_kernel_vec(
        iter,
        [=](scalar_t a) __ubsan_ignore_float_divide_by_zero__ -> scalar_t { return static_cast<scalar_t>(1.0) / a; },
        [=](Vectorized<scalar_t> a) { return a.reciprocal(); });
  });
}

// NB: Ignores the negative bit on tensors
void neg_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(kComplexHalf, kBFloat16, kHalf, iter.dtype(), "neg_cpu", [&]() {
    cpu_kernel_vec(
        iter,
        [=](scalar_t a) -> scalar_t { return -a; },
        [=](Vectorized<scalar_t> a) { return a.neg(); });
  });
}

static void sign_kernel(TensorIteratorBase& iter){
  if(iter.dtype() == ScalarType::Bool){
      cpu_kernel(iter, [=](bool x) -> bool { return x; });
  } else {
    AT_DISPATCH_ALL_TYPES_AND2(kBFloat16, ScalarType::Half, iter.dtype(), "sign_cpu", [&]() {
        auto zero_vec = Vectorized<scalar_t>(static_cast<scalar_t>(0));
        auto one_vec = Vectorized<scalar_t>(static_cast<scalar_t>(1));

        cpu_kernel_vec(
          iter,
          [=](scalar_t a) -> scalar_t { return (0 < a) - c10::is_negative(a); },
          [=](Vectorized<scalar_t> self_vec){

              // Comparison operators returns bitmask.
              auto left = Vectorized<scalar_t>::blendv(zero_vec, one_vec, zero_vec < self_vec);
              auto right = Vectorized<scalar_t>::blendv(zero_vec, one_vec, self_vec < zero_vec);

              return left - right;
          });
    });
  }
}

static void signbit_kernel(TensorIteratorBase& iter){
  // NOTE: signbit does not always support integral arguments.
  AT_DISPATCH_SWITCH(iter.input_dtype(), "signbit_cpu",
      AT_DISPATCH_CASE_INTEGRAL_TYPES([&] {
        cpu_kernel(iter, [](scalar_t a) -> bool { return c10::is_negative(a); });
      })
      AT_DISPATCH_CASE_FLOATING_TYPES_AND2(kBFloat16, ScalarType::Half, [&] {
        using opmath_t = at::opmath_type<scalar_t>;
        cpu_kernel(iter, [](scalar_t a) -> bool { return std::signbit(opmath_t{a}); });
      })
    );
}

static void sgn_kernel(TensorIteratorBase& iter) {
  auto dtype = iter.dtype();
  if (dtype == kComplexHalf) {
    using scalar_t = c10::complex<Half>;
    using opmath_t = at::opmath_type<scalar_t>;
    cpu_kernel(
        iter, [=](scalar_t a) -> scalar_t { return sgn_impl(opmath_t{a}); });
  } else {
    AT_DISPATCH_COMPLEX_TYPES(dtype, "sgn_cpu", [&]() {
      cpu_kernel_vec(
        iter,
        [=](scalar_t a) -> scalar_t { return sgn_impl(a); },
        [=](Vectorized<scalar_t> a) { return a.sgn(); });
    });
  }
}

static void sinc_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(kBFloat16, kHalf, iter.common_dtype(), "sinc_cpu", [&]() {
    cpu_kernel(
        iter,
        [=](scalar_t a) -> scalar_t {
          if (a == scalar_t(0)) {
            return scalar_t(1);
          } else {
            scalar_t product = c10::pi<scalar_t> * a;
            return std::sin(product) / product;
          }
        });
  });
}

static void sinh_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(kBFloat16, kHalf, iter.dtype(), "sinh_cpu", [&]() {
    cpu_kernel_vec(
        iter,
        [=](scalar_t a) -> scalar_t { return std::sinh(a); },
        [=](Vectorized<scalar_t> self_vec){return self_vec.sinh();});
  });
}

static void cosh_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(kBFloat16, kHalf, iter.dtype(), "cosh_cpu", [&]() {
    cpu_kernel_vec(
        iter,
        [=](scalar_t a) -> scalar_t { return std::cosh(a); },
        [=](Vectorized<scalar_t> self_vec){return self_vec.cosh();});
  });
}

static void acosh_kernel(TensorIteratorBase& iter) {
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(kBFloat16, kHalf, iter.dtype(), "acosh_cpu", [&]() {
      cpu_kernel_vec(
        iter,
        [=](scalar_t a) -> scalar_t { return std::acosh(a); },
        [=](Vectorized<scalar_t> self_vec){return self_vec.acosh();});
    });
}

static void asinh_kernel(TensorIteratorBase& iter) {
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(kBFloat16, kHalf, iter.dtype(), "asinh_cpu", [&]() {
      cpu_kernel(
        iter,
        [=](scalar_t a) -> scalar_t { return std::asinh(a); });
    });
}

static void atanh_kernel(TensorIteratorBase& iter) {
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(kBFloat16, kHalf, iter.dtype(), "atanh_cpu", [&]() {
      cpu_kernel(
        iter,
        [=](scalar_t a) -> scalar_t { return std::atanh(a); });
    });
}

static void digamma_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, iter.common_dtype(), "digamma", [&]() {
    cpu_kernel(
        iter,
        [=](scalar_t a) -> scalar_t { return calc_digamma(a); });
  });
}

static void trigamma_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, iter.dtype(), "trigamma", [&]() {
    cpu_kernel(
        iter,
        [=](scalar_t a) -> scalar_t { return trigamma(a); });
  });
}

static void exp2_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      kBFloat16, kHalf, iter.dtype(), "exp2", [&] {
    cpu_kernel_vec(
        iter,
        [](scalar_t a) -> scalar_t { return exp2_impl(a); },
        [](Vectorized<scalar_t> a) { return a.exp2(); });
  });
}

static void polygamma_kernel(TensorIteratorBase& iter, int64_t n) {
  if (n == 0) {
    digamma_kernel(iter);
  } else if (n == 1) {
    trigamma_kernel(iter);
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, iter.dtype(), "polygamma", [&]() {
      cpu_kernel(
          iter, [=](scalar_t a) -> scalar_t { return calc_polygamma(a, n); });
    });
  }
}

template<typename scalar_t>
inline scalar_t _nan_to_num_replace(scalar_t a, scalar_t nan_replacement, scalar_t pos_inf_replacement, scalar_t neg_inf_replacement) {
  return at::_isnan(a)
    ? nan_replacement
    : (a == std::numeric_limits<scalar_t>::infinity()
      ? pos_inf_replacement
      : (a == -std::numeric_limits<scalar_t>::infinity()
        ? neg_inf_replacement
        : a));
}

static void nan_to_num_kernel(
    TensorIteratorBase& iter,
    c10::optional<double> nan,
    c10::optional<double> pos_inf,
    c10::optional<double> neg_inf) {
  if (isComplexType(iter.dtype())) {
    AT_DISPATCH_COMPLEX_TYPES(iter.dtype(), "nan_to_num", [&]() {
      using value_t = scalar_t::value_type;
      value_t nan_replacement = static_cast<value_t>(nan.value_or(0.));
      value_t pos_inf_replacement = pos_inf.has_value()
          ? static_cast<value_t>(pos_inf.value())
          : std::numeric_limits<value_t>::max();
      value_t neg_inf_replacement = neg_inf.has_value()
          ? static_cast<value_t>(neg_inf.value())
          : std::numeric_limits<value_t>::lowest();

      cpu_kernel(iter, [=](scalar_t a) -> scalar_t {
        value_t res_real = _nan_to_num_replace(
          a.real(), nan_replacement, pos_inf_replacement, neg_inf_replacement);
        value_t res_imag = _nan_to_num_replace(
          a.imag(), nan_replacement, pos_inf_replacement, neg_inf_replacement);
        return scalar_t(res_real, res_imag);
      });
    });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, iter.dtype(), "nan_to_num", [&]() {
      scalar_t nan_replacement = static_cast<scalar_t>(nan.value_or(0.));
      scalar_t pos_inf_replacement = pos_inf.has_value()
          ? static_cast<scalar_t>(pos_inf.value())
          : std::numeric_limits<scalar_t>::max();
      scalar_t neg_inf_replacement = neg_inf.has_value()
          ? static_cast<scalar_t>(neg_inf.value())
          : std::numeric_limits<scalar_t>::lowest();

      cpu_kernel(iter, [=](scalar_t a) -> scalar_t {
        return _nan_to_num_replace(
          a, nan_replacement, pos_inf_replacement, neg_inf_replacement);
      });
    });
  }
}

static void kaiser_window_kernel(TensorIteratorBase& iter, int64_t window_length, double beta){
  AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, iter.dtype(), "kaiser_window_cpu", [&](){
    using opmath_t = at::opmath_type<scalar_t>;
    const opmath_t alpha = static_cast<opmath_t>((window_length - 1) / 2.0);
    const opmath_t beta_ = static_cast<opmath_t>(beta);
    cpu_kernel(iter, [=](scalar_t a){
        return calc_i0(beta_ * std::sqrt(1 - std::pow((static_cast<opmath_t>(a) - alpha) / alpha, static_cast<opmath_t>(2.0)))) / calc_i0(beta_);
    });
  });
}

void rsqrt_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(kBFloat16, kHalf, iter.common_dtype(), "rsqrt_cpu", [&] {
    cpu_kernel_vec(
        iter,
        [=](scalar_t a) __ubsan_ignore_float_divide_by_zero__ -> scalar_t {
          return (static_cast<scalar_t>(1)) / std::sqrt(a);
        },
        [=](Vectorized<scalar_t> a) { return a.rsqrt(); });
  });
}

static void entr_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      kBFloat16, kHalf, iter.common_dtype(), "entr_cpu", [&] {
        cpu_kernel(iter, [](scalar_t x) -> scalar_t {
          if (at::_isnan(x)) {
            return x;
          } else if (x > 0) {
            return -x * std::log(x);
          } else if (x == 0) {
            return static_cast<scalar_t>(0);
          }
          return static_cast<scalar_t>(-INFINITY);
        });
      });
}

static void frexp_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf,
    // The iter.dtype() here is the dtype of mantissa output.
    // It's a floating point type and must be the same as the input's dtype.
    iter.dtype(),
    "frexp_cpu", [&]() {
      cpu_kernel_multiple_outputs(
        iter,
        [](scalar_t a) -> std::tuple<scalar_t, int32_t> {
          int32_t exponent;
          scalar_t mantissa = std::frexp(a, &exponent);
          return std::tuple<scalar_t, int32_t>(mantissa, exponent);
        }
      );
  });
}

static void ndtri_kernel(TensorIteratorBase& iter) {
  TORCH_INTERNAL_ASSERT(iter.ntensors() == 2);
  AT_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "ndtri_cpu", [&]() {
        cpu_kernel(iter, [](scalar_t x) { return calc_ndtri(x); });
      });
}

static void log_ndtr_kernel(TensorIteratorBase& iter) {
  TORCH_INTERNAL_ASSERT(iter.ntensors() == 2);
  AT_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "log_ndtr_cpu", [&]() {
        cpu_kernel(iter, [](scalar_t x) { return calc_log_ndtr(x); });
      });
}

static void i0e_kernel(TensorIteratorBase& iter) {
  TORCH_INTERNAL_ASSERT(iter.ntensors() == 2);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      kBFloat16, kHalf, iter.common_dtype(), "i0e_cpu", [&]() {
        cpu_kernel_vec(
            iter,
            [](scalar_t x) { return calc_i0e(x); },
            [](Vectorized<scalar_t> x) { return x.i0e(); });
      });
}

static void i1_kernel(TensorIteratorBase& iter) {
  TORCH_INTERNAL_ASSERT(iter.ntensors() == 2);
  AT_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "i1_cpu", [&]() {
    cpu_kernel(iter, [](scalar_t x) { return calc_i1(x); });
  });
}

static void i1e_kernel(TensorIteratorBase& iter) {
  TORCH_INTERNAL_ASSERT(iter.ntensors() == 2);
  AT_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "i1e_cpu", [&]() {
    cpu_kernel(iter, [](scalar_t x) { return calc_i1e(x); });
  });
}

static void erfcx_kernel(TensorIteratorBase& iter){
  AT_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "erfcx_cpu", [&]() {
    cpu_kernel(
      iter,
      [](scalar_t a) -> scalar_t { return calc_erfcx(a); });
  });
}

static void round_decimals_kernel(TensorIteratorBase& iter, int64_t decimals) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      kBFloat16, kHalf, iter.dtype(), "round_cpu", [&]() {
        using opmath_t = at::opmath_type<scalar_t>;
        bool neg_flag = false;
        opmath_t ten_pow_decimals;
        if (decimals < 0) {
          decimals = -decimals;
          neg_flag = true;
        }
        ten_pow_decimals = static_cast<opmath_t>(std::pow(10, decimals));
        cpu_kernel(iter, [ten_pow_decimals, neg_flag](scalar_t a) -> scalar_t {
          return neg_flag ? std::nearbyint(static_cast<opmath_t>(a) / ten_pow_decimals) * ten_pow_decimals
                          : std::nearbyint(static_cast<opmath_t>(a) * ten_pow_decimals) / ten_pow_decimals;
        });
      });
}

static void bessel_j0_kernel(TensorIteratorBase& iterator) {
    TORCH_INTERNAL_ASSERT(iterator.ntensors() == 2);

    AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "bessel_j0_cpu", [&]() {
        cpu_kernel(iterator, [](scalar_t x) {
            return bessel_j0_forward(x);
        });
    });
} // bessel_j0_kernel(TensorIteratorBase& iterator)

static void bessel_j1_kernel(TensorIteratorBase& iterator) {
    TORCH_INTERNAL_ASSERT(iterator.ntensors() == 2);

    AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "bessel_j1_cpu", [&]() {
        cpu_kernel(iterator, [](scalar_t x) {
            return bessel_j1_forward(x);
        });
    });
} // bessel_j1_kernel(TensorIteratorBase& iterator)

static void bessel_y0_kernel(TensorIteratorBase& iterator) {
    TORCH_INTERNAL_ASSERT(iterator.ntensors() == 2);

    AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "bessel_y0_cpu", [&]() {
        cpu_kernel(iterator, [](scalar_t x) {
            return bessel_y0_forward(x);
        });
    });
} // bessel_y0_kernel(TensorIteratorBase& iterator)

static void bessel_y1_kernel(TensorIteratorBase& iterator) {
    TORCH_INTERNAL_ASSERT(iterator.ntensors() == 2);

    AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "bessel_y1_cpu", [&]() {
        cpu_kernel(iterator, [](scalar_t x) {
            return bessel_y1_forward(x);
        });
    });
} // bessel_y1_kernel(TensorIteratorBase& iterator)

static void modified_bessel_i0_kernel(TensorIteratorBase& iterator) {
    TORCH_INTERNAL_ASSERT(iterator.ntensors() == 2);

    AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "modified_bessel_i0_cpu", [&]() {
        cpu_kernel(iterator, [](scalar_t x) {
            return modified_bessel_i0_forward(x);
        });
    });
} // modified_bessel_i0_kernel(TensorIteratorBase& iterator)

static void modified_bessel_i1_kernel(TensorIteratorBase& iterator) {
    TORCH_INTERNAL_ASSERT(iterator.ntensors() == 2);

    AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "modified_bessel_i1_cpu", [&]() {
        cpu_kernel(iterator, [](scalar_t x) {
            return modified_bessel_i1_forward(x);
        });
    });
} // modified_bessel_i1_kernel(TensorIteratorBase& iterator)

static void modified_bessel_k0_kernel(TensorIteratorBase& iterator) {
    TORCH_INTERNAL_ASSERT(iterator.ntensors() == 2);

    AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "modified_bessel_k0_cpu", [&]() {
        cpu_kernel(iterator, [](scalar_t x) {
            return modified_bessel_k0_forward(x);
        });
    });
} // modified_bessel_k0_kernel(TensorIteratorBase& iterator)

static void modified_bessel_k1_kernel(TensorIteratorBase& iterator) {
    TORCH_INTERNAL_ASSERT(iterator.ntensors() == 2);

    AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "modified_bessel_k1_cpu", [&]() {
        cpu_kernel(iterator, [](scalar_t x) {
            return modified_bessel_k1_forward(x);
        });
    });
} // modified_bessel_k1_kernel(TensorIteratorBase& iterator)

// TODO: Disable cont. branch to test more risky code

#define IMPLEMENT_ITERATOR_LAMBDA(op)                                              \
          [&](char** data_, const int64_t* strides, int64_t n) {                   \
            scalar_t* out_data = reinterpret_cast<scalar_t*>(data_[0]);            \
            scalar_t* in_data = reinterpret_cast<scalar_t*>(data_[1]);             \
            int64_t out_stride = strides[0] / sizeof(scalar_t);                    \
            int64_t in_stride = strides[1] / sizeof(scalar_t);                     \
            if (out_stride == 1 && in_stride == 1) {                               \
              vml::v##op(out_data, in_data, n);                                    \
              return;                                                              \
            }                                                                      \
            static constexpr int64_t WIDTH = (8*1024) / sizeof(scalar_t);          \
            for (int64_t i = 0; i < n; i += WIDTH) {                               \
              scalar_t buffer[WIDTH];                                              \
              const int64_t width = std::min(WIDTH, n - i);                        \
              /* If either tensor is contiguous use it, otherwise copy into */     \
              /* a contiguous buffer so compute can still be vectorized */         \
              scalar_t * in_buffer = in_stride == 1 ? &in_data[i] : &buffer[0];    \
              scalar_t * out_buffer = out_stride == 1 ? &out_data[i] : &buffer[0]; \
              if (in_stride != 1)                                                  \
                for (const auto j : c10::irange(width))                            \
                  in_buffer[j] = in_data[in_stride * (i + j)];                     \
              vml::v##op(out_buffer, in_buffer, width);                            \
              if (out_stride != 1)                                                 \
                for (const auto j : c10::irange(width))                            \
                    out_data[out_stride * (i + j)] = out_buffer[j];                \
            }                                                                      \
          }

#define IMPLEMENT_FLOAT_KERNEL(op)                                                  \
  inline namespace CPU_CAPABILITY {                                                 \
  static void op##_kernel(TensorIteratorBase& iter) {                               \
    TORCH_INTERNAL_ASSERT(iter.ntensors() == 2);                                    \
    AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, iter.dtype(), #op "_vml_cpu", [&]() { \
      constexpr int64_t grain_size = 2048;                                          \
      iter.for_each(IMPLEMENT_ITERATOR_LAMBDA(op), grain_size);                     \
    });                                                                             \
    iter.cast_outputs();                                                            \
  }                                                                                 \
  }                                                                                 \
  REGISTER_DISPATCH(op##_stub, &CPU_CAPABILITY::op##_kernel)

#define IMPLEMENT_COMPLEX_KERNEL(op)                                                             \
  inline namespace CPU_CAPABILITY {                                                              \
  void op##_kernel(TensorIteratorBase& iter) {                                                   \
    TORCH_INTERNAL_ASSERT(iter.ntensors() == 2);                                                 \
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(kBFloat16, kHalf, iter.dtype(), #op "_vml_cpu", [&]() { \
        constexpr int64_t grain_size = 2048;                                                     \
        iter.for_each(IMPLEMENT_ITERATOR_LAMBDA(op), grain_size);                                \
    });                                                                                          \
    iter.cast_outputs();                                                                         \
  }                                                                                              \
  }                                                                                              \
  REGISTER_DISPATCH(op##_stub, &CPU_CAPABILITY::op##_kernel)

#define STATIC_IMPLEMENT_COMPLEX_KERNEL(op)                                                      \
  inline namespace CPU_CAPABILITY {                                                              \
  static void op##_kernel(TensorIteratorBase& iter) {                                            \
    TORCH_INTERNAL_ASSERT(iter.ntensors() == 2);                                                 \
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(kBFloat16, kHalf, iter.dtype(), #op "_vml_cpu", [&]() { \
        constexpr int64_t grain_size = 2048;                                                     \
        iter.for_each(IMPLEMENT_ITERATOR_LAMBDA(op), grain_size);                                \
    });                                                                                          \
    iter.cast_outputs();                                                                         \
  }                                                                                              \
  }                                                                                              \
  REGISTER_DISPATCH(op##_stub, &CPU_CAPABILITY::op##_kernel)

} // CPU_CAPABILITY namespace

REGISTER_DISPATCH(rsqrt_stub, &CPU_CAPABILITY::rsqrt_kernel);
REGISTER_DISPATCH(sigmoid_stub, &CPU_CAPABILITY::sigmoid_kernel);
REGISTER_DISPATCH(logit_stub, &CPU_CAPABILITY::logit_kernel);
REGISTER_DISPATCH(abs_stub, &CPU_CAPABILITY::abs_kernel);
REGISTER_DISPATCH(angle_stub, &CPU_CAPABILITY::angle_kernel);
REGISTER_DISPATCH(conj_physical_stub, &CPU_CAPABILITY::conj_kernel);
REGISTER_DISPATCH(exp2_stub, &CPU_CAPABILITY::exp2_kernel);
REGISTER_DISPATCH(bitwise_not_stub, &CPU_CAPABILITY::bitwise_not_kernel);
REGISTER_DISPATCH(logical_not_stub, &CPU_CAPABILITY::logical_not_kernel);
REGISTER_DISPATCH(frac_stub, &CPU_CAPABILITY::frac_kernel);
REGISTER_DISPATCH(reciprocal_stub, &CPU_CAPABILITY::reciprocal_kernel);
REGISTER_DISPATCH(nan_to_num_stub, &CPU_CAPABILITY::nan_to_num_kernel);
REGISTER_DISPATCH(neg_stub, &CPU_CAPABILITY::neg_kernel);
REGISTER_DISPATCH(sign_stub, &CPU_CAPABILITY::sign_kernel);
REGISTER_DISPATCH(signbit_stub, &CPU_CAPABILITY::signbit_kernel);
REGISTER_DISPATCH(sgn_stub, &CPU_CAPABILITY::sgn_kernel);
REGISTER_DISPATCH(sinc_stub, &CPU_CAPABILITY::sinc_kernel);
REGISTER_DISPATCH(sinh_stub, &CPU_CAPABILITY::sinh_kernel);
REGISTER_DISPATCH(cosh_stub, &CPU_CAPABILITY::cosh_kernel);
REGISTER_DISPATCH(acosh_stub, &CPU_CAPABILITY::acosh_kernel);
REGISTER_DISPATCH(asinh_stub, &CPU_CAPABILITY::asinh_kernel);
REGISTER_DISPATCH(atanh_stub, &CPU_CAPABILITY::atanh_kernel);
REGISTER_DISPATCH(digamma_stub, &CPU_CAPABILITY::digamma_kernel);
REGISTER_DISPATCH(trigamma_stub, &CPU_CAPABILITY::trigamma_kernel);
REGISTER_DISPATCH(polygamma_stub, &CPU_CAPABILITY::polygamma_kernel);
REGISTER_DISPATCH(kaiser_window_stub, &CPU_CAPABILITY::kaiser_window_kernel);
REGISTER_DISPATCH(special_entr_stub, &CPU_CAPABILITY::entr_kernel);
REGISTER_DISPATCH(frexp_stub, &CPU_CAPABILITY::frexp_kernel);
REGISTER_DISPATCH(special_i0e_stub, &CPU_CAPABILITY::i0e_kernel);
REGISTER_DISPATCH(special_ndtri_stub, &CPU_CAPABILITY::ndtri_kernel);
REGISTER_DISPATCH(special_log_ndtr_stub, &CPU_CAPABILITY::log_ndtr_kernel);
REGISTER_DISPATCH(special_i1_stub, &CPU_CAPABILITY::i1_kernel);
REGISTER_DISPATCH(special_i1e_stub, &CPU_CAPABILITY::i1e_kernel);
REGISTER_DISPATCH(special_erfcx_stub, &CPU_CAPABILITY::erfcx_kernel);
REGISTER_DISPATCH(round_decimals_stub, &CPU_CAPABILITY::round_decimals_kernel);
REGISTER_DISPATCH(special_bessel_j0_stub, &CPU_CAPABILITY::bessel_j0_kernel);
REGISTER_DISPATCH(special_bessel_j1_stub, &CPU_CAPABILITY::bessel_j1_kernel);
REGISTER_DISPATCH(special_bessel_y0_stub, &CPU_CAPABILITY::bessel_y0_kernel);
REGISTER_DISPATCH(special_bessel_y1_stub, &CPU_CAPABILITY::bessel_y1_kernel);
REGISTER_DISPATCH(special_modified_bessel_i0_stub, &CPU_CAPABILITY::modified_bessel_i0_kernel);
REGISTER_DISPATCH(special_modified_bessel_i1_stub, &CPU_CAPABILITY::modified_bessel_i1_kernel);
REGISTER_DISPATCH(special_modified_bessel_k0_stub, &CPU_CAPABILITY::modified_bessel_k0_kernel);
REGISTER_DISPATCH(special_modified_bessel_k1_stub, &CPU_CAPABILITY::modified_bessel_k1_kernel);

STATIC_IMPLEMENT_COMPLEX_KERNEL(acos)
STATIC_IMPLEMENT_COMPLEX_KERNEL(asin)
STATIC_IMPLEMENT_COMPLEX_KERNEL(atan)
IMPLEMENT_FLOAT_KERNEL(ceil)
STATIC_IMPLEMENT_COMPLEX_KERNEL(cos)
IMPLEMENT_FLOAT_KERNEL(erf)
IMPLEMENT_FLOAT_KERNEL(erfc)
IMPLEMENT_FLOAT_KERNEL(erfinv)
STATIC_IMPLEMENT_COMPLEX_KERNEL(exp)
STATIC_IMPLEMENT_COMPLEX_KERNEL(expm1)
IMPLEMENT_FLOAT_KERNEL(floor)
STATIC_IMPLEMENT_COMPLEX_KERNEL(log)
STATIC_IMPLEMENT_COMPLEX_KERNEL(log10)
STATIC_IMPLEMENT_COMPLEX_KERNEL(log1p)
STATIC_IMPLEMENT_COMPLEX_KERNEL(log2)
IMPLEMENT_FLOAT_KERNEL(i0)
IMPLEMENT_FLOAT_KERNEL(round)
STATIC_IMPLEMENT_COMPLEX_KERNEL(sin)
IMPLEMENT_COMPLEX_KERNEL(sqrt)
STATIC_IMPLEMENT_COMPLEX_KERNEL(tan)
STATIC_IMPLEMENT_COMPLEX_KERNEL(tanh)
IMPLEMENT_FLOAT_KERNEL(trunc)
IMPLEMENT_FLOAT_KERNEL(lgamma)

} // namespace at::native
