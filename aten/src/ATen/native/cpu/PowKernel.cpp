#include <cmath>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec256/vec256.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/Pow.h>
#include <ATen/native/cpu/Loops.h>

namespace at { namespace native {

namespace CPU_CAPABILITY {

void pow_tensor_tensor_kernel(TensorIteratorBase& iter) {
  const auto dtype = iter.common_dtype();
  if (isFloatingType(dtype) || isComplexType(dtype)) {
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(kHalf, kBFloat16, dtype, "pow", [&]() {

      using Vec = Vec256<scalar_t>;
      cpu_kernel_vec(iter,
        [=](scalar_t base, scalar_t exp) -> scalar_t {
          return std::pow(base, exp);
        },
        [&](Vec base, Vec exp) -> Vec {
          return base.pow(exp);
        }
      );
    });
  } else {
    AT_DISPATCH_INTEGRAL_TYPES(dtype, "pow", [&]() {
      cpu_kernel(iter,
        [=](scalar_t base, scalar_t exp) -> scalar_t {
          return native::powi(base, exp);
        }
      );
    });
  }
}

// The source-code of kernels for float, double and complex types is similar,
// barring a small distinction - even if the output dtype is float, a double
// exponent can be used. But Complex types' computation doesn't allow standard
// & double-precision to be mixed, since std::pow takes either complex64 inputs,
// or complex128 inputs, but not both. So, in order to provide a common path for
// float, double & complex types, template parameter cast_scalar_t is being used
// to resolve the aforementioned distinction. This approach also allows BFloat16
// to use this common-path. Half cannot currently use it, as AVX2 support for
// sqrt & rsqrt doesn't currently exist for it.
template <typename scalar_t, typename cast_scalar_t, typename exp_scalar_t>
void pow_tensor_scalar_optimized_kernel(TensorIteratorBase& iter, const exp_scalar_t exp) {
  using Vec = Vec256<scalar_t>;
  // .5 (sqrt), -.5 (rsqrt) and -1 (reciprocal) specializations are handled
  // in pow_tensor_scalar_kernel
  if (exp == 2.0) {
    cpu_kernel_vec(iter,
        [](scalar_t base) -> scalar_t {
          return base * base;
        },
        [](Vec base) -> Vec { return base * base; }
    );
  } else if (exp == 3.0) {
    cpu_kernel_vec(iter,
        [](scalar_t base) -> scalar_t {
          return base * base * base;
        },
        [](Vec base) -> Vec { return base * base * base; }
    );
  } else if (exp == -2.0) {
    cpu_kernel_vec(iter,
        [](scalar_t base) -> scalar_t {
          return static_cast<cast_scalar_t>(1.0) / (base * base); },
        [](Vec base) -> Vec { return (base * base).reciprocal(); }
    );
  } else {
    cpu_kernel_vec(iter,
        [=](scalar_t base) -> scalar_t {
          return std::pow(base, static_cast<cast_scalar_t>(exp));
        },
        [=](Vec base) -> Vec {
          return base.pow(static_cast<cast_scalar_t>(exp));
        }
    );
  }
}

// Forward declare some unary ops
void reciprocal_kernel(TensorIteratorBase& iter);
void rsqrt_kernel(TensorIteratorBase& iter);
void sqrt_kernel(TensorIteratorBase& iter);

void pow_tensor_scalar_kernel(
    TensorIteratorBase& iter,
    const Scalar& exp_scalar) {
  // prevent multiple calls to iter.common_dtype()
  const auto dtype = iter.common_dtype();

  if (dtype == ScalarType::Float || dtype == ScalarType::Double ||
      dtype == kBFloat16 || isComplexType(dtype)) {
    // Dispatch to fast specialization for sqrt, rsqrt and reciprocal
    if (exp_scalar.equal(.5)) {
      return sqrt_kernel(iter);
    } else if (exp_scalar.equal(-0.5)) {
      return rsqrt_kernel(iter);
    } else if (exp_scalar.equal(-1.0)) {
      return reciprocal_kernel(iter);
    }
  }

  if (dtype == ScalarType::Float || dtype == ScalarType::Double) {
    AT_DISPATCH_FLOATING_TYPES(dtype, "pow", [&]() {
      pow_tensor_scalar_optimized_kernel<scalar_t, double>(
          iter, exp_scalar.to<double>());
    });
  } else if (isComplexType(dtype)) {
    AT_DISPATCH_COMPLEX_TYPES(dtype, "pow", [&]() {
      pow_tensor_scalar_optimized_kernel<scalar_t, scalar_t>(
          iter, exp_scalar.to<c10::complex<double>>());
    });
  } else if (dtype == ScalarType::Half) {
    [&]() {
      using scalar_t =
          decltype(c10::impl::ScalarTypeToCPPType<ScalarType::Half>::t);
      const auto exp = exp_scalar.to<scalar_t>();
      using Vec = Vec256<scalar_t>;
      cpu_kernel_vec(iter,
          [=](scalar_t base) -> scalar_t {
            return std::pow(base, exp);
          },
          [=](Vec base) -> Vec { return base.pow(exp); }
      );
    }();
  } else if (dtype == ScalarType::BFloat16) {
      AT_DISPATCH_FLOATING_TYPES_AND(kBFloat16, dtype, "pow", [&]() {
        pow_tensor_scalar_optimized_kernel<scalar_t, scalar_t>(
            iter, exp_scalar.to<scalar_t>());
      });
  } else {
    AT_DISPATCH_INTEGRAL_TYPES(dtype, "pow", [&]() {
      const scalar_t exp = exp_scalar.to<scalar_t>();
      cpu_kernel(iter, [=](scalar_t base) -> scalar_t {
        return native::powi(base, exp);
      });
    });
  }
}

} // anonymous namespace

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_DISPATCH(pow_tensor_tensor_stub, &CPU_CAPABILITY::pow_tensor_tensor_kernel);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_DISPATCH(pow_tensor_scalar_stub, &CPU_CAPABILITY::pow_tensor_scalar_kernel);

}} // namespace at::native
