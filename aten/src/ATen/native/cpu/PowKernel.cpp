#include <cmath>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec256/vec256.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/Pow.h>
#include <ATen/native/cpu/Loops.h>

namespace at { namespace native {

namespace {

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

// Floating and complex types (minus Half) allow for optimized kernels
template <typename scalar_t, typename cast_scalar_t, typename exp_scalar_t>
void pow_tensor_scalar_optimized_kernel(TensorIteratorBase& iter, const exp_scalar_t exp) {
  using Vec = Vec256<scalar_t>;
  if (exp == 0.5) {
    cpu_kernel_vec(iter,
        [](scalar_t base) -> scalar_t {
        	return std::sqrt(base);
        },
        [](Vec base) -> Vec { return base.sqrt(); }
    );
  } else if (exp == 2.0) {
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
  } else if (exp == -0.5) {
    cpu_kernel_vec(iter,
        [](scalar_t base) __ubsan_ignore_float_divide_by_zero__ -> scalar_t {
          return scalar_t(1.0) / std::sqrt(base);
        },
        [](Vec base) -> Vec { return base.rsqrt(); }
    );
  } else if (exp == -1.0) {
    cpu_kernel_vec(iter,
        [](scalar_t base) -> scalar_t {
        	return scalar_t(1.0) / base;
        },
        [](Vec base) -> Vec { return base.reciprocal(); }
    );
  } else if (exp == -2.0) {
    cpu_kernel_vec(iter,
        [](scalar_t base) -> scalar_t {
        	return scalar_t(1.0) / (base * base); },
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

void pow_tensor_scalar_kernel(
    TensorIteratorBase& iter,
    const Scalar& exp_scalar) {
  // prevent multiple calls to iter.common_dtype()
  const auto dtype = iter.common_dtype();
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
      using scalar_t =
          decltype(c10::impl::ScalarTypeToCPPType<ScalarType::BFloat16>::t);
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

REGISTER_DISPATCH(pow_tensor_tensor_stub, &pow_tensor_tensor_kernel);
REGISTER_DISPATCH(pow_tensor_scalar_stub, &pow_tensor_scalar_kernel);

}} // namespace at::native
