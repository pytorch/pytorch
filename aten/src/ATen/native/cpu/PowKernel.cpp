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
  if (isFloatingType(iter.dtype()) || isComplexType(iter.dtype())) {
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(iter.dtype(), "pow", [&]() {
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
    AT_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "pow", [&]() {
      cpu_kernel(iter,
        [=](scalar_t base, scalar_t exp) -> scalar_t {
          return native::powi(base, exp);
        }
      );
    });
  }
}

void pow_tensor_scalar_kernel(TensorIteratorBase& iter, const Scalar& exp_scalar) {
  if (isFloatingType(iter.dtype())) {
    const auto exp = exp_scalar.to<double>();
    // Floating types allow AVX2 vector optimizations for pow/sqrt/rsqrt:
    AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "pow", [&]() {
      using Vec = Vec256<scalar_t>;
      if (exp == 0.5) {
        cpu_kernel_vec(iter,
          [](scalar_t base) -> scalar_t {
            return std::sqrt(base);
          },
          [](Vec base) -> Vec { return base.sqrt(); }
        );
      } else if (exp == 2) {
        cpu_kernel_vec(iter,
          [](scalar_t base) -> scalar_t {
            return base * base;
          },
          [](Vec base) -> Vec { return base * base; }
        );
      } else if (exp == 3) {
        cpu_kernel_vec(iter,
          [](scalar_t base) -> scalar_t {
            return base * base * base;
          },
          [](Vec base) -> Vec { return base * base * base; }
        );
      } else if (exp == -0.5) {
        cpu_kernel_vec(iter,
          [](scalar_t base) __ubsan_ignore_float_divide_by_zero__ -> scalar_t {
            return 1.0 / std::sqrt(base);
          },
          [](Vec base) -> Vec { return base.rsqrt(); }
        );
      } else if (exp == -1) {
        cpu_kernel_vec(iter,
          [](scalar_t base) -> scalar_t {
            return 1.0 / base;
          },
          [](Vec base) -> Vec { return base.reciprocal(); }
        );
      } else if (exp == -2) {
        cpu_kernel_vec(iter,
          [](scalar_t base) -> scalar_t {
            return 1.0 / (base * base);
          },
          [](Vec base) -> Vec { return (base * base).reciprocal(); }
        );
      } else {
        cpu_kernel_vec(iter,
          [=](scalar_t base) -> scalar_t {
            return std::pow(base, exp);
          },
          [=](Vec base) -> Vec { return base.pow(exp); }
        );
      }
    });
  } else if (isComplexType(iter.dtype())) {
    const auto exp = exp_scalar.to<c10::complex<double>>();
    // Floating types allow AVX2 vector optimizations for pow/sqrt/rsqrt:
    AT_DISPATCH_COMPLEX_TYPES(iter.dtype(), "pow", [&]() {
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
          [](scalar_t base) -> scalar_t {
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
            return scalar_t(1.0) / (base * base);
          },
          [](Vec base) -> Vec { return (base * base).reciprocal(); }
        );
      } else {
        cpu_kernel_vec(iter,
          [=](scalar_t base) -> scalar_t {
            return std::pow(base, scalar_t(exp));
          },
          [=](Vec base) -> Vec { return base.pow(scalar_t(exp)); } // std::pow cannot accept mixed complex data types.
        );
      }
    });
  } else {
    AT_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "pow", [&]() {
      const scalar_t exp = exp_scalar.to<scalar_t>();
      cpu_kernel(iter,
        [=](scalar_t base) -> scalar_t {
          return native::powi(base, exp);
        });
    });
  }
}

} // anonymous namespace

REGISTER_DISPATCH(pow_tensor_tensor_stub, &pow_tensor_tensor_kernel);
REGISTER_DISPATCH(pow_tensor_scalar_stub, &pow_tensor_scalar_kernel);

}} // namespace at::native
