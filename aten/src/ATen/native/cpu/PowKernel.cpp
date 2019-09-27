#include <cmath>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec256/vec256.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/Pow.h>
#include <ATen/native/cpu/Loops.h>

namespace at { namespace native {

namespace {

void pow_tensor_tensor_kernel(TensorIterator& iter) {
  if (isFloatingType(iter.dtype())) {
    AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "pow", [&]() {
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
          return std::pow(base, exp);
        }
      );
    });
  }
}

void pow_tensor_scalar_kernel(TensorIterator& iter, Scalar exp_scalar) {
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
          [](scalar_t base) -> scalar_t {
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
  } else {
    // Integral types do not allow AVX2 vector optimizations for pow/sqrt/rsqrt.
    // Trying to implement pow/sqrt/rsqrt as loop in vec256_int.h does not allow
    // powering integral tensor to float exponent. That's why we need this code
    // duplication:

    if (exp_scalar.isIntegral(true) && exp_scalar.to<int64_t>() >= 0) {
      // Specifically deal with an integer to the power of a positive integer for better efficiency.
      const auto exp = exp_scalar.to<int64_t>();

      AT_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "pow", [&]() {
        switch (exp) {
          case 2:
            cpu_kernel(iter,
              [](scalar_t base) -> scalar_t {
                return base * base;
              }
            );
            break;
          case 3:
            cpu_kernel(iter,
              [](scalar_t base) -> scalar_t {
                return base * base * base;
              }
            );
            break;
          default:
            cpu_kernel(iter,
              [=](scalar_t base) -> scalar_t {
                return std::pow(base, exp);
              }
            );
        }
      });
    } else {
      // Casting exponent to double(not tensor.dtype) allows powering integral
      // tensors to float exponent e.g. tensor([4]).pow(0.5) will be tensor([2])
      const auto exp = exp_scalar.to<double>();
      AT_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "pow", [&]() {
        if (exp == 0.5) {
          cpu_kernel(iter,
            [](scalar_t base) -> scalar_t {
              return std::sqrt(static_cast<long double>(base));
            }
          );
        } else if (exp == -0.5) {
          cpu_kernel(iter,
            [](scalar_t base) -> scalar_t {
              return 1.0 / std::sqrt(static_cast<long double>(base));
            }
          );
        } else if (exp == -1) {
          cpu_kernel(iter,
            [](scalar_t base) -> scalar_t {
              return 1.0 / static_cast<long double>(base);
            }
          );
        } else if (exp == -2) {
          cpu_kernel(iter,
            [](scalar_t base) -> scalar_t {
              return 1.0 / (base * base);
            }
          );
        } else {
          cpu_kernel(iter,
            [=](scalar_t base) -> scalar_t {
              return std::pow(static_cast<long double>(base), exp);
            }
          );
        }
      });
    }
  }
}

} // anonymous namespace

REGISTER_DISPATCH(pow_tensor_tensor_stub, &pow_tensor_tensor_kernel);
REGISTER_DISPATCH(pow_tensor_scalar_stub, &pow_tensor_scalar_kernel);

}} // namespace at::native
