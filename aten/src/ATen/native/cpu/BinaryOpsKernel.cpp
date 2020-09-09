#include <ATen/native/BinaryOps.h>

#include <cmath>
#include <iostream>

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec256/functional.h>
#include <ATen/cpu/vec256/vec256.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>
#include <ATen/native/Math.h>
#include <c10/macros/Macros.h>

namespace at {
namespace native {

namespace {

using namespace vec256;

// Note: Undefined behavior when performing addition is intentionally
// ignored.
void add_kernel(TensorIterator& iter, Scalar alpha_scalar) {
  if (iter.dtype() == ScalarType::Bool) {
      using scalar_t = bool;
      auto alpha = alpha_scalar.to<scalar_t>();
      cpu_kernel(iter,
        [=](scalar_t a, scalar_t b) __ubsan_ignore_undefined__ -> scalar_t { return a + alpha * b; });
  } else {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(kBFloat16, kHalf, iter.dtype(), "add_cpu/sub_cpu", [&]() {
      auto alpha = alpha_scalar.to<scalar_t>();
      auto alpha_vec = Vec256<scalar_t>(alpha);
      cpu_kernel_vec(iter,
        [=](scalar_t a, scalar_t b) __ubsan_ignore_undefined__ -> scalar_t { return a + alpha * b; },
        [=](Vec256<scalar_t> a, Vec256<scalar_t> b) __ubsan_ignore_undefined__ {
          return vec256::fmadd(b, alpha_vec, a);
        });
      });
  }
}

void add_clamp_kernel(TensorIterator& iter, Scalar alpha_scalar, Scalar min_val, Scalar max_val) {
  AT_DISPATCH_ALL_TYPES(iter.dtype(), "add_clamp_cpu", [&]() {
    auto alpha = alpha_scalar.to<scalar_t>();
    auto alpha_vec = Vec256<scalar_t>(alpha);
    auto min_scalar = min_val.to<scalar_t>();
    auto min_vec = Vec256<scalar_t>(min_scalar);
    auto max_scalar = max_val.to<scalar_t>();
    auto max_vec = Vec256<scalar_t>(max_scalar);
    cpu_kernel_vec(iter,
      [=](scalar_t a, scalar_t b) __ubsan_ignore_undefined__ -> scalar_t {
        return std::min(max_scalar, std::max(min_scalar, a + alpha * b));
      },
      [=](Vec256<scalar_t> a, Vec256<scalar_t> b) __ubsan_ignore_undefined__ {
        auto add_clamp_res = vec256::fmadd(b, alpha_vec, a);
        add_clamp_res = vec256::clamp_min(add_clamp_res, min_vec);
        add_clamp_res = vec256::clamp_max(add_clamp_res, max_vec);
        return add_clamp_res;
      });
    });
}

void atan2_kernel(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "atan2_cpu", [&]() {
    cpu_kernel_vec(iter, [=](scalar_t a, scalar_t b) -> scalar_t {
    return std::atan2(a, b);
  },
    [=](Vec256<scalar_t> a, Vec256<scalar_t> b) {
      return a.atan2(b);
    });
  });
}

// Note: Undefined behavior when performing subtraction is intentionally
// ignored.
void sub_kernel(TensorIterator& iter, Scalar alpha_scalar) __ubsan_ignore_undefined__ {
  add_kernel(iter, -alpha_scalar);
}

void mul_kernel(TensorIterator& iter) {
  if (iter.dtype() == ScalarType::Bool) {
    cpu_kernel(iter, [=](bool a, bool b) -> bool { return a && b; });
  } else {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(kBFloat16, kHalf, iter.dtype(), "mul_cpu", [&]() {
      cpu_kernel_vec(iter,
        [=](scalar_t a, scalar_t b) -> scalar_t { return a * b; },
        [=](Vec256<scalar_t> a, Vec256<scalar_t> b) {
          return a * b;
        });
    });
  }
}

void div_kernel(TensorIterator& iter) {
  if (isIntegralType(iter.dtype(), /*includeBool*/ false)) {
    // There's no SIMD integer division, so don't try to vectorize it.
    // TODO: if the divisor is a scalar, rewrite as multiplication by a constant.
    AT_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "div_cpu", [&]() {
      cpu_kernel(iter, [](scalar_t a, scalar_t b) -> scalar_t {
        TORCH_CHECK(b != 0, "ZeroDivisionError");
        return a / b;
      });
    });
  } else {
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(kBFloat16, kHalf, iter.dtype(), "div_cpu", [&]() {
      cpu_kernel_vec(iter,
        [](scalar_t a, scalar_t b) __ubsan_ignore_float_divide_by_zero__ -> scalar_t {
          return a / b;
        },
        [](Vec256<scalar_t> a, Vec256<scalar_t> b) {
          return a / b;
        });
    });
  }
}

void remainder_kernel(TensorIterator& iter) {
  if (isIntegralType(iter.dtype(), /*includeBool*/ false)) {
    AT_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "remainder_cpu", [&]() {
      cpu_kernel(iter, [](scalar_t a, scalar_t b) -> scalar_t {
        TORCH_CHECK(b != 0, "ZeroDivisionError");
        scalar_t r = a % b;
        if ((r != 0) && ((r < 0) != (b < 0))) {
          r += b;
        }
        return r;
      });
    });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(iter.dtype(), "remainder_cpu", [&]() {
      cpu_kernel_vec(iter,
        [=](scalar_t a, scalar_t b) __ubsan_ignore_float_divide_by_zero__ -> scalar_t {
          scalar_t mod = std::fmod(a, b);
          if ((mod != 0) && ((b < 0) != (mod < 0))) mod += b;
          return mod;
        },
        [=](Vec256<scalar_t> a, Vec256<scalar_t> b) {
          auto mod = a.fmod(b);
          const auto zero = Vec256<scalar_t>(0);
          auto mask = (mod != zero) & ((b < zero) ^ (mod < zero));
          return Vec256<scalar_t>::blendv(mod, mod + b, mask);
        });
    });
  }
}

void bitwise_and_kernel(TensorIterator& iter) {
  if (iter.dtype() == ScalarType::Bool) {
    cpu_kernel(
        iter,
        [](bool a, bool b) {
          return a && b;
        });
  } else {
    AT_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "bitwise_and_cpu", [&]() {
      cpu_kernel_vec(
          iter,
          [](scalar_t a, scalar_t b) -> scalar_t {
            return a & b;
          },
          [](Vec256<scalar_t> a, Vec256<scalar_t> b) {
            return a & b;
          });
    });
  }
}

void bitwise_or_kernel(TensorIterator& iter) {
  if (iter.dtype() == ScalarType::Bool) {
    cpu_kernel(
        iter,
        [](bool a, bool b) {
          return a || b;
        });
  } else {
    AT_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "bitwise_or_cpu", [&]() {
      cpu_kernel_vec(
          iter,
          [](scalar_t a, scalar_t b) -> scalar_t {
            return a | b;
          },
          [](Vec256<scalar_t> a, Vec256<scalar_t> b) {
            return a | b;
          });
    });
  }
}

void bitwise_xor_kernel(TensorIterator& iter) {
  if (iter.dtype() == ScalarType::Bool) {
    // Boolean type does not work with ^ (bitwise XOR) in C++. bitwise_xor wraps this operation for both Boolean and
    // integral types.
    cpu_kernel(
          iter,
          [](bool a, bool b) {
            return a != b;
          });
  } else {
    AT_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "bitwise_xor_cpu", [&]() {
      cpu_kernel_vec(
          iter,
          [](scalar_t a, scalar_t b) -> scalar_t {
            return a ^ b;
          },
          [](Vec256<scalar_t> a, Vec256<scalar_t> b) {
            return a ^ b;
          });
    });
  }
}

void lshift_kernel(TensorIterator& iter) {
  if (iter.dtype() == ScalarType::Float || iter.dtype() == ScalarType::Double) {
    AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "lshift_cpu", [&]() {
      auto base_vec = Vec256<scalar_t>((scalar_t)(2));
      cpu_kernel_vec(
        iter,
        [=](scalar_t a, scalar_t b) -> scalar_t {
          return a * std::pow((scalar_t)(2), b);
        },
        [=](Vec256<scalar_t> a, Vec256<scalar_t> b) {
          return a * base_vec.pow(b);
      });
    });
  } else {
    AT_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "lshift_cpu", [&]() {
      cpu_kernel(iter,
        [](scalar_t a, scalar_t b) -> scalar_t {
          return static_cast<std::make_unsigned_t<scalar_t>>(a) << b;
      });
    });
  }
}

void logical_and_kernel(TensorIterator& iter) {
  // We use if-else here specifically for bool instead of using iter.common_dtype() like the CUDA implementation because
  // common_dtype() is unavailable for bfloat16.
  if (iter.dtype() == ScalarType::Bool) {
    AT_DISPATCH_ALL_TYPES_AND3(kBool, kBFloat16, kHalf, iter.input_dtype(), "logical_and_cpu", [&]() {
      cpu_kernel(iter,
        [](scalar_t a, scalar_t b) -> bool {
          return a && b;
        });
    });
  } else {
    AT_DISPATCH_ALL_TYPES_AND2(kBFloat16, kHalf, iter.dtype(), "logical_and_cpu", [&]() {
      cpu_kernel(iter,
        [](scalar_t a, scalar_t b) -> scalar_t {
          return static_cast<scalar_t>(a && b);
        });
    });
  }
}

void logical_or_kernel(TensorIterator& iter) {
  // We use if-else here specifically for bool instead of using iter.common_dtype() like the CUDA implementation because
  // common_dtype() is unavailable for bfloat16.
  if (iter.dtype() == ScalarType::Bool) {
    AT_DISPATCH_ALL_TYPES_AND3(kBool, kBFloat16, kHalf, iter.input_dtype(), "logical_or_cpu", [&]() {
      cpu_kernel(iter,
        [](scalar_t a, scalar_t b) -> bool {
          return a || b;
        });
    });
  } else {
    AT_DISPATCH_ALL_TYPES_AND3(kBool, kBFloat16, kHalf, iter.dtype(), "logical_or_cpu", [&]() {
      cpu_kernel(iter,
        [](scalar_t a, scalar_t b) -> scalar_t {
          return static_cast<scalar_t>(a || b);
        });
      });
  }
}

void logical_xor_kernel(TensorIterator& iter) {
  // We use if-else here specifically for bool instead of using iter.common_dtype() like the CUDA implementation because
  // common_dtype() is unavailable for bfloat16.
  if (iter.dtype() == ScalarType::Bool) {
    AT_DISPATCH_ALL_TYPES_AND3(kBool, kBFloat16, kHalf, iter.input_dtype(), "logical_xor_cpu", [&]() {
      cpu_kernel(iter,
        [](scalar_t a, scalar_t b) -> bool {
          return bool(a) != bool(b);
        });
    });
  } else {
    AT_DISPATCH_ALL_TYPES_AND2(kBFloat16, kHalf, iter.dtype(), "logical_xor_cpu", [&]() {
      cpu_kernel(iter,
        [](scalar_t a, scalar_t b) -> scalar_t {
          return static_cast<scalar_t>(bool(a) != bool(b));
        });
    });
  }
}

void rshift_kernel(TensorIterator& iter) {
  if (iter.dtype() == ScalarType::Float || iter.dtype() == ScalarType::Double) {
    AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "rshift_cpu", [&]() {
      auto base_vec = Vec256<scalar_t>((scalar_t)(2));
      cpu_kernel_vec(
        iter,
        [=](scalar_t a, scalar_t b) -> scalar_t {
          return a / std::pow((scalar_t)(2), b);
        },
        [=](Vec256<scalar_t> a, Vec256<scalar_t> b) {
          return a / base_vec.pow(b);
      });
    });
  } else {
    AT_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "rshift_cpu", [&]() {
      cpu_kernel(iter,
        [](scalar_t a, scalar_t b) -> scalar_t {
          return a >> b;
      });
    });
  }
}

void lt_kernel(TensorIterator& iter) {
  if (iter.dtype() == ScalarType::Bool) {
    AT_DISPATCH_ALL_TYPES_AND3(kBool, kBFloat16, kHalf, iter.input_dtype(), "lt_cpu", [&]() {
      cpu_kernel(iter,
       [](scalar_t a, scalar_t b) -> bool {
         return a < b;
       });
    });
  } else {
    AT_DISPATCH_ALL_TYPES_AND2(kBFloat16, kHalf, iter.dtype(), "lt_cpu", [&]() {
      cpu_kernel_vec(
        iter,
        [](scalar_t a, scalar_t b) -> scalar_t {
          return a < b;
        },
        [](Vec256<scalar_t> a, Vec256<scalar_t> b) -> Vec256<scalar_t> {
          return a.lt(b);
        });
      });
  }
}

void le_kernel(TensorIterator& iter) {
  if (iter.dtype() == ScalarType::Bool) {
    AT_DISPATCH_ALL_TYPES_AND3(kBool, kBFloat16, kHalf, iter.input_dtype(), "le_cpu", [&]() {
      cpu_kernel(iter,
       [](scalar_t a, scalar_t b) -> bool {
         return a <= b;
       });
    });
  } else {
    AT_DISPATCH_ALL_TYPES_AND2(kBFloat16, kHalf, iter.dtype(), "le_cpu", [&]() {
      cpu_kernel_vec(
        iter,
        [](scalar_t a, scalar_t b) -> scalar_t {
          return a <= b;
        },
        [](Vec256<scalar_t> a, Vec256<scalar_t> b) -> Vec256<scalar_t> {
          return a.le(b);
        });
      });
  }
}

void gt_kernel(TensorIterator& iter) {
  if (iter.dtype() == ScalarType::Bool) {
    AT_DISPATCH_ALL_TYPES_AND3(kBool, kBFloat16, kHalf, iter.input_dtype(), "gt_cpu", [&]() {
      cpu_kernel(iter,
       [=](scalar_t a, scalar_t b) -> bool {
         return a > b;
       });
    });
  } else {
    AT_DISPATCH_ALL_TYPES_AND2(kBFloat16, kHalf, iter.dtype(), "gt_cpu", [&]() {
      cpu_kernel_vec(
        iter,
        [](scalar_t a, scalar_t b) -> scalar_t {
          return a > b;
        },
        [](Vec256<scalar_t> a, Vec256<scalar_t> b) -> Vec256<scalar_t> {
          return a.gt(b);
        });
      });
  }
}

void ge_kernel(TensorIterator& iter) {
  if (iter.dtype() == ScalarType::Bool) {
    AT_DISPATCH_ALL_TYPES_AND3(kBool, kBFloat16, kHalf, iter.input_dtype(), "ge_cpu", [&]() {
      cpu_kernel(iter,
       [](scalar_t a, scalar_t b) -> bool {
         return a >= b;
       });
    });
  } else {
    AT_DISPATCH_ALL_TYPES_AND2(kBFloat16, kHalf, iter.dtype(), "ge_cpu", [&]() {
      cpu_kernel_vec(
        iter,
        [](scalar_t a, scalar_t b) -> scalar_t {
          return a >= b;
        },
        [](Vec256<scalar_t> a, Vec256<scalar_t> b) -> Vec256<scalar_t> {
          return a.ge(b);
        });
      });
  }
}

void eq_kernel(TensorIterator& iter) {
  if (iter.dtype() == ScalarType::Bool) {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(kBool, kBFloat16, kHalf, iter.input_dtype(), "eq_cpu", [&]() {
      cpu_kernel(iter,
       [](scalar_t a, scalar_t b) -> bool {
         return a == b;
       });
    });
  } else {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(kBFloat16, kHalf, iter.dtype(), "eq_cpu", [&]() {
      cpu_kernel_vec(
        iter,
        [](scalar_t a, scalar_t b) -> scalar_t {
          return a == b;
        },
        [](Vec256<scalar_t> a, Vec256<scalar_t> b) -> Vec256<scalar_t> {
          return a.eq(b);
        });
      });
  }
}

void ne_kernel(TensorIterator& iter) {
  if (iter.dtype() == ScalarType::Bool) {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(kBool, kBFloat16, kHalf, iter.input_dtype(), "ne_cpu", [&]() {
      cpu_kernel(iter,
       [](scalar_t a, scalar_t b) -> bool {
         return a != b;
       });
    });
  } else {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(kBFloat16, kHalf, iter.dtype(), "ne_cpu", [&]() {
      cpu_kernel_vec(
        iter,
        [](scalar_t a, scalar_t b) -> scalar_t {
          return a != b;
        },
        [](Vec256<scalar_t> a, Vec256<scalar_t> b) -> Vec256<scalar_t> {
          return a.ne(b);
        });
      });
  }
}

void maximum_kernel(TensorIterator& iter) {
  if (iter.dtype() == ScalarType::Bool) {
    cpu_kernel(iter,
      [](bool a, bool b) -> bool {
        return a || b;
      });
  } else if (isIntegralType(iter.dtype(), /*includeBool=*/ false)) {
    AT_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "maximum_cpu", [&]() {
      cpu_kernel_vec(iter,
        [](scalar_t a, scalar_t b) -> scalar_t { return std::max(a, b); },
        [](Vec256<scalar_t> a, Vec256<scalar_t> b) { return at::vec256::maximum(a, b); });
    });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.dtype(), "maximum_cpu", [&]() {
      cpu_kernel_vec(iter,
        [](scalar_t a, scalar_t b) -> scalar_t {
          if (a != a || b != b) {
            return std::numeric_limits<scalar_t>::quiet_NaN();
          } else {
            return std::max(a, b);
          }
        },
        [](Vec256<scalar_t> a, Vec256<scalar_t> b) { return at::vec256::maximum(a, b); });
    });
  }
}

void minimum_kernel(TensorIterator& iter) {
  if (iter.dtype() == ScalarType::Bool) {
    cpu_kernel(iter,
      [](bool a, bool b) -> bool {
        return a && b;
      });
  } else if (isIntegralType(iter.dtype(), /*includeBool=*/ false)) {
    AT_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "minimum_cpu", [&]() {
      cpu_kernel_vec(iter,
        [](scalar_t a, scalar_t b) -> scalar_t { return std::min(a, b); },
        [](Vec256<scalar_t> a, Vec256<scalar_t> b) { return at::vec256::minimum(a, b); });
    });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.dtype(), "minimum_cpu", [&]() {
      cpu_kernel_vec(iter,
        [](scalar_t a, scalar_t b) -> scalar_t {
          if (a != a || b != b) {
            return std::numeric_limits<scalar_t>::quiet_NaN();
          } else {
            return std::min(a, b);
          }
        },
        [](Vec256<scalar_t> a, Vec256<scalar_t> b) { return at::vec256::minimum(a, b); });
    });
  }
}

void smooth_l1_kernel(TensorIterator& iter, double beta) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
        kBFloat16, kHalf, iter.dtype(), "smooth_l1_cpu", [&]() {
        using Vec = Vec256<scalar_t>;
        const Vec beta_vec(beta);
        const Vec point_five_vec(static_cast<scalar_t>(0.5));
        cpu_kernel_vec(
            iter,
            [&beta](scalar_t a, scalar_t b) -> scalar_t {
              auto z = std::abs(a - b);
              return z < beta
                  ? static_cast<scalar_t>(0.5) * z * z / beta
                  : z - static_cast<scalar_t>(0.5) * beta;
            },
            [&beta_vec, &point_five_vec](Vec a, Vec b) {
              auto z = (a - b).abs();
              return Vec::blendv(
                  point_five_vec * z * z / beta_vec, z - point_five_vec * beta_vec, z >= beta_vec);
            });
      });
}

void sigmoid_backward_kernel(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "sigmoid_backward_cpu", [&]() {
    auto one_vec = Vec256<scalar_t>((scalar_t)(1));
    cpu_kernel_vec(iter,
      [=](scalar_t a, scalar_t b) -> scalar_t {
        return a * (scalar_t(1) - b) * b;
      },
      [=](Vec256<scalar_t> a, Vec256<scalar_t> b) {
        return a * (one_vec - b) * b;
      });
  });
}

void logit_backward_kernel(TensorIterator& iter, Scalar eps_scalar) {
  AT_DISPATCH_FLOATING_TYPES_AND(
      kBFloat16, iter.dtype(), "logit_backward_cpu", [&]() {
        const scalar_t eps = eps_scalar.to<scalar_t>();
        const Vec256<scalar_t> kZeroVec(scalar_t(0));
        const Vec256<scalar_t> kOneVec(scalar_t(1));
        if (eps < scalar_t(0)) {
          const Vec256<scalar_t> kNanVec(
              std::numeric_limits<scalar_t>::quiet_NaN());
          cpu_kernel_vec(
              iter,
              [](scalar_t dy, scalar_t x) {
                return (x < scalar_t(0) || x > scalar_t(1))
                    ? std::numeric_limits<scalar_t>::quiet_NaN()
                    : ((x == scalar_t(0) || x == scalar_t(1))
                           ? (dy * std::numeric_limits<scalar_t>::infinity())
                           : (dy / (x * (scalar_t(1) - x))));
              },
              [kZeroVec, kOneVec, kNanVec](
                  Vec256<scalar_t> dy_vec, Vec256<scalar_t> x_vec) {
                return Vec256<scalar_t>::blendv(
                    kNanVec,
                    dy_vec / (x_vec * (kOneVec - x_vec)),
                    (x_vec >= kZeroVec) & (x_vec <= kOneVec));
              });
        } else {
          const scalar_t lo = eps;
          const scalar_t hi = scalar_t(1) - eps;
          const Vec256<scalar_t> lo_vec(lo);
          const Vec256<scalar_t> hi_vec(hi);
          cpu_kernel_vec(
              iter,
              [lo, hi](scalar_t dy, scalar_t x) {
                return (x < lo || x > hi)
                    ? scalar_t(0)
                    : ((x == scalar_t(0) || x == scalar_t(1))
                           ? dy * std::numeric_limits<scalar_t>::infinity()
                           : dy / (x * (scalar_t(1) - x)));
              },
              [kZeroVec, kOneVec, lo_vec, hi_vec](
                  Vec256<scalar_t> dy_vec, Vec256<scalar_t> x_vec) {
                return Vec256<scalar_t>::blendv(
                    kZeroVec,
                    dy_vec / (x_vec * (kOneVec - x_vec)),
                    (x_vec >= lo_vec) & (x_vec <= hi_vec));
              });
        }
      });
}

void tanh_backward_kernel(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(iter.dtype(), "tanh_backward_cpu", [&]() {
    auto one_vec = Vec256<scalar_t>(scalar_t{1});
    cpu_kernel_vec(
      iter,
      [=](scalar_t a, scalar_t b) -> scalar_t {
        return a * (scalar_t{1} - b * b);
      },
      [=](Vec256<scalar_t> a, Vec256<scalar_t> b) {
        return a * (one_vec - b * b);
      });
  });
}

void mse_kernel(TensorIterator& iter) {
  if (iter.dtype() == ScalarType::Half) {
    TORCH_WARN_ONCE("Applying the CPU mse kernel on half-type tensors. "
                    "This may be slower than using float or double-type tensors.");
  }

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(iter.dtype(), "mse_cpu", [&]() {
    cpu_kernel_vec(iter,
      [=](scalar_t a, scalar_t b) -> scalar_t {
        auto diff = a - b;
        return diff * diff;
      },
      [=](Vec256<scalar_t> a, Vec256<scalar_t> b) {
      auto diff =  a - b;
      return diff * diff;
      });
  });
}

void fmod_kernel(TensorIterator& iter) {
  if (isIntegralType(iter.dtype(), /*includeBool=*/ false)) {
    AT_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "fmod_cpu", [&]() {
      cpu_kernel(iter, [=](scalar_t x, scalar_t d) -> scalar_t {
        TORCH_CHECK(d != 0, "ZeroDivisionError");
        return x % d;
      });
    });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND(kHalf, iter.dtype(), "fmod_cpu", [&]() {
      cpu_kernel_vec(
        iter,
        [](scalar_t x, scalar_t d) -> scalar_t {
          return std::fmod(x, d);
        },
        [](Vec256<scalar_t> x, Vec256<scalar_t> d) {
          return x.fmod(d);
        });
      });
  }
}

void fmod_scalar_kernel(TensorIterator& iter, Scalar divisor) {
  if (isIntegralType(iter.dtype(), /*includeBool=*/ false)) {
    AT_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "fmod_scalar_cpu", [&]() {
      const auto div = divisor.to<scalar_t>();
      TORCH_CHECK(div != 0, "ZeroDivisionError");
      cpu_kernel(iter, [=](scalar_t x) -> scalar_t {
        return x % div;
      });
    });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND(kHalf, iter.dtype(), "fmod_scalar_cpu", [&]() {
      const auto div = divisor.to<scalar_t>();
      const auto div_vec = Vec256<scalar_t>(div);
      cpu_kernel_vec(
        iter,
        [=](scalar_t x) -> scalar_t {
          return std::fmod(x, div);
        },
        [=](Vec256<scalar_t> x) {
          return x.fmod(div_vec);
        });
      });
  }

}

void logaddexp_kernel(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "logaddexp_cpu", [&]() {
    cpu_kernel_vec(
        iter,
        [=](scalar_t a, scalar_t b) -> scalar_t {
          if (std::isinf(a) && a == b) {
            return a;
          } else {
            scalar_t m = std::max(a, b);
            return m + std::log((scalar_t)(1.0) + std::exp(-std::abs(a - b)));
          }
        },
        [=](Vec256<scalar_t> a, Vec256<scalar_t> b) {
          Vec256<scalar_t> inf(std::numeric_limits<scalar_t>::infinity());
          Vec256<scalar_t> one(1.0);
          Vec256<scalar_t> m = maximum(a, b);
          return Vec256<scalar_t>::blendv(
              m + (one + (a - b).abs().neg().exp()).log(),
              a,
              (a == b) & (a.abs() == inf));
        });
  });
}

void logaddexp2_kernel(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "logaddexp2_cpu", [&]() {
    cpu_kernel_vec(
        iter,
        [=](scalar_t a, scalar_t b) -> scalar_t {
          if (std::isinf(a) && a == b) {
            return a;
          } else {
            scalar_t m = std::max(a, b);
            return m + std::log2((scalar_t)(1.0) + std::pow((scalar_t)(2), -std::abs(a - b)));
          }
        },
        [=](Vec256<scalar_t> a, Vec256<scalar_t> b) {
          Vec256<scalar_t> inf(std::numeric_limits<scalar_t>::infinity());
          Vec256<scalar_t> one(1.0);
          Vec256<scalar_t> two(2.0);
          Vec256<scalar_t> m = maximum(a, b);
          return Vec256<scalar_t>::blendv(
              m + (one + two.pow((a - b).abs().neg())).log2(),
              a,
              (a == b) & (a.abs() == inf));
        });
  });
}

void gcd_kernel(TensorIterator& iter) {
  AT_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "gcd_cpu", [&]() {
      cpu_kernel(
          iter,
          [](scalar_t a, scalar_t b) -> scalar_t {
            return calc_gcd(a, b);
          });
    });
}

void lcm_kernel(TensorIterator& iter) {
  AT_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "lcm_cpu", [&]() {
      cpu_kernel(
          iter,
          [](scalar_t a, scalar_t b) -> scalar_t {
            scalar_t g = calc_gcd(a, b);
            return (g == 0) ? 0 : std::abs(a / g * b);
          });
    });
}

void hypot_kernel(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND(kBFloat16, iter.dtype(), "hypot_cpu", [&]() {
    cpu_kernel_vec(
        iter,
        [=](scalar_t a, scalar_t b) -> scalar_t {
            return std::hypot(a, b);
        },
        [=](Vec256<scalar_t> a, Vec256<scalar_t> b) {
            return a.hypot(b);
        });
  });
}

void nextafter_kernel(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "nextafter_cpu", [&]() {
    cpu_kernel_vec(
        iter,
        [=](scalar_t a, scalar_t b) -> scalar_t {
            return std::nextafter(a, b);
        },
        [=](Vec256<scalar_t> a, Vec256<scalar_t> b) {
            return a.nextafter(b);
        });
  });
}

void heaviside_kernel(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND3(kHalf, kBool, kBFloat16, iter.dtype(), "heaviside_cpu", [&]() {
    cpu_kernel(iter, [](scalar_t a, scalar_t b) -> scalar_t {
        return a == 0 ? b : static_cast<scalar_t>(a > 0);
    });
  });
}

} // namespace

REGISTER_DISPATCH(add_stub, &add_kernel);
REGISTER_DISPATCH(add_clamp_stub, &add_clamp_kernel);
REGISTER_DISPATCH(sub_stub, &sub_kernel);
REGISTER_DISPATCH(mul_stub, &mul_kernel);
REGISTER_DISPATCH(div_stub, &div_kernel);
REGISTER_DISPATCH(remainder_stub, &remainder_kernel);
REGISTER_DISPATCH(atan2_stub, &atan2_kernel);
REGISTER_DISPATCH(bitwise_and_stub, &bitwise_and_kernel);
REGISTER_DISPATCH(bitwise_or_stub, &bitwise_or_kernel);
REGISTER_DISPATCH(bitwise_xor_stub, &bitwise_xor_kernel);
REGISTER_DISPATCH(lshift_stub, &lshift_kernel);
REGISTER_DISPATCH(rshift_stub, &rshift_kernel);
REGISTER_DISPATCH(logical_xor_stub, &logical_xor_kernel);
REGISTER_DISPATCH(logical_and_stub, &logical_and_kernel);
REGISTER_DISPATCH(logical_or_stub, &logical_or_kernel);
REGISTER_DISPATCH(lt_stub, &lt_kernel);
REGISTER_DISPATCH(le_stub, &le_kernel);
REGISTER_DISPATCH(gt_stub, &gt_kernel);
REGISTER_DISPATCH(ge_stub, &ge_kernel);
REGISTER_DISPATCH(eq_stub, &eq_kernel);
REGISTER_DISPATCH(ne_stub, &ne_kernel);
REGISTER_DISPATCH(maximum_stub, &maximum_kernel);
REGISTER_DISPATCH(minimum_stub, &minimum_kernel);
REGISTER_DISPATCH(smooth_l1_stub, &smooth_l1_kernel);
REGISTER_DISPATCH(sigmoid_backward_stub, &sigmoid_backward_kernel);
REGISTER_DISPATCH(logit_backward_stub, &logit_backward_kernel);
REGISTER_DISPATCH(tanh_backward_stub, &tanh_backward_kernel);
REGISTER_DISPATCH(mse_stub, &mse_kernel);
REGISTER_DISPATCH(fmod_stub, &fmod_kernel);
REGISTER_DISPATCH(fmod_scalar_stub, &fmod_scalar_kernel);
REGISTER_DISPATCH(logaddexp_stub, &logaddexp_kernel);
REGISTER_DISPATCH(logaddexp2_stub, &logaddexp2_kernel);
REGISTER_DISPATCH(gcd_stub, &gcd_kernel);
REGISTER_DISPATCH(lcm_stub, &lcm_kernel);
REGISTER_DISPATCH(hypot_stub, &hypot_kernel);
REGISTER_DISPATCH(nextafter_stub, &nextafter_kernel);
REGISTER_DISPATCH(heaviside_stub, &heaviside_kernel);

} // namespace native
} // namespace at
