#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/native/BinaryOps.h>

#include <cmath>
#include <iostream>

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>
#include <ATen/native/Math.h>
#include <c10/macros/Macros.h>
#include <c10/util/TypeSafeSignMath.h>
#include <c10/util/copysign.h>

namespace at {
namespace native {

namespace {

using namespace vec;

void add_clamp_kernel(TensorIterator& iter, const Scalar& alpha_scalar, const Scalar& min_val, const Scalar& max_val) {
  AT_DISPATCH_ALL_TYPES(iter.dtype(), "add_clamp_cpu", [&]() {
    auto alpha = alpha_scalar.to<scalar_t>();
    auto alpha_vec = Vectorized<scalar_t>(alpha);
    auto min_scalar = min_val.to<scalar_t>();
    auto min_vec = Vectorized<scalar_t>(min_scalar);
    auto max_scalar = max_val.to<scalar_t>();
    auto max_vec = Vectorized<scalar_t>(max_scalar);
    cpu_kernel_vec(iter,
      [=](scalar_t a, scalar_t b) __ubsan_ignore_undefined__ -> scalar_t {
        return std::min(max_scalar, std::max(min_scalar, static_cast<scalar_t>(a + alpha * b)));
      },
      [=](Vectorized<scalar_t> a, Vectorized<scalar_t> b) __ubsan_ignore_undefined__ {
        auto add_clamp_res = vec::fmadd(b, alpha_vec, a);
        add_clamp_res = vec::clamp_min(add_clamp_res, min_vec);
        add_clamp_res = vec::clamp_max(add_clamp_res, max_vec);
        return add_clamp_res;
      });
    });
}

void atan2_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND(kBFloat16, iter.dtype(), "atan2_cpu", [&]() {
    cpu_kernel_vec(iter, [=](scalar_t a, scalar_t b) -> scalar_t {
    return std::atan2(a, b);
  },
    [=](Vectorized<scalar_t> a, Vectorized<scalar_t> b) {
      return a.atan2(b);
    });
  });
}

void mul_kernel(TensorIteratorBase& iter) {
  if (iter.dtype() == ScalarType::Bool) {
    cpu_kernel(iter, [=](bool a, bool b) -> bool { return a && b; });
  } else if (iter.dtype() == kComplexHalf) {
    cpu_kernel(
        iter,
        [=](c10::complex<at::Half> a,
            c10::complex<at::Half> b) -> c10::complex<at::Half> {
          using comp_t = c10::complex<float>;
          return comp_t{a} * comp_t{b};
        });
  } else {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(kBFloat16, kHalf, iter.dtype(), "mul_cpu", [&]() {
      cpu_kernel_vec(iter,
        [=](scalar_t a, scalar_t b) __ubsan_ignore_undefined__ -> scalar_t { return a * b; },
        [=](Vectorized<scalar_t> a, Vectorized<scalar_t> b) __ubsan_ignore_undefined__ {
          return a * b;
        });
    });
  }
}

void div_true_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(kBFloat16, kHalf, iter.common_dtype(), "div_cpu", [&]() {
    cpu_kernel_vec(iter,
      [](scalar_t a, scalar_t b) __ubsan_ignore_float_divide_by_zero__ -> scalar_t {
        return a / b;
      },
      [](Vectorized<scalar_t> a, Vectorized<scalar_t> b) {
        return a / b;
      });
  });
}

void div_trunc_kernel(TensorIteratorBase& iter) {
  const auto dtype = iter.common_dtype();
  if (isIntegralType(dtype, /*includeBool*/ false)) {
    // There's no SIMD integer division, so don't try to vectorize it.
    // TODO: if the divisor is a scalar, rewrite as multiplication by a constant.
    AT_DISPATCH_INTEGRAL_TYPES(dtype, "div_trunc_cpu", [&]() {
      cpu_kernel(iter, [](scalar_t a, scalar_t b) -> scalar_t {
        TORCH_CHECK(b != 0, "ZeroDivisionError");
        return a / b;
      });
    });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, dtype, "div_trunc_cpu", [&]() {
      cpu_kernel_vec(iter,
        [](scalar_t a, scalar_t b) __ubsan_ignore_float_divide_by_zero__ -> scalar_t {
          return std::trunc(a / b);
        },
        [](Vectorized<scalar_t> a, Vectorized<scalar_t> b) {
          return (a / b).trunc();
        });
    });
  }
}

// NOTE: [Floor Division in Python]
// Python's __floordiv__ operator is more complicated than just floor(a / b).
// It aims to maintain the property: a == (a // b) * b + remainder(a, b)
// which can otherwise fail due to rounding errors in the remainder.
// So, instead it is calculated as: a // b = (a - remainder(a, b)) / b
// With some additional fix-ups added to the result.
//
// For reference, see CPython's implementation:
// https://github.com/python/cpython/blob/ace008c531dd685a30c1dd68f9b5ba35f20171cf/Objects/floatobject.c#L636

void div_floor_kernel(TensorIteratorBase& iter) {
  const auto dtype = iter.common_dtype();
  if (dtype == kByte) {
    // In the special case of unsigned integer division, floor division is
    // equivalent to truncation division (since the signs of the divisor and
    // dividend are always the same)
    return div_trunc_kernel(iter);
  } else if (isIntegralType(dtype, /*includeBool*/ false)) {
    // There's no SIMD integer division, so don't try to vectorize it.
    AT_DISPATCH_INTEGRAL_TYPES(dtype, "div_floor_cpu", [&]() {
      cpu_kernel(iter, [](scalar_t a, scalar_t b) -> scalar_t {
        TORCH_CHECK(b != 0, "ZeroDivisionError");
        if (c10::is_negative(a) != c10::is_negative(b)) {
          // Subtracts one from the results of truncation division if the
          // divisor and dividend have different sign(bit)s and the remainder of
          // the division is nonzero
          const auto quot = a / b;
          const auto rem = a % b;
          return rem ? quot - 1 : quot;
        }

        return a / b;
      });
    });
  } else {
    // See NOTE: [Floor Division in Python]
    AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, dtype, "div_floor_cpu", [&]() {
      using vec_t = Vectorized<scalar_t>;
      cpu_kernel_vec(iter,
          [](scalar_t a, scalar_t b) __ubsan_ignore_float_divide_by_zero__ -> scalar_t {
            if (C10_UNLIKELY(b == 0)) {
              // Divide by zero: return standard IEEE result
              return a / b;
            }

            auto mod = std::fmod(a, b);
            auto div = (a - mod) / b;
            if ((mod != 0) && (b < 0) != (mod < 0)) {
              div -= scalar_t(1);
            }

            scalar_t floordiv;
            if (div != 0) {
              floordiv = std::floor(div);
              if (div - floordiv > scalar_t(0.5)) {
                floordiv += scalar_t(1.0);
              }
            } else {
              floordiv = c10::copysign(scalar_t(0), a / b);
            }
            return floordiv;
          },
          [](vec_t a, vec_t b) -> vec_t {
            auto mod = a.fmod(b);
            auto div = (a - mod) / b;
            const auto zero = vec_t(0);
            auto mask = (mod != zero) & ((b < zero) ^ (mod < zero));
            const auto one = vec_t(1);
            div = vec_t::blendv(div, div - one, mask);
            auto floordiv = div.floor();
            mask = (div - floordiv) > vec_t(0.5);
            floordiv = vec_t::blendv(floordiv, floordiv + one, mask);
            const auto basic_div = a / b;
            floordiv = vec_t::blendv(floordiv, zero.copysign(basic_div), div == zero);
            floordiv = vec_t::blendv(floordiv, basic_div, b == zero);
            return floordiv;
          });
    });
  }
}

void remainder_kernel(TensorIteratorBase& iter) {
  if (isIntegralType(iter.common_dtype(), /*includeBool*/ false)) {
    AT_DISPATCH_INTEGRAL_TYPES(iter.common_dtype(), "remainder_cpu", [&]() {
      cpu_kernel(iter, [](scalar_t a, scalar_t b) -> scalar_t {
        TORCH_CHECK(b != 0, "ZeroDivisionError");
        scalar_t r = a % b;
        if ((r != 0) && (c10::is_negative(r) != c10::is_negative(b))) {
          r += b;
        }
        return r;
      });
    });
  } else if (iter.common_dtype() == kBFloat16) {
    cpu_kernel_vec(iter,
      [=](BFloat16 a, BFloat16 b) __ubsan_ignore_float_divide_by_zero__ -> BFloat16 {
        float a0 = static_cast<float>(a);
        float b0 = static_cast<float>(b);
        float mod0 = std::fmod(a0, b0);
        if ((mod0 != 0) && ((b0 < 0) != (mod0 < 0))) {
          mod0 += b0;
        }
        return mod0;
      },
      [=](Vectorized<BFloat16> a, Vectorized<BFloat16> b) {
        Vectorized<float> a0, a1, b0, b1;
        std::tie(a0, a1) = convert_bfloat16_float(a);
        std::tie(b0, b1) = convert_bfloat16_float(b);
        auto mod0 = a0.fmod(b0);
        auto mod1 = a1.fmod(b1);
        const auto zero = Vectorized<float>(0);
        auto mask0 = (mod0 != zero) & ((b0 < zero) ^ (mod0 < zero));
        auto mask1 = (mod1 != zero) & ((b1 < zero) ^ (mod1 < zero));
        a0 = Vectorized<float>::blendv(mod0, mod0 + b0, mask0);
        a1 = Vectorized<float>::blendv(mod1, mod1 + b1, mask1);
        return convert_float_bfloat16(a0, a1);
      });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(iter.common_dtype(), "remainder_cpu", [&]() {
      cpu_kernel_vec(iter,
        [=](scalar_t a, scalar_t b) __ubsan_ignore_float_divide_by_zero__ -> scalar_t {
          scalar_t mod = std::fmod(a, b);
          if ((mod != 0) && ((b < 0) != (mod < 0))) mod += b;
          return mod;
        },
        [=](Vectorized<scalar_t> a, Vectorized<scalar_t> b) {
          auto mod = a.fmod(b);
          const auto zero = Vectorized<scalar_t>(0);
          auto mask = (mod != zero) & ((b < zero) ^ (mod < zero));
          return Vectorized<scalar_t>::blendv(mod, mod + b, mask);
        });
    });
  }
}

void bitwise_and_kernel(TensorIteratorBase& iter) {
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
          [](Vectorized<scalar_t> a, Vectorized<scalar_t> b) {
            return a & b;
          });
    });
  }
}

void bitwise_or_kernel(TensorIteratorBase& iter) {
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
          [](Vectorized<scalar_t> a, Vectorized<scalar_t> b) {
            return a | b;
          });
    });
  }
}

void bitwise_xor_kernel(TensorIteratorBase& iter) {
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
          [](Vectorized<scalar_t> a, Vectorized<scalar_t> b) {
            return a ^ b;
          });
    });
  }
}

void lshift_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "lshift_cpu", [&]() {
    cpu_kernel(iter,
      [](scalar_t a, scalar_t b) -> scalar_t {
        return static_cast<std::make_unsigned_t<scalar_t>>(a) << b;
    });
  });
}

void logical_and_kernel(TensorIterator& iter) {
  // See Note [special-case bool outputs]
  if (iter.dtype() == ScalarType::Bool) {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(kBool, kBFloat16, kHalf, iter.common_dtype(), "logical_and_cpu", [&]() {
      cpu_kernel(iter,
        [](scalar_t a, scalar_t b) -> bool {
          return a && b;
        });
    });
  } else {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(kBFloat16, kHalf, iter.common_dtype(), "logical_and_cpu", [&]() {
      cpu_kernel(iter,
        [](scalar_t a, scalar_t b) -> scalar_t {
          return static_cast<scalar_t>(a && b);
        });
    });
  }
}

void logical_or_kernel(TensorIterator& iter) {
  // See Note [special-case bool outputs]
  if (iter.dtype() == ScalarType::Bool) {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(kBool, kBFloat16, kHalf, iter.common_dtype(), "logical_or_cpu", [&]() {
      cpu_kernel(iter,
        [](scalar_t a, scalar_t b) -> bool {
          return a || b;
        });
    });
  } else {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(kBool, kBFloat16, kHalf, iter.common_dtype(), "logical_or_cpu", [&]() {
      cpu_kernel(iter,
        [](scalar_t a, scalar_t b) -> scalar_t {
          return static_cast<scalar_t>(a || b);
        });
    });
  }
}

void logical_xor_kernel(TensorIterator& iter) {
  // See Note [special-case bool outputs]
  if (iter.dtype() == ScalarType::Bool) {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(kBool, kBFloat16, kHalf, iter.common_dtype(), "logical_xor_cpu", [&]() {
      cpu_kernel(iter,
        [](scalar_t a, scalar_t b) -> bool {
          return bool(a) != bool(b);
        });
    });
  } else {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(kBFloat16, kHalf, iter.common_dtype(), "logical_xor_cpu", [&]() {
      cpu_kernel(iter,
        [](scalar_t a, scalar_t b) -> scalar_t {
          return static_cast<scalar_t>(bool(a) != bool(b));
        });
    });
  }
}

void rshift_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "rshift_cpu", [&]() {
    cpu_kernel(iter,
      [](scalar_t a, scalar_t b) -> scalar_t {
        return a >> b;
      });
  });
}

void lt_kernel(TensorIteratorBase& iter) {
  // See Note [special-case bool outputs]
  if (iter.dtype() == ScalarType::Bool) {
    AT_DISPATCH_ALL_TYPES_AND3(kBool, kBFloat16, kHalf, iter.common_dtype(), "lt_cpu", [&]() {
      cpu_kernel(iter,
        [](scalar_t a, scalar_t b) -> bool {
          return a < b;
        });
    });
  } else {
    AT_DISPATCH_ALL_TYPES_AND2(kBFloat16, kHalf, iter.common_dtype(), "lt_cpu", [&]() {
      cpu_kernel_vec(
        iter,
        [](scalar_t a, scalar_t b) -> scalar_t {
          return a < b;
        },
        [](Vectorized<scalar_t> a, Vectorized<scalar_t> b) -> Vectorized<scalar_t> {
          return a.lt(b);
        });
    });
  }
}

void le_kernel(TensorIteratorBase& iter) {
  // See Note [special-case bool outputs]
  if (iter.dtype() == ScalarType::Bool) {
    AT_DISPATCH_ALL_TYPES_AND3(kBool, kBFloat16, kHalf, iter.common_dtype(), "le_cpu", [&]() {
      cpu_kernel(iter,
        [](scalar_t a, scalar_t b) -> bool {
          return a <= b;
        });
    });
  } else {
    AT_DISPATCH_ALL_TYPES_AND2(kBFloat16, kHalf, iter.common_dtype(), "le_cpu", [&]() {
      cpu_kernel_vec(
        iter,
        [](scalar_t a, scalar_t b) -> scalar_t {
          return a <= b;
        },
        [](Vectorized<scalar_t> a, Vectorized<scalar_t> b) -> Vectorized<scalar_t> {
          return a.le(b);
        });
    });
  }
}

void gt_kernel(TensorIteratorBase& iter) {
  // See Note [special-case bool outputs]
  if (iter.dtype() == ScalarType::Bool) {
    AT_DISPATCH_ALL_TYPES_AND3(kBool, kBFloat16, kHalf, iter.common_dtype(), "gt_cpu", [&]() {
      cpu_kernel(iter,
        [](scalar_t a, scalar_t b) -> bool {
          return a > b;
        });
    });
  } else {
    AT_DISPATCH_ALL_TYPES_AND2(kBFloat16, kHalf, iter.common_dtype(), "gt_cpu", [&]() {
      cpu_kernel_vec(
        iter,
        [](scalar_t a, scalar_t b) -> scalar_t {
          return a > b;
        },
        [](Vectorized<scalar_t> a, Vectorized<scalar_t> b) -> Vectorized<scalar_t> {
          return a.gt(b);
        });
    });
  }
}

void ge_kernel(TensorIteratorBase& iter) {
  // See Note [special-case bool outputs]
  if (iter.dtype() == ScalarType::Bool) {
    AT_DISPATCH_ALL_TYPES_AND3(kBool, kBFloat16, kHalf, iter.common_dtype(), "ge_cpu", [&]() {
      cpu_kernel(iter,
        [](scalar_t a, scalar_t b) -> bool {
          return a >= b;
        });
    });
  } else {
    AT_DISPATCH_ALL_TYPES_AND2(kBFloat16, kHalf, iter.common_dtype(), "ge_cpu", [&]() {
      cpu_kernel_vec(
        iter,
        [](scalar_t a, scalar_t b) -> scalar_t {
          return a >= b;
        },
        [](Vectorized<scalar_t> a, Vectorized<scalar_t> b) -> Vectorized<scalar_t> {
          return a.ge(b);
        });
    });
  }
}

void eq_kernel(TensorIteratorBase& iter) {
  // See Note [special-case bool outputs]
  if (iter.dtype() == ScalarType::Bool) {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(kComplexHalf, kBool, kBFloat16, kHalf, iter.common_dtype(), "eq_cpu", [&]() {
      cpu_kernel(iter,
        [](scalar_t a, scalar_t b) -> bool {
          return a == b;
        });
    });
  } else {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(kComplexHalf, kBFloat16, kHalf, iter.common_dtype(), "eq_cpu", [&]() {
      cpu_kernel_vec(
        iter,
        [](scalar_t a, scalar_t b) -> scalar_t {
          return static_cast<scalar_t>(a == b);
        },
        [](Vectorized<scalar_t> a, Vectorized<scalar_t> b) -> Vectorized<scalar_t> {
          return a.eq(b);
        });
    });
  }
}

void ne_kernel(TensorIteratorBase& iter) {
  // See Note [special-case bool outputs]
  if (iter.dtype() == ScalarType::Bool) {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(kComplexHalf, kBool, kBFloat16, kHalf, iter.common_dtype(), "ne_cpu", [&]() {
      cpu_kernel(iter,
        [](scalar_t a, scalar_t b) -> bool {
          return a != b;
        });
    });
  } else {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(kComplexHalf, kBFloat16, kHalf, iter.common_dtype(), "ne_cpu", [&]() {
      cpu_kernel_vec(
        iter,
        [](scalar_t a, scalar_t b) -> scalar_t {
          return static_cast<scalar_t>(a != b);
        },
        [](Vectorized<scalar_t> a, Vectorized<scalar_t> b) -> Vectorized<scalar_t> {
          return a.ne(b);
        });
    });
  }
}

void maximum_kernel(TensorIteratorBase& iter) {
  if (iter.dtype() == ScalarType::Bool) {
    cpu_kernel(iter,
      [](bool a, bool b) -> bool {
        return a || b;
      });
  } else if (isIntegralType(iter.dtype(), /*includeBool=*/ false)) {
    AT_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "maximum_cpu", [&]() {
      cpu_kernel_vec(iter,
        [](scalar_t a, scalar_t b) -> scalar_t { return std::max(a, b); },
        [](Vectorized<scalar_t> a, Vectorized<scalar_t> b) { return at::vec::maximum(a, b); });
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
        [](Vectorized<scalar_t> a, Vectorized<scalar_t> b) { return at::vec::maximum(a, b); });
    });
  }
}

void minimum_kernel(TensorIteratorBase& iter) {
  if (iter.dtype() == ScalarType::Bool) {
    cpu_kernel(iter,
      [](bool a, bool b) -> bool {
        return a && b;
      });
  } else if (isIntegralType(iter.dtype(), /*includeBool=*/ false)) {
    AT_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "minimum_cpu", [&]() {
      cpu_kernel_vec(iter,
        [](scalar_t a, scalar_t b) -> scalar_t { return std::min(a, b); },
        [](Vectorized<scalar_t> a, Vectorized<scalar_t> b) { return at::vec::minimum(a, b); });
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
        [](Vectorized<scalar_t> a, Vectorized<scalar_t> b) { return at::vec::minimum(a, b); });
    });
  }
}

void fmax_kernel(TensorIteratorBase& iter) {
  if (isFloatingType(iter.common_dtype())) {
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.common_dtype(), "fmax_cpu", [&]() {
      cpu_kernel(iter,
        [](scalar_t a, scalar_t b) -> scalar_t {
          return std::fmax(a, b);
        });
    });
  } else {
    maximum_kernel(iter);
  }
}

void fmin_kernel(TensorIteratorBase& iter) {
  if (isFloatingType(iter.common_dtype())) {
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.common_dtype(), "fmin_cpu", [&]() {
      cpu_kernel(iter,
        [](scalar_t a, scalar_t b) -> scalar_t {
          return std::fmin(a, b);
        });
    });
  } else {
    minimum_kernel(iter);
  }
}

void smooth_l1_kernel(TensorIteratorBase& iter, double beta) {
  if (iter.dtype() == kBFloat16) {
    const float beta_val(beta);
    const Vectorized<float> beta_val_vec(beta_val);
    const Vectorized<float> point_five_vec(static_cast<float>(0.5));
    cpu_kernel_vec(
        iter,
        [&beta_val](BFloat16 a, BFloat16 b) -> BFloat16 {
          auto z = std::abs(float(a) - float(b));
          return z < beta_val
              ? static_cast<float>(0.5) * z * z / beta_val
              : z - static_cast<float>(0.5) * beta_val;
        },
        [&beta_val_vec, &point_five_vec](Vectorized<BFloat16> a, Vectorized<BFloat16> b) {
          Vectorized<float> a0, a1, b0, b1;
          std::tie(a0, a1) = convert_bfloat16_float(a);
          std::tie(b0, b1) = convert_bfloat16_float(b);
          auto z = (a0 - b0).abs();
          a0 =  Vectorized<float>::blendv(
              point_five_vec * z * z / beta_val_vec, z - point_five_vec * beta_val_vec, z >= beta_val_vec);
          z = (a1 - b1).abs();
          a1 =  Vectorized<float>::blendv(
              point_five_vec * z * z / beta_val_vec, z - point_five_vec * beta_val_vec, z >= beta_val_vec);
          return convert_float_bfloat16(a0, a1);
        });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND(
        kHalf, iter.dtype(), "smooth_l1_cpu", [&]() {
        using Vec = Vectorized<scalar_t>;
        const scalar_t beta_val(beta);
        const Vec beta_val_vec(beta_val);
        const Vec point_five_vec(static_cast<scalar_t>(0.5));
        cpu_kernel_vec(
            iter,
            [&beta_val](scalar_t a, scalar_t b) -> scalar_t {
              auto z = std::abs(a - b);
              return z < beta_val
                  ? static_cast<scalar_t>(0.5) * z * z / beta_val
                  : z - static_cast<scalar_t>(0.5) * beta_val;
            },
            [&beta_val_vec, &point_five_vec](Vec a, Vec b) {
              auto z = (a - b).abs();
              return Vec::blendv(
                  point_five_vec * z * z / beta_val_vec, z - point_five_vec * beta_val_vec, z >= beta_val_vec);
            });
      });
  }
}

void huber_kernel(TensorIterator& iter, double delta) {
  AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, iter.dtype(), "huber_cpu", [&]() {
    using Vec = Vectorized<scalar_t>;
    const scalar_t delta_val(delta);
    const Vec delta_val_vec(delta_val);
    const Vec point_five_vec(static_cast<scalar_t>(0.5));
    cpu_kernel_vec(
      iter,
      [&delta_val](scalar_t a, scalar_t b) -> scalar_t {
        auto z = std::abs(a - b);
        return z < delta_val ? static_cast<scalar_t>(0.5) * z * z :
        delta_val * (z - static_cast<scalar_t>(0.5) * delta_val);
      },
      [&delta_val_vec, &point_five_vec](Vec a, Vec b) {
        auto z = (a - b).abs();
        return Vec::blendv(point_five_vec * z * z,
          delta_val_vec * (z - point_five_vec * delta_val_vec),
          z >= delta_val_vec);
    });
  });
}

void sigmoid_backward_kernel(TensorIteratorBase& iter) {
  if (isComplexType(iter.dtype())) {
    AT_DISPATCH_COMPLEX_TYPES(iter.dtype(), "sigmoid_backward_cpu", [&]() {
      auto one_vec = Vectorized<scalar_t>(scalar_t{1});
      cpu_kernel_vec(
        iter,
        [=](scalar_t a, scalar_t b) -> scalar_t {
          return a * std::conj((scalar_t(1) - b) * b);
        },
        [=](Vectorized<scalar_t> a, Vectorized<scalar_t> b) {
          return a * ((one_vec - b) * b).conj();
        });
    });
  } else if (iter.dtype() == kBFloat16) {
    auto one_vec = Vectorized<float>((float)(1));
    cpu_kernel_vec(
      iter,
      [=](BFloat16 a, BFloat16 b) -> BFloat16 {
         float a0 = static_cast<float>(a);
         float b0 = static_cast<float>(b);
         return a0 * (float(1) - b0) * b0;
      },
      [=](Vectorized<BFloat16> a, Vectorized<BFloat16> b) {
         Vectorized<float> a0, a1, b0, b1;
         std::tie(a0, a1) = convert_bfloat16_float(a);
         std::tie(b0, b1) = convert_bfloat16_float(b);
         a0 = a0 * (one_vec - b0) * b0;
         a1 = a1 * (one_vec - b1) * b1;
         return convert_float_bfloat16(a0, a1);
      });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND(kHalf, iter.dtype(), "sigmoid_backward_cpu", [&]() {
      auto one_vec = Vectorized<scalar_t>((scalar_t)(1));
      cpu_kernel_vec(
        iter,
        [=](scalar_t a, scalar_t b) -> scalar_t {
          return a * (scalar_t(1) - b) * b;
        },
        [=](Vectorized<scalar_t> a, Vectorized<scalar_t> b) {
          return a * (one_vec - b) * b;
        });
    });
  }
}

void logit_backward_kernel(TensorIteratorBase& iter, const Scalar& eps_scalar) {
  AT_DISPATCH_FLOATING_TYPES_AND(
      kBFloat16, iter.dtype(), "logit_backward_cpu", [&]() {
        const scalar_t eps = eps_scalar.to<scalar_t>();
        const Vectorized<scalar_t> kZeroVec(scalar_t(0));
        const Vectorized<scalar_t> kOneVec(scalar_t(1));
        if (eps < scalar_t(0)) {
          const Vectorized<scalar_t> kNanVec(
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
                  Vectorized<scalar_t> dy_vec, Vectorized<scalar_t> x_vec) {
                return Vectorized<scalar_t>::blendv(
                    kNanVec,
                    dy_vec / (x_vec * (kOneVec - x_vec)),
                    (x_vec >= kZeroVec) & (x_vec <= kOneVec));
              });
        } else {
          const scalar_t lo = eps;
          const scalar_t hi = scalar_t(1) - eps;
          const Vectorized<scalar_t> lo_vec(lo);
          const Vectorized<scalar_t> hi_vec(hi);
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
                  Vectorized<scalar_t> dy_vec, Vectorized<scalar_t> x_vec) {
                return Vectorized<scalar_t>::blendv(
                    kZeroVec,
                    dy_vec / (x_vec * (kOneVec - x_vec)),
                    (x_vec >= lo_vec) & (x_vec <= hi_vec));
              });
        }
      });
}

void tanh_backward_kernel(TensorIteratorBase& iter) {
  if (isComplexType(iter.dtype())) {
    AT_DISPATCH_COMPLEX_TYPES(iter.dtype(), "tanh_backward_cpu", [&]() {
      auto one_vec = Vectorized<scalar_t>(scalar_t{1});
      cpu_kernel_vec(
        iter,
        [=](scalar_t a, scalar_t b) -> scalar_t {
          return a * std::conj(scalar_t{1} - b * b);
        },
        [=](Vectorized<scalar_t> a, Vectorized<scalar_t> b) {
          return a * (one_vec - b * b).conj();
        });
    });
  } else if (iter.dtype() == kBFloat16) {
    auto one_vec = Vectorized<float>(float{1});
    cpu_kernel_vec(
      iter,
      [=](BFloat16 a, BFloat16 b) -> BFloat16 {
        float a0 = float(a);
        float b0 = float(b);
        return a0 * (float{1} - b0 * b0);
      },
      [=](Vectorized<BFloat16> a, Vectorized<BFloat16> b) {
        Vectorized<float> a0, a1, b0, b1;
        std::tie(a0, a1) = convert_bfloat16_float(a);
        std::tie(b0, b1) = convert_bfloat16_float(b);
        a0 = a0 * (one_vec - b0 * b0);
        a1 = a1 * (one_vec - b1 * b1);
        return convert_float_bfloat16(a0, a1);
      });
  } else {
    AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "tanh_backward_cpu", [&]() {
      auto one_vec = Vectorized<scalar_t>(scalar_t{1});
      cpu_kernel_vec(
        iter,
        [=](scalar_t a, scalar_t b) -> scalar_t {
          return a * (scalar_t{1} - b * b);
        },
        [=](Vectorized<scalar_t> a, Vectorized<scalar_t> b) {
          return a * (one_vec - b * b);
        });
    });
  }
}

void mse_kernel(TensorIteratorBase& iter) {
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
      [=](Vectorized<scalar_t> a, Vectorized<scalar_t> b) {
      auto diff =  a - b;
      return diff * diff;
      });
  });
}

void fmod_kernel(TensorIteratorBase& iter) {
  if (isIntegralType(iter.common_dtype(), /*includeBool=*/ false)) {
    AT_DISPATCH_INTEGRAL_TYPES(iter.common_dtype(), "fmod_cpu", [&]() {
      cpu_kernel(iter, [=](scalar_t x, scalar_t d) -> scalar_t {
        TORCH_CHECK(d != 0, "ZeroDivisionError");
        return x % d;
      });
    });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, iter.common_dtype(), "fmod_cpu", [&]() {
      cpu_kernel_vec(
        iter,
        [](scalar_t x, scalar_t d) -> scalar_t {
          return std::fmod(x, d);
        },
        [](Vectorized<scalar_t> x, Vectorized<scalar_t> d) {
          return x.fmod(d);
        });
    });
  }
}

void logaddexp_kernel(TensorIteratorBase& iter) {
  if (iter.dtype() == kBFloat16) {
    cpu_kernel_vec(
        iter,
        [=](BFloat16 a, BFloat16 b) -> BFloat16 {
          float a0 = static_cast<float>(a);
          float b0 = static_cast<float>(b);
          if (std::isinf(a0) && a0 == b0) {
            return a0;
          } else {
            float m0 = std::max(a0, b0);
            return m0 + std::log(static_cast<float>(1.0) + std::exp(-std::abs(a0 - b0)));
          }
        },
        [=](Vectorized<BFloat16> a, Vectorized<BFloat16> b) {
          Vectorized<float> a0, a1, b0, b1;
          std::tie(a0, a1) = convert_bfloat16_float(a);
          std::tie(b0, b1) = convert_bfloat16_float(b);
          Vectorized<float> inf(std::numeric_limits<float>::infinity());
          Vectorized<float> one(1.0);
          Vectorized<float> m0 = maximum(a0, b0);
          Vectorized<float> m1 = maximum(a1, b1);
          a0 = Vectorized<float>::blendv(
              m0 + (one + (a0 - b0).abs().neg().exp()).log(),
              a0,
              (a0 == b0) & (a0.abs() == inf));
          a1 = Vectorized<float>::blendv(
              m1 + (one + (a1 - b1).abs().neg().exp()).log(),
              a1,
              (a1 == b1) & (a1.abs() == inf));
          return convert_float_bfloat16(a0, a1);
        });
  } else {
    AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "logaddexp_cpu", [&]() {
      cpu_kernel_vec(
          iter,
          [=](scalar_t a, scalar_t b) -> scalar_t {
            if (std::isinf(a) && a == b) {
              return a;
            } else {
              scalar_t m = std::max(a, b);
              return m + std::log(static_cast<scalar_t>(1.0) + std::exp(-std::abs(a - b)));
            }
          },
          [=](Vectorized<scalar_t> a, Vectorized<scalar_t> b) {
            Vectorized<scalar_t> inf(std::numeric_limits<scalar_t>::infinity());
            Vectorized<scalar_t> one(1.0);
            Vectorized<scalar_t> m = maximum(a, b);
            return Vectorized<scalar_t>::blendv(
                m + (one + (a - b).abs().neg().exp()).log(),
                a,
                (a == b) & (a.abs() == inf));
          });
    });
  }
}

void logaddexp2_kernel(TensorIteratorBase& iter) {
  if (iter.dtype() == kBFloat16) {
    cpu_kernel_vec(
        iter,
        [=](BFloat16 a, BFloat16 b) -> BFloat16 {
          float a0 = static_cast<float>(a);
          float b0 = static_cast<float>(b);
          if (std::isinf(a0) && a0 == b0) {
            return a0;
          } else {
            float m0 = std::max(a0, b0);
            return m0 + std::log2(static_cast<float>(1.0) + std::pow(static_cast<float>(2), -std::abs(a0 - b0)));
          }
        },
        [=](Vectorized<BFloat16> a, Vectorized<BFloat16> b) {
          Vectorized<float> a0, a1, b0, b1;
          std::tie(a0, a1) = convert_bfloat16_float(a);
          std::tie(b0, b1) = convert_bfloat16_float(b);
          Vectorized<float> inf(std::numeric_limits<float>::infinity());
          Vectorized<float> one(1.0);
          Vectorized<float> two(2.0);
          Vectorized<float> m0 = maximum(a0, b0);
          Vectorized<float> m1 = maximum(a1, b1);
          a0 = Vectorized<float>::blendv(
              m0 + (one + two.pow((a0 - b0).abs().neg())).log2(),
              a0,
              (a0 == b0) & (a0.abs() == inf));
          a1 = Vectorized<float>::blendv(
              m1 + (one + two.pow((a1 - b1).abs().neg())).log2(),
              a1,
              (a1 == b1) & (a1.abs() == inf));
          return convert_float_bfloat16(a0, a1);
        });
  } else {
    AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "logaddexp2_cpu", [&]() {
      cpu_kernel_vec(
          iter,
          [=](scalar_t a, scalar_t b) -> scalar_t {
            if (std::isinf(a) && a == b) {
              return a;
            } else {
              scalar_t m = std::max(a, b);
              return m + std::log2(static_cast<scalar_t>(1.0) + std::pow(static_cast<scalar_t>(2), -std::abs(a - b)));
            }
          },
          [=](Vectorized<scalar_t> a, Vectorized<scalar_t> b) {
            Vectorized<scalar_t> inf(std::numeric_limits<scalar_t>::infinity());
            Vectorized<scalar_t> one(1.0);
            Vectorized<scalar_t> two(2.0);
            Vectorized<scalar_t> m = maximum(a, b);
            return Vectorized<scalar_t>::blendv(
                m + (one + two.pow((a - b).abs().neg())).log2(),
                a,
                (a == b) & (a.abs() == inf));
          });
    });
  }
}

void gcd_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_INTEGRAL_TYPES(iter.common_dtype(), "gcd_cpu", [&]() {
      cpu_kernel(
          iter,
          [](scalar_t a, scalar_t b) -> scalar_t {
            return calc_gcd(a, b);
          });
    });
}

void lcm_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_INTEGRAL_TYPES(iter.common_dtype(), "lcm_cpu", [&]() {
      cpu_kernel(
          iter,
          [](scalar_t a, scalar_t b) -> scalar_t {
            scalar_t g = calc_gcd(a, b);
            return (g == 0) ? 0 : std::abs(a / g * b);
          });
    });
}

void hypot_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND(kBFloat16, iter.dtype(), "hypot_cpu", [&]() {
    cpu_kernel_vec(
        iter,
        [=](scalar_t a, scalar_t b) -> scalar_t {
            return std::hypot(a, b);
        },
        [=](Vectorized<scalar_t> a, Vectorized<scalar_t> b) {
            return a.hypot(b);
        });
  });
}

void igamma_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, iter.dtype(), "igamma_cpu", [&]() {
    cpu_kernel_vec(
        iter,
        [=](scalar_t a, scalar_t b) -> scalar_t {
            return calc_igamma(a, b);
        },
        [=](Vectorized<scalar_t> a, Vectorized<scalar_t> b) {
            return a.igamma(b);
        });
  });
}

void igammac_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, iter.dtype(), "igammac_cpu", [&]() {
    cpu_kernel_vec(
        iter,
        [=](scalar_t a, scalar_t b) -> scalar_t {
            return calc_igammac(a, b);
        },
        [=](Vectorized<scalar_t> a, Vectorized<scalar_t> b) {
            return a.igammac(b);
        });
  });
}

void nextafter_kernel(TensorIteratorBase& iter) {
  if (iter.common_dtype() == kBFloat16) {
    using scalar_t = c10::BFloat16;
    cpu_kernel(
        iter,
        [=](scalar_t a, scalar_t b) -> scalar_t {
            return std::nextafter(a, b);
        });
  } else {
    AT_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "nextafter_cpu", [&]() {
    cpu_kernel_vec(
        iter,
        [=](scalar_t a, scalar_t b) -> scalar_t {
            return std::nextafter(a, b);
        },
        [=](Vectorized<scalar_t> a, Vectorized<scalar_t> b) {
            return a.nextafter(b);
        });
  });
  }
}

void heaviside_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_ALL_TYPES_AND3(kHalf, kBool, kBFloat16, iter.dtype(), "heaviside_cpu", [&]() {
    cpu_kernel(iter, [](scalar_t a, scalar_t b) -> scalar_t {
        return a == 0 ? b : static_cast<scalar_t>(a > 0);
    });
  });
}

void copysign_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, iter.common_dtype(), "copysign_cpu", [&]() {
    cpu_kernel_vec(iter,
      [](scalar_t a, scalar_t b) -> scalar_t {
        return c10::copysign(a, b);
      },
      [](Vectorized<scalar_t> a, Vectorized<scalar_t> b) -> Vectorized<scalar_t> {
        return a.copysign(b);
      });
  });
}

void xlogy_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, iter.common_dtype(), "xlogy_cpu", [&]() {
    cpu_kernel(iter, [](scalar_t x, scalar_t y) -> scalar_t {
      if (at::_isnan(y)){
        return NAN;
      }
      if (x == 0){
        return 0;
      }
      return x * std::log(y);
    });
  });
}

void xlog1py_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, iter.common_dtype(), "xlog1py_cpu", [&]() {
    cpu_kernel(iter, [](scalar_t x, scalar_t y) -> scalar_t {
      if (at::_isnan(y)){
        return NAN;
      }
      if (x == 0){
        return 0;
      }
      return x * std::log1p(y);
    });
  });
}

void zeta_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "zeta_cpu", [&]() {
    cpu_kernel(iter, [](scalar_t x, scalar_t q) -> scalar_t {
      return zeta(x, q);
    });
  });
}

void chebyshev_polynomial_t_kernel(TensorIteratorBase& iterator) {
    AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "chebyshev_polynomial_t_cpu", [&]() {
        cpu_kernel(iterator, [](scalar_t x, scalar_t n) -> scalar_t {
            return chebyshev_polynomial_t_forward(x, n);
        });
    });
} // chebyshev_polynomial_t_kernel(TensorIteratorBase& iterator)

void chebyshev_polynomial_u_kernel(TensorIteratorBase& iterator) {
    AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "chebyshev_polynomial_u_cpu", [&]() {
        cpu_kernel(iterator, [](scalar_t x, scalar_t n) -> scalar_t {
            return chebyshev_polynomial_u_forward(x, n);
        });
    });
} // chebyshev_polynomial_u_kernel(TensorIteratorBase& iterator)

void chebyshev_polynomial_v_kernel(TensorIteratorBase& iterator) {
    AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "chebyshev_polynomial_v_cpu", [&]() {
        cpu_kernel(iterator, [](scalar_t x, scalar_t n) -> scalar_t {
            return chebyshev_polynomial_v_forward(x, n);
        });
    });
} // chebyshev_polynomial_v_kernel(TensorIteratorBase& iterator)

void chebyshev_polynomial_w_kernel(TensorIteratorBase& iterator) {
    AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "chebyshev_polynomial_w_cpu", [&]() {
        cpu_kernel(iterator, [](scalar_t x, scalar_t n) -> scalar_t {
            return chebyshev_polynomial_w_forward(x, n);
        });
    });
} // chebyshev_polynomial_w_kernel(TensorIteratorBase& iterator)

void hermite_polynomial_h_kernel(TensorIteratorBase& iterator) {
    AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "hermite_polynomial_h_cpu", [&]() {
        cpu_kernel(iterator, [](scalar_t x, scalar_t n) -> scalar_t {
            return hermite_polynomial_h_forward(x, n);
        });
    });
} // hermite_polynomial_h_kernel(TensorIteratorBase& iterator)

void hermite_polynomial_he_kernel(TensorIteratorBase& iterator) {
    AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "hermite_polynomial_he_cpu", [&]() {
        cpu_kernel(iterator, [](scalar_t x, scalar_t n) -> scalar_t {
            return hermite_polynomial_he_forward(x, n);
        });
    });
} // hermite_polynomial_he_kernel(TensorIteratorBase& iterator)

void laguerre_polynomial_l_kernel(TensorIteratorBase& iterator) {
    AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "laguerre_polynomial_l_cpu", [&]() {
        cpu_kernel(iterator, [](scalar_t x, scalar_t n) -> scalar_t {
            return laguerre_polynomial_l_forward(x, n);
        });
    });
} // laguerre_polynomial_l_kernel(TensorIteratorBase& iterator)

void legendre_polynomial_p_kernel(TensorIteratorBase& iterator) {
    AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "legendre_polynomial_p_cpu", [&]() {
        cpu_kernel(iterator, [](scalar_t x, scalar_t n) -> scalar_t {
            return legendre_polynomial_p_forward(x, n);
        });
    });
} // legendre_polynomial_p_kernel(TensorIteratorBase& iterator)

void shifted_chebyshev_polynomial_t_kernel(TensorIteratorBase& iterator) {
    AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "shifted_chebyshev_polynomial_t_cpu", [&]() {
        cpu_kernel(iterator, [](scalar_t x, scalar_t n) -> scalar_t {
            return shifted_chebyshev_polynomial_t_forward(x, n);
        });
    });
} // shifted_chebyshev_polynomial_t_kernel(TensorIteratorBase& iterator)

void shifted_chebyshev_polynomial_u_kernel(TensorIteratorBase& iterator) {
    AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "shifted_chebyshev_polynomial_u_cpu", [&]() {
        cpu_kernel(iterator, [](scalar_t x, scalar_t n) -> scalar_t {
            return shifted_chebyshev_polynomial_u_forward(x, n);
        });
    });
} // shifted_chebyshev_polynomial_u_kernel(TensorIteratorBase& iterator)

void shifted_chebyshev_polynomial_v_kernel(TensorIteratorBase& iterator) {
    AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "shifted_chebyshev_polynomial_v_cpu", [&]() {
        cpu_kernel(iterator, [](scalar_t x, scalar_t n) -> scalar_t {
            return shifted_chebyshev_polynomial_v_forward(x, n);
        });
    });
} // shifted_chebyshev_polynomial_v_kernel(TensorIteratorBase& iterator)

void shifted_chebyshev_polynomial_w_kernel(TensorIteratorBase& iterator) {
    AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "shifted_chebyshev_polynomial_w_cpu", [&]() {
        cpu_kernel(iterator, [](scalar_t x, scalar_t n) -> scalar_t {
            return shifted_chebyshev_polynomial_w_forward(x, n);
        });
    });
} // shifted_chebyshev_polynomial_w_kernel(TensorIteratorBase& iterator)

} // namespace

REGISTER_DISPATCH(add_clamp_stub, &add_clamp_kernel);
REGISTER_DISPATCH(mul_stub, &mul_kernel);
REGISTER_DISPATCH(div_true_stub, &div_true_kernel);
REGISTER_DISPATCH(div_trunc_stub, &div_trunc_kernel);
REGISTER_DISPATCH(div_floor_stub, &div_floor_kernel);
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
REGISTER_DISPATCH(fmax_stub, &fmax_kernel);
REGISTER_DISPATCH(fmin_stub, &fmin_kernel);
REGISTER_DISPATCH(smooth_l1_stub, &smooth_l1_kernel);
REGISTER_DISPATCH(huber_stub, &huber_kernel);
REGISTER_DISPATCH(sigmoid_backward_stub, &sigmoid_backward_kernel);
REGISTER_DISPATCH(logit_backward_stub, &logit_backward_kernel);
REGISTER_DISPATCH(tanh_backward_stub, &tanh_backward_kernel);
REGISTER_DISPATCH(mse_stub, &mse_kernel);
REGISTER_DISPATCH(fmod_stub, &fmod_kernel);
REGISTER_DISPATCH(logaddexp_stub, &logaddexp_kernel);
REGISTER_DISPATCH(logaddexp2_stub, &logaddexp2_kernel);
REGISTER_DISPATCH(gcd_stub, &gcd_kernel);
REGISTER_DISPATCH(lcm_stub, &lcm_kernel);
REGISTER_DISPATCH(hypot_stub, &hypot_kernel);
REGISTER_DISPATCH(igamma_stub, &igamma_kernel);
REGISTER_DISPATCH(igammac_stub, &igammac_kernel);
REGISTER_DISPATCH(nextafter_stub, &nextafter_kernel);
REGISTER_DISPATCH(heaviside_stub, &heaviside_kernel);
REGISTER_DISPATCH(copysign_stub, &copysign_kernel);
REGISTER_DISPATCH(xlogy_stub, &xlogy_kernel);
REGISTER_DISPATCH(xlog1py_stub, &xlog1py_kernel);
REGISTER_DISPATCH(zeta_stub, &zeta_kernel);
REGISTER_DISPATCH(chebyshev_polynomial_t_stub, &chebyshev_polynomial_t_kernel);
REGISTER_DISPATCH(chebyshev_polynomial_u_stub, &chebyshev_polynomial_u_kernel);
REGISTER_DISPATCH(chebyshev_polynomial_v_stub, &chebyshev_polynomial_v_kernel);
REGISTER_DISPATCH(chebyshev_polynomial_w_stub, &chebyshev_polynomial_w_kernel);
REGISTER_DISPATCH(hermite_polynomial_h_stub, &hermite_polynomial_h_kernel);
REGISTER_DISPATCH(hermite_polynomial_he_stub, &hermite_polynomial_he_kernel);
REGISTER_DISPATCH(laguerre_polynomial_l_stub, &laguerre_polynomial_l_kernel);
REGISTER_DISPATCH(legendre_polynomial_p_stub, &legendre_polynomial_p_kernel);
REGISTER_DISPATCH(shifted_chebyshev_polynomial_t_stub, &shifted_chebyshev_polynomial_t_kernel);
REGISTER_DISPATCH(shifted_chebyshev_polynomial_u_stub, &shifted_chebyshev_polynomial_u_kernel);
REGISTER_DISPATCH(shifted_chebyshev_polynomial_v_stub, &shifted_chebyshev_polynomial_v_kernel);
REGISTER_DISPATCH(shifted_chebyshev_polynomial_w_stub, &shifted_chebyshev_polynomial_w_kernel);

} // namespace native
} // namespace at
