#include <cmath>
#include <iostream>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec256/vec256.h>
#include <ATen/cpu/vec256/functional.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/cpu/Loops.h>
#include <c10/macros/Macros.h>

namespace at { namespace native {
namespace {

using namespace vec256;

void add_kernel(TensorIterator& iter, Scalar alpha_scalar) {
  if (iter.dtype() == ScalarType::Bool) {
      using scalar_t = bool;
      auto alpha = alpha_scalar.to<scalar_t>();
      cpu_kernel(iter,
        [=](scalar_t a, scalar_t b) -> scalar_t { return a + alpha * b; });
  } else {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND(kBFloat16, iter.dtype(), "add_cpu/sub_cpu", [&]() {
      auto alpha = alpha_scalar.to<scalar_t>();
      auto alpha_vec = Vec256<scalar_t>(alpha);
      cpu_kernel_vec(iter,
        [=](scalar_t a, scalar_t b) -> scalar_t { return a + alpha * b; },
        [=](Vec256<scalar_t> a, Vec256<scalar_t> b) {
          return vec256::fmadd(b, alpha_vec, a);
        });
      });
  }
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

void sub_kernel(TensorIterator& iter, Scalar alpha_scalar) {
  add_kernel(iter, -alpha_scalar);
}

void mul_kernel(TensorIterator& iter) {
  if (iter.dtype() == ScalarType::Bool) {
    cpu_kernel(iter, [=](bool a, bool b) -> bool { return a && b; });
  } else {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND(kBFloat16, iter.dtype(), "mul_cpu", [&]() {
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
  } else if (isComplexType(iter.dtype())) {
      AT_DISPATCH_COMPLEX_TYPES(iter.dtype(), "div_cpu", [&]() {
        cpu_kernel_vec(iter,
          [=](scalar_t a, scalar_t b) __ubsan_ignore_float_divide_by_zero__ -> scalar_t {
             return a / b;
          },
          [=](Vec256<scalar_t> a, Vec256<scalar_t> b) {
            return a / b;
          });
      });
    } else {
    AT_DISPATCH_FLOATING_TYPES_AND(kBFloat16, iter.dtype(), "div_cpu", [&]() {
      cpu_kernel_vec(iter,
        [=](scalar_t a, scalar_t b) __ubsan_ignore_float_divide_by_zero__ -> scalar_t {
           return a / b;
        },
        [=](Vec256<scalar_t> a, Vec256<scalar_t> b) {
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
          return a - b * at::native::floor_impl(a / b);
        },
        [=](Vec256<scalar_t> a, Vec256<scalar_t> b) {
          Vec256<scalar_t> r = a - b * (a / b).floor();
          return r;
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

template<typename scalar_t>
static inline scalar_t lshift_wrapper(scalar_t a, scalar_t b) {
  return a << b;
}

static inline int8_t lshift_wrapper(int8_t a, int8_t b) {
  return ((uint8_t)a) << b;
}

static inline int16_t lshift_wrapper(int16_t a, int16_t b) {
  return ((uint16_t)a) << b;
}

static inline int32_t lshift_wrapper(int32_t a, int32_t b) {
  return ((uint32_t)a) << b;
}

static inline int64_t lshift_wrapper(int64_t a, int64_t b) {
  return ((uint64_t)a) << b;
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
          return lshift_wrapper(a, b);
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

template<typename scalar_t>
static inline scalar_t rshift_wrapper(scalar_t a, scalar_t b) {
  return a >> b;
}

static inline int8_t rshift_wrapper(int8_t a, int8_t b) {
  return ((uint8_t)a) >> b;
}

static inline int16_t rshift_wrapper(int16_t a, int16_t b) {
  return ((uint16_t)a) >> b;
}

static inline int32_t rshift_wrapper(int32_t a, int32_t b) {
  return ((uint32_t)a) >> b;
}

static inline int64_t rshift_wrapper(int64_t a, int64_t b) {
  return ((uint64_t)a) >> b;
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
          return rshift_wrapper(a, b);
      });
    });
  }
}

void lt_kernel(TensorIterator& iter) {
  if (iter.dtype() == ScalarType::Bool) {
    AT_DISPATCH_ALL_TYPES_AND2(kBool, kBFloat16, iter.input_dtype(), "lt_cpu", [&]() {
      cpu_kernel(iter,
       [=](scalar_t a, scalar_t b) -> bool {
         return a < b;
       });
    });
  } else {
    AT_DISPATCH_ALL_TYPES_AND(kBFloat16, iter.dtype(), "lt_cpu", [&]() {
      cpu_kernel(iter,
       [=](scalar_t a, scalar_t b) -> scalar_t {
         return a < b;
       });
    });
  }
}

void le_kernel(TensorIterator& iter) {
  if (iter.dtype() == ScalarType::Bool) {
    AT_DISPATCH_ALL_TYPES_AND2(kBool, kBFloat16, iter.input_dtype(), "le_cpu", [&]() {
      cpu_kernel(iter,
       [=](scalar_t a, scalar_t b) -> bool {
         return a <= b;
       });
    });
  } else {
    AT_DISPATCH_ALL_TYPES_AND(kBFloat16, iter.dtype(), "le_cpu", [&]() {
      cpu_kernel(iter,
       [=](scalar_t a, scalar_t b) -> scalar_t {
         return a <= b;
       });
    });
  }
}

void gt_kernel(TensorIterator& iter) {
  if (iter.dtype() == ScalarType::Bool) {
    AT_DISPATCH_ALL_TYPES_AND2(kBool, kBFloat16, iter.input_dtype(), "gt_cpu", [&]() {
      cpu_kernel(iter,
       [=](scalar_t a, scalar_t b) -> bool {
         return a > b;
       });
    });
  } else {
    AT_DISPATCH_ALL_TYPES_AND(kBFloat16, iter.dtype(), "gt_cpu", [&]() {
      cpu_kernel(iter,
       [=](scalar_t a, scalar_t b) -> scalar_t {
         return a > b;
       });
    });
  }
}

void ge_kernel(TensorIterator& iter) {
  if (iter.dtype() == ScalarType::Bool) {
    AT_DISPATCH_ALL_TYPES_AND2(kBool, kBFloat16, iter.input_dtype(), "ge_cpu", [&]() {
      cpu_kernel(iter,
       [=](scalar_t a, scalar_t b) -> bool {
         return a >= b;
       });
    });
  } else {
    AT_DISPATCH_ALL_TYPES_AND(kBFloat16, iter.dtype(), "ge_cpu", [&]() {
      cpu_kernel(iter,
       [=](scalar_t a, scalar_t b) -> scalar_t {
         return a >= b;
       });
    });
  }
}

void eq_kernel(TensorIterator& iter) {
  if (iter.dtype() == ScalarType::Bool) {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(kBool, kBFloat16, iter.input_dtype(), "eq_cpu", [&]() {
      cpu_kernel(iter,
       [=](scalar_t a, scalar_t b) -> bool {
         return a == b;
       });
    });
  } else {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND(kBFloat16, iter.dtype(), "eq_cpu", [&]() {
      cpu_kernel(iter,
       [=](scalar_t a, scalar_t b) -> scalar_t {
         return a == b;
       });
    });
  }
}

void ne_kernel(TensorIterator& iter) {
  if (iter.dtype() == ScalarType::Bool) {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(kBool, kBFloat16, iter.input_dtype(), "ne_cpu", [&]() {
      cpu_kernel(iter,
       [=](scalar_t a, scalar_t b) -> bool {
         return a != b;
       });
    });
  } else {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND(kBFloat16, iter.dtype(), "ne_cpu", [&]() {
      cpu_kernel(iter,
       [=](scalar_t a, scalar_t b) -> scalar_t {
         return a != b;
       });
    });
  }
}

void max_elementwise_kernel(TensorIterator& iter) {
  if (iter.dtype() == ScalarType::Bool) {
    cpu_kernel(iter,
      [](bool a, bool b) -> bool {
        return a || b;
      });
  } else if (isIntegralType(iter.dtype(), /*includeBool=*/ false)) {
    AT_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "max_lementwise_cpu", [&]() {
      cpu_kernel_vec(iter,
        [](scalar_t a, scalar_t b) -> scalar_t { return std::max(a, b); },
        [](Vec256<scalar_t> a, Vec256<scalar_t> b) { return at::vec256::maximum(a, b); });
    });
  } else {
    AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "max_elementwise_cpu", [&]() {
      cpu_kernel_vec(iter,
        [](scalar_t a, scalar_t b) -> scalar_t {
          if (std::isnan(a) || std::isnan(b)) {
            return std::numeric_limits<scalar_t>::quiet_NaN();
          } else {
            return std::max(a, b);
          }
        },
        [](Vec256<scalar_t> a, Vec256<scalar_t> b) { return at::vec256::maximum(a, b); });
    });
  }
}

void min_elementwise_kernel(TensorIterator& iter) {
  if (iter.dtype() == ScalarType::Bool) {
    cpu_kernel(iter,
      [](bool a, bool b) -> bool {
        return a && b;
      });
  } else if (isIntegralType(iter.dtype(), /*includeBool=*/ false)) {
    AT_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "min_elementwise_cpu", [&]() {
      cpu_kernel_vec(iter,
        [](scalar_t a, scalar_t b) -> scalar_t { return std::min(a, b); },
        [](Vec256<scalar_t> a, Vec256<scalar_t> b) { return at::vec256::minimum(a, b); });
    });
  } else {
    AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "min_elementwise_cpu", [&]() {
      cpu_kernel_vec(iter,
        [](scalar_t a, scalar_t b) -> scalar_t {
          if (std::isnan(a) || std::isnan(b)) {
            return std::numeric_limits<scalar_t>::quiet_NaN();
          } else {
            return std::min(a, b);
          }
        },
        [](Vec256<scalar_t> a, Vec256<scalar_t> b) { return at::vec256::minimum(a, b); });
    });
  }
}

void smooth_l1_kernel(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND(kBFloat16, iter.dtype(), "smooth_l1_cpu", [&]() {
    cpu_kernel(iter, [=](scalar_t a, scalar_t b) -> scalar_t {
      auto z = std::abs(a - b);
      return z < scalar_t(1.) ? scalar_t(0.5) * z * z : z - scalar_t(0.5);
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

void tanh_backward_kernel(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "tanh_backward_cpu", [&]() {
    auto one_vec = Vec256<scalar_t>((scalar_t)(1));
    cpu_kernel_vec(iter,
      [=](scalar_t a, scalar_t b) -> scalar_t {
        return a * (scalar_t(1) - b * b);
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
        return x % d;
      });
    });
  } else {
    AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "fmod_cpu", [&]() {
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
      cpu_kernel(iter, [=](scalar_t x) -> scalar_t {
        return x % div;
      });
    });
  } else {
    AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "fmod_scalar_cpu", [&]() {
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

} // anonymous namespace


REGISTER_DISPATCH(add_stub, &add_kernel);
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
REGISTER_DISPATCH(max_elementwise_stub, &max_elementwise_kernel);
REGISTER_DISPATCH(min_elementwise_stub, &min_elementwise_kernel);
REGISTER_DISPATCH(smooth_l1_stub, &smooth_l1_kernel);
REGISTER_DISPATCH(sigmoid_backward_stub, &sigmoid_backward_kernel);
REGISTER_DISPATCH(tanh_backward_stub, &tanh_backward_kernel);
REGISTER_DISPATCH(mse_stub, &mse_kernel);
REGISTER_DISPATCH(fmod_stub, &fmod_kernel);
REGISTER_DISPATCH(fmod_scalar_stub, &fmod_scalar_kernel);

}} // namespace at::native
